# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Available Transfer Capability (ATC) and the simultaneous-feasibility test.

The contract-path / point-to-point world that *precedes* nodal pricing. Before a
balancing authority joins a nodal market, it sells transmission as **point-to-point
service** between a point of receipt (POR) and a point of delivery (POD), rated by
**Available Transfer Capability** under the NERC MOD standards:

    ATC = TTC - TRM - ETC_F - CBM            (firm, eq. in the notebook)

    TTC  Total Transfer Capability   the most that can flow on the path before a
                                     reliability limit binds
    TRM  Transmission Reliability Margin   uncertainty cushion
    CBM  Capacity Benefit Margin     set aside for generation-reliability imports
    ETC  Existing Transmission Commitments   capability already sold/committed

This module computes ATC at the **shift-factor** level, consistent with the DC
algebra in ``seams_engine`` (it reuses that module's ``PTDFData`` — it does not
re-derive a PTDF). A point-to-point reservation ``s -> k`` is a balanced
injection at ``s`` / withdrawal at ``k``; its effect on line ``l`` is the
**path shift factor**

    a_l(s, k) = SF_{l, s} - SF_{l, k}                                   (1)

so a 1 MW reservation loads line ``l`` by ``a_l`` MW, and the path's standalone
transfer limit is

    TTC(s, k) = min_l  F_bar_l / |a_l(s, k)|                            (2)

(the first line to bind as the reservation grows). The central object is the
**simultaneous-feasibility test (SFT)**: a *set* of awards ``{(s_j, k_j, q_j)}``
is simultaneously feasible iff the superposed flow respects every monitored limit,

    | Sum_j q_j a_l(s_j, k_j) | <= F_bar_l   for all monitored l.       (3)

This is the condition behind revenue adequacy for financial transmission rights
(Hogan): congestion rents collected at the clearing cover the rights' payouts
*only* when the awarded set is simultaneously feasible. ATC posted path-by-path,
or summed across uncoordinated balancing authorities, can violate (3) — the
combined awards **oversubscribe** the feasible set. ``book_sequentially`` shows
the within-footprint fix (decrement ATC by existing commitments, ETC), and
``ba_atc`` shows the cross-footprint failure (each BA rates a path through only
its own monitored lines, missing a neighbor's loop flow).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

    import numpy.typing as npt
    import pypsa
    from typing_extensions import Self, Unpack

    from footprints import Footprints
    from lab_types import (
        Area,
        Bus,
        BusValues,
        Fleet,
        FloatArray,
        Interface,
        Line,
        LineValues,
        Loads,
    )
    from seams_engine import EngineResult, FlowgateSet, PTDFData

    Monitored: TypeAlias = Literal["all"] | Collection[Line] | FlowgateSet
    Point: TypeAlias = Bus | BusValues
    ExtraPoint: TypeAlias = tuple[str, Point, float, Area]
    PointRow: TypeAlias = tuple[str, BusValues, float, Area, str]
    AreaOf: TypeAlias = Callable[[Bus], Area] | Mapping[Bus, Area]
    AwardLike: TypeAlias = "Award | Request | tuple[Bus, Bus, float]"
    BookGroup: TypeAlias = "RightsBook | dict[str, RightsBook] | Iterable[RightsBook]"
    BidRow: TypeAlias = (
        tuple[Bus, Bus, float, float]
        | tuple[Bus, Bus, float, float, str]
        | tuple[Bus, Bus, float, float, str, str]
    )

    class NominateOptions(TypedDict, total=False):
        """The keyword arguments :func:`union_test` passes to the common test."""

        direction: str
        reduction: str
        sentinel: float | None
        max_iter: int | None
        weights: npt.ArrayLike | None
        tol: float
        rf_min: float


# ──────────────────────────────────────────────────────────────────────────
# Monitored-line resolution
# ──────────────────────────────────────────────────────────────────────────
def _monitored_idx(pt: PTDFData, monitored: Monitored) -> list[int]:
    """Indices of the monitored lines (``"all"`` or a list of line names)."""
    if monitored == "all":
        return list(range(pt.n_line))
    return [pt.line_idx[str(l)] for l in monitored]  # type: ignore[union-attr]  # a FlowgateSet raises here


def _fgset(monitored: Monitored) -> FlowgateSet | None:
    """``monitored`` as a ``seams_engine.FlowgateSet`` — or None if it names
    lines. Duck-typed so this module keeps no import on the engine: in the
    West the RATED PATH is the flowgate, so the ATC/TTC/SFT machinery accepts
    the same constraint object the solvers enforce (MOD-029 rates each path
    alone; the SFT over the flowgate union is the holistic check it skips)."""
    return monitored if (hasattr(monitored, "S") and hasattr(monitored, "ids")  # type: ignore[return-value]  # a duck test, which mypy cannot narrow
                         and hasattr(monitored, "fwd")) else None


def _ttc_over(pt: PTDFData, a: FloatArray, monitored: Monitored, tol: float,
              sentinel: float | None = None) -> tuple[float, Line | None, float]:
    """``min cap/|a|`` over the monitored constraint set for a per-line loading
    vector ``a``. Per-line path: the original loop, op for op. Flowgate path:
    the loading of flowgate k is ``(S·a)_k`` and the cap is direction-aware
    (``fwd`` when the transfer loads it forward, ``rev`` when backward).

    Returns ``(mw, id, cap)``. ``mw`` is the flow-scaled transfer limit,
    ``id`` the constraint that sets it, and ``cap`` that constraint's own
    rating in the direction the transfer loads it, which is the catalog
    figure a rated path is posted at. A transfer that impacts no monitored
    constraint returns ``(inf, None, inf)``, because the model carries no
    limit other than the monitored set.

    ``sentinel`` is the value an unrated direction carries, compared by
    equality. A direction carrying it is skipped, because the catalog does
    not rate that direction and the sentinel marks the absence of a rating
    rather than a large one. The skip matters here beyond never binding.
    This function returns the binding row's ``cap``, and
    :func:`rights_book_rsp` posts that ``cap`` one for one under
    ``ttc="contract_path"``, so a sentinel read as a rating would be sold as
    capability. ``sentinel is None`` reads every rating as a number, which is
    what every caller did before, and the comparison is by equality rather
    than by size because a genuine rating can sit above the sentinel. This is
    the convention :func:`_unbounded_caps` applies in the reduction rules.
    """
    sent = None if sentinel is None else float(sentinel)
    fgs = _fgset(monitored)
    if fgs is not None:
        af = fgs.S @ a
        best_mw, best_id, best_cap = np.inf, None, np.inf
        for r in range(fgs.n_fg):
            if abs(af[r]) < tol:
                continue
            cap = float(fgs.fwd[r] if af[r] > 0 else fgs.rev[r])
            if sent is not None and cap == sent:
                continue
            lim = cap / abs(af[r])
            if lim < best_mw:
                best_mw, best_id, best_cap = lim, fgs.ids[r], cap
        return float(best_mw), best_id, float(best_cap)
    best_mw, best_line, best_cap = np.inf, None, np.inf
    for l in _monitored_idx(pt, monitored):
        if abs(a[l]) < tol:
            continue
        cap = float(pt.s_nom[l])
        if sent is not None and cap == sent:
            continue
        lim = cap / abs(a[l])
        if lim < best_mw:
            best_mw, best_line, best_cap = lim, pt.lines[l], cap
    return float(best_mw), best_line, float(best_cap)


# ──────────────────────────────────────────────────────────────────────────
# Path shift factors, TTC, the ATC identity
# ──────────────────────────────────────────────────────────────────────────
def path_shift_factors(pt: PTDFData, source: Bus, sink: Bus) -> FloatArray:
    """Per-MW loading of every line by a point-to-point reservation ``source->sink``.

    ``a_l = SF_{l, source} - SF_{l, sink}`` (eq. 1). The slack bus cancels, so the
    result is independent of the PTDF's slack choice for a balanced transaction.
    """
    return pt.ptdf[:, pt.bus_idx[str(source)]] - pt.ptdf[:, pt.bus_idx[str(sink)]]


def _ttc_pair(pt: PTDFData, a: FloatArray, monitored: Monitored, tol: float,
              sentinel: float | None = None) -> tuple[float, Line | None]:
    """The first two members of :func:`_ttc_over`, which is what a caller
    that posts a transfer limit and its binding element needs. ``sentinel``
    passes through unchanged."""
    mw, which, _cap = _ttc_over(pt, a, monitored, tol, sentinel=sentinel)
    return mw, which


def ttc(pt: PTDFData, source: Bus, sink: Bus, monitored: Monitored = "all",
        tol: float = 1e-6) -> tuple[float, Line | None]:
    """Total Transfer Capability of the path and the first line to bind (eq. 2).

    ``TTC = min_l F_bar_l / |a_l|`` over the monitored lines (a line the path
    barely touches, ``|a_l| < tol``, never binds and is skipped). Returns
    ``(mw, binding_line_name)``.
    """
    a = path_shift_factors(pt, source, sink)
    return _ttc_pair(pt, a, monitored, tol)


def atc(ttc_mw: float, etc: float = 0.0, trm: float = 0.0, cbm: float = 0.0) -> float:
    """The firm ATC identity ``ATC = TTC - TRM - ETC_F - CBM`` (floored at 0).

    Defaults ``TRM = CBM = 0`` for the illustrations; the four terms are named so
    the markdown can reference the full NERC definition. ``etc`` is the capability
    (MW, in this path's direction) already consumed by existing commitments.
    """
    return max(0.0, ttc_mw - trm - etc - cbm)


def hub_path_shift_factors(pt: PTDFData, weights_from: BusValues,
                           weights_to: BusValues) -> FloatArray:
    """Per-MW loading of every line by a HUB-TO-HUB transfer: 1 MW injected
    distributed over ``weights_from`` (``{bus: w}``, summing to 1) and withdrawn
    over ``weights_to``. The distributed generalization of
    :func:`path_shift_factors` — the flow footprint of a GAP-to-GAP scheduled
    transfer — and slack-independent for the same balanced-transaction reason."""
    a = np.zeros(pt.n_line)
    for b, w in weights_from.items():
        a += float(w) * pt.ptdf[:, pt.bus_idx[str(b)]]
    for b, w in weights_to.items():
        a -= float(w) * pt.ptdf[:, pt.bus_idx[str(b)]]
    return a


def interface_ttc(pt: PTDFData, interfaces: Iterable[Interface],
                  weights: dict[Area, BusValues], monitored: Monitored = "all",
                  tol: float = 1e-6) -> dict[Interface, tuple[float, Line | None]]:
    """Rated-path TTC for a set of balancing-authority interfaces, each rated
    INDEPENDENTLY — the MOD-029 posture applied interface by interface.

    For each footprint pair the hub-to-hub transfer is ramped on the otherwise
    unloaded network until the first monitored line reaches its rating,
    ``TTC_ab = min_l F̄_l / |a_l|`` with ``a`` from
    :func:`hub_path_shift_factors` (the teaching simplification of the MOD-029
    R1/R2 transfer study; area-to-area ramps also echo MOD-028's
    area-interchange structure, but the result is USED the MOD-029 way — a
    static scalar posted per path). Because each interface is rated ALONE, the
    scalars ignore cross-path interaction (MOD-029 handles it only through
    pre-computed nomograms), so a TSP with several interfaces is in general
    over-assigned relative to what its system can carry at once —
    :func:`interface_sft` quantifies that overcount for the same set. With the
    unloaded base case the rating is direction-symmetric, matching the
    symmetric released-capacity bounds of ``solve_engine_transfers``.

    ``interfaces`` is an iterable of footprint pairs ``(a, b)``; ``weights`` is
    ``{footprint: {bus: w}}`` (ex-ante aggregation points, e.g.
    ``revenue_allocation.hub_weights``). Returns
    ``{(a, b) sorted: (ttc_mw, binding_line)}``."""
    out = {}
    for k in interfaces:
        a, b = sorted((str(k[0]), str(k[1])))
        sf = hub_path_shift_factors(pt, weights[a], weights[b])
        out[(a, b)] = _ttc_pair(pt, sf, monitored, tol)
    return out


def interface_sft(pt: PTDFData,
                  ratings: Mapping[Interface, float | tuple[float, Line | None]],
                  weights: dict[Area, BusValues], monitored: Monitored = "all",
                  direction: str = "worst") -> pd.DataFrame:
    """The holistic check the independent ratings skip: superpose EVERY
    interface's hub-to-hub transfer at its posted rating (oriented toward the
    sorted pair's second footprint) and report each monitored line's loading —
    the MOD-030/SFT reading of a MOD-029-style book. Overloaded rows are the
    over-assignment from path-by-path accounting: capacity sold interface by
    interface that the transmission service provider's system cannot carry
    simultaneously.

    ``ratings`` is ``{(a, b): MW}`` (or the ``{(a, b): (MW, line)}`` output of
    :func:`interface_ttc`). ``direction='sorted'`` superposes every rating
    oriented toward the sorted pair's second footprint (one signed pattern,
    counterflows credited — the netted/FTR-style reading);
    ``direction='worst'`` sums each rating's ABSOLUTE loading per line (no
    counterflow credit — the firm/OATT reading: the ratings are sold as firm
    rights exercisable in either direction, so the book must stand up when the
    directions stack). Returns a DataFrame (line, flow, rating, loading
    fraction, overloaded)."""
    if direction not in ("sorted", "worst"):
        raise ValueError("direction must be 'sorted' or 'worst'")
    fgs = _fgset(monitored)
    if fgs is not None:
        # Superpose at the FLOWGATE level: each rating's absolute flowgate
        # loading (worst) or its signed loading (sorted), against fwd/rev.
        cf = np.zeros(fgs.n_fg)
        for k, v in ratings.items():
            a, b = sorted((str(k[0]), str(k[1])))
            mw = float(v[0] if isinstance(v, (tuple, list)) else v)
            fc = fgs.S @ (mw * hub_path_shift_factors(pt, weights[a], weights[b]))
            cf += np.abs(fc) if direction == "worst" else fc
        rows = []
        for r in range(fgs.n_fg):
            f = float(cf[r])
            if direction == "worst":
                cap = float(min(fgs.fwd[r], fgs.rev[r]))   # must stand either way
                over = f > cap + 1e-6
            else:
                cap = float(fgs.fwd[r] if f >= 0 else fgs.rev[r])
                over = f > float(fgs.fwd[r]) + 1e-6 or f < -float(fgs.rev[r]) - 1e-6
            rows.append(dict(line=fgs.ids[r], flow=f, rating=cap,
                             loading=abs(f) / cap if cap > 0 else np.inf,
                             overloaded=over))
        return pd.DataFrame(rows).set_index("line")
    flow = np.zeros(pt.n_line)
    for k, v in ratings.items():
        a, b = sorted((str(k[0]), str(k[1])))
        mw = float(v[0] if isinstance(v, (tuple, list)) else v)
        contrib = mw * hub_path_shift_factors(pt, weights[a], weights[b])
        flow += np.abs(contrib) if direction == "worst" else contrib
    rows = []
    for l in _monitored_idx(pt, monitored):
        rows.append(dict(line=pt.lines[l], flow=flow[l], rating=pt.s_nom[l],
                         loading=abs(flow[l]) / pt.s_nom[l],
                         overloaded=abs(flow[l]) > pt.s_nom[l] + 1e-6))
    return pd.DataFrame(rows).set_index("line")


# ──────────────────────────────────────────────────────────────────────────
# Awards and the simultaneous-feasibility test
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Award:
    """A granted point-to-point reservation: ``mw`` MW from ``source`` to ``sink``."""

    source: str
    sink: str
    mw: float

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.sink = str(self.sink)
        self.mw = float(self.mw)


def _as_weights(point: Point) -> BusValues:
    """A point of receipt or delivery as ``{bus: weight}``.

    A bus name normalizes to ``{bus: 1.0}``. A dict is taken as given and its
    weights must sum to one, because the point is one megawatt of injection or
    withdrawal spread over the buses, which is what makes the transfer
    balanced and the shift factors slack-independent.
    """
    if isinstance(point, dict):
        w = {str(b): float(v) for b, v in point.items()}
        if not w:
            raise ValueError("a point of receipt or delivery needs a bus")
        total = sum(w.values())
        if abs(total - 1.0) >= 1e-9:
            raise ValueError(f"point weights sum to {total!r}, not 1")
        return w
    return {str(point): 1.0}


def _point_name(point: Point, weights: BusValues) -> str:
    """A readable name for a point. A bus name or a one-bus set is the bus.
    A distributed point is named by its heaviest bus and the count of the
    rest, because an aggregation point has no single bus to name."""
    if not isinstance(point, dict) or len(weights) == 1:
        return next(iter(weights))
    top = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return f"hub:{top}+{len(weights) - 1}"


@dataclass
class HubAward(Award):
    """A granted transfer between two AGGREGATION POINTS.

    ``source_w`` and ``sink_w`` are ``{bus: weight}`` maps summing to one, and
    ``source``/``sink`` are the names the ledgers print. The loading is
    :func:`hub_path_shift_factors` rather than :func:`path_shift_factors`, so
    a hub-to-hub award and a bus-to-bus award superpose in one list through
    :func:`_award_sf`.

    The distinction is the one the transport layer already draws: a
    point-to-point reservation names two buses, while network integration
    service and a balancing-authority transfer name two aggregation points.
    """

    source_w: BusValues = field(default_factory=dict)
    sink_w: BusValues = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.source_w = _as_weights(self.source_w)
        self.sink_w = _as_weights(self.sink_w)


@dataclass
class Request:
    """A transmission service request waiting in an authority's queue.

    ``source`` and ``sink`` are each a bus name or a ``{bus: weight}`` map
    summing to one. ``kind`` is ``"PTP"`` for point-to-point service or
    ``"NITS"`` for network integration transmission service. ``priority``
    sorts the queue, lower first, and requests of equal priority keep their
    insertion order, because a queue is served in the order it was received.
    ``holder`` names the party that holds the right and ``label`` names the
    request in the ledgers. ``provider`` names the transmission service
    provider that sells the service, and defaults to ``holder``, which is the
    self-provided case in which an authority books its own system. The two are
    distinct because a right on one provider's system is held by a customer of
    that provider, and the return test reads the charge on the provider's side
    and the value on the holder's.

    ``tier`` is the curtailment tier: 0 network integration transmission
    service, 1 long-term firm point-to-point service, 2 short-term firm
    point-to-point service, 3 non-firm service. A reduction cuts the highest
    tier first. ``priority`` is the queue order, which is a different object:
    it sets who is served first when capability is posted, while ``tier`` sets
    who is cut first when the posted book does not fit.

    ``source_name`` and ``sink_name`` override the printed names; they
    default to the bus for a single-bus point and to the heaviest bus plus a
    count for a distributed one.
    """

    source: Point
    sink: Point
    mw: float
    holder: str = ""
    kind: str = "PTP"
    priority: int = 0
    label: str = ""
    source_name: str = ""
    sink_name: str = ""
    provider: str = ""
    tier: int = 1
    source_w: BusValues = field(default_factory=dict, init=False, repr=False)
    sink_w: BusValues = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("NITS", "PTP"):
            raise ValueError(f"kind must be 'NITS' or 'PTP', got {self.kind!r}")
        self.mw = float(self.mw)
        self.holder = str(self.holder)
        self.provider = str(self.provider) or self.holder
        self.tier = int(self.tier)
        if self.tier not in (0, 1, 2, 3):
            raise ValueError(
                "tier must be 0 (network), 1 (long-term firm), 2 (short-term "
                f"firm) or 3 (non-firm), got {self.tier!r}")
        self.label = str(self.label)
        self.source_w = _as_weights(self.source)
        self.sink_w = _as_weights(self.sink)
        self.source_name = (str(self.source_name) or
                            _point_name(self.source, self.source_w))
        self.sink_name = (str(self.sink_name) or
                          _point_name(self.sink, self.sink_w))

    def to_award(self, mw: float | None = None) -> Award:
        """The request as an award of ``mw`` MW, the full request by default.

        A request whose two points are each a single bus NAMED BY THAT BUS
        lowers to a plain :class:`Award`, because ``hub_path_shift_factors``
        on a one-bus point is ``path_shift_factors`` term for term. Every
        other request becomes a :class:`HubAward`, including a one-bus point
        carrying a name of its own, because a plain award reads its buses off
        its own names.
        """
        q = self.mw if mw is None else float(mw)
        if (len(self.source_w) == 1 and len(self.sink_w) == 1
                and self.source_name in self.source_w
                and self.sink_name in self.sink_w):
            return Award(self.source_name, self.sink_name, q)
        return HubAward(self.source_name, self.sink_name, q,
                        source_w=dict(self.source_w), sink_w=dict(self.sink_w))


def _as_award(a: AwardLike) -> Award:
    if isinstance(a, Request):
        return a.to_award()
    return a if isinstance(a, Award) else Award(*a)


def _as_request(r: AwardLike) -> Request:
    """A queue entry as a :class:`Request`. An award or a
    ``(source, sink, mw)`` tuple carries no holder, provider, kind, tier or
    priority, so it takes the defaults. A :class:`Request` is returned as it
    is, so its holder, provider and tier reach the ledgers unchanged."""
    if isinstance(r, Request):
        return r
    a = _as_award(r)
    if isinstance(a, HubAward):
        return Request(dict(a.source_w), dict(a.sink_w), a.mw,
                       source_name=a.source, sink_name=a.sink)
    return Request(a.source, a.sink, a.mw)


# ──────────────────────────────────────────────────────────────────────────
# A clearing's dispatch read as a rights book
# ──────────────────────────────────────────────────────────────────────────
def _bus_key(b: Bus) -> tuple[int, int, str]:
    """A sort key that orders numeric bus names by number and the rest by
    name, numeric names first."""
    s = str(b)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def _area_fn(area_of: AreaOf) -> Callable[[Bus], Area]:
    """``area_of`` as a function of a bus name: a callable is used as it is
    and a mapping is indexed, so a bus with no area raises ``KeyError``."""
    if callable(area_of):
        return lambda b: area_of(str(b))
    return lambda b: area_of[str(b)]


def _extra_points(points: Iterable[ExtraPoint], what: str,
                  tol: float) -> list[PointRow]:
    """The caller's extra sources or sinks as ``(key, weights, mw, area,
    name)`` rows, each ``(name, weights, mw, area)`` in the call."""
    out = []
    for p in points:
        if len(p) != 4:
            raise ValueError(f"an extra {what} is (name, weights, MW, area), "
                             f"got {p!r}")
        name, weights, mw, area = p
        mw = float(mw)
        if mw < -tol:
            raise ValueError(f"the extra {what} {name!r} carries {mw} MW; a "
                             f"{what} carries non-negative megawatts")
        if mw <= tol:
            continue
        out.append((str(name), _as_weights(weights), mw, str(area),
                    str(name)))
    return out


def _point_sf(pt: PTDFData, w: BusValues) -> FloatArray:
    """The shift-factor column of a point ``{bus: weight}`` on every line."""
    return sum(float(v) * pt.ptdf[:, pt.bus_idx[str(b)]] for b, v in w.items())  # type: ignore[return-value]  # an empty w returns the int 0


def _pairing_matrix(src: list[PointRow], snk: list[PointRow], pairing: str,
                    pt: PTDFData | None, lines: Iterable[Line] | None,
                    merit: list[float], D: float) -> FloatArray:
    """The volumes ``x[s, k]`` of the rights from each source to each sink
    under ``pairing``. ``src`` and ``snk`` are the point rows of
    :func:`dispatch_requests`, ``merit`` the sources' merit-order keys, and
    ``D`` the sinks' total. Every pairing delivers each source's whole output
    and fills each sink's whole withdrawal."""
    q_s = np.array([r[2] for r in src], dtype=float)
    q_k = np.array([r[2] for r in snk], dtype=float)
    if pairing == "proportional":
        return np.outer(q_s, q_k) / D
    # the distance of a pair: the summed absolute difference of the two
    # points' shift factors over the named constraints
    li = [pt.line_idx[m] for m in lines]  # type: ignore[union-attr]  # not None off the proportional pairing
    sf_s = [_point_sf(pt, r[1])[li] for r in src]  # type: ignore[arg-type]  # pt as on the line above
    sf_k = [_point_sf(pt, r[1])[li] for r in snk]  # type: ignore[arg-type]  # pt as on the line above
    C = np.array([[float(np.abs(a - b).sum()) for b in sf_k] for a in sf_s])
    # the sinks carry the sources' total, so that the two margins agree to
    # the last bit and the program is feasible within the balance tolerance
    q_k = q_k * (q_s.sum() / q_k.sum())
    if pairing == "nearest":
        X = np.zeros((len(src), len(snk)))
        room = q_k.copy()
        for i in sorted(range(len(src)), key=lambda j: (merit[j], j)):
            left = q_s[i]
            for k in sorted(range(len(snk)), key=lambda j: (C[i, j], j)):
                if left <= 0.0:
                    break
                take = min(left, room[k])
                if take > 0.0:
                    X[i, k] += take
                    room[k] -= take
                    left -= take
        return X
    from scipy.optimize import linprog
    n_s, n_k = len(src), len(snk)
    A_eq = np.zeros((n_s + n_k, n_s * n_k))
    for i in range(n_s):
        A_eq[i, i * n_k:(i + 1) * n_k] = 1.0
    for k in range(n_k):
        A_eq[n_s + k, k::n_k] = 1.0
    out = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([q_s, q_k]),
                  bounds=[(0.0, None)] * (n_s * n_k), method="highs")
    if out.status != 0:
        raise RuntimeError(f"the minimum-loading pairing failed: {out.message}")
    return np.maximum(out.x.reshape(n_s, n_k), 0.0)


PAIRINGS = ("proportional", "nearest", "min_loading")


def dispatch_requests(res: EngineResult, *, gens: Fleet | None = None,
                      loads: Loads | None = None, area_of: AreaOf,
                      buses: Iterable[Bus] | None = None,
                      basis: str = "gross", granularity: str = "unit",
                      sources: Iterable[ExtraPoint] = (),
                      sinks: Iterable[ExtraPoint] = (),
                      tier: Callable[[Area, Area], int] | None = None,
                      label: str = "", tol: float = 1e-6,
                      min_mw: float | None = None,
                      pairing: str = "proportional", pt: PTDFData | None = None,
                      lines: Iterable[Line] | None = None,
                      holder: str = "sink") -> list[Request]:
    """A clearing's dispatch read as a book of rights: each source's megawatts
    split over the sinks in proportion to their megawatts, held by the sink's
    area and provided by the source's.

    With sources ``s`` carrying ``q_s`` and sinks ``k`` carrying ``q_k``, and
    ``D = sum_k q_k``, the right from ``s`` to ``k`` is

        ``x_(s -> k) = q_s q_k / D``,

    so every source delivers its whole output and every sink receives its
    whole withdrawal. The superposed flow of the book is therefore
    ``sum_s q_s SF_s - sum_k q_k SF_k`` on every line, the flow the sources
    and sinks themselves place on the network. That requires ``sum_s q_s = D``, and the function raises
    ``ValueError`` where the two totals differ by more than ``tol`` times the
    larger of one megawatt and ``D``.

    ``basis`` names the two readings of the dispatch.

    1. ``"gross"``: the sources are the dispatched generation and the sinks
       the served load, the nominal load less the shed plus the cleared
       demand bids at the bus (``res.demand_by_bus``, which carries battery
       charging under the completed storage formulation, read where the
       result carries it). This is the rule of notebook 304a's
       ``dispatch_tags`` and of the note's tag equation: each unit's dispatch
       split over the served loads in proportion to load.
    2. ``"net"``: each bus nets its generation against its served load, a bus
       with a positive net injection is a source and a bus with a negative
       one a sink. This is construction M3 of notebook 111b, the book the
       dispatch supports.

    ``granularity`` names the points.

    1. ``"unit"``: one source per dispatched unit at its bus, one sink per
       load bus. The sources are ordered by offer cost, then by name, where
       ``gens`` carries a ``cost``. The gross basis only.
    2. ``"bus"``: one source per bus with generation, one sink per load bus.
    3. ``"area"``: one source and one sink per area, each a point spread over
       the area's buses in proportion to their megawatts and named
       ``"<area>.GEN"`` and ``"<area>.LOAD"``. The superposition and every
       per-area total are unchanged, and a right-by-right quantity is not.

    ``gens`` is ``{unit: spec}`` with each spec's ``bus`` (an engine's
    ``gens``), read against ``res.dispatch``; with ``gens=None`` the
    generation is ``res.gen_by_bus`` and the unit granularity is not
    available. ``loads`` is ``{bus: MW}``, the nominal loads, and defaults to
    ``res.load_by_bus``. ``buses`` restricts both to a bus set, a footprint's
    own, and ``None`` keeps every bus. ``area_of`` maps a bus to its area, a
    callable or a mapping.

    ``sources`` and ``sinks`` add named points ``(name, weights, MW, area)``,
    each kept as its own point under every granularity. They carry what the
    dispatch does not hold at a bus: an import from an area outside the
    footprint as a source at that area's aggregation point, an export as a
    sink there, and the two ends of a controllable link.

    Each right is a :class:`Request` held by the sink's area and provided by
    the source's, which is notebook 111b's ``request()`` rule: network
    integration service at ``tier=0`` where the two areas are one, and
    point-to-point service at ``tier=1`` across. ``tier`` overrides the tier
    as a function ``tier(source area, sink area) -> int``, which under the
    default sink rule is ``tier(provider, holder)``. Every request carries
    ``priority=0``, as 111b's do, and is labelled ``label`` followed by its
    two points. A right of ``min_mw`` megawatts or less is not issued.
    ``min_mw=None`` reads as ``tol``, which is 111b's rule; a caller that
    asserts the book's superposition at ``tol`` passes ``min_mw=0.0``, since
    the rights left out at ``tol`` each accumulate over many pairs of points. A point carries nothing at 1e-9 MW or less on the gross
    basis, 304a's threshold, and at ``tol`` or less on the net basis, 111b's.
    An extra point carries nothing at ``tol`` or less.

    The holder rule is a Lab choice, made so that the settlement regimes that
    value rights holder by holder are read per area, as the regimes that read
    flow contributions are. The pairing in proportion to megawatts is the
    note's rule. Note that the regimes that read only flow contributions do
    not depend on the pairing, and the regimes that value rights one by one do.

    ``pairing`` names three pairings, each of which delivers every source's
    whole output and fills every sink's whole withdrawal, so the book's
    superposed flow is the same under each (build item B2 of the
    independent-clearing argument diagram, 2026-10-23).

    1. ``"proportional"``, the default: ``x_(s -> k) = q_s q_k / D``, the rule
       above.
    2. ``"nearest"``: the sources in merit order, each delivering to the
       sinks nearest it first, where the distance of a source and a sink is
       ``sum_m |SF_(s,m) - SF_(k,m)|`` over the constraints ``lines``. The
       merit order is the offer cost at the unit granularity and the
       clearing's price at the source point otherwise (``res.lmp`` weighted
       by the point's weights), ties broken by the source's order above.
    3. ``"min_loading"``: the transportation program that minimizes the
       book's total absolute loading
       ``sum_(s,k) sum_m |x_(s -> k) (SF_(s,m) - SF_(k,m))|`` over the
       constraints ``lines`` subject to the source and sink totals, with
       ``s`` the source index and ``k`` the sink index. The objective is the
       sum over rights of each right's absolute loading, a firm-type sum in
       which no right's counterflow offsets another right's flow. It is
       linear because each right's loading on ``m`` is its volume
       ``x_(s -> k) >= 0`` times the fixed factor ``SF_(s,m) - SF_(k,m)``.
       Its optimum need not be unique, and the solver's vertex is returned.

    The two alternative pairings need ``pt``, the shift factors, and read
    ``lines``, every line of ``pt`` by default. They are Lab constructions,
    stated as such: the record states no pairing rule for a dispatch read as
    rights beyond the proportional one.

    ``holder`` names the holder rule: ``"sink"``, the default, holds each
    right in the sink's area, and ``"source"`` holds it in the source's area.
    The provider is the source's area under both. The kind and the tier are
    properties of the delivery path and not of the holder, so they read the
    pair (source area, sink area) under both rules: a right within one area
    is network integration service at tier 0 and a right across areas is
    point-to-point service at tier 1, and a ``tier`` callable is called as
    ``tier(source area, sink area)``. Under the default sink rule the sink's
    area is the holder, so that call is ``tier(provider, holder)``.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    if holder not in ("sink", "source"):
        raise ValueError(f"holder must be 'sink' or 'source', got {holder!r}")
    if pairing != "proportional" and pt is None:
        raise ValueError(f"the {pairing!r} pairing reads the shift factors, "
                         "and pt is None")
    if basis not in ("gross", "net"):
        raise ValueError(f"basis must be 'gross' or 'net', got {basis!r}")
    if granularity not in ("unit", "bus", "area"):
        raise ValueError("granularity must be 'unit', 'bus' or 'area', got "
                         f"{granularity!r}")
    if granularity == "unit" and basis == "net":
        raise ValueError("the net basis nets each bus, so its points are "
                         "buses or areas, not units")
    if granularity == "unit" and gens is None:
        raise ValueError("the unit granularity reads gens against "
                         "res.dispatch, and gens is None")
    area = _area_fn(area_of)
    keep = None if buses is None else {str(b) for b in buses}
    # the point threshold: 304a's 1e-9 MW on the gross basis, 111b's tol on
    # the net basis
    pt_tol = 1e-9 if basis == "gross" else float(tol)
    x_min = float(tol) if min_mw is None else float(min_mw)

    def _inside(b: Bus) -> bool:
        return keep is None or str(b) in keep

    # generation: by unit, and by bus
    units = []
    if gens is not None:
        disp = getattr(res, "dispatch", None) or {}
        for g, spec in gens.items():
            b = str(spec["bus"])
            mw = float(disp.get(g, 0.0))
            if _inside(b) and mw > pt_tol:
                units.append((str(g), b, mw, float(spec.get("cost", 0.0))))
        units.sort(key=lambda u: (u[3], u[0]))
        gen_bus: BusValues = {}
        for _g, b, mw, _c in units:
            gen_bus[b] = gen_bus.get(b, 0.0) + mw
    else:
        gen_bus = {str(b): float(v) for b, v in
                   (getattr(res, "gen_by_bus", None) or {}).items()
                   if _inside(b)}

    # served load: nominal less shed, plus the cleared demand bids
    nominal = (getattr(res, "load_by_bus", None) or {}) if loads is None \
        else loads
    shed = getattr(res, "shed_by_bus", None) or {}
    bids = getattr(res, "demand_by_bus", None) or {}
    served: BusValues = {}
    for b, v in nominal.items():
        if _inside(b):
            served[str(b)] = served.get(str(b), 0.0) + float(v)
    for b, v in shed.items():
        if _inside(b):
            served[str(b)] = served.get(str(b), 0.0) - float(v)
    for b, v in bids.items():
        if _inside(b):
            served[str(b)] = served.get(str(b), 0.0) + float(v)

    if basis == "net":
        net = {b: gen_bus.get(b, 0.0) - served.get(b, 0.0)
               for b in set(gen_bus) | set(served)}
        src_bus = {b: v for b, v in net.items() if v > pt_tol}
        snk_bus = {b: -v for b, v in net.items() if v < -pt_tol}
    else:
        src_bus = {b: v for b, v in gen_bus.items() if v > pt_tol}
        snk_bus = {b: v for b, v in served.items() if v > pt_tol}

    def _by_area(mw_by_bus: BusValues, suffix: str) -> list[PointRow]:
        grouped: dict[Area, BusValues] = {}
        for b in sorted(mw_by_bus, key=_bus_key):
            grouped.setdefault(area(b), {})[b] = mw_by_bus[b]
        rows = []
        for a in sorted(grouped):
            tot = sum(grouped[a].values())
            rows.append((f"{a}.{suffix}",
                         {b: v / tot for b, v in grouped[a].items()},
                         tot, str(a), f"{a}.{suffix}"))
        return rows

    def _by_bus(mw_by_bus: BusValues) -> list[PointRow]:
        return [(b, {b: 1.0}, mw_by_bus[b], str(area(b)), b)
                for b in sorted(mw_by_bus, key=_bus_key)]

    if granularity == "unit":
        src = [(g, {b: 1.0}, mw, str(area(b)), b) for g, b, mw, _c in units]
    elif granularity == "bus":
        src = _by_bus(src_bus)
    else:
        src = _by_area(src_bus, "GEN")
    snk = _by_area(snk_bus, "LOAD") if granularity == "area" \
        else _by_bus(snk_bus)
    src += _extra_points(sources, "source", tol)
    snk += _extra_points(sinks, "sink", tol)

    S = float(sum(r[2] for r in src))
    D = float(sum(r[2] for r in snk))
    if abs(S - D) > tol * max(1.0, abs(D)):
        raise ValueError(
            f"the sources carry {S:,.6f} MW and the sinks {D:,.6f} MW; a "
            "dispatch read as rights must balance its sinks")
    merit = []
    if pairing == "nearest":
        cost = {u[0]: u[3] for u in units} if granularity == "unit" else {}
        lmp = getattr(res, "lmp", None) or {}
        for key_s, w_s, _q, _a, _n in src:
            merit.append(cost[key_s] if key_s in cost else
                         float(sum(v * float(lmp.get(b, 0.0))
                                   for b, v in w_s.items())))
    X = _pairing_matrix(src, snk, pairing, pt,
                        list(pt.lines) if (pt is not None and lines is None)
                        else lines, merit, D) if src and snk else None
    out = []
    for i, (key_s, w_s, q_s, a_s, nm_s) in enumerate(src):
        for k, (key_k, w_k, q_k, a_k, nm_k) in enumerate(snk):
            x = float(X[i, k])  # type: ignore[index]  # X is None only when this loop is empty
            if x <= x_min:
                continue
            a_h = a_k if holder == "sink" else a_s
            # the kind and the tier belong to the delivery path, so they
            # read (source area, sink area) whatever the holder rule
            kind = "NITS" if a_s == a_k else "PTP"
            tr = (0 if kind == "NITS" else 1) if tier is None \
                else int(tier(a_s, a_k))
            out.append(Request(
                w_s if len(w_s) > 1 or nm_s not in w_s else nm_s,
                w_k if len(w_k) > 1 or nm_k not in w_k else nm_k,
                x, holder=a_h, provider=a_s, kind=kind, tier=tr,
                label=f"{label}{key_s}->{key_k}",
                source_name=nm_s, sink_name=nm_k))
    return out


def _award_sf(pt: PTDFData, award: AwardLike) -> FloatArray:
    """Per-MW loading of every line by one award.

    :func:`hub_path_shift_factors` for a :class:`HubAward` and
    :func:`path_shift_factors` for a plain :class:`Award`. Every superposition
    in this module reads an award's loading here, so the two kinds mix in one
    award list.
    """
    a = _as_award(award)
    if isinstance(a, HubAward):
        return hub_path_shift_factors(pt, a.source_w, a.sink_w)
    return path_shift_factors(pt, a.source, a.sink)


def _regrant(award: Award, mw: float) -> Award:
    """The same award at a different megawatt quantity, kind preserved."""
    if isinstance(award, HubAward):
        return HubAward(award.source, award.sink, mw,
                        source_w=dict(award.source_w),
                        sink_w=dict(award.sink_w))
    return Award(award.source, award.sink, mw)


def superposed_flow(pt: PTDFData, awards: Iterable[AwardLike]) -> FloatArray:
    """Signed flow on every line from the superposition of all awards: ``Sum_j q_j a_l``."""
    f = np.zeros(pt.n_line)
    for a in awards:
        a = _as_award(a)
        f += a.mw * _award_sf(pt, a)
    return f


def line_loadings(pt: PTDFData, awards: Iterable[AwardLike],
                  monitored: Monitored = "all") -> pd.DataFrame:
    """Per-line SFT table for a set of awards.

    Columns: ``from``/``to`` buses, signed ``flow`` (MW), ``limit`` (F_bar),
    ``loading_%``, and ``overload`` (``|flow| > limit``). Restricted to the
    monitored lines — also the source of ``line_flows``/``constrained_lines`` for
    the network figures.
    """
    f = superposed_flow(pt, awards)
    fgs = _fgset(monitored)
    if fgs is not None:
        cf = fgs.S @ f
        rows = []
        for r in range(fgs.n_fg):
            flow = float(cf[r])
            cap = float(fgs.fwd[r] if flow >= 0 else fgs.rev[r])
            rows.append({
                "line": fgs.ids[r], "from": "", "to": "",
                "flow": round(flow, 1), "limit": round(cap, 0),
                "loading_%": round(100 * abs(flow) / cap, 0) if cap > 0 else np.inf,
                "overload": (flow > float(fgs.fwd[r]) + 1e-6
                             or flow < -float(fgs.rev[r]) - 1e-6),
            })
        return pd.DataFrame(rows).set_index("line")
    idx = _monitored_idx(pt, monitored)
    rows = []
    for l in idx:
        b0, b1 = pt.line_buses[l]
        lim = float(pt.s_nom[l])
        flow = float(f[l])
        rows.append({
            "line": pt.lines[l], "from": b0, "to": b1,
            "flow": round(flow, 1), "limit": round(lim, 0),
            "loading_%": round(100 * abs(flow) / lim, 0) if lim > 0 else np.inf,
            "overload": abs(flow) > lim + 1e-6,
        })
    return pd.DataFrame(rows).set_index("line")


def simultaneous_feasibility(pt: PTDFData, awards: Iterable[AwardLike],
                             monitored: Monitored = "all",
                             ) -> tuple[bool, pd.DataFrame]:
    """Hogan's SFT (eq. 3): is the superposed flow within every monitored limit?

    Returns ``(feasible, loadings_table)``. ``feasible`` is True iff no monitored
    line overloads.
    """
    df = line_loadings(pt, awards, monitored=monitored)
    return (not bool(df["overload"].any()), df)


def overloaded_lines(pt: PTDFData, awards: Iterable[AwardLike],
                     monitored: Monitored = "all") -> list[Line]:
    """Names of the monitored lines that overload under the award set (for figures)."""
    df = line_loadings(pt, awards, monitored=monitored)
    return df.index[df["overload"]].tolist()


def firm_line_loadings(pt: PTDFData, awards: Iterable[AwardLike],
                       monitored: Monitored = "all") -> pd.DataFrame:
    """Per-line FIRM loading -- the conservative ATC standard that does NOT credit counterflow.

    Firm point-to-point rights must be deliverable even if the counter-flowing rights are not
    scheduled, so on each line the same-direction commitments are summed *without* netting the
    opposing ones: ``fwd = Sum_j max(0, q_j a_l)``, ``rev = Sum_j max(0, -q_j a_l)``, and the firm
    loading is ``max(fwd, rev)``. Contrast :func:`line_loadings`, which nets the flows by
    superposition -- the efficient-dispatch / financial-FTR notion that *does* credit counterflow.
    Columns: ``from``/``to``, ``fwd``/``rev`` directional sums, ``firm`` loading, ``limit``,
    ``loading_%`` (firm vs limit), ``overload`` (``firm > limit``).
    """
    fgs = _fgset(monitored)
    if fgs is not None:
        # Firm reading at the FLOWGATE level: each award's signed flowgate
        # loading, same-direction sums, no counterflow credit. The forward sum
        # tests against fwd_k, the reverse sum against rev_k (asymmetric
        # ratings keep their own directions; an unrated side never binds).
        per = []
        for a in awards:
            a = _as_award(a)
            per.append(fgs.S @ (a.mw * _award_sf(pt, a)))
        rows = []
        for r in range(fgs.n_fg):
            vals = [float(p[r]) for p in per]
            fwd_sum = sum(v for v in vals if v > 0)
            rev_sum = sum(-v for v in vals if v < 0)
            over = (fwd_sum > float(fgs.fwd[r]) + 1e-6
                    or rev_sum > float(fgs.rev[r]) + 1e-6)
            cap = float(fgs.fwd[r] if fwd_sum >= rev_sum else fgs.rev[r])
            firm = max(fwd_sum, rev_sum)
            rows.append({
                "line": fgs.ids[r], "from": "", "to": "",
                "fwd": round(fwd_sum, 1), "rev": round(rev_sum, 1),
                "firm": round(firm, 1), "limit": round(cap, 0),
                "loading_%": round(100 * firm / cap, 0) if cap > 0 else np.inf,
                "overload": over,
            })
        return pd.DataFrame(rows).set_index("line")
    idx = _monitored_idx(pt, monitored)
    contrib: dict[int, list[float]] = {l: [] for l in idx}
    for a in awards:
        a = _as_award(a)
        sf = _award_sf(pt, a)
        for l in idx:
            contrib[l].append(float(a.mw * sf[l]))
    rows = []
    for l in idx:
        b0, b1 = pt.line_buses[l]
        lim = float(pt.s_nom[l])
        fwd = sum(v for v in contrib[l] if v > 0)
        rev = sum(-v for v in contrib[l] if v < 0)
        firm = max(fwd, rev)
        rows.append({
            "line": pt.lines[l], "from": b0, "to": b1,
            "fwd": round(fwd, 1), "rev": round(rev, 1), "firm": round(firm, 1),
            "limit": round(lim, 0),
            "loading_%": round(100 * firm / lim, 0) if lim > 0 else np.inf,
            "overload": firm > lim + 1e-6,
        })
    return pd.DataFrame(rows).set_index("line")


def firm_feasibility(pt: PTDFData, awards: Iterable[AwardLike],
                     monitored: Monitored = "all") -> tuple[bool, pd.DataFrame]:
    """Firm-rights ATC feasibility: are the same-direction commitments within every monitored
    limit *without* crediting counterflow (:func:`firm_line_loadings`)?

    The conservative standard for firm point-to-point transmission rights -- a set can be firm
    even when a counter-flowing right would let the efficient (netted) dispatch carry more, and
    a set the netted :func:`simultaneous_feasibility` passes can still be firm-infeasible if it
    leans on counterflow. Returns ``(feasible, firm_loadings_table)``.
    """
    df = firm_line_loadings(pt, awards, monitored=monitored)
    return (not bool(df["overload"].any()), df)


def flow_dict(pt: PTDFData, awards: Iterable[AwardLike]) -> LineValues:
    """``{line: signed MW}`` of the superposed award flow (feeds ``plot_network_topology``)."""
    f = superposed_flow(pt, awards)
    return {pt.lines[l]: float(f[l]) for l in range(pt.n_line)}


# ──────────────────────────────────────────────────────────────────────────
# Sequential booking — decrement ATC by existing commitments (ETC)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class FlowgateLedger:
    """Existing transmission commitments (ETC) on a monitored constraint set.

    Three sums over the same constraints, each in constraint MW. ``fwd`` and
    ``rev`` are the same-direction sums of every booked award's loading, taken
    without crediting counterflow, because a firm right must be deliverable
    when the counter-flowing rights are not scheduled. ``net`` is the signed
    superposition, which is the netted reading that does credit counterflow.
    The two readings are the firm-versus-netted distinction of
    :func:`firm_line_loadings` versus :func:`line_loadings`, carried into the
    booking loop.

    A constraint here is a flowgate or a line. The constraint order is the one
    :func:`_constraint_loading` returns for the same monitored set.
    """

    fwd: FloatArray
    rev: FloatArray
    net: FloatArray

    @classmethod
    def zeros(cls, n: int) -> "FlowgateLedger":
        """An empty ledger over ``n`` constraints."""
        return cls(fwd=np.zeros(int(n)), rev=np.zeros(int(n)),
                   net=np.zeros(int(n)))

    def add(self, af: FloatArray, mw: float) -> "FlowgateLedger":
        """Book ``mw`` MW of an award whose per-MW constraint loading is ``af``.

        The award loads constraint ``k`` by ``af_k * mw``. That signed value
        enters ``net``; its positive part enters ``fwd`` and its negative part
        enters ``rev``, so the directional sums never net against each other.
        """
        v = np.asarray(af, dtype=float) * float(mw)
        self.net = self.net + v
        self.fwd = self.fwd + np.maximum(v, 0.0)
        self.rev = self.rev + np.maximum(-v, 0.0)
        return self

    def table(self, ids: Iterable[Line], cap_fwd: npt.ArrayLike,
              cap_rev: npt.ArrayLike, etc_basis: str = "firm") -> pd.DataFrame:
        """The ETC and AFC table, indexed by constraint id.

        Columns ``cap_fwd``/``cap_rev`` are the constraint's two ratings and
        ``etc_fwd``/``etc_rev``/``etc_net`` the three commitment sums, which
        are reported whatever the basis. ``afc_fwd``/``afc_rev`` are the
        remaining capability in each direction, ``cap - etc`` floored at zero,
        on the basis ``etc_basis`` selects. Under ``"firm"`` the deduction is
        the same-direction sum, which is the capability a further firm request
        can be granted against, and is the default. Under ``"netted"`` it is
        the signed net read in that direction, which credits counterflow and
        matches what ``available_atc`` grants on the netted basis. TRM and CBM
        are not deducted here. They are arguments of :func:`atc` and default to
        zero in this series.
        """
        if etc_basis not in ("netted", "firm"):
            raise ValueError(
                f"etc_basis must be 'netted' or 'firm', got {etc_basis!r}")
        cap_fwd = np.asarray(cap_fwd, dtype=float)
        cap_rev = np.asarray(cap_rev, dtype=float)
        used_fwd = self.fwd if etc_basis == "firm" else self.net
        used_rev = self.rev if etc_basis == "firm" else -self.net
        return pd.DataFrame({
            "id": list(ids),
            "cap_fwd": cap_fwd,
            "cap_rev": cap_rev,
            "etc_fwd": self.fwd,
            "etc_rev": self.rev,
            "etc_net": self.net,
            "afc_fwd": np.maximum(0.0, cap_fwd - used_fwd),
            "afc_rev": np.maximum(0.0, cap_rev - used_rev),
        }).set_index("id")


def _constraint_loading(pt: PTDFData, a: FloatArray, monitored: Monitored,
                        ) -> tuple[FloatArray, FloatArray, FloatArray, list[Line]]:
    """Per-MW impact of the per-line loading vector ``a`` on the monitored set.

    Returns ``(af, cap_fwd, cap_rev, ids)``. The flowgate branch takes
    ``af = S @ a`` with the flowgate set's own direction-aware ratings. The
    per-line branch takes ``af = a[idx]`` with ``s_nom`` in both directions,
    because a line rating is symmetric. A flowgate's ``S`` is a selection
    matrix at the identity set, so ``(S @ a)_k`` is ``a[l]`` exactly and the
    two branches agree bit for bit there.

    ``a`` is one loading vector over the lines, or a matrix whose columns are
    one loading vector each. Both branches select rows, so a matrix returns
    one row per constraint and one column per vector.
    """
    fgs = _fgset(monitored)
    if fgs is not None:
        return (np.asarray(fgs.S @ a, dtype=float),
                np.asarray(fgs.fwd, dtype=float),
                np.asarray(fgs.rev, dtype=float),
                list(fgs.ids))
    idx = _monitored_idx(pt, monitored)
    cap = np.array([float(pt.s_nom[l]) for l in idx])
    return (np.asarray(a, dtype=float)[idx], cap, cap.copy(),
            [pt.lines[l] for l in idx])


def _booked_ledger(pt: PTDFData, booked: Iterable[AwardLike],
                   monitored: Monitored) -> FlowgateLedger:
    """The ETC ledger of an award list over the monitored constraint set."""
    _, _, _, ids = _constraint_loading(pt, np.zeros(pt.n_line), monitored)
    led = FlowgateLedger.zeros(len(ids))
    for b in booked:
        b = _as_award(b)
        af, _, _, _ = _constraint_loading(pt, _award_sf(pt, b), monitored)
        led.add(af, b.mw)
    return led


def flowgate_available_atc(pt: PTDFData, a: FloatArray,
                           monitored: Monitored = "all",
                           ledger: FlowgateLedger | None = None,
                           etc_basis: str = "netted", rf_min: float = 0.0,
                           tol: float = 1e-6, sentinel: float | None = None,
                           ) -> tuple[float, Line | None]:
    """Remaining transfer capability of a path over a monitored constraint set.

    ``a`` is the path's per-line loading vector. A constraint is consulted when
    its per-MW impact ``|af_k|`` reaches ``max(rf_min, tol)``. ``rf_min`` is the
    MOD-030 response-factor threshold below which a flowgate is not charged for
    a transfer, and ``rf_min = 0.0`` means every impacted constraint is
    consulted. A constraint below the threshold is neither consulted nor
    decremented. MOD-030 defines the response factor on the outage transfer
    distribution factor, which is the limiting element under its worst
    contingency at a 5 percent threshold, while ``rf_min`` screens on the
    base-case PTDF path shift factor with no contingency step, which is a
    stated deviation.

    The headroom of constraint ``k`` is its rating in the push direction less
    the commitments already taken there: ``cap_fwd_k - used_k`` where the path
    loads it forward, ``cap_rev_k - used_k`` where the path loads it backward.
    Under ``etc_basis="netted"`` the used capability is the signed ledger net
    read in the push direction, which credits counterflow. Under
    ``etc_basis="firm"`` it is the same-direction sum ``fwd_k`` or ``rev_k``,
    which does not. Returns the tightest ``headroom_k / |af_k|``, floored at
    zero, and the constraint id that set it. Returns ``(inf, None)`` when the
    path impacts no constraint at the threshold, because the model carries no
    limit other than the monitored set.

    ``sentinel`` is the value an unrated direction carries, compared by
    equality. A direction carrying it is unbounded here, through
    :func:`_unbounded_caps`, so the sentinel never acts as a rating and never
    names itself as the limiting constraint. ``sentinel is None`` reads every
    rating as a number, which is what every caller did before.
    """
    if etc_basis not in ("netted", "firm"):
        raise ValueError(
            f"etc_basis must be 'netted' or 'firm', got {etc_basis!r}")
    af, cap_fwd, cap_rev, ids = _constraint_loading(pt, a, monitored)
    cap_fwd, cap_rev = _unbounded_caps(cap_fwd, cap_rev, sentinel)
    thresh = max(float(rf_min), float(tol))
    best_mw, best_id = np.inf, None
    for k in range(len(ids)):
        if abs(af[k]) < thresh:
            continue
        if af[k] > 0:
            cap = float(cap_fwd[k])
            used = 0.0 if ledger is None else (
                float(ledger.net[k]) if etc_basis == "netted"
                else float(ledger.fwd[k]))
        else:
            cap = float(cap_rev[k])
            used = 0.0 if ledger is None else (
                float(-ledger.net[k]) if etc_basis == "netted"
                else float(ledger.rev[k]))
        lim = max(0.0, cap - used) / abs(af[k])
        if lim < best_mw:
            best_mw, best_id = lim, ids[k]
    return float(max(0.0, best_mw)), best_id


def available_atc(pt: PTDFData, source: Bus, sink: Bus,
                  booked: Iterable[AwardLike], monitored: Monitored = "all",
                  tol: float = 1e-6, etc_basis: str = "netted",
                  rf_min: float = 0.0, sentinel: float | None = None) -> float:
    """Remaining ATC for ``source->sink`` given already-``booked`` awards (ETC).

    For each monitored line the headroom in the *push direction* of this path is
    ``F_bar_l - sign(a_l) * F^booked_l``; the available transfer is the tightest
    ``headroom_l / |a_l|`` (floored at 0). With no bookings this reduces to TTC.

    ``monitored`` also takes a ``seams_engine.FlowgateSet``, in which case the
    headroom is read on each flowgate against its own forward and reverse
    rating. ``etc_basis="firm"`` reads the existing commitments without
    crediting counterflow, and ``rf_min`` is the MOD-030 impact threshold. The
    per-line netted case with ``rf_min = 0.0`` keeps its own body, so the
    default call is unchanged; every other case routes through
    :func:`flowgate_available_atc`.

    A request that impacts no monitored constraint at the threshold is
    unbounded here, because the model carries no limit other than the monitored
    set. MOD-030 pairs its response-factor threshold with the provider's other
    operating-limit checks, which this module does not represent, so a large
    ``rf_min`` posts a capability no single number in this model bounds.

    ``sentinel`` is the value an unrated direction carries and passes through
    to :func:`flowgate_available_atc`, which reads that direction as
    unbounded. A call carrying a sentinel takes that route, because the
    per-line branch below reads ``s_nom`` in both directions and has no
    unrated direction to skip.
    """
    a = path_shift_factors(pt, source, sink)
    if (_fgset(monitored) is None and etc_basis == "netted"
            and rf_min == 0.0 and sentinel is None):
        f0 = superposed_flow(pt, booked)
        avail = np.inf
        for l in _monitored_idx(pt, monitored):
            if abs(a[l]) < tol:
                continue
            headroom = pt.s_nom[l] - np.sign(a[l]) * f0[l]
            avail = min(avail, max(0.0, headroom) / abs(a[l]))
        return float(max(0.0, avail))
    mw, _ = flowgate_available_atc(
        pt, a, monitored, ledger=_booked_ledger(pt, booked, monitored),
        etc_basis=etc_basis, rf_min=rf_min, tol=tol, sentinel=sentinel)
    return mw


def book_sequentially(pt: PTDFData, requests: Iterable[AwardLike],
                      monitored: Monitored = "all", etc_basis: str = "netted",
                      rf_min: float = 0.0, tol: float = 1e-6,
                      sentinel: float | None = None) -> list[Award]:
    """Grant requests in order, decrementing ATC by existing commitments (ETC).

    ``requests`` is a list of ``(source, sink, mw)``, :class:`Award`,
    :class:`HubAward` or :class:`Request` entries; each is granted up to the
    ATC still available after the prior bookings (``available_atc``). Granted MW
    of 0 still appears in the result (a fully-curtailed request). ``monitored``,
    ``etc_basis``, ``rf_min`` and ``tol`` pass through to
    :func:`available_atc`, so the book can be rated over a flowgate set and on
    the firm reading of ETC. A granted award keeps the kind of the request it
    came from, so a hub-to-hub request is granted as a :class:`HubAward`.
    ``sentinel`` passes through as well, so a direction the catalog does not
    rate is unbounded in the booking rather than posted at the sentinel's own
    value.

    At ``rf_min = 0.0`` the cumulative award set is simultaneously feasible by
    construction, because every constraint the path impacts is consulted and
    decremented. Above it the screened-out impacts accumulate unrecorded, so a
    book can overload a monitored constraint: on the 9-bus case with
    ``line_4`` rated 40 MW, one request from bus 3 to bus 5 at ``rf_min = 0.9``
    is granted 300 MW, against 160.2 MW at ``rf_min = 0.0``, and loads
    ``line_4`` to 187 percent and ``line_2`` to 150 percent. That residual is
    MOD-030's known gap between a per-flowgate posting and the holistic test,
    which :func:`simultaneous_feasibility` measures. It is the reason the
    threshold is a dial here and not a default.

    A request that impacts no monitored constraint at the threshold is granted
    in full, because the model carries no limit other than the monitored set
    and MOD-030 pairs its threshold with operating-limit checks this module
    does not represent.
    """
    booked: list[Award] = []
    for req in requests:
        r = _as_award(req)
        if isinstance(r, HubAward):
            # A hub-to-hub award has no two buses to name, so the remaining
            # capability is read from its own loading vector.
            avail, _ = flowgate_available_atc(
                pt, _award_sf(pt, r), monitored,
                ledger=_booked_ledger(pt, booked, monitored),
                etc_basis=etc_basis, rf_min=rf_min, tol=tol,
                sentinel=sentinel)
        else:
            avail = available_atc(pt, r.source, r.sink, booked,
                                  monitored=monitored, tol=tol,
                                  etc_basis=etc_basis, rf_min=rf_min,
                                  sentinel=sentinel)
        booked.append(_regrant(r, min(r.mw, avail)))
    return booked


# ──────────────────────────────────────────────────────────────────────────
# Rights books - one per authority, under MOD-029 and under MOD-030
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RightsBook:
    """One authority's granted book and the posting table behind it.

    ``awards[i]`` is the grant on ``requests[i]``, and both lists are in the
    book's served order, which is ``priority`` then insertion order. A zero
    grant keeps its row, as :func:`book_sequentially` does, because a fully
    curtailed request is a served request.

    ``ledger`` is the posting table of the method the book was built under.
    Under ``method="rsp"`` it has one row per rated path, which is what
    MOD-029 posts. Under ``method="afc"`` it has one row per flowgate, which
    is what MOD-030 posts. ``monitored`` is the constraint set the book was
    rated over.

    ``holder`` is the one holder the book's requests name, and is empty where
    they name several, which is what a provider's book does. ``provider`` is
    the one provider they name, and is empty where they name several. An
    authority booking its own system carries both.
    """

    holder: str
    method: str
    awards: list[Award]
    requests: list[Request]
    ledger: pd.DataFrame
    monitored: Monitored = "all"
    provider: str = ""

    @property
    def requested_mw(self) -> float:
        """Total megawatts asked for."""
        return float(sum(_as_award(r).mw for r in self.requests))

    @property
    def granted_mw(self) -> float:
        """Total megawatts granted."""
        return float(sum(a.mw for a in self.awards))


def _sorted_queue(queue: Iterable[AwardLike]) -> list[Request]:
    """The queue as :class:`Request` objects, served lowest priority first.

    ``sorted`` is stable, so requests of equal priority keep their insertion
    order, which is the order a queue is served in.
    """
    reqs = [_as_request(r) for r in queue]
    return sorted(reqs, key=lambda r: r.priority)


def _book_holder(reqs: Iterable[Request]) -> str:
    """The book's holder: the one holder its requests name, else empty."""
    hs = {r.holder for r in reqs}
    return hs.pop() if len(hs) == 1 else ""


def _book_provider(reqs: Iterable[Request]) -> str:
    """The book's provider: the one provider its requests name, else empty.

    A provider's book names one provider and several holders, so the book is
    named by its provider where it cannot be named by its holder.
    """
    ps = {r.provider for r in reqs}
    return ps.pop() if len(ps) == 1 else ""


def rights_book_rsp(pt: PTDFData, queue: Iterable[AwardLike], own: Monitored,
                    ttc: str = "contract_path", tol: float = 1e-6,
                    sentinel: float | None = None) -> RightsBook:
    """An authority's rights book under the Rated System Path method (MOD-029).

    Each request is rated ALONE on the authority's own constraint set: the
    transfer is ramped on the otherwise unloaded network until the first own
    constraint reaches its rating, which identifies the element that first
    binds, its rating ``cap``, and the path's flow-scaled Total Transfer
    Capability ``TTC = cap / |a|``. Existing transmission commitments ``ETC``
    are the megawatts already booked ON THAT ELEMENT, counted in transfer
    megawatts, and the grant is added to that element's ETC.

    ``ttc`` selects what the path is sold at.

    ``"contract_path"``, the default, grants ``min(requested, cap - ETC)``.
    The path is sold one for one at the posted capability of the element that
    rates it, which is how a Rated System Path is posted: Bonneville's
    available transfer capability implementation document rates each path
    under the Rated System Path methodology and posts a figure in path
    megawatts that a customer reserves against megawatt for megawatt
    (``bpa_atcid2024``). A megawatt of reservation consumes a megawatt of the
    posted path, whatever the transfer's shift factor on the rating element.

    ``"flow_scaled"`` grants ``min(requested, TTC - ETC)``, so the reservation
    is charged the flow it puts on the rating element rather than its own
    megawatts. ``TTC`` is the transfer-study figure for the request's own
    source-to-sink pair. It is kept as the record of the rule this function
    applied before 2026-09-22 and is not a posting practice this series
    claims.

    Two properties follow from rating each path alone, and both belong to the
    method rather than to this implementation.

    1. There is no impact scaling. A path's commitments are held against its
       first-binding element at one megawatt of ETC per megawatt of transfer,
       whatever the shift factor, so two paths that load the same element at
       different rates are charged the same. Nothing else the request loads is
       charged at all.

       Under ``bpa_atcid2024`` a Rated System Path's posted total transfer
       capability is the output of a transfer study on the path, and a
       reservation consumes its own megawatts of that posted figure one for
       one. This function reads the rating element's catalog rating as the
       posted figure, which is the Lab's stand-in for that study. The account
       is therefore one megawatt of existing commitments per megawatt of
       reservation against the path's rating, and both sides are in path
       megawatts rather than in megawatts of flow on the rating element.
       Where a request loads the binding element at ``|a| < 1`` per megawatt
       the element is under-sold by the factor ``|a|``: on the triangle of 112
       section 4 the three 40 MW grants put 15.39, 29.85 and 14.45 MW on the
       40 MW ``line_4``. Where ``|a|`` is near one, which is a transfer across
       the interface the path defines, the two readings agree. The ledger's
       ``loading_per_mw`` column carries ``|a|`` per row, so the factor is
       visible where the under-sale matters.
    2. There is no cross-constraint interaction. Two paths first bound by the
       same element share that element's ETC, but a path first bound elsewhere
       is charged nothing on it even when it loads it. The union of several
       such books therefore over-assigns the network, which is the
       over-assignment :func:`interface_sft` measures for the same set, and
       which :func:`nominate_and_test` resolves over the union.

    A request that impacts no own constraint is granted in full and its
    limiting element is recorded as ``None``. That is the loop-flow blind
    spot: the authority rates through its own elements only, so a transfer
    that loads a neighbour's element and none of its own is unconstrained
    here. Rating through the holder's own set is the teaching simplification of
    the MOD-029 R1/R2 transfer study, and a deviation from it: MOD-029 R1 rates
    a path on a wide-area model that includes contiguous and joint-operating
    agreement linked areas, so the standard's own study does capture parallel
    flow through a neighbour's elements, and this one does not.

    The request itself is the point-to-point capacity reservation of
    ``hogan1997reservations``, which the :class:`Award` object carries as a
    source, a sink and a quantity. The per-path accounting here is the
    link-based decomposition that paper rejects, because a reservation charged
    to one path is a right on a link rather than on a source-to-sink pair, and
    the coordination it calls for in place of that decomposition is the common
    test :func:`nominate_and_test` runs.

    Where network integration service registers is read two ways in practice.
    It can register on its first-binding path, as a point-to-point reservation
    does, or by its impact on every path the authority rates. This function
    takes the first reading, so a ``kind="NITS"`` request is served exactly as
    a ``"PTP"`` request is, and the second reading is the MOD-030 book
    :func:`rights_book_afc` builds.

    ``own`` is the authority's own constraint set: a ``seams_engine``
    ``FlowgateSet``, a list of line names, or ``"all"``. Returns a
    :class:`RightsBook` whose ledger has one row per rated path. The ledger's
    ``ttc_mw`` column is the flow-scaled transfer limit and ``cap_mw`` is the
    posted rating of the element that sets it, so both readings are on the
    table whichever one the grant was taken from. ``loading_per_mw`` is the
    request's own ``|a|`` on that element, and is ``NaN`` where the request
    impacts no own constraint. The ``atc_mw`` column is the availability the
    grant was taken against, so it follows ``ttc``: ``cap_mw - etc_mw`` under
    ``"contract_path"`` and ``ttc_mw - etc_mw`` under ``"flow_scaled"``.

    ``sentinel`` is the value an unrated direction carries, compared by
    equality, and a direction carrying it is skipped in the rating. It is
    load-bearing under ``"contract_path"``, because the grant is the binding
    row's rating and a sentinel read as a rating would be posted as
    capability. ``sentinel is None`` reads every rating as a number.
    """
    if ttc not in ("contract_path", "flow_scaled"):
        raise ValueError("ttc must be 'contract_path' or 'flow_scaled', "
                         f"got {ttc!r}")
    reqs = _sorted_queue(queue)
    etc: LineValues = {}
    awards, rows = [], []
    for r in reqs:
        aw = r.to_award()
        sf = _award_sf(pt, aw)
        ttc_mw, path, cap = _ttc_over(pt, sf, own, tol, sentinel=sentinel)
        posted = cap if ttc == "contract_path" else ttc_mw
        if path is None:
            used, avail, grant = 0.0, float("inf"), aw.mw
            loading = float("nan")
        else:
            used = float(etc.get(path, 0.0))
            avail = max(0.0, posted - used)
            grant = min(aw.mw, avail)
            etc[path] = used + grant
            af, _cf, _cr, ids = _constraint_loading(pt, sf, own)
            loading = abs(float(af[ids.index(path)]))
        awards.append(_regrant(aw, grant))
        rows.append({"holder": r.holder, "label": r.label, "kind": r.kind,
                     "source": r.source_name, "sink": r.sink_name,
                     "requested_mw": aw.mw, "ttc_mw": float(ttc_mw),
                     "cap_mw": float(cap), "loading_per_mw": loading,
                     "etc_mw": used, "atc_mw": avail,
                     "granted_mw": float(grant), "limiting": path})
    cols = ["holder", "label", "kind", "source", "sink", "requested_mw",
            "ttc_mw", "cap_mw", "loading_per_mw", "etc_mw", "atc_mw",
            "granted_mw", "limiting"]
    ledger = pd.DataFrame(rows, columns=cols)
    return RightsBook(holder=_book_holder(reqs), method="rsp", awards=awards,
                      requests=reqs, ledger=ledger, monitored=own,
                      provider=_book_provider(reqs))


def rights_book_afc(pt: PTDFData, queue: Iterable[AwardLike], own: Monitored,
                    rf_min: float = 0.05, etc_basis: str = "firm",
                    tol: float = 1e-6,
                    sentinel: float | None = None) -> RightsBook:
    """An authority's rights book under the Available Flowgate Capability
    method (MOD-030).

    Every request is booked against every own flowgate it impacts at or above
    the response-factor threshold ``rf_min``, each charged at its own impact,
    which is :func:`book_sequentially` over the authority's own constraint
    set. ``etc_basis="firm"`` reads the existing commitments without crediting
    counterflow, which is the firm service standard, and is the default here
    because a flowgate posting is sold as firm capability. The firm default is
    a stated deviation from MOD-030's netted available flowgate capability,
    which computes the capability against the net impact of existing
    commitments and therefore does credit counterflow. ``etc_basis="netted"``
    is the standard's own reading.

    The contrast with :func:`rights_book_rsp` is the whole of the two
    methods: MOD-029 charges a path's megawatts to one element, and MOD-030
    charges a path's impact to every element it reaches. Returns a
    :class:`RightsBook` whose ledger has one row per flowgate, with the two
    ratings, the three commitment sums, and the remaining capability in each
    direction.

    ``sentinel`` is the value an unrated direction carries, compared by
    equality, and a direction carrying it is unbounded in the booking rather
    than posted at the sentinel's own value. The reported ledger keeps the
    sentinel, as the reported table does everywhere in this module.
    ``sentinel is None`` reads every rating as a number.
    """
    reqs = _sorted_queue(queue)
    awards = book_sequentially(pt, [r.to_award() for r in reqs], monitored=own,
                               etc_basis=etc_basis, rf_min=rf_min, tol=tol,
                               sentinel=sentinel)
    led = _booked_ledger(pt, awards, own)
    _, cap_fwd, cap_rev, ids = _constraint_loading(pt, np.zeros(pt.n_line), own)
    return RightsBook(holder=_book_holder(reqs), method="afc", awards=awards,
                      requests=reqs,
                      ledger=led.table(ids, cap_fwd, cap_rev,
                                       etc_basis=etc_basis),
                      monitored=own, provider=_book_provider(reqs))


# ──────────────────────────────────────────────────────────────────────────
# The common test over the union of every book, and the reduction family
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Nomination:
    """The outcome of the common simultaneous feasibility test.

    ``entitlements`` has one row per nominated award: ``holder``,
    ``provider``, ``label``, ``kind``, ``tier``, ``source``, ``sink``,
    ``nominated``, ``entitled``, ``haircut_frac`` and ``limiting``. ``by_ba``
    aggregates it per holder and adds three fractions: ``kept_frac``, the
    share of its own nomination the holder keeps; ``haircut_frac``, the share
    of its own nomination it loses; and ``haircut_share``, the holder's share
    of the total reduction across every holder. ``by_provider`` is the same
    table grouped by provider, which is the side of the account the provider's
    return test reads.
    ``loadings`` has one row per constraint, before and after the reduction.
    ``duals`` holds the constraint multipliers where the rule produces them,
    and is empty where it does not. ``iterations`` counts the reduction steps
    taken. ``awards`` holds the nominated award objects in the row order of
    ``entitlements``, so a settlement reads each row's two points from the
    award that produced it.
    """

    entitlements: pd.DataFrame
    by_ba: pd.DataFrame
    loadings: pd.DataFrame
    duals: LineValues
    iterations: int
    reduction: str
    direction: str
    by_provider: pd.DataFrame = field(default_factory=pd.DataFrame)
    awards: list[Award] = field(default_factory=list)
    by_group: pd.DataFrame = field(default_factory=pd.DataFrame)


class _HolderNetted(str):
    """The ``"holder_netted"`` reading with the holder index of every award.

    The object is the string ``"holder_netted"``, so every message and every
    comparison that reads the direction as a string reads it unchanged, and it
    carries ``groups``, one integer per award naming the award's holder, which
    :func:`_directional_flows` reads to net the loadings within each holder.
    """

    def __new__(cls, groups: npt.ArrayLike) -> Self:
        obj = super().__new__(cls, "holder_netted")
        obj.groups = np.asarray(groups, dtype=int)  # type: ignore[attr-defined]  # set on the instance alone
        return obj


def _group_flows(A: FloatArray, x: FloatArray,
                 groups: npt.NDArray[np.int_]) -> FloatArray:
    """``(n_constraints, n_groups)``: each group's signed loading of every
    constraint, the awards of the group superposed with counterflow credited
    among them."""
    groups = np.asarray(groups, dtype=int)
    n_g = int(groups.max()) + 1 if groups.size else 0
    H = np.zeros((len(x), n_g))
    if groups.size:
        H[np.arange(len(x)), groups] = 1.0
    return A @ (np.asarray(x, dtype=float)[:, None] * H)


def _directional_flows(A: FloatArray, x: FloatArray,
                       direction: str) -> tuple[FloatArray, FloatArray]:
    """The constraint flows a nomination ``x`` produces, as
    ``(forward, reverse)`` non-negative arrays.

    Netted superposes the signed loadings and reports the net in whichever
    direction it points, which credits counterflow, and is the reading
    :func:`line_loadings` takes. Firm sums the two directions separately and
    credits nothing, which is the reading :func:`firm_line_loadings` takes,
    because a firm right must be deliverable when the counter-flowing rights
    are not scheduled. Holder-netted superposes the signed loadings within
    each holder's own book and sums the holders' nets direction by direction,
    so counterflow is credited inside one holder's book and not across
    holders.
    """
    if direction == "netted":
        f = A @ x
        return np.maximum(f, 0.0), np.maximum(-f, 0.0)
    if direction == "holder_netted":
        Fg = _group_flows(A, x, direction.groups)  # type: ignore[attr-defined]  # a plain str raises here
        return (np.maximum(Fg, 0.0).sum(axis=1),
                np.maximum(-Fg, 0.0).sum(axis=1))
    return np.maximum(A, 0.0) @ x, np.maximum(-A, 0.0) @ x


def _unbounded_caps(cap_fwd: FloatArray, cap_rev: FloatArray,
                    sentinel: float | None) -> tuple[FloatArray, FloatArray]:
    """The two rating arrays with every unrated direction set to infinity.

    A direction carrying ``sentinel`` is one the catalog does not rate, not
    one rated at the sentinel's value. A reduction rule reads a rating as a
    number, so the sentinel would otherwise act as a rating: it would report
    an overload at a flow above it, set the scale-back factor, and name
    itself as a limiting row. Replacing it with infinity makes the direction
    unbounded, which is the ``FlowgateSet`` convention that the sentinel never
    binds. The originals are left untouched for the reported table, where an
    unrated direction reads ``rated=False``.

    ``sentinel is None`` returns copies unchanged, so a caller that does not
    use the convention is unaffected. The comparison is by equality, never by
    size, because a genuine rating can sit above the sentinel.
    """
    f = np.asarray(cap_fwd, dtype=float).copy()
    r = np.asarray(cap_rev, dtype=float).copy()
    if sentinel is not None:
        f[f == float(sentinel)] = np.inf
        r[r == float(sentinel)] = np.inf
    return f, r


def _overloads(A: FloatArray, x: FloatArray, cap_fwd: FloatArray,
               cap_rev: FloatArray, direction: str,
               tol: float) -> list[tuple[int, float, float]]:
    """Every overloaded constraint as ``(row, side, scale)``.

    ``scale`` is the constraint's own scale-back factor ``cap / loading``, the
    fraction of the present loading the constraint can carry. ``side`` is
    ``+1`` forward and ``-1`` reverse.
    """
    fwd, rev = _directional_flows(A, x, direction)
    out = []
    for k in range(len(cap_fwd)):
        if fwd[k] > cap_fwd[k] + tol:
            out.append((k, 1.0, float(cap_fwd[k] / fwd[k])))
        if rev[k] > cap_rev[k] + tol:
            out.append((k, -1.0, float(cap_rev[k] / rev[k])))
    return out


def _reduce_pro_rata(A: FloatArray, t: FloatArray, cap_fwd: FloatArray,
                     cap_rev: FloatArray, ids: list[Line], direction: str,
                     max_iter: int, tol: float, rf_min: float = 0.05,
                     tier: npt.NDArray[np.int_] | None = None,
                     ) -> tuple[FloatArray, int, list[Line | None]]:
    """Cut the most overloaded constraint's material contributors, highest
    curtailment tier first, by the megawatts that constraint is over.

    Each iteration takes the most overloaded constraint ``k``, the one with
    the smallest scale-back factor, and removes its excess
    ``loading_k - cap_k`` from that constraint alone.

    A contributor is an award whose response factor on that constraint in the
    overloaded direction, ``A[k, i] * sgn``, reaches ``max(rf_min, tol)``.
    That is the materiality screen of the MOD-030 response-factor threshold,
    read here on the base-case shift factor rather than on the outage transfer
    distribution factor, which is the deviation
    :func:`flowgate_available_atc` already states. It exists because an award
    that loads a constraint at a hundredth of a megawatt per megawatt is not a
    party to that constraint's overload, and cutting it removes almost no flow
    while removing a great deal of service. The floor at ``tol`` is the same
    floor :func:`flowgate_available_atc` applies, and it is what keeps an
    award with no response on the row out of the screen at ``rf_min = 0.0``:
    an award that puts nothing on the constraint is not a party to its
    overload under any threshold.

    The contributors are cut by tier, highest first: 3 non-firm, then 2
    short-term firm, then 1 long-term firm, then 0 network. Within a tier the
    cut is pro rata to each contributor's own contribution ``A[k, i] * x_i``,
    so one common factor applies to that tier. A tier whose whole contribution
    is smaller than the remaining excess is set to zero and the next tier is
    taken. ``tier`` is one tier per award, and ``None`` puts every award in
    tier 1, under which the rule is per-constraint pro rata among the material
    contributors.

    Where the most overloaded constraint has no material contributor carrying
    megawatts the rule falls back to every award that loads it in the
    overloaded direction, and names the fallback in ``limiting`` as
    ``"<id> (no material contributor)"``, because a reduction that cut nothing
    would leave the constraint overloaded and the fallback has to be visible
    in the ledger.

    ``limiting`` names the constraint that FIRST cut each award. An award cut
    on one iteration keeps that iteration's name on every later iteration,
    because the reason its megawatts were reduced is the constraint that took
    them, and a later pass relabelling it would report a material contributor
    as a fallback.

    Under the firm reading the step is monotone: scaling a contributor down
    lowers its same-direction sum on every constraint and raises none, so the
    loop terminates. Under the netted reading it is not, because cutting an
    award that counterflows another constraint raises that constraint's net.
    The loop therefore runs at most ``max_iter`` times and raises
    ``RuntimeError`` naming the rows still overloaded. A rating is never
    loosened to make the test pass.

    Under the holder-netted reading a contributor must also belong to a holder
    whose own book nets onto the constraint in the overloaded direction,
    because a holder whose book nets against that direction adds nothing to
    the overload under that reading. That screen is a Lab modeling choice
    made with the reading itself. The step is not monotone under this reading
    either, so the same iteration bound and the same failure apply.

    The rule is a Lab modeling choice. It reads the pro rata curtailment
    practice of transmission loading relief under the open access transmission
    tariff, where the parties that materially load a constrained element are
    cut in proportion to their contributions and the non-firm service is cut
    before the firm service. The tariff curtailment sections and NERC's
    transmission loading relief procedure are not filed in ``sources/``, so
    this is a reading and not a transcription; ``nerc_mod_atc`` supplies the
    5 percent response-factor threshold. ``bushnellstoft1996contract`` is not
    its source. That paper states a feasibility CONDITION on which incremental
    contract set a grid modifier may take, and it carries no proportional
    reduction. It supplies the feasibility bound the common test enforces, not
    the way a reduction is shared among the parties.
    """
    n = len(t)
    x = t.astype(float).copy()
    limiting: list[Line | None] = [None] * n
    tiers = (np.ones(n, dtype=int) if tier is None
             else np.asarray(tier, dtype=int))
    if tiers.shape != (n,):
        raise ValueError(f"tier must have {n} entries, got {tiers.shape}")
    rf = max(float(rf_min), float(tol))
    for it in range(int(max_iter) + 1):
        over = _overloads(A, x, cap_fwd, cap_rev, direction, tol)
        if not over:
            return x, it, limiting
        if it == max_iter:
            rows = sorted({ids[k] for k, _, _ in over})
            raise RuntimeError(
                "pro_rata did not reach a feasible set in "
                f"{max_iter} iterations on the {direction} reading; still "
                f"overloaded: {rows}")
        k_min, sgn, _theta = min(over, key=lambda z: z[2])
        fwd, rev = _directional_flows(A, x, direction)
        loading = float(fwd[k_min] if sgn > 0 else rev[k_min])
        cap = float(cap_fwd[k_min] if sgn > 0 else cap_rev[k_min])
        excess = loading - cap
        resp = A[k_min] * sgn
        party = np.ones(n, dtype=bool)
        if direction == "holder_netted":
            # only a holder whose own book nets onto the constraint in the
            # overloaded direction is a party to its overload
            net_g = _group_flows(A[k_min:k_min + 1], x, direction.groups)[0]  # type: ignore[attr-defined]  # a plain str raises here
            party = sgn * net_g[direction.groups] > tol  # type: ignore[attr-defined]  # a plain str raises here
        material = np.nonzero((resp >= rf) & party)[0]
        name = ids[k_min]
        on_row = float(np.sum(resp[material] * x[material]))
        if material.size == 0 or on_row <= tol:
            material = np.nonzero((resp > tol) & party)[0]
            name = f"{ids[k_min]} (no material contributor)"
        for tr in sorted({int(tiers[i]) for i in material}, reverse=True):
            if excess <= tol:
                break
            grp = np.array([i for i in material if int(tiers[i]) == tr],
                           dtype=int)
            contrib = float(np.sum(resp[grp] * x[grp]))
            if contrib <= tol:
                continue
            if contrib <= excess:
                x[grp] = 0.0
                excess -= contrib
            else:
                x[grp] = x[grp] * (1.0 - excess / contrib)
                excess = 0.0
            for i in grp:
                if limiting[i] is None:
                    limiting[i] = name
    raise RuntimeError("unreachable")


def _reduce_pro_rata_global(A: FloatArray, t: FloatArray, cap_fwd: FloatArray,
                            cap_rev: FloatArray, ids: list[Line], direction: str,
                            max_iter: int, tol: float,
                            ) -> tuple[FloatArray, int, list[Line | None]]:
    """Scale back every award that loads ANY overloaded constraint, by the
    tightest constraint's single scale-back factor.

    Per overloaded constraint ``k`` the scale-back factor is
    ``s_k = cap_k / loading_k``. Every award that loads an overloaded
    constraint in the overloaded direction is multiplied by ``min_k s_k``, and
    the test is run again.

    This is the record of the rule ``reduction="pro_rata"`` applied before
    2026-09-22, kept so the numbers stored under it can be reproduced. It is
    not a posting or curtailment practice this series claims. One factor taken
    from the tightest constraint is applied to contributors to every other
    overloaded constraint as well, so the reduction removes far more than the
    excess: on the WECC map-mirror's Northwest book 101 of the 116 awards it
    cut load their constraint below the MOD-030 response-factor threshold.
    :func:`_reduce_pro_rata` is the corrected rule.
    """
    x = t.astype(float).copy()
    limiting: list[Line | None] = [None] * len(t)
    for it in range(int(max_iter) + 1):
        over = _overloads(A, x, cap_fwd, cap_rev, direction, tol)
        if not over:
            return x, it, limiting
        if it == max_iter:
            rows = sorted({ids[k] for k, _, _ in over})
            raise RuntimeError(
                "pro_rata_global did not reach a feasible set in "
                f"{max_iter} iterations on the {direction} reading; still "
                f"overloaded: {rows}")
        k_min, _, theta = min(over, key=lambda z: z[2])
        cut = np.zeros(len(t), dtype=bool)
        for k, sgn, _ in over:
            cut |= (A[k] * sgn > tol)
        x = np.where(cut, x * theta, x)
        for i in np.nonzero(cut)[0]:
            limiting[i] = ids[k_min]
    raise RuntimeError("unreachable")


def _reduce_uniform(A: FloatArray, t: FloatArray, cap_fwd: FloatArray,
                    cap_rev: FloatArray, ids: list[Line], direction: str,
                    max_iter: int, tol: float,
                    ) -> tuple[FloatArray, int, list[Line | None]]:
    """Scale EVERY award by one factor, the tightest constraint's scale-back.

    The factor is ``theta = min_k s_k`` over the overloaded constraints, and
    it is applied to the whole nomination rather than to the contributors
    alone, so an award that loads nothing overloaded is cut as much as one
    that causes the overload. The iteration and the failure are those of
    :func:`_reduce_pro_rata`.
    """
    x = t.astype(float).copy()
    limiting: list[Line | None] = [None] * len(t)
    for it in range(int(max_iter) + 1):
        over = _overloads(A, x, cap_fwd, cap_rev, direction, tol)
        if not over:
            return x, it, limiting
        if it == max_iter:
            rows = sorted({ids[k] for k, _, _ in over})
            raise RuntimeError(
                "uniform did not reach a feasible set in "
                f"{max_iter} iterations on the {direction} reading; still "
                f"overloaded: {rows}")
        k_min, _, theta = min(over, key=lambda z: z[2])
        x = x * theta
        limiting = [ids[k_min]] * len(t)
    raise RuntimeError("unreachable")


# The trust-constr solver's own accuracy. Its interior-point step stops a few
# parts in a million from the constraint boundary, so the least-squares rule
# tests feasibility, screens its multipliers and checks its own answer at this
# tolerance rather than at the caller's ``tol``, which the two iterative rules
# use because their steps are exact.
_QP_TOL = 1e-6


def _reduce_least_squares(A: FloatArray, t: FloatArray, cap_fwd: FloatArray,
                          cap_rev: FloatArray, ids: list[Line], direction: str,
                          weights: npt.ArrayLike | None, tol: float,
                          ) -> tuple[FloatArray, int, list[Line | None], LineValues]:
    """Award the feasible set closest to the nomination in weighted squared
    deviation.

    Solves ``min sum_i w_i (t_i - x_i)^2`` subject to
    ``-cap_rev_k <= sum_i af_k(i) x_i <= cap_fwd_k`` on every constraint and
    ``0 <= x_i <= t_i``, a convex quadratic program with linear constraints,
    through ``scipy.optimize.minimize(method="trust-constr")`` with an exact
    gradient and Hessian. The objective, the bounds and the starting point are
    fixed, so the solution is deterministic. The constraint multipliers are
    returned as the nomination's duals. A solve the solver refuses is retried
    once in the award nondimensionalized by the largest nomination, because
    the solver's tolerances are absolute while a nomination's size is not, and
    the retry's multipliers are scaled back into the caller's units.

    This is the least-squares reduction of the California ISO's congestion
    revenue rights allocation as the user characterizes it. Its primary
    source, the ISO tariff's congestion revenue rights allocation section and
    the congestion revenue rights Business Practice Manual, is not filed in
    ``sources/``, so the rule stated here is a reading of that description and
    not a verified transcription of the tariff. It is a Lab modeling choice
    until the primary source is filed.

    Only the netted reading has this form. The firm reading sums each
    direction separately, which is not one linear function of the award
    vector, so ``direction="firm"`` raises ``NotImplementedError``. The
    holder-netted reading sums the holders' nets direction by direction,
    which is not linear either, and raises the same error.

    A nomination that is already feasible is returned unchanged rather than
    solved for, because it is the exact solution and the interior-point solver
    would step a few parts in a million away from it. ``tol`` here is
    :data:`_QP_TOL`, the solver's own accuracy, and not the caller's ``tol``:
    it sets the feasibility test on entry, the multiplier screen, and the
    feasibility check on the answer.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    if direction != "netted":
        raise NotImplementedError(
            "least_squares is defined on the netted reading only: the firm "
            "reading sums each direction separately, so the constraint is "
            "not a single linear form in the award vector")
    n = len(t)
    if not _overloads(A, t, cap_fwd, cap_rev, direction, tol):
        return t.astype(float).copy(), 0, [None] * n, {}
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have {n} entries, got {w.shape}")
    tt = t.astype(float)

    def _solve(scale: float) -> tuple[Any, FloatArray]:
        """The program solved in the award ``y = x / scale``.

        Returns ``(result, scale)``. The objective scales by ``scale ** 2``
        and the feasible set is unchanged, so the minimizer in ``x`` is the
        same program's minimizer and only the solver's arithmetic differs.
        """
        yy = tt / scale

        def obj(y: FloatArray) -> float:
            return float(np.sum(w * (yy - y) ** 2))

        def jac(y: FloatArray) -> FloatArray:
            return -2.0 * w * (yy - y)

        def hess(y: FloatArray) -> FloatArray:
            return 2.0 * np.diag(w)

        return minimize(obj, yy.copy(), jac=jac, hess=hess,
                        method="trust-constr",
                        constraints=[LinearConstraint(A * scale, -cap_rev,
                                                      cap_fwd)],
                        bounds=Bounds(np.zeros(n), yy),
                        options=dict(gtol=1e-10, xtol=1e-14, maxiter=5000,
                                     verbose=0)), yy

    scale = 1.0
    res, yy = _solve(scale)
    if res.status not in (1, 2):
        # ``gtol`` is an ABSOLUTE tolerance on the objective's gradient, and
        # that gradient is ``2 w (t - x)``, which carries the size of the
        # nomination. A nomination of thousands of megawatts therefore cannot
        # reach the tolerance even when it is feasible and optimal: the WECC
        # map-mirror's Northwest book reaches a point violating nothing by
        # more than 4e-11 MW and is refused. The retry is the same program in
        # the nondimensional award, scaled by the largest nomination, which is
        # where the tolerance means what it says. The first attempt is kept so
        # that a nomination already near unit scale, as every 9-bus case is,
        # takes the same solver path it always did.
        scale = float(np.max(tt))
        if scale <= 0:
            raise RuntimeError(
                f"least_squares reduction failed: {res.message}")
        res, yy = _solve(scale)
        if res.status not in (1, 2):
            raise RuntimeError(
                f"least_squares reduction failed: {res.message}")
    x = np.clip(np.asarray(res.x, dtype=float), 0.0, yy) * scale
    still = _overloads(A, x, cap_fwd, cap_rev, direction, tol)
    if still:
        rows = sorted({ids[k] for k, _, _ in still})
        raise RuntimeError(
            "least_squares returned an award set that is still overloaded at "
            f"the solver's own accuracy of {tol}: {rows}")
    # A multiplier is the objective's derivative with respect to a rating, and
    # the ratings were never scaled, so the retry's multipliers come back in
    # the caller's units through the objective's own factor of ``scale ** 2``.
    v = (np.asarray(res.v[0], dtype=float) * scale * scale if len(res.v)
         else np.zeros(len(ids)))
    # The interior-point solver leaves a small multiplier on every slack row,
    # so the reported duals are screened relative to the largest one.
    thr = max(tol, tol * float(np.max(np.abs(v))) if len(v) else tol)
    duals = {ids[k]: float(v[k]) for k in range(len(ids)) if abs(v[k]) > thr}
    limiting: list[Line | None] = []
    for i in range(n):
        if tt[i] - x[i] <= tol:
            limiting.append(None)
            continue
        hit = [(abs(v[k]), ids[k]) for k in range(len(ids))
               if abs(A[k, i]) > tol and abs(v[k]) > tol]
        limiting.append(max(hit)[1] if hit else None)
    return x, 1, limiting, duals


def _loading_table(A: FloatArray, t: FloatArray, x: FloatArray,
                   cap_fwd: FloatArray, cap_rev: FloatArray, ids: list[Line],
                   direction: str, sentinel: float | None,
                   tol: float) -> pd.DataFrame:
    """The per-constraint table, before and after the reduction.

    Each row reports the direction the REDUCED nomination puts the most
    megawatts on, ties to forward, and reports the pre-reduction flow in that
    same direction so the pair is comparable. The direction is chosen on
    megawatts rather than on loading so that a row whose flow sits on an
    unrated direction reports that direction, which is where the flow is. An
    unrated direction carries the sentinel, which is compared by equality and
    never by size, and its loading is not a number, because an unrated
    direction has no rating to be a fraction of. ``binding`` is a rated row
    whose post-reduction loading reaches its rating.

    Note that the reduction tests BOTH directions of every constraint against
    their own ratings, so the feasibility of the entitled set does not depend
    on which direction a row reports.

    A row reporting its reverse direction carries a NEGATIVE flow here, because
    the sign names the direction. :func:`firm_line_loadings` reports the same
    quantity as a positive ``rev`` column beside a positive ``fwd`` column, so
    the two tables agree in magnitude and differ in sign on a reverse row.
    """
    fwd_pre, rev_pre = _directional_flows(A, t, direction)
    fwd_post, rev_post = _directional_flows(A, x, direction)
    rows = []
    for k in range(len(ids)):
        cf, cr = float(cap_fwd[k]), float(cap_rev[k])
        rated_f = sentinel is None or cf != sentinel
        rated_r = sentinel is None or cr != sentinel
        if fwd_post[k] >= rev_post[k]:
            cap, rated = cf, rated_f
            f_pre, f_post = fwd_pre[k], fwd_post[k]
        else:
            cap, rated = cr, rated_r
            f_pre, f_post = -rev_pre[k], -rev_post[k]
        load_post = (abs(f_post) / cap) if (rated and cap > 0) else np.nan
        load_pre = (abs(f_pre) / cap) if (rated and cap > 0) else np.nan
        binding = bool(rated and not np.isnan(load_post)
                       and load_post >= 1.0 - 1e-9)
        rows.append({"id": ids[k], "flow_pre": float(f_pre),
                     "flow_post": float(f_post), "cap": cap,
                     "loading_pre": float(load_pre),
                     "loading_post": float(load_post),
                     "rated": bool(rated), "binding": binding})
    return pd.DataFrame(rows).set_index("id")


def _group_table(ent: pd.DataFrame, t: FloatArray, x: FloatArray,
                 key: str) -> pd.DataFrame:
    """The per-party table of a nomination, grouped on ``key``.

    ``by_ba`` groups on ``holder`` and ``by_provider`` on ``provider``, and
    both are this function, because the two tables report the same quantities
    on the two sides of the same account. ``t`` and ``x`` are the nominated
    and entitled vectors, read for the total reduction so that
    ``haircut_share`` is a share of the same total in both tables.
    """
    total_cut = float(np.sum(t - x))
    rows = []
    for name, g in ent.groupby(key, sort=True):
        nom, entl = float(g["nominated"].sum()), float(g["entitled"].sum())
        cut = nom - entl
        rows.append({key: name, "nominated": nom, "entitled": entl,
                     "haircut_mw": cut,
                     "kept_frac": 1.0 if nom <= 0 else entl / nom,
                     "haircut_frac": 0.0 if nom <= 0 else cut / nom,
                     "haircut_share": 0.0 if total_cut <= 0 else cut / total_cut})
    return pd.DataFrame(rows, columns=[key, "nominated", "entitled",
                                       "haircut_mw", "kept_frac",
                                       "haircut_frac", "haircut_share"])


def nominate_and_test(books: Iterable[RightsBook], pt: PTDFData, fgs: Monitored,
                      direction: str = "netted", reduction: str = "pro_rata",
                      sentinel: float | None = None,
                      max_iter: int | None = None,
                      weights: npt.ArrayLike | None = None,
                      tol: float = 1e-9, rf_min: float = 0.05) -> Nomination:
    """The common simultaneous feasibility test over the union of every book.

    Each authority rates its own requests on its own constraint set, so the
    union of the books is in general not feasible on the shared network. This
    function nominates every award in every book onto ALL of ``fgs``, the
    shared constraint set, tests the union, and reduces it to a feasible set
    under one rule.

    Running one test over the union is the coordination
    ``hogan1997reservations`` requires once capacity reservations are held
    against paths rather than against source-to-sink pairs, and it is Concept 3
    of ``msc_cra_update2026``: one common simultaneous feasibility test over the
    open access tariff rights and the congestion revenue rights together, so
    that the two kinds of right are not tested against the network separately.

    ``direction="netted"`` superposes the signed loadings, which credits
    counterflow and is the test behind revenue adequacy for financial rights
    (``hogan2000flowgate``). ``direction="firm"`` sums each direction without
    crediting counterflow, which is the standard a firm point-to-point right
    is sold under. ``direction="holder_netted"`` superposes the signed
    loadings within each holder's own book and sums the holders' nets
    direction by direction, so counterflow is credited within one holder's
    book and not across holders. That is the reading under which a holder
    nets its own schedules and no holder is credited with another holder's
    counterflow, which is a Lab construct, stated as such: the union of
    independently built books is tested under it beside the two readings
    above, and it lies between them, since each holder's net is at most its
    firm sum and the holders' nets sum to at least the union's net.
    ``reduction`` is ``"pro_rata"``, ``"pro_rata_global"``,
    ``"uniform"`` or ``"least_squares"``, each documented on its own function.
    The least-squares rule is defined on the netted reading only.
    ``rf_min`` is the response-factor threshold the ``"pro_rata"`` rule
    screens contributors on, and the tiers that rule cuts in are read from
    each request's ``tier``. ``"pro_rata_global"`` is the record of the rule
    ``"pro_rata"`` named before 2026-09-22 and reads neither. ``max_iter``
    bounds the iterative rules and defaults to the number of constraints.
    ``sentinel`` is the value an unrated direction carries, compared by
    equality. An unrated direction is UNBOUNDED in the reduction: its cap is
    replaced by infinity before any rule reads it, so the sentinel never acts
    as a rating, never sets a scale-back factor, and never names a limiting
    row. That is the ``FlowgateSet`` convention, where the sentinel marks a
    direction the path catalog does not rate rather than a large rating. The
    reported table keeps the sentinel, so an unrated row reads
    ``rated=False`` with no loading. ``tol`` bounds the two iterative rules,
    and the least-squares rule uses :data:`_QP_TOL`, its solver's own
    accuracy.

    Returns a :class:`Nomination`. No award is ever raised above its
    nomination and no rating is ever loosened, so the entitled set is a subset
    of the nominated one and every rated row ends within its rating.
    """
    if direction not in ("netted", "firm", "holder_netted"):
        raise ValueError(
            "direction must be 'netted', 'firm' or 'holder_netted', got "
            f"{direction!r}")
    if reduction not in ("pro_rata", "pro_rata_global", "uniform",
                         "least_squares"):
        raise ValueError(
            "reduction must be 'pro_rata', 'pro_rata_global', 'uniform' or "
            f"'least_squares', got {reduction!r}")

    meta: list[dict[str, Any]]
    meta, cols, awards = [], [], []
    for bk in books:
        for aw, rq in zip(bk.awards, bk.requests):
            r = _as_request(rq)
            holder = r.holder or bk.holder
            meta.append({"holder": holder,
                         "provider": r.provider or bk.provider or holder,
                         "label": r.label, "kind": r.kind, "tier": int(r.tier),
                         "source": r.source_name, "sink": r.sink_name,
                         "nominated": float(aw.mw)})
            cols.append(_award_sf(pt, aw))
            awards.append(aw)
    _, cap_fwd, cap_rev, ids = _constraint_loading(pt, np.zeros(pt.n_line), fgs)
    n = len(meta)
    if n:
        A = np.asarray(
            _constraint_loading(pt, np.column_stack(cols), fgs)[0],
            dtype=float).reshape(len(ids), n)
    else:
        A = np.zeros((len(ids), 0))
    t = np.array([m["nominated"] for m in meta], dtype=float)
    reading = direction
    if direction == "holder_netted":
        hix = {h: j for j, h in enumerate(sorted({m["holder"] for m in meta}))}
        reading = _HolderNetted([hix[m["holder"]] for m in meta])

    red_fwd, red_rev = _unbounded_caps(cap_fwd, cap_rev, sentinel)
    if max_iter is None:
        max_iter = max(1, len(ids))
    duals: LineValues = {}
    if reduction == "least_squares":
        x, iters, limiting, duals = _reduce_least_squares(
            A, t, red_fwd, red_rev, ids, reading, weights, _QP_TOL)
    elif reduction == "uniform":
        x, iters, limiting = _reduce_uniform(
            A, t, red_fwd, red_rev, ids, reading, max_iter, tol)
    elif reduction == "pro_rata_global":
        x, iters, limiting = _reduce_pro_rata_global(
            A, t, red_fwd, red_rev, ids, reading, max_iter, tol)
    else:
        x, iters, limiting = _reduce_pro_rata(
            A, t, red_fwd, red_rev, ids, reading, max_iter, tol,
            rf_min=rf_min,
            tier=np.array([m["tier"] for m in meta], dtype=int))

    ent = pd.DataFrame(meta, columns=["holder", "provider", "label", "kind",
                                      "tier", "source", "sink", "nominated"])
    ent["entitled"] = x
    ent["haircut_frac"] = [
        0.0 if t[i] <= 0 else float((t[i] - x[i]) / t[i]) for i in range(n)]
    ent["limiting"] = limiting

    return Nomination(
        entitlements=ent, by_ba=_group_table(ent, t, x, "holder"),
        by_provider=_group_table(ent, t, x, "provider"),
        awards=awards,
        loadings=_loading_table(A, t, x, cap_fwd, cap_rev, ids, reading,
                                sentinel, tol),
        duals=duals, iterations=int(iters), reduction=reduction,
        direction=str(direction))


def settle_rights(nomination: Nomination | Iterable[AwardLike],
                  lmp: BusValues | None = None, *, form: str = "obligation",
                  scope: str = "nodal", pt: PTDFData | None = None,
                  duals: LineValues | None = None) -> pd.DataFrame:
    """Settle an entitled set against a clearing's nodal prices.

    Each entitled right is a source-to-sink claim, so its value is its
    megawatts times the price difference between its two points:

        ``dlambda_i = sum_b sink_w_i[b] lmp[b] - sum_b source_w_i[b] lmp[b]``

    with the two weight maps taken from the award the row came from
    (``nomination.awards``), and a plain :class:`Award` read as one megawatt
    at one bus. Under ``form="obligation"`` the value is
    ``entitled_i * dlambda_i`` and is signed, so a right pointed against the
    price gradient pays. Under ``form="option"`` it is
    ``max(0, entitled_i * dlambda_i)``, so the holder exercises only where the
    right is in the money.

    Columns: ``holder``, ``provider``, ``label``, ``kind``, ``tier``,
    ``entitled``, ``dlambda``, ``value``.

    A matched book, one right per source-to-sink delivery at the dispatch,
    settles to the clearing's own congestion rent, because the sum of the
    price differences over the deliveries is what load pays less what
    generation is paid. That is the revenue adequacy of ``hogan2000flowgate``
    read on this module's objects.

    An option book is revenue adequate only against a set tested WITHOUT
    crediting counterflow, because the holder drops the legs that would have
    paid. The function therefore warns when ``form="option"`` and the
    nomination was tested on the netted reading; ``direction="firm"`` is the
    test an option book needs.

    ``scope="constraints"`` values each right on the constraints in scope
    instead of at the nodal prices. ``duals`` is ``{line: mu_m}``, the signed
    shadow prices of the constraints in scope (a flowgate clearing passes its
    implied per-line duals ``S^T mu_fg``), and ``pt`` the shift factors. The
    price difference is then

        ``dlambda_i = -sum_(m in duals) mu_m (SF_src(i),m - SF_snk(i),m)``

    with each end's shift factor read through the award's weight map
    (:func:`_award_sf`), and the right's value is
    ``v_i = entitled_i * dlambda_i`` as before. The table gains one column
    ``v[m]`` per constraint in scope, the obligation value of the right's
    loading of that constraint, ``-mu_m entitled_i (SF_src(i),m -
    SF_snk(i),m)``; these columns sum to the obligation value, and under
    ``form="option"`` they stay the signed per-constraint contributions while
    ``value`` is the positive part of their sum. This is the valuation
    ``holder_regimes`` states, and it is the DC form of the right's value in
    ``hogan2000flowgate``, ``p = H^T mu``: a right's value is its
    distribution factors times the shadow prices. ``lmp`` is not read under
    this scope.

    The nodal price is the area energy price plus ``sum_m SF_n,m mu_m``
    (``lmp_n = lhat_a(n) + sum_m SF_n,m mu_m``). Where ``duals`` carries
    every binding constraint, the nodal value of a right therefore differs
    from its constraint value by the signed amount

        ``nodal_i - constraint_i = entitled_i (lhat_snk(i) - lhat_src(i))``,

    with each end's energy price ``lhat`` read through the award's weight
    map, so a :class:`HubAward` whose point spans several areas reads a
    weighted mix of their energy prices. The gap is zero wherever every
    declared interface is slack, since every area's energy price is then the
    same; where an interface binds, it is positive for a right pointed from
    the low-priced area to the high-priced one and negative for a right
    pointed the other way.

    ``nomination`` may also be a list of :class:`Request` or :class:`Award`
    objects, which is valued as a nomination whose every right is entitled
    at its nominated megawatts (:func:`value_requests`).
    """
    if not hasattr(nomination, "entitlements"):
        nomination = _requests_nomination(nomination)
    if form not in ("obligation", "option"):
        raise ValueError(
            f"form must be 'obligation' or 'option', got {form!r}")
    if scope not in ("nodal", "constraints"):
        raise ValueError(
            f"scope must be 'nodal' or 'constraints', got {scope!r}")
    if scope == "constraints" and (pt is None or duals is None):
        raise ValueError("scope='constraints' needs pt= and duals=")
    if scope == "nodal" and lmp is None:
        raise ValueError("scope='nodal' needs the clearing's lmp")
    if form == "option" and getattr(nomination, "direction", "") in (
            "netted", "holder_netted"):
        warnings.warn(
            f"an option book was tested on the {nomination.direction} "  # type: ignore[union-attr]  # a list became a Nomination above
            "reading, which credits counterflow; an option is feasible only "
            "under the firm test",
            stacklevel=2)
    ent = nomination.entitlements
    awards = list(nomination.awards)  # type: ignore[union-attr]  # a list became a Nomination above
    if len(awards) != len(ent):
        raise ValueError(
            f"the nomination carries {len(awards)} awards for {len(ent)} "
            "entitlement rows")

    def _w(award: Award, side: str) -> BusValues:
        w = getattr(award, f"{side}_w", None)
        return dict(w) if w else {str(getattr(award, side)): 1.0}

    def _price(w: BusValues) -> float:
        return float(sum(float(v) * float(lmp[b]) for b, v in w.items()))  # type: ignore[index, misc]  # refused above on the nodal scope

    lines = [str(m) for m in duals] if scope == "constraints" else []  # type: ignore[union-attr]  # refused above when None
    mu = {str(m): float(v) for m, v in (duals or {}).items()}
    rows = []
    for i, (_idx, r) in enumerate(ent.iterrows()):
        aw = awards[i]
        q = float(r["entitled"])
        per = {}
        if scope == "nodal":
            dl = _price(_w(aw, "sink")) - _price(_w(aw, "source"))
        else:
            sf = _award_sf(pt, aw)  # type: ignore[arg-type]  # refused above when None
            per = {m: -mu[m] * float(sf[pt.line_idx[m]]) for m in lines}  # type: ignore[union-attr]  # refused above when None
            dl = float(sum(per.values()))
        val = q * dl
        rows.append({"holder": r["holder"], "provider": r["provider"],
                     "label": r["label"], "kind": r["kind"],
                     "tier": int(r["tier"]), "entitled": q, "dlambda": dl,
                     "value": val if form == "obligation" else max(0.0, val),
                     **{f"v[{m}]": q * per[m] for m in lines}})
    return pd.DataFrame(rows, columns=["holder", "provider", "label", "kind",
                                       "tier", "entitled", "dlambda",
                                       "value"] + [f"v[{m}]" for m in lines])


def _requests_nomination(items: Iterable[AwardLike]) -> Nomination:
    """A list of :class:`Request` or :class:`Award` objects (or
    ``(source, sink, mw)`` tuples) as a :class:`Nomination` whose every right
    is entitled at its nominated megawatts, so that :func:`settle_rights`
    values the list as it stands. No feasibility test is run, and the
    ``by_ba``, ``loadings`` and ``duals`` fields are empty."""
    reqs = [_as_request(r) for r in items]
    awards = [r.to_award() for r in reqs]
    ent = pd.DataFrame(
        [{"holder": r.holder, "provider": r.provider, "label": r.label,
          "kind": r.kind, "tier": r.tier, "source": r.source_name,
          "sink": r.sink_name, "nominated": r.mw, "entitled": r.mw,
          "haircut_frac": 0.0, "limiting": ""} for r in reqs],
        columns=["holder", "provider", "label", "kind", "tier", "source",
                 "sink", "nominated", "entitled", "haircut_frac", "limiting"])
    return Nomination(entitlements=ent, by_ba=pd.DataFrame(),
                      loadings=pd.DataFrame(), duals={}, iterations=0,
                      reduction="none", direction="", awards=awards)


def value_requests(requests: Iterable[AwardLike], lmp: BusValues | None = None, *,
                   form: str = "obligation", scope: str = "nodal",
                   pt: PTDFData | None = None,
                   duals: LineValues | None = None) -> pd.DataFrame:
    """The value of a list of rights at a clearing's prices, each right at
    its full megawatts.

    ``requests`` is a list of :class:`Request` or :class:`Award` objects. The
    list is read as a nomination whose every right is entitled at its
    nominated megawatts, and :func:`settle_rights` values it under the same
    ``form``, ``scope``, ``pt`` and ``duals``, so the table and every column
    are that function's. A right's value is therefore its megawatts times the
    price difference between its sink and its source under the nodal scope,
    and its megawatts times ``-sum_m mu_m (SF_src,m - SF_snk,m)`` under the
    constraint scope. This is the entry point for a book valued before any
    feasibility test, such as the book a clearing's dispatch issues
    (:func:`dispatch_requests`).
    """
    return settle_rights(_requests_nomination(requests), lmp, form=form,
                         scope=scope, pt=pt, duals=duals)


def union_test(books_by_group: Mapping[str, BookGroup], pt: PTDFData,
               fgs: Monitored, **kw: Unpack[NominateOptions]) -> Nomination:
    """The common simultaneous feasibility test over the union of several
    groups' books, each row tagged with its group, and κ per group.

    ``books_by_group`` is ``{group: books}``, where ``books`` is a list of
    :class:`RightsBook`, a mapping whose values are books, or one book. The
    union of every group's books is passed to :func:`nominate_and_test` in
    the order the groups are given, so the test, the reduction and every
    table it returns are that function's own, and keyword arguments reach it
    unchanged.

    The result's ``entitlements`` gains a ``group`` column, and its
    ``by_group`` is the per-party table of :func:`nominate_and_test` grouped on
    that column and indexed by group, so a group's kept fraction is
    ``by_group.loc[group, "kept_frac"]``. That fraction is the group's
    ``κ^M = sum_(k in M) x~_k / sum_(k in M) x_k``. A group whose books carry
    no award keeps a row with nothing nominated and a kept fraction of one,
    because nothing of its nomination was cut.

    A group is the footprint whose engine built the book, as the experiment
    reads it, and not the holder, so a footprint's κ counts every right its
    dispatch issued, the boundary legs held outside it included.
    """
    groups, seq, tags = list(books_by_group), [], []
    for g in groups:
        bks = books_by_group[g]
        if isinstance(bks, RightsBook):
            bks = [bks]
        elif isinstance(bks, dict):
            bks = list(bks.values())
        for bk in bks:
            seq.append(bk)
            tags += [g] * len(list(zip(bk.awards, bk.requests)))
    nom = nominate_and_test(seq, pt, fgs, **kw)
    ent = nom.entitlements
    ent["group"] = pd.Series(tags, index=ent.index, dtype=object)
    t = ent["nominated"].to_numpy(dtype=float)
    x = ent["entitled"].to_numpy(dtype=float)
    tab = _group_table(ent, t, x, "group").set_index("group")
    empty = [g for g in groups if g not in tab.index]
    for g in empty:
        tab.loc[g] = {"nominated": 0.0, "entitled": 0.0, "haircut_mw": 0.0,
                      "kept_frac": 1.0, "haircut_frac": 0.0,
                      "haircut_share": 0.0}
    nom.by_group = tab.reindex(groups)
    return nom


#: The four settlement regimes that value entitled rights holder by holder,
#: under the column names ``wecc_rights.regimes_table`` prints.
HOLDER_REGIMES = ("3. rights-holder rights",
                  "4. registered-book contract",
                  "4o. registered-book option",
                  "5. hybrid")

#: The row that carries the rent no holder claims.
UNALLOCATED = "(unallocated)"


def holder_regimes(values: pd.DataFrame, R: float, *,
                   floor: float = 0.0) -> pd.DataFrame:
    """Regimes 3, 4, 4o and 5 from per-right values and the pool, with the
    (unallocated) rows.

    ``values`` has one row per entitled right, with its ``holder`` and its
    ``value`` ``v_k = -sum_m mu_m x~_k (SF_src(k),m - SF_snk(k),m)`` over the
    constraints in scope. ``R`` is the pool those constraints collect. With
    ``V_h = sum_(k in h) v_k`` the registered-book value of holder ``h`` and
    ``V+ = sum_h [V_h]+``, the regimes are:

    1. regime 3, rights-holder: ``A3_h = R [V_h]+ / V+``;
    2. regime 4, registered-book contract: ``A4_h = V_h``;
    3. regime 4o, registered-book option: ``A4o_h = sum_(k in h) [v_k]+``;
    4. regime 5, hybrid: ``A5_h = [V_h]+ min(1, R / V+)``.

    Regimes 3 and 5 each carry an ``(unallocated)`` row, ``R`` less what the
    holders are paid, where it differs from zero by more than 1e-9: regime 3
    leaves the whole pool there when no holder's value is positive, and
    regime 5 leaves what the hedges do not reach. Regimes 3 and 5 therefore
    sum to ``R`` with that row, and regimes 4 and 4o do not, because a
    registered book is paid at the shadow price on its own loading.

    The rows are the holders in sorted order, then the ``(unallocated)`` row
    where it is carried, and ``attrs["R"]`` is the pool. ``floor`` drops a
    holder whose largest entry is below that many dollars, and the
    ``(unallocated)`` row is never dropped.

    The accumulation runs right by right in the order of ``values`` and
    holder by holder in the order each holder first appears, which is the
    arithmetic ``wecc_rights.regimes_table`` performed before it called this
    function, so its output is unchanged to the last bit.
    """
    df = pd.DataFrame(values)
    holders = [str(h) for h in df["holder"]] if len(df) else []
    vals = [float(v) for v in df["value"]] if len(df) else []
    r4: dict[str, float] = {}
    r4o: dict[str, float] = {}
    for h, v in zip(holders, vals):
        r4[h] = r4.get(h, 0.0) + v
        r4o[h] = r4o.get(h, 0.0) + max(0.0, v)
    R = float(R)
    pos = {a: v for a, v in r4.items() if v > 0}
    sv = sum(pos.values())
    r3 = {a: R * v / sv for a, v in pos.items()} if sv > 0 else {}
    scale = min(1.0, R / sv) if sv > 1e-9 else 1.0
    r5 = {a: v * scale for a, v in pos.items()}
    unalloc_r3 = R - sum(r3.values())
    unalloc_r5 = R - sum(r5.values())

    cols = list(HOLDER_REGIMES)
    tab = pd.DataFrame({
        cols[0]: pd.Series(r3, dtype=float),
        cols[1]: pd.Series(r4, dtype=float),
        cols[2]: pd.Series(r4o, dtype=float),
        cols[3]: pd.Series(r5, dtype=float)},
        index=sorted(r4), columns=cols, dtype=float).fillna(0.0)
    if floor > 0 and len(tab):
        mag = tab.abs().max(axis=1)
        tab = tab.loc[[h for h in tab.index if mag[h] >= floor]]
    if abs(unalloc_r3) > 1e-9 or abs(unalloc_r5) > 1e-9:
        tab.loc[UNALLOCATED] = {cols[0]: unalloc_r3, cols[1]: 0.0,
                                cols[2]: 0.0, cols[3]: unalloc_r5}
    tab.attrs["R"] = R
    return tab


# ──────────────────────────────────────────────────────────────────────────
# Footprint-aware ATC — each BA rates a path through only its own lines
# ──────────────────────────────────────────────────────────────────────────
def ba_monitored_lines(pt: PTDFData, fp: Footprints, name: str) -> list[Line]:
    """A balancing authority's monitored set: its own internal lines plus the ties.

    Uses ``footprints.Footprints.line_kind``: a line internal to ``name`` or a
    tie is monitored by ``name``; a line internal to *another* footprint is not
    (the loop-flow blind spot). Ties are shared, so both BAs rate them.
    """
    mon = []
    for l in pt.lines:
        kind, owner = fp.line_kind(pt, l)
        if (kind == "internal" and owner == name) or kind == fp.tie_label:
            mon.append(l)
    return mon


def ba_atc(pt: PTDFData, fp: Footprints, name: str, source: Bus, sink: Bus,
           etc: float = 0.0) -> tuple[float, Line | None]:
    """ATC a balancing authority posts for ``source->sink`` seeing only its own lines.

    Returns ``(atc_mw, binding_line)`` — TTC through ``ba_monitored_lines(name)``,
    less ``etc``. A neighbor's internal line that this path loads is invisible
    here, which is exactly how combined ATC oversubscribes the network.
    """
    mon = ba_monitored_lines(pt, fp, name)
    ttc_mw, binding = ttc(pt, source, sink, monitored=mon)
    return atc(ttc_mw, etc=etc), binding


# ──────────────────────────────────────────────────────────────────────────
# The CRR / FTR auction — award the simultaneously feasible set by bid value
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Bid:
    """A transmission customer's bid into the CRR/FTR auction.

    A request for up to ``mw`` MW of a point-to-point right ``source -> sink`` at a
    hedge-value ``premium`` ($/MW) — what the customer will pay for the congestion
    hedge on that path. ``label`` names the customer for the ledger.
    """

    source: str
    sink: str
    mw: float
    premium: float
    label: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.sink = str(self.sink)
        self.mw = float(self.mw)
        self.premium = float(self.premium)
        self.group = str(self.group)


@dataclass
class CRRResult:
    """Outcome of :func:`crr_auction`.

    ``awards`` — per-bid table (source, sink, label, requested, premium, awarded MW,
    ``charge`` = the flowgate price it pays ``Sum_l a_l mu_l`` $/MW, and ``status``).
    ``mu`` — ``{line: signed shadow price}`` on the binding flowgates (the scarcity
    prices; slack lines omitted). ``rent`` — auction revenue ``Sum_i x_i * charge_i``,
    which equals the congestion rent ``Sum_l |mu_l F_l|`` of the awarded set. ``value``
    — the objective ``Sum_i premium_i * x_i`` (total hedge value awarded).
    """

    awards: pd.DataFrame
    mu: LineValues
    rent: float
    value: float


def crr_auction(pt: PTDFData, bids: Iterable[Bid] | Iterable[BidRow],
                monitored: Monitored = "all",
                obligations: Mapping[str, float] | None = None,
                tol: float = 1e-7) -> CRRResult:
    """Hogan's CRR/FTR auction: award point-to-point rights to maximize bid value
    subject to the simultaneous-feasibility test.

    Solves the linear program (Hogan 1992's *"maximizing the value of the selected
    bids"* objective over the flowgate SFT ``H t <= b`` of the flowgate paper)::

        max_x   Sum_i premium_i * x_i
        s.t.   | Sum_i a_l(s_i, k_i) x_i | <= F_bar_l    for every monitored line l
                Sum_{i in g} x_i <= cap_g                 for every obligation group g
                0 <= x_i <= requested_i

    where ``a_l`` is the :func:`path_shift_factors` of each bid and ``x_i`` its awarded
    MW. The counterflow-crediting (netted) SFT is used — the financial-CRR / obligation
    standard, the same test as :func:`simultaneous_feasibility`. The dual ``mu_l`` on a
    binding line is its scarcity price; by complementary slackness a bid clears
    (``x_i > 0``) only where its premium covers the flowgate charge it consumes,
    ``premium_i >= Sum_l a_l(s_i, k_i) mu_l`` — the endogenous, price-based award that
    replaces an administrative sort.

    ``obligations`` (optional) ``{group: cap_mw}`` caps the TOTAL award across all bids
    sharing that ``Bid.group`` — a transmission customer's single delivery obligation it
    may source over several candidate paths (POR/POD combinations). With it the auction
    *derives* the sourcing split (e.g. cheap path up to the flowgate limit, the rest via a
    costlier one) instead of being handed it. Returns a :class:`CRRResult`.

    ``monitored`` may be a ``seams_engine.FlowgateSet``, the case's
    constraint set, as in every other rating of this module (2026-10-21,
    plan item B2). The SFT row of flowgate ``k`` is then
    ``Sum_m S[k,m] a_m(s_i, k_i)``, the forward block is capped at ``fwd_k``
    and the reverse block at ``rev_k``, and ``mu`` is keyed by flowgate id.
    The rent then equals ``Sum_k |mu_k f_k|`` of the awarded set. With line
    names or ``"all"`` the auction is the per-line program, as before.
    """
    from scipy.optimize import linprog

    bids = [b if isinstance(b, Bid) else Bid(*b) for b in bids]
    fgs = _fgset(monitored)
    n = len(bids)
    if fgs is None:
        idx = _monitored_idx(pt, monitored)
        ids = [pt.lines[l] for l in idx]
        # A[l, i] = path shift factor of bid i on monitored line l (the SFT coefficient).
        A = np.zeros((len(idx), n))
        for i, b in enumerate(bids):
            A[:, i] = path_shift_factors(pt, b.source, b.sink)[idx]
        fbar = np.array([float(pt.s_nom[l]) for l in idx])
        fbar_rev = fbar
    else:
        ids = list(fgs.ids)
        # A[k, i] = flowgate shift factor of bid i: Sum_m S[k,m] a_m(s_i, k_i).
        A = np.zeros((fgs.n_fg, n))
        for i, b in enumerate(bids):
            A[:, i] = fgs.S @ path_shift_factors(pt, b.source, b.sink)
        fbar = np.asarray(fgs.fwd, dtype=float)
        fbar_rev = np.asarray(fgs.rev, dtype=float)
    # |A x| <= fbar  ->  two one-sided blocks (+A and -A); netting credits counterflow.
    blocks = [A, -A]
    rhs = [fbar, fbar_rev]
    # Obligation-group caps appended AFTER the line rows (so line-dual indexing is unchanged).
    for g, cap in (obligations or {}).items():
        blocks.append(np.array([[1.0 if b.group == g else 0.0 for b in bids]]))
        rhs.append(np.array([float(cap)]))
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    c = np.array([-b.premium for b in bids])                 # maximize value = minimize -value
    bounds = [(0.0, b.mw) for b in bids]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CRR auction LP failed: {res.message}")

    x = np.asarray(res.x)
    # HiGHS marginals for (A_ub x <= b_ub) are <= 0; the scarcity price is their negative.
    m = -np.asarray(res.ineqlin.marginals)
    nl = len(ids)
    mu_line = m[:nl] - m[nl:2 * nl]                          # signed dual per monitored constraint (constraint rows only)
    charge = A.T @ mu_line                                   # flowgate charge Sum_l a_l mu_l per bid
    mu = {ids[j]: round(float(mu_line[j]), 4)
          for j in range(nl) if abs(mu_line[j]) > tol}

    rows = []
    for i, b in enumerate(bids):
        xi = float(x[i])
        status = ("full" if xi > b.mw - 1e-6 else "partial" if xi > 1e-6 else "none")
        rows.append({"label": b.label, "from": b.source, "to": b.sink,
                     "requested (MW)": round(b.mw, 1), "premium ($/MW)": round(b.premium, 2),
                     "awarded (MW)": round(xi, 1), "charge ($/MW)": round(float(charge[i]), 2),
                     "status": status})
    awards = pd.DataFrame(rows)
    return CRRResult(awards=awards, mu=mu,
                     rent=round(float(x @ charge), 1), value=round(float(-res.fun), 1))


def awarded_rights(result: CRRResult, tier_key: str = "label",
                   min_mw: float = 1e-6) -> list[dict[str, str | float]]:
    """The awarded set as a ``rights`` list (``{source, sink, mw, tier}``) for the
    figure helpers (``rights_figure`` / ``draw_rights_arcs``), dropping zero awards.
    """
    out = []
    for _, r in result.awards.iterrows():
        if r["awarded (MW)"] > min_mw:
            out.append({"source": r["from"], "sink": r["to"],
                        "mw": float(r["awarded (MW)"]), "tier": r[tier_key]})
    return out


# ──────────────────────────────────────────────────────────────────────────
# Smoke test — reproduces the three results the notebook is built on
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import wscc9_model as wm
    import footprints as fpmod
    from seams_engine import compute_ptdf

    pt = wm.shift_factors()

    print("(2) standalone path TTCs:")
    for s, k in [("3", "7"), ("3", "5"), ("3", "9"), ("2", "9"), ("1", "9")]:
        mw, binding = ttc(pt, s, k)
        print(f"    {s}->{k}: TTC={mw:6.1f} MW  binds {binding}")

    print("\n(3) single footprint, naive 'serve all load from gen-3':")
    naive = [Award("3", "9", 125), Award("3", "7", 100), Award("3", "5", 90)]
    ok, df = simultaneous_feasibility(pt, naive)
    print(df[df["loading_%"] >= 90].to_string())
    print(f"    SFT feasible? {ok}  (expect False; line_3 105%, line_4 114%)")

    print("\n    after sequential booking (ETC decrement), same requests in order:")
    booked = book_sequentially(pt, [("3", "9", 125), ("3", "7", 100), ("3", "5", 90)])
    ok2, _ = simultaneous_feasibility(pt, booked)
    print("    granted:", [(a.source + "->" + a.sink, round(a.mw, 1)) for a in booked],
          "| SFT feasible?", ok2)

    print("\n(4) two BAs each sell all their ATC:")
    BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    fp = fpmod.make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"},
                    monitored=None, tie_label="tie")
    a1, b1 = ba_atc(pt, fp, "BA-1", "3", "7")
    a2, b2 = ba_atc(pt, fp, "BA-2", "1", "9")
    print(f"    BA-1 ATC 3->7 (own lines) = {a1:.0f}  binds {b1}")
    print(f"    BA-2 ATC 1->9 (own lines) = {a2:.0f}  binds {b2}")
    combined = [Award("3", "7", a1), Award("1", "9", a2)]
    ok3, df3 = simultaneous_feasibility(pt, combined)
    print(df3.loc[["line_3", "line_4"]].to_string())
    print(f"    network-wide SFT feasible? {ok3}  (expect False; line_4 ~121%)")

    print("\n(5) CRR/FTR auction — award oversubscribed requests by hedge value:")
    ptc = compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")
    r = crr_auction(ptc, [
        Bid("1", "9", 125, 8.0, "C1 -> bus 9"),
        Bid("1", "5", 90, 3.0, "C2 -> bus 5"),
        Bid("3", "7", 100, 12.0, "C3 -> bus 7 (cheap)")])
    print(r.awards.to_string(index=False))
    print(f"    flowgate prices mu = {r.mu}")
    print(f"    auction revenue = ${r.rent:,.0f}  (= congestion rent of the awarded set)")
# ---------------------------------------------------------------------------
# Contingency (N-1 / LODF) helpers, shared by 301 and 302.
# ---------------------------------------------------------------------------

def post_contingency_ptdf(net: pypsa.Network, drop_line: Line,
                          slack_bus: Bus = "1") -> PTDFData:
    """PTDF of ``net`` with ``drop_line`` removed -- the post-contingency
    operator (same injections, no redispatch)."""
    import seams_engine
    n = net.copy()
    n.remove("Line", drop_line)
    return seams_engine.compute_ptdf(n, slack_bus=slack_bus)


def post_contingency_flow(pt_ctg: PTDFData, res: EngineResult, line: Line) -> float:
    """The flow ``res``'s injections put on ``line`` under the
    post-contingency PTDF ``pt_ctg``."""
    return float((pt_ctg.ptdf @ res.injection)[pt_ctg.line_idx[line]])


def lodf_derate(pt_ctg: PTDFData, res: EngineResult, survivor: Line,
                rating: float) -> tuple[float, float]:
    """The outage distribution factor onto ``survivor`` and the N-1 de-rate:
    ``lodf`` is the share of the lost circuit's flow the survivor picks up,
    and ``derate`` is the base limit at which the post-contingency flow lands
    exactly at ``rating``. Returns ``(lodf, derate)``."""
    lodf = post_contingency_flow(pt_ctg, res, survivor) / res.flow_own[survivor] - 1.0
    return lodf, rating / (1.0 + lodf)


# ──────────────────────────────────────────────────────────────────────────
# Western Loading Relief (WLR): the relief assignment on one constraint
# (added 2026-10-13 for notebook 304a)
# ──────────────────────────────────────────────────────────────────────────
#: The NERC transmission priority codes mapped to the bucket level the WLR walk
#: takes lowest first. 7-F (firm point-to-point) and 7-FN (firm network) share
#: the firm bucket. Mirrors ``etag.NERC_PRIORITIES``; repeated here so this
#: module keeps no import on the tag module.
WLR_PRIORITY_LEVELS = {"0-NX": 0, "1-NS": 1, "2-NH": 2, "3-ND": 3, "4-NW": 4,
                       "5-NM": 5, "6-NN": 6, "7-F": 7, "7-FN": 7}


def wlr_gtl_component(pt: PTDFData, line: Line, gen_by_bus: BusValues,
                      load_by_bus: BusValues, *, market: str,
                      priority: str = "7-F",
                      tag: str | None = None) -> dict[str, str | float]:
    """One balancing authority's (or market area's) untagged generation-to-load
    as a WLR component on ``line``.

    The impact is the generation's shift factors measured against the area's
    load-weighted shift factor,

        I = sum_n g_n (SF_{n,m} - LSF_m),   LSF_m = sum_n d_n SF_{n,m} / sum_n d_n,

    the white paper's generator-to-load distribution factor applied to the
    untagged dispatch (``eccetf_futurestate2024`` p. 7 defines the GLDF as the
    generation shift factor less the load shift factor, p. 9 the GTL layer, and
    p. 12 makes untagged GTL firm by default and aggregates it per market area).
    The residual formula is the record's natural implementation of those
    definitions and is not quoted from a source (sweep_sources_304a Part 4,
    row 4: "GTL residual formula Inferred"). ``mw`` is the total untagged
    generation and ``tdf`` the impact per megawatt, ``I / mw``, which is the
    ratio the 5 percent eligibility screen reads; the GTL denominator is not
    stated publicly (row 5), so that ratio is a Lab reading.

    The impact equals the area's flow contribution only for a balanced book,
    generation equal to load. The white paper decrements generation for load
    already served by tags (row 4); the caller passes the untagged remainder
    of both, and this function does not check the balance.

    Returns a component dict for :func:`wlr_assign` with ``kind='GTL'``.
    """
    li = pt.line_idx[str(line)]
    sf = {b: float(pt.ptdf[li, pt.bus_idx[str(b)]]) for b in pt.buses}
    d_tot = float(sum(load_by_bus.values()))
    if d_tot <= 0:
        raise ValueError("a GTL component needs load to measure against")
    lsf = sum(float(d) * sf[str(b)] for b, d in load_by_bus.items()) / d_tot
    g_tot = float(sum(gen_by_bus.values()))
    impact = sum(float(g) * (sf[str(b)] - lsf) for b, g in gen_by_bus.items())
    return {"tag": tag or f"GTL {market}", "market": market, "kind": "GTL",
            "source": "gen", "sink": "load", "priority": priority,
            "mw": g_tot, "tdf": (impact / g_tot if g_tot > 0 else 0.0),
            "lsf": lsf}


def wlr_assign(tags: pd.DataFrame | Iterable[Mapping[str, Any]], pt: PTDFData,
               line: Line, rating: float | None = None,
               flow: float | None = None, *, threshold: float = 0.05,
               priority_order: Mapping[str, int] | None = None,
               tol: float = 1e-9) -> pd.DataFrame:
    """Assign the relief a constraint needs to the tags that load it, by the
    Western Loading Relief (WLR, NAESB WEQ-026) rule as the public record
    describes it. Returns the ledger, one row per component.

    ``tags`` is an iterable of dicts (or a DataFrame) with ``tag``, ``market``,
    ``source``, ``sink``, ``priority`` (a NERC code, see
    :data:`WLR_PRIORITY_LEVELS`) and ``mw``. A component may carry its own
    ``tdf`` (a GTL component from :func:`wlr_gtl_component` does), and
    ``kind`` (``'tag'`` by default, or ``'GTL'``). ``flow`` is the superposed
    flow on ``line`` and defaults to the sum of the components' impacts, which
    is the flow when every megawatt on the network is tagged; ``rating``
    defaults to the line's rating in ``pt``.

    The rule, with its status in the public record (sweep_sources_304a Part 4):

    1. **Distribution factor and impact.** ``TDF_k = SF_{src,m} - SF_{snk,m}``
       and ``I_k = MW_k * TDF_k`` (``eccetf_futurestate2024`` p. 7; quoted).
    2. **Relief requirement.** ``R = s F_m - rating`` with ``s`` the sign of the
       flow (the loading direction), and no relief when ``R <= 0``. The white
       paper's example is 600 MW of post-contingent flow on a 500 MW SOL needing
       100 MW of relief (p. 13); the formula as a rule is a reading of that
       example (row 3: "formula Inferred").
    3. **Eligibility.** A component is eligible iff ``s TDF_k >= threshold``
       (5 percent; ``naesb_weq_ecrec2025`` "5% or greater impact", quoted; the
       positive sign is member-only, ``naesb_wicmwg_minutes2025`` 17 Oct).
    4. **Counterflow.** A component with ``s I_k < 0`` is neither curtailed nor
       credited: it stays in the flow and never reduces ``R`` below the flow
       less the rating (member-only, the 17 Oct minutes: "counterflows are not
       included in relief obligation calculations").
    5. **Bucket walk.** Eligible components are taken by priority level, lowest
       first; every component in a bucket below the last one reached is cut to
       zero before the next bucket is touched (white paper p. 12 and p. 14;
       quoted as proposal; ratified wording member-only).
    6. **Pro rata by impact in the last bucket.** The residual ``R_res`` is
       split ``relief_k = R_res * I_k / sum_j I_j`` over the last bucket, and a
       tag's cut is ``cut_k = relief_k / TDF_k`` (white paper pp. 14-15, the
       worked example; the exact formula is member-only). Relief proportional
       to impact is the same as an equal fractional cut of every tag in the
       bucket, ``cut_k / MW_k = R_res / sum_j I_j``. This is the reading
       :func:`_reduce_pro_rata` already implements within one curtailment tier
       (a common factor on the tier's megawatts), and the opposite of a cut
       proportional to ``MW * TDF``, which would make relief proportional to
       ``MW * TDF**2``. A GTL component's relief is its obligation in
       megawatts of flow, discharged by redispatch, so it has no cut.

    A market's relief obligation is the sum of its components' ``relief``.
    Where the eligible impact is smaller than ``R`` the shortfall is left
    unassigned and reported in ``ledger.attrs['unassigned']``; no constraint
    is relaxed to hide it. Everything the walk does not cover (initiation, the
    hourly clock, the priority-assignment method for multi-segment tags, the BA
    veto) is outside this function. :func:`_reduce_pro_rata` is left unchanged,
    because it works on rights books, takes the excess from the books
    themselves rather than from a measured flow, and has no GTL component.

    The ledger columns are ``tag, market, kind, source, sink, priority,
    bucket, MW, TDF, impact, eligible, relief, cut, MW kept``; ``attrs`` carries
    ``R`` (the requirement), ``flow``, ``rating``, ``direction`` (``s``),
    ``last_bucket``, ``assigned`` and ``unassigned``.
    """
    levels = WLR_PRIORITY_LEVELS if priority_order is None else dict(priority_order)
    recs = (tags.to_dict("records") if isinstance(tags, pd.DataFrame)
            else [dict(t) for t in tags])
    li = pt.line_idx[str(line)]
    rows = []
    for t in recs:
        kind = t.get("kind", "tag")
        if t.get("tdf") is not None:
            tdf = float(t["tdf"])
        else:
            tdf = float(pt.ptdf[li, pt.bus_idx[str(t["source"])]]
                        - pt.ptdf[li, pt.bus_idx[str(t["sink"])]])
        pr = str(t["priority"])
        if pr not in levels:
            raise ValueError(f"unknown priority {pr!r} for {t.get('tag')}")
        mw = float(t["mw"])
        rows.append({"tag": str(t.get("tag", f"{t['source']}->{t['sink']}")),
                     "market": t.get("market"), "kind": kind,
                     "source": str(t["source"]), "sink": str(t["sink"]),
                     "priority": pr, "bucket": int(levels[pr]),
                     "MW": mw, "TDF": tdf, "impact": mw * tdf})
    led = pd.DataFrame(rows)
    rate = float(pt.s_nom[li]) if rating is None else float(rating)
    f = float(led["impact"].sum()) if flow is None else float(flow)
    s = 1.0 if f >= 0 else -1.0
    req = max(0.0, s * f - rate)
    led["eligible"] = (s * led["TDF"] >= threshold) & (led["MW"] > tol)
    led["relief"] = 0.0
    residual, last = req, None
    if req > tol:
        for b in sorted(led.loc[led["eligible"], "bucket"].unique()):
            if residual <= tol:
                break
            grp = led.index[led["eligible"] & (led["bucket"] == b)]
            avail = float((s * led.loc[grp, "impact"]).sum())
            if avail <= tol:
                continue
            last = int(b)
            if avail <= residual + tol:
                led.loc[grp, "relief"] = s * led.loc[grp, "impact"]
                residual -= avail
            else:
                led.loc[grp, "relief"] = residual * (s * led.loc[grp, "impact"]) / avail
                residual = 0.0
    is_tag = (led["kind"] != "GTL").to_numpy()
    rel = led["relief"].to_numpy(dtype=float)
    stdf = s * led["TDF"].to_numpy(dtype=float)
    cut = np.zeros(len(led))
    pos = is_tag & (rel > 0)
    cut[pos] = rel[pos] / stdf[pos]
    cut[~is_tag] = np.nan
    led["cut"] = cut
    led["MW kept"] = np.where(is_tag, led["MW"].to_numpy(dtype=float) - cut, np.nan)
    led.attrs.update(R=req, flow=f, rating=rate, direction=s, last_bucket=last,
                     assigned=float(led["relief"].sum()),
                     unassigned=(max(0.0, residual) if req > tol else 0.0))
    return led[["tag", "market", "kind", "source", "sink", "priority", "bucket",
                "MW", "TDF", "impact", "eligible", "relief", "cut", "MW kept"]]
