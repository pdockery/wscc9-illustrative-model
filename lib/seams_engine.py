# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Multi-engine DC-OPF on a shared network — teaching harness for the seams paper.

This module implements, in the most transparent way possible, the shift-factor
algebra of *Market Seams in the Western Interconnection* (Dockery, 2026) on the
WSCC/IEEE 9-bus test system. It is the computational backbone for
``seams_examples.ipynb``.

The objects map one-to-one onto the paper's notation:

    PTDF[l, n]      SF_{n,m,t}      shift factor of bus n on constraint m   (§2.1)
    MarketEngine    engine M        a resource/load/constraint partition    (§2.1)
    EngineResult.lmp        λ^M_{n,t}   engine M's LMP at every bus           (9),(14),(17)
    EngineResult.energy_price   λ^M_t / λ^M_{j,t}   energy component
    EngineResult.line_dual      μ^M_{m,t}   congestion shadow price           (5)
    EngineResult.flow_own       F^M_{m,t}   engine M's *own* flow component   (6)
    physical_flows()        F^phys_{m,t} = Σ_M F^M_{m,t} + F^non   superposition (30)
    actual_injection()      the same sum from each engine's own generation and
                            served load alone, which ``physical_flows(...,
                            exo_physical=False)`` and ``actual_flows()`` read,
                            for a schedule whose two legs are realized in two books

Each engine optimizes **only its own resources and loads**, enforces limits
**only on its own activated constraint set**, and computes flow **only against
its own injections** (eq. 6). Cross-engine injections enter as fixed,
price-taking schedules (``exo`` argument). The physical flow on any line is the
superposition of every engine's own component (eq. 30) — which is what produces
the overloads and price gaps the paper formalizes.

The DC-OPF is solved with ``scipy.optimize.linprog(method="highs")``; the LMP
decomposition is read directly off the HiGHS dual variables, so the nodal price
is literally ``energy_price + Σ_l PTDF[l, n] · μ_l`` (eq. 9).

The Case contract (2026-10-18). One copy of the shared libraries serves every
case, and the surface a case exposes to them is stated once in this module.
``CaseProtocol`` is the runtime-checkable Protocol every case satisfies (the
9-bus ``wscc9_model.WSCC9Case`` and the WECC ``wecc_case.WECCCase``);
``DataBackedCase`` is the extension a case pinned to a data bundle adds;
``check_case`` verifies an instance against both and returns the violations;
``CaseNetwork.constraint_set`` names the kind of constraint a built network
rates, ``"lines"`` or ``"flowgates"``, by the rule the solvers apply. The
ambient case is held by ``set_case`` and ``using``, and ``active()`` resolves
every shared-library default (fleet, loads, slack bus, layout) through a
conforming case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import SimpleNamespace

    import pypsa
    from footprints import Footprints
    from lab_types import (
        Area,
        AreaBuses,
        AreaValues,
        Bus,
        BusValues,
        Fleet,
        FloatArray,
        Interface,
        Interfaces,
        Line,
        LineValues,
        Loads,
        Resource,
    )
    from risk import LogitNormal, Normal


# ──────────────────────────────────────────────────────────────────────────
# Shift factors (PTDF)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PTDFData:
    """Shared-network shift factors and bookkeeping (the paper's N, M, SF)."""

    ptdf: FloatArray              # (n_line, n_bus) — SF_{n,m}
    buses: list[Bus]              # ordered bus names (N)
    lines: list[Line]             # ordered line names (M)
    bus_idx: dict[Bus, int]
    line_idx: dict[Line, int]
    line_buses: list[tuple[Bus, Bus]]   # (bus0, bus1) per line
    s_nom: FloatArray             # (n_line,) — operating limit F̄_m
    susceptance: FloatArray       # (n_line,) — b_m = 1/x_m ("slipperiness")
    slack_bus: Bus

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_line(self) -> int:
        return len(self.lines)

    def line_flow(self, injection: BusValues | FloatArray) -> FloatArray:
        """F_m = Σ_n SF_{n,m} · p_n for an injection vector (dict or array)."""
        p = self._as_vector(injection)
        return self.ptdf @ p

    def _as_vector(self, injection: BusValues | FloatArray) -> FloatArray:
        if isinstance(injection, np.ndarray):
            return injection
        p = np.zeros(self.n_bus)
        for bus, mw in injection.items():
            p[self.bus_idx[str(bus)]] += mw
        return p


def _bus_sort_key(b: object) -> tuple[int, int, str]:
    """Sort key for bus names: numeric names sort NUMERICALLY (all-digit names
    like the 9-bus case's "1".."9" keep exactly the permutation the old
    ``int(x)`` sort produced), non-numeric names (e.g. WECC's "st_1694_230")
    sort lexicographically after every numeric one."""
    s = str(b)
    return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)


def compute_ptdf(network: pypsa.Network | SimpleNamespace, slack_bus: Bus | None = None,
                 method: str = "auto", bus_order: list[Bus] | None = None) -> PTDFData:
    """Build the DC shift-factor matrix from a PyPSA network.

    Uses line reactances (``x_pu_eff`` / ``x``) and the standard reduced-
    susceptance inversion with one slack bus. The PTDF distribution depends only
    on *relative* susceptances, so WITHIN one voltage level the per-unit/ohm
    convention is immaterial. A mixed-voltage network in ohms must supply a
    per-line ``v_nom`` column (kV): the DC flow weight is then the per-unit
    susceptance ``V^2/x`` (2026-08-25; raw ``1/x`` over-weighted a 230 kV line
    against a 500 kV line by ``(500/230)^2``). Lines without ``v_nom`` fall
    back to ``1/x``, so a per-unit or single-voltage case (the 9-bus teaching
    case) is byte-identical to before.

    Parameters
    ----------
    method : "auto" | "dense" | "sparse"
        ``dense`` is the original full matrix inverse; ``sparse`` factorizes the
        reduced susceptance matrix with SuperLU and back-solves one RHS per bus
        (the WECC map-mirror's 1883-node case factorizes in milliseconds where
        the dense inverse would be avoidable work). ``auto`` = dense below 200
        buses — so the teaching case NEVER enters the new code path — else sparse.
    bus_order : list, optional
        Explicit bus ordering (a case may pin its own); default sorted by
        :func:`_bus_sort_key`.

    ``slack_bus=None`` resolves to the FIRST bus in sort order — on the 9-bus
    case that is bus "1", exactly the old hardcoded default; a case with named
    buses should always pass its own slack explicitly (``case.slack_bus``).
    """
    if bus_order is not None:
        buses = [str(b) for b in bus_order]
        assert set(buses) == set(map(str, network.buses.index)), \
            "bus_order must be a permutation of the network's buses"
    else:
        buses = sorted(network.buses.index.tolist(), key=_bus_sort_key)
    if slack_bus is None:
        slack_bus = buses[0]           # 9-bus: "1", the old hardcoded default
    lines = network.lines
    n_bus, n_line = len(buses), len(lines)
    bus_idx = {b: i for i, b in enumerate(buses)}

    b_line = np.zeros(n_line)
    C = np.zeros((n_line, n_bus))
    line_names, line_buses, s_nom = [], [], np.zeros(n_line)
    has_v = "v_nom" in getattr(lines, "columns", ())
    v_ref = None
    if has_v:
        _vv = np.asarray([v for v in lines["v_nom"] if v is not None and v == v],
                         dtype=float)
        if len(_vv):
            v_ref = float(np.median(_vv))
    for li, (name, row) in enumerate(lines.iterrows()):
        x = row.get("x_pu_eff", row.get("x_pu", row.get("x", 0.01)))
        if abs(x) < 1e-10:
            x = 0.01
        v = row.get("v_nom") if has_v else None
        if v_ref and v is not None and v == v:      # v == v filters NaN
            # mixed-voltage ohm network: per-unit susceptance V^2/x,
            # normalized by the network's median voltage so a
            # uniform-voltage case is bit-identical to plain 1/x
            b_line[li] = (float(v) / v_ref) ** 2 / x
        else:
            b_line[li] = 1.0 / x
        b0, b1 = str(row["bus0"]), str(row["bus1"])
        C[li, bus_idx[b0]] = 1.0
        C[li, bus_idx[b1]] = -1.0
        line_names.append(str(name))
        line_buses.append((b0, b1))
        s_nom[li] = row["s_nom"]

    s_i = bus_idx[slack_bus]
    keep = [i for i in range(n_bus) if i != s_i]
    resolved = method if method != "auto" else ("dense" if n_bus < 200 else "sparse")
    if resolved == "dense":
        # The original construction, untouched — the teaching case's numbers.
        B_bus = C.T @ np.diag(b_line) @ C
        B_inv = np.linalg.inv(B_bus[np.ix_(keep, keep)])
        B_inv_full = np.zeros((n_bus, n_bus))
        for ii, i in enumerate(keep):
            for jj, j in enumerate(keep):
                B_inv_full[i, j] = B_inv[ii, jj]
        ptdf = np.diag(b_line) @ C @ B_inv_full
    else:
        # Sparse LU on the reduced susceptance matrix, one back-solve per line.
        # PTDF row m = b_m · [B_red⁻¹ (e_{bus0(m)} − e_{bus1(m)})]ᵀ on the kept
        # buses (B is symmetric). At the WECC map-mirror's 1883 nodes this
        # factorizes in milliseconds and back-solves ~0.2 ms/line.
        import scipy.sparse as _sp
        from scipy.sparse.linalg import factorized as _factorized

        Cs = _sp.csr_matrix(C)
        B_red = (Cs.T @ _sp.diags(b_line) @ Cs).tocsc()[keep][:, [
            *keep]].tocsc()
        solve = _factorized(B_red)
        kpos = np.full(n_bus, -1)
        kpos[keep] = np.arange(len(keep))
        ptdf = np.zeros((n_line, n_bus))
        for li in range(n_line):
            rhs = np.zeros(len(keep))
            for b, sgn in ((line_buses[li][0], 1.0), (line_buses[li][1], -1.0)):
                j = bus_idx[b]
                if j != s_i:
                    rhs[kpos[j]] += sgn
            ptdf[li, keep] = b_line[li] * solve(rhs)

    return PTDFData(
        ptdf=ptdf,
        buses=buses,
        lines=line_names,
        bus_idx=bus_idx,
        line_idx={n: i for i, n in enumerate(line_names)},
        line_buses=line_buses,
        s_nom=s_nom,
        susceptance=b_line,
        slack_bus=slack_bus,
    )


def susceptance_widths(
    pt: PTDFData, wmin: float = 1.2, wmax: float = 7.0
) -> LineValues:
    """Map each line's susceptance b_m = 1/x_m to a drawing width ("slipperiness").

    Lower reactance ⇒ higher susceptance ⇒ a *wider* line: for a given angle
    difference, DC flow on a line is proportional to b_m, so a wider line is the
    one that "wants" more of the flow, all else equal. Returns ``{line: width}``
    scaled linearly across [wmin, wmax].
    """
    b = pt.susceptance
    lo, hi = float(b.min()), float(b.max())
    if hi - lo < 1e-9:
        return {pt.lines[i]: 0.5 * (wmin + wmax) for i in range(pt.n_line)}
    w = wmin + (b - lo) / (hi - lo) * (wmax - wmin)
    return {pt.lines[i]: float(w[i]) for i in range(pt.n_line)}


# ──────────────────────────────────────────────────────────────────────────
# Flowgates — signed linear combinations of line flows that carry limits
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class FlowgateSet:
    """The activated constraint set as FLOWGATES: ``f_k = Σ_m S[k,m]·F_m`` with
    ``−rev_k ≤ f_k ≤ fwd_k``.

    A monitored LINE is the special case ``S = e_mᵀ`` with ``fwd = rev = s_nom``
    (see :func:`identity_flowgates`) — so per-line limits and WECC rated paths
    are one object, and the LMP identity survives verbatim: with the implied
    per-line dual ``μ^impl = Sᵀ μ_fg``,

        Σ_k H[k,n] μ_k  =  Σ_m SF_{n,m} μ^impl_m ,      H = S · PTDF,

    so ``λ_n = λ̂_a + Σ_m SF_{n,m} μ_m`` holds on any case. A line inside a
    binding flowgate inherits ``sign_m · μ_k``; a line in no binding flowgate
    has ``μ^impl_m = 0``.

    An unrated direction carries a large sentinel (the upstream western-seams
    convention: a missing catalog reverse rating means UNRATED, never "equal to
    forward" — constraining it would invent phantom reverse congestion rent).
    """

    ids: list[Line]                    # flowgate names (== line names at S = I)
    S: FloatArray                      # (n_fg, n_line) signed membership
    fwd: FloatArray                    # (n_fg,) forward rating, MW
    rev: FloatArray                    # (n_fg,) reverse rating, MW (positive)
    members: dict[Line, list[tuple[Line, float]]] = field(
        default_factory=dict)   # {id: [(line, sign), ...]}
    _H: FloatArray | None = field(default=None, repr=False, compare=False)

    @property
    def n_fg(self) -> int:
        return len(self.ids)

    @property
    def fg_idx(self) -> dict[Line, int]:
        return {k: i for i, k in enumerate(self.ids)}

    def H(self, pt: PTDFData) -> FloatArray:
        """(n_fg, n_bus) bus→flowgate shift factors ``H = S @ PTDF``; cached."""
        if self._H is None:
            object.__setattr__(self, "_H", np.asarray(self.S @ pt.ptdf))
        return self._H  # type: ignore[return-value]  # set by object.__setattr__ above

    def subset(self, ids: Iterable[Line]) -> "FlowgateSet":
        """The flowgate set restricted to ``ids``, in the order given.

        Row slicing only. The column space is unchanged, so the subset's ``S``
        multiplies the same per-line flow vector the full set does, and a
        subset row's loading equals the full set's row for the same id. The
        cached ``H`` is dropped because it carries one row per flowgate.

        An authority's own constraint set is a subset of the shared catalog,
        so this is how a per-authority rating is taken without rebuilding the
        catalog.
        """
        idx = self.fg_idx
        sel = [idx[str(k)] for k in ids]
        keep = [self.ids[i] for i in sel]
        return FlowgateSet(
            ids=keep, S=np.asarray(self.S)[sel, :].copy(),
            fwd=np.asarray(self.fwd)[sel].copy(),
            rev=np.asarray(self.rev)[sel].copy(),
            members={k: self.members[k] for k in keep if k in self.members})

    def with_member(self, pid: Line, line: Line, sign: float,
                    pt: PTDFData) -> "FlowgateSet":
        """A copy of this set whose flowgate ``pid`` counts ``line`` at
        ``sign``, with every other row untouched.

        A rated path is a cut, so a new facility across the same cut is a
        MEMBER of that path and not a separate constraint. Raising the
        path's rating without adding the member would price the old
        facilities' flow at the new rating and leave the new facility's flow
        out of the cut.

        ``sign`` is the member's orientation in the path's forward
        direction, and the caller derives it from the network rather than
        reading it off the line's bus order. The derivation compares two
        flows under ONE injection: the flow on the new line, and the flow
        on the cut's existing members, which is the flow that carries the
        path's forward direction. The two agreeing gives ``+1`` and the two
        opposing gives ``-1``. The new line's own flow cannot supply the
        sign alone, because a passive DC network puts a non-negative flow
        on a line for an injection across that line's own ends.
        ``wecc_westtec.member_sign`` is that derivation.

        The bus-to-flowgate cache ``_H`` is dropped because the row changed.
        ``pt`` is the PTDF the set was built over. A set and a PTDF spanning
        different line counts raise, since the row's columns would then name
        different lines.
        """
        if self.S.shape[1] != pt.n_line:
            raise ValueError(
                f"the flowgate set spans {self.S.shape[1]} lines and the "
                f"PTDF spans {pt.n_line}; a member can only be added on the "
                f"network the set was built over")
        k = self.fg_idx[pid]
        j = pt.line_idx[str(line)]
        if any(str(m) == str(line) for m, _s in self.members.get(pid, [])):
            raise ValueError(f"{line} is already a member of {pid}; a cut "
                             f"counts a facility once")
        S = self.S.copy()
        S[k, j] += float(sign)
        members = {key: list(v) for key, v in self.members.items()}
        members.setdefault(pid, []).append((str(line), float(sign)))
        return FlowgateSet(ids=list(self.ids), S=S, fwd=self.fwd.copy(),
                           rev=self.rev.copy(), members=members)

    def with_flowgate(self, pid: Line,
                      members: list[tuple[Line, float]] | dict[Line, float],
                      fwd: float, rev: float, pt: PTDFData) -> "FlowgateSet":
        """A copy of this set with one new flowgate ``pid`` appended, whose
        row counts the given signed member lines and carries the given
        directional ratings, with every existing row of ``S``, ``fwd`` and
        ``rev`` unchanged.

        ``members`` is a list of ``(line, sign)`` pairs or a ``{line: sign}``
        map. ``sign`` is each member's orientation in the flowgate's forward
        direction, as in :meth:`with_member`. ``fwd`` is the forward rating
        and ``rev`` the reverse rating, both in MW and both non-negative, so
        the new row is enforced as ``-rev <= sum_m S[k,m] F_m <= fwd``.

        The new flowgate is a constraint of its own and not a member of an
        existing path, which is the distinction :meth:`with_member` draws. A
        facility rated as its own path, such as a line with a rating of its
        own in the WECC Path Rating Process, takes this method, and a
        facility across an existing cut takes :meth:`with_member`.

        The row is appended last, so every existing flowgate keeps its index
        and a loading computed on the old set equals the same row of the new
        one. The bus-to-flowgate cache ``_H`` is dropped because the set
        gained a row. A ``pid`` already in the set, a member line not in
        ``pt``, a line listed twice, an empty member list, a negative rating,
        and a set built over a different line count each raise.
        """
        if self.S.shape[1] != pt.n_line:
            raise ValueError(
                f"the flowgate set spans {self.S.shape[1]} lines and the "
                f"PTDF spans {pt.n_line}; a flowgate can only be added on the "
                f"network the set was built over")
        pid = str(pid)
        if pid in self.fg_idx:
            raise ValueError(f"{pid} is already a flowgate of this set")
        pairs = (list(members.items()) if isinstance(members, dict)
                 else [tuple(m) for m in members])  # type: ignore[misc]  # each m is a (line, sign) pair
        if not pairs:
            raise ValueError(f"flowgate {pid} needs at least one member line")
        fwd, rev = float(fwd), float(rev)
        if fwd < 0 or rev < 0:
            raise ValueError(f"flowgate {pid}: ratings are non-negative MW "
                             f"(fwd={fwd}, rev={rev})")
        row = np.zeros(pt.n_line)
        seen: list[tuple[str, float]] = []
        for line, sign in pairs:
            line = str(line)
            if line not in pt.line_idx:
                raise ValueError(f"{line} is not a line of the network")
            if any(m == line for m, _s in seen):
                raise ValueError(f"{line} is listed twice in {pid}; a "
                                 f"flowgate counts a facility once")
            row[pt.line_idx[line]] = float(sign)
            seen.append((line, float(sign)))
        S = np.vstack([np.asarray(self.S, dtype=float), row[None, :]])
        members_out = {key: list(v) for key, v in self.members.items()}
        members_out[pid] = seen
        return FlowgateSet(
            ids=list(self.ids) + [pid], S=S,
            fwd=np.append(np.asarray(self.fwd, dtype=float), fwd),
            rev=np.append(np.asarray(self.rev, dtype=float), rev),
            members=members_out)

    def implied_line_duals(self, fg_dual: LineValues) -> FloatArray:
        """``μ^impl = Sᵀ μ_fg`` — the per-line duals the flowgate duals imply."""
        mu = np.zeros(self.n_fg)
        idx = self.fg_idx
        for k, v in fg_dual.items():
            mu[idx[k]] = v
        return self.S.T @ mu


def identity_flowgates(pt: PTDFData, lines: str | Iterable[Line] = "all",
                       s_max_pu: float = 1.0) -> FlowgateSet:
    """Today's per-line constraint set as a ``FlowgateSet``: one flowgate per
    monitored line, ``S`` a selection matrix, ``fwd = rev = s_nom``."""
    idx = list(range(pt.n_line)) if lines == "all" else \
        [pt.line_idx[str(l)] for l in lines]
    S = np.zeros((len(idx), pt.n_line))
    S[np.arange(len(idx)), idx] = 1.0
    cap = pt.s_nom[idx] * float(s_max_pu)
    return FlowgateSet(
        ids=[pt.lines[i] for i in idx], S=S, fwd=cap.copy(), rev=cap.copy(),
        members={pt.lines[i]: [(pt.lines[i], 1.0)] for i in idx})


# ──────────────────────────────────────────────────────────────────────────
# Market engine definition (the paper's R^M, D^M, M^M_act)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MarketEngine:
    """One independent optimization over a subset of the shared network.

    Attributes
    ----------
    name : str
        Engine label (e.g. "A — WEM", "B — Markets+").
    gens : dict
        ``{gen_id: {"bus": str, "cost": $/MWh, "p_nom": MW}}`` — the engine's
        resource stack R^M.
    loads : dict
        ``{bus: MW}`` — the engine's served load D^M.
    activated_lines : "all" | list[str]
        The activated constraint set M^M_act. Lines outside this set have their
        limit F̄_m relaxed to +∞ inside this engine (silent congestion).
    """

    name: str
    gens: Fleet = field(default_factory=dict)
    loads: Loads = field(default_factory=dict)
    activated_lines: str | list[Line] = "all"      # "all" or list of line names

    @property
    def buses(self) -> set[Bus]:
        bs = {str(g["bus"]) for g in self.gens.values()}
        bs |= {str(b) for b in self.loads}
        return bs


@dataclass
class EngineResult:
    name: str
    dispatch: dict[Resource, float]         # gen_id -> MW
    gen_by_bus: BusValues
    load_by_bus: Loads
    exo_by_bus: BusValues
    injection: FloatArray                  # p^inj,M_n (own gens+loads+exo)
    lmp: BusValues                          # λ^M_n at every bus
    energy_price: float                     # λ^M_t  (or marginal energy comp.)
    line_dual: LineValues                   # μ^M_m (signed, nonzero if binding)
    flow_own: LineValues                    # F^M_m  (own injections only)
    total_cost: float
    status: str
    shed_by_bus: BusValues = field(default_factory=dict)  # u_n (unserved load; empty unless shed_price set)
    demand_cleared: dict[str, float] = field(default_factory=dict)  # price-sensitive demand bids: bid_id -> MW served
    # The cleared demand bids summed per bus, {bus: MW}: the bids of
    # ``demand_cleared`` placed at the bus each bid names (battery charging
    # under the completed storage formulation). Empty when no bid clears.
    demand_by_bus: BusValues = field(default_factory=dict)
    link_flow: dict[str, float] = field(default_factory=dict)    # DC-link transfers t_l (signed, bus0 -> bus1)
    # Flowgate layer (populated when the solver is given a FlowgateSet; at
    # S = I these coincide with line_dual / flow_own / (s_nom, s_nom)).
    fg_dual: LineValues = field(default_factory=dict)      # μ_k by flowgate id
    fg_flow: LineValues = field(default_factory=dict)      # f_k = Σ_m S[k,m] F_m
    fg_limit: dict[Line, tuple[float, float]] = field(
        default_factory=dict)     # (fwd_k, rev_k)
    # Membership of each flowgate, ``{id: [(line, sign), ...]}``, set only when
    # the solver was given a FlowgateSet (2026-10-21, plan item B2); empty for a
    # per-line clearing, so its emptiness tells the two constraint sets apart.
    fg_members: dict[Line, list[tuple[Line, float]]] = field(default_factory=dict)
    # EDAM-literal transfer form (solve_engine_transfers only; None otherwise)
    area_prices: AreaValues | None = None    # λ̂_a — each area's balance dual
    transfers: dict[Interface, float] | None = None            # {(a,b) sorted: net SCHEDULED t a→b, min-total-schedule tags}
    transfer_duals: dict[Interface, float] | None = None       # {(a,b) sorted: signed μ_T^{ab} (λ̂_b − λ̂_a when the limit binds)}
    transfer_limits: Interfaces | None = None      # {(a,b) sorted: released capacity T̄_ab (None = unlimited)}
    transfer_violations: dict[Interface, float] = field(
        default_factory=dict)  # {(a,b) sorted: signed VRL violation MW (interface_vrl only)}
    group_cap_dual: list[float] = field(default_factory=list)        # |dual| of each group_caps row, in the order passed
    # Intertie scheduling limits (solve_engine_transfers ``schedule_limits``
    # only; empty otherwise), keyed by limit id. The dual is nonnegative in
    # $/MWh and zero when the row is slack; the MW is the summed cleared
    # awards of the bids the limit names.
    schedule_limit_dual: dict[str, float] = field(default_factory=dict)
    schedule_limit_mw: dict[str, float] = field(default_factory=dict)


def _demand_by_bus(dem_bus: list[Bus], d_opt: FloatArray) -> BusValues:
    """The cleared demand bids summed per bus, ``{bus: MW}``.

    A bid is counted under the same threshold ``demand_cleared`` applies
    (more than 1e-9 MW), so the per-bus total equals the sum of the cleared
    bids that name the bus. Empty when no bid clears.
    """
    out: dict[str, float] = {}
    for b, q in zip(dem_bus, d_opt):
        if float(q) > 1e-9:
            out[str(b)] = out.get(str(b), 0.0) + float(q)
    return out


def own_injections(res: EngineResult | SimpleNamespace,
                   dc_links: dict[str, dict[str, Any]] | None = None,
                   decimals: int | None = 6) -> BusValues:
    """The engine's own net injection per bus, ``{bus: MW}``.

    The net injection at bus n is the generation the engine dispatches at n,
    less the nominal load at n, plus the load shed at n, less the cleared
    demand bids at n (``demand_by_bus``, which carries battery charging under
    the completed storage formulation). The nominal load less the shed is the
    served load, so the sum reads as generation less served load less cleared
    demand. Where ``dc_links`` is passed, each controllable HVDC link in
    ``res.link_flow`` adds its flow as a withdrawal at ``bus0`` and an
    injection at ``bus1``; a link absent from ``dc_links`` is not carried.

    The fixed injection ``exo`` the engine was handed is excluded, because it
    is not the engine's own dispatch. The result therefore equals
    ``res.injection`` less ``res.exo_by_bus`` whenever every link the engine
    carries is passed in ``dc_links``.

    This is the one object both halves build the forecast term
    ``rho * (g - d)`` from, and ``wecc_nonmarket.result_injections`` returns
    this function's output at ``decimals=6``. ``decimals`` rounds each entry,
    6 by default; ``None`` returns the unrounded
    values. A result without ``demand_by_bus`` (a stand-in clearing) is read
    as carrying no demand bids.
    """
    inj: dict[str, float] = {}
    for b, mw in (getattr(res, "gen_by_bus", None) or {}).items():
        inj[str(b)] = inj.get(str(b), 0.0) + float(mw)
    for b, mw in (getattr(res, "load_by_bus", None) or {}).items():
        inj[str(b)] = inj.get(str(b), 0.0) - float(mw)
    for b, mw in (getattr(res, "shed_by_bus", None) or {}).items():
        inj[str(b)] = inj.get(str(b), 0.0) + float(mw)
    for b, mw in (getattr(res, "demand_by_bus", None) or {}).items():
        inj[str(b)] = inj.get(str(b), 0.0) - float(mw)
    for lid, f in (getattr(res, "link_flow", None) or {}).items():
        spec = (dc_links or {}).get(lid)
        if spec is None:
            continue
        b0, b1 = str(spec["bus0"]), str(spec["bus1"])
        inj[b0] = inj.get(b0, 0.0) - float(f)
        inj[b1] = inj.get(b1, 0.0) + float(f)
    if decimals is None:
        return inj
    return {b: round(v, int(decimals)) for b, v in inj.items()}


def solve_engine_dispatch(
    pt: PTDFData,
    engine: MarketEngine,
    exo: BusValues | None = None,
    flow_offsets: LineValues | None = None,
    shed_price: float | None = None,
    demand_bids: list[dict[str, Any]] | None = None,
    flowgates: FlowgateSet | None = None,
    dc_links: dict[str, dict[str, Any]] | None = None,
) -> EngineResult:
    """Clear one engine's DC-OPF on the shared network.

    Parameters
    ----------
    pt : PTDFData
        Shared shift factors.
    engine : MarketEngine
        Resource/load/activated-constraint partition.
    exo : dict, optional
        ``{bus: MW}`` price-taking exogenous injections (positive = injection
        into this engine's footprint). This is how a neighboring engine's
        export shows up: a fixed, price-insensitive schedule (paper §2.3, §4.2).
    flow_offsets : dict, optional
        ``{line: MW}`` signed flow (in each line's reference direction) that
        this engine *accommodates* on its activated limits: the limit is
        enforced on ``F^M_m + offset_m`` rather than on ``F^M_m`` alone, so the
        engine leaves room for another party's anticipated flow. The offset is
        exogenous to the optimization (an ATC-style reservation), not a
        decision variable. Default ``None`` leaves the limits unchanged.
    shed_price : float, optional
        Power-balance relaxation penalty, $/MWh (load shedding). When set, an
        unserved-load variable ``0 ≤ u_n ≤ d_n`` is added at every load bus and
        priced at ``shed_price`` in the objective, so the clearing stays
        feasible when generation cannot reach load (e.g. behind a binding
        constraint). Mechanically ``u_n`` is a virtual generator at the load
        bus with cost ``shed_price``: nothing sheds while cheaper supply can
        reach the bus, and where shedding is interior the bus LMP equals
        ``shed_price`` (the penalty becomes the price cap). Mirrors the EDAM
        power-balance relaxation (per-BAA energy-supply-shortfall variables at
        penalty costs — Draft Technical Description §5.5/§11). Shed quantities
        return as ``shed_by_bus``; ``load_by_bus`` stays the NOMINAL load
        (served = load − shed). ``total_cost`` remains the production cost of
        dispatch only (the penalty term is not included). Default ``None``
        disables the relaxation (an unservable engine raises, as before).
    demand_bids : list of dict, optional
        Price-sensitive demand (the mirror of a supply bid): each entry
        ``{'id', 'bus', 'mw', 'price'}`` is a willingness-to-pay block —
        up to ``mw`` MW withdrawn at ``bus`` that clears only while the bus
        LMP is at or below ``price``. This is how an *economic export* is bid
        at an intertie: a demand offer whose bid is distinct from any
        generator's marginal cost. Mechanically a block is a variable
        ``0 ≤ q ≤ mw`` with objective coefficient ``−price`` (a benefit), a
        ``−1`` entry in the energy balance (it is load), and a ``−SF`` entry in
        every flow row (a withdrawal). Cleared quantities return as
        ``demand_cleared`` keyed by ``id``; ``total_cost`` stays the supply
        production cost (the demand benefit is not netted in). Default ``None``
        adds no demand bids.
    flowgates : FlowgateSet, optional
        Enforce limits on FLOWGATES (signed sums of line flows — WECC rated
        paths) instead of individual lines. ``engine.activated_lines`` then
        names flowgate ids; ``flow_offsets`` keys flowgate ids. ``line_dual``
        carries the IMPLIED per-line duals ``Sᵀμ`` so every downstream ledger
        and figure reads unchanged; ``fg_dual``/``fg_flow``/``fg_limit`` carry
        the flowgate layer itself. Default ``None`` = per-line limits (the
        original behavior, byte-identical).
    dc_links : dict, optional
        Controllable radial HVDC ties (PDCI, IPP, …): ``{link_id: {"bus0",
        "bus1", "p_max", "p_min"}}``. Each link adds ONE signed dispatch
        variable ``t ∈ [p_min, p_max]`` (MW, positive = bus0 → bus1;
        ``p_min`` defaults to 0, pass a negative value for reverse
        capability) at zero cost: injection ``−t`` at bus0 and ``+t`` at
        bus1, so the energy balance is untouched and each activated
        constraint row gains the column ``H[:,bus1] − H[:,bus0]``. The
        optimum arbitrages the two ends' LMPs until the link saturates —
        its rent is ``(λ_{bus1} − λ_{bus0})·t``. Flows return in
        ``link_flow``. Default ``None`` adds no links.

    Returns
    -------
    EngineResult
        Dispatch, the LMP at *every* bus (eq. 9), the energy price, signed line
        congestion duals, and the engine's own flow component (eq. 6).
    """
    exo = {str(k): float(v) for k, v in (exo or {}).items()}
    offsets = {str(k): float(v) for k, v in (flow_offsets or {}).items()}
    gen_ids = list(engine.gens)
    G = len(gen_ids)
    if G == 0:
        raise ValueError(f"Engine {engine.name!r} has no generators.")

    cost = np.array([engine.gens[g]["cost"] for g in gen_ids])
    cap = np.array([engine.gens[g]["p_nom"] for g in gen_ids])
    gen_bus = [str(engine.gens[g]["bus"]) for g in gen_ids]

    # Load-shed relaxation columns: one unserved-load variable u_n per load bus,
    # structurally a virtual generator at that bus with cost shed_price.
    shed_bus: list[str] = []
    shed_cap_l: list[float] = []
    if shed_price is not None:
        for b, v in engine.loads.items():
            if float(v) > 0:
                shed_bus.append(str(b))
                shed_cap_l.append(float(v))
    K = len(shed_bus)
    shed_cap = np.array(shed_cap_l)

    # Price-sensitive demand bids (the mirror of a supply bid): variable q ∈ [0, mw]
    # at a bus, a benefit (−price) in the objective, load (−1) in the balance, and a
    # withdrawal (−SF) in the flow rows. This is how an economic export bids at a tie.
    dem_ids: list[str] = []
    dem_bus: list[str] = []
    dem_cap_l: list[float] = []
    dem_price_l: list[float] = []
    for j, db in enumerate(demand_bids or []):
        if float(db["mw"]) <= 0:
            continue
        dem_ids.append(str(db.get("id", f"demand_{j}")))
        dem_bus.append(str(db["bus"]))
        dem_cap_l.append(float(db["mw"]))
        dem_price_l.append(float(db["price"]))
    D = len(dem_ids)
    dem_cap = np.array(dem_cap_l)

    # DC links: one signed transfer variable per link (see the docstring).
    link_ids = list(dc_links or {})
    L = len(link_ids)
    link_b0 = [str(dc_links[l]["bus0"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    link_b1 = [str(dc_links[l]["bus1"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    link_lo = [float(dc_links[l].get("p_min", 0.0)) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    link_hi = [float(dc_links[l]["p_max"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None

    # Fixed injection at each bus from loads (−) and exogenous schedules (+)
    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw

    total_load = load_vec.sum()
    total_exo = exo_vec.sum()

    # ── Energy balance:  Σ_g g + Σ_n u_n − Σ_j q_j = total_load − total_exo ──
    # gens (+1) and shed (+1) supply; demand bids (−1) are extra load served;
    # a DC link's two ends cancel, so its balance coefficient is 0.
    A_eq = np.concatenate([np.ones((1, G + K)), -np.ones((1, D))], axis=1)
    if L:
        A_eq = np.concatenate([A_eq, np.zeros((1, L))], axis=1)
    b_eq = np.array([total_load - total_exo])

    # ── Activated constraint layer: per-LINE (default) or per-FLOWGATE ───
    # With flowgates=None the aliases ARE the pt arrays (same objects), so the
    # arithmetic below is the original per-line construction, op for op.
    if flowgates is None:
        cids, Hmat, fwd, rev, cidx = pt.lines, pt.ptdf, pt.s_nom, pt.s_nom, pt.line_idx
    else:
        cids, Hmat = flowgates.ids, flowgates.H(pt)
        fwd, rev, cidx = flowgates.fwd, flowgates.rev, flowgates.fg_idx
    if engine.activated_lines == "all":
        act = list(range(len(cids)))
    else:
        act = [cidx[str(l)] for l in engine.activated_lines]

    # f_k = Σ_g H[k, bus_g] g_g + Ffix_k,  Ffix_k = Σ_n H[k,n](exo_n − load_n)
    Ffix = Hmat @ (exo_vec - load_vec)
    gen_sf = np.array([[Hmat[l, pt.bus_idx[gb]] for gb in gen_bus] for l in act]) \
        if act else np.zeros((0, G))
    shed_sf = np.array([[Hmat[l, pt.bus_idx[b]] for b in shed_bus] for l in act]) \
        if act else np.zeros((0, K))
    # demand bids withdraw, so their flow contribution is −SF (a negative injection)
    dem_sf = np.array([[-Hmat[l, pt.bus_idx[b]] for b in dem_bus] for l in act]) \
        if act else np.zeros((0, D))
    # a DC link transfers t from bus0 to bus1: column H[:,bus1] − H[:,bus0]
    link_sf = np.array([[Hmat[l, pt.bus_idx[b1]] - Hmat[l, pt.bus_idx[b0]]
                         for b0, b1 in zip(link_b0, link_b1)] for l in act]) \
        if act else np.zeros((0, L))

    A_ub, b_ub = [], []
    for row, l in enumerate(act):
        off = offsets.get(cids[l], 0.0)                # accommodated flow (signed)
        row_sf = np.concatenate([gen_sf[row], shed_sf[row], dem_sf[row],
                                 link_sf[row]])
        A_ub.append(row_sf)                            # +(flow+off) ≤ fwd − Ffix
        b_ub.append(fwd[l] - Ffix[l] - off)
        A_ub.append(-row_sf)                           # −(flow+off) ≤ rev + Ffix
        b_ub.append(rev[l] + Ffix[l] + off)

    A_ub = np.array(A_ub) if A_ub else None  # type: ignore[assignment]  # rebinds the row list
    b_ub = np.array(b_ub) if b_ub else None  # type: ignore[assignment]  # rebinds the row list

    c_vec = np.concatenate([
        cost,
        np.full(K, float(shed_price)) if K else np.zeros(0),  # type: ignore[arg-type]  # K > 0 only with shed_price set
        -np.array(dem_price_l) if D else np.zeros(0),   # demand bid: benefit = −price
        np.zeros(L),                                    # DC links transfer at no cost
    ])
    bounds = ([(0, c) for c in cap] + [(0, c) for c in shed_cap]
              + [(0, c) for c in dem_cap]
              + list(zip(link_lo, link_hi)))
    res = linprog(
        c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Engine {engine.name!r} infeasible: {res.message}")

    g_opt = res.x[:G]
    u_opt = res.x[G:G + K]
    d_opt = res.x[G + K:G + K + D]
    t_opt = res.x[G + K + D:G + K + D + L]
    dispatch = {gid: float(g_opt[i]) for i, gid in enumerate(gen_ids)}

    # ── Dual decomposition → nodal LMP (eq. 9) ───────────────────────────
    # HiGHS marginals follow ∂(objective)/∂(rhs):
    #   energy component  = eqlin marginal (positive = marginal $/MWh)
    #   line congestion   = Σ_l PTDF[l,n] · (m_upper_l − m_lower_l)
    energy_price = float(res.eqlin.marginals[0])
    m = res.ineqlin.marginals if (A_ub is not None) else np.array([])

    fg_dual = {cids[l]: 0.0 for l in range(len(cids))}
    cong = np.zeros(pt.n_bus)
    for row, l in enumerate(act):
        mu = float(m[2 * row] - m[2 * row + 1])        # signed congestion dual
        fg_dual[cids[l]] = mu
        cong += Hmat[l] * mu
    if flowgates is None:
        line_dual = dict(fg_dual)                      # cids ARE the lines
    else:
        impl = flowgates.implied_line_duals(fg_dual)   # μ^impl = Sᵀ μ_fg
        line_dual = {pt.lines[i]: float(impl[i]) for i in range(pt.n_line)}

    lmp = {pt.buses[n]: energy_price + cong[n] for n in range(pt.n_bus)}

    # ── Engine's own flow component F^M_m (eq. 6) ────────────────────────
    gen_by_bus: dict[str, float] = {}
    for i, gid in enumerate(gen_ids):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + g_opt[i]
    inj = np.zeros(pt.n_bus)
    for b, mw in gen_by_bus.items():
        inj[pt.bus_idx[b]] += mw
    shed_vec = np.zeros(pt.n_bus)
    for b, u in zip(shed_bus, u_opt):
        shed_vec[pt.bus_idx[b]] += u
    dem_vec = np.zeros(pt.n_bus)
    for b, q in zip(dem_bus, d_opt):
        dem_vec[pt.bus_idx[b]] += q
    inj += exo_vec - load_vec + shed_vec - dem_vec   # served load (d−u) plus cleared demand bids
    for b0, b1, t in zip(link_b0, link_b1, t_opt):   # DC-link transfers
        inj[pt.bus_idx[b0]] -= t
        inj[pt.bus_idx[b1]] += t
    flow_own = {pt.lines[l]: float((pt.ptdf[l] @ inj)) for l in range(pt.n_line)}

    return EngineResult(
        name=engine.name,
        dispatch=dispatch,
        gen_by_bus=gen_by_bus,
        load_by_bus={str(b): float(v) for b, v in engine.loads.items()},
        exo_by_bus=dict(exo),
        injection=inj,
        lmp=lmp,
        energy_price=energy_price,
        line_dual=line_dual,
        flow_own=flow_own,
        total_cost=float(cost @ g_opt),
        status=res.message,
        shed_by_bus={b: float(u) for b, u in zip(shed_bus, u_opt) if u > 1e-9},
        demand_cleared={i: float(q) for i, q in zip(dem_ids, d_opt) if q > 1e-9},
        demand_by_bus=_demand_by_bus(dem_bus, d_opt),
        link_flow={lid: float(t) for lid, t in zip(link_ids, t_opt)},
        fg_dual=fg_dual,
        fg_flow={cids[k]: float(Hmat[k] @ inj) for k in range(len(cids))},
        fg_limit={cids[k]: (float(fwd[k]), float(rev[k]))
                  for k in range(len(cids))},
        fg_members=({} if flowgates is None else
                    {k: list(flowgates.members.get(k, [])) for k in cids}),
    )


#: Active-set band for the QP's dual recovery, in MW plus a share of the line's
#: rating. ``trust-constr`` satisfies a binding limit only to its own convergence
#: tolerance and was measured stopping up to 2e-4 MW inside one, so a line at its
#: limit must still be recognised as binding from a little inside it.
_QP_ACTIVE_BAND_MW = 1e-3
_QP_ACTIVE_BAND_REL = 1e-6
#: A recovered dual at or below this magnitude means the candidate line is not
#: actually binding and is pruned from the active set.
_QP_MU_ZERO = 1e-7


def solve_engine_qp(
    pt: PTDFData,
    engine: MarketEngine,
    curves: dict[Resource, tuple[float, float]] | None = None,
    exo: BusValues | None = None,
    shed_price: float | None = None,
    binding_lines: list[Line] | None = None,
    tol: float = 1e-3,
    flowgates: FlowgateSet | None = None,
) -> EngineResult:
    """Clear one engine with **continuous linear marginal-cost curves** (a convex QP).

    The flat-LP ``solve_engine_dispatch`` gives every generator a single offer
    price; this variant lets a unit's marginal cost *rise with output* along a
    true straight line — no block staircase. For ``curves[g] = (a, b)`` the unit's
    cost is ``a·g + ½·b·g²`` so its marginal cost is ``MC_g(g) = a + b·g``; a unit
    absent from ``curves`` keeps its flat engine cost (``b = 0``, ``a = cost``).

    Everything else — the shared PTDF, the activated line limits, the ``exo``
    schedule, the load-shed relaxation, and the LMP decomposition
    ``λ_n = λ + Σ_m SF_{n,m} μ_m`` — is identical to ``solve_engine_dispatch``
    and the returned ``EngineResult`` carries the same fields and sign
    conventions, so every downstream figure/ledger reads it unchanged. With
    ``b = 0`` for all units it reproduces the flat LP exactly.

    Because there is no QP backend in the environment, the strictly-convex primal
    is solved with ``scipy.optimize.minimize(method="trust-constr")`` and the duals
    are recovered analytically from KKT stationarity at the *interior* generators:
    for any unit with ``0 < g_g < p_nom`` and bound multipliers zero,
    ``MC_g(g_g) = λ + Σ_m SF_{bus(g),m} μ_m``. With one balance
    dual and the binding-line duals as
    unknowns, the interior-gen equations pin them by least squares. Pass
    ``binding_lines`` (e.g. the active set from a companion flat solve) to fix the
    dual-carrying line set when the unconstrained dispatch is degenerate.

    Parameters mirror ``solve_engine_dispatch``; ``curves`` and ``binding_lines``
    are the only additions. ``demand_bids`` are not supported here.
    """
    curves = curves or {}
    exo = {str(k): float(v) for k, v in (exo or {}).items()}
    gen_ids = list(engine.gens)
    G = len(gen_ids)
    if G == 0:
        raise ValueError(f"Engine {engine.name!r} has no generators.")
    gen_bus = [str(engine.gens[g]["bus"]) for g in gen_ids]
    cap = np.array([engine.gens[g]["p_nom"] for g in gen_ids])
    a = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[0] for g in gen_ids])
    b = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[1] for g in gen_ids])

    # Load-shed columns: virtual flat gens at the load buses, cost shed_price.
    shed_bus: list[str] = []
    shed_cap_l: list[float] = []
    if shed_price is not None:
        for bb, v in engine.loads.items():
            if float(v) > 0:
                shed_bus.append(str(bb))
                shed_cap_l.append(float(v))
    K = len(shed_bus)
    shed_cap = np.array(shed_cap_l) if K else np.zeros(0)

    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw
    total_load = load_vec.sum()
    total_exo = exo_vec.sum()

    if flowgates is None:
        cids, Hmat, fwd, rev, cidx = pt.lines, pt.ptdf, pt.s_nom, pt.s_nom, pt.line_idx
    else:
        cids, Hmat = flowgates.ids, flowgates.H(pt)
        fwd, rev, cidx = flowgates.fwd, flowgates.rev, flowgates.fg_idx
    if engine.activated_lines == "all":
        act = list(range(len(cids)))
    else:
        act = [cidx[str(l)] for l in engine.activated_lines]
    Ffix = Hmat @ (exo_vec - load_vec)
    gen_sf = np.array([[Hmat[l, pt.bus_idx[gb]] for gb in gen_bus] for l in act]) \
        if act else np.zeros((0, G))
    shed_sf = np.array([[Hmat[l, pt.bus_idx[bb]] for bb in shed_bus] for l in act]) \
        if act else np.zeros((0, K))

    N = G + K
    H = np.zeros((N, N))
    H[np.arange(G), np.arange(G)] = b
    c_lin = np.concatenate([a, np.full(K, float(shed_price)) if K else np.zeros(0)])  # type: ignore[arg-type]  # K > 0 only with shed_price set

    cons = [LinearConstraint(np.concatenate([np.ones(G), np.ones(K)]).reshape(1, -1),
                             total_load - total_exo, total_load - total_exo)]
    if act:
        rows, lb, ub = [], [], []
        for r, l in enumerate(act):
            rows.append(np.concatenate([gen_sf[r], shed_sf[r]]))
            lb.append(-rev[l] - Ffix[l])
            ub.append(fwd[l] - Ffix[l])
        cons.append(LinearConstraint(np.array(rows), np.array(lb), np.array(ub)))

    bounds = Bounds(np.zeros(N), np.concatenate([cap, shed_cap]))
    x0 = np.minimum(bounds.ub, np.maximum(bounds.lb, np.full(N, (total_load - total_exo) / max(N, 1))))
    res = minimize(lambda x: 0.5 * x @ (H @ x) + c_lin @ x, x0, method="trust-constr",
                   jac=lambda x: H @ x + c_lin, hess=lambda x: H,
                   constraints=cons, bounds=bounds,
                   options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 2000})
    x = res.x
    g_opt = x[:G]
    u_opt = x[G:G + K]

    # ── Dual recovery from interior-gen stationarity ─────────────────────
    interior = [i for i in range(G) if tol < g_opt[i] < cap[i] - tol]

    def _fk(r: int, l: int) -> float:
        return (gen_sf[r] @ g_opt + (shed_sf[r] @ u_opt if K else 0.0)) + Ffix[l]

    def _kkt(
        bset: list[int],
    ) -> tuple[float, dict[int, float], float] | tuple[None, None, None]:
        """Least-squares KKT solve for (λ, μ) on a candidate active set.

        One row per interior generator (its marginal cost equals its bus price)
        and one per interior shed variable (its bus sits at the shed price).
        Returns ``(None, None, None)`` when the vertex is fully degenerate and
        stationarity determines nothing.
        """
        A_, rhs_ = [], []
        for i in interior:
            A_.append([1.0] + [Hmat[l, pt.bus_idx[gen_bus[i]]] for l in bset])
            rhs_.append(a[i] + b[i] * g_opt[i])
        for j in range(K):
            # An interior shed variable is marginal exactly like an interior
            # generator: its flat cost is the shed price, so stationarity pins
            # its bus at V (the flat LP's behavior). Without these rows an
            # interval whose only marginal variable is shed leaves the KKT
            # system empty.
            if tol < u_opt[j] < shed_cap[j] - tol:
                A_.append([1.0] + [Hmat[l, pt.bus_idx[shed_bus[j]]] for l in bset])
                rhs_.append(float(shed_price))  # type: ignore[arg-type]  # K > 0 only with shed_price set
        if not A_:
            return None, None, None
        A_ = np.array(A_)  # type: ignore[assignment]  # rebinds the row list to an array
        rhs_ = np.array(rhs_)  # type: ignore[assignment]  # rebinds the row list to an array
        sol, *_ = np.linalg.lstsq(A_, rhs_, rcond=None)
        resid = float(np.max(np.abs(A_ @ sol - rhs_))) if len(rhs_) else 0.0
        return float(sol[0]), {l: float(sol[1 + k]) for k, l in enumerate(bset)}, resid

    if binding_lines is not None:
        binding = [cidx[str(l)] for l in binding_lines]
        lam, mu, _resid = _kkt(binding)
    else:
        # The active set carries the congestion columns of the KKT system, so a
        # binding line dropped from it leaves only the balance column and every
        # price collapses to the average of the interior units' marginal costs.
        # ``trust-constr`` stops a little INSIDE a binding limit, so the
        # candidate band covers its stopping error and the set is then pruned by
        # complementarity: a line that is merely near its limit recovers a zero
        # dual and is removed. Pruning is accepted only while the stationarity
        # residual does not worsen, so the recovered (λ, μ) is unchanged wherever
        # a narrow band was already right.
        binding = [
            l for r, l in enumerate(act)
            if _fk(r, l) > fwd[l] - (_QP_ACTIVE_BAND_MW + _QP_ACTIVE_BAND_REL * fwd[l])
            or _fk(r, l) < -(rev[l] - (_QP_ACTIVE_BAND_MW + _QP_ACTIVE_BAND_REL * rev[l]))
        ]
        lam, mu, _resid = _kkt(binding)
        while mu is not None and binding:
            drop = [l for l in binding if abs(mu[l]) <= _QP_MU_ZERO]
            if not drop:
                break
            trial = [l for l in binding if l not in drop]
            t_lam, t_mu, t_resid = _kkt(trial)
            if t_mu is None or t_resid > max(_resid, _QP_MU_ZERO) * 1.000001 + _QP_MU_ZERO:  # type: ignore[operator, type-var]  # a None residual comes with a None mu
                break          # dropping them would break stationarity; keep the set
            binding, lam, mu, _resid = trial, t_lam, t_mu, t_resid
    if mu is None:
        # Fully degenerate vertex: every generator and shed variable sits at a
        # bound, so stationarity determines no dual. Support the dispatch with a
        # flat price at the dearest dispatched unit's marginal cost and carry no
        # line duals (no price separation, no rent) rather than crash.
        on = [i for i in range(G) if g_opt[i] > tol]
        lam = float(max(a[i] + b[i] * g_opt[i] for i in on)) if on else float(np.min(a))
        mu = {l: 0.0 for l in binding}

    fg_dual = {cids[l]: 0.0 for l in range(len(cids))}
    cong = np.zeros(pt.n_bus)
    for l in binding:
        fg_dual[cids[l]] = mu[l]
        cong += Hmat[l] * mu[l]
    if flowgates is None:
        line_dual = dict(fg_dual)
    else:
        impl = flowgates.implied_line_duals(fg_dual)
        line_dual = {pt.lines[i]: float(impl[i]) for i in range(pt.n_line)}
    lmp = {pt.buses[n]: lam + cong[n] for n in range(pt.n_bus)}

    gen_by_bus: dict[str, float] = {}
    for i in range(G):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + g_opt[i]
    inj = np.zeros(pt.n_bus)
    for bb, mw in gen_by_bus.items():
        inj[pt.bus_idx[bb]] += mw
    shed_vec = np.zeros(pt.n_bus)
    for bb, uu in zip(shed_bus, u_opt):
        shed_vec[pt.bus_idx[bb]] += uu
    inj += exo_vec - load_vec + shed_vec
    flow_own = {pt.lines[l]: float(pt.ptdf[l] @ inj) for l in range(pt.n_line)}

    return EngineResult(
        name=engine.name,
        dispatch={gid: float(g_opt[i]) for i, gid in enumerate(gen_ids)},
        gen_by_bus=gen_by_bus,
        load_by_bus={str(bb): float(v) for bb, v in engine.loads.items()},
        exo_by_bus=dict(exo),
        injection=inj,
        lmp=lmp,
        energy_price=lam,  # type: ignore[arg-type]  # lam is set wherever mu is
        line_dual=line_dual,
        flow_own=flow_own,
        total_cost=float(np.sum(a * g_opt + 0.5 * b * g_opt ** 2)),
        status=res.message,
        shed_by_bus={bb: float(uu) for bb, uu in zip(shed_bus, u_opt) if uu > 1e-9},
        fg_dual=fg_dual,
        fg_flow={cids[k]: float(Hmat[k] @ inj) for k in range(len(cids))},
        fg_limit={cids[k]: (float(fwd[k]), float(rev[k]))
                  for k in range(len(cids))},
        fg_members=({} if flowgates is None else
                    {k: list(flowgates.members.get(k, [])) for k in cids}),
    )


# ──────────────────────────────────────────────────────────────────────────
# EDAM-literal clearing: per-area balances + scheduled transfer variables
# ──────────────────────────────────────────────────────────────────────────
def solve_engine_transfers(
    pt: PTDFData,
    engine: MarketEngine,
    areas: AreaBuses,
    interfaces: Interfaces,
    curves: dict[Resource, tuple[float, float]] | None = None,
    exo: BusValues | None = None,
    shed_price: float | None = None,
    tol: float = 1e-3,
    flowgates: FlowgateSet | None = None,
    dc_links: dict[str, dict[str, Any]] | None = None,
    interface_vrl: float | None = None,
    fixed_transfers: dict[Interface, float] | None = None,
    flow_offsets: LineValues | None = None,
    gen_min: dict[Resource, float] | None = None,
    group_caps: list[tuple[list[Resource], float]] | None = None,
    demand_bids: list[dict[str, Any]] | None = None,
    schedule_limits: dict[str, tuple[list[str], float]] | None = None,
) -> EngineResult:
    """The EDAM-literal clearing of one engine, one power balance per area.

    Each area's balance has the dual ``λ̂_a``, the per-BAA energy price, and
    the balances are tied together by directional scheduled-transfer
    variables ``t_ab ≥ 0`` on the declared interfaces, each capped by its
    released capacity (``caiso_edam_techdesc`` sections 3, 3.3 and 5.2:
    matched export/import TSR pairs, net transfer in each BAA's balance).

    The transfer variables are FLOW-INERT (the GAP-to-GAP convention): physical
    line flows come from the actual nodal injections, and the hub-decomposition
    identity makes that exact for any transfer split consistent with the net
    positions. Transfers therefore do two jobs only — they carry the
    per-interface scheduling limits, and they are the tags the settlement rides
    on. Because the split does not touch flows, a slack interface CYCLE leaves
    the tags undetermined; the tags are pinned by a SECOND, lexicographic stage
    that minimizes total scheduled MW at the cost-optimal dispatch (tagged
    DIRECT wherever a direct interface with released capacity exists; the
    documented house convention — real EDAM leaves residual ties to the
    solver). The two-stage form leaves every price and dual untouched.

    Parameters
    ----------
    pt : PTDFData
        The shared network's shift factors, ``pt.ptdf[m, n]`` (line first).
    engine : MarketEngine
        The engine's generators, loads and activated constraint set.
    areas : dict
        ``{area: [buses]}`` — a DISJOINT partition covering every bus that
        carries this engine's generation or load.
    interfaces : dict
        ``{(a, b): limit}`` — the interfaces that EXIST (pairs may be in either
        order; stored sorted). ``limit`` is the released transfer capacity
        ``T̄_ab`` applied to each direction, or ``None`` for unlimited. A pair
        absent from the dict has NO interface: no schedule may ride it, however
        the transmission elements connect.
    curves : dict, optional
        ``{gen: (a, b)}`` rising linear marginal costs — clears the convex QP
        (``trust-constr``) instead of the flat LP, as in ``solve_engine_qp``.
    exo : dict, optional
        ``{bus: MW}`` price-taking exogenous injections (positive = injection),
        as in ``solve_engine_dispatch``. Every exo bus must lie in a declared
        area: the injection credits that area's power balance, enters the line
        flows through the PTDF, and moves the net position the tags settle —
        exactly the legacy fold-in of ``exo`` into the interchange.
    shed_price : float, optional
        Load-shed relaxation at penalty V, as in ``solve_engine_dispatch``.
    tol : float, default 1e-3
        Absolute tolerance in MW that the QP path (``curves``) uses to call
        a generator, shed or transfer interior in its KKT recovery.
    flowgates : FlowgateSet, optional
        Activated constraints as flowgates instead of lines, as in
        ``solve_engine_dispatch``; the per-line duals are then the implied
        duals ``μ^impl = Sᵀ μ_fg``.
    dc_links : dict, optional
        Controllable radial HVDC ties, as in ``solve_engine_dispatch`` —
        ``{link_id: {"bus0", "bus1", "p_max", "p_min"}}``, one signed
        PHYSICAL variable per link (unlike the flow-inert scheduled
        transfers, a link moves real flow AND both end-areas' balances).
        Both ends must lie in declared areas. Not supported with ``curves``.
    interface_vrl : float, optional
        Violation Relaxation Limit for the released-capacity constraints,
        $/MWh (SPP-style; Markets+ carries a $3,000 VRL on its transfer
        interface). When set, each finite ``t_ab ≤ T̄_ab`` becomes
        ``t_ab − s_ab ≤ T̄_ab`` with a violation variable ``s_ab ≥ 0`` priced
        at the VRL in the objective: a schedule may exceed the released
        capacity only at that penalty, the interface shadow price is CAPPED
        at the VRL, and scarcity that would otherwise be an infeasibility
        becomes a priced violation (reported in ``transfer_violations``).
        An interface ABSENT from ``interfaces`` stays structurally absent —
        no variable exists, so no penalty can buy a schedule on it. Default
        ``None`` keeps every released capacity hard. LP only (not
        ``curves``); the violation MW is excluded from ``total_cost``
        (production cost), like the shed penalty.
    fixed_transfers : dict, optional
        ``{(a, b): signed MW}`` — SELF-SCHEDULED tags. Each named pair's
        schedule is pinned at the given value (positive = the sorted-first
        area exports) instead of being optimized: the tag is price-taking,
        consumes the pair's released capacity like any schedule, and the
        engine optimizes only the remaining transfers — while the exporting
        area's balance must SUPPORT the pinned export from its own resources.
        Institutionally this is interchange with counterparties OUTSIDE a
        market footprint: the historic pairwise schedule is a fact the market
        carries, not a decision it makes — and when two engines study the
        same interval, passing the SAME dict to both keeps their cross-seam
        books consistent. Each pinned pair must be declared in
        ``interfaces`` and fit its released capacity (clear errors
        otherwise). LP only (not ``curves``).
    flow_offsets : dict, optional
        ``{constraint_id: MW}`` signed accommodated flow added to each
        activated constraint's computed flow before its limits are checked,
        exactly as in ``solve_engine_dispatch`` — the ex-ante reservation
        hook. Keys name flowgate ids when ``flowgates`` is given, line ids
        otherwise. Reported flows exclude the offset.
    gen_min : dict, optional
        ``{gen: MW}`` lower bound on a generator's dispatch, the MUST-RUN
        hook. The named generator's column takes the bound in place of
        zero, so the clearing runs at least that much of it whatever the
        price. A generator absent from the dict keeps its zero bound, so
        ``None`` (the default) leaves the program exactly as it was. A
        bound above the generator's capacity is an error, because it cannot
        be met. LP only (not ``curves``).
    group_caps : list, optional
        ``[(gen_ids, cap_MW)]`` one upper bound on the TOTAL dispatch of a
        set of generators. It exists because a storage's hourly capability
        is a fleet quantity: the members' own capacities may each be met
        while their sum exceeds what the fleet can deliver. Each entry adds
        one inequality row, placed after the activated-constraint rows and
        before the scheduling-limit and VRL rows, and its dual comes back in
        ``group_cap_dual`` in the order passed. LP only (not ``curves``).
    demand_bids : list of dict, optional
        ``[{"id", "bus", "mw", "price"}]`` price-sensitive demand, exactly
        as in ``solve_engine_dispatch``: one variable ``q_j`` in
        ``[0, mw]`` at the named bus, a benefit ``-price`` in the
        objective, load (``-1``) in its area's power balance, and a
        withdrawal (``-SF``) in the flow rows. A bid clears where the bus
        price sits at or below its price. It exists here because a
        battery's charging is a price-sensitive withdrawal: the battery
        buys energy where the local price is low enough against its own
        value. The cleared megawatts come back in ``demand_cleared``
        keyed by ``id``, and the withdrawal enters the net position the
        tags settle. The benefit is excluded from ``total_cost``, which
        stays the supply cost. LP only (not ``curves``).
    schedule_limits : dict, optional
        ``{limit_id: (demand_bid_ids, limit_mw)}``, the intertie scheduling
        limits (ISL), each a cap on the scheduled MW at one scheduling
        point. Each entry adds one inequality row with coefficient one on
        the cleared MW ``q_j`` of every named demand bid and no coefficient
        on any flow or other variable, ``Σ_{j∈J_s} q_j ≤ Q̄_s`` with dual
        ``μ^s ≥ 0``, where ``J_s`` is the set of named bid ids and
        ``Q̄_s`` is ``limit_mw`` in MW. The named bids are the export awards
        at the point. The keyword applies to exports the caller enters as
        demand bids, each a price-sensitive withdrawal at the point whose
        withdrawal still moves the flows through the bid's bus. In the WECC
        half, entering an interchange as bids replaces the fixed-injection
        boundary of ``CLAUDE.md`` section 6 rule 2 for that interchange, and
        the calling notebook names it as a methodology change. The
        rows sit after the ``group_caps`` rows and before the VRL rows. A
        bid id absent from ``demand_bids`` raises ``KeyError``. A bid that
        ``demand_bids`` carries with nonpositive ``mw`` has no column, so a
        limit naming it is accepted and counts it as zero. An empty id list
        or a negative or non-finite limit raises ``ValueError``, and a bare
        string in place of the id list raises ``TypeError``. ``None``
        (the default) and ``{}`` add no row. LP only (not ``curves``).
        ``solve_engine_dispatch`` carries no such argument and is unchanged.

    Returns
    -------
    EngineResult
        As usual, plus ``area_prices`` (``λ̂_a``), ``transfers`` (net scheduled
        MW per sorted pair, the stage-two tags), ``transfer_duals`` (signed
        ``μ_T^{ab} = λ̂_b − λ̂_a`` where a limit binds, else 0), and
        ``transfer_limits``. ``lmp`` is ``λ̂_{a(n)} + Σ_m SF_{n,m} μ_m`` and
        ``energy_price`` is the LMP at the PTDF slack bus (equal to the pooled
        energy price whenever no interface binds). With ``schedule_limits``,
        ``schedule_limit_dual`` is ``{limit_id: μ^s}`` in $/MWh,
        nonnegative and zero when the row is slack, and
        ``schedule_limit_mw`` is ``{limit_id: Σ_{j∈J_s} q_j}`` in MW; both
        are empty without it.

    Notes
    -----
    **The intertie scheduling limit.** ``caiso_edam_techdesc`` section 4
    indexes intertie scheduling limits in the generalized network
    constraint set ``m`` beside the facility limits, with the import and
    the export resources associated with each limit, and section 10 bounds
    every member of that set on both sides, ``LFL_m ≤ … ≤ UFL_m``, with a
    shadow price ``μ_m`` that enters the LMP through the term
    ``−Σ_m SF_{i,m} μ_m``. Section 10 gives no value for an ISL
    coefficient. ``caiso_dame_edam2023`` (letter p. 178, citing tariff
    appendix C, revised section A.3) does: the shift factor of an intertie
    resource to an intertie constraint or intertie scheduling limit at its
    scheduling point "is +1 for an import and −1 for an export", and zero
    for every other resource, so the row carries no flow. The export's −1
    is in the source's injection convention, in which an export schedule is
    a negative injection; on the award ``q_j``, a withdrawal, it is the +1
    of this row. With the source's sign, the export's price is
    ``λ_n − (−1) μ_m = λ_n + μ_m``. The sign map: ``μ^s`` equals the
    source's ``μ_m`` for an export-direction limit binding at its upper
    limit ``UFL_m``, and the limit's dual does not take the canon's flipped
    line-dual sign. ``caiso_dame_edam_tariff2023`` section 33.11.1 counts a binding
    intertie scheduling limit among the conditions that collect congestion
    revenue, which ``caiso_dame_edam2023`` (pp. 186 to 188) keeps apart
    from the transfer revenue of a binding EDAM transfer scheduling limit.
    The transfer scheduling limit stays the ``interfaces`` family with its
    dual ``μ_T^{ab}``, and ``schedule_limits`` is the limit at a scheduling
    point where awards leave the footprint.

    Three departures from the source, each a Lab modeling choice:

    1. The ISL is a gross, one-sided cap on cleared demand bids. The
       source's row nets imports (+1) against exports (−1) at the point and
       is bounded on both sides, so its limit is net and two-sided; this
       row sums the named export awards only, with an upper limit only. A
       caller with fixed imports ``I`` at the point reproduces the source's
       export-direction limit by setting the row's limit Q̄_s (``limit_mw``)
       to the source's net limit plus ``I``. The source's form
       is GridPath's ``net`` basis and this row is its ``gross`` basis.
    2. The ISL dual is reported as its own family, ``schedule_limit_dual``,
       and the per-bus ``lmp`` carries no ISL term. The source enters the
       ISL dual into the LMP of each scheduling-point resource through its
       ±1 coefficient. ``lmp`` is keyed by bus, and the row has no flow
       coefficient, so ``lmp`` stays ``λ_n = λ̂_{a(n)} + Σ_m SF_{n,m} μ_m``
       and every identity stated on it (the nodal price, ``R = Σ_m |μ_m
       F_m|``, ``Σ_a N_a = R + R_T``) holds as without the limit. The
       source's price of an export resource at the point is
       ``λ_n + μ^s``.
    3. The ISL rent, ``R^ISL = μ^s · schedule_limit_mw[s]``, equals
       ``μ^s · Q̄_s`` by complementary slackness and is outside ``R``,
       ``R^dc`` and ``R_T``. A settlement at ``lmp`` does not collect it,
       and a settlement that prices the named awards at ``λ_n + μ^s``
       collects it in addition. Its allocation is the caller's.

    **The KKT conditions, verified on this program.** HiGHS reports each
    inequality marginal as the derivative of the objective in the row's
    right-hand side, which is nonpositive on a binding ``≤`` row of a
    minimization, so ``μ^s`` is the negated marginal of row ``s``. The
    column of a bid ``j`` at bus ``n`` carries the benefit ``−p_j`` in the
    objective, ``−1`` in its area's balance ``[λ̂_a]``, ``−SF_{n,m}`` in each
    flow row ``[μ_m]`` and ``+1`` in each ISL row that names it, so its
    reduced cost is ``d_j = −p_j + λ_n + Σ_{s∋j} μ^s``. Optimality
    requires ``d_j ≥ 0`` where ``q_j = 0``, ``d_j = 0`` where
    ``0 < q_j < q̄_j`` and ``d_j ≤ 0`` where ``q_j = q̄_j``. Therefore:

    - The bus price ``λ_n`` is set as without the limit, by the marginal
      units and bids and the binding flow rows. The ISL moves it only by
      changing which lines bind or which units or bids are marginal.
    - A partially cleared award ``j*`` of a binding limit ``s`` that no
      other limit names gives ``μ^s = p_{j*} − λ_{n(j*)}``, the gap
      between the marginal award's bid and the bus price. The marginal
      award's bid equals ``λ_n + μ^s``, the source's resource price at
      the point, and not the bus price ``lmp[n]``.
    - Where no award of the limit is partial, ``μ^s`` is any value in
      ``[max_{q_j=0}(p_j − λ_n), min_{q_j=q̄_j}(p_j − λ_n)]`` intersected
      with ``[0, ∞)``, and the solver reports one point of that interval.

    """
    curves = curves or {}
    areas = {str(a): [str(b) for b in bs] for a, bs in areas.items()}
    names = list(areas)
    A = len(names)
    area_of: dict[str, str] = {}
    for a, bs in areas.items():
        for b in bs:
            if b in area_of:
                raise ValueError(f"bus {b} appears in two areas ({area_of[b]}, {a})")
            area_of[b] = a
    for b in engine.buses:
        if str(b) not in area_of:
            raise ValueError(f"engine bus {b} is not covered by `areas`")

    exo = {str(k): float(v) for k, v in (exo or {}).items() if float(v) != 0.0}
    for b in exo:
        if b not in area_of:
            raise ValueError(f"exo bus {b} is not covered by `areas`")
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw

    ifc: list[tuple[str, str]] = []
    lims: list[float | None] = []
    for k, v in interfaces.items():
        a, b = sorted((str(k[0]), str(k[1])))
        if a not in areas or b not in areas or a == b:
            raise ValueError(f"interface {k} does not join two distinct declared areas")
        ifc.append((a, b))
        lims.append(None if v is None else float(v))
    P = len(ifc)

    gen_ids = list(engine.gens)
    G = len(gen_ids)
    if G == 0:
        raise ValueError(f"Engine {engine.name!r} has no generators.")
    gen_bus = [str(engine.gens[g]["bus"]) for g in gen_ids]
    cap = np.array([engine.gens[g]["p_nom"] for g in gen_ids])
    a_c = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[0] for g in gen_ids])
    b_c = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[1] for g in gen_ids])

    shed_bus: list[str] = []
    shed_cap_l: list[float] = []
    if shed_price is not None:
        for bb, v in engine.loads.items():
            if float(v) > 0:
                shed_bus.append(str(bb))
                shed_cap_l.append(float(v))
    K = len(shed_bus)
    shed_cap = np.array(shed_cap_l) if K else np.zeros(0)

    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw

    # DC links (physical, signed; see the docstring). Columns sit AFTER the
    # interface pairs so every existing index is untouched.
    if dc_links and curves:
        raise NotImplementedError("dc_links with rising-MC curves is not supported")
    link_ids = list(dc_links or {})
    Lk = len(link_ids)
    link_b0 = [str(dc_links[l]["bus0"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    link_b1 = [str(dc_links[l]["bus1"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    for bb in link_b0 + link_b1:
        if bb not in area_of:
            raise ValueError(f"dc_link bus {bb} is not covered by `areas`")
    link_lo = [float(dc_links[l].get("p_min", 0.0)) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None
    link_hi = [float(dc_links[l]["p_max"]) for l in link_ids]  # type: ignore[index]  # link_ids is empty when dc_links is None

    # Price-sensitive demand bids. The columns sit AFTER the links and
    # BEFORE the VRL slacks, so every index built above is untouched.
    dem_ids: list[str] = []
    dem_bus: list[str] = []
    dem_cap_l: list[float] = []
    dem_price_l: list[float] = []
    for j, db in enumerate(demand_bids or []):
        if float(db["mw"]) <= 0:
            continue
        bb = str(db["bus"])
        if bb not in area_of:
            raise ValueError(f"demand bid bus {bb} is not covered by `areas`")
        dem_ids.append(str(db.get("id", f"demand_{j}")))
        dem_bus.append(bb)
        dem_cap_l.append(float(db["mw"]))
        dem_price_l.append(float(db["price"]))
    D = len(dem_ids)
    dem_cap = np.array(dem_cap_l) if D else np.zeros(0)

    if gen_min and curves:
        raise NotImplementedError("gen_min with rising-MC curves is not supported")
    if group_caps and curves:
        raise NotImplementedError("group_caps with rising-MC curves is not supported")
    if demand_bids and curves:
        raise NotImplementedError("demand_bids with rising-MC curves is not supported")
    if schedule_limits and curves:
        raise NotImplementedError(
            "schedule_limits with rising-MC curves is not supported")
    if interface_vrl is not None and curves:
        raise NotImplementedError("interface_vrl with rising-MC curves is not supported")
    if fixed_transfers and curves:
        raise NotImplementedError("fixed_transfers with rising-MC curves is not supported")
    # VRL relaxation: one violation variable per FINITE-capped direction. An
    # interface absent from `interfaces` has no variable at all -- the VRL
    # cannot buy a schedule on a relationship that does not exist.
    vrl_cols: list[int] = []
    vrl_caps: list[float] = []
    if interface_vrl is not None:
        for p_, v_ in enumerate(lims):
            if v_ is not None:
                for d_ in (0, 1):
                    vrl_cols.append(G + K + 2 * p_ + d_)
                    vrl_caps.append(float(v_))
    S_v = len(vrl_cols)

    # g, u, (fwd, rev) per interface, links, demand bids, VRL slacks
    N = G + K + 2 * P + Lk + D + S_v
    C_D = G + K + 2 * P + Lk             # first demand-bid column
    a_idx = {a: r for r, a in enumerate(names)}

    # ── Per-area balances:  Σ_a g + Σ_a u − Σ_out t + Σ_in t = load_a ──────
    A_bal = np.zeros((A, N))
    b_bal = np.zeros(A)
    for i, gb in enumerate(gen_bus):
        A_bal[a_idx[area_of[gb]], i] = 1.0
    for j, sb in enumerate(shed_bus):
        A_bal[a_idx[area_of[sb]], G + j] = 1.0
    for p, (x, y) in enumerate(ifc):
        A_bal[a_idx[x], G + K + 2 * p] = -1.0            # t_xy exports from x …
        A_bal[a_idx[y], G + K + 2 * p] = +1.0            # … and imports into y
        A_bal[a_idx[x], G + K + 2 * p + 1] = +1.0        # t_yx the reverse
        A_bal[a_idx[y], G + K + 2 * p + 1] = -1.0
    for q, (b0, b1) in enumerate(zip(link_b0, link_b1)):
        col = G + K + 2 * P + q                          # += so an intra-area link nets to 0
        A_bal[a_idx[area_of[b0]], col] += -1.0
        A_bal[a_idx[area_of[b1]], col] += +1.0
    for j, dbus in enumerate(dem_bus):                   # a cleared bid is extra load
        A_bal[a_idx[area_of[dbus]], C_D + j] += -1.0
    for a in names:
        b_bal[a_idx[a]] = sum(load_vec[pt.bus_idx[bb]] - exo_vec[pt.bus_idx[bb]]
                              for bb in areas[a])

    # ── Activated limits, per-LINE or per-FLOWGATE (transfers stay flow-inert:
    #    zero columns either way) ─────────────────────────────────────────────
    if flowgates is None:
        cids, Hmat, fwd, rev, cidx = pt.lines, pt.ptdf, pt.s_nom, pt.s_nom, pt.line_idx
    else:
        cids, Hmat = flowgates.ids, flowgates.H(pt)
        fwd, rev, cidx = flowgates.fwd, flowgates.rev, flowgates.fg_idx
    if engine.activated_lines == "all":
        act = list(range(len(cids)))
    else:
        act = [cidx[str(l)] for l in engine.activated_lines]
    Ffix = Hmat @ (exo_vec - load_vec)
    offsets = {str(k): float(v) for k, v in (flow_offsets or {}).items()}
    flow_rows = np.zeros((len(act), N))
    for r, l in enumerate(act):
        for i, gb in enumerate(gen_bus):
            flow_rows[r, i] = Hmat[l, pt.bus_idx[gb]]
        for j, sb in enumerate(shed_bus):
            flow_rows[r, G + j] = Hmat[l, pt.bus_idx[sb]]
        for q, (b0, b1) in enumerate(zip(link_b0, link_b1)):
            flow_rows[r, G + K + 2 * P + q] = (Hmat[l, pt.bus_idx[b1]]
                                               - Hmat[l, pt.bus_idx[b0]])
        for j, dbus in enumerate(dem_bus):                # a withdrawal is -SF
            flow_rows[r, C_D + j] = -Hmat[l, pt.bus_idx[dbus]]

    t_ub = np.array([np.inf if v is None else v for v in lims for _ in (0, 1)])
    t_ub1 = np.full(2 * P, np.inf) if S_v else t_ub    # VRL: caps move to rows
    lo = np.concatenate([np.zeros(G + K + 2 * P), np.array(link_lo),
                         np.zeros(D + S_v)])
    hi = np.concatenate([cap, shed_cap, t_ub1, np.array(link_hi), dem_cap,
                         np.full(S_v, np.inf)])
    # must-run lower bounds: the named generators run at least this much
    g_pos = {g: i for i, g in enumerate(gen_ids)}
    for g_, v_ in (gen_min or {}).items():
        i_ = g_pos.get(str(g_))
        if i_ is None:
            raise ValueError(f"gen_min names {g_!r}, which is not in this engine")
        v_ = float(v_)
        if v_ > cap[i_] + 1e-9:
            raise ValueError(f"gen_min {g_!r} = {v_:.4f} MW exceeds its capacity "
                             f"{cap[i_]:.4f} MW")
        lo[i_] = max(lo[i_], min(v_, float(cap[i_])))
    # group capability rows: one per entry, members' columns summed
    grp_rows: list[FloatArray] = []
    grp_caps: list[float] = []
    for members_, cap_ in (group_caps or []):
        row_ = np.zeros(N)
        for g_ in members_:
            i_ = g_pos.get(str(g_))
            if i_ is None:
                raise ValueError(f"group_caps names {g_!r}, which is not in this engine")
            row_[i_] = 1.0
        grp_rows.append(row_)
        grp_caps.append(float(cap_))
    # ── Intertie scheduling limits: Σ_{j∈J_s} q_j ≤ Q̄_s  [μ^s ≥ 0] ──────
    # One row per limit, coefficient one on each named demand-bid column and
    # none on any flow, so the row adds no shift-factor term to any price.
    # A bid passed with nonpositive mw has no column and counts as zero.
    bid_known = {str(db.get("id", f"demand_{j}"))
                 for j, db in enumerate(demand_bids or [])}
    bid_cols: dict[str, list[int]] = {}
    for j, d_id in enumerate(dem_ids):
        bid_cols.setdefault(d_id, []).append(C_D + j)
    isl_ids: list[str] = []
    isl_cols: list[list[int]] = []
    isl_rows: list[FloatArray] = []
    isl_caps: list[float] = []
    for s_id, (bids_, lim_) in (schedule_limits or {}).items():
        s_key = str(s_id)
        if isinstance(bids_, str):              # a bare id would split into characters
            raise TypeError(f"schedule limit {s_key!r} names {bids_!r}; pass a "
                            f"list of demand bid ids")
        names_ = [str(b_) for b_ in bids_]
        if not names_:
            raise ValueError(f"schedule limit {s_key!r} names no demand bid")
        for b_ in names_:
            if b_ not in bid_known:
                raise KeyError(f"schedule limit {s_key!r} names demand bid "
                               f"{b_!r}, which is not in demand_bids")
        lim_f = float(lim_)
        if not np.isfinite(lim_f) or lim_f < 0.0:
            raise ValueError(f"schedule limit {s_key!r} = {lim_f!r} MW is not "
                             f"a finite nonnegative number")
        cols_ = sorted({c_ for b_ in names_ for c_ in bid_cols.get(b_, [])})
        row_ = np.zeros(N)
        row_[cols_] = 1.0
        isl_ids.append(s_key)
        isl_cols.append(cols_)
        isl_rows.append(row_)
        isl_caps.append(lim_f)
    # ── Self-scheduled tags: pin the pair's directional variables ──────────
    fixed_idx: dict[int, float] = {}          # directional column -> pinned MW
    if fixed_transfers:
        pair_of = {k: p for p, k in enumerate(ifc)}
        for k, v in fixed_transfers.items():
            key = tuple(sorted((str(k[0]), str(k[1]))))
            if key not in pair_of:
                raise ValueError(f"fixed transfer {key} is not a declared interface")
            p_ = pair_of[key]  # type: ignore[index]  # key is a sorted pair
            if lims[p_] is not None and abs(float(v)) > lims[p_] + 1e-6:  # type: ignore[operator]  # the test above covers this index
                raise ValueError(
                    f"fixed transfer {key} = {float(v):+.1f} MW exceeds its "
                    f"released capacity {lims[p_]:.1f} MW")
            t_ = float(v)
            fixed_idx[G + K + 2 * p_] = max(t_, 0.0)
            fixed_idx[G + K + 2 * p_ + 1] = max(-t_, 0.0)
        for c_, v_ in fixed_idx.items():
            lo[c_] = hi[c_] = v_

    if not curves:
        # ---- flat-offer LP (HiGHS) ----------------------------------------
        c_vec = np.concatenate([a_c, np.full(K, float(shed_price)) if K else np.zeros(0),  # type: ignore[arg-type]  # K > 0 only with shed_price set
                                np.zeros(2 * P + Lk),
                                -np.array(dem_price_l) if D else np.zeros(0),
                                np.full(S_v, float(interface_vrl or 0.0))])
        A_ub, b_ub = [], []
        for r, l in enumerate(act):
            off_ = offsets.get(cids[l], 0.0)           # accommodated flow (signed)
            A_ub.append(flow_rows[r]);  b_ub.append(fwd[l] - Ffix[l] - off_)
            A_ub.append(-flow_rows[r]); b_ub.append(rev[l] + Ffix[l] + off_)
        for row_, cap_ in zip(grp_rows, grp_caps):
            A_ub.append(row_); b_ub.append(cap_)
        # the scheduling-limit rows, after the group caps and before the VRL
        for row_, cap_ in zip(isl_rows, isl_caps, strict=True):
            A_ub.append(row_)
            b_ub.append(cap_)
        for k2, (col_, cap_) in enumerate(zip(vrl_cols, vrl_caps)):
            row_ = np.zeros(N)
            row_[col_] = 1.0
            row_[C_D + D + k2] = -1.0                  # t - s <= T-bar
            A_ub.append(row_); b_ub.append(cap_)
        res = linprog(c_vec, A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=A_bal, b_eq=b_bal,
                      bounds=list(zip(lo, [None if np.isinf(h) else h for h in hi])),
                      method="highs")
        if not res.success:
            raise RuntimeError(f"Engine {engine.name!r} infeasible: {res.message}")
        x = res.x
        lamhat = {a: float(res.eqlin.marginals[a_idx[a]]) for a in names}
        m = res.ineqlin.marginals if A_ub else np.array([])
        fg_dual = {cids[l]: 0.0 for l in range(len(cids))}
        cong = np.zeros(pt.n_bus)
        for r, l in enumerate(act):
            mu = float(m[2 * r] - m[2 * r + 1])
            fg_dual[cids[l]] = mu
            cong += Hmat[l] * mu
        grp_dual = [abs(float(m[2 * len(act) + k2])) for k2 in range(len(grp_rows))]
        # HiGHS marginal = ∂objective/∂rhs ≤ 0 on a binding ≤ row, so the ISL
        # dual is its negation, nonnegative in $/MWh (+ 0.0 clears a -0.0).
        r_isl = 2 * len(act) + len(grp_rows)
        isl_dual = {s_: -float(m[r_isl + k2]) + 0.0
                    for k2, s_ in enumerate(isl_ids)}
    else:
        # ---- rising-MC QP (trust-constr), duals from KKT stationarity ------
        H = np.zeros((N, N))
        H[np.arange(G), np.arange(G)] = b_c
        c_lin = np.concatenate([a_c, np.full(K, float(shed_price)) if K else np.zeros(0),  # type: ignore[arg-type]  # K > 0 only with shed_price set
                                np.zeros(2 * P)])
        cons = [LinearConstraint(A_bal, b_bal, b_bal)]
        if act:
            cons.append(LinearConstraint(
                flow_rows,
                np.array([-rev[l] - Ffix[l] - offsets.get(cids[l], 0.0)
                          for l in act]),
                np.array([fwd[l] - Ffix[l] - offsets.get(cids[l], 0.0)
                          for l in act])))
        x0 = np.minimum(np.where(np.isinf(hi), load_vec.sum(), hi),
                        np.full(N, load_vec.sum() / max(N, 1)))
        r = minimize(lambda z: 0.5 * z @ (H @ z) + c_lin @ z, x0,
                     method="trust-constr", jac=lambda z: H @ z + c_lin,
                     hess=lambda z: H, constraints=cons,
                     bounds=Bounds(lo, hi),
                     options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 3000})
        x = r.x
        binding = [l for rr, l in enumerate(act)
                   if flow_rows[rr] @ x + Ffix[l] + offsets.get(cids[l], 0.0)
                   > fwd[l] - 1e-4
                   or flow_rows[rr] @ x + Ffix[l] + offsets.get(cids[l], 0.0)
                   < -(rev[l] - 1e-4)]
        # KKT unknowns: [λ̂_a … , μ_l for binding lines]; rows from interior
        # generators/shed (marginal cost pins its area's price + congestion) and
        # interior transfers (a slack interface with flow ties its two λ̂ together).
        rows, rhs = [], []
        def _row(area: Area, bus: Bus | None = None) -> list[float]:
            rc = [0.0] * A + ([Hmat[l, pt.bus_idx[bus]] for l in binding] if bus else [0.0] * len(binding))
            rc[a_idx[area]] = 1.0
            return rc
        for i in range(G):
            if tol < x[i] < cap[i] - tol:
                rows.append(_row(area_of[gen_bus[i]], gen_bus[i]))
                rhs.append(a_c[i] + b_c[i] * x[i])
        for j in range(K):
            if tol < x[G + j] < shed_cap[j] - tol:
                rows.append(_row(area_of[shed_bus[j]], shed_bus[j]))
                rhs.append(float(shed_price))  # type: ignore[arg-type]  # K > 0 only with shed_price set
        for p, (xx, yy) in enumerate(ifc):
            for col, (fr, to) in ((G + K + 2 * p, (xx, yy)), (G + K + 2 * p + 1, (yy, xx))):
                ub = hi[col]
                if x[col] > tol and (np.isinf(ub) or x[col] < ub - tol):
                    rc = [0.0] * (A + len(binding))
                    rc[a_idx[fr]] = 1.0
                    rc[a_idx[to]] = -1.0
                    rows.append(rc)                       # interior transfer: λ̂_fr = λ̂_to
                    rhs.append(0.0)
        if rows:
            sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        else:
            on = [i for i in range(G) if x[i] > tol]
            flat = float(max(a_c[i] + b_c[i] * x[i] for i in on)) if on else float(np.min(a_c))
            sol = np.array([flat] * A + [0.0] * len(binding))
        lamhat = {a: float(sol[a_idx[a]]) for a in names}
        fg_dual = {cids[l]: 0.0 for l in range(len(cids))}
        cong = np.zeros(pt.n_bus)
        for k2, l in enumerate(binding):
            mu = float(sol[A + k2])
            fg_dual[cids[l]] = mu
            cong += Hmat[l] * mu
        grp_dual = []
        isl_dual = {}                                 # schedule_limits is LP only

    g_opt, u_opt = x[:G], x[G:G + K]
    t_link = x[G + K + 2 * P:G + K + 2 * P + Lk]
    d_opt = x[C_D:C_D + D] if D else np.zeros(0)
    s_viol = x[C_D + D:C_D + D + S_v] if S_v else np.zeros(0)

    # ── Stage two: the tags — minimal total scheduled MW at this dispatch ──
    # An area's INTERFACE-borne position is its generation surplus plus what
    # the physical DC links already deliver into (or draw out of) it.
    link_net = {a: 0.0 for a in names}
    for q, (b0, b1) in enumerate(zip(link_b0, link_b1)):
        link_net[area_of[b0]] -= float(t_link[q])
        link_net[area_of[b1]] += float(t_link[q])
    ex = {a: float(sum(g_opt[i] for i in range(G) if area_of[gen_bus[i]] == a)
                   + sum(u_opt[j] for j in range(K) if area_of[shed_bus[j]] == a)
                   - sum(d_opt[j] for j in range(D) if area_of[dem_bus[j]] == a)
                   + link_net[a]
                   - b_bal[a_idx[a]]) for a in names}     # net export of each area
    transfers = {k: 0.0 for k in ifc}
    if P:
        A_t = A_bal[:A - 1, G + K:G + K + 2 * P]          # drop one redundant balance row
        b_t = np.array([-ex[a] for a in names[:A - 1]])   # −Σ_out t + Σ_in t = −e_a^out
        t_ub2 = t_ub.copy()                               # tags may carry awarded violations
        for k2, col_ in enumerate(vrl_cols):
            t_ub2[col_ - (G + K)] += float(s_viol[k2])
        t_lo2 = np.zeros(2 * P)
        for c_, v_ in fixed_idx.items():                  # pinned tags stay pinned
            t_lo2[c_ - (G + K)] = v_
            t_ub2[c_ - (G + K)] = v_
        r2 = linprog(np.ones(2 * P), A_eq=A_t, b_eq=b_t,
                     bounds=[(l_, None if np.isinf(u) else u)
                             for l_, u in zip(t_lo2, t_ub2)],
                     method="highs")
        if not r2.success:
            raise RuntimeError(f"transfer tagging stage infeasible: {r2.message}")
        for p, k in enumerate(ifc):
            transfers[k] = float(r2.x[2 * p] - r2.x[2 * p + 1])
            if abs(transfers[k]) < 1e-9:
                transfers[k] = 0.0

    transfer_violations: dict[Interface, float] = {}
    for k2, col_ in enumerate(vrl_cols):
        if s_viol[k2] > 1e-9:
            p_ = (col_ - (G + K)) // 2
            sgn = 1.0 if (col_ - (G + K)) % 2 == 0 else -1.0
            k_ = ifc[p_]
            transfer_violations[k_] = (transfer_violations.get(k_, 0.0)
                                       + sgn * float(s_viol[k2]))

    # Signed interface duals: λ̂_b − λ̂_a wherever the pair's limit binds.
    transfer_duals = {}
    for p, (xx, yy) in enumerate(ifc):
        binds = (lims[p] is not None
                 and (abs(transfers[(xx, yy)]) > lims[p] - 1e-4  # type: ignore[operator]  # the test above covers this index
                      or abs(x[G + K + 2 * p]) > lims[p] - 1e-4  # type: ignore[operator]  # the test above covers this index
                      or abs(x[G + K + 2 * p + 1]) > lims[p] - 1e-4))  # type: ignore[operator]  # the test above covers this index
        transfer_duals[(xx, yy)] = float(lamhat[yy] - lamhat[xx]) if binds else 0.0

    lmp = {bb: lamhat[area_of[bb]] + float(cong[pt.bus_idx[bb]]) if bb in area_of
           else float("nan") for bb in pt.buses}
    # buses outside the declared areas (none, per the cover check, unless the
    # partition covers only the engine's buses on a larger network): price them
    # off the slack area's λ̂ plus congestion so the dict stays total.
    for bb in pt.buses:
        if bb not in area_of:
            lmp[bb] = lamhat[names[0]] + float(cong[pt.bus_idx[bb]])

    gen_by_bus: dict[str, float] = {}
    for i, gid in enumerate(gen_ids):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + float(g_opt[i])
    inj = np.zeros(pt.n_bus)
    for bb, mw in gen_by_bus.items():
        inj[pt.bus_idx[bb]] += mw
    for j, sb in enumerate(shed_bus):
        inj[pt.bus_idx[sb]] += u_opt[j]
    for j, dbus in enumerate(dem_bus):                 # cleared bids withdraw
        inj[pt.bus_idx[dbus]] -= d_opt[j]
    inj += exo_vec - load_vec
    for b0, b1, t in zip(link_b0, link_b1, t_link):    # DC-link transfers
        inj[pt.bus_idx[b0]] -= t
        inj[pt.bus_idx[b1]] += t
    flow_own = {pt.lines[l]: float(pt.ptdf[l] @ inj) for l in range(pt.n_line)}

    if flowgates is None:
        line_dual = dict(fg_dual)                      # cids ARE the lines
    else:
        impl = flowgates.implied_line_duals(fg_dual)   # μ^impl = Sᵀ μ_fg
        line_dual = {pt.lines[i]: float(impl[i]) for i in range(pt.n_line)}

    return EngineResult(
        name=engine.name,
        dispatch={gid: float(g_opt[i]) for i, gid in enumerate(gen_ids)},
        gen_by_bus=gen_by_bus,
        load_by_bus={str(bb): float(v) for bb, v in engine.loads.items()},
        exo_by_bus=dict(exo),
        injection=inj,
        lmp=lmp,
        energy_price=float(lmp[pt.slack_bus]),
        line_dual=line_dual,
        flow_own=flow_own,
        total_cost=float(np.sum(a_c * g_opt + 0.5 * b_c * g_opt ** 2)),
        status="optimal (two-stage: dispatch, then min-total-schedule tags)",
        shed_by_bus={sb: float(uu) for sb, uu in zip(shed_bus, u_opt) if uu > 1e-9},
        demand_cleared={i: float(q) for i, q in zip(dem_ids, d_opt) if q > 1e-9},
        demand_by_bus=_demand_by_bus(dem_bus, d_opt),
        link_flow={lid: float(t) for lid, t in zip(link_ids, t_link)},
        area_prices=lamhat,
        transfers=transfers,
        transfer_duals=transfer_duals,
        transfer_limits={k: lims[p] for p, k in enumerate(ifc)},
        transfer_violations=transfer_violations,
        group_cap_dual=grp_dual,
        schedule_limit_dual=isl_dual,
        schedule_limit_mw={s_: float(sum(x[c_] for c_ in cols_))
                           for s_, cols_ in zip(isl_ids, isl_cols, strict=True)},
        fg_dual=fg_dual,
        fg_flow={cids[k]: float(Hmat[k] @ inj) for k in range(len(cids))},
        fg_limit={cids[k]: (float(fwd[k]), float(rev[k]))
                  for k in range(len(cids))},
        fg_members=({} if flowgates is None else
                    {k: list(flowgates.members.get(k, [])) for k in cids}),
    )


# ──────────────────────────────────────────────────────────────────────────
# Superposition of engine flows  (eq. 30):  F^phys = Σ_M F^M + F^non
# ──────────────────────────────────────────────────────────────────────────
def actual_injection(pt: PTDFData, results: list[EngineResult]) -> np.ndarray:
    """The injection vector of what the engines actually dispatched and served.

    Each engine's ``injection`` carries its cleared generation, less the load
    it served (its load net of any shed), less any price-sensitive demand bid
    it cleared, plus its DC-link transfers, plus the exogenous schedule ``exo``
    it was handed. This strips the ``exo`` back out and leaves the rest, so the
    vector is what the engines physically dispatched and served, and a schedule
    whose two legs are realized in two engines' books is counted once rather
    than twice.
    """
    p = np.zeros(pt.n_bus)
    for r in results:
        p += r.injection
        for bus, mw in r.exo_by_bus.items():
            p[pt.bus_idx[str(bus)]] -= float(mw)
    return p


def physical_flows(
    pt: PTDFData,
    results: list[EngineResult],
    non_market: BusValues | None = None,
    *,
    exo_physical: bool = True,
    flowgates: FlowgateSet | None = None,
) -> pd.DataFrame:
    """Combine engine flow components into the physical flow on every line.

    ``flowgates`` (added 2026-10-21, plan item B2) rates the superposition
    against the case's constraint set. When a ``FlowgateSet`` is given, the
    table is indexed by flowgate id, and each flow column is the flowgate
    flow ``f_k = Σ_m S[k,m] F_m`` of the per-line column it replaces. The
    ``rating`` column is the rating in the direction the physical flow
    loads the flowgate, ``fwd_k`` for a non-negative flow and ``rev_k``
    otherwise, and ``overload`` tests ``f_k > fwd_k`` or ``f_k < −rev_k``.
    The ``members`` column names the member lines with their signs, in
    place of the ``from`` and ``to`` buses of a line. The default ``None``
    rates every line against ``pt.s_nom``, as before.

    Returns a DataFrame indexed by line with each engine's own component, the
    non-market residual, the physical total, the rating, and an overload flag.
    The physical flow is computed from the *summed* injection vector, which is
    identical to summing the per-engine flow components (DC superposition).

    ``exo_physical`` states what each engine's exogenous schedule ``exo``
    stands for, and the two readings give different physical flows:

    1. ``True`` (the default) reads every ``exo`` as a physical injection or
       withdrawal that no engine in ``results`` dispatches or serves, so it
       belongs in the sum. A boundary balancing authority's scheduled
       interchange, placed on buses outside every engine's footprint, is that
       case.
    2. ``False`` reads every ``exo`` as the counterparty engine's leg of a
       schedule whose other leg is already in that engine's own book. A
       balanced schedule from a source bus to a sink bus is realized as
       generation in the source engine's book and as load in the sink
       engine's book, so summing the scheduled injections as well would count
       the schedule twice. The superposition is then each engine's cleared
       generation less its served load and nothing else, which is the rule
       :func:`actual_flows` states, and the bus balance of the summed vector
       is checked before the flow is formed.

    The per-engine column ``F[name]`` follows the same reading, so the engine
    columns always sum to the physical total. Under ``exo_physical=True`` it is
    the engine's book flow ``flow_own``, which carries the engine's own
    exogenous schedule. Under ``exo_physical=False`` it is the engine's actual
    flow contribution, its cleared generation less its served load, and the
    difference between the two is the flow the engine's own scheduled leg
    places where the engine models it.

    Raises ``ValueError`` under ``exo_physical=False`` when the summed
    injection does not balance, because an unbalanced sum says at least one
    ``exo`` is not a counterparty's leg and the default reading is the one
    that applies.
    """
    non = np.zeros(pt.n_bus)
    if non_market:
        for b, mw in non_market.items():
            non[pt.bus_idx[str(b)]] += mw
    f_non = pt.ptdf @ non

    cols = {}
    if exo_physical:
        p_total = np.zeros(pt.n_bus)
        for r in results:
            p_total += r.injection
            cols[r.name] = [r.flow_own[l] for l in pt.lines]
    else:
        p_total = actual_injection(pt, results)
        # The engines' own generation and served load balance the non-market
        # residual and nothing else, because every exogenous schedule in this
        # reading is the counterparty engine's leg of a schedule that engine
        # books. A sum that does not balance says one of them is not.
        resid = float(p_total.sum() + non.sum())
        scale = max(1.0, float(np.abs(p_total).sum()))
        if abs(resid) > 1e-6 * scale:
            raise ValueError(
                "physical_flows(exo_physical=False): the engines' generation "
                f"less their served load, plus the non-market residual, is "
                f"{resid:+.4f} MW out of balance, so at least one exogenous "
                "schedule is not the counterparty engine's leg of a balanced "
                "schedule. Use exo_physical=True for a schedule that no engine "
                "in this set dispatches or serves."
            )
        for r in results:
            if not r.exo_by_bus:
                # nothing to strip, so the book flow IS the actual contribution
                cols[r.name] = [r.flow_own[l] for l in pt.lines]
                continue
            e = np.zeros(pt.n_bus)
            for bus, mw in r.exo_by_bus.items():
                e[pt.bus_idx[str(bus)]] += float(mw)
            f_exo = pt.ptdf @ e
            cols[r.name] = [r.flow_own[l] - float(f_exo[pt.line_idx[l]])
                            for l in pt.lines]
    if non_market:
        p_total = p_total + non

    f_phys = pt.ptdf @ p_total

    if flowgates is not None:
        S = np.asarray(flowgates.S)
        fwd = np.asarray(flowgates.fwd, dtype=float)
        rev = np.asarray(flowgates.rev, dtype=float)
        g_phys = S @ f_phys
        df = pd.DataFrame(index=list(flowgates.ids))
        df.index.name = "flowgate"
        df["members"] = [", ".join(f"{'+' if s >= 0 else '-'}{m}"
                                   for m, s in flowgates.members.get(k, []))
                         for k in flowgates.ids]
        for name, vals in cols.items():
            df[f"F[{name}]"] = np.round(S @ np.asarray(vals, dtype=float), 1)
        if non_market:
            df["F[non-market]"] = np.round(S @ f_non, 1)
        df["F_phys"] = np.round(g_phys, 1)
        cap = np.where(g_phys >= 0, fwd, rev)
        df["rating"] = cap
        df["loading_%"] = np.round(100 * np.abs(g_phys) / cap, 0)
        df["overload"] = (g_phys > fwd + 1e-6) | (g_phys < -rev - 1e-6)
        return df

    df = pd.DataFrame(index=pt.lines)
    df.index.name = "line"
    df["from"] = [b0 for b0, _ in pt.line_buses]
    df["to"] = [b1 for _, b1 in pt.line_buses]
    for name, vals in cols.items():
        df[f"F[{name}]"] = np.round(vals, 1)
    if non_market:
        df["F[non-market]"] = np.round(f_non, 1)
    df["F_phys"] = np.round(f_phys, 1)
    df["rating"] = pt.s_nom
    df["loading_%"] = np.round(100 * np.abs(f_phys) / pt.s_nom, 0)
    df["overload"] = np.abs(f_phys) > pt.s_nom + 1e-6
    return df


#: Display default for :func:`to_supply_demand`'s ``show_backstops``. ``False`` (the
#: series default) hides an idle backstop and draws a dispatched one at its cleared
#: megawatts. ``True`` draws every backstop like any other generator, at full
#: capacity with the idle part faded; 304a sets it, because its backstops are small,
#: priced inside the figure's price axis, and the resource the relief examples use.
SHOW_IDLE_BACKSTOPS = False


def to_supply_demand(
    engine: MarketEngine, result: EngineResult, show_backstops: bool | None = None
) -> tuple[dict[Bus, list[dict[str, Any]]], Loads]:
    """Adapt an engine + result to ``nodal_plot`` ``supply_by_bus``/``demand_by_bus``.

    Lets the engine results be drawn with the existing ``plot_network_topology``
    / ``plot_nodal_circlize`` helpers (merit-order staircase + LMP-height loads).

    ``show_backstops`` (default: the module's :data:`SHOW_IDLE_BACKSTOPS`) chooses how
    a ``backstop_*`` unit is drawn. A display choice only; no number changes.
    """
    full = SHOW_IDLE_BACKSTOPS if show_backstops is None else bool(show_backstops)
    supply_by_bus: dict[Bus, list[dict[str, Any]]] = {}
    for gid, spec in engine.gens.items():
        bus = str(spec["bus"])
        acc = result.dispatch.get(gid, 0.0)
        # A residual BACKSTOP is a balancing resource, not part of the economic merit order:
        # by default hide it when idle, and draw only the block it actually balances when it
        # clears (so its large capacity does not stretch the merit-order axis at every load bus).
        # With ``full`` it is drawn like any other unit, at its capacity.
        is_backstop = gid.startswith("backstop_") and not full
        if is_backstop and acc <= 1e-6:
            continue
        vol = float(acc) if is_backstop else spec["p_nom"]
        # A unit may carry a linear marginal-cost curve ``curve=(a, b)`` (MC = a + b·g,
        # set by the rising-curve QP clearing). Then its merit-order "price" is the
        # marginal offer AT its cleared output (a + b·acc), and the curve params ride
        # along as ``mc0``/``mc_slope`` so ``plot_nodal_circlize`` can draw the supply
        # as a sloped WEDGE under the MC line instead of a flat rectangle.
        curve = spec.get("curve")
        entry = {
            "unit_id": gid,
            "price": spec["cost"] if curve is None else float(curve[0] + curve[1] * acc),
            "volume": vol,
            "capacity": vol,
            "accepted_volume": acc,
        }
        if curve is not None:
            entry["mc0"], entry["mc_slope"] = float(curve[0]), float(curve[1])
        if spec.get("hatch"):
            entry["hatch"] = spec["hatch"]
        supply_by_bus.setdefault(bus, []).append(entry)
    for bus in supply_by_bus:
        supply_by_bus[bus].sort(key=lambda g: g["price"])
    demand_by_bus = {str(b): float(v) for b, v in engine.loads.items() if v > 0}
    return supply_by_bus, demand_by_bus


def shed_segments(
    result: EngineResult, demand_by_bus: Loads, shed_alpha: float = 0.12
) -> dict[Bus, list[dict[str, float]]]:
    """A ``nodal_plot`` ``demand_segments`` dict that renders shed (unserved) load
    as a faint tail: each shed bus's load bar splits into a served segment (bus
    color at the demand fill) plus an unserved segment in the SAME bus color at
    the fainter ``shed_alpha`` — the convention used for idle generation capacity.
    Returns ``{}`` when nothing sheds, so ``shed_segments(...) or None`` is a clean
    default.
    """
    segs: dict[Bus, list[dict[str, float]]] = {}
    for b, u in (result.shed_by_bus or {}).items():
        if u <= 1e-6:
            continue
        served = float(demand_by_bus.get(str(b), 0.0)) - u
        segs[str(b)] = ([{"mw": served}] if served > 0.5 else []) \
            + [{"mw": u, "alpha": shed_alpha}]   # bus color (inherited), faint
    return segs


def served_by_bus(result: EngineResult, demand_by_bus: Loads) -> BusValues:
    """``{bus: served MW}`` at the buses that shed, for ``plot_network_topology``'s
    ``demand_served_by_bus`` ('served/total' load annotation). Empty when nothing
    sheds, so the annotation falls back to the plain total elsewhere.
    """
    return {str(b): float(demand_by_bus.get(str(b), 0.0)) - u
            for b, u in (result.shed_by_bus or {}).items() if u > 1e-6}


def served_demand(result: EngineResult, demand_by_bus: Loads) -> BusValues:
    """Full ``{bus: served MW}`` = demand − shed, for tracing gen→load chords on
    the SERVED dispatch. Passing this (not nominal demand) to
    ``compute_ptdf_flows`` keeps the trace balanced (Σ gen = Σ served), so each
    generator's chords match its dispatch and a shed bus receives only its
    served-MW of incoming chords (the unserved tail stays chord-free).
    """
    shed = result.shed_by_bus or {}
    return {str(b): float(v) - shed.get(str(b), 0.0) for b, v in demand_by_bus.items()}


def seam_dual_gap(results: list[EngineResult], buses: list[Bus]) -> pd.DataFrame:
    """Pairwise seam dual gap Δλ^{M,M'}_n at each bus (Proposition 1 / eq. 18)."""
    df = pd.DataFrame(index=buses)
    df.index.name = "bus"
    for r in results:
        df[f"λ[{r.name}]"] = [round(r.lmp[b], 2) for b in buses]
    names = [r.name for r in results]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df[f"Δλ[{a}→{b}]"] = (df[f"λ[{b}]"] - df[f"λ[{a}]"]).round(2)
    return df


# ──────────────────────────────────────────────────────────────────────────
# The CASE: the injectable scenario layer
# ──────────────────────────────────────────────────────────────────────────
# Everything a clearing needs that is not the algebra — the network builder,
# the fleet, the loads, the drawing layout, the risk drivers — lives in a
# *case* object. ``wscc9_model.CASE`` is one implementation (the 9-bus teaching
# case); the western-seams-lab repo supplies another (the real WECC map-mirror).
# The shared libraries resolve their defaults through ``active()`` instead of
# importing ``wscc9_model`` directly, which is what lets one copy of the
# library serve both series.
#
# This registry lives HERE, not in a new module, because every published
# notebook's Colab bootstrap fetches a fixed list of ten flat module files by
# name — adding an eleventh would break all of them, while ``seams_engine`` is
# already on the list and imports no case at module scope.

@dataclass
class CaseNetwork:
    """One built network with its shift factors (and, later, its flowgates).

    ``constraint_set`` names the kind of constraint the case rates, and it
    follows the rule the solvers apply: ``"lines"`` when ``fg`` is ``None``
    (the solvers take the per-line branch over ``pt.s_nom``), ``"flowgates"``
    whenever ``fg`` is set (the solvers take the flowgate branch over
    ``fg.ids``). A ``FlowgateSet`` with no ids therefore classifies as
    ``"flowgates"`` and clears with no activated constraint at all, which is
    what the solvers do with it. :func:`check_case` rejects a ``build()``
    that returns such a set, because a case that rates nothing is not a
    conformant case. The property reads ``fg`` and changes nothing else
    (2026-10-18).
    """

    network: pypsa.Network | SimpleNamespace   # pypsa.Network
    pt: PTDFData
    fg: FlowgateSet | None = None      # FlowgateSet (phase 4); None = per-line

    @property
    def constraint_set(self) -> str:
        if self.fg is None:
            return CONSTRAINT_LINES
        return CONSTRAINT_FLOWGATES


#: The two values ``CaseNetwork.constraint_set`` takes.
CONSTRAINT_LINES = "lines"
CONSTRAINT_FLOWGATES = "flowgates"


class Case:
    """The case contract (duck-typed; ``wscc9_model.WSCC9Case`` is the reference).

    The contract is stated as a ``typing.Protocol`` in :class:`CaseProtocol`
    (with the data-backed extension in :class:`DataBackedCase`) and checked by
    :func:`check_case`; this docstring is the narrative form.

    Attributes / properties
        name         : str            — display name ("WSCC-9", "WECC-map-1883")
        slack_bus    : str            — the PTDF slack bus of this case
        gen_fleet    : dict           — {gid: {"bus", "cost", "p_nom", ...}}
        loads        : dict           — {bus: MW}
        shed_price   : float          — the case's V (load-shed penalty)
        bus_to_area  : dict           — {bus: BA}; {} where BAs come from Footprints
        bus_colors, coords, ring_order, center_bus, rotation_deg — drawing layout
        risk_load_sd, risk_cost_sd    — the case's driver spreads (risk.py) (optional)

    Methods
        build(line_ratings=None, **variant) -> CaseNetwork
            Fresh network each call (notebooks mutate engines/networks in
            place); only the PTDF may be cached. Unknown ``variant`` kwargs
            must RAISE, never be silently ignored.
        make_engine(name, buses, gen_fleet=None, loads=None, activated="all")
        hub_kind() -> "auto" | "gap" | "elap"
        hub_weights(kind=None, fp=None, *, loads=None, gen_fleet=None) -> dict
            {area: {bus: w}}, the case's ex-ante aggregation-point weights.
        drivers(kinds=("load", "cost")) -> dict
            The case's canonical uncertain drivers ({key: dist} for risk.py).
        historical_days() -> dict                              (optional)
            {day_id: {driver_key: 24-array | scalar}} — feeds
            risk.day_bootstrap on a bundle-backed case.
    """


_ACTIVE: CaseProtocol | None = None    # the ambient case, set by set_case()

#: Lazy default: the 9-bus teaching case. Import happens on first ``active()``
#: call, never at module import, so this module works when wscc9_model is absent.
_DEFAULT_CASE = ("wscc9_model", "CASE")


def set_case(case: CaseProtocol | None) -> CaseProtocol | None:
    """Install ``case`` as the ambient case; returns the previous one."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, case
    return prev


def active() -> CaseProtocol:
    """The ambient case, lazily defaulting to the 9-bus teaching case.

    Library functions call this ONLY to fill in defaults the caller did not
    supply (fleet, loads, slack bus, layout). A caller that passes everything
    explicitly never touches the registry.
    """
    global _ACTIVE
    if _ACTIVE is None:
        import importlib

        mod, attr = _DEFAULT_CASE
        _ACTIVE = getattr(importlib.import_module(mod), attr)
    return _ACTIVE


class using:
    """Context manager: run a block under a different ambient case.

    >>> with seams_engine.using(my_case):
    ...     ra.allocate_congestion_rent(...)   # resolves defaults from my_case
    """

    def __init__(self, case: CaseProtocol) -> None:
        self.case = case

    def __enter__(self) -> CaseProtocol:
        self.prev = set_case(self.case)
        return self.case

    def __exit__(self, *exc: object) -> Literal[False]:
        set_case(self.prev)
        return False


# ──────────────────────────────────────────────────────────────────────────
# The Case contract as a Protocol, with its conformance check (2026-10-18)
# ──────────────────────────────────────────────────────────────────────────
# One copy of the shared libraries serves every case: the 9-bus teaching case
# (``wscc9_model.WSCC9Case``), the WECC map-mirror (``wecc_case.WECCCase``),
# and any case built after them. The surface the libraries read is stated
# here once, as two ``runtime_checkable`` Protocols, and ``check_case``
# verifies an instance against it. Nothing below changes a default, a
# signature, or a clearing.

#: Keys every unit of a ``gen_fleet`` carries. ``bus`` is the host bus name,
#: ``cost`` the offer in $/MWh, ``p_nom`` the available capacity in MW. The
#: solvers read all three (``solve_engine_dispatch``, ``solve_engine_qp``,
#: ``solve_engine_transfers``), as do the settlement and figure helpers.
FLEET_REQUIRED = ("bus", "cost", "p_nom")

#: Keys a unit may carry, with the consumer of each:
#: ``carrier`` (fuel or technology; ``risk.py`` factor drivers, the WECC
#: fleet), ``ba`` (the unit's own balancing authority where it differs from
#: its bus's; ``risk._gen_ba``, the WECC fleet), ``p_min`` (a dispatch floor
#: in MW; the completed storage formulation writes it and
#: ``wecc_nonmarket`` reads it into ``gen_min``), ``curve`` (a linear
#: marginal-cost pair ``(a, b)``; ``to_supply_demand`` and the QP figures),
#: ``hatch`` (a matplotlib hatch for the unit's accepted block;
#: ``to_supply_demand``), ``gas_basis`` (the ``(day, basis)`` pair of
#: strings ``wecc_lookahead.reprice_gas_fleet`` writes onto a repriced
#: natural-gas unit and reads back, so that a fleet the case loader already
#: repriced under ``WECCCase.gas_basis`` is not repriced again at the same
#: basis; every natural-gas unit of the WECC fleet carries it under the
#: shipped ``"daily"`` basis, and no unit carries it under ``gas_basis =
#: None``). A key outside both tuples is a schema violation, because no
#: shared consumer reads it.
FLEET_OPTIONAL = ("carrier", "ba", "p_min", "curve", "hatch", "gas_basis")

#: The values ``hub_kind()`` may return.
HUB_KINDS = ("auto", "gap", "elap")


@runtime_checkable
class CaseProtocol(Protocol):
    """The surface every case exposes to the shared libraries.

    Data members (attributes or properties; a property returns a copy):

    ``name``
        The display name, a string.
    ``slack_bus``
        The PTDF slack bus, a bus name present in ``build().pt.buses``.
    ``gen_fleet``
        ``{unit_id: spec}`` under the fleet schema (:data:`FLEET_REQUIRED`,
        :data:`FLEET_OPTIONAL`).
    ``loads``
        ``{bus: MW}``.
    ``shed_price``
        The case's load-shed penalty in $/MWh (the canon's $V$).
    ``bus_to_area``
        ``{bus: area}``; ``{}`` where the areas come from a Footprints
        partition, as on the 9-bus case.
    ``bus_colors``, ``coords``, ``ring_order``, ``center_bus``, ``rotation_deg``
        The drawing layout: ``{bus: color}``, ``{bus: (x, y)}``, the chord
        sector order, the bus placed across 12 o'clock (``None`` for no
        centering), and the topology rotation in degrees.

    Methods:

    ``build(line_ratings=None, **variant) -> CaseNetwork``
        A fresh network each call; only the PTDF may be cached. An unknown
        ``variant`` keyword raises.
    ``make_engine(name, buses, gen_fleet=None, loads=None, activated="all")``
        A ``MarketEngine`` owning the fleet units and loads at ``buses``.
    ``hub_kind() -> str``
        One of :data:`HUB_KINDS`.
    ``hub_weights(kind=None, fp=None, *, loads=None, gen_fleet=None) -> dict``
        The case's ex-ante aggregation-point weights ``{area: {bus: w}}``,
        each area's weights summing to one over its own buses. ``kind``
        defaults to ``hub_kind()``. ``fp`` is a ``Footprints`` partition (or
        any object carrying ``names`` and ``defs``) that names the areas. A
        case whose ``bus_to_area`` is empty takes its areas from ``fp``
        alone and raises ``ValueError`` without one. ``loads`` and
        ``gen_fleet`` default to the case's own. Added 2026-10-21 (plan
        item B3), so that every caller that weights a hub for the active
        case reads one method, and a new case adds its basis there rather
        than a further definition.
    ``drivers(kinds=("load", "cost")) -> dict``
        The case's uncertain drivers as ``{key: dist}`` for ``risk.py``.

    Decisions recorded against the plan's list (architecture plan
    2026-10-18, item B1). ``hub_kind`` and ``drivers`` are methods, because
    both existing cases implement them as methods. ``center_bus`` and
    ``rotation_deg`` are kept: the 9-bus case already exposes them as class
    attributes aliasing ``wscc9_model.CENTER_BUS`` and ``ROTATION_DEG``, and
    the WECC case exposes ``None`` and ``0.0``. ``risk_load_sd``,
    ``risk_cost_sd`` and ``historical_days()`` are optional and stay outside
    the Protocol, because ``risk.py`` reads them through ``getattr`` with a
    default and ``hasattr`` respectively.
    """

    name: str
    slack_bus: str
    gen_fleet: Fleet
    loads: Loads
    shed_price: float
    bus_to_area: dict[Bus, Area]
    bus_colors: dict[Bus, str]
    coords: dict[Bus, tuple[float, float]]
    ring_order: list[Bus]
    center_bus: str | None
    rotation_deg: float

    def build(
        self, line_ratings: LineValues | None = None, **variant: object
    ) -> CaseNetwork: ...

    def make_engine(self, name: str, buses: Iterable[Bus],
                    gen_fleet: Fleet | None = None, loads: Loads | None = None,
                    activated: str | list[Line] = "all") -> MarketEngine: ...

    def hub_kind(self) -> str: ...

    def hub_weights(self, kind: str | None = None,
                    fp: Footprints | SimpleNamespace | None = None, *,
                    loads: Loads | None = None,
                    gen_fleet: Fleet | None = None) -> dict[Area, BusValues]: ...

    def drivers(
        self, kinds: tuple[str, ...] = ("load", "cost")
    ) -> dict[tuple[str, ...], Normal | LogitNormal]: ...


@runtime_checkable
class DataBackedCase(Protocol):
    """The extension a case pinned to a data bundle exposes, in addition to
    :class:`CaseProtocol`. ``wecc_case.WECCCase`` is the reference.

    ``bundle``
        The frozen data bundle the case reads.
    ``day``, ``hour``
        The pinned interval: a day identifier string and an integer hour.
    ``at(day, hour)``
        A sibling case pinned to another interval of the same bundle.
    ``dc_links(buses=None) -> dict``
        ``{link_id: {"bus0", "bus1", "p_max", "p_min"}}`` in the form the
        solvers take; ``p_min`` is optional and defaults to 0 there.
    ``nameplate_fleet() -> dict``
        The fleet at nameplate capacity under the fleet schema.
    ``posture_line() -> str``
        One line naming the input posture the case prices under.
    """

    bundle: object
    day: str
    hour: int

    def at(self, day: str, hour: int) -> DataBackedCase: ...

    def dc_links(
        self, buses: Iterable[Bus] | None = None
    ) -> dict[str, dict[str, Any]]: ...

    def nameplate_fleet(self) -> Fleet: ...

    def posture_line(self) -> str: ...


def check_fleet(fleet: object, label: str = "gen_fleet") -> list[str]:
    """Violations of the fleet schema in ``fleet``; empty when conformant.

    Each unit is a dict carrying every key in :data:`FLEET_REQUIRED` and no
    key outside :data:`FLEET_REQUIRED` + :data:`FLEET_OPTIONAL`. ``cost`` and
    ``p_nom`` are finite numbers, ``p_nom`` is non-negative, ``p_min`` (when
    present) is a number no greater than ``p_nom``, and ``gas_basis`` (when
    present) is a ``(day, basis)`` pair of strings.
    """
    out: list[str] = []
    if not isinstance(fleet, dict):
        return [f"{label} is {type(fleet).__name__}, not dict"]
    known = set(FLEET_REQUIRED) | set(FLEET_OPTIONAL)
    for gid, spec in fleet.items():
        if not isinstance(spec, dict):
            out.append(f"{label}[{gid!r}] is {type(spec).__name__}, not dict")
            continue
        missing = [k for k in FLEET_REQUIRED if k not in spec]
        if missing:
            out.append(f"{label}[{gid!r}] lacks required key(s) {missing}")
        unknown = sorted(set(spec) - known)
        if unknown:
            out.append(f"{label}[{gid!r}] carries unknown key(s) {unknown}")
        for k in ("cost", "p_nom", "p_min"):
            if k in spec and not _is_finite_number(spec[k]):
                out.append(f"{label}[{gid!r}][{k!r}] = {spec[k]!r} is not a finite number")
        if "p_nom" in spec and _is_finite_number(spec["p_nom"]) and float(spec["p_nom"]) < 0:
            out.append(f"{label}[{gid!r}]['p_nom'] = {spec['p_nom']!r} is negative")
        if ("p_min" in spec and "p_nom" in spec
                and _is_finite_number(spec["p_min"]) and _is_finite_number(spec["p_nom"])
                and float(spec["p_min"]) > float(spec["p_nom"]) + 1e-9):
            out.append(f"{label}[{gid!r}]['p_min'] exceeds its 'p_nom'")
        if "gas_basis" in spec:
            gb = spec["gas_basis"]
            if not (isinstance(gb, tuple) and len(gb) == 2
                    and all(isinstance(x, str) for x in gb)):
                out.append(f"{label}[{gid!r}]['gas_basis'] = {gb!r} is not a "
                           f"(day, basis) pair of strings")
    return out


def _is_finite_number(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x))) and not isinstance(x, bool)  # type: ignore[arg-type]  # the check takes any object and catches the TypeError
    except (TypeError, ValueError):
        return False


def _member(case: object, name: str, out: list[str]) -> tuple[bool, Any]:
    """``(present, value)`` for one member; a missing or raising member is
    recorded in ``out`` and reported as absent."""
    try:
        return True, getattr(case, name)
    except AttributeError:
        out.append(f"missing member {name!r}")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"member {name!r} raised {type(exc).__name__}: {exc}")
    return False, None


def check_case(case: CaseProtocol, build: bool = True) -> list[str]:
    """Violations of the Case contract by ``case``; empty when conformant.

    The check reads every :class:`CaseProtocol` member and verifies its
    type, runs :func:`check_fleet` over ``gen_fleet``, and, when ``build`` is
    true, calls ``build()`` and verifies that it returns a
    :class:`CaseNetwork` whose ``fg`` is either ``None`` (a line-rated
    network) or a :class:`FlowgateSet` with at least one flowgate id and an
    ``S`` matrix spanning the same lines as ``pt`` (an empty set clears
    with no activated constraint). The ``constraint_set`` property itself
    returns ``"lines"`` or ``"flowgates"`` by construction, so the check
    does not verify it. A case whose ``bus_to_area`` is non-empty has
    ``hub_weights()`` called with its defaults, and each area's weights are
    checked to sum to one over that area's own buses. The check then verifies that the slack bus, every
    fleet bus and every load bus are buses of the built network, that an
    unknown ``variant`` keyword raises, and that ``make_engine`` over the
    network's buses returns a :class:`MarketEngine` holding every fleet
    unit. A case that exposes ``bundle`` is checked against
    :class:`DataBackedCase` as well. The check clears nothing.
    """
    out: list[str] = []

    ok, name = _member(case, "name", out)
    if ok and not isinstance(name, str):
        out.append(f"name is {type(name).__name__}, not str")
    ok, slack = _member(case, "slack_bus", out)
    if ok and not isinstance(slack, str):
        out.append(f"slack_bus is {type(slack).__name__}, not str")
    ok, fleet = _member(case, "gen_fleet", out)
    if ok:
        out.extend(check_fleet(fleet))
        if isinstance(fleet, dict) and not fleet:
            out.append("gen_fleet is empty")
    ok, loads = _member(case, "loads", out)
    if ok:
        if not isinstance(loads, dict):
            out.append(f"loads is {type(loads).__name__}, not dict")
        else:
            bad = [b for b, v in loads.items() if not _is_finite_number(v)]
            if bad:
                out.append(f"loads carry non-numeric MW at {bad[:5]}")
    ok, shed = _member(case, "shed_price", out)
    if ok and not _is_finite_number(shed):
        out.append(f"shed_price = {shed!r} is not a finite number")
    for nm in ("bus_to_area", "bus_colors", "coords"):
        ok, v = _member(case, nm, out)
        if ok and not isinstance(v, dict):
            out.append(f"{nm} is {type(v).__name__}, not dict")
    ok, ring = _member(case, "ring_order", out)
    if ok and not isinstance(ring, (list, tuple)):
        out.append(f"ring_order is {type(ring).__name__}, not list")
    ok, cb = _member(case, "center_bus", out)
    if ok and cb is not None and not isinstance(cb, str):
        out.append(f"center_bus = {cb!r} is neither None nor str")
    ok, rot = _member(case, "rotation_deg", out)
    if ok and not _is_finite_number(rot):
        out.append(f"rotation_deg = {rot!r} is not a finite number")
    for nm in ("build", "make_engine", "hub_kind", "hub_weights", "drivers"):
        ok, fn = _member(case, nm, out)
        if ok and not callable(fn):
            out.append(f"{nm} is not callable")
    ok, hk = _member(case, "hub_kind", out)
    if ok and callable(hk):
        try:
            kind = hk()
            if kind not in HUB_KINDS:
                out.append(f"hub_kind() returned {kind!r}, not one of {HUB_KINDS}")
        except Exception as exc:                    # noqa: BLE001
            out.append(f"hub_kind() raised {type(exc).__name__}: {exc}")
    ok, hw = _member(case, "hub_weights", out)
    ok_b, b2a = _member(case, "bus_to_area", [])
    if ok and callable(hw) and ok_b and isinstance(b2a, dict) and b2a:
        # a case that names its own areas weights them without a partition
        try:
            W = hw()
            if not isinstance(W, dict) or not W:
                out.append("hub_weights() returned no area")
            else:
                for area, w in W.items():
                    if abs(sum(float(v) for v in w.values()) - 1.0) > 1e-6:
                        out.append(f"hub_weights()[{area!r}] does not sum to one")
                    off = [b for b in w if b2a.get(b) != area]
                    if off:
                        out.append(f"hub_weights()[{area!r}] weights buses of "
                                   f"another area: {off[:3]}")
        except Exception as exc:                    # noqa: BLE001
            out.append(f"hub_weights() raised {type(exc).__name__}: {exc}")

    if build and not out:
        try:
            cn: CaseNetwork | None = case.build()
        except Exception as exc:                    # noqa: BLE001
            out.append(f"build() raised {type(exc).__name__}: {exc}")
            cn = None
        if cn is not None:
            if not isinstance(cn, CaseNetwork):
                out.append(f"build() returned {type(cn).__name__}, not CaseNetwork")
            else:
                if cn.constraint_set not in (CONSTRAINT_LINES, CONSTRAINT_FLOWGATES):
                    out.append(f"constraint_set is {cn.constraint_set!r}")
                if cn.fg is not None:
                    if not isinstance(cn.fg, FlowgateSet):
                        out.append(f"build().fg is {type(cn.fg).__name__}, not FlowgateSet")
                    elif len(cn.fg.ids) == 0:
                        out.append("build().fg is a FlowgateSet with no flowgate; "
                                   "the clearing would activate no constraint")
                    elif cn.fg.S.shape[1] != cn.pt.n_line:
                        out.append(f"build().fg spans {cn.fg.S.shape[1]} lines and "
                                   f"pt spans {cn.pt.n_line}")
                buses = set(cn.pt.buses)
                if cn.pt.slack_bus != slack:
                    out.append(f"pt.slack_bus {cn.pt.slack_bus!r} differs from slack_bus {slack!r}")
                if slack not in buses:
                    out.append(f"slack_bus {slack!r} is not a bus of the built network")
                off = sorted({str(s["bus"]) for s in fleet.values()} - buses)
                if off:
                    out.append(f"fleet buses outside the network: {off[:5]}")
                off = sorted({str(b) for b in loads} - buses)
                if off:
                    out.append(f"load buses outside the network: {off[:5]}")
                try:
                    eng = case.make_engine("conformance", cn.pt.buses)
                    if not isinstance(eng, MarketEngine):
                        out.append(f"make_engine() returned {type(eng).__name__}, not MarketEngine")
                    elif set(eng.gens) != set(fleet):
                        out.append("make_engine() over every bus does not hold every fleet unit")
                except Exception as exc:            # noqa: BLE001
                    out.append(f"make_engine() raised {type(exc).__name__}: {exc}")
            try:
                case.build(_conformance_probe=True)
                out.append("build() accepted an unknown variant keyword without raising")
            except Exception:                       # noqa: BLE001
                pass

    if hasattr(case, "bundle"):
        out.extend(_check_data_backed(case))  # type: ignore[arg-type]  # the hasattr test selects a data-backed case
    return out


def _check_data_backed(case: DataBackedCase) -> list[str]:
    out: list[str] = []
    ok, day = _member(case, "day", out)
    if ok and not isinstance(day, str):
        out.append(f"day is {type(day).__name__}, not str")
    ok, hour = _member(case, "hour", out)
    if ok and not isinstance(hour, int):
        out.append(f"hour is {type(hour).__name__}, not int")
    for nm in ("at", "dc_links", "nameplate_fleet", "posture_line"):
        ok, fn = _member(case, nm, out)
        if ok and not callable(fn):
            out.append(f"{nm} is not callable")
    if out:
        return out
    try:
        sib = case.at(case.day, case.hour)
        if (sib.day, sib.hour) != (case.day, case.hour):
            out.append("at(day, hour) did not return a case pinned to that interval")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"at() raised {type(exc).__name__}: {exc}")
    try:
        links = case.dc_links()
        if not isinstance(links, dict):
            out.append(f"dc_links() returned {type(links).__name__}, not dict")
        else:
            for lid, spec in links.items():
                missing = [k for k in ("bus0", "bus1", "p_max") if k not in spec]
                if missing:
                    out.append(f"dc_links()[{lid!r}] lacks {missing}")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"dc_links() raised {type(exc).__name__}: {exc}")
    try:
        out.extend(check_fleet(case.nameplate_fleet(), label="nameplate_fleet()"))
    except Exception as exc:                        # noqa: BLE001
        out.append(f"nameplate_fleet() raised {type(exc).__name__}: {exc}")
    try:
        line = case.posture_line()
        if not isinstance(line, str) or not line.strip():
            out.append("posture_line() did not return a non-empty string")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"posture_line() raised {type(exc).__name__}: {exc}")
    return out


if __name__ == "__main__":
    # Smoke: the EDAM-literal transfer form against the pooled engines.
    import footprints as fpmod
    import wscc9_model as wm

    pt = compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    fp = fpmod.make(pt, {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]},
                    {"BA-1": "#993AFF", "BA-2": "#2471A3"})
    pooled = solve_engine_dispatch(pt, eng)
    new = solve_engine_transfers(pt, eng, fp.defs, {("BA-1", "BA-2"): None})
    assert all(abs(pooled.lmp[b] - new.lmp[b]) < 1e-4 for b in pt.buses), "pooled equivalence"
    capped = solve_engine_transfers(pt, eng, fp.defs, {("BA-1", "BA-2"): 30.0})
    k12 = ("BA-1", "BA-2")
    assert abs(capped.transfers[k12] - 30.0) < 1e-4, "capped schedule at its released capacity"  # type: ignore[index]  # solve_engine_transfers sets this field
    assert abs(capped.transfer_duals[k12]  # type: ignore[index]  # solve_engine_transfers sets this field
               - (capped.area_prices["BA-2"]  # type: ignore[index]  # solve_engine_transfers sets this field
                  - capped.area_prices["BA-1"]  # type: ignore[index]  # solve_engine_transfers sets this field
                  )) < 1e-3, \
        "binding interface dual = lamhat gap"
    print("2-BA: pooled equivalence OK; capped interface "
          f"t = {capped.transfers[k12]:.1f} MW at dual {capped.transfer_duals[k12]:+.2f}")  # type: ignore[index]  # solve_engine_transfers sets this field

    # congested triangle with A's imports capped: every stream nontrivial
    BA3 = {"A": ["1", "4", "5"], "B": ["3", "6", "7"], "C": ["2", "8", "9"]}
    r = solve_engine_transfers(pt, eng, BA3,
                               {("A", "B"): 40.0, ("A", "C"): 20.0, ("B", "C"): None},
                               shed_price=150.0)
    Nsum = sum(r.lmp[b] * (wm.DEFAULT_LOADS.get(b, 0) - r.gen_by_bus.get(b, 0)) for b in pt.buses)
    R = sum(abs(r.line_dual[l] * r.flow_own[l]) for l in pt.lines)
    RT = sum(abs(r.transfer_duals[k] * r.transfers[k]) for k in r.transfers)  # type: ignore[index, union-attr]  # solve_engine_transfers sets these fields
    assert abs(Nsum - R - RT) < 1e-2 and RT > 100, "sum N_a = R + R_T (capped triangle)"
    assert all(abs(r.transfer_duals[(x, y)] - (r.area_prices[y] - r.area_prices[x])) < 1e-3  # type: ignore[index]  # solve_engine_transfers sets this field
               for (x, y) in [("A", "B"), ("A", "C")]), "binding dual = lamhat gap"
    lamhat_str = {a: round(v, 2) for a, v in r.area_prices.items()}  # type: ignore[union-attr]  # solve_engine_transfers sets this field
    print(f"3-BA capped triangle: sum N_a = {Nsum:,.2f} = R ({R:,.2f}) + R_T ({RT:,.2f}); "
          f"lamhat {lamhat_str}")
# ---------------------------------------------------------------------------
# Real-time relief and superposed loading helpers (shared by 204/204a/304).
# ---------------------------------------------------------------------------

def rt_relief(pt: PTDFData, eng: MarketEngine, r_da: EngineResult,
              nbr_flows: LineValues, premium: float,
              shed_price: float | None = 150.0,
              ratings_override: LineValues | None = None,
              exo: BusValues | None = None,
              flowgates: FlowgateSet | None = None,
              ) -> tuple[MarketEngine, EngineResult]:
    """Real-time re-clear of ``eng``: the day-ahead blocks re-offered at cost
    (decrements buy back at the offer), increments above the schedule at
    ``cost * (1 + premium)``, and the neighbor's realized flows fixed on the
    engine's activated lines as ``flow_offsets``. ``ratings_override`` maps
    {line: MW} to replace a rating for this re-clear (block-relief targets).
    Returns ``(rt_engine, result)``; the RT engine's units are the re-offered
    day-ahead blocks and the increments, so figures drawn from the result must
    use the RT engine.

    ``exo`` is the exogenous schedule the engine carries in the real-time
    clearing. The default ``None`` carries the day-ahead result's own
    schedule, ``r_da.exo_by_bus``, because a schedule fixed before the
    day-ahead clearing is still fixed in real time and its flow contribution
    is still in the engine's book. A day-ahead clearing with no schedule
    carries an empty one, so the default leaves such a re-clear unchanged.
    Pass a dict to state a different real-time schedule, or ``{}`` to clear
    the engine without one.

    ``flowgates`` (added 2026-10-21, plan item B2) is the case's constraint
    set as a ``FlowgateSet``. When it is given, the re-clear enforces the
    flowgates rather than the lines. ``eng.activated_lines`` then names
    flowgate ids, ``nbr_flows`` maps each flowgate id to the neighbor's
    realized flowgate flow, and ``ratings_override`` maps a flowgate id to a
    rating applied in both directions, on a copy of the set. The default
    ``None`` rates every activated line against ``pt.s_nom``, as before.
    """
    import copy as _copy
    gens = {}
    for g, s in eng.gens.items():
        da = float(r_da.dispatch.get(g, 0.0))
        if da > 1e-9:
            gens[f'{g}__da'] = {'bus': s['bus'], 'cost': s['cost'], 'p_nom': da}
        rem = float(s['p_nom']) - da
        if rem > 1e-9:
            gens[f'{g}__inc'] = {'bus': s['bus'],
                                 'cost': s['cost'] * (1.0 + premium), 'p_nom': rem}
    act = list(eng.activated_lines) if eng.activated_lines != 'all' else 'all'
    eng_rt = MarketEngine(name=eng.name, gens=gens, loads=dict(eng.loads),
                          activated_lines=act)   # same name: tables/figures line up
    if flowgates is not None:
        fg_use = flowgates
        if ratings_override:
            fwd = np.asarray(flowgates.fwd, dtype=float).copy()
            rev = np.asarray(flowgates.rev, dtype=float).copy()
            idx = flowgates.fg_idx
            for k, v in ratings_override.items():
                fwd[idx[str(k)]] = rev[idx[str(k)]] = float(v)
            fg_use = FlowgateSet(ids=list(flowgates.ids), S=flowgates.S,
                                 fwd=fwd, rev=rev, members=dict(flowgates.members))
        mon = act if act != 'all' else list(fg_use.ids)
        offs = {k: float(nbr_flows.get(k, 0.0)) for k in mon}
        exo_rt = dict(r_da.exo_by_bus) if exo is None else {
            str(b): float(v) for b, v in exo.items()}
        r_rt = solve_engine_dispatch(pt, eng_rt, exo=exo_rt or None,
                                     flow_offsets=offs, shed_price=shed_price,
                                     flowgates=fg_use)
        return eng_rt, r_rt
    pt_use = pt
    if ratings_override:
        pt_use = _copy.deepcopy(pt)
        for l, v in ratings_override.items():
            pt_use.s_nom[pt_use.line_idx[l]] = float(v)
    mon = act if act != 'all' else list(pt_use.lines)
    offs = {l: float(nbr_flows.get(l, 0.0)) for l in mon}
    exo_rt = dict(r_da.exo_by_bus) if exo is None else {
        str(b): float(v) for b, v in exo.items()}
    r_rt = solve_engine_dispatch(pt_use, eng_rt, exo=exo_rt or None,
                                 flow_offsets=offs, shed_price=shed_price)
    return eng_rt, r_rt


def actual_flows(pt: PTDFData, results: list[EngineResult]) -> FloatArray:
    """Superposed line loading with generators and loads counted ONCE across
    the books. A schedule booked in both clears (a full-path trade; a tie
    pair) has no bus of its own, so summing the booked flows would re-count
    its path; this sums the bus-level injections instead. Returns the flow
    vector in ``pt.lines`` order.

    This is the vector ``physical_flows(pt, results, exo_physical=False)``
    tabulates, and both read it from :func:`actual_injection`, so the two
    never drift apart.

    The value moved on 2026-09-19 for three kinds of clearing, and in each the
    new value is the flow the dispatch actually places. Where a result SHEDS,
    the withdrawal is now the served load ``d_n - u_n`` rather than the nominal
    load, so a Market A engine loaded past what it can reach reports 40.0 MW on
    a 40 MW line where the old form reported 102.6. Where a result clears a
    price-sensitive DEMAND BID, that bid is now counted as the withdrawal it
    is. Where a result carries a DC LINK, its transfers are now counted,
    because a link is physical and flow-carrying. In all three the new value
    equals the engine's own ``flow_own`` when the engine carries no exogenous
    schedule, which the old form contradicted. A result with none of the three
    is unchanged to the last bit.
    """
    return pt.ptdf @ actual_injection(pt, results)
