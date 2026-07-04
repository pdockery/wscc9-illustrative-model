# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
its own monitored lines, missing a neighbour's loop flow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# Monitored-line resolution
# ──────────────────────────────────────────────────────────────────────────
def _monitored_idx(pt, monitored) -> list[int]:
    """Indices of the monitored lines (``"all"`` or a list of line names)."""
    if monitored == "all":
        return list(range(pt.n_line))
    return [pt.line_idx[str(l)] for l in monitored]


# ──────────────────────────────────────────────────────────────────────────
# Path shift factors, TTC, the ATC identity
# ──────────────────────────────────────────────────────────────────────────
def path_shift_factors(pt, source, sink) -> np.ndarray:
    """Per-MW loading of every line by a point-to-point reservation ``source->sink``.

    ``a_l = SF_{l, source} - SF_{l, sink}`` (eq. 1). The slack bus cancels, so the
    result is independent of the PTDF's slack choice for a balanced transaction.
    """
    return pt.ptdf[:, pt.bus_idx[str(source)]] - pt.ptdf[:, pt.bus_idx[str(sink)]]


def ttc(pt, source, sink, monitored="all", tol: float = 1e-6) -> tuple[float, str]:
    """Total Transfer Capability of the path and the first line to bind (eq. 2).

    ``TTC = min_l F_bar_l / |a_l|`` over the monitored lines (a line the path
    barely touches, ``|a_l| < tol``, never binds and is skipped). Returns
    ``(mw, binding_line_name)``.
    """
    a = path_shift_factors(pt, source, sink)
    best_mw, best_line = np.inf, None
    for l in _monitored_idx(pt, monitored):
        if abs(a[l]) < tol:
            continue
        lim = pt.s_nom[l] / abs(a[l])
        if lim < best_mw:
            best_mw, best_line = lim, pt.lines[l]
    return float(best_mw), best_line


def atc(ttc_mw: float, etc: float = 0.0, trm: float = 0.0, cbm: float = 0.0) -> float:
    """The firm ATC identity ``ATC = TTC - TRM - ETC_F - CBM`` (floored at 0).

    Defaults ``TRM = CBM = 0`` for the illustrations; the four terms are named so
    the markdown can reference the full NERC definition. ``etc`` is the capability
    (MW, in this path's direction) already consumed by existing commitments.
    """
    return max(0.0, ttc_mw - trm - etc - cbm)


# ──────────────────────────────────────────────────────────────────────────
# Awards and the simultaneous-feasibility test
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Award:
    """A granted point-to-point reservation: ``mw`` MW from ``source`` to ``sink``."""

    source: str
    sink: str
    mw: float

    def __post_init__(self):
        self.source = str(self.source)
        self.sink = str(self.sink)
        self.mw = float(self.mw)


def _as_award(a) -> Award:
    return a if isinstance(a, Award) else Award(*a)


def superposed_flow(pt, awards) -> np.ndarray:
    """Signed flow on every line from the superposition of all awards: ``Sum_j q_j a_l``."""
    f = np.zeros(pt.n_line)
    for a in awards:
        a = _as_award(a)
        f += a.mw * path_shift_factors(pt, a.source, a.sink)
    return f


def line_loadings(pt, awards, monitored="all") -> pd.DataFrame:
    """Per-line SFT table for a set of awards.

    Columns: ``from``/``to`` buses, signed ``flow`` (MW), ``limit`` (F_bar),
    ``loading_%``, and ``overload`` (``|flow| > limit``). Restricted to the
    monitored lines — also the source of ``line_flows``/``constrained_lines`` for
    the network figures.
    """
    f = superposed_flow(pt, awards)
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


def simultaneous_feasibility(pt, awards, monitored="all") -> tuple[bool, pd.DataFrame]:
    """Hogan's SFT (eq. 3): is the superposed flow within every monitored limit?

    Returns ``(feasible, loadings_table)``. ``feasible`` is True iff no monitored
    line overloads.
    """
    df = line_loadings(pt, awards, monitored=monitored)
    return (not bool(df["overload"].any()), df)


def overloaded_lines(pt, awards, monitored="all") -> list[str]:
    """Names of the monitored lines that overload under the award set (for figures)."""
    df = line_loadings(pt, awards, monitored=monitored)
    return df.index[df["overload"]].tolist()


def firm_line_loadings(pt, awards, monitored="all") -> pd.DataFrame:
    """Per-line FIRM loading -- the conservative ATC standard that does NOT credit counterflow.

    Firm point-to-point rights must be deliverable even if the counter-flowing rights are not
    scheduled, so on each line the same-direction commitments are summed *without* netting the
    opposing ones: ``fwd = Sum_j max(0, q_j a_l)``, ``rev = Sum_j max(0, -q_j a_l)``, and the firm
    loading is ``max(fwd, rev)``. Contrast :func:`line_loadings`, which nets the flows by
    superposition -- the efficient-dispatch / financial-FTR notion that *does* credit counterflow.
    Columns: ``from``/``to``, ``fwd``/``rev`` directional sums, ``firm`` loading, ``limit``,
    ``loading_%`` (firm vs limit), ``overload`` (``firm > limit``).
    """
    idx = _monitored_idx(pt, monitored)
    contrib = {l: [] for l in idx}
    for a in awards:
        a = _as_award(a)
        sf = path_shift_factors(pt, a.source, a.sink)
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


def firm_feasibility(pt, awards, monitored="all") -> tuple[bool, pd.DataFrame]:
    """Firm-rights ATC feasibility: are the same-direction commitments within every monitored
    limit *without* crediting counterflow (:func:`firm_line_loadings`)?

    The conservative standard for firm point-to-point transmission rights -- a set can be firm
    even when a counter-flowing right would let the efficient (netted) dispatch carry more, and
    a set the netted :func:`simultaneous_feasibility` passes can still be firm-infeasible if it
    leans on counterflow. Returns ``(feasible, firm_loadings_table)``.
    """
    df = firm_line_loadings(pt, awards, monitored=monitored)
    return (not bool(df["overload"].any()), df)


def flow_dict(pt, awards) -> dict[str, float]:
    """``{line: signed MW}`` of the superposed award flow (feeds ``plot_network_topology``)."""
    f = superposed_flow(pt, awards)
    return {pt.lines[l]: float(f[l]) for l in range(pt.n_line)}


# ──────────────────────────────────────────────────────────────────────────
# Sequential booking — decrement ATC by existing commitments (ETC)
# ──────────────────────────────────────────────────────────────────────────
def available_atc(pt, source, sink, booked, monitored="all", tol: float = 1e-6) -> float:
    """Remaining ATC for ``source->sink`` given already-``booked`` awards (ETC).

    For each monitored line the headroom in the *push direction* of this path is
    ``F_bar_l - sign(a_l) * F^booked_l``; the available transfer is the tightest
    ``headroom_l / |a_l|`` (floored at 0). With no bookings this reduces to TTC.
    """
    a = path_shift_factors(pt, source, sink)
    f0 = superposed_flow(pt, booked)
    avail = np.inf
    for l in _monitored_idx(pt, monitored):
        if abs(a[l]) < tol:
            continue
        headroom = pt.s_nom[l] - np.sign(a[l]) * f0[l]
        avail = min(avail, max(0.0, headroom) / abs(a[l]))
    return float(max(0.0, avail))


def book_sequentially(pt, requests, monitored="all") -> list[Award]:
    """Grant requests in order, decrementing ATC by existing commitments (ETC).

    ``requests`` is a list of ``(source, sink, mw)``; each is granted up to the
    ATC still available after the prior bookings (``available_atc``), so the
    cumulative award set is simultaneously feasible by construction. Granted MW
    of 0 still appears in the result (a fully-curtailed request).
    """
    booked: list[Award] = []
    for req in requests:
        r = _as_award(req)
        avail = available_atc(pt, r.source, r.sink, booked, monitored=monitored)
        booked.append(Award(r.source, r.sink, min(r.mw, avail)))
    return booked


# ──────────────────────────────────────────────────────────────────────────
# Footprint-aware ATC — each BA rates a path through only its own lines
# ──────────────────────────────────────────────────────────────────────────
def ba_monitored_lines(pt, fp, name: str) -> list[str]:
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


def ba_atc(pt, fp, name: str, source, sink, etc: float = 0.0) -> tuple[float, str]:
    """ATC a balancing authority posts for ``source->sink`` seeing only its own lines.

    Returns ``(atc_mw, binding_line)`` — TTC through ``ba_monitored_lines(name)``,
    less ``etc``. A neighbour's internal line that this path loads is invisible
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

    def __post_init__(self):
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
    mu: dict
    rent: float
    value: float


def crr_auction(pt, bids, monitored="all", obligations=None, tol: float = 1e-7) -> CRRResult:
    """Hogan's CRR/FTR auction: award point-to-point rights to maximise bid value
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
    """
    from scipy.optimize import linprog

    bids = [b if isinstance(b, Bid) else Bid(*b) for b in bids]
    idx = _monitored_idx(pt, monitored)
    n = len(bids)
    # A[l, i] = path shift factor of bid i on monitored line l (the SFT coefficient).
    A = np.zeros((len(idx), n))
    for i, b in enumerate(bids):
        A[:, i] = path_shift_factors(pt, b.source, b.sink)[idx]
    fbar = np.array([float(pt.s_nom[l]) for l in idx])
    # |A x| <= fbar  ->  two one-sided blocks (+A and -A); netting credits counterflow.
    blocks = [A, -A]
    rhs = [fbar, fbar]
    # Obligation-group caps appended AFTER the line rows (so line-dual indexing is unchanged).
    for g, cap in (obligations or {}).items():
        blocks.append(np.array([[1.0 if b.group == g else 0.0 for b in bids]]))
        rhs.append(np.array([float(cap)]))
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    c = np.array([-b.premium for b in bids])                 # maximise value = minimise -value
    bounds = [(0.0, b.mw) for b in bids]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CRR auction LP failed: {res.message}")

    x = np.asarray(res.x)
    # HiGHS marginals for (A_ub x <= b_ub) are <= 0; the scarcity price is their negative.
    m = -np.asarray(res.ineqlin.marginals)
    nl = len(idx)
    mu_line = m[:nl] - m[nl:2 * nl]                          # signed dual per monitored line (line rows only)
    charge = A.T @ mu_line                                   # flowgate charge Sum_l a_l mu_l per bid
    mu = {pt.lines[l]: round(float(mu_line[j]), 4)
          for j, l in enumerate(idx) if abs(mu_line[j]) > tol}

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


def awarded_rights(result: CRRResult, tier_key="label", min_mw: float = 1e-6) -> list[dict]:
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
