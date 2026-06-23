# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Settlement, congestion- and transfer-revenue allocation, and position ledgers.

This is the methodology layer the congestion-revenue notebooks share (and the
fundamentals notebook borrows from). Every function operates on a cleared
:class:`seams_engine.EngineResult` and a :class:`footprints.Footprints`
partition, and returns plain pandas tables — there is no plotting here.

The pieces, grouped:

**Settlement** — `ba_settlement` (per-area generator revenue / served-load
payment at nodal LMPs), `settlement_by_bus`, `capacity_value_by_line`.

**Congestion rent** — `line_rent_table` (|μ|·|F| per line), `border_separation`
(the cross-border separation τ from its causes, eq. ΣΔSF·μ),
`allocate_congestion_rent` (Method 1 = managing area keeps its lines' rent;
Method 2 = rebate τ to the net-payer), `compare_methods`, and the homework-style
ledgers `revenue_table` / `autarky_vs_unified`.

**Transfer rent** — `transfer_rent` (|μ_T·E|), `allocate_transfer_rent`
(T1 fixed shares / T2 to the net-payer), `solve_with_transfer`, `transfer_ledger`.

**Position ledger** — `position_ledger`, the per-footprint cash-flow table
(consumers, generator surplus, trader/transfer, Total = −production cost) the
seams and two-settlement notebooks settle with; positions sum to −(production
cost) and any pure transfer price nets out at the system total.

**Independent operation** — `independent_clear`, each area as its own engine on
the full network (autarky baseline / infeasibility-as-a-finding), and `_agg`
(area payment / revenue / production cost), `cost_by_bus`.

Allocation *policy knobs* (which methods to tabulate, the T1 shares, the
transfer limit) stay **visible** in the notebooks and are passed in.
"""

from __future__ import annotations

import pandas as pd

import wscc9_model as wm
from seams_engine import compute_ptdf, solve_engine_dispatch

#: Display names for the transfer methodologies.
T_NAMES = {1: "T1 -- fixed shares (TRANSFER_SPLIT)",
           2: "T2 -- all to the net-payer BA"}

#: Row order for the autarky-vs-unified ledger.
ROWS = ["Autarky: payment / revenue", "Autarky: production cost",
        "Autarky: own congestion rent (CRR)", "Autarky: position",
        "Unified: payment / revenue", "Unified: production cost",
        "Congestion rent (CRR)", "Final position", "Delta vs autarky",
        "Pareto (Delta >= 0)"]


# ──────────────────────────────────────────────────────────────────────────
# Settlement
# ──────────────────────────────────────────────────────────────────────────
def ba_settlement(fp, res, loads):
    """Per-AREA generator revenue (lmp·g) and SERVED-load payment (lmp·(d−u)) at
    nodal LMPs. Areas = the footprints plus, when buses sit outside every
    footprint, a 'Non-market' area — it settles at LMP like any other
    (conservation needs its column) but is never allocated rent. Shed load pays
    nothing."""
    out = {}
    for area, buses in fp.areas.items():
        gen_rev = sum(res.lmp[b] * res.gen_by_bus.get(b, 0.0) for b in buses)
        load_pay = sum(res.lmp[b] * (float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0))
                       for b in buses)
        out[area] = dict(gen_rev=gen_rev, load_pay=load_pay, net_into_pool=load_pay - gen_rev)
    return out


def settlement_by_bus(res, buses, loads):
    """Bus-level settlement (LMP, gen, paid-to-gen, load, paid-by-load) + SUBTOTAL,
    over the given buses — the one-engine accounting of the fundamentals notebook."""
    t = pd.DataFrame(
        [{"bus": b,
          "LMP ($/MWh)": round(res.lmp[b], 2),
          "gen (MW)": round(res.gen_by_bus.get(b, 0.0), 1),
          "paid to gen ($/h)": round(res.lmp[b] * res.gen_by_bus.get(b, 0.0), 1),
          "load (MW)": round(float(loads.get(b, 0.0)), 1),
          "paid by load ($/h)": round(res.lmp[b] * float(loads.get(b, 0.0)), 1)}
         for b in buses]
    ).set_index("bus")
    t.loc["SUBTOTAL"] = ["", t["gen (MW)"].sum(), t["paid to gen ($/h)"].sum(),
                         t["load (MW)"].sum(), t["paid by load ($/h)"].sum()]
    return t


def capacity_value_by_line(res, pt, lines):
    """Constraint-level accounting: per line flow, rating, binding flag, shadow
    price |μ|, and rent |μ|·flow + TOTAL — the marginal value of line capacity."""
    t = pd.DataFrame(
        [{"line": l,
          "from": pt.line_buses[pt.line_idx[l]][0],
          "to": pt.line_buses[pt.line_idx[l]][1],
          "flow (MW)": round(res.flow_own[l], 1),
          "rating (MW)": round(pt.s_nom[pt.line_idx[l]], 0),
          "binding": abs(res.line_dual[l]) > 1e-3,
          "|mu| ($/MWh = value of +1 MW)": round(abs(res.line_dual[l]), 2),
          "rent |mu| x flow ($/h)": round(abs(res.line_dual[l]) * abs(res.flow_own[l]), 1)}
         for l in lines]
    ).set_index("line")
    t.loc["TOTAL"] = ["", "", "", "", "", "", round(t["rent |mu| x flow ($/h)"].sum(), 1)]
    return t


# ──────────────────────────────────────────────────────────────────────────
# Congestion rent
# ──────────────────────────────────────────────────────────────────────────
def line_rent_table(fp, res, pt):
    """Per-line congestion rent |μ|·|F|, tagged by topology (internal/tie) and by
    the managing footprint from the configurable line assignment."""
    recs = []
    for l in pt.lines:
        i = pt.line_idx[l]
        b0, b1 = pt.line_buses[i]
        mu, F = res.line_dual[l], res.flow_own[l]
        recs.append(dict(line=l, frm=b0, to=b1, kind=fp.line_kind(pt, l)[0],
                         home=fp.line_assign.get(l),
                         mu=round(mu, 2), flow=round(F, 1), rent=abs(mu) * abs(F)))
    return pd.DataFrame(recs).set_index("line")


def border_separation(fp, res, pt):
    """Cross-border separation rent per tie, priced from its causes:
    dlam_cong = Σ over binding lines of (SF at to-bus − SF at from-bus)·μ — the
    part of the price gap line congestion creates. dlam is the raw LMP gap;
    dlam_xfer = dlam − dlam_cong is the part a binding transfer constraint
    creates (±μ_T), settled separately as transfer rent. sep_rent prices the
    crossing flow at dlam_cong only."""
    recs = []
    for l in fp.ties:
        i = pt.line_idx[l]
        b0, b1 = pt.line_buses[i]
        F, dlam = res.flow_own[l], res.lmp[b1] - res.lmp[b0]
        dlam_cong = sum((pt.ptdf[pt.line_idx[m], pt.bus_idx[b1]]
                         - pt.ptdf[pt.line_idx[m], pt.bus_idx[b0]]) * mu
                        for m, mu in res.line_dual.items())
        imp = fp.fp_of(b1) if F >= 0 else fp.fp_of(b0)
        exp = fp.fp_of(b0) if F >= 0 else fp.fp_of(b1)
        recs.append(dict(line=l, flow=round(F, 1), dlam=round(dlam, 2),
                         dlam_cong=round(dlam_cong, 2),
                         dlam_xfer=round(dlam - dlam_cong, 2),
                         sep_rent=abs(dlam_cong * F), importing=imp, exporting=exp))
    return pd.DataFrame(recs).set_index("line")


def allocate_congestion_rent(fp, res, pt, loads, unassigned_split=0.5):
    """Allocate total congestion rent to the two footprints under both methods.
    Rent follows the line assignment; rent on unassigned lines is split
    unassigned_split / (1 − unassigned_split). Method 2 rebates the cross-border
    separation τ to the net-payer footprint, funded by the other."""
    lr, sep = line_rent_table(fp, res, pt), border_separation(fp, res, pt)
    R = lr["rent"].sum()
    R_unassigned = lr[lr.home.isna()]["rent"].sum()
    R_own = {ba: lr[lr.home == ba]["rent"].sum() for ba in fp.names}
    R_border = sep["sep_rent"].sum() if len(sep) else 0.0
    settle = ba_settlement(fp, res, loads)
    hedged_ba = max(fp.names, key=lambda ba: settle[ba]["net_into_pool"])
    funding_ba = [ba for ba in fp.names if ba != hedged_ba][0]

    alloc = {ba: dict(R_own=R_own[ba], unassigned_share=unassigned_split * R_unassigned,
                      method1=R_own[ba] + unassigned_split * R_unassigned) for ba in fp.names}
    tau = R_border
    alloc[hedged_ba]["method2"] = alloc[hedged_ba]["method1"] + tau
    alloc[funding_ba]["method2"] = alloc[funding_ba]["method1"] - tau
    summary = dict(R=R, R_unassigned=R_unassigned, R_own=R_own, R_border=R_border,
                   hedged_ba=hedged_ba, funding_ba=funding_ba, tau=tau)
    return pd.DataFrame(alloc).T[["R_own", "unassigned_share", "method1", "method2"]], summary, lr, sep


def compare_methods(fp, res, pt, loads):
    """Tidy side-by-side footprint settlement + Method 1 / Method 2 table."""
    alloc, summ, lr, sep = allocate_congestion_rent(fp, res, pt, loads)
    settle = ba_settlement(fp, res, loads)
    tbl = pd.DataFrame({ba: {
        "generator revenue (lmp*g)": settle[ba]["gen_rev"],
        "load payment (lmp*d)": settle[ba]["load_pay"],
        "own-line rent (assigned)": alloc.loc[ba, "R_own"],
        "unassigned-line share": alloc.loc[ba, "unassigned_share"],
        "congestion rent -- Method 1": alloc.loc[ba, "method1"],
        "congestion rent -- Method 2": alloc.loc[ba, "method2"],
    } for ba in fp.names})
    tbl["TOTAL"] = tbl.sum(axis=1)
    return tbl.round(1), summ, lr, sep


def _ledger(fp, alloc, s, method):
    """One allocation method's ledger — each area's consumers and generators side
    by side (cash out negative, cash in positive). Per area: Consumer, Generator,
    Area net. The bottom row carries the two readings of the area-position
    identity (Area net = A − N; Consumer = −(load cost net of CRR)); the TOTAL
    column checks conservation (rent row → R, net positions → 0)."""
    col = "method1" if method == 1 else "method2"
    data = {}
    for ba in fp.areas:
        gr, lp = s[ba]["gen_rev"], s[ba]["load_pay"]
        A = float(alloc.loc[ba, col]) if ba in alloc.index else 0.0
        rows_c = {"Energy settlement  (lmp x q)": -lp,
                  "Congestion rent allocated  (A)": A,
                  "Net position  (Area net: 0 => whole)": -lp + A}
        rows_g = {"Energy settlement  (lmp x q)": gr,
                  "Congestion rent allocated  (A)": "",
                  "Net position  (Area net: 0 => whole)": gr}
        data[(ba, "Consumer")] = rows_c
        data[(ba, "Generator")] = rows_g
        data[(ba, "Area net")] = {k: rows_c[k] + (rows_g[k] or 0.0) for k in rows_c}
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df[("TOTAL", "")] = sum(df[(ba, "Area net")] for ba in fp.areas)
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v)


def revenue_table(fp, res, pt, loads):
    """Two ledgers (one per allocation method), each area's consumers and
    generators side by side with an Area-net column (the autarky-vs-unified
    layout). Returns (method-1 table, method-2 table, summary)."""
    alloc, summ, lr, sep = allocate_congestion_rent(fp, res, pt, loads)
    s = ba_settlement(fp, res, loads)
    return _ledger(fp, alloc, s, 1), _ledger(fp, alloc, s, 2), summ


# ──────────────────────────────────────────────────────────────────────────
# Independent operation + aggregation
# ──────────────────────────────────────────────────────────────────────────
def cost_by_bus(gen_fleet=None):
    """``{bus: marginal cost}`` from a fleet (default the canonical teaching fleet)."""
    fleet = wm.DEFAULT_GEN_FLEET if gen_fleet is None else gen_fleet
    return {s["bus"]: s["cost"] for s in fleet.values()}


def _agg(fp, result, area, loads=None, cost=None):
    """(served-load payment, generator revenue, production cost) for an AREA at
    `result` prices — shed load pays nothing."""
    loads = wm.DEFAULT_LOADS if loads is None else loads
    cost = cost_by_bus() if cost is None else cost
    bs = fp.areas[area]
    L = sum(result.lmp[b] * (loads.get(b, 0.0) - result.shed_by_bus.get(b, 0.0)) for b in bs)
    R = sum(result.lmp[b] * result.gen_by_bus.get(b, 0.0) for b in bs)
    PC = sum(cost.get(b, 0.0) * result.gen_by_bus.get(b, 0.0) for b in bs)
    return L, R, PC


def independent_clear(fp, rat, gen_fleet=None, loads=None, shed_price=None, split_5_6=False):
    """Each AREA as its own independent engine on the full network — own gens
    serve own load, enforcing ONLY the lines assigned to it (rest relaxed), no
    interchange. A nodal DC-OPF per area; infeasibility is returned as ``None``
    (a finding) unless ``shed_price`` is given (then the area sheds at the
    penalty). Returns ``(pt, {area: EngineResult|None})``."""
    n = wm.build_network(rat, split_5_6=split_5_6)
    pt = compute_ptdf(n, slack_bus="1")
    out = {}
    for area, buses in fp.areas.items():
        act = [l for l in pt.lines if fp.line_assign.get(l) == area]
        try:
            eng = wm.make_engine(area, buses, gen_fleet=gen_fleet, loads=loads, activated=act)
            out[area] = solve_engine_dispatch(pt, eng, shed_price=shed_price)
        except (RuntimeError, ValueError):
            out[area] = None
    return pt, out


def autarky_vs_unified(fp, method, alloc, indep, resU, loads=None, cost=None):
    """Homework-style autarky (independent) vs unified ledger for one allocation
    method. `alloc` is the congestion allocation table, `indep` the per-area
    autarky results, `resU` the unified clearing. Cash out negative; the TOTAL
    column's positions sum to −(production cost)."""
    data = {}
    for ba in fp.areas:
        La, Ra, PCa = _agg(fp, indep[ba], ba, loads, cost)
        Lu, Ru, PCu = _agg(fp, resU, ba, loads, cost)
        col = "method1" if method == 1 else "method2"
        A = float(alloc.loc[ba, col]) if ba in alloc.index else 0.0
        Ra_int = La - Ra
        PSa, PSu = Ra - PCa, Ru - PCu
        cons_aut = -La + Ra_int
        cons_fin = -Lu + A
        cons_d = cons_fin - cons_aut
        gen_d = PSu - PSa
        data[(ba, "Consumer")] = {
            "Autarky: payment / revenue": -La, "Autarky: production cost": "",
            "Autarky: own congestion rent (CRR)": Ra_int, "Autarky: position": cons_aut,
            "Unified: payment / revenue": -Lu, "Unified: production cost": "",
            "Congestion rent (CRR)": A, "Final position": cons_fin,
            "Delta vs autarky": cons_d,
            "Pareto (Delta >= 0)": "yes" if cons_d >= -1e-6 else "no"}
        data[(ba, "Generator")] = {
            "Autarky: payment / revenue": Ra, "Autarky: production cost": -PCa,
            "Autarky: own congestion rent (CRR)": "", "Autarky: position": PSa,
            "Unified: payment / revenue": Ru, "Unified: production cost": -PCu,
            "Congestion rent (CRR)": "", "Final position": PSu,
            "Delta vs autarky": gen_d,
            "Pareto (Delta >= 0)": "yes" if gen_d >= -1e-6 else "no"}
    df = pd.DataFrame(data).reindex(ROWS)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)


# ──────────────────────────────────────────────────────────────────────────
# Transfer rent
# ──────────────────────────────────────────────────────────────────────────
def transfer_rent(res):
    """R_T = |μ_T · E| — zero unless the transfer constraint binds."""
    return abs((res.interchange_dual or 0.0) * (res.interchange_mw or 0.0))


def allocate_transfer_rent(fp, res, loads, t_method, transfer_split):
    """σ_a by transfer methodology: T1 = fixed shares (``transfer_split``, EDAM's
    equal split); T2 = the whole R_T to the NET-PAYER footprint (the transfer
    analogue of congestion Method 2). Returns ``{footprint: $ of R_T}``."""
    RT = transfer_rent(res)
    if t_method == 1:
        return {ba: transfer_split[ba] * RT for ba in fp.names}
    if t_method == 2:
        s = ba_settlement(fp, res, loads)
        payer = max(fp.names, key=lambda ba: s[ba]["net_into_pool"])
        return {ba: (RT if ba == payer else 0.0) for ba in fp.names}
    raise ValueError(f"unknown transfer methodology {t_method!r}")


def solve_with_transfer(fp, ratings, ebar, gen_fleet=None, loads=None,
                        shed_price=None, split_5_6=False):
    """Unified clearing with the net-interchange constraint |E| ≤ ebar on the
    FIRST footprint's bus set, plus the optional load-shed relaxation. Returns
    ``(n, pt, engine, res)``."""
    n = wm.build_network(ratings, split_5_6=split_5_6)
    p = compute_ptdf(n, slack_bus="1")
    e = wm.make_engine("UNIFIED", buses=p.buses, gen_fleet=gen_fleet, loads=loads)
    r = solve_engine_dispatch(p, e, interchange=(fp.defs[fp.names[0]], ebar),
                              shed_price=shed_price)
    return n, p, e, r


def transfer_ledger(fp, res, p, loads, method, t_method, indep, transfer_split, cost=None):
    """Autarky vs the transfer-constrained clearing — the per-area
    Consumer/Generator layout with both revenue streams shown COLLECTED
    separately from ALLOCATED (congestion by Method `method`; transfer by
    `t_method`). `indep` = autarky results on the SAME ratings. TOTAL positions
    sum to −(production cost)."""
    alloc, summ, lr, sep = allocate_congestion_rent(fp, res, p, loads)
    col = "method1" if method == 1 else "method2"
    RT = transfer_rent(res)
    AT_map = allocate_transfer_rent(fp, res, loads, t_method, transfer_split)
    data = {}
    for ba in fp.areas:
        assert indep[ba] is not None, (f"{ba} has no autarky baseline -- pass "
                                       "shed_price=SHED_PRICE to independent_clear")
        La, Ra, PCa = _agg(fp, indep[ba], ba, loads, cost)
        Lc, Rc, PCc = _agg(fp, res, ba, loads, cost)
        own_aut = La - Ra
        A = float(alloc.loc[ba, col]) if ba in alloc.index else 0.0
        AT = AT_map.get(ba, 0.0)
        cons_aut, gen_aut = -La + own_aut, Ra - PCa
        cons_fin, gen_fin = -Lc + A + AT, Rc - PCc
        data[(ba, "Consumer")] = {
            "Autarky: payment / revenue": -La, "Autarky: production cost": "",
            "Autarky: own congestion rent (CRR)": own_aut, "Autarky: position": cons_aut,
            "Constrained: payment / revenue": -Lc, "Constrained: production cost": "",
            "Congestion rent collected (assigned lines)": summ["R_own"].get(ba, 0.0),
            "Congestion rent allocated (CRR)": A,
            "Transfer rent collected (border)": "",
            "Transfer rent allocated (CRR)": AT,
            "Final position": cons_fin,
            "Delta vs autarky": cons_fin - cons_aut,
            "Pareto (Delta >= 0)": "yes" if cons_fin - cons_aut >= -1e-6 else "no"}
        data[(ba, "Generator")] = {
            "Autarky: payment / revenue": Ra, "Autarky: production cost": -PCa,
            "Autarky: own congestion rent (CRR)": "", "Autarky: position": gen_aut,
            "Constrained: payment / revenue": Rc, "Constrained: production cost": -PCc,
            "Congestion rent collected (assigned lines)": "",
            "Congestion rent allocated (CRR)": "",
            "Transfer rent collected (border)": "",
            "Transfer rent allocated (CRR)": "",
            "Final position": gen_fin,
            "Delta vs autarky": gen_fin - gen_aut,
            "Pareto (Delta >= 0)": "yes" if gen_fin - gen_aut >= -1e-6 else "no"}
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    tot = df.map(lambda v: v if isinstance(v, (int, float)) else 0.0).sum(axis=1)
    tot["Congestion rent collected (assigned lines)"] = summ["R"]
    tot["Transfer rent collected (border)"] = RT
    tot = tot.astype(object)
    tot["Pareto (Delta >= 0)"] = ""
    df[("TOTAL", "")] = tot
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v)


# ──────────────────────────────────────────────────────────────────────────
# Position ledger (generalized from the seams / two-settlement notebooks)
# ──────────────────────────────────────────────────────────────────────────
def position_ledger(fp, res, gen_fleet=None, loads=None, *, trader=0.0,
                    trader_label="Trader"):
    """Per-footprint net positions at ``res``'s prices — the seams/two-settlement
    settlement ledger:

    * consumers   = −Σ lmp·load over the footprint's loads;
    * generators  = Σ (lmp_bus − cost)·g over the footprint's units (producer
      surplus / inframarginal rent);
    * a single trader / transfer row (e.g. a bilateral or self-schedule leg —
      a pure transfer that nets out at the system total); and
    * Total = −production cost.

    Positions sum to −(production cost). Returns ``{(footprint, role): $/h}``.
    """
    fleet = wm.DEFAULT_GEN_FLEET if gen_fleet is None else gen_fleet
    loads = wm.DEFAULT_LOADS if loads is None else loads
    out = {}
    for m, buses in fp.defs.items():
        loads_m = {b: loads[b] for b in buses if b in loads}
        out[(m, "consumers")] = -sum(res.lmp[b] * mw for b, mw in loads_m.items())
        out[(m, "generators")] = sum(
            (res.lmp[str(fleet[g]["bus"])] - fleet[g]["cost"]) * mw
            for g, mw in res.dispatch.items() if str(fleet[g]["bus"]) in buses)
    out[(trader_label, "")] = float(trader)
    out[("Total", "production cost")] = -res.total_cost
    return out


def wheeling_settlement(source, sink, E, rA, rB, *,
                        transport_market="Market A", wheeling_market="Market B"):
    """Settle a transfer ``source``→``sink`` of ``E`` MW that clears in a
    **transport** market ``rA`` (two non-electrically-connected BAs, coordinated
    by a scheduling limit) and physically **wheels across** a contiguous market
    ``rB``.

    The transaction settles in BOTH markets. In the transport market its only
    revenue is the **transfer rent** μ_T·E = (λ^A_sink − λ^A_source)·E — the price
    separation there is purely the transfer constraint (the two ends share no
    line). The wheeling market settles the same megawatts as a **self-schedule**
    (imports at ``source`` @ λ^B_source, exports at ``sink`` @ λ^B_sink) and is
    allocated the **congestion rent** (λ^B_sink − λ^B_source)·E that its own
    constraints create between the two points. Returns ``(legs_table, summary)``.
    """
    src, snk = str(source), str(sink)
    laA_s, laA_k = rA.lmp[src], rA.lmp[snk]
    laB_s, laB_k = rB.lmp[src], rB.lmp[snk]
    muT, wheel = laA_k - laA_s, laB_k - laB_s
    rows = [
        {"market": transport_market, "leg": f"buy source @ bus {src}",
         "settles at $/MWh": round(laA_s, 2), "cash to txn $/h": round(-laA_s * E, 1)},
        {"market": transport_market, "leg": f"sell sink @ bus {snk}",
         "settles at $/MWh": round(laA_k, 2), "cash to txn $/h": round(+laA_k * E, 1)},
        {"market": wheeling_market, "leg": f"import @ bus {src}",
         "settles at $/MWh": round(laB_s, 2), "cash to txn $/h": round(-laB_s * E, 1)},
        {"market": wheeling_market, "leg": f"export @ bus {snk}",
         "settles at $/MWh": round(laB_k, 2), "cash to txn $/h": round(+laB_k * E, 1)},
    ]
    legs = pd.DataFrame(rows).set_index(["market", "leg"])
    summary = dict(E=E, muT=muT, transfer_rent=muT * E,
                   wheel_spread=wheel, wheeling_rent=wheel * E,
                   laA=(laA_s, laA_k), laB=(laB_s, laB_k))
    return legs, summary


if __name__ == "__main__":
    import footprints as fpmod

    pt = wm.shift_factors()
    fp = fpmod.make(pt, {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]},
                    {"BA-1": "#993AFF", "BA-2": "#2471A3"},
                    manage={"BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
                            "BA-2": ["line_0", "line_1", "line_7", "line_8"]})

    SCN = {"line_4": 40.0}
    netB = wm.build_network(SCN)
    ptB = compute_ptdf(netB, slack_bus="1")
    resU = solve_engine_dispatch(ptB, wm.make_engine("UNIFIED", buses=ptB.buses))
    _, indep = independent_clear(fp, SCN, shed_price=150.0)
    alloc, summ, _, _ = allocate_congestion_rent(fp, resU, ptB, wm.DEFAULT_LOADS)
    print(f"congestion rent R = ${summ['R']:.1f}/h ; tau = ${summ['tau']:.1f}")
    av = autarky_vs_unified(fp, 1, alloc, indep, resU)
    fin = av.loc["Final position"]
    tot = sum(v for v in fin.values if isinstance(v, (int, float)))
    print(f"Method-1 autarky-vs-unified: final positions sum = {tot:,.1f} "
          f"(= -production cost {-sum(_agg(fp, resU, a)[2] for a in fp.areas):,.1f}?)")
    print("\nposition ledger (unified) — consumers + generators + trader = -(R + cost):")
    pl = position_ledger(fp, resU)
    for k, v in pl.items():
        print(f"  {k}: {v:,.1f}")
    parties = sum(v for k, v in pl.items() if k[0] != "Total")
    print(f"  parties sum = {parties:,.1f}  vs  -(R + cost) "
          f"= {-(summ['R'] + sum(_agg(fp, resU, a)[2] for a in fp.areas)):,.1f}")
