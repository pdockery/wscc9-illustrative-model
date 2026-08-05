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

**Transfer rent** — `transfer_rent` (Σ |μ_T^{ab}·t_ab|), `allocate_transfer_rent`
(T1 fixed shares / T2 to the net-payer / T3 per-interface halves),
`solve_with_transfers`, `transfer_ledger`.

**Position ledger** — `position_ledger`, the per-footprint cash-flow table
(consumers, generator surplus, trader/transfer, Total = −production cost) the
seams and two-settlement notebooks settle with; positions sum to −(production
cost) and any pure transfer price nets out at the system total.

**Independent operation** — `independent_clear`, each area as its own engine on
the full network (autarky baseline / infeasibility-as-a-finding), and `_agg`
(area payment / revenue / production cost), `cost_by_bus`.

Allocation *policy knobs* (which methods to tabulate, the T1 shares, the
released transfer capacities) stay **visible** in the notebooks and are passed in.
"""

from __future__ import annotations

import pandas as pd

import numpy as np

import atc
from seams_engine import (active as _case, compute_ptdf, solve_engine_dispatch,
                          solve_engine_qp, solve_engine_transfers)


def _cn(ratings=None, **variant):
    """The ambient case's network + shift factors (a ``CaseNetwork``).

    This is the ONLY way this module builds a network: through the case, never
    by importing a scenario module. On the 9-bus case it reproduces the old
    ``compute_ptdf(wm.build_network(ratings), slack_bus="1")`` exactly (the
    case's slack bus IS "1"); on any other case it builds that case instead.
    """
    return _case().build(ratings, **variant)

#: Display names for the transfer methodologies.
T_NAMES = {1: "T1 -- fixed shares (TRANSFER_SPLIT)",
           2: "T2 -- all to the net-payer BA",
           3: "T3 -- per-interface halves (each interface's stream split between its two parties)"}

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


def settlement_by_bus(res, buses, loads, all_buses=None):
    """Bus-level settlement (LMP, gen, paid-to-gen, load, paid-by-load) + SUBTOTAL —
    the one-engine accounting of the fundamentals notebook.

    ``buses`` are the buses actually cleared/settled in this dispatch. Pass
    ``all_buses`` to also list the buses OUTSIDE the dispatch (the greyed,
    out-of-area buses of the per-BA autarky / accommodation views): the table then
    gains an ``in settlement`` (True/False) column, and an out-of-area bus shows its
    LMP but zero gen / load / payments (it carries no resources here). Without
    ``all_buses`` the table is exactly the original five columns over ``buses``. The
    SUBTOTAL sums only the in-settlement rows (the rest are zero)."""
    settled = {str(b) for b in buses}
    flag = all_buses is not None
    disp = [str(b) for b in (all_buses if flag else buses)]
    rows = []
    for b in disp:
        ins = b in settled
        g = res.gen_by_bus.get(b, 0.0) if ins else 0.0
        d = float(loads.get(b, 0.0)) if ins else 0.0
        row = {"bus": b}
        if flag:
            row["in settlement"] = ins
        row["LMP ($/MWh)"] = round(res.lmp[b], 2)
        row["gen (MW)"] = round(g, 1)
        row["paid to gen ($/h)"] = round(res.lmp[b] * g, 1)
        row["load (MW)"] = round(d, 1)
        row["paid by load ($/h)"] = round(res.lmp[b] * d, 1)
        rows.append(row)
    t = pd.DataFrame(rows).set_index("bus")
    lead = ["", ""] if flag else [""]
    t.loc["SUBTOTAL"] = lead + [t["gen (MW)"].sum(), t["paid to gen ($/h)"].sum(),
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


def _ba_shares(fp, spec, res=None, loads=None):
    """Normalize a share specification over the footprints to ``{ba: sigma_a}``
    summing to one.

    ``spec`` may be: a **dict** ``{ba: share}`` covering every footprint and
    summing to 1; the string ``'even'`` (1/N each); the string ``'interchange'``
    (proportional to each footprint's scheduled net interchange ``|e_a|`` at the
    clearing — the N-footprint generalization of EDAM's equal split between the
    two parties to a transfer: with two footprints ``|e_1| = |e_2|`` always, so
    it IS the legacy 50/50, and a footprint that schedules no interchange gets
    no share); or a **float** ``s`` (two footprints only — the legacy
    ``{first: s, second: 1-s}`` form)."""
    names = list(fp.names)
    if isinstance(spec, dict):
        missing = [b for b in names if b not in spec]
        if missing:
            raise ValueError(f"share dict missing footprints {missing}")
        tot = float(sum(spec.values()))
        if abs(tot - 1.0) > 1e-9:
            raise ValueError(f"shares must sum to 1 (got {tot})")
        return {b: float(spec[b]) for b in names}
    if spec == "even":
        return {b: 1.0 / len(names) for b in names}
    if spec == "interchange":
        loads = _case().loads if loads is None else loads
        e = {a: abs(sum(float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0)
                        - res.gen_by_bus.get(b, 0.0) for b in fp.defs[a])) for a in names}
        tot = sum(e.values())
        return {b: (e[b] / tot if tot > 1e-9 else 1.0 / len(names)) for b in names}
    s = float(spec)
    if len(names) != 2:
        raise ValueError("a scalar share is the two-footprint legacy form; with more "
                         "footprints pass a dict, 'even', or 'interchange'")
    return {names[0]: s, names[1]: 1.0 - s}


def pairwise_transfer_settlement(fp, res, pt, loads=None, gen_fleet=None, weights=None):
    """Per-interface (footprint-pair) transfer settlement.

    Interfaces are the unordered footprint pairs joined by at least one tie
    line. Two per-interface quantities are reported side by side:

    * ``t_physical`` — the ORIENTED net physical flow a→b summed over the
      pair's ties (what the transmission elements carry);
    * ``t_scheduled`` — the engine's cleared transfer tags when the clearing
      came from ``solve_engine_transfers`` (``res.transfers``); on a TREE
      interface topology the per-area balances pin these to the cut flows, on
      CYCLES they differ and their difference, the ``loop flow`` column, is
      the unscheduled parallel flow circulating through third parties.

    Each interface settles hub-to-hub on the SCHEDULED quantity when tags are
    available (the EDAM reading — settlement rides the schedule), else on the
    physical flow: ``T_ab = t_ab (lambda_b - lambda_a)``. Either way the
    pairwise settlements partition the pooled total exactly (same hub
    weights): ``sum_ab T_ab = T_E = sum_a lambda_a e_a`` by telescoping on the
    net positions. The ``half share each`` column is the EDAM allocation rule
    (each transfer's surplus split equally between the two BAAs party to it).
    """
    loads = _case().loads if loads is None else loads
    if weights is None:
        weights = hub_weights(fp, "auto", loads, gen_fleet)
    lam = {a: sum(w * res.lmp[b] for b, w in weights[a].items()) for a in fp.names}
    rows = {}
    for l in pt.lines:
        b0, b1 = pt.line_buses[pt.line_idx[l]]
        a0, a1 = fp.fp_of(b0), fp.fp_of(b1)
        if a0 == a1 or a0 is None or a1 is None:
            continue
        key = tuple(sorted((a0, a1)))
        f = res.flow_own[l] if (a0, a1) == key else -res.flow_own[l]
        r = rows.setdefault(key, {"ties": [], "t": 0.0})
        r["ties"].append(l)
        r["t"] += f
    sched = getattr(res, "transfers", None)
    if sched is not None:
        for k, v in sched.items():
            rows.setdefault(tuple(sorted(k)), {"ties": [], "t": 0.0})  # scheduled-only interfaces
    out = {}
    for (a, b), r in sorted(rows.items()):
        t_phys = r["t"]
        t_sch = None
        if sched is not None:
            t_sch = float(sched.get((a, b), -sched.get((b, a), 0.0)))
        t_settle = t_sch if t_sch is not None else t_phys
        T = t_settle * (lam[b] - lam[a])
        row = {"ties": ", ".join(r["ties"]) if r["ties"] else "---",
               "t_physical (MW)": t_phys}
        if t_sch is not None:
            row["t_scheduled (MW)"] = t_sch
            row["loop flow (MW)"] = t_phys - t_sch
        row.update({"hub price a": lam[a], "hub price b": lam[b],
                    "T_ab": T, "half share each": 0.5 * T})
        out[(a, b)] = row
    df = pd.DataFrame(out).T
    df.index.names = ["a", "b"]
    return df


def fp_interfaces(fp, pt, areas=None):
    """The interface topology a partition implies: the unordered pairs joined
    by at least one tie line, ``{(a, b) sorted: [ties]}``.

    ``areas`` (``{area: [buses]}``, optional) computes the interfaces of a
    FINER partition than the footprints — the WECC two-level structure, where
    BA areas nest inside market-engine footprints and the per-BA interfaces are
    what ``solve_engine_transfers`` schedules over. Default: the footprints
    themselves (the two-level structure collapsed, today's behaviour)."""
    if areas is not None:
        a_of = {str(b): a for a, bs in areas.items() for b in bs}
        out: dict = {}
        for l in pt.lines:
            b0, b1 = pt.line_buses[pt.line_idx[l]]
            a0, a1 = a_of.get(b0), a_of.get(b1)
            if a0 != a1 and a0 is not None and a1 is not None:
                out.setdefault(tuple(sorted((a0, a1))), []).append(l)
        return out
    out = {}
    for l in pt.lines:
        b0, b1 = pt.line_buses[pt.line_idx[l]]
        a0, a1 = fp.fp_of(b0), fp.fp_of(b1)
        if a0 != a1 and a0 is not None and a1 is not None:
            out.setdefault(tuple(sorted((a0, a1))), []).append(l)
    return out


def flowgate_rent_table(res) -> pd.DataFrame:
    """Per-FLOWGATE congestion rent ``|μ_k · f_k|`` (+ a TOTAL row).

    The flowgate analogue of :func:`line_rent_table` — with one deliberate
    non-identity: for a path whose members carry flow in mixed signs relative
    to the path orientation, ``Σ_m |μ^impl_m||F_m| ≠ |μ_k f_k|`` per line, so
    per-line abs-rent does not decompose a flowgate's rent. The SIGNED sums
    agree exactly (``Σ_m μ^impl_m F_m = Σ_k μ_k f_k``), so total ``R`` and
    every conservation identity are unaffected; read per-constraint rent off
    THIS table when the constraint set is flowgates."""
    rows = []
    for k, mu in res.fg_dual.items():
        f = res.fg_flow.get(k, 0.0)
        fwd, rev = res.fg_limit.get(k, (float("nan"), float("nan")))
        rows.append(dict(flowgate=k, flow=f, fwd=fwd, rev=rev, mu=mu,
                         rent=abs(mu * f), binding=abs(mu) > 1e-6))
    df = pd.DataFrame(rows).set_index("flowgate")
    if len(df):
        tot = df[["rent"]].sum()
        df.loc["TOTAL", "rent"] = float(tot["rent"])
    return df


def solve_with_transfers(fp, ratings, limits=None, gen_fleet=None, loads=None,
                         shed_price=None, split_5_6=False, curves=None, exo=None):
    """Unified clearing in the EDAM-literal form (``solve_engine_transfers``):
    one balance per footprint, scheduled transfer variables on every interface
    the tie topology implies, and per-interface released-capacity limits.

    ``limits`` is ``{(a, b): MW}`` (pairs in either order; a pair omitted gets
    ``None`` = unlimited), a scalar (single-interface shorthand, valid only
    when the partition has exactly one interface — the two-footprint case),
    the string
    ``'mod29'`` (every interface's released capacity DERIVED as its
    independently-rated hub-to-hub TTC, ``atc.interface_ttc`` — the MOD-029
    posture, over-assigned by construction relative to the simultaneous
    capability; see ``atc.interface_sft``), or ``None`` for all-unlimited. A
    dict value may itself be ``'mod29'`` to derive that pair while overriding
    others manually. ``exo`` passes price-taking fixed injections through to
    the engine (every exo bus must lie in a footprint). Returns
    ``(n, pt, engine, res)``."""
    _c = _cn(ratings, split_5_6=split_5_6)
    n, p = _c.network, _c.pt
    e = _case().make_engine("UNIFIED", buses=p.buses, gen_fleet=gen_fleet, loads=loads)
    pairs = fp_interfaces(fp, p)
    _W: list = []
    def _mod29(k):
        if not _W:
            _W.append(hub_weights(fp, "auto", loads, gen_fleet))
        return atc.interface_ttc(p, [k], _W[0])[k][0]
    if limits is None:
        ifc = {k: None for k in pairs}
    elif isinstance(limits, str) and limits == "mod29":
        ifc = {k: _mod29(k) for k in pairs}
    elif isinstance(limits, dict):
        norm = {tuple(sorted((str(a), str(b)))): v for (a, b), v in limits.items()}
        unknown = [k for k in norm if k not in pairs]
        if unknown:
            raise ValueError(f"limits given for non-existent interfaces {unknown}")
        ifc = {k: (_mod29(k) if norm.get(k) == "mod29" else norm.get(k)) for k in pairs}
    else:
        if len(pairs) != 1:
            raise ValueError("a scalar limit needs exactly one interface; pass a dict")
        ifc = {k: float(limits) for k in pairs}
    r = solve_engine_transfers(p, e, fp.defs, ifc, curves=curves, exo=exo,
                               shed_price=shed_price)
    return n, p, e, r


def mod29_interface_limits(fp, pt, loads=None, gen_fleet=None, monitored="all"):
    """The MOD-029-style default released capacities for a partition, with the
    holistic check the path-by-path accounting skips.

    Returns ``(limits, sft)``: ``limits`` a DataFrame per interface (rated TTC,
    first binding line — each rated ALONE, the over-assignment posture), and
    ``sft`` the ``atc.interface_sft`` line-loading table when every interface
    schedules its full rating at once — overloaded rows are capacity the
    ratings assign that the system cannot carry simultaneously (the
    MOD-029-versus-SFT gap, per interface)."""
    W = hub_weights(fp, "auto", loads, gen_fleet)
    pairs = fp_interfaces(fp, pt)
    rated = atc.interface_ttc(pt, pairs, W, monitored=monitored)
    limits = pd.DataFrame({k: {"TTC (MW)": v[0], "first binding line": v[1],
                               "ties": ", ".join(pairs[k])}
                           for k, v in rated.items()}).T
    limits.index.names = ["a", "b"]
    return limits, atc.interface_sft(pt, rated, W, monitored=monitored, direction="worst")


def allocate_congestion_rent(fp, res, pt, loads, unassigned_split=0.5, te_split=None,
                             gen_fleet=None):
    """Allocate total congestion rent to the footprints under both methods
    (any number of footprints; the narrative fields below read cleanest at two).

    **Method 1 — rent follows the transmission element.** Each footprint keeps the rent on the lines
    *assigned* to it (``R_own``); rent on unassigned lines is split by
    ``unassigned_split`` (a :func:`_ba_shares` spec — scalar ``s`` is the
    two-footprint legacy ``s / (1 - s)``, or pass a dict / ``'even'`` /
    ``'interchange'`` for more footprints).

    **Method 2 — rent follows the price impact.** Each footprint keeps its own
    congestion, ``N_a^c = -Σ_m μ_m F_m^a`` measured against its own ex-ante
    aggregation point (:func:`regional_congestion`) — every position paired with
    the footprint's own hub, so the number is reference-bus independent — plus its
    ``te_split`` share of the transfer settlement ``T_E``
    (:func:`transfer_settlement`). ``te_split`` is a :func:`_ba_shares` spec, or
    the string ``'interface'`` for the EDAM rule — each interface's settlement
    ``T_ab`` (:func:`pairwise_transfer_settlement`) split equally between the two
    footprints party to it, which at two footprints is identically the legacy
    50/50; ``None`` leaves ``T_E`` unallocated as the complementary stream the
    transfer methodology handles. Method 1 sums to R; Method 2 with shares sums
    to ``R + R_T``, and with ``te_split=None`` to ``R + R_T - T_E`` — the ledgers
    allocate each method's residual (``pool − Σ allocation``) by the transfer
    rule, so every pairing conserves.

    ``gen_fleet`` must be passed whenever the clearing used a non-default fleet:
    the ex-ante GAP hub weights are built from nameplate data, and defaulting to
    the canonical fleet would settle against the wrong aggregation points.

    (The tie-based :func:`border_separation` is still returned as ``sep`` for the
    tie price-gap decomposition of Section 5, but it does not set Method 2.)"""
    lr, sep = line_rent_table(fp, res, pt), border_separation(fp, res, pt)
    R = lr["rent"].sum()
    R_unassigned = lr[lr.home.isna()]["rent"].sum()
    R_own = {ba: lr[lr.home == ba]["rent"].sum() for ba in fp.names}
    W = hub_weights(fp, "auto", loads, gen_fleet)
    rc = regional_congestion(fp, res, pt, loads, weights=W)
    Nc = {ba: rc[ba]["line_congestion"] for ba in fp.names}   # N_a^c, the write-up's -Σ μ_m F_m^a
    settle = ba_settlement(fp, res, loads)
    hedged_ba = max(fp.names, key=lambda ba: settle[ba]["net_into_pool"])
    funding_ba = min(fp.names, key=lambda ba: settle[ba]["net_into_pool"])

    TE = transfer_settlement(fp, res, loads, weights=W)
    if isinstance(te_split, str) and te_split == "interface":
        pw = pairwise_transfer_settlement(fp, res, pt, loads, weights=W)
        te_alloc = {ba: 0.0 for ba in fp.names}
        for (a, b), row in pw.iterrows():
            te_alloc[a] += 0.5 * float(row["T_ab"])
            te_alloc[b] += 0.5 * float(row["T_ab"])
    else:
        sigma = (_ba_shares(fp, te_split, res, loads) if te_split is not None
                 else {ba: 0.0 for ba in fp.names})
        te_alloc = {ba: sigma[ba] * TE for ba in fp.names}
    u_shares = (_ba_shares(fp, unassigned_split, res, loads) if R_unassigned > 1e-9
                else {ba: 0.0 for ba in fp.names})
    alloc = {ba: dict(R_own=R_own[ba], unassigned_share=u_shares[ba] * R_unassigned,
                      method1=R_own[ba] + u_shares[ba] * R_unassigned,
                      method2=Nc[ba] + te_alloc[ba]) for ba in fp.names}
    tau = alloc[hedged_ba]["method2"] - alloc[hedged_ba]["method1"]   # cross-border rebate to the net-payer
    summary = dict(R=R, R_unassigned=R_unassigned, R_own=R_own, Nc=Nc, TE=TE,
                   R_border=sep["sep_rent"].sum() if len(sep) else 0.0,
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


def revenue_table(fp, res, pt, loads, te_split=None):
    """Two ledgers (one per allocation method), each area's consumers and
    generators side by side with an Area-net column (the autarky-vs-unified
    layout). Pass ``te_split`` (e.g. 0.5) to fold the transfer settlement into
    Method 2 by fixed shares so its ledger conserves without a separate transfer
    row. Returns (method-1 table, method-2 table, summary)."""
    alloc, summ, lr, sep = allocate_congestion_rent(fp, res, pt, loads, te_split=te_split)
    s = ba_settlement(fp, res, loads)
    return _ledger(fp, alloc, s, 1), _ledger(fp, alloc, s, 2), summ


# ──────────────────────────────────────────────────────────────────────────
# Independent operation + aggregation
# ──────────────────────────────────────────────────────────────────────────
def cost_by_bus(gen_fleet=None):
    """``{bus: marginal cost}`` from a fleet (default the canonical teaching fleet)."""
    fleet = _case().gen_fleet if gen_fleet is None else gen_fleet
    return {s["bus"]: s["cost"] for s in fleet.values()}


def _prod_cost(result, bs, cost, curves=None, gen_bus=None):
    """Production cost of the generators sitting in bus set ``bs``.

    Default (``curves=None``) is the flat ``sum_b cost[b]*g_b`` — marginal cost
    times output. With **rising linear MC curves** ``curves = {gen: (a, b)}``
    (``MC_g(g) = a + b*g``) the cost is the **area under marginal cost**,
    ``a*g + ½*b*g²``, summed per generator (``gen_bus = {gen: bus}`` maps each unit
    to its bus; defaults to the canonical fleet). A generator absent from ``curves``
    falls back to its flat ``cost[bus]*g``. Producer surplus is then
    ``LMP*g − (a*g + ½*b*g²)`` — positive for an inframarginal unit, zero only when
    the curve is flat and the unit sets its own price."""
    if not curves:
        return sum(cost.get(b, 0.0) * result.gen_by_bus.get(b, 0.0) for b in bs)
    if gen_bus is None:
        gen_bus = {g: s["bus"] for g, s in _case().gen_fleet.items()}
    bset = {str(x) for x in bs}
    pc = 0.0
    for g, gmw in result.dispatch.items():
        b = str(gen_bus[g])
        if b not in bset:
            continue
        if g in curves:
            a, bb = curves[g]
            pc += a * gmw + 0.5 * bb * gmw * gmw
        else:
            pc += cost.get(b, 0.0) * gmw
    return pc


def _agg(fp, result, area, loads=None, cost=None, curves=None, gen_bus=None):
    """(served-load payment, generator revenue, production cost) for an AREA at
    `result` prices — shed load pays nothing. Pass ``curves`` (``{gen:(a,b)}``,
    ``MC=a+b*g``) so the production cost — and hence producer surplus ``R − PC`` — is
    the area under marginal cost rather than the flat ``cost*g`` (which is degenerate,
    zero surplus, when each unit sets its own price)."""
    loads = _case().loads if loads is None else loads
    cost = cost_by_bus() if cost is None else cost
    bs = fp.areas[area]
    L = sum(result.lmp[b] * (loads.get(b, 0.0) - result.shed_by_bus.get(b, 0.0)) for b in bs)
    R = sum(result.lmp[b] * result.gen_by_bus.get(b, 0.0) for b in bs)
    PC = _prod_cost(result, bs, cost, curves, gen_bus)
    return L, R, PC


def independent_clear(fp, rat, gen_fleet=None, loads=None, shed_price=None, split_5_6=False,
                      curves=None):
    """Each AREA as its own independent engine on the full network — own gens
    serve own load, enforcing ONLY the lines assigned to it (rest relaxed), no
    interchange. A nodal DC-OPF per area; infeasibility is returned as ``None``
    (a finding) unless ``shed_price`` is given (then the area sheds at the
    penalty). Pass ``curves`` (``{gen:(a,b)}``) to clear against **rising linear
    MC curves** (``solve_engine_qp``) instead of flat offers. Returns
    ``(pt, {area: EngineResult|None})``."""
    pt = _cn(rat, split_5_6=split_5_6).pt
    out = {}
    for area, buses in fp.areas.items():
        act = [l for l in pt.lines if fp.line_assign.get(l) == area]
        try:
            eng = _case().make_engine(area, buses, gen_fleet=gen_fleet, loads=loads, activated=act)
            out[area] = (solve_engine_qp(pt, eng, curves=curves, shed_price=shed_price)
                         if curves else solve_engine_dispatch(pt, eng, shed_price=shed_price))
        except (RuntimeError, ValueError):
            out[area] = None
    return pt, out


def autarky_vs_unified(fp, method, alloc, indep, resU, loads=None, cost=None,
                       curves=None, gen_bus=None):
    """Homework-style autarky (independent) vs unified ledger for one allocation
    method. `alloc` is the congestion allocation table, `indep` the per-area
    autarky results, `resU` the unified clearing. Cash out negative; the TOTAL
    column's positions sum to −(production cost). Pass ``curves`` (``{gen:(a,b)}``)
    so generator producer surplus is the LMP payment minus the area under marginal
    cost (non-degenerate) rather than the flat ``Σ(λ−c)·g`` (≈ 0 when each unit sets
    its own price)."""
    data = {}
    for ba in fp.areas:
        La, Ra, PCa = _agg(fp, indep[ba], ba, loads, cost, curves, gen_bus)
        Lu, Ru, PCu = _agg(fp, resU, ba, loads, cost, curves, gen_bus)
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


def _fold_congestion(base, fp, resU, pt, loads, gen_fleet=None):
    """Fold the congestion-revenue derivation into the **Unified** block of a ROWS-shaped
    ledger (the output of ``autarky_vs_unified`` or ``methodology_ledger``) and add a
    ``TOTAL`` column, so the table walks the whole chain in one object. The Unified section
    reads, top to bottom:

      * **Unified: payment / revenue** and **Unified: production cost** (as before).
      * **Unified: congestion rent -- system (R)** -- the pooled line rent the unified
        dispatch collects (``TOTAL`` only; R is a system quantity).
      * **Unified: congestion by region (N_a^c)** -- how R splits across regions by the
        congestion each one's own injections create (``-sum_m mu_m F_m^a``; sums to R).
      * **Unified: Congestion Revenue Allocation** -- how much of R each region is *given*
        under the methodology (sums to R); the renamed ``Congestion rent (CRR)`` line,
        which feeds each consumer's **Final position** below.

    Showing ``N_a^c`` (what a region's prices *create*) next to the allocation (what it
    *keeps*) makes benefit-commensurability visible in the same ledger that tests individual
    rationality. ``resU`` is the unified clearing and ``pt`` the PTDF used for ``N_a^c``."""
    base = base.rename(index={"Congestion rent (CRR)": "Unified: Congestion Revenue Allocation"})
    base[("TOTAL", "")] = [
        round(sum(v for v in base.loc[r]
                  if isinstance(v, (int, float)) and not isinstance(v, bool)), 1)
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in base.loc[r])
        else "" for r in base.index]
    order = list(base.columns)
    rc = regional_congestion(fp, resU, pt, loads, gen_fleet=gen_fleet)
    nc = {a: round(rc[a]["line_congestion"], 1) for a in fp.areas}
    p = {b: resU.gen_by_bus.get(b, 0.0) - (float(loads.get(b, 0.0)) - resU.shed_by_bus.get(b, 0.0))
         for b in pt.buses}
    R = round(-sum(resU.line_dual[m] * sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * p[b]
              for b in pt.buses) for m in pt.lines), 1)
    TE = round(transfer_settlement(fp, resU, loads, gen_fleet=gen_fleet), 1)

    def _row(consumer_vals, total):
        return [total if c == ("TOTAL", "") else
                consumer_vals.get(c[0], "") if c[1] == "Consumer" else "" for c in order]

    idx = ["Unified: congestion rent -- system (R)",
           "Unified: congestion by region (N_a^c)"]
    body = [_row({}, R), _row(nc, R + round(transfer_rent(resU), 1) - TE)]
    if abs(TE) > 0.05:
        idx.append("Unified: transfer settlement (T_E)")
        body.append(_row({}, TE))
        pairs = fp_interfaces(fp, pt)
        if len(pairs) > 1:
            # the per-interface decomposition of T_E (scheduled tags when the
            # clearing carries them, physical cut flows otherwise)
            pw = pairwise_transfer_settlement(fp, resU, pt, loads, gen_fleet=gen_fleet)
            for (a, b), prow in pw.iterrows():
                idx.append(f"  T_ab: {a} to {b}")
                body.append(_row({}, round(float(prow["T_ab"]), 1)))
    extra = pd.DataFrame(body, index=idx, columns=pd.MultiIndex.from_tuples(order))
    out = pd.concat([base, extra])
    return out.reindex(
        ["Autarky: payment / revenue", "Autarky: production cost",
         "Autarky: own congestion rent (CRR)", "Autarky: position",
         "Unified: payment / revenue", "Unified: production cost",
         "Unified: congestion rent -- system (R)",
         "Unified: congestion by region (N_a^c)"]
        + idx[2:]
        + ["Unified: Congestion Revenue Allocation",
           "Final position", "Delta vs autarky", "Pareto (Delta >= 0)"])


def autarky_vs_unified_congestion(fp, method, alloc, indep, resU, pt, *,
                                  loads=None, cost=None, curves=None, gen_bus=None,
                                  gen_fleet=None):
    """``autarky_vs_unified`` (the ``N_a = L_a - G_a`` net-position ledger) with the
    congestion-revenue derivation folded into the Unified block by :func:`_fold_congestion`
    -- the system rent R, its split into ``N_a^c`` by region, and the allocation, above each
    region's consumer/generator Pareto positions. ``method`` / ``alloc`` / ``indep`` / ``resU``
    are as in ``autarky_vs_unified``; ``pt`` is the PTDF used for ``N_a^c``. Pass ``curves``
    (``{gen:(a,b)}``) to make generator producer surplus the area-under-MC form. Pass
    ``gen_fleet`` whenever the clearing uses a NON-DEFAULT fleet: the aggregation-point
    (hub) weights are generation-weighted, so omitting it silently builds each region's
    hub from the canonical fleet's nameplates and mis-sites ``N_a^c`` (the partition
    ``sum_a N_a^c + T_E`` still closes on R -- only the split moves)."""
    loads = _case().loads if loads is None else loads
    return _fold_congestion(
        autarky_vs_unified(fp, method, alloc, indep, resU, loads, cost, curves, gen_bus),
        fp, resU, pt, loads, gen_fleet=gen_fleet)


# ──────────────────────────────────────────────────────────────────────────
# Transmission service: embedded-cost scheduling rates and long-run recovery
# (notebook 203 — provision/appropriation consistency)
# ──────────────────────────────────────────────────────────────────────────
def embedded_rate(RR: float, atc: float) -> float:
    """Long-run scheduling-right rate ``r = RR / ATC`` ($/MW): a transmission
    system's embedded (legacy) revenue requirement ``RR`` spread over the firm
    scheduling capability ``ATC`` it provides. ``ATC = 0`` returns 0."""
    return RR / atc if atc else 0.0


def scheduling_subscription(fp, res, area: str, loads=None) -> float:
    """Firm scheduling-right subscription ``S_a`` a balancing authority's own
    customers buy at a clearing: the MW of the area's load served by the area's
    OWN generation (its internal generator→load delivery). Imports are delivered
    by the joint market without a per-MW embedded charge, so they do not count.

    ``S_a = min(Σ own gen, Σ own load)``. In autarky the area self-supplies, so
    ``S_a`` equals its load; under a joint dispatch that displaces the area's
    generation, ``S_a`` collapses and the embedded cost is stranded.
    """
    loads = _case().loads if loads is None else loads
    g = sum(res.gen_by_bus.get(b, 0.0) for b in fp.areas[area])
    d = sum(float(loads.get(b, 0.0)) for b in fp.areas[area])
    return min(g, d)


def _own_congestion(fp, res, pt, area: str) -> float:
    """Congestion rent |μ|·|F| on the lines this area manages, at ``res`` — the
    arbitrage the area's internal scheduling rights earn (0 if uncongested)."""
    return sum(abs(res.line_dual[l]) * abs(res.flow_own[l])
               for l in pt.lines if fp.line_assign.get(l) == area)


def transmission_service_recovery(fp, indep, resU, pt, *, embedded, atc_by_ba,
                                  loads=None):
    """Per-area embedded-cost recovery, autarky vs joint dispatch.

    For each area: the scheduling rate ``r_a = RR_a / ATC_a``, the firm
    subscription ``S_a`` (``scheduling_subscription``) in autarky (``indep[a]``)
    and under the joint clearing (``resU``), the recovery ``r_a·S_a`` against the
    embedded cost ``RR_a``, and whether it recovers. The consistency test of the
    notebook: both areas recover in autarky, but a joint dispatch that displaces
    one area's generation collapses its ``S`` and strands its embedded cost.
    """
    loads = _case().loads if loads is None else loads
    rows = []
    for a in fp.names:
        RR = float(embedded[a]); ATC = float(atc_by_ba[a])
        r = embedded_rate(RR, ATC)
        S0 = scheduling_subscription(fp, indep[a], a, loads)
        S1 = scheduling_subscription(fp, resU, a, loads)
        rec0, rec1 = r * S0, r * S1
        rows.append({"area": a, "RR (embedded $/h)": round(RR, 0),
                     "ATC (MW)": round(ATC, 0), "rate r ($/MW)": round(r, 2),
                     "S autarky (MW)": round(S0, 0), "S joint (MW)": round(S1, 0),
                     "recovery autarky": round(rec0, 0), "recovery joint": round(rec1, 0),
                     "stranded joint": round(max(0.0, RR - rec1), 0),
                     "recovers joint": rec1 >= RR - 1.0})
    return pd.DataFrame(rows).set_index("area")


def scheduling_service_ledger(fp, indep, resU, pt, *, embedded, atc_by_ba,
                              loads=None, cost=None):
    """Four-party provision/appropriation ledger — autarky vs joint dispatch.

    Per area, four parties (cash out negative):

    * **Generators** — producer surplus Σ(λ−c)·g.
    * **Load** — energy payment −Σ λ·d.
    * **Transmission Customer** — captures the area's congestion-rent arbitrage
      (Method-1: the area keeps its own lines' rent) and pays the scheduling
      charge ``r_a·S_a`` for the right to deliver gen→load.
    * **Transmission Service Provider** — collects ``r_a·S_a`` and pays the
      embedded cost ``RR_a`` (position 0 when it recovers, negative when stranded).

    Returns a MultiIndex table (columns = (area, party)) with the autarky and
    joint positions, the change, and a Pareto flag, plus a TOTAL column. All
    positions sum to ``−(production cost + embedded cost)`` in each state — the
    "positions sum to −production cost" identity extended by the sunk embedded
    cost. Pair with :func:`transmission_service_recovery` for the recovery rows.
    """
    loads = _case().loads if loads is None else loads
    allocU, _, _, _ = allocate_congestion_rent(fp, resU, pt, loads)
    data = {}
    for a in fp.names:
        RR = float(embedded[a]); r = embedded_rate(RR, float(atc_by_ba[a]))
        La, Ra, PCa = _agg(fp, indep[a], a, loads, cost)
        Lu, Ru, PCu = _agg(fp, resU, a, loads, cost)
        S0 = scheduling_subscription(fp, indep[a], a, loads)
        S1 = scheduling_subscription(fp, resU, a, loads)
        arb0 = _own_congestion(fp, indep[a], pt, a)
        arb1 = float(allocU.loc[a, "method1"]) if a in allocU.index else 0.0
        parties = {
            "Generators":  (Ra - PCa,           Ru - PCu),
            "Load":        (-La,                 -Lu),
            "Txn Customer": (arb0 - r * S0,      arb1 - r * S1),
            "TSP":         (r * S0 - RR,         r * S1 - RR),
        }
        for party, (base, joint) in parties.items():
            data[(a, party)] = {
                "Autarky position": base, "Joint position": joint,
                "Change (joint - autarky)": joint - base,
                "Pareto (change >= 0)": "yes" if joint - base >= -1e-6 else "no"}
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    tot = df.map(lambda v: v if isinstance(v, (int, float)) else 0.0).sum(axis=1)
    tot = tot.astype(object); tot["Pareto (change >= 0)"] = ""
    df[("TOTAL", "")] = tot
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v)


# ──────────────────────────────────────────────────────────────────────────
# Transfer rent
# ──────────────────────────────────────────────────────────────────────────
def transfer_rent(res):
    """R_T — the transfer-scheduling rent: the sum of every interface's
    ``|μ_T^{ab} · t_ab|`` on an EDAM-literal result
    (:func:`transfer_rent_by_interface`); ``0.0`` for a clearing with no
    scheduled transfers (a pooled or single-area result)."""
    if getattr(res, "transfer_duals", None):
        return float(sum(abs(res.transfer_duals[k] * res.transfers.get(k, 0.0))
                         for k in res.transfer_duals))
    return 0.0


def transfer_rent_by_interface(res):
    """``{(a, b): |μ_T^{ab} · t_ab|}`` per interface — the released-capacity
    rent each binding interface collects (empty for a pooled result)."""
    if not getattr(res, "transfer_duals", None):
        return {}
    return {k: abs(res.transfer_duals[k] * res.transfers.get(k, 0.0))
            for k in res.transfer_duals}


def allocate_transfer_rent(fp, res, loads, t_method, transfer_split, stream=None,
                           pt=None, gen_fleet=None):
    """σ_a by transfer methodology: T1 = fixed shares (``transfer_split``, EDAM's
    equal split stated as one share per footprint); T2 = the whole stream to the
    NET-PAYER footprint (the transfer analogue of congestion Method 2); T3 = the
    EDAM per-interface rule — each interface's dollars split equally between the
    two footprints party to it, N-footprint-ready (requires ``pt``). ``stream``
    is the dollars being shared -- default the transfer rent ``R_T``; the
    ledgers pass each congestion method's residual (``pool − Σ congestion
    allocation``: ``R_T`` under Method 1, the transfer settlement ``T_E`` under
    Method 2) so every pairing conserves. T3 reads the residual's per-interface
    decomposition (``R_T^{ab}`` when the stream is the scheduling rent, ``T_ab``
    when it is the transfer settlement — whichever total the stream matches),
    takes half of each interface's dollars to each party, and rescales exactly
    to the stream. Returns ``{footprint: $}``."""
    RT = transfer_rent(res) if stream is None else float(stream)
    if t_method == 1:
        # Route through _ba_shares so every documented spec form reaches T1
        # ('even' / 'interchange' / scalar / dict). A dict spec returns its own
        # values unchanged, so existing call sites are numerically untouched;
        # before this, T1 indexed the spec directly and the non-dict forms
        # raised. (Found by the phase-0 golden harness, fixed in phase 4.)
        sig = _ba_shares(fp, transfer_split, res, loads)
        return {ba: sig[ba] * RT for ba in fp.names}
    if t_method == 2:
        s = ba_settlement(fp, res, loads)
        payer = max(fp.names, key=lambda ba: s[ba]["net_into_pool"])
        return {ba: (RT if ba == payer else 0.0) for ba in fp.names}
    if t_method == 3:
        if pt is None:
            raise ValueError("t_method=3 (per-interface halves) needs pt")
        rti = transfer_rent_by_interface(res)
        TE = transfer_settlement(fp, res, loads,
                                 weights=hub_weights(fp, "auto", loads, gen_fleet))
        if rti and abs(RT - sum(rti.values())) <= abs(RT - TE):
            per = dict(rti)
        else:
            pw = pairwise_transfer_settlement(fp, res, pt, loads, gen_fleet)
            per = {k: float(v) for k, v in pw["T_ab"].items()}
        halves = {ba: 0.0 for ba in fp.names}
        # An interface may name a party that is NOT a footprint of this
        # allocation — a pure-junction BA left out of the overlay, or a
        # price-taking boundary area. Its half has no claimant here, so the
        # in-universe party keeps the interface's dollars rather than the
        # allocation raising KeyError on a real network.
        for (a, b), v in per.items():
            ina, inb = a in halves, b in halves
            if ina and inb:
                halves[a] += 0.5 * v
                halves[b] += 0.5 * v
            elif ina:
                halves[a] += v
            elif inb:
                halves[b] += v
        tot = sum(halves.values())
        if abs(tot) < 1e-9:
            return {ba: RT / len(fp.names) for ba in fp.names}
        return {ba: halves[ba] / tot * RT for ba in fp.names}
    raise ValueError(f"unknown transfer methodology {t_method!r}")


def transfer_ledger(fp, res, p, loads, method, t_method, indep, transfer_split, cost=None,
                    curves=None, gen_bus=None, gen_fleet=None):
    """Autarky vs the transfer-constrained clearing — the per-area
    Consumer/Generator layout with both revenue streams shown COLLECTED
    separately from ALLOCATED (congestion by Method `method`; transfer by
    `t_method`, incl. T3 per-interface halves). `res` is an EDAM-literal
    clearing (``solve_engine_transfers`` — the pool carries every interface's
    scheduling rent) or a pooled one with no transfers. `indep` =
    autarky results on the SAME ratings. TOTAL positions sum to −(production
    cost). Pass ``curves`` (``{gen:(a,b)}``) so generator producer surplus is
    the LMP payment minus the area under marginal cost."""
    alloc, summ, lr, sep = allocate_congestion_rent(fp, res, p, loads, gen_fleet=gen_fleet)
    col = "method1" if method == 1 else "method2"
    RT = transfer_rent(res)
    pool = summ["R"] + RT
    residual = pool - float(alloc[col].sum())      # R_T under M1; T_E under M2 -- conserves either way
    AT_map = allocate_transfer_rent(fp, res, loads, t_method, transfer_split, stream=residual,
                                    pt=p, gen_fleet=gen_fleet)
    data = {}
    for ba in fp.areas:
        assert indep[ba] is not None, (f"{ba} has no autarky baseline -- pass "
                                       "shed_price=SHED_PRICE to independent_clear")
        La, Ra, PCa = _agg(fp, indep[ba], ba, loads, cost, curves, gen_bus)
        Lc, Rc, PCc = _agg(fp, res, ba, loads, cost, curves, gen_bus)
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
            "Transfer settlement collected": "",
            "Transfer settlement allocated (CRR)": AT,
            "Final position": cons_fin,
            "Delta vs autarky": cons_fin - cons_aut,
            "Pareto (Delta >= 0)": "yes" if cons_fin - cons_aut >= -1e-6 else "no"}
        data[(ba, "Generator")] = {
            "Autarky: payment / revenue": Ra, "Autarky: production cost": -PCa,
            "Autarky: own congestion rent (CRR)": "", "Autarky: position": gen_aut,
            "Constrained: payment / revenue": Rc, "Constrained: production cost": -PCc,
            "Congestion rent collected (assigned lines)": "",
            "Congestion rent allocated (CRR)": "",
            "Transfer settlement collected": "",
            "Transfer settlement allocated (CRR)": "",
            "Final position": gen_fin,
            "Delta vs autarky": gen_fin - gen_aut,
            "Pareto (Delta >= 0)": "yes" if gen_fin - gen_aut >= -1e-6 else "no"}
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    tot = df.map(lambda v: v if isinstance(v, (int, float)) else 0.0).sum(axis=1)
    tot["Congestion rent collected (assigned lines)"] = summ["R"]
    tot["Transfer settlement collected"] = residual
    tot = tot.astype(object)
    tot["Pareto (Delta >= 0)"] = ""
    df[("TOTAL", "")] = tot
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v)


# ──────────────────────────────────────────────────────────────────────────
# Self-schedule incentive (how an allocation RULE changes behaviour)
# ──────────────────────────────────────────────────────────────────────────
def self_schedule_ledger(fp, ebar, source, sink, mw, *, ratings=None,
                         gen_fleet=None, loads=None):
    """Economic dispatch vs. a firm-rights self-schedule, as a conserving ledger.

    The released capacity ``ebar`` on the footprints' interface caps the scheduled transfer, so
    the exporting area sits at its cheap unit while the importing area sits at its
    dear unit -- the price gap a firm right spans. A resource at bus ``source``
    whose LMP sits below its cost is OFF in the economic clearing. A balanced firm
    self-schedule of ``mw`` from ``source`` to ``sink`` (modelled as the
    price-taking injection ``exo={source: +mw}``; because export is capped, the
    cheap exporter backs down by ``mw``) forces it ON. Under a rule that REBATES
    the schedule's out-of-area congestion charge ``(lmp_sink - lmp_source)*mw``,
    the owner pockets that rebate, so self-scheduling is privately profitable even
    though it raises production cost.

    Returns ``(ledger, summary)``. ``ledger`` is a per-party table (Economic /
    Self-schedule / Change) whose non-TOTAL positions sum to ``-(production cost)``
    in each column -- the conservation check. ``summary`` carries the dispatch
    shift, the private gain, the deadweight (extra production cost), and the
    rebate drawn from the rent pool.

    The scenario is built on its OWN network (``ratings`` line limits, default the
    base ratings) rather than any externally mutated shift factors, so the result
    is reproducible whatever clearing the notebook ran before.
    """
    loads = _case().loads if loads is None else loads
    fleet = _case().gen_fleet if gen_fleet is None else gen_fleet
    src, snk, S = str(source), str(sink), float(mw)
    src_cost = min(g["cost"] for g in fleet.values() if str(g["bus"]) == src)
    pt = _cn(ratings or {}).pt
    ifc = {k: float(ebar) for k in fp_interfaces(fp, pt)}
    mk = lambda: _case().make_engine("UNIFIED", buses=pt.buses, gen_fleet=fleet, loads=loads)
    base = solve_engine_transfers(pt, mk(), fp.defs, ifc)
    ss = solve_engine_transfers(pt, mk(), fp.defs, ifc, exo={src: +S})

    def positions(r, selfsched):
        gen_pay_src = r.lmp[src] * S if selfsched else 0.0
        rebate = (r.lmp[snk] - r.lmp[src]) * S if selfsched else 0.0
        out = {}
        for ba in fp.names:
            out[f"{ba} consumers"] = -sum(
                r.lmp[b] * (float(loads.get(b, 0.0)) - (S if b == snk else 0.0))
                for b in fp.defs[ba])
            out[f"{ba} generators"] = sum(
                (r.lmp[str(fleet[g]["bus"])] - fleet[g]["cost"]) * q
                for g, q in r.dispatch.items() if str(fleet[g]["bus"]) in fp.defs[ba])
        out["Self-schedule (firm right)"] = (
            gen_pay_src - src_cost * S - r.lmp[snk] * S + rebate
            if selfsched else -r.lmp[snk] * S)
        Ld = sum(r.lmp[b] * float(loads.get(b, 0.0)) for b in pt.buses)
        Gp = sum(r.lmp[b] * r.gen_by_bus.get(b, 0.0) for b in pt.buses)
        out["Congestion + transfer rent pool"] = (Ld - Gp - gen_pay_src) - rebate
        prod = r.total_cost + (src_cost * S if selfsched else 0.0)
        out["TOTAL (= -production cost)"] = -prod
        return out, prod, rebate

    pe, prod_e, _ = positions(base, False)
    ps, prod_s, rebate = positions(ss, True)
    ledger = pd.DataFrame({"Economic": pe, "Self-schedule": ps})
    ledger["Change"] = ledger["Self-schedule"] - ledger["Economic"]
    out_self = dict(ss.gen_by_bus)
    out_self[src] = out_self.get(src, 0.0) + S          # add the price-taking self-schedule
    summary = dict(
        source=src, sink=snk, mw=S, src_cost=src_cost,
        lmp_source=ss.lmp[src], lmp_sink=ss.lmp[snk], gap=ss.lmp[snk] - ss.lmp[src],
        private_gain=ps["Self-schedule (firm right)"] - pe["Self-schedule (firm right)"],
        deadweight=prod_s - prod_e, rebate=rebate,
        pool_econ=pe["Congestion + transfer rent pool"],
        pool_self=ps["Congestion + transfer rent pool"],
        prod_econ=prod_e, prod_self=prod_s,
        out_econ=dict(base.gen_by_bus), out_self=out_self,
        bus_cost={str(g["bus"]): g["cost"] for g in fleet.values()},
    )
    return ledger.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v), summary


# ──────────────────────────────────────────────────────────────────────────
# BA net congestion position (the "outcomes" allocation) + a general
# methodology ledger (autarky vs a clearing, for any allocation)
# ──────────────────────────────────────────────────────────────────────────
def net_congestion_position(fp, res, loads=None, weights=None, gen_fleet=None):
    """Each BA's net congestion position, the congestion *outcome* at its nodes,
    measured against the BA's OWN ex-ante aggregation point (hub).

    ``N_a^c = sum_{n in a} (lambda_n - lambda_a)(d_n - g_n)`` with
    ``lambda_a = sum_n w_n lmp_n`` the BA's energy price at its aggregation point
    -- equivalently ``-sum_m mu_m F_m^{a}`` with hub-relative shift factors
    ``SF_{n,m} - SF^a_m``. Every term is a price difference, so the number is
    independent of the power-flow calculation's reference bus. The aggregation
    points are configured EX ANTE from the same inputs that define the BAs (bus
    membership + nameplate / load data) via `hub_weights`; ``weights=None`` uses
    generation-weighted hubs where a BA has generation and load-weighted hubs
    otherwise. The scheduled transfer settles separately, hub-to-hub, as the transfer
    settlement (``hub_congestion_decomposition``); together they satisfy
    ``sum_a N_a^c + T_E = R + R_T``. This is the "make the BA whole" (outcomes)
    allocation -- distinct from giving each line's rent to the BA that owns it.
    Returns ``{ba: $/h}``.
    """
    loads = _case().loads if loads is None else loads
    if weights is None:
        weights = hub_weights(fp, "auto", loads, gen_fleet)
    out = {}
    for ba in fp.names:
        lam_a = sum(w * res.lmp[b] for b, w in weights[ba].items())
        out[ba] = sum((res.lmp[b] - lam_a) * (float(loads.get(b, 0.0)) - res.gen_by_bus.get(b, 0.0))
                      for b in fp.defs[ba])
    return out


def regional_congestion(fp, res, pt, loads=None, weights=None, gen_fleet=None):
    """Unpack each region's load bill into the generation it pays for and the
    scarcity rent it bears, each region measured against its OWN ex-ante
    aggregation point (hub).

    With hub weights ``w`` on each region's own buses (``hub_weights``: the same
    inputs that define the region -- bus membership plus nameplate / load data),
    the hub price is ``lambda_a = sum_n w_n lmp_n`` and each region's net payment
    decomposes exactly:

        L_a = G_a + lambda_a*e_a + N_a^c,

      L_a    load payment (lmp . d)              -- what region a's load pays
      G_a    generator revenue WITHIN a (lmp . g)-- gens physically in a
      e_a    net import (d - g) [MW]             -- MW drawn from the rest of the system
      lambda_a*e_a   the net import valued at the region's OWN hub price
      N_a^c  = sum_{n in a}(lmp_n - lambda_a)(d_n - g_n)
             = -sum_m mu_m F_m^a,  F_m^a = sum_{n in a}(SF_{n,m} - SF^a_m)(g_n - d_n)
                   -- the region's own congestion: every position paired with its
                   own hub, so every term is a price difference and the number is
                   independent of the power-flow reference bus. A binding
                   transfer's price adder is constant across the region's buses
                   and cancels inside (lmp_n - lambda_a).

    The scheduled transfer settles separately, hub-to-hub, as the transfer
    settlement ``T_E = sum_a lambda_a e_a`` (`transfer_settlement`); together
    ``sum_a N_a^c + T_E = R + R_T``. Returns ``{region: dict(load_pay,
    gen_rev_within, net_import, hub_price, energy_payment, gen_payment,
    line_congestion, transfer, congestion)}`` with ``congestion = line_congestion
    = N_a^c`` and ``transfer = 0.0`` (deprecated key -- the transfer settles
    system-wide as ``T_E``, not per region)."""
    loads = _case().loads if loads is None else loads
    if weights is None:
        weights = hub_weights(fp, "auto", loads, gen_fleet)
    out = {}
    for a in fp.names:
        lam_a = sum(w * res.lmp[b] for b, w in weights[a].items())
        L = G = e = nc = 0.0
        for b in fp.defs[a]:
            d = float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0)
            g = res.gen_by_bus.get(b, 0.0)
            L += res.lmp[b] * d
            G += res.lmp[b] * g
            e += d - g
            nc += (res.lmp[b] - lam_a) * (d - g)                    # N_a^c against the own hub
        out[a] = dict(load_pay=L, gen_rev_within=G, net_import=e, hub_price=lam_a,
                      energy_payment=lam_a * e, gen_payment=G + lam_a * e,
                      line_congestion=nc, transfer=0.0, congestion=nc)
    return out


def transfer_settlement(fp, res, loads=None, weights=None, gen_fleet=None):
    """The transfer settlement ``T_E = sum_a lambda_a e_a`` (two areas:
    ``E * (lambda_2 - lambda_1)``): the scheduled interchange bought at the
    exporter's hub and sold at the importer's. Carries the intertie scheduling
    limit's rent when the schedule binds, plus the line-congestion value of the
    transfer's own hub-to-hub flow -- so it is generally nonzero even when the
    scheduling limit is slack, and negative when that flow is net counterflow.
    Conservation: ``sum_a N_a^c + T_E = R + R_T``."""
    loads = _case().loads if loads is None else loads
    if weights is None:
        weights = hub_weights(fp, "auto", loads, gen_fleet)
    te = 0.0
    for a in fp.names:
        lam_a = sum(w * res.lmp[b] for b, w in weights[a].items())
        e_a = sum((float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0))
                  - res.gen_by_bus.get(b, 0.0) for b in fp.defs[a])
        te += lam_a * e_a
    return te


def congestion_shift_breakdown(fp, res, pt, loads=None):
    """Per-bus shift-factor build-up of the congestion revenue by region $N_a^c$.

    For each binding line ``m`` a bus contributes ``-mu_m * SF_{m,n} * (g_n - d_n)`` to
    $N_a^c$: its net injection ``g - d`` scaled by its shift factor onto line ``m`` and the
    shadow price ``mu_m``. A bus that pushes power ONTO the constrained line is a causer (+);
    one that counter-flows it is a reliever (-). Summed over a region's buses this is
    ``-mu_m * F_m^a = N_a^c`` -- the **network** (shift-factor) form of $N_a^c$, equal to the
    **price** form ``sum_{n in a}(lambda_n - lambda)(d_n - g_n)``. Returns a DataFrame: one row
    per injecting bus (region, net injection, the shift factor and flow contribution on each
    binding line, congestion $/h), plus a ``"<region> total"`` row whose flow column is
    ``F_m^a`` and whose congestion column is ``N_a^c``.
    """
    loads = _case().loads if loads is None else loads
    weights = hub_weights(fp, "auto", loads)
    binding = [m for m in pt.lines if abs(res.line_dual[m]) > 1e-3]
    sfa = {ba: {m: sum(w * pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] for b, w in weights[ba].items())
                for m in binding} for ba in fp.names}
    recs, E_a = [], {}
    for ba in fp.names:
        sub = {"bus": f"{ba} total", "region": "", "net inj g-d (MW)": 0.0}
        for m in binding:
            sub[f"flow on {m} (MW)"] = 0.0
        sub["congestion N_a^c ($/h)"] = 0.0
        bus_recs, E_a[ba] = [], 0.0
        for b in fp.defs[ba]:
            d = float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0)
            pn = res.gen_by_bus.get(b, 0.0) - d                          # net injection g - d
            E_a[ba] += pn
            if abs(pn) < 1e-6:
                continue
            rec = {"bus": b, "region": ba, "net inj g-d (MW)": pn}
            cong = 0.0
            for m in binding:
                sf = pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] - sfa[ba][m]   # hub-relative
                rec[f"SF to {m}"] = sf
                rec[f"flow on {m} (MW)"] = sf * pn
                cong += -res.line_dual[m] * sf * pn
                sub[f"flow on {m} (MW)"] += sf * pn
            rec["congestion N_a^c ($/h)"] = cong
            sub["net inj g-d (MW)"] += pn
            sub["congestion N_a^c ($/h)"] += cong
            bus_recs.append(rec)
        recs += bus_recs + [sub]
    xfer = {"bus": "Transfer (hub-to-hub)", "region": "", "net inj g-d (MW)": 0.0}
    tcong = 0.0
    for m in binding:
        fe = sum(sfa[ba][m] * E_a[ba] for ba in fp.names)                # F^E_m
        xfer[f"flow on {m} (MW)"] = fe
        tcong += -res.line_dual[m] * fe
    xfer["congestion N_a^c ($/h)"] = tcong
    return pd.DataFrame(recs + [xfer]).set_index("bus").fillna("")


def hub_weights(fp, kind="auto", loads=None, gen_fleet=None):
    """Ex-ante aggregation-point weights ``{footprint: {bus: w}}`` (each summing to 1).

    The aggregation points are derived from the SAME inputs that define the
    balancing authorities -- the bus membership lists plus nameplate / load data
    -- so declaring the footprints declares their hubs. ``kind='gap'`` --
    nameplate-generation-weighted (the footprint's Generation Aggregation Point);
    ``kind='elap'`` -- load-weighted (its Load Aggregation Point); ``kind='auto'``
    (default) -- GAP where the footprint has generation, ELAP otherwise. Weights
    are FIXED data (``p_nom`` / load levels), never dispatch outcomes, so the hub
    definition cannot be moved by the clearing it settles -- the same discipline
    as EDAM's predefined distribution factors."""
    loads = _case().loads if loads is None else loads
    fleet = _case().gen_fleet if gen_fleet is None else gen_fleet
    if kind not in ("auto", "gap", "elap"):
        raise ValueError(f"unknown hub kind {kind!r} (use 'auto', 'gap' or 'elap')")
    out = {}
    for ba in fp.names:
        gen_raw = {}
        for g in fleet.values():
            b = str(g["bus"])
            if b in fp.defs[ba]:
                gen_raw[b] = gen_raw.get(b, 0.0) + float(g["p_nom"])
        load_raw = {b: float(loads.get(b, 0.0)) for b in fp.defs[ba]}
        if kind == "gap" or (kind == "auto" and sum(gen_raw.values()) > 0):
            raw = gen_raw
        else:
            raw = load_raw
        tot = sum(raw.values())
        if tot <= 0:
            raise ValueError(f"{ba} has neither generation nor load to weight a hub")
        out[ba] = {b: v / tot for b, v in raw.items() if v > 0}
    return out


def hub_congestion_decomposition(fp, res, pt, weights=None, loads=None):
    """Slack-invariant per-area congestion decomposition on the EDAM per-BAA-price
    convention: each area settled against its OWN aggregation point (hub), the
    interchange settled hub-to-hub.

    With hub weights ``w`` on each area's own buses, the hub price is
    ``lambda_a = sum_n w_n lmp_n`` and the decomposition is

        intra:  N_a^{c,w} = sum_{n in a} (lmp_n - lambda_a)(d_n - g_n)
                          = -sum_m mu_m F_m^{a,w}
        transfer settlement:  T_E = sum_a lambda_a e_a   (two areas: E * (lambda_2 - lambda_1))
        partition:  sum_a N_a^{c,w} + T_E = R + R_T.

    Every term is a difference of LMPs (equivalently of shift factors), so the
    computational slack bus drops out of every number -- unlike the global-lambda
    ``N_a^c`` of `net_congestion_position` / `regional_congestion`, which shifts by
    ``(lambda' - lambda) * E_a`` for an area with nonzero interchange (the
    reference-bus ambiguity). A binding transfer's price adder is constant across
    an area's buses and cancels inside ``(lmp_n - lambda_a)``, so the intra term is
    pure line congestion; the transfer settlement carries the tie-congestion
    differential plus the intertie scheduling limit's rent. The ONE remaining
    convention is the hub definition: switching area ``a``'s hub moves its
    component by ``E_a * (lambda_a' - lambda_a)``, offset one-for-one in the
    transfer settlement -- bounded by the area's internal price spread, and an
    explicit design choice (ELAP vs GAP vs sub-hubs; the intertie-modeling
    question), not a solver artifact.

    ``weights`` defaults to `hub_weights(fp, 'auto', loads)`. Returns a DataFrame
    with one column per area plus ``Transfer (hub-to-hub)`` and ``TOTAL``; the TOTAL of
    the congestion row equals the collected ``R + R_T`` (conservation check row)."""
    loads = _case().loads if loads is None else loads
    weights = hub_weights(fp, "auto", loads) if weights is None else weights
    p = {b: res.gen_by_bus.get(b, 0.0)
            - (float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0))
         for b in pt.buses}                                   # net injection g - d (served)
    lam_a, E_a, intra = {}, {}, {}
    for ba in fp.names:
        w = weights[ba]
        lam_a[ba] = sum(w[b] * res.lmp[b] for b in w)
        E_a[ba] = sum(p[b] for b in fp.defs[ba])
        intra[ba] = sum((res.lmp[b] - lam_a[ba]) * (-p[b]) for b in fp.defs[ba])
    xfer = sum(lam_a[ba] * (-E_a[ba]) for ba in fp.names)     # transfer settlement T_E = sum_a lambda_a e_a
    R = -sum(res.line_dual[m] * sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * p[b]
             for b in pt.buses) for m in pt.lines)
    RT = transfer_rent(res)
    rows = {
        "Hub price lambda_a ($/MWh)": {**{ba: lam_a[ba] for ba in fp.names},
                                       "Transfer (hub-to-hub)": "", "TOTAL": ""},
        "Net export E_a (MW)": {**{ba: E_a[ba] for ba in fp.names},
                                "Transfer (hub-to-hub)": "", "TOTAL": sum(E_a.values())},
        "Congestion ($/h)": {**{ba: intra[ba] for ba in fp.names},
                             "Transfer (hub-to-hub)": xfer,
                             "TOTAL": sum(intra.values()) + xfer},
        "Collected R + R_T ($/h)": {**{ba: "" for ba in fp.names},
                                    "Transfer (hub-to-hub)": "", "TOTAL": R + RT},
    }
    df = pd.DataFrame(rows).T[list(fp.names) + ["Transfer (hub-to-hub)", "TOTAL"]]
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) else v)


def congestion_summary(fp, res, pt, allocations=None, loads=None, cost=None):
    """Per-region congestion accounting that pairs what the DISPATCH produces with what a
    METHODOLOGY allocates -- the frame for comparing allocation rules.

    Rows (columns are the regions plus a ``TOTAL``):

    * **Congestion rent -- system (R)** -- the total line rent the dispatch collects (in the
      ``TOTAL`` column; blank by region, because R is a system quantity).
    * **Congestion by region (N_a^c)** -- each region's net congestion position
      ``sum_{n in a}(lambda_n - lambda)(d_n - g_n) = -sum_m mu_m F_m^a``; these PARTITION R.
    * **Transfer by region (N_a^T)** -- each region's transfer scarcity (sums to ``R_T``); only
      shown when a transfer constraint binds.
    * **Generation cost (by region)** -- production cost ``sum c_i g_i`` of the generators
      located in each region.
    * **Allocated: <name>** -- one row per entry of ``allocations`` (a ``{name: {ba: $}}`` map),
      the rent each region is *given* under that rule (e.g. ownership ``R_a`` vs outcomes
      ``N_a^c``); the policy choice, set against the ``N_a^c`` the region actually creates.

    ``N_a^c`` (what a region's prices create) is the natural benchmark; each ``Allocated`` row is a
    rule's answer for how much of R that region keeps. Returns a rounded DataFrame.
    """
    loads = _case().loads if loads is None else loads
    cost = cost_by_bus() if cost is None else cost
    rc = regional_congestion(fp, res, pt, loads)
    p = {b: res.gen_by_bus.get(b, 0.0) - (float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0))
         for b in pt.buses}
    R = -sum(res.line_dual[m] * sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * p[b]
             for b in pt.buses) for m in pt.lines)
    RT = transfer_rent(res)
    TE = transfer_settlement(fp, res, loads)
    rows, totals = {}, {}
    rows["Congestion rent -- system (R)"] = {a: "" for a in fp.names}; totals["Congestion rent -- system (R)"] = R
    if RT > 1e-6:
        rows["Transfer rent -- system (R_T)"] = {a: "" for a in fp.names}; totals["Transfer rent -- system (R_T)"] = RT
    rows["Congestion by region (N_a^c)"] = {a: rc[a]["line_congestion"] for a in fp.names}
    if abs(TE) > 1e-6:
        rows["Transfer settlement (T_E)"] = {a: "" for a in fp.names}; totals["Transfer settlement (T_E)"] = TE
    rows["Generation cost (by region)"] = {a: _agg(fp, res, a, loads, cost)[2] for a in fp.names}
    for name, alloc in (allocations or {}).items():
        rows[f"Allocated: {name}"] = {a: float(alloc.get(a, 0.0)) for a in fp.names}
    df = pd.DataFrame(rows).T[list(fp.names)]
    df["TOTAL"] = [totals.get(r, sum(v for v in df.loc[r] if isinstance(v, (int, float)))) for r in df.index]
    return df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)


def loop_flow_triangle(fp, ratings, host, nbr, *, gen_fleet=None, loads=None,
                       cost=None, shed_price=150.0):
    """The "circular finger-pointing" triangle for a constraint internal to ``host``.

    One unified clearing on the ``ratings`` network. For each binding line the host manages,
    split the flow into the host's own contribution and the NEIGHBOUR's parallel flow,
    ``F_m = F_m^host + F_m^nbr`` with ``F_m^a = sum_{n in a} SF_{m,n}(g_n - d_n)``, and read the
    one shadow price ``mu_m`` three ways -- the three claims that point in a circle:

      * **host -> nbr** (pay for the neighbour's parallel flow on the host's line):
        ``-sum_m mu_m F_m^nbr``;
      * **nbr -> host** (compensate the price separation the constraint makes at the
        neighbour's nodes): ``N_nbr^c = sum_{n in nbr}(lambda_n - lambda)(d_n - g_n)``;
      * **host redispatch** (the production-cost penalty of hosting the neighbour's parallel
        flow versus a "no parallel flow" baseline in which the host had the full line):
        ``PC(ratings) - PC(ratings relaxed by F_m^nbr)``.

    All three coincide: ``N_nbr^c = -sum_m mu_m F_m^nbr`` identically (the price deviation at a
    bus IS the shift-factor-weighted shadow price), and the marginal redispatch cost of a
    binding line is by definition its shadow price -- so the finger-pointing is three names for
    one scarcity, which is the clean case for a "no one pays anyone" netting. Returns
    ``(df, summary)``: ``df`` is the three-claim table ($/h); ``summary`` carries the common value,
    ``R``, and per-binding-line ``mu`` / ``F_host`` / ``F_nbr``."""
    loads = _case().loads if loads is None else loads
    cost = cost_by_bus(gen_fleet) if cost is None else cost
    pt = _cn(ratings or {}).pt
    res = solve_engine_dispatch(pt, _case().make_engine("UNIFIED", buses=pt.buses,
                                gen_fleet=gen_fleet, loads=loads), shed_price=shed_price)
    lam = float(res.energy_price)
    p = {b: res.gen_by_bus.get(b, 0.0) - (float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0))
         for b in pt.buses}
    binding = [m for m in pt.lines if abs(res.line_dual[m]) > 1e-3 and fp.line_assign.get(m) == host]
    F_host = {m: sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * p[b] for b in fp.defs[host]) for m in binding}
    F_nbr  = {m: sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * p[b] for b in fp.defs[nbr])  for m in binding}
    leg_hn = -sum(res.line_dual[m] * F_nbr[m] for m in binding)
    leg_nh = sum((res.lmp[b] - lam) * (float(loads.get(b, 0.0)) - res.shed_by_bus.get(b, 0.0)
                 - res.gen_by_bus.get(b, 0.0)) for b in fp.defs[nbr])
    relaxed = dict(ratings or {})
    for m in binding:                                     # give the host the capacity the nbr consumed
        s = float(pt.s_nom[pt.line_idx[m]])
        relaxed[m] = s + F_nbr[m] * (1.0 if (F_host[m] + F_nbr[m]) >= 0 else -1.0)
    pt2 = _cn(relaxed).pt
    res2 = solve_engine_dispatch(pt2, _case().make_engine("UNIFIED", buses=pt2.buses,
                                 gen_fleet=gen_fleet, loads=loads), shed_price=shed_price)
    PC = lambda r, p_: sum(cost.get(b, 0.0) * r.gen_by_bus.get(b, 0.0) for b in p_.buses)
    leg_rd = PC(res, pt) - PC(res2, pt2)
    R = sum(abs(res.line_dual[l]) * abs(res.flow_own[l]) for l in pt.lines)
    df = pd.DataFrame([
        {"claim": f"{host} -> {nbr}: pay for {nbr}'s parallel flow on {host}'s line",
         "measured as": "-sum_m mu_m F_m^nbr", "$/h": round(leg_hn, 1)},
        {"claim": f"{nbr} -> {host}: compensate {nbr}'s price separation",
         "measured as": "N_nbr^c = sum (lmp-lambda)(d-g)", "$/h": round(leg_nh, 1)},
        {"claim": f"{host} redispatch to host {nbr}'s parallel flow",
         "measured as": "PC(rat) - PC(rat relaxed by F_m^nbr)", "$/h": round(leg_rd, 1)},
    ]).set_index("claim")
    summary = dict(R=float(round(R, 1)), common=float(round(leg_hn, 1)),
                   mu={m: float(round(res.line_dual[m], 2)) for m in binding},
                   F_host={m: float(round(F_host[m], 2)) for m in binding},
                   F_nbr={m: float(round(F_nbr[m], 2)) for m in binding}, binding=binding, res=res, pt=pt)
    return df, summary


def rights_makewhole(fp, ratings, rights, *, redispatch, gen_fleet=None, loads=None,
                     shed_price=150.0):
    """Can a transmission service provider make its firm-rights holders whole -- from congestion
    revenue alone, or only once the redispatch that coordination *avoids* is also credited?

    One unified clearing on ``ratings``. Each right ``{'source','sink','mw'}`` is valued at its
    perfect hedge ``H = mw*(lmp_sink - lmp_source)``; positive-hedge rights are the claims the TSP
    owes, grouped by the **sink's** balancing authority (the area whose customer holds the right).
    For each congestion-revenue allocation methodology -- **ownership** (``R_a``, the host keeps
    its own lines' rent), **outcomes** (``N_a^c``, the area's congestion position), and **hybrid**
    (settle the right financially: pay each holder its hedge from the system rent ``R``, pro-rata
    if ``R`` is short) -- the area's holders are tested against two funding bases:

      * **congestion revenue alone**: ``owed_a <= A_a``;
      * **congestion revenue + avoided redispatch**: ``owed_a <= A_a + redispatch[a]``, where
        ``redispatch[a]`` is the TSP-borne redispatch premium the area bore in autarky/bilateral
        (e.g. :func:`bilateral_self_solve`) and no longer bears under coordination.

    Returns a DataFrame indexed by ``(method, area)`` with the owed hedge, the methodology's
    allocation, the avoided redispatch, and the two whole/short verdicts -- the ledger that shows
    when coordination's efficiency saving, not the rent, is what funds the firm right."""
    loads = _case().loads if loads is None else loads
    pt = _cn(ratings or {}).pt
    res = solve_engine_dispatch(pt, _case().make_engine("UNIFIED", buses=pt.buses,
                                gen_fleet=gen_fleet, loads=loads), shed_price=shed_price)
    lmp = res.lmp
    owed = {a: 0.0 for a in fp.names}
    for r in rights:
        H = float(r["mw"]) * (lmp[str(r["sink"])] - lmp[str(r["source"])])
        if H > 1e-9:
            owed[fp.fp_of(str(r["sink"]))] += H
    rc = regional_congestion(fp, res, pt, loads)
    R = sum(rc[a]["line_congestion"] for a in fp.names)
    owed_total = sum(owed.values())
    scale = min(1.0, R / owed_total) if owed_total > 1e-9 else 1.0   # hybrid pays holders from R (pro-rata if short)
    alloc = {"ownership (R_a)": ownership_allocation(fp, res, pt),
             "outcomes (N_a^c)": {a: rc[a]["line_congestion"] for a in fp.names},
             "hybrid (pay the right)": {a: owed[a] * scale for a in fp.names}}
    rows = []
    for m, A in alloc.items():
        for a in fp.names:
            d = max(0.0, float(redispatch.get(a, 0.0))); Aa = float(A[a])   # credit only redispatch no longer borne
            rows.append({"method": m, "area": a, "owed (Sigma H)": round(owed[a], 0),
                         "cong. rev A_a": round(Aa, 0),
                         "whole? (rev only)": "yes" if owed[a] <= Aa + 1e-6 else "no",
                         "avoided redispatch": round(d, 0), "A_a + redispatch": round(Aa + d, 0),
                         "whole? (rev + redisp.)": "yes" if owed[a] <= Aa + d + 1e-6 else "no"})
    return pd.DataFrame(rows).set_index(["method", "area"])


def ownership_allocation(fp, res, pt, unassigned_split=0.5, transfer_split=0.5):
    """Each BA keeps the rent collected on the lines it *owns* (the "ownership"
    allocation -- where the constraint manifests). ``A_a = sum_{l owned by a}
    |mu_l F_l|``; rent on unowned/tie lines is split ``unassigned_split``. The
    inter-BA **transfer** rent ``R_T = |mu_T E|`` (the transport-layer rent of a
    binding inter-BA transfer constraint) is owned by neither BA's internal network,
    so it is split ``transfer_split`` to the first footprint and the remainder to the
    second -- the intertie transfer revenue shared 50/50 by default. Returns ``{ba: $/h}`` summing to the
    total congestion + transfer rent ``R + R_T``."""
    lr = line_rent_table(fp, res, pt)
    R_un = lr[lr.home.isna()]["rent"].sum()
    out = {ba: lr[lr.home == ba]["rent"].sum() + unassigned_split * R_un for ba in fp.names}
    RT = transfer_rent(res)
    if RT > 1e-9 and len(fp.names) >= 2:
        out[fp.names[0]] += transfer_split * RT
        out[fp.names[1]] += (1.0 - transfer_split) * RT
    return out


def methodology_ledger(fp, alloc, indep, resU, loads=None, cost=None):
    """Autarky vs a unified clearing under ONE congestion-revenue allocation.

    ``alloc`` is ``{ba: $ allocated to ba}`` (e.g. the ownership or net-congestion-
    position bookend). Homework-style ledger: per BA, consumers and generators side
    by side, the autarky position (each BA on its own engine on the full network),
    the unified position with the allocation ``A`` rebated to consumers, the change,
    and a Pareto (``Delta >= 0``) flag. Generalises :func:`autarky_vs_unified` to an
    arbitrary allocation dict -- the methodology-comparison frame. The unified
    positions sum to ``-(production cost)``.
    """
    data = {}
    for ba in fp.areas:
        La, Ra, PCa = _agg(fp, indep[ba], ba, loads, cost)
        Lu, Ru, PCu = _agg(fp, resU, ba, loads, cost)
        A = float(alloc.get(ba, 0.0))
        Ra_int = La - Ra
        PSa, PSu = Ra - PCa, Ru - PCu
        cons_aut, cons_fin = -La + Ra_int, -Lu + A
        cons_d, gen_d = cons_fin - cons_aut, PSu - PSa
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


def methodology_ledger_congestion(fp, alloc, indep, resU, pt, *, loads=None, cost=None,
                                  gen_fleet=None):
    """``methodology_ledger`` with the congestion-revenue derivation folded into the Unified
    block by :func:`_fold_congestion` -- the system rent R, its split into the congestion each
    region creates ``N_a^c``, and the rent THIS methodology allocates (renamed
    ``Unified: Congestion Revenue Allocation``), above each region's consumer/generator Pareto
    positions. ``N_a^c`` is the same across methodologies (the dispatch creates it); setting it
    beside each methodology's allocation is the benefit-commensurability comparison. ``alloc``
    is the ``{ba: $}`` allocation; ``pt`` the PTDF used for ``N_a^c``. Pass ``gen_fleet``
    with a non-default fleet (the hub weights are generation-weighted)."""
    loads = _case().loads if loads is None else loads
    return _fold_congestion(methodology_ledger(fp, alloc, indep, resU, loads, cost),
                            fp, resU, pt, loads, gen_fleet=gen_fleet)


def methodology_ledger_4p(fp, indep, resU, *, makewhole, path_hedge, uplift,
                          loads=None, cost=None):
    """Four-party autarky-vs-coordination ledger, separating the path-right holder
    from general load.

    Per BA, three parties carry the settlement (cash out negative): **Generators**
    (producer surplus), **Load** (general consumers: pay LMP, receive their
    make-whole allocation ``makewhole[a]``, pay ``uplift[a]``), and **Txn customer**
    (the path / flowgate right holder: receives ``path_hedge[a]``, and holds no
    right in autarky). Keeping the right-holder distinct from general load is what
    lets a methodology that funds a flowgate right *and* makes load whole avoid
    collapsing into a bookend.

    The collected rent is conserved at the **system** level iff
    ``sum_a (makewhole + path_hedge - uplift) = R`` (returned as ``residual`` in the
    summary); a positive residual is the methodology's revenue shortfall. Returns
    ``(ledger, summary)`` with the autarky and coordination positions, the change,
    and a Pareto flag per (BA, party).
    """
    loads = _case().loads if loads is None else loads
    data, R = {}, 0.0
    for a in fp.names:
        La, Ra, PCa = _agg(fp, indep[a], a, loads, cost)
        Lu, Ru, PCu = _agg(fp, resU, a, loads, cost)
        mw = float(makewhole.get(a, 0.0)); H = float(path_hedge.get(a, 0.0)); u = float(uplift.get(a, 0.0))
        parties = {"Generators": (Ra - PCa, Ru - PCu),
                   "Load": (-Ra, -Lu + mw - u),                 # autarky load keeps own congestion (La-Ra)
                   "Txn customer": (0.0, H)}
        for party, (aut, co) in parties.items():
            data[(a, party)] = {"Autarky position": aut, "Coordination position": co,
                                "Change": co - aut,
                                "Pareto (>= 0)": "yes" if co - aut >= -1e-6 else "no"}
    R = sum(_agg(fp, resU, a, loads, cost)[0] - _agg(fp, resU, a, loads, cost)[1] for a in fp.names)
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    summary = dict(R=R, makewhole_total=sum(makewhole.values()),
                   path_hedge_total=sum(path_hedge.values()), uplift_total=sum(uplift.values()),
                   residual=R - (sum(makewhole.values()) + sum(path_hedge.values()) - sum(uplift.values())))
    return (df.map(lambda v: round(v, 1) if isinstance(v, (int, float)) and not isinstance(v, bool) else v),
            summary)


# ──────────────────────────────────────────────────────────────────────────
# Bilateral self-solve with a redispatch premium (the 313 autarky baseline)
# ──────────────────────────────────────────────────────────────────────────
def bilateral_self_solve(fp, area, ratings=None, *, gen_fleet=None, loads=None,
                         voll=200.0):
    """A balancing authority **solving itself** — production cost plus its redispatch
    premium.

    The area serves its own load from its cheapest own generator on a path-by-path
    ATC posting (the 111 desk; ``atc.ttc`` per source->sink), fills any ATC shortfall
    with its **residual backstop** at the load bus, and then balances the rights-
    feasible (but possibly infeasible) schedule on the network it monitors
    (``bilateral.backstop_clear`` with ``bearer='tsp'``, so the premium falls on the
    transmission service provider that rated the ATC, not the scheduler). Returns
    ``(total_cost, redispatch_premium, shed)`` -- the honest "autarky carrying its
    budgeted redispatch" baseline the coordination methodologies are tested against.
    """
    import bilateral as bl
    from types import SimpleNamespace
    fleet = fleet_with_backstop() if gen_fleet is None else gen_fleet
    loads = _case().loads if loads is None else loads
    pt = _cn(ratings or {}).pt
    buses = set(fp.defs[area])
    mon = [l for l in pt.lines if fp.line_assign.get(l) == area]
    sup = [bl.Supplier(g, s["bus"], s["cost"], s["p_nom"])
           for g, s in fleet.items() if str(s["bus"]) in buses]
    src = min((s for s in sup if not s.gid.startswith("backstop")),
              key=lambda s: s.cost).bus
    own = {b: float(loads[b]) for b in fp.defs[area] if float(loads.get(b, 0.0)) > 0}
    aw, ss = [], {s.gid: 0.0 for s in sup}
    src_gid = next(s.gid for s in sup if s.bus == src)
    for snk, mw in own.items():
        cap, _ = atc.ttc(pt, src, snk, monitored=mon)
        a = min(mw, cap)
        aw.append(atc.Award(src, snk, a)); ss[src_gid] += a
        if mw - a > 1e-6:                       # ATC shortfall served locally by the backstop
            aw.append(atc.Award(snk, snk, mw - a))
            ss[f"backstop_{snk}"] = ss.get(f"backstop_{snk}", 0.0) + (mw - a)
    shim = SimpleNamespace(awards=aw, self_sched=ss, served=dict(own))
    bk = bl.backstop_clear(pt, sup, shim, voll=voll, monitored=mon, bearer="tsp")
    cost = {s.gid: s.cost for s in sup}
    total = sum(cost[g] * mw for g, mw in bk.dispatch.items())
    return total, bk.redispatch_cost, sum(bk.shed.values())


def pre_market_bilateral(fp, ratings=None, *, gen_fleet=None, loads=None, voll=200.0,
                         thickness=8, rounds=16, belief=None, rr=None, atc_by_ba=None):
    """The **pre-market bilateral** baseline: a near-competitive double auction across the
    whole network whose reserved paths are not simultaneously feasible, fixed by **local**
    redispatch.

    A high-liquidity / low-aversion repeated double auction (102's ``double_auction_clear``)
    trades cheap power to every load at near-competitive prices, on path-by-path ATC, so the
    reserved set jointly overloads. Each balancing authority then enforces **its own** lines:
    the schedule is curtailed to simultaneous feasibility, and the **sink** balancing authority
    backfills its curtailed load from **its own** merit-order resources (local generation, then
    its $90 backstop) -- dearer than the engine's global counter-trade, the premium **borne by
    the TSP**. Cross-border delivered megawatts pay **pancaked** transmission ``r_a = RR_a/ATC_a``
    (the 203 scheduling rate; a transfer that recovers each TSP's revenue requirement). Returns a
    summary dict (``served_cost`` = ideal cheap trade, ``total_cost`` = after local backfill,
    ``redispatch_premium`` = the difference, ``pancaked`` charges, and the curtailment).
    """
    import bilateral as bl
    fleet = fleet_with_backstop() if gen_fleet is None else gen_fleet
    loads = _case().loads if loads is None else loads
    pt = _cn(ratings or {}).pt
    sup = [bl.Supplier(g, s["bus"], s["cost"], s["p_nom"]) for g, s in fleet.items()]
    cost = {str(s["bus"]): float(s["cost"]) for s in fleet.values()}
    cap = {str(s["bus"]): float(s["p_nom"]) for s in fleet.values()}
    lcs = bl.calibrate_load_centers(pt, voll=voll, firmness=0.7)
    belief = belief or bl.Belief(price_risk=5.0, load_aversion=0.6)
    da = bl.double_auction_clear(pt, sup, lcs, thickness=thickness, rounds=rounds, belief=belief)

    # ideal cheap trade (the auction's intent) and the per-bus generation it implies
    served_cost = sum(cost[a.source] * a.mw for a in da.awards)
    g = {b: 0.0 for b in cost}
    # curtail the reserved paths to simultaneous feasibility (cheapest awards booked first)
    reqs = sorted([(a.source, a.sink, a.mw) for a in da.awards if a.mw > 1e-6],
                  key=lambda r: cost[r[0]])
    booked = atc.book_sequentially(pt, reqs, monitored="all")
    for a in booked:
        g[a.source] += a.mw
    cur = {}
    for a in da.awards:
        cur[a.sink] = cur.get(a.sink, 0.0) + a.mw
    for a in booked:
        cur[a.sink] = cur.get(a.sink, 0.0) - a.mw

    # the SINK BA backfills its curtailed load from its OWN merit-order resources
    backfill = {}
    for snk, mw in cur.items():
        if mw <= 1e-6:
            continue
        ba = fp.fp_of(snk)
        local = sorted([b for b in cost if fp.fp_of(b) == ba], key=lambda b: cost[b])
        need = mw
        for b in local:
            take = min(need, max(0.0, cap[b] - g[b]))
            g[b] += take; need -= take
            backfill[(ba, b)] = backfill.get((ba, b), 0.0) + take
            if need <= 1e-6:
                break

    total_cost = sum(cost[b] * mw for b, mw in g.items())

    # per-BA local redispatch premium = backfill cost at its sinks - the cheap trade curtailed there
    orig_ss, book_ss = {}, {}
    for a in da.awards:
        orig_ss[(a.source, a.sink)] = orig_ss.get((a.source, a.sink), 0.0) + a.mw
    for a in booked:
        book_ss[(a.source, a.sink)] = book_ss.get((a.source, a.sink), 0.0) + a.mw
    premium_by_ba = {a: 0.0 for a in fp.names}
    for (ba, b), mw in backfill.items():
        premium_by_ba[ba] += cost[b] * mw                       # what the local backfill cost
    for (src, snk), mw in orig_ss.items():
        c = mw - book_ss.get((src, snk), 0.0)
        ba = fp.fp_of(snk)
        if c > 1e-6 and ba is not None:
            premium_by_ba[ba] -= cost[src] * c                  # less the curtailed cheap trade

    # pancaked transmission on cross-border DELIVERED megawatts (RR recovery, a transfer)
    rr = rr or {a: 0.0 for a in fp.names}
    atc_by_ba = atc_by_ba or {a: 1.0 for a in fp.names}
    rate = {a: (float(rr[a]) / float(atc_by_ba[a]) if atc_by_ba.get(a) else 0.0) for a in fp.names}
    pancaked = 0.0
    for a in booked:
        sba, kba = fp.fp_of(a.source), fp.fp_of(a.sink)
        if sba is not None and kba is not None and sba != kba:
            pancaked += (rate[sba] + rate[kba]) * a.mw      # export + import rates pancake

    return dict(served_cost=served_cost, total_cost=total_cost,
                redispatch_premium=total_cost - served_cost, pancaked=pancaked,
                premium_by_ba={k: round(v, 1) for k, v in premium_by_ba.items()},
                curtailed={k: round(v, 1) for k, v in cur.items() if v > 1e-6},
                rate=rate, dispatch={b: round(v, 1) for b, v in g.items() if v > 1e-6})


def rr_margin_ledger(fp, *, embedded, margin, charge, coord_redispatch=0.0):
    """The **provision-layer** ledger: each TSP recovers a revenue requirement
    ``RR_a = embedded_a + delta_a``, where the margin ``delta_a`` is the headroom budgeted to
    absorb redispatch from non-simultaneously-feasible ATC -- sized to the dear **local**
    redispatch of the pre-market baseline. Coordination replaces that local redispatch with the
    engine's cheap integrated dispatch (``coord_redispatch``, system-wide), **freeing** the
    margin. A congestion-revenue methodology's per-BA ``charge`` (e.g. each BA's net congestion
    position) is then routed through the freed margin: ``absorbed = min(charge, delta)``, the
    remainder **uplifts to load**, and any unspent margin **rebates to load**. Returns a per-BA
    ledger (list of row dicts) and a summary. The teaching point lives in the summary: the
    *pooled* margin may cover the whole settlement while *per-BA* it sits in the wrong account.
    """
    rows = []
    for a in fp.names:
        m = max(0.0, float(margin.get(a, 0.0)))
        ch = float(charge.get(a, 0.0))
        absorbed = min(max(ch, 0.0), m)
        rows.append(dict(ba=a, embedded=float(embedded.get(a, 0.0)), margin=m, charge=ch,
                         absorbed=absorbed, uplift_to_load=max(0.0, ch - absorbed),
                         rebate_to_load=max(0.0, m - max(ch, 0.0))))
    tm = sum(r["margin"] for r in rows)
    tc = sum(r["charge"] for r in rows)
    summary = dict(total_margin=tm, total_charge=tc, coord_redispatch=float(coord_redispatch),
                   per_ba_uplift=sum(r["uplift_to_load"] for r in rows),
                   per_ba_rebate=sum(r["rebate_to_load"] for r in rows),
                   pooled_covers=tm + 1e-6 >= tc, pooled_residual=tm - tc)
    return rows, summary


def oatt_vs_tac(fp, res, *, rr, atc_by_ba, loads=None):
    """Two ways a TSP recovers its revenue requirement ``RR_a``, with opposite incidence.

    **OATT (firm-path):** a per-MW rate ``r_a = RR_a / ATC_a`` on firm reservations, **pancaked**
    across every system a schedule crosses -- a cross-border megawatt-hour pays the *sum* of the
    rates it traverses, a per-transaction adder that suppresses trade wherever it exceeds the
    source-to-sink cost gap. **TAC (measured-demand):** a volumetric rate ``t_a = RR_a / D_a`` on
    measured demand ``D_a = load_a + exports_a`` (energy withdrawn for end-use load plus energy
    exported out of the footprint; imports are supply, not demand, so the literal "load + *net*
    exports" would zero an importer's base -- gross exports is the operative quantity). The
    marginal cross-border megawatt-hour then pays **no** per-transaction transmission charge (the
    importing load is already in the base), but the **exporter** pays TAC on what it sends out (the
    contested export charge). Returns a per-BA ledger and a summary (the pancaked cross-border
    adder = Σ r_a). Generation/load are read from the clearing ``res`` (``gen_by_bus``/``load_by_bus``).
    """
    rows = []
    pancake = 0.0
    for a in fp.names:
        L = sum(v for b, v in res.load_by_bus.items() if fp.fp_of(str(b)) == a)
        G = sum(v for b, v in res.gen_by_bus.items() if fp.fp_of(str(b)) == a)
        exp = max(0.0, G - L)
        D = L + exp
        r = float(rr[a]) / float(atc_by_ba[a])
        t = float(rr[a]) / D if D > 1e-9 else float("nan")
        rows.append(dict(ba=a, rr=float(rr[a]), atc=float(atc_by_ba[a]), load=L, exports=exp,
                         measured_demand=D, oatt_rate=r, tac_rate=t,
                         tac_load_bill=L * t, tac_export_bill=exp * t))
        pancake += r
    summary = dict(pancaked_adder=pancake, total_rr=float(sum(rr.values())))
    return rows, summary


# ──────────────────────────────────────────────────────────────────────────
# Parallel-flow attribution (whose use caused the congestion on a line?)
# ──────────────────────────────────────────────────────────────────────────
def parallel_flow_attribution(fp, res, pt, loads=None):
    """Attribute each binding line's flow and congestion rent to each BA.

    Each BA's contribution pairs its net positions with its OWN ex-ante
    aggregation point (hub-relative shift factors), and the scheduled transfer
    carries its own hub-to-hub component, so the decomposition
    ``F_l = sum_a F_l^a + F_l^E`` is exact and independent of the power-flow
    reference bus. The rent collected on the line is ``|mu_l F_l|``; a share is
    NEGATIVE when the positions counterflow (relieve) the line. For an INTERNAL
    line, the manager BA's share is its own self-caused congestion and the other
    BA's share is the **parallel (loop) flow** it imposes on a neighbour's facility
    -- the congestion at the centre of the hold-harmless dispute. Returns a tidy
    table over the binding lines (one column pair per BA plus the Transfer pair).
    """
    loads = _case().loads if loads is None else loads
    weights = hub_weights(fp, "auto", loads)
    p = {b: res.gen_by_bus.get(b, 0.0) - float(loads.get(b, 0.0)) for b in pt.buses}
    E_a = {ba: sum(p[b] for b in fp.defs[ba]) for ba in fp.names}
    rows = []
    for l in pt.lines:
        mu = res.line_dual[l]
        if abs(mu) < 1e-3:
            continue
        li = pt.line_idx[l]
        Ftot = sum(pt.ptdf[li, pt.bus_idx[b]] * p[b] for b in pt.buses)
        kind, _ = fp.line_kind(pt, l)
        row = {"line": l, "kind": kind, "manager": fp.line_assign.get(l),
               "mu": round(mu, 1), "flow": round(Ftot, 1), "rent": round(abs(mu * Ftot), 0)}
        fe = 0.0
        for ba in fp.names:
            sfa = sum(w * pt.ptdf[li, pt.bus_idx[b]] for b, w in weights[ba].items())
            Fa = sum((pt.ptdf[li, pt.bus_idx[b]] - sfa) * p[b] for b in fp.defs[ba])
            fe += sfa * E_a[ba]
            row[f"{ba} flow"] = round(Fa, 1)
            row[f"{ba} rent"] = round(abs(mu) * Fa * np.sign(Ftot), 0)
        row["Transfer flow"] = round(fe, 1)                      # hub-to-hub component F^E_l
        row["Transfer rent"] = round(abs(mu) * fe * np.sign(Ftot), 0)
        rows.append(row)
    cols = ["line", "kind", "manager", "mu", "flow", "rent"]
    for ba in fp.names:
        cols += [f"{ba} flow", f"{ba} rent"]
    cols += ["Transfer flow", "Transfer rent"]
    return pd.DataFrame(rows, columns=cols).set_index("line")   # empty when no line binds


def loop_flow_charge(fp, res, pt, host, nbr, loads=None):
    """The neighbour's parallel-flow value on the host's binding lines, at the market price.

    For every line INTERNAL to ``host`` that binds, take its shadow price ``mu_m`` and the
    NEIGHBOUR's contribution to its flow, ``F_m^{nbr} = sum_{n in nbr} SF_{n,m}(g_n - d_n)``
    (hub-relative shift factors, so the split is reference-bus-invariant), and return

        tau = - sum_m mu_m F_m^{nbr}

    -- the ``loop_flow_triangle`` host->nbr leg, identically equal to ``N_nbr^c`` and to the
    host's marginal redispatch cost. It is SIGNED: negative in an interval whose neighbour
    flow relieves the constraint. Returns 0.0 when no host-internal line binds.
    """
    loads = _case().loads if loads is None else loads
    weights = hub_weights(fp, "auto", loads)
    p = {b: res.gen_by_bus.get(b, 0.0) - float(loads.get(b, 0.0)) for b in pt.buses}
    host_buses, nbr_buses = set(fp.defs[host]), fp.defs[nbr]
    tau = 0.0
    for l in pt.lines:
        mu = res.line_dual[l]
        if abs(mu) < 1e-3:
            continue
        b0, b1 = pt.line_buses[pt.line_idx[l]]
        if b0 not in host_buses or b1 not in host_buses:      # only the host's OWN facilities
            continue
        li = pt.line_idx[l]
        sfa = sum(w * pt.ptdf[li, pt.bus_idx[b]] for b, w in weights[nbr].items())
        Fnbr = sum((pt.ptdf[li, pt.bus_idx[b]] - sfa) * p[b] for b in nbr_buses)
        Ftot = sum(pt.ptdf[li, pt.bus_idx[b]] * p[b] for b in pt.buses)
        tau += abs(mu) * Fnbr * np.sign(Ftot)     # = -mu_m F_m^{nbr} (sign folds into |mu|*sign(F))
    return float(tau)


# ──────────────────────────────────────────────────────────────────────────
# Path rights as a perfect hedge (the Powerex "hold-harmless" concept)
# ──────────────────────────────────────────────────────────────────────────
def path_hedge_ledger(fp, ratings, rights, *, gen_fleet=None, loads=None,
                      shed_price=150.0):
    """A perfect hedge on path rights, with an inter-BA-before-intra-BA waterfall.

    Models the charitable reading of the Powerex concept: a firm point-to-point
    right (POR ``source`` -> POD ``sink``) honored under unified dispatch is given
    a perfect congestion hedge -- its holder receives ``H = mw*(lmp_sink -
    lmp_source)``, the merchandising surplus on its path. The question is whether
    the honored set is fundable from the rent the clearing actually collects.

    One unified clearing on the ``ratings`` network sets the LMPs and the
    collected congestion rent ``R = sum_l |mu_l F_l|``. Each right in ``rights``
    (a dict ``{'source','sink','mw','tier'}`` with ``tier`` in ``{'inter','intra'}``)
    is valued at ``H``. The WECC-rated **inter-BA** paths are a simultaneously
    feasible backbone, so they are paid FIRST from ``R``; **intra-BA** rights are
    paid from the residual. When the honored set fails the simultaneous-feasibility
    test (``atc.simultaneous_feasibility``) the total hedge exceeds ``R`` and the
    shortfall falls on the lower-priority intra-BA tier. Returns
    ``(ledger, summary)``; ``ledger`` is a per-right table (tier, path, MW, hedge,
    funded, shortfall) and ``summary`` carries ``R``, the per-tier totals, the
    waterfall, the SFT verdict, and a per-right ``honored`` flag for the figure.
    """
    loads = _case().loads if loads is None else loads
    _c = _cn(ratings or {})
    net, pt = _c.network, _c.pt
    res = solve_engine_dispatch(
        pt, _case().make_engine("UNIFIED", buses=pt.buses, gen_fleet=gen_fleet, loads=loads),
        shed_price=shed_price)
    R = sum(abs(res.line_dual[l]) * abs(res.flow_own[l]) for l in pt.lines)

    rights = [dict(r) for r in rights]
    for r in rights:
        r["source"], r["sink"] = str(r["source"]), str(r["sink"])
        r["tier"] = r.get("tier", "inter")
        r["hedge"] = float(r["mw"]) * (res.lmp[r["sink"]] - res.lmp[r["source"]])
    aw = [atc.Award(r["source"], r["sink"], r["mw"]) for r in rights]
    feasible, _ = atc.simultaneous_feasibility(pt, aw)
    overloaded = atc.overloaded_lines(pt, aw)

    H_inter = sum(r["hedge"] for r in rights if r["tier"] == "inter")
    H_intra = sum(r["hedge"] for r in rights if r["tier"] == "intra")
    fund_inter = min(H_inter, R)
    residual = R - fund_inter
    fund_intra = min(H_intra, max(0.0, residual))
    shortfall = max(0.0, (H_inter + H_intra) - R)
    # pro-rata funding within each tier (priority across tiers, share within)
    share = {"inter": (fund_inter / H_inter if H_inter > 1e-9 else 1.0),
             "intra": (fund_intra / H_intra if H_intra > 1e-9 else 1.0)}

    rows = []
    for r in rights:
        funded = r["hedge"] * share[r["tier"]]
        r["honored"] = funded >= r["hedge"] - 1e-6        # fully funded ?
        rows.append({"tier": r["tier"], "path": f'{r["source"]}->{r["sink"]}',
                     "MW": round(float(r["mw"]), 0),
                     "hedge $/h": round(r["hedge"], 0),
                     "funded $/h": round(funded, 0),
                     "shortfall $/h": round(r["hedge"] - funded, 0)})
    ledger = pd.DataFrame(rows)
    ledger.loc["collected rent R"] = ["", "", "", round(R, 0), round(R, 0), ""]

    summary = dict(
        R=R, rent_by_line={l: round(abs(res.line_dual[l]) * abs(res.flow_own[l]), 0)
                           for l in pt.lines if abs(res.line_dual[l]) > 1e-3},
        H_inter=H_inter, H_intra=H_intra, total=H_inter + H_intra,
        fund_inter=fund_inter, fund_intra=fund_intra, shortfall=shortfall,
        sft_feasible=feasible, overloaded=overloaded, rights=rights, res=res,
        pt=pt, net=net)
    return ledger, summary


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
    fleet = _case().gen_fleet if gen_fleet is None else gen_fleet
    loads = _case().loads if loads is None else loads
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


# ──────────────────────────────────────────────────────────────────────────
# Distributional Pareto: coalition accounting over uncertainty draws (314)
# ──────────────────────────────────────────────────────────────────────────
def coalition_positions(fp, pt, host, *, gen_fleet=None, loads=None, curves=None,
                        cost=None, load=None, te_split="interface", shed_price=None,
                        transfer_limits=None):
    """One interval of the coalition-accounting Pareto check, on drawn drivers.

    Works for any disjoint partition of the buses into footprints (two or more;
    single-bus footprints included). ``te_split`` is a :func:`_ba_shares` spec or
    ``'interface'``; the default ``'interface'`` is the EDAM rule — each
    interface's transfer settlement split equally between the two footprints
    party to it (:func:`pairwise_transfer_settlement`) — which with two
    footprints is identically the legacy 50/50 split.

    ``transfer_limits`` clears the unified case in the EDAM-literal form
    (``solve_engine_transfers``) instead of the pooled engine: a ``{(a, b):
    MW}`` dict of released capacities per interface (``None`` values =
    unlimited) or the string ``'mod29'`` (each interface's released capacity
    derived as its independently-rated TTC, ``atc.interface_ttc``). Binding
    interfaces collect scheduling rent ``R_T^{ab}``; Method 2 carries it inside
    the transfer settlement, and Method 1 is credited each interface's rent
    half-and-half to its parties, so BOTH methods allocate the full
    ``R + R_T`` pool and the conservation identity (eq. 4 of notebook 314)
    holds unchanged. Default ``None`` keeps the pooled clearing.

    The baseline is the ACCOMMODATION status quo: every area except ``host``
    self-serves (its own engine, only its own lines activated), their net
    injections sit on the grid as fixed exogenous inputs, and ``host`` clears
    around the loop flow they add — bearing the redispatch unpriced. The
    coalition outcome is the unified clearing of the same drawn interval.
    Both cases rebate each area's internal congestion to its consumers (the
    baseline keeps its own rent; the unified rebate is the Congestion Revenue
    Allocation under each method), so each constituency's change isolates the
    gains from trade plus the cross-border reallocation the methodology prices
    — the same convention as :func:`autarky_vs_unified`.

    ``cost`` ``{gen: $/MWh}`` and ``load`` ``{bus: MW}`` override the base
    fleet / loads for THIS interval (the drawn drivers, ``risk.py`` keys).
    With ``curves`` ``{gen: (a, b)}`` the drawn cost shifts the curve intercept
    (``a + drawn − base flat cost``) and every clearing is the rising-MC QP, so
    producer surplus is genuine; without ``curves`` the clearings are flat-offer
    (producer surplus degenerate whenever a unit sets its own price).

    Returns a dict: ``delta`` ``{(method, area, side): $/h}`` with side
    ``'Consumer'`` / ``'Generator'`` (the Generator change is
    method-independent but is stated under each method for a uniform key set),
    plus the interval diagnostics ``R``, ``TE``, ``Nc``, ``alloc``, ``gains``
    (baseline minus unified TOTAL cost — production plus unserved energy charged
    at the shed price), ``shed_base``, ``shed_unified``, and the clearings
    themselves (``base``, ``unified``). Unserved energy is likewise charged to
    the shedding area's consumers at the shed price, so an infeasible-but-shedding
    baseline does not read as cheap.
    """
    fleet = _case().gen_fleet if gen_fleet is None else gen_fleet
    base_loads = _case().loads if loads is None else loads
    loads_s = {b: float((load or {}).get(b, mw)) for b, mw in base_loads.items()}
    cost_s = {g: float((cost or {}).get(g, spec["cost"])) for g, spec in fleet.items()}
    fleet_s = {g: {**spec, "cost": cost_s[g]} for g, spec in fleet.items()}
    curves_s = ({g: (curves[g][0] + cost_s[g] - fleet[g]["cost"], curves[g][1])
                 for g in curves} if curves else None)
    gen_bus = {g: str(s["bus"]) for g, s in fleet.items()}
    cbb = {s["bus"]: cost_s[g] for g, s in fleet.items()}     # drawn flat cost by bus

    def _solve(engine, **kw):
        return (solve_engine_qp(pt, engine, curves=curves_s, shed_price=shed_price, **kw)
                if curves_s else solve_engine_dispatch(pt, engine, shed_price=shed_price, **kw))

    base, exo = {}, {}
    for a in fp.names:
        if a == host:
            continue
        act = [l for l in pt.lines if fp.line_assign.get(l) == a]
        base[a] = _solve(_case().make_engine(a, fp.defs[a], gen_fleet=fleet_s, loads=loads_s,
                                        activated=act))
        for b in fp.defs[a]:
            v = (base[a].gen_by_bus.get(b, 0.0) - loads_s.get(b, 0.0)
                 + base[a].shed_by_bus.get(b, 0.0))
            if abs(v) > 1e-9:
                exo[b] = exo.get(b, 0.0) + v
    act_h = [l for l in pt.lines if fp.line_assign.get(l) == host]
    base[host] = _solve(_case().make_engine(host, fp.defs[host], gen_fleet=fleet_s,
                                            loads=loads_s, activated=act_h), exo=exo)

    eng_u = _case().make_engine("UNIFIED", buses=pt.buses, gen_fleet=fleet_s, loads=loads_s)
    if transfer_limits is None:
        resU = _solve(eng_u)
    else:
        pairs = fp_interfaces(fp, pt)
        if isinstance(transfer_limits, str) and transfer_limits == "mod29":
            W29 = hub_weights(fp, "auto", base_loads, fleet)
            ifc = {k: atc.interface_ttc(pt, [k], W29)[k][0] for k in pairs}
        else:
            _n = {tuple(sorted((str(x), str(y)))): v
                  for (x, y), v in transfer_limits.items()}
            ifc = {k: _n.get(k) for k in pairs}
        resU = solve_engine_transfers(pt, eng_u, fp.defs, ifc, curves=curves_s,
                                      shed_price=shed_price)
    alloc, summ, _, _ = allocate_congestion_rent(fp, resU, pt, loads_s, te_split=te_split,
                                                 gen_fleet=fleet)
    # Method 1 stops at the line rent R; each binding interface's scheduling
    # rent is credited half to each party so both methods allocate R + R_T.
    rt_half = {a: 0.0 for a in fp.names}
    for (x, y), v in transfer_rent_by_interface(resU).items():
        rt_half[x] += 0.5 * v
        rt_half[y] += 0.5 * v

    # Unserved energy is charged to the area's consumers at the shed price (the
    # teaching VOLL): shed load pays no LMP, but the consumers lose the value of
    # the megawatts they did not get, so an unreliable autarky baseline must not
    # read as cheap. The same charge enters `gains` as avoided unserved energy.
    V = float(shed_price) if shed_price is not None else 0.0
    delta, cost_b, cost_u = {}, 0.0, 0.0
    for a in fp.names:
        Lb, Rb, PCb = _agg(fp, base[a], a, loads_s, cbb, curves_s, gen_bus)
        Lu, Ru, PCu = _agg(fp, resU, a, loads_s, cbb, curves_s, gen_bus)
        shed_ba = sum(base[a].shed_by_bus.get(b, 0.0) for b in fp.defs[a])
        shed_ua = sum(resU.shed_by_bus.get(b, 0.0) for b in fp.defs[a])
        cost_b += PCb + V * shed_ba
        cost_u += PCu + V * shed_ua
        d_gen = (Ru - PCu) - (Rb - PCb)
        cons_base = -Lb - V * shed_ba + (Lb - Rb)              # own internal rent rebated
        for m in (1, 2):
            A = float(alloc.loc[a, f"method{m}"]) + (rt_half[a] if m == 1 else 0.0)
            delta[(m, a, "Consumer")] = (-Lu - V * shed_ua + A) - cons_base
            delta[(m, a, "Generator")] = d_gen
    return dict(delta=delta, R=summ["R"], TE=summ["TE"], Nc=summ["Nc"], alloc=alloc,
                gains=cost_b - cost_u, R_T=transfer_rent(resU),
                shed_base=sum(sum(r.shed_by_bus.values()) for r in base.values()),
                shed_unified=sum(resU.shed_by_bus.values()),
                base=base, unified=resU)


def coalition_pareto_draws(fp, pt, scen, host, *, gen_fleet=None, loads=None, curves=None,
                           te_split="interface", shed_price=None, progress=None,
                           transfer_limits=None):
    """The coalition-accounting Pareto check over a ``risk.py`` scenario set.

    For every draw in ``scen`` (a ``risk.Scenarios`` — Monte-Carlo or
    Gauss-Hermite), clear the accommodation baseline and the unified dispatch
    on the SAME drawn drivers (:func:`coalition_positions`) and record each
    constituency's change under each allocation method. Returns
    ``(deltas, diag)``: ``deltas[(method, area, side)]`` are length-``S``
    arrays carrying ``scen.weight``, ready for
    :func:`pareto_distribution_ledger` or ``risk.py``'s weighted statistics;
    ``diag`` is a per-scenario DataFrame (``R``, ``TE``, ``gains``, shed MW)
    for conservation and coverage checks. ``progress`` prints a line every
    that-many intervals.
    """
    keys, out, rows = None, None, []
    for s in range(scen.S):
        cost = {k[1]: float(scen.draws[k][s]) for k in scen.keys if k[0] == "cost"}
        load = {k[1]: float(scen.draws[k][s]) for k in scen.keys if k[0] == "load"}
        one = coalition_positions(fp, pt, host, gen_fleet=gen_fleet, loads=loads,
                                  curves=curves, cost=cost, load=load,
                                  te_split=te_split, shed_price=shed_price,
                                  transfer_limits=transfer_limits)
        if out is None:
            keys = list(one["delta"])
            out = {k: np.empty(scen.S) for k in keys}
        for k in keys:
            out[k][s] = one["delta"][k]
        rows.append(dict(R=one["R"], TE=one["TE"], R_T=one["R_T"], gains=one["gains"],
                         shed_base=one["shed_base"], shed_unified=one["shed_unified"]))
        if progress and (s + 1) % progress == 0:
            print(f"  {s + 1}/{scen.S} intervals cleared")
    return out, pd.DataFrame(rows)


def pareto_distribution_ledger(deltas, weight, gammas=(0.0, 0.005, 0.02), q=0.05):
    """The distributional Pareto ledger — one row per (method, area,
    constituency), the columns the acceptance tests a coalition member could
    apply to its change distribution.

    ``E`` and ``sd`` are the risk-neutral view; ``P(>=0)`` is the
    single-interval Pareto test's pass frequency; ``q05`` (the lower
    ``q``-quantile) is the tail an ex-post make-whole would face; and the
    mean-variance certainty equivalent ``CE = E − (gamma/2)·Var`` for each
    ``gamma`` is the ex-ante test a risk-averse member actually signs on
    (``risk.py``'s canon, ``GAMMA_LADDER``). Ex-ante Pareto at ``gamma``
    requires every row's CE ≥ 0; per-interval Pareto requires every row's
    pass frequency = 1 — the two distributional versions of the
    single-interval check (:func:`pareto_verdicts` tabulates them).
    """
    w = np.asarray(weight, dtype=float)
    rows = {}
    for (m, a, side), v in deltas.items():
        v = np.asarray(v, dtype=float)
        mu = float(np.sum(w * v))
        var = float(np.sum(w * (v - mu) ** 2))
        order = np.argsort(v)
        qv = float(np.interp(q, np.cumsum(w[order]), v[order]))
        row = {"E": mu, "sd": var ** 0.5,
               "P(>=0)": float(np.sum(w * (v >= -1e-9))),
               f"q{int(round(q * 100)):02d}": qv}
        for g in gammas:
            row[f"CE g={g:g}"] = mu - 0.5 * g * var
        rows[(f"Method {m}", a, side + "s")] = row
    df = pd.DataFrame(rows).T
    df.index.names = ["method", "area", "constituency"]
    return df


def pareto_verdicts(ledger):
    """Verdict table per method from a :func:`pareto_distribution_ledger`:
    ``every interval`` (all pass frequencies equal 1 — the single-interval test
    holding draw by draw) and one ``ex ante`` column per certainty-equivalent
    ladder rung (all of that method's CEs ≥ 0 — the test a member applying that
    risk aversion signs on)."""
    ce_cols = [c for c in ledger.columns if c.startswith("CE")]
    out = {}
    for m in ledger.index.get_level_values(0).unique():
        blk = ledger.loc[m]
        row = {"every interval": bool((blk["P(>=0)"] >= 1.0 - 1e-9).all())}
        for c in ce_cols:
            row[f"ex ante {c[3:]}"] = bool((blk[c] >= -1e-6).all())
        out[m] = row
    return pd.DataFrame(out).T


if __name__ == "__main__":
    import footprints as fpmod
    import wscc9_model as wm   # the smoke test IS the 9-bus case, explicitly

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
    alloc, summ, _, _ = allocate_congestion_rent(fp, resU, ptB, wm.CASE.loads)
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

    # ── Transmission service: embedded-cost recovery, autarky vs joint (203) ──
    SCN2 = {"line_2": 60.0}
    pt2 = compute_ptdf(wm.build_network(SCN2), slack_bus="1")
    resU2 = solve_engine_dispatch(pt2, wm.make_engine("UNIFIED", buses=pt2.buses))
    _, indep2 = independent_clear(fp, SCN2, shed_price=150.0)
    RR = {"BA-1": 800.0, "BA-2": 3225.0}
    atc_by_ba = {a: scheduling_subscription(fp, indep2[a], a) for a in fp.names}
    rec = transmission_service_recovery(fp, indep2, resU2, pt2, embedded=RR, atc_by_ba=atc_by_ba)
    print("\ntransmission-service recovery (autarky vs joint dispatch):")
    print(rec.to_string())
    led = scheduling_service_ledger(fp, indep2, resU2, pt2, embedded=RR, atc_by_ba=atc_by_ba)
    jt = led.loc["Joint position", ("TOTAL", "")]
    target = -(sum(_agg(fp, resU2, a)[2] for a in fp.areas) + sum(RR.values()))
    print(f"\nledger conserves: joint positions sum = {jt:,.1f}  vs  -(prod cost + embedded) "
          f"= {target:,.1f}")
    assert abs(jt - target) < 1.0, "scheduling ledger does not conserve"
    print("OK -- both recover in autarky; joint dispatch strands "
          f"{rec.index[~rec['recovers joint']].tolist()}")
