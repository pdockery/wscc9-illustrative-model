"""The constraint set, the hub weights and the declared interfaces as arguments.

Each new argument is tested on the 9-bus case against its default.

1. A case whose ``build()`` returns a ``FlowgateSet`` is cleared on that
   set by ``revenue_allocation``. The test set is the per-line set plus one
   cut flowgate ``TIES`` over the two ties, rated below the cut flow of the
   per-line clearing, so the flowgate result differs from the per-line
   result by the cut binding. The implied per-line duals are ``Sᵀ μ_fg``,
   the LMP identity holds through them, and with
   ``rent_basis="constraints"`` the congestion rent and both allocation
   methods are stated on the constraint set. The default line basis is
   the per-line table, the form in which Method 1 is stated.
   ``physical_flows``, ``rt_relief`` and ``atc.crr_auction`` take the set,
   and the identity set reproduces their per-line results.
2. ``case.hub_weights(kind, fp)`` returns ``revenue_allocation.hub_weights``
   on the 9-bus case.
3. An explicit declared interface set equal to the derived one returns
   identical results from ``solve_with_transfers``,
   ``mod29_interface_limits`` and ``coalition_positions``.

    pytest tests/test_constraint_set_args.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "lib", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atc                         # noqa: E402
import footprints as fpmod         # noqa: E402
import revenue_allocation as ra    # noqa: E402
import seams_engine as se          # noqa: E402
import wscc9_model as wm           # noqa: E402

BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
MANAGE = {"BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
          "BA-2": ["line_0", "line_1", "line_7", "line_8"]}
COLORS = {"BA-1": "#993AFF", "BA-2": "#2471A3"}
TIES = ("line_2", "line_7")
#: The share of the per-line cut flow the cut flowgate is rated at.
CUT_SHARE = 0.8


def _fp(pt):
    return fpmod.make(pt, BA_DEFS, COLORS, manage=MANAGE, tie_label="tie")


@pytest.fixture(scope="module")
def lines_case():
    pt = wm.CASE.build().pt
    fp = _fp(pt)
    _n, p, _e, r = ra.solve_with_transfers(fp, {}, shed_price=150.0)
    return fp, p, r


def _cut_signs(pt, res):
    """Each tie's sign in the BA-1 to BA-2 direction, read off the flow."""
    out = {}
    for l in TIES:
        b0, b1 = pt.line_buses[pt.line_idx[l]]
        into_ba2 = 1.0 if b1 in BA_DEFS["BA-2"] else -1.0
        out[l] = into_ba2
    return out


def _cut_set(pt, signs, rating):
    """The per-line set plus the cut flowgate ``TIES``."""
    ident = se.identity_flowgates(pt)
    row = np.zeros(pt.n_line)
    for l, sg in signs.items():
        row[pt.line_idx[l]] = sg
    return se.FlowgateSet(
        ids=list(ident.ids) + ["TIES"], S=np.vstack([ident.S, row]),
        fwd=np.concatenate([ident.fwd, [rating]]),
        rev=np.concatenate([ident.rev, [1e6]]),
        members={**ident.members, "TIES": [(l, sg) for l, sg in signs.items()]})


class _FlowgateCase(wm.WSCC9Case):
    """The 9-bus case rated on a stated ``FlowgateSet``."""

    def __init__(self, fg):
        super().__init__()
        self._fg = fg

    def build(self, line_ratings=None, **variant):
        cn = super().build(line_ratings, **variant)
        return se.CaseNetwork(network=cn.network, pt=cn.pt, fg=self._fg)


@pytest.fixture(scope="module")
def cut(lines_case):
    fp, pt, r_lines = lines_case
    signs = _cut_signs(pt, r_lines)
    cut_flow = sum(sg * r_lines.flow_own[l] for l, sg in signs.items())
    assert cut_flow > 1.0, "the canonical case exports from BA-1 to BA-2"
    fg = _cut_set(pt, signs, CUT_SHARE * cut_flow)
    case = _FlowgateCase(fg)
    with se.using(case):
        _n, p, _e, r = ra.solve_with_transfers(fp, {}, shed_price=150.0)
    return dict(fp=fp, pt=pt, fg=fg, case=case, signs=signs, cut_flow=cut_flow,
                lines=r_lines, res=r)


# ── B2: the clearings on the case's constraint set ─────────────────────────
def test_flowgate_case_binds_the_cut_the_lines_leave_slack(cut):
    r, rl, fg = cut["res"], cut["lines"], cut["fg"]
    rating = CUT_SHARE * cut["cut_flow"]
    # the per-line clearing carries the cut above the rating; the flowgate
    # clearing holds it at the rating and prices it
    assert cut["cut_flow"] > rating + 1.0
    assert abs(r.fg_flow["TIES"] - rating) < 1e-6
    assert abs(r.fg_dual["TIES"]) > 1e-3
    assert r.fg_members["TIES"] == fg.members["TIES"]
    assert not rl.fg_members
    assert r.total_cost > rl.total_cost + 1e-6


def test_implied_duals_and_the_lmp_identity(cut):
    r, pt, fg = cut["res"], cut["pt"], cut["fg"]
    impl = fg.implied_line_duals(r.fg_dual)
    for i, l in enumerate(pt.lines):
        assert abs(r.line_dual[l] - impl[i]) < 1e-9
    # F_m = Σ_i S[m,i] F_i on the cut
    f = sum(sg * r.flow_own[l] for l, sg in cut["signs"].items())
    assert abs(f - r.fg_flow["TIES"]) < 1e-6
    # λ_n = λ̂_a(n) + Σ_m SF_{n,m} μ^impl_m
    a_of = {b: a for a, bs in BA_DEFS.items() for b in bs}
    for b in pt.buses:
        cong = sum(pt.ptdf[pt.line_idx[l], pt.bus_idx[b]] * r.line_dual[l]
                   for l in pt.lines)
        assert abs(r.lmp[b] - (r.area_prices[a_of[b]] + cong)) < 1e-6


def test_rent_and_both_methods_on_the_constraint_set(cut):
    fp, pt, r = cut["fp"], cut["pt"], cut["res"]
    with se.using(cut["case"]):
        alloc, summ, lr, _sep = ra.allocate_congestion_rent(
            fp, r, pt, wm.DEFAULT_LOADS, te_split=0.5, rent_basis="constraints")
        _a0, summ0, lr0, _s0 = ra.allocate_congestion_rent(
            fp, r, pt, wm.DEFAULT_LOADS, te_split=0.5)
        rc = ra.regional_congestion(fp, r, pt, wm.DEFAULT_LOADS)
        TE = ra.transfer_settlement(fp, r, wm.DEFAULT_LOADS)
    R = ra.constraint_rent(r, pt)
    RT = ra.transfer_rent(r)
    assert abs(summ["R"] - R) < 1e-6
    assert abs(R - sum(abs(mu * r.fg_flow[k]) for k, mu in r.fg_dual.items())) < 1e-9
    # the cut's members are managed by different footprints, so it is unassigned
    assert lr.loc["TIES", "kind"] == "flowgate" and lr.loc["TIES", "home"] is None
    assert abs(alloc["method1"].sum() - R) < 1e-6
    assert abs(alloc["method2"].sum() - (R + RT)) < 1e-6
    assert abs(sum(rc[a]["line_congestion"] for a in fp.names) + TE - (R + RT)) < 1e-6
    # the default line basis is the per-line table, the form WECC 201 states
    pd.testing.assert_frame_equal(lr0, ra.line_rent_table(fp, r, pt))
    assert abs(summ0["R"] - ra.line_rent_table(fp, r, pt)["rent"].sum()) < 1e-9
    own = ra.ownership_allocation(fp, r, pt, rent_basis="constraints")
    assert abs(sum(own.values()) - (R + RT)) < 1e-6
    with pytest.raises(ValueError):
        ra.allocate_congestion_rent(fp, r, pt, wm.DEFAULT_LOADS, rent_basis="paths")


def test_the_cut_case_independent_clear_enforces_the_cut_in_both_areas(cut):
    fp = cut["fp"]
    with se.using(cut["case"]):
        _pt, ind = ra.independent_clear(fp, {}, shed_price=150.0)
    for a in fp.names:
        assert ind[a] is not None
        assert "TIES" in ind[a].fg_members


def test_path_hedge_ledger_tests_the_rights_on_the_cut(cut):
    fp = cut["fp"]
    rights = [{"source": "3", "sink": "5", "mw": 200.0, "tier": "inter"}]
    with se.using(cut["case"]):
        _led, summ = ra.path_hedge_ledger(fp, {}, rights)
    # the clearing, the rent and the SFT read the same flowgate set
    assert "TIES" in summ["overloaded"]
    assert abs(summ["R"] - ra.constraint_rent(summ["res"], summ["pt"])) < 1e-9
    assert set(summ["rent_by_line"]) <= set(cut["fg"].ids)


def test_coalition_positions_conserve_on_the_cut(cut):
    """Both methods allocate the rent the flowgate clearing collects."""
    fp, pt, fg = cut["fp"], cut["pt"], cut["fg"]
    with se.using(cut["case"]):
        out = ra.coalition_positions(fp, pt, "BA-1", shed_price=150.0,
                                     flowgates=fg, transfer_limits="mod29")
    resU, R, RT = out["unified"], out["R"], out["R_T"]
    assert resU.fg_members
    assert abs(R - ra.constraint_rent(resU, pt)) < 1e-9
    alloc = out["alloc"]
    rt_half = sum(ra.transfer_rent_by_interface(resU).values())
    assert abs(alloc["method1"].sum() + rt_half - (R + RT)) < 1e-6
    assert abs(alloc["method2"].sum() - (R + RT)) < 1e-6


def test_a_mixed_sign_path_separates_the_two_bases(lines_case):
    """A path whose members carry flow of opposite sign: the line basis sums
    the members' absolute rents and overstates the rent collected, and the
    constraint basis equals the signed rent Method 2 partitions."""
    fp, pt, r_lines = lines_case
    f4, f5 = r_lines.flow_own["line_4"], r_lines.flow_own["line_5"]
    assert f4 > 0 > f5, "the members carry flow of opposite sign"
    ident = se.identity_flowgates(pt)
    row = np.zeros(pt.n_line)
    row[pt.line_idx["line_4"]] = row[pt.line_idx["line_5"]] = 1.0
    fg = se.FlowgateSet(
        ids=list(ident.ids) + ["MIX"], S=np.vstack([ident.S, row]),
        fwd=np.concatenate([ident.fwd, [0.7 * (f4 + f5)]]),
        rev=np.concatenate([ident.rev, [1e6]]),
        members={**ident.members, "MIX": [("line_4", 1.0), ("line_5", 1.0)]})
    case = _FlowgateCase(fg)
    with se.using(case):
        _n, _p, _e, r = ra.solve_with_transfers(fp, {}, shed_price=150.0)
        lines = ra.allocate_congestion_rent(fp, r, pt, wm.DEFAULT_LOADS, te_split=0.5)
        cons = ra.allocate_congestion_rent(fp, r, pt, wm.DEFAULT_LOADS, te_split=0.5,
                                           rent_basis="constraints")
        rc = ra.regional_congestion(fp, r, pt, wm.DEFAULT_LOADS)
        TE = ra.transfer_settlement(fp, r, wm.DEFAULT_LOADS)
        out = ra.coalition_positions(fp, pt, "BA-1", shed_price=150.0, flowgates=fg)
    assert abs(r.fg_dual["MIX"]) > 1e-3 and not r.shed_by_bus
    R_c, R_l, RT = cons[1]["R"], lines[1]["R"], ra.transfer_rent(r)
    assert R_l > R_c + 1.0
    # both members are managed by BA-1, so the path's home is BA-1
    assert cons[2].loc["MIX", "home"] == "BA-1"
    assert abs(sum(rc[a]["line_congestion"] for a in fp.names) + TE - (R_c + RT)) < 1e-6
    assert abs(cons[0]["method1"].sum() - R_c) < 1e-6
    assert abs(cons[0]["method2"].sum() - (R_c + RT)) < 1e-6
    # coalition_positions allocates the collected rent under both methods
    assert abs(out["R"] - ra.constraint_rent(out["unified"], pt)) < 1e-9
    assert abs(out["alloc"]["method1"].sum() - out["alloc"]["method2"].sum()) < 1e-6


def test_identity_set_reproduces_the_line_clearings(lines_case):
    fp, pt, r_lines = lines_case
    case = _FlowgateCase(se.identity_flowgates(pt))
    with se.using(case):
        _n, _p, _e, r = ra.solve_with_transfers(fp, {}, shed_price=150.0)
        _pt, ind = ra.independent_clear(fp, {}, shed_price=150.0)
    _pt0, ind0 = ra.independent_clear(fp, {}, shed_price=150.0)
    for b in pt.buses:
        assert abs(r.lmp[b] - r_lines.lmp[b]) < 1e-9
    for a in fp.areas:
        assert (ind[a] is None) == (ind0[a] is None)
        if ind[a] is None:
            continue
        for b in pt.buses:
            assert abs(ind[a].lmp[b] - ind0[a].lmp[b]) < 1e-9


def test_managed_constraints_follow_the_members(cut):
    fp, pt, fg = cut["fp"], cut["pt"], cut["fg"]
    for a in fp.names:
        lines = ra.managed_constraints(fp, a, pt)
        assert lines == [l for l in pt.lines if fp.line_assign.get(l) == a]
        # every one-line flowgate keeps its line's manager, and the cut, whose
        # members are managed by both areas, is enforced by both (the
        # membership rule of wecc_flowgates.monitored_paths)
        assert ra.managed_constraints(fp, a, pt, fg) == lines + ["TIES"]
    # the Method 1 home of the shared cut is no single footprint
    assert ra.constraint_assign(fp, fg)["TIES"] is None
    row = np.zeros(pt.n_line)
    row[pt.line_idx["line_4"]] = row[pt.line_idx["line_5"]] = 1.0
    one = se.FlowgateSet(ids=["P"], S=row[None, :],
                         fwd=np.array([50.0]), rev=np.array([50.0]),
                         members={"P": [("line_4", 1.0), ("line_5", 1.0)]})
    assert ra.constraint_assign(fp, one) == {"P": "BA-1"}
    assert ra.managed_constraints(fp, "BA-1", pt, one) == ["P"]
    assert ra.managed_constraints(fp, "BA-2", pt, one) == []


def test_physical_flows_on_the_constraint_set(cut):
    pt, fg, r = cut["pt"], cut["fg"], cut["res"]
    rl = cut["lines"]
    base = se.physical_flows(pt, [rl])
    assert list(base.index) == list(pt.lines)
    pf = se.physical_flows(pt, [rl], flowgates=fg)
    assert list(pf.index) == list(fg.ids)
    assert abs(pf.loc["TIES", "F_phys"] - round(cut["cut_flow"], 1)) < 0.11
    assert bool(pf.loc["TIES", "overload"])
    assert pf.loc["TIES", "rating"] == pytest.approx(CUT_SHARE * cut["cut_flow"])
    # the per-line rows of the set agree with the per-line table
    for l in pt.lines:
        assert pf.loc[l, "F_phys"] == base.loc[l, "F_phys"]
        assert bool(pf.loc[l, "overload"]) == bool(base.loc[l, "overload"])
    pf2 = se.physical_flows(pt, [r], flowgates=fg)
    assert not bool(pf2.loc["TIES", "overload"])


def test_rt_relief_on_the_constraint_set(lines_case, cut):
    fp, pt, _r = lines_case
    eng = wm.make_engine("BA-1", buses=BA_DEFS["BA-1"], activated=MANAGE["BA-1"])
    r_da = se.solve_engine_dispatch(pt, eng, shed_price=150.0)
    nbr = {l: 5.0 for l in MANAGE["BA-1"]}
    e0, r0 = se.rt_relief(pt, eng, r_da, nbr, 0.1)
    e1, r1 = se.rt_relief(pt, eng, r_da, nbr, 0.1,
                          flowgates=se.identity_flowgates(pt))
    assert set(e0.gens) == set(e1.gens)
    for g in e0.gens:
        assert abs(r0.dispatch[g] - r1.dispatch[g]) < 1e-9
    for b in pt.buses:
        assert abs(r0.lmp[b] - r1.lmp[b]) < 1e-9
    # a flowgate rating override applies to the copy, not the set passed
    fg = se.identity_flowgates(pt)
    before = fg.fwd.copy()
    _e2, r2 = se.rt_relief(pt, eng, r_da, {}, 0.1, flowgates=fg,
                           ratings_override={"line_4": 20.0})
    assert np.array_equal(fg.fwd, before)
    assert abs(r2.fg_flow["line_4"]) <= 20.0 + 1e-6


def test_crr_auction_on_the_constraint_set(cut):
    pt, fg = cut["pt"], cut["fg"]
    bids = [atc.Bid("3", "5", 200.0, 20.0, label="a"),
            atc.Bid("2", "9", 200.0, 15.0, label="b")]
    base = atc.crr_auction(pt, bids)
    same = atc.crr_auction(pt, bids, monitored=se.identity_flowgates(pt))
    pd.testing.assert_frame_equal(base.awards, same.awards)
    assert base.mu == same.mu and base.rent == same.rent
    tight = atc.crr_auction(pt, bids, monitored=fg)
    assert "TIES" in tight.mu
    assert tight.value <= base.value + 1e-6


# ── B3: the hub weights through the case ───────────────────────────────────
def test_hub_weights_method_on_the_nine_bus_case(lines_case):
    fp = lines_case[0]
    case = wm.CASE
    assert case.hub_weights(None, fp) == ra.hub_weights(fp, "auto")
    for kind in ("auto", "gap", "elap"):
        assert case.hub_weights(kind, fp) == ra.hub_weights(fp, kind)
    assert (case.hub_weights("auto", fp, loads=wm.LOADS_3BA, gen_fleet=wm.FLEET_3BA)
            == ra.hub_weights(fp, "auto", wm.LOADS_3BA, wm.FLEET_3BA))
    with pytest.raises(ValueError):
        case.hub_weights()
    # the module's internal route reads the method and returns the same numbers
    assert ra._hub(fp) == ra.hub_weights(fp, "auto")


def test_hub_weights_of_a_wheeling_area_are_equal_over_its_buses():
    """A footprint with buses but neither generation nor load (2026-10-25)
    takes equal weights over its buses under every kind, the WECC bundle's
    own rule for an area with no generation; a footprint with no bus still
    raises, and a footprint whose supply exists but is not of the requested
    kind raises as before."""
    from types import SimpleNamespace
    fp = SimpleNamespace(names=["W", "G"],
                         defs={"W": ["8", "9"], "G": ["1", "2"]})
    loads = {"2": 50.0}
    fleet = {"g1": {"bus": "1", "p_nom": 100.0}}
    for kind in ("auto", "gap", "elap"):
        w = ra.hub_weights(fp, kind, loads, fleet)
        assert w["W"] == {"8": 0.5, "9": 0.5}
        assert abs(sum(w["W"].values()) - 1.0) < 1e-12
    assert ra.hub_weights(fp, "auto", loads, fleet)["G"] == {"1": 1.0}
    assert ra.hub_weights(fp, "elap", loads, fleet)["G"] == {"2": 1.0}
    empty = SimpleNamespace(names=["E"], defs={"E": []})
    with pytest.raises(ValueError, match="nor a bus"):
        ra.hub_weights(empty, "auto", loads, fleet)
    only_load = SimpleNamespace(names=["L"], defs={"L": ["2"]})
    with pytest.raises(ValueError, match="no generation"):
        ra.hub_weights(only_load, "gap", loads, fleet)
    only_gen = SimpleNamespace(names=["G"], defs={"G": ["1"]})
    with pytest.raises(ValueError, match="no load"):
        ra.hub_weights(only_gen, "elap", loads, fleet)


def _pooled(pt, fp, dc_links=None, ratings=None):
    """The EDAM-literal clearing of ``fp`` on the default 9-bus case with
    every interface the partition implies declared and unlimited."""
    eng = wm.make_engine("U", buses=pt.buses)
    ifc = {k: None for k in ra.fp_interfaces(fp, pt)}
    return se.solve_engine_transfers(pt, eng, fp.defs, ifc, shed_price=150.0,
                                     dc_links=dc_links)


def test_interface_split_conserves_through_a_wheeling_area():
    """A bus-only area (bus 4: no load, no generation) sits between the bus-1
    unit and every load, so its two interfaces carry the same schedule and
    its net position is zero. Under ``te_split="interface"`` (2026-10-25) the
    pairwise settlements telescope to ``T_E``, Method 2 sums to ``R + R_T``,
    and the wheeling area's Method 2 share is half of each settlement it is
    party to."""
    pt = se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")
    defs = {"A": ["1"], "W": ["4"], "B": ["2", "3", "5", "6", "7", "8", "9"]}
    fp = fpmod.make(pt, defs, {"A": "#111", "W": "#222", "B": "#333"})
    res = _pooled(pt, fp)
    loads = wm.DEFAULT_LOADS
    W = ra._hub(fp, loads, wm.DEFAULT_GEN_FLEET)
    assert W["W"] == {"4": 1.0}
    pw = ra.pairwise_transfer_settlement(fp, res, pt, loads, weights=W)
    TE = ra.transfer_settlement(fp, res, loads, weights=W)
    assert abs(float(pw["T_ab"].sum()) - TE) < 1e-6 * max(1.0, abs(TE))
    alloc, summ, _lr, _sep = ra.allocate_congestion_rent(
        fp, res, pt, loads, unassigned_split="even", te_split="interface",
        gen_fleet=wm.DEFAULT_GEN_FLEET)
    pool = float(summ["R"]) + float(ra.transfer_rent(res))
    assert abs(float(alloc["method2"].sum()) - pool) < 1e-6 * max(1.0, pool)
    assert abs(float(summ["Nc"]["W"])) < 1e-9
    half = sum(0.5 * float(r["T_ab"]) for (a, b), r in pw.iterrows()
               if "W" in (a, b))
    assert abs(float(alloc.loc["W", "method2"]) - half) < 1e-9
    # no line is assigned to any area here, so Method 1 is the even split
    # of the unassigned rent and the wheeling area owns no line rent
    assert abs(float(alloc.loc["W", "R_own"])) < 1e-9


def test_interface_split_settles_a_controllable_link_between_its_end_areas(fp3):
    """With a DC link from area B (bus 3) into area A (bus 5) on a congested
    case, the per-area balances carry the link's delivery in each end area's
    position. Without the links the pairwise sum misses ``T_E`` by exactly
    the hub-priced link term; with ``dc_links`` passed (2026-10-25) it
    telescopes to ``T_E`` and Method 2 under the interface rule sums to
    ``R + R_dc + R_T``, the pool the hub decomposition partitions."""
    pt = se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")
    fp = fpmod.make(pt, BA3, {"A": "#111111", "B": "#222222", "C": "#333333"})
    link = {"dc1": {"bus0": "3", "bus1": "5", "p_max": 15.0}}
    res = _pooled(pt, fp, dc_links=link)
    assert res.link_flow["dc1"] > 1.0
    loads = wm.DEFAULT_LOADS
    W = ra._hub(fp, loads, wm.DEFAULT_GEN_FLEET)
    lam = {a: sum(w * res.lmp[b] for b, w in W[a].items()) for a in fp.names}
    TE = ra.transfer_settlement(fp, res, loads, weights=W)
    pw0 = ra.pairwise_transfer_settlement(fp, res, pt, loads, weights=W)
    pw1 = ra.pairwise_transfer_settlement(fp, res, pt, loads, weights=W,
                                          dc_links=link)
    link_term = res.link_flow["dc1"] * (lam["A"] - lam["B"])
    assert abs(link_term) > 1e-3
    assert abs(float(pw0["T_ab"].sum()) + link_term - TE) < 1e-6 * max(1.0, abs(TE))
    assert abs(float(pw1["T_ab"].sum()) - TE) < 1e-6 * max(1.0, abs(TE))
    assert abs(float(pw1.loc[("A", "B"), "t_link (MW)"]) + res.link_flow["dc1"]) < 1e-9
    R_dc = sum((res.lmp[link[k]["bus1"]] - res.lmp[link[k]["bus0"]]) * t
               for k, t in res.link_flow.items())
    alloc, summ, _lr, _sep = ra.allocate_congestion_rent(
        fp, res, pt, loads, unassigned_split="even", te_split="interface",
        gen_fleet=wm.DEFAULT_GEN_FLEET, dc_links=link)
    pool = float(summ["R"]) + R_dc + float(ra.transfer_rent(res))
    assert abs(float(alloc["method2"].sum()) - pool) < 1e-6 * max(1.0, pool)
    # the default carries no link and is the form every caller used before
    alloc0, _s, _l, _p = ra.allocate_congestion_rent(
        fp, res, pt, loads, unassigned_split="even", te_split="interface",
        gen_fleet=wm.DEFAULT_GEN_FLEET)
    assert abs(float(alloc0["method2"].sum()) + link_term - pool) < 1e-6 * max(1.0, pool)


# ── B4: the declared interface set ─────────────────────────────────────────
BA3 = {"A": ["1", "4", "5"], "B": ["3", "6", "7"], "C": ["2", "8", "9"]}


@pytest.fixture(scope="module")
def fp3():
    pt = wm.CASE.build().pt
    return pt, fpmod.make(pt, BA3, {"A": "#111111", "B": "#222222", "C": "#333333"})


def test_declared_set_equal_to_the_derived_one_clears_identically(fp3):
    pt, fp = fp3
    derived = ra.fp_interfaces(fp, pt)
    assert len(derived) == 3
    # both orientations, reversed order: the same declared set
    declared = [(b, a) for (a, b) in reversed(list(derived))] + list(derived)
    assert ra.declared_interfaces(fp, pt, declared) == derived
    lim = {("A", "B"): 40.0, ("A", "C"): 20.0}
    _n, _p, _e, r0 = ra.solve_with_transfers(fp, {}, limits=lim, shed_price=150.0)
    _n, _p, _e, r1 = ra.solve_with_transfers(fp, {}, limits=lim, shed_price=150.0,
                                             interfaces=declared)
    assert r0.lmp == r1.lmp and r0.transfers == r1.transfers
    assert r0.transfer_duals == r1.transfer_duals
    l0, s0 = ra.mod29_interface_limits(fp, pt)
    l1, s1 = ra.mod29_interface_limits(fp, pt, interfaces=declared)
    pd.testing.assert_frame_equal(l0, l1)
    pd.testing.assert_frame_equal(s0, s1)


def test_declared_subset_drops_an_interface(fp3):
    pt, fp = fp3
    _n, _p, _e, r = ra.solve_with_transfers(fp, {}, shed_price=150.0,
                                            interfaces=[("A", "B"), ("B", "C")])
    assert set(r.transfers) == {("A", "B"), ("B", "C")}
    with pytest.raises(ValueError):
        ra.declared_interfaces(fp, pt, [("A", "Z")])
    with pytest.raises(ValueError):
        ra.declared_interfaces(fp, pt, [("A", "A")])


def test_coalition_positions_with_a_declared_set(lines_case):
    fp, pt, _r = lines_case
    kw = dict(transfer_limits="mod29", shed_price=150.0)
    a = ra.coalition_positions(fp, pt, "BA-1", **kw)
    b = ra.coalition_positions(fp, pt, "BA-1", interfaces=[("BA-2", "BA-1")], **kw)
    assert a["delta"] == b["delta"] and a["R"] == b["R"] and a["R_T"] == b["R_T"]
    c = ra.coalition_positions(fp, pt, "BA-1", flowgates=se.identity_flowgates(pt), **kw)
    for k, v in a["delta"].items():
        assert abs(v - c["delta"][k]) < 1e-6
