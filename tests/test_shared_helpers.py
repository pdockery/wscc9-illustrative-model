# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Smoke tests for the helpers extracted from the notebooks (2026-08-20):
seams_engine.rt_relief / actual_flows / physical_flows(exo_physical=),
footprints.line_status_table / ba_line_colors, atc.post_contingency_* /
lodf_derate, wscc9_figures.install_latex_repr, seam_views.configure /
relief_view."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import atc
import footprints
import seam_views
import seams_engine
import wscc9_figures
import wscc9_model as wm
from seams_engine import actual_flows, compute_ptdf, rt_relief, solve_engine_dispatch
from wscc9_model import build_network, make_engine

MKT = {"Market A": ["2", "8", "7", "6", "3"], "Market B": ["1", "9", "4", "5"]}
MON = {"Market A": ["line_2", "line_3", "line_4", "line_5", "line_6"],
       "Market B": ["line_0", "line_1", "line_7", "line_8"]}


def _case():
    net = build_network({"line_4": 40.0})
    pt = compute_ptdf(net, slack_bus="1")
    fp = footprints.make(pt, MKT, {"Market A": "#993AFF", "Market B": "#2471A3"},
                         monitored=MON, tie_label="intertie")
    A = make_engine("Market A", MKT["Market A"], activated=MON["Market A"])
    B = make_engine("Market B", MKT["Market B"], activated=MON["Market B"])
    return pt, fp, A, B


def test_rt_relief_clears_the_overload():
    pt, _, A, B = _case()
    rA = solve_engine_dispatch(pt, A)
    rB = solve_engine_dispatch(pt, B)
    assert seams_engine.physical_flows(pt, [rA, rB])["overload"].any()
    eng_rt, r_rt = rt_relief(pt, A, rA, rB.flow_own, premium=0.5)
    pf = seams_engine.physical_flows(pt, [r_rt, rB])
    assert not pf["overload"].any()
    assert r_rt.total_cost >= rA.total_cost  # the relief is paid at the premium
    assert any(g.endswith("__inc") for g in eng_rt.gens)


def test_rt_relief_ratings_override():
    pt, _, A, B = _case()
    rA = solve_engine_dispatch(pt, A)
    rB = solve_engine_dispatch(pt, B)
    _, r_rt = rt_relief(pt, A, rA, rB.flow_own, premium=0.5,
                        ratings_override={"line_4": 35.0})
    assert abs(r_rt.flow_own["line_4"] + rB.flow_own["line_4"]) <= 35.0 + 1e-6


def test_actual_flows_matches_injections():
    pt, _, A, B = _case()
    rA = solve_engine_dispatch(pt, A)
    rB = solve_engine_dispatch(pt, B)
    f = actual_flows(pt, [rA, rB])
    assert np.allclose(f, pt.ptdf @ (rA.injection + rB.injection), atol=1e-9)


def _case_205a():
    """205a's exhibit case: the bus-1 unit at 300 MW on a 300 MW line_0, the
    realized loads the transaction exhibits carry, line_4 rated 40 MW."""
    fleet = {g: dict(s) for g, s in wm.DEFAULT_GEN_FLEET.items()}
    fleet["gen_slack_0"]["p_nom"] = 300.0
    net = build_network({"line_4": 40.0, "line_0": 300.0})
    pt = compute_ptdf(net, slack_bus="1")
    A = make_engine("Market A", MKT["Market A"], gen_fleet=fleet,
                    activated=MON["Market A"])
    A.loads["7"] = 90.0
    B = make_engine("Market B", MKT["Market B"], gen_fleet=fleet,
                    activated=MON["Market B"])
    B.loads["9"], B.loads["5"] = 173.0, 90.0
    return pt, A, B


def test_physical_flows_exo_physical_on_the_205a_import():
    """205b's rule on 205a's Example 2, the 25 MW import from bus 1 to bus 7.

    The schedule's two legs are realized in the two books, so the physical flow
    under ``exo_physical=False`` is Example 1's, 49.44 MW on line_4, and the
    default reading, which adds both scheduled injections, gives 61.11 MW."""
    pt, A, B = _case_205a()
    rA1 = solve_engine_dispatch(pt, A, shed_price=150.0)
    rB1 = solve_engine_dispatch(pt, B, shed_price=150.0)
    rA2 = solve_engine_dispatch(pt, A, exo={"1": +25.0}, shed_price=150.0)
    rB2 = solve_engine_dispatch(pt, B, exo={"7": -25.0}, shed_price=150.0)

    base = float(actual_flows(pt, [rA1, rB1])[pt.line_idx["line_4"]])
    assert abs(base - 49.4374) < 1e-3

    act = seams_engine.physical_flows(pt, [rA2, rB2], exo_physical=False)
    # the frame rounds its flow columns to 0.1 MW, so the check carries that
    assert abs(float(act.loc["line_4", "F_phys"]) - base) < 0.05
    # the per-engine columns are the actual contributions and sum to the total
    assert abs(float(act.loc["line_4", "F[Market A]"])
               + float(act.loc["line_4", "F[Market B]"])
               - float(act.loc["line_4", "F_phys"])) < 1e-6

    dflt = seams_engine.physical_flows(pt, [rA2, rB2])
    assert abs(float(dflt.loc["line_4", "F_phys"]) - 61.11) < 0.05
    # the default is unchanged by the new keyword: it still sums the injections
    assert np.allclose(dflt["F_phys"].to_numpy(float),
                       np.round(pt.ptdf @ (rA2.injection + rB2.injection), 1),
                       atol=1e-9)


def test_physical_flows_rejects_an_unbalanced_actual_sum():
    """An exogenous schedule that no engine in the set books leaves the summed
    generation and served load out of balance, and the actual-dispatch reading
    refuses it rather than returning a flow that misses the injection."""
    pt, A, B = _case_205a()
    rA = solve_engine_dispatch(pt, A, exo={"1": +25.0}, shed_price=150.0)
    rB = solve_engine_dispatch(pt, B, shed_price=150.0)
    try:
        seams_engine.physical_flows(pt, [rA, rB], exo_physical=False)
    except ValueError as exc:
        assert "out of balance" in str(exc)
    else:
        raise AssertionError("an unbalanced actual sum must raise")


def test_rt_relief_carries_the_host_schedule():
    """The host's own exogenous schedule stays in its real-time book, so its
    real-time generation still covers the schedule it carried day-ahead."""
    pt, A, B = _case_205a()
    rA = solve_engine_dispatch(pt, A, exo={"5": -30.0}, shed_price=150.0)
    rB = solve_engine_dispatch(pt, B, exo={"3": +30.0}, shed_price=150.0)
    phys = actual_flows(pt, [rA, rB])
    offs = {l: float(phys[pt.line_idx[l]]) - float(rA.flow_own[l]) for l in pt.lines}

    _, r_rt = rt_relief(pt, A, rA, offs, premium=0.0, shed_price=150.0)
    assert abs(sum(r_rt.dispatch.values()) - (90.0 + 30.0)) < 1e-6
    after = actual_flows(pt, [r_rt, rB])
    assert abs(float(after[pt.line_idx["line_4"]])) <= 40.0 + 1e-6
    # at a zero premium the bill is the shadow price times the exceedance
    bill = r_rt.total_cost - rA.total_cost
    excess = abs(float(phys[pt.line_idx["line_4"]])) - 40.0
    assert abs(bill - abs(r_rt.line_dual["line_4"]) * excess) < 0.01

    # dropping the schedule leaves the host serving its own load alone
    _, r_none = rt_relief(pt, A, rA, offs, premium=0.0, shed_price=150.0, exo={})
    assert abs(sum(r_none.dispatch.values()) - 90.0) < 1e-6


def test_relief_view_draws_the_host_chair():
    """seam_views.relief_view returns a figure and names the megawatts the host
    moved, on 205a's case with Example 3's export in Market A's book."""
    import warnings

    import matplotlib
    import matplotlib.pyplot as plt

    prev = matplotlib.get_backend()            # restored below; no global mutation
    plt.switch_backend("Agg")                  # no window; the figure is closed below

    pt, A, B = _case_205a()
    fleet = {g: dict(s) for g, s in wm.DEFAULT_GEN_FLEET.items()}
    fleet["gen_slack_0"]["p_nom"] = 300.0
    net = build_network({"line_4": 40.0, "line_0": 300.0})
    fp = footprints.make(pt, MKT, {"Market A": "#993AFF", "Market B": "#2471A3"},
                         monitored=MON, tie_label="intertie")
    seam_views.configure(fp=fp, gen_fleet=fleet, loads=dict(wm.DEFAULT_LOADS),
                         save_enabled=False)

    r_da = solve_engine_dispatch(pt, A, exo={"5": -30.0}, shed_price=150.0)
    rB = solve_engine_dispatch(pt, B, exo={"3": +30.0}, shed_price=150.0)
    phys = actual_flows(pt, [r_da, rB])
    offs = {l: float(phys[pt.line_idx[l]]) - float(r_da.flow_own[l]) for l in pt.lines}
    r_rt = solve_engine_dispatch(pt, A, exo={"5": -30.0}, flow_offsets=offs,
                                 shed_price=150.0)

    with warnings.catch_warnings():            # relief_view calls plt.show()
        warnings.simplefilter("ignore", UserWarning)
        fig = seam_views.relief_view(net, pt, fp, "Market A", A, r_da, r_rt, offs,
                                     suptitle="smoke", save=None)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        # the default legend note carries every bus whose generation rose, and
        # the megawatts it names are r_rt less r_da at that bus
        up = {b: float(r_rt.gen_by_bus.get(b, 0.0)) - float(r_da.gen_by_bus.get(b, 0.0))
              for b in set(r_da.gen_by_bus) | set(r_rt.gen_by_bus)}
        up = {b: mw for b, mw in up.items() if mw > 1e-6}
        assert up, "the relief must move at least one bus up"
        notes = [t.get_text() for t in fig.findobj(plt.Text)]
        blob = " ".join(notes)
        for b, mw in up.items():
            assert f"{mw:.2f} MW at bus {b}" in blob, (b, mw, blob[:400])
        # and it clears around the fixed flow it was handed
        worst = max(A.activated_lines, key=lambda l: abs(offs[l]))
        assert f"of fixed flow on {worst}" in blob
    finally:
        plt.close(fig)
        plt.switch_backend(prev)


def test_footprint_helpers():
    pt, fp, _, _ = _case()
    st = footprints.line_status_table(fp, pt, MON, MON)
    assert st.loc["line_4", "Market A"] == "enforces"
    assert st.loc["line_4", "Market B"] == "--"
    assert st.loc["line_2", "location"] == "intertie"
    cols = footprints.ba_line_colors(fp, pt)
    assert cols["line_4"] == "#993AFF"
    assert cols["line_2"] == footprints.UNASSIGNED_COLOR
    green = footprints.ba_line_colors(fp, pt, tie_color="#117A2B")
    assert green["line_2"] == "#117A2B"


def test_contingency_helpers():
    net = build_network(split_5_6=True)   # the 301/302 double-circuit 5-6 corridor
    pt = compute_ptdf(net, slack_bus="1")
    pt_ctg = atc.post_contingency_ptdf(build_network(split_5_6=True), "line_2_ba2")
    assert "line_2_ba2" not in list(pt_ctg.lines)
    uni = seams_engine.MarketEngine(
        name="U", gens={g: dict(s) for g, s in wm.DEFAULT_GEN_FLEET.items()},
        loads=dict(wm.DEFAULT_LOADS), activated_lines="all")
    r = solve_engine_dispatch(pt, uni)
    post = atc.post_contingency_flow(pt_ctg, r, "line_2_ba1")
    lodf, derate = atc.lodf_derate(pt_ctg, r, "line_2_ba1", 75.0)
    assert abs(post) > abs(r.flow_own["line_2_ba1"]) - 1e-6
    assert 0.0 < derate < 75.0 + 1e-6


def test_seam_positions_closure():
    import revenue_allocation as ra
    pt, fp, A, B = _case()
    rA = solve_engine_dispatch(pt, A)
    rB = solve_engine_dispatch(pt, B)
    out = ra.seam_positions((A, rA), (B, rB), unified_cost=0.0)
    for m in ("Market A", "Market B"):
        closes = (-out[(m, "consumers")]
                  - out[(m, "gen revenue")] - out[(m, "congestion revenue")])
        assert abs(closes) < 1e-6  # consumers = gen revenue + congestion (no trader leg)
    assert out[("Trader", "profit")] == 0.0


def test_display_and_views_config():
    wscc9_figures.install_latex_repr()   # no-op outside IPython, must not raise
    seam_views.configure(gen_fleet=dict(wm.DEFAULT_GEN_FLEET),
                         loads=dict(wm.DEFAULT_LOADS))
    try:
        seam_views.configure(bogus_key=1)
    except KeyError:
        pass
    else:
        raise AssertionError("configure accepted an unknown key")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("all shared-helper smoke tests pass")
