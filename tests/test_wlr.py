# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Guards on ``atc.wlr_assign``, the Western Loading Relief assignment 304a uses.

The rule (``eccetf_futurestate2024`` pp. 12-15 as the public record reads it):
relief requirement = flow less rating; a tag is eligible at a TDF of 5 percent
or more in the loading direction; counterflow is neither cut nor credited; the
priority buckets are walked lowest first, every bucket below the last zeroed;
the last bucket shares the residual pro rata by impact; a tag's cut is its
relief over its TDF. The case is 204's section-6 presentation case: line_4 at
40 MW, Market A's blind tags 3->7 and 2->7, Market B's 1->5 and 1->9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import atc
import etag
import wscc9_model as wm
from seams_engine import compute_ptdf
from wscc9_model import build_network

L4 = "line_4"


@pytest.fixture(scope="module")
def pt():
    return compute_ptdf(build_network({"line_4": 40.0, "line_0": 300.0}), slack_bus="1")


def _tags(pa="7-F", pb="7-F"):
    # the scenario-1 blind dispatch of 204's presentation case, as tags
    return [dict(tag="A1", market="Market A", source="3", sink="7", priority=pa, mw=40.8603),
            dict(tag="A2", market="Market A", source="2", sink="7", priority=pa, mw=49.0697),
            dict(tag="B1", market="Market B", source="1", sink="5", priority=pb, mw=90.0),
            dict(tag="B2", market="Market B", source="1", sink="9", priority=pb, mw=173.0699)]


def _flow(pt, tags):
    li = pt.line_idx[L4]
    return sum(t["mw"] * (pt.ptdf[li, pt.bus_idx[t["source"]]]
                          - pt.ptdf[li, pt.bus_idx[t["sink"]]]) for t in tags)


def test_impacts_and_requirement(pt):
    led = atc.wlr_assign(_tags(), pt, L4)
    imp = dict(zip(led["tag"], led["impact"]))
    assert imp["A1"] == pytest.approx(34.81, abs=0.01)
    assert imp["A2"] == pytest.approx(5.19, abs=0.01)
    assert imp["B1"] == pytest.approx(-12.16, abs=0.01)
    assert imp["B2"] == pytest.approx(21.61, abs=0.01)
    assert led.attrs["R"] == pytest.approx(_flow(pt, _tags()) - 40.0)


def test_conservation_sum_relief_equals_requirement(pt):
    for pa, pb in (("7-F", "7-F"), ("2-NH", "7-F"), ("7-F", "2-NH")):
        led = atc.wlr_assign(_tags(pa, pb), pt, L4)
        assert led["relief"].sum() == pytest.approx(led.attrs["R"], abs=1e-9)
        assert led.attrs["unassigned"] == pytest.approx(0.0, abs=1e-9)


def test_counterflow_neither_cut_nor_credited(pt):
    led = atc.wlr_assign(_tags(), pt, L4).set_index("tag")
    assert not led.loc["B1", "eligible"]
    assert led.loc["B1", "relief"] == 0.0 and led.loc["B1", "cut"] == 0.0
    # not credited: the requirement is the flow less the rating, the flow
    # including the counterflow tag; removing the counterflow raises R by its impact
    no_cf = [t for t in _tags() if t["tag"] != "B1"]
    led2 = atc.wlr_assign(no_cf, pt, L4)
    assert led2.attrs["R"] - atc.wlr_assign(_tags(), pt, L4).attrs["R"] == pytest.approx(12.16, abs=0.01)


def test_threshold_screens_small_tdf(pt):
    # 2->7 has TDF 0.106: eligible at 5 percent, screened out at 11 percent
    led = atc.wlr_assign(_tags(), pt, L4, threshold=0.11).set_index("tag")
    assert not led.loc["A2", "eligible"] and led.loc["A2", "relief"] == 0.0
    assert led.loc["A1", "eligible"] and led.loc["B2", "eligible"]
    assert led["relief"].sum() == pytest.approx(led.attrs["R"])


def test_pro_rata_by_impact_is_an_equal_fractional_cut(pt):
    led = atc.wlr_assign(_tags(), pt, L4)
    el = led[led["eligible"]]
    share = el["relief"] / el["impact"]
    assert np.allclose(share, share.iloc[0])                 # relief proportional to impact
    frac = el["cut"] / el["MW"]
    assert np.allclose(frac, frac.iloc[0])                   # equal fractional cut
    assert np.allclose(el["cut"] * el["TDF"], el["relief"])  # cut = relief / TDF
    a_share = led.loc[led["market"] == "Market A", "relief"].sum()
    assert a_share == pytest.approx(led.attrs["R"] * 40.0 / 61.61, abs=0.01)


def test_bucket_walk_lowest_priority_first(pt):
    led = atc.wlr_assign(_tags("2-NH", "7-F"), pt, L4).set_index("tag")
    assert led.loc["B2", "relief"] == 0.0                    # the firm bucket untouched
    assert led.loc[["A1", "A2"], "relief"].sum() == pytest.approx(led.attrs["R"])
    assert led.attrs["last_bucket"] == 2
    # a lower bucket too small for R is zeroed before the next is touched
    small = _tags("7-F", "7-F")
    small[1]["priority"] = "2-NH"                            # A2 alone non-firm, impact 5.19
    led2 = atc.wlr_assign(small, pt, L4).set_index("tag")
    assert led2.loc["A2", "MW kept"] == pytest.approx(0.0, abs=1e-9)
    assert led2.loc["A2", "relief"] == pytest.approx(5.19, abs=0.01)
    rest = led2.attrs["R"] - led2.loc["A2", "relief"]
    assert led2.loc[["A1", "B2"], "relief"].sum() == pytest.approx(rest)
    assert led2.attrs["last_bucket"] == 7


def test_matches_reduce_pro_rata_within_one_tier(pt):
    """On an all-tagged set at one priority, the cut equals the netted
    _reduce_pro_rata reduction with the 5 percent screen (relief by impact)."""
    tags = _tags()
    li = pt.line_idx[L4]
    A = np.array([[pt.ptdf[li, pt.bus_idx[t["source"]]] - pt.ptdf[li, pt.bus_idx[t["sink"]]]
                   for t in tags]])
    t0 = np.array([t["mw"] for t in tags])
    x, _, _ = atc._reduce_pro_rata(A, t0, np.array([40.0]), np.array([40.0]), [L4],
                                   "netted", 20, 1e-9, rf_min=0.05)
    led = atc.wlr_assign(tags, pt, L4)
    assert np.allclose(led["MW kept"].to_numpy(), x, atol=1e-6)


def test_gtl_component_equals_the_tags_it_replaces(pt):
    gtl = atc.wlr_gtl_component(pt, L4, {"3": 40.8603, "2": 49.0697}, {"7": 89.93},
                                market="Market A")
    assert gtl["kind"] == "GTL" and gtl["priority"] == "7-F"
    led = atc.wlr_assign([gtl] + _tags()[2:], pt, L4)
    tags_led = atc.wlr_assign(_tags(), pt, L4)
    assert led.loc[0, "impact"] == pytest.approx(
        tags_led.loc[tags_led["market"] == "Market A", "impact"].sum())
    assert np.isnan(led.loc[0, "cut"])
    assert led.loc[0, "relief"] == pytest.approx(
        tags_led.loc[tags_led["market"] == "Market A", "relief"].sum())


def test_no_relief_below_rating(pt):
    led = atc.wlr_assign(_tags(), pt, L4, rating=60.0)
    assert led.attrs["R"] == 0.0 and (led["relief"] == 0).all() and (led["cut"] == 0).all()


def test_etag_accepts_nerc_codes_and_keeps_legacy():
    s = etag.Segment("BA-B", por="1", pod="9", reservation_mw=10.0, priority="2-NH")
    assert not s.firm and etag.priority_level("2-NH") == 2
    assert etag.Segment("BA-B", "1", "9", 10.0, priority="7-F").firm
    assert etag.Segment("BA-B", "1", "9", 10.0).firm            # default 'firm'
    assert not etag.Segment("BA-B", "1", "9", 10.0, priority="non-firm").firm
    with pytest.raises(ValueError):
        etag.Segment("BA-B", "1", "9", 10.0, priority="8-X")


def test_fleet_with_backstop_per_bus_sizes_and_unchanged_default():
    f = wm.fleet_with_backstop(price=55.0, p_nom={"5": 5, "7": 25, "9": 80})
    assert {g: f[g]["p_nom"] for g in f if g.startswith("backstop_")} == {
        "backstop_5": 5.0, "backstop_7": 25.0, "backstop_9": 80.0}
    d = wm.fleet_with_backstop()
    assert d["backstop_9"] == {"bus": "9", "cost": 90.0, "p_nom": 125.0}
    assert wm.fleet_with_backstop(p_nom=10.0)["backstop_5"]["p_nom"] == 10.0


def test_priority_tables_agree():
    """atc keeps its own copy of the NERC priority levels; it must match etag's."""
    assert atc.WLR_PRIORITY_LEVELS == etag.NERC_PRIORITIES


def test_net_injection_panel_matches_gtl_component(pt):
    """The net-injection panel (304a's alternative GTL reading) boxes the impact
    sum_n SF_n x_n of a balanced injection set, which is wlr_gtl_component's
    impact on the same untagged generation and load, and rejects an unbalanced set."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import wscc9_figures as figs
    gen, load = {"1": 163.07}, {"9": 73.07, "5": 90.0}
    comp = atc.wlr_gtl_component(pt, L4, gen, load, market="Market B")
    x = {"1": 163.07, "9": -73.07, "5": -90.0}
    fig, ax = figs.net_injection_flow_panel(build_network({"line_4": 40.0, "line_0": 300.0}),
                                            pt, x, line=L4, label="B GTL")
    box = [t.get_text() for t in ax.texts if "impact" in t.get_text()][0]
    assert f"impact {comp['mw'] * comp['tdf']:+.1f} MW (counterflow)" in box
    assert f"impact / injected MW {comp['tdf']:+.3f}" in box
    assert "not eligible" in box
    plt.close(fig)
    with pytest.raises(ValueError):
        figs.net_injection_flow_panel(None, pt, {"1": 10.0, "9": -5.0}, line=L4)
