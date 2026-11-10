"""Phase-4 surfaces: atc-on-flowgates, the two-level partition, T1 spec forms,
and the flowgate rent table."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import atc                     # noqa: E402
import footprints as fpmod     # noqa: E402
import revenue_allocation as ra  # noqa: E402
import seams_engine as se      # noqa: E402
import wscc9_model as wm       # noqa: E402


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")


def test_atc_ttc_on_flowgates_matches_lines_at_identity(pt):
    fg = se.identity_flowgates(pt)
    mw_l, line_l = atc.ttc(pt, "3", "5")
    mw_f, id_f = atc.ttc(pt, "3", "5", monitored=fg)
    assert abs(mw_l - mw_f) < 1e-9 and line_l == id_f


def test_atc_sft_on_flowgates(pt):
    fg = se.identity_flowgates(pt)
    awards = [atc.Award("3", "5", 40.0), atc.Award("2", "9", 30.0)]
    ok_l, df_l = atc.simultaneous_feasibility(pt, awards)
    ok_f, df_f = atc.simultaneous_feasibility(pt, awards, monitored=fg)
    assert ok_l == ok_f
    assert (df_l["overload"].values == df_f["overload"].values).all()
    okF_l, dfF_l = atc.firm_feasibility(pt, awards)
    okF_f, dfF_f = atc.firm_feasibility(pt, awards, monitored=fg)
    assert okF_l == okF_f
    assert (dfF_l["firm"].values == dfF_f["firm"].values).all()


def test_atc_ttc_direction_aware_on_asymmetric_flowgate(pt):
    # a one-member path rated 30 fwd / unrated reverse: the 3->5 transfer loads
    # line_4 forward, so its TTC through this constraint uses the FWD cap ...
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx["line_4"]] = 1.0
    a = atc.path_shift_factors(pt, "3", "5")[pt.line_idx["line_4"]]
    assert a > 0
    fg = se.FlowgateSet(ids=["p4"], S=S, fwd=np.array([30.0]),
                        rev=np.array([9999.0]), members={})
    mw, which = atc.ttc(pt, "3", "5", monitored=fg)
    assert which == "p4" and abs(mw - 30.0 / a) < 1e-9
    # ... and the REVERSE transfer (5->3) hits the unrated sentinel instead.
    mw_r, _ = atc.ttc(pt, "5", "3", monitored=fg)
    assert mw_r > 100 * mw


def test_two_level_partition(pt):
    BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    AREAS = {"a21": ["2", "8"], "a22": ["7", "6", "3"],
             "a11": ["1", "9"], "a12": ["4", "5"]}
    SUB = {"BA-1": ["a21", "a22"], "BA-2": ["a11", "a12"]}
    fp = fpmod.make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"},
                    sub=SUB, area_defs=AREAS)
    assert set(fp.areas_of()) == set(AREAS)
    assert set(fp.areas_of("BA-1")) == {"a21", "a22"}
    # the finer interfaces exist and cover more pairs than the footprint ones
    fine = ra.fp_interfaces(fp, pt, areas=fp.areas_of())
    coarse = ra.fp_interfaces(fp, pt)
    assert len(fine) > len(coarse) >= 1
    # a non-tiling area set must raise
    with pytest.raises(ValueError):
        fpmod.make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"},
                   sub={"BA-1": ["a21"], "BA-2": ["a11", "a12"]},
                   area_defs={"a21": ["2", "8"], "a11": ["1", "9"],
                              "a12": ["4", "5"]})
    # no two-level structure declared -> areas_of() is the footprints
    fp0 = fpmod.make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"})
    assert fp0.areas_of() == fp0.defs


def test_t1_accepts_every_spec_form(pt):
    BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    fp = fpmod.make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"})
    n, p, e, res = ra.solve_with_transfers(fp, {"line_4": 40.0}, limits=120.0,
                                           shed_price=150.0)
    RT = ra.transfer_rent(res)
    d = ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS, 1,
                                  {"BA-1": 0.5, "BA-2": 0.5}, pt=p)
    ev = ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS, 1, "even", pt=p)
    sc = ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS, 1, 0.5, pt=p)
    ic = ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS, 1, "interchange",
                                   pt=p)
    for out in (d, ev, sc, ic):
        assert abs(sum(out.values()) - RT) < 1e-6
    assert d == ev == sc          # all 50/50 at two footprints
    # 'interchange': |e_1| = |e_2| in exact arithmetic at two footprints, so
    # the shares are 50/50 up to float noise in the two net positions
    for ba in d:
        assert abs(ic[ba] - d[ba]) < 1e-6 * max(1.0, abs(RT))


def test_flowgate_rent_table(pt):
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx["line_4"]] = 1.0
    fg = se.FlowgateSet(ids=["p4"], S=S, fwd=np.array([30.0]),
                        rev=np.array([9999.0]), members={})
    eng = wm.make_engine("U", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=150.0, flowgates=fg)
    df = ra.flowgate_rent_table(res)
    assert df.loc["p4", "binding"]
    assert abs(df.loc["p4", "rent"] - abs(res.fg_dual["p4"] * res.fg_flow["p4"])) < 1e-9
    assert abs(df.loc["TOTAL", "rent"] - df.loc["p4", "rent"]) < 1e-9
    # the signed flowgate rent equals the signed implied per-line rent
    signed_fg = sum(res.fg_dual[k] * res.fg_flow[k] for k in res.fg_dual)
    signed_ln = sum(res.line_dual[l] * res.flow_own[l] for l in res.line_dual)
    assert abs(signed_fg - signed_ln) < 1e-9
