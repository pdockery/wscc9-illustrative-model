"""The flowgate layer must be a STRICT generalization of per-line limits.

Three claims, each tested on the 9-bus case:

1. ``identity_flowgates`` reproduces the per-line clearing exactly — same
   dispatch, LMPs, duals, to machine precision.
2. A genuine multi-line flowgate binds as a signed aggregate, and the LMP
   identity survives through the implied per-line duals:
   ``λ_n = λ + Σ_m SF_{n,m} μ^impl_m`` with ``μ^impl = Sᵀ μ_fg``.
3. Asymmetric fwd/rev ratings bind on the correct side, and an unrated
   direction (large sentinel) never binds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seams_engine as se  # noqa: E402
import wscc9_model as wm   # noqa: E402


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")


@pytest.fixture(scope="module")
def engine(pt):
    return wm.make_engine("UNIFIED", buses=pt.buses)


def test_identity_flowgates_reproduce_lines(pt, engine):
    base = se.solve_engine_dispatch(pt, engine, shed_price=150.0)
    fg = se.identity_flowgates(pt)
    same = se.solve_engine_dispatch(pt, engine, shed_price=150.0, flowgates=fg)
    for g in base.dispatch:
        assert abs(base.dispatch[g] - same.dispatch[g]) < 1e-7
    for b in pt.buses:
        assert abs(base.lmp[b] - same.lmp[b]) < 1e-7
    for l in pt.lines:
        assert abs(base.line_dual[l] - same.line_dual[l]) < 1e-7
        assert abs(same.fg_dual[l] - same.line_dual[l]) < 1e-12   # S = I
    assert same.fg_limit["line_4"] == (40.0, 40.0)


def test_two_line_flowgate_binds_and_identity_holds(pt, engine):
    # A 2-member signed flowgate over the corridor lines 4 (6-7) and 5 (7-8):
    # cap the AGGREGATE below its unconstrained flow so it must bind.
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx["line_4"]] = 1.0
    S[0, pt.line_idx["line_5"]] = -1.0
    fg = se.FlowgateSet(ids=["corridor"], S=S,
                        fwd=np.array([25.0]), rev=np.array([9999.0]),
                        members={"corridor": [("line_4", 1.0), ("line_5", -1.0)]})
    res = se.solve_engine_dispatch(pt, engine, shed_price=150.0, flowgates=fg)

    # the aggregate sits at its forward rating and carries a dual
    f = res.fg_flow["corridor"]
    assert abs(f - 25.0) < 1e-6, f
    mu_k = res.fg_dual["corridor"]
    assert abs(mu_k) > 1e-6

    # implied per-line duals: sign_m * mu_k on members, 0 elsewhere
    assert abs(res.line_dual["line_4"] - mu_k) < 1e-9
    assert abs(res.line_dual["line_5"] + mu_k) < 1e-9
    for l in pt.lines:
        if l not in ("line_4", "line_5"):
            assert abs(res.line_dual[l]) < 1e-9

    # the LMP identity through the implied duals, at every bus, to 1e-9
    for n, b in enumerate(pt.buses):
        cong = sum(pt.ptdf[pt.line_idx[l], n] * res.line_dual[l]
                   for l in pt.lines)
        assert abs(res.lmp[b] - (res.energy_price + cong)) < 1e-9

    # flowgate flow equals the signed sum of member line flows
    agg = res.flow_own["line_4"] - res.flow_own["line_5"]
    assert abs(agg - f) < 1e-9


def test_asymmetric_ratings_bind_directionally(pt, engine):
    # Same aggregate, but rated only in REVERSE: forward unrated (sentinel).
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx["line_4"]] = 1.0
    fg = se.FlowgateSet(ids=["p4"], S=S,
                        fwd=np.array([9999.0]), rev=np.array([10.0]),
                        members={"p4": [("line_4", 1.0)]})
    res = se.solve_engine_dispatch(pt, engine, shed_price=150.0, flowgates=fg)
    # line_4's natural flow is FORWARD (~63 MW unconstrained) — the reverse
    # rating must not touch it and no dual should appear.
    assert res.fg_flow["p4"] > 11.0
    assert abs(res.fg_dual["p4"]) < 1e-9

    fg2 = se.FlowgateSet(ids=["p4"], S=S.copy(),
                         fwd=np.array([30.0]), rev=np.array([9999.0]),
                         members={"p4": [("line_4", 1.0)]})
    res2 = se.solve_engine_dispatch(pt, engine, shed_price=150.0, flowgates=fg2)
    assert abs(res2.fg_flow["p4"] - 30.0) < 1e-6      # forward rating binds
    assert abs(res2.fg_dual["p4"]) > 1e-6


def test_transfers_form_accepts_flowgates(pt, engine):
    fg = se.identity_flowgates(pt)
    areas = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    base = se.solve_engine_transfers(pt, engine, areas,
                                     {("BA-1", "BA-2"): None}, shed_price=150.0)
    same = se.solve_engine_transfers(pt, engine, areas,
                                     {("BA-1", "BA-2"): None}, shed_price=150.0,
                                     flowgates=fg)
    for b in pt.buses:
        assert abs(base.lmp[b] - same.lmp[b]) < 1e-7
    for a in base.area_prices:
        assert abs(base.area_prices[a] - same.area_prices[a]) < 1e-7


def test_qp_form_accepts_flowgates(pt, engine):
    curves = {"gen_slack_0": (50.0, 0.10), "gen_0": (35.0, 0.05),
              "gen_1": (20.0, 0.03)}
    fg = se.identity_flowgates(pt)
    base = se.solve_engine_qp(pt, engine, curves=curves, shed_price=150.0)
    same = se.solve_engine_qp(pt, engine, curves=curves, shed_price=150.0,
                              flowgates=fg)
    for b in pt.buses:
        assert abs(base.lmp[b] - same.lmp[b]) < 1e-5
    for l in pt.lines:
        assert abs(base.line_dual[l] - same.line_dual[l]) < 1e-5
