"""Must-run bounds and group capability caps on the transfers solver.

``solve_engine_transfers`` gains two inputs that a storage fleet needs and
a generator fleet does not. ``gen_min`` is a lower bound on a named
generator's dispatch, which is how a must-run block reaches the clearing.
``group_caps`` is an upper bound on the TOTAL of a set of generators,
which is how a storage's hourly capability reaches it: the capability is a
fleet quantity and the members' own capacities each admit more than the
fleet can deliver.

Both are inert when absent, so every existing call site clears the same
program it cleared before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seams_engine as se      # noqa: E402
import wscc9_model as wm       # noqa: E402

RATINGS = {"line_4": 40.0}
BA = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network(RATINGS), slack_bus="1")


def _eng(pt):
    return wm.make_engine("U", buses=pt.buses)


def _clear(pt, **kw):
    return se.solve_engine_transfers(pt, _eng(pt), BA,
                                     {("BA-1", "BA-2"): None},
                                     shed_price=150.0, **kw)


def test_absent_inputs_change_nothing(pt):
    """The canonical two-BA clearing reproduces exactly when the two new
    inputs are absent, passed as None, and passed empty."""
    base = _clear(pt)
    for kw in ({}, dict(gen_min=None, group_caps=None),
               dict(gen_min={}, group_caps=[])):
        r = _clear(pt, **kw)
        assert r.dispatch == base.dispatch
        assert r.lmp == base.lmp
        assert abs(r.total_cost - base.total_cost) < 1e-9
        assert r.group_cap_dual == []
    # the canonical congestion rent of the 201 anchor
    rent = sum(abs(mu) * abs(base.flow_own[m])
               for m, mu in base.line_dual.items())
    assert abs(rent - 804.0) < 1.0


def test_gen_min_raises_a_units_dispatch(pt):
    """A must-run bound runs the named generator at least that much, and
    production cost rises because the bound displaces cheaper energy."""
    base = _clear(pt)
    idle = [g for g, mw in base.dispatch.items() if mw < 1e-6]
    assert idle, "the canonical case needs an idle unit for this test"
    g = idle[0]
    r = _clear(pt, gen_min={g: 10.0})
    assert r.dispatch[g] >= 10.0 - 1e-6
    assert r.total_cost > base.total_cost + 1e-6
    r2 = _clear(pt, gen_min={g: 20.0})
    assert r2.dispatch[g] >= 20.0 - 1e-6
    assert r2.total_cost >= r.total_cost - 1e-6


def test_gen_min_rejects_a_bound_it_cannot_meet(pt):
    eng = _eng(pt)
    g = list(eng.gens)[0]
    big = float(eng.gens[g]["p_nom"]) + 1.0
    with pytest.raises(ValueError):
        _clear(pt, gen_min={g: big})
    with pytest.raises(ValueError):
        _clear(pt, gen_min={"not_a_unit": 1.0})


def test_group_cap_binds_with_a_positive_dual(pt):
    """A cap below two units' combined optimum binds exactly, carries a
    positive dual, and leaves the allocation between them to the engine."""
    base = _clear(pt)
    gs = sorted(base.dispatch, key=lambda g: -base.dispatch[g])[:2]
    tot = sum(base.dispatch[g] for g in gs)
    assert tot > 30.0
    cap = tot - 20.0
    r = _clear(pt, group_caps=[(gs, cap)])
    assert abs(sum(r.dispatch[g] for g in gs) - cap) < 1e-6
    assert len(r.group_cap_dual) == 1
    assert r.group_cap_dual[0] > 1e-6
    assert r.total_cost > base.total_cost + 1e-6
    # a cap ABOVE the optimum is inert
    slack = _clear(pt, group_caps=[(gs, tot + 50.0)])
    assert slack.dispatch == base.dispatch
    assert slack.group_cap_dual[0] < 1e-9


def test_group_caps_reject_an_unknown_member(pt):
    with pytest.raises(ValueError):
        _clear(pt, group_caps=[(["not_a_unit"], 10.0)])


def test_a_demand_bid_above_the_price_clears_in_full(pt):
    """A price-sensitive demand bid priced above the local price clears at
    its full size, raises the area's served load by that much, and comes
    back in demand_cleared."""
    base = _clear(pt)
    bus = "5"                                   # a BA-2 bus, the dear area
    p0 = base.lmp[bus]
    r = _clear(pt, demand_bids=[{"id": "charge", "bus": bus, "mw": 5.0,
                                 "price": p0 + 100.0}])
    assert abs(r.demand_cleared["charge"] - 5.0) < 1e-6
    gen0 = sum(base.dispatch.values())
    gen1 = sum(r.dispatch.values())
    shed0 = sum((base.shed_by_bus or {}).values())
    shed1 = sum((r.shed_by_bus or {}).values())
    assert abs((gen1 + shed1) - (gen0 + shed0) - 5.0) < 1e-6
    # the withdrawal is in the net position the tags settle
    k = ("BA-1", "BA-2")
    assert r.transfers[k] > base.transfers[k] - 1e-6


def test_a_demand_bid_below_the_price_does_not_clear(pt):
    """A bid priced under the local price clears nothing and leaves the
    program exactly where it was."""
    base = _clear(pt)
    bus = "5"
    r = _clear(pt, demand_bids=[{"id": "charge", "bus": bus, "mw": 5.0,
                                 "price": base.lmp[bus] - 5.0}])
    assert r.demand_cleared == {}
    assert r.dispatch == base.dispatch
    assert r.lmp == base.lmp


def test_demand_bids_are_inert_when_absent(pt):
    """Absent, None and empty all clear the canonical program."""
    base = _clear(pt)
    for kw in (dict(demand_bids=None), dict(demand_bids=[]),
               dict(demand_bids=[{"id": "z", "bus": "5", "mw": 0.0,
                                  "price": 999.0}])):
        r = _clear(pt, **kw)
        assert r.dispatch == base.dispatch
        assert r.lmp == base.lmp
        assert r.demand_cleared == {}


def test_demand_bids_coexist_with_the_vrl_slacks(pt):
    """The demand columns sit before the VRL slacks, so a capped interface
    still reports its violation with a bid in the program. A 400 MW bid in
    BA-2 exceeds that area's own 250 MW of capacity, so the schedule runs
    past the 5 MW released capacity and the violation is priced."""
    eng = _eng(pt)
    r = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 5.0},
                                  shed_price=150.0, interface_vrl=3000.0,
                                  demand_bids=[{"id": "c", "bus": "5",
                                                "mw": 400.0,
                                                "price": 9000.0}])
    assert abs(r.demand_cleared["c"] - 400.0) < 1e-6
    k = ("BA-1", "BA-2")
    v = r.transfer_violations[k]
    assert v > 1.0
    assert abs(r.transfers[k] - (5.0 + v)) < 1e-6


def test_demand_bid_rejects_a_bus_outside_the_areas(pt):
    eng = _eng(pt)
    with pytest.raises(ValueError):
        se.solve_engine_transfers(pt, eng, {"BA-1": BA["BA-1"]},
                                  {}, shed_price=150.0,
                                  demand_bids=[{"bus": "9", "mw": 1.0,
                                                "price": 1.0}])


def test_curves_reject_both_inputs(pt):
    eng = _eng(pt)
    g = list(eng.gens)[0]
    curves = {gg: (float(s["cost"]), 0.05) for gg, s in eng.gens.items()}
    with pytest.raises(NotImplementedError):
        _clear(pt, curves=curves, gen_min={g: 1.0})
    with pytest.raises(NotImplementedError):
        _clear(pt, curves=curves, group_caps=[([g], 1.0)])
    with pytest.raises(NotImplementedError):
        _clear(pt, curves=curves,
               demand_bids=[{"bus": "5", "mw": 1.0, "price": 1.0}])
