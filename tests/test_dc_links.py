"""DC links on the shared solvers: one signed, costless, capacity-bounded
physical transfer variable per link (the PDCI/IPP primitive for the WECC
case), byte-identical to the old path when absent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seams_engine as se      # noqa: E402
import wscc9_model as wm       # noqa: E402

RATINGS = {"line_4": 40.0}


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network(RATINGS), slack_bus="1")


def _eng(pt):
    return wm.make_engine("U", buses=pt.buses)


def test_absent_links_change_nothing(pt):
    base = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0)
    with_none = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0,
                                         dc_links=None)
    with_empty = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0,
                                          dc_links={})
    for r in (with_none, with_empty):
        assert r.dispatch == base.dispatch
        assert r.lmp == base.lmp
        assert r.link_flow == {}


def test_link_arbitrages_and_saturates(pt):
    """A DC link from the cheap 3-side to the constrained 7-side bypasses the
    binding AC path: it runs, lowers cost, and when the price gap survives it
    saturates with rent (lmp[b1]-lmp[b0])*t."""
    base = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0)
    gap0 = base.lmp["7"] - base.lmp["3"]
    assert gap0 > 1.0                      # line_4 at 40 MW separates 3 from 7
    link = {"dc1": {"bus0": "3", "bus1": "7", "p_max": 15.0}}
    r = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0, dc_links=link)
    t = r.link_flow["dc1"]
    assert t > 1.0
    assert r.total_cost < base.total_cost - 1e-6
    if r.lmp["7"] - r.lmp["3"] > 0.01:     # gap survives -> link at its cap
        assert abs(t - 15.0) < 1e-6
    # a link that would move power the WRONG way stays at zero
    bad = {"dcb": {"bus0": "7", "bus1": "3", "p_max": 15.0}}
    rb = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0, dc_links=bad)
    assert abs(rb.link_flow["dcb"]) < 1e-9
    # ... and with reverse capability it runs backwards to the same optimum
    bad_rev = {"dcb": {"bus0": "7", "bus1": "3", "p_max": 15.0, "p_min": -15.0}}
    rr = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0,
                                  dc_links=bad_rev)
    assert abs(rr.link_flow["dcb"] + t) < 1e-6
    assert abs(rr.total_cost - r.total_cost) < 1e-6


def test_link_flow_enters_physical_injection(pt):
    link = {"dc1": {"bus0": "3", "bus1": "7", "p_max": 15.0}}
    r = se.solve_engine_dispatch(pt, _eng(pt), shed_price=150.0, dc_links=link)
    # the AC network sees a withdrawal at 3 and an injection at 7
    inj = r.injection
    j3, j7 = pt.bus_idx["3"], pt.bus_idx["7"]
    manual = np.zeros(pt.n_bus)
    for b, g in r.gen_by_bus.items():
        manual[pt.bus_idx[b]] += g
    for b, d in r.load_by_bus.items():
        manual[pt.bus_idx[b]] -= d
    for b, u in r.shed_by_bus.items():
        manual[pt.bus_idx[b]] += u
    manual[j3] -= r.link_flow["dc1"]
    manual[j7] += r.link_flow["dc1"]
    assert np.allclose(inj, manual, atol=1e-8)
    assert abs(inj.sum()) < 1e-8


def test_links_in_transfers_form(pt):
    """Transfers form: the link is physical (moves flows AND area balances);
    with it, the two areas' interfaces schedule less."""
    BA = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    eng = _eng(pt)
    base = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None},
                                     shed_price=150.0)
    link = {"dc1": {"bus0": "3", "bus1": "9", "p_max": 20.0}}
    r = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None},
                                  shed_price=150.0, dc_links=link)
    assert "optimal" in r.status
    assert r.total_cost <= base.total_cost + 1e-6
    t = r.link_flow["dc1"]
    k = ("BA-1", "BA-2")
    # the tags + the physical link together carry each area's net position:
    # net export of BA-1 = schedule + link flow (3 is in BA-1, 9 in BA-2)
    gen1 = sum(r.gen_by_bus.get(b, 0.0) for b in BA["BA-1"])
    load1 = sum(r.load_by_bus.get(b, 0.0) for b in BA["BA-1"])
    shed1 = sum(r.shed_by_bus.get(b, 0.0) for b in BA["BA-1"])
    assert abs((gen1 + shed1 - load1) - (r.transfers[k] + t)) < 1e-4
    # QP curves + links is explicitly unsupported
    with pytest.raises(NotImplementedError):
        se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None},
                                  curves={"gen_0": (35.0, 0.1)},
                                  shed_price=150.0, dc_links=link)


def test_transfers_absent_links_unchanged(pt):
    BA = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    eng = _eng(pt)
    a = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 30.0},
                                  shed_price=150.0)
    b = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 30.0},
                                  shed_price=150.0, dc_links={})
    assert a.dispatch == b.dispatch
    assert a.transfers == b.transfers
    assert a.area_prices == b.area_prices
