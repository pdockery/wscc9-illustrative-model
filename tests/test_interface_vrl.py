"""interface_vrl: the SPP/Markets+-style Violation Relaxation Limit on the
released-capacity constraints of solve_engine_transfers."""
import numpy as np
import pytest

import footprints as fpmod  # noqa: F401
import seams_engine as se
import wscc9_model as wm

BA = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}


@pytest.fixture(scope="module")
def pt():
    return wm.shift_factors()


@pytest.fixture(scope="module")
def eng():
    return wm.make_engine("uni", [b for bs in BA.values() for b in bs])


def test_vrl_inactive_when_scarcity_below_it(pt, eng):
    """With the VRL above the interface's scarcity value, the clearing is
    identical to the hard-capped one and no violation is awarded."""
    hard = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 30.0})
    soft = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 30.0},
                                     interface_vrl=3000.0)
    assert soft.total_cost == pytest.approx(hard.total_cost, abs=1e-6)
    assert soft.transfer_violations == {}
    assert soft.transfer_duals[("BA-1", "BA-2")] == pytest.approx(
        hard.transfer_duals[("BA-1", "BA-2")], abs=1e-6)


def test_vrl_caps_the_shadow_price_and_reports_the_violation(pt, eng):
    """With the VRL below the scarcity value the market violates at the
    penalty: mu_T == VRL exactly, the violation MW is reported, and the
    dispatch reaches the unlimited optimum (penalty not in production cost)."""
    free = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None})
    soft = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 30.0},
                                     interface_vrl=10.0)
    assert soft.transfer_duals[("BA-1", "BA-2")] == pytest.approx(10.0, abs=1e-6)
    v = soft.transfer_violations[("BA-1", "BA-2")]
    assert v == pytest.approx(free.transfers[("BA-1", "BA-2")] - 30.0, abs=1e-4)
    assert soft.total_cost == pytest.approx(free.total_cost, abs=1e-6)
    assert soft.transfers[("BA-1", "BA-2")] == pytest.approx(
        free.transfers[("BA-1", "BA-2")], abs=1e-4)


def test_vrl_cannot_create_an_absent_interface(pt, eng):
    """A pair absent from `interfaces` has no variable: the VRL cannot buy a
    schedule on a relationship that does not exist."""
    r = se.solve_engine_transfers(pt, eng, BA, {}, interface_vrl=10.0)
    assert r.transfers == {} and r.transfer_violations == {}


def test_fixed_transfers_pin_and_guard(pt, eng):
    """A self-scheduled tag is honored exactly, the exporter supports it, and
    the guards reject an undeclared pair or one beyond its capacity."""
    free = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None})
    r = se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): None},
                                  fixed_transfers={("BA-1", "BA-2"): 100.0})
    assert r.transfers[("BA-1", "BA-2")] == pytest.approx(100.0, abs=1e-6)
    assert r.total_cost > free.total_cost  # pinning below the optimum costs
    with pytest.raises(ValueError, match="exceeds its released capacity"):
        se.solve_engine_transfers(pt, eng, BA, {("BA-1", "BA-2"): 150.0},
                                  fixed_transfers={("BA-1", "BA-2"): 200.0})
    with pytest.raises(ValueError, match="not a declared interface"):
        se.solve_engine_transfers(pt, eng, BA, {},
                                  fixed_transfers={("BA-1", "BA-2"): 50.0})
