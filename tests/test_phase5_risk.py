"""Phase-5 surfaces: factor driver kinds (lf/li/fuel/ca/cf), LogitNormal,
DriverModel/ba_drivers, day_bootstrap, and the propagate pass-throughs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import risk                    # noqa: E402
import seams_engine as se      # noqa: E402
import wscc9_model as wm       # noqa: E402

B2A = {"2": "BA-1", "8": "BA-1", "7": "BA-1", "6": "BA-1", "3": "BA-1",
       "1": "BA-2", "9": "BA-2", "4": "BA-2", "5": "BA-2"}


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")


def _scen1(values: dict) -> risk.Scenarios:
    """One deterministic scenario holding exactly ``values`` {key: value}."""
    return risk.Scenarios(draws={k: np.array([float(v)]) for k, v in values.items()},
                          weight=np.array([1.0]), keys=list(values))


# ── factor expansion ─────────────────────────────────────────────────────
def test_lf_scales_every_load_in_the_ba(pt):
    scen = _scen1({("lf", "BA-1"): 0.10})
    vals, _ = risk.propagate(pt, scen, lambda r: r.total_cost, bus_to_area=B2A)
    manual = risk.clear(pt, load={b: mw * 1.10 for b, mw in wm.CASE.loads.items()
                                  if B2A[b] == "BA-1"})
    assert abs(vals[0] - manual.total_cost) < 1e-9


def test_lf_plus_li_shocks_add_within_a_bus(pt):
    scen = _scen1({("lf", "BA-1"): 0.10, ("li", "7"): 0.05})
    vals, _ = risk.propagate(pt, scen, lambda r: r.total_cost, bus_to_area=B2A)
    ld = {b: mw * (1.10 if B2A[b] == "BA-1" else 1.0) for b, mw in wm.CASE.loads.items()}
    ld["7"] = wm.CASE.loads["7"] * 1.15                    # 1 + 0.10 + 0.05, not 1.10*1.05
    manual = risk.clear(pt, load=ld)
    assert abs(vals[0] - manual.total_cost) < 1e-9


def _tagged_fleet():
    fleet = wm.CASE.gen_fleet
    for g in fleet:
        fleet[g]["carrier"] = "gas" if g == "gen_slack_0" else "hydro"
    return fleet


def test_fuel_factor_scales_carrier_costs(pt):
    fleet = _tagged_fleet()
    scen = _scen1({("fuel", "gas"): 0.50})
    vals, _ = risk.propagate(pt, scen, lambda r: r.total_cost, fleet=fleet)
    manual = risk.clear(pt, cost={"gen_slack_0": fleet["gen_slack_0"]["cost"] * 1.50},
                        fleet=fleet)
    assert abs(vals[0] - manual.total_cost) < 1e-9


def test_ca_factor_scales_ba_fleet_costs(pt):
    scen = _scen1({("ca", "BA-2"): 0.20})                  # bus-1 slack unit only
    vals, _ = risk.propagate(pt, scen, lambda r: r.total_cost, bus_to_area=B2A)
    manual = risk.clear(pt, cost={"gen_slack_0": wm.CASE.gen_fleet["gen_slack_0"]["cost"] * 1.20})
    assert abs(vals[0] - manual.total_cost) < 1e-9


def test_cf_derates_capacity_and_moves_the_dispatch(pt):
    fleet = _tagged_fleet()
    scen = _scen1({("cf", None, "hydro"): 0.30})
    vals, _ = risk.propagate(pt, scen, lambda r: r.dispatch["gen_1"], fleet=fleet)
    manual = risk.clear(pt, pmax={g: fleet[g]["p_nom"] * 0.30
                                  for g in ("gen_0", "gen_1")}, fleet=fleet)
    assert abs(vals[0] - manual.dispatch["gen_1"]) < 1e-9
    assert vals[0] <= fleet["gen_1"]["p_nom"] * 0.30 + 1e-9
    base = risk.clear(pt, fleet=fleet)
    assert manual.total_cost > base.total_cost + 1.0       # the derate binds


def test_spec_ba_tag_beats_bus_area(pt):
    # a generator tagged with its OWN BA resolves there even when its bus
    # sits in another BA's footprint (the real-West pattern)
    fleet = wm.CASE.gen_fleet
    fleet["gen_1"]["ba"] = "BA-2"                          # bus 3 is in BA-1
    scen = _scen1({("ca", "BA-2"): 0.20})
    vals, _ = risk.propagate(pt, scen, lambda r: r.total_cost, fleet=fleet,
                             bus_to_area=B2A)
    manual = risk.clear(pt, cost={g: fleet[g]["cost"] * 1.20
                                  for g in ("gen_slack_0", "gen_1")}, fleet=fleet)
    assert abs(vals[0] - manual.total_cost) < 1e-9


def test_unknown_kind_and_empty_factor_raise(pt):
    with pytest.raises(ValueError, match="unknown driver kind"):
        risk.propagate(pt, _scen1({("windgust", "x"): 0.1}), lambda r: 0.0)
    with pytest.raises(ValueError, match="matches no bus/generator"):
        risk.propagate(pt, _scen1({("lf", "BA-9"): 0.1}), lambda r: 0.0, bus_to_area=B2A)
    with pytest.raises(ValueError, match="matches no bus/generator"):
        risk.propagate(pt, _scen1({("fuel", "coal"): 0.1}), lambda r: 0.0)  # untagged fleet


# ── LogitNormal ──────────────────────────────────────────────────────────
def test_logitnormal_draws_live_in_unit_interval():
    d = {("cf", None, None): risk.LogitNormal(0.9, 0.8)}
    scen = risk.monte_carlo(d, n=500, rng=np.random.default_rng(3))
    x = scen.draws[("cf", None, None)]
    assert (x > 0).all() and (x < 1).all()
    # degenerate sd=0 pins the mean; GH at mean .5 stays .5 by symmetry
    assert (risk.monte_carlo({("cf", None, None): risk.LogitNormal(0.7, 0.0)}, n=5,
                             rng=np.random.default_rng(0)).draws[("cf", None, None)] == 0.7).all()
    gh = risk.gauss_hermite({("cf", None, None): risk.LogitNormal(0.5, 1.0)}, nodes=9)
    assert abs(risk.emean(gh.draws[("cf", None, None)], gh.weight) - 0.5) < 1e-9


# ── DriverModel / ba_drivers ─────────────────────────────────────────────
def test_ba_drivers_builds_the_expected_structure():
    dm = risk.ba_drivers(bus_to_area=B2A, load_sd={"BA-1": 0.08, "BA-2": 0.12},
                         load_idio_sd=0.03, fuel_sd={"gas": 0.15},
                         cf={("BA-1", "hydro"): (0.9, 0.5)})
    keys = set(dm.dists)
    assert ("lf", "BA-1") in keys and ("lf", "BA-2") in keys
    assert all(("li", b) in keys for b in wm.CASE.loads)
    assert ("fuel", "gas") in keys and ("cf", "BA-1", "hydro") in keys
    assert isinstance(dm.dists[("cf", "BA-1", "hydro")], risk.LogitNormal)
    assert len(dm.drivers) == len(dm.dists)
    scen = dm.sample(40, rng=np.random.default_rng(0))
    assert scen.S == 40 and set(scen.keys) == keys


def test_ba_drivers_without_area_map_raises():
    with pytest.raises(ValueError, match="bus_to_area"):
        risk.ba_drivers(load_sd={"BA-1": 0.1})             # 9-bus case has bus_to_area = {}


# ── day_bootstrap ────────────────────────────────────────────────────────
def _days():
    hrs = np.arange(24.0)
    return {"d0": {("load", "7"): 100 + hrs, ("fuel", "gas"): 0.1},
            "d1": {("load", "7"): 200 + hrs, ("fuel", "gas"): -0.1}}


def test_day_bootstrap_draws_are_consistent_within_a_scenario():
    scen = risk.day_bootstrap(_days(), n=100, rng=np.random.default_rng(7))
    assert scen.S == 100 and set(scen.keys) == set(_days()["d0"])
    for s in range(scen.S):
        day, hr = scen.meta["day"][s], int(scen.meta["hour"][s])
        base = 100.0 if day == "d0" else 200.0
        assert scen.draws[("load", "7")][s] == base + hr           # joint (day, hour) draw
        assert scen.draws[("fuel", "gas")][s] == (0.1 if day == "d0" else -0.1)
    assert set(scen.meta["day"]) == {"d0", "d1"}


def test_day_bootstrap_hours_and_overlay_and_errors():
    scen = risk.day_bootstrap(_days(), n=50, rng=np.random.default_rng(1), hours=5,
                              overlay={("lf", "BA-1"): risk.Normal(0.0, 0.05)})
    assert (scen.meta["hour"] == 5).all()
    assert ("lf", "BA-1") in scen.keys and scen.draws[("lf", "BA-1")].std() > 0
    pool = risk.day_bootstrap(_days(), n=200, rng=np.random.default_rng(2),
                              hours=range(16, 21))
    assert set(pool.meta["hour"]) <= set(range(16, 21))
    with pytest.raises(ValueError, match="already drawn"):
        risk.day_bootstrap(_days(), n=5, rng=np.random.default_rng(0),
                           overlay={("load", "7"): risk.Normal(0.0, 1.0)})
    bad = _days()
    bad["d1"] = {("load", "9"): np.arange(24.0)}
    with pytest.raises(ValueError, match="different driver keys"):
        risk.day_bootstrap(bad, n=5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one"):
        risk.day_bootstrap({}, n=5)


def test_day_bootstrap_scenarios_propagate(pt):
    scen = risk.day_bootstrap(_days(), n=8, rng=np.random.default_rng(4),
                              overlay={("lf", "BA-1"): risk.Normal(0.0, 0.05)})
    vals, w = risk.propagate(pt, scen, risk.path_congestion("3", "7"),
                             fleet=_tagged_fleet(), bus_to_area=B2A)
    assert len(vals) == 8 and abs(w.sum() - 1.0) < 1e-12


# ── pass-throughs and the case surface ───────────────────────────────────
def test_propagate_with_identity_flowgates_matches_lines(pt):
    dists = {k: risk.load_dists()[k] for k in list(risk.load_dists())[:2]}
    scen = risk.gauss_hermite(dists, nodes=3)
    v0, _ = risk.propagate(pt, scen, risk.path_congestion("3", "9"))
    v1, _ = risk.propagate(pt, scen, risk.path_congestion("3", "9"),
                           flowgates=se.identity_flowgates(pt))
    assert np.allclose(v0, v1, atol=1e-7)
    l0, _, _ = risk.induced_lmps(pt, scen, buses=["5", "9"])
    l1, _, _ = risk.induced_lmps(pt, scen, buses=["5", "9"],
                                 flowgates=se.identity_flowgates(pt))
    assert np.allclose(l0, l1, atol=1e-7)


def test_case_drivers_surface():
    d = wm.CASE.drivers()
    assert d.keys() == {**risk.load_dists(), **risk.cost_dists()}.keys()
    assert set(k[0] for k in wm.CASE.drivers(kinds=("load",))) == {"load"}
