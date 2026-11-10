"""The 9-bus teaching case against the Case contract.

``seams_engine.CaseProtocol`` states the surface every case exposes to the
shared libraries, ``seams_engine.check_case`` verifies an instance against
it, and this file runs that check over ``wscc9_model.CASE``, the reference
implementation.

    pytest tests/test_case_contract.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "lib", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seams_engine as se           # noqa: E402
import wscc9_model as wm            # noqa: E402


@pytest.fixture(scope="module")
def case():
    return wm.CASE


def test_check_case_passes(case):
    assert se.check_case(case) == []


def test_isinstance_case_protocol(case):
    assert isinstance(case, se.CaseProtocol)
    # the 9-bus case reads no bundle, so it is not a data-backed case
    assert not isinstance(case, se.DataBackedCase)


def test_fleet_schema(case):
    fleet = case.gen_fleet
    assert fleet, "the fleet is empty"
    known = set(se.FLEET_REQUIRED) | set(se.FLEET_OPTIONAL)
    for gid, spec in fleet.items():
        assert set(se.FLEET_REQUIRED) <= set(spec), (gid, spec)
        assert set(spec) <= known, (gid, sorted(set(spec) - known))
    assert se.check_fleet(fleet) == []
    # the backstop variant of the fleet carries the same schema
    assert se.check_fleet(wm.fleet_with_backstop()) == []
    assert se.check_fleet(wm.FLEET_3BA) == []


def test_fleet_schema_names_violations():
    bad = {"u": {"bus": "1", "cost": 10.0}}
    assert any("lacks required" in v for v in se.check_fleet(bad))
    bad = {"u": {"bus": "1", "cost": 10.0, "p_nom": 5.0, "colour": "red"}}
    assert any("unknown key" in v for v in se.check_fleet(bad))
    # the gas-basis marker is the (day, basis) pair reprice_gas_fleet writes
    ok = {"u": {"bus": "1", "cost": 10.0, "p_nom": 5.0,
                "gas_basis": ("2024-07-12", "daily")}}
    assert se.check_fleet(ok) == []
    bad = {"u": {"bus": "1", "cost": 10.0, "p_nom": 5.0, "gas_basis": "daily"}}
    assert any("gas_basis" in v for v in se.check_fleet(bad))


def test_build_returns_line_rated_network(case):
    cn = case.build()
    assert isinstance(cn, se.CaseNetwork)
    assert cn.fg is None
    assert cn.constraint_set == se.CONSTRAINT_LINES == "lines"
    assert cn.pt.slack_bus == case.slack_bus
    assert case.slack_bus in cn.pt.buses
    # a fresh network each call, the PTDF cached
    cn2 = case.build()
    assert cn2.network is not cn.network
    assert cn2.pt is cn.pt


def test_build_rejects_unknown_variant(case):
    with pytest.raises(TypeError):
        case.build(not_a_variant=True)


def test_make_engine_clears(case):
    cn = case.build()
    eng = case.make_engine("UNIFIED", cn.pt.buses)
    assert isinstance(eng, se.MarketEngine)
    assert set(eng.gens) == set(case.gen_fleet)
    assert set(eng.loads) == set(case.loads)
    res = se.solve_engine_dispatch(cn.pt, eng, shed_price=case.shed_price)
    assert "Optimal" in res.status
    served = sum(case.loads.values())
    assert abs(sum(res.dispatch.values()) - served) < 1e-6


def test_hub_kind_and_drivers(case):
    assert case.hub_kind() in se.HUB_KINDS
    d = case.drivers()
    assert isinstance(d, dict) and d


def test_constraint_set_property_reads_fg():
    cn = wm.CASE.build()
    assert se.CaseNetwork(network=cn.network, pt=cn.pt, fg=None).constraint_set == "lines"
    fg = se.identity_flowgates(cn.pt)
    assert se.CaseNetwork(network=cn.network, pt=cn.pt, fg=fg).constraint_set == "flowgates"


def test_empty_flowgate_set_follows_the_solver_branch():
    """An empty ``FlowgateSet`` is classified the way the solvers treat it:
    the flowgate branch, which activates no constraint. The property and
    the clearing agree, and ``check_case`` rejects a case that builds it."""
    base = wm.CASE.build()
    cn = wm.CASE.build({m: 60.0 for m in base.pt.lines})
    empty = se.identity_flowgates(cn.pt).subset([])
    assert se.CaseNetwork(network=cn.network, pt=cn.pt, fg=empty).constraint_set == "flowgates"
    eng = wm.make_engine("UNIFIED", buses=cn.pt.buses)
    lines = se.solve_engine_dispatch(cn.pt, eng, shed_price=wm.CASE.shed_price)
    nothing = se.solve_engine_dispatch(cn.pt, eng, shed_price=wm.CASE.shed_price,
                                       flowgates=empty)
    # the per-line clearing holds every flow at its 60 MW rating; the empty
    # flowgate set holds none of them and prices no constraint
    assert max(abs(f) for f in lines.flow_own.values()) <= 60.0 + 1e-6
    assert max(abs(f) for f in nothing.flow_own.values()) > 60.0
    assert nothing.fg_dual == {} and not any(abs(v) > 1e-9 for v in nothing.line_dual.values())

    class RatesNothing(wm.WSCC9Case):
        def build(self, line_ratings=None, **variant):
            cn = super().build(line_ratings, **variant)
            return se.CaseNetwork(network=cn.network, pt=cn.pt,
                                  fg=se.identity_flowgates(cn.pt).subset([]))

    bad = RatesNothing()
    out = se.check_case(bad)
    assert any("no flowgate" in v for v in out), out


def test_hub_weights_is_part_of_the_contract(case):
    """``hub_weights`` (plan item B3, 2026-10-21) is a Protocol member, and
    ``check_case`` reports a case without it. On the 9-bus case the method
    needs the Footprints partition, because ``bus_to_area`` is empty."""
    import footprints as fpmod
    import revenue_allocation as ra

    assert callable(case.hub_weights)
    pt = case.build().pt
    fp = fpmod.make(pt, {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]},
                    {"BA-1": "#993AFF", "BA-2": "#2471A3"})
    assert case.hub_weights(fp=fp) == ra.hub_weights(fp, case.hub_kind())
    with pytest.raises(ValueError):
        case.hub_weights()

    class NoHubWeights:
        def __getattr__(self, name):
            if name == "hub_weights":
                raise AttributeError(name)
            return getattr(case, name)

    assert any("hub_weights" in v for v in se.check_case(NoHubWeights(), build=False))
