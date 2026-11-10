"""The bilateral baselines build their backstop fleet on the ambient case.

``revenue_allocation.bilateral_self_solve`` and ``pre_market_bilateral`` call
``fleet_with_backstop`` when no ``gen_fleet`` is given. The name was never imported, so
both raised ``NameError`` on their default path from the initial commit until
2026-11-02. They now build the backstop through ``_fleet_with_case_backstop``, which
takes the base fleet and the load buses from the ambient case. These tests run the
default path, and check that on the 9-bus case it equals passing
``wscc9_model.fleet_with_backstop()`` explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "lib", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import footprints as fpmod  # noqa: E402
import revenue_allocation as ra  # noqa: E402
import wscc9_model as wm  # noqa: E402

BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
MANAGE = {
    "BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
    "BA-2": ["line_0", "line_1", "line_7", "line_8"],
}
COLORS = {"BA-1": "#993AFF", "BA-2": "#2471A3"}


@pytest.fixture(scope="module")
def fp():
    return fpmod.make(
        wm.CASE.build().pt, BA_DEFS, COLORS, manage=MANAGE, tie_label="tie"
    )


def test_case_backstop_equals_the_9bus_default():
    assert ra._fleet_with_case_backstop(wm.CASE.loads) == wm.fleet_with_backstop()


@pytest.mark.parametrize("area", ["BA-1", "BA-2"])
def test_bilateral_self_solve_default_fleet(fp, area):
    default = ra.bilateral_self_solve(fp, area)
    explicit = ra.bilateral_self_solve(fp, area, gen_fleet=wm.fleet_with_backstop())
    assert default == explicit


def test_backstop_only_at_positive_load_buses():
    fleet = ra._fleet_with_case_backstop({"5": 125.0, "7": 0.0, "9": 100.0})
    assert {k for k in fleet if k.startswith("backstop_")} == {
        "backstop_5",
        "backstop_9",
    }
    assert fleet["backstop_5"]["p_nom"] == 125.0


def test_backstop_refuses_a_case_with_a_dearer_unit(monkeypatch):
    dear = dict(wm.CASE.gen_fleet)
    dear["peaker"] = {"bus": "1", "cost": 120.0, "p_nom": 10.0}
    monkeypatch.setattr(
        type(wm.CASE), "gen_fleet", property(lambda self: dear), raising=False
    )
    with pytest.raises(ValueError, match="not a last resort"):
        ra._fleet_with_case_backstop(wm.CASE.loads)


def test_pre_market_bilateral_refuses_two_units_at_one_bus(fp):
    with pytest.raises(ValueError, match="one unit per bus"):
        ra.pre_market_bilateral(fp, loads=wm.LOADS_3BA)


def test_pre_market_bilateral_default_fleet(fp):
    default = ra.pre_market_bilateral(fp)
    explicit = ra.pre_market_bilateral(fp, gen_fleet=wm.fleet_with_backstop())
    assert default.keys() == explicit.keys()
    for k in default:
        assert repr(default[k]) == repr(explicit[k]), k
