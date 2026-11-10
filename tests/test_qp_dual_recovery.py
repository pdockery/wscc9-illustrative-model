"""``solve_engine_qp`` must recover a binding line's dual at every volume.

``trust-constr`` satisfies a binding limit only to its own convergence
tolerance, and was measured stopping up to 2e-4 MW inside one. The dual
recovery's active set is what carries the congestion columns of the KKT
system, so a line dropped from it leaves only the balance column and every
price collapses to the least-squares average of the interior units' marginal
costs. That produced silently wrong prices at scattered volumes: the LMP
identity ``λ_n = λ + Σ_m SF_{n,m} μ_m`` still held exactly, because both sides
used the same missing dual, so nothing downstream caught it.

The regression walks a genuinely binding constraint across a volume sweep and
checks the recovered dual against the shadow price measured independently, as
the derivative of dispatch cost with respect to the line's rating.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seams_engine as se  # noqa: E402
import wscc9_model as wm   # noqa: E402

MKT_A = ["2", "8", "7", "6", "3"]
ACT_A = ["line_2", "line_3", "line_4", "line_5", "line_6"]
CURVES = {"gen_slack_0": (45.0, 0.0233), "gen_0": (30.0, 0.0284), "gen_1": (15.0, 0.0360)}
RATING = 40.0
#: volumes that include the ones the narrow active-set band used to miss
SWEEP = [0.0, 5.0, 10.0, 15.0, 18.0, 19.0, 20.0, 21.0, 25.0, 30.0,
         115.0, 118.0, 120.0, 122.0, 125.0]


def _fleet():
    gf = {g: dict(s) for g, s in wm.DEFAULT_GEN_FLEET.items()}
    gf["gen_slack_0"]["p_nom"] = 300.0
    return gf


def _clear(export_mw, rating=RATING):
    pt = se.compute_ptdf(wm.build_network({"line_4": rating, "line_0": 300.0}),
                         slack_bus="1")
    eng = wm.make_engine("Market A", MKT_A, gen_fleet=_fleet(), activated=ACT_A)
    eng.loads["7"] = 90.0
    return se.solve_engine_qp(pt, eng, curves=CURVES,
                              exo={"8": -float(export_mw)}, shed_price=150.0)


@pytest.mark.parametrize("export_mw", SWEEP)
def test_binding_line_keeps_its_dual(export_mw):
    """The line sits at its rating at every volume, so its dual is never zero."""
    res = _clear(export_mw)
    flow = float(res.flow_own["line_4"])
    assert abs(flow) == pytest.approx(RATING, abs=1e-3), (
        f"line_4 is not at its limit at {export_mw} MW: flow {flow}")
    assert abs(float(res.line_dual["line_4"])) > 1.0, (
        f"binding line lost its dual at {export_mw} MW "
        f"(flow {flow:.9f} of {RATING})")


@pytest.mark.parametrize("export_mw", [0.0, 5.0, 19.0, 20.0, 30.0, 120.0])
def test_dual_matches_the_measured_shadow_price(export_mw):
    """μ equals −d(cost)/d(rating), measured by a central difference."""
    h = 0.05
    mu = float(_clear(export_mw).line_dual["line_4"])
    up = _clear(export_mw, RATING + h).total_cost
    dn = _clear(export_mw, RATING - h).total_cost
    measured = -(up - dn) / (2.0 * h)      # cost falls as the rating rises
    assert abs(mu) == pytest.approx(measured, rel=5e-3), (
        f"at {export_mw} MW the recovered dual {mu:.4f} does not match the "
        f"measured shadow price {measured:.4f}")


def test_price_is_monotone_in_the_export():
    """Market A's price at the export bus rises smoothly with the export.

    A dropped dual shows up here as a downward spike, because the price falls
    back to the flat average of the marginal costs.
    """
    lam = [float(_clear(E).lmp["8"]) for E in SWEEP]
    diffs = np.diff(lam)
    assert (diffs > 0).all(), (
        "price is not monotone in the export volume: "
        + ", ".join(f"{E:.0f}MW=${p:.3f}" for E, p in zip(SWEEP, lam)))


def test_flat_curves_still_match_the_lp():
    """With b = 0 the QP must reproduce the flat LP, duals included."""
    pt = se.compute_ptdf(wm.build_network({"line_4": RATING, "line_0": 300.0}),
                         slack_bus="1")
    eng = wm.make_engine("Market A", MKT_A, gen_fleet=_fleet(), activated=ACT_A)
    eng.loads["7"] = 90.0
    flat = {g: (float(eng.gens[g]["cost"]), 0.0) for g in eng.gens}
    qp = se.solve_engine_qp(pt, eng, curves=flat, shed_price=150.0)
    lp = se.solve_engine_dispatch(pt, eng, shed_price=150.0)
    for b in pt.buses:
        assert float(qp.lmp[b]) == pytest.approx(float(lp.lmp[b]), abs=1e-4)
    for l in pt.lines:
        assert float(qp.line_dual[l]) == pytest.approx(float(lp.line_dual[l]), abs=1e-4)
