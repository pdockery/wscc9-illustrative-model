# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""The intertie scheduling limit on the transfers solver.

``solve_engine_transfers(..., schedule_limits=...)`` caps the summed cleared
MW of named demand bids, the export awards at a scheduling point, with one
row that has coefficient one on the awards and no flow coefficient (the
limit of ``caiso_edam_techdesc`` section 4, with the export's -1 of
``caiso_dame_edam2023`` p. 178 read on the award as a withdrawal). The
tests run on 201's
canonical clearing (line_4 rated 40 MW, the two-BA partition, one unlimited
interface) and guard four claims.

1. Absent, ``None`` and empty limits leave every result unchanged, and the
   clearing reproduces the frozen 201 rent and the pooled clearing.
2. A binding limit caps the awards exactly, and its dual is the one the KKT
   conditions of the program give: the marginal award's bid less the bus
   price, which is also the objective's rise per MW the limit tightens.
3. The limit is the only new binding row: the binding line set and the bus
   prices are those of the clearing without bids, while the awards still
   move the line flows.
4. A limit naming a bid id not in ``demand_bids`` raises, and so do the
   other malformed inputs.

    cd wscc9-illustrative-models && PYTHONPATH=lib python -m pytest tests/test_schedule_limit.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_runner as gr  # noqa: E402
import seams_engine as se  # noqa: E402
import wscc9_model as wm  # noqa: E402

RATINGS = {"line_4": 40.0}
BA = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
SHED = 150.0  # the 9-bus shed price, $/MWh
BUS = "9"  # the scheduling point, a BA-2 load bus
BID_MW = 20.0  # each export bid's size, MW
HI_MARGIN, LO_MARGIN = 15.0, 10.0  # each bid's price above the bus price, $/MWh
LIMIT = 25.0  # the scheduling limit, MW: one bid full, one partial
TOL = 1e-6  # absolute tolerance, MW or $/MWh
BIND = 1e-6  # absolute |dual| above which a row binds, $/MWh
# relative tolerance against the golden, which is frozen at 9 significant figures
GOLDEN_REL = 1e-8


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network(RATINGS), slack_bus="1")


def _clear(pt, **kw):
    eng = wm.make_engine("U", buses=pt.buses)
    return se.solve_engine_transfers(
        pt, eng, BA, {("BA-1", "BA-2"): None}, shed_price=SHED, **kw
    )


@pytest.fixture(scope="module")
def base(pt):
    """The canonical clearing, called without the new keyword."""
    return _clear(pt)


@pytest.fixture(scope="module")
def bids(base):
    """Two export bids at one bus, priced above the canonical bus price."""
    p0 = base.lmp[BUS]
    return [
        {"id": "x_hi", "bus": BUS, "mw": BID_MW, "price": p0 + HI_MARGIN},
        {"id": "x_lo", "bus": BUS, "mw": BID_MW, "price": p0 + LO_MARGIN},
    ]


def _isl(limit):
    return {"isl": (["x_hi", "x_lo"], limit)}


def _objective(res, bids):
    """The program's objective: supply cost plus shed penalty less bid value."""
    price = {b["id"]: b["price"] for b in bids}
    return (
        res.total_cost
        + SHED * sum(res.shed_by_bus.values())
        - sum(price[j] * q for j, q in res.demand_cleared.items())
    )


def _binding(res):
    return {k for k, v in res.fg_dual.items() if abs(v) > BIND}


def _same(a, b):
    """Exact equality of every field the transfer form reports."""
    for f in (
        "dispatch",
        "lmp",
        "line_dual",
        "flow_own",
        "area_prices",
        "transfers",
        "transfer_duals",
        "demand_cleared",
        "shed_by_bus",
        "fg_dual",
        "group_cap_dual",
    ):
        assert getattr(a, f) == getattr(b, f), f
    assert a.total_cost == b.total_cost
    assert a.energy_price == b.energy_price


# ── 1. Inert when absent ───────────────────────────────────────────────────
def test_schedule_limits_absent_leave_every_result_unchanged(pt, base, bids):
    """None and {} clear the program the call without the keyword clears,
    with and without demand bids, and the new fields stay empty."""
    for kw in ({"schedule_limits": None}, {"schedule_limits": {}}):
        _same(_clear(pt, **kw), base)
        _same(_clear(pt, demand_bids=bids, **kw), _clear(pt, demand_bids=bids))
    assert base.schedule_limit_dual == {}
    assert base.schedule_limit_mw == {}


def test_canonical_clearing_reproduces_the_frozen_rent_and_the_pool(pt, base):
    """The clearing without the keyword is 201's: its rent R = Σ|μ_m F_m|
    equals the golden 201 rent, and its prices and duals equal the pooled
    clearing's, the existing path the transfer form coincides with when the
    interface is unlimited."""
    rent = sum(abs(mu) * abs(base.flow_own[m]) for m, mu in base.line_dual.items())
    pooled = se.solve_engine_dispatch(
        pt, wm.make_engine("U", buses=pt.buses), shed_price=SHED
    )
    for n in pt.buses:
        assert base.lmp[n] == pytest.approx(pooled.lmp[n], abs=TOL)
    for m in pt.lines:
        assert base.line_dual[m] == pytest.approx(pooled.line_dual[m], abs=TOL)
    if not gr.GOLDEN.exists():
        pytest.skip(f"no golden at {gr.GOLDEN}; run golden_runner.py --write")
    golden = json.loads(gr.GOLDEN.read_text(encoding="utf-8"))
    assert rent == pytest.approx(golden["line4_allocation"]["cmp[1].R"], rel=GOLDEN_REL)


def test_a_slack_limit_is_inert(pt, bids):
    """A limit above the bids' sum binds nothing: both bids clear in full,
    the dual is zero and the prices are those of the clearing without it."""
    free = _clear(pt, demand_bids=bids)
    r = _clear(pt, demand_bids=bids, schedule_limits=_isl(2 * BID_MW + 5.0))
    assert r.demand_cleared == pytest.approx(free.demand_cleared, abs=TOL)
    assert r.schedule_limit_mw["isl"] == pytest.approx(2 * BID_MW, abs=TOL)
    assert r.schedule_limit_dual["isl"] == pytest.approx(0.0, abs=TOL)
    for n in r.lmp:
        assert r.lmp[n] == pytest.approx(free.lmp[n], abs=TOL)


# ── 2. A binding limit and its dual ────────────────────────────────────────
def test_binding_limit_caps_the_awards_at_the_kkt_dual(pt, bids):
    """With the limit below the bids' sum, the summed awards equal the limit,
    the dual is positive, and the reduced-cost conditions hold: the partial
    award's bid equals the bus price plus the dual, and the full award's bid
    is at least that."""
    free = _clear(pt, demand_bids=bids)
    assert sum(free.demand_cleared.values()) > LIMIT + 1.0, "the limit must bind"
    r = _clear(pt, demand_bids=bids, schedule_limits=_isl(LIMIT))
    q = {b["id"]: r.demand_cleared.get(b["id"], 0.0) for b in bids}
    assert r.schedule_limit_mw["isl"] == pytest.approx(LIMIT, abs=TOL)
    assert sum(q.values()) == pytest.approx(LIMIT, abs=TOL)
    mu = r.schedule_limit_dual["isl"]
    assert mu > BIND
    partial = [b for b in bids if TOL < q[b["id"]] < b["mw"] - TOL]
    full = [b for b in bids if q[b["id"]] > b["mw"] - TOL]
    assert len(partial) == 1, "the case needs one marginal award"
    lam = r.lmp[BUS]
    # d_j = −p_j + λ_n + μ: zero on the partial award, nonpositive on a full one
    assert mu == pytest.approx(partial[0]["price"] - lam, abs=TOL)
    for b in full:
        assert -b["price"] + lam + mu <= TOL
    # the source's resource price at the point, λ_n + μ, is the marginal bid
    assert lam + mu == pytest.approx(partial[0]["price"], abs=TOL)


def test_the_dual_is_the_objective_rise_per_mw_of_tightening(pt, bids):
    """Tightening the limit by one MW, inside the same basis, raises the
    objective by the dual: the sign and size of μ^ISL read from the program
    itself rather than from a solver convention."""
    r = _clear(pt, demand_bids=bids, schedule_limits=_isl(LIMIT))
    r1 = _clear(pt, demand_bids=bids, schedule_limits=_isl(LIMIT - 1.0))
    rise = _objective(r1, bids) - _objective(r, bids)
    assert rise == pytest.approx(r.schedule_limit_dual["isl"], abs=TOL)


# ── 3. The only new binding row; the awards move the flows ─────────────────
def test_the_limit_is_the_only_new_binding_row_and_moves_the_flows(pt, base, bids):
    """At two binding limits, the binding line set and every bus price are
    those of the clearing without bids, because the row has no flow
    coefficient and no new line binds, while the line flows move with the
    awards through the withdrawal at the bus."""
    r = _clear(pt, demand_bids=bids, schedule_limits=_isl(LIMIT))
    r2 = _clear(pt, demand_bids=bids, schedule_limits=_isl(LIMIT - 3.0))
    for res in (r, r2):
        assert res.schedule_limit_dual["isl"] > BIND
        assert _binding(res) == _binding(base)
        for n in pt.buses:
            assert res.lmp[n] == pytest.approx(base.lmp[n], abs=TOL)
    moved = max(abs(r.flow_own[m] - r2.flow_own[m]) for m in pt.lines)
    assert moved > 0.1


# ── 4. Malformed inputs ────────────────────────────────────────────────────
def test_a_bid_id_not_in_demand_bids_raises(pt, bids):
    with pytest.raises(KeyError):
        _clear(
            pt,
            demand_bids=bids,
            schedule_limits={"isl": (["x_hi", "not_a_bid"], LIMIT)},
        )
    with pytest.raises(KeyError):
        _clear(pt, schedule_limits={"isl": (["x_hi"], LIMIT)})


def test_malformed_limits_raise(pt, bids):
    with pytest.raises(ValueError):
        _clear(pt, demand_bids=bids, schedule_limits={"isl": ([], LIMIT)})
    with pytest.raises(ValueError):
        _clear(pt, demand_bids=bids, schedule_limits=_isl(-1.0))
    with pytest.raises(ValueError):
        _clear(pt, demand_bids=bids, schedule_limits=_isl(float("nan")))
    with pytest.raises(TypeError):
        _clear(pt, demand_bids=bids, schedule_limits={"isl": ("x_hi", LIMIT)})


def test_a_zero_size_bid_counts_as_zero(pt, bids):
    """A bid that demand_bids carries at zero MW has no column, so a limit
    may name it and counts it as zero."""
    zero = {"id": "x_zero", "bus": BUS, "mw": 0.0, "price": bids[0]["price"]}
    r = _clear(
        pt, demand_bids=[*bids, zero], schedule_limits={"isl": (["x_zero"], LIMIT)}
    )
    assert r.schedule_limit_mw["isl"] == 0.0
    assert r.schedule_limit_dual["isl"] == pytest.approx(0.0, abs=TOL)


def test_curves_reject_schedule_limits(pt, bids):
    eng = wm.make_engine("U", buses=pt.buses)
    curves = {g: (float(s["cost"]), 0.05) for g, s in eng.gens.items()}
    with pytest.raises(NotImplementedError):
        _clear(pt, curves=curves, schedule_limits=_isl(LIMIT))
