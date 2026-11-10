# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""Slide checks for ``workshop2_congestion_management.ipynb``, the August 18 deck.

The notebook clears the deck's two congestion management examples on the 9-bus network.
These tests run its code cells once, through :func:`slide_check.run_notebook`, and compare
its clears with the numbers slides 3 to 17 print, through :func:`slide_check.check_slide`,
each row at half a unit of the slide's last printed digit. The 228 rows are those the
notebook's cells asserted before the checks moved here on 2026-11-08. The tests also
assert the anchor of Market A's first clear, the notebook's cost-of-an-input identity for
each input Market A carries, the closing of its two cost ledgers, and the slides'
statements that a clear leaves a quantity unchanged.

A model value is read from the notebook's namespace or computed from its objects. A typed
number is a slide's printed value, an input or the series anchor. A number the notebook's
text states is read from its markdown cell by :func:`text_value`, never typed, so a test
guards the text against the run.

    python -m pytest tests/test_slides_workshop2.py -q
    python -m pytest tests -q -m slides
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import nbformat
import pytest

from revenue_allocation import capacity_value_by_line, settlement_by_bus
from slide_check import (
    ROOT,
    Row,
    as_row,
    check_slide,
    flow_rows,
    injection_rows,
    price_rows,
    run_notebook,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from seams_engine import EngineResult

pytestmark = pytest.mark.slides

NOTEBOOK = "workshop2_congestion_management.ipynb"
#: A number in the notebook's text: one decimal numeral, as the text prints it.
NUMERAL = r"(\d+(?:\.\d+)?)"

#: Absolute tolerance of an identity the formulation states exactly, $/h or MW, at the
#: solver's precision.
IDENTITY_ABS = 1e-6
#: Absolute tolerance of two computations of one quantity from the same numbers.
EXACT_ABS = 1e-9
#: The series anchor, Market A's first clear: line 6-7 at its 40 MW rating with
#: |mu| = $20.10/MWh and R = $804.1/h, each to half a unit of its last digit.
ANCHOR_FLOW, ANCHOR_FLOW_TOL = 40.0, 0.05
ANCHOR_MU, ANCHOR_MU_TOL = 20.10, 0.005
ANCHOR_RENT, ANCHOR_RENT_TOL = 804.1, 0.05
#: Half a unit of a slide's last printed digit, absolute, in the row's unit: a whole
#: number, a tenth and a hundredth.
HALF_UNIT, HALF_TENTH, HALF_HUNDREDTH = 0.5, 0.05, 0.005
#: Slide 14's printed redispatch of each unit, MW, which the run does not reproduce.
SLIDE_14_REDISPATCH, SLIDE_14_REDISPATCH_TOL = 15.45, 0.005


@pytest.fixture(scope="module")
def ns() -> dict[str, Any]:
    """The namespace of one run of the notebook's code cells."""
    return run_notebook(NOTEBOOK)


def text_value(heading: str, pattern: str) -> tuple[float, float]:
    """A number the notebook's text states, with half a unit of its last printed digit.

    ``heading`` opens the one markdown cell that states it, and ``pattern`` holds
    :data:`NUMERAL` once, as the number's group. The tolerance is absolute.
    """
    nb = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    cells = [
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and c.source.startswith(heading)
    ]
    assert len(cells) == 1, f"{len(cells)} markdown cells open with {heading!r}"
    found = re.search(pattern, cells[0])
    assert found, f"{pattern!r} is not in the cell headed {heading!r}"
    numeral = found.group(1)
    places = len(numeral.partition(".")[2])
    return float(numeral), 0.5 * 10.0**-places


def rows_of(rows: Sequence[Sequence[Any]]) -> list[Row]:
    """A slide's own rows, each with its compared value, ready to join the harness rows."""
    return [as_row(r) for r in rows]


def net_injection(ns: dict[str, Any], *results: EngineResult) -> dict[str, float]:
    """The summed net injection at each bus, {bus: MW}, as the figures' node labels draw it."""
    pt = ns["pt"]
    return {
        b: float(sum(r.injection[pt.bus_idx[b]] for r in results)) for b in pt.buses
    }


def superposed(ns: dict[str, Any], *results: EngineResult) -> dict[str, float]:
    """The superposed flow, the markets' flow contributions summed, {line: MW}, unrounded."""
    return {m: float(sum(r.flow_own[m] for r in results)) for m in ns["pt"].lines}


def shift_factor(ns: dict[str, Any], bus: str) -> float:
    """The shift factor of ``bus`` on the study line; the array is indexed line first."""
    pt = ns["pt"]
    return float(pt.ptdf[pt.line_idx[ns["STUDY_LINE"]], pt.bus_idx[bus]])


def abs_mu(ns: dict[str, Any], res: EngineResult) -> float:
    """The absolute dual of the study line in ``res``, $/MWh."""
    return abs(float(res.line_dual[ns["STUDY_LINE"]]))


def rating(ns: dict[str, Any]) -> float:
    """The study line's rating, MW."""
    pt = ns["pt"]
    return float(pt.s_nom[pt.line_idx[ns["STUDY_LINE"]]])


def offer(ns: dict[str, Any], unit: str) -> float:
    """The offer price of ``unit`` in the EDIT cell's fleet, $/MWh."""
    return float(ns["GEN_FLEET"][unit]["cost"])


def redispatch(res_1: EngineResult, res_0: EngineResult, unit: str) -> float:
    """The change in ``unit``'s dispatch from ``res_0`` to ``res_1``, MW."""
    return float(res_1.dispatch[unit] - res_0.dispatch[unit])


def test_slide_03_case_counts_match(ns: dict[str, Any]) -> None:
    """Slide 3 prints the case's counts: 3 generators, 9 lines and 3 loads."""
    check_slide(
        3,
        [
            ("generators", len(ns["GEN_FLEET"]), 3, HALF_UNIT),
            ("lines", len(ns["pt"].lines), 9, HALF_UNIT),
            (
                "loads",
                sum(v > 0 for v in ns["CONFIGURED_LOADS"].values()),
                3,
                HALF_UNIT,
            ),
        ],
    )


def test_slide_04_case_inputs_match(ns: dict[str, Any]) -> None:
    """Slide 4 prints the nine line ratings, the three offers and capacities, and the loads.

    The loads are the configured 125, 100 and 90 MW, not the drawn loads the clears serve.
    """
    pt, fleet, loads = ns["pt"], ns["GEN_FLEET"], ns["CONFIGURED_LOADS"]
    ratings = {
        "line_6": 250,
        "line_7": 250,
        "line_8": 250,
        "line_0": 300,
        "line_5": 250,
        "line_1": 250,
        "line_4": 40,
        "line_2": 150,
        "line_3": 300,
    }
    check_slide(
        4,
        [
            (f"rating {m} (MW)", float(pt.s_nom[pt.line_idx[m]]), v, HALF_UNIT)
            for m, v in ratings.items()
        ]
        + [
            (f"{g} offer ($/MWh)", fleet[g]["cost"], v, HALF_UNIT)
            for g, v in {"gen_0": 35, "gen_slack_0": 50, "gen_1": 20}.items()
        ]
        + [
            (f"{g} capacity (MW)", fleet[g]["p_nom"], v, HALF_UNIT)
            for g, v in {"gen_0": 300, "gen_slack_0": 300, "gen_1": 170}.items()
        ]
        + [
            (f"configured load bus {b} (MW)", loads[b], v, HALF_UNIT)
            for b, v in {"9": 125, "7": 100, "5": 90}.items()
        ],
    )


def test_slide_06_market_a_clear_holds_anchor_and_matches(ns: dict[str, Any]) -> None:
    """Slide 6 prints Market A's first clear: $2,535/h, the injections, flows and ring prices.

    The slide prints neither the dual of $20.10/MWh nor the rent of $804.1/h, so the
    anchor is asserted apart from the slide's rows. Line 6-7 binds at 40 MW, the
    settlement total and the rent-by-line total each equal R, and the dual is the interior
    pair's offer gap over their shift-factor gap, (35 - 20) / (SF_3 - SF_2).
    """
    a1, ra1 = ns["A1"], ns["rA1"]
    flow = float(ra1.flow_own[ns["STUDY_LINE"]])
    mu = abs_mu(ns, ra1)
    settle = settlement_by_bus(ra1, ns["MARKETS"]["Market A"], a1.loads)
    rent = capacity_value_by_line(ra1, ns["pt"], a1.activated_lines)
    r_settle = float(
        settle.loc["SUBTOTAL", "paid by load ($/h)"]
        - settle.loc["SUBTOTAL", "paid to gen ($/h)"]
    )
    r_lines = float(rent.loc["TOTAL", "rent |mu| x flow ($/h)"])
    mu_pair = (offer(ns, "gen_0") - offer(ns, "gen_1")) / (
        shift_factor(ns, "3") - shift_factor(ns, "2")
    )

    assert abs(flow) == pytest.approx(ANCHOR_FLOW, abs=ANCHOR_FLOW_TOL)
    assert mu == pytest.approx(ANCHOR_MU, abs=ANCHOR_MU_TOL)
    for r in (mu * abs(flow), r_settle, r_lines):
        assert r == pytest.approx(ANCHOR_RENT, abs=ANCHOR_RENT_TOL)
    assert mu == pytest.approx(mu_pair, abs=EXACT_ABS)

    check_slide(
        6,
        rows_of([("Market A production cost ($/h)", ra1.total_cost, 2535, HALF_UNIT)])
        + injection_rows(net_injection(ns, ra1), {"2": 49, "3": 41, "7": -90})
        + flow_rows(
            ra1.flow_own,
            {
                "line_6": 49,
                "line_5": 50,
                "line_4": 40,
                "line_3": 41,
                "line_2": 1,
                "line_1": 1,
                "line_7": 1,
                "line_8": 1,
            },
        )
        + price_rows(
            ra1.lmp,
            {
                "2": 35.0,
                "3": 20.0,
                "7": 37.1,
                "9": 30.2,
                "4": 27.7,
                "1": 27.7,
                "5": 25.0,
            },
        ),
    )


def test_slide_07_bus_7_walk_equals_lmp_and_matches(ns: dict[str, Any]) -> None:
    """Slide 7 prints the bus-7 price as the cost of one more MW of load there.

    The slide prints $39.90 less $2.80, $37.10, the products of the coefficients rounded
    to 1.14 MW and 0.14 MW. The unrounded products are $39.96 and $2.83, and their
    difference is the engine's $37.126/MWh, which the ring on slide 6 labels $37.1. The
    rows rebuild the slide's products at its rounding, and the walk's total is asserted
    against the LMP unrounded.
    """
    walk, ra1 = ns["WALK_7"], ns["rA1"]
    c_hi, c_lo = offer(ns, "gen_0"), offer(ns, "gen_1")
    dg2 = float(walk.loc["gen_0", "redispatch (MW)"])
    dg3 = float(walk.loc["gen_1", "redispatch (MW)"])
    assert float(walk.loc["TOTAL", "cost change ($/h)"]) == pytest.approx(
        ra1.lmp["7"], abs=EXACT_ABS
    )
    check_slide(
        7,
        [
            ("bus 2 unit up (MW)", dg2, 1.14, HALF_HUNDREDTH),
            ("bus 3 unit down (MW)", -dg3, 0.14, HALF_HUNDREDTH),
            (
                "bus 2 unit cost on the rounded coefficient ($/h)",
                round(dg2, 2) * c_hi,
                39.90,
                HALF_HUNDREDTH,
            ),
            (
                "bus 3 unit saving on the rounded coefficient ($/h)",
                round(-dg3, 2) * c_lo,
                2.80,
                HALF_HUNDREDTH,
            ),
            (
                "bus 7 on the rounded coefficients ($/MWh)",
                round(dg2, 2) * c_hi - round(-dg3, 2) * c_lo,
                37.10,
                HALF_HUNDREDTH,
            ),
            ("bus 7 LMP, ring label ($/MWh)", ra1.lmp["7"], 37.1, HALF_TENTH),
        ],
    )


def test_slide_08_line_walk_equals_dual_and_matches(ns: dict[str, Any]) -> None:
    """Slide 8 prints the line 6-7 dual as the cost of one MW off the line.

    The slide multiplies the rounded 1.34 MW, which gives $46.90 and $26.80. The unrounded
    products are $46.91 and $26.80, and their difference is $20.10/MWh either way. The
    walk's redispatch is 1 / (SF_3 - SF_2) MW and its cost is |mu|, both unrounded.
    """
    walk = ns["WALK_L"]
    mu = abs_mu(ns, ns["rA1"])
    c_hi, c_lo = offer(ns, "gen_0"), offer(ns, "gen_1")
    dg2 = float(walk.loc["gen_0", "redispatch (MW)"])
    dg3 = float(walk.loc["gen_1", "redispatch (MW)"])
    assert float(walk["cost change ($/h)"].sum()) == pytest.approx(mu, abs=EXACT_ABS)
    assert dg2 == pytest.approx(
        1.0 / (shift_factor(ns, "3") - shift_factor(ns, "2")), abs=EXACT_ABS
    )
    check_slide(
        8,
        [
            ("bus 2 unit up (MW)", dg2, 1.34, HALF_HUNDREDTH),
            ("bus 3 unit down (MW)", -dg3, 1.34, HALF_HUNDREDTH),
            (
                "bus 2 unit cost on the rounded coefficient ($/h)",
                round(dg2, 2) * c_hi,
                46.90,
                HALF_HUNDREDTH,
            ),
            (
                "bus 3 unit saving on the rounded coefficient ($/h)",
                round(-dg3, 2) * c_lo,
                26.80,
                HALF_HUNDREDTH,
            ),
            ("|mu| line 6-7 ($/MWh)", mu, 20.10, HALF_HUNDREDTH),
        ],
    )


def test_slide_09_market_b_clear_matches(ns: dict[str, Any]) -> None:
    """Slide 9 prints Market B's clear: 9.4 MW on line 6-7 and every price at $50/MWh.

    The slide's ring prints no price at bus 4.
    """
    rb, pt = ns["rB"], ns["pt"]
    check_slide(
        9,
        rows_of(
            [
                (
                    "Market B flow on line 6-7 (MW)",
                    rb.flow_own[ns["STUDY_LINE"]],
                    9.4,
                    HALF_TENTH,
                )
            ]
        )
        + injection_rows(net_injection(ns, rb), {"9": -173, "1": 263, "5": -90})
        + flow_rows(
            rb.flow_own,
            {
                "line_0": 263,
                "line_8": 164,
                "line_1": 99,
                "line_7": 9,
                "line_5": 9,
                "line_4": 9,
                "line_2": 9,
            },
        )
        + price_rows(rb.lmp, {b: 50.0 for b in pt.buses if b != "4"}),
    )


def test_slide_10_superposition_overloads_and_matches(ns: dict[str, Any]) -> None:
    """Slide 10 prints the superposition of the two clears, which overloads line 6-7.

    The slide prints the superposed flow on line 6-7 only as the label 49/40, and the
    test compares the run with the 49.4 MW the slide 10 cell's text states. Its line labels are those of
    ``seam_views.combined_views``, the flow rounded to 0.1 MW and then to a whole MW,
    half to even, so the flow rows compare that label.
    """
    ra1, rb = ns["rA1"], ns["rB"]
    fs = superposed(ns, ra1, rb)
    text, tol = text_value(
        "### Slide 10", rf"{NUMERAL} MW against its {NUMERAL} MW rating"
    )
    assert bool(ns["PF1"].loc[ns["STUDY_LINE"], "overload"])
    assert fs[ns["STUDY_LINE"]] == pytest.approx(text, abs=tol)
    check_slide(
        10,
        injection_rows(
            net_injection(ns, ra1, rb),
            {"2": 49, "9": -173, "1": 263, "7": -90, "5": -90, "3": 41},
        )
        + flow_rows(
            fs,
            {
                "line_6": 49,
                "line_7": 9,
                "line_0": 263,
                "line_8": 164,
                "line_5": 40,
                "line_1": 99,
                "line_4": 49,
                "line_2": 9,
                "line_3": 41,
            },
            two_step=True,
        ),
    )


def test_slide_11_accommodation_ledger_closes_and_matches(ns: dict[str, Any]) -> None:
    """Slide 11 prints Market A's redispatch to Market B's realized flow: $2,725/h and $190/h.

    The slide writes 9.4 MW at $20.10/MWh as $190. The product of the rounded factors is
    $188.94, and the unrounded factors give $189.89, which rounds to $190, so the row
    compares the unrounded product. The cost change equals |mu| times Market B's flow by
    the notebook's cost-of-an-input identity, the superposed flow on line 6-7 sits at the
    rating, and the notebook's accommodation ledger closes: its unit rows sum to the cost
    change and its residual is zero.
    """
    m = ns["STUDY_LINE"]
    ra1, ra1r, rb = ns["rA1"], ns["rA1r"], ns["rB"]
    mu, fb = abs_mu(ns, ra1), float(rb.flow_own[m])
    dg2, dg3 = redispatch(ra1r, ra1, "gen_0"), redispatch(ra1r, ra1, "gen_1")
    change = ra1r.total_cost - ra1.total_cost
    fs = superposed(ns, ra1r, rb)

    assert change == pytest.approx(mu * fb, abs=IDENTITY_ABS)
    assert fs[m] == pytest.approx(rating(ns), abs=IDENTITY_ABS)
    ledger = ns["ACC_LEDGER"]["cost ($/h)"]
    assert float(ledger["change in production cost, the sum"]) == pytest.approx(
        change, abs=IDENTITY_ABS
    )
    assert float(ledger["residual, the sum less (7)"]) == pytest.approx(
        0.0, abs=IDENTITY_ABS
    )

    mkt_a = ns["MARKETS"]["Market A"]
    check_slide(
        11,
        rows_of(
            [
                ("Market A production cost ($/h)", ra1r.total_cost, 2725, HALF_UNIT),
                ("independent run ($/h)", ra1.total_cost, 2535, HALF_UNIT),
                ("bus 2 unit up (MW)", dg2, 12.66, HALF_HUNDREDTH),
                ("bus 3 unit down (MW)", -dg3, 12.66, HALF_HUNDREDTH),
                ("bus 2 unit cost ($/h)", dg2 * offer(ns, "gen_0"), 443, HALF_UNIT),
                ("bus 3 unit saving ($/h)", -dg3 * offer(ns, "gen_1"), 253, HALF_UNIT),
                ("change in production cost ($/h)", change, 190, HALF_UNIT),
                ("Market B flow on line 6-7 (MW)", fb, 9.4, HALF_TENTH),
                ("|mu| line 6-7 ($/MWh)", abs_mu(ns, ra1r), 20.10, HALF_HUNDREDTH),
                ("|mu| x F^B, unrounded ($/h)", mu * fb, 190, HALF_UNIT),
            ]
        )
        + injection_rows(
            net_injection(ns, ra1r, rb),
            {"2": 62, "3": 28, "9": -173, "1": 263, "7": -90, "5": -90},
        )
        + flow_rows(
            fs,
            {
                "line_6": 62,
                "line_7": 12,
                "line_8": 161,
                "line_0": 263,
                "line_5": 50,
                "line_1": 102,
                "line_4": 40,
                "line_2": 12,
                "line_3": 28,
            },
            two_step=True,
        )
        + price_rows(
            {**rb.lmp, **{b: ra1r.lmp[b] for b in mkt_a}},
            {"9": 50.0, "1": 50.0, "5": 50.0, "2": 35.0, "7": 37.1, "3": 20.0},
        ),
    )


def test_slide_13_forecast_book_matches(ns: dict[str, Any]) -> None:
    """Slide 13 prints Market A's forecast of Market B's book: 11.5 MW on line 6-7.

    Market A's side of the slide is drawn as configured, at nameplate and with the
    configured 100 MW load at bus 7.
    """
    bf, rbf, fleet = ns["Bf"], ns["rBf"], ns["GEN_FLEET"]
    check_slide(
        13,
        rows_of(
            [
                (
                    "forecast flow contribution on line 6-7 (MW)",
                    rbf.flow_own[ns["STUDY_LINE"]],
                    11.5,
                    HALF_TENTH,
                ),
                ("forecast bus 9 load (MW)", bf.loads["9"], 190, HALF_UNIT),
                ("forecast bus 1 dispatch (MW)", rbf.gen_by_bus["1"], 280, HALF_UNIT),
                ("bus 5 load (MW)", bf.loads["5"], 90, HALF_UNIT),
                (
                    "bus 2 capacity, as configured (MW)",
                    fleet["gen_0"]["p_nom"],
                    300,
                    HALF_UNIT,
                ),
                (
                    "bus 7 load, as configured (MW)",
                    ns["CONFIGURED_LOADS"]["7"],
                    100,
                    HALF_UNIT,
                ),
                (
                    "bus 3 capacity, as configured (MW)",
                    fleet["gen_1"]["p_nom"],
                    170,
                    HALF_UNIT,
                ),
            ]
        )
        + injection_rows(net_injection(ns, rbf), {"9": -190, "1": 280, "5": -90})
        + flow_rows(
            rbf.flow_own,
            {
                "line_0": 280,
                "line_8": 178,
                "line_1": 102,
                "line_7": 12,
                "line_5": 12,
                "line_4": 12,
                "line_2": 12,
            },
        ),
    )


def test_slide_14_forecast_clear_follows_identity_and_matches(
    ns: dict[str, Any],
) -> None:
    """Slide 14 prints Market A's clear against the forecast: $2,767/h and $232/h.

    The one exception in the deck: slide 14 prints the redispatch of each unit as
    15.45 MW, and the run gives 15.47 MW. The test asserts that the run's redispatch is the
    forecast flow times the line walk's 1 / (SF_3 - SF_2) MW per MW, that each unit's
    redispatch matches the 15.47 MW the slide 14 cell's text states, and that the slide's
    15.45 MW lies outside half a unit of the run's value. The slide's dollar figures hold at either value. Its
    11.5 MW at $20.10/MWh is $231.15 on the rounded factors and $232.04 on the unrounded
    ones, which the slide prints as $232, so the row compares the unrounded product. The
    cost change equals |mu| times the forecast flow by the cost-of-an-input identity.
    """
    m = ns["STUDY_LINE"]
    ra1, ra2, rbf = ns["rA1"], ns["rA2"], ns["rBf"]
    mu, fb_hat = abs_mu(ns, ra1), float(rbf.flow_own[m])
    dg2, dg3 = redispatch(ra2, ra1, "gen_0"), redispatch(ra2, ra1, "gen_1")
    change = ra2.total_cost - ra1.total_cost

    assert change == pytest.approx(mu * fb_hat, abs=IDENTITY_ABS)
    assert dg2 == pytest.approx(
        fb_hat / (shift_factor(ns, "3") - shift_factor(ns, "2")), abs=IDENTITY_ABS
    )
    text, tol = text_value("### Slide 14", rf"falls by {NUMERAL} MW each")
    assert dg2 == pytest.approx(text, abs=tol)
    assert -dg3 == pytest.approx(text, abs=tol)
    assert abs(dg2 - SLIDE_14_REDISPATCH) > SLIDE_14_REDISPATCH_TOL

    mkt_a = ns["MARKETS"]["Market A"]
    check_slide(
        14,
        rows_of(
            [
                ("Market A production cost ($/h)", ra2.total_cost, 2767, HALF_UNIT),
                ("independent run ($/h)", ra1.total_cost, 2535, HALF_UNIT),
                ("bus 2 unit cost ($/h)", dg2 * offer(ns, "gen_0"), 541, HALF_UNIT),
                ("bus 3 unit saving ($/h)", -dg3 * offer(ns, "gen_1"), 309, HALF_UNIT),
                ("change in production cost ($/h)", change, 232, HALF_UNIT),
                ("forecast flow on line 6-7 (MW)", fb_hat, 11.5, HALF_TENTH),
                ("|mu| line 6-7 ($/MWh)", abs_mu(ns, ra2), 20.10, HALF_HUNDREDTH),
                ("|mu| x forecast, unrounded ($/h)", mu * fb_hat, 232, HALF_UNIT),
            ]
        )
        + injection_rows(
            net_injection(ns, ra2, rbf),
            {"2": 65, "3": 25, "7": -90, "9": -190, "1": 280, "5": -90},
        )
        + flow_rows(
            superposed(ns, ra2, rbf),
            {
                "line_6": 65,
                "line_7": 15,
                "line_8": 175,
                "line_0": 280,
                "line_5": 50,
                "line_1": 105,
                "line_4": 40,
                "line_2": 15,
                "line_3": 25,
            },
        )
        + price_rows(
            {**rbf.lmp, **{b: ra2.lmp[b] for b in mkt_a}},
            {
                "9": 50.0,
                "1": 50.0,
                "5": 50.0,
                "4": 50.0,
                "2": 35.0,
                "7": 37.1,
                "3": 20.0,
            },
        ),
    )


def test_slide_15_market_b_clear_unchanged_and_matches(ns: dict[str, Any]) -> None:
    """Slide 15 prints Market B's clear with its forecast of Market A in gray.

    No line Market B enforces binds with or without its input, so its cost and dispatch
    are those of slide 9. Its forecast of Market A holds line 6-7 at the rating, which the
    slide prints no row for. The studied superposition on line 6-7 is printed only as the
    label 49/40, and the test compares the run with the 49.4 MW the slide 15 cell's text
    states.
    """
    m = ns["STUDY_LINE"]
    rb, rb2, rap = ns["rB"], ns["rB2"], ns["rAp"]
    assert rb2.total_cost == pytest.approx(rb.total_cost, abs=IDENTITY_ABS)
    for g in rb.dispatch:
        assert rb2.dispatch[g] == pytest.approx(rb.dispatch[g], abs=IDENTITY_ABS)
    view = superposed(ns, rb2, rap)
    assert abs(float(rap.flow_own[m])) == pytest.approx(rating(ns), abs=HALF_TENTH)
    text, tol = text_value(
        "### Slide 15", rf"{NUMERAL} MW on line 6-7 against the {NUMERAL} MW rating"
    )
    assert view[m] == pytest.approx(text, abs=tol)

    mkt_b = ns["MARKETS"]["Market B"]
    check_slide(
        15,
        injection_rows(
            net_injection(ns, rb2, rap),
            {"2": 41, "3": 42, "7": -83, "9": -173, "1": 263, "5": -90},
        )
        + flow_rows(
            view,
            {
                "line_6": 41,
                "line_7": 8,
                "line_8": 165,
                "line_0": 263,
                "line_5": 34,
                "line_1": 98,
                "line_4": 49,
                "line_2": 8,
                "line_3": 42,
            },
        )
        + price_rows(
            {**rap.lmp, **{b: rb2.lmp[b] for b in mkt_b}},
            {
                "9": 50.0,
                "1": 50.0,
                "5": 50.0,
                "2": 35.0,
                "8": 35.0,
                "7": 37.1,
                "6": 20.0,
                "3": 20.0,
            },
        ),
    )


def test_slide_16_superposition_below_rating_and_matches(ns: dict[str, Any]) -> None:
    """Slide 16 prints the superposition of Example 2, line 6-7 below its rating.

    The slide prints the superposed flow on line 6-7 only as the label 38/40, and the
    test compares the run with the 37.9 MW the slide 16 cell's text states. The slide labels four lines one MW below the run's
    flow rounded directly (line 8-2 at 64 MW for 64.54 MW, and lines 8-9, 4-5 and 5-6),
    because ``seam_views.combined_views`` rounds to 0.1 MW and then to a whole MW, half to
    even, so the flow rows compare that label.
    """
    ra2, rb2 = ns["rA2"], ns["rB2"]
    fs = superposed(ns, ra2, rb2)
    assert not bool(ns["PF2"].loc[ns["STUDY_LINE"], "overload"])
    text, tol = text_value(
        "### Slide 16", rf"{NUMERAL} MW against its {NUMERAL} MW rating"
    )
    assert fs[ns["STUDY_LINE"]] == pytest.approx(text, abs=tol)
    check_slide(
        16,
        injection_rows(
            net_injection(ns, ra2, rb2),
            {"2": 65, "9": -173, "1": 263, "7": -90, "5": -90, "3": 25},
        )
        + flow_rows(
            fs,
            {
                "line_6": 64,
                "line_7": 12,
                "line_8": 161,
                "line_0": 263,
                "line_5": 52,
                "line_1": 102,
                "line_4": 38,
                "line_2": 12,
                "line_3": 25,
            },
            two_step=True,
        ),
    )


def test_slide_17_phantom_ledger_closes_and_matches(ns: dict[str, Any]) -> None:
    """Slide 17 prints the cost of phantom congestion: $2,767/h less $2,725/h, $42/h.

    The slide's two costs are Market A's clear with headroom on the constraint, on the
    forecast, and its clear at the limit, on Market B's realized flow.

    The slide's 2.1 MW at $20.10/MWh is $42.21 on the rounded factors and $42.15 on the
    unrounded ones, and both print as $42. The difference equals |mu| times the
    over-forecast by the cost-of-an-input identity applied to both clears, |mu| is the
    same in every Market A clear, and the notebook's phantom-congestion ledger closes: each
    residual is zero and the unused capacity equals the over-forecast.
    """
    m = ns["STUDY_LINE"]
    ra1, ra1r, ra2, rap, rb2 = ns["rA1"], ns["rA1r"], ns["rA2"], ns["rAp"], ns["rB2"]
    mu = abs_mu(ns, ra1)
    over = float(ns["rBf"].flow_own[m] - ns["rB"].flow_own[m])
    headroom = ra2.total_cost - ra1r.total_cost

    assert headroom == pytest.approx(mu * over, abs=IDENTITY_ABS)
    for r in (ra1r, ra2, rap):
        assert abs_mu(ns, r) == pytest.approx(mu, abs=EXACT_ABS)
    ledger = ns["PHANTOM_LEDGER"]
    for res in ledger["residual ($/h)"].dropna():
        assert float(res) == pytest.approx(0.0, abs=IDENTITY_ABS)
    unused = ledger.loc["unused capacity, rating less superposed flow"].iloc[0]
    assert float(unused) == pytest.approx(over, abs=IDENTITY_ABS)

    fs = superposed(ns, ra2, rb2)
    mkt_a = ns["MARKETS"]["Market A"]
    check_slide(
        17,
        rows_of(
            [
                ("Market A cost with headroom ($/h)", ra2.total_cost, 2767, HALF_UNIT),
                ("Market A cost at the limit ($/h)", ra1r.total_cost, 2725, HALF_UNIT),
                ("increased production cost ($/h)", headroom, 42, HALF_UNIT),
                ("over-forecast on line 6-7 (MW)", over, 2.1, HALF_TENTH),
                ("|mu| line 6-7 ($/MWh)", mu, 20.10, HALF_HUNDREDTH),
                (
                    "2.1 MW x $20.10 on the rounded factors ($/h)",
                    round(over, 1) * round(mu, 2),
                    42,
                    HALF_UNIT,
                ),
            ]
        )
        + flow_rows(
            fs,
            {
                "line_6": 64,
                "line_7": 12,
                "line_8": 161,
                "line_0": 263,
                "line_5": 52,
                "line_1": 102,
                "line_4": 38,
                "line_2": 12,
                "line_3": 25,
            },
            two_step=True,
        )
        + injection_rows(
            net_injection(ns, ra2, rb2),
            {"2": 65, "9": -173, "1": 263, "7": -90, "5": -90, "3": 25},
        )
        + price_rows(
            {**rb2.lmp, **{b: ra2.lmp[b] for b in mkt_a}},
            {"9": 50.0, "1": 50.0, "5": 50.0, "2": 35.0, "7": 37.1, "3": 20.0},
        ),
    )
