# SPDX-FileCopyrightText: Market Design Lab contributors
#
# SPDX-License-Identifier: MIT

"""Slide checks for ``workshop4_loading_relief.ipynb``, the October 22, 2026 deck.

The module executes the notebook's code cells once, through
:func:`slide_check.run_notebook`, and holds the clears in that namespace against the
numbers the deck prints. One test covers one slide. A slide row's tolerance is half a
unit of the slide's last printed digit, one of ``WHOLE_TOL``, ``TENTH_TOL``,
``HUNDREDTH_TOL`` and ``THOUSANDTH_TOL``. The anchor rows hold Market A's first clear
to the series anchor, line 6-7 at 40 MW with a dual of $20.10/MWh and a rent of
$804.1/h, at the ``ANCHOR_*_TOL`` tolerances. An identity between two computations
from the same numbers, such as a sum of tag impacts against a flow contribution, holds
to ``EXACT_ABS``. A solved cost, flow, dispatch or dual against its identity or its
limit holds to ``IDENTITY_ABS``, and the superposition of the two capped clears against
line 6-7's rating to ``SUPERPOSITION_ABS``. Every tolerance is absolute. A number the
notebook's text states is read from its markdown cell and compared there with the run,
printed as the text prints it, so no run-produced number is typed into this module.

The relief tables of slides 70, 76, 80 and 81 are compared as printed text, cell for
cell, because their tag names, priorities and "not eligible" entries are text.

The first reconciliation picture of slides 63 and 85 is the August workshop's picture,
line 6-7 at 43/40 with net injections of +58 MW at bus 2 and +32 MW at bus 3. The
notebook does not run that dispatch, and no test here checks it. Slides 52, 53, 56, 64,
72, 78, 79, 85 and 86 carry no number the notebook clears apart from those the other
slides print.

    python -m pytest tests -q -m slides
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nbformat
import numpy as np
import pytest

from slide_check import (
    ROOT,
    check_slide,
    flow_rows,
    injection_rows,
    price_rows,
    run_notebook,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import pandas as pd

pytestmark = pytest.mark.slides

NOTEBOOK = "workshop4_loading_relief.ipynb"

#: Absolute tolerance of two computations of one quantity from the same numbers, MW,
#: $/h or $/MWh, at floating-point precision.
EXACT_ABS = 1e-9
#: Absolute tolerance of a solved cost, flow, dispatch or dual against the identity or
#: the limit the formulation states for it, $/h, MW or $/MWh, at the solver's precision.
IDENTITY_ABS = 1e-6
#: Absolute tolerance of the superposition of Example 4's two capped clears against
#: line 6-7's rating, MW. Each clear holds its flow at its limit to ``IDENTITY_ABS``
#: and the two limits sum to the rating to ``EXACT_ABS``, so the sum of the two flows
#: sits within twice ``IDENTITY_ABS``.
SUPERPOSITION_ABS = 2 * IDENTITY_ABS
#: The series anchor, Market A's first clear, as NOTEBOOK_RECORD.md states it: line
#: 6-7 at its 40 MW rating, MW, with |mu| = $20.10/MWh and R = $804.1/h, each with an
#: absolute tolerance of half a unit of its last stated digit.
ANCHOR_FLOW, ANCHOR_FLOW_TOL = 40.0, 0.05
ANCHOR_MU, ANCHOR_MU_TOL = 20.10, 0.005
ANCHOR_RENT, ANCHOR_RENT_TOL = 804.1, 0.05
#: Absolute tolerance of a slide value printed to a whole unit, half that unit, in the
#: unit of the row's quantity (MW, $/h, $/MWh or a count).
WHOLE_TOL = 0.5
#: Absolute tolerance of a slide value printed to 0.1 of its unit, half of 0.1.
TENTH_TOL = 0.05
#: Absolute tolerance of a slide value printed to 0.01 of its unit, half of 0.01.
HUNDREDTH_TOL = 0.005
#: Absolute tolerance of a slide value printed to 0.001 of its unit, half of 0.001.
THOUSANDTH_TOL = 0.0005
#: Absolute tolerance of a label's visible digits, which the row compares as a whole
#: number read off the label, so it is zero.
LABEL_DIGITS_TOL = 0.0


@pytest.fixture(scope="module")
def ns() -> dict[str, Any]:
    """The namespace of one execution of the notebook's code cells."""
    return run_notebook(NOTEBOOK)


def markdown_of(heading: str) -> str:
    """The source of the notebook's one markdown cell that opens with ``heading``."""
    nb = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    cells = [
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and c.source.startswith(heading)
    ]
    assert len(cells) == 1, f"{len(cells)} markdown cells open with {heading!r}"
    return str(cells[0])


# ─── Row builders the slides share ───────────────────────────────────────────


def panel_rows(
    flows: Mapping[str, float],
    inj: Mapping[str, float],
    slide_flows: Mapping[str, float],
    slide_inj: Mapping[str, float],
    box: Mapping[str, tuple[float, float, float]],
) -> list[tuple[str, float, float, float, float]]:
    """The rows of one tag panel: its line labels, its node chips and its box.

    ``box`` maps a box entry to ``(model, slide, tolerance)``.
    """
    return (
        flow_rows(flows, slide_flows)
        + injection_rows(inj, slide_inj)
        + [(f"box {k}", v, s, tol, v) for k, (v, s, tol) in box.items()]
    )


def rows(*parts: Iterable[Sequence[Any]]) -> list[Sequence[Any]]:
    """The row lists ``parts`` joined in order, as one slide's rows."""
    return [r for part in parts for r in part]


def prefixed(prefix: str, part: Iterable[Sequence[Any]]) -> list[tuple[Any, ...]]:
    """The rows of ``part`` with ``prefix`` before each quantity name."""
    return [(f"{prefix} {r[0]}", *r[1:]) for r in part]


def neighbor_rows(
    ns: Mapping[str, Any],
    side: str,
    eng: Any,
    res: Any,
    slide: Mapping[str, float],
    offsets: Mapping[str, float] | None = None,
) -> list[tuple[str, float, float, float]]:
    """Rows comparing an inputs frame's chips at the neighbor's buses with the slide.

    The chips are ``assumed_injections(side, res)``, the values the notebook's inputs
    frame of clear ``res`` draws. The function asserts that ``res`` is ``eng``'s clear
    for ``side``, that the chips sit at the slide's buses, and that the flow input
    ``offsets`` the clear carried equals the flow the chips place on its activated
    lines, the sum over the chips of SF_{n,m} times the chip.
    """
    pt = ns["pt"]
    assert eng.name == side, f"{eng.name} is not {side}'s engine"
    assert set(res.dispatch) == set(eng.gens), f"{side}: res is not {eng.name}'s clear"
    x = ns["assumed_injections"](side, res)
    assert set(x) == set(slide), (
        f"{side}: chips at {sorted(x)}, slide at {sorted(slide)}"
    )
    for m in eng.activated_lines:
        placed = sum(pt.ptdf[pt.line_idx[m], pt.bus_idx[b]] * v for b, v in x.items())
        assert abs(placed - (offsets or {}).get(m, 0.0)) < EXACT_ABS, (
            f"{side}: {m} input differs"
        )
    return [
        (f"assumed injection bus {b} (MW)", x[b], v, WHOLE_TOL)
        for b, v in slide.items()
    ]


def assert_table(
    slide: int, text: pd.DataFrame, expected: Sequence[Sequence[Any]]
) -> None:
    """Assert that the printed table ``text`` reads, cell for cell, as the slide's table."""
    got = text.reset_index().astype(str).to_numpy().tolist()
    exp = [[str(v) for v in row] for row in expected]
    assert len(got) == len(exp), f"slide {slide}: {len(got)} rows against {len(exp)}"
    bad = [(g, e) for g, e in zip(got, exp, strict=True) if g != e]
    assert not bad, f"slide {slide}: rows differ: {bad}"


def host_lmp(
    ns: Mapping[str, Any], pairs: Sequence[tuple[Any, Any]]
) -> dict[str, float]:
    """Each bus's LMP in the clear of the market that holds it, as a ring reads it."""
    fp, pt = ns["fp"], ns["pt"]
    return {
        b: float(next(r for e, r in pairs if fp.fp_of(b) == e.name).lmp[b])
        for b in pt.buses
    }


def label_two_digits(flow: float) -> float:
    """The first two digits of a three-digit whole-MW label, where an arc covers the third."""
    return float(int(np.round(abs(flow))) // 10)


# ─── The case ─────────────────────────────────────────────────────────────────


def test_slide_54_case_counts_match(ns: dict[str, Any]) -> None:
    """Slide 54 prints the case's counts: 3 generators, 9 lines and 3 loads."""
    check_slide(
        54,
        [
            ("generators", len(ns["GEN_FLEET"]), 3, WHOLE_TOL),
            ("lines", len(ns["pt"].lines), 9, WHOLE_TOL),
            ("loads", sum(v > 0 for v in ns["LOADS"].values()), 3, WHOLE_TOL),
        ],
    )


def test_slide_55_ratings_offers_capacities_and_loads_match(ns: dict[str, Any]) -> None:
    """Slide 55 prints each line's rating, each unit's offer and capacity, and each load.

    Ratings, offers and capacities are whole numbers. The loads are rounded to whole MW.
    """
    pt, fleet, loads = ns["pt"], ns["FLEET"], ns["LOADS"]
    ratings = {
        "line_7": 250,
        "line_8": 250,
        "line_6": 250,
        "line_0": 300,
        "line_5": 250,
        "line_1": 250,
        "line_4": 40,
        "line_2": 150,
        "line_3": 300,
    }
    offers = {
        "gen_0": 35,
        "gen_slack_0": 50,
        "gen_1": 20,
        "backstop_9": 55,
        "backstop_7": 55,
        "backstop_5": 55,
    }
    capacities = {
        "gen_0": 300,
        "gen_slack_0": 300,
        "gen_1": 170,
        "backstop_9": 80,
        "backstop_7": 80,
        "backstop_5": 80,
    }
    check_slide(
        55,
        rows(
            [
                (f"rating {m} (MW)", float(pt.s_nom[pt.line_idx[m]]), v, WHOLE_TOL)
                for m, v in ratings.items()
            ],
            [
                (f"{g} offer ($/MWh)", fleet[g]["cost"], v, WHOLE_TOL)
                for g, v in offers.items()
            ],
            [
                (f"{g} capacity (MW)", fleet[g]["p_nom"], v, WHOLE_TOL)
                for g, v in capacities.items()
            ],
            [
                (f"load bus {b} (MW)", loads[b], v, WHOLE_TOL)
                for b, v in {"9": 173, "7": 90, "5": 90}.items()
            ],
        ),
    )


# ─── Example 1 ────────────────────────────────────────────────────────────────


def test_slide_57_market_a_clearing_matches(ns: dict[str, Any]) -> None:
    """Slide 57 prints Market A's clear: its cost, net injections, flows and LMPs.

    The cost is to the dollar, the injections and flows to whole MW and the LMPs to
    $0.1/MWh. The slide prints neither the dual nor the rent. The anchor rows hold
    Market A's clear and the base clearing, one engine over both footprints, to line
    6-7 at 40 MW, |mu| = $20.10/MWh and R = $804.1/h, the rent both as the settlement
    total and as |mu| times |F| over the activated lines.
    """
    rA1 = ns["rA1"]
    check_slide(
        57,
        rows(
            [("Market A production cost ($/h)", rA1.total_cost, 2535, WHOLE_TOL)],
            injection_rows(ns["net_injection"](rA1), {"2": 49, "3": 41, "7": -90}),
            flow_rows(
                rA1.flow_own,
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
            ),
            price_rows(
                rA1.lmp,
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
        ),
    )
    line = ns["STUDY_LINE"]
    rBASE, settle_base, base = ns["rBASE"], ns["SETTLE_BASE"], ns["BASE"]
    r_settle_base = (
        settle_base.loc["SUBTOTAL", "paid by load ($/h)"]
        - settle_base.loc["SUBTOTAL", "paid to gen ($/h)"]
    )
    r_lines_base = sum(
        abs(rBASE.line_dual[m] * rBASE.flow_own[m]) for m in base.activated_lines
    )
    check_slide(
        "57, the anchor",
        [
            (
                "Market A |F| line 6-7 (MW)",
                abs(ns["F_A1"]),
                ANCHOR_FLOW,
                ANCHOR_FLOW_TOL,
            ),
            ("Market A |mu| line 6-7 ($/MWh)", ns["MU"], ANCHOR_MU, ANCHOR_MU_TOL),
            (
                "Market A R, settlement total ($/h)",
                ns["R_SETTLE"],
                ANCHOR_RENT,
                ANCHOR_RENT_TOL,
            ),
            (
                "Market A R, line total ($/h)",
                ns["R_LINES"],
                ANCHOR_RENT,
                ANCHOR_RENT_TOL,
            ),
            (
                "Market A R, |mu| x |F^A| ($/h)",
                ns["R_A1"],
                ANCHOR_RENT,
                ANCHOR_RENT_TOL,
            ),
            (
                "base clearing |F| line 6-7 (MW)",
                abs(rBASE.flow_own[line]),
                ANCHOR_FLOW,
                ANCHOR_FLOW_TOL,
            ),
            (
                "base clearing |mu| line 6-7 ($/MWh)",
                abs(rBASE.line_dual[line]),
                ANCHOR_MU,
                ANCHOR_MU_TOL,
            ),
            (
                "base clearing R, settlement total ($/h)",
                r_settle_base,
                ANCHOR_RENT,
                ANCHOR_RENT_TOL,
            ),
            (
                "base clearing R, |mu| x |F| over the activated lines ($/h)",
                r_lines_base,
                ANCHOR_RENT,
                ANCHOR_RENT_TOL,
            ),
        ],
    )
    assert ns["BINDING_BASE"] == [line]
    assert ns["PAIR"]["base clearing"] == ns["PAIR"]["Market A"] == ["gen_0", "gen_1"]


def test_slide_58_bus7_price_walk_matches(ns: dict[str, Any]) -> None:
    """Slide 58 prints the bus-7 price walk on coefficients rounded to 0.01 MW.

    The slide prints $37.10 as $39.90 less $2.80, the offers times the redispatch
    coefficients rounded to 1.14 MW and 0.14 MW. The rows rebuild those products at
    that rounding, to the cent. The unrounded walk gives the engine's $37.126/MWh,
    which the ring labels $37.1, and its total equals the bus-7 LMP.
    """
    walk, rA1 = ns["WALK_7"], ns["rA1"]
    c_hi, c_lo = ns["C_HI"], ns["C_LO"]
    dg2 = walk.loc["gen_0", "redispatch (MW)"]
    dg3 = walk.loc["gen_1", "redispatch (MW)"]
    assert abs(walk.loc["TOTAL", "cost change ($/h)"] - rA1.lmp["7"]) < EXACT_ABS
    check_slide(
        58,
        [
            ("bus 2 unit up (MW)", dg2, 1.14, HUNDREDTH_TOL),
            ("bus 3 unit down (MW)", -dg3, 0.14, HUNDREDTH_TOL),
            (
                "bus 2 unit cost on the rounded coefficient ($/h)",
                round(dg2, 2) * c_hi,
                39.90,
                HUNDREDTH_TOL,
            ),
            (
                "bus 3 unit saving on the rounded coefficient ($/h)",
                round(-dg3, 2) * c_lo,
                2.80,
                HUNDREDTH_TOL,
            ),
            (
                "bus 7 on the rounded coefficients ($/MWh)",
                round(dg2, 2) * c_hi - round(-dg3, 2) * c_lo,
                37.10,
                HUNDREDTH_TOL,
            ),
            ("bus 7 LMP, ring label ($/MWh)", rA1.lmp["7"], 37.1, TENTH_TOL),
        ],
    )


def test_slide_59_line_6_7_walk_matches(ns: dict[str, Any]) -> None:
    """Slide 59 prints the line 6-7 walk on coefficients rounded to 0.01 MW.

    The slide multiplies the rounded 1.34 MW and prints $46.90 and $26.80, whose
    difference is $20.10/MWh rounded or not. The walk's unrounded total equals |mu|, and
    its bus-2 coefficient equals 1 / (SF_3 - SF_2).
    """
    walk, sf = ns["WALK_L"], ns["SF"]
    c_hi, c_lo, mu = ns["C_HI"], ns["C_LO"], ns["MU"]
    dg2 = walk.loc["gen_0", "redispatch (MW)"]
    dg3 = walk.loc["gen_1", "redispatch (MW)"]
    assert abs(walk["cost change ($/h)"].sum() - mu) < EXACT_ABS
    assert abs(dg2 - 1.0 / (sf["3"] - sf["2"])) < EXACT_ABS
    check_slide(
        59,
        [
            ("bus 2 unit up (MW)", dg2, 1.34, HUNDREDTH_TOL),
            ("bus 3 unit down (MW)", -dg3, 1.34, HUNDREDTH_TOL),
            (
                "bus 2 unit cost on the rounded coefficient ($/h)",
                round(dg2, 2) * c_hi,
                46.90,
                HUNDREDTH_TOL,
            ),
            (
                "bus 3 unit saving on the rounded coefficient ($/h)",
                round(-dg3, 2) * c_lo,
                26.80,
                HUNDREDTH_TOL,
            ),
            ("|mu| line 6-7 ($/MWh)", mu, 20.10, HUNDREDTH_TOL),
        ],
    )


def test_slide_60_market_b_clearing_matches(ns: dict[str, Any]) -> None:
    """Slide 60 prints Market B's clear: its flow on line 6-7 to 0.1 MW, and its injections,
    flows to whole MW and LMPs to $0.1/MWh."""
    rB = ns["rB"]
    check_slide(
        60,
        rows(
            [("Market B flow on line 6-7 (MW)", ns["FB"], 9.4, TENTH_TOL)],
            injection_rows(ns["net_injection"](rB), {"9": -173, "1": 263, "5": -90}),
            flow_rows(
                rB.flow_own,
                {
                    "line_0": 263,
                    "line_8": 164,
                    "line_1": 99,
                    "line_7": 9,
                    "line_5": 9,
                    "line_4": 9,
                    "line_2": 9,
                },
            ),
            price_rows(rB.lmp, dict.fromkeys(ns["pt"].buses, 50.0)),
        ),
    )


def test_slide_61_blind_superposition_matches(ns: dict[str, Any]) -> None:
    """Slide 61 prints the blind superposition, line 6-7 labelled 49/40.

    The line labels are those of ``seam_views.combined_views``, the flow rounded to
    0.1 MW and then to whole MW, half to even. A row holds the superposed flow on line
    6-7 to the 49.4 MW the slide prints, and the figure's table marks the line
    overloaded. The notebook's text states the flow to 0.001 MW, which test_slide_70
    reads from its markdown.
    """
    rA1, rB, fs1 = ns["rA1"], ns["rB"], ns["FS1"]
    line = ns["STUDY_LINE"]
    assert ns["PF1"].loc[line, "overload"]
    check_slide(
        61,
        rows(
            [("F^s line 6-7 (MW)", fs1[line], 49.4, TENTH_TOL)],
            injection_rows(
                ns["net_injection"](rA1, rB),
                {"2": 49, "9": -173, "1": 263, "7": -90, "5": -90, "3": 41},
            ),
            flow_rows(
                fs1,
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
        ),
    )


def test_slide_62_market_a_redispatch_to_market_b_flow_matches(
    ns: dict[str, Any],
) -> None:
    """Slide 62 prints Market A's redispatch to Market B's realized flow.

    Costs and the cost change are to the dollar, the redispatch to 0.01 MW and the
    LMPs to $0.1/MWh. The slide rounds the cost change, $189.89/h, to $190. The change
    equals |mu| times Market B's flow by the notebook's equation (6), the redispatched
    superposition holds line 6-7 at its rating, and every backstop is idle in the blind
    clears and the redispatch, its $55/MWh offer above every LMP there.
    """
    rA1, rA1r, rB = ns["rA1"], ns["rA1r"], ns["rB"]
    line, rate, mu, fb = ns["STUDY_LINE"], ns["RATE"], ns["MU"], ns["FB"]
    c_hi, c_lo = ns["C_HI"], ns["C_LO"]
    dg2 = rA1r.dispatch["gen_0"] - rA1.dispatch["gen_0"]
    dg3 = rA1r.dispatch["gen_1"] - rA1.dispatch["gen_1"]
    assert abs((rA1r.total_cost - rA1.total_cost) - mu * fb) < IDENTITY_ABS
    assert abs(ns["FS1R"][line] - rate) < IDENTITY_ABS
    for r in (rA1, rB, rA1r):
        assert all(
            v < IDENTITY_ABS for g, v in r.dispatch.items() if g.startswith("backstop_")
        )
    lmp_max = max(max(r.lmp.values()) for r in (rA1, rB, rA1r))
    assert lmp_max < ns["BACKSTOP_PRICE"]
    host = {**rB.lmp, **{b: rA1r.lmp[b] for b in ns["MARKETS"]["Market A"]}}
    check_slide(
        62,
        rows(
            [
                ("Market A production cost ($/h)", rA1r.total_cost, 2725, WHOLE_TOL),
                ("independent run ($/h)", rA1.total_cost, 2535, WHOLE_TOL),
                ("bus 2 unit up (MW)", dg2, 12.66, HUNDREDTH_TOL),
                ("bus 3 unit down (MW)", -dg3, 12.66, HUNDREDTH_TOL),
                ("bus 2 unit cost ($/h)", dg2 * c_hi, 443, WHOLE_TOL),
                ("bus 3 unit saving ($/h)", -dg3 * c_lo, 253, WHOLE_TOL),
                (
                    "change in production cost ($/h)",
                    rA1r.total_cost - rA1.total_cost,
                    190,
                    WHOLE_TOL,
                ),
                ("Market B flow on line 6-7 (MW)", fb, 9.4, TENTH_TOL),
                (
                    "|mu| line 6-7 ($/MWh)",
                    abs(rA1r.line_dual[line]),
                    20.10,
                    HUNDREDTH_TOL,
                ),
                ("|mu| x F^B, unrounded ($/h)", mu * fb, 190, WHOLE_TOL),
            ],
            price_rows(
                host, {"9": 50.0, "1": 50.0, "5": 50.0, "2": 35.0, "7": 37.1, "3": 20.0}
            ),
        ),
    )


def test_slide_63_parallel_process_matches(ns: dict[str, Any]) -> None:
    """Slide 63 prints the parallel process: inputs, optimization and reconciliation.

    The inputs column's chips at the neighbor's buses read 0 MW in each blind clear.
    The reconciliation column's second and third pictures are the redispatched
    superposition, labelled as ``combined_views`` labels it (0.1 MW, then whole MW, half
    to even), with its ring to whole MW and its LMPs to $0.1/MWh. The column's first
    picture, the August workshop's 43/40, is not run.
    """
    A1, rA1, B, rB = ns["A1"], ns["rA1"], ns["B"], ns["rB"]
    rA1r = ns["rA1r"]
    host = {**rB.lmp, **{b: rA1r.lmp[b] for b in ns["MARKETS"]["Market A"]}}
    ring = {
        **{b: rA1r.gen_by_bus[b] for b in ("2", "3")},
        "1": rB.gen_by_bus["1"],
        "9": rB.load_by_bus["9"],
        "5": rB.load_by_bus["5"],
        "7": rA1r.load_by_bus["7"],
    }
    check_slide(
        63,
        rows(
            neighbor_rows(ns, "Market A", A1, rA1, {"9": 0, "1": 0, "5": 0}),
            neighbor_rows(ns, "Market B", B, rB, {"2": 0, "7": 0, "3": 0}),
            injection_rows(
                ns["net_injection"](rA1r, rB),
                {"2": 62, "3": 28, "9": -173, "1": 263, "7": -90, "5": -90},
            ),
            flow_rows(
                ns["FS1R"],
                {
                    "line_4": 40,
                    "line_6": 62,
                    "line_5": 50,
                    "line_3": 28,
                    "line_2": 12,
                    "line_8": 161,
                    "line_7": 12,
                    "line_1": 102,
                    "line_0": 263,
                },
                two_step=True,
            ),
            price_rows(
                host, {"9": 50.0, "1": 50.0, "5": 50.0, "2": 35.0, "7": 37.1, "3": 20.0}
            ),
            [
                (f"ring MW bus {b}", ring[b], v, WHOLE_TOL)
                for b, v in {
                    "2": 62,
                    "3": 28,
                    "1": 263,
                    "9": 173,
                    "7": 90,
                    "5": 90,
                }.items()
            ],
        ),
    )


# ─── Example 2 ────────────────────────────────────────────────────────────────


def test_slide_65_example1_tags_over_blind_flow_match(ns: dict[str, Any]) -> None:
    """Slide 65 prints the four tags to 0.1 MW over the blind superposed flow.

    The node and line labels are whole MW, rounded directly. The tags' flows rebuild
    the superposed flow on every line, and each market's tags rebuild its flow
    contribution on line 6-7, by the notebook's equation (8).
    """
    flows_tags, tags, line = ns["FLOWS_TAGS"], ns["TAGS"], ns["STUDY_LINE"]
    resid = (flows_tags["sum of tags"] - flows_tags["F^s"]).abs().max()
    assert resid < EXACT_ABS
    for mkt, r in (("Market A", ns["rA1"]), ("Market B", ns["rB"])):
        imp = sum(flows_tags.loc[line, t.tag_id] for t, _ in tags if t.gca == mkt)
        assert abs(imp - r.flow_own[line]) < EXACT_ABS, mkt
    tag = ns["TAG"]
    check_slide(
        65,
        rows(
            [
                (f"tag {k} (MW)", tag[k].mw, v, TENTH_TOL)
                for k, v in {"B2": 173.1, "A2": 49.1, "B1": 90.0, "A1": 40.9}.items()
            ],
            injection_rows(
                ns["net_injection"](ns["rA1"], ns["rB"]),
                {"2": 49, "9": -173, "1": 263, "7": -90, "5": -90, "3": 41},
            ),
            flow_rows(
                ns["FS1"],
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
            ),
        ),
    )


def tag_slide_rows(
    ns: Mapping[str, Any],
    tid: str,
    mw: float,
    slide_flows: Mapping[str, float],
    slide_inj: Mapping[str, float],
    tdf: float,
    impact: float,
) -> list[tuple[Any, ...]]:
    """One tag slide's rows: the path table's MW, and its panel's labels and box.

    The path table prints the tag's MW to 0.1 MW. The panel's line and node labels are
    whole MW, the box prints the TDF to 0.001 and the impact to 0.1 MW.
    """
    t, sf, line = ns["TAG"][tid], ns["SF"], ns["STUDY_LINE"]
    assert t.path[0].priority == "7-F"
    return [("path table title (MW)", t.mw, mw, TENTH_TOL)] + panel_rows(
        ns["FLOWS_TAGS"][tid],
        {t.source: t.mw, t.sink: -t.mw},
        slide_flows,
        slide_inj,
        {
            "TDF": (sf[t.source] - sf[t.sink], tdf, THOUSANDTH_TOL),
            "impact (MW)": (ns["FLOWS_TAGS"].loc[line, tid], impact, TENTH_TOL),
        },
    )


def test_slide_66_tag_a1_matches(ns: dict[str, Any]) -> None:
    """Slide 66 prints tag A1 (3 to 7) at 40.9 MW and 7-F, with TDF +0.852 and impact +34.8 MW."""
    check_slide(
        66,
        tag_slide_rows(
            ns,
            "A1",
            40.9,
            {
                "line_4": 35,
                "line_3": 41,
                "line_7": 6,
                "line_8": 6,
                "line_5": 6,
                "line_1": 6,
                "line_2": 6,
            },
            {"3": 41, "7": -41},
            0.852,
            34.8,
        ),
    )


def test_slide_67_tag_a2_matches(ns: dict[str, Any]) -> None:
    """Slide 67 prints tag A2 (2 to 7) at 49.1 MW and 7-F, with TDF +0.106 and impact +5.2 MW."""
    check_slide(
        67,
        tag_slide_rows(
            ns,
            "A2",
            49.1,
            {
                "line_6": 49,
                "line_5": 44,
                "line_7": 5,
                "line_8": 5,
                "line_1": 5,
                "line_4": 5,
                "line_2": 5,
            },
            {"2": 49, "7": -49},
            0.106,
            5.2,
        ),
    )


def test_slide_68_tag_b1_matches(ns: dict[str, Any]) -> None:
    """Slide 68 prints tag B1 (1 to 5) at 90.0 MW and 7-F, with TDF -0.135 and impact -12.2 MW."""
    check_slide(
        68,
        tag_slide_rows(
            ns,
            "B1",
            90.0,
            {
                "line_0": 90,
                "line_1": 78,
                "line_7": 12,
                "line_8": 12,
                "line_5": 12,
                "line_4": 12,
                "line_2": 12,
            },
            {"1": 90, "5": -90},
            -0.135,
            -12.2,
        ),
    )


def test_slide_69_tag_b2_matches(ns: dict[str, Any]) -> None:
    """Slide 69 prints tag B2 (1 to 9) at 173.1 MW and 7-F, with TDF +0.125 and impact +21.6 MW."""
    check_slide(
        69,
        tag_slide_rows(
            ns,
            "B2",
            173.1,
            {
                "line_0": 173,
                "line_8": 151,
                "line_7": 22,
                "line_5": 22,
                "line_1": 22,
                "line_4": 22,
                "line_2": 22,
            },
            {"1": 173, "9": -173},
            0.125,
            21.6,
        ),
    )


def test_slide_70_example2_relief_table_and_tags_match(ns: dict[str, Any]) -> None:
    """Slide 70 prints the relief table of Example 2 and the tags over the reconciled flow.

    The table prints the relief to 0.01 MW, the shares to 0.1 percent and every other
    megawatt column to 0.1 MW. The figure labels each tag's kept MW to 0.1 MW and each
    line to whole MW, rounded directly. An arc covers the third digit of the line 1-4
    label, so its row holds the two visible digits, 23. The obligations sum to the
    requirement and B1 takes no relief. The notebook's text for this slide states the
    superposed flow and the requirement to 0.001 MW, and both read as the run prints them
    to that resolution.
    """
    led, ob, fs2 = ns["LED2"], ns["OB2"], ns["FS2"]
    assert abs(sum(ob.values()) - led.attrs["R"]) < EXACT_ABS
    assert float(led.loc[led["tag"] == "B1", "relief"].iloc[0]) == 0.0
    text = markdown_of("### Slide 70")
    for name, mw in (
        ("superposed flow", ns["FS1"][ns["STUDY_LINE"]]),
        ("relief requirement", led.attrs["R"]),
    ):
        assert f"{mw:.3f} MW" in text, (
            f"slide 70 text: {name} {mw:.3f} MW is not stated"
        )
    assert_table(
        70,
        ns["RT2"],
        [
            ["A1 (3→7)", "7-F", "40.9", "34.8", "56.5%", "5.34", "6.3", "34.6"],
            ["A2 (2→7)", "7-F", "49.1", "5.2", "8.4%", "0.80", "7.5", "41.5"],
            ["B2 (1→9)", "7-F", "173.1", "21.6", "35.1%", "3.31", "26.5", "146.5"],
            [
                "B1 (1→5, counterflow)",
                "7-F",
                "90.0",
                "-12.2",
                "not eligible",
                "0.00",
                "0.0",
                "90.0",
            ],
            ["Total", "", "353.0", "49.4", "100.0%", "9.45", "40.3", "312.7"],
        ],
    )
    kept = dict(zip(led["tag"], led["MW"] - led["instructed MW"], strict=True))
    check_slide(
        70,
        rows(
            [
                (f"tag {k} kept (MW)", kept[k], v, TENTH_TOL)
                for k, v in {"B2": 146.5, "A2": 41.5, "A1": 34.6, "B1": 90.0}.items()
            ],
            flow_rows(
                fs2,
                {
                    "line_7": 5,
                    "line_8": 141,
                    "line_6": 42,
                    "line_5": 36,
                    "line_1": 95,
                    "line_4": 40,
                    "line_2": 5,
                    "line_3": 35,
                },
            ),
            [
                (
                    "|F| line_0, label's first two digits",
                    abs(fs2["line_0"]),
                    23,
                    LABEL_DIGITS_TOL,
                    label_two_digits(fs2["line_0"]),
                )
            ],
        ),
    )


def test_slide_71_example2_reconciled_dispatch_matches(ns: dict[str, Any]) -> None:
    """Slide 71 prints Example 2's reconciled dispatch.

    The dispatch and node labels are whole MW, rounded directly. The line labels are
    those of ``combined_views``, the flow rounded to 0.1 MW and then to whole MW, half to
    even, so line 1-4 at 236.534 MW reads 236 beside bus 1's +237 MW. Each bus reads the
    blind clear's price, to $0.1/MWh. Line 6-7 holds at its rating and each backstop
    within its capacity.
    """
    ex2, fs2, disp2 = ns["EX2"], ns["FS2"], ns["DISP2"]
    bk = ns["backstop_mw"](ex2)
    assert abs(fs2[ns["STUDY_LINE"]] - ns["RATE"]) < IDENTITY_ABS
    assert all(bk[b] <= ns["BACKSTOP_MW"][b] + IDENTITY_ABS for b in ns["BACKSTOP_MW"])
    check_slide(
        71,
        rows(
            [
                (f"{g} (MW)", disp2[g], v, WHOLE_TOL)
                for g, v in {
                    "gen_0": 42,
                    "gen_1": 35,
                    "backstop_7": 14,
                    "backstop_9": 27,
                    "gen_slack_0": 237,
                }.items()
            ],
            injection_rows(
                ns["net_injection"](*(r for _, r in ex2)),
                {"2": 42, "3": 35, "7": -76, "9": -147, "1": 237, "5": -90},
            ),
            flow_rows(
                fs2,
                {
                    "line_6": 42,
                    "line_7": 5,
                    "line_8": 141,
                    "line_0": 236,
                    "line_5": 36,
                    "line_1": 95,
                    "line_4": 40,
                    "line_2": 5,
                    "line_3": 35,
                },
                two_step=True,
            ),
            price_rows(
                host_lmp(ns, ns["PAIRS"]),
                {"2": 35.0, "3": 20.0, "7": 37.1, "9": 50.0, "1": 50.0, "5": 50.0},
            ),
        ),
    )


# ─── Example 3 ────────────────────────────────────────────────────────────────


def test_slide_73_example3_tags_over_blind_flow_match(ns: dict[str, Any]) -> None:
    """Slide 73 prints Example 3's three tags to 0.1 MW over the blind superposed flow.

    The node and line labels are whole MW, rounded directly. Market A's untagged
    generation equals its untagged load.
    """
    assert abs(sum(ns["G_UN_A"].values()) - sum(ns["D_UN_A"].values())) < EXACT_ABS
    tag3 = ns["TAG3"]
    check_slide(
        73,
        rows(
            [
                (f"tag {k} (MW)", tag3[k].mw, v, TENTH_TOL)
                for k, v in {"B2": 173.1, "B1": 90.0, "A1": 25.0}.items()
            ],
            injection_rows(
                ns["net_injection"](ns["rA1"], ns["rB"]),
                {"2": 49, "9": -173, "1": 263, "7": -90, "5": -90, "3": 41},
            ),
            flow_rows(
                ns["FS1"],
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
            ),
        ),
    )


def test_slide_74_market_b_tag_panels_match(ns: dict[str, Any]) -> None:
    """Slide 74 prints Market B's two tag panels, B2 and B1.

    Each panel's line and node labels are whole MW, its TDF is to 0.001 and its
    impact to 0.1 MW.
    """
    tag3, sf, flows_tags, line = (
        ns["TAG3"],
        ns["SF"],
        ns["FLOWS_TAGS"],
        ns["STUDY_LINE"],
    )
    rows: list[tuple[Any, ...]] = []
    panels = (
        (
            "B2",
            {
                "line_0": 173,
                "line_8": 151,
                "line_7": 22,
                "line_5": 22,
                "line_1": 22,
                "line_4": 22,
                "line_2": 22,
            },
            {"1": 173, "9": -173},
            0.125,
            21.6,
        ),
        (
            "B1",
            {
                "line_0": 90,
                "line_1": 78,
                "line_7": 12,
                "line_8": 12,
                "line_5": 12,
                "line_4": 12,
                "line_2": 12,
            },
            {"1": 90, "5": -90},
            -0.135,
            -12.2,
        ),
    )
    for tid, fl, inj, tdf, imp in panels:
        t = tag3[tid]
        rows += prefixed(
            tid,
            panel_rows(
                flows_tags[tid],
                {t.source: t.mw, t.sink: -t.mw},
                fl,
                inj,
                {
                    "TDF": (sf[t.source] - sf[t.sink], tdf, THOUSANDTH_TOL),
                    "impact (MW)": (flows_tags.loc[line, tid], imp, TENTH_TOL),
                },
            ),
        )
    check_slide(74, rows)


def test_slide_75_tag_a1_and_untagged_generation_match(ns: dict[str, Any]) -> None:
    """Slide 75 prints tag A1 at 25 MW and Market A's untagged generation-to-load.

    Each panel's line and node labels are whole MW. A1's box prints its TDF to 0.001
    and its impact to 0.1 MW. The GTL box prints the impact to 0.1 MW and the impact
    per injected MW to 0.001. The GTL impact equals the flow its net injections place on
    line 6-7.
    """
    pt, sf, line = ns["pt"], ns["SF"], ns["STUDY_LINE"]
    x_a, i_gtl, gtl_a = ns["X_A"], ns["I_GTL"], ns["GTL_A"]
    a1_3, flows_a1_3 = ns["A1_3"], ns["FLOWS_A1_3"]
    assert abs(i_gtl - sum(sf[b] * v for b, v in x_a.items())) < EXACT_ABS
    inj = np.zeros(pt.n_bus)
    for b, v in x_a.items():
        inj[pt.bus_idx[b]] = v
    flows_gtl = {m: float(pt.ptdf[pt.line_idx[m]] @ inj) for m in pt.lines}
    check_slide(
        75,
        rows(
            prefixed(
                "A1",
                panel_rows(
                    flows_a1_3,
                    {"3": a1_3.mw, "7": -a1_3.mw},
                    {
                        "line_4": 21,
                        "line_3": 25,
                        "line_7": 4,
                        "line_8": 4,
                        "line_5": 4,
                        "line_1": 4,
                        "line_2": 4,
                    },
                    {"3": 25, "7": -25},
                    {
                        "TDF": (sf["3"] - sf["7"], 0.852, THOUSANDTH_TOL),
                        "impact (MW)": (flows_a1_3[line], 21.3, TENTH_TOL),
                    },
                ),
            ),
            prefixed(
                "GTL",
                panel_rows(
                    flows_gtl,
                    x_a,
                    {
                        "line_6": 49,
                        "line_5": 46,
                        "line_4": 19,
                        "line_3": 16,
                        "line_7": 3,
                        "line_8": 3,
                        "line_1": 3,
                        "line_2": 3,
                    },
                    {"2": 49, "3": 16, "7": -65},
                    {
                        "impact (MW)": (i_gtl, 18.7, TENTH_TOL),
                        "impact / injected MW": (gtl_a["tdf"], 0.288, THOUSANDTH_TOL),
                    },
                ),
            ),
        ),
    )


def test_slide_76_example3_relief_table_and_tags_match(ns: dict[str, Any]) -> None:
    """Slide 76 prints the relief table of Example 3 and the tags over the reconciled flow.

    The table prints the relief to 0.01 MW, the shares to 0.1 percent and every other
    megawatt column to 0.1 MW. The figure labels each tag's kept MW and each bus's net
    injection to 0.1 MW, and each line to whole MW, rounded directly. An arc covers the
    third digit of the line 1-4 label, so its row holds the two visible digits, 23.
    Each market's obligation equals Example 2's.
    """
    led, fs3 = ns["LED3"], ns["FS3"]
    ob2, ob3 = ns["OB2"], ns["OB3"]
    assert all(abs(ob3[k] - ob2[k]) < EXACT_ABS for k in ns["MARKETS"])
    assert_table(
        76,
        ns["RT3"],
        [
            ["A1 (3→7)", "7-F", "25.0", "21.3", "34.6%", "3.27", "3.8", "21.2"],
            [
                "A GTL (net injections), untagged",
                "7-F",
                "64.9",
                "18.7",
                "30.4%",
                "2.87",
                "10.0",
                "55.0",
            ],
            ["gen at bus 2", "", "49.1", "5.2", "8.4%", "0.80", "7.5", "41.5"],
            ["gen at bus 3", "", "15.9", "13.5", "21.9%", "2.07", "2.4", "13.4"],
            ["B2 (1→9)", "7-F", "173.1", "21.6", "35.1%", "3.31", "26.5", "146.5"],
            [
                "B1 (1→5, counterflow)",
                "7-F",
                "90.0",
                "-12.2",
                "not eligible",
                "0.00",
                "0.0",
                "90.0",
            ],
            ["Total", "", "353.0", "49.4", "100.0%", "9.45", "40.3", "312.7"],
        ],
    )
    kept = dict(zip(led["tag"], led["MW"] - led["instructed MW"], strict=True))
    net3 = ns["NET3"]
    injections = {"2": 41.5, "9": -146.5, "1": 236.5, "7": -76.1, "5": -90.0, "3": 34.6}
    check_slide(
        76,
        rows(
            [
                (f"tag {k} kept (MW)", kept[k], v, TENTH_TOL)
                for k, v in {"B2": 146.5, "B1": 90.0, "A1": 21.2}.items()
            ],
            [
                (f"net injection bus {b} (MW)", net3[b], v, TENTH_TOL)
                for b, v in injections.items()
            ],
            flow_rows(
                fs3,
                {
                    "line_7": 5,
                    "line_8": 141,
                    "line_6": 42,
                    "line_5": 36,
                    "line_1": 95,
                    "line_4": 40,
                    "line_2": 5,
                    "line_3": 35,
                },
            ),
            [
                (
                    "|F| line_0, label's first two digits",
                    abs(fs3["line_0"]),
                    23,
                    LABEL_DIGITS_TOL,
                    label_two_digits(fs3["line_0"]),
                )
            ],
        ),
    )


def test_slide_77_example3_reconciled_dispatch_matches(ns: dict[str, Any]) -> None:
    """Slide 77 prints Example 3's reconciled dispatch, which repeats slide 71's numbers.

    The roundings are slide 71's: the dispatch and node labels whole MW, rounded
    directly, the line labels those of ``combined_views``, and the prices to $0.1/MWh.
    The dispatch equals Example 2's unit by unit, and line 6-7 holds at its rating.
    """
    disp2, disp3, fs3 = ns["DISP2"], ns["DISP3"], ns["FS3"]
    assert disp2.keys() == disp3.keys()
    assert all(abs(disp2[g] - disp3[g]) < EXACT_ABS for g in disp2)
    assert abs(fs3[ns["STUDY_LINE"]] - ns["RATE"]) < IDENTITY_ABS
    check_slide(
        77,
        rows(
            [
                (f"{g} (MW)", disp3[g], v, WHOLE_TOL)
                for g, v in {
                    "gen_0": 42,
                    "gen_1": 35,
                    "backstop_7": 14,
                    "backstop_9": 27,
                    "gen_slack_0": 237,
                }.items()
            ],
            injection_rows(
                ns["NET3"], {"2": 42, "3": 35, "7": -76, "9": -147, "1": 237, "5": -90}
            ),
            flow_rows(
                fs3,
                {
                    "line_6": 42,
                    "line_7": 5,
                    "line_8": 141,
                    "line_0": 236,
                    "line_5": 36,
                    "line_1": 95,
                    "line_4": 40,
                    "line_2": 5,
                    "line_3": 35,
                },
                two_step=True,
            ),
            price_rows(
                host_lmp(ns, ns["PAIRS"]),
                {"2": 35.0, "3": 20.0, "7": 37.1, "9": 50.0, "1": 50.0, "5": 50.0},
            ),
        ),
    )


# ─── Example 4 ────────────────────────────────────────────────────────────────


def test_slide_80_market_a_flow_limit_table_matches(ns: dict[str, Any]) -> None:
    """Slide 80 prints Market A's market flow limit table and its clear's inputs.

    The table prints each row's relief to 0.01 MW and the totals and the limit to
    0.1 MW. The limit equals the notebook's equation (16). The inputs frame's chips at
    Market B's buses read 0 MW.
    """
    lim_a, f0, ob3, s_m = ns["LIM_A"], ns["F0"], ns["OB3"], ns["S_M"]
    assert abs(lim_a - (s_m * f0["Market A"] - ob3["Market A"])) < EXACT_ABS
    assert_table(
        80,
        ns["LT_A"],
        [
            ["A1 (3→7)", "7-F", "25.0", "21.3", "3.27"],
            ["A GTL (net injections), untagged", "7-F", "64.9", "18.7", "2.87"],
            ["Total", "", "89.9", "40.0", "6.1"],
            ["Market Flow Limit", "", "", "", "33.9"],
        ],
    )
    check_slide(
        80,
        neighbor_rows(ns, "Market A", ns["A4"], ns["rA4"], {"9": 0, "1": 0, "5": 0}),
    )


def test_slide_81_market_b_flow_limit_table_matches(ns: dict[str, Any]) -> None:
    """Slide 81 prints Market B's market flow limit table and its clear's inputs.

    The table prints each row's relief to 0.01 MW and the totals and the limit to
    0.1 MW. The limit equals the notebook's equation (16), and the two limits sum to
    line 6-7's rating. The inputs frame's chips at Market A's buses read 0 MW.
    """
    lim_a, lim_b, f0, ob3, s_m = (
        ns["LIM_A"],
        ns["LIM_B"],
        ns["F0"],
        ns["OB3"],
        ns["S_M"],
    )
    assert abs(lim_b - (s_m * f0["Market B"] - ob3["Market B"])) < EXACT_ABS
    assert abs(lim_a + lim_b - ns["RATE"]) < EXACT_ABS
    assert_table(
        81,
        ns["LT_B"],
        [
            ["B2 (1→9)", "7-F", "173.1", "21.6", "3.31"],
            ["B1 (1→5, counterflow)", "7-F", "90.0", "-12.2", "0.00"],
            ["Total", "", "263.1", "9.4", "3.3"],
            ["Market Flow Limit", "", "", "", "6.1"],
        ],
    )
    check_slide(
        81,
        neighbor_rows(ns, "Market B", ns["B4"], ns["rB4"], {"2": 0, "7": 0, "3": 0}),
    )


def test_slide_82_market_a_clear_against_limit_matches(ns: dict[str, Any]) -> None:
    """Slide 82 prints Market A's clear against its limit.

    The limit label, injections and flows are whole MW and the LMPs $0.1/MWh. The
    slide prints no dual, and the row holds it to slide 59's $20.10/MWh. Market A's
    flow on line 6-7 sits at its limit.
    """
    rA4, lim_a, line = ns["rA4"], ns["LIM_A"], ns["STUDY_LINE"]
    assert abs(rA4.flow_own[line] - lim_a) < IDENTITY_ABS
    check_slide(
        82,
        rows(
            [
                (
                    "|mu| line 6-7, slide 59's value ($/MWh)",
                    abs(rA4.line_dual[line]),
                    20.10,
                    HUNDREDTH_TOL,
                ),
                ("line 6-7 limit label (MW)", lim_a, 34, WHOLE_TOL),
            ],
            injection_rows(ns["net_injection"](rA4), {"2": 57, "3": 33, "7": -90}),
            flow_rows(
                rA4.flow_own,
                {
                    "line_6": 57,
                    "line_5": 56,
                    "line_4": 34,
                    "line_3": 33,
                    "line_7": 1,
                    "line_8": 1,
                    "line_1": 1,
                    "line_2": 1,
                },
            ),
            price_rows(
                rA4.lmp,
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
        ),
    )


def test_slide_83_market_b_clear_against_limit_matches(ns: dict[str, Any]) -> None:
    """Slide 83 prints Market B's clear against its limit.

    The limit label, the backstop, injections and flows are whole MW and the LMPs
    $0.1/MWh. Market B's flow on line 6-7 sits at its limit. The slide prints no dual.
    The bus-1 unit and the bus-9 backstop are both interior, so the line walk over that
    pair gives |mu| as the offer difference over the shift-factor difference, the form
    test_slide_59 holds for the bus-2 and bus-3 units. The notebook's text states |mu|
    to the cent, and it reads as the run prints it to that resolution.
    """
    rB4, lim_b, line = ns["rB4"], ns["LIM_B"], ns["STUDY_LINE"]
    fleet, sf = ns["FLEET"], ns["SF"]
    assert abs(rB4.flow_own[line] - lim_b) < IDENTITY_ABS
    for g in ("gen_slack_0", "backstop_9"):
        assert IDENTITY_ABS < rB4.dispatch[g] < fleet[g]["p_nom"] - IDENTITY_ABS, g
    mu = abs(rB4.line_dual[line])
    walk = (fleet["backstop_9"]["cost"] - fleet["gen_slack_0"]["cost"]) / abs(
        sf[fleet["backstop_9"]["bus"]] - sf[fleet["gen_slack_0"]["bus"]]
    )
    assert abs(mu - walk) < IDENTITY_ABS
    assert f"{mu:.2f}$/MWh" in markdown_of("### Slide 83"), (
        f"slide 83 text: |mu| ${mu:.2f}/MWh is not stated"
    )
    check_slide(
        83,
        rows(
            [
                ("line 6-7 limit label (MW)", lim_b, 6, WHOLE_TOL),
                ("backstop_9 (MW)", rB4.dispatch["backstop_9"], 27, WHOLE_TOL),
            ],
            injection_rows(ns["net_injection"](rB4), {"1": 237, "9": -147, "5": -90}),
            flow_rows(
                rB4.flow_own,
                {
                    "line_0": 237,
                    "line_8": 140,
                    "line_1": 96,
                    "line_7": 6,
                    "line_5": 6,
                    "line_4": 6,
                    "line_2": 6,
                },
            ),
            price_rows(
                rB4.lmp,
                {
                    "9": 55.0,
                    "1": 50.0,
                    "5": 44.6,
                    "2": 64.5,
                    "8": 64.5,
                    "7": 68.7,
                    "6": 34.6,
                    "3": 34.6,
                },
            ),
        ),
    )


def test_slide_84_capped_clears_superposition_matches(ns: dict[str, Any]) -> None:
    """Slide 84 prints the superposition of the two capped clears.

    The node labels are whole MW, rounded directly, and the line labels those of
    ``combined_views``, so line 1-4 at 236.5 MW reads 236. Each bus reads its own
    market's LMP, to $0.1/MWh. Line 6-7 holds at its rating.
    """
    rA4, rB4, fs4 = ns["rA4"], ns["rB4"], ns["FS4"]
    assert abs(fs4[ns["STUDY_LINE"]] - ns["RATE"]) < SUPERPOSITION_ABS
    check_slide(
        84,
        rows(
            injection_rows(
                ns["net_injection"](rA4, rB4),
                {"2": 57, "3": 33, "1": 237, "9": -147, "7": -90, "5": -90},
            ),
            flow_rows(
                fs4,
                {
                    "line_6": 57,
                    "line_7": 7,
                    "line_8": 139,
                    "line_0": 236,
                    "line_5": 50,
                    "line_1": 97,
                    "line_4": 40,
                    "line_2": 7,
                    "line_3": 33,
                },
                two_step=True,
            ),
            price_rows(
                host_lmp(ns, ns["PAIRS4"]),
                {"9": 55.0, "2": 35.0, "1": 50.0, "7": 37.1, "5": 44.6, "3": 20.0},
            ),
        ),
    )
