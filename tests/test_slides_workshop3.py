"""Slide checks for ``workshop3_transactions_across_boundaries.ipynb``.

The notebook clears the four examples of the September 16, 2026 deck *Transactions
across market boundaries* (``caiso_spp_workshop_sep2026``), numbered 1 to 34 from the
deck's own title slide. The module fixture runs the notebook's code cells once, and
each test compares named results in that namespace with the numbers one slide prints
(participant direction 2026-11-08).

A ring or network label is compared at the figure's rounding: a net injection and a
line's own flow to a whole MW, a nodal price to $0.1/MWh, each rounded once from the
run. A row then holds only where the label and the slide agree. Where a slide rounds
its inputs before it multiplies, the test rebuilds the slide's number from the run's
values at the slide's rounding.

The run's unrounded figures are checked against the notebook's formulation, not against
typed values. Each clearing meets its power balance (1), its price identity (3), the
objective of (1) or (11) at its dispatch, and the marginal-cost condition of each unit
inside its bounds; each price walk meets (5) and (6); each settlement meets (8), (10) or
(12) at the cleared prices; and each market's change in production cost equals its
amount on the transaction, negated. A figure the notebook's text states is read from
its markdown and compared at the digits the text prints.

A typed number is a slide's printed value, an input, a tolerance, or the series anchor.

    python -m pytest tests/test_slides_workshop3.py -q
    python -m pytest tests -q -m slides
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import nbformat
import pytest

from slide_check import ROOT, Row, as_row, check_slide, run_notebook

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from seams_engine import EngineResult

pytestmark = pytest.mark.slides

NOTEBOOK = "workshop3_transactions_across_boundaries.ipynb"

#: Absolute tolerance of an identity of a linear clearing, MW, $/MWh or $/h, at the LP
#: solver's precision.
IDENTITY_ABS = 1e-6
#: Absolute tolerance of an identity of a rising-curve clearing or of a bisected award,
#: MW, $/MWh or $/h: the QP tolerance the notebook's clearing holds its shed to.
QP_ABS = 1e-3
#: Tolerance of a slide value that restates an input exactly, in the input's unit.
EXACT = 0.0
#: Half a unit of a whole-MW label, MW, absolute.
WHOLE_MW_TOL = 0.5
#: Half a unit of a $0.1/MWh price label, $/MWh, absolute.
TENTH_PRICE_TOL = 0.05
#: The series anchor, the canonical clearing: line 6-7 at its 40 MW rating with
#: |mu| = $20.10/MWh and R = $804.1/h, each to half a unit of its last digit, absolute.
ANCHOR_FLOW, ANCHOR_FLOW_TOL = 40.0, 0.05
ANCHOR_MU, ANCHOR_MU_TOL = 20.10, 0.005
ANCHOR_RENT, ANCHOR_RENT_TOL = 804.1, 0.05
#: The study line, line 6-7, as the notebook's EDIT cell names it.
STUDY_LINE = "line_4"


@pytest.fixture(scope="module")
def nb() -> dict[str, Any]:
    """The namespace of one run of the notebook's code cells."""
    return run_notebook(NOTEBOOK)


@pytest.fixture(scope="module")
def text() -> str:
    """The notebook's markdown cells joined, the text whose stated figures are read."""
    book = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    return "\n\n".join(c.source for c in book.cells if c.cell_type == "markdown")


def label(value: float, digits: int) -> float:
    """``value`` as a figure or a slide prints it, at ``digits`` decimals."""
    return float(f"{value:.{digits}f}")


def half_unit(digits: int) -> float:
    """Half a unit of the last printed digit at ``digits`` decimals, absolute."""
    return 0.5 * 10.0**-digits


def label_row(quantity: str, value: float, slide: float, digits: int) -> Row:
    """A row comparing ``value``'s printed label at ``digits`` decimals with the slide's."""
    return as_row((quantity, value, slide, half_unit(digits), label(value, digits)))


def stated(text: str, pattern: str) -> tuple[str, ...]:
    """The figures one passage of the notebook's text states, as printed.

    ``pattern`` must match the text once; each group is one figure.
    """
    found = re.findall(pattern, text)
    assert len(found) == 1, (
        f"{pattern!r} matches the notebook's text {len(found)} times"
    )
    first = found[0]
    return first if isinstance(first, tuple) else (first,)


def text_row(quantity: str, value: float, figure: str) -> Row:
    """A row comparing ``value`` with a figure of the notebook's text, at its printed digits.

    ``figure`` is the text's string, with its dollar sign, Unicode minus and thousands
    separators as printed.
    """
    plain = figure.replace("\\$", "").replace("$", "").replace("−", "-")
    plain = plain.replace(",", "")
    return label_row(quantity, value, float(plain), len(plain.partition(".")[2]))


def assert_residuals(
    slide: int | str, residuals: Mapping[str, float], tol: float
) -> None:
    """Every residual of one slide's identities lies within ``tol``, absolute.

    The assertion names each residual beyond ``tol`` with its value.
    """
    bad = {q: f"{r:+.3e}" for q, r in residuals.items() if not abs(r) <= tol}
    assert not bad, f"slide {slide}: residual beyond {tol:g}: {bad}"


def clearing_residuals(
    nb: Mapping[str, Any],
    mkt: str,
    res: EngineResult,
    *,
    exo: float = 0.0,
    curves: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """The residuals of market ``mkt``'s clearing ``res`` against its formulation.

    ``exo`` is the signed MW the clearing carries at its transaction node, positive
    into the network; ``curves`` maps a unit to ``(a, b)`` of (11), and without it the
    offers are the flat offers of the fleet. The residuals are the power balance of (1),
    the price identity (3) at every bus, the objective at the dispatch, and the
    marginal-cost condition ``MC_i(g_i) = lambda_bus(i)`` of each unit inside its
    bounds, of which there is at least one.
    """
    pt, fleet = nb["PT"], nb["FLEET"]
    load = sum(v for b, v in nb["LOADS"].items() if b in nb["MKT_DEFS"][mkt])
    out = {"balance (1) (MW)": sum(res.dispatch.values()) - load + exo}
    for n in pt.buses:
        lam = res.energy_price + sum(
            pt.ptdf[pt.line_idx[m], pt.bus_idx[n]] * mu
            for m, mu in res.line_dual.items()
        )
        out[f"price (3) bus {n} ($/MWh)"] = float(res.lmp[n]) - lam
    cost, interior = 0.0, 0
    for g, x in res.dispatch.items():
        a, b = curves[g] if curves else (float(fleet[g]["cost"]), 0.0)
        cost += a * x + 0.5 * b * x * x
        if QP_ABS < x < float(fleet[g]["p_nom"]) - QP_ABS:
            interior += 1
            bus = str(fleet[g]["bus"])
            out[f"marginal cost {g} bus {bus} ($/MWh)"] = (
                a + b * x - float(res.lmp[bus])
            )
    assert interior, f"{mkt}: no unit inside its bounds"
    out["objective at the dispatch ($/h)"] = res.total_cost - cost
    return out


def rating_residual(nb: Mapping[str, Any], res: EngineResult) -> dict[str, float]:
    """The study line's own flow in ``res`` less its rating, MW: zero where it binds."""
    return {
        "|F| line 6-7 at its rating (MW)": abs(float(res.flow_own[STUDY_LINE]))
        - float(nb["LINE4_RATING"])
    }


def walk_residuals(
    nb: Mapping[str, Any], res: EngineResult, walk: pd.DataFrame, bus: str
) -> dict[str, float]:
    """The residuals of one price walk of ``res`` at ``bus`` against (5) and (6).

    The walk delivers one MW at ``bus``, as load where ``WALK_DIRECTION`` reads
    withdrawal and as generation otherwise, so its right-hand sides and its price carry
    the sign ``+1`` or ``-1``.
    """
    pt, fleet = nb["PT"], nb["FLEET"]
    sign = 1.0 if nb["WALK_DIRECTION"][bus] == "withdrawal" else -1.0
    units = walk.drop(index="TOTAL")
    dg = units["redispatch (MW)"]
    out = {"(5) the MW delivered": float(dg.sum()) - sign}
    for m, mu in res.line_dual.items():
        if abs(mu) > IDENTITY_ABS:
            sf = pt.ptdf[pt.line_idx[m]]
            placed = sum(
                sf[pt.bus_idx[str(fleet[g]["bus"])]] * dg[g] for g in units.index
            )
            out[f"(5) flow on {m} (MW)"] = float(placed - sign * sf[pt.bus_idx[bus]])
    for g in units.index:
        cost = float(units.loc[g, "cost change ($/h)"])
        out[f"(6) cost change {g} ($/h)"] = cost - float(fleet[g]["cost"]) * float(
            dg[g]
        )
    total = float(walk.loc["TOTAL", "cost change ($/h)"])
    out["(6) the price ($/MWh)"] = total - sign * float(res.lmp[bus])
    return out


def cost_change(res: EngineResult, base: EngineResult) -> float:
    """The change in production cost from ``base`` to ``res``, $/h."""
    return res.total_cost - base.total_cost


def letter_rows(
    nb: Mapping[str, Any],
    res: EngineResult,
    inj: Mapping[str, float],
    flows: Mapping[str, float],
    prices: Mapping[str, float],
) -> list[Row]:
    """Rows comparing one market's letter with a slide's node, line and ring labels.

    ``inj`` maps a bus to the slide's net injection in MW, positive into the network;
    ``flows`` maps a line's bus pair ``"from-to"`` to the slide's own flow
    ``|F^M_m|`` in MW; ``prices`` maps a bus to the slide's nodal price in $/MWh. The
    compared value is the run's label, a whole MW or $0.1/MWh, and the tolerance half
    a unit of that digit.
    """
    pt = nb["PT"]
    line_of = {f"{b0}-{b1}": m for m, (b0, b1) in zip(pt.lines, pt.line_buses)}
    rows: list[Row] = []
    for b, v in inj.items():
        x = float(res.injection[pt.bus_idx[b]])
        rows.append(
            as_row((f"net injection bus {b} (MW)", x, v, WHOLE_MW_TOL, label(x, 0)))
        )
    for pair, v in flows.items():
        x = abs(float(res.flow_own[line_of[pair]]))
        rows.append(as_row((f"|F| line {pair} (MW)", x, v, WHOLE_MW_TOL, label(x, 0))))
    for b, v in prices.items():
        x = float(res.lmp[b])
        rows.append(
            as_row((f"LMP bus {b} ($/MWh)", x, v, TENTH_PRICE_TOL, label(x, 1)))
        )
    return rows


def pair_rows(nb: Mapping[str, Any]) -> list[Row]:
    """Slides 13 and 15: the flow the balanced pair of (7) places on each line, MW.

    The flow is ``S (SF_{s,m} - SF_{k,m})`` by (2), with ``PT.ptdf`` stored line first;
    each slide labels its absolute value to a whole MW.
    """
    pt, s, src, snk = nb["PT"], nb["S_SCHED"], nb["SOURCE_BUS"], nb["SINK_BUS"]
    slide = {
        "1-4": 25,
        "8-9": 13,
        "9-4": 13,
        "7-8": 13,
        "4-5": 12,
        "5-6": 12,
        "6-7": 12,
    }
    rows: list[Row] = []
    for m, (b0, b1) in zip(pt.lines, pt.line_buses):
        pair = f"{b0}-{b1}"
        if pair in slide:
            sf = pt.ptdf[pt.line_idx[m]]
            x = abs(s * (sf[pt.bus_idx[src]] - sf[pt.bus_idx[snk]]))
            rows.append(label_row(f"|pair flow| line {pair} (MW)", x, slide[pair], 0))
    assert len(rows) == len(slide)
    return rows


def test_canonical_anchor_line_6_7_binds_at_rating_with_rent(
    nb: dict[str, Any],
) -> None:
    """The canonical clearing: line 6-7 at its 40 MW rating, |mu| = $20.10/MWh, R = $804.1/h.

    One engine over all nine buses, the stock fleet and loads, line 6-7 rated 40 MW.
    """
    check_slide(
        "anchor",
        [
            (
                "|F| line 6-7 (MW)",
                abs(float(nb["_res0"].flow_own[STUDY_LINE])),
                ANCHOR_FLOW,
                ANCHOR_FLOW_TOL,
            ),
            ("|mu| line 6-7 ($/MWh)", abs(nb["MU_CANON"]), ANCHOR_MU, ANCHOR_MU_TOL),
            ("R ($/h)", nb["R_CANON"], ANCHOR_RENT, ANCHOR_RENT_TOL),
        ],
    )


def test_slide_03_case_counts_match(nb: dict[str, Any]) -> None:
    """Slide 3: the WSCC 9-bus case, three generators, nine lines and three loads."""
    check_slide(
        3,
        [
            ("generators", len(nb["FLEET"]), 3, EXACT),
            ("lines", len(nb["PT"].lines), 9, EXACT),
            ("loads", len(nb["LOADS"]), 3, EXACT),
        ],
    )


def test_slide_04_ratings_offers_and_loads_match(nb: dict[str, Any]) -> None:
    """Slide 4: the nine line ratings, three offers and the ring's loads.

    The slide prints each rating in whole MW and each offer as $/MWh by MW. Its ring
    draws the stock loads of the 9-bus case, 125 MW at bus 9 and 100 MW at bus 7,
    where every clearing on the deck serves 173 MW and 90 MW; bus 5's 90 MW agrees.
    """
    pt, fleet = nb["PT"], nb["FLEET"]
    ratings = {
        "8-2": 250,
        "8-9": 250,
        "9-4": 250,
        "7-8": 250,
        "4-5": 250,
        "1-4": 300,
        "3-6": 300,
        "5-6": 150,
        "6-7": 40,
    }
    offers = {"2": (35, 300), "1": (50, 300), "3": (20, 170)}  # $/MWh, MW
    rows: list[Row] = []
    for m, (b0, b1) in zip(pt.lines, pt.line_buses):
        pair = f"{b0}-{b1}"
        rows.append(
            label_row(f"rating {pair} (MW)", pt.s_nom[pt.line_idx[m]], ratings[pair], 0)
        )
    by_bus = {str(s["bus"]): s for s in fleet.values()}
    assert set(by_bus) == set(offers)
    for b, (cost, cap) in offers.items():
        rows.append(as_row((f"offer bus {b} ($/MWh)", by_bus[b]["cost"], cost, EXACT)))
        rows.append(as_row((f"capacity bus {b} (MW)", by_bus[b]["p_nom"], cap, EXACT)))
    stock = nb["wm"].DEFAULT_LOADS
    rows += [
        as_row(("load bus 5 (MW)", nb["LOADS"]["5"], 90, EXACT)),
        as_row(("ring load bus 9, stock (MW)", stock["9"], 125, EXACT)),
        as_row(("ring load bus 7, stock (MW)", stock["7"], 100, EXACT)),
    ]
    check_slide(4, rows)


def test_slide_06_market_a_clearing_matches(nb: dict[str, Any], text: str) -> None:
    """Slide 6: Market A's clearing, line 6-7 binding at 40 MW.

    The slide prints 3 injections and 8 flows in whole MW and 7 prices at $0.1/MWh.
    Market A's own clearing reproduces the canonical anchor's dual and rent, and its
    two units are both marginal, so its prices at buses 2 and 3 are their offers.
    """
    res = nb["res_A1"]
    check_slide(
        6,
        letter_rows(
            nb,
            res,
            {"2": 49, "3": 41, "7": -90},
            {
                "8-2": 49,
                "7-8": 50,
                "6-7": 40,
                "3-6": 41,
                "8-9": 1,
                "9-4": 1,
                "4-5": 1,
                "5-6": 1,
            },
            {
                "9": 30.2,
                "4": 27.7,
                "1": 27.7,
                "5": 25.0,
                "2": 35.0,
                "3": 20.0,
                "7": 37.1,
            },
        ),
    )
    p3, p7 = stated(
        text, r"from \\\$([\d.]+)/MWh at bus 3 to \\\$([\d.]+)/MWh at bus 7"
    )
    check_slide(
        "6, the notebook's text",
        [
            text_row("LMP bus 3 ($/MWh)", res.lmp["3"], p3),
            text_row("LMP bus 7 ($/MWh)", res.lmp["7"], p7),
        ],
    )
    residuals = clearing_residuals(nb, "Market A", res)
    marginal = [k for k in residuals if k.startswith("marginal cost")]
    assert len(marginal) == len(res.dispatch), f"not every unit is marginal: {marginal}"
    assert_residuals(
        6,
        {
            **residuals,
            **rating_residual(nb, res),
            "R^A less R ($/h)": nb["R_A1"] - nb["R_CANON"],
            "mu^A less mu ($/MWh)": res.line_dual[STUDY_LINE] - nb["MU_CANON"],
        },
        IDENTITY_ABS,
    )


def test_slide_07_market_a_bus7_price_walk_matches(nb: dict[str, Any]) -> None:
    """Slide 7: Market A's price at bus 7 as the cost of one more MW of load there.

    The slide rounds the redispatch to 1.14 MW and 0.14 MW before it multiplies, so it
    prints $39.90, $2.80 and $37.10. The run's unrounded walk is checked against (5)
    and (6), its total against Market A's price at bus 7.
    """
    w7 = nb["walks_A"]["7"]
    fleet = nb["FLEET"]
    mw = "redispatch (MW)"
    mw_2, mw_3 = round(w7.loc["gen_0", mw], 2), round(-w7.loc["gen_1", mw], 2)
    c_2, c_3 = fleet["gen_0"]["cost"], fleet["gen_1"]["cost"]
    check_slide(
        7,
        [
            label_row("gen_0 bus 2 redispatch (MW)", w7.loc["gen_0", mw], 1.14, 2),
            label_row("gen_1 bus 3 redispatch (MW)", w7.loc["gen_1", mw], -0.14, 2),
            label_row("bus 2 cost at the slide's MW ($/h)", mw_2 * c_2, 39.90, 2),
            label_row("bus 3 cost at the slide's MW ($/h)", mw_3 * c_3, 2.80, 2),
            label_row(
                "price at the slide's MW ($/MWh)", mw_2 * c_2 - mw_3 * c_3, 37.10, 2
            ),
        ],
    )
    assert_residuals(7, walk_residuals(nb, nb["res_A1"], w7, "7"), IDENTITY_ABS)


def test_slide_09_market_a_bus1_price_walk_matches(nb: dict[str, Any]) -> None:
    """Slide 9: Market A's price at bus 1 as the value of one more MW of generation.

    The slide prints the redispatch at two decimals and each dollar amount from the
    four-decimal redispatch, so its $18.05, $9.69 and $27.74 are the run's.
    """
    w1 = nb["walks_A"]["1"]
    mw, usd = "redispatch (MW)", "cost change ($/h)"
    check_slide(
        9,
        [
            label_row("gen_0 bus 2 redispatch (MW)", w1.loc["gen_0", mw], -0.52, 2),
            label_row("gen_1 bus 3 redispatch (MW)", w1.loc["gen_1", mw], -0.48, 2),
            label_row("gen_0 bus 2 cost change ($/h)", w1.loc["gen_0", usd], -18.05, 2),
            label_row("gen_1 bus 3 cost change ($/h)", w1.loc["gen_1", usd], -9.69, 2),
            label_row(
                "change in production cost ($/h)", w1.loc["TOTAL", usd], -27.74, 2
            ),
        ],
    )
    assert_residuals(9, walk_residuals(nb, nb["res_A1"], w1, "1"), IDENTITY_ABS)


def test_slide_10_market_b_clearing_matches(nb: dict[str, Any]) -> None:
    """Slide 10: Market B's clearing, its unit at bus 1 setting $50/MWh everywhere.

    The slide prints 3 injections and 7 flows in whole MW and 9 prices at $0.1/MWh,
    and Market B's own flow on line 6-7, 9.4 MW, which with Market A's 40 MW makes
    49.4 MW; both at one decimal. Market B does not activate line 6-7.
    """
    res, pt = nb["res_B1"], nb["PT"]
    assert STUDY_LINE not in nb["ACTIVATED_LINES"]["Market B"]
    check_slide(
        10,
        [
            *letter_rows(
                nb,
                res,
                {"1": 263, "9": -173, "5": -90},
                {
                    "1-4": 263,
                    "9-4": 164,
                    "4-5": 99,
                    "8-9": 9,
                    "7-8": 9,
                    "6-7": 9,
                    "5-6": 9,
                },
                dict.fromkeys(pt.buses, 50.0),
            ),
            label_row(
                "Market B own flow line 6-7 (MW)", res.flow_own[STUDY_LINE], 9.4, 1
            ),
            label_row("superposed flow line 6-7 (MW)", nb["F_SUM_1"], 49.4, 1),
        ],
    )
    assert_residuals(10, clearing_residuals(nb, "Market B", res), IDENTITY_ABS)


def test_slide_11_market_b_bus7_price_walk_matches(nb: dict[str, Any]) -> None:
    """Slide 11: Market B's price at bus 7, +1.0000 MW at bus 1 times $50 = $50.00."""
    w = nb["walks_B"]["7"]
    check_slide(
        11,
        [
            label_row(
                "gen_slack_0 bus 1 redispatch (MW)",
                w.loc["gen_slack_0", "redispatch (MW)"],
                1.0,
                4,
            ),
            label_row(
                "change in production cost ($/h)",
                w.loc["TOTAL", "cost change ($/h)"],
                50.00,
                2,
            ),
        ],
    )
    assert_residuals(11, walk_residuals(nb, nb["res_B1"], w, "7"), IDENTITY_ABS)


def test_slide_13_market_a_input_leg_matches(nb: dict[str, Any]) -> None:
    """Slide 13: Market A's input, whose leg is +25 MW at bus 1.

    The slide prints the pair's flow on 7 lines in whole MW, Market A's leg at bus 1,
    and the 65 MW of bus 7's 90 MW load the schedule does not cover.
    """
    leg = nb["legs_2"]["Market A"]
    s = nb["S_SCHED"]
    assert leg["node"] == "1"
    check_slide(
        13,
        [
            *pair_rows(nb),
            ("Market A leg at bus 1 (MW)", leg["sign"] * s, 25.0, EXACT),
            ("load block bus 7 (MW)", nb["LOADS"][nb["SINK_BUS"]] - s, 65.0, EXACT),
        ],
    )


def test_slide_14_market_a_scheduled_clearing_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 14: Market A's clearing with the schedule.

    The slide prints 4 injections and 9 flows in whole MW and 7 prices at $0.1/MWh.
    Line 6-7 still binds and both units stay marginal, so every Market A price is
    unchanged, and Market A's production cost falls by its bus 1 price times the
    schedule.
    """
    a2 = nb["st2"]["Market A"]
    res, res0 = a2["res"], a2["res0"]
    s = nb["S_SCHED"]
    check_slide(
        14,
        letter_rows(
            nb,
            res,
            {"2": 36, "3": 29, "1": 25, "7": -90},
            {
                "8-2": 36,
                "7-8": 50,
                "6-7": 40,
                "3-6": 29,
                "8-9": 14,
                "9-4": 14,
                "4-5": 11,
                "5-6": 11,
                "1-4": 25,
            },
            {
                "9": 30.2,
                "4": 27.7,
                "1": 27.7,
                "5": 25.0,
                "2": 35.0,
                "3": 20.0,
                "7": 37.1,
            },
        ),
    )
    g2_0, g3_0, g2, g3 = stated(text, r"from (\d+) and (\d+) MW to (\d+) and (\d+) MW")
    check_slide(
        "14, the notebook's text",
        [
            text_row(
                "gen_0 bus 2 without the schedule (MW)", res0.dispatch["gen_0"], g2_0
            ),
            text_row(
                "gen_1 bus 3 without the schedule (MW)", res0.dispatch["gen_1"], g3_0
            ),
            text_row("gen_0 bus 2 with the schedule (MW)", res.dispatch["gen_0"], g2),
            text_row("gen_1 bus 3 with the schedule (MW)", res.dispatch["gen_1"], g3),
        ],
    )
    assert_residuals(
        14,
        {
            **clearing_residuals(nb, "Market A", res, exo=a2["sign"] * s),
            **rating_residual(nb, res),
            **{
                f"price change bus {n} ($/MWh)": res.lmp[n] - res0.lmp[n]
                for n in nb["PT"].buses
            },
            "dC^A + lambda^A_1 S ($/h)": cost_change(res, res0) + res.lmp["1"] * s,
        },
        IDENTITY_ABS,
    )


def test_slide_15_market_b_input_leg_matches(nb: dict[str, Any]) -> None:
    """Slide 15: Market B's input, whose leg is -25 MW at bus 7.

    The slide prints the pair's flow on 7 lines in whole MW, Market B's leg at bus 7,
    and the bus 1 unit's remaining offer as "$50 x 275 MW".
    """
    leg = nb["legs_2"]["Market B"]
    s = nb["S_SCHED"]
    unit = nb["FLEET"]["gen_slack_0"]
    assert leg["node"] == "7"
    check_slide(
        15,
        [
            *pair_rows(nb),
            ("Market B leg at bus 7 (MW)", leg["sign"] * s, -25.0, EXACT),
            label_row("bus 1 offer ($/MWh)", unit["cost"], 50, 0),
            label_row(
                "bus 1 offer beyond the schedule (MW)", unit["p_nom"] - s, 275, 0
            ),
        ],
    )


def test_slide_16_market_b_scheduled_clearing_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 16: Market B's clearing with the schedule.

    The slide prints 4 injections and 7 flows in whole MW and 9 prices at $0.1/MWh.
    Market B's production cost rises by its bus 7 price times the schedule.
    """
    b2, pt = nb["st2"]["Market B"], nb["PT"]
    res, res0 = b2["res"], b2["res0"]
    s = nb["S_SCHED"]
    check_slide(
        16,
        letter_rows(
            nb,
            res,
            {"1": 288, "9": -173, "5": -90, "7": -25},
            {
                "1-4": 288,
                "9-4": 177,
                "4-5": 111,
                "8-9": 4,
                "7-8": 4,
                "6-7": 21,
                "5-6": 21,
            },
            dict.fromkeys(pt.buses, 50.0),
        ),
    )
    g0, g1 = stated(text, r"rises from (\d+) MW to (\d+) MW at")
    check_slide(
        "16, the notebook's text",
        [
            text_row(
                "gen_slack_0 without the schedule (MW)",
                res0.dispatch["gen_slack_0"],
                g0,
            ),
            text_row(
                "gen_slack_0 with the schedule (MW)", res.dispatch["gen_slack_0"], g1
            ),
        ],
    )
    assert_residuals(
        16,
        {
            **clearing_residuals(nb, "Market B", res, exo=b2["sign"] * s),
            "dC^B - lambda^B_7 S ($/h)": cost_change(res, res0) - res.lmp["7"] * s,
        },
        IDENTITY_ABS,
    )


def test_slide_17_schedule_settlement_matches(nb: dict[str, Any]) -> None:
    """Slide 17: the schedule's settlement by (8); slide 18 repeats the totals.

    The slide multiplies the rounded price $27.74, so it prints $693.50 and a total of
    -$556.50. The run's unrounded settlement is checked against (8) at the cleared
    prices, and against the two markets' change in production cost, negated.
    """
    st2, dc = nb["st2"], nb["dC_2"]
    pay, chg = st2["Market A"], st2["Market B"]
    s = nb["S_SCHED"]
    pay_r = s * round(pay["price"], 2)
    check_slide(
        17,
        [
            label_row("Market A price at bus 1 ($/MWh)", pay["price"], 27.74, 2),
            label_row("Market B charge ($/h)", -chg["amount"], 1250.00, 2),
            label_row("Market A payment at $27.74 ($/h)", pay_r, 693.50, 2),
            label_row("Pi at $27.74 ($/h)", pay_r + chg["amount"], -556.50, 2),
        ],
    )
    lam_a = float(pay["res"].lmp[pay["node"]])
    lam_b = float(chg["res"].lmp[chg["node"]])
    assert_residuals(
        17,
        {
            "Pi less (8) ($/h)": st2["pi"] - (lam_a * s - lam_b * s),
            "Pi + (dC^A + dC^B) ($/h)": st2["pi"] + sum(dc.values()),
        },
        IDENTITY_ABS,
    )


def test_slide_21_market_a_flat_offer_award_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 21: Market A's award on the flat-offer bid, 30 MW at $25.02/MWh.

    The slide labels the price $25/MWh, and prints 4 injections and 8 flows in whole
    MW and 7 prices at $0.1/MWh. The award is the whole bid quantity, and Market A's
    price at bus 5 is its Example 1 price, flat over the volume.
    """
    a3 = nb["st3"]["Market A"]
    res, base = a3["res"], nb["res_A1"]
    check_slide(
        21,
        [
            label_row("Market A price at bus 5 ($/MWh)", a3["price"], 25, 0),
            *letter_rows(
                nb,
                res,
                {"2": 59, "3": 61, "5": -30, "7": -90},
                {
                    "8-2": 59,
                    "7-8": 50,
                    "6-7": 40,
                    "3-6": 61,
                    "5-6": 21,
                    "8-9": 9,
                    "9-4": 9,
                    "4-5": 9,
                },
                {
                    "9": 30.2,
                    "4": 27.7,
                    "1": 27.7,
                    "5": 25.0,
                    "2": 35.0,
                    "3": 20.0,
                    "7": 37.1,
                },
            ),
        ],
    )
    (p5,) = stated(text, r"price at bus 5 is \\\$([\d.]+)/MWh at every volume")
    g2, g3 = stated(text, r"rise to (\d+) and (\d+) MW to serve it")
    check_slide(
        "21, the notebook's text",
        [
            text_row("Market A price at bus 5 ($/MWh)", a3["price"], p5),
            text_row("gen_0 bus 2 (MW)", res.dispatch["gen_0"], g2),
            text_row("gen_1 bus 3 (MW)", res.dispatch["gen_1"], g3),
        ],
    )
    node, award = a3["node"], a3["award"]
    assert_residuals(
        21,
        {
            **clearing_residuals(nb, "Market A", res, exo=a3["sign"] * award),
            **rating_residual(nb, res),
            "award less the bid quantity (MW)": award - nb["QTY_3"],
            "price less the cleared price ($/MWh)": a3["price"] - res.lmp[node],
            "price less the Example 1 price ($/MWh)": a3["price"] - base.lmp[node],
            "dC^A - lambda^A_5 S_A ($/h)": cost_change(res, base) - a3["price"] * award,
        },
        IDENTITY_ABS,
    )


def test_slide_23_market_b_flat_offer_award_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 23: Market B's award on the flat-offer bid, 30 MW at $50/MWh.

    The slide labels the price $50/MWh, and prints 4 injections and 8 flows in whole
    MW and 9 prices at $0.1/MWh. The award is the whole bid quantity, and Market B's
    production cost falls by its bus 3 price times the award.
    """
    b3, pt = nb["st3"]["Market B"], nb["PT"]
    res, base = b3["res"], nb["res_B1"]
    check_slide(
        23,
        [
            label_row("Market B price at bus 3 ($/MWh)", b3["price"], 50, 0),
            *letter_rows(
                nb,
                res,
                {"1": 233, "3": 30, "9": -173, "5": -90},
                {
                    "1-4": 233,
                    "9-4": 152,
                    "4-5": 81,
                    "8-9": 21,
                    "7-8": 21,
                    "6-7": 21,
                    "5-6": 9,
                    "3-6": 30,
                },
                dict.fromkeys(pt.buses, 50.0),
            ),
        ],
    )
    g0, g1 = stated(text, r"backs down from (\d+) MW to (\d+) MW")
    check_slide(
        "23, the notebook's text",
        [
            text_row(
                "gen_slack_0 without the bid (MW)", base.dispatch["gen_slack_0"], g0
            ),
            text_row(
                "gen_slack_0 with the award (MW)", res.dispatch["gen_slack_0"], g1
            ),
        ],
    )
    node, award = b3["node"], b3["award"]
    assert_residuals(
        23,
        {
            **clearing_residuals(nb, "Market B", res, exo=b3["sign"] * award),
            "award less the bid quantity (MW)": award - nb["QTY_3"],
            "price less the cleared price ($/MWh)": b3["price"] - res.lmp[node],
            "price less the Example 1 price ($/MWh)": b3["price"] - base.lmp[node],
            "dC^B + lambda^B_3 S_B ($/h)": cost_change(res, base) + b3["price"] * award,
        },
        IDENTITY_ABS,
    )


def test_slide_24_flat_offer_settlement_matches(nb: dict[str, Any]) -> None:
    """Slide 24: the flat-offer bid's settlement by (10); slide 25 repeats the totals.

    The slide rounds Market A's price to $25.00, so it prints a charge of $750 and a
    total of $750. The run's unrounded settlement is checked against (10) at the
    cleared prices, and against the two markets' change in production cost, negated.
    No position is left open, so no real-time leg settles.
    """
    st3 = nb["st3"]
    a3, b3 = st3["Market A"], st3["Market B"]
    chg_r = a3["award"] * round(a3["price"])
    assert st3["rt"] is None
    check_slide(
        24,
        [
            label_row("Market B payment ($/h)", b3["amount"], 1500.00, 2),
            label_row("Market A charge at $25 ($/h)", chg_r, 750, 0),
            label_row("Pi at $25 ($/h)", b3["amount"] - chg_r, 750, 0),
        ],
    )
    lam_a = float(a3["res"].lmp[a3["node"]])
    lam_b = float(b3["res"].lmp[b3["node"]])
    dc = cost_change(a3["res"], nb["res_A1"]) + cost_change(b3["res"], nb["res_B1"])
    assert_residuals(
        24,
        {
            "Pi less (10) ($/h)": st3["pi"]
            - (lam_b * b3["award"] - lam_a * a3["award"]),
            "Pi + (dC^A + dC^B) ($/h)": st3["pi"] + dc,
            "open position (MW)": st3["open"],
        },
        IDENTITY_ABS,
    )


def test_slide_28_market_a_rising_curve_award_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 28: Market A's award under rising curves, 39.79 MW at the $40 bid.

    The slide labels the award 40 MW and the unawarded rest 80 MW, the bid's 120 MW
    less the rounded award. It prints 4 injections and 8 flows in whole MW and 7
    prices at $0.1/MWh. The award is partial, so its clearing prices bus 5 at the bid,
    by (9).
    """
    a4, qty, bid = nb["st4"]["Market A"], nb["QTY_4"], nb["BID_4"]
    res = a4["res"]
    assert a4["node"] == "5"
    assert 0.0 < a4["award"] < qty, "Market A's award is not partial"
    check_slide(
        28,
        [
            label_row("Market A award (MW)", a4["award"], 40, 0),
            label_row(
                "unawarded at the rounded award (MW)", qty - round(a4["award"]), 80, 0
            ),
            *letter_rows(
                nb,
                res,
                {"2": 62, "3": 67, "5": -40, "7": -90},
                {
                    "8-2": 62,
                    "7-8": 50,
                    "6-7": 40,
                    "3-6": 67,
                    "5-6": 27,
                    "8-9": 12,
                    "9-4": 12,
                    "4-5": 12,
                },
                {
                    "9": 42.5,
                    "4": 41.3,
                    "1": 41.3,
                    "5": 40.0,
                    "2": 44.9,
                    "3": 37.6,
                    "7": 45.9,
                },
            ),
        ],
    )
    award_t, rest_t = stated(
        text, r"reaches the \\\$40 bid at ([\d.]+) MW\. The bid's remaining ([\d.]+) MW"
    )
    check_slide(
        "28, the notebook's text",
        [
            text_row("Market A award (MW)", a4["award"], award_t),
            text_row("unawarded (MW)", qty - a4["award"], rest_t),
        ],
    )
    assert_residuals(
        28,
        {
            **clearing_residuals(
                nb,
                "Market A",
                res,
                exo=a4["sign"] * a4["award"],
                curves=nb["CURVES_TX"],
            ),
            **rating_residual(nb, res),
            "price at bus 5 less the bid ($/MWh)": res.lmp["5"] - bid,
            "settled price less the bid ($/MWh)": a4["price"] - bid,
        },
        QP_ABS,
    )


def test_slide_30_market_b_rising_curve_award_matches(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 30: Market B's award under rising curves, 103 MW at the $40 bid.

    The slide labels the award 103 MW, and prints 4 injections and 8 flows in whole MW
    and 9 prices at $0.1/MWh. The award is partial, so its clearing prices bus 3 at
    the bid, by (9).
    """
    b4, qty, bid, pt = nb["st4"]["Market B"], nb["QTY_4"], nb["BID_4"], nb["PT"]
    res = b4["res"]
    assert b4["node"] == "3"
    assert 0.0 < b4["award"] < qty, "Market B's award is not partial"
    check_slide(
        30,
        [
            label_row("Market B award (MW)", b4["award"], 103, 0),
            *letter_rows(
                nb,
                res,
                {"1": 160, "3": 103, "9": -173, "5": -90},
                {
                    "1-4": 160,
                    "9-4": 124,
                    "4-5": 36,
                    "8-9": 49,
                    "7-8": 49,
                    "6-7": 49,
                    "5-6": 54,
                    "3-6": 103,
                },
                dict.fromkeys(pt.buses, 40.0),
            ),
        ],
    )
    award_t, g1_t = stated(
        text,
        r"reaches the \\\$40 bid at (\d+) MW, where its bus 1 unit runs at (\d+) MW",
    )
    check_slide(
        "30, the notebook's text",
        [
            text_row("Market B award (MW)", b4["award"], award_t),
            text_row("gen_slack_0 bus 1 (MW)", res.dispatch["gen_slack_0"], g1_t),
        ],
    )
    assert_residuals(
        30,
        {
            **clearing_residuals(
                nb,
                "Market B",
                res,
                exo=b4["sign"] * b4["award"],
                curves=nb["CURVES_TX"],
            ),
            "price at bus 3 less the bid ($/MWh)": res.lmp["3"] - bid,
            "settled price less the bid ($/MWh)": b4["price"] - bid,
        },
        QP_ABS,
    )


def test_slide_31_open_position_and_real_time_close_match(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 31: the open position and Market B's real-time close by (12).

    The slide rounds the awards to 40 MW and 103 MW and the open position to 63 MW,
    and the real-time price to $46.30/MWh, so it prints Market A's charge as -$1,600
    and the buy-back as -$2,916.90. Its right ring prints $46.3/MWh at every bus. The
    real-time clearing carries the tagged volume, the lesser award, at bus 3.
    """
    st4, pt = nb["st4"], nb["PT"]
    a4, b4, rt = st4["Market A"], st4["Market B"], st4["rt"]
    q_a, q_rt = round(a4["award"]), round(st4["open"])
    amt_a = -q_a * round(a4["price"], 2)
    amt_rt = -q_rt * round(rt["price"], 1)
    check_slide(
        31,
        [
            label_row("open position (MW)", st4["open"], 63, 0),
            label_row("real-time price at bus 3 ($/MWh)", rt["price"], 46.3, 1),
            *letter_rows(nb, rt["res"], {}, {}, dict.fromkeys(pt.buses, 46.3)),
            label_row("Market A charge at 40 MW ($/h)", amt_a, -1600.00, 2),
            label_row("buy-back at 63 MW and $46.30 ($/h)", amt_rt, -2916.90, 2),
        ],
    )
    award_b, award_a = stated(
        text, r"Market B awarded (\d+) MW and Market A ([\d.]+) MW"
    )
    (short,) = stated(text, r"the scheduler is ([\d.]+) MW short")
    (p_rt,) = stated(text, r"rises to \\\$([\d.]+)/MWh")
    check_slide(
        "31, the notebook's text",
        [
            text_row("Market B award (MW)", b4["award"], award_b),
            text_row("Market A award (MW)", a4["award"], award_a),
            text_row("open position (MW)", st4["open"], short),
            text_row("real-time price at bus 3 ($/MWh)", rt["price"], p_rt),
        ],
    )
    node = b4["node"]
    assert rt["market"] == "Market B"
    assert_residuals(
        31,
        {
            **clearing_residuals(
                nb,
                "Market B",
                rt["res"],
                exo=b4["sign"] * st4["tagged"],
                curves=nb["CURVES_TX"],
            ),
            "open less (S_B - S_A) (MW)": st4["open"] - (b4["award"] - a4["award"]),
            "tagged less the lesser award (MW)": st4["tagged"]
            - min(a4["award"], b4["award"]),
            "real-time price less the cleared price ($/MWh)": rt["price"]
            - rt["res"].lmp[node],
            "Market A charge + S_A P ($/h)": a4["amount"] + a4["award"] * nb["BID_4"],
            "buy-back + open lambda^rt ($/h)": st4["rt_amount"]
            + st4["open"] * rt["price"],
        },
        QP_ABS,
    )


def test_slide_32_example4_scheduler_transactions_match(
    nb: dict[str, Any], text: str
) -> None:
    """Slide 32: the scheduler's transactions in Example 4.

    The slide's title reads "Example #3", and its content is Example 4. It prints the
    positions -40, +103 and -63 MW, each signed as its amount, and a net of -$396.90,
    the sum at its rounded inputs (4,120 - 1,600 - 2,916.90). The run's unrounded net
    is checked against (12) and the figure the notebook's text states. The positions
    sum to zero.
    """
    st4, tx = nb["st4"], nb["tx_4"]
    a4, b4, rt = st4["Market A"], st4["Market B"], st4["rt"]
    q_a, q_b, q_rt = (round(v) for v in (a4["award"], b4["award"], st4["open"]))
    net_r = (
        q_b * round(b4["price"], 2)
        - q_a * round(a4["price"], 2)
        - q_rt * round(rt["price"], 1)
    )
    pos, amt = tx["position (MW)"], tx["amount ($/h)"]
    check_slide(
        32,
        [
            label_row("Market B payment ($/h)", b4["amount"], 4120.00, 2),
            label_row("Pi at the slide's rounding ($/h)", net_r, -396.90, 2),
            *(
                label_row(f"position {name} (MW)", pos.iloc[i], v, 0)
                for i, (name, v) in enumerate(
                    (
                        ("day-ahead buy", -40),
                        ("day-ahead sell", 103),
                        ("real-time", -63),
                    )
                )
            ),
        ],
    )
    (net_t,) = stated(text, r"amounts sum to (−?\\\$[\d.,]+)/h")
    check_slide("32, the notebook's text", [text_row("Pi ($/h)", st4["pi"], net_t)])
    bid = nb["BID_4"]
    assert_residuals(
        32,
        {
            "Pi less (12) ($/h)": st4["pi"]
            - (bid * b4["award"] - bid * a4["award"] - st4["open"] * rt["price"]),
            "Net amount less the transactions ($/h)": amt.iloc[-1]
            - amt.iloc[:-1].sum(),
            "net position (MW)": pos.iloc[-1],
        },
        QP_ABS,
    )


def test_slide_33_open_position_matches(nb: dict[str, Any]) -> None:
    """Slide 33: the open position, 63 MW at the slide's whole-MW rounding."""
    check_slide(33, [label_row("open position (MW)", nb["st4"]["open"], 63, 0)])
