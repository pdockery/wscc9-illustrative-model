"""Slide checks for the deck notebooks, run during development.

A deck notebook clears its deck's examples and prints the ledgers its reading
needs. Whether those clears reproduce the numbers the slides print is checked
here, outside the notebook: :func:`run_notebook` executes a notebook's code cells
in one namespace, and each ``tests/test_slides_<deck>.py`` compares named results
in that namespace with the slide's printed values through :func:`check_slide`.

A row is ``(quantity, model, slide, tolerance)`` or ``(quantity, model, slide,
tolerance, compared)``. ``compared`` is the value the slide's figure prints from
the run, which differs from ``model`` only where the figure rounds a label, and
defaults to ``model``. A row holds when ``|compared - slide| <= tolerance``; the
tolerance is half a unit of the slide's last printed digit.

    python -m pytest tests -q -m slides
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import nbformat
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

Row = tuple[str, float, float, float, float]

#: Absolute tolerance of a whole-MW slide label: half a unit of its last printed digit, MW.
WHOLE_MW_ABS = 0.5

#: Absolute tolerance of a slide price printed to $0.1/MWh: half its last digit, $/MWh.
TENTH_PRICE_ABS = 0.05

#: Absolute slack added to every row's tolerance, so a value that lands on the
#: half-unit boundary in binary floating point still holds. Units of the row.
ROW_SLACK_ABS = 1e-9


def run_notebook(name: str) -> dict[str, Any]:
    """Execute the code cells of root notebook ``name`` and return their namespace.

    The cells run in order in one namespace, from the repository root, on the
    non-interactive backend. ``plt.show`` and ``display`` draw nothing, and every
    figure is closed after its cell, so the run writes no file and keeps no figure.
    Lines that start with ``%`` or ``!`` are dropped, as IPython syntax.
    """
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    nb = nbformat.read(ROOT / name, as_version=4)
    ns: dict[str, Any] = {"__name__": "__main__", "display": lambda *_a, **_k: None}
    show = plt.show
    cwd = Path.cwd()
    plt.show = lambda *_a, **_k: None  # no display outside a kernel
    os.chdir(ROOT)
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            src = "\n".join(
                ln
                for ln in cell.source.splitlines()
                if not ln.lstrip().startswith(("%", "!"))
            )
            exec(compile(src, f"{name}[{i}]", "exec"), ns)  # noqa: S102  # the notebook is the code under test
            plt.close("all")
    finally:
        plt.show = show
        os.chdir(cwd)
    return ns


def as_row(row: Sequence[Any]) -> Row:
    """Return ``row`` with its ``compared`` value, which defaults to ``model``."""
    quantity, model, slide, tol, *rest = row
    compared = rest[0] if rest else model
    return (str(quantity), float(model), float(slide), float(tol), float(compared))


def check_slide(slide: int | str, rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Assert every row of one slide and return the table of rows.

    The assertion message carries the whole table, so a failing slide shows each
    row's model value, compared value, slide value and tolerance.
    """
    table = pd.DataFrame(
        [as_row(r) for r in rows],
        columns=["quantity", "model", "slide", "tolerance", "compared"],
    )
    gap = (table["compared"] - table["slide"]).abs()
    table["holds"] = gap <= table["tolerance"] + ROW_SLACK_ABS
    bad = table.loc[~table["holds"], "quantity"].tolist()
    assert not bad, f"slide {slide}: outside tolerance: {bad}\n" + table.to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"
    )
    return table


def drawn_flow(flow: float) -> float:
    """The whole-MW flow label ``seam_views.combined_views`` draws.

    The figure rounds a flow to 0.1 MW and then to a whole MW, half to even, so a
    flow of 64.54 MW is drawn as 64 where direct rounding gives 65.
    """
    return float(f"{np.round(abs(flow), 1):.0f}")


def flow_rows(
    flows: Mapping[str, float], slide: Mapping[str, float], *, two_step: bool = False
) -> list[Row]:
    """Rows comparing each line's absolute flow with the slide's label, to half a MW.

    With ``two_step`` the compared value is :func:`drawn_flow`, the label of a
    ``seam_views.combined_views`` figure.
    """
    rows = []
    for m, v in slide.items():
        f = abs(float(flows[m]))
        compared = drawn_flow(f) if two_step else f
        rows.append((f"|F| {m} (MW)", f, float(v), WHOLE_MW_ABS, compared))
    return rows


def injection_rows(inj: Mapping[str, float], slide: Mapping[str, float]) -> list[Row]:
    """Rows comparing net injections with the slide's node labels, to half a MW."""
    return [
        as_row((f"net injection bus {b} (MW)", inj[b], v, WHOLE_MW_ABS))
        for b, v in slide.items()
    ]


def price_rows(prices: Mapping[str, float], slide: Mapping[str, float]) -> list[Row]:
    """Rows comparing LMPs with the slide's ring labels, to $0.05/MWh."""
    return [
        as_row((f"LMP bus {b} ($/MWh)", prices[b], v, TENTH_PRICE_ABS))
        for b, v in slide.items()
    ]
