# SPDX-FileCopyrightText: Market Design Lab contributors
#
# SPDX-License-Identifier: MIT

"""The figure strip of ``seam_views`` (user decision 2026-11-07).

The public notebooks draw their figures with no suptitle, no panel title and no
legend, because the deck slide carries them. ``seam_views.strip`` removes all
three, and ``configure(strip_inline=True)`` applies it to each frame's figure
before the frame's own ``plt.show()``. The tests guard the strip on a
``footprint_figure`` of the canonical case (line_4 rated 40 MW, the 204
partition) and the inline flag on ``own_view``, with the flag's default
``False`` leaving the displayed figure as drawn.

Run from ``wscc9-illustrative-models``:
``PYTHONPATH=lib python -m pytest tests/test_seam_views_strip.py -q``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import pytest

import footprints
import seam_views
import wscc9_figures
from seams_engine import compute_ptdf, solve_engine_dispatch
from wscc9_model import build_network, make_engine

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure

MKT = {"Market A": ["2", "8", "7", "6", "3"], "Market B": ["1", "9", "4", "5"]}
MON = {
    "Market A": ["line_2", "line_3", "line_4", "line_5", "line_6"],
    "Market B": ["line_0", "line_1", "line_7", "line_8"],
}
COLORS = {"Market A": "#993AFF", "Market B": "#2471A3"}


@pytest.fixture
def case() -> Iterator[tuple]:
    """The canonical case with Market A cleared, on a backend that opens no window.

    The backend and ``seam_views.CFG`` are restored afterwards, so no test leaks
    the strip flag into another.
    """
    prev_backend = matplotlib.get_backend()
    prev_cfg = dict(seam_views.CFG)
    plt.switch_backend("Agg")
    net = build_network({"line_4": 40.0})
    pt = compute_ptdf(net, slack_bus="1")
    fp = footprints.make(pt, MKT, COLORS, monitored=MON, tie_label="intertie")
    eng = make_engine("Market A", MKT["Market A"], activated=MON["Market A"])
    res = solve_engine_dispatch(pt, eng)
    try:
        yield net, pt, fp, eng, res
    finally:
        seam_views.CFG.clear()
        seam_views.CFG.update(prev_cfg)
        plt.close("all")
        plt.switch_backend(prev_backend)


def _labels(fig: Figure) -> tuple[bool, list[str], int, int]:
    """The suptitle's visibility, every axes title, and the legend counts."""
    sup = getattr(fig, "_suptitle", None)
    visible = sup is not None and sup.get_visible() and bool(sup.get_text())
    titles = [
        ax.get_title(loc=loc) for ax in fig.axes for loc in ("left", "center", "right")
    ]
    n_ax_legends = sum(ax.get_legend() is not None for ax in fig.axes)
    return visible, titles, n_ax_legends, len(fig.legends)


def test_strip_removes_suptitle_titles_and_legends(case: tuple) -> None:
    """``strip`` leaves no legend, no axes title and no visible suptitle."""
    net, pt, fp, eng, res = case
    fig = wscc9_figures.footprint_figure(
        net, pt, fp, eng, res, highlight="Market A", suptitle="canonical case"
    )
    visible, titles, n_ax, n_fig = _labels(fig)
    assert visible, "the drawn figure carries its suptitle"
    assert any(titles), "the drawn figure carries its panel titles"
    assert n_fig >= 1, "footprint_figure always draws a figure legend"

    out = seam_views.strip(fig)
    assert out is fig
    visible, titles, n_ax, n_fig = _labels(fig)
    assert not visible
    assert all(t == "" for t in titles), titles
    assert n_ax == 0
    assert n_fig == 0
    assert seam_views.strip(fig) is fig  # idempotent on a stripped figure


def test_configure_strip_inline_strips_the_displayed_own_view(
    case: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``strip_inline=True`` strips ``own_view``'s figure before its display."""
    net, pt, fp, eng, res = case
    shown: list[Figure] = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.gcf()))
    stripped: list[Figure] = []
    real_strip = seam_views.strip

    def _recorded_strip(fig: Figure, size: tuple[float, float] | None = None) -> Figure:
        stripped.append(fig)
        return real_strip(fig, size)

    monkeypatch.setattr(seam_views, "strip", _recorded_strip)

    def _draw() -> Figure:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            seam_views.own_view(net, pt, fp, "Market A", eng, res, suptitle="inline")
        return shown[-1]

    assert seam_views.CFG["strip_inline"] is False
    visible, titles, _, n_fig = _labels(_draw())
    assert visible, "the default draws the suptitle"
    assert any(titles), "the default draws the panel titles"
    assert n_fig >= 1, "the default draws the figure legend"
    # the default display is the bare plt.show() of the frame before 2026-11-07
    assert stripped == [], "the default never reaches strip"

    seam_views.configure(strip_inline=True)
    visible, titles, n_ax, n_fig = _labels(_draw())
    assert stripped == [shown[-1]], "the flag strips the displayed figure once"
    assert not visible
    assert all(t == "" for t in titles), titles
    assert n_ax == 0
    assert n_fig == 0
