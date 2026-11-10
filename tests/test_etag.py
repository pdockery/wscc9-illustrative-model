# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Guards on ``etag.py``, the interchange tag used by 212a.

The tag is the vehicle the two-market transaction rides on: what each market
engine is handed (``exo``), what the tag itself says (``table``), what a
curtailment does to it (``curtail``), and the path figure (``tag_figure``).
The configuration is 212's: Market A = {bus 2, bus 3}, Market B = the other
seven buses, and a 3 to 2 wheel that lies outside Market B at both ends.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import etag
import footprints
from etag import ETag, Segment
from seams_engine import compute_ptdf
from wscc9_model import build_network

matplotlib.use("Agg", force=False)

MKT = {"Market A": ["2", "3"],
       "Market B": ["1", "4", "5", "6", "7", "8", "9"]}
BA_A2 = ["3"]          # the source balancing authority, a Market A area
E_TAG = 30.0


def _tag(mw=E_TAG):
    return ETag("TAG-001", pse="the transmission customer", gca="BA-A2",
                lca="BA-A1", source="3", sink="2", mw=mw,
                path=[Segment("BA-B", por="3", pod="2", reservation_mw=E_TAG,
                              priority="firm", oasis_ref="OASIS-2024-0001")])


def test_exo_sums_to_zero_when_both_legs_are_outside():
    e = _tag().exo(MKT["Market B"])
    assert e == {"3": +E_TAG, "2": -E_TAG}
    assert abs(sum(e.values())) < 1e-12


def test_exo_is_empty_when_both_legs_are_inside():
    """Market A holds both ends in its own book, so it is handed nothing."""
    assert _tag().exo(MKT["Market A"]) == {}


def test_exo_hands_over_only_the_leg_outside_the_footprint():
    """The source leg inside BA-A2's own area is not handed to that area."""
    e = _tag().exo(BA_A2)
    assert e == {"2": -E_TAG}
    assert "3" not in e


def test_curtail_reduces_both_legs_and_leaves_the_reservation():
    t = _tag().curtail(10.0)
    assert t.mw == pytest.approx(20.0)
    e = t.exo(MKT["Market B"])
    assert e["3"] == pytest.approx(+20.0)
    assert e["2"] == pytest.approx(-20.0)
    assert t.path[0].reservation_mw == pytest.approx(E_TAG)
    assert _tag().curtail(100.0).mw == pytest.approx(0.0)
    assert _tag().mw == pytest.approx(E_TAG)      # the original is untouched


def test_table_carries_the_nerc_fields():
    tbl = _tag().table()
    fields = list(tbl.index)
    for f in ("Tag identifier", "Purchasing-selling entity (PSE)",
              "Generation control area (GCA)", "Load control area (LCA)",
              "Source", "Sink", "Energy profile (MW)"):
        assert f in fields
    for f in ("transmission provider", "point of receipt (POR)",
              "point of delivery (POD)", "reservation (MW)",
              "curtailment priority", "OASIS reference"):
        assert any(f in x for x in fields), f
    assert tbl.loc["Energy profile (MW)", "value"] == "30.0"


def test_a_bad_priority_is_refused():
    with pytest.raises(ValueError):
        Segment("BA-B", por="3", pod="2", reservation_mw=30.0, priority="maybe")


def _figure():
    net = build_network({"line_4": 80.0})
    pt = compute_ptdf(net, slack_bus="1")
    fp = footprints.make(pt, MKT, {"Market A": "#7D3C98", "Market B": "#2471A3"},
                         tie_label="seam")
    return etag.tag_figure(net, pt, _tag(), fp=fp, line_colors=fp.line_colors(pt))


def test_tag_figure_returns_a_figure():
    import matplotlib.pyplot as plt
    fig = _figure()
    assert fig is not None
    assert len(fig.axes) >= 2          # the network panel and the segment table
    plt.close(fig)


def test_the_segment_table_header_fits_its_columns():
    """rights_figure fixes the column widths to its own shorter default labels
    and sets an explicit font size, so a longer header would overprint its
    neighbour unless tag_figure shrinks it."""
    import matplotlib.pyplot as plt
    fig = _figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header = [cell for ax in fig.axes for tbl in ax.tables
              for (row, _), cell in tbl.get_celld().items() if row == 0]
    assert len(header) == len(etag.SEGMENT_COLUMNS)
    for cell in header:
        filled = (cell.get_text().get_window_extent(renderer).width
                  / cell.get_window_extent(renderer).width)
        assert filled <= 0.95, (cell.get_text().get_text(), filled)
    plt.close(fig)
