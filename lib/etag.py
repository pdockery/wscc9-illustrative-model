# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The interchange tag, as the vehicle a transaction between two markets rides on.

An interchange transaction between balancing authorities is represented on an
e-Tag built to the NERC INT standards and the NAESB WEQ-004 business practice
standards (``naesb_nerc_etag``). The tag carries the generation control area and
the load control area, the source and the sink, an energy profile, the
transmission-service detail for each segment of the path, and a curtailment
priority. :class:`ETag` and :class:`Segment` hold exactly those fields, and the
segment's ``reservation_mw`` is the transmission reservation the schedule rides
on, the object the NERC MOD standards define and rate per path
(``nerc_mod_atc``).

Three methods connect the tag to a clearing on the shared network.

1. :meth:`ETag.exo` returns the fixed injection pair a market engine is handed.
   A leg at a bus inside the market's own footprint is already in that market's
   own book, as a portion of the generation it dispatches or of the load it
   serves, so only a leg outside the footprint is handed over. A wheel-through
   lies outside the wheeling market at both ends, so that market carries the
   whole balanced pair. This is the reading of the day-ahead market's
   self-schedule as a fixed injection in ``caiso_edam_techdesc``, and of
   transmission service exercised as a balanced price-taking self-schedule in
   ``pacificorp_edam_oatt2025``.
2. :meth:`ETag.table` returns the tag itself as a copyable table, one row per
   field, with the path rows after the header rows.
3. :func:`tag_figure` draws the tag's physical path as point-of-receipt to
   point-of-delivery arcs over the network, through
   :func:`wscc9_figures.rights_figure`, with the segment table beside it.

**Modeling choices.** The tag is simplified to what a single-period,
energy-only, bid-based DC clearing can read, and each simplification is stated
here rather than in a notebook cell.

1. One segment per transmission provider. A tag in the field carries a segment
   per provider per product, with physical-path and market-path detail; here a
   provider appears once, with one point of receipt and one point of delivery.
2. The energy profile is one number for one interval. A tag in the field carries
   a ramp profile in start/stop blocks; the series clears one period, so the
   profile is the megawatts of that period.
3. There are no losses and no loss-provision detail, because the clearing is
   lossless.
4. Transmission service is recorded as firm or non-firm, which is the
   curtailment-priority field the standards require, without the OASIS product
   classes beneath it. ``oasis_ref`` is a display string and is never read.
   Since 2026-10-13 (304a) a segment may instead carry a NERC transmission
   priority code, ``'0-NX'`` through ``'7-FN'`` (:data:`NERC_PRIORITIES`), the
   priority field a loading-relief procedure walks lowest first; the code maps
   to firm at level 7 and to non-firm below it (:attr:`Segment.firm`), so a
   figure that draws firm and non-firm draws a coded segment the same way. The
   codes are the NERC tagging priorities as ``pjm_nerctagpriority2003`` lists
   them (that 2003 summary predates the 7-FN designation, which is taken from
   the current NAESB tagging tables it names), and the white paper's bucket walk
   in ``eccetf_futurestate2024`` reads them in this order.
5. :meth:`ETag.curtail` reduces the energy profile and leaves every segment's
   reservation unchanged, because a curtailment reduces the schedule rather than
   the right it rides on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Literal

import pandas as pd

import wscc9_figures as figs

if TYPE_CHECKING:
    from collections.abc import Collection

    import pypsa
    from matplotlib.figure import Figure

    from footprints import Footprints
    from lab_types import Bus, BusValues, Line, LineValues, Resource
    from seams_engine import EngineResult, FlowgateSet, MarketEngine, PTDFData

__all__ = ["Segment", "ETag", "tag_figure", "fit_table_header", "SEGMENT_COLUMNS",
           "NERC_PRIORITIES", "priority_level", "tier_of", "dispatch_tags"]

#: The NERC transmission priority codes a tag segment may carry, mapped to the
#: priority level a loading-relief procedure walks lowest first: 0-NX next-hour
#: market, 1-NS non-firm secondary, 2-NH non-firm hourly point-to-point, 3-ND
#: daily, 4-NW weekly, 5-NM monthly, 6-NN non-firm network, 7-F firm
#: point-to-point and 7-FN firm network. The two legacy labels map to the ends
#: of the scale: ``'firm'`` to 7 and ``'non-firm'`` to 0.
NERC_PRIORITIES = {"0-NX": 0, "1-NS": 1, "2-NH": 2, "3-ND": 3, "4-NW": 4,
                   "5-NM": 5, "6-NN": 6, "7-F": 7, "7-FN": 7}
_LEGACY_PRIORITIES = {"firm": 7, "non-firm": 0}


def priority_level(code: str) -> int:
    """The priority level (0 lowest, 7 firm) of a segment priority, a NERC code
    or one of the legacy labels ``'firm'`` / ``'non-firm'``.

    The legacy labels carry no NERC code, so they map to the ends of the scale
    for ordering only. :func:`atc.wlr_assign` needs a NERC code on every tag,
    because the loading-relief bucket walk orders by the code, and it raises on
    a legacy label rather than guess a bucket."""
    if code in NERC_PRIORITIES:
        return NERC_PRIORITIES[code]
    if code in _LEGACY_PRIORITIES:
        return _LEGACY_PRIORITIES[code]
    raise ValueError(f"unknown transmission priority {code!r}")


@dataclass
class Segment:
    """One provider's leg of the tagged path.

    ``provider`` is the transmission provider the segment is taken on, which in
    the 9-bus series is a balancing authority. ``por`` and ``pod`` are the point
    of receipt and the point of delivery, each a bus of the shared network.
    ``reservation_mw`` is the transmission reservation the segment rides on, and
    ``priority`` is the curtailment priority, ``'firm'`` or ``'non-firm'`` or a
    NERC transmission priority code from :data:`NERC_PRIORITIES`.
    ``oasis_ref`` is a reference string for display and is never read by a
    calculation.
    """

    provider: str
    por: str
    pod: str
    reservation_mw: float
    priority: str = "firm"
    oasis_ref: str = ""

    def __post_init__(self) -> None:
        self.provider = str(self.provider)
        self.por = str(self.por)
        self.pod = str(self.pod)
        self.reservation_mw = float(self.reservation_mw)
        if self.priority not in _LEGACY_PRIORITIES and self.priority not in NERC_PRIORITIES:
            raise ValueError("priority must be 'firm', 'non-firm' or a NERC "
                             f"priority code {sorted(NERC_PRIORITIES)}")

    @property
    def firm(self) -> bool:
        """True for firm service: ``'firm'`` or a NERC code at level 7."""
        return priority_level(self.priority) == 7


@dataclass
class ETag:
    """An interchange tag: who moves how much from where to where, on what service.

    ``pse`` is the purchasing-selling entity, the party that holds the
    transaction. ``gca`` is the generation control area and ``lca`` the load
    control area, the two balancing authorities the standards name on every tag.
    ``source`` and ``sink`` are buses of the shared network, ``mw`` is the energy
    profile for the interval, and ``path`` is the ordered list of
    :class:`Segment` the tag is taken on.
    """

    tag_id: str
    pse: str
    gca: str
    lca: str
    source: str
    sink: str
    mw: float
    path: list[Segment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.sink = str(self.sink)
        self.mw = float(self.mw)
        self.path = list(self.path)
        if self.mw < 0:
            raise ValueError("the energy profile mw must be non-negative")

    # -- what a market engine is handed ------------------------------------
    def exo(self, market_buses: Iterable[str]) -> BusValues:
        """The fixed injection pair ``{bus: MW}`` a market engine carries.

        ``market_buses`` is the market's own footprint. The source leg enters as
        ``+mw`` and the sink leg as ``-mw``, and a leg at a bus inside the
        footprint is omitted, because that leg is already a portion of the
        generation the market dispatches or of the load it serves. The returned
        pair therefore sums to zero for a market that carries both legs and to
        the single leg's megawatts for a market that carries one, and it is empty
        for a market that holds both ends inside its own footprint.
        """
        buses = {str(b) for b in market_buses}
        out: dict[str, float] = {}
        if self.source not in buses:
            out[self.source] = out.get(self.source, 0.0) + self.mw
        if self.sink not in buses:
            out[self.sink] = out.get(self.sink, 0.0) - self.mw
        return out

    # -- the tag itself -----------------------------------------------------
    def table(self) -> pd.DataFrame:
        """The tag as a copyable table, one row per field.

        The header rows carry the tag identifier, the purchasing-selling entity,
        the generation and load control areas, the source and sink, and the
        energy profile. The path rows follow, one block per segment, carrying the
        transmission provider, the point of receipt, the point of delivery, the
        reservation, the curtailment priority and the OASIS reference.
        """
        rows = [
            ("Tag identifier", self.tag_id),
            ("Purchasing-selling entity (PSE)", self.pse),
            ("Generation control area (GCA)", self.gca),
            ("Load control area (LCA)", self.lca),
            ("Source", f"bus {self.source}"),
            ("Sink", f"bus {self.sink}"),
            ("Energy profile (MW)", f"{self.mw:,.1f}"),
        ]
        for i, seg in enumerate(self.path, start=1):
            rows += [
                (f"Segment {i}: transmission provider", seg.provider),
                (f"Segment {i}: point of receipt (POR)", f"bus {seg.por}"),
                (f"Segment {i}: point of delivery (POD)", f"bus {seg.pod}"),
                (f"Segment {i}: reservation (MW)", f"{seg.reservation_mw:,.1f}"),
                (f"Segment {i}: curtailment priority", seg.priority),
                (f"Segment {i}: OASIS reference", seg.oasis_ref),
            ]
        return pd.DataFrame(rows, columns=["field", "value"]).set_index("field")

    def curtail(self, mw: float) -> "ETag":
        """A copy of the tag with the energy profile reduced by ``mw`` megawatts.

        The reduction applies to the one energy profile, so both legs of the
        :meth:`exo` pair fall by the same amount. Every segment's reservation is
        unchanged, because a curtailment reduces the schedule and not the right
        it rides on. The profile is floored at zero.
        """
        return replace(self, mw=max(0.0, self.mw - float(mw)),
                       path=[replace(s) for s in self.path])


#: The segment table's header, wrapped one word per line where a label is longer
#: than the column carrying it. :func:`wscc9_figures.rights_figure` fixes its
#: column widths to its own default labels (``Right``, ``MW``, ``ATC``,
#: ``%ATC``), so the two rightmost columns are narrow.
SEGMENT_COLUMNS = ("Segment\n(POR→POD)", "Tagged\nMW",
                   "Reservation\n(MW)", "% of\nreservation")


def fit_table_header(fig: Figure, max_size: float = 10.0, min_size: float = 6.0,
                     fill: float = 0.95) -> float | None:
    """Shrink a figure's table header until every label fits its own column.

    :func:`wscc9_figures.rights_figure` fixes the table's column widths to its
    own default labels and turns matplotlib's automatic font sizing off with an
    explicit 10 point size, so a longer header label neither shrinks nor wraps
    and overprints its neighbour. This steps the header row's font size down by
    half a point at a time until the widest header cell's text occupies no more
    than ``fill`` of that cell's width, and it stops at ``min_size``.

    Returns the size it settled on, or ``None`` when the figure carries no table.
    The measurement needs a renderer, so a backend that cannot supply one leaves
    the header at ``max_size``.
    """
    header = [cell for ax in fig.axes for tbl in ax.tables
              for (row, _), cell in tbl.get_celld().items() if row == 0]
    if not header:
        return None
    size = float(max_size)
    while True:
        for cell in header:
            cell.get_text().set_fontsize(size)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # the except below takes a backend without it
            filled = max(cell.get_text().get_window_extent(renderer).width
                         / cell.get_window_extent(renderer).width
                         for cell in header)
        except Exception:                       # no measurable renderer
            return size
        if filled <= fill or size <= min_size:
            return size
        size = max(float(min_size), size - 0.5)


def tag_figure(net: pypsa.Network, pt: PTDFData, tag: ETag, *,
               fp: Footprints | None = None,
               monitored: Literal["all"] | Collection[Line] | FlowgateSet = "all",
               coords: dict[Bus, tuple[float, float]] | None = None,
               bus_colors: dict[Bus, str] | None = None,
               line_colors: dict[Line, str] | None = None,
               line_widths: LineValues | None = None,
               tier_colors: dict[str, str] | None = None,
               title: str | None = None, font_scale: float = 1.0,
               col_labels: Iterable[str] | None = None,
               header_fontsize: float = 10.0) -> Figure:
    """The tagged path over the network, with the segment table beside it.

    Each segment is drawn as a point-of-receipt to point-of-delivery arc through
    :func:`wscc9_figures.rights_figure`, which also rides the superposed schedule
    flow on the lines as ``flow / limit`` and turns a line red where the schedule
    overloads it. The table beside it carries one row per segment: the path, the
    tagged megawatts, the reservation the segment rides on, and the share of that
    reservation the tag uses.

    The header takes :data:`SEGMENT_COLUMNS` unless ``col_labels`` overrides it,
    and :func:`fit_table_header` then shrinks that header from ``header_fontsize``
    until every label fits its own column, because the underlying figure's column
    widths are fixed to its own shorter default labels.

    Returns the figure.
    """
    rights, caps, pcts = [], [], []
    for seg in tag.path:
        rights.append({"source": seg.por, "sink": seg.pod, "mw": tag.mw,
                       "tier": "inter" if seg.firm else "intra"})
        caps.append(seg.reservation_mw)
        pcts.append(100.0 * tag.mw / seg.reservation_mw
                    if seg.reservation_mw > 1e-9 else 0.0)
    fig = figs.rights_figure(
        net, pt, rights, fp=fp, coords=coords, monitored=monitored,
        bus_colors=bus_colors, line_colors=line_colors, line_widths=line_widths,
        tier_colors=tier_colors, show_atc=False, show_sft=True,
        atc_values=caps, pct_values=pcts,
        col_labels=list(col_labels if col_labels is not None else SEGMENT_COLUMNS),
        title=title or f"e-Tag {tag.tag_id} -- the tagged path on the network",
        font_scale=font_scale)
    fit_table_header(fig, max_size=header_fontsize)
    return fig


# -- the tag's physical path as a deck table (304a) ---------------------------
#: Columns of the physical-path table, in the order of the Markets+ seams
#: workshop's tag tables (SPP, Panel 3 deck, 16 September 2026), with the NERC
#: transmission priority added as the last column.
PATH_COLUMNS = ("BA", "TSP", "MO", "PSE", "POR", "POD", "S/E", "Priority")
_PATH_HEADER = "#1B9BD7"     # the deck's title bar
_PATH_BAND = "#CDEBF8"       # the deck's light-blue band rows
_PATH_RULE = "#7FC3E8"


def physical_path_figure(tag: "ETag", *, names: dict[str, str] | None = None,
                         mo: dict[str, str] | None = None,
                         se: dict[Bus, str] | None = None,
                         title: str | None = None,
                         priority_colors: dict[str, str] | None = None,
                         figsize: tuple[float, float | None] = (10.0, None),
                         fontsize: float = 11) -> Figure:
    """The tag's physical path as a table in the style of the Markets+ seams deck.

    One row for the source (BA = the generation control area, POR = the source),
    one row per :class:`Segment` (TSP, MO, PSE, POR, POD, S/E, priority), and one
    row for the sink (BA = the load control area, POR = the sink), under a title
    bar, with the band rows shaded as the deck shades them. The NERC priority
    column is the addition to the deck's layout, and its text takes the
    priority's colour from :data:`wscc9_figures.PRIORITY_COLORS`, so a row reads
    against the tag's arc on the network panel.

    ``names`` maps the model's party names (``'Market A'``) to display codes;
    ``mo`` maps a segment's provider to its market operator and ``se`` maps a
    POD bus to its scheduling entity (both default to the provider and to the
    tag's load control area). Display only; no calculation reads it.
    Returns the figure.
    """
    import matplotlib.pyplot as plt

    nm = (lambda s: (names or {}).get(s, s))
    mo_of = (lambda p: nm((mo or {}).get(p, p)))
    se_of = (lambda b: nm((se or {}).get(str(b), tag.lca)))
    pcol = dict(figs.PRIORITY_COLORS if priority_colors is None else priority_colors)
    pse = nm(tag.pse)
    rows = [[nm(tag.gca), "", mo_of(tag.gca), pse, f"Source: bus {tag.source}", "", "", ""]]
    for seg in tag.path:
        rows.append(["", nm(seg.provider), mo_of(seg.provider), pse,
                     f"bus {seg.por}", f"bus {seg.pod}", se_of(seg.pod), seg.priority])
    rows.append([nm(tag.lca), "", mo_of(tag.lca), pse, f"Sink: bus {tag.sink}", "", "", ""])
    n = len(rows) + 1                           # the label row and the body
    w, h = figsize
    h = h if h is not None else 0.42 * (n + (1 if title != "" else 0))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    ax.set_position([0.0, 0.0, 1.0, 1.0])  # type: ignore[arg-type]  # stub omits list
    ttl = title if title is not None else (
        f"EXAMPLE: Physical Path for tag {tag.tag_id} "
        f"(bus {tag.source} to bus {tag.sink}, {tag.mw:,.1f} MW)")
    frac = n / (n + 1) if ttl else 1.0          # the title bar takes one row height
    tb = ax.table(cellText=rows, colLabels=list(PATH_COLUMNS), cellLoc="left",
                  colLoc="left", bbox=[0.0, 0.0, 1.0, frac],  # type: ignore[arg-type]  # stub omits list
                  colWidths=[0.09, 0.12, 0.12, 0.12, 0.17, 0.11, 0.09, 0.10])
    tb.auto_set_font_size(False)
    tb.set_fontsize(fontsize)
    last = len(rows)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(_PATH_RULE)
        cell.set_linewidth(0.8)
        if r == 0:                                  # the column-label row
            cell.set_facecolor(_PATH_BAND)
            cell.get_text().set_fontweight("bold")
            continue
        cell.set_facecolor(_PATH_BAND if r % 2 == 0 else "white")
        if r in (1, last) and c == 0:               # the source and sink BA
            cell.get_text().set_fontweight("bold")
        if c == len(PATH_COLUMNS) - 1:
            code = cell.get_text().get_text()
            if code:
                cell.get_text().set_color(pcol.get(code, "#222222"))
                cell.get_text().set_fontweight("bold")
    if ttl:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((0.0, frac), 1.0, 1.0 - frac, transform=ax.transAxes,
                               facecolor=_PATH_HEADER, edgecolor=_PATH_RULE,
                               linewidth=0.8, clip_on=False))
        ax.text(0.5, frac + (1.0 - frac) / 2, ttl, transform=ax.transAxes,
                ha="center", va="center", fontsize=fontsize + 1,
                fontweight="bold", color="#0B2545")
    return fig


# ──────────────────────────────────────────────────────────────────────────
# A clearing's dispatch read as tags, and the priority code read as a tier
# ──────────────────────────────────────────────────────────────────────────
#: The curtailment tier of each NERC transmission priority code, the Lab rule
#: :func:`tier_of` states.
_TIER_OF_CODE = {"7-FN": 0, "7-F": 1, **{c: 3 for c, lvl in NERC_PRIORITIES.items()
                                          if lvl < 7}}


def tier_of(code: str) -> int:
    """The curtailment tier of a NERC transmission priority code, the tier
    ``atc.Request.tier`` carries and the ``pro_rata`` rule of
    ``atc.nominate_and_test`` cuts highest first.

    The mapping is a Lab rule, stated as follows:

    1. ``'7-FN'``, firm network integration transmission service, maps to
       tier 0, the network service tier.
    2. ``'7-F'``, firm point-to-point service, maps to tier 1, the long-term
       firm point-to-point tier. The NERC code does not separate long-term
       from short-term firm service, so tier 2 is reached by no code.
    3. ``'0-NX'`` through ``'6-NN'``, the non-firm codes, map to tier 3, the
       non-firm tier.

    The rule is how a tag's priority (notebook 304a's ``VARIANTS``) reaches
    the curtailment order of a rights book. The legacy labels ``'firm'`` and
    ``'non-firm'`` carry no NERC code, so they raise rather than guess a tier,
    as :func:`atc.wlr_assign` raises on them.
    """
    code = str(code)
    if code in _TIER_OF_CODE:
        return _TIER_OF_CODE[code]
    if code in _LEGACY_PRIORITIES:
        raise ValueError(
            f"{code!r} is a legacy label with no NERC priority code; a tier "
            f"is read from a code in {sorted(NERC_PRIORITIES)}")
    raise ValueError(f"unknown transmission priority {code!r}")


def _bus_key(b: Bus) -> tuple[int, int, str]:
    """Sort key for bus names: numeric names numerically, the rest after them
    lexicographically (``seams_engine._bus_sort_key``'s rule)."""
    s = str(b)
    return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)


def _gross_split(eng: MarketEngine, res: EngineResult,
                 tol: float = 1e-6) -> list[tuple[Resource, Bus, Bus, float]]:
    """Each unit's dispatch split over the engine's served loads in proportion
    to load, as ``[(unit, source bus, sink bus, MW), ...]``, cheapest unit
    first (cost, then unit name), then by sink bus.

    A load bus is a load of more than 1e-9 MW, and its served load is the
    load less the shed at the bus. The dispatch must equal the served load
    within ``tol`` MW, or the split raises, because a unit's megawatts with no
    load to serve have no sink in the rule.
    """
    loads = {str(b): float(v) for b, v in eng.loads.items() if v > 1e-9}
    shed = res.shed_by_bus or {}
    served = {b: v - float(shed.get(b, 0.0)) for b, v in loads.items()}
    D = sum(served.values())
    units = sorted([(g, float(mw)) for g, mw in res.dispatch.items() if mw > 1e-9],
                   key=lambda gm: (eng.gens[gm[0]]["cost"], gm[0]))
    G = sum(mw for _, mw in units)
    if abs(G - D) >= tol:
        raise ValueError(
            f"{eng.name}: the dispatch ({G:.6f} MW) must balance the served "
            f"load ({D:.6f} MW); the gross split has no sink for the difference")
    out = []
    for g, mw in units:
        src = str(eng.gens[g]["bus"])
        for b in sorted(served, key=_bus_key):
            out.append((g, src, b, mw * served[b] / D))
    return out


def _check_against_requests(eng: MarketEngine, res: EngineResult,
                            split: list[tuple[Resource, Bus, Bus, float]],
                            tol: float = 1e-9) -> None:
    """Compare the gross split with ``atc.dispatch_requests`` where that
    function exists, source bus by sink bus, and raise on a difference."""
    try:
        import atc as _atc
    except ImportError:            # pragma: no cover - atc ships with etag
        return
    fn = getattr(_atc, "dispatch_requests", None)
    if fn is None:
        return
    area_of = {str(b): eng.name for b in eng.buses}
    reqs = fn(res, gens=eng.gens, loads=eng.loads, area_of=area_of,
              basis="gross", granularity="unit")

    def _bus(point_w: BusValues) -> Bus:
        if len(point_w) != 1:
            raise ValueError("a unit-granularity request names one bus at "
                             "each end")
        return next(iter(point_w))

    theirs: dict[tuple[Bus, Bus], float] = {}
    for r in reqs:
        k = (_bus(r.source_w), _bus(r.sink_w))
        theirs[k] = theirs.get(k, 0.0) + float(r.mw)
    ours: dict[tuple[Bus, Bus], float] = {}
    for _g, src, snk, q in split:
        ours[(src, snk)] = ours.get((src, snk), 0.0) + q
    scale = tol * max(1.0, sum(ours.values()))
    for k in set(ours) | set(theirs):
        if abs(ours.get(k, 0.0) - theirs.get(k, 0.0)) > scale:
            raise ValueError(
                f"{eng.name}: atc.dispatch_requests and the gross split "
                f"differ on {k[0]} -> {k[1]}: {theirs.get(k, 0.0)!r} against "
                f"{ours.get(k, 0.0)!r} MW")


def dispatch_tags(eng: MarketEngine, res: EngineResult, priority: str,
                  prefix: str) -> list[tuple[ETag, Resource]]:
    """Generator-to-load tags of one market's dispatch: each unit's dispatch
    split over the market's served loads in proportion to load.

    Returns ``[(ETag, unit), ...]``, cheapest unit first (by cost, then by
    unit name), then by sink bus. The tag from unit ``i`` to load bus ``n``
    carries ``g_i * d_n / sum_n' d_n'`` MW, where ``d_n`` is the served load,
    the load less the shed at the bus. Each tag names the market as its
    generation control area, its load control area and its one segment's
    provider, takes the market's scheduling coordinator as its
    purchasing-selling entity, carries a reservation equal to its megawatts,
    and takes
    ``priority`` as its segment's priority code. Tag identifiers are
    ``prefix`` followed by a running count from 1.

    The rule is notebook 304a's (cell 18), lifted unchanged, and it is the
    gross basis of the independent-clearing rights construction, under which
    the tags' induced flows sum to the market's own flow contribution on
    every line, because the dispatch balances the served load. The function
    raises when the dispatch does not balance the served load within 1e-6 MW,
    as 304a asserts, which includes a clearing with cleared demand bids,
    since a bid has no load bus in the rule.

    Where ``atc.dispatch_requests`` exists it is called with
    ``basis="gross"`` and ``granularity="unit"``, and its volumes, summed per
    source bus and sink bus, must equal the tags' within 1e-9 times the
    served load, or the function raises. The tags themselves are built from
    the rule here, because a tag names its unit and 304a's order, which the
    request book does not have to carry.
    """
    split = _gross_split(eng, res)
    _check_against_requests(eng, res, split)
    out: list[tuple[ETag, Resource]] = []
    for g, src, b, q in split:
        tid = f"{prefix}{len(out) + 1}"
        out.append((ETag(tag_id=tid, pse=f"{eng.name} scheduling coordinator",
                         gca=eng.name, lca=eng.name, source=src, sink=b, mw=q,
                         path=[Segment(provider=eng.name, por=src, pod=b,
                                       reservation_mw=q, priority=priority)]), g))
    return out
