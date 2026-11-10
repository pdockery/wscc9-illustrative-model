# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Teaching figure compositions for the WSCC 9-bus notebooks.

``nodal_plot`` provides the reusable primitives (network topology panel +
circlize nodal-dispatch panel + ``plot_combined_letter`` that puts them side by
side). This module holds the teaching *compositions* the notebooks reuse — one
clearing drawn as a [network + nodal dispatch] pair, colored by footprint, with
the standard shared legend.

* :func:`footprint_figure` — the composite for one clearing. Color one
  footprint and gray the rest (``highlight=`` or explicit ``dim_buses=``), draw
  ties/managed lines in footprint colors, put a footprint band on the circlize
  ring, and annotate any self-schedule on the network panel. This single
  function replaces the per-notebook ``example_figure`` (congestion notebooks,
  grays the other BA) and ``market_figure`` (seams notebook, grays the other
  market).
* :func:`composite_figure` — the unified two-panel view (nothing grayed), the
  figure each downstream notebook prints in its set-up cell.
* :func:`draw_net_dispatch` / :func:`transfer_figure` — the "Transfers" inset
  (each footprint as one bubble, the scheduled transfer as a flow/limit label)
  and the composite carrying it.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import nodal_plot
from nodal_plot import plot_combined_letter, compute_ptdf_flows, plot_network_topology
from seams_engine import (
    active as _case,
    to_supply_demand, susceptance_widths, shed_segments, served_demand,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence

    import pandas as pd
    import pypsa
    from atc import Award
    from bilateral import OrderBookResult, Supplier
    from footprints import Footprints
    from lab_types import Bus, BusValues, Line, LineValues, Loads, Resource
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from seams_engine import EngineResult, FlowgateSet, MarketEngine, PTDFData
    from typing_extensions import Unpack

    _Blocks: TypeAlias = dict[Bus, list[dict[str, Any]]]
    _SupDem: TypeAlias = tuple[_Blocks, Loads]
    _Coords: TypeAlias = dict[Bus, tuple[float, float]]
    _Curves: TypeAlias = dict[Resource, tuple[float, float]]
    _Flows: TypeAlias = Sequence[tuple[Bus, Bus, float] | tuple[Bus, Bus, float, str]]
    _Lazy: TypeAlias = dict[Bus, str] | _Coords | list[Bus] | Bus | None
    _Monitored: TypeAlias = Literal["all"] | Collection[Line] | FlowgateSet

    class _FootprintKw(TypedDict, total=False):
        highlight: str | None
        dim_buses: Iterable[Bus] | None
        exo_sched: BusValues | None
        exo_annotate: bool
        sup_dem: _SupDem | None
        bus_colors: dict[Bus, str] | None
        demand_segments: _Blocks | None
        shed: bool
        legend_note: str | None
        annotate_roles: bool
        axis_key: bool
        node_net_mw: bool | BusValues
        network_show_lmp: bool | None
        line_flows: LineValues | None
        constrained_lines: Iterable[Line] | None
        bus_lmps: BusValues | None
        bus_coords: _Coords | None
        ring_order: list[Bus] | None
        center_bus: Bus | None
        curves: _Curves | None
        title_left: str
        title_right: str
        figsize: tuple[float, float]
        panel_ratios: tuple[float, float]

# Drawing-layout defaults now resolve through the ambient case at CALL time
# (they used to be imported from wscc9_model at module import, which welded this
# module to the 9-bus case). Every public function still accepts explicit
# bus_colors= / bus_coords= / ring_order= / center_bus= overrides.


def _BUS_COLORS() -> dict[Bus, str]:
    return _case().bus_colors


def _COORDS() -> _Coords:
    return _case().coords


def _RING_ORDER() -> list[Bus]:
    return _case().ring_order


def _CENTER_BUS() -> Bus | None:
    return _case().center_bus


def __getattr__(name: str) -> _Lazy:
    """PEP 562: keep ``wscc9_figures.BUS_COLORS``-style access working."""
    lazies: dict[str, Callable[[], _Lazy]] = {
        "BUS_COLORS": _BUS_COLORS, "COORDS": _COORDS,
        "RING_ORDER": _RING_ORDER, "CENTER_BUS": _CENTER_BUS}
    if name in lazies:
        return lazies[name]()
    raise AttributeError(name)

#: Tier colors for sold transmission rights drawn on the network.
RIGHT_TIER_COLORS = {"inter": "#117A65", "intra": "#B9770E"}
#: Arc colours by NERC transmission priority code (304a): firm service in the
#: firm-right green of :data:`RIGHT_TIER_COLORS`, non-firm hourly in amber, the
#: other non-firm codes in a lighter amber. Passed as ``tier_colors`` with each
#: arc's ``tier`` set to its code.
PRIORITY_COLORS = {"7-F": "#117A65", "7-FN": "#0E6655", "6-NN": "#CA6F1E",
                   "5-NM": "#CA6F1E", "4-NW": "#CA6F1E", "3-ND": "#CA6F1E",
                   "2-NH": "#B9770E", "1-NS": "#D4AC0D", "0-NX": "#D4AC0D"}
#: Induced-flow panel (304a): a line the tag loads in the constraint's loading
#: direction (orange, because red is reserved for a binding or overloaded line),
#: and a line it counterflows.
INDUCED_LOADING = "#D35400"
INDUCED_COUNTER = "#1F618D"
_WECC_PATH = "#B8860B"   # gold: a WECC-rated inter-BA path (tie)

_DIM = "#C8CCCE"        # grayed (out-of-focus) footprint
_UNASSIGNED = "#AAB7B8"  # line assigned to nobody
_SELFSCHED = "#95A5A6"   # self-schedule: a price-taking block the dispatch did not optimize


def footprint_figure(
    net: pypsa.Network,
    pt: PTDFData,
    fp: Footprints,
    engine: MarketEngine,
    res: EngineResult,
    *,
    highlight: str | None = None,
    dim_buses: Iterable[Bus] | None = None,
    exo_sched: BusValues | None = None,
    exo_annotate: bool = True,
    sup_dem: _SupDem | None = None,
    bus_colors: dict[Bus, str] | None = None,
    demand_segments: _Blocks | None = None,
    shed: bool = True,
    legend_note: str | None = None,
    suptitle: str | None = None,
    annotate_roles: bool = True,
    axis_key: bool = True,
    node_net_mw: bool | BusValues = True,
    network_show_lmp: bool | None = None,
    line_flows: LineValues | None = None,
    constrained_lines: Iterable[Line] | None = None,
    bus_lmps: BusValues | None = None,
    bus_coords: _Coords | None = None,
    ring_order: list[Bus] | None = None,
    center_bus: Bus | None = None,
    curves: _Curves | None = None,
    title_left: str = "Network -- DC power flow",
    title_right: str = "Nodal dispatch -- merit order, demand & flows",
    figsize: tuple[float, float] = (11, 6.2),
    panel_ratios: tuple[float, float] = (1, 1),
) -> Figure:
    """Combined network + nodal-dispatch composite for one clearing.

    Parameters
    ----------
    net, pt : pypsa.Network, PTDFData
        The network and its shift factors.
    fp : footprints.Footprints
        The partition (line colors, bands, footprint membership).
    engine, res : MarketEngine, EngineResult
        The cleared engine and its result.
    highlight : str, optional
        Footprint to keep in color; every other footprint's buses are grayed.
        (Mutually exclusive with ``dim_buses``.)
    dim_buses : iterable, optional
        Explicit buses to gray (the congestion-notebook per-BA view).
    exo_sched : dict, optional
        ``{bus: MW}`` self-schedule (+ = delivery INTO the focus footprint).
        Drawn as it enters the clear (import = unpriced supply at the sink,
        export = scheduled demand at the source) and annotated on the network
        panel. Ignored when ``sup_dem`` is given (then used only for the label).
    sup_dem : (dict, dict), optional
        Override ``(supply_by_bus, demand_by_bus)`` to take full control of the
        bars (e.g. drawing a trade as a dedicated block).
    shed : bool
        Track load shedding — trace gen→load flows on SERVED demand and draw the
        unserved tail as a faint segment. Default True (off for trade views that
        pass their own ``demand_segments``).
    node_net_mw : True, dict, or False
        Annotate each network node with its NET injection (+) / withdrawal (−).
        ``True`` (the default, the series standard since 2026-09-12) reads the
        clearing's own ``res.injection`` (dispatched gen − served load + exo);
        a dict ``{bus: MW}`` overrides it; ``False`` drops the labels and
        prints the LMP on the network instead. Pairs with ``network_show_lmp``
        (default then: hide the network-panel LMP, since price is read off
        the dispatch ring).
    network_show_lmp : bool, optional
        Whether the NETWORK panel prints the bus LMP. Default ``False`` while
        the net-MW labels are drawn, ``True`` when ``node_net_mw=False``. The
        dispatch ring always keeps its LMP labels.
    line_flows : dict, optional
        ``{line: MW}`` drawn on the network panel instead of the clearing's own
        ``res.flow_own``. Use for a STUDY view, where the market draws its own
        book superposed on a proxy of the neighbor's.
    constrained_lines : iterable, optional
        Lines drawn red instead of the clearing's binding set (``res.line_dual``).
        Pairs with ``line_flows``: a studied line can be flagged without any dual.
    bus_lmps : dict, optional
        ``{bus: $/MWh}`` instead of ``res.lmp`` -- e.g. each footprint priced by
        its own clearing when two books are drawn on one ring.
    """
    if curves:                              # attach rising MC curves so the ring draws each
        for _g in engine.gens:              # dispatched unit as a sloped WEDGE (a + b*g), not a
            if _g in curves:                # flat rectangle -- matches the QP the dispatch cleared with
                engine.gens[_g]["curve"] = curves[_g]
        # Bars handed in through ``sup_dem`` were built from the engine BEFORE
        # the curve was attached (self_schedule_bars, bid_bars), so stamp the
        # curve onto those blocks too, else they draw flat.
        if sup_dem is not None:
            for _blocks in sup_dem[0].values():
                for _u in _blocks:
                    if _u.get("unit_id") in curves:
                        _u["mc0"], _u["mc_slope"] = map(float, curves[_u["unit_id"]])

    if node_net_mw is True:
        bus_net = {b: float(res.injection[pt.bus_idx[b]]) for b in pt.buses}
    elif isinstance(node_net_mw, dict):
        bus_net = {str(k): float(v) for k, v in node_net_mw.items()}
    else:
        bus_net = None
    if network_show_lmp is None:
        network_show_lmp = bus_net is None

    has_selfsched = False
    if sup_dem is not None:
        sup, dem = sup_dem
        has_selfsched = any(u.get("unit_id") == "self_schedule" or u.get("color") == _SELFSCHED
                            for _blocks in sup.values() for u in _blocks)
    else:
        sup, dem = to_supply_demand(engine, res)
        for b, mw in (exo_sched or {}).items():
            b = str(b)
            if mw >= 0:
                # A self-schedule is price-taking and not optimized: draw it gray,
                # with the bar AT the bus LMP (price=lmp, no bid) so it sits on the
                # dashed LMP line rather than showing a marginal-cost/rent gap.
                sup.setdefault(b, []).append({"unit_id": "self_schedule",
                                              "price": res.lmp[b], "volume": mw,
                                              "capacity": mw, "accepted_volume": mw,
                                              "color": _SELFSCHED,
                                              "price_label": False,
                                              "force_label": True,
                                              "lane": "self_schedule"})
                has_selfsched = True
            else:
                dem[b] = dem.get(b, 0.0) - mw

    if demand_segments is None and shed and sup_dem is None:
        demand_segments = shed_segments(res, dem) or None
    has_shed = any(v > 1e-6 for v in (getattr(res, "shed_by_bus", None) or {}).values())

    # flow tracing: on served demand when tracking shed, else on the drawn demand
    if sup_dem is None and shed:
        flow_dem = served_demand(res, dem)
    else:
        flow_dem = dem
    flows = _traced_flows(net, sup, flow_dem, demand_segments)

    # which buses to gray
    if dim_buses is not None:
        dim = {str(b) for b in dim_buses}
    elif highlight is not None:
        dim = {b for b in pt.buses if fp.fp_of(b) != highlight}
    else:
        dim = set()

    colors = dict(_BUS_COLORS() if bus_colors is None else bus_colors)
    for b in dim:
        colors[b] = _DIM

    # line colors: a managed line takes its footprint's color; gray a line
    # between grayed buses or assigned to a fully-grayed footprint.
    lcolors = fp.line_colors(pt)
    dim_fps = {name for name in fp.names if all(str(b) in dim for b in fp.defs[name])}
    for l in pt.lines:
        i = pt.line_idx[l]
        b0, b1 = pt.line_buses[i]
        if (b0 in dim and b1 in dim) or fp.line_assign.get(l) in dim_fps:
            lcolors[l] = _DIM

    gcolors = {name: (_DIM if name in dim_fps else fp.colors[name]) for name in fp.names}
    if constrained_lines is not None:
        binding = {str(l) for l in constrained_lines}
    else:
        binding = {l for l, mu in res.line_dual.items() if abs(mu) > 1e-3}

    fig, (ax_net, ax_circ) = plot_combined_letter(
        net, sup, dem,
        bus_colors=colors, dim_buses=dim,
        bus_lmps=dict(res.lmp) if bus_lmps is None else {str(k): float(v) for k, v in bus_lmps.items()},
        bus_net_mw=bus_net, network_show_lmp=network_show_lmp,
        line_flows=({l: res.flow_own[l] for l in pt.lines} if line_flows is None
                    else {str(k): float(v) for k, v in line_flows.items()}),
        line_widths=susceptance_widths(pt), line_colors=lcolors,
        constrained_lines=binding,
        flows=flows,
        clearing_price=res.energy_price,
        demand_segments=demand_segments,
        lmp_line=True, bus_groups=fp.groups(pt), group_colors=gcolors,
        show_group_labels=False,
        annotate_roles=annotate_roles, axis_key=axis_key,
        all_buses=pt.buses,
        sector_order=ring_order if ring_order is not None else _RING_ORDER(),
        bus_coords=bus_coords if bus_coords is not None else _COORDS(),
        center_bus=center_bus if center_bus is not None else _CENTER_BUS(),
        title_left=title_left, title_right=title_right, suptitle=suptitle,
        figsize=figsize, panel_ratios=panel_ratios,
    )

    _coords = bus_coords if bus_coords is not None else _COORDS()
    _exo = (exo_sched or {}) if exo_annotate else {}   # exo_annotate=False keeps the ring wedge, drops the network label
    # A balanced source/sink pair (two entries) would stack two labels above adjacent
    # buses; separate them horizontally and shorten the text. A single entry keeps the
    # centered-above placement (unchanged for the fundamentals notebook's lone import).
    _pair = len(_exo) == 2
    _cx = (sum(_coords[str(b)][0] for b in _exo) / 2.0) if _pair else None
    for b, mw in _exo.items():
        if _pair:
            _dx = 1 if _coords[str(b)][0] >= _cx else -1  # type: ignore[operator]  # _cx is set when _pair
            _xt, _ha = (34 * _dx, 18), ("left" if _dx > 0 else "right")
            _txt = f"self-sched {'+' if mw >= 0 else '−'}{abs(mw):.0f} MW"
        else:
            _xt, _ha = (0, 26), "center"
            _txt = f"{'self-schedule import' if mw >= 0 else 'self-schedule export'} {abs(mw):.0f} MW"
        ax_net.annotate(_txt, _coords[str(b)], fontsize=8, fontweight="bold",
                        color="#B03A2E", xytext=_xt, textcoords="offset points",
                        ha=_ha, bbox=dict(boxstyle="round", fc="#FDEDEC", ec="#B03A2E"))

    handles: list[mpatches.Patch | Line2D] = [
        mpatches.Patch(fc=fp.colors[name], ec="#555", label=f"{name} lines / band")
        for name in fp.names
    ]
    if any(fp.line_assign.get(l) is None for l in pt.lines):
        handles.append(mpatches.Patch(fc=_UNASSIGNED, ec="#555",
                                      label="Line assigned to neither footprint"))
    if has_selfsched:
        handles.append(mpatches.Patch(fc=_SELFSCHED, ec="#555",
                                      label="Self-schedule (price-taking; bar at LMP, not optimized)"))
    if has_shed:
        handles.append(mpatches.Patch(fc="#888", alpha=0.3, ec="#555",
                                      label="Load shed (unserved -- bus color, faint; LMP = SHED_PRICE)"))
    handles += [
        Line2D([0], [0], color="#E74C3C", lw=3, label="Congested line (binding)"),
        Line2D([0], [0], color="#17202A", lw=1.6, ls="--",
               label="Bus LMP (bar fill = marginal cost; gap = inframarginal rent)"),
    ]
    if legend_note:
        handles.append(mpatches.Patch(fc="white", ec="#555", label=legend_note))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, fontsize=7.5, framealpha=0.9, edgecolor="gray")
    return fig


def composite_figure(
    net: pypsa.Network,
    pt: PTDFData,
    fp: Footprints,
    engine: MarketEngine,
    res: EngineResult,
    *,
    suptitle: str | None = None,
    **kw: Unpack[_FootprintKw],
) -> Figure:
    """The unified two-panel composite (nothing grayed) — the set-up figure each
    downstream notebook prints. Thin wrapper over :func:`footprint_figure`."""
    return footprint_figure(net, pt, fp, engine, res, suptitle=suptitle, **kw)


def transfer_inset(
    ax: Axes,
    labels: tuple[str, str],
    tam: dict[str, float],
    surplus: dict[str, float],
    E: float,
    ebar: float,
    muT: float,
    colors: dict[str, str],
    tie_cap: float,
    title: str = "Transfers",
) -> None:
    """The "Transfers" inset, drawn from **explicit** values so the SAME transfer
    summary can overlay ANY composite (it never reads ``fp``/``res``).

    Two bubbles — ``labels = (left, right)`` — each at the network panel's
    bus-bubble scale (s = max(3·MW, 200), MW from ``tam[label]`` = the addressable
    market) and labeled with its net ``surplus[label]``. The connecting line
    carries the transfer ``E``/``ebar`` (flow/limit); its width is ``ebar``
    relative to ``tie_cap`` (thin = tight). When ``muT`` ≠ 0 the line turns red and
    ``|muT|`` prints below it. ``E`` is signed left→right (positive = left exports).
    """
    names = list(labels)
    size = {n: max(tam[n] * 3.0, 200.0) for n in names}
    fig = ax.figure
    pos = ax.get_position()
    w_pts = pos.width * fig.get_size_inches()[0] * 72.0  # type: ignore[union-attr]  # no SubFigure method
    h_pts = pos.height * fig.get_size_inches()[1] * 72.0  # type: ignore[union-attr]  # no SubFigure method
    r_pts = {n: (size[n] / pi) ** 0.5 for n in names}
    shrink = min(1.0, 0.30 * h_pts / max(r_pts.values()))
    size = {n: size[n] * shrink ** 2 for n in names}
    r_pts = {n: r * shrink for n, r in r_pts.items()}

    centers = {names[0]: (0.22, 0.45), names[1]: (0.78, 0.45)}
    binding = abs(muT) > 1e-3
    lw = 0.8 + 4.2 * min(1.0, ebar / tie_cap)
    color = "#E74C3C" if binding else "#7F8C8D"
    e = float(E)
    rx = {n: r_pts[n] / w_pts for n in names}
    x0 = centers[names[0]][0] + rx[names[0]]
    x1 = centers[names[1]][0] - rx[names[1]]
    a, b = ((x0, x1) if e >= 0 else (x1, x0))
    ax.annotate("", xy=(b, 0.45), xytext=(a, 0.45),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=8 + lw, shrinkA=0, shrinkB=0))
    for n, (cx, cy) in centers.items():
        ax.scatter(cx, cy, s=size[n], c=colors[n], ec="#333", lw=2, alpha=0.85, zorder=3)
        ax.text(cx, cy, f"{n}\n{surplus[n]:+,.0f}", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)
    ax.text(0.5, 0.70, f"{abs(e):.0f}/{ebar:.0f}",
            fontsize=9, color="#C0392B" if binding else "#7F8C8D",
            fontweight="bold" if binding else "normal", ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    if binding:
        ax.text(0.5, 0.16, f"${abs(muT):.2f}", fontsize=9.5, color="#C0392B",
                fontweight="bold", ha="center", va="center")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_net_dispatch(
    ax: Axes,
    fp: Footprints,
    res: EngineResult,
    ebar: float,
    engine: MarketEngine,
    tie_cap: float,
) -> None:
    """The "Transfers" inset for a footprint pair — reads the scheduled transfer
    and its interface dual off an EDAM-literal result (``res.transfers`` /
    ``res.transfer_duals``, signed as the first footprint's net export; zero for
    a clearing with no declared interface), computes the bubble sizes and
    surpluses from ``fp``, and defers the drawing to :func:`transfer_inset`."""
    names = fp.names
    sup, dem = to_supply_demand(engine, res)
    tam, surplus = {}, {}
    for name in names:
        bs = fp.defs[name]
        tam[name] = (sum(g.get("capacity", g["volume"]) for b in bs for g in sup.get(b, []))
                     + sum(dem.get(b, 0.0) for b in bs))
        surplus[name] = (sum(res.gen_by_bus.get(b, 0.0) for b in bs)
                         - sum(res.load_by_bus.get(b, 0.0) - res.shed_by_bus.get(b, 0.0)
                               for b in bs))
    k = tuple(sorted(names[:2]))
    t = float((res.transfers or {}).get(k, 0.0)) if getattr(res, "transfers", None) else 0.0  # type: ignore[arg-type]  # sorted() of a pair is a pair
    e = t if names[0] == k[0] else -t
    mu = (abs((res.transfer_duals or {}).get(k, 0.0))  # type: ignore[arg-type]  # sorted() of a pair is a pair
          if getattr(res, "transfer_duals", None) else 0.0)
    transfer_inset(ax, (names[0], names[1]), tam, surplus, e, ebar, mu,
                   fp.colors, tie_cap)


#: Default rectangle (figure fraction) for the Transfers inset in the network panel.
INSET_RECT = (0.005, 0.02, 0.19, 0.235)


def transfer_figure(
    net: pypsa.Network,
    pt: PTDFData,
    fp: Footprints,
    engine: MarketEngine,
    res: EngineResult,
    ebar: float,
    tie_cap: float,
    suptitle: str | None = None,
    **kw: Unpack[_FootprintKw],
) -> Figure:
    """:func:`footprint_figure` plus the "Transfers" inset in the network panel's
    lower-left, at the network panel's own bubble scale. Extra keyword arguments
    (e.g. ``node_net_mw``, ``highlight``) pass through to :func:`footprint_figure`."""
    fig = footprint_figure(net, pt, fp, engine, res, suptitle=suptitle, **kw)
    draw_net_dispatch(fig.add_axes(INSET_RECT), fp, res, ebar, engine, tie_cap)
    return fig


def _traced_flows(
    net: pypsa.Network,
    sup: _Blocks,
    dem: Loads,
    demand_segments: _Blocks | None,
) -> _Flows:
    """Gen-to-load chords. A balanced self-schedule, a lane block at its source
    and a lane segment at its sink, is its own chord from the block to the
    segment; the market's dispatch is traced over everything else. Without both
    ends, the block is traced with the rest, as before.

    A demand segment flagged ``unserved`` is drawn but not traced -- the
    unawarded part of a price-sensitive load bid, like the shed tail -- so the
    chords carry only the volume the market actually awarded."""
    dem = {b: float(v) for b, v in dem.items()}
    for b, segs in (demand_segments or {}).items():
        un = sum(float(s_["mw"]) for s_ in segs if s_.get("unserved"))
        if un > 1e-9:
            dem[b] = max(dem.get(b, 0.0) - un, 0.0)
    dem = {b: v for b, v in dem.items() if v > 1e-6}
    # A lane is paired by NAME: the block carrying ``lane == name`` at its source
    # bus and the segments carrying the same name at their sink buses. A lane
    # with both ends is its own set of chords, one per sink segment; a lane block
    # with no sinks (a one-legged self-schedule) is traced with the rest.
    lane_blocks = {}                         # name -> (bus, mw)
    for b, blocks in sup.items():
        for u in blocks:
            if u.get("lane"):
                lane_blocks[u["lane"]] = (b, float(u["volume"]))
    lane_sinks: dict[str, BusValues] = {}    # name -> {bus: mw}
    for b, segs in (demand_segments or {}).items():
        for s_ in segs:
            if s_.get("lane") and float(s_["mw"]) > 1e-6:
                lane_sinks.setdefault(s_["lane"], {})
                lane_sinks[s_["lane"]][b] = lane_sinks[s_["lane"]].get(b, 0.0) + float(s_["mw"])
    live = [n for n in lane_blocks if n in lane_sinks]
    if not live:
        return compute_ptdf_flows(net, sup, dem)
    sup_t = {b: [u for u in blocks if u.get("lane") not in live] for b, blocks in sup.items()}
    sup_t = {b: v for b, v in sup_t.items() if v}
    dem_t = dict(dem)
    lane_flows = []
    for n in live:
        src_b, _mw = lane_blocks[n]
        for b, v in lane_sinks[n].items():
            dem_t[b] = dem_t.get(b, 0.0) - v
            lane_flows.append((src_b, b, v, n))
    dem_t = {b: v for b, v in dem_t.items() if v > 1e-6}
    rest = compute_ptdf_flows(net, sup_t, dem_t) if sup_t and dem_t else []
    return list(rest) + lane_flows


def redispatch_bars(
    engine: MarketEngine,
    res: EngineResult,
    source: Bus,
    mw: float,
    *,
    sup_dem: _SupDem | None = None,
    curves: _Curves | None = None,
    hatch: str = "////",
) -> tuple[_SupDem, None]:
    """Build ``(sup_dem, None)`` marking a market's OWN incremental dispatch at
    ``source`` -- the megawatts it moves in real time above the dispatch a
    day-ahead award assumed.

    The increment is the LAST ``mw`` of the dispatched stack: the slice of the
    marginal unit's own offer curve between the two dispatches. That unit's block
    is cut at the lower point and the same curve continues as a second block,
    hatched, so the supply curve on the ring is unchanged and the hatch is the
    segment of it the market moved through. ``curves`` is the ``{unit: (a, b)}``
    the clearing used; a unit without one is flat.

    The increment stays in the BUS's own colour and carries no lane, because it
    is the market's own generation serving its own load. A counterparty's
    position is not drawn here: it belongs at the node that counterparty
    transacted at, as a block of its own (see :func:`bid_bars`).
    """
    S, src = float(mw), str(source)
    sup, dem = (sup_dem if sup_dem is not None else to_supply_demand(engine, res))
    curves = curves or {}
    out: list[dict[str, Any]]
    remaining, out = S, []
    for u in reversed(sup.get(src, [])):          # top of the stack first
        acc = float(u.get("accepted_volume", u["volume"]))  # type: ignore[arg-type]  # get with an Any default reads as maybe None
        take = min(remaining, acc) if remaining > 1e-9 else 0.0
        if take <= 1e-9:
            out.insert(0, u)
            continue
        a, b = curves.get(u["unit_id"], (float(u["price"]), 0.0))
        lo = acc - take                           # the dispatch before the move
        cap = float(u.get("capacity", u["volume"]))  # type: ignore[arg-type]  # get with an Any default reads as maybe None
        base = dict(u)
        base["accepted_volume"] = base["volume"] = base["capacity"] = lo
        if b:
            base["mc0"], base["mc_slope"] = float(a), float(b)
        inc = dict(u)
        inc.update({"unit_id": f"{u['unit_id']} (redispatch)",
                    "price": float(a + b * lo), "volume": cap - lo,
                    "capacity": cap - lo, "accepted_volume": take,
                    "hatch": hatch, "price_label": False, "force_label": True})
        if b:
            inc["mc0"], inc["mc_slope"] = float(a + b * lo), float(b)
        out.insert(0, inc)
        if lo > 1e-9:
            out.insert(0, base)
        remaining -= take
    sup[src] = out
    return (sup, dem), None

def bid_bars(
    engine: MarketEngine,
    res: EngineResult,
    node: Bus,
    kind: str,
    bid_mw: float,
    awarded: float,
    price: float,
    *,
    sup_dem: _SupDem | None = None,
    color: str = _SELFSCHED,
    solid: float = 0.55,
    faint: float = 0.15,
) -> tuple[_SupDem, _Blocks | None]:
    """Build ``(sup_dem, demand_segments)`` for a PRICE-SENSITIVE bid at ``node``.

    A bid is drawn like any other resource or load, in the self-schedule gray:
    the bid quantity ``bid_mw`` is the offered envelope and ``awarded`` is the
    part the market took, so a fully awarded bid reads solid and a partly
    awarded one reads part solid, part faded. Unlike a price-taking
    self-schedule the block keeps its price label, because a bid has a price.

    1. ``kind='resource'`` puts the bid on the supply side at ``node``, at bar
       height ``price``, its capacity the bid quantity and its accepted volume
       the award, which is the same three-level fill every other unit gets.
    2. ``kind='load'`` puts it on the demand side, the awarded part at ``solid``
       alpha and the unawarded part at ``faint``. The unawarded part is flagged
       ``unserved``, so :func:`_traced_flows` draws it without tracing it.

    The award is traced into the chords exactly like any other volume. Only the
    balanced self-schedule of :func:`self_schedule_bars` gets a chord lane of its
    own, because only there is there a gray block at BOTH ends to join.
    """
    nd, Q, W, P = str(node), float(bid_mw), float(awarded), float(price)
    sup, dem = (sup_dem if sup_dem is not None else to_supply_demand(engine, res))
    if kind not in ("resource", "load"):
        raise ValueError("kind must be 'resource' or 'load'")
    segs: _Blocks | None = None
    if kind == "resource":
        sup.setdefault(nd, []).insert(0, {
            "unit_id": "bid", "price": P, "volume": Q, "capacity": Q,
            "accepted_volume": W, "color": color, "force_label": True})
    else:
        rest = dem.get(nd, 0.0)
        dem[nd] = rest + Q
        segs = {nd: []}
        if W > 0.01:
            segs[nd].append({"mw": W, "price": P, "color": color, "alpha": solid})
        if Q - W > 0.01:
            segs[nd].append({"mw": Q - W, "price": P, "color": color,
                             "alpha": faint, "unserved": True})
        if rest > 0.01:
            segs[nd].append({"mw": rest})
    return (sup, dem), segs


def self_schedule_bars(
    engine: MarketEngine,
    res: EngineResult,
    source: Bus | None = None,
    sink: Bus | None = None,
    mw: float = 0.0,
    *,
    sup_dem: _SupDem | None = None,
    source_price: float | None = None,
    sink_price: float | None = None,
    sink_rest_price: float | None = None,
    served_by_source: bool = False,
    add_sink_demand: bool = True,
    color: str = _SELFSCHED,
) -> tuple[_SupDem, _Blocks | None]:
    """Build ``(sup_dem, demand_segments)`` that draw a price-taking self-schedule as a
    distinct gray block — the single, shared convention across the series (101's one-sided
    injection, 201's balanced gen→load, 205's cross-market trade), cleaner than the
    automatic ``exo_sched`` wedge. Pass the returned ``sup_dem`` / ``demand_segments``
    straight to :func:`footprint_figure` / :func:`transfer_figure` (they take over the
    bars, so drop ``exo_sched`` on the ring; the network panel's net-MW labels come from
    ``res.injection`` and already include the schedule).

    The scheduled ``mw`` is drawn as a fixed block at the BOTTOM of the ``source`` bus's
    supply stack (priced at ``source_price``, default $\\lambda_{source}$) and/or as the
    first slice of the ``sink`` bus's demand bar (priced at ``sink_price``, default
    $\\lambda_{sink}$; the remainder at ``sink_rest_price``, default the same). Give only
    ``source`` for a one-sided injection (the 101 case → a single clean wedge), or both for
    a balanced schedule. ``source``/``sink`` default to ``None`` (that leg is skipped).

    A price-taking block carries no offer, so neither leg is labelled with a price: the
    supply block sets ``price_label: False`` and the demand slice is labelled in MW like
    every other demand segment. The block's height is the node's price and its volume is
    printed inside it. Both legs carry ``lane: 'self_schedule'``, so with a source AND a
    sink the ring draws the schedule as its own gray chord from the block to the slice
    and traces the market's dispatch over the rest (:func:`_traced_flows`).

    Options for the cross-market case:
    ``served_by_source=True`` first backs the source unit's priced stack down by ``mw``
    (the source *generator* is dispatched to serve the schedule); the default adds the
    block on top (a fixed exogenous injection separate from the cleared generation).
    ``add_sink_demand=False`` treats the schedule as part of the sink's EXISTING load
    (splits the bar without growing it) rather than adding ``mw`` on top. Pass an existing
    ``sup_dem`` to modify it in place (e.g. bars already carrying a seam interchange)
    instead of rebuilding from ``engine``/``res``. ``source_price``/``sink_price`` let each
    leg settle at another market's LMP (the four-leg cross-valuation).
    """
    S = float(mw)
    sup, dem = (sup_dem if sup_dem is not None else to_supply_demand(engine, res))
    segs = None
    if source is not None and S > 1e-9:
        src = str(source)
        sp = res.lmp[src] if source_price is None else float(source_price)
        if served_by_source:                          # back the source unit down by S first
            taken = 0.0
            for u in sup.get(src, []):
                take: float = min(S - taken, u.get("accepted_volume", u["volume"]))  # type: ignore[type-var,assignment]  # as below
                u["accepted_volume"] = u.get("accepted_volume", u["volume"]) - take  # type: ignore[operator]  # get with an Any default reads as maybe None
                u["volume"] = u["capacity"] = u["capacity"] - take
                taken += take
            sup[src] = [u for u in sup.get(src, []) if u["volume"] > 1e-9]
        sup.setdefault(src, []).insert(0, {"unit_id": "self_schedule", "price": sp,
            "volume": S, "capacity": S, "accepted_volume": S, "color": color,
            "price_label": False, "force_label": True, "lane": "self_schedule"})
    if sink is not None and S > 1e-9:
        snk = str(sink)
        kp = res.lmp[snk] if sink_price is None else float(sink_price)
        rp = kp if sink_rest_price is None else float(sink_rest_price)
        if add_sink_demand:
            dem[snk] = dem.get(snk, 0.0) + S
        total = dem.get(snk, 0.0)
        segs = {snk: [{"mw": min(S, total), "price": kp, "color": color,
                       "lane": "self_schedule"}]
                     + ([{"mw": total - S, "price": rp}] if total - S > 0.01 else [])}
    return (sup, dem), segs


def draw_rights_arcs(
    ax: Axes,
    rights: Sequence[dict[str, Any]],
    coords: _Coords | None = None,
    *,
    tier_colors: dict[str, str] | None = None,
    rad: float = 0.22,
    label: bool = True,
    atc_caps: Sequence[float | None] | None = None,
    label_mw: bool = True,
    label_fontsize: float = 7.5,
) -> Axes:
    """Overlay sold point-to-point transmission rights as directed POR->POD arcs.

    ``rights`` is a list of dicts ``{'source','sink','mw', 'tier'?, 'honored'?}``:
    a curved arrow from the source bus to the sink bus, width growing with ``mw``,
    colored by ``tier`` ('inter' = inter-BA path right / WECC backbone, 'intra' =
    intra-BA right) via ``tier_colors`` (default :data:`RIGHT_TIER_COLORS`). A
    ``honored=False`` right is drawn dashed (a curtailed / unfunded right).

    Two optional keys (304a, 2026-10-13; absent keys draw exactly as before):
    ``tag_id`` prefixes the arc's label with the tag's identifier, so the arc
    maps to a row of a tag table; ``mw_cut`` draws a partial curtailment, the
    solid arc at the megawatts kept (``mw - mw_cut``) and a dashed arc beside it
    at the megawatts cut, labelled ``"40.9 -> 34.6 MW"``. A tag coloured by its
    NERC priority passes ``tier=<code>`` with ``tier_colors=PRIORITY_COLORS``. Drawn
    ON TOP of an existing network panel (e.g. the ``ax`` returned by
    :func:`rights_figure` / ``plot_network_topology``) so the financial rights sit
    over the physical lines they load.

    ``atc_caps`` (the ATC layer) is an optional list aligned with ``rights``; when
    given, each arc's label reads ``MW / ATC`` -- the scheduled quantity against the
    path's standalone transfer capability -- so an arc near its own cap is visible.

    A label that lands on a bus marker, another text, a drawn line, a flow arrow
    or an arc slides along its arc, or out along the chord normal or into the
    bulge, when the figure draws; a label that clears keeps its place
    (:func:`nodal_plot.add_arc_label_clearance`, 2026-11-07). A flow label of
    the panel that an arc crosses moves off it the same way.
    """
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    coords = _COORDS() if coords is None else coords
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    for ri, r in enumerate(rights):
        s, k, mw = str(r["source"]), str(r["sink"]), float(r["mw"])
        col = tcol.get(r.get("tier", "inter"), "#117A65")
        honored = r.get("honored", True)
        cut = float(r.get("mw_cut", 0.0) or 0.0)
        p0, p1 = np.array(coords[s]), np.array(coords[k])
        if cut > 1e-9:
            kept = max(0.0, mw - cut)
            if kept > 1e-9:
                ax.add_patch(FancyArrowPatch(
                    p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",  # type: ignore[arg-type]  # array, stub says tuple
                    mutation_scale=18, lw=1.6 + 3.6 * min(1.0, kept / 150.0),
                    color=col, ls="-", alpha=0.9, zorder=6, shrinkA=14, shrinkB=16))
            ax.add_patch(FancyArrowPatch(
                p0, p1, connectionstyle=f"arc3,rad={rad * 1.45}", arrowstyle="-|>",  # type: ignore[arg-type]  # array, stub says tuple
                mutation_scale=14, lw=1.2 + 3.6 * min(1.0, cut / 150.0), color=col,
                ls=(0, (4, 3)), alpha=0.75, zorder=6, shrinkA=14, shrinkB=16))
        else:
            ax.add_patch(FancyArrowPatch(
                p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",  # type: ignore[arg-type]  # array, stub says tuple
                mutation_scale=18, lw=1.6 + 3.6 * min(1.0, mw / 150.0), color=col,
                ls="-" if honored else (0, (4, 3)), alpha=0.9 if honored else 0.7,
                zorder=6, shrinkA=14, shrinkB=16))
        if label:
            mid = 0.5 * (p0 + p1)
            d = p1 - p0
            # matplotlib's arc3 bows toward (dy, -dx) by 0.5*rad*|chord|; put the
            # label on that arc midpoint (so it scales with chord length and sits on
            # the curve), nudged a touch further out so the box clears the line.
            bulge = np.array([d[1], -d[0]])
            bunit = bulge / (np.linalg.norm(bulge) + 1e-9)
            tid = r.get("tag_id")
            path = f"{tid} {s}→{k}" if tid else f"{s}→{k}"
            if cut > 1e-9:
                tag = (f"{path}\n{mw:.1f} → {max(0.0, mw - cut):.1f} MW"
                       if label_mw else f"{path} (cut {cut:.1f})")
            elif label_mw:
                cap = None if atc_caps is None else atc_caps[ri]
                mwtag = f"{mw:.0f} MW" if cap is None else f"{mw:.0f} / {cap:.0f} MW"
                if tid and cap is None:
                    mwtag = f"{mw:.1f} MW"
                tag = f"{path}\n{mwtag}" + ("" if honored else "\n(curtailed)")
            else:
                # path only; the MW / ATC live in the table beside the graph
                tag = path + ("" if honored else " (curt.)")
            ann = ax.annotate(tag, mid + 0.5 * rad * bulge + 0.30 * bunit,
                              ha="center", va="center", fontsize=label_fontsize,
                              color=col, fontweight="bold", zorder=8,
                              bbox={"boxstyle": "round,pad=0.2", "fc": "white",
                                    "ec": col, "lw": 1, "alpha": 0.92})
            # Measured when the axes draws, against the chips and flow labels as
            # they are finally placed.
            nodal_plot.add_arc_label_clearance(ax, ann, p0, p1, mid + rad * bulge,
                                               bunit, coords)
    return ax


def induced_flow_panel(
    net: pypsa.Network,
    pt: PTDFData,
    source: Bus,
    sink: Bus,
    mw: float,
    *,
    line: Line,
    ax: Axes | None = None,
    coords: _Coords | None = None,
    bus_colors: dict[Bus, str] | None = None,
    line_widths: LineValues | None = None,
    tag_id: str | None = None,
    priority: str | None = None,
    tier_colors: dict[str, str] | None = None,
    cut_mw: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.5, 7.5),
    threshold: float = 0.05,
    direction: int = +1,
) -> tuple[Figure, Axes]:
    """The flow one point-to-point transaction places on the network (304a).

    A tag of ``mw`` MW from ``source`` to ``sink`` is the balanced injection pair
    ``+mw`` at the source and ``-mw`` at the sink, so it places
    ``mw * (SF_{src,m} - SF_{snk,m})`` on every line ``m`` (the transfer
    distribution factor times the megawatts, ``eccetf_futurestate2024`` p. 7;
    the same quantity :func:`seam_views.input_view` draws for a balanced
    schedule). The panel is the network topology with that pair on the nodes
    and those flows on the lines, the tag drawn as its arc (coloured by
    ``priority`` through ``tier_colors``, default :data:`PRIORITY_COLORS`), and
    the constraint ``line`` drawn in :data:`INDUCED_LOADING` where the tag loads
    it and :data:`INDUCED_COUNTER` where it counterflows it, with a box carrying
    the tag's TDF on that line, its impact, and whether it clears the 5 percent
    eligibility screen. ``cut_mw`` (a curtailment) draws the arc as kept plus
    cut and states the backstop leg: the cut megawatts replaced at the sink bus.

    ``direction`` is the loading direction ``s`` of the constraint, as in
    :func:`atc.wlr_assign`: ``+1`` (the default) when the superposed flow runs
    in the line's reference direction (``from`` to ``to``), as on 304a's
    constraint, and ``-1`` when it runs ``to`` to ``from``. The loading test and
    the eligibility screen use ``direction * TDF``. Returns ``(fig, ax)``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from seams_engine import susceptance_widths

    coords = _COORDS() if coords is None else coords
    s, k, S = str(source), str(sink), float(mw)
    a = pt.ptdf[:, pt.bus_idx[s]] - pt.ptdf[:, pt.bus_idx[k]]
    flows = {l: float(S * a[pt.line_idx[l]]) for l in pt.lines}
    li = pt.line_idx[str(line)]
    tdf = float(a[li])
    sgn = 1.0 if direction >= 0 else -1.0
    impact = S * tdf
    loads = sgn * impact > 0
    lcol = INDUCED_LOADING if loads else INDUCED_COUNTER
    lcolors = {l: "#CBD0D3" for l in pt.lines}
    lcolors[str(line)] = lcol
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]  # Figure unless on a SubFigure
    plot_network_topology(
        net, bus_colors=_BUS_COLORS() if bus_colors is None else bus_colors,
        bus_coords=coords, bus_net_mw={s: +S, k: -S}, lmp_only=True,
        line_flows=flows, line_colors=lcolors,
        line_widths=susceptance_widths(pt) if line_widths is None else line_widths,
        flow_labels=True, constrained_lines=set(), ax=ax,
        title=title or f"{tag_id + ' ' if tag_id else ''}{s}→{k}: induced flow")
    tcol = PRIORITY_COLORS if tier_colors is None else tier_colors
    arc = {"source": s, "sink": k, "mw": S, "tier": priority or "7-F"}
    if tag_id:
        arc["tag_id"] = tag_id
    if cut_mw:
        arc["mw_cut"] = float(cut_mw)
    draw_rights_arcs(ax, [arc], coords, tier_colors=tcol, label_mw=True)
    b0, b1 = pt.line_buses[li]
    elig = sgn * tdf >= threshold
    kind = "loading" if loads else "counterflow"
    txt = (f"{line} ({b0}-{b1})\nTDF {tdf:+.3f}\nimpact {impact:+.1f} MW ({kind})\n"
           + ("eligible (TDF ≥ 5%)" if elig else "not eligible"))
    if cut_mw:
        txt += f"\ncut {cut_mw:.1f} MW; backstop +{cut_mw:.1f} MW at bus {k}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8, color=lcol,
            fontweight="bold", va="bottom", ha="left", zorder=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=lcol, lw=1.2,
                      alpha=0.95))
    return fig, ax


def net_injection_flow_panel(
    net: pypsa.Network,
    pt: PTDFData,
    injections: BusValues,
    *,
    line: Line,
    ax: Axes | None = None,
    coords: _Coords | None = None,
    bus_colors: dict[Bus, str] | None = None,
    line_widths: LineValues | None = None,
    label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.5, 7.5),
    threshold: float = 0.05,
    direction: int = +1,
    tol: float = 1e-6,
) -> tuple[Figure, Axes]:
    """The flow a balanced set of net injections places on the network (304a).

    The companion of :func:`induced_flow_panel` for a component with no single
    source and sink: a market area's untagged generation-to-load read as its net
    injections and withdrawals ``x_n`` (``{bus: MW}``, generation positive), which
    place ``sum_n SF_{n,m} x_n`` on every line ``m``. The injections must balance,
    so the flows do not depend on the reference bus, and the impact equals
    :func:`atc.wlr_gtl_component`'s on the same generation and load; unbalanced
    injections raise ``ValueError``. The panel labels each bus with its ``x_n``, draws the flows,
    colours the constraint ``line`` :data:`INDUCED_LOADING` where the injections
    load it and :data:`INDUCED_COUNTER` where they counterflow it, and boxes the
    impact and the impact per injected megawatt (the generation total), the ratio
    :func:`atc.wlr_gtl_component` passes to the 5 percent screen for a GTL
    component (a Lab reading; the GTL denominator is not public). ``direction``
    is the loading direction ``s``, as in :func:`induced_flow_panel`. Returns
    ``(fig, ax)``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from seams_engine import susceptance_widths

    x = {str(b): float(v) for b, v in injections.items()}
    if abs(sum(x.values())) > tol:
        raise ValueError(f"the net injections must balance (sum {sum(x.values()):+.6f} MW)")
    coords = _COORDS() if coords is None else coords
    inj = np.zeros(len(pt.buses))
    for b, v in x.items():
        inj[pt.bus_idx[b]] += v
    fl = pt.ptdf @ inj
    flows = {l: float(fl[pt.line_idx[l]]) for l in pt.lines}
    li = pt.line_idx[str(line)]
    impact = float(fl[li])
    g_tot = sum(v for v in x.values() if v > 0)
    ratio = impact / g_tot if g_tot > 0 else 0.0
    sgn = 1.0 if direction >= 0 else -1.0
    loads = sgn * impact > 0
    lcol = INDUCED_LOADING if loads else INDUCED_COUNTER
    lcolors = {l: "#CBD0D3" for l in pt.lines}
    lcolors[str(line)] = lcol
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]  # Figure unless on a SubFigure
    plot_network_topology(
        net, bus_colors=_BUS_COLORS() if bus_colors is None else bus_colors,
        bus_coords=coords, bus_net_mw=x, lmp_only=True,
        line_flows=flows, line_colors=lcolors,
        line_widths=susceptance_widths(pt) if line_widths is None else line_widths,
        flow_labels=True, constrained_lines=set(), ax=ax,
        title=title or f"{label + ': ' if label else ''}net injections, induced flow")
    b0, b1 = pt.line_buses[li]
    kind = "loading" if loads else "counterflow"
    elig = sgn * ratio >= threshold
    txt = (f"{label + chr(10) if label else ''}{line} ({b0}-{b1})\n"
           f"impact {impact:+.1f} MW ({kind})\nimpact / injected MW {ratio:+.3f}\n"
           + (f"eligible (≥ {threshold:.0%})" if elig else "not eligible"))
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8, color=lcol,
            fontweight="bold", va="bottom", ha="left", zorder=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=lcol, lw=1.2,
                      alpha=0.95))
    return fig, ax


def rights_figure(
    net: pypsa.Network,
    pt: PTDFData,
    rights: Sequence[dict[str, Any]],
    *,
    fp: Footprints | None = None,
    ties: Iterable[Line] | None = None,
    coords: _Coords | None = None,
    monitored: _Monitored = "all",
    title: str | None = None,
    bus_colors: dict[Bus, str] | None = None,
    tier_colors: dict[str, str] | None = None,
    line_colors: dict[Line, str] | None = None,
    line_widths: LineValues | None = None,
    show_atc: bool = False,
    show_sft: bool = True,
    supply_by_bus: _Blocks | None = None,
    demand_by_bus: Loads | None = None,
    bus_net_mw: BusValues | None = None,
    col_labels: list[str] | None = None,
    atc_values: Iterable[float | None] | None = None,
    pct_values: Sequence[float] | None = None,
    font_scale: float = 1.0,
) -> Figure:
    """The standard "transmission rights" view: network panel + rights table.

    Two panels side by side so the graph stays uncluttered and the numbers are
    readable:

    * **Left -- the network.** Each ``rights`` entry (see :func:`draw_rights_arcs`)
      is a POR->POD arc, width growing with MW, colored by tier (inter-BA gold-path
      vs intra-BA), labeled by **path only** (``2->7``). The physical, *superposed*
      award flow rides on the lines as a direction arrow + ``flow / limit`` (the
      share of capacity); any line the rights overload (``show_sft``) turns **red**
      -- the simultaneous-feasibility test failing, path-by-path rights colliding on
      the shared grid.
    * **Right -- the rights table.** One row per right: ``Right`` (path, colored to
      match its arc), scheduled ``MW``, and -- with ``show_atc`` -- the path's
      standalone ``ATC`` (:func:`atc.ttc`) and the ``%ATC`` booked.

    Returns the figure. (Layers 3 parallel-flow and 4 transfer-cutset are added as
    the series reaches the notebooks that teach them.)
    """
    import atc
    import matplotlib.pyplot as plt
    from seams_engine import susceptance_widths

    # Line WIDTH encodes susceptance b_m = 1/x_m (the 101 convention): a wider line is more
    # "slippery", so it wants more of any given flow -- the intuition for parallel flow.
    line_widths = susceptance_widths(pt) if line_widths is None else line_widths
    coords = _COORDS() if coords is None else coords
    tielist = list(ties) if ties is not None else (list(fp.ties) if fp is not None else [])
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    aw = [atc.Award(r["source"], r["sink"], r["mw"]) for r in rights]

    overl = set(atc.overloaded_lines(pt, aw, monitored)) if show_sft else set()
    # Base line color: a caller-supplied map (e.g. by balancing authority) if given, else the
    # gold WECC-path / gray scheme. An overload (SFT failing) flags red and overrides either.
    def _base(l: Line) -> str:
        if line_colors is not None:
            return line_colors.get(l, "#CBD0D3")
        return _WECC_PATH if l in tielist else "#CBD0D3"
    lcolors = {l: ("#C0392B" if l in overl else _base(l)) for l in pt.lines}

    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.02)
    axn = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")

    # left: the network. The superposed-award flow rides on the lines (direction +
    # flow/limit, red where the SFT fails); the rights are arcs labeled by path
    # only -- the MW / ATC live in the table on the right.
    plot_network_topology(
        net, bus_colors=_BUS_COLORS() if bus_colors is None else bus_colors, bus_coords=coords,
        supply_by_bus=supply_by_bus, demand_by_bus=demand_by_bus, bus_net_mw=bus_net_mw,
        lmp_only=bus_net_mw is not None,
        line_flows=atc.flow_dict(pt, aw), line_colors=lcolors, line_widths=line_widths,
        flow_labels=True, constrained_lines=overl, ax=axn, title=title or "Scheduled bilateral rights",
        node_number_fontsize=10 * font_scale, annot_fontsize=6 * font_scale,
        flow_label_fontsize=9 * font_scale, title_fontsize=13 * font_scale,
        net_label_offset=28.0 / font_scale,
        title_pad=(None if font_scale == 1.0 else 6.0 + 45.0 * (font_scale - 1.0)))
    draw_rights_arcs(axn, rights, coords, tier_colors=tier_colors, label_mw=False,
                     label_fontsize=7.5 * font_scale)

    # right: the rights table -- path, scheduled MW, the ATC, and the % booked. ``atc_values``
    # (e.g. a BA-level ATC the caller computed) overrides the standalone per-path TTC.
    if atc_values is not None:
        caps: list[float | None] = list(atc_values)
    elif show_atc:
        caps = [atc.ttc(pt, r["source"], r["sink"], monitored=monitored)[0] for r in rights]
    else:
        caps = [None] * len(rights)
    cells, pathcols = [], []
    for idx, (r, cap) in enumerate(zip(rights, caps)):
        path = f"{r['source']}→{r['sink']}"
        pathcols.append(tcol.get(r.get("tier", "inter"), "#117A65"))
        if cap is not None:
            pct = pct_values[idx] if pct_values is not None else 100 * r["mw"] / cap
            cells.append([path, f"{r['mw']:g}", f"{cap:.0f}", f"{pct:.0f}%"])
        else:
            cells.append([path, f"{r['mw']:g}"])
    if col_labels is not None:
        collab = col_labels
    else:
        collab = ["Right", "MW", "ATC", "%ATC"] if show_atc else ["Right", "MW"]
    cw = [0.30, 0.30, 0.20, 0.20] if len(collab) == 4 else None
    tbl = axt.table(cellText=cells, colLabels=collab, loc="center", cellLoc="center",
                    bbox=[0.0, 0.14, 1.0, 0.72], colWidths=cw)  # type: ignore[arg-type]  # bounds, stub says Bbox
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    nrow = len(cells) + 1
    for (rr, cc), cell in tbl.get_celld().items():        # taller header row
        cell.set_edgecolor("#999")
        cell.set_height((1.8 if rr == 0 else 1.0) / (nrow + 0.8))
    for j in range(len(collab)):                          # header row
        tbl[(0, j)].set_text_props(fontweight="bold", color="white")
        tbl[(0, j)].set_facecolor("#34495E")
    for i, col in enumerate(pathcols):                    # path cell colored to its arc
        tbl[(i + 1, 0)].get_text().set_color(col)
        tbl[(i + 1, 0)].get_text().set_fontweight("bold")
    return fig


def rights_payoff_figure(
    net: pypsa.Network,
    pt: PTDFData,
    rights: Sequence[dict[str, Any]],
    res: EngineResult,
    *,
    fp: Footprints | None = None,
    coords: _Coords | None = None,
    bus_colors: dict[Bus, str] | None = None,
    line_colors: dict[Line, str] | None = None,
    line_widths: LineValues | None = None,
    tier_colors: dict[str, str] | None = None,
    title: str | None = None,
    supply_by_bus: _Blocks | None = None,
    demand_by_bus: Loads | None = None,
    bus_net_mw: BusValues | None = None,
    font_scale: float = 1.0,
) -> Figure:
    """The **settlement** companion to :func:`rights_figure`: the same rights drawn
    over the *unified-clearing* network --- nodal LMPs at every bus and the
    dispatch flows on every line --- with each right's **FTR payoff** in the table.

    Where :func:`rights_figure` asks *do these rights fit?* (the SFT / ATC view),
    this asks *what do they pay?* A financial right from POR ``s`` to POD ``k`` for
    ``q`` MW settles at the locational price spread it spans,
    ``payoff = q * (lmp_k - lmp_s)`` --- positive where it delivers into a dearer
    bus, negative where it counter-flows into a cheaper one. The table breaks that
    out as **volume**, **price difference**, and **total payoff**, and totals to the
    congestion rent when the set reconstructs the merchandising surplus.

    Parameters mirror :func:`rights_figure`; ``res`` is the
    :class:`~seams_engine.EngineResult` of the unified clearing (its ``lmp``,
    ``flow_own`` and ``line_dual`` supply the prices, flows and binding lines).
    Returns the figure.
    """
    import matplotlib.pyplot as plt
    from seams_engine import susceptance_widths

    coords = _COORDS() if coords is None else coords
    line_widths = susceptance_widths(pt) if line_widths is None else line_widths
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    lmp = res.lmp
    binding = {l for l in pt.lines if abs(res.line_dual.get(l, 0.0)) > 1e-3}

    def _base(l: Line) -> str:
        if line_colors is not None:
            return line_colors.get(l, "#CBD0D3")
        return "#CBD0D3"
    lcolors = {l: ("#C0392B" if l in binding else _base(l)) for l in pt.lines}

    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.02)
    axn = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")

    # left: the priced network -- nodal LMPs + the unified dispatch flows, with the
    # rights as POR->POD arcs on top (binding lines red).
    plot_network_topology(
        net, bus_colors=_BUS_COLORS() if bus_colors is None else bus_colors,
        bus_coords=coords, bus_lmps=lmp,
        supply_by_bus=supply_by_bus, demand_by_bus=demand_by_bus, bus_net_mw=bus_net_mw,
        lmp_only=bus_net_mw is not None,
        line_flows={l: float(res.flow_own[l]) for l in pt.lines},
        line_colors=lcolors, line_widths=line_widths, constrained_lines=binding,
        flow_labels=True, ax=axn, title=title or "Rights settled at the unified prices",
        node_number_fontsize=10 * font_scale, annot_fontsize=6 * font_scale,
        flow_label_fontsize=9 * font_scale, title_fontsize=13 * font_scale,
        net_label_offset=28.0 / font_scale,
        title_pad=(None if font_scale == 1.0 else 6.0 + 45.0 * (font_scale - 1.0)))
    draw_rights_arcs(axn, rights, coords, tier_colors=tier_colors, label_mw=False,
                     label_fontsize=7.5 * font_scale)

    # right: the payoff table -- volume, price difference, total payoff.
    cells, pathcols, total = [], [], 0.0
    for r in rights:
        s, k, q = str(r["source"]), str(r["sink"]), float(r["mw"])
        dp = lmp[k] - lmp[s]; pay = q * dp; total += pay
        pathcols.append(tcol.get(r.get("tier", "inter"), "#117A65"))
        cells.append([f"{s}→{k}", f"{q:g}", f"{dp:+.2f}", f"{pay:+,.0f}"])
    cells.append(["total", "", "", f"{total:+,.0f}"])
    collab = ["SFT Right\n(Path)", "SFT Volume\n(MW)", "Price diff\n($/MW)", "Payoff\n($)"]
    tbl = axt.table(cellText=cells, colLabels=collab, loc="center", cellLoc="center",
                    bbox=[0.0, 0.14, 1.0, 0.72], colWidths=[0.29, 0.29, 0.24, 0.18])  # type: ignore[arg-type]  # bounds, stub says Bbox
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    nrow = len(cells) + 1                                  # header + data + total
    for (r, c), cell in tbl.get_celld().items():  # type: ignore[assignment]  # taller header row
        cell.set_edgecolor("#999")
        cell.set_height((1.8 if r == 0 else 1.0) / (nrow + 0.8))
    for j in range(len(collab)):
        tbl[(0, j)].set_text_props(fontweight="bold", color="white")
        tbl[(0, j)].set_facecolor("#34495E")
    for i, col in enumerate(pathcols):
        tbl[(i + 1, 0)].get_text().set_color(col)
        tbl[(i + 1, 0)].get_text().set_fontweight("bold")
    for j in range(len(collab)):                          # total row
        tbl[(len(cells), j)].set_text_props(fontweight="bold")
    return fig


def self_schedule_figure(
    summary: dict[str, Any], *, suptitle: str | None = None
) -> Figure:
    """Two-panel view of the self-schedule incentive, from a
    :func:`revenue_allocation.self_schedule_ledger` ``summary``.

    Left -- generator output by bus, economic vs self-schedule: the merit-order
    inversion (the cheap exporter backs down so the dearer self-scheduled unit can
    run within the capped export). Right -- the conservation decomposition: the
    rebate the rule hands the self-schedule splits exactly into the owner's private
    gain and pure deadweight loss, and the rent pool shrinks by the whole rebate.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    s = summary
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # ── Panel A: dispatch, economic vs self-schedule ────────────────────────
    buses = [b for b in sorted(set(s["out_econ"]) | set(s["out_self"]),
                               key=lambda b: s["bus_cost"].get(b, 0.0))
             if s["out_econ"].get(b, 0.0) > 0.5 or s["out_self"].get(b, 0.0) > 0.5]
    x = np.arange(len(buses)); w = 0.38
    e = [s["out_econ"].get(b, 0.0) for b in buses]
    f = [s["out_self"].get(b, 0.0) for b in buses]
    axL.bar(x - w / 2, e, w, label="economic dispatch", color="#BDC3C7", ec="#7F8C8D")
    cols = ["#C0392B" if b == s["source"]                       # forced-on dear unit
            else "#2E86C1" if s["out_self"].get(b, 0.0) < s["out_econ"].get(b, 0.0) - 0.5
            else "#7F8C8D" for b in buses]                      # backed-down cheap unit
    axL.bar(x + w / 2, f, w, label="with self-schedule", color=cols, ec="#34495E")
    for xi, b in zip(x, buses):
        axL.annotate(f"${s['bus_cost'].get(b, 0):.0f}", (xi, max(s["out_econ"].get(b, 0.0),
                     s["out_self"].get(b, 0.0))), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8, color="#34495E")
    axL.set_xticks(x); axL.set_xticklabels([f"bus {b}" for b in buses])
    axL.set_ylabel("output (MW)")
    axL.set_title(f"Dispatch: self-scheduling the ${s['src_cost']:.0f} unit "
                  f"displaces the cheap unit", fontsize=10)
    axL.legend(fontsize=8, framealpha=0.9)
    axL.spines[["top", "right"]].set_visible(False)

    # ── Panel B: conservation -- the rebate = private gain + deadweight ──────
    gain, dw, reb = s["private_gain"], s["deadweight"], s["rebate"]
    axR.bar(0, gain, 0.5, color="#27AE60", ec="#1E8449")
    axR.bar(0, dw, 0.5, bottom=gain, color="#922B21", ec="#641E16")
    axR.annotate(f"private gain\nto self-scheduler\n+${gain:,.0f}", (0, gain / 2),
                 ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    axR.annotate(f"deadweight loss\n(extra prod. cost)\n${dw:,.0f}",
                 (0, gain + dw / 2), ha="center", va="center", fontsize=8,
                 color="white", fontweight="bold")
    axR.annotate(f"rebate drawn from\nrent pool: ${reb:,.0f}", (0.32, reb / 2),
                 ha="left", va="center", fontsize=9,
                 arrowprops=dict(arrowstyle="-[, widthB=2.6", color="#34495E"))
    axR.set_xlim(-0.5, 1.1); axR.set_xticks([])
    axR.set_ylabel("$/h")
    axR.set_title(f"Rent pool: {s['pool_econ']:,.0f} to {s['pool_self']:,.0f} per hour\n"
                  f"(the rebate is pure transfer + waste)", fontsize=10)
    axR.spines[["top", "right"]].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import footprints as fpmod
    import wscc9_model as wm
    from seams_engine import compute_ptdf, solve_engine_dispatch

    net = wm.build_network({"line_4": 40.0})
    pt = compute_ptdf(net, slack_bus="1")
    fp = fpmod.make(pt, {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]},
                    {"BA-1": "#993AFF", "BA-2": "#2471A3"},
                    manage={"BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
                            "BA-2": ["line_0", "line_1", "line_7", "line_8"]})
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = solve_engine_dispatch(pt, eng, shed_price=150.0)
    fig = composite_figure(net, pt, fp, eng, res, suptitle="smoke test")
    fig.savefig("_fig_smoke.png", dpi=70, bbox_inches="tight")
    print("composite_figure OK ->", fig.get_size_inches(), "saved _fig_smoke.png")
def install_latex_repr() -> None:
    """Booktabs LaTeX rendering for DataFrames on PDF export (HTML tables stay
    in Jupyter). No effect outside an IPython session."""
    import pandas as pd

    def _df_to_latex(df: pd.DataFrame | pd.Series) -> str:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        body = df.to_latex(escape=True, na_rep='',
                           float_format=lambda v: f'{(0.0 if v == 0 else v):,.6g}')
        ncol = df.shape[1] + df.index.nlevels
        if ncol > 6:
            return '\\begin{center}\\resizebox{\\linewidth}{!}{%\n' + body + '}\\end{center}'
        return '\\begin{center}{\\small\n' + body + '}\\end{center}'

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        ltx = ip.display_formatter.formatters['text/latex']
        ltx.for_type(pd.DataFrame, _df_to_latex)
        ltx.for_type(pd.Series, _df_to_latex)
    except Exception:
        pass


def award_supply_demand(awards: Iterable[Award]) -> _SupDem:
    """Net injection per bus from point-to-point awards (source +, sink -),
    rendered as the supply/demand annotations the network figure takes."""
    inj: BusValues = {}
    for a in awards:
        inj[a.source] = inj.get(a.source, 0.0) + a.mw
        inj[a.sink] = inj.get(a.sink, 0.0) - a.mw
    sup: _Blocks
    sup, dem = {}, {}
    for b, v in inj.items():
        if v > 1e-6:
            sup.setdefault(b, []).append(
                {'unit_id': f'POR@{b}', 'price': 0.0, 'volume': v,
                 'accepted_volume': v, 'capacity': v})
        elif v < -1e-6:
            dem[b] = -v
    return sup, dem


def award_figure(
    pt: PTDFData,
    awards: Sequence[Award],
    title: str,
    net: pypsa.Network | None = None,
    monitored: _Monitored = 'all',
) -> None:
    """Draw an ATC award set on the network: the drawn line flows are the
    award flows (an infeasible set shows its overloads in red)."""
    import atc
    import matplotlib.pyplot as plt
    import wscc9_model as _wm
    net = _wm.build_network() if net is None else net
    sup, dem = award_supply_demand(awards)
    plot_network_topology(
        net, supply_by_bus=sup, demand_by_bus=dem,
        bus_colors=_wm.BUS_COLORS, bus_coords=_wm.COORDS,
        line_flows=atc.flow_dict(pt, awards),
        constrained_lines=set(atc.overloaded_lines(pt, awards, monitored)),
        title=title)
    plt.show()   # display once; returning the fig too would double it inline
def bilateral_dispatch_figure(
    pt: PTDFData,
    da: OrderBookResult,
    suppliers: Iterable[Supplier],
    title: str,
    monitored: _Monitored = 'all',
    line_ratings: LineValues | None = None,
) -> None:
    """The bilateral clearing on the network + nodal dispatch letter.
    GENERATORS: the solid bar is the FIXED marginal cost; the dotted staircase
    over it is the band of cleared (sale) prices its blocks fetched. LOADS:
    the receive-bar is segmented by the prices paid. Chords are the actual
    bilateral gen->load contracts. ``da`` is a bilateral OrderBookResult
    (trades / cleared / awards / premium)."""
    import atc
    import matplotlib.pyplot as plt
    import wscc9_model as _wm
    sup_rows: dict[Bus, dict[tuple[int, int], float]]
    dem_seg: dict[Bus, dict[int, float]]
    agg: dict[tuple[Bus, Bus], float]
    num: BusValues
    den: BusValues
    sup_rows, dem_seg, agg, num, den = {}, {}, {}, {}, {}
    for t in da.trades:
        sup_rows.setdefault(t['sbus'], {})
        skey = (round(t['s_cost']), round(t['price']))
        sup_rows[t['sbus']][skey] = sup_rows[t['sbus']].get(skey, 0.0) + t['qty']
        dem_seg.setdefault(t['center'], {})
        dem_seg[t['center']][round(t['price'])] = dem_seg[t['center']].get(round(t['price']), 0.0) + t['qty']
        agg[(t['sbus'], t['center'])] = agg.get((t['sbus'], t['center']), 0.0) + t['qty']
        for b in (t['sbus'], t['center']):
            num[b] = num.get(b, 0.0) + t['price'] * t['qty']
            den[b] = den.get(b, 0.0) + t['qty']
    cap_by_bus: BusValues
    base_cost_by_bus: BusValues
    cap_by_bus, base_cost_by_bus = {}, {}
    for s in suppliers:
        cap_by_bus[s.bus] = cap_by_bus.get(s.bus, 0.0) + s.p_nom
        base_cost_by_bus.setdefault(s.bus, s.cost)
    supply_by_bus, cleared_price_line = {}, {}
    for b, rows in sup_rows.items():
        blocks: list[dict[str, Any]] = []
        for ((c, p), mw) in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            uid = f'{b}.{len(blocks)}'
            blocks.append({'unit_id': uid, 'price': float(c), 'volume': mw,
                           'accepted_volume': mw, 'capacity': mw})
            cleared_price_line[uid] = float(p)
        supply_by_bus[b] = blocks
    for b, cap in cap_by_bus.items():
        disp = sum(blk['accepted_volume'] for blk in supply_by_bus.get(b, []))
        idle = cap - disp
        if idle > 0.5:
            idle_cost = (sum(blk['price'] * blk['accepted_volume'] for blk in supply_by_bus.get(b, []))
                         / disp) if disp > 0.5 else base_cost_by_bus[b]
            blocks = supply_by_bus.setdefault(b, [])
            blocks.append({'unit_id': f'{b}.idle{len(blocks)}', 'price': float(idle_cost),
                           'volume': round(idle, 1), 'accepted_volume': 0.0,
                           'capacity': round(idle, 1)})
    demand_by_bus = {c: round(v, 1) for c, v in da.cleared.items() if v > 0.5}
    demand_segments = {c: [{'mw': mw, 'price': float(p)} for p, mw in sorted(seg.items())]
                       for c, seg in dem_seg.items()}
    flows = [(s, c, round(mw, 1)) for (s, c), mw in agg.items() if mw > 0.5]
    bus_lmps = {b: num[b] / den[b] for b in num if den[b] > 0}
    awards = da.awards
    net_mw: BusValues = {}
    for a in awards:
        net_mw[a.source] = net_mw.get(a.source, 0.0) + a.mw
        net_mw[a.sink] = net_mw.get(a.sink, 0.0) - a.mw
    plot_combined_letter(
        _wm.build_network(line_ratings=line_ratings or {}), supply_by_bus, demand_by_bus,
        bus_colors=_wm.BUS_COLORS, bus_coords=_wm.COORDS, flows=flows,
        bus_lmps=bus_lmps, bus_net_mw=net_mw,
        gen_marginal_costs=cleared_price_line, lmp_line=False,
        gen_bid_labels=False, gen_cost_labels=True, block_mw_unit=False,
        line_flows=atc.flow_dict(pt, awards),
        constrained_lines=set(atc.overloaded_lines(pt, awards, monitored)),
        demand_segments=demand_segments, all_buses=_wm.RING_ORDER,
        sector_order=_wm.RING_ORDER, center_bus=_wm.CENTER_BUS,
        title_left='Network -- physical (PTDF) flow of the bilateral schedule',
        title_right='Bilateral dispatch -- gen cost bar + cleared-price staircase, load price bands',
        suptitle=title)
    plt.show()
