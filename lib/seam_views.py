# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Presentation frames for the seam notebooks (204, 204a, 205a, 205b).

Every exhibit is drawn the same way: the inputs a market solves around
(``input_view``), the book a price-sensitive bid arrives in (``bid_view``),
one market's chair (``own_view``), the
forecast book alone (``forecast_view``), the host's real-time relief with the
megawatts it moved hatched (``relief_view``), then the superposition as the
network + nodal dispatch letter and as the standalone net-injection network
(``combined_views``). ``export_pair`` writes the settlement and line CSVs, and
``save_clean`` writes the deck PNG copies with a transparent background and one
canvas size per graph type.

The notebook passes its context once through :func:`configure`: the footprint
partition ``fp``, the exhibit fleet and loads, the export knobs and the inline
strip. The frame functions keep the signatures the notebooks call.

The deck slide carries each exhibit's suptitle, panel titles and legends, so
:func:`strip` removes all three from a figure, and ``save_clean`` strips every
PNG it writes. Every frame displays its own figure through ``plt.show()`` before
it returns, so a notebook cannot strip the inline copy after the call.
Therefore, ``configure(strip_inline=True)`` strips each frame's figure before
its display (user decision 2026-11-07, for the public notebooks). The default
``False`` displays each figure as it is drawn. Every title in the frames and in
the figure helpers they call is set through ``set_title`` or ``suptitle``, so
the strip reaches all of them. Text drawn with ``ax.text`` or ``ax.annotate``,
an inset's label among it, is kept. The flag reaches the frames of this module
alone. ``wscc9_figures.award_figure``, ``wscc9_figures.bilateral_dispatch_figure``
and the ``intertie`` figures call ``plt.show()`` themselves and keep their titles
and legends. None of the workshop notebooks (204, 204a, 205a, 205b, 304a) calls
them on 2026-11-07.

The frames, :func:`strip` and ``save_clean`` are display code. They draw the
engine result they are handed and change no number, so they map to no source.
Two functions tabulate numbers from that result
and clear nothing. ``export_pair`` settles each bus at its home market's LMP and
states the canon's rent ``R = sum_m |mu_m F^M_m|`` on each market's own book
flow, and :func:`price_walk` reads a node's price as the cost of the
redispatch that serves one more MW there.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nodal_plot
import wscc9_figures as figs
from nodal_plot import plot_combined_letter
from seams_engine import physical_flows, susceptance_widths, to_supply_demand
from wscc9_model import BUS_COLORS, CENTER_BUS, COORDS, RING_ORDER

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import pypsa
    from matplotlib.figure import Figure

    from footprints import Footprints
    from lab_types import Bus, BusValues, Line, LineValues, Resource
    from seams_engine import EngineResult, MarketEngine, PTDFData

    Blocks: TypeAlias = dict[Bus, list[dict[str, Any]]]
    SupDem: TypeAlias = tuple[Blocks, BusValues]
    Curves: TypeAlias = dict[Resource, tuple[float, float]]
    EnginePair: TypeAlias = tuple[MarketEngine, EngineResult]

#: The grayed-equals-forecast cue shared by every frame.
GRAY = "#A6ACAF"

__all__ = [
    "CFG",
    "GRAY",
    "bid_view",
    "combined_views",
    "configure",
    "export_pair",
    "forecast_view",
    "input_view",
    "market_inputs_view",
    "own_view",
    "price_walk",
    "relief_view",
    "save_clean",
    "strip",
    "walk_text",
]

#: The three axes title locations matplotlib keeps, each emptied by :func:`strip`.
_TITLE_LOCS: tuple[Literal["left", "center", "right"], ...] = (
    "left", "center", "right")

#: Module context, set by :func:`configure` from the notebook's EDIT cell.
#: ``strip_inline`` (default ``False``) strips each frame's figure before its
#: inline display, so the public notebooks draw no suptitle, panel title or
#: legend (user decision 2026-11-07).
CFG: dict[str, Any] = {
    "fp": None,                 # Footprints partition (fp_of, colors, line_colors)
    "gen_fleet": None,          # the exhibit fleet {gen: {bus, cost, p_nom}}
    "loads": None,              # {bus: MW}
    "save_enabled": False,      # file writes off until the notebook opts in
    "export_dir": "presentation_exhibits",
    "export_prefix": "seams_cm_",   # CSV filename stem: 204/204a "seams_cm_", 205a "seams_tx_"
    "export_dpi": 600,
    "export_size": (13.33, 7.5),
    "net_export_size": (7.5, 7.5),
    "panel_export_size": (6.67, 7.5),
    "strip_inline": False,      # strip each frame's figure before plt.show()
}


def configure(**kw: object) -> None:
    """Set the module context. Unknown keys are rejected."""
    for k in kw:
        if k not in CFG:
            raise KeyError(f"seam_views.configure: unknown key {k!r}")
    CFG.update(kw)


def _show(fig: Figure) -> None:
    """A frame's inline display, stripped first when ``strip_inline`` is set."""
    if CFG["strip_inline"]:
        strip(fig)
    plt.show()


def _sup_dem(pairs: Sequence[EnginePair]) -> SupDem:
    sup, dem = {}, {}
    for _e, _r in pairs:
        _s, _d = to_supply_demand(_e, _r)
        sup.update(_s)
        dem.update(_d)
    return sup, dem


def _net_mw(pt_: PTDFData, *results: EngineResult) -> BusValues:
    return {b: float(sum(r.injection[pt_.bus_idx[b]] for r in results)) for b in pt_.buses}


def own_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints,
    side: str,
    eng: MarketEngine,
    res: EngineResult,
    proxy: EnginePair | None = None,
    binding: Collection[Line] | None = None,
    suptitle: str = '',
    save: str | None = None,
    sup_dem: SupDem | None = None,
    demand_segments: Blocks | None = None,
    legend_note: str | None = None,
    curves: Curves | None = None,
) -> None:
    """One market's chair. With ``proxy=(engine, result)`` the neighbor is drawn
    grayed as that market's FORECAST of it, its net injections annotated at the
    neighbor's buses, and the network panel carries the studied superposition.

    ``sup_dem`` and ``demand_segments`` take over the ring's bars, which is how a
    price-taking self-schedule is drawn as its own gray block: build them with
    :func:`wscc9_figures.self_schedule_bars`, the shared convention across the
    series, and pass them straight through. An explicit ``sup_dem`` overrides the
    proxy branch's superposed bars.
    """
    if proxy is None:
        fig = figs.footprint_figure(net_, pt_, fp_, eng, res, highlight=side,
                                    node_net_mw=True, sup_dem=sup_dem,
                                    demand_segments=demand_segments,
                                    legend_note=legend_note, curves=curves,
                                    constrained_lines=binding, suptitle=suptitle)
    else:
        p_eng, p_res = proxy
        sup, dem = _sup_dem([(eng, res), (p_eng, p_res)])
        act = list(eng.activated_lines) if eng.activated_lines != 'all' else list(pt_.lines)
        flagged = (binding if binding is not None else
                   [l for l in act if abs(res.line_dual.get(l, 0.0)) > 1e-3])
        fig = figs.footprint_figure(
            net_, pt_, fp_, eng, res, highlight=side,
            sup_dem=(sup_dem if sup_dem is not None else (sup, dem)), shed=False,
            demand_segments=demand_segments, legend_note=legend_note,
            node_net_mw=_net_mw(pt_, res, p_res),
            line_flows={l: res.flow_own[l] + p_res.flow_own[l] for l in pt_.lines},
            constrained_lines=flagged,
            bus_lmps={b: float((res if fp_.fp_of(b) == side else p_res).lmp[b])
                      for b in pt_.buses},
            title_left=f'Network -- own book + forecast of '
                       f'{"B" if side == "Market A" else "A"} (grayed = forecast)',
            suptitle=suptitle)
    _show(fig)
    if save:
        save_clean(fig, save)


def relief_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints | None,
    side: str,
    eng: MarketEngine,
    r_da: EngineResult,
    r_rt: EngineResult,
    offsets: LineValues,
    *,
    curves: Curves | None = None,
    binding: Collection[Line] | None = None,
    legend_note: str | None = None,
    suptitle: str = '',
    save: str | None = None,
    sup_dem: SupDem | None = None,
    demand_segments: Blocks | None = None,
    hatch: str = "////",
) -> Figure:
    """The host market's REAL-TIME RELIEF, on its own chair.

    The host re-clears its own book against the flow on each line that its book
    does not carry (204's mechanism), and this frame draws that re-clearing the
    way the market itself reads it: its own dispatch on the ring, its own flows
    on the network, its own footprint in colour and the counterparty's grayed.

    The frame is drawn against the DAY-AHEAD engine ``eng`` and a relief result
    ``r_rt`` cleared at the host's own offers, so every unit on the ring is a
    unit of the host's book. A result from
    :func:`seams_engine.rt_relief` cannot be drawn here, because its units are
    the re-offered day-ahead blocks ``{g}__da`` and the increments ``{g}__inc``
    and they match no unit of ``eng``; at a zero premium the two clearings have
    the same dispatch and the same cost, and above it the premium menu's bill
    is reported beside the frame rather than drawn.

    Every bus whose generation rose between ``r_da`` and ``r_rt`` carries the
    megawatts it moved as a hatched block at the top of its own stack, through
    :func:`wscc9_figures.redispatch_bars`, in the bus's own colour, because the
    increment is the host's own generation serving its own load. ``offsets``
    is the ``{line: MW}`` fixed flow the relief cleared against; it names the
    megawatts in the default legend note and is not drawn on the network,
    because the network panel is the host's own book.

    ``sup_dem`` and ``demand_segments`` take over the ring's bars before the
    hatch is applied, which is how a schedule the host still carries in real
    time is drawn in the shared self-schedule gray: build them with
    :func:`wscc9_figures.self_schedule_bars` on ``r_rt`` and pass them through,
    as :func:`own_view` takes them for a day-ahead frame.
    """
    fp_ = fp_ or CFG['fp']
    sup_dem = to_supply_demand(eng, r_rt) if sup_dem is None else sup_dem
    moves = {b: float(r_rt.gen_by_bus.get(b, 0.0)) - float(r_da.gen_by_bus.get(b, 0.0))
             for b in set(r_da.gen_by_bus) | set(r_rt.gen_by_bus)}
    up = {b: mw for b, mw in sorted(moves.items()) if mw > 1e-6}
    for b, mw in up.items():
        sup_dem, _ = figs.redispatch_bars(eng, r_rt, b, mw, sup_dem=sup_dem,
                                          curves=curves, hatch=hatch)
    act = list(eng.activated_lines) if eng.activated_lines != 'all' else list(pt_.lines)
    flagged = (binding if binding is not None else
               [l for l in act if abs(r_rt.line_dual.get(l, 0.0)) > 1e-3])
    if legend_note is None:
        worst = max(act, key=lambda l: abs(float(offsets.get(l, 0.0))), default=None)
        moved = ', '.join(f'{mw:.2f} MW at bus {b}' for b, mw in up.items()) or 'nothing'
        legend_note = (f'Hatched = the {moved} this market moved in real time'
                       + (f'; it cleared around {float(offsets.get(worst, 0.0)):+.2f} MW '
                          f'of fixed flow on {worst}' if worst else ''))
    fig = figs.footprint_figure(net_, pt_, fp_, eng, r_rt, highlight=side,
                                node_net_mw=True, sup_dem=sup_dem, shed=False,
                                demand_segments=demand_segments,
                                legend_note=legend_note, curves=curves,
                                constrained_lines=flagged, suptitle=suptitle)
    _show(fig)
    if save:
        save_clean(fig, save)
    return fig


def _grayed(
    pt_: PTDFData, fp_: Footprints, side: str
) -> tuple[set[Bus], dict[Bus, str], dict[Line, str], dict[str, str]]:
    """Gray every footprint but ``side``: its buses, the lines between them, and
    its band. The shared cue of every frame that gives one market the chair."""
    dim = {b for b in pt_.buses if fp_.fp_of(b) != side}
    colors = dict(BUS_COLORS)
    for b in dim:
        colors[b] = figs._DIM
    lcolors = fp_.line_colors(pt_)
    dim_fps = {n for n in fp_.names if all(str(b) in dim for b in fp_.defs[n])}
    for l in pt_.lines:
        b0, b1 = pt_.line_buses[pt_.line_idx[l]]
        if (b0 in dim and b1 in dim) or fp_.line_assign.get(l) in dim_fps:
            lcolors[l] = figs._DIM
    gcolors = {n: (figs._DIM if n in dim_fps else fp_.colors[n]) for n in fp_.names}
    return dim, colors, lcolors, gcolors


def bid_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints | None,
    side: str,
    eng: MarketEngine,
    node: Bus,
    mw: float | None,
    price: float | None,
    kind: Literal['resource', 'load'],
    *,
    blocks: Sequence[tuple[float, float]] | None = None,
    ref_price: float | None = None,
    curves: Curves | None = None,
    suptitle: str = '',
    save: str | None = None,
    title_left: str | None = None,
    title_right: str | None = None,
) -> Figure:
    """One market's book AS CONFIGURED, with a price-sensitive bid at ``node``.

    A bid is not a self-schedule. It carries a price, it is not balanced, and the
    market decides how much of it to award, so the market's book holds only the
    end of the transaction that falls at a node outside its own footprint:

    1. ``kind='resource'`` draws the bid as gray supply at ``node``: energy
       offered into this market at ``price``, up to ``mw``, none of it accepted
       yet. The block keeps its offer label, because a bid has a price where a
       price-taking self-schedule has none.
    2. ``kind='load'`` draws it as gray demand at ``node``: energy this market is
       asked to serve, bid at ``price``, up to ``mw``.

    Nothing else in the book has cleared either, so every resource stands at its
    full nameplate and the network panel carries no flows. The counter-leg sits
    in the OTHER market's book and never enters this one, which is why there is
    no chord on the ring and no flow on the network: until the market runs, the
    bid is a quantity offered at a price and nothing more.

    A bid may also be a stack of blocks, ``blocks=[(mw, price), ...]``, in place
    of the single block ``(mw, price)``. The stack is drawn at ``node`` as
    consecutive gray blocks in merit order, each at its own price, through the
    same ring entries the single block uses: a supply stack in ascending price,
    each block carrying its ``$p x q MW`` offer label, and a demand stack in
    descending price, each block carrying its MW inside the bar and its price as
    the bar's height, as the single demand block does. A stack of one block draws
    the same figure as the single-block call. The stack is the September frame
    generalized to a price-sensitive bid of many blocks (Lab display choice,
    2026-11-10), and the frame changes no number, so it maps to no source.

    Parameters
    ----------
    net_ : pypsa.Network
        The network the panels draw.
    pt_ : PTDFData
        The shift factors of ``net_``, for the bus and line sets and the line
        widths.
    fp_ : Footprints or None
        The market partition. ``None`` takes ``CFG['fp']``.
    side : str
        The market whose book is drawn, a key of the partition.
    eng : MarketEngine
        The market's engine, whose ``loads`` (MW by bus) are drawn as its demand.
    node : Bus
        The bus at which the bid's end in this market falls.
    mw : float or None
        The single block's quantity, MW, positive. ``None`` when ``blocks`` is
        given.
    price : float or None
        The single block's price, $/MWh. ``None`` when ``blocks`` is given.
    kind : {'resource', 'load'}
        Whether the bid is supply offered at ``node`` or demand bid at ``node``.
    blocks : sequence of (float, float), optional
        The bid as a stack, one ``(mw, price)`` pair per block, MW positive and
        $/MWh. It replaces ``(mw, price)``, which must then both be ``None``.
        The order given is not kept, because the frame sorts the stack into merit
        order. Default ``None`` draws the single block ``(mw, price)``.
    ref_price : float, optional
        The bar height of every active bus and of the load the bid does not
        cover, $/MWh. Default the highest fleet cost or bid price.
    curves : dict, optional
        ``{unit: (a, b)}`` draws the market's own resources as rising
        marginal-cost wedges, ``a + b g`` in $/MWh with ``g`` in MW, the offer
        model of the curve examples, in place of flat blocks.
    suptitle : str, optional
        The figure's suptitle.
    save : str, optional
        The file stem ``save_clean`` writes, when exports are enabled.
    title_left, title_right : str, optional
        The panel titles. Defaults name the market and describe the bid.

    Returns
    -------
    matplotlib.figure.Figure
        The network panel and the nodal book, displayed before the return.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``'resource'`` nor ``'load'``, if ``blocks`` is
        given together with ``mw`` or ``price``, if neither is given in full, or
        if ``blocks`` is empty or holds a block of nonpositive MW.

    """
    fp_ = fp_ or CFG['fp']
    fleet = CFG['gen_fleet']
    nd = str(node)
    if kind not in ('resource', 'load'):
        raise ValueError(f"kind must be 'resource' or 'load', got {kind!r}")
    stack: list[tuple[float, float]]
    if blocks is None:
        if mw is None or price is None:
            raise ValueError(
                f"mw and price are both required without blocks, got mw={mw!r}, "
                f"price={price!r}")
        stack = [(float(mw), float(price))]
    else:
        if mw is not None or price is not None:
            raise ValueError(
                f"blocks replaces mw and price, which must both be None, got "
                f"mw={mw!r}, price={price!r}")
        stack = [(float(q), float(p)) for q, p in blocks]
        if not stack or any(q <= 0.0 for q, _ in stack):
            raise ValueError(
                f"blocks must hold at least one block, each of positive MW, got "
                f"{blocks!r}")
    # Merit order: supply rises in price, demand falls (stable for equal prices).
    stack.sort(key=lambda b: b[1] if kind == 'resource' else -b[1])
    Q = sum(q for q, _ in stack)
    P_lo, P_hi = min(p for _, p in stack), max(p for _, p in stack)

    sup: Blocks
    sup, dem = {}, {}
    for gid, spec in fleet.items():
        b = str(spec['bus'])
        if fp_.fp_of(b) != side:          # only what this market may dispatch
            continue
        cap = float(spec['p_nom'])
        sup.setdefault(b, []).append({'unit_id': gid, 'price': float(spec['cost']),
                                      'volume': cap, 'capacity': cap,
                                      'accepted_volume': cap})
    for b in sup:
        sup[b].sort(key=lambda g: g['price'])
        for u in sup[b]:
            if curves and u['unit_id'] in curves:
                u['mc0'], u['mc_slope'] = map(float, curves[u['unit_id']])
                u['price'] = u['mc0']          # the wedge starts at its intercept
    for b, v in eng.loads.items():
        if float(v) > 1e-9:
            dem[str(b)] = float(v)

    ref = (float(ref_price) if ref_price is not None
           else max([float(spec['cost']) for spec in fleet.values()] + [P_hi]))
    gray = {'color': figs._SELFSCHED}
    segs = None
    if kind == 'resource':
        # One gray entry per block, ahead of the bus's own units, as the single
        # block stands; a stack's ids are numbered in merit order.
        sup.setdefault(nd, [])[:0] = [
            {'unit_id': 'bid' if len(stack) == 1 else f'bid_{k + 1}', 'price': p,
             'volume': q, 'capacity': q, 'accepted_volume': 0.0,
             'force_label': True, **gray}
            for k, (q, p) in enumerate(stack)]
        # A stack's title carries two dollar signs, escaped, because matplotlib
        # reads an unescaped pair as mathtext.
        what = (f'a resource offering up to {Q:.0f} MW at ${P_lo:,.0f}/MWh'
                if len(stack) == 1 else
                f'a resource offering up to {Q:.0f} MW in {len(stack)} blocks '
                f'at \\${P_lo:,.0f} to \\${P_hi:,.0f}/MWh')
    else:
        dem[nd] = dem.get(nd, 0.0) + Q
        total = dem[nd]
        # Nothing is awarded on this frame, so the whole bid is drawn at the
        # faint alpha an unawarded volume carries, the same cue the awarded
        # frames use for the part of a bid a market did not take.
        segs = {nd: [dict(mw=q, price=p, alpha=0.15, **gray) for q, p in stack]
                    + ([{'mw': total - Q, 'price': ref}] if total - Q > 0.01 else [])}
        what = (f'a load bidding ${P_hi:,.0f}/MWh for up to {Q:.0f} MW'
                if len(stack) == 1 else
                f'a load bidding \\${P_hi:,.0f} down to \\${P_lo:,.0f}/MWh for up '
                f'to {Q:.0f} MW in {len(stack)} blocks')

    dim, colors, lcolors, gcolors = _grayed(pt_, fp_, side)
    active = set(sup) | {b for b, v in dem.items() if v > 1e-9}

    fig, _ax = plot_combined_letter(
        net_, sup, dem,
        bus_colors=colors,
        bus_lmps={b: (ref if b in active else 0.0) for b in pt_.buses},
        show_bus_price=False, lmp_line=False,
        bus_net_mw=None, network_show_lmp=False,
        line_flows={l: 0.0 for l in pt_.lines},   # nothing cleared, so no flows
        line_widths=susceptance_widths(pt_), line_colors=lcolors,
        flows=[], demand_segments=segs,           # no counter-leg, so no chord
        dim_buses=dim,
        bus_groups={b: fp_.fp_of(b) for b in pt_.buses},  # type: ignore[misc]  # fp_of is None only for a bus off the partition, and every bus is on it
        group_colors=gcolors, show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER,
        bus_coords=COORDS, center_bus=CENTER_BUS,
        annotate_roles=True, axis_key=True,
        title_left=title_left or f'Network -- {side} meets the transaction at bus {nd}',
        title_right=title_right or f'Nodal book as configured -- the bid is {what}',
        suptitle=suptitle)
    _show(fig)
    if save:
        save_clean(fig, save)
    return fig


def input_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints | None,
    side: str,
    eng: MarketEngine,
    source: Bus,
    sink: Bus,
    mw: float,
    *,
    ref_price: float | None = None,
    suptitle: str = '',
    save: str | None = None,
    title_left: str | None = None,
    title_right: str | None = None,
) -> Figure:
    """One market's INPUTS, before it clears: the capability it may dispatch, the
    load it must serve, and the schedule it has to solve around.

    Nothing on this frame is a result. Every resource stands at its full
    nameplate, no price has formed, and the bars share one reference height, so
    the ring reads as a statement of quantities rather than of a merit order.

    The schedule is a BALANCED price-taking self-schedule of ``mw`` MW from
    ``source`` to ``sink``, and both markets are handed the same one. Each market
    draws both of its legs in the shared self-schedule gray, tied by a chord, and
    which leg is exogenous to that market follows from the partition:

    1. A leg at a bus INSIDE the market's footprint is a portion of the market's
       own track. At the source it is the part of that bus's capability the
       schedule commits, and at the sink it is the part of the load the market
       already serves that the schedule covers. Neither adds to the total.
    2. A leg at a bus OUTSIDE it is a fixed quantity the market solves around: an
       injection the market did not dispatch, or a withdrawal it serves no load
       behind. It is what makes the schedule visible in that market's clearing.

    The network panel carries the schedule whole, as the balanced pair it is: the
    injection, the withdrawal that balances it, and the flows the pair places on
    the network, ``S(SF_{s,m}-SF_{k,m})`` on each line ``m``. Both markets are
    handed the same pair, so both panels carry the same annotations and the same
    flows, and the panels differ only in which market is grayed.
    """
    fp_ = fp_ or CFG['fp']
    fleet = CFG['gen_fleet']
    src, snk, S = str(source), str(sink), float(mw)
    src_in, snk_in = fp_.fp_of(src) == side, fp_.fp_of(snk) == side

    sup: Blocks
    sup, dem = {}, {}
    for gid, spec in fleet.items():
        b = str(spec['bus'])
        if fp_.fp_of(b) != side:          # only what this market may dispatch
            continue
        cap = float(spec['p_nom'])
        sup.setdefault(b, []).append({'unit_id': gid, 'price': float(spec['cost']),
                                      'volume': cap, 'capacity': cap,
                                      'accepted_volume': cap})
    for b in sup:
        sup[b].sort(key=lambda g: g['price'])
    for b, v in eng.loads.items():
        if float(v) > 1e-9:
            dem[str(b)] = float(v)

    ref = (float(ref_price) if ref_price is not None
           else max(float(spec['cost']) for spec in fleet.values()))
    gray = {'color': figs._SELFSCHED}
    legs = []

    # -- the source leg -------------------------------------------------------
    if src_in:                            # a portion of this market's own capability
        taken = 0.0
        for u in sup.get(src, []):
            take = min(S - taken, u['capacity'])
            u['volume'] = u['capacity'] = u['capacity'] - take
            u['accepted_volume'] = u['capacity']
            taken += take
        sup[src] = [u for u in sup.get(src, []) if u['capacity'] > 1e-9]
        legs.append(f'{S:.0f} MW of the capability at bus {src}')
    else:                                 # an injection this market did not dispatch
        legs.append(f'an injection of {S:.0f} MW at bus {src}')
    sup.setdefault(src, []).insert(0, {
        'unit_id': 'self_schedule', 'price': ref, 'volume': S, 'capacity': S,
        'accepted_volume': S, 'price_label': False, 'force_label': True,
        'lane': 'self_schedule', **gray})

    # -- the sink leg ---------------------------------------------------------
    if snk_in:                            # a portion of the load it already serves
        legs.append(f'{S:.0f} MW of the load at bus {snk}')
    else:                                 # a withdrawal it serves no load behind
        dem[snk] = dem.get(snk, 0.0) + S
        legs.append(f'a withdrawal of {S:.0f} MW at bus {snk}')

    # The network panel carries the schedule as the balanced pair it is: the
    # injection at the source, the withdrawal at the sink that balances it, and
    # the flows the pair places on the network. Both markets are handed the same
    # pair, so both panels carry the same annotations and the same flows,
    # whichever leg each market holds inside its own footprint.
    netmw = {src: +S, snk: -S}
    _sched = S * (pt_.ptdf[:, pt_.bus_idx[src]] - pt_.ptdf[:, pt_.bus_idx[snk]])
    sched_flows = {l: float(_sched[pt_.line_idx[l]]) for l in pt_.lines}
    total = dem.get(snk, 0.0)
    segs = {snk: [dict(mw=min(S, total), price=ref, lane='self_schedule', **gray)]
                 + ([{'mw': total - S, 'price': ref}] if total - S > 0.01 else [])}

    dim, colors, lcolors, gcolors = _grayed(pt_, fp_, side)
    # Only a bus carrying something takes the reference bar height. The rest read
    # as plain numbered buses, because no price has formed on this frame and a
    # grayed bus with a price would print one.
    active = set(sup) | {b for b, v in dem.items() if v > 1e-9}

    fig, _ax = plot_combined_letter(
        net_, sup, dem,
        bus_colors=colors,
        bus_lmps={b: (ref if b in active else 0.0) for b in pt_.buses},
        show_bus_price=False, lmp_line=False,
        bus_net_mw=netmw, network_show_lmp=False,
        line_flows=sched_flows,
        line_widths=susceptance_widths(pt_), line_colors=lcolors,
        flows=[(src, snk, S, 'self_schedule')], demand_segments=segs,
        dim_buses=dim,
        bus_groups={b: fp_.fp_of(b) for b in pt_.buses},  # type: ignore[misc]  # fp_of is None only for a bus off the partition, and every bus is on it
        group_colors=gcolors, show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER,
        bus_coords=COORDS, center_bus=CENTER_BUS,
        annotate_roles=True, axis_key=True,
        title_left=(title_left or f'Network -- the balanced schedule, bus {src} to '
                    f'bus {snk}, and the flows it places'),
        title_right=title_right or ('Nodal inputs -- capability offered, load to '
                                    'be served, and the balanced self-schedule'),
        suptitle=suptitle)
    _show(fig)
    if save:
        save_clean(fig, save)
    return fig


def market_inputs_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints | None,
    side: str,
    eng: MarketEngine,
    *,
    neighbor_mw: BusValues | None = None,
    ref_price: float | None = None,
    suptitle: str = '',
    save: str | None = None,
    title_left: str | None = None,
    title_right: str | None = None,
    line_colors: dict[Line, str] | None = None,
) -> Figure:
    """One market's INPUTS to its own clearing, with no schedule: the capability it
    may dispatch, the load it must serve, and the injections and withdrawals it
    assumes at the neighbor's buses (304a's parallel-process slide).

    Nothing on this frame is a result. Every resource of ``side`` stands at its
    full nameplate, backstops included, the load stands at the reference height
    (the dearest offer, unless ``ref_price`` is given), and no price has formed.
    The neighbor's footprint is grayed. ``neighbor_mw`` is the forecast
    ``{bus: MW}`` of the neighbor's net injections the market clears around,
    default zero at every neighbor bus that carries a resource or a load, as in
    a blind clearing that assumes no flow from the neighbor; the network panel
    labels those buses with the assumed megawatts and carries the flows they
    place, ``sum_n SF_{n,m} x_n`` on each line ``m`` (zero for a zero forecast).
    ``line_colors`` ``{line: colour}`` overrides the footprint colouring of the named
    lines, so a line the market enforces in the neighbor's footprint (304a's flow
    cap on ``line_4`` for Market B) is drawn rather than grayed. Default: no override.
    """
    fp_ = fp_ or CFG['fp']
    fleet, loads = CFG['gen_fleet'], CFG['loads']
    sup: Blocks
    sup, dem = {}, {}
    for gid, spec in fleet.items():
        b = str(spec['bus'])
        if fp_.fp_of(b) != side:
            continue
        cap = float(spec['p_nom'])
        sup.setdefault(b, []).append({'unit_id': gid, 'price': float(spec['cost']),
                                      'volume': cap, 'capacity': cap,
                                      'accepted_volume': cap})
    for b in sup:
        sup[b].sort(key=lambda g: g['price'])
    for b, v in eng.loads.items():
        if float(v) > 1e-9:
            dem[str(b)] = float(v)
    ref = (float(ref_price) if ref_price is not None
           else max(float(spec['cost']) for spec in fleet.values()))
    if neighbor_mw is None:
        nb = {str(spec['bus']) for spec in fleet.values()
              if fp_.fp_of(str(spec['bus'])) != side}
        nb |= {str(b) for b, v in loads.items()
               if fp_.fp_of(str(b)) != side and float(v) > 1e-9}
        neighbor_mw = {b: 0.0 for b in sorted(nb, key=int)}
    x = {str(b): float(v) for b, v in neighbor_mw.items()}
    inj = [x.get(b, 0.0) for b in pt_.buses]
    import numpy as _np
    fl = pt_.ptdf @ _np.array(inj)
    flows = {l: float(fl[pt_.line_idx[l]]) for l in pt_.lines}
    dim, colors, lcolors, gcolors = _grayed(pt_, fp_, side)
    if line_colors:
        lcolors = {**lcolors, **{str(l): c for l, c in line_colors.items()}}
    active = set(sup) | set(dem)
    fig, _ax = plot_combined_letter(
        net_, sup, dem,
        bus_colors=colors,
        bus_lmps={b: (ref if b in active else 0.0) for b in pt_.buses},
        show_bus_price=False, lmp_line=False,
        bus_net_mw=x, network_show_lmp=False, net_mw_zero_label=True,
        line_flows=flows,
        line_widths=susceptance_widths(pt_), line_colors=lcolors,
        flows=[], dim_buses=dim,
        bus_groups={b: fp_.fp_of(b) for b in pt_.buses},  # type: ignore[misc]  # fp_of is None only for a bus off the partition, and every bus is on it
        group_colors=gcolors, show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER,
        bus_coords=COORDS, center_bus=CENTER_BUS,
        annotate_roles=True, axis_key=True,
        title_left=(title_left or f"Network -- {side}'s assumed neighbor injections "
                    "and the flows they place"),
        title_right=title_right or (f'Nodal inputs -- {side}: capability offered '
                                    'and load to be served'),
        suptitle=suptitle)
    _show(fig)
    if save:
        save_clean(fig, save)
    return fig


def combined_views(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints,
    pairs: Sequence[EnginePair],
    label: str,
    binding: Collection[Line] | None = None,
    note: str | None = None,
    save: str | None = None,
    network_panel: bool = False,
) -> pd.DataFrame:
    """The superposition, twice: the combined letter, then the standalone
    network carrying only net injections/withdrawals and the flows.

    The superposed flow is read with ``exo_physical=False``: in the two-engine
    seam representation every exogenous schedule an engine carries is the other
    engine's leg of a schedule that engine already books, so the physical flow
    is each market's cleared generation less its served load and nothing else.
    """
    ress = [r for _, r in pairs]
    sup, dem = _sup_dem(pairs)
    pf = physical_flows(pt_, ress, exo_physical=False).rename(
        columns={'F_phys': 'F^s'})
    phys = {l: float(pf.loc[l, 'F^s']) for l in pt_.lines}
    red = set(pf.index[pf['overload']]) if binding is None else set(binding)
    host = {b: float(next(r for e, r in pairs if fp_.fp_of(b) == e.name).lmp[b])
            for b in pt_.buses}
    # The node labels read the same actual injection the flows are formed from:
    # each market's cleared generation less its served load, with no scheduled
    # injection, so the labels and the line flows state one dispatch. Written as
    # the net injection less each market's own scheduled leg, so a pair of
    # results with no schedule gives the plain net injection unchanged.
    netmw = _net_mw(pt_, *ress)
    for r in ress:
        for bus, mw in r.exo_by_bus.items():
            netmw[str(bus)] -= float(mw)
    fig_cl, _ = plot_combined_letter(
        net_, sup, dem,
        bus_colors=dict(BUS_COLORS), bus_lmps=host, bus_net_mw=netmw,
        network_show_lmp=False, line_flows=phys,
        line_widths=susceptance_widths(pt_), line_colors=fp_.line_colors(pt_),
        constrained_lines=red, flows=nodal_plot.compute_ptdf_flows(net_, sup, dem),
        lmp_line=True, bus_groups={b: fp_.fp_of(b) for b in pt_.buses},  # type: ignore[misc]  # fp_of is None only for a bus off the partition, and every bus is on it
        group_colors=dict(fp_.colors), show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER, bus_coords=COORDS,
        center_bus=CENTER_BUS, annotate_roles=True, axis_key=True,
        title_left=note or 'Network -- superposed flow (red = overload)',
        suptitle=f'{label} -- superposed flow  F^s = F[Market A] + F[Market B]'
                 f'   (net injections on the network; prices on the ring)')
    _show(fig_cl)
    fig_nt, _ax_nt = nodal_plot.plot_network_topology(
        net_, bus_colors={b: fp_.colors[fp_.fp_of(b)] for b in pt_.buses},  # type: ignore[index]  # fp_of is None only for a bus off the partition, and every bus is on it
        bus_net_mw=netmw, bus_coords=COORDS, line_flows=phys,
        line_widths=susceptance_widths(pt_), line_colors=fp_.line_colors(pt_),
        constrained_lines=red,
        title=f'{label} -- combined net injections and superposed flows')
    _show(fig_nt)
    if save:
        save_clean(fig_cl, f'{save}_combined')
        save_clean(fig_nt, f'{save}_network', size=CFG['net_export_size'])
    if save and network_panel:
        # the combined letter's network panel alone: per-bus colors, no ring
        fig_np, _ax_np = nodal_plot.plot_network_topology(
            net_, bus_colors=dict(BUS_COLORS), bus_net_mw=netmw,
            bus_coords=COORDS, line_flows=phys,
            line_widths=susceptance_widths(pt_), line_colors=fp_.line_colors(pt_),
            constrained_lines=red,
            number_position='outside', box_node_header=False, lmp_only=True,
            node_number_fontsize=11, annot_fontsize=7.5, title_fontsize=11,
            title=f'{label} -- network panel')
        _show(fig_np)
        save_clean(fig_np, f'{save}_network_panel', size=CFG['panel_export_size'])
    return pf


def forecast_view(
    net_: pypsa.Network,
    pt_: PTDFData,
    fp_: Footprints,
    fc_eng: MarketEngine,
    fc_res: EngineResult,
    label: str,
    save: str | None = None,
) -> None:
    """The forecast book alone: the forecast market's cleared dispatch on the
    ring and its flow contributions on the shared network. The other market's
    resources are drawn as configured -- full nameplate, solid, load at the
    reference height, no net injections and no dispatch -- so every megawatt
    on the network panel is the forecast book's."""
    gen_fleet, loads = CFG['gen_fleet'], CFG['loads']
    cfg_side = 'Market A' if fc_eng.name == 'Market B' else 'Market B'
    supF, demF = to_supply_demand(fc_eng, fc_res)
    sup = {b: list(v) for b, v in supF.items()}
    dem = dict(demF)
    for gid, spec in gen_fleet.items():
        b = str(spec['bus'])
        if fp_.fp_of(b) != cfg_side:
            continue
        cap = float(spec['p_nom'])
        sup.setdefault(b, []).append({
            'unit_id': gid, 'price': float(spec['cost']),
            'volume': cap, 'capacity': cap,
            'accepted_volume': cap,          # configuration style: full = solid
        })
        sup[b].sort(key=lambda g: g['price'])
    for b, v in loads.items():
        if fp_.fp_of(str(b)) == cfg_side and v > 0:
            dem[str(b)] = float(v)
    p_ref = max(float(s['cost']) for s in gen_fleet.values())
    lmps = {b: (float(fc_res.lmp[b]) if fp_.fp_of(b) == fc_eng.name else p_ref)
            for b in pt_.buses}
    # the forecast area drawn gray -- nodes, lines, band, bars and chords --
    # the same grayed-equals-forecast cue the own-market views use
    bus_cols = {b: (GRAY if fp_.fp_of(b) == fc_eng.name else BUS_COLORS[b])
                for b in pt_.buses}
    line_cols = {l: (GRAY if c == fp_.colors[fc_eng.name] else c)
                 for l, c in fp_.line_colors(pt_).items()}
    grp_cols = {g: (GRAY if g == fc_eng.name else c)
                for g, c in fp_.colors.items()}
    netmw = {b: float(fc_res.injection[pt_.bus_idx[b]])
             for b in pt_.buses if fp_.fp_of(b) == fc_eng.name}
    fig, _axes = plot_combined_letter(
        net_, sup, dem,
        bus_colors=bus_cols,
        bus_lmps=lmps, show_bus_price=False,   # heights only; no printed price
        bus_net_mw=netmw,                      # forecast side annotated only
        network_show_lmp=False,
        line_flows={l: float(fc_res.flow_own[l]) for l in pt_.lines},
        line_widths=susceptance_widths(pt_), line_colors=line_cols,
        flows=nodal_plot.compute_ptdf_flows(net_, supF, demF),
        lmp_line=False, bus_groups={b: fp_.fp_of(b) for b in pt_.buses},  # type: ignore[misc]  # fp_of is None only for a bus off the partition, and every bus is on it
        group_colors=grp_cols, show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER, bus_coords=COORDS,
        center_bus=CENTER_BUS, annotate_roles=True, axis_key=True,
        title_left=f"Network -- {fc_eng.name}'s forecast flow contributions",
        title_right=f"Nodal -- {fc_eng.name}'s forecast book; {cfg_side} as configured",
        suptitle=f"{label}: the forecast book alone")
    _show(fig)
    if save:
        save_clean(fig, save)


def export_pair(
    pt_: PTDFData,
    pairs: Sequence[EnginePair],
    slug: str,
    offsets: dict[str, LineValues] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The deck's two tables: settlement by node at home-market prices, and the
    line table with each market's book flow, dual and rent. Root CSVs.

    The ``F^s`` column is the superposition under ``exo_physical=False``, the
    rule :func:`combined_views` states: each market's cleared generation less
    its served load, with no scheduled injection added, because a balanced
    schedule's two legs are already realized in the two markets' books. 205a's
    stored ``F^s`` on ``line_4`` therefore reads 124 percent in its Example 2
    and 134 percent in its Example 3b, where summing the scheduled injections
    as well had given 153 percent and 152 percent. That is a correction, and
    205a's own scope note disclaims the combined-flow column.
    """
    fp_ = CFG['fp']
    rows = []
    for b in pt_.buses:
        eng, r = next((e, x) for e, x in pairs if fp_.fp_of(b) == e.name)
        load = float(eng.loads.get(b, 0.0))
        served = load - float(r.shed_by_bus.get(b, 0.0))
        gen = float(r.gen_by_bus.get(b, 0.0))
        lmp = float(r.lmp[b])
        rows.append({'bus': b, 'market': eng.name, 'LMP ($/MWh)': round(lmp, 2),
                     'gen (MW)': round(gen, 1), 'paid to gen ($/h)': round(lmp * gen, 1),
                     'load (MW)': round(load, 1), 'served (MW)': round(served, 1),
                     'paid by load ($/h)': round(lmp * served, 1)})
    settle = pd.DataFrame(rows).set_index('bus')
    for eng, _ in pairs:
        sub = settle[settle['market'] == eng.name]
        settle.loc[f'SUBTOTAL {eng.name}'] = [
            eng.name, '', round(sub['gen (MW)'].sum(), 1),
            round(sub['paid to gen ($/h)'].sum(), 1), round(sub['load (MW)'].sum(), 1),
            round(sub['served (MW)'].sum(), 1), round(sub['paid by load ($/h)'].sum(), 1)]
        # Two rows, two objects (2026-10-17; before, the residual alone carried
        # the label CONGESTION RENT). The residual is what the market's load
        # paid less what its generation received at the market's own prices;
        # it equals the congestion rent only when the market carries no
        # exogenous schedule, because a schedule's leg is settled at the
        # market's price but sits in neither column. The congestion rent is
        # the canon's R = sum_m |mu_m F^M_m| on the market's own book flow.
        settle.loc[f'SETTLEMENT RESIDUAL {eng.name} (load - gen)'] = [
            eng.name, '', '', '', '', '',
            round(sub['paid by load ($/h)'].sum() - sub['paid to gen ($/h)'].sum(), 1)]
        settle.loc[f'CONGESTION RENT {eng.name} (sum |mu F|)'] = [
            eng.name, '', '', '', '', '',
            round(sum(abs(float(r.line_dual.get(l, 0.0)) * float(r.flow_own[l]))
                      for l in pt_.lines), 1)]
    pf = physical_flows(pt_, [r for _, r in pairs], exo_physical=False).rename(
        columns={'F_phys': 'F^s (MW)'})
    lines = pf[['from', 'to', 'F[Market A]', 'F[Market B]', 'F^s (MW)',
                'rating', 'loading_%', 'overload']].copy()
    for _c in ('F[Market A]', 'F[Market B]', 'F^s (MW)', 'rating'):
        lines[_c] = lines[_c].astype(float).round(1)
    lines['loading_%'] = lines['loading_%'].astype(float).round(0)
    off = offsets or {}
    for eng, r in pairs:
        own = {l: float(r.flow_own[l]) for l in pt_.lines}
        offs = {l: float(off.get(eng.name, {}).get(l, 0.0)) for l in pt_.lines}
        book = {l: own[l] + offs[l] for l in pt_.lines}
        mu = {l: abs(float(r.line_dual.get(l, 0.0))) for l in pt_.lines}
        lines[f'book flow {eng.name} (MW)'] = [round(book[l], 1) for l in lines.index]
        lines[f'|mu| {eng.name} ($/MWh)'] = [round(mu[l], 2) for l in lines.index]
        # the rent is |mu| times the market's OWN flow component F^M_m, which
        # is what the settlement's load-minus-generation collects; the value
        # of a reserved offset (|mu| times the neighbour's forecast flow the
        # market holds headroom for) is reported in its own column (2026-10-17;
        # before, the rent column read |mu| x |F^M_m + offset| and disagreed
        # with the settlement row by exactly the offset's value)
        lines[f'rent {eng.name} ($/h)'] = [round(mu[l] * abs(own[l]), 1) for l in lines.index]
        if any(abs(v) > 1e-9 for v in offs.values()):
            lines[f'reservation value {eng.name} ($/h)'] = [round(mu[l] * abs(offs[l]), 1)
                                                            for l in lines.index]
    if CFG['save_enabled']:
        os.makedirs(CFG['export_dir'], exist_ok=True)
        pre = CFG['export_prefix']
        fs = os.path.join(CFG['export_dir'], f'{pre}{slug}_settlement.csv')
        fl = os.path.join(CFG['export_dir'], f'{pre}{slug}_lines.csv')
        settle.to_csv(fs, encoding='utf-8-sig')
        lines.to_csv(fl, encoding='utf-8-sig')
        print(f'wrote {fs} and {fl}')
    try:
        display(settle)  # type: ignore[name-defined]  # the notebook's builtin
        display(lines)  # type: ignore[name-defined]  # the notebook's builtin
    except NameError:
        pass
    return settle, lines


def price_walk(
    pt_: PTDFData,
    eng: MarketEngine,
    res: EngineResult,
    bus: Bus,
    tol: float = 1e-6,
    direction: Literal['withdrawal', 'injection'] = 'withdrawal',
) -> tuple[pd.DataFrame, list[Line]]:
    """The production-cost change behind one node's price.

    Serving one more MW of load at bus ``bus`` redispatches the market's
    marginal units subject to its binding constraints. The redispatch
    ``dg_i`` solves the balance row ``sum_i dg_i = 1`` together with one row
    per binding line ``m``, ``sum_i SF_{bus(i),m} dg_i = SF_{bus,m}``, and the
    node's price is the cost of that redispatch, ``sum_i c_i dg_i``.

    ``direction`` selects which increment is walked, and the two readings are
    the same arithmetic with the sign of the increment reversed.

    1. ``'withdrawal'`` (the default) adds one MW of load at the bus. The
       market's marginal units are redispatched UP, production cost RISES, and
       the total is ``+res.lmp[bus]``: the marginal cost of serving the node.
    2. ``'injection'`` adds one MW of generation at the bus. The right-hand
       side becomes ``-1`` and ``-SF_{bus,m}``, so every redispatch reverses:
       the market's own units are backed DOWN, production cost FALLS, and the
       total is ``-res.lmp[bus]``. The node's price is therefore the marginal
       value of generation at the bus, the saving the market realizes from a
       megawatt delivered there.

    The calculation reads the market's own offers and the shared shift
    factors only, so it prices a bus INSIDE the market's footprint and a bus
    OUTSIDE it by the same arithmetic. Returns the per-unit rows and the
    total, which reproduces ``res.lmp[bus]`` up to that sign;
    :func:`walk_text` prints them.
    """
    if direction not in ('withdrawal', 'injection'):
        raise ValueError("direction must be 'withdrawal' or 'injection'")
    sgn = -1.0 if direction == 'injection' else 1.0
    binding = [l for l, v in res.line_dual.items() if abs(v) > tol]
    marginal = [g for g, v in res.dispatch.items()
                if tol < v < float(eng.gens[g]['p_nom']) - tol]
    if not marginal:
        raise ValueError(f"{eng.name}: no interior marginal unit at bus {bus}; "
                         "the price is set at a bound and has no redispatch walk")
    rows_A = [[1.0] * len(marginal)]
    rhs = [sgn]
    for l in binding:
        li = pt_.line_idx[l]
        rows_A.append([pt_.ptdf[li, pt_.bus_idx[str(eng.gens[g]['bus'])]] for g in marginal])
        rhs.append(sgn * pt_.ptdf[li, pt_.bus_idx[str(bus)]])
    dg, *_ = np.linalg.lstsq(np.asarray(rows_A, float), np.asarray(rhs, float), rcond=None)
    out = pd.DataFrame([{
        'unit': g,
        'bus': str(eng.gens[g]['bus']),
        'offer ($/MWh)': float(eng.gens[g]['cost']),
        'redispatch (MW)': float(dg[i]),
        'cost change ($/h)': float(eng.gens[g]['cost']) * float(dg[i]),
    } for i, g in enumerate(marginal)]).set_index('unit')
    out.loc['TOTAL'] = ['', '', out['redispatch (MW)'].sum(),
                        out['cost change ($/h)'].sum()]
    return out, binding


def walk_text(
    walk: pd.DataFrame,
    bus: Bus,
    market: str,
    lmp: float,
    inside: bool,
    direction: Literal['withdrawal', 'injection'] = 'withdrawal',
    binding: Collection[Line] | None = None,
    label_width: int = 52,
) -> str:
    """The walk as a block of text, one line per marginal resource.

    Each line multiplies the resource's redispatch by its own offer, in the
    order equation (4) multiplies them, and the block closes on the total, which
    is the node's price. The block is the exhibit for this calculation, so it is
    written to be read and copied straight out of the cell output.
    """
    body = walk.drop(index='TOTAL')
    what = ('the value of one more MW of generation'
            if direction == 'injection' else 'the cost of one more MW of load')
    where = 'inside' if inside else 'outside'
    out = [f"{market} -- bus {bus}, {where} its footprint: {what}", ""]
    for unit, r in body.iterrows():
        lhs = (f"{unit} at bus {r['bus']}: "
               f"{float(r['redispatch (MW)']):+.4f} MW"
               f"  x  ${float(r['offer ($/MWh)']):,.2f}/MWh")
        out.append(f"  {lhs:<{label_width}}=  {_money(float(r['cost change ($/h)']))}/h")
    total = float(walk.loc['TOTAL', 'cost change ($/h)'])
    out.append("  " + "-" * (label_width + 14))
    out.append(f"  {'change in production cost':<{label_width}}=  {_money(total)}/h")
    note = '' if not binding else f"   (binding: {', '.join(map(str, binding))})"
    label = f"{market}'s price at bus {bus}"
    out.append(f"  {label:<{label_width}}=  {_money(lmp)}/MWh{note}")
    return "\n".join(out)


def _money(v: float) -> str:
    """A signed dollar figure in a fixed width, so a column of them lines up."""
    return f"{'-$' if v < 0 else ' $'}{abs(v):>9,.2f}"


def strip(fig: Figure, size: tuple[float, float] | None = None) -> Figure:
    """The figure without the suptitle, panel titles and legends its slide carries.

    The strip hides the suptitle, empties every axes title at all three title
    locations, removes every axes legend and every figure legend, and sets the
    canvas when ``size`` is given. A frame's legend note sits inside its figure
    legend and is removed with it. Text drawn with ``ax.text`` or
    ``ax.annotate``, an inset's label among it, is kept. The strip is
    idempotent, so a figure stripped before its inline display is stripped again
    by :func:`save_clean` with no further change.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to strip, changed in place.
    size : tuple of float, optional
        The canvas ``(width, height)`` in inches. Default ``None`` keeps the
        canvas as drawn.

    Returns
    -------
    matplotlib.figure.Figure
        ``fig`` itself, stripped.

    """
    sup = getattr(fig, "_suptitle", None)  # a private attribute, not in the stubs
    if sup is not None:
        sup.set_visible(False)
    for ax_ in fig.axes:
        for loc in _TITLE_LOCS:
            ax_.set_title("", loc=loc)
        leg = ax_.get_legend()
        if leg is not None:
            leg.remove()
    for leg_ in list(fig.legends):
        leg_.remove()
    if size is not None:
        fig.set_size_inches(*size)
    return fig


def save_clean(
    fig: Figure, name: str, size: tuple[float, float] | None = None
) -> None:
    """The deck PNG copy of one exhibit, written only when saving is enabled.

    The copy is the figure after :func:`strip` at the canvas ``size``, saved at
    ``CFG['export_dpi']`` with a transparent background and a full-canvas
    bounding box, so every PNG of a graph type lands at the same pixel size. No
    file is written unless the notebook set ``configure(save_enabled=True)``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The exhibit, stripped and resized in place.
    name : str
        The file stem, written as ``<export_dir>/<name>.png``.
    size : tuple of float, optional
        The canvas ``(width, height)`` in inches. Default
        ``CFG['export_size']``.

    """
    if not CFG['save_enabled']:
        return
    size = size or CFG['export_size']
    os.makedirs(CFG['export_dir'], exist_ok=True)
    strip(fig, size)
    # explicit full-canvas bbox: the inline backend's tight-bbox default would
    # otherwise crop/expand each figure to its content and break the uniform size
    from matplotlib.transforms import Bbox
    import time as _time
    final = os.path.join(CFG['export_dir'], f'{name}.png')
    tmp = os.path.join(CFG['export_dir'], f'.{name}.tmp.png')
    fig.savefig(tmp, dpi=CFG['export_dpi'], transparent=True,
                bbox_inches=Bbox([[0, 0], [size[0], size[1]]]))  # type: ignore[arg-type]  # the lists join to object
    # Windows: an exhibit open in a viewer, an Explorer preview or a scan can hold the
    # old file for a moment; write beside it and swap it in, retrying briefly.
    for _try in range(10):
        try:
            os.replace(tmp, final)
            break
        except OSError:
            if _try == 9:
                raise
            _time.sleep(0.5)
    print(f"saved {CFG['export_dir']}/{name}.png")
