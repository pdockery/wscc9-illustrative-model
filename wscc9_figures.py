# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Teaching figure compositions for the WSCC 9-bus notebooks.

``nodal_plot`` provides the reusable primitives (network topology panel +
circlize nodal-dispatch panel + ``plot_combined_letter`` that puts them side by
side). This module holds the teaching *compositions* the notebooks reuse — one
clearing drawn as a [network + nodal dispatch] pair, coloured by footprint, with
the standard shared legend.

* :func:`footprint_figure` — the composite for one clearing. Colour one
  footprint and grey the rest (``highlight=`` or explicit ``dim_buses=``), draw
  ties/managed lines in footprint colours, put a footprint band on the circlize
  ring, and annotate any self-schedule on the network panel. This single
  function replaces the per-notebook ``example_figure`` (congestion notebooks,
  greys the other BA) and ``market_figure`` (seams notebook, greys the other
  market).
* :func:`composite_figure` — the unified two-panel view (nothing greyed), the
  figure each downstream notebook prints in its set-up cell.
* :func:`draw_net_dispatch` / :func:`transfer_figure` — the "Transfers" inset
  (each footprint as one bubble, the interchange as a flow/limit label) and the
  composite carrying it.
"""

from __future__ import annotations

from math import pi

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import nodal_plot
from nodal_plot import plot_combined_letter, compute_ptdf_flows, plot_network_topology
from seams_engine import (
    to_supply_demand, susceptance_widths, shed_segments, served_demand,
)
from wscc9_model import BUS_COLORS, COORDS, RING_ORDER, CENTER_BUS

#: Tier colours for sold transmission rights drawn on the network.
RIGHT_TIER_COLORS = {"inter": "#117A65", "intra": "#B9770E"}
_WECC_PATH = "#B8860B"   # gold: a WECC-rated inter-BA path (tie)

_DIM = "#C8CCCE"        # greyed (out-of-focus) footprint
_UNASSIGNED = "#AAB7B8"  # line assigned to nobody
_SELFSCHED = "#95A5A6"   # self-schedule: a price-taking block the dispatch did not optimise


def footprint_figure(
    net, pt, fp, engine, res, *,
    highlight=None, dim_buses=None, exo_sched=None, sup_dem=None,
    bus_colors=None, demand_segments=None, shed=True, legend_note=None, suptitle=None,
    annotate_roles=True, axis_key=True,
    node_net_mw=None, network_show_lmp=None,
    bus_coords=None, ring_order=None, center_bus=None,
    title_left="Network -- DC power flow",
    title_right="Nodal dispatch -- merit order, demand & flows",
    figsize=(11, 6.2),
    panel_ratios=(1, 1),
):
    """Combined network + nodal-dispatch composite for one clearing.

    Parameters
    ----------
    net, pt : pypsa.Network, PTDFData
        The network and its shift factors.
    fp : footprints.Footprints
        The partition (line colours, bands, footprint membership).
    engine, res : MarketEngine, EngineResult
        The cleared engine and its result.
    highlight : str, optional
        Footprint to keep in colour; every other footprint's buses are greyed.
        (Mutually exclusive with ``dim_buses``.)
    dim_buses : iterable, optional
        Explicit buses to grey (the congestion-notebook per-BA view).
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
    node_net_mw : True or dict, optional
        Annotate each network node with its NET injection (+) / withdrawal (−)
        instead of (or alongside) the LMP. ``True`` reads the clearing's own
        ``res.injection`` (dispatched gen − served load + exo); a dict ``{bus: MW}``
        overrides it. Pairs with ``network_show_lmp`` (default then: hide the
        network-panel LMP, since price is read off the dispatch ring).
    network_show_lmp : bool, optional
        Whether the NETWORK panel prints the bus LMP. Default ``True`` unless
        ``node_net_mw`` is given (then ``False``). The dispatch ring always keeps
        its LMP labels.
    """
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
    else:
        sup, dem = to_supply_demand(engine, res)
        for b, mw in (exo_sched or {}).items():
            b = str(b)
            if mw >= 0:
                # A self-schedule is price-taking and not optimised: draw it grey,
                # with the bar AT the bus LMP (price=lmp, no bid) so it sits on the
                # dashed LMP line rather than showing a marginal-cost/rent gap.
                sup.setdefault(b, []).append({"unit_id": "self_schedule",
                                              "price": res.lmp[b], "volume": mw,
                                              "capacity": mw, "accepted_volume": mw,
                                              "color": _SELFSCHED})
                has_selfsched = True
            else:
                dem[b] = dem.get(b, 0.0) - mw

    if demand_segments is None and shed and sup_dem is None:
        demand_segments = shed_segments(res, dem) or None
    has_shed = demand_segments is not None

    # flow tracing: on served demand when tracking shed, else on the drawn demand
    if sup_dem is None and shed:
        flow_dem = served_demand(res, dem)
    else:
        flow_dem = dem
    flows = compute_ptdf_flows(net, sup, flow_dem)

    # which buses to grey
    if dim_buses is not None:
        dim = {str(b) for b in dim_buses}
    elif highlight is not None:
        dim = {b for b in pt.buses if fp.fp_of(b) != highlight}
    else:
        dim = set()

    colors = dict(BUS_COLORS if bus_colors is None else bus_colors)
    for b in dim:
        colors[b] = _DIM

    # line colours: a managed line takes its footprint's colour; grey a line
    # between greyed buses or assigned to a fully-greyed footprint.
    lcolors = fp.line_colors(pt)
    dim_fps = {name for name in fp.names if all(str(b) in dim for b in fp.defs[name])}
    for l in pt.lines:
        i = pt.line_idx[l]
        b0, b1 = pt.line_buses[i]
        if (b0 in dim and b1 in dim) or fp.line_assign.get(l) in dim_fps:
            lcolors[l] = _DIM

    gcolors = {name: (_DIM if name in dim_fps else fp.colors[name]) for name in fp.names}
    binding = {l for l, mu in res.line_dual.items() if abs(mu) > 1e-3}

    fig, (ax_net, ax_circ) = plot_combined_letter(
        net, sup, dem,
        bus_colors=colors, bus_lmps=res.lmp,
        bus_net_mw=bus_net, network_show_lmp=network_show_lmp,
        line_flows={l: res.flow_own[l] for l in pt.lines},
        line_widths=susceptance_widths(pt), line_colors=lcolors,
        constrained_lines=binding,
        flows=flows,
        clearing_price=res.energy_price,
        demand_segments=demand_segments,
        lmp_line=True, bus_groups=fp.groups(pt), group_colors=gcolors,
        show_group_labels=False,
        annotate_roles=annotate_roles, axis_key=axis_key,
        all_buses=pt.buses,
        sector_order=ring_order if ring_order is not None else RING_ORDER,
        bus_coords=bus_coords if bus_coords is not None else COORDS,
        center_bus=center_bus if center_bus is not None else CENTER_BUS,
        title_left=title_left, title_right=title_right, suptitle=suptitle,
        figsize=figsize, panel_ratios=panel_ratios,
    )

    _coords = bus_coords if bus_coords is not None else COORDS
    for b, mw in (exo_sched or {}).items():
        kind = "self-schedule import" if mw >= 0 else "self-schedule export"
        ax_net.annotate(f"{kind} {abs(mw):.0f} MW", _coords[str(b)], fontsize=8,
                        fontweight="bold", color="#B03A2E", xytext=(0, 26),
                        textcoords="offset points", ha="center",
                        bbox=dict(boxstyle="round", fc="#FDEDEC", ec="#B03A2E"))

    handles = [mpatches.Patch(fc=fp.colors[name], ec="#555", label=f"{name} lines / band")
               for name in fp.names]
    if any(fp.line_assign.get(l) is None for l in pt.lines):
        handles.append(mpatches.Patch(fc=_UNASSIGNED, ec="#555",
                                      label="Line assigned to neither footprint"))
    if has_selfsched:
        handles.append(mpatches.Patch(fc=_SELFSCHED, ec="#555",
                                      label="Self-schedule (price-taking; bar at LMP, not optimised)"))
    if has_shed:
        handles.append(mpatches.Patch(fc="#888", alpha=0.3, ec="#555",
                                      label="Load shed (unserved -- bus colour, faint; LMP = SHED_PRICE)"))
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


def composite_figure(net, pt, fp, engine, res, *, suptitle=None, **kw):
    """The unified two-panel composite (nothing greyed) — the set-up figure each
    downstream notebook prints. Thin wrapper over :func:`footprint_figure`."""
    return footprint_figure(net, pt, fp, engine, res, suptitle=suptitle, **kw)


def transfer_inset(ax, labels, tam, surplus, E, ebar, muT, colors, tie_cap, title="Transfers"):
    """The "Transfers" inset, drawn from **explicit** values so the SAME transfer
    summary can overlay ANY composite (it never reads ``fp``/``res``).

    Two bubbles — ``labels = (left, right)`` — each at the network panel's
    bus-bubble scale (s = max(3·MW, 200), MW from ``tam[label]`` = the addressable
    market) and labelled with its net ``surplus[label]``. The connecting line
    carries the transfer ``E``/``ebar`` (flow/limit); its width is ``ebar``
    relative to ``tie_cap`` (thin = tight). When ``muT`` ≠ 0 the line turns red and
    ``|muT|`` prints below it. ``E`` is signed left→right (positive = left exports).
    """
    names = list(labels)
    size = {n: max(tam[n] * 3.0, 200.0) for n in names}
    fig = ax.figure
    pos = ax.get_position()
    w_pts = pos.width * fig.get_size_inches()[0] * 72.0
    h_pts = pos.height * fig.get_size_inches()[1] * 72.0
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


def draw_net_dispatch(ax, fp, res, ebar, engine, tie_cap):
    """The "Transfers" inset for a footprint pair cleared with an ``interchange``
    constraint — computes the bubble sizes / surpluses / E / μ_T from ``fp`` and
    ``res`` and defers the drawing to :func:`transfer_inset`."""
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
    transfer_inset(ax, (names[0], names[1]), tam, surplus,
                   float(res.interchange_mw or 0.0), ebar,
                   abs(res.interchange_dual or 0.0), fp.colors, tie_cap)


#: Default rectangle (figure fraction) for the Transfers inset in the network panel.
INSET_RECT = (0.005, 0.02, 0.19, 0.235)


def transfer_figure(net, pt, fp, engine, res, ebar, tie_cap, suptitle=None, **kw):
    """:func:`footprint_figure` plus the "Transfers" inset in the network panel's
    lower-left, at the network panel's own bubble scale. Extra keyword arguments
    (e.g. ``node_net_mw``, ``highlight``) pass through to :func:`footprint_figure`."""
    fig = footprint_figure(net, pt, fp, engine, res, suptitle=suptitle, **kw)
    draw_net_dispatch(fig.add_axes(INSET_RECT), fp, res, ebar, engine, tie_cap)
    return fig


def draw_rights_arcs(ax, rights, coords=None, *, tier_colors=None, rad=0.22,
                     label=True, atc_caps=None, label_mw=True, label_fontsize=7.5):
    """Overlay sold point-to-point transmission rights as directed POR->POD arcs.

    ``rights`` is a list of dicts ``{'source','sink','mw', 'tier'?, 'honored'?}``:
    a curved arrow from the source bus to the sink bus, width growing with ``mw``,
    coloured by ``tier`` ('inter' = inter-BA path right / WECC backbone, 'intra' =
    intra-BA right) via ``tier_colors`` (default :data:`RIGHT_TIER_COLORS`). A
    ``honored=False`` right is drawn dashed (a curtailed / unfunded right). Drawn
    ON TOP of an existing network panel (e.g. the ``ax`` returned by
    :func:`rights_figure` / ``plot_network_topology``) so the financial rights sit
    over the physical lines they load.

    ``atc_caps`` (the ATC layer) is an optional list aligned with ``rights``; when
    given, each arc's label reads ``MW / ATC`` -- the scheduled quantity against the
    path's standalone transfer capability -- so an arc near its own cap is visible.
    """
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    coords = COORDS if coords is None else coords
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    for ri, r in enumerate(rights):
        s, k, mw = str(r["source"]), str(r["sink"]), float(r["mw"])
        col = tcol.get(r.get("tier", "inter"), "#117A65")
        honored = r.get("honored", True)
        p0, p1 = np.array(coords[s]), np.array(coords[k])
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
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
            if label_mw:
                cap = None if atc_caps is None else atc_caps[ri]
                mwtag = f"{mw:.0f} MW" if cap is None else f"{mw:.0f} / {cap:.0f} MW"
                tag = f"{s}→{k}\n{mwtag}" + ("" if honored else "\n(curtailed)")
            else:
                # path only; the MW / ATC live in the table beside the graph
                tag = f"{s}→{k}" + ("" if honored else " (curt.)")
            ax.annotate(tag, mid + 0.5 * rad * bulge + 0.30 * bunit,
                        ha="center", va="center", fontsize=label_fontsize,
                        color=col, fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                                  lw=1, alpha=0.92))
    return ax


def rights_figure(net, pt, rights, *, fp=None, ties=None, coords=None,
                  monitored="all", title=None, bus_colors=None, tier_colors=None,
                  line_colors=None, line_widths=None, show_atc=False, show_sft=True,
                  supply_by_bus=None, demand_by_bus=None, bus_net_mw=None,
                  col_labels=None, atc_values=None, pct_values=None, font_scale=1.0):
    """The standard "transmission rights" view: network panel + rights table.

    Two panels side by side so the graph stays uncluttered and the numbers are
    readable:

    * **Left -- the network.** Each ``rights`` entry (see :func:`draw_rights_arcs`)
      is a POR->POD arc, width growing with MW, coloured by tier (inter-BA gold-path
      vs intra-BA), labelled by **path only** (``2->7``). The physical, *superposed*
      award flow rides on the lines as a direction arrow + ``flow / limit`` (the
      share of capacity); any line the rights overload (``show_sft``) turns **red**
      -- the simultaneous-feasibility test failing, path-by-path rights colliding on
      the shared grid.
    * **Right -- the rights table.** One row per right: ``Right`` (path, coloured to
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
    coords = COORDS if coords is None else coords
    tielist = list(ties) if ties is not None else (list(fp.ties) if fp is not None else [])
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    aw = [atc.Award(r["source"], r["sink"], r["mw"]) for r in rights]

    overl = set(atc.overloaded_lines(pt, aw, monitored)) if show_sft else set()
    # Base line colour: a caller-supplied map (e.g. by balancing authority) if given, else the
    # gold WECC-path / grey scheme. An overload (SFT failing) flags red and overrides either.
    def _base(l):
        if line_colors is not None:
            return line_colors.get(l, "#CBD0D3")
        return _WECC_PATH if l in tielist else "#CBD0D3"
    lcolors = {l: ("#C0392B" if l in overl else _base(l)) for l in pt.lines}

    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.02)
    axn = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")

    # left: the network. The superposed-award flow rides on the lines (direction +
    # flow/limit, red where the SFT fails); the rights are arcs labelled by path
    # only -- the MW / ATC live in the table on the right.
    plot_network_topology(
        net, bus_colors=BUS_COLORS if bus_colors is None else bus_colors, bus_coords=coords,
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
        caps = list(atc_values)
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
                    bbox=[0.0, 0.14, 1.0, 0.72], colWidths=cw)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    nrow = len(cells) + 1
    for (rr, cc), cell in tbl.get_celld().items():        # taller header row
        cell.set_edgecolor("#999")
        cell.set_height((1.8 if rr == 0 else 1.0) / (nrow + 0.8))
    for j in range(len(collab)):                          # header row
        tbl[(0, j)].set_text_props(fontweight="bold", color="white")
        tbl[(0, j)].set_facecolor("#34495E")
    for i, col in enumerate(pathcols):                    # path cell coloured to its arc
        tbl[(i + 1, 0)].get_text().set_color(col)
        tbl[(i + 1, 0)].get_text().set_fontweight("bold")
    return fig


def rights_payoff_figure(net, pt, rights, res, *, fp=None, coords=None,
                         bus_colors=None, line_colors=None, line_widths=None,
                         tier_colors=None, title=None,
                         supply_by_bus=None, demand_by_bus=None, bus_net_mw=None,
                         font_scale=1.0):
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

    coords = COORDS if coords is None else coords
    line_widths = susceptance_widths(pt) if line_widths is None else line_widths
    tcol = RIGHT_TIER_COLORS if tier_colors is None else tier_colors
    lmp = res.lmp
    binding = {l for l in pt.lines if abs(res.line_dual.get(l, 0.0)) > 1e-3}

    def _base(l):
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
        net, bus_colors=BUS_COLORS if bus_colors is None else bus_colors,
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
                    bbox=[0.0, 0.14, 1.0, 0.72], colWidths=[0.29, 0.29, 0.24, 0.18])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    nrow = len(cells) + 1                                  # header + data + total
    for (r, c), cell in tbl.get_celld().items():           # taller header row
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


def self_schedule_figure(summary, *, suptitle=None):
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
