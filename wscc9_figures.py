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
from nodal_plot import plot_combined_letter, compute_ptdf_flows
from seams_engine import (
    to_supply_demand, susceptance_widths, shed_segments, served_demand,
)
from wscc9_model import BUS_COLORS, COORDS, RING_ORDER, CENTER_BUS

_DIM = "#C8CCCE"        # greyed (out-of-focus) footprint
_UNASSIGNED = "#AAB7B8"  # line assigned to nobody
_SELFSCHED = "#95A5A6"   # self-schedule: a price-taking block the dispatch did not optimise


def footprint_figure(
    net, pt, fp, engine, res, *,
    highlight=None, dim_buses=None, exo_sched=None, sup_dem=None,
    bus_colors=None, demand_segments=None, shed=True, legend_note=None, suptitle=None,
    annotate_roles=True, axis_key=True,
    bus_coords=None, ring_order=None, center_bus=None,
    title_left="Network -- DC power flow",
    title_right="Nodal dispatch -- merit order, demand & flows",
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
    """
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


def transfer_figure(net, pt, fp, engine, res, ebar, tie_cap, suptitle=None):
    """:func:`footprint_figure` plus the "Transfers" inset in the network panel's
    lower-left, at the network panel's own bubble scale."""
    fig = footprint_figure(net, pt, fp, engine, res, suptitle=suptitle)
    draw_net_dispatch(fig.add_axes(INSET_RECT), fp, res, ebar, engine, tie_cap)
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
