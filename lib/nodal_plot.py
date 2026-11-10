"""
Nodal circlize visualization for ASSUME market results.

Each bus is a sector with a unique color (consistent with network diagrams):
- Left half (GEN): merit order staircase — width = capacity, height = bid price
  - Solid fill = accepted dispatch, faded = available but not cleared
  - Gen bars use the bus color (darker for accepted, lighter for unaccepted)
- Right half (LOAD): demand bar sized by MW, uses bus color
- Inner chords: power flows from gen to load, colored by source bus
- Inner ring: bus color, solid

In a pay-as-clear (copperplate) market, flows are proportional:
local gen serves local load first, surplus distributed to deficit buses.
In nodal markets (Phase 4), actual OPF line flows replace these.

Portable: only depends on pycirclize, pandas, matplotlib, numpy.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.colors import to_rgba, to_hex
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from pycirclize import Circos
from pycirclize.utils.plot import get_label_params_by_rad

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence

    import pypsa
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.projections.polar import PolarAxes

    from lab_types import Bus, BusValues, Line, LineValues, Loads, Resource

__all__ = [
    "BUS_PALETTE",
    "IEEE9_COORDS",
    "TRANSIT_COLOR",
    "add_arc_label_clearance",
    "assign_bus_colors",
    "compute_proportional_flows",
    "compute_ptdf_flows",
    "extract_line_flows",
    "plot_combined_letter",
    "plot_network_topology",
    "plot_nodal_circlize",
    "plot_nodal_from_results",
    "prepare_bus_data",
    "rotate_coords",
    "trace_gen_to_load",
]


# --- 9+ distinct bus colors (works for any network up to ~12 buses) ---
# Designed for visual distinction on both circlize and network topology plots.
BUS_PALETTE = [
    '#E74C3C',  # red
    '#3498DB',  # blue
    '#2ECC71',  # green
    '#F39C12',  # orange
    '#9B59B6',  # purple
    '#1ABC9C',  # teal
    '#E67E22',  # dark orange
    '#34495E',  # dark slate
    '#E91E63',  # pink
    '#00BCD4',  # cyan
    '#8BC34A',  # lime
    '#FF5722',  # deep orange
]

TRANSIT_COLOR = '#BDC3C7'  # gray for buses with no gen or load


def _darken(color: str, factor: float = 0.7) -> str:
    """Darken a color for borders/edges."""
    r, g, b, a = to_rgba(color)
    return to_hex((r * factor, g * factor, b * factor, a))


def assign_bus_colors(
    all_buses: list[Bus],
    supply_by_bus: dict[Bus, list[dict[str, Any]]] | None = None,
    demand_by_bus: Loads | None = None,
) -> dict[Bus, str]:
    """
    Assign a unique color to each bus. Transit buses get gray.

    Parameters
    ----------
    all_buses : list of str
    supply_by_bus : dict, optional
    demand_by_bus : dict, optional

    Returns
    -------
    dict : {bus: color_hex}
    """
    supply_by_bus = supply_by_bus or {}
    demand_by_bus = demand_by_bus or {}

    colors = {}
    palette_idx = 0
    for bus in all_buses:
        has_gen = bus in supply_by_bus and len(supply_by_bus.get(bus, [])) > 0
        has_load = bus in demand_by_bus and demand_by_bus.get(bus, 0) > 0
        if has_gen or has_load:
            colors[bus] = BUS_PALETTE[palette_idx % len(BUS_PALETTE)]
            palette_idx += 1
        else:
            colors[bus] = TRANSIT_COLOR
    return colors


def compute_proportional_flows(
    supply_by_bus: dict[Bus, list[dict[str, Any]]], demand_by_bus: Loads
) -> list[tuple[Bus, Bus, float]]:
    """
    Compute proportional flows for copperplate (pay-as-clear) markets.

    1. At mixed buses (both gen and load), local gen serves local load first.
       This self-serving flow is included as a same-bus chord.
    2. Remaining surplus is exported proportionally to all deficit buses.

    Returns [(src_bus, dst_bus, flow_MW), ...] including self-serving flows.
    """
    all_buses = set(list(supply_by_bus.keys()) + list(demand_by_bus.keys()))

    # Compute gen, load, and self-serve at each bus
    bus_gen = {}
    bus_load = {}
    self_serve = {}
    for bus in all_buses:
        g = sum(g_['accepted_volume'] for g_ in supply_by_bus.get(bus, []))
        d = demand_by_bus.get(bus, 0)
        bus_gen[bus] = g
        bus_load[bus] = d
        self_serve[bus] = min(g, d)

    # After self-serving: net surplus and deficit
    surplus = {}
    deficit = {}
    for bus in all_buses:
        remaining_gen = bus_gen[bus] - self_serve[bus]
        remaining_load = bus_load[bus] - self_serve[bus]
        if remaining_gen > 0.5:
            surplus[bus] = remaining_gen
        if remaining_load > 0.5:
            deficit[bus] = remaining_load

    total_deficit = sum(deficit.values())

    flows = []

    # Self-serving flows (same bus, gen→load)
    for bus, ss in self_serve.items():
        if ss > 0.5:
            flows.append((bus, bus, round(ss, 1)))

    # Surplus → deficit proportionally
    if total_deficit > 0:
        for src, surplus_mw in surplus.items():
            for dst, deficit_mw in deficit.items():
                flow = surplus_mw * (deficit_mw / total_deficit)
                if flow > 0.5:
                    flows.append((src, dst, round(flow, 1)))

    return flows


def extract_line_flows(
    flows_dict: dict[tuple[Line, int], float],
    grid_data: dict[str, Any],
    hour_idx: int,
) -> list[tuple[Bus, Bus, float]]:
    """
    Extract actual line flows from Egret results for a given hour.

    Converts Egret's (line_name, t) → MW dict into the
    [(src_bus, dst_bus, flow_MW), ...] format used by plot_nodal_circlize.

    Positive flow means from_bus → to_bus; negative is reversed.

    Parameters
    ----------
    flows_dict : dict
        {(line_name, t_idx): flow_MW} from clear_sced() results.
    grid_data : dict
        Must have 'lines' DataFrame with 'bus0', 'bus1' columns.
    hour_idx : int
        Time period index to extract.

    Returns
    -------
    list of (src_bus, dst_bus, abs_flow_MW)
    """
    lines_df = grid_data['lines']
    result = []
    for line_name in lines_df.index:
        flow = flows_dict.get((str(line_name), hour_idx), 0.0)
        if abs(flow) < 0.5:
            continue
        bus0 = str(lines_df.loc[line_name, 'bus0'])
        bus1 = str(lines_df.loc[line_name, 'bus1'])
        if flow > 0:
            result.append((bus0, bus1, round(abs(flow), 1)))
        else:
            result.append((bus1, bus0, round(abs(flow), 1)))
    return result


def trace_gen_to_load(
    line_flows: list[tuple[Bus, Bus, float]],
    gen_by_bus: BusValues,
    load_by_bus: Loads,
) -> list[tuple[Bus, Bus, float]]:
    """
    Proportional power tracing (Bialek's method) to compute gen-to-load flows.

    Given actual directed line flows and bus-level generation/load, compute
    how much of each generator bus's output serves each load bus.

    Algorithm:
    1. Build directed flow graph from line flows.
    2. At each bus, gross inflow = local generation + sum of incoming flows.
    3. Each outgoing flow carries the same proportional mix of sources.
    4. Solve the linear system to get each generator's share at each bus.
    5. Load at each bus is served proportionally by the shares present.

    Parameters
    ----------
    line_flows : list of (src_bus, dst_bus, flow_MW)
        Directed line flows (positive = src→dst direction).
    gen_by_bus : dict
        {bus: total_generation_MW} for buses with generation.
    load_by_bus : dict
        {bus: total_load_MW} for buses with load.

    Returns
    -------
    list of (gen_bus, load_bus, flow_MW)
        Gen-to-load allocation suitable for circlize chords.
    """
    import numpy as np

    # Collect all buses
    all_buses = sorted(set(
        list(gen_by_bus.keys()) +
        list(load_by_bus.keys()) +
        [s for s, d, f in line_flows] +
        [d for s, d, f in line_flows]
    ))
    bus_idx = {b: i for i, b in enumerate(all_buses)}
    n = len(all_buses)

    # Build inflow matrix: inflow[i][j] = flow from bus j to bus i
    inflow = np.zeros((n, n))
    for src, dst, flow_mw in line_flows:
        if flow_mw > 0.01:
            i_src = bus_idx[src]
            i_dst = bus_idx[dst]
            inflow[i_dst][i_src] += flow_mw

    # Gross through-flow at each bus = local gen + incoming line flows
    gross = np.zeros(n)
    for bus, gen_mw in gen_by_bus.items():
        gross[bus_idx[bus]] += gen_mw
    for dst_i in range(n):
        gross[dst_i] += inflow[dst_i].sum()

    # Build proportional sharing matrix A:
    # share[g_bus][bus] = fraction of power at 'bus' that originated from 'g_bus'
    #
    # At bus j: share[g][j] = (gen_g_at_j / gross_j) +
    #           sum over upstream i: (inflow[j][i] / gross_j) * share[g][i]
    #
    # This is: S = D_gen + D_inflow * S  =>  (I - D_inflow) * S = D_gen
    # where D_inflow[j][i] = inflow[j][i] / gross[j]

    gen_buses = sorted(gen_by_bus.keys())
    n_gen = len(gen_buses)

    # Build the system for each gen bus
    # (I - D_inflow) * s_g = d_g  where d_g[j] = gen_mw / gross[j] if j is gen bus g
    D_inflow = np.zeros((n, n))
    for j in range(n):
        if gross[j] > 0.01:
            for i in range(n):
                D_inflow[j][i] = inflow[j][i] / gross[j]

    A = np.eye(n) - D_inflow

    # Solve for each generator's share vector
    gen_shares = {}  # {gen_bus: array of shares at each bus}
    for g_bus in gen_buses:
        g_idx = bus_idx[g_bus]
        rhs = np.zeros(n)
        if gross[g_idx] > 0.01:
            rhs[g_idx] = gen_by_bus[g_bus] / gross[g_idx]
        try:
            shares = np.linalg.solve(A, rhs)
            gen_shares[g_bus] = np.maximum(shares, 0)  # clip numerical noise
        except np.linalg.LinAlgError:
            gen_shares[g_bus] = np.zeros(n)

    # Compute gen-to-load allocations
    result = []
    for g_bus in gen_buses:
        for l_bus, load_mw in load_by_bus.items():
            l_idx = bus_idx[l_bus]
            # Load at l_bus served by g_bus = load_mw * share_of_g at l_bus
            alloc = load_mw * gen_shares[g_bus][l_idx]
            if alloc > 0.5:
                result.append((g_bus, l_bus, round(alloc, 1)))

    return result


def compute_ptdf_flows(
    network: pypsa.Network,
    supply_by_bus: dict[Bus, list[dict[str, Any]]],
    demand_by_bus: Loads,
) -> list[tuple[Bus, Bus, float]]:
    """
    Compute gen-to-load flow allocation using PTDF-based proportional tracing.

    1. Compute PTDF matrix from network impedances.
    2. Compute net injection vector from gen dispatch and load.
    3. Compute actual line flows: f = PTDF * P_net.
    4. Decompose each line flow by generator: f_g_l = PTDF(l, bus_g) * P_g.
       The slack bus contribution is the residual (total flow - sum of others).
    5. At each bus, compute the fraction of incoming power from each generator
       (proportional sharing at junctions), then allocate load accordingly.

    Parameters
    ----------
    network : pypsa.Network
        Must have buses and lines with reactance (x_pu_eff or x).
    supply_by_bus : dict
        {bus: [{'unit_id', 'price', 'volume', 'accepted_volume'}, ...]}
    demand_by_bus : dict
        {bus: total_demand_MW}

    Returns
    -------
    list of (gen_bus, load_bus, flow_MW)
        Gen-to-load allocation suitable for circlize chords.
    """
    buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))
    lines = network.lines
    n_bus = len(buses)
    n_line = len(lines)
    bus_idx = {b: i for i, b in enumerate(buses)}

    # Line susceptances (1/x) and incidence matrix
    b_line = np.zeros(n_line)
    C = np.zeros((n_line, n_bus))
    line_buses = []  # (bus0, bus1) for each line

    for li, (line_name, row) in enumerate(lines.iterrows()):
        x = row.get('x_pu_eff', row.get('x_pu', row.get('x', 0.01)))
        if abs(x) < 1e-10:
            x = 0.01
        b_line[li] = 1.0 / x
        bus0 = str(row['bus0'])
        bus1 = str(row['bus1'])
        C[li, bus_idx[bus0]] = 1.0
        C[li, bus_idx[bus1]] = -1.0
        line_buses.append((bus0, bus1))

    # Bus susceptance matrix B = C^T * diag(b) * C
    B_bus = C.T @ np.diag(b_line) @ C

    # Remove slack bus (first bus) to make B invertible
    slack_idx = 0
    slack_bus = buses[slack_idx]
    keep = [i for i in range(n_bus) if i != slack_idx]
    B_reduced = B_bus[np.ix_(keep, keep)]

    try:
        B_inv = np.linalg.inv(B_reduced)
    except np.linalg.LinAlgError:
        return compute_proportional_flows(supply_by_bus, demand_by_bus)

    # Expand B_inv back to full size (slack row/col = 0)
    B_inv_full = np.zeros((n_bus, n_bus))
    for ii, i in enumerate(keep):
        for jj, j in enumerate(keep):
            B_inv_full[i, j] = B_inv[ii, jj]

    PTDF = np.diag(b_line) @ C @ B_inv_full  # (n_line x n_bus)

    # Collect gen dispatch by bus
    gen_dispatch = {}
    for bus, gens in supply_by_bus.items():
        total = sum(g['accepted_volume'] for g in gens)
        if total > 0.1:
            gen_dispatch[bus] = total

    gen_buses = sorted(gen_dispatch.keys(), key=lambda x: int(x))
    if not gen_buses:
        return []

    # Net injection vector: gen - load at each bus
    P_net = np.zeros(n_bus)
    for bus, mw in gen_dispatch.items():
        P_net[bus_idx[bus]] += mw
    for bus, mw in demand_by_bus.items():
        P_net[bus_idx[bus]] -= mw

    # Copperplate co-located service. A generator and a load at the SAME bus are
    # joined by a zero-impedance node, not a line — there is no shift factor from
    # gen to its own load — so the bus's own gen serves its own load FIRST, and
    # only the NET injection (gen - load) enters the PTDF trace below. The locally
    # served amount min(gen, load) becomes a same-bus self-loop chord. This keeps
    # the tracing methodology identical (it just runs on net injections, as a
    # shift factor is defined) and stops a net-sink bus from "exporting" its gen.
    local_service = {b: min(gen_dispatch.get(b, 0.0), demand_by_bus.get(b, 0.0))
                     for b in set(gen_dispatch) | set(demand_by_bus)}
    net_gen = {b: gen_dispatch[b] - local_service.get(b, 0.0) for b in gen_dispatch}

    # Total line flows: f = PTDF * P_net  (P_net already nets gen against load)
    line_flows = PTDF @ P_net  # positive = bus0→bus1

    # Decompose each line flow by generator using PTDF
    # For non-slack gen g: contribution to line l = PTDF(l, bus_g) * P_g
    # Slack gen gets the residual: f_l - sum(non-slack contributions)
    # This ensures contributions sum exactly to total line flow.
    # flow_by_gen[l][gen_bus] = MW contribution of gen to line l (signed)
    flow_by_gen = np.zeros((n_line, len(gen_buses)))
    gen_bus_idx_map = {b: gi for gi, b in enumerate(gen_buses)}

    for gi, g_bus in enumerate(gen_buses):
        if g_bus == slack_bus:
            continue
        g_idx = bus_idx[g_bus]
        for li in range(n_line):
            flow_by_gen[li, gi] = PTDF[li, g_idx] * gen_dispatch[g_bus]

    # Slack gets residual
    if slack_bus in gen_bus_idx_map:
        si = gen_bus_idx_map[slack_bus]
        for li in range(n_line):
            flow_by_gen[li, si] = line_flows[li] - sum(
                flow_by_gen[li, gi] for gi in range(len(gen_buses)) if gi != si
            )

    # Now use proportional tracing with PTDF-decomposed line flows.
    # At each bus, incoming power = local gen + incoming line flows (by gen).
    # Each generator's share of outgoing flows is proportional to its share
    # of total incoming power at that bus.
    #
    # We solve this iteratively in topological order (upstream → downstream).
    # gen_share[bus][gen_bus] = fraction of power at bus from gen_bus

    # Build directed flow graph from line flows
    # For each line: if flow > 0, it goes bus0→bus1; if < 0, bus1→bus0
    # incoming[bus] = list of (from_bus, line_idx, abs_flow, direction_sign)
    incoming: dict[Bus, list[tuple[Bus, int, float]]] = {b: [] for b in buses}
    for li in range(n_line):
        bus0, bus1 = line_buses[li]
        f = line_flows[li]
        if abs(f) < 0.01:
            continue
        if f > 0:
            # Flow from bus0 to bus1
            incoming[bus1].append((bus0, li, abs(f)))
        else:
            # Flow from bus1 to bus0
            incoming[bus0].append((bus1, li, abs(f)))

    # At each bus, compute the gen composition of incoming power
    # gen_share[bus] = {gen_bus: MW from that gen at this bus}
    gen_at_bus: dict[Bus, dict[Bus, float]] = {}
    for b in buses:
        gen_at_bus[b] = {}
        if net_gen.get(b, 0.0) > 0.1:        # only NET (surplus) gen enters the network
            gen_at_bus[b][b] = net_gen[b]

    # Iterative proportional sharing (Bialek). Each bus's throughput, decomposed by
    # gen origin, is its OWN net (surplus) gen plus the inflow on each incoming line
    # carrying that line's upstream composition. Restart every iteration from the
    # local gen only and read the PREVIOUS iteration's shares (Jacobi), so a bus is
    # never re-added to itself: the earlier in-place/accumulating form re-added the
    # inflows on top of the already-accumulated shares, inflating a generator's
    # downstream reach and breaking gen-side conservation (its chords summed to more
    # than its dispatch) — which surfaces once a bus hosts BOTH gen and load, or
    # flows form a loop.
    base = {b: ({b: net_gen[b]} if net_gen.get(b, 0.0) > 0.1 else {}) for b in buses}
    for _iteration in range(60):
        changed = False
        next_gab = {}
        for b in buses:
            new_shares = dict(base[b])  # local surplus gen only; inflows added below

            for from_bus, li, abs_flow in incoming[b]:
                from_shares = gen_at_bus.get(from_bus, {})   # previous iteration
                from_total = sum(from_shares.values())
                if from_total < 0.01:
                    continue
                # This line carries abs_flow MW from from_bus to b; split it by
                # from_bus's gen composition.
                for g_bus, g_mw in from_shares.items():
                    new_shares[g_bus] = new_shares.get(g_bus, 0.0) + (g_mw / from_total) * abs_flow

            for g in set(new_shares) | set(gen_at_bus.get(b, {})):
                if abs(gen_at_bus.get(b, {}).get(g, 0.0) - new_shares.get(g, 0.0)) > 0.05:
                    changed = True
            next_gab[b] = new_shares

        gen_at_bus = next_gab
        if not changed:
            break

    # Extract gen-to-load allocations from gen_at_bus. Only the NET load
    # (load - locally served) is imported from the network; the locally served
    # part is emitted as a same-bus self-loop below.
    result = []
    for load_bus, load_mw in demand_by_bus.items():
        net_load = load_mw - local_service.get(load_bus, 0.0)
        if net_load < 0.1:
            continue
        shares = gen_at_bus.get(load_bus, {})
        total = sum(shares.values())
        if total < 0.1:
            continue

        for g_bus, g_share in shares.items():
            alloc = net_load * (g_share / total)
            if alloc > 0.5:
                result.append((g_bus, load_bus, round(alloc, 1)))

    # Copperplate self-loops: a bus's own gen serving its own co-located load.
    for b, lc in local_service.items():
        if lc > 0.5:
            result.append((b, b, round(lc, 1)))

    return result


def prepare_bus_data(
    orders_hour: pd.DataFrame, network: pypsa.Network
) -> tuple[dict[Bus, list[dict[str, Any]]], Loads, pd.DataFrame, list[Bus]]:
    """
    Organize hourly market orders into per-bus supply and demand data.

    Parameters
    ----------
    orders_hour : DataFrame
        Market orders for a single hour (from market_orders.csv).
    network : pypsa.Network
        PyPSA network with buses and lines.

    Returns
    -------
    supply_by_bus : dict
        {bus: [{'unit_id', 'price', 'volume', 'accepted_volume'}, ...]} sorted by price
    demand_by_bus : dict
        {bus: total_demand_MW}
    lines_df : DataFrame
        Line data with bus0, bus1, s_nom columns
    all_buses : list
        Sorted bus names
    """
    all_buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))
    lines_df = network.lines[['bus0', 'bus1', 's_nom']]

    supply_by_bus = {}
    for bus in all_buses:
        gens = orders_hour[
            (orders_hour['volume'] > 0) & (orders_hour['node'] == int(bus))
        ].sort_values('price')
        if len(gens) > 0:
            supply_by_bus[bus] = gens[
                ['unit_id', 'price', 'volume', 'accepted_volume']
            ].to_dict('records')

    demand_by_bus = {}
    for bus in all_buses:
        loads = orders_hour[
            (orders_hour['volume'] < 0) & (orders_hour['node'] == int(bus))
        ]
        if len(loads) > 0:
            demand_by_bus[bus] = abs(loads['volume'].sum())

    return supply_by_bus, demand_by_bus, lines_df, all_buses


def plot_nodal_circlize(
    supply_by_bus: dict[Bus, list[dict[str, Any]]],
    demand_by_bus: Loads,
    all_buses: list[Bus],
    flows: Sequence[tuple[Bus, Bus, float] | tuple[Bus, Bus, float, str]] | None = None,
    clearing_price: float | None = None,
    bus_lmps: BusValues | None = None,
    title: str | None = None,
    min_sector: float = 40,
    bus_colors: dict[Bus, str] | None = None,
    figsize: tuple[float, float] | None = None,
    price_cap: float = 200,
    gen_marginal_costs: dict[Resource, float] | None = None,
    ax: PolarAxes | None = None,
    label_fontsize: float = 9,
    compact: bool = False,
    show_legend: bool = True,
    sector_order: Iterable[Bus] | None = None,
    start: float = 0,
    center_bus: Bus | None = None,
    dim_buses: Collection[Bus] | None = None,
    track_fontsize: float | None = None,
    lmp_line: bool = False,
    bus_groups: dict[Bus, str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_label_fontsize: float | None = None,
    show_group_labels: bool = True,
    annotate_roles: bool = False,
    axis_key: bool = False,
    demand_segments: dict[Bus, list[dict[str, Any]]] | None = None,
    gen_bid_labels: bool = True,
    gen_cost_labels: bool = False,
    block_mw_unit: bool = True,
    show_bus_price: bool = True,
) -> Figure:
    """
    Create a circlize chord diagram showing nodal generation, demand, and flows.

    Each bus gets a unique color. Gen and load at the same bus share that color.
    Merit order staircase shows multiple generators as steps within the gen side.
    Load bar height = bus clearing price (LMP). Shadow price bar shown outside.

    Parameters
    ----------
    supply_by_bus : dict
        {bus: [{'unit_id', 'price', 'volume', 'accepted_volume'}, ...]}
    demand_by_bus : dict
        {bus: total_demand_MW}
    all_buses : list
        Ordered bus names (strings)
    flows : list of tuples, optional
        [(src_bus, dst_bus, flow_MW), ...]. Auto-computed if None.
    clearing_price : float, optional
        System clearing price for annotation (used as fallback for all buses).
    bus_lmps : dict, optional
        {bus: lmp_price}. Per-bus locational marginal prices.
        Falls back to clearing_price for buses not in dict.
    title : str, optional
        Plot title.
    min_sector : float
        Minimum sector width in MW.
    bus_colors : dict, optional
        {bus: color_hex}. Auto-assigned if None.
    figsize : tuple, optional
        Figure size.
    price_cap : float
        Visual cap for price normalization. Prices above this are drawn at cap
        height with a label showing the actual value (avoids squishing everything
        when a bus has an extreme shadow price like $10,000).
    gen_marginal_costs : dict, optional
        {unit_id: marginal_cost}. If provided, draws a dotted line across each
        generator's bar at the marginal cost height for comparison with bid price.
    lmp_line : bool
        If True, draw a dashed line across each bus's dispatched generation at
        the bus LMP height. With the solid bar height = the unit's marginal cost
        (its bid), the gap between the bar top and this line is the visual
        inframarginal rent (LMP − marginal cost) the unit earns; where the LMP
        falls below a unit's cost the unit is not dispatched. One channel shows
        marginal cost (the fill), the other the cleared LMP (the dashed line).
    bus_groups : dict, optional
        {bus: group_label}. When given (with more than one distinct group), an
        outer band is drawn around each sector in its group's color and each
        contiguous run of a group is labeled (e.g. "BA-1", "BA-2") — so a
        multi-BA / multi-market ring shows which operator owns which quadrant.
    group_colors : dict, optional
        {group_label: color_hex}. Color for each group's outer band/label.
        Auto-filled from the bus palette if omitted.
    group_label_fontsize : float, optional
        Font size for the outer group labels. Defaults to ``label_fontsize``.
    show_group_labels : bool
        If True (default) draw the on-ring group labels ("BA-1", …). Set False
        when a figure legend already identifies the bands, to keep the labels
        from colliding with the bus / LMP text outside the ring.
    annotate_roles : bool
        If True, draw a one-time set of read-the-chart callouts near 12 o'clock:
        "supply" on a dispatched generator's inner send-bar, "demand" on a load's
        inner receive-bar, and "dispatch" on a flow chord — each on the element
        closest to the top.
    axis_key : bool
        If True, draw a small illustrative axis on one bar that follows the polar
        layout — a radial arrow up the bar's left edge labeled "price" and an arc
        bent along the track base labeled "volume" — showing that a bar's radial
        height encodes price and its tangential width encodes volume. Direction
        only (not a measured scale). Anchored to the first bar clockwise from the
        top (skipping the demand-labeled bar), chosen dynamically so it does not
        depend on any particular bus name/number. A legend-on-the-figure so the inner ring and chords
        explain themselves once.
    show_bus_price : bool
        Print the price under each outer ring label ("Bus N / $X/MWh"). Set
        False for a CONFIGURATION view, where the demand-bar height is a chosen
        reference — the dearest unit's offer, say — rather than a cleared LMP:
        the bars still take their height from ``bus_lmps``, but the ring reads
        plain "Bus N" so no number is presented as a price it is not.

    An outer bus label that another text reaches moves outward when the figure
    draws, and a ring with no collision draws as placed (``_clear_ring_labels``,
    2026-11-07). Before it, a bid label, an in-bar megawatt label, the axis
    key's labels and the role labels that a drawn ring line, the key's arrows
    or another text reaches move to the nearest clear placement
    (``_clear_chord_labels``, 2026-11-07). Display only: no number changes.

    Returns
    -------
    fig : matplotlib Figure
    """
    # Colors are assigned against the ORIGINAL bus order so they stay tied to
    # bus identity; the sector placement order is then set independently below.
    if bus_colors is None:
        bus_colors = assign_bus_colors(all_buses, supply_by_bus, demand_by_bus)

    # sector_order controls the clockwise arrangement of sectors (pycirclize
    # places dict order clockwise from 12 o'clock). Default keeps all_buses order.
    if sector_order is not None:
        ordered = [b for b in sector_order if b in all_buses]
        ordered += [b for b in all_buses if b not in ordered]  # any omitted buses
        all_buses = ordered

    if flows is None:
        flows = compute_proportional_flows(supply_by_bus, demand_by_bus)

    # Build per-bus LMP lookup (fallback to system clearing price)
    _lmps = {}
    for bus in all_buses:
        if bus_lmps and bus in bus_lmps:
            _lmps[bus] = bus_lmps[bus]
        elif clearing_price is not None:
            _lmps[bus] = clearing_price
        else:
            _lmps[bus] = 0

    # Price normalization: auto-scale to max bid price in the data,
    # capped at price_cap. This makes the staircase fill the radial space.
    max_bid = max(
        (g['price'] for gens in supply_by_bus.values() for g in gens),
        default=0,
    )
    max_lmp = max(_lmps.values()) if _lmps else 0
    # also cover any opt-in dotted lines (e.g. the bilateral cleared-price staircase), so
    # they fit the radial range instead of clamping at the top.
    max_line = max((gen_marginal_costs or {}).values(), default=0)
    max_seg = max((float(s.get('price', 0)) for segs in (demand_segments or {}).values()
                   for s in segs), default=0)
    norm_price = max(min(max(max_bid, max_lmp, max_line, max_seg) * 1.1, price_cap), 1)

    # --- Sector sizing (all buses, transit buses get small fixed size) ---
    gap = 5  # MW gap between gen and load within a sector
    transit_sector = 15  # small fixed width for empty transit buses

    bus_gen_width = {}
    bus_load_width = {}
    bus_sizes = {}
    for bus in all_buses:
        # Use nameplate capacity if available, otherwise bid volume
        gw = sum(g.get('capacity', g['volume']) for g in supply_by_bus.get(bus, []))
        lw = demand_by_bus.get(bus, 0)
        bus_gen_width[bus] = gw
        bus_load_width[bus] = lw
        content = gw + lw + (gap if gw > 0 and lw > 0 else 0)
        if content < 1:
            # a dimmed out-of-area bus carries a "$XX/MWh" price label, so give it a
            # touch more angular room than a bare transit bus so the labels don't collide
            bus_sizes[bus] = (30 if (dim_buses is not None and bus in dim_buses)
                              else transit_sector)
        else:
            bus_sizes[bus] = max(content, min_sector)

    # --- Build Circos ---
    # `start` rotates the whole ring (degrees clockwise from 12 o'clock). If
    # `center_bus` is given, compute the start offset that lands that bus's
    # sector center exactly at 12 o'clock.
    sector_dict = {f"Bus {b}": bus_sizes[b] for b in all_buses}
    _space = 5
    start_deg = start
    if center_bus is not None and center_bus in all_buses:
        sizes = [bus_sizes[b] for b in all_buses]
        total = sum(sizes) or 1
        avail = 360 - len(sizes) * _space
        widths = [s / total * avail for s in sizes]
        k = all_buses.index(center_bus)
        before = sum(widths[i] + _space for i in range(k))
        start_deg = -(before + widths[k] / 2)
    circos = Circos(sector_dict, space=_space,
                    start=start_deg, end=start_deg + 360)

    bus_gen_range: dict[Bus, tuple[float, float]] = {}  # (x_start, x_end) of gen region
    bus_load_range = {}  # (x_start, x_end) of load region
    # A LANE is a block on the gen arc / a segment on the load arc that a
    # self-schedule owns (``lane`` key). Its range is reserved for the schedule's
    # own chord, keyed (x0, x1, color) per bus.
    gen_lane, load_lane = {}, {}
    # The outer bus labels as (text, angle in radians, radius), for the
    # collision pass after rendering.
    _ring_labels: list[tuple[str, float, float]] = []
    # The texts the chord pass may move, and the ring lines they keep clear of
    # as (theta, r, lw), for the collision pass after rendering
    # (_clear_chord_labels, 2026-11-07).
    _chord_labels: list[_ChordLabel] = []
    _role_labels: list[_ChordLabel] = []
    _ring_lines: list[tuple[np.ndarray, np.ndarray, float]] = []

    def _line_at(sec: Any, x0: float, x1: float, r0: float, lw: float,
                 r1: float | None = None) -> None:
        """Record a drawn ring line from sector coordinate x0 to x1."""
        if x1 - x0 > 0.01:
            _ring_lines.append((*_ring_arc(sec.x_to_rad(x0), sec.x_to_rad(x1),
                                           r0, r1), lw))

    def _sides_at(sec: Any, xs: Iterable[float], r0: float, r1: float,
                  lw: float) -> None:
        """Record the radial sides of a drawn bar at sector coordinates xs."""
        _ring_lines.extend((*_ring_arc(sec.x_to_rad(x), sec.x_to_rad(x), r0, r1), lw)
                           for x in xs)

    def _in_block(text: str, sec: Any, x: float, r: float, x0: float, x1: float,
                  r_lo: float, r_hi: float) -> None:
        """Record an in-bar label that keeps to its block ``[x0, x1] x [r_lo, r_hi]``.

        Its candidates slide it along the ring and the radius, then turn it to
        run along the radius; its block's two sides are lines it clears. A
        block too small for its label in either direction leaves it the
        space next above the block's top, within the block's angles, and
        last the radii above the block's centre out to the outer band.
        """
        th, th0, th1 = sec.x_to_rad(x), sec.x_to_rad(x0), sec.x_to_rad(x1)
        th_mid = sec.x_to_rad((x0 + x1) / 2)
        _chord_labels.append(_ChordLabel(
            text, th, r,
            _block_candidates(th, r, th0, th1, r_lo, r_hi,
                              ("horizontal", "vertical"))
            + [(th_mid, r_hi + 1.0, "horizontal")]
            + _block_candidates(th_mid, r_hi + 1.0, th0, th1,
                                r_hi + 0.5, 96.0, ("horizontal",))
            + [(th_mid, r_hi + 0.5 * k, "horizontal")
               for k in range(2, int(2 * (96.0 - r_hi)) + 1)],
            [_ring_arc(th0, th0, r_lo, r_hi), _ring_arc(th1, th1, r_lo, r_hi)]))

    # In-track label sizes. track_fontsize, when given, overrides the size of the
    # bid, dispatched-MW and load-MW labels drawn inside the sectors (e.g. to hold
    # a >=10 pt floor for a page-width standalone figure). Otherwise the compact
    # flag drives them (smaller still in the dense side-by-side composite).
    _fs_gen = track_fontsize if track_fontsize is not None else (7 if compact else 5)
    _fs_load = track_fontsize if track_fontsize is not None else (7 if compact else 6)

    # Group (BA / market) banding setup. Only active when bus_groups names more
    # than one distinct group; otherwise the ring is single-operator and the
    # outer band is skipped (keeping the default single-market look unchanged).
    _draw_groups = bool(bus_groups) and len(set(bus_groups.values())) > 1  # type: ignore[union-attr]  # bool() guards None, mypy does not narrow
    _group_colors = dict(group_colors or {})
    if _draw_groups:
        _gpal = 0
        for g in dict.fromkeys(bus_groups.values()):  # type: ignore[union-attr]  # stable first-seen order
            if g not in _group_colors:
                _group_colors[g] = BUS_PALETTE[_gpal % len(BUS_PALETTE)]
                _gpal += 1

    for bus in all_buses:
        sector = circos.get_sector(f"Bus {bus}")
        gens = supply_by_bus.get(bus, [])
        dem = demand_by_bus.get(bus, 0)
        size = bus_sizes[bus]
        gw = bus_gen_width[bus]
        lw = bus_load_width[bus]
        bc = bus_colors[bus]
        bc_dark = _darken(bc, 0.6)

        # --- Outer track: merit order + demand ---
        # Main track for gen staircase and load bars (thick band)
        track = sector.add_track((55, 96))
        # Shadow price track (thin, just outside main)
        shadow_track = sector.add_track((97, 100))

        bus_lmp = _lmps.get(bus, 0)

        # Helper: normalize price to r_lim within main track [55, 96]
        # Negative prices get a minimal bar height with a label showing the value
        _MIN_BAR_R = 58  # minimum bar top for negative bids (thin visible bar)

        def _price_to_r(price: float) -> float:
            capped = min(price, norm_price)
            if capped <= 0:
                return _MIN_BAR_R
            return 55 + (capped / norm_price) * 41

        # SUPPLY side (left): merit order staircase in bus color
        # Three fill levels: nameplate capacity (very faded), bid volume (faded),
        # accepted dispatch (solid). Width always = nameplate capacity.
        x_pos = 0
        for i, gen in enumerate(gens):
            nameplate = gen.get('capacity', gen['volume'])
            bid_vol = gen['volume']
            price = gen['price']
            accepted = min(gen['accepted_volume'], bid_vol)
            # Opt-in per-unit color override (e.g. a gray self-schedule block
            # inside an otherwise bus-colored stack); default = bus color.
            gcol = gen.get('color', bc)
            gcol_dark = _darken(gcol) if 'color' in gen else bc_dark

            # Clamp coordinates to sector bounds to avoid pycirclize ValueError
            x_end_nameplate = min(x_pos + nameplate, size)
            x_end_bid = min(x_pos + bid_vol, size)
            x_end_acc = min(x_pos + accepted, size)
            if gen.get('lane'):
                gen_lane[bus] = (x_pos, x_end_nameplate, gcol)

            # A unit carrying a linear marginal-cost curve (``mc0`` + ``mc_slope``·g)
            # is drawn as a sloped WEDGE under its MC line rather than a flat bar:
            # the bar top rises with output, so the rising offer is visible and the
            # gap to the dashed LMP line is the per-MW inframarginal rent that varies
            # along the curve. A flat unit keeps the original three-level rectangle.
            has_curve = 'mc_slope' in gen
            # A unit entry may carry ``hatch``; it applies to the accepted fill
            # on both the flat and the curved path (an incremental redispatch
            # above the dispatch a day-ahead award assumed is drawn this way).
            _hk = {'hatch': gen['hatch']} if gen.get('hatch') else {}
            if has_curve:
                _mc0, _slope = gen['mc0'], gen['mc_slope']
                bar_top = _price_to_r(min(_mc0 + _slope * nameplate, norm_price))

                def _wedge(xa: float, xb: float, **kw: str | float) -> None:
                    if xb <= xa + 0.01:
                        return
                    xs = np.linspace(xa, xb, 16)
                    vs = np.clip(_mc0 + _slope * (xs - x_pos), 0.5, norm_price)
                    track.fill_between(xs.tolist(), vs.tolist(), 0,
                                       vmin=0, vmax=norm_price, **kw)

                _wedge(x_pos, x_end_nameplate, fc=gcol, alpha=0.08, ec=gcol, lw=0.5, ls='--')
                _wedge(x_pos, x_end_bid, fc=gcol, alpha=0.25, ec=gcol, lw=0.5)
                _wedge(x_pos, x_end_acc, fc=gcol, alpha=0.7, ec=gcol_dark, lw=0.8, **_hk)
                if x_end_nameplate > x_pos + 0.01:
                    _xs = np.linspace(x_pos, x_end_nameplate, 16)
                    _ring_lines.append((
                        np.array([sector.x_to_rad(v) for v in _xs]),
                        np.array([_price_to_r(min(_mc0 + _slope * (v - x_pos), norm_price))
                                  for v in _xs]),
                        0.8))
            else:
                bar_top = _price_to_r(price)

                # Nameplate capacity (very faded — withheld capacity)
                if x_end_nameplate > x_pos + 0.01:
                    track.rect(x_pos, x_end_nameplate, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.08, ec=gcol, lw=0.5,
                               ls='--', **_hk)

                # Bid volume (faded — offered but not accepted)
                if bid_vol > 0 and x_end_bid > x_pos + 0.01:
                    track.rect(x_pos, x_end_bid, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.25, ec=gcol, lw=0.5, **_hk)

                # Accepted dispatch (solid bus color)
                if accepted > 0 and x_end_acc > x_pos + 0.01:
                    track.rect(x_pos, x_end_acc, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.7, ec=gcol_dark, lw=0.8,
                               **_hk)
                _line_at(sector, x_pos, x_end_nameplate, bar_top, 0.8)
                _sides_at(sector, {x for x, drawn in (
                    (x_pos, x_end_nameplate > x_pos + 0.01),
                    (x_end_nameplate, x_end_nameplate > x_pos + 0.01),
                    (x_end_bid, bid_vol > 0 and x_end_bid > x_pos + 0.01),
                    (x_end_acc, accepted > 0 and x_end_acc > x_pos + 0.01))
                    if drawn}, 55, bar_top, 0.8)
            _line_at(sector, x_pos, x_end_nameplate, 55, 0.8)

            # Label: the BID (price × volume) on each gen bar, mirroring the
            # load side which prints its MW in-track. (Unit name dropped.) Opt out
            # via gen_bid_labels=False when the bar height is a cost the caller would
            # rather not annotate (e.g. the bilateral book, which labels the cleared
            # price on the dotted line instead -- see gen_cost_labels below).
            # A dense offer staircase (a storage curve entered as many thin
            # blocks) cannot carry per-block labels legibly: skip the text on
            # any block narrower than 15% of its sector.
            # An entry may also opt out on its own through ``price_label: False``.
            # A price-taking block (a self-schedule) carries no offer, so its bar
            # height is the bus LMP rather than a bid and a "$p x q MW" label would
            # read as an offer that was never made; the accepted-MW label inside the
            # bar carries its volume.
            # ``force_label`` overrides the density guard for a block that is
            # the subject of the frame (a self-schedule beside a large unit's
            # nameplate is narrow, and dropping its MW would hide the quantity
            # the frame is about).
            _dense_blk = (nameplate < 0.15 * max(size, 1e-9)
                          and not gen.get('force_label'))
            if (gen_bid_labels and not has_curve and not _dense_blk
                    and gen.get('price_label', True)):
                mid_x = x_pos + nameplate / 2
                if price < 0:
                    price_str = f"-${abs(price):.0f}"
                elif price > norm_price:
                    price_str = f"${price:,.0f}"
                else:
                    price_str = f"${price:.0f}"
                track.text(f"{price_str} × {bid_vol:.0f} MW",
                           x=mid_x, r=bar_top + 5,
                           fontsize=_fs_gen, color=gcol_dark,
                           fontweight='bold')
                # A bid label that a ring line or another text reaches steps
                # along the radius, nearest first, between the bar base and
                # the outer band, and last outside the band (a label wider
                # than a narrow sector fits nowhere inside it).
                _th = sector.x_to_rad(mid_x)
                _chord_labels.append(_ChordLabel(
                    f"{price_str} × {bid_vol:.0f} MW", _th, bar_top + 5,
                    [(_th, bar_top + 5 + d, None)
                     for d in sorted((0.25 * k for k in range(-160, 161) if k),
                                     key=lambda v: (abs(v), -v))
                     if 56.0 <= bar_top + 5 + d <= 96.0]
                    + [(_th, 100.5 + 0.25 * k, None) for k in range(47)],
                    []))

            # Dispatched MW printed INSIDE the gen track (mirrors the load side,
            # which prints its MW in-track), while the bid stays outside the bar.
            # Radius = midpoint of the FILLED bar (55 -> bar_top) so the label
            # always sits inside the bar regardless of its height; a fixed floor
            # would push a short bar's label out on top of the track.
            if accepted > 0.5 and not _dense_blk:
                # For a wedge, place the MW label inside the bar at the ACCEPTED
                # edge's height (the peak ``bar_top`` is the nameplate edge, well above).
                if has_curve:
                    _r_acc = _price_to_r(min(_mc0 + _slope * accepted / 2, norm_price))
                    _lbl_r = (55 + _r_acc) / 2
                else:
                    _lbl_r = (55 + bar_top) / 2
                track.text(f"{accepted:.0f}{' MW' if block_mw_unit else ''}",
                           x=x_pos + accepted / 2, r=_lbl_r,
                           fontsize=_fs_gen, color=gcol_dark,
                           fontweight='bold')
                # the label keeps to the dispatched part of the block it reads
                _in_block(f"{accepted:.0f}{' MW' if block_mw_unit else ''}",
                          sector, x_pos + accepted / 2, _lbl_r,
                          x_pos, x_end_acc, 55,
                          _r_acc if has_curve else bar_top)

            # Staircase step line between generators
            if i > 0:
                track.line([x_pos, x_pos], [55, bar_top],
                           color=bc_dark, lw=0.8, ls='--')

            x_pos += nameplate

        bus_gen_range[bus] = (0, x_pos)

        # Marginal cost dotted lines (drawn after bars so they appear on top)
        # Use raw price as y-value with vmin/vmax matching the price normalization,
        # so the line maps to the correct radial position within the track.
        if gen_marginal_costs:
            mc_x = 0
            for gen in gens:
                nameplate = gen.get('capacity', gen['volume'])
                uid = gen['unit_id']
                if uid in gen_marginal_costs:
                    mc_val = gen_marginal_costs[uid]
                    mc_draw = min(mc_val, norm_price)   # clamp to the radial range (label keeps mc_val)
                    mc_x_end = min(mc_x + nameplate, size)
                    if mc_val > 0 and mc_x_end > mc_x + 1:
                        track.line(
                            [mc_x + 0.5, mc_x_end - 0.5],
                            [mc_draw, mc_draw],
                            vmin=0, vmax=norm_price,
                            color=bc_dark, lw=1.5, ls=':', zorder=10,
                        )
                        _line_at(sector, mc_x + 0.5, mc_x_end - 0.5,
                                 _price_to_r(mc_draw), 1.5)
                        # Opt-in: print the line's $value above each segment (used by the
                        # bilateral book to show the cleared price of each staircase step).
                        if gen_cost_labels:
                            track.text(f"${mc_val:.0f}",
                                       x=(mc_x + mc_x_end) / 2,
                                       r=min(99, _price_to_r(mc_val) + 4),
                                       fontsize=_fs_gen, color=bc_dark,
                                       fontweight='bold', zorder=11)
                mc_x += nameplate

        # LMP dashed line at the cleared bus LMP. Each solid gen bar top is that
        # unit's marginal cost (its bid); the gap to this line is the dispatched
        # unit's inframarginal rent (LMP − marginal cost), and a bar rising ABOVE
        # the line is out of merit (undispatched). Drawn over the generation
        # region where the bus has gen; for a TRANSIT bus (no gen, no load) it is
        # drawn across the whole empty sector so its LMP is still visible. A
        # load-only bus is skipped — its load bar height already IS the LMP.
        if lmp_line and bus_lmp > 0:
            if gw > 0:
                lmp_x0, lmp_x1 = 0.5, min(gw, size) - 0.5
            elif dem <= 0:
                lmp_x0, lmp_x1 = 0.5, size - 0.5      # transit: span the sector
            else:
                lmp_x0 = lmp_x1 = 0.0                 # load-only: bar = LMP already
            if lmp_x1 - lmp_x0 > 1.0:
                track.line(
                    [lmp_x0, lmp_x1],
                    [min(bus_lmp, norm_price), min(bus_lmp, norm_price)],
                    vmin=0, vmax=norm_price,
                    color='#17202A', lw=1.6, ls=(0, (4, 2)), zorder=12,
                )
                _line_at(sector, lmp_x0, lmp_x1, _price_to_r(bus_lmp), 1.6)

        # LOAD side (right): demand bar, height = bus LMP. An opt-in
        # demand_segments entry splits the SAME total into consecutive
        # segments, each with its own height/color (e.g. the portion of a
        # load served by a self-schedule at the other market's price).
        if dem > 0:
            load_start = gw + (gap if gw > 0 else 0)
            load_end = min(load_start + dem, size)
            segs = (demand_segments or {}).get(bus)

            if segs:
                x0 = load_start
                for seg in segs:
                    mw = float(seg['mw'])
                    if mw <= 0.01:
                        continue
                    x1 = min(x0 + mw, load_end)
                    seg_top = _price_to_r(float(seg.get('price', bus_lmp)))
                    scol = seg.get('color', bc)
                    scol_dark = _darken(scol) if 'color' in seg else bc_dark
                    if seg.get('lane'):
                        load_lane[bus] = (x0, x1, scol)
                    # Per-segment alpha (default = the demand fill 0.35). A fainter
                    # alpha draws e.g. shed/unserved load in the bus color, the
                    # way idle generation capacity is drawn faint on the supply side.
                    track.rect(x0, x1, r_lim=(55, seg_top),
                               fc=scol, alpha=float(seg.get('alpha', 0.35)),
                               ec=scol_dark, lw=0.8)
                    _line_at(sector, x0, x1, seg_top, 0.8)
                    _line_at(sector, x0, x1, 55, 0.8)
                    _sides_at(sector, (x0, x1), 55, seg_top, 0.8)
                    if mw > 0.5:
                        track.text(f"{mw:.0f}{' MW' if block_mw_unit else ''}", x=(x0 + x1) / 2,
                                   r=(55 + seg_top) / 2,
                                   fontsize=_fs_load, color=scol_dark,
                                   fontweight='bold')
                        _in_block(f"{mw:.0f}{' MW' if block_mw_unit else ''}",
                                  sector, (x0 + x1) / 2, (55 + seg_top) / 2,
                                  x0, x1, 55, seg_top)
                    x0 = x1
                # A split load bar no longer encodes the bus LMP by its
                # height, so draw the dashed LMP line across the load region
                # (mirrors the gen-region line) to keep the price visible.
                if lmp_line and bus_lmp > 0 and load_end - load_start > 1.0:
                    track.line([load_start + 0.5, load_end - 0.5],
                               [min(bus_lmp, norm_price)] * 2,
                               vmin=0, vmax=norm_price,
                               color='#17202A', lw=1.6, ls=(0, (4, 2)), zorder=12)
                    _line_at(sector, load_start + 0.5, load_end - 0.5,
                             _price_to_r(bus_lmp), 1.6)
            else:
                load_bar_top = _price_to_r(bus_lmp)

                # Demand bar at LMP height
                track.rect(load_start, load_end, r_lim=(55, load_bar_top),
                           fc=bc, alpha=0.35, ec=bc_dark, lw=0.8)

                mid_load = load_start + dem / 2
                # Radius = midpoint of the filled load bar (55 -> load_bar_top), so the
                # MW label stays inside the bar even when the LMP (bar height) is low.
                track.text(f"{dem:.0f}{' MW' if block_mw_unit else ''}", x=mid_load, r=(55 + load_bar_top) / 2,
                           fontsize=_fs_load, color=bc_dark,
                           fontweight='bold')
                _line_at(sector, load_start, load_end, load_bar_top, 0.8)
                _line_at(sector, load_start, load_end, 55, 0.8)
                _sides_at(sector, (load_start, load_end), 55, load_bar_top, 0.8)
                _in_block(f"{dem:.0f}{' MW' if block_mw_unit else ''}",
                          sector, mid_load, (55 + load_bar_top) / 2,
                          load_start, load_end, 55, load_bar_top)
                # Price label on load bar (suppressed in compact mode and whenever a
                # track_fontsize floor is set — it is a tiny, duplicate of the bus LMP
                # already shown in the outer ring label and the shared key).
                if not compact and track_fontsize is None and show_bus_price:
                    lmp_label = f"${bus_lmp:.1f}"
                    if bus_lmp > norm_price:
                        lmp_label = f"${bus_lmp:,.1f}"
                    track.text(lmp_label, x=mid_load, r=58,
                               fontsize=4, color=bc_dark)

            bus_load_range[bus] = (load_start, load_end)
        else:
            bus_load_range[bus] = (0, 0)

        # --- Outer band: shadow-price ghost OR group (BA/market) band ---
        # When buses are grouped by operator, the (97,100) ring carries the
        # group's color band instead of the LMP ghost (the LMP is already the
        # load-bar height and, with lmp_line, a dashed line on the gen bars).
        if _draw_groups and bus in bus_groups:  # type: ignore[operator]  # guarded by _draw_groups
            gcol = _group_colors.get(bus_groups[bus], TRANSIT_COLOR)  # type: ignore[index]  # guarded by _draw_groups
            shadow_track.rect(0, size, r_lim=(97, 100),
                              fc=gcol, alpha=0.55, ec=_darken(gcol), lw=0.5)
            _line_at(sector, 0, size, 97, 0.5)
            _line_at(sector, 0, size, 100, 0.5)
        elif bus_lmp > 0 and (dem > 0 or gw > 0):
            # Shadow price bar (ghost bar outside main track): bus LMP outline.
            shadow_r = 97 + min(bus_lmp / norm_price, 1.0) * 3
            region_start = 0
            region_end = gw + dem + (gap if gw > 0 and dem > 0 else 0)
            if region_end > 0 and shadow_r > 97 + 1e-6:
                # a zero-height rect (LMP tiny vs the normalization) is an
                # invalid pycirclize track range -- skip the ghost bar
                shadow_track.rect(region_start, min(region_end, size),
                                  r_lim=(97, shadow_r),
                                  fc=bc, alpha=0.15, ec=bc, lw=0.3)
                _line_at(sector, region_start, min(region_end, size), 97, 0.3)
                _line_at(sector, region_start, min(region_end, size), shadow_r, 0.3)

        # --- Inner ring: a "send" sub-bar under the generator and a "receive"
        #     sub-bar under the load, in the same bus color, so a bus with
        #     co-located gen + load shows its two roles separately. The gen→load
        #     chords (incl. the same-bus self-loop) then visibly run from the
        #     send bar to the receive bar. ---
        label_track = sector.add_track((48, 53))
        gen_disp_local = sum(g['accepted_volume'] for g in gens)
        drew_inner = False
        if gw > 0:
            # "send" under the generator: idle capacity faint, dispatched solid
            if gw > gen_disp_local + 0.01:
                label_track.rect(min(gen_disp_local, gw), gw, r_lim=(48, 53),
                                 fc=bc, alpha=0.12, ec=bc, lw=0.4)
            if gen_disp_local > 0.01:
                label_track.rect(0, min(gen_disp_local, gw), r_lim=(48, 53),
                                 fc=bc, alpha=0.6, ec=bc_dark, lw=0.5)
            drew_inner = True
        ls_inner, le_inner = bus_load_range.get(bus, (0, 0))
        if dem > 0 and le_inner > ls_inner:
            # "receive" under the load: outline only
            label_track.rect(ls_inner, le_inner, r_lim=(48, 53),
                             fc='white', alpha=0.85, ec=bc, lw=1.5)
            drew_inner = True
        if not drew_inner:
            # transit — thin outline across the whole sector
            label_track.rect(0, size, r_lim=(48, 53),
                             fc='white', alpha=0.5, ec=bc, lw=0.5)

        # Bus label. A transit bus (no generation and no load) carries no price
        # worth surfacing here, so it mimics the network diagram's node — just
        # the bus number inside a circle (a parenthesized "(N)" fallback if the
        # circle bbox can't be drawn). Active buses keep "Bus N" + their LMP.
        bus_lmp = _lmps.get(bus, 0)
        is_transit = (gw <= 0) and (dem <= 0)
        _dimmed = dim_buses is not None and bus in dim_buses
        # Each outer label is recorded with its angle and radius, so the collision
        # pass after rendering (_clear_ring_labels) finds the text it drew.
        _ring_rad = sector.x_to_rad(sector.center)
        if is_transit and _dimmed and bus_lmp > 0:
            # A grayed bus OUTSIDE the modeled area: no gen/load here, but its LMP
            # is defined (lambda + shift-factor congestion) -- surface just the price
            # ($/MWh, no "Bus N" label), the out-of-area price the ring still prices.
            sector.text(f"${bus_lmp:.1f}", r=104, fontsize=label_fontsize * 0.8,
                        fontweight='bold', color=bc_dark)
            _ring_labels.append((f"${bus_lmp:.1f}", _ring_rad, 104))
        elif is_transit:
            try:
                sector.text(f"{bus}", r=103, fontsize=label_fontsize,
                            fontweight='bold', color=bc_dark,
                            bbox=dict(boxstyle='circle,pad=0.3', fc='white',
                                      ec=bc_dark, lw=1.2))
                _ring_labels.append((f"{bus}", _ring_rad, 103))
            except Exception:
                sector.text(f"({bus})", r=103, fontsize=label_fontsize,
                            fontweight='bold', color=bc_dark)
                _ring_labels.append((f"({bus})", _ring_rad, 103))
        elif bus_lmp > 0 and show_bus_price:
            sector.text(f"Bus {bus}\n${bus_lmp:.1f}/MWh", r=103,
                        fontsize=label_fontsize, fontweight='bold', color=bc_dark)
            _ring_labels.append((f"Bus {bus}\n${bus_lmp:.1f}/MWh", _ring_rad, 103))
        else:
            sector.text(f"Bus {bus}", r=103, fontsize=label_fontsize,
                        fontweight='bold', color=bc_dark)
            _ring_labels.append((f"Bus {bus}", _ring_rad, 103))

    # --- Flow chords: gen→load, anchored to gen dispatch and load demand ---
    # Source side: chord occupies fraction of gen region = alloc_MW / gen_dispatch_MW
    # Dest side: chord occupies fraction of load region = alloc_MW / load_demand_MW
    gen_cursor = {b: r[0] for b, r in bus_gen_range.items()}
    load_cursor = {b: r[0] for b, r in bus_load_range.items()}
    # A lane drawn first on its arc is reserved for the lane's own chord, so the
    # regular chords start after it. Reserve only where a lane flow is actually
    # present: a one-legged bid carries a lane block with no counter-leg, and
    # reserving there would leave its ordinary chord no room to be drawn.
    _lane_pairs = {(f[0], f[1]) for f in (flows or []) if len(f) > 3}
    for b, (x0, x1, _c) in gen_lane.items():
        if (any(s_ == b for s_, _d in _lane_pairs)
                and abs(x0 - bus_gen_range.get(b, (0, 0))[0]) < 0.01):
            gen_cursor[b] = x1
    for b, (x0, x1, _c) in load_lane.items():
        if (any(d_ == b for _s, d_ in _lane_pairs)
                and abs(x0 - bus_load_range.get(b, (0, 0))[0]) < 0.01):
            load_cursor[b] = x1

    # Accepted dispatch per bus (for chord proportional sizing)
    bus_dispatch = {}
    for bus in all_buses:
        bus_dispatch[bus] = sum(g['accepted_volume'] for g in supply_by_bus.get(bus, []))

    if flows:
        sorted_flows = sorted(flows, key=lambda f: -abs(f[2]))
        lane_cursor: dict[Bus, float] = {}     # per lane block: where the next lane chord leaves it

        for flow in sorted_flows:
            src, dst, flow_mw = flow[0], flow[1], flow[2]
            if abs(flow_mw) < 0.5:
                continue
            src_name = f"Bus {src}"
            dst_name = f"Bus {dst}"

            # A lane flow ``(src, dst, mw, lane)`` runs from the lane block at
            # its source to the lane segment at its sink, in the block's color.
            if len(flow) > 3 and src in gen_lane and dst in load_lane:
                gx0, gx1, gcol_ = gen_lane[src]
                lx0, lx1, _lc = load_lane[dst]
                # one lane block may feed several lane segments: walk a cursor
                # across the block, each chord as wide as its own MW
                lc = lane_cursor.setdefault(src, gx0)
                w = min(abs(flow_mw), gx1 - lc)
                if w >= 0.5 and lx1 - lx0 >= 0.5:
                    circos.link((src_name, lc, lc + w - 0.01),
                                (dst_name, lx0, lx1 - 0.01),
                                color=gcol_, alpha=0.5)
                    lane_cursor[src] = lc + w
                continue

            # Source: chord width = flow_mw directly (MW units match sector coords).
            # This anchors chords to the dispatched (dark) portion of the gen bar,
            # since the cursor starts at 0 and total chord width = total dispatch.
            sg_start, sg_end = bus_gen_range.get(src, (0, 0))
            gen_total = bus_dispatch.get(src, 0)
            if sg_end - sg_start < 0.5:
                continue  # skip if source has no gen region
            chord_w_src = max(flow_mw, 1) if gen_total > 0 else 1

            # Dest: width within load region proportional to alloc / total load
            dl_start, dl_end = bus_load_range.get(dst, (0, 0))
            load_range = dl_end - dl_start
            load_total = demand_by_bus.get(dst, 0)
            if load_range < 0.5:
                continue  # skip if dest has no load region
            chord_w_dst = max((flow_mw / load_total) * load_range, 1) \
                if load_total > 0 else 1

            color = bus_colors.get(src, '#888')

            src_pos = gen_cursor.get(src, sg_start)
            dst_pos = load_cursor.get(dst, dl_start)

            # Clamp to region boundaries (with small epsilon for float safety)
            eps = 0.01
            src_end = min(src_pos + chord_w_src, sg_end - eps)
            dst_end = min(dst_pos + chord_w_dst, dl_end - eps)
            if src_end - src_pos < 0.5 or dst_end - dst_pos < 0.5:
                continue

            circos.link(
                (src_name, src_pos, src_end),
                (dst_name, dst_pos, dst_end),
                color=color, alpha=0.35,
            )

            gen_cursor[src] = src_end
            load_cursor[dst] = dst_end

    # --- Group (BA / market) labels: one per contiguous run of a group ---
    # Placed just outside the bus labels at the center degree of each run so the
    # multi-operator ring reads "this arc is BA-1, that arc is BA-2". Font held at
    # label_fontsize (group_label_fontsize to override) so it stays readable.
    if _draw_groups and show_group_labels:
        _glf = group_label_fontsize if group_label_fontsize is not None else label_fontsize
        runs: list[tuple[str, list[Bus]]] = []  # (group, [buses in this contiguous run])
        for b in all_buses:
            g = bus_groups.get(b)  # type: ignore[union-attr]  # guarded by _draw_groups
            if g is None:
                continue
            if runs and runs[-1][0] == g:
                runs[-1][1].append(b)
            else:
                runs.append((g, [b]))
        for g, run_buses in runs:
            mid_bus = run_buses[len(run_buses) // 2]
            sec = circos.get_sector(f"Bus {mid_bus}")
            d0, d1 = sec.deg_lim
            deg_mid = (d0 + d1) / 2
            gcol = _group_colors.get(g, TRANSIT_COLOR)
            circos.text(g, r=109, deg=deg_mid, adjust_rotation=False,
                        fontsize=_glf, fontweight='bold', color=_darken(gcol),
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  ec=gcol, lw=1.2, alpha=0.95))

    # --- Read-the-chart labels: subtle "supply" / "demand" / "dispatch" near 12 --
    # Styled like the in-bar MW numbers (small, bold, the bus's own dark color,
    # no box) and tucked INTO the relevant element: "supply" on the inner
    # send-bar, "demand" on the inner receive-bar, "dispatch" on a chord — each on
    # the element closest to the top, so the diagram explains itself once.
    if annotate_roles:
        import math as _math

        def _ang_dist(d: float) -> float:
            d %= 360
            return min(d, 360 - d)

        _centers = {}
        for _b in all_buses:
            _d0, _d1 = circos.get_sector(f"Bus {_b}").deg_lim
            _centers[_b] = ((_d0 + _d1) / 2) % 360

        _hfs = max(4.5, _fs_load - 1)   # a touch smaller than the MW labels

        def _hlabel(text: str, bus: Bus, xc: float, r: float, color: str,
                    x0: float, x1: float) -> None:
            sec = circos.get_sector(f"Bus {bus}")
            deg = _math.degrees(sec.x_to_rad(xc)) % 360
            circos.text(text, r=r, deg=deg, adjust_rotation=False,
                        fontsize=_hfs, fontweight='bold', color=color)
            # A role label that collides slides along its inner bar [x0, x1],
            # then turns to lie along the bar (a level label beside 3 or 9
            # o'clock is wider than the bar is deep).
            _th = _math.radians(deg)
            _role_labels.append(_ChordLabel(
                text, _th, r,
                _block_candidates(sec.x_to_rad(xc), r, sec.x_to_rad(x0),
                                  sec.x_to_rad(x1), 48.0, 53.0,
                                  (None, "horizontal")),
                []))

        # supply: dispatched generator nearest the top — on its inner send-bar
        _gen = [b for b in all_buses
                if bus_dispatch.get(b, 0) > 0.5 and bus_gen_range.get(b, (0, 0))[1] > 0.5]
        if _gen:
            gb = min(_gen, key=lambda b: _ang_dist(_centers[b]))
            _hlabel('supply', gb,
                    min(bus_dispatch.get(gb, 0), bus_gen_range[gb][1]) / 2,
                    50.5, _darken(bus_colors[gb], 0.55),
                    0.0, min(bus_dispatch.get(gb, 0), bus_gen_range[gb][1]))
        # demand: load nearest the top — on its inner receive-bar
        _load = [b for b in all_buses if demand_by_bus.get(b, 0) > 0.5
                 and bus_load_range.get(b, (0, 0))[1] > bus_load_range.get(b, (0, 0))[0]]
        if _load:
            lb = min(_load, key=lambda b: _ang_dist(_centers[b]))
            ls_, le_ = bus_load_range[lb]
            _hlabel('demand', lb, (ls_ + le_) / 2, 50.5,
                    _darken(bus_colors[lb], 0.55), ls_, le_)
        # dispatch: a flow chord near the top — small neutral label in the center
        if flows and _load:
            _sec = circos.get_sector(f"Bus {lb}")
            _deg = _math.degrees(_sec.x_to_rad((ls_ + le_) / 2)) % 360
            circos.text('dispatch', r=26, deg=_deg, adjust_rotation=False,
                        fontsize=_hfs, fontweight='bold', color='#5D6D7E')
            _th = _math.radians(_deg)
            _role_labels.append(_ChordLabel(
                'dispatch', _th, 26,
                _block_candidates(_th, 26, _th - 0.3, _th + 0.3, 18.0, 34.0), []))

    # --- Render ---
    # Draw into a caller-supplied PolarAxes (for side-by-side composite figures)
    # or create a standalone figure.
    if ax is not None:
        circos.plotfig(ax=ax)
        fig: Figure = ax.get_figure()  # type: ignore[assignment]  # stub adds None
    else:
        fig = circos.plotfig(figsize=figsize or (10, 10))

    # Illustrative axis following the polar layout: a radial arrow up a bar's
    # left edge = "price", and an arc bent along the track base = "volume".
    # Direction only (not a measured scale). Anchored to the FIRST bus clockwise
    # from the top that has a bar, skipping the demand-labeled bus — so it never
    # crowds "demand" and is picked dynamically (no reliance on any bus number).
    _key_artists: list[Artist] = []
    _key_labels: list[_ChordLabel] = []
    if axis_key:
        _pax = ax if ax is not None else fig.axes[0]

        def _ad(d: float) -> float:
            d %= 360
            return min(d, 360 - d)

        _cn = {}
        for _b in all_buses:
            _d0, _d1 = circos.get_sector(f"Bus {_b}").deg_lim
            _cn[_b] = ((_d0 + _d1) / 2) % 360

        def _has_bar(b: Bus) -> bool:
            return (bus_gen_range.get(b, (0, 0))[1] > 0.5
                    or bus_load_range.get(b, (0, 0))[1]
                    > bus_load_range.get(b, (0, 0))[0])

        # demand bus (top load) to keep the axis away from it
        _ldb = [b for b in all_buses if demand_by_bus.get(b, 0) > 0.5
                and bus_load_range.get(b, (0, 0))[1] > bus_load_range.get(b, (0, 0))[0]]
        _demb = min(_ldb, key=lambda b: _ad(_cn[b])) if _ldb else None

        # walk clockwise from the sector after the top; first bar that isn't demand
        _N = len(all_buses)
        _topb = min(all_buses, key=lambda b: _ad(_cn[b]))
        _ti = all_buses.index(_topb)
        _axb = None
        for _k in range(1, _N + 1):
            _b = all_buses[(_ti + _k) % _N]
            if _b != _demb and _has_bar(_b):
                _axb = _b
                break

        if _axb is not None:
            _sec = circos.get_sector(f"Bus {_axb}")
            _gx0, _gx1 = bus_gen_range.get(_axb, (0, 0))
            if _gx1 - _gx0 > 0.5:                      # prefer the generation bar
                _x0, _x1 = _gx0, _gx1
            else:                                      # else the load bar
                _x0, _x1 = bus_load_range.get(_axb, (0, 0))
            _col = _darken(bus_colors[_axb], 0.5)
            _afs = max(6.0, _fs_load - 1)
            _arr = dict(arrowstyle='-|>', color=_col, lw=1.4, mutation_scale=10,
                        shrinkA=0, shrinkB=0)
            # Origin sits in the white gap between the inner and middle tracks
            # (inner ends ~53, middle starts at 55) so the corner is off the bars.
            _r0 = 54.0
            # price: radial arrow up the bar's left edge from the gap, pointing out
            _radL = _sec.x_to_rad(_x0 + 0.6)
            _key_artists.append(_pax.annotate(
                '', xy=(_radL, 90), xytext=(_radL, _r0),
                arrowprops=_arr, annotation_clip=False, zorder=21))
            _pax.text(_radL, 93, 'price', ha='center', va='center',
                      fontsize=_afs, fontweight='bold', color=_col,
                      clip_on=False, zorder=21)
            # "price" that collides steps out along its arrow, then aside
            _key_labels.append(_ChordLabel(
                'price', _radL, 93,
                _block_candidates(_radL, 93, _radL - 0.12, _radL + 0.12, 91.0, 97.0),
                []))
            # volume: arc along the gap, in the increasing-volume direction
            _xr = _x0 + 0.6 + min((_x1 - _x0) - 1.2, max((_x1 - _x0) * 0.7, 6))
            _ths = np.linspace(_sec.x_to_rad(_x0 + 0.6), _sec.x_to_rad(_xr), 24)
            _key_artists += _pax.plot(_ths, [_r0] * 24, color=_col, lw=1.4, zorder=21,
                                      clip_on=False, solid_capstyle='round')
            _key_artists.append(_pax.annotate(
                '', xy=(_ths[-1], _r0), xytext=(_ths[-3], _r0),
                arrowprops=_arr, annotation_clip=False, zorder=21))
            _pax.text(_sec.x_to_rad(_xr), _r0 - 3.0, 'volume', ha='center',
                      va='center', fontsize=_afs, fontweight='bold', color=_col,
                      clip_on=False, zorder=21)
            # "volume" that its arrowhead or another text reaches moves back
            # under the arc's shaft, level and then lying along the arc, each
            # a step further in, and last past the arrow's tip on the arc's
            # line, up to 0.2 rad.
            _tha, _thb = float(_ths[0]), float(_ths[-1])
            _key_labels.append(_ChordLabel(
                'volume', _sec.x_to_rad(_xr), _r0 - 3.0,
                [(_thb - f * (_thb - _tha), rr, orient)
                 for orient in (None, "horizontal")
                 for rr in (_r0 - 3.0, _r0 - 3.5, _r0 - 4.0, _r0 - 4.5)
                 for f in np.linspace(0.1, 1.0, 10)]
                + [(_thb + np.sign(_thb - _tha) * d, rr, None)
                   for rr in (_r0, _r0 - 1.0)
                   for d in np.linspace(0.02, 0.2, 10)],
                []))

    # An outer bus label that a bid label (or any other text) reaches moves
    # outward until it clears; a ring with no collision is unchanged.
    # Measured when the axes draws, so the ring's final size is the one used.
    _rax = ax if ax is not None else fig.axes[0]

    def _clear_on_draw(renderer: RendererBase) -> None:
        placed = _clear_chord_labels(_rax, renderer,
                                     _chord_labels + _key_labels + _role_labels,
                                     _ring_lines, _key_artists, _ring_labels)
        _clear_ring_labels(_rax, renderer, _ring_labels, inner_texts=placed)

    _rax.add_artist(_LabelClearance(_clear_on_draw, list(_rax.texts)))

    if title and ax is None:
        fig.text(0.5, 0.985, title, ha='center', fontsize=13, fontweight='bold')

    # (System clearing price annotation removed — LMPs shown per-bus in labels)

    if not show_legend:
        return fig

    # --- Legend: bus colors ---
    legend_patches = []
    for bus in all_buses:
        bc = bus_colors[bus]
        has_gen = bus in supply_by_bus and len(supply_by_bus.get(bus, [])) > 0
        has_load = demand_by_bus.get(bus, 0) > 0
        if has_gen or has_load:
            parts = []
            if has_gen:
                units = ', '.join(g['unit_id'] for g in supply_by_bus[bus])
                parts.append(f"Gen: {units}")
            if has_load:
                parts.append(f"Load: {demand_by_bus[bus]:.0f} MW")
            label = f"Bus {bus} — {'; '.join(parts)}"
        else:
            label = f"Bus {bus} — Transit"
        legend_patches.append(Patch(fc=bc, alpha=0.7, ec=_darken(bc),
                                    label=label))

    fig.legend(handles=legend_patches, loc='lower left',
               fontsize=7, framealpha=0.9, title='Bus Legend', title_fontsize=8,
               edgecolor='gray')

    return fig


def plot_nodal_from_results(
    orders: pd.DataFrame,
    meta: pd.DataFrame,
    network: pypsa.Network,
    time: pd.Timestamp | None = None,
    flows: Sequence[tuple[Bus, Bus, float] | tuple[Bus, Bus, float, str]]
    | Literal['proportional']
    | None = 'proportional',
    bus_colors: dict[Bus, str] | None = None,
    bus_lmps: BusValues | None = None,
) -> Figure:
    """
    High-level entry point: pick a timestep and plot.

    Parameters
    ----------
    orders : DataFrame
        Full market_orders.csv
    meta : DataFrame
        Full market_meta.csv
    network : pypsa.Network
        PyPSA network
    time : Timestamp, optional
        Hour to plot. Defaults to peak supply hour.
    flows : list or 'proportional', optional
        If 'proportional' (default), auto-compute copperplate flows.
        If list of (src, dst, MW) tuples, use those directly.
        If None, no chords.
    bus_colors : dict, optional
        {bus: color_hex}. Auto-assigned if None.
        Pass the same dict to a network topology plot for visual consistency.
    bus_lmps : dict, optional
        {bus: lmp_price}. Per-bus LMPs from nodal clearing.
        Falls back to system clearing_price for all buses.

    Returns
    -------
    fig : matplotlib Figure
    """
    if time is None:
        time = meta.loc[meta['supply_volume_energy'].idxmax(), 'time']

    hour = orders[orders['start_time'] == time].copy()
    clearing_price = meta.loc[meta['time'] == time, 'price'].iloc[0]

    supply_by_bus, demand_by_bus, lines_df, all_buses = prepare_bus_data(
        hour, network
    )

    if flows == 'proportional':
        flows = compute_proportional_flows(supply_by_bus, demand_by_bus)

    title = f"Nodal View — {time}"

    return plot_nodal_circlize(
        supply_by_bus, demand_by_bus, all_buses,
        flows=flows,
        clearing_price=clearing_price,
        bus_lmps=bus_lmps,
        title=title,
        bus_colors=bus_colors,
    )


# --- Standard IEEE 9-bus layout coordinates ---
# Hexagonal ring (4-5-6-7-8-9) with gen spurs (1 off 4, 3 off 6, 2 off 8).
# Arranged so no lines cross.
IEEE9_COORDS = {
    '4': (-1.5, -0.5),  # transit (ring, left-bottom)
    '5': (-1.5,  1.5),  # load (ring, left-top)
    '6': ( 0.0,  2.5),  # transit (ring, top)
    '7': ( 1.5,  1.5),  # load (ring, right-top)
    '8': ( 1.5, -0.5),  # transit (ring, right-bottom)
    '9': ( 0.0, -1.5),  # load (ring, bottom)
    '1': (-3.0, -1.5),  # slack gen (spur from 4)
    '3': ( 0.0,  4.5),  # gen (spur from 6)
    '2': ( 3.0, -1.5),  # gen (spur from 8)
}


def rotate_coords(
    coords: dict[Bus, tuple[float, float]], deg: float
) -> dict[Bus, tuple[float, float]]:
    """Rotate a {bus: (x, y)} layout about the origin by ``deg`` degrees
    (counter-clockwise). 180 flips top↔bottom; the rigid rotation keeps the
    no-crossing IEEE9 layout intact. Returns a new dict (does not mutate)."""
    import math
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return {b: (x * c - y * s, x * s + y * c) for b, (x, y) in coords.items()}


def _compute_dc_line_flows(
    network: pypsa.Network,
    supply_by_bus: dict[Bus, list[dict[str, Any]]] | None,
    demand_by_bus: Loads | None,
) -> LineValues:
    """
    Compute DC power flow on each line from gen dispatch and load.

    Returns dict: {line_name: flow_MW} where positive = bus0→bus1.
    """
    buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))
    lines = network.lines
    n_bus = len(buses)
    n_line = len(lines)
    bus_idx = {b: i for i, b in enumerate(buses)}

    b_line = np.zeros(n_line)
    C = np.zeros((n_line, n_bus))
    for li, (line_name, row) in enumerate(lines.iterrows()):
        x = row.get('x_pu_eff', row.get('x_pu', row.get('x', 0.01)))
        if abs(x) < 1e-10:
            x = 0.01
        b_line[li] = 1.0 / x
        C[li, bus_idx[str(row['bus0'])]] = 1.0
        C[li, bus_idx[str(row['bus1'])]] = -1.0

    B_bus = C.T @ np.diag(b_line) @ C
    keep = [i for i in range(n_bus) if i != 0]
    B_reduced = B_bus[np.ix_(keep, keep)]
    try:
        B_inv = np.linalg.inv(B_reduced)
    except np.linalg.LinAlgError:
        return {}

    B_inv_full = np.zeros((n_bus, n_bus))
    for ii, i in enumerate(keep):
        for jj, j in enumerate(keep):
            B_inv_full[i, j] = B_inv[ii, jj]

    PTDF = np.diag(b_line) @ C @ B_inv_full

    P_net = np.zeros(n_bus)
    for bus, gens in (supply_by_bus or {}).items():
        P_net[bus_idx[bus]] += sum(g['accepted_volume'] for g in gens)
    for bus, mw in (demand_by_bus or {}).items():
        P_net[bus_idx[bus]] -= mw

    line_flows_vec = PTDF @ P_net
    return {lines.index[li]: line_flows_vec[li] for li in range(n_line)}


def plot_network_topology(
    network: pypsa.Network,
    supply_by_bus: dict[Bus, list[dict[str, Any]]] | None = None,
    demand_by_bus: Loads | None = None,
    bus_colors: dict[Bus, str] | None = None,
    bus_coords: dict[Bus, tuple[float, float]] | None = None,
    bus_lmps: BusValues | None = None,
    bus_net_mw: BusValues | None = None,
    demand_served_by_bus: BusValues | None = None,
    line_flows: LineValues | None = None,
    line_widths: LineValues | None = None,
    line_colors: dict[Line, str] | None = None,
    constrained_lines: Collection[Line] | None = None,
    title: str = 'IEEE 9-Bus Network',
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 7),
    number_position: str = 'inside',
    box_node_header: bool = False,
    node_number_fontsize: float = 10,
    annot_fontsize: float = 6,
    title_fontsize: float = 13,
    lmp_only: bool = False,
    flow_label_fontsize: float = 9,
    flow_labels: bool = True,
    parallel_gap: float = 0.12,
    net_label_offset: float = 28.0,
    title_pad: float | None = None,
    net_mw_zero_label: bool = False,
    net_mw_decimals: int = 0,
) -> tuple[Figure, Axes]:
    """
    Draw the network topology diagram with bus colors matching the circlize plot.

    ``net_mw_zero_label`` (default False) labels every bus that is a key of
    ``bus_net_mw`` even when its value is zero, as a gray "0 MW" chip (304a's
    market-inputs view, where a market assumes zero injections at the neighbor's
    buses). By default a net injection under 0.5 MW is not labelled.
    ``net_mw_decimals`` (default 0) sets the decimals of those chips, so a figure
    whose chips must match a table to 0.1 MW (304a's change-in-injection view) can
    show them; the default keeps every existing figure unchanged.

    Labels that collide move when the figure draws, and a figure with no
    collision draws as placed (2026-11-07): a flow label moves off a chip, a
    flow arrow, a drawn line or another flow label (``_clear_flow_obstacles``),
    and a chip or info box that a drawn line crosses moves to a clear offset
    (``_clear_chips``).

    Each node is colored by its bus color, sized by generation + load.
    Lines are drawn between connected buses with flow arrows and magnitudes.
    Constrained lines (flow >= 95% of s_nom) are drawn in red.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with buses and lines.
    supply_by_bus : dict, optional
        {bus: [{'unit_id', 'price', 'volume', 'accepted_volume'}, ...]}
    demand_by_bus : dict, optional
        {bus: total_demand_MW} (requested)
    bus_colors : dict, optional
        {bus: color_hex}. Must match circlize plot colors.
    bus_coords : dict, optional
        {bus: (x, y)}. Defaults to IEEE9_COORDS.
    bus_lmps : dict, optional
        {bus: lmp_price}. Annotated on each bus if provided.
    demand_served_by_bus : dict, optional
        {bus: served_MW}. If provided, load annotations show served/requested.
    line_flows : dict, optional
        {line_name: flow_MW}. Positive = bus0→bus1. If None, computed from
        DC power flow using network impedances and supply/demand data.
    line_widths : dict, optional
        {line_name: linewidth}. Encodes a per-line drawing width — e.g. line
        "slipperiness" (susceptance b = 1/x; wider = lower reactance = carries
        more flow per unit angle). When given, width is this orthogonal channel
        and color still flags congestion (red). When None, the legacy behavior
        applies (lw=2, lw=3 for constrained lines).
    line_colors : dict, optional
        {line_name: color_hex}. The "ownership" color of a line — e.g. the
        balancing authority / market that manages it. Used as the line's base
        color so the diagram reads which operator monitors which corridor. A
        line absent from the dict stays gray (e.g. a tie that no single operator
        owns). Congestion (``constrained_lines``) still overrides to red on top
        of this — a congested line is red regardless of its owner.
    constrained_lines : set or list of str, optional
        The set of genuinely constrained lines — those whose congestion shadow
        price μ is non-zero (a binding transmission limit), or physically
        overloaded lines. A line is drawn red **iff** it is in this set. This is
        the only thing that triggers the constrained (red) treatment: a line that
        merely reaches its rating because the generator behind it is maxed out
        has μ = 0 and stays gray. When None (or a line is absent from the set),
        the line is not constrained. Flow magnitude never drives the color.
    title : str, optional
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    supply_by_bus = supply_by_bus or {}
    demand_by_bus = demand_by_bus or {}

    all_buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))

    if bus_colors is None:
        bus_colors = assign_bus_colors(all_buses, supply_by_bus, demand_by_bus)

    coords = bus_coords or IEEE9_COORDS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]  # stub adds None

    # --- Line flows: use solver flows if provided, otherwise compute DC ---
    if line_flows is None:
        line_flows = _compute_dc_line_flows(network, supply_by_bus, demand_by_bus)

    # --- Draw lines with flow arrows ---
    lines = network.lines[['bus0', 'bus1', 's_nom']]
    # Parallel circuits (same bus pair) get a small perpendicular offset so they
    # render side-by-side rather than overplotting. Single lines are unaffected.
    _groups: dict[frozenset[Bus], list[Line]] = {}
    for _ln, _lr in lines.iterrows():
        _groups.setdefault(frozenset((_lr['bus0'], _lr['bus1'])), []).append(_ln)
    _par_offset = {}
    for _grp in _groups.values():
        _k = len(_grp)
        for _j, _ln in enumerate(sorted(_grp)):
            _par_offset[_ln] = (_j - (_k - 1) / 2.0) * parallel_gap
    # Each flow label with its line's midpoint and normal offset, and each bus chip,
    # for the collision pass at the end (_clear_flow_labels).
    _flow_texts: list[tuple[Text, float, float, float, float]] = []
    _chips: list[Annotation] = []
    # The obstacles of the passes run at draw time (_clear_chips and
    # _clear_flow_obstacles): each flow arrow with its shaft width (points), each
    # drawn line as (x0, y0, x1, y1, width in points), each bus marker as
    # (x, y, radius in points), each bus-number chip, and each chip with the unit
    # vector from the chart centre to its bus.
    _arrows: list[tuple[Annotation, float]] = []
    _segments: list[tuple[float, float, float, float, float]] = []
    _discs: list[tuple[float, float, float]] = []
    _numbers: list[Annotation] = []
    _chip_dirs: list[tuple[Annotation, float, float]] = []
    for line_name, line in lines.iterrows():
        b0, b1 = line['bus0'], line['bus1']
        if b0 not in coords or b1 not in coords:
            continue
        x0, y0 = coords[b0]
        x1, y1 = coords[b1]
        off = _par_offset.get(line_name, 0.0)
        if off:
            _dx, _dy = x1 - x0, y1 - y0
            _L = (_dx * _dx + _dy * _dy) ** 0.5 or 1.0
            _ox, _oy = -_dy / _L * off, _dx / _L * off
            x0, y0, x1, y1 = x0 + _ox, y0 + _oy, x1 + _ox, y1 + _oy
        s_nom = line['s_nom']
        flow = line_flows.get(line_name, 0.0)

        # Red ONLY for genuinely constrained lines: those whose congestion shadow
        # price is non-zero (or are passed as overloaded). Flow merely reaching
        # the rating is NOT congestion — a radial line at its limit because the
        # generator behind it is maxed out has zero shadow price and stays gray.
        # A line is red iff it is in the caller-supplied set; with no set, none.
        constrained = (constrained_lines is not None
                       and line_name in constrained_lines)
        # Color priority: congestion red > owner (BA/market) color > gray.
        # A congested line is always red; otherwise it takes its owner's color
        # if one was supplied, and falls back to gray (e.g. an unowned tie).
        if constrained:
            line_color = '#E74C3C'
        elif line_colors is not None and line_name in line_colors:
            line_color = line_colors[line_name]
        else:
            line_color = '#AAB7B8'
        if line_widths is not None:
            # Width encodes slipperiness (susceptance); color still flags congestion.
            line_lw = line_widths.get(line_name, 2)
        else:
            line_lw = 3 if constrained else 2

        ax.plot([x0, x1], [y0, y1], color=line_color, lw=line_lw, zorder=1,
                solid_capstyle='round')
        _segments.append((x0, y0, x1, y1, float(line_lw)))

        # Flow arrow at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if abs(flow) > 0.5:
            # Arrow direction: positive flow = bus0→bus1
            if flow > 0:
                dx, dy = x1 - x0, y1 - y0
            else:
                dx, dy = x0 - x1, y0 - y1
            # Normalize; arrow length, shaft, and head all scale with line width
            # so a slippery (wide) line also carries a visually heavier arrow.
            arrow_half = 0.18 + 0.022 * line_lw
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                dx, dy = dx / length * arrow_half, dy / length * arrow_half
            _arrows.append((ax.annotate(
                '', xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                arrowprops={'arrowstyle': '-|>', 'color': line_color,
                            'lw': max(1.0, 0.9 * line_lw),
                            'mutation_scale': 8 + 2.4 * line_lw},
                zorder=2), max(1.0, 0.9 * line_lw)))

        # Flow label: "flow/capacity" offset perpendicular to line (opt-out via
        # flow_labels=False). Skip lines carrying ~no flow so dead paths don't print
        # a "0/limit" tag; a constrained line is always labeled.
        if flow_labels and (abs(flow) > 0.5 or constrained):
            flow_label = f"{abs(flow):.0f}/{s_nom:.0f}"
            label_color = '#C0392B' if constrained else '#7F8C8D'
            fontweight = 'bold' if constrained else 'normal'
            # Compute perpendicular offset so label sits beside the arrow
            ldx, ldy = x1 - x0, y1 - y0
            ll = (ldx**2 + ldy**2) ** 0.5
            if ll > 0:
                perp_x, perp_y = -ldy / ll * 0.25, ldx / ll * 0.25
            else:
                perp_x, perp_y = 0, 0.25
            _flow_texts.append((ax.text(
                mx + perp_x, my + perp_y, flow_label,
                fontsize=flow_label_fontsize, color=label_color,
                fontweight=fontweight, ha='center', va='center',
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1)), mx, my, perp_x, perp_y))

    # --- Draw buses ---
    # Chart center (mean of the drawn buses) -- used to push per-bus labels
    # radially OUTWARD (left nodes label left, right right, top up, bottom down)
    # so the chips clear the line-flow tags instead of all shifting one way.
    _bxy = [coords[b] for b in all_buses if b in coords]
    _cx = sum(p[0] for p in _bxy) / len(_bxy) if _bxy else 0.0
    _cy = sum(p[1] for p in _bxy) / len(_bxy) if _bxy else 0.0

    for bus in all_buses:
        if bus not in coords:
            continue
        x, y = coords[bus]
        bc = bus_colors.get(bus, TRANSIT_COLOR)
        bc_dark = _darken(bc, 0.6)

        # Size by nameplate capacity + load (min size for transit)
        gen_mw = sum(g.get('capacity', g['volume']) for g in supply_by_bus.get(bus, []))
        load_mw = demand_by_bus.get(bus, 0)
        total_mw = gen_mw + load_mw
        node_size = max(total_mw * 3, 200)  # scale for visibility

        ax.scatter(x, y, s=node_size, c=bc, ec=bc_dark, lw=2,
                   zorder=3, alpha=0.85)
        # marker radius in points: sqrt of the area in points squared, halved,
        # plus half the 2-point edge
        _discs.append((x, y, node_size ** 0.5 / 2 + 1.0))
        _rx, _ry = x - _cx, y - _cy
        _rl = (_rx * _rx + _ry * _ry) ** 0.5 or 1.0
        _n_chips = len(_chips)

        # Bus number label. 'inside' = white number in the bubble (default);
        # 'outside' = dark number on a small chip beside the bubble, so the
        # colored marker stays clean for print at half-page width.
        if number_position == 'outside':
            _numbers.append(ax.annotate(
                bus, (x, y), fontsize=node_number_fontsize,
                fontweight='bold', color=bc_dark, zorder=6,
                ha='center', va='center',
                xytext=(-13, 11), textcoords='offset points',
                bbox={'boxstyle': 'circle,pad=0.18', 'fc': 'white',
                      'ec': bc_dark, 'lw': 1.2, 'alpha': 0.95}))
        else:
            ax.text(x, y, bus, fontsize=node_number_fontsize, fontweight='bold',
                    ha='center', va='center', color='white', zorder=4)

        # lmp_only: skip the gen/bid/load info box entirely (that detail now lives
        # in the circlize panel) and print only a per-bus annotation, no info box.
        # With bus_net_mw given, label the NET INJECTION (dispatched generation +,
        # load -) in a green/maroon chip -- for a flow-focused panel where price is
        # better read off the dispatch ring; otherwise fall back to the LMP.
        if lmp_only:
            # radial push-out: offset the chip away from the chart center, and anchor
            # the text on the side facing the node so it reads outward.
            rdx, rdy = x - _cx, y - _cy
            rl = (rdx * rdx + rdy * rdy) ** 0.5 or 1.0
            ox, oy = net_label_offset * rdx / rl, net_label_offset * rdy / rl
            _ha = 'left' if ox > 3 else ('right' if ox < -3 else 'center')
            _va = 'bottom' if oy > 3 else ('top' if oy < -3 else 'center')
            _show_net = bus_net_mw is not None and (
                abs(bus_net_mw.get(bus, 0.0)) > 0.5
                or (net_mw_zero_label and bus in bus_net_mw))
            if _show_net:
                mw = bus_net_mw[bus]  # type: ignore[index]  # _show_net implies a dict
                if abs(mw) <= 0.5:                      # an explicit zero (opt-in)
                    txt, col = "0 MW", '#7F8C8D'
                else:
                    txt = f"{mw:+.{int(net_mw_decimals)}f} MW"
                    col = '#1E8449' if mw > 0 else '#922B21'
                if bus_lmps and bus in bus_lmps:        # price alongside the injection
                    txt += f"\n${bus_lmps[bus]:.1f}"
                _chips.append(ax.annotate(
                    txt, (x, y), fontsize=annot_fontsize + 2, color=col,
                    fontweight='bold', ha=_ha, va=_va,
                    xytext=(ox, oy), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec=col, lw=1.2, alpha=0.92),
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.7,
                                    alpha=0.5, shrinkA=2, shrinkB=3), zorder=5))
            elif bus_lmps and bus in bus_lmps:
                _chips.append(ax.annotate(
                    f"${bus_lmps[bus]:.1f}", (x, y),
                    fontsize=annot_fontsize + 2, color=bc_dark,
                    fontweight='bold', ha=_ha, va=_va,
                    xytext=(ox, oy), textcoords='offset points', zorder=5))
            _chip_dirs += [(c, _rx / _rl, _ry / _rl) for c in _chips[_n_chips:]]
            continue

        # Build annotation text. Optional color-matched node header so the info
        # box is identifiable when the number is outside the bubble.
        annotations = []
        if box_node_header:
            annotations.append(f"Bus {bus}")
        if gen_mw > 0:
            for g in supply_by_bus[bus]:
                acc = g.get('accepted_volume', 0)
                nameplate = g.get('capacity', g['volume'])
                bid_vol = g['volume']
                # Line 1: unit name with dispatch/capacity
                annotations.append(f"{g['unit_id']}: {acc:.0f}/{nameplate:.0f} MW")
                # Line 2: bid details (price × volume offered)
                annotations.append(f"  Bid: ${g['price']:.1f} × {bid_vol:.0f} MW")
        if load_mw > 0:
            if demand_served_by_bus and bus in demand_served_by_bus:
                served = demand_served_by_bus[bus]
                annotations.append(f"Load: {served:.0f}/{load_mw:.0f} MW")
            else:
                annotations.append(f"Load: {load_mw:.0f} MW")
        # Net injection (+) / withdrawal (−) chip in the info box: dispatched gen −
        # served load + exo, so a flow-focused topology reads what each bus puts on
        # the transmission elements, not just its price. Only when bus_net_mw is supplied.
        if bus_net_mw is not None and abs(bus_net_mw.get(bus, 0.0)) > 0.5:
            _nmw = bus_net_mw[bus]
            annotations.append(f"Net: +{_nmw:.0f} MW" if _nmw > 0 else f"Net: {_nmw:.0f} MW")
        if bus_lmps and bus in bus_lmps:
            annotations.append(f"LMP: ${bus_lmps[bus]:.1f}")

        if annotations:
            label = '\n'.join(annotations)
            _chips.append(ax.annotate(
                label, (x, y), fontsize=annot_fontsize, color=bc_dark,
                fontweight='bold',
                xytext=(12, -12), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=bc, lw=1.4, alpha=0.9),
                zorder=5))
            _chip_dirs.append((_chips[-1], _rx / _rl, _ry / _rl))

    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_pad)
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.axis('off')
    _clear_flow_labels(ax, _flow_texts, _chips)
    # The lines and the flow arrows are obstacles as well, and a chip that a line
    # crosses moves out; both are measured when the axes draws, against the
    # limits a later overlay (a rights arc) leaves.
    if _flow_texts or _chip_dirs:

        def _clear_on_draw(renderer: RendererBase) -> None:
            _clear_chips(ax, renderer, _chip_dirs, _segments, _discs, _numbers)
            _clear_flow_obstacles(ax, renderer, _flow_texts, _chips, _arrows,
                                  _segments)

        ax.add_artist(_LabelClearance(
            _clear_on_draw, [*(t for t, *_ in _flow_texts), *_chips]))

    return fig, ax


def _clear_flow_labels(
    ax: Axes,
    flow_texts: Sequence[tuple[Text, float, float, float, float]],
    chips: Sequence[Annotation],
) -> None:
    """Move a line's flow label off a bus chip that covers it, along the line's normal.

    Display only: a label that overlaps no chip stays where it was, so a figure
    without a collision is unchanged. A covered label tries the far side of its
    line, then 1.6 times the offset on the far and the near side, and takes the
    first position that clears every chip and every other flow label. A label no
    position clears stays where it was. ``flow_texts`` holds each label with its
    line's midpoint and normal offset in data units, and ``chips`` the bus chips
    and info boxes, placed in offset points from their bus.
    """
    get_renderer = getattr(ax.figure.canvas, 'get_renderer', None)
    if not flow_texts or not chips or get_renderer is None:
        return
    renderer = get_renderer()
    ax.autoscale_view()                  # the limits and aspect the draw will use
    ax.apply_aspect()
    pad = renderer.points_to_pixels(2.0)

    def box(t: Text) -> Bbox:
        if isinstance(t, Annotation):
            t.update_positions(renderer)
        b = Text.get_window_extent(t, renderer)   # the text, not a chip's leader line
        return Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)

    chip_boxes = [box(c) for c in chips]
    for k, (t, mx, my, px, py) in enumerate(flow_texts):
        if not any(box(t).overlaps(c) for c in chip_boxes):
            continue
        others = [box(o[0]) for j, o in enumerate(flow_texts) if j != k]
        for scale in (-1.0, -1.6, 1.6):
            t.set_position((mx + scale * px, my + scale * py))
            b = box(t)
            if not any(b.overlaps(c) for c in chip_boxes + others):
                break
        else:
            t.set_position((mx + px, my + py))


# ─── Label clearance (display only) ─────────────────────────────────────────
# The passes below move a label only where its box overlaps another drawn
# element, so a figure without a collision draws as before. They change no
# number the figure states. The passes added on 2026-11-07 run when the axes
# draws (``_LabelClearance``), after the axes has applied its final limits and
# aspect, so they measure the geometry the labels are drawn in and leave the
# axes state that a later overlay sees unchanged. The placement rules are a
# display choice of the Lab's own; no source prescribes them.

#: Candidate placements of a covered flow label, as (normal, along) multiples of
#: its 0.25-unit offset from the line midpoint. The first three are the far side
#: and 1.6 times the offset on the far and the near side, the order of
#: ``_clear_flow_labels``; the rest step further out along the normal, then
#: along the line past the arrow, then out to 3.5 times the offset for a label
#: wider than a wide line's clearance (2026-11-07).
_FLOW_LABEL_STEPS: tuple[tuple[float, float], ...] = (
    (-1.0, 0.0), (-1.6, 0.0), (1.6, 0.0),
    (1.4, 0.0), (-1.4, 0.0), (1.8, 0.0), (-1.8, 0.0), (2.2, 0.0), (-2.2, 0.0),
    (1.0, 1.8), (1.0, -1.8), (-1.0, 1.8), (-1.0, -1.8),
    (1.4, 1.8), (1.4, -1.8), (-1.4, 1.8), (-1.4, -1.8),
    (2.6, 0.0), (-2.6, 0.0), (3.0, 0.0), (-3.0, 0.0), (3.5, 0.0), (-3.5, 0.0),
)


class _LabelClearance(Artist):
    """An invisible artist that runs a label-clearance pass when its axes draws.

    ``Axes.draw`` applies the autoscaled limits and the aspect before it draws
    any child, and this artist sorts before every text (zorder 0), so ``run``
    receives the renderer of the draw and measures the final geometry. Each draw
    first returns every text of ``movable`` to the position and rotation it had
    when the artist was made, so repeated draws place the labels the same way.
    The texts'
    stale callbacks are held while the pass moves them, so a move inside a draw
    does not request another draw. The artist draws nothing and takes no part in
    a layout or a tight bounding box.
    """

    def __init__(
        self,
        run: Callable[[RendererBase], None],
        movable: Sequence[Text],
    ) -> None:
        super().__init__()
        self._run = run
        self._movable = list(movable)
        self._home = [(t.get_position(), t.get_rotation()) for t in self._movable]
        self.set_zorder(0)
        self.set_in_layout(False)

    def draw(self, renderer: RendererBase) -> None:
        """Return the movable texts to their placements, then run the pass."""
        if not self.get_visible():
            return
        held = [t.stale_callback for t in self._movable]
        for t in self._movable:
            t.stale_callback = None
        try:
            for t, (home, rot) in zip(self._movable, self._home, strict=True):
                if t.get_position() != home:
                    t.set_position(home)
                if t.get_rotation() != rot:
                    t.set_rotation(rot)
            self._run(renderer)
        finally:
            for t, cb in zip(self._movable, held, strict=True):
                t.stale_callback = cb
        self.stale = False


def _px(renderer: RendererBase, points: float) -> float:
    """A length in points as display pixels at the renderer's resolution."""
    return float(np.asarray(renderer.points_to_pixels(points)))


def _text_box(t: Text, renderer: RendererBase, pad: float) -> Bbox:
    """The display box of a text alone, padded by ``pad`` pixels on each side.

    An annotation's leader line and drawn bbox patch are left out; the position
    is brought up to date first, as the draw does.
    """
    if isinstance(t, Annotation):
        t.update_positions(renderer)
    b = Text.get_window_extent(t, renderer)   # the text, not a chip's leader line
    return Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)


def _drawn_box(t: Text, renderer: RendererBase) -> Bbox:
    """The display box a text occupies as drawn, its bbox patch and edge included.

    A text with no bbox patch is its own extent padded by 2 points.
    """
    if isinstance(t, Annotation):
        t.update_positions(renderer)
    patch = t.get_bbox_patch()
    if patch is None:
        return _text_box(t, renderer, _px(renderer, 2.0))
    t.update_bbox_position_size(renderer)
    return patch.get_window_extent(renderer).padded(
        _px(renderer, patch.get_linewidth() / 2 + 0.5))


def _segment_paths(
    ax: Axes,
    segments: Sequence[tuple[float, float, float, float, float]],
    renderer: RendererBase,
) -> list[tuple[Path, float]]:
    """Each drawn line as a display-space path with its half width in pixels."""
    return [(Path(ax.transData.transform([(x0, y0), (x1, y1)])),
             _px(renderer, lw / 2))
            for x0, y0, x1, y1, lw in segments]


def _arrow_paths(
    patch: FancyArrowPatch, renderer: RendererBase, lw: float | None = None
) -> list[tuple[Path, float]]:
    """An arrow patch as drawn: its shaft and head polygons in display pixels.

    Each polygon comes with half the line width in pixels (``lw`` in points,
    the patch's own width by default). The polygons are returned one by one,
    because the arrow's CLOSEPOLY vertex is a placeholder at the origin that a
    whole-path test reads as a point. The patch scales its head and width by a
    dpi factor that it sets only in its own draw, after a clearance pass; it is
    set here as the draw sets it, so the first draw measures the arrow it
    paints.
    """
    patch._dpi_cor = _px(renderer, 1.0)  # type: ignore[attr-defined]  # a matplotlib attribute the stubs omit
    drawn = patch.get_transform().transform_path(patch.get_path())
    hw = _px(renderer, (patch.get_linewidth() if lw is None else lw) / 2)
    polys = [np.asarray(q) for q in drawn.to_polygons(closed_only=False)]
    return [(Path(q), hw) for q in polys if len(q) > 1]


def _box_hits_disc(b: Bbox, cx: float, cy: float, r: float) -> bool:
    """Whether a display box comes within ``r`` pixels of the point ``(cx, cy)``."""
    dx = max(b.x0 - cx, 0.0, cx - b.x1)
    dy = max(b.y0 - cy, 0.0, cy - b.y1)
    return dx * dx + dy * dy < r * r


def _clear_chips(
    ax: Axes,
    renderer: RendererBase,
    chip_dirs: Sequence[tuple[Annotation, float, float]],
    segments: Sequence[tuple[float, float, float, float, float]],
    discs: Sequence[tuple[float, float, float]],
    numbers: Sequence[Annotation],
) -> None:
    """Move a bus chip that a drawn line crosses to the nearest clear offset.

    Display only, run when the axes draws. ``chip_dirs`` holds every chip and
    info box with the unit vector from the chart centre to its bus, ``segments``
    each drawn line as ``(x0, y0, x1, y1, lw)`` in data units and points,
    ``discs`` each bus marker as ``(x, y, r)`` in data units and points, and
    ``numbers`` the bus-number chips. A chip whose drawn box no line crosses
    keeps its offset, so a figure without a collision is unchanged.

    A crossed chip with offset ``o`` (points, length ``L``) from its bus tries,
    in order, ``o + k L u`` for ``k`` = 0.5 and 1 along its unit vector ``u``
    (1.5 and 2 times the offset of a radial chip of the ``lmp_only`` view),
    ``o`` turned about its bus by 30 and 60 degrees each way at 1 and 1.5 times
    its length, and ``o + k L u`` for ``k`` = 1.5 to 3.5 in steps of 0.5. It
    takes the first placement that no line crosses, that clears every other
    chip, every bus-number chip and every bus marker, and whose nearest bus is
    its own, so a chip is never read as a neighbour's; with none clear it keeps
    its offset. The chip keeps its alignment, which faces its bus.
    """
    if not chip_dirs or not segments:
        return
    lines = _segment_paths(ax, segments, renderer)
    markers = [(*ax.transData.transform((x, y)), _px(renderer, r))
               for x, y, r in discs]

    def crossed(b: Bbox) -> bool:
        return any(p.intersects_bbox(b.padded(hw), filled=False) for p, hw in lines)

    def gap(b: Bbox, cx: float, cy: float) -> float:
        return float(np.hypot(max(b.x0 - cx, 0.0, cx - b.x1),
                              max(b.y0 - cy, 0.0, cy - b.y1)))

    for chip, ux, uy in chip_dirs:
        if not crossed(_drawn_box(chip, renderer)):
            continue
        ox, oy = chip.xyann
        size = (ox * ox + oy * oy) ** 0.5
        own_x, own_y = ax.transData.transform(chip.xy)
        others = [_drawn_box(c, renderer) for c, *_ in chip_dirs if c is not chip]
        others += [_drawn_box(n, renderer) for n in numbers]
        turned = [(f * (ox * np.cos(t) - oy * np.sin(t)),
                   f * (ox * np.sin(t) + oy * np.cos(t)))
                  for f in (1.0, 1.5)
                  for t in np.radians((30.0, -30.0, 60.0, -60.0))]
        candidates = ([(ox + k * size * ux, oy + k * size * uy) for k in (0.5, 1.0)]
                      + turned
                      + [(ox + k * size * ux, oy + k * size * uy)
                         for k in (1.5, 2.0, 2.5, 3.0, 3.5)])
        for cand in candidates:
            chip.xyann = (float(cand[0]), float(cand[1]))
            b = _drawn_box(chip, renderer)
            if (not crossed(b) and not any(b.overlaps(o) for o in others)
                    and not any(_box_hits_disc(b, *m) for m in markers)
                    and gap(b, own_x, own_y) <= min(gap(b, mx, my)
                                                    for mx, my, _r in markers)):
                break
        else:
            chip.xyann = (ox, oy)


def _clear_flow_obstacles(
    ax: Axes,
    renderer: RendererBase,
    flow_texts: Sequence[tuple[Text, float, float, float, float]],
    chips: Sequence[Annotation],
    arrows: Sequence[tuple[Annotation, float]],
    segments: Sequence[tuple[float, float, float, float, float]],
) -> None:
    """Move a line's flow label off a chip, a flow arrow, a line or another label.

    Display only, run when the axes draws, after ``_clear_chips``. A label that
    overlaps nothing stays where it was, so a figure without a collision is
    unchanged. ``flow_texts`` holds each label with its line's midpoint and
    normal offset in data units, ``chips`` the bus chips and info boxes,
    ``arrows`` each flow arrow with its shaft width in points, and ``segments``
    each drawn line as ``(x0, y0, x1, y1, lw)`` in data units and points. The
    rights arcs an overlay has added to the axes by the time it draws (each
    ``FancyArrowPatch`` of ``ax.patches``, ``wscc9_figures.draw_rights_arcs``)
    are obstacles as well (2026-11-07). A label
    overlaps a chip or another label when their text boxes, padded by 2 points,
    overlap, and an arrow, an arc or a line when its drawn path, widened by half
    its line width, crosses the label's padded box. A covered label tries the placements
    of ``_FLOW_LABEL_STEPS`` in order and takes the first that clears every
    obstacle; with none clear it takes the placement with the fewest overlaps,
    its own position first.
    """
    if not flow_texts:
        return
    pad = _px(renderer, 2.0)

    def box(t: Text) -> Bbox:
        return _text_box(t, renderer, pad)

    chip_boxes = [box(c) for c in chips]
    paths: list[tuple[Path, float]] = []
    for ann, lw in arrows:
        ann.update_positions(renderer)
        if ann.arrow_patch is not None:
            paths += _arrow_paths(ann.arrow_patch, renderer, lw)
    paths += _segment_paths(ax, segments, renderer)
    for arc in ax.patches:
        if isinstance(arc, FancyArrowPatch):
            paths += _arrow_paths(arc, renderer)

    def overlaps(b: Bbox, others: list[Bbox]) -> int:
        n = sum(b.overlaps(c) for c in chip_boxes + others)
        return n + sum(p.intersects_bbox(b.padded(hw), filled=False)
                       for p, hw in paths)

    for k, (t, mx, my, px, py) in enumerate(flow_texts):
        others = [box(o[0]) for j, o in enumerate(flow_texts) if j != k]
        best_n = overlaps(box(t), others)
        if not best_n:
            continue
        best = t.get_position()
        for s, a in _FLOW_LABEL_STEPS:
            # (py, -px) runs along the line, with the normal offset's length
            t.set_position((mx + s * px + a * py, my + s * py - a * px))
            n = overlaps(box(t), others)
            if n < best_n:
                best, best_n = t.get_position(), n
            if not n:
                break
        t.set_position(best)


def _text_quad(t: Text, renderer: RendererBase) -> np.ndarray:
    """The four display corners of a text's rotated box, in pixels, shape ``(4, 2)``.

    The box is the text's unrotated extent turned by its rotation about the
    centre of its rendered extent, so a tangential label on the ring is measured
    as drawn and not by its axis-aligned envelope. The unrotated extent is read
    from an unrotated copy, so the text itself is not touched (setting a
    rotation back can renormalize it, 360 to 0, and move its antialiasing).
    """
    env = t.get_window_extent(renderer)
    rot = t.get_rotation()
    flat_text = Text()
    flat_text.update_from(t)
    flat_text.set_text(t.get_text())
    flat_text.set_rotation(0)
    owner = t.get_figure()
    if owner is not None:
        flat_text.set_figure(owner)
    flat = flat_text.get_window_extent(renderer)
    hw, hh = flat.width / 2, flat.height / 2
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    centre = np.array([(env.x0 + env.x1) / 2, (env.y0 + env.y1) / 2])
    return corners @ np.array([[c, s], [-s, c]]) + centre


def _quad_gap(a: np.ndarray, b: np.ndarray) -> float:
    """The separation of two convex quadrilaterals along their edge normals, in pixels.

    Positive is a gap; zero or negative means the two overlap (the separating
    axis test).
    """
    gap = -np.inf
    for q in (a, b):
        for i in range(4):
            e = q[(i + 1) % 4] - q[i]
            nn = float(np.hypot(e[0], e[1]))
            if nn == 0.0:
                continue
            n = np.array([-e[1], e[0]]) / nn
            pa, pb = a @ n, b @ n
            gap = max(gap, float(pb.min() - pa.max()), float(pa.min() - pb.max()))
    return float(gap)


def _clear_ring_labels(
    ax: Axes,
    renderer: RendererBase,
    labels: Sequence[tuple[str, float, float]],
    clearance_pt: float = 4.0,
    step: float = 0.25,
    r_limit: float = 130.0,
    inner_texts: Sequence[Text] = (),
) -> None:
    """Move a ring's outer bus label outward until it clears every other text.

    Display only, run when the axes draws. ``labels`` holds each outer label's
    string, angle in radians and radius, as passed to its sector; the radius is
    pycirclize's 0 to 100 track scale. A label whose rotated box clears every
    other text by ``clearance_pt`` points keeps its radius, so a ring without a
    collision is unchanged. A label closer than that (a bid label at
    ``bar_top + 5`` reaches it on a high-priced unit, or a neighbouring bus's
    label touches it) steps outward by ``step`` until its box clears every text
    inside the ring and every other outer label by ``clearance_pt`` points. The
    test that starts the search is the test that ends it (2026-11-07; before, a
    label moved only when its box overlapped). A label that no
    radius up to ``r_limit`` clears keeps its placed radius. A text anchored
    outside the labels (a group label) that a moved label then overlaps steps
    outward the same way. A text of ``inner_texts`` (a ring text the chord
    pass placed, which may sit outside the band) counts inside the ring
    wherever it ends.
    """
    keys = set(labels)
    ring = [t for t in ax.texts if (t.get_text(), *t.get_position()) in keys]
    if not ring:
        return
    r_ring = max(r for _s, _a, r in labels)
    outer = [t for t in ax.texts if t not in ring and t.get_transform() is ax.transData
             and t.get_position()[1] > r_ring
             and not any(t is i for i in inner_texts)]
    inner = [_text_quad(t, renderer) for t in ax.texts
             if t not in ring and t not in outer and t.get_text()]
    clear = _px(renderer, clearance_pt)
    quads = {id(t): _text_quad(t, renderer) for t in ring}

    def step_out(t: Text, obstacles: list[np.ndarray]) -> bool:
        if all(_quad_gap(quads[id(t)], o) >= clear for o in obstacles):
            return False
        theta, r = t.get_position()
        r0 = r
        while r < r_limit:
            r += step
            t.set_position((theta, r))
            quads[id(t)] = _text_quad(t, renderer)
            if all(_quad_gap(quads[id(t)], o) >= clear for o in obstacles):
                break
        else:
            t.set_position((theta, r0))
            quads[id(t)] = _text_quad(t, renderer)
            return False
        return True

    moved = False
    for t in ring:
        moved |= step_out(t, inner + [quads[id(o)] for o in ring if o is not t])
    if moved:
        for t in outer:
            quads[id(t)] = _text_quad(t, renderer)
            step_out(t, [quads[id(o)] for o in ring + outer if o is not t])


class _ChordLabel(NamedTuple):
    """A text of the dispatch ring that the chord pass may move (display only).

    ``text``, ``theta`` and ``r`` identify the text as placed: its string, its
    angle in radians and its radius on pycirclize's 0 to 100 scale.
    ``candidates`` holds the placements it may take, in order, as
    ``(theta, r, orientation)``, where the orientation is ``None`` to keep the
    rotation, ``"horizontal"`` to lie along the ring at the new angle and
    ``"vertical"`` to run along the radius. ``edges`` holds the polylines
    ``(theta, r)`` of the block that contains the text, which the text clears
    as well.
    """

    text: str
    theta: float
    r: float
    candidates: list[tuple[float, float, str | None]]
    edges: list[tuple[np.ndarray, np.ndarray]]


def _ring_arc(
    th0: float, th1: float, r0: float, r1: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """A ring polyline from ``(th0, r0)`` to ``(th1, r1)``, sampled every 0.02 rad.

    With ``r1`` omitted the polyline is an arc at radius ``r0``.
    """
    n = max(2, int(abs(th1 - th0) / 0.02) + 2)
    return np.linspace(th0, th1, n), np.linspace(r0, r0 if r1 is None else r1, n)


def _block_candidates(
    theta: float,
    r: float,
    th0: float,
    th1: float,
    r_lo: float,
    r_hi: float,
    orientations: Sequence[str | None] = (None,),
) -> list[tuple[float, float, str | None]]:
    """Placements of a text inside the block ``[th0, th1] x (r_lo, r_hi)``.

    Steps of a twelfth of the block's angular width and of 0.5 in radius about
    the text's own placement, for each orientation, nearest first. A step
    costs its count of twelfths over 6 plus its count of radius steps over 8,
    and each orientation after the first adds 0.5, so a text turns before it
    slides more than three twelfths. The placed position itself is left out for
    the first orientation.
    """
    lo, hi = min(th0, th1), max(th0, th1)
    step = (hi - lo) / 12
    scored = [(abs(k) / 6 + abs(j) / 8 + 0.5 * i, theta + k * step, r + 0.5 * j, o)
              for i, o in enumerate(orientations)
              for k in range(-6, 7) for j in range(-8, 9)]
    return [(t, rr, o) for d, t, rr, o in sorted(scored, key=lambda c: c[0])
            if lo <= t <= hi and r_lo < rr < r_hi and d > 0]


def _polyline_quad_gap(poly: np.ndarray, quad: np.ndarray) -> float:
    """The distance in pixels from an open polyline ``(n, 2)`` to a convex quad.

    Zero where the polyline crosses the quad or has a vertex inside it.
    """
    q0, q1 = quad, np.roll(quad, -1, axis=0)
    e = q1 - q0
    # a polyline vertex inside the quad lies on the same side of all four edges
    side = (e[None, :, 0] * (poly[:, None, 1] - q0[None, :, 1])
            - e[None, :, 1] * (poly[:, None, 0] - q0[None, :, 0]))
    if ((side >= 0).all(axis=1) | (side <= 0).all(axis=1)).any():
        return 0.0
    a, b = poly[:-1], poly[1:]

    def cross(o: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return ((u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1])
                - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0]))

    pa, pb = a[:, None, :], b[:, None, :]
    qa, qb = q0[None, :, :], q1[None, :, :]
    if ((cross(qa, qb, pa) * cross(qa, qb, pb) < 0)
            & (cross(pa, pb, qa) * cross(pa, pb, qb) < 0)).any():
        return 0.0

    def point_seg(pts: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> float:
        v = s1 - s0
        vv = (v * v).sum(axis=-1)
        vv = np.where(vv == 0, 1.0, vv)
        t = np.clip(((pts[:, None, :] - s0[None]) * v[None]).sum(-1) / vv[None], 0, 1)
        d = pts[:, None, :] - (s0[None] + t[..., None] * v[None])
        return float(np.hypot(d[..., 0], d[..., 1]).min())

    return min(point_seg(poly, q0, q1), point_seg(quad, a, b))


def _clear_chord_labels(
    ax: Axes,
    renderer: RendererBase,
    labels: Sequence[_ChordLabel],
    arcs: Sequence[tuple[np.ndarray, np.ndarray, float]],
    key_artists: Sequence[Artist],
    ring_labels: Sequence[tuple[str, float, float]],
    clearance_pt: float = 2.0,
) -> list[Text]:
    """Move a dispatch-ring text off a drawn line, the axis key or another text.

    Display only, run when the axes draws, before ``_clear_ring_labels``.
    ``labels`` holds the movable texts in the order they are tried: the bid
    labels, the in-bar megawatt labels, the axis key's "price" and "volume",
    then the role labels "supply", "demand" and "dispatch". ``arcs`` holds the
    drawn ring lines as ``(theta, r, lw)``: the bar tops and bases, the dashed
    LMP lines, the marginal-cost lines and the outer band's edges.
    ``key_artists`` holds the axis key's arrows and arc. A text clears when its
    rotated box keeps ``clearance_pt`` points from every other text, from every
    line widened by half its width and from the edges of its own block. The
    outer bus labels of ``ring_labels`` are left out, because
    ``_clear_ring_labels`` moves them afterwards.

    A text that clears keeps its placement, so a ring without a collision is
    unchanged. A text that does not tries its candidates in order and takes the
    first that clears; with none clear it takes the placement with the fewest
    obstacles in reach and, among those, the widest gap, its own placement
    first. Returns the texts of ``labels`` found on the axes, which
    ``_clear_ring_labels`` counts inside the ring wherever they end.
    """
    by_key = {(t.get_text(), *t.get_position()): t for t in ax.texts}
    ring_keys = set(ring_labels)
    ring = {id(t) for t in ax.texts if (t.get_text(), *t.get_position()) in ring_keys}
    clear = _px(renderer, clearance_pt)
    lines: list[tuple[np.ndarray, float]] = [
        (ax.transData.transform(np.column_stack([th, rs])), _px(renderer, lw / 2))
        for th, rs, lw in arcs]
    for art in key_artists:
        if isinstance(art, Annotation):
            art.update_positions(renderer)
            if art.arrow_patch is not None:
                lines += [(np.asarray(path.vertices), hw)
                          for path, hw in _arrow_paths(art.arrow_patch, renderer)]
        elif isinstance(art, Line2D):
            lines.append((ax.transData.transform(art.get_xydata()),
                          _px(renderer, art.get_linewidth() / 2)))
    quads = {id(t): _text_quad(t, renderer) for t in ax.texts
             if t.get_text() and t.get_visible() and id(t) not in ring}

    def score(t: Text, edges: list[np.ndarray]) -> tuple[int, float]:
        q = quads[id(t)]
        gaps = [_quad_gap(q, o) for k, o in quads.items() if k != id(t)]
        gaps += [_polyline_quad_gap(p, q) - hw for p, hw in lines]
        gaps += [_polyline_quad_gap(p, q) for p in edges]
        return sum(g < clear for g in gaps), min(gaps, default=float("inf"))

    managed: list[Text] = []
    for lab in labels:
        t = by_key.get((lab.text, lab.theta, lab.r))
        if t is None or id(t) not in quads:
            continue
        managed.append(t)
        edges = [ax.transData.transform(np.column_stack([th, rs]))
                 for th, rs in lab.edges]
        n, g = score(t, edges)
        if not n:
            continue
        best = (n, -g, t.get_position(), t.get_rotation())
        for th, r, orient in lab.candidates:
            t.set_position((th, r))
            if orient is not None:
                t.set_rotation(get_label_params_by_rad(
                    th, orient, only_rotation=True)["rotation"])
            quads[id(t)] = _text_quad(t, renderer)
            n, g = score(t, edges)
            if (n, -g) < best[:2]:
                best = (n, -g, t.get_position(), t.get_rotation())
            if not n:
                break
        t.set_position(best[2])
        if t.get_rotation() != best[3]:
            t.set_rotation(best[3])
        quads[id(t)] = _text_quad(t, renderer)
    return managed


#: Placements of a covered rights-arc label as (curve parameter, offset along
#: the chord normal in data units): the six of the first pass, then the wider
#: search of 2026-11-07 at 0.55 out, 0.30 inside the bulge and 0.80 out, each
#: from the midpoint outward along the curve. A label inside the bulge sits
#: between its arc and the chord.
_ARC_LABEL_STEPS: tuple[tuple[float, float], ...] = (
    (0.6, 0.30), (0.7, 0.30), (0.4, 0.30), (0.3, 0.30), (0.5, 0.55), (0.5, 0.80),
    *((t_, off) for off in (0.55, -0.30, 0.80)
      for t_ in (0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2)),
)


def _clear_arc_label(
    ax: Axes,
    label: Annotation,
    p0: np.ndarray,
    p1: np.ndarray,
    ctrl: np.ndarray,
    bunit: np.ndarray,
    coords: dict[Bus, tuple[float, float]],
    renderer: RendererBase,
) -> None:
    """Slide a rights-arc label off a bus marker, a line or another text on the panel.

    Display only, run when the axes draws (:func:`add_arc_label_clearance`).
    ``p0`` and ``p1`` are the source and sink in data units, ``ctrl`` the arc3
    curve's control point and ``bunit`` the unit chord normal toward the bulge,
    as ``draw_rights_arcs`` computes them. The label starts on the curve's
    midpoint, 0.30 data units out along ``bunit``. A label whose drawn box
    clears every bus marker and every other text on the axes keeps that
    position, so a figure without a collision is unchanged. A covered label
    tries points on the curve toward the sink (curve parameter 0.6, 0.7), then
    toward the source (0.4, 0.3), each 0.30 out along the normal, then the
    midpoint at 0.55 and 0.80 out; then (2026-11-07) the curve parameters 0.5,
    0.6, 0.4, 0.7, 0.3, 0.8 and 0.2 at 0.55 out, at 0.30 inside the bulge and
    at 0.80 out, in that order. It takes the first that clears; with none clear it takes the
    placement with the fewest overlaps, the default first.

    Since 2026-11-07 the drawn network lines (each ``Line2D`` of the axes), the
    flow arrows and the rights arcs (each ``FancyArrowPatch`` of the axes, the
    label's own included) are obstacles as well: a label box that their path,
    widened by half its line width, crosses is covered.

    A bus marker is the disc of the largest scatter marker drawn at the bus's
    coordinates, its edge included, or a 10-point disc where none is drawn. The
    bus numbers, the chips and the flow labels that ``plot_network_topology``
    placed are texts of the axes and are tested as boxes.
    """
    # Marker radius in points at each bus: a scatter size is the marker's area
    # in points squared, so its radius is sqrt(size) / 2.
    radius_pt = dict.fromkeys(coords, 0.0)
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        offs, sizes = np.asarray(coll.get_offsets()), coll.get_sizes()
        lws = np.atleast_1d(coll.get_linewidth())
        if not len(sizes):
            continue
        for i, xy in enumerate(offs):
            for b, c in coords.items():
                if np.allclose(xy, c):
                    lw = float(lws[i % len(lws)]) if len(lws) else 0.0
                    r_pt = np.sqrt(float(sizes[i % len(sizes)])) / 2 + lw / 2
                    radius_pt[b] = max(radius_pt[b], r_pt)
    discs = [(ax.transData.transform(coords[b]),
              _px(renderer, r if r > 0 else 10.0))
             for b, r in radius_pt.items()]
    boxes = [_drawn_box(t, renderer) for t in ax.texts
             if t is not label and t.get_visible() and t.get_text()]
    paths: list[tuple[Path, float]] = [
        (Path(ax.transData.transform(ln.get_xydata())),
         _px(renderer, ln.get_linewidth() / 2))
        for ln in ax.lines if ln.get_visible()]
    for t in ax.texts:
        if isinstance(t, Annotation) and not t.get_text() and t.arrow_patch is not None:
            t.update_positions(renderer)
            paths += _arrow_paths(t.arrow_patch, renderer)
    for arc in ax.patches:
        if isinstance(arc, FancyArrowPatch):
            paths += _arrow_paths(arc, renderer)

    def overlaps() -> int:
        b = _drawn_box(label, renderer)
        n = sum(b.overlaps(o) for o in boxes)
        n += sum(p.intersects_bbox(b.padded(hw), filled=False) for p, hw in paths)
        return n + sum(_box_hits_disc(b, float(c[0]), float(c[1]), r)
                       for c, r in discs)

    best_n = overlaps()
    if not best_n:
        return
    best = label.xyann
    for t_, off in _ARC_LABEL_STEPS:
        on_curve = (1 - t_) ** 2 * p0 + 2 * t_ * (1 - t_) * ctrl + t_ ** 2 * p1
        label.xyann = tuple(on_curve + off * bunit)
        n = overlaps()
        if n < best_n:
            best, best_n = label.xyann, n
        if not n:
            break
    label.xyann = best


def add_arc_label_clearance(
    ax: Axes,
    label: Annotation,
    p0: np.ndarray,
    p1: np.ndarray,
    ctrl: np.ndarray,
    bunit: np.ndarray,
    coords: dict[Bus, tuple[float, float]],
) -> None:
    """Register the draw-time clearance of a rights-arc label on ``ax``.

    Display only. The clearance runs when the axes draws, against the chips and
    flow labels as they are finally placed, and moves ``label`` as
    :func:`_clear_arc_label` states. ``p0`` and ``p1`` are the arc's source and
    sink in data units, ``ctrl`` its arc3 control point, ``bunit`` the unit
    chord normal toward the bulge and ``coords`` the bus coordinates of the
    panel.
    """
    ax.add_artist(_LabelClearance(
        partial(_clear_arc_label, ax, label, p0, p1, ctrl, bunit, coords),
        [label]))


def _shared_bus_legend_handles(
    all_buses: list[Bus],
    supply_by_bus: dict[Bus, list[dict[str, Any]]] | None,
    demand_by_bus: Loads | None,
    bus_lmps: BusValues | None,
    bus_colors: dict[Bus, str],
) -> list[Patch]:
    """Build one color-keyed legend entry per active bus carrying the bid and
    LMP, so the two side-by-side panels need not repeat that text. Color does
    the matching: the same swatch identifies the bus in both panels."""
    supply_by_bus = supply_by_bus or {}
    demand_by_bus = demand_by_bus or {}
    bus_lmps = bus_lmps or {}
    handles = []
    for bus in all_buses:
        gens = supply_by_bus.get(bus, [])
        load = demand_by_bus.get(bus, 0)
        if not gens and load <= 0:
            continue
        parts = []
        for g in gens:
            cap = g.get('capacity', g.get('volume', 0))
            parts.append(f"{g['unit_id']} ${g['price']:.0f}x{cap:.0f}")
        if load > 0:
            parts.append(f"load {load:.0f} MW")
        lmp = bus_lmps.get(bus)
        lmp_txt = f"  |  LMP ${lmp:.0f}" if lmp is not None else ""
        label = f"Bus {bus}: " + "; ".join(parts) + lmp_txt
        handles.append(Patch(fc=bus_colors.get(bus, TRANSIT_COLOR),
                             ec='#555', alpha=0.8, label=label))
    return handles


def plot_combined_letter(
    network: pypsa.Network,
    supply_by_bus: dict[Bus, list[dict[str, Any]]],
    demand_by_bus: Loads,
    *,
    bus_colors: dict[Bus, str] | None = None,
    bus_lmps: BusValues | None = None,
    bus_net_mw: BusValues | None = None,
    line_flows: LineValues | None = None,
    line_widths: LineValues | None = None,
    line_colors: dict[Line, str] | None = None,
    constrained_lines: Collection[Line] | None = None,
    flows: Sequence[tuple[Bus, Bus, float] | tuple[Bus, Bus, float, str]] | None = None,
    clearing_price: float | None = None,
    gen_marginal_costs: dict[Resource, float] | None = None,
    lmp_line: bool = False,
    bus_groups: dict[Bus, str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_label_fontsize: float | None = None,
    show_group_labels: bool = True,
    annotate_roles: bool = False,
    axis_key: bool = False,
    demand_segments: dict[Bus, list[dict[str, Any]]] | None = None,
    gen_bid_labels: bool = True,
    gen_cost_labels: bool = False,
    block_mw_unit: bool = True,
    show_bus_price: bool = True,
    all_buses: list[Bus] | None = None,
    title_left: str = 'Network — DC power flow',
    title_right: str = 'Nodal dispatch - - merit order, demand, PTDF gen->load',
    suptitle: str | None = None,
    figsize: tuple[float, float] = (11, 6.2),
    label_fontsize: float = 10,
    sector_order: Iterable[Bus] | None = None,
    bus_coords: dict[Bus, tuple[float, float]] | None = None,
    center_bus: Bus | None = None,
    dim_buses: Collection[Bus] | None = None,
    start: float = 0,
    network_show_lmp: bool = True,
    panel_ratios: tuple[float, float] = (1, 1),
    net_mw_zero_label: bool = False,
) -> tuple[Figure, tuple[Axes, PolarAxes]]:
    """Letter-size composite: network topology (left) + circlize/chord (right).

    Designed to drop onto one landscape Letter page with both panels sharing the
    width and a single color-keyed legend (bid size, bid, LMP) along the bottom,
    so nothing has to be repeated on each panel. Node numbers sit beside the
    colored bubbles (not inside), and the circlize sector labels are drawn at
    ``label_fontsize`` (10 by default) so they stay legible at half-page width.

    Parameters mirror ``plot_network_topology`` / ``plot_nodal_circlize``;
    ``bus_colors`` is shared across both panels so the legend matches both.

    Returns
    -------
    fig : matplotlib Figure
    (ax_net, ax_circ) : tuple of Axes
    """
    if all_buses is None:
        all_buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))
    if bus_colors is None:
        bus_colors = assign_bus_colors(all_buses, supply_by_bus, demand_by_bus)

    fig = plt.figure(figsize=figsize)
    _compact = figsize[0] < 10          # the taller 8.5-in-wide print layout
    gs = fig.add_gridspec(1, 2, width_ratios=list(panel_ratios), wspace=0.02,
                          left=0.02, right=0.99, top=(0.895 if _compact else 0.92),
                          bottom=0.03)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_circ: PolarAxes = fig.add_subplot(gs[0, 1], projection='polar')  # type: ignore[assignment]  # stub reads Axes for any projection
    # Pull the circlize axes in slightly so pycirclize's peripheral labels
    # (drawn out to r~103) don't clip against the page edge.
    box = ax_circ.get_position()
    ax_circ.set_position([box.x0 + 0.01, box.y0 + (0.01 if _compact else 0.02),  # type: ignore[arg-type]  # a list unpacks at run time, the stub says tuple
                          box.width * (0.95 if _compact else 0.90),
                          box.height * (0.96 if _compact else 0.90)])

    # Left: network topology -- numbers outside the bubble, color-matched boxes.
    plot_network_topology(
        network, supply_by_bus, demand_by_bus,
        bus_colors=bus_colors, bus_lmps=(bus_lmps if network_show_lmp else None),
        bus_net_mw=bus_net_mw, bus_coords=bus_coords,
        line_flows=line_flows, line_widths=line_widths, line_colors=line_colors,
        constrained_lines=constrained_lines,
        number_position='outside', box_node_header=False, lmp_only=True,
        node_number_fontsize=11, annot_fontsize=7.5, title_fontsize=11,
        ax=ax_net, title=title_left, net_mw_zero_label=net_mw_zero_label,
    )
    if _compact:               # lift the panel title into the margin, clear of the top net-mw box
        ax_net.set_title(title_left, fontsize=11, fontweight='bold', pad=20)

    # Right: circlize/chord drawn into the polar axes, compact labels at fs=10.
    plot_nodal_circlize(
        supply_by_bus, demand_by_bus, all_buses,
        flows=flows, clearing_price=clearing_price, bus_lmps=bus_lmps,
        dim_buses=dim_buses,
        bus_colors=bus_colors, gen_marginal_costs=gen_marginal_costs,
        lmp_line=lmp_line, bus_groups=bus_groups, group_colors=group_colors,
        group_label_fontsize=group_label_fontsize, show_group_labels=show_group_labels,
        annotate_roles=annotate_roles, axis_key=axis_key,
        demand_segments=demand_segments,
        gen_bid_labels=gen_bid_labels, gen_cost_labels=gen_cost_labels,
        block_mw_unit=block_mw_unit, show_bus_price=show_bus_price,
        ax=ax_circ, label_fontsize=label_fontsize, compact=True,
        show_legend=False, sector_order=sector_order,
        start=start, center_bus=center_bus,
    )
    ax_circ.set_title(title_right, fontsize=11, fontweight='bold', pad=(20 if _compact else 10))

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=(0.99 if _compact else 0.985))

    return fig, (ax_net, ax_circ)
