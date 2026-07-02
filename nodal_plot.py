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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba, to_hex
from pycirclize import Circos


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


def _darken(color, factor=0.7):
    """Darken a color for borders/edges."""
    r, g, b, a = to_rgba(color)
    return to_hex((r * factor, g * factor, b * factor, a))


def assign_bus_colors(all_buses, supply_by_bus=None, demand_by_bus=None):
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


def compute_proportional_flows(supply_by_bus, demand_by_bus):
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


def extract_line_flows(flows_dict, grid_data, hour_idx):
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


def trace_gen_to_load(line_flows, gen_by_bus, load_by_bus):
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


def compute_ptdf_flows(network, supply_by_bus, demand_by_bus):
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
    incoming = {b: [] for b in buses}
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
    gen_at_bus = {}
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


def prepare_bus_data(orders_hour, network):
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
    supply_by_bus,
    demand_by_bus,
    all_buses,
    flows=None,
    clearing_price=None,
    bus_lmps=None,
    title=None,
    min_sector=40,
    bus_colors=None,
    figsize=None,
    price_cap=200,
    gen_marginal_costs=None,
    ax=None,
    label_fontsize=9,
    compact=False,
    show_legend=True,
    sector_order=None,
    start=0,
    center_bus=None,
    track_fontsize=None,
    lmp_line=False,
    bus_groups=None,
    group_colors=None,
    group_label_fontsize=None,
    show_group_labels=True,
    annotate_roles=False,
    axis_key=False,
    demand_segments=None,
    gen_bid_labels=True,
    gen_cost_labels=False,
    block_mw_unit=True,
):
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
        outer band is drawn around each sector in its group's colour and each
        contiguous run of a group is labelled (e.g. "BA-1", "BA-2") — so a
        multi-BA / multi-market ring shows which operator owns which quadrant.
    group_colors : dict, optional
        {group_label: color_hex}. Colour for each group's outer band/label.
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
        layout — a radial arrow up the bar's left edge labelled "price" and an arc
        bent along the track base labelled "volume" — showing that a bar's radial
        height encodes price and its tangential width encodes volume. Direction
        only (not a measured scale). Anchored to the first bar clockwise from the
        top (skipping the demand-labelled bar), chosen dynamically so it does not
        depend on any particular bus name/number. A legend-on-the-figure so the inner ring and chords
        explain themselves once.

    Returns
    -------
    fig : matplotlib Figure
    """
    # Colours are assigned against the ORIGINAL bus order so they stay tied to
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
            bus_sizes[bus] = transit_sector
        else:
            bus_sizes[bus] = max(content, min_sector)

    # --- Build Circos ---
    # `start` rotates the whole ring (degrees clockwise from 12 o'clock). If
    # `center_bus` is given, compute the start offset that lands that bus's
    # sector centre exactly at 12 o'clock.
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

    bus_gen_range = {}   # (x_start, x_end) of gen region
    bus_load_range = {}  # (x_start, x_end) of load region

    # In-track label sizes. track_fontsize, when given, overrides the size of the
    # bid, dispatched-MW and load-MW labels drawn inside the sectors (e.g. to hold
    # a >=10 pt floor for a page-width standalone figure). Otherwise the compact
    # flag drives them (smaller still in the dense side-by-side composite).
    _fs_gen = track_fontsize if track_fontsize is not None else (7 if compact else 5)
    _fs_load = track_fontsize if track_fontsize is not None else (7 if compact else 6)

    # Group (BA / market) banding setup. Only active when bus_groups names more
    # than one distinct group; otherwise the ring is single-operator and the
    # outer band is skipped (keeping the default single-market look unchanged).
    _draw_groups = bool(bus_groups) and len(set(bus_groups.values())) > 1
    _group_colors = dict(group_colors or {})
    if _draw_groups:
        _gpal = 0
        for g in dict.fromkeys(bus_groups.values()):   # stable first-seen order
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

        def _price_to_r(price):
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
            # Opt-in per-unit colour override (e.g. a grey self-schedule block
            # inside an otherwise bus-coloured stack); default = bus colour.
            gcol = gen.get('color', bc)
            gcol_dark = _darken(gcol) if 'color' in gen else bc_dark

            # Clamp coordinates to sector bounds to avoid pycirclize ValueError
            x_end_nameplate = min(x_pos + nameplate, size)
            x_end_bid = min(x_pos + bid_vol, size)
            x_end_acc = min(x_pos + accepted, size)

            # A unit carrying a linear marginal-cost curve (``mc0`` + ``mc_slope``·g)
            # is drawn as a sloped WEDGE under its MC line rather than a flat bar:
            # the bar top rises with output, so the rising offer is visible and the
            # gap to the dashed LMP line is the per-MW inframarginal rent that varies
            # along the curve. A flat unit keeps the original three-level rectangle.
            has_curve = 'mc_slope' in gen
            if has_curve:
                _mc0, _slope = gen['mc0'], gen['mc_slope']
                bar_top = _price_to_r(min(_mc0 + _slope * nameplate, norm_price))

                def _wedge(xa, xb, **kw):
                    if xb <= xa + 0.01:
                        return
                    xs = np.linspace(xa, xb, 16)
                    vs = np.clip(_mc0 + _slope * (xs - x_pos), 0.5, norm_price)
                    track.fill_between(xs.tolist(), vs.tolist(), 0,
                                       vmin=0, vmax=norm_price, **kw)

                _wedge(x_pos, x_end_nameplate, fc=gcol, alpha=0.08, ec=gcol, lw=0.5, ls='--')
                _wedge(x_pos, x_end_bid, fc=gcol, alpha=0.25, ec=gcol, lw=0.5)
                _wedge(x_pos, x_end_acc, fc=gcol, alpha=0.7, ec=gcol_dark, lw=0.8)
            else:
                bar_top = _price_to_r(price)

                # Nameplate capacity (very faded — withheld capacity)
                if x_end_nameplate > x_pos + 0.01:
                    track.rect(x_pos, x_end_nameplate, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.08, ec=gcol, lw=0.5, ls='--')

                # Bid volume (faded — offered but not accepted)
                if bid_vol > 0 and x_end_bid > x_pos + 0.01:
                    track.rect(x_pos, x_end_bid, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.25, ec=gcol, lw=0.5)

                # Accepted dispatch (solid bus color)
                if accepted > 0 and x_end_acc > x_pos + 0.01:
                    track.rect(x_pos, x_end_acc, r_lim=(55, bar_top),
                               fc=gcol, alpha=0.7, ec=gcol_dark, lw=0.8)

            # Label: the BID (price × volume) on each gen bar, mirroring the
            # load side which prints its MW in-track. (Unit name dropped.) Opt out
            # via gen_bid_labels=False when the bar height is a cost the caller would
            # rather not annotate (e.g. the bilateral book, which labels the cleared
            # price on the dotted line instead -- see gen_cost_labels below).
            if gen_bid_labels and not has_curve:
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

            # Dispatched MW printed INSIDE the gen track (mirrors the load side,
            # which prints its MW in-track), while the bid stays outside the bar.
            # Radius = midpoint of the FILLED bar (55 -> bar_top) so the label
            # always sits inside the bar regardless of its height; a fixed floor
            # would push a short bar's label out on top of the track.
            if accepted > 0.5:
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

        # LOAD side (right): demand bar, height = bus LMP. An opt-in
        # demand_segments entry splits the SAME total into consecutive
        # segments, each with its own height/colour (e.g. the portion of a
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
                    # Per-segment alpha (default = the demand fill 0.35). A fainter
                    # alpha draws e.g. shed/unserved load in the bus colour, the
                    # way idle generation capacity is drawn faint on the supply side.
                    track.rect(x0, x1, r_lim=(55, seg_top),
                               fc=scol, alpha=float(seg.get('alpha', 0.35)),
                               ec=scol_dark, lw=0.8)
                    if mw > 0.5:
                        track.text(f"{mw:.0f}{' MW' if block_mw_unit else ''}", x=(x0 + x1) / 2,
                                   r=(55 + seg_top) / 2,
                                   fontsize=_fs_load, color=scol_dark,
                                   fontweight='bold')
                    x0 = x1
                # A split load bar no longer encodes the bus LMP by its
                # height, so draw the dashed LMP line across the load region
                # (mirrors the gen-region line) to keep the price visible.
                if lmp_line and bus_lmp > 0 and load_end - load_start > 1.0:
                    track.line([load_start + 0.5, load_end - 0.5],
                               [min(bus_lmp, norm_price)] * 2,
                               vmin=0, vmax=norm_price,
                               color='#17202A', lw=1.6, ls=(0, (4, 2)), zorder=12)
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
                # Price label on load bar (suppressed in compact mode and whenever a
                # track_fontsize floor is set — it is a tiny, duplicate of the bus LMP
                # already shown in the outer ring label and the shared key).
                if not compact and track_fontsize is None:
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
        # group's colour band instead of the LMP ghost (the LMP is already the
        # load-bar height and, with lmp_line, a dashed line on the gen bars).
        if _draw_groups and bus in bus_groups:
            gcol = _group_colors.get(bus_groups[bus], TRANSIT_COLOR)
            shadow_track.rect(0, size, r_lim=(97, 100),
                              fc=gcol, alpha=0.55, ec=_darken(gcol), lw=0.5)
        elif bus_lmp > 0 and (dem > 0 or gw > 0):
            # Shadow price bar (ghost bar outside main track): bus LMP outline.
            shadow_r = 97 + min(bus_lmp / norm_price, 1.0) * 3
            region_start = 0
            region_end = gw + dem + (gap if gw > 0 and dem > 0 else 0)
            if region_end > 0:
                shadow_track.rect(region_start, min(region_end, size),
                                  r_lim=(97, shadow_r),
                                  fc=bc, alpha=0.15, ec=bc, lw=0.3)

        # --- Inner ring: a "send" sub-bar under the generator and a "receive"
        #     sub-bar under the load, in the same bus colour, so a bus with
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
        # the bus number inside a circle (a parenthesised "(N)" fallback if the
        # circle bbox can't be drawn). Active buses keep "Bus N" + their LMP.
        bus_lmp = _lmps.get(bus, 0)
        is_transit = (gw <= 0) and (dem <= 0)
        if is_transit:
            try:
                sector.text(f"{bus}", r=103, fontsize=label_fontsize,
                            fontweight='bold', color=bc_dark,
                            bbox=dict(boxstyle='circle,pad=0.3', fc='white',
                                      ec=bc_dark, lw=1.2))
            except Exception:
                sector.text(f"({bus})", r=103, fontsize=label_fontsize,
                            fontweight='bold', color=bc_dark)
        elif bus_lmp > 0:
            sector.text(f"Bus {bus}\n${bus_lmp:.1f}/MWh", r=103,
                        fontsize=label_fontsize, fontweight='bold', color=bc_dark)
        else:
            sector.text(f"Bus {bus}", r=103, fontsize=label_fontsize,
                        fontweight='bold', color=bc_dark)

    # --- Flow chords: gen→load, anchored to gen dispatch and load demand ---
    # Source side: chord occupies fraction of gen region = alloc_MW / gen_dispatch_MW
    # Dest side: chord occupies fraction of load region = alloc_MW / load_demand_MW
    gen_cursor = {b: r[0] for b, r in bus_gen_range.items()}
    load_cursor = {b: r[0] for b, r in bus_load_range.items()}

    # Accepted dispatch per bus (for chord proportional sizing)
    bus_dispatch = {}
    for bus in all_buses:
        bus_dispatch[bus] = sum(g['accepted_volume'] for g in supply_by_bus.get(bus, []))

    if flows:
        sorted_flows = sorted(flows, key=lambda f: -abs(f[2]))

        for src, dst, flow_mw in sorted_flows:
            if abs(flow_mw) < 0.5:
                continue
            src_name = f"Bus {src}"
            dst_name = f"Bus {dst}"

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
    # Placed just outside the bus labels at the centre degree of each run so the
    # multi-operator ring reads "this arc is BA-1, that arc is BA-2". Font held at
    # label_fontsize (group_label_fontsize to override) so it stays readable.
    if _draw_groups and show_group_labels:
        _glf = group_label_fontsize if group_label_fontsize is not None else label_fontsize
        runs = []  # (group, [buses in this contiguous run])
        for b in all_buses:
            g = bus_groups.get(b)
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
    # Styled like the in-bar MW numbers (small, bold, the bus's own dark colour,
    # no box) and tucked INTO the relevant element: "supply" on the inner
    # send-bar, "demand" on the inner receive-bar, "dispatch" on a chord — each on
    # the element closest to the top, so the diagram explains itself once.
    if annotate_roles:
        import math as _math

        def _ang_dist(d):
            d %= 360
            return min(d, 360 - d)

        _centers = {}
        for _b in all_buses:
            _d0, _d1 = circos.get_sector(f"Bus {_b}").deg_lim
            _centers[_b] = ((_d0 + _d1) / 2) % 360

        _hfs = max(4.5, _fs_load - 1)   # a touch smaller than the MW labels

        def _hlabel(text, bus, xc, r, color):
            sec = circos.get_sector(f"Bus {bus}")
            deg = _math.degrees(sec.x_to_rad(xc)) % 360
            circos.text(text, r=r, deg=deg, adjust_rotation=False,
                        fontsize=_hfs, fontweight='bold', color=color)

        # supply: dispatched generator nearest the top — on its inner send-bar
        _gen = [b for b in all_buses
                if bus_dispatch.get(b, 0) > 0.5 and bus_gen_range.get(b, (0, 0))[1] > 0.5]
        if _gen:
            gb = min(_gen, key=lambda b: _ang_dist(_centers[b]))
            _hlabel('supply', gb,
                    min(bus_dispatch.get(gb, 0), bus_gen_range[gb][1]) / 2,
                    50.5, _darken(bus_colors[gb], 0.55))
        # demand: load nearest the top — on its inner receive-bar
        _load = [b for b in all_buses if demand_by_bus.get(b, 0) > 0.5
                 and bus_load_range.get(b, (0, 0))[1] > bus_load_range.get(b, (0, 0))[0]]
        if _load:
            lb = min(_load, key=lambda b: _ang_dist(_centers[b]))
            ls_, le_ = bus_load_range[lb]
            _hlabel('demand', lb, (ls_ + le_) / 2, 50.5,
                    _darken(bus_colors[lb], 0.55))
        # dispatch: a flow chord near the top — small neutral label in the centre
        if flows and _load:
            _sec = circos.get_sector(f"Bus {lb}")
            _deg = _math.degrees(_sec.x_to_rad((ls_ + le_) / 2)) % 360
            circos.text('dispatch', r=26, deg=_deg, adjust_rotation=False,
                        fontsize=_hfs, fontweight='bold', color='#5D6D7E')

    # --- Render ---
    # Draw into a caller-supplied PolarAxes (for side-by-side composite figures)
    # or create a standalone figure.
    if ax is not None:
        circos.plotfig(ax=ax)
        fig = ax.get_figure()
    else:
        fig = circos.plotfig(figsize=figsize or (10, 10))

    # Illustrative axis following the polar layout: a radial arrow up a bar's
    # left edge = "price", and an arc bent along the track base = "volume".
    # Direction only (not a measured scale). Anchored to the FIRST bus clockwise
    # from the top that has a bar, skipping the demand-labelled bus — so it never
    # crowds "demand" and is picked dynamically (no reliance on any bus number).
    if axis_key:
        _pax = ax if ax is not None else fig.axes[0]

        def _ad(d):
            d %= 360
            return min(d, 360 - d)

        _cn = {}
        for _b in all_buses:
            _d0, _d1 = circos.get_sector(f"Bus {_b}").deg_lim
            _cn[_b] = ((_d0 + _d1) / 2) % 360

        def _has_bar(b):
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
            _pax.annotate('', xy=(_radL, 90), xytext=(_radL, _r0),
                          arrowprops=_arr, annotation_clip=False, zorder=21)
            _pax.text(_radL, 93, 'price', ha='center', va='center',
                      fontsize=_afs, fontweight='bold', color=_col,
                      clip_on=False, zorder=21)
            # volume: arc along the gap, in the increasing-volume direction
            _xr = _x0 + 0.6 + min((_x1 - _x0) - 1.2, max((_x1 - _x0) * 0.7, 6))
            _ths = np.linspace(_sec.x_to_rad(_x0 + 0.6), _sec.x_to_rad(_xr), 24)
            _pax.plot(_ths, [_r0] * 24, color=_col, lw=1.4, zorder=21,
                      clip_on=False, solid_capstyle='round')
            _pax.annotate('', xy=(_ths[-1], _r0), xytext=(_ths[-3], _r0),
                          arrowprops=_arr, annotation_clip=False, zorder=21)
            _pax.text(_sec.x_to_rad(_xr), _r0 - 3.0, 'volume', ha='center',
                      va='center', fontsize=_afs, fontweight='bold', color=_col,
                      clip_on=False, zorder=21)

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


def plot_nodal_from_results(orders, meta, network, time=None,
                            flows='proportional', bus_colors=None,
                            bus_lmps=None):
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


def rotate_coords(coords, deg):
    """Rotate a {bus: (x, y)} layout about the origin by ``deg`` degrees
    (counter-clockwise). 180 flips top↔bottom; the rigid rotation keeps the
    no-crossing IEEE9 layout intact. Returns a new dict (does not mutate)."""
    import math
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return {b: (x * c - y * s, x * s + y * c) for b, (x, y) in coords.items()}


def _compute_dc_line_flows(network, supply_by_bus, demand_by_bus):
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
    network,
    supply_by_bus=None,
    demand_by_bus=None,
    bus_colors=None,
    bus_coords=None,
    bus_lmps=None,
    bus_net_mw=None,
    demand_served_by_bus=None,
    line_flows=None,
    line_widths=None,
    line_colors=None,
    constrained_lines=None,
    title='IEEE 9-Bus Network',
    ax=None,
    figsize=(8, 7),
    number_position='inside',
    box_node_header=False,
    node_number_fontsize=10,
    annot_fontsize=6,
    title_fontsize=13,
    lmp_only=False,
    flow_label_fontsize=9,
    flow_labels=True,
    parallel_gap=0.12,
    net_label_offset=28.0,
    title_pad=None,
):
    """
    Draw the network topology diagram with bus colors matching the circlize plot.

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
        and colour still flags congestion (red). When None, the legacy behaviour
        applies (lw=2, lw=3 for constrained lines).
    line_colors : dict, optional
        {line_name: color_hex}. The "ownership" colour of a line — e.g. the
        balancing authority / market that manages it. Used as the line's base
        colour so the diagram reads which operator monitors which corridor. A
        line absent from the dict stays grey (e.g. a tie that no single operator
        owns). Congestion (``constrained_lines``) still overrides to red on top
        of this — a congested line is red regardless of its owner.
    constrained_lines : set or list of str, optional
        The set of genuinely constrained lines — those whose congestion shadow
        price μ is non-zero (a binding transmission limit), or physically
        overloaded lines. A line is drawn red **iff** it is in this set. This is
        the only thing that triggers the constrained (red) treatment: a line that
        merely reaches its rating because the generator behind it is maxed out
        has μ = 0 and stays grey. When None (or a line is absent from the set),
        the line is not constrained. Flow magnitude never drives the colour.
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
        fig = ax.get_figure()

    # --- Line flows: use solver flows if provided, otherwise compute DC ---
    if line_flows is None:
        line_flows = _compute_dc_line_flows(network, supply_by_bus, demand_by_bus)

    # --- Draw lines with flow arrows ---
    lines = network.lines[['bus0', 'bus1', 's_nom']]
    # Parallel circuits (same bus pair) get a small perpendicular offset so they
    # render side-by-side rather than overplotting. Single lines are unaffected.
    _groups = {}
    for _ln, _lr in lines.iterrows():
        _groups.setdefault(frozenset((_lr['bus0'], _lr['bus1'])), []).append(_ln)
    _par_offset = {}
    for _grp in _groups.values():
        _k = len(_grp)
        for _j, _ln in enumerate(sorted(_grp)):
            _par_offset[_ln] = (_j - (_k - 1) / 2.0) * parallel_gap
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
        # generator behind it is maxed out has zero shadow price and stays grey.
        # A line is red iff it is in the caller-supplied set; with no set, none.
        constrained = (constrained_lines is not None
                       and line_name in constrained_lines)
        # Colour priority: congestion red > owner (BA/market) colour > grey.
        # A congested line is always red; otherwise it takes its owner's colour
        # if one was supplied, and falls back to grey (e.g. an unowned tie).
        if constrained:
            line_color = '#E74C3C'
        elif line_colors is not None and line_name in line_colors:
            line_color = line_colors[line_name]
        else:
            line_color = '#AAB7B8'
        if line_widths is not None:
            # Width encodes slipperiness (susceptance); colour still flags congestion.
            line_lw = line_widths.get(line_name, 2)
        else:
            line_lw = 3 if constrained else 2

        ax.plot([x0, x1], [y0, y1], color=line_color, lw=line_lw, zorder=1,
                solid_capstyle='round')

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
            ax.annotate('', xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                        arrowprops=dict(arrowstyle='-|>', color=line_color,
                                        lw=max(1.0, 0.9 * line_lw),
                                        mutation_scale=8 + 2.4 * line_lw),
                        zorder=2)

        # Flow label: "flow/capacity" offset perpendicular to line (opt-out via
        # flow_labels=False). Skip lines carrying ~no flow so dead paths don't print
        # a "0/limit" tag; a constrained line is always labelled.
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
            ax.text(mx + perp_x, my + perp_y, flow_label,
                    fontsize=flow_label_fontsize, color=label_color,
                    fontweight=fontweight, ha='center', va='center',
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

    # --- Draw buses ---
    # Chart centre (mean of the drawn buses) -- used to push per-bus labels
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

        # Bus number label. 'inside' = white number in the bubble (default);
        # 'outside' = dark number on a small chip beside the bubble, so the
        # coloured marker stays clean for print at half-page width.
        if number_position == 'outside':
            ax.annotate(bus, (x, y), fontsize=node_number_fontsize,
                        fontweight='bold', color=bc_dark, zorder=6,
                        ha='center', va='center',
                        xytext=(-13, 11), textcoords='offset points',
                        bbox=dict(boxstyle='circle,pad=0.18', fc='white',
                                  ec=bc_dark, lw=1.2, alpha=0.95))
        else:
            ax.text(x, y, bus, fontsize=node_number_fontsize, fontweight='bold',
                    ha='center', va='center', color='white', zorder=4)

        # lmp_only: skip the gen/bid/load info box entirely (that detail now lives
        # in the circlize panel) and print only a per-bus annotation, no info box.
        # With bus_net_mw given, label the NET INJECTION (dispatched generation +,
        # load -) in a green/maroon chip -- for a flow-focused panel where price is
        # better read off the dispatch ring; otherwise fall back to the LMP.
        if lmp_only:
            # radial push-out: offset the chip away from the chart centre, and anchor
            # the text on the side facing the node so it reads outward.
            rdx, rdy = x - _cx, y - _cy
            rl = (rdx * rdx + rdy * rdy) ** 0.5 or 1.0
            ox, oy = net_label_offset * rdx / rl, net_label_offset * rdy / rl
            _ha = 'left' if ox > 3 else ('right' if ox < -3 else 'center')
            _va = 'bottom' if oy > 3 else ('top' if oy < -3 else 'center')
            if bus_net_mw is not None and abs(bus_net_mw.get(bus, 0.0)) > 0.5:
                mw = bus_net_mw[bus]
                txt = f"+{mw:.0f} MW" if mw > 0 else f"{mw:.0f} MW"
                if bus_lmps and bus in bus_lmps:        # price alongside the injection
                    txt += f"\n${bus_lmps[bus]:.1f}"
                col = '#1E8449' if mw > 0 else '#922B21'
                ax.annotate(txt, (x, y), fontsize=annot_fontsize + 2, color=col,
                            fontweight='bold', ha=_ha, va=_va,
                            xytext=(ox, oy), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec=col, lw=1.2, alpha=0.92),
                            arrowprops=dict(arrowstyle='-', color=col, lw=0.7,
                                            alpha=0.5, shrinkA=2, shrinkB=3), zorder=5)
            elif bus_lmps and bus in bus_lmps:
                ax.annotate(f"${bus_lmps[bus]:.1f}", (x, y),
                            fontsize=annot_fontsize + 2, color=bc_dark,
                            fontweight='bold', ha=_ha, va=_va,
                            xytext=(ox, oy), textcoords='offset points', zorder=5)
            continue

        # Build annotation text. Optional colour-matched node header so the info
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
        # the wires, not just its price. Only when bus_net_mw is supplied.
        if bus_net_mw is not None and abs(bus_net_mw.get(bus, 0.0)) > 0.5:
            _nmw = bus_net_mw[bus]
            annotations.append(f"Net: +{_nmw:.0f} MW" if _nmw > 0 else f"Net: {_nmw:.0f} MW")
        if bus_lmps and bus in bus_lmps:
            annotations.append(f"LMP: ${bus_lmps[bus]:.1f}")

        if annotations:
            label = '\n'.join(annotations)
            ax.annotate(label, (x, y), fontsize=annot_fontsize, color=bc_dark,
                        fontweight='bold',
                        xytext=(12, -12), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec=bc, lw=1.4, alpha=0.9),
                        zorder=5)

    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_pad)
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.axis('off')

    return fig, ax


def _shared_bus_legend_handles(all_buses, supply_by_bus, demand_by_bus,
                               bus_lmps, bus_colors):
    """Build one colour-keyed legend entry per active bus carrying the bid and
    LMP, so the two side-by-side panels need not repeat that text. Colour does
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
    network,
    supply_by_bus,
    demand_by_bus,
    *,
    bus_colors=None,
    bus_lmps=None,
    bus_net_mw=None,
    line_flows=None,
    line_widths=None,
    line_colors=None,
    constrained_lines=None,
    flows=None,
    clearing_price=None,
    gen_marginal_costs=None,
    lmp_line=False,
    bus_groups=None,
    group_colors=None,
    group_label_fontsize=None,
    show_group_labels=True,
    annotate_roles=False,
    axis_key=False,
    demand_segments=None,
    gen_bid_labels=True,
    gen_cost_labels=False,
    block_mw_unit=True,
    all_buses=None,
    title_left='Network — DC power flow',
    title_right='Nodal dispatch - - merit order, demand, PTDF gen->load',
    suptitle=None,
    figsize=(11, 6.2),
    label_fontsize=10,
    sector_order=None,
    bus_coords=None,
    center_bus=None,
    start=0,
    network_show_lmp=True,
    panel_ratios=(1, 1),
):
    """Letter-size composite: network topology (left) + circlize/chord (right).

    Designed to drop onto one landscape Letter page with both panels sharing the
    width and a single colour-keyed legend (bid size, bid, LMP) along the bottom,
    so nothing has to be repeated on each panel. Node numbers sit beside the
    coloured bubbles (not inside), and the circlize sector labels are drawn at
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
    ax_circ = fig.add_subplot(gs[0, 1], projection='polar')
    # Pull the circlize axes in slightly so pycirclize's peripheral labels
    # (drawn out to r~103) don't clip against the page edge.
    box = ax_circ.get_position()
    ax_circ.set_position([box.x0 + 0.01, box.y0 + (0.01 if _compact else 0.02),
                          box.width * (0.95 if _compact else 0.90),
                          box.height * (0.96 if _compact else 0.90)])

    # Left: network topology -- numbers outside the bubble, colour-matched boxes.
    plot_network_topology(
        network, supply_by_bus, demand_by_bus,
        bus_colors=bus_colors, bus_lmps=(bus_lmps if network_show_lmp else None),
        bus_net_mw=bus_net_mw, bus_coords=bus_coords,
        line_flows=line_flows, line_widths=line_widths, line_colors=line_colors,
        constrained_lines=constrained_lines,
        number_position='outside', box_node_header=False, lmp_only=True,
        node_number_fontsize=11, annot_fontsize=7.5, title_fontsize=11,
        ax=ax_net, title=title_left,
    )
    if _compact:               # lift the panel title into the margin, clear of the top net-mw box
        ax_net.set_title(title_left, fontsize=11, fontweight='bold', pad=20)

    # Right: circlize/chord drawn into the polar axes, compact labels at fs=10.
    plot_nodal_circlize(
        supply_by_bus, demand_by_bus, all_buses,
        flows=flows, clearing_price=clearing_price, bus_lmps=bus_lmps,
        bus_colors=bus_colors, gen_marginal_costs=gen_marginal_costs,
        lmp_line=lmp_line, bus_groups=bus_groups, group_colors=group_colors,
        group_label_fontsize=group_label_fontsize, show_group_labels=show_group_labels,
        annotate_roles=annotate_roles, axis_key=axis_key,
        demand_segments=demand_segments,
        gen_bid_labels=gen_bid_labels, gen_cost_labels=gen_cost_labels,
        block_mw_unit=block_mw_unit,
        ax=ax_circ, label_fontsize=label_fontsize, compact=True,
        show_legend=False, sector_order=sector_order,
        start=start, center_bus=center_bus,
    )
    ax_circ.set_title(title_right, fontsize=11, fontweight='bold', pad=(20 if _compact else 10))

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=(0.99 if _compact else 0.985))

    return fig, (ax_net, ax_circ)
