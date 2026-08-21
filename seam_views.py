# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Presentation frames for the seam notebooks (204, 204a).

Every exhibit is drawn the same way: one market's chair (``own_view``), the
forecast book alone (``forecast_view``), then the superposition as the
network + nodal dispatch letter and as the standalone net-injection network
(``combined_views``). ``export_pair`` writes the settlement and line CSVs and
``save_clean`` writes the deck PNG copies (suptitle, panel titles and legends
removed, transparent background, one canvas size per graph type).

The notebook passes its context once through :func:`configure` — the
footprint partition ``fp``, the exhibit fleet and loads, and the export
knobs — and the frame functions keep the signatures the notebooks call.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

import nodal_plot
import wscc9_figures as figs
from nodal_plot import plot_combined_letter
from seams_engine import physical_flows, susceptance_widths, to_supply_demand
from wscc9_model import BUS_COLORS, CENTER_BUS, COORDS, RING_ORDER

#: The grayed-equals-forecast cue shared by every frame.
GRAY = "#A6ACAF"

#: Module context, set by :func:`configure` from the notebook's EDIT cell.
CFG = {
    "fp": None,                 # Footprints partition (fp_of, colors, line_colors)
    "gen_fleet": None,          # the exhibit fleet {gen: {bus, cost, p_nom}}
    "loads": None,              # {bus: MW}
    "save_enabled": False,      # file writes off until the notebook opts in
    "export_dir": "presentation_exhibits",
    "export_dpi": 600,
    "export_size": (13.33, 7.5),
    "net_export_size": (7.5, 7.5),
    "panel_export_size": (6.67, 7.5),
}


def configure(**kw):
    """Set the module context. Unknown keys are rejected."""
    for k in kw:
        if k not in CFG:
            raise KeyError(f"seam_views.configure: unknown key {k!r}")
    CFG.update(kw)


def _sup_dem(pairs):
    sup, dem = {}, {}
    for _e, _r in pairs:
        _s, _d = to_supply_demand(_e, _r)
        sup.update(_s)
        dem.update(_d)
    return sup, dem


def _net_mw(pt_, *results):
    return {b: float(sum(r.injection[pt_.bus_idx[b]] for r in results)) for b in pt_.buses}


def own_view(net_, pt_, fp_, side, eng, res, proxy=None, binding=None,
             suptitle='', save=None):
    """One market's chair. With ``proxy=(engine, result)`` the neighbor is drawn
    grayed as that market's FORECAST of it, its net injections annotated at the
    neighbor's buses, and the network panel carries the studied superposition."""
    if proxy is None:
        fig = figs.footprint_figure(net_, pt_, fp_, eng, res, highlight=side,
                                    node_net_mw=True,
                                    constrained_lines=binding, suptitle=suptitle)
    else:
        p_eng, p_res = proxy
        sup, dem = _sup_dem([(eng, res), (p_eng, p_res)])
        act = list(eng.activated_lines) if eng.activated_lines != 'all' else list(pt_.lines)
        flagged = (binding if binding is not None else
                   [l for l in act if abs(res.line_dual.get(l, 0.0)) > 1e-3])
        fig = figs.footprint_figure(
            net_, pt_, fp_, eng, res, highlight=side,
            sup_dem=(sup, dem), shed=False,
            node_net_mw=_net_mw(pt_, res, p_res),
            line_flows={l: res.flow_own[l] + p_res.flow_own[l] for l in pt_.lines},
            constrained_lines=flagged,
            bus_lmps={b: float((res if fp_.fp_of(b) == side else p_res).lmp[b])
                      for b in pt_.buses},
            title_left=f'Network -- own book + forecast of '
                       f'{"B" if side == "Market A" else "A"} (grayed = forecast)',
            suptitle=suptitle)
    plt.show()
    if save:
        save_clean(fig, save)


def combined_views(net_, pt_, fp_, pairs, label, binding=None, note=None,
                   save=None, network_panel=False):
    """The superposition, twice: the combined letter, then the standalone
    network carrying only net injections/withdrawals and the flows."""
    ress = [r for _, r in pairs]
    sup, dem = _sup_dem(pairs)
    pf = physical_flows(pt_, ress).rename(columns={'F_phys': 'F^s'})
    phys = {l: float(pf.loc[l, 'F^s']) for l in pt_.lines}
    red = set(pf.index[pf['overload']]) if binding is None else set(binding)
    host = {b: float(next(r for e, r in pairs if fp_.fp_of(b) == e.name).lmp[b])
            for b in pt_.buses}
    netmw = _net_mw(pt_, *ress)
    fig_cl, _ = plot_combined_letter(
        net_, sup, dem,
        bus_colors=dict(BUS_COLORS), bus_lmps=host, bus_net_mw=netmw,
        network_show_lmp=False, line_flows=phys,
        line_widths=susceptance_widths(pt_), line_colors=fp_.line_colors(pt_),
        constrained_lines=red, flows=nodal_plot.compute_ptdf_flows(net_, sup, dem),
        lmp_line=True, bus_groups={b: fp_.fp_of(b) for b in pt_.buses},
        group_colors=dict(fp_.colors), show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER, bus_coords=COORDS,
        center_bus=CENTER_BUS, annotate_roles=True, axis_key=True,
        title_left=note or 'Network -- superposed flow (red = overload)',
        suptitle=f'{label} -- superposed flow  F^s = F[Market A] + F[Market B]'
                 f'   (net injections on the network; prices on the ring)')
    plt.show()
    fig_nt, _ax_nt = nodal_plot.plot_network_topology(
        net_, bus_colors={b: fp_.colors[fp_.fp_of(b)] for b in pt_.buses},
        bus_net_mw=netmw, bus_coords=COORDS, line_flows=phys,
        line_widths=susceptance_widths(pt_), line_colors=fp_.line_colors(pt_),
        constrained_lines=red,
        title=f'{label} -- combined net injections and superposed flows')
    plt.show()
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
        plt.show()
        save_clean(fig_np, f'{save}_network_panel', size=CFG['panel_export_size'])
    return pf


def forecast_view(net_, pt_, fp_, fc_eng, fc_res, label, save=None):
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
        lmp_line=False, bus_groups={b: fp_.fp_of(b) for b in pt_.buses},
        group_colors=grp_cols, show_group_labels=False,
        all_buses=pt_.buses, sector_order=RING_ORDER, bus_coords=COORDS,
        center_bus=CENTER_BUS, annotate_roles=True, axis_key=True,
        title_left=f"Network -- {fc_eng.name}'s forecast flow contributions",
        title_right=f"Nodal -- {fc_eng.name}'s forecast book; {cfg_side} as configured",
        suptitle=f"{label}: the forecast book alone")
    plt.show()
    if save:
        save_clean(fig, save)


def export_pair(pt_, pairs, slug, offsets=None):
    """The deck's two tables: settlement by node at home-market prices, and the
    line table with each market's book flow, dual and rent. Root CSVs."""
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
        settle.loc[f'CONGESTION RENT {eng.name}'] = [
            eng.name, '', '', '', '', '',
            round(sub['paid by load ($/h)'].sum() - sub['paid to gen ($/h)'].sum(), 1)]
    pf = physical_flows(pt_, [r for _, r in pairs]).rename(columns={'F_phys': 'F^s (MW)'})
    lines = pf[['from', 'to', 'F[Market A]', 'F[Market B]', 'F^s (MW)',
                'rating', 'loading_%', 'overload']].copy()
    for _c in ('F[Market A]', 'F[Market B]', 'F^s (MW)', 'rating'):
        lines[_c] = lines[_c].astype(float).round(1)
    lines['loading_%'] = lines['loading_%'].astype(float).round(0)
    off = offsets or {}
    for eng, r in pairs:
        book = {l: float(r.flow_own[l]) + float(off.get(eng.name, {}).get(l, 0.0))
                for l in pt_.lines}
        mu = {l: abs(float(r.line_dual.get(l, 0.0))) for l in pt_.lines}
        lines[f'book flow {eng.name} (MW)'] = [round(book[l], 1) for l in lines.index]
        lines[f'|mu| {eng.name} ($/MWh)'] = [round(mu[l], 2) for l in lines.index]
        lines[f'rent {eng.name} ($/h)'] = [round(mu[l] * abs(book[l]), 1) for l in lines.index]
    if CFG['save_enabled']:
        os.makedirs(CFG['export_dir'], exist_ok=True)
        fs = os.path.join(CFG['export_dir'], f'seams_cm_{slug}_settlement.csv')
        fl = os.path.join(CFG['export_dir'], f'seams_cm_{slug}_lines.csv')
        settle.to_csv(fs, encoding='utf-8-sig')
        lines.to_csv(fl, encoding='utf-8-sig')
        print(f'wrote {fs} and {fl}')
    try:
        display(settle)
        display(lines)
    except NameError:
        pass
    return settle, lines


def save_clean(fig, name, size=None):
    """Deck PNG copy: suptitle, panel titles and legends removed, transparent
    background, the canvas fixed so every PNG of a graph type lands the same
    pixel size. No file is written unless the notebook enabled saving."""
    if not CFG['save_enabled']:
        return
    size = size or CFG['export_size']
    os.makedirs(CFG['export_dir'], exist_ok=True)
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_visible(False)
    for ax_ in fig.axes:
        ax_.set_title('')
        if ax_.get_legend() is not None:
            ax_.get_legend().remove()
    for leg_ in list(fig.legends):
        leg_.remove()
    fig.set_size_inches(*size)
    # explicit full-canvas bbox: the inline backend's tight-bbox default would
    # otherwise crop/expand each figure to its content and break the uniform size
    from matplotlib.transforms import Bbox
    fig.savefig(os.path.join(CFG['export_dir'], f'{name}.png'), dpi=CFG['export_dpi'],
                transparent=True,
                bbox_inches=Bbox([[0, 0], [size[0], size[1]]]))
    print(f"saved {CFG['export_dir']}/{name}.png")
