# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Intertie schedule placement — SP-Tie vs GAP-Tie — for notebooks 222 and 322.

An intertie schedule is an import or export agreed at a scheduling point on the
seam between two markets. The placement of that schedule on the network is the
only thing that varies here:

    SP-Tie   the whole schedule E sits at the tie bus;
    GAP-Tie  the schedule is spread over the counterparty's generation
             aggregation point, bus n taking the share E·GDF_n.

Each clear is rendered as a **3-set**: (1) the importer's own model of the
schedule, (2) the exporter's own model of it as demand at the counterparty
location, and (3) the *combined actual* — both real dispatches on the one
network, with the flows that physically result. The gap between (1) and (3) is
the phantom congestion the notebooks price.

222 clears once; 322 clears the same machinery twice, day-ahead on a load
forecast and real-time on the load that shows up, and threads a ``period``
label and the settlement's own load through the same functions.

The notebook passes its context once through :func:`configure` — the two
markets, the tie bus and the aggregation points, the flowgate, the schedule and
its bid — and the functions keep the signatures the notebooks call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np

import nodal_plot
import wscc9_figures as figs
from nodal_plot import plot_combined_letter
from seams_engine import (
    compute_ptdf, solve_engine_dispatch, susceptance_widths, to_supply_demand,
)
from wscc9_model import (
    BUS_COLORS, CENTER_BUS, COORDS, RING_ORDER, build_network, make_engine,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pypsa
    from matplotlib.figure import Figure

    from footprints import Footprints
    from lab_types import Bus, BusValues, Fleet, Line, LineValues, Loads
    from seams_engine import EngineResult, MarketEngine, PTDFData

    class _Config(TypedDict):
        """The context once configure has run; CFG opens most slots as None."""

        importer: str
        exporter: str
        b_buses: list[Bus]
        a_buses: list[Bus]
        b_lines: list[Line]
        a_lines: list[Line]
        tie_bus: Bus
        dgap: BusValues
        b_gap: BusValues
        flowgate: Line
        flowgate_mw: float
        schedule_mw: float
        import_bid: float
        shed_price: float
        all_loads: Loads
        fp: Footprints
        mkt_colors: dict[str, str]
        cost: BusValues
        import_color: str

#: Module context, set by :func:`configure` from the notebook's EDIT cell.
CFG: _Config = {
    "importer": "Market B",     # the market that buys the schedule
    "exporter": "Market A",     # the market that sources it
    "b_buses": None,  # type: ignore[typeddict-item]  # the importer's buses
    "a_buses": None,  # type: ignore[typeddict-item]  # the exporter's buses
    "b_lines": None,  # type: ignore[typeddict-item]  # the importer's activated constraint set
    "a_lines": None,  # type: ignore[typeddict-item]  # the exporter's activated constraint set
    "tie_bus": None,  # type: ignore[typeddict-item]  # the scheduling point (SP-Tie location)
    "dgap": None,  # type: ignore[typeddict-item]  # {bus: GDF} the exporter's DGAP, GDFs summing to 1
    "b_gap": None,  # type: ignore[typeddict-item]  # {bus: GDF} the importer's GAP (the delivery location)
    "flowgate": None,  # type: ignore[typeddict-item]  # the studied constraint
    "flowgate_mw": None,  # type: ignore[typeddict-item]  # its rating (MW)
    "schedule_mw": None,  # type: ignore[typeddict-item]  # the intertie schedule E (MW)
    "import_bid": None,  # type: ignore[typeddict-item]  # the schedule's offer price ($/MWh)
    "shed_price": 300.0,        # an intertie VOLL: a shed schedule is a firmness failure
    "all_loads": None,  # type: ignore[typeddict-item]  # {bus: MW} every load on the transmission element
    "fp": None,  # type: ignore[typeddict-item]  # Footprints partition (fp_of, colors, line_colors)
    "mkt_colors": None,  # type: ignore[typeddict-item]  # {market: color} for the combined-actual bands
    "cost": None,  # type: ignore[typeddict-item]  # {bus: $/MWh} marginal cost, for the combined-actual bars
    "import_color": "#E67E22",  # the schedule drawn orange wherever it is placed
}


def configure(**kw: object) -> None:
    """Set the module context. Unknown keys are rejected."""
    for k in kw:
        if k not in CFG:
            raise KeyError(f"intertie.configure: unknown key {k!r}")
    CFG.update(kw)  # type: ignore[typeddict-item]  # the loop admits CFG's keys alone


def build_net() -> pypsa.Network:
    """The shared network with the studied flowgate derated to its rating."""
    return build_network({CFG["flowgate"]: CFG["flowgate_mw"]})


def tie_inject(mode: str, E: float | None = None) -> Fleet:
    """The schedule as bid generation, placed per the tie model: at the tie bus (SP)
    or blended over the exporter's DGAP buses (GAP). Returns ``{gen_id: spec}`` to
    add to the importer's fleet."""
    E = CFG["schedule_mw"] if E is None else E
    bid = CFG["import_bid"]
    if mode == "SP":
        return {"import": {"bus": CFG["tie_bus"], "cost": bid, "p_nom": E}}
    return {f"import_{b}": {"bus": b, "cost": bid, "p_nom": E * w}
            for b, w in CFG["dgap"].items()}


def tie_clear(
    mode: str, E: float | None = None, loads: Loads | None = None
) -> tuple[pypsa.Network, PTDFData, MarketEngine, EngineResult]:
    """Clear the importer's market with the schedule modeled SP-tie or GAP-tie.
    ``loads`` overrides the importer's own load — the day-ahead forecast or the
    real-time actual in the two-settlement notebook."""
    net = build_net()
    pt = compute_ptdf(net, slack_bus="1")
    e = make_engine(CFG["importer"], CFG["b_buses"], loads=loads,
                    activated=CFG["b_lines"])
    e.gens = dict(e.gens)
    e.gens.update(tie_inject(mode, E))
    r = solve_engine_dispatch(pt, e, shed_price=CFG["shed_price"])
    return net, pt, e, r


def import_mw(r: EngineResult) -> float:
    """The schedule's cleared megawatts, summed over however many buses the
    placement spread it across."""
    return sum(v for k, v in r.dispatch.items() if str(k).startswith("import"))


def tie_figure(
    mode: str,
    net: pypsa.Network,
    pt: PTDFData,
    e: MarketEngine,
    r: EngineResult,
    suptitle: str,
) -> Figure:
    """The importer's own model of the schedule: its footprint in color, the rest
    grayed, the scheduled block in orange wherever the tie model places it."""
    sup, dem = to_supply_demand(e, r)
    for b in sup:
        for g in sup[b]:
            if str(g["unit_id"]).startswith("import"):
                g["color"] = CFG["import_color"]
    loc = "tie bus" if mode == "SP" else "DGAP buses"
    return figs.footprint_figure(
        net, pt, CFG["fp"], e, r, highlight=CFG["importer"],
        dim_buses=CFG["a_buses"], sup_dem=(sup, dem),
        legend_note=f"Import bid ({CFG['schedule_mw']:.0f} MW @ "
                    f"${CFG['import_bid']:.0f}) drawn in orange at its {loc}",
        suptitle=suptitle)


def neigh_clear(
    mode: str, E: float
) -> tuple[pypsa.Network, PTDFData, MarketEngine, EngineResult, BusValues]:
    """The other footprint serves the transaction. For an import, the exporter
    dispatches its own units to meet the export as added DEMAND, placed per the
    methodology — at the scheduling point (SP-Tie) or at the importer's GAP
    (GAP-Tie), the counterparty location."""
    net = build_net()
    pt = compute_ptdf(net, slack_bus="1")
    e = make_engine(CFG["exporter"], CFG["a_buses"], activated=CFG["a_lines"])
    exo = ({CFG["tie_bus"]: -E} if mode == "SP"
           else {b: -E * w for b, w in CFG["b_gap"].items()})
    r = solve_engine_dispatch(pt, e, exo=exo, shed_price=CFG["shed_price"])
    return net, pt, e, r, exo


def neigh_figure(
    mode: str,
    net: pypsa.Network,
    pt: PTDFData,
    e: MarketEngine,
    r: EngineResult,
    exo: BusValues,
    suptitle: str,
    highlight: str | None = None,
    dim: Iterable[Bus] | None = None,
) -> Figure:
    """The other footprint's view. Its schedule draws via ``exo_sched`` — an export
    as demand, an import as gray price-taking supply. ``mode`` is carried for
    call-site symmetry with the other 3-set panels."""
    return figs.footprint_figure(
        net, pt, CFG["fp"], e, r,
        highlight=CFG["exporter"] if highlight is None else highlight,
        dim_buses=CFG["b_buses"] if dim is None else dim,
        exo_sched=exo, suptitle=suptitle)


def combined_actual(
    rc: EngineResult,
    rn: EngineResult,
    pt: PTDFData | None = None,
    all_loads: Loads | None = None,
) -> tuple[BusValues, LineValues]:
    """The real physical state the two models approximate: each market's REAL
    generation (the importer's own unit; the exporter's DGAP units) serving every
    load, and the actual flows. Flows are rating-independent; ``pt`` only carries
    the scenario's flowgate rating for the check."""
    pt = _ptdf(pt)
    all_loads = CFG["all_loads"] if all_loads is None else all_loads
    gap_buses = list(CFG["b_gap"]) + list(CFG["dgap"])
    realgen = {b: (rc if b in CFG["b_gap"] else rn).gen_by_bus.get(b, 0.0)
               for b in gap_buses}
    inj = np.zeros(pt.n_bus)
    for b, g in realgen.items():
        inj[pt.bus_idx[b]] += g
    for b, d in all_loads.items():
        inj[pt.bus_idx[str(b)]] -= d
    lflow = {l: float(pt.ptdf[pt.line_idx[l]] @ inj) for l in pt.lines}
    return realgen, lflow


def combined_figure(
    rc: EngineResult,
    rn: EngineResult,
    suptitle: str,
    net: pypsa.Network | None = None,
    pt: PTDFData | None = None,
    all_loads: Loads | None = None,
) -> tuple[Figure, LineValues]:
    """The combined actual as the network + nodal letter: the real dispatch on the
    one line, each side's buses priced at its OWN market's LMP."""
    pt = _ptdf(pt)
    net = build_net() if net is None else net
    all_loads = CFG["all_loads"] if all_loads is None else all_loads
    realgen, lflow = combined_actual(rc, rn, pt, all_loads)
    cost = CFG["cost"]
    sup = {b: [{"unit_id": f"g{b}", "price": cost[b], "volume": g,
                "capacity": g, "accepted_volume": g}]
           for b, g in realgen.items() if g > 0.5}
    dem = {str(b): float(v) for b, v in all_loads.items()}
    lmp = {b: (rc.lmp[b] if b in CFG["b_buses"] else rn.lmp[b]) for b in pt.buses}
    over = {l for l in pt.lines
            if abs(lflow[l]) > pt.s_nom[pt.line_idx[l]] + 1e-6}
    fig, _ = plot_combined_letter(
        net, sup, dem, bus_colors=dict(BUS_COLORS), bus_lmps=lmp,
        line_flows=lflow, line_widths=susceptance_widths(pt),
        line_colors=CFG["fp"].line_colors(pt), constrained_lines=over,
        flows=nodal_plot.compute_ptdf_flows(net, sup, dem),
        lmp_line=True, bus_groups=CFG["fp"].groups(pt),
        group_colors=dict(CFG["mkt_colors"]), show_group_labels=False,
        all_buses=pt.buses, sector_order=RING_ORDER, bus_coords=COORDS,
        center_bus=CENTER_BUS,
        title_left="Network -- the one line, actual flows", suptitle=suptitle)
    return fig, lflow


def three_set(
    mode: str, loads: Loads | None = None, period: str | None = None
) -> tuple[EngineResult, EngineResult, LineValues]:
    """The 3-set for one clear: (1) the importer modeled, (2) the exporter modeled,
    and (3) the combined ACTUAL dispatch and flows — the truth both approximations
    stand in for. ``loads`` clears on a settlement's own load and ``period`` labels
    it, the two-settlement notebook's day-ahead / real-time thread."""
    fg, fg_mw = CFG["flowgate"], CFG["flowgate_mw"]
    imp, exp = CFG["importer"], CFG["exporter"]
    tag = f"{period} | " if period else ""
    netC, ptC, eC, rC = tie_clear(mode, loads=loads)
    E = import_mw(rC)
    muC = rC.line_dual.get(fg, 0.0)
    netN, ptN, eN, rN, exoN = neigh_clear(mode, E)
    # the settlement's own load replaces the forecast load at the importer's buses
    all_loads = {**(CFG["all_loads"] or {}),
                 **{str(b): float(v) for b, v in (loads or {}).items()}}
    _, lflow = combined_actual(rC, rN, ptC, all_loads)
    l1 = lflow[fg]
    gap_bus = ", ".join(CFG["b_gap"])
    delivered_at = (f"the scheduling point (bus {CFG['tie_bus']})" if mode == "SP"
                    else f"{imp} GAP (bus {gap_bus})")
    tie_figure(mode, netC, ptC, eC, rC,
               f"{mode}-Tie | {tag}(1) {imp} modeled (importer) -- import {E:.0f} MW; "
               f"flowgate {fg} {'BINDS' if abs(muC) > 1e-3 else 'slack'}")
    plt.show()
    neigh_figure(mode, netN, ptN, eN, rN, exoN,
                 f"{mode}-Tie | {tag}(2) {exp} modeled (exporter) -- serves the "
                 f"{E:.0f} MW export as demand at {delivered_at}")
    plt.show()
    verdict = ("REAL congestion" if abs(l1) > fg_mw + 1 else
               "within limit -- the modeled congestion is PHANTOM"
               if abs(muC) > 1e-3 else "within limit")
    combined_figure(rC, rN,
                    f"{mode}-Tie | {tag}(3) Combined ACTUAL -- {fg} carries "
                    f"{abs(l1):.0f}/{fg_mw:.0f} MW ({verdict})",
                    net=netC, pt=ptC, all_loads=all_loads)
    plt.show()
    return rC, rN, lflow


def _ptdf(pt: PTDFData | None) -> PTDFData:
    """The passed shift factors, or the base network's when a caller omits them."""
    return compute_ptdf(build_net(), slack_bus="1") if pt is None else pt
