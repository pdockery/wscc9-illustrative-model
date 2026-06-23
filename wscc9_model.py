# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""WSCC/IEEE 9-bus teaching scenario: network, shift factors, and market engine.

This is the scenario layer the illustrative notebooks share. It sits on top of
two primitive modules and adds the one fixed teaching set-up they all reuse:

    ``ieee9_network.build_ieee9_network``  →  the canonical 9-bus PyPSA case
    ``seams_engine.compute_ptdf``          →  the DC shift-factor matrix (PTDF)
    ``seams_engine.MarketEngine``          →  a resource/load/constraint partition

On that base it pins down the **teaching fleet** (which unit sits where, at what
cost), the **load pattern**, and the **drawing layout** (ring order, rotation,
bus colours) so that every notebook — and every figure within a notebook — reads
the same way.

Design rule (the notebooks keep their knobs visible): this module holds *logic*
and the *immutable base case* only. The canonical fleet/loads are exposed as
``DEFAULT_GEN_FLEET`` / ``DEFAULT_LOADS`` and used as defaults, but a notebook's
own ``# --EDIT--`` cell may override any number and pass it back in (e.g.
``make_engine(..., gen_fleet=my_fleet)`` or re-bidding an engine's gens after the
fact). Nothing here decides a market *issue*; that is the job of the issue
notebooks.

Generators (the teaching price gradient — note bus 3 is the CHEAPEST, unlike the
raw ``ieee9_network`` cost map):

    gen_slack_0  bus 1  $50/MWh  250 MW   — slack, most expensive
    gen_0        bus 2  $35/MWh  300 MW   — mid-merit
    gen_1        bus 3  $20/MWh  170 MW   — cheapest

Loads: 90 MW @ bus 5, 100 MW @ bus 7, 125 MW @ bus 9.
"""

from __future__ import annotations

import pandas as pd

import nodal_plot
from ieee9_network import build_ieee9_network
from seams_engine import MarketEngine, compute_ptdf
from nodal_plot import assign_bus_colors


# ──────────────────────────────────────────────────────────────────────────
# The teaching scenario (fleet, loads) — base defaults; notebooks may override
# ──────────────────────────────────────────────────────────────────────────
#: Resource stack on the shared network. ``{gen_id: {bus, cost $/MWh, p_nom MW}}``.
DEFAULT_GEN_FLEET = {
    "gen_slack_0": {"bus": "1", "cost": 50.0, "p_nom": 250.0},   # slack unit
    "gen_0":       {"bus": "2", "cost": 35.0, "p_nom": 300.0},   # mid-merit
    "gen_1":       {"bus": "3", "cost": 20.0, "p_nom": 170.0},   # cheapest unit
}

#: Fixed load at each bus (MW).
DEFAULT_LOADS = {"5": 90.0, "7": 100.0, "9": 125.0}

#: Three-BA teaching scenario (notebooks 112 / 212). Nodes 2 and 3 each carry gen +
#: load and are individually self-sufficient (node 3 cheaper + more load; node 2
#: pricier + less load); the two canonical cheap units move to the formerly-transit
#: nodes 6 and 8 so they sit inside the "rest of network" BA. The clearing uses these
#: via ``make_engine(gen_fleet=FLEET_3BA, loads=LOADS_3BA)`` — the base PyPSA network
#: (topology / reactances) is unchanged.
FLEET_3BA = {
    "gen_1": {"bus": "1", "cost": 50.0, "p_nom": 250.0},   # slack (rest-of-network BA)
    "gen_6": {"bus": "6", "cost": 20.0, "p_nom": 170.0},   # cheapest — moved from node 3
    "gen_8": {"bus": "8", "cost": 35.0, "p_nom": 300.0},   # mid-merit — moved from node 2
    "gen_3": {"bus": "3", "cost": 25.0, "p_nom": 160.0},   # node-3 BA: cheaper, surplus exports
    "gen_2": {"bus": "2", "cost": 48.0, "p_nom": 90.0},    # node-2 BA: pricier
}
LOADS_3BA = {"2": 60.0, "3": 120.0, "5": 90.0, "7": 100.0, "9": 125.0}

#: Ordered bus names of the case (sorted as the engine sorts them).
BUSES = [str(i) for i in range(1, 10)]


# ──────────────────────────────────────────────────────────────────────────
# Drawing layout — applied to every node + circlize plot so they read alike
# ──────────────────────────────────────────────────────────────────────────
#: Circlize sector order: clockwise from 12 o'clock, following the network ring
#: so the chord diagram reads in the same spatial order as the topology diagram.
RING_ORDER = ["3", "6", "7", "8", "2", "9", "4", "1", "5"]

#: Node-diagram rotation (deg): bus 3 at the bottom, bus 2 NW, bus 9 at top.
ROTATION_DEG = 180

#: Circlize centre: bus 9 placed across 12 o'clock.
CENTER_BUS = "9"

#: Rotated bus coordinates for the network topology panel.
COORDS = nodal_plot.rotate_coords(nodal_plot.IEEE9_COORDS, ROTATION_DEG)

#: Canonical per-bus colours, assigned ONCE from the full fleet (which buses host
#: gen/load) so every figure — unified, scenario, or per-footprint — uses the
#: SAME colour for a given bus. A subset clearing never restarts the palette.
_sup_full: dict[str, list] = {}
for _g, _s in DEFAULT_GEN_FLEET.items():
    _sup_full.setdefault(_s["bus"], []).append(_s)
BUS_COLORS = assign_bus_colors(BUSES, _sup_full, DEFAULT_LOADS)


# ──────────────────────────────────────────────────────────────────────────
# Network construction
# ──────────────────────────────────────────────────────────────────────────
def build_network(line_ratings: dict | None = None, split_5_6: bool = False):
    """Build the 9-bus PyPSA network, applying any line-rating overrides.

    Parameters
    ----------
    line_ratings : dict, optional
        ``{line: MW}`` overrides applied to ``s_nom`` (e.g. ``{'line_4': 40}``).
        Applied AFTER any disaggregation, so a split circuit can still be re-rated.
    split_5_6 : bool, optional
        If True, the 5-6 interface ``line_2`` is modelled as a parallel DOUBLE
        CIRCUIT — one circuit per balancing authority. ``line_2`` is disaggregated
        by the standard PyPSA-Eur/Earth rule: N circuits each carry ``s_nom/N`` at
        ``N·x`` (and ``N·r``), so the pair is electrically IDENTICAL to the single
        line. Here N=2 → ``line_2_ba1`` + ``line_2_ba2``, each 75 MW at 2·x.
        ``n.calculate_dependent_values()`` is then required so the fresh lines pick
        up ``x_pu_eff`` (``compute_ptdf`` prefers it over raw ``x``; without it the
        pair would NOT be equivalent). Default False (single ``line_2``).

    Returns
    -------
    pypsa.Network
    """
    n = build_ieee9_network(periods=1)
    if split_5_6:
        x2, r2, s2 = (float(n.lines.loc["line_2", k]) for k in ("x", "r", "s_nom"))
        b0, b1 = n.lines.loc["line_2", "bus0"], n.lines.loc["line_2", "bus1"]
        n.remove("Line", "line_2")
        n.add("Line", "line_2_ba1", bus0=b0, bus1=b1, x=2 * x2, r=2 * r2, s_nom=s2 / 2)
        n.add("Line", "line_2_ba2", bus0=b0, bus1=b1, x=2 * x2, r=2 * r2, s_nom=s2 / 2)
        n.calculate_dependent_values()   # recompute x_pu_eff → electrically identical to line_2
    for ln, mw in (line_ratings or {}).items():
        n.lines.loc[ln, "s_nom"] = float(mw)
    return n


def build_transport_network(buses, line_ratings: dict | None = None, x: float = 0.1):
    """A tiny PyPSA network of ``buses`` joined in a chain by *transfer* lines.

    The transport-layer abstraction for balancing authorities that are **not
    electrically connected** and coordinate only through scheduling limits (e.g.
    Markets+ BAAs): there is no physical grid here, just one "transfer line" per
    consecutive bus pair, named ``transfer_<a>_<b>``, at reactance ``x`` and a
    rating Ē taken from ``line_ratings`` (keyed by the line name, default very
    large = slack). ``compute_ptdf`` + ``solve_engine_dispatch`` then clear it; on
    the resulting 2-bus network the line carries the net interchange E and its
    dual is the transfer shadow price μ_T. ``slack_bus`` must be one of ``buses``.

    Parameters
    ----------
    buses : iterable
        The transport nodes (e.g. the single-bus BAs of one market).
    line_ratings : dict, optional
        ``{transfer_line_name: Ē MW}`` scheduling limits; default 1e4 (slack).
    """
    import pypsa

    buses = [str(b) for b in buses]
    ratings = line_ratings or {}
    n = pypsa.Network()
    for b in buses:
        n.add("Bus", b, v_nom=345.0)
    for a, b in zip(buses[:-1], buses[1:]):
        name = f"transfer_{a}_{b}"
        n.add("Line", name, bus0=a, bus1=b, x=x, r=0.0,
              s_nom=float(ratings.get(name, 1e4)))
    n.calculate_dependent_values()
    return n


def shift_factors(network=None, slack_bus: str = "1"):
    """DC shift-factor matrix (PTDF) of the network (default: the base case)."""
    if network is None:
        network = build_network()
    return compute_ptdf(network, slack_bus=slack_bus)


def sf_table(pt) -> pd.DataFrame:
    """Readable shift-factor table ``SF[line (a-b), bus]`` for display."""
    return pd.DataFrame(
        pt.ptdf,
        index=[f"{l} ({a}-{b})" for l, (a, b) in zip(pt.lines, pt.line_buses)],
        columns=[f"bus {b}" for b in pt.buses],
    ).round(3)


# ──────────────────────────────────────────────────────────────────────────
# Market engine construction
# ──────────────────────────────────────────────────────────────────────────
def make_engine(
    name: str,
    buses,
    gen_fleet: dict | None = None,
    loads: dict | None = None,
    activated="all",
) -> MarketEngine:
    """A ``MarketEngine`` owning every fleet gen and load whose bus is in ``buses``.

    Parameters
    ----------
    name : str
        Engine label.
    buses : iterable
        The bus set this engine optimises over.
    gen_fleet, loads : dict, optional
        Override the canonical fleet/loads (default ``DEFAULT_GEN_FLEET`` /
        ``DEFAULT_LOADS``) — the visible-knob hook for a notebook EDIT cell.
    activated : "all" | list[str]
        The activated constraint set ℳ^M_act (lines this engine enforces limits on).
    """
    fleet = DEFAULT_GEN_FLEET if gen_fleet is None else gen_fleet
    loads = DEFAULT_LOADS if loads is None else loads
    buses = {str(b) for b in buses}
    gens = {g: dict(s) for g, s in fleet.items() if str(s["bus"]) in buses}
    own_loads = {b: mw for b, mw in loads.items() if b in buses}
    return MarketEngine(name=name, gens=gens, loads=own_loads, activated_lines=activated)


if __name__ == "__main__":
    pt = shift_factors()
    print("fleet:", {g: (s["bus"], s["cost"], s["p_nom"]) for g, s in DEFAULT_GEN_FLEET.items()})
    print("loads:", DEFAULT_LOADS)
    print("ring order:", RING_ORDER, "| rotation", ROTATION_DEG, "| centre", CENTER_BUS)
    print("bus colours:", BUS_COLORS)
    print("\nshift factors (slack = bus 1):")
    print(sf_table(pt))

    # base single-line vs split double-circuit are electrically identical
    pt1 = compute_ptdf(build_network(), slack_bus="1")
    pt2 = compute_ptdf(build_network(split_5_6=True), slack_bus="1")
    eng = make_engine("UNIFIED", buses=pt1.buses)
    from seams_engine import solve_engine_dispatch
    f1 = solve_engine_dispatch(pt1, eng).flow_own
    eng2 = make_engine("UNIFIED", buses=pt2.buses)
    r2 = solve_engine_dispatch(pt2, eng2)
    # the split circuit's two halves should sum to the single line's flow
    split = r2.flow_own["line_2_ba1"] + r2.flow_own["line_2_ba2"]
    print(f"\nline_2 flow: single {f1['line_2']:.3f}  vs  split-sum {split:.3f}  "
          f"(Δ {abs(f1['line_2'] - split):.5f} — should be ~0)")
