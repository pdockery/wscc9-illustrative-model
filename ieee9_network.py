# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
IEEE 9-bus test network as a PyPSA Network.

Loads the canonical Anderson & Fouad 9-bus system from pandapower, converts to
PyPSA, then enriches with time-varying load profiles and marginal costs
suitable for the ASSUME ``load_pypsa()`` workflow.

Source: pandapower.networks.case9() (PYPOWER origin, Anderson & Fouad 1980)
"""

import logging

import numpy as np
import pandas as pd
import pandapower.networks as pn
import pypsa

logger = logging.getLogger(__name__)


def _fix_pandapower_names(pp_net) -> None:
    """Assign unique names to pandapower components.

    pandapower's ``case9`` already ships canonical bus names (integers 1-9, equal
    to ``bus index + 1`` -- the standard PYPOWER/MATPOWER case9 numbering), but its
    generator, ext_grid, line, and load components have non-unique/None names,
    which makes PyPSA's ``import_from_pandapower_net`` raise
    ``ValueError: Names must be unique``. We assign unique names to those
    components here; the bus reassignment to ``str(i + 1)`` reproduces the
    shipped 1-9 numbering (so bus "1" is the slack, "5/7/9" the load buses, etc.).
    """
    pp_net.bus["name"] = [str(i + 1) for i in pp_net.bus.index]
    pp_net.gen["name"] = [f"gen_{i}" for i in pp_net.gen.index]
    pp_net.ext_grid["name"] = [f"gen_slack_{i}" for i in pp_net.ext_grid.index]
    pp_net.load["name"] = [f"load_{i}" for i in pp_net.load.index]
    pp_net.line["name"] = [f"line_{i}" for i in pp_net.line.index]
    if len(pp_net.trafo) > 0:
        pp_net.trafo["name"] = [f"trafo_{i}" for i in pp_net.trafo.index]


def build_ieee9_network(
    start: str = "2025-01-01",
    periods: int = 24,
    freq: str = "1h",
    load_scale: float = 1.0,
    profile_cfg: dict | None = None,
) -> pypsa.Network:
    """Load IEEE 9-bus from pandapower, convert to PyPSA, and enrich.

    The conversion preserves the canonical case9 topology and electrical
    parameters (bus voltages, line reactances, thermal ratings, generator
    limits).  On top of that we add:

    - **Snapshots**: A DatetimeIndex for time-series simulation.
    - **Marginal costs**: Differentiated across generators for price separation.
    - **Load profiles**: Synthetic 24h diurnal pattern on each load bus.
    - **UC parameters**: ramp limits, min up/down times for SCUC readiness.

    Generator cost assignment (creating a price gradient):
      - gen_slack_0 (bus 1, 250 MW): $20/MWh  — baseload
      - gen_0       (bus 2, 300 MW): $35/MWh  — mid-merit
      - gen_1       (bus 3, 270 MW): $50/MWh  — peaker

    Args:
        start: Start datetime for snapshots.
        periods: Number of time periods.
        freq: Snapshot frequency (e.g., '1h', '15min').
        load_scale: Multiplier applied to all loads.

    Returns:
        PyPSA Network ready for ``load_pypsa()``.
    """
    # ── Load from pandapower ───────────────────────────────────────────
    pp_net = pn.case9()
    _fix_pandapower_names(pp_net)

    n = pypsa.Network()
    n.import_from_pandapower_net(pp_net)

    # ── Fix missing capacity values ───────────────────────────────────
    # PyPSA's pandapower importer doesn't carry over p_nom for generators
    # or s_nom for lines.  Reconstruct from the original pandapower data.

    # Generator p_nom: from pp ext_grid.max_p_mw and gen.max_p_mw
    gen_capacity = {}
    gen_p_nom_min = {}
    for _, row in pp_net.ext_grid.iterrows():
        name = row["name"]
        gen_capacity[name] = row["max_p_mw"]
        gen_p_nom_min[name] = 0.0  # No must-run minimum (SCED handles commitment)
    for _, row in pp_net.gen.iterrows():
        name = row["name"]
        gen_capacity[name] = row["max_p_mw"]
        gen_p_nom_min[name] = 0.0  # No must-run minimum (SCED handles commitment)
    n.generators["p_nom"] = n.generators.index.map(gen_capacity)
    n.generators["p_nom_min"] = n.generators.index.map(gen_p_nom_min)

    # Line s_nom and impedances from pandapower
    # PyPSA's import_from_pandapower_net doesn't carry over impedances, so
    # we compute per-unit values manually: x_pu = x_ohm_per_km * length_km / Z_base
    v_nom = pp_net.bus.vn_kv.iloc[0]  # 345 kV uniform
    base_mva = 100.0
    z_base = v_nom**2 / base_mva  # ohm

    line_s_nom = {}
    line_x_pu = {}
    line_r_pu = {}
    for _, row in pp_net.line.iterrows():
        name = row["name"]
        line_s_nom[name] = row["max_i_ka"] * np.sqrt(3) * v_nom  # MVA
        line_x_pu[name] = row["x_ohm_per_km"] * row["length_km"] / z_base
        line_r_pu[name] = row["r_ohm_per_km"] * row["length_km"] / z_base
    n.lines["s_nom"] = n.lines.index.map(line_s_nom)
    n.lines["x_pu_eff"] = n.lines.index.map(line_x_pu)
    n.lines["r_pu_eff"] = n.lines.index.map(line_r_pu)
    # Also set the standard PyPSA impedance fields (per-unit, used by PyPSA OPF)
    n.lines["x"] = n.lines.index.map(line_x_pu)
    n.lines["r"] = n.lines.index.map(line_r_pu)

    logger.info(
        f"Imported IEEE 9-bus: {len(n.buses)} buses, {len(n.generators)} gens, "
        f"{len(n.loads)} loads, {len(n.lines)} lines"
    )

    # ── Snapshots ──────────────────────────────────────────────────────
    snapshots = pd.date_range(start, periods=periods, freq=freq)
    n.set_snapshots(snapshots)

    # ── Generator enrichment ───────────────────────────────────────────
    # Assign differentiated marginal costs and UC parameters.
    # The canonical case9 has: ext_grid on bus 1 (slack), gen on bus 2, gen on bus 3.
    # After import, PyPSA names them gen_slack_0, gen_0, gen_1.
    marginal_costs = {}
    carriers = {}
    for gen_name in n.generators.index:
        bus = n.generators.at[gen_name, "bus"]
        if bus == "1":
            marginal_costs[gen_name] = 20.0
            carriers[gen_name] = "nuclear"
        elif bus == "2":
            marginal_costs[gen_name] = 35.0
            carriers[gen_name] = "gas"
        elif bus == "3":
            marginal_costs[gen_name] = 50.0
            carriers[gen_name] = "gas"
        else:
            marginal_costs[gen_name] = 40.0
            carriers[gen_name] = "other"

    n.generators["marginal_cost"] = pd.Series(marginal_costs)
    n.generators["carrier"] = pd.Series(carriers)
    n.generators["ramp_limit_start_up"] = 0.5
    n.generators["ramp_limit_shut_down"] = 0.5
    n.generators["min_up_time"] = 2
    n.generators["min_down_time"] = 1

    # ── Load profiles ──────────────────────────────────────────────────
    # Diurnal pattern driven by profile_cfg (from scenarios.yaml).
    # Defaults reproduce the original hardcoded shape: peak at H12, trough at H0/H23.
    # Use actual hour-of-day from snapshots so non-midnight start times work.
    pcfg = profile_cfg or {}
    base        = pcfg.get("base",        0.6)
    amplitude   = pcfg.get("amplitude",   0.4)
    phase_shift = pcfg.get("phase_shift", 4)
    period      = pcfg.get("period",      32)   # full sine period in hours
    clip_min    = pcfg.get("clip_min",    0.5)
    clip_max    = pcfg.get("clip_max",    1.0)

    hours = snapshots.hour.values
    profile = base + amplitude * np.sin(2 * np.pi * (hours - phase_shift) / period)
    profile = np.clip(profile, clip_min, clip_max)

    for load_name in n.loads.index:
        p_base = n.loads.at[load_name, "p_set"] * load_scale
        n.loads_t.p_set[load_name] = pd.Series(
            p_base * profile, index=snapshots
        )

    return n


# ── Zone partitioning ──────────────────────────────────────────────────
# Using 1-indexed bus names from the canonical case9 (as converted by PyPSA).
# Zone A: buses 1, 2, 4, 7, 8  (gen_slack_0 on bus1, gen_0 on bus2; load on bus7)
# Zone B: buses 3, 5, 6, 9     (gen_1 on bus3; loads on bus5 and bus9)
ZONE_A_BUSES = {"1", "2", "4", "7", "8"}
ZONE_B_BUSES = {"3", "5", "6", "9"}


def assign_zones(network: pypsa.Network) -> pypsa.Network:
    """Add ``zone`` column to buses DataFrame.

    Zone A: buses 1, 2, 4, 7, 8 — two generators (cheaper), one load
    Zone B: buses 3, 5, 6, 9    — one generator (expensive), two loads
    """
    network.buses["zone"] = network.buses.index.map(
        lambda b: "zone_a" if b in ZONE_A_BUSES else "zone_b"
    )
    return network


def get_seam_lines(network: pypsa.Network) -> pd.DataFrame:
    """Return lines that cross zone boundaries (the seam)."""
    bus_zone = network.buses["zone"]
    mask = network.lines.bus0.map(bus_zone) != network.lines.bus1.map(bus_zone)
    return network.lines[mask]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = build_ieee9_network()
    assign_zones(n)

    print(f"Buses: {n.buses.index.tolist()}")
    print(f"  zones: {n.buses['zone'].to_dict()}")
    print(f"\nGenerators:")
    print(n.generators[["bus", "p_nom", "marginal_cost", "carrier"]].to_string())
    print(f"\nLoads:")
    print(n.loads[["bus", "p_set"]].to_string())
    print(f"\nLoad profile (first 6h):")
    print(n.loads_t.p_set.head(6).to_string())
    print(f"\nLines:")
    print(n.lines[["bus0", "bus1", "s_nom"]].to_string())
    print(f"\nSeam lines: {get_seam_lines(n).index.tolist()}")
    print(f"Snapshots: {n.snapshots[0]} to {n.snapshots[-1]} ({len(n.snapshots)} periods)")

    # Validate canonical structure
    assert len(n.buses) == 9, f"Expected 9 buses, got {len(n.buses)}"
    assert len(n.generators) == 3, f"Expected 3 generators, got {len(n.generators)}"
    assert len(n.loads) == 3, f"Expected 3 loads, got {len(n.loads)}"
    assert len(n.snapshots) == 24
    assert n.loads_t.p_set.shape == (24, 3), "Load time series shape mismatch"
    print("\nAll validations passed.")
