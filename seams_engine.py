# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multi-engine DC-OPF on a shared network — teaching harness for the seams paper.

This module implements, in the most transparent way possible, the shift-factor
algebra of *Market Seams in the Western Interconnection* (Dockery, 2026) on the
WSCC/IEEE 9-bus test system. It is the computational backbone for
``seams_examples.ipynb``.

The objects map one-to-one onto the paper's notation:

    PTDF[l, n]      SF_{n,m,t}      shift factor of bus n on constraint m   (§2.1)
    MarketEngine    engine M        a resource/load/constraint partition    (§2.1)
    EngineResult.lmp        λ^M_{n,t}   engine M's LMP at every bus           (9),(14),(17)
    EngineResult.energy_price   λ^M_t / λ^M_{j,t}   energy component
    EngineResult.line_dual      μ^M_{m,t}   congestion shadow price           (5)
    EngineResult.flow_own       F^M_{m,t}   engine M's *own* flow component   (6)
    physical_flows()        F^phys_{m,t} = Σ_M F^M_{m,t} + F^non   superposition (30)

Each engine optimises **only its own resources and loads**, enforces limits
**only on its own activated constraint set**, and computes flow **only against
its own injections** (eq. 6). Cross-engine injections enter as fixed,
price-taking schedules (``exo`` argument). The physical flow on any line is the
superposition of every engine's own component (eq. 30) — which is what produces
the overloads and price gaps the paper formalises.

The DC-OPF is solved with ``scipy.optimize.linprog(method="highs")``; the LMP
decomposition is read directly off the HiGHS dual variables, so the nodal price
is literally ``energy_price + Σ_l PTDF[l, n] · μ_l`` (eq. 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ──────────────────────────────────────────────────────────────────────────
# Shift factors (PTDF)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PTDFData:
    """Shared-network shift factors and bookkeeping (the paper's N, M, SF)."""

    ptdf: np.ndarray              # (n_line, n_bus) — SF_{n,m}
    buses: list[str]              # ordered bus names (N)
    lines: list[str]              # ordered line names (M)
    bus_idx: dict[str, int]
    line_idx: dict[str, int]
    line_buses: list[tuple[str, str]]   # (bus0, bus1) per line
    s_nom: np.ndarray             # (n_line,) — operating limit F̄_m
    susceptance: np.ndarray       # (n_line,) — b_m = 1/x_m ("slipperiness")
    slack_bus: str

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_line(self) -> int:
        return len(self.lines)

    def line_flow(self, injection: dict[str, float] | np.ndarray) -> np.ndarray:
        """F_m = Σ_n SF_{n,m} · p_n for an injection vector (dict or array)."""
        p = self._as_vector(injection)
        return self.ptdf @ p

    def _as_vector(self, injection) -> np.ndarray:
        if isinstance(injection, np.ndarray):
            return injection
        p = np.zeros(self.n_bus)
        for bus, mw in injection.items():
            p[self.bus_idx[str(bus)]] += mw
        return p


def compute_ptdf(network, slack_bus: str = "1") -> PTDFData:
    """Build the DC shift-factor matrix from a PyPSA network.

    Uses line reactances (``x_pu_eff`` / ``x``) and the standard reduced-
    susceptance inversion with one slack bus. The PTDF distribution depends only
    on *relative* reactances, so the per-unit/ohm convention is immaterial here.
    """
    buses = sorted(network.buses.index.tolist(), key=lambda x: int(x))
    lines = network.lines
    n_bus, n_line = len(buses), len(lines)
    bus_idx = {b: i for i, b in enumerate(buses)}

    b_line = np.zeros(n_line)
    C = np.zeros((n_line, n_bus))
    line_names, line_buses, s_nom = [], [], np.zeros(n_line)
    for li, (name, row) in enumerate(lines.iterrows()):
        x = row.get("x_pu_eff", row.get("x_pu", row.get("x", 0.01)))
        if abs(x) < 1e-10:
            x = 0.01
        b_line[li] = 1.0 / x
        b0, b1 = str(row["bus0"]), str(row["bus1"])
        C[li, bus_idx[b0]] = 1.0
        C[li, bus_idx[b1]] = -1.0
        line_names.append(str(name))
        line_buses.append((b0, b1))
        s_nom[li] = row["s_nom"]

    B_bus = C.T @ np.diag(b_line) @ C
    s_i = bus_idx[slack_bus]
    keep = [i for i in range(n_bus) if i != s_i]
    B_inv = np.linalg.inv(B_bus[np.ix_(keep, keep)])
    B_inv_full = np.zeros((n_bus, n_bus))
    for ii, i in enumerate(keep):
        for jj, j in enumerate(keep):
            B_inv_full[i, j] = B_inv[ii, jj]
    ptdf = np.diag(b_line) @ C @ B_inv_full

    return PTDFData(
        ptdf=ptdf,
        buses=buses,
        lines=line_names,
        bus_idx=bus_idx,
        line_idx={n: i for i, n in enumerate(line_names)},
        line_buses=line_buses,
        s_nom=s_nom,
        susceptance=b_line,
        slack_bus=slack_bus,
    )


def susceptance_widths(
    pt: PTDFData, wmin: float = 1.2, wmax: float = 7.0
) -> dict[str, float]:
    """Map each line's susceptance b_m = 1/x_m to a drawing width ("slipperiness").

    Lower reactance ⇒ higher susceptance ⇒ a *wider* line: for a given angle
    difference, DC flow on a line is proportional to b_m, so a wider line is the
    one that "wants" more of the flow, all else equal. Returns ``{line: width}``
    scaled linearly across [wmin, wmax].
    """
    b = pt.susceptance
    lo, hi = float(b.min()), float(b.max())
    if hi - lo < 1e-9:
        return {pt.lines[i]: 0.5 * (wmin + wmax) for i in range(pt.n_line)}
    w = wmin + (b - lo) / (hi - lo) * (wmax - wmin)
    return {pt.lines[i]: float(w[i]) for i in range(pt.n_line)}


# ──────────────────────────────────────────────────────────────────────────
# Market engine definition (the paper's R^M, D^M, M^M_act)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MarketEngine:
    """One independent optimisation over a subset of the shared network.

    Attributes
    ----------
    name : str
        Engine label (e.g. "A — WEM", "B — Markets+").
    gens : dict
        ``{gen_id: {"bus": str, "cost": $/MWh, "p_nom": MW}}`` — the engine's
        resource stack R^M.
    loads : dict
        ``{bus: MW}`` — the engine's served load D^M.
    activated_lines : "all" | list[str]
        The activated constraint set M^M_act. Lines outside this set have their
        limit F̄_m relaxed to +∞ inside this engine (silent congestion).
    """

    name: str
    gens: dict[str, dict] = field(default_factory=dict)
    loads: dict[str, float] = field(default_factory=dict)
    activated_lines: object = "all"      # "all" or list of line names

    @property
    def buses(self) -> set[str]:
        bs = {str(g["bus"]) for g in self.gens.values()}
        bs |= {str(b) for b in self.loads}
        return bs


@dataclass
class EngineResult:
    name: str
    dispatch: dict[str, float]              # gen_id -> MW
    gen_by_bus: dict[str, float]
    load_by_bus: dict[str, float]
    exo_by_bus: dict[str, float]
    injection: np.ndarray                  # p^inj,M_n (own gens+loads+exo)
    lmp: dict[str, float]                   # λ^M_n at every bus
    energy_price: float                     # λ^M_t  (or marginal energy comp.)
    line_dual: dict[str, float]             # μ^M_m (signed, nonzero if binding)
    flow_own: dict[str, float]              # F^M_m  (own injections only)
    total_cost: float
    status: str


def solve_engine_dispatch(
    pt: PTDFData,
    engine: MarketEngine,
    exo: dict[str, float] | None = None,
    flow_offsets: dict[str, float] | None = None,
) -> EngineResult:
    """Clear one engine's DC-OPF on the shared network.

    Parameters
    ----------
    pt : PTDFData
        Shared shift factors.
    engine : MarketEngine
        Resource/load/activated-constraint partition.
    exo : dict, optional
        ``{bus: MW}`` price-taking exogenous injections (positive = injection
        into this engine's footprint). This is how a neighbouring engine's
        export shows up: a fixed, price-insensitive schedule (paper §2.3, §4.2).
    flow_offsets : dict, optional
        ``{line: MW}`` signed flow (in each line's reference direction) that
        this engine *accommodates* on its activated limits: the limit is
        enforced on ``F^M_m + offset_m`` rather than on ``F^M_m`` alone, so the
        engine leaves room for another party's anticipated flow. The offset is
        exogenous to the optimisation (an ATC-style reservation), not a
        decision variable. Default ``None`` leaves the limits unchanged.

    Returns
    -------
    EngineResult
        Dispatch, the LMP at *every* bus (eq. 9), the energy price, signed line
        congestion duals, and the engine's own flow component (eq. 6).
    """
    exo = {str(k): float(v) for k, v in (exo or {}).items()}
    offsets = {str(k): float(v) for k, v in (flow_offsets or {}).items()}
    gen_ids = list(engine.gens)
    G = len(gen_ids)
    if G == 0:
        raise ValueError(f"Engine {engine.name!r} has no generators.")

    cost = np.array([engine.gens[g]["cost"] for g in gen_ids])
    cap = np.array([engine.gens[g]["p_nom"] for g in gen_ids])
    gen_bus = [str(engine.gens[g]["bus"]) for g in gen_ids]

    # Fixed injection at each bus from loads (−) and exogenous schedules (+)
    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw

    total_load = load_vec.sum()
    total_exo = exo_vec.sum()

    # ── Energy balance:  Σ_g g = total_load − total_exo ──────────────────
    A_eq = np.ones((1, G))
    b_eq = np.array([total_load - total_exo])

    # ── Activated line limits ────────────────────────────────────────────
    if engine.activated_lines == "all":
        act = list(range(pt.n_line))
    else:
        act = [pt.line_idx[str(l)] for l in engine.activated_lines]

    # F_l = Σ_g PTDF[l, bus_g] g_g + Ffix_l,  Ffix_l = Σ_n PTDF[l,n](exo_n − load_n)
    Ffix = pt.ptdf @ (exo_vec - load_vec)
    gen_sf = np.array([[pt.ptdf[l, pt.bus_idx[gb]] for gb in gen_bus] for l in act]) \
        if act else np.zeros((0, G))

    A_ub, b_ub = [], []
    for row, l in enumerate(act):
        off = offsets.get(pt.lines[l], 0.0)            # accommodated flow (signed)
        A_ub.append(gen_sf[row])                       # +(flow+off) ≤ F̄ − Ffix
        b_ub.append(pt.s_nom[l] - Ffix[l] - off)
        A_ub.append(-gen_sf[row])                      # −(flow+off) ≤ F̄ + Ffix
        b_ub.append(pt.s_nom[l] + Ffix[l] + off)
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    res = linprog(
        cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, c) for c in cap], method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Engine {engine.name!r} infeasible: {res.message}")

    g_opt = res.x
    dispatch = {gid: float(g_opt[i]) for i, gid in enumerate(gen_ids)}

    # ── Dual decomposition → nodal LMP (eq. 9) ───────────────────────────
    # HiGHS marginals follow ∂(objective)/∂(rhs):
    #   energy component  = eqlin marginal (positive = marginal $/MWh)
    #   line congestion   = Σ_l PTDF[l,n] · (m_upper_l − m_lower_l)
    energy_price = float(res.eqlin.marginals[0])
    m = res.ineqlin.marginals if (A_ub is not None) else np.array([])

    line_dual = {pt.lines[l]: 0.0 for l in range(pt.n_line)}
    cong = np.zeros(pt.n_bus)
    for row, l in enumerate(act):
        mu = float(m[2 * row] - m[2 * row + 1])        # signed congestion dual
        line_dual[pt.lines[l]] = mu
        cong += pt.ptdf[l] * mu
    lmp = {pt.buses[n]: energy_price + cong[n] for n in range(pt.n_bus)}

    # ── Engine's own flow component F^M_m (eq. 6) ────────────────────────
    gen_by_bus: dict[str, float] = {}
    for i, gid in enumerate(gen_ids):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + g_opt[i]
    inj = np.zeros(pt.n_bus)
    for b, mw in gen_by_bus.items():
        inj[pt.bus_idx[b]] += mw
    inj += exo_vec - load_vec
    flow_own = {pt.lines[l]: float((pt.ptdf[l] @ inj)) for l in range(pt.n_line)}

    return EngineResult(
        name=engine.name,
        dispatch=dispatch,
        gen_by_bus=gen_by_bus,
        load_by_bus={str(b): float(v) for b, v in engine.loads.items()},
        exo_by_bus=dict(exo),
        injection=inj,
        lmp=lmp,
        energy_price=energy_price,
        line_dual=line_dual,
        flow_own=flow_own,
        total_cost=float(cost @ g_opt),
        status=res.message,
    )


# ──────────────────────────────────────────────────────────────────────────
# Superposition of engine flows  (eq. 30):  F^phys = Σ_M F^M + F^non
# ──────────────────────────────────────────────────────────────────────────
def physical_flows(
    pt: PTDFData,
    results: list[EngineResult],
    non_market: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine engine flow components into the physical flow on every line.

    Returns a DataFrame indexed by line with each engine's own component, the
    non-market residual, the physical total, the rating, and an overload flag.
    The physical flow is computed from the *summed* injection vector, which is
    identical to summing the per-engine flow components (DC superposition).
    """
    p_total = np.zeros(pt.n_bus)
    cols = {}
    for r in results:
        p_total += r.injection
        cols[r.name] = [r.flow_own[l] for l in pt.lines]

    non = np.zeros(pt.n_bus)
    if non_market:
        for b, mw in non_market.items():
            non[pt.bus_idx[str(b)]] += mw
        p_total += non
    f_non = pt.ptdf @ non

    f_phys = pt.ptdf @ p_total

    df = pd.DataFrame(index=pt.lines)
    df.index.name = "line"
    df["from"] = [b0 for b0, _ in pt.line_buses]
    df["to"] = [b1 for _, b1 in pt.line_buses]
    for name, vals in cols.items():
        df[f"F[{name}]"] = np.round(vals, 1)
    if non_market:
        df["F[non-market]"] = np.round(f_non, 1)
    df["F_phys"] = np.round(f_phys, 1)
    df["rating"] = pt.s_nom
    df["loading_%"] = np.round(100 * np.abs(f_phys) / pt.s_nom, 0)
    df["overload"] = np.abs(f_phys) > pt.s_nom + 1e-6
    return df


def to_supply_demand(
    engine: MarketEngine, result: EngineResult
) -> tuple[dict, dict]:
    """Adapt an engine + result to ``nodal_plot`` ``supply_by_bus``/``demand_by_bus``.

    Lets the engine results be drawn with the existing ``plot_network_topology``
    / ``plot_nodal_circlize`` helpers (merit-order staircase + LMP-height loads).
    """
    supply_by_bus: dict[str, list] = {}
    for gid, spec in engine.gens.items():
        bus = str(spec["bus"])
        supply_by_bus.setdefault(bus, []).append({
            "unit_id": gid,
            "price": spec["cost"],
            "volume": spec["p_nom"],
            "capacity": spec["p_nom"],
            "accepted_volume": result.dispatch.get(gid, 0.0),
        })
    for bus in supply_by_bus:
        supply_by_bus[bus].sort(key=lambda g: g["price"])
    demand_by_bus = {str(b): float(v) for b, v in engine.loads.items() if v > 0}
    return supply_by_bus, demand_by_bus


def seam_dual_gap(results: list[EngineResult], buses: list[str]) -> pd.DataFrame:
    """Pairwise seam dual gap Δλ^{M,M'}_n at each bus (Proposition 1 / eq. 18)."""
    df = pd.DataFrame(index=buses)
    df.index.name = "bus"
    for r in results:
        df[f"λ[{r.name}]"] = [round(r.lmp[b], 2) for b in buses]
    names = [r.name for r in results]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df[f"Δλ[{a}→{b}]"] = (df[f"λ[{b}]"] - df[f"λ[{a}]"]).round(2)
    return df
