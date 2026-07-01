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
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds


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
    interchange_mw: float | None = None     # E (net export of the interchange bus set)
    interchange_dual: float | None = None   # μ_T (signed, nonzero if the limit binds)
    interchange_limit: float | None = None  # Ē (the scheduling limit enforced)
    shed_by_bus: dict[str, float] = field(default_factory=dict)  # u_n (unserved load; empty unless shed_price set)
    demand_cleared: dict[str, float] = field(default_factory=dict)  # price-sensitive demand bids: bid_id -> MW served


def solve_engine_dispatch(
    pt: PTDFData,
    engine: MarketEngine,
    exo: dict[str, float] | None = None,
    flow_offsets: dict[str, float] | None = None,
    interchange: tuple | None = None,
    shed_price: float | None = None,
    demand_bids: list[dict] | None = None,
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
    interchange : (buses, limit), optional
        A net-interchange scheduling limit (an EDAM-style transfer constraint):
        ``buses`` is one footprint's bus set, and the engine's net export out
        of it, ``E = Σ_{i: bus(i)∈buses} g_i + Σ_{n∈buses} (exo_n − d_n)``
        (lossless DC: identical to the summed tie flow across the cutset), is
        held within ``±limit``. The signed dual is returned as
        ``interchange_dual``, ``E`` as ``interchange_mw``, and every bus inside
        ``buses`` carries the same ``+μ_T`` term in ``lmp`` — the per-footprint
        energy-price separation a transfer constraint creates. Default ``None``
        adds nothing.
    shed_price : float, optional
        Power-balance relaxation penalty, $/MWh (load shedding). When set, an
        unserved-load variable ``0 ≤ u_n ≤ d_n`` is added at every load bus and
        priced at ``shed_price`` in the objective, so the clearing stays
        feasible when generation cannot reach load (e.g. behind a binding
        constraint). Mechanically ``u_n`` is a virtual generator at the load
        bus with cost ``shed_price``: nothing sheds while cheaper supply can
        reach the bus, and where shedding is interior the bus LMP equals
        ``shed_price`` (the penalty becomes the price cap). Mirrors the EDAM
        power-balance relaxation (per-BAA energy-supply-shortfall variables at
        penalty costs — Draft Technical Description §5.5/§11). Shed quantities
        return as ``shed_by_bus``; ``load_by_bus`` stays the NOMINAL load
        (served = load − shed). ``total_cost`` remains the production cost of
        dispatch only (the penalty term is not included). Default ``None``
        disables the relaxation (an unservable engine raises, as before).
    demand_bids : list of dict, optional
        Price-sensitive demand (the mirror of a supply bid): each entry
        ``{'id', 'bus', 'mw', 'price'}`` is a willingness-to-pay block —
        up to ``mw`` MW withdrawn at ``bus`` that clears only while the bus
        LMP is at or below ``price``. This is how an *economic export* is bid
        at an intertie: a demand offer whose bid is distinct from any
        generator's marginal cost. Mechanically a block is a variable
        ``0 ≤ q ≤ mw`` with objective coefficient ``−price`` (a benefit), a
        ``−1`` entry in the energy balance (it is load), and a ``−SF`` entry in
        every flow row (a withdrawal). Cleared quantities return as
        ``demand_cleared`` keyed by ``id``; ``total_cost`` stays the supply
        production cost (the demand benefit is not netted in). Default ``None``
        adds no demand bids.

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

    # Load-shed relaxation columns: one unserved-load variable u_n per load bus,
    # structurally a virtual generator at that bus with cost shed_price.
    shed_bus: list[str] = []
    shed_cap_l: list[float] = []
    if shed_price is not None:
        for b, v in engine.loads.items():
            if float(v) > 0:
                shed_bus.append(str(b))
                shed_cap_l.append(float(v))
    K = len(shed_bus)
    shed_cap = np.array(shed_cap_l)

    # Price-sensitive demand bids (the mirror of a supply bid): variable q ∈ [0, mw]
    # at a bus, a benefit (−price) in the objective, load (−1) in the balance, and a
    # withdrawal (−SF) in the flow rows. This is how an economic export bids at a tie.
    dem_ids: list[str] = []
    dem_bus: list[str] = []
    dem_cap_l: list[float] = []
    dem_price_l: list[float] = []
    for j, db in enumerate(demand_bids or []):
        if float(db["mw"]) <= 0:
            continue
        dem_ids.append(str(db.get("id", f"demand_{j}")))
        dem_bus.append(str(db["bus"]))
        dem_cap_l.append(float(db["mw"]))
        dem_price_l.append(float(db["price"]))
    D = len(dem_ids)
    dem_cap = np.array(dem_cap_l)

    # Fixed injection at each bus from loads (−) and exogenous schedules (+)
    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw

    total_load = load_vec.sum()
    total_exo = exo_vec.sum()

    # ── Energy balance:  Σ_g g + Σ_n u_n − Σ_j q_j = total_load − total_exo ──
    # gens (+1) and shed (+1) supply; demand bids (−1) are extra load served.
    A_eq = np.concatenate([np.ones((1, G + K)), -np.ones((1, D))], axis=1)
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
    shed_sf = np.array([[pt.ptdf[l, pt.bus_idx[b]] for b in shed_bus] for l in act]) \
        if act else np.zeros((0, K))
    # demand bids withdraw, so their flow contribution is −SF (a negative injection)
    dem_sf = np.array([[-pt.ptdf[l, pt.bus_idx[b]] for b in dem_bus] for l in act]) \
        if act else np.zeros((0, D))

    A_ub, b_ub = [], []
    for row, l in enumerate(act):
        off = offsets.get(pt.lines[l], 0.0)            # accommodated flow (signed)
        row_sf = np.concatenate([gen_sf[row], shed_sf[row], dem_sf[row]])
        A_ub.append(row_sf)                            # +(flow+off) ≤ F̄ − Ffix
        b_ub.append(pt.s_nom[l] - Ffix[l] - off)
        A_ub.append(-row_sf)                           # −(flow+off) ≤ F̄ + Ffix
        b_ub.append(pt.s_nom[l] + Ffix[l] + off)

    # ── Net-interchange / transfer-path constraint:  −Ē ≤ E ≤ Ē ─────────
    # ``interchange[0]`` is either a footprint BUS set (E = its net export; the
    # weight on each member bus is 1) or a transport-layer PATH given as the line
    # cutset it crosses — a list of line names, or a ``{line: ±1 orientation}``
    # dict — for which E = Σ_{m∈cutset} s_m F_m and the per-bus weight is the cutset
    # shift factor w_n = Σ_m s_m SF_{n,m}. Both reduce to one linear constraint on
    # the injections, Σ_n w_n p_n; a footprint's own boundary cutset reproduces its
    # net export exactly. The line limits stay enforced underneath either way.
    w_vec = a_ix = e_fix = None
    if interchange is not None:
        spec, ix_limit = interchange[0], float(interchange[1])
        w_vec = np.zeros(pt.n_bus)
        spec_keys = list(spec)
        if spec_keys and all(str(x) in pt.line_idx for x in spec_keys):
            orient = spec if isinstance(spec, dict) else {l: 1.0 for l in spec_keys}
            for line, s_m in orient.items():
                w_vec += float(s_m) * pt.ptdf[pt.line_idx[str(line)]]
        else:
            for b in spec_keys:
                w_vec[pt.bus_idx[str(b)]] = 1.0
        a_ix = np.array([w_vec[pt.bus_idx[gb]] for gb in gen_bus])
        a_ixu = np.array([w_vec[pt.bus_idx[b]] for b in shed_bus])
        a_ixd = np.array([-w_vec[pt.bus_idx[b]] for b in dem_bus])
        a_row = np.concatenate([a_ix, a_ixu, a_ixd])    # shed raises net export; a demand bid lowers it
        e_fix = float(w_vec @ (exo_vec - load_vec))
        A_ub.append(a_row)                             # +E ≤ Ē  (E = a·g + a_u·u + e_fix)
        b_ub.append(ix_limit - e_fix)
        A_ub.append(-a_row)                            # −E ≤ Ē
        b_ub.append(ix_limit + e_fix)

    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    c_vec = np.concatenate([
        cost,
        np.full(K, float(shed_price)) if K else np.zeros(0),
        -np.array(dem_price_l) if D else np.zeros(0),   # demand bid: benefit = −price
    ])
    bounds = ([(0, c) for c in cap] + [(0, c) for c in shed_cap]
              + [(0, c) for c in dem_cap])
    res = linprog(
        c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Engine {engine.name!r} infeasible: {res.message}")

    g_opt = res.x[:G]
    u_opt = res.x[G:G + K]
    d_opt = res.x[G + K:G + K + D]
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

    # Interchange / transfer-path dual: shifts every bus by its weight × μ_T.
    interchange_mw = interchange_dual = interchange_limit = None
    if interchange is not None:
        ix_row = 2 * len(act)
        mu_t = float(m[ix_row] - m[ix_row + 1])
        cong += w_vec * mu_t
        interchange_mw = float(a_ix @ g_opt + (a_ixu @ u_opt if K else 0.0)
                               + (a_ixd @ d_opt if D else 0.0) + e_fix)
        interchange_dual = mu_t
        interchange_limit = ix_limit

    lmp = {pt.buses[n]: energy_price + cong[n] for n in range(pt.n_bus)}

    # ── Engine's own flow component F^M_m (eq. 6) ────────────────────────
    gen_by_bus: dict[str, float] = {}
    for i, gid in enumerate(gen_ids):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + g_opt[i]
    inj = np.zeros(pt.n_bus)
    for b, mw in gen_by_bus.items():
        inj[pt.bus_idx[b]] += mw
    shed_vec = np.zeros(pt.n_bus)
    for b, u in zip(shed_bus, u_opt):
        shed_vec[pt.bus_idx[b]] += u
    dem_vec = np.zeros(pt.n_bus)
    for b, q in zip(dem_bus, d_opt):
        dem_vec[pt.bus_idx[b]] += q
    inj += exo_vec - load_vec + shed_vec - dem_vec   # served load (d−u) plus cleared demand bids
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
        interchange_mw=interchange_mw,
        interchange_dual=interchange_dual,
        interchange_limit=interchange_limit,
        shed_by_bus={b: float(u) for b, u in zip(shed_bus, u_opt) if u > 1e-9},
        demand_cleared={i: float(q) for i, q in zip(dem_ids, d_opt) if q > 1e-9},
    )


def solve_engine_qp(
    pt: PTDFData,
    engine: MarketEngine,
    curves: dict[str, tuple] | None = None,
    exo: dict[str, float] | None = None,
    interchange: tuple | None = None,
    shed_price: float | None = None,
    binding_lines: list[str] | None = None,
    tol: float = 1e-3,
) -> EngineResult:
    """Clear one engine with **continuous linear marginal-cost curves** (a convex QP).

    The flat-LP ``solve_engine_dispatch`` gives every generator a single offer
    price; this variant lets a unit's marginal cost *rise with output* along a
    true straight line — no block staircase. For ``curves[g] = (a, b)`` the unit's
    cost is ``a·g + ½·b·g²`` so its marginal cost is ``MC_g(g) = a + b·g``; a unit
    absent from ``curves`` keeps its flat engine cost (``b = 0``, ``a = cost``).

    Everything else — the shared PTDF, the activated line limits, the ``exo``
    schedule, the interchange constraint, the load-shed relaxation, and the LMP
    decomposition ``λ_n = λ + Σ_m SF_{n,m} μ_m + 𝟙{n∈set}·μ_T`` — is identical to
    ``solve_engine_dispatch`` and the returned ``EngineResult`` carries the same
    fields and sign conventions, so every downstream figure/ledger reads it
    unchanged. With ``b = 0`` for all units it reproduces the flat LP exactly.

    Because there is no QP backend in the environment, the strictly-convex primal
    is solved with ``scipy.optimize.minimize(method="trust-constr")`` and the duals
    are recovered analytically from KKT stationarity at the *interior* generators:
    for any unit with ``0 < g_g < p_nom`` and bound multipliers zero,
    ``MC_g(g_g) = λ + Σ_m SF_{bus(g),m} μ_m + w_{bus(g)} μ_T``. With one balance
    dual, the binding-line duals, and (if it binds) the interchange dual as
    unknowns, the interior-gen equations pin them by least squares. Pass
    ``binding_lines`` (e.g. the active set from a companion flat solve) to fix the
    dual-carrying line set when the unconstrained dispatch is degenerate.

    Parameters mirror ``solve_engine_dispatch``; ``curves`` and ``binding_lines``
    are the only additions. ``demand_bids`` are not supported here.
    """
    curves = curves or {}
    exo = {str(k): float(v) for k, v in (exo or {}).items()}
    gen_ids = list(engine.gens)
    G = len(gen_ids)
    if G == 0:
        raise ValueError(f"Engine {engine.name!r} has no generators.")
    gen_bus = [str(engine.gens[g]["bus"]) for g in gen_ids]
    cap = np.array([engine.gens[g]["p_nom"] for g in gen_ids])
    a = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[0] for g in gen_ids])
    b = np.array([curves.get(g, (engine.gens[g]["cost"], 0.0))[1] for g in gen_ids])

    # Load-shed columns: virtual flat gens at the load buses, cost shed_price.
    shed_bus: list[str] = []
    shed_cap_l: list[float] = []
    if shed_price is not None:
        for bb, v in engine.loads.items():
            if float(v) > 0:
                shed_bus.append(str(bb))
                shed_cap_l.append(float(v))
    K = len(shed_bus)
    shed_cap = np.array(shed_cap_l) if K else np.zeros(0)

    load_vec = np.zeros(pt.n_bus)
    for bus, mw in engine.loads.items():
        load_vec[pt.bus_idx[str(bus)]] += mw
    exo_vec = np.zeros(pt.n_bus)
    for bus, mw in exo.items():
        exo_vec[pt.bus_idx[bus]] += mw
    total_load = load_vec.sum()
    total_exo = exo_vec.sum()

    if engine.activated_lines == "all":
        act = list(range(pt.n_line))
    else:
        act = [pt.line_idx[str(l)] for l in engine.activated_lines]
    Ffix = pt.ptdf @ (exo_vec - load_vec)
    gen_sf = np.array([[pt.ptdf[l, pt.bus_idx[gb]] for gb in gen_bus] for l in act]) \
        if act else np.zeros((0, G))
    shed_sf = np.array([[pt.ptdf[l, pt.bus_idx[bb]] for bb in shed_bus] for l in act]) \
        if act else np.zeros((0, K))

    N = G + K
    H = np.zeros((N, N))
    H[np.arange(G), np.arange(G)] = b
    c_lin = np.concatenate([a, np.full(K, float(shed_price)) if K else np.zeros(0)])

    cons = [LinearConstraint(np.concatenate([np.ones(G), np.ones(K)]).reshape(1, -1),
                             total_load - total_exo, total_load - total_exo)]
    if act:
        rows, lb, ub = [], [], []
        for r, l in enumerate(act):
            rows.append(np.concatenate([gen_sf[r], shed_sf[r]]))
            lb.append(-pt.s_nom[l] - Ffix[l])
            ub.append(pt.s_nom[l] - Ffix[l])
        cons.append(LinearConstraint(np.array(rows), np.array(lb), np.array(ub)))

    w_vec = a_row_full = e_fix = None
    if interchange is not None:
        spec, ix_limit = interchange[0], float(interchange[1])
        w_vec = np.zeros(pt.n_bus)
        spec_keys = list(spec)
        if spec_keys and all(str(x) in pt.line_idx for x in spec_keys):
            orient = spec if isinstance(spec, dict) else {l: 1.0 for l in spec_keys}
            for line, s_m in orient.items():
                w_vec += float(s_m) * pt.ptdf[pt.line_idx[str(line)]]
        else:
            for bb in spec_keys:
                w_vec[pt.bus_idx[str(bb)]] = 1.0
        a_ix = np.array([w_vec[pt.bus_idx[gb]] for gb in gen_bus])
        a_ixu = np.array([w_vec[pt.bus_idx[bb]] for bb in shed_bus]) if K else np.zeros(0)
        a_row_full = np.concatenate([a_ix, a_ixu])
        e_fix = float(w_vec @ (exo_vec - load_vec))
        cons.append(LinearConstraint(a_row_full.reshape(1, -1),
                                     -ix_limit - e_fix, ix_limit - e_fix))

    bounds = Bounds(np.zeros(N), np.concatenate([cap, shed_cap]))
    x0 = np.minimum(bounds.ub, np.maximum(bounds.lb, np.full(N, (total_load - total_exo) / max(N, 1))))
    res = minimize(lambda x: 0.5 * x @ (H @ x) + c_lin @ x, x0, method="trust-constr",
                   jac=lambda x: H @ x + c_lin, hess=lambda x: H,
                   constraints=cons, bounds=bounds,
                   options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 2000})
    x = res.x
    g_opt = x[:G]
    u_opt = x[G:G + K]

    # ── Dual recovery from interior-gen stationarity ─────────────────────
    interior = [i for i in range(G) if tol < g_opt[i] < cap[i] - tol]
    if binding_lines is not None:
        binding = [pt.line_idx[str(l)] for l in binding_lines]
    else:
        binding = [l for r, l in enumerate(act)
                   if abs((gen_sf[r] @ g_opt + (shed_sf[r] @ u_opt if K else 0.0)) + Ffix[l])
                   > pt.s_nom[l] - 1e-4]
    ix_binds = False
    if interchange is not None:
        E = float(a_row_full @ x + e_fix)
        ix_binds = abs(abs(E) - interchange[1]) < 1e-4
    A, rhs = [], []
    for i in interior:
        rowc = [1.0] + [pt.ptdf[l, pt.bus_idx[gen_bus[i]]] for l in binding]
        if ix_binds:
            rowc.append(w_vec[pt.bus_idx[gen_bus[i]]])
        A.append(rowc)
        rhs.append(a[i] + b[i] * g_opt[i])
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(rhs), rcond=None)
    lam = float(sol[0])
    mu = {l: float(sol[1 + k]) for k, l in enumerate(binding)}
    mu_T = float(sol[1 + len(binding)]) if ix_binds else 0.0

    line_dual = {pt.lines[l]: 0.0 for l in range(pt.n_line)}
    cong = np.zeros(pt.n_bus)
    for l in binding:
        line_dual[pt.lines[l]] = mu[l]
        cong += pt.ptdf[l] * mu[l]
    if ix_binds:
        cong += w_vec * mu_T
    lmp = {pt.buses[n]: lam + cong[n] for n in range(pt.n_bus)}

    gen_by_bus: dict[str, float] = {}
    for i in range(G):
        gen_by_bus[gen_bus[i]] = gen_by_bus.get(gen_bus[i], 0.0) + g_opt[i]
    inj = np.zeros(pt.n_bus)
    for bb, mw in gen_by_bus.items():
        inj[pt.bus_idx[bb]] += mw
    shed_vec = np.zeros(pt.n_bus)
    for bb, uu in zip(shed_bus, u_opt):
        shed_vec[pt.bus_idx[bb]] += uu
    inj += exo_vec - load_vec + shed_vec
    flow_own = {pt.lines[l]: float(pt.ptdf[l] @ inj) for l in range(pt.n_line)}

    interchange_mw = interchange_dual = interchange_limit = None
    if interchange is not None:
        interchange_mw = float(a_row_full @ x + e_fix)
        interchange_dual = mu_T
        interchange_limit = interchange[1]

    return EngineResult(
        name=engine.name,
        dispatch={gid: float(g_opt[i]) for i, gid in enumerate(gen_ids)},
        gen_by_bus=gen_by_bus,
        load_by_bus={str(bb): float(v) for bb, v in engine.loads.items()},
        exo_by_bus=dict(exo),
        injection=inj,
        lmp=lmp,
        energy_price=lam,
        line_dual=line_dual,
        flow_own=flow_own,
        total_cost=float(np.sum(a * g_opt + 0.5 * b * g_opt ** 2)),
        status=res.message,
        interchange_mw=interchange_mw,
        interchange_dual=interchange_dual,
        interchange_limit=interchange_limit,
        shed_by_bus={bb: float(uu) for bb, uu in zip(shed_bus, u_opt) if uu > 1e-9},
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
        acc = result.dispatch.get(gid, 0.0)
        # A residual BACKSTOP is a balancing resource, not part of the economic merit order:
        # hide it when idle, and draw only the block it actually balances when it clears (so its
        # large capacity does not stretch the merit-order axis at every load bus).
        is_backstop = gid.startswith("backstop_")
        if is_backstop and acc <= 1e-6:
            continue
        vol = float(acc) if is_backstop else spec["p_nom"]
        # A unit may carry a linear marginal-cost curve ``curve=(a, b)`` (MC = a + b·g,
        # set by the rising-curve QP clearing). Then its merit-order "price" is the
        # marginal offer AT its cleared output (a + b·acc), and the curve params ride
        # along as ``mc0``/``mc_slope`` so ``plot_nodal_circlize`` can draw the supply
        # as a sloped WEDGE under the MC line instead of a flat rectangle.
        curve = spec.get("curve")
        entry = {
            "unit_id": gid,
            "price": spec["cost"] if curve is None else float(curve[0] + curve[1] * acc),
            "volume": vol,
            "capacity": vol,
            "accepted_volume": acc,
        }
        if curve is not None:
            entry["mc0"], entry["mc_slope"] = float(curve[0]), float(curve[1])
        supply_by_bus.setdefault(bus, []).append(entry)
    for bus in supply_by_bus:
        supply_by_bus[bus].sort(key=lambda g: g["price"])
    demand_by_bus = {str(b): float(v) for b, v in engine.loads.items() if v > 0}
    return supply_by_bus, demand_by_bus


def shed_segments(result, demand_by_bus, shed_alpha=0.12):
    """A ``nodal_plot`` ``demand_segments`` dict that renders shed (unserved) load
    as a faint tail: each shed bus's load bar splits into a served segment (bus
    colour at the demand fill) plus an unserved segment in the SAME bus colour at
    the fainter ``shed_alpha`` — the convention used for idle generation capacity.
    Returns ``{}`` when nothing sheds, so ``shed_segments(...) or None`` is a clean
    default.
    """
    segs: dict[str, list] = {}
    for b, u in (result.shed_by_bus or {}).items():
        if u <= 1e-6:
            continue
        served = float(demand_by_bus.get(str(b), 0.0)) - u
        segs[str(b)] = ([{"mw": served}] if served > 0.5 else []) \
            + [{"mw": u, "alpha": shed_alpha}]   # bus colour (inherited), faint
    return segs


def served_by_bus(result, demand_by_bus):
    """``{bus: served MW}`` at the buses that shed, for ``plot_network_topology``'s
    ``demand_served_by_bus`` ('served/total' load annotation). Empty when nothing
    sheds, so the annotation falls back to the plain total elsewhere.
    """
    return {str(b): float(demand_by_bus.get(str(b), 0.0)) - u
            for b, u in (result.shed_by_bus or {}).items() if u > 1e-6}


def served_demand(result, demand_by_bus):
    """Full ``{bus: served MW}`` = demand − shed, for tracing gen→load chords on
    the SERVED dispatch. Passing this (not nominal demand) to
    ``compute_ptdf_flows`` keeps the trace balanced (Σ gen = Σ served), so each
    generator's chords match its dispatch and a shed bus receives only its
    served-MW of incoming chords (the unserved tail stays chord-free).
    """
    shed = result.shed_by_bus or {}
    return {str(b): float(v) - shed.get(str(b), 0.0) for b, v in demand_by_bus.items()}


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
