# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strategic bilateral arbitrage over transmission rights, and its reconciliation.

The behavioral layer that *precedes* a uniform-price nodal market. Under an OATT,
a generator does not bid a marginal-cost offer into a central auction; it holds a
**portfolio of point-to-point transmission rights** (rated by Available Transfer
Capability, see ``atc.py``) and uses them to **arbitrage** between its own bus and a
profitable load center -- buy low at the generator, sell high at the load. Because
this is *not* a uniform-price marginal-cost auction, it is **not incentive
compatible**: the arbitraging agent is strategic and withholds to keep the spread
wide. This is the **no-arbitrage bilateral Cournot model** of Hobbs (2001) and
Metzler, Hobbs & Pang (2003) -- producers price-discriminate across nodes, facing
affine demand, taking transmission prices as given -- with one teaching
simplification: the generator is its OWN arbitrager (it holds the right and
captures the spread), so there is no separate price-equalizing arbitrager and nodal
spreads need not collapse to the transmission charge. (Hobbs's with-arbitrage variant
is still strategic and equals a POOLCO Cournot clearing; it is NOT modelled here. The
``competitive_clear`` benchmark is instead the perfectly competitive, price-taking
efficient dispatch -- the yardstick that exposes the Cournot withholding wedge.)

Three pieces, all on the shared DC shift-factor algebra of ``seams_engine``:

1. **Order book.** Each load center ``k`` posts an affine inverse demand (a
   willingness-to-pay curve) ``p_k(D) = a_k - b_k D``; each generator is a strategic
   supplier with a cost and a capacity. (``LoadCenter``, ``Supplier``.)

2. **The spot iteration.** Each supplier chooses how much to deliver to each load
   center it holds a right to, as a **Cournot best response** to rivals' current
   deliveries, capped by each right's ATC. Sweeping supplier-by-supplier to a fixed
   point is the **Gauss-Seidel diagonalization** that computes the Nash-Cournot
   equilibrium -- the iterative counterpart to solving the monotone mixed LCP whose
   existence and uniqueness Metzler, Hobbs & Pang (2003) establish; the bid-adjustment
   iteration of Cherukuri & Cortes (2017) converges to the same kind of object, and the
   AMES test bed of Tesfatsion is the teaching precedent. The factor of 2 in the best
   response (eq. 3) is the marginal-revenue (Cournot) wedge -- strategic quantity is
   below the competitive quantity. The producer takes the transmission/congestion price
   as given (Hobbs's key LCP assumption), here the optional exogenous term ``w`` (default
   0). (``clear``, ``best_response``.)

3. **Reconciliation against actual power flow.** The decentralized self-schedules
   respect each right's ATC but **not** the simultaneous network limits, so the
   booked set can overload a line. ``power_flow_check`` runs the DC power flow on the
   awarded schedules; if a line is over its rating, ``redispatch`` runs a minimum-cost
   **counter-trading** DC-OPF (curtail the schedule causing the constraint, start up a
   feasible substitute generator) and the **causing arbitrager pays** the uplift --
   the causer-pays-vs-socialized-uplift distinction of real-system schedule
   curtailment and redispatch.

The library reuses ``atc.py`` for rights/feasibility (a right is an ``atc.Award``
path; ATC caps come from ``atc.ttc``/``atc.available_atc``; feasibility from
``atc.simultaneous_feasibility``) and ``seams_engine`` for the competitive DC-OPF
benchmark the strategic equilibrium is measured against. Used by notebook ``111``
(the portfolio-planning step) and ``102`` (the spot iteration + reconciliation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import atc
from seams_engine import MarketEngine, solve_engine_dispatch


# ──────────────────────────────────────────────────────────────────────────
# Order book: load-center inverse demand and strategic suppliers
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class LoadCenter:
    """A load bus with an inelastic forecast ``d_nom`` and a risk-averse premium curve.

    Demand is a fixed forecast ``d_nom`` that must be served. The strategic (cheap) units
    deliver ``Q`` toward it; the rest is filled by a price-taking **backstop** (an
    expensive fringe unit, then a value-of-lost-load shed). The price the strategic firms
    face is therefore an affine **residual inverse demand** in their delivery ``Q``,

        ``p_k(Q) = clamp( a_k - b_k Q,  [p_comp, voll] )``,

    set in ``calibrate_load_centers`` so that at full strategic service ``Q = d_nom`` the
    price is the competitive ``p_comp`` (no premium), and as the cheap units withhold the
    backstop runs and the price rises toward ``voll`` (the value of lost load). ``b_k`` is
    the scarcity slope ``(voll - p_comp)/(firmness*d_nom)``; ``voll`` caps the price.
    """

    bus: str
    a: float
    b: float
    d_nom: float = 0.0
    p_comp: float = 0.0
    voll: float = 1000.0

    def __post_init__(self):
        self.bus = str(self.bus)
        self.a, self.b, self.d_nom = float(self.a), float(self.b), float(self.d_nom)
        self.p_comp, self.voll = float(self.p_comp), float(self.voll)

    def price(self, Q: float) -> float:
        """Clearing price when the strategic units deliver ``Q`` (backstop fills the rest):
        the residual demand ``a - b Q`` clamped to ``[p_comp, voll]``."""
        return float(min(self.voll, max(self.p_comp, self.a - self.b * float(Q))))

    def premium(self, Q: float) -> float:
        """Risk-averse premium over the competitive price at strategic delivery ``Q``."""
        return self.price(Q) - self.p_comp

    def firmness_gap(self) -> float:
        """Strategic shortfall at which the backstop is exhausted and price hits ``voll``:
        ``(a - voll)/b`` below the forecast, i.e. ``d_nom - (a - voll)/b``."""
        return self.d_nom - (self.a - self.voll) / self.b if self.b > 0 else float("inf")


@dataclass
class Supplier:
    """A generator: fixed marginal ``cost`` $/MWh at ``bus``, ``p_nom`` MW cap.

    ``price_risk`` is the generator's own perceived spot-price risk (sigma, $/MWh) for the
    two-sided forward model; ``0`` falls back to the network-wide ``Belief.price_risk``.
    """

    gid: str
    bus: str
    cost: float
    p_nom: float
    price_risk: float = 0.0

    def __post_init__(self):
        self.gid, self.bus = str(self.gid), str(self.bus)
        self.cost, self.p_nom = float(self.cost), float(self.p_nom)
        self.price_risk = float(self.price_risk)


# ──────────────────────────────────────────────────────────────────────────
# Rights and portfolios
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Right:
    """A point-to-point transmission right ``source->sink`` rated at ``cap_mw`` ATC.

    ``planned_mw`` is the quantity the holder intends to schedule on it (set by
    ``plan_portfolio``), ``<= cap_mw``. ``binding_line`` is the line that set the ATC.
    """

    source: str
    sink: str
    cap_mw: float
    binding_line: str = ""
    planned_mw: float = 0.0

    def __post_init__(self):
        self.source, self.sink = str(self.source), str(self.sink)
        self.cap_mw, self.planned_mw = float(self.cap_mw), float(self.planned_mw)

    def award(self, mw: float | None = None) -> atc.Award:
        """As an ``atc.Award`` (``planned_mw`` if ``mw`` not given)."""
        return atc.Award(self.source, self.sink, self.planned_mw if mw is None else mw)


@dataclass
class Portfolio:
    """One supplier's book of rights ``{sink: Right}`` from its bus."""

    gid: str
    bus: str
    rights: list[Right] = field(default_factory=list)

    def cap_to(self, sink: str) -> float:
        """ATC cap of the right to ``sink`` (0 if none held)."""
        for r in self.rights:
            if r.sink == str(sink):
                return r.cap_mw
        return 0.0

    def sinks(self) -> list[str]:
        return [r.sink for r in self.rights]

    def requests(self) -> list[tuple[str, str, float]]:
        """``(source, sink, planned_mw)`` for every right with a planned schedule."""
        return [(r.source, r.sink, r.planned_mw) for r in self.rights if r.planned_mw > 1e-9]


def rate_right(pt, source, sink, etc: float = 0.0, monitored="all") -> Right:
    """ATC-rate one ``source->sink`` right: ``cap = ATC(TTC, etc)`` (``atc.ttc``/``atc.atc``)."""
    ttc_mw, binding = atc.ttc(pt, source, sink, monitored=monitored)
    return Right(str(source), str(sink), atc.atc(ttc_mw, etc=etc), binding or "")


# ──────────────────────────────────────────────────────────────────────────
# Calibration of the order book to the competitive clearing
# ──────────────────────────────────────────────────────────────────────────
def _competitive_lmp(pt, gen_fleet=None, loads=None) -> dict[str, float]:
    """LMP at every bus of the fixed-load DC-OPF (line limits enforced)."""
    import wscc9_model as wm
    fleet = wm.DEFAULT_GEN_FLEET if gen_fleet is None else gen_fleet
    loads = wm.DEFAULT_LOADS if loads is None else loads
    eng = MarketEngine(
        name="BENCH",
        gens={g: dict(s) for g, s in fleet.items()},
        loads={str(b): float(v) for b, v in loads.items()},
        activated_lines="all",
    )
    return solve_engine_dispatch(pt, eng).lmp


def calibrate_load_centers(pt, loads=None, voll: float = 1000.0, firmness: float = 0.5,
                           competitive_lmp=None) -> dict[str, LoadCenter]:
    """Build ``{bus: LoadCenter}`` for the must-serve forecast + risk-averse premium.

    Each center's residual inverse demand is pinned by two anchors: at full strategic
    service ``Q = d_nom`` the price equals the **competitive** ``p_comp`` (the fixed-load
    DC-OPF LMP -- no premium), and as the cheap units withhold to ``Q = d_nom -
    firmness*d_nom`` the backstop is exhausted and the price reaches ``voll``. Hence the
    scarcity slope ``b_k = (voll - p_comp)/(firmness*d_nom)`` and ``a_k = p_comp + b_k*d_nom``.
    ``firmness`` is the share of forecast the price-taking backstop can cover before VOLL
    (ample backstop -> flat residual -> small premium; tight -> steep -> cheap units pivotal).
    """
    import wscc9_model as wm
    loads = wm.DEFAULT_LOADS if loads is None else loads
    lmp = competitive_lmp if competitive_lmp is not None else _competitive_lmp(pt, loads=loads)
    centers = {}
    for bus, d_nom in loads.items():
        bus = str(bus)
        p = float(lmp[bus])
        d = float(d_nom)
        gap = max(1e-6, firmness * d)
        b = (voll - p) / gap
        if b <= 0:                       # voll below competitive (degenerate): tiny positive slope
            b = max(1e-3, p) / max(1.0, d)
        a = p + b * d
        centers[bus] = LoadCenter(bus=bus, a=a, b=b, d_nom=d, p_comp=p, voll=voll)
    return centers


# ──────────────────────────────────────────────────────────────────────────
# 111: least-cost portfolio planning
# ──────────────────────────────────────────────────────────────────────────
def plan_portfolio(pt, supplier: Supplier, load_centers: dict[str, LoadCenter], *,
                   etc: float = 0.0, monitored="all", booked=None) -> Portfolio:
    """Build a supplier's book of rights, most-profitable load center first.

    For each center the **margin** is ``p_k(d_nom) - cost - transmission cost`` (the
    spread the right would capture); the **revenue opportunity** is that margin times
    the schedulable quantity ``min(d_nom_k, ATC)``. Centers with a positive margin are
    ranked by revenue opportunity, high-to-low (the most valuable wheel first), and a
    right is taken to each, scheduled up to ``min(d_nom_k, ATC)``. ``booked`` (a list of
    prior ``atc.Award``) tightens the available capability via ``atc.available_atc`` so
    a cumulative book across suppliers stays consistent; ``booked=None`` rates each
    right standalone (its own TTC). The planned schedule is the request the ATC desk
    then rates.
    """
    prior = list(booked) if booked else []
    scored = []
    for k, lc in load_centers.items():
        r = rate_right(pt, supplier.bus, lc.bus, etc=etc, monitored=monitored)
        margin = lc.price(lc.d_nom) - supplier.cost - etc
        if margin <= 1e-6 or r.cap_mw <= 1e-9:   # a break-even unit holds no arbitrage rights
            continue
        opportunity = margin * min(lc.d_nom, r.cap_mw)
        scored.append((opportunity, k, lc, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    rights: list[Right] = []
    for opportunity, k, lc, r in scored:
        if booked is not None:
            avail = atc.available_atc(pt, supplier.bus, lc.bus, prior, monitored=monitored)
        else:
            avail = r.cap_mw
        qty = min(lc.d_nom, avail)
        if qty <= 1e-6:
            continue
        r.planned_mw = float(qty)
        rights.append(r)
        if booked is not None:
            prior.append(r.award())
    return Portfolio(gid=supplier.gid, bus=supplier.bus, rights=rights)


def plan_all(pt, suppliers: list[Supplier], load_centers: dict[str, LoadCenter], *,
             etc: float = 0.0, monitored="all", share_atc: bool = False) -> dict[str, Portfolio]:
    """``plan_portfolio`` for every supplier.

    With ``share_atc=True`` the suppliers book against a shared, decrementing ATC pool
    (the ``atc.book_sequentially`` idiom); the default plans each supplier standalone.
    """
    booked: list[atc.Award] | None = [] if share_atc else None
    out: dict[str, Portfolio] = {}
    for s in suppliers:
        pf = plan_portfolio(pt, s, load_centers, etc=etc, monitored=monitored, booked=booked)
        out[s.gid] = pf
        if share_atc:
            booked.extend(r.award() for r in pf.rights)
    return out


# ──────────────────────────────────────────────────────────────────────────
# The strategic spot iteration (Gauss-Seidel best response)
# ──────────────────────────────────────────────────────────────────────────
def best_response(supplier: Supplier, portfolio: Portfolio,
                  load_centers: dict[str, LoadCenter],
                  rival_delivery: dict[str, float], *, w=None) -> dict[str, float]:
    """A supplier's Cournot best response over the load centers it holds rights to.

    Solves ``max_q sum_k [p_k(q_k + Q_{-g,k}) - cost - w_{g,k}] q_k`` over ``q_k`` in
    ``[0, ATC_k]`` with the unit cap ``sum_k q_k <= p_nom`` and ``Q_{-g,k}`` the rivals'
    fixed deliveries. The unconstrained interior optimum is ``q_k = (a_k - cost -
    w_{g,k} - b_k Q_{-g,k}) / (2 b_k)`` (eq. 3; the factor 2 is the Cournot wedge); the
    box is applied by clipping, and the unit cap by water-filling a common marginal-
    profit shadow price ``lam`` so ``sum_k q_k = p_nom`` when it binds.
    """
    w = w or {}
    sinks = portfolio.sinks()

    def q_at(lam: float) -> dict[str, float]:
        out = {}
        for k in sinks:
            lc = load_centers[k]
            Qm = float(rival_delivery.get(k, 0.0))
            wk = float(w.get((supplier.gid, k), 0.0))   # congestion charge, keyed (gid, sink) as in clear()
            qk = (lc.a - supplier.cost - wk - lc.b * Qm - lam) / (2 * lc.b)
            out[k] = float(min(max(0.0, qk), portfolio.cap_to(k)))
        return out

    q = q_at(0.0)                                   # interior, box-clipped
    if sum(q.values()) <= supplier.p_nom + 1e-9:
        return q
    # unit cap binds: raise lam until sum_k q_k = p_nom (q decreases monotonically in lam)
    lo, hi = 0.0, max(lc.a for lc in load_centers.values())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sum(q_at(mid).values()) > supplier.p_nom:
            lo = mid
        else:
            hi = mid
    return q_at(hi)


@dataclass
class BilateralResult:
    """The strategic equilibrium of the spot iteration (load served in full)."""

    deliveries: dict[tuple[str, str], float]      # (gid, sink) -> strategic MW
    center_price: dict[str, float]                # sink -> clearing price (VOLL-capped)
    center_demand: dict[str, float]               # sink -> strategic delivery Q_strat_k
    premium: dict[str, float]                      # sink -> price - competitive
    served: dict[str, float]                      # sink -> MW served (forecast minus shed)
    fringe: dict[str, float]                      # sink -> MW filled by the backstop
    shed: dict[str, float]                        # sink -> MW shed at VOLL (scarcity tail)
    self_sched: dict[str, float]                  # gid -> total MW produced (incl. fringe)
    margins: dict[tuple[str, str], float]         # (gid, sink) -> premium-bearing spread
    profit: dict[str, float]                      # gid -> arbitrage rent (strategic units)
    awards: list[atc.Award]                       # strategic + fringe deliveries
    loadings: pd.DataFrame
    trace: list[dict]
    sweeps: int
    converged: bool


def clear(pt, strategic: list[Supplier], portfolios: dict[str, Portfolio],
          load_centers: dict[str, LoadCenter], fringe: Supplier, *, tol: float = 1e-3,
          max_sweeps: int = 50, order=None, jacobi: bool = False, damp: float = 1.0,
          monitored="all") -> BilateralResult:
    """Gauss-Seidel diagonalization to the strategic equilibrium, with load served in full.

    The ``strategic`` (cheap) units play ``best_response`` against the residual demand,
    sweeping to a fixed point (Gauss-Seidel; ``jacobi`` freezes rivals each sweep, ``damp``
    blends). Each center's forecast ``d_nom`` is then **served in full**: the strategic
    delivery ``Q_strat_k`` plus the ``fringe`` backstop filling the gap up to the firmness
    gap ``G_k``, and a value-of-lost-load **shed** only beyond that (the scarcity tail). The
    clearing price is the residual demand at ``Q_strat_k``, capped at VOLL -- so the rent is
    a price **premium** above competitive, not unserved load. Network feasibility of the
    combined (strategic + fringe) schedule is reconciled afterwards by ``backstop_clear``.
    """
    sup = {s.gid: s for s in strategic}
    order = list(order) if order else [s.gid for s in strategic]
    deliv: dict[tuple[str, str], float] = {
        (g, k): 0.0 for g in order for k in portfolios[g].sinks()
    }
    trace: list[dict] = []
    converged = False
    sweeps = 0

    for t in range(1, max_sweeps + 1):
        sweeps = t
        base = dict(deliv) if jacobi else deliv      # jacobi reads the frozen snapshot
        max_step = 0.0
        for g in order:
            pf = portfolios[g]
            rival = {k: sum(base[(h, k)] for h in order if h != g and (h, k) in base)
                     for k in pf.sinks()}
            br = best_response(sup[g], pf, load_centers, rival)
            for k, qk in br.items():
                new = damp * qk + (1 - damp) * deliv[(g, k)]
                max_step = max(max_step, abs(new - deliv[(g, k)]))
                deliv[(g, k)] = new
        D = _center_demand(deliv, load_centers)
        trace.append({"sweep": t, "max_step": float(max_step),
                      **{f"D_{k}": float(v) for k, v in D.items()}})
        if max_step < tol:
            converged = True
            break

    D = _center_demand(deliv, load_centers)
    price = {k: load_centers[k].price(D[k]) for k in load_centers}
    premium = {k: load_centers[k].premium(D[k]) for k in load_centers}

    # Serve the forecast in full: strategic Q, then the fringe backstop, then VOLL shed.
    gap = {k: max(0.0, load_centers[k].d_nom - D[k]) for k in load_centers}
    Gk = {k: max(0.0, load_centers[k].firmness_gap()) for k in load_centers}
    fringe_fill = {k: min(gap[k], Gk[k]) for k in load_centers}
    # cap the backstop unit's total output; any overflow becomes shed
    tot_fr = sum(fringe_fill.values())
    if tot_fr > fringe.p_nom + 1e-9 and tot_fr > 0:
        scale = fringe.p_nom / tot_fr
        fringe_fill = {k: v * scale for k, v in fringe_fill.items()}
    shed = {k: gap[k] - fringe_fill[k] for k in load_centers}
    served = {k: load_centers[k].d_nom - shed[k] for k in load_centers}

    margins = {(g, k): price[k] - sup[g].cost for (g, k) in deliv}
    profit = {g: sum(margins[(g, k)] * deliv[(g, k)] for k in portfolios[g].sinks())
              for g in order}
    self_sched = {g: sum(deliv[(g, k)] for k in portfolios[g].sinks()) for g in order}
    self_sched[fringe.gid] = sum(fringe_fill.values())

    awards = [atc.Award(sup[g].bus, k, mw) for (g, k), mw in deliv.items() if mw > 1e-9]
    awards += [atc.Award(fringe.bus, k, mw) for k, mw in fringe_fill.items() if mw > 1e-9]
    loadings = atc.line_loadings(pt, awards, monitored=monitored)
    return BilateralResult(
        deliveries=deliv, center_price=price, center_demand=D, premium=premium,
        served=served, fringe=fringe_fill, shed=shed, self_sched=self_sched,
        margins=margins, profit=profit, awards=awards, loadings=loadings,
        trace=trace, sweeps=sweeps, converged=converged,
    )


def _center_demand(deliv, load_centers) -> dict[str, float]:
    D = {k: 0.0 for k in load_centers}
    for (g, k), mw in deliv.items():
        if k in D:
            D[k] += mw
    return D


# ──────────────────────────────────────────────────────────────────────────
# Reconciliation: DC power-flow feasibility and counter-trading redispatch
# ──────────────────────────────────────────────────────────────────────────
def power_flow_check(pt, awards, monitored="all") -> tuple[bool, pd.DataFrame]:
    """DC power-flow feasibility of the self-scheduled awards (``atc.simultaneous_feasibility``).

    Returns ``(feasible, loadings_table)`` -- the equilibrium book may respect every
    right's ATC yet overload a line once the schedules superpose on the real network.
    """
    return atc.simultaneous_feasibility(pt, awards, monitored=monitored)


@dataclass
class BackstopResult:
    """Network reconciliation of the strategic + backstop schedule, with a VOLL shed valve."""

    dispatch: dict[str, float]                # gid -> post-reconciliation MW
    up: dict[str, float]                      # gid -> MW incremented (backstop ramps up)
    down: dict[str, float]                    # gid -> MW curtailed (strategic causer)
    shed: dict[str, float]                    # center bus -> MW shed at VOLL (last resort)
    redispatch_cost: float                    # production-cost uplift (excl. shed penalty)
    charge_by_arb: dict[str, float]           # gid -> causer-pays charge (bearer='causer')
    tsp_charge: float                         # whole uplift to the rating TSP (bearer='tsp')
    feasible_before: bool
    feasible_after: bool
    loadings: pd.DataFrame


def backstop_clear(pt, suppliers: list[Supplier], result: BilateralResult, *,
                   voll: float = 1000.0, buyback=None, monitored="all",
                   bearer: str = "causer") -> BackstopResult:
    """Reconcile the combined (strategic + backstop) schedule to a feasible power flow.

    The strategic self-schedules plus the backstop's gap-fill (``result.awards`` /
    ``result.self_sched``) respect each right's ATC but can overload a line. This is the
    minimum-cost counter-trade that restores feasibility::

        min  sum_g (cost_g up_g + price_g down_g) + voll * sum_n shed_n
        s.t. sum_g (up_g - down_g) + sum_n shed_n = 0            (served load adjusts by shed)
             |Ffix_l + sum_g SF[l,g](up_g - down_g) + sum_n SF[l,n] shed_n| <= F_bar_l
             0 <= up_g <= p_nom_g - s_g,  0 <= down_g <= s_g,  0 <= shed_n <= served_n

    Both supply legs are paid (ramp the backstop up at its offer, buy back ``price_g`` of
    the curtailed schedule, default the unit's own cost; override via ``buyback={gid:$}``),
    so a feasible schedule is a no-op. Curtailing the strategic schedule that loads the
    constrained line and ramping the backstop relieves it; **shed at VOLL** is the last
    resort if even that cannot deliver. The production-cost uplift is charged to the
    curtailed (causing) units (causer-pays).
    """
    feas_before, _ = power_flow_check(pt, result.awards, monitored=monitored)

    sup = list(suppliers)
    G = len(sup)
    s = np.array([result.self_sched.get(x.gid, 0.0) for x in sup])
    cap = np.array([x.p_nom for x in sup])
    cost = np.array([x.cost for x in sup])
    price = np.array([(buyback or {}).get(x.gid, x.cost) for x in sup], dtype=float)
    busg = [x.bus for x in sup]

    centers = list(result.served)
    C = len(centers)
    shed_cap = np.array([result.served[c] for c in centers]) if C else np.zeros(0)

    Ffix = atc.superposed_flow(pt, result.awards)          # combined-schedule line flows
    act = atc._monitored_idx(pt, monitored)
    sf_g = np.array([[pt.ptdf[l, pt.bus_idx[b]] for b in busg] for l in act]) \
        if act else np.zeros((0, G))
    sf_c = np.array([[pt.ptdf[l, pt.bus_idx[c]] for c in centers] for l in act]) \
        if act else np.zeros((0, C))                       # shedding adds +SF (less withdrawal)

    c_obj = np.concatenate([cost, price, np.full(C, float(voll))])
    A_eq = np.concatenate([np.ones((1, G)), -np.ones((1, G)), np.ones((1, C))], axis=1)
    b_eq = np.array([0.0])
    A_ub, b_ub = [], []
    for row, l in enumerate(act):
        srow = np.concatenate([sf_g[row], -sf_g[row], sf_c[row]])
        A_ub.append(srow);  b_ub.append(pt.s_nom[l] - Ffix[l])
        A_ub.append(-srow); b_ub.append(pt.s_nom[l] + Ffix[l])
    bounds = ([(0.0, max(0.0, cap[i] - s[i])) for i in range(G)]
              + [(0.0, max(0.0, s[i])) for i in range(G)]
              + [(0.0, max(0.0, float(shed_cap[i]))) for i in range(C)])
    res = linprog(c_obj, A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"backstop_clear infeasible: {res.message}")

    up = res.x[:G]
    down = res.x[G:2 * G]
    shed = res.x[2 * G:]
    g_final = s + up - down
    redispatch_cost = float(cost @ (up - down))            # production-cost uplift (true costs)
    total_down = float(down.sum())
    charge, tsp_charge = {}, 0.0
    if bearer == "tsp":
        # the transmission service provider that rated the ATC bears the whole uplift
        # (it provisioned the capability; the redispatch is its provision cost, not the
        # scheduler's) -- the 313 framing. Default 'causer' keeps the 102 narrative.
        tsp_charge = max(0.0, redispatch_cost)
    elif total_down > 1e-9 and redispatch_cost > 1e-9:
        for i, x in enumerate(sup):
            if down[i] > 1e-9:
                charge[x.gid] = redispatch_cost * float(down[i]) / total_down

    # post-reconciliation loadings: flows of the final injection (gens, withdrawals, shed)
    inj = np.zeros(pt.n_bus)
    for a in result.awards:
        inj[pt.bus_idx[a.sink]] -= a.mw                      # withdrawal at load center
    for i, x in enumerate(sup):
        inj[pt.bus_idx[x.bus]] += g_final[i]
    for j, c in enumerate(centers):
        inj[pt.bus_idx[c]] += shed[j]                        # shed = restored injection
    flow = pt.ptdf @ inj
    rows = []
    for l in atc._monitored_idx(pt, monitored):
        b0, b1 = pt.line_buses[l]
        lim = float(pt.s_nom[l])
        rows.append({"line": pt.lines[l], "from": b0, "to": b1,
                     "flow": round(float(flow[l]), 1), "limit": round(lim, 0),
                     "loading_%": round(100 * abs(flow[l]) / lim, 0) if lim > 0 else np.inf,
                     "overload": abs(flow[l]) > lim + 1e-6})
    loadings = pd.DataFrame(rows).set_index("line")
    feas_after = not bool(loadings["overload"].any())

    return BackstopResult(
        dispatch={x.gid: float(g_final[i]) for i, x in enumerate(sup)},
        up={x.gid: float(up[i]) for i, x in enumerate(sup) if up[i] > 1e-9},
        down={x.gid: float(down[i]) for i, x in enumerate(sup) if down[i] > 1e-9},
        shed={c: float(shed[j]) for j, c in enumerate(centers) if shed[j] > 1e-9},
        redispatch_cost=redispatch_cost,
        charge_by_arb=charge,
        tsp_charge=tsp_charge,
        feasible_before=feas_before,
        feasible_after=feas_after,
        loadings=loadings,
    )


# Backward-compatible aliases (the prior names).
RedispatchResult = BackstopResult
redispatch = backstop_clear


# ──────────────────────────────────────────────────────────────────────────
# Competitive DC-OPF benchmark and the comparison
# ──────────────────────────────────────────────────────────────────────────
def competitive_clear(pt, suppliers: list[Supplier], load_centers: dict[str, LoadCenter],
                      monitored="all"):
    """The perfectly-competitive nodal clearing over the **inelastic forecast**.

    Builds a ``MarketEngine`` from all units (marginal-cost offers) serving the fixed
    forecast ``d_nom`` with line limits enforced -- the efficient, no-premium dispatch.
    Returns ``(EngineResult, {sink: d_nom})``: the competitive LMP is the ``p_comp`` floor
    the strategic premium is measured against, and every center is served in full.
    """
    act = "all" if monitored == "all" else list(monitored)
    eng = MarketEngine(
        name="COMPETITIVE",
        gens={s.gid: {"bus": s.bus, "cost": s.cost, "p_nom": s.p_nom} for s in suppliers},
        loads={lc.bus: lc.d_nom for lc in load_centers.values()},
        activated_lines=act,
    )
    res = solve_engine_dispatch(pt, eng)
    D = {k: load_centers[k].d_nom for k in load_centers}
    return res, D


def compare(result: BilateralResult, backstop: BackstopResult, competitive) -> pd.DataFrame:
    """Per-center strategic-vs-competitive table: load served, the risk-averse premium.

    Both clearings **serve the full forecast**; the difference is the price. Columns:
    forecast, served (strategic), the backstop fringe-fill and any VOLL shed, competitive
    vs strategic price, and the **premium** (strategic - competitive). A TOTAL row carries
    the served totals and the causer-pays reconciliation uplift.
    """
    comp_res, _ = competitive
    rows = []
    for k in result.center_demand:
        rows.append({
            "center": k,
            "forecast": round(result.served[k] + result.shed[k], 1),
            "served": round(result.served[k], 1),
            "fringe": round(result.fringe[k], 1),
            "shed": round(result.shed[k], 1),
            "price_comp": round(float(comp_res.lmp[k]), 1),
            "price_strat": round(result.center_price[k], 1),
            "premium": round(result.premium[k], 1),
        })
    df = pd.DataFrame(rows).set_index("center")
    df.loc["TOTAL"] = {
        "forecast": round(df["forecast"].sum(), 1), "served": round(df["served"].sum(), 1),
        "fringe": round(df["fringe"].sum(), 1), "shed": round(df["shed"].sum(), 1),
        "price_comp": np.nan, "price_strat": np.nan, "premium": np.nan,  # prices don't sum
    }
    return df


# ──────────────────────────────────────────────────────────────────────────
# The decentralized order book (double auction, discriminatory midpoint price)
# ──────────────────────────────────────────────────────────────────────────
# The Cournot machinery above is the *insufficient-competition* model: a handful of
# strategic firms withhold, and the rent vanishes only as the firm count -> infinity
# (Hobbs 2001). The order book below is the *canonical bilateral market*: many sellers
# and buyers post shaded asks and bids, the book is matched and **cleared as submitted**
# (NOT a least-cost optimization), and each matched pair settles at the **midpoint** of
# its bid and ask -- the discriminatory k = 1/2 double auction of Nicolaisen, Petrov &
# Tesfatsion (2001). Three results give it three teaching layers:
#
#   * **Institution (Gode & Sunder 1993).** Even *zero-intelligence* traders bidding at
#     random reach ~99% of the gains from trade, provided a **budget constraint** holds --
#     a seller never asks below cost, a buyer never bids above its value; the institution
#     does the work. (``order_book_clear`` scores allocative efficiency = realized / first-best.)
#   * **Convergence (Rustichini, Satterthwaite & Williams 1994).** With private values the
#     equilibrium shading is ``O(1/m)`` and the welfare loss ``O(1/m^2)``: thicken the book
#     (the ``thickness`` knob ``N``) and the strategic wedge and marginal premium collapse to
#     the competitive nodal (OPF) clearing. Hobbs's thesis made a *rate*, not an asymptote.
#   * **Network microstructure (Borenstein; NPT defer congestion to future work).** A trade
#     ``g -> k`` consumes a transmission right; with ``network="rights"`` each contract respects
#     only its own ATC, so the awarded set can jointly **overload** -- the notebook-111
#     simultaneous-feasibility failure -- which ``order_book_redispatch`` then balances.
#
# THREE premium layers, each with its own lever:
#   (1) **Friction** (``shade``): the strategic bid-ask wedge, ``O(1/m)``, which the thickness
#       sweep collapses to zero -- the market converges to the competitive (OPF) price.
#   (2) **Forward risk** (``Belief``, Bessembinder-Lemmon 2002): the day-ahead/UC context is a
#       two-sided FORWARD market over an uncertain, spike-prone spot price. Generators have a
#       risk APPETITE for the upside (ask ``cost + gen_appetite*sigma*max(0,z)``); loads are
#       risk-AVERSE to it (bid ``min(value, p_comp + load_aversion*sigma)``); they meet at a
#       premium above marginal cost. A preference over real price risk -- it survives full
#       competition (unlike the friction). Each generator keeps a SINGLE fixed marginal cost.
#   (3) **Balancing** (``order_book_redispatch``): the min-cost counter-trade that restores
#       simultaneous feasibility, its production-cost uplift charged to the curtailed schedule
#       (causer-pays) -- the same machinery the Cournot model uses (``backstop_clear``).
# The primitives: the seller's FIXED marginal cost and its risk-appetite ask, and the buyer's
# TRUE value/budget and its risk-averse forward bid (see ``Belief``/``build_orders``).
@dataclass
class Belief:
    """The two-sided forward-market risk attitudes (Bessembinder-Lemmon 2002).

    Each generator's marginal **cost is fixed and known to it**; what is uncertain is the
    **spot price**, a right-skewed, spike-prone distribution with perceived risk ``price_risk``
    (sigma, $/MWh; a generator may carry its own via ``Supplier.price_risk``). The two sides
    hedge that price asymmetrically -- the forward-premium intuition of Bessembinder-Lemmon:

    * **Generators have a risk APPETITE for the upside tail.** Each interval has a spot signal
      ``z`` (standard normal); a generator asks ``cost + gen_appetite * sigma * max(0, z)`` --
      reaching up for margin when the market is tight, undercutting toward its cost otherwise.
      Asking too high risks being **cut out** of the clearing (lost opportunity).
    * **Loads are risk-AVERSE to the upside tail.** A load makes a forward commitment, bidding
      ``E[price] + load_aversion * sigma`` above its expected cost (the competitive ``p_comp``),
      **capped by its value/budget** (Gode-Sunder: it never bids above its true value).

    Deterministically (no ``rng``) ``z = 0``: generators ask at cost and the premium is the
    load's forward hedge. With draws, ``z`` varies and the Monte Carlo traces the resulting
    forward-premium / firmness distribution. Both attitudes ``0`` is the risk-neutral book.
    """

    price_risk: float = 0.0       # sigma of the perceived spot-price distribution ($/MWh)
    gen_appetite: float = 0.0     # generator appetite for the upside tail (ask markup multiplier)
    load_aversion: float = 0.0    # load aversion to the upside tail (forward bid premium, in sigma)
    value_sd: float = 0.0         # SD of loads' true value of energy


@dataclass
class Order:
    """One block in the order book: a seller's ask or a buyer's bid.

    ``limit`` is the TRUE primitive scored for surplus and the Gode-Sunder budget constraint:
    the seller's realized cost floor or the buyer's true value of energy. ``price`` is the
    SUBMITTED ask/bid -- the perceived cost/value, plus a risk hedge, plus the strategic
    ``strat`` shade. The matcher sorts and clears on ``price``; surplus is scored on ``limit``.
    """

    oid: str
    bus: str
    side: str          # "sell" or "buy"
    qty: float         # block size, MW
    limit: float       # TRUE cost floor (sell) or value (buy), $/MWh
    price: float = 0.0  # submitted ask/bid (perceived + risk hedge + strategic shade)
    strat: float = 0.0  # the strategic O(1/m) component of price (signed), for diagnostics

    def __post_init__(self):
        self.oid, self.bus, self.side = str(self.oid), str(self.bus), str(self.side)
        self.qty, self.limit, self.price = float(self.qty), float(self.limit), float(self.price)
        self.strat = float(self.strat)


@dataclass
class OrderBookResult:
    """The cleared decentralized order book."""

    trades: list[dict]                 # each matched fill: seller/buyer, qty, ask/bid, midpoint
    awards: list[atc.Award]            # aggregated (source_bus -> center) deliveries
    cleared: dict[str, float]          # center -> MW matched
    forecast: dict[str, float]         # center -> MW demanded
    price: dict[str, float]            # center -> volume-weighted (discriminatory) trade price
    marg_price: dict[str, float]       # center -> marginal-pair midpoint (the LMP analogue)
    premium: dict[str, float]          # center -> avg trade price - competitive (p_comp)
    marg_premium: dict[str, float]     # center -> marginal price - competitive (-> 0 thick book)
    seller_profit: dict[str, float]    # source bus -> rent captured
    buyer_surplus: dict[str, float]    # center -> consumer surplus
    realized_surplus: float            # gains from trade actually captured (true limits)
    best_surplus: float                # gains from trade of the truthful (first-best) book
    efficiency: float                  # realized / best (allocative efficiency)
    shade_wedge: float                 # mean |strategic shade| over orders (-> 0 thick book)
    n_sell: int
    n_buy: int
    loadings: pd.DataFrame


def build_orders(suppliers: list[Supplier], load_centers: dict[str, LoadCenter], *,
                 thickness: int = 1, shade: float = 0.0, belief: Belief | None = None,
                 rng=None) -> tuple[list[Order], list[Order]]:
    """Turn the fleet and the load into a thick book of belief-driven sell and buy orders.

    Each supplier becomes ``thickness`` sell blocks, each load center ``thickness`` buy blocks.
    Faithful to the NPT primitives (a fixed marginal cost / value behind every offer) and the
    Bessembinder-Lemmon two-sided forward premium:

    * **Seller.** Fixed marginal ``cost`` (the ``limit`` -- never drawn; the generator knows it).
      A spot signal ``z`` (standard normal, drawn per interval) drives a risk-appetite **ask**
      ``cost + gen_appetite * sigma_g * max(0, z)`` -- reaching for the upside tail in a tight
      interval, undercutting toward cost otherwise -- plus an ``O(1/m)`` strategic markup.
    * **Buyer.** TRUE value = the affine demand value ``a_k - b_k q`` (high inframarginal,
      ``p_comp`` at the forecast margin) + ``value_sd`` noise -- the surplus basis and the hard
      budget cap. The **bid** depends on the mode:

      - ``belief=None`` (value double auction, RSW): the buyer bids its true value -- the
        convergence configuration; as the book thickens the price -> the competitive ``p_comp``.
      - ``Belief`` supplied (**forward procurement**): the load makes a risk-averse forward
        commitment, ``min(value, p_comp + load_aversion * sigma)`` -- bidding above expected
        cost to be served firmly, but never above its value/budget (Gode-Sunder). ``z`` does
        NOT move the load (it is hedged); it moves the generators' asks.

    ``shade`` is the friction wedge (``O(1/m)``, vanishes with thickness). Deterministically
    (``rng=None``) ``z = 0`` so generators ask at cost and the premium is the load's hedge;
    the Monte Carlo over ``z`` traces the two-sided premium/firmness distribution.
    """
    bel = belief if belief is not None else Belief()
    forward = belief is not None
    thickness = max(1, int(thickness))
    draw = rng is not None
    z = float(rng.normal(0.0, 1.0)) if draw else 0.0             # interval spot signal (tightness)
    cap = max((lc.voll for lc in load_centers.values()), default=1.0)

    sells: list[Order] = []
    for s in suppliers:
        q = s.p_nom / thickness
        sigma_g = s.price_risk if s.price_risk > 0 else bel.price_risk
        # risk-appetite ASK above the FIXED cost: a forward reach (baseline 1) the interval's
        # spot signal z tightens (z>0, reach further) or loosens (z<-1, undercut toward cost).
        ask = s.cost + (bel.gen_appetite * sigma_g * max(0.0, 1.0 + z) if forward else 0.0)
        for i in range(thickness):
            sells.append(Order(f"{s.gid}.{i}", s.bus, "sell", q, s.cost, ask))   # limit = fixed cost
    buys: list[Order] = []
    hedge = bel.load_aversion * bel.price_risk if forward else 0.0   # forward bid premium ($/MWh)
    for kbus, lc in load_centers.items():
        q_block = lc.d_nom / thickness
        for i in range(thickness):
            q_mid = (i + 0.5) * q_block                              # cumulative quantity
            base_val = min(lc.voll, max(lc.p_comp, lc.a - lc.b * q_mid))   # affine true WTP
            vnoise = float(rng.normal(0.0, bel.value_sd)) if (draw and bel.value_sd > 0) else 0.0
            value_true = min(lc.voll, max(lc.p_comp, base_val + vnoise))
            bid = min(value_true, lc.p_comp + hedge) if forward else value_true   # budget cap stands
            buys.append(Order(f"L{kbus}.{i}", kbus, "buy", q_block, value_true, bid))

    m_sell, m_buy = max(1, len(sells)), max(1, len(buys))
    for o in sells:
        o.strat = shade * max(0.0, cap - o.limit) / m_sell          # strategic markup, O(1/m)
        o.price += o.strat
    for o in buys:
        o.strat = -shade * o.price / m_buy                          # strategic shave, O(1/m)
        o.price = max(0.0, o.price + o.strat)
    return sells, buys


def _path_cap(pt, source, sink, monitored, cache) -> float:
    """ATC (standalone TTC) of the ``source->sink`` contract path, cached; inf within a bus."""
    if source == sink:
        return float("inf")
    key = (source, sink)
    if key not in cache:
        cache[key] = float(atc.ttc(pt, source, sink, monitored=monitored)[0])
    return cache[key]


def _match(pt, sells: list[Order], buys: list[Order], k: float, monitored,
           network: str = "rights") -> list[dict]:
    """Match the book: highest-value buyer walks down the ask-ascending sellers.

    Buyers are served in bid-descending order (path-dependent, as a real book is); each takes
    the cheapest **individually rational** (``bid >= ask``) seller it can still reach, and the
    fill settles at its ``k``-midpoint ``k*bid + (1-k)*ask`` -- discriminatory pricing,
    dispersed across pairs (NPT). Reachability depends on ``network``:

    * ``"rights"`` (default): each contract path ``g -> k`` is capped only by its OWN ATC
      (a point-to-point transmission right). Like notebook 111, the individually feasible
      contracts can jointly **overload** the grid -- the simultaneous-feasibility failure that
      ``order_book_redispatch`` then balances.
    * ``"simultaneous"``: each fill is checked against ``atc.available_atc`` of the running
      awards, so the cleared book is network-feasible by construction (no redispatch needed).
    """
    S = sorted(sells, key=lambda o: o.price)
    B = sorted(buys, key=lambda o: -o.price)
    rs = {id(o): o.qty for o in S}
    rb = {id(o): o.qty for o in B}
    booked: list[atc.Award] = []
    used: dict[tuple[str, str], float] = {}
    ttc_cache: dict[tuple[str, str], float] = {}
    trades: list[dict] = []
    for b in B:
        for s in S:
            if rb[id(b)] <= 1e-9:
                break
            if rs[id(s)] <= 1e-9:
                continue
            if b.price + 1e-9 < s.price:      # asks ascending: no IR seller remains for b
                break
            q = min(rb[id(b)], rs[id(s)])
            if network == "simultaneous":
                q = min(q, atc.available_atc(pt, s.bus, b.bus, booked, monitored=monitored))
            else:                             # per-right path cap (can overload jointly)
                cap = _path_cap(pt, s.bus, b.bus, monitored, ttc_cache)
                q = min(q, cap - used.get((s.bus, b.bus), 0.0))
            if q <= 1e-6:                     # path full: walk to the next seller
                continue
            trades.append(dict(seller=s.oid, sbus=s.bus, buyer=b.oid, center=b.bus,
                               qty=float(q), ask=s.price, bid=b.price,
                               price=k * b.price + (1 - k) * s.price,
                               s_cost=s.limit, b_val=b.limit))
            if network == "simultaneous":
                booked.append(atc.Award(s.bus, b.bus, q))
            else:
                used[(s.bus, b.bus)] = used.get((s.bus, b.bus), 0.0) + q
            rs[id(s)] -= q
            rb[id(b)] -= q
    return trades


def order_book_clear(pt, sells: list[Order], buys: list[Order],
                     load_centers: dict[str, LoadCenter] | None = None, *,
                     k: float = 0.5, monitored="all", network: str = "rights") -> OrderBookResult:
    """Clear the decentralized order book and score it against the first-best.

    Matches the submitted book (``_match``), then re-matches a **truthful** copy (every order
    priced at its ``limit``) to get the first-best gains from trade; allocative ``efficiency``
    is the ratio (Gode-Sunder). Per center it reports both the volume-weighted **discriminatory
    price** the load actually pays and the **marginal-pair midpoint** -- the price of the last
    accepted trade, the LMP analogue that converges to the competitive ``p_comp`` as the book
    thickens. With ``network="rights"`` (default) the awards respect each contract's ATC but can
    jointly overload -- pass the result to ``order_book_redispatch`` to balance the schedule.
    """
    trades = _match(pt, sells, buys, k, monitored, network=network)
    truthful = _match(pt,
                      [Order(o.oid, o.bus, o.side, o.qty, o.limit, o.limit) for o in sells],
                      [Order(o.oid, o.bus, o.side, o.qty, o.limit, o.limit) for o in buys],
                      k, monitored, network=network)
    realized = sum((t["b_val"] - t["s_cost"]) * t["qty"] for t in trades)
    best = sum((t["b_val"] - t["s_cost"]) * t["qty"] for t in truthful)
    efficiency = min(1.0, realized / best) if best > 1e-9 else 1.0   # clip greedy-benchmark noise

    centers = sorted({o.bus for o in buys})
    forecast = {c: sum(o.qty for o in buys if o.bus == c) for c in centers}
    cleared = {c: 0.0 for c in centers}
    pxvol = {c: 0.0 for c in centers}
    lo_bid = {c: float("inf") for c in centers}     # marginal accepted bid at the center
    hi_ask = {c: float("-inf") for c in centers}    # marginal accepted ask delivered there
    buyer_surplus = {c: 0.0 for c in centers}
    seller_profit: dict[str, float] = {}
    agg: dict[tuple[str, str], float] = {}
    for t in trades:
        c, q = t["center"], t["qty"]
        cleared[c] += q
        pxvol[c] += t["price"] * q
        lo_bid[c] = min(lo_bid[c], t["bid"])
        hi_ask[c] = max(hi_ask[c], t["ask"])
        buyer_surplus[c] += (t["b_val"] - t["price"]) * q
        seller_profit[t["sbus"]] = seller_profit.get(t["sbus"], 0.0) + (t["price"] - t["s_cost"]) * q
        agg[(t["sbus"], c)] = agg.get((t["sbus"], c), 0.0) + q

    p_comp = {c: (load_centers[c].p_comp if load_centers and c in load_centers else 0.0)
              for c in centers}
    price = {c: (pxvol[c] / cleared[c] if cleared[c] > 1e-9 else float("nan")) for c in centers}
    marg_price = {c: (k * lo_bid[c] + (1 - k) * hi_ask[c] if cleared[c] > 1e-9 else float("nan"))
                  for c in centers}
    premium = {c: (price[c] - p_comp[c] if cleared[c] > 1e-9 else float("nan")) for c in centers}
    marg_premium = {c: (marg_price[c] - p_comp[c] if cleared[c] > 1e-9 else float("nan"))
                    for c in centers}
    wedge = [abs(o.strat) for o in (list(sells) + list(buys))]
    awards = [atc.Award(src, c, mw) for (src, c), mw in agg.items() if mw > 1e-9]
    loadings = atc.line_loadings(pt, awards, monitored=monitored)

    return OrderBookResult(
        trades=trades, awards=awards, cleared=cleared, forecast=forecast, price=price,
        marg_price=marg_price, premium=premium, marg_premium=marg_premium,
        seller_profit=seller_profit, buyer_surplus=buyer_surplus,
        realized_surplus=float(realized), best_surplus=float(best), efficiency=float(efficiency),
        shade_wedge=float(np.mean(wedge)) if wedge else 0.0,
        n_sell=len(sells), n_buy=len(buys), loadings=loadings,
    )


def _assemble(pt, trades, forecast, load_centers, k, monitored, best_surplus, n_sell, n_buy):
    """Aggregate a list of cleared ``trades`` into an ``OrderBookResult``."""
    centers = sorted(forecast)
    cleared = {c: 0.0 for c in centers}
    pxvol = {c: 0.0 for c in centers}
    lo_bid = {c: float("inf") for c in centers}
    hi_ask = {c: float("-inf") for c in centers}
    buyer_surplus = {c: 0.0 for c in centers}
    seller_profit: dict[str, float] = {}
    agg: dict[tuple[str, str], float] = {}
    spread_vol = 0.0
    for t in trades:
        c, q = t["center"], t["qty"]
        cleared[c] += q
        pxvol[c] += t["price"] * q
        lo_bid[c] = min(lo_bid[c], t["bid"])
        hi_ask[c] = max(hi_ask[c], t["ask"])
        buyer_surplus[c] += (t["b_val"] - t["price"]) * q
        seller_profit[t["sbus"]] = seller_profit.get(t["sbus"], 0.0) + (t["price"] - t["s_cost"]) * q
        agg[(t["sbus"], c)] = agg.get((t["sbus"], c), 0.0) + q
        spread_vol += (t["bid"] - t["ask"]) * q
    p_comp = {c: (load_centers[c].p_comp if c in load_centers else 0.0) for c in centers}
    price = {c: (pxvol[c] / cleared[c] if cleared[c] > 1e-9 else float("nan")) for c in centers}
    marg_price = {c: (k * lo_bid[c] + (1 - k) * hi_ask[c] if cleared[c] > 1e-9 else float("nan"))
                  for c in centers}
    premium = {c: (price[c] - p_comp[c] if cleared[c] > 1e-9 else float("nan")) for c in centers}
    marg_premium = {c: (marg_price[c] - p_comp[c] if cleared[c] > 1e-9 else float("nan"))
                    for c in centers}
    realized = sum((t["b_val"] - t["s_cost"]) * t["qty"] for t in trades)
    vol = sum(t["qty"] for t in trades)
    awards = [atc.Award(src, c, mw) for (src, c), mw in agg.items() if mw > 1e-9]
    loadings = atc.line_loadings(pt, awards, monitored=monitored)
    return OrderBookResult(
        trades=trades, awards=awards, cleared=cleared, forecast=forecast, price=price,
        marg_price=marg_price, premium=premium, marg_premium=marg_premium,
        seller_profit=seller_profit, buyer_surplus=buyer_surplus,
        realized_surplus=float(realized),
        best_surplus=float(best_surplus) if best_surplus else float(realized),
        efficiency=min(1.0, realized / best_surplus) if best_surplus and best_surplus > 1e-9 else 1.0,
        shade_wedge=float(spread_vol / vol) if vol > 1e-9 else 0.0,
        n_sell=n_sell, n_buy=n_buy, loadings=loadings,
    )


def double_auction_clear(pt, suppliers: list[Supplier], load_centers: dict[str, LoadCenter], *,
                         belief: Belief | None = None, rng=None, thickness: int = 4,
                         rounds: int = 12, k: float = 0.5, network: str = "rights", monitored="all",
                         reach0: float = 1.5, concede: float = 0.3, climb: float = 0.35
                         ) -> OrderBookResult:
    """The **repeated discriminatory double auction** that approximates bilateral trading.

    There is no central clearing: over ``rounds`` of bargaining, every still-unfilled seller
    and buyer posts a fresh offer DRAWN from its perceived spot-price distribution, and the
    book is matched at the midpoint where a bid meets an ask. Because offers are stochastic and
    the market is not incentive-compatible, a buyer's draw often lands below a seller's -- no
    deal that round -- so the two sides **concede** over rounds (sellers lower their reach
    toward cost, buyers raise their bid toward budget) until the must-serve load procures its
    **full** volume. Each block:

    * **Seller** (fixed cost ``c_i``, price-risk ``sigma_i``) asks, in round ``r``,
      ``c_i + gen_appetite*sigma_i*max(0, reach0 - concede*r) + sigma_i*eps`` (floored at cost):
      it reaches for the upside early and concedes toward cost as it stays unsold.
    * **Buyer** (budget ``v_j`` = value ceiling, expected ``p_comp``, price-risk ``sigma``) bids
      ``p_comp + load_aversion*sigma*(climb*r) + sigma*eps``, capped at ``v_j``: it raises its
      offer as it stays unserved, never above budget.

    ``rng=None`` drops the ``eps`` dispersion (a clean deterministic concession path). Returns
    an ``OrderBookResult``; ``trades`` carry a ``round`` field, and ``shade_wedge`` is the
    realized volume-weighted bid-ask spread. With ``network="rights"`` the awards respect each
    contract's ATC but can jointly overload -- pass to ``order_book_redispatch`` to balance.
    """
    bel = belief if belief is not None else Belief()
    thickness = max(1, int(thickness))
    draw = rng is not None

    sellers, buyers = [], []
    for s in suppliers:
        sg = s.price_risk if s.price_risk > 0 else bel.price_risk
        q = s.p_nom / thickness
        for i in range(thickness):
            sellers.append({"id": f"{s.gid}.{i}", "bus": s.bus, "cost": s.cost, "sig": sg, "qty": q})
    for kbus, lc in load_centers.items():
        q = lc.d_nom / thickness
        for i in range(thickness):
            buyers.append({"id": f"L{kbus}.{i}", "center": kbus, "budget": lc.voll,
                           "pcomp": lc.p_comp, "sig": bel.price_risk, "qty": q})
    rem_s = {a["id"]: a["qty"] for a in sellers}
    rem_b = {a["id"]: a["qty"] for a in buyers}
    used: dict[tuple[str, str], float] = {}
    ttc_cache: dict[tuple[str, str], float] = {}
    trades: list[dict] = []
    disp = 1.0 / np.sqrt(thickness)            # liquidity: more parties -> tighter offer dispersion

    for r in range(int(rounds)):
        act_b = [a for a in buyers if rem_b[a["id"]] > 1e-9]
        if not act_b:
            break
        act_s = [a for a in sellers if rem_s[a["id"]] > 1e-9]
        for a in act_s:
            reach = bel.gen_appetite * a["sig"] * max(0.0, reach0 - concede * r)
            eps = float(rng.normal(0.0, 1.0)) if draw else 0.0
            a["ask"] = max(a["cost"], a["cost"] + reach + a["sig"] * disp * eps)
        for a in act_b:
            up = bel.load_aversion * a["sig"] * (climb * r)
            eps = float(rng.normal(0.0, 1.0)) if draw else 0.0
            a["bid"] = min(a["budget"], a["pcomp"] + up + a["sig"] * disp * eps)
        S = sorted(act_s, key=lambda a: a["ask"])
        B = sorted(act_b, key=lambda a: -a["bid"])
        for b in B:
            for s in S:
                if rem_b[b["id"]] <= 1e-9:
                    break
                if rem_s[s["id"]] <= 1e-9:
                    continue
                if b["bid"] + 1e-9 < s["ask"]:        # asks ascending: no IR seller remains
                    break
                q = min(rem_b[b["id"]], rem_s[s["id"]])
                if network == "rights":
                    cap = _path_cap(pt, s["bus"], b["center"], monitored, ttc_cache)
                    q = min(q, cap - used.get((s["bus"], b["center"]), 0.0))
                else:
                    bk = [atc.Award(src, snk, mw) for (src, snk), mw in used.items()]
                    q = min(q, atc.available_atc(pt, s["bus"], b["center"], bk, monitored=monitored))
                if q <= 1e-6:
                    continue
                trades.append(dict(seller=s["id"], sbus=s["bus"], buyer=b["id"], center=b["center"],
                                   qty=float(q), ask=s["ask"], bid=b["bid"],
                                   price=k * b["bid"] + (1 - k) * s["ask"],
                                   s_cost=s["cost"], b_val=b["budget"], round=r))
                used[(s["bus"], b["center"])] = used.get((s["bus"], b["center"]), 0.0) + q
                rem_s[s["id"]] -= q
                rem_b[b["id"]] -= q

    forecast = {kbus: lc.d_nom for kbus, lc in load_centers.items()}
    # first-best: every MW served by the cheapest feasible seller (truthful single shot)
    tru = _match(pt,
                 [Order(f"{s.gid}.{i}", s.bus, "sell", s.p_nom / thickness, s.cost, s.cost)
                  for s in suppliers for i in range(thickness)],
                 [Order(f"L{kbus}.{i}", kbus, "buy", lc.d_nom / thickness, lc.voll, lc.voll)
                  for kbus, lc in load_centers.items() for i in range(thickness)],
                 k, monitored, network=network)
    best = sum((t["b_val"] - t["s_cost"]) * t["qty"] for t in tru)
    return _assemble(pt, trades, forecast, load_centers, k, monitored, best,
                     len(sellers), len(buyers))


def convergence_sweep(pt, suppliers: list[Supplier], load_centers: dict[str, LoadCenter],
                      thicknesses, *, shade: float = 0.4, belief: Belief | None = None,
                      k: float = 0.5, monitored="all", network: str = "simultaneous") -> pd.DataFrame:
    """Re-clear the (deterministic) order book at each book thickness ``N``.

    The convergence demonstration: as ``N`` grows the strategic shade is ``O(1/N)`` and the
    welfare loss ``O(1/N^2)`` (Rustichini-Satterthwaite-Williams), so ``shade_wedge`` and the
    ``marg_premium`` (the marginal price's gap to competitive) collapse toward zero -- the
    book converges to the OPF clearing -- while ``avg_premium`` (the discriminatory price the
    load pays, lifted by the persistent risk hedge in ``belief``) does **not** vanish.
    """
    rows = []
    for N in thicknesses:
        sells, buys = build_orders(suppliers, load_centers, thickness=int(N),
                                   shade=shade, belief=belief)
        r = order_book_clear(pt, sells, buys, load_centers, k=k, monitored=monitored, network=network)
        prem = [v for v in r.premium.values() if v == v]
        mprem = [v for v in r.marg_premium.values() if v == v]
        rows.append({"N": int(N), "blocks": r.n_sell + r.n_buy,
                     "shade_wedge": round(r.shade_wedge, 2),
                     "marg_premium": round(float(np.mean(mprem)) if mprem else float("nan"), 2),
                     "avg_premium": round(float(np.mean(prem)) if prem else float("nan"), 2),
                     "efficiency": round(r.efficiency, 4),
                     "cleared_MW": round(sum(r.cleared.values()), 1)})
    return pd.DataFrame(rows).set_index("N")


def monte_carlo(pt, suppliers: list[Supplier], load_centers: dict[str, LoadCenter], *,
                belief: Belief, draws: int = 200, thickness: int = 4, shade: float = 0.0,
                seed: int = 0, k: float = 0.5, monitored="all", network: str = "rights") -> pd.DataFrame:
    """Repeat the order book over ``draws`` intervals -- the forward-premium experiment.

    Each draw is one interval with a fresh spot signal ``z`` (and ``value_sd`` noise) from
    ``belief`` (seeded ``rng``): in a tight draw (``z>0``) generators reach up for the upside
    tail. Returns one row per draw with the served fraction of forecast, the volume-weighted
    price premium over competitive, and efficiency. Comparing a risk-neutral (``load_aversion=0``)
    against a risk-averse load shows the trade it makes -- a higher average premium bought in
    exchange for firmer service when the market tightens (the Bessembinder-Lemmon forward premium).
    """
    rng = np.random.default_rng(seed)
    fc = sum(lc.d_nom for lc in load_centers.values())
    rows = []
    for d in range(int(draws)):
        sells, buys = build_orders(suppliers, load_centers, thickness=thickness,
                                   shade=shade, belief=belief, rng=rng)
        r = order_book_clear(pt, sells, buys, load_centers, k=k, monitored=monitored, network=network)
        served = sum(r.cleared.values())
        prem = [v for v in r.premium.values() if v == v]
        rows.append({"draw": d, "served_frac": served / fc if fc > 0 else 1.0,
                     "premium": float(np.mean(prem)) if prem else float("nan"),
                     "efficiency": r.efficiency})
    return pd.DataFrame(rows)


def order_book_redispatch(pt, suppliers: list[Supplier], result: OrderBookResult, *,
                          voll: float = 1000.0, monitored="all",
                          bearer: str = "causer") -> BackstopResult:
    """Balance the bilateral schedule and charge the uplift to the curtailed schedule.

    Each order-book contract respects its own transmission right, but (like notebook 111's
    simultaneous-feasibility failure) the awarded set can overload the grid. This runs the same
    minimum-cost **counter-trading redispatch** as the Cournot model (``backstop_clear``): curtail
    the schedule loading the constrained line, ramp a feasible substitute, shed at VOLL only as a
    last resort. The production-cost uplift is the **balancing-cost premium**, charged to the
    curtailed (causing) schedule -- causer pays. Returns a ``BackstopResult`` with the
    redispatch, the ``redispatch_cost`` uplift, the causer-pays ``charge_by_arb``, and
    before/after feasibility.
    """
    from types import SimpleNamespace
    bus_to_gid = {s.bus: s.gid for s in suppliers}
    self_sched = {s.gid: 0.0 for s in suppliers}
    for a in result.awards:
        if a.source in bus_to_gid:
            self_sched[bus_to_gid[a.source]] += a.mw
    shim = SimpleNamespace(awards=result.awards, self_sched=self_sched, served=result.cleared)
    return backstop_clear(pt, suppliers, shim, voll=voll, monitored=monitored, bearer=bearer)


# ──────────────────────────────────────────────────────────────────────────
# Smoke test — the spot iteration, its reconciliation, and the benchmark
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import wscc9_model as wm

    suppliers = [Supplier(g, s["bus"], s["cost"], s["p_nom"])
                 for g, s in wm.DEFAULT_GEN_FLEET.items()]

    # ── Scenario A: the 111 planning use (default, uncongested network) ──────
    # The bus-3 supplier's planned book must reduce to 111's naive requests, so the
    # ATC desk in 111 rates exactly the same set.
    ptA = wm.shift_factors()
    lcsA = calibrate_load_centers(ptA)
    booksA = plan_all(ptA, suppliers, lcsA)
    bus3 = next(s for s in suppliers if s.bus == "3")
    reqs = sorted(booksA[bus3.gid].requests())
    print("bus-3 planned book (111 scenario):", reqs)
    for r in booksA[bus3.gid].rights:
        assert abs(r.cap_mw - atc.ttc(ptA, r.source, r.sink)[0]) < 1e-6, "cap != TTC"
    assert [(s, k, round(m)) for s, k, m in reqs] == \
        [("3", "5", 90), ("3", "7", 100), ("3", "9", 125)], reqs

    # ── Scenario B: the 102 spot iteration (must-serve load + fringe + VOLL) ──
    # Default ratings are uncongested (flat $35); tighten line_4 / line_2 so the
    # competitive prices separate and the rights are worth something.
    RATINGS = {"line_4": 60.0, "line_2": 90.0}
    VOLL, FIRMNESS = 200.0, 0.7
    pt = wm.shift_factors(wm.build_network(line_ratings=RATINGS))
    strategic = [s for s in suppliers if s.bus in ("2", "3")]   # cheap arbitragers (Cournot)
    fringe = next(s for s in suppliers if s.bus == "1")         # price-taking backstop
    lcs = calibrate_load_centers(pt, voll=VOLL, firmness=FIRMNESS)
    print("\ncompetitive price p_comp @ centers:", {k: round(lcs[k].p_comp, 1) for k in lcs})
    books = plan_all(pt, strategic, lcs)

    res = clear(pt, strategic, books, lcs, fringe)
    print(f"converged={res.converged} in {res.sweeps} sweeps")
    assert res.converged and res.sweeps < 50

    # best-response fixed point
    for g in books:
        rival = {k: sum(res.deliveries[(h, k)] for h in books if h != g and (h, k) in res.deliveries)
                 for k in books[g].sinks()}
        br = best_response(next(s for s in strategic if s.gid == g), books[g], lcs, rival)
        for k, qk in br.items():
            assert abs(qk - res.deliveries[(g, k)]) < 1e-2, (g, k, qk, res.deliveries[(g, k)])
    print("best-response fixed point: OK")

    # load served in full; the rent is a price premium bounded by VOLL
    print("served:", {k: round(res.served[k], 1) for k in res.served},
          "| fringe:", {k: round(res.fringe[k], 1) for k in res.fringe},
          "| shed:", {k: round(v, 1) for k, v in res.shed.items() if v > 1e-6})
    assert sum(res.shed.values()) < 1.0, "base case should serve the full forecast (no shed)"
    for k in res.premium:
        assert -1e-6 <= res.premium[k] <= VOLL + 1e-6
    assert sum(res.premium.values()) > 0, "the equilibrium should carry a premium"
    print("premium:", {k: round(res.premium[k], 1) for k in res.premium})
    print("equilibrium strategic deliveries:",
          {f"{g}->{k}": round(v, 1) for (g, k), v in res.deliveries.items() if v > 1e-6})

    # benchmark (full forecast served at competitive cost) + backstop reconciliation
    comp = competitive_clear(pt, suppliers, lcs)
    bk = backstop_clear(pt, strategic + [fringe], res, voll=VOLL)
    print("\ncompare:\n", compare(res, bk, comp).to_string())
    print(f"\nfeasible before backstop_clear: {bk.feasible_before} | after: {bk.feasible_after}")
    print("curtail:", {g: round(v, 1) for g, v in bk.down.items()},
          "| ramp up:", {g: round(v, 1) for g, v in bk.up.items()},
          f"| uplift ${bk.redispatch_cost:.0f}")
    assert bk.feasible_after, "backstop_clear must restore feasibility"

    # ── Scenario C: the repeated bilateral double auction ─────────────────────
    # The numerical model of bilateral trading (no central clearing): over rounds of
    # back-and-forth, sellers and buyers draw offers from their perceived spot-price
    # distributions and concede until the must-serve load procures its FULL volume.
    #   * number of parties -> liquidity (the bid-ask spread tightens);
    #   * risk aversion / price-risk sigma -> the forward premium (Bessembinder-Lemmon);
    #   * the rights-feasible schedule can overload -> causer-pays balancing redispatch.
    print("\n--- Scenario C: repeated bilateral double auction ---")
    bel = Belief(price_risk=20.0, gen_appetite=1.0, load_aversion=1.5)
    da = double_auction_clear(pt, suppliers, lcs, belief=bel, rng=np.random.default_rng(3),
                              thickness=4, network="rights")
    served, fc = sum(da.cleared.values()), sum(da.forecast.values())
    nrounds = max(t["round"] for t in da.trades) + 1
    print(f"served {served:.0f}/{fc:.0f} MW in {nrounds} rounds | premium "
          f"{{ {', '.join(f'{c}:{v:+.0f}' for c, v in da.premium.items())} }} | spread ${da.shade_wedge:.0f}")
    # generators keep their FIXED cost; the area-track cost never moves with the draws.
    assert all(abs(t["s_cost"] - next(s.cost for s in suppliers if s.bus == t["sbus"])) < 1e-9
               for t in da.trades), "seller cost is the fixed marginal cost"
    assert served > 0.999 * fc, "the must-serve load procures its full volume"

    def _avg(N=4, av=1.5, sg=20.0, seeds=6):
        rs = [double_auction_clear(pt, suppliers, lcs, thickness=N, rng=np.random.default_rng(s),
              belief=Belief(price_risk=sg, gen_appetite=1.0, load_aversion=av)) for s in range(seeds)]
        prem = float(np.mean([np.nanmean([v for v in r.premium.values() if v == v]) for r in rs]))
        spread = float(np.mean([r.shade_wedge for r in rs]))
        return prem, spread

    _, s1 = _avg(N=1); _, s16 = _avg(N=16)
    print(f"liquidity: N=1 spread ${s1:.0f} -> N=16 spread ${s16:.0f}")
    assert s16 < s1, "more parties tighten the bid-ask spread (liquidity)"
    plo, _ = _avg(av=0.5); phi, _ = _avg(av=3.0)
    print(f"risk: aversion 0.5 premium ${plo:+.0f} -> 3.0 premium ${phi:+.0f}")
    assert phi > plo + 1e-6, "more load aversion raises the forward premium"

    rd = order_book_redispatch(pt, suppliers, da, voll=VOLL)
    print(f"redispatch: feasible {rd.feasible_before}->{rd.feasible_after} | "
          f"curtail {{ {', '.join(f'{g}:{v:.0f}' for g, v in rd.down.items())} }} | uplift ${rd.redispatch_cost:.0f}")
    assert rd.feasible_after, "redispatch restores network feasibility"

    print("\nall smoke assertions passed.")
