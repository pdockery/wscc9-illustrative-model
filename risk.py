# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical per-node distributions, OPF propagation, and risk preferences.

One shared source of truth for the uncertainty and risk-preference parameters that were, until
now, redefined (and quietly *disagreed*) in every seams / intertie notebook. The architecture is
**primitives + OPF propagation**: we do NOT store per-node price distributions directly (a set of
nodal prices that no dispatch produces). Instead we store the *drivers* — a per-bus **load**
distribution and a per-generator **cost** distribution — draw scenarios over them (Monte-Carlo, or
Gauss-Hermite quadrature for a low-dimensional slice), and run the DC-OPF for each draw. The nodal
prices, congestion, and any payoff are then *induced* by the clearing, so they stay physically
consistent with the network and revenue-adequate by construction.

    primitives (load, cost dists)  ->  scenarios (MC / GH)  ->  clear DC-OPF each  ->  induced
    nodal prices / congestion / payoff  ->  E, Var, quantiles, certainty-equivalent, risk-loaded premium

The risk-preference vocabulary matches the repo canon: risk aversion is ``gamma``
everywhere, with the mean-variance certainty equivalent ``CE = E[Pi] - (gamma/2) Var[Pi]``. The
Bessembinder-Lemmon forward-premium primitives (``price_risk``/``gen_appetite``/``load_aversion``)
used by the bilateral double auction are re-exported here as the single canonical ``Belief`` so that
notebook draws from the same place.

**Canonical parameters** (edit here, once): ``DEFAULT_LOAD_SD``, ``DEFAULT_COST_SD``,
``GAMMA_LADDER``, ``DEFAULT_BELIEF``. These reconcile the six previously-uncoordinated setups
(202/212/222/102/`revenue_allocation`); the values are illustrative and meant to be tuned to the
payoff scale — they are documented, not sacred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import wscc9_model as wm
from seams_engine import solve_engine_dispatch


# ──────────────────────────────────────────────────────────────────────────
# Canonical primitives — the one source of truth
# ──────────────────────────────────────────────────────────────────────────
# Per-bus LOAD standard deviation (MW); the mean is ``wscc9_model.DEFAULT_LOADS``.
# ~15% coefficient of variation; bus 7's N(100, 15) matches the seams notebook's load risk.
DEFAULT_LOAD_SD: dict[str, float] = {"5": 13.0, "7": 15.0, "9": 19.0}

# Per-generator COST standard deviation ($/MWh); the mean is the ``DEFAULT_GEN_FLEET`` cost.
# The dear bus-1 unit ($50) is the most volatile; the cheap bus-3 unit ($20) the most stable.
DEFAULT_COST_SD: dict[str, float] = {"gen_slack_0": 12.0, "gen_0": 7.0, "gen_1": 3.0}

# Canonical risk-aversion ladder — the mean-variance (CARA-style) coefficient ``gamma``
# (risk-neutral / moderate / high). Reconciles the old [0,.001,.005] and [0,.006,.018] ladders;
# tune to the payoff variance scale of the notebook using it.
GAMMA_LADDER: tuple[float, ...] = (0.0, 0.005, 0.02)


@dataclass
class Belief:
    """Bessembinder-Lemmon forward-premium primitives (the bilateral double-auction risk model),
    kept here as the single canonical instance. ``price_risk`` = sigma of the perceived spot price
    ($/MWh); ``gen_appetite`` = generator upside-tail ask markup; ``load_aversion`` = load
    forward-bid premium (in sigma); ``value_sd`` = spread of loads' true value of energy."""

    price_risk: float = 20.0
    gen_appetite: float = 1.0
    load_aversion: float = 1.5
    value_sd: float = 0.0


DEFAULT_BELIEF = Belief()


# ──────────────────────────────────────────────────────────────────────────
# Distributions and the canonical dist dictionaries
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Normal:
    """A one-dimensional normal driver: ``mean`` and ``sd`` (a degenerate ``sd=0`` is a constant)."""

    mean: float
    sd: float


def load_dists(loads: dict | None = None, sd: dict | None = None) -> dict[tuple[str, str], Normal]:
    """Per-bus load drivers ``{('load', bus): Normal(mean, sd)}``. Mean = ``DEFAULT_LOADS`` **updated by**
    ``loads``; sd = ``DEFAULT_LOAD_SD`` **updated by** ``sd`` — both keyed by bus, so a *partial* override
    tweaks only the named buses and leaves the rest at their defaults."""
    means = {**wm.DEFAULT_LOADS, **(loads or {})}
    spreads = {**DEFAULT_LOAD_SD, **(sd or {})}
    return {("load", b): Normal(float(mw), float(spreads.get(b, 0.0))) for b, mw in means.items()}


def cost_dists(fleet: dict | None = None, cost: dict | None = None,
               sd: dict | None = None) -> dict[tuple[str, str], Normal]:
    """Per-gen cost drivers ``{('cost', gen): Normal(mean, sd)}``. Mean = ``cost[gen]`` if supplied
    else the fleet cost; sd = ``sd[gen]`` if supplied else ``DEFAULT_COST_SD``. ``cost`` and ``sd`` are
    keyed by **generator id** — ``gen_slack_0`` = bus 1, ``gen_0`` = bus 2, ``gen_1`` = bus 3."""
    fleet = wm.DEFAULT_GEN_FLEET if fleet is None else fleet
    spreads = {**DEFAULT_COST_SD, **(sd or {})}
    cost = cost or {}
    return {("cost", g): Normal(float(cost.get(g, spec["cost"])), float(spreads.get(g, 0.0)))
            for g, spec in fleet.items()}


# ──────────────────────────────────────────────────────────────────────────
# Scenarios — Monte-Carlo and Gauss-Hermite draws over a set of drivers
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Scenarios:
    """A weighted set of ``S`` draws over the named drivers.

    ``draws[key]`` is a length-``S`` array of that driver's value in each scenario (a driver not
    sampled keeps its mean in every scenario); ``weight`` (length ``S``, sums to 1) is the quadrature
    weight — uniform ``1/S`` for Monte-Carlo. ``keys`` preserves order.
    """

    draws: dict
    weight: np.ndarray
    keys: list

    @property
    def S(self) -> int:
        return len(self.weight)


def monte_carlo(dists: dict, n: int = 2000, rng=None) -> Scenarios:
    """``n`` independent Monte-Carlo draws over ``dists`` ``{key: Normal}`` (uniform weights).

    Pass an explicit ``rng`` (``numpy.random.Generator``) for reproducibility — this module never
    seeds one for you.
    """
    if rng is None:
        rng = np.random.default_rng()
    keys = list(dists)
    draws = {k: (rng.normal(dists[k].mean, dists[k].sd, n) if dists[k].sd > 0
                 else np.full(n, dists[k].mean)) for k in keys}
    return Scenarios(draws=draws, weight=np.full(n, 1.0 / n), keys=keys)


def gauss_hermite(dists: dict, nodes: int = 7) -> Scenarios:
    """Tensor-product Gauss-Hermite quadrature over ``dists`` — ``nodes ** len(dists)`` scenarios.

    Exact for polynomials of the payoff in the (independent, normal) drivers, and *deterministic*
    (no RNG). Practical only for a **low-dimensional** slice (1–3 uncertain drivers); it raises if
    the tensor grid would exceed 20000 points — use :func:`monte_carlo` for the full per-node set.
    """
    keys = list(dists)
    if nodes ** len(keys) > 20000:
        raise ValueError(f"Gauss-Hermite grid {nodes}**{len(keys)} too large; use monte_carlo for "
                         f"{len(keys)} drivers")
    x, w = np.polynomial.hermite_e.hermegauss(nodes)     # nodes/weights for the standard normal
    w = w / w.sum()
    grids_x = np.meshgrid(*[x] * len(keys), indexing="ij")
    grids_w = np.meshgrid(*[w] * len(keys), indexing="ij")
    weight = np.prod([g.ravel() for g in grids_w], axis=0)
    draws = {k: dists[k].mean + dists[k].sd * grids_x[i].ravel() for i, k in enumerate(keys)}
    return Scenarios(draws=draws, weight=weight, keys=keys)


# ──────────────────────────────────────────────────────────────────────────
# Propagation — clear the DC-OPF for each scenario, induce the outcome
# ──────────────────────────────────────────────────────────────────────────
def clear(pt, cost=None, load=None, *, fleet=None, loads=None, shed_price=150.0):
    """Clear one DC-OPF scenario. ``cost`` ``{gen: $/MWh}`` and ``load`` ``{bus: MW}`` override the
    base ``fleet`` / ``loads`` (defaults ``DEFAULT_GEN_FLEET`` / ``DEFAULT_LOADS``); returns the
    ``EngineResult``."""
    fleet = wm.DEFAULT_GEN_FLEET if fleet is None else fleet
    loads = wm.DEFAULT_LOADS if loads is None else loads
    f = {g: {**spec, "cost": (cost or {}).get(g, spec["cost"])} for g, spec in fleet.items()}
    ld = {b: (load or {}).get(b, mw) for b, mw in loads.items()}
    eng = wm.make_engine("UNIFIED", buses=pt.buses, gen_fleet=f, loads=ld)
    return solve_engine_dispatch(pt, eng, shed_price=shed_price)


def propagate(pt, scen: Scenarios, quantity, *, fleet=None, loads=None, shed_price=150.0):
    """Run ``quantity(res)`` over every scenario's clearing; return ``(values, weights)``.

    ``quantity`` is a callable on the ``EngineResult`` (e.g. :func:`path_congestion`). For scenario
    ``s`` the drawn ``('cost', g)`` / ``('load', b)`` drivers override the base fleet/loads, the
    DC-OPF is cleared, and ``quantity`` is evaluated. The returned ``values`` (length ``S``) carry
    the scenario ``weights`` — feed both to :func:`certainty_equivalent`, :func:`emean`, etc.
    """
    vals = np.empty(scen.S)
    for s in range(scen.S):
        cost = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "cost"}
        load = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "load"}
        vals[s] = float(quantity(clear(pt, cost=cost, load=load, fleet=fleet, loads=loads,
                                        shed_price=shed_price)))
    return vals, scen.weight


def path_congestion(source, sink):
    """A ``quantity`` for :func:`propagate`: the price separation ``lambda_sink - lambda_source`` a
    point-to-point right on ``source -> sink`` spans (the congestion the hedge pays)."""
    source, sink = str(source), str(sink)
    return lambda res: res.lmp[sink] - res.lmp[source]


def induced_lmps(pt, scen: Scenarios, buses=None, *, fleet=None, loads=None, shed_price=150.0):
    """The induced **nodal LMP** at every bus, one DC-OPF solve per scenario (one pass).

    Returns ``(lmps, weights, buses)`` where ``lmps`` is ``(S, n_bus)`` — column ``j`` is the LMP
    distribution at ``buses[j]`` (default ``pt.buses``) across the scenarios, carrying ``weights``.
    Congestion on a path is a column difference: ``lmps[:, buses.index(k)] - lmps[:, buses.index(s)]``,
    so the whole nodal picture comes from a single propagation rather than one solve per quantity.
    """
    buses = list(pt.buses) if buses is None else [str(b) for b in buses]
    lmps = np.empty((scen.S, len(buses)))
    for s in range(scen.S):
        cost = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "cost"}
        load = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "load"}
        res = clear(pt, cost=cost, load=load, fleet=fleet, loads=loads, shed_price=shed_price)
        lmps[s] = [res.lmp[b] for b in buses]
    return lmps, scen.weight, buses


# ──────────────────────────────────────────────────────────────────────────
# Weighted statistics and risk preferences
# ──────────────────────────────────────────────────────────────────────────
def emean(values, weights) -> float:
    """Weighted expectation ``E[X]``."""
    return float(np.sum(weights * values))


def evar(values, weights) -> float:
    """Weighted variance ``Var[X]``."""
    m = emean(values, weights)
    return float(np.sum(weights * (values - m) ** 2))


def equantile(values, weights, q: float) -> float:
    """Weighted ``q``-quantile (``q`` in [0,1]) — for fan charts of the payoff distribution."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w)
    return float(np.interp(q, cw, v))


def certainty_equivalent(values, weights, gamma: float) -> float:
    """Mean-variance certainty equivalent ``CE = E[X] - (gamma/2) Var[X]`` (the repo canon)."""
    return emean(values, weights) - 0.5 * gamma * evar(values, weights)


def risk_loaded_premium(cong_values, weights, gamma: float) -> float:
    """The **risk-loaded hedge value** of a congestion exposure — the premium a risk-averse customer
    will pay for a right that pays the (random) congestion ``c = lambda_k - lambda_s``.

    Unhedged the customer bears ``-c``; hedged it pays a certain ``pi`` and receives ``c``, for a
    certain ``-pi``. It hedges while ``-pi >= CE(-c) = -E[c] - (gamma/2) Var[c]``, so its maximum
    willingness to pay is

        pi = E[c] + (gamma/2) Var[c].

    At ``gamma = 0`` this is the **expected** congestion; ``gamma > 0`` loads it by the variance — the
    reason a bid can clear on *hedge value* rather than any single realized congestion. This is the
    premium the 411 auction takes as each customer's bid.
    """
    return emean(cong_values, weights) + 0.5 * gamma * evar(cong_values, weights)


# ──────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from seams_engine import compute_ptdf

    ptC = compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")
    rng = np.random.default_rng(0)

    # Uncertain drivers: all loads + all generator costs.
    dists = {**load_dists(), **cost_dists()}
    print("drivers:", [f"{k[0]}:{k[1]}" for k in dists])

    # Induce the congestion distribution on 3->7 (the corridor-spanning path) by Monte-Carlo.
    vals, w = propagate(ptC, monte_carlo(dists, n=400, rng=rng), path_congestion("3", "7"))
    print(f"\ninduced congestion 3->7:  E = ${emean(vals, w):.2f}/MW   sd = ${evar(vals, w) ** .5:.2f}"
          f"   5-95% = [{equantile(vals, w, .05):.1f}, {equantile(vals, w, .95):.1f}]")
    for g in GAMMA_LADDER:
        print(f"  gamma={g:<6}  CE(hedge) = ${certainty_equivalent(vals, w, g):7.2f}   "
              f"risk-loaded premium = ${risk_loaded_premium(vals, w, g):7.2f}/MW")

    # Gauss-Hermite on a 2-driver slice (bus-1 cost + bus-7 load) — deterministic.
    slice2 = {("cost", "gen_slack_0"): Normal(50.0, 12.0), ("load", "7"): Normal(100.0, 15.0)}
    v2, w2 = propagate(ptC, gauss_hermite(slice2, nodes=7), path_congestion("3", "7"))
    print(f"\nGauss-Hermite (49 pts) congestion 3->7:  E = ${emean(v2, w2):.2f}/MW   "
          f"premium(gamma=.02) = ${risk_loaded_premium(v2, w2, 0.02):.2f}/MW")
