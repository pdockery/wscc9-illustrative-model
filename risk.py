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
payoff scale — they are documented, not sacred. The per-bus spreads resolve through the ambient
case (``case.risk_load_sd`` / ``case.risk_cost_sd``); the 9-bus values below stay the canonical
knobs, mirrored by ``WSCC9Case``.

**Factor drivers (the per-BA structure).** On a many-bus case, independent per-bus draws are a
trap: a balancing authority's aggregate load error shrinks like ``1/sqrt(n_buses)``, so a 329-bus
BA would show ~18x less relative risk than a 1-bus BA purely as an artefact of resolution. Every
real driver is therefore **a common factor plus an idiosyncratic residual**:

    d_n(s)  = d̄_n  · (1 + ε_{a(n)}(s) + η_n(s))      ε_a common per BA, η_n idiosyncratic
    c_g(s)  = c̄_g  · (1 + φ_carrier(s) + ψ_{a(g)}(s))
    p̄_g(s) = p_nom_g · cf(s)                          cf logit-normal per (BA, carrier)

Scenario keys carry the structure: alongside the absolute-level kinds ``('load', bus)`` /
``('cost', gen)``, the propagation understands the **relative factor kinds** ``('lf', ba)`` /
``('li', bus)`` (load common / idiosyncratic), ``('fuel', carrier)`` / ``('ca', ba)`` (cost
common by carrier / by BA), and ``('cf', ba, carrier)`` (available-capacity fraction). Factor
membership resolves through ``case.bus_to_area`` and each generator's ``carrier`` tag; a factor
that matches nothing **raises** rather than silently dropping out. :func:`ba_drivers` builds the
whole structure as a :class:`DriverModel`; :func:`day_bootstrap` draws jointly-consistent base
scenarios from historical whole days (preserving intra-day shape and every cross-BA correlation
with no covariance matrix to defend), with an ``overlay`` of factor shocks for forecast error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seams_engine import active as _case, solve_engine_dispatch


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


@dataclass
class LogitNormal:
    """A (0,1)-valued driver ``x = sigmoid(N(logit(mean), sd))`` — the canonical shape for an
    **available-capacity fraction** (``('cf', ba, carrier)``): bounded on both sides, right-skewed
    near 1 (a mostly-available fleet loses capacity in lumps), degenerate at ``sd=0``. ``mean`` is
    stated on the (0,1) scale; ``sd`` on the logit scale."""

    mean: float
    sd: float


def _logit(p: float) -> float:
    p = float(p)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def load_dists(loads: dict | None = None, sd: dict | None = None) -> dict[tuple[str, str], Normal]:
    """Per-bus load drivers ``{('load', bus): Normal(mean, sd)}``. Mean = the case loads **updated
    by** ``loads``; sd = the case's ``risk_load_sd`` (9-bus: ``DEFAULT_LOAD_SD``) **updated by**
    ``sd`` — both keyed by bus, so a *partial* override tweaks only the named buses and leaves the
    rest at their defaults."""
    c = _case()
    means = {**c.loads, **(loads or {})}
    spreads = {**getattr(c, "risk_load_sd", DEFAULT_LOAD_SD), **(sd or {})}
    return {("load", b): Normal(float(mw), float(spreads.get(b, 0.0))) for b, mw in means.items()}


def cost_dists(fleet: dict | None = None, cost: dict | None = None,
               sd: dict | None = None) -> dict[tuple[str, str], Normal]:
    """Per-gen cost drivers ``{('cost', gen): Normal(mean, sd)}``. Mean = ``cost[gen]`` if supplied
    else the fleet cost; sd = ``sd[gen]`` if supplied else the case's ``risk_cost_sd`` (9-bus:
    ``DEFAULT_COST_SD``). ``cost`` and ``sd`` are keyed by **generator id** — ``gen_slack_0`` =
    bus 1, ``gen_0`` = bus 2, ``gen_1`` = bus 3."""
    c = _case()
    fleet = c.gen_fleet if fleet is None else fleet
    spreads = {**getattr(c, "risk_cost_sd", DEFAULT_COST_SD), **(sd or {})}
    cost = cost or {}
    return {("cost", g): Normal(float(cost.get(g, spec["cost"])), float(spreads.get(g, 0.0)))
            for g, spec in fleet.items()}


# ──────────────────────────────────────────────────────────────────────────
# Factor drivers — the per-BA common + idiosyncratic structure
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Driver:
    """One named uncertain factor: a scenario ``key`` (see the module docstring's kind
    vocabulary) and its marginal ``dist`` (:class:`Normal` relative shock, mean 0, or
    :class:`LogitNormal` fraction for ``cf``)."""

    key: tuple
    dist: object


@dataclass
class DriverModel:
    """A bundle of :class:`Driver` factors plus the membership metadata that resolves them.

    ``dists`` is exactly the ``{key: dist}`` mapping :func:`monte_carlo` / :func:`gauss_hermite`
    consume, so a model slots into the existing scenario machinery unchanged; ``bus_to_area`` is
    carried so the same object can be handed to :func:`propagate` for factor resolution."""

    dists: dict
    bus_to_area: dict = field(default_factory=dict)

    @property
    def drivers(self) -> list:
        return [Driver(k, d) for k, d in self.dists.items()]

    def sample(self, n: int = 200, rng=None) -> "Scenarios":
        """Monte-Carlo draws over every factor (independent across factors; the common/idio
        *structure* is what creates cross-bus correlation after expansion)."""
        return monte_carlo(self.dists, n=n, rng=rng)

    def quadrature(self, nodes: int = 7) -> "Scenarios":
        """Gauss-Hermite tensor grid over the factors (low-dimensional slices only)."""
        return gauss_hermite(self.dists, nodes=nodes)


def ba_drivers(*, bus_to_area: dict | None = None, loads: dict | None = None,
               load_sd: dict | None = None, load_idio_sd=None,
               fuel_sd: dict | None = None, ca_sd: dict | None = None,
               cf: dict | None = None) -> DriverModel:
    """Build the per-BA common-factor + idiosyncratic :class:`DriverModel`.

    Parameters (all keyword-only; omit any block you don't want)
    ----------
    bus_to_area : {bus: ba} — default ``case.bus_to_area``; required (non-empty) for any
        BA-keyed factor.
    load_sd : {ba: sigma} — common load factors ``('lf', ba): Normal(0, sigma)``.
    load_idio_sd : float or {bus: sigma} — idiosyncratic ``('li', bus)`` shocks; a float applies
        to every load bus (``loads`` defaults to the case loads).
    fuel_sd : {carrier: sigma} — ``('fuel', carrier)`` cost shocks common to a fuel.
    ca_sd : {ba: sigma} — ``('ca', ba)`` cost shocks common to a BA's fleet.
    cf : {(ba, carrier): (mean, sd)} — ``('cf', ba, carrier): LogitNormal(mean, sd)`` available-
        capacity fractions; either slot may be ``None`` for "any".
    """
    c = _case()
    if bus_to_area is None:
        bus_to_area = dict(getattr(c, "bus_to_area", {}) or {})
    dists: dict = {}
    if load_sd:
        if not bus_to_area:
            raise ValueError("load_sd is BA-keyed but no bus_to_area map is available")
        dists.update({("lf", str(ba)): Normal(0.0, float(s)) for ba, s in load_sd.items()})
    if load_idio_sd:
        idio = ({b: load_idio_sd for b in (loads if loads is not None else c.loads)}
                if np.isscalar(load_idio_sd) else load_idio_sd)
        dists.update({("li", str(b)): Normal(0.0, float(s)) for b, s in idio.items() if s})
    if fuel_sd:
        dists.update({("fuel", str(cr)): Normal(0.0, float(s)) for cr, s in fuel_sd.items()})
    if ca_sd:
        if not bus_to_area:
            raise ValueError("ca_sd is BA-keyed but no bus_to_area map is available")
        dists.update({("ca", str(ba)): Normal(0.0, float(s)) for ba, s in ca_sd.items()})
    if cf:
        for (ba, cr), (m, s) in cf.items():
            dists[("cf", None if ba is None else str(ba),
                   None if cr is None else str(cr))] = LogitNormal(float(m), float(s))
    return DriverModel(dists=dists, bus_to_area=bus_to_area)


# ──────────────────────────────────────────────────────────────────────────
# Scenarios — Monte-Carlo and Gauss-Hermite draws over a set of drivers
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Scenarios:
    """A weighted set of ``S`` draws over the named drivers.

    ``draws[key]`` is a length-``S`` array of that driver's value in each scenario (a driver not
    sampled keeps its mean in every scenario); ``weight`` (length ``S``, sums to 1) is the quadrature
    weight — uniform ``1/S`` for Monte-Carlo. ``keys`` preserves order. ``meta`` carries optional
    per-scenario bookkeeping that is **not** a driver (e.g. :func:`day_bootstrap` records the drawn
    ``day`` and ``hour``).
    """

    draws: dict
    weight: np.ndarray
    keys: list
    meta: dict = field(default_factory=dict)

    @property
    def S(self) -> int:
        return len(self.weight)


def _draw_mc(dist, n: int, rng) -> np.ndarray:
    """One driver's ``n`` Monte-Carlo draws (Normal on its own scale; LogitNormal through the
    sigmoid; ``sd=0`` degenerates to the mean either way)."""
    if isinstance(dist, LogitNormal):
        if dist.sd > 0:
            return _sigmoid(rng.normal(_logit(dist.mean), dist.sd, n))
        return np.full(n, float(dist.mean))
    return rng.normal(dist.mean, dist.sd, n) if dist.sd > 0 else np.full(n, dist.mean)


def monte_carlo(dists: dict, n: int = 2000, rng=None) -> Scenarios:
    """``n`` independent Monte-Carlo draws over ``dists`` ``{key: Normal|LogitNormal}`` (uniform
    weights).

    Pass an explicit ``rng`` (``numpy.random.Generator``) for reproducibility — this module never
    seeds one for you.
    """
    if rng is None:
        rng = np.random.default_rng()
    keys = list(dists)
    draws = {k: _draw_mc(dists[k], n, rng) for k in keys}
    return Scenarios(draws=draws, weight=np.full(n, 1.0 / n), keys=keys)


def _gh_nodes(dist, x: np.ndarray) -> np.ndarray:
    """Map standard-normal quadrature nodes onto one driver's scale (a LogitNormal is a smooth
    transform of a normal, so the same nodes/weights integrate it exactly as a function of the
    underlying normal)."""
    if isinstance(dist, LogitNormal):
        return _sigmoid(_logit(dist.mean) + dist.sd * x)
    return dist.mean + dist.sd * x


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
    draws = {k: _gh_nodes(dists[k], grids_x[i].ravel()) for i, k in enumerate(keys)}
    return Scenarios(draws=draws, weight=weight, keys=keys)


def day_bootstrap(days, n: int = 200, rng=None, hours=None, overlay: dict | None = None) -> Scenarios:
    """Block-bootstrap :class:`Scenarios` over **historical whole days** — the base case for a
    bundle-backed case, per the module docstring: drawing a day (and an hour within it) keeps
    intra-day shape and every cross-driver correlation exactly, with no covariance matrix.

    Parameters
    ----------
    days : a case exposing ``historical_days()``, or the mapping itself —
        ``{day_id: {driver_key: 24-array | scalar}}``. Every day must carry the same keys;
        array-valued entries are hourly profiles, scalars hold for the whole day.
    n : number of scenarios; each is one (day, hour) draw, uniform with replacement.
    rng : ``numpy.random.Generator`` — pass one explicitly for reproducibility.
    hours : ``None`` (uniform over the profile hours), an int (that fixed hour), or a sequence
        to draw from (e.g. ``range(16, 21)`` for evening-peak conditioning).
    overlay : ``{driver_key: Normal|LogitNormal}`` extra factors drawn **independently on top** of
        the historical base — the forecast-error layer (e.g. ``('lf', ba)`` shocks around the
        drawn day). A key already present in the days raises rather than being silently doubled.

    The drawn ``day`` and ``hour`` land in ``Scenarios.meta`` for conditioning and fan charts.
    """
    if hasattr(days, "historical_days"):
        days = days.historical_days()
    if not days:
        raise ValueError("day_bootstrap needs at least one historical day")
    if rng is None:
        rng = np.random.default_rng()
    day_ids = list(days)
    keys = list(days[day_ids[0]])
    for d in day_ids:
        if list(days[d]) != keys:
            raise ValueError(f"day {d!r} carries different driver keys than {day_ids[0]!r}")
    n_hours = max([len(np.atleast_1d(v)) for v in days[day_ids[0]].values()] or [1])

    di = rng.integers(0, len(day_ids), n)
    if hours is None:
        hi = rng.integers(0, n_hours, n)
    elif np.isscalar(hours):
        hi = np.full(n, int(hours))
    else:
        pool = np.asarray(list(hours), dtype=int)
        hi = pool[rng.integers(0, len(pool), n)]
    if n_hours and (hi.max(initial=0) >= n_hours or hi.min(initial=0) < 0):
        raise ValueError(f"hour draw outside the 0..{n_hours - 1} profile range")

    draws = {}
    for k in keys:
        vals = np.empty(n)
        for s in range(n):
            v = days[day_ids[di[s]]][k]
            vals[s] = float(np.atleast_1d(v)[hi[s]] if np.ndim(v) else v)
        draws[k] = vals
    if overlay:
        for k, dist in overlay.items():
            if k in draws:
                raise ValueError(f"overlay key {k!r} already drawn from the historical days")
            draws[k] = _draw_mc(dist, n, rng)
    return Scenarios(draws=draws, weight=np.full(n, 1.0 / n), keys=list(draws),
                     meta={"day": np.array([day_ids[i] for i in di]), "hour": hi})


# ──────────────────────────────────────────────────────────────────────────
# Propagation — clear the DC-OPF for each scenario, induce the outcome
# ──────────────────────────────────────────────────────────────────────────
def clear(pt, cost=None, load=None, *, fleet=None, loads=None, shed_price=150.0,
          pmax=None, flowgates=None):
    """Clear one DC-OPF scenario. ``cost`` ``{gen: $/MWh}``, ``load`` ``{bus: MW}`` and ``pmax``
    ``{gen: MW}`` override the base ``fleet`` / ``loads`` (defaults from the ambient case);
    ``flowgates`` forwards a ``FlowgateSet`` to the solver. Returns the ``EngineResult``."""
    c = _case()
    fleet = c.gen_fleet if fleet is None else fleet
    loads = c.loads if loads is None else loads
    if pmax:
        f = {g: {**spec, "cost": (cost or {}).get(g, spec["cost"]),
                 "p_nom": float(pmax.get(g, spec["p_nom"]))} for g, spec in fleet.items()}
    else:
        f = {g: {**spec, "cost": (cost or {}).get(g, spec["cost"])} for g, spec in fleet.items()}
    ld = {b: (load or {}).get(b, mw) for b, mw in loads.items()}
    eng = c.make_engine("UNIFIED", buses=pt.buses, gen_fleet=f, loads=ld)
    return solve_engine_dispatch(pt, eng, shed_price=shed_price, flowgates=flowgates)


def _gen_ba(spec, bus_to_area):
    """A generator's BA: its own ``'ba'`` tag when the fleet carries one (a
    unit can connect at another BA's bus — common on the real West), else the
    BA of its bus."""
    return spec.get("ba", bus_to_area.get(str(spec["bus"])))


def _factor_members(keys, fleet, loads, bus_to_area) -> dict:
    """Resolve each **factor** key to the ids it hits — buses for the load kinds, generators for
    the cost/cf kinds; the absolute kinds need no resolution. An unknown kind, or a factor that
    matches nothing, **raises**: a silently-dropped driver is a wrong answer, not a default."""
    members: dict = {}
    for k in keys:
        kind = k[0]
        if kind in ("load", "cost"):
            continue
        if kind == "lf":
            ids = [b for b in loads if bus_to_area.get(str(b)) == k[1]]
        elif kind == "li":
            ids = [k[1]] if k[1] in loads else []
        elif kind == "fuel":
            ids = [g for g, s in fleet.items() if s.get("carrier") == k[1]]
        elif kind == "ca":
            ids = [g for g, s in fleet.items() if _gen_ba(s, bus_to_area) == k[1]]
        elif kind == "cf":
            ba, carrier = k[1], k[2]
            ids = [g for g, s in fleet.items()
                   if (ba is None or _gen_ba(s, bus_to_area) == ba)
                   and (carrier is None or s.get("carrier") == carrier)]
        else:
            raise ValueError(f"unknown driver kind in scenario key {k!r}")
        if not ids:
            raise ValueError(f"driver {k!r} matches no bus/generator "
                             f"(check bus_to_area / the fleet's 'carrier' tags)")
        members[k] = ids
    return members


def _scenario_overrides(scen: Scenarios, s: int, fleet, loads, members):
    """Scenario ``s``'s engine-level ``(cost, load, pmax)`` overrides: absolute draws first, then
    the factor shocks composed per the module docstring — load/cost shocks **add** within a
    bus/gen (``base·(1+ε+η)``), cf fractions **multiply** if two hit the same generator."""
    cost = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "cost"}
    load = {k[1]: scen.draws[k][s] for k in scen.keys if k[0] == "load"}
    if not members:
        return cost, load, None
    lsh: dict = {}
    csh: dict = {}
    pmax: dict = {}
    for k, ids in members.items():
        v = scen.draws[k][s]
        if k[0] in ("lf", "li"):
            for b in ids:
                lsh[b] = lsh.get(b, 0.0) + v
        elif k[0] in ("fuel", "ca"):
            for g in ids:
                csh[g] = csh.get(g, 0.0) + v
        else:                                   # 'cf'
            for g in ids:
                pmax[g] = pmax.get(g, float(fleet[g]["p_nom"])) * v
    for b, e in lsh.items():
        load[b] = load.get(b, float(loads[b])) * (1.0 + e)
    for g, e in csh.items():
        cost[g] = cost.get(g, float(fleet[g]["cost"])) * (1.0 + e)
    return cost, load, (pmax or None)


def propagate(pt, scen: Scenarios, quantity, *, fleet=None, loads=None, shed_price=150.0,
              flowgates=None, bus_to_area=None):
    """Run ``quantity(res)`` over every scenario's clearing; return ``(values, weights)``.

    ``quantity`` is a callable on the ``EngineResult`` (e.g. :func:`path_congestion`). For scenario
    ``s`` the drawn drivers — absolute ``('cost', g)`` / ``('load', b)`` levels and the relative
    factor kinds (``lf``/``li``/``fuel``/``ca``/``cf``, resolved through ``bus_to_area``, default
    the case's) — override the base fleet/loads, the DC-OPF is cleared (with ``flowgates`` if
    given), and ``quantity`` is evaluated. The returned ``values`` (length ``S``) carry the
    scenario ``weights`` — feed both to :func:`certainty_equivalent`, :func:`emean`, etc.
    """
    c = _case()
    fleet = c.gen_fleet if fleet is None else fleet          # hoisted: one copy, not one per draw
    loads = c.loads if loads is None else loads
    members = _factor_members(scen.keys, fleet, loads,
                              c.bus_to_area if bus_to_area is None else bus_to_area)
    vals = np.empty(scen.S)
    for s in range(scen.S):
        cost, load, pmax = _scenario_overrides(scen, s, fleet, loads, members)
        vals[s] = float(quantity(clear(pt, cost=cost, load=load, fleet=fleet, loads=loads,
                                        shed_price=shed_price, pmax=pmax,
                                        flowgates=flowgates)))
    return vals, scen.weight


def path_congestion(source, sink):
    """A ``quantity`` for :func:`propagate`: the price separation ``lambda_sink - lambda_source`` a
    point-to-point right on ``source -> sink`` spans (the congestion the hedge pays)."""
    source, sink = str(source), str(sink)
    return lambda res: res.lmp[sink] - res.lmp[source]


def induced_lmps(pt, scen: Scenarios, buses=None, *, fleet=None, loads=None, shed_price=150.0,
                 flowgates=None, bus_to_area=None):
    """The induced **nodal LMP** at every bus, one DC-OPF solve per scenario (one pass).

    Returns ``(lmps, weights, buses)`` where ``lmps`` is ``(S, n_bus)`` — column ``j`` is the LMP
    distribution at ``buses[j]`` (default ``pt.buses``) across the scenarios, carrying ``weights``.
    Congestion on a path is a column difference: ``lmps[:, buses.index(k)] - lmps[:, buses.index(s)]``,
    so the whole nodal picture comes from a single propagation rather than one solve per quantity.
    Accepts the same factor kinds / ``flowgates`` / ``bus_to_area`` as :func:`propagate`.
    """
    c = _case()
    fleet = c.gen_fleet if fleet is None else fleet
    loads = c.loads if loads is None else loads
    members = _factor_members(scen.keys, fleet, loads,
                              c.bus_to_area if bus_to_area is None else bus_to_area)
    buses = list(pt.buses) if buses is None else [str(b) for b in buses]
    lmps = np.empty((scen.S, len(buses)))
    for s in range(scen.S):
        cost, load, pmax = _scenario_overrides(scen, s, fleet, loads, members)
        res = clear(pt, cost=cost, load=load, fleet=fleet, loads=loads, shed_price=shed_price,
                    pmax=pmax, flowgates=flowgates)
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
    ptC = _case().build({"line_4": 40.0}).pt
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

    # Per-BA factor drivers: a two-BA overlay on the 9 buses — one common load
    # factor per BA plus small idiosyncratic residuals (the WECC structure in miniature).
    B2A = {b: ("BA-1" if b in ("2", "8", "7", "6", "3") else "BA-2") for b in ptC.buses}
    dm = ba_drivers(bus_to_area=B2A, load_sd={"BA-1": 0.08, "BA-2": 0.12},
                    load_idio_sd=0.03)
    v3, w3 = propagate(ptC, dm.sample(300, rng), path_congestion("3", "7"), bus_to_area=B2A)
    print(f"per-BA factors ({len(dm.dists)} drivers) congestion 3->7:  "
          f"E = ${emean(v3, w3):.2f}/MW   sd = ${evar(v3, w3) ** .5:.2f}")

    # Day bootstrap over three synthetic historical days (base + overlay forecast error).
    hrs = np.arange(24.0)
    days = {f"d{i}": {("load", "7"): 90 + 10 * i + 8 * np.sin((hrs - 6 + i) * np.pi / 12)}
            for i in range(3)}
    scen = day_bootstrap(days, n=200, rng=rng, hours=range(16, 21),
                         overlay={("lf", "BA-1"): Normal(0.0, 0.05)})
    v4, w4 = propagate(ptC, scen, path_congestion("3", "7"), bus_to_area=B2A)
    print(f"day-bootstrap (evening hours, lf overlay):  E = ${emean(v4, w4):.2f}/MW   "
          f"days drawn = {sorted({str(d) for d in scen.meta['day']})}")
