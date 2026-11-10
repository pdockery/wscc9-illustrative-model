"""Golden-number harness for the library.

``lib/`` is a copy of the shared 9-bus library, and
``lib/MANIFEST.md`` records each module's SHA-256 and the date of the copy.
``tests/golden/golden.json`` is the Lab's frozen record of what the library
computes. This file reruns every scenario and compares, so a copy that
moves a number fails here. The suite needs no notebook.

Each scenario is a named function returning ``{label: object}``. ``canon()``
flattens whatever comes back -- DataFrames, Series, dicts, EngineResults -- into
a flat ``{dotted.key: number|string}`` map with floats rounded to 9 significant
digits. That is far inside solver noise but tight enough to catch a swapped
default or a re-ordered constraint.

The scenarios are fixed in this file. They were harvested from the
``# -- EDIT --`` cells of the Lab's notebooks, and the notebook numbers in the
comments below name those notebooks, which this repository does not carry.

Usage
-----
    python tests/golden_runner.py             # compare against tests/golden/golden.json
    python tests/golden_runner.py --only atc  # run a subset (substring match)
    python tests/golden_runner.py --write --local   # re-freeze, which this repository does not do

golden.json is copied with the library and re-frozen only by the Lab, so
``--write`` refuses without ``--local``.

Determinism: HiGHS and ``trust-constr`` are deterministic, and ``risk.monte_carlo``
requires an explicit ``rng``, so every scenario below is reproducible by
construction. ``--write`` refuses to run unless the suite reproduces itself twice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# the library is lib/ at the repository root; the runner also runs as a script,
# outside tests/conftest.py, so it puts lib/ on the path itself
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

GOLDEN = Path(__file__).resolve().parent / "golden" / "golden.json"

import atc                                     # noqa: E402
import footprints as fpmod                     # noqa: E402
import intertie                                # noqa: E402
import revenue_allocation as ra                # noqa: E402
import risk                                    # noqa: E402
import seams_engine as se                      # noqa: E402
import wscc9_model as wm                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalisation
# ─────────────────────────────────────────────────────────────────────────────
SIGFIGS = 9


def _num(v):
    """Round to SIGFIGS significant digits; normalise -0.0 and non-finites."""
    f = float(v)
    if not np.isfinite(f):
        return str(f)
    if f == 0.0:
        return 0.0
    return float(f"{f:.{SIGFIGS}g}")


def canon(obj, prefix: str = "") -> dict:
    """Flatten an arbitrary result object into {dotted.key: number|str}."""
    out: dict = {}

    def put(key, val):
        if isinstance(val, (bool, np.bool_)):
            out[key] = bool(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            out[key] = _num(val)
        elif val is None:
            out[key] = None
        else:
            out[key] = str(val)

    def walk(o, p):
        if isinstance(o, pd.DataFrame):
            for col in o.columns:
                for idx, val in o[col].items():
                    walk(val, f"{p}.{col}.{idx}")
        elif isinstance(o, pd.Series):
            for idx, val in o.items():
                walk(val, f"{p}.{idx}")
        elif isinstance(o, dict):
            for k, val in o.items():
                key = ".".join(str(x) for x in k) if isinstance(k, tuple) else str(k)
                walk(val, f"{p}.{key}" if p else key)
        elif isinstance(o, se.EngineResult):
            for f in ("dispatch", "gen_by_bus", "load_by_bus", "lmp", "line_dual",
                      "flow_own", "shed_by_bus", "area_prices", "transfers",
                      "transfer_duals"):
                v = getattr(o, f, None)
                if v:
                    walk(v, f"{p}.{f}")
            walk(o.energy_price, f"{p}.energy_price")
            walk(o.total_cost, f"{p}.total_cost")
            walk(o.status, f"{p}.status")
        elif isinstance(o, np.ndarray):
            for i, val in enumerate(o.ravel()):
                walk(val, f"{p}[{i}]")
        elif isinstance(o, (list, tuple)):
            for i, val in enumerate(o):
                walk(val, f"{p}[{i}]")
        else:
            put(p, o)

    walk(obj, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures (mirroring the notebooks' -- EDIT -- cells)
# ─────────────────────────────────────────────────────────────────────────────
RATINGS_L4 = {"line_4": 40.0}
SHED = 150.0

BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
BA_LINES = {"BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
            "BA-2": ["line_0", "line_1", "line_7", "line_8"]}
BA_COLORS = {"BA-1": "#993AFF", "BA-2": "#2471A3"}
SHARES_2BA = {"BA-1": 0.5, "BA-2": 0.5}

BA3_DEFS = {"BA-1": ["1", "4", "5"], "BA-2": ["2", "7", "8"], "BA-3": ["3", "6", "9"]}
BA3_COLORS = {"BA-1": "#993AFF", "BA-2": "#2471A3", "BA-3": "#117A2B"}

# The cycle partition of 112 section 4, as its cell 11 builds it, with one
# interface held by each authority, oriented so each loads line_4 forward.
TRI_DEFS = {"A": ["1", "4", "5"], "B": ["3", "6", "7"], "C": ["2", "8", "9"]}
TRI_MANAGE = {"A": ["line_0", "line_1"], "B": ["line_3", "line_4"],
              "C": ["line_6", "line_7"]}
TRI_COLORS = {"A": "#E67E22", "B": "#27AE60", "C": "#2471A3"}
TRI_HOLDS = {"A": ("B", "A"), "B": ("B", "C"), "C": ("A", "C")}

SEAM_DEFS = {"Market A": ["1", "4", "5", "9"], "Market B": ["2", "3", "6", "7", "8"]}
SEAM_MON = {"Market A": ["line_0", "line_1", "line_8"],
            "Market B": ["line_3", "line_4", "line_5", "line_6"]}


def _fp2(pt):
    return fpmod.make(pt, BA_DEFS, BA_COLORS, manage=BA_LINES, tie_label="tie")


def _fp3(pt):
    return fpmod.make(pt, BA3_DEFS, BA3_COLORS, tie_label="tie")


def _pt(ratings=None, split_5_6=False):
    return se.compute_ptdf(wm.build_network(ratings, split_5_6=split_5_6),
                           slack_bus="1")


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────
def s_base_unified():
    pt = _pt()
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    return {"res": res, "sf": wm.sf_table(pt), "widths": se.susceptance_widths(pt)}


def s_line4_unified():
    pt = _pt(RATINGS_L4)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    return {"res": res, "phys": se.physical_flows(pt, [res])}


def s_line4_allocation():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    alloc, summary, lr, sep = ra.allocate_congestion_rent(
        fp, res, pt, wm.DEFAULT_LOADS)
    return {"alloc": alloc, "summary": summary, "lr": lr, "sep": sep,
            "revtab": ra.revenue_table(fp, res, pt, wm.DEFAULT_LOADS),
            "cmp": ra.compare_methods(fp, res, pt, wm.DEFAULT_LOADS),
            "settle": ra.ba_settlement(fp, res, wm.DEFAULT_LOADS),
            "bybus": ra.settlement_by_bus(res, pt.buses, wm.DEFAULT_LOADS),
            "linerent": ra.line_rent_table(fp, res, pt),
            "border": ra.border_separation(fp, res, pt),
            "capval": ra.capacity_value_by_line(res, pt, pt.lines),
            "own": ra.ownership_allocation(fp, res, pt)}


def s_hub_decomposition():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    return {"w_auto": ra.hub_weights(fp, "auto"),
            "w_gap": ra.hub_weights(fp, "gap"),
            "w_elap": ra.hub_weights(fp, "elap", loads=wm.DEFAULT_LOADS),
            "nc": ra.net_congestion_position(fp, res),
            "reg": ra.regional_congestion(fp, res, pt),
            "te": ra.transfer_settlement(fp, res),
            "hubdec": ra.hub_congestion_decomposition(fp, res, pt),
            "shift": ra.congestion_shift_breakdown(fp, res, pt),
            "summary": ra.congestion_summary(fp, res, pt),
            "parallel": ra.parallel_flow_attribution(fp, res, pt),
            "loop": ra.loop_flow_charge(fp, res, pt, "BA-1", "BA-2")}


def s_transfers_mod29():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    n, p, eng, res = ra.solve_with_transfers(fp, RATINGS_L4, limits="mod29",
                                             shed_price=SHED)
    lim, sft = ra.mod29_interface_limits(fp, p)
    return {"res": res, "limits": lim, "sft": sft,
            "ifaces": ra.fp_interfaces(fp, p),
            "pair": ra.pairwise_transfer_settlement(fp, res, p),
            "rent": ra.transfer_rent(res),
            "rent_by": ra.transfer_rent_by_interface(res),
            "hubdec": ra.hub_congestion_decomposition(fp, res, p)}


def s_transfers_scalar():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    n, p, eng, res = ra.solve_with_transfers(fp, RATINGS_L4, limits=120.0,
                                             shed_price=SHED)
    return {"res": res,
            # t_method keys are ints 1/2/3 (see ra.T_NAMES). NOTE: the T1 branch
            # indexes transfer_split directly, so it requires a {ba: share} dict
            # -- it does NOT route through _ba_shares, so the 'even'/'interchange'
            # /scalar spec forms documented for that helper do not reach T1.
            "alloc_T1": ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS,
                                                  1, SHARES_2BA, pt=p),
            "alloc_T2": ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS,
                                                  2, SHARES_2BA, pt=p),
            "alloc_T3": ra.allocate_transfer_rent(fp, res, wm.DEFAULT_LOADS,
                                                  3, SHARES_2BA, pt=p),
            "pair": ra.pairwise_transfer_settlement(fp, res, p),
            "pos": ra.position_ledger(fp, res)}


def s_three_ba():
    pt = _pt(RATINGS_L4)
    fp = _fp3(pt)
    n, p, eng, res = ra.solve_with_transfers(
        fp, RATINGS_L4, limits="mod29", gen_fleet=wm.FLEET_3BA,
        loads=wm.LOADS_3BA, shed_price=SHED)
    return {"res": res,
            "hubdec": ra.hub_congestion_decomposition(fp, res, p,
                                                      loads=wm.LOADS_3BA),
            "pair": ra.pairwise_transfer_settlement(fp, res, p,
                                                    loads=wm.LOADS_3BA,
                                                    gen_fleet=wm.FLEET_3BA),
            "nc": ra.net_congestion_position(fp, res, loads=wm.LOADS_3BA,
                                             gen_fleet=wm.FLEET_3BA),
            "limits": ra.mod29_interface_limits(fp, p, loads=wm.LOADS_3BA,
                                                gen_fleet=wm.FLEET_3BA)[0]}


def s_independent():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    resU = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    _pti, indep = ra.independent_clear(fp, RATINGS_L4, shed_price=SHED)
    alloc, summary, lr, sep = ra.allocate_congestion_rent(
        fp, resU, pt, wm.DEFAULT_LOADS)
    return {"indep": indep,
            "avu": ra.autarky_vs_unified_congestion(fp, "method1", alloc, indep,
                                                    resU, pt),
            "mled": ra.methodology_ledger_congestion(fp, alloc, indep, resU, pt)}


def s_atc():
    pt = _pt()
    fp = _fp2(pt)
    awards = [atc.Award("3", "5", 40.0), atc.Award("2", "9", 30.0)]
    ttc_mw, ttc_line = atc.ttc(pt, "3", "5")
    ok_n, df_n = atc.simultaneous_feasibility(pt, awards)
    ok_f, df_f = atc.firm_feasibility(pt, awards)
    return {"psf_3_5": atc.path_shift_factors(pt, "3", "5"),
            "ttc_mw": ttc_mw, "ttc_line": ttc_line,
            "atc_identity": atc.atc(ttc_mw, etc=10.0, trm=5.0, cbm=2.0),
            "flow": atc.flow_dict(pt, awards),
            "loadings": atc.line_loadings(pt, awards),
            "firm_loadings": atc.firm_line_loadings(pt, awards),
            "sft_netted_ok": ok_n, "sft_netted": df_n,
            "sft_firm_ok": ok_f, "sft_firm": df_f,
            "overloaded": atc.overloaded_lines(pt, awards),
            "avail": atc.available_atc(pt, "3", "5", awards),
            "booked": [(a.source, a.sink, a.mw)
                       for a in atc.book_sequentially(
                           pt, [atc.Award("3", "5", 60.0),
                                atc.Award("2", "9", 60.0)])],
            "ba_mon": atc.ba_monitored_lines(pt, fp, "BA-1"),
            "ba_atc": atc.ba_atc(pt, fp, "BA-1", "3", "5")}


def s_atc_interfaces():
    pt = _pt({"line_4": 40.0})
    fp = _fp3(pt)
    w = ra.hub_weights(fp, "auto", loads=wm.LOADS_3BA, gen_fleet=wm.FLEET_3BA)
    ifc = ra.fp_interfaces(fp, pt)
    ttcs = atc.interface_ttc(pt, ifc, w)
    ratings = {k: v[0] for k, v in ttcs.items()}
    return {"ttc": {f"{a}|{b}": v for (a, b), v in ttcs.items()},
            "sft_worst": atc.interface_sft(pt, ratings, w, direction="worst"),
            "sft_sorted": atc.interface_sft(pt, ratings, w, direction="sorted"),
            "hub_psf": atc.hub_path_shift_factors(pt, w["BA-1"], w["BA-2"])}


def _round6(obj):
    """Every float in a scenario value rounded to 1e-6.

    The least-squares rule is an interior-point solve, so its last few digits
    are solver noise and freezing them would pin the solver's build rather than
    the rule. The proportional and uniform arms are exact arithmetic and are
    stored at full precision.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.round(6)
    if isinstance(obj, dict):
        return {k: round(float(v), 6) for k, v in obj.items()}
    return round(float(obj), 6)


def s_rights_books():
    """The triangle of 112 section 4 as three rights books, both methods, the
    common test on both readings and all three reduction rules.

    Every least-squares output is rounded to 1e-6 before it is stored, because
    the rule solves a quadratic program whose last digits are solver noise.
    """
    pt = _pt(RATINGS_L4)
    fp = fpmod.make(pt, TRI_DEFS, TRI_COLORS, manage=TRI_MANAGE)
    lim, _sft = ra.mod29_interface_limits(fp, pt)
    w = ra.hub_weights(fp, "auto")
    ident = se.identity_flowgates(pt)

    queues = {}
    for holder, (src, snk) in TRI_HOLDS.items():
        key = tuple(sorted((src, snk)))
        queues[holder] = [atc.Request(w[src], w[snk],
                                      float(lim.loc[key, "TTC (MW)"]),
                                      holder=holder, label=f"{src}->{snk}",
                                      source_name=src, sink_name=snk)]

    out = {}
    books = {"rsp": [], "afc": []}
    for holder, q in queues.items():
        rsp = atc.rights_book_rsp(pt, q, "all")
        afc = atc.rights_book_afc(pt, q, ident, rf_min=0.05)
        books["rsp"].append(rsp)
        books["afc"].append(afc)
        out[f"rsp_ledger_{holder}"] = rsp.ledger
        out[f"rsp_granted_{holder}"] = rsp.granted_mw
        out[f"afc_granted_{holder}"] = afc.granted_mw
        out[f"afc_ledger_{holder}"] = afc.ledger

    for method, bks in books.items():
        for direction in ("netted", "firm"):
            for rule in ("pro_rata", "uniform", "least_squares"):
                if rule == "least_squares" and direction == "firm":
                    continue
                nom = atc.nominate_and_test(bks, pt, ident,
                                            direction=direction,
                                            reduction=rule)
                tag = f"{method}_{direction}_{rule}"
                fix = _round6 if rule == "least_squares" else (lambda z: z)
                out[f"ent_{tag}"] = fix(nom.entitlements)
                out[f"byba_{tag}"] = fix(nom.by_ba)
                out[f"load_{tag}"] = fix(nom.loadings)
                out[f"duals_{tag}"] = fix(nom.duals)
                out[f"iters_{tag}"] = nom.iterations
    return out


def s_crr_auction():
    pt = _pt(RATINGS_L4)
    bids = [atc.Bid("3", "5", 40.0, 12.0, label="A", group="g1"),
            atc.Bid("2", "9", 35.0, 8.0, label="B", group="g1"),
            atc.Bid("3", "9", 25.0, 15.0, label="C", group="g2")]
    r = atc.crr_auction(pt, bids)
    r2 = atc.crr_auction(pt, bids, obligations={"g1": 50.0})
    return {"awards": r.awards, "mu": r.mu, "rent": r.rent, "value": r.value,
            "obl_awards": r2.awards, "obl_rent": r2.rent, "obl_value": r2.value,
            "rights": atc.awarded_rights(r)}


def s_qp():
    pt = _pt(RATINGS_L4)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    curves = {"gen_slack_0": (50.0, 0.10), "gen_0": (35.0, 0.05),
              "gen_1": (20.0, 0.03)}
    res = se.solve_engine_qp(pt, eng, curves=curves, shed_price=SHED)
    return {"res": res}


def s_two_engine_seam():
    pt = _pt(RATINGS_L4)
    fp = fpmod.make(pt, SEAM_DEFS,
                    {"Market A": "#993AFF", "Market B": "#2471A3"},
                    monitored=SEAM_MON, tie_label="seam")
    ra_eng = wm.make_engine("Market A", buses=SEAM_DEFS["Market A"],
                            activated=SEAM_MON["Market A"])
    rb_eng = wm.make_engine("Market B", buses=SEAM_DEFS["Market B"],
                            activated=SEAM_MON["Market B"])
    resA = se.solve_engine_dispatch(pt, ra_eng, shed_price=SHED)
    resB = se.solve_engine_dispatch(pt, rb_eng, shed_price=SHED)
    return {"A": resA, "B": resB,
            "phys": se.physical_flows(pt, [resA, resB]),
            "gap": se.seam_dual_gap([resA, resB], pt.buses),
            "kinds": {l: fp.line_kind(pt, l) for l in pt.lines}}


def s_exogenous():
    """The `exo=` representation -- the convention the WECC boundary will use."""
    pt = _pt(RATINGS_L4)
    eng = wm.make_engine("SUB", buses=["1", "4", "5", "9"])
    exo = {"6": 40.0, "8": -15.0}
    res = se.solve_engine_dispatch(pt, eng, exo=exo, shed_price=SHED)
    off = se.solve_engine_dispatch(pt, eng, exo=exo,
                                   flow_offsets={"line_4": 5.0},
                                   shed_price=SHED)
    return {"res": res, "with_offsets": off}


def s_self_schedule():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    return {"led": ra.self_schedule_ledger(fp, 120.0, "3", "9", 50.0,
                                           ratings=RATINGS_L4)}


def s_path_hedge():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    rights = [{"source": "3", "sink": "9", "mw": 30.0, "holder": "BA-2"}]
    return {"led": ra.path_hedge_ledger(fp, RATINGS_L4, rights)}


def s_provision():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    _pti, indep = ra.independent_clear(fp, RATINGS_L4, shed_price=SHED)
    rr = {"BA-1": 1000.0, "BA-2": 800.0}
    atc_by = {"BA-1": 50.0, "BA-2": 40.0}
    return {"rate": ra.embedded_rate(1000.0, 50.0),
            "sub_1": ra.scheduling_subscription(fp, res, "BA-1"),
            "sub_2": ra.scheduling_subscription(fp, res, "BA-2"),
            "recovery": ra.transmission_service_recovery(
                fp, indep, res, pt, embedded=rr, atc_by_ba=atc_by),
            "oatt": ra.oatt_vs_tac(fp, res, rr=rr, atc_by_ba=atc_by)}


def s_wheeling():
    """212's object: a transfer clearing in a transport market and wheeled by
    a contiguous one. rA/rB are EngineResults, not prices."""
    ptT = se.compute_ptdf(wm.build_transport_network(["2", "3"]), slack_bus="2")
    engT = wm.make_engine("Market A", buses=["2", "3"],
                          gen_fleet=wm.FLEET_3BA, loads=wm.LOADS_3BA)
    rA = se.solve_engine_dispatch(ptT, engT, shed_price=SHED)
    ptB = _pt(RATINGS_L4)
    rB = se.solve_engine_dispatch(ptB, wm.make_engine("Market B", buses=ptB.buses),
                                  shed_price=SHED)
    legs, summary = ra.wheeling_settlement("2", "3", 30.0, rA, rB)
    return {"legs": legs, "summary": summary}


def s_risk_gauss_hermite():
    pt = _pt(RATINGS_L4)
    dists = {**risk.load_dists(), **risk.cost_dists()}
    small = {k: dists[k] for k in list(dists)[:2]}
    scen = risk.gauss_hermite(small, nodes=3)
    vals, wts = risk.propagate(pt, scen, risk.path_congestion("3", "9"))
    lmps = risk.induced_lmps(pt, scen, buses=["5", "9"])
    return {"keys": [".".join(k) for k in scen.keys],
            "weight": scen.weight, "draws": scen.draws,
            "mean": risk.emean(vals, wts), "var": risk.evar(vals, wts),
            "q05": risk.equantile(vals, wts, 0.05),
            "ce": risk.certainty_equivalent(vals, wts, 0.005),
            "prem": risk.risk_loaded_premium(vals, wts, 0.005),
            "lmps": lmps}


def s_risk_monte_carlo():
    pt = _pt(RATINGS_L4)
    dists = {**risk.load_dists(), **risk.cost_dists()}
    scen = risk.monte_carlo(dists, n=24, rng=np.random.default_rng(0))
    vals, wts = risk.propagate(pt, scen, risk.path_congestion("3", "9"))
    return {"mean": risk.emean(vals, wts), "var": risk.evar(vals, wts),
            "q05": risk.equantile(vals, wts, 0.05),
            "ce0": risk.certainty_equivalent(vals, wts, 0.0),
            "ce2": risk.certainty_equivalent(vals, wts, 0.02),
            "vals": vals}


def s_coalition_pareto():
    pt = _pt(RATINGS_L4)
    fp = _fp2(pt)
    dists = {**risk.load_dists(), **risk.cost_dists()}
    scen = risk.monte_carlo(dists, n=16, rng=np.random.default_rng(0))
    deltas, detail = ra.coalition_pareto_draws(fp, pt, scen, "BA-1",
                                               shed_price=SHED)
    led = ra.pareto_distribution_ledger(deltas, scen.weight)
    return {"ledger": led, "verdicts": ra.pareto_verdicts(led),
            "detail": detail}


def s_split_5_6():
    pt = _pt(RATINGS_L4, split_5_6=True)
    eng = wm.make_engine("UNIFIED", buses=pt.buses)
    res = se.solve_engine_dispatch(pt, eng, shed_price=SHED)
    return {"res": res, "lines": pt.lines, "s_nom": pt.s_nom,
            "sf": wm.sf_table(pt)}


def s_fleet_backstop():
    return {"bs": wm.fleet_with_backstop(),
            "bs_90": wm.fleet_with_backstop(price=90.0),
            "cost_by_bus": ra.cost_by_bus()}


def s_transport_network():
    n = wm.build_transport_network(["2", "3"])
    pt = se.compute_ptdf(n, slack_bus="2")
    return {"lines": pt.lines, "buses": pt.buses, "s_nom": pt.s_nom}


# 222a, the scheduling limit against the line limit (2026-11-06): the notebook's
# EDIT-cell case on the base ratings, the scheduling point at bus 9 on line_7.
SL_SP = "9"          # the scheduling point, BA-2's end of the 8-9 tie
SL_TIE = "line_7"    # the tie carrying it
SL_Q = 100.0         # the intertie scheduling limit, MW
SL_TIE_SIDE = 40.0   # a tie rating at which the tie is the tighter limit, MW
SL_ISL_SIDE = 60.0   # a tie rating at which the scheduling limit is the tighter, MW


def s_scheduling_limit():
    """222a's key numbers: BA-2's willingness to pay as merged blocks of its
    own avoided cost, BA-1's market under the scheduling limit and under the
    flow emulation, the one engine with the tie at the combined-actual flow
    and with the seam rated, and both limits declared on either side of the
    crossing."""
    pt = _pt()
    # 201's assignment, as 222a carries it: line_2 to BA-1, line_7 to BA-2
    fp = _fp2(pt)
    e2 = wm.make_engine("BA-2", BA_DEFS["BA-2"], activated=BA_LINES["BA-2"])

    def c2(x):
        r = se.solve_engine_dispatch(pt, e2, exo={SL_SP: x} if x > 0 else None,
                                     shed_price=SHED)
        return r.total_cost + SHED * sum(r.shed_by_bus.values()), r

    d2 = sum(wm.DEFAULT_LOADS.get(b, 0.0) for b in BA_DEFS["BA-2"])
    # the import levels at C_2's breakpoints, as 222a places them: where the residual
    # load d2 - x crosses a cumulative capacity of BA-2's units, cheapest first
    caps = np.cumsum([float(s["p_nom"]) for s in sorted(
        (s for s in wm.DEFAULT_GEN_FLEET.values() if str(s["bus"]) in BA_DEFS["BA-2"]),
        key=lambda s: s["cost"])])
    levels = sorted({0.0, d2, *(float(d2 - c) for c in caps if 0.0 < d2 - c < d2)})
    obj = [c2(x)[0] for x in levels]
    blocks: list = []
    for k in range(1, len(levels)):
        p = (obj[k - 1] - obj[k]) / (levels[k] - levels[k - 1])
        if blocks and abs(blocks[-1]["price"] - p) < 1e-6:
            blocks[-1]["mw"] += levels[k] - levels[k - 1]
        else:
            blocks.append({"price": p, "mw": levels[k] - levels[k - 1]})
    bids = [{"id": f"imp_{k}", "bus": SL_SP, "mw": float(b["mw"]),
             "price": float(b["price"])} for k, b in enumerate(blocks)]

    def ba1(q_bar=None, tie=None):
        ptx = _pt({} if tie is None else {SL_TIE: tie})
        act = BA_LINES["BA-1"] + ([SL_TIE] if tie is not None else [])
        eng = wm.make_engine("BA-1", BA_DEFS["BA-1"], activated=act)
        kw = ({} if q_bar is None else
              {"schedule_limits": {"isl": ([b["id"] for b in bids], q_bar)}})
        return se.solve_engine_transfers(
            ptx, eng, {"BA-1": BA_DEFS["BA-1"] + [SL_SP]}, {}, shed_price=SHED,
            demand_bids=bids, **kw)

    isl = ba1(q_bar=SL_Q)
    emul = ba1(tie=SL_Q)
    _, r2 = c2(sum(isl.demand_cleared.values()))
    w = ra.hub_weights(fp, "auto")
    intertie.configure(b_gap=w["BA-2"], dgap=w["BA-1"], all_loads=wm.DEFAULT_LOADS)
    _, act_flow = intertie.combined_actual(r2, isl, pt, wm.DEFAULT_LOADS)
    f_star = act_flow[SL_TIE]
    ifc = {("BA-1", "BA-2"): None}
    pt_j = _pt({SL_TIE: f_star})
    joint = se.solve_engine_transfers(pt_j, wm.make_engine("J", buses=pt_j.buses),
                                      BA_DEFS, ifc, shed_price=SHED)
    # Method 1 on the lines basis, each line's rent |mu^impl F| to its holder (the form
    # WECC 201 and 201a state). On the constraints basis the seam flowgate below is held
    # by neither area and splits half each; 222a prints that basis as a comparison only.
    alloc_j, summ_j, _, _ = ra.allocate_congestion_rent(
        fp, joint, pt_j, wm.DEFAULT_LOADS, te_split=0.5, rent_basis="lines")
    signs = {m: (1.0 if pt.line_buses[pt.line_idx[m]][0] in BA_DEFS["BA-1"] else -1.0)
             for m in fp.ties}
    fg = se.identity_flowgates(pt).with_flowgate(
        "seam", signs, fwd=SL_Q,
        rev=float(sum(pt.s_nom[pt.line_idx[m]] for m in fp.ties)), pt=pt)
    seam = se.solve_engine_transfers(pt, wm.make_engine("J", buses=pt.buses), BA_DEFS,
                                     ifc, shed_price=SHED, flowgates=fg)
    alloc_s, summ_s, _, _ = ra.allocate_congestion_rent(
        fp, seam, pt, wm.DEFAULT_LOADS, te_split=0.5, rent_basis="lines")
    tie_side = ba1(q_bar=SL_Q, tie=SL_TIE_SIDE)
    isl_side = ba1(q_bar=SL_Q, tie=SL_ISL_SIDE)
    return {"bids": bids,
            "isl": isl, "isl_awards": isl.demand_cleared,
            "isl_dual": isl.schedule_limit_dual, "isl_mw": isl.schedule_limit_mw,
            "residual": r2, "f_star": f_star,
            "emul": emul, "emul_awards": emul.demand_cleared,
            "joint": joint, "joint_alloc": alloc_j, "joint_summary": summ_j,
            "seam": seam, "seam_dual": seam.fg_dual, "seam_alloc": alloc_s,
            "seam_summary": summ_s,
            "tie_side": tie_side, "tie_side_awards": tie_side.demand_cleared,
            "tie_side_dual": tie_side.schedule_limit_dual,
            "isl_side": isl_side, "isl_side_awards": isl_side.demand_cleared,
            "isl_side_dual": isl_side.schedule_limit_dual}


SCENARIOS = {
    "base_unified": s_base_unified,
    "line4_unified": s_line4_unified,
    "line4_allocation": s_line4_allocation,
    "hub_decomposition": s_hub_decomposition,
    "transfers_mod29": s_transfers_mod29,
    "transfers_scalar": s_transfers_scalar,
    "three_ba": s_three_ba,
    "independent": s_independent,
    "atc": s_atc,
    "atc_interfaces": s_atc_interfaces,
    "rights_books": s_rights_books,
    "crr_auction": s_crr_auction,
    "qp": s_qp,
    "two_engine_seam": s_two_engine_seam,
    "exogenous": s_exogenous,
    "self_schedule": s_self_schedule,
    "path_hedge": s_path_hedge,
    "provision": s_provision,
    "wheeling": s_wheeling,
    "risk_gauss_hermite": s_risk_gauss_hermite,
    "risk_monte_carlo": s_risk_monte_carlo,
    "coalition_pareto": s_coalition_pareto,
    "split_5_6": s_split_5_6,
    "fleet_backstop": s_fleet_backstop,
    "transport_network": s_transport_network,
    "scheduling_limit": s_scheduling_limit,
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_all(only: str | None = None, verbose: bool = False) -> dict:
    out: dict = {}
    for name, fn in SCENARIOS.items():
        if only and only not in name:
            continue
        try:
            res = fn()
        except Exception as exc:                       # noqa: BLE001
            out[name] = {"__error__": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"  {name}: ERROR {type(exc).__name__}: {exc}")
            continue
        flat = canon(res, "")
        out[name] = flat
        if verbose:
            print(f"  {name}: {len(flat)} values")
    return out


def diff(a: dict, b: dict) -> list[str]:
    """Human-readable differences between two golden maps."""
    msgs = []
    for scen in sorted(set(a) | set(b)):
        if scen not in a:
            msgs.append(f"{scen}: MISSING from golden")
            continue
        if scen not in b:
            msgs.append(f"{scen}: NEW scenario (not in golden)")
            continue
        ka, kb = a[scen], b[scen]
        for key in sorted(set(ka) | set(kb)):
            va, vb = ka.get(key, "<absent>"), kb.get(key, "<absent>")
            if va != vb:
                msgs.append(f"{scen}.{key}: {vb!r} -> {va!r}")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="freeze the current numbers into golden.json")
    ap.add_argument("--local", action="store_true",
                    help="allow --write here; golden.json is copied with lib/")
    ap.add_argument("--only", default=None, help="substring filter on scenario name")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cur = run_all(args.only, args.verbose)
    n_err = sum(1 for v in cur.values() if "__error__" in v)
    n_val = sum(len(v) for v in cur.values())
    print(f"{len(cur)} scenarios, {n_val} values, {n_err} errored")
    for name, v in cur.items():
        if "__error__" in v:
            print(f"  ERROR {name}: {v['__error__']}")

    if args.write and not args.local:
        print("golden.json is copied with lib/ (lib/MANIFEST.md) and re-frozen "
              "by the Lab. --write --local overrides.")
        return 2

    if args.write:
        if n_err:
            print("refusing to freeze goldens while scenarios error")
            return 2
        # A golden that cannot reproduce itself is not a golden.
        again = run_all(args.only)
        if again != cur:
            print("NON-DETERMINISTIC: two runs disagree; refusing to write")
            for m in diff(again, cur)[:20]:
                print("   ", m)
            return 3
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {GOLDEN.relative_to(ROOT)} ({GOLDEN.stat().st_size/1024:.0f} KB)")
        return 0

    if not GOLDEN.exists():
        print(f"no golden at {GOLDEN} -- run with --write first")
        return 1
    ref = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.only:
        ref = {k: v for k, v in ref.items() if args.only in k}
    msgs = diff(cur, ref)
    if msgs:
        print(f"\n{len(msgs)} DIFFERENCES:")
        for m in msgs[:60]:
            print("   ", m)
        if len(msgs) > 60:
            print(f"    ... and {len(msgs)-60} more")
        return 1
    print("MATCH -- libraries reproduce the frozen numbers exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
