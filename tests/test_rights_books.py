"""Flowgate-aware booking in ``atc``: the generalization must be exact.

Seven claims, each on the 9-bus case with ``line_4`` rated 40 MW (the binding
corridor of 111 and 112).

1. ``book_sequentially`` over ``identity_flowgates`` equals the per-line book
   award by award, on exact float equality.
2. ``available_atc`` agrees between the two branches on every hexagon pair.
3. An asymmetric flowgate binds on its rated side only. The unrated direction
   carries the sentinel, and the sentinel never binds.
4. A signed two-line corridor books to its own aggregate rating divided by the
   aggregate shift factor.
5. The firm reading of existing commitments books strictly less than the netted
   reading on counterflowing requests, and the same on same-direction ones.
6. The ledger table reports both readings, and the AFC columns follow the basis
   asked for.
7. A response-factor threshold above zero screens constraints out of the
   booking, so the book overloads them. The behaviour is pinned because it is a
   stated deviation, not a defect.

The rights books and the common simultaneous feasibility test add eight more.

8. One authority, one constraint, one request along it: the Rated System Path
   grant is the constraint's own rating, the Available Flowgate Capability
   grant is the path's flow-scaled Total Transfer Capability, and no rule cuts
   either.
9. The three-authority triangle of 112 section 4: each book's rated path is
   first bound by the shared element, the union overloads it, and the
   reduction brings it to its rating.
10. The Available Flowgate Capability book at a zero threshold is firm feasible
    on its own set, and at a threshold of 0.05 the exceedance is bounded by the
    threshold times the book.
11. The netted reading entitles at least as much as the firm reading, the
    reduction conserves megawatts, and a feasible union is returned unchanged.
12. The Available Flowgate Capability book at a zero threshold on the netted
    reading is the sequential book.
13. The least-squares reduction is feasible, is at least as close to the
    nomination as the proportional rule, coincides with it on a symmetric case,
    and parts from it when the shift factors differ.
14. The unrated sentinel is read by equality: its row is unrated and never
    binds.
15. The loadings table reproduces ``line_loadings`` and ``firm_line_loadings``
    at their own rounding.
16. The common test reads a per-line monitored set as well as a flowgate set,
    and the two agree on the identity set.
17. An unrated direction is UNBOUNDED in the reduction. A nomination loading it
    above the sentinel's own value is returned untouched under every rule, and
    no award is limited by an unrated row.

The provider, the contract-path posting, the corrected reduction and the
settlement add six more.

18. A request left without a provider takes its holder, and the provider
    reaches the entitlement table and the per-provider table.
19. The Rated System Path book sells the binding element's posted rating one
    for one, and ``ttc="flow_scaled"`` reproduces the flow-scaled grant the
    book made before 2026-09-22.
20. The corrected proportional rule cuts the material contributor alone and
    removes exactly the excess, where the rule it replaces cuts both awards by
    one factor. Where no contributor is material it falls back to every
    contributor and names the fallback.
21. With every award material and one tier the corrected rule and the rule it
    replaces coincide on the triangle.
22. With two tiers on one gate the higher tier is cut first.
23. A matched book settles to the clearing's congestion rent, and the option
    form is non-negative and at least the obligation form on every row.

The reviewer's four findings of 2026-09-22 add four more.

24. At a zero threshold an award with no response on the overloaded row is not
    a material contributor and is not cut.
25. The Rated System Path ledger reports the loading the grant places on the
    element that rates it, and the contract-path grant under-sells that
    element by the request's own response factor.
26. An unrated direction is skipped in the rating, so the sentinel never
    reaches the posted capability the contract-path grant is taken from.
27. The limiting name is the constraint that first cut the award, and a later
    pass does not relabel it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import atc                # noqa: E402
import footprints as fpmod  # noqa: E402
import revenue_allocation as ra  # noqa: E402
import seams_engine as se  # noqa: E402
import wscc9_model as wm   # noqa: E402


# The hexagon reserved book of 111 section 6: each generator to its two
# nearest loads.
HEX = [("3", "7", 100), ("3", "5", 90),
       ("2", "7", 100), ("2", "9", 125),
       ("1", "9", 125), ("1", "5", 90)]


@pytest.fixture(scope="module")
def pt():
    return se.compute_ptdf(wm.build_network({"line_4": 40.0}), slack_bus="1")


@pytest.fixture(scope="module")
def ident(pt):
    return se.identity_flowgates(pt)


def _one_line_gate(pt, line, fwd, rev):
    """A flowgate over a single line with its own two ratings."""
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx[line]] = 1.0
    return se.FlowgateSet(ids=[line], S=S, fwd=np.array([float(fwd)]),
                          rev=np.array([float(rev)]),
                          members={line: [(line, 1.0)]})


def test_identity_flowgates_book_exactly_as_lines(pt, ident):
    per_line = atc.book_sequentially(pt, HEX)
    per_gate = atc.book_sequentially(pt, HEX, monitored=ident)
    assert len(per_line) == len(per_gate) == len(HEX)
    for a, b in zip(per_line, per_gate):
        assert (a.source, a.sink) == (b.source, b.sink)
        assert a.mw == b.mw, (a, b)


def test_available_atc_agrees_on_every_hexagon_pair(pt, ident):
    booked = atc.book_sequentially(pt, HEX)
    for s, k, _ in HEX:
        line = atc.available_atc(pt, s, k, booked)
        gate = atc.available_atc(pt, s, k, booked, monitored=ident)
        assert line == gate, (s, k, line, gate)


def test_the_unrated_direction_never_binds(pt):
    a = atc.path_shift_factors(pt, "3", "5")
    af = float(a[pt.line_idx["line_4"]])
    assert abs(af) > 0.1
    fwd_pair = ("3", "5") if af > 0 else ("5", "3")
    rev_pair = ("5", "3") if af > 0 else ("3", "5")

    fg = _one_line_gate(pt, "line_4", 30.0, 9999.0)
    ask = 500.0

    fwd_book = atc.book_sequentially(pt, [(*fwd_pair, ask)], monitored=fg)
    assert fwd_book[0].mw == pytest.approx(float(fg.fwd[0]) / abs(af), rel=1e-12)
    assert fwd_book[0].mw < ask

    # The reverse direction is unrated, so the sentinel leaves the request
    # bounded by nothing but its own size.
    rev_book = atc.book_sequentially(pt, [(*rev_pair, ask)], monitored=fg)
    assert rev_book[0].mw == ask
    mw, bid = atc.flowgate_available_atc(
        pt, atc.path_shift_factors(pt, *rev_pair), fg)
    assert bid == "line_4"
    assert mw == pytest.approx(float(fg.rev[0]) / abs(af), rel=1e-12)
    assert mw > ask


def test_a_signed_corridor_books_to_its_aggregate_rating(pt):
    S = np.zeros((1, pt.n_line))
    S[0, pt.line_idx["line_4"]] = 1.0
    S[0, pt.line_idx["line_5"]] = -1.0
    fg = se.FlowgateSet(ids=["corridor"], S=S, fwd=np.array([25.0]),
                        rev=np.array([9999.0]),
                        members={"corridor": [("line_4", 1.0), ("line_5", -1.0)]})

    # Bus 7 is the corridor's midpoint, so the signed aggregate reads the
    # injection at bus 7 and the path 3 to 7 impacts it at 1.0 per MW.
    a = atc.path_shift_factors(pt, "3", "7")
    af = float((S @ a)[0])
    assert af > 0
    book = atc.book_sequentially(pt, [("3", "7", 500.0)], monitored=fg)
    assert book[0].mw == pytest.approx(float(fg.fwd[0]) / abs(af), rel=1e-12)

    # The same construction with both members positive gives a non-unit
    # aggregate shift factor, and the grant divides by it.
    S2 = np.zeros((1, pt.n_line))
    S2[0, pt.line_idx["line_4"]] = 1.0
    S2[0, pt.line_idx["line_5"]] = 1.0
    fg2 = se.FlowgateSet(ids=["corridor_sum"], S=S2, fwd=np.array([25.0]),
                         rev=np.array([9999.0]),
                         members={"corridor_sum": [("line_4", 1.0),
                                                   ("line_5", 1.0)]})
    a2 = atc.path_shift_factors(pt, "3", "5")
    af2 = float((S2 @ a2)[0])
    assert af2 > 1e-3 and abs(af2 - 1.0) > 1e-3
    book2 = atc.book_sequentially(pt, [("3", "5", 500.0)], monitored=fg2)
    assert book2[0].mw == pytest.approx(float(fg2.fwd[0]) / abs(af2), rel=1e-12)


def test_firm_etc_books_no_more_than_netted(pt):
    counter = [("3", "5", 60.0), ("5", "3", 400.0)]
    netted = atc.book_sequentially(pt, counter)
    firm = atc.book_sequentially(pt, counter, etc_basis="firm")
    assert netted[0].mw == firm[0].mw
    assert firm[1].mw < netted[1].mw

    same = [("3", "5", 30.0), ("3", "5", 400.0)]
    netted_s = atc.book_sequentially(pt, same)
    firm_s = atc.book_sequentially(pt, same, etc_basis="firm")
    for a, b in zip(netted_s, firm_s):
        assert a.mw == pytest.approx(b.mw, rel=1e-12)


def test_the_ledger_table_reports_both_readings(pt, ident):
    booked = atc.book_sequentially(pt, HEX)
    led = atc._booked_ledger(pt, booked, ident)
    af, cap_f, cap_r, ids = atc._constraint_loading(pt, np.zeros(pt.n_line), ident)
    df = led.table(ids, cap_f, cap_r)
    assert list(df.index) == list(ids)
    f = atc.superposed_flow(pt, booked)
    for l in pt.lines:
        assert df.loc[l, "etc_net"] == pytest.approx(f[pt.line_idx[l]], abs=1e-9)
        assert df.loc[l, "etc_fwd"] >= 0.0
        assert df.loc[l, "etc_rev"] >= 0.0
        assert df.loc[l, "afc_fwd"] == pytest.approx(
            max(0.0, df.loc[l, "cap_fwd"] - df.loc[l, "etc_fwd"]), abs=1e-9)

    # The netted basis credits counterflow, so it never posts less capability
    # than the firm basis on the same ledger.
    dn = led.table(ids, cap_f, cap_r, etc_basis="netted")
    for l in pt.lines:
        assert dn.loc[l, "afc_fwd"] == pytest.approx(
            max(0.0, dn.loc[l, "cap_fwd"] - dn.loc[l, "etc_net"]), abs=1e-9)
        assert dn.loc[l, "afc_fwd"] >= df.loc[l, "afc_fwd"] - 1e-9
        assert dn.loc[l, "afc_rev"] >= df.loc[l, "afc_rev"] - 1e-9
    with pytest.raises(ValueError):
        led.table(ids, cap_f, cap_r, etc_basis="gross")


def test_a_response_factor_threshold_screens_constraints_out(pt):
    """The MOD-030 threshold posts capability the network cannot carry.

    A request from bus 3 to bus 5 impacts ``line_4`` at 0.25 per MW, so a
    threshold of 0.9 screens ``line_4`` out of the booking. The grant then
    exceeds the unscreened grant and the book fails the simultaneous
    feasibility test. This is the residual MOD-030 leaves between a
    per-constraint posting and the holistic test.
    """
    ask = [("3", "5", 5000.0)]
    full = atc.book_sequentially(pt, ask)
    screened = atc.book_sequentially(pt, ask, rf_min=0.9)
    assert screened[0].mw > full[0].mw

    ok_full, _ = atc.simultaneous_feasibility(pt, full)
    assert ok_full
    ok_screened, df = atc.simultaneous_feasibility(pt, screened)
    assert not ok_screened
    assert df.loc["line_4", "overload"]
    assert df.loc["line_4", "loading_%"] > 100.0

    # A request that impacts nothing at the threshold is granted in full,
    # because the model carries no limit other than the monitored set.
    unbounded = atc.book_sequentially(pt, [("3", "5", 1e9)], rf_min=1.5)
    assert unbounded[0].mw == 1e9


# ──────────────────────────────────────────────────────────────────────────
# The two books, the common test, and the reduction family
# ──────────────────────────────────────────────────────────────────────────
# The cycle partition of 112 section 4, built as its cell 11 builds it.
TRI_DEFS = {"A": ["1", "4", "5"], "B": ["3", "6", "7"], "C": ["2", "8", "9"]}
TRI_MANAGE = {"A": ["line_0", "line_1"], "B": ["line_3", "line_4"],
              "C": ["line_6", "line_7"]}
TRI_RATINGS = {"line_4": 40.0}
TRI_COLORS = {"A": "#E67E22", "B": "#27AE60", "C": "#2471A3"}

# One interface per authority, so every interface is held once. Each is
# oriented so its transfer loads line_4 forward. On the unloaded base case the
# rated capability is direction symmetric, so the orientation does not change
# any posted number, and it makes the firm union stack on one side of line_4.
TRI_HOLDS = {"A": ("B", "A"), "B": ("B", "C"), "C": ("A", "C")}


@pytest.fixture(scope="module")
def tri():
    """The triangle: ``(pt, fp, limits, sft, hub weights)``."""
    pt = se.compute_ptdf(wm.build_network(TRI_RATINGS), slack_bus="1")
    fp = fpmod.make(pt, TRI_DEFS, TRI_COLORS, manage=TRI_MANAGE)
    lim, sft = ra.mod29_interface_limits(fp, pt)
    return pt, fp, lim, sft, ra.hub_weights(fp, "auto")


def _tri_books(tri):
    """One Rated System Path book per authority, each holding its interface's
    request at that interface's posted capability."""
    pt, _fp, lim, _sft, w = tri
    books = []
    for holder, (src, snk) in TRI_HOLDS.items():
        key = tuple(sorted((src, snk)))
        req = atc.Request(w[src], w[snk], float(lim.loc[key, "TTC (MW)"]),
                          holder=holder, label=f"{src}->{snk}",
                          source_name=src, sink_name=snk)
        books.append(atc.rights_book_rsp(pt, [req], "all"))
    return books


def _entitled_awards(books, nom):
    """Every book's awards at the megawatts the nomination entitled them to."""
    out, i = [], 0
    for bk in books:
        for aw in bk.awards:
            out.append(atc._regrant(aw, float(nom.entitlements["entitled"].iloc[i])))
            i += 1
    return out


def test_one_authority_one_constraint_one_path(pt):
    """Claim 8, on the contract-path posting. The Rated System Path book sells
    the binding element's own rating one for one, the Available Flowgate
    Capability book sells the flow-scaled capability, and the flow-scaled
    Rated System Path knob reproduces the second."""
    gate = _one_line_gate(pt, "line_4", 40.0, 40.0)
    ttc_mw, binding = atc.ttc(pt, "3", "5", monitored=gate)
    cap = float(gate.fwd[0])
    assert binding == "line_4"
    assert ttc_mw > cap          # the path loads line_4 at less than 1 per MW

    req = atc.Request("3", "5", 500.0, holder="A", label="p1")
    rsp = atc.rights_book_rsp(pt, [req], gate)
    scaled = atc.rights_book_rsp(pt, [req], gate, ttc="flow_scaled")
    afc = atc.rights_book_afc(pt, [req], gate, rf_min=0.0)
    assert rsp.ledger["ttc_mw"].iloc[0] == pytest.approx(ttc_mw, rel=1e-12)
    assert rsp.ledger["cap_mw"].iloc[0] == pytest.approx(cap, rel=1e-12)
    assert rsp.granted_mw == pytest.approx(cap, rel=1e-12)
    assert scaled.granted_mw == pytest.approx(ttc_mw, rel=1e-12)
    assert afc.granted_mw == pytest.approx(ttc_mw, rel=1e-12)

    with pytest.raises(ValueError):
        atc.rights_book_rsp(pt, [req], gate, ttc="catalog")

    for rule in ("pro_rata", "pro_rata_global", "uniform", "least_squares"):
        nom = atc.nominate_and_test([rsp], pt, gate, reduction=rule)
        assert nom.entitlements["haircut_frac"].iloc[0] == pytest.approx(
            0.0, abs=1e-9)
        assert nom.iterations == 0
        assert nom.by_ba["entitled"].iloc[0] == pytest.approx(cap, rel=1e-9)


def test_the_triangle_books_reproduce_the_posted_capability(tri):
    """Claims 9 part one and 19. Each book's rated path is first bound by
    ``line_4``, and the contract-path grant is that element's own rating less
    its existing commitments, one megawatt of reservation per megawatt of
    posted path."""
    pt, _fp, lim, _sft, _w = tri
    rating = float(pt.s_nom[pt.line_idx["line_4"]])
    for book, (holder, (src, snk)) in zip(_tri_books(tri), TRI_HOLDS.items()):
        key = tuple(sorted((src, snk)))
        assert book.holder == holder
        assert book.provider == holder
        assert book.ledger["limiting"].iloc[0] == "line_4"
        # The flow-scaled figure is still reported beside the posted one.
        assert book.ledger["ttc_mw"].iloc[0] == pytest.approx(
            float(lim.loc[key, "TTC (MW)"]), abs=1e-9)
        assert book.ledger["cap_mw"].iloc[0] == pytest.approx(rating, abs=1e-9)
        assert book.ledger["etc_mw"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert book.granted_mw == pytest.approx(rating, abs=1e-9)


def test_the_flow_scaled_knob_reproduces_the_earlier_book(tri):
    """Claim 19, second half. ``ttc="flow_scaled"`` grants the interface's own
    independently rated Total Transfer Capability, which is what the book
    granted before 2026-09-22 and what the stored ledger carried."""
    pt, _fp, lim, _sft, w = tri
    for holder, (src, snk) in TRI_HOLDS.items():
        key = tuple(sorted((src, snk)))
        ttc_mw = float(lim.loc[key, "TTC (MW)"])
        req = atc.Request(w[src], w[snk], ttc_mw, holder=holder,
                          label=f"{src}->{snk}", source_name=src,
                          sink_name=snk)
        book = atc.rights_book_rsp(pt, [req], "all", ttc="flow_scaled")
        assert book.ledger["ttc_mw"].iloc[0] == pytest.approx(ttc_mw, abs=1e-9)
        assert book.ledger["atc_mw"].iloc[0] == pytest.approx(ttc_mw, abs=1e-9)
        assert book.granted_mw == pytest.approx(ttc_mw, abs=1e-9)


def test_the_triangle_union_overloads_and_reduces(tri):
    """Claim 9, part two. The firm union of the three contract-path books
    loads line_4 past its rating, and the proportional rule brings it to that
    rating with every authority cut by the same fraction, because all three
    nominate the same megawatts and all three are material contributors in one
    tier."""
    pt, _fp, _lim, sft, w = tri
    books = _tri_books(tri)
    ident = se.identity_flowgates(pt)

    # The union's own firm loading, built from the three grants and their hub
    # shift factors rather than taken from the reduction.
    l4 = pt.line_idx["line_4"]
    rating = float(pt.s_nom[l4])
    flow = sum(bk.granted_mw
               * abs(atc.hub_path_shift_factors(pt, w[src], w[snk])[l4])
               for bk, (src, snk) in zip(books, TRI_HOLDS.values()))

    nom = atc.nominate_and_test(books, pt, ident, direction="firm",
                                reduction="pro_rata")
    row = nom.loadings.loc["line_4"]
    assert row["loading_pre"] == pytest.approx(flow / rating, abs=1e-9)
    assert row["loading_pre"] > 1.0
    assert row["loading_post"] == pytest.approx(1.0, abs=1e-9)
    assert bool(row["binding"])
    assert nom.iterations == 1

    # The reduction is the same fraction for all three.
    keep = rating / flow
    for _i, r in nom.by_ba.iterrows():
        assert r["entitled"] / r["nominated"] == pytest.approx(keep, abs=1e-9)
        assert r["kept_frac"] == pytest.approx(keep, abs=1e-9)
        assert r["haircut_frac"] == pytest.approx(1 - keep, abs=1e-9)
        assert r["kept_frac"] + r["haircut_frac"] == pytest.approx(1.0,
                                                                  abs=1e-12)
    assert nom.by_ba["haircut_share"].sum() == pytest.approx(1.0, abs=1e-9)

    # The threefold over-assignment ``interface_sft`` measures belongs to the
    # flow-scaled book, which posts each interface's whole rated capability.
    scaled = [atc.rights_book_rsp(pt, bk.requests, "all", ttc="flow_scaled")
              for bk in books]
    nom_s = atc.nominate_and_test(scaled, pt, ident, direction="firm",
                                  reduction="pro_rata")
    assert nom_s.loadings.loc["line_4", "loading_pre"] == pytest.approx(
        float(sft.loc["line_4", "loading"]), abs=1e-9)
    assert nom_s.loadings.loc["line_4", "loading_pre"] == pytest.approx(
        3.0, abs=1e-9)
    for _i, r in nom_s.by_ba.iterrows():
        assert r["kept_frac"] == pytest.approx(1 / 3, abs=1e-9)


def test_afc_at_a_zero_threshold_is_firm_feasible(pt):
    """Claim 10. Seeded random queues, booked over the whole network."""
    own = se.identity_flowgates(pt)
    buses = list(pt.buses)
    rng = np.random.default_rng(20260918)
    for trial in range(8):
        queue = []
        for j in range(6):
            s, k = rng.choice(buses, size=2, replace=False)
            queue.append(atc.Request(str(s), str(k), float(rng.uniform(10, 200)),
                                     holder="A", label=f"t{trial}r{j}"))
        strict = atc.rights_book_afc(pt, queue, own, rf_min=0.0)
        ok, df = atc.firm_feasibility(pt, strict.awards, monitored=own)
        assert ok, df[df["overload"]].to_string()

        loose = atc.rights_book_afc(pt, queue, own, rf_min=0.05)
        ok2, df2 = atc.firm_feasibility(pt, loose.awards, monitored=own)
        if not ok2:
            bound = 0.05 * loose.granted_mw
            worst = (df2[["fwd", "rev"]].max(axis=1) - df2["limit"]).max()
            assert worst <= bound + 1e-6, (worst, bound)


def test_the_reduction_conserves_and_never_raises_an_award(tri):
    """Claim 11."""
    pt, _fp, _lim, _sft, _w = tri
    books = _tri_books(tri)
    ident = se.identity_flowgates(pt)

    firm = atc.nominate_and_test(books, pt, ident, direction="firm")
    netted = atc.nominate_and_test(books, pt, ident, direction="netted")
    f_by = firm.by_ba.set_index("holder")["entitled"]
    n_by = netted.by_ba.set_index("holder")["entitled"]
    for holder in f_by.index:
        assert n_by[holder] >= f_by[holder] - 1e-9

    for nom in (firm, netted):
        nominated = float(nom.entitlements["nominated"].sum())
        entitled = float(nom.entitlements["entitled"].sum())
        assert entitled <= nominated + 1e-9
        assert float(nom.by_ba["haircut_mw"].sum()) == pytest.approx(
            nominated - entitled, abs=1e-9)
        assert (nom.entitlements["entitled"]
                <= nom.entitlements["nominated"] + 1e-9).all()
        rated = nom.loadings[nom.loadings["rated"]]
        assert (rated["loading_post"] <= 1 + 1e-9).all()
        # Both directions of every constraint, not only the reported one.
        entitled_awards = _entitled_awards(books, nom)
        test = (atc.firm_feasibility if nom.direction == "firm"
                else atc.simultaneous_feasibility)
        ok, df = test(pt, entitled_awards, monitored=ident)
        assert ok, df[df["overload"]].to_string()

    # A union already within every rating is returned unchanged.
    small = [atc.rights_book_rsp(pt, [atc.Request("3", "5", 10.0, holder="A")],
                                 "all")]
    quiet = atc.nominate_and_test(small, pt, ident)
    assert quiet.iterations == 0
    assert quiet.entitlements["entitled"].iloc[0] == pytest.approx(10.0,
                                                                   abs=1e-12)


def test_afc_on_identity_flowgates_is_the_sequential_book(pt, ident):
    """Claim 12."""
    queue = [atc.Request(s, k, mw, holder="A") for s, k, mw in HEX]
    book = atc.rights_book_afc(pt, queue, ident, rf_min=0.0,
                               etc_basis="netted")
    plain = atc.book_sequentially(pt, HEX)
    assert len(book.awards) == len(plain)
    for a, b in zip(book.awards, plain):
        assert (a.source, a.sink) == (b.source, b.sink)
        assert a.mw == b.mw, (a, b)


def test_least_squares_against_pro_rata(pt):
    """Claim 13.

    The two rules coincide when every award loads the constraint at the same
    shift factor AND nominates the same megawatts, because the least-squares
    reduction spreads one equal absolute cut over the awards while the
    proportional rule spreads one equal fractional cut, and the two agree only
    on a symmetric nomination.
    """
    gate = _one_line_gate(pt, "line_4", 40.0, 40.0)

    def _book(holder, src, snk, mw):
        return atc.rights_book_rsp(
            pt, [atc.Request(src, snk, mw, holder=holder)], gate)

    # Symmetric: the same path posted twice, at the same megawatts.
    # The interior-point solver stops a few parts in a million inside the
    # constraint, so the comparison is made at that accuracy.
    sym = [_book("A", "3", "5", 120.0), _book("B", "3", "5", 120.0)]
    pr = atc.nominate_and_test(sym, pt, gate, reduction="pro_rata")
    ls = atc.nominate_and_test(sym, pt, gate, reduction="least_squares")
    assert (ls.entitlements["entitled"].to_numpy()
            == pytest.approx(pr.entitlements["entitled"].to_numpy(), abs=1e-3))

    # Unequal shift factors, same direction: the rules part. Path 3 to 5
    # loads line_4 at 0.25 per MW and path 3 to 7 at 0.85, so the proportional
    # rule cuts them in proportion to their megawatts while the least-squares
    # rule cuts them in proportion to their shift factors.
    asym = [_book("A", "3", "5", 120.0), _book("B", "3", "7", 120.0)]
    pr2 = atc.nominate_and_test(asym, pt, gate, reduction="pro_rata")
    ls2 = atc.nominate_and_test(asym, pt, gate, reduction="least_squares")
    xp = pr2.entitlements["entitled"].to_numpy()
    xl = ls2.entitlements["entitled"].to_numpy()
    assert np.max(np.abs(xp - xl)) > 1.0

    t = pr2.entitlements["nominated"].to_numpy()
    assert float(np.sum((t - xl) ** 2)) <= float(np.sum((t - xp) ** 2)) + 1e-6
    assert float(np.sum(xl)) <= float(np.sum(t)) + 1e-9
    assert ls2.loadings.loc["line_4", "loading_post"] <= 1 + 1e-9
    assert "line_4" in ls2.duals

    with pytest.raises(NotImplementedError):
        atc.nominate_and_test(asym, pt, gate, direction="firm",
                              reduction="least_squares")


def test_the_sentinel_row_is_unrated_and_never_binds(pt):
    """Claim 14. The sentinel is compared by equality."""
    a = atc.path_shift_factors(pt, "3", "5")
    af = float(a[pt.line_idx["line_4"]])
    rev_pair = ("5", "3") if af > 0 else ("3", "5")
    fg = _one_line_gate(pt, "line_4", 30.0, 9999.0)
    book = atc.rights_book_rsp(
        pt, [atc.Request(rev_pair[0], rev_pair[1], 400.0, holder="A")], fg)
    nom = atc.nominate_and_test([book], pt, fg, sentinel=9999.0)
    row = nom.loadings.loc["line_4"]
    assert row["cap"] == 9999.0
    assert not bool(row["rated"])
    assert not bool(row["binding"])
    assert np.isnan(row["loading_post"])
    assert nom.entitlements["entitled"].iloc[0] == pytest.approx(400.0,
                                                                 abs=1e-12)
    # Without a sentinel the same row is rated and reads as a fraction.
    plain = atc.nominate_and_test([book], pt, fg)
    assert bool(plain.loadings.loc["line_4", "rated"])


def test_an_unrated_direction_is_unbounded_in_the_reduction(pt, ident):
    """Claim 17. The sentinel marks a direction the catalog does not rate, so
    it must not act as a rating inside the reduction.

    The nomination here loads the unrated direction with 12,000 MW, above the
    sentinel's own value of 9,999. It is rated over an EMPTY constraint set,
    which grants it in full, because the rating layer is not what this claim
    is about. Under the sentinel every rule must return it untouched and name
    no limiting row. Without the sentinel the same row is a 9,999 MW rating
    and the proportional rule cuts to it, which is what makes the claim
    non-vacuous.
    """
    a = atc.path_shift_factors(pt, "3", "5")
    af = float(a[pt.line_idx["line_4"]])
    src, snk = ("5", "3") if af > 0 else ("3", "5")
    fg = _one_line_gate(pt, "line_4", 30.0, 9999.0)
    mw = 12000.0 / abs(af)
    book = atc.rights_book_rsp(
        pt, [atc.Request(src, snk, mw, holder="A")], ident.subset([]))
    assert book.granted_mw == pytest.approx(mw, rel=1e-12)

    for rule in ("pro_rata", "uniform", "least_squares"):
        nom = atc.nominate_and_test([book], pt, fg, sentinel=9999.0,
                                    reduction=rule)
        ent = nom.entitlements
        assert ent["entitled"].iloc[0] == pytest.approx(mw, rel=1e-12), rule
        assert all(v is None for v in ent["limiting"]), rule
        row = nom.loadings.loc["line_4"]
        assert abs(float(row["flow_post"])) > 9999.0, rule
        assert not bool(row["rated"]) and not bool(row["binding"]), rule

    cut = atc.nominate_and_test([book], pt, fg, reduction="pro_rata")
    assert cut.entitlements["entitled"].iloc[0] < mw - 1e-6
    assert cut.entitlements["limiting"].iloc[0] == "line_4"


def test_the_loadings_table_matches_the_line_tables(tri):
    """Claim 15. The nomination's own table is the same quantity the two
    loading tables report, unrounded."""
    pt, _fp, _lim, _sft, _w = tri
    books = _tri_books(tri)
    awards = [a for b in books for a in b.awards]
    ident = se.identity_flowgates(pt)

    nn = atc.nominate_and_test(books, pt, ident, direction="netted")
    ln = atc.line_loadings(pt, awards, monitored=ident)
    for l in pt.lines:
        assert nn.loadings.loc[l, "flow_pre"] == pytest.approx(
            float(ln.loc[l, "flow"]), abs=0.05)

    nf = atc.nominate_and_test(books, pt, ident, direction="firm")
    lf = atc.firm_line_loadings(pt, awards, monitored=ident)
    for l in pt.lines:
        assert abs(nf.loadings.loc[l, "flow_pre"]) == pytest.approx(
            float(lf.loc[l, "firm"]), abs=0.05)


def test_flowgate_subset_is_row_slicing(pt, ident):
    """``FlowgateSet.subset`` selects rows and changes nothing else."""
    sub = ident.subset(["line_4", "line_2"])
    assert sub.ids == ["line_4", "line_2"]
    assert sub.S.shape == (2, pt.n_line)
    for i, k in enumerate(sub.ids):
        j = ident.fg_idx[k]
        assert np.array_equal(sub.S[i], ident.S[j])
        assert sub.fwd[i] == ident.fwd[j]
        assert sub.rev[i] == ident.rev[j]
        assert sub.members[k] == ident.members[k]
    assert sub._H is None


def test_the_common_test_reads_a_per_line_monitored_set(tri):
    """Claim 16. ``nominate_and_test`` takes the same monitored argument the
    rest of the module takes, so a per-line set and the identity flowgate set
    return the same entitlements and the same loadings."""
    pt, _fp, _lim, _sft, _w = tri
    books = _tri_books(tri)
    ident = se.identity_flowgates(pt)
    for direction in ("firm", "netted"):
        per_line = atc.nominate_and_test(books, pt, "all", direction=direction)
        per_gate = atc.nominate_and_test(books, pt, ident, direction=direction)
        assert list(per_line.loadings.index) == list(per_gate.loadings.index)
        assert np.array_equal(per_line.entitlements["entitled"].to_numpy(),
                              per_gate.entitlements["entitled"].to_numpy())
        for col in ("flow_pre", "flow_post", "loading_pre", "loading_post"):
            assert np.array_equal(per_line.loadings[col].to_numpy(),
                                  per_gate.loadings[col].to_numpy())


# ──────────────────────────────────────────────────────────────────────────
# The provider, the corrected reduction, and the settlement
# ──────────────────────────────────────────────────────────────────────────
def test_a_request_without_a_provider_takes_its_holder(pt):
    """Claim 18. The provider defaults to the holder, which is the
    self-provided case, and it reaches both tables."""
    own = atc.Request("3", "5", 30.0, holder="A", label="own")
    assert own.provider == "A"
    assert own.tier == 1

    sold = atc.Request("3", "7", 20.0, holder="LSE", provider="TSP",
                       label="sold", tier=2)
    assert sold.provider == "TSP"
    assert sold.tier == 2
    with pytest.raises(ValueError):
        atc.Request("3", "5", 10.0, holder="A", tier=4)

    ident = se.identity_flowgates(pt)
    book = atc.rights_book_rsp(pt, [own, sold], ident.subset([]))
    assert book.holder == ""          # two holders
    assert book.provider == ""        # two providers
    nom = atc.nominate_and_test([book], pt, ident.subset(["line_4"]))
    ent = nom.entitlements.set_index("label")
    assert ent.loc["own", "provider"] == "A"
    assert ent.loc["sold", "provider"] == "TSP"
    assert int(ent.loc["sold", "tier"]) == 2
    assert set(nom.by_provider["provider"]) == {"A", "TSP"}
    assert set(nom.by_ba["holder"]) == {"A", "LSE"}
    assert nom.by_provider["nominated"].sum() == pytest.approx(
        nom.by_ba["nominated"].sum(), abs=1e-12)
    assert len(nom.awards) == len(nom.entitlements)


def test_the_corrected_rule_cuts_the_material_contributor_alone():
    """Claim 20. One gate, two awards, ten megawatts of excess.

    Award 1 loads the gate at 0.5 per megawatt and award 2 at 0.02, which is
    below the 0.05 response-factor threshold. The corrected rule removes the
    excess from award 1 alone, which takes exactly 10 / 0.5 megawatts. The
    rule it replaces scales both by the gate's own scale-back factor, so it
    cuts an award that puts a fiftieth of a megawatt per megawatt on the
    constrained element.
    """
    A = np.array([[0.5, 0.02]])
    cap = np.array([100.0])
    t = np.array([200.0, 500.0])
    loading = float((A @ t)[0])
    excess = loading - float(cap[0])
    assert excess == pytest.approx(10.0, abs=1e-12)

    x, iters, limiting = atc._reduce_pro_rata(
        A, t, cap, cap, ["g"], "firm", 5, 1e-9)
    assert iters == 1
    assert x[0] == pytest.approx(t[0] - excess / A[0, 0], abs=1e-9)
    assert x[1] == pytest.approx(t[1], abs=1e-12)
    assert float((A @ x)[0]) == pytest.approx(float(cap[0]), abs=1e-9)
    assert limiting == ["g", None]

    xg, _itg, limg = atc._reduce_pro_rata_global(
        A, t, cap, cap, ["g"], "firm", 5, 1e-9)
    theta = float(cap[0]) / loading
    assert xg[0] == pytest.approx(t[0] * theta, abs=1e-9)
    assert xg[1] == pytest.approx(t[1] * theta, abs=1e-9)
    assert limg == ["g", "g"]
    # The rule it replaces takes more megawatts off the book for the same
    # relief on the same constraint.
    assert float(np.sum(t - xg)) > float(np.sum(t - x))


def test_the_corrected_rule_falls_back_where_nothing_is_material():
    """Claim 20, second half. Both awards sit below the threshold, so the
    fallback cuts every contributor and names itself in ``limiting``."""
    A = np.array([[0.01, 0.02]])
    cap = np.array([2.0])
    t = np.array([100.0, 100.0])
    x, _it, limiting = atc._reduce_pro_rata(
        A, t, cap, cap, ["g"], "firm", 5, 1e-9)
    assert float((A @ x)[0]) == pytest.approx(float(cap[0]), abs=1e-9)
    assert limiting == ["g (no material contributor)"] * 2
    # One factor, because the two awards are in one tier.
    assert x[0] / t[0] == pytest.approx(x[1] / t[1], abs=1e-12)


def test_the_corrected_rule_matches_the_rule_it_replaces_on_the_triangle(tri):
    """Claim 21. Every award is material on line_4 and all three are in one
    tier, so the corrected rule is the per-constraint proportional cut and the
    two rules coincide."""
    pt, _fp, _lim, _sft, _w = tri
    books = _tri_books(tri)
    ident = se.identity_flowgates(pt)
    new = atc.nominate_and_test(books, pt, ident, direction="firm",
                                reduction="pro_rata")
    old = atc.nominate_and_test(books, pt, ident, direction="firm",
                                reduction="pro_rata_global")
    assert (new.entitlements["entitled"].to_numpy()
            == pytest.approx(old.entitlements["entitled"].to_numpy(),
                             abs=1e-9))
    assert new.iterations == old.iterations


def test_the_higher_tier_is_cut_first(pt):
    """Claim 22. Two awards on one gate at the same shift factor, one non-firm
    and one long-term firm. The non-firm award carries the whole reduction."""
    gate = _one_line_gate(pt, "line_4", 40.0, 40.0)
    empty = se.identity_flowgates(pt).subset([])
    a4 = abs(float(atc.path_shift_factors(pt, "3", "5")[pt.line_idx["line_4"]]))
    firm = atc.Request("3", "5", 100.0, holder="F", label="firm", tier=1)
    nonfirm = atc.Request("3", "5", 100.0, holder="N", label="nonfirm",
                          tier=3)
    book = atc.rights_book_rsp(pt, [firm, nonfirm], empty)
    assert book.granted_mw == pytest.approx(200.0, abs=1e-9)

    nom = atc.nominate_and_test([book], pt, gate, direction="firm",
                                reduction="pro_rata")
    ent = nom.entitlements.set_index("label")
    excess = 200.0 * a4 - float(gate.fwd[0])
    assert excess > 0
    assert ent.loc["firm", "entitled"] == pytest.approx(100.0, abs=1e-9)
    assert ent.loc["nonfirm", "entitled"] == pytest.approx(
        100.0 - excess / a4, abs=1e-9)
    assert nom.loadings.loc["line_4", "loading_post"] == pytest.approx(
        1.0, abs=1e-9)

    # The rule it replaces reads no tier and cuts both by one factor.
    old = atc.nominate_and_test([book], pt, gate, direction="firm",
                                reduction="pro_rata_global")
    e_old = old.entitlements.set_index("label")
    assert e_old.loc["firm", "entitled"] == pytest.approx(
        e_old.loc["nonfirm", "entitled"], abs=1e-9)


def _matched_book(pt, res):
    """111's hexagon carrying the deliveries of a clearing.

    Each generator delivers to its two nearest loads, and the six volumes are
    the non-negative solution of the hexagon's own marginals: every generator
    ships what it generated and every load receives what it consumed. The
    volumes are solved for, never assumed, and the solution is checked against
    the marginals before it is used.
    """
    from scipy.optimize import nnls

    gens = sorted({s for s, _k, _mw in HEX})
    loads = sorted({k for _s, k, _mw in HEX})
    pairs = [(s, k) for s, k, _mw in HEX]
    M = np.zeros((len(gens) + len(loads), len(pairs)))
    for j, (s, k) in enumerate(pairs):
        M[gens.index(s), j] = 1.0
        M[len(gens) + loads.index(k), j] = 1.0
    b = np.array([float(res.gen_by_bus.get(g, 0.0)) for g in gens]
                 + [float(res.load_by_bus.get(l, 0.0)) for l in loads])
    x, _resid = nnls(M, b)
    assert np.max(np.abs(M @ x - b)) < 1e-9
    return [atc.Request(s, k, float(v), holder=f"LSE_{k}",
                        provider=f"TSP_{s}", label=f"{s}->{k}")
            for (s, k), v in zip(pairs, x)]


def test_the_matched_book_settles_to_the_congestion_rent(pt):
    """Claim 23. The hexagon carrying 201's canonical clearing settles to that
    clearing's own congestion rent, which is the revenue-adequacy identity."""
    res = se.solve_engine_dispatch(
        pt, wm.make_engine("UNIFIED", buses=pt.buses), shed_price=150.0)
    rent = -sum(float(res.line_dual.get(l, 0.0)) * float(res.flow_own[l])
                for l in pt.lines)
    assert rent > 0

    ident = se.identity_flowgates(pt)
    book = atc.rights_book_rsp(pt, _matched_book(pt, res), ident.subset([]))
    nom = atc.nominate_and_test([book], pt, ident, direction="netted")
    # The book is the dispatch, so the common test takes it unchanged.
    assert nom.iterations == 0
    assert float((nom.entitlements["nominated"]
                  - nom.entitlements["entitled"]).sum()) == pytest.approx(
        0.0, abs=1e-9)

    obl = atc.settle_rights(nom, res.lmp)
    assert float(obl["value"].sum()) == pytest.approx(rent, abs=1e-6)
    assert list(obl.columns) == ["holder", "provider", "label", "kind",
                                 "tier", "entitled", "dlambda", "value"]
    assert set(obl["provider"]) == {"TSP_1", "TSP_2", "TSP_3"}

    with pytest.warns(UserWarning):
        opt = atc.settle_rights(nom, res.lmp, form="option")
    assert (opt["value"].to_numpy() >= -1e-12).all()
    # An option is exercised only in the money, so it pays at least what the
    # obligation pays on every row and more over the book.
    assert (opt["value"].to_numpy()
            >= obl["value"].to_numpy() - 1e-12).all()
    assert float(opt["value"].sum()) > float(obl["value"].sum())

    with pytest.raises(ValueError):
        atc.settle_rights(nom, res.lmp, form="forward")


def test_a_zero_response_award_is_never_a_material_contributor():
    """Claim 24. At ``rf_min = 0.0`` the materiality screen still floors at
    ``tol``, so an award with no response on the overloaded row is not a
    contributor and is not cut. The row is named as the limiting constraint
    for the award that does load it, and for that award alone."""
    A = np.array([[0.5, 0.0]])
    cap = np.array([100.0])
    t = np.array([300.0, 500.0])
    excess = float((A @ t)[0]) - float(cap[0])
    assert excess == pytest.approx(50.0, abs=1e-12)

    x, iters, limiting = atc._reduce_pro_rata(
        A, t, cap, cap, ["g"], "firm", 5, 1e-9, rf_min=0.0)
    assert iters == 1
    assert x[0] == pytest.approx(t[0] - excess / A[0, 0], abs=1e-9)
    assert x[1] == pytest.approx(t[1], abs=1e-12)
    assert float((A @ x)[0]) == pytest.approx(float(cap[0]), abs=1e-9)
    assert limiting == ["g", None]


def test_the_ledger_reports_the_loading_the_grant_places(tri):
    """Claim 25. ``loading_per_mw`` is the request's own response factor on
    the element that rates its path, so the megawatts of flow the grant places
    on that element are the grant times that column. The three grants are 40
    MW each against a 40 MW element and place less than 40 MW on it, which is
    the under-sale the contract-path posting carries."""
    pt, _fp, _lim, _sft, _w = tri
    rating = float(pt.s_nom[pt.line_idx["line_4"]])
    for book in _tri_books(tri):
        row = book.ledger.iloc[0]
        assert row["limiting"] == "line_4"
        loading = float(row["loading_per_mw"])
        assert 0.0 < loading < 1.0
        placed = float(row["granted_mw"]) * loading
        # ``line_loadings`` rounds its flow column to a tenth of a megawatt.
        table = atc.line_loadings(pt, book.awards)
        assert placed == pytest.approx(
            abs(float(table.loc["line_4", "flow"])), abs=0.05)
        assert placed < rating
        # The flow-scaled figure is the same rating read through that loading.
        assert float(row["ttc_mw"]) * loading == pytest.approx(
            float(row["cap_mw"]), rel=1e-9)


def test_an_unrated_direction_is_never_posted_as_capability(pt):
    """Claim 26. ``_ttc_over`` reads the sentinel by equality and skips the
    direction carrying it, so the unrated direction never sets the transfer
    limit and never reaches ``cap``. Under the contract-path posting that
    matters twice over, because the grant is the binding row's own rating."""
    S = np.zeros((2, pt.n_line))
    S[0, pt.line_idx["line_2"]] = 1.0
    S[1, pt.line_idx["line_4"]] = 1.0
    sentinel = 9999.0
    gate = se.FlowgateSet(
        ids=["line_2", "line_4"], S=S,
        fwd=np.array([150.0, 20000.0]),
        rev=np.array([sentinel, 20000.0]),
        members={"line_2": [("line_2", 1.0)], "line_4": [("line_4", 1.0)]})
    a = atc.path_shift_factors(pt, "3", "5")
    assert a[pt.line_idx["line_2"]] < 0      # the path loads line_2 in reverse

    blind_mw, blind_id, blind_cap = atc._ttc_over(pt, a, gate, 1e-6)
    assert blind_id == "line_2"
    assert blind_cap == sentinel

    mw, which, cap = atc._ttc_over(pt, a, gate, 1e-6, sentinel=sentinel)
    assert which == "line_4"
    assert cap == 20000.0
    assert cap != sentinel
    assert mw > blind_mw

    req = atc.Request("3", "5", 15000.0, holder="A", label="p1")
    blind = atc.rights_book_rsp(pt, [req], gate)
    book = atc.rights_book_rsp(pt, [req], gate, sentinel=sentinel)
    assert blind.granted_mw == pytest.approx(sentinel, rel=1e-12)
    assert book.granted_mw == pytest.approx(15000.0, rel=1e-12)
    assert list(book.ledger["limiting"]) == ["line_4"]

    # The same convention reaches the Available Flowgate Capability book,
    # which charges impacts rather than transfer megawatts.
    afc = atc.rights_book_afc(pt, [req], gate, rf_min=0.0, sentinel=sentinel)
    blind_afc = atc.rights_book_afc(pt, [req], gate, rf_min=0.0)
    assert afc.granted_mw > blind_afc.granted_mw


def test_the_limiting_name_is_the_constraint_that_first_cut_the_award():
    """Claim 27. An award cut on the first pass as a material contributor
    keeps that pass's constraint name when a second pass cuts it again, so a
    material cut is never relabelled as a fallback or as another row."""
    A = np.array([[1.0, 0.0], [1.0, 1.0]])
    cap = np.array([50.0, 130.0])
    t = np.array([100.0, 100.0])

    x, iters, limiting = atc._reduce_pro_rata(
        A, t, cap, cap, ["g0", "g1"], "firm", 5, 1e-9)
    assert iters == 2                      # two cuts, then a feasible test
    assert limiting == ["g0", "g1"]
    assert x[0] == pytest.approx(50.0 * (1.0 - 20.0 / 150.0), abs=1e-9)
    assert x[1] == pytest.approx(100.0 * (1.0 - 20.0 / 150.0), abs=1e-9)
    flows = A @ x
    assert flows[0] <= cap[0] + 1e-9
    assert flows[1] == pytest.approx(cap[1], abs=1e-9)
