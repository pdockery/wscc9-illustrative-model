# NOTEBOOK_RECORD.md: the record of the public sandbox

This file holds the record of this repository: what each root notebook carries and which deck it follows, an index of the library's public functions, the publishing steps, and the canon's dated deviations. `AGENTS.md` carries the rules that hold in every session, and `CANON.md` the annotated canon. New entries are added here, dated.

## Contents

1. The notebooks
2. The library
3. Publishing
4. The canon's dated deviations

---

## 1. The notebooks

One entry per root notebook, dated. Each entry carries the deck (date, title, `references.bib` key, the excerpt file under `sources/workshops/` and its page range in the posted deck), the slide map (deck page against the notebook's section and cell, in the deck's order, with the excerpt file its page numbers index), the anchors the notebook reproduces from the deck, the stamp, the builder and the checks run. The sandbox entry lists each configurable feature the sandbox exposes and its knob, and the features its closing section says it does not carry.

### `workshop2_congestion_management.ipynb`

Entry dated 2026-11-07, revised 2026-11-08: the slide checks moved to `tests/test_slides_workshop2.py`, and the tables became Market A's production-cost ledgers (user direction 2026-11-08).

**Deck.** *Seams Workshop: Congestion Management Examples*, the joint CAISO and SPP market seams workshop of August 18, 2026 (`caiso_spp_workshop_aug2026`). The excerpt is `sources/workshops/2026-08-18_congestion_management_examples.pdf`, pages 91 to 109 of the posted 113-page deck, and a slide number in the notebook is the excerpt's page, so slide $n$ is posted page $90+n$.

**What it carries.** Two market engines clear independently on the 9-bus network, and line 6-7 (`line_4`, rated 40 MW) binds inside Market A.
1. The formulation has 7 numbered equations: the market program with its flow input, the flow contribution, the superposed flow, the non-implication that each market within its limit leaves the line within its rating, the nodal price, the congestion rent, and the cost of an input.
2. The slide map, in the deck's order: slides 3 and 4 (the footprints and the case as configured); Example 1, no forecast of the neighbor's flow, slides 6 to 11 (Market A clears, the bus-7 price and the line-6-7 dual each as a production-cost change, Market B clears, the superposed flow, Market A's redispatch to Market B's realized flow); Example 2, an over-forecast of the neighbor's flow, slides 13 to 17 (Market A's input, each market's clear on its forecast, the superposed flow, the cost of phantom congestion); slide 18, the parallel process, in prose; slide 19, which prints no number.
3. The anchor is the Market A clear of Example 1: line 6-7 binds at 40 MW with $\lvert\mu\rvert=\$20.10$/MWh and a congestion rent of \$804.1/h. The slide 6 cell prints the settlement by bus and the rent by line, each totalling the rent.
4. The tables are Market A's production cost before and after accommodation, built by the notebook-local helpers `cost_table`, `accommodation_ledger` and `phantom_ledger`. After slide 11, `ACC_TABLE` and `ACC_LEDGER` set Market A's clear with no input, \$2,534.65/h with a superposed flow of 49.446 MW on line 6-7, beside its clear on Market B's realized flow, \$2,724.54/h at 40.000 MW, and close the rise of \$189.89/h by (7) as $\lvert\mu\rvert$ times Market B's 9.446 MW, with the unit-by-unit redispatch and a zero residual. At slide 17, `PHANTOM_TABLE` adds the clear on the forecast, \$2,766.69/h, and `PHANTOM_LEDGER` closes the cost of phantom congestion, \$42.15/h, as $\lvert\mu\rvert$ times the 2.097 MW over-forecast, which equals the capacity of line 6-7 that the superposed flow of 37.903 MW leaves unused. Example 2's opening defines phantom congestion as congestion that Market A prices and relieves on line 6-7 and that the superposed flow does not produce, the phrase slide 12 uses.
5. The EDIT cell has 14 knobs: `MARKETS`, `MARKET_COLORS`, `MONITORED_LINES`, `ACTIVATED_LINES`, `LINE_RATINGS`, `GEN_FLEET`, `CONFIGURED_LOADS`, the four drawn loads `LOAD_7`, `LOAD_9`, `LOAD_9_FORECAST_BY_A` and `LOAD_7_FORECAST_BY_B`, `SHED_PRICE`, `SLACK_BUS` and `STUDY_LINE`.
6. The closing Sandbox section reruns the chain through `run_chain()` on the EDIT cell's values, and takes the bus-7 load, the bus-9 load and Market A's forecast as arguments. It prints the bottom lines of the two ledgers, the cost of accommodation and the cost of phantom congestion, each against (7) with the superposed flow against the rating.

**Slide checks.** `tests/test_slides_workshop2.py` executes the notebook through `tests/slide_check.py` and holds 13 tests, one per slide, with 228 rows: slide 3, 3; slide 4, 18; slide 6, 19; slide 7, 6; slide 8, 5; slide 9, 19; slide 10, 15; slide 11, 31; slide 13, 17; slide 14, 30; slide 15, 23; slide 16, 15; slide 17, 27. Beside the rows the tests assert the anchor, the dual as the interior pair's offer gap over their shift-factor gap, both identities of (7), Market B's unchanged clear on slide 15, the common $\lvert\mu\rvert$ across Market A's clears, and the closing of both ledgers. The one exception is slide 14's redispatch: the slide prints 15.45 MW, the run gives 15.47 MW, the number the slide 14 cell states, and the test reads it from that cell. The slide's dollar figures hold at either value. The notebook has 14 figures.

**Stamp.** 47 cells, executed with `tools/nb_execute.py` at 2026-11-08T17:15:44Z. The stamp's `lib_commit` is `ce66306` with `lib_dirty` true, because `lib/` was copied in on 2026-11-07 after that commit, and its `lib_manifest.sha256` is `cec8c27c5df3f385824463c14c9804ce745299b0b3371c1afd2ae5180ade5e03`, the SHA-256 of `lib/MANIFEST.md`, which lists the 14 modules' hashes. Re-executed with `tools/nb_execute.py` at 2026-11-11T00:58:10Z, its stamp's `lib_commit` `eb48135` with `lib_dirty` false and its `lib_manifest.sha256` `520d446bc3f9e3ab3641f710b0e97969d8bfa769a4225093d587a6392c1a308d`. Every text output and figure count is as before.

### `workshop3_transactions_across_boundaries.ipynb`

Entry dated 2026-11-07, revised 2026-11-08: the slide checks moved to `tests/test_slides_workshop3.py` (user direction 2026-11-08).

**Deck.** *Transactions across market boundaries*, the joint CAISO and SPP market seams workshop of September 16, 2026 (`caiso_spp_workshop_sep2026`). The excerpt is `sources/workshops/2026-09-16_transactions_across_market_boundaries.pdf`, pages 1 to 34 of the 34-page deck. It is link-only, so the file is untracked and `manifest.json` gives the posting. A slide number in the notebook is the deck's page.

**What it carries.** Two market engines on the 9-bus network, and four examples of a transaction between them.
1. The formulation has 12 numbered equations: the clearing with a fixed exogenous injection, the own flow, the nodal price, the congestion rent, the redispatch behind a price and the price as its cost, the self-schedule legs, the scheduler's settlement, the award of a price-sensitive bid, its settlement, the rising offer curve, and the settlement with an open position at the real-time price. It adds the transaction legs (`tx_legs`), the bisection of an award, and the scheduler's net settlement $\Pi$.
2. The slide map: slides 3 and 4 (the case); Example 1, each market prices every node, slides 5 to 11; Example 2, a balanced self-schedule of 25 MW from bus 1 to bus 7, slides 12 to 18; Example 3, a price-sensitive bid of 30 MW at \$40 from bus 3 to bus 5 under flat offers, slides 19 to 25; Example 4, the same bid at 120 MW under rising offer curves, slides 26 to 33; slide 34, the key points, which prints no number.
3. The anchor is the canonical clearing of one engine over all nine buses: line 6-7 binds at a signed dual of $-\$20.10$/MWh with a rent of \$804.1/h. Market A's clear in Example 1 reproduces it, and the cell prints Market A's rent and dual against the canonical ones with their residuals.
4. The EDIT cell has 21 knobs: `MKT_DEFS`, `MARKET_COLORS`, `MONITORED_LINES`, `ACTIVATED_LINES`, `P_NOM_B1`, `LINE0_RATING`, `LINE4_RATING`, `LOADS`, `SHED_PRICE`, the price-walk buses `WALK_A`, `WALK_B` and `WALK_DIRECTION`, Example 2's `SOURCE_BUS`, `SINK_BUS` and `S_SCHED`, and `TX_BID`, `BID_3`, `QTY_3`, `BID_4`, `QTY_4` and `CURVES_TX` for Examples 3 and 4. The deck does not print `SHED_PRICE` or `CURVES_TX`.
5. The closing Sandbox section runs `build_case()` and `run_chain()` on the EDIT cell's values, draws no figure and checks no slide. A clearing that sheds load stops the chain with an `AssertionError` that names the market.

**Slide checks.** `tests/test_slides_workshop3.py` executes the notebook through `tests/slide_check.py` and holds 21 tests: the canonical anchor and one per slide for slides 3, 4, 6, 7, 9, 10, 11, 13 to 17, 21, 23, 24, 28 and 30 to 33. They carry the 95 checks the notebook asserted before 2026-11-08: slide 4, nine ratings and three offers; slide 6, 3 injections, 8 flows and 7 prices; slide 10, 3 injections, 7 flows and 9 prices; slides 13 and 15, 7 pair flows; slide 14, 4 injections, 9 flows and 7 prices; slide 16, 4 injections, 7 flows and 9 prices; slide 21, 4 injections, 8 flows and 7 prices; slide 23, 4 injections, 8 flows and 9 prices; slide 28, 4 injections, 8 flows and 7 prices; slide 30, 4 injections, 8 flows and 9 prices; and the prices, amounts and positions of slides 7, 9, 11, 17, 18, 24, 25 and 31 to 33. Where the notebook asserted the run's own unrounded figures, the test checks each clearing's identities instead (the power balance, the price identity at every bus, the objective at the dispatch, each interior unit's offer at its bus price, the settlements recomputed from the prices, and $\Pi+(\Delta C^A+\Delta C^B)=0$), and reads the figures the markdown states from the markdown, because a run-produced number is not typed into a test (`CODE_STYLE.md` 8.1). The named exceptions are slides where the deck rounds before it multiplies, and the test rebuilds the slide's number at its rounding. Slide 4's ring draws the stock loads of 125 MW at bus 9 and 100 MW at bus 7, and every clear serves 173 MW and 90 MW. Slide 7 prints \$39.90, \$2.80 and \$37.10 where the run gives \$39.96, \$2.83 and \$37.13. Slide 24 prints \$750 where the run gives \$750.59. Slides 31 and 32 print -\$1,600 and -\$2,916.90 where the run gives -\$1,591.45 and -\$2,928.15, and the net is -\$396.90 on the slide's rounding and -\$399.60 in the run. Slide 32's title reads Example #3 and its content is Example 4. The notebook has 17 figures.

**Stamp.** 70 cells, executed with `tools/nb_execute.py` at 2026-11-08T16:52:59Z. The stamp's `lib_commit` is `ce66306` with `lib_dirty` true, because `lib/` was copied in on 2026-11-07 after that commit, and its `lib_manifest.sha256` is `cec8c27c5df3f385824463c14c9804ce745299b0b3371c1afd2ae5180ade5e03`, the SHA-256 of `lib/MANIFEST.md`, which lists the 14 modules' hashes. Re-executed with `tools/nb_execute.py` at 2026-11-11T01:01:04Z, its stamp's `lib_commit` `eb48135` with `lib_dirty` false and its `lib_manifest.sha256` `520d446bc3f9e3ab3641f710b0e97969d8bfa769a4225093d587a6392c1a308d`. Every text output and figure count is as before, except one pandas Styler object's memory address.

### `workshop4_loading_relief.ipynb`

Entry dated 2026-11-07, revised 2026-11-08: the slide checks moved to `tests/test_slides_workshop4.py` (user direction 2026-11-08).

**Deck.** *Seams Workshop: Western Loading Relief process examples*, the joint CAISO and SPP market seams workshop of October 22, 2026 (`caiso_spp_workshop_oct2026`). The excerpt is `sources/workshops/2026-10-22_loading_relief_examples.pdf`, pages 52 to 86 of the posted 95-page deck, and a slide number in the notebook is the posted deck's page, printed in its footer. The procedure is the Western Interconnection Loading Relief business practice standards as the NAESB recommendation summarizes them (`naesb_weq_ecrec2025`).

**What it carries.** Two market engines clear independently, their superposed flow overloads line 6-7, and the Western Loading Relief process assigns the relief.
1. The formulation has 16 numbered equations. Equations (1) to (6) are the market program, the flow contribution and its superposition, the nodal price, the production-cost walk, the rent and the cost of an input. Equations (7) to (16) add the e-tag objects (`etag.ETag`): a tag's transfer distribution factor and impact, the sum of tag impacts to the flow, the relief requirement, the eligibility threshold, the assignment by priority level, the curtailment, the redispatch to a backstop unit, the generation-to-load impact and its allocation over untagged generators, and the market flow limit.
2. The slide map: slides 54 and 55 (the footprints and the case, with one backstop unit of \$55/MWh and 80 MW at each load bus); Example 1, no forecast, slides 56 to 63; Example 2, everything tagged, slides 64 to 71; Example 3, tagged schedules and generation-to-load, slides 72 to 77; Example 4, relief through each market's flow limit, slides 78 to 84; slides 85 and 86 as a note, slide 86 printing no number.
3. The anchor is Example 1's Market A clear, the first example of the August workshop on a fleet with the backstops added: line 6-7 binds at 40 MW with $\lvert\mu\rvert=\$20.10$/MWh. The blind superposed flow is 49.446 MW and the relief requirement 9.446 MW. The Example 2 and Example 3 cells print the agreement of the two markets' obligations and of the dispatch as ledger rows, each with its residual held below a printed tolerance.
4. The EDIT cell has 20 knobs: `MARKETS`, `MARKET_COLORS`, `MONITORED_LINES`, `ACTIVATED_LINES`, `LINE_RATINGS`, `GEN_FLEET`, `BACKSTOP_PRICE`, `BACKSTOP_MW`, `LOAD_7`, `LOAD_9`, `LOAD_5`, `SHED_PRICE`, `SLACK_BUS`, `STUDY_LINE`, `WLR_THRESHOLD`, `TAG_PRIORITY`, `GTL_PRIORITY`, `A_TAG_MW`, `PATH_NAMES` and `PATH_MO`.
5. The closing Sandbox section reruns the chain through `run_chain()`: the blind clears, Market A's redispatch, Example 2's all-tagged assignment, Example 3's partly tagged assignment with the generator allocation, and Example 4's clears against the market flow limits. It prints the requirement, the host's cost change and the two limits of 33.867 MW and 6.133 MW.

**Slide checks.** `tests/test_slides_workshop4.py` executes the notebook through `tests/slide_check.py` and holds 26 tests, one per slide from slide 54 to slide 84 except 56, 64, 72, 78 and 79, which print no number the notebook computes. The rows are the 421 the notebook asserted before 2026-11-08, plus 12 former plain asserts turned into rows, and the relief tables are compared as text. By slide: slide 54, 3 of 3; slide 55, 24 of 24; slide 57, 19; slide 58, 6; slide 59, 5; slide 60, 20; slide 61, 15; slide 62, 16; slide 63, 33; slide 65, 19; slides 66 to 69, 12 each; slide 70, 13; slide 71, 26; slide 73, 18; slide 74, 22; slide 75, 24; slide 76, 18; slide 77, 26; slides 80 and 81, 3 each; slide 82, 19; slide 83, 20; slide 84, 21, before the 12 added rows. The market flow limits are checked through (16) and their sum against the rating, and Market B's line 6-7 dual on slide 83 through its interior pair's offer gap over their shift-factor gap, so no run-produced number is typed into the test. The one exception is a picture from the August workshop, line 6-7 at 43/40 in the first reconciliation column, which slides 63 and 85 repeat and the notebook does not run. The notebook has 34 figures.

**Stamp.** 75 cells, executed with `tools/nb_execute.py` at 2026-11-08T17:20:54Z. The stamp's `lib_commit` is `ce66306` with `lib_dirty` true, because `lib/` was copied in on 2026-11-07 after that commit, and its `lib_manifest.sha256` is `cec8c27c5df3f385824463c14c9804ce745299b0b3371c1afd2ae5180ade5e03`, the SHA-256 of `lib/MANIFEST.md`, which lists the 14 modules' hashes. Re-executed with `tools/nb_execute.py` at 2026-11-11T01:06:04Z, its stamp's `lib_commit` `eb48135` with `lib_dirty` false and its `lib_manifest.sha256` `520d446bc3f9e3ab3641f710b0e97969d8bfa769a4225093d587a6392c1a308d`. Every text output and figure count is as before.

### `101_fundamentals.ipynb`

Entry dated 2026-11-10 (owner direction 2026-11-10: a starting framework for understanding the market from the simplest market design, with the fewest formulas and variables, the network and nodal dispatch figure for each example, and no settlement ledger or congestion revenue allocation).

**What it carries.** The 9-bus case cleared as a single-period, energy-only, lossless DC dispatch with offers at cost. The formulation has 5 numbered equations: the program with one power balance per area, the flow, the nodal price, each area's production cost and load payment, and the sum of two markets' flows. The nomenclature table has 14 rows. No deck is followed. The notebook has 7 figures: the configuration view, one figure per example, and three in the sandbox.

**The examples.** Example 1 is one market at the case ratings: no line binds, every bus prices at \$35/MWh, the production cost is \$8,475/h and load pays \$11,025/h. Example 2 rates line 6-7 at 40 MW: $\mu=-\$20.10$/MWh, the production cost is \$8,941/h and load pays \$9,745.1/h, asserted against `tests/golden/golden.json` (`line4_unified`) with eq. (3) checked at every bus. Example 3 clears the same market as two balancing authority areas joined by an unlimited interface. Its dispatch and nodal prices are asserted equal to Example 2's. BA A carries the whole \$8,941/h of production cost and transfers 215 MW to BA B, BA A's load pays \$3,712.6/h and BA B's \$6,032.5/h.

**The sandbox.** The EDIT cell has 6 knobs: `SB_RATINGS`, `SB_FLEET`, `SB_LOADS`, `MARKETS`, `ACTIVATED` and `MARKET_COLORS`. As stored, Market A and Market B clear independently, each enforcing its own lines. Market A binds line 6-7 at $-\$20.10$/MWh. The two markets' production costs are \$2,908.5/h and \$10,750/h, \$4,717.5/h above one market on the same knobs, and their summed flow on line 6-7 is 43.4 MW against its 40 MW rating.

**Stamp.** 23 cells, executed with `tools/nb_execute.py` on 2026-11-10. The stamp's `lib_commit` is `febf7d4` with `lib_dirty` false, and its `lib_manifest.sha256` is `520d446bc3f9e3ab3641f710b0e97969d8bfa769a4225093d587a6392c1a308d`.

### `sandbox.ipynb`

Entry dated 2026-11-07. On 2026-11-10 it moved from 101_sandbox.ipynb, its former name, to `sandbox.ipynb`, unchanged except its title cell, when `101_fundamentals.ipynb` took the simple formulation (owner direction 2026-11-10). It was re-executed on 2026-11-10, its stamp's `lib_commit` `febf7d4` with `lib_dirty` false.

**What it carries.** The 9-bus case cleared from one configuration cell, as one market (unified mode, the footprints being two balancing-authority areas) or as two independent markets (two-markets mode). It carries the engine, the settlement and allocation ledgers, an intertie scheduling limit and the loading relief, over one period, energy only, on a lossless DC network. No deck is followed. The formulation has 13 numbered equations.

**The slide-free checks.** Example 1 is the canonical anchor at the defaults: line 6-7 at 40 MW, $\mu=-\$20.10$/MWh and a rent of \$804.1/h, asserted against `tests/golden/golden.json`, with the settlement tables totalling the rent. Example 2 is two markets with a scheduling point at bus 8 under a 100 MW limit, stated as a hypothesis with six checks: the awards equal the limit, $\mu^s$ equals the bid less the bus-8 price, the export resource's price equals Market B's price at bus 8, the limit's rent equals $\mu^s$ times the awards, both forms of (7) close in Market A, and (6) closes in Market B. A third cell runs the scheduling limit at bus 9 and asserts its dual, awards, cost and price against the golden. The notebook has 7 figures.

**The knobs.** The EDIT cell has 45, in groups. The case: `LINE_RATINGS`, `SPLIT_5_6`, `FLEET`, `LOADS`, `SHED_PRICE`, `OFFER_MODE` (flat, curves or tranches), `CURVES`, `TRANCHE_MW`, `BACKSTOP`, `SHOW_IDLE_BACKSTOPS`, `GEN_MIN`, `DC_LINKS`, `FLOWGATES`. The markets: `MODE`, `MARKETS`, `MARKET_COLORS`, `MONITORED`, `ACTIVATED`, `OWN_LINES`, `ENGINE_RATINGS`, `RESERVATION`, `PSI`, `PSI_REFERENCE`, `FORECAST`. Interfaces and transfers: `INTERFACES`, `INTERFACE_VRL`, `FIXED_TRANSFERS`. Schedules, bids and offers: `SELF_SCHEDULES`, `FIXED_INJECTIONS`, `EXPORT_BIDS`, `IMPORT_OFFERS`, `SCHED_POINT`, `BID_STACK`. Real time: `RT_RELIEF`, `RT_PREMIUM`. Tags and loading relief: `TAGS`, `TAG_PRIORITY`, `WLR_LINE`, `WLR_THRESHOLD`. The ledgers: `UNASSIGNED_SPLIT`, `TE_SPLIT`, `CONGESTION_METHODS`, `TRANSFER_METHODS`, `TRANSFER_SPLIT`, `HUB_KIND`. The deck notebooks' knobs are listed in their own entries above: 14 in the congestion management notebook, 21 in the transactions notebook and 20 in the loading relief notebook.

**What it does not carry.** The closing Sandbox section lists 15 limits. The main ones: one period with no commitment, reserves, ramping, storage or sequential pricing; a rising offer cleared as a curve excludes bids, the scheduling limit, pinned transfers, must-run, the violation price and HVDC links; bids, offers and the scheduling point sit in one market at a time; in two-markets mode the interface, pinned-transfer, HVDC and flowgate knobs enter only the unified benchmark; the bid awarded by bisection under a continuous curve, the day-ahead to real-time chain with two settlements, and the loading-relief reconciliation by backstop or by market flow limit are not carried; balancing authorities nested inside a market, the ATC, TTC and rights books, the CRR auction, the N-1 de-rate and the risk layer are not exposed; the drawing layout is fixed to the 9-bus ring.

**Stamp.** 35 cells, executed with `tools/nb_execute.py` at 2026-11-08T07:41:21Z. The stamp's `lib_commit` is `ce66306` with `lib_dirty` true, because `lib/` was copied in on 2026-11-07 after that commit, and its `lib_manifest.sha256` is `cec8c27c5df3f385824463c14c9804ce745299b0b3371c1afd2ae5180ade5e03`, the SHA-256 of `lib/MANIFEST.md`, which lists the 14 modules' hashes.

---

## 2. The library

`lib/` is a byte copy of a shared 9-bus library, and `lib/MANIFEST.md` records each module's SHA-256 and the date of the copy. The index below gives each module's purpose and its public functions and classes, one line each, read from the module's docstrings. A name with a leading underscore is internal to its module and is not listed. The docstrings carry the full statement, with each function's arguments, units and signs.

### `seams_engine.py`

The engine. It clears the PTDF DC-OPF of one or more market engines on a shared network with `scipy.optimize.linprog` (HiGHS), decomposes each nodal price into its energy and congestion parts, and superposes the engines' flows. Its module docstring maps the code to the notation.

- `PTDFData` (class): the shared network's shift factors and its bus and line bookkeeping, stored line by bus.
- `compute_ptdf`: the DC shift-factor matrix of a PyPSA network.
- `susceptance_widths`: each line's susceptance, $b_m=1/x_m$, mapped to a drawing width.
- `FlowgateSet` (class): the activated constraint set as flowgates, each a weighted sum of line flows with a forward and a reverse limit.
- `identity_flowgates`: the per-line constraint set as a `FlowgateSet`, one flowgate per monitored line.
- `MarketEngine` (class): one independent optimization over a subset of the shared network.
- `EngineResult` (class): one engine's clearing, with its dispatch, nodal prices, line duals, own flows and the optional transfer, flowgate and limit fields.
- `own_injections`: the engine's own net injection per bus, in MW.
- `solve_engine_dispatch`: one engine's DC-OPF on the shared network, with a pooled energy balance.
- `solve_engine_qp`: one engine cleared with continuous linear marginal-cost curves, a convex QP.
- `solve_engine_transfers`: the EDAM-literal clearing of one engine, one power balance per area with scheduled transfers on the declared interfaces and their released-capacity duals.
- `actual_injection`: the injection vector of what the engines dispatched and served.
- `physical_flows`: the engines' flow components combined into the physical flow on every line.
- `to_supply_demand`: an engine and its result adapted to the supply and demand bars of `nodal_plot`.
- `shed_segments`: demand segments that draw shed load as a faint tail on each shedding bus's load bar.
- `served_by_bus`: the served MW at the buses that shed.
- `served_demand`: the served MW at every bus, demand less shed.
- `seam_dual_gap`: the pairwise gap between two engines' prices at each bus.
- `CaseNetwork` (class): one built network with its shift factors and its flowgates.
- `Case` (class): the case contract, which `wscc9_model.WSCC9Case` implements.
- `set_case`: installs a case as the ambient case and returns the previous one.
- `active`: the ambient case, the 9-bus case by default.
- `using` (class): a context manager that runs a block under a different ambient case.
- `CaseProtocol` (class): the surface every case exposes to the shared library.
- `DataBackedCase` (class): the extension a case built on a data bundle exposes; the 9-bus case does not implement it.
- `check_fleet`: the violations of the fleet schema, empty when the fleet conforms.
- `check_case`: the violations of the case contract, empty when the case conforms.
- `rt_relief`: the real-time re-clear of an engine, with day-ahead blocks re-offered at cost, increments at a premium, and the neighbor's realized flows fixed on the engine's activated lines.
- `actual_flows`: the superposed line loading with each generator and load counted once across the books.

### `wscc9_model.py`

The 9-bus teaching case: the network, the fleet and the loads, the market-engine factory and the layout constants.

- `fleet_with_backstop`: the teaching fleet plus a residual backstop unit at each load bus, priced above every real unit and sized to the bus's load by default.
- `build_network`: the 9-bus PyPSA network, with any line-rating overrides applied.
- `build_transport_network`: a small PyPSA network of buses joined in a chain by transfer lines.
- `shift_factors`: the DC shift-factor matrix of the network, the base case by default.
- `sf_table`: a readable shift-factor table, line by bus, for display.
- `make_engine`: a `MarketEngine` that owns every fleet unit and load at the named buses.
- `WSCC9Case` (class): the 9-bus case as a `seams_engine.Case`, the reference implementation of the contract.

### `ieee9_network.py`

The IEEE 9-bus test network as a PyPSA network, built from pandapower's case.

- `build_ieee9_network`: the IEEE 9-bus case loaded from pandapower, converted to PyPSA and enriched.
- `assign_zones`: a `zone` column added to the buses table.
- `get_seam_lines`: the lines that cross zone boundaries.

### `footprints.py`

The partition of the shared 9-bus network into market footprints, balancing authorities or markets, and the assignment of each line.

- `Footprints` (class): a two-footprint partition of the shared network, with each line's assignment.
- `make`: a `Footprints` built from bus definitions and the network topology.
- `ba_line_colors`: each internal line coloured by its footprint, with ties and unowned lines in their own colours.
- `line_status_table`: each line's location, internal or tie, and whether each footprint enforces it, monitors it, or neither.

### `revenue_allocation.py`

Settlement, congestion-rent and transfer-rent allocation, and the position ledgers, with the hub decomposition and the coalition and Pareto machinery.

- `ba_settlement`: each area's generator revenue and served-load payment at nodal prices.
- `settlement_by_bus`: the bus-level settlement of one engine, with a subtotal.
- `capacity_value_by_line`: each line's flow, rating, binding flag, shadow price and rent, with a total.
- `line_rent_table`: each line's congestion rent, $\lvert\mu_m\rvert\lvert F_m\rvert$, tagged by topology and by the managing footprint.
- `border_separation`: the cross-border separation rent per tie, the part of the price gap that line congestion creates.
- `pairwise_transfer_settlement`: the transfer settlement per interface, footprint pair by footprint pair.
- `fp_interfaces`: the interfaces a partition implies, the pairs joined by at least one tie.
- `declared_interfaces`: the declared interface set, $\mathcal I^M$, as sorted area pairs.
- `constraint_assign`: the home footprint of each flowgate for Method 1 on the constraint basis.
- `managed_constraints`: the activated constraint set of an area's own engine.
- `constraint_rent`: the congestion rent, $R$, over the result's constraint set.
- `constraint_rent_table`: the line rent table on the result's constraint set.
- `flowgate_rent_table`: each flowgate's congestion rent, with a total.
- `solve_with_transfers`: the unified clearing in the EDAM-literal form, with one balance per footprint and released-capacity limits on the interfaces.
- `mod29_interface_limits`: the MOD-029-style released capacities for a partition, with the simultaneous check across interfaces that path-by-path accounting skips.
- `allocate_congestion_rent`: the total congestion rent allocated to the footprints under Method 1 and Method 2.
- `compare_methods`: the footprint settlement beside the Method 1 and Method 2 allocations.
- `revenue_table`: two ledgers, one per allocation method, with each area's consumers and generators side by side.
- `settlement_regimes`: the settlement regimes of the independent-clearing rights construction on one clearing, each column summing to $R$.
- `cost_by_bus`: the marginal cost at each bus from a fleet.
- `independent_clear`: each area cleared as its own engine on the full network, enforcing only its own lines, with no interchange.
- `autarky_vs_unified`: the net-position ledger of independent operation against the unified clearing, for one allocation method.
- `autarky_vs_unified_congestion`: the same ledger with the congestion derivation, $R$, $N_a^c$ and the allocation, folded into the unified block.
- `embedded_rate`: the long-run scheduling-right rate, a revenue requirement spread over the firm scheduling capability.
- `scheduling_subscription`: the firm scheduling-right subscription an area's own customers buy at a clearing.
- `transmission_service_recovery`: each area's embedded-cost recovery under independent operation and joint dispatch.
- `scheduling_service_ledger`: the four-party provision and appropriation ledger under independent operation and joint dispatch.
- `transfer_rent`: the transfer rent, $R_T$, summed over the interfaces of an EDAM-literal result.
- `transfer_rent_by_interface`: the rent each binding interface collects.
- `allocate_transfer_rent`: the transfer-settlement shares, $\sigma_a$, under the fixed-share, net-payer and per-interface rules.
- `transfer_ledger`: independent operation against the transfer-constrained clearing, with each revenue stream shown collected and allocated.
- `self_schedule_ledger`: economic dispatch against a forced self-schedule, with the incentive ledger.
- `net_congestion_position`: each area's net congestion position, measured against its own ex-ante hub.
- `regional_congestion`: each region's load bill split into the generation it pays for and the congestion it bears, $N_a^c$.
- `transfer_settlement`: the transfer settlement, $T_E$, the scheduled interchange bought at the exporter's hub and sold at the importer's.
- `congestion_shift_breakdown`: the per-bus shift-factor build-up of the congestion by region, $N_a^c$.
- `hub_weights`: the ex-ante aggregation-point weights of each footprint, each summing to one.
- `hub_congestion_decomposition`: the slack-invariant congestion decomposition, each area settled against its own hub and the interchange settled hub to hub.
- `hub_flow_contributions`: each area's hub-form flow contribution on every line.
- `position_s`: the joining area's congestion by flow contribution on the binding lines outside it, returned to it, a Lab construction.
- `congestion_summary`: per region, what the dispatch produces beside what an allocation method allocates.
- `loop_flow_triangle`: one shadow price on a constraint internal to the host, read three ways.
- `rights_makewhole`: whether a transmission service provider can make its firm-rights holders whole from congestion revenue alone, or only once the avoided redispatch is credited.
- `ownership_allocation`: each area keeps the rent collected on the lines it owns.
- `methodology_ledger`: independent operation against a unified clearing under one congestion-rent allocation.
- `methodology_ledger_congestion`: the same ledger with the congestion derivation folded into the unified block.
- `methodology_ledger_4p`: the four-party ledger, with the path-right holder separated from general load.
- `bilateral_self_solve`: a balancing authority solving itself, production cost plus its redispatch premium.
- `pre_market_bilateral`: the pre-market bilateral baseline, a double auction whose reserved paths are not simultaneously feasible, fixed by local redispatch.
- `rr_margin_ledger`: the provision-layer ledger, each provider's revenue requirement with the margin that absorbs redispatch.
- `oatt_vs_tac`: two ways a provider recovers its revenue requirement, with opposite incidence.
- `parallel_flow_attribution`: each binding line's flow and congestion rent attributed to each area.
- `loop_flow_charge`: the neighbor's parallel-flow value on the host's binding lines, at the market price.
- `path_hedge_ledger`: a perfect hedge on path rights, with an inter-area-before-intra-area waterfall.
- `position_ledger`: each footprint's net position at a result's prices.
- `wheeling_settlement`: the settlement of a transfer that clears in a transport market and wheels across a contiguous market.
- `coalition_positions`: one interval of the coalition Pareto check, on drawn drivers.
- `coalition_pareto_draws`: the coalition Pareto check over a `risk.py` scenario set.
- `pareto_distribution_ledger`: one row per method, area and constituency, with the acceptance tests a member could apply.
- `pareto_verdicts`: the verdict per method, every interval and ex ante at each risk-aversion rung.
- `flow_ledger`: the money flows of an economic dispatch and a forced self-schedule.
- `seam_positions`: two-engine settlement positions, consumer payments negative.
- `seam_positions_unified`: the unified clearing in the same positions format.

### `atc.py`

Available transfer capability and the simultaneous feasibility test: path ratings, the netted and firm feasibility notions, the rights books under MOD-029 and MOD-030, the common test, the CRR auction, the N-1 helpers and the Western Loading Relief assignment.

- `path_shift_factors`: the per-MW loading of every line by a point-to-point reservation.
- `ttc`: the total transfer capability of a path and the first line to bind.
- `atc`: the firm ATC identity, TTC less TRM, ETC and CBM, floored at zero.
- `hub_path_shift_factors`: the per-MW loading of every line by a hub-to-hub transfer.
- `interface_ttc`: the rated-path TTC of each interface, each rated alone.
- `interface_sft`: every interface's transfer superposed at its posted rating, with each monitored line's loading.
- `Award` (class): a granted point-to-point reservation.
- `HubAward` (class): a granted transfer between two aggregation points.
- `Request` (class): a transmission service request in an authority's queue, with its holder, provider and tier.
- `dispatch_requests`: a clearing's dispatch read as a book of rights, each source's MW split over the sinks.
- `superposed_flow`: the signed flow on every line from the superposition of all awards.
- `line_loadings`: the per-line feasibility table for a set of awards.
- `simultaneous_feasibility`: Hogan's simultaneous feasibility test, which nets counterflow.
- `overloaded_lines`: the monitored lines that overload under an award set.
- `firm_line_loadings`: the per-line firm loading, which does not credit counterflow.
- `firm_feasibility`: whether the same-direction commitments sit within every monitored limit.
- `flow_dict`: the superposed award flow, by line.
- `FlowgateLedger` (class): the existing transmission commitments on a monitored constraint set.
- `flowgate_available_atc`: the remaining transfer capability of a path over a monitored constraint set.
- `available_atc`: the remaining ATC of a path given the awards already booked.
- `book_sequentially`: requests granted in order, each decrementing the ATC.
- `RightsBook` (class): one authority's granted book and the posting table behind it.
- `rights_book_rsp`: an authority's rights book under the rated system path method, MOD-029.
- `rights_book_afc`: an authority's rights book under the available flowgate capability method, MOD-030.
- `Nomination` (class): the outcome of the common simultaneous feasibility test.
- `nominate_and_test`: the common simultaneous feasibility test over the union of every book.
- `settle_rights`: an entitled set settled against a clearing's nodal prices or its constraint duals.
- `value_requests`: the value of a list of rights at a clearing's prices, each at its full MW.
- `union_test`: the common test over the union of several groups' books, with the loadings kept per group.
- `holder_regimes`: the holder-by-holder settlement regimes from per-right values and the pool.
- `ba_monitored_lines`: an authority's monitored set, its internal lines plus the ties.
- `ba_atc`: the ATC an authority posts for a path, seeing only its own lines.
- `Bid` (class): a transmission customer's bid into the CRR auction.
- `CRRResult` (class): the outcome of the CRR auction.
- `crr_auction`: Hogan's CRR auction, point-to-point rights awarded to maximize bid value subject to simultaneous feasibility.
- `awarded_rights`: the awarded set as a list of rights for the figure helpers.
- `post_contingency_ptdf`: the shift factors with one line removed.
- `post_contingency_flow`: the flow a result's injections place on a line after the contingency.
- `lodf_derate`: the outage distribution factor onto a surviving line and the N-1 derate.
- `wlr_gtl_component`: one area's untagged generation-to-load as a Western Loading Relief component on a line.
- `wlr_assign`: the relief a constraint needs assigned to the tags that load it, by the Western Loading Relief rule as the public record describes it.

### `bilateral.py`

Bilateral trading over transmission rights and its reconciliation to a feasible flow: the repeated double auction and the Cournot must-serve model.

- `LoadCenter` (class): a load bus with an inelastic forecast and a risk-averse premium curve.
- `Supplier` (class): a generator with a fixed marginal cost and a capacity.
- `Right` (class): a point-to-point transmission right rated at its ATC.
- `Portfolio` (class): one supplier's book of rights from its bus.
- `rate_right`: the ATC rating of one point-to-point right.
- `calibrate_load_centers`: the load centers for the must-serve forecast and the risk-averse premium.
- `plan_portfolio`: a supplier's book of rights, the most profitable load center first.
- `plan_all`: the portfolio plan for every supplier.
- `best_response`: a supplier's Cournot best response over the load centers it holds rights to.
- `BilateralResult` (class): the strategic equilibrium of the spot iteration.
- `clear`: the Gauss-Seidel iteration to the strategic equilibrium, with load served in full.
- `power_flow_check`: the DC power-flow feasibility of the self-scheduled awards.
- `BackstopResult` (class): the network reconciliation of the strategic and backstop schedule.
- `backstop_clear`: the combined schedule reconciled to a feasible power flow.
- `competitive_clear`: the competitive nodal clearing over the inelastic forecast.
- `compare`: the strategic and competitive outcomes per load center.
- `Belief` (class): the two-sided forward-market risk attitudes of Bessembinder and Lemmon.
- `Order` (class): one block in the order book, an ask or a bid.
- `OrderBookResult` (class): the cleared order book.
- `build_orders`: the fleet and the load turned into a book of belief-driven orders.
- `order_book_clear`: the order book cleared and scored against the first best.
- `double_auction_clear`: the repeated discriminatory double auction that approximates bilateral trading.
- `convergence_sweep`: the order book re-cleared at each book thickness.
- `monte_carlo`: the order book repeated over a number of draws, the forward-premium experiment.
- `order_book_redispatch`: the bilateral schedule balanced, with the uplift charged to the curtailed schedule.

### `risk.py`

The canonical per-node load and per-generator cost distributions, their propagation through the DC-OPF by Monte Carlo or Gauss-Hermite quadrature, and the risk preferences, with risk aversion, $\gamma$, and the mean-variance certainty equivalent.

- `Belief` (class): the forward-premium primitives of the double auction, kept here as the one canonical instance.
- `Normal` (class): a one-dimensional normal driver.
- `LogitNormal` (class): a driver bounded on (0, 1), the shape of an available-capacity fraction.
- `load_dists`: the per-bus load drivers.
- `cost_dists`: the per-generator cost drivers.
- `Driver` (class): one named uncertain factor with its marginal distribution.
- `DriverModel` (class): a bundle of drivers with the membership that resolves them.
- `ba_drivers`: the per-area common-factor and idiosyncratic driver model.
- `Scenarios` (class): a weighted set of draws over the named drivers.
- `monte_carlo`: independent Monte Carlo draws with uniform weights.
- `gauss_hermite`: tensor-product Gauss-Hermite quadrature over the drivers.
- `day_bootstrap`: a block bootstrap over historical whole days, for a case built on a data bundle.
- `clear`: one DC-OPF scenario cleared.
- `propagate`: a quantity evaluated over every scenario's clearing, with its weights.
- `path_congestion`: the price separation a point-to-point right spans, as a quantity for `propagate`.
- `induced_lmps`: the induced nodal price at every bus, one solve per scenario.
- `emean`: the weighted expectation.
- `evar`: the weighted variance.
- `equantile`: the weighted quantile.
- `certainty_equivalent`: the mean-variance certainty equivalent, the mean less half the risk aversion times the variance.
- `risk_loaded_premium`: the risk-loaded hedge value of a congestion exposure.

### `etag.py`

The interchange tag as data, the vehicle a transaction between two markets rides on, with its NERC priority codes and its figures. It does no clearing of its own.

- `priority_level`: the priority level of a segment, 0 lowest and 7 firm.
- `Segment` (class): one provider's leg of the tagged path.
- `ETag` (class): an interchange tag, who moves how much from where to where on what service.
- `fit_table_header`: a figure's table header shrunk until every label fits its column.
- `tag_figure`: the tagged path over the network, with the segment table beside it.
- `physical_path_figure`: the tag's physical path as a table in the layout of the Markets+ seams deck.
- `tier_of`: the curtailment tier of a NERC transmission priority code.
- `dispatch_tags`: the generator-to-load tags of one market's dispatch, each unit's output split over the served loads.

### `intertie.py`

The placement of an intertie schedule, at the tie bus (SP-Tie) or over the exporter's aggregation buses (GAP-Tie), with the clearings and figures that compare the two.

- `configure`: sets the module context.
- `build_net`: the shared network with the studied flowgate derated to its rating.
- `tie_inject`: the schedule as bid generation, placed by the tie model.
- `tie_clear`: the importer's market cleared with the schedule modeled as SP-Tie or GAP-Tie.
- `import_mw`: the schedule's cleared MW, summed over the buses the placement spreads it across.
- `tie_figure`: the importer's own model of the schedule.
- `neigh_clear`: the other footprint serving the transaction.
- `neigh_figure`: the other footprint's view.
- `combined_actual`: the physical state the two models approximate, each market's real generation serving every load.
- `combined_figure`: the combined actual dispatch on the network, each side priced at its own market's prices.
- `three_set`: the importer's model, the exporter's model and the combined actual dispatch for one clearing.

### `seam_views.py`

The presentation frames of the seam exhibits: each market's own view, the inputs it clears around, the superposition, the deck tables and the price walk.

- `configure`: sets the module context.
- `own_view`: one market's own view of its clearing.
- `relief_view`: the host market's real-time relief, drawn against the day-ahead engine.
- `bid_view`: one market's book as configured, with a price-sensitive bid at a node.
- `input_view`: one market's inputs before it clears, with the schedule it solves around.
- `market_inputs_view`: one market's inputs with no schedule, with the injections it assumes at the neighbor's buses.
- `combined_views`: the superposition, as the combined figure and then the network with net injections and flows.
- `forecast_view`: the forecast book alone, its dispatch and its flow contributions.
- `export_pair`: the deck's two tables, settlement by node and the line table with each market's flow, dual and rent.
- `price_walk`: the production-cost change behind one node's price.
- `walk_text`: the walk as text, one line per marginal resource.
- `strip`: the figure without the suptitle, panel titles and legends its slide carries.
- `save_clean`: the deck PNG copy of one exhibit, written only when saving is enabled.

### `wscc9_figures.py`

The shared figure compositions: the network and nodal-dispatch composite, the transfers inset, the rights view and the induced-flow panels.

- `footprint_figure`: the network and nodal-dispatch composite for one clearing.
- `composite_figure`: the unified two-panel composite, nothing grayed.
- `transfer_inset`: the transfers inset, drawn from explicit values.
- `draw_net_dispatch`: the transfers inset for a footprint pair, read from an EDAM-literal result.
- `transfer_figure`: the footprint figure with the transfers inset.
- `redispatch_bars`: the supply and demand bars that mark a market's own incremental dispatch at a bus.
- `bid_bars`: the supply and demand bars for a price-sensitive bid at a node.
- `self_schedule_bars`: the supply and demand bars that draw a price-taking self-schedule as a gray block.
- `draw_rights_arcs`: sold point-to-point rights overlaid as directed arcs from the point of receipt to the point of delivery.
- `induced_flow_panel`: the flow one point-to-point transaction places on the network, with its impact on a constraint.
- `net_injection_flow_panel`: the flow a balanced set of net injections places on the network.
- `rights_figure`: the transmission-rights view, the network panel and the rights table.
- `rights_payoff_figure`: the rights drawn over the unified clearing, with each right's FTR payoff.
- `self_schedule_figure`: the two-panel view of the self-schedule incentive.
- `install_latex_repr`: booktabs LaTeX rendering of DataFrames on PDF export.
- `award_supply_demand`: the net injection per bus from point-to-point awards.
- `award_figure`: an ATC award set drawn on the network, with overloads in red.
- `bilateral_dispatch_figure`: the bilateral clearing on the network and the nodal dispatch figure.

### `nodal_plot.py`

The network-topology figure and the nodal chord figure of generation, demand and flows.

- `assign_bus_colors`: a distinct colour for each bus.
- `compute_proportional_flows`: proportional flows for a copperplate market.
- `extract_line_flows`: the line flows of one hour from EGRET results.
- `trace_gen_to_load`: generation-to-load flows by proportional power tracing, Bialek's method.
- `compute_ptdf_flows`: the generation-to-load flow allocation by PTDF-based proportional tracing.
- `prepare_bus_data`: market orders organized into per-bus supply and demand.
- `plot_nodal_circlize`: the chord figure of nodal generation, demand and flows.
- `plot_nodal_from_results`: one timestep chosen and plotted.
- `rotate_coords`: a bus layout rotated about the origin.
- `plot_network_topology`: the network topology figure, with bus colours matching the chord figure.
- `plot_combined_letter`: the letter-size composite of the topology and the chord figure.

### `lab_types.py`

The repository's type aliases, imported only under `TYPE_CHECKING`, so that one object carries one name in every hint and no notebook gains a runtime import. It defines no function. The aliases are `Bus`, `Line`, `Area`, `Resource`, `Interface`, `BusValues`, `LineValues`, `AreaValues`, `Loads`, `Unit`, `Fleet`, `AreaBuses`, `Interfaces` and `FloatArray`.

---

## 3. Publishing

The library is copied in from its shared source as one set of 14 modules, and `lib/MANIFEST.md` is regenerated with each module's SHA-256 and the date of the copy. The guard and the modules it imports arrive the same way, recorded in `.claude/hooks/MANIFEST.md`. A manifest hash is the SHA-256 of the file byte for byte as stored in git, which is the file with its line endings brought to LF under `.gitattributes`. A Windows working copy with CRLF endings therefore hashes the same as a Colab, Linux or macOS checkout. `tools/doctor.py`, the session brief and `tools/tests/test_githooks.py` compare each file with its row in this form. The notebooks are executed headless with `tools/nb_execute.py` before a commit, so that each notebook's stamp carries the library hash.

---

## 4. The canon's dated deviations

The deviations and renames of the canon that bear on the notebooks here, with their dates.

- **2026-06-27 and 2026-10-18, the risk vocabulary.** Risk aversion is $\gamma$ everywhere (2026-06-27). The ATC reservation margin, first written $\beta$, is $\psi$ (user decision 2026-10-18). The docstring cost shock $\psi_{a(g)}(s)$ in `risk.py` is a separate symbol, which no notebook renders.
- **2026-07-14, the pairwise transfer form.** The transport layer is part of the canonical program from the start, in its pairwise form.
- **2026-07-30, the `*In words.*` blurb.** Every numbered display equation is followed by an `*In words.*` blurb.
- **2026-07-31, the three-tier line distinction.** The network line set, each engine's monitored lines and each engine's activated constraint set are distinct sets.
- **2026-08-05, "Read it." retired.** The "Read it." marker under a figure or printed result is retired, because it labelled what reads as prose.
- **2026-08-06, the allocation-versus-coordination framing retired.** The label pair and the "two instances of one allocation or appropriation problem" framing for congestion revenue allocation and seams are retired; `CLAUDE.md` states the retirement.
- **2026-08-20, the declared interface set.** The declared interface set is $\mathcal I^M$, script $\mathcal I$ and not script $\mathcal T$.
- **2026-11-06, Method 1's general form.** Method 1 is $A^{(1)}_a=R^{\mathrm{own}}_a+s_aR^{\mathrm{unassigned}}$: each area keeps the rent on the lines assigned to it, and the rent on lines assigned to no area is split by the shares $s_a$, half each by default (`unassigned_split` of `allocate_congestion_rent`). The form $A^{(1)}_a=R^{\mathrm{int}}_a+\tfrac12R^{\mathrm{tie}}$ is the case with both ties unassigned.
