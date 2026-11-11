# The nomenclature canon

This file holds the nomenclature canon of the 9-bus series as this repository's notebooks use it: the conventions, the annotated symbol tables with their code names and the naming rules, the allocation methods, the hub decomposition, the seams and transfer-risk vocabulary, the seams-track mechanics and the open fix-on-touch items. It is read before writing or reading math, a symbol table or a ledger. A notebook is named by its file. The dated history of the canon (the deviations found and resolved, the renames and the retired conventions) is `NOTEBOOK_RECORD.md` section 4. `python .claude/skills/writing-voice/scripts/prose_lint.py --list-rules` names the conventions below that are checked mechanically.

The series shares one mathematical vocabulary, held in the tables below and in the nomenclature table of `sandbox.ipynb`. A notebook's math is written and edited to match this canon, and the dated deviations in the record distinguish drift from intent.

## Contents

1. Conventions
2. The fundamentals chain
3. The annotated symbol tables and the naming rules
4. The 9-bus partition and the two allocation methods
5. The hub decomposition and the reference bus
6. The seams and transfer-risk vocabulary
7. The seams-track mechanics
8. Open items, fixed on touch

---

## 1. Conventions

- **Sets are script capitals; the element is the matching lower-case letter; upper-case names a market engine.** Case is load-bearing: lower-case $m$ is a *line*, upper-case $M$ a *market engine*, script $\mathcal M$ the *set* of lines.
- **Line index is $m$** (never $\ell$/$l$); **BA / area index is $a$**.
- **Shift factor is $SF_{n,m}$, bus first, line second** (italic `SF`, not `\mathrm{SF}`/`\mathit{SF}`).
- **The line dual $\mu_m$ is *signed*; the LMP adds $+\sum_m SF_{n,m}\mu_m$.** (This is the opposite sign from the EDAM tech description's non-negative $\mu$ with $-\sum SF\,\mu$. The signed-$\mu$/`+` form is the house convention.)
- **Notebook math is `$…$` / `$$…$$` only, never `\(…\)` / `\[…\]`** (VS Code renders the latter as raw text; Jupyter renders both).
- **Every numbered display equation gets an "In words." blurb.** Immediately after the equation, a short paragraph opening `*In words.*` reads the equation in plain language: follow the equation's structure (left to right; objective, then each constraint with its dual), name each quantity by its nomenclature-table name and keep the symbol inline where it anchors the reading ("the absolute shadow price $|\mu_m|$ times the absolute flow $|F_m|$"), and say what a symbol *is* before using it ("Market $M$'s congestion rent $R^M$ is…", never a bare "$R^M$ is…"). The blurb states what the equation **says**. The interpretation and the so-what live in the surrounding prose, not the blurb. The formulation cells of `sandbox.ipynb` carry the construct.
- **`*In words.*` is the only labelled construct, and it belongs to equations alone.** Interpretation of a figure or a printed result is ordinary prose under the output: no marker, no bolded lead-in, no "Read it.". Such prose is kept limited, concise, factual and matter-of-fact, on the same terms as every other markdown cell.
- **A notebook's formulation section opens with the nomenclature table**, `| Symbol | Code | Definition |`, mapping every symbol the notebook uses to the object's name and to where it lives in the shared libraries (the formulation section of `sandbox.ipynb` is the model). The canon's names are reused verbatim; a sibling notebook's table is the starting point, trimmed to the symbols actually used.
- **Self-contained numbering.** Each notebook numbers its *own* equations from (1) and never cites another notebook's §/equation numbers, or an external paper's. Inside a notebook, `(4)` always means *that* notebook's (4). A sibling is named by file. A load-bearing derivation, definition or number is stated in the notebook that needs it, re-derived and renumbered from (1). A pointer may say where to read *more*; it never says where to read *this*.

## 2. The fundamentals chain

The (1)-(7) chain is stated in `.claude/skills/modify-example/SKILL.md`, "The formulation", item 2. Two details that skill item leaves implicit are stated here.

1. The nodal LMP (6) is an identity in two forms: the solver's ($\hat\lambda_a$ + $SF\mu$, the per-BAA form `solve_engine_transfers` literally solves) equals the aggregation-point form ($\lambda_a$ + hub-relative $SF\mu$). The aggregation-point form is EDAM §10's $LMP_i=\lambda_j/LPF_i-\sum SF\mu_m$, read lossless ($LPF{=}1$) and with the opposite $\mu$ sign.
2. The scheduled-transfer (transport) constraints ride in the canonical program from the start. A sibling that needs the chain re-derives it and renumbers from (1).

## 3. The annotated symbol tables and the naming rules

**Core symbols: the one-engine nodal clearing.**

| Symbol | Meaning |
|---|---|
| $\mathcal N,n$ / $\mathcal M,m$ / $\mathcal R,i$ / $\mathcal E,M$ | buses / the **network line set** (every line in the shared network) / resources / market engines |
| $\mathcal M^{M}_{\mathrm{act}}\subseteq\mathcal M^{M}_{\mathrm{mon}}\subseteq\mathcal M$ | the **three-tier line distinction**: the network line set holds every line; each engine's **monitored** lines $\mathcal M^{M}_{\mathrm{mon}}$ are the subset it models and watches (flows computed and reported); its **activated** constraint set $\mathcal M^{M}_{\mathrm{act}}$ is the configurable subset of the monitored lines whose limits enter its clearing. "Monitored" is per-engine, not a name for the shared set. Older notebooks still say "monitored lines" for $\mathcal M$ (section 8). |
| $c_i,\bar g_i,g_i,\mathrm{bus}(i)$ | resource cost, capacity, dispatch, host bus |
| $d_n$ / $\mathrm{exo}_n$ / $u_n$ / $V$ | load / fixed exogenous injection / shed MW / shed price |
| $SF_{n,m}$ / $F_m$ / $\bar F_m$ / $p^{\mathrm{inj}}_n$ | shift factor / flow / rating / net injection $g_n-d_n+\mathrm{exo}_n$ |
| $\lambda$ / $\lambda_n$ / $\mu_m$ | pooled energy price / nodal LMP $=\lambda+\sum_m SF_{n,m}\mu_m$ / signed line dual |
| $R$ | congestion rent $=\sum_m\lvert\mu_m F_m\rvert=\sum_n\lambda_n d_n-\sum_n\lambda_n g_n$, what load pays less what generation is paid; no binding line means no price separation and $R=0$ |

**Transport (transfer) layer.**

| Symbol | Meaning |
|---|---|
| $\mathcal N_a$ / $a$ | BA $a$'s bus set / the BA index ($\mathcal N_1$ is BA-1's) |
| $\mathcal A$ / $\hat\lambda_a$ | the set of BA areas **in the market** (the $\mathcal N_a$ partition $\mathcal N$) / per-BAA power-balance dual, the area's energy price (one balance per area, coupled by the transfer variables; `res.area_prices`); the transfer FOC $\mu_T^{ab}=\hat\lambda_b-\hat\lambda_a$ (binding; $\hat\lambda_a=\hat\lambda_b$ slack) ties the areas' prices together, every $\hat\lambda_a$ collapses to the pooled $\lambda$ when no interface binds, and the hub price is $\lambda_a=\hat\lambda_a+\sum_m SF^a_m\mu_m$ |
| $\mathcal M^{\mathrm{tie}}$ / $\partial\mathcal N_1$ / $s_m$ | ties / BA-1's boundary lines / boundary orientation $\pm1$ |
| $\mathcal I^M\subseteq\mathcal A\times\mathcal A$ | the **declared interface set**: the ordered area pairs $(a,b)$ on which the engine carries a scheduled-transfer variable (both orientations of a declared interface are members; `interfaces` in `solve_engine_transfers`). Quantifies the released-capacity family $t_{ab}\le\bar T_{ab},\ (a,b)\in\mathcal I^M$ and the Lagrangian transfer sum, in parallel to $m\in\mathcal M^M_{\mathrm{act}}$. Letter is script $\mathcal I$, NOT script $\mathcal T$. EDAM reduction: EDAM's transfer index is $(j,k,l)$ with possibly several transfers per intertie, and its scheduling limits are members of the generalized constraint set $m$ with the single dual $\mu_m$; the model declares at most one interface per area pair, so the pair is the full index and the interface duals stay their own family $\mu_T^{ab}$ |
| $t_{ab}$ / $\bar T_{ab}$ / $\mu_T^{ab}$ | scheduled transfer $a\to b$ (flow-inert; the min-total-schedule tags, `res.transfers`) / the interface's released capacity / the interface dual $=\hat\lambda_b-\hat\lambda_a$ when its released capacity binds ($\ge 0$ in the export direction) |
| $E_a$ / $E,\bar E,\mu_T$ | net interchange $E_a=\sum_b(t_{ab}-t_{ba})$ (export +) / the TWO-AREA abbreviation: $E\equiv t_{12}$, $\bar E\equiv\bar T_{12}$, $\mu_T\equiv\mu_T^{12}$ |
| $R_T$ | transfer rent $=\sum_{ab}\mu_T^{ab}\,t_{ab}$ (the **full** stream; per-interface streams $R_T^{ab}$, `transfer_rent_by_interface`) |
| $J_s$ / $\bar Q_s$ / $\mu^s$ / $R^{\mathrm{ISL}}$ | an intertie scheduling limit $s$ (2026-11-06, `schedule_limits` in `solve_engine_transfers`): the set of export awards (demand bids at the scheduling point) it names / its limit in MW / its dual $\ge0$, the gap between the marginal award's bid and the bus price, carried outside the per-bus LMP (the export resource's price is $\lambda_n+\mu^s$) / its rent $\mu^s\sum_{j\in J_s}q_j$, outside $R$, $R^{dc}$ and $R_T$: settled at `lmp` the conservation identity is unchanged, and a settlement that prices the awards at $\lambda_n+\mu^s$ adds $R^{\mathrm{ISL}}$ to its right side; where no award of the limit is partial, $\mu^s$ is whichever point of its bracketing interval the solver reports (`res.schedule_limit_dual`, `res.schedule_limit_mw`); the coefficient is the source's $-1$ on an export at its scheduling point (`caiso_dame_edam2023` p. 178, tariff appendix C A.3), $+1$ on the award as a withdrawal, and the source's limit is net and two-sided where the row is gross and one-sided |

**Per-BA settlement and allocation.**

| Symbol | Meaning |
|---|---|
| $G_a$ / $L_a$ / $N_a$ | BA gen revenue $\sum\lambda_n g_n$ / load pay $\sum\lambda_n d_n$ / **net position** $L_a-G_a$ |
| $\sum_a N_a$ | $=R$ with no transfer binding, $=R+R_T$ when the transfer binds |
| $N_a^c$ / $N_a^T$ / $e_a$ / $F_m^a$ | **congestion by region** $=\sum_{n\in a}(\lambda_n-\lambda)(d_n-g_n)=-\sum_m\mu_m F_m^a$ (partitions $R$; `regional_congestion`, `congestion_summary`) / transfer scarcity / net import / BA $a$'s flow contribution $\sum_{n\in a}SF_{n,m}(g_n-d_n)$ |
| $R_a$ / $\tau$ | line-ownership rent $\sum_{m\in\mathcal M_a}\lvert\mu_m F_m\rvert$ / cross-border separation rent $\sum_{\mathrm{tie}}\lvert\Delta\lambda^{\mathrm{cong}}_m F_m\rvert$ (`border_separation`, the tie price-gap object); $\tau$ has a second sense (naming rule 3) |
| $\lambda_a^{w}$ / $N_a^{c,w}$ | hub (aggregation-point) price $\sum_{n\in a} w^a_n\lambda_n$, the ex-ante weighted average of the area's **own** LMPs ($w$ = ex-ante ELAP/GAP weights, never dispatch-weighted; `hub_weights`) / **slack-invariant congestion by region** $\sum_{n\in a}(\lambda_n-\lambda_a^{w})(d_n-g_n)$; partition $\sum_a N_a^{c,w}+\sum_a\lambda_a^{w}e_a=R+R_T$ (`hub_congestion_decomposition`) |
| $T_E$ | transfer settlement, the scheduled transfer at the hub-price gap, $T_E=\sum_a\lambda_a^{w}e_a$ |
| $A^{(1)}_a,A^{(2)}_a$ / $\sigma_a,A^T_a$ | Method 1 / Method 2 congestion allocation (section 4) / transfer-settlement share (Method 2's split of $T_E$; `te_split`, the knob `TE_SPLIT` of `sandbox.ipynb`) & transfer-rent allocation |

**Naming rules.**

1. **$N_a$ is not congestion.** The net position $N_a=L_a-G_a$ (`ba_settlement(...)['net_into_pool']`) is dominated by the energy of inter-region trade, because an importer's $N_a$ is mostly its import bill, which pays *other* regions' generators. Only the system total $\sum_a N_a$ equals $R$ (or $R+R_T$ with an interface binding). A per-region $L_a-G_a$ is never labelled "congestion rent". The congestion revenue by region is $N_a^c$, which alone partitions $R$, and the decomposition is $L_a=G_a+\lambda e_a+N_a^T+N_a^c$. `sandbox.ipynb` states its hub form as its (8). $N_a^c$ depends on the reference bus for an unbalanced area, so the slack-invariant statement is the hub form, $\sum_a N_a^{c,w}+T_E=R+R_T$, against ex-ante hub weights (section 5).
2. **$\hat\lambda_a$ is not $\lambda_a$.** The area energy price $\hat\lambda_a$ (`res.area_prices`) is not the hub price $\lambda_a$ ($\lambda_a^w$, `ra.hub_weights`), against which $N_a^{c,w}$ and $T_E$ are measured, and a text names which it means.
3. **$\tau$ carries two senses**: the cross-border separation rent (`border_separation`) and the Method 1 → Method 2 bridge $A^{(2)}_h-A^{(1)}_h$ for the net payer $h$ (`revenue_table`, `summ['tau']`, printed in the allocation table of `sandbox.ipynb`). Every use names which one is meant. A third $\tau$, a jointly-impacted-constraint qualification threshold, is unrelated to both.

## 4. The 9-bus partition and the two allocation methods

**The BA partition is an accounting overlay** on the *same* physics. BA-1 = buses {2,8,7,6,3} (the cheap fleet); BA-2 = buses {1,9,4,5} (the dear unit). Two tie lines: `line_2` (5–6) and `line_7` (8–9). Cheap BA-1 exports to dear BA-2. `sandbox.ipynb` names the two areas by their footprints, the Market A area (BA-1's buses) and the Market B area (BA-2's).

**The allocation question.** Split `R` between the BAs. Per-BA net payment `N_a = L_a − G_a` (these sum to `R`). Tag rent by line ownership: `R = R_tie + Σ_a R_int_a`.

- **Method 1 (home-keeps-it):** `A¹_a = R_own_a + s_a R_unassigned`, the rent on the lines assigned to `a` plus `a`'s share `s_a` (half each by default, `unassigned_split=0.5`) of the rent on lines assigned to no area (`allocate_congestion_rent`). A tie is assigned or unassigned like any other line: the default `OWN_LINES` of `sandbox.ipynb` assigns `line_2` to BA-1 and `line_7` to BA-2, so no tie's rent is split there, and the half-split form `A¹_a = R_int_a + ½R_tie` is the case with both ties unassigned. The rent follows the *transmission element*, even when an internal constraint separated prices in the *other* BA.
- **Method 2 (net congestion position):** `A²_a = N_a^c + σ_a T_E`. Each BA keeps the congestion its own gen-to-load use creates on the binding constraints, wherever they sit, plus its share `σ_a` of the transfer settlement (`Σ σ_a = 1`; `sandbox.ipynb` defaults to the even split, `TE_SPLIT = 0.5`). Method 2 differs from Method 1 by the single cross-BA bridge `τ = A²_h − A¹_h` for the net payer `h`, the opposite amount moving for the other BA. It conserves total rent. The sign of `τ` is a property of the scenario, not of the rule: `T_E` is negative whenever the hub-to-hub exchange is net counterflow on the binding lines. The canonical clearing, Example 1 of `sandbox.ipynb`, binds `line_4` (bus 6 to bus 7) at 40 MW with the signed dual $\mu_{\mathrm{line\_4}}=-\$20.10$/MWh and $R=\$804.1$/h. There Method 1 gives the whole rent to BA-1, which holds `line_4`, and Method 2 at $\sigma_a=\frac12$ gives BA-1 \$932.46/h and BA-2 $-\$128.37$/h.

## 5. The hub decomposition and the reference bus

**`N_a^c` is reference-bus-dependent for an unbalanced area, so the convention is named.** Under a slack change `N_a^c` shifts by `(λ_r' − λ_r)·E_a` (net export × reference-price change). The **slack-invariant statement is the per-BAA hub decomposition** (the EDAM per-BAA-λ convention): `N_a^{c,w} = Σ_{n∈a}(λ_n − λ_a^w)(d_n − g_n)` against the area's **own** ex-ante aggregation point, plus the hub-to-hub transfer settlement `T_E = Σ_a λ_a^w e_a`; partition `Σ N_a^{c,w} + T_E = R + R_T`. Every term is a price/shift-factor *difference*, so the slack drops out. The residual convention is the hub definition (ELAP vs GAP), bounded by the area's internal spread × interchange, and an explicit design object: the SP-Tie/GAP-Tie question. It is computed via `hub_congestion_decomposition` + `hub_weights` (weights fixed ex ante, never dispatch-weighted). The legacy `N_a^c` equals the hub-at-slack-bus special case: fine in a notebook that states its slack, wrong to present as convention-free. Balanced books (rights sets, balanced self-schedules) are invariant without any hub. The objection that established the rule and its worked example are in `NOTEBOOK_RECORD.md` section 2, `revenue_allocation.py`.

## 6. The seams and transfer-risk vocabulary

Risk aversion is **$\gamma$** everywhere (domain-standard mean-variance/Arrow–Pratt coefficient) and the ATC reservation margin is **$\psi$** everywhere (renamed from $\beta$ on 2026-10-18, `NOTEBOOK_RECORD.md` section 4).

- **Seams and transfer risk:** $\psi$ = ATC reservation **margin** (the knob `PSI` of `sandbox.ipynb` reserves $\psi$ times the other market's reference flow on each activated line, its (11)); $\gamma$ = **risk-aversion** coefficient ($\mathrm{CE}_\gamma=\mathbb E\Pi-\tfrac\gamma2\mathrm{Var}\,\Pi$ for a payoff $\Pi$; for a COST the penalty adds, $\mathrm{CE}_\gamma[C]=\mathbb E C+\tfrac\gamma2\mathrm{Var}\,C$, and a notebook that uses it states the mirror); $W$ seam cost.

## 7. The seams-track mechanics

The seams track holds one distinction, intra-market against inter-market, keyed by the activated-constraint set and shown numerically in `sandbox.ipynb`, whose `MODE` knob clears the same knobs as one market or as two independent markets. The two regimes are never labelled an allocation versus coordination pair, and congestion revenue allocation and seams are never presented as two instances of one allocation or appropriation problem (retired 2026-08-06, `NOTEBOOK_RECORD.md` section 4). A text states which engine enforces which constraint, whose flow component it prices, and what the superposition leaves unpriced. A **market** is an independent clearing engine and footprint, and a **balancing authority (BA/BAA)** is an area within a market. A market is never called a balancing area. The counterfactual is independent operation, not autarky: each BA as its own seams-style engine on the full network, enforcing only its own internal lines. The mechanics below are the detail each seams notebook needs.

- *Intra-market (one engine).* One co-optimised clearing over all footprints; the BA partition is an accounting overlay; one engine enforces every activated constraint and prices the whole flow on it (Example 1 of `sandbox.ipynb`).
- *Inter-market (two engines).* Two independent clearings on the shared grid with **disjoint** activated sets; each seam line is allocated to ONE managing market (line_2→A, line_7→B, the one-tie-each assignment of the default `MONITORED` of `sandbox.ipynb`), but a manager enforces only its OWN flow component, so the superposition is priced by *neither*, with the interchange settlement riding on top. `workshop2_congestion_management.ipynb` and `workshop4_loading_relief.ipynb` work the shared-constraint side (no forecast, over-forecast, the redispatch to the realized flow, the loading relief); `workshop3_transactions_across_boundaries.ipynb` works the interchange side (each market pricing every node, a balanced self-schedule, a price-sensitive bid).
- **Interchange representation.** In the two-independent-markets case the seam schedule is a **fixed external injection** (`exo=`), not a co-optimised transfer; SP-Tie/GAP-Tie is the *placement* of that fixed schedule. Example 2 of `sandbox.ipynb` enters Market A's awards in Market B's clearing at bus 8 this way. `solve_engine_transfers` (the co-optimised `t_ab`/`μ_T` form) is the **intra-market** transport layer (the unified mode of `sandbox.ipynb`). A transfer that one market clears and the other wheels is co-optimised *within* the first and enters the second as `exo`. Two independent markets are not modelled with `solve_engine_transfers`.
- **Congestion detection = the SFT over the UNION of both books.** `atc.simultaneous_feasibility` (netted, credits counterflow) / `atc.firm_feasibility` (no counterflow) is the shared tool for all three seam faces (interference / phantom / contingency). No notebook here calls it; the superposed flow (`physical_flows`, the workshop notebooks' `superposed`) is the same union read on the cleared dispatch.
- **One LODF/N-1/ψ contingency apparatus for both regimes**: nested day-ahead ⊂ real-time in one market, and two independent markets. The helpers `post_contingency_ptdf` / `post_contingency_flow` / `lodf_derate` live in `atc.py`, and a notebook in either regime calls them. No notebook here does.
- **`SHED_PRICE`.** `SHED_PRICE` is 150 in every notebook of this repository. A notebook that adopts another value, such as an intertie VOLL of 300, states it and checks the phantom price it can move.

## 8. Open items, fixed on touch

1. **"Monitored lines" for the network line set.** Older notebooks say "monitored lines" for $\mathcal M$. The fix on touch is "network line set", because "monitored" is per-engine (section 3).
2. **The limit index $s$ against the orientation $s_m$.** The intertie scheduling limit's index $s$ ($J_s$, $\bar Q_s$, $\mu^s$, 2026-11-06) and the boundary orientation $s_m$ of section 3 share a letter, and the sandbox's scheduling limit carries the index. One of the two takes another letter on touch, and the choice is the participant's (found 2026-11-06).
