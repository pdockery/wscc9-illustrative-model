---
name: modify-example
description: Changes a deck notebook or the sandbox under the series conventions, EDIT knobs first and then cells, with the methodology check after every code change and a stamped re-execution. Use for any change to a root notebook, and for a proposed change to the library the notebooks call.
disable-model-invocation: false
---

# Modifying an example

You are changing a notebook whose stored outputs are the record of a workshop example. Each deck notebook follows its deck's slide sequence, and the sandbox notebook exposes the engine's configurable features and lists, in its closing section, the ones it does not carry. The ask arrives as `$ARGUMENTS` or in the chat; where it names no notebook, ask which one before reading. `AGENTS.md` carries the rules that hold in every session, `CANON.md` at the root carries the annotated canon, `figures.md` beside this file the combined figure, `NOTEBOOK_RECORD.md` the dated record, and `orientation` has run. The reconciliation exchange of that skill applies to every instruction here: restate the ask, place it in the literature, propose, and wait where the ask is large or the record contradicts it.

## Intake

If `orientation` has not run in this session, say in one sentence that you are running it first because every skill assumes its reading, invoke it, deliver its brief, and then begin the interview. Skip it only where the participant has said to skip it, and then say which parts of the reading you are proceeding without.

The participant may have pasted only the starter, with no inputs, and may be new to the repository. Where an input is missing, interview the participant for it before reading anything, one question at a time, with the AskUserQuestion tool where it is available. Each question says in a sentence what the input is for and gives one example drawn from the repository. Where an input has a sensible default, offer the default and ask only for confirmation.

1. **The notebook.** The example: "`sandbox.ipynb`".
2. **The change**, as one of four kinds: knob values (the example: "`line_4` at 30 MW in the August Example 1"); a new example in the sandbox (the example: "Market B also enforces `line_4`"); a code or markdown cell (the example: "print the redispatch ledger under the Example 2 figure"); a library change (the example: "move the October notebook's relief table into `atc.py`").
3. **Whether stored results are replaced.** The default is no, and the change runs on a scratch copy under `sandbox_exhibits/`. The example: "yes, re-run in place after the change".

When the set is complete, state it back in the form of the list above and proceed.

## The rules of a change here

1. **The deck notebooks follow the deck.** A change that adds or reorders an example is proposed against the slide sequence in `NOTEBOOK_RECORD.md` section 1, and a new example goes in the sandbox.
2. **Knobs live in `# -- EDIT --` cells.** An exploration sets knobs on a scratch copy and leaves the notebook's stored results alone.
3. **Edit by cell**, with `NotebookEdit` or `nbformat`, never by regenerating the file. A notebook is its own source of truth, and no builder script regenerates one.
4. **Markdown goes through `writing-voice`** in the notebook voice. The formulation opens with the `| Symbol | Code | Definition |` table, and every numbered equation carries its `*In words.*` blurb.
5. **Figures follow `figures.md`.** A deck notebook's figure carries no suptitle, no panel title and no legend, because the slide carries them.
6. **A change to the library is proposed in `dev_notes.md`'s open decisions** and arrives as a fresh copy of `lib/` with a regenerated `lib/MANIFEST.md`; a notebook-local helper carries it until then. The proposal carries its diff and runs only on a scratch branch. A local change to `lib/` is a divergence from `lib/MANIFEST.md`, reported as such.
7. **The methodology check** below runs after every code change.
8. **Re-execute through `run-example`**, so that the stored outputs carry the stamp.

## The series conventions

1. **Self-contained math.** A notebook numbers its own equations from (1), restates any derivation it needs, and never cites another notebook's or a paper's section or equation numbers, because those renumber. A notebook may name a sibling by its file name. A reader who opens a notebook cold follows every claim without opening another.
2. **The formulation opens with the nomenclature table and states the program whole**, as the next section sets out. Every numbered display equation is followed by an `*In words.*` blurb, the only labelled construct a notebook uses; a figure or a printed result is read in ordinary prose.
3. **Math is `$…$` and `$$…$$`**, because the VS Code preview renders only those.
4. **The canon governs the symbols.** Sets are script capitals with lower-case elements, the line index is $m$, the shift factor is $SF_{n,m}$ bus first, $\mu_m$ is signed with $+\sum SF\mu$, $N_a^c$ is the congestion by region in hub form and $N_a$ the net position, $R$ and $R_T$ are the full streams, the declared interface set is script $\mathcal I$, and $\mu_T^{ab}=\hat\lambda_b-\hat\lambda_a$ where $a\to b$ binds. The risk vocabulary uses $\gamma$ for risk aversion and $\psi$ for the ATC margin. The annotated tables, the allocation methods, the hub decomposition and the open fix-on-touch items are in `CANON.md`.
5. **Seams vocabulary.** "Accommodation" names the response miss and "forecast" the input miss. A market is an independent clearing engine, a balancing authority area is an area within one market, and an intra-market interface takes intertie, transfer and transfer settlement. The retired framings of `CANON.md` section 7 stay retired, and the seams mechanics (interchange as a fixed injection, the SFT over the union of both books) are `CANON.md` section 7.
6. **Prose is the notebook voice** of the `writing-voice` skill. Charts use direct encodings with raw values visible.

## The formulation

A notebook states the math it uses, in its own numbering, before it clears anything. `exemplars/notebook.md` in the writing-voice skill shows each piece in its tested form.

1. **The nomenclature table comes first**, `| Symbol | Code | Definition |`, mapping every symbol to the object's name in the library and to a definition in words. A sibling's table is the starting point, trimmed to the symbols used; notation is never invented for an existing object.
2. **The program is stated whole**, an objective subject to its constraint families, each family's dual in brackets, in the order of the fundamentals chain. (1) The program minimizes dispatch cost subject to one power balance per balancing authority area $[\hat\lambda_a]$ coupled by scheduled transfers, the activated line limits $[\mu_m]$, and each declared interface's released capacity $[\mu_T^{ab}]$. (2) The pooled energy balance $[\lambda]$ is the per-area balances summed, the case `solve_engine_dispatch` solves. (3) Flow and the activated limits, with the net injection defined. (4) Scheduled transfers and released capacity, with the net interchange. (5) The Lagrangian and stationarity, giving the marginal unit's price and the transfer condition $\mu_T^{ab}=\hat\lambda_b-\hat\lambda_a$ where a limit binds. (6) The nodal price in two forms, the solver's $\hat\lambda_a+\sum_m SF_{n,m}\mu_m$ and the aggregation-point form $\lambda_a+\sum_m(SF_{n,m}-SF^a_m)\mu_m$. (7) The rents $R=\sum_m|\mu_m F_m|=\sum_n\lambda_n d_n-\sum_n\lambda_n g_n$ and $R_T=\sum\mu_T^{ab}t_{ab}$. A notebook that needs part of the chain restates that part and renumbers from (1). A notebook that adds an object (a flowgate, a scheduling limit, a flow offset, a reservation) adds it to the program in the same shape, with its dual in brackets and its row in the table.
3. **The `*In words.*` blurb** reads the equation left to right in the equation's own order, names every quantity by its table name with the symbol inline, and says what a symbol is before using it. It may keep the "subject to three constraint families: A (dual X); B (dual Y)" shape, because that shape mirrors the equation. The blurb states what the equation says, and the interpretation lives in the surrounding prose.
4. **The formulation names the EDAM object it mirrors and the divergence from it**: the signed $\mu$ with $+\sum SF\mu$, the lossless $LPF=1$, and the one-interface-per-pair transfer layer with its own dual, all stated in `CANON.md` sections 1 to 3. The market's own statement is `caiso_edam_techdesc` Sections 3, 4, 5 and 10.

## The deck notebooks

A deck notebook has a fixed form: the title cell, which names the deck, its date and its excerpt under `sources/workshops/`; the nomenclature table; the formulation with numbered equations and `*In words.*` blurbs; the configuration with its `# -- EDIT --` knobs; then the examples in the deck's slide order. It carries little exposition and no editorializing. A deck notebook closes with a sandbox section, which reruns the notebook's chain on the values of its `# -- EDIT --` cell in a final code cell. The sandbox exposes the engine's configurable features, including a scheduling point and an interchange scheduling limit at a transition bus, and its `MODE` knob clears the same knobs as one market or as two independent markets. Its closing section lists the features it does not carry.

## The pattern of an example

1. **One constraint isolates one mechanism.** An example changes one thing against a case the reader already holds: the deck's own anchor, cross-checked against the canonical clearing (`line_4` at 40 MW on the teaching fleet, $R=\$804.1$/h, the signed $\mu_{\mathrm{line\_4}}=-\$20.10$/MWh, Example 1 of `sandbox.ipynb`). Where an example in the sandbox tests a claim, a markdown cell states the hypothesis and the reading rule before the run, and the result is read against that rule in one or two sentences after it. A result that surprises is reported as a finding.
2. **The first figure of every example is the interval's dispatch as the standard combined figure**, network panel and dispatch panel from the same result, drawn to the defaults in `figures.md`, with no suptitle, panel title or legend in a deck notebook.
3. **The settlement is printed under the figure**: `settlement_by_bus` and `capacity_value_by_line`, whose totals both equal $R$, and for an allocation example the folded ledger with its conservation row visible. A ledger with raw components beats a derived metric.
4. **The prose under the output reads the number and stops.**
5. **The knobs are visible** in `# -- EDIT --` cells. A deck notebook ends in the sandbox section of its fixed form, and `sandbox.ipynb` reruns the whole chain on whatever the reader sets.

## Reading and running

`NOTEBOOK_RECORD.md` is read before the notebook: section 1's entry for the notebook at hand, with its slide map and anchors, and section 2's entries for the modules it calls. A dated entry you add goes in that record, in the same manner, and `CLAUDE.md` changes only where a rule that holds in every session changes.

The files are large, so extract the cells with `nbformat` under `PYTHONIOENCODING=utf-8` and read three things together: the markdown, the stored outputs and the `# -- EDIT --` knobs. The stored outputs are the experiment and carry the stamp of their last execution in `metadata.lab_execution`.

`CLAUDE.md` names the environment, the headless run command and the test command. `MPLBACKEND=Agg` is never set, because it strips every stored figure. A notebook is re-executed with `tools/nb_execute.py`, not bare nbconvert, so that its stored results carry the library commit and the manifest, and `python tools/nb_staleness.py` is reported before and after.

### Where a notebook writes

An export path is resolved from the repository, never from the working directory, because the kernel's working directory depends on where it was started. The repository root is the folder that holds `lib/` and the notebooks. A public notebook writes no file by default (`seam_views.configure(save_enabled=False)`). A scratch export goes to `outputs/` or `sandbox_exhibits/`, which `.gitignore` excludes, because the notebook is the source and its stored outputs already carry the figures and tables. A new export destination needs a matching `.gitignore` rule.

## Ground rules for a change

- Restate the ask and the cells it touches before acting.
- Notebook code and any proposed library code are written to `CODE_STYLE.md`. A docstring of formulation code names the equation it implements, its `references.bib` key, its units and signs, and any modeling choice of this repository's own.
- Methodology belongs in the library so every notebook sees it. A change to the library is proposed in `dev_notes.md`'s open decisions and arrives as a fresh copy of `lib/` with a regenerated `lib/MANIFEST.md`; a notebook-local helper carries it until then.
- A claim follows a run. A dispatch, a dual, a rent or an allocation comes from a run in this session, with the command and its output shown.
- Before presenting, the numbers reconcile across cells: $\sum_a N_a$ is $R$ with no interface binding and $R+R_T$ with one, $\sum_a N_a^c+T_E=R+R_T$ closes, and congestion by region is read as $N_a^c$. A result that contradicts the framing or another cell is stated as such and dug into.
- `python -m pytest tests -q` runs before a change to `lib/` or a test is done. A change that moves a golden is a deliberate decision, stated as such.
- A change to a deck notebook is done when `python -m pytest tests -q -m slides` passes. The slide checks live in `tests/test_slides_<deck>.py`, never in the notebook, whose tables are the ledgers its reading needs (user decision 2026-11-08).

## The methodology check, after every code change

The series exists for model-to-model experiments, so a number is a result only if every step that produced it maps onto a published formulation or onto the methodology the codebase already states. An agent reaching for an outcome can take a shortcut that produces the number and breaks that mapping, such as the net position $L_a-G_a$ reported as the congestion by region $N_a^c$, or a rating widened so that the program solves. The check catches the next one. Before it, `CANON.md` is read for the conventions the diff touches, and `sources/excerpts/<key>.md` for each source whose construct the change implements; a sign, unit or reference mapping that the change alters is a stated difference in the formulation cell or a failure here. It runs on the diff of every change to a library, a script or a code cell, before the change is reported, and a formulation cell that states what the code does is read as part of the code. Answer three questions against the code as written, fix each failure in the code, and run the questions again until a pass finds nothing.

1. **Is the change consistent with the modeling choices the codebase states?** This is a question of method, and a change that reproduces every stored number can still fail it. The change stays inside the model's stated form (single-period, energy-only, bid-based, lossless DC with PTDF shift factors, lines or flowgates with signed duals, flow-inert transfers between area balances, HVDC as physical links, an engine's boundary as fixed injections placed by ex-ante hub weights), or it names itself as a methodology change with the siblings it reaches. It uses the library's objects for the library's concepts rather than re-implementing one. It keeps the case's stated inputs (the teaching fleet, the loads, the ratings and the `SHED_PRICE` of 150), or states and prints the different value. It keeps the identities the model asserts, restated with any new term. It keeps the canon's names, index order and signs. It is stated where such choices are stated: a docstring, the formulation cell and table, and `AGENTS.md` or `CANON.md` where other notebooks inherit it. The anchors (the deck numbers each notebook reproduces, checked by `tests/test_slides_<deck>.py` and listed in its `NOTEBOOK_RECORD.md` section 1 entry, with the canonical clearing's $R=804.1$ at `line_4`$=40$ as the cross-check) and the suites are diagnostics: an anchor that moves is explained by the stated choice, with the stored results it supersedes named, and an anchor that holds through a methodology change is the finding.
2. **Does the change map onto a reasonable reading of the sources?** The source whose construct it implements is named by `references.bib` key, with the construct, in one sentence per construct. Every deviation is named with its reason. The source is read as it characterizes itself. Where no source can be named, the change is a modeling choice of this repository's own, stated as such in the cell and the report and flagged for the participant.
3. **Was a shortcut taken?** Confirm each absent: a number typed in where a run should produce it; a stored output edited or a figure drawn from data the notebook did not compute; a constraint dropped, relaxed or screened out so the program solves; a default changed silently rather than pinned and printed; a test weakened or a special case for the configuration under test; a term dropped from an identity or a sign flipped; the formulation's object replaced by a proxy that coincides on the case at hand; the wrong object substituted because it is at hand ($L_a-G_a$ for $N_a^c$, $\lambda_a$ written as $\hat\lambda_a$, a scheduled injection summed into the physical flow where the actual dispatch is meant).

The report closes with a three-line record, one line per question, each ending in its evidence, written from the code after the last fix.

The deck anchors are the numbers on the slides. In the August notebook they are Market A's $2,535, bus 7 at $37.10, the line 6 to 7 dual of $20.10/MWh, Market B's 9.4 MW, the redispatch at $2,725, the over-forecast 11.5 MW at $2,767 and the $42 of phantom congestion. The August Market A clearing binds `line_4` (bus 6 to bus 7) at 40 MW with the same marginal pair as the canonical clearing, and the first run of the August notebook checks that its rent is the canonical $804.1/h.

<example>
1. Method: passes the sandbox knob `SCHED_POINT["limit_mw"]` to the engine's own `schedule_limits=` row of `solve_engine_transfers`, stated in the sandbox's formulation cell and table with its dual $\mu^s$ outside $R$; the suite passes; the canonical anchor holds at $R=804.1$ because `SCHED_POINT` defaults to `None`.
2. Sources: `caiso_dame_edam2023` p. 178 for the $-1$ coefficient on an export at its scheduling point; deviation stated: the row is gross and one-sided where the source's limit is net and two-sided.
3. Shortcuts: none; the limit is read from the knob, no bound was widened, the default is printed with the configuration.
</example>

For a change to the library, a formulation cell or any function that produces a number a notebook cites, a fresh-context reviewer runs the three questions on the diff, the `AGENTS.md` and `CANON.md` sections the change touches, and `sources/excerpts/<key>.md` and the bib `note` of the sources named under question 2, told to report only what affects the mapping or the identities. The reviewer's record is the one reported. The fresh-context reviewer is the `methodology-checker` agent, and `methodology-checker-light` takes a small diff to scaffolding. A change to scaffolding alone takes your own check.

**The stop rule.** The check runs again after each fix until a pass finds nothing. A loop of three rounds with an open fail or should-fix is escalated to the participant, who resumes it, holds the finding or changes the task. The loop is never ended by reaching the cap, and a verdict of `Revise` or `Block` is never reported as done. Where the saved workflow `build-and-check` (`.claude/workflows/build-and-check.js`) is available, it runs the build, a conventions review and a fresh check each round under this rule, and commits nothing.

## Your first task

Restate the ask and the cells it touches. Read the target notebook's markdown, stored outputs and knobs, and the library functions it calls. Report, leading with the numbers that bear on the ask: the notebook's structure as the cells show it, with the stamp its outputs carry; anything stale, inconsistent or surprising, cited to the cell; the proposed change at cell level with its knock-on cells (figures, tables, the goldens); the figures the change will produce, each as the standard letter plus stated additions or as an approved variant; and the modeling choice the change makes, the source construct it maps onto by key, and which stored numbers it is expected to move and to hold. Where the ask is exploration, run the cells on a scratch copy or a minimal standalone on the engine, show the numbers, and read them against the deck. Implement a change after the participant confirms.

## Sub-agent models

Sub-agents run on the cheapest model tier that fits their task, the rule `CLAUDE.md` states (user decision 2026-10-18); the Agent and Workflow tools inherit the session's model unless `model` is passed, so every brief names it. In this skill the fresh methodology check of a cell or scaffolding diff runs on `sonnet` (`methodology-checker-light`); the check of a formulation cell or a library diff runs on `opus` (`methodology-checker`), and so does a notebook rewrite or a new notebook (`notebook-builder`); the voice check of markdown runs on `sonnet` (`voice-checker`); a headless re-execution and its report run on `sonnet` (`notebook-reexecutor`); a read-only review of a diff against `CANON.md`, `figures.md` and `CODE_STYLE.md` runs on `sonnet` (`conventions-reviewer`); an adversarial check of one stated claim runs on `opus` (`refuter`); a test run or a cell extraction alone runs on `haiku` (`test-runner`, `cell-extractor`). A role this directory does not define takes an explicit `model`, and judging roles report on the scale of `methodology-checker.md`, section "Severity and verdict". The session's top-tier model is not used for a tightly scoped sub-agent task.
