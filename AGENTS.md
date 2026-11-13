# AGENTS.md

This file orients any agent working in this repository, whichever tool or model runs it. A person can start a session with "Read AGENTS.md, then help me with …". Claude Code reads it through `CLAUDE.md`, which adds the Claude-specific layer.

## Purpose and scope

- **Purpose.** Give people who work on electricity markets runnable notebooks behind the examples in public presentation materials on Western market seams.
- **Scope.** Limited to the 9-bus examples of the CAISO-SPP joint market seams workshops of 2026 and the clearing, settlement and seams mechanics those examples rest on.
- **Objective.** Reproduce each example from its stated formulation, so that a reader can check every number a slide prints and change the inputs that produced it.
- **Activity.** Build on the published market rules, the economics of locational pricing and the literature on coordination between markets, each cited to a public source.
- **Deliverable.** Produce educational, informational and directional insights from an illustrative network, with each example reproducible and each number cited to the cell that prints it.

## What this repository holds

The network is the WSCC/IEEE 9-bus test case (`anderson1977power`, `thurner2018pandapower`), cleared as a lossless DC dispatch with shift factors, with one market or two markets on one network. Each workshop deck has one notebook at the root, and each deck notebook follows its deck's slide sequence: `workshop2_congestion_management.ipynb` (August 18, 2026), `workshop3_transactions_across_boundaries.ipynb` (September 16, 2026) and `workshop4_loading_relief.ipynb` (October 22, 2026). `101_fundamentals.ipynb` states the market's clearing at its simplest and clears it as one market, as two balancing authority areas within it, and in its closing sandbox as two markets on one network. `sandbox.ipynb` exposes the engine's configurable features in one EDIT cell, and its closing section lists the features it does not carry. `README.md` holds the workshop index, with each deck's excerpt and Colab badge.

A deck notebook has a fixed form: the title cell, the nomenclature table, the formulation with numbered equations and `*In words.*` blurbs, the configuration with its `# -- EDIT --` knobs, then the examples in the deck's slide order. It closes with a sandbox section that reruns its chain on the values of its EDIT cell. A figure in a deck notebook carries no suptitle, no panel title and no legend, because the slide carries them, and it is drawn inline at the default inline dpi (user decision 2026-11-07). A deck notebook writes no file by default. The checks that its clears reproduce the slides' numbers live outside it, in `tests/test_slides_<deck>.py` through `tests/slide_check.py` (user decision 2026-11-08). A new example goes in the sandbox, so that each deck notebook stays on its deck.

`lib/` is a byte copy of a shared 9-bus library maintained outside this repository. `lib/MANIFEST.md` records each module's SHA-256 and the date of the copy. A change to the library is proposed in the open decisions of `dev_notes.md` and arrives as a fresh copy, and a notebook-local helper carries the change until then. A local edit to `lib/` reads as a divergence from the manifest. A docstring in `lib/` keeps its inherited text, including the names of modules and notebooks this repository does not carry.

## How to use this repository

An agent working here helps a person find and understand what the repository holds, and bring a question of their own.

1. Restate the question in one sentence, and name any term the record defines differently, with the record's term.
2. Look in the record, in this order.
   - **The notebooks.** Each root notebook states its question in its title cell, and its findings are in its stored outputs. `NOTEBOOK_RECORD.md` section 1 holds each notebook's entry and slide map.
   - **The decks.** `sources/workshops/` holds the deck excerpts, named by date, and `sources/workshops/manifest.json` gives each posted deck's URL, page range and SHA-256.
   - **The sources.** Each entry of `sources/references.bib` carries a `url` and a `note` on its bearing, and a `file` where the document is tracked. Read `sources/excerpts/<key>.md` first, then the PDF where tracked, otherwise the url.
   - **The library.** `NOTEBOOK_RECORD.md` section 2 indexes the public functions module by module, and the module docstring of `lib/seams_engine.py` maps the code to the notation.
3. Answer from what was found, citing the file and the cell, page or section. A number comes from a stored output with its provenance, never from memory. The provenance is the notebook's `lab_execution` metadata stamp or, for a notebook not yet stamped, its row in `tools/nb_staleness.py`.
4. If the record does not address the question, say so, show the nearest material, and sketch which knobs of `sandbox.ipynb` could answer it. Nothing is built until the person asks.
5. The open decisions are listed in the newest `dev_notes.md` entry that restates them, by `D-YYYYMMDD-n` id. They are read when a question touches them.

## The Western context

The California ISO operates the Western Energy Imbalance Market (WEIM) in real time and the Extended Day-Ahead Market (EDAM) a day ahead. Balancing authorities that join the Western Energy Imbalance Market or the Extended Day-Ahead Market confer commitment, dispatch, and pricing functions to the market, and maintain responsibility for investment, procurement, and the recovery of their costs (`caiso_dame_edam2023`, pp. 4, 5, 37-38, 132, 144). EDAM is "a voluntary regional day-ahead market" (`caiso_dame_edam2023`, p. 12).

| Feature | Statement | Sources |
|---|---|---|
| Incremental steps by willing participants into centralized markets | A participating balancing authority can elect to confer commitment, dispatch, and pricing responsibilities to the market incrementally and retains its remaining functional responsibility including reliability and transmission service tariff. | `caiso_dame_edam2023`, pp. 4, 12, 37; `pgp2021retrospective`, p. 36 |
| Institutional diversity across the West | Participants differ in business model, commercial arrangements, customer relationships, and regulatory regime, so the market's rules must work alongside institutions with diversity of policies, goals, and objectives. | `pgp2021retrospective`, pp. 11, 20; `kiesling2026rto`, p. 1 |
| Decentralized stewardship of the grid and its resources | Provision (i.e., the supply and maintenance of the shared resource) of the grid and its resources stays with a diverse set of institutions, who own, maintain, and invest in transmission and generation, procure adequate resources under their own authority, and recover the cost of those decisions through their own contracts, rates, and tariffs. | `caiso_dame_edam2023`, pp. 5, 37-38, 132, 144; `ostrom1990governing`, ch. 2 |
| Loosely meshed network with long-distance transmission paths | Long transmission paths join the West's load centers to generation across a loosely meshed grid, and power flow from many network users concentrates on the high-voltage, low-impedance paths that a few transmission providers operate. | `nerc_oscillation2019`, pp. xii-xiii; `pgp2021retrospective`, p. 11 |
| A shared network that is non-excludable and rivalrous | The western grid is a common-pool resource on which the injection and withdrawal of power contributes flow to interconnected facilities regardless of ownership or jurisdictional boundaries, so a portion of the network cannot exclude external flow contribution without isolating itself from the network, and when a constraint binds one participant's use displaces another's. | `ostrom2009nobellecture`, p. 413; `kiesling2026rto`, p. 4 |

The Western Interconnection "historically has been characterized by long transmission lines that carry power from generation rich regions" to load centers in California, while the Eastern Interconnection "has a mesh-like transmission line structure" (`nerc_oscillation2019`, pp. xii-xiii). The arrangement is polycentric, with balancing authorities, transmission providers, the states that regulate them and the markets themselves as formally independent centers of decision making on one network. Elinor Ostrom, in her Nobel lecture, writes that "'Polycentric' connotes many centers of decision making that are formally independent of each other" (`ostrom2009nobellecture`, p. 411).

The functions conferred to the market (commitment, dispatch, and pricing) tend to be short-run appropriation functions. The responsibilities maintained outside the market (investment, procurement, and the recovery of costs) tend to be long-run provision functions (`ostrom1990governing`, ch. 2). Ostrom's design principle 2B reads "Appropriation rules are congruent with provision rules; the distribution of costs is proportional to the distribution of benefits" (`ostrom2009nobellecture`, p. 422). The Public Generating Pool's retrospective of the organized-market initiatives in the West finds that all of them struggled "over issues of transmission cost allocation, governance, and cost relative to benefits" (`pgp2021retrospective`, p. 8). The examples here are short-run: each clears one interval and settles it, and the provision functions stay outside the model.

## How the model runs

**One engine, one clearing.** `lib/seams_engine.py` clears a single-period, energy-only, bid-based security-constrained economic dispatch as a linear program on a DC network, in which shift factors map each bus's injection to each line's flow. In the EDAM-literal form (`solve_engine_transfers`), each balancing authority area has its own power balance with dual $\hat\lambda_a$, each activated line limit has the signed dual $\mu_m$, and each declared interface's released capacity has the dual $\mu_T^{ab}$. The congestion rent is $R=\sum_m|\mu_m||F_m|$, what load pays less what generation is paid, so no binding line means no price separation and a rent of zero.

**Markets and areas.** A market is an independent clearing engine and its footprint, and a balancing authority (BA) is an area within a market. A market is never called a balancing area. The BA partition is an accounting overlay on the same physics. Intra-market, one engine prices the whole flow on each activated constraint. Inter-market, two independent clearings on the shared grid have disjoint activated sets, each manager enforces only its own flow component, and neither prices the superposition. A text states which engine enforces which constraint, whose flow component it prices, and what the superposition leaves unpriced. The counterfactual is independent operation, not autarky.

**Settlement.** `lib/revenue_allocation.py` settles each clearing into ledgers: what load pays, what generation is paid, and how the rent is divided among areas. An allocation rule is a design option to be tested, not a fixed output of the model.

**Bounds.** The model is illustrative, not a forecast. Offers are cost-based, each clearing is a single period, and the network is lossless.

## Dated assumptions

| assumption | value | where the code sets it | dated or held |
|---|---|---|---|
| shed price $V$ | \$150/MWh | `SHED_PRICE` in the configuration cell of each root notebook | a notebook that adopts another value states it (`CANON.md` section 7) |
| figure label rounding | two paths: a flow rounded to 0.1 MW and then to a whole MW, or directly to a whole MW | `combined_views` and `own_view` in `lib/seam_views.py` | held for the participant, D-20261107-7 |
| slide tolerances | 0.5 MW on a whole-MW label, \$0.05/MWh on a price printed to \$0.1, and a floating-point slack of 1e-9 | `WHOLE_MW_ABS`, `TENTH_PRICE_ABS` and `ROW_SLACK_ABS` in `tests/slide_check.py` | 2026-11-08 |
| Method 1's split of the rent on unassigned lines | half each, $s_a=0.5$ | `unassigned_split` of `allocate_congestion_rent` in `lib/revenue_allocation.py`; `UNASSIGNED_SPLIT` in `sandbox.ipynb` | 2026-11-06 (`NOTEBOOK_RECORD.md` section 4) |
| Method 2's transfer-settlement share | half each, $\sigma_a=0.5$ | `TE_SPLIT` in `sandbox.ipynb` | `CANON.md` section 4 |

## Notation

The series shares one notation, held in `CANON.md`. Read `CANON.md` before writing or reading math, a symbol table or a ledger. Five conventions recur everywhere.

1. The line index is $m$ and the area index is $a$; a set is a script capital, and an upper-case $M$ names a market engine.
2. The shift factor is $SF_{n,m}$, bus first.
3. The line dual $\mu_m$ is signed, and the nodal price adds $+\sum_m SF_{n,m}\mu_m$, the opposite sign from the EDAM technical description.
4. A market is an independent clearing engine and its footprint, and a balancing authority is an area within a market; the two terms are never interchanged.
5. Notebook math is written `$…$` and `$$…$$`, and every numbered display equation is followed by an `*In words.*` reading.

## Evidence

- A claim follows a run in the session or cites the stored cell it comes from.
- A source is quoted verbatim at its pinpoint.
- Public documents of the ISOs, RTOs, FERC, NERC, NAESB and the workshop hosts are tracked under `sources/`. Academic papers and books are link-only, cited by key, page and url, with at most a short quotation, and their local copies stay under the ignored `sources/academic-literature/`.
- The September 16 deck and the workshop agenda deck are link-only, because the posted files carry internal markings. They are cited by slide and url, and no further passage of either is copied into the repository.

## Working rules

- **Robustness lives in code**, in the library or the notebook, never in hand-edited result files.
- **Notebooks are edited by cell** (with `nbformat` or an editor's cell tools) and are their own source of truth. No builder script regenerates a notebook.
- **The methodology check** (user instruction 2026-09-12). Every code change is checked from its diff for consistency of method with the codebase's modeling choices and for mappability to the sources, then audited for shortcuts, and each failure is fixed until the check passes. The check reads `CANON.md` and `sources/excerpts/<key>.md` for the conventions and sources the change touches, and `.claude/skills/modify-example/SKILL.md` holds the full statement. A number reached by a shortcut that breaks the mapping is not a result.
- **Code** follows `CODE_STYLE.md`, linted by `ruff.toml` and type-checked through `tools/run_mypy.py` under `mypy.ini`.
- **Prose** follows the notebook and chat voices in `.claude/skills/writing-voice/`, whose register files and exemplars any agent can read.
- **Replies in the chat** are written in the TD voice, the register of the California ISO's *Extended Day-Ahead Market Draft Technical Description*, which `register/td.md` and `exemplars/td.md` in `.claude/skills/writing-voice/` describe and quote (owner decision 2026-11-12). A reply opens with the answer, names each term in full on every mention, gives a design choice its reason in the same sentence, and states a number with its unit and provenance. A contrast stays only where both of its sides are defined and it carries its own claim, so "it is not X, it is Y" and its two-sentence form are cut (`register/cut_on_sight.md`). Math in a reply is written in Unicode characters (λ̂_a, λ_n, μ_m, SF_{n,m}, Σ_m SF_{n,m} μ_m, θ₁ − θ₂, ≤, ≥, ×) and never in LaTeX, because the chat does not render it, and a symbol follows the plain name of its quantity, set off in commas ("the shadow price of line 6-7, μ_m").
- **The stop rule.** A methodology check runs again after each fix until a pass finds nothing, and a loop of three rounds with an open fail or should-fix goes to the participant; reaching the cap never ends it (`.claude/skills/modify-example/SKILL.md`).
- **Items held for the participant** are decided by the participant, never by the agent.
- **Record entries.** A `dev_notes.md` entry is written in the five parts the file's header names, and each held item carries a `D-YYYYMMDD-n` id. The form binds the maintainer and is recommended to contributors.
- **Git.** Nothing is committed or pushed unless the participant asks. Every path is staged by name, and `git add` with `-A`, `--all` or `.` is never run. No commit carries an AI authorship trailer. Nothing is opened or posted on an upstream repository.
- **Processes.** A Python process is ended by its PID, never by its image name, because a kill by image name ends the participant's kernels and other runs. In Claude Code the guard refuses the kill by name (`CLAUDE.md`, section 5).

## Running and verifying

- **Python.** Python 3.10 or later, Colab-compatible. `requirements.txt` names the packages the notebooks and `lib/` import, unpinned for Colab. `requirements-dev.txt` adds the harness packages, and `constraints.txt` pins every package of the tested environment: `pip install -r requirements-dev.txt -c constraints.txt`. `tools/env_lock.json` records the versions of that environment, and a run under other versions is warned and never refused. Every command runs from the repository root.
- **The interpreter.** `LAB_PY`, where it is set, names the interpreter that `tools/doctor.py` and `tools/check.py` examine, and otherwise each uses the interpreter that runs it.
- **Commit hooks.** `git config core.hooksPath .githooks`, run once after cloning, activates the commit hooks of `.githooks/`. `commit-msg` refuses an AI authorship trailer. `pre-commit` refuses a staged file over 50 MB, a path under `sources/academic-literature/` and a `sources/` path that an ignore rule lists.
- **Readiness.** `python tools/doctor.py` reports the interpreter, each package against its pin, the notebook kernels, each `lib/` module and copied hook against its manifest, the data files the tests need, the process count and `core.hooksPath`. It changes nothing.
- **Verification.** `python tools/check.py` runs the lint of the changed Python files, the harness tests and the model tests, and `python tools/check.py --full` adds the slide checks, which execute each deck notebook. A change to a deck notebook is done when `--full` passes. A fresh copy of `lib/` is done when `python tools/check.py --status --full` shows a passing line for each tier. The single suites stay available: `python -m pytest tests -q`, and the slide checks alone `python -m pytest tests -q -m slides`.
- **Notebooks.** A headless re-run is `python tools/nb_execute.py <notebook>.ipynb`, which compiles every code cell first, refuses the launch on a syntax error, and stamps the library state into the notebook's metadata. `python tools/nb_staleness.py` lists the notebooks whose stored results predate a library change. `MPLBACKEND=Agg` is never set, because it strips every stored figure.
- **Lint and types.** `ruff check --config ruff.toml <files>` and `python tools/run_mypy.py [<module>...]`.
- **The process ceiling.** A pytest, nbconvert or `nb_execute.py` launch is held while the count of Python interpreter processes is at the ceiling, `CEILING` in `.claude/hooks/_limits.py`: seven, or the positive integer that `LAB_PROCESS_CEILING` sets. A launch sent to the background needs the count `BACKGROUND_MARGIN` below the ceiling, because the next count does not see it. `tools/check.py` and `tools/nb_execute.py` hold to the same ceiling.
- **Colab.** Each notebook's first code cell installs `pypsa`, `pandapower` and `pycirclize` and fetches the 14 modules from this repository's `lib/` on `main`, so a library change reaches Colab only when it is pushed.
- **Exports.** An export path is resolved from the repository, never from the working directory. Scratch exports go to `outputs/` or `sandbox_exhibits/`, both ignored.

## Where things live

| path | holds |
|---|---|
| `*.ipynb` at the root | the three deck notebooks and the sandbox |
| `lib/` | the 14 library modules and `MANIFEST.md` |
| `tests/` | the suite, the slide checks and the frozen goldens; `tests/conftest.py` puts `lib/` on the path |
| `tools/` | the stamped execution wrapper, the staleness report, the readiness report, the verify command, the environment check and the type check; `tools/README.md` describes each |
| `sources/` | `references.bib`, the deck excerpts under `workshops/`, the page-pinned passages under `excerpts/`, and the public documents the notebooks cite |
| `notes/writing_profile/` | the measured register profile the writing-voice rules cite |
| `prompts/` | starters for people; `prompts/01-orientation.md` is the page for a person joining |
| `.claude/` | the Claude Code layer: skills, agent roles, hooks, the saved workflow and `settings.json` |
| `.githooks/` | the commit hooks, activated by `git config core.hooksPath .githooks` |
| `requirements.txt`, `requirements-dev.txt`, `constraints.txt` | the notebook packages, the harness packages, and the pins of the tested environment |
| `CANON.md` | the notation |
| `NOTEBOOK_RECORD.md` | the notebooks and their slide maps, the library's function index, the publishing steps, the canon's dated deviations |
| `dev_notes.md` | the dated development log, newest first, with the open decisions |
| `CODE_STYLE.md`, `ruff.toml`, `mypy.ini` | the code style guide, its lint and its type check |
| `LICENSE` | the MIT licence for the code; `sources/` stays under its publishers' terms |

## Open decisions

The decisions held for the participant carry `D-YYYYMMDD-n` ids in `dev_notes.md`, and a decision stays open until a later line closes it. The newest entry that restates them is the current list.
