# Code style guide

<!-- voice: td. The mentor pins are in section 2; the counts of this repository are a measurement of 2026-11-07. -->

## 1. Purpose and scope

This guide states the standard to which code in this repository is written, by a participant or by a participant's agent. The scope is this repository alone: the code cells of the root notebooks, `tests/`, `tools/` and the scripts under `.claude/`. `lib/` is a byte copy of a shared 9-bus library, and `lib/MANIFEST.md` records each module's SHA-256 and the date of the copy. `lib/` is not edited here (rule 7.8). Its modules are the exemplars of the domain rules and are measured below, and a gap the measurement shows in `lib/` is closed in a later copy. Existing code in `tools/`, `tests/` and `.claude/` is brought to the standard on touch and never in a mass reformat: the lint, type-hint and docstring rules apply in the function or cell being changed.

Notebook markdown and the nomenclature canon are governed by the `writing-voice` skill, `AGENTS.md` and `CANON.md`, and this guide points to them instead of restating them.

A general rule is stated in four parts.

1. The rule, in one sentence.
2. **Mentors.** The mentor practice the rule comes from, with a pinpoint.
3. **Choice.** The practice chosen where the mentors differ, with its reason.
4. **Here.** The state of this repository, with the gap.

A domain rule is stated as this repository's own. A pinpoint names the project, the commit of section 2 and `file:line`. A line number in this repository moves when a file is edited, so an exemplar here also names its object.

## 2. Where the rules come from

The general rules of this guide come from mentor codebases. The domain rules come from the canon of `CANON.md`. A mentor's modeling convention is never imported with its code style. PyPSA reports a line's shadow prices as two non-negative duals, `mu_lower` and `mu_upper` (PyPSA `pypsa/data/component_attrs/lines.csv:42-43`), and EGRET solves in per unit and unscales its results before it returns them (EGRET `egret/models/dcopf.py:602`). Code in this repository keeps the canon's signed line dual, $\mu_m$, and the canon's units, because the canon's sign and unit conventions are load-bearing in every identity the tests assert.

The mentors were read on 2026-11-02 from clones at the commits below.

| mentor | role | repository and commit | licence |
|---|---|---|---|
| PyPSA | primary | `PyPSA/PyPSA` `d21e028d3198490b6d5d310ccb199a76393775a2` (master, 2026-11-02) | MIT for code and the component attribute tables (`REUSE.toml`); CC-BY-4.0 for the documentation pages |
| GridPath | primary | `sylvan-energy/gridpath` `508c00f5af93dfb41cac39594f65f85e9a167827` (develop, 2026-10-28) | Apache-2.0 |
| linopy | secondary | `PyPSA/linopy` `2ace7b39e30a77aaebaca8bbc17f5b989a0ab121` (master, 2026-11-02) | MIT |
| EGRET | secondary | `grid-parity-exchange/Egret` `28c324f9bf4fb3913b0a60dba3dd732b46084c3b` (main, 2026-05-29) | Revised BSD, Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (`LICENSE.txt`) |

Short excerpts below are quoted from PyPSA and linopy under MIT and from GridPath under Apache-2.0, with the project named. EGRET is paraphrased and not quoted, because its licence conditions the redistribution of source on keeping its notice.

PyPSA and GridPath are the primary mentors because both are open power-system optimization models with long maintenance records, and between them they cover two modeling layers: PyPSA assembles its programs with linopy, and GridPath assembles its programs with Pyomo. linopy and EGRET are read where they add a practice the primary mentors lack. Where the mentors differ, this guide follows PyPSA's practice unless a stated reason favours another mentor, because PyPSA is the only mentor with a complete lint, type-checking and test configuration.

The measured practice of the four mentors over their package modules without tests, and of the 14 modules of `lib/`, is:

| measure | PyPSA | GridPath | EGRET | linopy | `lib/` |
|---|---|---|---|---|---|
| modules | 88 | 413 | 71 | 46 | 14 |
| functions | 1,214 | 2,966 | 1,088 | 1,696 | 405 |
| functions with a return annotation | 100% | 0% | 2% | 100% | 100% |
| functions with a docstring | 85% | 77% | 28% | 61% | 79% |
| docstrings with numpydoc sections | 488 | 0 | 77 | 144 | 20 |
| docstrings with Sphinx `:param:` fields | 0 | 1,383 | 0 | 0 | 0 |
| modules that call `logging.getLogger` | 41 | 0 | 10 | 10 | 1 |
| `print` calls | 4 | 103 | 71 | 4 | 79 |
| `warnings.warn` calls | 53 | 37 | 9 | 11 | 2 |
| modules with `from __future__ import annotations` | 62 | 0 | 3 | 40 | 14 |
| formatter | ruff format | Black | none | ruff format | ruff format on 2 of 14 |

The measurement is a standard-library `ast` walk.

## 3. Layout and formatting

3.1. A new module is formatted by `ruff format` at 88 characters with double quotes from its creation, and so is every module off the `[format]` exclude list of `ruff.toml`.
- Mentors. PyPSA runs `ruff-format` in pre-commit (PyPSA `.pre-commit-config.yaml:20-28`), linopy does the same (linopy `.pre-commit-config.yaml:12-16`), and GridPath runs Black in continuous integration (GridPath `.github/workflows/black.yml:19-20`, `pyproject.toml:208-211`). EGRET runs no formatter. All three formatters default to 88 characters.
- Choice. The mentors agree on a formatter. A module on the exclude list keeps its layout until it is formatted in a change of its own (rule 14.6).
- Here. On 2026-11-07, 12 of the 14 modules of `lib/` and 17 modules of `tests/` sit on the exclude list. Off the list, `ruff format --check` reports 4 modules of `tools/` and `.claude/` that the formatter would change.

3.2. A changed line in a module on the exclude list is held to 88 characters where the statement allows, without reflowing its neighbours.
- Mentors. PyPSA and linopy ignore `E501` and leave line length to the formatter (PyPSA `pyproject.toml:237`, linopy `pyproject.toml:198`).

3.3. Imports sit at the top of the module in three isort groups (standard library, third party, first party), and a module imports a library module by its flat name.
- Mentors. PyPSA and linopy select isort (`I`) and ban relative imports (PyPSA `pyproject.toml:206` and `293-295`, linopy `pyproject.toml:190` and `232-235`).
- Choice. The flat import (`import seams_engine as se`) is this repository's layout, because `lib/` is placed on `sys.path` by `tests/conftest.py` and by each notebook's setup cell. The `src` list of `ruff.toml` makes isort read these names as first party. A module does not modify `sys.path` outside those two places, and a new module's name does not shadow a standard-library module (`A005`).
- Here. `I001` reports 23 findings in `lib/`.

3.4. A module begins with `from __future__ import annotations`.
- Mentors. PyPSA (62 of 88 modules, for example PyPSA `pypsa/optimization/constraints.py:7`) and linopy (40 of 46).
- Here. 14 of 14 modules of `lib/`.

3.5. A long module is divided by the banner comment `# ───` above and below a section title. GridPath divides a module by banner comments (GridPath `gridpath/project/operations/operational_types/gen_hydro.py:531-532`). Exemplar: `lib/seams_engine.py:83-85`, the banner above `PTDFData`.

3.6. A hand-aligned table or matrix, whose columns carry its meaning, is kept by a `# fmt: off` and `# fmt: on` pair, with the reason in the comment above it, and a single statement never takes the pair. A comment the formatter leaves on a closing line moves to its own line above the statement. Prose-width formatting of anything else is left to the formatter.

3.7. A module created here opens with `# SPDX-FileCopyrightText: Market Design Lab contributors` and `# SPDX-License-Identifier: MIT`. The root `LICENSE` is the MIT licence, scoped to the code. A file headed `SPDX-FileCopyrightText: ASSUME Developers` keeps that notice, because the MIT licence requires it.
- Mentors. All four keep a licence file at the root (PyPSA `LICENSE`, GridPath `LICENSE.md`, EGRET and linopy `LICENSE.txt`), and PyPSA heads each module with these two lines (PyPSA `pypsa/optimization/constraints.py:1-3`).
- Here. 13 modules of `lib/` carry the ASSUME notice, and `lib/lab_types.py` carries the notice of this rule.

## 4. Names and the public interface

4.1. Functions and variables are `snake_case`, classes `CapWords`, and module constants `UPPER_CASE`, and a leading underscore marks a name internal to its module.
- Mentors. PyPSA and linopy follow PEP 8 names. PyPSA marks internal modules and helpers with a leading underscore (PyPSA `pypsa/components/_types/`, `pypsa/common.py:715` `_convert_to_series`).
- Choice. A canon symbol keeps its case in code (`F`, `SF`, `R_T`, `A_ub`), because the canon's case is load-bearing. No mentor selects the pep8-naming rules (`N`), so the lint does not conflict.
- Here. 102 of the 405 functions of `lib/` carry a leading underscore.

4.2. Network and fleet attributes take the PyPSA names the code already uses (`p_nom`, `s_nom`, `bus0`, `bus1`, `x`), and buses, lines and areas are keyed by their string names (`"7"`, `"line_4"`, `"BA-1"`).
- Mentors. PyPSA declares each attribute with its type, unit, default and description in a table per component (PyPSA `pypsa/data/component_attrs/lines.csv:1-11`).
- Choice. A unit is stated in the docstring or field comment (rule 6.4) and not in the name. GridPath writes the unit into the name instead (`GenHydro_Gross_Power_MW`, GridPath `gridpath/project/operations/operational_types/gen_hydro.py:458`), which the canon's fixed names do not allow.

4.3. A module that other modules or notebooks import names its public interface in `__all__`.
- Mentors. PyPSA (10 modules, PyPSA `pypsa/__init__.py:54`) and linopy (8 modules). GridPath and EGRET declare none.
- Choice. PyPSA's practice, because a notebook bootstrap fetches modules by name and an explicit interface states what a notebook may call.
- Here. 2 of the 14 modules of `lib/`, `etag.py` and `seam_views.py`. `NOTEBOOK_RECORD.md` section 2 lists the public functions of every module.

4.4. A public name or argument that is renamed keeps its old spelling for one recorded period, through a wrapper that issues `DeprecationWarning` naming the new spelling.
- Mentors. PyPSA's `deprecated_kwargs` decorator (PyPSA `pypsa/common.py:513-552`) and its `@deprecated` decorator with "Deprecated since" in the docstring (PyPSA `pypsa/common.py:894-917`).
- Choice. The period is stated in `dev_notes.md` and not as a version, because this repository has no releases.
- Here. No deprecation is in use. A rename that arrives with a copy of `lib/` breaks the notebook cells that call the old name, so the notebooks are re-executed with every copy (`NOTEBOOK_RECORD.md` section 3).

## 5. Type hints

5.1. A new or changed function annotates every argument and its return, and a mapping's keys, shape and unit are stated in the docstring as well.
- Mentors. PyPSA annotates the return of every function (1,214 of 1,214), selects `ANN` (PyPSA `pyproject.toml:211`) and runs mypy with `disallow_untyped_defs` on its package (PyPSA `pyproject.toml:170-183`). linopy annotates every return and runs mypy with the same setting (linopy `pyproject.toml:168-175`). GridPath annotates none (0 of 2,966), and EGRET almost none.
- Choice. PyPSA's practice, because an annotation is checked by a tool while a docstring is not, and because the two mentors that annotate are the two whose configurations this guide follows. `Any` is permitted (`ANN401` is ignored, as in PyPSA).
- Here. Every function of `lib/` annotates its return, and `ANN` reports no finding there.

5.2. mypy checks `lib/`, `tools/`, `.claude/hooks/` and the writing-voice scripts, and the tests are out of its scope.
- Mentors. `mypy.ini` carries the settings of PyPSA (`pyproject.toml:170-183`) and linopy (`pyproject.toml:168-175`), with one section per package that ships no type information in place of PyPSA's global `ignore_missing_imports`.
- Here. `python tools/run_mypy.py` checks the whole scope, and `python tools/run_mypy.py <module>` checks one module and prints its findings alone, both from the repository root. On 2026-11-07 the whole scope, 25 source files, carried no finding.

5.3. A hint pass changes no behaviour, and a defect mypy reveals is reported, carries `# type: ignore[code]  # reason` on its line, and is fixed in a later change. The pragma stands first in the comment, because mypy reads it only there. Exemplar: `lib/atc.py:2333`, in `settle_rights`.

5.4. The shared type aliases live in `lib/lab_types.py`, whose docstring states how to choose between an alias and a concrete pandas or numpy type. A module imports them, and any name used only in a hint, under `if TYPE_CHECKING:`, so that no notebook or Colab run gains a runtime import, and it opens with `from __future__ import annotations` (rule 3.4), because Python otherwise evaluates each annotation when it defines the function.

## 6. Docstrings and comments

6.1. Every library module opens with a docstring that states what the module carries, maps its objects to the canon's notation, and names its sources by `references.bib` key.
- Mentors. PyPSA's module docstrings are one line (PyPSA `pypsa/optimization/constraints.py:5`). GridPath's module docstring describes the modeled object and its costs (GridPath `gen_hydro.py:15-26`).
- Choice. GridPath's descriptive form, extended by the library's map from code to notation. This is the repository's own extension, because the canon is the series' own.
- Here. 14 of 14 modules of `lib/`. Exemplar: `lib/seams_engine.py:5` (the map from code to notation) and `lib/etag.py:5` (the object, its standards and the library's simplifications).

6.2. A public function has a docstring that follows numpydoc: a one-line declarative summary, a blank line, an extended summary in prose, then `Parameters` and `Returns` sections that give each argument's and each returned quantity's type, shape, unit and sign, with `Notes` and `Examples` where they apply.
- Mentors. PyPSA writes numpydoc (488 docstrings with sections; PyPSA `pypsa/optimization/constraints.py:1315-1361`, `pypsa/network/power_flow.py:1113-1123`), and so do linopy (144) and EGRET (EGRET `egret/models/dcopf.py:505-532`). GridPath writes Sphinx `:param:` fields (GridPath `gridpath/auxiliary/validations.py:338-353`). The lint enforces PyPSA's layout rules: the summary on the first line (`D212`), a blank line after it (`D205`), and the closing quotes on their own line (`D209`).
- Choice. numpydoc, the practice of three of the four mentors. The summary is a declarative noun phrase ("The ambient case, ...") and never an imperative, following linopy's ignore of `D401` (linopy `pyproject.toml:203`) and the TD register, which cuts the imperative where the declarative is available. The declarative summary applies to every docstring in the repository. The `Parameters` and `Returns` sections apply to library code, and a harness function states its part of the hook contract in prose instead (rule 13.6).
- Here. 79 percent of the functions of `lib/` and 83 percent of its public functions have a docstring, mostly in prose with code names in double backticks. 20 docstrings carry a numpydoc section. `D` reports 473 findings in `lib/`, chiefly `D205` (156) and `D209` (105).

6.3. A function that defines a constraint family states the family's name, its index set and its dual in its docstring, with a worked number where one helps.
- Mentors. GridPath opens each constraint rule's docstring with "**Constraint Name**: GenHydro_Max_Power_Constraint" and "**Enforced Over**: GEN_HYDRO_OPR_TMPS", followed by a worked example in MW (GridPath `gen_hydro.py:535-549`, Apache-2.0). PyPSA defines one family per function, `define_nodal_balance_constraints` (PyPSA `pypsa/optimization/constraints.py:1307`).
- Choice. GridPath's form, with the dual named in the canon's notation, because the canon reads every constraint with its dual.
- Here. `solve_engine_transfers` (`lib/seams_engine.py:1128`) assembles its rows inline in one function, and its docstring names the families with their duals, the intertie scheduling limit among them (`lib/seams_engine.py:1308`).

6.4. Every quantity a function returns states its unit (MW, $/MWh, $/h), and every signed quantity states its positive direction.
- Mentors. PyPSA's attribute tables give a unit for every attribute and the positive direction of every signed one, for example "Active power at bus (positive if net generation)" (PyPSA `pypsa/data/component_attrs/generators.csv:48`) and "positive if branch is withdrawing power from `bus0`" (PyPSA `lines.csv:31`).
- Choice. The unit and the sign go in the docstring and the field comment, because this repository keeps no attribute tables.
- Here. Exemplar: `lib/seams_engine.py:625` (`exo`, "positive = injection", in `solve_engine_dispatch`) and the field comments of `EngineResult`, `lib/seams_engine.py:499`.

6.5. Math in a new module's docstring is LaTeX between `$` signs in a raw docstring, in the canon's notation.
- Mentors. PyPSA writes "According to formula $r / (1 - (1 + r)^{-n})$" (PyPSA `pypsa/common.py:904`) and `$f \leq F$` in its attribute tables. GridPath writes reStructuredText `:math:` roles (19 uses, GridPath `gridpath/objective/max_npv.py:79`). Math in docstrings is rare in all four mentors.
- Choice. PyPSA's `$...$`, because the notebooks write the canon in the same notation. A backslash requires a raw docstring (`D301`).
- Here. `lib/` writes ASCII and Unicode math, and an existing module keeps its own form.

6.6. A module-level constant or knob is documented by a `#:` comment block above it, which states its meaning, its unit, its default and the decision that set it. linopy documents module attributes this way (linopy `linopy/persistent/diff.py:36`), and Sphinx reads the `#:` form. Exemplar: `lib/revenue_allocation.py:3069` (`BACKSTOP_PRICE`, with its unit and its label as a modeling choice). `lib/` carries 116 `#:` lines.

6.7. A dated decision is cited in the docstring or `#:` block where the code carries it, with the date and the owner, as in "(user decision 2026-10-25)". This is the repository's own practice.

6.8. A comment states a reason, a unit or a sign, and prose in a docstring or comment follows the TD register of the `writing-voice` skill, with no em-dashes. Exemplar: `lib/etag.py`, `lib/seam_views.py` and `lib/lab_types.py`, which carry no em-dash. The other 11 modules of `lib/` carry 260 lines with one.

6.9. Open work is recorded in the open decisions of `dev_notes.md` and not in a `TODO` comment. PyPSA keeps `TODO` comments (PyPSA `pyproject.toml:134`, `docs/contributing/contributing.md:39`), and `lib/` carries none.

## 7. The formulation in code

The rules of this section are the repository's own. No mentor supplies them.

7.1. A code name follows the name the library already gives the canon's object, and a new object takes a name in the same pattern. The established names are:

| Canon object | Code name | Note |
|---|---|---|
| nodal price, $\lambda_n$ | `lmp` | `EngineResult.lmp`, keyed by bus |
| pooled energy price, $\lambda$ | `energy_price` | |
| area energy price, $\hat\lambda_a$ | `area_prices` | never the hub price |
| hub price, $\lambda_a^w$ | `hub_weights`, `hub_congestion_decomposition` | ex-ante weights |
| line dual, $\mu_m$ | `line_dual` | signed |
| interface dual, $\mu_T^{ab}$ | `transfer_duals` | signed, keyed by sorted pair |
| scheduled transfer, $t_{ab}$ | `transfers` | flow-inert |
| released capacity, $\bar T_{ab}$ | `transfer_limits` | `None` is unlimited |
| engine's own flow, $F_m$ | `flow_own` | |
| shift factor, $SF_{n,m}$ | `PTDFData.ptdf` | stored line by bus, `(n_line, n_bus)` |
| line rating, $\bar F_m$ | `s_nom` | MW |
| resource capacity, $\bar g_i$ | `p_nom` | MW |
| exogenous injection, $\mathrm{exo}_n$ | `exo` | positive is injection |
| shed price, $V$ | `shed_price` | $/MWh |
| congestion by region, $N_a^c$ | `regional_congestion` | |

The array `PTDFData.ptdf` is stored line first while the canon writes the shift factor bus first. A new function that indexes it writes `pt.ptdf[m, n]` and states the order in its docstring. Exemplar: `lib/seams_engine.py:87` (`PTDFData`).

7.2. The line dual is signed and the nodal price is the area energy price plus the shift-factor-weighted sum of the line duals, $\lambda_n=\hat\lambda_{a(n)}+\sum_m SF_{n,m}\mu_m$, in every function, whatever convention a solver or a reference implementation uses.

7.3. The docstring of a function that implements a formulation names the equation it implements (the program, the pooled balance, the nodal price identity) and the source construct by `references.bib` key with its section. Exemplar: `solve_engine_transfers`, whose docstring cites `caiso_edam_techdesc` sections 3, 3.3 and 5.2 for the per-area balance and the released capacity (`lib/seams_engine.py:1152`).

7.4. A docstring cites a notebook's equations by name and never by number, because a notebook renumbers its equations from (1). A paper's equation number appears only beside the paper's name.

7.5. A dual is read from the solver once, in the function that calls the solver, and that function states the solver's sign convention and the conversion to the canon's signed line dual, $\mu_m$. Exemplar: `lib/seams_engine.py:818-823`, in `solve_engine_dispatch`, whose comment states that the HiGHS marginals are the derivative of the objective in the right-hand side. EGRET reads its nodal price from the balance dual in the solving function in the same way (EGRET `egret/models/dcopf.py:583-597`).

7.6. A choice that no source makes is labelled "a modeling choice" in the docstring of the object that carries it, with its reason, and a departure from a source is listed with the source. Exemplar: `lib/seams_engine.py:1334`, the three departures of the intertie scheduling limit from its source, and `lib/atc.py:1757`. `lib/` carries the label 7 times.

7.7. A program is assembled one constraint family at a time, each family with its own named block (a function, or a commented block of rows) and the docstring of rule 6.3. GridPath assembles each family through its own rule function (GridPath `gen_hydro.py:499-527`), PyPSA through one `define_*` function per family, and EGRET through its `declare_*` library functions.

7.8. `lib/` is not edited here. A change the notebooks need from the library is proposed in the open decisions of `dev_notes.md` and arrives with a fresh copy, and a notebook-local helper carries it until then. A local change to `lib/` reads as a divergence from the hashes of `lib/MANIFEST.md`.

7.9. The methodology check of `.claude/skills/modify-example/SKILL.md` runs on the diff of every code change, and its three-line record is reported with the change. This guide does not restate the check.

## 8. Numbers and data

The rules of this section are the repository's own. They are stricter than the mentors' test practice: EGRET types an expected objective value into a test (EGRET `egret/models/tests/test_dcopf.py:131`) and GridPath an expected build quantity (GridPath `tests/test_examples.py:1792`), which this repository does not do, because a typed-in number cannot be traced to the run that produced it.

8.1. A number that a run produces (a price, a flow, a dual, a rent, an allocation) is computed, and is never written as a literal in library code, in a test expectation or in a notebook cell. A golden value is frozen by its runner, in the way GridPath and EGRET keep expected results in data files (GridPath `tests/test_data/test_scenario_objective_function_values.csv:1`; EGRET `egret/models/tests/test_dcopf.py:24-25`). Here `tests/golden/golden.json` is copied with `lib/`, and `tests/golden_runner.py --write` refuses without `--local` (`tests/golden_runner.py:724`).

8.2. An input number comes from a named source: a `references.bib` source cited where the constant is defined, a deck page the notebook reproduces, or a constant labelled a modeling choice in its `#:` block. Exemplar: `lib/revenue_allocation.py:3069` (`BACKSTOP_PRICE`).

8.3. A parameter a reader may change is a knob: an `UPPER_CASE` assignment in a notebook's EDIT cell, a module-level constant with a `#:` block, or a keyword argument with its default stated in the docstring. Exemplar: `lib/seams_engine.py:2028` (`SHOW_IDLE_BACKSTOPS`).

8.4. A default that changes a result is never changed silently. The change is stated in `dev_notes.md`, and a notebook whose result depends on the default sets it in its EDIT cell.

8.5. A numerical tolerance is a named argument or module constant, and a comparison states whether it is absolute or relative. PyPSA names its test tolerance (`TOLERANCE = 1e-5`, PyPSA `test/test_capital_cost_split.py:17`) and writes `pytest.approx(25.0, abs=0.1)` and `rel=1e-3`; EGRET compares objectives with `rel_tol=1e-6`. Here 29 functions of `lib/` take a named tolerance, as `solve_engine_transfers` takes `tol: float = 1e-3` (`lib/seams_engine.py:1136`).

## 9. Result objects and solvers

9.1. A solver result is a `@dataclass`, a ledger or table is a pandas `DataFrame`, and a per-bus, per-line or per-area map is a `dict` keyed by name.
- Mentors. linopy returns its solve as nested dataclasses, `Result`, `Status`, `Solution` and `SolverReport` (linopy `linopy/constants.py:258-330`). PyPSA writes its results onto the network object and returns the solver's status and termination condition (PyPSA `pypsa/optimization/optimize.py:561`, `707`). EGRET writes its results into the `ModelData` it was given. GridPath writes results to files and a database.
- Choice. linopy's dataclass, because a result here is compared across engines and must not mutate its inputs.
- Here. 31 of the 40 classes of `lib/` are dataclasses. Exemplar: `lib/seams_engine.py:499` (`EngineResult`).

9.2. A field added to a result dataclass takes a default (`None` or `field(default_factory=dict)`), so that every existing constructor stays valid. linopy's `Solution` and `SolverReport` default every field (linopy `linopy/constants.py:291-317`). Exemplar: `lib/seams_engine.py:512` (`shed_by_bus`) and the fields after it.

9.3. A result names the solver that formed it.
- Mentors. linopy's `Result.solver_name` (linopy `linopy/constants.py:328`).
- Here. `EngineResult` does not carry the solver. The rule matters because two solvers can return different optimal duals on a dual-degenerate program.

9.4. A linear program is solved by `scipy.optimize.linprog(method="highs")`. HiGHS is also the default solver of PyPSA and GridPath (`highspy` in PyPSA `pyproject.toml:49` and GridPath `pyproject.toml:23`). `seams_engine.py`, `atc.py` and `bilateral.py` call `linprog`.

## 10. Errors, warnings, logging and dependencies

10.1. An invalid input raises `ValueError`, or a subclass named for the check, with the offending value in the message (`!r`); a missing name raises `KeyError`; a solver failure raises `RuntimeError`.
- Mentors. PyPSA raises `ValueError` with the value in the message (PyPSA `pypsa/network/power_flow.py:380-381`) and subclasses it for its consistency checks, `ConsistencyError(ValueError)` (PyPSA `pypsa/consistency.py:37`); linopy subclasses `ValueError` and `RuntimeError` (linopy `linopy/restrict/matrix.py:38`).
- Choice. The message is written inline in the `raise`, as linopy, GridPath and EGRET write it. PyPSA assigns it to `msg` first (the `EM` rules), which is not adopted.
- Here. 100 of the 131 raises in `lib/` are `ValueError`. Exemplar: `lib/seams_engine.py:370`, in `FlowgateSet`.

10.2. A check that inspects a whole case runs before the solve, in its own function, and either raises or reports each failure.
- Mentors. PyPSA's consistency module raises in strict mode and logs a warning otherwise (PyPSA `pypsa/consistency.py:45-49`). GridPath validates every input in a separate `validate_inputs` pass that collects error messages (GridPath `gridpath/auxiliary/validations.py:335-382`).
- Here. `check_fleet` and `check_case` return a list of violations, empty when the case conforms (`lib/seams_engine.py:2433` and `2493`). Other validation is inline in the solving functions.

10.3. A library function never uses `assert` to validate an input or a result, because Python drops `assert` under `-O`.
- Mentors. PyPSA selects `S` and so flags `assert` in its package (`S101`), and ignores it in tests (PyPSA `pyproject.toml:214`, `259-271`).
- Here. `lib/` carries 29 `assert` statements.

10.4. A diagnostic goes to `logging.getLogger(__name__)`, a fallback or a condition the caller must act on goes to `warnings.warn(..., stacklevel=2)` with a stated category, and `print` appears only in a function whose purpose is to print a report, marked `# noqa: T201 - <reason>`.
- Mentors. PyPSA declares a module logger (PyPSA `pypsa/optimization/constraints.py:47`, 41 modules), uses `warnings.warn` with `stacklevel` for deprecations (PyPSA `pypsa/common.py:504-508`), and selects `T20` (4 `print` calls in its package). linopy logs in the same way (10 modules with a logger, 4 `print` calls) without selecting `T20`. GridPath prints (103 calls) and logs nothing.
- Choice. PyPSA's practice, because a notebook reader can silence a logger and cannot silence a `print`.
- Here. `lib/ieee9_network.py:28` declares the one logger of `lib/`, and `lib/` carries 79 `print` calls. Exemplar of a warned condition: `lib/atc.py:2332`, in `settle_rights`, which warns when an option book was tested on a reading that credits counterflow.

10.5. A broad `except Exception` carries `# noqa: BLE001 - <reason>` and re-raises, warns or falls back loudly, and never passes silently. This is the repository's own rule; no mentor selects `BLE001`. `lib/` carries 10 such annotations. Exemplar: `lib/seams_engine.py:2560`, in `check_case`, which records the exception as a violation.

10.6. An optional dependency (`pypsa`, `pandapower`, `pycirclize`) is imported inside the function that needs it, on a line marked `# noqa: PLC0415 - optional dependency`, or under `if TYPE_CHECKING:` where only a hint needs it, so that the module imports without it.
- Mentors. PyPSA declares optional dependencies as extras (PyPSA `pyproject.toml:59-66`), imports them inside functions on lines marked `# noqa: PLC0415` (26 such lines, for example PyPSA `pypsa/common.py:919`), and raises an `ImportError` with an install message through `check_optional_dependency` (PyPSA `pypsa/common.py:704-713`). GridPath declares `gurobi` as an extra (GridPath `pyproject.toml:59`).
- Here. `lib/` carries 55 function-level imports, none marked. Exemplar of the hint-only import: `lib/seams_engine.py:63`.

10.7. A path is resolved from `Path(__file__).resolve()` or from the repository root, never from the working directory, and a path operation uses `pathlib`. PyPSA selects `PTH` (PyPSA `pyproject.toml:226`). Exemplar: `.claude/hooks/code_check.py:42` (`ROOT`). The harness carries 62 `Path(` calls and 50 `os.path` calls.

10.8. A text file is opened with `encoding="utf-8"`, because the Windows default code page cannot decode the Greek letters `λ` and `μ`, and subprocess output is decoded with `encoding="utf-8", errors="replace"`. This is the repository's own rule (`PLW1514`). No mentor selects it, although all four test on Windows in continuous integration (GridPath `.github/workflows/test_gridpath.yml:4`). Exemplar: `.claude/hooks/session_brief.py:79`.

## 11. Tests

11.1. The suites run under pytest, with shared cases as fixtures in `conftest.py`.
- Mentors. PyPSA's suite is pytest with 31 fixtures in its conftest (PyPSA `test/conftest.py:166`, `ac_dc_network`), and linopy's is pytest (linopy `AGENTS.md:52-54`). GridPath writes `unittest` classes and collects them with pytest (GridPath `pyproject.toml:60-64`); EGRET writes `unittest` with `parameterized` (EGRET `egret/models/tests/test_dcopf.py:19`).
- Choice. pytest, as in PyPSA and linopy. The harness tests stay on standard-library `unittest` (rule 13.6).
- Here. `tests/` holds 18 test modules, and pytest collects 226 tests from them on 2026-11-07.

11.2. A test is named for the object under test and the claim it asserts, `test_<object>_<claim>`.
- Mentors. PyPSA names a test for the function and the case, `test_define_fixed_operational_constraints_positive` (PyPSA `test/test_lopf_basic_constraints.py:331`).
- Choice. The claim is written as a phrase, because a test here guards an identity and its name states the identity. Exemplar: `tests/test_constraint_set_args.py:192` (`test_coalition_positions_conserve_on_the_cut`).

11.3. A test module opens with a docstring that states what it guards and how to run it. This is the repository's own rule; PyPSA exempts test modules from docstrings (PyPSA `pyproject.toml:275`). Exemplar: `tests/test_structure.py:1`.

11.4. A new function's test asserts three things where they apply: the identity the formulation states, on a case where a constraint binds; the sign convention on a binding row; and agreement with an existing path where the two coincide. This is the repository's own rule. Exemplar: `tests/test_flowgates.py:37` (`test_identity_flowgates_reproduce_lines`).

11.5. A comparison against a reference implementation lives in a differential test that solves one case in this repository's formulation and in the reference, compares the objective, the prices and the flows at a stated tolerance, and skips when the reference package is not importable.
- Mentors. PyPSA tests its power flow against PYPOWER and pandapower (PyPSA `docs/contributing/contributing.md:120-121`, `test/test_lpf_against_pypower.py:6-9`). EGRET tests against stored reference solutions at `rel_tol=1e-6` (EGRET `egret/models/tests/test_dcopf.py:55`).
- Here. This repository has no differential test. A difference such a test finds is never closed by loosening the tolerance.

11.6. A golden moves only with a stated reason, recorded in the commit and in `dev_notes.md` with the stored results it supersedes, and `golden_runner.py --write` refuses to freeze unless the suite reproduces itself twice. GridPath keeps an expected value per platform where solvers differ (GridPath `tests/test_data/test_scenario_objective_function_values.csv:1`). Here the golden arrives with `lib/` (rule 8.1). Exemplar: `tests/test_golden.py:48` (`test_scenario_matches_golden`).

11.7. A test that needs an absent input (a golden, an optional package) skips or fails with a message that says how to produce the input. PyPSA skips with a reason (PyPSA `test/conftest.py:390`). Exemplar: `tests/test_golden.py:24`, whose message states that the golden is copied with `lib/`.

11.8. A slow test is marked `@pytest.mark.slow`, as linopy registers it (linopy `pyproject.toml:142-145`). No test here carries the mark.

11.9. A test never special-cases the configuration under test and never weakens an assertion so that it passes, which is question 3 of the methodology check.

## 12. Notebooks

The rules of this section are the repository's own. PyPSA strips notebook outputs before a commit (PyPSA `.pre-commit-config.yaml:41-45`) and lints notebooks (`extend-include = ['*.ipynb']`, PyPSA `pyproject.toml:199`). This repository keeps the outputs, because a stored output is the citable record of a run, and it does not lint notebooks (rule 12.6). Cell numbers below are `nbformat` indices, counted from 0.

12.1. The first code cell is the `#@title ▶ Run me first` setup cell, which installs the Colab packages and fetches the 14 modules of `lib/`. The four root notebooks place it at cell 1, after the title cell.

12.2. A notebook's knobs sit in EDIT cells, each a fenced block of `UPPER_CASE` assignments marked `# -- EDIT --`, with a unit or meaning comment on each line, followed by the code that uses them. Exemplar: `workshop2_congestion_management.ipynb` cell 7 and `101_sandbox.ipynb` cell 11.

12.3. An export path is resolved from the repository, never from the working directory, and a scratch export goes to `outputs/` or `sandbox_exhibits/`, both ignored. Exemplar: `101_sandbox.ipynb` cells 1 and 2.

12.4. A notebook is edited by cell, with `NotebookEdit` or `nbformat`, and executed headless with `tools/nb_execute.py`, which runs `python -m nbconvert --to notebook --execute --inplace` and stamps the library state into the notebook's metadata. `MPLBACKEND=Agg` is never set.

12.5. A figure in a deck notebook carries no suptitle, no panel title and no legend, because the slide carries them (`AGENTS.md`, "What this repository holds").

12.6. Notebooks are outside the lint (`ruff.toml` excludes `*.ipynb`), because a notebook lint would report the imports and names a deck notebook keeps for its reader, and not new drift.

## 13. Tools and hooks

A tool is a script under `tools/`. A harness script is a hook or a skill script under `.claude/`. Rules 13.3 to 13.6 are the repository's own, because no mentor runs a Claude Code harness.

13.1. A tool opens with a docstring that names its inputs, its outputs and its use, parses its arguments with `argparse`, and runs from `main()` under `if __name__ == "__main__"`. GridPath's command-line modules expose `main()` and `parse_arguments()` (GridPath `pyproject.toml:141-148`). Exemplar: `tools/nb_execute.py` (`main`).

13.2. A tool fails loudly: an unexpected input raises, and the tool writes only to the paths its docstring names. A tool reports on stdout, so `ruff.toml` exempts `tools/` from `T20`, as PyPSA exempts its examples (PyPSA `pyproject.toml:260-266`).

13.3. A harness script uses the standard library only, because Claude Code runs hooks under the system `python`, which does not carry the scientific stack. Exemplar: `.claude/hooks/session_brief.py:1`.

13.4. A hook fails open: it catches every error, writes one line to stderr and exits 0. The one exception is a blocking guard, which exits 2 for a rule it enforces. Exemplar: `.claude/hooks/guard.py:202` (`main`) and `.claude/hooks/code_check.py:286` (`main`).

13.5. A harness script is Windows-safe: text is read and written as UTF-8, a subprocess runs with `encoding="utf-8"`, `errors="replace"`, `stdin=subprocess.DEVNULL`, a timeout and an explicit working directory, and paths are built with `pathlib`. Exemplar: `.claude/hooks/code_check.py:124` (`run_ruff`).

13.6. A hook's module docstring states its contract (the event, the input, the output and the exit codes). Each function's docstring opens with a declarative summary and states its part of the contract in prose, without the numpydoc sections of rule 6.2. The hook has standard-library `unittest` tests under `.claude/hooks/tests/`, which `ruff.toml` exempts from the pytest style (`PT`). Exemplar: `.claude/hooks/code_check.py` and `.claude/hooks/tests/test_code_check.py`.

## 14. The lint and the formatter

14.1. `ruff.toml` at the root selects the rule families of PyPSA's configuration (PyPSA `pyproject.toml:198-295`), less those that no other mentor selects and that a house rule makes redundant (`EM`, `TRY003`), plus a set of defect rules no mentor selects. The file lists each family with the mentor that selects it and each departure with its reason.

14.2. The modules carry inherited findings. On 2026-11-07 `ruff check --config ruff.toml` reports 150 findings over `tools/` (60), `.claude/hooks/` (36) and `tests/` (54), 173 over `.claude/skills/`, and 736 over `lib/`, so a whole-folder run exits 1. The gate is therefore no new finding on a touched line.

14.3. `.claude/hooks/code_check.py` runs the lint after every Write, Edit or MultiEdit of a `.py` file and reports the findings on the touched lines to the agent, so the inherited findings do not flood an edit. For a module off the `[format]` exclude list, the hook also reports a module that `ruff format` would change.

14.4. A finding is fixed in the code. A deliberate case keeps `# noqa: <code> - <reason>` on the line, as PyPSA keeps `# noqa: PLC0415` and `# noqa: TC004` (PyPSA `pypsa/optimization/constraints.py:41`).

14.5. The lint is run by hand from the repository root as `ruff check --config ruff.toml <files>`, because ruff resolves the configuration's relative paths against the working directory. `ruff check --fix` is applied only to the lines being changed. `ruff format` is run by hand only on a module off the `[format]` exclude list, because an explicit path overrides the list.

14.6. `lib/` arrives in the layout its source gives it and is not reformatted here, and a module of `lib/` leaves the `[format]` exclude list when a copy brings it formatted. A module of `tests/`, `tools/` or `.claude/` on the list is formatted in a change of its own that changes nothing else, and leaves the list in the same change.

14.7. A rule family is added to `ruff.toml` only after it is measured on the whole repository, and its count is recorded in the file's comment.

## 15. Open decisions

Each item states the mentor practice, the state of this repository, and a proposed default for a participant to confirm.

1. **The docstring convention on touch.** Rule 6.2 adopts numpydoc. The proposed default is that an edited docstring in `tools/`, `tests/` or `.claude/` gains its `Parameters` and `Returns` sections in the same change, and an unedited docstring is left as it stands.
2. **Warnings as errors in the suite.** PyPSA turns `DeprecationWarning`, `FutureWarning`, `RuntimeWarning` and `UserWarning` into errors in its suite and allows named exceptions (PyPSA `pyproject.toml:124-152`). The suite here sets no warning filter. The proposed default is to measure the warnings the suite emits and then turn `DeprecationWarning` and `FutureWarning` into errors.
