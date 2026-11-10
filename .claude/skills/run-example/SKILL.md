---
name: run-example
description: Runs a notebook headless with the stamped wrapper, reports what moved against the stored outputs, and reports the staleness of every notebook before and after. Use when stored results must be refreshed or confirmed, or after a fresh copy of the library arrives.
disable-model-invocation: false
---

# Running an example

You are re-executing one or more root notebooks so that their stored outputs carry the library state they were produced under. The stored outputs are the record, and the stamp in `metadata.lab_execution` says which library they came from. `CLAUDE.md` names the environment and the commands, and `orientation` has run.

## Intake

If `orientation` has not run in this session, say in one sentence that you are running it first because this skill assumes its reading, invoke it, deliver its brief, and then begin the interview. Skip it only where the participant has said to skip it.

Where an input is missing, interview the participant for it, one question at a time, with the AskUserQuestion tool where it is available. Each question says what the input is for and gives one example.

1. **The notebook or notebooks.** The example: "`workshop2_congestion_management.ipynb`", or "every stale notebook".
2. **In place or on a scratch copy.** The default is a scratch copy under `sandbox_exhibits/`, which `.gitignore` excludes, unless the stored results are meant to be replaced. The example: "in place, after the library copy of today".
3. **A per-cell timeout.** The default is none. The example: "600 seconds".

When the set is complete, state it back and proceed.

## Procedure

1. `python tools/nb_staleness.py`, reported as it prints.
2. Where `lib/` differs from HEAD (`git status -- lib`), `python -m pytest tests -q` runs first, and a failure stops the run.
3. A scratch copy is a copy of the file under `sandbox_exhibits/`. The copy sits inside this repository, so its stamp records this repository's `lib/`. A copy outside the repository takes `--root <this repository>`.
4. `python tools/nb_execute.py <notebook>.ipynb [--timeout SECONDS] [--root DIR]`, one notebook at a time, in the foreground. The wrapper refuses at seven Python interpreters and when `MPLBACKEND` names Agg. `MPLBACKEND=Agg` is never set, because it strips every stored figure.
5. Read the new stamp: `date`, `python`, `lib_commit`, `lib_dirty`, `lib_diff_sha256`, `lib_manifest`, `command`.
6. Compare the stored outputs with the previous version cell by cell (`git show HEAD:<notebook>` for an in-place run, the original file for a scratch copy). Report each printed number that moved, with its old and new value and its cell, and each anchor the notebook asserts that held.
7. `python tools/nb_staleness.py` again.

A cell that fails is reported with its traceback, and the notebook is not edited to make it pass. A number that moved is reported, and the reason is found through `modify-example`, not here.

## Colab

Each notebook's first cell installs `pypsa pandapower pycirclize` on Colab and fetches the 14 modules from `https://raw.githubusercontent.com/pdockery/wscc9-illustrative-model/main/lib/`. A library change reaches Colab only when it is pushed to `main`. The badge for a notebook is

`https://colab.research.google.com/github/pdockery/wscc9-illustrative-model/blob/main/<notebook>.ipynb`

## Your first task

Restate the notebooks and the mode. Report the staleness before, run, and report the stamp, the moved numbers with their cells, the anchors that held, and the staleness after.

## Sub-agent models

Sub-agents run on the cheapest model tier that fits their task, the rule `CLAUDE.md` states (user decision 2026-10-18), and every brief names the model. In this skill a headless re-execution and its report run on `sonnet` (`notebook-reexecutor`), and the test run alone runs on `haiku` (`test-runner`). The comparison against the stored outputs is the supervisor's own.
