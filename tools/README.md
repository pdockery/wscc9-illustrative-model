# tools

Repository tools that act on the notebooks, `lib/` and the machine. Each is standard library plus the packages named. The notebooks are the `*.ipynb` files at the repository root, and `lib/` is a byte copy of a shared 9-bus library. `lib/MANIFEST.md` records each module's SHA-256 and the date of the copy. Each tool runs from the repository root, and the caches it writes sit under `tools/.cache/`, which `tools/.gitignore` ignores.

## The environment: env_lock.json, lab_env.py and constraints.txt

`constraints.txt` at the root pins every package of the tested environment, and `pip install -r requirements-dev.txt -c constraints.txt` installs it. `requirements.txt` stays unpinned, so that Colab installs the notebook packages beside its own.

`tools/env_lock.json` records the versions of that environment. Each package carries a role: a `model` package (numpy, pandas, scipy, matplotlib, pypsa, pandapower, pycirclize) moves a stored result when its version moves, and a tooling package does not. The Python release the lock records is information only, because the repository admits Python 3.10 and later.

`tools/lab_env.py` compares the running interpreter with the lock. `check_environment()` compares each model package by its exact version and issues a warning that names each difference and the constraints install line. A difference is never an error, because the Colab path runs the notebooks under other versions, and where the lock is absent the check does nothing. `tools/nb_execute.py` and `tools/check.py` call it; the notebooks' bootstrap and `tests/conftest.py` do not.

## doctor.py

`python tools/doctor.py [--json] [--out PATH]`

The report states the machine's readiness to run the suites and the notebooks, and changes nothing. It covers the interpreter it examines (`LAB_PY` where it is set, the running one otherwise), each package of `tools/env_lock.json` against its pin, the notebook tooling and the kernelspecs the root notebooks name, each `lib/` module and each copied hook against its manifest, the data files of `tools/required_data.txt` with the tests that need them, the count of Python processes against the ceiling, and git's `core.hooksPath`. The examined interpreter is probed in a subprocess, so the script runs under any Python 3.10 or later.

The Python release and `core.hooksPath` are information and never a problem. The process count is a problem only at the ceiling. A manifest hash is compared in the form of `NOTEBOOK_RECORD.md` section 3, the file byte for byte as stored in git.

The report prints as text, or as JSON with `--json`, and the JSON is written to `tools/.cache/doctor.json`, which the session brief reads. The exit status is 0 when no problem is found, 1 when one is, and 2 when the script itself fails.

`tools/required_data.txt` lists the data files the tests need, one per line: the path, whether it is tracked or local, and the tests that skip or fail without it. A skip is silent in a passing suite, so the report names every absent path.

## check.py

`python tools/check.py [--quick | --full | --tiers T[,T...]] [--python PATH] [--suite-timeout S] [--mypy | --no-mypy]`

`python tools/check.py --status [--quick | --full | --tiers T[,T...]]`

The verify command runs four tiers in order, one at a time, and stops at the first that fails.

1. `lint`: ruff over the changed Python files, which fails when a file carries more findings of any rule than its version at HEAD. With `--mypy`, or by default with `--full`, `tools/run_mypy.py` runs over the same files, and its count is recorded without failing the tier.
2. `harness`: one pytest run per test folder under `tools/` and `.claude/`, which includes `.claude/skills/writing-voice/tests`.
3. `tests`: the model tests of `tests/`, run with `-m "not slides"`.
4. `slides`: the slide checks, run with `-m slides`. Each executes a deck notebook, so this tier is the slow one.

`--quick` runs the first three tiers and is the default, and `--full` runs all four. The interpreter is `--python`, else `LAB_PY`, else the one running the script. Each pytest run starts only when the process count leaves room under the ceiling, and a run past its timeout is ended by its PID with its children.

A tier's failures are compared with `tools/known_failures.toml`, which holds no entry. A failure the file does not list fails the tier, and so does a listed failure that now passes, because its entry is stale. A pytest tier that collects no test is recorded as `not-run` and stops the run as a failure does.

After each tier one JSON line is appended to `tools/.cache/check_runs.jsonl`, with the tier, its counts and verdict, and the library fingerprint: `lib_commit`, `lib_diff_sha256`, `lib_dirty` and the sha256 of `lib/MANIFEST.md`. `--status` reports, per tier, whether a passing line matches the present fingerprint. The exit status is 0 when every tier run passes, 1 when a tier fails, 2 on a usage or configuration error, and 3 when a launch is refused at the process ceiling. The tool stages and commits nothing.

## nb_execute.py

`python tools/nb_execute.py NOTEBOOK... [--timeout SECONDS] [--root DIR] [--no-prose-lint]`

The script runs the house headless re-run, `python -m nbconvert --to notebook --execute --inplace`, under the interpreter that invokes it. The Matplotlib backend variable is not set, and the script stops when it names Agg. `requirements-dev.txt` names the packages the run needs beside `requirements.txt`.

**The pre-flight.** Before any notebook starts, every code cell of every named notebook is compiled. A cell magic is skipped, and a line magic or a shell escape is masked when the cell fails to compile as it stands. A cell that still fails refuses the whole launch, and the message names the notebook, the cell id and the line. `prose_lint.py` then lints each notebook and reports its findings, which never refuse a launch; `--no-prose-lint` skips the lint. Last, `lab_env.check_environment()` compares the interpreter with the lock.

**The stamp.** On success the notebook's top-level metadata receives `lab_execution`, with these fields.

1. `date` is the UTC time of the stamp.
2. `python` is the interpreter's file name and version.
3. `lib_commit` is `git rev-parse HEAD`.
4. `lib_dirty` is true when `lib/` has uncommitted changes.
5. `lib_diff_sha256` is the hash of `git diff HEAD`, the porcelain status with every untracked file listed, and the sha256 of each untracked file, over `lib/`. A copy that changes an untracked file therefore changes the hash.
6. `lib_manifest` is the path and sha256 of `lib/MANIFEST.md`, which records each module's SHA-256 and the date of the copy.
7. `lib_changed_during_run` is present and true when the library state at the stamp differs from the state at launch.
8. `packages` is the version of each model package of `tools/env_lock.json` and of nbconvert, nbclient and ipykernel, null for one not installed.
9. `kernel` is the notebook's kernelspec name.
10. `host` is the operating system, its release and the machine type.
11. `command` is the nbconvert command line, with the notebook's repository-relative path.

The stamp holds no absolute path, no host name and no environment variable. Cells and outputs are not changed. The stamp is written with the notebook's own layout (indent, key order, escaping and line endings).

**The ledger.** On failure no stamp is written, and the failing cell and the exception are read from nbconvert's error report and printed. Every attempt, refused, failed or stamped, appends one JSON line to `tools/.cache/nb_runs.jsonl`: the notebook, the start and end, the seconds, the exit status, the outcome, the failing cell, the library state at launch, the host and the command. After each stamp the staleness summary is rewritten.

The repository is the one that holds the notebook. `--root DIR` names the repository for a notebook outside every repository, such as a scratch copy, and the stamp then records that repository's `lib/`. A notebook outside every repository is skipped without `--root`.

The script stops before execution when the process count is at the ceiling.

## nb_staleness.py

`python tools/nb_staleness.py [--json] [--root DIR] [--no-cache]`

The report classifies each root notebook.

1. Current: a stamp exists, no file under `lib/` has changed in code since `lib_commit` (committed or in the working tree), and an executed-dirty stamp matches the present diff hash.
2. Stale: lib files changed in code after `lib_commit`, named in the report, or the stamp records uncommitted lib changes that no longer match, or `lib_commit` does not resolve in this repository, so the lib state cannot be compared.
3. Unstamped: no stamp. The report estimates by comparing the notebook's last commit time with the last commit that changed code in `lib/`, and labels the result an estimate.

A changed file is one that `git diff --name-only` lists or an untracked file that `.gitignore` does not cover, and it is a change only when its code differs. A Python file whose runtime fingerprint at `lib_commit` equals its working-tree fingerprint is dropped from the list, because a reformat, a comment edit or a type hint does not move a stored result, and the row's detail states how many files were dropped. The runtime fingerprint is the code fingerprint with hints removed by `strip_hints` in `_common.py`, so a `cast()` or an assert stays a change. A file that is not Python, that `lib_commit` does not hold, or that does not parse stays a change. `lib/MANIFEST.md` is such a file, so a fresh copy of the library marks every stamped notebook stale. A dropped file does not clear a stamp written with uncommitted lib changes, because that stamp's code is not the code at `lib_commit`. The estimate applies the same test to each lib commit, newest first, and stops at the first that changed code, and the row's detail counts the later commits it passed over.

The stamp is read from the last megabyte of the file, where nbformat writes the sorted metadata block, so a 20 MB notebook costs one small read. A fingerprint is kept under `tools/.cache/fingerprints`, keyed by the Python release and the file's content. Every run without `--no-cache` writes `tools/.cache/staleness.json` through a temporary file and a rename: the counts per class, the time it was generated and the commit classified (`head`). `nb_execute.py` rewrites it after each stamp. The session brief reads it only when it is newer than the newest change to `lib/`. `--json` prints the rows under the key `root`.

### The code fingerprint

`code_fingerprint` in `_common.py` is the sha256 of `ast.dump` of the parsed module, which excludes positions, with three forms the formatter changes brought to one reading: a docstring's whitespace, the `kind` of a constant, which a dropped `u` prefix changes, and a `del` of one parenthesized tuple. A `SyntaxError` returns None, and None equals nothing. The fingerprint does not see comments, layout, parentheses, trailing commas, quoting, string prefixes, line endings or a byte order mark. A fingerprint depends on the interpreter's `ast.dump`, so a fingerprint is compared within one Python release.

## run_mypy.py

`python tools/run_mypy.py` checks the whole scope, and `python tools/run_mypy.py MODULE...` checks named modules. `--cache NAME` gives a caller a cache folder of its own. `mypy.ini` holds the settings (`CODE_STYLE.md` section 5).

One pass reads the `files` of `mypy.ini`: `lib/`, `tools/`, `.claude/hooks/` and the writing-voice scripts. `tests/` is excluded. The pass imports with `--follow-imports=silent`. `lib/` is a copy, so a finding there is fixed at the library's source and arrives with the next copy.

A run over named modules prints only the findings in those modules, with its own count, and returns 1 only when one of them has a finding. mypy 1.19.1 replays the cached findings of an imported module whatever `--follow-imports` says, so a whole run uses `tools/.cache/mypy/shared` and a named run `tools/.cache/mypy/shared-modules`.

## Process ceiling

The ceiling is the count of Python interpreter processes at which a heavy launch is refused: `CEILING` in `.claude/hooks/_limits.py`, seven by default, or the positive integer that the environment variable `LAB_PROCESS_CEILING` sets. An unset or invalid value leaves the default. `.claude/hooks/guard.py` blocks a Bash or PowerShell call that launches pytest, nbconvert or `nb_execute.py` at the ceiling. A launch sent to the background needs the count `BACKGROUND_MARGIN` (two) below the ceiling, because it keeps running after the call returns and the next count does not see it. `nb_execute.py` and `check.py` hold to the same ceiling. The guard reads the command word of each segment, so a rule's text inside `echo`, `grep` or a quoted string passes, and the quoted payload of a wrapper such as `bash -c` is read as a command. A count that cannot be taken allows the call.

A Python process is ended by its PID, never by its image name, because a kill by image name also ends the participant's kernels and other runs. The guard refuses `taskkill /IM python*`, `Stop-Process -Name python*`, `pkill python` and their kind.

Every block, ask and warning of the guard is appended to `.claude/logs/hooks.jsonl` by `.claude/hooks/_log.py`. The folder carries its own ignore file, so the log never reaches a commit, and the session brief prints its counts.

## Tests

`python tools/check.py --tiers harness` runs the harness tests, one pytest run per test folder under `tools/` and `.claude/`. A single folder runs as `python -m pytest tools/tests -q` or `python -m pytest .claude/hooks/tests -q` from the repository root, or `python -m unittest discover -s tools/tests -v`. The execution round-trip test is skipped when nbconvert or ipykernel is absent, and the hook tests that run ruff are skipped when ruff is absent. The tools tests work on temporary repositories in this repository's layout.
