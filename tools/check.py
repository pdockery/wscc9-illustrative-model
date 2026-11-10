# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
r"""One verify command: the lint, the harness tests, the model tests and the slide checks.

Usage::

    python tools/check.py [--quick | --full | --tiers T[,T...]] [--python PATH]
                          [--suite-timeout S] [--mypy | --no-mypy]
    python tools/check.py --status [--quick | --full | --tiers T[,T...]]

The tiers run in this order, one at a time, and the run stops at the first tier that fails.
Each tier after it prints one line that names the stopping tier and why, such as
``harness: not run (lint failed)``, and writes no ledger line.

1. ``lint``: ruff over the changed Python files, which fails when a file carries more findings
   of any rule than its version at HEAD. With ``--mypy``, or by default with ``--full``,
   ``tools/run_mypy.py`` runs over the same files, and its count is recorded and does not
   fail the tier, because the scope carries findings that were never cleared.
2. ``harness``: each test folder under ``.claude/`` and ``tools/``, one pytest run per folder,
   which includes ``.claude/skills/writing-voice/tests``.
3. ``tests``: the model tests of ``tests/``, run with ``-m "not slides"`` and
   ``PYTHONPATH=lib``.
4. ``slides``: the slide checks of the deck notebooks, run with ``-m slides``. Each one
   executes a deck notebook, so this tier is the slow one.

``--quick`` runs the first three tiers and is the default, and ``--full`` runs all four
and records the mypy count. Each pytest run is one process, started only when the count of
other Python processes leaves room for it and this script under the ceiling, and a run past
its timeout is ended by its PID with its children. The ceiling is ``LAB_PROCESS_CEILING``
when set, else the ceiling of ``.claude/hooks/_limits.py``. The interpreter is ``--python``,
else ``LAB_PY``, else the one running this script.

A tier's failures are compared with ``tools/known_failures.toml``. A failure the file does not
list fails the tier, and so does a listed failure that now passes, because its entry is stale.
A pytest tier that collected no test is recorded with status ``not-run`` and ``ok`` false,
and it stops the run as a failure does. After each tier one JSON line is appended to
``tools/.cache/check_runs.jsonl`` (ignored by ``.gitignore``) with the tier, its counts and
verdict, and the library fingerprint: ``lib_commit``, the last commit that touched ``lib/``,
``lib_diff_sha256`` and ``lib_dirty`` from ``tools/_common.lib_state``, which hashes the
content of each untracked file, and ``manifest_sha256`` of ``lib/MANIFEST.md``. ``--status``
reports, per tier, whether a passing line matches the present fingerprint on all three of
``MATCH_KEYS``. The tool stages and commits nothing.

Exit status: 0 when every tier run passes (or, with ``--status``, when every tier has a
matching passing line), 1 when a tier fails, 2 on a usage or configuration error, 3 when a
launch is refused at the process ceiling. Standard library only (CODE_STYLE.md 13.3).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import importlib.util
import json
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import _common as C  # noqa: E402

TIERS = ("lint", "harness", "tests", "slides")
QUICK = ("lint", "harness", "tests")
#: The library folders whose state is the fingerprint of a ledger line.
LIB_DIRS = (C.LIB_DIR,)
#: The fingerprint keys a ledger line must equal for ``--status`` to count it.
MATCH_KEYS = ("lib_commit", "lib_diff_sha256", "manifest_sha256")
LEDGER = ROOT / "tools" / ".cache" / "check_runs.jsonl"
KNOWN = ROOT / "tools" / "known_failures.toml"
KNOWN_FIELDS = ("id", "tier", "cause", "since", "record", "remove_when")
#: Seconds allowed to one run of a test folder or a whole suite.
SUITE_TIMEOUT = 3600
#: The ceiling when ``.claude/hooks/_limits.py`` cannot be read.
DEFAULT_CEILING = 7
#: Paths outside the lint, as ruff.toml's extend-exclude holds them.
LINT_SKIP = ("sources/", "outputs/", "sandbox_exhibits/")
#: The pytest options of every run: quiet, a summary line per test, no cache folder written,
#: and the repository root as the rootdir, so a node id is relative to the root whichever
#: folder the run names (without an ini file pytest takes the run's own folder as the rootdir).
PYTEST_OPTS = ("-q", "-rA", "-p", "no:cacheprovider", "--rootdir=.")
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL) (\S+)")
_COUNT = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)"
)
#: The line ``tools/run_mypy.py`` prints at the end of each pass over named modules.
_MYPY_PASS = re.compile(r"^\d+ named, (\d+) errors?\r?$", re.M)


class ConfigError(ValueError):
    """A malformed known-failures file or an unknown tier."""


class CeilingError(RuntimeError):
    """A launch refused because the Python processes are at the ceiling."""


@dataclasses.dataclass
class Run:
    """One subprocess of a tier: its label, its command, its folder and its time limit."""

    label: str
    argv: list[str]
    cwd: Path
    timeout: float
    env: dict[str, str] | None = None
    base: str = ""


@dataclasses.dataclass
class TierResult:
    """What one tier found, compared with the known failures."""

    tier: str
    ok: bool = True
    status: str = "pass"
    counts: Counter = dataclasses.field(default_factory=Counter)
    passed: set[str] = dataclasses.field(default_factory=set)
    failed: set[str] = dataclasses.field(default_factory=set)
    new_failures: list[str] = dataclasses.field(default_factory=list)
    known_failures: list[str] = dataclasses.field(default_factory=list)
    known_now_passing: list[str] = dataclasses.field(default_factory=list)
    known_not_run: list[str] = dataclasses.field(default_factory=list)
    timed_out: list[str] = dataclasses.field(default_factory=list)
    #: The runs that collected no test (pytest exit 5).
    not_run: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    tail: str = ""
    duration_s: float = 0.0


def _load_module(name: str, path: Path) -> Any:
    """A module loaded from a file path, or None when the file is absent or fails to import."""
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        return None
    return mod


def _limits() -> Any:
    """The hooks' shared ``_limits`` module, or None where it cannot be read."""
    return _load_module("_limits", ROOT / ".claude" / "hooks" / "_limits.py")


def ceiling() -> int:
    """The Python process ceiling: ``LAB_PROCESS_CEILING`` when a positive integer, else the hooks' own."""
    raw = os.environ.get("LAB_PROCESS_CEILING", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    mod = _limits()
    value = getattr(mod, "CEILING", None)
    return value if isinstance(value, int) and value > 0 else DEFAULT_CEILING


def python_count() -> int | None:
    """The other Python interpreter processes now running, or None when they cannot be counted."""
    mod = _limits()
    counter = getattr(mod, "python_count", None)
    if counter is None:
        return None
    try:
        return counter()
    except Exception:
        return None


def ruff_argv(python: str) -> list[str] | None:
    """The command that runs ruff, or None when ruff is absent.

    The finder is ``find_ruff`` of ``.claude/hooks/code_check.py``. Where it finds nothing,
    ``python -m ruff`` stands in when that interpreter has the module.
    """
    mod = _load_module("code_check", ROOT / ".claude" / "hooks" / "code_check.py")
    found = mod.find_ruff() if mod is not None and hasattr(mod, "find_ruff") else None
    if found:
        return [found]
    probe = subprocess.run(
        [python, "-m", "ruff", "--version"],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return [python, "-m", "ruff"] if probe.returncode == 0 else None


def _parse_known_text(text: str) -> list[dict[str, str]]:
    """The ``[[failure]]`` tables of the known-failures file, read without ``tomllib``.

    The reader takes the file's own subset of TOML, a table header and ``key = "string"``
    lines, for an interpreter older than 3.11. Any other line raises ``ConfigError``.
    """
    entries: list[dict[str, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s == "[[failure]]":
            entries.append({})
            continue
        m = re.fullmatch(r'([A-Za-z_]+)\s*=\s*"((?:[^"\\]|\\.)*)"', s)
        if not m or not entries:
            raise ConfigError(
                f'known_failures.toml line {i}: not a key = "string" line'
            )
        entries[-1][m.group(1)] = json.loads(f'"{m.group(2)}"')
    return entries


def load_known(path: Path = KNOWN) -> dict[str, dict[str, str]]:
    """The known failures by node id, each entry checked for every field and a known tier."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415 - Python 3.11 and later; the text reader stands in below

        try:
            entries = tomllib.loads(text).get("failure", [])
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path.name}: {exc}") from exc
    else:
        entries = _parse_known_text(text)
    out = {}
    for n, e in enumerate(entries, 1):
        missing = [f for f in KNOWN_FIELDS if not str(e.get(f, "")).strip()]
        if missing:
            raise ConfigError(f"{path.name} entry {n}: missing {', '.join(missing)}")
        if e["tier"] not in TIERS:
            raise ConfigError(f"{path.name} entry {n}: unknown tier {e['tier']!r}")
        if e["id"] in out:
            raise ConfigError(f"{path.name} entry {n}: {e['id']} is listed twice")
        out[e["id"]] = {k: str(v) for k, v in e.items()}
    return out


def node_id(raw: str, base: str = "") -> str:
    """A summary line's node id, relative to the repository root with forward slashes.

    ``base`` is the run's folder relative to the root, empty for the root itself.
    """
    path, sep, rest = raw.partition("::")
    path = path.replace("\\", "/")
    if base:
        path = posixpath.normpath(f"{base}/{path}")
    return f"{path}{sep}{rest}"


def parse_pytest(output: str, base: str = "") -> tuple[set[str], set[str], Counter]:
    """The passed node ids, the failed or erroring node ids, and the counts of a pytest run.

    The ids come from the ``-rA`` summary lines, through ``node_id`` with ``base``, and the
    counts from the last line that reports them. An XPASS counts as passed, because a
    non-strict one does not fail the run.
    """
    passed: set[str] = set()
    failed: set[str] = set()
    counts: Counter = Counter()
    for line in output.splitlines():
        m = _RESULT.match(line)
        if m:
            kind, nodeid = m.group(1), node_id(m.group(2), base)
            if kind in ("PASSED", "XPASS"):
                passed.add(nodeid)
            elif kind in ("FAILED", "ERROR"):
                failed.add(nodeid)
    for line in reversed(output.splitlines()):
        found = _COUNT.findall(line)
        if found and " in " in line:
            for n, word in found:
                key = {"error": "errors", "warning": "warnings"}.get(word, word)
                counts[key] += int(n)
            break
    return passed, failed, counts


def compare(result: TierResult, known: dict[str, dict[str, str]]) -> TierResult:
    """The tier's verdict against the known failures of its tier.

    A failure the file does not list is new. A listed failure that passed is stale. A listed
    failure that neither failed nor passed (skipped, deselected or not collected) is noted.
    """
    mine = {k for k, v in known.items() if v["tier"] == result.tier}
    result.new_failures = sorted(result.failed - mine)
    result.known_failures = sorted(result.failed & mine)
    result.known_now_passing = sorted(mine & result.passed)
    result.known_not_run = sorted(mine - result.failed - result.passed)
    if (
        result.new_failures
        or result.known_now_passing
        or result.timed_out
        or result.errors
    ):
        result.ok = False
        result.status = "fail"
    return result


def _kill_tree(proc: subprocess.Popen) -> None:
    """The process and its children ended by the process's own PID, never by image name."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def room_for_launch(count: int | None, cap: int) -> bool:
    """Whether one more Python process fits: the others, this script and the child.

    An unknown count allows the launch, as the guard allows it (fail open).
    """
    return count is None or count + 2 <= cap


def execute(
    run: Run, *, cap: int, counter: Any = python_count
) -> tuple[int | None, str]:
    """One subprocess run to its end or its timeout, as (exit status or None, output).

    ``CeilingError`` is raised before the launch when the process count leaves no room.
    """
    n = counter()
    if not room_for_launch(n, cap):
        raise CeilingError(
            f"{n} other Python processes run and the ceiling is {cap}; "
            f"{run.label} was not started. End orphaned processes by PID and rerun."
        )
    env = None
    if run.env:
        env = {**os.environ, **run.env}
    kwargs: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {}
    proc = subprocess.Popen(
        run.argv,
        cwd=str(run.cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    try:
        out, _ = proc.communicate(timeout=run.timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            out = ""
        return None, out or ""
    return proc.returncode, out or ""


#: The pytest counts that mean a test was collected and reached an outcome.
COLLECTED = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")


def collected(res: TierResult) -> int:
    """The tests a tier collected, from its summary counts and its parsed node ids."""
    return sum(res.counts.get(k, 0) for k in COLLECTED) or len(res.passed | res.failed)


def _tail(out: str, label: str, n: int = 30) -> str:
    lines = out.rstrip().splitlines()[-n:]
    return f"--- {label} (last {len(lines)} lines)\n" + "\n".join(lines) + "\n"


def pytest_tier(
    tier: str,
    runs: list[Run],
    known: dict[str, dict[str, str]],
    *,
    cap: int,
    counter: Any = python_count,
) -> TierResult:
    """A tier made of pytest runs, one after another, merged and compared.

    Exit 0 and 1 are read from the summary. Exit 5 (no test collected) puts the run in
    ``not_run``. Any other exit, or an exit 1 with no failure parsed, is an error of the tier,
    with the run's last lines kept in ``tail``. A tier that fails on neither count but has a
    run in ``not_run``, or collected no test over all its runs, is recorded as ``not-run``,
    never as a pass.
    """
    res = TierResult(tier)
    start = time.monotonic()
    for run in runs:
        code, out = execute(run, cap=cap, counter=counter)
        passed, failed, counts = parse_pytest(out, run.base)
        res.passed |= passed
        res.failed |= failed
        res.counts.update(counts)
        if code is None:
            res.timed_out.append(f"{run.label} (over {run.timeout:g} s)")
            res.tail += _tail(out, run.label)
        elif code == 5:
            res.not_run.append(run.label)
            res.tail += _tail(out, run.label)
        elif code not in (0, 1) or (code == 1 and not failed):
            res.errors.append(f"{run.label}: pytest exit {code} with no failure parsed")
            res.tail += _tail(out, run.label)
        elif failed:
            res.tail += _tail(out, run.label)
    res.duration_s = round(time.monotonic() - start, 1)
    res = compare(res, known)
    if res.ok and (res.not_run or not collected(res)):
        res.ok, res.status = False, "not-run"
        res.notes.append(
            f"{', '.join(res.not_run)}: pytest collected no test (exit 5)"
            if res.not_run
            else "the tier collected no test"
        )
    return res


def harness_dirs(root: Path) -> list[Path]:
    """The test folders of the harness.

    A folder counts when it is named ``tests``, sits under ``.claude/`` or ``tools/`` outside
    a cache, and holds a ``test_*.py``.
    """
    found = set()
    for base in (root / ".claude", root / "tools"):
        if not base.is_dir():
            continue
        for f in base.rglob("test_*.py"):
            parts = f.relative_to(root).parts
            if f.parent.name == "tests" and not any(
                p in (".cache", "__pycache__", "node_modules") for p in parts
            ):
                found.add(f.parent)
    return sorted(found)


def build_runs(
    tier: str, root: Path, python: str, *, suite_timeout: float
) -> list[Run]:
    """The pytest runs of a harness or model tier, each started from ``root``."""
    base = [python, "-m", "pytest"]
    opts = list(PYTEST_OPTS)
    if tier == "harness":
        return [
            Run(
                d.relative_to(root).as_posix(),
                [*base, d.relative_to(root).as_posix(), *opts],
                root,
                suite_timeout,
            )
            for d in harness_dirs(root)
        ]
    if tier in ("tests", "slides"):
        pp = os.pathsep.join(
            p for p in (str(root / C.LIB_DIR), os.environ.get("PYTHONPATH")) if p
        )
        marker = "not slides" if tier == "tests" else "slides"
        return [
            Run(
                tier,
                [*base, "tests", *opts, "-m", marker],
                root,
                suite_timeout,
                {"PYTHONPATH": pp},
            )
        ]
    raise ConfigError(f"no pytest runs for tier {tier!r}")


def changed_python_files(root: Path) -> list[str]:
    """The Python files that git reports as modified, added, renamed or untracked.

    Deleted files and the paths outside the lint (``LINT_SKIP``) are left out.
    """
    out = C.git(str(root), "status", "--porcelain", "-z", "--untracked-files=all")
    files = []
    items = out.split("\0")
    i = 0
    while i < len(items):
        item = items[i]
        i += 1
        if len(item) < 4:
            continue
        code, path = item[:2], item[3:]
        if "R" in code or "C" in code:
            i += 1  # the original path of a rename or copy follows
        if "D" in code or not path.endswith(".py"):
            continue
        if path.startswith(LINT_SKIP):
            continue
        files.append(path)
    return sorted(set(files))


def _ruff_counts(
    root: Path, ruff: list[str], args: list[str], stdin: str | None = None
) -> Counter:
    """The findings of one ruff call, counted by (repository path, rule)."""
    r = subprocess.run(
        [
            *ruff,
            "check",
            "--config",
            "ruff.toml",
            "--force-exclude",
            "--no-cache",
            "--exit-zero",
            "--output-format",
            "json",
            *args,
        ],
        input=stdin,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(root),
        timeout=600,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ruff check failed: {r.stderr.strip()[:300]}")
    counts: Counter = Counter()
    for f in json.loads(r.stdout or "[]"):
        name = Path(f["filename"])
        rel = (name.relative_to(root) if name.is_absolute() else name).as_posix()
        counts[(rel, f.get("code") or "syntax-error")] += 1
    return counts


#: Concurrent ruff calls over the HEAD versions. Ruff is not a Python process in the sense of
#: the ceiling, so these do not count against it.
HEAD_WORKERS = 8


def head_counts(root: Path, ruff: list[str], paths: list[str]) -> Counter:
    """The ruff findings of each path's version at HEAD, counted by (path, rule).

    The versions are read in one ``git cat-file --batch`` (``_common.git_blobs``) and each is
    linted from standard input under its own path, so its per-file ignores apply, with up to
    ``HEAD_WORKERS`` ruff calls at once. A path HEAD does not hold, or holds empty, has no
    findings.
    """
    blobs = C.git_blobs(str(root), "HEAD", paths)
    texts = [
        (path, blob.decode("utf-8", "replace"))
        for path, blob in zip(paths, blobs, strict=True)
        if blob
    ]
    before: Counter = Counter()
    if not texts:
        return before
    with ThreadPoolExecutor(max_workers=min(HEAD_WORKERS, len(texts))) as pool:
        for counts in pool.map(
            lambda item: _ruff_counts(
                root, ruff, ["--stdin-filename", item[0], "-"], item[1]
            ),
            texts,
        ):
            before.update(counts)
    return before


def mypy_findings(code: int | None, out: str) -> tuple[int | None, str]:
    """The mypy count of a run of ``tools/run_mypy.py`` over named modules, and its note.

    The driver prints ``<n> named, <e> errors`` once per pass and never mypy's own ``Found``
    line, so the count is the sum of ``<e>`` over those lines. Where no such line is printed
    the ``: error:`` lines are counted. A run past its time limit, or an exit of 2 or more (a
    usage or crash report), has no count, and its note says so.
    """
    if code is None:
        return None, "mypy error: the run passed its time limit; no count"
    if code >= 2:
        return None, f"mypy error: exit {code}, a usage or crash report; no count"
    per_pass = _MYPY_PASS.findall(out)
    if per_pass:
        n = sum(int(e) for e in per_pass)
    else:
        n = sum(": error:" in line for line in out.splitlines())
    return n, f"mypy exit {code}, {n} findings (recorded only)"


def lint_tier(
    root: Path,
    python: str,
    *,
    cap: int,
    mypy: bool = True,
    counter: Any = python_count,
    files: list[str] | None = None,
    ruff: list[str] | None = None,
) -> TierResult:
    """The lint of the changed files against their versions at HEAD.

    A file fails when any rule finds more in the working file than in its HEAD version, a
    new file against none. The mypy count of the same files is recorded as a note.
    """
    res = TierResult("lint")
    start = time.monotonic()
    files = changed_python_files(root) if files is None else files
    res.counts["files"] = len(files)
    if ruff is None and files:
        ruff = ruff_argv(python)
    if not files:
        res.notes.append("no changed Python file")
    elif ruff is None:
        res.errors.append("ruff not found (CODE_CHECK_RUFF, PATH, or `python -m ruff`)")
    else:
        now: Counter = Counter()
        for k in range(0, len(files), 100):
            now.update(_ruff_counts(root, ruff, files[k : k + 100]))
        before = head_counts(root, ruff, sorted({path for path, _ in now}))
        rose = sorted(
            f"{path}: {code} {before[(path, code)]} -> {n}"
            for (path, code), n in now.items()
            if n > before[(path, code)]
        )
        res.counts["findings"] = sum(now.values())
        res.counts["findings_at_head"] = sum(before.values())
        if rose:
            res.errors.append(f"{len(rose)} rules found more than at HEAD")
            res.tail = "\n".join(rose) + "\n"
    run_mypy = root / "tools" / "run_mypy.py"
    targets = [f for f in files if "/tests/" not in f"/{f}"]
    if mypy and targets and run_mypy.is_file():
        code, out = execute(
            Run("mypy", [python, str(run_mypy), *targets], root, SUITE_TIMEOUT),
            cap=cap,
            counter=counter,
        )
        n, note = mypy_findings(code, out)
        if n is not None:
            res.counts["mypy"] = n
        res.notes.append(note)
    elif mypy and targets:
        res.notes.append("tools/run_mypy.py is absent; mypy not run")
    elif targets:
        res.notes.append("mypy not run; --mypy or --full records its count")
    res.duration_s = round(time.monotonic() - start, 1)
    if res.errors:
        res.ok, res.status = False, "fail"
    return res


def fingerprint(root: Path) -> dict[str, Any]:
    """The library state a ledger line carries and ``--status`` matches.

    ``lib_state`` hashes the diff, the status and the content of each untracked file under
    ``lib/``, so a fresh copy of the library changes the fingerprint whether or not it is
    committed. ``manifest_sha256`` is the hash of ``lib/MANIFEST.md``, empty where it is absent.
    """
    dirs = list(LIB_DIRS)
    dirty, sha = C.lib_state(str(root), dirs)
    _, manifest = C.manifest_hash(str(root))
    return {
        "lib_commit": C.git(str(root), "log", "-1", "--format=%H", "--", *dirs).strip(),
        "lib_diff_sha256": sha,
        "manifest_sha256": manifest or "",
        "lib_dirty": dirty,
        "head": C.git(str(root), "rev-parse", "HEAD").strip(),
    }


def ledger_line(
    res: TierResult, fp: dict[str, Any], python: str, mode: str
) -> dict[str, Any]:
    """The JSON line one tier appends to the ledger.

    The interpreter is recorded by its release and its file name, never by its path.
    """
    return {
        "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tier": res.tier,
        "ok": res.ok,
        "status": res.status,
        "mode": mode,
        "counts": dict(res.counts),
        "new_failures": res.new_failures,
        "known_failures": res.known_failures,
        "known_now_passing": res.known_now_passing,
        "known_not_run": res.known_not_run,
        "timed_out": res.timed_out,
        "not_run": res.not_run,
        "errors": res.errors,
        "notes": res.notes,
        "duration_s": res.duration_s,
        "python": Path(python).name,
        **fp,
    }


def append_ledger(line: dict[str, Any], path: Path = LEDGER) -> None:
    """One line appended to the ledger, the folder made where absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def summary(res: TierResult) -> str:
    """One line per tier: name, verdict, counts and time."""
    order = ("passed", "failed", "errors", "skipped", "deselected", "files", "findings")
    parts = [f"{res.counts[k]} {k}" for k in order if res.counts.get(k)]
    if res.known_failures:
        parts.append(f"{len(res.known_failures)} known")
    verdict = {
        "pass": "PASS",
        "fail": "FAIL",
        "ceiling": "REFUSED",
        "not-run": "NOT RUN",
    }[res.status]
    return f"{res.tier:<8} {verdict:<8} {', '.join(parts) or '-'}  {res.duration_s:g} s"


def details(res: TierResult) -> list[str]:
    """The lines that explain a verdict: new, stale and unrun entries, timeouts and errors."""
    out = [f"  new failure: {i}" for i in res.new_failures]
    out += [
        f"  now passes, remove from known_failures.toml: {i}"
        for i in res.known_now_passing
    ]
    out += [f"  known, did not run: {i}" for i in res.known_not_run]
    out += [f"  timed out: {i}" for i in res.timed_out]
    out += [f"  collected no test: {i}" for i in res.not_run]
    out += [f"  error: {e}" for e in res.errors]
    out += [f"  note: {n}" for n in res.notes]
    return out


def select_tiers(args: argparse.Namespace) -> tuple[list[str], str]:
    """The tiers named on the command line, in run order, and the mode's name."""
    if args.tiers:
        names = [t.strip() for t in args.tiers.split(",") if t.strip()]
        unknown = [t for t in names if t not in TIERS]
        if unknown:
            raise ConfigError(
                f"unknown tier {', '.join(unknown)}; the tiers are {', '.join(TIERS)}"
            )
        return [t for t in TIERS if t in names], "tiers"
    if args.full:
        return list(TIERS), "full"
    return list(QUICK), "quick"


def status(
    root: Path, tiers: list[str], ledger: Path = LEDGER
) -> tuple[bool, list[str]]:
    """The newest ledger line per tier at the present fingerprint, and whether all pass.

    A line matches when each key of ``MATCH_KEYS`` equals the present one. A line written
    before ``manifest_sha256`` was recorded matches nothing.
    """
    fp = fingerprint(root)
    newest: dict[str, dict[str, Any]] = {}
    if ledger.is_file():
        for raw in ledger.read_text(encoding="utf-8").splitlines():
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if all(line.get(k) == fp[k] for k in MATCH_KEYS):
                newest[line.get("tier", "")] = line
    lines = [
        f"fingerprint {fp['lib_commit'][:9] or 'uncommitted'} {fp['lib_diff_sha256'][:12]}"
        f" {fp['manifest_sha256'][:12] or 'no-manifest'}{' (dirty)' if fp['lib_dirty'] else ''}"
    ]
    ok = True
    for t in tiers:
        line = newest.get(t)
        if line is None:
            ok = False
            lines.append(f"{t:<8} no run at this fingerprint")
        else:
            ok &= bool(line.get("ok"))
            word = (
                "PASS"
                if line.get("ok")
                else "NOT RUN"
                if line.get("status") == "not-run"
                else "FAIL"
            )
            lines.append(f"{t:<8} {word} at {line.get('time')}")
    return ok, lines


#: How a stopping tier is named in the line of each later tier, ``<tier>: not run (...)``.
STOP_REASON = {
    "fail": "failed",
    "ceiling": "refused at the process ceiling",
    "not-run": "collected no test",
}


def run_tiers(
    tiers: list[str],
    *,
    root: Path = ROOT,
    python: str,
    mode: str,
    known: dict[str, dict[str, str]],
    suite_timeout: float = SUITE_TIMEOUT,
    mypy: bool = True,
    ledger: Path = LEDGER,
    counter: Any = python_count,
    cap: int | None = None,
    runs: dict[str, list[Run]] | None = None,
    emit: Any = print,
) -> int:
    """The tiers run in order, each summarised and recorded, stopping at the first failure.

    ``runs`` replaces the pytest runs of named tiers, which the tests use for a fake tier.
    """
    cap = ceiling() if cap is None else cap
    for tier in tiers:
        try:
            if tier == "lint" and not (runs and tier in runs):
                res = lint_tier(root, python, cap=cap, mypy=mypy, counter=counter)
            else:
                plan = (runs or {}).get(tier) or build_runs(
                    tier, root, python, suite_timeout=suite_timeout
                )
                res = pytest_tier(tier, plan, known, cap=cap, counter=counter)
        except CeilingError as exc:
            res = TierResult(tier, ok=False, status="ceiling", errors=[str(exc)])
        append_ledger(ledger_line(res, fingerprint(root), python, mode), ledger)
        emit(summary(res))
        for line in details(res):
            emit(line)
        if not res.ok:
            if res.tail:
                emit(res.tail.rstrip())
            why = STOP_REASON[res.status]
            for later in tiers[tiers.index(tier) + 1 :]:
                emit(f"{later}: not run ({tier} {why})")
            return 3 if res.status == "ceiling" else 1
    return 0


def mypy_wanted(args: argparse.Namespace, mode: str) -> bool:
    """Whether the lint tier records the mypy count.

    It does with ``--mypy``, and with ``--full`` unless ``--no-mypy`` is given. The count
    never decides the tier, and it is the cost of the tier, so ``--quick`` and ``--tiers``
    leave it out.
    """
    if args.no_mypy:
        return False
    return bool(args.mypy) or mode == "full"


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", action="store_true", help="lint, harness and tests (the default)"
    )
    group.add_argument("--full", action="store_true", help="all four tiers")
    group.add_argument("--tiers", help=f"a comma list of {', '.join(TIERS)}")
    ap.add_argument(
        "--status", action="store_true", help="report the ledger, run nothing"
    )
    ap.add_argument("--python", help="the interpreter of the runs (else LAB_PY)")
    ap.add_argument("--suite-timeout", type=float, default=SUITE_TIMEOUT)
    mypy_group = ap.add_mutually_exclusive_group()
    mypy_group.add_argument(
        "--mypy",
        action="store_true",
        help="record the mypy count in the lint tier (the default with --full)",
    )
    mypy_group.add_argument(
        "--no-mypy", action="store_true", help="skip the mypy count, with --full too"
    )
    args = ap.parse_args(argv)
    with contextlib.suppress(ValueError, AttributeError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # a TextIOWrapper on every platform the repository runs on
    try:
        tiers, mode = select_tiers(args)
        if args.status:
            ok, lines = status(ROOT, tiers)
            print("\n".join(lines))  # noqa: T201 - a report tool
            return 0 if ok else 1
        known = load_known()
    except ConfigError as exc:
        print(f"check: {exc}", file=sys.stderr)  # noqa: T201 - a report tool
        return 2
    python = args.python or os.environ.get("LAB_PY") or sys.executable
    print(f"check: {mode} ({', '.join(tiers)}) under {python}", flush=True)  # noqa: T201 - a report tool
    return run_tiers(
        tiers,
        python=python,
        mode=mode,
        known=known,
        suite_timeout=args.suite_timeout,
        mypy=mypy_wanted(args, mode),
        emit=lambda s: print(s, flush=True),  # noqa: T201 - a report tool
    )


if __name__ == "__main__":
    sys.exit(main())
