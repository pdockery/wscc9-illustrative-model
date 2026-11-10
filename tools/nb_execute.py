#!/usr/bin/env python
"""Stamped headless execution of notebooks.

Usage: python tools/nb_execute.py NOTEBOOK... [--timeout SECONDS] [--root DIR] [--no-prose-lint]

Runs the house re-run, `python -m nbconvert --to notebook --execute --inplace`, under the
interpreter that invokes this script. The Matplotlib backend variable is never set, because a
non-inline backend strips the stored figures; the script refuses to run when it names Agg.

Pre-flight. Before any notebook starts, every code cell of every named notebook is compiled
to an AST. A cell magic is skipped, and a line magic or shell escape (`%...`, `!...`,
`x = !...`) is masked when the cell fails to compile as it stands. A cell that still fails
refuses the whole launch, naming the notebook, the cell id and the line. prose_lint.py then
lints each notebook and its findings are reported; they never refuse a launch, and
--no-prose-lint skips them. Last, the running interpreter is compared with tools/env_lock.json
(tools/lab_env.py), and a difference is a warning.

On success the notebook's top-level metadata receives metadata["lab_execution"]:

  date            UTC ISO 8601 time the stamp was written
  python          interpreter file name and version
  lib_commit      `git rev-parse HEAD` of the repository
  lib_dirty       true when lib/ has uncommitted changes
  lib_diff_sha256 hash of `git diff HEAD`, the porcelain status and each untracked file's
                  sha256 over lib/
  lib_manifest    the path and sha256 of lib/MANIFEST.md, which records each module's
                  SHA-256 and the date of the copy
  lib_changed_during_run  present and true when the lib state at the stamp differs from
                  the lib state at launch
  packages        the version of each model package of tools/env_lock.json and of nbconvert,
                  nbclient and ipykernel (importlib.metadata), null for one not installed
  kernel          the notebook's kernelspec name
  host            the operating system, its release and the machine type
  command         the nbconvert command line, with the notebook's repository-relative path

The stamp holds no absolute path, no host name and no environment variable. A notebook
outside the repository, run with --root, is named in the command by its file name alone.
The packages are those of this interpreter; the kernel resolves to this interpreter's
environment when the kernelspec is installed there.

Cells and outputs are untouched by the stamp. On failure no stamp is written; the failing cell
and the exception are read from nbconvert's error report and printed. Every attempt, refused,
failed or stamped, appends one JSON line to tools/.cache/nb_runs.jsonl under the repository:
notebook, start, end, seconds, exit, outcome, failing cell, the lib state at launch, host and
command. After each stamp the staleness summary tools/.cache/staleness.json is rewritten
(nb_staleness.refresh_cache).

The repository is the one that holds the notebook. --root names it for a notebook outside
every repository, such as a scratch copy, and the stamp then records that repository's lib/.

The script refuses to start when the count of running Python interpreter processes reaches
the ceiling of .claude/hooks/_limits.py (seven unless that module sets another). Standard
library plus nbconvert.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100, PTH120 - the tools folder on sys.path
import _common as C  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

TOOLS = Path(__file__).resolve().parent
PROSE_LINT = TOOLS.parent / ".claude/skills/writing-voice/scripts/prose_lint.py"
LIMITS = TOOLS.parent / ".claude/hooks/_limits.py"
LOCK = TOOLS / "env_lock.json"
#: The ceiling used when .claude/hooks/_limits.py cannot be loaded.
DEFAULT_CEILING = 7

#: The model packages a stamp records when tools/env_lock.json is absent or unreadable.
MODEL_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pypsa",
    "pandapower",
    "pycirclize",
)
#: The execution packages every stamp records beside the model packages.
EXECUTION_PACKAGES = ("nbconvert", "nbclient", "ipykernel")

_IPYTHON_LINE = re.compile(r"^(\s*)(?:[!%]|[\w.]+\s*=\s*[!%])")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EXCEPTION_LINE = re.compile(r"^([A-Z][\w.]*)(?::\s?(.*))?$")
_RULE = re.compile(r"^-{18,}$")
TAIL_LINES = 2000  # stderr lines kept for the failure report
LEDGER_TAIL = 40  # stderr lines kept in a failed attempt's ledger record


# ---- process ceiling --------------------------------------------------------------------


def _limits() -> ModuleType | None:
    """The hooks' _limits module, loaded from its file, or None when it cannot be loaded."""
    try:
        spec = importlib.util.spec_from_file_location("_limits", LIMITS)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]  # spec is None for an unloadable path; the except below returns None
        spec.loader.exec_module(mod)  # type: ignore[union-attr]  # spec or loader None raises AttributeError, caught below
    except Exception:
        return None
    return mod


def ceiling() -> int:
    """The process ceiling of .claude/hooks/_limits.py, or DEFAULT_CEILING without it."""
    mod = _limits()
    value = getattr(mod, "CEILING", None) if mod is not None else None
    return value if isinstance(value, int) and value > 0 else DEFAULT_CEILING


def python_count() -> int | None:
    """The count of running Python interpreters from _limits.py; None when unavailable."""
    mod = _limits()
    try:
        return mod.python_count() if mod is not None else None
    except Exception:
        return None


# ---- the stamp --------------------------------------------------------------------------


def write_stamp(path: str, stamp: dict[str, Any]) -> None:
    """Set metadata[lab_execution], keeping nbformat's layout (indent 1, sorted keys, UTF-8) and the file's line endings."""
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw[:20000]
    ascii_only = raw.isascii()  # a notebook saved with \u escapes stays escaped; otherwise UTF-8 as nbformat writes
    nb = json.loads(raw.decode("utf-8"))
    nb.setdefault("metadata", {})[C.STAMP_KEY] = stamp
    text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=ascii_only) + "\n"
    data = text.replace("\n", "\r\n" if crlf else "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)


def utc_now() -> str:
    """The present UTC time in ISO 8601 to the second."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def shown_path(nb_path: str, root: str) -> str:
    """The notebook's path relative to root with forward slashes, or its absolute path outside root."""
    try:
        rel = os.path.relpath(nb_path, root)
    except ValueError:  # another drive
        rel = os.pardir
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        # abspath normalizes the path without resolving a link, as the report shows it
        rel = os.path.abspath(nb_path)  # noqa: PTH100
    return rel.replace("\\", "/")


def stamp_path(nb_path: str, root: str) -> str:
    """The notebook as the stamp names it: relative to root, or its file name outside root."""
    shown = shown_path(nb_path, root)
    return Path(nb_path).name if Path(shown).is_absolute() else shown


def model_packages(lock_path: Path = LOCK) -> tuple[str, ...]:
    """The packages whose role is "model" in the lock, or MODEL_PACKAGES without a readable lock."""
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        names = tuple(
            sorted(
                name
                for name, spec in (lock.get("packages") or {}).items()
                if isinstance(spec, dict) and spec.get("role") == "model"
            )
        )
    except (OSError, ValueError, AttributeError):
        return MODEL_PACKAGES
    return names or MODEL_PACKAGES


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """The installed version of each package, None for a package that is not installed."""
    return {name: _version(name) for name in names}


def _version(name: str) -> str | None:
    """The installed version of one package, or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def host_state() -> dict[str, str]:
    """The operating system, its release and the machine type; never the host name."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def interpreter() -> str:
    """The running interpreter's file name and version, without its directory."""
    return f"{Path(sys.executable).name} {platform.python_version()}"


def kernel_name(nb_path: str) -> str | None:
    """The kernelspec name in a notebook's metadata, or None."""
    try:
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return ((nb.get("metadata") or {}).get("kernelspec") or {}).get("name")


def lib_snapshot(root: str) -> dict[str, Any]:
    """The lib state of the repository at root: its HEAD, whether lib/ is dirty, and its hash."""
    dirty, h = C.lib_state(root, C.lib_dirs())
    return {
        "lib_commit": C.git(root, "rev-parse", "HEAD").strip(),
        "lib_dirty": dirty,
        "lib_diff_sha256": h,
    }


def build_stamp(root: str, nb_path: str, command: str) -> dict[str, Any]:
    """The lab_execution stamp of a run in root, with lib/ read as it stands now."""
    rel, mh = C.manifest_hash(root)
    stamp: dict[str, Any] = {"date": utc_now(), "python": interpreter()}
    stamp.update(lib_snapshot(root))
    stamp["command"] = command
    if mh:
        stamp["lib_manifest"] = {"path": rel, "sha256": mh}
    stamp["packages"] = package_versions(model_packages() + EXECUTION_PACKAGES)
    stamp["kernel"] = kernel_name(nb_path)
    stamp["host"] = host_state()
    return stamp


# ---- pre-flight -------------------------------------------------------------------------


def compile_cell(src: str) -> SyntaxError | None:
    """The SyntaxError a cell's source raises when compiled, or None when it compiles."""
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(src, "<cell>", "exec", flags)
    except SyntaxError as e:
        return e
    except ValueError as e:  # a null byte in the source
        return SyntaxError(str(e))
    return None


def mask_ipython(src: str) -> str:
    """The source with each line magic and shell escape replaced by `pass` at its indentation."""
    lines = src.split("\n")
    for i, line in enumerate(lines):
        m = _IPYTHON_LINE.match(line)
        if m:
            lines[i] = m.group(1) + "pass"
    return "\n".join(lines)


def cell_source(cell: dict[str, Any]) -> str:
    """A cell's source as one string, whether stored as a string or a list of lines."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def syntax_problems(nb_path: str) -> list[dict[str, Any]]:
    """Every code cell of a notebook that does not compile, with its id, index, line and message.

    A cell magic (`%%` on the first line) is skipped, because its body need not be Python. A
    cell that fails as it stands is compiled again with its line magics and shell escapes
    masked, and only a cell that fails both ways is reported, with the masked error.
    """
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    out = []
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if src.lstrip().startswith("%%") or compile_cell(src) is None:
            continue
        err = compile_cell(mask_ipython(src))
        if err is None:
            continue
        out.append(
            {
                "id": cell.get("id"),
                "index": index,
                "line": err.lineno,
                "message": err.msg or str(err),
            }
        )
    return out


def readable_problems(nb: str) -> list[dict[str, Any]]:
    """syntax_problems of one notebook, with an unreadable notebook reported as a problem."""
    try:
        return syntax_problems(nb)
    except (OSError, ValueError) as e:
        return [{"id": None, "index": None, "line": None, "message": str(e)}]


def prose_findings(nb_path: str) -> dict[str, Any] | None:
    """prose_lint.py's JSON report on one notebook, or None when the lint cannot run."""
    if not PROSE_LINT.is_file():
        return None
    try:
        r = subprocess.run(
            [sys.executable, str(PROSE_LINT), "--json", nb_path],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=300,
            cwd=str(TOOLS.parent),
        )
        return json.loads(r.stdout) if r.returncode in (0, 1) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def report_prose(rel: str, report: dict[str, Any] | None) -> None:
    """Print a one-line summary of the prose lint and its first findings."""
    if report is None:
        print(f"nb_execute: prose_lint did not run on {rel}", file=sys.stderr)
        return
    errors, warns = report.get("errors", 0), report.get("warnings", 0)
    if not errors and not warns:
        return
    print(
        f"nb_execute: prose_lint reports {errors} error(s) and {warns} warning(s) in "
        f"{rel}; the run goes ahead",
        file=sys.stderr,
    )
    for f in report.get("findings", [])[:8]:
        where = f.get("cell", f.get("line", ""))
        print(
            f"  {f.get('severity', '')} {f.get('rule', '')} {where}: "
            f"{str(f.get('message', ''))[:160]}",
            file=sys.stderr,
        )


def check_environment() -> list[str] | None:
    """tools/lab_env.py's comparison of this interpreter with the lock; None when it cannot run.

    lab_env warns and never raises, and a failure to import it is reported on one line.
    """
    try:
        import lab_env  # noqa: PLC0415 - a sibling tool, loaded only for the pre-flight

        return list(lab_env.check_environment())
    except Exception as e:
        print(f"nb_execute: environment check did not run ({e!r})", file=sys.stderr)
        return None


# ---- execution and failure capture ------------------------------------------------------


def run_streaming(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    """Run a command, echoing its stderr as it arrives, and return its exit and stderr tail."""
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    with subprocess.Popen(
        cmd, env=env, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE
    ) as p:
        assert p.stderr is not None  # noqa: S101 - stderr=PIPE guarantees it
        for raw in p.stderr:
            line = raw.decode("utf-8", "replace")
            sys.stderr.write(line)
            sys.stderr.flush()
            tail.append(line)
        rc = p.wait()
    return rc, "".join(tail)


def _between_rules(lines: list[str], start: int) -> tuple[str | None, int]:
    """The text between the first two rule lines at or after start, and the index after it."""
    rules = [k for k in range(start, len(lines)) if _RULE.match(lines[k].strip())]
    if len(rules) < 2:
        return None, start
    return "\n".join(lines[rules[0] + 1 : rules[1]]), rules[1] + 1


def match_cell(
    nb_path: str, source: str | None, prefix: bool = False
) -> dict[str, Any]:
    """The id and index of the code cell whose source is the given text, or its first lines."""
    if not source:
        return {}
    want = source.strip()
    try:
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        got = cell_source(cell).strip()
        if got == want or (prefix and want and got.startswith(want)):
            return {"id": cell.get("id"), "index": index}
    return {}


def failing_cell(stderr_text: str, nb_path: str) -> dict[str, Any] | None:
    """The failing cell and exception read from nbconvert's error report, or None.

    nbclient's CellExecutionError prints the cell's source between two rule lines, then the
    kernel's traceback, whose last "Name: message" line is the exception. A timeout prints a
    preview of the cell's first lines, and a dead kernel names no cell.
    """
    lines = _ANSI.sub("", stderr_text).splitlines()
    out: dict[str, Any] = {}
    marks = [k for k, x in enumerate(lines) if "An error occurred while executing" in x]
    timeout = [k for k, x in enumerate(lines) if "Here is a preview of the cell" in x]
    if marks:
        source, after = _between_rules(lines, marks[-1])
        out.update(match_cell(nb_path, source))
        exc = [m for x in lines[after:] if (m := _EXCEPTION_LINE.match(x.strip()))]
        if exc:
            out["exception"] = exc[-1].group(0)
        if source and source.strip():
            out["source_head"] = source.strip().splitlines()[0][:160]
    elif timeout:
        source, _ = _between_rules(lines, timeout[-1])
        out.update(match_cell(nb_path, source, prefix=True))
        out["exception"] = "CellTimeoutError"
    else:
        named = [x.strip() for x in lines if "DeadKernelError" in x]
        if named:
            out["exception"] = named[-1][:300]
    return out or None


# ---- run ledger and staleness summary ---------------------------------------------------


def ledger_path(root: str) -> Path:
    """The run ledger of the repository at root, under its ignored tools/.cache."""
    return C.cache_file(root, "nb_runs.jsonl")


def append_run(root: str, record: dict[str, Any]) -> None:
    """Append one attempt to the run ledger; a failed write is reported and not raised."""
    path = ledger_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as e:
        print(f"nb_execute: run ledger not written ({e})", file=sys.stderr)


def refresh_staleness(root: str) -> None:
    """Rewrite the staleness summary of the repository at root; a failure is reported only."""
    try:
        import nb_staleness  # noqa: PLC0415 - a sibling tool, loaded only after a stamp

        nb_staleness.refresh_cache(root)
    except Exception as e:
        print(f"nb_execute: staleness summary not refreshed ({e!r})", file=sys.stderr)


def root_or_none(nb: str, root: str | None = None) -> str | None:
    """The repository named by --root, or the one that holds the notebook, or None."""
    try:
        return C.repo_root(root or nb)
    except RuntimeError:
        return None


def enclosing_repository(nb: str) -> str | None:
    """The nearest directory above a notebook that holds a .git entry, found without git."""
    for parent in Path(nb).resolve().parents:
        if (parent / ".git").exists():
            return str(parent)
    return None


def named(nb: str, root: str | None) -> str:
    """A notebook as a message names it: relative to --root or its repository, else its file name."""
    base = root or enclosing_repository(nb)
    return shown_path(nb, base) if base else Path(nb).name


def refuse_all(
    notebooks: list[str], root: str | None, outcome: str, message: str
) -> int:
    """Write a refusal to stderr and to the run ledger, and return exit 2.

    The ledger is that of --root when it is given, and otherwise that of the repository that
    holds each notebook, found from its .git entry. No git process is started, so a refusal at
    the process ceiling starts no process. A notebook outside every repository has no ledger.
    """
    sys.stderr.write(message)
    now = utc_now()
    for nb in notebooks:
        base = root or enclosing_repository(nb)
        if base is None:
            continue
        append_run(
            base,
            {
                "notebook": shown_path(nb, base),
                "start": now,
                "end": now,
                "exit": 2,
                "outcome": outcome,
                "host": host_state(),
            },
        )
    return 2


def preflight(notebooks: list[str], lint: bool, root: str | None = None) -> int:
    """Refuse the launch (exit 2) when a code cell does not compile; report the prose lint and the environment."""
    problems = [(nb, p) for nb in notebooks for p in readable_problems(nb)]
    if problems:
        text = "".join(
            f"nb_execute: {named(nb, root)} cell {p['id']} "
            f"(index {p['index']}) line {p['line']}: {p['message']}\n"
            for nb, p in problems
        )
        return refuse_all(
            sorted({nb for nb, _ in problems}),
            root,
            "refused: syntax",
            text + "nb_execute: no notebook was started.\n",
        )
    if lint:
        for nb in notebooks:
            report_prose(named(nb, root), prose_findings(nb))
    check_environment()
    return 0


def execute_one(nb: str, root: str, timeout: int | None, env: dict[str, str]) -> int:
    """Run one notebook, stamp it on success, and append the attempt to the run ledger."""
    args = ["-m", "nbconvert", "--to", "notebook", "--execute", "--inplace"]
    cmd = [sys.executable, *args, nb]
    if timeout:
        cmd.insert(-1, f"--ExecutePreprocessor.timeout={timeout:d}")
    rel = shown_path(nb, root)
    shown = " ".join(["python", *args, stamp_path(nb, root)])
    record: dict[str, Any] = {"notebook": rel, "command": shown, "host": host_state()}
    launch = lib_snapshot(root)
    record.update(launch)
    record["python"] = interpreter()
    record["start"] = utc_now()
    t0 = time.time()
    rc, tail = run_streaming(cmd, env)
    record.update(end=utc_now(), seconds=round(time.time() - t0, 1), exit=rc)
    if rc != 0:
        cell = failing_cell(tail, nb)
        record.update(outcome="failed", failing_cell=cell)
        record["stderr_tail"] = _ANSI.sub("", tail).splitlines()[-LEDGER_TAIL:]
        where = ""
        if cell:
            exc = cell.get("exception", "exception not found")
            where = f" at cell {cell.get('id')} (index {cell.get('index')}): {exc}"
        sys.stderr.write(
            f"nb_execute: execution failed for {rel} (exit {rc}){where}; "
            "no stamp written.\n"
        )
        append_run(root, record)
        return 1
    stamp = build_stamp(root, nb, shown)
    if any(stamp[k] != launch[k] for k in ("lib_commit", "lib_diff_sha256")):
        stamp["lib_changed_during_run"] = True
        record["lib_changed_during_run"] = True
    write_stamp(nb, stamp)
    record["outcome"] = "stamped"
    append_run(root, record)
    print(f"stamped {rel}")
    refresh_staleness(root)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Check the ceiling, the backend and the pre-flight, then run and stamp each notebook."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="per-cell timeout in seconds (nbconvert default otherwise)",
    )
    ap.add_argument(
        "--ceiling",
        type=int,
        default=None,
        help="the process ceiling (default: .claude/hooks/_limits.py)",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="the repository whose lib/ the stamp records (default: the notebook's own)",
    )
    ap.add_argument(
        "--no-prose-lint",
        action="store_true",
        help="skip the prose lint of the pre-flight (the syntax check always runs)",
    )
    a = ap.parse_args(argv)
    notebooks = [os.path.abspath(nb) for nb in a.notebooks]  # noqa: PTH100 - abspath keeps a link as named
    limit = a.ceiling if a.ceiling is not None else ceiling()
    n = python_count()
    if n is not None and n >= limit:
        return refuse_all(
            notebooks,
            a.root,
            "refused: ceiling",
            f"nb_execute: {n} Python interpreter processes are running (ceiling "
            f"{limit}). Wait for them to finish; no notebook was started.\n",
        )
    env = dict(os.environ)
    if env.get("MPLBACKEND", "").lower() == "agg":
        return refuse_all(
            notebooks,
            a.root,
            "refused: Agg backend",
            "nb_execute: the Matplotlib backend variable names Agg, which strips "
            "stored figures. Unset it.\n",
        )
    env.pop("MPLBACKEND", None)
    if preflight(notebooks, lint=not a.no_prose_lint, root=a.root):
        return 2
    rc = 0
    for nb in notebooks:
        root = root_or_none(nb, a.root)
        if root is None:
            sys.stderr.write(
                f"nb_execute: {nb} is not inside a git repository; skipped\n"
            )
            rc = 1
            continue
        rc = max(rc, execute_one(nb, root, a.timeout, env))
    return rc


if __name__ == "__main__":
    sys.exit(main())
