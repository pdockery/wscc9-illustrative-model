#!/usr/bin/env python
"""PostToolUse hook: runs the repository's code lint on a .py file or notebook cell an agent has just written or edited.

Claude Code runs this after every Write, Edit, MultiEdit or NotebookEdit call, with the call's
JSON on stdin (https://code.claude.com/docs/en/hooks). The hook lints the file with ruff under
the root ruff.toml and keeps the findings on the touched lines. An Edit or MultiEdit touches the
lines spanned by each occurrence of its new_string in the file after the call. A Write touches
the whole file, and at most MAX_WRITE findings are reported for it. A syntax error is reported
wherever it sits, because the file no longer imports.

A NotebookEdit of a code cell is checked in two steps. The cell's source is parsed with
ast.parse, with IPython magics, shell escapes and help queries replaced by `pass`, so a syntax
error is caught even when ruff is absent. When the cell parses and ruff is found, ruff checks
the notebook with the F and E9 rules only (NOTEBOOK_SELECT) and the findings in the edited cell
are kept. ruff.toml excludes notebooks from the full lint, and ruff lints a file named on its
command line even when the configuration excludes it, so the notebook check needs no change to
ruff.toml. A markdown cell and a deleted cell are not checked. .claude/hooks/notebook_check.py
applies the same checks to a notebook a shell command changed through nbformat.

When ruff.toml has a [format] section, the hook also runs `ruff format --check
--force-exclude` on the file and reports a module the formatter covers that is not
formatted. The [format] exclude list holds the modules not yet formatted, so the
check reaches new modules and modules that have left the list. lib/ is a copy of the shared
9-bus library, and lib/MANIFEST.md records each module's SHA-256 and the date of the copy, so a
listed lib/ module leaves the list when a formatted copy arrives.

ruff runs with the repository root as its working directory, because ruff resolves the
relative paths of a --config file (the per-file ignores and the format exclude list)
against the working directory.

The findings reach the agent through hookSpecificOutput.additionalContext. A clean or ignored
call prints nothing. The hook ignores a file outside the repository, a file that is not .py or
.ipynb, and a file under sources/, outputs/, sandbox_exhibits/ or .ipynb_checkpoints/.

ruff is the CODE_CHECK_RUFF environment variable where it is set. Otherwise it is the first ruff
found beside the interpreter that runs the hook, in the active conda environment
(CONDA_PREFIX), on PATH, and in the named environments of a conda installation under the home
folder or /opt (CONDA_ROOTS), in that order. When no ruff is found, the hook says
"code_check: ruff not found; lint skipped" once per session, through a flag file in the
temporary directory, and then skips the lint in silence. CODE_CHECK_RUFF set to an empty string
turns the lint off without the message. The hook skips silently when ruff.toml is absent. It
always exits 0 and fails open on any error, as .claude/hooks/guard.py does, because a lint must
never block the edit it reports on. Standard library only. CODE_STYLE.md section 14 states the
lint and formatter policy.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "ruff.toml"
#: The conda installations whose named environments are searched for ruff, after PATH.
CONDA_ROOTS = (
    Path.home() / "miniconda3",
    Path.home() / "anaconda3",
    Path.home() / "miniforge3",
    Path("/opt/anaconda3"),
    Path("/opt/miniconda3"),
)
SKIP_PARTS = {"sources", "outputs", "sandbox_exhibits", ".ipynb_checkpoints"}
MAX_WRITE = 15
MAX_EDIT = 25
TIMEOUT = 8
#: The ruff rules run on notebook code: the pyflakes family and the syntax errors.
NOTEBOOK_SELECT = "F,E9"
RUFF_MISSING = "code_check: ruff not found; lint skipped"
#: The folder of the once-per-session flag files; CODE_CHECK_STATE_DIR overrides it for the tests.
STATE_DIR_NAME = "claude-code-check"
#: The cell magics whose body is Python, which the syntax check parses without the magic line.
PY_CELL_MAGICS = {"time", "timeit", "capture", "prun"}
_MAGIC_LINE = re.compile(r"^(\s*)(%|!)")
_MAGIC_ASSIGN = re.compile(r"^(\s*)[\w.]+(\s*,\s*[\w.]+)*\s*=\s*[%!]")
_HELP_LINE = re.compile(r"^(\s*)[\w.]+\?{1,2}\s*$")


def env_ruffs(prefix: Path) -> list[Path]:
    """The places ruff sits in a conda environment folder: Scripts/ on Windows, bin/ elsewhere."""
    return [prefix / "Scripts" / "ruff.exe", prefix / "bin" / "ruff"]


def near_candidates() -> list[Path]:
    """The ruff beside the hook's interpreter, then the ruff of the active conda environment.

    A virtual environment keeps ruff in the interpreter's own folder (bin/, or Scripts/ on
    Windows), and a conda environment on Windows keeps it in Scripts/ under the interpreter's
    folder.
    """
    here = Path(sys.executable).parent
    out = [here / "ruff", here / "ruff.exe", here / "Scripts" / "ruff.exe"]
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        out += env_ruffs(Path(conda))
    return out


def conda_candidates(roots: tuple[Path, ...] | None = None) -> list[Path]:
    """The ruff of each named environment of each conda installation, environments in name order."""
    out: list[Path] = []
    for base in CONDA_ROOTS if roots is None else roots:
        try:
            envs = sorted(p for p in (base / "envs").iterdir() if p.is_dir())
        except OSError:
            continue
        for env in envs:
            out += env_ruffs(env)
    return out


def find_ruff() -> str | None:
    """The ruff executable, or None when it is absent.

    The order is CODE_CHECK_RUFF when set, then the interpreter's own folder and CONDA_PREFIX,
    then PATH, then the named conda environments. CODE_CHECK_RUFF set to an empty string
    disables the hook.
    """
    env = os.environ.get("CODE_CHECK_RUFF")
    if env is not None:
        return env if env and Path(env).is_file() else None
    for cand in near_candidates():
        if cand.is_file():
            return str(cand)
    on_path = shutil.which("ruff")
    if on_path:
        return on_path
    for cand in conda_candidates():
        if cand.is_file():
            return str(cand)
    return None


def ruff_disabled() -> bool:
    """True when CODE_CHECK_RUFF is set to an empty string, which turns the lint off in silence."""
    return os.environ.get("CODE_CHECK_RUFF") == ""


def state_dir() -> Path:
    """The folder of the once-per-session flag files, under the temporary directory."""
    env = os.environ.get("CODE_CHECK_STATE_DIR")
    return Path(env) if env else Path(tempfile.gettempdir()) / STATE_DIR_NAME


def ruff_missing_note(data: dict) -> str:
    """The ruff-not-found line on the first call of a session, '' after it or when the lint is off.

    The session is the payload's session_id. A flag file marks the line as said. When the flag
    cannot be written the line is said again on the next call, which is the fail-open side.
    """
    if ruff_disabled():
        return ""
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(data.get("session_id") or "nosession"))
    flag = state_dir() / f"ruff-missing-{sid[:80]}"
    if flag.exists():
        return ""
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_bytes(b"1\n")
    except OSError:
        pass
    return RUFF_MISSING


def strip_magics(src: str) -> str | None:
    """The cell source with IPython syntax replaced by `pass`, or None for a non-Python cell magic.

    A line magic (%), a shell escape (!), an assignment from either, and a help query (name?)
    become `pass` at the same indentation, so the line numbers are kept. A cell magic whose body
    is Python (PY_CELL_MAGICS) loses its first line; any other cell magic is not Python.
    """
    lines = src.split("\n")
    first = next((k for k, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].lstrip().startswith("%%"):
        name = lines[first].lstrip()[2:].split(maxsplit=1)
        if not name or name[0] not in PY_CELL_MAGICS:
            return None
        lines[first] = ""
    out = []
    for line in lines:
        m = (
            _MAGIC_LINE.match(line)
            or _MAGIC_ASSIGN.match(line)
            or _HELP_LINE.match(line)
        )
        out.append(m.group(1) + "pass" if m else line)
    return "\n".join(out)


def parse_quietly(src: str) -> None:
    """Parse the source with ast.parse and drop its SyntaxWarnings, which would reach stderr."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ast.parse(src)


def cell_syntax_error(src: str) -> tuple[int, str] | None:
    """(line, message) of a syntax error in a code cell's source, or None when it parses.

    The source is parsed as written first, so a line that only looks like a magic (an operator
    continuing an expression, a line inside a string) is never rewritten when the cell parses.
    """
    try:
        parse_quietly(src)
    except SyntaxError:
        pass
    except ValueError as e:  # a null byte in the source
        return 1, str(e)
    else:
        return None
    stripped = strip_magics(src)
    if stripped is None:
        return None
    try:
        parse_quietly(stripped)
    except SyntaxError as e:
        return e.lineno or 1, e.msg or "invalid syntax"
    except ValueError as e:
        return 1, str(e)
    return None


def cell_source(cell: dict) -> str:
    """A notebook cell's source as one string, from either the list or the string form."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def edited_cells(nb: dict, ti: dict) -> set[int]:
    """The 0-based indices of the cells a NotebookEdit wrote, found as prose_lint.py finds them.

    The cell whose source equals new_source wins, then a match without surrounding whitespace,
    then the cell_id (the cell after it for an insert), then a "cell-N" index.
    """
    cells = nb.get("cells") or []
    new_src = ti.get("new_source")
    if isinstance(new_src, str):
        for key in (lambda s: s, lambda s: s.strip()):
            hits = {
                i for i, c in enumerate(cells) if key(cell_source(c)) == key(new_src)
            }
            if hits:
                return hits
    cid = ti.get("cell_id")
    if cid is not None:
        insert = (ti.get("edit_mode") or "replace") == "insert"
        for i, c in enumerate(cells):
            if c.get("id") == cid:
                return {i + 1} if insert else {i}
        m = re.match(r"(?:cell-)?(\d+)$", str(cid))
        if m:
            return {int(m.group(1))}
    return set()


def ruff_notebook(ruff: str, config: Path, path: Path) -> list[dict]:
    """Ruff's F and E9 findings for a notebook; each carries "cell", 1-based over all cells.

    The notebook is named on the command line, so ruff lints it although ruff.toml excludes
    *.ipynb (the exclusion binds only under --force-exclude). `--extension ipynb:python` is not
    passed, because it makes ruff read the notebook's JSON as one Python expression.
    """
    cmd = [
        ruff,
        "check",
        "--config",
        str(config),
        "--select",
        NOTEBOOK_SELECT,
        "--no-cache",
        "--exit-zero",
        "--quiet",
        "--output-format",
        "json",
        str(path),
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(config.parent),
    )
    out = r.stdout.strip()
    if not out:
        return []
    found = json.loads(out)
    return found if isinstance(found, list) else []


def notebook_rows(
    nb: dict, cells: set[int], ruff: str | None, config: Path, path: Path
) -> list[str]:
    """The report rows for the given code cells of a notebook: syntax first, then ruff's F and E9.

    Cells are named by their 0-based index, as prose_lint.py names them. Ruff runs only when
    every checked cell parses, because a syntax error makes its pyflakes findings unreliable.
    """
    nbcells = nb.get("cells") or []
    rows = []
    for i in sorted(cells):
        if i >= len(nbcells) or nbcells[i].get("cell_type") != "code":
            continue
        err = cell_syntax_error(cell_source(nbcells[i]))
        if err:
            rows.append(f"- syntax cell {i} line {err[0]}: {err[1]}")
    if rows or not ruff:
        return rows
    found = [
        f
        for f in ruff_notebook(ruff, config, path)
        if (f.get("cell") or 0) - 1 in cells
    ]
    found.sort(key=lambda f: (f["cell"], (f.get("location") or {}).get("row") or 0))
    for f in found:
        code = "syntax" if is_syntax(f) else f.get("code")
        row = (f.get("location") or {}).get("row", "?")
        rows.append(f"- {code} cell {f['cell'] - 1} line {row}: {f.get('message', '')}")
    return rows


def rel_in_root(path: Path, root: Path) -> Path | None:
    """The path relative to root, or None outside root or under a skipped folder."""
    try:
        rel = Path(os.path.normcase(str(path.resolve()))).relative_to(
            os.path.normcase(str(root.resolve()))
        )
    except (ValueError, OSError):
        return None
    if any(part in SKIP_PARTS for part in rel.parts):
        return None
    return rel


def line_spans(text: str, needle: str) -> set[int]:
    """The 1-based line numbers spanned by every occurrence of needle in text."""
    lines: set[int] = set()
    if not needle:
        return lines
    start = 0
    while True:
        j = text.find(needle, start)
        if j < 0:
            break
        first = text.count("\n", 0, j) + 1
        last = first + needle.count("\n")
        if needle.endswith("\n"):
            last -= 1
        lines.update(range(first, max(first, last) + 1))
        start = j + max(1, len(needle))
    return lines


def touched_lines(data: dict, text: str) -> set[int] | None:
    """The set of touched line numbers, or None for the whole file (Write)."""
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    if tool == "Write":
        return None
    if tool == "Edit":
        news = [ti.get("new_string")]
    elif tool == "MultiEdit":
        news = [
            e.get("new_string") for e in (ti.get("edits") or []) if isinstance(e, dict)
        ]
    else:
        return set()
    region: set[int] = set()
    for new in news:
        if isinstance(new, str):
            region |= line_spans(text, new)
    return region


def run_ruff(ruff: str, config: Path, path: Path) -> list[dict]:
    """Ruff's JSON findings for one file, or [] when ruff reports nothing."""
    cmd = [
        ruff,
        "check",
        "--config",
        str(config),
        "--no-cache",
        "--exit-zero",
        "--quiet",
        "--output-format",
        "json",
        str(path),
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(config.parent),
    )
    out = r.stdout.strip()
    if not out:
        return []
    found = json.loads(out)
    return found if isinstance(found, list) else []


def has_format_section(config: Path) -> bool:
    """True when the ruff config has a [format] section, which turns the format check on."""
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(line.strip() == "[format]" for line in text.splitlines())


def format_pending(ruff: str, config: Path, path: Path) -> bool:
    """True when the formatter covers the file and would reformat it.

    `ruff format --check` exits 1 for a file it would reformat, 0 for a formatted or excluded
    file, and 2 for an error such as a syntax error, which the lint already reports.
    """
    cmd = [
        ruff,
        "format",
        "--check",
        "--force-exclude",
        "--config",
        str(config),
        "--no-cache",
        "--quiet",
        str(path),
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(config.parent),
    )
    return r.returncode == 1


def is_syntax(finding: dict) -> bool:
    """True for a parse error, which ruff 0.15 codes "invalid-syntax" and older releases null or E999."""
    code = finding.get("code")
    return not code or code in ("invalid-syntax", "E999")


def format_findings(
    rel: Path, items: list[dict], whole: bool, total: int, unformatted: bool = False
) -> str:
    """The report: one row per kept finding, then the format row, then the policy line."""
    count = total + (1 if unformatted else 0)
    scope = "whole file" if whole else "touched lines"
    plural = "" if count == 1 else "s"
    rows = [f"code_check: {count} finding{plural} in {rel.as_posix()} ({scope})."]
    for f in items:
        loc = f.get("location") or {}
        code = f.get("code") or "syntax"
        rows.append(f"- {code} line {loc.get('row', '?')}: {f.get('message', '')}")
    if total > len(items):
        rows.append(
            f"... and {total - len(items)} more; run ruff check --config ruff.toml "
            "on the file for the full list."
        )
    if unformatted:
        rows.append(
            "- format: the formatter covers this module (it is not on the [format] exclude "
            "list of ruff.toml, which holds the modules not yet formatted) and "
            "would reformat it; run "
            f"`ruff format --config ruff.toml {rel.as_posix()}` from the repository root "
            "(CODE_STYLE.md 3.1)."
        )
    rows.append(
        "Fix the code, or keep a deliberate case with `# noqa: <code> - <reason>`. "
        "The rules are ruff.toml; the policy is CODE_STYLE.md section 14."
    )
    return "\n".join(rows)


def true_path(path: Path, root: Path, rel: Path) -> tuple[Path, Path]:
    """The file's resolved path and its path relative to root, in the case the disk holds.

    A tool call can carry a lower-case drive letter (c: for C:). ruff matches the per-file
    ignores and the [format] exclude list of ruff.toml case-sensitively, so the file is linted
    by its resolved path. rel_in_root compares case-folded paths, and its result is the
    fallback for the relative path.
    """
    real = path.resolve()
    try:
        return real, real.relative_to(root.resolve())
    except ValueError:
        return real, rel


def target_of(data: dict, root: Path, suffix: str) -> tuple[Path, Path] | None:
    """(resolved path, relative path) of the call's file when it has the suffix and is checked.

    The result is None outside the repository, under a skipped folder, when the file is absent,
    or when ruff.toml is absent.
    """
    ti = data.get("tool_input") or {}
    fp = ti.get("notebook_path") if suffix == ".ipynb" else ti.get("file_path")
    if not fp or not str(fp).lower().endswith(suffix):
        return None
    path = Path(fp)
    if not path.is_absolute():
        path = Path(data.get("cwd") or root) / path
    rel = rel_in_root(path, root)
    if rel is None or not path.is_file() or not (root / CONFIG_NAME).is_file():
        return None
    return true_path(path, root, rel)


def python_report(data: dict, root: Path, ruff: str | None) -> str:
    """The lint report for a .py file a Write, Edit or MultiEdit touched, or ''."""
    target = target_of(data, root, ".py")
    if target is None:
        return ""
    path, rel = target
    config = root / CONFIG_NAME
    ruff = ruff or find_ruff()
    if not ruff:
        return ruff_missing_note(data)
    text = path.read_text(encoding="utf-8", errors="replace")
    region = touched_lines(data, text)
    if region is not None and not region:
        return ""
    found = run_ruff(ruff, config, path)
    keep = []
    for f in found:
        row = (f.get("location") or {}).get("row")
        if is_syntax(f):
            f["code"] = None
        if f.get("code") is None or region is None or row in region:
            keep.append(f)
    syntax = any(f.get("code") is None for f in keep)
    unformatted = (
        not syntax and has_format_section(config) and format_pending(ruff, config, path)
    )
    if not keep and not unformatted:
        return ""
    keep.sort(
        key=lambda f: ((f.get("location") or {}).get("row") or 0, f.get("code") or "")
    )
    limit = MAX_WRITE if region is None else MAX_EDIT
    return format_findings(rel, keep[:limit], region is None, len(keep), unformatted)


def notebook_report(data: dict, root: Path, ruff: str | None) -> str:
    """The code check of the cell a NotebookEdit wrote, or '' for a clean, markdown or deleted cell."""
    ti = data.get("tool_input") or {}
    mode = ti.get("edit_mode") or "replace"
    if mode == "delete" or ti.get("cell_type") == "markdown":
        return ""
    target = target_of(data, root, ".ipynb")
    if target is None:
        return ""
    path, rel = target
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return ""  # a notebook mid-write is not checked
    if not isinstance(nb, dict):
        return ""
    nbcells = nb.get("cells") or []
    cells = {
        i
        for i in edited_cells(nb, ti)
        if i < len(nbcells) and nbcells[i].get("cell_type") == "code"
    }
    if not cells:
        return ""
    ruff = ruff or find_ruff()
    rows = notebook_rows(nb, cells, ruff, root / CONFIG_NAME, path)
    note = "" if ruff else ruff_missing_note(data)
    if not rows:
        return note
    which = ", ".join(str(i) for i in sorted(cells))
    plural = "" if len(rows) == 1 else "s"
    head = f"code_check: {len(rows)} finding{plural} in {rel.as_posix()} (edited cell {which})."
    tail = (
        "Notebook code is checked for syntax and with ruff's F and E9 rules only; cells are "
        "numbered from 0, as prose_lint.py numbers them. Fix the cell, or keep a deliberate "
        "case with `# noqa: <code> - <reason>`."
    )
    return "\n".join(x for x in (head, *rows[:MAX_EDIT], tail, note) if x)


def run_hook(
    stdin_text: str, root: Path | str | None = None, ruff: str | None = None
) -> str:
    """The hook's stdout text, which is '' on a clean or ignored call."""
    root = Path(root) if root is not None else ROOT
    data = json.loads(stdin_text)
    if not isinstance(data, dict):
        return ""
    if data.get("tool_name") == "NotebookEdit":
        ctx = notebook_report(data, root, ruff)
    else:
        ctx = python_report(data, root, ruff)
    if not ctx:
        return ""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": ctx,
            }
        }
    )


def main() -> int:
    """The entry point, which reads the call from stdin, writes the report to stdout and returns 0.

    It returns 0 in every case, because the hook fails open.
    """
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        out = run_hook(raw)
        if out:
            sys.stdout.write(out + "\n")
    except Exception as e:
        try:
            sys.stderr.write(f"code_check.py: {type(e).__name__}; nothing checked\n")
        except Exception:  # noqa: S110 - a hook fails open, even when stderr is closed
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
