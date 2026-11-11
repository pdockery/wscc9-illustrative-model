#!/usr/bin/env python
"""PostToolUse hook: checks the notebooks a shell command changed through nbformat.

Claude Code runs this after every Bash or PowerShell call, with the call's JSON on stdin
(https://code.claude.com/docs/en/hooks). A notebook edited by a patch script that uses nbformat
(the working agreements of CLAUDE.md allow the route) never passes through the NotebookEdit
hooks, so neither prose_lint.py nor code_check.py sees it. This hook stands in for them on that
route.

The hook acts only when the command used nbformat: the command text names nbformat (an inline
`python -c` or a heredoc), or a .py script the command names mentions nbformat. It then takes
the candidate notebooks, which are the .ipynb files git reports as modified or untracked and the
notebooks the command or the script names, and keeps those written within WINDOW seconds. A
named notebook is found as given, against the command's working directory, and then against the
repository root, where the deck notebooks, 101_fundamentals.ipynb and sandbox.ipynb sit. The window covers the Bash
tool's longest foreground run. For each notebook (at most MAX_NOTEBOOKS, newest first), the
changed cells are the cells whose source differs from HEAD, or every cell of a notebook HEAD
lacks. Three checks run on the changed cells:

- each changed code cell is parsed with ast.parse, IPython magics skipped (code_check.py);
- when every changed cell parses and ruff is found, ruff's F and E9 rules (code_check.py);
- prose_lint.py --json on the notebook, keeping the findings in the changed cells.

A notebook HEAD lacks reports every cell as changed, so an untracked notebook's report is
capped at MAX_ROWS rows like any other.

The report reaches the agent through hookSpecificOutput.additionalContext, and a clean or
ignored call prints nothing. The hook is advisory. It always exits 0 and fails open on any
error, as code_check.py does, and it stops starting new checks after DEADLINE seconds. Standard
library only.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
sys.path.insert(0, str(HOOKS))
import code_check as cc  # noqa: E402 - the sibling hook module, found through its folder

ROOT = HOOKS.parents[1]
PROSE_LINT = Path(".claude") / "skills" / "writing-voice" / "scripts" / "prose_lint.py"
#: The Bash tool's longest foreground run is 600 s; the window adds a minute.
WINDOW = 660
MAX_NOTEBOOKS = 5
MAX_ROWS = 15
MAX_SCRIPT_BYTES = 2_000_000
DEADLINE = 20.0
GIT_TIMEOUT = 10
LINT_TIMEOUT = 10
_PY_TOKEN = re.compile(
    r"\"([^\"]+?\.py)\"|'([^']+?\.py)'|([^\s\"'<>|;&()]+?\.py)(?=$|[\s\"'<>|;&()])"
)
_NB_TOKEN = re.compile(
    r"\"([^\"]+?\.ipynb)\"|'([^']+?\.ipynb)'|([^\s\"'<>|;&()]+?\.ipynb)(?=$|[\s\"'<>|;&()])"
)
_GIT_BASH_DRIVE = re.compile(r"^/([A-Za-z])/")


def tokens(pattern: re.Pattern[str], text: str) -> list[str]:
    """Every path the pattern finds in text, quoted or bare, in order of appearance."""
    return [next(g for g in m.groups() if g) for m in pattern.finditer(text)]


def as_path(token: str, base: Path) -> Path:
    """A path token as a Path, a Git Bash drive (/c/...) converted on Windows, relative to base."""
    if os.name == "nt":
        token = _GIT_BASH_DRIVE.sub(lambda m: m.group(1).upper() + ":/", token)
    p = Path(token)
    return p if p.is_absolute() else base / p


def read_script(path: Path) -> str:
    """A script's text, or '' when it is absent, too large or unreadable."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_SCRIPT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def nbformat_texts(command: str, base: Path) -> list[str] | None:
    """The texts that used nbformat (the command, the scripts it names), or None when none did."""
    texts = [command] + [
        read_script(as_path(t, base)) for t in tokens(_PY_TOKEN, command)
    ]
    used = [t for t in texts if re.search(r"\bnbformat\b", t)]
    return used or None


def named_notebooks(texts: list[str], base: Path, root: Path) -> list[Path]:
    """The notebooks the texts name, found as given against base, then against root."""
    out = []
    for text in texts:
        for tok in tokens(_NB_TOKEN, text):
            p = as_path(tok, base)
            if not p.is_file():
                p = as_path(tok, root)
            if p.is_file():
                out.append(p)
    return out


def git(root: Path, *args: str) -> bytes | None:
    """The stdout of a git command, or None when git fails."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            timeout=GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def git_notebooks(root: Path) -> list[Path]:
    """The modified, added or untracked .ipynb files git reports (ignored files are not listed)."""
    raw = git(root, "status", "--porcelain", "-z", "--untracked-files=all")
    if raw is None:
        return []
    parts = raw.decode("utf-8", "replace").split("\0")
    out, i = [], 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        code, rel = entry[:2], entry[3:]
        if code[0] in "RC":
            i += 1  # the source path of a rename follows
        if "D" not in code and rel.lower().endswith(".ipynb"):
            out.append(root / rel)
    return out


def candidates(paths: list[Path], root: Path, now: float) -> list[tuple[Path, Path]]:
    """(resolved path, relative path) of the notebooks in root written within WINDOW, newest first."""
    seen: dict[str, tuple[float, Path, Path]] = {}
    for p in paths:
        rel = cc.rel_in_root(p, root)
        if rel is None:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if now - mtime > WINDOW:
            continue
        real, rel = cc.true_path(p, root, rel)
        seen.setdefault(os.path.normcase(str(real)), (mtime, real, rel))
    ranked = sorted(seen.values(), key=lambda x: -x[0])
    return [(real, rel) for _, real, rel in ranked[:MAX_NOTEBOOKS]]


def changed_cells(root: Path, rel: Path, nb: dict) -> set[int]:
    """The 0-based indices of the cells whose source differs from HEAD; every cell when HEAD lacks it."""
    now = [cc.cell_source(c) for c in nb.get("cells") or []]
    raw = git(root, "show", f"HEAD:{rel.as_posix()}")
    try:
        head = json.loads(raw.decode("utf-8", "replace")) if raw is not None else None
    except ValueError:
        head = None
    if not isinstance(head, dict):
        return set(range(len(now)))
    before = [cc.cell_source(c) for c in head.get("cells") or []]
    out: set[int] = set()
    matcher = difflib.SequenceMatcher(None, before, now, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            out.update(range(j1, j2))
    return out


def prose_rows(root: Path, path: Path, cells: set[int]) -> list[str]:
    """prose_lint.py's findings in the given cells, one row each, or [] when it cannot run."""
    script = root / PROSE_LINT
    if not script.is_file():
        return []
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--json", str(path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=LINT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        found = (
            json.loads(r.stdout).get("findings", []) if r.returncode in (0, 1) else []
        )
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return []
    rows = []
    for f in found:
        if f.get("cell") not in cells:
            continue
        sev = str(f.get("severity", "")).upper()
        rows.append(
            f"- {sev} {f.get('rule')} {f.get('pin')}: {f.get('message')} | {f.get('excerpt')}"
        )
    return rows


def check_notebook(
    root: Path, path: Path, rel: Path, ruff: str | None
) -> tuple[list[str], int]:
    """(report rows, number of changed cells) for one notebook."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return [
            f"- the notebook does not parse as JSON; {rel.as_posix()} is damaged"
        ], 0
    if not isinstance(nb, dict):
        return [], 0
    cells = changed_cells(root, rel, nb)
    if not cells:
        return [], 0
    rows = cc.notebook_rows(nb, cells, ruff, root / cc.CONFIG_NAME, path)
    rows += prose_rows(root, path, cells)
    return rows, len(cells)


def run_hook(
    stdin_text: str,
    root: Path | str | None = None,
    ruff: str | None = None,
    now: float | None = None,
) -> str:
    """The hook's stdout text, which is '' on a clean or ignored call."""
    t0 = time.monotonic()
    root = Path(root) if root is not None else ROOT
    data = json.loads(stdin_text)
    if not isinstance(data, dict):
        return ""
    command = (data.get("tool_input") or {}).get("command")
    if not isinstance(command, str) or (
        "nbformat" not in command and ".py" not in command
    ):
        return ""
    base = Path(data.get("cwd") or root)
    texts = nbformat_texts(command, base)
    if texts is None:
        return ""
    now = time.time() if now is None else now
    found = candidates(
        git_notebooks(root) + named_notebooks(texts, base, root), root, now
    )
    if not found:
        return ""
    ruff = ruff or cc.find_ruff()
    blocks = []
    for path, rel in found:
        if time.monotonic() - t0 > DEADLINE:
            blocks.append(
                f"notebook_check: stopped before {rel.as_posix()} (time budget)."
            )
            break
        rows, n = check_notebook(root, path, rel, ruff)
        if not rows:
            continue
        head = (
            f"notebook_check: {len(rows)} finding{'' if len(rows) == 1 else 's'} in "
            f"{rel.as_posix()} ({n} {'cell differs' if n == 1 else 'cells differ'} from HEAD; "
            "cells numbered from 0)."
        )
        more = len(rows) - MAX_ROWS
        tail = [f"... and {more} more."] if more > 0 else []
        blocks.append("\n".join([head, *rows[:MAX_ROWS], *tail]))
    if not blocks:
        return ""
    blocks.append(
        "A shell command changed these notebooks through nbformat, which the NotebookEdit "
        "hooks do not see, so this report stands in for prose_lint.py and code_check.py. "
        "Run prose_lint.py on a notebook for its full list."
    )
    note = "" if ruff else cc.ruff_missing_note(data)
    ctx = "\n\n".join(x for x in (*blocks, note) if x)
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
            sys.stderr.write(
                f"notebook_check.py: {type(e).__name__}; nothing checked\n"
            )
        except Exception:  # noqa: S110 - a hook fails open, even when stderr is closed
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
