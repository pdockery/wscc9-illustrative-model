#!/usr/bin/env python
"""Stop and PreCompact hook: the clock-out check, advisory and once per session.

Contract (https://code.claude.com/docs/en/hooks). The input on stdin is the event's JSON
(``session_id``, ``hook_event_name``, ``stop_hook_active``, ``cwd`` and others). The hook
never blocks and always exits 0.

The SessionStart brief records a signature of the working tree in the gitignored
``.claude/.state/<session_id>.json`` (``record_baseline``). The signature hashes the
``git status --porcelain`` entries with each listed file's size and modification time, so
that an edit to a file already modified also changes it, and the state keeps each entry's
own signature, so that the check can tell which files changed in this session. This hook
stays silent when the state file is absent, when the signature is unchanged, when
``stop_hook_active`` is set, when it has already fired for the event in this session, or
when it has nothing to report.

The checks that show to everyone:

1. each ``lib/`` module whose SHA-256 differs from ``lib/MANIFEST.md``, hashed as the brief
   hashes it (CRLF read as LF);
2. each deck notebook changed in this session without a passing slide-check run after the
   change. A deck notebook is one a ``tests/test_slides_*.py`` module names as
   ``NOTEBOOK = "<name>.ipynb"``, or every root notebook when no such module exists. The run
   is read from ``tools/.cache/check_runs.jsonl`` when that ledger exists (a line whose
   ``ok`` is true, whose ``tier`` or ``tiers`` names ``slides``, and whose ``ts`` is later
   than the notebook's modification time); without the ledger the check is a reminder;
3. each untracked ``.py`` file that a modified ``.py`` file imports;
4. each path that this session's record text cites in backticks, that exists and that git
   ignores, so that it is on this disk only.

The dev_notes check, whether ``dev_notes.md`` has an entry dated today, shows only when the
environment variable ``LAB_MAINTAINER`` is set and not empty, and so does the closing text,
which names the five parts of an entry and the ``D-YYYYMMDD-n`` ids of held decisions.

Output contract, from the hooks reference (code.claude.com/docs/en/hooks). On Stop the hook
prints one JSON object, ``{"hookSpecificOutput": {"hookEventName": "Stop",
"additionalContext": note}}``, which the reference lists as non-error feedback that continues
the conversation. The hook never emits ``decision``, and the ``fired`` list makes it continue
the turn at most once per session. On PreCompact the stdout of a hook that exits 0 becomes the
instructions for the compaction summary, cut at 10,000 characters, so the note goes to stdout
as plain text held under ``NOTE_CAP``.

The hook runs no suite and no staleness report. Standard library only. Any error writes one
line to stderr and exits 0.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HOOKS = Path(__file__).resolve().parent
STATE_DIR = Path(".claude") / ".state"
GIT_TIMEOUT = 5
LEDGER = Path("tools") / ".cache" / "check_runs.jsonl"
#: The records whose text this session adds and whose citations are read.
RECORDS = ("dev_notes.md", "NOTEBOOK_RECORD.md")
#: The variable that turns on the dev_notes check and the closing text.
MAINTAINER_VAR = "LAB_MAINTAINER"
#: The command that runs the slide checks when no ledger records one.
SLIDES_COMMAND = "python -m pytest tests -q -m slides"
HEADING_DATE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})")
BACKTICK = re.compile(r"`([^`\s]+/[^`\s]*)`")
SLIDE_NOTEBOOK = re.compile(r"""^NOTEBOOK\s*=\s*["']([^"']+\.ipynb)["']""", re.M)
#: Names printed per finding before the rest is elided.
SHOWN = 4
#: Characters of note kept, below the 10,000 at which Claude Code cuts hook output.
NOTE_CAP = 9000


def git(root: Path, *args: str, binary: bool = False) -> Any:
    """The stdout of a git command in ``root``, or an empty value on any failure."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            timeout=GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return b"" if binary else ""
    if r.returncode != 0 and not (args and args[0] == "check-ignore"):
        return b"" if binary else ""
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def porcelain(root: Path, *extra: str) -> list[tuple[str, str]]:
    """(status code, path) of each ``git status --porcelain -z`` entry."""
    raw = git(root, "status", "--porcelain", "-z", *extra)
    parts = raw.split("\0")
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if code[0] in "RC":
            i += 1  # the original path of a rename or copy follows
        out.append((code, path))
    return out


def entry_signatures(root: Path) -> dict[str, str]:
    """Each dirty entry's path and its status code, size and modification time."""
    out: dict[str, str] = {}
    for code, path in porcelain(root):
        try:
            st = (root / path).stat()
            out[path] = f"{code} {st.st_size} {st.st_mtime_ns}"
        except OSError:  # noqa: PERF203 - a deleted entry has no size
            out[path] = f"{code} -"
    return out


def tree_signature(root: Path, entries: dict[str, str] | None = None) -> str:
    """A hash of the dirty entries with each one's size and modification time."""
    entries = entry_signatures(root) if entries is None else entries
    h = hashlib.sha256()
    for path in sorted(entries):
        h.update(f"{path} {entries[path]}\n".encode())
    return h.hexdigest()


def state_file(root: Path, session_id: str) -> Path:
    """The state file of one session, its id reduced to word characters and hyphens."""
    safe = re.sub(r"[^\w-]", "_", session_id)[:80] or "unknown"
    return root / STATE_DIR / f"{safe}.json"


def read_state(path: Path) -> dict[str, Any] | None:
    """The state a session recorded, or None when absent or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_state(path: Path, data: dict[str, Any]) -> None:
    """Write a session's state with LF endings, creating the folder with a ``*`` .gitignore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ignore = path.parent / ".gitignore"
    if not ignore.exists():
        ignore.write_bytes(b"*\n")
    path.write_bytes((json.dumps(data, indent=1) + "\n").encode("utf-8"))


def record_baseline(root: Path, session_id: str, source: str = "") -> bool:
    """Record the tree signature at a session's start; False when one is already recorded.

    The SessionStart brief calls this on every start, resume, clear and compaction, and the
    first signature of a session id is kept, so that a compaction does not reset it.
    """
    if not session_id:
        return False
    path = state_file(root, session_id)
    if path.exists():
        return False
    entries = entry_signatures(root)
    write_state(
        path,
        {
            "session_id": session_id,
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "signature": tree_signature(root, entries),
            "entries": entries,
            "fired": [],
        },
    )
    return True


def changed_since(baseline: object, now: dict[str, str]) -> list[str]:
    """The dirty paths of ``now`` whose signature differs from the baseline's, or are new."""
    base = baseline if isinstance(baseline, dict) else {}
    return sorted(p for p, sig in now.items() if base.get(p) != sig)


def manifest_divergence(root: Path) -> list[str]:
    """The ``lib/`` modules that differ from ``lib/MANIFEST.md``, as the brief counts them."""
    name = "session_brief"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, HOOKS / "session_brief.py")
        if spec is None or spec.loader is None:
            return []
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001 - an unimportable brief drops this check
            sys.modules.pop(name, None)
            return []
    st = mod.manifest_status(str(root))
    return list(st["differ"]) if st else []


def deck_notebooks(root: Path) -> set[str] | None:
    """The notebooks the slide checks name, or None when no slide-check module exists."""
    names: set[str] = set()
    found = False
    for p in sorted((root / "tests").glob("test_slides_*.py")):
        found = True
        try:
            names.update(SLIDE_NOTEBOOK.findall(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return names if found else None


def _when(rec: dict[str, Any]) -> float | None:
    """The POSIX time of a ledger line's ``ts`` (ISO 8601), or None."""
    raw = rec.get("ts") or rec.get("time")
    if not isinstance(raw, str):
        return None
    try:
        t = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.astimezone()
    return t.timestamp()


def passing_slides_time(root: Path) -> float | None:
    """The time of the newest passing slide-check line in the ledger, or None.

    The ledger is ``tools/.cache/check_runs.jsonl``. A line counts when its ``ok`` is true and
    its ``tier`` (a string) or ``tiers`` (a list) names ``slides``.
    """
    try:
        lines = (root / LEDGER).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    best = None
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:  # noqa: PERF203 - one malformed line is skipped
            continue
        if not isinstance(rec, dict) or rec.get("ok") is not True:
            continue
        tiers = rec.get("tiers")
        tiers = tiers if isinstance(tiers, list) else [rec.get("tier")]
        if "slides" not in tiers:
            continue
        t = _when(rec)
        if t is not None and (best is None or t > best):
            best = t
    return best


def unchecked_notebooks(root: Path, changed: list[str]) -> tuple[list[str], bool]:
    """(deck notebooks changed in this session with no later passing run, ledger present)."""
    decks = deck_notebooks(root)
    notebooks = [
        p
        for p in changed
        if "/" not in p
        and p.endswith(".ipynb")
        and (root / p).is_file()
        and (decks is None or p in decks)
    ]
    ledger = (root / LEDGER).is_file()
    if not notebooks:
        return [], ledger
    passed = passing_slides_time(root) if ledger else None
    if passed is None:
        return notebooks, ledger
    late = [p for p in notebooks if (root / p).stat().st_mtime > passed]
    return late, ledger


def dev_notes_dates(root: Path) -> list[str]:
    """The dates of the ``## YYYY-MM-DD`` entries of the root dev_notes.md."""
    try:
        text = (root / "dev_notes.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    return [m.group(1) for m in map(HEADING_DATE.match, text.splitlines()) if m]


def imported_names(path: Path) -> set[str]:
    """Every module and name a Python file imports, dotted parts included."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.update(a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            names.update(a.name for a in node.names)
    return names


def untracked_imports(root: Path) -> list[tuple[str, list[str]]]:
    """(untracked .py path, modified files that import it) for each such import."""
    untracked = [
        p
        for c, p in porcelain(root, "--untracked-files=all", "--", "*.py")
        if c == "??"
    ]
    if not untracked:
        return []
    modified = [
        p
        for c, p in porcelain(root, "--untracked-files=no", "--", "*.py")
        if c != "??" and "D" not in c
    ]
    imports = {p: imported_names(root / p) for p in modified}
    out = []
    for u in untracked:
        stem = Path(u).stem
        users = [Path(p).name for p, names in imports.items() if stem in names]
        if users:
            out.append((u, users))
    return out


def cited_paths(root: Path, since: str) -> list[tuple[str, str]]:
    """(path, record) for each backticked path in this session's record text.

    This session's text is the dev_notes.md entries dated on or after ``since`` and the
    lines added to NOTEBOOK_RECORD.md in the working tree.
    """
    found: list[tuple[str, str]] = []
    try:
        text = (root / RECORDS[0]).read_text(encoding="utf-8")
    except (OSError, ValueError):
        text = ""
    keep = False
    for line in text.splitlines():
        m = HEADING_DATE.match(line)
        if m:
            keep = m.group(1) >= since
        if keep:
            found.extend((p, RECORDS[0]) for p in BACKTICK.findall(line))
    diff = git(root, "diff", "-U0", "HEAD", "--", *RECORDS[1:])
    record = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            record = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("+") and record:
            found.extend((p, Path(record).name) for p in BACKTICK.findall(line))
    return found


def ignored_citations(root: Path, since: str) -> list[tuple[str, str]]:
    """(path, record) for each cited path that exists and that git ignores."""
    resolved: dict[str, str] = {}
    for raw, record in cited_paths(root, since):
        p = raw.strip().rstrip(".,;:").rstrip("/")
        if not p or any(ch in p for ch in "<>*?{}$"):
            continue
        if (root / p).exists():
            resolved.setdefault(p, record)
    if not resolved:
        return []
    out = git(root, "check-ignore", "--", *resolved)
    ignored = {line.strip().replace("\\", "/") for line in out.splitlines()}
    return [(p, r) for p, r in resolved.items() if p in ignored]


def _names(items: list[str], sep: str = ", ") -> str:
    """The first few names, with the count of the rest when more follow."""
    more = len(items) - SHOWN
    return sep.join(items[:SHOWN]) + (f"{sep}and {more} more" if more > 0 else "")


def maintainer() -> bool:
    """Whether LAB_MAINTAINER is set and not empty."""
    return bool(os.environ.get(MAINTAINER_VAR, "").strip())


def findings(root: Path, changed: list[str], started: str, today: str) -> list[str]:
    """The sentences of the checks that show to everyone, empty when none finds anything."""
    parts: list[str] = []
    differ = manifest_divergence(root)
    if differ:
        parts.append(
            f"{len(differ)} lib/ modules differ from lib/MANIFEST.md ({_names(differ)}); "
            "lib/ is a copy, so a change to it arrives as a fresh copy with a "
            "regenerated manifest."
        )
    late, ledger = unchecked_notebooks(root, changed)
    if late and ledger:
        parts.append(
            "No passing slides run in tools/.cache/check_runs.jsonl is later than the "
            f"change to {_names(late)}; a deck notebook change is done when the slide "
            "checks pass."
        )
    elif late:
        parts.append(
            f"Deck notebooks changed in this session: {_names(late)}. A deck notebook "
            f"change is done when `{SLIDES_COMMAND}` passes, and no ledger here records "
            "that run."
        )
    imports = untracked_imports(root)
    if imports:
        parts.append(
            "Untracked modules imported by modified files: "
            + _names([f"{u} (by {_names(users)})" for u, users in imports], "; ")
            + "."
        )
    cited = ignored_citations(root, started[:10] or today)
    if cited:
        parts.append(
            "Paths the records cite that git ignores, so they exist on this disk only: "
            + _names([f"{p} ({r})" for p, r in cited])
            + "."
        )
    return parts


def build_note(
    root: Path,
    started: str,
    today: str,
    changed: list[str] | None = None,
    is_maintainer: bool | None = None,
) -> str:
    """The clock-out note, or "" when it has nothing to report."""
    is_maintainer = maintainer() if is_maintainer is None else is_maintainer
    parts = findings(root, changed or [], started, today)
    if is_maintainer:
        dated = today in dev_notes_dates(root)
        parts.insert(
            0,
            f"dev_notes.md has {'an' if dated else 'no'} entry dated {today}.",
        )
    if not parts:
        return ""
    note = (
        "Clock-out check (session_close.py, advisory, once per session): the working tree "
        "changed in this session. " + " ".join(parts)
    )
    if is_maintainer:
        note += (
            " Before the session ends, write the dev_notes.md entry dated today in five "
            "parts: what was done; what verified it, by command and counts; what is "
            "unverified; what is held, by its D-YYYYMMDD-n id; the next step. A record "
            "entry names no other repository."
        )
    return note


def render(event: str, note: str) -> str:
    """The stdout for ``event``: PreCompact summary instructions or the Stop JSON object."""
    if len(note) > NOTE_CAP:
        note = note[: NOTE_CAP - 12].rstrip() + " [cut here]"
    if event == "PreCompact":
        return "Carry this into the summary. " + note + "\n"
    out = {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": note}}
    return json.dumps(out) + "\n"


def main() -> None:
    """Read the event, decide whether to speak, and print the note in the event's form."""
    raw = sys.stdin.read() if sys.stdin is not None and not sys.stdin.isatty() else ""
    hook = json.loads(raw) if raw.strip() else {}
    if not isinstance(hook, dict) or hook.get("stop_hook_active"):
        return
    event = str(hook.get("hook_event_name") or "Stop")
    if event not in ("Stop", "PreCompact"):
        return
    sid = str(hook.get("session_id") or "")
    if not sid:
        return
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or hook.get("cwd") or Path.cwd())
    path = state_file(root, sid)
    state = read_state(path)
    if state is None or event in state.get("fired", []):
        return
    entries = entry_signatures(root)
    if tree_signature(root, entries) == state.get("signature"):
        return
    changed = changed_since(state.get("entries"), entries)
    today = dt.date.today().isoformat()
    note = build_note(root, str(state.get("started", "")), today, changed)
    if not note:
        return
    state.setdefault("fired", []).append(event)
    write_state(path, state)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(render(event, note))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - the hook fails open (CODE_STYLE.md 13.4)
        sys.stderr.write(f"session_close: skipped ({type(e).__name__}).\n")
    sys.exit(0)
