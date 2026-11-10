#!/usr/bin/env python
"""SessionStart hook: prints the repository state brief as plain text on stdout.

Contract (https://code.claude.com/docs/en/hooks): on SessionStart, plain stdout on
exit 0 is added as context at the start of the conversation. The hook input on
stdin is JSON (session_id, cwd, hook_event_name, source, ...). It is read when
present and never required.

The brief is a list of blocks, each with its own word budget (``BUDGETS``), so that a long
block is cut on its own and never pushes a later block out. The blocks, in order:

- the opening line;
- the branch, the uncommitted entries as counts by kind, and the recent commits;
- the last commit's age, the oldest uncommitted change and the dev_notes entries since;
- the unmerged local branches older than 30 days;
- the worktrees;
- the Python interpreter count against the ceiling of ``.claude/hooks/_limits.py``;
- the library line: the module count, the date of the copy, and the modules whose SHA-256
  differs from ``lib/MANIFEST.md``;
- the notebook staleness from ``tools/.cache/staleness.json``, read only when it is newer
  than the newest change to ``lib/``;
- the hook log's counts from ``.claude/logs/hooks.jsonl``;
- the readiness report from ``tools/.cache/doctor.json``, when present;
- the newest dated entry of the root ``dev_notes.md``;
- the open decisions, the ``D-YYYYMMDD-n`` ids that dev_notes.md raises and does not close.

The manifest hashes each module's bytes with CRLF line endings read as LF, the bytes git
stores for a text file. A module that matches only in its CRLF working bytes is counted
separately, because its manifest row was written under the earlier convention.

A block whose source is absent prints nothing, except the blocks that report a missing
input by design. The hook also records the working tree's signature for the clock-out
check of ``.claude/hooks/session_close.py`` in the gitignored ``.claude/.state/``.

Any failure of a block drops that block and leaves the brief whole. A failure of the whole
prints one line, "Session state brief unavailable (<ErrorType>).", and the exit status is 0.
Standard library only. Wired by .claude/settings.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from types import ModuleType

HOOKS = Path(__file__).resolve().parent
#: The ceiling when .claude/hooks/_limits.py cannot be imported.
FALLBACK_CEILING = 7
GIT_TIMEOUT = 3
#: Days after which an unmerged local branch is reported.
BRANCH_AGE_DAYS = 30
#: The library folder and its manifest, repository-relative.
LIB_DIR = "lib"
MANIFEST = "lib/MANIFEST.md"
#: Module names a divergence line prints before it elides the rest.
SHOWN = 4
#: Words each block may use; a block over its budget is cut and marked " ...".
BUDGETS = {
    "head": 12,
    "status": 55,
    "age": 25,
    "branches": 22,
    "worktrees": 25,
    "python": 14,
    "library": 30,
    "staleness": 30,
    "hooklog": 20,
    "doctor": 20,
    "dev_notes": 25,
    "decisions": 14,
}
#: The cap on the whole brief, the sum of the budgets.
MAX_WORDS = sum(BUDGETS.values())
DAY = 86400
#: An open-decision id: ``D-`` and the date of the entry that raises it, then a number.
DECISION_ID = re.compile(r"\bD-(\d{8})-(\d+)\b")
#: A line that closes a decision: "Closed D-..." or "Decided D-...", or the id followed by
#: "closed" or "decided".
CLOSED_BEFORE = re.compile(r"(?i)\b(?:closed|decided)\b[\s:*_]*(D-\d{8}-\d+)\b")
CLOSED_AFTER = re.compile(
    r"(?i)\b(D-\d{8}-\d+)\b[\s:*_(),-]*(?:is\s+|was\s+)?(?:closed|decided)\b"
)


def project_root() -> str:
    """The project folder: CLAUDE_PROJECT_DIR, or the working directory."""
    return os.environ.get("CLAUDE_PROJECT_DIR") or str(Path.cwd())


def read_hook_input() -> dict[str, Any]:
    """Return the SessionStart JSON as a dict, or {} when absent or unreadable."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            if sys.stdin is None or sys.stdin.isatty():
                return
            raw = sys.stdin.read()
            if raw.strip():
                box["data"] = json.loads(raw)
        except Exception:  # noqa: S110 - an unreadable input leaves the brief whole
            pass

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(0.4)
    data = box.get("data")
    return data if isinstance(data, dict) else {}


def git(root: str, *args: str) -> str:
    """The stdout of a git command in ``root``, or "" on any failure."""
    try:
        r = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:  # noqa: BLE001 - a failed git call drops its block
        return ""
    return r.stdout if r.returncode == 0 else ""


def load_module(name: str, path: Path) -> ModuleType | None:
    """A standard-library module of the harness imported from its file, or None."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a module that fails to import drops its block
        sys.modules.pop(name, None)
        return None
    return mod


def clip_block(lines: list[str], budget: int) -> list[str]:
    """The lines of a block cut to ``budget`` words, the cut line ending in " ..."."""
    out: list[str] = []
    left = budget
    for line in lines:
        words = line.split()
        if len(words) <= left:
            out.append(line)
            left -= len(words)
            continue
        if left > 0:
            out.append(" ".join(words[:left]) + " ...")
        break
    return out


def kind(path: str) -> str:
    """The kind of an uncommitted path, by folder and extension."""
    p = path.replace("\\", "/").lower()
    if p.startswith((".claude/skills/", ".claude/hooks/")) or p.endswith("claude.md"):
        return "harness"
    if p.startswith("sources/"):
        return "sources"
    if p.endswith(".ipynb"):
        return "notebook"
    if p.endswith(".py"):
        return "py"
    if p.endswith((".tex", ".bib")):
        return "tex"
    if p.endswith((".pdf", ".docx", ".dotx")):
        return "pdf/docx"
    if p.endswith(".md"):
        return "md"
    if "/data/" in p or p.endswith((".csv", ".json", ".xlsx")):
        return "data"
    return "other"


def porcelain_entries(text: str) -> list[tuple[str, str]]:
    """(status code, path) of each ``git status --porcelain`` line."""
    out = []
    for line in text.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.append((line[:2], path.strip('"')))
    return out


def status_block(root: str, entries: list[tuple[str, str]]) -> list[str]:
    """The branch, the uncommitted entries as counts by kind, and the recent commits."""
    head = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if not head:
        raise RuntimeError("git state unavailable")
    ab = git(root, "rev-list", "--left-right", "--count", "HEAD...@{u}").split()
    ab_s = f"ahead {ab[0]}, behind {ab[1]}" if len(ab) == 2 else "no upstream"
    out = [f"Branch {head}, {ab_s}."]
    if entries:
        counts: dict[str, int] = {}
        for _, path in entries:
            counts[kind(path)] = counts.get(kind(path), 0) + 1
        untracked = sum(1 for c, _ in entries if c == "??")
        staged = sum(1 for c, _ in entries if c[0] not in " ?")
        by_kind = ", ".join(
            f"{k} {n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        out.append(
            f"Uncommitted {len(entries)}: {by_kind}; "
            f"{untracked} untracked, {staged} staged."
        )
    else:
        out.append("Working tree clean.")
    log = git(root, "log", "-3", "--date=short", "--format=%h %ad %s").splitlines()
    if log:
        out.append("Recent commits: " + " | ".join(s[:60] for s in log) + ".")
    return out


def dev_notes_text(root: str) -> str | None:
    """The root dev_notes.md, or None when it is absent or unreadable."""
    try:
        return (Path(root) / "dev_notes.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def dev_notes_dates(root: str) -> list[str]:
    """The dates of the dated ``##`` entries of the root dev_notes.md, in file order."""
    text = dev_notes_text(root) or ""
    pat = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})")
    return [m.group(1) for m in map(pat.match, text.splitlines()) if m]


def _day(ts: float) -> str:
    return dt.date.fromtimestamp(ts).isoformat()


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _words(text: object, n: int) -> str:
    w = str(text).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _names(items: list[str]) -> str:
    return ", ".join(items[:SHOWN]) + (", ..." if len(items) > SHOWN else "")


def age_line(
    root: str, entries: list[tuple[str, str]], now: float | None = None
) -> str | None:
    """The last commit's age, the oldest uncommitted change and the entries since."""
    now = time.time() if now is None else now
    ct = git(root, "log", "-1", "--format=%ct").strip()
    if not ct.isdigit():
        return None
    commit_day = _day(int(ct))
    days = int((now - int(ct)) // DAY)
    parts = [f"Last commit {_plural(days, 'day')} ago ({commit_day})"]
    mtimes = []
    for _, path in entries:
        p = Path(root) / path
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    if mtimes:
        parts.append(f"oldest uncommitted change {_day(min(mtimes))}")
    since = sum(1 for d in dev_notes_dates(root) if d > commit_day)
    parts.append(
        f"{since} dev_notes " + ("entry" if since == 1 else "entries") + " since"
    )
    return "; ".join(parts) + "."


def branches_line(root: str, now: float | None = None) -> str | None:
    """The unmerged local branches whose last commit is older than BRANCH_AGE_DAYS."""
    now = time.time() if now is None else now
    unmerged = set(
        git(root, "branch", "--no-merged", "HEAD", "--format=%(refname:short)").split()
    )
    refs = git(
        root,
        "for-each-ref",
        "--format=%(refname:short)\t%(committerdate:unix)",
        "refs/heads",
    )
    old = []
    for line in refs.splitlines():
        name, _, ct = line.partition("\t")
        if name in unmerged and ct.isdigit() and now - int(ct) > BRANCH_AGE_DAYS * DAY:
            old.append((int(ct), name))
    if not old:
        return None
    old.sort()
    names = ", ".join(n for _, n in old[:3]) + (", ..." if len(old) > 3 else "")
    return (
        f"Unmerged branches older than {BRANCH_AGE_DAYS} days: {len(old)} "
        f"({names}; oldest {_day(old[0][0])})."
    )


def worktrees(root: str) -> list[str]:
    """The worktrees besides main, each with its branch and last commit."""
    blocks: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in [*git(root, "worktree", "list", "--porcelain").splitlines(), ""]:
        if not line:
            if cur:
                blocks.append(cur)
                cur = {}
        else:
            k, _, v = line.partition(" ")
            cur[k] = v
    extra = blocks[1:]
    if not extra:
        return ["Worktrees: main only."]
    parts = []
    for w in extra[:4]:
        last = git(w["worktree"], "log", "-1", "--format=%h %s").strip()[:60]
        branch = Path(w.get("branch", "detached")).name
        parts.append(f"{Path(w['worktree']).name} [{branch}] {last}")
    return ["Worktrees besides main: " + "; ".join(parts) + "."]


def _local_python_count() -> int | None:
    """The count of python.exe and pythonw.exe processes, or None when unavailable.

    The count leaves out this process, and its parent launcher when the parent is also a
    python process.
    """
    if os.name != "nt":
        return None
    import ctypes  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415 - Windows only

    class PE(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - the ctypes layout
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t),
            ("mod", wintypes.DWORD),
            ("thr", wintypes.DWORD),
            ("ppid", wintypes.DWORD),
            ("pri", ctypes.c_long),
            ("flags", wintypes.DWORD),
            ("exe", ctypes.c_wchar * 260),
        ]

    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(2, 0)
    pe = PE()
    pe.dwSize = ctypes.sizeof(PE)
    procs = {}
    ok = k.Process32FirstW(snap, ctypes.byref(pe))
    while ok:
        procs[pe.pid] = (pe.exe.lower(), pe.ppid)
        ok = k.Process32NextW(snap, ctypes.byref(pe))
    k.CloseHandle(snap)
    names = ("python.exe", "pythonw.exe")
    mine = os.getpid()
    skip = {mine}
    parent = procs.get(mine, ("", 0))[1]
    if procs.get(parent, ("", 0))[0] in names:
        skip.add(parent)
    return sum(1 for pid, (exe, _) in procs.items() if exe in names and pid not in skip)


def limits() -> tuple[int, Callable[[], int | None]]:
    """The ceiling and the process counter, from _limits.py or the local fallback.

    ``.claude/hooks/_limits.py`` holds the one ceiling the guard and the brief share. When it
    cannot be imported the brief uses FALLBACK_CEILING and its own count.
    """
    mod = load_module("_limits", HOOKS / "_limits.py")
    if mod is None:
        return FALLBACK_CEILING, _local_python_count
    ceiling = int(getattr(mod, "CEILING", FALLBACK_CEILING))
    fn = getattr(mod, "python_count", None)
    if callable(fn):
        return ceiling, fn
    return ceiling, _local_python_count


def python_line() -> str:
    """The interpreter count against the ceiling, or why there is none."""
    ceiling, counter = limits()
    try:
        n = counter()
    except Exception:  # noqa: BLE001 - a failed count is reported, not raised
        n = None
    if n is None:
        if os.name != "nt":
            return "Python processes: ceiling not enforced on this OS."
        return "Python processes: count unavailable."
    full = " (at the ceiling)" if n >= ceiling else ""
    return f"Python interpreter processes running: {n} of {ceiling}{full}."


def lf_sha256(data: bytes) -> str:
    """The SHA-256 of ``data`` with each CRLF read as LF, the bytes git stores for text."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


#: A manifest row: the backticked file name, its backticked SHA-256, and any later cells.
MANIFEST_ROW = re.compile(
    r"^\|\s*`([\w.-]+\.py)`\s*\|\s*`([0-9a-f]{64})`\s*\|(.*)$", re.M
)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def manifest_status(root: str) -> dict[str, Any] | None:
    """The manifest's rows checked against ``lib/``, or None when the manifest is unreadable.

    The keys are ``rows`` (the module count), ``date`` (the copy date, from the prose
    "copied in on YYYY-MM-DD" or the newest date in a row's later cells), ``differ`` (the
    modules whose LF bytes do not hash to their row) and ``crlf_only`` (those of ``differ``
    whose raw working bytes do).
    """
    lib = Path(root) / LIB_DIR
    try:
        text = (Path(root) / MANIFEST).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    rows = MANIFEST_ROW.findall(text)
    dates = [d for _, _, rest in rows for d in ISO_DATE.findall(rest)]
    prose = re.search(r"copied in on (\d{4}-\d{2}-\d{2})", text)
    date = prose.group(1) if prose else (max(dates) if dates else "")
    if not rows and not date:
        return None
    differ, crlf_only = [], []
    for name, sha, _ in rows:
        try:
            data = (lib / name).read_bytes()
        except OSError:
            differ.append(name)
            continue
        if lf_sha256(data) == sha:
            continue
        differ.append(name)
        if hashlib.sha256(data).hexdigest() == sha:
            crlf_only.append(name)
    return {"rows": len(rows), "date": date, "differ": differ, "crlf_only": crlf_only}


def library_line(root: str) -> str | None:
    """The module count and the date of the copy lib/MANIFEST.md records, or None.

    The line also counts the modules whose SHA-256 differs from the manifest.
    """
    st = manifest_status(root)
    if st is None:
        return None
    on = f", copied in on {st['date']}" if st["date"] else ""
    line = f"Library: {st['rows']} modules{on}"
    if not st["rows"]:
        return line + "."
    differ, crlf = st["differ"], st["crlf_only"]
    if not differ:
        return f"{line}; all match the manifest."
    out = (
        f"{line}; {len(differ)} of {st['rows']} modules differ from the manifest "
        f"({_names(differ)})"
    )
    if crlf:
        out += f", {len(crlf)} of them only in CRLF line endings (an older manifest)"
    return out + "."


def lib_change(root: str) -> tuple[float, str] | None:
    """The time and the kind of the newest change to ``lib/``, or None when none is known.

    The candidates are the manifest's modification time (a copy), the newest module's
    modification time (an edit) and the newest commit that touched ``lib/``.
    """
    lib = Path(root) / LIB_DIR
    found: list[tuple[float, str]] = []
    try:
        found.append(((Path(root) / MANIFEST).stat().st_mtime, "lib copy"))
    except OSError:
        pass
    try:
        mtimes = [p.stat().st_mtime for p in lib.glob("*.py")]
    except OSError:
        mtimes = []
    if mtimes:
        found.append((max(mtimes), "lib module change"))
    ct = git(root, "log", "-1", "--format=%ct", "--", LIB_DIR).strip()
    if ct.isdigit():
        found.append((float(ct), "lib commit"))
    return max(found) if found else None


def _count(value: object) -> int:
    """A staleness count: an integer, or the sum of a nested mapping's counts."""
    if isinstance(value, dict):
        return sum(_count(v) for v in value.values())
    try:
        return int(value)  # type: ignore[call-overload]  # a JSON number or numeric string
    except (TypeError, ValueError):
        return 0


def staleness_line(root: str) -> str:
    """The notebook counts from tools/.cache/staleness.json when newer than ``lib/``.

    The cache is read only when it was written after the newest change to ``lib/``
    (``lib_change``); otherwise the line says that it is stale.
    """
    path = Path(root) / "tools" / ".cache" / "staleness.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        written = path.stat().st_mtime
    except (OSError, ValueError):
        return "Notebook staleness: no cache; run python tools/nb_staleness.py."
    change = lib_change(root)
    if change is not None and written < change[0]:
        return (
            f"Notebook staleness cache stale (written {_day(written)}, before the "
            f"{change[1]} of {_day(change[0])}); run python tools/nb_staleness.py."
        )
    counts = data.get("counts") if isinstance(data, dict) else None
    counts = counts if isinstance(counts, dict) else {}
    parts = ", ".join(
        f"{_count(counts.get(k, 0))} {k}" for k in ("current", "stale", "unstamped")
    )
    return f"Notebook staleness ({_day(written)}): {parts}."


def hooklog_line(root: str) -> str | None:
    """Counts from .claude/logs/hooks.jsonl, one JSON object per hook decision, if present.

    The log is the one ``.claude/hooks/_log.py`` writes (keys ``time``, ``hook``, ``kind``,
    ``rule``). An entry is counted by its kind (block, ask or warn) and under its rule, or
    its hook when it names no rule.
    """
    path = Path(root) / ".claude" / "logs" / "hooks.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    counts: dict[str, int] = {}
    kinds: dict[str, int] = {}
    first = ""
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:  # noqa: PERF203 - one malformed line is skipped
            continue
        if not isinstance(rec, dict):
            continue
        first = first or str(rec.get("time") or "")[:10]
        key = str(rec.get("rule") or rec.get("hook") or "unnamed")
        counts[key] = counts.get(key, 0) + 1
        k = str(rec.get("kind") or "entry")
        kinds[k] = kinds.get(k, 0) + 1
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    since = f" since {first}" if first else ""
    by_kind = ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
    listed = ", ".join(f"{k} {n}" for k, n in top)
    return f"Hook log: {sum(counts.values())} entries{since} ({by_kind}); {listed}."


def doctor_line(root: str) -> str | None:
    """One line from tools/.cache/doctor.json, or None while the report is absent.

    The keys read are ``ok``, ``problems`` (a list of strings) and ``interpreter``.
    """
    path = Path(root) / "tools" / ".cache" / "doctor.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        when = _day(path.stat().st_mtime)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    problems = [str(p) for p in data.get("problems") or []]
    interp = data.get("interpreter")
    with_interp = f", interpreter {interp}" if interp else ""
    if data.get("ok") and not problems:
        return f"Doctor ({when}): ok{with_interp}."
    listed = "; ".join(problems[:3]) or "not ok"
    return (
        f"Doctor ({when}): {_plural(len(problems), 'problem')}{with_interp}: {listed}."
    )


def dev_notes_line(root: str) -> str:
    """The newest dated entry of the root dev_notes.md, by date and title."""
    text = dev_notes_text(root)
    if text is None:
        return "Newest dev_notes entry: file not found."
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}")
    best = None
    for line in text.splitlines():
        s = line.strip()
        title = None
        h = re.match(r"#+\s+(.*)", s)
        b = re.match(r"\*\*(.+?)\*\*", s)
        if h:
            title = h.group(1)
        elif b:
            title = b.group(1)
        if not title:
            continue
        d = date_re.search(title)
        if d and (best is None or d.group(0) > best[0]):
            best = (d.group(0), title.strip().rstrip(".").strip())
    if best is None:
        return "Newest dev_notes entry: no dated heading found."
    title = best[1]
    if title.startswith(best[0]):
        title = title[len(best[0]) :].lstrip(" :-")
    return f"Newest dev_notes entry: {best[0]}, {_words(title, 16)}."


def open_decisions(text: str) -> list[str]:
    """The ``D-YYYYMMDD-n`` ids the text raises and no line closes, oldest first.

    A line closes an id when it reads "Closed D-..." or "Decided D-...", or when the id is
    followed by "closed" or "decided".
    """
    raised = {m.group(0) for m in DECISION_ID.finditer(text)}
    closed: set[str] = set()
    for line in text.splitlines():
        closed.update(m.group(1) for m in CLOSED_BEFORE.finditer(line))
        closed.update(m.group(1) for m in CLOSED_AFTER.finditer(line))

    def key(i: str) -> tuple[str, int]:
        _, date, n = i.split("-")
        return date, int(n)

    return sorted(raised - closed, key=key)


def decisions_line(root: str) -> str | None:
    """The count of open decisions and the oldest id, or None when dev_notes.md has none."""
    text = dev_notes_text(root)
    if text is None or not DECISION_ID.search(text):
        return None
    ids = open_decisions(text)
    if not ids:
        return "Open decisions: none open."
    return f"Open decisions: {len(ids)}, oldest {ids[0]}."


def record_baseline(root: str, hook: dict[str, Any]) -> None:
    """Record the tree signature for the clock-out check; any failure is ignored."""
    sid = hook.get("session_id")
    if not isinstance(sid, str) or not sid:
        return
    mod = load_module("session_close", HOOKS / "session_close.py")
    if mod is None:
        return
    try:
        mod.record_baseline(Path(root), sid, str(hook.get("source") or ""))
    except Exception:  # noqa: S110 - the clock-out baseline is advisory
        pass


def _opt(fn: Callable[..., str | None], *args: Any) -> list[str]:
    """A one-line block as a list, empty when the line is None or its source fails."""
    try:
        line = fn(*args)
    except Exception:  # noqa: BLE001 - a failed block is dropped
        return []
    return [line] if line else []


def build_brief(root: str, hook: dict[str, Any]) -> str:
    """The brief's text, each block cut to its budget."""
    entries = porcelain_entries(git(root, "status", "--porcelain"))
    with ThreadPoolExecutor(12) as ex:
        fs: dict[str, Future[Any]] = {
            "status": ex.submit(status_block, root, entries),
            "age": ex.submit(_opt, age_line, root, entries),
            "branches": ex.submit(_opt, branches_line, root),
            "worktrees": ex.submit(worktrees, root),
            "python": ex.submit(_opt, python_line),
            "library": ex.submit(_opt, library_line, root),
            "staleness": ex.submit(_opt, staleness_line, root),
            "hooklog": ex.submit(_opt, hooklog_line, root),
            "doctor": ex.submit(_opt, doctor_line, root),
            "dev_notes": ex.submit(_opt, dev_notes_line, root),
            "decisions": ex.submit(_opt, decisions_line, root),
            "baseline": ex.submit(record_baseline, root, hook),
        }
        res = {k: f.result() for k, f in fs.items()}  # status raises without git
    src = hook.get("source")
    via = f" (source: {src})" if isinstance(src, str) and src else ""
    blocks: dict[str, list[str]] = {
        "head": [f"Session state brief, built by the SessionStart hook{via}."]
    }
    for key in BUDGETS:
        if key != "head":
            blocks[key] = list(res[key] or [])
    lines: list[str] = []
    for key, budget in BUDGETS.items():
        lines.extend(clip_block(blocks.get(key, []), budget))
    return "\n".join(lines)


def main() -> None:
    """Print the brief for the project at CLAUDE_PROJECT_DIR."""
    # titles carry Greek letters; the Windows console is cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    hook = read_hook_input()
    root = project_root()
    if not Path(root).is_dir():
        raise FileNotFoundError(root)
    sys.stdout.write(build_brief(root, hook) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - the hook fails open (CODE_STYLE.md 13.4)
        sys.stdout.write(f"Session state brief unavailable ({type(e).__name__}).\n")
    sys.exit(0)
