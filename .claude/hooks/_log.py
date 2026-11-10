"""The hooks' decision log: one JSON line per block or warning in `.claude/logs/hooks.jsonl`.

A hook calls `record(hook, kind, rule, detail)` when it blocks a call or warns, and the line
holds the time, the hook, the kind ("block", "ask" or "warn"), the rule's name, the session and
a shortened detail. The log answers which rules fire and how often, which no other record
holds. The folder is created on first use with a `.gitignore` of `*`, so the log never reaches
a commit.

The log fails open: any error in writing it is swallowed, and the hook's decision stands. The
file is rotated to `hooks.jsonl.1` when it passes `MAX_BYTES`. Standard library only
(CODE_STYLE.md 13.3).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

#: The log folder, `.claude/logs` beside this module's `.claude/hooks`.
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
#: The log file's name.
LOG_NAME = "hooks.jsonl"
#: The size past which the log is rotated once.
MAX_BYTES = 1_000_000
#: The longest detail kept in a line.
MAX_DETAIL = 300


def record(
    hook: str,
    kind: str,
    rule: str,
    detail: str = "",
    session: str = "",
    log_dir: Path | None = None,
) -> None:
    """Append one decision line to the log, and swallow every error in doing so."""
    try:
        folder = log_dir if log_dir is not None else LOG_DIR
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
        ignore = folder / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
        path = folder / LOG_NAME
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            path.replace(folder / (LOG_NAME + ".1"))
        line = {
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "hook": hook,
            "kind": kind,
            "rule": rule,
            "session": session,
            "detail": " ".join(detail.split())[:MAX_DETAIL],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        return
