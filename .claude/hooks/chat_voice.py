"""Stop hook: log the tells of machine prose in the agent's last chat reply.

The hook reads the reply that just ended, from the payload's ``last_assistant_message`` or else
from the transcript, and runs on its prose the tells of
``.claude/skills/writing-voice/scripts/style_check.py`` (``find_tells`` and ``pair_tells``), and
three chat rules: math written in LaTeX, which the chat does not render, an em dash in running
prose, and a question that asks what the writer should do. Each hit is one line in ``.claude/logs/hooks.jsonl`` with kind ``voice``, so the session
brief counts the tells by rule. The hook never blocks and prints nothing: a reply streams to the
participant as it is written, so a block could only append a correction to a reply already read.

Code blocks, inline code, table rows and quoted passages are left out of the scan. The rule of
three and the scare-quote tells are left out too, because a reply lists real triples and quotes
file names. The hook fails open on any error. Standard library only (CODE_STYLE.md 13.3).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

HOOKS = Path(__file__).resolve().parent
sys.path.insert(0, str(HOOKS))
import _log  # noqa: E402

#: The style check whose tells the hook reads.
STYLE_CHECK = (
    HOOKS.parents[0] / "skills" / "writing-voice" / "scripts" / "style_check.py"
)
#: Tells left out of a chat scan, by the start of their name. A reply addresses the participant,
#: so the reader-address tell, which counts every "you", gives way to DRAMA below.
SKIP_TELLS = ("rule of three", "scare quotes", "reader address")
#: Reader address as drama, the forms register/cut_on_sight.md names, in a chat reply.
DRAMA = re.compile(
    r"\b(?:notice (?:that|how)|look at (?:who|what|how)|watch (?:what|how)|"
    r"here['’]s the (?:thing|catch|kicker|short version)|the (?:catch|trap|kicker) is|let['’]s)\b",
    re.I,
)
#: A question that asks what the writer should do ("Should I hold?"). A question whether to
#: proceed is kept (user instruction 2026-11-12).
PERMISSION = re.compile(r"\bshould I\b[^.!?\n]*\?", re.I)
#: The most lines one reply adds to the log.
MAX_HITS = 12
#: LaTeX in a chat reply: a dollar-delimited span, a \( or \[ delimiter, or a math command.
LATEX = re.compile(
    r"(?<![\\\w$])\$[^$\n]{1,80}\$(?!\w)|\\\(|\\\[|\\(?:frac|hat|bar|sum|mathcal|mathrm|lambda|mu|"
    r"theta|le|ge|cdot|times|text|sqrt|alpha|beta|gamma|delta|sigma|Delta)(?![A-Za-z])"
)
EM_DASH = re.compile("—")
FENCE = re.compile(r"```.*?(```|$)", re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
QUOTE = re.compile(r'"[^"\n]{12,}"|“[^”\n]{12,}”')


def load_style_check() -> ModuleType | None:
    """The style check module, or None when it cannot be loaded."""
    try:
        spec = importlib.util.spec_from_file_location("style_check", STYLE_CHECK)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return None
    else:
        return mod


def reply_text(payload: dict) -> str:
    """The last reply's text: the payload's field, or the transcript's assistant text after the last user turn."""
    text = payload.get("last_assistant_message")
    if isinstance(text, str) and text.strip():
        return text
    path = payload.get("transcript_path")
    if not path or not Path(path).is_file():
        return ""
    parts: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if rec.get("type") == "user":
            is_prompt = isinstance(content, str) or any(
                isinstance(c, dict) and c.get("type") == "text" for c in content or []
            )
            if is_prompt:
                parts = []
        elif rec.get("type") == "assistant" and isinstance(content, list):
            parts += [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
    return "\n\n".join(p for p in parts if p)


def prose_lines(text: str) -> list[str]:
    """The reply's prose, one paragraph or list item per entry, without code, tables, links or quotations."""
    text = FENCE.sub("\n", text)
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("|", ">")):
            continue
        s = re.sub(r"^(#+|[-*+]|\d+[.)])\s+", "", s)
        s = LINK.sub(r"\1", INLINE_CODE.sub(" code ", s))
        out.append(QUOTE.sub(" quotation ", s))
    return out


def check(text: str, sc: ModuleType | None) -> list[tuple[str, str]]:
    """Each tell in the reply as (rule, the sentence or excerpt that carries it)."""
    hits: list[tuple[str, str]] = []
    bare = INLINE_CODE.sub(" ", FENCE.sub("\n", text))
    for m in LATEX.finditer(bare):
        hits.append(("chat LaTeX math", bare[max(0, m.start() - 40) : m.end() + 40]))
        break
    lines = prose_lines(text)
    hits.extend(("chat em dash", line) for line in lines if EM_DASH.search(line))
    hits.extend(("chat reader address", line) for line in lines if DRAMA.search(line))
    hits.extend(("chat permission question", line) for line in lines if PERMISSION.search(line))
    if sc is not None:
        sents = [s for line in lines for s in sc.sentences(line)]
        found = sc.find_tells(sents) + (
            sc.pair_tells(sents) if hasattr(sc, "pair_tells") else []
        )
        hits += [(n, s) for n, s in found if not n.startswith(SKIP_TELLS)]
    return hits[:MAX_HITS]


def main(stdin: str, log_dir: Path | None = None) -> int:
    """Read the Stop payload, log each tell of the last reply, and return 0 whatever happens."""
    try:
        payload = json.loads(stdin or "{}")
        text = reply_text(payload)
        if not text:
            return 0
        for rule, excerpt in check(text, load_style_check()):
            _log.record(
                "chat_voice",
                "voice",
                rule,
                excerpt,
                str(payload.get("session_id", "")),
                log_dir=log_dir,
            )
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.stdin.read()))
