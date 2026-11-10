"""Instruction-integrity tests for the agent's own instructions (standard library; PyYAML optional).

Run: python -m pytest tools/tests/test_instruction_integrity.py -q

The tests read the instruction files, not the model. They check that each skill and agent file
parses, that the model tiers the skills state match the agent files, that the paths the grounding
documents name exist, that every hook in `.claude/settings.json` runs on a canned payload and
exits 0, that the grounding documents agree once AGENTS.md and CANON.md exist, and that no tracked
text carries a banned name. The banned-name patterns live in an ignored local file
(`.git/info/banned_names.txt`, one regular expression per line, `exclude: <path regex>` lines for
exempt paths), and that test skips where the file is absent, so no tracked file lists the names.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # the frontmatter tests skip without PyYAML
    yaml = None

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"
AGENTS = ROOT / ".claude" / "agents"
TIERS = ("haiku", "sonnet", "opus")
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
READ_ONLY = re.compile(r"read-only|writes nothing|changes no file", re.IGNORECASE)

#: Grounding documents whose backticked paths must exist.
PATH_DOCS = (
    "CLAUDE.md",
    "AGENTS.md",
    "CANON.md",
    "README.md",
    "NOTEBOOK_RECORD.md",
    "tools/README.md",
)
PATH_EXT = r"(?:py|md|json|ipynb|txt|tex|bib|yml|yaml|toml|ini|js|bat|csv)"
PATH_TOKEN = re.compile(r"`([^`\s]+)`")


def git(*args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""


def frontmatter(path: Path) -> tuple[dict[str, object], str]:
    """Parse the leading YAML block; return (fields, raw block)."""
    text = path.read_bytes().decode("utf-8")
    if not text.startswith("---\n"):
        return {}, ""
    end = text.find("\n---", 4)
    block = text[4:end] if end > 0 else ""
    fields = yaml.safe_load(block) if yaml else {}
    return (fields if isinstance(fields, dict) else {}), block


def instruction_files() -> list[tuple[Path, str]]:
    """Every skill file and agent file with its expected name."""
    found = [(p, p.parent.name) for p in sorted(SKILLS.glob("*/SKILL.md"))]
    found += [(p, p.stem) for p in sorted(AGENTS.glob("*.md"))]
    return found


# ------------------------------------------------------------------ frontmatter ----------
class Frontmatter(unittest.TestCase):
    def setUp(self) -> None:
        if yaml is None:
            self.skipTest("PyYAML is not installed")

    def test_files_found(self) -> None:
        self.assertTrue(instruction_files())

    def test_every_skill_folder_has_a_skill_file(self) -> None:
        missing = [
            d.name
            for d in sorted(SKILLS.iterdir())
            if d.is_dir() and not (d / "SKILL.md").exists()
        ]
        self.assertEqual(missing, [])

    def test_parses_with_name_and_description(self) -> None:
        problems = []
        for path, expected in instruction_files():
            raw = path.read_bytes()
            rel = path.relative_to(ROOT).as_posix()
            if b"\r" in raw:
                problems.append(f"{rel} carries a carriage return")
            try:
                fields, _ = frontmatter(path)
            except yaml.YAMLError as exc:
                problems.append(f"{rel} frontmatter does not parse: {exc}")
                continue
            if fields.get("name") != expected:
                problems.append(
                    f"{rel} name is {fields.get('name')!r}, expected {expected!r}"
                )
            desc = fields.get("description")
            if not isinstance(desc, str) or not desc.strip():
                problems.append(f"{rel} has no description")
            elif len(desc) > 1024:
                problems.append(
                    f"{rel} description is {len(desc)} characters, over 1024"
                )
        self.assertEqual(problems, [])


# ------------------------------------------------------------------ roles ----------------
def agent_records() -> dict[str, dict[str, object]]:
    out = {}
    for path in sorted(AGENTS.glob("*.md")):
        fields, _ = frontmatter(path)
        out[path.stem] = fields
    return out


TIER_THEN_ROLES = re.compile(
    r"`(haiku|sonnet|opus)`\s*\(((?:`[a-z]+(?:-[a-z]+)*`(?:,\s*)?)+)\)"
)
ROLE_THEN_TIER = re.compile(r"`([a-z]+(?:-[a-z]+)+)`\s+on\s+`(haiku|sonnet|opus)`")


def tier_statements(text: str) -> list[tuple[str, str]]:
    """The (role, tier) pairs of a skill's tier statement, in both forms the skills use."""
    pairs = []
    for m in TIER_THEN_ROLES.finditer(text):
        pairs += [(r, m.group(1)) for r in re.findall(r"`([a-z-]+)`", m.group(2))]
    pairs += [(m.group(1), m.group(2)) for m in ROLE_THEN_TIER.finditer(text)]
    return pairs


class Roles(unittest.TestCase):
    def setUp(self) -> None:
        if yaml is None:
            self.skipTest("PyYAML is not installed")

    def test_model_is_a_named_tier(self) -> None:
        bad = {
            n: f.get("model")
            for n, f in agent_records().items()
            if f.get("model") not in TIERS
        }
        self.assertEqual(bad, {})

    def test_read_only_roles_carry_no_write_tool(self) -> None:
        bad = []
        for name, fields in agent_records().items():
            tools = {t.strip() for t in str(fields.get("tools", "")).split(",")}
            if (
                READ_ONLY.search(str(fields.get("description", "")))
                and tools & WRITE_TOOLS
            ):
                bad.append(f"{name}: {sorted(tools & WRITE_TOOLS)}")
        self.assertEqual(bad, [])

    def test_skill_tier_statements_match_agent_files(self) -> None:
        agents = agent_records()
        problems = []
        seen = 0
        for path in sorted(SKILLS.glob("*/SKILL.md")):
            for role, tier in tier_statements(path.read_text(encoding="utf-8")):
                seen += 1
                rel = path.relative_to(ROOT).as_posix()
                if role not in agents:
                    problems.append(
                        f"{rel} names `{role}` beside `{tier}`, and no agent file defines it"
                    )
                elif agents[role].get("model") != tier:
                    problems.append(
                        f"{rel} puts `{role}` on `{tier}`, the agent file says `{agents[role].get('model')}`"
                    )
        self.assertGreater(seen, 0, "no tier statement found")
        self.assertEqual(problems, [])


# ------------------------------------------------------------------ paths ----------------
def looks_like_path(token: str) -> bool:
    """A repository path, a directory under one, or a file name; not a ratio, a command or a URL."""
    if re.search(r"[<>*{}$|=\"'(),;!?]|^-|^https?:|^\.\.\.|\\", token):
        return False
    if token.startswith("/"):
        return False  # a slash command such as /orientation
    if re.search(r"(?<=[\w-])\." + PATH_EXT + r"(?::\d+(?:-\d+)?)?$", token):
        return not token.startswith("~") and " " not in token
    if "/" in token:
        # no extension: a path only when it starts at a top-level folder of the repository
        return token.split("/", 1)[0] in {p.name for p in ROOT.iterdir() if p.is_dir()}
    return False


def inline_tokens(text: str) -> list[str]:
    """Backticked tokens outside fenced code blocks."""
    text = re.sub(r"(?s)```.*?```", "", text)
    return PATH_TOKEN.findall(text)


def basename_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for rel in git("ls-files", "-co", "--exclude-standard").splitlines():
        index.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    return index


def index_paths(index: dict[str, list[str]]) -> list[str]:
    return [rel for rels in index.values() for rel in rels]


class Paths(unittest.TestCase):
    """A backticked path in a grounding document, a prompt, a skill or an agent file exists."""

    def resolves(self, doc: Path, token: str, index: dict[str, list[str]]) -> bool:
        token = re.sub(r":\d+(?:-\d+)?$", "", token)
        bare = token.rstrip("/")
        if not bare:
            return True
        # the document's own folder and each folder above it, up to the repository root
        for base in (doc.parent, *doc.parent.parents):
            if (base / bare).exists():
                return True
            if base == ROOT:
                break
        if "/" not in bare and bare in index:
            return True
        # a path written from a skill's own folder (register/td.md) or a folder named on its own (workshops/)
        if any(
            ("/" + rel + "/").find("/" + bare + "/") >= 0 or rel.endswith("/" + bare)
            for rel in index_paths(index)
        ):
            return True
        # a path the repository ignores (a cache, an output folder, a local-only source) is legitimate
        for candidate in (token, bare + "/"):
            r = subprocess.run(
                ["git", "-C", str(ROOT), "check-ignore", "-q", "--", candidate],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if r.returncode == 0:
                return True
        return False

    def test_backticked_paths_exist(self) -> None:
        docs = [ROOT / d for d in PATH_DOCS]
        docs += sorted((ROOT / "prompts").glob("*.md"))
        docs += sorted(SKILLS.rglob("*.md")) + sorted(AGENTS.glob("*.md"))
        index = basename_index()
        problems = []
        for doc in docs:
            if not doc.exists():
                continue
            rel = doc.relative_to(ROOT).as_posix()
            tokens = inline_tokens(doc.read_text(encoding="utf-8"))
            problems.extend(
                f"{rel}: `{token}`"
                for token in tokens
                if looks_like_path(token) and not self.resolves(doc, token, index)
            )
        self.assertEqual(sorted(set(problems)), [])


# ------------------------------------------------------------------ hooks ----------------
def hook_payload(event: str, matcher: str) -> dict[str, object]:
    tool = (matcher.split("|")[0] if matcher else "") or "Bash"
    target = "tools/_common.py"
    if event == "SessionStart":
        return {
            "hook_event_name": event,
            "source": "startup",
            "session_id": "integrity-test",
        }
    if tool in ("Bash", "PowerShell"):
        inp: dict[str, object] = {"command": "git status --short"}
    elif tool == "NotebookEdit":
        inp = {
            "notebook_path": target,
            "cell_id": "x",
            "new_source": "x",
            "edit_mode": "replace",
        }
    elif tool == "Edit" or tool == "MultiEdit":
        inp = {"file_path": target, "old_string": "x", "new_string": "x"}
    else:
        inp = {"file_path": target, "content": "x"}
    return {
        "hook_event_name": event,
        "session_id": "integrity-test",
        "tool_name": tool,
        "tool_input": inp,
        "tool_response": {"success": True},
    }


def settings_hooks() -> list[tuple[str, str, str]]:
    settings = json.loads(
        (ROOT / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    found = []
    for event, entries in settings.get("hooks", {}).items():
        for entry in entries:
            found.extend(
                (event, entry.get("matcher", ""), hook["command"])
                for hook in entry.get("hooks", [])
                if hook.get("type") == "command"
            )
    return found


class Hooks(unittest.TestCase):
    """Each hook command runs on a canned payload aimed at tools/_common.py, exits 0 and prints no traceback."""

    def test_every_hook_command_runs(self) -> None:
        hooks = settings_hooks()
        self.assertTrue(hooks)
        env = dict(os.environ, CLAUDE_PROJECT_DIR=str(ROOT))
        problems = []
        for event, matcher, command in hooks:
            args = shlex.split(
                command.replace("$CLAUDE_PROJECT_DIR", ROOT.as_posix()), posix=True
            )
            if args and args[0] in ("python", "python3"):
                args[0] = sys.executable
            payload = json.dumps(hook_payload(event, matcher))
            r = subprocess.run(
                args,
                input=payload.encode("utf-8"),
                capture_output=True,
                cwd=str(ROOT),
                env=env,
                timeout=60,
                check=False,
            )
            err = r.stderr.decode("utf-8", "replace")
            if r.returncode != 0 or "Traceback" in err:
                problems.append(
                    f"{event} {command}: exit {r.returncode} {err.strip()[-300:]}"
                )
        self.assertEqual(problems, [])


# ------------------------------------------------------------------ grounding ------------
def text_files() -> list[Path]:
    out = []
    for rel in git("ls-files", "-co", "--exclude-standard").splitlines():
        p = ROOT / rel
        if (
            p.suffix.lower()
            in {".md", ".py", ".json", ".js", ".toml", ".ini", ".txt", ".yml", ".yaml"}
            and p.is_file()
        ):
            out.append(p)
    return out


class Grounding(unittest.TestCase):
    """Guards that bind once AGENTS.md and CANON.md exist; until then they skip."""

    def setUp(self) -> None:
        if not (ROOT / "AGENTS.md").exists() or not (ROOT / "CANON.md").exists():
            self.skipTest("AGENTS.md and CANON.md are not in place")

    def test_at_imports_resolve(self) -> None:
        problems = []
        for name in ("CLAUDE.md", "AGENTS.md"):
            doc = ROOT / name
            if not doc.exists():
                continue
            for line in doc.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^@(\S+)\s*$", line)
                if m and not (doc.parent / m.group(1)).exists():
                    problems.append(
                        f"{name} imports @{m.group(1)}, which does not exist"
                    )
        self.assertEqual(problems, [])

    def test_no_file_names_the_retired_canon_path(self) -> None:
        old = ".claude/skills/modify-example/" + "canon.md"
        records = {
            "dev_notes.md",
            "NOTEBOOK_RECORD.md",
        }  # dated entries keep their wording
        hits = [
            p.relative_to(ROOT).as_posix()
            for p in text_files()
            if p.name not in records
            and old in p.read_text(encoding="utf-8", errors="replace")
        ]
        self.assertEqual(hits, [])


# ------------------------------------------------------------------ banned names ---------
def banned_file() -> Path | None:
    out = git("rev-parse", "--git-path", "info/banned_names.txt").strip()
    if not out:
        return None
    p = Path(out)
    p = p if p.is_absolute() else ROOT / p
    return p if p.exists() else None


def sweep_files(exclude: list[re.Pattern[str]]) -> list[tuple[str, str]]:
    """(relative path, text) of every non-ignored text file outside lib/ and the inherited tests."""
    out = []
    for rel in git("ls-files", "-co", "--exclude-standard").splitlines():
        if re.search(r"^(lib|tests)/", rel) or any(x.search(rel) for x in exclude):
            continue
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 4_000_000:
            continue
        raw = p.read_bytes()
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", "replace")
        if p.suffix == ".ipynb":
            try:
                text = "\n".join(
                    "".join(c.get("source", []))
                    for c in json.loads(text).get("cells", [])
                )
            except ValueError:
                continue
        out.append((rel, text))
    return out


def load_banned(path: Path) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    patterns, exclude = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("exclude:"):
            exclude.append(re.compile(line[len("exclude:") :].strip()))
        else:
            patterns.append(re.compile(line))
    return patterns, exclude


class BannedNames(unittest.TestCase):
    def test_no_banned_name_in_tracked_text(self) -> None:
        path = banned_file()
        if path is None:
            self.skipTest("no .git/info/banned_names.txt on this machine")
        patterns, exclude = load_banned(path)
        hits = []
        for rel, text in sweep_files(exclude):
            for pat in patterns:
                m = pat.search(text)
                if m:
                    hits.append(f"{rel}: {m.group(0)}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
