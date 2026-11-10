"""Tests for .claude/hooks/session_brief.py, the SessionStart state brief.

Standard library unittest. Run from the repository root:

    python -m unittest discover -s .claude/hooks/tests -p "test_session_brief.py" -v

The brief's git calls are answered by a fake, so the fixture can hold 60 dirty paths without a
repository of that size. Each test writes its inputs in a temporary root: three library
modules with their manifest, one notebook, a dev_notes.md with two dated entries and three
decision ids, the staleness and doctor caches and a hook log.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HOOKS = Path(__file__).resolve().parents[1]
SCRIPT = HOOKS / "session_brief.py"
sys.path.insert(0, str(HOOKS))
import session_brief as sb  # noqa: E402

NOW = time.time()
DAY = 86400
KINDS = ("py", "md", "ipynb", "tex", "csv")
MODULES = ("mod_01.py", "mod_02.py", "mod_03.py")
HEADERS = (
    "Session state brief",
    "Branch main",
    "Uncommitted 60:",
    "Recent commits:",
    "Last commit 5 days ago",
    "Unmerged branches older than 30 days: 2",
    "Worktrees: main only.",
    "Python interpreter processes running: 3 of 7.",
    "Library: 3 modules, copied in on 2026-11-02; all match the manifest.",
    "Notebook staleness (",
    "Hook log: 3 entries since 2026-11-09 (block 2, warn 1)",
    "Doctor (",
    "Newest dev_notes entry: 2026-11-08, the newest entry.",
    "Open decisions: 2, oldest D-20261101-2.",
)
DEV_NOTES = """# dev_notes.md

## 2026-11-08: the newest entry

**Held.** D-20261108-1 asks which tolerance the check keeps.
Closed D-20261101-1: the owner chose the first option.

## 2026-11-01: the first entry

**Open decisions.**
1. D-20261101-1 asks one question.
2. D-20261101-2 asks another.
"""


def write(root: Path, rel: str, data: str | bytes) -> Path:
    """Write ``data`` at ``root/rel`` as bytes, LF kept, creating folders."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return p


def manifest(rows: dict[str, str], date: str = "2026-11-02") -> str:
    """A lib/MANIFEST.md with one row per module name and SHA-256."""
    body = "".join(f"| `{name}` | `{sha}` |\n" for name, sha in rows.items())
    return (
        "# Library manifest\n\n"
        f"lib/ is a copy of a shared library, copied in on {date}.\n\n"
        "| module | SHA-256 |\n|---|---|\n" + body
    )


def dirty_porcelain(n: int = 60) -> str:
    """A porcelain listing of ``n`` dirty paths of mixed kinds, a fifth untracked."""
    lines = []
    for i in range(n):
        ext = KINDS[i % len(KINDS)]
        code = "??" if i % 5 == 0 else (" M" if i % 7 else "M ")
        lines.append(f"{code} folder{i % 9}/file_{i}.{ext}")
    return "\n".join(lines) + "\n"


class FakeGit:
    """Answers the brief's git calls from canned text, keyed by the subcommand."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.lib_commit: int | None = int(NOW - 3 * DAY)

    def __call__(self, where: str, *args: str) -> str:
        sub = args[0]
        if sub == "rev-parse":
            return "main\n"
        if sub == "rev-list":
            return "0\t0\n"
        if sub == "status":
            return dirty_porcelain()
        if sub == "log" and "%ct" in " ".join(args) and "--" in args:
            return "" if self.lib_commit is None else f"{self.lib_commit}\n"
        if sub == "log" and "%ct" in " ".join(args):
            return f"{int(NOW - 5 * DAY - 60)}\n"
        if sub == "log":
            return "abc1234 2026-11-04 First change\ndef5678 2026-11-03 Second change\n"
        if sub == "branch":
            return "old-a\nold-b\nfresh\n"
        if sub == "for-each-ref":
            old = int(NOW - 60 * DAY)
            return f"main\t{int(NOW)}\nold-a\t{old}\nold-b\t{old + 10}\nfresh\t{int(NOW)}\n"
        if sub == "worktree":
            return f"worktree {self.root}\nHEAD abc\nbranch refs/heads/main\n\n"
        return ""


class BriefFixture(unittest.TestCase):
    """A temporary root with every input the brief reads."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = {}
        for i, name in enumerate(MODULES):
            data = f"X = {i}\n".encode()
            write(self.root, f"lib/{name}", data)
            rows[name] = hashlib.sha256(data).hexdigest()
        write(self.root, "lib/MANIFEST.md", manifest(rows))
        write(self.root, "note_01.ipynb", "{}\n")
        write(self.root, "dev_notes.md", DEV_NOTES)
        cache = self.root / "tools" / ".cache"
        write(
            self.root,
            "tools/.cache/doctor.json",
            json.dumps({"ok": False, "interpreter": "3.12", "problems": ["p1", "p2"]}),
        )
        write(
            self.root,
            "tools/.cache/staleness.json",
            json.dumps(
                {
                    "generated": "2026-11-09T08:00:00Z",
                    "counts": {"current": 2, "stale": 1, "unstamped": 1},
                }
            ),
        )
        log = [
            {
                "time": "2026-11-09T08:00:00",
                "hook": "guard",
                "kind": "block",
                "rule": "a",
            },
            {
                "time": "2026-11-09T09:00:00",
                "hook": "guard",
                "kind": "block",
                "rule": "a",
            },
            {
                "time": "2026-11-09T10:00:00",
                "hook": "guard",
                "kind": "warn",
                "rule": "b",
            },
        ]
        write(
            self.root,
            ".claude/logs/hooks.jsonl",
            "\n".join(map(json.dumps, log)) + "\nnot json\n",
        )
        old = NOW - 20 * DAY
        for p in [*(self.root / "lib").iterdir()]:
            os.utime(p, (old, old))
        self.set_mtime(cache / "staleness.json", NOW - DAY)
        self.fake = FakeGit(str(self.root))
        self.patches = [
            mock.patch.object(sb, "git", self.fake),
            mock.patch.object(sb, "limits", lambda: (7, lambda: 3)),
            mock.patch.object(sb.time, "time", lambda: NOW),
        ]
        for p in self.patches:
            p.start()
        sys.modules.pop("session_close", None)

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        sys.modules.pop("session_close", None)
        self.tmp.cleanup()

    @staticmethod
    def set_mtime(path: Path, when: float) -> None:
        os.utime(path, (when, when))

    def brief(self, hook: dict | None = None) -> str:
        return sb.build_brief(str(self.root), hook or {"source": "startup"})


class BuildBriefTest(BriefFixture):
    """The whole brief over 60 dirty paths."""

    def test_every_block_header_appears(self) -> None:
        text = self.brief()
        for header in HEADERS:
            self.assertIn(header, text)

    def test_brief_stays_under_the_cap(self) -> None:
        self.assertLessEqual(len(self.brief().split()), sb.MAX_WORDS)

    def test_budgets_sum_to_the_cap(self) -> None:
        self.assertEqual(sum(sb.BUDGETS.values()), sb.MAX_WORDS)

    def test_uncommitted_block_is_counts_not_names(self) -> None:
        line = next(s for s in self.brief().splitlines() if s.startswith("Uncommitted"))
        self.assertNotIn("file_", line)
        self.assertIn("12 untracked", line)
        self.assertIn("py 12", line)

    def test_a_long_block_is_cut_to_its_budget_and_later_blocks_survive(self) -> None:
        long_line = " ".join(f"w{i}" for i in range(200))
        with mock.patch.object(sb, "hooklog_line", lambda root: long_line):
            text = self.brief()
        line = next(s for s in text.splitlines() if s.startswith("w0 "))
        self.assertEqual(len(line.split()), sb.BUDGETS["hooklog"] + 1)
        self.assertTrue(line.endswith(" ..."))
        self.assertIn("Newest dev_notes entry: 2026-11-08", text)
        self.assertIn("Open decisions: 2", text)

    def test_each_block_keeps_its_budget(self) -> None:
        long_line = " ".join(f"w{i}" for i in range(400))
        blocks = (
            ("age", "age_line"),
            ("library", "library_line"),
            ("staleness", "staleness_line"),
            ("doctor", "doctor_line"),
            ("dev_notes", "dev_notes_line"),
            ("decisions", "decisions_line"),
        )
        for key, name in blocks:
            with mock.patch.object(sb, name, lambda *a: long_line):
                text = self.brief()
            line = next(s for s in text.splitlines() if s.startswith("w0 "))
            self.assertEqual(len(line.split()), sb.BUDGETS[key] + 1, key)

    def test_baseline_is_recorded_for_the_session(self) -> None:
        self.brief({"session_id": "abc-1", "source": "startup"})
        state = self.root / ".claude" / ".state" / "abc-1.json"
        self.assertTrue(state.is_file())
        ignore = self.root / ".claude" / ".state" / ".gitignore"
        self.assertEqual(ignore.read_bytes(), b"*\n")

    def test_optional_blocks_drop_when_absent(self) -> None:
        for rel in ("tools/.cache/doctor.json", ".claude/logs/hooks.jsonl"):
            (self.root / rel).unlink()
        (self.root / "lib" / "MANIFEST.md").unlink()
        write(self.root, "dev_notes.md", "# Notes\n\n## 2026-11-08: one entry\n")
        text = self.brief()
        for absent in ("Doctor", "Hook log", "Library:", "Open decisions"):
            self.assertNotIn(absent, text)

    def test_brief_names_no_private_block(self) -> None:
        text = self.brief()
        for absent in ("Public clone", "reformat", "programmes", "Held for", "prompts"):
            self.assertNotIn(absent, text)


class StalenessLineTest(BriefFixture):
    """The cache is read only when it is newer than the newest change to lib/."""

    def cache(self) -> Path:
        return self.root / "tools" / ".cache" / "staleness.json"

    def test_cache_newer_than_every_lib_change_is_read(self) -> None:
        line = sb.staleness_line(str(self.root))
        self.assertIn("2 current, 1 stale, 1 unstamped", line)

    def test_cache_older_than_the_lib_commit_is_stale(self) -> None:
        self.set_mtime(self.cache(), NOW - 10 * DAY)
        line = sb.staleness_line(str(self.root))
        self.assertIn("cache stale", line)
        self.assertIn("before the lib commit of", line)
        self.assertIn("tools/nb_staleness.py", line)

    def test_cache_older_than_the_manifest_is_stale(self) -> None:
        self.fake.lib_commit = None
        self.set_mtime(self.root / "lib" / "MANIFEST.md", NOW)
        line = sb.staleness_line(str(self.root))
        self.assertIn("before the lib copy of", line)
        self.assertIn("tools/nb_staleness.py", line)

    def test_cache_older_than_a_module_edit_is_stale(self) -> None:
        self.fake.lib_commit = None
        self.set_mtime(self.root / "lib" / "mod_02.py", NOW)
        line = sb.staleness_line(str(self.root))
        self.assertIn("before the lib module change of", line)

    def test_missing_cache_is_reported(self) -> None:
        self.cache().unlink()
        self.assertIn("no cache", sb.staleness_line(str(self.root)))

    def test_nested_counts_are_summed(self) -> None:
        nested = {"counts": {"current": {"a": 2, "b": 1}, "stale": 0}}
        self.cache().write_bytes(json.dumps(nested).encode())
        self.set_mtime(self.cache(), NOW - DAY)
        line = sb.staleness_line(str(self.root))
        self.assertIn("3 current, 0 stale, 0 unstamped", line)


class LibraryLineTest(BriefFixture):
    """The library line counts the modules whose LF bytes differ from the manifest."""

    def test_a_changed_module_is_named(self) -> None:
        write(self.root, "lib/mod_02.py", "X = 99\n")
        line = sb.library_line(str(self.root))
        self.assertIn("1 of 3 modules differ from the manifest (mod_02.py)", line)
        self.assertNotIn("CRLF", line)

    def test_a_missing_module_differs(self) -> None:
        (self.root / "lib" / "mod_03.py").unlink()
        self.assertIn("(mod_03.py)", sb.library_line(str(self.root)))

    def test_crlf_working_bytes_match_an_lf_manifest(self) -> None:
        write(self.root, "lib/mod_01.py", b"X = 0\r\n")
        self.assertIn("all match the manifest", sb.library_line(str(self.root)))

    def test_a_crlf_manifest_row_is_counted_apart(self) -> None:
        data = b"Y = 1\r\nZ = 2\r\n"
        write(self.root, "lib/mod_01.py", data)
        rows = {"mod_01.py": hashlib.sha256(data).hexdigest()}
        write(self.root, "lib/MANIFEST.md", manifest(rows))
        line = sb.library_line(str(self.root))
        self.assertIn("1 of 1 modules differ", line)
        self.assertIn("1 of them only in CRLF line endings", line)

    def test_a_date_column_gives_the_copy_date(self) -> None:
        shas = [hashlib.sha256(f"X = {i}\n".encode()).hexdigest() for i in range(3)]
        text = "| file | SHA-256 | copied |\n|---|---|---|\n" + "".join(
            f"| `{n}` | `{shas[i]}` | 2026-11-0{i + 3} |\n"
            for i, n in enumerate(MODULES)
        )
        write(self.root, "lib/MANIFEST.md", "# Library manifest\n\n" + text)
        line = sb.library_line(str(self.root))
        self.assertIn("3 modules, copied in on 2026-11-05; all match", line)


class DecisionsTest(BriefFixture):
    """The open decisions are the D- ids raised and not closed."""

    def test_count_and_oldest(self) -> None:
        self.assertEqual(
            sb.decisions_line(str(self.root)), "Open decisions: 2, oldest D-20261101-2."
        )

    def test_closing_forms(self) -> None:
        text = (
            "D-20261101-1 raised. D-20261101-2 raised. D-20261101-3 raised.\n"
            "D-20261102-10 raised. D-20261102-9 raised.\n"
            "**Decided** D-20261101-1 for the first option.\n"
            "D-20261101-2 is closed by the copy.\n"
            "D-20261101-3 (decided 2026-11-04).\n"
        )
        self.assertEqual(sb.open_decisions(text), ["D-20261102-9", "D-20261102-10"])

    def test_all_closed(self) -> None:
        write(self.root, "dev_notes.md", "D-20261101-1 asks.\nClosed D-20261101-1.\n")
        self.assertEqual(
            sb.decisions_line(str(self.root)), "Open decisions: none open."
        )

    def test_no_ids_drops_the_line(self) -> None:
        write(
            self.root, "dev_notes.md", "# Notes\n\n1. An open decision without an id.\n"
        )
        self.assertIsNone(sb.decisions_line(str(self.root)))


class LineTest(unittest.TestCase):
    """The one-line blocks and helpers in isolation."""

    def test_clip_block_cuts_at_the_budget(self) -> None:
        out = sb.clip_block(["one two three", "four five six seven"], 5)
        self.assertEqual(out, ["one two three", "four five ..."])
        self.assertEqual(sb.clip_block(["a b"], 0), [])

    def test_ceiling_not_enforced_message_when_the_count_is_none(self) -> None:
        with (
            mock.patch.object(sb, "limits", lambda: (7, lambda: None)),
            mock.patch.object(sb.os, "name", "posix"),
        ):
            self.assertEqual(
                sb.python_line(), "Python processes: ceiling not enforced on this OS."
            )

    def test_at_the_ceiling(self) -> None:
        with mock.patch.object(sb, "limits", lambda: (7, lambda: 7)):
            self.assertIn("7 of 7 (at the ceiling)", sb.python_line())

    def test_limits_fall_back_when_the_shared_module_is_absent(self) -> None:
        with mock.patch.object(sb, "load_module", lambda name, path: None):
            ceiling, counter = sb.limits()
        self.assertEqual(ceiling, sb.FALLBACK_CEILING)
        self.assertIs(counter, sb._local_python_count)

    def test_limits_read_the_shared_module(self) -> None:
        if not (HOOKS / "_limits.py").is_file():
            self.skipTest("_limits.py not present")
        sys.modules.pop("_limits", None)
        ceiling, counter = sb.limits()
        self.assertIsInstance(ceiling, int)
        self.assertTrue(callable(counter))

    def test_porcelain_entries_take_the_rename_target(self) -> None:
        self.assertEqual(
            sb.porcelain_entries('R  old.py -> new.py\n?? "a b.md"\n'),
            [("R ", "new.py"), ("??", "a b.md")],
        )

    def test_lf_sha256_reads_crlf_as_lf(self) -> None:
        self.assertEqual(
            sb.lf_sha256(b"a\r\nb\n"), hashlib.sha256(b"a\nb\n").hexdigest()
        )


class ScriptTest(unittest.TestCase):
    """The hook as Claude Code runs it fails open outside a repository."""

    def test_outside_a_repository_prints_one_line_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, CLAUDE_PROJECT_DIR=d, GIT_CEILING_DIRECTORIES=d)
            r = subprocess.run(
                [sys.executable, str(SCRIPT)],
                input="{}",
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
                check=False,
            )
        self.assertEqual(r.returncode, 0)
        self.assertEqual(
            r.stdout.strip(), "Session state brief unavailable (RuntimeError)."
        )


if __name__ == "__main__":
    unittest.main()
