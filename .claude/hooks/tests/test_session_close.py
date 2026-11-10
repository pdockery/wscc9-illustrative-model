"""Tests for .claude/hooks/session_close.py, the Stop and PreCompact clock-out check.

Standard library unittest. Run from the repository root:

    python -m unittest discover -s .claude/hooks/tests -p "test_session_close.py" -v

Each test builds a throwaway git repository in a temporary folder, so no file is written into
the working tree. The tests need git on PATH and skip without it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HOOKS = Path(__file__).resolve().parents[1]
SCRIPT = HOOKS / "session_close.py"
sys.path.insert(0, str(HOOKS))
import session_close as sc  # noqa: E402

HAS_GIT = shutil.which("git") is not None
TODAY = dt.date.today().isoformat()


def run_git(root: Path, *args: str) -> None:
    """Run git in ``root`` with a throwaway identity, raising on failure."""
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.org",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(root),
            *args,
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )


def write(root: Path, rel: str, text: str) -> Path:
    """Write ``text`` at ``root/rel`` with LF endings, creating folders."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))
    return p


def run_hook(
    root: Path, payload: dict, maintainer: bool = False
) -> subprocess.CompletedProcess:
    """Run the hook as Claude Code does, with the payload on stdin."""
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(root))
    env.pop(sc.MAINTAINER_VAR, None)
    if maintainer:
        env[sc.MAINTAINER_VAR] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=False,
    )


def iso(ts: float) -> str:
    """The local ISO 8601 time, with its offset, of a POSIX time."""
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


@unittest.skipUnless(HAS_GIT, "git not found")
class ClockOutFixture(unittest.TestCase):
    """A committed repository with a library, its manifest, a deck notebook and a record."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        run_git(self.root, "init", "-q")
        write(self.root, "dev_notes.md", "# Notes\n\n## 2026-11-01: an old entry\n")
        data = "X = 1\n"
        write(self.root, "lib/mod_01.py", data)
        sha = hashlib.sha256(data.encode()).hexdigest()
        write(
            self.root,
            "lib/MANIFEST.md",
            "# Library manifest\n\nCopied in on 2026-11-01.\n\n"
            f"| module | SHA-256 |\n|---|---|\n| `mod_01.py` | `{sha}` |\n",
        )
        write(self.root, "note_01.ipynb", "{}\n")
        write(self.root, "note_02.ipynb", "{}\n")
        write(
            self.root,
            "tests/test_slides_note_01.py",
            'NOTEBOOK = "note_01.ipynb"\n',
        )
        run_git(self.root, "add", "-A")
        run_git(self.root, "commit", "-q", "-m", "base")
        sys.modules.pop("session_brief", None)

    def tearDown(self) -> None:
        sys.modules.pop("session_brief", None)
        self.tmp.cleanup()

    def baseline(self, sid: str = "s1") -> None:
        self.assertTrue(sc.record_baseline(self.root, sid, "startup"))

    def note(self, changed: list[str] | None = None, maintainer: bool = False) -> str:
        return sc.build_note(self.root, TODAY, TODAY, changed, maintainer)


class BaselineTest(ClockOutFixture):
    """The baseline is kept once per session and never reaches git status."""

    def test_record_baseline_keeps_the_first_signature(self) -> None:
        self.baseline()
        first = sc.read_state(sc.state_file(self.root, "s1"))
        write(self.root, "new.txt", "x\n")
        self.assertFalse(sc.record_baseline(self.root, "s1", "compact"))
        again = sc.read_state(sc.state_file(self.root, "s1"))
        self.assertEqual(first["signature"], again["signature"])
        ignore = self.root / ".claude" / ".state" / ".gitignore"
        self.assertEqual(ignore.read_bytes(), b"*\n")

    def test_state_is_written_with_lf(self) -> None:
        self.baseline()
        self.assertNotIn(b"\r", sc.state_file(self.root, "s1").read_bytes())

    def test_state_folder_never_shows_in_git_status(self) -> None:
        self.baseline()
        out = subprocess.run(
            ["git", "-C", str(self.root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertNotIn(".claude", out)

    def test_signature_moves_when_a_dirty_file_changes_again(self) -> None:
        write(self.root, "lib/mod_01.py", "X = 2\n")
        first = sc.tree_signature(self.root)
        (self.root / "lib/mod_01.py").write_bytes(b"X = 3  # longer\n")
        self.assertNotEqual(first, sc.tree_signature(self.root))

    def test_changed_since_leaves_out_entries_dirty_before_the_session(self) -> None:
        write(self.root, "note_02.ipynb", '{"before": 1}\n')
        self.baseline()
        write(self.root, "note_01.ipynb", '{"during": 1}\n')
        state = sc.read_state(sc.state_file(self.root, "s1"))
        changed = sc.changed_since(state["entries"], sc.entry_signatures(self.root))
        self.assertEqual(changed, ["note_01.ipynb"])


class HookTest(ClockOutFixture):
    """The hook as Claude Code runs it: silent cases, once per event, and the two forms."""

    def test_silent_without_a_baseline(self) -> None:
        write(self.root, "new.txt", "x\n")
        r = run_hook(self.root, {"session_id": "none", "hook_event_name": "Stop"}, True)
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_silent_when_the_tree_is_unchanged(self) -> None:
        self.baseline()
        r = run_hook(self.root, {"session_id": "s1", "hook_event_name": "Stop"}, True)
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_silent_when_stop_hook_active(self) -> None:
        self.baseline()
        write(self.root, "new.txt", "x\n")
        payload = {
            "session_id": "s1",
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }
        r = run_hook(self.root, payload, True)
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_silent_for_a_contributor_with_nothing_to_report(self) -> None:
        self.baseline()
        write(self.root, "new.txt", "x\n")
        r = run_hook(self.root, {"session_id": "s1", "hook_event_name": "Stop"})
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        state = sc.read_state(sc.state_file(self.root, "s1"))
        self.assertEqual(state["fired"], [])

    def test_stop_fires_once_with_additional_context(self) -> None:
        self.baseline()
        write(self.root, "new.txt", "x\n")
        payload = {"session_id": "s1", "hook_event_name": "Stop"}
        r = run_hook(self.root, payload, True)
        self.assertEqual(r.returncode, 0)
        out = json.loads(r.stdout)
        hso = out["hookSpecificOutput"]
        self.assertEqual(hso["hookEventName"], "Stop")
        self.assertIn(f"has no entry dated {TODAY}", hso["additionalContext"])
        self.assertNotIn("decision", out)
        again = run_hook(self.root, payload, True)
        self.assertEqual(again.stdout, "")

    def test_contributor_sees_the_other_checks(self) -> None:
        self.baseline()
        write(self.root, "lib/mod_01.py", "X = 2\n")
        r = run_hook(self.root, {"session_id": "s1", "hook_event_name": "Stop"})
        note = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("1 lib/ modules differ from lib/MANIFEST.md (mod_01.py)", note)
        self.assertNotIn("dev_notes.md", note)
        self.assertNotIn("five parts", note)

    def test_precompact_writes_summary_instructions_to_stdout(self) -> None:
        self.baseline()
        write(self.root, "new.txt", "x\n")
        r = run_hook(
            self.root, {"session_id": "s1", "hook_event_name": "PreCompact"}, True
        )
        self.assertTrue(r.stdout.startswith("Carry this into the summary."))
        stop = run_hook(
            self.root, {"session_id": "s1", "hook_event_name": "Stop"}, True
        )
        self.assertIn("additionalContext", stop.stdout)

    def test_silent_on_another_event(self) -> None:
        self.baseline()
        write(self.root, "new.txt", "x\n")
        payload = {"session_id": "s1", "hook_event_name": "SubagentStop"}
        r = run_hook(self.root, payload, True)
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        state = sc.read_state(sc.state_file(self.root, "s1"))
        self.assertEqual(state["fired"], [])

    def test_fails_open_on_bad_input(self) -> None:
        env = dict(os.environ, CLAUDE_PROJECT_DIR=str(self.root))
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input="{not json",
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertIn("session_close: skipped", r.stderr)


class RenderTest(unittest.TestCase):
    """The two output forms of the hooks reference."""

    def test_stop_output_matches_the_documented_contract(self) -> None:
        out = json.loads(sc.render("Stop", "a note"))
        self.assertEqual(set(out), {"hookSpecificOutput"})
        self.assertEqual(
            out["hookSpecificOutput"],
            {"hookEventName": "Stop", "additionalContext": "a note"},
        )

    def test_precompact_output_is_plain_text_under_the_cap(self) -> None:
        text = sc.render("PreCompact", "x" * 20000)
        self.assertLess(len(text), 10000)
        self.assertTrue(text.rstrip().endswith("[cut here]"))
        with self.assertRaises(ValueError):
            json.loads(text)
        stop = json.loads(sc.render("Stop", "y" * 20000))
        self.assertLessEqual(
            len(stop["hookSpecificOutput"]["additionalContext"]), sc.NOTE_CAP
        )

    def test_maintainer_reads_the_variable(self) -> None:
        with mock.patch.dict(os.environ, {sc.MAINTAINER_VAR: "yes"}):
            self.assertTrue(sc.maintainer())
        with mock.patch.dict(os.environ, {sc.MAINTAINER_VAR: " "}):
            self.assertFalse(sc.maintainer())


class NoteTest(ClockOutFixture):
    """Each check of the note, on its own."""

    def test_maintainer_note_names_todays_entry_and_the_five_parts(self) -> None:
        write(self.root, "dev_notes.md", f"# Notes\n\n## {TODAY}: this session\n")
        note = self.note(maintainer=True)
        self.assertIn(f"has an entry dated {TODAY}", note)
        self.assertIn("five parts", note)
        self.assertIn("D-YYYYMMDD-n", note)
        self.assertIn("A record entry names no other repository.", note)

    def test_contributor_note_is_empty_with_nothing_to_report(self) -> None:
        self.assertEqual(self.note(), "")

    def test_a_clean_library_says_nothing(self) -> None:
        self.assertNotIn("MANIFEST", self.note(maintainer=True))

    def test_a_crlf_working_copy_matches_the_lf_manifest(self) -> None:
        (self.root / "lib/mod_01.py").write_bytes(b"X = 1\r\n")
        self.assertNotIn("MANIFEST", self.note(maintainer=True))

    def test_a_deck_notebook_without_a_ledger_gets_the_reminder(self) -> None:
        write(self.root, "note_01.ipynb", '{"x": 1}\n')
        note = self.note(["note_01.ipynb"])
        self.assertIn("Deck notebooks changed in this session: note_01.ipynb", note)
        self.assertIn(sc.SLIDES_COMMAND, note)

    def test_a_notebook_no_slide_check_names_is_left_out(self) -> None:
        write(self.root, "note_02.ipynb", '{"x": 1}\n')
        self.assertEqual(self.note(["note_02.ipynb"]), "")

    def test_every_root_notebook_counts_without_slide_checks(self) -> None:
        (self.root / "tests/test_slides_note_01.py").unlink()
        write(self.root, "note_02.ipynb", '{"x": 1}\n')
        self.assertIn("note_02.ipynb", self.note(["note_02.ipynb"]))

    def test_the_ledger_clears_a_notebook_only_after_its_change(self) -> None:
        nb = write(self.root, "note_01.ipynb", '{"x": 1}\n')
        mtime = nb.stat().st_mtime
        ledger = [
            {"ts": iso(mtime - 120), "tier": "slides", "ok": True},
            {"ts": iso(mtime + 120), "tier": "slides", "ok": False},
            {"ts": iso(mtime + 120), "tier": "tests", "ok": True},
        ]
        write(
            self.root,
            "tools/.cache/check_runs.jsonl",
            "\n".join(map(json.dumps, ledger)) + "\nnot json\n",
        )
        note = self.note(["note_01.ipynb"])
        self.assertIn("No passing slides run in tools/.cache/check_runs.jsonl", note)
        ledger.append(
            {"ts": iso(mtime + 120), "tiers": ["tests", "slides"], "ok": True}
        )
        write(
            self.root,
            "tools/.cache/check_runs.jsonl",
            "\n".join(map(json.dumps, ledger)) + "\n",
        )
        self.assertEqual(self.note(["note_01.ipynb"]), "")

    def test_note_names_an_untracked_module_a_modified_file_imports(self) -> None:
        write(self.root, "lib/mod_01.py", "import mod_types\nX = 1\n")
        write(self.root, "lib/mod_types.py", "T = int\n")
        write(self.root, "lib/unused_mod.py", "U = 1\n")
        found = sc.untracked_imports(self.root)
        self.assertEqual(found, [("lib/mod_types.py", ["mod_01.py"])])
        self.assertIn("mod_types.py (by mod_01.py)", self.note())

    def test_from_import_counts(self) -> None:
        write(self.root, "lib/mod_01.py", "from lib import helper_mod\n")
        write(self.root, "lib/helper_mod.py", "H = 1\n")
        self.assertEqual(len(sc.untracked_imports(self.root)), 1)

    def test_note_names_a_cited_path_that_git_ignores(self) -> None:
        write(self.root, ".gitignore", "outputs/\n")
        run_git(self.root, "add", ".gitignore")
        run_git(self.root, "commit", "-q", "-m", "ignore")
        write(self.root, "outputs/table_x.csv", "a\n")
        write(
            self.root,
            "dev_notes.md",
            "# Notes\n\n## 2026-11-01: old\n\nCites `outputs/old.csv`.\n\n"
            f"## {TODAY}: this session\n\nThe table is `outputs/table_x.csv`; "
            "the library is `lib/mod_01.py`.\n",
        )
        cited = sc.ignored_citations(self.root, TODAY)
        self.assertEqual(cited, [("outputs/table_x.csv", "dev_notes.md")])
        self.assertIn("outputs/table_x.csv (dev_notes.md)", self.note())

    def test_record_lines_added_in_the_working_tree_are_read(self) -> None:
        write(self.root, ".gitignore", "outputs/\n")
        write(self.root, "NOTEBOOK_RECORD.md", "# Record\n")
        run_git(self.root, "add", "-A")
        run_git(self.root, "commit", "-q", "-m", "record")
        write(self.root, "outputs/run.csv", "a\n")
        write(
            self.root,
            "NOTEBOOK_RECORD.md",
            "# Record\n\nThe run wrote `outputs/run.csv`.\n",
        )
        cited = sc.ignored_citations(self.root, "2099-01-01")
        self.assertEqual(cited, [("outputs/run.csv", "NOTEBOOK_RECORD.md")])


if __name__ == "__main__":
    unittest.main()
