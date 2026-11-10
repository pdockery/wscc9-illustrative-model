"""Tests for .claude/hooks/notebook_check.py, the PostToolUse check of nbformat edits.

Standard library unittest. Run from the repository root:

    python -m unittest discover -s .claude/hooks/tests -v

Each hook test builds a temporary git repository with a committed notebook at its root, as the
deck notebooks sit, writes the edit a patch script would make, and calls the hook with the shell
command that ran the script. No file is written into the working tree. The tests that need ruff
skip when ruff is not found.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HOOKS = Path(__file__).resolve().parents[1]
REPO = HOOKS.parents[1]
SCRIPT = HOOKS / "notebook_check.py"
sys.path.insert(0, str(HOOKS))
import code_check as cc  # noqa: E402
import notebook_check as nc  # noqa: E402

RUFF = cc.find_ruff() if "CODE_CHECK_RUFF" not in os.environ else None
needs_ruff = unittest.skipIf(RUFF is None, "ruff not found")
HAS_GIT = shutil.which("git") is not None


def notebook(sources):
    """A minimal nbformat-4 notebook; a source starting with '# ' is a markdown cell."""
    cells = []
    for i, src in enumerate(sources):
        kind = "markdown" if src.startswith("# ") else "code"
        cell = {"cell_type": kind, "id": f"c{i}", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def payload(command, cwd, tool="Bash"):
    return json.dumps(
        {
            "tool_name": tool,
            "tool_input": {"command": command},
            "cwd": str(cwd),
            "session_id": "nbcheck-test",
            "hook_event_name": "PostToolUse",
        }
    )


def context(out):
    if not out:
        return None
    hso = json.loads(out)["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    return hso["additionalContext"]


def run_git(root, *args):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "core.hooksPath=",
            *args,
        ],
        cwd=str(root),
        check=True,
        capture_output=True,
    )


class Tokens(unittest.TestCase):
    def test_bare_and_quoted_scripts(self):
        cmd = "python a.py && python \"C:/x y/b.py\" ; python 'c.py' > out.txt"
        self.assertEqual(nc.tokens(nc._PY_TOKEN, cmd), ["a.py", "C:/x y/b.py", "c.py"])

    def test_pyc_and_words_ignored(self):
        self.assertEqual(nc.tokens(nc._PY_TOKEN, "ls x.pyc; echo copy"), [])

    def test_notebook_tokens(self):
        text = (
            'nb = ROOT / "workshop3_transactions_across_boundaries.ipynb"; p=a/b.ipynb'
        )
        self.assertEqual(
            nc.tokens(nc._NB_TOKEN, text),
            ["workshop3_transactions_across_boundaries.ipynb", "p=a/b.ipynb"],
        )

    @unittest.skipUnless(os.name == "nt", "Git Bash drive paths are a Windows matter")
    def test_git_bash_drive(self):
        self.assertEqual(
            nc.as_path("/c/Users/x/p.py", Path("D:/")).as_posix(), "C:/Users/x/p.py"
        )


class Detection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_inline_nbformat(self):
        cmd = 'python -c "import nbformat; nbformat.read(...)"'
        self.assertEqual(nc.nbformat_texts(cmd, self.base), [cmd])

    def test_script_importing_nbformat(self):
        (self.base / "patch.py").write_text("import nbformat\n", encoding="utf-8")
        texts = nc.nbformat_texts("python patch.py", self.base)
        self.assertEqual(texts, ["import nbformat\n"])

    def test_script_without_nbformat(self):
        (self.base / "other.py").write_text("import json\n", encoding="utf-8")
        self.assertIsNone(nc.nbformat_texts("python other.py", self.base))

    def test_missing_script(self):
        self.assertIsNone(nc.nbformat_texts("python gone.py", self.base))

    def test_no_python_no_work(self):
        out = nc.run_hook(payload("git status", self.base), root=self.base)
        self.assertEqual(out, "")


class NamedNotebooks(unittest.TestCase):
    """A named notebook is found against the command's folder, then against the root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "scratch"
        self.base.mkdir()
        self.nb = self.root / "101_sandbox.ipynb"
        self.nb.write_text("{}", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_found_at_root_by_bare_name(self):
        found = nc.named_notebooks(["P = ROOT / '101_sandbox.ipynb'"], self.base, self.root)
        self.assertEqual(found, [self.root / "101_sandbox.ipynb"])

    def test_found_against_base_first(self):
        local = self.base / "101_sandbox.ipynb"
        local.write_text("{}", encoding="utf-8")
        found = nc.named_notebooks(["'101_sandbox.ipynb'"], self.base, self.root)
        self.assertEqual(found, [local])

    def test_absent_notebook_dropped(self):
        self.assertEqual(nc.named_notebooks(["'gone.ipynb'"], self.base, self.root), [])

    def test_no_subfolder_lookup_by_name(self):
        """A bare name is not searched for in subfolders."""
        (self.root / "notes").mkdir()
        (self.root / "notes" / "deep.ipynb").write_text("{}", encoding="utf-8")
        self.assertEqual(nc.named_notebooks(["'deep.ipynb'"], self.base, self.root), [])


@unittest.skipUnless(HAS_GIT, "git not found")
class Hook(unittest.TestCase):
    """The hook on a temporary repository whose notebook a patch script has changed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._state = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        run_git(self.root, "init", "-q")
        self.nb = self.root / "workshop9_test.ipynb"
        self.base_cells = ["# Title", "x = 1\n", "def broken(:\n"]
        self.nb.write_text(json.dumps(notebook(self.base_cells)), encoding="utf-8")
        shutil.copy(REPO / "ruff.toml", self.root / "ruff.toml")
        run_git(self.root, "add", "workshop9_test.ipynb", "ruff.toml")
        run_git(self.root, "commit", "-q", "-m", "base")
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        (self.scratch / "patch.py").write_text(
            "import nbformat\nnb = nbformat.read('workshop9_test.ipynb', 4)\n",
            encoding="utf-8",
        )
        self.env = mock.patch.dict(
            os.environ, {"CODE_CHECK_STATE_DIR": self._state.name}
        )
        self.env.start()
        os.environ.pop("CODE_CHECK_RUFF", None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()
        self._state.cleanup()

    def edit(self, cells):
        self.nb.write_text(json.dumps(notebook(cells)), encoding="utf-8")

    def hook(self, command="python scratch/patch.py", ruff=None, now=None):
        with mock.patch.object(cc, "find_ruff", return_value=ruff):
            return nc.run_hook(
                payload(command, self.root), root=self.root, ruff=ruff, now=now
            )

    def test_new_broken_cell_reported_old_one_not(self):
        self.edit([*self.base_cells, "y = (\n"])
        ctx = context(self.hook())
        self.assertIn("syntax cell 3 line 1", ctx)
        self.assertNotIn("cell 2", ctx)
        self.assertIn("1 cell differs from HEAD", ctx)
        self.assertIn("nbformat", ctx)
        self.assertIn(cc.RUFF_MISSING, ctx)

    def test_unchanged_notebook_silent(self):
        self.assertEqual(self.hook(), "")

    def test_old_write_outside_window_silent(self):
        self.edit([*self.base_cells, "y = (\n"])
        self.assertEqual(self.hook(now=time.time() + nc.WINDOW + 60), "")

    def test_command_without_nbformat_silent(self):
        self.edit([*self.base_cells, "y = (\n"])
        (self.scratch / "plain.py").write_text("print(1)\n", encoding="utf-8")
        self.assertEqual(self.hook("python scratch/plain.py"), "")

    def test_untracked_notebook_checks_every_cell(self):
        other = self.root / "workshop8_new.ipynb"
        other.write_text(json.dumps(notebook(["a = (\n", "b = 1\n"])), encoding="utf-8")
        ctx = context(self.hook())
        self.assertIn("workshop8_new.ipynb (2 cells differ from HEAD", ctx)
        self.assertIn("syntax cell 0 line 1", ctx)

    def test_untracked_notebook_rows_capped(self):
        other = self.root / "workshop8_new.ipynb"
        cells = [f"v{i} = (\n" for i in range(nc.MAX_ROWS + 4)]
        other.write_text(json.dumps(notebook(cells)), encoding="utf-8")
        ctx = context(self.hook())
        self.assertIn("... and 4 more.", ctx)
        self.assertEqual(ctx.count("- syntax cell"), nc.MAX_ROWS)

    def test_ignored_notebook_found_by_name_in_script(self):
        (self.root / ".gitignore").write_text("*_ignored.ipynb\n", encoding="utf-8")
        ign = self.root / "draft_ignored.ipynb"
        ign.write_text(json.dumps(notebook(["c = (\n"])), encoding="utf-8")
        (self.scratch / "patch2.py").write_text(
            "import nbformat\nP = ROOT / 'draft_ignored.ipynb'\n",
            encoding="utf-8",
        )
        ctx = context(self.hook("python scratch/patch2.py"))
        self.assertIn("draft_ignored.ipynb", ctx)

    def test_checkpoint_copy_not_checked(self):
        ck = self.root / ".ipynb_checkpoints"
        ck.mkdir()
        (ck / "workshop9_test-checkpoint.ipynb").write_text(
            json.dumps(notebook(["q = (\n"])), encoding="utf-8"
        )
        self.assertEqual(self.hook(), "")

    def test_damaged_notebook_reported(self):
        self.nb.write_text("{not json", encoding="utf-8")
        self.assertIn("does not parse as JSON", context(self.hook()))

    def test_prose_lint_findings_filtered_to_changed_cells(self):
        stub = self.root / nc.PROSE_LINT
        stub.parent.mkdir(parents=True)
        findings = [
            {
                "rule": "R-A",
                "severity": "error",
                "cell": 3,
                "pin": "cell 3 line 1",
                "message": "changed-cell finding",
                "excerpt": "e",
            },
            {
                "rule": "R-B",
                "severity": "warning",
                "cell": 0,
                "pin": "cell 0",
                "message": "old-cell finding",
                "excerpt": "e",
            },
            {
                "rule": "R-C",
                "severity": "error",
                "cell": None,
                "pin": "file",
                "message": "file finding",
                "excerpt": "e",
            },
        ]
        stub.write_text(
            "import json, sys\n"
            f"print(json.dumps({{'findings': {findings!r}}}))\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        self.edit([*self.base_cells, "# New text"])
        ctx = context(self.hook())
        self.assertIn("ERROR R-A cell 3 line 1: changed-cell finding", ctx)
        self.assertNotIn("old-cell finding", ctx)
        self.assertNotIn("file finding", ctx)

    def test_prose_lint_crash_fails_open(self):
        stub = self.root / nc.PROSE_LINT
        stub.parent.mkdir(parents=True)
        stub.write_text("raise SystemExit(2)\n", encoding="utf-8")
        self.edit([*self.base_cells, "# New text"])
        self.assertEqual(self.hook(), "")

    @needs_ruff
    def test_ruff_findings_in_changed_cells(self):
        self.edit(["# Title", "x = 1\n", "z = undefined_name\n"])
        ctx = context(self.hook(ruff=RUFF))
        self.assertIn("F821 cell 2 line 1", ctx)
        self.assertNotIn(cc.RUFF_MISSING, ctx)

    def test_powershell_command(self):
        self.edit([*self.base_cells, "y = (\n"])
        script = self.scratch / "patch.py"
        out = nc.run_hook(
            payload(f'& python "{script}"', self.root, tool="PowerShell"),
            root=self.root,
            ruff=None,
        )
        self.assertIn("syntax cell 3", context(out))


class FailOpen(unittest.TestCase):
    def test_script_exits_zero_on_bad_input(self):
        for raw in ("not json", "", "[1]", '{"tool_input": {"command": 3}}'):
            r = subprocess.run(
                [sys.executable, str(SCRIPT)],
                input=raw,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            self.assertEqual(r.returncode, 0, raw)
            self.assertEqual(r.stdout.strip(), "", raw)

    def test_script_fast_on_ordinary_command(self):
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=payload("git status --porcelain", REPO),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "")
        self.assertLess(time.perf_counter() - t0, 2.0)


if __name__ == "__main__":
    unittest.main()
