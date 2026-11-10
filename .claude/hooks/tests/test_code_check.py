"""Tests for .claude/hooks/code_check.py, the PostToolUse code lint.

Standard library unittest. Run from the repository root:

    python -m unittest discover -s .claude/hooks/tests -v

The tests that need ruff skip when ruff is not found. Each such test lints files in a temporary
root that holds a copy of the repository's ruff.toml, so no file is written into the working tree.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

HOOKS = Path(__file__).resolve().parents[1]
REPO = HOOKS.parents[1]
SCRIPT = HOOKS / "code_check.py"
sys.path.insert(0, str(HOOKS))
import code_check as cc  # noqa: E402

RUFF = cc.find_ruff() if "CODE_CHECK_RUFF" not in os.environ else None
needs_ruff = unittest.skipIf(
    RUFF is None or not (REPO / "ruff.toml").is_file(), "ruff or ruff.toml not found"
)


def payload(tool, path, **ti):
    ti["file_path"] = str(path)
    return json.dumps(
        {"tool_name": tool, "tool_input": ti, "hook_event_name": "PostToolUse"}
    )


def context(out):
    """The additionalContext text of a hook output, or None for an empty output."""
    if not out:
        return None
    data = json.loads(out)
    hso = data["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    return hso["additionalContext"]


class LineSpans(unittest.TestCase):
    def test_single_line(self):
        self.assertEqual(cc.line_spans("a\nb\nc\n", "b"), {2})

    def test_multi_line_and_trailing_newline(self):
        self.assertEqual(cc.line_spans("a\nb\nc\nd\n", "b\nc\n"), {2, 3})

    def test_every_occurrence(self):
        self.assertEqual(cc.line_spans("x = 1\ny = 2\nx = 1\n", "x = 1"), {1, 3})

    def test_empty_needle(self):
        self.assertEqual(cc.line_spans("a\n", ""), set())


class Ignored(unittest.TestCase):
    """Calls the hook ignores print nothing, with or without ruff."""

    def test_outside_repo(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.py"
            p.write_text("undefined_name\n", encoding="utf-8")
            self.assertEqual(
                cc.run_hook(payload("Write", p), root=REPO, ruff=RUFF or "ruff"), ""
            )

    def test_not_python(self):
        self.assertEqual(
            cc.run_hook(payload("Write", REPO / "CLAUDE.md"), root=REPO, ruff="ruff"),
            "",
        )

    def test_skipped_folders(self):
        for part in ("sources", "outputs", "sandbox_exhibits", ".ipynb_checkpoints"):
            self.assertIsNone(cc.rel_in_root(REPO / part / "x.py", REPO))
        self.assertIsNotNone(cc.rel_in_root(REPO / "lib" / "atc.py", REPO))

    def test_no_file_path(self):
        self.assertEqual(
            cc.run_hook(
                json.dumps({"tool_name": "Write", "tool_input": {}}), root=REPO
            ),
            "",
        )

    def test_non_dict_payload(self):
        self.assertEqual(cc.run_hook("[1, 2]", root=REPO), "")


class FindRuff(unittest.TestCase):
    """The search order: CODE_CHECK_RUFF, the interpreter's folder, CONDA_PREFIX, PATH, conda."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("CODE_CHECK_RUFF", None)
        os.environ.pop("CONDA_PREFIX", None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def exe(self, *parts):
        p = self.base.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return p

    def test_none_found(self):
        with (
            mock.patch.object(cc, "near_candidates", return_value=[]),
            mock.patch.object(cc, "conda_candidates", return_value=[]),
            mock.patch.object(cc.shutil, "which", return_value=None),
        ):
            self.assertIsNone(cc.find_ruff())

    def test_env_override_disables(self):
        with mock.patch.dict(os.environ, {"CODE_CHECK_RUFF": ""}):
            self.assertIsNone(cc.find_ruff())

    def test_env_override_wins(self):
        p = self.exe("chosen", "ruff")
        with mock.patch.dict(os.environ, {"CODE_CHECK_RUFF": str(p)}):
            self.assertEqual(cc.find_ruff(), str(p))

    def test_env_override_missing_file_is_none(self):
        with mock.patch.dict(os.environ, {"CODE_CHECK_RUFF": str(self.base / "gone")}):
            self.assertIsNone(cc.find_ruff())

    def test_conda_prefix_listed(self):
        with mock.patch.dict(os.environ, {"CONDA_PREFIX": str(self.base / "env")}):
            near = cc.near_candidates()
        self.assertIn(self.base / "env" / "Scripts" / "ruff.exe", near)
        self.assertIn(self.base / "env" / "bin" / "ruff", near)

    def test_interpreter_folder_listed_first(self):
        here = Path(sys.executable).parent
        self.assertEqual(cc.near_candidates()[0], here / "ruff")

    def test_conda_envs_in_name_order(self):
        b = self.exe("conda", "envs", "beta", "bin", "ruff")
        a = self.exe("conda", "envs", "alpha", "Scripts", "ruff.exe")
        found = [p for p in cc.conda_candidates((self.base / "conda",)) if p.is_file()]
        self.assertEqual(found, [a, b])

    def test_absent_conda_root_skipped(self):
        self.assertEqual(cc.conda_candidates((self.base / "none",)), [])

    def test_path_before_conda_envs(self):
        conda = self.exe("conda", "envs", "e", "bin", "ruff")
        with (
            mock.patch.object(cc, "near_candidates", return_value=[]),
            mock.patch.object(cc, "conda_candidates", return_value=[conda]),
            mock.patch.object(cc.shutil, "which", return_value="/on/path/ruff"),
        ):
            self.assertEqual(cc.find_ruff(), "/on/path/ruff")
        with (
            mock.patch.object(cc, "near_candidates", return_value=[]),
            mock.patch.object(cc, "conda_candidates", return_value=[conda]),
            mock.patch.object(cc.shutil, "which", return_value=None),
        ):
            self.assertEqual(cc.find_ruff(), str(conda))


class RuffAbsent(unittest.TestCase):
    def test_hook_says_ruff_missing_once_per_session(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as s:
            root = Path(d)
            (root / "ruff.toml").write_text(
                '[lint]\nselect = ["F"]\n', encoding="utf-8"
            )
            p = root / "x.py"
            p.write_text("undefined_name\n", encoding="utf-8")
            raw = json.loads(payload("Write", p))
            raw["session_id"] = "abc-123"
            raw = json.dumps(raw)
            with (
                mock.patch.object(cc, "find_ruff", return_value=None),
                mock.patch.dict(os.environ, {"CODE_CHECK_STATE_DIR": s}),
            ):
                os.environ.pop("CODE_CHECK_RUFF", None)
                self.assertEqual(context(cc.run_hook(raw, root=root)), cc.RUFF_MISSING)
                self.assertEqual(cc.run_hook(raw, root=root), "")
                other = raw.replace("abc-123", "def-456")
                self.assertEqual(
                    context(cc.run_hook(other, root=root)), cc.RUFF_MISSING
                )

    def test_missing_note_silent_when_disabled(self):
        with mock.patch.dict(os.environ, {"CODE_CHECK_RUFF": ""}):
            self.assertEqual(cc.ruff_missing_note({"session_id": "x"}), "")

    def test_missing_note_unwritable_flag_still_says_it(self):
        with tempfile.TemporaryDirectory() as s:
            blocker = Path(s) / "file"
            blocker.write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CODE_CHECK_STATE_DIR": str(blocker)}):
                os.environ.pop("CODE_CHECK_RUFF", None)
                self.assertEqual(cc.ruff_missing_note({}), cc.RUFF_MISSING)

    def test_script_silent_without_ruff(self):
        """The script itself, with ruff disabled, on a real library module: exit 0, no output."""
        env = dict(os.environ, CODE_CHECK_RUFF="")
        target = REPO / "lib" / "revenue_allocation.py"
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=payload("Write", target),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "")


class Malformed(unittest.TestCase):
    def test_script_fails_open(self):
        for raw in ("not json", "", "{", '{"tool_name": "Write", "tool_input": 3}'):
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

    def test_notebook_payload_fails_open(self):
        raw = '{"tool_name": "NotebookEdit", "tool_input": {"notebook_path": 3}}'
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=raw,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "")


@needs_ruff
class WithRuff(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        shutil.copy(REPO / "ruff.toml", self.root / "ruff.toml")
        (self.root / "lib").mkdir()
        self.path = self.root / "lib" / "mod.py"

    def tearDown(self):
        self._tmp.cleanup()

    def hook(self, raw):
        return context(cc.run_hook(raw, root=self.root, ruff=RUFF))

    def test_write_reports_whole_file(self):
        self.path.write_text(
            "import os\n\n\ndef f():\n    return undefined_name\n", encoding="utf-8"
        )
        ctx = self.hook(payload("Write", self.path))
        self.assertIn("whole file", ctx)
        self.assertIn("F401 line 1", ctx)
        self.assertIn("F821 line 5", ctx)

    def test_write_clean_prints_nothing(self):
        """A module that passes the lint and is formatted prints nothing."""
        src = '"""A clean module."""\n\n\ndef f(x: int) -> int:\n    """Return x."""\n    return x\n'
        self.path.write_text(src, encoding="utf-8")
        self.assertEqual(
            cc.run_hook(payload("Write", self.path), root=self.root, ruff=RUFF), ""
        )

    def test_write_caps_findings(self):
        body = '"""Caps."""\n\n' + "".join(f"v{i} = undefined_{i}\n" for i in range(20))
        self.path.write_text(body, encoding="utf-8")
        ctx = self.hook(payload("Write", self.path))
        self.assertEqual(ctx.count("\n- F821"), cc.MAX_WRITE)
        self.assertIn("... and 5 more", ctx)
        self.assertTrue(ctx.startswith("code_check: 20 findings"))

    def test_edit_keeps_touched_lines_only(self):
        new = "def g():\n    return other_undefined\n"
        self.path.write_text("import os\n\n\n" + new, encoding="utf-8")
        ctx = self.hook(payload("Edit", self.path, old_string="pass\n", new_string=new))
        self.assertIn("touched lines", ctx)
        self.assertIn("F821 line 5", ctx)
        self.assertNotIn("F401", ctx)

    def test_edit_outside_findings_prints_nothing(self):
        self.path.write_text(
            "import os\n\n\ndef g():\n    return 1\n", encoding="utf-8"
        )
        out = cc.run_hook(
            payload("Edit", self.path, old_string="return 0", new_string="return 1"),
            root=self.root,
            ruff=RUFF,
        )
        self.assertEqual(out, "")

    def test_edit_deletion_prints_nothing(self):
        self.path.write_text("import os\n", encoding="utf-8")
        out = cc.run_hook(
            payload("Edit", self.path, old_string="x = 1\n", new_string=""),
            root=self.root,
            ruff=RUFF,
        )
        self.assertEqual(out, "")

    def test_multiedit_unions_regions(self):
        self.path.write_text(
            "a = undefined_a\nimport os\nb = undefined_b\n", encoding="utf-8"
        )
        edits = [
            {"old_string": "x", "new_string": "a = undefined_a"},
            {"old_string": "y", "new_string": "b = undefined_b"},
        ]
        ctx = self.hook(payload("MultiEdit", self.path, edits=edits))
        self.assertIn("F821 line 1", ctx)
        self.assertIn("F821 line 3", ctx)
        self.assertNotIn("F401", ctx)

    def test_syntax_error_reported_anywhere(self):
        self.path.write_text("def f(:\n    pass\n\nx = 1\n", encoding="utf-8")
        ctx = self.hook(
            payload("Edit", self.path, old_string="x = 0", new_string="x = 1")
        )
        self.assertIn("syntax line 1", ctx)

    def test_relative_path_resolves_against_cwd(self):
        self.path.write_text("y = undefined_name\n", encoding="utf-8")
        raw = json.dumps(
            {
                "tool_name": "Write",
                "cwd": str(self.root),
                "tool_input": {"file_path": "lib/mod.py"},
            }
        )
        self.assertIn("F821", self.hook(raw))

    def test_unicode_source(self):
        self.path.write_text(
            '"""λ̂_a, the area energy price, in $/MWh."""\nmu = undefined_μ\n',
            encoding="utf-8",
        )
        self.assertIn("F821 line 2", self.hook(payload("Write", self.path)))

    def test_new_module_unformatted_reported(self):
        """A module outside the [format] exclude list that the formatter would change is reported."""
        self.path.write_text('"""New."""\n\nx = {  "a":1 }\n', encoding="utf-8")
        ctx = self.hook(payload("Write", self.path))
        self.assertIn("- format:", ctx)
        self.assertIn("CODE_STYLE.md section 14", ctx)

    def test_new_module_formatted_and_clean_prints_nothing_on_edit(self):
        self.path.write_text('"""New."""\n\nx = {"a": 1}\n', encoding="utf-8")
        out = cc.run_hook(
            payload("Edit", self.path, old_string="0", new_string='x = {"a": 1}'),
            root=self.root,
            ruff=RUFF,
        )
        self.assertEqual(out, "")

    def test_syntax_error_skips_format_check(self):
        self.path.write_text("def f(:\n    pass\n", encoding="utf-8")
        ctx = self.hook(payload("Write", self.path))
        self.assertIn("syntax line 1", ctx)
        self.assertNotIn("- format:", ctx)


@needs_ruff
class FormatConfig(unittest.TestCase):
    """The format check follows the [format] section of the root ruff.toml."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "lib").mkdir()
        self.path = self.root / "lib" / "mod.py"
        self.path.write_text('x = {  "a":1 }\n', encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_with(self, config_text):
        (self.root / "ruff.toml").write_text(config_text, encoding="utf-8")
        return cc.run_hook(payload("Write", self.path), root=self.root, ruff=RUFF)

    def test_no_format_section_no_check(self):
        self.assertEqual(self.run_with('[lint]\nselect = ["F"]\n'), "")

    def test_format_section_checks_new_module(self):
        self.assertIn(
            "- format:", context(self.run_with('[lint]\nselect = ["F"]\n\n[format]\n'))
        )

    def test_excluded_module_not_checked(self):
        cfg = '[lint]\nselect = ["F"]\n\n[format]\nexclude = ["lib/mod.py"]\n'
        self.assertEqual(self.run_with(cfg), "")


@needs_ruff
class RepoConfig(unittest.TestCase):
    """The repository's ruff.toml resolves its relative paths from the root, whatever the cwd."""

    def setUp(self):
        self.config = REPO / "ruff.toml"
        self._cwd = Path.cwd()
        os.chdir(tempfile.gettempdir())

    def tearDown(self):
        os.chdir(self._cwd)

    def test_existing_module_is_format_exempt(self):
        target = REPO / "lib" / "atc.py"
        self.assertFalse(cc.format_pending(RUFF, self.config, target))

    def test_per_file_ignores_apply(self):
        found = cc.run_ruff(RUFF, self.config, Path(__file__).resolve())
        self.assertFalse([f for f in found if (f.get("code") or "").startswith("PT0")])


@needs_ruff
class Speed(unittest.TestCase):
    def test_largest_library_module_under_budget(self):
        files = list((REPO / "lib").glob("*.py"))
        if not files:
            self.skipTest("no library modules")
        largest = max(files, key=lambda p: p.stat().st_size)
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=payload("Write", largest),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        dt = time.perf_counter() - t0
        self.assertEqual(r.returncode, 0)
        self.assertLess(dt, 1.5, f"{largest.name} took {dt:.2f} s")


def notebook(cells):
    """A minimal nbformat-4 notebook with the given (cell_type, source) cells."""
    return {
        "cells": [
            {
                "cell_type": kind,
                "id": f"c{i}",
                "metadata": {},
                "source": src,
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for i, (kind, src) in enumerate(cells)
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def nb_payload(path, new_source, **ti):
    ti.update(notebook_path=str(path), new_source=new_source)
    return json.dumps(
        {"tool_name": "NotebookEdit", "tool_input": ti, "session_id": "nb-test"}
    )


class Magics(unittest.TestCase):
    def test_line_magics_and_escapes_parse(self):
        src = "%matplotlib inline\n!pip install x\nfiles = !ls\nimport os\nos?\n"
        self.assertIsNone(cc.cell_syntax_error(src))

    def test_indented_escape_keeps_block(self):
        """The Colab bootstrap installs packages under an if."""
        src = "if IN_COLAB:\n    !pip install -q pypsa\nx = 1\n"
        self.assertIsNone(cc.cell_syntax_error(src))

    def test_python_cell_magic_body_parsed(self):
        self.assertIsNone(cc.cell_syntax_error("%%time\nx = 1\n"))
        self.assertEqual(cc.cell_syntax_error("%%time\nx = (\n")[0], 2)

    def test_other_cell_magic_skipped(self):
        self.assertIsNone(cc.cell_syntax_error("%%bash\nif [ -f x ]; then echo; fi\n"))

    def test_operator_line_not_rewritten(self):
        """A line that opens with % inside an expression parses as written."""
        self.assertIsNone(cc.cell_syntax_error("x = (10\n     % 3)\n"))

    def test_error_line_number_kept(self):
        line, msg = cc.cell_syntax_error("%load_ext autoreload\nx = 1\ndef f(:\n")
        self.assertEqual(line, 3)
        self.assertTrue(msg)

    def test_null_byte_reported(self):
        self.assertEqual(cc.cell_syntax_error("x = 1\0\n")[0], 1)

    def test_syntax_warning_kept_off_stderr(self):
        """An invalid escape (a LaTeX string without r) parses and warns nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(cc.cell_syntax_error('s = "\\$x$"\n'))

    def test_strip_keeps_line_count(self):
        src = "%time f()\n!ls\nx = 1"
        self.assertEqual(cc.strip_magics(src).count("\n"), src.count("\n"))


class EditedCells(unittest.TestCase):
    def setUp(self):
        self.nb = notebook(
            [("markdown", "# T"), ("code", "x = 1\n"), ("code", "y = 2")]
        )

    def test_by_source(self):
        self.assertEqual(cc.edited_cells(self.nb, {"new_source": "y = 2"}), {2})

    def test_by_stripped_source(self):
        self.assertEqual(cc.edited_cells(self.nb, {"new_source": "x = 1"}), {1})

    def test_by_cell_id_and_insert(self):
        ti = {"new_source": "zzz", "cell_id": "c1"}
        self.assertEqual(cc.edited_cells(self.nb, ti), {1})
        ti["edit_mode"] = "insert"
        self.assertEqual(cc.edited_cells(self.nb, ti), {2})

    def test_by_index_form(self):
        self.assertEqual(cc.edited_cells(self.nb, {"cell_id": "cell-2"}), {2})

    def test_list_source(self):
        nb = notebook([("code", ["a = 1\n", "b = 2"])])
        self.assertEqual(cc.edited_cells(nb, {"new_source": "a = 1\nb = 2"}), {0})

    def test_nothing_found(self):
        self.assertEqual(cc.edited_cells(self.nb, {"new_source": "zzz"}), set())


class NotebookEditHook(unittest.TestCase):
    """A NotebookEdit is checked by ast without ruff, and with ruff's F and E9 when ruff exists."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._state = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        shutil.copy(REPO / "ruff.toml", self.root / "ruff.toml")
        self.path = self.root / "workshop9_test.ipynb"
        self.env = mock.patch.dict(
            os.environ, {"CODE_CHECK_STATE_DIR": self._state.name}
        )
        self.env.start()
        os.environ.pop("CODE_CHECK_RUFF", None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()
        self._state.cleanup()

    def write(self, cells):
        self.path.write_text(json.dumps(notebook(cells)), encoding="utf-8")

    def test_syntax_error_without_ruff(self):
        bad = "def f(:\n    pass\n"
        self.write([("code", "import os\n"), ("code", bad)])
        with mock.patch.object(cc, "find_ruff", return_value=None):
            ctx = context(cc.run_hook(nb_payload(self.path, bad), root=self.root))
        self.assertIn("syntax cell 1 line 1", ctx)
        self.assertIn("edited cell 1", ctx)
        self.assertIn(cc.RUFF_MISSING, ctx)

    def test_clean_cell_without_ruff_says_note_only(self):
        self.write([("code", "x = 1\n")])
        with mock.patch.object(cc, "find_ruff", return_value=None):
            out = cc.run_hook(nb_payload(self.path, "x = 1\n"), root=self.root)
            self.assertEqual(context(out), cc.RUFF_MISSING)
            out = cc.run_hook(nb_payload(self.path, "x = 1\n"), root=self.root)
            self.assertEqual(out, "")

    def test_markdown_and_delete_ignored(self):
        self.write([("markdown", "def f(:")])
        for extra in ({"cell_type": "markdown"}, {"edit_mode": "delete"}):
            raw = nb_payload(self.path, "def f(:", **extra)
            self.assertEqual(cc.run_hook(raw, root=self.root, ruff="ruff"), "")

    def test_markdown_cell_found_by_source_ignored(self):
        self.write([("markdown", "def f(:")])
        with mock.patch.object(cc, "find_ruff", return_value=None):
            self.assertEqual(
                cc.run_hook(nb_payload(self.path, "def f(:"), root=self.root), ""
            )

    def test_outside_root_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.ipynb"
            p.write_text(json.dumps(notebook([("code", "def f(:")])), encoding="utf-8")
            self.assertEqual(
                cc.run_hook(nb_payload(p, "def f(:"), root=self.root, ruff="ruff"), ""
            )

    def test_checkpoint_copy_ignored(self):
        ck = self.root / ".ipynb_checkpoints"
        ck.mkdir()
        p = ck / "workshop9_test-checkpoint.ipynb"
        p.write_text(json.dumps(notebook([("code", "def f(:")])), encoding="utf-8")
        self.assertEqual(
            cc.run_hook(nb_payload(p, "def f(:"), root=self.root, ruff="ruff"), ""
        )

    def test_damaged_notebook_ignored(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(
            cc.run_hook(nb_payload(self.path, "x"), root=self.root, ruff="ruff"), ""
        )

    def test_python_file_payload_not_taken_as_notebook(self):
        """A Write of a .py file never reaches the notebook path, whatever it holds."""
        self.write([("code", "def f(:")])
        raw = payload("Write", self.path)
        with mock.patch.object(cc, "find_ruff", return_value=None):
            self.assertEqual(cc.run_hook(raw, root=self.root), "")

    @needs_ruff
    def test_ruff_findings_in_edited_cell_only(self):
        self.write(
            [
                ("code", "import os\n"),
                ("markdown", "# Text"),
                ("code", "y = undefined_name\n"),
            ]
        )
        raw = nb_payload(self.path, "y = undefined_name\n")
        ctx = context(cc.run_hook(raw, root=self.root, ruff=RUFF))
        self.assertIn("F821 cell 2 line 1", ctx)
        self.assertNotIn("F401", ctx)

    @needs_ruff
    def test_ruff_sees_names_from_earlier_cells(self):
        self.write([("code", "z = 1\n"), ("code", "print(z)\n")])
        raw = nb_payload(self.path, "print(z)\n")
        self.assertEqual(cc.run_hook(raw, root=self.root, ruff=RUFF), "")

    @needs_ruff
    def test_ruff_skips_rules_outside_f_and_e9(self):
        # E401 is outside the notebook selection.
        src = "import os, sys\nprint(os, sys)\n"
        self.write([("code", src)])
        self.assertEqual(
            cc.run_hook(nb_payload(self.path, src), root=self.root, ruff=RUFF), ""
        )

    @needs_ruff
    def test_magic_cell_clean_under_ruff(self):
        src = "%matplotlib inline\nimport os\nprint(os)\n"
        self.write([("code", src)])
        self.assertEqual(
            cc.run_hook(nb_payload(self.path, src), root=self.root, ruff=RUFF), ""
        )


@unittest.skipUnless(os.name == "nt", "drive letters are a Windows matter")
class DriveLetterCase(unittest.TestCase):
    def test_lower_case_drive_resolved(self):
        target = REPO / "lib" / "atc.py"
        lowered = Path(str(target)[0].lower() + str(target)[1:])
        rel = cc.rel_in_root(lowered, REPO)
        real, rel2 = cc.true_path(lowered, REPO, rel)
        self.assertEqual(str(real)[0], str(target.resolve())[0])
        self.assertEqual(rel2.as_posix(), "lib/atc.py")

    @needs_ruff
    def test_per_file_ignores_hold_for_lower_case_drive(self):
        """The hooks' own subprocess calls carry S603, which ruff.toml ignores under .claude/."""
        lowered = str(SCRIPT)[0].lower() + str(SCRIPT)[1:]
        out = cc.run_hook(payload("Write", lowered), root=REPO, ruff=RUFF)
        self.assertNotIn("S603", context(out) or "")


class Settings(unittest.TestCase):
    """.claude/settings.json wires code_check.py to notebook edits and notebook_check.py to shell calls."""

    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads(
            (REPO / ".claude" / "settings.json").read_text(encoding="utf-8")
        )

    def post_entries(self):
        return [
            (entry.get("matcher"), h)
            for entry in self.settings.get("hooks", {}).get("PostToolUse", [])
            for h in entry["hooks"]
        ]

    def test_code_check_covers_notebook_edit(self):
        hits = [m for m, h in self.post_entries() if "code_check.py" in h["command"]]
        self.assertEqual(hits, ["Write|Edit|MultiEdit|NotebookEdit"])

    def test_notebook_check_on_shell_calls(self):
        hits = [
            (m, h.get("timeout"))
            for m, h in self.post_entries()
            if "notebook_check.py" in h["command"]
        ]
        self.assertEqual(hits, [("Bash|PowerShell", 30)])
        self.assertTrue((HOOKS / "notebook_check.py").is_file())


if __name__ == "__main__":
    unittest.main()
