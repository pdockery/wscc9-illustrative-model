"""Tests for the pre-flight, the failure capture and the run ledger of tools/nb_execute.py.

The pre-flight compiles every code cell before any notebook starts, reports the prose lint and
runs the environment check. A failed run is read for its failing cell and exception, and every
attempt appends one line to the run ledger. Every notebook is a small synthetic one written to
a temporary repository in the public layout, lib/ and root notebooks; no repository notebook is
executed.

Run: python -m pytest tools/tests/test_nb_preflight.py -q
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import _common as C  # noqa: E402
import nb_execute  # noqa: E402
import nb_staleness  # noqa: E402

NOTEBOOK = "101_sandbox.ipynb"

FAILED_STDERR = (
    "[NbConvertApp] Converting notebook 101_sandbox.ipynb to notebook\n"
    "Traceback (most recent call last):\n"
    '  File "<frozen runpy>", line 88, in _run_module_as_main\n'
    "nbclient.exceptions.CellExecutionError: An error occurred while executing the "
    "following cell:\n"
    "------------------\n"
    "print('hello')\n"
    "y = x / 0\n"
    "------------------\n"
    "\n"
    "----- stdout -----\n"
    "hello\n"
    "------------------\n"
    "\n"
    "\x1b[31m-----------------------------------------------------------------\x1b[39m\n"
    "\x1b[31mZeroDivisionError\x1b[39m                         Traceback (most recent "
    "call last)\n"
    "\x1b[36mCell\x1b[39m\x1b[36m \x1b[39m\x1b[32mIn[2]\x1b[39m\x1b[32m, line 2\x1b[39m\n"
    "\n"
    "\x1b[31mZeroDivisionError\x1b[39m: division by zero\n"
)

TIMEOUT_STDERR = (
    "nbclient.exceptions.CellTimeoutError: A cell timed out while it was being executed, "
    "after 5 seconds.\n"
    "The message was: Cell execution timed out.\n"
    "Here is a preview of the cell contents:\n"
    "-------------------\n"
    "['import time', 'time.sleep(60)']\n"
    "-------------------\n"
)


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def write_nb(path, cells, metadata=None):
    """Write a notebook of (cell_type, source, outputs) triples with ids c0, c1, ..."""
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"c{i}",
                "metadata": {},
                "source": source,
                **(
                    {"execution_count": None, "outputs": outputs or []}
                    if kind == "code"
                    else {}
                ),
            }
            for i, (kind, source, outputs) in enumerate(cells)
        ],
        "metadata": metadata
        or {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_bytes((json.dumps(nb, indent=1, sort_keys=True) + "\n").encode())


class TempRepo(unittest.TestCase):
    """A temporary repository with lib/, one root notebook and one commit."""

    def setUp(self):
        self.root = os.path.normpath(tempfile.mkdtemp())
        r = Path(self.root)
        (r / "lib").mkdir()
        (r / "lib" / "engine.py").write_bytes(b"A = 1\n")
        self.nb = str(r / NOTEBOOK)
        write_nb(self.nb, [("code", "x = 1", None), ("code", "print(x + 1)", None)])
        git(self.root, "init", "-q")
        git(self.root, "config", "user.email", "t@example.org")
        git(self.root, "config", "user.name", "t")
        git(self.root, "config", "commit.gpgsign", "false")
        git(self.root, "add", "lib/engine.py", NOTEBOOK)
        git(self.root, "commit", "-qm", "initial")

    def ledger(self):
        lines = nb_execute.ledger_path(self.root).read_text(encoding="utf-8")
        return [json.loads(x) for x in lines.splitlines()]


class Syntax(TempRepo):
    def test_a_broken_cell_is_named_with_its_line(self):
        write_nb(
            self.nb, [("code", "x = 1", None), ("code", "y = 2\nif y:\nprint(y)", None)]
        )
        self.assertEqual(
            [
                (p["id"], p["index"], p["line"])
                for p in nb_execute.syntax_problems(self.nb)
            ],
            [("c1", 1, 3)],
        )

    def test_magics_escapes_and_bootstrap_pass(self):
        bootstrap = (
            "import sys, subprocess\n"
            "if 'google.colab' in sys.modules:\n"
            "    !pip install -q pypsa\n"
            "    %cd /content\n"
            "else:\n"
            "    files = !ls\n"
        )
        write_nb(
            self.nb,
            [
                ("code", "%matplotlib inline\nimport math", None),
                ("code", "%%bash\nls -la | grep (", None),
                ("code", bootstrap, None),
                ("code", "y = (10\n     % 3)\n", None),
                ("code", "await something()", None),
                ("markdown", "if (:", None),
            ],
        )
        self.assertEqual(nb_execute.syntax_problems(self.nb), [])

    def test_a_magic_does_not_hide_a_real_error(self):
        write_nb(self.nb, [("code", "%matplotlib inline\ndef f(:\n    pass", None)])
        problems = nb_execute.syntax_problems(self.nb)
        self.assertEqual([(p["id"], p["line"]) for p in problems], [("c0", 2)])

    def test_source_stored_as_a_list_of_lines(self):
        write_nb(self.nb, [("code", ["x = (\n", "1\n"], None)])
        self.assertEqual([p["id"] for p in nb_execute.syntax_problems(self.nb)], ["c0"])

    def test_an_unreadable_notebook_is_a_problem(self):
        Path(self.nb).write_bytes(b"not json")
        problems = nb_execute.readable_problems(self.nb)
        self.assertEqual(len(problems), 1)
        self.assertIsNone(problems[0]["id"])


class Preflight(TempRepo):
    def test_main_refuses_before_any_launch_and_logs_it(self):
        write_nb(self.nb, [("code", "def f(:\n    pass", None)])
        err = io.StringIO()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "run_streaming") as run,
            mock.patch.object(nb_execute, "prose_findings") as lint,
            mock.patch.object(nb_execute, "check_environment") as envcheck,
            mock.patch("sys.stderr", err),
        ):
            rc = nb_execute.main([self.nb])
        self.assertEqual(rc, 2)
        run.assert_not_called()
        lint.assert_not_called()
        envcheck.assert_not_called()
        self.assertIn(f"{NOTEBOOK} cell c0 (index 0) line 1", err.getvalue())
        self.assertIn("no notebook was started", err.getvalue())
        rec = self.ledger()[-1]
        self.assertEqual(rec["outcome"], "refused: syntax")
        self.assertEqual(rec["notebook"], NOTEBOOK)
        self.assertEqual(rec["exit"], 2)

    def test_one_broken_notebook_refuses_the_whole_launch(self):
        good = str(Path(self.root) / "workshop2_congestion_management.ipynb")
        write_nb(good, [("code", "x = 1", None)])
        write_nb(self.nb, [("code", "if:", None)])
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "run_streaming") as run,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(nb_execute.main([good, self.nb]), 2)
        run.assert_not_called()
        self.assertEqual([r["notebook"] for r in self.ledger()], [NOTEBOOK])

    def test_prose_findings_are_reported_and_never_refuse(self):
        report = {
            "errors": 1,
            "warnings": 2,
            "findings": [
                {
                    "severity": "error",
                    "rule": "MATH-DELIM",
                    "cell": 3,
                    "message": "use $",
                },
            ],
        }
        err = io.StringIO()
        with (
            mock.patch.object(nb_execute, "prose_findings", return_value=report),
            mock.patch.object(nb_execute, "check_environment", return_value=[]),
            mock.patch("sys.stderr", err),
        ):
            self.assertEqual(nb_execute.preflight([self.nb], lint=True), 0)
        self.assertIn("1 error(s) and 2 warning(s)", err.getvalue())
        self.assertIn("MATH-DELIM", err.getvalue())

    def test_an_absent_lint_is_named(self):
        err = io.StringIO()
        with (
            mock.patch.object(nb_execute, "PROSE_LINT", Path(self.root) / "absent.py"),
            mock.patch.object(nb_execute, "check_environment", return_value=[]),
            mock.patch("sys.stderr", err),
        ):
            self.assertEqual(nb_execute.preflight([self.nb], lint=True), 0)
        self.assertIn(f"prose_lint did not run on {NOTEBOOK}", err.getvalue())

    def test_no_prose_lint_skips_the_lint(self):
        with (
            mock.patch.object(nb_execute, "prose_findings") as lint,
            mock.patch.object(nb_execute, "check_environment", return_value=[]) as env,
        ):
            self.assertEqual(nb_execute.preflight([self.nb], lint=False), 0)
        lint.assert_not_called()
        env.assert_called_once()

    def test_prose_lint_runs_on_a_notebook_path(self):
        report = nb_execute.prose_findings(self.nb)
        if report is not None:
            self.assertIn("errors", report)
            self.assertIn("warnings", report)


class Environment(unittest.TestCase):
    def test_a_missing_lab_env_is_reported_on_one_line(self):
        err = io.StringIO()
        with (
            mock.patch.dict(sys.modules, {"lab_env": None}),
            mock.patch("sys.stderr", err),
        ):
            self.assertIsNone(nb_execute.check_environment())
        self.assertEqual(len(err.getvalue().splitlines()), 1)
        self.assertIn("environment check did not run", err.getvalue())

    def test_lab_env_differences_are_returned(self):
        fake = mock.Mock()
        fake.check_environment.return_value = ["numpy 0 (pinned 1)"]
        with mock.patch.dict(sys.modules, {"lab_env": fake}):
            self.assertEqual(nb_execute.check_environment(), ["numpy 0 (pinned 1)"])

    def test_a_raising_lab_env_does_not_stop_the_preflight(self):
        fake = mock.Mock()
        fake.check_environment.side_effect = RuntimeError("x")
        with (
            mock.patch.dict(sys.modules, {"lab_env": fake}),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertIsNone(nb_execute.check_environment())


class Failure(TempRepo):
    def test_failing_cell_from_a_cell_execution_error(self):
        write_nb(
            self.nb,
            [("code", "x = 1", None), ("code", "print('hello')\ny = x / 0", None)],
        )
        cell = nb_execute.failing_cell(FAILED_STDERR, self.nb)
        self.assertEqual(cell["id"], "c1")
        self.assertEqual(cell["index"], 1)
        self.assertEqual(cell["exception"], "ZeroDivisionError: division by zero")
        self.assertEqual(cell["source_head"], "print('hello')")

    def test_failing_cell_from_a_timeout(self):
        write_nb(self.nb, [("code", "import time\ntime.sleep(60)", None)])
        cell = nb_execute.failing_cell(TIMEOUT_STDERR, self.nb)
        self.assertEqual(cell["exception"], "CellTimeoutError")

    def test_failing_cell_from_a_dead_kernel(self):
        cell = nb_execute.failing_cell(
            "nbclient.exceptions.DeadKernelError: Kernel died\n", self.nb
        )
        self.assertIn("DeadKernelError", cell["exception"])
        self.assertIsNone(nb_execute.failing_cell("nothing useful\n", self.nb))

    def run_main(self, rc, tail, side_effect=None):
        err = io.StringIO()

        def launch(cmd, env):
            if side_effect:
                side_effect()
            return rc, tail

        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "run_streaming", side_effect=launch),
            mock.patch.object(nb_execute, "prose_findings", return_value=None),
            mock.patch.object(nb_execute, "check_environment", return_value=[]),
            mock.patch("sys.stderr", err),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            out = nb_execute.main([self.nb])
        return out, err.getvalue(), self.ledger()[-1]

    def test_a_failed_run_is_logged_with_its_cell_and_not_stamped(self):
        write_nb(
            self.nb,
            [("code", "x = 1", None), ("code", "print('hello')\ny = x / 0", None)],
        )
        rc, err, rec = self.run_main(1, FAILED_STDERR)
        self.assertEqual(rc, 1)
        self.assertIn("at cell c1 (index 1): ZeroDivisionError", err)
        self.assertEqual(rec["outcome"], "failed")
        self.assertEqual(rec["failing_cell"]["id"], "c1")
        for key in (
            "start",
            "end",
            "seconds",
            "exit",
            "lib_commit",
            "lib_dirty",
            "lib_diff_sha256",
            "host",
            "command",
            "stderr_tail",
        ):
            self.assertIn(key, rec)
        self.assertEqual(rec["notebook"], NOTEBOOK)
        self.assertIsNone(C.read_stamp(self.nb))

    def test_a_stamped_run_is_logged_and_refreshes_the_summary(self):
        rc, _, rec = self.run_main(0, "")
        self.assertEqual(rc, 0)
        self.assertEqual(rec["outcome"], "stamped")
        stamp = C.read_stamp(self.nb)
        self.assertEqual(stamp["lib_commit"], rec["lib_commit"])
        self.assertNotIn("lib_changed_during_run", stamp)
        self.assertNotIn("lib_changed_during_run", rec)
        summary = json.loads(
            nb_staleness.cache_path(self.root).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["counts"], {"current": 1, "stale": 0, "unstamped": 0})

    def test_a_lib_change_during_the_run_is_marked(self):
        def edit():
            (Path(self.root) / "lib" / "engine.py").write_bytes(b"A = 2\n")

        rc, _, rec = self.run_main(0, "", side_effect=edit)
        self.assertEqual(rc, 0)
        self.assertTrue(C.read_stamp(self.nb)["lib_changed_during_run"])
        self.assertTrue(rec["lib_changed_during_run"])
        self.assertFalse(rec["lib_dirty"])  # the launch state, before the edit

    def test_a_failed_summary_refresh_is_one_stderr_line(self):
        err = io.StringIO()
        with (
            mock.patch.object(
                nb_staleness, "refresh_cache", side_effect=OSError("locked")
            ),
            mock.patch("sys.stderr", err),
        ):
            nb_execute.refresh_staleness(self.root)
        self.assertEqual(len(err.getvalue().splitlines()), 1)
        self.assertIn("staleness summary not refreshed", err.getvalue())

    @unittest.skipUnless(
        importlib.util.find_spec("nbconvert") and importlib.util.find_spec("ipykernel"),
        "nbconvert and ipykernel are required",
    )
    def test_a_real_failure_names_the_cell(self):
        write_nb(self.nb, [("code", "x = 1", None), ("code", "y = x / 0", None)])
        err = io.StringIO()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "prose_findings", return_value=None),
            mock.patch.object(nb_execute, "check_environment", return_value=[]),
            mock.patch("sys.stderr", err),
        ):
            rc = nb_execute.main([self.nb])
        self.assertEqual(rc, 1)
        rec = self.ledger()[-1]
        self.assertEqual(rec["failing_cell"]["id"], "c1")
        self.assertTrue(
            rec["failing_cell"]["exception"].startswith("ZeroDivisionError")
        )


if __name__ == "__main__":
    unittest.main()
