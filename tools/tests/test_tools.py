"""Tests for the stamp of tools/nb_execute.py and the classes and summary of tools/nb_staleness.py.

Run: python -m pytest tools/tests/test_tools.py -q   (under an interpreter with nbconvert and
ipykernel for the execution round-trip; that test is skipped otherwise). The pre-flight, the
failure capture and the run ledger are tested in test_nb_preflight.py.
"""

import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import _common as C  # noqa: E402
import nb_execute  # noqa: E402
import nb_staleness  # noqa: E402


def run(cwd, *args, env=None):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, env=env)


def tiny_notebook(path, source=("x = 1", "print(x + 1)")):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [s],
            }
            for s in source
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = "c%d" % i
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, sort_keys=True)
        f.write("\n")


def make_repo(tmp):
    """A temporary repository in the public layout: lib/, a root notebook, and an initial commit."""
    (Path(tmp) / "lib").mkdir()
    run(tmp, "init", "-q")
    run(tmp, "config", "user.email", "t@example.org")
    run(tmp, "config", "user.name", "t")
    run(tmp, "config", "commit.gpgsign", "false")
    (Path(tmp) / "lib" / "engine.py").write_text("A = 1\n", encoding="utf-8")
    tiny_notebook(str(Path(tmp) / "101_sandbox.ipynb"))
    run(tmp, "add", "lib/engine.py", "101_sandbox.ipynb")
    run(tmp, "commit", "-qm", "initial")


def commit_lib_change(tmp, text, when):
    (Path(tmp) / "lib" / "engine.py").write_text(text, encoding="utf-8")
    run(tmp, "add", "lib/engine.py")
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    run(tmp, "commit", "-qm", "lib change", env=env)


def write_lib(tmp, name, text):
    (Path(tmp) / "lib" / name).write_text(text, encoding="utf-8", newline="\n")


def head(tmp):
    return C.git(tmp, "rev-parse", "HEAD").strip()


def status_of(tmp):
    rows = nb_staleness.classify(tmp)
    assert len(rows) == 1
    return rows[0]


def quiet_preflight():
    """Patches that keep the pre-flight's prose lint and environment check from running."""
    return (
        mock.patch.object(nb_execute, "prose_findings", return_value=None),
        mock.patch.object(nb_execute, "check_environment", return_value=[]),
    )


class StampOutsideTheRepository(unittest.TestCase):
    def test_root_option_stamps_a_scratch_notebook(self):
        repo = tempfile.mkdtemp()
        make_repo(repo)
        (Path(repo) / "lib" / "MANIFEST.md").write_text("m\n", encoding="utf-8")
        nb = str(Path(tempfile.mkdtemp()) / "s.ipynb")
        tiny_notebook(nb)
        lint, envcheck = quiet_preflight()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "run_streaming", return_value=(0, "")),
            lint,
            envcheck,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(nb_execute.main([nb, "--root", repo]), 0)
        st = C.read_stamp(nb)
        self.assertEqual(st["lib_commit"], head(repo))
        self.assertTrue(st["lib_dirty"])  # the untracked manifest
        self.assertEqual(st["lib_manifest"]["path"], "lib/MANIFEST.md")
        # a notebook outside the repository is named by its file name, never its absolute path
        self.assertTrue(st["command"].endswith(" s.ipynb"), st["command"])
        self.assertNotIn(str(Path(nb).parent).replace("\\", "/"), st["command"])
        self.assertTrue(nb_staleness.cache_path(repo).is_file())

    def test_notebook_outside_every_repository_is_skipped(self):
        nb = str(Path(tempfile.mkdtemp()) / "s.ipynb")
        tiny_notebook(nb)
        _, envcheck = quiet_preflight()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.object(nb_execute, "run_streaming") as launch,
            envcheck,
            mock.patch("sys.stderr", io.StringIO()),
            mock.patch.object(nb_execute.C, "repo_root", side_effect=RuntimeError("x")),
        ):
            self.assertEqual(nb_execute.main([nb, "--no-prose-lint"]), 1)
        launch.assert_not_called()


class StampRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        make_repo(self.tmp)
        self.nb = str(Path(self.tmp) / "101_sandbox.ipynb")

    def test_write_stamp_keeps_cells_and_outputs(self):
        before = json.load(open(self.nb, encoding="utf-8"))
        stamp = nb_execute.build_stamp(self.tmp, self.nb, "cmd")
        nb_execute.write_stamp(self.nb, stamp)
        after = json.load(open(self.nb, encoding="utf-8"))
        self.assertEqual(after["cells"], before["cells"])
        self.assertEqual(
            after["metadata"]["lab_execution"]["lib_commit"], head(self.tmp)
        )
        self.assertFalse(after["metadata"]["lab_execution"]["lib_dirty"])
        self.assertEqual(C.read_stamp(self.nb)["lib_commit"], head(self.tmp))

    def test_crlf_is_preserved(self):
        raw = open(self.nb, "rb").read().replace(b"\n", b"\r\n")
        open(self.nb, "wb").write(raw)
        nb_execute.write_stamp(self.nb, {"lib_commit": "x"})
        out = open(self.nb, "rb").read()
        self.assertNotIn(b"\r\r", out)
        self.assertEqual(out.count(b"\n"), out.count(b"\r\n"))

    @unittest.skipUnless(
        importlib.util.find_spec("nbconvert") and importlib.util.find_spec("ipykernel"),
        "nbconvert and ipykernel are required",
    )
    def test_execute_and_stamp(self):
        lint, envcheck = quiet_preflight()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            lint,
            envcheck,
        ):
            rc = nb_execute.main([self.nb])
        self.assertEqual(rc, 0)
        nb = json.load(open(self.nb, encoding="utf-8"))
        self.assertEqual(nb["cells"][1]["outputs"][0]["text"], ["2\n"])
        st = nb["metadata"]["lab_execution"]
        for key in (
            "date",
            "python",
            "lib_commit",
            "lib_dirty",
            "lib_diff_sha256",
            "command",
            "packages",
            "kernel",
            "host",
        ):
            self.assertIn(key, st)
        self.assertEqual(st["kernel"], "python3")
        self.assertNotIn("lib_changed_during_run", st)
        self.assertEqual(st["lib_commit"], head(self.tmp))
        self.assertNotIn("lib_manifest", st)  # the repository holds no lib/MANIFEST.md
        self.assertEqual(nb_staleness.classify(self.tmp)[0]["status"], "current")

    def test_refuses_at_the_ceiling(self):
        err = io.StringIO()
        with (
            mock.patch.object(nb_execute, "python_count", return_value=7),
            mock.patch("sys.stderr", err),
            mock.patch("subprocess.run") as sub,
        ):
            rc = nb_execute.main([self.nb])
        self.assertEqual(rc, 2)
        sub.assert_not_called()
        self.assertIn("7", err.getvalue())

    def test_refusal_at_the_ceiling_is_logged_without_git(self):
        with (
            mock.patch.object(nb_execute, "python_count", return_value=9),
            mock.patch("sys.stderr", io.StringIO()),
            mock.patch("subprocess.run") as sub,
            mock.patch("subprocess.Popen") as popen,
        ):
            self.assertEqual(nb_execute.main([self.nb, "--ceiling", "9"]), 2)
        sub.assert_not_called()
        popen.assert_not_called()
        rec = json.loads(
            nb_execute.ledger_path(self.tmp)
            .read_text(encoding="utf-8")
            .splitlines()[-1]
        )
        self.assertEqual(rec["outcome"], "refused: ceiling")
        self.assertEqual(rec["notebook"], "101_sandbox.ipynb")

    def test_the_ceiling_comes_from_the_hooks_limits(self):
        limits = nb_execute._limits()
        if limits is None:
            self.skipTest(".claude/hooks/_limits.py is absent")
        self.assertEqual(nb_execute.ceiling(), limits.CEILING)
        with mock.patch.object(nb_execute, "_limits", return_value=None):
            self.assertEqual(nb_execute.ceiling(), nb_execute.DEFAULT_CEILING)
            self.assertIsNone(nb_execute.python_count())

    def test_refuses_agg_backend(self):
        with (
            mock.patch.object(nb_execute, "python_count", return_value=0),
            mock.patch.dict(os.environ, {"MPLBACKEND": "agg"}),
            mock.patch("subprocess.run") as sub,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(nb_execute.main([self.nb]), 2)
        sub.assert_not_called()


class Staleness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        make_repo(self.tmp)
        self.nb = str(Path(self.tmp) / "101_sandbox.ipynb")

    def stamp(self):
        nb_execute.write_stamp(
            self.nb, nb_execute.build_stamp(self.tmp, self.nb, "cmd")
        )

    def test_unstamped_estimate(self):
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "unstamped")
        self.assertTrue(r["estimate"])
        self.assertFalse(r["estimated_stale"])
        commit_lib_change(self.tmp, "A = 2\n", "2030-01-01T00:00:00")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "unstamped")
        self.assertTrue(r["estimated_stale"])

    def test_current_then_stale_after_lib_commit(self):
        self.stamp()
        self.assertEqual(status_of(self.tmp)["status"], "current")
        commit_lib_change(self.tmp, "A = 2\n", "2030-01-01T00:00:00")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/engine.py"])

    def test_dirty_lib_at_execution_is_stale_until_unchanged_head(self):
        (Path(self.tmp) / "lib" / "engine.py").write_text("A = 3\n", encoding="utf-8")
        self.stamp()
        self.assertTrue(C.read_stamp(self.nb)["lib_dirty"])
        self.assertEqual(
            status_of(self.tmp)["status"], "current"
        )  # same dirty state, same HEAD
        (Path(self.tmp) / "lib" / "engine.py").write_text("A = 4\n", encoding="utf-8")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")

    def test_uncommitted_change_after_clean_stamp_is_stale(self):
        self.stamp()
        (Path(self.tmp) / "lib" / "engine.py").write_text("A = 5\n", encoding="utf-8")
        self.assertEqual(status_of(self.tmp)["status"], "stale")

    def test_non_lib_commit_does_not_stale(self):
        self.stamp()
        with open(os.path.join(self.tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write("x\n")
        run(self.tmp, "add", "README.md")
        run(self.tmp, "commit", "-qm", "doc")
        self.assertEqual(status_of(self.tmp)["status"], "current")

    def test_reformat_commit_is_not_a_change(self):
        self.stamp()
        commit_lib_change(self.tmp, "A  =  1   # a note\n", "2030-01-01T00:00:00")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "current")
        self.assertIn("1 file differs in format, comments or hints only", r["detail"])
        self.assertNotIn("changed_files", r)

    def test_comment_edit_in_the_working_tree_is_not_a_change(self):
        self.stamp()
        write_lib(self.tmp, "engine.py", "# SPDX-License-Identifier: MIT\nA = 1\n")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "current")
        self.assertIn("differs in format, comments or hints only", r["detail"])

    def test_unchanged_lib_has_no_note(self):
        self.stamp()
        self.assertNotIn("format or comments", status_of(self.tmp)["detail"])

    def test_code_change_beside_a_reformat_names_only_the_code_change(self):
        write_lib(self.tmp, "second.py", "B = 1\n")
        write_lib(self.tmp, "third.py", "C = 1\n")
        run(self.tmp, "add", "lib")
        run(self.tmp, "commit", "-qm", "two more modules")
        self.stamp()
        write_lib(self.tmp, "engine.py", "A  =  1\n")
        write_lib(self.tmp, "second.py", "B  =  1\n")
        write_lib(self.tmp, "third.py", "C = 2\n")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/third.py"])
        self.assertIn("2 files differ in format, comments or hints only", r["detail"])

    def test_docstring_word_change_is_a_change(self):
        write_lib(self.tmp, "engine.py", '"""One."""\nA = 1\n')
        run(self.tmp, "add", "lib")
        run(self.tmp, "commit", "-qm", "docstring")
        self.stamp()
        write_lib(self.tmp, "engine.py", '"""Two."""\nA = 1\n')
        self.assertEqual(status_of(self.tmp)["status"], "stale")

    def test_file_that_no_fingerprint_clears_stays_a_change(self):
        write_lib(self.tmp, "table.json", "{}\n")
        run(self.tmp, "add", "lib")
        run(self.tmp, "commit", "-qm", "data")
        self.stamp()
        write_lib(self.tmp, "table.json", "{ }\n")  # not Python
        self.assertEqual(status_of(self.tmp)["status"], "stale")
        write_lib(self.tmp, "table.json", "{}\n")
        write_lib(self.tmp, "engine.py", "def f(:\n")  # no fingerprint
        self.assertEqual(status_of(self.tmp)["status"], "stale")
        write_lib(self.tmp, "engine.py", "A = 1\n")
        self.assertEqual(status_of(self.tmp)["status"], "current")
        (Path(self.tmp) / "lib" / "engine.py").unlink()  # deleted
        self.assertEqual(status_of(self.tmp)["status"], "stale")

    def test_new_module_at_head_is_a_change(self):
        self.stamp()
        write_lib(self.tmp, "added.py", "B = 1\n")
        run(self.tmp, "add", "lib")
        run(self.tmp, "commit", "-qm", "new module")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/added.py"])

    def test_dirty_stamp_is_not_cleared_by_a_reformat(self):
        write_lib(self.tmp, "engine.py", "A = 3\n")
        self.stamp()
        write_lib(
            self.tmp, "engine.py", "A  =  3\n"
        )  # same code as the run, not as lib_commit
        self.assertEqual(status_of(self.tmp)["status"], "stale")
        run(self.tmp, "checkout", "--", "lib/engine.py")
        write_lib(
            self.tmp, "engine.py", "A  =  1\n"
        )  # same code as lib_commit, not as the run
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertIn("1 file differs in format, comments or hints only", r["detail"])

    def test_hint_commit_is_not_a_change(self):
        write_lib(self.tmp, "engine.py", "def f(x):\n    return x\n")
        run(self.tmp, "add", "lib/engine.py")
        run(self.tmp, "commit", "-qm", "function")
        self.stamp()
        commit_lib_change(
            self.tmp,
            "from __future__ import annotations\n\n\n"
            "def f(x: int) -> int:\n    return x\n",
            "2030-01-01T00:00:00",
        )
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "current")
        self.assertIn("1 file differs in format, comments or hints only", r["detail"])

    def test_hint_edit_in_the_working_tree_is_not_a_change(self):
        write_lib(self.tmp, "engine.py", "def f(x):\n    return x\n")
        run(self.tmp, "add", "lib/engine.py")
        run(self.tmp, "commit", "-qm", "function")
        self.stamp()
        write_lib(self.tmp, "engine.py", "def f(x: int) -> int:\n    return x\n")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "current")
        self.assertIn("hints only", r["detail"])

    def test_hint_beside_a_code_change_is_a_change(self):
        write_lib(self.tmp, "engine.py", "def f(x=1):\n    return x\n")
        run(self.tmp, "add", "lib/engine.py")
        run(self.tmp, "commit", "-qm", "function")
        self.stamp()
        write_lib(self.tmp, "engine.py", "def f(x: int = 2) -> int:\n    return x\n")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/engine.py"])

    def test_cast_is_a_change(self):
        write_lib(self.tmp, "engine.py", "def f(x):\n    return x\n")
        run(self.tmp, "add", "lib/engine.py")
        run(self.tmp, "commit", "-qm", "function")
        self.stamp()
        write_lib(
            self.tmp,
            "engine.py",
            "from typing import cast\n\n\ndef f(x):\n    return cast(int, x)\n",
        )
        self.assertEqual(status_of(self.tmp)["status"], "stale")

    def test_hint_commit_does_not_move_the_unstamped_estimate(self):
        commit_lib_change(self.tmp, "A: int = 1\n", "2030-01-01T00:00:00")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "unstamped")
        self.assertFalse(r["estimated_stale"])
        self.assertIn(
            "1 later lib commit changes format, comments or hints", r["detail"]
        )

    def test_licence_commit_does_not_move_the_unstamped_estimate(self):
        commit_lib_change(
            self.tmp, "# SPDX-License-Identifier: MIT\nA = 1\n", "2030-01-01T00:00:00"
        )
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "unstamped")
        self.assertFalse(r["estimated_stale"])
        self.assertIn(
            "1 later lib commit changes format, comments or hints only", r["detail"]
        )
        commit_lib_change(
            self.tmp, "# SPDX-License-Identifier: MIT\nA = 2\n", "2031-01-01T00:00:00"
        )
        r = status_of(self.tmp)
        self.assertTrue(r["estimated_stale"])

    def test_walk_past_a_page_of_comment_commits(self):
        commit_lib_change(self.tmp, "A = 2\n", "2030-01-01T00:00:00")
        for i in range(3):
            commit_lib_change(
                self.tmp, "# %d\nA = 2\n" % i, "2030-01-0%dT00:00:00" % (i + 2)
            )
        dirs = ["lib"]
        with mock.patch.object(nb_staleness, "LOG_PAGE", 2):
            when, passed = nb_staleness.last_code_commit(self.tmp, dirs)
        self.assertEqual(len(passed), 3)
        self.assertLess(when, min(passed))

    def test_root_commit_counts_as_a_code_change(self):
        when, passed = nb_staleness.last_code_commit(self.tmp, ["lib"])
        self.assertGreater(when, 0)
        self.assertEqual(passed, [])

    def test_lib_commit_that_does_not_resolve_is_stale(self):
        self.stamp()
        write_lib(self.tmp, "engine.py", "A = 3\n")
        for commit in ("deadbeef" * 5, ""):
            nb_execute.write_stamp(self.nb, {"lib_commit": commit})
            r = status_of(self.tmp)
            self.assertEqual(r["status"], "stale")
            self.assertIn("does not resolve", r["detail"])

    def test_new_untracked_module_is_a_change(self):
        self.stamp()
        write_lib(self.tmp, "c.py", "C = 1\n")
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/c.py"])

    def test_ignored_file_under_lib_is_not_a_change(self):
        with open(os.path.join(self.tmp, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("*.pyc\n")
        run(self.tmp, "add", ".gitignore")
        run(self.tmp, "commit", "-qm", "ignore")
        self.stamp()
        write_lib(self.tmp, "c.pyc", "x")
        self.assertEqual(status_of(self.tmp)["status"], "current")

    def test_cli_json(self):
        out = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "tools", "nb_staleness.py"),
                "--root",
                self.tmp,
                "--json",
                "--no-cache",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)["root"][0]["status"], "unstamped")

    def test_notebook_below_the_root_is_not_classified(self):
        below = Path(self.tmp) / "sandbox_exhibits"
        below.mkdir()
        tiny_notebook(below / "copy.ipynb")
        notebooks = [r["notebook"] for r in nb_staleness.classify(self.tmp)]
        self.assertEqual(notebooks, ["101_sandbox.ipynb"])

    def test_manifest_change_is_a_change(self):
        manifest = Path(self.tmp) / "lib" / "MANIFEST.md"
        manifest.write_text(
            "lib/ is the shared 9-bus library, copied in on 2026-11-07.\n",
            encoding="utf-8",
        )
        run(self.tmp, "add", "lib/MANIFEST.md")
        run(self.tmp, "commit", "-qm", "manifest")
        self.stamp()
        self.assertEqual(
            C.read_stamp(self.nb)["lib_manifest"]["path"], "lib/MANIFEST.md"
        )
        self.assertEqual(status_of(self.tmp)["status"], "current")
        manifest.write_text(
            "lib/ is the shared 9-bus library, copied in on 2026-11-08.\n",
            encoding="utf-8",
        )
        r = status_of(self.tmp)
        self.assertEqual(r["status"], "stale")
        self.assertEqual(r["changed_files"], ["lib/MANIFEST.md"])

    def test_summary_cache_counts(self):
        row = {"status": "stale", "notebook": "101_sandbox.ipynb", "detail": "d"}
        with (
            mock.patch.object(nb_staleness, "classify", return_value=[row]),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(nb_staleness.main(["--root", self.tmp]), 0)
        cache = nb_staleness.cache_path(self.tmp)
        self.assertEqual(cache, Path(self.tmp) / "tools" / ".cache" / "staleness.json")
        raw = cache.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        summary = json.loads(raw.decode("utf-8"))
        # the counts stay flat, the form the session brief reads
        self.assertEqual(summary["counts"], {"current": 0, "stale": 1, "unstamped": 0})
        self.assertEqual(summary["head"], head(self.tmp))
        self.assertIn("generated", summary)
        leftovers = [p.name for p in cache.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])  # no temporary file left behind

    def test_every_full_run_writes_the_summary_and_no_cache_writes_none(self):
        cache = nb_staleness.cache_path(self.tmp)
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            nb_staleness.main(["--root", self.tmp, "--no-cache"])
            self.assertFalse(cache.exists())
            nb_staleness.main(["--root", self.tmp])
        self.assertEqual(
            json.loads(cache.read_text(encoding="utf-8"))["counts"]["unstamped"], 1
        )

    def test_refresh_cache_classifies_and_rewrites(self):
        self.stamp()
        path = nb_staleness.refresh_cache(self.tmp)
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"], {"current": 1, "stale": 0, "unstamped": 0})
        write_lib(self.tmp, "engine.py", "A = 9\n")
        nb_staleness.refresh_cache(self.tmp)
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"]["stale"], 1)


class StampFields(unittest.TestCase):
    """The stamp of a published notebook holds no absolute path and no host name."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        make_repo(self.tmp)
        (Path(self.tmp) / "lib" / "MANIFEST.md").write_text("m\n", encoding="utf-8")
        self.nb = str(Path(self.tmp) / "101_sandbox.ipynb")
        self.stamp = nb_execute.build_stamp(
            self.tmp, self.nb, nb_execute.stamp_path(self.nb, self.tmp)
        )

    def strings(self, value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield str(k)
                yield from self.strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from self.strings(v)
        elif value is not None:
            yield str(value)

    def test_fields(self):
        self.assertEqual(
            set(self.stamp),
            {
                "date",
                "python",
                "lib_commit",
                "lib_dirty",
                "lib_diff_sha256",
                "lib_manifest",
                "command",
                "packages",
                "kernel",
                "host",
            },
        )
        self.assertEqual(set(self.stamp["host"]), {"system", "release", "machine"})
        for name in ("numpy", "pandas", "nbconvert", "nbclient", "ipykernel"):
            self.assertIn(name, self.stamp["packages"])

    def test_no_field_holds_a_path_separator_or_the_node_name(self):
        node = platform.node()
        # lib_manifest.path is the one field that names a file, and it is repository-relative
        for key, value in self.stamp.items():
            if key == "lib_manifest":
                self.assertEqual(value["path"], "lib/MANIFEST.md")
                continue
            for text in self.strings(value):
                self.assertNotIn("/", text, key)
                self.assertNotIn("\\", text, key)
                if node:
                    self.assertNotIn(node, text, key)
        self.assertEqual(self.stamp["command"], "101_sandbox.ipynb")
        self.assertEqual(self.stamp["python"].split()[0], Path(sys.executable).name)

    def test_model_packages_come_from_the_lock(self):
        lock = Path(self.tmp) / "env_lock.json"
        lock.write_text(
            json.dumps(
                {
                    "packages": {
                        "numpy": {"role": "model", "version": "1"},
                        "ruff": {"role": "tooling", "version": "1"},
                    }
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(nb_execute.model_packages(lock), ("numpy",))
        lock.write_text("not json", encoding="utf-8")
        self.assertEqual(nb_execute.model_packages(lock), nb_execute.MODEL_PACKAGES)
        self.assertEqual(
            nb_execute.model_packages(Path(self.tmp) / "absent.json"),
            nb_execute.MODEL_PACKAGES,
        )

    def test_a_missing_package_is_null(self):
        self.assertEqual(
            nb_execute.package_versions(("no-such-package-xyz",)),
            {"no-such-package-xyz": None},
        )


if __name__ == "__main__":
    unittest.main()
