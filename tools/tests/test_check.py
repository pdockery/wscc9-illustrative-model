# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""Tests for tools/check.py and tools/known_failures.toml.

No model suite runs here. The tiers are fake: a Python one-liner that prints a pytest summary,
and one real pytest run over a two-test fixture. The ledger is written to a temporary file.

Run: python -m pytest tools/tests/test_check.py -q   (under an interpreter with pytest)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import check  # noqa: E402

KNOWN = {
    "tests/test_a.py::test_known": {
        "id": "tests/test_a.py::test_known",
        "tier": "tests",
        "cause": "c",
        "since": "2026-11-10",
        "record": "r",
        "remove_when": "w",
    }
}
FIELDS = "\n".join(f'{k} = "v"' for k in ("cause", "since", "record", "remove_when"))


def fake_run(
    label: str, lines: list[str], code: int, *, timeout: float = 60
) -> check.Run:
    """A run whose command prints the given lines and exits with the given status."""
    prog = f"import sys; print({chr(10).join(lines)!r}); sys.exit({code})"
    return check.Run(label, [sys.executable, "-c", prog], ROOT, timeout)


def no_processes() -> int:
    return 0


class TestParse(unittest.TestCase):
    def test_node_id_is_brought_to_the_root(self) -> None:
        self.assertEqual(
            check.node_id("tests\\test_a.py::T::test_b[x::y]", "sub"),
            "sub/tests/test_a.py::T::test_b[x::y]",
        )
        self.assertEqual(
            check.node_id("tools/tests/test_a.py"), "tools/tests/test_a.py"
        )

    def test_summary_lines_and_counts(self) -> None:
        out = "\n".join(
            [
                "PASSED tests/test_a.py::test_ok",
                "XPASS tests/test_a.py::test_lucky",
                "FAILED tests/test_a.py::test_bad - assert False",
                "ERROR tests/test_b.py - ImportError",
                "SKIPPED [1] tests/test_c.py:3: absent",
                "2 failed, 408 passed, 3 skipped, 1 error, 1 warning in 12.34s",
            ]
        )
        passed, failed, counts = check.parse_pytest(out)
        self.assertEqual(
            passed, {"tests/test_a.py::test_ok", "tests/test_a.py::test_lucky"}
        )
        self.assertEqual(failed, {"tests/test_a.py::test_bad", "tests/test_b.py"})
        self.assertEqual(
            counts,
            Counter(failed=2, passed=408, skipped=3, errors=1, warnings=1),
        )


class TestCompare(unittest.TestCase):
    def _res(self, passed: set, failed: set) -> check.TierResult:
        res = check.TierResult("tests", passed=passed, failed=failed)
        return check.compare(res, KNOWN)

    def test_known_failure_passes_the_tier(self) -> None:
        res = self._res(set(), {"tests/test_a.py::test_known"})
        self.assertTrue(res.ok)
        self.assertEqual(res.known_failures, ["tests/test_a.py::test_known"])

    def test_new_failure_fails_the_tier(self) -> None:
        res = self._res(set(), {"tests/test_a.py::test_new"})
        self.assertFalse(res.ok)
        self.assertEqual(res.new_failures, ["tests/test_a.py::test_new"])

    def test_known_failure_that_passes_fails_the_tier(self) -> None:
        res = self._res({"tests/test_a.py::test_known"}, set())
        self.assertFalse(res.ok)
        self.assertEqual(res.known_now_passing, ["tests/test_a.py::test_known"])

    def test_known_failure_that_did_not_run_is_noted(self) -> None:
        res = self._res(set(), set())
        self.assertTrue(res.ok)
        self.assertEqual(res.known_not_run, ["tests/test_a.py::test_known"])

    def test_another_tiers_entry_is_not_compared(self) -> None:
        res = check.compare(check.TierResult("slides", failed=set()), KNOWN)
        self.assertEqual(res.known_not_run, [])


class TestKnownFile(unittest.TestCase):
    def _load(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.toml"
            p.write_text(text, encoding="utf-8")
            return check.load_known(p)

    def test_entries_are_checked(self) -> None:
        good = f'[[failure]]\nid = "a.py::t"\ntier = "slides"\n{FIELDS}\n'
        self.assertEqual(list(self._load(good)), ["a.py::t"])
        with self.assertRaisesRegex(check.ConfigError, "missing cause"):
            self._load('[[failure]]\nid = "a.py::t"\ntier = "slides"\n')
        with self.assertRaisesRegex(check.ConfigError, "unknown tier"):
            self._load(good.replace('"slides"', '"slow"'))
        with self.assertRaisesRegex(check.ConfigError, "listed twice"):
            self._load(good + good)

    def test_text_reader_agrees_with_tomllib(self) -> None:
        text = f'[[failure]]\nid = "a.py::t"\ntier = "tests"\n{FIELDS}\n'
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib needs Python 3.11")
        self.assertEqual(check._parse_known_text(text), tomllib.loads(text)["failure"])

    def test_tracked_file_has_no_entry_and_names_only_dev_notes(self) -> None:
        path = ROOT / "tools" / "known_failures.toml"
        self.assertEqual(check.load_known(path), {})
        header = path.read_text(encoding="utf-8")
        self.assertIn("dev_notes.md", header)
        self.assertNotIn("\r", header)

    def test_tracked_entries_name_tests_that_exist(self) -> None:
        known = check.load_known(ROOT / "tools" / "known_failures.toml")
        for nodeid in known:
            path, _, name = nodeid.partition("::")
            src = (ROOT / path).read_text(encoding="utf-8")
            self.assertIn(f"def {name}(", src, nodeid)


class TestLaunch(unittest.TestCase):
    def test_room_counts_this_script_and_the_child(self) -> None:
        self.assertTrue(check.room_for_launch(5, 7))
        self.assertFalse(check.room_for_launch(6, 7))
        self.assertTrue(check.room_for_launch(None, 7))

    def test_ceiling_refuses_before_the_launch(self) -> None:
        run = fake_run("x", ["never"], 0)
        with self.assertRaises(check.CeilingError):
            check.execute(run, cap=7, counter=lambda: 6)

    def test_ceiling_reads_the_override_then_the_hooks(self) -> None:
        with mock.patch.dict(os.environ, {"LAB_PROCESS_CEILING": "11"}):
            self.assertEqual(check.ceiling(), 11)
        with mock.patch.dict(os.environ, {"LAB_PROCESS_CEILING": "x"}):
            self.assertGreater(check.ceiling(), 0)
        env = {k: v for k, v in os.environ.items() if k != "LAB_PROCESS_CEILING"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(check.ceiling(), check.DEFAULT_CEILING)

    def test_timeout_ends_the_run_by_pid(self) -> None:
        run = check.Run(
            "sleeper",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(120)",
            ],
            ROOT,
            3,
        )
        start = time.monotonic()
        code, _out = check.execute(run, cap=7, counter=no_processes)
        self.assertIsNone(code)
        self.assertLess(time.monotonic() - start, 60)


class TestRunTiers(unittest.TestCase):
    def _run(
        self, runs: dict, tiers: list[str], **kw: object
    ) -> tuple[int, list, list]:
        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "check_runs.jsonl"
            fp = {
                "lib_commit": "abc",
                "lib_diff_sha256": "f00",
                "manifest_sha256": "e3b",
                "lib_dirty": True,
                "head": "abc",
            }
            with mock.patch.object(check, "fingerprint", return_value=fp):
                code = check.run_tiers(
                    tiers,
                    root=ROOT,
                    python=sys.executable,
                    mode="tiers",
                    known=KNOWN,
                    ledger=ledger,
                    counter=kw.get("counter", no_processes),
                    cap=7,
                    runs=runs,
                    emit=emitted.append,
                )
            lines = [
                json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()
            ]
        return code, lines, emitted

    def test_stops_at_the_first_failing_tier_and_records_each_run(self) -> None:
        runs = {
            "harness": [fake_run("h", ["PASSED t.py::a", "1 passed in 0.1s"], 0)],
            "tests": [
                fake_run(
                    "w",
                    ["FAILED tests/test_a.py::test_new - x", "1 failed in 0.1s"],
                    1,
                )
            ],
            "slides": [fake_run("never", ["PASSED x.py::y", "1 passed in 0.1s"], 0)],
        }
        code, lines, emitted = self._run(runs, ["harness", "tests", "slides"])
        self.assertEqual(code, 1)
        self.assertEqual([x["tier"] for x in lines], ["harness", "tests"])
        self.assertEqual([x["ok"] for x in lines], [True, False])
        self.assertEqual(lines[1]["new_failures"], ["tests/test_a.py::test_new"])
        for x in lines:
            self.assertEqual(x["lib_diff_sha256"], "f00")
            self.assertEqual(x["lib_commit"], "abc")
            self.assertEqual(x["python"], Path(sys.executable).name)
        self.assertTrue(emitted[0].startswith("harness  PASS"))
        self.assertTrue(
            any("new failure: tests/test_a.py::test_new" in e for e in emitted)
        )
        self.assertEqual(emitted[-1], "slides: not run (tests failed)")

    def test_known_failure_alone_passes(self) -> None:
        runs = {
            "tests": [
                fake_run(
                    "w",
                    ["FAILED tests/test_a.py::test_known - x", "1 failed in 0.1s"],
                    1,
                )
            ]
        }
        code, lines, _ = self._run(runs, ["tests"])
        self.assertEqual(code, 0)
        self.assertEqual(lines[0]["known_failures"], ["tests/test_a.py::test_known"])

    def test_unexplained_exit_fails(self) -> None:
        code, lines, _ = self._run({"slides": [fake_run("w", ["boom"], 2)]}, ["slides"])
        self.assertEqual(code, 1)
        self.assertIn("pytest exit 2", lines[0]["errors"][0])

    def test_empty_collection_is_not_run(self) -> None:
        # An exit 5 is never a pass, so a slides tier that collected nothing cannot write an
        # ok ledger line that --status would count as the library verified.
        runs = {
            "slides": [fake_run("w", ["no tests ran in 0.1s"], 5)],
            "harness": [fake_run("h", ["1 passed in 0.1s"], 0)],
        }
        code, lines, emitted = self._run(runs, ["harness", "slides"])
        self.assertEqual(code, 1)
        self.assertEqual([x["tier"] for x in lines], ["harness", "slides"])
        self.assertFalse(lines[1]["ok"])
        self.assertEqual(lines[1]["status"], "not-run")
        self.assertEqual(lines[1]["not_run"], ["w"])
        self.assertTrue(any(e.startswith("slides   NOT RUN") for e in emitted))

    def test_a_tier_that_collects_nothing_is_not_run(self) -> None:
        code, lines, _ = self._run(
            {"tests": [fake_run("w", ["5 deselected in 0.1s"], 0)]}, ["tests"]
        )
        self.assertEqual(code, 1)
        self.assertEqual(lines[0]["status"], "not-run")
        self.assertEqual(lines[0]["notes"], ["the tier collected no test"])

    def test_each_later_tier_prints_a_not_run_line(self) -> None:
        failed = check.TierResult("lint", ok=False, status="fail", errors=["x"])
        with mock.patch.object(check, "lint_tier", return_value=failed):
            code, lines, emitted = self._run({}, ["lint", "harness"])
        self.assertEqual(code, 1)
        self.assertEqual([x["tier"] for x in lines], ["lint"])
        self.assertEqual(emitted[-1], "harness: not run (lint failed)")

    def test_ceiling_refusal_is_recorded_and_returns_three(self) -> None:
        runs = {"tests": [fake_run("w", ["1 passed in 0.1s"], 0)]}
        code, lines, _ = self._run(runs, ["tests"], counter=lambda: 6)
        self.assertEqual(code, 3)
        self.assertEqual(lines[0]["status"], "ceiling")


class TestBuildRuns(unittest.TestCase):
    def test_tests_and_slides_split_on_the_marker(self) -> None:
        (tests,) = check.build_runs("tests", ROOT, sys.executable, suite_timeout=1)
        (slides,) = check.build_runs("slides", ROOT, sys.executable, suite_timeout=1)
        self.assertEqual(tests.argv[-2:], ["-m", "not slides"])
        self.assertEqual(slides.argv[-2:], ["-m", "slides"])
        for run in (tests, slides):
            self.assertIn("tests", run.argv)
            self.assertIn("--rootdir=.", run.argv)
            self.assertEqual(run.cwd, ROOT)
            self.assertTrue(run.env["PYTHONPATH"].startswith(str(ROOT / "lib")))

    def test_harness_runs_one_folder_each(self) -> None:
        runs = check.build_runs("harness", ROOT, sys.executable, suite_timeout=1)
        labels = {r.label for r in runs}
        self.assertIn("tools/tests", labels)
        self.assertIn(".claude/hooks/tests", labels)
        self.assertTrue(all(label.endswith("tests") for label in labels))

    def test_unknown_tier_is_a_config_error(self) -> None:
        with self.assertRaises(check.ConfigError):
            check.build_runs("lint", ROOT, sys.executable, suite_timeout=1)

    def test_harness_dirs_skip_caches(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in (
                "tools/tests/test_a.py",
                ".claude/skills/s/tests/test_b.py",
                "tools/.cache/tests/test_c.py",
                "tools/helpers/test_d.py",
            ):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")
            got = [p.relative_to(root).as_posix() for p in check.harness_dirs(root)]
        self.assertEqual(got, [".claude/skills/s/tests", "tools/tests"])


class TestRealPytest(unittest.TestCase):
    def test_a_real_run_gives_root_relative_ids(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "test_a.py").write_text(
                "def test_ok():\n    assert True\n\n\ndef test_known():\n    assert False\n",
                encoding="utf-8",
            )
            run = check.Run(
                "tests",
                [sys.executable, "-m", "pytest", "tests", *check.PYTEST_OPTS],
                root,
                120,
            )
            res = check.pytest_tier("tests", [run], KNOWN, cap=7, counter=no_processes)
        self.assertEqual(res.passed, {"tests/test_a.py::test_ok"})
        self.assertEqual(res.failed, {"tests/test_a.py::test_known"})
        self.assertTrue(res.ok, res.errors)
        self.assertEqual(res.counts["passed"], 1)


class TestSelectionAndStatus(unittest.TestCase):
    def test_tiers_keep_the_run_order(self) -> None:
        ns = check.argparse.Namespace(tiers="slides,lint", full=False)
        self.assertEqual(check.select_tiers(ns), (["lint", "slides"], "tiers"))
        ns = check.argparse.Namespace(tiers=None, full=False)
        self.assertEqual(check.select_tiers(ns)[0], ["lint", "harness", "tests"])
        ns = check.argparse.Namespace(tiers=None, full=True)
        self.assertEqual(
            check.select_tiers(ns), (["lint", "harness", "tests", "slides"], "full")
        )
        with self.assertRaises(check.ConfigError):
            check.select_tiers(check.argparse.Namespace(tiers="slow", full=False))

    def test_status_matches_lines_by_fingerprint(self) -> None:
        fp = {
            "lib_commit": "abc",
            "lib_diff_sha256": "f00",
            "manifest_sha256": "e3b",
            "lib_dirty": False,
            "head": "abc",
        }
        other = {**fp, "lib_diff_sha256": "bad"}
        new_manifest = {**fp, "manifest_sha256": "bad"}
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "l.jsonl"
            rows = [
                {"tier": "lint", "ok": True, "time": "t1", **fp},
                {"tier": "harness", "ok": True, "time": "t2", **other},
                {"tier": "harness", "ok": True, "time": "t3", **new_manifest},
            ]
            ledger.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            with mock.patch.object(check, "fingerprint", return_value=fp):
                ok, _lines = check.status(ROOT, ["lint"], ledger)
                self.assertTrue(ok)
                ok, lines = check.status(ROOT, ["lint", "harness"], ledger)
        self.assertFalse(ok)
        self.assertIn("no run at this fingerprint", lines[-1])

    def test_fingerprint_carries_the_ledger_keys(self) -> None:
        fp = check.fingerprint(ROOT)
        self.assertEqual(
            set(fp),
            {"lib_commit", "lib_diff_sha256", "manifest_sha256", "lib_dirty", "head"},
        )
        self.assertEqual(len(fp["lib_diff_sha256"]), 64)
        self.assertTrue(set(check.MATCH_KEYS) <= set(fp))

    def test_an_edit_to_an_untracked_lib_file_ends_the_match(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}

            def git(*a: str) -> None:
                subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, env=env
                )

            lib = root / check.LIB_DIRS[0]
            lib.mkdir(parents=True)
            (lib / "kept.py").write_text("x = 1\n", encoding="utf-8")
            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            git("add", f"{check.LIB_DIRS[0]}/kept.py")
            git("commit", "-q", "-m", "init")
            new = lib / "new_module.py"
            new.write_text("y = 1\n", encoding="utf-8")
            before = check.fingerprint(root)
            ledger = root / "l.jsonl"
            ledger.write_text(
                json.dumps({"tier": "lint", "ok": True, "time": "t", **before}) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(check.status(root, ["lint"], ledger)[0])
            new.write_text("y = 2\n", encoding="utf-8")
            ok, lines = check.status(root, ["lint"], ledger)
        self.assertFalse(ok)
        self.assertIn("no run at this fingerprint", lines[-1])

    def test_a_changed_manifest_ends_the_match(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lib = root / check.LIB_DIRS[0]
            lib.mkdir(parents=True)
            manifest = lib / "MANIFEST.md"
            manifest.write_bytes(b"one\n")
            with (
                mock.patch.object(check.C, "git", return_value=""),
                mock.patch.object(check.C, "lib_state", return_value=(False, "f00")),
            ):
                before = check.fingerprint(root)
                ledger = root / "l.jsonl"
                ledger.write_text(
                    json.dumps({"tier": "lint", "ok": True, "time": "t", **before})
                    + "\n",
                    encoding="utf-8",
                )
                self.assertTrue(check.status(root, ["lint"], ledger)[0])
                manifest.write_bytes(b"two\n")
                self.assertFalse(check.status(root, ["lint"], ledger)[0])


class TestLint(unittest.TestCase):
    def test_changed_files_read_git_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}

            def git(*a: str) -> None:
                subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, env=env
                )

            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            for name in ("keep.py", "gone.py", "old.py", "notes.md"):
                (Path(d) / name).write_text("x = 1\n", encoding="utf-8")
            git("add", "keep.py", "gone.py", "old.py", "notes.md")
            git("commit", "-q", "-m", "init")
            (Path(d) / "keep.py").write_text("x = 2\n", encoding="utf-8")
            (Path(d) / "gone.py").unlink()
            git("mv", "old.py", "new.py")
            (Path(d) / "sources").mkdir()
            (Path(d) / "sources" / "s.py").write_text("", encoding="utf-8")
            (Path(d) / "fresh.py").write_text("", encoding="utf-8")
            got = check.changed_python_files(Path(d))
        self.assertEqual(got, ["fresh.py", "keep.py", "new.py"])

    def test_a_rule_that_finds_more_than_at_head_fails(self) -> None:
        def counts(
            root: Path, ruff: list, args: list, stdin: str | None = None
        ) -> Counter:
            n = 1 if stdin is not None else 2
            return Counter({("a.py", "F401"): n, ("a.py", "E501"): 1})

        with (
            mock.patch.object(check, "_ruff_counts", side_effect=counts),
            mock.patch.object(check.C, "git_blobs", return_value=[b"import os\n"]),
        ):
            res = check.lint_tier(
                ROOT,
                sys.executable,
                cap=7,
                mypy=False,
                files=["a.py"],
                ruff=["ruff"],
            )
        self.assertFalse(res.ok)
        self.assertEqual(res.tail.strip(), "a.py: F401 1 -> 2")

    def test_ruff_falls_back_to_the_module(self) -> None:
        class Probe:
            returncode = 0

        with (
            mock.patch.object(check, "_load_module", return_value=None),
            mock.patch.object(check.subprocess, "run", return_value=Probe()),
        ):
            self.assertEqual(check.ruff_argv("py"), ["py", "-m", "ruff"])
        Probe.returncode = 1
        with (
            mock.patch.object(check, "_load_module", return_value=None),
            mock.patch.object(check.subprocess, "run", return_value=Probe()),
        ):
            self.assertIsNone(check.ruff_argv("py"))

    def test_missing_ruff_fails_the_tier(self) -> None:
        with mock.patch.object(check, "ruff_argv", return_value=None):
            res = check.lint_tier(
                ROOT, sys.executable, cap=7, mypy=False, files=["tools/check.py"]
            )
        self.assertFalse(res.ok)
        self.assertIn("ruff not found", res.errors[0])

    def _lint_with_mypy(self, lines: list[str], code: int) -> check.TierResult:
        """The lint tier with no ruff finding and a mypy run that prints ``lines``."""
        real = check.execute

        def swap(run: check.Run, *, cap: int, counter: object) -> tuple:
            return real(fake_run(run.label, lines, code), cap=cap, counter=counter)

        with (
            mock.patch.object(check, "_ruff_counts", return_value=Counter()),
            mock.patch.object(check, "execute", side_effect=swap),
        ):
            return check.lint_tier(
                ROOT,
                sys.executable,
                cap=7,
                counter=no_processes,
                files=["tools/_common.py", "lib/seams_engine.py"],
                ruff=["ruff"],
            )

    def test_mypy_count_sums_the_passes_of_run_mypy(self) -> None:
        out = [
            "1 named, 0 errors",
            "lib\\x.py:62: error: Argument 1 has incompatible type  [arg-type]",
            "1 named, 1 errors",
            "tools\\x.py:3: error: a second finding  [misc]",
            "1 named, 2 errors",
        ]
        res = self._lint_with_mypy(out, 1)
        self.assertTrue(res.ok)
        self.assertEqual(res.counts["mypy"], 3)
        self.assertEqual(res.notes, ["mypy exit 1, 3 findings (recorded only)"])

    def test_mypy_crash_is_an_error_note_with_no_count(self) -> None:
        res = self._lint_with_mypy(["mypy: error: bad config", "Found 9 errors"], 2)
        self.assertNotIn("mypy", res.counts)
        self.assertEqual(
            res.notes, ["mypy error: exit 2, a usage or crash report; no count"]
        )
        self.assertIsNone(check.mypy_findings(None, "")[0])
        self.assertEqual(
            check.mypy_findings(1, "a.py:1: error: x\na.py:2: error: y\n")[0], 2
        )

    def test_head_versions_are_read_in_one_batch(self) -> None:
        seen: list[tuple] = []

        def counts(
            root: Path, ruff: list, args: list, stdin: str | None = None
        ) -> Counter:
            seen.append((args[1], stdin))
            return Counter({(args[1], "F401"): 1})

        with (
            mock.patch.object(
                check.C, "git_blobs", return_value=[b"import os\n", None, b""]
            ) as blobs,
            mock.patch.object(check, "_ruff_counts", side_effect=counts),
        ):
            got = check.head_counts(ROOT, ["ruff"], ["a.py", "new.py", "empty.py"])
        blobs.assert_called_once_with(str(ROOT), "HEAD", ["a.py", "new.py", "empty.py"])
        self.assertEqual(seen, [("a.py", "import os\n")])
        self.assertEqual(got, Counter({("a.py", "F401"): 1}))

    def test_mypy_runs_with_full_or_the_flag(self) -> None:
        def ns(**kw: bool) -> check.argparse.Namespace:
            return check.argparse.Namespace(**{"mypy": False, "no_mypy": False, **kw})

        self.assertFalse(check.mypy_wanted(ns(), "quick"))
        self.assertFalse(check.mypy_wanted(ns(), "tiers"))
        self.assertTrue(check.mypy_wanted(ns(), "full"))
        self.assertTrue(check.mypy_wanted(ns(mypy=True), "quick"))
        self.assertFalse(check.mypy_wanted(ns(no_mypy=True), "full"))

    def test_lint_without_mypy_says_so(self) -> None:
        with mock.patch.object(check, "_ruff_counts", return_value=Counter()):
            res = check.lint_tier(
                ROOT,
                sys.executable,
                cap=7,
                mypy=False,
                files=["tools/_common.py"],
                ruff=["ruff"],
            )
        self.assertTrue(res.ok)
        self.assertEqual(
            res.notes, ["mypy not run; --mypy or --full records its count"]
        )

    def test_no_changed_file_passes(self) -> None:
        res = check.lint_tier(ROOT, sys.executable, cap=7, mypy=False, files=[])
        self.assertTrue(res.ok)
        self.assertEqual(res.notes, ["no changed Python file"])


class TestInterpreter(unittest.TestCase):
    def test_python_comes_from_the_flag_then_lab_py_then_this_one(self) -> None:
        seen: list[str] = []

        def fake_run_tiers(tiers: list, **kw: object) -> int:
            seen.append(str(kw["python"]))
            return 0

        with mock.patch.object(check, "run_tiers", side_effect=fake_run_tiers):
            with mock.patch.dict(os.environ, {"LAB_PY": "from-env"}):
                check.main(["--tiers", "harness"])
                check.main(["--tiers", "harness", "--python", "from-flag"])
            env = {k: v for k, v in os.environ.items() if k != "LAB_PY"}
            with mock.patch.dict(os.environ, env, clear=True):
                check.main(["--tiers", "harness"])
        self.assertEqual(seen, ["from-env", "from-flag", sys.executable])


if __name__ == "__main__":
    unittest.main()
