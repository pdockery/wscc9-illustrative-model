# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""Tests for tools/doctor.py and tools/required_data.txt.

The report is built from an injected lock, probe report and process count on a temporary
root, so no probe of an interpreter runs here. The tracked data list is checked against the
test files it names.

Run: python -m pytest tools/tests/test_doctor.py -q   (any Python 3.10 or later)
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import doctor  # noqa: E402

LOCK = {
    "environment": "tested",
    "python": "3.12.12",
    "packages": {"numpy": {"version": "2.3.5", "role": "model"}},
}
GOOD_PROBE = {
    "executable": "/env/python",
    "version": "3.12.12",
    "packages": {
        "numpy": "2.3.5",
        "nbformat": "5.10.4",
        "nbconvert": "7.17.0",
        "nbclient": "0.10.4",
        "ipykernel": "7.2.0",
    },
    "kernelspecs": {"python3": {"argv0": "/env/python"}},
}
KEYS = {
    "ok",
    "written",
    "interpreter",
    "examined",
    "packages",
    "notebook",
    "manifests",
    "data",
    "processes",
    "hooks_path",
    "problems",
}


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _manifest(rows: dict[str, str]) -> str:
    body = "".join(f"| `{n}` | `{h}` |\n" for n, h in rows.items())
    return f"# Manifest\n\n| file | SHA-256 |\n|---|---|\n{body}"


def _root(d: str, data: str = "") -> Path:
    root = Path(d)
    _put(root / "tools" / "required_data.txt", data)
    return root


def _diagnose(root: Path, **kw: object) -> dict:
    args = {"lock": LOCK, "report": GOOD_PROBE, "processes": 0, **kw}
    env = {k: v for k, v in os.environ.items() if k != "LAB_PY"}
    with mock.patch.dict(os.environ, env, clear=True):
        return doctor.diagnose(root, **args)


class TestRequiredData(unittest.TestCase):
    def test_entries_parse_and_bad_lines_raise(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = _put(Path(d) / "r.txt", "# c\n\na/b.csv | local | t.py::x, t.py::y\n")
            self.assertEqual(
                doctor.read_required_data(p),
                [{"path": "a/b.csv", "kind": "local", "tests": ["t.py::x", "t.py::y"]}],
            )
            _put(p, "a/b.csv | elsewhere | t.py::x\n")
            with self.assertRaisesRegex(ValueError, "line 1"):
                doctor.read_required_data(p)

    def test_presence_reads_globs_files_and_folders(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _put(root / "f" / "X_1.zip", "")
            got = doctor.data_status(
                root,
                [
                    {"path": "f/X_*.zip", "kind": "tracked", "tests": []},
                    {"path": "f/Y_*.zip", "kind": "tracked", "tests": []},
                    {"path": "f", "kind": "tracked", "tests": []},
                    {"path": "g.csv", "kind": "local", "tests": []},
                ],
            )
        self.assertEqual([e["present"] for e in got], [True, False, True, False])

    def test_tracked_list_names_the_golden_and_tests_that_exist(self) -> None:
        entries = doctor.read_required_data(ROOT / "tools" / "required_data.txt")
        self.assertIn("tests/golden/golden.json", [e["path"] for e in entries])
        for e in entries:
            self.assertTrue(e["tests"], e["path"])
            for t in e["tests"]:
                path, _, name = t.partition("::")
                src = (ROOT / path).read_text(encoding="utf-8")
                self.assertRegex(src, rf"(?m)^def {re.escape(name)}\b", t)


class TestCeiling(unittest.TestCase):
    def test_limits_file_sets_the_ceiling_and_its_absence_keeps_seven(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(doctor.ceiling(root), 7)
            _put(root / ".claude" / "hooks" / "_limits.py", "CEILING = 5\n")
            self.assertEqual(doctor.ceiling(root), 5)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _put(root / ".claude" / "hooks" / "_limits.py", "CEILING = (\n")
            self.assertEqual(doctor.ceiling(root), 7)

    def test_the_loaded_module_sets_the_ceiling_an_override_included(self) -> None:
        src = (
            "import os\n"
            "CEILING = int(os.environ.get('DOCTOR_TEST_CEILING', '7'))\n"
            "def python_count():\n    return 0\n"
            "def count_from_table(procs, mine):\n    return 0\n"
        )
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(os.environ, {"DOCTOR_TEST_CEILING": "4"}),
        ):
            root = Path(d)
            _put(root / ".claude" / "hooks" / "_limits.py", src)
            self.assertEqual(doctor.ceiling(root), 4)

    def test_python_count_is_a_count_or_none(self) -> None:
        n = doctor.python_count()
        self.assertTrue(n is None or n >= 0)


#: A process table of pid to (image, parent pid). Process 10 is the caller, and its parent 9
#: is a Python interpreter, so the guard's rule leaves both out.
TABLE = {
    1: ("explorer.exe", 0),
    2: ("python.exe", 1),
    3: ("python3.exe", 1),
    4: ("python3.12.exe", 1),
    5: ("pythonw.exe", 1),
    6: ("/usr/bin/python3", 1),
    7: ("python-launcher.exe", 1),
    8: ("pytest.exe", 1),
    9: ("C:\\envs\\lab\\python.exe", 1),
    10: ("python.exe", 9),
}


class TestCountAgreesWithTheGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = doctor.load_limits()
        if self.limits is None:
            self.skipTest(".claude/hooks/_limits.py does not load")

    def test_the_fallback_rule_matches_the_guards_rule(self) -> None:
        want = self.limits.count_from_table(TABLE, 10)
        self.assertEqual(want, 5)
        self.assertEqual(doctor.count_table(TABLE, 10), want)
        no_parent = {k: v for k, v in TABLE.items() if k != 9}
        self.assertEqual(
            doctor.count_table(no_parent, 10),
            self.limits.count_from_table(no_parent, 10),
        )

    def test_python_count_is_the_guards_count(self) -> None:
        with (
            mock.patch.object(self.limits, "windows_table", return_value=TABLE),
            mock.patch.object(self.limits, "posix_table", return_value=TABLE),
            mock.patch.object(os, "getpid", return_value=10),
            mock.patch.object(doctor, "fallback_count", return_value=-1),
        ):
            self.assertEqual(doctor.python_count(), 5)

    def test_an_unloadable_module_falls_back_to_the_reader(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            broken = _put(Path(d) / "_limits.py", "raise RuntimeError('broken')\n")
            self.assertIsNone(doctor.load_limits(broken))
            with mock.patch.object(doctor, "fallback_count", return_value=3):
                self.assertEqual(doctor.python_count(broken), 3)

    def test_the_readers_parse_their_tables(self) -> None:
        tasklist = (
            '"python3.exe","4","Console","1","10,000 K"\n'
            '"explorer.exe","1","Console","1","90,000 K"\n'
        )
        self.assertEqual(
            doctor.parse_tasklist(tasklist),
            {4: ("python3.exe", 0), 1: ("explorer.exe", 0)},
        )
        ps = "  10     9 python3\n   9     1 /usr/bin/python3.12\n"
        table = doctor.parse_ps(ps)
        self.assertEqual(table, self.limits.parse_ps(ps))
        self.assertEqual(doctor.count_table(table, 10), 0)


class TestManifests(unittest.TestCase):
    def test_each_state_is_read_from_the_lf_form(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lf = b"A = 1\nB = 2\n"
            crlf = lf.replace(b"\n", b"\r\n")
            _put(root / "lib" / "same.py", crlf)  # a CRLF checkout of an LF blob
            _put(root / "lib" / "old.py", crlf)
            _put(root / "lib" / "edited.py", b"A = 3\n")
            _put(root / "lib" / "extra.py", lf)
            rows = {
                "same.py": hashlib.sha256(lf).hexdigest(),
                "old.py": hashlib.sha256(crlf).hexdigest(),
                "edited.py": hashlib.sha256(lf).hexdigest(),
                "gone.py": hashlib.sha256(lf).hexdigest(),
            }
            _put(root / "lib" / "MANIFEST.md", _manifest(rows))
            st = doctor.manifest_status(root, "lib", "lib/MANIFEST.md", True)
            partial = doctor.manifest_status(root, "lib", "lib/MANIFEST.md", False)
        self.assertEqual(
            st["files"],
            {
                "edited.py": "differs",
                "gone.py": "absent",
                "old.py": "crlf",
                "same.py": "match",
            },
        )
        self.assertEqual(st["unlisted"], ["extra.py"])
        self.assertEqual(partial["unlisted"], [])

    def test_manifest_problems_reach_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _root(d)
            _put(root / "lib" / "a.py", "A = 1\n")
            _put(root / "lib" / "b.py", "B = 1\n")
            _put(root / "lib" / "MANIFEST.md", _manifest({"a.py": "0" * 64}))
            _put(root / ".claude" / "hooks" / "own_hook.py", "X = 1\n")
            good = hashlib.sha256(b"G = 1\n").hexdigest()
            _put(root / ".claude" / "hooks" / "guard.py", "G = 1\n")
            _put(
                root / ".claude" / "hooks" / "MANIFEST.md",
                _manifest({"guard.py": good}),
            )
            rep = _diagnose(root)
        text = "\n".join(rep["problems"])
        self.assertIn("lib/a.py differs from lib/MANIFEST.md", text)
        self.assertIn("lib/b.py is not listed in lib/MANIFEST.md", text)
        self.assertNotIn("own_hook.py", text)
        self.assertEqual(
            rep["manifests"][".claude/hooks"]["files"], {"guard.py": "match"}
        )

    def test_a_folder_without_its_manifest_is_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _root(d)
            _put(root / "lib" / "a.py", "A = 1\n")
            rep = _diagnose(root)
        self.assertIn("lib/MANIFEST.md is absent", rep["problems"])


class TestDiagnose(unittest.TestCase):
    def test_matching_machine_is_ok_with_every_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rep = _diagnose(_root(d))
        self.assertEqual(set(rep), KEYS)
        self.assertEqual(rep["problems"], [])
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["examined"]["from"], "the running interpreter")

    def test_each_difference_is_a_problem(self) -> None:
        probe = {
            **GOOD_PROBE,
            "packages": {**GOOD_PROBE["packages"], "numpy": "2.5.2", "nbconvert": None},
            "kernelspecs": {},
        }
        data = "missing.csv | local | tests/test_x.py::test_y\n"
        with tempfile.TemporaryDirectory() as d:
            root = _root(d, data)
            nb = {"metadata": {"kernelspec": {"name": "python3"}}, "cells": []}
            _put(root / "a.ipynb", json.dumps(nb))
            rep = _diagnose(root, report=probe, processes=7)
        text = "\n".join(rep["problems"])
        self.assertFalse(rep["ok"])
        self.assertIn("numpy 2.5.2 (pinned 2.3.5)", text)
        self.assertIn("pip install -r requirements-dev.txt -c constraints.txt", text)
        self.assertIn("notebook tooling absent: nbconvert", text)
        self.assertIn(
            "no Jupyter kernelspec 'python3' is installed; it is named by a.ipynb", text
        )
        self.assertIn(
            "missing.csv (local) is absent; these need it: tests/test_x.py::test_y",
            text,
        )
        self.assertIn("by PID", text)

    def test_the_python_release_and_hooks_path_are_information(self) -> None:
        probe = {**GOOD_PROBE, "version": "3.10.4"}
        with tempfile.TemporaryDirectory() as d:
            root = _root(d)
            (root / ".githooks").mkdir()
            rep = _diagnose(root, report=probe)
        self.assertEqual(rep["problems"], [])
        self.assertEqual(rep["examined"]["version"], "3.10.4")
        self.assertEqual(rep["examined"]["tested"], "3.12.12")
        self.assertEqual(rep["hooks_path"]["expected"], ".githooks")

    def test_below_the_ceiling_the_count_is_information(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rep = _diagnose(_root(d), processes=6)
        self.assertEqual(rep["problems"], [])
        self.assertEqual(rep["processes"], {"count": 6, "ceiling": 7})

    def test_absent_lock_and_probe_error_are_problems(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _root(d)
            rep = _diagnose(root, lock={})
            self.assertIn("tools/env_lock.json is absent", rep["problems"][0])
            probe = {
                "executable": "/env/python",
                "error": "the probe did not run: boom",
            }
            rep = _diagnose(root, report=probe)
        self.assertIn("boom", rep["problems"][0])
        self.assertEqual(len(rep["problems"]), 1)

    def test_lab_py_names_the_examined_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _root(d)
            exe = _put(root / "py.exe", "")
            seen = {}

            def fake_probe(python: Path, names: list, root: Path) -> dict:
                seen["python"] = python
                return {**GOOD_PROBE, "executable": str(python)}

            with (
                mock.patch.dict(os.environ, {"LAB_PY": str(exe)}),
                mock.patch.object(doctor, "probe", side_effect=fake_probe),
            ):
                rep = doctor.diagnose(root, lock=LOCK, processes=0)
        self.assertEqual(seen["python"], exe)
        self.assertEqual(rep["examined"]["from"], "LAB_PY")
        self.assertEqual(rep["examined"]["path"], str(exe))

    def test_main_writes_the_cache_and_returns_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rep = _diagnose(_root(d))
            out = Path(d) / "cache" / "doctor.json"
            with (
                mock.patch.object(doctor, "diagnose", return_value=rep),
                mock.patch("sys.stdout"),
            ):
                self.assertEqual(doctor.main(["--out", str(out)]), 0)
            raw = out.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(set(json.loads(raw)), KEYS)
            bad = {**rep, "ok": False, "problems": ["x"]}
            with (
                mock.patch.object(doctor, "diagnose", return_value=bad),
                mock.patch("sys.stdout"),
            ):
                self.assertEqual(doctor.main(["--out", str(out)]), 1)
            with (
                mock.patch.object(doctor, "diagnose", side_effect=OSError("disk")),
                mock.patch("sys.stderr"),
            ):
                self.assertEqual(doctor.main(["--out", str(out)]), 2)

    def test_render_names_each_check(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            text = doctor.render(_diagnose(_root(d)))
        for label in (
            "examined",
            "information only",
            "packages",
            "kernelspecs",
            "manifest",
            "data",
            "processes",
            "core.hooksPath",
        ):
            self.assertIn(label, text)

    def test_doctor_uses_the_standard_library_only(self) -> None:
        import ast  # noqa: PLC0415 - used by this test only

        src = (ROOT / "tools" / "doctor.py").read_text(encoding="utf-8")
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertLessEqual(
            mods - {"_common", "lab_env"}, set(sys.stdlib_module_names)
        )


if __name__ == "__main__":
    unittest.main()
