# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""Tests for tools/lab_env.py, tools/env_lock.json, constraints.txt and requirements-dev.txt.

The check warns and never raises, ``LAB_PY`` names the examined interpreter first, and each
version the lock records equals its pin in constraints.txt and requirements-dev.txt.

Run: python -m pytest tools/tests/test_lab_env.py -q   (any Python 3.10 or later)
"""

import ast
import io
import json
import os
import re
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import lab_env  # noqa: E402

LOCK = {
    "environment": "tested",
    "python": "3.12.12",
    "packages": {
        "numpy": {"version": "2.3.5", "role": "model"},
        "pandas": {"version": "2.3.3", "role": "model"},
        "ruff": {"version": "0.15.5", "role": "tooling"},
    },
}
#: The packages whose version moves a stored result, and those that move a check.
MODEL = {"numpy", "pandas", "scipy", "matplotlib", "pypsa", "pandapower", "pycirclize"}
TOOLING = {"pytest", "nbformat", "nbconvert", "ipykernel", "ruff", "mypy", "pyyaml"}


def _write(folder: str, name: str, text: str) -> Path:
    p = Path(folder) / name
    p.write_bytes(text.encode("utf-8"))
    return p


def _norm(name: str) -> str:
    """A distribution name in the normalized form of PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    """The ``name==version`` lines of a requirements or constraints file, by normalized name."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;#]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(3)
    return pins


class TestProblems(unittest.TestCase):
    def test_matching_interpreter_has_no_problem(self) -> None:
        got = lab_env.environment_problems(
            LOCK, versions={"numpy": "2.3.5", "pandas": "2.3.3"}
        )
        self.assertEqual(got, [])

    def test_model_versions_are_compared_and_tooling_is_not(self) -> None:
        got = lab_env.environment_problems(
            LOCK, versions={"numpy": "2.5.2", "pandas": None, "ruff": "0.1.0"}
        )
        self.assertEqual(
            got, ["numpy 2.5.2 (pinned 2.3.5)", "pandas absent (pinned 2.3.3)"]
        )

    def test_the_python_release_is_information_only(self) -> None:
        lock = {**LOCK, "python": "2.7.18"}
        got = lab_env.environment_problems(
            lock, versions={"numpy": "2.3.5", "pandas": "2.3.3"}
        )
        self.assertEqual(got, [])


class TestCheck(unittest.TestCase):
    def test_absent_lock_is_silent(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            self.assertEqual(lab_env.check_environment(Path(d) / "none.json"), [])
        self.assertEqual(w, [])

    def test_mismatch_warns_once_and_names_the_install(self) -> None:
        lock = {**LOCK, "packages": {"numpy": {"version": "0.0.1", "role": "model"}}}
        with (
            tempfile.TemporaryDirectory() as d,
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            p = _write(d, "lock.json", json.dumps(lock))
            got = lab_env.check_environment(p)
        self.assertIn("pinned 0.0.1", got[0])
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, lab_env.EnvironmentMismatchWarning)
        text = str(w[0].message)
        self.assertTrue(text.startswith("ENVIRONMENT MISMATCH"))
        self.assertIn(sys.executable, text)
        self.assertIn(lab_env.INSTALL_LINE, text)
        self.assertIn("LAB_PY", text)

    def test_warning_states_only_what_the_lock_records(self) -> None:
        lock = {**LOCK, "written": "2026-11-10"}
        text = lab_env._message(["numpy 2.5.2 (pinned 2.3.5)"], lock)
        self.assertIn(
            "tools/env_lock.json records the tested environment as written on 2026-11-10",
            text,
        )
        self.assertNotIn("were produced", text)
        self.assertIn("A number computed here may differ from the stored one.", text)
        undated = lab_env._message(["x"], LOCK)
        self.assertIn("records the tested environment. Differences: x.", undated)

    def test_warning_filter_set_to_error_does_not_raise(self) -> None:
        lock = {**LOCK, "packages": {"numpy": {"version": "0.0.1", "role": "model"}}}
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter("error")
            p = _write(d, "lock.json", json.dumps(lock))
            with mock.patch.object(sys, "stderr", err):
                got = lab_env.check_environment(p)
        self.assertTrue(got)
        self.assertIn("ENVIRONMENT MISMATCH", err.getvalue())

    def test_malformed_lock_warns_and_does_not_raise(self) -> None:
        for text in ("{not json", "[1, 2]"):
            with (
                tempfile.TemporaryDirectory() as d,
                warnings.catch_warnings(record=True) as w,
            ):
                warnings.simplefilter("always")
                got = lab_env.check_environment(_write(d, "lock.json", text))
            self.assertEqual(len(got), 1, text)
            self.assertIn("check failed", got[0])
            self.assertIn("ENVIRONMENT CHECK FAILED", str(w[0].message))

    def test_non_object_lock_raises_type_error_from_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "lock.json", "[1]")
            with self.assertRaises(TypeError):
                lab_env.load_lock(p)


class TestInterpreter(unittest.TestCase):
    def test_lab_py_comes_first(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            exe = _write(d, "python.exe", "")
            with mock.patch.dict(os.environ, {"LAB_PY": str(exe)}):
                self.assertEqual(lab_env.lab_interpreter(), exe)

    def test_the_running_interpreter_stands_otherwise(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LAB_PY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(lab_env.lab_interpreter(), Path(sys.executable))
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(os.environ, {"LAB_PY": str(Path(d) / "absent")}),
        ):
            self.assertEqual(lab_env.lab_interpreter(), Path(sys.executable))


class TestTrackedFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = lab_env.load_lock(ROOT / "tools" / "env_lock.json")
        self.assertIsNotNone(self.lock)

    def test_lock_has_the_fields_the_tools_read_and_no_machine_fields(self) -> None:
        lock = self.lock
        for key in ("about", "written", "environment", "python", "packages"):
            self.assertIn(key, lock)
        self.assertEqual(lock["environment"], "tested")
        self.assertRegex(lock["written"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(lock["python"], r"^3\.\d+\.\d+$")
        self.assertNotIn("interpreters", lock)
        self.assertNotIn("optional", lock)
        self.assertNotRegex(lock["about"], r"[A-Za-z]:[\\/]|[\\/]Users[\\/]|~[\\/]")
        roles = {d: s.get("role") for d, s in lock["packages"].items()}
        self.assertEqual({d for d, r in roles.items() if r == "model"}, MODEL)
        self.assertEqual({d for d, r in roles.items() if r == "tooling"}, TOOLING)
        for dist, spec in lock["packages"].items():
            self.assertRegex(str(spec.get("version")), r"^\d", dist)

    def test_every_lock_version_equals_its_pin_in_constraints(self) -> None:
        pins = _pins(ROOT / "constraints.txt")
        for dist, spec in self.lock["packages"].items():
            self.assertEqual(pins.get(_norm(dist)), spec["version"], dist)

    def test_every_tooling_version_equals_its_pin_in_requirements_dev(self) -> None:
        pins = _pins(ROOT / "requirements-dev.txt")
        for dist, spec in self.lock["packages"].items():
            if spec["role"] == "tooling":
                self.assertEqual(pins.get(_norm(dist)), spec["version"], dist)

    def test_requirements_dev_pins_agree_with_constraints(self) -> None:
        dev = _pins(ROOT / "requirements-dev.txt")
        pins = _pins(ROOT / "constraints.txt")
        for dist, ver in dev.items():
            self.assertEqual(pins.get(dist), ver, dist)

    def test_constraints_name_no_local_path_or_editable_install(self) -> None:
        text = (ROOT / "constraints.txt").read_text(encoding="utf-8")
        for pat in (r"[A-Za-z]:[\\/]", r"file://", r"(?m)^\s*-e\s", r" @ "):
            self.assertNotRegex(text, pat)
        self.assertIn("pip install -r requirements-dev.txt -c constraints.txt", text)

    def test_lab_env_uses_the_standard_library_only(self) -> None:
        src = (ROOT / "tools" / "lab_env.py").read_text(encoding="utf-8")
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertLessEqual(
            mods,
            {
                "__future__",
                "json",
                "os",
                "sys",
                "warnings",
                "importlib",
                "pathlib",
                "typing",
            },
        )


if __name__ == "__main__":
    unittest.main()
