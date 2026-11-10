# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""The environment check that the headless re-run and the verify command run first.

The module carries no model. It compares the running interpreter with
``tools/env_lock.json``, which records the versions of the packages of the tested
environment. A package whose role is ``"model"`` moves a stored result when its version
moves, so each one is compared by its exact version. A difference is a warning and never an
error, because the documented Colab path runs the notebooks under other versions. Where the
lock is absent the check does nothing.

The lock records the Python release it was tested under as information only. The
repository admits Python 3.10 and later, so a different release is not a difference.

``tools/nb_execute.py`` and ``tools/check.py`` call ``check_environment()``. The notebooks'
bootstrap does not, and neither does ``tests/conftest.py``. The module uses the standard
library only, so that ``tools/doctor.py`` imports it under any interpreter. ``LAB_PY`` names
the interpreter that ``tools/doctor.py`` and ``tools/check.py`` examine, where it is set.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "INSTALL_LINE",
    "LAB_PY_VAR",
    "LOCK_PATH",
    "EnvironmentMismatchWarning",
    "check_environment",
    "environment_problems",
    "lab_interpreter",
    "load_lock",
]

#: The repository root, one level above ``tools/``.
ROOT = Path(__file__).resolve().parents[1]
#: The lock file.
LOCK_PATH = ROOT / "tools" / "env_lock.json"
#: The environment variable that names the interpreter to examine.
LAB_PY_VAR = "LAB_PY"
#: The install that brings an environment to the lock's versions.
INSTALL_LINE = "pip install -r requirements-dev.txt -c constraints.txt"


class EnvironmentMismatchWarning(UserWarning):
    """A model package of the running interpreter differs from ``tools/env_lock.json``."""


def load_lock(path: Path | str | None = None) -> dict[str, Any] | None:
    """The lock as a mapping, or None when the file is absent.

    Parameters
    ----------
    path : Path or str, optional
        The lock file. The default is ``LOCK_PATH``.

    Returns
    -------
    dict or None
        The parsed lock, or None when no file exists at the path.

    Raises
    ------
    TypeError
        When the file does not hold a JSON object.

    """
    p = LOCK_PATH if path is None else Path(path)
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{p} holds a {type(data).__name__!r}, not a JSON object")
    return data


def lab_interpreter() -> Path:
    """The interpreter to examine: ``LAB_PY`` where it names a file, the running one otherwise.

    Returns
    -------
    Path
        The interpreter's path, with ``~`` expanded in ``LAB_PY``.

    """
    env = os.environ.get(LAB_PY_VAR)
    if env and Path(env).expanduser().is_file():
        return Path(env).expanduser()
    return Path(sys.executable)


def _installed(dist: str) -> str | None:
    """The installed version of a distribution, or None when it is absent."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def environment_problems(
    lock: dict[str, Any], *, versions: dict[str, str | None] | None = None
) -> list[str]:
    """The model packages that differ from the lock, one phrase each.

    Each package whose role is ``"model"`` is compared by its exact version, and an absent
    model package is a difference. A ``"tooling"`` package is left to ``tools/doctor.py``,
    because it moves a check and not a stored result. The Python release is not compared.

    Parameters
    ----------
    lock : dict
        The parsed lock.
    versions : dict, optional
        Installed versions by distribution name, None for an absent one. The default reads
        ``importlib.metadata``.

    Returns
    -------
    list of str
        One phrase per difference, as ``"numpy 2.5.2 (pinned 2.3.5)"``. Empty when every
        model package matches.

    """
    problems: list[str] = []
    for dist, spec in sorted((lock.get("packages") or {}).items()):
        if not isinstance(spec, dict) or spec.get("role") != "model":
            continue
        want = str(spec.get("version", ""))
        have = versions.get(dist) if versions is not None else _installed(dist)
        if have is None:
            problems.append(f"{dist} absent (pinned {want})")
        elif have != want:
            problems.append(f"{dist} {have} (pinned {want})")
    return problems


def _message(problems: list[str], lock: dict[str, Any]) -> str:
    """The warning text: what differs, which interpreter runs, and the install that fixes it.

    The text states only what the lock records, the environment and the date the lock was
    written. It does not say which environment produced a stored result, because the lock
    carries no record of that.
    """
    env = str(lock.get("environment", "tested"))
    written = str(lock.get("written", "")).strip()
    when = f" as written on {written}" if written else ""
    return (
        f"ENVIRONMENT MISMATCH. This run uses Python {sys.version.split()[0]} at "
        f"{sys.executable}. tools/env_lock.json records the {env} environment{when}. "
        f"Differences: {'; '.join(problems)}. Install the recorded versions with "
        f"{INSTALL_LINE}, or set {LAB_PY_VAR} to an interpreter that carries them. "
        "A number computed here may differ from the stored one."
    )


def check_environment(
    lock_path: Path | str | None = None, *, stacklevel: int = 2
) -> list[str]:
    """The differences from the lock, warned once and never raised.

    The function issues one ``EnvironmentMismatchWarning`` that lists every difference and
    names the install that removes them. Any error of its own, a malformed lock included,
    becomes a warning of the same category. A warnings filter set to ``"error"`` does not
    make it raise: the text then goes to standard error instead.

    Parameters
    ----------
    lock_path : Path or str, optional
        The lock file. The default is ``LOCK_PATH``.
    stacklevel : int, optional
        The ``warnings.warn`` stack level. The default of 2 points at the caller.

    Returns
    -------
    list of str
        The differences, empty when every model package matches or the lock is absent.

    """
    try:
        lock = load_lock(lock_path)
        if lock is None:
            return []
        problems = environment_problems(lock)
        if not problems:
            return []
        text = _message(problems, lock)
    except Exception as exc:  # noqa: BLE001 - the check must never stop a run; it warns instead
        problems = [f"the environment check failed: {exc!r}"]
        text = f"ENVIRONMENT CHECK FAILED: {exc!r}. The interpreter was not compared."
    try:
        warnings.warn(text, EnvironmentMismatchWarning, stacklevel=stacklevel)
    except Exception:  # noqa: BLE001 - a filter set to "error" raises here; the text goes to stderr
        print(text, file=sys.stderr)  # noqa: T201 - the fallback when the warning cannot be issued
    return problems
