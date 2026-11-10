#!/usr/bin/env python
# SPDX-FileCopyrightText: Market Design Lab contributors
#
# SPDX-License-Identifier: MIT

"""Run mypy over the repository in scope, or over named modules.

Usage:
  python tools/run_mypy.py [--cache NAME]
  python tools/run_mypy.py MODULE... [--cache NAME]

One pass reads the `files` of mypy.ini: lib/, tools/, .claude/hooks/ and the
writing-voice scripts. The pass imports with --follow-imports=silent, so a module's
findings print once, and a named module prints its own findings with the types of what
it imports.

The pass keeps its cache under tools/.cache/mypy/<pass>, and a run over named modules
keeps a second one under tools/.cache/mypy/<pass>-modules. mypy 1.19.1 replays the cached
findings of an imported module whatever --follow-imports says, so a run over named modules
prints only the findings in those modules, with its own count. --cache NAME adds a folder
level, which gives a parallel agent a cache of its own, and is not needed for a correct
report. The exit status is 0 clean, 1 findings, 2 a usage or crash report.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import _common as C


class MypyPass(NamedTuple):
    """One mypy run.

    Attributes
    ----------
    name : str
        The pass, which names its cache folder.
    half : str
        A folder placed on MYPYPATH, empty for none.
    paths : tuple of str
        The paths the pass reads, empty for the `files` of mypy.ini.

    """

    name: str
    half: str
    paths: tuple[str, ...]


#: The one pass, over the `files` of mypy.ini.
PASSES = (MypyPass("shared", "", ()),)


def pass_of(rel: str) -> MypyPass:
    """The pass that holds a module, which is the one pass.

    Parameters
    ----------
    rel : str
        The module's repository-relative path with forward slashes.

    Returns
    -------
    MypyPass
        The shared pass.

    """
    del rel
    return PASSES[0]


def own_findings(output: str, paths: list[str]) -> list[str]:
    """The lines of mypy's output that fall in the named modules.

    Parameters
    ----------
    output : str
        mypy's standard output.
    paths : list of str
        The named modules or folders, repository-relative with forward slashes.

    Returns
    -------
    list of str
        The finding and note lines of those paths, as mypy printed them.

    """
    starts = tuple(f"{path}{mark}" for path in paths for mark in (":", "/"))
    return [
        line
        for line in output.splitlines()
        if line.replace("\\", "/").startswith(starts)
    ]


def run_pass(root: Path, item: MypyPass, paths: list[str], cache: str) -> int:
    """Run one pass and print its findings.

    Parameters
    ----------
    root : Path
        The repository root, which is the working directory of the run.
    item : MypyPass
        The pass.
    paths : list of str
        The modules to read, or empty for all the pass holds.
    cache : str
        The caller's folder under tools/.cache/mypy, empty for none.

    Returns
    -------
    int
        The exit status of the pass.

    """
    name = f"{item.name}-modules" if paths else item.name
    folder = Path("tools/.cache/mypy", cache, name).as_posix()
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        "mypy.ini",
        "--follow-imports=silent",
        "--cache-dir",
        folder,
        *(paths or item.paths),
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    if item.half:
        env["MYPYPATH"] = item.half
    if not paths:
        return subprocess.run(command, cwd=root, env=env).returncode
    done = subprocess.run(
        command, cwd=root, env=env, capture_output=True, encoding="utf-8"
    )
    if done.returncode > 1:
        print(done.stdout + done.stderr)
        return done.returncode
    kept = own_findings(done.stdout, paths)
    errors = sum(": error:" in line for line in kept)
    print(*kept, f"{len(paths)} named, {errors} errors", sep="\n")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    """The entry point.

    Parameters
    ----------
    argv : list of str, optional
        The arguments, sys.argv by default.

    Returns
    -------
    int
        The exit status stated in the module docstring.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("modules", nargs="*", metavar="MODULE")
    parser.add_argument("--cache", default="", help="a cache folder of this caller")
    args = parser.parse_args(argv)
    root = Path(C.repo_root(str(Path(__file__).resolve().parent)))
    runs: list[tuple[MypyPass, list[str]]] = [(item, []) for item in PASSES]
    if args.modules:
        grouped: dict[MypyPass, list[str]] = {}
        for name in args.modules:
            rel = Path(name).resolve().relative_to(root.resolve()).as_posix()
            grouped.setdefault(pass_of(rel), []).append(rel)
        runs = list(grouped.items())
    return max(run_pass(root, item, paths, args.cache) for item, paths in runs)


if __name__ == "__main__":
    sys.exit(main())
