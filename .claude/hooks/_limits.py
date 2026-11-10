"""The machine's process ceiling and the count of Python interpreters, shared by the hooks.

The ceiling is the number of running Python interpreter processes at which a new heavy launch
(a test suite, a notebook execution or a clearing script) is refused. Orphaned suites and
parallel clearing runs have exhausted the machine's memory at this level before. The
environment variable `LAB_PROCESS_CEILING`, a positive integer, sets another ceiling, and an
unset or invalid value leaves the default. `guard.py`
imports `CEILING`, `BACKGROUND_MARGIN` and `python_count`, and the session brief and the agent
role files cite this module rather than a number of their own.

The count reads the Toolhelp32 snapshot on Windows and `ps` on POSIX. It leaves out the calling
process and its parent when the parent is itself a Python interpreter, which is the launcher of
a hook run. A count that cannot be taken returns None, and every caller treats None as "allow".
Standard library only (CODE_STYLE.md 13.3).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

#: The environment variable that sets the ceiling.
CEILING_VAR = "LAB_PROCESS_CEILING"
#: Python interpreter processes at which a heavy launch is refused, by default. It stays a
#: plain integer assignment, because a readiness report can read the literal from the source.
CEILING = 7
#: How far below the ceiling the count must sit for a heavy launch sent to the background,
#: which keeps running after the call returns and so is not seen by the next count.
BACKGROUND_MARGIN = 2
#: A process image name that is a Python interpreter: python, python3, python3.12, pythonw.
PYTHON_IMAGE = re.compile(r"^pythonw?(\d+(\.\d+)?)?(\.exe)?$", re.I)


def ceiling_from_env(default: int, environ: dict[str, str] | None = None) -> int:
    """The ceiling `LAB_PROCESS_CEILING` sets, or `default` when it is unset or invalid."""
    env = os.environ if environ is None else environ
    raw = env.get(CEILING_VAR, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


#: The ceiling in force: the default, or the value `LAB_PROCESS_CEILING` sets.
CEILING = ceiling_from_env(CEILING)


def is_python_image(name: str) -> bool:
    """Whether a process image name or path names a Python interpreter."""
    base = re.split(r"[\\/]", name.strip())[-1]
    return PYTHON_IMAGE.match(base) is not None


def count_from_table(procs: dict[int, tuple[str, int]], mine: int) -> int:
    """The interpreters in a process table of pid to (image, parent pid).

    The process `mine` is left out, and so is its parent when the parent is an interpreter.
    """
    skip = {mine}
    parent = procs.get(mine, ("", 0))[1]
    if is_python_image(procs.get(parent, ("", 0))[0]):
        skip.add(parent)
    return sum(
        1 for pid, (exe, _) in procs.items() if pid not in skip and is_python_image(exe)
    )


def windows_table() -> dict[int, tuple[str, int]]:
    """The Windows process table from the Toolhelp32 snapshot; empty on another platform."""
    if sys.platform != "win32":
        return {}
    import ctypes  # noqa: PLC0415 - read only on Windows
    from ctypes import wintypes  # noqa: PLC0415 - wintypes is Windows-only

    class PE(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t),
            ("mod", wintypes.DWORD),
            ("thr", wintypes.DWORD),
            ("ppid", wintypes.DWORD),
            ("pri", ctypes.c_long),
            ("flags", wintypes.DWORD),
            ("exe", ctypes.c_wchar * 260),
        ]

    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(2, 0)
    pe = PE()
    pe.dwSize = ctypes.sizeof(PE)
    procs: dict[int, tuple[str, int]] = {}
    ok = k.Process32FirstW(snap, ctypes.byref(pe))
    while ok:
        procs[pe.pid] = (pe.exe.lower(), pe.ppid)
        ok = k.Process32NextW(snap, ctypes.byref(pe))
    k.CloseHandle(snap)
    return procs


def parse_ps(text: str) -> dict[int, tuple[str, int]]:
    """The process table from the output of `ps -A -o pid= -o ppid= -o comm=`."""
    procs: dict[int, tuple[str, int]] = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs[int(parts[0])] = (parts[2].strip(), int(parts[1]))
    return procs


def posix_table() -> dict[int, tuple[str, int]] | None:
    """The POSIX process table from `ps`, or None when `ps` fails or is absent."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "comm="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return parse_ps(out.stdout)


def python_count() -> int | None:
    """The Python interpreter processes now running, less this process and a Python parent.

    None when the process table cannot be read.
    """
    if os.name == "nt":
        procs: dict[int, tuple[str, int]] | None = windows_table()
    else:
        procs = posix_table()
    if procs is None:
        return None
    return count_from_table(procs, os.getpid())
