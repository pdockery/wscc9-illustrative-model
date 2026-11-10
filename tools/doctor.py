# SPDX-FileCopyrightText: Market Design Lab contributors
# SPDX-License-Identifier: MIT
"""The machine's readiness to run the suites and the notebooks, reported and never changed.

Usage: ``python tools/doctor.py [--json] [--out PATH]``

The report covers the interpreter it examines (``LAB_PY`` where it is set, the running one
otherwise), each package of ``tools/env_lock.json`` against its pin, the notebook tooling
and the kernelspecs the root notebooks name, each ``lib/`` module and each copied hook
against its manifest, the data files of ``tools/required_data.txt`` with the tests that need
them, the count of Python processes against the ceiling of ``.claude/hooks/_limits.py``, and
git's ``core.hooksPath``. The examined interpreter is probed in a subprocess, so the script
runs under any Python 3.10 or later, the bare ``python`` that runs the hooks included.

The Python release and ``core.hooksPath`` are information and never a problem. The process
count is a problem only at the ceiling. A manifest hash is the SHA-256 of the file with its
line endings brought to LF, the form git stores under ``.gitattributes``.

Output: a text report on stdout (or the JSON with ``--json``), and the JSON written to
``tools/.cache/doctor.json``, which ``.gitignore`` ignores. The JSON's keys are ``ok``,
``written``, ``interpreter``, ``examined``, ``packages``, ``notebook``, ``manifests``,
``data``, ``processes``, ``hooks_path`` and ``problems``. ``ok`` is true when ``problems``
is empty.

Exit status: 0 when no problem is found, 1 when one is, 2 when the script itself fails.
Standard library only (CODE_STYLE.md 13.3). The script writes nothing but its cache file.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import _common as C  # noqa: E402
import lab_env  # noqa: E402

#: The ceiling on Python interpreter processes when .claude/hooks/_limits.py does not state one.
DEFAULT_CEILING = 7
#: The hooks' shared module that holds the ceiling and the count the guard compares with it.
LIMITS = ROOT / ".claude" / "hooks" / "_limits.py"
#: The fallback's image rule, the same as ``PYTHON_IMAGE`` in ``LIMITS``: python, python3,
#: python3.12, pythonw, each with or without ``.exe``.
FALLBACK_IMAGE = re.compile(r"^pythonw?(\d+(\.\d+)?)?(\.exe)?$", re.I)
#: The cache file.
CACHE = ROOT / "tools" / ".cache" / "doctor.json"
#: Seconds allowed to the probe of the examined interpreter.
PROBE_TIMEOUT = 120
#: The notebook tooling the headless re-run needs (tools/nb_execute.py).
NOTEBOOK_PACKAGES = ("nbformat", "nbconvert", "nbclient", "ipykernel")
#: The tracked git hooks folder; checked only where it exists.
HOOKS_DIR = ".githooks"
#: The manifests: the folder each records, its file, and whether every ``*.py`` file of the
#: folder must appear in it. ``lib/`` is a copy as a whole; ``.claude/hooks/`` also holds
#: hooks of this repository's own, which its manifest does not list.
MANIFESTS = (
    ("lib", "lib/MANIFEST.md", True),
    (".claude/hooks", ".claude/hooks/MANIFEST.md", False),
)
#: A manifest table row: a backquoted file name and a backquoted 64-digit SHA-256.
MANIFEST_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", re.M)

#: The probe the examined interpreter runs. It reads versions and kernelspecs, and prints
#: one line that starts with the marker.
PROBE = r"""
import json, sys
from importlib import metadata
names = json.loads(sys.argv[1])
def ver(n):
    try:
        return metadata.version(n)
    except metadata.PackageNotFoundError:
        return None
out = {"executable": sys.executable, "version": sys.version.split()[0],
       "packages": {n: ver(n) for n in names}}
try:
    from jupyter_client.kernelspec import KernelSpecManager
    specs = KernelSpecManager().get_all_specs()
    out["kernelspecs"] = {k: {"argv0": ((v.get("spec") or {}).get("argv") or [None])[0]}
                          for k, v in specs.items()}
except Exception as e:
    out["kernelspecs_error"] = repr(e)
print("DOCTOR_PROBE " + json.dumps(out))
"""
PROBE_MARK = "DOCTOR_PROBE "


@functools.cache
def _load_limits(path: Path) -> Any | None:
    try:
        spec = importlib.util.spec_from_file_location("_doctor_hook_limits", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - any failure of the hooks module falls back to the reader here
        return None
    if not all(
        callable(getattr(mod, f, None)) for f in ("python_count", "count_from_table")
    ):
        return None
    return mod


def load_limits(path: Path = LIMITS) -> Any | None:
    """The hooks' ``_limits`` module loaded from its file, or None when it cannot load.

    The module is loaded once per resolved path, under a name of its own, and is not entered
    in ``sys.modules``. It counts only when it defines ``python_count`` and
    ``count_from_table``.
    """
    return _load_limits(Path(path).resolve())


def ceiling(root: Path = ROOT) -> int:
    """The process ceiling, as the hooks' ``_limits.py`` states it.

    The loaded module's ``CEILING`` stands first, so an override the module reads from the
    environment applies here too. Where the module does not load, ``CEILING = <int>`` is
    read from the parsed file. ``DEFAULT_CEILING`` stands where both fail.
    """
    p = root / ".claude" / "hooks" / "_limits.py"
    mod = load_limits(p) if p.is_file() else None
    value = getattr(mod, "CEILING", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return DEFAULT_CEILING
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CEILING" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    return DEFAULT_CEILING


def count_table(procs: dict[int, tuple[str, int]], mine: int) -> int:
    """The interpreters in a table of pid to (image, parent pid), by the guard's rule.

    The process ``mine`` is left out, and so is its parent when the parent is an interpreter.
    The rule is ``count_from_table`` of ``LIMITS``, kept here for the fallback.
    """

    def is_python(name: str) -> bool:
        return FALLBACK_IMAGE.match(re.split(r"[\\/]", name.strip())[-1]) is not None

    skip = {mine}
    parent = procs.get(mine, ("", 0))[1]
    if is_python(procs.get(parent, ("", 0))[0]):
        skip.add(parent)
    return sum(
        1 for pid, (exe, _) in procs.items() if pid not in skip and is_python(exe)
    )


def parse_tasklist(text: str) -> dict[int, tuple[str, int]]:
    """The table from ``tasklist /FO CSV /NH``, whose rows carry no parent (set to 0)."""
    procs: dict[int, tuple[str, int]] = {}
    for line in text.splitlines():
        cells = [c.strip('"') for c in line.strip().split('","')]
        if len(cells) > 1 and cells[1].isdigit():
            procs[int(cells[1])] = (cells[0], 0)
    return procs


def parse_ps(text: str) -> dict[int, tuple[str, int]]:
    """The table from ``ps -A -o pid= -o ppid= -o comm=``."""
    procs: dict[int, tuple[str, int]] = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs[int(parts[0])] = (parts[2].strip(), int(parts[1]))
    return procs


def fallback_count() -> int | None:
    """The count from ``tasklist`` on Windows or ``ps`` elsewhere, through ``count_table``.

    ``tasklist`` gives no parent, so on Windows a Python parent is counted, and the count can
    be one above the guard's. None when the command cannot run.
    """
    if os.name == "nt":
        argv, parse = ["tasklist", "/FO", "CSV", "/NH"], parse_tasklist
    else:
        argv, parse = ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "comm="], parse_ps
    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return count_table(parse(out or ""), os.getpid())


def python_count(limits_path: Path = LIMITS) -> int | None:
    """The Python interpreter processes now running, as the guard counts them; None if unknown.

    The count is ``_limits.python_count`` of the hooks, so this report and the guard refuse
    at one count. Where the module cannot load or its count fails, ``fallback_count`` stands.
    A process is ended by its PID and never by its image name.
    """
    limits = load_limits(limits_path)
    if limits is None:
        return fallback_count()
    try:
        return limits.python_count()
    except Exception:  # noqa: BLE001 - a failing count falls back to the reader
        return fallback_count()


def read_required_data(path: Path) -> list[dict[str, Any]]:
    """The entries of ``tools/required_data.txt``: path, kind and the tests that need it.

    A blank line or a ``#`` line is skipped. A line without three fields raises
    ``ValueError`` naming its number, because a malformed entry would hide an absent file.
    """
    entries = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(" | ")]
        if len(parts) != 3 or parts[1] not in ("local", "tracked") or not parts[0]:
            raise ValueError(
                f"{path.name} line {i}: expected 'path | local|tracked | tests'"
            )
        tests = [t.strip() for t in parts[2].split(", ") if t.strip()]
        entries.append({"path": parts[0], "kind": parts[1], "tests": tests})
    return entries


def data_status(root: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each entry with ``present`` set: a glob matches, or the path exists as a file or folder."""
    out = []
    for e in entries:
        if any(c in e["path"] for c in "*?["):
            present = next(iter(root.glob(e["path"])), None) is not None
        else:
            present = (root / e["path"]).exists()
        out.append({**e, "present": present})
    return out


def lf_sha256(path: Path) -> str:
    """The SHA-256 of a file with CRLF line endings brought to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest_status(
    root: Path, folder: str, manifest: str, complete: bool
) -> dict[str, Any]:
    """One manifest compared with its folder: each listed file's hash, and the unlisted ones.

    Parameters
    ----------
    root : Path
        The repository root.
    folder : str
        The folder the manifest records, repository-relative.
    manifest : str
        The manifest file, repository-relative.
    complete : bool
        Whether every ``*.py`` file of the folder must be listed.

    Returns
    -------
    dict
        ``manifest`` (the path, or None when absent), ``files`` (name to ``"match"``,
        ``"differs"``, ``"crlf"`` or ``"absent"``) and ``unlisted`` (the ``*.py`` files of a
        complete folder that the manifest does not list). ``"crlf"`` marks a hash that matches
        the file's CRLF bytes and not its LF form.

    """
    mp = root / manifest
    if not mp.is_file():
        return {"manifest": None, "files": {}, "unlisted": []}
    rows = dict(MANIFEST_ROW.findall(mp.read_text(encoding="utf-8")))
    files: dict[str, str] = {}
    for name, want in sorted(rows.items()):
        p = root / folder / name
        if not p.is_file():
            files[name] = "absent"
        elif lf_sha256(p) == want:
            files[name] = "match"
        elif hashlib.sha256(p.read_bytes()).hexdigest() == want:
            files[name] = "crlf"
        else:
            files[name] = "differs"
    unlisted = (
        sorted(p.name for p in (root / folder).glob("*.py") if p.name not in rows)
        if complete
        else []
    )
    return {"manifest": manifest, "files": files, "unlisted": unlisted}


def notebook_kernels(root: Path) -> dict[str, str | None]:
    """The kernelspec name each root notebook's metadata names, None where it names none."""
    out: dict[str, str | None] = {}
    for nb in sorted(root.glob(C.NOTEBOOK_GLOB)):
        try:
            with nb.open(encoding="utf-8") as f:
                meta = json.load(f).get("metadata") or {}
        except (OSError, ValueError):
            out[nb.name] = None
            continue
        out[nb.name] = (meta.get("kernelspec") or {}).get("name")
    return out


def probe(python: Path, names: list[str], root: Path = ROOT) -> dict[str, Any]:
    """The examined interpreter's own report, from one subprocess that runs ``PROBE``.

    The result carries ``error`` when the subprocess fails, times out or prints no marker.
    """
    try:
        r = subprocess.run(
            [str(python), "-I", "-c", PROBE, json.dumps(names)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT,
            cwd=str(root),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"the probe did not run: {exc!r}"}
    for line in r.stdout.splitlines():
        if line.startswith(PROBE_MARK):
            try:
                return json.loads(line[len(PROBE_MARK) :])
            except ValueError:
                break
    tail = (r.stderr or "").strip().splitlines()[-1:] or [f"exit {r.returncode}"]
    return {"error": f"the probe printed no report: {tail[0]}"}


def hooks_path(root: Path) -> dict[str, Any]:
    """The value of ``core.hooksPath`` in git, and the tracked hooks folder where it exists."""
    value = C.git(str(root), "config", "--get", "core.hooksPath").strip()
    expected = HOOKS_DIR if (root / HOOKS_DIR).is_dir() else None
    return {"value": value or None, "expected": expected}


def _package_problems(
    root: Path, rep: dict[str, Any], lock: dict[str, Any], report: dict[str, Any]
) -> list[str]:
    """Set ``packages`` and ``notebook`` in the report, and return their problems.

    A package off its pin adds one closing problem that names the install of the lock's
    versions.
    """
    problems: list[str] = []
    probed = bool(report) and not report.get("error")
    have = report.get("packages") or {}
    packages = {}
    for dist, spec in sorted((lock.get("packages") or {}).items()):
        want, got = spec.get("version"), have.get(dist)
        packages[dist] = {"pinned": want, "installed": got, "role": spec.get("role")}
        if probed and got != want:
            problems.append(f"{dist} {got or 'absent'} (pinned {want})")
    if problems:
        problems.append(f"install the recorded versions with {lab_env.INSTALL_LINE}")
    rep["packages"] = packages

    specs = report.get("kernelspecs")
    kernels = notebook_kernels(root)
    rep["notebook"] = {
        "packages": {n: have.get(n) for n in NOTEBOOK_PACKAGES},
        "kernelspecs": specs,
        "kernels_named": kernels,
        "error": report.get("kernelspecs_error"),
    }
    if probed:
        missing = [n for n in NOTEBOOK_PACKAGES if not have.get(n)]
        if missing:
            problems.append(f"notebook tooling absent: {', '.join(missing)}")
        for name in sorted({k for k in kernels.values() if k}):
            if name not in (specs or {}):
                users = ", ".join(nb for nb, k in kernels.items() if k == name)
                problems.append(
                    f"no Jupyter kernelspec {name!r} is installed; it is named by {users}"
                )
    return problems


def diagnose(
    root: Path = ROOT,
    *,
    lock: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    processes: int | None = None,
) -> dict[str, Any]:
    """The full report as a mapping, with ``problems`` listed and ``ok`` set.

    ``lock``, ``report`` (the probe's result) and ``processes`` replace what the function
    would read or run, which the tests use. A lock that is absent is itself a problem.
    """
    problems: list[str] = []
    if lock is None:
        lock = lab_env.load_lock(root / "tools" / "env_lock.json") or {}
    if not lock:
        problems.append("tools/env_lock.json is absent, so no package is compared")
    rep: dict[str, Any] = {
        "ok": False,
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interpreter": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
    }

    exe = lab_env.lab_interpreter()
    if report is None:
        names = sorted(set(lock.get("packages") or {}) | set(NOTEBOOK_PACKAGES))
        report = probe(exe, names, root)
    rep["examined"] = {
        "path": report.get("executable") or str(exe),
        "from": lab_env.LAB_PY_VAR
        if os.environ.get(lab_env.LAB_PY_VAR)
        else "the running interpreter",
        "version": report.get("version"),
        "tested": lock.get("python"),
    }
    if report.get("error"):
        problems.append(f"the examined interpreter {exe}: {report['error']}")
    problems.extend(_package_problems(root, rep, lock, report))

    manifests = {}
    for folder, manifest, complete in MANIFESTS:
        st = manifest_status(root, folder, manifest, complete)
        manifests[folder] = st
        if st["manifest"] is None:
            if (root / folder).is_dir():
                problems.append(f"{manifest} is absent")
            continue
        for name, state in st["files"].items():
            if state == "crlf":
                problems.append(
                    f"{manifest} hashes {folder}/{name} with CRLF line endings; "
                    "regenerate it from the LF form"
                )
            elif state != "match":
                problems.append(f"{folder}/{name} {state} from {manifest}")
        problems.extend(
            f"{folder}/{name} is not listed in {manifest}" for name in st["unlisted"]
        )
    rep["manifests"] = manifests

    req = root / "tools" / "required_data.txt"
    try:
        data = data_status(root, read_required_data(req)) if req.is_file() else []
    except (OSError, ValueError) as exc:
        data = []
        problems.append(f"tools/required_data.txt: {exc}")
    rep["data"] = data
    problems.extend(
        f"{d['path']} ({d['kind']}) is absent; these need it: {', '.join(d['tests'])}"
        for d in data
        if not d["present"]
    )

    n = python_count() if processes is None else processes
    cap = ceiling(root)
    rep["processes"] = {"count": n, "ceiling": cap}
    if n is not None and n >= cap:
        problems.append(
            f"{n} Python processes run, at the ceiling of {cap}; end orphaned ones by PID"
        )

    rep["hooks_path"] = hooks_path(root)
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def render(rep: dict[str, Any]) -> str:
    """The report as text, one line per check and the problems last."""
    ex = rep["examined"]
    lines = [
        f"interpreter     {rep['interpreter']['version']} at {rep['interpreter']['executable']}",
        f"examined        {ex.get('version') or '?'} at {ex.get('path')} ({ex.get('from')})",
        f"python          {ex.get('version') or '?'} (tested under {ex.get('tested') or '?'}, "
        "information only)",
    ]
    off = [k for k, v in rep["packages"].items() if v["installed"] != v["pinned"]]
    lines.append(
        f"packages        {len(rep['packages']) - len(off)} of {len(rep['packages'])} "
        "at the pinned version"
    )
    specs = rep["notebook"]["kernelspecs"] or {}
    lines.append(f"kernelspecs     {', '.join(sorted(specs)) or 'none'}")
    named = sorted({k for k in rep["notebook"]["kernels_named"].values() if k})
    lines.append(f"notebooks name  {', '.join(named) or 'no kernelspec'}")
    for folder, st in rep["manifests"].items():
        good = sum(1 for s in st["files"].values() if s == "match")
        lines.append(
            f"manifest        {st['manifest'] or folder + ' (none)'}: "
            f"{good} of {len(st['files'])} match"
        )
    absent = sum(1 for d in rep["data"] if not d["present"])
    lines.append(
        f"data            {len(rep['data']) - absent} of {len(rep['data'])} present"
    )
    pr = rep["processes"]
    lines.append(f"processes       {pr['count']} Python (ceiling {pr['ceiling']})")
    hp = rep["hooks_path"]
    note = f" (tracked folder {hp['expected']})" if hp["expected"] else ""
    lines.append(f"core.hooksPath  {hp['value'] or 'unset'}{note}")
    lines.append("ok" if rep["ok"] else f"{len(rep['problems'])} problems:")
    lines.extend(f"  - {p}" for p in rep["problems"])
    return "\n".join(lines)


def write_cache(rep: dict[str, Any], path: Path = CACHE) -> None:
    """The report written as JSON with LF endings, through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(rep, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    """The command line: diagnose, write the cache, print the report, return the status."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="print the JSON report")
    ap.add_argument("--out", type=Path, default=CACHE, help="the cache file to write")
    args = ap.parse_args(argv)
    try:
        rep = diagnose()
        write_cache(rep, args.out)
    except Exception as exc:  # noqa: BLE001 - report-only tool; its own failure is status 2
        print(f"doctor: failed: {exc!r}", file=sys.stderr)  # noqa: T201 - a report tool
        return 2
    with contextlib.suppress(ValueError, AttributeError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # a TextIOWrapper on every platform the repository runs on
    print(json.dumps(rep, indent=1) if args.json else render(rep))  # noqa: T201 - a report tool
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
