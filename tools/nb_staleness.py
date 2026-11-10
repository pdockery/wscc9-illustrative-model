#!/usr/bin/env python
"""Report which notebooks' stored results predate a library change.

Usage: python tools/nb_staleness.py [--json] [--root DIR] [--no-cache]

The notebooks are the *.ipynb files at the repository root, and the library is lib/, the shared
9-bus library; lib/MANIFEST.md records each module's SHA-256 and the date of the copy. A change to
lib/MANIFEST.md alone is a lib change that is not Python, so a copy event marks every stamped
notebook stale.

Classes per notebook:
  current    stamped, no lib file changed since the stamp's lib_commit, and the lib state matches the stamp
  stale      stamped, and lib files changed after lib_commit (named), or lib_dirty at execution, or a
             lib_commit this repository cannot resolve, whose lib state cannot be compared
  unstamped  no lab_execution stamp; an ESTIMATE compares the notebook's last commit time with the
             last commit that changed code in the lib directories

A changed lib file is one `git diff --name-only` lists or an untracked one, and it is a change only
when its code differs. A Python file whose runtime fingerprint at lib_commit equals its working-tree
fingerprint (tools/_common.py) is dropped from the list, because a reformat, a comment edit or a
type hint does not move a stored result. The row's detail counts the files dropped. A dropped file cannot clear a
stamp written with uncommitted lib changes, because that stamp's code is not the code at
lib_commit. The estimate applies the same test to each lib commit, newest first, and stops at the
first that changed code, whose row detail counts any later commit it passed over.

The stamp is read from the tail of the file (nbformat writes the sorted metadata block last), so
a 100 MB notebook costs one small read. A fingerprint is memoized by file content under
tools/.cache/fingerprints, so only the first run after a change parses the files. Every run
without --no-cache writes its summary to tools/.cache/staleness.json under the repository it
classified, through a temporary file and a rename, for a SessionStart line to read, and
nb_execute.py rewrites that summary after each stamp (refresh_cache). The summary holds
"generated" (the UTC time it was written), "counts" (the current, stale and unstamped counts)
and "head" (the commit classified).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

LOG_PAGE = 20  # lib commits read at a time, most of which a walk never reaches


def last_commit_times(root: str, rel_dir: str) -> dict[str, int]:
    """Map notebook path -> epoch of its last commit, in one `git log` pass."""
    out = C.git(root, "log", "--format=@%ct", "--name-only", "--", rel_dir)
    times, cur = {}, None  # type: ignore[var-annotated]  # times maps a path to an epoch
    for line in out.splitlines():
        if line.startswith("@"):
            cur = int(line[1:])
        elif line.strip() and cur is not None:
            times.setdefault(line.strip(), cur)
    return times


def last_code_commit(root: str, dirs: list[str]) -> tuple[int, list[int]]:
    """The epoch of the newest commit that changed code under dirs, 0 if none, and the epochs passed over.

    A merge lists no files, so it stops the walk, which errs toward the estimate saying stale. A root
    commit's files have no parent version, so they count as changed. The log is read one page at a time,
    because the walk stops early and the whole history under lib costs more than a run allows.
    """
    passed: list[int] = []
    skip = 0
    while True:
        out = C.git(
            root,
            "log",
            "--format=@%H %ct",
            "--name-only",
            "-n",
            str(LOG_PAGE),
            "--skip",
            str(skip),
            "--",
            *dirs,
        )
        commits: list[tuple[str, int, list[str]]] = []
        for line in out.splitlines():
            if line.startswith("@"):
                sha, when = line[1:].split()
                commits.append((sha, int(when), []))
            elif line.strip() and commits:
                commits[-1][2].append(line.strip())
        for sha, when, files in commits:  # type: ignore[assignment]  # when is str here and int in the loop below
            if not files or len(
                C.same_code_between(root, sha + "^", sha, files, runtime=True)
            ) < len(files):
                return when, passed  # type: ignore[return-value]  # when is typed str by the split above
            passed.append(when)  # type: ignore[arg-type]  # when is typed str by the split above
        if len(commits) < LOG_PAGE:
            return 0, passed
        skip += LOG_PAGE


def same_code_note(dropped: int) -> str:
    """The clause of a row's detail that counts the lib files dropped as format, comment or hint edits."""
    if not dropped:
        return ""
    verb = "file differs" if dropped == 1 else "files differ"
    return f"; {dropped} {verb} in format, comments or hints only"


def later_note(later: int) -> str:
    """The clause of an estimate's detail that counts the later lib commits with no code change."""
    if not later:
        return ""
    verb = "commit changes" if later == 1 else "commits change"
    return f"; {later} later lib {verb} format, comments or hints only"


def classify(root: str) -> list[dict[str, Any]]:
    """One row per root notebook: its path, its class and the detail behind the class."""
    dirs = C.lib_dirs()
    nbs = sorted(str(p) for p in Path(root).glob(C.NOTEBOOK_GLOB))
    _, now_hash = C.lib_state(root, dirs)
    head = C.git(root, "rev-parse", "HEAD").strip()
    untracked = [
        x
        for x in C.git(
            root, "ls-files", "--others", "--exclude-standard", "--", *dirs
        ).splitlines()
        if x.strip()
    ]
    lib_last, passed = None, []  # type: ignore[var-annotated]  # lib_last is an epoch or None, passed a list of epochs
    nb_times = None
    changed_cache: dict[str, tuple[list[str], int, bool]] = {}
    rows = []
    for nb in nbs:
        rel = os.path.relpath(nb, root).replace("\\", "/")
        row: dict[str, Any] = {"notebook": rel}
        stamp = C.read_stamp(nb)
        if not stamp:
            if nb_times is None:
                # under git's glob magic `*` does not cross `/`, so this is the root notebooks alone
                nb_times = last_commit_times(root, ":(glob)" + C.NOTEBOOK_GLOB)
                lib_last, passed = last_code_commit(root, dirs)
            t = nb_times.get(rel)
            row.update(
                status="unstamped",
                estimate=True,
                detail=(
                    "notebook last committed before the last lib code commit"
                    if t is not None and t < lib_last  # type: ignore[operator]  # lib_last is None until the first unstamped notebook sets it
                    else "notebook committed after the last lib code commit"
                    if t is not None
                    else "notebook not committed"
                )
                + later_note(sum(t is None or t < x for x in passed)),
                estimated_stale=(t is None or t < lib_last),  # type: ignore[operator]  # lib_last is None until the first unstamped notebook sets it
            )
        else:
            c = stamp.get("lib_commit", "")
            if c not in changed_cache:
                resolved = (
                    C.git(
                        root, "rev-parse", "--verify", "--quiet", c + "^{commit}"
                    ).strip()
                    if c
                    else ""
                )
                out = (
                    C.git(root, "diff", "--name-only", c, "--", *dirs)
                    if resolved
                    else ""
                )
                listed = [x for x in out.splitlines() if x.strip()] + untracked
                same = C.same_code(root, c, listed, runtime=True) if resolved else set()
                changed_cache[c] = (
                    [x for x in listed if x not in same],
                    len(same),
                    bool(resolved),
                )
            changed, dropped, resolved = changed_cache[c]  # type: ignore[assignment]  # resolved is a str above and a bool here
            note = same_code_note(dropped)
            row["stamp_date"] = stamp.get("date")
            if not resolved:
                row.update(
                    status="stale",
                    detail=f"lib_commit {c[:8] or '(none)'} does not resolve here, so the lib "
                    "state cannot be compared",
                    changed_files=[],
                )
            elif (
                stamp.get("lib_dirty")
                and stamp.get("lib_diff_sha256") == now_hash
                and c == head
            ):
                row.update(
                    status="current",
                    detail="executed with uncommitted lib changes, unchanged since",
                )
            elif changed:
                row.update(
                    status="stale",
                    detail=f"lib changed since {c[:8]}{note}",
                    changed_files=changed,
                )
            elif stamp.get("lib_dirty"):
                row.update(
                    status="stale",
                    detail=f"executed with uncommitted lib changes{note}",
                    changed_files=[],
                )
            else:
                row.update(
                    status="current", detail=f"no lib change since {c[:8]}{note}"
                )
        rows.append(row)
    return rows


def cache_path(root: str) -> Path:
    """The staleness summary of the repository at root, under its ignored tools/.cache."""
    return C.cache_file(root, "staleness.json")


def counts_of(rows: list[dict[str, Any]]) -> dict[str, int]:
    """The number of rows in each class, current, stale and unstamped."""
    return {
        s: sum(1 for r in rows if r["status"] == s)
        for s in ("current", "stale", "unstamped")
    }


def write_cache(root: str, rows: list[dict[str, Any]]) -> Path:
    """Write the summary of a full run to cache_path(root), atomically, as LF text, and return the path.

    The keys "generated" and "counts" keep the flat form the SessionStart brief reads. The key
    "head" was added on 2026-11-10.
    """
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts_of(rows),
        "head": C.git(root, "rev-parse", "HEAD").strip(),
    }
    path = cache_path(root)
    C.write_atomic(path, (json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    return path


def refresh_cache(root: str) -> Path:
    """Classify every root notebook of the repository at root and rewrite its summary."""
    return write_cache(root, classify(root))


def main(argv: list[str] | None = None) -> int:
    """Print the report, or the rows as JSON with --json, write the summary, and return 0."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = (
        os.path.normpath(a.root)
        if a.root
        else C.repo_root(os.path.dirname(os.path.abspath(__file__)))
    )
    rows = classify(root)
    elapsed = time.time() - t0
    counts = counts_of(rows)
    if a.json:
        print(json.dumps({"root": rows}, indent=1))
    else:
        shown = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(f"notebooks: {shown or 'none'}")
        for r in rows:
            if r["status"] == "current":
                continue
            extra = ""
            if r["status"] == "stale" and r.get("changed_files"):
                extra = " [%s]" % ", ".join(
                    os.path.basename(x) for x in r["changed_files"][:6]
                )
            if r["status"] == "unstamped":
                extra = " (estimate: %s)" % (
                    "stale" if r["estimated_stale"] else "current"
                )
            print(
                "  %-9s %s: %s%s"
                % (r["status"], os.path.basename(r["notebook"]), r["detail"], extra)
            )
        print("runtime %.2f s" % elapsed, file=sys.stderr)
    if not a.no_cache:
        try:
            write_cache(root, rows)
        except OSError as e:
            print(f"nb_staleness: summary not cached ({e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
