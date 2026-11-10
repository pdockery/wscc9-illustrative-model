"""Tests for .githooks/commit-msg and .githooks/pre-commit, and for the two hook manifests.

Run: python -m pytest tools/tests/test_githooks.py -q

The hooks run against canned commits in a temporary repository whose core.hooksPath
points at this repository's .githooks folder. The manifest test hashes each listed
file's bytes with CRLF folded to LF, which equals the stored blob under .gitattributes.
"""

import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".githooks"

TRAILERS = [
    "Co-Authored-By: Claude <noreply@anthropic.com>",
    "Generated with [Claude Code](https://claude.com/claude-code)",
    "Contact: someone@noreply@anthropic.com",
]

HAVE_GIT_BASH = shutil.which("git") is not None and shutil.which("bash") is not None


def git(cwd, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check
    )


def make_repo(tmp):
    git(tmp, "init", "-q")
    git(tmp, "config", "user.name", "Test")
    git(tmp, "config", "user.email", "test@example.org")
    git(tmp, "config", "core.autocrlf", "false")
    git(tmp, "config", "core.hooksPath", HOOKS.as_posix())


def write(path, data=b"x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@unittest.skipUnless(HAVE_GIT_BASH, "git and bash are required")
class CommitMsgTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        make_repo(self.tmp)
        write(self.tmp / "a.txt")
        git(self.tmp, "add", "--", "a.txt")

    def test_clean_message_passes(self):
        r = git(self.tmp, "commit", "-m", "Add a file", check=False)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_each_trailer_is_refused(self):
        for trailer in TRAILERS:
            with self.subTest(trailer=trailer):
                r = git(
                    self.tmp, "commit", "-m", f"Add a file\n\n{trailer}", check=False
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("this repository", r.stderr)

    def test_message_names_no_private_repository(self):
        r = git(self.tmp, "commit", "-m", "x\n\n" + TRAILERS[0], check=False)
        for word in ("public clone", "CLAUDE.md", "user instruction"):
            self.assertNotIn(word, r.stderr)


@unittest.skipUnless(HAVE_GIT_BASH, "git and bash are required")
class PreCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        make_repo(self.tmp)

    def commit(self):
        return git(self.tmp, "commit", "-q", "-m", "test", check=False)

    def test_ordinary_files_pass(self):
        write(self.tmp / "lib" / "m.py")
        write(self.tmp / "sources" / "institutional-context" / "doc.pdf")
        git(self.tmp, "add", "--", "lib/m.py", "sources/institutional-context/doc.pdf")
        r = self.commit()
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_academic_literature_is_refused(self):
        p = "sources/academic-literature/paper.pdf"
        write(self.tmp / p)
        git(self.tmp, "add", "-f", "--", p)
        r = self.commit()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(p, r.stderr)
        self.assertIn("link-only", r.stderr)

    def test_ignored_sources_path_is_refused_on_forced_add(self):
        write(self.tmp / "sources" / ".gitignore", b"held.pdf\n")
        write(self.tmp / "sources" / "held.pdf")
        git(self.tmp, "add", "-f", "--", "sources/held.pdf", "sources/.gitignore")
        r = self.commit()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sources/held.pdf", r.stderr)
        self.assertNotIn("sources/.gitignore is", r.stderr)

    def test_oversize_file_is_refused_with_public_fix_line(self):
        write(self.tmp / "big.ipynb", b"0" * (51 * 1024 * 1024))
        git(self.tmp, "add", "--", "big.ipynb")
        r = self.commit()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("over the 50 MB limit", r.stderr)
        self.assertIn("Re-execute the notebook", r.stderr)

    def test_no_private_rule_text(self):
        text = (HOOKS / "pre-commit").read_text(encoding="utf-8")
        for word in ("responses", "shrink_notebook_figures"):
            self.assertNotIn(word, text)


class AttributesTests(unittest.TestCase):
    def test_required_rules_present(self):
        lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        rules = {ln for ln in lines if ln and not ln.startswith("#")}
        for rule in (
            "*.md text eol=lf",
            ".claude/** text eol=lf",
            "lib/** text eol=lf",
            "tests/** text eol=lf",
            "tools/** text eol=lf",
            ".githooks/* text eol=lf",
            "*.png binary",
            "*.pdf binary",
        ):
            self.assertIn(rule, rules)

    def test_hooks_have_lf_endings(self):
        for name in ("commit-msg", "pre-commit"):
            self.assertNotIn(b"\r", (HOOKS / name).read_bytes(), name)


class ManifestTests(unittest.TestCase):
    ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$")

    def check(self, folder):
        rows = []
        for line in (folder / "MANIFEST.md").read_text(encoding="utf-8").splitlines():
            m = self.ROW.match(line)
            if m:
                rows.append(m.groups())
        self.assertTrue(rows, f"no rows in {folder}/MANIFEST.md")
        for name, digest in rows:
            data = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(
                hashlib.sha256(data).hexdigest(), digest, f"{folder.name}/{name}"
            )

    def test_lib_manifest_matches_lf_bytes(self):
        self.check(ROOT / "lib")

    def test_hooks_manifest_matches_lf_bytes(self):
        self.check(ROOT / ".claude" / "hooks")


if __name__ == "__main__":
    unittest.main()
