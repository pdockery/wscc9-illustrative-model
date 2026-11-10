"""Tests for .claude/hooks/guard.py and its shared modules _limits.py and _log.py.

Standard library unittest. Run from the repository root:

    python -m pytest .claude/hooks/tests/test_guard.py -q

The three hook files are copies, listed with their hashes in .claude/hooks/MANIFEST.md, and
this test belongs to this repository. The must-block and must-allow tables read each command
through `guard.decide` with the process count fixed. Each runs in a temporary folder that
holds three trees with a `.git` folder each: a public tree with the marker `lib/MANIFEST.md`,
a plain tree without it, and a tree whose `origin` belongs to another owner. No file is
written into the working tree. The rules file is replaced by `PUBLIC_RULES`, this
repository's facts, so a table does not depend on the file on disk; the file itself is read
by its own test. The decision log is replaced by a mock where a test reads its calls, and
every other write of `_log.record` goes to a temporary folder, so no test writes
`.claude/logs/`. The tests of `.githooks/` are in tools/tests/test_githooks.py.
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HOOKS = Path(__file__).resolve().parents[1]
REPO = HOOKS.parents[1]
sys.path.insert(0, str(HOOKS))
import _limits  # noqa: E402
import _log  # noqa: E402
import guard  # noqa: E402

#: This repository's facts, as the tables expect them and as repo_rules.json states them.
PUBLIC_RULES = guard.RepoRules(
    repository="public",
    commit_hook_rules="the trailer, file-size and sources rules",
    allowed_owners=["pdockery"],
)

_LOG_TMP = Path(tempfile.mkdtemp())
_LOG_PATCH = mock.patch.object(_log, "LOG_DIR", _LOG_TMP / "logs")


def setUpModule() -> None:
    """Send every unmocked log write of the module to a temporary folder."""
    _LOG_PATCH.start()


def tearDownModule() -> None:
    """Restore the log folder and remove the temporary one."""
    _LOG_PATCH.stop()
    shutil.rmtree(_LOG_TMP, True)


def make_tree(
    root: Path, name: str, owner: str = "someone", public: bool = False
) -> Path:
    """A folder that reads as a git working tree, with an origin remote of `owner`.

    A public tree holds the marker `lib/MANIFEST.md`.
    """
    top = root / name
    (top / ".git").mkdir(parents=True)
    (top / ".git" / "config").write_bytes(
        f'[remote "origin"]\n\turl = https://github.com/{owner}/{name}.git\n'.encode()
    )
    if public:
        (top / "lib").mkdir()
        (top / "lib" / "MANIFEST.md").write_bytes(b"# manifest\n")
    return top


class Trees(unittest.TestCase):
    """A public, a plain and an upstream tree in one temporary folder."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pub = make_tree(self.tmp / "public", "model", public=True)
        self.plain = make_tree(self.tmp / "plain", "work")
        self.upstream = make_tree(self.tmp / "ref", "toolbox", owner="otherorg")
        self.msg_bad = self.pub / "msg_bad.txt"
        self.msg_bad.write_bytes(
            b"fix\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
        )
        self.msg_ok = self.pub / "msg_ok.txt"
        self.msg_ok.write_bytes(b"fix the guard\n")
        rec = mock.patch.object(guard, "record")
        self.record = rec.start()
        self.addCleanup(rec.stop)
        rules = mock.patch.object(guard, "RULES", PUBLIC_RULES)
        rules.start()
        self.addCleanup(rules.stop)

    def where(self, key: str) -> str:
        return str({"pub": self.pub, "plain": self.plain, "up": self.upstream}[key])

    def fmt(self, cmd: str) -> str:
        return cmd.format(
            pub=self.pub, plain=self.plain, bad=self.msg_bad, ok=self.msg_ok
        )

    def decide(
        self, where: str, cmd: str, count: int | None = 0, background: bool = False
    ) -> guard.Context:
        data = {
            "tool_input": {"command": self.fmt(cmd), "run_in_background": background},
            "cwd": self.where(where),
        }
        with mock.patch.object(guard, "python_count", return_value=count):
            return guard.decide(data)


#: (where, command, rule) that must be blocked, with the process count at zero.
MUST_BLOCK = [
    # 1. broad add in the public tree, direct and through every bypass form
    ("pub", "git add -A", "public-broad-add"),
    ("pub", "git add --all", "public-broad-add"),
    ("pub", "git add .", "public-broad-add"),
    ("pub", "git add ./", "public-broad-add"),
    ("pub", "git add -- .", "public-broad-add"),
    ("pub", 'git add "."', "public-broad-add"),
    ("pub", "git add -vA", "public-broad-add"),
    ("pub", "git add -fA", "public-broad-add"),
    ("pub", "git add '*.py'", "public-broad-add"),
    ("pub", "git add lib/*.py", "public-broad-add"),
    ("pub", "git add :/", "public-broad-add"),
    ("pub", "git add {pub}", "public-broad-add"),
    ("pub", "git status; git add --all", "public-broad-add"),
    ("pub", "( cd lib; git add . )", "public-broad-add"),
    # a subshell, a brace group, a script block and a substitution under any word
    ("pub", "(cd lib; git add .)", "public-broad-add"),
    ("pub", "(git add .)", "public-broad-add"),
    ("pub", "(git status | git add -A)", "public-broad-add"),
    ("pub", "{{ git add .; }}", "public-broad-add"),
    ("pub", "git status; {{ cd lib; git add .; }}", "public-broad-add"),
    ("pub", "& {{ git add . }}", "public-broad-add"),
    ("pub", "1 | ForEach-Object {{ git add -A }}", "public-broad-add"),
    ("pub", "echo $(git add -A)", "public-broad-add"),
    ("pub", 'echo "$(git add -A)"', "public-broad-add"),
    ("pub", "echo $(cd lib; git add .)", "public-broad-add"),
    ("pub", "Write-Output $(git add -A)", "public-broad-add"),
    ("pub", "echo `git add -A`", "public-broad-add"),
    ("pub", 'git commit -m "note `git add -A`"', "public-broad-add"),
    ("pub", "echo $(echo $(git add .))", "public-broad-add"),
    ("pub", "echo $(git add sources/academic-literature/a.pdf)", "public-sources"),
    ("pub", "echo $(MPLBACKEND=Agg python x.py)", "mplbackend-agg"),
    ("pub", "cd lib; git add .", "public-broad-add"),
    ("pub", "git add --pathspec-from-file=absent.txt", "public-broad-add"),
    ("pub", "git ls-files -o | git add --pathspec-from-file=-", "public-broad-add"),
    # 1. the sources: academic-literature never, and no forced add under sources/
    ("pub", "git add sources/academic-literature/a.pdf", "public-sources"),
    ("pub", "git add sources/Academic-Literature", "public-sources"),
    ("pub", "git add sources/x/../academic-literature/a.pdf", "public-sources"),
    ("pub", "git -C sources add academic-literature/a.pdf", "public-sources"),
    ("pub", "git add -f sources", "public-sources"),
    ("pub", "git add --force sources/workshops/deck.pdf", "public-sources"),
    ("pub", "git add -fv sources/references.bib", "public-sources"),
    # a pathspec magic is read as git reads it, and one the guard cannot know is broad
    ("pub", "git add -f :/sources/academic-literature/a.pdf", "public-sources"),
    (
        "pub",
        "git add -f ':(literal)sources/academic-literature/a.pdf'",
        "public-sources",
    ),
    (
        "pub",
        "git add ':(top,icase)Sources/Academic-Literature/a.pdf'",
        "public-sources",
    ),
    ("pub", "git add :/:sources/academic-literature/a.pdf", "public-sources"),
    ("pub", "git add -f :/sources/workshops/a.pdf", "public-sources"),
    ("pub", "git -C sources add -f :/sources/a.pdf", "public-sources"),
    ("pub", "git add ':!lib'", "public-broad-add"),
    ("pub", "git add ':^lib'", "public-broad-add"),
    ("pub", "git add ':(exclude)lib'", "public-broad-add"),
    ("pub", "git add ':(attr:x)lib'", "public-broad-add"),
    ("pub", "git add ':(top'", "public-broad-add"),
    ("pub", 'git add -f "$(echo sources/a.pdf)"', "public-broad-add"),
    ("pub", "git add $(git ls-files -o)", "public-broad-add"),
    ("pub", "git add `git ls-files -o`", "public-broad-add"),
    ("pub", "git add $P", "public-broad-add"),
    ("pub", "git add ${{P}}", "public-broad-add"),
    # piped text: escapes expanded, an unknown escape or a joining format is broad
    (
        "pub",
        "printf 'lib/x.py\\nsources/academic-literature/a.pdf\\n' "
        "| git add -f --pathspec-from-file=-",
        "public-sources",
    ),
    (
        "pub",
        "echo -e 'lib/x.py\\nsources/academic-literature/a.pdf' "
        "| git add --pathspec-from-file=-",
        "public-sources",
    ),
    (
        "pub",
        "printf '%s\\n' lib/x.py sources/academic-literature/a.pdf "
        "| git add --pathspec-from-file=-",
        "public-sources",
    ),
    (
        "pub",
        "printf '%s%s\\n' sources/academic- literature/a.pdf "
        "| git add --pathspec-from-file=-",
        "public-broad-add",
    ),
    (
        "pub",
        "printf 'sources/\\141cademic-literature/a.pdf' | git add --pathspec-from-file=-",
        "public-broad-add",
    ),
    (
        "pub",
        'Write-Output "lib/x.py`nsources/academic-literature/a.pdf" '
        "| git add --pathspec-from-file=-",
        "public-broad-add",
    ),
    ("pub", "echo $(ls) | git add --pathspec-from-file=-", "public-broad-add"),
    ("pub", "git add --pathspec-from-file=- <<EOF\n`ls`\nEOF", "public-broad-add"),
    # the wrappers and the git spellings
    ("pub", "git.exe add -A", "public-broad-add"),
    ("pub", "git --no-pager add -A", "public-broad-add"),
    ("pub", "git stage -A", "public-broad-add"),
    ("pub", 'bash -c "git add -A"', "public-broad-add"),
    ("pub", "sh -c 'git add .'", "public-broad-add"),
    ("pub", 'powershell -NoProfile -Command "git add -A"', "public-broad-add"),
    ("pub", 'pwsh -c "git add -A"', "public-broad-add"),
    ("pub", "cmd /c git add -A", "public-broad-add"),
    ("pub", "powershell -Command \"bash -c 'git add .'\"", "public-broad-add"),
    ("pub", 'echo "git add -A" | bash', "public-broad-add"),
    ("pub", "eval 'git add -A'", "public-broad-add"),
    ("pub", "bash <<'EOF'\ngit add -A\nEOF", "public-broad-add"),
    # the public tree reached from another folder
    ("plain", "git -C {pub} add -A", "public-broad-add"),
    ("plain", 'git -C "{pub}" -c core.quotepath=off add .', "public-broad-add"),
    ("plain", "git --work-tree={pub} add -A", "public-broad-add"),
    ("plain", "cd {pub} && git add -A", "public-broad-add"),
    ("plain", "Set-Location {pub}; git add -A", "public-broad-add"),
    ("plain", "git -C {pub} add sources/academic-literature/a.pdf", "public-sources"),
    ("plain", "git -C {pub} add -f sources/a.pdf", "public-sources"),
    # 2. MPLBACKEND=Agg
    ("pub", "MPLBACKEND=Agg python -m nbconvert --execute a.ipynb", "mplbackend-agg"),
    (
        "pub",
        "$env:MPLBACKEND='Agg'; python tools/nb_execute.py a.ipynb",
        "mplbackend-agg",
    ),
    ("pub", "export MPLBACKEND=Agg", "mplbackend-agg"),
    ("pub", "env MPLBACKEND=Agg jupyter nbconvert --execute a.ipynb", "mplbackend-agg"),
    # 3. the trailer, in -m, through -c and -C, in a heredoc, a here-string and a -F file
    (
        "pub",
        'git commit -m "x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"',
        "claude-trailer",
    ),
    (
        "pub",
        'git -c user.name=x commit -m "x\n\nCo-Authored-By: Claude Opus"',
        "claude-trailer",
    ),
    (
        "plain",
        "git -C {pub} commit -m 'fix\n\nGenerated with [Claude Code](https://claude.com)'",
        "claude-trailer",
    ),
    (
        "pub",
        "git commit -m \"$(cat <<'EOF'\nfix\n\nCo-Authored-By: Claude\nEOF\n)\"",
        "claude-trailer",
    ),
    (
        "pub",
        "git commit -m @'\nfix it's done\nCo-Authored-By: Claude\n'@",
        "claude-trailer",
    ),
    ("pub", "git commit -F {bad}", "claude-trailer"),
    ("pub", "git commit --file={bad}", "claude-trailer"),
    ("pub", "git commit -F msg_bad.txt", "claude-trailer"),
    ("pub", "cat {bad} | git commit -F -", "claude-trailer"),
    ("pub", "git commit --amend -m 'x noreply@anthropic.com'", "claude-trailer"),
    ("pub", "bash -c \"git commit -m 'Co-Authored-By: Claude'\"", "claude-trailer"),
    # 6. kill by image name
    ("pub", "taskkill /F /IM python.exe", "kill-by-name"),
    ("pub", "taskkill /im pythonw.exe /f", "kill-by-name"),
    ("pub", 'taskkill /F /FI "IMAGENAME eq python.exe"', "kill-by-name"),
    ("pub", "Stop-Process -Name python -Force", "kill-by-name"),
    ("pub", "Stop-Process -Name python,pythonw", "kill-by-name"),
    ("pub", "spps -Name python*", "kill-by-name"),
    ("pub", "Get-Process python | Stop-Process", "kill-by-name"),
    ("pub", "Get-Process -Name python* | Stop-Process -Force", "kill-by-name"),
    (
        "pub",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Invoke-CimMethod -MethodName Terminate",
        "kill-by-name",
    ),
    ("pub", "pkill python", "kill-by-name"),
    ("pub", "killall python3", "kill-by-name"),
    ("pub", "ps aux | grep python | awk '{{print $2}}' | xargs kill", "kill-by-name"),
    ("pub", 'powershell -Command "Stop-Process -Name python"', "kill-by-name"),
    # 7. a junction or a link that names data/
    ("pub", "cmd /c mklink /J wt\\model\\data C:\\main\\data", "junction-into-data"),
    ("pub", "mklink /D data ..\\main\\data", "junction-into-data"),
    (
        "pub",
        "New-Item -ItemType Junction -Path wt/data -Target C:/main/data",
        "junction-into-data",
    ),
    (
        "pub",
        "New-Item -Path wt\\data -ItemType SymbolicLink -Value C:\\main\\data",
        "junction-into-data",
    ),
    ("pub", "ni -it junction wt/model/data -Target ../data", "junction-into-data"),
    ("pub", "ln -s ../main/data data", "junction-into-data"),
    ("pub", "ln -sfn /x/data wt/data", "junction-into-data"),
    # 9. git clean of ignored files
    ("pub", "git clean -fdx", "clean-ignored"),
    ("pub", "git clean -xdf", "clean-ignored"),
    ("pub", "git clean -fX", "clean-ignored"),
    ("plain", "git -C {pub} clean -f -x", "clean-ignored"),
    # 10. posting to a repository of an owner not allowed
    ("pub", "gh issue create -R otherorg/toolbox -t x -b y", "upstream-post"),
    ("pub", "gh issue comment 12 --repo otherorg/toolbox -b x", "upstream-post"),
    ("pub", "gh pr create --repo=otherorg/toolbox -t x", "upstream-post"),
    (
        "pub",
        "gh pr comment https://github.com/otherorg/toolbox/pull/1 -b hi",
        "upstream-post",
    ),
    ("pub", "gh pr review 3 -R otherorg/toolbox --approve", "upstream-post"),
    ("pub", "gh api repos/otherorg/toolbox/issues -f title=x", "upstream-post"),
    (
        "pub",
        "gh api -X POST /repos/otherorg/toolbox/issues/5/comments",
        "upstream-post",
    ),
    (
        "pub",
        "gh api graphql -f query='mutation {{ addComment(input: {{}}) {{ clientMutationId }} }}'",
        "upstream-post",
    ),
    ("up", "gh issue create -t x -b y", "upstream-post"),
    # 11. wmic
    ("pub", "wmic process list brief", "wmic"),
    ("pub", "wmic.exe process where name='python.exe' get processid", "wmic"),
    # 12. a program fed to python on stdin that writes a notebook
    (
        "pub",
        "python - <<'EOF'\nimport nbformat\nnb = nbformat.read('a.ipynb', 4)\n"
        "nbformat.write(nb, 'a.ipynb')\nEOF",
        "heredoc-notebook",
    ),
    (
        "pub",
        "python <<EOF\nimport json\njson.dump(nb, open('a.ipynb', 'w'))\nEOF",
        "heredoc-notebook",
    ),
    (
        "pub",
        "cat <<'EOF' | python -\nimport nbformat\nnbformat.write(nb, f)\nEOF",
        "heredoc-notebook",
    ),
    (
        "pub",
        "@'\nfrom pathlib import Path\nPath('a.ipynb').write_text(s)\n'@ | python -",
        "heredoc-notebook",
    ),
    (
        "pub",
        "/opt/conda/envs/x/bin/python -X utf8 - <<'EOF'\n"
        "import nbformat\nnbformat.write(nb, 'a.ipynb')\nEOF",
        "heredoc-notebook",
    ),
]

#: (where, command) that must be allowed with the process count at nine, past the ceiling.
MUST_ALLOW = [
    # printing, searching or reading a rule's text is not a call
    ("pub", 'echo "git add -A"'),
    ("pub", "grep -rn 'git add -A' ."),
    ("pub", "cat <<'EOF'\ngit add -A\nEOF"),
    ("pub", 'rg "git add \\." CLAUDE.md'),
    ("pub", "rg 'git add --all|git add \\.' tools"),
    ("pub", 'echo "git add ." > list.txt'),
    ("pub", "echo \"git commit -m 'Co-Authored-By: Claude'\""),
    ("pub", 'echo "Co-Authored-By: Claude"'),
    ("pub", 'git log --grep "Co-Authored-By: Claude"'),
    ("pub", 'grep -n "noreply@anthropic.com" .claude/hooks/guard.py'),
    ("pub", 'grep -n "MPLBACKEND=Agg" CLAUDE.md'),
    ("pub", "echo MPLBACKEND=Agg"),
    ("pub", 'echo "taskkill /IM python.exe"'),
    ("pub", "echo wmic"),
    ("pub", "cat <<'EOF'\nimport nbformat\nEOF"),
    # a group's text in single quotes, escaped, or a brace in double quotes is not a call
    ("pub", "echo '$(git add -A)'"),
    ("pub", "echo \\$(git add -A)"),
    ("pub", "grep -n '{{ git add .; }}' CLAUDE.md"),
    ("pub", 'rg "{{git add -A}}" -g "*.{{py,md}}"'),
    ("pub", "git add lib/{{a,b}}.py"),
    ("pub", "(git add lib/a.py)"),
    ("pub", "{{ git add lib/a.py; }}"),
    ("pub", "echo $(git status --short)"),
    # staging by name in the public tree
    ("pub", "git add lib/seams_engine.py lib/MANIFEST.md"),
    ("pub", "git add ./lib/atc.py"),
    ("pub", "git add .gitignore README.md"),
    ("pub", "git add -u"),
    ("pub", "git status"),
    ("pub", "git diff --cached -- sources"),
    ("pub", 'git commit -m "note the add -A rule"'),
    ("pub", 'git add "sources/references.bib"'),
    ("pub", "git add sources/references.bib sources/workshops/deck.pdf"),
    ("pub", "git add sources"),
    ("pub", "git add -f lib/MANIFEST.md"),
    ("pub", "git add sources/academic-literature-index.md"),
    ("pub", "git add :/lib/a.py ':(literal)lib/b.py' ':(top,icase)LIB/c.py'"),
    ("pub", "git -C lib add :/sources/references.bib"),
    ("pub", "printf '%s\\n' lib/a.py lib/b.py | git add --pathspec-from-file=-"),
    ("pub", "printf 'lib/a.py\\nlib/b.py\\n' | git add --pathspec-from-file=-"),
    ("pub", "echo lib/a.py | git add --pathspec-from-file=-"),
    ("pub", "git commit -m 'fix the guard'"),
    ("pub", "git commit -F {ok}"),
    ("pub", 'git commit -m "fix pytest run"'),
    # a tree without the marker is not the public repository
    ("plain", "git add -A"),
    ("plain", "git add -f sources/a.pdf"),
    ("plain", "git -C {pub} add sources/references.bib"),
    # processes by PID, and process queries
    ("pub", "taskkill /PID 1234 /F"),
    ("pub", "Stop-Process -Id 1234"),
    ("pub", "kill 1234"),
    ("pub", "Get-Process python"),
    ("pub", "Get-CimInstance Win32_Process"),
    (
        "pub",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object ProcessId, CommandLine",
    ),
    (
        "pub",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' AND CommandLine LIKE "
        "'%pytest%'\" | ForEach-Object {{ Stop-Process -Id $_.ProcessId }}",
    ),
    # links and folders that do not name data/
    ("pub", "New-Item -ItemType Directory -Path wt/data"),
    ("pub", "mklink /J wt\\lib C:\\main\\lib"),
    ("pub", "Copy-Item -Recurse C:/main/data wt/data"),
    # git clean without ignored files, or a dry run
    ("pub", "git clean -nx"),
    ("pub", "git clean -fd"),
    ("pub", "git clean -n -X"),
    # python heredocs that touch no notebook, and a patch script run as a file
    ("pub", "python - <<'EOF'\nprint(1)\nEOF"),
    ("pub", "python tools/patch_notebook.py a.ipynb"),
    (
        "pub",
        "python - <<'EOF'\nimport json\nnb = json.load(open('a.ipynb', encoding='utf-8'))\n"
        "print(len(nb['cells']))\nEOF",
    ),
    # gh calls that only read, or that post to an allowed owner
    ("pub", "gh issue create -R pdockery/x -t x"),
    ("pub", "gh issue create -R someone/model -t x"),
    ("pub", "gh issue list -R otherorg/toolbox"),
    ("pub", "gh issue view 5 -R otherorg/toolbox"),
    ("pub", "gh api repos/otherorg/toolbox/issues"),
    ("pub", "gh pr create -t x -b y"),
    # the hooks path set to .githooks, or read
    ("pub", "git config core.hooksPath .githooks"),
    ("pub", "git config --get core.hooksPath"),
    # commands that are not launches, at the ceiling
    ("pub", "python script.py"),
    ("pub", "ls tests"),
]

#: Heavy launches, each blocked at the ceiling.
HEAVY = [
    "python -m pytest tests -q",
    "pytest -x",
    "cd x && PYTHONPATH=lib python -m pytest tests/",
    "python -m nbconvert --to notebook --execute a.ipynb",
    "jupyter nbconvert --execute a.ipynb",
    "python tools/nb_execute.py a.ipynb",
    "/opt/conda/bin/python -m pytest x 2>&1 | tail -5",
    'bash -c "cd x && python -m pytest tests -q"',
    'powershell -Command "python tools/nb_execute.py a.ipynb"',
    "bash -c pytest",
]

#: (where, command, rule) that must go to the participant.
MUST_ASK = [
    ("pub", "git commit --no-verify -m x", "hook-bypass"),
    ("pub", "git commit -nm x", "hook-bypass"),
    ("pub", "git -c core.hooksPath=/dev/null commit -m x", "hook-bypass"),
    ("pub", "git config --unset core.hooksPath", "hookspath-config"),
    ("pub", "git config core.hooksPath /tmp/none", "hookspath-config"),
]


class Tables(Trees):
    def test_must_block(self) -> None:
        for where, cmd, rule in MUST_BLOCK:
            with self.subTest(cmd=cmd):
                ctx = self.decide(where, cmd)
                blocks = [f.rule for f in ctx.findings if f.kind == "block"]
                self.assertIn(rule, blocks, f"{where}: {cmd!r} -> {blocks}")

    def test_must_allow(self) -> None:
        for where, cmd in MUST_ALLOW:
            with self.subTest(cmd=cmd):
                ctx = self.decide(where, cmd, count=9)
                found = [(f.kind, f.rule) for f in ctx.findings]
                self.assertEqual(found, [], f"{where}: {cmd!r}")
                self.assertFalse(ctx.heavy, cmd)

    def test_must_ask(self) -> None:
        for where, cmd, rule in MUST_ASK:
            with self.subTest(cmd=cmd):
                ctx = self.decide(where, cmd)
                self.assertEqual(
                    [(f.kind, f.rule) for f in ctx.findings], [("ask", rule)]
                )


class Marker(Trees):
    """The public repository is found by its marker, not by its folder name."""

    def test_the_marker_makes_a_tree_public(self) -> None:
        fork = make_tree(self.tmp / "fork", "my-fork", public=True)
        self.assertTrue(guard.in_public(str(fork)))
        self.assertTrue(guard.in_public(str(fork / "lib")))
        ctx = guard.decide({"tool_input": {"command": "git add -A"}, "cwd": str(fork)})
        self.assertEqual([f.rule for f in ctx.findings], ["public-broad-add"])

    def test_the_folder_name_alone_does_not(self) -> None:
        named = make_tree(self.tmp / "named", "model")
        self.assertFalse(guard.in_public(str(named)))
        ctx = guard.decide({"tool_input": {"command": "git add -A"}, "cwd": str(named)})
        self.assertEqual(ctx.findings, [])

    def test_a_marker_folder_is_not_the_marker(self) -> None:
        (self.plain / "lib" / "MANIFEST.md").mkdir(parents=True)
        self.assertFalse(guard.in_public(str(self.plain)))

    def test_the_rules_file_names_its_own_tree_public(self) -> None:
        with mock.patch.object(guard, "ROOT", self.plain):
            self.assertTrue(guard.in_public(str(self.plain)))
            self.assertFalse(guard.in_public(str(self.upstream)))
        with mock.patch.object(guard, "RULES", guard.RepoRules()):
            self.assertFalse(guard.in_public(str(self.plain)))

    def test_this_repository_is_public(self) -> None:
        self.assertTrue((REPO / "lib" / "MANIFEST.md").is_file())
        self.assertTrue(guard.in_public(str(REPO)))

    def test_no_tree_is_not_public(self) -> None:
        self.assertFalse(guard.in_public(""))
        self.assertFalse(guard.in_public(str(self.tmp)))


class Groups(Trees):
    """The commands inside a group are read; a backtick pair substitutes outside PowerShell.

    The tables above hold the subshell, brace-group and substitution cases. These tests
    read the hook input's `tool_name`, which the tables leave out.
    """

    def rules(self, cmd: str, tool: str) -> list[str]:
        data = {"tool_name": tool, "tool_input": {"command": cmd}, "cwd": str(self.pub)}
        with mock.patch.object(guard, "python_count", return_value=0):
            return [f.rule for f in guard.decide(data).findings]

    def test_a_backtick_pair_substitutes_in_bash(self) -> None:
        cmd = 'git commit -m "note `git add -A`"'
        self.assertEqual(self.rules(cmd, "Bash"), ["public-broad-add"])

    def test_a_backtick_is_an_escape_in_powershell(self) -> None:
        cmd = 'git commit -m "note `git add -A` here"'
        self.assertEqual(self.rules(cmd, "PowerShell"), [])

    def test_a_subexpression_runs_in_powershell(self) -> None:
        cmd = 'Write-Output "$(git add -A)"'
        self.assertEqual(self.rules(cmd, "PowerShell"), ["public-broad-add"])

    def test_an_escaped_subexpression_is_text_in_powershell(self) -> None:
        cmd = 'Write-Output "`$(git add -A)"'
        self.assertEqual(self.rules(cmd, "PowerShell"), [])

    def test_a_closing_mark_is_cut_from_the_last_word_only(self) -> None:
        words = guard.command(guard.lex("git add lib/a.py)")[0])
        self.assertEqual([w.text for w in words], ["git", "add", "lib/a.py"])
        words = guard.command(guard.lex("git add 'a)'")[0])
        self.assertEqual([w.text for w in words], ["git", "add", "a)"])
        words = guard.command(guard.lex("git add lib/f(1).py")[0])
        self.assertEqual([w.text for w in words], ["git", "add", "lib/f(1).py"])


class PathspecFile(Trees):
    """The paths of `--pathspec-from-file` are read and checked one by one."""

    def write(self, name: str, text: str) -> str:
        (self.pub / name).write_bytes(text.encode("utf-8"))
        return name

    def rules(self, where: str, cmd: str) -> list[str]:
        return [f.rule for f in self.decide(where, cmd).findings]

    def test_named_paths_are_allowed(self) -> None:
        f = self.write("list.txt", "lib/a.py\nsources/references.bib\n\n")
        self.assertEqual(self.rules("pub", f"git add --pathspec-from-file={f}"), [])
        self.assertEqual(self.rules("pub", f"git add --pathspec-from-file {f}"), [])

    def test_each_line_is_checked(self) -> None:
        cases = [
            ("lib/a.py\n.\n", "public-broad-add"),
            ("lib/*.py\n", "public-broad-add"),
            ("sources/academic-literature/a.pdf\n", "public-sources"),
            ('"sources/academic-literature/a b.pdf"\n', "public-sources"),
        ]
        for text, rule in cases:
            with self.subTest(text=text):
                f = self.write("list.txt", text)
                self.assertIn(
                    rule, self.rules("pub", f"git add --pathspec-from-file={f}")
                )

    def test_a_forced_source_is_blocked(self) -> None:
        f = self.write("list.txt", "sources/workshops/deck.pdf\n")
        self.assertEqual(self.rules("pub", f"git add --pathspec-from-file={f}"), [])
        self.assertEqual(
            self.rules("pub", f"git add -f --pathspec-from-file={f}"),
            ["public-sources"],
        )

    def test_nul_separated_paths(self) -> None:
        f = self.write("list.bin", "lib/a.py\0sources/academic-literature/a.pdf\0")
        cmd = f"git add --pathspec-file-nul --pathspec-from-file={f}"
        self.assertEqual(self.rules("pub", cmd), ["public-sources"])
        # without the option a NUL still separates, a superset of what git reads
        self.assertEqual(
            self.rules("pub", f"git add --pathspec-from-file={f}"), ["public-sources"]
        )

    def test_a_c_quoted_line_is_decoded(self) -> None:
        f = self.write("list.txt", '"sources/\\141cademic-literature/a.pdf"\n')
        cmd = f"git add --pathspec-from-file={f}"
        self.assertEqual(self.rules("pub", cmd), ["public-sources"])
        self.write("list.txt", '"lib/a\\tb.py"\n"lib/\\"c\\".py"\n')
        self.assertEqual(self.rules("pub", cmd), [])
        self.write("list.txt", '"sources/\\qx.pdf"\n')
        self.assertEqual(self.rules("pub", cmd), ["public-broad-add"])

    def test_stdin_from_a_heredoc_echo_or_cat(self) -> None:
        f = self.write("list.txt", "sources/academic-literature/a.pdf\n")
        blocked = [
            "git add --pathspec-from-file=- <<'EOF'\nsources/academic-literature/a.pdf\nEOF",
            "echo sources/academic-literature/a.pdf | git add --pathspec-from-file=-",
            f"cat {f} | git add --pathspec-from-file=-",
        ]
        for cmd in blocked:
            with self.subTest(cmd=cmd):
                self.assertEqual(self.rules("pub", cmd), ["public-sources"])
        ok = "git add --pathspec-from-file=- <<'EOF'\nlib/a.py\nEOF"
        self.assertEqual(self.rules("pub", ok), [])
        for head in ("cat", "type", "Get-Content"):
            absent = f"{head} absent.txt | git add -f --pathspec-from-file=-"
            with self.subTest(cmd=absent):
                self.assertIn("public-broad-add", self.rules("pub", absent))

    def test_a_plain_tree_has_no_pathspec_rule(self) -> None:
        (self.plain / "list.txt").write_bytes(b"sources/academic-literature/a.pdf\n")
        cmd = "git add --pathspec-from-file=list.txt"
        self.assertEqual(self.rules("plain", cmd), [])
        self.assertEqual(
            self.rules("plain", "git add --pathspec-from-file=absent.txt"), []
        )


class Rules(Trees):
    """The rules file: its defaults, its reading, and the messages it words."""

    def test_defaults_are_neutral(self) -> None:
        r = guard.RepoRules()
        self.assertEqual(
            (r.repository, r.never_committed, r.ignored_only_copy, r.allowed_owners),
            ("", [], [], []),
        )
        self.assertEqual(
            (r.commit_hook_rules, r.defect_destination, r.notes), ("", "", {})
        )
        self.assertEqual(r.note("claude-trailer"), "")

    def test_a_missing_or_broken_file_leaves_the_defaults(self) -> None:
        self.assertEqual(guard.load_rules(self.tmp / "absent.json"), guard.RepoRules())
        bad = self.tmp / "bad.json"
        for text in ("{not json", "[1, 2]", '"text"'):
            with self.subTest(text=text):
                bad.write_bytes(text.encode("utf-8"))
                self.assertEqual(guard.load_rules(bad), guard.RepoRules())
        self.assertTrue(self.record.called)

    def test_a_bad_value_falls_back_alone(self) -> None:
        p = self.tmp / "rules.json"
        data = {
            "_about": "a comment",
            "never_committed": "drafts/",
            "allowed_owners": ["me"],
            "notes": {"wmic": 3},
            "colour": "blue",
        }
        p.write_bytes(json.dumps(data).encode("utf-8"))
        r = guard.load_rules(p)
        self.assertEqual(
            (r.never_committed, r.allowed_owners, r.notes), ([], ["me"], {})
        )
        warned = " ".join(str(c.args) for c in self.record.call_args_list)
        for key in ("never_committed", "notes", "colour"):
            self.assertIn(key, warned)
        self.assertNotIn("_about", warned)

    def test_this_repository_file_matches_the_tables(self) -> None:
        r = guard.load_rules(guard.RULES_FILE)
        self.assertEqual(r, PUBLIC_RULES)
        self.record.assert_not_called()
        self.assertEqual(guard.load_rules(HOOKS / "repo_rules.json"), PUBLIC_RULES)

    def test_this_repository_messages(self) -> None:
        msg = {
            f.rule: f.message
            for cmd in (
                "git clean -fdx",
                "git commit --no-verify -m x",
                "gh issue create -R otherorg/toolbox -t x",
            )
            for f in self.decide("pub", cmd).findings
        }
        self.assertIn("an ignored file can be the only copy", msg["clean-ignored"])
        self.assertIn(
            "enforce the trailer, file-size and sources rules,", msg["hook-bypass"]
        )
        self.assertIn("is for the participant", msg["upstream-post"])

    def test_neutral_messages_name_no_folder(self) -> None:
        with mock.patch.object(guard, "RULES", guard.RepoRules()):
            texts = []
            for cmd in (
                "git clean -fdx",
                "git commit --no-verify -m x",
                "gh issue create -R otherorg/toolbox -t x",
                "git commit -m 'Co-Authored-By: Claude'",
            ):
                texts += [f.message for f in self.decide("plain", cmd).findings]
            self.assertEqual(len(texts), 4)
            joined = " ".join(texts)
            self.assertNotIn("2026-", joined)
            self.assertIn("the repository's commit rules", joined)
            self.assertIn("is for the participant", joined)

    def test_a_listed_folder_is_never_committed(self) -> None:
        rules = guard.RepoRules(
            never_committed=["drafts/"], notes={"never-committed-add": "It is local."}
        )
        with mock.patch.object(guard, "RULES", rules):
            for cmd in ("git add drafts/a.md", "git add -f drafts", "git add -f -A"):
                with self.subTest(cmd=cmd):
                    found = [f.rule for f in self.decide("plain", cmd).findings]
                    self.assertEqual(found, ["never-committed-add"])
            msg = self.decide("plain", "git add drafts/a.md").findings[0].message
            self.assertIn("drafts/ is never committed", msg)
            self.assertIn("It is local.", msg)
            self.assertEqual(self.decide("plain", "git add lib/a.py").findings, [])

    def test_no_listed_folder_means_no_rule(self) -> None:
        self.assertEqual(self.decide("pub", "git add drafts/a.md").findings, [])


class OriginOwner(Trees):
    """Rule 10 in the public repository allows the owner of its origin remote."""

    def post(self, cwd: Path, cmd: str) -> list[str]:
        data = {"tool_input": {"command": cmd}, "cwd": str(cwd)}
        return [f.rule for f in guard.decide(data).findings]

    def test_the_origin_owner_may_post_from_a_public_tree(self) -> None:
        fork = make_tree(self.tmp / "fork", "fork", owner="Someone", public=True)
        self.assertEqual(self.post(fork, "gh issue create -t x"), [])
        self.assertEqual(self.post(fork, "gh issue create -R someone/fork -t x"), [])
        self.assertEqual(
            self.post(fork, "gh issue create -R otherorg/toolbox -t x"),
            ["upstream-post"],
        )

    def test_the_rules_owner_stays_allowed(self) -> None:
        fork = make_tree(self.tmp / "fork", "fork", public=True)
        self.assertEqual(self.post(fork, "gh pr create -R pdockery/x -t y"), [])

    def test_only_the_origin_section_counts(self) -> None:
        fork = make_tree(self.tmp / "fork", "fork", public=True)
        (fork / ".git" / "config").write_bytes(
            b'[remote "upstream"]\n\turl = https://github.com/otherorg/toolbox.git\n'
            b'[remote "origin"]\n\turl = git@github.com:someone/fork.git\n'
        )
        self.assertEqual(guard.origin_owner(str(fork)), "someone")
        self.assertEqual(
            self.post(fork, "gh issue create -R otherorg/toolbox -t x"),
            ["upstream-post"],
        )

    def test_a_plain_tree_does_not_allow_its_origin(self) -> None:
        ctx = self.decide("up", "gh issue create -t x -b y")
        self.assertEqual([f.rule for f in ctx.findings], ["upstream-post"])


class Ceiling(Trees):
    def test_heavy_blocked_at_ceiling(self) -> None:
        for cmd in HEAVY:
            with self.subTest(cmd=cmd):
                ctx = self.decide("pub", cmd, count=_limits.CEILING)
                self.assertTrue(ctx.heavy, cmd)
                self.assertIn("ceiling", [f.rule for f in ctx.findings])

    def test_heavy_allowed_below_ceiling(self) -> None:
        for cmd in HEAVY:
            with self.subTest(cmd=cmd):
                ctx = self.decide("pub", cmd, count=_limits.CEILING - 1)
                self.assertEqual(ctx.findings, [])

    def test_background_needs_the_margin(self) -> None:
        cmd = "python -m pytest tests -q"
        limit = _limits.CEILING - _limits.BACKGROUND_MARGIN
        ctx = self.decide("pub", cmd, count=limit + 1, background=True)
        self.assertEqual([f.rule for f in ctx.findings], ["ceiling-background"])
        ctx = self.decide("pub", cmd, count=limit, background=True)
        self.assertEqual(
            [(f.kind, f.rule) for f in ctx.findings], [("advise", "background-pid")]
        )
        self.assertIn("PID", ctx.findings[0].message)

    def test_count_unavailable_allows(self) -> None:
        ctx = self.decide("pub", "python -m pytest", count=None)
        self.assertEqual(ctx.findings, [])

    def test_a_heavy_list_adds_its_scripts(self) -> None:
        listed = self.tmp / "heavy_entrypoints.txt"
        listed.write_bytes(
            b"# heavy scripts\nscripts/long_run.py\nscripts/sweep_*.py\n"
        )
        with mock.patch.object(guard, "HEAVY_LIST", listed):
            for cmd in (
                "python scripts/long_run.py",
                "cd scripts && python sweep_2024.py",
            ):
                with self.subTest(cmd=cmd):
                    self.assertTrue(self.decide("pub", cmd, 9).heavy)
            self.assertFalse(self.decide("pub", "python scripts/short.py", 9).heavy)

    def test_missing_heavy_list_leaves_only_the_fixed_launches(self) -> None:
        with mock.patch.object(guard, "HEAVY_LIST", self.tmp / "absent.txt"):
            self.assertFalse(self.decide("pub", "python scripts/long_run.py", 9).heavy)
            self.assertTrue(self.decide("pub", "python -m pytest", 9).heavy)


class Worktree(Trees):
    def make_link(self, link: Path, target: Path) -> bool:
        target.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            r = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
            return r.returncode == 0
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            return False
        return True

    def test_forced_remove_with_a_junction_is_blocked(self) -> None:
        wt = self.pub / "wt"
        (wt / "model").mkdir(parents=True)
        if not self.make_link(wt / "model" / "lnk", self.tmp / "target"):
            self.skipTest("cannot create a junction here")
        self.addCleanup(os.rmdir, wt / "model" / "lnk")
        ctx = self.decide("pub", "git worktree remove --force wt")
        self.assertEqual([f.rule for f in ctx.findings], ["worktree-force-reparse"])
        self.assertIn("lnk", ctx.findings[0].message)
        ctx = self.decide("pub", "git worktree remove -f -f wt")
        self.assertEqual([f.rule for f in ctx.findings], ["worktree-force-reparse"])
        self.assertEqual(self.decide("pub", "git worktree remove wt").findings, [])

    def test_forced_remove_without_a_link_is_allowed(self) -> None:
        (self.pub / "wt" / "a" / "b").mkdir(parents=True)
        self.assertEqual(
            self.decide("pub", "git worktree remove --force wt").findings, []
        )

    def test_unknown_after_budget_is_blocked(self) -> None:
        (self.pub / "wt" / "a").mkdir(parents=True)
        with mock.patch.object(guard, "REPARSE_BUDGET_DIRS", 0):
            ctx = self.decide("pub", "git worktree remove --force wt")
        self.assertEqual([f.rule for f in ctx.findings], ["worktree-force-reparse"])
        self.assertIn("could not confirm", ctx.findings[0].message)


class Protocol(Trees):
    """The hook contract: exit codes, stderr, the JSON on stdout and the log."""

    def run_main(self, payload: object) -> tuple[int, str, str]:
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO(raw)),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", err),
            mock.patch.object(guard, "python_count", return_value=0),
        ):
            rc = guard.main()
        return rc, out.getvalue(), err.getvalue()

    def payload(self, cmd: str, **extra: object) -> dict[str, object]:
        ti: dict[str, object] = {"command": cmd, **extra}
        return {"tool_input": ti, "cwd": str(self.pub), "session_id": "s1"}

    def test_block_exits_2_with_the_message_and_logs(self) -> None:
        rc, out, err = self.run_main(self.payload("wmic process list"))
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("Get-CimInstance", err)
        self.record.assert_called_once()
        args = self.record.call_args.args
        self.assertEqual(args[:3], ("guard", "block", "wmic"))
        self.assertEqual(args[4], "s1")

    def test_allow_is_silent(self) -> None:
        self.assertEqual(self.run_main(self.payload("git status")), (0, "", ""))
        self.record.assert_not_called()

    def test_ask_returns_the_decision(self) -> None:
        rc, out, _ = self.run_main(self.payload("git commit --no-verify -m x"))
        self.assertEqual(rc, 0)
        hso = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(hso["hookEventName"], "PreToolUse")
        self.assertEqual(hso["permissionDecision"], "ask")
        self.assertIn("--no-verify", hso["permissionDecisionReason"])

    def test_advice_returns_context_without_a_decision(self) -> None:
        rc, out, _ = self.run_main(
            self.payload("python -m pytest tests", run_in_background=True)
        )
        self.assertEqual(rc, 0)
        hso = json.loads(out)["hookSpecificOutput"]
        self.assertNotIn("permissionDecision", hso)
        self.assertIn("PID", hso["additionalContext"])

    def test_unparseable_and_non_object_input_allow(self) -> None:
        self.assertEqual(self.run_main("not json")[0], 0)
        self.assertEqual(self.run_main("[1, 2]")[0], 0)
        self.assertEqual(self.run_main({"tool_input": None})[0], 0)

    def test_internal_error_allows(self) -> None:
        with mock.patch.object(guard, "decide", side_effect=RuntimeError("x")):
            rc, _, err = self.run_main(self.payload("wmic"))
        self.assertEqual(rc, 0)
        self.assertIn("allowing the call", err)

    def test_a_failing_count_allows(self) -> None:
        raw = json.dumps(self.payload("pytest"))
        with (
            mock.patch("sys.stdin", io.StringIO(raw)),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
            mock.patch.object(guard, "python_count", side_effect=OSError("x")),
        ):
            self.assertEqual(guard.main(), 0)

    def test_one_failing_rule_leaves_the_others(self) -> None:
        boom = mock.Mock(side_effect=RuntimeError("x"))
        boom.__name__ = "rule_kill"
        with mock.patch.object(guard, "rule_kill", boom):
            rc, _, _ = self.run_main(
                self.payload("taskkill /IM python.exe; wmic os get")
            )
        self.assertEqual(rc, 2)
        kinds = [c.args[1:3] for c in self.record.call_args_list]
        self.assertIn(("warn", "rule_kill"), kinds)
        self.assertIn(("block", "wmic"), kinds)

    def test_ceiling_message_names_the_count_and_says_by_pid(self) -> None:
        with mock.patch.object(guard, "python_count", return_value=_limits.CEILING):
            ctx = guard.decide(self.payload("python -m pytest"))
        msg = ctx.findings[0].message
        self.assertIn(f"{_limits.CEILING} Python interpreter processes", msg)
        self.assertIn("by PID", msg)
        self.assertIn("LAB_PROCESS_CEILING", msg)


class Limits(unittest.TestCase):
    def test_python_image_names(self) -> None:
        for n in (
            "python.exe",
            "pythonw.exe",
            "python3",
            "python3.12",
            "/opt/x/bin/python",
        ):
            self.assertTrue(_limits.is_python_image(n), n)
        for n in ("pycharm64.exe", "ipython", "jupyter-lab", "py.exe", "pythonista"):
            self.assertFalse(_limits.is_python_image(n), n)

    def test_count_skips_self_and_a_python_parent(self) -> None:
        procs = {
            1: ("explorer.exe", 0),
            10: ("python.exe", 1),
            11: ("python.exe", 10),
            12: ("python3.12", 1),
            13: ("bash", 1),
        }
        self.assertEqual(_limits.count_from_table(procs, 11), 1)
        self.assertEqual(_limits.count_from_table(procs, 13), 3)

    def test_parse_ps(self) -> None:
        text = "  1     0 launchd\n 50     1 /opt/anaconda3/bin/python3.12\n bad line\n"
        self.assertEqual(
            _limits.parse_ps(text),
            {1: ("launchd", 0), 50: ("/opt/anaconda3/bin/python3.12", 1)},
        )

    def test_posix_table_fails_to_none(self) -> None:
        with mock.patch.object(_limits.subprocess, "run", side_effect=OSError("no ps")):
            self.assertIsNone(_limits.posix_table())

    def test_count_on_this_machine(self) -> None:
        n = _limits.python_count()
        self.assertTrue(n is None or n >= 0)

    def test_guard_loads_by_path_without_a_module_entry(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "guard_by_path", HOOKS / "guard.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ctx = mod.decide({"tool_input": {"command": "wmic os get"}, "cwd": str(REPO)})
        self.assertEqual([f.rule for f in ctx.findings], ["wmic"])

    def test_ceiling_from_the_environment(self) -> None:
        var = _limits.CEILING_VAR
        self.assertEqual(var, "LAB_PROCESS_CEILING")
        self.assertEqual(_limits.ceiling_from_env(7, {}), 7)
        self.assertEqual(_limits.ceiling_from_env(7, {var: "12"}), 12)
        self.assertEqual(_limits.ceiling_from_env(7, {var: " 4 "}), 4)
        for bad in ("", "x", "0", "-3", "2.5"):
            with self.subTest(value=bad):
                self.assertEqual(_limits.ceiling_from_env(7, {var: bad}), 7)

    def test_the_module_reads_the_variable_at_import(self) -> None:
        for value, want in (("11", 11), ("none", 7)):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {_limits.CEILING_VAR: value}),
            ):
                spec = importlib.util.spec_from_file_location(
                    "limits_by_path", HOOKS / "_limits.py"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.assertEqual(mod.CEILING, want)

    def test_guard_uses_the_shared_ceiling(self) -> None:
        self.assertEqual(guard.CEILING, _limits.CEILING)
        self.assertEqual(guard.BACKGROUND_MARGIN, _limits.BACKGROUND_MARGIN)


class Log(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp()) / "logs"
        self.addCleanup(shutil.rmtree, self.dir.parent, True)

    def test_writes_one_line_and_an_ignore_file(self) -> None:
        _log.record("guard", "block", "wmic", "wmic   os\nget", "s1", log_dir=self.dir)
        self.assertEqual((self.dir / ".gitignore").read_text(encoding="utf-8"), "*\n")
        lines = (self.dir / _log.LOG_NAME).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        line = json.loads(lines[0])
        self.assertEqual(
            {k: line[k] for k in ("hook", "kind", "rule", "session", "detail")},
            {"hook": "guard", "kind": "block", "rule": "wmic", "session": "s1",
             "detail": "wmic os get"},
        )  # fmt: skip

    def test_rotates_past_the_size(self) -> None:
        with mock.patch.object(_log, "MAX_BYTES", 10):
            _log.record("guard", "warn", "a", "x" * 50, log_dir=self.dir)
            _log.record("guard", "warn", "b", log_dir=self.dir)
        self.assertTrue((self.dir / (_log.LOG_NAME + ".1")).is_file())
        lines = (self.dir / _log.LOG_NAME).read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["rule"], "b")

    def test_fails_open(self) -> None:
        self.dir.parent.mkdir(exist_ok=True)
        self.dir.write_bytes(b"a file where the folder should be")
        _log.record("guard", "block", "x", log_dir=self.dir)  # no exception

    def test_the_log_folder_is_ignored(self) -> None:
        self.assertEqual(_log.LOG_DIR, _LOG_TMP / "logs")
        text = (REPO / ".gitignore").read_bytes().decode("utf-8")
        self.assertIn(".claude/logs/", text.splitlines())


if __name__ == "__main__":
    unittest.main()
