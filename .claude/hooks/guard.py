#!/usr/bin/env python
"""PreToolUse guard for the rules that must hold every time.

Contract. Claude Code runs this script before every Bash or PowerShell call, with the call's
JSON on stdin: `tool_input.command`, `tool_input.run_in_background`, `cwd` and `session_id`.
Exit 2 blocks the call and returns the message on stderr to the agent. Exit 0 lets the call
run. With exit 0 a JSON object on stdout can carry an "ask" decision, which puts the call to the
participant, or an advisory as `additionalContext`. Every block, ask, advisory and internal
warning is appended to `.claude/logs/hooks.jsonl` by `_log.py`. An error of the guard's own
allows the call with one line on stderr (CODE_STYLE.md 13.4). A rule stated in the
repository's instructions or a skill is advisory, and this script is the enforcement.

One file for every repository. The same file, byte for byte, serves each repository that
carries these hooks. The facts that differ between repositories are read at start from
`repo_rules.json` beside this file, whose keys `RepoRules` names. A missing or unreadable
rules file leaves the neutral defaults: no never-committed folder, no named owner and generic
messages. A missing `tools/heavy_entrypoints.txt` leaves the fixed launches of rule 5.

How a command is read. PowerShell here-strings and shell heredocs are lifted out first, and
each body is kept with the segment that opened it. The command is then split into segments on
`;`, `&`, `|` and newlines outside quotes and braces, and each segment into words, quotes
resolved. A rule reads the command word of a segment, after leading variable assignments,
prefixes such as `env` or `sudo`, the reserved words after which a command follows (`if`,
`then`, `elif`, `else`, `while`, `until`, `do`, `!`, `coproc`), the head `case <word> in` and
a case pattern such as `x)`. So an add inside an `if`, a loop or a case arm is read as the
add it is, and `echo`, `grep` or `cat` of a rule's text passes. The payload of a wrapper
(`bash -c`, `sh -c`, `zsh -c`, `cmd /c`, `powershell -Command`, `pwsh`, `eval`, `iex`) and
the text piped or fed into a shell are read again as commands, four levels deep at most.
Under any command word, the payload of a command substitution `$( ... )`, the body of a brace
group or script block `{ ... }` and, in a call that is not PowerShell, the payload of a
backtick pair are read again as commands too, so `echo $(git add -A)` is read as the add it
runs, and a segment that opens with `{` is read only through its body. A heredoc opened
inside such a command is read again with its body, so a heredoc in a brace group or a
substitution reads as it does outside. Text in single quotes is not read, a brace in double
quotes is text, and `${...}` is a variable. A `)` or `}` beyond the groups a segment's words
open is cut from its last word, so the subshell `(cd lib; git add .)` reads the pathspec `.`.
A `cd`, `Set-Location` or `pushd` segment moves the working directory for the segments after
it. A git call is read after its global options: `-C <dir>` moves the working directory, and
`-c <key=value>`, `--git-dir`, `--work-tree` and `--no-pager` are consumed. The public
repository is the git working tree whose root holds `lib/MANIFEST.md`, the manifest of its
copied library, or this file's own tree when the rules file names the repository "public".

The rules, by the name each carries in the log:

1. public-broad-add and public-sources. In the public repository: no `git add` of -A, --all,
   `.`, `:/` or a glob; no `git add` of a path under `sources/academic-literature/`, whose
   sources are cited and never published; and no forced add (`-f`, `--force`) of a path
   under `sources/`, which would stage a file its ignore rules keep out. The paths of a
   `--pathspec-from-file` file are read and checked one by one. A pathspec file the guard
   cannot read whole, because it is absent, unreadable or longer than `READ_LIMIT`, or
   that lists more than `PATHSPEC_LIMIT` paths, counts as a broad add, whether the option
   names it or `cat`, `type` or `Get-Content` pipes it in. The two limits hold the check
   of a listed file inside the hook timeout `settings.json` sets, past which the call
   would run unchecked, because a hook that times out does not block. A leading pathspec
   magic is read as git reads it (`:/` and `top` from the root), and a pathspec the guard
   cannot know counts as broad: an exclude or `attr` magic, a substitution, a variable, a
   backtick, or piped text with an escape the guard does not expand. An add inside a
   subshell, a brace group or a command substitution is read as the same add outside it.
2. mplbackend-agg. No `MPLBACKEND=Agg` on a run, because it strips every stored figure.
3. claude-trailer. No Claude authorship trailer in a commit, merge or tag message. The rule
   reads the whole command, every heredoc body and the file of `commit -F`.
4. never-committed-add. No `git add` of a path under a folder the rules file lists as never
   committed, and no forced add of a broad pathspec or of a folder that holds one.
5. ceiling. No launch of pytest, nbconvert, tools/nb_execute.py or a heavy entry point of
   `tools/heavy_entrypoints.txt` while the Python interpreter count is at the ceiling of
   `_limits.py`, which `LAB_PROCESS_CEILING` can set. A launch with `run_in_background` needs
   the count `BACKGROUND_MARGIN` below the ceiling, and an allowed one is told to record its
   PID. A failed count allows the call.
6. kill-by-name. No ending of processes by image name (`taskkill /IM python*`,
   `Stop-Process -Name python*`, `pkill python`, `Get-Process python | Stop-Process`), which
   also ends the participant's kernels. A process is ended by its PID.
7. junction-into-data. No `mklink /J` or `/D`, `New-Item -ItemType Junction` or
   `SymbolicLink`, or `ln -s` that names a data/ path.
8. worktree-force-reparse. No `git worktree remove --force` while the worktree holds a
   junction or symbolic link, which a forced remove deletes through.
9. clean-ignored. No `git clean` with -x or -X except a dry run: an ignored file can be the
   only copy.
10. upstream-post. No `gh issue` or `gh pr` create, comment, review or edit, and no writing
    `gh api` call on issues or pulls, against a repository of an owner not allowed. The
    allowed owners are those the rules file lists and, in the public repository, the owner
    of its `origin` remote.
11. wmic. No `wmic`, which is deprecated; `Get-CimInstance` replaces it.
12. heredoc-notebook. No program fed to python on stdin (a heredoc or a here-string, with
    no script or with `-`) that names `.ipynb` or `nbformat` and writes a file: the shell or
    the inline string can strip the script's backslashes. The script is written with Write
    and run as a file. A program that only reads a notebook, and data piped into a named
    script, are allowed.
13. hook-bypass and hookspath-config (ask, not block). A commit or merge with `--no-verify`
    or with `core.hooksPath` set on the call, and a `git config` that unsets or redirects
    `core.hooksPath` away from `.githooks`, go to the participant.
"""

# No `from __future__ import annotations`: with it the dataclasses below fail when the file is
# loaded by path without a sys.modules entry, as a test may load it.
import fnmatch
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

try:
    from _limits import BACKGROUND_MARGIN, CEILING, python_count  # noqa: E402
except Exception:  # the other rules still run without the shared limits
    CEILING, BACKGROUND_MARGIN = 7, 2

    def python_count() -> int | None:
        """The count's stand-in when `_limits.py` cannot be imported: unavailable."""
        return None


try:
    from _log import record  # noqa: E402
except Exception:

    def record(
        hook: str,
        kind: str,
        rule: str,
        detail: str = "",
        session: str = "",
        log_dir: Path | None = None,
    ) -> None:
        """The log's stand-in when `_log.py` cannot be imported: nothing is written."""
        return


#: The repository root, two levels above this folder (`.claude/hooks`).
ROOT = HOOKS.parents[1]
#: The tracked list of heavy entry points.
HEAVY_LIST = ROOT / "tools" / "heavy_entrypoints.txt"
#: This repository's own facts, read at start.
RULES_FILE = HOOKS / "repo_rules.json"
#: The tracked file, relative to a tree's root, that marks the public repository.
PUBLIC_MARKER = ("lib", "MANIFEST.md")
#: The public repository's sources folder, and its folder of sources never published.
SOURCES_DIR = "sources"
ACADEMIC_DIR = "sources/academic-literature"
#: The deepest a wrapper's payload is read again as a command.
MAX_DEPTH = 4
#: The most characters the guard reads of a file; a pathspec file past it is unknown.
READ_LIMIT = 1_000_000
#: The most paths the guard checks from a pathspec file; a file that lists more is unknown.
PATHSPEC_LIMIT = 20_000
#: The time and folder budget of the junction search under a worktree.
REPARSE_BUDGET_S = 3.0
REPARSE_BUDGET_DIRS = 20_000
#: Windows' FILE_ATTRIBUTE_REPARSE_POINT.
REPARSE_ATTR = 0x400

TRAILER = re.compile(
    r"Co-Authored-By:\s*Claude|Generated with \[Claude Code\]|noreply@anthropic\.com",
    re.I,
)
MPL_AGG = re.compile(r"MPLBACKEND['\"\]]*\s*=\s*['\"]?Agg", re.I)
LAUNCH = re.compile(
    r"-m\s+pytest\b|\bnbconvert\b|\bnb_execute\.py\b|\bpy\.test\b", re.I
)
PYTHON_NAME = re.compile(r"pythonw?(\d+(\.\d+)?)?|py", re.I)
ASSIGN = re.compile(r"^[A-Za-z_]\w*=")
HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)\s*(['\"]?)([A-Za-z_][\w.-]*)\2")
#: Text the shell rewrites before git reads it: a substitution, a variable or a backtick.
SHELL_EXPANDS = re.compile(r"\$[({A-Za-z_]|`")
HERESTRING = re.compile(r"@(['\"])[ \t]*\r?\n(.*?)\r?\n\1@", re.S)
NOTEBOOK = re.compile(r"\.ipynb\b|nbformat")
WRITES = re.compile(
    r"\.write\(|json\.dump|write_text|write_bytes|\.to_file\("
    r"|open\([^)]*['\"](?:[wax]b?\+?|r\+)['\"]"
)
GITHUB_OWNER = re.compile(r"github\.com[/:]([^/\s'\"]+)/", re.I)
#: The words read only for their text, whose segments no text rule reads.
READ_ONLY = frozenset(
    {
        "grep", "rg", "egrep", "fgrep", "git", "echo", "printf", "cat", "ls", "head", "tail",
        "type", "findstr", "select-string", "sls", "get-content", "gc", "write-output",
        "write-host", "write-error", "less", "more", "wc", "dir",
    }
)  # fmt: skip
#: The words that run the rest of the segment as a command.
PREFIXES = frozenset(
    {"sudo", "nohup", "time", "env", "command", "exec", "builtin", "noglob"}
)
#: The shell's reserved words after which a command word follows, PowerShell's `elseif` too.
RESERVED = frozenset(
    {"!", "if", "then", "elif", "elseif", "else", "while", "until", "do", "coproc"}
)
#: A word that opens a redirection, such as `>`, `2>>`, `<` or `&>`.
REDIRECT = re.compile(r"^(?:\d*[<>]|&>)")
#: The shells and evaluators whose payload or input is itself a command.
SHELLS = frozenset(
    {"bash", "sh", "zsh", "dash", "cmd", "powershell", "pwsh", "eval", "iex",
     "invoke-expression"}
)  # fmt: skip
#: A wrapper's options that take the next word as their value.
WRAPPER_VALUE_OPTS = frozenset(
    {"-executionpolicy", "-ep", "-windowstyle", "-version", "-inputformat", "-outputformat",
     "-configurationname", "-workingdirectory", "-file", "-encodedcommand"}
)  # fmt: skip
#: The options of `gh api` that take the next word as their value.
GH_API_VALUE_OPTS = frozenset(
    {"-X", "--method", "-f", "-F", "--field", "--raw-field", "-H", "--header", "--input",
     "-q", "--jq", "-t", "--template", "-p", "--preview", "--hostname", "--cache"}
)  # fmt: skip
CD_WORDS = frozenset({"cd", "chdir", "set-location", "sl", "pushd", "push-location"})
KILL_VERBS = frozenset({"stop-process", "spps", "kill"})
#: git global options that take the next word as their value.
GIT_VALUE_OPTS = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--config-env",
        "--super-prefix",
    }
)


@dataclass
class Word:
    """One word of a segment: its text with quotes resolved, and whether any part was quoted."""

    text: str
    quoted: bool = False


@dataclass
class Segment:
    """One simple command between separators.

    `raw` keeps the quotes, `bare` replaces each quoted string by the placeholder `Q`,
    `payloads` holds the quoted strings' contents, `sep` is the separator before the segment
    and `bodies` the heredoc bodies the segment opened.
    """

    raw: str = ""
    bare: str = ""
    words: list[Word] = field(default_factory=list)
    payloads: list[str] = field(default_factory=list)
    sep: str = ""
    bodies: list[str] = field(default_factory=list)


@dataclass
class Finding:
    """A rule's decision on the call: "block", "ask" or "advise", with the rule's name."""

    kind: str
    rule: str
    message: str


@dataclass
class Context:
    """What the reading of one call collects: the findings and whether it launches heavy work."""

    background: bool = False
    heavy: bool = False
    #: Whether a backtick pair is a substitution, as in Bash; in PowerShell it escapes.
    backticks: bool = True
    findings: list[Finding] = field(default_factory=list)
    heavy_globs: list[str] | None = None

    def add(self, kind: str, rule: str, message: str) -> None:
        """Keep a finding once per rule."""
        if all(f.rule != rule for f in self.findings):
            self.findings.append(Finding(kind, rule, message))


@dataclass
class RepoRules:
    """The facts of one repository, read from `repo_rules.json`; the defaults are neutral.

    `repository` is "public" when this file's own tree is the public repository.
    `never_committed` lists the folders no `git add` may stage (rule 4). `ignored_only_copy`
    lists the ignored folders the clean-ignored message names (rule 9). `commit_hook_rules`
    names what the commit hooks enforce, for the hook-bypass message (rule 13).
    `allowed_owners` lists the GitHub owners a post may go to (rule 10), and
    `defect_destination` names where the upstream-post message sends a defect instead.
    `notes` maps a rule's log name to one sentence appended to that rule's message.
    """

    repository: str = ""
    never_committed: list[str] = field(default_factory=list)
    ignored_only_copy: list[str] = field(default_factory=list)
    commit_hook_rules: str = ""
    allowed_owners: list[str] = field(default_factory=list)
    defect_destination: str = ""
    notes: dict[str, str] = field(default_factory=dict)

    def note(self, rule: str) -> str:
        """The rule's note with a leading space, or ""."""
        text = self.notes.get(rule, "").strip()
        return " " + text if text else ""


def valid_rule_value(value: object, default: object) -> bool:
    """Whether a rules-file value has the type of its default: a string, strings or a map."""
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if isinstance(default, dict):
        return isinstance(value, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        )
    return False


def load_rules(path: Path) -> RepoRules:
    """The repository's facts from a rules file, each bad or missing value at its default.

    A key opening with `_` is a comment. An unreadable file, a file that is not a JSON
    object, an unknown key and a value of the wrong type are each logged as a warning.
    """
    rules = RepoRules()
    try:
        if not path.is_file():
            return rules
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # the neutral defaults stand
        record("guard", "warn", "repo-rules", f"{type(e).__name__}: {e}")
        return rules
    if not isinstance(data, dict):
        record("guard", "warn", "repo-rules", "the rules file is not a JSON object")
        return rules
    for key, value in data.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if key not in rules.__dataclass_fields__:
            record("guard", "warn", "repo-rules", f"unknown key {key!r} ignored")
        elif not valid_rule_value(value, getattr(rules, key)):
            record(
                "guard", "warn", "repo-rules", f"{key!r} has the wrong type; ignored"
            )
        else:
            setattr(rules, key, value)
    return rules


#: This repository's facts, read once at start.
RULES = load_rules(RULES_FILE)


def join_names(names: list[str]) -> str:
    """Names joined as a list in prose: "a", "a and b", "a, b and c"."""
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# --- reading the command -------------------------------------------------------------------


def lift_herestrings(cmd: str) -> str:
    """The command with each PowerShell here-string turned into one single-quoted word.

    Quote characters inside the body are replaced by control characters, so that the body
    reads as one quoted string; no rule reads a quote character.
    """
    return HERESTRING.sub(
        lambda m: "'" + m.group(2).replace("'", "\x1f").replace('"', "\x1e") + "'", cmd
    )


def lift_heredocs(cmd: str) -> tuple[str, list[tuple[int, str]]]:
    """The command with its heredoc bodies removed, and each body with its opener's offset."""
    lines = cmd.split("\n")
    out: list[str] = []
    bodies: list[tuple[int, str]] = []
    offset, i = 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        start = offset
        offset += len(line) + 1
        i += 1
        for m in HEREDOC.finditer(line):
            delim, strip_tabs = m.group(3), m.group(1) == "-"
            body: list[str] = []
            while i < len(lines):
                cand = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if cand.strip() == delim:
                    break
                body.append(lines[i - 1])
            bodies.append((start + m.start(), "\n".join(body)))
    return "\n".join(out), bodies


def lex(cmd: str) -> list[Segment]:
    """The command's segments, each with its words, payloads, separator and heredoc bodies."""
    text, bodies = lift_heredocs(lift_herestrings(cmd))
    segs: list[Segment] = []
    seg, starts = Segment(), [0]
    word: list[str] = []
    quoted = False
    inword = False
    depth = 0
    sep = ""
    i, n = 0, len(text)

    def end_word() -> None:
        nonlocal word, quoted, inword
        if inword:
            seg.words.append(Word("".join(word), quoted))
        word, quoted, inword = [], False, False

    def end_segment(at: int) -> None:
        nonlocal seg, sep
        end_word()
        if seg.words or seg.raw.strip():
            seg.sep = sep
            segs.append(seg)
            starts.append(at)
            sep = ""
        seg = Segment()

    while i < n:
        ch = text[i]
        if ch in "\"'":
            j, buf = i + 1, []
            while j < n and text[j] != ch:
                if ch == '"' and text[j] in "\\`" and j + 1 < n and text[j + 1] == '"':
                    buf.append('"')
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            content = "".join(buf)
            seg.payloads.append(content)
            seg.raw += text[i : j + 1]
            seg.bare += " Q "
            word.append(content)
            quoted = inword = True
            i = j + 1
            continue
        if ch in "\\`" and i + 1 < n and text[i + 1] in "\"' ":
            word.append(text[i + 1])
            inword = True
            seg.raw += text[i : i + 2]
            seg.bare += text[i : i + 2]
            i += 2
            continue
        redirect_amp = ch == "&" and (
            (i > 0 and text[i - 1] in "<>") or (i + 1 < n and text[i + 1] == ">")
        )
        if ch in ";&|\n" and depth == 0 and not redirect_amp:
            end_segment(i)
            sep += ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        seg.raw += ch
        seg.bare += ch
        if ch.isspace():
            end_word()
        else:
            word.append(ch)
            inword = True
        i += 1
    end_segment(n)
    # attach each heredoc body to the segment whose span holds its opener: the segments
    # that ended before the opener are counted, and `starts` holds each segment's end
    for at, body in bodies:
        if segs:
            k = min(sum(1 for s in starts[1:] if s <= at), len(segs) - 1)
            segs[k].bodies.append(body)
    return segs


def clean(text: str) -> str:
    """A word stripped of the subshell and substitution marks around it."""
    t = text.lstrip("(`").removeprefix("$(").lstrip("(")
    if t.count(")") > t.count("("):
        t = t.rstrip(")")
    return t


def name(text: str) -> str:
    """A command word's name: its last path part, lower case, without `.exe`."""
    base = re.split(r"[\\/]", clean(text))[-1].lower()
    return base.removesuffix(".exe")


def command(seg: Segment) -> list[Word]:
    """The segment's words from its command word on.

    Leading assignments, prefixes, reserved words, a `case <word> in` head and a case pattern
    are skipped. A segment that opens with `{` has no command word of its own: the brace
    group's body is read again by `inner_commands`, and reading its words here as well would
    take the group's later commands for the arguments of its first.
    """
    words = list(seg.words)
    while words:
        w = words[0]
        t = clean(w.text)
        bare = "" if w.quoted else w.text.lower()
        assigns = ASSIGN.match(t) is not None and not w.quoted
        if not t or t == "&" or assigns or bare in RESERVED:
            words.pop(0)
        elif bare == "{":
            return []
        elif bare == "case":
            # `case <word> in`; the pattern after it is skipped on the next pass
            ins = [i for i, x in enumerate(words) if not x.quoted and x.text == "in"]
            words = words[ins[0] + 1 :] if ins else []
        elif is_case_pattern(words):
            words.pop(0)
        elif name(t) in PREFIXES:
            words.pop(0)
            while words and words[0].text.startswith("-"):
                words.pop(0)
        else:
            break
    return close_group(words)


def is_case_pattern(words: list[Word]) -> bool:
    """Whether the first word is a case pattern such as `x)` or `(*.py)`.

    It is a word with a `)` it does not open, an opening `(` aside, followed by a word that
    is neither an option nor a redirection: after the `)` that closes a subshell only a
    redirection or a separator may follow.
    """
    if len(words) < 2 or words[0].quoted or words[1].text.startswith("-"):
        return False
    t = words[0].text
    t = t[1:] if t.startswith("(") else t
    return t.count(")") > t.count("(") and not REDIRECT.match(words[1].text)


#: Each closing mark with its opening mark.
GROUP_MARKS = ((")", "("), ("}", "{"))


def close_group(words: list[Word]) -> list[Word]:
    """The words with each `)` or `}` beyond the groups they open cut from the last word.

    A subshell or a substitution split at a separator leaves its closing mark glued to the
    segment's last word, as in `git add .)`. The command word is counted as `name` reads
    it, without its opening marks, and a quoted word is neither counted nor cut.
    """
    if not words:
        return words
    texts = [clean(words[0].text)] + [w.text for w in words[1:] if not w.quoted]
    joined = "".join(texts)
    out = list(words)
    for closer, opener in GROUP_MARKS:
        excess = joined.count(closer) - joined.count(opener)
        while excess > 0 and out and not out[-1].quoted:
            t = out[-1].text
            if not t.endswith(closer):
                break
            t = t[:-1]
            excess -= 1
            if t or len(out) == 1:
                out[-1] = Word(t, False)
            else:
                out.pop()
    return out


def match_close(raw: str, start: int, opener: str, closer: str) -> int:
    """The index of the mark that closes a group opened before `start`, or the text's end.

    Marks inside single quotes are skipped.
    """
    depth, single = 1, False
    for j in range(start, len(raw)):
        c = raw[j]
        if single:
            single = c != "'"
        elif c == "'":
            single = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return j
    return len(raw)


def inner_commands(raw: str, backticks: bool = True) -> list[str]:
    """The commands a segment holds inside it, each read again under any command word.

    They are the payload of each command substitution `$( ... )`, the body of each brace
    group or script block `{ ... }` (a `${...}` variable excepted) and, when `backticks` is
    true, the payload of each backtick pair. Text in single quotes is not read, a brace
    in double quotes is text, and a mark after the escape character (a backslash, or the
    backtick of PowerShell) is text. A group left open runs to the segment's end, as a
    group split at a separator does.
    """
    return [raw[i:j] for i, j in inner_spans(raw, backticks)]


def inner_spans(raw: str, backticks: bool = True) -> list[tuple[int, int]]:
    """The start and end in `raw` of each command `inner_commands` reads, blank ones left out."""
    escape = "\\" if backticks else "`"
    out: list[tuple[int, int]] = []
    single = double = False
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if single:
            single = ch != "'"
        elif ch == escape:
            i += 2
            continue
        elif ch == "'" and not double:
            single = True
        elif ch == '"':
            double = not double
        elif ch == "$" and raw[i + 1 : i + 2] == "(":
            j = match_close(raw, i + 2, "(", ")")
            out.append((i + 2, j))
            i = j + 1
            continue
        elif ch == "{" and not double and raw[i - 1 : i] != "$":
            j = match_close(raw, i + 1, "{", "}")
            out.append((i + 1, j))
            i = j + 1
            continue
        elif ch == "`" and backticks:
            j = raw.find("`", i + 1)
            j = n if j < 0 else j
            out.append((i + 1, j))
            i = j + 1
            continue
        i += 1
    return [(i, j) for i, j in out if raw[i:j].strip()]


def with_bodies(text: str, bodies: list[str]) -> str:
    """The text with each heredoc body put back after the line of its opener, in order.

    `lex` lifts the bodies off a segment's `raw`, so a command read from inside it would
    find each opener with no body, and would take the lines after the opener for its body.
    Each body is put back with its delimiter, so the text reads as the shell reads it.
    """
    if not bodies:
        return text
    queue = list(bodies)
    out: list[str] = []
    for line in text.split("\n"):
        out.append(line)
        for m in HEREDOC.finditer(line):
            body = queue.pop(0) if queue else ""
            if body:
                out.append(body)
            out.append(m.group(3))
    return "\n".join(out)


def inner_texts(seg: Segment, backticks: bool = True) -> list[str]:
    """The commands `inner_commands` reads in a segment, each with its heredoc bodies.

    The segment's bodies follow its openers in order, so the openers before a command's
    span and inside it say which of the bodies are its own.
    """
    out: list[str] = []
    for i, j in inner_spans(seg.raw, backticks):
        before = sum(1 for _ in HEREDOC.finditer(seg.raw[:i]))
        inside = sum(1 for _ in HEREDOC.finditer(seg.raw[i:j]))
        out.append(with_bodies(seg.raw[i:j], seg.bodies[before : before + inside]))
    return out


def is_python(text: str) -> bool:
    """Whether a word names a Python interpreter (python, python3.12, pythonw, py)."""
    return PYTHON_NAME.fullmatch(name(text)) is not None


def resolve(cwd: str, path: str) -> str:
    """A path resolved against the working directory, with an MSYS drive path made native."""
    p = str(Path(path.strip() or ".").expanduser())
    m = re.match(r"^/([a-zA-Z])(/.*)?$", p.replace("\\", "/"))
    if os.name == "nt" and m:
        p = m.group(1).upper() + ":" + (m.group(2) or "/")
    if not cwd or Path(p).is_absolute() or re.match(r"^[a-zA-Z]:", p):
        return os.path.normpath(p)
    return os.path.normpath(Path(cwd) / p)


def repo_top(path: str) -> Path | None:
    """The root of the git working tree that holds `path`, or None."""
    if not path:
        return None
    p = Path(path)
    try:
        return next((d for d in (p, *p.parents) if (d / ".git").exists()), None)
    except OSError:
        return None


def same_path(a: Path, b: Path) -> bool:
    """Whether two paths name the same folder, case folded where the platform folds it."""
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(
            os.path.realpath(b)
        )
    except (OSError, ValueError):
        return False


def in_public(path: str) -> bool:
    """Whether `path` lies in the public repository's working tree.

    The tree is public when its root holds the marker `lib/MANIFEST.md`, or when it is this
    file's own tree and the rules file names the repository "public".
    """
    top = repo_top(path)
    if top is None:
        return False
    try:
        if top.joinpath(*PUBLIC_MARKER).is_file():
            return True
    except OSError:
        pass
    return RULES.repository.lower() == "public" and same_path(top, ROOT)


def rel_from(root: Path | None, cwd: str, path: str) -> str | None:
    """A path relative to the tree root `root`, with forward slashes, or None.

    A caller that resolves many paths finds the root once with `repo_top`, a walk up the
    folders, and resolves each path by string work alone.
    """
    if root is None:
        return None
    try:
        rel = os.path.relpath(resolve(cwd, path), root)
    except ValueError:
        return None
    return rel.replace("\\", "/")


# --- the rules ----------------------------------------------------------------------------


def evaluate(cmd: str, cwd: str, ctx: Context, depth: int = 0) -> None:
    """Read a command and add every finding to `ctx`; a wrapper's payload is read again."""
    segs = lex(cmd)
    for k, seg in enumerate(segs):
        words = command(seg)
        if not words:
            # a brace group's segment has no command word, and its body is read below
            if depth < MAX_DEPTH:
                for text in inner_texts(seg, ctx.backticks):
                    evaluate(text, cwd, ctx, depth + 1)
            continue
        head = name(words[0].text)
        piped = seg.sep.strip() == "|"
        prev = segs[k - 1] if k > 0 else None
        stdin = list(seg.bodies)
        if piped and prev is not None:
            stdin += prev.payloads + prev.bodies
        # the rules on one segment, each failing open on its own
        for rule in (
            rule_text,
            rule_git,
            rule_kill,
            rule_junction,
            rule_wmic,
            rule_gh,
            rule_python_heredoc,
            rule_heavy,
        ):
            try:
                rule(seg, words, head, cwd, ctx, stdin, segs, k)
            except Exception as e:  # noqa: PERF203 - each rule fails open on its own
                rule_name = getattr(rule, "__name__", "rule")
                record("guard", "warn", rule_name, f"{type(e).__name__}: {e}")
        if head in CD_WORDS:
            args = [
                w.text
                for w in words[1:]
                if not w.text.startswith("-") and w.text.lower() != "/d"
            ]
            if args:
                cwd = resolve(cwd, args[0])
        if depth < MAX_DEPTH and head in SHELLS:
            for text in [*seg.payloads, wrapped_rest(head, words), *stdin]:
                if text:
                    evaluate(text, cwd, ctx, depth + 1)
        if depth < MAX_DEPTH:
            for text in inner_texts(seg, ctx.backticks):
                evaluate(text, cwd, ctx, depth + 1)
    rule_kill_pipe(segs, ctx)


def wrapped_rest(head: str, words: list[Word]) -> str:
    """The words after a wrapper's own options, rejoined as a command, quoted words requoted."""
    rest = words[1:]
    while rest:
        t = rest[0].text
        if t.lower() in WRAPPER_VALUE_OPTS and len(rest) > 1:
            rest = rest[2:]
        elif not rest[0].quoted and (
            t.startswith("-") or (head == "cmd" and t.startswith("/"))
        ):
            rest = rest[1:]
        else:
            break
    out = []
    for w in rest:
        if w.quoted:
            q = "'" if "'" not in w.text else '"'
            out.append(q + w.text + q)
        else:
            out.append(w.text)
    return " ".join(out)


def rule_text(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 2: MPLBACKEND=Agg set on a segment that is not only read or printed."""
    if head in READ_ONLY:
        return
    if MPL_AGG.search(seg.raw):
        ctx.add(
            "block",
            "mplbackend-agg",
            "MPLBACKEND=Agg is blocked: it suppresses the inline backend and strips every "
            "stored figure from a notebook run.",
        )


@dataclass
class GitCall:
    """A git call after its global options: the working directory, configs, subcommand, args."""

    cwd: str
    config: list[str]
    sub: str
    args: list[Word]


def parse_git(words: list[Word], cwd: str) -> GitCall | None:
    """The git call of a segment whose command word is git, or None without a subcommand."""
    config: list[str] = []
    i = 1
    while i < len(words):
        t = words[i].text
        key, eq, val = t.partition("=")
        if key in GIT_VALUE_OPTS:
            if eq:
                v = val
            elif i + 1 < len(words):
                v = words[i + 1].text
                i += 1
            else:
                return None
            if key == "-C":
                cwd = resolve(cwd, v)
            elif key == "-c":
                config.append(v)
            elif key == "--work-tree":
                cwd = resolve(cwd, v)
            elif key == "--git-dir":
                gd = resolve(cwd, v)
                cwd = str(Path(gd).parent) if Path(gd).name == ".git" else gd
            i += 1
            continue
        if t.startswith("-"):
            i += 1
            continue
        return GitCall(cwd, config, t.lower(), words[i + 1 :])
    return None


def short_flags(args: list[Word], stop: str = "") -> set[str]:
    """The single-letter flags of the option words, a cluster read up to a value-taking letter."""
    flags: set[str] = set()
    for w in args:
        t = w.text
        if t == "--":
            break
        if re.fullmatch(r"-[A-Za-z]+", t):
            for ch in t[1:]:
                flags.add(ch)
                if ch in stop:
                    break
    return flags


def positionals(
    args: list[Word], value_opts: frozenset[str] = frozenset()
) -> list[str]:
    """The non-option words, every word after `--` included."""
    out: list[str] = []
    skip = False
    after = False
    for w in args:
        t = w.text
        if skip:
            skip = False
            continue
        if after or not t.startswith("-") or t == "-":
            out.append(t)
        elif t == "--":
            after = True
        elif t in value_opts:
            skip = True
    return out


def rule_git(
    seg: Segment,
    words: list[Word],
    head: str,
    cwd: str,
    ctx: Context,
    stdin: list[str],
    segs: list[Segment],
    k: int,
) -> None:
    """Rules 1, 3, 4, 8, 9 and 13 on a git call."""
    if head != "git":
        return
    call = parse_git(words, cwd)
    if call is None:
        return
    sub, args, gcwd = call.sub, call.args, call.cwd
    longs = {a.text.split("=")[0] for a in args if a.text.startswith("--")}
    if sub in ("add", "stage"):
        check_add(args, gcwd, ctx, segs, k)
    if sub in ("commit", "commit-tree", "merge", "tag"):
        texts = ["\n".join(s.raw for s in segs)]
        texts += [b for s in segs for b in s.bodies]
        texts += message_files(args, gcwd, segs, k)
        if any(TRAILER.search(t) for t in texts):
            ctx.add(
                "block",
                "claude-trailer",
                "A Claude authorship trailer is blocked in a commit, merge or tag message. "
                "The guard reads the -m text, heredocs and a -F file. Remove the trailer."
                + RULES.note("claude-trailer"),
            )
    if sub in ("commit", "merge", "am", "rebase", "cherry-pick", "revert"):
        hooks_off = any(c.lower().startswith("core.hookspath") for c in call.config)
        no_verify = "--no-verify" in longs or (
            sub == "commit" and "n" in short_flags(args, stop="mFcCt")
        )
        if hooks_off or no_verify:
            enforced = RULES.commit_hook_rules or "the repository's commit rules"
            ctx.add(
                "ask",
                "hook-bypass",
                "This git call skips the repository's commit hooks (--no-verify, or "
                f"core.hooksPath set on the call). The hooks enforce {enforced}, so the "
                "participant decides." + RULES.note("hook-bypass"),
            )
    if sub == "config":
        check_hookspath(args, ctx)
    if sub == "clean":
        flags = short_flags(args, stop="e")
        dry = "n" in flags or "--dry-run" in longs
        if ({"x", "X"} & flags) and not dry:
            folders = join_names(RULES.ignored_only_copy)
            where = (
                f"the ignored files under {folders}" if folders else "an ignored file"
            )
            ctx.add(
                "block",
                "clean-ignored",
                "git clean with -x or -X is blocked: it deletes ignored files, and "
                f"{where} can be the only copy. Delete named files by path, or list what "
                "would go with git clean -n first." + RULES.note("clean-ignored"),
            )
    if sub == "worktree" and args and args[0].text == "remove":
        rest = args[1:]
        force = "--force" in longs or "f" in short_flags(rest)
        paths = positionals(rest)
        if force and paths:
            check_worktree(resolve(gcwd, paths[0]), ctx)


#: The pathspecs that name a whole tree.
BROAD_PATHSPECS = ("", ".", ":", ":/", ":/*", ":(top)")
#: The commands whose output piped into `--pathspec-from-file=-` the guard can read.
ECHO_WORDS = frozenset({"echo", "printf", "write-output"})
CAT_WORDS = frozenset({"cat", "type", "get-content", "gc"})
#: The escapes `printf` and `echo -e` turn into a line break or a tab, which the guard expands.
ESCAPES = {"n": "\n", "r": "\r", "t": "\t"}
#: The escapes git decodes in a double-quoted line of a pathspec file, besides octal.
C_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}
#: The printf conversion that prints each argument on a line of its own.
PRINTF_LINE = "%s\\n"
#: The long pathspec magic words that keep a path's meaning; `top` reads it from the root.
PLAIN_MAGIC = frozenset({"top", "literal", "icase", "glob"})


def under(rel: str, folder: str) -> bool:
    """Whether a forward-slash, lower-case relative path is the folder or lies under it."""
    rel = rel.rstrip("/")
    return rel == folder or rel.startswith(folder + "/")


def expand_escapes(text: str) -> str | None:
    r"""The text with `\n`, `\r`, `\t` and `\0` expanded, or None with any other escape.

    A `\0` followed by a digit is an octal escape, and `\\` can stand for one backslash or
    two after the shell's quoting, so both make the text unknown.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1 : i + 2]
        if nxt in ESCAPES:
            out.append(ESCAPES[nxt])
        elif nxt == "0" and not text[i + 2 : i + 3].isdigit():
            out.append("\0")
        else:
            return None
        i += 2
    return "".join(out)


def echoed(head: str, words: list[str]) -> list[str] | None:
    r"""What `echo`, `printf` or `Write-Output` prints for its words, or None when unknown.

    Each word counts both as written and with its escapes expanded, because `echo` expands
    them in some shells and not in others. A word the shell rewrites, or one with an escape
    `expand_escapes` does not read, is unknown, and so is a printf format with a conversion
    other than `%s\n`, which can join two arguments into one path.
    """
    if any(SHELL_EXPANDS.search(w) for w in words):
        return None
    if head == "printf" and words and "%" in words[0].replace(PRINTF_LINE, ""):
        return None
    out: list[str] = []
    for w in words:
        expanded = expand_escapes(w)
        if expanded is None:
            return None
        out += [w, expanded] if expanded != w else [w]
    return out


def unquote_line(line: str) -> str | None:
    r"""A pathspec file's line with git's C-style quotes removed, or None when unknown.

    Git reads a line in double quotes as a C string, so `\"`, `\\`, `\n`, `\t` and an octal
    escape such as `\056` are decoded. Any other escape makes the line unknown.
    """
    s = line.strip()
    if not (len(s) >= 2 and s[0] == s[-1] == '"'):
        return s
    body, out, i = s[1:-1], [], 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1 : i + 2]
        octal = re.match(r"[0-3][0-7]{2}", body[i + 1 : i + 4])
        if octal:
            out.append(chr(int(octal.group(0), 8)))
            i += 4
        elif nxt in C_ESCAPES:
            out.append(C_ESCAPES[nxt])
            i += 2
        else:
            return None
    return "".join(out)


def pathspec_target(cwd: str, spec: str, root: Path | None) -> str | None:
    """The forward-slash, lower-case path a pathspec names from its tree's root, or None.

    A leading magic is read as git reads it: `:/` and `:(top)` name a path from the root,
    and `literal`, `icase` and `glob` leave the path as written. None means the guard
    cannot know what the pathspec selects: an exclude (`:!`, `:^`, `:(exclude)`), an
    `attr` or another magic word, a substitution, a variable or a backtick. `root` is the
    root of the tree that holds `cwd`, as `repo_top` finds it.
    """
    if SHELL_EXPANDS.search(spec):
        return None
    top, rest = False, spec
    if spec.startswith(":("):
        close = spec.find(")")
        if close < 0:
            return None
        for magic in spec[2:close].split(","):
            word = magic.strip().lower()
            if word not in PLAIN_MAGIC:
                return None
            top = top or word == "top"
        rest = spec[close + 1 :]
    elif spec.startswith(":"):
        i = 1
        while i < len(spec) and spec[i] in "/!^":
            if spec[i] != "/":
                return None
            top, i = True, i + 1
        if spec[i : i + 1] == ":":
            i += 1
        rest = spec[i:]
    if top:
        rel = os.path.normpath(rest.replace("\\", "/") or ".")
    else:
        rel = rel_from(root, cwd, rest) or rest
    return rel.replace("\\", "/").lower()


def pathspec_file(args: list[Word]) -> str | None:
    """The file named by `--pathspec-from-file`, or None without the option."""
    for i, w in enumerate(args):
        t = w.text
        if t.startswith("--pathspec-from-file="):
            return t.split("=", 1)[1]
        if t == "--pathspec-from-file":
            return args[i + 1].text if i + 1 < len(args) else ""
    return None


def piped_stdin(cwd: str, segs: list[Segment], k: int) -> str | None:
    """The text a segment reads on stdin, or None when the guard cannot know it.

    The text is the segment's own heredoc bodies, and, when the segment is piped from
    `echo`, `printf`, `Write-Output`, `cat`, `type` or `Get-Content`, what that command
    prints: its words read by `echoed`, its heredoc bodies, or the files it names. A named
    file that is not a readable file at decision time makes the text unknown, and so do
    named files whose text, with the bodies, is longer than `READ_LIMIT` together, and a
    heredoc body the shell would rewrite.
    """
    seg = segs[k]
    texts = list(seg.bodies)
    prev = segs[k - 1] if k > 0 else None
    if prev is not None and seg.sep.strip() == "|":
        pw = command(prev)
        head = name(pw[0].text) if pw else ""
        if head not in ECHO_WORDS | CAT_WORDS:
            return None
        texts += prev.bodies
        words = [w.text for w in pw[1:] if not w.text.startswith(("-", "<", ">"))]
        if head in CAT_WORDS:
            files = [resolve(cwd, t) for t in words]
            if not all(Path(p).is_file() for p in files):
                return None
            for p in files:
                whole = read_whole(p)
                if whole is None or sum(map(len, texts)) + len(whole) > READ_LIMIT:
                    return None
                texts.append(whole)
        else:
            printed = echoed(head, words)
            if printed is None:
                return None
            texts += printed
        if any(SHELL_EXPANDS.search(b) for b in prev.bodies):
            return None
    if any(SHELL_EXPANDS.search(b) for b in seg.bodies):
        return None
    return "\n".join(texts) if texts else None


def pathspec_paths(
    args: list[Word], cwd: str, segs: list[Segment], k: int
) -> list[str] | None:
    """The paths a `--pathspec-from-file` file lists, [] without one, None when unknown.

    The file is unknown when it cannot be read whole: absent, unreadable, or longer than
    `READ_LIMIT`. It is unknown too when it lists more than `PATHSPEC_LIMIT` paths, more
    than the guard checks within its hook timeout. Line breaks and NUL characters both
    separate paths, which covers what git reads with and without `--pathspec-file-nul`. A
    line in double quotes is decoded by `unquote_line`, and a line it cannot decode makes
    the file unknown.
    """
    nm = pathspec_file(args)
    if nm is None:
        return []
    if nm == "-":
        text = piped_stdin(cwd, segs, k)
    else:
        path = resolve(cwd, nm) if nm else ""
        text = read_whole(path) if path and Path(path).is_file() else None
    if text is None:
        return None
    out: list[str] = []
    for line in re.split(r"\0|\r\n|\n|\r", text):
        s = unquote_line(line)
        if s is None:
            return None
        out += [part for part in s.split("\0") if part]
        if len(out) > PATHSPEC_LIMIT:
            return None
    return out


def check_add(
    args: list[Word], cwd: str, ctx: Context, segs: list[Segment], k: int
) -> None:
    """Rules 1 and 4 on the arguments of a `git add`, a pathspec file's paths included."""
    flags = short_flags(args)
    longs = {a.text.split("=")[0] for a in args if a.text.startswith("--")}
    named = positionals(args, frozenset({"--chmod", "--pathspec-from-file"}))
    listed = pathspec_paths(args, cwd, segs, k)
    paths = named + (listed or [])
    root = repo_top(cwd)
    targets = {p: pathspec_target(cwd, p, root) for p in paths}
    rels = {
        p: t if t is not None else p.replace("\\", "/").lower()
        for p, t in targets.items()
    }
    force = "f" in flags or "--force" in longs
    broad = (
        "A" in flags
        or "--all" in longs
        or listed is None
        or any(
            p.rstrip("/\\") in BROAD_PATHSPECS
            or targets[p] is None
            or rels[p] in ("", ".")
            or re.search(r"[*?\[]", p)
            for p in paths
        )
    )
    if in_public(cwd):
        if broad:
            ctx.add(
                "block",
                "public-broad-add",
                "git add -A, --all, `.`, a glob, or a pathspec file the guard cannot read "
                "is blocked in the public repository: its working tree carries untracked "
                "files that are never published. Stage each path by name, and write a "
                "pathspec file before the call that reads it."
                + RULES.note("public-broad-add"),
            )
        for p in paths:
            if under(rels[p], ACADEMIC_DIR):
                ctx.add(
                    "block",
                    "public-sources",
                    f"Staging {p} is blocked: the public repository cites the sources under "
                    f"{ACADEMIC_DIR}/ by their bibliography url and never publishes them."
                    + RULES.note("public-sources"),
                )
            elif force and under(rels[p], SOURCES_DIR):
                ctx.add(
                    "block",
                    "public-sources",
                    f"A forced add of {p} is blocked: under {SOURCES_DIR}/ the public "
                    "repository publishes only what its ignore rules let through. Stage a "
                    "source without -f, or leave it out."
                    + RULES.note("public-sources"),
                )
    never = [f.replace("\\", "/").strip("/").lower() for f in RULES.never_committed]
    never = [f for f in never if f]
    if not never:
        return
    hit = force and broad
    for p in paths:
        rel, raw = rels[p], p.replace("\\", "/").lower()
        holds = rel in ("", ".") or any(
            f.startswith(rel.rstrip("/") + "/") for f in never
        )
        if any(f in rel or f in raw for f in never) or (force and holds):
            hit = True
    if hit:
        folders = join_names([f + "/" for f in never])
        ctx.add(
            "block",
            "never-committed-add",
            f"{folders} {'is' if len(never) == 1 else 'are'} never committed in this "
            "repository. Leave it out of the staged paths, and force-add only named files."
            + RULES.note("never-committed-add"),
        )


def message_files(args: list[Word], cwd: str, segs: list[Segment], k: int) -> list[str]:
    """The text of each message file named by -F or --file, and of the file piped into `-F -`."""
    names: list[str] = []
    for i, w in enumerate(args):
        t = w.text
        if t in ("-F", "--file") and i + 1 < len(args):
            names.append(args[i + 1].text)
        elif t.startswith("--file="):
            names.append(t.split("=", 1)[1])
        elif re.fullmatch(r"-[A-Za-z]*F.+", t):
            names.append(t[t.index("F") + 1 :])
    out: list[str] = []
    for nm in names:
        if nm == "-":
            prev = segs[k - 1] if k > 0 else None
            if prev is not None and segs[k].sep.strip() == "|":
                pw = command(prev)
                if pw and name(pw[0].text) in ("cat", "type", "get-content", "gc"):
                    out += [read_text(resolve(cwd, w.text)) for w in pw[1:]]
            continue
        out.append(read_text(resolve(cwd, nm)))
    return out


def read_text(path: str) -> str:
    """A file's text, at most `READ_LIMIT` characters, or "" when it cannot be read."""
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            return f.read(READ_LIMIT)
    except OSError:
        return ""


def read_whole(path: str) -> str | None:
    """A file's whole text, or None when it cannot be read or is longer than `READ_LIMIT`."""
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            text = f.read(READ_LIMIT + 1)
    except OSError:
        return None
    return None if len(text) > READ_LIMIT else text


def check_hookspath(args: list[Word], ctx: Context) -> None:
    """Rule 13 on a `git config` that unsets or redirects core.hooksPath."""
    texts = [a.text for a in args]
    keys = [i for i, t in enumerate(texts) if t.lower() == "core.hookspath"]
    if not keys:
        return
    i = keys[0]
    unset = any(t in ("--unset", "--unset-all") for t in texts)
    value = texts[i + 1] if i + 1 < len(texts) else None
    redirected = value is not None and not value.rstrip("/\\").endswith(".githooks")
    if unset or redirected:
        ctx.add(
            "ask",
            "hookspath-config",
            "This git config call unsets core.hooksPath or points it away from .githooks, "
            "which turns off the commit backstop. The participant decides.",
        )


def entry_kind(e: os.DirEntry[str]) -> str:
    """A folder entry's kind: "link" for a junction or a link to a folder, "dir", or "other"."""
    try:
        if not e.is_dir(follow_symlinks=False):
            return "link" if e.is_symlink() and Path(e.path).is_dir() else "other"
        attrs = getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return "other"
    return "link" if e.is_symlink() or attrs & REPARSE_ATTR else "dir"


def has_reparse_point(top: str) -> tuple[bool | None, str]:
    """Whether a folder holds a junction or a symbolic link to a folder, and the first found.

    The search does not follow links, skips `.git`, and stops at the time or folder budget,
    when the answer is None (unknown).
    """
    t0 = time.monotonic()
    stack = [top]
    seen = 0
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            kind = entry_kind(e)
            if kind == "link":
                return True, e.path
            if kind == "dir" and e.name != ".git":
                stack.append(e.path)
        seen += 1
        if seen > REPARSE_BUDGET_DIRS or time.monotonic() - t0 > REPARSE_BUDGET_S:
            return None, ""
    return False, ""


def check_worktree(path: str, ctx: Context) -> None:
    """Rule 8 on a forced `git worktree remove` of `path`."""
    if not Path(path).is_dir():
        return
    found, where = has_reparse_point(path)
    if found is False:
        return
    if found:
        message = (
            f"git worktree remove --force is blocked: the worktree holds a junction or "
            f"symbolic link ({where}), and a forced remove deletes through it into the "
            "target. Remove the link first (rmdir <link> on Windows, rm <link> on POSIX), "
            "then retry."
        )
    else:
        message = (
            "git worktree remove --force is blocked: the guard could not confirm within "
            "its budget that the worktree holds no junction or symbolic link. List them "
            "(dir /AL /S, or find -type l), remove each, then retry."
        )
    ctx.add(
        "block",
        "worktree-force-reparse",
        message + RULES.note("worktree-force-reparse"),
    )


KILL_MESSAGE = (
    "Ending processes by image name is blocked: it also ends the participant's notebook "
    "kernels and other sessions' runs. Find the process with Get-CimInstance Win32_Process "
    "(Name, ProcessId, CommandLine) and stop only the PIDs you started, with "
    "Stop-Process -Id <pid> or taskkill /PID <pid>."
)


def rule_kill(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 6 on one segment: taskkill /IM, Stop-Process -Name, pkill and killall of python."""
    texts = [w.text.lower() for w in words[1:]]

    def pythonish(t: str) -> bool:
        return t.startswith(("python", "py*", "py.exe", "*")) or t == "py"

    hit = False
    if head == "taskkill":
        for i, t in enumerate(texts):
            if t in ("/im", "-im") and i + 1 < len(texts) and pythonish(texts[i + 1]):
                hit = True
            if t in ("/fi", "-fi") and i + 1 < len(texts):
                f = texts[i + 1]
                hit |= "imagename" in f and "py" in f
    elif head in KILL_VERBS:
        for i, t in enumerate(texts):
            if t.startswith(("-name", "-processname", "-n")) and ":" in t:
                hit |= pythonish(t.split(":", 1)[1])
            elif t in ("-name", "-processname", "-n", "-na") and i + 1 < len(texts):
                hit |= any(pythonish(v.strip()) for v in texts[i + 1].split(","))
    elif head in ("pkill", "killall"):
        hit = any("python" in t for t in texts if not t.startswith("-"))
    if hit:
        ctx.add("block", "kill-by-name", KILL_MESSAGE)


def rule_kill_pipe(segs: list[Segment], ctx: Context) -> None:
    """Rule 6 on a pipe chain: a process query that names python, piped into a stop."""
    chain: list[Segment] = []
    for seg in [*segs, Segment(sep=";")]:
        if seg.sep.strip() == "|" and chain:
            chain.append(seg)
            continue
        if len(chain) > 1:
            first = command(chain[0])
            heads = [name(command(s)[0].text) for s in chain if command(s)]
            text = "\n".join(s.raw for s in chain).lower()
            query = bool(first) and name(first[0].text) in (
                "get-process", "gps", "ps", "get-ciminstance", "gcim", "get-wmiobject", "gwmi"
            )  # fmt: skip
            stop = any(h in KILL_VERBS or h == "remove-ciminstance" for h in heads) or (
                re.search(r"\b(kill|terminate)\b|\.kill\(", text) is not None
            )
            narrowed = re.search(
                r"commandline|\.id\s*-(eq|in)\b|processid\s*-(eq|in)\b|-id\s+\d", text
            )
            if query and stop and "python" in text and not narrowed:
                ctx.add("block", "kill-by-name", KILL_MESSAGE)
        chain = [seg]


def rule_junction(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 7: a junction or a symbolic link created with a data/ path among its words."""
    texts = [w.text for w in words[1:]]
    low = [t.lower() for t in texts]
    link = False
    if head == "mklink":
        link = any(t in ("/j", "/d") for t in low)
    elif head in ("new-item", "ni"):
        for i, t in enumerate(low):
            v = (
                t.split(":", 1)[1]
                if ":" in t
                else (low[i + 1] if i + 1 < len(low) else "")
            )
            if t.split(":")[0] in ("-itemtype", "-type", "-it"):
                link |= v.strip() in ("junction", "symboliclink")
    elif head == "ln":
        link = any(re.fullmatch(r"-[a-z]*s[a-z]*", t) for t in low)
    if link and any(re.search(r"(^|[\\/:])data([\\/]|$)", t) for t in low):
        ctx.add(
            "block",
            "junction-into-data",
            "Creating a junction or symbolic link that names a data/ folder is blocked: a "
            "later forced worktree remove deletes through it into the main tree's data. "
            "Copy the needed files into the worktree instead.",
        )


def rule_wmic(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 11: wmic."""
    if head == "wmic":
        ctx.add(
            "block",
            "wmic",
            "wmic is blocked: it is deprecated on Windows 11 and may be absent. Use "
            "Get-CimInstance Win32_Process (or Get-CimInstance <class>) instead.",
        )


def git_config_text(cwd: str) -> str | None:
    """The text of the git config of the tree that holds `cwd`, a worktree's included."""
    top = repo_top(cwd)
    if top is None:
        return None
    config = top / ".git" / "config"
    if not config.is_file():
        gitfile = read_text(str(top / ".git"))
        m = re.match(r"gitdir:\s*(.+)", gitfile.strip())
        if not m:
            return None
        common = Path(resolve(str(top), m.group(1))).parents[1]
        config = common / "config"
    return read_text(str(config))


def remote_owners(cwd: str) -> set[str] | None:
    """The GitHub owners of the git remotes of the tree that holds `cwd`, or None."""
    text = git_config_text(cwd)
    if text is None:
        return None
    return {m.group(1).lower() for m in GITHUB_OWNER.finditer(text)}


def origin_owner(cwd: str) -> str | None:
    """The GitHub owner of the `origin` remote of the tree that holds `cwd`, or None."""
    section = ""
    for line in (git_config_text(cwd) or "").splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.lower().replace(" ", "")
        elif section == '[remote"origin"]' and re.match(r"url\s*=", s, re.I):
            m = GITHUB_OWNER.search(s)
            return m.group(1).lower() if m else None
    return None


def allowed_owners(cwd: str) -> frozenset[str]:
    """The owners a post may go to: the rules file's, and in the public tree its origin's."""
    owners = {o.lower() for o in RULES.allowed_owners}
    if in_public(cwd):
        origin = origin_owner(cwd)
        if origin:
            owners.add(origin)
    return frozenset(owners)


def rule_gh(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 10: a gh call that posts to a repository of an owner not allowed."""
    if head != "gh" or len(words) < 2:
        return
    texts = [w.text for w in words[1:]]
    allowed = allowed_owners(cwd)

    def message(owners: str) -> str:
        dest = RULES.defect_destination.strip()
        remedy = (
            f"A defect in a reference codebase stays in {dest}."
            if dest
            else "Posting on another repository is for the participant."
        )
        return (
            "Posting an issue, pull request, comment or review to a repository of another "
            f"owner is blocked. {remedy}{RULES.note('upstream-post')} Owner named: {owners}."
        )

    if texts[0] in ("issue", "pr") and len(texts) > 1:
        if texts[1] not in ("create", "comment", "review", "edit", "reopen", "close"):
            return
        repo = None
        for i, t in enumerate(texts):
            if t in ("-R", "--repo") and i + 1 < len(texts):
                repo = texts[i + 1]
            elif t.startswith("--repo="):
                repo = t.split("=", 1)[1]
            elif t.startswith("-R") and len(t) > 2:
                repo = t[2:]
        owners: set[str] | None
        if repo:
            parts = [p for p in re.split(r"[/:]", repo) if p]
            owners = {parts[-2].lower()} if len(parts) >= 2 else None
        else:
            urls = {m.group(1).lower() for t in texts for m in GITHUB_OWNER.finditer(t)}
            owners = urls or remote_owners(cwd)
        bad = sorted((owners or set()) - allowed)
        if bad:
            ctx.add("block", "upstream-post", message(", ".join(bad)))
        return
    if texts[0] == "api":
        rest = texts[1:]
        low = [t.lower() for t in rest]
        method = "GET"
        for i, t in enumerate(low):
            if t in ("-x", "--method") and i + 1 < len(low):
                method = low[i + 1].upper()
            elif t.startswith(("--method=", "-x")) and len(t) > 2 and t != "-x":
                method = t.split("=", 1)[-1].removeprefix("-x").upper()
        fields = any(
            t in ("-f", "-F", "--field", "--raw-field", "--input")
            or t.startswith(("--field=", "--raw-field=", "--input="))
            for t in rest
        )
        writes = method != "GET" or fields
        endpoint = next(iter(positionals(words[2:], GH_API_VALUE_OPTS)), "")
        if endpoint == "graphql":
            body = " ".join(rest).lower()
            if "mutation" in body and re.search(
                r"createissue|addcomment|createpullrequest|addpullrequestreview", body
            ):
                ctx.add("block", "upstream-post", message("unknown (graphql)"))
            return
        m = re.match(r"/?repos/([^/]+)/[^/]+/(issues|pulls)\b", endpoint)
        if m and writes:
            owner = m.group(1).lower()
            # gh's own placeholder for the current repository
            placeholder = "{" + "owner}"
            owners = remote_owners(cwd) if owner == placeholder else {owner}
            bad = sorted((owners or set()) - allowed)
            if bad:
                ctx.add("block", "upstream-post", message(", ".join(bad)))


def rule_python_heredoc(
    seg: Segment,
    words: list[Word],
    head: str,
    cwd: str,
    ctx: Context,
    stdin: list[str],
    *_: object,
) -> None:
    """Rule 12: a program fed to python on stdin that names a notebook and writes a file.

    The program is a heredoc or a here-string, read by python with no script or with `-`.
    Text piped into a named script is that script's data, and a program that only reads a
    notebook is allowed.
    """
    py = next((i for i, w in enumerate(words[:6]) if is_python(w.text)), None)
    if py is None or not reads_program_from_stdin(words[py + 1 :]):
        return
    if any(NOTEBOOK.search(t) and WRITES.search(t) for t in stdin):
        ctx.add(
            "block",
            "heredoc-notebook",
            "A heredoc or here-string fed to python that writes a notebook is blocked: the "
            "shell or the inline string can strip the script's backslashes, which corrupts "
            "LaTeX in the cells. Write the patch script with the Write tool, run it as a "
            "file, and edit the notebook by cell.",
        )


def reads_program_from_stdin(args: list[Word]) -> bool:
    """Whether python, given these arguments, reads its program from stdin."""
    skip = False
    for w in args:
        t = w.text
        if skip:
            skip = False
        elif re.fullmatch(r"\d*(<<?-?|>>?|[<>]&)", t):
            skip = True  # a redirection whose target is the next word
        elif re.match(r"\d*[<>]", t):
            continue  # a redirection with its target attached
        elif t == "-":
            return True
        elif t in ("-c", "-m") or not t.startswith("-"):
            return False
        elif t in ("-X", "-W"):
            skip = True
    return True


def heavy_globs(ctx: Context) -> list[str]:
    """The globs of `tools/heavy_entrypoints.txt`, read once per call; none when it is absent."""
    if ctx.heavy_globs is None:
        globs: list[str] = []
        for line in read_text(str(HEAVY_LIST)).splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                globs.append(s.replace("\\", "/").lower())
        ctx.heavy_globs = globs
    return ctx.heavy_globs


def is_heavy_script(path: str, globs: list[str]) -> bool:
    """Whether a resolved script path ends in a path one of the globs matches."""
    p = path.replace("\\", "/").lower()
    return any(
        fnmatch.fnmatchcase(p, g) or fnmatch.fnmatchcase(p, "*/" + g) for g in globs
    )


def rule_heavy(
    seg: Segment, words: list[Word], head: str, cwd: str, ctx: Context, *_: object
) -> None:
    """Rule 5, the detection: a test, notebook or clearing launch in this segment."""
    if head in READ_ONLY:
        return
    if LAUNCH.search(seg.bare) or head in ("pytest", "py.test"):
        ctx.heavy = True
        return
    if head == "jupyter" and len(words) > 1 and words[1].text.lower() == "nbconvert":
        ctx.heavy = True
        return
    py = next((i for i, w in enumerate(words) if is_python(w.text)), None)
    if py is None:
        return
    globs = heavy_globs(ctx)
    for w in words[py + 1 :]:
        t = w.text
        if t.lower().endswith(".py") and is_heavy_script(resolve(cwd, t), globs):
            ctx.heavy = True
            return


def check_ceiling(ctx: Context) -> None:
    """Rule 5, the count: the ceiling, and the margin for a background launch."""
    if not ctx.heavy:
        return
    n = python_count()
    if n is None:
        record("guard", "warn", "ceiling", "process count unavailable; launch allowed")
        return
    if n >= CEILING:
        ctx.add(
            "block",
            "ceiling",
            f"{n} Python interpreter processes are running, at the ceiling of {CEILING} "
            "(.claude/hooks/_limits.py, or LAB_PROCESS_CEILING when set). Launching a test "
            "suite, a notebook execution or a "
            "clearing script now risks exhausting the machine. Wait for running work to "
            "finish, or end orphaned processes by PID (find them with Get-CimInstance "
            "Win32_Process), then retry once.",
        )
        return
    if ctx.background:
        limit = CEILING - BACKGROUND_MARGIN
        if n > limit:
            ctx.add(
                "block",
                "ceiling-background",
                f"{n} Python interpreter processes are running. A heavy launch sent to the "
                f"background needs the count at {limit} or below (the ceiling {CEILING} less "
                f"a margin of {BACKGROUND_MARGIN}), because the next check cannot see what "
                "the background run will start. Wait, or run it in the foreground.",
            )
        else:
            ctx.add(
                "advise",
                "background-pid",
                "This heavy launch runs in the background. Record its PID when it starts, "
                "and end it by that PID only, if it is orphaned.",
            )


def decide(data: dict[str, object]) -> Context:
    """Every finding on one hook input."""
    tool_input = data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    cmd = str(tool_input.get("command", ""))
    cwd = str(data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or "")
    ctx = Context(
        background=bool(tool_input.get("run_in_background")),
        backticks=data.get("tool_name") != "PowerShell",
    )
    evaluate(cmd, cwd, ctx)
    try:
        check_ceiling(ctx)
    except Exception as e:
        sys.stderr.write(f"guard.py: process check failed ({e}); allowing the call\n")
        record("guard", "warn", "ceiling", f"process check failed: {e}")
    return ctx


def main() -> int:
    """Read the hook input, apply the rules, and report: exit 2 with messages, or exit 0."""
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception as e:
        sys.stderr.write(
            f"guard.py: could not parse hook input ({e}); allowing the call\n"
        )
        return 0
    if not isinstance(data, dict):
        sys.stderr.write("guard.py: hook input is not an object; allowing the call\n")
        return 0
    try:
        ctx = decide(data)
    except Exception as e:
        sys.stderr.write(f"guard.py: internal error ({e}); allowing the call\n")
        record("guard", "warn", "internal", f"{type(e).__name__}: {e}")
        return 0
    tool_input = data.get("tool_input")
    cmd = str(tool_input.get("command", "")) if isinstance(tool_input, dict) else ""
    session = str(data.get("session_id", ""))
    for f in ctx.findings:
        record("guard", f.kind, f.rule, cmd, session)
    blocks = [f.message for f in ctx.findings if f.kind == "block"]
    if blocks:
        sys.stderr.write("\n".join(blocks) + "\n")
        return 2
    asks = [f.message for f in ctx.findings if f.kind == "ask"]
    advice = [f.message for f in ctx.findings if f.kind == "advise"]
    out: dict[str, object] = {"hookEventName": "PreToolUse"}
    if asks:
        out["permissionDecision"] = "ask"
        out["permissionDecisionReason"] = " ".join(asks)
    if advice:
        out["additionalContext"] = " ".join(advice)
    if len(out) > 1:
        print(json.dumps({"hookSpecificOutput": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
