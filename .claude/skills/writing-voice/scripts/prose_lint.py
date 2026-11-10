#!/usr/bin/env python
"""Hard-rule lint for prose and teaching notebooks.

    python .claude/skills/writing-voice/scripts/prose_lint.py FILE [FILE ...]
    python .claude/skills/writing-voice/scripts/prose_lint.py --changed
    python .claude/skills/writing-voice/scripts/prose_lint.py --all [--json]
    python .claude/skills/writing-voice/scripts/prose_lint.py --list-rules
    python .claude/skills/writing-voice/scripts/prose_lint.py --hook < posttooluse.json

The lint states the absolute rules of AGENTS.md, the writing-voice skill and the
modify-example skill as line, cell and file checks on notebooks (.ipynb), LaTeX notes
(.tex), record and skill files (.md) and the bibliography (.bib). It is the prose
counterpart of .claude/hooks/guard.py. style_check.py is a separate, voice-relative
diagnostic that never sets a target, and the lint does not replace it.

Each rule is a record in RULES with its id, severity, file kinds, hook flag and source
pinpoint; `--list-rules` prints the registry. An error breaks a rule that the record
treats as settled. A warning is a candidate that a person reads. A warning is promoted
to an error only after a 20-hit sample shows a false-positive rate under 5 percent.

The cut-on-sight lexicon is imported by name from style_check.TELLS, so that each phrase
is defined once. The symbol rules R1 to R3 for .tex files come from
symbol_check.check_file. Both modules are loaded by file path from this directory, so
the hook works from any working directory.

Scope. FILES lints whole files. `--all` lints every notebook at the repository root, every
.tex file under notes/, and sources/references.bib. `--changed` lints the modified and
untracked files that git reports and, for a tracked file, reports only the lines or
cells that differ from HEAD. `--hook` reads the PostToolUse JSON on stdin and reports
only the region the tool call touched: the lines spanned by each new string of an Edit
or MultiEdit, the edited cell of a NotebookEdit together with the adjacency checks that
involve it, and the whole file for a Write. The whole-notebook structure rules (the
sandbox and the EDIT cells) and CANON-MONITORED do not run in the hook.

Exceptions. An inline pragma names an accepted exception on its own line or the line
before: `<!-- lint: allow ID -->`, `% lint: allow ID` or `# lint: allow ID`.
`lint: allow-file ID` anywhere in a file exempts the whole file from that rule. Quoted
source text is exempt from the lexical rules, because the source's words are quoted
verbatim. The path exemptions are stated in each rule's record.

Exit codes: 0 when clean or warnings only, 1 when at least one error is reported, 2 on a
usage or internal error. `--hook` always exits 0, prints nothing on a clean result, and
reports findings through hookSpecificOutput.additionalContext.

Standard library only. The environment variable PROSE_LINT_ROOT overrides the
repository root, for the test suite.
"""

from __future__ import annotations

import argparse, difflib, importlib.util, io, json, os, re, subprocess, sys, time
from collections import Counter, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator
    from types import ModuleType

    from typing_extensions import Unpack

    class _FindingKw(TypedDict, total=False):
        cell: int | None
        line: int | None
        span: tuple[int, int] | None
        excerpt: str
        note: str
        severity: str | None

    class _LocKw(TypedDict, total=False):
        cell: int | None
        line: int | None
        span: tuple[int, int] | None


HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
DEFAULT_ROOT = SKILL.parents[2]
HOOK_MAX_FINDINGS = 15


def repo_root() -> Path:
    r = os.environ.get("PROSE_LINT_ROOT")
    return Path(r).resolve() if r else DEFAULT_ROOT


# ------------------------------------------------------------------ sibling modules ------
def _load(name: str) -> ModuleType:
    """Load a sibling script by file path, so the import does not depend on the cwd."""
    path = HERE / (name + ".py")
    spec = importlib.util.spec_from_file_location("_prose_lint_" + name, str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]  # spec is None when the path is unreadable
    spec.loader.exec_module(mod)  # type: ignore[union-attr]  # spec and its loader may be None
    return mod


_MODULES: dict[str, ModuleType | None] = {}


def sibling(name: str) -> ModuleType | None:
    """Return the loaded module, or None with one stderr line when it cannot be loaded."""
    if name not in _MODULES:
        try:
            _MODULES[name] = _load(name)
        except Exception as e:  # a parallel edit may leave it broken
            sys.stderr.write(
                "prose_lint: could not load %s.py (%s); its rules are skipped\n"
                % (name, e)
            )
            _MODULES[name] = None
    return _MODULES[name]


# The style_check tells that the lint imports, by rule. Names must exist in TELLS.
TELL_NAMES = {
    "CUT-PHRASES": ["intensifiers and evaluation", "meta-commentary on the document"],
    "CUT-READER-ADDRESS": ["reader address"],
    "LEX-NETWORK-METAPHOR": ["metaphor for the network"],
    "LEX-TRANSMISSION-OWNER": ["generic transmission owner"],
}


def _tell_flags(pat: str) -> int:
    """The regex flags style_check.find_tells uses for a tell pattern."""
    return (
        re.I if not pat.startswith("^(And|But)") and "Participating" not in pat else 0
    )


_TELL_RX: dict[str, list[re.Pattern[str]]] = {}


def tell_patterns(rule_id: str) -> list[re.Pattern[str]]:
    """Compiled tell patterns for a rule; an empty list when style_check is unavailable."""
    if rule_id not in _TELL_RX:
        sc = sibling("style_check")
        out = []
        if sc is not None:
            table = dict(getattr(sc, "TELLS", []))
            for name in TELL_NAMES.get(rule_id, []):
                if name in table:
                    out.append(re.compile(table[name], _tell_flags(table[name])))
                else:
                    sys.stderr.write(
                        "prose_lint: style_check.TELLS has no tell named %r\n" % name
                    )
        _TELL_RX[rule_id] = out
    return _TELL_RX[rule_id]


# ------------------------------------------------------------------ the rule registry ----
Rule = namedtuple("Rule", "id severity kinds scope hook source message")
# kinds: nbmd notebook markdown, nbcode notebook code, nb whole notebook, tex, md, bib
# scope: line, cell or file. A file-scope finding is reported in the hook only for a Write.
# the public notebooks that carry the reservation margin psi
MARGIN_NB = r"(^|/)(workshop2_congestion_management|101_sandbox)\.ipynb$"
RULE_DOCS = r"^(\.claude/|notes/writing_profile/|CLAUDE\.md$)"  # files that state the rules quote the phrases
_R = [
    # notebook math and structure
    (
        "MATH-DELIM",
        "error",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1 (Conventions); AGENTS.md, Notation; modify-example SKILL.md, series conventions 3",
        "Notebook math is $...$ or $$...$$; replace \\(...\\) and \\[...\\].",
    ),
    (
        "INW-MISSING",
        "error",
        "nbmd",
        "cell",
        True,
        "CANON.md, section 1 (In words); modify-example SKILL.md, formulation 3",
        "Every numbered display equation is followed directly by an *In words.* blurb.",
    ),
    (
        "INW-DEFERRED",
        "warning",
        "nbmd",
        "cell",
        True,
        "modify-example SKILL.md, formulation 3",
        "A where-clause sits between the numbered display and its *In words.* blurb; move the definitions into the blurb.",
    ),
    (
        "INW-ORPHAN",
        "warning",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1 (*In words.* belongs to equations alone)",
        "*In words.* belongs to an equation; this blurb follows no display.",
    ),
    (
        "INW-FORM",
        "error",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1 (In words)",
        "The blurb label is literally *In words.* (italic, with the period).",
    ),
    (
        "LABEL-READIT",
        "error",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1 (Read it. retired 2026-08-05)",
        'The "Read it." label is retired; read the output in ordinary prose.',
    ),
    (
        "LABEL-LEADIN",
        "warning",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1; writing-voice register/cut_on_sight.md, Sentence mechanics",
        "Prose under an output carries no labelled lead-in; drop the label.",
    ),
    (
        "NOMEN-TABLE",
        "error",
        "nb",
        "cell",
        True,
        "CANON.md, section 1 (nomenclature table); modify-example SKILL.md, formulation 1",
        "The formulation opens with the | Symbol | Code | Definition | table, before the first numbered display.",
    ),
    (
        "EQ-NUMBERING",
        "error",
        "nb",
        "cell",
        True,
        "CANON.md, section 1 (self-contained numbering); modify-example SKILL.md, series conventions 1",
        "A notebook numbers its own equations from (1), each number once.",
    ),
    (
        "XREF-SIBLING-NUM",
        "error",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 1 (self-contained numbering); AGENTS.md, Notation",
        "Name a sibling notebook by its file name; never cite its section or equation numbers.",
    ),
    (
        "XREF-EXTERNAL-EQ",
        "warning",
        "nbmd",
        "line",
        True,
        "CANON.md, section 1 (self-contained numbering; a paper's models are named, not numbered)",
        "Do not cite an external paper's equation numbers; state the construct in this notebook's numbering.",
    ),
    (
        "XREF-SIBLING-DESC",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 1 (name siblings by file)",
        "Name the sibling notebook by its file name, not by a description.",
    ),
    # canon
    (
        "CANON-SF-ROMAN",
        "error",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 1 (italic SF)",
        "The shift factor is italic SF_{n,m}; drop \\mathrm{SF} or \\mathit{SF}.",
    ),
    (
        "CANON-SF-ORDER",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 1 (bus first)",
        "The shift factor is SF_{n,m}, bus first and line second.",
    ),
    (
        "CANON-LINE-INDEX",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 1 (line index m)",
        "The line index is m, never \\ell or l (F^{dc}_\\ell is the HVDC link index).",
    ),
    (
        "CANON-INTERFACE-T",
        "warning",
        "nbmd",
        "line",
        True,
        "CANON.md, section 3 (transport layer table; script I, not script T)",
        "The declared interface set is script I; script T is reserved for a time set.",
    ),
    (
        "CANON-MARGIN-BETA",
        "warning",
        "nbmd",
        "line",
        True,
        "CANON.md, section 6 (the margin); NOTEBOOK_RECORD.md, section 4",
        "The ATC reservation margin is psi, renamed from beta on 2026-10-18.",
    ),
    (
        "CANON-MONITORED",
        "warning",
        "nbmd",
        "line",
        False,
        "CANON.md, section 3 (three-tier line distinction), section 8",
        '"Monitored" is per engine; the shared set is the network line set. Fix on touch where the shared set is meant.',
    ),
    (
        "CANON-LAGA",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "CANON.md, section 3",
        "L_a - G_a is the net settlement position N_a; congestion by region is N_a^c.",
    ),
    # lexicon and the cut-on-sight list
    (
        "LEX-BALANCING-AREA",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "AGENTS.md, How the model runs (Markets and areas)",
        "A market is not a balancing area; say market, or balancing authority area for an area within a market.",
    ),
    (
        "LEX-RETIRED-FRAME",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "CANON.md, section 7 (the intra-market and inter-market distinction)",
        "The allocation-versus-coordination framing is retired; state which engine enforces which constraint.",
    ),
    (
        "LEX-TRANSMISSION-OWNER",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "notes/writing_profile/source_register_profile.md, lexicon; style_check TELLS",
        "Name the specific transmission owner or the Participating Transmission Owner, not a generic one.",
    ),
    (
        "LEX-NETWORK-METAPHOR",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "notes/writing_profile/source_register_profile.md, no physical-plant metaphor; style_check TELLS",
        "Name the network as a facility, a constraint, an area or a flow contribution, not as an image.",
    ),
    (
        "LEX-EURO-KV",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "scoped out: no 9-bus file names a line class by voltage, and the public canon states no voltage convention",
        "Name a line class by its US nominal voltage (500/345/230/115 kV), never a European band.",
    ),
    (
        "CUT-PHRASES",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "writing-voice register/cut_on_sight.md; style_check TELLS",
        "Cut the intensifier, the evaluation or the meta-commentary; state the fact.",
    ),
    (
        "CUT-READER-ADDRESS",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "writing-voice register/cut_on_sight.md (reader-address and drama); style_check TELLS",
        "State what the dispatch does; do not address the reader.",
    ),
    (
        "CUT-IMPERATIVE",
        "warning",
        "nbmd",
        "line",
        True,
        "writing-voice register/cut_on_sight.md (the imperative)",
        'Use the declarative where it is available ("The clearing minimizes ...", not "Minimize ...").',
    ),
    (
        "VOICE-FIRST-PERSON",
        "warning",
        "nbmd",
        "line",
        True,
        "notes/writing_profile/source_register_profile.md, section 7 (Choosing a register)",
        "The notebook voice is impersonal; rewrite without we, our or us.",
    ),
    # punctuation, numbers, citation
    (
        "PUNCT-EM-DASH",
        "error",
        "nbmd,tex,md",
        "line",
        True,
        "writing-voice register/cut_on_sight.md, Sentence mechanics (em-dashes in titles and table cells only)",
        "Em-dashes live in section titles and table cells only; split the sentence.",
    ),
    (
        "PUNCT-DASH-ASIDE",
        "warning",
        "nbmd,tex",
        "line",
        True,
        "notes/writing_profile/source_register_profile.md, section 5 (Sentence mechanics)",
        "A spaced en-dash is an aside; split the sentence.",
    ),
    (
        "NUM-PERCENT",
        "warning",
        "tex",
        "line",
        True,
        "the mentor primer profile, not carried here, numbers (mentor voice)",
        "The mentor voice spells out percent.",
    ),
    (
        "NUM-THOUSANDS",
        "warning",
        "tex",
        "line",
        True,
        "the mentor primer profile, not carried here, numbers (mentor voice)",
        "A number of four or more digits takes a thousands separator in the mentor voice.",
    ),
    (
        "CITE-AUTHOR-DATE",
        "warning",
        "nbmd,tex,md",
        "line",
        True,
        "writing-voice SKILL.md, hard constraints (a person's work is introduced at first prose mention)",
        "Introduce the person and the context at first mention, not a bare Name (Year).",
    ),
    # notebook code and authoring
    (
        "NB-CAPTION-CELL",
        "warning",
        "nbmd",
        "cell",
        True,
        "writing-voice register/cut_on_sight.md, Sentence mechanics (no caption cells)",
        "No caption cell narrates a figure whose suptitle carries the content.",
    ),
    (
        "NB-EDIT-KNOBS",
        "warning",
        "nb",
        "file",
        False,
        "modify-example SKILL.md, pattern of an example 5",
        "The knobs are visible in a # -- EDIT -- cell.",
    ),
    (
        "NB-SANDBOX",
        "warning",
        "nb",
        "file",
        False,
        "modify-example SKILL.md, pattern of an example 5",
        "The notebook ends in a sandbox under a heading that names it.",
    ),
    (
        "CODE-MPLBACKEND",
        "error",
        "nbcode",
        "line",
        True,
        "AGENTS.md, Running and verifying; .claude/hooks/guard.py; modify-example SKILL.md, Reading and running",
        "Never set the Agg backend; it strips every stored figure.",
    ),
    (
        "CODE-EXPORT-CWD",
        "error",
        "nbcode",
        "line",
        True,
        "modify-example SKILL.md, Where a notebook writes; AGENTS.md, Running and verifying",
        "Resolve an export path from the repository (walk up to the root), never from the working directory.",
    ),
    (
        "CODE-SHED-PRICE",
        "warning",
        "nbcode",
        "line",
        True,
        "CANON.md, section 7 (SHED_PRICE)",
        "SHED_PRICE is 150 in every notebook; reconcile on touch.",
    ),
    # LaTeX and bib
    (
        "TEX-ONE-SENTENCE",
        "warning",
        "tex",
        "line",
        True,
        "writing-voice SKILL.md, Hard constraints",
        "One sentence per source line; break the line at the sentence boundary.",
    ),
    (
        "TEX-VOICE-HEADER",
        "warning",
        "tex",
        "file",
        True,
        "writing-voice SKILL.md, The workflow",
        "Declare the voice in the header comment (% Voice: ...).",
    ),
    (
        "TEX-EQREF",
        "warning",
        "tex",
        "line",
        True,
        "CANON.md, section 1 (self-contained numbering)",
        "Reference an equation with \\eqref, not \\ref.",
    ),
    (
        "CITE-KEY-MISSING",
        "warning",
        "tex,nbmd",
        "line",
        True,
        "AGENTS.md, How to use this repository (each references.bib entry carries url and note)",
        "The cited key is not an entry in sources/references.bib (a cite command in a note, a backticked key in notebook markdown).",
    ),
    (
        "SYM-R1",
        "warning",
        "tex",
        "line",
        True,
        "writing-voice SKILL.md, hard constraints; symbol_check.py R1",
        "Name the quantity first, then the symbol as an appositive in commas.",
    ),
    (
        "SYM-R2",
        "warning",
        "tex",
        "line",
        True,
        "writing-voice SKILL.md, hard constraints; symbol_check.py R2",
        "Read a relation as a sentence with each side named.",
    ),
    (
        "SYM-R3",
        "warning",
        "tex",
        "line",
        True,
        "writing-voice SKILL.md, hard constraints; symbol_check.py R3 (error in a td-voice file)",
        "Follow each numbered equation with an In-words block that cites it.",
    ),
    (
        "BIB-NOTE-SAFE",
        "error",
        "bib",
        "line",
        True,
        "AGENTS.md, How to use this repository (each references.bib entry carries url and note)",
        "The note field prints in the bibliography; escape %, _, & and #, and wrap math.",
    ),
]
RULES = {
    r[0]: Rule(r[0], r[1], tuple(r[2].split(",")), r[3], r[4], r[5], r[6]) for r in _R
}

# Path exemptions and path restrictions, by rule (regexes on the repository-relative path).
PATH_EXEMPT = {
    "CUT-PHRASES": [RULE_DOCS],
    "CUT-READER-ADDRESS": [RULE_DOCS],
    "LEX-NETWORK-METAPHOR": [RULE_DOCS],
    "LEX-TRANSMISSION-OWNER": [RULE_DOCS],
    "LEX-BALANCING-AREA": [RULE_DOCS],
    "LEX-RETIRED-FRAME": [RULE_DOCS],
    "CITE-AUTHOR-DATE": [RULE_DOCS],
    "PUNCT-EM-DASH": [RULE_DOCS],
    # D19: no 9-bus file names a line class by voltage, and the public canon states no voltage convention
    "LEX-EURO-KV": [r".*"],
}
PATH_ONLY = {
    "CANON-MARGIN-BETA": [MARGIN_NB],
    "CODE-SHED-PRICE": [r"^[^/]+\.ipynb$"],
}


def path_applies(rule_id: str, rel: str) -> bool:
    if any(re.search(p, rel) for p in PATH_EXEMPT.get(rule_id, [])):
        return False
    only = PATH_ONLY.get(rule_id)
    return not only or any(re.search(p, rel) for p in only)


# ------------------------------------------------------------------ findings -------------
class Finding(object):
    __slots__ = ("rule", "severity", "rel", "cell", "line", "span", "excerpt", "note")

    def __init__(
        self,
        rule: str,
        rel: str,
        cell: int | None = None,
        line: int | None = None,
        span: tuple[int, int] | None = None,
        excerpt: str = "",
        note: str = "",
        severity: str | None = None,
    ) -> None:
        self.rule, self.rel, self.cell, self.line = rule, rel, cell, line
        self.span, self.excerpt, self.note = span, excerpt, note
        self.severity = severity or RULES[rule].severity

    def pin(self) -> str:
        if self.cell is not None:
            return "cell %d" % self.cell + (" line %d" % self.line if self.line else "")
        return "line %d" % self.line if self.line else "file"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.rel,
            "cell": self.cell,
            "line": self.line,
            "pin": self.pin(),
            "excerpt": self.excerpt,
            "note": self.note,
            "message": RULES[self.rule].message,
        }


def read_text(path: str | Path) -> str:
    """The file as UTF-8 text, undecodable bytes replaced."""
    with io.open(str(path), encoding="utf8", errors="replace") as fh:
        return fh.read()


def clip(s: str | None, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


PRAGMA = re.compile(
    r"lint:\s*allow\s+([A-Z0-9][A-Z0-9\-]*(?:\s*,\s*[A-Z0-9][A-Z0-9\-]*)*)"
)
PRAGMA_FILE = re.compile(
    r"lint:\s*allow-file\s+([A-Z0-9][A-Z0-9\-]*(?:\s*,\s*[A-Z0-9][A-Z0-9\-]*)*)"
)


def pragma_ids(text: str | None) -> set[str]:
    out: set[str] = set()
    for m in PRAGMA.finditer(text or ""):
        out.update(x.strip() for x in m.group(1).split(","))
    return out


def file_pragma_ids(text: str | None) -> set[str]:
    out: set[str] = set()
    for m in PRAGMA_FILE.finditer(text or ""):
        out.update(x.strip() for x in m.group(1).split(","))
    return out


class Sink(object):
    """Collects findings for one file, applying path rules and pragmas."""

    def __init__(self, rel: str, kind_of_file: str, file_text: str = "") -> None:
        self.items: list[Finding]
        self.rel, self.kind, self.items = rel, kind_of_file, []
        self.file_allow = file_pragma_ids(file_text)
        self._applies: dict[str, bool] = {}

    def applies(self, rule_id: str) -> bool:
        if rule_id not in self._applies:
            self._applies[rule_id] = rule_id not in self.file_allow and path_applies(
                rule_id, self.rel
            )
        return self._applies[rule_id]

    def add(
        self, rule_id: str, allow: Collection[str] = (), **kw: Unpack[_FindingKw]
    ) -> None:
        if not self.applies(rule_id) or rule_id in allow:
            return
        self.items.append(Finding(rule_id, self.rel, **kw))


# ------------------------------------------------------------------ shared patterns ------
CUT_EXTRA = re.compile(
    r"\b(the whole point of the exercise|look at who does what|watch what happens|that is the trap|"
    r"naming the pieces first|let'?s now turn to|as we saw above|two sides of the same coin|"
    r"two instances of one|one (?:quantity|problem|question) with two (?:readings|faces))\b",
    re.I,
)
BAL_AREA = re.compile(r"\bbalancing areas?\b", re.I)
BAL_AREA_OK = re.compile(
    r"\bIntegrated Balancing Areas?\b|\bbalancing area authorit", re.I
)
RETIRED_FRAME = re.compile(
    r"\ballocation\W{0,3}\s+(?:versus|vs\.?)\s+\W{0,3}coordination\b"
    r"|\b(?:an?|the) (?:allocation|appropriation)(?:/| and | or )(?:coordination) (?:pair|problem|framing)\b",
    re.I,
)
AUTHOR_DATE = re.compile(
    r"(?<![\w(])((?:[A-Z][a-zA-Z'\-]+)(?: and [A-Z][a-zA-Z'\-]+| et al\.)?) \(((?:19|20)\d\d[a-z]?)\)"
)
EURO_KV = re.compile(r"\b(380|400|132|110)\s*(?:~|\s)?kV\b")
SIBLING_DESC = re.compile(r"\bthe (?:\w+[- ]){1,2}notebook\b", re.I)
SIBLING_DESC_OK = re.compile(
    r"\b(this|first|next|present|same|previous|preceding|companion|teaching|whole|entire|each|every)\b",
    re.I,
)
SIBLING_NUM = re.compile(
    r"\b[123]\d\d[a-c]?(?:'s|\u2019s)?\s*(?:\u00a7\s*\d|Section\s+\d|Sec\.\s*\d|[Ee]q(?:uation)?s?\.?\s*\(?\d|\(\d+[a-z]?\)(?!\s*(?:MW|\$|percent|%)))"
    r"|(?:\u00a7|Section|[Ee]quation|[Ee]q\.)\s*\(?\d+(?:\.\d+)?\)?\s+(?:of|in)\s+(?:notebook\s+)?[123]\d\d[a-c]?\b"
)
EXT_EQ = re.compile(
    r"(?:[Ee]quation|[Ee]q\.|\u00a7)\s*\(?\d+[a-z]?\)?\s+(?:of|in)\s+(?:the\s+)?(?:[A-Z][a-z]+|EDAM|\\cite)"
    r"|\b(?:[Hh]is|[Hh]er|[Tt]heir|[Tt]he paper'?s|[Tt]he source'?s)\s+\(?\d+[a-z]\)?"
)
IMPERATIVE_OPEN = re.compile(
    r"(?:Notice|Look|Watch|Recall|Observe|Imagine|Let's|Let us|See|Try|Run|Set|Change|Compare|Give|Minimi[sz]e|Maximi[sz]e)\b"
)
LABEL_READIT = re.compile(r"^\s*(?:\*\*|\*|_)Read it\s*[.:]?\s*(?:\*\*|\*|_)", re.I)
LABEL_LEADIN = re.compile(
    r"^\s*(?:\*\*|\*|_)(?:Reading|Takeaway|Interpretation|Key (?:point|takeaway|result)s?|What (?:this|it) shows|"
    r"Bottom line|So what|Observation|Insight|Results?)\s*[.:]?\s*(?:\*\*|\*|_)",
    re.I,
)
INW_VARIANT = re.compile(
    r"\*\*In words\.?\*\*|\bIn words:|_In words\._|\*In words\*(?!\.)|\*In words:\*"
)
INW_BLURB = re.compile(r"^\s*\*In words[^*]{0,30}\.\*")
SF_ROMAN = re.compile(r"\\(?:mathrm|mathit|text|operatorname)\s*\{\s*SF\s*\}")
SF_REVERSED = re.compile(r"SF_\{\s*(?:m|\\ell|l)\s*,\s*[a-z]")
LINE_IDX = re.compile(
    r"(?:\\bar\s*F|\\overline\{F\}|(?<![A-Za-z\\])F|\\mu|(?<![A-Za-z\\])R)(\^\{[^{}]*\}|\^[A-Za-z])?_\{?\s*(?:\\ell|l)(?![A-Za-z])"
    r"|SF_\{[^{},]*,\s*(?:\\ell|l)\s*\}"
    r"|(?:\\ell|(?<![A-Za-z\\])l)\s*\\in\s*\\mathcal\s*\{?\s*M"
)
INTERFACE_T = re.compile(r"\\mathcal\s*\{?\s*T(?![A-Za-z])")
MARGIN_BETA = re.compile(r"(?<!tilde)(?<!tilde\{)\\beta(?![A-Za-z])")
MONITORED = re.compile(r"\bmonitored lines?\b", re.I)
LA_GA = re.compile(
    r"L_a\s*-\s*G_a[^.]{0,80}congestion (?:rent|revenue)|congestion (?:rent|revenue)[^.]{0,80}L_a\s*-\s*G_a",
    re.I,
)
EMDASH = re.compile("\u2014")
DASH_ASIDE = re.compile("\\s\u2013\\s")
PCT = re.compile(r"\d\s*\\%")
THOUSANDS = re.compile(
    r"(?<![\d,.\-/])\d{4,}(?:\.\d+)?(?:~|\s|\\,)*(?:MW|MWh|GW|GWh|km|miles|dollars)\b"
)
FIRST_PERSON = re.compile(r"\b(?:[Ww]e|[Oo]ur|[Oo]urs|us)\b")
WIP_BANNER = re.compile(r"^\s*>\s*\*\*Work in progress\*\*")
RUNIN = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)?\*\*[^*]+\*\*")


def unquote(line: str, min_len: int = 1) -> str:
    """Replace quoted source text with a placeholder; the lexical rules skip it."""
    line = re.sub(
        r"\"[^\"]{%d,}\"|\u201c[^\u201d]{%d,}\u201d" % (min_len, min_len), " Q ", line
    )
    return line


def _exempt_match(rule_id: str, rx_match: re.Match[str], text: str) -> bool:
    """Rule-specific exemptions applied to a regex match."""
    if rule_id == "LEX-BALANCING-AREA":
        return bool(BAL_AREA_OK.search(text))
    if rule_id == "XREF-SIBLING-DESC":
        after = text[rx_match.end() :]
        # a number, an inline code name, or a \path{...} file name follows the description
        return bool(
            SIBLING_DESC_OK.search(rx_match.group(0))
            or re.match(r"\s*(?:\(?\s*)?(?:CODE\s*|C\s+)?[123]\d\d", after)
            or re.match(r"\s*(?:CODE|C)\b", after)
        )
    if rule_id == "CITE-AUTHOR-DATE":
        before = text[: rx_match.start()]
        # the label of a source list item ("\item Hogan (1998): ...") names the document, not a person's work in prose
        if re.match(r"\s*\\item\b[^:]*$", before) and ":" in text[rx_match.end() :]:
            return True
        return bool(
            re.search(r"\b(?:[Tt]he|[Ii]ts|[Tt]his|[Tt]hat)\s+$", before)
        ) or rx_match.group(1) in (
            "Order",
            "No",
            "Docket",
            "Section",
            "Stage",
            "Phase",
            "Appendix",
            "Table",
            "Figure",
            "Volume",
            "Part",
        )
    return False


def prose_rules(
    sink: Sink,
    kind: str,
    prose: str,
    allow: Collection[str],
    cell: int | None = None,
    line: int | None = None,
    span: tuple[int, int] | None = None,
    raw: str = "",
) -> None:
    """The lexical rules shared by notebook markdown, .tex and .md prose (quotations already removed)."""
    ex = clip(raw or prose)

    def add(rid: str) -> None:
        sink.add(rid, allow, cell=cell, line=line, span=span, excerpt=ex)

    def on(rid: str) -> bool:
        return kind in RULES[rid].kinds and sink.applies(rid) and rid not in allow

    if on("CUT-PHRASES") and (
        CUT_EXTRA.search(prose)
        or any(rx.search(prose) for rx in tell_patterns("CUT-PHRASES"))
    ):
        add("CUT-PHRASES")
    if on("CUT-READER-ADDRESS") and any(
        rx.search(prose) for rx in tell_patterns("CUT-READER-ADDRESS")
    ):
        add("CUT-READER-ADDRESS")
    if on("LEX-NETWORK-METAPHOR") and any(
        rx.search(prose) for rx in tell_patterns("LEX-NETWORK-METAPHOR")
    ):
        add("LEX-NETWORK-METAPHOR")
    if on("LEX-TRANSMISSION-OWNER") and any(
        rx.search(prose) for rx in tell_patterns("LEX-TRANSMISSION-OWNER")
    ):
        add("LEX-TRANSMISSION-OWNER")
    for rid, rx in (
        ("LEX-BALANCING-AREA", BAL_AREA),
        ("LEX-RETIRED-FRAME", RETIRED_FRAME),
        ("LEX-EURO-KV", EURO_KV),
        ("XREF-SIBLING-DESC", SIBLING_DESC),
        ("XREF-SIBLING-NUM", SIBLING_NUM),
        ("CANON-MONITORED", MONITORED),
        ("PUNCT-DASH-ASIDE", DASH_ASIDE),
        ("XREF-EXTERNAL-EQ", EXT_EQ),
        ("VOICE-FIRST-PERSON", FIRST_PERSON),
    ):
        if on(rid):
            m = rx.search(prose)
            if m and not _exempt_match(rid, m, prose):
                add(rid)
    if on("CITE-AUTHOR-DATE"):
        for m in AUTHOR_DATE.finditer(prose):
            if not _exempt_match("CITE-AUTHOR-DATE", m, prose):
                add("CITE-AUTHOR-DATE")
                break


def math_rules(
    sink: Sink,
    kind: str,
    text: str,
    allow: Collection[str],
    cell: int | None = None,
    line: int | None = None,
    span: tuple[int, int] | None = None,
    raw: str = "",
) -> None:
    """Canon checks on text that still carries its math."""
    ex = clip(raw or text)

    def chk(
        rid: str,
        rx: re.Pattern[str],
        extra_ok: Callable[[re.Match[str]], bool] | None = None,
    ) -> None:
        if kind in RULES[rid].kinds and sink.applies(rid) and rid not in allow:
            m = rx.search(text)
            if m and not (extra_ok and extra_ok(m)):
                sink.add(rid, allow, cell=cell, line=line, span=span, excerpt=ex)

    chk("CANON-SF-ROMAN", SF_ROMAN)
    chk("CANON-SF-ORDER", SF_REVERSED)
    chk(
        "CANON-LINE-INDEX",
        LINE_IDX,
        lambda m: (
            bool(re.search(r"dc", m.group(1) or ""))
            or bool(re.search(r"\bHVDC\b|\blinks?\b|F\^\{?\\?(?:mathrm\{)?dc", text))
        ),
    )
    chk("CANON-INTERFACE-T", INTERFACE_T)
    chk("CANON-MARGIN-BETA", MARGIN_BETA)
    chk("CANON-LAGA", LA_GA)


# ------------------------------------------------------------------ notebooks ------------
EQ_TAG = re.compile(r"\\tag\*?\{\s*(\d+[a-z]?)\s*\}")
EQ_QUAD = re.compile(
    r"\\q?quad\s*\(\s*(\d+[a-z]?)\s*\)(?:\s|\\q?quad|\\,|\\;|~)*[.,]?\s*$"
)
TABLE_HEAD = re.compile(r"\|\s*Symbol\s*\|\s*Code\s*\|\s*Definition\s*\|")
CONT_OPEN = re.compile(r"\s*(?:with|where|and|for|in which|subject to)\b")


def eq_number(body: str) -> str | None:
    m = EQ_TAG.search(body) or EQ_QUAD.search(body.strip())
    return m.group(1) if m else None


def md_pieces(src: str) -> list[tuple[str, str, int]]:
    """Split a markdown cell into ('display', body, line) and ('text', line_text, line) pieces.

    Fenced code is blanked so that line numbers stay true. Line numbers are 1-based."""
    src = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    out, pos = [], 0
    for m in re.finditer(r"\$\$(.*?)\$\$", src, re.S):
        start_line = src.count("\n", 0, m.start()) + 1
        seg = src[pos : m.start()]
        base = src.count("\n", 0, pos) + 1
        for k, ln in enumerate(seg.split("\n")):
            if ln.strip():
                out.append(("text", ln, base + k))
        out.append(("display", m.group(1), start_line))
        pos = m.end()
    base = src.count("\n", 0, pos) + 1
    for k, ln in enumerate(src[pos:].split("\n")):
        if ln.strip():
            out.append(("text", ln, base + k))
    return out


def strip_code(line: str) -> str:
    return re.sub(r"`[^`]*`", " CODE ", line)


def strip_inline(line: str) -> str:
    line = strip_code(line)
    line = re.sub(r"\$[^$]*\$", " X ", line)
    line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"<!--.*?-->", " ", line)
    return line


def scan_notebook(
    path: Path,
    rel: str,
    nb: dict[str, Any] | None = None,
    raw_text: str | None = None,
) -> list[Finding]:
    if nb is None:
        raw_text = read_text(path)
        nb = json.loads(raw_text)
    cells = nb.get("cells", [])
    sink = Sink(rel, "ipynb", "")
    # file pragmas live in any cell
    for c in cells:
        src = "".join(c.get("source", []))
        if "allow-file" in src:
            sink.file_allow |= file_pragma_ids(src)
    srcs = ["".join(c.get("source", [])) for c in cells]
    sandbox_from = next(
        (
            i
            for i, (c, s) in enumerate(zip(cells, srcs))
            if c.get("cell_type") == "markdown"
            and re.search(r"(?m)^#+ .*[Ss]andbox", s)
        ),
        None,
    )
    edit_cells = {
        i
        for i, (c, s) in enumerate(zip(cells, srcs))
        if c.get("cell_type") == "code" and re.search(r"#\s*--\s*EDIT\s*--", s)
    }
    eqs = []  # (cell, line, number)
    table_at = None  # (cell, line)
    pending = None  # (cell, line, number) of a numbered display awaiting its blurb
    cont = None  # a where-clause seen after the pending display
    blurb_open = False  # a blurb was read since the last display
    prev_kind = None
    prev_image = False
    for ci, (c, src) in enumerate(zip(cells, srcs)):
        ctype = c.get("cell_type")
        if ctype == "code":
            # the blurb comes immediately after the equation, before any code cell
            if pending is not None:
                sink.add(
                    "INW-MISSING",
                    cell=pending[0],
                    line=pending[1],
                    span=(pending[0], ci),
                    excerpt="eq (%s) is followed by a code cell" % pending[2],
                )
                pending, cont = None, None
            scan_code_cell(sink, ci, src)
            prev_image = any(
                "image/png" in (o.get("data") or {}) for o in c.get("outputs", []) or []
            )
            continue
        if ctype != "markdown":
            continue
        lines = src.split("\n")

        def allow_for(ln: int) -> set[str]:
            a = pragma_ids(lines[ln - 1]) if 0 < ln <= len(lines) else set()
            if ln >= 2:
                a |= pragma_ids(lines[ln - 2])
            return a

        if prev_image and re.match(
            r"\s*(?:Figure|Fig\.|The figure|Above|The (?:left|right|top|bottom) panel|The chart|The plot)\b",
            src,
        ):
            sink.add(
                "NB-CAPTION-CELL",
                allow_for(1),
                cell=ci,
                line=1,
                span=(ci - 1, ci),
                excerpt=clip(src),
            )
        prev_image = False
        th = TABLE_HEAD.search(src)
        if th and table_at is None:
            table_at = (ci, src.count("\n", 0, th.start()) + 1)
        in_sandbox = sandbox_from is not None and ci >= sandbox_from
        before_edit = (ci + 1) in edit_cells
        prev_blank = True
        last_ln = 0
        for kind, piece, ln in md_pieces(src):
            if kind == "display":
                allow = allow_for(ln)
                math_rules(
                    sink,
                    "nbmd",
                    piece,
                    allow,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    raw=piece,
                )
                n = eq_number(piece)
                if n:
                    if pending is not None:
                        sink.add(
                            "INW-MISSING",
                            allow,
                            cell=pending[0],
                            line=pending[1],
                            span=(pending[0], ci),
                            excerpt="eq (%s) is followed by eq (%s) before any *In words.*"
                            % (pending[2], n),
                        )
                    pending = (ci, ln, n)
                    cont = None
                    eqs.append((ci, ln, n))
                blurb_open = False
                prev_kind = "display"
                prev_blank = False
                last_ln = ln + piece.count("\n")
                continue
            # a text line
            if ln > last_ln + 1:
                prev_blank = True
            last_ln = ln
            line = piece
            allow = allow_for(ln)
            stripped = line.strip()
            is_heading = stripped.startswith("#")
            is_table = stripped.startswith("|")
            nocode = strip_code(line)
            math_rules(
                sink, "nbmd", nocode, allow, cell=ci, line=ln, span=(ci, ci), raw=line
            )
            noinline = strip_inline(line)
            cite_key_nb(sink, line, allow, ci, ln)
            if re.search(r"(?<!\\)\\[(\[]", noinline):
                sink.add(
                    "MATH-DELIM",
                    allow,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt=clip(line),
                )
            if LABEL_READIT.match(line):
                sink.add(
                    "LABEL-READIT",
                    allow,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt=clip(line),
                )
            elif LABEL_LEADIN.match(line):
                sink.add(
                    "LABEL-LEADIN",
                    allow,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt=clip(line),
                )
            if INW_VARIANT.search(line):
                sink.add(
                    "INW-FORM",
                    allow,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt=clip(line),
                )
            # the In-words state machine
            if INW_BLURB.match(line):
                if pending is not None and cont is not None:
                    sink.add(
                        "INW-DEFERRED",
                        allow,
                        cell=pending[0],
                        line=pending[1],
                        span=(pending[0], ci),
                        excerpt="eq (%s): %s" % (pending[2], clip(cont, 120)),
                    )
                elif pending is None and prev_kind != "display":
                    label = re.match(r"\s*\*(In words[^*]*)\*", line).group(1)  # type: ignore[union-attr]  # the match is None where the line does not open with the blurb
                    if not (blurb_open and label != "In words."):
                        sink.add(
                            "INW-ORPHAN",
                            allow,
                            cell=ci,
                            line=ln,
                            span=(ci, ci),
                            excerpt=clip(line),
                        )
                pending, cont, blurb_open = None, None, True
            elif pending is not None:
                if cont is None and CONT_OPEN.match(line) and not is_heading:
                    cont = line
                elif cont is not None and not re.search(r"[.:]\s*$", cont):
                    cont += " " + line
                else:
                    sink.add(
                        "INW-MISSING",
                        allow,
                        cell=pending[0],
                        line=pending[1],
                        span=(pending[0], ci),
                        excerpt="eq (%s) is followed by: %s"
                        % (pending[2], clip(line, 120)),
                    )
                    pending, cont = None, None
            prev_kind = "text"
            para_start = prev_blank
            prev_blank = False
            if is_table:
                if sink.applies("XREF-SIBLING-NUM") and SIBLING_NUM.search(noinline):
                    sink.add(
                        "XREF-SIBLING-NUM",
                        allow,
                        cell=ci,
                        line=ln,
                        span=(ci, ci),
                        excerpt=clip(line),
                    )
                continue
            if is_heading:
                continue
            body = re.sub(r"^\s*>\s?", "", line)
            prose_raw = strip_inline(body)
            prose = unquote(prose_raw)
            prose_rules(
                sink, "nbmd", prose, allow, cell=ci, line=ln, span=(ci, ci), raw=line
            )
            emdash_check(sink, line, prose, allow, cell=ci, line=ln, span=(ci, ci))
            if (
                para_start
                and not in_sandbox
                and not before_edit
                and "CUT-IMPERATIVE" not in allow
            ):
                s = re.sub(r"^[\s>*_\-0-9.]+", "", prose)
                if IMPERATIVE_OPEN.match(s):
                    sink.add(
                        "CUT-IMPERATIVE",
                        allow,
                        cell=ci,
                        line=ln,
                        span=(ci, ci),
                        excerpt=clip(line),
                    )
    if pending is not None:
        sink.add(
            "INW-MISSING",
            cell=pending[0],
            line=pending[1],
            span=(pending[0], len(cells)),
            excerpt="eq (%s) is the last numbered display and has no *In words.*"
            % pending[2],
        )
    # whole-notebook rules
    if eqs:
        first = eqs[0]
        if int(re.match(r"\d+", first[2]).group()) != 1:  # type: ignore[union-attr]  # the match is None for a number with no leading digit
            sink.add(
                "EQ-NUMBERING",
                cell=first[0],
                line=first[1],
                span=(first[0], first[0]),
                excerpt="the first numbered equation is (%s)" % first[2],
            )
        seen: dict[str, int] = {}
        for ci, ln, n in eqs:
            if n in seen:
                sink.add(
                    "EQ-NUMBERING",
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt="(%s) is numbered twice (first in cell %d)" % (n, seen[n]),
                )
            else:
                seen[n] = ci
        if table_at is None:
            sink.add(
                "NOMEN-TABLE",
                cell=first[0],
                line=first[1],
                span=(0, first[0]),
                excerpt="%d numbered equations and no | Symbol | Code | Definition | table"
                % len(eqs),
            )
        elif table_at > (first[0], first[1]):
            sink.add(
                "NOMEN-TABLE",
                cell=table_at[0],
                line=table_at[1],
                span=(first[0], table_at[0]),
                excerpt="the table (cell %d) follows equation (%s) in cell %d"
                % (table_at[0], first[2], first[0]),
            )
    if not edit_cells:
        sink.add("NB-EDIT-KNOBS", excerpt="no # -- EDIT -- code cell")
    if sandbox_from is None:
        sink.add("NB-SANDBOX", excerpt="no heading that names a sandbox")
    return sink.items


def emdash_check(
    sink: Sink, raw_line: str, prose: str, allow: Collection[str], **loc: Unpack[_LocKw]
) -> None:
    """PUNCT-EM-DASH on a body line: the banner and a bold run-in heading are titles."""
    if (
        not EMDASH.search(prose)
        or not sink.applies("PUNCT-EM-DASH")
        or "PUNCT-EM-DASH" in allow
    ):
        return
    if WIP_BANNER.match(raw_line):
        return
    m = RUNIN.match(prose)
    if m and not EMDASH.search(prose[m.end() :]):
        return
    sev = "warning" if sink.kind == "md" else None
    sink.add("PUNCT-EM-DASH", allow, excerpt=clip(raw_line), severity=sev, **loc)


EXPORT_CALL = re.compile(
    r"\b(savefig|to_csv|to_excel|to_parquet|to_json|to_latex|write_text|write_bytes|open)\(\s*(?:[fFrRbB]{0,2})(['\"])([^'\"]+?\.(?:png|pdf|svg|jpg|csv|xlsx|json|txt|tex|parquet|html|md))\2([^)]{0,40})"
)
EXPORT_PATHLIT = re.compile(
    r"\bPath\(\s*(?:[rR]?)(['\"])(?:\./)?(?:notes|outputs|presentation_exhibits)(?:[/\\][^'\"]*)?\1\s*\)"
)
CWD_CALL = re.compile(r"os\.getcwd\(\)|Path\.cwd\(\)")
ROOT_WALK = re.compile(
    r"\bwhile\b|\.parents\b|\bfor\b[^\n]*\bin\b[^\n]*\.parent|os\.path\.dirname\("
)
AGG = re.compile(
    r"matplotlib\.use\(\s*['\"]agg|MPLBACKEND['\"]?\]?\s*[=:,]\s*['\"]?agg|%env\s+MPLBACKEND\s*=?\s*agg",
    re.I,
)


def scan_code_cell(sink: Sink, ci: int, src: str) -> None:
    lines = src.split("\n")

    def loc(off: int) -> tuple[int, set[str]]:
        ln = src.count("\n", 0, off) + 1
        a = pragma_ids(lines[ln - 1]) | (
            pragma_ids(lines[ln - 2]) if ln >= 2 else set()
        )
        return ln, a

    for m in AGG.finditer(src):
        ln, a = loc(m.start())
        if not lines[ln - 1].lstrip().startswith("#"):
            sink.add(
                "CODE-MPLBACKEND",
                a,
                cell=ci,
                line=ln,
                span=(ci, ci),
                excerpt=clip(lines[ln - 1]),
            )
    for m in EXPORT_CALL.finditer(src):
        func, p, rest = m.group(1), m.group(3), m.group(4)
        ln, a = loc(m.start())
        if (
            lines[ln - 1].lstrip().startswith("#")
            or p.startswith("{")
            or re.match(r"[A-Za-z]:[\\/]|/|~", p)
        ):
            continue
        if func == "open" and not re.match(r"\s*,\s*(?:mode\s*=\s*)?['\"][wax]", rest):
            continue
        sink.add(
            "CODE-EXPORT-CWD",
            a,
            cell=ci,
            line=ln,
            span=(ci, ci),
            excerpt=clip(lines[ln - 1]),
        )
    for m in EXPORT_PATHLIT.finditer(src):
        ln, a = loc(m.start())
        if not lines[ln - 1].lstrip().startswith("#"):
            sink.add(
                "CODE-EXPORT-CWD",
                a,
                cell=ci,
                line=ln,
                span=(ci, ci),
                excerpt=clip(lines[ln - 1]),
            )
    if CWD_CALL.search(src) and not ROOT_WALK.search(src):
        for m in CWD_CALL.finditer(src):
            ln, a = loc(m.start())
            if not lines[ln - 1].lstrip().startswith("#"):
                sink.add(
                    "CODE-EXPORT-CWD",
                    a,
                    cell=ci,
                    line=ln,
                    span=(ci, ci),
                    excerpt=clip(lines[ln - 1]),
                )
    m = re.search(r"^\s*SHED_PRICE\s*=\s*([\d_.]+)", src, re.M)  # type: ignore[assignment]  # m is a Match from the loops above, and a search can return None
    if m:
        try:
            v = float(m.group(1).replace("_", ""))
        except ValueError:
            v = None
        if v is not None and v != 150:
            ln, a = loc(m.start(1))
            sink.add(
                "CODE-SHED-PRICE",
                a,
                cell=ci,
                line=ln,
                span=(ci, ci),
                excerpt=clip(lines[ln - 1]),
            )


# ------------------------------------------------------------------ LaTeX ----------------
TEX_SKIP_ENV = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray|tabular|tabularx|longtable|tikzpicture|verbatim|lstlisting|table|figure|displaymath|array|quote|quotation|displayquote)\*?\}"
)
ABBREV = re.compile(
    r"\b(?:e\.g|i\.e|cf|vs|al|No|Nos|pp?|Fig|Figs|Sec|Secs|Eq|Eqs|Dr|Mr|Ms|Inc|Co|Corp|Ltd|U\.S|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|St|Vol|Ch|approx|resp)\.",
    re.I,
)


def tex_voice(text: str) -> str | None:
    """The voice the header comment declares: td, msc, mentor, or None."""
    head = (
        text.split("\\begin{document}", 1)[0]
        if "\\begin{document}" in text
        else "\n".join(text.split("\n")[:40])
    )
    comments = "\n".join(l for l in head.split("\n") if l.lstrip().startswith("%"))
    m = re.search(r"(?is)\bVoice:\s*(.{0,160})", comments)
    if not m:
        return None
    v = m.group(1)
    if re.search(r"mentor", v, re.I):
        return "mentor"
    if re.search(
        r"Technical Description|\(TD\)|technical description register", v, re.I
    ):
        return "td"
    if re.search(r"\bMSC\b", v):
        return "msc"
    return None


def tex_prose_lines(text: str) -> Iterator[tuple[int, str | None]]:
    """Yield (lineno, line) for prose lines of the body; None marks a break in the prose."""
    lines = text.split("\n")
    in_doc = "\\begin{document}" not in text
    env = None
    for i, raw in enumerate(lines, 1):
        line = raw
        if not in_doc:
            if "\\begin{document}" in line:
                in_doc = True
            continue
        if "\\end{document}" in line:
            break
        if env:
            if re.search(r"\\end\{%s\*?\}" % env, line):
                env = None
            yield i, None
            continue
        m = TEX_SKIP_ENV.search(line)
        if m:
            if not re.search(r"\\end\{%s\*?\}" % m.group(1), line):
                env = m.group(1)
            yield i, None
            continue
        if re.match(r"\s*%", line) or not line.strip():
            yield i, None
            continue
        if re.search(r"(^|[^\\])&", line) or re.match(
            r"\s*\\(toprule|midrule|bottomrule|hline|node|draw|fill|path)", line
        ):
            continue
        yield i, line


def tex_unquote(line: str, state: bool) -> tuple[str, bool]:
    """Replace ``...'' and \\enquote{...} spans with a placeholder; state carries an open quote across lines."""
    out, i = [], 0
    s = line
    if state:
        j = s.find("''")
        if j < 0:
            return " Q ", True
        out.append(" Q ")
        i = j + 2
        state = False
    while True:
        j = s.find("``", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = s.find("''", j + 2)
        if k < 0:
            out.append(" Q ")
            state = True
            break
        out.append(" Q ")
        i = k + 2
    res = "".join(out)
    res = re.sub(r"\\enquote\{[^}]*\}", " Q ", res)
    return res, state


def tex_clean(line: str) -> str:
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = re.sub(r"\$[^$]*\$", " X ", line)
    line = re.sub(
        r"\\(cite[pt]?|ref|eqref|label|pageref|href|url|path|includegraphics|input|include)\*?(\[[^\]]*\])*\{[^}]*\}",
        " C ",
        line,
    )
    line = re.sub(r"\\(textbf|textit|emph|textsc|texttt)\{([^}]*)\}", r"\2", line)
    line = line.replace("---", "\u2014").replace(" -- ", " \u2013 ")
    # LaTeX section sign and tie
    line = re.sub(r"\\S(?![A-Za-z])\s*", "\u00a7", line).replace("~", " ")
    return line


_BIB_KEYS: dict[str, Any] = {}
BIB_KEY_RX = re.compile(
    r"(?m)^[ \t]*@(?!comment\b|string\b|preamble\b)\w+[ \t]*[{(][ \t]*([^,\s{}]+)[ \t]*,",
    re.I,
)
CITE_CMD_RX = re.compile(r"\\cite[a-zA-Z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}")


def bib_keys() -> set[str] | None:
    """The keys of sources/references.bib under the root, parsed once per run and again when the file's mtime changes."""
    bib = repo_root() / "sources" / "references.bib"
    try:
        stamp = (str(bib), bib.stat().st_mtime_ns)
    except OSError:
        return None
    if _BIB_KEYS.get("stamp") != stamp:
        _BIB_KEYS["stamp"] = stamp
        _BIB_KEYS["keys"] = set(BIB_KEY_RX.findall(read_text(bib)))
    return _BIB_KEYS["keys"]


def cite_key_rule(sink: Sink, lines: list[str]) -> None:
    """CITE-KEY-MISSING: a key in a cite command that is not an entry of the bibliography (silent when there is no bibliography)."""
    keys = bib_keys()
    if not keys:
        return
    for i, line in enumerate(lines, 1):
        nocomment = re.sub(r"(?<!\\)%.*", "", line)
        for m in CITE_CMD_RX.finditer(nocomment):
            for k in (k.strip() for k in m.group(1).split(",")):
                if k and k != "*" and k not in keys:
                    a = pragma_ids(line) | (
                        pragma_ids(lines[i - 2]) if i >= 2 else set()
                    )
                    sink.add("CITE-KEY-MISSING", a, line=i, span=(i, i), excerpt=k)


#: A backticked token that reads as a bibliography key: lower-case words with a four-digit year inside.
NB_KEY_RX = re.compile(r"`([a-z][a-z0-9_]*?(?:19|20)\d\d[a-z0-9_]*)`")


def cite_key_nb(
    sink: Sink, line: str, allow: Collection[str], cell: int, ln: int
) -> None:
    """CITE-KEY-MISSING in notebook markdown: a backticked key that is not an entry of the bibliography."""
    if not sink.applies("CITE-KEY-MISSING"):
        return
    keys = bib_keys()
    if not keys:
        return
    for m in NB_KEY_RX.finditer(line):
        if m.group(1) not in keys:
            sink.add(
                "CITE-KEY-MISSING",
                allow,
                cell=cell,
                line=ln,
                span=(cell, cell),
                excerpt=m.group(1),
            )


def scan_tex(path: Path, rel: str, text: str | None = None) -> list[Finding]:
    if text is None:
        text = read_text(path)
    sink = Sink(rel, "tex", text)
    lines = text.split("\n")
    voice = tex_voice(text)
    is_part = "\\begin{document}" not in text
    if not is_part and not re.search(
        r"(?mi)^\s*%.*\bvoice\b", text.split("\\begin{document}", 1)[0]
    ):
        sink.add(
            "TEX-VOICE-HEADER", line=1, excerpt="no header comment declares the voice"
        )

    def allow_for(i: int) -> set[str]:
        a = pragma_ids(lines[i - 1])
        if i >= 2:
            a |= pragma_ids(lines[i - 2])
        return a

    prev = None
    qstate = False
    for i, line in tex_prose_lines(text):
        if line is None:
            prev, qstate = None, False
            continue
        allow = allow_for(i)
        span = (i, i)
        nocomment = re.sub(r"(?<!\\)%.*", "", line)
        if re.match(r"\s*\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})*\s*$", line):
            prev = None
            continue
        is_heading = re.match(
            r"\s*\\(part|chapter|section|subsection|subsubsection|title)\*?(\[[^\]]*\])?\{",
            line,
        )
        math_rules(sink, "tex", nocomment, allow, line=i, span=span, raw=line)
        if re.search(r"\\ref\{eq", line):
            sink.add("TEX-EQREF", allow, line=i, span=span, excerpt=clip(line))
        if is_heading:
            prev = None
            continue
        # a run-in title (\paragraph{...} or a leading \textbf{...}) is a title; the sentence after it is body
        body = re.sub(
            r"^\s*(?:\\item\s*(?:\[[^\]]*\])?\s*)?(?:\\paragraph\*?\{[^}]*\}|\\textbf\{[^}]*\})",
            " ",
            nocomment,
        )
        unq, qstate = tex_unquote(body, qstate)
        prose = tex_clean(unq)
        prose_rules(sink, "tex", prose, allow, line=i, span=span, raw=line)
        emdash_check(sink, line, prose, allow, line=i, span=span)
        if voice == "mentor":
            if PCT.search(unq):
                sink.add("NUM-PERCENT", allow, line=i, span=span, excerpt=clip(line))
            if THOUSANDS.search(prose):
                sink.add("NUM-THOUSANDS", allow, line=i, span=span, excerpt=clip(line))
        # one sentence per source line
        probe = re.sub(r"^\s*\\item\s*(\[[^\]]*\])?\s*", " ", line)
        probe = re.sub(
            r"^\s*(\\textbf\{[^}]*\}|\\textit\{[^}]*\}|\\paragraph\{[^}]*\}|\\emph\{[^}]*\})",
            " ",
            probe,
        )
        probe = re.sub(r"^\s*\((?:[A-Z][^()]{0,100}|[ivx]+|[a-z]|\d+)\)\s*", " ", probe)
        probe = tex_clean(re.sub(r"``.*?''|``.*$", " Q ", probe))
        probe = ABBREV.sub(lambda m: m.group(0).replace(".", "#"), probe)
        probe = re.sub(r"\b[A-Z]\.", "#", probe)
        if re.search(r"[a-z0-9)\]X'\"][.!?]['\")\]]*\s+(?:``)?[A-Z(]", probe):
            sink.add(
                "TEX-ONE-SENTENCE",
                allow,
                line=i,
                span=span,
                excerpt=clip(line),
                note="two sentences on one line",
            )
        if prev is not None:
            pl, pc = prev
            if (
                not re.search(r"([.!?:;,]['\")\]}]*|\\\\|\{|%)\s*$", pc.rstrip())
                and re.match(r"\s*[a-z(]", prose)
                and not re.match(r"\s*\\", line)
            ):
                sink.add(
                    "TEX-ONE-SENTENCE",
                    allow_for(pl),
                    line=pl,
                    span=(pl, i),
                    excerpt=clip(pc + " / " + prose),
                    note="sentence wrapped across lines",
                )
        prev = (i, prose)
    cite_key_rule(sink, lines)
    sym_rules(sink, path, text, voice)
    return sink.items


def sym_rules(sink: Sink, path: Path, text: str, voice: str | None) -> None:
    """SYM-R1 to R3 from symbol_check.check_file, passed through with their line numbers."""
    sc = sibling("symbol_check")
    if sc is None or not os.path.exists(str(path)):
        return
    try:
        flags = sc.check_file(path)
    except Exception as e:
        sys.stderr.write("prose_lint: symbol_check failed on %s (%s)\n" % (sink.rel, e))
        return
    lines = text.split("\n")
    for ln, rule, problem, _t in flags:
        rid = "SYM-" + rule
        if rid not in RULES:
            continue
        a = pragma_ids(lines[ln - 1]) if 0 < ln <= len(lines) else set()
        span = (ln, ln)
        if rule == "R3":
            end = ln
            for k in range(ln - 1, min(len(lines), ln + 80)):
                if re.search(r"\\end\{(equation|align)\*?\}", lines[k]):
                    end = k + 1
                    break
            span = (ln, end + 3)
        sev = "error" if (rule == "R3" and voice == "td") else None
        sink.add(rid, a, line=ln, span=span, excerpt=clip(problem), severity=sev)


# ------------------------------------------------------------------ markdown and bib -----
def scan_md(path: Path, rel: str, text: str | None = None) -> list[Finding]:
    if text is None:
        text = read_text(path)
    sink = Sink(rel, "md", text)
    lines = text.split("\n")
    in_fence = False
    front = lines and lines[0].strip() == "---"
    for i, line in enumerate(lines, 1):
        if front:
            if i > 1 and line.strip() == "---":
                front = False
            continue
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip():
            continue
        s = line.strip()
        if s.startswith("#") or s.startswith("|"):
            continue
        allow = pragma_ids(line) | (pragma_ids(lines[i - 2]) if i >= 2 else set())
        prose = unquote(strip_inline(re.sub(r"^\s*>\s?", "", line)))
        prose_rules(sink, "md", prose, allow, line=i, span=(i, i), raw=line)
        emdash_check(sink, line, prose, allow, line=i, span=(i, i))
    return sink.items


def scan_bib(path: Path, rel: str, text: str | None = None) -> list[Finding]:
    if text is None:
        text = read_text(path)
    sink = Sink(rel, "bib", text)
    for m in re.finditer(r"(?im)^\s*note\s*=\s*\{", text):
        j, d = m.end(), 1
        while d and j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            d += {"{": 1, "}": -1}.get(ch, 0)
            j += 1
        body = text[m.end() : j - 1]
        ln = text.count("\n", 0, m.start()) + 1
        nomath = re.sub(r"\$[^$]*\$", "", body)
        nomath = re.sub(r"\\(?:url|href)\{[^}]*\}", "", nomath)
        if re.search(r"(?<!\\)[%_&#]", nomath) or "\u2014" in body:
            sink.add(
                "BIB-NOTE-SAFE",
                pragma_ids(text[text.rfind("\n", 0, m.start()) + 1 : m.start()]),
                line=ln,
                span=(ln, ln + body.count("\n")),
                excerpt=clip(body),
            )
    return sink.items


# ------------------------------------------------------------------ dispatch -------------
KINDS: dict[str, Callable[..., list[Finding]]] = {
    ".ipynb": scan_notebook,
    ".tex": scan_tex,
    ".md": scan_md,
    ".bib": scan_bib,
}
EXCLUDED = re.compile(
    r"(^|/)(outputs|presentation_exhibits|sandbox_exhibits|\.ipynb_checkpoints)(/|$)|^\.git/"
)


def rel_of(path: str | Path, root: str | Path) -> str | None:
    """The repository-relative path with forward slashes, or None when outside the root."""
    a = os.path.normcase(os.path.abspath(str(path)))
    r = os.path.normcase(os.path.abspath(str(root)))
    if a == r or not a.startswith(r.rstrip("\\/") + os.sep):
        return None
    return os.path.relpath(
        os.path.abspath(str(path)), os.path.abspath(str(root))
    ).replace("\\", "/")


def in_scope(rel: str | None) -> bool:
    """True for a lintable file under the repository outside the excluded folders."""
    if rel is None or Path(rel).suffix.lower() not in KINDS:
        return False
    if EXCLUDED.search(rel):
        return False
    if rel.startswith("sources/") and rel != "sources/references.bib":
        return False
    return True


def lint_file(path: str | Path, root: Path | None = None) -> list[Finding]:
    root = root or repo_root()
    rel = rel_of(path, root) or Path(path).as_posix()
    fn = KINDS.get(Path(path).suffix.lower())
    if fn is None:
        return []
    return fn(Path(path), rel)


def all_files(root: Path) -> list[Path]:
    out = []
    out += sorted(root.glob("*.ipynb"))
    out += sorted(
        p for p in (root / "notes").glob("**/*.tex") if in_scope(rel_of(p, root))
    )
    bib = root / "sources" / "references.bib"
    if bib.exists():
        out.append(bib)
    return out


# ------------------------------------------------------------------ regions --------------
def in_region(f: Finding, region: set[int] | None, hook: bool) -> bool:
    """region None means the whole file. A file-scope finding needs the whole file in the hook."""
    if region is None:
        return True
    if f.span is None:
        return not hook
    lo, hi = f.span
    return any(lo <= x <= hi for x in region)


def git(root: Path, *args: str) -> bytes:
    r = subprocess.run(("git",) + args, cwd=str(root), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            "git %s failed: %s"
            % (" ".join(args), r.stderr.decode("utf8", "replace").strip())
        )
    return r.stdout


def changed_files(root: Path) -> list[tuple[Path, set[int] | None]]:
    """(path, region) for each modified or untracked lintable file; region None for untracked."""
    raw = git(root, "status", "--porcelain", "-z", "--untracked-files=all").decode(
        "utf8", "replace"
    )
    parts = raw.split("\0")
    out: list[tuple[Path, set[int] | None]]
    out, i = [], 0
    while i < len(parts):
        e = parts[i]
        i += 1
        if len(e) < 4:
            continue
        code, rel = e[:2], e[3:]
        if code[0] in "RC":
            i += 1  # the source path of a rename follows
        if "D" in code or not in_scope(rel):
            continue
        p = root / rel
        if not p.exists():
            continue
        if code == "??" or "A" in code:
            out.append((p, None))
            continue
        try:
            head = git(root, "show", "HEAD:" + rel).decode("utf8", "replace")
        except RuntimeError:
            out.append((p, None))
            continue
        out.append((p, diff_region(p, head)))
    return out


def diff_region(p: Path, head_text: str) -> set[int] | None:
    """Changed lines (text) or cells (notebook) of p against its HEAD text."""
    now = read_text(p)
    if p.suffix.lower() == ".ipynb":
        try:
            a = [
                "".join(c.get("source", []))
                for c in json.loads(head_text).get("cells", [])
            ]
            b = ["".join(c.get("source", [])) for c in json.loads(now).get("cells", [])]
        except ValueError:
            return None
        base = 0
    else:
        a, b = head_text.split("\n"), now.split("\n")
        base = 1
    region: set[int] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if j2 > j1:
            region.update(range(j1 + base, j2 + base))
        else:
            # a deletion touches its neighbour
            region.add(min(j1, max(len(b) - 1, 0)) + base)
    return region


# ------------------------------------------------------------------ the hook -------------
def _cells_of_offsets(raw: str, offsets: Iterable[int]) -> tuple[set[int], int]:
    """Map character offsets in a notebook's raw JSON to cell indices, by the cell_type keys."""
    marks = [m.start() for m in re.finditer(r'(?<!\\)"cell_type"\s*:', raw)]
    out: set[int] = set()
    for off in offsets:
        k = -1
        for idx, mpos in enumerate(marks):
            if mpos <= off:
                k = idx
            else:
                break
        if k >= 0:
            out.add(k)
    return out, len(marks)


def _line_spans(
    text: str, needle: str | None, all_occurrences: bool = True
) -> set[int]:
    region: set[int] = set()
    if not needle:
        return region
    start = 0
    while True:
        j = text.find(needle, start)
        if j < 0:
            break
        lo = text.count("\n", 0, j) + 1
        region.update(range(lo, lo + needle.count("\n") + 1))
        start = j + max(len(needle), 1)
        if not all_occurrences:
            break
    return region


def hook_region(
    data: dict[str, Any],
    path: Path,
    raw: str,
    nb: dict[str, Any] | None = None,
) -> set[int] | None:
    """The touched region for a PostToolUse payload: None for the whole file, a set otherwise.

    raw is the file's text after the tool call; nb is the parsed notebook for an .ipynb file."""
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    suffix = path.suffix.lower()
    if tool == "Write":
        return None
    if tool in ("Edit", "MultiEdit"):
        news: list[Any] = (
            [ti.get("new_string")]
            if tool == "Edit"
            else [e.get("new_string") for e in (ti.get("edits") or [])]
        )
        news = [n for n in news if n]
        if suffix == ".ipynb":
            offs = []
            for n in news:
                start = 0
                while True:
                    j = raw.find(n, start)
                    if j < 0:
                        break
                    offs.append(j)
                    start = j + len(n)
            cells, nmarks = _cells_of_offsets(raw, offs)
            if nb is None:
                return set()
            return cells if nmarks == len(nb.get("cells", [])) else None
        region: set[int] = set()
        for n in news:
            region |= _line_spans(raw, n)
        return region
    if tool == "NotebookEdit":
        if nb is None:
            return set()
        cells = nb.get("cells", [])
        mode = ti.get("edit_mode") or "replace"
        if mode == "delete":
            return set()
        new_src = ti.get("new_source")
        if new_src is not None:
            hits = {
                i
                for i, c in enumerate(cells)
                if "".join(c.get("source", [])) == new_src
            }
            if hits:
                return hits
            hits = {
                i
                for i, c in enumerate(cells)
                if "".join(c.get("source", [])).strip() == new_src.strip()
            }
            if hits:
                return hits
        cid = ti.get("cell_id")
        if cid is not None:
            for i, c in enumerate(cells):
                if c.get("id") == cid:
                    return {i + 1} if mode == "insert" else {i}
            m = re.match(r"(?:cell-)?(\d+)$", str(cid))
            if m:
                return {int(m.group(1))}
        return set()
    return set()


def format_findings(
    rel: str, items: list[Finding], scope_note: str, limit: int | None = None
) -> str:
    ne = sum(1 for f in items if f.severity == "error")
    nw = len(items) - ne
    head = "prose_lint: %d error%s, %d warning%s in %s%s." % (
        ne,
        "" if ne == 1 else "s",
        nw,
        "" if nw == 1 else "s",
        rel,
        scope_note,
    )
    rows = sorted(
        items,
        key=lambda f: (
            f.severity != "error",
            f.cell if f.cell is not None else -1,
            f.line or 0,
            f.rule,
        ),
    )
    shown = rows if limit is None else rows[:limit]
    out = [head]
    for f in shown:
        note = " (%s)" % f.note if f.note else ""
        out.append(
            "- %s %s %s%s: %s | %s"
            % (
                f.severity.upper(),
                f.rule,
                f.pin(),
                note,
                RULES[f.rule].message,
                f.excerpt,
            )
        )
    if limit is not None and len(rows) > limit:
        out.append(
            "... and %d more; run prose_lint.py on the file for the full list."
            % (len(rows) - limit)
        )
    return "\n".join(out)


def run_hook(stdin_text: str) -> str:
    """Return the hook's stdout text ('' on a clean or ignored call)."""
    try:
        data = json.loads(stdin_text)
    except Exception as e:
        sys.stderr.write(
            "prose_lint: could not parse hook input (%s); nothing checked\n" % e
        )
        return ""
    if not isinstance(data, dict):
        return ""
    ti = data.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("notebook_path")
    if not fp:
        return ""
    root = repo_root()
    path = Path(fp)
    rel = rel_of(path, root)
    if not in_scope(rel) or not path.exists():
        return ""
    raw = read_text(path)
    nb = None
    if path.suffix.lower() == ".ipynb":
        try:
            nb = json.loads(raw)
        except ValueError:
            return ""  # a notebook mid-write is not linted
    region = hook_region(data, path, raw, nb)
    if region is not None and not region:
        return ""
    fn = KINDS[path.suffix.lower()]
    found = fn(path, rel, nb, raw) if nb is not None else fn(path, rel, raw)
    items = [f for f in found if RULES[f.rule].hook and in_region(f, region, hook=True)]
    if not items:
        return ""
    note = " (whole file)" if region is None else " (touched region)"
    ctx = format_findings(rel, items, note, limit=HOOK_MAX_FINDINGS)  # type: ignore[arg-type]  # rel is None only for a path in_scope has refused
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": ctx,
            }
        }
    )


# ------------------------------------------------------------------ CLI ------------------
def list_rules() -> str:
    rows = [("id", "severity", "kinds", "hook", "source")]
    for r in RULES.values():
        rows.append(
            (r.id, r.severity, ",".join(r.kinds), "yes" if r.hook else "no", r.source)
        )
    w = [max(len(x[k]) for x in rows) for k in range(4)]
    out = []
    for x in rows:
        out.append("  ".join(x[k].ljust(w[k]) for k in range(4)) + "  " + x[4])
    out.append("")
    out.append(
        "Severity overrides: PUNCT-EM-DASH is a warning in .md files; SYM-R3 is an error in a .tex file whose header declares the td voice."
    )
    out.append(
        "Path exemptions: "
        + "; ".join(
            "%s %s" % (k, ", ".join(v))
            for k, v in PATH_EXEMPT.items()
            if v != [RULE_DOCS]
        )
        + "; the lexical rules skip %s." % RULE_DOCS
    )
    out.append(
        "Path restrictions: "
        + "; ".join("%s only %s" % (k, ", ".join(v)) for k, v in PATH_ONLY.items())
        + "."
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prose_lint.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--changed", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rules", default="")
    ap.add_argument("--min-severity", choices=["error", "warning"], default="warning")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hook", action="store_true")
    ap.add_argument("--list-rules", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if not a.hook and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.hook:
        try:
            out = run_hook(sys.stdin.read())
            if out:
                sys.stdout.write(out + "\n")
        except Exception as e:  # the hook never blocks or fails the edit
            sys.stderr.write(
                "prose_lint: internal error in the hook (%r); nothing reported\n" % e
            )
        return 0
    if a.list_rules:
        write_out(list_rules() + "\n")
        return 0
    try:
        return run_cli(a)
    except Exception as e:
        sys.stderr.write("prose_lint: internal error (%r)\n" % e)
        return 2


def run_cli(a: argparse.Namespace) -> int:
    root = repo_root()
    want = set(x.strip() for x in a.rules.split(",") if x.strip())
    unknown = want - set(RULES)
    if unknown:
        sys.stderr.write(
            "prose_lint: unknown rule id(s): %s\n" % ", ".join(sorted(unknown))
        )
        return 2
    targets: list[tuple[Path, set[int] | None]] = []
    if a.all:
        targets += [(p, None) for p in all_files(root)]
    if a.changed:
        targets += changed_files(root)
    for f in a.files:
        p = Path(f)
        if not p.exists():
            sys.stderr.write("prose_lint: no such file: %s\n" % f)
            return 2
        targets.append((p, None))
    if not targets and not (a.all or a.changed):
        sys.stderr.write("prose_lint: give FILES, --changed or --all (see --help)\n")
        return 2
    t0 = time.time()
    findings = []
    seen = set()
    for p, region in targets:
        key = os.path.normcase(os.path.abspath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        items = lint_file(p, root)
        items = [f for f in items if in_region(f, region, hook=False)]
        if want:
            items = [f for f in items if f.rule in want]
        if a.min_severity == "error":
            items = [f for f in items if f.severity == "error"]
        findings += items
    secs = time.time() - t0
    counts = Counter(f.rule for f in findings)
    errors = sum(1 for f in findings if f.severity == "error")
    if a.json:
        out = {
            "files": len(seen),
            "seconds": round(secs, 2),
            "errors": errors,
            "warnings": len(findings) - errors,
            "counts": dict(sorted(counts.items())),
            "counts_by_severity": dict(
                sorted(
                    Counter("%s:%s" % (f.rule, f.severity) for f in findings).items()
                )
            ),
            "findings": [f.as_dict() for f in findings],
        }
        write_out(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    else:
        by_file: dict[str, list[Finding]] = {}
        for f in findings:
            by_file.setdefault(f.rel, []).append(f)
        text = [
            format_findings(rel, by_file[rel], "") + "\n\n" for rel in sorted(by_file)
        ]
        text.append(
            "%d file(s), %d error(s), %d warning(s) in %.1f s.\n"
            % (len(seen), errors, len(findings) - errors, secs)
        )
        write_out("".join(text))
    return 1 if errors else 0


def write_out(text: str) -> None:
    """Write UTF-8 to stdout whatever the console code page is."""
    buf = getattr(sys.stdout, "buffer", None)
    if buf is None:
        sys.stdout.write(text)
        return
    sys.stdout.flush()
    buf.write(text.encode("utf8"))
    buf.flush()


if __name__ == "__main__":
    sys.exit(main())
