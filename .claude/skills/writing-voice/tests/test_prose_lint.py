"""Tests for prose_lint.py (standard library unittest; fixtures are built in code).

    python .claude/skills/writing-voice/tests/test_prose_lint.py            # run the suite
    python .claude/skills/writing-voice/tests/test_prose_lint.py --refresh-golden

The suite holds one positive and one negative case per rule, the PostToolUse hook
contract, the hook's timing on the largest notebook, the exemplar cells of
exemplars/notebook.md, parity of the SYM counts with symbol_check, the existence of the
imported style_check tells, and the frozen corpus counts in prose_lint_golden.json.
`--refresh-golden` rewrites the counts in the golden file and keeps its change log; a
refresh that moves a count is recorded in that log with the reason.
"""

import importlib.util, io, json, os, re, shutil, subprocess, sys, tempfile, time, unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
LINT = SCRIPTS / "prose_lint.py"
GOLDEN = SCRIPTS / "prose_lint_golden.json"
REPO = HERE.parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pl = _load("prose_lint_under_test", LINT)


# ------------------------------------------------------------------ fixtures -------------
def md(src):
    return {
        "cell_type": "markdown",
        "id": "m%d" % (abs(hash(src)) % 10**8),
        "metadata": {},
        "source": src.splitlines(True),
    }


def code(src, image=False):
    outs = (
        [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo=", "text/plain": "<Figure>"},
                "metadata": {},
            }
        ]
        if image
        else []
    )
    return {
        "cell_type": "code",
        "id": "c%d" % (abs(hash(src)) % 10**8),
        "metadata": {},
        "execution_count": None,
        "outputs": outs,
        "source": src.splitlines(True),
    }


def notebook(cells):
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


TITLE = md(
    "# A fixture notebook\n\n> **Work in progress** \u2014 an illustrative teaching notebook.\n\nThe notebook clears one market."
)
TABLE = md(
    "| Symbol | Code | Definition |\n|---|---|---|\n| $\\lambda$ | `res.energy_price` | the energy price |"
)
EDIT = code("# -- EDIT -- inputs\nSHED_PRICE = 150.0\n")
SANDBOX = md("## 5. Sandbox\n\nThe sandbox reruns the chain on the configured inputs.")


def wrap(*cells):
    """A notebook with the standard title, table, EDIT cell and sandbox around the test cells."""
    return notebook([TITLE, TABLE, EDIT] + list(cells) + [SANDBOX])


class Fixture(object):
    """A temporary repository root, set as PROSE_LINT_ROOT for the duration of a test."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="prose_lint_test_"))
        self._old = os.environ.get("PROSE_LINT_ROOT")
        os.environ["PROSE_LINT_ROOT"] = str(self.root)

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):
            content = json.dumps(content, indent=1, ensure_ascii=False)
        with io.open(p, "w", encoding="utf8", newline="\n") as fh:
            fh.write(content)
        return p

    def close(self):
        if self._old is None:
            os.environ.pop("PROSE_LINT_ROOT", None)
        else:
            os.environ["PROSE_LINT_ROOT"] = self._old
        shutil.rmtree(str(self.root), ignore_errors=True)


class LintCase(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()

    def tearDown(self):
        self.fx.close()

    def lint(self, rel, content):
        p = self.fx.write(rel, content)
        return pl.lint_file(p, self.fx.root)

    def rules(self, rel, content):
        return Counter(f.rule for f in self.lint(rel, content))

    def nb(
        self,
        *cells,
        name="150_fixture.ipynb",
        bare=False,
    ):
        return self.rules(name, notebook(list(cells)) if bare else wrap(*cells))

    def assertHit(self, rule, counts):
        self.assertIn(rule, counts, "expected %s in %s" % (rule, dict(counts)))

    def assertNoHit(self, rule, counts):
        self.assertNotIn(rule, counts, "unexpected %s in %s" % (rule, dict(counts)))


TEX_HEAD = "\\documentclass{article}\n%% Voice: %s\n\\begin{document}\n"


def tex(body, voice="the EDAM Draft Technical Description (TD)."):
    return TEX_HEAD % voice + body + "\n\\end{document}\n"


EQ1 = "$$x=1 \\qquad (1)$$"
EQ2 = "$$y=2 \\tag{2}$$"
BLURB = "*In words.* The quantity $x$ equals one."


# ------------------------------------------------------------------ rule sources -------
SOURCE_FILES = {
    "CLAUDE.md": "CLAUDE.md",
    "AGENTS.md": "AGENTS.md",
    "CANON.md": "CANON.md",
    "modify-example SKILL.md": ".claude/skills/modify-example/SKILL.md",
    "writing-voice SKILL.md": ".claude/skills/writing-voice/SKILL.md",
    "writing-voice register/cut_on_sight.md": ".claude/skills/writing-voice/register/cut_on_sight.md",
    "NOTEBOOK_RECORD.md": "NOTEBOOK_RECORD.md",
    "notes/writing_profile/source_register_profile.md": "notes/writing_profile/source_register_profile.md",
}


def _headings(path):
    return [
        re.sub(r"^#+\s*", "", ln).strip().lower()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.startswith("#")
    ]


class RuleSources(unittest.TestCase):
    """Each 'file, section N' or 'file, heading' source of a rule names a heading that exists in the file."""

    def test_sources_resolve(self):
        problems = []
        for rule in pl.RULES.values():
            for part in (s.strip() for s in rule.source.split(";")):
                head, sep, rest = part.partition(", ")
                if head not in SOURCE_FILES:
                    continue  # not a file pinpoint (a script, a scope note)
                path = REPO / SOURCE_FILES[head]
                if not path.exists():
                    problems.append("%s: %s names a missing file" % (rule.id, part))
                    continue
                heads = _headings(path)
                sec = re.match(r"section (\d+)", rest)
                if sec:
                    ok = any(h.startswith(sec.group(1) + ".") for h in heads)
                else:
                    label = re.sub(r"\s*\(.*$", "", rest).strip().lower()
                    label = re.sub(r"\s+\d+$", "", label)
                    label = re.sub(r"^the ", "", label)
                    ok = any(label in h for h in heads) or label in path.read_text(
                        encoding="utf-8"
                    ).lower()
                if not ok:
                    problems.append("%s: %s has no matching heading" % (rule.id, part))
        self.assertEqual(problems, [])


# ------------------------------------------------------------------ notebook rules -------
class NotebookMarkdownRules(LintCase):
    def test_math_delim(self):
        self.assertHit(
            "MATH-DELIM", self.nb(md("The price \\(\\lambda\\) is the dual."))
        )
        self.assertNoHit(
            "MATH-DELIM",
            self.nb(md("The price $\\lambda$ is the dual.\n\n$$a \\\\[2pt] b$$")),
        )

    def test_inw_missing(self):
        self.assertHit(
            "INW-MISSING", self.nb(md(EQ1 + "\n\nThe next paragraph states a fact."))
        )
        self.assertHit("INW-MISSING", self.nb(md(EQ1 + "\n\n" + EQ2 + "\n\n" + BLURB)))
        self.assertNoHit("INW-MISSING", self.nb(md(EQ1 + "\n\n" + BLURB)))
        # a code cell or a table between the display and its blurb
        self.assertHit("INW-MISSING", self.nb(md(EQ1), code("x = 1\n"), md(BLURB)))
        self.assertHit(
            "INW-MISSING",
            self.nb(md(EQ1 + "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n" + BLURB)),
        )
        self.assertNoHit("INW-MISSING", self.nb(md(EQ1), md(BLURB)))

    def test_inw_deferred(self):
        self.assertHit(
            "INW-DEFERRED", self.nb(md(EQ1 + "\n\nwhere $x$ is the flow.\n\n" + BLURB))
        )
        self.assertNoHit("INW-DEFERRED", self.nb(md(EQ1 + "\n\n" + BLURB)))

    def test_inw_orphan(self):
        self.assertHit("INW-ORPHAN", self.nb(md("The market clears.\n\n" + BLURB)))
        self.assertNoHit("INW-ORPHAN", self.nb(md("$$z=3$$\n\n" + BLURB)))
        second = (
            EQ1
            + "\n\n"
            + BLURB
            + "\n\n*In words, second form.* The same quantity read again."
        )
        self.assertNoHit("INW-ORPHAN", self.nb(md(second)))

    def test_inw_form(self):
        self.assertHit(
            "INW-FORM", self.nb(md(EQ1 + "\n\n**In words.** The quantity is one."))
        )
        self.assertNoHit("INW-FORM", self.nb(md(EQ1 + "\n\n" + BLURB)))

    def test_label_readit(self):
        self.assertHit(
            "LABEL-READIT", self.nb(md("**Read it.** The price rises to \\$37."))
        )
        self.assertNoHit("LABEL-READIT", self.nb(md("The price rises to \\$37.")))

    def test_label_leadin(self):
        self.assertHit("LABEL-LEADIN", self.nb(md("**Takeaway:** the price rises.")))
        self.assertNoHit(
            "LABEL-LEADIN", self.nb(md("**The superposed flow.** The flow is the sum."))
        )

    def test_nomen_table(self):
        self.assertHit(
            "NOMEN-TABLE",
            self.nb(TITLE, md(EQ1 + "\n\n" + BLURB), EDIT, SANDBOX, bare=True),
        )
        self.assertHit(
            "NOMEN-TABLE",
            self.nb(TITLE, md(EQ1 + "\n\n" + BLURB), TABLE, EDIT, SANDBOX, bare=True),
        )
        self.assertNoHit("NOMEN-TABLE", self.nb(md(EQ1 + "\n\n" + BLURB)))

    def test_eq_numbering(self):
        self.assertHit("EQ-NUMBERING", self.nb(md(EQ2 + "\n\n" + BLURB)))
        self.assertHit(
            "EQ-NUMBERING",
            self.nb(md(EQ1 + "\n\n" + BLURB + "\n\n" + EQ1 + "\n\n" + BLURB)),
        )
        self.assertNoHit(
            "EQ-NUMBERING",
            self.nb(md(EQ1 + "\n\n" + BLURB + "\n\n" + EQ2 + "\n\n" + BLURB)),
        )

    def test_xref_sibling_num(self):
        self.assertHit(
            "XREF-SIBLING-NUM",
            self.nb(md("The configuration is 231 Section 6's screen.")),
        )
        self.assertHit(
            "XREF-SIBLING-NUM", self.nb(md("The rule follows 131's equation (4)."))
        )
        self.assertNoHit(
            "XREF-SIBLING-NUM",
            self.nb(md("The configuration is 231's screen at 300 (5) MW.")),
        )

    def test_xref_external_eq(self):
        text = md("Ramping limits (the paper's (6l) and (6m)) are not included.")
        self.assertHit("XREF-EXTERNAL-EQ", self.nb(text))
        # no notebook is exempt: a paper's models are named, not numbered
        self.assertHit(
            "XREF-EXTERNAL-EQ",
            self.nb(md("The ramping limit is the paper's (6l)."), name="101_sandbox.ipynb"),
        )
        self.assertNoHit(
            "XREF-EXTERNAL-EQ", self.nb(md("The exhibit above is eq. (4) in numbers."))
        )

    def test_xref_sibling_desc(self):
        self.assertHit(
            "XREF-SIBLING-DESC",
            self.nb(md("The derivation lives in the allocation notebook.")),
        )
        self.assertNoHit(
            "XREF-SIBLING-DESC",
            self.nb(
                md(
                    "The derivation lives in the allocation notebook 231, and this notebook reads it."
                )
            ),
        )


class CanonRules(LintCase):
    def test_sf_roman(self):
        self.assertHit(
            "CANON-SF-ROMAN",
            self.nb(md("The shift factor $\\mathrm{SF}_{n,m}$ is fixed.")),
        )
        self.assertNoHit(
            "CANON-SF-ROMAN", self.nb(md("The shift factor $SF_{n,m}$ is fixed."))
        )

    def test_sf_order(self):
        self.assertHit(
            "CANON-SF-ORDER", self.nb(md("The shift factor $SF_{m,n}$ is fixed."))
        )
        self.assertNoHit(
            "CANON-SF-ORDER", self.nb(md("The shift factor $SF_{n,m}$ is fixed."))
        )

    def test_line_index(self):
        self.assertHit(
            "CANON-LINE-INDEX", self.nb(md("The flow $F_\\ell$ on line $\\ell$ binds."))
        )
        self.assertNoHit(
            "CANON-LINE-INDEX", self.nb(md("The HVDC flow $F^{dc}_\\ell$ is bounded."))
        )

    def test_interface_t(self):
        text = md("The interface set $\\mathcal T$ is declared.")
        self.assertHit("CANON-INTERFACE-T", self.nb(text))
        self.assertHit("CANON-INTERFACE-T", self.nb(text, name="101_sandbox.ipynb"))
        self.assertNoHit(
            "CANON-INTERFACE-T", self.nb(md("The interface set $\\mathcal I$ is declared."))
        )

    def test_margin_beta(self):
        text = md("The reservation margin $\\beta$ is swept.")
        self.assertHit(
            "CANON-MARGIN-BETA",
            self.nb(text, name="workshop2_congestion_management.ipynb"),
        )
        self.assertNoHit("CANON-MARGIN-BETA", self.nb(text))
        self.assertNoHit(
            "CANON-MARGIN-BETA",
            self.nb(
                md("The state-of-charge dual $\\tilde\\beta$ is read."),
                name="workshop2_congestion_management.ipynb",
            ),
        )

    def test_monitored(self):
        self.assertHit(
            "CANON-MONITORED", self.nb(md("The engine prices the monitored lines."))
        )
        self.assertNoHit(
            "CANON-MONITORED", self.nb(md("The engine prices the network line set."))
        )
        self.assertFalse(pl.RULES["CANON-MONITORED"].hook)

    def test_laga(self):
        self.assertHit(
            "CANON-LAGA",
            self.nb(md("The congestion rent of the region is $L_a - G_a$.")),
        )
        self.assertNoHit(
            "CANON-LAGA", self.nb(md("The net settlement position is $L_a - G_a$."))
        )


class LexiconRules(LintCase):
    def test_balancing_area(self):
        self.assertHit(
            "LEX-BALANCING-AREA",
            self.nb(md("Each balancing area clears its own engine.")),
        )
        self.assertNoHit(
            "LEX-BALANCING-AREA",
            self.nb(md('The ISO wrote "each balancing area clears" in 2023.')),
        )
        self.assertNoHit(
            "LEX-BALANCING-AREA",
            self.nb(md("The Integrated Balancing Area is a proper name.")),
        )

    def test_retired_frame(self):
        self.assertHit(
            "LEX-RETIRED-FRAME",
            self.nb(md("The two regimes are allocation versus coordination.")),
        )
        self.assertNoHit(
            "LEX-RETIRED-FRAME",
            self.nb(md("The allocation of rent follows the dispatch.")),
        )

    def test_transmission_owner(self):
        self.assertHit(
            "LEX-TRANSMISSION-OWNER",
            self.nb(md("The transmission owner receives the rent.")),
        )
        self.assertNoHit(
            "LEX-TRANSMISSION-OWNER",
            self.nb(md("The Participating Transmission Owner receives the rent.")),
        )

    def test_network_metaphor(self):
        self.assertHit(
            "LEX-NETWORK-METAPHOR",
            self.nb(md("The 500 kV system is the backbone of the West.")),
        )
        self.assertNoHit(
            "LEX-NETWORK-METAPHOR",
            self.nb(md("The 500 kV facilities carry the flow contribution.")),
        )

    def test_euro_kv_scoped_out(self):
        # no 9-bus file names a line class by voltage, so the rule is path-exempt everywhere
        self.assertNoHit("LEX-EURO-KV", self.nb(md("The class is the 380 kV type.")))
        self.assertNoHit("LEX-EURO-KV", self.nb(md("The class is the 345 kV type.")))
        # the detector itself still works once the exemption is lifted
        saved = pl.PATH_EXEMPT.pop("LEX-EURO-KV")
        try:
            self.assertHit(
                "LEX-EURO-KV", self.nb(md("The class is the 380 kV type."))
            )
        finally:
            pl.PATH_EXEMPT["LEX-EURO-KV"] = saved

    def test_cut_phrases(self):
        self.assertHit("CUT-PHRASES", self.nb(md("Crucially, the price separates.")))
        self.assertHit(
            "CUT-PHRASES", self.nb(md("The two rules are two sides of the same coin."))
        )
        self.assertNoHit(
            "CUT-PHRASES", self.nb(md("The price separates when the line binds."))
        )
        self.assertNoHit(
            "CUT-PHRASES",
            self.nb(md('Hogan writes that "most importantly, the rent is collected".')),
        )

    def test_reader_address(self):
        self.assertHit(
            "CUT-READER-ADDRESS", self.nb(md("Notice that the price separates."))
        )
        self.assertNoHit("CUT-READER-ADDRESS", self.nb(md("The price separates.")))

    def test_imperative(self):
        self.assertHit(
            "CUT-IMPERATIVE", self.nb(md("Compare the two ledgers. The prices agree."))
        )
        self.assertNoHit(
            "CUT-IMPERATIVE", self.nb(md("Consider the case of one binding line."))
        )
        self.assertNoHit(
            "CUT-IMPERATIVE",
            self.nb(md("The ledgers differ.\nCompare is not a sentence opener here.")),
        )

    def test_first_person(self):
        self.assertHit(
            "VOICE-FIRST-PERSON", self.nb(md("Below the figure we print the ledger."))
        )
        self.assertNoHit(
            "VOICE-FIRST-PERSON", self.nb(md("The class is named by its US voltage."))
        )


class PunctuationAndCitationRules(LintCase):
    def test_em_dash(self):
        self.assertHit(
            "PUNCT-EM-DASH",
            self.nb(md("The dual \u2014 the shadow price \u2014 is signed.")),
        )
        self.assertHit(
            "PUNCT-EM-DASH",
            self.nb(md("**The flow.** The dual \u2014 the price \u2014 is signed.")),
        )
        clean = md(
            "## 2. The dual \u2014 a title\n\n| a | b \u2014 c |\n|---|---|\n\n**The flow \u2014 superposed.** The flow is the sum."
        )
        self.assertNoHit("PUNCT-EM-DASH", self.nb(clean))
        # the work-in-progress banner of the title cell is exempt (TITLE carries it)
        self.assertNoHit("PUNCT-EM-DASH", self.nb())

    def test_em_dash_severity_by_kind(self):
        f = [
            x
            for x in self.lint("notes/x.md", "# T\n\nThe dual \u2014 the price.\n")
            if x.rule == "PUNCT-EM-DASH"
        ]
        self.assertEqual([x.severity for x in f], ["warning"])
        f = [
            x
            for x in self.lint("notes/x.tex", tex("The dual---the price---is signed."))
            if x.rule == "PUNCT-EM-DASH"
        ]
        self.assertEqual([x.severity for x in f], ["error"])
        self.assertEqual(
            [
                x.rule
                for x in self.lint(
                    ".claude/skills/x/SKILL.md", "The dual \u2014 the price.\n"
                )
            ],
            [],
        )

    def test_dash_aside(self):
        self.assertHit(
            "PUNCT-DASH-ASIDE",
            self.nb(md("The dual \u2013 the price \u2013 is signed.")),
        )
        self.assertNoHit("PUNCT-DASH-ASIDE", self.nb(md("Line 6\u20137 binds.")))

    def test_percent(self):
        mentor = "the mentor primer (notes/writing_profile/mentor_primer.md)."
        self.assertHit(
            "NUM-PERCENT",
            self.rules(
                "notes/a.tex", tex("The share is 5\\% of the rent.", voice=mentor)
            ),
        )
        self.assertNoHit(
            "NUM-PERCENT",
            self.rules("notes/a.tex", tex("The share is 5\\% of the rent.")),
        )

    def test_thousands(self):
        mentor = "the mentor primer."
        self.assertHit(
            "NUM-THOUSANDS",
            self.rules("notes/a.tex", tex("The path carries 2500 MW.", voice=mentor)),
        )
        self.assertNoHit(
            "NUM-THOUSANDS",
            self.rules("notes/a.tex", tex("The path carries 2,500 MW.", voice=mentor)),
        )

    def test_author_date(self):
        self.assertHit(
            "CITE-AUTHOR-DATE",
            self.nb(md("Mays (2024) shows that the price is sequential.")),
        )
        self.assertNoHit(
            "CITE-AUTHOR-DATE", self.nb(md("The ISO (2023) states the design."))
        )
        self.assertNoHit(
            "CITE-AUTHOR-DATE",
            self.rules(
                "notes/a.tex",
                tex(
                    "\\begin{itemize}\n\\item Hogan (1998): The price is the opportunity cost.\n\\end{itemize}"
                ),
            ),
        )


class NotebookCodeRules(LintCase):
    def test_caption_cell(self):
        self.assertHit(
            "NB-CAPTION-CELL",
            self.nb(code("plot()", image=True), md("The figure shows the dispatch.")),
        )
        self.assertNoHit(
            "NB-CAPTION-CELL",
            self.nb(code("plot()"), md("The figure shows the dispatch.")),
        )

    def test_edit_knobs(self):
        self.assertHit("NB-EDIT-KNOBS", self.nb(TITLE, TABLE, SANDBOX, bare=True))
        self.assertNoHit("NB-EDIT-KNOBS", self.nb())
        self.assertFalse(pl.RULES["NB-EDIT-KNOBS"].hook)

    def test_sandbox(self):
        self.assertHit("NB-SANDBOX", self.nb(TITLE, TABLE, EDIT, bare=True))
        self.assertNoHit("NB-SANDBOX", self.nb())
        self.assertFalse(pl.RULES["NB-SANDBOX"].hook)

    def test_mplbackend(self):
        self.assertHit(
            "CODE-MPLBACKEND",
            self.nb(code('import matplotlib\nmatplotlib.use("Agg")\n')),
        )
        self.assertHit(
            "CODE-MPLBACKEND", self.nb(code("os.environ['MPLBACKEND'] = 'Agg'\n"))
        )
        self.assertNoHit(
            "CODE-MPLBACKEND",
            self.nb(code("# never set MPLBACKEND=Agg here\nplt.show()\n")),
        )

    def test_export_cwd(self):
        self.assertHit(
            "CODE-EXPORT-CWD", self.nb(code('fig.savefig("outputs/x.png", dpi=200)\n'))
        )
        self.assertHit(
            "CODE-EXPORT-CWD",
            self.nb(code("OUT = Path('outputs'); OUT.mkdir(exist_ok=True)\n")),
        )
        self.assertHit(
            "CODE-EXPORT-CWD",
            self.nb(code("OUT = os.path.join(os.getcwd(), 'outputs')\n")),
        )
        walk = "d = Path.cwd()\nwhile not (d / 'notes').is_dir():\n    d = d.parent\nOUT = d / 'outputs'\nfig.savefig(OUT / 'x.png')\nfig.savefig(f\"{OUT}/y.png\")\n"
        self.assertNoHit("CODE-EXPORT-CWD", self.nb(code(walk)))

    def test_shed_price(self):
        cell = code("SHED_PRICE = 300\n")
        self.assertHit("CODE-SHED-PRICE", self.nb(cell, name="101_sandbox.ipynb"))
        self.assertHit("CODE-SHED-PRICE", self.nb(cell, name="222_x.ipynb"))
        self.assertNoHit(
            "CODE-SHED-PRICE", self.nb(cell, name="sandbox_exhibits/x/201_x.ipynb")
        )
        self.assertNoHit("CODE-SHED-PRICE", self.nb(code("SHED_PRICE = 150.0\n")))


class TexAndBibRules(LintCase):
    def test_one_sentence(self):
        self.assertHit(
            "TEX-ONE-SENTENCE",
            self.rules("notes/a.tex", tex("The line binds. The price separates.")),
        )
        self.assertHit(
            "TEX-ONE-SENTENCE",
            self.rules("notes/a.tex", tex("The line binds and the\nprice separates.")),
        )
        self.assertNoHit(
            "TEX-ONE-SENTENCE",
            self.rules(
                "notes/a.tex",
                tex("The line binds, e.g. at 40 MW.\nThe price separates."),
            ),
        )

    def test_voice_header(self):
        self.assertHit(
            "TEX-VOICE-HEADER",
            self.rules(
                "notes/a.tex",
                "\\documentclass{article}\n\\begin{document}\nText.\n\\end{document}\n",
            ),
        )
        self.assertNoHit("TEX-VOICE-HEADER", self.rules("notes/a.tex", tex("Text.")))
        self.assertNoHit(
            "TEX-VOICE-HEADER",
            self.rules("notes/part.tex", "A part file has no preamble.\n"),
        )

    def test_eqref(self):
        self.assertHit(
            "TEX-EQREF",
            self.rules("notes/a.tex", tex("The rent follows from \\ref{eq:rent}.")),
        )
        self.assertNoHit(
            "TEX-EQREF",
            self.rules("notes/a.tex", tex("The rent follows from \\eqref{eq:rent}.")),
        )

    def test_sym_r1(self):
        self.assertHit("SYM-R1", self.rules("notes/a.tex", tex("The value $k$ rises.")))
        self.assertNoHit(
            "SYM-R1", self.rules("notes/a.tex", tex("The tariff, $k$, rises."))
        )

    def test_sym_r2(self):
        self.assertHit(
            "SYM-R2",
            self.rules("notes/a.tex", tex("The right is exercised when $v_X > k$.")),
        )
        self.assertNoHit(
            "SYM-R2",
            self.rules(
                "notes/a.tex",
                tex("The value of the right, $v$, exceeds the tariff, $k$."),
            ),
        )

    def test_sym_r3(self):
        bad = (
            "\\begin{equation}\nR = 1 \\label{eq:r}\n\\end{equation}\nThe rent is one."
        )
        good = bad.replace(
            "The rent is one.",
            "\\begin{inwords}\nEquation~\\eqref{eq:r} states that the rent is one.\n\\end{inwords}",
        )
        hits = [f for f in self.lint("notes/a.tex", tex(bad)) if f.rule == "SYM-R3"]
        self.assertEqual([f.severity for f in hits], ["error"])  # td voice
        hits = [
            f
            for f in self.lint("notes/b.tex", tex(bad, voice="the MSC voice."))
            if f.rule == "SYM-R3"
        ]
        self.assertEqual([f.severity for f in hits], ["warning"])
        self.assertNoHit("SYM-R3", self.rules("notes/a.tex", tex(good)))
        # a labelled equation and a labelled align that share one In-words block
        shared = (
            "\\begin{equation}\nR = 1 \\label{eq:a}\n\\end{equation}\n\\begin{align}\nS &= 2 \\label{eq:b}\n\\end{align}\n"
            "\\begin{inwords}\nEquation~\\eqref{eq:a} states that the rent is one, and \\eqref{eq:b} that the surplus is two.\n\\end{inwords}"
        )
        self.assertNoHit("SYM-R3", self.rules("notes/a.tex", tex(shared)))

    def test_xref_sibling_num_tex(self):
        for s in (
            "The rule is 132's equation~(3).",
            "The screen is 231~Section~6.",
            "The screen is Section~6 of 231.",
            "The screen is 231's \\S6.",
            "The screen is 231's \\S 6.",
        ):
            self.assertHit("XREF-SIBLING-NUM", self.rules("notes/a.tex", tex(s)))
        self.assertNoHit(
            "XREF-SIBLING-NUM",
            self.rules("notes/a.tex", tex("The screen in 231 prices the path.")),
        )

    def test_cite_key_missing(self):
        self.fx.write(
            "sources/references.bib",
            "@article{hogan2018primer,\n  note = {A note.},\n}\n@techreport{b2,\n  note = {n},\n}\n",
        )
        self.assertHit(
            "CITE-KEY-MISSING",
            self.rules("notes/a.tex", tex("The rent accrues \\citep{nosuchkey}.")),
        )
        self.assertHit(
            "CITE-KEY-MISSING",
            self.rules(
                "notes/a.tex",
                tex("The rent accrues \\citep[p.~5]{hogan2018primer, nosuchkey}."),
            ),
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.rules(
                "notes/a.tex",
                tex("The rent accrues \\citep[p.~5]{hogan2018primer,b2}."),
            ),
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.rules("notes/a.tex", tex("The rent accrues. % \\citep{nosuchkey}")),
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.rules(
                "notes/a.tex",
                tex(
                    "% lint: allow CITE-KEY-MISSING\nThe rent accrues \\citep{nosuchkey}."
                ),
            ),
        )
        self.fx.write(
            "sources/references.bib",
            "@article{hogan2018primer,\n  note = {A note.},\n}\n@article{newkey,\n  note = {n},\n}\n",
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.rules("notes/a.tex", tex("The rent accrues \\citep{newkey}.")),
        )

    def test_cite_key_missing_backticked_in_notebook_markdown(self):
        self.fx.write(
            "sources/references.bib",
            "@article{caiso_dame_edam2023,\n  note = {A note.},\n}\n",
        )
        self.assertHit(
            "CITE-KEY-MISSING",
            self.nb(md("The transmittal letter, `caiso_nosuch2023`, states the rule.")),
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.nb(md("The transmittal letter, `caiso_dame_edam2023`, states the rule.")),
        )
        # a backticked code name with no year is not read as a key
        self.assertNoHit(
            "CITE-KEY-MISSING", self.nb(md("The function `solve_engine_transfers` clears."))
        )
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.nb(md("<!-- lint: allow CITE-KEY-MISSING -->\nSee `caiso_nosuch2023`.")),
        )

    def test_cite_key_missing_silent_without_bibliography(self):
        self.assertNoHit(
            "CITE-KEY-MISSING",
            self.rules("notes/a.tex", tex("The rent accrues \\citep{nosuchkey}.")),
        )

    def test_bib_note(self):
        self.assertHit(
            "BIB-NOTE-SAFE",
            self.rules(
                "sources/references.bib",
                "@article{a,\n  note = {Half of the rent, 50% of R_T.},\n}\n",
            ),
        )
        self.assertNoHit(
            "BIB-NOTE-SAFE",
            self.rules(
                "sources/references.bib",
                "@article{a,\n  note = {Half of the rent, 50\\% of $R_T$.},\n}\n",
            ),
        )


class PragmasAndRegistry(LintCase):
    def test_line_pragma(self):
        self.assertNoHit(
            "PUNCT-EM-DASH",
            self.nb(
                md("<!-- lint: allow PUNCT-EM-DASH -->\nThe dual \u2014 the price.")
            ),
        )
        self.assertNoHit(
            "TEX-EQREF",
            self.rules(
                "notes/a.tex", tex("% lint: allow TEX-EQREF\nSee \\ref{eq:rent}.")
            ),
        )

    def test_file_pragma(self):
        self.assertNoHit(
            "CANON-MONITORED",
            self.nb(
                md("<!-- lint: allow-file CANON-MONITORED -->"),
                md("The monitored lines bind."),
            ),
        )

    def test_every_rule_has_a_test(self):
        src = pl.read_text(__file__)
        # each rule needs a positive case (assertHit, or a severity check on f.rule) and a negative case (assertNoHit)
        missing = [
            r
            for r in pl.RULES
            if not (
                re.search(
                    rf'(assertHit\(\s*"{re.escape(r)}"|f\.rule == "{re.escape(r)}")',
                    src,
                )
                and re.search(rf'assertNoHit\(\s*"{re.escape(r)}"', src)
            )
        ]
        self.assertEqual(missing, [])

    def test_first_errors(self):
        errors = {r.id for r in pl.RULES.values() if r.severity == "error"}
        self.assertEqual(
            errors,
            {
                "MATH-DELIM",
                "INW-MISSING",
                "INW-FORM",
                "LABEL-READIT",
                "NOMEN-TABLE",
                "EQ-NUMBERING",
                "XREF-SIBLING-NUM",
                "CANON-SF-ROMAN",
                "PUNCT-EM-DASH",
                "CODE-MPLBACKEND",
                "CODE-EXPORT-CWD",
                "BIB-NOTE-SAFE",
            },
        )

    def test_list_rules(self):
        r = subprocess.run(
            [sys.executable, str(LINT), "--list-rules"], capture_output=True
        )
        self.assertEqual(r.returncode, 0)
        out = r.stdout.decode("utf8")
        for rid in pl.RULES:
            self.assertIn(rid, out)

    def test_cli_exit_codes(self):
        bad = self.fx.write("notes/bad.md", "The dual \u2014 the price.\n")
        nb = self.fx.write(
            "150_x.ipynb",
            wrap(md("The dual \u2014 the price.")),
        )
        env = dict(os.environ)
        run = lambda *a: (
            subprocess.run(
                [sys.executable, str(LINT)] + list(a), capture_output=True, env=env
            ).returncode
        )
        self.assertEqual(run(str(bad)), 0)  # a warning only
        self.assertEqual(run(str(nb)), 1)  # an error
        self.assertEqual(run(str(nb), "--rules", "CANON-MONITORED"), 0)
        self.assertEqual(run("--rules", "NO-SUCH-RULE", str(nb)), 2)
        self.assertEqual(run("--no-such-option"), 2)
        self.assertEqual(run(), 2)

    def test_imported_tells_exist(self):
        sc = _load("style_check_for_test", SCRIPTS / "style_check.py")
        names = {n for n, _ in sc.TELLS}
        for rid, wanted in pl.TELL_NAMES.items():
            for n in wanted:
                self.assertIn(n, names, "%s imports the missing tell %r" % (rid, n))


# ------------------------------------------------------------------ the hook -------------
def run_hook(payload, env_root=None, raw=None):
    env = dict(os.environ)
    if env_root is not None:
        env["PROSE_LINT_ROOT"] = str(env_root)
    else:
        env.pop("PROSE_LINT_ROOT", None)
    data = raw if raw is not None else json.dumps(payload)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(LINT), "--hook"],
        input=data.encode("utf8"),
        capture_output=True,
        env=env,
    )
    return (
        r.returncode,
        r.stdout.decode("utf8"),
        r.stderr.decode("utf8"),
        time.time() - t0,
    )


class HookContract(LintCase):
    def payload(self, tool, path, **ti):
        key = "notebook_path" if tool == "NotebookEdit" else "file_path"
        tool_input = dict(ti)
        tool_input[key] = str(path).replace("/", "\\")
        return {
            "session_id": "t",
            "hook_event_name": "PostToolUse",
            "tool_name": tool,
            "tool_input": tool_input,
            "tool_response": {"filePath": str(path), "success": True},
            "cwd": str(self.fx.root),
        }

    def context(self, out):
        self.assertTrue(out.strip(), "expected hook output")
        d = json.loads(out)
        self.assertEqual(d["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        return d["hookSpecificOutput"]["additionalContext"]

    def test_write_dirty_and_clean(self):
        p = self.fx.write(
            "notes/a.md", "# T\n\nThe 500 kV system is the backbone of the West.\n"
        )
        code_, out, _, _ = run_hook(self.payload("Write", p, content=""), self.fx.root)
        self.assertEqual(code_, 0)
        self.assertIn("LEX-NETWORK-METAPHOR", self.context(out))
        p = self.fx.write(
            "notes/b.md", "# T\n\nThe 500 kV facilities carry the flow.\n"
        )
        code_, out, _, _ = run_hook(self.payload("Write", p, content=""), self.fx.root)
        self.assertEqual((code_, out), (0, ""))

    def test_edit_scopes_to_touched_lines(self):
        text = "# T\n\nThe dual \u2014 the price.\n\nThe line binds at 40 MW.\n"
        p = self.fx.write("notes/a.md", text)
        code_, out, _, _ = run_hook(
            self.payload(
                "Edit", p, old_string="binds", new_string="The line binds at 40 MW."
            ),
            self.fx.root,
        )
        # the pre-existing dash is outside the edit
        self.assertEqual((code_, out), (0, ""))
        code_, out, _, _ = run_hook(
            self.payload(
                "Edit", p, old_string="x", new_string="The dual \u2014 the price."
            ),
            self.fx.root,
        )
        self.assertEqual(code_, 0)
        self.assertIn("PUNCT-EM-DASH", self.context(out))

    def test_multiedit(self):
        p = self.fx.write(
            "notes/a.tex",
            tex("The line binds.\nCrucially, the price separates.\nSee \\ref{eq:x}."),
        )
        edits = [
            {"old_string": "a", "new_string": "Crucially, the price separates."},
            {"old_string": "b", "new_string": "See \\ref{eq:x}."},
        ]
        code_, out, _, _ = run_hook(
            self.payload("MultiEdit", p, edits=edits), self.fx.root
        )
        ctx = self.context(out)
        self.assertEqual(code_, 0)
        self.assertIn("CUT-PHRASES", ctx)
        self.assertIn("TEX-EQREF", ctx)
        self.assertNotIn("TEX-VOICE-HEADER", ctx)

    def test_edit_on_notebook_json(self):
        nbk = wrap(
            md("The dual \u2014 the price."), md("The monitored line binds at 40 MW.")
        )
        p = self.fx.write("150_x.ipynb", nbk)
        code_, out, _, _ = run_hook(
            self.payload("Edit", p, old_string="x", new_string="binds at 40 MW."),
            self.fx.root,
        )
        # CANON-MONITORED does not run in the hook
        self.assertEqual((code_, out), (0, ""))
        code_, out, _, _ = run_hook(
            self.payload("Edit", p, old_string="x", new_string="the price."),
            self.fx.root,
        )
        self.assertIn("PUNCT-EM-DASH", self.context(out))

    def test_notebookedit(self):
        dirty = "The dual \u2014 the price."
        clean = "The line binds at 40 MW."
        nbk = wrap(md(EQ1 + "\n\n" + BLURB), md(dirty), md(clean))
        p = self.fx.write("150_x.ipynb", nbk)
        code_, out, _, _ = run_hook(
            self.payload(
                "NotebookEdit", p, cell_id="x", new_source=clean, edit_mode="replace"
            ),
            self.fx.root,
        )
        self.assertEqual((code_, out), (0, ""))
        code_, out, _, _ = run_hook(
            self.payload(
                "NotebookEdit", p, cell_id="x", new_source=dirty, edit_mode="replace"
            ),
            self.fx.root,
        )
        self.assertIn("PUNCT-EM-DASH", self.context(out))

    def test_notebookedit_adjacency(self):
        # the blurb cell after a numbered display is replaced with prose: INW-MISSING spans both cells
        prose = "The next paragraph states a fact."
        nbk = wrap(md(EQ1), md(prose))
        p = self.fx.write("150_x.ipynb", nbk)
        code_, out, _, _ = run_hook(
            self.payload(
                "NotebookEdit", p, cell_id="x", new_source=prose, edit_mode="replace"
            ),
            self.fx.root,
        )
        self.assertIn("INW-MISSING", self.context(out))

    def test_finding_cap(self):
        lines = "\n".join("Line %d \u2014 a dash." % i for i in range(30))
        p = self.fx.write("notes/a.tex", tex(lines))
        code_, out, _, _ = run_hook(self.payload("Write", p, content=""), self.fx.root)
        ctx = self.context(out)
        self.assertEqual(
            sum(1 for l in ctx.split("\n") if l.startswith("- ")), pl.HOOK_MAX_FINDINGS
        )
        self.assertRegex(ctx, r"\.\.\. and \d+ more")

    def test_ignored_paths(self):
        outside = Path(tempfile.mkdtemp(prefix="prose_lint_outside_"))
        try:
            q = outside / "x.md"
            q.write_text("The dual \u2014 the price.\n", encoding="utf8")
            self.assertEqual(
                run_hook(self.payload("Write", q, content=""), self.fx.root)[:2],
                (0, ""),
            )
        finally:
            shutil.rmtree(str(outside), ignore_errors=True)
        for rel in (
            "outputs/x.md",
            "sources/academic-literature/x.md",
            "presentation_exhibits/d/x.md",
            "sandbox_exhibits/x.md",
            "notes/x.py",
        ):
            p = self.fx.write(rel, "The dual \u2014 the price.\nCrucially, it binds.\n")
            self.assertEqual(
                run_hook(self.payload("Write", p, content=""), self.fx.root)[:2],
                (0, ""),
                rel,
            )

    def test_malformed_and_unrelated_input(self):
        code_, out, err, _ = run_hook(None, self.fx.root, raw="{not json")
        self.assertEqual((code_, out), (0, ""))
        self.assertIn("could not parse", err)
        self.assertEqual(
            run_hook(
                {"tool_name": "Bash", "tool_input": {"command": "ls"}}, self.fx.root
            )[:2],
            (0, ""),
        )
        self.assertEqual(run_hook([1, 2], self.fx.root)[:2], (0, ""))


class HookTiming(unittest.TestCase):
    def test_largest_notebook_under_one_second(self):
        cands = sorted(REPO.glob("*.ipynb"), key=lambda q: q.stat().st_size)
        if not cands:
            self.skipTest("no notebook at the repository root")
        p = cands[-1]
        cells = json.loads(pl.read_text(p))["cells"]
        ci = next(
            i
            for i, c in enumerate(cells)
            if c["cell_type"] == "markdown" and "$$" in "".join(c["source"])
        )
        payload = {
            "tool_name": "NotebookEdit",
            "hook_event_name": "PostToolUse",
            "tool_input": {
                "notebook_path": str(p),
                "cell_id": cells[ci].get("id"),
                "new_source": "".join(cells[ci]["source"]),
                "edit_mode": "replace",
            },
            "tool_response": {},
        }
        best = min(run_hook(payload)[3] for _ in range(2))
        self.assertLess(best, 1.0, "hook took %.2f s on %s" % (best, p.name))


# ------------------------------------------------------------------ corpus checks --------
class Exemplars(unittest.TestCase):
    def test_exemplar_cells_have_no_errors(self):
        nbp = next(iter(sorted(REPO.glob("*nodal_market_fundamentals.ipynb"))), REPO / "absent.ipynb")
        if not nbp.exists():
            self.skipTest("the exemplar notebook is in commit ce66306, not the working tree")
        ex = pl.read_text(HERE.parent / "exemplars" / "notebook.md")
        cells = {
            int(m.group(1))
            for m in re.finditer(r"^\|\s*NB-\d+\s*\|[^|]*\|\s*(\d+)\s*\|", ex, re.M)
        }
        self.assertTrue(cells)
        errs = [
            f
            for f in pl.lint_file(nbp, REPO)
            if f.severity == "error" and f.cell in cells
        ]
        self.assertEqual([(f.rule, f.pin(), f.excerpt) for f in errs], [])


_CORPUS = {}


def corpus_report():
    """One --all --json run, shared by the golden and parity tests."""
    if "r" not in _CORPUS:
        env = dict(os.environ)
        env.pop("PROSE_LINT_ROOT", None)
        r = subprocess.run(
            [sys.executable, str(LINT), "--all", "--json"],
            capture_output=True,
            env=env,
            cwd=str(HERE),
        )
        if r.returncode not in (0, 1):
            raise RuntimeError(r.stderr.decode("utf8", "replace"))
        _CORPUS["r"] = json.loads(r.stdout.decode("utf8"))
    return _CORPUS["r"]


class Corpus(unittest.TestCase):
    def test_full_sweep_time(self):
        self.assertLess(corpus_report()["seconds"], 20.0)

    def test_golden_counts(self):
        golden = json.loads(pl.read_text(GOLDEN))
        now = corpus_report()["counts_by_severity"]
        moved = {
            k: (golden["counts_by_severity"].get(k, 0), now.get(k, 0))
            for k in set(golden["counts_by_severity"]) | set(now)
            if golden["counts_by_severity"].get(k, 0) != now.get(k, 0)
        }
        self.assertEqual(
            moved,
            {},
            "per-rule counts moved (golden, now). A rule change records why in the golden "
            "file's changes list; refresh with --refresh-golden.",
        )

    def test_sym_parity_with_symbol_check(self):
        sc = _load("symbol_check_for_test", SCRIPTS / "symbol_check.py")
        direct = Counter()
        for p in sorted((REPO / "notes").glob("**/*.tex")):
            if pl.in_scope(pl.rel_of(p, REPO)):
                direct.update("SYM-" + r for _, r, _, _ in sc.check_file(p))
        counts = corpus_report()["counts"]
        self.assertEqual(
            {k: counts.get(k, 0) for k in ("SYM-R1", "SYM-R2", "SYM-R3")},
            {k: direct.get(k, 0) for k in ("SYM-R1", "SYM-R2", "SYM-R3")},
        )


def refresh_golden():
    old = json.loads(pl.read_text(GOLDEN)) if GOLDEN.exists() else {}
    rep = corpus_report()
    out = {
        "_about": old.get("_about", ""),
        "generated": time.strftime("%Y-%m-%d"),
        "command": "python .claude/skills/writing-voice/scripts/prose_lint.py --all --json",
        "files": rep["files"],
        "seconds": rep["seconds"],
        "counts": rep["counts"],
        "counts_by_severity": rep["counts_by_severity"],
        "changes": old.get("changes", []),
    }
    with io.open(GOLDEN, "w", encoding="utf8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", GOLDEN)


if __name__ == "__main__":
    if "--refresh-golden" in sys.argv:
        refresh_golden()
    else:
        unittest.main()
