#!/usr/bin/env python
"""Post-draft style diagnostic for prose, read against the voice's own corpus.

    python .claude/skills/writing-voice/scripts/style_check.py <draft.tex|.md|.txt> --voice notebook
    python .claude/skills/writing-voice/scripts/style_check.py --rebuild        # recompute corpus_stats.json

The check runs AFTER drafting. It never sets a target. Each distributional measure of the
draft (sentence length and its spread, lexical diversity, function-word profile, the rates
of participial clauses, nominalizations, connectives and punctuation) is reported beside the
corpus's median and its 10th to 90th percentile band, computed over 600-word windows of the
voice's exemplar documents, and flagged when the draft falls outside the band. The catalogued
tells of machine prose (the register profile's cut-on-sight list, the editors' lists of AI
tells, the pattern lists of blader/humanizer (MIT, see LICENSE-humanizer.txt), the stylometry literature's features) are reported as counts with the offending
sentences quoted, so the revise step can edit the sentence toward the nearest exemplar.

Standard library only. `--rebuild` reads the documents named in corpus_sources.json, resolved
against the repository root: a notebook's markdown cells with the standard library, and a PDF
with pymupdf.
"""

from __future__ import annotations

import argparse, io, json, math, os, re, statistics, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ROOT = SKILL.parents[2]
STATS = SKILL / "corpus_stats.json"
SOURCES = SKILL / "corpus_sources.json"
WINDOW = 600

# ------------------------------------------------------------------ text preparation ----
ABBREV = r"(?:No|Nos|p|pp|Fig|Figs|Sec|Secs|vs|cf|e\.g|i\.e|et al|Dr|Mr|Ms|Inc|Co|Corp|Ltd|U\.S|P|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\."


def detach_footnotes(t: str) -> str:
    """Move each \\footnote{...} out of its sentence into a paragraph of its own at the end."""
    out, notes, i = [], [], 0
    tag = "\\footnote{"
    while True:
        j = t.find(tag, i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k = j + len(tag)
        d = 1
        while d and k < len(t):
            d += {"{": 1, "}": -1}.get(t[k], 0)
            k += 1
        notes.append(t[j + len(tag) : k - 1])
        i = k
    return "".join(out) + "\n\n" + "\n\n".join(notes)


def strip_latex(t: str) -> str:
    t = re.sub(r"(?m)^%.*$", "", t)
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = t.split(r"\begin{document}", 1)[-1]
    t = re.split(r"\\bibliographystyle|\\bibliography\{|\\end\{document\}", t)[0]
    t = re.sub(
        r"\\begin\{(longtable|tabular|table|figure|equation|align)\*?\}.*?\\end\{\1\*?\}",
        " ",
        t,
        flags=re.S,
    )
    t = re.sub(
        r"\\(cite[pt]?|ref|eqref|label|pageref|href|url|path|includegraphics)\*?(\[[^\]]*\])*\{[^}]*\}",
        "",
        t,
    )
    t = detach_footnotes(t)  # footnote prose becomes its own paragraphs
    t = re.sub(
        r"\\(section|subsection|paragraph|principle|element|item|caption)\*?(\[[^\]]*\])?\{[^}]*\}",
        "\n\n",
        t,
    )
    t = re.sub(r"\\(textbf|textit|emph)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = (
        t.replace("``", '"')
        .replace("''", '"')
        .replace("~", " ")
        .replace("--", "\u2013")
    )
    t = re.sub(r"[{}]", "", t)
    return t


def strip_markdown(t: str) -> str:
    t = re.sub(r"(?m)^\s*#+ .*$", "\n\n", t)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"(?m)^\s*\|.*$", "", t)
    t = re.sub(r"(?m)^\s*[-*] ", "", t)
    t = re.sub(r"\*\*([^*]*)\*\*", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.replace("`", "")
    return t


def load_text(path: Path) -> tuple[str, int]:
    raw = io.open(path, encoding="utf8", errors="replace").read()
    bold = len(re.findall(r"\\textbf\{|\*\*[^*]+\*\*", raw))
    if path.suffix == ".tex":
        t = strip_latex(raw)
    elif path.suffix in (".md", ".markdown"):
        t = strip_markdown(raw)
    else:
        t = raw
    return t, bold


def clean_corpus(t: str) -> str:
    """Drop PDF page markers, running heads, page numbers, and the reference list."""
    t = re.sub(r"===== PDF PAGE \d+ =====", "\n\n", t)
    lines = [l.strip() for l in t.split("\n")]
    freq = Counter(l for l in lines if 0 < len(l) < 90)
    out = []
    for l in lines:
        if not l:
            out.append("")
            continue
        if freq[l] >= 4 and len(l) < 90:  # a running header or footer
            continue
        if re.fullmatch(r"[\divxlc]+|Page \d+( of \d+)?|\d+ \| P a g e", l):
            continue
        out.append(l)
    t = "\n".join(out)
    t = re.split(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", t)[0]
    t = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", t)  # line-break hyphenation
    t = re.sub(r"(?<![.\n])\n(?!\n)", " ", t)  # unwrap lines inside a paragraph
    return t


def paragraphs(t: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", t)
        if len(p.split()) >= 8
    ]


def sentences(p: str) -> list[str]:
    p = re.sub(ABBREV, lambda m: m.group(0).replace(".", "\u2024"), p)
    ss = re.split(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z\"(\[$])", p)
    return [s.replace("\u2024", ".").strip() for s in ss if len(s.split()) >= 3]


WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def words(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


# ------------------------------------------------------------------ measures -------------
FUNCTION_WORDS = (
    "the of and to in a is that for on as with by be are this it or an from at which "
    "not its would be can than when where if each under between into their these those "
    "may will does do has have had was were been being only also so because such any all "
    "no more most other some there then while within without through over about after"
).split()

NOMINAL = re.compile(r"\b\w+(tion|sions?|ments?|ness|ities|ity|ances?|ences?)\b", re.I)
PARTICIPIAL = re.compile(r"(?:^|, )(?:[A-Z]?[a-z]+ )?\b\w+ing\b(?= [a-z])")
CONNECTIVES = {
    "However,": r"^However,",
    "Therefore,": r"^(Therefore|Thus|Hence|Consequently),",
    "Moreover,": r"^(Moreover|Furthermore|Additionally|In addition),",
    "Note that": r"^Note that",
    "summary closer": r"^(In summary|In short|Overall|Ultimately|In sum|To summarize)\b",
}


def measures(
    text: str,
) -> tuple[dict[str, float], dict[str, float], list[str], list[str]] | None:
    paras = paragraphs(text)
    sents = [s for p in paras for s in sentences(p)]
    if not sents:
        return None
    lens = [len(words(s)) for s in sents]
    allw = [w for s in sents for w in words(s)]
    n_w = len(allw) or 1
    # moving-average type-token ratio, window 200, length independent
    win = 200
    if len(allw) >= win:
        mattr = statistics.mean(
            len(set(allw[i : i + win])) / win for i in range(0, len(allw) - win + 1, 50)
        )
    else:
        mattr = len(set(allw)) / n_w
    fw = Counter(w for w in allw if w in FUNCTION_WORDS)
    m = {
        "words": n_w,
        "sentences": len(sents),
        "sentence length, mean": statistics.mean(lens),
        "sentence length, sd": statistics.pstdev(lens) if len(lens) > 1 else 0.0,
        "sentence length, coefficient of variation": (
            statistics.pstdev(lens) / statistics.mean(lens)
        )
        if len(lens) > 1
        else 0.0,
        "share of sentences under 10 words": sum(l < 10 for l in lens) / len(lens),
        "share of sentences over 40 words": sum(l > 40 for l in lens) / len(lens),
        "sentences per paragraph, mean": len(sents) / max(len(paras), 1),
        "lexical diversity (moving type-token ratio)": mattr,
        "nominalizations per 100 words": 100
        * len(NOMINAL.findall(" ".join(sents)))
        / n_w,
        "participial clauses per 100 sentences": 100
        * sum(bool(PARTICIPIAL.search(s)) for s in sents)
        / len(sents),
        "em or en dashes per 1000 words": 1000
        * sum(s.count("\u2014") + s.count("\u2013") for s in sents)
        / n_w,
        "semicolons per 1000 words": 1000 * sum(s.count(";") for s in sents) / n_w,
        "colons per 1000 words": 1000 * sum(s.count(":") for s in sents) / n_w,
        "parentheses per 1000 words": 1000 * sum(s.count("(") for s in sents) / n_w,
        "first person plural per 1000 words": 1000
        * sum(1 for w in allw if w in ("we", "our", "us"))
        / n_w,
        "passive constructions per 100 sentences": 100
        * sum(
            bool(
                re.search(
                    r"\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?\w+(ed|en)\b", s
                )
            )
            for s in sents
        )
        / len(sents),
    }
    for name, pat in CONNECTIVES.items():
        m["sentence-initial %s per 100 sentences" % name] = (
            100 * sum(bool(re.match(pat, s)) for s in sents) / len(sents)
        )
    fwv = {w: 1000 * fw[w] / n_w for w in FUNCTION_WORDS}
    return m, fwv, sents, paras


# ------------------------------------------------------------------ the tells ------------
TELLS = [
    (
        "negative parallelism (not only ... but; not X, but Y)",
        r"\bnot (only|just|merely|simply)\b.*\bbut\b|\bnot\b[^,.;]{1,40}, but\b",
    ),
    (
        "the X-and-not-Y contrast",
        r"\b(and|but) (does|do|is|are|did|was|were) not\b|\brather than\b.*\brather than\b",
    ),
    ("flourish contrast (X, not Y)", r"[a-z], not [a-z][^,.;]{1,30}[.;]"),
    (
        "rule of three (three coordinated single words)",
        r"\b\w+, \w+,? and \w+\b(?! \w+ \w+ \w+ \w+)",
    ),
    (
        "announcing colon",
        r"\b(is|states|holds|follows|means|remains|are)[^.:]{0,40}: [a-z]",
    ),
    (
        "metaphor for the network",
        r"\b(backbone|spine|arter(y|ies)|highway|skeleton|lifeblood|tapestry|landscape)\b",
    ),
    (
        "hedged identity (reads as, amounts to, is essentially)",
        r"\b(reads as|amounts to|is essentially|is really just|boils down to|at its core|in essence)\b",
    ),
    (
        "agency verbs on facilities",
        r"\b(line|path|constraint|network|grid|facility|interface)s? (wants?|sees?|prefers?|attracts?|carr(y|ies) flow|feeds? into|hang(s|ing) off)\b",
    ),
    (
        "intensifiers and evaluation",
        r"\b(crucial(ly)?|importantly|notabl[ey]|significantly|it is worth noting|worth noting|powerful|elegant|striking(ly)?|genuine(ly)?|pivotal|robust|seamless(ly)?|nuanced|holistic|comprehensive)\b",
    ),
    (
        "machine lexicon (delve, underscore, leverage, foster ...)",
        r"\b(delve[sd]?|underscor(e|es|ed|ing)|leverag(e|es|ed|ing)|foster(s|ed|ing)?|navigat(e|es|ing)|testament|intricate|multifaceted|realm|serves as|stands as|in today's|ever-evolving|a key (aspect|role|part))\b",
    ),
    (
        "vague attribution",
        r"\b(experts|observers|industry reports|many believe|it is widely|some argue|critics)\b",
    ),
    (
        "meta-commentary on the document",
        r"\b(in this (section|document|note|paper)|as (we|was) (saw|discussed|noted) (above|earlier)|the (goal|purpose) of this (section|document)|that sentence|let us now|we now turn)\b",
    ),
    ("reader address", r"\b(you|your|notice that|look at|watch what|imagine)\b"),
    (
        "expository survey voice (X lists, X states, X adds)",
        r"(?<!the )(?<!The )\b[A-Z][a-z]+ (lists|states|adds|notes|observes|argues|writes|points out|reminds us) that\b",
    ),
    (
        "sign-posting counts",
        r"\b(Two|Three|Four|Several) (things|points|reasons|answers|observations) (follow|are|stand out)\b",
    ),
    ("sentence-initial And or But", r"^(And|But) "),
    (
        "generic transmission owner",
        r"\btransmission owners?\b(?<!Participating Transmission Owner)",
    ),
    ("incentivize", r"\bincentivi[sz]e[sd]?\b"),
    ("scare quotes", r"(?<![\w.])\"[a-z][^\"]{1,25}\"(?![\w.,])"),
    # The tells below are adapted from the pattern lists of blader/humanizer SKILL.md v3.1.0 (https://github.com/blader/humanizer, commit 0ef9fe5bc3572734fd3ae6c448fe00a30cd5dcf0, 2026-10-28), MIT License, Copyright (c) 2025 Siqi Chen, itself based on Wikipedia:Signs_of_AI_writing. Used as diagnostics only; humanizer is not adopted as a skill. The licence text is in LICENSE-humanizer.txt beside this script.
    # Each name differs from every name above. The rate of em and en dashes is measured above and enforced by prose_lint, so no dash tell is listed here.
    (
        "flipped or split negative contrast",
        r"^(This|That|It|These|They) (does|do|did) not (mean|imply|suggest|say)\b|\b(it|this|that)(?: is|['’]s| was) not (?:just |only |merely |simply )?[^,.;]{1,60}[,;] (?:it|this|that)(?: is|['’]s| was)\b|, no [a-z]+( [a-z]+)?\.$",
    ),
    (
        "negation corrected in the same sentence (any subject)",
        r"\b(?:is|are|was|were)(?:n['’]t| not) (?:just |only |merely |simply |about |really )?[^,.;:]{1,60}[,;] (?:it|this|that|they|the \w+)(?: is|['’]s| are|['’]re| was| were)\b",
    ),
    (
        "preemptive negation",
        r"\b(?:and|but) none of (?:them|these|those)\b|\bI (?:have not|haven['’]t|did not|didn['’]t) (?:yet )?(?:edit|change|touch|run|commit|push)\w*|\bnothing (?:else )?(?:has been|was|is|is being) (?:run|changed|edited|committed|pushed)\b",
    ),
    (
        "not X but Y, without the comma",
        r"\b(?:is|are|was|were)(?:n['’]t| not)\b(?! only| just| merely| simply)[^,.;:]{1,40}\bbut (?:rather |instead )?(?:a|an|the|about|in|on|for|to)\b",
    ),
    (
        "restating closer",
        r"\b(that is the real|that distinction matters|read that again|let that sink in|the message was clear|it was a lesson in|(this|that) (shows|illustrates|demonstrates|underscores) the importance of|that is the (point|key|takeaway)|the (lesson|takeaway) is)\b",
    ),
    (
        "saying that sounds deep",
        r"\b(the real (question|issue|problem)|what really matters|at the heart of|the heart of the matter|the deeper (issue|point|problem)|fundamentally|in reality|the (language|currency|architecture) of|not a tool but|becomes a trap)\b",
    ),
    (
        "staged run-up",
        r"\b(let['’]?s (dive|explore|break|be honest|look|take a look)|here['’]?s (what|the thing|why|how)|without further ado|the thing is|real talk|heads up|quick note|now let['’]?s)\b|^(Honestly|Look|Frankly),",
    ),
    (
        "arguing with no one",
        r"\b(i['’]?m not saying|to be clear|don['’]?t get me wrong|this is not to say|some might say|a tempting approach|one might be tempted|an obvious approach would|you might think|it would be easy to|this isn['’]?t (mainly |really |just )?about|is not (mainly|primarily) about)\b",
    ),
    (
        "stacked qualifiers",
        r"\b(could potentially|might arguably|may possibly|could possibly|might possibly|it is also possible|to be fair|arguably|this is an inference)\b",
    ),
    (
        "machine lexicon, second list",
        r"\b(bolster(s|ed|ing)?|deep dive|garner(s|ed|ing)?|interplay|meticulous(ly)?|quietly|showcas(e|es|ed|ing)|vibrant)\b",
    ),
    (
        "inflated significance",
        r"\b(plays? an? (key|crucial|pivotal|vital|important) role|underscor\w+ (its|the|their) importance|reflects? a broader|(enduring|lasting) legacy|setting the stage for|evolving landscape|indelible mark|the future looks bright|exciting times|a step in the right direction|despite these challenges|continues to thrive|marks? a (pivotal|key|significant|turning))\b",
    ),
    (
        "-ing rider",
        r", (highlighting|underscoring|emphasizing|ensuring|reflecting|symbolizing|contributing to|cultivating|fostering|encompassing|showcasing)\b",
    ),
    (
        "sales language",
        r"\b(rich (cultural|history|heritage|tapestry)|profound|exemplifies|commitment to|natural beauty|nestled|in the heart of|groundbreaking|renowned|diverse array|breathtaking|must-visit|stunning)\b",
    ),
    (
        "borrowed authority",
        r"\b(experts (argue|believe|say|agree)|observers have|industry reports|several publications|featured in|profiled in|trade publications|independent coverage|active social media)\b",
    ),
    (
        "copula avoidance",
        r"\b(functions as|operates as|represents an?|boasts|maintains an?|refers to)\b",
    ),
    (
        "chatbot residue",
        r"\b(i hope this helps|of course!|certainly!|great question|you['’]?re absolutely right|let me know|would you like|want me to|should i continue)\b",
    ),
    (
        "knowledge-limit disclaimer",
        r"\b(up to my (last|knowledge)|while specific details|based on available (information|sources)|not publicly available|not widely (documented|disclosed)|in the (provided|available) sources|maintains a low profile|it is believed that|likely (grew|studied|began))\b",
    ),
    (
        "document about itself, second list",
        r"\b(the (table|figure|section|list|matrix) (below|above|that follows) (compares|shows|lists|summari[sz]es)|this section is organi[sz]ed|(was|were) added to replace|generated from|compiled from|flagged rather than|the following (table|figure|list))\b",
    ),
]

# Tells that a voice uses on purpose, as (voice, tell name) pairs. A hit is dropped from the
# report for that voice, the way the instruction-file exemption in report() drops "reader address".
VOICE_EXEMPT = {
    (
        "msc",
        "stacked qualifiers",
    ),  # the Committee stacks hedges before a flat recommendation (the MSC register)
    ("msc", "arguing with no one"),  # "To be clear" occurs in the MSC corpus
}
# Tells reported for review but not counted, as (voice, tell name) pairs.
VOICE_REVIEW_ONLY = {
    ("mentor", "-ing rider"),  # the exemplar primer contains one ", ensuring"
}


def voice_filter(
    voice: str, tells: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split the tell hits into (counted, review_only, exempted) for a voice."""
    counted, review, exempt = [], [], []
    for name, s in tells:
        if (voice, name) in VOICE_EXEMPT:
            exempt.append((name, s))
        elif (voice, name) in VOICE_REVIEW_ONLY:
            review.append((name, s))
        else:
            counted.append((name, s))
    return counted, review, exempt


QUOTED = re.compile(r'"[^"]{12,}("|$)')


def find_tells(sents: list[str]) -> list[tuple[str, str]]:
    """Scan the draft's own words: quoted source text is removed first, except for the scare-quote check."""
    hits = []
    for name, pat in TELLS:
        rx = re.compile(
            pat,
            re.I
            if not pat.startswith("^(And|But)") and "Participating" not in pat
            else 0,
        )
        for s in sents:
            probe = s if name == "scare quotes" else QUOTED.sub(" ", s)
            if rx.search(probe):
                hits.append((name, s))
    return hits


#: The two-sentence form of the empty contrast, "It isn't X. It's Y.": a short copular negation
#: whose next sentence restates its subject affirmatively.
PAIR_TELL = "negation then correction across two sentences"
_PAIR_NEG = re.compile(
    r"^(?:It|This|That|They|These|Those|The \w+)(?: is|['’]s| are|['’]re| was| were)?"
    r"(?: not|n['’]t)\b|^(?:It|This|That|They|These|Those|The \w+) (?:isn['’]t|aren['’]t|wasn['’]t)\b"
)
_PAIR_FIX = re.compile(r"^(?:It|This|That|They|These|Those)(?: is|['’]s| are|['’]re| was| were)\b")


def pair_tells(sents: list[str]) -> list[tuple[str, str]]:
    """The empty contrast split over two adjacent sentences, each pair returned as one hit."""
    hits = []
    for a, b in zip(sents, sents[1:]):
        if len(a.split()) <= 14 and len(b.split()) <= 20 and _PAIR_NEG.search(a) and _PAIR_FIX.search(b):
            hits.append((PAIR_TELL, f"{a} {b}"))
    return hits


def paragraph_openers(paras: list[str]) -> list[str]:
    """First sentences under nine words, the aphoristic-opener shape."""
    out = []
    for p in paras:
        ss = sentences(p)
        if ss and len(words(ss[0])) < 9 and not ss[0].endswith("?"):
            out.append(ss[0])
    return out


# ------------------------------------------------------------------ corpus stats ---------
def notebook_markdown(path: Path) -> str:
    """The prose of a notebook's markdown cells, with display math and tables removed."""
    nb = json.load(io.open(path, encoding="utf8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    t = "\n\n".join(cells)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", "X", t)
    return strip_markdown(t)


def rebuild() -> None:
    srcs = json.load(io.open(SOURCES, encoding="utf8"))
    stats = {}
    for voice, paths in srcs.items():
        chunks = []
        fwsum: Counter[str] = Counter()
        nw: float = 0
        for rel in paths:
            if rel.endswith(".ipynb"):
                t = notebook_markdown(ROOT / rel)
            else:
                import pymupdf  # noqa: PLC0415  only a PDF corpus needs it

                doc = pymupdf.open(str(ROOT / rel))
                t = clean_corpus(
                    "\n".join(
                        "===== PDF PAGE %d =====\n%s" % (i + 1, p.get_text())
                        for i, p in enumerate(doc)
                    )
                )
            paras = paragraphs(t)
            win = {"mentor": 200, "notebook": 120}.get(voice, WINDOW)
            buf, n = [], 0
            for p in paras:
                buf.append(p)
                n += len(p.split())
                if n >= win:
                    r = measures("\n\n".join(buf))
                    if r:
                        chunks.append(r[0])
                        for w, v in r[1].items():
                            fwsum[w] += v * r[0]["words"]  # type: ignore[assignment]  # float added to an int-valued Counter
                        nw += r[0]["words"]
                    buf, n = [], 0
        keys = [k for k in chunks[0] if k not in ("words", "sentences")]
        dist = {}
        for k in keys:
            vals = sorted(c[k] for c in chunks)
            q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]
            dist[k] = {"p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "n": len(vals)}
        stats[voice] = {
            "windows": len(chunks),
            "measures": dist,
            "function_words": {w: fwsum[w] / max(nw, 1) for w in FUNCTION_WORDS},
            "sources": paths,
        }
        print("%-7s %3d windows from %d document(s)" % (voice, len(chunks), len(paths)))
    with STATS.open("w", encoding="utf8", newline="\n") as fh:
        json.dump(stats, fh, indent=1)
    print("wrote", STATS)


# ------------------------------------------------------------------ report ---------------
def delta(fwv: dict[str, float], ref: dict[str, float]) -> float:
    """Burrows-style distance: mean absolute difference of per-mille function-word rates, scaled by the reference rate."""
    d = [abs(fwv[w] - ref[w]) / (ref[w] + 0.5) for w in FUNCTION_WORDS]
    return statistics.mean(d)


def report(path: Path, voice: str, as_json: bool = False) -> int:
    text, bold = load_text(path)
    r = measures(text)
    if r is None:
        print("no prose found in", path)
        return 1
    m, fwv, sents, paras = r
    stats = json.load(io.open(STATS, encoding="utf8"))
    if voice not in stats:
        print("no corpus statistics for voice %r; run --rebuild" % voice)
        return 1
    S = stats[voice]
    rows, flags = [], 0
    for k, v in m.items():
        if k in ("words", "sentences"):
            continue
        d = S["measures"].get(k)
        if not d:
            continue
        out = "" if d["p10"] <= v <= d["p90"] else ("LOW" if v < d["p10"] else "HIGH")
        flags += bool(out)
        rows.append((k, v, d["p50"], d["p10"], d["p90"], out))
    fw_delta = delta(fwv, S["function_words"])
    tells, voice_review, _exempt = voice_filter(voice, find_tells(sents) + pair_tells(sents))
    openers = paragraph_openers(paras)
    if as_json:
        print(
            json.dumps(
                {
                    "file": str(path),
                    "voice": voice,
                    "words": m["words"],
                    "sentences": m["sentences"],
                    "measures": rows,
                    "function_word_delta": fw_delta,
                    "bold_marks": bold,
                    "tells": tells,
                    "review_only": voice_review,
                    "short_openers": openers,
                },
                indent=1,
            )
        )
        return 0
    print(
        "STYLE CHECK  %s  against the %s corpus (%d windows)"
        % (path.name, voice, S["windows"])
    )
    print(
        "%d words, %d sentences, %d paragraphs. Bands are the corpus's 10th to 90th percentile; a flag means outside the band, not a target to hit."
        % (m["words"], m["sentences"], len(paras))
    )
    if m["words"] < 500:
        print(
            "The draft is under 500 words, so the function-word distance and the shares are noisy; read the tells and the sentence-length rows, and weigh the rest lightly."
        )
    print()
    print(
        "%-48s %8s %8s %18s  %s"
        % ("measure", "draft", "corpus", "band p10-p90", "flag")
    )
    for k, v, p50, p10, p90, out in rows:
        print("%-48s %8.2f %8.2f %8.2f - %7.2f  %s" % (k, v, p50, p10, p90, out))
    print(
        "%-48s %8.2f %8s %18s  %s"
        % (
            "function-word distance from the corpus",
            fw_delta,
            "",
            "under 0.35 is near",
            "HIGH" if fw_delta > 0.5 else "",
        )
    )
    print("%-48s %8d" % ("bold marks in the source", bold))
    review = [(n, s) for n, s in tells if n.startswith("rule of three")]
    tells = [(n, s) for n, s in tells if not n.startswith("rule of three")]
    if ".claude" in str(path).replace("/", os.sep).split(os.sep):
        # an instruction file addresses the model directly, by Anthropic's guidance
        tells = [(n, s) for n, s in tells if n != "reader address"]
    print(
        "\nTELLS (%d), each with the sentence to edit toward the nearest exemplar:"
        % len(tells)
    )
    for name, s in tells:
        print("  [%s]\n     %s" % (name, s[:260]))
    if review:
        print(
            "\nLISTS OF THREE (%d), review for the rule-of-three flourish; a list of three real items stands:"
            % len(review)
        )
        for name, s in review:
            print("     " + s[:200])
    if voice_review:
        print(
            "\nREVIEW ONLY (%d), not counted in the %s voice; keep each only if the exemplar documents use the construction:"
            % (len(voice_review), voice)
        )
        for name, s in voice_review:
            print("  [%s]\n     %s" % (name, s[:200]))
    if openers:
        print(
            "\nSHORT PARAGRAPH OPENERS (%d), check each for the aphoristic beat:"
            % len(openers)
        )
        for s in openers:
            print("     " + s)
    print(
        "\n%d distributional flag(s), %d tell(s). The revise step edits the quoted sentences and re-runs once; it does not regenerate."
        % (flags, len(tells))
    )
    return 0


def main() -> int:
    # the Windows console is cp1252; quoted sentences carry Unicode
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("draft", nargs="?", help=".tex, .md or .txt file to check")
    ap.add_argument(
        "--voice", default="notebook", choices=["notebook"]
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="recompute corpus_stats.json from the documents in corpus_sources.json; "
        "the notebook corpus is the markdown of sandbox.ipynb",
    )
    a = ap.parse_args()
    if a.rebuild:
        rebuild()
        return 0
    if not a.draft:
        ap.error("a draft file is required unless --rebuild")
    return report(Path(a.draft), a.voice, a.json)


if __name__ == "__main__":
    sys.exit(main())
