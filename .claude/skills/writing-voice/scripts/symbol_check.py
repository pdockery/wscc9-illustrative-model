r"""Check a LaTeX file against the symbol-naming standard (user instruction 2026-10-31).

The standard, stated as three rules the script can test:

  R1  Name first, then the symbol.  An inline symbol in prose follows the plain-language name of its
      quantity as an appositive set off by commas: "the tariff, $k$".  A symbol may appear bare later
      in the SAME sentence (source line) once it has been introduced that way, and a symbol that is
      the whole subject of a definition sentence in a symbol table is exempt.
  R2  A relation is read, not written.  An inline math span in prose that carries =, <, >, ge, le,
      \gtrless or \ne is written as a sentence with each side named ("the value of the right, $v_X$,
      exceeds the tariff, $k$"), never as "$v_X>k$" or "($x=1$)".
  R3  Every numbered equation is read.  Each \begin{equation} ... \label{L} ... \end{equation} is followed
      by a \begin{inwords} block whose text cites \eqref{L} and opens "Equation~\eqref{L} states that".
      A labelled align, align* or equation* environment is held to the same test; an unlabelled one is
      not checked.  A file that declares "% readings: inline" (the rules-and-games worksheets, user
      2026-11-03) reads its equations in the running text: there an equation passes R3 when its
      lead-in reading, a prose line, sits directly before \begin, with no In-words block required.

What the script skips: comment lines, the preamble (before \begin{document}), display math bodies,
tabular/longtable rows (lines containing " & " or \toprule etc.), tikz/figure code (lines made of the
matrix macros \o... , \node, \draw, \fill), \label/\ref/\cite-only lines, and the first column of a
symbol table.  Captions and \item lines are prose and are checked.

Usage:  python symbol_check.py FILE [FILE ...] [--json]
Exit code is 0; the report is the product.  Every flag is a candidate, read by a person.
"""

from __future__ import annotations
import re, sys, json
from pathlib import Path

MATH = re.compile(r"\$([^$]+)\$")
RELATION = re.compile(
    r"(?<![\\a-zA-Z])(=|<|>|\\ge\b|\\le\b|\\geq\b|\\leq\b|\\gtrless\b|\\ne\b|\\neq\b)"
)
SKIP_LINE = re.compile(
    r"^\s*(%|\\(begin|end)\{|\\(label|ref|eqref|cite[tp]?|caption\{\\label)|\\(o[a-z]+|node|draw|fill|path|addcontentsline|phantomsection|setcounter|newcommand|input|section|subsection|paragraph|subparagraph|centering|toprule|midrule|bottomrule|endhead|endfirsthead|hline|addlinespace|vspace|noindent|item\s*$))"
)
TABLE_ROW = re.compile(r"(^|[^\\])&")
DISPLAY_OPEN = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|eqnarray\*?)\}|\\\["
)
DISPLAY_CLOSE = re.compile(
    r"\\end\{(equation\*?|align\*?|gather\*?|eqnarray\*?)\}|\\\]"
)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


NOT_PROSE = re.compile(r"^\s*(%|\\(begin|end|label)\b|\\(textit|textbf)\{[^}]*\}\s*$)")


def read_inline(text: str, start: int) -> bool:
    """Return True when a prose line, the equation's lead-in reading, sits directly before the display."""
    before = text[:start].split("\n")[-2] if text[:start].endswith("\n") else ""
    return bool(before.strip()) and not NOT_PROSE.match(before)


def check_file(path: str) -> list[tuple[int, str, str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    flags = []
    in_doc = "\\begin{document}" not in text  # a part file has no preamble
    in_display = False
    inline = re.search(r"^%\s*readings:\s*inline", text, re.M) is not None
    # R3: equations and their In-words blocks
    for m in re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S):
        labels = re.findall(r"\\label\{([^}]*)\}", m.group(1))
        if (
            inline
            and labels
            and not re.match(r"\s*\\begin\{inwords\}", text[m.end() :])
        ):
            if not read_inline(text, m.start()):
                lineno = text[: m.start()].count("\n") + 1
                flags.append(
                    (
                        lineno,
                        "R3",
                        f"equation {','.join(labels)} has no lead-in reading before it",
                        "",
                    )
                )
            continue
        after = text[m.end() : m.end() + 1500]
        iw = re.search(r"\\begin\{inwords\}(.*?)\\end\{inwords\}", after, re.S)
        lineno = text[: m.start()].count("\n") + 1
        if not labels:
            flags.append((lineno, "R3", "equation without a label", ""))
            continue
        # a run of equations may share one In-words block that cites each of them
        between = (
            re.sub(
                r"\\begin\{(equation|align)\*?\}.*?\\end\{(equation|align)\*?\}",
                "",
                after[: iw.start()],
                flags=re.S,
            ).strip()
            if iw
            else "x"
        )
        if not iw or between:
            flags.append(
                (
                    lineno,
                    "R3",
                    "equation %s is not followed directly by an In-words block"
                    % ",".join(labels),
                    "",
                )
            )
            continue
        for lab in labels:
            if "\\eqref{%s}" % lab not in iw.group(1):
                flags.append(
                    (
                        lineno,
                        "R3",
                        "the In-words block does not cite \\eqref{%s}" % lab,
                        "",
                    )
                )
        if not re.search(r"Equation~\\eqref\{[^}]*\}\s+states that", iw.group(1)):
            flags.append(
                (
                    lineno,
                    "R3",
                    "the In-words block does not open 'Equation~\\eqref{...} states that'",
                    "",
                )
            )
    # R3 for labelled align, align* and equation* (2026-11-01). In both loops a run of equation and align
    # displays may share one In-words block that cites each of them.
    for m in re.finditer(
        r"\\begin\{(align\*?|equation\*)\}(.*?)\\end\{\1\}", text, re.S
    ):
        labels = re.findall(r"\\label\{([^}]*)\}", m.group(2))
        if not labels:
            continue
        env = m.group(1)
        if inline and not re.match(r"\s*\\begin\{inwords\}", text[m.end() :]):
            if not read_inline(text, m.start()):
                lineno = text[: m.start()].count("\n") + 1
                flags.append(
                    (
                        lineno,
                        "R3",
                        f"{env} {','.join(labels)} has no lead-in reading before it",
                        "",
                    )
                )
            continue
        after = text[m.end() : m.end() + 1500]
        iw = re.search(r"\\begin\{inwords\}(.*?)\\end\{inwords\}", after, re.S)
        lineno = text[: m.start()].count("\n") + 1
        between = (
            re.sub(
                r"\\begin\{(equation|align)\*?\}.*?\\end\{(equation|align)\*?\}",
                "",
                after[: iw.start()],
                flags=re.S,
            ).strip()
            if iw
            else "x"
        )
        if not iw or between:
            flags.append(
                (
                    lineno,
                    "R3",
                    "%s %s is not followed directly by an In-words block"
                    % (env, ",".join(labels)),
                    "",
                )
            )
            continue
        for lab in labels:
            if "\\eqref{%s}" % lab not in iw.group(1):
                flags.append(
                    (
                        lineno,
                        "R3",
                        "the In-words block does not cite \\eqref{%s}" % lab,
                        "",
                    )
                )
        if not re.search(r"Equation~\\eqref\{[^}]*\}\s+states that", iw.group(1)):
            flags.append(
                (
                    lineno,
                    "R3",
                    "the In-words block does not open 'Equation~\\eqref{...} states that'",
                    "",
                )
            )
    # R1, R2 line by line
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not in_doc:
            if "\\begin{document}" in line:
                in_doc = True
            continue
        if in_display:
            if DISPLAY_CLOSE.search(line):
                in_display = False
            continue
        if DISPLAY_OPEN.search(line):
            if not DISPLAY_CLOSE.search(line):
                in_display = True
            continue
        if not line or SKIP_LINE.match(line):
            continue
        if TABLE_ROW.search(line):
            continue
        spans = list(MATH.finditer(line))
        if not spans:
            continue
        introduced = set()
        prev_ok = False
        for sp in spans:
            body = sp.group(1).strip()
            key = norm(body)
            if re.fullmatch(r"[-+]?[0-9.,\s]+|\\%|[-+]?[0-9]+\s*\\%", body):
                continue  # a number
            if re.fullmatch(r"[<>=]|\\(ge|le|geq|leq)", body):
                prev_ok = False
                continue  # a named relation sign ("residual $>$ return", "marked $=$")
            before = line[: sp.start()]
            appositive = before.endswith(", ") or before.endswith(",")
            # an index letter after its noun: "in hour $t$", "indexed by $j$", "cell $(r,c)$"
            if re.search(
                r"(hour|hours|period|periods|index|indexed by|cell|row|column|design|case|worksheet|letter)\s$",
                before,
                re.I,
            ):
                introduced.add(key)
                prev_ok = True
                continue
            # a second symbol in a pair that continues an appositive: "$b_{11}$ or $b_{21}$", "$a_{11}$ over $a_{21}$"
            continuation = prev_ok and re.search(
                r"\$\s+(or|and|over|to|against)\s$", before
            )
            if RELATION.search(body):
                flags.append(
                    (
                        i,
                        "R2",
                        "relation written as math in prose: $%s$" % body,
                        line[:160],
                    )
                )
                prev_ok = False
                continue
            if appositive or continuation:
                introduced.add(key)
                prev_ok = True
                continue
            if key in introduced:
                prev_ok = True
                continue
            flags.append(
                (i, "R1", "symbol not introduced by its name: $%s$" % body, line[:160])
            )
            prev_ok = False
    return flags


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    out = {}
    total = 0
    for f in args:
        fl = check_file(f)
        out[f] = [dict(line=l, rule=r, problem=p, text=t) for l, r, p, t in fl]
        total += len(fl)
        if not as_json:
            print("== %s: %d flag(s)" % (f, len(fl)))
            for l, r, p, t in fl:
                print("  %s:%d [%s] %s" % (Path(f).name, l, r, p))
                if t:
                    print("      %s" % t)
    if as_json:
        print(json.dumps(out, indent=1))
    else:
        print(
            "%d flag(s) in %d file(s). R1 name-then-symbol; R2 relation read as a sentence; R3 In-words after every numbered equation."
            % (total, len(args))
        )


if __name__ == "__main__":
    main()
