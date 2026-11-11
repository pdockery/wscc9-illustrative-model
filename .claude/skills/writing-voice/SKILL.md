---
name: writing-voice
description: "Drafts and revises every piece of prose the agent authors in this repository, in one of the named voices (teaching notebook, chat reply), from exemplar passages, then checks the draft against the voice's corpus. Invoke it EVERY time you author prose: a record entry, a skill or README paragraph, a docstring paragraph, notebook markdown cell, a sub-agent brief, or a chat reply to the user (user instruction 2026-10-19)."
disable-model-invocation: false
---

# Writing in the repository's voices

You are writing as the author of the document's declared voice, with its exemplar passages in front of you. The repository's prose matches the documents it reasons from, because the notebooks are read beside the ISO's own papers and the decks they follow. A document declares its voice by its kind (a notebook is the notebook voice), and holds that voice for the whole document (user decision 2026-09-24). A passage that cannot be placed in one of the named voices is rewritten until it can.

## The voices

| voice | for | exemplars | corpus the check reads |
|---|---|---|---|
| `notebook` | the markdown of a teaching notebook, and the records, skills and README of this repository | `exemplars/notebook.md` | the markdown cells of `sandbox.ipynb` |
| `chat` | a reply to the user in the chat | `register/notebook.md`, `register/td.md` | none; the notebook and TD register with Unicode math (section below) |

Each exemplar file opens with an index of passages by document part (an opening, a definition, a result reading, a close). `register/<voice>.md` is a short characterization of that voice's exemplars, read with them and never instead of them. `register/td.md` and `exemplars/td.md` carry the technical description's register, which the notebook voice applies and the chat voice reads. `register/cut_on_sight.md` holds the cut-on-sight list and the sentence mechanics, which apply in every voice.

The EDAM Draft Technical Description became the named model for notebook markdown on 2026-08-20 (user decision).

## Chat replies (user instruction 2026-10-19)

A reply to the user in the chat is authored prose and takes this skill like any other document. It is written in the register of the notebook and TD voices: declarative sentences, one claim each, the subject next to its verb, the number stated and then the stop, no intensifiers, no em-dashes, no reader-address, no meta-commentary, and the cut-on-sight list of `register/cut_on_sight.md` applied in full. The one modification is the math: a chat reply writes math in readable Unicode characters (λ̂_a, μ_m, SF_{n,m}, Σ_a N_a^c + T_E = R + R_T, θ₁ − θ₂, ≤, ≥, ×), never `$…$` or other LaTeX source, because the chat does not render it. The exemplar files are not reread for every reply; `register/notebook.md` and `register/td.md` carry the register, and `style_check.py` is not run on a chat reply. Notebook markdown and record entries keep their own voice and their LaTeX math.

## The workflow

1. **Name the voice** from the document's kind, and the document part you are about to write.
2. **Read three to five exemplars for that part**, from the voice's exemplar file, varying the topic so that you copy the voice and not the subject. Read `register/<voice>.md`.
3. **Draft as a continuation.** Where the document exists, the text before the insertion point comes last in your reading, so the draft continues it. Write the passage whole. Let paragraphs run to their argument's length. State the plain sentence under any abstraction. A person is named in prose only where the claim is that person's judgment.
4. **Run the check**: `python .claude/skills/writing-voice/scripts/style_check.py <file> --voice notebook`, then `python .claude/skills/writing-voice/scripts/prose_lint.py <file>` for the hard rules. The style check's report places each distributional measure of the draft inside or outside the corpus's 10th to 90th percentile band, and quotes every sentence that carries a catalogued tell. For notebook markdown, write the cells to a `.md` file first.
5. **Revise by editing.** For each quoted sentence, find the exemplar that handles the same document part and edit the sentence toward it. Make the fewest edits that clear the flag. A distributional flag on its own is information about the draft, and the sentences the tells quote are where the edits go. Re-run once. Editing a draft leaves less of a machine footprint than regenerating it, so a section is not regenerated to clear a flag.
6. **Stop after the second run.** Report the remaining flags with the draft, so the author decides. Gains from a revise pass come early, and a third pass trades the argument for the statistics.

## Hard constraints, which hold in every voice

These are the conventions the record depends on, so they are rules rather than style.

- **A symbol in prose follows the plain-language name of its quantity, as an appositive in commas** ("the tariff, $k$"), and a relation in prose is read as a sentence with each side named ("the value of the right, $v_X$, exceeds the tariff, $k$"), never written as bare math ("$v_X>k$") (user instruction 2026-10-31). In a notebook, every numbered equation is followed by an `*In words.*` blurb, the rule stated below.
- **A number carries its provenance**: a source page, or a notebook and cell, because a number without provenance cannot be checked and the series exists to be checked.
- **Source wording is quoted verbatim, with its pinpoint**, because the record's authority is the source's own words (user instruction 2026-10-01). The page-pinned passages are in `sources/excerpts/<key>.md`.
- **A person's work is introduced at its first prose mention** (user decision 2026-09-03). A bare author-date shorthand ("Mays (2024)") does not appear at first mention. The person and the context are named, and the surname alone follows. A parenthetical citation names the document at first use. The worked before and after is `notes/writing_profile/source_register_profile.md` section 6.
- **The canon's names, index order and signs** (`AGENTS.md`, the notation section; `CANON.md`, the annotated tables) are used as the canon states them, because two documents that name one object differently are read as stating two objects.
- **In a notebook**, the formulation opens with the nomenclature table `| Symbol | Code | Definition |`, the program is stated whole with each constraint family's dual in brackets, and every numbered display equation is followed by an `*In words.*` blurb that reads the equation left to right, naming each quantity by its table name with the symbol inline. These are the series' teaching devices, and the blurb is the only labelled construct a notebook uses. NB-03, NB-04 and NB-05 in `exemplars/notebook.md` show the three.

## What the check is

Prose is drafted from the exemplar passages and checked after drafting against the voice's own corpus (user decision 2026-10-12). The check is a diagnostic read after drafting. It never sets a target. Its bands are the voice's own documents, measured in 600-word windows (`corpus_stats.json`), and its tells are the register profile's cut-on-sight list, the editors' lists of machine-writing patterns, and the stylometry literature's discriminating features. Prose drafted to hit the measures comes out with a performed rhythm (the texture finding of 2026-10-05), which is why the measures live here and not in the drafting instructions. The measured profile, `notes/writing_profile/source_register_profile.md`, is the provenance of the tells and the exemplar selection, and is read for that purpose.

## Files

- `exemplars/notebook.md`, `td.md`: verbatim passages with page or cell pinpoints, indexed by document part.
- `register/notebook.md`, `td.md`: one short characterization per voice; `register/cut_on_sight.md`: the cut-on-sight list and the sentence mechanics for every voice.
- `scripts/style_check.py`: the diagnostic; standard library only; `--json` for a machine-readable report.
- `scripts/prose_lint.py`: the hard prose and notebook rules, run by the PostToolUse hook on every write and by hand in step 4; `--list-rules` names the rules it checks. `scripts/symbol_check.py` is the symbol module `prose_lint.py` loads. `style_check.py` stays a diagnostic that never sets a target.
- `tests/test_prose_lint.py`: the lint's tests, with `scripts/prose_lint_golden.json` as the frozen per-rule counts.
- `corpus_sources.json`, `corpus_stats.json`: the document the notebook bands are computed from, and the bands. The source is the markdown of `sandbox.ipynb`, and `style_check.py --rebuild` recomputes the bands from it. The exemplar passages stay verbatim from the nodal market fundamentals notebook of commit ce66306.
