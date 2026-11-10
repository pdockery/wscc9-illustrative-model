# Source register profile

**What this is.** A writing profile of the documents this repository writes like, derived by
reading them rather than by recalling them. It exists so that any agent or author can
match the register of the Western market-design record without re-reading six PDFs.

**Built** 2026-08-26 by three agents
reading the corpus below end to end and reporting verbatim quotation, lexical counts, and
sentence statistics. This copy, of 2026-11-07, is cut to the registers this repository
uses: the technical description, which the notebook voice applies, and the reference
measurements of the ISO's other documents that the rules cite. It quotes public CAISO
documents for style matching. Nothing here is a research finding.

**Relationship to `CLAUDE.md` and the writing-voice skill.** The repository's working
agreement names the EDAM Draft Technical Description as the prose model, and the cut-on-sight
list and the sentence mechanics live in `.claude/skills/writing-voice/register/cut_on_sight.md`
(moved from `CLAUDE.md` on 2026-11-01). This file is the evidence behind that agreement and
extends it with the lexicon and the verb discipline. Where the two disagree, `CLAUDE.md` and
the skill govern; nothing here currently disagrees with them.

---

## 1. The corpus

Read in full or near-full. The technical description is tracked here, at the path given
from the repository root. The other five are public CAISO documents that the measurements
below cite, not carried here.

| Tag | Document | Register |
|---|---|---|
| **TD** | `sources/ferc-filings-and-iso-policy/caiso_edam_techdesc.pdf` (`caiso_edam_techdesc`) | Formulation. 27 pp. |
| **MSC-25** | Market Surveillance Committee, opinion on EDAM congestion revenue allocation, June 13, 2025 | Economist opinion, evidence-first. 35 pp. |
| **MSC-07** | Market Surveillance Committee, final opinion on congestion revenue rights, 2007 | Economist opinion, principle-first. 7 pp. |
| **FP** | CAISO, *EDAM Congestion Revenue Allocation*, Final Proposal, June 6, 2025 | Policy narrative. 34 pp. |
| **IP** | CAISO, EDAM congestion revenue allocation Issue Paper, March 17, 2025 | Policy narrative. 17 pp. |
| **S2** | CAISO, EDAM congestion revenue allocation Stage 2 Analysis Paper, April 14, 2026 | Quantitative analysis. 25 pp. |

---

## 2. The finding that governs everything else

**These documents contain no physical-plant metaphor, and almost no physical-plant
vocabulary at all.** Verified counts, case-insensitive, across full extracted text:

| term | TD | MSC-25 | MSC-07 | FP | IP | S2 |
|---|---|---|---|---|---|---|
| backbone | 0 | 0 | 0 | 0 | 0 | 1 |
| spine | 0 | 0 | 0 | 0 | 0 | 0 |
| corridor | 0 | 0 | 0 | 0 | 0 | 1 |
| artery / highway | 0 | 0 | 0 | 0 | 0 | 0 |
| high-voltage | 0 | 0 | 0 | — | — | — |
| kV / conductor / circuit | 0 | 0 | 0 | rare | rare | rare |

The single "backbone" is an offhand aside in S2 p. 17 ("key backbone elements like Path26
and the COI corridor"), not a defined term.

The network is named three ways, and only these three:

1. **As a constraint.** "the congested network constraint" (MSC-25 p. 1) · "constraints
   located in other balancing areas" (MSC-25 p. 7) · "binding transmission constraints" ·
   "Generalized network constraint index, including intertie scheduling limits" (TD p. 7) ·
   "internal constraints" / "external constraints".
2. **As a facility, system, or area.** "its own transmission system" (FP p. 28) ·
   "adjoining systems" · "the neighboring system" · "the interconnected transmission grid"
   (FP p. 7) · "interconnected transmission systems" · "a transmission element" ·
   "transmission facilities and their constraints in the FNM" (IP p. 15) · "balancing
   authority area" · "the market footprint".
3. **As a flow contribution.** "generation in one EDAM area may contribute flow on a
   transmission constraint in an adjacent EDAM area" (FP p. 5) · "flow contributions from
   schedules at that location in relation to the constraint" (FP p. 7).

A specific facility, when named at all, is named by its **constraint identifier**:
`30055_GATES1_500_30060_MIDWAY_500_BR_1_1`, "Gates-Midway 500kV line", "6110 COI S-N",
"Devers 230 kV Transformer".

**Operating rule.** Never reach for an image. If a sentence needs the network to be
*like* something, the sentence is wrong. Name the facility, the constraint, the area, or
the flow contribution.

---

## 3. Lexicon

Use these terms. They are the record's own.

**Flow physics.** parallel flow (the governing term, ~120 uses in FP/IP) · loop flow and
unscheduled flow (given once as synonyms, in quotation marks, then dropped) · flow
contribution · contribute flow on a constraint · effectiveness / effectiveness
contribution · shift factor · power transfer distribution factor (PTDF) · prevailing flow
vs counterflow · minimum effectiveness threshold · uncompensated flow · unpriced flow ·
simultaneously feasible.

Externality is **never called an externality**. It is "the external cost of imposing
congestion on a neighboring BAA", "cost shifts", "uncompensated flows", "flow impacts on
constraints in other balancing areas".

**Settlement.** congestion revenue · congestion rent · marginal congestion component (MCC)
· shadow price · accrue · materialize · allocate / sub-allocate · collect · bear the
effects · hedge · congestion cost exposure · revenue adequacy.

**Entities.** balancing authority area (BAA) · EDAM entity · transmission provider ·
transmission service provider · the market operator (CAISO when performing a market
function) · the ISO / the CAISO (the institution) · transmission customer · load serving
entity · scheduling coordinator · market participant.

> **Never write "transmission owner" as a generic noun.** The Western OATT idiom is
> *transmission provider* or *transmission service provider*, and the generic
> "transmission owner" appears nowhere in FP, IP, or S2.
>
> **The one exception is the CAISO tariff term of art**, capitalized:
> *Participating Transmission Owner*, the entity that places facilities under CAISO
> operational control while retaining ownership. Use it only in that exact,
> capitalized form and only when the CAISO operational-control relationship is the
> subject. The §12 checklist search will flag it; that hit is expected.

**Judgment vocabulary is small and plain** (MSC): good / bad, better / worse, simpler,
prudent, workable, durable, material, trivial, a step backward, worse than nothing.
Never elegant, powerful, striking, crucial, notable.

---

## 4. Verb discipline

The corpus keeps these strictly separate. Copying this division is the fastest way to
sound like the record.

| Subject | Verb | Example |
|---|---|---|
| flow | **contributes** | "contribute flow on a transmission constraint"; "contribute to the overload" |
| a constraint | **binds**, **materializes** | "the balancing area in which the internal transmission constraint materialized" |
| price | **impacts**, **has an effect on** | "impacts congestion prices at locations in the neighboring area" |
| revenue | **accrues**, **is collected**, **is allocated** | "the balancing area where the congestion revenues accrued" |
| an area | **bears** | "bears the effects of the constraint"; "bears the costs and actions to manage" |
| capacity | **is reserved**, **is released** | "Transmission capacity is reserved for this transfer on the associated intertie" |
| a constraint set | **is enforced**, **is relaxed**, **is ignored** | "the following constraints are enforced in the IFM" |
| a design | **supports**, **seeks to**, **recognizes that** | "This allocation method recognizes that…" |
| a schedule | **displaces**, **is dispatched** | "would necessarily displace a MW of other flows" |

Note the two that are easiest to get wrong: **flow contributes, price impacts.** Never
"flow impacts a constraint." And **a line does not carry, attract, take, or want flow** —
the construction "a line carries flow" appears nowhere in TD.

Use **incent**, not "incentivize" (FP usage).

---

## 5. Sentence mechanics

Measured, per register:

| | TD | MSC-25 | FP | IP | S2 |
|---|---|---|---|---|---|
| mean words/sentence | 22–35 | ~29 | 38.1 | 38.5 | **26.8** |
| median | — | — | 32 | 32 | **22** |
| em-dashes | **0** | 3 in 35 pp | **0** | **0** | **0** |
| spaced en-dash asides | 0 | 0 | 41 | 23 | **0** |
| semicolons | ~10 | 7 / 18,300 words | 3 | 7 | 3 |

**Rules that hold across every register.**

- **Em-dashes: zero.** Not one in the technical description, the Final Proposal, the Issue
  Paper, or the Stage 2 paper. Three in 35 pages of MSC. Treat as forbidden.
- **Semicolons: effectively absent** from body prose. Join with a comma and a coordinator,
  or start a new sentence.
- **Colons do two jobs only:** introduce a list, or introduce an equation after "as
  follows". Never a rhetorical pivot.
- **Parentheses are heavy and mechanical**, never commentary. Four jobs: acronym on first
  use; naming the symbol for the thing just named; scoping a category; a restrictive gloss.
  Definitional appositive on first use is a signature MSC move: "a shift factor
  (contribution of a schedule to flow over a line)".
- **Scare quotes mark a contested or borrowed term**, and are then dropped: "loop flow",
  "unscheduled flow", "shift-factor relationship", "use it or lose it", "fairness".
- **No first person** in TD, FP, IP, or S2. MSC uses a committee "we" throughout.
- **The reader is never addressed.** No "notice", "consider", "watch what happens",
  "as we saw above". The one permitted hypothetical form is "For example, consider a
  resource with a cost of $35/MWh…".
- **Percentages are spelled "percent"** in S2 prose, not "%".
- Repeat full names rather than pronouns wherever a pronoun could be ambiguous.

**Connectives, by register.** TD allows only: `Furthermore,` (additive) and
`Therefore,` / `Consequently,` / `thus` / `hence` (consequence). MSC's signature is
`However,` opening a paragraph, 34 times in 35 pages, marking the pivot from a favorable
finding to its residual risk. FP/IP open with `Under the [regime], …` more than anything
else, and use `To the extent …` as the conditional workhorse.

**Section and figure references are flat.** "Figure 2 shows the EDAM timeline." "Figure 10
provides a breakdown by the top transmission constraints in the California ISO area."
State the figure, give the reading, stop. FP once hedges a figure as "attempts to
illustrate", which is the most self-conscious the corpus ever gets.


**Provenance of the sentence mechanics and the In-words exemption** (moved from `CLAUDE.md`
on 2026-11-01; the rules themselves are `.claude/skills/writing-voice/register/cut_on_sight.md`).

- **The sentence mechanics** were distilled on 2026-08-21 from the user's five editorial
  passes. The user's words: "complicated subclauses with colons, hyphens, m-dashes … I've
  read and tried to state them in simpler terms without dramatization".
- **The In-words exemption.** The `*In words.*` form overrides the sentence-splitting
  mechanics inside its own blocks, a user flag of 2026-08-20 raised after splitting damaged
  the readings.

---

## 6. The house overlay: active voice, nouns close to verbs

**This modifies the sources and must be applied on top of them** (author instruction,
2026-08-26).

The ISO documents are heavily passive and agentless where allocation is concerned:
"are allocated", "is distributed", "will be sub-allocated", "congestion revenues accrue".
That is a regulatory drafting habit that avoids naming who acts. This repository does not use it.

Write the actor as the subject, and keep the subject next to its verb.

| Source habit | House form |
|---|---|
| "Flow on line 6-7 is contributed by Market B's dispatch." | "Market B's dispatch contributes flow to line 6-7." |
| "Each seam line is assigned to the market that manages it by the model." | "The model assigns each seam line to the market that manages it." |
| "Line 6-7 is enforced by Market A." | "Market A enforces line 6-7." |

Keep the passive only where the actor is genuinely the market mechanism and naming an
agent would mislead ("congestion revenue accrues to the area where the constraint binds").

**Introduce the work, not the citation** (author instruction, 2026-09-03). Notes
and meeting summaries are not academic papers, and a bare author-date reference
assumes a reader who knows the work by its author. The first prose mention of a person's
work names the person and the context the work was produced in; later mentions use the
surname alone, and a direct quote or "he says" / "she says" follows naturally once the
person is introduced. Parenthetical citations carry the document's title or docket at
first use and may shorten after. The scholarly `.tex` that carries a bibliography keeps
standard `\citet`/`\citep` practice.

| Before | After |
|---|---|
| "Kavulla's first lesson states the mechanism (Kavulla, AD25-7, pp. 1, 8)." | "Travis Kavulla, in comments to FERC's 2025 technical conference on resource adequacy (Docket AD25-7), makes the conceptual connection between a prompt showing and forward procurement. He says a relatively prompt showing of adequate capacity, subject to adverse financial outcomes or penalties for any shortfall, 'all but guarantees that those who bear the purchase obligation will purchase the commodity in advance if they possibly can' (AD25-7, pp. 1, 8)." |
| "the price-formation argument of Mays' work on sequential pricing (Mays, 2024)" | "Jacob Mays, a member of the ISO's Market Surveillance Committee, makes the price-formation argument in his working paper *Sequential Pricing of Electricity* (2024): …" |

---

## 7. Choosing a register

**The named voices (author decision, 2026-09-24).** An agent writing in this repository
uses one of the named voices of the `writing-voice` skill and holds it. This repository
carries two, the notebook voice and the chat voice. The notebook voice is the EDAM Draft
Technical Description's register (TD below) applied to the parts of a notebook, and it
covers the records, the skills and the README as well. The chat voice is the same
register with Unicode math. S2, FP and IP remain reference profiles of the ISO's own
documents, cited by the measurements above and below. A document takes its voice from its
kind, and a passage that cannot be placed in one of the named voices is rewritten until it
can.

**TD, for formulation sections and notebook markdown.** Declarative statement of what
something is, then `, as follows:` and the equation, then a paragraph opening `Where …`
that glosses it. `Note that …` is the only device for a mechanical consequence the reader
might miss. Say "for simplicity" when you simplify, and say what you dropped. State
difficulty without apology: "The implementation of this constraint is very complex."

> Notebook prose additionally carries the repository's own conventions: the nomenclature
> table, self-contained equation numbering from (1), and the `*In words.*` blurb. Those are
> in `CANON.md` and are not part of this profile.

---

## 8. Caveats and limitations, the corpus form

S2 is the model. The pattern is: **name the limitation, state its directional effect on
the estimate, state the decision it forces.** No apology, no defensiveness.

> "Congestion management in WEIM optimizes resources across a footprint larger than EDAM,
> so EDAM's congestion frequency and severity may differ. Therefore, results based on WEIM
> data should be interpreted with caution." (S2 p. 13)

> "Relying on these schedules could therefore overestimate congestion revenue allocation to
> qualifying self-scheduled transmission rights." (S2 p. 14)

MSC's form names the **specific missing dataset** and who would have to produce it. "We do
not know" is always paired with exactly what would need to be known: "We are not aware of a
comprehensive attempt to quantify the potential magnitudes of these problems, although such
a task is not unreasonably burdensome."

State scope exclusions as a flat sentence: "The initiative focuses narrowly on … and does
not seek to address …".

---

## 9. How numbers attach to claims

- Source, date, and dimension every figure. MSC footnotes the report, month, and table.
- Round deliberately and mark it: "around $11/MWh", "approximately $2.47 million per month",
  "on average", "a few cents".
- Narrate counterfactual arithmetic with `say`: "Multiplying such amounts by ~2000
  hours/quarter by, say, a flow of 1000 MW, could amount to as much as $100,000, which
  although not large could be of some concern."
- Judge the magnitude in plain words immediately after stating it, against a named
  baseline. A number with no size judgment is an incomplete sentence in this corpus.
- Illustrative examples use round pedagogical numbers and are flagged as hypothetical.

---

## 10. Cut on sight

Beyond the categories already in `.claude/skills/writing-voice/register/cut_on_sight.md` (intensifiers, reader address,
meta-commentary, imperatives), the corpus reading and the author's edits add:

- **Any metaphor for the network.** backbone, spine, artery, corridor as an image, "the
  grid's skeleton", "the highway".
- **"Reads as", "amounts to", "is essentially", "is really just".** Hedged identity claims.
  State what the thing is.
- **Casual spatial verbs.** "hanging off", "runs the length of", "terminates at", "sits on
  top of", "feeds into" — unless it is literally the defined term.
- **"Attracts", "wants", "sees", "prefers"** applied to a facility or a network. No agency.
- **The empty contrast ("X, not Y").** A contrast sentence earns its place only when
  both sides are defined in the document, the sentence adds a claim its neighbors do
  not already carry, and it is not a mirrored restatement of an adjacent sentence.
  Cut the flourish form ("complements, not substitutes" → "complementary"), the
  aphorism ("if everyone is accountable, no one is accountable"), and the closer that
  restates the preceding sentence with the terms swapped. Keep the contrast that
  carries its own testable claim ("the failure belongs to the integration and not to
  the forward program, since the same showing composed with an uncapped daily
  evaluation would leave the near-term consequence intact"). (Author edits,
  2026-09-03.)
- **The manufactured parallel ("two instances of one question").** A sentence that
  stages two subjects as sharing an underlying identity ("side by side as two
  instances of one design question", "two sides of the same coin", "the same
  instrument applied to the two commodities") instead of stating the shared property
  as a claim. If the parallel is real, the common property can be stated with its own
  truth conditions and support ("both are short-run eligibility criteria for actors
  with long-run independence"), and the staging sentence is then unnecessary.
  Promoting a source's hedged simile ("is like") to an identity ("is the same
  instrument") is the citation-facing form of the same fault. Note that `CLAUDE.md`'s
  retired framing, congestion revenue allocation and seams as two instances of one
  allocation and appropriation problem, is an instance. (Author edits, 2026-09-04.)
- **The aphoristic thesis opener ("one X with two Y").** A sentence that announces
  that the subject has a hidden structure without stating the structure ("Congestion
  rent in a nodal market is one quantity with two readings", "one problem with two
  faces"). The sentence carries no claim of its own, because its content arrives only
  in the sentences that follow it, and it usually opens a paragraph, where it costs
  the reader the most. Cut it and open with the first fact it was withholding ("When
  a transmission constraint binds, prices separate and load pays more for energy than
  generation is paid"). (Author edit, 2026-09-24.)
- **Self-labelling metadiscourse.** A sentence in which the document names, ranks, or
  glosses its own sentences rather than its subject: "That sentence is the key
  line of the formulation, and the rest of this cell is subordinate to it"; "It restates, in
  this notebook's words, the dispatch …". In Hyland's taxonomy of metadiscourse these are
  endophoric markers (pointing at another part of the text) and frame markers
  (announcing the text's structure), with an attribution gloss on top. The subject
  matter is unchanged by the label, so delete the label and let the sentence carry
  its own weight. (Author edit, 2026-09-24.)
- **The negative scope catalogue.** A scope stated as a chain of what the thing is
  not: "The sandbox is used to …. It is not a forecast, it does not rank market
  designs, and it does not model losses." State the scope affirmatively. Where a
  reader would otherwise assume the opposite, state one exclusion flatly and once
  (§8's scope-exclusion form), never a list. (Author edit, 2026-09-24.)
- **The expository survey voice.** The register of a seminar paper or literature
  review: narrated attribution in which each sentence reports who said what ("Hogan
  lists …", "Cramton states …", "Kiesling adds …", "the Committee has repeatedly held
  …"); sign-posting counts ("Two things follow", "Three answers are available");
  sentence-initial "And" or "But"; fragments staged for emphasis ("and the reason is
  physical"); causal tags hung on the end of a sentence ("which is why", "for the same
  reason"). It reads as a student showing that the reading was done. The MSC and TD
  form states the claim, attaches the citation with `\citep`, and stops; a person is
  named in prose only where the claim is that person's judgment rather than a fact
  the source establishes. (Author reaction, 2026-09-24: "sophomoric".)
- **The announcing colon.** A sentence that names the kind of thing it is about to say and
  then delivers it after a colon: "The revenue adequacy theorem states the bound: the
  congestion rent collected in the actual dispatch is always sufficient …"; "The
  Commission's statement is that …"; "Two things follow." The announcement is
  flourish. State the claim and attach the citation: "The congestion rent collected in the
  actual dispatch is sufficient to pay every simultaneously feasible right
  \citep[p.~N]{<key>}." The same holds for the appositive stack ("the
  surplus, the congestion rent, is what funds …") and for a sentence that carries three
  coordinated claims where three sentences would do. (Author reaction, 2026-09-25:
  "precision, simple writing, clarity, without flourish".)
- **Analogy as an argument.** "one system is the other rotated ninety degrees" is a claim
  that either has a measurement behind it or does not belong. In the case that prompted
  this profile, the measurement contradicted it.
- **Em-dashes.** See §5.

**Watch for the metaphor returning through a quoted source.** Research reports, utility
marketing, and trade press use "backbone" freely, and the word re-enters a draft when a
source's phrasing is carried across. Restate it instead: write "that network runs at
765 kV" rather than "its backbone is 765 kV." This happened once in the note that
prompted this profile, after the word had already been removed everywhere else, so run
the §12 search again after any research is folded in.

---

## 11. Worked before and after

Constructed on the canonical 9-bus clearing, in which line 6-7 binds at 40 MW with a signed
dual of $-\$20.10$/MWh and a rent of \$804.1/h (`NOTEBOOK_RECORD.md` section 1). The left
column is what an unprofiled model produces; the right is the corpus register with the house
overlay.

| Before | After |
|---|---|
| "Line 6-7 is the bottleneck the whole network leans on." | "Line 6-7 binds at its 40 MW rating, with a signed dual of $-\$20.10$/MWh." |
| "the binding line attracts flow from injections across the network regardless of who scheduled them" | "Each bus's injection contributes flow to line 6-7 in proportion to its shift factor, whichever market scheduled it." |
| "Congestion is a toll the cheap fleet pays to reach the load." | "The congestion rent is \$804.1/h, what load pays less what generation is paid." |

The lesson in the first row is the general one: **an image stands in for a measurement.**
Stating the rating and the dual is both more accurate and shorter than the image was.

---

## 12. Review checklist

Run against any draft before it ships.

1. Search the draft for: backbone, spine, corridor (as image), artery, highway, skeleton,
   reads as, hanging off, attracts, wants, sees, feeds into. Zero hits expected.
2. Search for `—` and `–`. Zero expected in body prose.
3. Search for `;`. Near zero expected.
4. Every number has a source, a dimension, and a size judgment against a named baseline.
5. Every sentence has one claim. If it needs a colon, semicolon, or dash to stand up,
   split it.
6. Subject is the actor, and sits next to its verb.
7. Flow contributes; price impacts; revenue accrues; a constraint binds or materializes;
   an area bears effects.
8. Entities are named in Western OATT idiom, never "transmission owner".
9. Caveats name the limitation, its directional effect, and the decision it forces.
10. Scope exclusions are stated flatly and once.
11. No claim rests on an analogy that has not been measured.
12. The first sentence of every paragraph and section states a fact. No aphoristic
    thesis opener ("X is one Y with two Z").
13. No sentence names, ranks, or glosses another sentence of the same document ("That
    sentence is …", "in this notebook's words"). No scope stated as a chain of negations.

---

## 13. Refreshing or extending this profile

The profile was produced by three parallel agents, one per document group, each asked for
a *writing* profile rather than a content summary, with heavy verbatim quotation and
lexical counts. To extend it to another document, reuse this prompt shape:

> Read [path] with the Read tool's `pages` parameter. Produce a WRITING PROFILE of the
> prose for register matching. Do NOT summarize content; focus entirely on HOW it is
> written. Report: (1) verbatim noun phrases for the physical network, and specifically
> whether "backbone" / "spine" / "corridor" appear and what is used instead; (2) 8–12
> verbatim sentences describing flow physics or one party's use of another's system;
> (3) how a finding, an implication, and a caveat are each stated, verbatim; (4) sentence
> openings and connectives with frequency markers; (5) characteristic verbs with their
> habitual objects; (6) sentence-length statistics and punctuation habits, counting
> em-dashes and semicolons explicitly; (7) ten verbatim sentences most representative of
> the voice, with page numbers. Quote liberally and exactly.

Asking for counts is what turned assumption into evidence here. The "backbone appears zero
times" finding is the one that changed the document, and it only came from counting.
