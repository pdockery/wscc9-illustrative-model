# Cut on sight, and the sentence mechanics

This list holds in every voice and in chat replies. It moved here from `CLAUDE.md` on 2026-11-01. The dated provenance of each item is `notes/writing_profile/source_register_profile.md`: the sentence mechanics in section 5, the overlay in section 6, the worked before-and-after in section 11, and the dated additions to this list in section 10. `python .claude/skills/writing-voice/scripts/prose_lint.py --list-rules` names the items checked mechanically; `scripts/style_check.py` quotes the sentences that carry the rest as tells.

## The repository overlay

The repository overlay on top of the sources is **active voice with the subject next to its verb**, which departs from the ISO documents' agentless passive.

## Seven things to cut on sight

- **Intensifiers and evaluation**: "crucially", "importantly", "notably", "it is worth noting", "powerful", "elegant", "striking", "a genuine improvement", "the whole point of the exercise". If a result matters, the number shows it.
- **Reader-address and drama**: "notice…", "look at who does what", "watch what happens", "that is the trap". State what the dispatch does; the reader draws the inference.
- **Meta-commentary that narrates the document instead of the subject**: "the frame", "naming the pieces first", "in this section we will…", "let's now turn to…", "as we saw above", "the goal of this section is…". Open with substance. Two further forms. **Self-labelling metadiscourse**: the document names, ranks, or glosses its own sentences ("That sentence is the purpose statement, and the rest of this document is subordinate to it"; "It restates, in the repository's words, …"), which is metadiscourse in Hyland's sense (endophoric and frame markers). Delete the label and let the sentence stand. **The negative scope catalogue**: a scope stated as a chain of what the thing is not ("They are not a tariff, they do not rank …, and they do not address …"). State the scope affirmatively, and state one exclusion flatly, once, only where a reader would otherwise assume the opposite.
- **The imperative where the declarative is available**: "The clearing minimizes production cost subject to…", not "Minimize production cost…"; "BA $a$'s Method 2 allocation $A^{(2)}_a$ is…", not "Give each area…".
- **The empty contrast**: an "X, not Y" sentence whose two sides are not both defined in the document, that adds no claim its neighbors lack, or that restates the adjacent sentence with the terms mirrored ("complements, not substitutes" → "complementary"; the aphorism "if everyone is accountable, no one is accountable"; a closer re-labelling the previous sentence). A contrast stays only when it carries its own testable claim.
- **The manufactured parallel**: staging two subjects as one underlying thing ("side by side as two instances of one design question", "two sides of the same coin", "the same instrument applied to the two commodities") instead of stating the shared property as its own claim ("both are short-run eligibility criteria for actors with long-run independence"). Promoting a source's "is like" to "is the same" is the same fault; the retired CRA-and-seams framing is an instance.
- **The aphoristic thesis opener**: a sentence that announces that the subject has a hidden structure without stating it ("Congestion rent in a nodal market is one quantity with two readings", "one problem with two faces"), so the claim arrives only in the sentences after it. It conveys no information. Open with the first fact the teaser was withholding ("When a transmission constraint binds, prices separate and load pays more for energy than generation is paid").

Interpretation of a figure or printed result is ordinary prose under the output, with no marker and no bolded lead-in. `*In words.*` is the only labelled construct and it belongs to numbered equations alone.

## Sentence mechanics

These were distilled from the user's five editorial passes; the user's words and the date are in the profile, section 5.

- Short declarative sentences, one claim each. If a sentence needs a colon, semicolon, or dash to hold together, split it. Colons only to introduce a list or a definition.
- Em-dashes live in section titles and table cells only. No asides, appositives, or option-catalogs in body prose.
- Bold is for headings, paragraph subject statements ("**The superposed flow.**"), and a term's first definition. Never mid-sentence emphasis.
- Say each thing once, in its home: sibling pointers at the technical hand-off or a Related-notebooks section, never repeated in closings; facts a table, matrix, or figure suptitle carries are not restated in prose.
- State the number, then stop. No interpretive coda unless it adds a new fact.
- No caption cells narrating figures whose suptitles already carry the content.
- Impersonal voice for the notebook's devices ("A dial is used in this notebook"). Scope limits are one flat list sentence ("X, Y, and Z are addressed only narratively here and left for subsequent notebooks").
- Prefer the formulation's own terms ("exogenous", "flow contribution") over informal glosses ("sized from").

## The In-words exemption

**The `*In words.*` form overrides the sentence-splitting mechanics inside its own blocks**, because splitting damaged the readings. The blurb reads the equation left to right following the equation's structure, names every quantity by its nomenclature-table name with the symbol inline ("the absolute shadow price $|\mu_m|$ times the absolute flow $|F_m|$"), defines a symbol before using it, and may keep the "subject to two constraint families: A (dual X), and B (dual Y)" colon-list shape because it mirrors the equation. NB-04 in `exemplars/notebook.md` is the exemplar. A symbol is never detached from its named quantity to satisfy a splitting rule.
