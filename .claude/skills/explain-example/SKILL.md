---
name: explain-example
description: Explains one example of a deck notebook, slide by slide, from the notebook's stored outputs and the page-pinned source excerpts. Use when a participant asks what a slide shows, where a number comes from, or which source a step rests on. Changes no files.
disable-model-invocation: false
---

# Explaining an example

You are explaining one example of a deck notebook to a reader who has seen, or is about to see, the deck. Each deck notebook follows its deck's slide sequence, so a slide maps to a section and a set of cells, and each number on the slide is a stored output of one of those cells. The explanation states each number from its cell, each source claim from its excerpt, and each modeling choice of this repository as one. `CANON.md` carries the canon, and `orientation` has run. The reconciliation exchange of that skill applies: restate the ask, place it in the literature, answer.

## Intake

If `orientation` has not run in this session, say in one sentence that you are running it first because this skill assumes its reading, invoke it, deliver its brief, and then begin the interview. Skip it only where the participant has said to skip it, and then say which parts of the reading you are proceeding without.

Where an input is missing, interview the participant for it before reading anything, one question at a time, with the AskUserQuestion tool where it is available. Each question says in a sentence what the input is for and gives one example from the repository.

1. **The notebook or the deck date.** The example: "the September 16 deck", which is `workshop3_transactions_across_boundaries.ipynb`.
2. **The slide or the example.** The example: "page 17 of the August deck", or "Example 2, the over-forecast".
3. **The reader's starting point.** The default is "has seen the deck". Another example: "knows nodal pricing but has not seen the deck".

When the set is complete, state it back in the form of the list above and proceed.

## Procedure

1. **Extract the example.** Read the markdown, the stored outputs and the `# -- EDIT --` knobs of the section with `nbformat` under `PYTHONIOENCODING=utf-8`, and the notebook's `metadata.lab_execution` stamp.
2. **Map the slide to cells** through the notebook's slide map in `NOTEBOOK_RECORD.md` section 1. The page numbers there index the excerpt file under `sources/workshops/` that the map names. The August excerpt's page 1 is page 91 of the posted deck, and the October excerpt's pages 1 to 35 are the joint deck's pages 52 to 86 (`sources/workshops/manifest.json`).
3. **State each number from its stored output**, cited as notebook and cell, with the stamp date. Where the stamp is stale against `python tools/nb_staleness.py`, say so and offer `run-example`. A number the slide rounds differently from the cell is stated with both values.
4. **State each source claim with its `references.bib` key and page**, quoting the passage verbatim from `sources/excerpts/<key>.md`. The excerpt file is read before the PDF. Where the needed passage is absent, `quote-locator` reads the tracked PDF at the page and returns it. The passage is then proposed to the participant as an addition to the excerpt file, and this skill does not write it.
5. **A link-only source** (an academic paper or book) is cited by key, page and `url`, and paraphrased beyond the short quotation its excerpt file holds.
6. **Name each modeling choice of this repository as one**, from the notebook's formulation cell: the lossless DC network with PTDF shift factors, the signed dual $\mu_m$ with $+\sum_m SF_{n,m}\mu_m$, the market boundary as fixed injections, and any choice the formulation cell names.
7. **Answer in the chat register** of `writing-voice`: declarative sentences, the number and then the stop, math in Unicode characters.

The explanation follows the slide's own order. Where the reader asks why a number takes its value, the explanation walks the clearing that produced it: which line binds, which units are marginal, and how the price follows from the marginal units' offers and the shift factors.

## Your first task

Restate the slide or example and the cells it maps to. Extract them, then give the explanation, leading with the slide's numbers and their cells. Close with the sources cited, each by key and page, and with any number the stamp leaves stale.

## Sub-agent models

Sub-agents run on the cheapest model tier that fits their task, the rule `CLAUDE.md` states (user decision 2026-10-18), and every brief names the model. In this skill the extraction of cells and stored outputs runs on `haiku` (`cell-extractor`), and the location of a passage in a tracked PDF runs on `sonnet` (`quote-locator`). The explanation is the supervisor's own.
