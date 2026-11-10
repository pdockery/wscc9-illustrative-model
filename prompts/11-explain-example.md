# Starter: have an example explained

Copy the block below into the chat. There is nothing to fill in; the agent runs orientation first if it has not run, then asks you for what it needs.

```
/explain-example
I am new to this repository and I want an example from one of the workshop decks explained. Invoke the explain-example skill. Before you read anything, tell me in a few sentences what the skill does, then walk me through the inputs you need, one question at a time, with an example for each. Confirm the set back to me, then begin.
```

Examples of an ask: "why is bus 7 at $37.10 on slide 7 of the August excerpt (deck page 97)"; "where does the 63 MW in the September Example 4 come from"; "which source says relief is assigned at a 5% impact". A number you recall is a premise the agent checks against the notebook's stored output before carrying it.

**What the skill does.** It maps a slide to the notebook cells that reproduce it, states each number from the cell's stored output with the date the notebook was last executed, quotes each source claim verbatim from the page-pinned excerpt under `sources/excerpts/`, and names each modeling choice the notebook makes. It changes no files.

**What happens next.** The agent asks for the deck or notebook, the slide or example, and where you are starting from, then explains the example in the slide's own order. Where the notebook's stored results are older than the library, it says so and offers `/run-example`.

**Where to read while you wait.** The deck excerpt for the example, under `sources/workshops/`, and the notebook's formulation section.
