# Starter: change a notebook or the sandbox

Copy the block below into the chat. There is nothing to fill in; the agent runs orientation first if it has not run, then asks you for what it needs.

```
/modify-example
I am new to this repository and I want to change a notebook or add an example to the sandbox. Invoke the modify-example skill. Before you read anything, tell me in a few sentences what the skill does, then walk me through the inputs you need, one question at a time, with an example for each. Confirm the set back to me, then begin.
```

Examples of an ask: "cap line_4 at 30 MW in the August Example 1 and show the redispatch"; "add a scheduling limit at bus 8 to the sandbox"; "print the relief table under the October Example 3 figure". A number you recall is a premise the agent checks against a run before carrying it.

**What the skill does.** It carries the series conventions (self-contained math, the nomenclature table, the program stated whole with its duals, the `*In words.*` blurb after every numbered equation), the deck notebook's fixed form, the standard combined figure, and the methodology check that runs after every code change. The deck notebooks follow their decks, so a new example goes in the sandbox. The library is a byte copy of a shared 9-bus library, so a library change is proposed in `dev_notes.md`'s open decisions.

**What happens next.** The agent restates the ask and the cells it touches, reads the notebook's markdown, stored outputs and knobs, and reports the proposed change at cell level with the figures it will produce and the stored numbers it is expected to move and to hold. It implements the change after you confirm, on a scratch copy unless you ask for the stored results to be replaced, and every code change closes with the three-line methodology record.

**Where to read while you wait.** The nomenclature canon, `CANON.md`, with its conventions, annotated symbol tables and naming rules.
