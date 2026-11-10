# Starter: re-run a notebook

Copy the block below into the chat. There is nothing to fill in; the agent runs orientation first if it has not run, then asks you for what it needs.

```
/run-example
I am new to this repository and I want a notebook re-run. Invoke the run-example skill. Before you read anything, tell me in a few sentences what the skill does, then walk me through the inputs you need, one question at a time, with an example for each. Confirm the set back to me, then begin.
```

Examples of an ask: "re-run the October notebook after today's library copy"; "run the sandbox on a scratch copy and show what moved"; "which notebooks are stale".

**What the skill does.** It reports the staleness of every notebook, runs the named notebook headless through `tools/nb_execute.py`, which stamps the library commit and the library manifest into the notebook's metadata, and compares the new outputs with the stored ones cell by cell. It runs on a scratch copy under `sandbox_exhibits/` unless you ask for the stored results to be replaced.

**What happens next.** The agent asks for the notebook, the mode and a timeout, runs the tests first when the library has changed, runs the notebook, and reports the stamp, each number that moved with its old and new value, the deck anchors that held, and the staleness after the run.

**Where to read while you wait.** `tools/README.md`, which states the stamp's fields and how a notebook is classed stale.
