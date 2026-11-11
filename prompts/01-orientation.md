# Joining the repository

This page is for a person opening the repository for the first time. It says what to install, how to start a session with an agent, and which starter to use for each kind of work. What the repository is for, and how an agent finds its way in it, is in `AGENTS.md` at the root, and that file is the first thing to read.

## What to install

- Python 3.10 or later with the packages in `requirements.txt` for the notebooks, and `requirements-dev.txt` for the tests and the headless re-run.
- An agent tool opened at the repository root. The repository's own harness is built for Claude Code, in the terminal or the IDE, and any tool that can read files in the repository can be oriented with the prompt below.

## How to start a session

**In Claude Code.** Open it at the repository root and type `/orientation`. The agent reads the state of the repository, tells you briefly what it holds, and asks what you want to do. It changes no files. A short summary of the repository's state is injected at the start of every session, and `/orientation` builds on it.

**In any other agent tool.** Paste this:

```
Read AGENTS.md at the root of this repository and follow it. Then tell me briefly what the repository holds, and ask which example I want explained, run or changed, or what question I have.
```

Then answer in plain language. A short instruction is enough, and dictated speech is fine. You can ask what a deck example shows, ask the agent to find a number or a source, or bring a question of your own. When the record has not addressed your question, the agent says so, shows the nearest material, and sketches which knobs of `sandbox.ipynb` could answer it before building anything.

Every number the agent cites comes from a source page or a notebook cell, and it says which. Where the record contradicts something in your request, it says so and says what the record supports. You can reaffirm your direction, and the work proceeds with the disagreement recorded.

## The starters

In Claude Code, each starter in this folder opens with a block to copy into the chat. The block names a skill and asks the agent to explain what it does and then interview you, one question at a time with an example for each, for the inputs that kind of work needs. You do not have to run `/orientation` first: a skill that finds orientation has not run in the session runs it before anything else and tells you so. The exception is `32-github-repository.md`, whose block is the slash command alone, because that skill runs only when you type it and does not run orientation first.

| starter | use it to |
|---|---|
| `11-explain-example.md` | have one example of a deck notebook explained slide by slide, with each number cited to its cell and each source claim to its page |
| `12-run-example.md` | re-run a notebook headless and see what moved against its stored outputs |
| `13-modify-example.md` | change a notebook's knobs or cells, or add an example to the sandbox, with the methodology check after every code change |
| `32-github-repository.md` | commit, push or clean up the working tree, staging only the paths you name |

Writing in any of these goes through the `writing-voice` skill, which the agent invokes on its own when it drafts prose.

## Where to read next

`AGENTS.md` first. Then `README.md` and the August notebook, `workshop2_congestion_management.ipynb`, the first deck of the series. Its formulation section opens with the nomenclature table, which maps every symbol to the code and to words, and `CANON.md` holds the full notation.
