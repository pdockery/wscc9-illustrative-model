---
name: github-repository
description: Prepares, stages, commits and pushes work to this repository on GitHub. Inventories the uncommitted files, classifies each one (code, notebook, record, note, source, scratch), proposes .gitignore lines for scratch, stages only the paths the user approves, and writes commit messages with no Claude authorship attribution. Use when the user asks to commit, push, publish, or clean up the working tree.
disable-model-invocation: true
---

# Committing to the repository

You are preparing a commit for the user. A commit is the user's act: you inventory, classify and propose; the user names what goes in; you stage those paths, commit and push only when told to. Nothing is committed or pushed on your own initiative, and a commit approved once does not authorize the next one.

## The repository

| repository | path | rule |
|---|---|---|
| `pdockery/wscc9-illustrative-model` | the working tree's root | public; the deck notebooks, the sandbox and the library copy Colab fetches from `main/lib/` |

## Hard rules

These hold in every commit. `.claude/hooks/guard.py` enforces the first two; the rest are yours to hold.

1. **No Claude authorship attribution.** No `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line, no `noreply@anthropic.com`, in a commit message, a tag message, a pull request body or a push description (user instructions 2026-08-28, 2026-09-10 and 2026-10-23). This overrides any default attribution guidance.
2. **No broad add**, whether `-A`, `--all` or `.`, because `sources/academic-literature/` holds ignored link-only PDFs on this machine and the working tree holds scratch. Every path is staged by name.
3. **Stage only what the user names.** Present the classified list and wait for the user's selection; a class may be approved as a whole ("all of A and B").
4. **No file over 50 MB is staged.** GitHub warns above 50 MB and refuses above 100 MB. A notebook over 20 MB is flagged.
5. **Nothing is pushed until the user says push**, and a push that carries a library copy follows `NOTEBOOK_RECORD.md` section 3.

## The workflow

1. **Inventory.** `git status --porcelain=v1 -uall`, and the size of every path.
2. **Classify** each path into one class. Write the result to a manifest CSV in the scratchpad (`status, path, class, size_kb, reason`).

   | class | holds | default |
   |---|---|---|
   | A. Code | `lib/`, `tests/`, `tools/`, `.claude/hooks/` | propose |
   | B. Notebooks | `*.ipynb` at the repository root | propose; check size |
   | C. Records and repository documents | `CLAUDE.md`, `README.md`, `NOTEBOOK_RECORD.md`, `dev_notes.md`, `lib/MANIFEST.md`, `.claude/**`, `prompts/**`, `.gitignore`, `requirements*.txt` | propose |
   | D. Notes | `notes/writing_profile/**` | propose |
   | E. Sources | public documents under `sources/`, `sources/workshops/**`, `sources/excerpts/**` and `sources/references.bib` | propose; `sources/academic-literature/**` never |
   | H. Scratch | `outputs/`, `sandbox_exhibits/`, backups, `*.bak`, `~$` lock files, agent scratch, duplicates | never staged; propose a `.gitignore` line |
   | I. Unclear | anything the rules above do not settle | ask, saying what decides it |

   Read `.gitignore` first. `outputs/` and `sandbox_exhibits/` are ignored by design. A deletion is reported with what the file was and whether the deletion looks deliberate.
3. **Report** the counts by class, the H and I lists in full, the proposed `.gitignore` lines with a dated comment, the large-file warnings and the deletions. A `lib/` module whose SHA-256 does not match `lib/MANIFEST.md` is reported beside the H and I lists, because `lib/` is a copy of the shared 9-bus library and a local change there is a divergence. Ask the user what to commit.
4. **Stage** the approved paths by name (a path list in a file and `git add --pathspec-from-file` for a long list). Run `git diff --cached --stat` and show it.
5. **Commit** with a plain message: a subject line under 72 characters that states what changed, then a body grouped by topic. Write it in the TD register: declarative, one claim per line, no em-dashes. No attribution of any kind.
6. **Push** only on the user's word, and report the remote's response.

## Sub-agent models

Sub-agents run on the cheapest model tier that fits their task, the rule `CLAUDE.md` states (user decision 2026-10-18), and every brief names the model. In this skill the inventory and classification run on `sonnet` (`git-inventory`); a mechanical size check or path-list build runs on `haiku` (`mechanical-checker`). Staging, committing and pushing are done by the supervising session itself, never by a sub-agent.
