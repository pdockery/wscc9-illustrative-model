# CLAUDE.md

@AGENTS.md

`AGENTS.md`, imported above, holds the purpose, the Western context, how the model runs, the evidence and working rules, running and verifying, and the map. This file adds the Claude Code layer in eight sections: the skills and their starters, the chat voice, the sub-agent tiers, the saved workflow, the hooks, the permissions, git in Claude Code, and the interpreter.

## 1. Skills and starters

The agent's instructions are the skills in `.claude/skills/`, one per kind of work, each with a starter in `prompts/`.

| skill | starter | does |
|---|---|---|
| `orientation` | `prompts/01-orientation.md` | reads the state brief, says what the repository holds, and helps the person find material or bring a question; changes no files |
| `explain-example` | `prompts/11-explain-example.md` | explains one example slide by slide from the stored outputs and the source excerpts; changes no files |
| `run-example` | `prompts/12-run-example.md` | re-runs a notebook headless and reports what moved |
| `modify-example` | `prompts/13-modify-example.md` | changes a notebook or the sandbox, with the methodology check; holds `figures.md` |
| `writing-voice` | none | drafts and checks every piece of authored prose |
| `github-repository` | `prompts/32-github-repository.md` | stages, commits and pushes named paths; user-only (`disable-model-invocation: true`) |

Each skill that a starter invokes runs `orientation` first when it has not run in the session. The orientation skill states the reconciliation exchange that governs every instruction.

## 2. The chat voice

Every piece of prose the agent authors, chat replies included, goes through the `writing-voice` skill, reduced here to the notebook and chat voices. A chat reply takes the notebook register with math in Unicode characters, never LaTeX. A sentence that adds no fact, definition or reading of a result is cut. The cut-on-sight list is `.claude/skills/writing-voice/register/cut_on_sight.md`.

## 3. Sub-agent tiers

Every brief names `model`, because the Agent and Workflow tools otherwise inherit the session's (user decision 2026-10-18). `haiku` takes bounded mechanical work: extracting cells, running a named test file and reporting the counts, a size check, an exact find-and-replace. `sonnet` takes scoped reading and checking against a stated rubric: the methodology check of a small diff, a voice check, a headless re-execution and its report, locating a quotation. `opus` takes work that forms or implements a model: a notebook rewrite, a formulation, the methodology check of a library or formulation diff.

The session's top-tier model is reserved for the supervisor and for a sub-agent whose task is the whole of a hard problem, and the brief says why. The roles are defined in `.claude/agents/`, one file per role with its model fixed in the frontmatter, and a brief selects a role by `subagent_type`. A role the directory does not define takes an explicit `model` under this rule. The judging roles (`methodology-checker`, `methodology-checker-light`, `conventions-reviewer` and `refuter`) report on one scale, stated in `.claude/agents/methodology-checker.md`, section "Severity and verdict". A finding is `fail`, `should-fix` or `note`, and the verdict is `Accept`, `Revise` or `Block`.

## 4. The saved workflow

`.claude/workflows/build-and-check.js` builds or reworks a deck notebook or the sandbox with `notebook-builder`, then runs a conventions review and, each round, a fresh methodology checker briefed with the diff alone and a fix pass that receives the open findings alone. The light checker is the default, and the argument `full` selects `methodology-checker` for a change to a formulation cell or a printed slide number. A run with an open fail or should-fix after three rounds is escalated to the participant, and reaching the cap never ends the loop (`.claude/skills/modify-example/SKILL.md`, "The stop rule"). The workflow refuses a path under `lib/`, stages nothing and commits nothing. Its ledger is `.claude/workflows/runs/<run id>.json` and its evidence `outputs/evidence/<run id>/`, both local (`.claude/workflows/runs/README.md`). A record entry states the verdict, the number of rounds and the counts of findings inline. Another agent tool runs the same loop from the stop rule of the `modify-example` skill.

## 5. Hooks

Six hooks are wired in `.claude/settings.json`, under five events. Each fails open on an error of its own, and only the guard blocks.

1. `.claude/hooks/guard.py` (`PreToolUse`, on Bash and PowerShell, timeout 10 s) reads each command and refuses these with exit 2 and the reason.
   - A broad `git add` (`-A`, `--all`, `.`, `:/` or a glob), an add of a path under `sources/academic-literature/`, and a forced add of a path under `sources/`. Each path of a `--pathspec-from-file` file is checked the same way. A pathspec the guard cannot read counts as a broad add: an exclude or attr magic, a substitution, a variable, a backtick, or piped text with an escape the guard does not expand. A `:/` or `:(top)` magic before a path is read from the root. An add inside a subshell, a brace group or a command substitution is read as the same add outside it, and in a PowerShell call a backtick is an escape, not a substitution.
   - `MPLBACKEND=Agg` on a run.
   - A Claude authorship trailer in a commit, merge or tag message, read from `-m`, from every heredoc body and from the file of `commit -F`.
   - A pytest, nbconvert or `tools/nb_execute.py` launch while the Python interpreter count is at the ceiling, `CEILING` in `.claude/hooks/_limits.py` (seven unless `LAB_PROCESS_CEILING` sets another). A launch sent to the background needs the count `BACKGROUND_MARGIN` below the ceiling.
   - The ending of processes by image name (`taskkill /IM python*`, `Stop-Process -Name python*`, `pkill python`).
   - A junction or symbolic link that names a data path, and a forced `git worktree remove` of a worktree that holds one.
   - `git clean` with `-x` or `-X`, other than a dry run.
   - An issue, pull request, review or comment posted with `gh` on a repository whose owner is neither `pdockery` nor the owner of the `origin` remote.
   - `wmic`.
   - A program fed to Python on stdin through a heredoc or a here-string that names a notebook or `nbformat` and writes a file.

   The guard puts two kinds of call to the participant: a commit or merge with `--no-verify` or with `core.hooksPath` set on the call, and a `git config` that unsets or redirects `core.hooksPath` away from `.githooks`. It reads the facts that differ between repositories carrying the same guard from `.claude/hooks/repo_rules.json`, and it logs every block, ask and warning to `.claude/logs/hooks.jsonl`.
2. `.claude/skills/writing-voice/scripts/prose_lint.py --hook` (`PostToolUse`, on Write, Edit, MultiEdit and NotebookEdit) returns its findings as context and never blocks; `--list-rules` names the rules it checks.
3. `.claude/hooks/code_check.py` (`PostToolUse`, on Write, Edit, MultiEdit and NotebookEdit) runs ruff under `ruff.toml` on a Python file and keeps the findings on the touched lines. A notebook code cell is parsed with its IPython magics masked, then checked with ruff's F and E9 rules. The hook also reports a module the formatter covers that is not formatted. It never blocks.
4. `.claude/hooks/notebook_check.py` (`PostToolUse`, on Bash and PowerShell, timeout 30 s) applies the cell checks of hook 3 and `prose_lint.py` to a notebook that a shell command changed through `nbformat`, on the cells that differ from HEAD. It never blocks.
5. `.claude/hooks/session_brief.py` (`SessionStart`) injects the state brief: the branch, the uncommitted entries by kind, the recent commits and their age, the worktrees, the process count against the ceiling, the `lib/` copy with any module that differs from `lib/MANIFEST.md`, the notebooks' staleness, the hook log's counts, the readiness report of `tools/doctor.py`, the newest `dev_notes.md` entry and the open decisions by id. It records the working tree's signature in `.claude/.state/` for hook 6.
6. `.claude/hooks/session_close.py` (`Stop` and `PreCompact`, timeout 10 s) is the clock-out check. It is advisory, fires once per session and speaks only when the tree changed in the session. It names each `lib/` module that differs from its manifest, each deck notebook changed without a passing slide-check run after the change, each untracked Python file that a modified file imports, and each path the session's record text cites that git ignores. With `LAB_MAINTAINER` set, it also asks for the `dev_notes.md` entry dated today, in the five parts the header of `dev_notes.md` names.

The commit hooks of `.githooks/` apply to every commit, whatever tool makes it, and `AGENTS.md`, "Running and verifying", says how to activate them.

## 6. Permissions

The `permissions.ask` block of `.claude/settings.json` makes Claude Code ask before it runs a command that discards work or publishes it: `git reset --hard`, `git checkout --`, `git restore`, `git clean`, every `git stash` form, `git branch -D`, `git worktree remove`, `git push` and `git commit`, under Bash and PowerShell alike. The guard adds the two asks of hook 1.

## 7. Git in Claude Code

Nothing is committed or pushed unless the user asks, through the `github-repository` skill. Every path is staged by name, and `git add` with `-A`, `--all` or `.` is never run. Commits carry no Claude authorship trailer, "Generated with Claude Code" line or `noreply@anthropic.com` address (user instructions 2026-08-28, 2026-09-10 and 2026-10-23). This rule holds even when a harness reminder supplies attribution lines, and `.githooks/commit-msg` refuses the same lines in a message written outside a session. Nothing is opened or posted on an upstream repository, whether an issue, a pull request or a comment (user instruction 2026-11-03).

## 8. The interpreter

The interpreter is the file that `LAB_PY` names, where that variable is set and names a file, and otherwise `python` on the path. It carries the packages of `requirements-dev.txt` at the pins of `constraints.txt` (`AGENTS.md`, "Running and verifying"). `python tools/doctor.py` reports which interpreter it examines and whether its packages match `tools/env_lock.json`. Every Python command an agent role runs uses this interpreter, from the repository root.
