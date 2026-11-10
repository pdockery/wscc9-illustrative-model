<!-- voice: TD (technical description) -->
# Workflow run ledgers

A saved workflow in `.claude/workflows/` records the status of each of its items in one ledger file here, `<run id>.json`. A paused or failed run can then be resumed from any session, because the ledger is on disk and not in a session transcript. The ledgers are local state and are not committed.

## The run id

The run id is `<date>_<workflow>-<n>`, for example `2026-11-10_build-and-check-1`. The date is the local date the run started. `n` is the smallest positive integer for which neither the ledger nor the evidence directory exists. The workflow's setup agent chooses it, because a workflow script cannot read the clock or the disk.

## The evidence directory

Each run writes what it cites to `outputs/evidence/<run id>/`, which `outputs/` in the root `.gitignore` covers, so the evidence stays on the machine that ran the workflow. The directory holds the checker returns, the attribution and patch scripts a fix pass writes, and the baseline copies of the owned paths taken before the build. A record entry states the verdict, the number of rounds and the counts of findings inline, and does not depend on the evidence directory being available to a reader. A file moved aside during a run goes to `outputs/quarantine/<run id>/`.

## The ledger

The ledger is one JSON object, rewritten whole at each checkpoint.

```json
{
  "run_id": "2026-11-10_build-and-check-1",
  "workflow": "build-and-check",
  "date": "2026-11-10",
  "args": {},
  "status": "running",
  "checkpoint": "round 2",
  "evidence": "outputs/evidence/2026-11-10_build-and-check-1/",
  "items": {
    "check": {"status": "done", "stage": "check", "detail": "round 2 Accept"}
  }
}
```

1. `status` is the run's: `running`, `ok`, `failed`, `blocked` or `escalated`.
2. `checkpoint` names the last phase or round that completed.
3. `items` maps each item (the build, the check, the verification) to its own `status`, one of `pending`, `running`, `done`, `failed`, `blocked` or `escalated`, with the `stage` it reached and a one-line `detail`.
4. `rounds` lists each check round with its verdict, the verdict of the conventions review and the number of open findings.

A workflow resumed with `resume_run_id` reads its ledger in the setup step and skips every item whose status is `done`. It repeats a step only for the items that did not complete that step.

## Status meanings

1. `failed`: a command the workflow depends on returned a non-zero exit code, or an agent returned nothing.
2. `blocked`: a check still finds a defect that the allowed fix rounds did not close.
3. `escalated`: the round cap was reached with an open fail or should-fix, and the decision belongs to the participant. The loop is not ended, because the methodology check runs again until a pass finds nothing. The participant resumes the run, holds the finding, or changes the task.
