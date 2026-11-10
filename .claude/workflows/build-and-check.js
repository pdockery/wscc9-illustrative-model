export const meta = {
  name: 'build-and-check',
  description: 'Build or rework a deck notebook or the sandbox, review its conventions, then run a fresh methodology check and a findings-only fix pass each round until a check accepts',
  whenToUse: 'A notebook rewrite or build (notebook-builder) whose methodology check must come from a fresh reader each round. Refuses a path under lib/. Escalates to the participant after three rounds with an open fail or should-fix. Commits nothing.',
  phases: [
    { title: 'Setup', detail: 'the repository root, the interpreter, the date, the run id, the ledger and the baseline copies', model: 'haiku' },
    { title: 'Build', detail: 'notebook-builder', model: 'opus' },
    { title: 'Check', detail: 'a conventions review, and a fresh methodology checker each round briefed with the diff only' },
    { title: 'Fix', detail: 'a fix pass that receives the open findings only', model: 'opus' },
    { title: 'Verify', detail: 'the stamped re-execution, then the test commands, one after another' },
  ],
}

// args: {
//   task: string,            the build brief: what to build or change, in which notebook
//   paths: string[],         the repository-relative files the builder owns; nothing else is edited
//   keys?: string[],         the references.bib keys the change implements, for the checker
//   full?: boolean,          methodology-checker in place of methodology-checker-light; set it when a
//                            formulation cell or a printed slide number changes
//   execute?: boolean,       re-execute each .ipynb path through tools/nb_execute.py after acceptance (default true)
//   tests?: string[],        test commands run after acceptance, one after another
//   skip_build?: boolean,    check a change already in the tree
//   base?: string,           the commit the baseline is read from when skip_build is set
//   max_rounds?: number,     rounds before escalation in this invocation (default 3)
//   resume_run_id?: string,  resume a run from its ledger in .claude/workflows/runs/
// }
// Evidence: outputs/evidence/<run id>/ (ignored, local only). Ledger: .claude/workflows/runs/<run id>.json.
// The workflow commits nothing, stages nothing, stashes nothing and adds no authorship trailer.

const NAME = 'build-and-check'
const A = args || {}
const CHECKER = A.full ? 'methodology-checker' : 'methodology-checker-light'
const CHECKER_MODEL = A.full ? 'opus' : 'sonnet'
const MAX_ROUNDS = A.max_rounds || 3
const PATHS = (A.paths || []).map(p => String(p).replace(/\\/g, '/'))
const KEYS = A.keys || []
const EXECUTE = A.execute !== false
// The repository root and the interpreter come from the setup agent at run time.
let ROOT = '.'

const SETUP = {
  type: 'object',
  properties: {
    root: { type: 'string', description: 'the repository root, from git rev-parse --show-toplevel' },
    python: { type: 'string', description: 'the interpreter of CLAUDE.md section 8, as a command' },
    date: { type: 'string', description: 'YYYY-MM-DD' },
    has_check: { type: 'boolean', description: 'whether tools/check.py exists' },
    untracked: { type: 'array', items: { type: 'string' }, description: 'each owned path that git does not track' },
    run_id: { type: 'string' },
    ledger: { type: 'string', description: 'the resumed ledger file text, or an empty string' },
  },
  required: ['root', 'python', 'date', 'has_check', 'untracked', 'run_id', 'ledger'],
}
const BUILT = {
  type: 'object',
  properties: {
    changed: { type: 'array', items: { type: 'string' }, description: 'each file changed, with the cells changed for a notebook' },
    outside: { type: 'array', items: { type: 'string' }, description: 'any file changed outside the owned paths, with the reason' },
    tests: { type: 'array', items: { type: 'string' }, description: 'each test command run, with its counts' },
    numbers: { type: 'array', items: { type: 'string' }, description: 'stored numbers that moved and held, old and new' },
    report: { type: 'string' },
  },
  required: ['changed', 'outside', 'tests', 'numbers', 'report'],
}
// The checker schema of methodology-checker.md, section "Severity and verdict".
const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          line: { type: 'string', description: 'the line, the line range or the notebook cell id' },
          question: { type: 'string', description: '1, 2 or 3 for the methodology check; the rubric item for the conventions review' },
          severity: { type: 'string', enum: ['fail', 'should-fix', 'note'] },
          fix: { type: 'string' },
        },
        required: ['file', 'line', 'question', 'severity', 'fix'],
      },
    },
    record: { type: 'string', description: 'the three-line record, or one line per rubric item read' },
    verdict: { type: 'string', enum: ['Accept', 'Revise', 'Block'] },
  },
  required: ['findings', 'record', 'verdict'],
}
const SUITE = {
  type: 'object',
  properties: {
    command: { type: 'string' },
    launched: { type: 'boolean', description: 'false when the process ceiling stopped the launch' },
    exit_code: { type: 'integer' },
    passed: { type: 'integer' },
    failed: { type: 'integer' },
    skipped: { type: 'integer' },
    errors: { type: 'integer' },
    failures: { type: 'array', items: { type: 'string' } },
  },
  required: ['command', 'launched', 'exit_code', 'passed', 'failed', 'skipped', 'errors', 'failures'],
}
const EXEC = {
  type: 'object',
  properties: {
    notebook: { type: 'string' },
    launched: { type: 'boolean', description: 'false when the process ceiling stopped the launch' },
    exit_code: { type: 'integer' },
    stamped: { type: 'boolean', description: 'whether the notebook metadata carries a fresh execution stamp' },
    size_mb: { type: 'number' },
    changed: { type: 'array', items: { type: 'string' }, description: 'each printed number that differs from the stored value, old and new' },
  },
  required: ['notebook', 'launched', 'exit_code', 'stamped', 'size_mb', 'changed'],
}

const RULES = () => `Working rules for this run:
1. Edit only these files: ${PATHS.join(', ')}. A change needed elsewhere is reported under "outside" and not made. A file under lib/ is never edited.
2. A notebook is edited by cell, through NotebookEdit or an nbformat script written with Write. No shell heredoc writes a patch script, no notebook is regenerated by a builder script, and no notebook JSON is edited as text.
3. No git command that changes the index, a branch or the working tree: no add, commit, stash, reset, checkout, restore or push.
4. A Python command runs under the interpreter named below. A test, a notebook execution or a clearing script launches only while python_count() in .claude/hooks/_limits.py sits below CEILING, and one runs at a time, in the foreground.
5. Any patch script, attribution script or probe you write goes in the evidence directory named below, not in a session scratchpad, so that a later session can open it.`

const SETUP_BRIEF = resume => `Return six values and write nothing.
1. root: the output of \`git rev-parse --show-toplevel\` from the current folder.
2. python: the interpreter of CLAUDE.md section 8 as a command that runs from the root: the file the LAB_PY variable names where it is set and is a file, and otherwise \`python\`. Confirm that \`<command> -m pytest --version\` prints a version, and return the command.
3. date: today's local date as YYYY-MM-DD, from \`date +%F\`.
4. has_check: whether tools/check.py exists under the root.
5. untracked: each of these paths that \`git ls-files --error-unmatch <path>\` does not report as tracked:
${PATHS.join('\n')}
6. run_id and ledger: ${resume
    ? `run_id is "${resume}", and ledger is the text of .claude/workflows/runs/${resume}.json, verbatim, or an empty string if it does not exist.`
    : `run_id is the smallest n >= 1 for which neither .claude/workflows/runs/<date>_${NAME}-<n>.json nor outputs/evidence/<date>_${NAME}-<n> exists, returned as <date>_${NAME}-<n>, and ledger is an empty string.`}`

const BASELINE_BRIEF = ev => `From ${ROOT}, copy each of these files that exists into ${ev}baseline/ under its repository-relative path, creating folders as needed, and list the ones that do not exist:
${PATHS.join('\n')}
${A.skip_build
    ? `Copy each file as it stands at ${A.base} (git show ${A.base}:<path>), not the working tree, because the change under check is already in the tree.`
    : 'Copy the working-tree file as it stands now, before the build.'}
Write nothing else and run no git command that changes the index or the working tree.`

const DIFF = ev => `The change is the difference between the baseline copies and the working tree, for these paths only:
${PATHS.map(p => `git diff --no-index ${ev}baseline/${p} ${p}`).join('\n')}
A path with no baseline copy is new: read it whole. For a notebook, extract the changed cells with nbformat under PYTHONIOENCODING=utf-8 and compare them by cell id with the baseline copy; never execute a notebook. Uncommitted hunks of another change can sit in the same files at HEAD, so the baseline copy, not HEAD, is the reference.`

const BUILD_BRIEF = (py, ev) => `Repository: ${ROOT}. Interpreter: ${py}. Evidence directory: ${ev}.

${A.task || '(no task given)'}

${RULES()}

Before returning, run the three questions of the methodology check on your own diff and fix each failure. Return the files and cells changed, any file outside the owned paths, the tests run with their counts, the stored numbers that moved and held, and your report with the three-line record.`

const CHECK_BRIEF = (py, ev, round) => `Repository: ${ROOT}. Interpreter: ${py}. Round ${round} of a fresh methodology check.

${DIFF(ev)}

${KEYS.length ? `The change implements these references.bib keys: ${KEYS.join(', ')}.` : 'The brief names no references.bib key: name the source the change implements, or flag it as a modeling choice of this repository.'}

Answer the three questions of the methodology check from the code as it stands. You receive the diff only: no builder report and no earlier check, by design. Rate each finding fail, should-fix or note on the scale of your role file, and close with the three-line record and the verdict Accept, Revise or Block. Read-only.`

const CONVENTIONS_BRIEF = (py, ev) => `Repository: ${ROOT}. Interpreter: ${py}.

${DIFF(ev)}

Review the change against CANON.md (sections 1, 3 and 8), figures.md and CODE_STYLE.md (changed lines only), by your role file's rubric. Rate each finding fail, should-fix or note, and close with the verdict. Read-only.`

const FIX_BRIEF = (py, ev, open) => `Repository: ${ROOT}. Interpreter: ${py}. Evidence directory: ${ev}.

A fresh reader checked the change to these files and found the items below. Fix each one in the code or the notebook, or, where an item is wrong on the code as it stands, say why with the evidence. Change nothing the items do not reach.
${JSON.stringify(open, null, 2)}

${RULES()}

Run the tests the change reaches, one file at a time. Return the files and cells changed, any file outside the owned paths, the tests run with their counts, the stored numbers that moved and held, and, in the report, each item by number with what was done.`

// The ledger and the evidence files are written by a mechanical agent, because a workflow script has no file access.
const ledger = { run_id: '', workflow: NAME, date: '', args: A, status: 'running', checkpoint: '', evidence: '', items: {}, rounds: [] }
async function persist(checkpoint, status, files) {
  ledger.checkpoint = checkpoint
  if (status) ledger.status = status
  const all = (files || []).concat([{ path: `.claude/workflows/runs/${ledger.run_id}.json`, content: JSON.stringify(ledger, null, 2) }])
  await agent(`From ${ROOT}, write each file below with exactly the content given, with LF line endings, creating folders as needed. Write nothing else and run nothing else.
${all.map(f => `=== ${f.path}\n${f.content}`).join('\n')}`, { label: 'record:' + checkpoint, phase: 'Setup', model: 'haiku' })
}
const isOpen = f => f.severity === 'fail' || f.severity === 'should-fix'

phase('Setup')
if (!PATHS.length || (!A.task && !A.skip_build)) {
  log('args.paths and args.task are required (args.task may be omitted with skip_build)')
  return { ok: false, stage: 'args' }
}
const libPaths = PATHS.filter(p => /^(\.\/)?lib\//.test(p))
if (libPaths.length) {
  log(`refused: this workflow does not edit lib/, which is copied from its source and never edited here (${libPaths.join(', ')}). A library change is proposed in dev_notes.md's open decisions.`)
  return { ok: false, stage: 'args', refused: libPaths }
}
const resume = A.resume_run_id
const env = await agent(SETUP_BRIEF(resume), { label: 'setup', phase: 'Setup', model: 'haiku', schema: SETUP })
if (!env) {
  log('the setup agent returned nothing; the run stops before any change')
  return { ok: false, stage: 'setup' }
}
ROOT = env.root || ROOT
const PY = env.python || 'python'
const EV = `outputs/evidence/${env.run_id}/`
if (A.skip_build && env.untracked.length && !A.base) {
  // HEAD holds no copy of an untracked path, so a baseline taken from it would be empty or wrong.
  log(`refused: skip_build needs a base commit while these paths are untracked: ${env.untracked.join(', ')}. Pass args.base, or run the build.`)
  return { ok: false, stage: 'args', untracked: env.untracked }
}
if (A.skip_build && !A.base) A.base = 'HEAD'
ledger.run_id = env.run_id
ledger.date = env.date
ledger.evidence = EV
let prior = null
if (resume && env.ledger) {
  try { prior = JSON.parse(env.ledger) } catch (e) { log('the resumed ledger does not parse; the run starts from the build') }
}
if (prior) {
  ledger.items = prior.items || {}
  ledger.rounds = prior.rounds || []
}
const built = Boolean(A.skip_build || (prior && prior.items && prior.items.build && prior.items.build.status === 'done'))
if (!prior) await agent(BASELINE_BRIEF(EV), { label: 'baseline', phase: 'Setup', model: 'haiku' })
log(`run ${ledger.run_id}, ${CHECKER}, evidence ${EV}`)

phase('Build')
let build = null
if (!built) {
  ledger.items.build = { status: 'running', stage: 'build', detail: 'notebook-builder' }
  build = await agent(BUILD_BRIEF(PY, EV), { label: 'build', phase: 'Build', agentType: 'notebook-builder', model: 'opus', schema: BUILT })
  if (!build) {
    ledger.items.build = { status: 'failed', stage: 'build', detail: 'the builder returned nothing' }
    await persist('build', 'failed')
    return { ok: false, run_id: ledger.run_id, stage: 'build' }
  }
  ledger.items.build = { status: 'done', stage: 'build', detail: build.changed.join('; ') }
  await persist('build', null, [{ path: `${EV}build.json`, content: JSON.stringify(build, null, 2) }])
  if (build.outside.length) log(`the builder reports changes outside the owned paths: ${build.outside.join('; ')}`)
}

// Each round runs a fresh checker on the diff. The conventions review runs in the first round of this
// invocation and again while it has an open item. A fix pass receives the open items only. The loop ends
// when a checker accepts or the round cap escalates; reaching the cap is never a pass.
const start = ledger.rounds.length
let conventionsOpen = true
let last = null
let status = 'running'
for (let k = 1; ; k++) {
  const round = start + k
  phase('Check')
  const [check, conv] = await parallel([
    () => agent(CHECK_BRIEF(PY, EV, round), { label: `check:${round}`, phase: 'Check', agentType: CHECKER, model: CHECKER_MODEL, schema: FINDINGS }),
    () => (conventionsOpen
      ? agent(CONVENTIONS_BRIEF(PY, EV), { label: `conventions:${round}`, phase: 'Check', agentType: 'conventions-reviewer', model: 'sonnet', schema: FINDINGS })
      : Promise.resolve(null)),
  ])
  if (!check) {
    status = 'failed'
    ledger.rounds.push({ round, verdict: 'none', open: 0, detail: 'the checker returned nothing' })
    await persist(`round ${round}`, status)
    break
  }
  const convOpen = conv ? conv.findings.filter(isOpen) : []
  conventionsOpen = convOpen.length > 0
  const open = check.findings.filter(isOpen).map(f => Object.assign({ from: 'methodology' }, f))
    .concat(convOpen.map(f => Object.assign({ from: 'conventions' }, f)))
  last = { round, check, conventions: conv, open }
  ledger.rounds.push({
    round,
    verdict: check.verdict,
    conventions: conv ? conv.verdict : 'not run',
    open: open.length,
    evidence: `${EV}round${round}_check.json`,
  })
  const files = [{ path: `${EV}round${round}_check.json`, content: JSON.stringify(check, null, 2) }]
  if (conv) files.push({ path: `${EV}round${round}_conventions.json`, content: JSON.stringify(conv, null, 2) })
  if (!open.length && check.verdict !== 'Accept') {
    // A verdict that its findings do not support is not resolved by a fix pass.
    status = 'escalated'
    ledger.items.check = { status: 'escalated', stage: 'check', detail: `round ${round}: verdict ${check.verdict} with no open finding` }
    await persist(`round ${round}`, status, files)
    log(`ESCALATE TO THE PARTICIPANT: the checker of round ${round} returned ${check.verdict} with no open fail or should-fix. Read ${EV}round${round}_check.json.`)
    break
  }
  if (!open.length) {
    status = 'ok'
    ledger.items.check = { status: 'done', stage: 'check', detail: `round ${round} ${check.verdict}` }
    await persist(`round ${round}`, null, files)
    break
  }
  if (k >= MAX_ROUNDS) {
    status = 'escalated'
    ledger.items.check = { status: 'escalated', stage: 'check', detail: `${open.length} open after round ${round}` }
    await persist(`round ${round}`, status, files)
    log(`ESCALATE TO THE PARTICIPANT: ${open.length} open fail or should-fix items after ${k} rounds in this invocation. Resume with resume_run_id ${ledger.run_id}, hold the items, or change the task.`)
    break
  }
  ledger.items.check = { status: 'running', stage: 'fix', detail: `${open.length} open after round ${round}` }
  await persist(`round ${round}`, null, files)

  phase('Fix')
  const fix = await agent(FIX_BRIEF(PY, EV, open), { label: `fix:${round}`, phase: 'Fix', agentType: 'notebook-builder', model: 'opus', schema: BUILT })
  if (!fix) {
    status = 'failed'
    ledger.items.check = { status: 'failed', stage: 'fix', detail: `the fix pass of round ${round} returned nothing` }
    await persist(`fix ${round}`, status)
    break
  }
  await persist(`fix ${round}`, null, [{ path: `${EV}round${round}_fix.json`, content: JSON.stringify(fix, null, 2) }])
  if (fix.outside.length) log(`the fix pass reports changes outside the owned paths: ${fix.outside.join('; ')}`)
}

phase('Verify')
const execs = []
const tests = []
if (status === 'ok') {
  // One Python process at a time (test-runner.md): each command runs in the foreground, after the one before.
  if (EXECUTE) {
    for (const nb of PATHS.filter(p => p.endsWith('.ipynb'))) {
      const out = await agent(`Re-execute ${nb} from ${ROOT} with \`${PY} tools/nb_execute.py ${nb}\` in the foreground, so that the library commit, the diff hash and the manifest are stamped into its metadata. MPLBACKEND is never set. Report whether it launched, the exit code, whether the metadata carries a fresh stamp, the file size in MB, and each printed number that differs from the stored value, old and new.`,
        { label: 'execute', phase: 'Verify', agentType: 'notebook-reexecutor', model: 'sonnet', schema: EXEC })
      execs.push({ notebook: nb, out })
    }
  }
  const commands = (A.tests && A.tests.length)
    ? A.tests
    : (env.has_check
      ? [`${PY} tools/check.py --full`]
      : [`${PY} -m pytest tests -q`, `${PY} -m pytest tests -q -m slides`])
  for (const c of commands) {
    const out = await agent(`Run this from ${ROOT} in the foreground and report the counts and every failure verbatim:\n${c}`,
      { label: 'test', phase: 'Verify', agentType: 'test-runner', model: 'haiku', schema: SUITE })
    tests.push({ command: c, out })
  }
  const execOk = execs.every(e => e.out && e.out.launched && e.out.exit_code === 0 && e.out.stamped)
  const testsOk = tests.every(t => t.out && t.out.launched && t.out.exit_code === 0)
  if (!execOk || !testsOk) status = 'failed'
  ledger.items.verify = {
    status: execOk && testsOk ? 'done' : 'failed',
    stage: 'verify',
    detail: execs.map(e => `${e.notebook}: ${e.out ? e.out.exit_code : 'none'}`).concat(tests.map(t => `${t.command}: ${t.out ? t.out.exit_code : 'none'}`)).join('; '),
  }
  await persist('verify', status, [{ path: `${EV}verify.json`, content: JSON.stringify({ execs, tests }, null, 2) }])
} else if (status !== 'escalated') {
  await persist('stop', status)
}

return {
  ok: status === 'ok',
  status,
  run_id: ledger.run_id,
  ledger: `.claude/workflows/runs/${ledger.run_id}.json`,
  evidence: EV,
  build,
  rounds: ledger.rounds,
  open: last ? last.open : [],
  record: last && last.check ? last.check.record : '',
  verdict: last && last.check ? last.check.verdict : '',
  execs,
  tests,
}
