# Devpost submission copy

> Paste each section into the matching Devpost field.
> **`⟨FILL⟩` markers are numbers from the final run — replace them before submitting.**
> Do not invent them.

---

## Project name

**GOAT-LeBron — an ML research agent that diagnoses before it prescribes**

## Elevator pitch (one line)

We didn't build a recommender. We built the machine that builds recommenders — and it
has to explain, in writing, why each experiment was worth running.

---

## Inspiration

An ML engineer's day is a loop: read the metrics, guess what's wrong, write code,
train, look at the score, decide what to try next. Most agents that automate this
loop skip the first two steps — they mutate code and keep whatever makes the number
go up. That works, and it teaches you nothing.

The failure mode we cared about is subtler than "the score didn't improve". It's
**the score improved, but not for the reason you think**:

> The doctor says the kid is bad at maths → you hire a maths tutor → the term score
> goes up 3 points → but maths is unchanged; the gain came from an easy literature
> paper. Record that as "tutoring works" and you will keep buying maths tutoring.

An agent that cannot tell those two apart accumulates confident nonsense. So we made
the agent state a hypothesis *before* each experiment, and made **code**, not a
prompt, check afterwards whether the specific number it pointed at actually moved.

---

## What it does

You give it a dataset and a scoring rule. Then nobody touches the keyboard.

Each round runs four LLM roles with plain code between them:

| | Role | What it does |
|---|---|---|
| ① | **Doctor** | Reads the scorecard, names what is wrong, ranks by severity |
| | *card matcher* | *pure code — intersects symptoms with the method-card library* |
| ② | **Strategist** | Picks 3 candidate remedies, each with a written causal chain |
| | *scheduler* | *pure code — picks 1 by cost/benefit, chooses the data fidelity* |
| ③ | **Implementer** | Writes the actual code — a config change, or a new module |
| | *training* | *real training, real scoring, new scorecard* |
| ④ | **Reflector** | Judges whether the hypothesis held, updates the card's trust score |

The three italic steps call no model. The LLM wakes up at four decision points only —
which is how the token budget stays small.

Rounds are chained: this round's scorecard is the next round's input; failed cards are
blacklisted; already-applied cards are never proposed twice. The run stops on
convergence (no gain above ε for N rounds), budget exhaustion, or "nothing left to
diagnose at full data".

**Three things make it different from a code-mutation search loop:**

1. **Diagnose, then prescribe.** A 12-entry symptom vocabulary is the shared language
   between the doctor and a 14-card method library. The doctor physically cannot emit
   a symptom outside the vocabulary — it is enum-locked in the output schema. Card
   matching is therefore a set intersection, not another paid LLM call.

2. **It cannot lie to itself.** The anti-self-deception rules are validators in
   `agent/roles.py`, not sentences in a prompt. A prompt is a sign on the wall; a
   validator is the wall. Rejected output is not discarded — the exact violation is
   fed back and the role is asked again.

3. **It knows its own measurement error.** "Cold-start bucket is 0.03 below the hot
   bucket" is only a finding if 0.03 is bigger than how much that number wobbles on
   its own. We measure the wobble (same config, different seeds) and cross-check it
   against the Hanley–McNeil analytic standard error. Anything smaller than the band
   is noise and may not be reported as a symptom or claimed as a gain.

---

## How we addresses the problem statement

The track asks for an agent that autonomously improves a recommender on KuaiRand-Pure
and, crucially, that is judged on *what it chose to try and why* rather than on the
implementation alone.

- **Autonomous iteration** — `run_session` is a closed outer loop: diagnosis →
  proposal → implementation → training → reflection → next round's diagnosis, with
  cross-round state (card trust ledger, training-time ledger, shelved proposals,
  blacklist) persisted every round.
- **Reasoning that is auditable** — every round records the hypothesis, the causal
  chain from evidence to root cause to remedy, the full code diff, the resulting
  GAUC/nDCG@5, and the verdict. That log *is* the deliverable.
- **Beyond hyperparameter tuning** — the agent writes real modules
  (`FeatureOp` / `ModelOp` / `TrainOp`) into `modules/`, which are loaded into a deep
  training path (embedding tables, epoch loop, per-epoch validation, best-weight
  rollback). It is not restricted to turning knobs.
- **Robustness as a scored property** — each of the five stages has its own fuse, so
  a bad round is wasted, never the session. Every stumble and recovery is logged.
- **Honest accounting** — token totals, wall-clock, GPU-hours, iteration count and
  manual-intervention count are emitted by the system, not typed in by hand.

---

## How we built it

```
CLAUDE.md               12 hard rules (leakage, train-only statistics, holdout,
                        write-scope, no magic numbers) — read before any code
knowledge/
  symptoms.yaml         12 symptoms — the shared vocabulary
  cards/                14 method cards (treats / why / how / failure signals / source)
agent/
  roles.py              the four roles + the validators that make self-deception fail
  loop.py               one round, one session, three ledgers, the shelf
  schemas.py            structured-output schemas; symptom names enum-locked
  noise.py              noise bands: empirical (multi-seed) + Hanley–McNeil
  offline.py            fake model + fake executor — rehearse a whole session for $0
harness/                training path: feature/model/train op loading, deep loop
kuairand_goat_bridge/   KuaiRand adapter: official data split, official evaluator,
                        subprocess sandbox with a hard wall-clock timeout,
                        per-round diagnostics fed back into the scorecard
```

Two design decisions did most of the work:

**The scorecard carries evidence, not just a score.** Six of the twelve symptoms are
judged on *grouped* numbers, not on the total: overfitting needs a train-set score,
cold-start needs per-exposure-bucket scores, new-user needs seen/unseen split, and so
on. When the scorecard held only three validation numbers, the doctor correctly and
repeatedly answered "I cannot tell" — and the strategist and implementer were never
even invoked. Adding the grouped-evidence blocks is what turned the loop on.

**The official baseline is deliberately withheld from the agent.** The competition
ranks by delta over the official baseline — that is the judges' ruler, not the agent's
input. Given the number, the agent degenerates into tuning against a constant: above
it, "no findings"; below it, a vague "underfitting". It stops looking at the
train/validation gap, the buckets, the user composition — the evidence that actually
localises a cause. The baseline is recorded end-to-end in `final_summary.json` for the
humans; a validator asserts none of those figures can appear in what the agent reads.

---

## Development tools

- VS Code
- Claude Code (Anthropic) — used as a coding assistant during development
- Git / GitHub
- pytest

## APIs used

- **DeepSeek API** — the default LLM provider for the four agent roles
- **Anthropic Claude API** — alternative provider (both are supported behind one
  interface with structured-output enforcement and token accounting)

No other external APIs. The agent has no internet access at run time.

## Libraries and frameworks

| | |
|---|---|
| PyTorch | the deep training path (embedding tables, epoch loop) |
| NumPy | the official baseline FM and all metric computation |
| pandas / PyArrow | data handling |
| PyYAML | configs, symptom vocabulary, method cards |
| pytest | ⟨FILL: test count⟩ offline tests |
| `multiprocessing` (spawn) | cross-platform training sandbox with a hard timeout |

The official KuaiRand starter kit (`data.py`, `evaluate.py`, `submit.py`) is vendored
**unmodified** and called directly — we never reimplement the metric, because a
metric that is "almost the official one" only reveals itself at submission time.

## Datasets and assets

- **KuaiRand-Pure** (required benchmark). Label `long_view`; within-user ranking;
  GAUC and nDCG@5, primary = their mean. Official date split: train 2022-04-08→04-21,
  validation 04-22→04-28, test 04-29→05-08.
- **Official baseline** from the starter kit: a factorization machine (k=16, lr=0.001,
  5 categorical fields, NumPy only). Validation GAUC 0.6674 / nDCG@5 0.5357 /
  primary 0.6016.
- No external or manually labelled data.

---

## Challenges we ran into

**A doctor with no evidence is a doctor who says "no finding" forever.** Our first
real run produced `no_finding` every single round and converged having written zero
lines of code. The doctor was right: eleven of twelve symptom rules ask for grouped
numbers that the scorecard simply did not contain. We had migrated the symptom
vocabulary to the new dataset and forgotten to migrate the evidence with it.

**One wrong constant inverted a conclusion, silently.** The symptom table hard-coded
"official baseline GAUC 0.6016". That figure is the *primary*, not the GAUC — the real
GAUC baseline is 0.6674. Our model scored 0.6638: below baseline on all three metrics.
The doctor, reading our wrong number, computed "clearly above baseline" and reported
nothing. Nothing crashed. That single number is why we now refuse to hard-code any
baseline anywhere, and why the agent no longer sees one at all.

**Bugs that only fire on the last step.** Three separate defects in selecting and
submitting the best round would each have thrown *after* the full training session —
hours of compute, no submission file. A wrong nesting level (`KeyError`), a missing
argument (`TypeError`), and `if self._selected_round` treating **round 0 as falsy**,
which silently submitted the last round instead of the chosen one. The existing test
could not catch the third: its history had only one round, so "round 0" and "the last
round" were the same round.

**A one-line config change turned the agent back into a knob-turner.** Switching the
trainer to the official FM (a legitimate step for baseline reproduction) also disabled
code writing — that trainer rejects new files by design. Five consecutive rounds were
burned while the strategist worked this out for itself, each costing a real training
run and four LLM calls. Nothing errored; the runs just quietly stopped producing code.

The lesson underneath all four is the same, and it shaped the codebase: **the
dangerous failures are the silent ones.** So the constraints live in validators and
tests, not in comments — including a test that fails if the production config points
at a trainer that cannot write code.

---

## Accomplishments that we're proud of

- **The anti-self-deception layer is enforced, not requested.** Six distinct ways for
  a reflection to be internally inconsistent are rejected by code, with the violation
  quoted back for a retry.
- **The agent is told its own measurement error** and is forbidden from calling
  anything smaller than it a finding — including the honest case where the empirical
  band measures 0.0000 because the sampling scheme cannot perturb that metric, and we
  fall back to the analytic band rather than accept a threshold of zero.
- **The doctor's negative results are as well-argued as the positive ones.** In our
  run it ruled out cold-start ("lowest bucket GAUC 0.6667 is *above* the hottest
  bucket's 0.655"), new-user ("unseen 0.6877 *above* seen 0.6668") and drift ("no
  monotone decline; the last two days rise") — and flagged that the smallest bucket
  had 166 positives and was therefore unusable as evidence.
- **A full offline rehearsal mode** — fake model, fake executor, injectable failures —
  so the whole loop, including recovery paths, is testable with no network and no cost.

---

## What we learned

- **Prompts are signs on the wall; validators are the wall.** Every constraint we
  wrote as an instruction was eventually violated. Every constraint we wrote as a
  validator held.
- **"I cannot tell" is a feature.** Forcing the doctor to say so, instead of
  manufacturing a finding, is what made its findings worth reading — and it pointed
  straight at the missing evidence.
- **Withholding information can make an agent smarter.** Removing the baseline was
  the single change that most improved diagnosis quality.
- **A metric that can only ever be zero is not a measurement.** The manual-intervention
  count is scored, so "0" is only meaningful if non-zero was reachable. Ours wasn't —
  the command wrote to one file and the run read another — until we fixed it.

---

## What's next

- Move symptom judgement from the LLM into deterministic code, leaving the doctor to
  rank, weigh confidence, and spot what the rules do not cover.
- Rewrite the noise-band measurement for a within-user ranking metric — the current
  implementation is still shaped for the previous dataset's click/conversion funnel,
  so thresholds currently fall back to a fixed floor.
- Automatic detection of human intervention (config or module changes not attributable
  to the agent's own patch), so the autonomy number does not depend on self-reporting.
- The bonus benchmarks (KuaiRand-1k / 27k).

---

## Built With

`python` · `pytorch` · `numpy` · `pandas` · `pyyaml` · `pytest` · `deepseek-api` ·
`anthropic-claude-api` · `kuairand-pure`

---

## Team member contributions

⟨FILL: confirm names and split — the rows below are inferred from git history⟩

| Member | Contribution |
|---|---|
| Wang Jingjie | Method library and modelling: baseline reproduction, real executor, method cards, role prompts, web console |
| 许叔尧 (David Xu) | Agent core: the four roles and their validators, the outer loop, the three ledgers, noise bands, deliverable packaging |
| HYF | Dataset bridges: KuaiRand and AliCCP adapters, official starter-kit integration, preflight checks |
| Stephen Zhu | Bridge and modules: official FM trainer integration, baseline reproduction config, training diagnostics |

---

## Results

⟨FILL — from `deliverables/session_summary.json` after the final run.
Leave this section out entirely rather than guessing.⟩

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| **Ours (validation-best)** | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |
| **Absolute delta** | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |

Resource usage to reach convergence: ⟨FILL⟩ LLM tokens (input + output),
⟨FILL⟩ agent wall-clock, ⟨FILL⟩ of 50 iterations used, ⟨FILL⟩ GPU-hours,
⟨FILL⟩ manual interventions.
