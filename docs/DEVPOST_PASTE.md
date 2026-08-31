# Devpost — paste-ready, field by field

Deadline-mode version. Each block below maps to one field in the Devpost form.
`⟨FILL⟩` = a number from the final run. **Never guess one.**

---

## STEP 2 · Project overview

### Project name
```
GOAT-LeBron
```

### Elevator pitch  (200 char limit — this one is 192)
```
An ML research agent that diagnoses before it prescribes: it names what's wrong with the model, argues for a fix, writes the code — then code, not a prompt, checks whether the hypothesis held.
```

---

## STEP 3 · Project details → "About the project"

```
## Inspiration

An ML engineer's day is a loop: read the metrics, guess what's wrong, write code,
train, look at the score, decide what to try next. Most agents that automate this loop
skip the first two steps — they mutate code and keep whatever makes the number go up.
That works, and it teaches you nothing.

The failure mode we cared about is quieter than "the score didn't improve". It is
**the score improved, but not for the reason you think**:

> The doctor says the kid is bad at maths → you hire a maths tutor → the term score
> goes up 3 points → but maths is unchanged; the gain came from an easy literature
> paper. Record that as "tutoring works" and you will keep buying maths tutoring.

An agent that cannot separate those two cases accumulates confident nonsense. So our
agent must state a hypothesis *before* each experiment, and afterwards **code** — not
a prompt — checks whether the specific number it pointed at actually moved.

## What it does

You give it a dataset and a scoring rule. Then nobody touches the keyboard.

Each round runs four LLM roles with plain code between them:

| Role | What it does |
|---|---|
| **Doctor** | Reads the scorecard, names what is wrong, ranks by severity |
| *card matcher* | *pure code — intersects symptoms with the method-card library* |
| **Strategist** | Picks 3 candidate remedies, each with a written causal chain |
| *scheduler* | *pure code — picks 1 by cost/benefit, chooses the data fidelity* |
| **Implementer** | Writes the actual code — a config change, or a new module |
| *training* | *real training, real scoring, new scorecard* |
| **Reflector** | Judges whether the hypothesis held; updates that card's trust score |

The italic steps call no model. The LLM wakes at four decision points only — which is
how the token budget stays small, and token usage is scored.

Rounds are chained: this round's scorecard is the next doctor's input, failed cards are
blacklisted, applied cards are never re-proposed, and unchosen proposals are shelved
and re-offered while still relevant. The run stops on convergence (no gain above
ε=0.002 for 3 rounds), budget exhaustion, or nothing left to diagnose at full data.

## Three things that make it more than a code-mutation search

**1. Diagnose, then prescribe.** A 12-entry symptom vocabulary is the shared language
between the doctor and a 14-card method library. The doctor's output schema enum-locks
it, so it *physically cannot* name a symptom no card treats. Card matching is therefore
a set intersection — free, deterministic, and it never hallucinates a match.

**2. It cannot lie to itself.** The anti-self-deception rules are validators, not
sentences in a prompt. A reflection is rejected if it claims success while admitting no
target symptom improved; if it claims a symptom is resolved while its own before/after
are identical; if the change is smaller than the seed-to-seed noise; or if it fails to
account for every symptom the proposal claimed to treat. Rejected output isn't
discarded — the exact violation is quoted back and the role is asked again.

*Prompts are signs on the wall. Validators are the wall.* Every constraint we wrote as
an instruction was eventually violated. Every constraint we wrote as a validator held.

**3. It knows its own measurement error.** "The cold-start bucket is 0.03 below the hot
bucket" is only a finding if 0.03 exceeds how much that number wobbles by itself. We
measure the wobble across seeds and cross-check it against the Hanley–McNeil analytic
standard error. Anything under the band may not be reported as a symptom, nor claimed
as a gain.

## How it addresses the problem statement

- **Autonomous iteration** — a closed outer loop with cross-round state persisted every
  round: a card-trust ledger, a training-time ledger, a proposal shelf, a blacklist.
- **Auditable reasoning** — every round records the hypothesis, the causal chain from
  evidence to root cause to remedy, the full code diff, the resulting GAUC/nDCG@5, and
  the verdict. That log *is* the deliverable.
- **Beyond hyperparameter tuning** — the agent writes real modules (FeatureOp /
  ModelOp / TrainOp) into `modules/`, loaded into a deep training path with embedding
  tables, an epoch loop, per-epoch validation and best-weight rollback.
- **Robustness as a scored property** — each of five stages has its own fuse, so a bad
  round is wasted, never the session. Training runs in a spawned subprocess with a hard
  wall-clock timeout. Every stumble and recovery is logged.
- **Honest accounting** — tokens, wall-clock, GPU-hours, iteration count and manual
  interventions are emitted by the system, not typed in by hand.

## How we built it

Two decisions did most of the work.

**The scorecard carries evidence, not just a score.** Six of the twelve symptoms are
judged on *grouped* numbers: overfitting needs a train-set score, cold-start needs
per-exposure-bucket scores, new-user needs a seen/unseen split. When the scorecard held
only three validation numbers, the doctor correctly and repeatedly answered "I cannot
tell" — and the strategist and implementer were never even invoked. Adding the grouped
evidence is what turned the loop on.

**The official baseline is deliberately withheld from the agent.** The competition ranks
by delta over that baseline — the judges' ruler, not the agent's input. Given the
number, the agent degenerates into tuning against a constant: above it, "no findings";
below it, a vague "underfitting". It stops reading the train/validation gap, the
buckets, the user composition — the evidence that actually localises a cause. The
baseline is recorded end-to-end for humans in `final_summary.json`; a test asserts none
of those figures can reach anything the agent reads.

## Development tools

VS Code · Claude Code (Anthropic) as a coding assistant · Git/GitHub · pytest

## APIs used

**DeepSeek API** (default provider for the four agent roles) and **Anthropic Claude
API** (alternative) — both behind one interface with structured-output enforcement and
token accounting. No other external APIs; the agent has no internet access at run time.

## Libraries and frameworks

PyTorch (deep training path) · NumPy (baseline FM and all metric computation) ·
pandas / PyArrow · PyYAML (configs, symptom vocabulary, method cards) · pytest ·
`multiprocessing` spawn (cross-platform training sandbox with a hard timeout)

The official KuaiRand starter kit (`data.py`, `evaluate.py`, `submit.py`) is vendored
**unmodified** and called directly. We never reimplement the metric — a metric that is
"almost the official one" only reveals itself at submission time.

## Datasets and assets

**KuaiRand-Pure** (required benchmark). Label `long_view`; within-user ranking; GAUC and
nDCG@5, primary = their mean. Official date split: train 2022-04-08→04-21, validation
04-22→04-28, test 04-29→05-08 (verified 1,141,112 / 124,909 / 170,588 rows).

**Official baseline** from the starter kit: a factorization machine (k=16, lr=0.001,
five categorical fields, NumPy only) — validation GAUC 0.6674 / nDCG@5 0.5357 /
primary 0.6016. No external or manually labelled data.

## Challenges we ran into

**A doctor with no evidence says "no finding" forever.** Our first real run produced
`no_finding` every round and converged having written zero lines of code. The doctor was
right — eleven of twelve symptom rules ask for grouped numbers the scorecard did not
contain. We had migrated the symptom vocabulary to the new dataset and forgotten to
migrate the evidence with it.

**One wrong constant inverted a conclusion, silently.** The symptom table hard-coded
"official baseline GAUC 0.6016". That figure is the *primary*; the real GAUC baseline is
0.6674. Our model scored 0.6638 — below baseline on all three metrics — and the doctor,
reading our wrong number, concluded "clearly above baseline" and reported nothing.
Nothing crashed. That one number is why the agent no longer sees a baseline at all.

**Bugs that only fire on the last step.** Three separate defects in selecting and
submitting the best round would each have thrown *after* a full training session — hours
of compute, no submission file. A wrong nesting level (KeyError), a missing argument
(TypeError), and `if self._selected_round` treating **round 0 as falsy**, silently
submitting the last round instead of the chosen one. The existing test structurally
could not catch the third: its history had one round, so "round 0" and "the last round"
were the same round.

**A one-line config change turned the agent back into a knob-turner.** Switching the
trainer to the official FM — legitimate for baseline reproduction — also disabled code
writing, because that trainer rejects new files by design. Five consecutive rounds were
burned while the strategist worked this out for itself, each costing a real training run
and four LLM calls. Nothing errored; the runs just quietly stopped producing code.

All four say the same thing, and it shaped the codebase: **the dangerous failures are
the silent ones.** So constraints live in validators and tests, not comments —
including a test that fails if the production config points at a trainer that cannot
write code.

## Accomplishments that we're proud of

- Six distinct ways for a reflection to be internally inconsistent are rejected **by
  code**, with the violation quoted back for a retry.
- The agent is told its own measurement error and forbidden from calling anything
  smaller a finding — including the honest case where the empirical band measures
  0.0000 because the sampling scheme cannot perturb that metric, and we fall back to the
  analytic band rather than accept a threshold of zero.
- **Its negative results are as well-argued as its positive ones.** In our run it ruled
  out cold-start ("lowest bucket GAUC 0.6667 is *above* the hottest bucket's 0.655"),
  new-user ("unseen 0.6877 *above* seen 0.6668") and drift ("no monotone decline; the
  last two days rise") — and flagged that the smallest bucket had 166 positives and was
  therefore unusable as evidence.
- A full offline rehearsal mode — fake model, fake executor, injectable failures — so
  the whole loop including recovery paths is testable with no network and no cost.

## What we learned

- **"I cannot tell" is a feature.** Forcing the doctor to say so, instead of
  manufacturing a finding, is what made its findings worth reading — and it pointed
  straight at the missing evidence.
- **Withholding information can make an agent smarter.** Removing the baseline was the
  single change that most improved diagnosis quality.
- **A metric that can only ever be zero is not a measurement.** The manual-intervention
  count is scored, so "0" only means something if non-zero was reachable. Ours wasn't —
  the command wrote to one file and the run read another — until we fixed it.

## What's next

Move symptom judgement from the LLM into deterministic code, leaving the doctor to rank,
weigh confidence and spot what the rules miss. Rewrite noise-band measurement for a
within-user ranking metric. Detect human intervention automatically instead of relying
on self-reporting. Attempt the bonus benchmarks (KuaiRand-1k / 27k).

## Results

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| **Ours (validation-best)** | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |
| **Absolute delta** | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |

Resource usage to convergence: ⟨FILL⟩ LLM tokens (input + output), ⟨FILL⟩ agent
wall-clock, ⟨FILL⟩ of 50 iterations, ⟨FILL⟩ GPU-hours, ⟨FILL⟩ manual interventions.
```

### Built With  (tags)
```
python  pytorch  numpy  pandas  pyyaml  pytest  deepseek  anthropic-claude  kuairand
```

### Try it out (links)
```
https://github.com/davidxu277/GOAT-LeBron
```

---

## STEP 4 · Additional info

**Team member contributions** — ⟨FILL: confirm, inferred from git history⟩

```
Wang Jingjie — Method library and modelling: baseline reproduction, the real executor,
method cards, role prompts, web console, noise-band methodology.

David Xu (许叔尧) — Agent core: the four roles and their validators, the outer loop,
the three ledgers, the proposal shelf, scorecard diagnostics, deliverable packaging.

HYF — Dataset bridges: KuaiRand and AliCCP adapters, official starter-kit integration,
preflight and data checks.

Stephen Zhu — Bridge and modules: official FM trainer integration, baseline
reproduction config, per-round training diagnostics, fidelity sampling.
```

---

## Before you hit Submit

- [ ] Every `⟨FILL⟩` replaced with a real number from `deliverables/session_summary.json`
- [ ] **Repository is public**
- [ ] Deliverable #3 attached — per-iteration log + intervention count
- [ ] Deliverable #4 attached — the submission file itself, not the score
- [ ] If the run did not finish in time: **delete the Results section** rather than
      guessing. An unverifiable number is worse than an absent one.
