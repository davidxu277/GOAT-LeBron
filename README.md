# GOAT-LeBron

**TikTok TechJam · Track 2 — An autonomous ML research agent for recommender systems**

> We didn't build a recommender. We built the machine that builds recommenders — and
> it has to explain, in writing, why each experiment was worth running.

*[中文版 README](README.zh-CN.md)*

---

## 1. Project overview

An ML engineer's day is a loop:

> read the metrics → guess what's wrong → write code → train → look at the score →
> decide what to try next → repeat until the score stops moving

**This project automates that loop, including the guessing.** You hand it a dataset
and a scoring rule; then nobody touches the keyboard. It reads its own scorecard,
names what is wrong, picks a remedy and argues for it, writes the code, trains,
and judges whether its own hypothesis held.

It produces two things: a trained ranking model, and a complete record of what it was
thinking at every step.

### Why not just mutate code and keep what scores higher?

Because of a failure mode that is quieter than "the score didn't improve":

> The doctor says the kid is bad at maths → you hire a maths tutor → the term score
> goes up 3 points → **but maths is unchanged**; the gain came from an easy
> literature paper. Record that as "tutoring works" and you will keep buying maths
> tutoring — and keep getting further from the actual problem.

An agent that cannot separate those two cases accumulates confident nonsense. So this
agent must state a hypothesis *before* each experiment, and afterwards **code** —
not a prompt — checks whether the specific number it pointed at actually moved.

---

## 2. How it works

### One session

```
  Human provides two things: the data + the scoring rule
      ↓
  Round 0 · run as-is, obtain the first scorecard
      ↓
  ┌─→ One round · diagnose → prescribe → implement → train → reflect
  │       ↓
  │   new scorecard
  │       ↓
  │   anything left to improve?
  │       │
  └───────┤ yes — the new scorecard becomes the next doctor's input
          │
          │ no — stop on any of:
          │        · no gain above ε for N consecutive rounds
          │        · token budget exhausted
          │        · nothing diagnosable even at full data
          ↓
  Select the round with the best validation score → that recipe is what we submit
```

Rounds are chained. Failed cards are blacklisted; already-applied cards are never
proposed again; unchosen proposals are shelved and re-offered while still relevant.

### One round: four roles in relay

```
  scorecard →  ① Doctor  →  match cards  →  ② Strategist →  scheduler
                symptoms      set intersect     3 remedies     pick 1
                + severity    (no LLM)          + reasoning    (no LLM)
                                                                   ↓
  next round ← ④ Reflector ←   train    ←   ③ Implementer ←────────┘
                did it hold?   scorecard      writes code
```

| | Role | What it does |
|---|---|---|
| ① | **Doctor** | Reads the scorecard, names the symptoms, ranks them by severity |
| | *card matcher* | *pure code — intersects symptoms with the card library* |
| ② | **Strategist** | Picks 3 remedies; each states which symptom, expected gain, cost, failure signals |
| | *scheduler* | *pure code — picks 1 by cost/benefit and chooses the data fidelity* |
| ③ | **Implementer** | Turns it into code — a config change, or a new module under `modules/` |
| | *training* | *real training, real scoring, new scorecard* |
| ④ | **Reflector** | Judges whether the hypothesis held; updates the card's trust score |

**The three italic steps call no model.** The LLM is woken at four decision points
only. That is what keeps the token budget small — and token usage is scored.

---

## 3. What makes it different

### 3.1 Diagnose, then prescribe

Code-mutation agents search blindly: change something, see if the number moved. They
never form a view of *what is actually wrong*.

```
Evidence   train primary 0.6909 vs validation 0.6015 — a gap of 0.0894
             ↓
Diagnosis  "memorising the training set" (severity 0.6, confidence high)
             ↓
Remedy     from the cards that treat it → e.g. user-level normalisation
           Reason: the model is fitting per-user rating scale rather than
           within-user preference order, which is what GAUC actually measures
```

The link between doctor and cards is a **12-entry symptom vocabulary**
(`knowledge/symptoms.yaml`). The doctor's output schema enum-locks it, so the doctor
*physically cannot* emit a symptom that no card treats. Card matching is therefore a
set intersection — free, deterministic, and it never hallucinates a match.

### 3.2 It cannot lie to itself

These are validators in `agent/roles.py`, not sentences in a prompt:

```python
# claims the hypothesis held, while admitting no target symptom improved
if all_targets_unresolved and verdict == "correct":        reject

# subtler: claims a symptom is resolved, but its own before/after are identical
if resolved in ("yes", "partly") and before == after:      reject

# the change is smaller than the seed-to-seed wobble
if max_change < noise_band and verdict == "correct":       reject

# a proposal claiming to treat 3 symptoms must account for all 3
if any target symptom is unaccounted for:                  reject
```

Rejection is not discarding — the exact violation is quoted back and the role is
asked again.

> **Prompts are signs on the wall. Validators are the wall.** Every constraint we
> wrote as an instruction was eventually violated; every constraint we wrote as a
> validator held.

### 3.3 It knows its own measurement error

Is "the cold bucket is 0.03 below the hot bucket" a finding? Only if 0.03 is larger
than how much that number wobbles by itself.

We measure it: same config, different random seeds, N runs — and cross-check against
the Hanley–McNeil analytic standard error computed from the positive/negative counts
(`agent/noise.py`). Anything below the band may not be reported as a symptom, and may
not be claimed as a gain.

The band is also **per-metric**. Two metrics whose wobble differs by an order of
magnitude cannot share one threshold: a real gain in the stable one gets drowned by
the noisy one's band, while the noisy one's pure jitter clears the shared threshold
and earns undeserved trust.

### 3.4 The official baseline is deliberately withheld from the agent

The competition ranks by delta over the official baseline. **That is the judges'
ruler, not the agent's input.**

Given the number, the agent degenerates into tuning against a constant: above it,
"no findings"; below it, a vague "underfitting". It stops reading the train/validation
gap, the buckets, the user composition — the evidence that actually localises a cause.

We learned this the expensive way. The symptom table once hard-coded
"official baseline GAUC 0.6016". That figure is the **primary**, not the GAUC; the
real GAUC baseline is 0.6674. Our model scored 0.6638 — *below* baseline on all three
metrics — and the doctor, reading our wrong constant, concluded "clearly above
baseline" and reported nothing. Nothing crashed. A wrong constant inverted a
conclusion in perfect silence.

The baseline is still recorded end to end, in `final_summary.json`, for humans. A test
asserts that none of those figures can appear in anything the agent reads.

---

## 4. Staying alive

Robustness is scored, so it is engineered rather than hoped for.

| Failure | Response |
|---|---|
| A role misbehaves (bad format, network drop) | Each of the five stages has its own fuse — **the round is wasted, never the session** |
| The implementer's code fails validation | Retry with the error fed back; then fall through to backup proposals |
| Training crashes or times out | The reflection is synthesised **in code** — no LLM call wasted on a run with no result |
| The same idea keeps coming back | Tried cards blacklisted; applied cards never re-proposed |
| Several rounds with nothing diagnosable | Escalate one data fidelity; if already at full data, declare convergence |
| Process dies mid-session | Logs and all three ledgers are flushed every round; restart resumes |

Training runs in a spawned subprocess with a hard wall-clock timeout (`terminate`,
escalating to `kill`), so a hung trainer cannot consume the session budget. Every
stumble and recovery is written to the log — that is the evidence of robustness.

---

## 5. Task and scoring

**Dataset** — KuaiRand-Pure. Label `long_view`. Within-user ranking over logged
impressions; no full-corpus retrieval.

| | |
|---|---|
| Metrics | **GAUC** and **nDCG@5**; primary = their mean |
| GAUC scope | users with `0 < positives < impressions`, weighted by positives |
| nDCG scope | all users; zero-positive users score 0 |
| Split | train 2022-04-08→04-21 · validation 04-22→04-28 · test 04-29→05-08 |
| Convergence | ε = 0.002 on primary, patience = 3 |
| Budget | ≤ 50 training attempts, ≤ 6 h wall-clock |

**Official baseline** (starter kit): a factorization machine, k=16, lr=0.001, five
categorical fields, NumPy only.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Validation | 0.6674 | 0.5357 | 0.6016 |
| Hidden test | 0.6610 | 0.5282 | 0.5946 |

Ranking uses the absolute delta over that baseline, taken from the **validation-best**
round. The scoring also weighs the agent itself — the quality of its reasoning, how
few times a human intervened, and the compute and tokens it consumed. **The log is
part of the score.**

---

## 6. Repository map

```
CLAUDE.md / AGENTS.md       12 hard rules — leakage, train-only statistics, holdout
                            discipline, write scope, no magic numbers
agent/                      the agent core
  roles.py                    four roles + the validators (the anti-self-deception wall)
  loop.py                     one round, one session, three ledgers, the shelf
  knowledge.py                loads the vocabulary and cards; matches by symptom
  schemas.py                  structured-output schemas; symptom names enum-locked
  noise.py                    noise bands: multi-seed empirical + Hanley–McNeil
  offline.py                  fake model + fake executor — rehearse a session for $0
  prompts/                    the four role prompts
knowledge/
  symptoms.yaml               12 symptoms — the doctor↔card vocabulary
  cards/                      14 method cards
harness/                    training path: op loading, deep loop, R2/R5 guards
modules/                    replaceable parts — the ONLY place the agent may write
config/pipeline.yaml        pipeline config — the other thing the agent may change
kuairand_goat_bridge/       KuaiRand adapter
  official_starter_kit/       vendored, unmodified: data.py, evaluate.py, submit.py
  src/kuairand_bridge/        dataset views, runner, evaluator, diagnostics,
                              subprocess sandbox, session entry point
  examples/goat_trainer.py    research trainer — the agent CAN write modules
  examples/official_fm_trainer.py  baseline trainer — config-only, by design
  configs/kuairand_task.yaml  ← the real run
  configs/fm_baseline.yaml    ← baseline reproduction only
logs/ · deliverables/       per-round logs, ledgers, snapshots; the submission bundle
```

The starter kit is **vendored unmodified and called directly**. We never reimplement
the metric — a metric that is "almost the official one" only reveals itself at
submission time.

---

## 7. Setup and installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/davidxu277/GOAT-LeBron.git
cd GOAT-LeBron

python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
.venv/bin/pip install -e kuairand_goat_bridge      # makes `python -m kuairand_bridge` work
```

Without the editable install, prefix bridge commands with
`PYTHONPATH=kuairand_goat_bridge/src`.

Download **KuaiRand-Pure** and point the config at the directory containing the CSVs:

```yaml
# kuairand_goat_bridge/configs/kuairand_task.yaml
data_dir: /absolute/path/to/KuaiRand-Pure/data
```

Verify the data before spending anything:

```bash
.venv/bin/python -m kuairand_bridge preflight --data-dir /path/to/KuaiRand-Pure/data
# verified: train 1,141,112 · valid 124,909 · test 170,588
# `official_row_counts_match: true` means the split matches the starter kit exactly
```

LLM credentials (the agent roles; not needed for the offline checks below):

```bash
export AGENT_PROVIDER=deepseek DEEPSEEK_API_KEY=...
# or: export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 8. Steps to reproduce our results

### Step 0 — free checks (no network, no cost)

```bash
.venv/bin/python -m agent.cli check                      # vocabulary ↔ cards consistent
.venv/bin/python -m pytest tests/ kuairand_goat_bridge/tests/ -q
.venv/bin/python -m agent.cli run --offline --rounds 8   # rehearse a whole session
```

The rehearsal uses a fake model and a fake executor to exercise the **wiring**: does
state carry across rounds, does a crashed role recover, does it stop when it should.
Failures can be injected on demand:

```bash
.venv/bin/python -m agent.cli run --offline --rounds 8 --fail-round 3 --fail-role-call 2
```

Rehearsal logs go to `logs/offline/` and never contaminate the deliverable logs.

### Step 1 — reproduce the official baseline (~30 s)

```bash
.venv/bin/python -m kuairand_bridge goat-run \
    --config kuairand_goat_bridge/configs/fm_baseline.yaml --dry-run
```

This config sets `require_baseline_reproduction: true`, so round 0 must land within
0.003 of the official primary or the run aborts. **Do this first** — it proves the
harness reproduces the official number, so any later gain is attributable to the agent
rather than to a harness discrepancy.

### Step 2 — noise band: **not available on this dataset yet**

`agent.cli noise` still requires the previous dataset's file layout
(`--train` / `--val-features`) and its analytic fallback assumes a plain AUC rather
than a within-user GAUC. It has never been run on KuaiRand.

Nothing needs doing here — but be aware of what it means: every "is this a real
improvement?" threshold falls back to a fixed floor of **0.0005**, which is a guess,
not a measurement of this data. The session announces this at startup rather than
pretending otherwise:

```
⚠️ Noise band never measured — using the R11 fallback floor 0.0005
   (a guessed number, not one measured on this data)
```

This is limitation 2, and it is the first thing we would fix with more time.

### Step 3 — the autonomous run (nobody touches the keyboard)

```bash
.venv/bin/python -m kuairand_bridge goat-run \
    --config kuairand_goat_bridge/configs/kuairand_task.yaml
```

> ⚠️ `configs/kuairand_task.yaml` must point at `examples/goat_trainer.py`. The other
> trainer only accepts six hyperparameters and rejects new files by design — pointing
> the real run at it silently reduces the agent to a knob-turner. `tests/test_configs.py`
> guards this.

If you intervene at any point, record it — the autonomy score depends on this number
being a real observation rather than a hard-coded zero:

```bash
.venv/bin/python -m agent.cli intervene "round 7 hit OOM, reduced batch size" --round 7
```

### Step 4 — package the deliverables

```bash
.venv/bin/python -m agent.cli finalize
```

| Output | Deliverable |
|---|---|
| `rounds.jsonl` | **#3** per-iteration log: hypothesis, full code diff, metrics, errors and recoveries |
| `narrative.md` | **#3** human-readable: the whole session as one storyline |
| `session_summary.json` | **#4** results table: best scores, delta over baseline, tokens, wall-clock, GPU-hours, interventions |
| `best_pipeline/` | the **recipe** of the best round (config + module code) |
| `dashboard.html` | round-by-round replay for reviewers |

> **Deliverable #4 is the model output, not the score.** The agent's edits are
> *cumulative*: by round 20 the disk holds only the final stack, and round 5's state
> was overwritten long ago. `best_pipeline/` is how the best round is reconstructed
> and re-run to produce the actual submission file.

### What counts as a manual intervention

Fixed before the run, not adjusted afterwards:

| Not an intervention (setup) | An intervention (touching a live run) |
|---|---|
| Downloading and preprocessing data | Changing config or code mid-run |
| Writing cards, prompts, the vocabulary | Killing a round, restarting the process |
| Choosing round count, budget, starting fidelity | Manually choosing which version to submit |
| Installing the environment; fixing our own bugs beforehand | Fixing a bug mid-run and continuing |

---

## 9. Limitations, and what we would improve with more time

These are problems we know about, stated before anyone had to point them out.

**1. Symptom judgement is done by the LLM, not by code.** The detection rules are
passed as text and the model does the arithmetic. This is not reproducible — the same
scorecard can yield different findings on two runs — and the arithmetic can be wrong.
*Improvement:* compute the rules deterministically in code and leave the doctor to
weigh confidence, rank severity, and notice what the rules do not cover. Diagnosis
should be deterministic; the innovation is in the strategist's reasoning, not in the
doctor's subtraction.

**2. Noise bands have never been measured on this dataset.** `agent/noise.py` is still
shaped for the previous dataset's click/conversion funnel — it reads fields KuaiRand
does not have, and its analytic fallback assumes a plain AUC rather than a within-user
GAUC. Until it is rewritten, every threshold falls back to a fixed floor of 0.0005,
which is a guess. The system says so out loud at startup rather than pretending
otherwise. *Improvement:* rewrite the measurement for within-user ranking metrics.

**3. The reflector's before/after numbers are still self-reported.** Code checks that
they are *mutually consistent* with the reflector's own verdict — claiming a symptom
is resolved while reporting identical before/after is rejected — but cannot verify
they were copied from the scorecard rather than invented. Fixing this properly
requires limitation 1.

**4. The train-set score uses the inference path.** The "memorising the training set"
diagnosis compares train and validation scores, but the train score is produced with
each module's `transform` rather than the out-of-fold path used during fitting. For
target-encoding-style modules this makes the train score slightly optimistic. The
scorecard carries an explicit caveat so the doctor discounts marginal gaps.

**5. One change per round.** This is deliberate: two changes in one round and a score
increase cannot be attributed, which would defeat the entire "did the symptom
actually improve" mechanism. Combinations are handled by *composite cards* — a single
card that packages changes which must ship together. The cost is slower coverage.

**6. Prompt quality is validated on very few samples.** The test suite covers the
*enforcement* layer thoroughly (does the evidence contain numbers, are forbidden
fields blocked, can the reflector deceive itself, does state carry across rounds).
Whether the prompts elicit good reasoning can only be judged by reading real runs, and
we have read few.

**7. Bonus benchmarks not attempted.** Only the required KuaiRand-Pure. Attempting
KuaiRand-1k and 27k in the available time would most likely have compromised both.

**8. Two AliCCP-era items remain unfixed** because they cannot affect this task: a
purchase-AUC field naming mismatch, and rows with `click=0, conversion=1` being warned
about rather than cleaned. Both live on the retired dataset path.

---

## 10. Team member contributions

| Member | Contribution |
|---|---|
| **Wang Jingjie** | Method library and modelling — baseline reproduction, the real executor, method cards, role prompts, web console, noise-band methodology |
| **许叔尧 (David Xu)** | Agent core — the four roles and their validators, the outer loop, the three ledgers, the shelf, scorecard diagnostics, deliverable packaging |
| **HYF** | Dataset bridges — KuaiRand and AliCCP adapters, official starter-kit integration, preflight and data checks |
| **Stephen Zhu** | Bridge and modules — official FM trainer integration, baseline reproduction config, per-round training diagnostics, fidelity sampling |
| *All* | Each member logs their own changes in `docs/开发日志.md`, one line per change |

---

## 11. Further reading

| File | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | The 12 hard rules, danger signals, pre-commit checklist |
| [docs/DEVPOST.md](docs/DEVPOST.md) | Submission copy |
| [docs/开发日志.md](docs/开发日志.md) | Development log — one line per change, newest first |
| [docs/四个角色接口.md](docs/四个角色接口.md) | The input/output contract of the four roles |
| [knowledge/symptoms.yaml](knowledge/symptoms.yaml) | The 12 symptoms and their detection rules |
| [knowledge/卡片格式.md](knowledge/卡片格式.md) | Method card format and an annotated example |
| [README.zh-CN.md](README.zh-CN.md) | Chinese README |
