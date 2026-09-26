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
  ┌─→ One round · diagnose → prescribe → implement → trial run → train → reflect
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
  Select the round with the best development-set score
      ↓
  Holdout exam · retrain round 0 and the best round once each; how much did
  each gain on users no decision ever looked at?
      ↓
  Retrain the best round on full data → that is what we submit
```

Rounds are chained. Failed cards are blacklisted; already-applied cards are never
proposed again; unchosen proposals are shelved and re-offered while still relevant.

### One round: four roles in relay

```
  scorecard →  ① Doctor  →  match cards  →  ② Strategist →  scheduler
                symptoms      set intersect     3 remedies     pick 1
                + severity    (no LLM)          + reasoning    (no LLM)
                                                                   ↓
  next round ← ④ Reflector ←   train    ←  trial run  ←  ③ Implementer
                did it hold?   scorecard     2% data,        writes code
                                             1 epoch            ↑
                                             crash ─traceback───┘
```

| | Role | What it does |
|---|---|---|
| ① | **Doctor** | Reads the scorecard, names the symptoms, ranks them by severity |
| | *card matcher* | *pure code — intersects symptoms with the card library* |
| ② | **Strategist** | Picks 3 remedies; each states which symptom, expected gain, cost, failure signals |
| | *scheduler* | *pure code — picks 1 by cost/benefit and chooses the data fidelity* |
| ③ | **Implementer** | Turns it into code — a config change, or a new module under `modules/` |
| | *trial run* | *pure code — 2% of the data, 1 epoch, no training attempt used; a crash sends the full traceback back to the implementer, at most 2 more tries per proposal* |
| | *training* | *real training, real scoring, new scorecard* |
| ④ | **Reflector** | Judges whether the hypothesis held; updates the card's trust score |

**The four italic steps call no model.** The LLM is woken at four decision points
only. That is what keeps the token budget small — and token usage is scored.

**You can watch it think.** Every step prints one plain line to the terminal as it
finishes: what the doctor found, what the strategist proposed, which attempt the
implementer is on, whether the trial run passed, the scores, the reflector's verdict.
Each round is also written out in full as markdown — every version of the
implementer's code (the broken ones included), full tracebacks, and time and tokens
per role. See section 8, step 3.

---

## 3. What makes it different

### 3.1 Diagnose, then prescribe

Code-mutation agents search blindly: change something, see if the number moved. They
never form a view of *what is actually wrong*.

```
Evidence   exposure bucket "<10 views": GAUC 0.611, vs 0.684 for ">1000 views"
           — a gap of 0.073, on 22% of impressions
             ↓
Diagnosis  "cold videos can't rank" (severity 0.7, confidence high)
             ↓
Remedy     from the cards that treat it → e.g. author fallback
           Reason: a video seen fewer than 10 times cannot learn its own
           embedding; borrowing its author's statistics targets exactly that
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

The design: same config, different random seeds, N runs — anything below that band may
not be reported as a symptom, and may not be claimed as a gain. A band measured at one
data fidelity is never silently reused at another; if it does not fit, the session
drops it and says so in the results table.

**Measured on this dataset** (2026-09-26). Same config, same data subset, only the
training seed changes, five runs; band = 2×√2×standard deviation, i.e. two runs of the
identical config land within the band about 95% of the time:

| Fidelity | GAUC | nDCG@5 | primary |
|---|---|---|---|
| small (15% of train) | 0.0054 | 0.0024 | 0.0037 |
| medium (40% of train) | 0.0042 | 0.0026 | 0.0031 |

(With the holdout on, every round is scored on 80% of users, which wobbles a little
more: 0.0065 / 0.0028 / 0.0045 on the small development set.)

The data subset is pinned while measuring and only the training seed varies — within a
session the data never changes between rounds, so that is the only wobble a round-to-
round comparison sees. Resampling the data too would count sampling noise and inflate
the threshold.

The numbers overturned three assumptions:

- The guessed fallback floor of 0.0005 was **5–10× too tight**. A +0.001 GAUC "win"
  earned a "hypothesis held" verdict and a trust bump — at the scale of a reseed.
- The official convergence threshold ε = 0.002 is **smaller than the primary's own
  wobble**; a single-seed comparison cannot resolve a gain that small.
- Going from 15% to 40% of the data narrowed the band by only 20%: the wobble comes
  mostly from initialisation, not from too little data. More seeds, not more data.

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

### 3.5 An exam it has never seen

The agent looks at the same validation set for N rounds and keeps the best one. At the
wobble measured above, picking the highest of 20 rounds finds several thousandths of
"improvement" from luck alone.

So the real run locks away 20% of validation users (`holdout_users: 0.2`). During the
session, diagnosis, per-round scoring and early stopping inside training all see only
the other 80% — the development set. After the session, round 0 and the best round are
each retrained once at the same fidelity and three numbers are reported: **the gain on
the development set, the gain on the holdout, and the difference — luck picked up by
selection.**

It compares **gains**, not scores. On real data, round 0 already differs by 0.018
between the two sides — the holdout users are simply harder to rank. Reporting "best
round's development score minus holdout score" would misread that 0.018 of population
difference as overfitting to the development set.

The submission is still scored on the **full** validation set, directly comparable to
the official baseline.

---

## 4. Staying alive

Robustness is scored, so it is engineered rather than hoped for.

| Failure | Response |
|---|---|
| A role misbehaves (bad format, network drop) | Each of the five stages has its own fuse — **the round is wasted, never the session** |
| The implementer's code fails validation | Retry with the error fed back; then fall through to backup proposals |
| The implementer's code crashes when run | A 1-epoch trial run on 2% of the data exposes it in seconds, **using no training attempt**; the full traceback goes back to the implementer, at most 2 more tries per proposal, then backups |
| Training crashes or times out | The reflection is synthesised **in code** — no LLM call wasted on a run with no result |
| The same idea keeps coming back | Tried cards blacklisted; applied cards never re-proposed |
| Several rounds with nothing diagnosable | Escalate one data fidelity; if already at full data, declare convergence |
| Process dies mid-session | Logs and all three ledgers are flushed every round; restart resumes |
| A change silently has no effect (the part never loaded, the config key is read by nobody) | Validation predictions are fingerprinted; identical to last round → verdict "had no effect", and the card's trust score is protected — the fault is the harness, not the idea |
| A failed round leaves its patch behind | Patch history is rolled back, so one bad round cannot poison every round after it |
| Round after round fails to even run | Stop after `patience` consecutive failures instead of burning the whole budget |

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
  trace.py                    each step as one plain line; each round as full markdown
  knowledge.py                loads the vocabulary and cards; matches by symptom
  schemas.py                  structured-output schemas; symptom names enum-locked
  offline.py                  fake model + fake executor — rehearse a session for $0
  prompts/                    the four role prompts
knowledge/
  symptoms.yaml               12 symptoms — the doctor↔card vocabulary
  cards/                      14 method cards
harness/                    training path: op loading, deep loop, auto-binning of
                            continuous features, loss mount point, R2/R5 guards
modules/                    replaceable parts — the ONLY place the agent may write
kuairand_goat_bridge/       KuaiRand adapter
  official_starter_kit/       vendored, unmodified: data.py, evaluate.py, submit.py
  src/kuairand_bridge/        dataset views (incl. development/holdout split), runner,
                              evaluator, diagnostics, subprocess sandbox, trial run,
                              noise-band measurement (noise.py), session entry point
  examples/goat_trainer.py    research trainer — the agent CAN write modules
  examples/official_fm_trainer.py  baseline trainer — config-only, by design
  configs/kuairand_task.yaml  ← the real run; its trainer_config (features / model /
                                train) is the other thing the agent may change
  configs/fm_baseline.yaml    ← baseline reproduction only
logs/                       offline-rehearsal logs and ledgers (a real run writes
                            everything under its own output_dir)
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

### Step 2 — measure the noise band (~1 min at small fidelity)

```bash
.venv/bin/python -m kuairand_bridge noise \
    --config kuairand_goat_bridge/configs/kuairand_task.yaml --fidelity 小份 --seeds 5
```

Same config, same data subset, five training seeds. The result goes to
`output_dir/logs/noise_bands.json` and step 3 picks it up automatically. `--fidelity`
must match the fidelity the real run starts at — a band is only valid at the fidelity
it was measured on; if they differ, the session does not use it and says so in the
results table.

Skipping this still works, but every "is this a real improvement?" threshold falls back
to the guessed 0.0005 (5–10× tighter than measured), and the session warns at startup.

### Step 3 — the autonomous run (nobody touches the keyboard)

```bash
.venv/bin/python -m kuairand_bridge goat-run \
    --config kuairand_goat_bridge/configs/kuairand_task.yaml
```

> ⚠️ `configs/kuairand_task.yaml` must point at `examples/goat_trainer.py`. The other
> trainer only accepts six hyperparameters and rejects new files by design — pointing
> the real run at it silently reduces the agent to a knob-turner. `tests/test_configs.py`
> guards this.

The terminal while it runs (a real round on real data; no LLM key was configured yet,
so the four roles were scripted — only the trial runs, training and scores are real.
The script makes the implementer read a column that does not exist on its first try):

```
第 1 轮 · 小份数据
  医生    发现 1 个病：冷门视频排不上去（严重 0.7，把握高）
          证据：按视频曝光次数分组：最低桶 GAUC 0.612，最高桶 0.681，差 0.069
  筛卡    对症的卡 3 张：类目兜底(0.60) popularity_prior(0.60) 目标编码(0.60)
  军师    提了 3 个方案，第一个「类目兜底」，预计 GAUC +0.0010 · nDCG@5 +0.0040
  调度    选「类目兜底」，小份数据，备胎：popularity_prior、目标编码
  工兵    第 1 次（类目兜底）：新写 modules/features/_offline_probe_hot.py，配置改 5 行
  试跑    挂了 —— ChildRunnerError: 训练子进程失败：KeyError: '不存在的列'
  工兵    第 2 次（类目兜底）：新写 modules/features/_offline_probe_hot.py，配置改 5 行
  试跑    过了（5.4 秒）
  训练    GAUC 0.6379 · nDCG@5 0.5238（11.9 秒）
  复盘官  猜对了：GAUC +0.0008 · nDCG@5 +0.0031；冷门视频排不上去 部分
          下一步：去看新用户那一组
  本轮    20.4 秒 · 1.8 万 token · 出错并恢复 1 次
```

(Doctor → card match → strategist → scheduler → implementer, attempt 1 → trial run
crashes with the traceback → implementer, attempt 2 → trial run passes → training →
reflector → round total.)

If you intervene at any point, record it — the autonomy score depends on this number
being a real observation rather than a hard-coded zero:

```bash
.venv/bin/python -m agent.cli intervene "round 7 hit OOM, reduced batch size" --round 7
```

### Step 4 — collect the deliverables

Nothing to run: the session writes everything under the config's `output_dir`
(`output/agent_run/` by default).

| Output | Deliverable |
|---|---|
| `logs/rounds.jsonl` | **#3** per-iteration log: hypothesis, full code diff, metrics, errors and recoveries |
| `logs/narrative.md` | **#3** human-readable: the whole session as one storyline |
| `logs/session_summary.json` | **#4** results table: best scores, delta over baseline, tokens, wall-clock, GPU-hours, interventions |
| `logs/rounds/<run>/round_NNN.md` | each round in full: every role's output, every version of the implementer's code (broken ones included), full trial-run tracebacks, time and tokens per step |
| `logs/holdout_report.json` | the holdout exam: gains of round 0 → best round on the development set and on the holdout, and the difference |
| `logs/noise_bands.json` | the noise band from step 2 (only if measured) |
| `logs/snapshots/` | each round's config and module code — any round can be reconstructed |
| `final/` | the validation-best round replayed on full data, with the checked test-set submission |
| `final_summary.json` | best scores, the baseline-reproduction check, the submission, and the whole session summary |

> **The submission comes from the best round, not the last one.** The agent's edits
> are *cumulative*: by round 20 the pipeline holds the whole stack, and round 5's state
> is long gone. After convergence the bridge replays exactly the best round's patch
> history on full data to produce the submission — including round 0, the official
> baseline, when no round beat it.

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

**2. A single comparison can only resolve so much.** The band comes from five seeds,
so the standard deviation itself carries roughly ±35% error — trust the order of
magnitude, not the third decimal. More important is what it says: the primary's wobble
(about 0.003–0.004) exceeds the official convergence threshold of 0.002, and the holdout,
with only 20% of users, wobbles more. "This round beat the last one" can only resolve
changes larger than the band.
*Improvement:* run key changes and the final comparison over multiple seeds and report
mean and interval.

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

**8. The post-competition changes have not yet been run end to end with a real LLM.**
The trial run, per-round trace, noise band and holdout were all verified on real data,
but with the four roles scripted. How the prompts behave under the new mechanisms is
the next thing to find out (see `docs/待做清单.md`).

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
| [docs/待做清单.md](docs/待做清单.md) | What is left to do; every finished item says how it was verified |
| [docs/开发日志.md](docs/开发日志.md) | Development log — one line per change, newest first |
| [docs/四个角色接口.md](docs/四个角色接口.md) | The input/output contract of the four roles |
| [knowledge/symptoms.yaml](knowledge/symptoms.yaml) | The 12 symptoms and their detection rules |
| [knowledge/卡片格式.md](knowledge/卡片格式.md) | Method card format and an annotated example |
| [README.zh-CN.md](README.zh-CN.md) | Chinese README |

---

## 12. After the competition

The competition submission is pinned by the tag `submission-techjam2026`. Work since
then went into making the agent's own failures impossible rather than patching them one
at a time — every change below comes with a test that fails without it.

- **Knowledge that cannot silently go stale.** Cards, prompts, the implementer's example
  files and the hard rules are checked against derived facts: the current task's metric
  names, files that actually exist, fields that actually exist. The mid-competition
  dataset switch had left 11 of 14 cards quoting the previous task's metrics.
- **One task, one pipeline.** The previous dataset's pipeline is gone; a guard test keeps
  its field and metric names out of the code.
- **Mount points the cards were promising.** A model part's own `loss` now actually runs
  (it used to be silently overridden), so the pairwise-ranking and time-decay cards can
  be implemented. Continuous features are binned on train quantiles instead of being
  treated as IDs — in a synthetic check, 37% of validation rows used to fall out of
  vocabulary.
- **You can watch it run.** One plain line per step in the terminal, one full markdown
  file per round. The real-run path used to print nothing for a whole session, and the
  implementer's "broke it, then fixed it" was lost as soon as the round ended.
- **Broken code gets a trial run first.** 2% of the data, 1 epoch; a crash sends the
  traceback back to the implementer (up to 2 more tries) without using a training
  attempt. In one real 7-round run, 4 rounds crashed on start, each burning an attempt
  and docking the card's trust score.
- **Noise band measured, holdout carved out** (sections 3.3 and 3.5). The measurement
  overturned the fallback threshold and showed where single-seed comparisons stop being
  able to tell.
- **Feature modules can no longer see validation answers.** Validation data used to
  reach feature modules with the label column attached; one read of it and the score
  was fiction until the hidden test set. The label is now removed before `transform`
  and reattached after; a module that changes the row count is rejected.
- **No more daily false alarm from the self-check.** "High share of degenerate users"
  is a property of the metric, not something a model can fix; the vocabulary now says
  so, and the self-check only warns about treatable symptoms with no card.
- **Bugs that corrupted results without a single error.** Target encoding output a
  constant on integer IDs; real runs handed the implementer a stale config-only trainer
  as its example; and a leftover score-renaming adapter made the results table report
  every run as 0.67 below the official baseline.
