## Inspiration

An ML engineer's day is a loop: read the metrics, guess what's wrong, write code, train, look at the score, decide what to try next. Most agents that automate this loop skip the first two steps — they mutate code and keep whatever makes the number go up. That works, and it teaches you nothing.

The failure mode we cared about is quieter than "the score didn't improve". It is **the score improved, but not for the reason you think**:

*The doctor says the kid is bad at maths. You hire a maths tutor. The term score goes up 3 points. But maths is unchanged — the gain came from an easy literature paper. Record that as "tutoring works" and you will keep buying maths tutoring, and keep getting further from the actual problem.*

An agent that cannot separate those two cases accumulates confident nonsense. So our agent must state a hypothesis **before** each experiment, and afterwards **code** — not a prompt — checks whether the specific number it pointed at actually moved.

## What it does

You give it a dataset and a scoring rule. Then nobody touches the keyboard.

Each round runs four LLM roles, with plain deterministic code between them:

- **Doctor** — reads the scorecard, names what is wrong, ranks findings by severity
- *card matcher* — **pure code**: intersects the symptoms with the method-card library
- **Strategist** — picks 3 candidate remedies, each with a written causal chain from evidence to root cause to fix
- *scheduler* — **pure code**: picks 1 by cost/benefit and chooses the data fidelity
- **Implementer** — writes the actual code: a config change, or a new module
- *training* — real training, real scoring, a new scorecard
- **Reflector** — judges whether the hypothesis held, and updates that card's trust score

The two *pure code* steps call no model. The LLM is woken at four decision points only — which is how the token budget stays small, and token usage is scored.

Rounds are chained. This round's scorecard is the next doctor's input; failed cards are blacklisted; applied cards are never re-proposed; unchosen proposals are shelved and re-offered while still relevant. The session stops on convergence (no gain above ε = 0.002 for 3 consecutive rounds), budget exhaustion, or nothing left to diagnose at full data.

## Three things that make it more than a code-mutation search

**1. Diagnose, then prescribe.**

A 12-entry symptom vocabulary is the shared language between the doctor and a 14-card method library. The doctor's structured-output schema enum-locks that vocabulary, so it *physically cannot* name a symptom that no card treats. Card matching is therefore a set intersection — free, deterministic, and incapable of hallucinating a match.

**2. It cannot lie to itself.**

The anti-self-deception rules are validators in code, not sentences in a prompt. A reflection is rejected if it claims success while admitting no target symptom improved; if it claims a symptom is resolved while its own before/after numbers are identical; if the change is smaller than the seed-to-seed noise; or if it fails to account for every symptom the proposal claimed to treat.

Rejection is not discarding — the exact violation is quoted back and the role is asked again.

*Prompts are signs on the wall. Validators are the wall.* Every constraint we wrote as an instruction was eventually violated. Every constraint we wrote as a validator held.

**3. It knows its own measurement error.**

"The cold-start bucket is 0.03 below the hot bucket" is only a finding if 0.03 exceeds how much that number wobbles on its own. We measure the wobble across random seeds and cross-check it against the Hanley–McNeil analytic standard error computed from the positive/negative counts. Anything under the band may not be reported as a symptom, and may not be claimed as a gain.

## How it addresses the problem statement

- **Autonomous iteration** — a closed outer loop with cross-round state persisted every round: a card-trust ledger, a training-time ledger, a proposal shelf, and a blacklist.
- **Auditable reasoning** — every round records the hypothesis, the causal chain, the full code diff, the resulting GAUC/nDCG@5, and the verdict. That log *is* the deliverable.
- **Beyond hyperparameter tuning** — the agent writes real modules (FeatureOp / ModelOp / TrainOp) into `modules/`, loaded into a deep training path with embedding tables, an epoch loop, per-epoch validation, and best-weight rollback.
- **Robustness as a scored property** — each of the five stages has its own fuse, so a bad round is wasted but never the session. Training runs in a spawned subprocess under a hard wall-clock timeout. Every stumble and recovery is logged.
- **Honest accounting** — tokens, wall-clock, GPU-hours, iteration count and manual interventions are emitted by the system, not typed in by hand.

## How we built it

Two decisions did most of the work.

**The scorecard carries evidence, not just a score.** Six of the twelve symptoms are judged on *grouped* numbers rather than the total: overfitting needs a train-set score, cold-start needs per-exposure-bucket scores, new-user needs a seen/unseen split. When the scorecard held only three validation numbers, the doctor correctly and repeatedly answered "I cannot tell" — and the strategist and implementer were never even invoked. Adding the grouped evidence is what turned the loop on.

**The agent builds its own codebase, not just its own config.** An agent limited to hyperparameters is a knob-turner, and the track explicitly rewards going past that. So we gave it three places to write real Python, behind three protocols in `modules/base.py`:

- `FeatureOp` — feature engineering. `needs()` declares which raw columns it reads, `fit()` learns from the training split only, `transform()` applies. Any new column it produces is appended to the feature table automatically, so a feature the agent invented in round 6 is simply part of the model from round 6 onward.
- `ModelOp` — network architecture. `build()` receives the feature spec and returns a model; `predict()` defines how to score.
- `TrainOp` — training strategy, as callbacks on `on_train_begin` / `on_epoch_end` / `on_train_end`. Early stopping, weight averaging and schedules are written as ordinary code hooked into the loop.

The modules the agent writes are imported by path at run time and dropped into a real deep training path: ID vocabularies built from the training split only, embedding tables, an epoch loop, per-epoch validation, TrainOp callbacks fired between epochs, and a rollback to the best epoch's weights at the end.

The freedom is bounded by the same rules a human contributor follows, enforced in code rather than requested in a prompt. It may create files under `modules/` and nothing else — a path containing `..`, or one that would overwrite existing human code, is rejected outright. Every statistic is fitted on the training split; a module that computes target statistics is driven through an out-of-fold path so a row can never be encoded using its own label.

This is the difference between an agent that reports "I raised the learning rate" and one that reports "I wrote a video-popularity bucketing feature, here is the file, here is what it changed." After a session, the new `.py` files in the repository are its work, and they are in the log with full diffs.

## Development tools

VS Code · Claude Code (Anthropic) as a coding assistant · Git and GitHub · pytest

## APIs used

**DeepSeek API** — the default provider for the four agent roles.
**Anthropic Claude API** — the alternative provider.

Both sit behind one interface with structured-output enforcement and token accounting. No other external APIs; the agent has no internet access at run time.

## Libraries and frameworks

- **PyTorch** — the deep training path (embedding tables, epoch loop)
- **NumPy** — the official baseline factorization machine, and all metric computation
- **pandas / PyArrow** — data handling
- **PyYAML** — configs, the symptom vocabulary, the method cards
- **pytest** — the offline test suite
- **multiprocessing (spawn)** — a cross-platform training sandbox with a hard timeout

The official KuaiRand starter kit (`data.py`, `evaluate.py`, `submit.py`) is vendored **unmodified** and called directly. We never reimplement the metric — a metric that is "almost the official one" only reveals itself at submission time.

## Datasets and assets

**KuaiRand-Pure** (the required benchmark). Label `long_view`; within-user ranking over logged impressions; GAUC and nDCG@5, with primary being their mean. Official date split — train 2022-04-08 to 04-21, validation 04-22 to 04-28, test 04-29 to 05-08. Verified row counts: 1,141,112 / 124,909 / 170,588.

**Official baseline** from the starter kit: a factorization machine (k=16, lr=0.001, five categorical fields, NumPy only), scoring validation GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016.

No external or manually labelled data.

## Challenges we ran into

**A doctor with no evidence says "no finding" forever.**
Our first real run produced `no_finding` every single round and converged having written zero lines of code. The doctor was right — eleven of the twelve symptom rules ask for grouped numbers that the scorecard simply did not contain. We had migrated the symptom vocabulary to the new dataset and forgotten to migrate the evidence with it.

**One wrong constant inverted a conclusion, silently.**
The symptom table hard-coded "official baseline GAUC 0.6016". That figure is the *primary*, not the GAUC — the real GAUC baseline is 0.6674. Our model scored 0.6638, below baseline on all three metrics. The doctor, reading our wrong number, concluded "clearly above baseline" and reported nothing. Nothing crashed. That single number is why the agent no longer sees a baseline at all.

**Bugs that only fire on the very last step.**
Three separate defects in selecting and submitting the best round would each have thrown *after* a full training session — hours of compute, and no submission file. A wrong nesting level (KeyError), a missing argument (TypeError), and `if self._selected_round` treating **round 0 as falsy**, which silently submitted the last round instead of the chosen one. The existing test structurally could not catch the third: its history contained one round, so "round 0" and "the last round" were the same round.

**A one-line config change turned the agent back into a knob-turner.**
Switching the trainer to the official FM — entirely legitimate for baseline reproduction — also disabled code writing, because that trainer rejects new files by design. Five consecutive rounds were burned while the strategist worked this out for itself, each costing a real training run and four LLM calls. Nothing errored. The runs just quietly stopped producing code.

All four say the same thing, and it shaped the codebase: **the dangerous failures are the silent ones.** So constraints live in validators and tests rather than in comments — including a test that fails if the production config points at a trainer that cannot write code.

## Accomplishments that we're proud of

- Six distinct ways for a reflection to be internally inconsistent are rejected **by code**, with the violation quoted back for a retry.
- The agent is told its own measurement error and forbidden from calling anything smaller a finding — including the honest case where the empirical band measures 0.0000 because the sampling scheme cannot perturb that metric, and we fall back to the analytic band rather than accept a threshold of zero.
- **Its negative results are as well argued as its positive ones.** In one run it ruled out cold-start (lowest bucket GAUC 0.6667 is *above* the hottest bucket's 0.655), new-user (unseen 0.6877 *above* seen 0.6668), and temporal drift (no monotone decline; the last two days rise) — and flagged that the smallest bucket held only 166 positives and was therefore unusable as evidence.
- A full offline rehearsal mode — fake model, fake executor, injectable failures — so the entire loop including its recovery paths is testable with no network and no cost.

## What we learned

**"I cannot tell" is a feature.** Forcing the doctor to say so, instead of manufacturing a finding, is what made its findings worth reading — and it pointed us straight at the missing evidence.

**Withholding information can make an agent smarter.** Removing the baseline was the single change that most improved diagnosis quality.

**A metric that can only ever be zero is not a measurement.** The manual-intervention count is scored, so reporting "0" only means something if non-zero was reachable. Ours wasn't — the command wrote to one file and the run read another — until we fixed it.

## What's next for GOAT-LeBron

Every direction below has the same shape: move work that the model is doing badly into code, so the model can spend its budget on the one thing only it can do — deciding what is worth trying next.

**Let the agent debug its own code.** Today it can repair a module that fails validation, because the validator hands back the exact violation. But a module that passes validation and then *crashes at run time* costs a whole round: the error becomes a hint for the next round, and a different remedy may be chosen entirely. The fix is a three-stage preflight before the expensive training starts — parse, import, then run the module against a hundred rows — and, when something still breaks, one repair attempt with the full traceback and the offending source, charged to the same round. Writing code is only half of autonomy; the other half is fixing it.

**Make diagnosis deterministic.** The detection rules are currently passed to the doctor as text, and it does the arithmetic. The same scorecard can therefore yield different findings twice, and the arithmetic can simply be wrong. Computing the rules in code makes diagnosis reproducible and leaves the doctor to do what a model is actually good at: weighing confidence, ranking severity, and noticing the thing no rule anticipated. The same move fixes measurement — the noise bands need rewriting for a within-user ranking metric, and once thresholds are computed rather than described, they can be enforced rather than suggested.

**Let it write its own method cards.** The card library is the ceiling on what the agent can propose, and today every card was written by a human. When the agent invents a remedy that survives reflection — the hypothesis held, the target symptom actually moved, the gain cleared the noise band — it has already produced everything a card needs: what it treats, why it should work, how it was implemented, and what failure looked like. Writing that back as a new card is the difference between an agent that solves a task and an agent whose library grows every time it runs. The trust ledger already persists across sessions; the knowledge should too.

**Measure autonomy instead of asking for it.** The manual-intervention count is self-reported: a human is trusted to run a command after touching a live run. Since we know exactly which files the agent wrote each round, any other change to the config or to `modules/` is, by elimination, a human's. Detecting that automatically turns the autonomy number from a promise into an observation — which is the same standard we already hold the agent to everywhere else.

## Results

```
                                  GAUC     nDCG@5   primary
Official baseline (validation)   0.6674    0.5357    0.6016
Ours (validation-best)           ⟨FILL⟩    ⟨FILL⟩    ⟨FILL⟩
Absolute delta                   ⟨FILL⟩    ⟨FILL⟩    ⟨FILL⟩
```

Resource usage to reach convergence: ⟨FILL⟩ LLM tokens (input + output), ⟨FILL⟩ agent wall-clock, ⟨FILL⟩ of the 50 permitted iterations, ⟨FILL⟩ GPU-hours, and ⟨FILL⟩ manual interventions.
