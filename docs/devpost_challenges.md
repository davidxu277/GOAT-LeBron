## Challenges we ran into

**The doctor kept saying "I can't tell."**

Our first real run was a disaster of a particular kind. Every single round, the doctor
looked at the scorecard and reported nothing wrong. The session converged having
written zero lines of code. The strategist and the engineer were never even woken up.

The obvious fix is to lower the bar — tell it to find *something*. We didn't, and we're
glad. We went and read what it actually wrote, and it had been right every time:
*"needs the training-set score, not here", "needs the per-bucket numbers, not here",
"needs users split into seen and unseen, not here."*

Most of what a doctor diagnoses is not the total score. Overfitting is the gap between
training and validation. Cold-start is one bucket against another. New users are one
group against another. Our scorecard had three validation numbers and nothing else. We
had asked it to diagnose from a thermometer reading.

So we rebuilt the scorecard instead of the doctor. It now carries the training-set
score, per-exposure-bucket scores, seen-versus-unseen users, day-by-day scores, and the
composition of the user base. The first run afterwards, it found two real problems and
ruled out four others with numbers for each. Nothing about the doctor changed.

**A library of methods is always going to run out.**

We wrote a library of method cards — each one a technique from the recommender-systems
literature, with what it treats, why it works, how to build it, and what failure looks
like. The obvious problem: the agent can only ever be as good as our reading list.

The tempting fix is to hand it the papers. Fill the context window with literature and
let it work things out. We think that's worse, not better. Every round it would have to
re-read everything and decide by feel which paper applies — an expensive, fuzzy
judgement, repeated, with a fresh chance to hallucinate a connection each time.

We did two other things instead.

First, we distilled rather than dumped. A card is a paper compressed into the four
things you need at the moment of choosing, and it is *labelled* with which problems it
treats — using the same fixed vocabulary the doctor is restricted to. That makes
matching a set intersection: instant, free, and incapable of inventing a match that
isn't there. The doctor's output schema physically cannot produce a symptom name that
no card treats.

Second, we gave the strategist an escape hatch. If nothing in the library fits the
diagnosis, or everything that fits has already been tried, it is allowed to invent a
remedy that exists on no card — it just has to write the implementation sketch itself
and argue for it like any other proposal. Those proposals go through exactly the same
scrutiny afterwards. In our runs, several of the agent's own inventions came from
reading the error message of something that had failed two rounds earlier.

The library is the floor, not the ceiling.

**The score went up, and it meant nothing.**

This is the failure we were most afraid of, because it looks like success.

The agent says "cold-start items are the problem", applies a fix, and the score
improves. Except the cold-start bucket didn't move at all — something unrelated did.
Record that as "the fix worked" and it will keep reaching for that fix, and keep
getting further from the real problem. Do it for twenty rounds and you have a
confidently wrong system with a rising score.

So the reviewer role isn't allowed to just look at the total. It has to name the number
it was aiming at and report what that number was before and after. And those rules are
written as checks that fail, not as instructions in a prompt. It cannot claim success
while admitting nothing improved. It cannot claim a symptom is cured while its own
before and after are identical. It cannot claim a win smaller than the noise. If a
proposal said it would treat three problems, it must account for all three.

Everything we wrote as an instruction was eventually ignored. Everything we wrote as a
check held.

**Is 0.03 a big difference? Nobody knew.**

The rules for diagnosis are full of thresholds — this bucket is 0.03 below that one, so
report it. But 0.03 means completely different things in different places. In a small
bucket with fifty positive examples, the number moves by more than that on its own if
you just change the random seed. In a large one it barely moves at all. The same
threshold turns noise into a diagnosis in one place and hides a real problem in another.

So we measured it. Run the same configuration several times, changing nothing but the
seed, and watch how much each number wobbles by itself. Then cross-check that against
the textbook formula for the same quantity. The agent is handed its own measurement
error, and forbidden from calling anything smaller than it a finding — or a gain.

The interesting case was when the measurement came back as exactly zero. That does not
mean the number is stable; it means our way of measuring couldn't move it. Accepting
that would have set the threshold to nothing, and every random flicker would have
counted as a victory. So a measured zero falls back to the analytic estimate — an
honest "we can't tell right now" instead of a flattering "no noise here".

**One wrong number, and the agent reached the opposite conclusion. Quietly.**

The rule for "not learning enough" compared our score against the official baseline.
Somebody had written that baseline into the rules as 0.6016. That figure is the
*combined* score. The GAUC baseline is actually 0.6674.

Our model scored 0.6638 — below the official baseline on all three metrics. The doctor,
reading our number, calculated that we were comfortably above it, and reported nothing
wrong. Nothing crashed. No test failed. One typo-grade mistake produced the exact
opposite of the truth, and it would have kept doing so for the whole run.

The fix was not to correct the number. It was to stop giving the agent a baseline at
all. Ranking against the official baseline is the *judges'* job. Handed that number,
the agent stops being a diagnostician and becomes a thermostat: above the line, nothing
to do; below it, "we need to try harder". It stops reading the train-validation gap and
the buckets and the user mix — the evidence that actually tells you what is wrong. It
now compares against one thing only: its own previous round. The baseline is still
recorded from end to end, for us. A test makes sure it never reaches the agent.

**The pharmacy stocked medicine the hospital couldn't administer.**

For a while, most of the cards described things our training path physically could not
do. The agent would choose one, the engineer would write it, the run would come back
unchanged — and the reviewer, quite reasonably, would conclude the method was useless
and mark the card down.

That is a very expensive kind of wrong. The card was fine. Our pipeline was missing a
capability. But the verdict goes into a ledger that persists, so the agent learns to
stop suggesting some of the best-known techniques in the field, for a reason we caused.
Twenty rounds later that mistake is frozen into what the system believes.

So "we couldn't run it" and "it didn't work" are now two different outcomes with two
different bookkeeping paths, and only the second one costs the method any credibility.
Then we went and built the missing capability, so the agent could write real model and
training code rather than only adjusting numbers.

**The bugs that only appear at the very end.**

Three separate defects sat in the final step — choosing the best round and producing
the submission. Each would have thrown *after* the entire session finished. Hours of
training, and then no file.

The third one is our favourite. Round zero is a perfectly valid choice — if the agent
never improves on the starting point, round zero is exactly what you should submit. But
the code asked `if selected_round:`, and in Python zero is false. So "submit round
zero" was silently read as "nobody chose", and it submitted the last round instead.

There was already a test for this, and it could not have caught it. The test only ever
created one round — so "round zero" and "the last round" were the same round, and both
branches gave the same answer. The test now runs two.

The pattern behind all three is what changed how we work. **The dangerous failures
don't raise errors.** They finish successfully and hand you the wrong thing. So the
rules live in checks and tests, including one that fails if somebody points the real
run at a trainer that can't write code — which is exactly the one-line change that
quietly turned our agent back into a knob-turner for five rounds.
