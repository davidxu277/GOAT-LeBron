## Challenges we ran into

We expected the hard part to be making the agent clever. It wasn't. Every problem that
actually cost us time had the same shape: it looked like one thing and turned out to be
another. A cautious agent that was really a blind one. A limit on its knowledge that was
really a decision we had made. And a success that was really nothing at all.

**The doctor kept saying it couldn't tell.**

We came back to the first real run expecting to read arguments. We read the same sentence
twenty times. *No findings this round.* The session had converged having written nothing,
and the two roles that do the actual work had never been woken up.

Our instinct was that the doctor was too timid. Before loosening it we read what it had
written, which we should have done first:

*Needs the training-set score. Not here.*
*Needs the per-bucket numbers. Not here.*
*Needs users split into seen and unseen. Not here.*

It was right. Almost nothing a doctor diagnoses is the total score — overfitting is a
distance, cold start is one bucket against another. Our scorecard had three validation
numbers on it. We had handed it a thermometer and asked for a diagnosis.

So we rebuilt the scorecard and left the doctor alone. Next run: two real problems found,
four ruled out, numbers for all six. Not a word of the doctor changed.

When the agent says it cannot tell, that is information about us.

**A library of methods always runs out.**

Every technique we knew went onto a card — what it treats, why it works, how to build it,
what failure looks like. Which means, on the face of it, the agent can never be smarter
than our reading list.

The obvious answer is to hand it the papers. We think that is worse: it would re-read the
shelf every round and decide by feel what applied, with a fresh chance each time to invent
a connection that isn't there.

So we distilled instead of dumping. A card is a paper reduced to the four things you need
at the moment of choosing, labelled with the problems it treats — using the same fixed
vocabulary the doctor is restricted to. Finding the right card is then not a judgement at
all. It is an intersection of two sets: free, instant, and unable to invent a match.

Then we gave the strategist a way out of the library. When nothing fits, or everything
that fits has already failed, it may propose something that exists on no card — it just
has to sketch the implementation itself and face the same review afterwards. That path
gets used. Some of its inventions came from reading the error message of something that
had broken two rounds earlier.

The library turned out to be the floor, not the ceiling.

**The score went up and it meant nothing.**

This is the failure we feared most, because it arrives dressed as success.

The agent decides cold-start items are the problem, writes a fix, and the score improves.
Except the cold-start bucket never moved — the gain came from somewhere else. Write that
down as "it worked" and it reaches for the same fix next time, and the card's trust score
climbs. Twenty rounds later you have a rising number and a completely wrong picture of
your own model.

It is the maths tutor problem. The child is bad at maths, you hire a maths tutor, the term
score goes up three points — and maths is unchanged, the gain came from an easy literature
paper. Now you will hire that tutor again.

So the reviewer may not look at the total. It has to name the number it was aiming at and
report what that number did. And we wrote those rules as checks that fail, not
instructions that ask: it cannot claim success while admitting nothing improved, cannot
call a problem fixed while its own before and after are identical, cannot claim a win
smaller than the noise, and cannot promise to treat three problems and account for one.

Every rule we wrote as an instruction was eventually ignored. Every rule we wrote as a
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
