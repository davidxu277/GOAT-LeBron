## Team member contributions

We were building a hospital for machine learning models. Four of us, four parts of it.

**The first of us got the patient ready.**
None of this means anything without data you can trust. They took the raw dataset,
cleaned it up, and split it the way the organisers split it — not the way that would
have made our scores look better. They are also the reason we score with the
organisers' own code instead of writing our own version. A scorer that is *nearly*
right is the worst kind. It agrees with you all week, then disagrees on the one day
that counts.

**The second of us wrote the pharmacy.**
They read their way through recommender systems — how people actually fix a ranking
model when it is broken — and turned each technique into a card. Every card says the
same four things: what it treats, why it should work, how to build it, and what it
looks like when it fails. So when the agent has a diagnosis, it has somewhere to look
it up.

Then they did the harder half. They gave the agent hands. It doesn't only pick cards
off a shelf — it can write the code itself. New features, new model parts, new training
tricks. Whatever it writes goes straight into the pipeline and gets trained.

**The third of us built the medical team.**
One model staring at one scorecard was never going to be enough. So the thinking got
split into four. A doctor, who reads the report and says what is wrong. A strategist,
who picks the treatment and has to argue for it. An engineer, who writes it. And a
reviewer, who checks afterwards whether the thing they were treating actually got
better.

Getting them to talk to each other was the easy part. The hard part was making sure the
reviewer couldn't quietly let the strategist off the hook — so those checks are code
that fails, not instructions we hoped would be followed. And any one of the four can
fall over without bringing the whole run down with it.

**The fourth of us ran it for real.**
Everything above is a theory until somebody presses go on the real dataset and watches
it break. They ran the real sessions, found where it fell apart, and fixed it. Most of
what we now know about this system, we know because they ran it and it didn't work.

None of the four was optional, and we found that out in order. Until the data was
right, nobody could tell whether the agent was wrong or the exam was. Until the agent
had somewhere to look things up, it had good instincts and nothing to do with them.
And until someone ran the whole thing end to end, we were all just fairly confident.
