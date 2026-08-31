## What we learned

**Machine learning research can be autonomous, and watching it happen changes how you
think about what these models are for.**

Before this we thought of an LLM as something you ask questions. What we built asks its
own questions. It reads a result, decides what is wrong, argues for what to try next,
writes the code, runs it, and then tells us whether it was right — including when the
answer is no. The loop that we thought was the human part of the job turned out to be
mechanisable, and the interesting work moved somewhere else: to deciding what evidence
it gets to see, and what it is not allowed to conclude.

**Prompts are signs on the wall. Validators are the wall.**

Every constraint we wrote as an instruction was eventually ignored — not maliciously,
just under pressure, on a long run, in a case we hadn't imagined. Every constraint we
wrote as a check that fails held every time. We stopped asking the agent to be careful
and started making carelessness impossible to express.

**"I can't tell" is a feature, and we nearly deleted it.**

An agent that always finds something wrong is useless, because you cannot distinguish
its findings from its habits. When ours reported nothing for twenty rounds our instinct
was to lower the bar. Reading its reasoning instead was the single most useful hour we
spent — it had been telling us what was missing, in detail, the entire time. Honest
uncertainty is not a failure to answer. It is the answer, and usually it is about you.

**Giving an agent less can make it smarter.**

We removed the official baseline from what the agent sees, and its diagnoses got
sharper. With the number, it compared itself to a constant and stopped looking. Without
it, it had to look at the actual evidence — the gap between training and validation, one
bucket against another, the shape of the user base. Not every piece of context you can
give a model is worth giving it.

**The dangerous failures don't raise errors.**

Almost nothing that hurt us crashed. A wrong constant produced the exact opposite
conclusion. A one-line config change quietly turned the agent back into a knob-turner
for five rounds. `if round:` treated round zero as "nobody chose". Every one of them
finished successfully and handed us the wrong thing. Loud failures are cheap; you find
them in a minute. We now spend our review time asking what could go wrong here without
anyone noticing.

**A number that can only ever be zero is not a measurement.**

We report how many times a human intervened. It read zero — and then we found the
command wrote to one file while the run read another, so it could only ever have read
zero. The number was true and worthless. If you are going to report a metric, first make
sure the bad value was reachable.
