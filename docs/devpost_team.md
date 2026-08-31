## Team member contributions

We thought of the project as a school with four things in it. Each of us took one.

**The exam hall — data and the runner.**
Somebody has to be able to say "try this idea and tell me the score." That is harder
than it sounds. The hall has to set the same exam every time, and it must not fall down
no matter how strange the idea being tested. So: get the real dataset in, split it
exactly the way the organisers split it, and score it with the organisers' own code
rather than our own copy of it — a scorer that is *almost* the official one only shows
you the difference on the day you submit. Every experiment runs in its own process with
a timer on it, so one badly written idea can hang forever without taking the day down
with it.

**The student — the model.**
First build a recommender by hand that scores as well as the official one, so we have a
fair thing to beat. Then take it apart, so that every piece — the features, the network,
the way it is trained — can be replaced by someone who is not you. The second half is
the one that matters. **An agent can only change what somebody made changeable.**

**The teacher — the agent.**
Read the student's report card, work out what is actually wrong, decide what to try
next, write the code, and know when to stop. That is four different jobs, so we made
four roles and put plain code between them. The hardest part was not getting it to have
ideas; it was stopping it from marking its own homework too kindly. Those rules are
written as checks that fail, not as instructions we hope it follows.

**The camera — the log.**
Record all of it, because more than half the marks are for the thinking rather than the
score. The format had to be agreed on the first day, before the other three wrote a
line — if the camera turns up late, everyone else has to go back and redo their work to
fit it. Every round we keep what the agent hoped would happen, the code it wrote, the
score it got, and anything that broke along the way.

The order mattered more than we expected. Until the exam hall actually worked, the
teacher and the student were both just guessing.

And the split was never clean. We all read each other's code, and nearly every serious
bug in this project was found by the person who did not write it.
