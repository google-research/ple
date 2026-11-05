# Rules and scoring

## The dev set and the test set

The competition has two task sets:

- **Dev set (29 tasks):** published, graders included, in the
  [Hugging Face dataset](https://huggingface.co/datasets/google/physical-engineerings-last-exam) that
  [`eval/dev_set.yaml`](../eval/dev_set.yaml) names. You run it on your machine
  with `eval/run.py`, and the `artifacts/` from that run go into your
  submission. Their grade is your dev score.
- **Test set (56 tasks):** hidden. The evaluation server runs every scored
  submission three times on it, and the mean of the three runs ranks the
  leaderboard.

While the competition is open, build and check your entry with the kit, run the
dev set, and submit your harness commit together with its `artifacts/`. When
the competition closes, submissions stop. Submissions made before the deadline
get all their test runs, and the leaderboard is the final standings once every
one of them is scored. Organizers review the code and artifacts of the top
entries before publishing the results. Dates are posted on the [competition site](https://ple-bench.com).

## Submissions

Sign in to the [competition site](https://ple-bench.com) with your GitHub account (one account per participant) and [register](https://ple-bench.com/competition/register) before your first submission. A submission is a single commit of a GitHub repository containing `submission.yaml`, your harness ([`kit/README.md`](../kit/README.md#what-you-submit)), and the `artifacts/` from your dev-set run ([`artifacts.md`](artifacts.md)). The repository can be public, or private with the competition's GitHub App installed. Your harness must call one of the Google-sponsored `google-cloud/<id>` models ([`models.md`](models.md)); the competition takes no API keys.

The submit page checks `artifacts/` before queuing your submission. A commit refused at submit time ([`artifacts.md`](artifacts.md#refused-at-submit-time-no-submission-slot-is-used)) does not use a submission slot.

You have three scored submissions for the entire competition, and you can have
one in the queue or under evaluation at a time: submit the next one once the
previous one is scored or has failed. A submission takes a slot once queued.
The slot is returned if the submission fails before task evaluation begins (for
example, if it is invalid, fails to build, or fails the smoke task) or fails
because of a fault on our side, even after one or two of its runs; such a
submission does not rank.

An evaluation takes about a day once it starts, and up to about three days when
your harness uses most of the 6-hour limit on many tasks; time in the queue
comes on top. We build your harness, run the smoke task, grade your committed
dev artifacts, run the reproduction check, and then run the test set three
times at once (about 36 tasks in flight, 12 per run). Your submission page
tracks its progress.

## Scoring

Each task scores from `0` to `1` (the reward its grader returns divided by the
task's maximum).

Tasks graded on constraints and a design objective (in the dev set, tasks whose
grader includes `tests/paper_score.py`) score each deliverable item as
`s = 0.4 p + 0.6 G O'`, and take the mean of `s` across items:

- `p` is the fraction of constraints met.
- `G` is `1` if every hard constraint (safety and feasibility) is met and `0` otherwise, so breaking any hard constraint caps the item score at `0.4`.
- `O' = clip((O - O_naive) / (1 - O_naive), 0, 1)` compares the design's objective `O` (`1` only at the task's certified bound—for cost minimization, the certified lower bound divided by the design's cost) against `O_naive`, the objective of an unoptimized compliant baseline (`0` if the task has no baseline).

A test run's score is the mean across every task of the test set, and a task
without a result in that run scores `0`. A submission's test score is the mean
of its three run scores; the leaderboard shows it with ± the standard deviation
of the three, which does not affect rank. An entry scored on a single run
shows without ±.

Faults on our side (Docker daemon errors, task container failures, or outages
and rate limits on Google-sponsored models) are retried, up to three attempts
per task. A task still without a result after its attempts counts as `0` in
that run, and a test run in which no task could be graded runs again; the runs
already scored are kept.

Failures of your harness count as the task's outcome: a crash, a protocol
violation, hitting the task's time limit, or an invalid request to the model
(such as one answered with `400`). Deliverables already written in the task
container before the failure are still graded. A harness that exits with an
error before its model produced any output counts as not started and runs again,
up to three attempts in all; a task whose harness never starts scores `0` in
that run.

The leaderboard ranks participants by their best scored submission, using the
test score rounded to the three decimals it shows. Ties go to the earlier
submission. A submission appears on the leaderboard once all of its runs are
scored; until then, your earlier best stays.

Rubric tasks and `checks-judge` tasks are scored by a fixed LLM judge chosen by
the organizers. Local dev-set runs use the judge you configure, so local scores
may differ slightly from the leaderboard.

## Dev score and reproduction check

Your dev score is the grade of the `artifacts/` committed in your submission, scored by each dev task's grader with the organizers' judge ([how](artifacts.md#how-we-grade-them)). A dev task without usable artifacts scores `0`. The local score recorded in your `manifest.json` is shown alongside ours for reference only.

After grading your artifacts, we run your harness live on a random sample of dev
tasks (chosen after you submit) and compare those live scores with your
artifacts' scores on the same tasks. If the live run scores more than `0.25`
below your artifacts on average, the submission is flagged for review. Because
model output varies from run to run, an organizer inspects every flagged
submission before making a decision.

Your leaderboard rank depends only on the test set. The leaderboard
displays each entry's dev score, and reproduction check details are visible to
you on your submission page.

## Fair play

- Do not hard-code task answers or detect specific tasks to branch on them, dev
  tasks included. The dev-set graders and reference files are public so you can
  evaluate locally, and your harness must never read or use them. A harness that
  reads its own `artifacts/` at build or run time (from disk, git history, or
  the network) breaks this rule.
- Your committed `artifacts/` must be exported by `eval/run.py` from the harness
  you submit, running the `model_api` declared in `submission.yaml`. Re-running
  individual dev tasks and exporting their results is fine, but do not create or
  edit files in `artifacts/` by hand, with another harness, or with another
  model.
- Do not bypass the egress proxy, connect to any host other than your model's
  endpoint (web search, including a provider's own search tools, and external
  APIs are not allowed), inspect grading scripts, or tamper with the evaluation
  containers.
- Do not log, store, publish, or share the prompts or files of the test tasks.
- The `model_api` in `submission.yaml` must match the model your harness actually calls.

Organizers may disqualify a submission that breaks these rules.
