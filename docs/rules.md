# Rules and scoring

## The dev set and the test set

The competition has two task sets:

- **Dev set:** published, graders included, in the dataset
  [`eval/dev_set.yaml`](../eval/dev_set.yaml) names. You run it on your machine
  with `eval/run.py`, and the `artifacts/` from that run go into your
  submission. Their grade is your dev score.
- **Test set:** hidden. The evaluation server runs every scored submission once
  on it, and that score ranks the leaderboard.

While the competition is open, build and check your entry with the kit, run the
dev set, and submit your harness commit together with its `artifacts/`. When
the competition closes, submissions stop and the leaderboard is the final
standings. Organizers review the code and artifacts of the top entries before
publishing the results. Dates are posted on the competition site.

## Submissions

Sign in to the competition site with your GitHub account (one account per participant). A submission is a single commit of a GitHub repository containing `submission.yaml`, your harness ([`kit/README.md`](../kit/README.md#what-you-submit)), and the `artifacts/` from your dev-set run ([`artifacts.md`](artifacts.md)). Any repository with a `.env` file must be private.

The submit page checks `artifacts/` before queuing a run. A commit refused at submit time ([`artifacts.md`](artifacts.md#refused-at-submit-time-no-submission-slot-is-used)) does not use a submission slot.

You have three scored submissions for the entire competition. A submission
takes a slot once queued. The slot is returned if the submission fails before
task evaluation begins (for example, if it is invalid, fails to build, or fails
the smoke task) or fails because of a fault on our side.

A scored submission takes several hours, and up to a day when your harness uses
most of the 6 hours each task allows: we build your harness, run the smoke
task, grade your committed dev artifacts, run the reproduction check, and
evaluate the test set. Your submission page tracks its progress.

## Scoring

Each task scores from `0` to `1` (the reward its grader returns divided by the
task's maximum). A submission's score is the mean across every task of the
test set, and a task without a result scores `0`.

Faults on our side (Docker daemon errors, task container failures, or outages
and rate limits on Google-sponsored models) are retried up to three attempts
total. A task still without a result after retries counts as `0`, and a run
where no task could be graded is put back in the queue.

Failures of your harness count as your run's outcome: a crash, a protocol
violation, hitting the task's time limit, or errors from a third-party or
self-hosted model endpoint. Deliverables already written in the task container
before the failure are still graded.

The leaderboard ranks participants by their best scored submission. Ties go to
the earlier submission.

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
  endpoint (web search and external APIs are not allowed), inspect grading
  scripts, or tamper with the evaluation containers.
- Do not log, store, publish, or share the prompts or files of the test tasks.
  Because test prompts reach the model endpoint your harness calls, any
  self-hosted endpoint (`openai-compatible/...`) must have prompt and request
  logging turned off.
- The `model_api` in `submission.yaml` must match the model your harness actually calls.

Organizers may disqualify a submission that breaks these rules.
