# Rules and scoring

## The dev set and the test set

The competition has two task sets:

- **Dev set (29 tasks):** published, graders included, in the
  [Hugging Face dataset](https://huggingface.co/datasets/google/physical-engineerings-last-exam) that
  [`eval/dev_set.yaml`](../eval/dev_set.yaml) names. You run it on your machine
  with `eval/run.py`, and the `artifacts/` from that run go into your entry.
  Their grade is your dev score.
- **Test set (56 tasks):** hidden. How entries are evaluated and ranked on it
  will be announced.

Submission details, including dates, will be announced on the
[competition site](https://ple-bench.com). Until then, you can build and check
an entry with the kit and run the dev set locally.

## Entries

An entry is a single commit of a GitHub repository containing `submission.yaml`, your harness ([`kit/README.md`](../kit/README.md#what-you-submit)), and the `artifacts/` from your dev-set run ([`artifacts.md`](artifacts.md)). Your harness must call one of the Google-sponsored `google-cloud/<id>` models ([`models.md`](models.md)), and the entry contains no API keys. How to submit an entry will be announced on the [competition site](https://ple-bench.com).

## Scoring

Each task scores from `0` to `1` (the reward its grader returns divided by the
task's maximum).

Tasks graded on constraints and a design objective (in the dev set, tasks whose
grader includes `tests/paper_score.py`) score each deliverable item as
`s = 0.4 p + 0.6 G O'`, and take the mean of `s` across items:

- `p` is the fraction of constraints met.
- `G` is `1` if every hard constraint (safety and feasibility) is met and `0` otherwise, so breaking any hard constraint caps the item score at `0.4`.
- `O' = clip((O - O_naive) / (1 - O_naive), 0, 1)` compares the design's objective `O` (`1` only at the task's certified bound; for cost minimization, the certified lower bound divided by the design's cost) against `O_naive`, the objective of an unoptimized compliant baseline (`0` if the task has no baseline).

A run's score is the mean across all tasks in the set, and any task without a
result in that run scores `0`.

Failures of your harness count as the task's outcome: a crash, a protocol
violation, hitting the task's 6-hour limit, or an invalid request to the model
(such as one answered with `400`). Deliverables already written in the task
container before the failure are still graded. A harness that exits with an
error before its model produced any output counts as not started and runs again,
up to three attempts in all; a task whose harness never starts scores `0` in
that run.

Rubric tasks and `checks-judge` tasks are scored by an LLM judge. Local dev-set
runs use the judge you configure ([`eval/README.md`](../eval/README.md#before-you-run)),
so scores can vary slightly across judges.

## Dev score

Your dev score is the grade of the `artifacts/` committed in your entry, scored by each dev task's grader ([how](artifacts.md#how-artifacts-are-graded)). A dev task without usable artifacts scores `0`. `eval/run.py` records your local score in `manifest.json`.

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

Organizers may disqualify an entry that breaks these rules.
