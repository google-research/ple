# Dev set evaluation

`eval/run.py` runs your entry on the nine public dev tasks, grades the results with the leaderboard's graders, and writes your harness's outputs into your entry's `artifacts/` folder, which you commit and submit ([`docs/artifacts.md`](../docs/artifacts.md)). `eval/run.py --artifacts` grades that folder offline just as the server does ([below](#check-your-artifacts-before-you-submit)).

The tasks live in the Hugging Face dataset named in
[`dev_set.yaml`](dev_set.yaml) (`tasks_repo` at commit `tasks_revision`). On
first run, `eval/run.py` downloads the needed tasks into
`~/.cache/plebench/tasks/` and verifies each archive against its `sha256` in
`dev_set.yaml`.

## Before you run

What you need:

- Docker running locally with about 10 GB of free disk space for images, plus the CPUs and memory described in [Time and cost](#time-and-cost).
- [uv](https://docs.astral.sh/uv/) and Harbor 0.20.0 with `websockets`: `uv tool install harbor==0.20.0 --with websockets`. `eval/run.py` checks Harbor's version and `websockets` before starting.
- Access to your declared model:
  - For a `google-cloud/<id>` model, set `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` pointing to a local proxy ([`docs/models.md`](../docs/models.md#running-google-cloud-models-on-your-machine)).
  - For commercial providers, put the required API key in your entry's `.env`.
  - For a self-hosted model, see [A model you serve yourself](#a-model-you-serve-yourself).
- A judge for the `rubric` and `checks-judge` tasks. `eval/run.py` uses the first available option:
  1. `JUDGE_BASE_URL` with `JUDGE_API_KEY` (any OpenAI-compatible endpoint; setting `JUDGE_API_KEY` without `JUDGE_BASE_URL` stops the run so keys are never sent to the wrong host)
  2. `MODEL_PROXY_URL` with `MODEL_PROXY_KEY`
  3. Your entry's own model and key (when not using `google-cloud/<id>`)

  `JUDGE_MODEL` defaults to `claude-opus-5-5` (or to your entry's model when
  option 3 is used). The leaderboard always judges with Claude Opus 5.5, which
  another judge model only approximates.
- Internet access to `huggingface.co` the first time a task runs. While the
  dataset is private, set `HF_TOKEN` to a Hugging Face token with read access,
  or pass `--tasks-dir DIR` to use pre-downloaded task folders.

Keep local proxy, judge, and Hugging Face settings in an env file outside your
entry repository and pass it with `--env-file` (keep only the keys your harness
uses in your entry's `.env`):

```bash
# local.env
MODEL_PROXY_URL=http://localhost:4000
MODEL_PROXY_KEY=<the key you gave kit/model_proxy/start.sh>
# JUDGE_BASE_URL=https://<an OpenAI-compatible endpoint>/v1
# JUDGE_API_KEY=<its key>
# HF_TOKEN=<a Hugging Face token, while the dataset is private>
```

## Run the dev set

From the root of this repository (with your entry in its own separate Git repository; see [Quick start](../README.md#quick-start)):

```bash
# Run all nine dev tasks
uv run python eval/run.py ../my-harness --env-file local.env

# Run specific tasks only
uv run python eval/run.py ../my-harness --env-file local.env --tasks jobshop-scheduler,refinery-pipeline
```

The kit builds the entry the same way the server does, reusing the last build
when nothing changed (`--rebuild` forces a rebuild). Common flags:

- `--trials N`: How many tasks to run in parallel
- `--attempts N`: Max attempts per task when an evaluator-side fault occurs (default `3`)
- `--no-export`: Leave your entry's `artifacts/` untouched
- `--out DIR`: Write the run output to a specific folder instead of `runs/<name>-<time>/`

Pressing `Ctrl-C` asks Harbor to cancel running trials and remove their
containers (up to 120 s; press `Ctrl-C` again to kill Harbor immediately).
Finished tasks stay in `results.json` and `export.json`:

- Re-grade tasks whose grader failed without re-running your harness: `uv run python eval/run.py --regrade <run> --env-file local.env`
- Copy a run's artifacts into your entry: `uv run python -m kit artifacts <entry> --from <run>`

### A model you serve yourself

`--model-base-url http://<address>:<port>/v1` overrides `model_base_url` for a local run. Inside the serve container, `localhost` in that URL is rewritten to `host.docker.internal`, so on Linux your server must listen on `0.0.0.0` (or the Docker bridge IP), not only `127.0.0.1`. Unless `JUDGE_BASE_URL` or `MODEL_PROXY_URL` is set, verifier containers also use that server as the judge. See [`docs/local-models.md`](../docs/local-models.md) for how to publish it for the evaluation server.

### Debugging against a running server

```bash
uv run python -m kit up ../my-harness           # prints the server's ws:// address
uv run python eval/run.py --url ws://127.0.0.1:<port> --name my-harness --tasks jobshop-scheduler
```

`kit up` runs your server inside its container behind the egress proxy; a
`serve.py` started directly on your host runs without either, which is handy for
attaching a debugger. With `--url`, set `JUDGE_*` or `MODEL_PROXY_*`
explicitly. Because a `--url` run does not record a harness digest in
`manifest.json`, generate the artifacts you submit with `eval/run.py <entry>`.

## Time and cost

- Each task container uses 2 to 4 CPUs and 4 to 8 GB of memory, plus 2 CPUs and
  4.25 GB for its serve and egress containers. By default, `eval/run.py` runs as
  many tasks in parallel as Docker's CPUs and memory allow, sizing each slot for
  the largest selected task (6 CPUs and 12.25 GB for the full dev set, so an
  8-CPU machine runs one task at a time). Grading then starts a verifier
  container per task. Timed verifiers (`jobshop-scheduler` gives each instance 5
  seconds) score lower when tasks contend for CPUs or run under `linux/amd64`
  emulation on Apple silicon.
- Harnesses stop at the time limit, 6 hours for every task, as on the evaluation
  server (54 hours total if run one at a time, plus setup and grading; most
  harnesses finish far sooner). On a 208-CPU machine running all nine tasks at once, a reference
  harness on Claude Opus 5.5 finished in 32 to 45 minutes of wall time (2 to 3.2
  hours of cumulative trial time).
- That reference run used about 30 million input tokens and 0.44 million output
  tokens. The judge answers 31 calls across the rubric tasks (one per criterion
  of `refinery-pipeline` and `pipeline-layout-cost`, and one for the
  `fire-test-validation-memo` rubric) plus one call per `checks-judge` task.
- Follow live progress with `tail -f runs/<run>/attempt1/harbor.log` (note that Harbor's logged "Mean" is the raw reward before scaling).

## Read the results

The summary table prints each task's normalized score from `0` to `1` (`reward /
reward_scale`), status, and harness runtime, followed by the mean. `eval/run.py`
exits with `0` on success, `1` if any task has no measurement or unusable
artifacts, or `130` when interrupted.

```
runs/<name>-<time>/              (runs/<name>-artifacts-<time>/ for --artifacts)
  results.json                   overall and per-task scores, statuses, and token usage
  export.json, artifacts/        export record and exported artifacts with manifest.json (not for --artifacts)
  tasks/                         per-run copy of each task (configured so judge verifiers can reach the judge)
  attempt1/                      retries go to attempt2/ and attempt3/
    harbor.log                   Harbor console output
    job.yaml, task-limits.yaml   Harbor job config and container resource limits
    jobs/dev/<task>__<id>/       one trial (`harbor view runs/<run>/attempt1/jobs` browses them)
      result.json                Harbor's trial record
      agent/                     <name>.txt (protocol transcript), serve.log, egress.log, trajectory.json
      artifacts/                 deliverables Harbor passed to the verifier container
      deliverables/              deliverables copied out by the kit (used by --regrade)
      verifier/                  verifier output: test-stdout.txt, grade.json, grade.log, or GRADER_FAILED.txt
  attempt1-regrade1/             grader replays for trials whose grader failed (2 and 3 if it fails again)
```

## Check your artifacts before you submit

Before submitting, run the checks in [`docs/artifacts.md`](../docs/artifacts.md#check-before-you-submit). One of them is `eval/run.py --artifacts ../my-harness`, which copies each task's committed files into a fresh task container and grades them just like the server does. Each row prints `your run scored X` alongside the new score when `manifest.json` recorded a previous score (LLM-judged scores typically vary by a few points between calls).

## How grading works

Every task is graded by Harbor in a dedicated verifier container that receives
only the task's declared deliverables (and `final_message.md` for tasks that
read it). Your harness never sees the graders.

- `checks`: Programmatic checks with no network access.
- `rubric`: An LLM judge scores the deliverable against an expert rubric.
- `checks-judge`: Programmatic checks that also query the LLM judge on parts of
  the deliverable. The grader runs as an unprivileged user and sends judge
  queries through a local relay inside the verifier container that holds the API
  key.

Verifier containers for `rubric` and `checks-judge` tasks can reach only the
judge host. `eval/run.py` passes `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and
`JUDGE_MODEL` through Harbor's environment so the key is never written to files
in the run directory.

| Trial outcome | What `eval/run.py` does |
|---|---|
| Your harness ended (`done`, `error`, `AgentTimeoutError`, or `NonZeroAgentExitCodeError`) | Measured: whatever deliverables are in the container are graded. |
| The grader failed (`GRADER_FAILED.txt`, usually the judge) | Re-graded from the snapshotted deliverables in a fresh task container up to 3 times, without re-running your harness. |
| The evaluator failed otherwise (`VerifierTimeoutError`, missing reward, or Harbor/Docker/network error) | Not measured; the task trial runs again, up to `--attempts` times. |
| Broken `task.toml` | Not measured and not re-run. |

After fixing a judge issue (such as a missing key or rate limit), re-grade the
unmeasured tasks without re-running your harness:

```bash
uv run python eval/run.py --regrade runs/<name>-<time> --env-file local.env
```

## The dev set

| Task | Discipline | Graded by | What it asks for |
|---|---|---|---|
| `jobshop-scheduler` | Industrial engineering | `checks` | Replace a naive job-shop dispatcher with a scheduler that finds short schedules for unseen 20 × 15 instances within 5 seconds per instance |
| `refinery-pipeline` | Civil engineering | `rubric` | Model and solve a station-siting and pipeline-network optimization problem, reporting the results in the final message |
| `data-hall-mep-routing` | Industrial engineering | `checks` | Reconstruct a data hall from seven drawing sheets and route cable tray, busway, and chilled-water piping clash-free as an IFC4 model generated by a script |
| `pipeline-layout-cost` | Civil engineering | `rubric` | Find the lowest-cost pipeline network for two refineries, where a line's cost per kilometer depends on which refineries it serves and urban segments carry a surcharge |
| `fire-test-validation-memo` | Mechanical engineering | `rubric` | Evaluate whether a fire-resistance simulation is validated against furnace test data and write an engineering validation memo as a Word document |
| `data-hall-mep-routing-offset` | Industrial engineering | `checks` | Route MEP systems in a harder data hall where the structural grid is offset from the floor tiles, navigating a cross-aisle and 104 sprinkler heads |
| `eor-review-corrections` | Industrial engineering | `checks-judge` | Turn Engineer-of-Record review markups into an issued drawing revision across two bid packages, including comment dispositions, a re-issue sweep, a sheet index, and change documentation in a `.zip` package |
| `foundation-strategy` | Civil engineering | `checks-judge` | Select spread footings or aggregate piers for six building pads from geotechnical records, delivering a design basis, schedule, and constraint register in a `.zip` package |
| `seismic-design-basis` | Structural engineering | `checks-judge` | Establish the seismic design basis across thirteen data-hall sites, from site class to design base shear, within a portfolio budget, packaged as a `.zip` archive |

[`dev_set.yaml`](dev_set.yaml) lists each task's time limit, deliverable paths,
and archive `sha256`. The task definitions and graders live in the Hugging Face
dataset under `tasks/<id>/`.
