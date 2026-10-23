# Dev set evaluation

`eval/run.py` runs your entry on the 29 public dev tasks, grades the results with the leaderboard's graders, and writes your harness's outputs into your entry's `artifacts/` folder, which you commit and submit ([`docs/artifacts.md`](../docs/artifacts.md)). `eval/run.py --artifacts` grades that folder offline just as the server does ([below](#check-your-artifacts-before-you-submit)).

The tasks live in the Hugging Face dataset named in
[`dev_set.yaml`](dev_set.yaml) (`tasks_repo`). Each run uses the dataset's
latest commit: `eval/run.py` downloads the tasks it needs into
`~/.cache/plebench/tasks/<commit>/` the first time it sees that commit, and
records the commit in `results.json`, so `--regrade` grades with the same
tasks.

## Before you run

What you need:

- Docker running locally with about 10 GB of free disk space for base images, plus room for the world images and the CPUs and memory described in [Time and cost](#time-and-cost).
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
  option 3 is used). The leaderboard always grades with Claude Opus 5.5, which
  other judge models only approximate. When calling `MODEL_PROXY_URL` (or a
  `JUDGE_BASE_URL` pointing to the same server), `eval/run.py` requests
  `<model>-judge` so `kit/model_proxy` uses the model's standard default
  reasoning effort—matching the evaluation server's judge—rather than the
  maximum effort used for harnesses. (If you point `JUDGE_BASE_URL` at
  `kit/model_proxy` without setting `MODEL_PROXY_URL`, set
  `JUDGE_MODEL=claude-opus-5-5-judge` explicitly.) If Claude Opus 5.5 refuses a
  judge request, `kit/model_proxy` sends that request once to Claude Opus 5 and
  returns its reply, as the evaluation server's judge does. Harness requests are
  never re-sent.
- Any local judge on `localhost` must be reachable from inside Docker
  containers. On Linux, verifier containers connect via Docker's bridge IP, so a
  server bound only to `127.0.0.1` will not be reachable (`kit/model_proxy/start.sh`
  binds to `0.0.0.0` by default unless `MODEL_PROXY_HOST` overrides it).
  On Linux, when the selected tasks include judge tasks and the judge runs on
  this machine, `eval/run.py` checks before building that the judge's port
  accepts a TCP connection at Docker's bridge address, and stops if it does not.
- Internet access to `huggingface.co` on first run, and read access to
  [the dataset](https://huggingface.co/datasets/physical-engineerings-last-exam/physical-engineerings-last-exam).
  Request access on the dataset page while signed in to Hugging Face, then set
  `HF_TOKEN` to a read token from that account (or run `uv run hf auth login`
  once). You can also pass `--tasks-dir DIR` to use pre-downloaded task folders,
  which are used as they are. Without internet access, a run uses the newest
  download it finished.

Keep local proxy, judge, and Hugging Face settings in an env file outside your
entry repository and pass it with `--env-file` (keep only the keys your harness
uses in your entry's `.env`):

```bash
# local.env
MODEL_PROXY_URL=http://localhost:4000
MODEL_PROXY_KEY=<the key you gave kit/model_proxy/start.sh>
# JUDGE_BASE_URL=https://<an OpenAI-compatible endpoint>/v1
# JUDGE_API_KEY=<its key>
# HF_TOKEN=<a Hugging Face token, if needed>
```

### Platform notes

- **macOS and Windows:** Docker Desktop runs containers in a VM with its own CPU, memory, and disk limits (Settings > Resources). Give it at least 6 CPUs, 16.25 GB of memory (enough for the largest dev task plus its serve stack; see [Time and cost](#time-and-cost)), and enough disk space for the task and world images.
- **Windows:** Run everything inside WSL 2. Clone this repository and keep your entry inside the Linux filesystem (under `~`, not `/mnt/c`), using WSL's `git`, `uv`, and Python. Repositories cloned with Git for Windows or stored on `/mnt/c` often set `core.symlinks=false` (which `kit check` refuses) and `core.autocrlf=true` (which converts shell scripts to CRLF line endings that break under Linux `bash`). In your entry's repository, set `git config core.autocrlf input` or add `* text=auto eol=lf` to `.gitattributes`.

## Run the dev set

From the root of this repository (with your entry in its own separate Git repository; see [Quick start](../README.md#quick-start)):

```bash
# Run all 29 dev tasks
uv run python eval/run.py ../my-harness --env-file local.env

# Run specific tasks only
uv run python eval/run.py ../my-harness --env-file local.env --tasks data-center-site-model,engineers-estimate-check
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

- Re-grade tasks whose judge grading failed without re-running your harness: `uv run python eval/run.py --regrade <run> --env-file local.env`
- Copy a run's artifacts into your entry: `uv run python -m kit artifacts <entry> --from <run>`

### A model you serve yourself

`--model-base-url http://<address>:<port>/v1` overrides `model_base_url` for a local run. Inside the serve container, `localhost` in that URL is rewritten to `host.docker.internal`, so on Linux your server must listen on `0.0.0.0` (or the Docker bridge IP), not only `127.0.0.1`. Unless `JUDGE_BASE_URL` or `MODEL_PROXY_URL` is set, verifier containers also use that server as the judge, and reach it as the Docker bridge IP on Linux or `host.docker.internal` on macOS. See [`docs/local-models.md`](../docs/local-models.md) for how to publish it for the evaluation server.

### Debugging against a running server

```bash
uv run python -m kit up ../my-harness           # prints the server's ws:// address
uv run python eval/run.py --url ws://127.0.0.1:<port> --name my-harness --tasks data-center-site-model
```

`kit up` runs your server inside its container behind the egress proxy; a
`serve.py` started directly on your host runs without either, which is handy for
attaching a debugger. With `--url`, `--name` and `--model` say what to call the
harness and its model in the results, and set `JUDGE_*` or `MODEL_PROXY_*`
explicitly. Because a `--url` run does not record a harness digest in
`manifest.json`, generate the artifacts you submit with `eval/run.py <entry>`.

## Time and cost

- Each task container uses 2 to 4 CPUs and 4 to 12 GB of memory, plus 2 CPUs
  and 4.25 GB for its serve and egress containers. By default, `eval/run.py`
  runs as many tasks in parallel as Docker's CPUs and memory allow, sizing each
  slot for the largest selected task (6 CPUs and 16.25 GB for the full dev set,
  so an 8-CPU machine runs one task at a time). Grading then starts a verifier
  container per task. Timed verifiers score lower when tasks contend for CPUs
  or run under `linux/amd64` emulation on Apple silicon.
- 12 rubric tasks run inside world environments that serve project
  applications (email, documents, spreadsheets, CAD) over MCP. Their Docker
  images are large (about 17 GB) and take a while to build, so `eval/run.py`
  pre-builds them once before starting the run. Each image gets at least an
  hour to build locally (overriding any shorter `task.toml` timeout meant for
  the evaluation server's warm cache). If an image build times out, that task is
  marked unmeasured while the remaining tasks still run, and concurrent
  `eval/run.py` processes share a single image build.
- Harnesses stop at the time limit, 6 hours for every task, as on the evaluation
  server (174 hours total if run one at a time, plus setup and grading; most
  harnesses finish far sooner).
- The judge answers one call per rubric task (or one per rendered document
  group when a rubric inspects rendered pages) and a few targeted calls per
  `checks-judge` task, plus retries of calls that fail.
- Follow live progress with `tail -f runs/<run>/attempt1/harbor.log` (note that Harbor's logged "Mean" is the raw reward before scaling).

## Read the results

The summary table prints each task's normalized score from `0` to `1` (`reward /
reward_scale`), status, and harness runtime, followed by the mean. `eval/run.py`
exits with `0` on success, `1` if any task has no measurement (it counts `0`,
so the mean is a lower bound) or unusable artifacts, or `130` when interrupted.

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
  attempt1-regrade1/             grader replays for trials whose judge grading failed (2 and 3 if it fails again)
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
| Your harness exited with an error before its model produced any output (zero output tokens reported, or, when none are reported, no tool or model activity in its trajectory) | Not started: run again, up to `--attempts` runs in all, or marked unmeasured (counting `0`) if it never starts. Exception: a harness that ended on a context-length overflow or a provider's limit (such as a tool-count limit) is graded as-is, matching the evaluation server. |
| A `rubric` or `checks-judge` grader failed (`GRADER_FAILED.txt` or `VerifierTimeoutError`) | Re-graded from the snapshotted deliverables in a fresh container up to 3 times without re-running your harness; marked unmeasured if all 3 fail. |
| Any other evaluator fault (`checks` grader failure, missing reward, Harbor/Docker/network error, or a harness cut off by an endpoint/network error, even when its grader also failed) | Not measured: the full task trial runs again, up to `--attempts` runs in all. |
| Broken `task.toml`, or a task image that did not build | Not measured and not re-run. |

After fixing a judge issue (such as a missing key or rate limit), re-grade the
tasks whose judge grading failed without re-running your harness (any other
unmeasured task needs a new run):

```bash
uv run python eval/run.py --regrade runs/<name>-<time> --env-file local.env
```

`--regrade` grades with the exact task definitions recorded in the run's
`results.json`: it downloads that dataset revision again if your checkout has
since moved to a newer one, or checks that your `--tasks-dir` folders have not
changed.

## The dev set

| Task | Discipline | Graded by | What it asks for |
|---|---|---|---|
| `data-hall-mep-routing` | Mechanical engineering | `checks` | Reconstruct a data hall from seven drawing sheets and generate a clash-free IFC4 model routing cable tray, busway, and chilled-water piping |
| `fire-test-validation-memo` | Mechanical engineering | `rubric` | Evaluate a fire-resistance simulation against furnace test data and write an engineering validation memo in Word |
| `data-hall-mep-routing-offset` | Mechanical engineering | `checks` | Route MEP systems in a data hall where the structural grid is offset from the floor tiles, navigating a cross-aisle and 104 sprinkler heads |
| `eor-review-corrections` | Structural engineering | `checks-judge` | Turn engineer-of-record review markups into an issued drawing revision across two bid packages, including comment dispositions, a sheet index, and change documentation |
| `foundation-strategy` | Civil engineering | `checks-judge` | Choose between spread footings and aggregate piers for six building pads using geotechnical records, and deliver a design basis, schedule, and constraint register |
| `landscape-sheet-revision` | Civil engineering | `rubric` | Incorporate a street-tree comment into a landscape drawing set and verify the package for agency resubmittal |
| `data-center-site-model` | Civil engineering | `checks` | Lay out a two-phase data-center campus that meets every binding requirement in its approval document, as an IFC model |
| `engineers-estimate-check` | Civil engineering | `rubric` | Audit a section of an engineer's estimate against the project plans before it goes to the surety broker |
| `footing-settlement-verification` | Civil engineering | `rubric` | Verify the long-term settlement of a building block's spread footings on clay and issue the verification memo |
| `solar-supplemental-review` | Electrical engineering | `rubric` | Write the supplemental interconnection review memo for a solar application on a distribution feeder |
| `solar-witness-test-package` | Electrical engineering | `rubric` | Prepare the witness-test scope and schedule for municipal solar arrays that require testing, and list those that do not |
| `frame-extension-method` | Civil engineering | `checks` | Prepare a construction method package for a steel frame extension over an occupied floor, verified against safety regulations |
| `fire-solve-convergence` | Mechanical engineering | `rubric` | Provide a signed engineering disposition on a non-converging steady-state fire simulation before a fourth run is attempted |
| `les-mesh-resolution` | Mechanical engineering | `rubric` | Document the mesh-resolution basis of a large-eddy fire simulation to close a peer-review item |
| `mv-conveyance-study` | Electrical engineering | `checks` | Produce a conveyance study and cost workbook for seven 13.8 kV medium-voltage distribution circuits |
| `data-hall-device-takeoff` | Electrical engineering | `checks` | Complete an electrical device takeoff from a data hall drawing set and price the bill of materials |
| `division-26-specifications` | Electrical engineering | `checks` | Write Division 26 electrical specifications for an Army facility from its coordinated design package |
| `heat-rejection-life-cycle-cost` | Mechanical engineering | `checks` | Design a condenser-water heat rejection replacement for a federal campus and justify it with a life-cycle cost analysis |
| `cooling-water-flush-plan` | Mechanical engineering | `checks` | Create a cleaning, flushing, and passivation plan for a data hall's process cooling-water system prior to turnover |
| `wind-diaphragm-shear` | Structural engineering | `checks` | Calculate wind roof-diaphragm shears and shear-wall loads for a single-story data hall |
| `seismic-diaphragm-shear` | Structural engineering | `checks` | Calculate seismic diaphragm shears and shear-wall loads along a data-center building's lateral load path |
| `roof-deck-design` | Structural engineering | `checks` | Design a data center's steel roof deck for gravity and lateral loads, specifying gage, fastener pattern, and diaphragm capacity |
| `fill-compaction-acceptance` | Civil engineering | `rubric` | Decide whether slab-on-grade fill compaction testing and its records meet the earthwork specification, as a materials acceptance record |
| `shoring-design-verification` | Civil engineering | `rubric` | Verify a revised excavation shoring design against the logged groundwater and subsurface conditions before the excavation starts |
| `emergency-spillway-check` | Civil engineering | `rubric` | Check whether a stormwater basin's emergency spillway as drawn can answer a plan-check comment, and set out the revisions in a memo |
| `pre-application-capacity-response` | Electrical engineering | `rubric` | Write a utility's response to a pre-application request for a feeder's available capacity, and close the case in its tracking |
| `permission-to-operate-memo` | Electrical engineering | `rubric` | Advise whether permission to operate can issue for a PV site on the requested date, and open a work package for what stands in the way |
| `backfeed-energization-reply` | Electrical engineering | `rubric` | Reply to a community solar developer on whether its site can be energized for backfeed on the requested date, and open a work package |
| `lv-cable-sizing` | Electrical engineering | `checks-judge` | Size a building's low-voltage cable schedule for voltage drop, duct-bank ampacity, and cost, with a constraint register and design basis |

[`dev_set.yaml`](dev_set.yaml) lists each task's time limit (`time_limit_min`,
360 minutes for every task, matching `task.toml`, `eval/run.py`, and the
evaluation server), and deliverable paths. The task
definitions and graders live in the Hugging Face dataset under `tasks/<id>/`.
