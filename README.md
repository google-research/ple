# Physical-Engineering’s Last Exam

**Physical-Engineering’s Last Exam (PLE)** is a benchmark of real-world engineering work. Each task asks for a deliverable an engineer produces on a project: a calculation package, a design review, a drawing, or a technical memo. Domain experts reviewed every task, grading rubric, and automated check.

In the PLE harness competition, you enter with a **harness** (the server that drives a model through each task), the **model** it calls, and the **`artifacts/`** it produced on the public dev set. For each submission, we:

1. **Grade your committed `artifacts/`** to compute your dev score.
2. **Run your harness live** on a sample of dev tasks to verify that your dev score reproduces.
3. **Evaluate your harness on the hidden test set**, which determines your leaderboard rank.

This repository provides everything you need to build, test, and score an entry locally:

- [`sample_submissions/`](sample_submissions/): eight working starter entries
- [`kit/`](kit/): `python -m kit check` validates, builds, and smoke-tests your harness the way the evaluation server does
- [`eval/`](eval/): `python eval/run.py` runs the dev set, grades the results, and exports `artifacts/`

## Quick start

**Prerequisites:**

- Linux, macOS, or Windows with WSL 2
- Git, Python 3.10+, and [uv](https://docs.astral.sh/uv/)
- A running Docker daemon with sufficient disk, CPU, and memory ([`eval/README.md`](eval/README.md#before-you-run))
- Internet access to Hugging Face for the dev-set tasks (set `HF_TOKEN` while the dataset is private)
- An OpenAI API key for `custom-openai` below (whose model also serves as the local judge), or the local proxy setup in [Models](#models) if you start from [`minimal`](sample_submissions/minimal/) with a Google-sponsored model

> **Note:** Containers run as `linux/amd64` to match the evaluation server. On Apple Silicon, Docker emulates `amd64`, which runs slower and can affect timed verifiers ([Time and cost](eval/README.md#time-and-cost)).

```bash
# 1. Install Harbor (runs the dev tasks; the kit's agent needs websockets)
uv tool install harbor==0.20.0 --with websockets

# 2. Copy a starter entry into its own Git repository outside this clone
cp -r sample_submissions/custom-openai ../my-harness
git -C ../my-harness init -q

# 3. Add your OpenAI API key to ../my-harness/.env
cp ../my-harness/.env.example ../my-harness/.env

# 4. Build, start, test the model, and run the smoke and MCP checks
uv run python -m kit check ../my-harness

# 5. Run and score the dev set; writes ../my-harness/artifacts/
uv run python eval/run.py ../my-harness
```

Keep your entry in its own Git repository outside this clone: this repository's `.gitignore` excludes `.env`, and `kit check` requires your entry to sit at the root of its own repository.

## Before you submit

See [`docs/artifacts.md`](docs/artifacts.md#check-before-you-submit) for details on each check:

1. **Validate exported artifacts:** Run `uv run python -m kit artifacts ../my-harness` and confirm there are no `REFUSED` lines (ideally `artifacts/: 30 of 30 dev tasks usable`; any dev task without usable artifacts scores `0`).
2. **Optional offline regrade:** Run `uv run python eval/run.py --artifacts ../my-harness` to grade `artifacts/` directly without re-running your harness.
3. **Commit all files**, including `artifacts/` and `.env`:
   ```bash
   git -C ../my-harness add -A && git -C ../my-harness commit -m "harness and dev artifacts"
   ```
   If your global gitignore excludes `.env`, force-add it with `git -C ../my-harness add -f .env`.
4. **Run the final readiness check:** Run `uv run python -m kit check ../my-harness` and confirm it finishes with `ready to submit`.
5. **Push to GitHub:** Push the commit to a GitHub repository owned by your personal account. Any repository with a `.env` file must be private and have the competition's GitHub App installed (linked on the competition site).
6. **Submit:** Submit the commit on the competition site.

## How an entry runs

An entry is a single commit in a GitHub repository containing `submission.yaml`, your harness code, the exported `artifacts/`, and an optional `.env` file at its root ([`kit/README.md`](kit/README.md#what-you-submit)).

Your harness runs as a WebSocket server in its own container alongside an isolated task container:

```
 your server (serve.py)  <---- websocket ---->  evaluator  ---- runs your requests ---->  task container
         |                                                                                (project files,
         +---- egress proxy ----> your model's API, and no other host                      no network)
```

For each task, the evaluator sends the task prompt over the WebSocket, and your harness interacts with the task container through a small JSON protocol: run a shell command, read or write a file, or call one of the task's local MCP servers. See [`kit/README.md`](kit/README.md) for the container contract and [`docs/protocol.md`](docs/protocol.md) for the WebSocket protocol.

## Models

Your harness can use any of the thirteen Google-sponsored `google-cloud/<id>` models, a commercial model with your own API key, or a self-hosted model server. [`docs/models.md`](docs/models.md) lists all supported configurations and the environment variables passed to your container.

- **Google-sponsored models:** Run free of charge on the evaluation server. To run the dev set locally with a `google-cloud/<id>` model, point `kit/model_proxy` at your own Vertex AI project (billed to that project) or use another proxy that serves the same model names ([setup instructions](docs/models.md#running-google-cloud-models-on-your-machine)).
- **Self-hosted models:** To serve a model from your own GPUs to the evaluation server, see [`docs/local-models.md`](docs/local-models.md).

## Scoring

Each task is scored from `0` to `1`. Your submission's test score is the mean across all **37 hidden test tasks** (any task without a grade scores `0`).

Each scored submission runs once on the hidden test set, and the leaderboard ranks participants by their best test score. Your dev score and reproduction check are shown for reference and do not affect your leaderboard rank. Full details are in [`docs/rules.md`](docs/rules.md).

## The benchmark

| Discipline | Tasks | Typical deliverables |
|---|--:|---|
| Structural engineering | 10 | Gravity, wind, and seismic load checks, member and connection checks, seismic design bases, drawing revisions from review comments |
| Civil engineering | 15 | Settlement and bearing capacity checks, foundation strategy, site and pipeline layout, construction sequencing and temporary works, cost estimates and fee positions |
| Electrical engineering | 25 | Load and capacity calculations, plan-review responses, energization checks, cable sizing, specifications, drawing records |
| Mechanical engineering | 17 | Cooling and controls, clash-free MEP routing, equipment equivalency, storage-tank service life, fire-test validation |

Each task runs inside an isolated Linux container preloaded with the project's files and no internet access. Where needed, tasks also run local MCP tool servers (documents, spreadsheets, email, CAD). The harness writes its deliverable files to disk, finishes with a final message, and has up to 6 hours per task. Tasks are graded by automated checks, an LLM rubric judge, or both.

The **30 public dev tasks** are downloaded automatically from Hugging Face by `eval/run.py` ([`eval/README.md`](eval/README.md#the-dev-set)); the remaining **37 tasks** form the hidden test set, and the full benchmark covers all **67 tasks**.

## Sample submissions

[`sample_submissions/`](sample_submissions/README.md) includes eight working starter entries:

- **Compact custom starters:** [`minimal`](sample_submissions/minimal/), [`plan-verify`](sample_submissions/plan-verify/), and [`node-protocol`](sample_submissions/node-protocol/) (Node.js)
- **Full-featured custom harnesses:** [`custom-openai`](sample_submissions/custom-openai/) and [`local-model`](sample_submissions/local-model/) (for self-hosted endpoints)
- **Adapters for existing agents:** [`hermes-opus`](sample_submissions/hermes-opus/) (Hermes Agent), [`pi-opus`](sample_submissions/pi-opus/) (Pi), and [`agy-gemini`](sample_submissions/agy-gemini/) (Antigravity CLI)

## Repository layout

```
README.md
docs/
  protocol.md            WebSocket protocol: messages, limits, timeouts, and fault handling
  models.md              model_api options, runtime environment variables, and local proxy setup
  local-models.md        serving a self-hosted model from your own GPUs
  artifacts.md           artifacts/ layout, validation rules, and pre-submission checks
  rules.md               competition rules, scoring, reproduction checks, and fair play
kit/                     python -m kit: validate, build, and test an entry locally
  README.md              container contract, resource limits, and CLI commands
  model_proxy/           LiteLLM proxy for running google-cloud models via Vertex AI
  builder/Dockerfile     base image used to build and run harnesses
eval/                    python eval/run.py: run and grade the dev set
  dev_set.yaml           dev-set task list, deliverables, standalone timeouts, and dataset revision
sample_submissions/      eight working starter entries (see sample_submissions/README.md)
  taskbridge.py          Python helper implementing the WebSocket protocol and local subprocess relay
  minimal/  plan-verify/  node-protocol/  custom-openai/  local-model/
  hermes-opus/  pi-opus/  agy-gemini/
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
