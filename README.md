# PLEBench: Physical-Engineering’s Last Exam Harness Competition

Physical-Engineering’s Last Exam (PLE) is a benchmark of real-world engineering work. Each task asks for a deliverable an engineer produces on a project: a calculation package, a design review, a drawing, or a technical memo. Domain experts reviewed every task, grading rubric, and automated check.

You enter with a harness (the program that drives a model through a task), the
model it calls, and the `artifacts/` it produced on the public dev set. For each
submission, we:

1. Grade your dev-set artifacts to get your dev score.
2. Run your harness on a few dev tasks to check that it reproduces your artifacts' scores.
3. Run your harness on the hidden test set, which ranks your entry on the leaderboard.

This repository has the tools to build, test, and score an entry on your
machine:

- [`sample_submissions/`](sample_submissions/) holds eight starter entries.
- [`kit/`](kit/) (`python -m kit check`) runs your entry through the evaluation server's build and smoke-test steps.
- [`eval/`](eval/) (`python eval/run.py`) scores your entry on the dev set and exports the `artifacts/` folder you submit.

## Quick start

Prerequisites:

- Linux, macOS, or Windows with WSL 2
- Git, Python 3.10+, and [uv](https://docs.astral.sh/uv/)
- A running Docker daemon with sufficient disk, CPU, and memory for a dev run ([eval/README.md](eval/README.md#before-you-run))
- Internet access to Hugging Face for the dev-set tasks (set `HF_TOKEN` while the dataset is private)
- An OpenAI API key for `custom-openai` below (whose model also serves as the local judge), or the local proxy setup in [Models](#models) if you start from [`minimal`](sample_submissions/minimal/) with a Google-sponsored model

Containers run as `linux/amd64` (matching the evaluation server), so Docker emulates `amd64` on Apple silicon. Emulation runs slower, which can reduce scores on timed verifiers compared to the leaderboard ([Time and cost](eval/README.md#time-and-cost)).

```bash
# Install Harbor (runs the dev tasks; the kit's agent needs websockets)
uv tool install harbor==0.20.0 --with websockets

# Copy a starter entry into its own Git repository outside this clone
cp -r sample_submissions/custom-openai ../my-harness
git -C ../my-harness init -q

# Put your OpenAI API key in ../my-harness/.env
cp ../my-harness/.env.example ../my-harness/.env

# Build, test the model, and run the smoke and MCP tasks
uv run python -m kit check ../my-harness

# Run and score the dev set; writes ../my-harness/artifacts/
uv run python eval/run.py ../my-harness
```

Keep your entry outside this clone as its own Git repository. This repository's
`.gitignore` ignores `.env`, and `kit check` requires your entry to sit at the
root of its own repository.

## Before you submit

[docs/artifacts.md](docs/artifacts.md#check-before-you-submit) explains each check in detail:

1. Check the exported artifacts with
   `uv run python -m kit artifacts ../my-harness`. Make sure there are no
   `REFUSED` lines and it prints `artifacts/: 9 of 9 dev tasks usable` (any dev
   task without usable artifacts scores `0`).
2. Optionally, grade `artifacts/` the way the server does without running your harness: `uv run python eval/run.py --artifacts ../my-harness`.
3. Commit everything, including `artifacts/` and `.env`:
   ```bash
   git -C ../my-harness add -A && git -C ../my-harness commit -m "harness and dev artifacts"
   ```
   If your global gitignore ignores `.env`, force-add it with
   `git -C ../my-harness add -f .env`.
4. Run `uv run python -m kit check ../my-harness` and confirm it ends with `ready to submit`.
5. Push the commit to a GitHub repository owned by your account. Any repository with a `.env` file must be private and have the competition's GitHub App installed (linked on the competition site).
6. Submit the commit on the competition site.

## How an entry runs

An entry is a commit in a GitHub repository with `submission.yaml`, your harness, the exported `artifacts/`, and a `.env` file if your harness needs API keys ([kit/README.md](kit/README.md#what-you-submit)).

Your harness is a WebSocket server. For each task, we start it in its own
container next to an isolated task container:

```
 your server (serve.py)  <---- websocket ---->  evaluator  ---- runs your requests ---->  task container
         |                                                                                (project files,
         +---- egress proxy ----> your model's API, and no other host                      no network)
```

The evaluator sends the task over the WebSocket. Your harness works inside the task container through a small JSON protocol: run a shell command, read or write a file, or call one of the task's local MCP servers. See [`kit/README.md`](kit/README.md) for the container contract and [`docs/protocol.md`](docs/protocol.md) for the protocol.

## Models

Your harness can call one of thirteen Google-sponsored `google-cloud/<id>` models, a provider's model on your own API key, or a model server you run yourself. [`docs/models.md`](docs/models.md) lists all supported models and the environment variables passed to your server.

The sponsored models are free on our servers. To run the dev set on your machine with a `google-cloud/<id>` model, run `kit/model_proxy` against your own Vertex AI project (billed to that project) or use another proxy that serves the same model names ([how](docs/models.md#running-google-cloud-models-on-your-machine)). [`docs/local-models.md`](docs/local-models.md) shows how to serve a model from your own GPUs to the evaluation server.

## Scoring

Each task scores from `0` to `1`. A submission's score is the mean across the
tasks of the test set, where a task without a result counts as `0`.

The dev set is published and you run it yourself; the test set is hidden, and the evaluation server runs every scored submission once on it. The test score ranks the leaderboard, and when the competition closes, the leaderboard is the final standings. Your dev score and the reproduction check do not affect your leaderboard rank. See [`docs/rules.md`](docs/rules.md) for the full rules.

## The benchmark

| Discipline | Tasks | Typical deliverables |
|---|--:|---|
| Structural engineering | 9 | Gravity, wind, and seismic load checks, member and connection checks, seismic design bases, temporary structures |
| Civil engineering | 10 | Settlement and bearing capacity checks, foundation strategy, site and pipeline layout, cost estimates |
| Electrical engineering | 22 | Load and capacity calculations, plan-review responses, energization checks, cable sizing, specifications |
| Mechanical engineering | 16 | Cooling and controls, equipment equivalency, storage-tank service life, fire-test validation |
| Industrial engineering | 10 | Clash-free MEP routing, job-shop schedules, drawing revisions and review comments, scope and change claims |

Each task runs in an isolated Linux container with the project's files and no
internet access. Some tasks also serve project applications (documents,
spreadsheets, email, CAD) as local MCP tool servers. The harness writes its
deliverables to disk and finishes with a final message. The harness has 6 hours
for each task, and tasks are graded by programmatic checks, an LLM judge using
an expert rubric, or both.

The nine dev-set tasks are in a Hugging Face dataset that `eval/run.py` downloads automatically ([`eval/README.md`](eval/README.md#the-dev-set)). The leaderboard runs on the hidden test set.

## Sample submissions

[`sample_submissions/`](sample_submissions/README.md) has eight working entries:

- Short custom starters: [`minimal`](sample_submissions/minimal/), [`plan-verify`](sample_submissions/plan-verify/), and [`node-protocol`](sample_submissions/node-protocol/) (JavaScript)
- Full custom harnesses: [`custom-openai`](sample_submissions/custom-openai/) and [`local-model`](sample_submissions/local-model/) (for a model you serve yourself)
- Wrappers around existing agents: [`hermes-opus`](sample_submissions/hermes-opus/) (Hermes Agent), [`pi-opus`](sample_submissions/pi-opus/) (Pi), and [`agy-gemini`](sample_submissions/agy-gemini/) (Antigravity CLI)

## Repository layout

```
README.md
docs/
  protocol.md            the serve protocol: messages, limits, timeouts, faults
  models.md              model_api, server environment variables, running google-cloud models locally
  local-models.md        serving a model from your own GPUs to the evaluation server
  artifacts.md           artifacts/: export, layout, server rules, pre-submit checks
  rules.md               the dev and test sets, submissions, scoring, reproduction check, fair play
kit/                     python -m kit: check, build, and start an entry on your machine
  README.md              container contract, limits, and kit commands
  model_proxy/           LiteLLM proxy serving google-cloud models from your Vertex AI project
  builder/Dockerfile     image that builds and runs your harness
eval/                    python eval/run.py: run and grade the dev set
  dev_set.yaml           dev task time limits, deliverables, and dataset reference
sample_submissions/      eight working entries (README.md compares them)
  taskbridge.py          serve protocol helper for Python harnesses
  minimal/  plan-verify/  node-protocol/  custom-openai/  local-model/
  hermes-opus/  pi-opus/  agy-gemini/
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/); see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

All software (the kit, sample submissions, and evaluation scripts) is licensed
under the Apache License, Version 2.0 (Apache 2.0); see [`LICENSE`](LICENSE) or
https://www.apache.org/licenses/LICENSE-2.0. All other materials, including the
documentation and the dev-set tasks (in the Hugging Face dataset named in
`eval/dev_set.yaml`), are licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/legalcode. Unless required by
applicable law or agreed to in writing, all software and materials distributed
here under the Apache 2.0 or CC BY 4.0 licenses are distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the licenses for the specific language governing permissions and limitations
under those licenses. Please keep the dev-set tasks out of training data for
models that will be evaluated on this benchmark.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
