# Physical-Engineering’s Last Exam

<p align="center">
  <a href="https://arxiv.org/abs/2610.30537"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="PLE-Bench paper on arXiv"></a>
  <a href="https://ple-bench.com"><img src="https://img.shields.io/badge/Website-PLE--Bench-009FDB?style=for-the-badge" alt="PLE-Bench competition and leaderboard website"></a>
  <a href="https://huggingface.co/datasets/google/physical-engineerings-last-exam"><img src="https://img.shields.io/badge/Hugging_Face-PLE--Bench-FFD21E?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="PLE-Bench dataset on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2EA44F?style=for-the-badge" alt="Licensed under the Apache 2.0 license"></a>
</p>

**Physical-Engineering’s Last Exam (PLE)** is an 85-task benchmark of
real-world engineering work across structural, civil, electrical, and
mechanical engineering. Each task runs in an isolated container and asks for a
project deliverable—a calculation package, design review, drawing, or technical
memo—graded from `0` to `1` by automated checks, an LLM rubric judge, or both.

In the [PLE competition](https://ple-bench.com), an entry is a GitHub commit containing your **harness** (the server that drives a model through each task), the Google-sponsored **model** it calls, and the **`artifacts/`** it produced on the **29 public dev tasks** ([dataset](https://huggingface.co/datasets/google/physical-engineerings-last-exam)). How to submit entries, and how they will be evaluated and ranked on the **56 hidden test tasks**, will be announced on the [competition site](https://ple-bench.com), which also hosts the benchmark leaderboard ([rules and scoring](docs/rules.md)).

## Quick start

You need [Docker](https://docs.docker.com/engine/install/) and [uv](https://docs.astral.sh/uv/).

The dev tasks are in a gated [Hugging Face dataset](https://huggingface.co/datasets/google/physical-engineerings-last-exam): accept its terms on the dataset page, then create a [read token](https://huggingface.co/settings/tokens).

```bash
# 1. Install Harbor (runs the dev tasks; the kit's agent needs websockets)
uv tool install harbor==0.20.0 --with websockets
export HF_TOKEN=<your Hugging Face read token>

# 2. Copy a starter entry into its own Git repository outside this clone
cp -r sample_submissions/pi-opus ../my-harness
git -C ../my-harness init -q

# 3. Start the local proxy for the google-cloud models (billed to your Google Cloud project)
gcloud auth application-default login
export VERTEXAI_PROJECT=<your-gcp-project> MODEL_PROXY_KEY=$(openssl rand -hex 24)
kit/model_proxy/start.sh &
export MODEL_PROXY_URL=http://localhost:4000

# 4. Build, start, test the model, and run the smoke and MCP checks
uv run python -m kit check ../my-harness

# 5. Run and score the dev set; writes ../my-harness/artifacts/
uv run python eval/run.py ../my-harness
```

See [`kit/README.md`](kit/README.md) for local checks and container limits, and
[`eval/README.md`](eval/README.md) for dev-set evaluation options and the full
list of dev tasks.

## Before you submit

1. **Commit your harness and `artifacts/`**:
   ```bash
   git -C ../my-harness add -A && git -C ../my-harness commit -m "harness and dev artifacts"
   ```
2. **Run the readiness check** and confirm it finishes with `ready to submit` (see [`docs/artifacts.md`](docs/artifacts.md) for artifact validation and offline regrading with `eval/run.py --artifacts`):
   ```bash
   uv run python -m kit check ../my-harness
   ```
3. **Push the commit to a GitHub repository.** Submission details will be announced on the [competition site](https://ple-bench.com).

## How an entry runs

Your harness runs as a WebSocket server in its own container alongside an
isolated task container:

```
 your server (serve.py)  <---- websocket ---->  evaluator  ---- runs your requests ---->  task container
         |                                                                                (project files,
         +---- egress proxy ----> your model's API, and no other host                      no network)
```

For each task (up to 6 hours), the evaluator sends the task prompt over the WebSocket, and your harness interacts with the task container via a small JSON protocol: run a shell command, read or write a file, or call one of the task's local MCP servers (documents, spreadsheets, email, CAD). See [`kit/README.md`](kit/README.md) for the container contract and [`docs/protocol.md`](docs/protocol.md) for the WebSocket protocol.

## Models

Set `model_api` in `submission.yaml` to any of the eight Google-sponsored
`google-cloud/<id>` models. Your harness reads the model's URL and key from
environment variables, so an entry contains no API keys. To run these models
locally, use [`kit/model_proxy`](docs/models.md#running-google-cloud-models-on-your-machine).
For local runs only, the kit and `eval/run.py` can also run your harness on a
commercial model with your own key or on a
[model you serve](docs/local-models.md), with the key in your environment or
`--env-file`; such an entry is not accepted for submission. See
[`docs/models.md`](docs/models.md) for the models and runtime environment
variables.

## Sample submissions

[`sample_submissions/`](sample_submissions/README.md) includes eight working
starter entries, all on Google-sponsored models except `local-model`, which is
for local runs:

- **Compact custom starters:** [`minimal`](sample_submissions/minimal/), [`plan-verify`](sample_submissions/plan-verify/), and [`node-protocol`](sample_submissions/node-protocol/) (Node.js)
- **Full-featured custom harnesses:**
  [`custom-openai`](sample_submissions/custom-openai/) and
  [`local-model`](sample_submissions/local-model/) (the same harness on a model
  you serve, local runs only)
- **Adapters for existing agents:** [`hermes-opus`](sample_submissions/hermes-opus/) (Hermes Agent), [`pi-opus`](sample_submissions/pi-opus/) (Pi), and
  [`agy-gemini`](sample_submissions/agy-gemini/) (Antigravity CLI)

## The benchmark

| Discipline | Tasks | Typical deliverables |
|---|--:|---|
| Structural engineering | 10 | Gravity, wind, and seismic load checks, member and connection checks, seismic design bases, drawing revisions from review comments |
| Civil engineering | 18 | Settlement and bearing capacity checks, foundation strategy, site layout, grading and drainage, construction sequencing and temporary works, cost estimates and fee positions |
| Electrical engineering | 27 | Load and capacity calculations, plan-review responses, energization checks, cable sizing, specifications, drawing records |
| Mechanical engineering | 30 | Cooling and controls, clash-free MEP routing, equipment equivalency, storage-tank service life, fire-test validation |

## Citation

If you use PLE in your research, please cite the
[paper](https://arxiv.org/abs/2610.30537):

```bibtex
@misc{han2026ple,
  title         = {{PLE}: Physical-Engineering's Last Exam},
  author        = {Rujun Han and Yufan Zhuang and Arijit Ray and Han Yu and
                   Zhongying CuiZhu and George Lee and Yifei Ming and Binz Roy and
                   Varun Sakalkar and Vishy Tirumalashetty and Burak Gokturk and
                   Tomas Pfister and Chen-Yu Lee},
  year          = {2026},
  eprint        = {2610.30537},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.30537}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
