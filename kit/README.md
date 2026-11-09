# The submission kit

`python -m kit check` runs your entry through the evaluation steps on your own machine: it builds your bundle in the builder image, starts your WebSocket server behind the egress proxy, tests model connectivity, runs a smoke task and an MCP task, and reviews your `artifacts/`. The [last section](#what-kit-check-does-not-cover) lists what `kit check` does not cover.

## What you submit

An entry is a single commit of a GitHub repository. How to submit it will be
announced on the [competition site](https://ple-bench.com).

At the root of your repository:

```
submission.yaml      how to install and start your harness, and which model it calls
serve.py             your harness: a WebSocket server (any language; `serve` runs the command you give)
artifacts/           what your harness produced on the dev set, exported by eval/run.py (docs/artifacts.md)
...                  anything else it needs: requirements.txt, prompts, plugins, a vendored agent
```

`artifacts/` is graded separately ([`docs/artifacts.md`](../docs/artifacts.md)). The build leaves it out, along with any top-level folder named `artifact` or `artifacts` (in any letter case, with or without a `dev`, `dev-`, or `dev_` prefix) and `runs/<run>/artifacts/`. Do not put harness code in folders with those names.

### `submission.yaml`

```yaml
name: my-harness                          # leaderboard entry name: 2-39 characters of a-z, 0-9, and -
model_api: google-cloud/claude-opus-5-5   # the model your harness calls (docs/models.md)
install: >-                               # optional: runs once at build time as root, with internet access
  uv venv --python 3.12 /opt/harness/venv
  && uv pip install --python /opt/harness/venv/bin/python -r {src}/requirements.txt
serve: /opt/harness/venv/bin/python {src}/serve.py    # required: starts your WebSocket server on $PORT
env:                                      # optional: non-secret settings, as quoted strings
  LOG_LEVEL: "info"
```

| Field | Rules |
|---|---|
| `name` | 2 to 39 characters of `a-z`, `0-9`, and `-`, starting with a letter or digit |
| `model_api` | `google-cloud/<id>`, one of the Google-sponsored models ([`docs/models.md`](../docs/models.md)). `openai/<model>`, `anthropic/<model>`, `gemini/<model>`, and `openai-compatible/<model>` with a public HTTPS `model_base_url` run locally only ([your own model](../docs/models.md#your-own-model-local-runs-only)) |
| `install` | Shell command run during the build. `{src}` becomes `/opt/harness/src` (`${src}` is left to the shell) |
| `serve` | Shell command run for each task. `{src}` and `{port}` are filled in (the port is also in `$PORT`) |
| `env` | Quoted `NAME: "value"` pairs for non-secret settings. Never put a key here: the competition takes none |

`env` cannot override variables set by the evaluator: `PORT`, `HOME`, `USER`, the proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `NODE_USE_ENV_PROXY`), or the model variables in [`docs/models.md`](../docs/models.md#what-your-server-receives). Unknown fields are refused, and the file may be at most 16 KiB.

## What runs where

```
  build container (once)              serve container (one per task)          task container (one per task)
  debian:bookworm + uv + Node 22      builder image + your /opt/harness       the task's files and tools
  internet access, no task data       your server listening on $PORT          no network access
  install runs as root                internet only via the egress proxy      your requests run here
          |                                    ^         |
          +--> /opt/harness (the bundle) ------+         +-- websocket <--> evaluator --> exec / read / write / http
```

### Build

The build clones your repository at the submitted commit, including `.git` (so `setuptools-scm` and `hatch-vcs` can read a version) and submodules, into a clean `debian:bookworm` container built from [`builder/Dockerfile`](builder/Dockerfile). The image includes `uv` (`/opt/harness/bin/uv`), Node 22 (`/opt/harness/node`), `git`, `curl`, compilers, and `python3`.

- `install` runs as root with internet access for up to 30 minutes, and no single file it writes may exceed 8 GiB. Afterward, `.git` is removed and only `/opt/harness` is packed into the bundle.
- Install everything under `/opt/harness`. System packages installed via
  `apt-get` do not carry over to the serve container. Create Python environments
  with `uv venv --python 3.12 /opt/harness/venv` (which places Python under
  `/opt/harness/python`); do not use `python3 -m venv`, which symlinks to
  Debian's Python outside `/opt/harness`.
- The clone is anonymous and does not run Git LFS or carry cloud credentials. Submodules must use public `https://` URLs, and large files must be committed directly or downloaded in `install`.
- Locally, `kit` stages your working tree (tracked and untracked files not ignored by git, preserving `0755`/`0644` file modes) and builds for `linux/amd64` on your local Docker daemon.

### Serve

For each task, the evaluator starts a fresh `linux/amd64` container from the
builder image with your `/opt/harness` bundle mounted, and runs your `serve`
command in `/opt/harness/src` as an unprivileged user (`uid 1000`) with a
read-only root filesystem. The container gets 2 CPUs, 4 GB of memory, up to 512
processes, and 1 GB each of writable `/tmp` and `$HOME`. Your server must
accept a WebSocket connection on `0.0.0.0:$PORT` within 120 seconds.

Outbound traffic goes strictly through the egress proxy (`HTTPS_PROXY`,
`HTTP_PROXY`, `ALL_PROXY`, and `NODE_USE_ENV_PROXY=1`), which connects only
to your model's API host over TLS (with matching SNI). Web search and external
APIs are blocked, and connections idle for 30 minutes are closed. Provider-side
tools (Gemini Google Search, URL context, and code execution; Claude web search,
web fetch, and code execution; OpenAI `web_search_options` and hosted tools)
are also forbidden; for `google-cloud` models, both the model gateway and
`kit/model_proxy` reject them with HTTP `400`. On Docker 28+, the serve
container is also blocked from reaching host services on your machine (older
Docker versions may still allow connections to host services on `0.0.0.0` via
the bridge gateway, which evaluation runs block).

### Task

The evaluator connects to your server and sends the task. Your harness sends
requests to run shell commands, read and write files, and call the task's local
MCP servers inside the task container (which has no internet access). When the
session ends, the files at the task's declared deliverable paths and your final
message are graded. Graders receive only those paths as regular files: symlinks
or helper files left outside the deliverable paths are not graded.

[`docs/protocol.md`](../docs/protocol.md) defines the protocol messages, [`docs/models.md`](../docs/models.md#what-your-server-receives) lists the injected model environment variables, and [`sample_submissions/taskbridge.py`](../sample_submissions/taskbridge.py) implements the protocol for Python harnesses.

## Resource and protocol limits

| What | Limit |
|---|---|
| Server startup | WebSocket accepted on `$PORT` within 120 s |
| Serve container | 2 CPUs, 4 GB memory, 512 processes, 1 GB `/tmp`, 1 GB `$HOME` |
| Build | 30 minutes; `/opt/harness` up to 2 GiB packed and 8 GiB unpacked; no file over 8 GiB; no filename or symlink target over 1,024 chars; repository with `.git` history up to 1 GiB |
| Task container | No file over 8 GiB; host-graded deliverables up to 1 GiB per file and 4 GiB total; local runs cap processes at 4,096 (8 per CPU on a machine with more than 512 CPUs) |
| Requests | Command, file, HTTP, and frame limits are in [`docs/protocol.md`](../docs/protocol.md#2-requests-harness-to-evaluator) |

## Checking an entry locally

```bash
uv run python -m kit check ../my-harness [--cancel-test] [--no-smoke]
```

`kit check` runs these stages in order and stops at the first failure:

| Stage | What it checks |
|---|---|
| `submission.yaml` | Every field and rule of `submission.yaml` |
| `model` (settings) | Before anything is built, the local model settings: `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` for a `google-cloud` model, or your own model's key and any `--model-base-url` for a local run on a model of your own (from the environment or `--env-file`) |
| `files` | Files named in `serve` exist; scripts that `install` and `serve` run have LF line endings (CRLF warns); no file holds what looks like an API key or private key (the competition takes no keys); no harness file uses Git LFS; symlinks are checked out as links (`core.symlinks` is not `false`); submodules can be fetched anonymously; the entry is the root of its own repository with everything committed |
| `build` | Runs `install` in the builder container and validates the bundle (cached unless files change or `--rebuild` is passed) |
| `model` | Checks that your model answers a short prompt from your serve image through the egress proxy |
| `listen` | Verifies your server accepts a WebSocket connection within 120 s |
| `smoke` | Runs a small task where your harness writes `/app/hello.txt` and sends `done` |
| `mcp` | Runs a task whose answer comes from a tool on the task's MCP server (warns if your harness does not use MCP) |
| `cancel` | With `--cancel-test`: cancels a task after 45 s and checks that your harness still sends `done` |
| `artifacts` | Validates `artifacts/` against the rules in [`docs/artifacts.md`](../docs/artifacts.md#check-before-you-submit) and checks that it is committed |

It prints `ready to submit` only when every stage passes, `artifacts/` is valid,
`model_api` is a `google-cloud/<id>` model, and your entry is at the root of a
Git repository with everything committed. Otherwise, the last line tells you
what to do next. It exits `0` when every stage it ran passed (even if the entry
is not yet ready to submit), `1` when a stage failed or a stage you asked for
did not run (such as `--cancel-test`), and `130` when interrupted. It needs a
local Docker daemon: a remote `DOCKER_HOST` is not supported.

Logs are written to `~/.cache/plebench/<name>/` (or `--work <dir>`):
`build.log`, `serve.log`, `egress.log`, `relay.log`, stage-specific logs
(`egress-model.log`, `serve-mcp.log`, etc.), and `check.json`. The values of
key settings (`MODEL_PROXY_KEY`, or your own model's key in a local run) are
redacted as `[NAME]` in console output and `check.json`, while the raw log
files under `~/.cache/plebench/` keep your server's unredacted output.

- `--no-smoke` skips the smoke and MCP tasks for faster iteration (`listen` and `--cancel-test` still run). The check then never prints `ready to submit`, which requires passing the smoke task.
- `--keep` leaves the containers running and prints the `docker rm -f` command to remove them.
- `--env-file` loads `NAME=value` pairs for the local run, kept outside your entry: `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` for `google-cloud` models, or your own model's key for a local-only entry ([`docs/models.md`](../docs/models.md#your-own-model-local-runs-only)). Variables exported in your shell work too.
- `--model-base-url http://<host>:<port>/v1` points a local run of an
  `openai-compatible` model at a server you run, in place of `model_base_url`
  ([`docs/local-models.md`](../docs/local-models.md)).

Other commands:

```bash
uv run python -m kit validate ../my-harness   # check submission.yaml only (no Docker)
uv run python -m kit build ../my-harness      # build the bundle and serve image only
uv run python -m kit up ../my-harness         # start your server in its container and print its ws:// address
uv run python -m kit artifacts ../my-harness  # review artifacts/ (--from runs/<run> imports a run first, --json)
uv run python -m kit models                   # list the google-cloud models
```

## What kit check does not cover

- Model key scoping: in evaluation, a `google-cloud` entry's key works only for its declared model ([`docs/models.md`](../docs/models.md#what-your-server-receives)), while your local proxy serves all eight.
- Submodule `.git` directories: the evaluation build checks out each submodule with its `.git` directory (`depth 1`), whereas `kit` stages submodule files without `.git`.
- Uncached builds: `kit` reuses cached builds when harness files haven't changed, while the evaluation build always starts from scratch. Pin your dependencies, or test once with `--rebuild`.
- Dev-set grading and concurrency: `kit check` only runs the smoke and MCP tasks. Use `eval/run.py` to run the 29 dev tasks ([`eval/README.md`](../eval/README.md)). In evaluation, your harness may run on many tasks at once, each in its own container.
