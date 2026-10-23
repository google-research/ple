# hermes-opus: Hermes Agent on Claude Opus 5.5

Wraps [hermes-agent](https://github.com/NousResearch/hermes-agent)
(`v0.22.2`) as a harness running on the Google-sponsored
`google-cloud/claude-opus-5-5` model.

## How it works

For each task, `serve.py`:

1. Creates an isolated Hermes home directory containing `config.yaml` and the `task_env` plugin.
2. Starts a `taskbridge.Relay` so the `hermes` subprocess can communicate with the task container over local HTTP.
3. Runs `hermes chat --query-file <instruction> --format stream-json` with a `--run-budget` sized so that Hermes's wrap-up notice (sent at 80% of the budget, telling the model to stop verifying and deliver) arrives `BUDGET_MARGIN_SEC` before the task deadline.
4. Returns Hermes's final response as `final_message` and logs token usage.

The `task_env/` plugin implements a custom Hermes terminal backend that executes
every shell command inside the task container via the relay. Because Hermes runs
its file tools (`read_file`, `write_file`, `patch`, `search_files`) through
the terminal backend, file operations automatically target the task container
too. When a task provides MCP servers, `serve.py` registers their relay URLs
under `mcp_servers` in `config.yaml` so Hermes exposes their tools to the model.

Built-in Hermes toolsets that require internet access, a browser, a GUI desktop,
or human interaction (`web`, `browser`, `vision`, `image_gen`, `clarify`,
etc.) are disabled in `serve.py`.

Two Hermes behaviors are adjusted for the evaluation gateway, which Hermes
otherwise treats as a third-party endpoint:

- By default, Hermes strips Claude's prior thinking blocks whenever the base URL
  is not `api.anthropic.com`, forcing Claude to re-derive its reasoning on every
  turn. At build time, `install` runs `thinking_patch.py` to teach Hermes to
  preserve signed thinking blocks when `HERMES_NATIVE_THINKING_BASE_URL` matches
  `ANTHROPIC_BASE_URL` (the build fails if the patch does not apply cleanly).
- Hermes's Anthropic adapter blocks auxiliary calls to non-`api.anthropic.com`
  URLs, which would cause context-compression summarization to fail and drop
  middle turns. `serve.py` routes compression summaries through the
  OpenAI-compatible `MODEL_BASE_URL` instead (`auxiliary.compression` in
  `config.yaml`).

`serve.py` also sets a 1-hour request timeout and a 30-minute stream-idle
timeout in `config.yaml` and the `HERMES_*_TIMEOUT` environment variables so
long max-effort reasoning pauses are not cut short. It requests each model's
highest reasoning effort (`max` for Claude and GLM 5.2, and `high` for other
models; the model proxy strips `reasoning_effort` for models that do not
support it).

| File | Purpose |
|---|---|
| `submission.yaml` | Clones the pinned `hermes-agent` release (checking its commit), installs it into `/opt/harness/venv` with `uv`, patches it, and runs `serve.py` |
| `thinking_patch.py` | Install-time patch so Hermes sends Claude's earlier thinking back on the evaluation gateway |
| `serve.py` | Generates per-task Hermes config, launches `hermes`, and returns its final output |
| `task_env/` | Hermes terminal backend plugin that routes shell and file operations to the task container |
| `taskbridge.py` | Serve protocol and local HTTP relay (shared across Python samples) |

Note: `serve.py` needs only `websockets` (via `taskbridge.py`), which `hermes-agent`'s lockfile already installs. It writes `config.yaml` as JSON, which YAML parsers also read, because Hermes does not install PyYAML.

## Customizing

To switch models, change `model_api` in `submission.yaml` to any model in [`docs/models.md`](../../docs/models.md). `serve.py` routes Claude models (`google-cloud/claude-*`, or `anthropic/<model>` with `ANTHROPIC_API_KEY` in `.env`) through Hermes's native Anthropic provider, and all other models through its OpenAI-compatible provider using `MODEL_BASE_URL`.

In `serve.py`, you can also tune `MAX_TURNS` (maximum tool-calling turns,
500 by default as in Hermes, so the task's time limit ends a run),
`BUDGET_MARGIN_SEC` (how early before the task deadline Hermes is told to wrap
up), `AUX_TIMEOUT_SEC` (the timeout of the compressor's summary call),
`REQUEST_TIMEOUT_SEC` and `STREAM_IDLE_SEC` (the model calls' timeouts), and
`DISABLED_TOOLSETS`.
