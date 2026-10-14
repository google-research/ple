# hermes-opus: Hermes Agent on Claude Opus 5.5

Wraps [hermes-agent](https://github.com/NousResearch/hermes-agent)
(`v2026.9.24`) as a harness running on the Google-sponsored
`google-cloud/claude-opus-5-5` model.

## How it works

For each task, `serve.py`:

1. Creates an isolated Hermes home directory containing `config.yaml` and the `task_env` plugin.
2. Starts a `taskbridge.Relay` so the `hermes` subprocess can communicate with the task container over local HTTP.
3. Runs `hermes chat --query-file <instruction> --format stream-json` with a `--run-budget` set just short of the task deadline so Hermes wraps up cleanly on time.
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

| File | Purpose |
|---|---|
| `submission.yaml` | Clones the pinned `hermes-agent` release, installs it into `/opt/harness/venv` with `uv`, and runs `serve.py` |
| `serve.py` | Generates per-task Hermes config, launches `hermes`, and returns its final output |
| `task_env/` | Hermes terminal backend plugin that routes shell and file operations to the task container |
| `taskbridge.py` | Serve protocol and local HTTP relay (shared across Python samples) |

Note: `serve.py` imports `yaml` (PyYAML) and `websockets` (via `taskbridge.py`), both of which are already installed by `hermes-agent`'s lockfile.

## Customizing

To switch models, change `model_api` in `submission.yaml` to any model in [`docs/models.md`](../../docs/models.md). `serve.py` routes Claude models (`google-cloud/claude-*`, or `anthropic/<model>` with `ANTHROPIC_API_KEY` in `.env`) through Hermes's native Anthropic provider, and all other models through its OpenAI-compatible provider using `MODEL_BASE_URL`.

In `serve.py`, you can also tune `MAX_TURNS` (maximum tool-calling turns),
`BUDGET_MARGIN_SEC` (how early before the task deadline Hermes is told to wrap
up), and `DISABLED_TOOLSETS`.
