# pi-opus: Pi Coding Agent on Claude Opus 5.5

Wraps the open-source [Pi](https://github.com/badlogic/pi-mono) coding agent
(`@earendil-works/pi-coding-agent`, pinned to `1.1.0`) on the Google-sponsored
`google-cloud/claude-opus-5-5` model. `harness.py` and the TypeScript extension
`task-tools.ts` bridge Pi to the serve protocol.

## How it works

`serve.py` starts the WebSocket server (`taskbridge.run(harness.solve)`). For
each task connection, `harness.py`:

1. Starts a local `taskbridge.Relay` so Pi can reach the task container over HTTP.
2. Writes per-task Pi configuration files into a scratch directory: `models.json`, `settings.json`, and `mcp.json` (pointing to the relay URLs of the task's MCP servers).
3. Runs `pi --print --mode json` with the task instruction piped to stdin. Pi handles the agent loop, tool execution, retries, and context compaction.
4. Streams Pi's JSON events into `log` and `usage` messages and returns Pi's final answer as `final_message`.

Because Pi runs inside the serve container while the project files live in the
task container, the TypeScript extension `task-tools.ts` overrides Pi's `read`,
`write`, `edit`, and `bash` tools to execute inside the task container via the
relay (resolving relative paths and running commands in the task's `workdir`).

| File | Purpose |
|---|---|
| `submission.yaml` | Installs Pi via `npm ci` and a Python venv for the wrapper, then runs `serve.py` |
| `package.json`, `package-lock.json` | Pinned Pi package and dependency lockfile |
| `serve.py` | WebSocket entrypoint (`taskbridge.run(harness.solve)`) |
| `harness.py` | Per-task Pi configuration, subprocess management, and event streaming |
| `task-tools.ts` | Pi extension redirecting `read`, `write`, `edit`, and `bash` to the task container |
| `taskbridge.py` | Serve protocol and local HTTP relay (shared across Python samples) |

## Customizing

In `harness.py`, you can edit `NOTE` (and `TIME_NOTE`) to append instructions to Pi's system
prompt, or adjust `THINKING` (`off` through `max`, default `max`, the highest) to change
the reasoning budget. To customize tools, modify `task-tools.ts` or pass
additional extensions via `-e` in `harness.py`. To upgrade Pi itself, update
the version in `package.json`, run `npm install --package-lock-only` to refresh
`package-lock.json`, and commit both files.

To change the model, set `model_api` in `submission.yaml` to any model in
[`docs/models.md`](../../docs/models.md). Claude models
(`google-cloud/claude-*`, or `anthropic/*` in a local run on your own key) use
Pi's native `anthropic` provider, and Gemini models (`google-cloud/gemini-*`, or
`gemini/*` in a local run) use Pi's native `google` provider, inheriting context
limits, thinking support, and vision support from Pi's built-in catalog. Any
model not in Pi 1.1.0's catalog, including OpenAI-compatible models and
`google-cloud/gemini-4-argon`, falls back to `UNKNOWN_MODEL` in `harness.py`
(128k context, 16k max output tokens, text-only). Update those limits to match
your model (for Gemini 4 Argon: 1,048,576 context and 65,536 max output
tokens). For Argon, also leave `reasoning` unset: Pi's `reasoning: true` sends a
token budget that Argon rejects with `400`.

## Notes

- Only `read`, `bash`, `edit`, and `write` are enabled. Pi's built-in `grep`, `find`, and `ls` tools are disabled because they would run against the
  serve container's filesystem rather than the task container (`bash` handles
  those inside the task container).
- Individual `bash` calls can run for up to 600 seconds (the protocol maximum).
- `harness.py` appends the task's remaining minutes to Pi's system prompt, stops Pi 90 seconds before the task deadline, and sends Pi's latest response as `final_message`.
- Pi is configured (`SETTINGS` in `harness.py`) to retry failed model requests up to 8 times over ~4 minutes, allow up to an hour for silent reasoning streams, and keep Claude's prompt cache warm during long commands.
- Task MCP servers are passed to Pi via `mcp.json`.
- `npm ci` installs exact versions from `package-lock.json`, so every build gets the same packages.
