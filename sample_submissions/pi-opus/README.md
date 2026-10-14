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

In `harness.py`, you can edit `NOTE` to append instructions to Pi's system
prompt, or adjust `THINKING` (`off` through `max`, default `max`, the highest) to change
the reasoning budget. To customize tools, modify `task-tools.ts` or pass
additional extensions via `-e` in `harness.py`. To upgrade Pi itself, update
the version in `package.json`, run `npm install --package-lock-only` to refresh
`package-lock.json`, and commit both files.

To change the model, set `model_api` in `submission.yaml` to any model in [`docs/models.md`](../../docs/models.md). Claude models (`google-cloud/claude-*`, `anthropic/*`) use Pi's native `anthropic` provider, and Gemini models (`google-cloud/gemini-*`, `gemini/*`) use Pi's native `google` provider, automatically inheriting context limits, thinking support, and vision support from Pi's built-in catalog. All other models use the OpenAI-compatible `MODEL_BASE_URL` with the fallback limits in `UNKNOWN_MODEL` inside `harness.py` (128k context, 16k max output tokens, text-only; update `UNKNOWN_MODEL` to match your model's specs). Pi 1.1.0's catalog has no Gemini 4 Argon, so `google-cloud/gemini-4-argon` also gets `UNKNOWN_MODEL` (set its limits to a 1,048,576-token context and 65,536 output tokens) and thinks at the server's default level, `high`. Leave `reasoning` out of `UNKNOWN_MODEL` for it: with `reasoning: true`, Pi sends a thinking budget, which Argon refuses with `400`.

## Notes

- Only `read`, `bash`, `edit`, and `write` are enabled. Pi's built-in `grep`, `find`, and `ls` tools are disabled because they would run against the
  serve container's filesystem rather than the task container (`bash` handles
  those inside the task container).
- Individual `bash` calls can run for up to 600 seconds (the protocol maximum).
- `harness.py` stops Pi 90 seconds before the task deadline so its last answer is sent cleanly as `final_message`.
- `npm ci` installs exact versions from `package-lock.json`, ensuring reproducible builds on the evaluation server.
