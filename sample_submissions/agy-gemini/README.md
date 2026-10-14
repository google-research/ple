# agy-gemini: Antigravity CLI on Gemini 3.8 Flash

Runs [Antigravity CLI](https://antigravity.google/docs/cli/overview) (`agy`)
headless as a harness using the Google-sponsored `google-cloud/gemini-3.8-flash`
model.

## How it works

For each task, `serve.py` spawns an `agy -p` subprocess inside the serve
container. Because `agy`'s built-in file and shell tools would operate on the
serve container rather than the task container, `serve.py` disables built-in
tools (`tools: []`) and exposes the task container via MCP instead.

A lightweight stdio MCP server in `task_mcp.py` provides `shell`, `read_file`,
`write_file`, and `edit_file` tools backed by the task container via
`taskbridge.Relay` (`read_file` returns image files as native MCP image blocks
so Gemini can inspect engineering drawings directly). Any task-provided MCP
servers (email, documents, CAD, etc.) are also proxied into the task container
through `Relay.mcp_url`. Every MCP server is registered with `"eager": true` in
`mcp_config.json`, and `serve.py` pre-fetches each server's tool list before
launching `agy` so the model sees all tool schemas immediately.

`agy` reads `GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL`, which the evaluator
sets automatically for Gemini models (`modelProvider: "gemini"`). The final text
response from `--output-format stream-json` becomes `final_message` (or the last
streamed text block if the task times out first). If `agy` produces no output
for 15 minutes due to a stalled model request, `serve.py` restarts it and
resumes the session via `--conversation`.

| File | Purpose |
|---|---|
| `submission.yaml` | Model declaration, build script, and server startup command |
| `install.sh` | Downloads the pinned `agy` binary (with SHA-256 verification) and creates the Python venv |
| `serve.py` | Sets up an isolated `agy` home and MCP config per task, runs `agy`, and returns its output |
| `task_mcp.py` | Stdio MCP server exposing `shell`, `read_file`, `write_file`, and `edit_file` in the task container |
| `taskbridge.py` | Serve protocol and local HTTP relay (shared across Python samples) |

## Configuration

Set `AGY_EFFORT` under `env:` in `submission.yaml` to control reasoning effort:
`low`, `medium`, or `high` (the default, Gemini's highest).

You can switch `model_api` to `google-cloud/gemini-3.1-pro-preview`, or to
`gemini/<model>` with your own `GEMINI_API_KEY` in `.env`. `serve.py` derives
`agy`'s internal model identifier by stripping `-preview` and appending the
effort level (e.g., `gemini-3.1-pro-high`; note that Gemini 3.1 Pro only
supports `low` and `high`). You can also override the identifier directly by
setting `AGY_MODEL` under `env:` (run `agy models` to see valid names). Gemini 4
Argon is not in that list, so this sample cannot use
`google-cloud/gemini-4-argon`: `agy` sends a model outside its list a thinking
budget, which Argon refuses. Because `agy` uses the native Gemini API, this
sample only works with Gemini models.

## Notes

- On startup, `agy` requests a short conversation title from a lightweight model; if the endpoint does not serve that auxiliary model, the title request fails harmlessly and the task continues.
- `agy` attempts to contact telemetry, feature-flag, and update endpoints on
  startup; the egress proxy blocks those requests without affecting execution,
  and `serve.py` sets `AGY_CLI_DISABLE_AUTO_UPDATE=true`.
- Each concurrent session uses ~300–400 MB of RAM. On tasks with large MCP catalogs, eager tool schemas add to each prompt (e.g., 264 tools add ~75k input tokens).
