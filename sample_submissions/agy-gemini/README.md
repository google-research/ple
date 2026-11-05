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
streamed text block if the task times out first). If `agy` stalls with no output
for 40 minutes, loses its connection before producing an answer, or hits a rate
limit (waiting `60s * attempt`), `serve.py` resumes the conversation with
`--conversation` for up to 4 total attempts.

| File | Purpose |
|---|---|
| `submission.yaml` | Model declaration, build script, and server startup command |
| `install.sh` | Downloads the pinned `agy` 1.4.10 binary (with SHA-512 verification) and creates the Python venv |
| `serve.py` | Sets up an isolated `agy` home and MCP config per task, runs `agy`, and returns its output |
| `task_mcp.py` | Stdio MCP server exposing `shell`, `read_file`, `write_file`, and `edit_file` in the task container |
| `taskbridge.py` | Serve protocol and local HTTP relay (shared across Python samples) |

## Configuration

Set `AGY_EFFORT` under `env:` in `submission.yaml` to control reasoning effort:
`low`, `medium`, or `high` (the default, Gemini's highest), for the models in
`agy`'s own list; a custom model (`-byom`) gets `agy`'s dynamic thinking budget
instead. Because `agy` uses
the native Gemini API, this sample only works with Gemini models.

You can switch `model_api` to `google-cloud/gemini-4-argon` or
`google-cloud/gemini-3.1-pro-preview`, or, for a local run only, to
`gemini/<model>` with your own `GEMINI_API_KEY` in your environment or
`--env-file`:

- **Built-in `agy` models** (Gemini 4 Argon, Gemini 3.8, 3.7, and 3.6 Flash, and
  Gemini 3.1 Pro): `serve.py` strips `-preview` and appends the effort level
  (for example, `gemini-3.1-pro-high`; note that Gemini 3.1 Pro only supports
  `low` and `high`), and `agy` uses its built-in context compaction threshold.
  You can also set `AGY_MODEL` under `env:` to pass a specific `agy models`
  identifier directly.
- **Other Gemini models**: `serve.py` registers `<model>-byom` in
  `settings.json` (`customModelsConfig`) with an explicit context budget so
  `agy` compacts the conversation before hitting the context limit (`agy`
  triggers compaction roughly 70,000 tokens below the configured budget, which
  `serve.py` sets to `max(W - 70000, W / 2 + 70000)` for `AGY_CONTEXT_WINDOW`
  `W`, default `1048576`).

## Notes

- On startup, `agy` requests a short conversation title from a lightweight model; if the endpoint does not serve that auxiliary model, the title request fails harmlessly and the task continues.
- `serve.py` turns off `agy`'s telemetry in `settings.json` and its updates
  with `AGY_CLI_DISABLE_AUTO_UPDATE=true`; the egress proxy blocks any other
  startup request (feature flags) without affecting execution.
- Each concurrent session uses ~300–400 MB of RAM. On tasks with large MCP catalogs, eager tool schemas add to each prompt (e.g., 264 tools add ~75k input tokens).
