# Sample submissions

Every folder here is a complete, working entry. If you're writing a new harness from scratch, start with [`minimal/`](minimal/).

| Folder | Harness | Default `model_api` | What it demonstrates |
|---|---|---|---|
| [`minimal/`](minimal/) | Custom (Python) | `google-cloud/gemini-3.8-flash` | Smallest complete harness (~120 lines): a single tool-calling loop |
| [`plan-verify/`](plan-verify/) | Custom (Python) | `google-cloud/claude-opus-5-5` | `minimal`'s loop split into planner, worker, and verifier roles with per-step time budgets |
| [`node-protocol/`](node-protocol/) | Custom (JavaScript) | `google-cloud/claude-opus-5-5` | Speaks the WebSocket serve protocol directly without `taskbridge.py` |
| [`custom-openai/`](custom-openai/) | Custom (Python) | `openai/gpt-6-astra` | Full-featured reference harness: MCP tools, streaming, context compaction, and deadline-aware retries |
| [`local-model/`](local-model/) | `custom-openai` | `openai-compatible/gpt-oss-120b` | Serves a model from your own GPUs over public HTTPS (see [`docs/local-models.md`](../docs/local-models.md)) |
| [`hermes-opus/`](hermes-opus/) | [Hermes Agent](https://github.com/NousResearch/hermes-agent) | `google-cloud/claude-opus-5-5` | Wraps an existing agent whose terminal runs inside the task container via a plugin backend |
| [`pi-opus/`](pi-opus/) | [Pi](https://github.com/badlogic/pi-mono) | `google-cloud/claude-opus-5-5` | Wraps an existing coding agent with a TypeScript extension that redirects file and shell tools to the task container |
| [`agy-gemini/`](agy-gemini/) | [Antigravity CLI](https://antigravity.google/product/antigravity-cli) | `google-cloud/gemini-3.8-flash` | Wraps a closed-source CLI agent that interacts with the task container through an MCP server |

## Using a sample

Copy the sample folder outside this repository into a standalone Git repository and follow the [Quick start](../README.md#quick-start) steps (only copy `.env.example` to `.env` if the sample includes one).

- If the sample uses a `google-cloud/<id>` model, set `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` for local testing ([setup guide](../docs/models.md#running-google-cloud-models-on-your-machine)).
- Samples do not include pre-generated `artifacts/`. Run `eval/run.py` to generate them, then follow [Before you submit](../README.md#before-you-submit).

## Changing the model

Update `model_api` in `submission.yaml` (adding `model_base_url` only for `openai-compatible` models) and keep `.env` in sync by adding any required API key and removing unused ones ([`docs/models.md`](../docs/models.md)).

- `minimal`, `plan-verify`, `node-protocol`, `custom-openai`, and
  `local-model` use the OpenAI-compatible endpoint provided for every model, so
  they work with any model in `docs/models.md`. For models with smaller context
  windows, lower `MAX_OUTPUT_TOKENS` (default `32000`) under `env:` in
  `submission.yaml`.
- `hermes-opus` and `pi-opus` also support any model in `docs/models.md` (using the native Anthropic API for Claude models, and `pi-opus` using the native Gemini API for Gemini models).
- `agy-gemini` only supports Gemini 3 models.

Every sample asks its model for its highest reasoning level: `max` for Claude
and `high` for Gemini and the others (`AGY_EFFORT` in `agy-gemini`, `THINKING`
in `pi-opus`, `REASONING_EFFORT` under `env:` in the rest).

Always generate your `artifacts/` using the exact `model_api` you submit.

## `taskbridge.py`

All Python samples share [`taskbridge.py`](taskbridge.py), which implements [the serve protocol](../docs/protocol.md). Each sample directory contains its own copy so you can `cp -r` any folder as a self-contained repository.

| API | Description |
|---|---|
| `run(solve)` | Runs a WebSocket server on `$PORT`, calling `solve(session)` per task connection and sending its return string as `final_message` |
| `Session.instruction`, `.workdir`, `.limits`, `.mcp_servers` | Fields from the initial `task` message |
| `Session.exec(command, timeout_sec=120)` | Runs `bash -c` in the task container; returns `exit_code`, `output`, `truncated`, and `timed_out` |
| `Session.read_file(path)`, `Session.write_file(path, data, append=False)` | Reads a file (up to 10 MB) as `bytes`, or writes `str`/`bytes` (automatically chunking large writes) |
| `Session.http(url, method, headers, body)` | Sends an HTTP request from inside the task container to one of its MCP servers |
| `Session.log(text)`, `Session.usage(model, input_tokens, output_tokens, cache_read_tokens)` | Sends one-way log and token-usage messages |
| `Session.time_left()` | Remaining task time in seconds, or `None` |
| `Session.final_message` | Fallback final message sent if the task is cancelled or `solve` raises an exception |
| `Cancelled` | Exception raised when the evaluator cancels the task (`taskbridge` then sends `done` automatically) |
| `TaskError` | Exception raised when a request returns `ok: false` or would be rejected as a malformed frame |
| `Relay(session)` | Local HTTP bridge to the task container for subprocess agents (`async with Relay(session) as relay:`, passing `relay.env()` and `relay.mcp_url(name)`) |
| `MCPClient(session, url)` | Client for task MCP servers (`initialize()`, `list_tools()`, `call_tool(name, arguments)`) |
| `run_process(argv, session, on_line=...)` | Runs an agent subprocess, streams stdout lines to `on_line`, and stops it cleanly before the task deadline |
