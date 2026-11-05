# custom-openai: full-featured custom harness on the OpenAI API

The most complete reference harness in this repository. It talks to the model via the OpenAI Chat Completions API (the Responses API for OpenAI's GPT-6 models) using `MODEL_BASE_URL`, `MODEL_API_KEY`, and `MODEL_NAME`, so the exact same code runs against any `google-cloud/<id>` model (by default `google-cloud/gpt-oss-120b`) and, for local runs, against OpenAI or a model you serve ([`local-model`](../local-model/) uses `serve.py` unchanged). For a stripped-down version of the core loop in under 250 lines, see [`minimal`](../minimal/).

| File | Purpose |
|---|---|
| `serve.py` | Agent loop, container and MCP tools, context compaction, and deadline management |
| `taskbridge.py` | Serve protocol implementation (shared across Python samples) |
| `requirements.txt` | Pinned `openai` and `websockets` dependencies |
| `submission.yaml` | Model declaration, build command, and server startup command |

## How it works

### Tools and MCP discovery
The model gets four core tools that execute inside the task container: `bash`,
`read_file`, `write_file` (with `append` support for large files), and
`edit_file` (exact string replacement).

When a task provides MCP servers (documents, spreadsheets, email, CAD, etc.),
small catalogs (up to 128 total tools and under 40,000 characters of schema
JSON) are exposed directly as function tools. When the catalog is larger, the
prompt lists the available MCP servers and provides `mcp_find` and `mcp_call`
tools so the model can search and invoke them on demand without bloating every
prompt.

### GPT-6 on the Responses API
In a local run on `openai/<model>`, OpenAI's GPT-6 models (such as
`gpt-6-astra`) reject requests that combine function tools with
`reasoning_effort` on `/v1/chat/completions`, so `serve.py` calls them through
the Responses API at `max` effort (and automatically falls back to the Responses
API if any other model returns that `400`). The harness keeps its internal
message history in Chat Completions format and converts it on each call, passing
`store: false` and round-tripping output items (including encrypted reasoning
items) across turns. All other models, including `local-model`, use Chat
Completions.

### Streaming and deadline handling
Every model call is streamed. Tool calls execute and return results along with
the remaining time from `limits.agent_timeout_sec`. If a response is truncated
at the token limit or empty, the harness prompts the model to continue and
stitches truncated replies together.

`serve.py` caps every `bash` command's timeout so it ends before the wrap-up
window. Three minutes before the task deadline, it interrupts any in-flight
stream and asks the model for its final summary. If time runs out before the summary finishes,
whatever text has streamed so far is sent as `final_message`.

### Retries and context compaction
Assistant messages preserve provider-specific fields (such as Gemini thought
signatures and Claude signed thinking blocks), and Claude calls mark the system
prompt and newest message with `cache_control` for prompt caching. Timeouts,
`429` rate limits, `5xx` errors, dropped connections, mid-stream errors, and
streams that end without a finish reason are retried up to five times with
backoff (`RETRY_WAITS`, 5 seconds to 2 minutes, stopping before the wrap-up
window) without adding failed partial responses to the conversation. If an
endpoint rejects `max_completion_tokens`, the harness falls back to `max_tokens`
or lowers the requested limit automatically.

To keep long runs within the context window, large tool outputs are head/tail
truncated immediately. Once the prompt exceeds `COMPACT_AT_TOKENS` (or if the
endpoint returns a context-length error), the harness first trims older tool
outputs that the model has already seen. If the conversation is still too large,
it asks the model to summarize older turns into concise notes while preserving
Claude's signed thinking blocks on the recent turns kept in history.

## Configuring the model

```yaml
model_api: google-cloud/gpt-oss-120b
```

The evaluator supplies the key for `google-cloud/<id>` models, so the entry
holds none. To switch models, set `model_api` to another `google-cloud/<id>`
model. For a local run on your own OpenAI key, set `model_api: openai/<model>`
and put `OPENAI_API_KEY` in your environment or `--env-file`; such an entry is
not accepted for submission ([`docs/models.md`](../../docs/models.md#your-own-model-local-runs-only)).

Optional settings under `env:` in `submission.yaml`:

| Variable | Default | Description |
|---|---|---|
| `MAX_OUTPUT_TOKENS` | `32000` | Max completion tokens per call (lowered automatically if the endpoint enforces a smaller cap). For GPT 6 Astra in a local run, whose reasoning counts toward it, set its maximum, `128000` |
| `COMPACT_AT_TOKENS` | `150000` | Prompt token threshold that triggers context compaction (this sample sets `90000` for gpt-oss's 131,072-token window; raise it when you switch to a model with a larger one) |
| `REASONING_EFFORT` | the model's highest | Reasoning level sent with each call: `max` for Claude, GLM, GPT-6 and DeepSeek-V4, `high` for the others |

## Debugging on your host

To run `serve.py` directly on your host outside Docker:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
MODEL_BASE_URL=http://localhost:4000/v1 MODEL_API_KEY=$MODEL_PROXY_KEY MODEL_NAME=gpt-oss-120b PORT=8765 \
    .venv/bin/python serve.py
```

Then connect `eval/run.py` to it from the competition repository:

```bash
uv run python eval/run.py --url ws://127.0.0.1:8765 --name custom-openai --tasks data-center-site-model
```
