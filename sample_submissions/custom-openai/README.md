# custom-openai: full-featured custom harness on the OpenAI API

The most complete reference harness in this repository. It talks to the model via the OpenAI Chat Completions API (the Responses API for OpenAI's GPT-6 models) using `MODEL_BASE_URL`, `MODEL_API_KEY`, and `MODEL_NAME`, so the exact same code runs against OpenAI, any `google-cloud/<id>` model, or a self-hosted model ([`local-model`](../local-model/) uses `serve.py` unchanged). For a stripped-down version of the core loop in under 250 lines, see [`minimal`](../minimal/).

> **Cost:** With its default model (`openai/gpt-6-astra` at `max` effort, up to 128,000 output tokens per call), this sample can cost tens of dollars per task, billed to your OpenAI key. Unless you set `JUDGE_*` or `MODEL_PROXY_*`, `eval/run.py` also uses that key and model as the judge ([`eval/README.md`](../../eval/README.md#before-you-run)), so the judge's calls bill the same key. Try a couple of tasks first with `--tasks`.

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
OpenAI's GPT-6 models (such as `gpt-6-astra`) reject requests that combine
function tools with `reasoning_effort` on `/v1/chat/completions`, so `serve.py`
calls them through the Responses API at `max` effort (and automatically falls
back to the Responses API if any other model returns that `400`). The harness
keeps its internal message history in Chat Completions format and converts it on
each call, passing `store: false` and round-tripping output items (including
encrypted reasoning items) across turns. All other models, including
`local-model`, use Chat Completions.

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
model_api: openai/gpt-6-astra
```

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and commit `.env` to a
private repository. To switch models, update `model_api` in `submission.yaml`
and adjust `.env` to match (for example, `google-cloud/qwen3-coder-480b` needs
no API key, so remove `OPENAI_API_KEY` from `.env`).

Optional settings under `env:` in `submission.yaml`:

| Variable | Default | Description |
|---|---|---|
| `MAX_OUTPUT_TOKENS` | `32000` | Max completion tokens per call (lowered automatically if the endpoint enforces a smaller cap). `submission.yaml` sets `128000` for GPT 6 Astra, whose reasoning counts toward it (its maximum) |
| `COMPACT_AT_TOKENS` | `150000` | Prompt token threshold that triggers context compaction |
| `REASONING_EFFORT` | the model's highest | Reasoning level sent with each call: `max` for Claude, GLM, GPT-6 and DeepSeek-V4, `high` for the others |

## Debugging on your host

To run `serve.py` directly on your host outside Docker:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -a && . ./.env && set +a
MODEL_BASE_URL=https://api.openai.com/v1 MODEL_API_KEY=$OPENAI_API_KEY MODEL_NAME=gpt-6-astra PORT=8765 \
    .venv/bin/python serve.py
```

Then connect `eval/run.py` to it from the competition repository:

```bash
uv run python eval/run.py --url ws://127.0.0.1:8765 --name custom-openai --tasks data-center-site-model
```
