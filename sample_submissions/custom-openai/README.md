# custom-openai: full-featured custom harness on the OpenAI API

The most complete reference harness in this repository. It talks to the model via the OpenAI Chat Completions API using `MODEL_BASE_URL`, `MODEL_API_KEY`, and `MODEL_NAME`, so the exact same code runs against OpenAI, any `google-cloud/<id>` model, or a self-hosted model ([`local-model`](../local-model/) uses `serve.py` unchanged). For a stripped-down ~120-line version of the core loop, see [`minimal`](../minimal/).

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

### Streaming and deadline handling
Every model call is streamed. Tool calls execute and return results along with
the remaining time from `limits.agent_timeout_sec`. If a response is truncated
at the token limit or empty, the harness prompts the model to continue and
stitches truncated replies together.

Three minutes before the task deadline, any in-flight stream is interrupted and
the model is asked for its final summary. If time runs out before the summary
finishes, whatever text has streamed so far is sent as `final_message`.

### Retries and context compaction
Assistant messages round-trip provider-specific fields (such as Gemini thought
signatures and Claude signed thinking blocks). Timeouts, `429` rate limits, and
`5xx` errors are retried with exponential backoff as long as enough time remains
before the wrap-up window, and if an endpoint rejects `max_completion_tokens`,
the harness falls back to `max_tokens` or lowers the requested limit
automatically.

To keep long runs within the context window, large tool outputs are head/tail
truncated immediately. Once the prompt exceeds `COMPACT_AT_TOKENS` (or if the
endpoint returns a context-length error), the harness first trims older tool
outputs that the model has already seen. If the conversation is still too large,
it asks the model to summarize older turns into concise notes and strips
invalidated thinking blocks from subsequent turns so Claude endpoints accept the
compacted history.

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
| `MAX_OUTPUT_TOKENS` | `32000` | Max completion tokens per call (lowered automatically if the endpoint enforces a smaller cap) |
| `COMPACT_AT_TOKENS` | `150000` | Prompt token threshold that triggers context compaction |

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
uv run python eval/run.py --url ws://127.0.0.1:8765 --name custom-openai --tasks jobshop-scheduler
```
