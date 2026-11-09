# Models

The `model_api` field in `submission.yaml` declares which model your harness
calls. Entries use one of the Google-sponsored `google-cloud/<id>` models and
contain no API keys: your server reads the model's URL and key from the
`MODEL_*` and `OPENAI_*` environment variables. Because every model uses the
same variable names, your harness can switch models without code changes.

For local runs only, `model_api` can also name a model of your own
([below](#your-own-model-local-runs-only)).

## Google-sponsored models (`google-cloud/<id>`)

Entries can use these eight models:

| `model_api` | Model | Supported APIs |
|---|---|---|
| `google-cloud/gemini-4-argon` | Gemini 4 Argon | OpenAI Chat Completions, Gemini API |
| `google-cloud/gemini-3.8-flash` | Gemini 3.8 Flash | OpenAI Chat Completions, Gemini API |
| `google-cloud/gemini-3.1-pro-preview` | Gemini 3.1 Pro (preview) | OpenAI Chat Completions, Gemini API |
| `google-cloud/claude-opus-5-5` | Claude Opus 5.5 | OpenAI Chat Completions, Anthropic Messages |
| `google-cloud/gpt-oss-120b` | gpt-oss 120B | OpenAI Chat Completions |
| `google-cloud/glm-5.2` | GLM 5.2 | OpenAI Chat Completions |
| `google-cloud/gemma-4-26b` | Gemma 4 26B A4B | OpenAI Chat Completions |
| `google-cloud/llama-4-maverick` | Llama 4 Maverick 17B 128E | OpenAI Chat Completions |

All eight support tool calling, with or without streaming. Run
`uv run python -m kit models` to print this list.

### Running google-cloud models on your machine

For local runs, you need a proxy that serves the same model names. [`kit/model_proxy/start.sh`](../kit/model_proxy/start.sh) starts a LiteLLM proxy backed by Vertex AI in your own Google Cloud project (billed to that project):

```bash
gcloud auth application-default login
export VERTEXAI_PROJECT=<your-gcp-project> MODEL_PROXY_KEY=$(openssl rand -hex 24)
kit/model_proxy/start.sh &                     # serves google-cloud models on http://localhost:4000
export MODEL_PROXY_URL=http://localhost:4000
uv run python -m kit check ../my-harness
```

`kit check` and `eval/run.py` read `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` from
your environment or `--env-file`.

Any other LiteLLM or OpenAI-compatible proxy that serves those names works too
(for example, routing `claude-opus-5-5` to your own Anthropic key), as long as
it also speaks the native API your harness uses: Anthropic Messages for
`hermes-opus` and `pi-opus` on Claude, the Gemini API for `agy-gemini` and for
`pi-opus` on Gemini. Everything else uses OpenAI Chat Completions.

A proxy on your own machine should serve plain `http://`. The serve container
reaches your host as `host.docker.internal`, a hostname TLS certificates rarely
cover.

## What your server receives

| Variable | Value for `google-cloud/<id>` |
|---|---|
| `MODEL_API` | `model_api` as written |
| `MODEL_NAME` | `<id>` |
| `MODEL_BASE_URL`, `MODEL_API_KEY` | Gateway URL (`.../v1`) and the entry's key |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | Gateway URL without `/v1` and the same key (Claude models only) |
| `GOOGLE_GEMINI_BASE_URL`, `GEMINI_API_KEY` | Gateway URL without `/v1` and the same key (Gemini models only) |

Call only the model you declared. In evaluation, the key works for that model
alone, and requests for any other model get `403`. Locally, the kit passes your
proxy's URL and `MODEL_PROXY_KEY`, and the proxy serves all eight models.

`OPENAI_BASE_URL` and `OPENAI_API_KEY` always match `MODEL_BASE_URL` and
`MODEL_API_KEY`, so the OpenAI SDK needs no arguments:

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
MODEL = os.environ["MODEL_NAME"]
CACHE = {"type": "ephemeral"}


def marked(messages):
    """Mark the system prompt and newest message for Claude prompt caching (other models cache automatically)."""
    if "claude" not in MODEL.lower():
        return messages
    out = list(messages)
    for i in {0, len(out) - 1}:
        if isinstance(out[i].get("content"), str) and out[i]["content"]:
            out[i] = {**out[i], "content": [{"type": "text", "text": out[i]["content"], "cache_control": CACHE}]}
    return out


async def ask(messages, tools):
    return await client.chat.completions.create(model=MODEL, messages=marked(messages), tools=tools)
```

Claude only caches prompt prefixes at explicit cache breakpoints, so mark the
system prompt and the newest message (see
[Tips for calling models](#tips-for-calling-models)). Without those marks,
every turn re-processes the full conversation.

For Claude and Gemini models, the Anthropic and Gemini SDKs read their
respective variables from the table above.

## Your own model (local runs only)

`kit check`, `kit up`, and `eval/run.py` can also run your harness on a model
you pay for or serve, for example to compare models on the dev set. The
competition accepts only `google-cloud/<id>` models, so `kit check` never
reports such an entry ready to submit. Its key goes in your environment or in
`--env-file` (kept outside your entry, like `MODEL_PROXY_KEY`), never in your
repository.

| `model_api` | Provider | Key (environment or `--env-file`) | Outbound host allowed |
|---|---|---|---|
| `openai/<model>` | OpenAI | `OPENAI_API_KEY` | `api.openai.com` |
| `anthropic/<model>` | Anthropic | `ANTHROPIC_API_KEY` | `api.anthropic.com` |
| `gemini/<model>` | Google AI Studio | `GEMINI_API_KEY` | `generativelanguage.googleapis.com` |
| `openai-compatible/<model>` + `model_base_url` | A server you run (vLLM, SGLang, etc.) or any compatible provider | `MODEL_API_KEY` | Host of `model_base_url`, or of `--model-base-url` |

Your server receives the provider's OpenAI-compatible URL (or `model_base_url`)
in `MODEL_BASE_URL` and `OPENAI_BASE_URL`, and the key in `MODEL_API_KEY` and
`OPENAI_API_KEY`. For `anthropic/` it also gets `ANTHROPIC_BASE_URL`
(`https://api.anthropic.com`) and `ANTHROPIC_API_KEY`; for `gemini/`,
`GOOGLE_GEMINI_BASE_URL` (`https://generativelanguage.googleapis.com`) and
`GEMINI_API_KEY`.

For `openai-compatible/`, any server that implements the OpenAI Chat Completions
API with tool calls (in both streamed and non-streamed responses) works. The
part after `openai-compatible/` is the model name sent in requests.
`model_base_url` must be an `https://` URL on a public DNS name; to reach a
server on your machine or network, pass `--model-base-url` instead.
[`local-models.md`](local-models.md) shows how to serve a model from your own
GPUs, and [`sample_submissions/local-model/`](../sample_submissions/local-model/)
is a complete entry configured this way.

## Tips for calling models

- Stream long generations, or give your client generous timeouts. Reasoning
  models can think silently for several minutes before the first token, and the
  egress proxy closes any connection that transfers no bytes for 30 minutes.
- If your harness wraps the Claude Code CLI, set `API_FORCE_IDLE_TIMEOUT=0` in
  its environment. Otherwise the CLI drops any stream that stays silent for 5 to
  6 minutes ("API Error: The operation timed out."), regardless of other timeout
  settings.
- Claude requires explicit cache breakpoints: put
  `"cache_control": {"type": "ephemeral"}` on the system prompt and the newest
  message (up to 4 breakpoints are allowed). See `minimal`, `plan-verify`, and
  `node-protocol` for examples. Gemini, OpenAI, and vLLM cache prefixes
  automatically.
- Retry HTTP `429`, timeouts, and `5xx` responses with exponential backoff; do
  not retry `400` responses without changing the request. Also retry dropped
  connections, mid-stream error events, and streams that end without a finish
  reason, and discard any partial response so the model never sees an incomplete
  turn in its conversation history.
- When you call a Google-sponsored model without specifying a reasoning level,
  the model gateway and `kit/model_proxy` default to the model's highest
  setting: `max` for Claude and GLM, and `high` for Gemini and gpt-oss. Pass
  `reasoning_effort` (or `thinking`) explicitly to use a lower
  level. LLM judge calls are the one exception: they run at the model's standard
  default (via `<model>-judge` on `kit/model_proxy`). If Claude Opus 5.5 refuses
  a judge request, the model gateway and `kit/model_proxy` send it once to
  Claude Opus 5; harness requests are never re-sent.
- Gemini 4 Argon takes a thinking level of `low`, `medium`, or `high` (via
  `reasoning_effort`, or `thinkingConfig.thinkingLevel` in the Gemini API) and
  returns `400` if you pass a token budget (`thinkingBudget`, even `0`). Use
  `high` for maximum reasoning (the Gemini API also accepts `MAX`, but Argon
  reasons less at `MAX` than at `high`). In OpenAI Chat Completions, a
  `thinking` budget is ignored and Argon falls back to `medium`. Argon supports
  up to 65,536 output tokens within a 1,048,576-token context window shared
  across input, thinking, and output.
- Claude Opus 5.5 returns `400` if you set `temperature`, `top_p`, `top_k`, a
  thinking budget (`budget_tokens`), `thinking: {"type": "disabled"}`, or a
  forced tool choice (`tool_choice: "required"` or a named function; `any` or
  `tool` in Anthropic Messages). Use `auto` or `none`, and set `max_tokens`
  generously (`32000` is a good default).
- Track the remaining time in `limits.agent_timeout_sec` and finish your model calls before it runs out.
- Provider-hosted tools are not allowed ([Fair play](rules.md#fair-play)). For
  `google-cloud` models, both the model gateway and `kit/model_proxy` reject
  requests that enable them with HTTP `400`: Gemini Google Search, URL context,
  and code execution; Claude web search, web fetch, and code execution; and
  OpenAI `web_search_options` or hosted tools (web search, file search, code
  interpreter, computer use, remote MCP). Always use function tools that run
  inside the task container.
  `kit/model_proxy` only exposes the endpoints harnesses use (Chat Completions,
  Anthropic Messages, the Responses API, and Gemini
  `generateContent`/`streamGenerateContent`) and returns `400` for any other
  route.
