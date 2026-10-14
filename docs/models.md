# Models

The `model_api` field in `submission.yaml` declares which model your harness
calls. Whatever the provider, your server receives the same `MODEL_*` and
`OPENAI_*` environment variables, so a harness that reads them can switch models
without code changes.

| `model_api` | Provider | Required in `.env` | Outbound host allowed |
|---|---|---|---|
| `google-cloud/<id>` | Google-sponsored models | None | Sponsored model endpoint |
| `openai/<model>` | OpenAI (your account) | `OPENAI_API_KEY=...` | `api.openai.com` |
| `anthropic/<model>` | Anthropic (your account) | `ANTHROPIC_API_KEY=...` | `api.anthropic.com` |
| `gemini/<model>` | Google AI Studio (your account) | `GEMINI_API_KEY=...` | `generativelanguage.googleapis.com` |
| `openai-compatible/<model>` + `model_base_url` | Your own server (vLLM, SGLang, etc.) or any compatible provider | `MODEL_API_KEY=...` | Host of `model_base_url` |

The leaderboard shows the model beside each entry. Put any required API key in `.env` ([`kit/README.md`](../kit/README.md#env)).

## Google-sponsored models (`google-cloud/<id>`)

These thirteen models are hosted for the competition, need no API key, and are
free on our servers:

| `model_api` | Model | Supported APIs |
|---|---|---|
| `google-cloud/claude-opus-5-5` | Claude Opus 5.5 | OpenAI Chat Completions, Anthropic Messages |
| `google-cloud/gemini-4-argon` | Gemini 4 Argon | OpenAI Chat Completions, Gemini API |
| `google-cloud/gemini-3.8-flash` | Gemini 3.8 Flash | OpenAI Chat Completions, Gemini API |
| `google-cloud/gemini-3.1-pro-preview` | Gemini 3.1 Pro (preview) | OpenAI Chat Completions, Gemini API |
| `google-cloud/gpt-oss-120b` | gpt-oss 120B | OpenAI Chat Completions |
| `google-cloud/qwen3-coder-480b` | Qwen3 Coder 480B A35B | OpenAI Chat Completions |
| `google-cloud/qwen3-next-80b` | Qwen3-Next 80B A3B Instruct | OpenAI Chat Completions |
| `google-cloud/deepseek-v3.2` | DeepSeek-V3.2 | OpenAI Chat Completions |
| `google-cloud/kimi-k2-thinking` | Kimi K2 Thinking | OpenAI Chat Completions |
| `google-cloud/minimax-m2` | MiniMax M2 | OpenAI Chat Completions |
| `google-cloud/glm-5.2` | GLM 5.2 | OpenAI Chat Completions |
| `google-cloud/gemma-4-26b` | Gemma 4 26B A4B | OpenAI Chat Completions |
| `google-cloud/llama-4-maverick` | Llama 4 Maverick 17B 128E | OpenAI Chat Completions |

All thirteen support tool calling, with or without streaming. Run
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

| Variable | `google-cloud/<id>` | `openai/` | `anthropic/` | `gemini/` | `openai-compatible/` |
|---|---|---|---|---|---|
| `MODEL_API` | As written | As written | As written | As written | As written |
| `MODEL_NAME` | `<id>` | `<model>` | `<model>` | `<model>` | `<model>` |
| `MODEL_BASE_URL` | Gateway URL (`.../v1`) | `https://api.openai.com/v1` | `https://api.anthropic.com/v1` | `https://generativelanguage.googleapis.com/v1beta/openai` | `model_base_url` |
| `MODEL_API_KEY` | Per-submission key | Your key | Your key | Your key | Your key |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | Gateway URL without `/v1` (Claude models) | Not set | `https://api.anthropic.com`, your key | Not set | Not set |
| `GOOGLE_GEMINI_BASE_URL`, `GEMINI_API_KEY` | Gateway URL without `/v1` (Gemini models) | Not set | Not set | `https://generativelanguage.googleapis.com`, your key | Not set |

On the evaluation server, a `google-cloud` entry's key is generated for that
submission and works only for the declared model; requests for any other model
get `403`. Locally, the kit passes your proxy's URL and `MODEL_PROXY_KEY`
instead.

`OPENAI_BASE_URL` and `OPENAI_API_KEY` always match `MODEL_BASE_URL` and
`MODEL_API_KEY`, so the OpenAI SDK needs no arguments:

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY


async def ask(messages, tools):
    return await client.chat.completions.create(model=os.environ["MODEL_NAME"], messages=messages, tools=tools)
```

For Claude and Gemini models, the Anthropic and Gemini SDKs read their
respective variables from the table above.

## Serving your own model (`openai-compatible/<model>`)

Any server that implements the OpenAI Chat Completions API with tool calls (in
both streamed and non-streamed responses) works. Because the evaluation server
reaches it over the internet:

- `model_base_url` must be an `https://` URL (typically ending in `/v1`) on a DNS name with a public IPv4 address and no private IP resolutions.
- The server must require `Authorization: Bearer <MODEL_API_KEY>`, accept the model name after `openai-compatible/`, and handle about 12 concurrent tasks until your run finishes.
- Test-set prompts and files reach your server, so it must not log or store them.

[`local-models.md`](local-models.md) shows how to serve a model from your own GPUs, and [`sample_submissions/local-model/`](../sample_submissions/local-model/) is a complete entry configured this way.

## Tips for calling models

- Stream long generations, or give your client generous timeouts. Reasoning
  models can think silently for several minutes before the first token, and the
  egress proxy closes any connection that transfers no bytes for 30 minutes.
- Retry HTTP `429`, timeouts, and `5xx` responses with exponential backoff. A `400` means the request itself was invalid, so retrying it unchanged won't help.
- A Google-sponsored model called without a reasoning level thinks at its
  highest: `max` for Claude and GLM, `high` for Gemini, gpt-oss, Kimi, and
  MiniMax. The evaluation server and `kit/model_proxy` both add it. Set
  `reasoning_effort` (or `thinking`) yourself to use a lower level.
- Gemini 4 Argon takes a thinking level, `low`, `medium`, or `high`
  (`reasoning_effort`, or `thinkingConfig.thinkingLevel` in the Gemini API), and
  returns `400` for a thinking budget (`thinkingBudget`, even `0`). The Gemini
  API also accepts `MAX`, but Argon thinks less at `MAX` than at `high`, so
  `high` is its highest level. In OpenAI Chat Completions a `thinking` budget is
  not refused but not applied either: Argon then thinks at its own default,
  `medium`. It writes at most 65,536 output tokens, and its 1,048,576-token
  context holds the input, the thinking, and the output together.
- Claude Opus 5.5 returns `400` if you set `temperature`, `top_p`, `top_k`, a
  thinking budget (`budget_tokens`), `thinking: {"type": "disabled"}`, or a
  forced tool choice (`tool_choice: "required"` or a named function; `any` or
  `tool` in Anthropic Messages). Use `auto` or `none`, and set `max_tokens`
  generously (`32000` is a good default).
- Track the remaining time in `limits.agent_timeout_sec` and finish your model calls before it runs out.
