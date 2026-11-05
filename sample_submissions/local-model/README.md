# local-model: the custom harness on a model you serve (local runs only)

Runs the reference harness from [`custom-openai`](../custom-openai/) (`serve.py`
and `taskbridge.py`) against `gpt-oss-120b` served from your own GPUs, so you
can try your own model on the dev set. It is for local runs only: the
competition accepts only `google-cloud/<id>` models, so `kit check` never
reports this entry ready to submit. The helper configs in `host/` run on your
GPU host, not inside the serve container.

| File | Purpose |
|---|---|
| `submission.yaml` | Configures `model_api: openai-compatible/gpt-oss-120b`, placeholder `model_base_url: https://llm.example.org/v1`, and token limits tuned for `gpt-oss`'s 131,072-token context window |
| `local.env.example` | A template for the `local.env` your local runs pass with `--env-file` (`MODEL_API_KEY`); keep the filled-in copy outside your entry |
| `host/Caddyfile` | Caddy auth gate: forwards `/v1/...` requests bearing `MODEL_API_KEY` to the local model server and returns `401` to everything else |
| `host/cloudflared.yml` | Cloudflare Tunnel configuration for a gate on another machine behind NAT |

## Setup steps

Copy this folder into a standalone Git repository outside this clone (see [Quick start](../../README.md#quick-start)):

1. Start your model server and auth gate:

   ```bash
   # Terminal 1: start vLLM on localhost
   vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --host 127.0.0.1 --port 8000 \
     --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
   ```

   ```bash
   # Terminal 2 (inside your copied entry directory): generate the key, save it for --env-file, start Caddy
   export MODEL_API_KEY=$(openssl rand -hex 24)
   echo "MODEL_API_KEY=$MODEL_API_KEY" >> <competition repository>/local.env
   caddy run --config host/Caddyfile
   ```

2. Validate and run the dev set from the root of the competition repository, pointing the run at the gate:

   ```bash
   uv run python -m kit check ../my-harness --env-file local.env --model-base-url http://localhost:8090/v1
   uv run python eval/run.py ../my-harness --env-file local.env --model-base-url http://localhost:8090/v1
   ```

See [`docs/local-models.md`](../../docs/local-models.md) for SGLang,
`llama.cpp`, and Ollama setups, and for a model server on another machine. To
use a different model, set the name after `openai-compatible/` to match
`/v1/models` and adjust `MAX_OUTPUT_TOKENS` and `COMPACT_AT_TOKENS` for its
context window.
