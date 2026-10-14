# local-model: running the custom harness against your own GPU server

Runs the reference harness from [`custom-openai`](../custom-openai/) (`serve.py`
and `taskbridge.py`) against `gpt-oss-120b` hosted on your own GPUs. During
official evaluation, our servers connect to your model over the internet at
`model_base_url`. The helper configs in `host/` run on your GPU host, not
inside the evaluation container.

| File | Purpose |
|---|---|
| `submission.yaml` | Configures `model_api: openai-compatible/gpt-oss-120b`, placeholder `model_base_url: https://llm.example.org/v1`, and token limits tuned for `gpt-oss`'s 131,072-token context window |
| `host/Caddyfile` | Caddy auth gate: forwards `/v1/...` requests bearing `MODEL_API_KEY` to the local model server and returns `401` to everything else |
| `host/cloudflared.yml` | Cloudflare Tunnel configuration for exposing the gate from behind NAT |

## Setup steps

Copy this folder into a standalone Git repository outside this clone (see [Quick start](../../README.md#quick-start)):

1. Start your model server and auth gate:

   ```bash
   # Terminal 1: start vLLM on localhost
   vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --host 127.0.0.1 --port 8000 \
     --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
   ```

   ```bash
   # Terminal 2 (inside your copied entry directory): generate key and start Caddy
   export MODEL_API_KEY=$(openssl rand -hex 24)
   echo "MODEL_API_KEY=$MODEL_API_KEY" > .env
   caddy run --config host/Caddyfile
   ```

2. Publish the gate over HTTPS and update `model_base_url` in `submission.yaml`
   to `https://<your-hostname>/v1`. If your host has a public IP, run
   `GATE_ADDRESS=llm.example.org caddy run --config host/Caddyfile` so Caddy
   serves HTTPS directly. If it sits behind NAT, fill in `host/cloudflared.yml`
   and run `cloudflared tunnel --config host/cloudflared.yml run`.

3. Validate and run the dev set from the root of the competition repository (without `--model-base-url`, so traffic goes through your public HTTPS URL):

   ```bash
   uv run python -m kit check ../my-harness
   uv run python eval/run.py ../my-harness
   ```

See [`docs/local-models.md`](../../docs/local-models.md) for SGLang, `llama.cpp`, and Ollama setups, alternative tunneling options, and external reachability
checks. To use a different model, set the name after `openai-compatible/` to
match `/v1/models` and adjust `MAX_OUTPUT_TOKENS` and `COMPACT_AT_TOKENS` for
its context window.
