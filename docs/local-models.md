# Running the dev set on a model you serve

For a local run, an entry with `model_api: openai-compatible/<model>` can call a
model server you run, for example to try a model on your own GPUs on the dev
set. This is for local runs only: the competition accepts only
`google-cloud/<id>` models ([`models.md`](models.md#your-own-model-local-runs-only)),
so `kit check` never reports such an entry ready to submit, and the evaluation
server never calls your server.

A clean setup keeps the model server on `127.0.0.1` and puts a small auth gate
in front of it that checks your key. [`sample_submissions/local-model/`](../sample_submissions/local-model/)
is a complete entry set up this way, with the gate config in its `host/` folder.

## Settings

- `model_base_url` in `submission.yaml` must be an `https://` URL on a DNS name
  (the kit checks it like every field: IP addresses, single-label names, and
  names ending in `.local`, `.internal`, or `.localhost` are refused). For a
  server on your machine or network, keep a placeholder (the sample has
  `https://llm.example.org/v1`) and pass
  `--model-base-url http://<host>:<port>/v1` to `kit check` and `eval/run.py`.
- Inside the serve container, `localhost` in `--model-base-url` becomes
  `host.docker.internal`, so on Linux the address must listen on `0.0.0.0` (or
  the Docker bridge IP), not only `127.0.0.1`. The gate below listens on all
  addresses while the model server stays on `127.0.0.1`.
- The part of `model_api` after `openai-compatible/` is sent as the `model`
  field in API requests, so it must match a name your server accepts (listed at
  `/v1/models`). For `openai-compatible/gpt-oss:20b`, the harness sends
  `gpt-oss:20b`.
- Put the key in your environment or in `--env-file` as `MODEL_API_KEY`, never in your entry. The harness sends it as `Authorization: Bearer <key>`, and the gate checks it.
- The egress proxy closes any connection that transfers no bytes in either direction for 30 minutes.
- The endpoint must return tool calls in both streamed and non-streamed
  responses (the `local-model` sample streams every call), and serve one model
  call per task you run at once (`eval/run.py --trials N`).

## 1. Start the model server

Run your server on `127.0.0.1:8000` with tool calling enabled. For
`gpt-oss-120b` with vLLM:

```bash
vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
```

On a single 80 GB GPU, vLLM 0.29.0 starts `gpt-oss-120b` with default settings,
but only has room for about 180,000 tokens of KV cache, so several long sessions
will wait on each other. Using two GPUs (`--tensor-parallel-size 2`) leaves much
more room.

| Server | Tool calls | Context and parallel requests | Built-in auth | Tested by us |
|---|---|---|---|---|
| [vLLM](https://docs.vllm.ai) | `--enable-auto-tool-choice --tool-call-parser <parser>` | `--max-model-len`, `--max-num-seqs` | `--api-key` (only guards `/v1` and a few prefixes) | 0.29.0 (`gpt-oss-20b`, `gpt-oss-120b`) |
| [SGLang](https://docs.sglang.io) | `--tool-call-parser <parser>` | `--context-length`, `--max-running-requests` | `--api-key` | No |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) `llama-server` | `--jinja` (on by default); `--alias` sets the `/v1/models` name | `-c`, `-np` (with `-np`, add `--kv-unified` or `-c` is divided across slots) | `--api-key` | No |
| [Ollama](https://docs.ollama.com) | Built into the model template (`gpt-oss` has it) | `OLLAMA_CONTEXT_LENGTH` (4k to 256k by GPU memory), `OLLAMA_NUM_PARALLEL` (default 1) | None | 0.40.3 (`gpt-oss:20b`) |

The tool parser flag depends on the model family:

- For `gpt-oss`: `openai` in vLLM (with `--reasoning-parser openai_gptoss`), or `gpt-oss` in SGLang (with `--reasoning-parser gpt-oss`).
- Other vLLM parsers include `qwen3_coder`, `glm47`, `kimi_k2`, `deepseek_v32`, and `minimax_m2`.

Without a parser, the server cannot turn model output into tool calls for
`tool_choice: auto`, which the sample harnesses send.

Ollama defaults to one request at a time, a 4k context on GPUs under 24 GB, and
unloading idle models after five minutes. Start it with:
```bash
OLLAMA_CONTEXT_LENGTH=65536 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1 ollama serve
```

## 2. Put the gate in front

[`host/Caddyfile`](../sample_submissions/local-model/host/Caddyfile) configures [Caddy](https://caddyserver.com) as a gate: it passes `/v1/...` requests carrying `Authorization: Bearer $MODEL_API_KEY` to the model server and returns `401` to everything else. Use it in front of every server. vLLM's `--api-key` only guards `/v1` and a few other prefixes, leaving `/invocations` (unauthenticated inference) and control routes like `/scale_elastic_ep` open, and Ollama has no built-in key check at all.

From your entry's folder:

```bash
export MODEL_API_KEY=$(openssl rand -hex 24)
echo "MODEL_API_KEY=$MODEL_API_KEY" >> <competition repository>/local.env   # for --env-file, outside your entry
caddy run --config host/Caddyfile                                            # plain HTTP on port 8090
```

- Set `MODEL_SERVER` if the model server is not on `127.0.0.1:8000` (Ollama's
  default is `127.0.0.1:11434`). The gate rewrites the `Host` header to the
  upstream address because Ollama rejects non-local `Host` headers.
- To run the gate in Docker:
  ```bash
  docker run -d --restart unless-stopped --network host -e MODEL_API_KEY -e MODEL_SERVER \
    -v "$PWD/host/Caddyfile:/etc/caddy/Caddyfile:ro" caddy:2
  ```

## 3. A model server on another machine

If the GPUs are on another machine, run the model server and the gate there and
pass the gate's address across your network: `--model-base-url
http://<machine>:8090/v1` (private addresses are allowed with this flag).

If your machine cannot reach that one (behind NAT on another network), give the
gate a public HTTPS name and pass that instead, for example `--model-base-url
https://llm.example.org/v1`:

- On a machine with a public IPv4 address, run
  `GATE_ADDRESS=llm.example.org caddy run --config host/Caddyfile`, point the
  name's `A` record at the machine, and open ports 80 and 443; Caddy fetches the
  TLS certificate automatically.
- Behind NAT, use a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/) with [`host/cloudflared.yml`](../sample_submissions/local-model/host/cloudflared.yml) (a domain on Cloudflare DNS).

Cloudflare returns `524` if a response does not start within 125 seconds.
Streaming responses start right away, and in our tests streams stayed open
across two-minute pauses between events, but cold-loading a model on demand can
still time out. Also, Cloudflare's Browser Integrity Check (on by default for
custom domains) can return `403` to non-browser user agents; if `kit check` gets
a `403` page from Cloudflare, turn off Browser Integrity Check for the hostname
with a Configuration Rule.

## 4. Check it

From the machine that runs `kit check`, with the gate on `localhost:8090` (use
its address from section 3 if it runs elsewhere):

```bash
# Should print 401 (gate requires the key)
curl -4 -s -o /dev/null -w '%{http_code}\n' http://localhost:8090/v1/models

# Should print the start of a "bash" tool call
curl -4 -sN http://localhost:8090/v1/chat/completions -H "Authorization: Bearer $MODEL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model": "gpt-oss-120b", "stream": true,
  "messages": [{"role": "user", "content": "List the files in /app."}],
  "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object",
  "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]}' | grep -m1 -E '"name": ?"bash"'
```

If the second command prints nothing, run it without `grep` to inspect the raw
response. A plain-text answer usually means tool-call parsing is not enabled on
the model server.

Next, from the root of this repository:

```bash
uv run python -m kit check ../my-harness --env-file local.env --model-base-url http://localhost:8090/v1
uv run python eval/run.py ../my-harness --env-file local.env --model-base-url http://localhost:8090/v1
```

Unless `JUDGE_BASE_URL` or `MODEL_PROXY_URL` is set, `eval/run.py` also uses
your server as the judge ([`eval/README.md`](../eval/README.md#before-you-run)).
`--trials N` sets how many tasks run at once, if your machine has enough CPUs (4
to 6 per task).
