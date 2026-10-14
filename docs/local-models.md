# Serving your own model

An entry with `model_api: openai-compatible/<model>` calls a model server you
run at the `model_base_url` in `submission.yaml`. Our evaluation server calls
that URL over the internet from Google Cloud for the smoke test, the dev
reproduction check, and the test set.

A clean way to expose a model on your own GPUs is to start the model server on `127.0.0.1`, put a small auth gate in front of it that checks your key, and publish the gate under a public HTTPS name. [`sample_submissions/local-model/`](../sample_submissions/local-model/) is a complete entry set up this way, with the gate and tunnel configs in its `host/` folder.

## Rules for `model_base_url`

The kit and the evaluation server enforce these rules:

- It must be an `https://` URL on a DNS name. IP addresses, single-label names, and names ending in `.local`, `.internal`, or `.localhost` are refused.
- Every address the hostname resolves to must be public. The egress proxy
  refuses private and shared ranges, including Tailscale (`100.64.0.0/10`). The
  evaluation server has no IPv6, so the hostname needs an `A` record.
- Any port works (for example, `https://llm.example.org:8443/v1`).
- The path normally ends in `/v1`, and clients append `/chat/completions`.
- The part of `model_api` after `openai-compatible/` is sent as the `model`
  field in API requests, so it must match a name your server accepts (listed at
  `/v1/models`). For `openai-compatible/gpt-oss:20b`, the harness sends
  `gpt-oss:20b`.
- Put the key in `.env` as `MODEL_API_KEY`. The harness sends it as `Authorization: Bearer <key>`.
- The egress proxy closes any connection that transfers no bytes in either direction for 30 minutes.

## What the endpoint has to handle

- Tool calls in both streamed and non-streamed responses (the `local-model` sample streams every call).
- About 12 tasks at once, each with one model call in flight.
- Continuous uptime from the moment you submit: the submission waits in the queue and then runs for up to a day (each task allows 6 hours). Errors and timeouts from your endpoint count against your run ([Scoring](rules.md#scoring)).
- The test set's prompts and files. Turn off request logging on the model server and anything in front of it, and never store task content ([Fair play](rules.md#fair-play)).

## 1. Start the model server

Run your server on `127.0.0.1:8000` with tool calling enabled. For
`gpt-oss-120b` with vLLM:

```bash
vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
```

On a single 80 GB GPU, vLLM 0.29.0 starts `gpt-oss-120b` with default settings,
but only has room for about 180,000 tokens of KV cache, so 12 long sessions will
wait on each other. Using two GPUs (`--tensor-parallel-size 2`) leaves much more
room.

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
echo "MODEL_API_KEY=$MODEL_API_KEY" > .env                       # the harness sends this same key
caddy run --config host/Caddyfile                                # plain HTTP on port 8090 (for a tunnel)
GATE_ADDRESS=llm.example.org caddy run --config host/Caddyfile   # or HTTPS on a machine with a public IP
```

- On a machine with a public IP, point the domain's `A` record at the machine and open ports 80 and 443; Caddy fetches the TLS certificate automatically.
- Set `MODEL_SERVER` if the model server is not on `127.0.0.1:8000` (Ollama's
  default is `127.0.0.1:11434`). The gate rewrites the `Host` header to the
  upstream address because Ollama rejects non-local `Host` headers.
- To run the gate in Docker:
  ```bash
  docker run -d --restart unless-stopped --network host -e MODEL_API_KEY -e MODEL_SERVER \
    -v "$PWD/host/Caddyfile:/etc/caddy/Caddyfile:ro" caddy:2
  ```
  On a public IP, add `-e GATE_ADDRESS -v caddy-data:/data` so the certificate
  survives restarts.

## 3. Publish the gate

| Method | Behind NAT | What you need | Verdict |
|---|---|---|---|
| Public IP with Caddy serving HTTPS | No | Public IPv4 address, ports 80 and 443, DNS name ([DuckDNS](https://www.duckdns.org) is free) | Best choice if the machine has a public IPv4 address |
| [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/) with [`host/cloudflared.yml`](../sample_submissions/local-model/host/cloudflared.yml) | Yes | Domain using Cloudflare DNS (free plan works) | Best choice behind NAT if you have a domain; TLS terminates at Cloudflare |
| [Tailscale Funnel](https://tailscale.com/kb/1223/funnel): `tailscale funnel 8090` | Yes | Tailscale account (Funnel is in beta) | Good behind NAT without a domain (public ports 443, 8443, and 10000 only) |
| SSH reverse tunnel to a small cloud VM | Yes | Cloud VM with a public IPv4 address | Good behind NAT without a tunnel service; run Caddy on the VM |
| Cloudflare quick tunnel (`--url http://127.0.0.1:8090`) | Yes | Nothing | For testing only: hostname changes on restart and has no uptime guarantee |
| `ngrok` (free plan) | Yes | Account | For testing only: 1 GB transfer and 20,000 requests/month can run out mid-eval |

### Notes on tunnels

Cloudflare returns `524` if a response does not start within 125 seconds.
Streaming responses start right away, and in our tests streams stayed open
across two-minute pauses between events, but cold-loading a model on demand can
still time out. Also, Cloudflare's Browser Integrity Check (on by default for
custom domains) can return `403` to non-browser user agents; if `kit check` gets
a `403` page from Cloudflare, turn off Browser Integrity Check for the hostname
with a Configuration Rule.

With Tailscale Funnel, a Funnel hostname resolves to a `100.x` address inside
your own tailnet, which the egress proxy refuses. Always test Funnel URLs from
outside your tailnet.

For an SSH reverse tunnel, run Caddy on the VM with `GATE_ADDRESS` set to the
VM's DNS name, and keep the tunnel open from your GPU machine using systemd or a
loop:
```bash
ssh -NT -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R 8000:127.0.0.1:8000 <vm>
```

We tested the gate in front of vLLM and Ollama published through a Cloudflare
quick tunnel (`kit check` passed against both, and `eval/run.py` ran dev tasks
against vLLM).

## 4. Check it the way the evaluation server will

From a machine outside your network (such as a cloud VM or Cloud Shell, since
your LAN or tailnet may resolve the hostname differently):

```bash
# Should print 401 (gate requires the key)
curl -4 -s -o /dev/null -w '%{http_code}\n' https://llm.example.org/v1/models

# Should print the start of a "bash" tool call
curl -4 -sN https://llm.example.org/v1/chat/completions -H "Authorization: Bearer $MODEL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model": "gpt-oss-120b", "stream": true,
  "messages": [{"role": "user", "content": "List the files in /app."}],
  "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object",
  "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]}' | grep -m1 -E '"name": ?"bash"'
```

If the second command prints nothing, run it without `grep` to inspect the raw
response. A plain-text answer usually means tool-call parsing is not enabled on
the model server.

Next, from the root of this repository, run `kit check` and `eval/run.py`
without `--model-base-url` so requests go through the egress proxy to your
public URL:

```bash
uv run python -m kit check ../my-harness
uv run python eval/run.py ../my-harness
```

The evaluation server runs about 12 tasks at once. You can test similar load
locally with `eval/run.py --trials 9` if your machine has enough CPUs (4 to 6
per task). Before your public hostname is live,
`--model-base-url http://localhost:8090/v1` points both commands at your local
gate, though `kit check` will only print `ready to submit` when run without
`--model-base-url`.

## Keep it running

Run the model server, gate, and tunnel as services that restart on failure and
start on boot (systemd units or Docker with `--restart unless-stopped`). Keep
them running until your submission page shows the evaluation has finished.
Rotate the key when the competition ends.
