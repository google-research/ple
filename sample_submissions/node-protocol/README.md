# node-protocol: the serve protocol in JavaScript, on Claude Opus 5.5

A single-file JavaScript harness (`serve.mjs`) that implements
[the serve protocol](../../docs/protocol.md) directly over WebSockets without
`taskbridge.py` or a provider SDK. It calls the Google-sponsored
`google-cloud/claude-opus-5-5` model via standard `fetch` against its
OpenAI-compatible endpoint. Any language with a WebSocket server and HTTP client
can integrate the same way.

## How it works

The agent loop mirrors [`minimal`](../minimal/): three tools (`bash`, `read_file`, `write_file`), remaining minutes appended after each tool round, a final summary requested four minutes before the deadline, and `MAX_OUTPUT_TOKENS` (default `32000`) per model call. Like `minimal`, it does not call task MCP servers (`kit check` will warn at the `mcp` stage).

Inside `serve.mjs`, the protocol layer reads the initial `task` message to
track the task deadline, pending requests, and latest summary. Each `exec`,
`read_file`, or `write_file` call gets an incrementing `id` and resolves its
promise when the matching `result` arrives (the protocol supports up to 8
concurrent requests in flight). Outgoing requests are validated before sending
so empty paths/commands, NUL bytes, or unpaired surrogates return to the model
as tool errors instead of triggering malformed-frame penalties on the evaluator,
and incoming WebSocket frames are capped at 16 MiB (`maxPayload`).

Model calls stream SSE responses and preserve provider-specific reasoning
fields (`reasoning_content`, Gemini thought signatures, and Claude signed
`thinking_blocks`). Because Node's built-in `fetch` drops streams that stay
silent for 5 minutes (shorter than a max-effort reasoning pause), `serve.mjs`
uses `fetch` from the pinned `undici` package with a 30-minute body/header
timeout (`MODEL_IDLE_SEC`, matching the egress proxy) and routes traffic
through `HTTPS_PROXY` / `HTTP_PROXY`. On `cancel` or socket close, the server
aborts any in-flight `fetch` call, rejects pending tool requests, and
immediately sends `done` with the latest summary (including partial streamed
text) within the 10-second grace window.

Transient failures (`408`, `409`, `429`, `5xx`, dropped connections, mid-stream
errors other than invalid-request codes, or streams that end without a finish
reason or `[DONE]`) are logged and retried with backoff (`RETRY_WAITS`, 5
seconds to 2 minutes) without adding partial output to the conversation. If a
`400` indicates the context window is full, the older half of long tool outputs
is trimmed to the first 1,000 characters (`CUT_KEEP`) and the call is retried.

| File | Purpose |
|---|---|
| `serve.mjs` | WebSocket server, protocol client, and agent loop |
| `package.json`, `package-lock.json` | Pinned `ws` and `undici` dependencies |
| `submission.yaml` | Runs `npm ci` at build time and `node serve.mjs` at runtime |

## Running locally outside Docker

To run `serve.mjs` directly on your host (for example, under a debugger):

```bash
npm ci
PORT=8765 MODEL_NAME=claude-opus-5-5 MODEL_BASE_URL=http://localhost:4000/v1 MODEL_API_KEY=$MODEL_PROXY_KEY node serve.mjs
```

Then point `eval/run.py` at it from the root of the competition repository:

```bash
uv run python eval/run.py --url ws://127.0.0.1:8765 --name node-protocol --tasks data-center-site-model --env-file local.env
```
