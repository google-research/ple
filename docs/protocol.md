# The serve protocol (version 1)

Your harness runs as a WebSocket server. For each task, the evaluator opens one
connection to it, sends the task, and executes the requests your harness sends
back until the task ends. Every frame is a UTF-8 text frame holding one JSON
object of at most 16 MiB.

[`sample_submissions/taskbridge.py`](../sample_submissions/taskbridge.py) implements the protocol for Python harnesses, and [`sample_submissions/node-protocol/`](../sample_submissions/node-protocol/) speaks it directly in JavaScript. This page describes the wire format if you are writing your own client.

## 1. The task message

As soon as the WebSocket connects, the evaluator sends the first message:

```json
{
  "type": "task",
  "protocol": 1,
  "session": "3f2c9a1e0b7d4c55",
  "instruction": "<the task, verbatim>",
  "workdir": "/app",
  "limits": {
    "agent_timeout_sec": 21600,
    "exec_timeout_sec": 600,
    "max_output_bytes": 100000,
    "max_write_bytes": 10000000
  },
  "mcp_servers": [
    {"name": "apps", "url": "http://127.0.0.1:8000/mcp/", "transport": "streamable-http"}
  ]
}
```

- `workdir` is the directory where shell commands start and against which relative file paths are resolved.
- `limits.agent_timeout_sec` is the time the task has left when this message is sent, rounded up to whole seconds (at least `1`), or `null` when untimed.
- `mcp_servers` is usually empty (`[]`). When populated, the task serves project
  applications (documents, spreadsheets, email, CAD, and so on) as MCP tool
  servers at those URLs, reachable only from inside the task container.

## 2. Requests (harness to evaluator)

Every request carries an `id` your harness chooses: a non-empty UTF-8 string of
at most 128 characters, with no unpaired surrogates. Each valid request gets one
`{"type": "result", "id": ...}` reply.

Up to 8 requests can run at once. One-way messages (`log`, `usage`, `done`,
`error`) take no slot and are handled as soon as they are read. If a 9th
request arrives while 8 are already running, the evaluator pauses reading new
frames until one of the 8 finishes, so a `done` sent behind that 9th request
waits too. Run your server without keepalive pings (`ping_interval=None` in
Python `websockets`) or with a `ping_timeout` longer than your longest command.

| Request | Result fields |
|---|---|
| `{"type": "exec", "id": "1", "command": "ls -la", "timeout_sec": 120}` | `exit_code`, `output`, `truncated`, `timed_out` |
| `{"type": "read_file", "id": "2", "path": "/app/data.csv"}` | `ok`, `content_b64`, `size`, `truncated` |
| `{"type": "write_file", "id": "3", "path": "/app/out/report.md", "content_b64": "...", "append": false}` | `ok`, `size` |
| `{"type": "http", "id": "4", "method": "POST", "url": "http://127.0.0.1:8000/mcp/", "headers": {...}, "body_b64": "..."}` | `ok`, `status`, `headers`, `body_b64`, `truncated` |

### Request details

#### `exec`
Runs `bash -c <command>` inside the task container in `workdir` as the task's
user. Each command runs in a fresh subshell: changes to files persist, but `cd`
and exported environment variables do not carry over to the next call.

- `stdout` and `stderr` come back together in `output`. Past 100 KB
  (`limits.max_output_bytes`, counted in UTF-8 bytes after invalid sequences are
  replaced), the evaluator keeps the head and tail and sets `truncated: true`.
- Commands time out after `timeout_sec` seconds (default 120, capped at
  `exec_timeout_sec` and clamped to the task's remaining time minus 2 seconds,
  minimum 1 second). On timeout, `timed_out` is `true` and `exit_code` is `124`. Commands too long for an argv argument are executed via a temporary script
  file (up to the 16 MiB frame limit).

#### `read_file` and `write_file`
Transfer raw file bytes encoded as Base64. Paths may be absolute or relative to
`workdir`.

- Each read returns at most 10 MB, and each write accepts at most 10 MB. To write a larger file, make multiple calls with `append: true`.
- `write_file` creates missing parent directories automatically.
- Both follow symbolic links. `read_file` returns `ok: false` for anything that is not a regular file (such as a FIFO, socket, or device).

#### `http`
Sends an HTTP request from inside the task container to one of the task's
`mcp_servers` URLs (or a path/query under one) and waits up to `timeout_sec`
(default 120 s, max 600 s).

- URLs with `.` or `..` path segments, backslashes, percent-encoded slashes (`%2F`, `%5C`), or path parameters (`;` or `%3B` in the path) are refused.
- Redirects (`3xx`) are not followed; you receive the `3xx` status and its `Location` header.
- Request and response bodies are capped at 10 MB, and larger responses are truncated with `truncated: true`.
- The evaluator's HTTP client drops hop-by-hop and framing headers (`Host`, `Content-Length`, `Transfer-Encoding`, `Connection`, `Keep-Alive`, `Upgrade`, `TE`, `Trailer`, `Proxy-*`) and sets its own.
- Task MCP servers use Streamable HTTP: send JSON-RPC `POST` requests
  (`initialize`, `notifications/initialized`, `tools/list`, `tools/call`)
  with `Accept: application/json, text/event-stream`, reuse the returned
  `Mcp-Session-Id` header, pass the `MCP-Protocol-Version` returned by
  `initialize` on subsequent requests, and handle either JSON or SSE replies.
  Note that a task's tool catalog can hold hundreds of tools (~500 KB).

### Errors and malformed frames

When a request cannot be carried out in the task container (missing file,
permission error, full disk, exhausted container memory or PIDs, or a Docker
file copy stalling past 60 seconds), the evaluator replies with `ok: false` and
an `error` message, and the task continues. A command that runs and returns a
non-zero exit code replies normally with `exit_code` and `output`.

A malformed frame also receives `ok: false` when it has a valid `id` to reply
to, and counts toward the limit of 20 malformed frames per task. A frame is
malformed if it has:

- Invalid JSON or an unknown `type`
- A missing or `null` required field (`id`, `command`, `path`, `url`, or
  `content_b64` on `write_file`), or an empty `id`, `command`, `path`, or
  `url` (an empty `content_b64` is valid and writes an empty file)
- A NUL byte or invalid UTF-8 in `command` or `path`
- A `write_file` payload over 10 MB
- An `http` URL that does not match any of the task's `mcp_servers`
- A `usage` report whose `model` is neither a string nor `null`, or whose token counts are not integers in `0 .. 2^53 - 1` (a missing or `null` count defaults to `0`)

Optional fields (`timeout_sec`, `append`, and `http`'s `method`, `headers`,
and `body_b64`) may be omitted or set to `null`.

## 3. One-way messages and finishing a task

These messages receive no reply:

| Message | Meaning |
|---|---|
| `{"type": "log", "text": "..."}` | Appends a line to your run's log |
| `{"type": "usage", "model": "...", "input_tokens": 1200, "output_tokens": 300, "cache_read_tokens": 0}` | Records token usage for one model call |
| `{"type": "done", "final_message": "..."}` | Finishes the task normally; `final_message` is passed to the grader |
| `{"type": "error", "message": "..."}` | Ends the task early; deliverables already written in the container are still graded |

Rubric graders read `final_message` alongside your deliverable files, so finish
with a clear summary of what you produced and where the files live. When a task
lets you choose a deliverable format, prefer one the graders extract directly:
plain text (`.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.py`,
`.html`, `.xml`), Microsoft Office (`.docx`, `.xlsx`, `.pptx`), or `.pdf`.
Other binary formats (`.odt`, `.ods`, `.odp`, etc.) are listed only by
filename and size.

## 4. Timeouts and network limits

When `limits.agent_timeout_sec` expires:

1. The evaluator sends `{"type": "cancel", "reason": "timeout"}`.
2. In-flight requests inside the task container are stopped and answered with `ok: false` and `"cancelled: the time is up"`, as are any subsequent requests.
3. The evaluator keeps the WebSocket open for up to 10 seconds (or until you send `done` or `error`) so your harness can send a final summary and flush `log` and `usage` messages.

Budget your agent's turns to finish before `agent_timeout_sec`, or catch
`cancel` and immediately send `done` with your best summary so far.

Calls to your model go through the egress proxy, your server's only outbound
network path. It closes any connection that transfers no bytes in either
direction for 30 minutes (1800 s), so stream long model calls to keep the
connection active. It also rejects malformed HTTP headers and ambiguous framing
with `400 Bad Request`.

## 5. Early termination and fault attribution

A task ends early as your harness's outcome (grading whatever deliverables are
already in the container) when:

- Your server closes the WebSocket or crashes before sending `done` or `error`.
- Your server sends more than 20 malformed frames, or any frame larger than 16 MiB.
- Your server fails to accept the WebSocket connection within 120 seconds of starting.
- Your server's outbound connections exhaust the egress proxy's 256 MiB memory limit.

Requests that exhaust the task container's disk, memory, or process limit fail
with their exit code or `ok: false` (`the task container is not answering: out of processes or memory?`).
Because container resource exhaustion comes from your harness's commands, it counts as your harness's outcome and the task is not
re-run.

Faults on the evaluator's side do not count against your harness: a Docker daemon error, a vanished task container, the egress proxy stopping (unless your own traffic exhausted its memory), or a host/network fault. The evaluator aborts the trial and re-runs the task ([Scoring](rules.md#scoring)).

When a trial ends (normally, by timeout, or by cancellation), the evaluator stops all processes started inside the task container during the trial—regardless of user or environment—before collecting deliverables for grading, ensuring files cannot change while being graded. During a local dev-set run (`eval/run.py`), deliverables are also exported to your [artifacts](artifacts.md) as soon as the session ends. Before copying, the exporter stops processes launched by your commands using an environment marker; however, any background process started with a cleared environment (`env -i`) will not carry that marker and could continue modifying files during the copy. Always stop such processes yourself before sending `done`.

## 6. Minimal server example

This Python server shows the wire protocol one request at a time; see [`node-protocol/serve.mjs`](../sample_submissions/node-protocol/serve.mjs) for a full implementation with an agent loop.

```python
import asyncio, json, os
import websockets

async def session(ws):
    task = json.loads(await ws.recv())
    n = 0

    async def call(request):                     # send one request and wait for its result
        nonlocal n
        n += 1
        await ws.send(json.dumps({**request, "id": str(n)}))
        while True:
            message = json.loads(await ws.recv())
            if message.get("type") == "cancel":
                raise TimeoutError
            if message.get("type") == "result" and message.get("id") == str(n):
                return message

    summary = "Nothing delivered yet."
    try:
        listing = await call({"type": "exec", "command": "ls -la", "timeout_sec": 60})
        # ... call your model, run its tool calls via call(...), and update summary ...
        summary = "Wrote /app/report.md: ..."
    except TimeoutError:
        pass                                     # time is up: send done with what we have so far
    await ws.send(json.dumps({"type": "done", "final_message": summary}))

async def main():
    async with websockets.serve(
        session, "0.0.0.0", int(os.environ["PORT"]), max_size=16 * 2**20, ping_interval=None
    ):
        await asyncio.Future()

asyncio.run(main())
```
