# minimal: a short tool-calling loop on Gemini 3.8 Flash

The smallest complete harness in this repository (~120 lines of `serve.py` built
on `taskbridge.py`) and the easiest starting point for a custom agent. It calls
the Google-sponsored `google-cloud/gemini-3.8-flash` model, so it needs no API
key and no `.env` file.

## How it works

`solve(session)` in `serve.py` implements the entire agent loop:

1. Sends a concise system prompt, the task instruction, and three tools (`bash`, `read_file`, and `write_file`).
2. Executes each tool call inside the task container via `session.exec`,
   `session.read_file`, or `session.write_file`, and returns the output to the
   model along with the remaining task time in minutes.
3. When the model responds without calling a tool, that response becomes
   `final_message`. If a response is empty or truncated at the token limit, the
   loop nudges the model to continue (up to `MAX_NUDGES = 3` times) and
   concatenates truncated replies.
4. Four minutes before the task deadline (`WRAP_UP_SEC`), any in-flight model call is cancelled and the model is asked for its final summary with `tool_choice="none"`.

Every non-empty text response updates `session.final_message` so `taskbridge`
still sends the latest summary if the task is cancelled. Token usage is reported
after each call via `session.usage`. Assistant messages are appended via
`model_dump()` so provider-specific fields (like Gemini's thought signatures)
are preserved across turns.

| File | Purpose |
|---|---|
| `serve.py` | Agent loop and tool definitions |
| `taskbridge.py` | Serve protocol implementation (shared across Python samples) |
| `requirements.txt` | Pinned `openai` and `websockets` dependencies |
| `submission.yaml` | Model declaration, build command, and server startup command |

## Customizing

Because the OpenAI SDK reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` (which the evaluator sets for every model), you can switch to any model in [`docs/models.md`](../../docs/models.md) just by changing `model_api` in `submission.yaml`.

- For models with smaller context windows, set a lower `MAX_OUTPUT_TOKENS` (default `32000`) under `env:` in `submission.yaml`.
- If you want to test locally without a Google Cloud project for `kit/model_proxy`, set `model_api: gemini/gemini-3.8-flash` and put `GEMINI_API_KEY` in `.env`.
- Always generate your `artifacts/` with the same `model_api` you submit.

To keep the code minimal, this starter omits task MCP tools (so `kit check` prints a warning at the `mcp` stage), context compaction, streaming, and deadline-aware retries. See [`custom-openai`](../custom-openai/) for a full harness with those features, or [`plan-verify`](../plan-verify/) for a multi-role planner/worker/verifier loop built on `minimal`.
