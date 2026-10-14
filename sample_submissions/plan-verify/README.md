# plan-verify: planner, worker, and verifier on Claude Opus 5.5

Builds on [`minimal`](../minimal/) by running the same agent loop across three
distinct roles (planner, worker, and verifier) in separate conversations using
`google-cloud/claude-opus-5-5`. Run `diff -u ../minimal/serve.py serve.py` to
see the exact changes from `minimal`.

## How it works

`agent()` wraps `minimal`'s tool-calling loop as a reusable function that takes
a message history, a tool list, and a step deadline. `solve()` calls it across
three stages:

1. The planner (`bash`, `read_file`) explores the project files and writes a
   structured plan covering each required deliverable, its path and format, the
   engineering method, and how to verify the result.
2. The worker (`bash`, `read_file`, `write_file`) takes the task instruction and the planner's plan, produces the deliverables, and writes the summary used as `final_message`.
3. The verifier (`bash`, `read_file`) starts a fresh conversation containing
   only the task instruction, the plan, and the worker's summary (without the
   worker's intermediate reasoning). It inspects the generated deliverables and
   replies either `PASS` or a numbered list of issues. Any issues are fed back
   into the worker's conversation for a fix round (up to two verify-and-fix
   rounds total).

### Time budget allocation

| Step | Wraps up when |
|---|---|
| Planner | 10% of total task time is used (`PLAN_SHARE`) |
| Worker | 25% of total task time remains (`CHECK_SHARE`) |
| Verifier | 12.5% of total task time remains |
| Fix round | 4 minutes remain (`WRAP_UP_SEC`) |

A verification or fix step only starts if at least 2 minutes remain before its
wrap-up deadline (`MIN_STEP_SEC`); otherwise it is skipped so the harness
doesn't start a toolless turn. Each step's final wrap-up call gets up to 4
minutes (`WRAP_UP_SEC`). Like `minimal`, this starter does not call task MCP
servers, so `kit check` prints a warning at the `mcp` stage.

| File | Purpose |
|---|---|
| `serve.py` | Multi-role loop, time budgeting, and role prompts |
| `taskbridge.py` | Serve protocol implementation (shared across Python samples) |
| `requirements.txt` | Pinned `openai` and `websockets` dependencies |
| `submission.yaml` | Model declaration, build command, and server startup command |

## Customizing

You can switch `model_api` in `submission.yaml` to any model in [`docs/models.md`](../../docs/models.md). The code reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`, and `MAX_OUTPUT_TOKENS` (default `32000`, configurable under `env:`). Role prompts (`PLANNER`, `WORKER`, `VERIFIER`) and time-share constants live at the top of `serve.py`.
