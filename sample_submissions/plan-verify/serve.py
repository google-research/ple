# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan-work-verify harness running three specialized agent roles per task.

1. The planner inspects the task's files and writes a plan describing each
   deliverable, its path and format, and how to verify it.
2. The worker executes the task using the plan; its answer becomes the final
   message.
3. The verifier starts with a fresh conversation containing only the task, the
   plan, and the worker's summary. It inspects the deliverables and replies
   `PASS` or lists specific gaps, which are sent back to the worker for up to
   `MAX_FIX_ROUNDS` rounds while sufficient time remains.

The planner and verifier receive only `bash` and `read_file` and are instructed
not to modify any files.
"""

import asyncio
import json
import os
import re

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
# a fix round wraps up this long before the time limit, and a wrap-up call may
# take as long
WRAP_UP_SEC = 240
WRAP_UP = {
    "role": "user",
    "content": "Time is up for this step. Stop and reply with your answer.",
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()
# the planner wraps up once it has used this share of the task's time
PLAN_SHARE = 0.1
# the worker wraps up when this share is left, to keep time for checking and
# fixing
CHECK_SHARE = 0.25
# a check or a fix round starts only with this long to work before its own
# wrap-up
MIN_STEP_SEC = 120
MAX_FIX_ROUNDS = 2

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Because
# calls are non-streaming, a max-effort response can take over 15 minutes
# before any bytes arrive, so the timeout matches the egress proxy's 30-minute
# idle limit.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

PLANNER = """You plan an engineering task that another engineer will carry out in a Linux container with no internet \
access. Inspect the task's files with your tools (bash starts in {workdir}), but do not create or change any file. \
Then reply with the plan: each deliverable with its exact path and format, what it must contain, the data and method \
to use, and how to check it."""

WORKER = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. A plan for the task follows it; use it, and correct it where the files \
show it is wrong. Compute with code, not by hand. Write each deliverable exactly where and in the format the task \
asks, then open it again to check it. When you are done, reply without calling a tool: a short summary of each \
deliverable, where it is, and the key results."""

VERIFIER = """You check an engineer's work on a task in a Linux container. You get the task, the plan, and the \
engineer's summary of what was delivered. Open every deliverable with your tools (bash starts in {workdir}) and check \
it against the task: the path, the format, every part the task asks for, and the key numbers, recomputed. Do not \
create or change any file. If everything holds, reply with the single word PASS and nothing else. Otherwise reply \
with a numbered list of the gaps, each specific enough to fix."""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]
READ_ONLY = TOOLS[:2]


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


async def run_tool(session, fn):
  """Executes a single tool call in the task container and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Any tool failure is
  returned as an error string for the model.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    fn: Function-call dictionary containing `name` and `arguments`.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate
  except (taskbridge.TaskError, KeyError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a rate
  limit, server error, or connection drop, sleeps for the next interval in
  `RETRY_WAITS` (capped at the step's wrap-up deadline). Errors are logged
  rather than added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the step wraps up.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


def conversation(system, session, prompt):
  """Creates the initial `[system, user]` message list for a role.

  Args:
    system: System prompt template containing `{workdir}`.
    session: Active `taskbridge.Session` providing `workdir`.
    prompt: Initial user message content.

  Returns:
    A two-element list of OpenAI chat message dictionaries.
  """
  return [
      {"role": "system", "content": system.format(workdir=session.workdir)},
      {"role": "user", "content": prompt},
  ]


async def agent(session, messages, tools, until, final=False):
  """Runs the tool-calling loop on `messages` until the model answers.

  Prompts the model to wrap up immediately once fewer than `until` seconds
  remain on the task clock. When `final` is `True`, updates
  `session.final_message` with each non-empty response.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: Mutable conversation history list.
    tools: List of OpenAI tool definitions available to this role.
    until: Remaining task time threshold (in seconds) at which this step wraps
      up.
    final: Whether to record responses in `session.final_message`.

  Returns:
    The final text response produced by this agent step.
  """
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  while True:
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < until
    if wrap_up and messages[-1:] != [WRAP_UP]:  # asked once, also on a retry
      messages.append(dict(WRAP_UP))
    # A call still running when the wrap-up starts is stopped. The wrap-up call
    # itself gets at most WRAP_UP_SEC and stops at the time limit; if it takes
    # longer, the step returns head, its answer so far.
    if left is None:
      timeout = None
    elif wrap_up:
      timeout = min(WRAP_UP_SEC, left)
    else:
      timeout = left - until
    try:
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(messages),
              tools=tools,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          timeout,
      )
    except asyncio.TimeoutError:
      if wrap_up:
        return head
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, until):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if final and text:
      session.final_message = text
    # sent back as returned: Gemini refuses a tool call without its thought
    # signature
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      turn.pop("tool_calls", None)  # calls made at the wrap-up are not run
      if message.content:
        # a fix round, or the request to go on after a cut-off reply, continues
        # from this answer
        messages.append(turn)
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return text
      nudges, head = nudges + 1, text
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue, and finish with your answer."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"])
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1]["content"] += (
          f"\n[{int((session.time_left() - until) // 60)} minutes left for this"
          " step]"
      )


async def solve(session):
  """Executes the planner, worker, and verifier loop for `session`.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the worker.
  """
  total = session.time_left()

  def until(share):
    """Computes the remaining seconds when `share` of total task time is left.

    Args:
      share: Fraction of total task time remaining (`0.0` to `1.0`).

    Returns:
      Remaining seconds threshold, or `0` if the task has no time limit.
    """
    return 0 if total is None else total * share

  def time_for(wrap_up):
    """Checks whether at least `MIN_STEP_SEC` remains before `wrap_up`.

    Args:
      wrap_up: Remaining seconds threshold at which the next step wraps up.

    Returns:
      `True` if there is no deadline or at least `MIN_STEP_SEC` before
      `wrap_up`.
    """
    left = session.time_left()
    return left is None or left - wrap_up >= MIN_STEP_SEC

  task = session.instruction
  plan = await agent(
      session,
      conversation(PLANNER, session, task),
      READ_ONLY,
      until(1 - PLAN_SHARE),
  )
  await session.log(f"plan: {plan[:1000]}")
  work = conversation(WORKER, session, f"{task}\n\n## Plan\n{plan}")
  summary = await agent(session, work, TOOLS, until(CHECK_SHARE), final=True)
  for _ in range(MAX_FIX_ROUNDS):
    if not time_for(until(CHECK_SHARE / 2)):
      break
    review = (
        f"## Task\n{task}\n\n## Plan\n{plan}\n\n## The engineer's"
        f" summary\n{summary}"
    )
    verdict = await agent(
        session,
        conversation(VERIFIER, session, review),
        READ_ONLY,
        until(CHECK_SHARE / 2),
    )
    await session.log(f"verifier: {verdict[:1000]}")
    # PASS alone in any letter case (Markdown emphasis and a full stop aside),
    # or an empty reply, names no gaps
    if verdict.strip("*_. \r\n\t").upper() in ("", "PASS") or not time_for(
        WRAP_UP_SEC
    ):
      break
    work.append({
        "role": "user",
        "content": (
            "A reviewer checked your files and found these"
            f" gaps:\n{verdict}\n\nFix them, then reply with your updated"
            " summary."
        ),
    })
    summary = (
        await agent(session, work, TOOLS, WRAP_UP_SEC, final=True) or summary
    )
  return summary or session.final_message


if __name__ == "__main__":
  taskbridge.run(solve)
