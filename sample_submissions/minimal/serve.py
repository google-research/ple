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

"""A minimal harness: one model, three tools, one loop.

The model gets the task and three tools that act in the task container. Each
tool call runs there and its result goes back to the model, until the model
answers without calling a tool. That answer is the final message.
"""

import asyncio
import json
import os

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, high for the others
# (Gemini refuses max). REASONING_EFFORT overrides it.
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if "claude" in MODEL.lower() else "high"
)
WRAP_UP_SEC = 240  # ask for the summary this long before the time limit
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model, so the
# client needs no arguments for them. The SDK retries rate limits and server
# errors on its own.
client = openai.AsyncOpenAI(max_retries=5, timeout=900)

SYSTEM = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. Read the task and the files it points to before you start. Compute with \
code, not by hand. Write each deliverable exactly where and in the format the task asks, then open it again to check \
it. When you are done, reply without calling a tool: a short summary of each deliverable, where it is, and the key \
results."""


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


async def solve(session):
  """Runs the minimal tool-calling loop on `session` until completion.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the model.
  """
  messages = [
      {"role": "system", "content": SYSTEM.format(workdir=session.workdir)},
      {"role": "user", "content": session.instruction},
  ]
  # head: the start of an answer cut off at the output limit
  nudges, head = 0, ""
  while True:
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < WRAP_UP_SEC
    if wrap_up:
      messages.append({
          "role": "user",
          "content": "Time is almost up. Stop and reply with your summary.",
      })
    try:  # a call still running when the wrap-up starts is stopped
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=messages,
              tools=TOOLS,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          None if left is None or wrap_up else left - WRAP_UP_SEC,
      )
    except asyncio.TimeoutError:
      continue
    if reply.usage:
      await session.usage(
          MODEL, reply.usage.prompt_tokens, reply.usage.completion_tokens
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if text:
      # what taskbridge sends if the time runs out first
      session.final_message = text
    # Send the message back as the endpoint returned it: Gemini refuses a tool
    # call that comes back without its thought signature, which is one of the
    # endpoint's own fields.
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return session.final_message
      nudges, head = nudges + 1, text
      if message.content:  # cut off at the output limit
        messages.append(turn)
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue the task, and finish with your summary."
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
      messages[-1][
          "content"
      ] += f"\n[{int(session.time_left() // 60)} minutes left]"


if __name__ == "__main__":
  taskbridge.run(solve)
