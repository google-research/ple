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

"""Antigravity CLI (`agy`) harness that runs one headless session per task.

`agy` runs in the serve container with its built-in local tools disabled
(`tools: []`). Instead, the model interacts with the task container through MCP:
`task_mcp.py` exposes container tools (`shell`, `read_file`, `write_file`,
`edit_file`) over `taskbridge.Relay`, and the task's own MCP servers are
proxied through the same relay. All tools are loaded eagerly so the model sees
every schema up front.

Because a hung model call could block until the task deadline, any run whose
output is silent for `STALL_SEC` is terminated and its conversation is resumed,
up to `MAX_RUNS` total runs.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import pathlib
import signal
import sys
import tempfile
import time
from typing import Any

import taskbridge

HERE = pathlib.Path(__file__).resolve().parent
AGY = "/opt/harness/bin/agy"
EFFORT = os.environ.get("AGY_EFFORT", "high")
STOP_BEFORE_SEC = 90
STALL_SEC = 900  # longer than one command may run (600 s)
MAX_RUNS = 4
# bytes; Linux refuses one argument over 128 KiB (MAX_ARG_STRLEN)
MAX_ARGV_PROMPT = 100_000
TASK_TOOLS = ("shell", "read_file", "write_file", "edit_file")
RESUME = (
    "Your previous run was interrupted. Continue the task where you left off."
)

AGENT = """---
name: task-agent
description: Works an engineering task in a remote Linux container through the task tools.
tools: []
---

# System Prompt
You are an engineering agent. The task's files are in a remote Linux container with no internet access; your local
workspace is empty and is not where the work happens. Act on the container only through the tools of the `{bridge}`
server: `mcp_{bridge}_shell` runs bash there (each call starts in {workdir}), `mcp_{bridge}_read_file`,
`mcp_{bridge}_write_file` and `mcp_{bridge}_edit_file` read and change its files. Paths are container paths. {apps}

How to work:
- Read the whole instruction first, then inspect the files it points to before deciding anything.
- Work in small, verified steps. Run calculations with python3 in the container rather than by hand.
- Write each deliverable exactly where and in the format the instruction asks (path, file name, file type), early, and
  improve it as you go. Check that it opens and holds what was asked.
- You have about {minutes} minutes. Leave time to finish the deliverables.
- End with a short summary of what you delivered and where, the key results, and any assumptions you made.
"""


def prompt_input(prompt: str) -> tuple[list[str], bytes | None]:
  """Builds `agy` command-line prompt arguments and optional stdin payload.

  Prompts exceeding `MAX_ARGV_PROMPT` bytes are passed via standard input as a
  single `stream-json` user message using `--input-format stream-json`.

  Args:
    prompt: The user prompt string to send to `agy`.

  Returns:
    A tuple of `(cli_args, stdin_bytes)`.
  """
  if len(prompt.encode()) <= MAX_ARGV_PROMPT:
    return ["-p", prompt], None
  message = {"event": "user", "message": {"role": "user", "content": prompt}}
  return ["--input-format", "stream-json", "-p="], (
      json.dumps(message) + "\n"
  ).encode()


def model_slug() -> str:
  """Returns the `agy` model identifier string.

  Uses `AGY_MODEL` if set; otherwise strips `-preview` from `MODEL_NAME` and
  appends `EFFORT` (e.g. `gemini-3.8-flash-high`).

  Returns:
    The model slug string for `agy --model`.
  """
  return (
      os.environ.get("AGY_MODEL")
      or f"{os.environ['MODEL_NAME'].removesuffix('-preview')}-{EFFORT}"
  )


async def mcp_tools(session: taskbridge.Session, url: str) -> list[str]:
  """Lists the tool names offered by one of the task's MCP servers.

  Args:
    session: Active task session used to reach the MCP server.
    url: URL of the MCP server inside the task container.

  Returns:
    A list of tool names exposed by the server.
  """
  client = taskbridge.MCPClient(session, url, client="agy-gemini")
  await client.initialize()
  return [tool["name"] for tool in await client.list_tools()]


def read_stream(path: pathlib.Path) -> dict[str, Any]:
  """Parses a single run's `stream-json` output file.

  Args:
    path: Path to the JSONL stream output file written by `agy`.

  Returns:
    A dictionary containing `conversation`, `response`, `status`, `error`,
    `usage`, `tools` (Counter), and `texts` (list of non-empty agent responses).
  """
  out: dict[str, Any] = {
      "conversation": None,
      "response": None,
      "status": None,
      "error": None,
      "usage": {},
      "tools": collections.Counter(),
      "texts": [],
  }
  texts: dict[int, str] = {}
  lines = path.read_text(errors="replace").splitlines() if path.exists() else []
  for line in lines:
    try:
      event = json.loads(line)
    except ValueError:
      continue
    if not isinstance(event, dict):
      continue
    step = (
        event["step_update"]
        if isinstance(event.get("step_update"), dict)
        else {}
    )
    if event.get("event") == "init":
      out["conversation"] = event.get("conversation_id")
    elif step.get("step_type") == "agent_response" and step.get("text_delta"):
      index = (
          step["step_index"] if isinstance(step.get("step_index"), int) else 0
      )
      texts[index] = texts.get(index, "") + str(step["text_delta"])
    elif step.get("step_type") == "tool" and step.get("state") == "DONE":
      out["tools"][str(step.get("tool_name") or "?")] += 1
    elif event.get("event") == "result":
      result = event["result"] if isinstance(event.get("result"), dict) else {}
      response = result.get("response")
      out.update(
          response=response if isinstance(response, str) else None,
          status=result.get("status"),
          error=result.get("error"),
          usage=result["usage"]
          if isinstance(result.get("usage"), dict)
          else {},
      )
  out["texts"] = [texts[i].strip() for i in sorted(texts) if texts[i].strip()]
  return out


NO_ANSWER = (
    "Stopped before a final answer; the work so far is in the task container."
)


def final_message(runs: list[dict[str, Any]]) -> str:
  """Extracts the final answer from completed `agy` runs.

  Prefers the last run's final response, falling back to the last streamed text
  if `agy` was stopped early. Strips any leading JSON line echoed from the
  finish tool's arguments.

  Args:
    runs: List of parsed stream dictionaries from `read_stream`.

  Returns:
    The cleaned final answer string, or `""` if no text was produced.
  """
  texts: list[str] = []
  for run in runs:
    texts.extend(run["texts"])
  answer = (runs[-1]["response"] or "").strip() or (texts[-1] if texts else "")
  first, _, rest = answer.partition("\n")
  if first.startswith("{") and '"toolAction"' in first:
    answer = rest.strip()
  return answer


async def watch(stream: pathlib.Path, pidfile: pathlib.Path) -> None:
  """Terminates the `agy` process group if its stream is silent for `STALL_SEC`.

  Args:
    stream: Path to the run's JSONL stream output file.
    pidfile: Path to the file holding the `agy` process group leader PID.
  """
  while True:
    await asyncio.sleep(30)
    with contextlib.suppress(OSError):
      if time.time() - stream.stat().st_mtime > STALL_SEC:
        with contextlib.suppress(OSError, ValueError):
          os.killpg(int(pidfile.read_text()), signal.SIGTERM)
        return


async def configure(
    session: taskbridge.Session, relay: taskbridge.Relay, home: pathlib.Path
) -> None:
  """Writes `agy` settings, MCP server configs, and the custom agent definition.

  Args:
    session: Active task session.
    relay: Local HTTP relay for the task container.
    home: Temporary `HOME` directory for `agy`.
  """
  # task_mcp.py's server, under a name none of the task's servers has
  bridge, n = "task", 1
  while bridge in {server["name"] for server in session.mcp_servers}:
    n += 1
    bridge = f"task{n}"
  servers = {
      bridge: {
          "command": sys.executable,
          "args": [str(HERE / "task_mcp.py")],
          "env": {"TASK_RELAY_URL": relay.url, "TASK_WORKDIR": session.workdir},
          "tools": {name: {"eager": True} for name in TASK_TOOLS},
      }
  }
  apps = []
  for server in session.mcp_servers:
    try:
      names = await mcp_tools(session, server["url"])
    except Exception as exc:  # pylint: disable=broad-exception-caught
      await session.log(
          f"MCP server {server['name']}: could not list its tools ({exc})"
      )
      continue
    await session.log(f"MCP server {server['name']}: {len(names)} tools")
    servers[server["name"]] = {
        "serverUrl": relay.mcp_url(server["name"]),
        "tools": {name: {"eager": True} for name in names},
    }
    apps.append(
        f"`{server['name']}` ({len(names)} tools, named"
        f" mcp_{server['name']}_<tool>)"
    )
  config = home / ".gemini" / "config"
  (config / "agents").mkdir(parents=True)
  (config / "mcp_config.json").write_text(
      json.dumps({"mcpServers": servers}, indent=2)
  )
  (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
  (home / ".gemini" / "antigravity-cli" / "settings.json").write_text(
      json.dumps({"modelProvider": "gemini"})
  )
  note = (
      f"The task also offers apps through these MCP servers: {', '.join(apps)}."
      if apps
      else ""
  )
  left = session.time_left()
  minutes = "unlimited" if left is None else int(left // 60)
  (config / "agents" / "task-agent.md").write_text(
      AGENT.format(
          workdir=session.workdir, minutes=minutes, apps=note, bridge=bridge
      )
  )
  (home / "workspace").mkdir()


async def solve(session: taskbridge.Session) -> str:
  """Runs `agy` on the task, resuming on stall, and returns the final message.

  Args:
    session: Active task session.

  Returns:
    The final response string from `agy`, or `NO_ANSWER` if none was produced.

  Raises:
    RuntimeError: If `agy` exits with an error before using any tools or
      streaming any text.
  """
  runs: list[dict[str, Any]] = []
  with tempfile.TemporaryDirectory(
      prefix="agy-",
      dir=os.environ.get("HOME") or "/tmp",
      ignore_cleanup_errors=True,
  ) as scratch:
    home = pathlib.Path(scratch)
    async with taskbridge.Relay(session) as relay:
      await configure(session, relay, home)
      prompt, conversation = session.instruction, None
      for n in range(MAX_RUNS):
        stream, pidfile = home / f"stream{n}.jsonl", home / f"agy{n}.pid"
        env = relay.env({
            **os.environ,
            "HOME": str(home),
            "AGY_STREAM": str(stream),
            "AGY_PID": str(pidfile),
            "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
        })
        prompt_args, stdin = prompt_input(prompt)
        # the shell writes its pid and then becomes agy, so watch() can signal
        # agy's process group
        argv = [
            "/bin/sh",
            "-c",
            'echo $$ > "$AGY_PID"; exec "$0" "$@" > "$AGY_STREAM"',
            AGY,
            *prompt_args,
            "--agent",
            "task-agent",
            "--model",
            model_slug(),
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            *(["--conversation", conversation] if conversation else []),
        ]
        watcher = asyncio.create_task(watch(stream, pidfile))
        try:
          await taskbridge.run_process(
              argv,
              session,
              env=env,
              cwd=str(home / "workspace"),
              stdin=stdin,
              stop_before_sec=STOP_BEFORE_SEC,
              log_prefix="agy: ",
          )
        finally:
          stalled = watcher.done() and not watcher.cancelled()
          watcher.cancel()
          runs.append(read_stream(stream))
          session.final_message = final_message(runs) or session.final_message
        conversation = runs[-1]["conversation"] or conversation
        left = session.time_left()
        # resume only with at least two minutes to work before run_process would
        # stop agy
        if (
            not stalled
            or not conversation
            or (left is not None and left < STOP_BEFORE_SEC + 120)
        ):
          break
        await session.log(
            f"agy was silent for {STALL_SEC // 60} minutes; resuming its"
            " conversation"
        )
        prompt = RESUME
  usage = collections.Counter()
  for run in runs:
    usage.update({k: v for k, v in run["usage"].items() if isinstance(v, int)})
  await session.usage(
      os.environ["MODEL_NAME"],
      usage["input_tokens"],
      usage["output_tokens"],
      usage["cache_read_tokens"],
  )
  tools = sum((run["tools"] for run in runs), collections.Counter())
  await session.log(
      f"agy: {len(runs)} run(s), last status {runs[-1]['status']}, tools"
      f" {dict(tools)}"
  )
  if (
      runs[-1]["status"] == "ERROR"
      and not tools
      and not any(run["texts"] for run in runs)
  ):
    raise RuntimeError(f"agy: {runs[-1]['error']}")
  return session.final_message or NO_ANSWER


if __name__ == "__main__":
  taskbridge.run(solve)
