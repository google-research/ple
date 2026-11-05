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

Runs that lose their connection before answering, hit a rate limit, or produce
no output for `STALL_SEC` seconds are automatically resumed in the same
conversation (up to `MAX_RUNS` total runs).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import pathlib
import re
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
# agy is stopped and resumed after this long without output (longer than one
# command may run, 600 s)
STALL_SEC = 2400
WATCH_SEC = 30
# SIGKILL follows SIGTERM after this long
KILL_GRACE_SEC = 10
MAX_RUNS = 4
# a rate-limited run is resumed after this long times its number
BACKOFF_SEC = 60
# bytes; Linux refuses one argument over 128 KiB (MAX_ARG_STRLEN)
MAX_ARGV_PROMPT = 100_000
TASK_TOOLS = ("shell", "read_file", "write_file", "edit_file")
# Models built into agy 1.4.10 for a Gemini API key (`agy models`); agy applies
# its own compaction limit to each
CATALOG = (
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
)
# Any other model runs as a custom model, which agy compacts only if
# settings.json gives it a token budget, once its token estimate passes the
# budget minus its output reserve (about 70k tokens, measured on agy 1.3.3
# and 1.4.10)
CUSTOM_SUFFIX = "-byom"
AGY_RESERVE = 70_000
# tokens; Gemini's context window (set AGY_CONTEXT_WINDOW for another)
CONTEXT_WINDOW = int(os.environ.get("AGY_CONTEXT_WINDOW") or 1_048_576)
RESUME = (
    "Your previous run was interrupted. Continue the task where you left off."
)
# How agy (a Go binary) reports a connection it lost or could not make, and the
# gateway's 500 for a stream its upstream ended early
TRANSPORT = re.compile(
    r"error sending request|unexpected EOF|: EOF\b|connection reset by peer|"
    r"connection refused|broken pipe|i/o timeout|TLS handshake timeout|"
    r"no such host|network issue connecting to the server|\(stream cut\)",
    re.IGNORECASE,
)
# A rate limit as agy reports it: its own `Error 429`, or a 5xx whose message is
# a rate limit or an exhausted quota
RATE_LIMIT = re.compile(
    r"Error 429\b|Error 5\d\d\b[\s\S]*?(?:RateLimitError|RESOURCE_EXHAUSTED|"
    r"Resource exhausted|resource_exhausted|Too Many Requests|Quota exceeded|"
    r"throttled|\b429\b)",
    re.IGNORECASE,
)
# agy's settings.json: the Gemini API route, no prompts, no telemetry
SETTINGS = {
    "modelProvider": "gemini",
    "toolPermission": "always-proceed",
    "artifactReviewPolicy": "always-proceed",
    "allowNonWorkspaceAccess": True,
    "enableTelemetry": False,
    "showFeedbackSurvey": False,
    "showTips": False,
    "notifications": False,
}

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


def custom_model() -> tuple[str, dict[str, Any]] | None:
  """Returns the custom model slug and settings entry for non-catalog models.

  Configures a token budget of
  `max(CONTEXT_WINDOW - AGY_RESERVE, CONTEXT_WINDOW // 2 + AGY_RESERVE)` so
  `agy` triggers conversation compaction before overflowing the context window.

  Returns:
    A tuple of `(name, entry)`, or `None` for a catalog model or when
    `AGY_MODEL` is set explicitly.
  """
  name = os.environ["MODEL_NAME"]
  if os.environ.get("AGY_MODEL") or name.removesuffix("-preview") in CATALOG:
    return None
  budget = max(CONTEXT_WINDOW - AGY_RESERVE, CONTEXT_WINDOW // 2 + AGY_RESERVE)
  return name + CUSTOM_SUFFIX, {
      "modelName": f"gemini-api://models/{name}",
      "maxTokens": budget,
  }


def model_slug() -> str:
  """Returns the `agy` model identifier string.

  Uses `AGY_MODEL` if set; for a model outside agy's catalog, its name from
  `custom_model`; otherwise strips `-preview` from `MODEL_NAME` and appends
  `EFFORT` (e.g. `gemini-3.8-flash-high`).

  Returns:
    The model slug string for `agy --model`.
  """
  custom = custom_model()
  if custom:
    return custom[0]
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
    `usage`, `tools` (Counter), `texts` (list of non-empty agent responses),
    `answered` (whether the last completed step was a response with text), and
    `ended_on_error` (whether no response or tool step followed agy's last
    error step).
  """
  out: dict[str, Any] = {
      "conversation": None,
      "response": None,
      "status": None,
      "error": None,
      "usage": {},
      "tools": collections.Counter(),
      "texts": [],
      "answered": False,
      "ended_on_error": True,
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
    kind, state = step.get("step_type"), step.get("state")
    if event.get("event") == "init":
      out["conversation"] = event.get("conversation_id")
    elif kind == "agent_response":
      index = (
          step["step_index"] if isinstance(step.get("step_index"), int) else 0
      )
      texts[index] = texts.get(index, "") + str(step.get("text_delta") or "")
      out["ended_on_error"] = False
      if state == "DONE":
        out["answered"] = bool(texts[index].strip())
    elif kind:
      if kind in ("tool", "error_message"):
        out["ended_on_error"] = kind == "error_message"
      if kind == "tool" and state == "DONE":
        out["tools"][str(step.get("tool_name") or "?")] += 1
      # any later step means the preceding text was not the final answer
      if kind != "error_message" and state in ("ACTIVE", "DONE"):
        out["answered"] = False
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


def resume_wait(run: dict[str, Any], stalled: bool, n: int) -> float | None:
  """Returns the delay in seconds before resuming a run, or `None` to stop.

  Stalled runs and runs that lost their connection before producing an answer
  resume immediately (`0`). Runs that ended on a rate-limit error resume after
  `BACKOFF_SEC * (n + 1)` seconds. Because `agy` records the first error of a
  conversation in `result.error` even if an internal retry later succeeded,
  rate-limit recovery checks `ended_on_error` to confirm the run actually
  stopped on that error.

  Args:
    run: The run's parsed stream, from `read_stream`.
    stalled: Whether the run's `Watch` stopped it.
    n: The run's zero-based index.

  Returns:
    The seconds to wait before resuming, or `None` to end the task.
  """
  if stalled:
    return 0
  error = str(run["error"] or "") if run["status"] == "ERROR" else ""
  if TRANSPORT.search(error) and not run["answered"]:
    return 0
  if RATE_LIMIT.search(error) and run["ended_on_error"]:
    return BACKOFF_SEC * (n + 1)
  return None


class Watch:
  """Watchdog that stops `agy` once it produces no output for `STALL_SEC`.

  Only `agy` stdout/stderr output counts toward activity, since local file I/O
  may continue while a model call hangs.

  Attributes:
    stalled: Set when the watchdog stops the run, before `agy` is signalled.
  """

  def __init__(self, stream: pathlib.Path, pidfile: pathlib.Path):
    self.stream, self.pidfile = stream, pidfile
    self.stalled = False

  def silent_for(self, now: float) -> float:
    """Returns how long agy has printed nothing, in seconds.

    Args:
      now: The current time, in seconds since the epoch.

    Raises:
      OSError: If agy's output file does not exist yet.
    """
    return now - self.stream.stat().st_mtime

  async def run(self) -> None:
    """Checks every `WATCH_SEC` and stops agy's process group when stalled."""
    while True:
      await asyncio.sleep(WATCH_SEC)
      with contextlib.suppress(OSError, ValueError):
        if self.silent_for(time.time()) > STALL_SEC:
          self.stalled = True
          await stop(int(self.pidfile.read_text()))
          return


async def stop(pgid: int) -> None:
  """Sends a process group SIGTERM, then SIGKILL after `KILL_GRACE_SEC`.

  Args:
    pgid: The process group ID.
  """
  with contextlib.suppress(OSError):
    os.killpg(pgid, signal.SIGTERM)
    await asyncio.sleep(KILL_GRACE_SEC)
    os.killpg(pgid, signal.SIGKILL)


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
  settings = SETTINGS
  custom = custom_model()
  if custom:
    name, entry = custom
    settings = {
        **SETTINGS,
        "customModelsConfig": {"customModels": {name: entry}},
    }
  (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
  (home / ".gemini" / "antigravity-cli" / "settings.json").write_text(
      json.dumps(settings)
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
  """Runs `agy` on the task and returns the final message.

  A run that stalled, lost its connection before answering, or was rate limited
  is resumed in the same conversation (`resume_wait`).

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
        # the shell writes its pid and then becomes agy, so Watch can signal
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
        watch = Watch(stream, pidfile)
        watcher = asyncio.create_task(watch.run())
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
          watcher.cancel()
          runs.append(read_stream(stream))
          session.final_message = final_message(runs) or session.final_message
        conversation = runs[-1]["conversation"] or conversation
        wait = resume_wait(runs[-1], watch.stalled, n)
        left = session.time_left()
        # resume only with at least two minutes to work before run_process would
        # stop agy
        if (
            wait is None
            or not conversation
            or n + 1 == MAX_RUNS
            or (left is not None and left < wait + STOP_BEFORE_SEC + 120)
        ):
          break
        why = (
            f"was silent for {STALL_SEC // 60} minutes"
            if watch.stalled
            else f"ended on {str(runs[-1]['error'])[:200]!r}"
        )
        await session.log(f"agy {why}; resuming its conversation in {wait} s")
        await asyncio.sleep(wait)
        prompt = RESUME
  usage = collections.Counter()
  for run in runs:
    usage.update({k: v for k, v in run["usage"].items() if isinstance(v, int)})
  # agy's input_tokens excludes cache reads, which the protocol's input count
  # includes
  await session.usage(
      os.environ["MODEL_NAME"],
      usage["input_tokens"] + usage["cache_read_tokens"],
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
