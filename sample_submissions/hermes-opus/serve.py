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

"""Hermes agent harness that runs one `hermes chat` process per task.

The `task_env` plugin runs Hermes terminal commands in the task container, and
Hermes file tools operate through that terminal. The task's MCP servers are
reached through the local relay.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

import taskbridge
import yaml

HERE = pathlib.Path(__file__).resolve().parent
HERMES = "/opt/harness/venv/bin/hermes"
MAX_TURNS = 300
# hermes wraps up this long before the limit: at max effort one model call can
# take minutes
BUDGET_MARGIN_SEC = 600
STOP_BEFORE_SEC = 60
# Claude models come with an Anthropic endpoint (ANTHROPIC_BASE_URL) and use
# hermes's Anthropic provider. Any other model uses an OpenAI-compatible
# provider on MODEL_BASE_URL, defined in config.yaml under this name.
OPENAI_PROVIDER = "task-model"
OPENAI_KEY_ENV = "TASK_MODEL_API_KEY"
# The model's highest reasoning level: max for Claude, high for the others
# (Gemini refuses max). REASONING_EFFORT overrides it.
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if os.environ.get("ANTHROPIC_BASE_URL") else "high"
)
# Toolsets switched off: they need the internet, a display or a person, keep
# state between runs, or are not needed for one task in a container.
DISABLED_TOOLSETS = [
    "web",
    "search",
    "x_search",
    "browser",
    "vision",
    "video",
    "image_gen",
    "video_gen",
    "tts",
    "memory",
    "session_search",
    "cronjob",
    "clarify",
    "code_execution",
    "computer_use",
    "homeassistant",
    "kanban",
    "connections",
]


def run_budget(left: float) -> int:
  """Computes the time budget in seconds passed to Hermes.

  Reserves `BUDGET_MARGIN_SEC` (or 40% of remaining time on a short task) so
  Hermes wraps up before `run_process` stops it at `STOP_BEFORE_SEC`.

  Args:
    left: Remaining task time in seconds.

  Returns:
    The time budget in seconds (at least 1).
  """
  return max(
      1, int(taskbridge.time_share(left, BUDGET_MARGIN_SEC, fraction=0.4))
  )


def model_provider() -> tuple[str, dict[str, Any], dict[str, str]]:
  """Returns the Hermes provider, config entries, and API key environment.

  Returns:
    A tuple of `(provider_name, config_dict, env_dict)` for the task's model.
  """
  name = os.environ["MODEL_NAME"]
  if os.environ.get("ANTHROPIC_BASE_URL"):
    return "anthropic", {"model": name}, {}
  entry = {
      "base_url": os.environ["MODEL_BASE_URL"],
      "key_env": OPENAI_KEY_ENV,
      "api_mode": "chat_completions",
  }
  return (
      OPENAI_PROVIDER,
      {
          "model": {"default": name, "provider": OPENAI_PROVIDER},
          "providers": {OPENAI_PROVIDER: entry},
      },
      {OPENAI_KEY_ENV: os.environ.get("MODEL_API_KEY", "")},
  )


def hermes_config(
    session: taskbridge.Session,
    relay: taskbridge.Relay,
    model: dict[str, Any],
) -> dict[str, Any]:
  """Builds the Hermes `config.yaml` dictionary for a task session.

  Args:
    session: The active task session.
    relay: The local HTTP relay for the task container.
    model: Model configuration entries from `model_provider`.

  Returns:
    The configuration dictionary to serialize as `config.yaml`.
  """
  return {
      **model,
      "toolsets": ["hermes-cli"],
      "agent": {
          "max_turns": MAX_TURNS,
          "disabled_toolsets": DISABLED_TOOLSETS,
          "reasoning_effort": EFFORT,
      },
      "memory": {"memory_enabled": False, "user_profile_enabled": False},
      "compression": {"enabled": True, "threshold": 0.85},
      "terminal": {"backend": "task", "cwd": session.workdir, "timeout": 180},
      "checkpoints": {"enabled": False},
      "auxiliary": {"title_generation": {"enabled": False}},
      "curator": {"enabled": False},
      "security": {"allow_lazy_installs": False, "tirith_enabled": False},
      "updates": {"check": False},
      "plugins": {"enabled": ["task_env"]},
      "mcp_servers": {
          s["name"]: {"url": relay.mcp_url(s["name"])}
          for s in session.mcp_servers
      },
  }


class Events:
  """Collects Hermes stream-json events and tracks the latest response text.

  The text emitted after the last tool call is stored as the session's final
  message so far, ensuring a cancelled task or an early exit still has an
  answer. The `result` event holds the final answer and token counts.
  """

  def __init__(self, session: taskbridge.Session):
    self.session, self.result, self.tail = session, {}, []

  async def on_line(self, line: bytes) -> None:
    """Processes one line of stream-json output from Hermes.

    Args:
      line: Raw line bytes read from the Hermes stdout stream.
    """
    try:
      event = json.loads(line)
    except ValueError:
      return
    kind = event.get("type") if isinstance(event, dict) else None
    if kind == "result":
      self.result = event
    elif kind in ("tool_use", "tool_result"):
      self.tail = []
    elif kind == "text" and isinstance(event.get("text"), str):
      self.tail.append(event["text"])
      self.session.final_message = (
          "".join(self.tail).strip() or self.session.final_message
      )

  @property
  def answer(self) -> str:
    """Returns the final answer text from the result event."""
    text = self.result.get("text")
    return text.strip() if isinstance(text, str) else ""


async def solve(session: taskbridge.Session) -> str:
  """Runs Hermes on the task session and returns the final answer.

  Args:
    session: The active task session.

  Returns:
    The final message produced by Hermes.
  """
  with tempfile.TemporaryDirectory(
      prefix="hermes-", ignore_cleanup_errors=True
  ) as scratch:
    home = pathlib.Path(scratch)
    shutil.copytree(HERE / "task_env", home / "plugins" / "task_env")
    # hermes saves long command output to a local file and points the model at
    # it, but the model's file tools read the task container. A file at that
    # directory's path turns this off.
    (home / "cache").mkdir()
    (home / "cache" / "terminal-output").touch()
    query = home / "task.md"
    query.write_text(session.instruction)
    provider, model, model_env = model_provider()
    async with taskbridge.Relay(session) as relay:
      (home / "config.yaml").write_text(
          yaml.safe_dump(hermes_config(session, relay, model), sort_keys=False)
      )
      # hermes would pick up OPENAI_* on its own; it gets the key only under its
      # provider's name (ANTHROPIC_API_KEY, or OPENAI_KEY_ENV from model_env)
      env = {
          k: v
          for k, v in relay.env().items()
          if not k.startswith(("OPENAI_", "MODEL_"))
      }
      env.update(
          model_env,
          HERMES_HOME=str(home),
          TERMINAL_ENV="task",
          TERMINAL_CWD=session.workdir,
          TASK_WORKDIR=session.workdir,
          TASKBRIDGE_DIR=str(HERE),
          NO_COLOR="1",
          TERM="dumb",
      )
      argv = [
          HERMES,
          "--yolo",
          "chat",
          "--query-file",
          str(query),
          "--format",
          "stream-json",
          "--model",
          os.environ["MODEL_NAME"],
          "--provider",
          provider,
          "--ignore-rules",
          "--source",
          "tool",
      ]
      left = session.time_left()
      if left is not None:
        argv += ["--run-budget", str(run_budget(left))]
      events = Events(session)
      done = await taskbridge.run_process(
          argv,
          session,
          env=env,
          cwd=str(home),
          on_line=events.on_line,
          stop_before_sec=STOP_BEFORE_SEC,
          log_prefix="hermes: ",
      )
  tokens = events.result.get("tokens")
  if isinstance(tokens, dict) and tokens:
    n = {k: v for k, v in tokens.items() if isinstance(v, int)}
    # input_tokens counts cached prompt tokens too
    read = n.get("cache_read", 0)
    await session.usage(
        os.environ["MODEL_NAME"],
        cache_read_tokens=read,
        output_tokens=n.get("output", 0),
        input_tokens=n.get("input", 0) + read + n.get("cache_write", 0),
    )
  return (
      events.answer
      or session.final_message
      or f"hermes exited with code {done.code} without a final answer."
  )


if __name__ == "__main__":
  taskbridge.run(solve)
