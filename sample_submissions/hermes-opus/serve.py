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

HERE = pathlib.Path(__file__).resolve().parent
HERMES = "/opt/harness/venv/bin/hermes"
# hermes's own default turn cap: the task's time limit is what ends a run
MAX_TURNS = 500
# hermes is told to wrap up this long before the limit: at max effort one model
# call can take minutes
BUDGET_MARGIN_SEC = 600
STOP_BEFORE_SEC = 60
# Claude models come with an Anthropic endpoint (ANTHROPIC_BASE_URL) and use
# hermes's Anthropic provider. Any other model uses an OpenAI-compatible
# provider on MODEL_BASE_URL, defined in config.yaml under this name.
OPENAI_PROVIDER = "task-model"
OPENAI_KEY_ENV = "TASK_MODEL_API_KEY"
# thinking_patch.py makes hermes send Claude's earlier signed thinking back to
# the endpoint this variable names (otherwise hermes strips it for any endpoint
# but Anthropic's own)
NATIVE_THINKING_ENV = "HERMES_NATIVE_THINKING_BASE_URL"
# request timeout of hermes's side calls (the compressor's summary, image
# descriptions), and how long the summary may stream nothing: at max effort it
# can think for minutes before its first token
AUX_TIMEOUT_SEC = 1800
# Request timeout and stream-idle timeout for model calls, raised from hermes's
# defaults (1800 s and 90 to 180 s) so a max-effort reply that thinks silently
# for minutes is not cut off. They are set both in the provider config and in
# the environment, which some hermes code paths read instead.
REQUEST_TIMEOUT_SEC = 3600
STREAM_IDLE_SEC = 1800
TIMEOUTS = {
    "request_timeout_seconds": REQUEST_TIMEOUT_SEC,
    "stale_timeout_seconds": STREAM_IDLE_SEC,
}
TIMEOUT_ENV = {
    "HERMES_API_TIMEOUT": str(REQUEST_TIMEOUT_SEC),
    "HERMES_STREAM_STALE_TIMEOUT": str(STREAM_IDLE_SEC),
    "HERMES_API_CALL_STALE_TIMEOUT": str(STREAM_IDLE_SEC),
    "HERMES_STREAM_READ_TIMEOUT": str(STREAM_IDLE_SEC),
}
# Claude's context window, which the compressor's summary call is told (any
# other model's window hermes looks up itself)
CLAUDE_CONTEXT = 1_000_000
# turns a delegated subagent may take (hermes's default is 250)
DELEGATE_MAX_ITERATIONS = 50
# The model's highest reasoning level, as the evaluation server picks for a
# request that names none: max for Claude and GLM, high for the others (Gemini
# and gpt-oss refuse max). REASONING_EFFORT overrides it.
HIGHEST_EFFORT = (("claude", "max"), ("glm", "max"))
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
    "kanban",
    "connections",
]


def reasoning_effort() -> str:
  """Returns the reasoning level hermes asks the task's model for.

  Returns:
    `REASONING_EFFORT` if set, else the model's level in `HIGHEST_EFFORT`, else
    `high`.
  """
  model = os.environ.get("MODEL_NAME", "").lower()
  return os.environ.get("REASONING_EFFORT") or next(
      (level for name, level in HIGHEST_EFFORT if name in model), "high"
  )


def run_budget(left: float) -> int:
  """Computes the time budget in seconds passed to Hermes.

  Hermes emits a wrap-up notice at 80% of `--run-budget`. Dividing the desired
  wrap-up time by 0.8 triggers that notice `BUDGET_MARGIN_SEC` (or 40% of
  remaining time on short tasks) before the deadline, ahead of `run_process`
  stopping Hermes at `STOP_BEFORE_SEC`.

  Args:
    left: Remaining task time in seconds.

  Returns:
    The time budget in seconds (at least 1).
  """
  wrap_up = taskbridge.time_share(left, BUDGET_MARGIN_SEC, fraction=0.4)
  return max(1, int(wrap_up / 0.8))


def model_provider() -> tuple[str, dict[str, Any], dict[str, str]]:
  """Returns the Hermes provider, its config entries, and its environment.

  Returns:
    A tuple of `(provider_name, config_dict, env_dict)` for the task's model.
    `config_dict` names the model and the provider with its `TIMEOUTS`.
    `env_dict` holds the API key under `OPENAI_KEY_ENV` and, for Claude,
    the endpoint whose thinking is sent back under `NATIVE_THINKING_ENV`.
  """
  name = os.environ["MODEL_NAME"]
  # the key of the OpenAI-compatible endpoint, which `summary_route` uses too
  env = {OPENAI_KEY_ENV: os.environ.get("MODEL_API_KEY", "")}
  if os.environ.get("ANTHROPIC_BASE_URL"):
    env[NATIVE_THINKING_ENV] = os.environ["ANTHROPIC_BASE_URL"]
    config = {"model": name, "providers": {"anthropic": TIMEOUTS}}
    return "anthropic", config, env
  entry = {
      "base_url": os.environ["MODEL_BASE_URL"],
      "key_env": OPENAI_KEY_ENV,
      "api_mode": "chat_completions",
      **TIMEOUTS,
  }
  return (
      OPENAI_PROVIDER,
      {
          "model": {"default": name, "provider": OPENAI_PROVIDER},
          "providers": {OPENAI_PROVIDER: entry},
      },
      env,
  )


def summary_route() -> dict[str, Any]:
  """Builds the configuration for Hermes's context-compaction summary calls.

  Hermes summarizes older turns via a side call, but its Anthropic helper
  rejects non-Anthropic URLs (such as the evaluation gateway) and drops turns
  without summarizing. Routing compaction through the OpenAI-compatible
  endpoint (`MODEL_BASE_URL`) avoids this check. For Claude, `context_length` is
  also set explicitly so Hermes does not fall back to a smaller guess.

  Returns:
    The `auxiliary.compression` entry of `config.yaml`.
  """
  route = {
      "provider": "custom",
      "model": os.environ["MODEL_NAME"],
      "base_url": os.environ["MODEL_BASE_URL"],
      "key_env": OPENAI_KEY_ENV,
      "timeout": AUX_TIMEOUT_SEC,
      "no_progress_timeout": AUX_TIMEOUT_SEC,
  }
  if os.environ.get("ANTHROPIC_BASE_URL"):
    route["context_length"] = CLAUDE_CONTEXT
  return route


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
          "reasoning_effort": reasoning_effort(),
      },
      "memory": {"memory_enabled": False, "user_profile_enabled": False},
      "compression": {"enabled": True, "threshold": 0.85},
      "terminal": {"backend": "task", "cwd": session.workdir, "timeout": 180},
      "delegation": {"max_iterations": DELEGATE_MAX_ITERATIONS},
      "checkpoints": {"enabled": False},
      "auxiliary": {
          "title_generation": {"enabled": False},
          "compression": summary_route(),
          "vision": {"timeout": AUX_TIMEOUT_SEC},
      },
      "curator": {"enabled": False},
      "security": {"allow_lazy_installs": False},
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
      # written as JSON, which is valid YAML, so serve.py does not need PyYAML
      (home / "config.yaml").write_text(
          json.dumps(hermes_config(session, relay, model), indent=2)
      )
      # hermes would pick up OPENAI_* on its own; it gets the key only under the
      # names its config reads (ANTHROPIC_API_KEY, OPENAI_KEY_ENV in model_env)
      env = {
          k: v
          for k, v in relay.env().items()
          if not k.startswith(("OPENAI_", "MODEL_"))
      }
      env.update(
          model_env,
          **TIMEOUT_ENV,
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
