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

"""Hermes terminal backend that executes commands in the task container.

Routes every command through the harness's local relay (`taskbridge.Relay` at
`$TASK_RELAY_URL`). Because Hermes file tools run through the terminal, they
also operate inside the task container.
"""

import base64
import json
import os
import pathlib
import shlex
import sys
from typing import Any
import urllib.request
import uuid

from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _ThreadedProcessHandle

# taskbridge.py is in the submission's folder, which serve.py passes as
# TASKBRIDGE_DIR (hermes loads this plugin from a copy in its home). It goes at
# the end of sys.path so it cannot shadow one of hermes's own modules.
sys.path.append(
    os.environ.get("TASKBRIDGE_DIR")
    or str(pathlib.Path(__file__).resolve().parents[1])
)
import taskbridge  # pylint: disable=g-import-not-at-top,g-bad-import-order

# Output up to this size comes back inline: it fits exec's 100 KB cut even when
# every byte is invalid UTF-8, which the cut counts as 3 bytes. Larger output is
# read back with read_file.
INLINE_BYTES = 33_000
# a longer command runs from a script file: Linux caps one argument at 128 KiB
ARG_BYTES = 100_000
LARGE = "__task_output_in_file__"
# the relay is local: skip any HTTP proxy
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(route: str, fields: dict[str, Any], timeout: float) -> dict[str, Any]:
  """Sends a JSON POST request to the local task relay.

  Args:
    route: Relay endpoint name (e.g., `exec`, `read_file`, `write_file`).
    fields: Request payload dictionary.
    timeout: Request timeout in seconds.

  Returns:
    The parsed JSON response dictionary from the relay.
  """
  request = urllib.request.Request(
      f"{os.environ['TASK_RELAY_URL']}/{route}",
      data=json.dumps(fields).encode(),
      method="POST",
      headers={"Content-Type": "application/json"},
  )
  with _OPENER.open(request, timeout=timeout) as response:
    return json.loads(response.read())


def _upload(path: str, data: bytes) -> None:
  """Writes `data` to `path` in the task container in protocol-sized chunks.

  Args:
    path: Destination path inside the task container.
    data: Raw bytes to write.

  Raises:
    OSError: If the relay reports an error writing a chunk.
  """
  for start in range(0, max(len(data), 1), taskbridge.WRITE_CHUNK):
    done = _post(
        "write_file",
        {
            "path": path,
            "append": start > 0,
            "content_b64": (
                base64.b64encode(
                    data[start : start + taskbridge.WRITE_CHUNK]
                ).decode()
            ),
        },
        120,
    )
    if not done.get("ok"):
      raise OSError(done.get("error"))


def _run(
    command: str, stdin: bytes | None, timeout: int, login: bool
) -> tuple[str, int]:
  """Executes `command` in the task container and returns output and exit code.

  Redirects command output to a temporary file inside the container so outputs
  exceeding `exec`'s 100 KB limit can be retrieved via `read_file` (up to
  10 MB). Passes `stdin` and oversized commands via temporary files as well.

  Args:
    command: Shell command string to execute with bash.
    stdin: Optional bytes to pipe to standard input.
    timeout: Command timeout in seconds.
    login: Whether to invoke bash as a login shell (`bash -l`).

  Returns:
    A tuple of `(output_text, exit_code)`.
  """
  tag = uuid.uuid4().hex
  out, inp, src = (
      f"/tmp/.task-out-{tag}",
      f"/tmp/.task-in-{tag}",
      f"/tmp/.task-cmd-{tag}",
  )
  shell = "bash -l" if login else "bash"
  try:
    call = f"{shell} -c {shlex.quote(command)}"
    if len(call.encode("utf-8", "surrogateescape")) > ARG_BYTES:
      _upload(src, command.encode("utf-8", "surrogateescape"))
      call = f"{shell} {src}"
    if stdin is not None:
      _upload(inp, stdin)
      call += f" < {inp}"
    # a redirect that could not be made (a full or read-only /tmp) leaves no
    # file: bash's own error is the output
    script = (
        f"{call} > {out} 2>&1; status=$?; rm -f {inp} {src}; if [ ! -f {out} ];"
        " then "
        f"echo '[the output could not be kept in {out}: is /tmp full or"
        " read-only?]'; "
        f'elif [ "$(stat -c %s {out} 2>/dev/null || echo 0)" -le {INLINE_BYTES}'
        f" ]; then cat {out}; "
        f"rm -f {out}; else echo {LARGE}; fi; exit $status"
    )
    result = _post(
        "exec", {"command": script, "timeout_sec": timeout}, timeout + 60
    )
    if not result.get("ok", True):
      return f"error: {result.get('error')}", 1
    output = result.get("output") or ""
    if output.strip() == LARGE or result.get("timed_out"):
      got = _post("read_file", {"path": out}, 120)
      if got.get("ok"):
        output = base64.b64decode(got["content_b64"]).decode("utf-8", "replace")
        if got.get("truncated"):
          output += (
              f"\n[output cut: {got.get('size')} bytes in all, the first 10 MB"
              " shown]"
          )
      # so the model never sees the marker as the output
      elif output.strip() == LARGE:
        output = (
            "[the output was too long to show inline and could not be read"
            f" back: {got.get('error')}]"
        )
      _post(
          "exec", {"command": f"rm -f {out} {inp} {src}", "timeout_sec": 30}, 90
      )
    return output, int(result.get("exit_code", 1))
  # a relay failure goes back to the model as the command's output
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return f"the task container could not be reached: {exc}", 1


class TaskEnvironment(BaseEnvironment):
  """Hermes terminal environment backed by the task container relay."""

  def __init__(self, cwd: str, timeout: int):
    super().__init__(cwd=cwd, timeout=timeout)
    self.init_session()

  def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
    """Runs a bash command in the task container on a background thread.

    Args:
      cmd_string: Command string to execute.
      login: Whether to run bash as a login shell.
      timeout: Maximum execution time in seconds.
      stdin_data: Optional string input for standard input.

    Returns:
      A `_ThreadedProcessHandle` wrapping the command execution.
    """
    stdin = (
        None
        if stdin_data is None
        else stdin_data.encode("utf-8", "surrogateescape")
    )
    limit = taskbridge.positive_int(timeout, 120, taskbridge.MAX_TIMEOUT_SEC)
    return _ThreadedProcessHandle(lambda: _run(cmd_string, stdin, limit, login))

  def cleanup(self):
    """Cleans up environment resources (no-op for the remote container)."""


class TaskEnvironmentProvider(TerminalEnvironmentProvider):
  """Registers the `task` terminal environment provider with Hermes."""

  is_remote = True
  is_container = True

  @property
  def name(self) -> str:
    """Returns the backend identifier (`task`)."""
    return "task"

  @property
  def env_description(self) -> str:
    """Returns a short description of the task container environment."""
    return "the task's Linux container (no internet access)"

  def is_available(self) -> bool:
    """Returns whether the task relay URL is configured in the environment."""
    return bool(os.environ.get("TASK_RELAY_URL"))

  def create_environment(
      self,
      *,
      cwd,
      timeout,
      task_id="default",
      image=None,
      container_config=None,
      **kwargs,
  ):
    """Creates a `TaskEnvironment` for the current task container.

    Args:
      cwd: Fallback working directory if `TASK_WORKDIR` is not set.
      timeout: Default command timeout in seconds.
      task_id: Unused task identifier from Hermes.
      image: Unused container image name.
      container_config: Unused container configuration.
      **kwargs: Additional unused keyword arguments.

    Returns:
      A configured `TaskEnvironment` instance.
    """
    del task_id, image, container_config, kwargs
    return TaskEnvironment(
        cwd=os.environ.get("TASK_WORKDIR") or cwd, timeout=timeout
    )


def register(ctx):
  """Registers `TaskEnvironmentProvider` with the Hermes plugin context.

  Args:
    ctx: The Hermes plugin registration context.
  """
  ctx.register_terminal_environment_provider(TaskEnvironmentProvider())
