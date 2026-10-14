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

"""Harbor agent that runs a submission's serve stack and relays tasks to it.

Referenced by `eval/run.py` in generated Harbor job files. Starts a submission's
server next to each task container (or connects to an already-running server)
and relays task execution over the PLEBench serve protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import pathlib
import threading
from typing import Any
import uuid

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.trajectories import Trajectory
from kit import capture
from kit import protocol as P
from kit import relay
from kit import stack


def task_agent_timeout(text: str) -> float | None:
  """Parses `[agent].timeout_sec` from `task.toml` content.

  Args:
    text: Raw TOML content of `task.toml`.

  Returns:
    The configured timeout in seconds as a float, or None if unset.
  """
  # tomllib needs Python 3.11: Harbor's Python has it, but the kit's tests also
  # import this module on 3.10
  import tomllib  # pylint: disable=g-import-not-at-top

  value = (tomllib.loads(text).get("agent") or {}).get("timeout_sec")
  return None if value is None else float(value)


def agent_timeout(
    config: dict[str, Any], task_timeout_sec: float | None
) -> float | None:
  """Computes the agent phase time limit matching Harbor 0.20.0's `Trial`.

  Since Harbor does not pass its computed timeout to the agent, this replicates
  `_compute_agent_timeout_sec`: taking the job's `override_timeout_sec` (or
  `task_timeout_sec`), capping at `max_timeout_sec`, and multiplying by
  `agent_timeout_multiplier` (or `timeout_multiplier`).

  Args:
    config: Parsed `config.json` dictionary for the trial.
    task_timeout_sec: Fallback `[agent].timeout_sec` from the task's
      `task.toml`.

  Returns:
    The effective agent timeout in seconds, or None if no timeout applies.
  """
  agent = config.get("agent") or {}
  base = agent.get("override_timeout_sec") or task_timeout_sec
  if base is None:
    return None
  multiplier = config.get("agent_timeout_multiplier")
  if multiplier is None:
    multiplier = config.get("timeout_multiplier", 1.0)
  return min(float(base), float(agent.get("max_timeout_sec") or "inf")) * float(
      multiplier
  )


OutputOverrun = relay.OutputOverrun


async def _bounded_output(process, *, timeout_sec: int | None) -> ExecResult:
  """Reads up to `P.MAX_EXEC_READ_BYTES` from `process` stdout with a timeout.

  Replaces `DockerEnvironment._collect_buffered_output` for commands without
  stdin (Harbor merges stderr into stdout). If output exceeds
  `P.MAX_EXEC_READ_BYTES`, kills the Docker client process and raises
  `OutputOverrun`.

  Args:
    process: Async subprocess running `docker compose exec`.
    timeout_sec: Optional command timeout in seconds.

  Returns:
    An `ExecResult` containing decoded stdout and the process return code.

  Raises:
    OutputOverrun: If the command output exceeds `P.MAX_EXEC_READ_BYTES`.
    RuntimeError: If the command times out after `timeout_sec` seconds.
  """

  async def read() -> bytes:
    data, over = bytearray(), False
    while chunk := await process.stdout.read(1 << 16):
      if over:
        continue  # drained to the end, so the pipe closes and the wait returns
      data += chunk
      if len(data) > P.MAX_EXEC_READ_BYTES:
        over = True
        with contextlib.suppress(ProcessLookupError):
          process.kill()
    await process.wait()
    if over:
      raise OutputOverrun
    return bytes(data)

  try:
    data = await asyncio.wait_for(read(), timeout_sec or None)
  except BaseException as exc:
    with contextlib.suppress(ProcessLookupError):
      process.kill()
    await process.communicate()  # reads what is left, then reaps
    if isinstance(exc, asyncio.TimeoutError):
      raise RuntimeError(
          f"Command timed out after {timeout_sec} seconds"
      ) from None
    raise
  return ExecResult(
      stdout=data.decode(errors="replace") or None,
      stderr=None,
      return_code=process.returncode or 0,
  )


def bound_reads(environment) -> None:
  """Wraps `environment._collect_buffered_output` to bound stdout reads.

  Commands without stdin (including verifier commands) are routed through
  `_bounded_output`, while commands with `stdin_data` use Harbor's original
  collector. Environments without `_collect_buffered_output` are left unchanged.

  Args:
    environment: The Harbor task environment to patch.
  """
  harbor_collect = getattr(type(environment), "_collect_buffered_output", None)
  if harbor_collect is None or "_collect_buffered_output" in vars(environment):
    return

  async def collect(process, *, timeout_sec, stdin_data=None):
    if stdin_data is not None:
      return await harbor_collect(
          process, timeout_sec=timeout_sec, stdin_data=stdin_data
      )
    return await _bounded_output(process, timeout_sec=timeout_sec)

  # pylint: disable=protected-access
  environment._collect_buffered_output = collect
  # pylint: enable=protected-access


class HarborExecutor:
  """Adapts a Harbor task environment to the `relay.Relay` executor interface.

  The relay runs each command under `timeout` inside the container (exiting 124
  on timeout). Exceptions raised by Harbor or `docker compose` are forwarded to
  the relay (`Relay._failure`) to distinguish harness errors from evaluator
  infrastructure failures. Each command is tagged with `relay.EXEC_TAG` so
  interrupted commands are also terminated inside the container via
  `relay.stop_command`.
  """

  STOP_SEC = 20

  def __init__(self, environment):
    """Initializes HarborExecutor with a Harbor task environment.

    Args:
      environment: The Harbor task environment.
    """
    self.environment = environment

  async def exec(
      self, command: str, *, cwd: str, timeout_sec: int, user: str | None = None
  ) -> tuple[int, str]:
    """Executes `command` in the task container and returns `(rc, output)`.

    Args:
      command: Shell command to run.
      cwd: Working directory inside the container.
      timeout_sec: Maximum execution time in seconds.
      user: Optional user account under which to run the command.

    Returns:
      A tuple `(return_code, stdout_text)`.

    Raises:
      relay.ExecutorError: If `docker compose exec` fails and the container is
        unreachable.
    """
    who = {"user": user} if user else {}
    tag = uuid.uuid4().hex
    try:
      result = await self.environment.exec(
          command=command,
          cwd=cwd,
          timeout_sec=timeout_sec,
          env={relay.EXEC_TAG: tag},
          **who,
      )
    # cancelled (time limit, session done) or Harbor failed: stop it in the
    # container too
    except BaseException:
      await asyncio.shield(self._stop(tag, who))
      raise
    out = result.stdout or ""
    # Harbor merges stdout and stderr, so text that looks like docker's may be
    # the command's own. As on the server, it counts as docker's only when the
    # container then fails the probe too.
    failure = relay.docker_failure(result.return_code, out)
    if failure and not await self.answers():
      # the probe's answer goes with the error, so that the relay does not probe
      # again (Relay._failure)
      raise relay.ExecutorError(
          f"docker compose exec: {failure}", reachable=False
      )
    return result.return_code, out

  async def answers(self) -> bool:
    """Returns whether the task container responds to a trivial probe command.

    Returns:
      True if `true` exits with status 0 inside the container, False otherwise.
    """
    try:
      probe = await self.environment.exec(
          command="true", cwd="/", timeout_sec=relay.PROBE_TIMEOUT_SEC
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return False
    return probe.return_code == 0

  async def stop_all(self) -> None:
    """Stops all background processes left running by relay commands."""
    await self._stop("", {})

  async def _stop(self, tag: str, who: dict[str, Any]) -> None:
    """Stops processes started by the command tagged `tag` inside the container.

    When `tag` is empty, stops processes from all relay exec calls. Runs best-
    effort within `STOP_SEC` as the command's user and never raises.

    Args:
      tag: Value of `relay.EXEC_TAG` identifying the command, or `""` for all.
      who: Optional `{"user": str}` keyword arguments for `environment.exec`.
    """
    with contextlib.suppress(Exception):
      await asyncio.wait_for(
          self.environment.exec(
              command=relay.stop_command(tag),
              cwd="/",
              timeout_sec=self.STOP_SEC - 5,
              **who,
          ),
          self.STOP_SEC,
      )

  async def upload(self, local: str, remote: str) -> None:
    """Uploads a local file at `local` to `remote` inside the task container.

    Args:
      local: Local filesystem path of the source file.
      remote: Destination path inside the task container.
    """
    await self.environment.upload_file(local, remote)

  async def healthy(self) -> bool:
    """Returns whether the task's `main` container is still running or OOMed.

    Returns:
      True if the task's main container is alive according to
      `stack.container_alive`, or False if it stopped or Docker is unreachable.
    """
    # pylint: disable=g-import-not-at-top
    from harbor.environments.docker.docker import (
        _sanitize_docker_compose_project_name as project_of,
    )
    # pylint: enable=g-import-not-at-top

    project = project_of(getattr(self.environment, "session_id", None) or "")

    def check() -> bool:
      try:
        r = stack.docker(
            "ps",
            "-a",
            "-q",
            "--filter",
            f"label={stack.COMPOSE_PROJECT}={project}",
            "--filter",
            "label=com.docker.compose.service=main",
            timeout=30,
            check=False,
        )
      except stack.InfraError:
        return False
      ids = r.stdout.split() if r.returncode == 0 else []
      return bool(ids) and stack.container_alive(ids[0])

    return await asyncio.to_thread(check)


class ServeAgent(BaseAgent):
  """Harbor agent that manages a submission's serve stack and relays a trial."""

  SUPPORTS_ATIF = True

  def __init__(
      self,
      *args,
      harness: str = "harness",
      image: str | None = None,
      serve: str = "",
      allow: list[str] | None = None,
      exceptions: list[str] | None = None,
      env_file: str | None = None,
      url: str | None = None,
      run: str | None = None,
      deliverables: dict[str, list[str]] | None = None,
      **kwargs,
  ):
    """Initializes ServeAgent with container stack or external server settings.

    Args:
      *args: Positional arguments forwarded to `BaseAgent`.
      harness: Name of the submission harness.
      image: Docker image for the serve container when starting a per-trial
        stack.
      serve: Command string used to start the server inside `image`.
      allow: Allowed outbound `host:port` endpoints for the egress proxy.
      exceptions: Private `host:port` endpoints allowed through the egress
        proxy.
      env_file: Path to the environment file passed to the serve container.
      url: WebSocket URL of an already-running server (instead of `image` and
        `serve`).
      run: Evaluation run ID used to label and sweep the trial's serve stack.
      deliverables: Mapping from task folder names to deliverable container
        paths to capture after the session.
      **kwargs: Keyword arguments forwarded to `BaseAgent`.

    Raises:
      ValueError: If neither `url` nor both `image` and `serve` are provided.
    """
    super().__init__(*args, **kwargs)
    if not url and not (image and serve):
      raise ValueError(
          "ServeAgent: give `image` and `serve`, or the `url` of a running"
          " server"
      )
    self.harness, self.image, self.serve, self.url, self.run_id = (
        harness,
        image,
        serve,
        url,
        run,
    )
    self.allow, self.exceptions, self.env_file = (
        list(allow or []),
        list(exceptions or []),
        env_file,
    )
    self.stack: stack.Stack | None = None
    # run()'s teardown (_teardown in a thread), once begun
    self.ending: asyncio.Future[Any] | None = None
    # populate's teardown of a stack run() never reached
    self.late: threading.Thread | None = None
    self.outcome = relay.Outcome()
    self.logs: dict[str, bytes] = {}
    self.deliverables = dict(deliverables or {})
    # this trial's deliverables, once setup found its task
    self.captured: list[str] | None = None
    self.baseline: dict[str, Any] | None = None
    self.capture_note = ""  # why there is no baseline

  @staticmethod
  def name() -> str:
    """Returns the registered Harbor agent name."""
    return "plebench-serve"

  def version(self) -> str | None:
    """Returns the agent version string combining harness and image or URL."""
    return f"{self.harness} ({self.image or self.url})"

  async def setup(self, environment) -> None:
    """Captures baseline deliverables and starts the trial's serve stack.

    Starting the serve stack during `setup` keeps container creation time out of
    the agent phase timeout (though the server's startup until it accepts the
    WebSocket connection is counted). Uses `stack.settled` so that if `setup` is
    cancelled, it waits for the background start thread and removes the stack
    before returning.

    Args:
      environment: The Harbor task environment.
    """
    await self._baseline(environment)
    if self.url:
      return
    st = self.stack = stack.Stack(run=self.run_id)
    try:
      await stack.settled(
          functools.partial(
              st.start,
              image=self.image,
              command=self.serve,
              env_file=self.env_file,
              allow=self.allow,
              exceptions=self.exceptions,
          ),
          st.stop,
      )
    except BaseException:
      # removed: by settled() when cancelled, else by start() itself
      self.stack = None
      raise

  async def _baseline(self, environment) -> None:
    """Records the pre-run deliverable listing for the trial's task.

    If listing fails, `self.baseline` remains None so every deliverable file is
    treated as modified, and the failure reason is recorded in
    `self.capture_note`.

    Args:
      environment: The Harbor task environment.
    """
    if not self.deliverables:
      return
    _, task_dir = self._trial(environment)
    paths = self.deliverables.get(task_dir.name) if task_dir else None
    if not paths:
      return
    self.captured = list(paths)
    try:
      self.baseline = await capture.listing(environment, paths)
    # the run goes on; its export says what is missing
    except Exception as exc:  # pylint: disable=broad-exception-caught
      note = (
          f"no baseline ({type(exc).__name__}: {exc}): every file counts as"
          " changed"
      )
      self.capture_note = note[:300]

  async def _capture(self, environment) -> None:
    """Stops leftover processes and copies modified deliverables from the task.

    Runs within `capture.CAPTURE_SEC`. Any cancellation arriving during capture
    is held until the copy finishes so the verifier never runs while files are
    still being copied out.

    Args:
      environment: The Harbor task environment.
    """
    dest = pathlib.Path(environment.trial_paths.trial_dir) / capture.FOLDER
    capture.start(dest)

    async def copy() -> None:
      await HarborExecutor(environment).stop_all()
      await capture.capture(
          environment,
          self.captured,
          self.baseline,
          dest,
          self.outcome.final_message,
          self.capture_note,
      )

    job = asyncio.ensure_future(asyncio.wait_for(copy(), capture.CAPTURE_SEC))
    try:
      await asyncio.shield(job)
    except (asyncio.CancelledError, KeyboardInterrupt):
      await stack._outlast(job)  # pylint: disable=protected-access
      raise
    # past CAPTURE_SEC: capture.json records it
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          f"serve: the deliverables were not copied out ({type(exc).__name__}:"
          f" {exc})"
      )

  def _trial(self, environment) -> tuple[dict[str, Any], pathlib.Path | None]:
    """Reads the trial's `config.json` and resolves its task directory.

    Falls back to the parent of `environment.environment_dir` when `config.json`
    is unavailable.

    Args:
      environment: The Harbor task environment.

    Returns:
      A tuple `(config_dict, task_dir_path)`.
    """
    config: dict[str, Any] = {}
    try:
      config = json.loads(
          (environment.trial_paths.trial_dir / "config.json").read_text()
      )
      return config, pathlib.Path(config["task"]["path"])
    # config.json missing, or not in the form expected
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          f"serve: the trial's config.json named no task ({type(exc).__name__}:"
          f" {exc}); the task is found beside its environment"
      )
      config = config if isinstance(config, dict) else {}
      env_dir = getattr(environment, "environment_dir", None)
      return config, pathlib.Path(env_dir).parent if env_dir else None

  def _agent_timeout(self, environment) -> float | None:
    """Computes the agent phase timeout from `config.json` and `task.toml`.

    Args:
      environment: The Harbor task environment.

    Returns:
      The agent timeout in seconds, or None if unavailable.
    """
    config, task_dir = self._trial(environment)
    try:
      task_limit = (
          task_agent_timeout((task_dir / "task.toml").read_text())
          if task_dir
          else None
      )
      limit = agent_timeout(config, task_limit)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          "serve: the agent time limit could not be read"
          f" ({type(exc).__name__}: {exc}); the task message carries none"
      )
      return None
    if limit is None:
      self.logger.warning(
          "serve: no agent time limit found; the task message carries none"
      )
    return limit

  async def run(self, instruction, environment, context) -> None:
    """Connects to the serve container, relays the task, and captures outputs.

    Args:
      instruction: The task instruction string.
      environment: The Harbor task environment.
      context: The Harbor execution context.

    Raises:
      NonZeroAgentExitCodeError: If the server connection fails or the session
        does not end in `"completed"`.
      stack.InfraError: If an evaluator infrastructure failure occurs during
        relaying.
    """
    bound_reads(environment)
    executor = HarborExecutor(environment)
    servers = [
        {"name": s.name, "url": s.url, "transport": s.transport}
        for s in self.mcp_servers
        if s.url
    ]
    st = self.stack
    rl = relay.Relay(
        executor,
        workdir="/",
        mcp_servers=relay.neutral_names(servers),
        agent_timeout_sec=self._agent_timeout(environment),
        stack=st,
    )
    rl.start_clock()
    try:
      rl.workdir = await self._workdir(environment)
      alive = (lambda: asyncio.to_thread(st.running)) if st else None
      try:
        ws = await rl.connect(
            self.url or st.url, alive=alive, start=st.started_at if st else None
        )
      except ConnectionError as exc:
        # docker inspect blocks, so it runs in a thread while the other trials
        # on Harbor's event loop go on
        exited = await asyncio.to_thread(stack.exit_note, st) if st else None
        rl.outcome.status, rl.outcome.reason = (
            "error",
            f"{exc}; {exited}" if exited else str(exc),
        )
        raise NonZeroAgentExitCodeError(f"serve: {rl.outcome.reason}") from None
      rl.outcome.note(f"Harbor session {self.session_id}")
      await rl.run(ws, instruction)
    finally:
      self.outcome = rl.outcome
      try:
        # before the verifier, which may change the files
        if self.captured is not None:
          await self._capture(environment)
      finally:
        # Read the logs and remove the stack before run() returns, even when a
        # cancel (Harbor's time limit) arrives meanwhile: the thread keeps
        # running, so wait it out, then re-raise the cancel.
        # populate_context_post_run then finds both done. _teardown never
        # raises, so the outcome stands.
        self.ending = asyncio.get_running_loop().run_in_executor(
            None, self._teardown
        )
        try:
          await asyncio.shield(self.ending)
        except (asyncio.CancelledError, KeyboardInterrupt):
          await stack._outlast(self.ending)  # pylint: disable=protected-access
          raise
    if rl.outcome.infra:
      raise stack.InfraError(
          f"the evaluator failed while relaying: {rl.outcome.infra}"
      )
    if rl.outcome.status != "completed":
      raise NonZeroAgentExitCodeError(f"serve: {rl.outcome.reason}")

  async def _workdir(self, environment) -> str:
    """Resolves the task container's initial working directory via `pwd`.

    Args:
      environment: The Harbor task environment.

    Returns:
      The absolute working directory path inside the task container.

    Raises:
      stack.InfraError: If `pwd` does not output an absolute path.
    """
    result = await environment.exec(command="pwd", timeout_sec=30)
    lines = [
        ln for ln in (result.stdout or "").splitlines() if ln.startswith("/")
    ]
    if not lines:
      raise stack.InfraError(
          "the task container's working directory could not be resolved (pwd"
          " gave no absolute path)"
      )
    return lines[-1]

  def _teardown(self) -> None:
    """Collects serve and egress container logs and stops the stack.

    Never raises so that teardown errors cannot overwrite the trial's outcome.
    """
    st, self.stack = self.stack, None
    if st is None:
      return
    try:
      self.logs = {
          "serve.log": st.logs("serve"),
          "egress.log": st.logs("egress"),
      }
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          f"the logs of serve stack {st.id} could not be read: {exc}"
      )
    try:
      st.stop()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(f"serve stack {st.id} could not be removed: {exc}")

  def _trajectory(self) -> str | None:
    """Builds the ATIF trajectory JSON string from `self.outcome`.

    Returns:
      Formatted JSON string for `trajectory.json`, or None if fewer than two
      steps were recorded.
    """
    o = self.outcome
    steps = []
    for s in o.steps:
      step = {
          "step_id": len(steps) + 1,
          "source": s["source"],
          "message": s.get("message", ""),
      }
      if s["source"] == "agent":
        step["tool_calls"] = [{
            "tool_call_id": s["id"],
            "function_name": s["tool"],
            "arguments": s["arguments"],
        }]
        step["observation"] = {
            "results": [
                {"source_call_id": s["id"], "content": s["observation"]}
            ]
        }
      steps.append(step)
    if o.final_message.strip():
      steps.append({
          "step_id": len(steps) + 1,
          "source": "agent",
          "message": o.final_message,
      })
    if len(steps) < 2:
      return None
    trajectory = Trajectory.model_validate({
        "schema_version": "ATIF-v1.7",
        "session_id": self.session_id or self.harness,
        "agent": {
            "name": self.harness,
            "version": self.version() or "",
            "model_name": ", ".join(o.models) or self.model_name,
        },
        "steps": steps,
        "notes": f"serve stop condition: {o.reason}",
        "final_metrics": {
            "total_prompt_tokens": o.usage["input_tokens"] or None,
            "total_completion_tokens": o.usage["output_tokens"] or None,
            "total_cached_tokens": o.usage["cache_read_tokens"] or None,
            "total_steps": len(steps),
        },
    })
    return json.dumps(trajectory.to_json_dict(), indent=2)

  def populate_context_post_run(self, context) -> None:
    """Populates `context` with token counts, metadata, and log files.

    Never raises so Harbor does not retry or mark a completed trial as failed.

    Args:
      context: The Harbor execution context to update.
    """
    try:
      self._populate(context)
    # whatever failed here, the trial's outcome stands
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          "serve: the run's record could not be completed"
          f" ({type(exc).__name__}: {exc})"
      )

  def _populate(self, context) -> None:
    """Writes token usage, metadata, console output, and logs for `context`.

    Args:
      context: The Harbor execution context to update.
    """
    if self.stack is not None and self.ending is None and self.late is None:
      # run() never reached its teardown, so the trial ended before its harness
      # ran. Harbor calls this on the event loop that carries every other
      # trial's relay, so a thread removes the stack and writes its logs once
      # read, without being waited for here (the process waits for it before
      # exiting). The run's sweep (stack.RUN_LABEL) removes whatever is left.
      self.late = threading.Thread(
          target=self._late_teardown, name="serve-teardown"
      )
      self.late.start()
    o = self.outcome
    if o.usage["input_tokens"] or o.usage["output_tokens"]:
      context.n_input_tokens = o.usage["input_tokens"]
      context.n_output_tokens = o.usage["output_tokens"]
      context.n_cache_tokens = o.usage["cache_read_tokens"]
    context.metadata = {
        **(context.metadata or {}),
        "serve_status": o.status,
        "serve_requests": o.requests,
        "serve_models": o.models,
    }
    console = (
        "\n".join(o.log)
        + f"\nserve status={o.status}\nserve stop condition: {o.reason}\n"
    )
    if self.late is not None:
      console += (
          "serve stack: removed in the background; serve.log and egress.log"
          " follow once read\n"
      )
    if o.final_message:
      console += (
          json.dumps({"type": "result", "result": o.final_message}) + "\n"
      )
    files = {
        f"{self.harness}.txt": lambda: console,
        "trajectory.json": self._trajectory,
    }
    self._write(
        {**files, **({} if self.late is not None else self._log_files())}
    )

  def _late_teardown(self) -> None:
    """Tears down a stack that `run()` never reached and writes its logs."""
    self._teardown()
    try:
      self._write(self._log_files())
    # the stack is removed already; only its logs are lost
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.logger.warning(
          "serve: the serve stack's logs could not be written"
          f" ({type(exc).__name__}: {exc})"
      )

  def _log_files(self) -> dict[str, Any]:
    """Returns a mapping of log filenames to lazy UTF-8 decoder callables."""
    return {
        name: lambda b=b: b.decode("utf-8", errors="replace")
        for name, b in self.logs.items()
    }

  def _write(self, files: dict[str, Any]) -> None:
    """Writes each file in `files` to `self.logs_dir` without following links.

    Args:
      files: Mapping of filename to a zero-arg callable returning text content
        (or None to skip writing the file).
    """
    self.logs_dir.mkdir(parents=True, exist_ok=True)
    for name, render in files.items():
      try:
        text = render()
        if text is not None:
          capture.write_text(self.logs_dir / name, text)
      # a file that cannot be written does not stop the others
      except Exception as exc:  # pylint: disable=broad-exception-caught
        self.logger.warning(f"{name} could not be written: {exc}")
