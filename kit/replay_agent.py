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

"""Harbor agent that replays a submission's dev-set artifacts into a container.

Used by `eval/run.py --artifacts` to grade artifacts without running a harness
or model, matching the evaluation server. For its trial's task, it reads the
folder written by `artifacts.materialize()`, deletes files listed in
`deleted.txt`, writes each deliverable file at its target path and mode as root,
and records the final message in the agent's log directory for graders.

If the container rejects a file write while still responsive (e.g., because a
parent path is a regular file in the task image), the agent notes the failure
and the verifier grades whatever was written. If Docker fails to complete a
command, the container stops responding, or the task has no materialized folder,
`ReplayError` is raised as an evaluator failure so the task is retried.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import tempfile
from typing import Any
import uuid

from harbor.agents.base import BaseAgent
from harbor.models.trajectories import Trajectory
from kit import artifacts
from kit.capture import exit_status
from kit.capture import write_text

STEP_SEC = 600  # time limit of one command in the task container
UPLOADS_AT_ONCE = 8  # as on the evaluation server
# Each command below reads a NUL-separated list of paths from a file, since
# thousands of paths as arguments would pass the kernel's limit. The server's
# replay runs the same commands.

# regular files only: a link or a folder the task's own image has at that path
# stays
REMOVE_FILES = (
    'xargs -0 -r sh -c \'for p; do if [ -f "$p" ] && [ ! -L "$p" ]; then rm -f'
    ' -- "$p"; fi; done\' sh'
)
# mkdir -p still makes every folder it can when another fails (a parent that is
# a file in the image)
MAKE_FOLDERS = "xargs -0 -r mkdir -p --"
# A link, or a folder that holds nothing but folders, at a file's own path is
# removed first, so docker cp neither writes through the link (into its target,
# maybe outside the deliverables) nor puts the file inside the folder. A folder
# that holds anything else, or that find cannot read through, stays, and its
# path is printed after BLOCKED.
CLEAR_TARGETS = (
    'xargs -0 -r sh -c \'for p; do if [ -L "$p" ]; then rm -f -- "$p"; elif ['
    ' -d "$p" ]; then f=$(find "$p" ! -type d -print -quit) && [ -z "$f" ] &&'
    ' rm -r -- "$p" 2>/dev/null || echo "BLOCKED:$p"; fi; done; exit 0\' sh'
)
BLOCKED = "BLOCKED:"


class ReplayError(RuntimeError):
  """Raised when the task container cannot be populated with artifacts.

  Indicates an evaluator infrastructure failure (such as a Docker failure or a
  full disk) rather than an artifact issue, causing the task to be retried.
  """


class ReplayAgent(BaseAgent):
  """Harbor agent that replays materialized task artifacts without a model."""

  SUPPORTS_ATIF = True

  def __init__(
      self,
      *args,
      artifacts: str = "",  # pylint: disable=redefined-outer-name
      tasks: dict[str, str] | None = None,
      harness: str = "replay",
      **kwargs,
  ):
    """Initializes ReplayAgent with materialized artifact paths and task map.

    Args:
      *args: Positional arguments forwarded to `BaseAgent`.
      artifacts: Root directory containing one materialized folder per dev task
        ID.
      tasks: Optional mapping from task folder names to task IDs where they
        differ.
      harness: Harness name recorded in logs and trajectories.
      **kwargs: Keyword arguments forwarded to `BaseAgent`.
    """
    super().__init__(*args, **kwargs)
    self.root, self.tasks, self.harness = (
        pathlib.Path(artifacts),
        dict(tasks or {}),
        harness,
    )
    self.notes: list[str] = []
    self.final_message = ""

  @staticmethod
  def name() -> str:
    """Returns the registered Harbor agent name."""
    return "plebench-replay"

  def version(self) -> str | None:
    """Returns the agent version string identifying the replayed harness."""
    return f"artifacts of {self.harness}"

  async def setup(self, environment) -> None:
    """Performs agent setup (no-op for replay).

    Args:
      environment: The Harbor task environment.
    """
    return None

  def _task_id(self, environment) -> str:
    """Resolves the dev task ID from the trial's `config.json`.

    Args:
      environment: The Harbor task environment.

    Returns:
      The resolved task ID string.

    Raises:
      ReplayError: If `config.json` cannot be read or does not name a task.
    """
    try:
      config = json.loads(
          (
              pathlib.Path(environment.trial_paths.trial_dir) / "config.json"
          ).read_text()
      )
      name = pathlib.Path(config["task"]["path"]).name
    except (OSError, ValueError, KeyError, TypeError) as exc:
      raise ReplayError(
          f"the trial's config.json names no task ({type(exc).__name__}: {exc})"
      ) from None
    return self.tasks.get(name, name)

  async def _exec(self, environment, command: str) -> str:
    """Runs `command` as root inside the task container and returns stdout.

    Running as root matches the evaluation server's replay so tasks whose agents
    run as non-root users still receive every file. Each command ends with
    `rm -f`, so any non-zero exit status means the command failed to finish.

    Args:
      environment: The Harbor task environment.
      command: Shell command to execute as root.

    Returns:
      The command's standard output string.

    Raises:
      ReplayError: If Docker fails to run or complete the command.
    """
    try:
      r = await environment.exec(
          command=command, cwd="/", timeout_sec=STEP_SEC, user="root"
      )
    # Docker's failure, not the artifacts'
    except Exception as exc:  # noqa: BLE001
      raise ReplayError(
          f"the task container did not run a command ({type(exc).__name__}:"
          f" {exc})"
      ) from exc
    if r.return_code:
      raise ReplayError(
          f"the task container did not finish a command (exit {r.return_code}: "
          f"{(r.stdout or '').strip()[-300:]})"
      )
    return r.stdout or ""

  async def _answers(self, environment) -> bool:
    """Returns whether the task container still responds to a trivial command.

    If the container responds, a rejected file write is attributed to the
    artifacts rather than a Docker failure.

    Args:
      environment: The Harbor task environment.

    Returns:
      True if `true` exits with 0 inside the container, False otherwise.
    """
    try:
      r = await environment.exec(command="true", cwd="/", timeout_sec=30)
    except Exception:  # pylint: disable=broad-exception-caught
      return False
    return r.return_code == 0

  async def run(self, instruction, environment, context) -> None:
    """Replays materialized task artifacts into `environment` and logs results.

    Args:
      instruction: The task instruction string.
      environment: The Harbor task environment.
      context: The Harbor execution context.

    Raises:
      ReplayError: If materialized artifacts are missing or container operations
        fail.
    """
    task = self._task_id(environment)
    folder = self.root / task
    if not (folder / artifacts.REPLAY).is_file():
      # only usable tasks are replayed: grading the untouched task would score a
      # baseline for missing artifacts
      raise ReplayError(f"no materialized artifacts for task {task}")
    data = artifacts.read_replay(folder)
    if data["deleted"]:
      rc, _ = await self._each(
          environment, "deleted.txt's list", data["deleted"], REMOVE_FILES
      )
      self.notes.append(
          f"deleted.txt: {len(data['deleted'])} path(s)"
          + ("" if rc == 0 else f", exit {rc}")
      )
    if data["files"]:
      await self._write(environment, folder, data["files"])
    if data.get("final_message"):
      self.final_message = (folder / data["final_message"]).read_text(
          encoding="utf-8", errors="replace"
      )
    self._record(instruction)

  async def _each(
      self, environment, what: str, paths: list[str], command: str
  ) -> tuple[int, str]:
    """Uploads `paths` as a NUL-separated list to `/tmp` and pipes to `command`.

    Args:
      environment: The Harbor task environment.
      what: Description of the path list for error reporting.
      paths: List of container paths to pass to `command`.
      command: Shell command that reads NUL-separated paths from stdin.

    Returns:
      A tuple `(exit_code, output)` from executing `command`.

    Raises:
      ReplayError: If uploading the path list fails or the command does not
        complete.
    """
    remote = f"/tmp/.plebench-replay-{uuid.uuid4().hex[:12]}.list"
    with tempfile.TemporaryDirectory(prefix="plebench-replay-") as tmp:
      local = pathlib.Path(tmp) / "paths"
      local.write_bytes(b"".join(p.encode("utf-8") + b"\0" for p in paths))
      try:
        await environment.upload_file(str(local), remote)
      # Docker's failure, or a full disk
      except Exception as exc:  # noqa: BLE001
        raise ReplayError(
            f"{what} could not be copied into the task container"
            f" ({type(exc).__name__}: {exc})"
        ) from exc
    out = await self._exec(
        environment, f'{command} < {remote} 2>&1; echo "rc=$?"; rm -f {remote}'
    )
    if (rc := exit_status(out)) is None:
      raise ReplayError(
          f"{what}: the task container did not finish a command"
          f" ({out.strip()[-300:]})"
      )
    return rc, out

  async def _write(
      self, environment, folder: pathlib.Path, files: list[dict[str, Any]]
  ) -> None:
    """Writes deliverable files into the container with their recorded modes.

    Creates parent directories first, clears symlinks or empty directory trees
    at target paths via `CLEAR_TARGETS`, and uploads up to `UPLOADS_AT_ONCE`
    files concurrently. Any target path occupied by a non-empty directory is
    skipped.

    Args:
      environment: The Harbor task environment.
      folder: Local directory holding materialized task artifacts.
      files: List of file descriptors with `src`, `path`, and `mode` entries.

    Raises:
      ReplayError: If a file upload fails and the container is no longer
        responsive.
    """
    parents = sorted({os.path.dirname(f["path"]) for f in files} - {"/", ""})
    if parents:
      rc, _ = await self._each(
          environment, "the folder list", parents, MAKE_FOLDERS
      )
      if rc:
        self.notes.append(f"some folders could not be made (exit {rc})")
    _, out = await self._each(
        environment,
        "the files' paths",
        [f["path"] for f in files],
        CLEAR_TARGETS,
    )
    blocked = {
        line[len(BLOCKED) :]
        for line in out.splitlines()
        if line.startswith(BLOCKED)
    }
    if blocked:
      self.notes.append(
          f"{len(blocked)} file(s) not written: a folder that holds something"
          " is at the path"
      )
    gate = asyncio.Semaphore(UPLOADS_AT_ONCE)

    async def upload(f: dict[str, Any]) -> str | None:
      src = folder / f["src"]
      # docker cp keeps the mode; the owner becomes root
      os.chmod(src, int(f["mode"], 8))
      async with gate:
        try:
          await environment.upload_file(str(src), f["path"])
        # most likely this file's path; _answers() catches Docker failing
        except Exception:  # pylint: disable=broad-exception-caught
          return f["path"]
      return None

    failed = sorted(blocked) + [
        p
        for p in await asyncio.gather(
            *(upload(f) for f in files if f["path"] not in blocked)
        )
        if p
    ]
    if failed and not await self._answers(environment):
      raise ReplayError(
          f"writing {failed[0]} failed and the task container does not answer"
      )
    self.notes.append(f"wrote {len(files) - len(failed)} file(s)")
    if failed:
      self.notes.append(
          f"not written: {', '.join(failed[:5])}"
          + (" ..." if len(failed) > 5 else "")
      )

  def _record(self, instruction: str) -> None:
    """Writes the console log and `trajectory.json` in `ServeAgent` format.

    Args:
      instruction: The task instruction string.
    """
    self.logs_dir.mkdir(parents=True, exist_ok=True)
    console = (
        "\n".join(self.notes)
        + "\nreplay of the submitted artifacts: no harness, no model\n"
    )
    if self.final_message:
      console += (
          json.dumps({"type": "result", "result": self.final_message}) + "\n"
      )
    files = {f"{self.harness}.txt": console}
    if self.final_message.strip():
      trajectory = Trajectory.model_validate({
          "schema_version": "ATIF-v1.7",
          "session_id": self.session_id or self.harness,
          "agent": {
              "name": self.harness,
              "version": self.version() or "",
              "model_name": "replay",
          },
          "steps": [
              {"step_id": 1, "source": "user", "message": instruction},
              {"step_id": 2, "source": "agent", "message": self.final_message},
          ],
          "notes": "replay of the submitted artifacts",
      })
      files["trajectory.json"] = json.dumps(trajectory.to_json_dict(), indent=2)
    for name, text in files.items():
      write_text(self.logs_dir / name, text)

  def populate_context_post_run(self, context) -> None:
    """Populates `context.metadata` with replay status and notes after the run.

    Args:
      context: The Harbor execution context to update.
    """
    context.metadata = {
        **(context.metadata or {}),
        "replay": True,
        "replay_notes": self.notes,
    }
