#!/usr/bin/env python3

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

"""Score a submission on the public dev set with the leaderboard's graders.

  uv run python eval/run.py <submission dir> [--tasks id,id]
      [--env-file local.env] [--out runs/x]
  uv run python eval/run.py --url ws://127.0.0.1:8765 [--name my-harness]
  uv run python eval/run.py --regrade runs/<name>-<time> [--env-file local.env]
  uv run python eval/run.py --artifacts <submission dir>

Tasks are downloaded from the latest commit of the Hugging Face dataset named in
`dev_set.yaml` (`tasks_repo`) into `~/.cache/plebench/tasks`, and each run
records that commit in `results.json` for `--regrade` (or uses `--tasks-dir`
folders directly). The kit builds your submission the same way the evaluation
server does (reusing cached builds when nothing changed), runs each task
alongside your harness via Harbor, and grades the result in an isolated verifier
container.

- If a `rubric` or `checks-judge` verifier fails, the task is re-graded up to
  3 times from the captured deliverables without re-running your harness.
- If an evaluator fault occurs (or the harness exits with an error before the
  model produces any output, unless it ended on a context overflow or a
  provider's limit), the task runs again, up to `--attempts` runs in all.
- Measured tasks' modified deliverable files are exported to
  `runs/<run>/artifacts/` and (unless `--no-export` is passed) to
  `<submission dir>/artifacts/`.

Harness API keys are read from the submission's `.env`. `MODEL_PROXY_URL/KEY`,
`JUDGE_BASE_URL/API_KEY/MODEL`, and `HF_TOKEN` are read from the environment or
`--env-file`. Requires Docker, `uv`, and Harbor with `websockets` (see
`eval/README.md`).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import copy
import dataclasses
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.parse
import uuid

try:
  import tomllib  # pylint: disable=g-import-not-at-top
# Python 3.10 fallback (from PyPI via pyproject.toml)
except ModuleNotFoundError:
  import tomli as tomllib  # pylint: disable=g-import-not-at-top

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from kit import artifacts  # pylint: disable=g-import-not-at-top
from kit import build  # pylint: disable=g-import-not-at-top
from kit import check  # pylint: disable=g-import-not-at-top
from kit import export  # pylint: disable=g-import-not-at-top
from kit import protocol  # pylint: disable=g-import-not-at-top
from kit import spec  # pylint: disable=g-import-not-at-top
from kit import stack  # pylint: disable=g-import-not-at-top
from kit.models import openai_base  # pylint: disable=g-import-not-at-top
import yaml  # pylint: disable=g-import-not-at-top

AGENT = "kit.harbor_agent:ServeAgent"
REPLAY_AGENT = "kit.replay_agent:ReplayAgent"
# Harbor Docker environment with an egress sidecar that tolerates judge reply
# headers arriving after 15 s
ENVIRONMENT = "kit.harbor_docker:KitDockerEnvironment"
# Pinned Harbor version required by kit/harbor_agent.py and container cleanup
HARBOR_VERSION = "0.20.0"
HARBOR_INSTALL = f"uv tool install harbor=={HARBOR_VERSION} --with websockets"
KIT_ARTIFACTS = "uv run python -m kit artifacts"
# Max verifier regrade passes before a judge task is marked unmeasured
REGRADE_PASSES = 3
# Grace period (seconds) for Harbor to cancel trials and remove containers on
# Ctrl-C
HARBOR_STOP_SEC = 120
# Interval (seconds) between checkpoints of finished trials to results.json and
# export.json
CHECKPOINT_SEC = 30
# Umask for Harbor so task and verifier containers can read replayed
# deliverables
CHILD_UMASK = 0o022
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
# Dev-set grading modes that call the LLM judge from the verifier container
JUDGE_GRADING = ("rubric", "checks-judge")
JUDGE_VARS = ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL")
# Model-name suffix marking judge calls routed through the kit's model proxy
# (kit/model_proxy/effort_default.py)
JUDGE_SUFFIX = "-judge"
# Subdirectory under ~/.cache/plebench storing downloaded task revisions
TASKS_CACHE = "tasks"
HF_REVISION_RE = re.compile(r"[0-9a-f]{40}")
# Matches TOML table headers `[name]` or `[[name]]` (group 1)
TABLE_HEADER = re.compile(r"\s*\[\[?([^\[\]]+)\]\]?\s*(#.*)?")
# Harness termination exceptions that count as a completed agent trial (matching
# the evaluation server). Any other Harbor exception leaves the trial
# unmeasured.
AGENT_OUTCOMES = frozenset({
    "AgentTimeoutError",
    "NonZeroAgentExitCodeError",
    "AgentSafetyRefusalError",
    "OutputTokenExceededError",
    "ContextWindowExceededError",
})
# Harbor 0.20 endpoint, transport, sandbox, and cancellation exceptions that
# indicate the trial was cut off externally and should be retried.
CUT_OFFS = frozenset({
    "AgentAuthenticationError",
    "AgentSetupTimeoutError",
    "ApiConnectionClosedError",
    "ApiConnectionError",
    "ApiError",
    "ApiInternalServerError",
    "ApiOverloadedError",
    "ApiProviderResourceNotFoundError",
    "ApiRateLimitError",
    "ApiResponseStalledError",
    "ApiUsageLimitError",
    "CancelledError",
    "EnvironmentStartTimeoutError",
    "HealthcheckError",
    "ModelNotFoundError",
    "NetworkConnectionError",
    "SandboxBuildFailedError",
    "UnknownApiError",
})
# Matches provider context-length errors (including Vertex and xAI phrasing) so
# an UnknownApiError is classified as ContextWindowExceededError rather than a
# cut-off.
CONTEXT_OVERFLOW = re.compile(
    r"longer than the model\\*'s context length|prompt is too long|"
    r"context_length_exceeded|maximum context length|"
    r"maximum prompt length is \S+ but the request contains",
    re.IGNORECASE,
)
# Matches mid-stream connection/server errors during automatic prompt
# compaction; despite containing "Prompt is too long", this is a transport
# cut-off and is retried.
COMPACTION_CUT = re.compile(
    r"automatic compaction failed: API Error: (?:Connection lost|Server error)"
    r" mid-response",
    re.IGNORECASE,
)
# Matches deterministic provider request limits (e.g., Grok's 350-tool cap) that
# would fail identically on retry, so the trial is graded as-is (never_started).
PROVIDER_LIMIT = re.compile(r"Maximum tools limit reached")
# Per-trial CPU and memory overhead for the serve container + egress proxy
SERVE_CPUS = int(stack.CPUS)
SERVE_BYTES = sum(
    int(size[:-1]) << {"m": 20, "g": 30}[size[-1]]
    for size in (stack.MEMORY, stack.EGRESS_MEMORY)
)
# Minimal host environment passed to Harbor so it can talk to Docker without
# exposing user API keys.
HARBOR_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_DEFAULT_PLATFORM",
    "DOCKER_API_VERSION",
    "XDG_RUNTIME_DIR",
    "UV_CACHE_DIR",
)
# Extra Docker Compose overlay applied to every task for pids/ulimit/label
# settings
TASK_LIMITS = "task-limits.yaml"


class HarborFailed(RuntimeError):  # pylint: disable=g-bad-exception-name
  """Raised when `harbor run` exits non-zero and leaves no trial result."""


@dataclasses.dataclass
class Result:
  """Evaluation outcome and metadata for a single benchmark task."""

  id: str
  # Normalized score (reward / reward_scale) in [0, 1]; 0 when not measured
  score: float = 0.0
  reward: float | None = None
  measured: bool = False
  status: str = "not run"  # Human-readable status shown in the summary table
  # Machine-readable status category used for branching
  kind: str = "not run"
  # True when an evaluator fault occurred and run_attempts should retry the task
  retry: bool = True
  seconds: float | None = None
  trial: str = ""
  # Token counts reported via usage messages (input, cache, output)
  tokens: dict[str, Any] | None = None
  # Regrade trial ID when graded by a verifier replay
  graded_by: str = ""
  # --artifacts: review status for this task's exported artifacts
  artifact: str = ""
  # --artifacts: prior local score recorded in manifest.json
  claimed: float | None = None


class Unreadable(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when a trial's JSON file cannot be read or parsed."""


def read_json(path: pathlib.Path) -> dict[str, Any]:
  """Reads and parses a trial's JSON file as a dictionary.

  Args:
    path: Path to the JSON file.

  Returns:
    The parsed JSON object.

  Raises:
    Unreadable: If the file cannot be read, fails to parse, or is not a JSON
      object.
  """
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError) as exc:
    raise Unreadable(
        f"{path.name} cannot be read ({type(exc).__name__})"
    ) from None
  if not isinstance(data, dict):
    raise Unreadable(f"{path.name} holds no JSON object")
  return data


def finite(value: Any) -> float | None:
  """Converts a value to a finite float if possible.

  Args:
    value: Value to convert (booleans are excluded).

  Returns:
    The value as a float if it is a finite int or float, or None otherwise.
  """
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(value)
  ):
    return None
  return float(value)


class TaskError(ValueError):
  """Raised when a task's `task.toml`, `reward_scale` or grading is invalid."""


def task_toml(task: pathlib.Path) -> dict[str, Any]:
  """Reads and parses `task.toml` from a task directory.

  Args:
    task: Path to the task directory.

  Returns:
    The parsed TOML content as a dictionary.
  """
  return tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))


def _task_data(task: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
  """Reads `task.toml` and its `[metadata]` table.

  Args:
    task: Path to the task directory.

  Returns:
    A tuple of `(task_toml_data, metadata_table)`, the latter empty when absent.

  Raises:
    TaskError: If `task.toml` cannot be read or parsed.
  """
  try:
    data = task_toml(task)
  # tomllib.TOMLDecodeError is a subclass of ValueError.
  except (OSError, ValueError) as exc:
    raise TaskError(f"{task.name}/task.toml: {exc}") from None
  meta = data.get("metadata")
  return data, meta if isinstance(meta, dict) else {}


def reward_scale(task: pathlib.Path) -> float:
  """Returns the reward scale for a perfect result from `task.toml`.

  Reads `reward_scale` from `[metadata]` or top-level in `task.toml`, defaulting
  to 1.0 if absent.

  Args:
    task: Path to the task directory.

  Returns:
    The positive reward scale factor.

  Raises:
    TaskError: If `task.toml` cannot be read or parsed, or if `reward_scale` is
      not a positive finite number.
  """
  data, meta = _task_data(task)
  value = meta.get("reward_scale", data.get("reward_scale", 1.0))
  scale = finite(value)
  if scale is None or scale <= 0:
    raise TaskError(
        f"{task.name}/task.toml: reward_scale {value!r} is not a positive"
        " number"
    )
  return scale


def asks_judge(task: pathlib.Path) -> bool:
  """Returns whether the task's verifier uses an LLM judge.

  Reads `[metadata] grading` from `task.toml` (`checks`, `rubric`, or
  `checks-judge`).

  Args:
    task: Path to the task directory.

  Returns:
    True if the task's grading is in `JUDGE_GRADING`, False for `checks`.

  Raises:
    TaskError: If `task.toml` is missing or invalid, or if its grading mode is
      not recognized.
  """
  grading = _task_data(task)[1].get("grading")
  if grading not in ("checks", *JUDGE_GRADING):
    raise TaskError(
        f"{task.name}/task.toml: [metadata] grading {grading!r} is not checks,"
        f" {' or '.join(JUDGE_GRADING)}"
    )
  return grading in JUDGE_GRADING


# ---------------------------------------------------------------- LLM judge
class JudgeError(ValueError):
  """Raised when judge configuration would send an API key to a wrong host."""


NO_JUDGE = (
    "graded by an LLM judge, so set JUDGE_BASE_URL and JUDGE_API_KEY (any"
    " OpenAI-compatible endpoint) or MODEL_PROXY_URL and MODEL_PROXY_KEY"
)


def judge_env(
    base: dict[str, Any], own: dict[str, Any] | None = None
) -> dict[str, Any]:
  """Resolves `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL` settings.

  Uses `JUDGE_BASE_URL` and `JUDGE_API_KEY` from `base`, falling back to
  `MODEL_PROXY_URL` with `MODEL_PROXY_KEY`, and finally to `own` (the
  submission's own model and key). An API key is only sent to the endpoint it
  was configured with. `JUDGE_MODEL` defaults to the submission's model when
  `own` is used, or `DEFAULT_JUDGE_MODEL` otherwise.

  Args:
    base: Environment variables dictionary for the current run.
    own: Optional fallback judge settings from the submission's own model.

  Returns:
    A dictionary containing the resolved `JUDGE_*` configuration.

  Raises:
    JudgeError: If `JUDGE_API_KEY` is set without `JUDGE_BASE_URL`.
  """
  judge = {k: base.get(k) or "" for k in JUDGE_VARS}
  if not judge["JUDGE_BASE_URL"]:
    if judge["JUDGE_API_KEY"]:
      raise JudgeError(
          "JUDGE_API_KEY is set but JUDGE_BASE_URL is not: set the endpoint"
          " that key belongs to, or unset JUDGE_API_KEY (it is not sent to any"
          " other endpoint)"
      )
    if base.get("MODEL_PROXY_URL"):
      judge["JUDGE_BASE_URL"] = openai_base(base["MODEL_PROXY_URL"])
      judge["JUDGE_API_KEY"] = base.get("MODEL_PROXY_KEY") or ""
    elif own:
      judge["JUDGE_BASE_URL"], judge["JUDGE_API_KEY"] = (
          own["JUDGE_BASE_URL"],
          own["JUDGE_API_KEY"],
      )
      judge["JUDGE_MODEL"] = judge["JUDGE_MODEL"] or own["JUDGE_MODEL"]
  judge["JUDGE_MODEL"] = judge["JUDGE_MODEL"] or DEFAULT_JUDGE_MODEL
  if base.get("MODEL_PROXY_URL") and same_server(
      judge["JUDGE_BASE_URL"], base["MODEL_PROXY_URL"]
  ):
    # Record which setting provided the judge URL for error reporting.
    judge["from"] = "MODEL_PROXY_URL"
  return judge


def same_server(url: str, other: str) -> bool:
  """Returns whether two model server URLs name the same OpenAI-compatible API.

  A trailing slash or `/v1`, a default port, and the spelling of this machine
  (`localhost`, `127.0.0.1`) do not count.

  Args:
    url: A model server URL.
    other: Another model server URL.

  Returns:
    True if both reach the same scheme, host, port and API path.
  """

  def key(u: str) -> tuple[str, str, int, str] | None:
    try:
      parts = urllib.parse.urlsplit(openai_base(u.strip()))
      port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
      return None
    host = parts.hostname or ""
    host = "localhost" if check.this_machine(host) else host
    return parts.scheme, host, port, parts.path

  return key(url) is not None and key(url) == key(other)


def judge_problem(judge: dict[str, Any]) -> str | None:
  """Checks whether the judge configuration has a valid endpoint URL.

  Args:
    judge: Resolved judge configuration dictionary.

  Returns:
    An error description if the endpoint is missing or invalid (naming the
    setting without repeating the URL), or None if valid.
  """
  url = judge.get("JUDGE_BASE_URL") or ""
  if not url:
    return NO_JUDGE
  return check.url_problem(url, judge.get("from") or "JUDGE_BASE_URL")


def bridge_gateway() -> str:
  """Returns the host gateway IP on Docker's default bridge network.

  Returns:
    The gateway IP address string.

  Raises:
    stack.InfraError: If Docker's bridge network reports no gateway address.
  """
  out = stack.docker(
      "network",
      "inspect",
      "bridge",
      "--format",
      "{{range .IPAM.Config}}{{.Gateway}} {{end}}",
      timeout=60,
  ).stdout.split()
  if not out:
    raise stack.InfraError("Docker's bridge network names no gateway address")
  return out[0]


def judge_in_containers(
    judge: dict[str, Any], darwin: bool | None = None
) -> tuple[dict[str, str], str]:
  """Rewrites judge settings for access from inside a verifier container.

  A judge running on the host machine (`localhost` or `127.0.0.1`) is rewritten
  to `host.docker.internal` on macOS or the Docker bridge gateway on Linux. A
  judge reached through the kit's model proxy (`MODEL_PROXY_URL`) is asked for
  `<model>-judge`, which the proxy serves with `<model>` at its default effort,
  as the evaluation server's judge runs, rather than at the effort the proxy
  applies to harness calls (kit/model_proxy/effort_default.py).

  Args:
    judge: Resolved judge configuration dictionary.
    darwin: Whether the host OS is macOS; auto-detected from `sys.platform` if
      None.

  Returns:
    A tuple of `(verifier_judge_env, allowed_host)` for the verifier container.
  """
  proxied = judge.get("from") == "MODEL_PROXY_URL"
  if proxied and not judge["JUDGE_MODEL"].endswith(JUDGE_SUFFIX):
    judge = {**judge, "JUDGE_MODEL": judge["JUDGE_MODEL"] + JUDGE_SUFFIX}
  parts = urllib.parse.urlsplit(judge["JUDGE_BASE_URL"].strip())
  host = parts.hostname or ""
  if check.this_machine(host):
    mac = sys.platform == "darwin" if darwin is None else darwin
    host = "host.docker.internal" if mac else bridge_gateway()
    netloc = (f"[{host}]" if ":" in host else host) + (
        f":{parts.port}" if parts.port else ""
    )
    judge = {
        **judge,
        "JUDGE_BASE_URL": urllib.parse.urlunsplit(
            parts._replace(netloc=netloc)
        ),
    }
  return {k: judge[k] for k in JUDGE_VARS}, host


# Seconds the judge's port gets to accept a connection (judge_reach_problem)
REACH_SEC = 5


def reachable(host: str, port: int) -> bool:
  """Returns whether a TCP connection to `host:port` is accepted.

  Args:
    host: Host name or IP address.
    port: TCP port.

  Returns:
    True if the connection opened within `REACH_SEC`.
  """
  try:
    with socket.create_connection((host, port), timeout=REACH_SEC):
      return True
  except OSError:
    return False


def judge_reach_problem(
    judge: dict[str, Any], darwin: bool | None = None
) -> str | None:
  """Checks that verifier containers can reach a judge on this machine.

  On Linux they reach it at Docker's bridge address (`judge_in_containers`),
  where a server listening only on 127.0.0.1 does not answer and grading would
  fail with a connection error inside the verifier. Docker Desktop forwards
  `host.docker.internal` to the host's loopback, so macOS is not checked, nor is
  a judge on another machine.

  Args:
    judge: Resolved judge configuration dictionary.
    darwin: Whether the host OS is macOS; auto-detected from `sys.platform` if
      None.

  Returns:
    What to do if the judge's port does not answer at the bridge address, or
    None.
  """
  url = (judge.get("JUDGE_BASE_URL") or "").strip()
  try:
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
  except ValueError:
    return None  # Invalid URL syntax is reported by judge_problem().
  mac = sys.platform == "darwin" if darwin is None else darwin
  if mac or not check.this_machine(parts.hostname or ""):
    return None
  gateway = bridge_gateway()
  if reachable(gateway, port):
    return None
  return (
      f"the judge at {check.redact_url(url)} is on this machine, but nothing"
      f" answers at {gateway}:{port}, Docker's bridge address, where the"
      " verifier containers reach it: a server listening on 127.0.0.1 alone"
      " cannot be reached from a container. Start it on 0.0.0.0 (as"
      " kit/model_proxy/start.sh does unless MODEL_PROXY_HOST says otherwise)"
      f" or on {gateway}, then run again"
  )


# ---------------------------------------------------------------- Dev tasks
class TasksError(Exception):
  """Raised when dev tasks cannot be obtained."""


class HubUnreachable(TasksError):  # pylint: disable=g-bad-exception-name
  """Raised when Hugging Face cannot be reached due to a network error."""


class RevisionGone(TasksError):  # pylint: disable=g-bad-exception-name
  """Raised when the requested dataset commit no longer exists on the hub."""


class ImageNotBuilt(TasksError):
  """Raised when a task's Docker image fails to build."""

  def __init__(self, task: str, reason: str):
    super().__init__(f"{task}: {reason}")
    self.reason = reason


def tree_sha256(folder: pathlib.Path) -> str:
  """Computes the SHA-256 digest of all regular files in a task folder.

  Hashes a sorted manifest of `<file_sha256>  <relative_posix_path>` lines,
  matching the digest recorded in `dev_set.yaml` and `release.json`. Symlinks
  are skipped; `has_link` rejects folders that contain any.

  Args:
    folder: Path to the task directory.

  Returns:
    The hexadecimal SHA-256 digest of the folder's file listing.
  """
  files = sorted(
      (p.relative_to(folder).as_posix(), p)
      for p in folder.rglob("*")
      if p.is_file() and not p.is_symlink()
  )
  listing = "".join(
      f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}\n"
      for rel, p in files
      if "__pycache__" not in rel.split("/")
  )
  return hashlib.sha256(listing.encode()).hexdigest()


def has_link(folder: pathlib.Path) -> bool:
  """Returns whether a task folder contains any symlink.

  The dataset contains no symlinks and `tree_sha256` skips them, so any folder
  containing a symlink is rejected regardless of its digest.

  Args:
    folder: Path to the task directory.

  Returns:
    True if any path under `folder` is a symlink.
  """
  return any(p.is_symlink() for p in folder.rglob("*"))


def dataset_repo(dev_file: pathlib.Path | None = None) -> str:
  """Returns the Hugging Face dataset repository ID from `dev_set.yaml`.

  Args:
    dev_file: Path to `dev_set.yaml`; defaults to `artifacts.DEV_SET_FILE`.

  Returns:
    The dataset's `tasks_repo` repository ID.

  Raises:
    TasksError: If `dev_file` does not specify `tasks_repo`.
  """
  dev_file = dev_file or artifacts.DEV_SET_FILE
  data = (
      yaml.safe_load(pathlib.Path(dev_file).read_text(encoding="utf-8")) or {}
  )
  repo = str(data.get("tasks_repo") or "")
  if not repo:
    raise TasksError(
        f"{dev_file} names no dataset (tasks_repo): give the task folders with"
        " --tasks-dir"
    )
  return repo


def _hub_failure(exc: Exception, where: str, token: str | None) -> TasksError:
  """Translates a `huggingface_hub` exception into an actionable `TasksError`.

  Args:
    exc: The exception raised by `huggingface_hub`.
    where: The dataset URL on Hugging Face.
    token: The Hugging Face token used, if any.

  Returns:
    A `TasksError` for access refusal or `HubUnreachable` for network errors.
  """
  name = type(exc).__name__
  if name in ("GatedRepoError", "RepositoryNotFoundError"):
    hint = (
        "set HF_TOKEN to a token that can read it"
        if not token
        else "HF_TOKEN cannot read it"
    )
    return TasksError(
        f"{where} refused the download ({name}): {hint} (while the dataset is"
        " gated, request access on that page, signed in to Hugging Face, and"
        " use a token of the account that was granted it)"
    )
  return HubUnreachable(
      f"the dev tasks could not be downloaded from {where} ({name}: {exc});"
      " check this machine's internet access and run again, or give"
      " --tasks-dir"
  )


def latest_revision(repo: str, token: str | None) -> str:
  """Returns the current commit SHA of the dataset's `main` branch.

  Each run uses the dataset's latest commit and records it in `results.json` for
  `--regrade`.

  Args:
    repo: Hugging Face dataset repository ID.
    token: Optional Hugging Face authentication token.

  Returns:
    The commit SHA of the dataset's `main` branch.

  Raises:
    TasksError: If `huggingface_hub` is missing, access is refused, or no valid
      commit SHA is returned.
    HubUnreachable: If Hugging Face cannot be reached.
  """
  try:
    # pylint: disable=g-import-not-at-top
    from huggingface_hub import HfApi
    # pylint: enable=g-import-not-at-top
  except ImportError:
    raise TasksError(
        "huggingface_hub is not installed: run eval/run.py with `uv run` from"
        " this repository, which installs it"
    ) from None
  where = f"https://huggingface.co/datasets/{repo}"
  try:
    info = HfApi().repo_info(
        repo, repo_type="dataset", revision="main", token=token or None
    )
  # huggingface_hub exceptions do not share a single base class.
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise _hub_failure(exc, where, token) from None
  sha = str(getattr(info, "sha", "") or "")
  if not HF_REVISION_RE.fullmatch(sha):
    raise TasksError(
        f"{where} answered no commit for its main branch ({sha!r})"
    )
  return sha


def _finished(cache: pathlib.Path, task: str) -> pathlib.Path:
  """Returns the completion marker path for `task` inside `cache`."""
  return cache / ".finished" / task


def _usable(cache: pathlib.Path, task: str) -> bool:
  """Returns whether `cache` has a complete, symlink-free download of `task`."""
  folder = cache / "tasks" / task
  return (
      _finished(cache, task).is_file()
      and (folder / "task.toml").is_file()
      and not has_link(folder)
  )


def _newest_download(ids: list[str]) -> pathlib.Path | None:
  """Returns the most recent cached revision directory containing all `ids`."""
  root = stack.cache_root() / TASKS_CACHE
  folders = [
      d
      for d in (root.iterdir() if root.is_dir() else ())
      if HF_REVISION_RE.fullmatch(d.name)
      and all(_usable(d, t) for t in ids)
  ]
  return max(folders, key=lambda d: d.stat().st_mtime, default=None)


def download_tasks(
    repo: str,
    revision: str,
    ids: list[str],
    dest: pathlib.Path,
    token: str | None,
) -> None:
  """Downloads task folders from a Hugging Face dataset repository.

  Args:
    repo: Hugging Face dataset repository ID.
    revision: Git commit SHA to download from.
    ids: Task IDs to download (`tasks/<id>/`).
    dest: Local destination directory.
    token: Optional Hugging Face authentication token.

  Raises:
    TasksError: If `huggingface_hub` is missing or the download fails.
    RevisionGone: If `revision` no longer exists in the dataset repository.
  """
  try:
    # pylint: disable=g-import-not-at-top
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RevisionNotFoundError
    # pylint: enable=g-import-not-at-top
  except ImportError:
    raise TasksError(
        "huggingface_hub is not installed: run eval/run.py with `uv run` from"
        " this repository, which installs it"
    ) from None
  where = f"https://huggingface.co/datasets/{repo}"
  try:
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        token=token or None,
        allow_patterns=[f"tasks/{t}/**" for t in ids],
    )
  except RevisionNotFoundError:
    raise RevisionGone(
        f"{where} has no revision {revision} any more (its history was"
        " rewritten since)"
    ) from None
  # huggingface_hub exceptions do not share a single base class.
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise _hub_failure(exc, where, token) from None


def resolve_tasks(
    ids: list[str],
    tasks_dir: pathlib.Path | None,
    values: dict[str, Any],
    pin: tuple[str, str] | None = None,
) -> dict[str, pathlib.Path]:
  """Locates or downloads the requested dev task directories.

  Uses `tasks_dir` directly if provided. Otherwise downloads the tasks from the
  dataset's latest commit (or `pin`) into the kit cache on first use. Task
  folders containing symlinks are rejected so containers cannot read arbitrary
  host files. If Hugging Face is unreachable, falls back to the most recent
  complete cached download.

  Args:
    ids: List of task IDs to resolve.
    tasks_dir: Optional local directory containing task subfolders.
    values: Environment variables dictionary (used for Hugging Face tokens).
    pin: Optional `(tasks_repo, tasks_revision)` to download from instead of
      the dataset's latest commit (used by `--regrade`).

  Returns:
    A mapping from task ID to its resolved local directory path.

  Raises:
    TasksError: If any task folder is missing, contains a symlink, or cannot be
      downloaded.
  """
  if tasks_dir is not None:
    found = {t: (tasks_dir / t).resolve() for t in ids}
    missing = [t for t, d in found.items() if not (d / "task.toml").is_file()]
    if missing:
      raise TasksError(
          f"--tasks-dir {tasks_dir} holds no task folder for"
          f" {', '.join(missing)}"
      )
    linked = [t for t in ids if has_link(found[t])]
    if linked:
      raise TasksError(
          f"{', '.join(linked)} under --tasks-dir {tasks_dir} holds a symbolic"
          " link, which the dataset's tasks never do, so it is not graded"
      )
    return found
  token = values.get("HF_TOKEN") or values.get("HUGGING_FACE_HUB_TOKEN")
  if pin is not None:
    repo, revision = pin
  else:
    repo = dataset_repo()
    try:
      revision = latest_revision(repo, token)
    except HubUnreachable:
      newest = _newest_download(ids)
      if newest is None:
        raise
      print(
          f"note: huggingface.co could not be reached, so the dev tasks are"
          f" those of the newest finished download (commit {newest.name[:12]})",
          flush=True,
      )
      return {t: newest / "tasks" / t for t in ids}
  cache = stack.cache_root() / TASKS_CACHE / revision
  found = {t: cache / "tasks" / t for t in ids}
  # Serialize downloads so concurrent runs do not delete in-progress folders.
  with build.locked(cache, "downloading the dev set"):
    stale = [t for t in ids if not _usable(cache, t)]
    if stale:
      print(
          f"downloading {len(stale)} dev task(s) from"
          f" huggingface.co/datasets/{repo} (commit {revision[:12]}) into"
          f" {cache} ...",
          flush=True,
      )
      for t in stale:
        _finished(cache, t).unlink(missing_ok=True)
        shutil.rmtree(found[t], ignore_errors=True)
      download_tasks(repo, revision, stale, cache, token)
      for t in stale:
        if not (found[t] / "task.toml").is_file():
          raise TasksError(
              f"{t}: the dataset at commit {revision[:12]} holds no task folder"
              " for it"
          )
        if has_link(found[t]):
          raise TasksError(
              f"{t}: the task folder downloaded at commit {revision[:12]} holds"
              " a symbolic link, which the dataset's tasks never do, so it is"
              " not graded"
          )
        _finished(cache, t).parent.mkdir(parents=True, exist_ok=True)
        _finished(cache, t).touch()
  return found


def tasks_record(
    found: dict[str, pathlib.Path], tasks_dir: pathlib.Path | None
) -> dict[str, Any]:
  """Records the task source metadata for `results.json` and `--regrade`.

  Args:
    found: Mapping from task ID to resolved task directory.
    tasks_dir: The `--tasks-dir` directory, or None when using the dataset.

  Returns:
    A dictionary recording `tasks_dir`, or `tasks_repo` and `tasks_revision`
    when all tasks come from a single cached dataset commit.
  """
  if tasks_dir is not None:
    return {"tasks_dir": str(tasks_dir)}
  record: dict[str, Any] = {"tasks_repo": dataset_repo()}
  root = stack.cache_root() / TASKS_CACHE
  revisions = set()
  for path in found.values():
    try:
      revisions.add(pathlib.Path(path).relative_to(root).parts[0])
    except (ValueError, IndexError):
      revisions.add("")
  if len(revisions) == 1 and HF_REVISION_RE.fullmatch(next(iter(revisions))):
    record["tasks_revision"] = revisions.pop()
  return record


def recorded_tasks(
    ids: list[str],
    saved: dict[str, Any],
    tasks_dir: pathlib.Path | None,
    values: dict[str, Any],
) -> dict[str, pathlib.Path]:
  """Resolves the task directories recorded in `saved` for `--regrade`.

  Uses `tasks_dir` (from the CLI or `saved["tasks_dir"]`) when present, or the
  dataset commit recorded in `saved`. Falls back to the dataset's latest commit
  with a note if `saved` records no commit or if that commit no longer exists on
  the hub.

  Args:
    ids: Task IDs to locate.
    saved: The run's `results.json` dictionary.
    tasks_dir: Optional `--tasks-dir` override for the regrade.
    values: Environment variables dictionary (used for Hugging Face tokens).

  Returns:
    A mapping from task ID to its resolved task directory.
  """
  folder = tasks_dir or (
      pathlib.Path(saved["tasks_dir"]) if saved.get("tasks_dir") else None
  )
  if folder is not None:
    return tasks_or_exit(ids, folder, values)
  repo = str(saved.get("tasks_repo") or "")
  revision = str(saved.get("tasks_revision") or "")
  if not repo or not HF_REVISION_RE.fullmatch(revision):
    print(
        "note: results.json does not record which task folders the run graded"
        " with (a run of an earlier kit), so they are graded with the dataset's"
        " latest commit (or --tasks-dir), which may differ",
        flush=True,
    )
    return tasks_or_exit(ids, tasks_dir, values)
  try:
    return resolve_tasks(ids, None, values, pin=(repo, revision))
  except RevisionGone:
    print(
        f"note: the dataset no longer has the commit the run graded with"
        f" ({revision[:12]}; its history was rewritten since), so its tasks are"
        " graded with the dataset's latest commit, which may differ",
        flush=True,
    )
    return tasks_or_exit(ids, None, values)
  except TasksError as exc:
    sys.exit(f"dev tasks: {exc}")


def allowlisted(text: str, host: str) -> str:
  """Rewrites `task.toml` content so verifier networks allow only `host`.

  Args:
    text: Original `task.toml` text.
    host: Hostname or IP address to allowlist in `[verifier]` and
      `[verifier.environment]`.

  Returns:
    The updated `task.toml` text.
  """
  out, table, done = [], "", set()
  tables = ("verifier", "verifier.environment")
  allow = [
      'network_mode = "allowlist"',
      f"allowed_hosts = [{json.dumps(host)}]",
  ]

  def close() -> None:
    if table in tables and table not in done:
      out.extend(allow)
      done.add(table)

  for line in text.splitlines():
    header = TABLE_HEADER.fullmatch(line)
    if header:
      close()
      table = header.group(1).strip()
      out.append(line)
      continue
    if table in tables and re.match(
        r"\s*(network_mode|allowed_hosts)\s*=", line
    ):
      continue
    out.append(line)
  close()
  for name in tables:
    if name not in done:
      out.extend(["", f"[{name}]", *allow])
  return "\n".join(out) + "\n"


def _brackets(line: str) -> int:
  """Returns the net count of unclosed `[` brackets on a TOML line.

  Args:
    line: One line of TOML text.

  Returns:
    The number of `[` characters minus `]` characters outside string literals
    and comments (positive when an array continues onto subsequent lines).
  """
  depth, quote, i = 0, "", 0
  while i < len(line):
    c = line[i]
    if quote:
      if c == "\\" and quote == '"':
        i += 1
      elif c == quote:
        quote = ""
    elif c in "\"'":
      quote = c
    elif c == "#":
      break
    elif c == "[":
      depth += 1
    elif c == "]":
      depth -= 1
    i += 1
  return depth


def without_collect(text: str) -> str:
  """Removes `[verifier]` collect hooks from `task.toml` content.

  Used during replay grading because the replayed deliverables already include
  files produced by collect hooks in the original harness container, and running
  the hooks again in a fresh container would overwrite them.

  Args:
    text: Original `task.toml` text.

  Returns:
    The `task.toml` text without `[verifier] collect` and
    `[[verifier.collect]]` tables.
  """
  out, table, open_value = [], "", 0
  for line in text.splitlines():
    # Skip continuation lines of a multi-line collect array.
    if open_value > 0:
      open_value += _brackets(line)
      continue
    header = TABLE_HEADER.fullmatch(line)
    if header:
      table = header.group(1).strip()
      # Drop [[verifier.collect]] tables entirely.
      if table == "verifier.collect":
        continue
    elif table == "verifier.collect":
      continue
    elif table == "verifier" and re.match(r"\s*collect\s*=", line):
      open_value = _brackets(line.split("=", 1)[1])
      continue
    out.append(line)
  return "\n".join(out) + "\n"


def agent_limited(text: str, seconds: int) -> str:
  """Sets `timeout_sec` in the `[agent]` table of `task.toml` content.

  Args:
    text: Original `task.toml` content.
    seconds: The agent timeout in seconds.

  Returns:
    The updated `task.toml` text with `[agent] timeout_sec` set to `seconds`.
  """
  return with_setting(text, "agent", "timeout_sec", seconds)


def with_setting(text: str, name: str, key: str, value: float) -> str:
  """Sets `key = value` in table `[name]` of `task.toml` content.

  Args:
    text: Original `task.toml` content.
    name: The TOML table name (e.g. `"agent"`).
    key: The key to set within `[name]`.
    value: The numeric value to write.

  Returns:
    The updated `task.toml` text, appending `[name]` if not already present.
  """
  out, table, done = [], "", False

  def close() -> None:
    nonlocal done
    if table == name and not done:
      out.append(f"{key} = {value}")
      done = True

  for line in text.splitlines():
    header = TABLE_HEADER.fullmatch(line)
    if header:
      close()
      table = header.group(1).strip()
    elif table == name and re.match(rf"\s*{re.escape(key)}\s*=", line):
      continue
    out.append(line)
  close()
  if not done:
    out.extend(["", f"[{name}]", f"{key} = {value}"])
  return "\n".join(out) + "\n"


# Minimum time (seconds) a task image gets to build locally: applies to
# task_image and to Harbor's environment and verifier builds (the staged copy's
# [environment] build_timeout_sec).
LOCAL_BUILD_SEC = 3600


def task_image(task: pathlib.Path) -> str:
  """Builds the task's Docker image once and returns its tag.

  Builds `task/environment/` tagged by its content hash and reuses the image
  while the directory is unchanged. Tasks with `verifier_image = "task"` in
  `task.toml` `[metadata]` build their verifier image from this base tag. A
  per-tag file lock serializes concurrent builds across processes without
  counting wait time against the build timeout.

  Args:
    task: Path to the task directory.

  Returns:
    The built Docker image tag.

  Raises:
    ImageNotBuilt: If the image build fails or times out.
  """
  env_dir = task / "environment"
  tag = f"plebench-task-{task.name}:{tree_sha256(env_dir)[:20]}"
  with build.locked(
      stack.cache_root() / "task-images" / tag.replace(":", "-"),
      f"building {task.name}'s image",
  ):
    return _built(task, env_dir, tag)


def _built(task: pathlib.Path, env_dir: pathlib.Path, tag: str) -> str:
  """Builds `tag` from `env_dir` if not already present in Docker.

  Args:
    task: Path to the task directory.
    env_dir: Path to the task's `environment/` directory.
    tag: Target Docker image tag.

  Returns:
    The built Docker image tag.

  Raises:
    ImageNotBuilt: If the image build fails or times out.
  """
  inspected = subprocess.run(
      ["docker", "image", "inspect", tag], capture_output=True, check=False
  )
  if inspected.returncode == 0:
    return tag
  # Allow at least LOCAL_BUILD_SEC for cold local builds that pull base layers.
  limit = max(
      float(
          (task_toml(task).get("environment") or {}).get("build_timeout_sec")
          or 0
      ),
      LOCAL_BUILD_SEC,
  )
  print(
      f"building {task.name}'s image, which its verifier starts from (up to"
      f" {limit / 60:.0f} min) ...",
      flush=True,
  )
  try:
    r = subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(env_dir)],
        capture_output=True,
        text=True,
        timeout=limit,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
        check=False,
    )
  except subprocess.TimeoutExpired:
    raise ImageNotBuilt(
        task.name, f"its image did not build within {limit:.0f} s"
    ) from None
  if r.returncode != 0:
    raise ImageNotBuilt(
        task.name,
        f"its image did not build: {(r.stderr or r.stdout)[-800:]}",
    )
  return tag


def fresh_copy(src: pathlib.Path, target: pathlib.Path) -> None:
  """Copies `src` to `target` with fresh modification times on all files.

  Excludes `__pycache__` directories. Updating mtimes ensures BuildKit does not
  reuse stale cached context files that happen to match in path, size, and
  mtime.

  Args:
    src: Source task directory.
    target: Destination directory (must not exist).
  """
  shutil.copytree(
      src,
      target,
      symlinks=True,
      ignore=shutil.ignore_patterns("__pycache__"),
      copy_function=shutil.copy,
  )


def replay_staged(
    task_dirs: dict[str, pathlib.Path], dest: pathlib.Path
) -> dict[str, pathlib.Path]:
  """Prepares staged task directories for replay grading.

  Tasks with `[verifier]` collect hooks are copied into `dest/<id>/` with their
  collect hooks removed (`without_collect`); other tasks reuse their existing
  staged directory.

  Args:
    task_dirs: Mapping from task ID to its staged task directory.
    dest: Directory for task copies with collect hooks stripped.

  Returns:
    A mapping from task ID to the task directory used for replay grading.

  Raises:
    TasksError: If `task.toml` cannot be parsed or its collect hooks cannot be
      removed.
  """
  out = {}
  for t, src in task_dirs.items():
    try:
      if not (task_toml(src).get("verifier") or {}).get("collect"):
        out[t] = src
        continue
      target = dest / t
      if target.exists():
        shutil.rmtree(target)
      fresh_copy(src, target)
      toml = target / "task.toml"
      toml.write_text(
          without_collect(toml.read_text(encoding="utf-8")), encoding="utf-8"
      )
      left = (task_toml(target).get("verifier") or {}).get("collect")
    except ValueError as exc:
      raise TasksError(
          f"{t}/task.toml cannot be read for the replay: {exc}"
      ) from None
    if left:
      raise TasksError(
          f"{t}/task.toml: its collect hooks could not be taken out for the"
          " replay"
      )
    out[t] = target
  return out


def stage_tasks(
    found: dict[str, pathlib.Path],
    dest: pathlib.Path,
    judge_host: str | None,
    unbuilt: dict[str, str] | None = None,
) -> dict[str, pathlib.Path]:
  """Copies tasks into the run's staging directory and configures judge access.

  Sets the competition's agent timeout (`spec.AGENT_TIMEOUT_SEC`) and a minimum
  build timeout (`LOCAL_BUILD_SEC`) in each staged `task.toml`. Tasks with
  `verifier_image = "task"` get a `tests/Dockerfile` built on top of
  `task_image`. Tasks graded by an LLM judge (`asks_judge`) have their verifier
  network allowlist updated to permit `judge_host`. No API keys are written to
  disk; the verifier receives `JUDGE_*` variables from Harbor's environment.

  Args:
    found: Mapping from task ID to source task directory.
    dest: Destination directory for staged task copies.
    judge_host: Hostname or IP of the LLM judge, or None if unused.
    unbuilt: Optional mapping populated with `(task_id, reason)` when a task
      image fails to build; if None, `ImageNotBuilt` is raised instead.

  Returns:
    A mapping from task ID to its staged task directory path.

  Raises:
    TasksError: If a staged `task.toml` is invalid or cannot be updated, or if
      a task image fails to build when `unbuilt` is None.
  """
  staged = {}
  for t, src in found.items():
    target = dest / t
    if target.exists():
      shutil.rmtree(target)
    fresh_copy(src, target)
    toml = target / "task.toml"
    toml.write_text(
        agent_limited(
            toml.read_text(encoding="utf-8"), spec.AGENT_TIMEOUT_SEC
        ),
        encoding="utf-8",
    )
    try:
      limit = (task_toml(target).get("agent") or {}).get("timeout_sec")
    except ValueError as exc:
      raise TasksError(
          f"{t}/task.toml cannot be read once the agent time limit is set:"
          f" {exc}"
      ) from None
    if limit != spec.AGENT_TIMEOUT_SEC:
      raise TasksError(f"{t}/task.toml: the agent time limit could not be set")
    # Ensure cold local builds have at least LOCAL_BUILD_SEC.
    own = (task_toml(target).get("environment") or {}).get("build_timeout_sec")
    try:
      build_sec = max(float(own or 0), LOCAL_BUILD_SEC)
    except (TypeError, ValueError):
      raise TasksError(
          f"{t}/task.toml: [environment] build_timeout_sec {own!r} is no"
          " number"
      ) from None
    if build_sec != own:
      toml.write_text(
          with_setting(
              toml.read_text(encoding="utf-8"),
              "environment",
              "build_timeout_sec",
              build_sec,
          ),
          encoding="utf-8",
      )
      try:
        built = task_toml(target).get("environment") or {}
      except ValueError as exc:
        raise TasksError(
            f"{t}/task.toml cannot be read once its build time is set: {exc}"
        ) from None
      if built.get("build_timeout_sec") != build_sec:
        raise TasksError(f"{t}/task.toml: the build time could not be set")
    meta = task_toml(target).get("metadata") or {}
    if meta.get("verifier_image") == "task":
      # Build the verifier image from the task's base image plus tests/.
      try:
        image = task_image(target)
      except ImageNotBuilt as exc:
        if unbuilt is None:
          raise
        # Record the build failure for this task so other tasks can still run.
        unbuilt[t] = exc.reason
        continue
      (target / "tests" / "Dockerfile").write_text(
          f"FROM {image}\nCOPY . /tests\n", encoding="utf-8"
      )
    try:
      by_judge = asks_judge(target)
    except TaskError as exc:
      raise TasksError(str(exc)) from None
    if by_judge and judge_host:
      toml.write_text(
          allowlisted(toml.read_text(encoding="utf-8"), judge_host),
          encoding="utf-8",
      )
      try:
        verifier = task_toml(target).get("verifier") or {}
        opened = [verifier, verifier.get("environment") or {}]
      except ValueError as exc:
        raise TasksError(
            f"{t}/task.toml cannot be read once its verifier may reach the"
            f" judge: {exc}"
        ) from None
      if any(v.get("allowed_hosts") != [judge_host] for v in opened):
        raise TasksError(
            f"{t}/task.toml: its verifier could not be given the judge's host"
        )
    staged[t] = target
  return staged


def select_tasks(arg: str | None, dev: dict[str, Any]) -> list[str]:
  """Parses and validates the `--tasks` selection against the dev set.

  Args:
    arg: Comma-separated task IDs from `--tasks`, or None for all dev tasks.
    dev: Dev set task metadata mapping.

  Returns:
    Deduplicated list of selected task IDs in the order given.
  """
  wanted = (
      list(dict.fromkeys(t.strip() for t in arg.split(",") if t.strip()))
      if arg
      else list(dev)
  )
  unknown = [t for t in wanted if t not in dev]
  if unknown or not wanted:
    sys.exit(
        f"not dev tasks: {', '.join(unknown) or '(none given)'} (the dev set:"
        f" {', '.join(dev)})"
    )
  return wanted


def harbor() -> str:
  """Locates the `harbor` CLI executable or exits if not installed.

  Returns:
    Path to the `harbor` executable.
  """
  found = os.environ.get("HARBOR") or shutil.which("harbor")
  if not found:
    sys.exit(f"Harbor is not installed: {HARBOR_INSTALL}")
  return found


def harbor_python(path: str) -> pathlib.Path | None:
  """Finds the Python interpreter that runs Harbor's launcher script.

  Resolves the interpreter from the launcher's `#!` line or looks for a sibling
  `python` binary in the same directory.

  Args:
    path: Path to the `harbor` executable.

  Returns:
    Path to the Python interpreter (even if broken, so `check_harbor` can
    advise reinstalling), or None if no interpreter is found.
  """
  launcher = pathlib.Path(path).resolve()
  try:
    with launcher.open("rb") as fh:
      line = fh.readline(4096).decode(errors="replace")
  except OSError:
    line = ""
  words = line[2:].split() if line.startswith("#!") else []
  if (
      len(words) > 1
      and pathlib.Path(words[0]).name == "env"
      and "python" in words[1]
  ):
    return pathlib.Path(shutil.which(words[1]) or words[1])
  if (
      words
      and os.path.isabs(words[0])
      and "python" in pathlib.Path(words[0]).name
  ):
    return pathlib.Path(words[0])
  python = launcher.with_name("python")
  return python if os.path.lexists(python) else None


def check_harbor(path: str) -> None:
  """Verifies Harbor's version and required Python modules in its environment.

  Exits if Harbor's version mismatches or its environment cannot import
  `websockets` and `kit.harbor_agent`. Prints a warning if Harbor's Python
  interpreter cannot be located.

  Args:
    path: Path to the `harbor` executable.
  """
  python = harbor_python(path)
  reinstall = f"reinstall Harbor with `{HARBOR_INSTALL} --reinstall`"
  if python is None:
    print(
        f"warning: there is no Python beside {path}, so Harbor's version and"
        f" its websockets are not checked; if trials fail, {reinstall}",
        flush=True,
    )
    return
  probe = (
      "import importlib.metadata as m, websockets, kit.harbor_agent;"
      " print(m.version('harbor'))"
  )
  try:
    r = subprocess.run(
        [str(python), "-c", probe],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
        check=False,
    )
    why = (
        (r.stderr.strip().splitlines() or ["no output"])[-1]
        if r.returncode != 0
        else ""
    )
  # Interpreter is missing (e.g. after a Python upgrade) or timed out.
  except (OSError, subprocess.TimeoutExpired) as exc:
    why = str(exc)
  if why:
    sys.exit(
        f"the kit's agent does not load in Harbor's environment ({why}):"
        f" {reinstall}"
    )
  version = (r.stdout.split() or ["(no version)"])[-1]
  if version != HARBOR_VERSION:
    sys.exit(
        f"Harbor {version} is installed and the kit needs {HARBOR_VERSION}:"
        f" {reinstall}"
    )


def _seconds(phase: dict[str, Any] | None) -> float | None:
  """Computes elapsed seconds between `started_at` and `finished_at` timestamps.

  Args:
    phase: Dictionary containing ISO-8601 `started_at` and `finished_at` keys.

  Returns:
    Rounded duration in seconds, or None if timestamps are missing or invalid.
  """
  try:
    return round(
        (
            datetime.datetime.fromisoformat(phase["finished_at"].rstrip("Z"))
            - datetime.datetime.fromisoformat(phase["started_at"].rstrip("Z"))
        ).total_seconds()
    )
  except (TypeError, KeyError, ValueError):
    return None


def _written_by_verifier(path: pathlib.Path, trial: pathlib.Path) -> bool:
  """Checks whether `path` is a regular file (not a symlink) inside `trial`.

  Prevents symlinks left by the task harness inside `verifier/` from being
  trusted as verifier output.

  Args:
    path: File path to inspect.
    trial: Root directory of the trial.

  Returns:
    True if `path` is a regular file resolving inside `trial`, False otherwise.
  """
  try:
    return stat.S_ISREG(
        os.lstat(path).st_mode
    ) and path.resolve().is_relative_to(trial.resolve())
  except OSError:
    return False


def _grader_said(path: pathlib.Path) -> str:
  """Returns the last non-empty line from `GRADER_FAILED.txt`, truncated.

  Args:
    path: Path to `GRADER_FAILED.txt`.

  Returns:
    Up to 160 characters of the last non-empty line, or an empty string.
  """
  try:
    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    ]
  except OSError:
    return ""
  said = next((line for line in reversed(lines) if line), "")
  return said[:160]


def _has_content(message: Any) -> bool:
  """Returns whether an ATIF message contains any non-whitespace text.

  Args:
    message: A message string or nested list/dict of content parts.

  Returns:
    True if the message contains any non-blank text.
  """
  if isinstance(message, str):
    return bool(message.strip())
  if isinstance(message, dict):
    return any(_has_content(v) for v in message.values())
  if isinstance(message, list):
    return any(_has_content(v) for v in message)
  return False


def _tokens(metrics: Any) -> bool:
  """Returns whether ATIF metrics report any positive token count.

  Args:
    metrics: A step's or the trajectory's metrics mapping.

  Returns:
    True if any key containing `"token"` has a positive numeric value.
  """
  return isinstance(metrics, dict) and any(
      isinstance(v, (int, float)) and v > 0
      for k, v in metrics.items()
      if "token" in k
  )


def never_started(res: dict[str, Any], trial: pathlib.Path) -> bool:
  """Returns whether the harness failed before its model produced any output.

  Follows the leaderboard's rule: the harness exited with an error and reported
  zero output tokens, or omitted output token counts and `trajectory.json` shows
  no model activity (a tool call, a positive token count, or at least two
  non-empty agent turns). Such a trial is retried unless it failed due to a
  context overflow or a deterministic provider limit (`PROVIDER_LIMIT`).

  Args:
    res: The trial's parsed `result.json`.
    trial: Path to the trial directory.

  Returns:
    True if the trial failed before starting model work.
  """
  out_tokens = (res.get("agent_result") or {}).get("n_output_tokens")
  info = res.get("exception_info") or {}
  said = str(info.get("exception_message") or "")[-200_000:]
  if PROVIDER_LIMIT.search(said) or CONTEXT_OVERFLOW.search(said):
    return False
  if trial_exception(res) != "NonZeroAgentExitCodeError" or (
      out_tokens is not None and out_tokens != 0
  ):
    return False
  if out_tokens == 0:
    return True
  traj = trial / "agent" / "trajectory.json"
  try:
    data = (
        json.loads(traj.read_text(encoding="utf-8"))
        if traj.is_file() and not traj.is_symlink()
        else {}
    )
  except (OSError, ValueError):
    data = {}
  if not isinstance(data, dict):
    return True
  steps = [s for s in data.get("steps") or [] if isinstance(s, dict)]
  agent = [s for s in steps if s.get("source") == "agent"]
  worked = (
      any(s.get("tool_calls") for s in agent)
      or any(_tokens(s.get("metrics")) for s in steps)
      or _tokens(data.get("final_metrics"))
      or sum(1 for s in agent if _has_content(s.get("message"))) >= 2
  )
  return not worked


def trial_exception(res: dict[str, Any]) -> str:
  """Returns the normalized exception type recorded in a trial's `result.json`.

  Reclassifies `UnknownApiError` matching `CONTEXT_OVERFLOW` (and not
  `COMPACTION_CUT`) as `ContextWindowExceededError`.

  Args:
    res: The trial's parsed `result.json`.

  Returns:
    The exception type string, or `""` if no exception is recorded.
  """
  info = res.get("exception_info") or {}
  exc = str(info.get("exception_type") or "")
  message = str(info.get("exception_message") or "")
  if (
      exc == "UnknownApiError"
      and CONTEXT_OVERFLOW.search(message)
      and not COMPACTION_CUT.search(message)
  ):
    return "ContextWindowExceededError"
  return exc


def outcome(
    task_id: str, task: pathlib.Path, trial: pathlib.Path | None
) -> Result:
  """Evaluates a single trial directory according to leaderboard grading rules.

  Applies grading checks in the leaderboard's order (see `eval/README.md`).
  Trials interrupted by external cut-offs (`CUT_OFFS`), failed `checks`
  verifiers, or harnesses that exited before model output (`never_started`) are
  marked for retry. Judge tasks whose verifier failed or timed out are marked
  `"grader failed"` so their captured deliverables can be re-graded without
  re-running the harness.

  Args:
    task_id: Identifier of the benchmark task.
    task: Path to the task definition directory.
    trial: Path to Harbor's trial output directory, or None if no trial ran.

  Returns:
    The computed `Result` for the trial.
  """
  if trial is None:
    return Result(task_id, status="no trial", kind="no trial")
  try:
    res = read_json(trial / "result.json")
  # Treat an unreadable result.json as a missing trial so the task is retried.
  except Unreadable as exc:
    return Result(
        task_id, status=f"no trial: {exc}", kind="no trial", trial=str(trial)
    )
  usage = res.get("agent_result") or {}
  tokens = {
      k: usage[f"n_{k}_tokens"]
      for k in ("input", "cache", "output")
      if usage.get(f"n_{k}_tokens")
  }
  r = Result(
      task_id,
      trial=str(trial),
      seconds=_seconds(res.get("agent_execution")),
      tokens=tokens or None,
  )
  exc = trial_exception(res)
  if exc in CUT_OFFS:
    r.status, r.kind = f"evaluator: {exc}", "evaluator"
    return r
  raw = finite(
      ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
  )
  failed = trial / "verifier" / "GRADER_FAILED.txt"
  grader_failed = _written_by_verifier(failed, trial)
  by_judge = False
  # Harbor records only the first exception, so a judge verifier timeout after
  # an agent outcome appears as that outcome with no verifier result.
  unverified = (
      exc in AGENT_OUTCOMES
      and raw is None
      and res.get("verifier_result") is None
  )
  if grader_failed or exc == "VerifierTimeoutError" or unverified:
    try:
      by_judge = asks_judge(task)
    except TaskError as err:
      r.status, r.kind, r.retry = f"task error: {err}", "task error", False
      return r
  if grader_failed:
    said = _grader_said(failed)
    r.status = "grader failed" + (f": {said}" if said else "")
    r.kind = "grader failed" if by_judge else "checks failed"
  elif exc == "VerifierTimeoutError" and by_judge:
    r.status, r.kind = (
        "grader failed: the verifier ran out of time (VerifierTimeoutError)",
        "grader failed",
    )
  elif exc and exc not in AGENT_OUTCOMES:
    r.status, r.kind = f"evaluator: {exc}", "evaluator"
  elif never_started(res, trial):
    r.status, r.kind = (
        "not started: your harness exited before its model produced any"
        " output",
        "not started",
    )
  elif unverified and by_judge:
    r.status, r.kind = (
        f"grader failed: the verifier left no result after {exc}",
        "grader failed",
    )
  elif raw is None:
    r.status, r.kind = "no reward", "no reward"
  else:
    try:
      scale = reward_scale(task)
    except TaskError as err:
      r.status, r.kind, r.retry = f"task error: {err}", "task error", False
      return r
    r.reward, r.measured, r.retry = raw, True, False
    r.score = min(max(raw / scale, 0.0), 1.0)
    r.status, r.kind = (f"harness: {exc}", "harness") if exc else ("ok", "ok")
  return r


def trials_by_task(
    job: pathlib.Path, quiet: bool = False
) -> dict[str, pathlib.Path]:
  """Finds completed trial directories in `job` keyed by task ID.

  Trials with unreadable `config.json` files or missing `result.json` files are
  skipped so their tasks can be retried.

  Args:
    job: Path to the Harbor job directory.
    quiet: Skip such trials without a note (for polling while Harbor is still
      writing them).

  Returns:
    A mapping from task ID to its trial directory path.
  """
  out = {}
  for p in sorted(job.glob("*/config.json")):
    try:
      out[pathlib.Path(read_json(p)["task"]["path"]).name] = p.parent
    except (Unreadable, KeyError, TypeError) as exc:
      if not quiet:
        print(
            f"note: {p.parent.name} is skipped:"
            f" {exc if isinstance(exc, Unreadable) else 'no task path'}",
            flush=True,
        )
  return {k: v for k, v in out.items() if (v / "result.json").is_file()}


@contextlib.contextmanager
def checkpoints(
    job: pathlib.Path,
    todo: list[str],
    task_dirs: dict[str, pathlib.Path],
    results: dict[str, Result],
    keep: Callable[[dict[str, Result]], None],
):
  """Periodically checkpoints completed trial results while a Harbor job runs.

  Polls `job` every `CHECKPOINT_SEC` seconds in a background thread and invokes
  `keep` whenever new `result.json` files appear for tasks in `todo`, preserving
  completed results if the process is terminated abruptly. Does not mutate
  `results` in place.

  Args:
    job: Harbor's job directory for the current attempt.
    todo: Task IDs in the current attempt.
    task_dirs: Mapping from task ID to staged task directory.
    results: Completed results from earlier attempts (not mutated).
    keep: Callback that persists a mapping from task ID to `Result`.

  Yields:
    None while the background checkpoint thread is active.
  """
  stop = threading.Event()

  def watch() -> None:
    seen = {}
    while not stop.wait(CHECKPOINT_SEC):
      try:
        found = trials_by_task(job, quiet=True)
        found = {t: p for t, p in found.items() if t in todo}
        stamp = {}
        for t, p in found.items():
          st = (p / "result.json").stat()
          stamp[t] = (p, st.st_mtime_ns, st.st_size)
        if stamp != seen:
          now = {t: outcome(t, task_dirs[t], p) for t, p in found.items()}
          keep({**results, **now})
          # Retry reading result.json on the next poll if it was only partially
          # written.
          seen = {t: s for t, s in stamp.items() if now[t].kind != "no trial"}
      # Ignore transient errors while Harbor is still writing trial files.
      except (OSError, ValueError):
        continue

  thread = threading.Thread(target=watch, daemon=True)
  thread.start()
  try:
    yield
  finally:
    stop.set()
    thread.join()


def own_judge(
    sub: spec.Submission, dotenv: dict[str, Any], base_url: str | None = None
) -> dict[str, str] | None:
  """Builds fallback `JUDGE_*` settings from the submission's own model and key.

  Args:
    sub: Loaded submission specification.
    dotenv: Environment variables loaded from the submission's `.env`.
    base_url: Optional `--model-base-url` override.

  Returns:
    A dictionary of `JUDGE_*` settings, or None for `google-cloud` models.

  Raises:
    check.SetupError: If `base_url` is not valid for replacing the model's
      endpoint.
  """
  problem = check.model_base_url_flag_problem(sub, base_url)
  if problem:
    raise check.SetupError("model", problem)
  if sub.model.provider == "google-cloud":
    return None
  return {
      "JUDGE_BASE_URL": base_url or sub.model.endpoint,
      "JUDGE_MODEL": sub.model.name,
      "JUDGE_API_KEY": dotenv.get(sub.model.key_secret) or "",
  }


def replay_folder(trial: pathlib.Path, dest: pathlib.Path) -> str:
  """Stages a trial's captured deliverables into `dest` for replay grading.

  Writes the captured files, `deleted.txt`, `final_message.txt`, and
  `replay.json` index into `dest`.

  Args:
    trial: Source trial directory containing the capture.
    dest: Destination directory for the replay agent.

  Returns:
    An empty string on success, or a reason string if the trial has no complete
    capture to replay.
  """
  content, why = export.task_folder(trial, dest)
  if content is None:
    return why
  files = [
      {
          "path": "/" + rel,
          "src": f"{artifacts.FILES}/{rel}",
          "mode": (
              "755"
              if os.stat(dest / artifacts.FILES / rel).st_mode & 0o100
              else "644"
          ),
      }
      for rel in content["files"]
  ]
  said = (dest / artifacts.FINAL_MESSAGE).read_bytes().strip()
  (dest / artifacts.REPLAY).write_text(
      json.dumps(
          {
              "task": dest.name,
              "files": files,
              "deleted": content["deleted"],
              "final_message": artifacts.FINAL_MESSAGE if said else None,
          },
          indent=2,
      )
      + "\n",
      encoding="utf-8",
  )
  return ""


def _ending(trial: pathlib.Path) -> str:
  """Determines how the harness ended in `trial` for status reporting.

  Args:
    trial: Path to the trial directory.

  Returns:
    `"ok"`, `"harness: <exception_type>"`, or `"evaluator: <exception_type>"`
    for a cut-off, whose deliverables measure nothing (`regrade_by_replay`
    replays none).
  """
  with contextlib.suppress(Unreadable):
    exc = trial_exception(read_json(trial / "result.json"))
    if exc in AGENT_OUTCOMES:
      return f"harness: {exc}"
    if exc in CUT_OFFS:
      return f"evaluator: {exc}"
  return "ok"


def regrade_by_replay(
    results: dict[str, Result],
    ids: list[str],
    task_dirs: dict[str, pathlib.Path],
    out: pathlib.Path,
    trials: int,
    judge: dict[str, Any] | None,
    harness: str,
    label: str,
) -> None:
  """Re-grades tasks by replaying captured deliverables in fresh containers.

  Runs up to `REGRADE_PASSES` passes using the replay agent without re-running
  the harness, each task in a copy without its collect hooks
  (`replay_staged`). Updates `results` in place: tasks successfully graded keep
  their original harness trial and record the replay trial in `graded_by`;
  tasks with missing captures, and cut-offs an earlier kit filed as gradings
  that failed, are marked for retry; tasks whose grader still fails remain
  unmeasured.

  Args:
    results: Mapping from task ID to current `Result`, updated in place.
    ids: Task IDs whose graders failed and should be re-graded.
    task_dirs: Mapping from task ID to staged task directory.
    out: Run output directory.
    trials: Maximum number of concurrent Harbor trials.
    judge: Judge environment variables for verifier containers, or None.
    harness: Name of the harness for replay metadata.
    label: Prefix label for temporary replay and regrade folders.
  """
  staged = out / f"{label}-replay"
  todo = []
  try:
    for t in ids:
      if not results[t].trial:
        continue
      ending = _ending(pathlib.Path(results[t].trial))
      # Reclassify cut-offs that an earlier kit recorded as failed gradings.
      if ending.startswith("evaluator: "):
        print(
            f"  {t}: its harness was cut off ({ending}), so its deliverables"
            " measure nothing; the task runs again",
            flush=True,
        )
        results[t].status, results[t].kind, results[t].retry = (
            ending,
            "evaluator",
            True,
        )
        continue
      why = replay_folder(pathlib.Path(results[t].trial), staged / t)
      if why:
        print(
            f"  {t}: its deliverables cannot be graded again ({why}); the task"
            " runs again",
            flush=True,
        )
        results[t].retry = True
      else:
        todo.append(t)
    try:
      # Strip collect hooks since captured deliverables already include their
      # output.
      replay_dirs = replay_staged(
          {t: task_dirs[t] for t in todo}, out / f"{label}-replay-tasks"
      )
    except TasksError as exc:
      print(
          f"  the deliverables cannot be graded again ({exc}); the tasks run"
          " again",
          flush=True,
      )
      for t in todo:
        results[t].retry = True
      return
    agent = {
        "import_path": REPLAY_AGENT,
        "model_name": "replay",
        "kwargs": {"artifacts": str(staged), "harness": harness},
    }
    passes, stopped = 0, ""
    for n in range(1, REGRADE_PASSES + 1):
      if not todo:
        return
      print(
          f"  grading {', '.join(todo)} again from the deliverables your"
          f" harness left (pass {n} of {REGRADE_PASSES}; your harness does not"
          " run)",
          flush=True,
      )
      try:
        found = trials_by_task(
            run_harbor(
                agent,
                [replay_dirs[t] for t in todo],
                out / f"{label}-regrade{n}",
                trials,
                None,
                judge,
            )
        )
      except HarborFailed as exc:
        print(f"  Harbor failed: {exc}", flush=True)
        stopped = "; Harbor failed before the rest"
        break
      passes = n
      for t in list(todo):
        r = outcome(t, replay_dirs[t], found.get(t))
        if r.measured:
          first = results[t]
          r.status = (
              f"{_ending(pathlib.Path(first.trial))}; graded again from its"
              " deliverables"
          )
          r.graded_by, r.trial, r.seconds, r.tokens = (
              r.trial,
              first.trial,
              first.seconds,
              first.tokens,
          )
          results[t] = r
          todo.remove(t)
        elif r.kind == "grader failed":
          results[t].status = r.status
    # Do not re-run the harness; `--regrade` can retry grading later.
    for t in todo:
      results[t].retry = False
      results[t].status += (
          f" ({passes} grading{'s' * (passes != 1)} of its"
          f" deliverables{stopped})"
      )
  finally:
    # Remove temporary replay staging; each trial directory retains what was
    # graded.
    shutil.rmtree(staged, ignore_errors=True)


def task_resources(task: pathlib.Path) -> tuple[int, int]:
  """Reads CPU count and memory (MiB) for a task container from `task.toml`.

  Args:
    task: Path to the task directory.

  Returns:
    A tuple of `(cpus, memory_mb)`, defaulting to `(2, 4096)` when unset or
    invalid.
  """
  try:
    env = task_toml(task).get("environment") or {}
  except (OSError, ValueError):
    env = {}

  def whole(key: str, default: int) -> int:
    value = env.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else default
    )

  return whole("cpus", 2), whole("memory_mb", 4096)


def default_trials(tasks: list[pathlib.Path], serve: bool = True) -> int:
  """Calculates default trial concurrency based on Docker daemon resources.

  Sizes concurrency so tasks do not compete for CPUs or memory, accounting for
  the maximum task resource request plus the per-trial serve stack when `serve`
  is True.

  Args:
    tasks: List of task directories to be run.
    serve: Whether each trial also starts a harness serve stack.

  Returns:
    The number of concurrent trials to run (at least 1).
  """
  cpus, mib = (max(values) for values in zip(*map(task_resources, tasks)))
  ncpu, total = build.daemon_resources() or (1, 0)
  by_cpu = ncpu // (cpus + (SERVE_CPUS if serve else 0))
  by_memory = total // ((mib << 20) + (SERVE_BYTES if serve else 0))
  return max(1, min(len(tasks), by_cpu, by_memory))


# Harbor runs as the child of this guard, in the guard's process group. The
# guard reads a pipe whose other end only the process that started it holds, so
# the pipe closes however that process dies (kill -9, the OOM killer). When it
# closes, or on SIGINT or SIGTERM, the guard sends Harbor one SIGINT, as Ctrl-C
# does, waits up to the seconds it was given for Harbor to stop, then kills the
# rest of the group. The guard reaps Harbor itself, so it never signals a
# process id Harbor no longer holds.
GUARD = """import contextlib, os, signal, subprocess, sys, threading
fd, stop_sec, cmd = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
stop = threading.Event()
signal.signal(signal.SIGINT, lambda *_: stop.set())
signal.signal(signal.SIGTERM, lambda *_: stop.set())
child = subprocess.Popen(cmd)
def watch():
    while os.read(fd, 1):
        pass
    stop.set()
def stopper():
    stop.wait()
    child.send_signal(signal.SIGINT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(stop_sec)
    os.killpg(0, signal.SIGKILL)
ender = threading.Thread(target=stopper, daemon=True)
for thread in (threading.Thread(target=watch, daemon=True), ender):
    thread.start()
code = child.wait()
if stop.is_set():
    ender.join()
sys.exit(128 - code if code < 0 else code)
"""
# Guard processes read _WATCHED; _ALIVE remains open only in this parent
# process.
_WATCHED, _ALIVE = os.pipe()


def guarded(
    cmd: list[str], stop_sec: float, **popen: Any
) -> subprocess.Popen[Any]:
  """Starts `cmd` under the `GUARD` process-group supervisor.

  The guard runs in its own session so its PID is the process group ID, and
  terminates the child group if this process exits or receives a signal.

  Args:
    cmd: Command and arguments to execute.
    stop_sec: Grace period in seconds after SIGINT before SIGKILL.
    **popen: Additional keyword arguments passed to `subprocess.Popen`.

  Returns:
    The started `subprocess.Popen` instance for the guard process.
  """
  return subprocess.Popen(
      [sys.executable, "-I", "-c", GUARD, str(_WATCHED), str(stop_sec), *cmd],
      start_new_session=True,
      pass_fds=(_WATCHED,),
      umask=CHILD_UMASK,
      **popen,
  )


def wait_harbor(cmd: list[str], env: dict[str, str], log: Any) -> int:
  """Runs Harbor under `GUARD` and returns its exit code.

  Forwards a first `KeyboardInterrupt` to the guard so Harbor can clean up its
  trials and containers, and kills the process group immediately on a second
  interrupt.

  Args:
    cmd: Harbor command line to execute.
    env: Environment variables for the Harbor process.
    log: Writable file object for stdout and stderr output.

  Returns:
    The exit status of the Harbor process.
  """
  proc = guarded(
      cmd,
      HARBOR_STOP_SEC,
      env=env,
      cwd=ROOT,
      stdout=log,
      stderr=subprocess.STDOUT,
  )
  try:
    return proc.wait()
  except KeyboardInterrupt:
    with contextlib.suppress(ProcessLookupError):
      proc.send_signal(signal.SIGINT)
    # Print after forwarding SIGINT and ignore broken-pipe errors so Harbor
    # still shuts down cleanly.
    with contextlib.suppress(OSError):
      print(
          "stopping Harbor: it cancels its trials and removes their containers"
          f" (up to {HARBOR_STOP_SEC} s; interrupt again to kill it now)",
          flush=True,
      )
    try:
      proc.wait()  # Bounded by the guard process's shutdown timeout.
    except KeyboardInterrupt:
      with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
      proc.wait()
    raise


def new_run_dir(name: str) -> pathlib.Path:
  """Creates and returns a unique `runs/<name>-<timestamp>` directory.

  Appends `-2`, `-3`, etc. if another run directory was created in the same
  second.

  Args:
    name: Submission or harness name prefix.

  Returns:
    The resolved path to the newly created run directory.
  """
  base = ROOT / "runs" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
  base.parent.mkdir(parents=True, exist_ok=True)
  for n in range(1, 1000):
    out = base if n == 1 else base.with_name(f"{base.name}-{n}")
    with contextlib.suppress(FileExistsError):
      out.mkdir()
      return out.resolve()
  sys.exit(f"{base} and 998 more like it exist already: give an --out")


def out_problem(
    out: pathlib.Path, entry: pathlib.Path | None = None
) -> str | None:
  """Checks whether `out` is a valid output directory for a run.

  Args:
    out: Candidate output directory path.
    entry: Optional submission directory path that `out` must not lie inside.

  Returns:
    An error message if `out` lies inside `entry` or is a non-empty existing
    path, or None if valid.
  """
  out = out.resolve()
  if entry is not None and (out == entry or entry in out.parents):
    return (
        f"--out {out} is inside the submission {entry}: choose a folder outside"
        " it (by default a run goes to runs/<name>-<time> in this repository)"
    )
  if out.exists() and (not out.is_dir() or any(out.iterdir())):
    return f"--out {out} is not an empty folder: give a new or empty one"
  return None


def task_limits(run: str) -> dict[str, Any]:
  """Builds the Docker Compose overlay for task container limits and labels.

  Applies the server's file-size limit (`build.MAX_FILE_BYTES`), process limit
  (`check.TASK_PIDS`), and the run label (`stack.RUN_LABEL`).

  Args:
    run: Run identifier string used to label the main task container.

  Returns:
    A Docker Compose configuration dictionary for `TASK_LIMITS`.
  """
  return {
      "services": {
          "main": {
              "ulimits": {"fsize": build.MAX_FILE_BYTES},
              "pids_limit": check.TASK_PIDS,
              "labels": {stack.RUN_LABEL: run},
          }
      }
  }


def remove_task_projects(run: str) -> None:
  """Removes Docker Compose projects for task containers belonging to `run`.

  Cleans up task containers and networks left behind if Harbor is killed.

  Args:
    run: Run identifier label value to filter containers by.
  """
  with contextlib.suppress(stack.InfraError):
    mains = stack.docker(
        "ps",
        "-a",
        "--filter",
        f"label={stack.RUN_LABEL}={run}",
        "--format",
        '{{.Label "' + stack.COMPOSE_PROJECT + '"}}',
        timeout=60,
    )
    stack.remove_projects(set(mains.stdout.split()), timeout=60)


def run_harbor(
    agent: dict[str, Any],
    tasks: list[pathlib.Path],
    root: pathlib.Path,
    trials: int,
    env_values: dict[str, str] | None,
    judge: dict[str, str] | None = None,
) -> pathlib.Path:
  """Executes a single `harbor run` invocation over `tasks` in `root`.

  Writes the serve container's environment to a private temporary file outside
  the shared run folder, and passes `judge` variables via Harbor's process
  environment for verifier containers. Cleans up leftover task projects and
  serve stacks on exit.

  Args:
    agent: Harbor agent configuration dictionary.
    tasks: List of staged task directory paths to run.
    root: Attempt directory where `job.yaml` and `harbor.log` are written.
    trials: Maximum number of concurrent trials.
    env_values: Optional environment variables for the serve container.
    judge: Optional `JUDGE_*` variables for verifier containers.

  Returns:
    Path to the Harbor job output directory (`root / "jobs" / "dev"`).

  Raises:
    HarborFailed: If `harbor run` exits non-zero without producing any trial
      result.
  """
  root.mkdir(parents=True)
  agent = copy.deepcopy(agent)
  env = {k: v for k, v in os.environ.items() if k in HARBOR_ENV}
  env.update(judge or {})
  env["PYTHONPATH"] = str(ROOT)
  # Match the evaluation server's default container platform (linux/amd64).
  env["DOCKER_DEFAULT_PLATFORM"] = stack.docker_env()["DOCKER_DEFAULT_PLATFORM"]
  job = root / "jobs" / "dev"
  # Assign a run label even for --url and --artifacts so remove_task_projects()
  # can clean up their task containers.
  run_id = agent["kwargs"].get("run") or uuid.uuid4().hex[:12]
  try:
    with contextlib.ExitStack() as held:
      if env_values is not None:
        agent["kwargs"]["env_file"] = str(
            held.enter_context(check.private_env_file(env_values))
        )
      (root / TASK_LIMITS).write_text(
          yaml.safe_dump(task_limits(run_id), sort_keys=False), encoding="utf-8"
      )
      config = {
          "job_name": job.name,
          "jobs_dir": str(job.parent),
          "n_concurrent_trials": min(trials, len(tasks)),
          "environment": {
              "import_path": ENVIRONMENT,
              "extra_docker_compose": [str(root / TASK_LIMITS)],
          },
          "agents": [agent],
          "tasks": [{"path": str(t)} for t in tasks],
      }
      (root / "job.yaml").write_text(
          yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
      )
      with (root / "harbor.log").open("w", encoding="utf-8") as log:
        rc = wait_harbor(
            [harbor(), "run", "-y", "-c", str(root / "job.yaml")], env, log
        )
  finally:
    # A killed Harbor leaves the task containers of its unfinished trials, and
    # Harbor removes no serve stack of a trial that stopped between its setup
    # and its agent phase. The task containers go first, since they are found by
    # the run's label, which the sweep of the serve stacks also matches.
    remove_task_projects(run_id)
    if agent["kwargs"].get("run"):
      with contextlib.suppress(stack.InfraError):
        stack.sweep(run=agent["kwargs"]["run"])
  # If at least one trial directory exists (even if unreadable), let the caller
  # inspect and retry individual tasks rather than failing the whole Harbor job.
  if rc != 0 and not any(job.glob("*/result.json")):
    raise HarborFailed(
        f"`harbor run` exited with status {rc} without running a trial; see"
        f" {root / 'harbor.log'}"
    )
  return job


def write_results(
    out: pathlib.Path,
    name: str,
    model: str,
    rows: list[Result],
    judge: dict[str, Any],
    source: dict[str, Any] | None,
) -> float:
  """Writes `results.json` atomically to `out` and returns the mean score.

  Args:
    out: Run output directory.
    name: Submission or harness name.
    model: Model identifier string.
    rows: List of per-task `Result` objects.
    judge: Judge configuration dictionary.
    source: Optional metadata dictionary (`submission_dir`, `model_base_url`,
      `artifacts`, `no_export`, and the task folders' record, `tasks_record`)
      preserved for `--regrade`.

  Returns:
    The mean score across all tasks in `rows`.
  """
  source = dict(source or {})
  # Redact user, password, and query parameters that might contain credentials.
  if source.get("model_base_url"):
    source["model_base_url"] = check.redact_url(source["model_base_url"])
  mean = sum(r.score for r in rows) / len(rows) if rows else 0.0
  tasks = [
      {
          **{k: v for k, v in dataclasses.asdict(r).items() if k != "retry"},
          "score": round(r.score, 6),
          "reward": None if r.reward is None else round(r.reward, 6),
          "trial": os.path.relpath(r.trial, out) if r.trial else "",
          "graded_by": os.path.relpath(r.graded_by, out) if r.graded_by else "",
      }
      for r in rows
  ]
  out.mkdir(parents=True, exist_ok=True)
  # Write to a PID/thread-scoped temporary file and atomically replace
  # results.json so concurrent checkpoint writes never leave a partial file.
  tmp = out / f".results.json.{os.getpid()}.{threading.get_ident()}.tmp"
  tmp.write_text(
      json.dumps(
          {
              "submission": name,
              "model": model,
              "score": round(mean, 6),
              "complete": all(r.measured for r in rows),
              "unmeasured": [r.id for r in rows if not r.measured],
              "judge_model": judge["JUDGE_MODEL"],
              "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              **source,
              "tasks": tasks,
          },
          indent=2,
      ),
      encoding="utf-8",
  )
  os.replace(tmp, out / "results.json")
  return mean


def rerun(args: str, env_file: str | None) -> str:
  """Formats a `uv run python eval/run.py` command string for user hints.

  Args:
    args: Command-line arguments to pass to `eval/run.py`.
    env_file: Optional `--env-file` path from the current run to preserve.

  Returns:
    The formatted command string.
  """
  extra = (
      f" --env-file {shlex.quote(os.path.abspath(env_file))}"
      if env_file
      else ""
  )
  return f"uv run python eval/run.py {args}{extra}"


def report(
    name: str,
    model: str,
    rows: list[Result],
    judge: dict[str, Any],
    out: pathlib.Path,
    source: dict[str, Any] | None = None,
    env_file: str | None = None,
) -> int:
  """Writes `results.json` and prints the score table and diagnostic hints.

  Args:
    name: Submission or harness name.
    model: Model identifier string.
    rows: List of per-task `Result` objects.
    judge: Judge configuration dictionary.
    out: Run output directory.
    source: Optional run source metadata for `results.json`.
    env_file: Optional `--env-file` path for follow-up command hints.

  Returns:
    Exit status code: 1 if any task is unmeasured, or 0 if all tasks were
    measured.
  """
  # Write results.json before printing so a broken stdout pipe cannot lose it.
  mean = write_results(out, name, model, rows, judge, source)
  width = max((len(r.id) for r in rows), default=len("task"))
  print(f"\n{'task':<{width}}  score  status")
  for r in rows:
    claimed = (
        f"; your run scored {r.claimed:.3f}" if r.claimed is not None else ""
    )
    print(
        f"{r.id:<{width}}  {r.score:.3f}  {r.status}"
        + (f" ({r.seconds:.0f} s)" if r.seconds else "")
        + claimed
    )
  print(f"\n{name} ({model}): {mean:.3f} over {len(rows)} task(s)")
  where = shlex.quote(str(out))
  crashed = [r.id for r in rows if r.kind == "harness"]
  if crashed:
    print(
        f"note: {', '.join(crashed)} ended with your harness's own error; what"
        f" it printed is in {where}/attempt*/jobs/dev/<task>__*/agent/"
        f" (serve.log, {name}.txt)"
    )
  unmeasured = [r for r in rows if not r.measured]
  if unmeasured:
    hints = []
    kinds = {r.kind for r in unmeasured}
    if "grader failed" in kinds:
      hints.append(
          "A grader that failed is usually the judge (JUDGE_*, or your model"
          " server): fix it and grade the deliverables again with"
          f" `{rerun('--regrade ' + where, env_file)}`, which does not run your"
          " harness."
      )
    if "not started" in kinds:
      hints.append(
          "A task not started ended with your harness's error before its model"
          " produced any output: what it printed is in"
          f" {where}/attempt*/jobs/dev/<task>__*/agent/ (serve.log,"
          f" {name}.txt)."
      )
    if "task error" in kinds:
      hints.append(
          "A task error is in the task's own task.toml (its reward_scale), not"
          " in your harness or the judge: update this repository (git pull) and"
          " run that task again."
      )
    verifiers = [
        str(pathlib.Path(r.graded_by or r.trial) / "verifier")
        for r in unmeasured
        if r.kind in ("no reward", "grader failed", "checks failed") and r.trial
    ]
    if verifiers:
      hints.append(
          "What the verifier printed is in the trial's verifier folder"
          " (test-stdout.txt, grade.log, and GRADER_FAILED.txt from a grader"
          f" that gave up): {', '.join(verifiers)}."
      )
    logs = attempt_logs(out)
    if logs and kinds & {
        "no trial",
        "harbor failed",
        "evaluator",
        "no reward",
        "grader failed",
        "checks failed",
    }:
      hints.append(f"What Harbor printed is in {logs[-1]}.")
    print(
        f"warning: {len(unmeasured)} task(s) have no measurement:"
        f" {', '.join(f'{r.id} ({r.status})' for r in unmeasured)}. They count"
        " 0 above, as on the leaderboard once its attempts are used up. "
        + " ".join(hints)
    )
  print(f"results: {out / 'results.json'}")
  return 1 if unmeasured else 0


def attempt_logs(out: pathlib.Path) -> list[pathlib.Path]:
  """Returns `harbor.log` paths from `attempt<N>` folders in attempt order.

  Args:
    out: Run output directory.

  Returns:
    Sorted list of existing `harbor.log` paths.
  """
  found = [
      (int(m.group(1)), d / "harbor.log")
      for d in out.glob("attempt*")
      if (m := re.fullmatch(r"attempt(\d+)", d.name))
      and d.is_dir()
      and (d / "harbor.log").is_file()
  ]
  return [log for _, log in sorted(found)]


def judge_or_exit(
    values: dict[str, Any], own: dict[str, Any] | None
) -> dict[str, Any]:
  """Resolves judge environment variables or exits on configuration error.

  Args:
    values: Run environment variables dictionary.
    own: Optional fallback judge settings from the submission's model.

  Returns:
    The resolved `JUDGE_*` configuration dictionary.
  """
  try:
    return judge_env(values, own)
  except JudgeError as exc:
    sys.exit(str(exc))


def judged(found: dict[str, pathlib.Path]) -> list[str]:
  """Filters `found` to tasks whose verifiers require an LLM judge.

  Reads each task's `task.toml` grading (`asks_judge`), as `outcome` and
  `stage_for_run` read it. Exits if a task's grading cannot be read.

  Args:
    found: Mapping from task ID to task directory.

  Returns:
    List of task IDs in `found` with `rubric` or `checks-judge` grading.
  """
  try:
    return [t for t, task in found.items() if asks_judge(task)]
  except TaskError as exc:
    sys.exit(f"dev tasks: {exc}")


def need_judge(
    found: dict[str, pathlib.Path],
    judge: dict[str, Any],
    skippable: bool = True,
) -> None:
  """Ensures a valid LLM judge is configured when any task in `found` needs one.

  Exits with an error message if a judge task lacks a valid judge endpoint, or
  prints the active judge model and redacted URL.

  Args:
    found: Mapping from selected task ID to task directory.
    judge: Resolved judge configuration dictionary.
    skippable: Whether the error hint may suggest excluding judge tasks with
      `--tasks`.
  """
  asked = judged(found)
  problem = judge_problem(judge) if asked else None
  if problem == NO_JUDGE:
    sys.exit(
        f"{', '.join(asked)}: {problem}"
        + (", or leave them out with --tasks" if skippable else "")
    )
  if problem:
    sys.exit(problem)
  # Redact credentials or query parameters before printing the judge URL.
  if asked:
    print(
        f"rubric and checks-judge tasks are judged by {judge['JUDGE_MODEL']} at"
        f" {check.redact_url(judge['JUDGE_BASE_URL'])} (JUDGE_* to change)",
        flush=True,
    )


def reach_judge(
    found: dict[str, pathlib.Path], judge: dict[str, Any]
) -> None:
  """Exits if verifier containers cannot reach the configured judge endpoint.

  Requires Docker to inspect the bridge gateway, so this runs after `preflight`
  and before building or grading.

  Args:
    found: Mapping from selected task ID to task directory.
    judge: Resolved judge configuration dictionary.
  """
  problem = judge_reach_problem(judge) if judged(found) else None
  if problem:
    sys.exit(problem)


def preflight() -> None:
  """Checks Harbor and Docker availability and sweeps leftover resources."""
  check_harbor(harbor())
  build.require_local_daemon()
  check.sweep_leftovers()


def tasks_or_exit(
    ids: list[str], tasks_dir: pathlib.Path | None, values: dict[str, Any]
) -> dict[str, pathlib.Path]:
  """Resolves task directories for `ids` or exits on error.

  Args:
    ids: Task IDs to resolve.
    tasks_dir: Optional local task directory override.
    values: Environment variables dictionary.

  Returns:
    A mapping from task ID to its local task directory path.
  """
  try:
    return resolve_tasks(ids, tasks_dir, values)
  except TasksError as exc:
    sys.exit(f"dev tasks: {exc}")


def stage_for_run(
    found: dict[str, pathlib.Path],
    out: pathlib.Path,
    judge: dict[str, Any],
) -> tuple[dict[str, pathlib.Path], dict[str, str] | None, dict[str, Result]]:
  """Stages task copies under `out/tasks` and prepares container judge settings.

  Configures verifier network allowlists for judge tasks (`asks_judge`). Tasks
  whose Docker images fail to build are excluded from the staged mapping and
  returned as unmeasured `Result` entries so the remaining tasks can still run.

  Args:
    found: Mapping from task ID to resolved source task directory.
    out: Run output directory.
    judge: Resolved judge configuration dictionary.

  Returns:
    A tuple of `(staged_task_dirs, verifier_judge_env, unbuilt_results)`.
  """
  unbuilt: dict[str, str] = {}
  try:
    asked = any(asks_judge(p) for p in found.values())
    values, host = judge_in_containers(judge) if asked else (None, None)
    staged = stage_tasks(found, out / "tasks", host, unbuilt)
  except (TaskError, TasksError) as exc:
    sys.exit(f"dev tasks: {exc}")
  for t, reason in unbuilt.items():
    print(f"note: {t}: {reason}; it is not run", flush=True)
  return (
      staged,
      values,
      {
          t: Result(
              t,
              status=f"{reason}; rerun to build it again",
              kind="image",
              retry=False,
          )
          for t, reason in unbuilt.items()
      },
  )


def trial_path(run_dir: pathlib.Path, recorded: str) -> pathlib.Path | None:
  """Resolves a trial path inside `run_dir` from its `attempt<N>` segment.

  Ensures a moved or copied run folder always references its own trials rather
  than another directory's.

  Args:
    run_dir: Path to the run directory.
    recorded: Trial path string recorded in `results.json`.

  Returns:
    The resolved trial path inside `run_dir`, or None if invalid.
  """
  parts = pathlib.Path(recorded).parts
  marks = [
      i for i, part in enumerate(parts) if re.fullmatch(r"attempt\d+", part)
  ]
  return (
      run_dir.joinpath(*parts[marks[-1] :])
      if marks and ".." not in parts
      else None
  )


def saved_result(row: dict[str, Any], run_dir: pathlib.Path) -> Result:
  """Reconstructs a `Result` from a `results.json` task entry.

  Args:
    row: Dictionary for a single task from `results.json`.
    run_dir: Path to the run directory.

  Returns:
    The reconstructed `Result` with trial paths resolved under `run_dir`.
  """
  fields = {
      k: row[k]
      for k in Result.__dataclass_fields__
      if k in row and k not in ("trial", "graded_by")
  }
  r = Result(**fields, retry=False)
  r.trial = str(trial_path(run_dir, row.get("trial") or "") or "")
  r.graded_by = str(run_dir / row["graded_by"]) if row.get("graded_by") else ""
  return r


def regrade(
    run_dir: pathlib.Path,
    values: dict[str, Any],
    env_file: str | None = None,
    tasks_dir: pathlib.Path | None = None,
) -> int:
  """Re-grades a finished run's failed judge gradings by replaying deliverables.

  Grades again, by replaying the deliverables their harness left (never the
  harness itself), the tasks whose judge grading did not finish ("grader
  failed"), with the judge chosen as the run chose it and the task folders it
  graded with (`recorded_tasks`); updates `results.json`, and exports artifacts
  for newly measured tasks. Any other unmeasured task (a cut-off, no reward, a
  checks grader that failed) needs its harness run again, as on the server, so
  it is left as it is.

  Args:
    run_dir: Path to the finished run directory.
    values: Environment variables dictionary.
    env_file: Optional `--env-file` path for command hints.
    tasks_dir: Optional local task directory override.

  Returns:
    Process exit status code (0 on complete measurement and export, 1
    otherwise).
  """
  try:
    saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    earlier = {t["id"]: t for t in saved["tasks"]}
  except (OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit(
        f"{run_dir} holds no results.json of eval/run.py ({type(exc).__name__}:"
        f" {exc})"
    )
  if not earlier:
    sys.exit(
        f"{run_dir / 'results.json'} lists no tasks: nothing to grade again"
    )
  if saved.get("artifacts"):
    sys.exit(
        "an --artifacts run grades the entry's artifacts/: run it again (`"
        "uv run python eval/run.py --artifacts <entry>`)"
    )
  results = {t: saved_result(row, run_dir) for t, row in earlier.items()}
  dev = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
  again = [
      t
      for t, r in results.items()
      if not r.measured and r.kind not in ("grader failed", "task error")
  ]
  if again:
    command = rerun(f"<entry> --tasks {','.join(again)}", env_file)
    print(
        f"note: {', '.join(again)} are unmeasured for a reason other than a"
        " judge grading that failed, so they need a run of your harness, not"
        f" another grading (`{command}`)",
        flush=True,
    )
  todo = [
      t
      for t, r in results.items()
      if not r.measured
      and r.kind == "grader failed"
      and r.trial
      and pathlib.Path(r.trial, "result.json").is_file()
  ]
  # When the run recorded a dataset commit or --tasks-dir, use that record;
  # otherwise restrict regrading to tasks listed in this checkout's dev set.
  own_record = bool(saved.get("tasks_dir")) or bool(
      HF_REVISION_RE.fullmatch(str(saved.get("tasks_revision") or ""))
  )
  for t in [t for t in todo if t not in dev and not own_record]:
    results[t].status = "task not in this checkout's dev set"
  todo = [t for t in todo if t in dev or own_record]
  if not todo:
    print(
        f"nothing to grade again: no task of {run_dir} has a judge grading that"
        " failed and a trial to replay"
    )
    # Match report(): return exit status 1 if any task remains unmeasured.
    return 1 if any(not r.measured for r in results.values()) else 0
  own = None
  if saved.get("submission_dir"):
    try:
      entry = pathlib.Path(saved["submission_dir"])
      own = own_judge(
          spec.load(entry), check.entry_env(entry), saved.get("model_base_url")
      )
    except (OSError, spec.SpecError, check.SetupError):
      print(
          f"note: {saved['submission_dir']} no longer holds a valid"
          f" submission.yaml and {spec.ENV_FILE}, so its model cannot be the"
          " judge of last resort",
          flush=True,
      )
  judge = judge_or_exit(values, own)
  found = recorded_tasks(todo, saved, tasks_dir, values)
  need_judge(found, judge, skippable=False)
  preflight()
  reach_judge(found, judge)
  # Preserve earlier results for any task whose image fails to build.
  task_dirs, verifier_env, unbuilt = stage_for_run(found, run_dir, judge)
  todo = [t for t in todo if t not in unbuilt]
  if todo:
    trials = default_trials(list(task_dirs.values()), serve=False)
    regrade_by_replay(
        results,
        todo,
        task_dirs,
        run_dir,
        trials,
        verifier_env,
        saved.get("submission") or "harness",
        time.strftime("%Y%m%d-%H%M%S"),
    )
  source = {
      k: saved[k]
      for k in (
          "submission_dir",
          "model_base_url",
          "artifacts",
          "no_export",
          "tasks_repo",
          "tasks_revision",
          "tasks_dir",
      )
      if k in saved
  }
  rows = [results[t] for t in earlier]
  rc = report(
      saved.get("submission", "harness"),
      saved.get("model", "unknown"),
      rows,
      judge,
      run_dir,
      source,
      env_file,
  )
  # Export any newly measured tasks that are not yet present in artifacts/.
  held = (export.read_manifest(run_dir / artifacts.ROOT) or {}).get(
      "tasks"
  ) or {}
  missing = [t for t in todo if results[t].measured and t not in held]
  if missing:
    rc = export_regraded(results, missing, run_dir, saved, env_file) or rc
  return rc


def export_regraded(
    results: dict[str, Result],
    missing: list[str],
    run_dir: pathlib.Path,
    saved: dict[str, Any],
    env_file: str | None = None,
) -> int:
  """Exports artifacts for tasks newly measured during `--regrade`.

  Adds the missing tasks' artifacts to `run_dir/artifacts` and, unless
  `--no-export` was set or there is no submission directory, to the entry's
  `artifacts/`.

  Args:
    results: Mapping from task ID to updated `Result`.
    missing: Task IDs that are now measured but absent from the run's manifest.
    run_dir: Path to the run directory.
    saved: Original `results.json` dictionary.
    env_file: Optional `--env-file` path for command hints.

  Returns:
    0 on success, or 1 if exporting or reviewing artifacts failed.
  """
  try:
    record = json.loads((run_dir / export.RECORD).read_text(encoding="utf-8"))
    harness = record["harness"]
  except (OSError, ValueError, KeyError, TypeError):
    print(
        f"note: the artifacts of {', '.join(missing)} are not exported:"
        f" {run_dir} keeps no {export.RECORD} (the harness that ran), so run"
        f" them again (`{rerun('<entry> --tasks <id>', env_file)}`)"
    )
    return 0
  entry = (
      pathlib.Path(saved["submission_dir"])
      if saved.get("submission_dir") and not saved.get("no_export")
      else None
  )
  with tempfile.TemporaryDirectory(
      prefix=f"{export.STAGING}regrade-",
      dir=run_dir,
      ignore_cleanup_errors=True,
  ) as staging:
    return export_artifacts(
        results,
        missing,
        run_dir,
        harness,
        record.get("dev_set") or "",
        entry,
        staging=pathlib.Path(staging),
    )


def export_rows(rows: list[Result], out: pathlib.Path) -> list[dict[str, Any]]:
  """Formats task results into row dictionaries for `export.json`.

  Args:
    rows: List of per-task `Result` objects.
    out: Run output directory used to relativize trial paths.

  Returns:
    List of row dictionaries consumed by `export.export_run`.
  """
  return [
      {
          "task": r.id,
          "trial": os.path.relpath(r.trial, out) if r.trial else None,
          "measured": r.measured,
          "reward": round(r.score, 6) if r.measured else None,
          "ended": r.status,
          "run": out.name,
      }
      for r in rows
  ]


def _record_rows(
    out: pathlib.Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  """Merges updated task `rows` into existing `export.json` rows in `out`.

  Args:
    out: Run output directory containing `export.json`.
    rows: New row dictionaries from re-graded tasks.

  Returns:
    The merged list of row dictionaries.
  """
  try:
    earlier = (
        json.loads((out / export.RECORD).read_text(encoding="utf-8")).get(
            "rows"
        )
        or []
    )
  except (OSError, ValueError, AttributeError):
    earlier = []
  mine = {r["task"]: r for r in rows}
  merged = [
      mine.pop(r["task"], r)
      for r in earlier
      if isinstance(r, dict) and "task" in r
  ]
  return merged + list(mine.values())


def harness_at_start(
    entry: pathlib.Path, name: str, model_api: str
) -> dict[str, Any]:
  """Reads harness provenance metadata for the manifest before a run starts.

  Exits if any harness file cannot be read or if `git` fails.

  Args:
    entry: Path to the submission directory.
    name: Submission name.
    model_api: Model API identifier string.

  Returns:
    The `harness` provenance dictionary from `export.provenance`.
  """
  try:
    return export.provenance(entry, name, model_api)
  except artifacts.Unreadable as exc:
    sys.exit(exc.problem)
  except artifacts.GitFailed as exc:
    sys.exit(artifacts.git_stop(entry, exc.said, exc.kind))


def load_entry(path: str) -> tuple[pathlib.Path, spec.Submission]:
  """Resolves and validates a submission directory and its `submission.yaml`.

  Args:
    path: Path to the submission directory or its `submission.yaml` file.

  Returns:
    A tuple of `(resolved_entry_dir, submission_spec)`.
  """
  try:
    entry = spec.entry_dir(path).resolve()
    return entry, spec.load(entry)
  except spec.EntryError as exc:
    sys.exit(str(exc))
  except FileNotFoundError:
    sys.exit(f"no {spec.FILE} in {path}")
  except spec.SpecError as exc:
    sys.exit(
        "submission.yaml is not valid:\n  - " + "\n  - ".join(exc.problems)
    )


def build_image(
    entry: pathlib.Path, sub: spec.Submission, rebuild: bool
) -> tuple[str, dict[str, Any]]:
  """Builds the submission's container image and records harness provenance.

  Verifies that the submission files did not change while being staged for the
  build, and exits with a diagnostic message if the build fails.

  Args:
    entry: Path to the submission directory.
    sub: Loaded submission specification.
    rebuild: Whether to force a rebuild even if cached layers match.

  Returns:
    A tuple of `(image_tag, harness_provenance)`.
  """
  try:
    work = build.work_dir(sub.name, src=entry)
  except build.BuildError as exc:
    sys.exit(f"build failed: {exc}")
  print(f"building {sub.name} ...", flush=True)
  try:
    with build.locked(work):
      harness = harness_at_start(entry, sub.name, sub.model.api)
      source = build.stage(entry, work / "source")
      try:
        same = spec.load(source.path) == sub
      except (OSError, spec.SpecError):
        same = False
      if (
          not same
          or harness_at_start(entry, sub.name, sub.model.api) != harness
      ):
        sys.exit(
            f"{entry} changed while the kit copied it for the build: run again"
            " once your edits are saved"
        )
      prep = build.prepare(sub, source, work, rebuild=rebuild)
  except build.BuildError as exc:
    sys.exit(f"build failed: {exc} (logs in {work})")
  except OSError as exc:
    # A file error on this machine. The work directory is outside the entry, so
    # a path inside the entry is one of the entry's files that could not be
    # read.
    where = os.fsdecode(exc.filename or "")
    if where and (os.path.realpath(where) + os.sep).startswith(
        str(entry) + os.sep
    ):
      sys.exit(
          f"build failed: {where} could not be read ({exc.strerror or exc}): a"
          " disk or I/O problem on this machine, not your harness; fix it,"
          " then run again"
      )
    sys.exit(
        f"build failed: {exc.strerror or exc} ({exc.filename or work}); free"
        " disk space or fix the cause, then run again"
    )
  if prep.reused:
    print(
        "  reusing the last build: files, install and builder unchanged"
        " (--rebuild to build anyway)",
        flush=True,
    )
  return prep.image, harness


def main(argv: list[str] | None = None) -> int:
  """Runs the CLI entrypoint with signal handling and error reporting.

  Args:
    argv: Optional command-line arguments list; defaults to `sys.argv[1:]`.

  Returns:
    Process exit status code.
  """
  # Configure stdout and stderr to replace unencodable characters on write.
  check.tolerant_output()
  try:
    # Translate SIGTERM and SIGHUP into KeyboardInterrupt so Harbor and temp
    # env files are cleaned up on termination.
    with check.stops_on_signals():
      return _main(argv)
  except stack.InfraError as exc:
    sys.exit(f"Docker failed: {exc}")
  # Handle errors in the submission's .env or local model server configuration.
  except check.SetupError as exc:
    sys.exit(f"{exc.stage}: {exc}")
  except KeyboardInterrupt:
    print(
        "interrupted: what the run started (Harbor) is stopped, and the env"
        " file with your keys removed",
        file=sys.stderr,
        flush=True,
    )
    return 130


def positive(text: str) -> int:
  """Parses a positive integer (>= 1) for `--trials` and `--attempts`.

  Args:
    text: String argument from the command line.

  Returns:
    The parsed positive integer.

  Raises:
    ValueError: If `text` is not an integer >= 1.
  """
  value = int(text)
  if value < 1:
    raise ValueError(text)
  return value


def _main(argv: list[str] | None) -> int:
  """Parses CLI arguments and dispatches evaluation, regrading, or replay.

  Args:
    argv: Command-line arguments list, or None to read `sys.argv[1:]`.

  Returns:
    Process exit status code.
  """
  ap = argparse.ArgumentParser(
      description=__doc__.splitlines()[0],
      epilog=__doc__.split("\n\n", 1)[1],
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument(
      "submission",
      nargs="?",
      help="a directory holding submission.yaml (or that file)",
  )
  ap.add_argument(
      "--url",
      help=(
          "use a server that is already running instead of building one"
          " (ws://host:port)"
      ),
  )
  ap.add_argument(
      "--name", help="with --url: what to call the harness in the results"
  )
  ap.add_argument(
      "--model",
      default="",
      help="with --url: the model it calls, for the results",
  )
  ap.add_argument(
      "--tasks", help="comma-separated dev task ids (default: all of them)"
  )
  ap.add_argument(
      "--trials",
      type=positive,
      help=(
          "tasks run (and graded) at once (default: as many as Docker's "
          "CPUs and memory hold; 1 with --url)"
      ),
  )
  ap.add_argument(
      "--attempts",
      type=positive,
      default=3,
      help="runs per task while the evaluator fails it",
  )
  ap.add_argument(
      "--env-file",
      help=(
          "NAME=value lines for this run: MODEL_PROXY_URL/KEY, JUDGE_* (your"
          " keys go in the submission's .env)"
      ),
  )
  ap.add_argument(
      "--model-base-url",
      help=(
          "use this server instead of model_base_url (a model you serve"
          " locally)"
      ),
  )
  ap.add_argument(
      "--out",
      type=pathlib.Path,
      help="a new or empty folder for the run (default runs/<name>-<time>)",
  )
  ap.add_argument(
      "--rebuild",
      action="store_true",
      help="build even if nothing changed since the last build",
  )
  ap.add_argument(
      "--tasks-dir",
      type=pathlib.Path,
      metavar="DIR",
      help=(
          "use the dev task folders in DIR (one per task id) instead of the"
          " dataset dev_set.yaml pins"
      ),
  )
  ap.add_argument(
      "--regrade",
      type=pathlib.Path,
      metavar="RUN_DIR",
      help=(
          "grade the tasks of a finished run (runs/<name>-<time>) whose judge"
          " grading failed again from the deliverables its harness left, and"
          " rewrite its results"
      ),
  )
  ap.add_argument(
      "--no-export",
      action="store_true",
      help=(
          "leave the entry's artifacts/ as it is (the run's own artifacts/ is"
          " written anyway)"
      ),
  )
  ap.add_argument(
      "--artifacts",
      metavar="SUBMISSION_DIR",
      help=(
          "grade the entry's artifacts/ the way the server does: your harness"
          " does not run; rubric and checks-judge tasks still call the judge"
      ),
  )
  a = ap.parse_args(argv)
  if a.regrade and {k for k, v in vars(a).items() if v != ap.get_default(k)} - {
      "regrade",
      "env_file",
      "tasks_dir",
  }:
    ap.error(
        "--regrade takes only the run directory (and --env-file, --tasks-dir)"
    )
  if a.artifacts and (a.submission or a.url):
    ap.error(
        "--artifacts takes the entry's directory (no other submission, no"
        " --url)"
    )
  if not a.regrade and not a.artifacts and bool(a.submission) == bool(a.url):
    ap.error("give a submission directory, --url, --artifacts or --regrade")
  if (a.name or a.model) and not a.url:
    ap.error(
        "--name and --model go with --url: a submission names its harness and"
        " model itself"
    )
  if a.url:
    # Strip whitespace before validation so the relay receives the checked URL.
    a.url = a.url.strip()
    problem = check.url_problem(a.url, "--url", ("ws", "wss"))
    if problem:
      ap.error(problem)
  # Validate --name since it is used for the run directory and log filenames.
  if a.name is not None and not spec.NAME_RE.fullmatch(a.name):
    ap.error(
        f"--name {a.name!r}: 2-39 characters of a-z, 0-9 and '-', starting with"
        " a letter or digit (an entry's name)"
    )
  a.out = a.out.resolve() if a.out else None
  a.tasks_dir = a.tasks_dir.resolve() if a.tasks_dir else None
  values = {**os.environ, **check.read_env_file(a.env_file)}
  if a.regrade:
    return regrade(a.regrade.resolve(), values, a.env_file, a.tasks_dir)
  if a.artifacts:
    return replay(a, values)
  dev = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
  wanted = select_tasks(a.tasks, dev)
  entry, sub = load_entry(a.submission) if a.submission else (None, None)
  problem = out_problem(a.out, entry) if a.out else None
  if problem:
    sys.exit(problem)
  own = rt = None
  if sub:
    dotenv = check.entry_env(entry)
    rt = check.runtime(sub, dotenv, values, base_url=a.model_base_url)
    for note in rt.notes:
      print(f"warning: {note}", flush=True)
    own = own_judge(sub, dotenv, a.model_base_url)
  judge = judge_or_exit(values, own)
  # Resolve tasks before building so download errors surface immediately.
  found = tasks_or_exit(wanted, a.tasks_dir, values)
  need_judge(found, judge)
  preflight()
  reach_judge(found, judge)

  if a.url:
    name, model = a.name or "harness", a.model or "unknown"
    agent = {
        "import_path": AGENT,
        "model_name": model,
        "kwargs": {"harness": name, "url": a.url},
    }
    env_values, trials = None, a.trials or 1
    harness = {
        "name": name,
        "model_api": model,
        "tree": None,
        "commit": None,
        "dirty": False,
    }
  else:
    name, model = sub.name, sub.model.api
    image, harness = build_image(entry, sub, a.rebuild)
    # Tag per-trial serve stacks with a run ID so run_harbor() can sweep them on
    # exit.
    agent = {
        "import_path": AGENT,
        "model_name": model,
        "kwargs": {
            "harness": name,
            "image": image,
            "serve": spec.fill(sub.serve, protocol.PORT),
            "allow": rt.allow,
            "exceptions": rt.exceptions,
            "run": uuid.uuid4().hex[:12],
        },
    }
    env_values, trials = rt.env, a.trials or default_trials(
        list(found.values())
    )
  # Copy each task's deliverable paths out of the container when its session
  # ends (see kit/capture.py).
  agent["kwargs"]["deliverables"] = {
      t: list(dev[t].deliverables) for t in wanted
  }
  out = a.out or new_run_dir(name)
  task_dirs, verifier_env, unbuilt = stage_for_run(found, out, judge)
  if sub:
    source = {
        "submission_dir": str(entry),
        "model_base_url": a.model_base_url,
        "no_export": a.no_export,
    }
  else:
    source = {}
  source.update(tasks_record(found, a.tasks_dir))
  digest = artifacts.dev_set_digest(artifacts.DEV_SET_FILE)

  def save(results: dict[str, Result]) -> None:
    rows = [results.get(t) or unbuilt[t] for t in wanted]
    write_results(out, name, model, rows, judge, source)
    export.save_record(out, export_rows(rows, out), harness, digest)

  runnable = [t for t in wanted if t not in unbuilt]
  results = {
      **unbuilt,
      **(
          run_attempts(
              agent,
              runnable,
              task_dirs,
              out,
              trials,
              env_values,
              verifier_env,
              a.attempts,
              save,
              a.env_file,
              regrade=name,
          )
          if runnable
          else {}
      ),
  }
  if not runnable:
    save(results)
  rc = report(
      name, model, [results[t] for t in wanted], judge, out, source, a.env_file
  )
  return (
      export_artifacts(
          results, wanted, out, harness, digest, None if a.no_export else entry
      )
      or rc
  )


def run_attempts(
    agent: dict[str, Any],
    wanted: list[str],
    task_dirs: dict[str, pathlib.Path],
    out: pathlib.Path,
    trials: int,
    env_values: dict[str, str] | None,
    judge: dict[str, str] | None,
    attempts: int,
    save: Callable[[dict[str, Result]], None],
    env_file: str | None = None,
    regrade: str | None = None,  # pylint: disable=redefined-outer-name
) -> dict[str, Result]:
  """Runs tasks through Harbor with retries on evaluator failure.

  Retries tasks that fail due to evaluator errors up to `attempts` times. When
  `regrade` is set (to the harness name), trials whose graders failed are
  re-graded from captured deliverables via `regrade_by_replay` instead of
  re-running the harness. Calls `save` after each attempt, each time a trial
  finishes while Harbor runs (`checkpoints`), and when the run is interrupted
  or fails.

  Args:
    agent: Harbor agent configuration dictionary.
    wanted: List of task IDs to evaluate.
    task_dirs: Mapping from task ID to staged task directory.
    out: Run output directory.
    trials: Maximum number of concurrent trials.
    env_values: Optional environment variables for the serve container.
    judge: Optional `JUDGE_*` variables for verifier containers.
    attempts: Maximum number of attempts per task on evaluator failure.
    save: Callback invoked with current results after each attempt, each
      finished trial, or an interrupt or failure.
    env_file: Optional `--env-file` path for interrupt hints.
    regrade: Optional harness name enabling deliverable replay when a grader
      fails.

  Returns:
    A mapping from task ID to its final `Result`.
  """
  results = {t: Result(t) for t in wanted}

  def keep(current: dict[str, Result] | None = None) -> None:
    try:
      save(results if current is None else current)
    except OSError as exc:
      print(
          f"warning: the results so far could not be saved ({exc})", flush=True
      )

  todo = list(wanted)
  for attempt in range(1, attempts + 1):
    root = out / f"attempt{attempt}"
    print(
        f"attempt {attempt}: {len(todo)} task(s), {min(trials, len(todo))} at a"
        f" time; Harbor writes {root / 'harbor.log'} (tail -f it to follow)",
        flush=True,
    )
    try:
      with checkpoints(root / "jobs" / "dev", todo, task_dirs, results, keep):
        job = run_harbor(
            agent,
            [task_dirs[t] for t in todo],
            root,
            trials,
            env_values,
            judge,
        )
      found = trials_by_task(job)
      results.update({t: outcome(t, task_dirs[t], found.get(t)) for t in todo})
      failed = [t for t in todo if results[t].kind == "grader failed"]
      if regrade and failed:
        regrade_by_replay(
            results,
            failed,
            task_dirs,
            out,
            trials,
            judge,
            regrade,
            f"attempt{attempt}",
        )
    except HarborFailed as exc:
      print(f"  Harbor failed: {exc}", flush=True)
      results.update({
          t: Result(
              t,
              status="no trial: Harbor failed",
              retry=False,
              kind="harbor failed",
          )
          for t in todo
      })
      break
    # On Ctrl-C or unexpected exceptions, save completed trials before
    # re-raising.
    # pylint: disable-next=broad-exception-caught
    except (KeyboardInterrupt, Exception):
      found = trials_by_task(root / "jobs" / "dev")
      results.update({
          t: outcome(t, task_dirs[t], found[t])
          for t in todo
          if t in found and not results[t].measured
      })
      keep()
      where = shlex.quote(str(out))
      hints: list[str] = []
      if regrade:
        hints.append(
            f"`{rerun('--regrade ' + where, env_file)}` grades again the"
            " deliverables of tasks whose grader did not finish"
        )
      # Skip export hint during --artifacts replay runs, which write no
      # export.json.
      if (out / export.RECORD).is_file():
        hints.append(
            f"`{KIT_ARTIFACTS} <entry> --from {where}` writes the measured"
            " tasks' artifacts into your entry"
        )
      print(
          f"kept what finished in {out}"
          + (": " + ", and ".join(hints) if hints else ""),
          flush=True,
      )
      raise
    keep()
    todo = [t for t in todo if results[t].retry]
    if not todo:
      break
    print(
        "  the evaluator failed"
        f" {', '.join(f'{t} ({results[t].status})' for t in todo)}; running"
        " again",
        flush=True,
    )
  return results


def export_artifacts(
    results: dict[str, Result],
    wanted: list[str],
    out: pathlib.Path,
    harness: dict[str, Any],
    digest: str,
    entry: pathlib.Path | None,
    staging: pathlib.Path | None = None,
) -> int:
  """Writes captured task artifacts to the run directory and submission entry.

  Exports measured task captures stamped with `harness` and `digest` into
  `out/artifacts` (or merges via `staging` during a regrade), then copies them
  into `entry/artifacts` if `entry` is provided and prints the artifact review.

  Args:
    results: Mapping from task ID to `Result`.
    wanted: Task IDs to export.
    out: Run output directory.
    harness: Harness provenance dictionary recorded at run start.
    digest: SHA-256 digest of `dev_set.yaml`.
    entry: Submission directory to copy artifacts into, or None if `--no-export`
      or `--url`.
    staging: Optional temporary staging directory for merging regraded tasks.

  Returns:
    0 if all measured task artifacts were exported and accepted by review, or 1
    otherwise.
  """
  dev = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
  rows = export_rows([results[t] for t in wanted], out)
  src = staging or out / artifacts.ROOT
  notes: list[str] = []
  target = shlex.quote(str(entry)) if entry else "<your entry>"
  retry = f"{KIT_ARTIFACTS} {target} --from {shlex.quote(str(out))}"
  try:
    export.save_record(
        out, _record_rows(out, rows) if staging else rows, harness, digest
    )
    done = export.export_run(rows, src, harness, digest, notes)
    if staging:
      # Merge newly regraded task artifacts into the run's artifacts/ folder.
      export.into_entry(src, out, dev, history=False, run=True)
  except (export.ExportError, OSError) as exc:
    print(
        f"\nthe run's artifacts could not be written: {export.reason(exc)}. The"
        " scores above stand: fix the cause (free disk space, a file you"
        f" cannot read), then `{retry}` writes them"
    )
    return 1
  exported = [d.task for d in done if d.entry is not None]
  # Count measured tasks whose deliverable capture failed.
  failed = int(any(d.entry is None and results[d.task].measured for d in done))
  print(
      f"\nartifacts: {len(exported)} task(s) in {out / artifacts.ROOT}"
      if exported
      else "\nartifacts: none"
  )
  for d in done:
    if d.entry is None:
      print(export.not_exported(d, entry))
    elif d.entry.get("skipped"):
      shown = "; ".join(
          f"{s.get('path')} ({s.get('reason')})" for s in d.entry["skipped"][:3]
      )
      print(
          f"  {d.task}: left out: {shown}"
          + (" ..." if len(d.entry["skipped"]) > 3 else "")
      )
  if not exported:
    return failed
  if entry is None:
    print("".join(f"  note: {n}\n" for n in notes), end="")
    own = artifacts.review(
        artifacts.from_directory(out, hashes=False, git=None),
        dev,
        read=artifacts.read_local(out),
    )
    verdict = server_verdict(own, exported)
    if verdict:
      print("\n".join([*own.lines(), verdict]))
    print(
        f"  copy them into your entry with `{KIT_ARTIFACTS} <entry> --from"
        f" {shlex.quote(str(out))}`"
    )
    return 1 if verdict else failed
  # Clear notes from exporting the run's copy before exporting into the entry.
  notes.clear()
  try:
    before = artifacts.review_directory(entry, dev, harness=False)
    copied = export.into_entry(src, entry, dev, notes)
  # The entry was not modified; the unreadable file is in the run directory.
  except export.RunUnreadable as exc:
    print(
        f"the run's artifacts cannot be read: {exc}. The scores above stand:"
        f" fix that, then `{retry}` copies them in"
    )
    return 1
  except (export.ExportError, OSError) as exc:
    print(
        f"the entry's artifacts/ could not be written: {export.reason(exc)}."
        f" The run keeps them in {out / artifacts.ROOT}: fix the cause, then"
        f" `{retry}` copies them in"
    )
    return 1
  print(
      f"  {entry / artifacts.ROOT}: replaced {', '.join(copied) or 'nothing'}"
  )
  print("".join(f"  note: {n}\n" for n in notes), end="")
  nothing = artifacts.blob_id(b"")
  empty = [
      d.task
      for d in done
      if d.entry is not None
      and d.task in copied
      and not d.entry["files"]
      and d.entry["final_message"] == nothing
  ]
  if empty:
    restore = (
        f"`git -C {shlex.quote(str(entry))} restore {artifacts.ROOT}` brings"
        " them back"
    )
    lost = [t for t in empty if before.tasks[t].status == artifacts.OK]
    # If the pre-export review stopped on an unreadable file, prior task status
    # is unknown.
    if before.unreadable:
      said = (
          " Whether they held usable artifacts before could not be checked"
          " (the kit stopped on the entry then); if those were committed,"
          f" {restore}."
      )
    elif lost:
      said = (
          f" The usable artifacts {', '.join(lost)} had before were replaced;"
          f" if they were committed, {restore}."
      )
    else:
      said = ""
    print(
        f"warning: your harness left nothing to grade in {len(empty)} task(s):"
        f" {', '.join(empty)}. What it did is in"
        f" {out}/attempt*/jobs/dev/<task>__*/agent/ (serve.log,"
        f" {harness.get('name') or 'harness'}.txt)."
        + said
    )
  if harness.get("dirty"):
    print(
        "  note: the harness had uncommitted changes when the run started;"
        " commit exactly those files, or the server sees a different harness"
        " than the one that made these artifacts"
    )
  try:
    rv = artifacts.review_directory(entry, dev)
  except OSError as exc:
    print(
        f"the entry's artifacts/ could not be reviewed ({type(exc).__name__}:"
        f" {exc}); `{KIT_ARTIFACTS} {shlex.quote(str(entry))}` reviews them"
        " once the cause is fixed"
    )
    return 1
  print("\n".join(rv.lines()))
  where = shlex.quote(str(entry))
  verdict = server_verdict(rv, copied)
  if rv.accepted:
    print(
        "commit them with the"
        f" harness{', though ' + verdict if verdict else ''}: git -C {where}"
        f" add -A && git -C {where} commit -m 'dev artifacts'"
    )
    return 1 if verdict else failed
  # Distinguish local filesystem/git errors from validation failures.
  if rv.unreadable:
    print(
        "not reviewed: the kit stopped on what the STOP line above names; fix"
        f" it as that line says, then `{KIT_ARTIFACTS} {where}` reviews them"
    )
  else:
    print(
        f"not ready to commit: {verdict}; fix what they say, run the dev set"
        " again, and commit then"
    )
  return 1


def server_verdict(rv: artifacts.Review, tasks: list[str]) -> str:
  """Summarizes whether the server would refuse or zero-score reviewed tasks.

  Args:
    rv: Completed artifact `Review` object.
    tasks: List of task IDs to check within `rv`.

  Returns:
    A warning message if the review was refused or any task in `tasks` is
    unusable, or an empty string if all `tasks` would be graded.
  """
  if not rv.accepted:
    return "the server would refuse these artifacts (the REFUSED lines above)"
  unusable = [t for t in tasks if rv.tasks[t].status != artifacts.OK]
  return (
      f"{', '.join(unusable)} would count 0 on the server (see above)"
      if unusable
      else ""
  )


def replay(a: argparse.Namespace, values: dict[str, Any]) -> int:
  """Reviews and grades an entry's `artifacts/` directory without running it.

  Replays each usable task's artifacts into a fresh container for verification
  and scores tasks without usable artifacts as 0, matching server evaluation.

  Args:
    a: Parsed CLI arguments namespace (`--artifacts`).
    values: Environment variables dictionary.

  Returns:
    Process exit status code.
  """
  entry, sub = load_entry(a.artifacts)
  problem = out_problem(a.out, entry) if a.out else None
  if problem:
    sys.exit(problem)
  dev = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
  wanted = select_tasks(a.tasks, dev)
  try:
    rv = artifacts.review_directory(entry, dev)
  # Top-level directory cannot be listed (unreadable inner files are reported in
  # rv.unreadable).
  except OSError as exc:
    sys.exit(f"could not read {exc.filename or entry}: {exc.strerror or exc}")
  print("\n".join(rv.lines()), flush=True)
  if rv.unreadable:
    print(
        "nothing graded: the kit stopped on what the STOP line above names; fix"
        " it as that line says, then run again"
    )
    return 1
  if not rv.accepted:
    print(
        "the server would refuse these artifacts when you submit: fix the"
        " REFUSED lines above"
    )
    return 1
  judge = judge_or_exit(
      values, own_judge(sub, check.entry_env(entry), a.model_base_url)
  )
  usable = [t for t in wanted if rv.tasks[t].status == artifacts.OK]
  found = tasks_or_exit(usable, a.tasks_dir, values) if usable else {}
  need_judge(found, judge)
  out = a.out or new_run_dir(f"{sub.name}-artifacts")
  staged = out / "replay"
  source = {
      "submission_dir": str(entry),
      "model_base_url": a.model_base_url,
      "artifacts": True,
  }

  def rows(results: dict[str, Result]) -> list[Result]:
    """Builds the ordered result rows combining replay grades and review status.

    Args:
      results: Mapping from task ID to replay `Result` for usable tasks.

    Returns:
      List of `Result` objects for all `wanted` tasks.
    """
    listed = []
    for t in wanted:
      r = rv.tasks[t]
      row = results.get(t) or Result(
          t,
          measured=True,
          retry=False,
          kind="artifact",
          status=(
              f"artifacts {r.status}: {(r.problems or ['no folder'])[0][:160]}"
          ),
      )
      row.artifact, row.claimed = r.status, r.claimed
      listed.append(row)
    return listed

  def save(current: dict[str, Result]) -> None:
    # Include unbuilt tasks alongside current results from run_attempts().
    write_results(
        out,
        sub.name,
        sub.model.api,
        rows({**results, **current}),
        judge,
        source,
    )

  results: dict[str, Result] = {}
  try:
    try:
      artifacts.materialize(
          rv,
          dev,
          lambda rel: open(
              os.open(entry / rel, os.O_RDONLY | os.O_NOFOLLOW), "rb"
          ),
          staged,
          only=wanted,
      )
    # Abort grading if local filesystem or permission errors prevent staging.
    except OSError as exc:
      print(
          "the artifacts could not be copied for grading:"
          f" {exc.strerror or exc} ({exc.filename or staged}); fix that and run"
          " again"
      )
      return 1
    # Re-filter usable tasks in case materialize() rejected files modified after
    # the initial review.
    usable = [t for t in usable if rv.tasks[t].status == artifacts.OK]
    if usable:
      preflight()
      reach_judge({t: found[t] for t in usable}, judge)
      task_dirs, verifier_env, unbuilt = stage_for_run(
          {t: found[t] for t in usable}, out, judge
      )
      results.update(unbuilt)
      usable = [t for t in usable if t not in unbuilt]
    if usable:
      # Strip collect hooks since the exported artifacts already include their
      # output.
      try:
        task_dirs = replay_staged(task_dirs, out / "replay-tasks")
      except TasksError as exc:
        sys.exit(f"dev tasks: {exc}")
      agent = {
          "import_path": REPLAY_AGENT,
          "model_name": "replay",
          "kwargs": {"artifacts": str(staged), "harness": sub.name},
      }
      trials = a.trials or default_trials(list(task_dirs.values()), serve=False)
      results.update(
          run_attempts(
              agent,
              usable,
              task_dirs,
              out,
              trials,
              None,
              verifier_env,
              a.attempts,
              save,
              a.env_file,
          )
      )
  finally:
    # Remove temporary replay staging; each trial directory retains what was
    # graded.
    shutil.rmtree(staged, ignore_errors=True)
  rc = report(
      sub.name, sub.model.api, rows(results), judge, out, source, a.env_file
  )
  if not rv.accepted:  # materialize() rejected the last usable task.
    print(
        "the server would refuse these artifacts when you submit:"
        f" {'; '.join(rv.problems)}"
    )
    return 1
  if len(wanted) < len(dev):
    print(
        f"graded {len(wanted)} of the {len(dev)} dev tasks (--tasks); the"
        " server grades all of them"
    )
  print(
      "a preview: the official dev score is the server's grade of the artifacts"
      " in the commit you submit"
  )
  return rc


if __name__ == "__main__":
  sys.exit(main())
