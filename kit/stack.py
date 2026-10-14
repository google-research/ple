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

"""Manages serve container stacks and egress proxies for PLEBench evaluation.

Runs each serve container and its egress proxy on an internal Docker network
so that the serve container reaches only allowed hosts and publishes its
WebSocket on `127.0.0.1`:

  serve container --(internal net)--> egress proxy --(egress net)--> allowed
  evaluator --> 127.0.0.1:<published> --> egress proxy --> serve:8765
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

from kit.protocol import PORT
from kit.spec import AGENT_TIMEOUT_SEC

LABEL = "plebench.serve"
# on a serve image: the entry (work directory) it was built for
ENTRY_LABEL = "plebench.entry"
# on a stack eval/run.py started: its run, which it sweeps when the job ends
RUN_LABEL = "plebench.run"
# Docker Compose's label on a project's containers and networks
COMPOSE_PROJECT = "com.docker.compose.project"
RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# a registry's kind (register): a directory of the kit's cache
KIND_RE = re.compile(r"[a-z]{1,32}")
# the registry kind of images live processes start containers from
HELD = "held"
BUILDER_REPO = "plebench-builder"  # kit/build.py's builder images
# old serve images per entry, and builder images, kept besides those in use
KEEP_IMAGES = 2
# an image this young may still start trials of a run in progress (eval/run.py)
RECENT_IMAGE_SEC = AGENT_TIMEOUT_SEC + 3600
# the evaluation server's; Docker on Apple Silicon emulates it
PLATFORM = "linux/amd64"
EGRESS_IMAGE = "python:3.12-slim"
EGRESS_SCRIPT = pathlib.Path(__file__).with_name("egress.py")
EGRESS_NETWORK = "plebench-serve-egress"
PROXY_PORT = 3128
FORWARD_PORT = 8000
IMAGE_REPO = "plebench-serve"
HOME = "/home/harness"
CPUS, MEMORY, PIDS = "2", "4g", "512"
EGRESS_MEMORY = "256m"
LOG_BYTES = 4 << 20
# The kit reads both containers' output with `docker logs` (the egress proxy's
# readiness too), so the log driver is pinned: the daemon's default may be one
# docker logs cannot read (none, or a remote one).
LOG_OPTS = (
    "--log-driver",
    "json-file",
    "--log-opt",
    "max-size=8m",
    "--log-opt",
    "max-file=2",
)
# docker's and podman's wording
NO_SUCH_CONTAINER = re.compile(r"no such (?:container|object)", re.IGNORECASE)
EGRESS_READY_SEC = 30
# a stack this old was left behind: no trial takes the agent limit plus 2 h
SWEEP_AGE_SEC = AGENT_TIMEOUT_SEC + 2 * 3600

# The bundle is packed world-readable (build.py), so ADD alone gives the right
# modes, in one layer.
SERVE_DOCKERFILE = """\
FROM {base}
ADD bundle.tar.gz /
ENV HOME={home} USER=harness
WORKDIR /opt/harness/src
"""


class InfraError(RuntimeError):
  """Raised when Docker or the egress proxy fails on the evaluator side."""


def cache_root() -> pathlib.Path:
  """Returns the root directory for the kit's cached files.

  Located under `$HOME`, which Docker Desktop, Colima, and Lima share with
  their VMs (unlike the system temporary directory), so any file bind-mounted
  into a container is placed under this root.

  Returns:
    Path to `~/.cache/plebench`.
  """
  return pathlib.Path.home() / ".cache" / "plebench"


def kit_root() -> pathlib.Path:
  """Returns the directory for the kit's internal bookkeeping.

  Starts with a dot so no entry work directory (such as one named `held` or
  `egress`) can collide with it.

  Returns:
    Path to `~/.cache/plebench/.kit`.
  """
  return cache_root() / ".kit"


def egress_root() -> pathlib.Path:
  """Returns the base directory where per-stack copies of `egress.py` live.

  Each stack stores its copy at `<run or "-">/<stack id>/egress.py`.

  Returns:
    Path to the `egress` subdirectory under `kit_root()`.
  """
  return kit_root() / "egress"


def registry_root(kind: str) -> pathlib.Path:
  """Returns the directory holding active process registrations for `kind`.

  Used by `register()` and `live()` to track `up` stacks, `held` images, and
  `secrets` directories.

  Args:
    kind: Registry category name.

  Returns:
    Path to `kit_root() / kind`.
  """
  return kit_root() / kind


# the other platform DOCKER_DEFAULT_PLATFORM named, once warned about
_warned_platform: list[str] = []


def docker_env() -> dict[str, str]:
  """Returns the environment for Docker CLI calls pinned to `linux/amd64`.

  Overrides any `DOCKER_DEFAULT_PLATFORM` targeting another platform (emitting
  a one-time warning) so images and containers always match the evaluation
  server's architecture.

  Returns:
    Environment dictionary with `DOCKER_DEFAULT_PLATFORM` set to `PLATFORM`.
  """
  env = dict(os.environ)
  asked = env.get("DOCKER_DEFAULT_PLATFORM", "")
  if (
      asked
      and asked.split("/")[:2] != PLATFORM.split("/")
      and not _warned_platform
  ):
    _warned_platform.append(asked)
    print(
        f"kit: DOCKER_DEFAULT_PLATFORM={asked} is ignored: images and"
        f" containers run as {PLATFORM}, as on the evaluation server",
        file=sys.stderr,
        flush=True,
    )
  env["DOCKER_DEFAULT_PLATFORM"] = PLATFORM
  return env


# the docker-driver builder's name, once `docker context show` gave it
_daemon_builder: str | None = None


def build_env() -> dict[str, str]:
  """Returns the environment for `docker build` using the daemon's builder.

  Sets `BUILDX_BUILDER` to the current Docker context name so `docker build`
  uses the `docker`-driver builder even if a `docker-container` buildx builder
  is selected (which would not load built images into the daemon's image store
  or resolve local `FROM` images).

  Returns:
    Environment dictionary configured for `docker build`.
  """
  global _daemon_builder
  env = docker_env()
  if _daemon_builder is None:
    try:
      r = docker("context", "show", timeout=30, check=False)
    except InfraError:  # no docker CLI: the build says so
      return env
    if r.returncode != 0 or not r.stdout.strip():
      return env
    _daemon_builder = r.stdout.strip()
  env["BUILDX_BUILDER"] = _daemon_builder
  return env


def docker(
    *args: str,
    timeout: float = 120,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  """Runs a Docker CLI command and returns the completed process.

  Args:
    *args: Command-line arguments passed to `docker`.
    timeout: Maximum time in seconds to wait for the command.
    check: Whether to raise `InfraError` on a non-zero exit code.
    env: Environment variables for the subprocess; defaults to `docker_env()`.

  Returns:
    The completed subprocess result with captured UTF-8 output.

  Raises:
    InfraError: If the command fails to start, times out, or exits non-zero
      when `check` is True.
  """
  try:
    r = subprocess.run(
        ["docker", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env or docker_env(),
        check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise InfraError(f"docker {args[0]}: {exc}") from exc
  if check and r.returncode != 0:
    raise InfraError(
        f"docker {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-600:]}"
    )
  return r


def _exists(kind: str, name: str) -> bool:
  """Returns whether a Docker object of `kind` and `name` exists."""
  return docker(kind, "inspect", name, check=False).returncode == 0


def docker_answers() -> bool:
  """Whether the Docker daemon answers."""
  try:
    return (
        docker(
            "info", "--format", "{{.ServerVersion}}", timeout=30, check=False
        ).returncode
        == 0
    )
  except InfraError:
    return False


def container_alive(container: str) -> bool:
  """Checks whether `container` is still running or was OOM-killed.

  In either case, the container's state reflects the workload inside it rather
  than an evaluator infrastructure failure.

  Args:
    container: Name or ID of the container to inspect.

  Returns:
    True if the container is running or was OOM-killed; False otherwise.
  """
  try:
    r = docker(
        "inspect",
        "-f",
        "{{.State.Running}} {{.State.OOMKilled}}",
        container,
        timeout=30,
        check=False,
    )
  except InfraError:
    return False
  return r.returncode == 0 and "true" in r.stdout.split()


def image_id(tag: str) -> str:
  """The local image's id; "" when there is none."""
  r = docker("image", "inspect", "-f", "{{.Id}}", tag, check=False)
  return r.stdout.strip() if r.returncode == 0 else ""


# whether `docker image inspect --platform` works here (Docker 28.1 on)
_inspect_platform = [True]


def ensure_image(image: str) -> None:
  """Ensures `image` is present locally for `PLATFORM`, pulling it if needed.

  Uses `--platform` when supported by `docker image inspect` so multi-platform
  image stores report the `linux/amd64` copy rather than the host's native
  architecture.

  Args:
    image: Docker image reference to verify or pull.
  """
  fmt = ("-f", "{{.Os}}/{{.Architecture}}")
  if _inspect_platform[0]:
    have = docker(
        "image", "inspect", "--platform", PLATFORM, *fmt, image, check=False
    )
    # a docker CLI or daemon older than the flag
    if have.returncode != 0 and "--platform" in have.stderr:
      _inspect_platform[0] = False
  if not _inspect_platform[0]:
    have = docker("image", "inspect", *fmt, image, check=False)
  if have.returncode != 0 or have.stdout.strip() != PLATFORM:
    docker("pull", image, timeout=600)


def _used_images() -> set[str]:
  """Returns the IDs of all images referenced by any existing container.

  Includes running, stopped, and created containers.

  Returns:
    Set of image IDs currently used by containers.
  """
  ids = docker("ps", "-a", "-q", "--no-trunc", check=False).stdout.split()
  if not ids:
    return set()
  out = docker("inspect", "-f", "{{.Image}}", *ids, check=False).stdout
  return set(out.split())


def _process_started(pid: int) -> str | None:
  """Reads the kernel start time for `pid` from `/proc/<pid>/stat`.

  Distinguishes `pid` from a later process assigned the same PID on Linux;
  returns None when `/proc` is unavailable (such as on macOS).

  Args:
    pid: Process ID to inspect.

  Returns:
    The start-time field from `/proc/<pid>/stat`, or None if unavailable.
  """
  try:
    return (
        pathlib.Path(f"/proc/{pid}/stat")
        .read_text()
        .rpartition(")")[2]
        .split()[19]
    )
  except (OSError, IndexError):
    return None


def _alive(pid: int, started: str | None) -> bool:
  """Checks whether `pid` is still running and matches its `started` timestamp.

  On systems without `/proc` (such as macOS), only PID existence can be
  checked.

  Args:
    pid: Process ID to check.
    started: Kernel start timestamp recorded at registration, or None.

  Returns:
    True if the process exists and matches `started` (when available).
  """
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  # another user's process has the pid; its start time decides
  except PermissionError:
    pass
  now = _process_started(pid)
  return started is None or now is None or now == started


def register(
    kind: str, payload: dict[str, Any], key: str | None = None
) -> pathlib.Path:
  """Registers `payload` under `kind` for the lifetime of the current process.

  Writes `registry_root(kind)/<key>.json` containing the current PID, process
  start time, and `payload`. `live(kind)` returns the payload while this
  process runs and cleans it up once the process exits.

  Args:
    kind: Registry category name matching `KIND_RE`.
    payload: Dictionary of metadata to store with the registration.
    key: Optional unique key for the registration file; a second registration
      with the same key replaces the first.

  Returns:
    Path to the registration JSON file, which can be passed to `unregister()`.

  Raises:
    ValueError: If `kind` does not match `KIND_RE`.
  """
  if not KIND_RE.fullmatch(kind):
    raise ValueError(f"register: {kind!r} is not a registry kind")
  pid = os.getpid()
  key = key or f"{pid}-{uuid.uuid4().hex[:12]}"
  if not RUN_RE.fullmatch(key):
    key = hashlib.sha256(key.encode()).hexdigest()[:32]
  root = registry_root(kind)
  root.mkdir(parents=True, exist_ok=True)
  path = root / f"{key}.json"
  # renamed into place, so never half written
  partial = root / f".{key}.{uuid.uuid4().hex[:8]}.partial"
  partial.write_text(
      json.dumps(
          {"pid": pid, "started": _process_started(pid), "payload": payload}
      ),
      encoding="utf-8",
  )
  partial.replace(path)
  return path


def unregister(path: pathlib.Path) -> None:
  """Remove one registration (register's path) before its process exits."""
  with contextlib.suppress(OSError):
    pathlib.Path(path).unlink()


def live(kind: str, gone=None) -> list[dict[str, Any]]:
  """Returns payloads registered under `kind` by processes that are still alive.

  For any registration whose owning process has exited, calls `gone(payload)`
  (if provided) to clean up left-behind resources before removing the
  registration file. If `gone` raises `InfraError`, the registration is kept
  for the next call.

  Args:
    kind: Registry category name.
    gone: Optional callback invoked with `payload` for dead processes.

  Returns:
    List of payload dictionaries belonging to currently running processes.
  """
  root, out = registry_root(kind), []
  for path in (
      sorted(root.glob("*.json"))
      if KIND_RE.fullmatch(kind) and root.is_dir()
      else ()
  ):
    try:
      entry = json.loads(path.read_text())
      pid, payload = int(entry["pid"]), entry["payload"]
      if not isinstance(payload, dict):
        raise TypeError("the payload is not a mapping")
    except FileNotFoundError:  # another process dropped it meanwhile
      continue
    except (OSError, ValueError, KeyError, TypeError):
      unregister(path)
      continue
    if _alive(pid, entry.get("started")):
      out.append(payload)
      continue
    if gone is not None:
      try:
        gone(payload)
      except InfraError:
        continue
    unregister(path)
  return out


def hold_image(tag: str) -> None:
  """Registers `tag` as held by the current process until it exits.

  Prevents `prune_images()` from removing an image that an active process
  (such as `eval/run.py` starting trials across hours) still needs.

  Args:
    tag: Docker image tag to protect from pruning.
  """
  register(
      HELD,
      {"image": tag},
      key=f"{os.getpid()}-{hashlib.sha256(tag.encode()).hexdigest()[:16]}",
  )


def _held_images() -> set[str]:
  """Returns the set of image tags currently held by live processes.

  Drops registrations belonging to processes that have exited.

  Returns:
    Set of held Docker image tags.
  """
  return {p["image"] for p in live(HELD) if isinstance(p.get("image"), str)}


def prune_images(
    repository: str,
    keep: int = KEEP_IMAGES,
    *,
    label: str = "",
    protect: str = "",
    used: set[str] | None = None,
    held: set[str] | None = None,
) -> int:
  """Removes old kit images from `repository` while keeping active ones.

  Spares `protect`, images held by live processes (`hold_image`), images used
  by any container, images built within `RECENT_IMAGE_SEC`, and the `keep`
  newest remaining images.

  Args:
    repository: Docker repository name whose images should be pruned.
    keep: Number of newest eligible images to retain.
    label: Optional Docker label filter (`key=value`).
    protect: Specific image tag to spare from removal.
    used: Precomputed set of image IDs in use by containers, or None to query.
    held: Precomputed set of image tags held by live processes, or None to
      query.

  Returns:
    Number of Docker image tags removed.
  """
  filters = ["--filter", f"reference={repository}"] + (
      ["--filter", f"label={label}"] if label else []
  )
  listed = docker(
      "image",
      "ls",
      *filters,
      "--format",
      "{{.Repository}}:{{.Tag}}",
      check=False,
  ).stdout.split()
  tags = [
      t
      for t in listed
      if t.startswith(f"{repository}:") and not t.endswith(":<none>")
  ]
  if not tags:
    return 0
  # one docker call for all the tags; each line names its image's tags, so one
  # that vanished meanwhile is skipped
  born, ids = {}, {}
  out = docker(
      "image",
      "inspect",
      "-f",
      "{{.Created}}|{{.Id}}|{{json .RepoTags}}",
      *tags,
      check=False,
  ).stdout
  for line in out.splitlines():
    stamp, iid, repo_tags = (line.split("|", 2) + ["", ""])[:3]
    try:
      names = json.loads(repo_tags) or []
    except ValueError:
      continue
    for tag in set(names) & set(tags):
      born[tag], ids[tag] = created_epoch(stamp), iid
  used = _used_images() if used is None else used
  held = _held_images() if held is None else held
  now = time.time()
  rest = [
      t
      for t in tags
      if t != protect
      and t not in held
      and t in ids
      and ids[t] not in used
      and born[t] is not None
      and now - born[t] >= RECENT_IMAGE_SEC
  ]
  removed = 0
  for tag in sorted(rest, key=born.get, reverse=True)[keep:]:
    if docker("image", "rm", tag, check=False).returncode == 0:
      removed += 1
  return removed


def entry_label(name: str) -> str:
  """Sanitizes an entry work directory `name` for use as `ENTRY_LABEL`.

  Args:
    name: Raw directory name of the entry.

  Returns:
    Sanitized label value of up to 128 characters.
  """
  return re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:128] or "entry"


def serve_image(
    bundle: pathlib.Path,
    base_image: str,
    log: pathlib.Path | None = None,
    digest: str | None = None,
    base_id: str | None = None,
) -> str:
  """Builds (or reuses) the serve image unpacking `bundle` onto `base_image`.

  Tags the image by a hash of `digest`, `base_id`, and `SERVE_DOCKERFILE`,
  registers it via `hold_image()`, and prunes older serve and builder images.

  Args:
    bundle: Path to the `bundle.tar.gz` archive.
    base_image: Tag of the builder base image.
    log: Optional file path where build output is written.
    digest: Optional precomputed SHA-256 hex digest of `bundle`.
    base_id: Optional precomputed Docker image ID of `base_image`.

  Returns:
    The tag of the built or cached serve image.

  Raises:
    InfraError: If `base_image` is missing or `docker build` fails.
  """
  if digest is None:
    h = hashlib.sha256()
    with bundle.open("rb") as f:
      for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
    digest = h.hexdigest()
  base_id = base_id or image_id(base_image)
  if not base_id:
    raise InfraError(
        f"the image {base_image} is not there to build the serve image from"
    )
  dockerfile = SERVE_DOCKERFILE.format(base=base_image, home=HOME)
  # the tag covers everything that shapes the image, so a change to
  # SERVE_DOCKERFILE (HOME, WORKDIR) builds anew
  shape = "\n".join([digest, base_id, dockerfile])
  tag = f"{IMAGE_REPO}:{hashlib.sha256(shape.encode()).hexdigest()[:16]}"
  hold_image(tag)
  if image_id(tag):
    return tag
  # the context sits beside the bundle, so the bundle (up to 2 GiB) is linked
  # into it rather than copied
  with tempfile.TemporaryDirectory(
      prefix="plebench-serve-", dir=bundle.parent
  ) as ctx:
    try:
      os.link(bundle, pathlib.Path(ctx) / "bundle.tar.gz")
    except OSError:
      shutil.copyfile(bundle, pathlib.Path(ctx) / "bundle.tar.gz")
    (pathlib.Path(ctx) / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    label = f"{ENTRY_LABEL}={entry_label(bundle.parent.name)}"
    r = docker(
        "build",
        "-q",
        "--label",
        label,
        "-t",
        tag,
        ctx,
        timeout=1800,
        check=False,
        env=build_env(),
    )
  if log is not None:
    log.write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
  if r.returncode != 0:
    raise InfraError(
        "building the serve image failed:"
        f" {(r.stderr or r.stdout).strip()[-600:]}"
    )
  try:
    # one scan of the containers and holders for both prunes
    used, held = _used_images(), _held_images()
    prune_images(IMAGE_REPO, label=label, protect=tag, used=used, held=held)
    if base_image.startswith(f"{BUILDER_REPO}:"):
      prune_images(BUILDER_REPO, protect=base_image, used=used, held=held)
  except InfraError:  # tidying up is best effort; the image is built
    pass
  return tag


def _ensure_egress_base() -> None:
  """Ensures `EGRESS_IMAGE` is present and `EGRESS_NETWORK` exists.

  Raises:
    InfraError: If pulling the image or creating the network fails.
  """
  ensure_image(EGRESS_IMAGE)
  if not _exists("network", EGRESS_NETWORK):
    r = docker(
        "network",
        "create",
        "--label",
        f"{LABEL}=shared",
        EGRESS_NETWORK,
        check=False,
    )
    if r.returncode != 0 and not _exists("network", EGRESS_NETWORK):
      raise InfraError(f"creating {EGRESS_NETWORK}: {r.stderr.strip()[-300:]}")


async def settled(start, undo) -> None:
  """Runs `start` in a worker thread, cleaning up with `undo` if interrupted.

  If the awaiting coroutine is cancelled or interrupted while `start` is
  running in the executor thread, waits for `start` to finish, runs `undo` in
  an executor thread to remove any created containers, and re-raises.

  Args:
    start: Zero-argument callable that starts containers.
    undo: Zero-argument callable that removes what `start` created.
  """
  loop = asyncio.get_running_loop()
  started = loop.run_in_executor(None, start)
  try:
    await asyncio.shield(started)
  except (asyncio.CancelledError, KeyboardInterrupt):
    await _outlast(started)
    await _outlast(loop.run_in_executor(None, undo))
    raise


async def _outlast(future: asyncio.Future[Any]) -> None:
  """Waits for `future` to complete across repeated cancellations.

  Suppresses `CancelledError` and `KeyboardInterrupt` until `future` is done
  and consumes any exception on `future` so asyncio does not log an unretrieved
  exception.

  Args:
    future: The future to await until completion.
  """
  while not future.done():
    with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
      await asyncio.wait({future})
  if not future.cancelled():
    # retrieved, so asyncio does not log it as never retrieved
    future.exception()


def _unread(why: str) -> bytes:
  """Formats a fallback log line when `docker logs` fails.

  Prevents Docker's own error message from being mistaken for output from the
  container.

  Args:
    why: Explanation of why `docker logs` failed.

  Returns:
    UTF-8 encoded fallback line.
  """
  return (
      "(the container's output could not be read: docker logs failed:"
      f" {why[:300]})\n".encode()
  )


def _last_line(data: bytes) -> str:
  """Extracts the last non-empty line from `data`.

  Args:
    data: Raw output bytes from `docker`.

  Returns:
    The last stripped line of text, or an empty string if `data` has no lines.
  """
  lines = data.decode("utf-8", errors="replace").strip().splitlines()
  return lines[-1].strip() if lines else ""


def _capped(data: bytes) -> bytes:
  """Truncates `data` to `LOG_BYTES` by keeping its head and tail."""
  half = LOG_BYTES // 2
  return (
      data
      if len(data) <= LOG_BYTES
      else data[:half] + b"\n... [cut] ...\n" + data[-half:]
  )


class Stack:
  """One task's serve container, egress proxy and internal network.

  `run` labels them with the run that started them (RUN_LABEL), so that
  sweep(run=...) can remove them however the run's trials ended.
  """

  def __init__(self, keep: bool = False, run: str | None = None):
    if run is not None and not RUN_RE.fullmatch(run):
      raise ValueError(
          f"run {run!r}: 1-64 characters of A-Z, a-z, 0-9, '_', '.' and '-'"
      )
    self.id = uuid.uuid4().hex[:12]
    self.owner = "keep" if keep else self.id
    self.run = run
    self.network = f"plebench-net-{self.id}"
    self.egress = f"plebench-egress-{self.id}"
    self.serve = f"plebench-serve-{self.id}"
    self.forward_port: int | None = None
    self.started_at: float | None = None
    self.egress_copy: pathlib.Path | None = None

  @property
  def url(self) -> str:
    return f"ws://127.0.0.1:{self.forward_port}/"

  def start(
      self,
      *,
      image: str,
      command: str,
      env_file: str | None,
      allow: list[str],
      exceptions: list[str] = (),
  ) -> None:
    """Starts the internal network, egress proxy, and serve container.

    Cleans up all created resources via `stop()` if any step fails. Passes only
    the environment variables from `env_file` plus fixed stack variables
    (`PORT`, `HOME`, `USER`, and proxy settings) to the serve container.

    Args:
      image: Serve container Docker image tag.
      command: Shell command to run inside the serve container.
      env_file: Optional path to an environment file for the serve container.
      allow: List of host patterns permitted by the egress proxy.
      exceptions: Host patterns excluded from `allow`.
    """
    try:
      self._start(image, command, env_file, allow, list(exceptions))
    except BaseException:
      self.stop()
      raise

  def _start(self, image, command, env_file, allow, exceptions) -> None:
    """Creates and starts the network, egress proxy, and serve container.

    Args:
      image: Serve container Docker image tag.
      command: Shell command to run inside the serve container.
      env_file: Optional path to an environment file for the serve container.
      allow: List of host patterns permitted by the egress proxy.
      exceptions: List of host patterns excluded from `allow`.

    Raises:
      InfraError: If any Docker command or egress readiness check fails.
    """
    _ensure_egress_base()
    label = f"{LABEL}={self.owner}"
    labels = ["--label", label] + (
        ["--label", f"{RUN_LABEL}={self.run}"] if self.run else []
    )
    docker("network", "create", "--internal", *labels, self.network)
    docker(
        "create",
        "--name",
        self.egress,
        *labels,
        "--network",
        EGRESS_NETWORK,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"127.0.0.1::{FORWARD_PORT}",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--memory",
        EGRESS_MEMORY,
        "--pids-limit",
        "256",
        *LOG_OPTS,
        "-v",
        f"{self._egress_script()}:/egress.py:ro,z",
        EGRESS_IMAGE,
        "python3",
        "-I",
        "/egress.py",
        "--allow",
        ",".join(allow),
        "--except",
        ",".join(exceptions),
        "--forward",
        f"serve:{PORT}",
        "--proxy-port",
        str(PROXY_PORT),
        "--forward-port",
        str(FORWARD_PORT),
    )
    docker("network", "connect", "--alias", "egress", self.network, self.egress)
    docker("start", self.egress)
    mapped = docker("port", self.egress, f"{FORWARD_PORT}/tcp").stdout.split()
    try:
      self.forward_port = int(mapped[0].rsplit(":", 1)[1])
    except (IndexError, ValueError):
      raise InfraError(
          f"the egress proxy published no port ({mapped})"
      ) from None
    self._wait_for_egress()
    proxy = f"http://egress:{PROXY_PORT}"
    args = [
        "run",
        "-d",
        "--name",
        self.serve,
        *labels,
        "--network",
        self.network,
        "--network-alias",
        "serve",
        "--user",
        "1000:1000",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=1g,mode=1777",
        "--tmpfs",
        f"{HOME}:rw,exec,nosuid,size=1g,uid=1000,gid=1000",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        CPUS,
        "--memory",
        MEMORY,
        "--pids-limit",
        PIDS,
        *LOG_OPTS,
    ]
    if env_file:
      args += ["--env-file", env_file]
    fixed = {
        "PORT": str(PORT),
        "HOME": HOME,
        "USER": "harness",
        "HTTPS_PROXY": proxy,
        "HTTP_PROXY": proxy,
        "ALL_PROXY": proxy,
        "https_proxy": proxy,
        "http_proxy": proxy,
        "all_proxy": proxy,
        "NO_PROXY": "",
        "no_proxy": "",
        "NODE_USE_ENV_PROXY": "1",
    }
    for k, v in fixed.items():
      args += ["-e", f"{k}={v}"]
    docker(*args, image, "bash", "-c", command)
    self.started_at = time.monotonic()

  def _wait_for_egress(self, timeout: float = EGRESS_READY_SEC) -> None:
    """Waits until the egress proxy logs that its ports are bound.

    Streams `docker logs --follow` until `"kind": "start"` appears so the
    serve container's first request finds the proxy listening.

    Args:
      timeout: Maximum seconds to wait for the proxy to start listening.

    Raises:
      InfraError: If `docker logs` fails, the proxy exits early, or `timeout`
        expires.
    """
    try:
      proc = subprocess.Popen(
          ["docker", "logs", "--follow", self.egress],
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=docker_env(),
      )
    except OSError as exc:
      raise InfraError(f"docker logs: {exc}") from exc
    deadline, seen = time.monotonic() + timeout, bytearray()
    try:
      while b'"kind": "start"' not in seen:
        left = deadline - time.monotonic()
        if left <= 0:
          raise InfraError(
              f"the egress proxy was not listening within {timeout:.0f} s"
          )
        if not select.select([proc.stdout], [], [], left)[0]:
          continue
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:  # the proxy exited (or its logs could not be read)
          raise InfraError(
              "the egress proxy exited before listening: "
              f"{seen.decode(errors='replace').strip()[-300:]}"
          )
        seen += chunk
        del seen[:-LOG_BYTES]
    finally:
      proc.kill()
      proc.wait()

  def _egress_script(self) -> pathlib.Path:
    """Writes a world-readable (`0644`) copy of `egress.py` in `cache_root()`.

    Allows the egress container (running as UID 65534) to read `egress.py` even
    when the repository checkout has restrictive permissions (`0600` under
    umask `077`).

    Returns:
      Path to the copied `egress.py` file.

    Raises:
      InfraError: If the copy cannot be written.
    """
    self.egress_copy = egress_root() / (self.run or "-") / self.id
    try:
      # another kit's _sweep_egress_copies may remove the run's folder
      # just made
      for _ in range(3):
        with contextlib.suppress(FileNotFoundError):
          self.egress_copy.mkdir(parents=True, exist_ok=True)
          break
      script = self.egress_copy / "egress.py"
      shutil.copyfile(EGRESS_SCRIPT, script)
      os.chmod(script, 0o644)
    except OSError as exc:
      raise InfraError(
          f"the egress proxy's copy of egress.py could not be written: {exc}"
      ) from exc
    return script

  def _container(self, which: str) -> str:
    """The serve container's name, or the egress proxy's for which="egress"."""
    return self.serve if which == "serve" else self.egress

  def running(self, which: str = "serve") -> bool | None:
    """Checks whether the `which` container is currently running.

    Args:
      which: Either `"serve"` or `"egress"`.

    Returns:
      True if running, False if Docker reports it has stopped, or None if
      Docker could not inspect the container.
    """
    try:
      return bool(self.state(which).get("Running"))
    except InfraError:
      return None

  def gone(self, which: str = "serve") -> bool:
    """Checks whether the `which` container has been removed outside the kit.

    Args:
      which: Either `"serve"` or `"egress"`.

    Returns:
      True if Docker explicitly reports no such container; False if it exists
      or if Docker does not answer.
    """
    try:
      r = docker(
          "container",
          "inspect",
          "-f",
          "{{.Id}}",
          self._container(which),
          timeout=30,
          check=False,
      )
    except InfraError:
      return False
    return r.returncode != 0 and bool(NO_SUCH_CONTAINER.search(r.stderr))

  def state(self, which: str = "serve") -> dict[str, Any]:
    """Returns Docker's `.State` dictionary for the `which` container.

    Args:
      which: Either `"serve"` or `"egress"`.

    Returns:
      The parsed `.State` dictionary from `docker inspect`.

    Raises:
      InfraError: If `docker inspect` fails, times out, or the container does
        not exist.
    """
    r = docker(
        "inspect",
        "-f",
        "{{json .State}}",
        self._container(which),
        timeout=30,
        check=False,
    )
    try:
      state = json.loads(r.stdout) if r.returncode == 0 else None
    except ValueError:
      state = None
    if not isinstance(state, dict):
      raise InfraError(
          f"docker inspect of the {which} container failed:"
          f" {(r.stderr or r.stdout).strip()[-300:]}"
      )
    return state

  def logs(self, which: str = "serve") -> bytes:
    """Fetches combined stdout and stderr logs for the `which` container.

    Preserves the interleaved ordering replayed by `docker logs` and caps the
    output to `LOG_BYTES`. Returns a labelled fallback line (`_unread`) if
    `docker logs` fails.

    Args:
      which: Either `"serve"` or `"egress"`.

    Returns:
      Capped container log bytes, or a fallback error line.
    """
    try:
      r = subprocess.run(
          ["docker", "logs", self._container(which)],
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          timeout=60,
          env=docker_env(),
          check=False,
      )
    except subprocess.TimeoutExpired:
      return _unread("it did not answer within 60 s")
    except OSError as exc:
      return _unread(str(exc))
    if r.returncode != 0:
      return _unread(_last_line(r.stdout) or f"exit {r.returncode}")
    return _capped(r.stdout)

  def exec(
      self, *argv: str, timeout: float = 120
  ) -> subprocess.CompletedProcess[str]:
    return docker("exec", self.serve, *argv, timeout=timeout, check=False)

  def stop(self) -> None:
    """Removes the serve container, egress proxy, network, and egress script.

    Attempts every removal step even if an earlier one fails or times out, and
    never raises. Any resource still left behind is later cleaned up by
    `sweep()`.
    """
    for args in (
        ("rm", "-f", self.serve),
        ("rm", "-f", self.egress),
        ("network", "rm", self.network),
    ):
      with contextlib.suppress(InfraError):
        docker(*args, timeout=60, check=False)
    if self.egress_copy is not None:
      shutil.rmtree(self.egress_copy, ignore_errors=True)
      self.egress_copy = None


def exit_wording(state: dict[str, Any], memory: str) -> str:
  """Describes how a stopped container exited from its `.State` dictionary.

  Args:
    state: Container `.State` mapping from `Stack.state()`.
    memory: Memory limit string (for example, `"4g"`) for OOM messages.

  Returns:
    Human-readable exit description (OOM-killed or exit code).
  """
  if state.get("OOMKilled"):
    return f"killed for using more than {memory} of memory"
  return f"exited with status {state.get('ExitCode')}"


def output_tail(data: bytes, lines: int = 15) -> str:
  """Returns the last `lines` of container output, indented for messages."""
  tail = data.decode(errors="replace").strip().splitlines()[-lines:]
  return "    " + "\n    ".join(tail or ["(nothing)"])


def exit_note(st: Stack) -> str | None:
  """Formats an exit note if `st`'s serve container has stopped.

  Used to enrich the relay's error message when a server accepts no connection.
  Because `docker inspect` blocks, async callers run this in a worker thread.

  Args:
    st: The `Stack` whose serve container should be inspected.

  Returns:
    `"the server <how it exited>"` if the serve container stopped, or None if
    it is still running or Docker could not inspect it.
  """
  try:
    state = st.state()
  except InfraError:
    return None
  return (
      None
      if state.get("Running")
      else f"the server {exit_wording(state, MEMORY)}"
  )


def write_env_file(path: pathlib.Path, values: dict[str, str]) -> None:
  """A Docker env file readable only by this user, created anew."""
  bad = sorted(k for k, v in values.items() if any(c in v for c in "\n\r\x00"))
  if bad:
    raise ValueError(
        f"{', '.join(bad)}: a value with a line break cannot reach the serve"
        " container"
    )
  path.unlink(missing_ok=True)
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  # docker reads it as UTF-8, whatever the locale
  with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.writelines(f"{k}={v}\n" for k, v in values.items())


_RFC3339 = re.compile(
    r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)"
)


def created_epoch(created: str) -> float | None:
  """Parses a `docker inspect` creation timestamp into seconds since the epoch.

  Accepts either Unix epoch seconds (`{{.Created.Unix}}`, used for networks) or
  an RFC 3339 timestamp (`{{.Created}}`, used for containers and images).

  Args:
    created: Timestamp string printed by `docker inspect`.

  Returns:
    Seconds since the Unix epoch, or None if `created` cannot be parsed.
  """
  if created.isdigit():
    return float(created)
  m = _RFC3339.fullmatch(created)
  if not m:
    return None
  stamp = (
      datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
      .replace(tzinfo=datetime.timezone.utc)
      .timestamp()
  )
  zone = m.group(3)
  if zone == "Z":
    offset = 0
  else:
    sign = 1 if zone[0] == "+" else -1
    offset = sign * (int(zone[1:3]) * 3600 + int(zone[-2:]) * 60)
  return stamp + float("0." + (m.group(2) or "0")) - offset


def sweep(*, run: str | None = None) -> int:
  """Removes stale or run-specific serve stacks and left-behind task projects.

  When `run` is None, removes stacks and Harbor task projects older than
  `SWEEP_AGE_SEC` (excluding `kit up` stacks and shared networks). When `run`
  is given, removes every stack belonging to that run regardless of age.

  Args:
    run: Optional run identifier (`RUN_LABEL`) to sweep immediately.

  Returns:
    Total number of Docker containers and networks removed.

  Raises:
    ValueError: If `run` is not a valid run identifier.
    InfraError: If Docker fails to list or inspect containers or networks.
  """
  if run is not None and not RUN_RE.fullmatch(run):
    raise ValueError(f"run {run!r}: not a run id")
  removed, now, active = 0, time.time(), set()
  selector = f"label={RUN_LABEL}={run}" if run else f"label={LABEL}"
  for kind, listing, labels, created in (
      ("container", ["ps", "-a"], ".Config.Labels", "{{.Created}}"),
      ("network", ["network", "ls"], ".Labels", "{{.Created.Unix}}"),
  ):
    oids = docker(
        *listing, "--filter", selector, "--format", "{{.ID}}"
    ).stdout.split()
    if not oids:
      continue
    # one docker call for them all; each line carries its object's id, so one
    # removed meanwhile is just missing
    r = docker(
        kind,
        "inspect",
        "-f",
        f'{{{{.Id}}}}|{{{{.Name}}}}|{created}|{{{{index {labels} "{LABEL}"}}}}',
        *oids,
        check=False,
    )
    if r.returncode != 0 and not docker_answers():
      raise InfraError(f"docker {kind} inspect: {r.stderr.strip()[-300:]}")
    for line in r.stdout.splitlines():
      oid, name, stamp, owner = (line.strip().split("|", 3) + ["", "", ""])[:4]
      if not oid:
        continue
      # None for a time this docker prints otherwise, not taken as old
      born = created_epoch(stamp)
      if run or (
          owner
          and owner not in ("shared", "keep")
          and born is not None
          and now - born > SWEEP_AGE_SEC
      ):
        rm = (
            ["rm", "-f", oid] if kind == "container" else ["network", "rm", oid]
        )
        if docker(*rm, check=False).returncode == 0:
          removed += 1
          continue
      # a stack still there keeps its copy of egress.py
      active.add(name.rsplit("-", 1)[-1])
  if run is None:
    removed += _sweep_task_projects(now)
  _sweep_egress_copies(run, active, now)
  return removed


def _sweep_task_projects(now: float) -> int:
  """Removes Harbor task Compose projects older than `SWEEP_AGE_SEC`.

  Identifies abandoned task containers carrying both `RUN_LABEL` and
  `COMPOSE_PROJECT` and removes their entire Compose projects via
  `remove_projects()`.

  Args:
    now: Current Unix epoch timestamp in seconds.

  Returns:
    Number of Docker containers and networks removed.

  Raises:
    InfraError: If Docker fails to list or inspect task containers.
  """
  oids = docker(
      "ps",
      "-a",
      "--filter",
      f"label={RUN_LABEL}",
      "--filter",
      f"label={COMPOSE_PROJECT}",
      "--format",
      "{{.ID}}",
  ).stdout.split()
  if not oids:
    return 0
  r = docker(
      "container",
      "inspect",
      "-f",
      f'{{{{.Created}}}}|{{{{index .Config.Labels "{COMPOSE_PROJECT}"}}}}',
      *oids,
      check=False,
  )
  if r.returncode != 0 and not docker_answers():
    raise InfraError(f"docker container inspect: {r.stderr.strip()[-300:]}")
  old = set()
  for line in r.stdout.splitlines():
    stamp, _, project = line.strip().partition("|")
    # None for a time this docker prints otherwise, not taken as old
    born = created_epoch(stamp)
    if project and born is not None and now - born > SWEEP_AGE_SEC:
      old.add(project)
  return remove_projects(old)


def remove_projects(projects, timeout: float = 120) -> int:
  """Removes all containers and networks belonging to `projects`.

  Args:
    projects: Iterable of Docker Compose project names (`COMPOSE_PROJECT`).
    timeout: Timeout in seconds for each Docker CLI call.

  Returns:
    Total number of containers and networks removed.

  Raises:
    InfraError: If Docker fails to list project resources.
  """
  removed = 0
  for listing, rm in (
      (("ps", "-a"), ("rm", "-f")),
      (("network", "ls"), ("network", "rm")),
  ):
    oids: list[str] = []
    for name in sorted(projects):
      out = docker(
          *listing,
          "--filter",
          f"label={COMPOSE_PROJECT}={name}",
          "--format",
          "{{.ID}}",
          timeout=timeout,
      ).stdout
      oids.extend(out.split())
    if oids:
      removed += len(
          docker(*rm, *oids, timeout=timeout, check=False).stdout.split()
      )
  return removed


def _sweep_egress_copies(run: str | None, active: set[str], now: float) -> None:
  """Removes stale per-stack `egress.py` copies under `egress_root()`.

  Args:
    run: Optional run ID whose egress copies should be removed immediately.
    active: Set of stack IDs that still have live containers or networks.
    now: Current Unix epoch timestamp in seconds.
  """
  root = egress_root()
  groups = (
      [root / run] if run else (sorted(root.iterdir()) if root.is_dir() else [])
  )
  for group in groups:
    try:
      # read before the removals below touch it
      idle = run or now - group.stat().st_mtime > SWEEP_AGE_SEC
      for copy in sorted(group.iterdir()):
        if copy.name not in active and (
            run or now - copy.stat().st_mtime > SWEEP_AGE_SEC
        ):
          shutil.rmtree(copy, ignore_errors=True)
      if idle:
        group.rmdir()  # only once it is empty
    except OSError:
      pass
