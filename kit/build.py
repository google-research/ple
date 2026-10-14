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

"""Builds a submission into a bundle and serve image as the server does.

Submission files are copied into a clean builder container (with internet access
and no task data), `install` runs there as root, and `/opt/harness` is packed
into the bundle used to create the serve image. Docker runs `linux/amd64` images
on any host machine (emulated on Apple Silicon).
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable
import uuid
import zlib

from kit import artifacts
from kit import spec
from kit import stack

BUILDER_DIR = pathlib.Path(__file__).with_name("builder")
BUILD_TIMEOUT_SEC = 1800
# one anonymous fetch of a submodule's commit (submodule_problems)
FETCH_SEC = 120
# git's message for a fetch that never reached the repository: no network, a
# proxy or TLS failure, a server error, no answer in time. Every curl failure
# reads "unable to access", and of those only 401, 403 and 404 are the
# repository's.
UNREACHED = re.compile(
    r"unable to access (?!.*error: 40[134]\b)|^no answer within"
)
MAX_SOURCE_BYTES = 1 << 30
MAX_BUNDLE_BYTES = 2 << 30
MAX_UNPACKED_BYTES = 8 << 30
MAX_MEMBERS = 200_000
# characters in a bundle member's name or link target, as on the server
MAX_NAME = 1024
# symlinks one path may pass through, a symlink's own included: Linux's limit
MAX_LINK_HOPS = 40
# path characters _resolve may look up for the whole archive
LINK_CHECK_BUDGET = 50_000_000
ROOT = "opt/harness"
KEY_FILE = "bundle.key"
# beside a staging directory the kit made: only it, or an empty one, is replaced
STAGED_MARK = ".staged-by-kit"
# the server's builder limits
MAX_CPUS, MAX_MEMORY, MAX_PIDS = 8, 16 << 30, 4096
# the server's too: no file the install writes may be larger (RLIMIT_FSIZE)
MAX_FILE_BYTES = 8 << 30
# The builder's command runs before build.sh: it prints a line showing the
# container started, then checks its mounted inputs. A folder the Docker daemon
# cannot see (outside Docker Desktop's file sharing, or a Colima or Lima VM's
# mounts) is mounted as an empty directory, so a missing input is a Docker
# setting to fix, not a failed install. No line at all means Docker could not
# start the container.
BUILDER_STARTED = "--- builder started"
MOUNT_MISSING = "@@plebench-mount-missing@@"
BUILDER_COMMAND = (
    f"echo '{BUILDER_STARTED}' >&2; "
    'for f in /mnt/build.sh /mnt/install.sh; do [ -f "$f" ] || { echo'
    f' "{MOUNT_MISSING} $f" >&2; '
    "exit 1; }; done; "
    f"[ -e /mnt/src/{spec.FILE} ] || [ -L /mnt/src/{spec.FILE} ] || "
    f'{{ echo "{MOUNT_MISSING} /mnt/src" >&2; exit 1; }}; '
    "exec bash /mnt/build.sh"
)
SHARING = (
    "the Docker daemon must see the work directory: Docker Desktop shares your"
    " home directory by default (Settings > Resources > File sharing for"
    " others), a Colima or Lima VM sees only what it mounts; or choose a --work"
    " under your home directory"
)


class BuildError(Exception):  # pylint: disable=g-bad-exception-name
  """A build failure the participant can fix; the message explains how."""


@dataclasses.dataclass(frozen=True)
class Source:
  """Staged submission source files and metadata."""

  path: pathlib.Path
  files: int
  uncommitted: list[str]
  # of the staged files' paths, modes and contents (or link targets), and
  # _version
  key: str
  # each staged regular file's sha256 by its path, as staging took it (what kit
  # check's key scan compares)
  digests: dict[str, str] = dataclasses.field(
      default_factory=dict, compare=False, repr=False
  )
  # staged files a commit holds as Git LFS pointers (_git_files)
  lfs: list[str] = dataclasses.field(default_factory=list)
  # left-out folders holding files, besides the reviewed one
  left_out: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Bundle:
  """Packed harness bundle archive metadata."""

  path: pathlib.Path
  size: int
  seconds: float
  sha256: str = ""  # of the packed bundle, which names its serve image


@dataclasses.dataclass(frozen=True)
class Prepared:
  """Prepared bundle and serve image for a staged submission."""

  bundle: Bundle
  # the serve image: the builder image with the bundle at /opt/harness
  image: str
  # the bundle of an earlier build, whose inputs have not changed since
  reused: bool


def work_dir(
    name: str,
    work: str | pathlib.Path | None = None,
    src: str | pathlib.Path | None = None,
) -> pathlib.Path:
  """Returns the work directory for an entry's staged files, bundle, and logs.

  Uses `work` as an absolute path (Docker treats relative mounts as volume
  names), or defaults to `~/.cache/plebench/<name>` (`stack.cache_root`). When
  `src` is provided, a work directory equal to or inside the submission is
  rejected before creation so nothing (such as `.lock` or `check.json`) is ever
  written into the submission.

  Args:
    name: Submission entry name.
    work: Optional work directory path override.
    src: Optional submission source directory path.

  Returns:
    The resolved work directory path.
  """
  path = pathlib.Path(work).resolve() if work else stack.cache_root() / name
  if src is not None:
    _refuse_inside(pathlib.Path(src), path)
  path.mkdir(parents=True, exist_ok=True)
  return path


def _refuse_inside(src: pathlib.Path, work: pathlib.Path) -> None:
  src, work = src.resolve(), work.resolve()
  if work == src or src in work.parents:
    raise BuildError(
        f"the work directory {work} is inside the submission {src}: choose a"
        " --work outside it (the default is ~/.cache/plebench/<name>)"
    )


@contextlib.contextmanager
def locked(work: pathlib.Path):
  """Acquires an exclusive lock on an entry's work directory during builds.

  A second kit process on the same entry (e.g., `kit check` while `eval/run.py`
  builds, or two concurrent eval runs) waits here and then reuses the first
  process's bundle when nothing changed instead of replacing files in use.

  Args:
    work: Work directory to lock.
  """
  work.mkdir(parents=True, exist_ok=True)
  fd = os.open(work / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
  try:
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      print(
          f"another kit run is building in {work}; waiting for it to finish",
          flush=True,
      )
      fcntl.flock(fd, fcntl.LOCK_EX)
    yield
  finally:
    os.close(fd)  # closing the descriptor releases the lock


def remote_daemon() -> str | None:
  """Returns the Docker endpoint URL if reached over the network, else None.

  The kit bind-mounts local files into its containers and connects to the serve
  container at `127.0.0.1`, so it requires a local Docker daemon rather than a
  `tcp://`, `ssh://`, or `http(s)://` endpoint.

  Returns:
    The remote Docker endpoint string, or None if the daemon is local.
  """
  host = os.environ.get("DOCKER_HOST", "")
  if not host:
    try:
      r = stack.docker(
          "context",
          "inspect",
          "--format",
          "{{.Endpoints.docker.Host}}",
          timeout=30,
          check=False,
      )
    except stack.InfraError:
      # docker missing or not answering: the next docker command says so
      return None
    host = r.stdout.strip() if r.returncode == 0 else ""
  return (
      host
      if host.split("://", 1)[0].lower() in ("tcp", "ssh", "http", "https")
      else None
  )


def require_local_daemon() -> None:
  """Verifies that the active Docker daemon is local.

  Raises:
    stack.InfraError: If the Docker daemon is remote (`remote_daemon`).
  """
  host = remote_daemon()
  if host:
    raise stack.InfraError(
        f"the kit needs a Docker daemon on this machine, and {host} is remote"
        " (the kit bind-mounts local files and connects to 127.0.0.1): unset"
        " DOCKER_HOST or switch back with `docker context use default`"
    )


# the Docker daemon's (CPUs, bytes of memory), once it answered
_resources: tuple[int, int] | None = None


def daemon_resources() -> tuple[int, int] | None:
  """Returns the `(cpus, memory_bytes)` reported by the Docker daemon.

  Queries the daemon until it answers and caches the result for the process,
  returning None while it does not answer. Host resources are never substituted
  because a VM (such as Docker Desktop) often has fewer CPUs than the host and
  `docker run --cpus` fails if requested CPUs exceed the daemon's count.

  Returns:
    A `(cpus, memory_bytes)` tuple, or None if the daemon did not respond.
  """
  global _resources
  if _resources is None:
    try:
      ncpu, mem = (
          int(x)
          for x in stack.docker(
              "info",
              "--format",
              "{{.NCPU}} {{.MemTotal}}",
              timeout=30,
              check=False,
          ).stdout.split()
      )
    except (stack.InfraError, ValueError):
      return None
    _resources = ncpu, mem
  return _resources


def limits() -> list[str]:
  """Returns Docker CLI resource limit flags for the builder container.

  Applies the evaluation server's limits (8 CPUs, 16 GiB RAM, 4096 processes,
  8 GiB file size), capping CPUs and memory at what the local Docker daemon
  provides because `dockerd` rejects `--cpus` above its own count. When the
  daemon does not report its resources, `--cpus` is omitted and the server's
  memory limit is used.

  Returns:
    A list of `docker run` resource limit flags.
  """
  known = daemon_resources()
  rest = ["--pids-limit", str(MAX_PIDS), "--ulimit", f"fsize={MAX_FILE_BYTES}"]
  if known is None:
    return ["--memory", str(MAX_MEMORY), *rest]
  ncpu, mem = known
  return [
      "--cpus",
      str(max(1, min(MAX_CPUS, ncpu))),
      "--memory",
      str(max(1 << 30, min(MAX_MEMORY, mem))),
      *rest,
  ]


def builder_image(log: pathlib.Path) -> tuple[str, str]:
  """Ensures the builder image is built and returns its `(tag, image_id)`.

  Builds the image from `kit/builder` once per Dockerfile and platform.

  Args:
    log: Path to the build log file.

  Returns:
    A `(tag, image_id)` tuple for the builder image.

  Raises:
    stack.InfraError: If building the builder image times out or fails.
  """
  env = stack.build_env()
  recipe = (BUILDER_DIR / "Dockerfile").read_bytes() + env[
      "DOCKER_DEFAULT_PLATFORM"
  ].encode()
  tag = f"plebench-builder:{hashlib.sha256(recipe).hexdigest()[:16]}"
  found = stack.image_id(tag)
  if not found:
    with log.open("a") as fh:
      try:
        r = subprocess.run(
            ["docker", "build", "-q", "-t", tag, str(BUILDER_DIR)],
            stdout=fh,
            stderr=fh,
            timeout=BUILD_TIMEOUT_SEC,
            env=env,
            check=False,
        )
      except subprocess.TimeoutExpired:
        raise stack.InfraError(
            "building the builder image did not finish within"
            f" {BUILD_TIMEOUT_SEC // 60} minutes: its downloads are slow on"
            " this network, or linux/amd64 is emulated on this machine; run it"
            f" again, and see {log}"
        ) from None
      except OSError as exc:
        raise stack.InfraError(f"docker build: {exc}") from exc
    if r.returncode != 0:
      raise stack.InfraError(f"building the builder image failed; see {log}")
    found = stack.image_id(tag)
    # a build that exited 0 but left no image (a builder that loads nothing)
    if not found:
      raise stack.InfraError(
          f"building the builder image left no image {tag}; see {log}"
      )
  return tag, found


def _git(
    src: pathlib.Path, *args: str, errors: list[str] | None = None
) -> str | None:
  """Runs git in `src` and returns its decoded output, or None on failure.

  Uses `os.fsdecode` so paths that are not valid UTF-8 still open the same
  filesystem entries and round-trip through `os.fsencode`.

  Args:
    src: Directory in which to run git.
    *args: Arguments passed to git.
    errors: Optional list to append stderr messages to on failure.

  Returns:
    The decoded stdout string, or None if the git command failed.
  """
  out = artifacts.git_output(src, *args, errors=errors)
  return None if out is None else os.fsdecode(out)


def _unreadable(src: pathlib.Path, rel: str, why: str) -> BuildError:
  """Creates a `BuildError` for an unreadable harness file or folder.

  The build stops when a file or folder cannot be read locally because the
  bundle would silently omit it while the server's build of the commit would
  include it.

  Args:
    src: Submission root directory.
    rel: Path relative to `src` that could not be read.
    why: Description of the read error.

  Returns:
    A `BuildError` describing how to fix the unreadable path.
  """
  return BuildError(
      artifacts.unreadable_stop(
          src, [(rel, why)], "the build would leave it out"
      )
  )


def _submodules(git: Callable[..., bytes | None]) -> list[tuple[str, str]]:
  """Returns `(state, relative_path)` for each submodule under the entry.

  Queries `git` (the strict runner from `artifacts.work_tree`): `' '` when
  checked out at the recorded commit, `'+'` at another commit, and `'-'` when
  not checked out. Only call this when the entry has paths in the index, since
  `-- .` matching nothing is an error.

  Args:
    git: Strict git runner callable for the work tree.

  Returns:
    A list of `(state, relative_path)` tuples for each submodule.
  """
  out = os.fsdecode(git("submodule", "status", "--recursive", "--", ".") or b"")
  found = []
  for line in out.split("\n"):
    m = re.fullmatch(r"([ +\-U])[0-9a-f]{40,64} (.*?)(?: \(.*\))?", line)
    if m:
      found.append((m.group(1), m.group(2)))
  return found


def _git_files(
    src: pathlib.Path,
) -> tuple[list[str], list[str], list[str], dict[str, bool]] | None:
  """Lists tracked/untracked files, changes, LFS pointers, and executable modes.

  Inspects git's index for `src` (including checked-out submodules, since the
  server runs `git submodule update --init --recursive`). Because the server
  clones without Git LFS, index entries that are LFS pointers are reported;
  repositories with `core.fileMode=false` take executable bits from the index.

  Args:
    src: Submission directory path.

  Returns:
    A tuple `(files_to_build, uncommitted_files, lfs_pointers, executable_map)`,
    or None if `src` is outside a git work tree.

  Raises:
    BuildError: If a submodule is not checked out, symbolic links were checked
      out as plain files (`core.symlinks=false`), files are unreadable, or any
      git command fails.
  """
  git, refused = artifacts.work_tree(src)
  if refused:
    raise BuildError(refused)
  if git is None:
    return None

  def files(*args: str, warnings: list[str] | None = None) -> list[str]:
    return [
        p
        for p in os.fsdecode(git(*args, warnings=warnings) or b"").split("\0")
        if p
    ]

  warned: list[str] = []
  try:
    tracked = files("ls-files", "-z", "--recurse-submodules", "--", ".")
    untracked = files(
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--",
        ".",
        warnings=warned,
    )
    # the index's own paths: a submodule that was never fetched has none of its
    # files listed above
    subs = (
        _submodules(git)
        if tracked or files("ls-files", "-z", "--", ".")
        else []
    )
    repos = [("", git)] + [
        (f"{p}/", artifacts.GitRunner(src / p, ""))
        for state, p in subs
        if state != "-"
    ]
    pointers, links = [], []
    # each repository's 100755 files, or None where git reads the bit from
    # the disk
    marked = {}
    for top, runner in repos:
      # every file's index version, whatever the attributes say now: a removed
      # Git LFS rule leaves its pointers in the index while the disk holds the
      # whole files
      pointers += [top + p for p in artifacts.index_state(runner, ["."], {})[0]]
      off = {
          k
          for k in ("fileMode", "symlinks")
          if runner("config", "--bool", f"core.{k}", ok=(0, 1)) == b"false\n"
      }
      if off:
        raw_staged = (runner("ls-files", "-s", "-z", "--", ".") or b"").split(
            b"\0"
        )
        staged = [
            (rec[:6], top + os.fsdecode(rec.partition(b"\t")[2]))
            for rec in raw_staged
            if rec
        ]
      else:
        staged = []
      marked[top] = (
          {p for mode, p in staged if mode == b"100755"}
          if "fileMode" in off
          else None
      )
      links += [
          p for mode, p in staged if mode == b"120000" and "symlinks" in off
      ]
    executable = {}
    # a file belongs to the deepest repository holding it
    for rel in tracked + untracked:
      index = marked[
          max((top for top in marked if rel.startswith(top)), key=len)
      ]
      if index is not None:
        executable[rel] = rel in index
    # git skipped what these folders hold
    blocked = artifacts.could_not_open(warned, src, git)
    # before the first commit, every file is new (`git rev-parse --verify` ends
    # with code 1 when there is no HEAD)
    head = git("rev-parse", "--verify", "-q", "HEAD", ok=(0, 1))
    changed = (
        files("diff", "--name-only", "-z", "--relative", "HEAD", "--", ".")
        if head
        else tracked
    )
  except artifacts.GitFailed as exc:
    raise BuildError(artifacts.git_stop(src, exc.said, exc.kind)) from None
  if links:
    raise BuildError(
        f"core.symlinks is false, so git checked out {artifacts.shown(links)}"
        " as plain files where the commit holds symbolic links: run `git"
        " config core.symlinks true` in each repository holding one, then `git"
        " checkout -- <link>` for each"
    )
  if blocked:
    them = "it" if len(blocked) == 1 else "them"
    raise BuildError(
        artifacts.unreadable_stop(
            src, blocked, f"the build would leave {them} out"
        )
    )
  missing = [path for state, path in subs if state == "-"]
  if missing:
    raise BuildError(
        f"the submodule {', '.join(missing)} is not checked out here, but the"
        " server's build fetches it: run `git submodule update --init"
        " --recursive`"
    )
  # the server checks out the recorded commit
  moved = [path for state, path in subs if state == "+"]
  return (
      tracked + untracked,
      sorted(set(untracked) | set(changed) | set(moved)),
      pointers,
      executable,
  )


def _gitlinks(repo: pathlib.Path) -> list[tuple[str, str]]:
  """Returns `(commit, path)` for each submodule in the index under `repo`."""
  out = _git(repo, "ls-files", "-s", "-z") or ""
  return [
      (info.split()[1], path)
      for info, _, path in (rec.partition("\t") for rec in out.split("\0"))
      if info.startswith("160000 ")
  ]


def _relative_url(base: str, url: str) -> str:
  """Resolves a relative submodule `url` (`./` or `../`) against `base`.

  Args:
    base: Base remote URL of the parent repository.
    url: Relative submodule URL to resolve.

  Returns:
    The resolved URL string.
  """
  base = base.rstrip("/")
  while url.startswith(("./", "../")):
    if url.startswith("../"):
      base = base.rsplit("/", 1)[0]
    url = url.split("/", 1)[1]
  return f"{base}/{url}".rstrip("/")


def _fetch_problem(url: str, commit: str) -> str | None:
  """Checks whether `commit` can be fetched anonymously from `url`.

  Runs `git fetch --depth 1` in an empty repository with a temporary `HOME` so
  no local git configuration (credential helpers, URL rewrites, `~/.netrc`) is
  used, matching the evaluation server's anonymous fetch.

  Args:
    url: Submodule repository URL.
    commit: Commit hash to fetch.

  Returns:
    The last stderr line from git when the fetch fails, or None on success.
  """
  with tempfile.TemporaryDirectory(prefix="plebench-fetch-") as home:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        HOME=home,
        XDG_CONFIG_HOME=home,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        SSH_ASKPASS="",
        LC_ALL="C",
    )
    try:
      for args in (
          ["init", "-q"],
          ["fetch", "-q", "--depth", "1", "--filter=tree:0", "--", url, commit],
      ):
        p = subprocess.run(
            ["git", "-c", "credential.helper=", "-C", home, *args],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=FETCH_SEC,
            check=False,
        )
        if p.returncode:
          return (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
    except subprocess.TimeoutExpired:
      return f"no answer within {FETCH_SEC} s"
    except OSError as exc:
      return str(exc)
  return None


def _without_userinfo(text: str) -> str:
  """Strips user and password credentials from all URLs in `text`.

  Args:
    text: Input string potentially containing URLs with userinfo.

  Returns:
    `text` with URL userinfo (`user:pass@`) removed.
  """
  return re.sub(r"(?<=://)[^\s/@]*@", "", text)


def submodule_problems(
    src: pathlib.Path, origin: str | None = None, top: str = ""
) -> list[tuple[str, str]]:
  """Checks whether all submodules under `src` can be fetched by the server.

  The evaluation server runs `git submodule update --init --recursive --depth 1`
  without credentials or git config, so each submodule must have an `https://`
  URL whose recorded commit can be fetched anonymously. Relative URLs are
  resolved against the parent repository's remote (`origin`). Submodules in
  excluded artifact directories (`artifacts.bundle_excluded`) are skipped, and
  credentials are stripped from all reported URLs.

  Args:
    src: Repository or submodule directory path.
    origin: Optional remote origin URL of `src` for resolving relative URLs.
    top: Path prefix for nested submodules.

  Returns:
    A list of `(status, message)` tuples where `status` is `"fail"` or `"warn"`.
  """
  if origin is None:
    remote = (_git(src, "remote", "get-url", "origin") or "").strip()
    origin = _without_userinfo(
        re.sub(
            r"^(?:ssh://)?git@github\.com[:/]", "https://github.com/", remote
        )
    )
  conf = (
      _git(
          src,
          "config",
          "-z",
          "-f",
          ".gitmodules",
          "--get-regexp",
          r"^submodule\..*\.(path|url)$",
      )
      or ""
  )
  pairs = dict(rec.split("\n", 1) for rec in conf.split("\0") if "\n" in rec)
  urls = {
      path: pairs.get(key.removesuffix("path") + "url", "")
      for key, path in pairs.items()
      if key.endswith(".path")
  }
  problems = []
  for commit, path in _gitlinks(src):
    if not top and artifacts.bundle_excluded(path):
      continue
    name, url = top + path, urls.get(path, "")
    if url.startswith(("./", "../")):
      if not origin:
        problems.append((
            "fail",
            (
                f"the submodule {name} has the relative URL {url}, which git"
                " resolves against the repository's origin remote, and there"
                " is none: add the GitHub repository you submit (`git remote"
                " add origin https://github.com/<owner>/<repository>`)"
            ),
        ))
        continue
      url = _relative_url(origin, url)
    if not url.startswith("https://"):
      problems.append((
          "fail",
          (
              f"the submodule {name} has the URL"
              f" {url or '(none in .gitmodules)'}: the server fetches"
              " submodules over https:// without credentials"
          ),
      ))
    elif (said := _fetch_problem(url, commit)) and UNREACHED.search(said):
      problems.append((
          "warn",
          (
              f"could not check that the server can fetch the submodule {name}"
              f" (commit {commit[:12]} of {url}): git, run without this"
              " machine's git config as on the server, could not reach it (git"
              f" says: {said})"
          ),
      ))
    elif said:
      problems.append((
          "fail",
          (
              f"the server could not fetch the submodule {name} (commit"
              f" {commit[:12]} of {url}) without credentials (git says:"
              f" {said}): make that repository public and push the commit"
          ),
      ))
    elif (src / path / ".git").exists():
      problems += submodule_problems(src / path, url, f"{name}/")
  return [(status, _without_userinfo(line)) for status, line in problems]


def flagged(src: pathlib.Path) -> list[str]:
  """Returns files in `src` marked `assume-unchanged` or `skip-worktree`.

  These files have a lowercase or `S` tag in `git ls-files -v` and are hidden
  from `git diff` and `git status` regardless of working-tree changes.

  Args:
    src: Submission directory path.

  Returns:
    A list of relative file paths with `assume-unchanged` or `skip-worktree`.
  """
  out = _git(src, "ls-files", "-v", "-z", "--", ".") or ""
  return [
      rec[2:] for rec in out.split("\0") if rec[:1].islower() or rec[:1] == "S"
  ]


def env_ignored(src: pathlib.Path) -> bool:
  """Returns True if `src` has a `.env` file that git ignores.

  Args:
    src: Submission directory path.

  Returns:
    True if `.env` exists in `src` and is ignored by git, False otherwise.
  """
  return (
      os.path.lexists(src / spec.ENV_FILE)
      and _git(src, "check-ignore", "-q", "--", spec.ENV_FILE) is not None
  )


def repository_root(src: pathlib.Path) -> pathlib.Path | None:
  """Returns the root of the git work tree containing `src`, or None."""
  top = (_git(src, "rev-parse", "--show-toplevel") or "").strip()
  return pathlib.Path(top) if top else None


def competition_clone(src: pathlib.Path) -> pathlib.Path | None:
  """Returns the competition repository root if `src` is inside a clone of it.

  A competition repository clone contains both `kit/check.py` and
  `sample_submissions/`; an entry inside it is not a standalone repository and
  its `.gitignore` excludes `.env`.

  Args:
    src: Submission directory path.

  Returns:
    The root `Path` of the competition clone, or None.
  """
  root = repository_root(src)
  if (
      root
      and (root / "kit" / "check.py").is_file()
      and (root / "sample_submissions").is_dir()
  ):
    return root
  return None


def _is_toplevel(src: pathlib.Path) -> bool:
  root = repository_root(src)
  return root is not None and root.resolve() == src.resolve()


def _tree_bytes(root: pathlib.Path) -> int:
  """Returns the total byte size of regular files under `root`.

  Args:
    root: Directory root to measure (symlinks not followed).

  Returns:
    The total size in bytes of all regular files under `root`.
  """
  total = 0
  for folder, _, names in os.walk(root):
    for n in names:
      st = os.lstat(os.path.join(folder, n))
      total += st.st_size if stat.S_ISREG(st.st_mode) else 0
  return total


def _version(src: pathlib.Path) -> str:
  """Computes the git history component of the build key for `src`.

  Returns the last commit that modified a bundled path along with its
  `git describe --tags --always` string. Because tools like `setuptools-scm` and
  `hatch-vcs` read `HEAD` directly, committing or tagging only `.env` or
  artifacts leaves this build key unchanged so the bundle is reused.

  Args:
    src: Submission directory path.

  Returns:
    A string combining the commit hash and `git describe` output, or `""`.
  """
  top = (_git(src, "ls-tree", "--name-only", "-z", "HEAD") or "").split("\0")
  skip = [
      f":(exclude,literal){n}"
      for n in top
      if n and (n == spec.ENV_FILE or artifacts.bundle_excluded(n))
  ]
  last = (
      _git(
          src,
          "log",
          "-1",
          "--format=%H",
          "--",
          ".",
          *skip,
          ":(exclude,glob)runs/*/artifacts/**",
      )
      or ""
  ).strip()
  if not last:
    return ""
  desc = (_git(src, "describe", "--tags", "--always", last) or "").strip()
  return f"{last} {desc}"


def _stage_git(
    src: pathlib.Path, dest: pathlib.Path, local: bool = True
) -> int:
  """Clones the repository's `.git` history into `dest/.git`.

  Provides the repository history in `dest/.git` during `install` (removed by
  `build.sh` before packing) with the commit's index so `git status` shows only
  uncommitted changes. A `local` clone hard-links objects where possible;
  otherwise git's transport copies only objects reachable from branches, tags,
  and remote-tracking branches, matching the server's GitHub clone.

  Args:
    src: Source repository path.
    dest: Destination staging directory path.
    local: Whether to use a local clone that hard-links git objects.

  Returns:
    The size in bytes of `dest/.git`, which counts toward the repository limit.

  Raises:
    BuildError: If cloning the repository's git history fails.
  """
  tmp = pathlib.Path(tempfile.mkdtemp(prefix=".git-clone-", dir=dest.parent))
  try:
    errors: list[str] = []
    remotes = "remote.origin.fetch=+refs/remotes/origin/*:refs/remotes/github/*"
    fetch = [] if local else ["--no-local", "-c", remotes]
    if (
        artifacts.git_output(
            tmp,
            "clone",
            "--quiet",
            "--no-checkout",
            *fetch,
            "--",
            os.path.abspath(src),
            "repo",
            errors=errors,
        )
        is None
    ):
      raise BuildError(
          "could not copy the repository's git history for the build "
          f"({errors[0] if errors else 'git clone failed'})"
      )
    (tmp / "repo" / ".git").rename(dest / ".git")
  finally:
    shutil.rmtree(tmp, ignore_errors=True)
  artifacts.git_output(dest, "reset", "--quiet")
  return _tree_bytes(dest / ".git")


def _check_places(src: pathlib.Path, dest: pathlib.Path) -> None:
  """Validates that `src` and `dest` do not overlap and `dest` is replaceable.

  Args:
    src: Submission source directory path.
    dest: Staging destination directory path.

  Raises:
    BuildError: If the work directory is equal to or inside `src`, if staging
      into `dest` would overwrite `src`, or if `dest` already exists and is
      neither empty nor previously created by the kit.
  """
  src, work, target = src.resolve(), dest.parent.resolve(), dest.resolve()
  _refuse_inside(src, work)
  if target == src or target in src.parents:
    raise BuildError(
        f"staging into {target} would replace the submission itself: choose"
        " another --work"
    )
  staged = dest.is_dir() and (_mark(dest).is_file() or not any(dest.iterdir()))
  if dest.is_symlink() or dest.exists() and not staged:
    if stack.cache_root().resolve() in work.parents:
      fix = (
          "it is in the kit's own cache, so remove it (`rm -rf"
          f" {shlex.quote(str(dest))}`) and run again"
      )
    else:
      fix = "choose another --work"
    raise BuildError(
        f"{dest} exists and was not made by the kit, which would replace it:"
        f" {fix}"
    )


def _mark(dest: pathlib.Path) -> pathlib.Path:
  return dest.with_name(f".{dest.name}{STAGED_MARK}")


def _walked_files(src: pathlib.Path) -> list[str]:
  """Lists files and symlinks in `src` when `src` is outside a git work tree.

  Walks `src` via `artifacts.walk()` without following symlinks and excluding
  `.git`. Excluded artifact folders (`artifacts.bundle_excluded`) are still
  included up to the review depth because all but the reviewed folder count
  toward the repository size limit, though unreadable paths inside them are
  skipped.

  Args:
    src: Submission directory path.

  Returns:
    A list of relative file and symlink paths under `src`.

  Raises:
    BuildError: If any non-excluded file or folder cannot be read.
  """
  # room for runs/<run>/artifacts above an artifacts folder's depth
  deepest = artifacts.DEPTH + 3

  def too_deep(rel: str, _: pathlib.Path) -> str:
    return (
        artifacts.TOO_DEEP
        if artifacts.bundle_excluded(rel) and rel.count("/") > deepest
        else ""
    )

  names = []
  for rel, _, st, why in artifacts.walk(src, stop=too_deep):
    if why:
      if artifacts.bundle_excluded(rel):
        continue
      raise _unreadable(src, rel, why)
    if st is not None and (
        stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)
    ):
      names.append(rel)
  return names


def _listing(
    src: pathlib.Path,
) -> tuple[
    list[tuple[str, os.stat_result, str | None]],
    list[str],
    int,
    list[str],
    dict[str, bool],
    list[str],
]:
  """Collects files to stage, uncommitted paths, size, LFS pointers, and modes.

  Computes the repository size across staged files and non-reviewed artifact
  folders just as the evaluation server does.

  Args:
    src: Submission directory path.

  Returns:
    A tuple `(entries, uncommitted, total_bytes, lfs_pointers, executable_map,
    left_out_folders)`.

  Raises:
    BuildError: If a bundled file is unreadable or total size exceeds
      `MAX_SOURCE_BYTES`.
  """
  listed = _git_files(src)
  names, uncommitted, pointers, executable = (
      listed if listed is not None else (_walked_files(src), [], [], {})
  )
  # .env goes to the serve container's environment, never into the bundle. The
  # artifacts folders are graded apart and kept out of the bundle, so a harness
  # never finds its own dev answers (artifacts.bundle_excluded); all but the one
  # review() reads, which has limits of its own, still count toward the size.
  exempt = artifacts.reviewed_root(names)
  entries, total, left_out = [], 0, set()
  for rel in sorted(set(names)):
    if rel == spec.ENV_FILE:
      continue
    try:
      st = os.lstat(src / rel)
    # listed by git but deleted in the working tree
    except (FileNotFoundError, NotADirectoryError):
      continue
    except OSError as exc:
      # never bundled: the review says what it cannot read there
      if artifacts.bundle_excluded(rel):
        continue
      raise _unreadable(src, rel, exc.strerror or type(exc).__name__) from None
    if artifacts.bundle_excluded(rel):
      if stat.S_ISREG(st.st_mode) and not (
          exempt and (rel == exempt or rel.startswith(exempt + "/"))
      ):
        total += st.st_size
        parts = rel.split("/")
        left_out.add(
            parts[0]
            if artifacts.bundle_excluded(parts[0])
            else "/".join(parts[:3])
        )
    elif stat.S_ISLNK(st.st_mode):
      try:
        entries.append((rel, st, os.readlink(src / rel)))
      except OSError as exc:
        raise _unreadable(
            src, rel, exc.strerror or type(exc).__name__
        ) from None
    elif stat.S_ISREG(st.st_mode):
      total += st.st_size
      entries.append((rel, st, None))
    if total > MAX_SOURCE_BYTES:
      raise BuildError(
          f"the repository is larger than {MAX_SOURCE_BYTES >> 30} GiB"
      )
  return entries, uncommitted, total, pointers, executable, sorted(left_out)


def _copy(src: pathlib.Path, out: pathlib.Path) -> str:
  """Copies `src` to a new file `out` and returns the SHA-256 hex digest.

  Computes the digest in the same pass as the copy. Read errors set
  `OSError.filename` to `src` so the caller can distinguish unreadable source
  files from write errors on `out`.

  Args:
    src: Source file path to read.
    out: Destination file path to create.

  Returns:
    The lowercase hexadecimal SHA-256 digest of the file's contents.
  """
  digest = hashlib.sha256()
  with src.open("rb") as fin, out.open("xb") as fout:
    while True:
      try:
        chunk = fin.read(1 << 20)
      except OSError as exc:
        exc.filename = exc.filename or str(src)
        raise
      if not chunk:
        break
      digest.update(chunk)
      fout.write(chunk)
  shutil.copystat(src, out)
  return digest.hexdigest()


def stage(src: pathlib.Path, dest: pathlib.Path) -> Source:
  """Stages a clean copy of the submission in `src` into `dest`.

  In a git repository, copies tracked and non-ignored untracked files (plus
  checked-out submodules) and records which files differ from `HEAD`. When `src`
  is the repository root, also stages `.git` history via `_stage_git` so
  `install` can inspect it, and incorporates `_version` into the content key.
  The root `.env` and dev-set artifacts (`artifacts.bundle_excluded`) are
  excluded from the staged tree, matching the evaluation server.

  Args:
    src: Submission source directory path.
    dest: Destination directory path for staged files.

  Returns:
    A `Source` instance describing the staged files, content key, and metadata.

  Raises:
    BuildError: If `src` is not a directory, paths are invalid or unreadable, or
      the repository size exceeds `MAX_SOURCE_BYTES`.
    OSError: If writing to `dest` fails.
  """
  if not src.is_dir():
    raise BuildError(
        f"{src} is not a directory: give the directory that holds {spec.FILE}"
    )
  _check_places(src, dest)
  entries, uncommitted, total, pointers, executable, left_out = _listing(src)
  history = (
      _is_toplevel(src)
      and _git(src, "rev-parse", "--verify", "-q", "HEAD") is not None
  )
  if dest.exists():
    shutil.rmtree(dest)
  dest.mkdir(parents=True)
  _mark(dest).write_text(
      "made by python -m kit; replaced on every build\n", encoding="utf-8"
  )
  key = hashlib.sha256(os.fsencode(f"G {_version(src)}\0") if history else b"")
  digests: dict[str, str] = {}
  for rel, st, target in entries:
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if target is not None:
      os.symlink(target, out)
      key.update(os.fsencode(f"L {rel} {target}\0"))
      continue
    try:
      digests[rel] = _copy(src / rel, out)
    # tell a failed read of the entry's file from a failed write of the copy
    except OSError as exc:
      why, read = exc.strerror or type(exc).__name__, exc.filename == str(
          src / rel
      )
      if (
          not os.access(src / rel, os.R_OK)
          or read
          and exc.errno in (errno.EACCES, errno.EPERM)
      ):
        raise _unreadable(src, rel, why) from None
      if read:  # a failing drive under the entry, say
        raise BuildError(
            f"{rel} could not be read ({why}): a disk or I/O problem on this"
            " machine, not your harness; fix it, then run again"
        ) from None
      raise
    # the mode git records, which the server's checkout gives the file: the
    # owner's executable bit, or the index's where core.fileMode is false
    # (_git_files)
    mode = 0o755 if executable.get(rel, st.st_mode & stat.S_IXUSR) else 0o644
    os.chmod(out, mode)
    # os.fsencode, since a name that is not UTF-8 holds a lone surrogate, which
    # str.encode() refuses
    key.update(os.fsencode(f"F {rel} {mode:o} {st.st_size} {digests[rel]}\0"))
  if history:
    history_bytes = _stage_git(src, dest)
    # a local clone keeps a squashed history's old objects
    if total + history_bytes > MAX_SOURCE_BYTES:
      shutil.rmtree(dest / ".git")
      history_bytes = _stage_git(src, dest, local=False)
    total += history_bytes
  if total > MAX_SOURCE_BYTES:
    shutil.rmtree(dest)
    raise BuildError(
        "the repository, with its git history, is larger than"
        f" {MAX_SOURCE_BYTES >> 30} GiB as the server clones it: every"
        " committed version of artifacts/ stays in the history, so keep the"
        " artifacts small or squash the history before you submit"
    )
  return Source(
      dest,
      len(entries),
      uncommitted,
      key.hexdigest(),
      digests,
      sorted(set(pointers) & set(digests)),
      left_out,
  )


def _script(has_install: bool) -> str:
  return (
      "\n".join([
          "set -eo pipefail",
          # fd 3 carries the bundle; everything else goes to the log
          "exec 3>&1 1>&2",
          f"cp -a /mnt/src {spec.SRC}",
          f"chown -R 0:0 {spec.SRC}",
          'mkdir -p "$HOME"',
          f"cd {spec.SRC}",
          "echo '--- install'",
          "bash -eo pipefail /mnt/install.sh 3>&-"
          if has_install
          else "echo '(none)'",
          "echo '--- packing /opt/harness'",
          f"rm -rf {spec.SRC}/.git",
          # no setuid bits; readable by the serve container's user
          "chmod -R u-s,g-s,a+rX /opt/harness",
          "tar -czf - -C / opt/harness >&3",
      ])
      + "\n"
  )


def _docker_failed(rc: int, log: pathlib.Path) -> None:
  """Raises `stack.InfraError` if the builder container failed before `install`.

  Checks `log` for `MOUNT_MISSING` (a bind mount the Docker daemon could not
  see) or a missing `BUILDER_STARTED` marker (Docker failed to start the
  container at all). Returns normally if the container started with all inputs
  present, meaning the non-zero exit code came from `install` itself.

  Args:
    rc: Process exit code from `docker run`.
    log: Path to the build log file.

  Raises:
    stack.InfraError: If a mounted input was missing or Docker failed to start
      the container.
  """
  # the markers come first, before anything the install prints; Docker's own
  # error comes last
  try:
    with log.open("rb") as fh:
      head = fh.read(64 << 10).decode("utf-8", errors="replace")
      fh.seek(max(0, fh.seek(0, os.SEEK_END) - 4096))
      tail = fh.read().decode("utf-8", errors="replace")
  except OSError:
    head = tail = ""
  missing = next(
      (
          line.split(MOUNT_MISSING, 1)[1].strip()
          for line in head.splitlines()
          if MOUNT_MISSING in line
      ),
      None,
  )
  if missing is not None:
    raise stack.InfraError(
        f"the builder container got an empty directory in place of {missing}:"
        f" {SHARING}"
    )
  if BUILDER_STARTED not in head:
    said = next(
        (line.strip() for line in reversed(tail.splitlines()) if line.strip()),
        "",
    )
    hint = f"; {SHARING}" if "mount" in said.lower() else ""
    raise stack.InfraError(
        f"Docker could not start the builder container (exit {rc}):"
        f" {said[-400:]}{hint}"
    )


def build(
    sub: spec.Submission,
    source: Source,
    work: pathlib.Path,
    builder: str | None = None,
) -> Bundle:
  """Builds the submission bundle at `work/bundle.tar.gz`.

  Callers must hold `locked(work)`.

  Args:
    sub: Parsed submission specification.
    source: Staged submission source files.
    work: Work directory for build outputs and logs.
    builder: Optional builder image tag; resolved via `builder_image` if None.

  Returns:
    A `Bundle` describing the built `bundle.tar.gz` archive.

  Raises:
    BuildError: If `install` times out, exits non-zero, or produces an invalid
      bundle archive.
    stack.InfraError: If Docker fails to start the builder container.
  """
  work.mkdir(parents=True, exist_ok=True)
  log = work / "build.log"
  log.write_text("", encoding="utf-8")
  started = time.time()
  image = builder or builder_image(log)[0]
  # in the work directory, which the daemon must be able to mount (SHARING)
  scripts = pathlib.Path(tempfile.mkdtemp(prefix="build-scripts-", dir=work))
  # UTF-8 whatever the locale, as the server writes them: install may hold any
  # text
  (scripts / "install.sh").write_text(
      spec.fill(sub.install) + "\n", encoding="utf-8"
  )
  (scripts / "build.sh").write_text(
      _script(bool(sub.install)), encoding="utf-8"
  )
  name = f"plebench-build-{sub.name}-{uuid.uuid4().hex[:8]}"
  # The label names the build itself as owner, so stack.sweep removes a builder
  # a killed kit left running once it is older than any build can take. The
  # mounts are the kit's own copies, relabelled (z) so that a daemon with
  # SELinux lets the container read them; elsewhere z does nothing.
  cmd = [
      "docker",
      "run",
      "--rm",
      "--name",
      name,
      "--label",
      f"{stack.LABEL}={name}",
      "--log-driver",
      "none",
      *limits(),
      "-v",
      f"{source.path}:/mnt/src:ro,z",
      "-v",
      f"{scripts}/install.sh:/mnt/install.sh:ro,z",
      "-v",
      f"{scripts}/build.sh:/mnt/build.sh:ro,z",
      image,
      "bash",
      "-c",
      BUILDER_COMMAND,
  ]
  # left by a killed build: none can still be running
  for stale in work.glob("bundle.tar.gz.partial-*"):
    with contextlib.suppress(OSError):
      if time.time() - stale.stat().st_mtime > BUILD_TIMEOUT_SEC + 600:
        stale.unlink()
  # this build's own: renamed into place whole
  partial = work / f"bundle.tar.gz.partial-{uuid.uuid4().hex[:8]}"
  digest, members = hashlib.sha256(), Members()
  try:
    try:
      rc = _run(cmd, log, partial, digest, members)
    finally:
      problem = members.problem()
    if rc == -9:
      raise BuildError(
          f"install did not finish within {BUILD_TIMEOUT_SEC // 60} minutes"
      )
    if rc != 0:
      _docker_failed(rc, log)
      raise BuildError(f"install failed (exit {rc}); the end of {log} says why")
    if problem:
      raise BuildError(f"the bundle cannot be used: {problem}")
  except BaseException:
    # time-bounded, and never raised in place of the original error: with a hung
    # daemon or no docker CLI, stack.sweep removes the builder later
    with contextlib.suppress(stack.InfraError):
      stack.docker("rm", "-f", name, timeout=60, check=False)
    partial.unlink(missing_ok=True)
    raise
  finally:
    shutil.rmtree(scripts, ignore_errors=True)
  bundle = work / "bundle.tar.gz"
  partial.replace(bundle)
  size = bundle.stat().st_size
  with log.open("a") as fh:
    fh.write(
        f"--- bundle {size / 2**20:.0f} MiB, built in"
        f" {time.time() - started:.0f} s\n"
    )
  return Bundle(bundle, size, time.time() - started, digest.hexdigest())


def build_key(sub: spec.Submission, source: Source, builder_id: str) -> str:
  """Computes the cache key that decides whether the last bundle is reused.

  Combines the staged files digest (`source.key`, which includes file paths,
  modes, contents or link targets, and git version), the `install` command, the
  build script, and the builder image ID.

  Args:
    sub: Parsed submission specification.
    source: Staged submission source metadata.
    builder_id: Docker image ID of the builder image.

  Returns:
    The SHA-256 hex digest serving as the build cache key.
  """
  h = hashlib.sha256(f"files {source.key}\0".encode())
  h.update(
      f"install {spec.fill(sub.install)}\0{_script(bool(sub.install))}\0builder"
      f" {builder_id}".encode()
  )
  return h.hexdigest()


def prepare(
    sub: spec.Submission,
    source: Source,
    work: pathlib.Path,
    rebuild: bool = False,
) -> Prepared:
  """Prepares the bundle and serve image for a staged submission.

  Reuses the previous bundle and serve image in `work` when the staged files,
  git version, `install` command, and builder image match `KEY_FILE`, unless
  `rebuild` is True.

  Args:
    sub: Parsed submission specification.
    source: Staged submission source metadata.
    work: Work directory path.
    rebuild: Whether to force a fresh build even if the build key matches.

  Returns:
    A `Prepared` instance containing the `Bundle`, serve image tag, and whether
    the bundle was reused.
  """
  builder, builder_id = builder_image(work / "builder.log")
  key = build_key(sub, source, builder_id)
  marker, path = work / KEY_FILE, work / "bundle.tar.gz"
  saved = marker.read_text().split() if marker.is_file() else []
  if not rebuild and path.is_file() and len(saved) == 2 and saved[0] == key:
    bundle, reused = Bundle(path, path.stat().st_size, 0.0, saved[1]), True
  else:
    marker.unlink(missing_ok=True)
    bundle, reused = build(sub, source, work, builder), False
    marker.write_text(f"{key} {bundle.sha256}\n", encoding="utf-8")
  image = stack.serve_image(
      bundle.path,
      builder,
      work / "serve-image.log",
      digest=bundle.sha256,
      base_id=builder_id,
  )
  return Prepared(bundle, image, reused)


def _run(
    cmd: list[str], log: pathlib.Path, sink: pathlib.Path, *taps: Any
) -> int:
  """Runs `cmd`, appending stderr to `log` and streaming stdout to `sink`.

  Feeds each stdout chunk to every object in `taps` via `.update(chunk)` and
  enforces `MAX_BUNDLE_BYTES` and `BUILD_TIMEOUT_SEC`.

  Args:
    cmd: Command and arguments to execute.
    log: Log file path for stderr output.
    sink: Output file path for stdout bytes.
    *taps: Consumers implementing `.update(bytes)` (e.g., a hash or `Members`).

  Returns:
    The process return code, or `-9` if killed on timeout.

  Raises:
    BuildError: If stdout exceeds `MAX_BUNDLE_BYTES`.
  """
  with log.open("ab") as err, sink.open("xb") as out, subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=err,
      start_new_session=True,
      env=stack.docker_env(),
  ) as proc:
    timer = threading.Timer(
        BUILD_TIMEOUT_SEC, os.killpg, (proc.pid, signal.SIGKILL)
    )
    timer.start()
    try:
      size = 0
      while chunk := proc.stdout.read(1 << 20):
        size += len(chunk)
        if size > MAX_BUNDLE_BYTES:
          os.killpg(proc.pid, signal.SIGKILL)
          raise BuildError(
              f"/opt/harness is larger than {MAX_BUNDLE_BYTES >> 30} GiB once"
              " packed"
          )
        out.write(chunk)
        for tap in taps:
          tap.update(chunk)
      proc.wait()
    finally:
      timer.cancel()
  return proc.returncode


def _resolve(
    path: str, links: dict[str, str], budget: list[int], hops: int = 0
) -> str | None:
  """Resolves `path` by following archive symlinks in `links`.

  Args:
    path: Archive path to resolve.
    links: Mapping of symlink archive paths to their link targets.
    budget: Single-element list tracking remaining path characters to inspect.
    hops: Number of symlinks already followed to reach `path`.

  Returns:
    The resolved normalized path, or None if symlink resolution exceeds
    `MAX_LINK_HOPS` or exhausts `budget`.
  """
  todo, done = (
      collections.deque(p for p in path.split("/") if p not in ("", ".")),
      [],
  )
  while todo:
    part = todo.popleft()
    if part == "..":
      if done:
        done.pop()
      continue
    done.append(part)
    key = "/".join(done)
    budget[0] -= len(key) + 1
    target = links.get(key)
    if budget[0] < 0 or target is not None and hops == MAX_LINK_HOPS:
      return None
    if target is not None:
      hops += 1
      done.pop()
      todo.extendleft(
          reversed([p for p in target.split("/") if p not in ("", ".")])
      )
      if target.startswith("/"):
        done = []
  return "/".join(done)


class _BadHeader(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when `_Header` rejects a tar header before `tarfile` reads its body.

  Does not inherit from `tarfile.HeaderError` because `tarfile` treats some
  header errors mid-archive as end-of-archive.
  """


class _Header(tarfile.TarInfo):
  """Tar member header validator that rejects unsafe or oversized headers early.

  Refuses pax headers, sparse headers, and oversized GNU long name/link headers
  before `tarfile` allocates memory to read their payloads, matching the
  evaluation server's checks.
  """

  def _proc_member(self, archive):  # pylint: disable=arguments-renamed
    if self.type in (
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_SPARSE,
    ):
      raise _BadHeader(
          "it has a pax or sparse header, which the builder's tar does not"
          " write"
      )
      # tarfile's hook for each header it reads
    if (
        self.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
        and self.size > 4 * MAX_NAME + 1
    ):
      raise _BadHeader(
          f"it has a long name or link header of {self.size} bytes, more than a"
          f" name of {MAX_NAME} characters takes"
      )
    return super()._proc_member(archive)


class Members:
  """Streaming reader that inspects bundle tar members via a background thread.

  Reads archive bytes written through `update()` over an OS pipe so the bundle
  is validated on the fly without decompressing it a second time. Calling
  `problem()` closes the stream and returns the validation result from
  `members_problem`.
  """

  def __init__(self):
    r, w = os.pipe()
    self._in, self._out = os.fdopen(r, "rb"), os.fdopen(w, "wb")
    self._members: list[tarfile.TarInfo] = []
    self._problem: str | None = None
    self._thread = threading.Thread(target=self._read, daemon=True)
    self._thread.start()

  def _read(self) -> None:
    try:
      self._problem = _read_members(self._in, self._members)
    # whatever the bytes are, the build gets an answer
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._problem = f"not a readable tar.gz ({type(exc).__name__}: {exc})"
    finally:
      # drain the pipe, so the writer cannot block on a reader that
      # stopped
      while self._in.read(1 << 20):
        pass
      self._in.close()

  def update(self, chunk: bytes) -> None:
    """Writes a chunk of compressed archive bytes to the reader pipe.

    Args:
      chunk: Compressed archive bytes.
    """
    self._out.write(chunk)

  def problem(self) -> str | None:
    """Closes the stream and returns why the bundle is unsafe, or None if valid.

    Returns:
      An error description if the bundle is invalid, or None if it is safe.
    """
    if not self._out.closed:
      self._out.close()
    self._thread.join()
    return self._problem or members_problem(self._members)


def _read_members(fileobj: Any, into: list[tarfile.TarInfo]) -> str | None:
  """Streams tar.gz members from `fileobj` into `into` and checks limits.

  Args:
    fileobj: Readable binary stream containing the `.tar.gz` archive.
    into: List to append parsed `tarfile.TarInfo` members to.

  Returns:
    An error description if reading fails, `_Header` rejects a header, or the
    member count exceeds `MAX_MEMBERS`; otherwise None.
  """
  try:
    with tarfile.open(fileobj=fileobj, mode="r|gz", tarinfo=_Header) as tar:
      for m in tar:
        into.append(m)
        if len(into) > MAX_MEMBERS:
          return f"it has more than {MAX_MEMBERS} files"
  except _BadHeader as exc:
    return str(exc)
  except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
    return f"not a readable tar.gz ({exc})"
  return None


def members_problem(members: list[tarfile.TarInfo]) -> str | None:
  """Returns why an archive with `members` cannot be unpacked safely, or None.

  Verifies that each member name is unique, at most `MAX_NAME` characters (along
  with its link target), and stays within `opt/harness` after resolving archive
  symlinks without writing through a symlink. Also checks that hard links point
  to earlier regular files, no devices/FIFOs/setuid/setgid files exist, and
  total members and unpacked bytes stay within `MAX_MEMBERS` and
  `MAX_UNPACKED_BYTES`.

  Args:
    members: List of `tarfile.TarInfo` entries from the bundle archive.

  Returns:
    A string describing the first safety violation found, or None if valid.
  """
  if len(members) > MAX_MEMBERS:
    return f"it has more than {MAX_MEMBERS} files"
  if sum(m.size for m in members) > MAX_UNPACKED_BYTES:
    return (
        f"/opt/harness is larger than {MAX_UNPACKED_BYTES >> 30} GiB unpacked"
    )

  def inside(rel: str | None) -> bool:
    return rel is not None and (rel == ROOT or rel.startswith(ROOT + "/"))

  names = [posixpath.normpath(m.name.removeprefix("./")) for m in members]
  seen: set[str] = set()
  for name, m in zip(names, members):
    if len(m.name) > MAX_NAME or len(m.linkname) > MAX_NAME:
      return (
          f"{m.name[:80]!r}... has a name or link target longer than {MAX_NAME}"
          " characters"
      )
    if name in seen:
      return (
          f"{m.name!r} is in the archive twice (unpacked, the second replaces"
          " the first)"
      )
    seen.add(name)
  links = {n: m.linkname for n, m in zip(names, members) if m.issym()}
  # the regular files stored so far: what a hard link may name
  files: set[str] = set()
  budget = [LINK_CHECK_BUDGET]
  for name, m in zip(names, members):
    # the bundle root's parent: let through, as the server's check does
    if name == "opt" and m.isdir():
      continue
    if m.name.startswith("/") or ".." in m.name.split("/") or not inside(name):
      return f"{m.name!r} lies outside /opt/harness"
    if m.mode & 0o6000:
      return f"{m.name!r} is setuid or setgid"
    if m.issym():
      start = (
          m.linkname
          if m.linkname.startswith("/")
          else posixpath.join(posixpath.dirname(name), m.linkname)
      )
      # the link itself is the first followed
      if not inside(_resolve(start, links, budget, hops=1)):
        hint = ""
        if re.search(r"(^|/)python[0-9.]*$", m.linkname):
          hint = (
              "; a venv made with `python3 -m venv` links to the system Python,"
              " which the bundle does not keep: create it with uv, `uv venv"
              " --python 3.12 /opt/harness/venv`"
          )
        return (
            f"the link {m.name!r} points outside /opt/harness"
            f" ({m.linkname}){hint}"
        )
    elif m.islnk():
      target = posixpath.normpath(m.linkname.removeprefix("./").lstrip("/"))
      if (
          ".." in m.linkname.split("/")
          or not inside(target)
          or target not in files
          or _resolve(posixpath.dirname(target), links, budget)
          != posixpath.dirname(target)
      ):
        return (
            f"the hard link {m.name!r} does not name, by a plain path, a"
            f" regular file stored before it under /opt/harness ({m.linkname})"
        )
    elif not (m.isfile() or m.isdir()):
      return f"{m.name!r} is a device or FIFO"
    if m.isfile():
      files.add(name)
  ordered = sorted(names)
  for link in links:
    i = bisect.bisect_left(ordered, link + "/")
    if i < len(ordered) and ordered[i].startswith(link + "/"):
      return f"{ordered[i]!r} would be written through the symlink {link!r}"
  return None
