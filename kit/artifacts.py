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

"""Validation, inspection, and materialization of dev-set submission artifacts.

The kit, the submit page, and the evaluation server all read a submission's
artifacts through `review()`, ensuring they agree on what counts. `review()`
operates on a repository listing (`Entry`: path, git mode, size, blob ID), which
comes from a Git tree on the server and from files on disk in the kit;
`materialize()` then copies the usable files into one folder per task in the
layout read by the replay agents (`replay.json`).
"""

from __future__ import annotations

import codecs
import dataclasses
import difflib
import errno
import hashlib
import json
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
from typing import Any, BinaryIO, Callable, Iterable, Iterator

import yaml

FORMAT = 1
ROOT = "artifacts"
MANIFEST = "manifest.json"
FINAL_MESSAGE = "final_message.md"
FILES = "files"
DELETED = "deleted.txt"
TRAJECTORY = "trajectory.json"
# materialize()'s index of one task's folder, for the replay agents
REPLAY = "replay.json"
ATTRIBUTES = ".gitattributes"
IGNORE = ".gitignore"
# the kit's own .gitattributes and .gitignore in an artifacts folder
# (export._layout)
ATTRIBUTES_TEXT = (
    "# written by the kit: git keeps these files byte for byte (no line-ending"
    " conversion, no filter such as Git LFS, no $Id$ expansion, no"
    " re-encoding)\n* -text -filter -ident -working-tree-encoding\n"
)
IGNORE_TEXT = (
    "# written by the kit: git commits every file here, whatever a .gitignore"
    " above leaves out\n!*\n"
)
# the kit's own files directly in the artifacts folder
LAYOUT_FILES = frozenset({MANIFEST, ATTRIBUTES, IGNORE})
# with .git/, never exported or graded
GIT_OWN = frozenset({".gitignore", ".gitattributes", ".gitmodules"})
# prefix of what an export writes beside the task folders before swapping it in
KIT_STAGING = ".plebench-"
# <task>.earlier: where an earlier kit's export kept a folder it could not put
# back
EARLIER = ".earlier"
EARLIER_RE = re.compile(
    r"(?P<task>.+)" + re.escape(EARLIER) + r"(?:-[0-9a-f]{4})?"
)
ENV_FILE = ".env"
DEV_SET_FILE = (
    pathlib.Path(__file__).resolve().parents[1] / "eval" / "dev_set.yaml"
)

MAX_FILE_BYTES = 95 << 20  # GitHub refuses a file over 100 MiB
MAX_TASK_BYTES = 256 << 20
MAX_TASK_FILES = 2000
MAX_TOTAL_BYTES = 512 << 20
MAX_TOTAL_FILES = 10_000
MAX_MESSAGE_BYTES = 1 << 20
MAX_TRAJECTORY_BYTES = 32 << 20
MAX_MANIFEST_BYTES = 4 << 20
MAX_DELETED_BYTES = 256 << 10
# the manifest lists the harness's files up to this many (harness.files)
MAX_HARNESS_FILES = 5000
# Path limits, measured on the path as committed under the task folder
# (files/<container path>). No deliverable comes close to the folder limit.
MAX_PATH_FOLDERS = 64
MAX_PATH_BYTES = 1024
# one file or folder name: what Linux, macOS and Windows file systems allow
MAX_NAME_BYTES = 255
SHOWN = 5  # paths named in one warning; the rest are counted
# What a file system answers for a name it cannot hold (too long, not valid
# there, a file where a folder goes). macOS caps a whole path at 1,024 bytes, so
# a path review() accepts can still be too long once a local folder is added.
NAME_ERRNOS = frozenset({
    errno.ENAMETOOLONG,
    errno.EILSEQ,
    errno.EINVAL,
    errno.ENOTDIR,
    errno.EISDIR,
    errno.EEXIST,
})

# 100664: old git versions wrote it for regular files
FILE_MODES = {"100644", "100755", "100664"}
LINK, SUBMODULE = "120000", "160000"
ROOT_ALIAS_RE = re.compile(r"(dev[-_]?)?artifacts?", re.IGNORECASE)
RUN_FOLDER = r"runs/[^/]+/artifacts"
# runs/ or a run folder: one this machine cannot list stops the review when no
# artifacts/ is read, else it warns
RUNS_FOLDER_RE = re.compile(r"runs(/[^/]+)?")
# a runs/<run>/artifacts folder at the start of a longer path
RUN_ROOT_RE = re.compile(RUN_FOLDER + r"(?=/)")
ARCHIVE_RE = re.compile(
    r"(?P<stem>.+?)\.(zip|tar|tgz|tar\.gz|tar\.bz2|tar\.xz|tar\.zst|7z|rar|gz)",
    re.IGNORECASE,
)
# what Harbor puts after a trial folder's task name and "__"
TRIAL_SUFFIX_RE = re.compile(r"[A-Za-z0-9]{4,}")
# in order of preference
MESSAGE_ALIASES = (
    "final_message.txt",
    "final_message",
    "final-message.md",
    "message.md",
)
JUNK = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", "__MACOSX"})
TRIAL_PARTS = frozenset({
    "agent",
    "verifier",
    "result.json",
    "config.json",
    "trial.log",
    "artifacts",
})
LFS_POINTER = b"version https://git-lfs.github.com/spec/"
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# documentation, by file name: README, LICENSE or CHANGELOG, bare or as .md,
# .txt, .rst or .markdown
DOCS_RE = re.compile(
    r"(readme|licen[cs]e|changelog)(\.(md|txt|rst|markdown))?", re.IGNORECASE
)

OK, EMPTY, REFUSED, MISSING = "ok", "empty", "refused", "missing"


@dataclasses.dataclass(frozen=True)
class DevTask:
  """Metadata and deliverable specification for a dev-set task."""

  id: str
  # checks, rubric or checks-judge: how its verifier container grades it
  grading: str = "checks"
  # container paths; one ending in "/" is a folder, everything under it
  deliverables: tuple[str, ...] = ()
  final_message: bool = False  # its grader reads the final message
  title: str = ""

  @property
  def deliverable_files(self) -> tuple[str, ...]:
    return tuple(d for d in self.deliverables if not d.endswith("/"))

  def covers(self, path: str) -> bool:
    """Returns whether `path` is covered by this task's deliverables.

    Args:
      path: Container path to check.

    Returns:
      True if `path` matches a deliverable file or lies under a deliverable
      directory.
    """
    return covered(self.deliverables, path)


def covered(deliverables: Iterable[str], path: str) -> bool:
  """Returns whether `path` matches or lies under one of `deliverables`.

  A deliverable ending in `"/"` covers any path under that folder; a file
  deliverable covers only itself. Shared by artifact capture and `review()`.

  Args:
    deliverables: Deliverable container paths for a task.
    path: Container path to check.

  Returns:
    True if `path` is covered by `deliverables`.
  """
  return any(
      path == d or (d.endswith("/") and path.startswith(d))
      for d in deliverables
  )


def load_dev_set(path: str | pathlib.Path) -> dict[str, DevTask]:
  """eval/dev_set.yaml's tasks, by id."""
  data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
  out = {}
  for t in data.get("tasks") or []:
    grading = t.get("grading", "checks")
    out[t["id"]] = DevTask(
        id=t["id"],
        grading=grading,
        deliverables=tuple(t.get("deliverables") or ()),
        final_message=bool(t.get("final_message", grading == "rubric")),
        title=t.get("title", ""),
    )
  return out


def dev_set_digest(path: str | pathlib.Path) -> str:
  """Computes the SHA-256 hex digest of the dev-set file with LF line endings.

  Normalizes CRLF to LF so a Windows checkout produces the same digest as the
  server's copy.

  Args:
    path: Path to the `dev_set.yaml` file.

  Returns:
    The lowercase hexadecimal SHA-256 digest string.
  """
  return hashlib.sha256(
      pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")
  ).hexdigest()


def bundle_excluded(rel: str) -> bool:
  """Returns whether `rel` is or lies inside an artifacts folder.

  Matches top-level `artifacts`-like folders and `runs/<run>/artifacts` so that
  neither build includes submission artifacts in the harness bundle.

  Args:
    rel: Repository-relative path to check.

  Returns:
    True if `rel` should be excluded from the harness bundle.
  """
  parts = rel.strip("/").split("/")
  return bool(ROOT_ALIAS_RE.fullmatch(parts[0])) or (
      len(parts) >= 3 and parts[0] == "runs" and parts[2] == ROOT
  )


def blob_id(data: bytes) -> str:
  """Computes Git's SHA-1 blob object ID for `data`.

  Args:
    data: Raw file bytes.

  Returns:
    The 40-character hexadecimal SHA-1 git blob digest.
  """
  return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def file_blob_id(path: pathlib.Path) -> str:
  h = hashlib.sha1(b"blob %d\x00" % os.stat(path).st_size)
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


# The helpers below never raise and never follow a link. Path.is_dir() and
# friends raise on a permission error before Python 3.14, and os.walk is
# recursive before 3.12.


def _why(exc: BaseException) -> str:
  return getattr(exc, "strerror", None) or type(exc).__name__


def lstat(path: str | pathlib.Path) -> os.stat_result | None:
  """Runs `os.lstat(path)`, returning `None` if `path` cannot be inspected.

  Args:
    path: Filesystem path to stat without following symlinks.

  Returns:
    The `os.stat_result`, or `None` on permission or path-limit errors.
  """
  try:
    return os.lstat(path)
  except (OSError, ValueError):
    return None


def is_folder(path: str | pathlib.Path) -> bool:
  """A folder, never a link to one."""
  st = lstat(path)
  return st is not None and stat.S_ISDIR(st.st_mode)


def is_regular(path: str | pathlib.Path) -> bool:
  """A regular file, never a link to one."""
  st = lstat(path)
  return st is not None and stat.S_ISREG(st.st_mode)


TOO_DEEP = "nested too deep to list"
SUBMODULE_STOP = "a git repository of its own"
# How many folders below an artifacts folder a walk enters: a wrapper, a task
# folder and the deepest path review() accepts, plus one level, which it lists
# as too deep so that path_problem() refuses that task for its depth.
DEPTH = MAX_PATH_FOLDERS + 3


def walk(
    top: pathlib.Path,
    rel: str = "",
    depth: int | None = None,
    stop: Callable[[str, pathlib.Path], str] | None = None,
) -> Iterator[tuple[str, pathlib.Path, os.stat_result | None, str]]:
  """Walks `top` depth-first in sorted order without following symlinks.

  Never enters `.git` directories. Non-directory entries are yielded with an
  empty reason `""`. Directories that are not entered are yielded with the stop
  reason: `stop(rel, path)`, `TOO_DEEP` when deeper than `depth`, or the OS
  error string when the directory cannot be listed and entered. Entries that
  cannot be `lstat`-ed are yielded with `None` as the stat result.

  Args:
    top: Root path to walk (or a single file path).
    rel: Relative path prefix corresponding to `top`.
    depth: Optional maximum directory nesting depth below `top`.
    stop: Optional callback returning a non-empty reason to skip entering a
      directory.

  Yields:
    Tuples of `(relative_path, full_path, lstat_result, stop_or_error_reason)`.
  """
  stack = [(top, rel, 0)]
  while stack:
    path, here, level = stack.pop()
    try:
      st = os.lstat(path)
    # no permission, or a name past the system's limits
    except (OSError, ValueError) as exc:
      yield here, path, None, _why(exc)
      continue
    if not stat.S_ISDIR(st.st_mode):
      yield here, path, st, ""
      continue
    why = stop(here, path) if stop else ""
    if not why and depth is not None and level > depth:
      why = TOO_DEEP
    if not why and not os.access(path, os.R_OK | os.X_OK):
      # without x its entries cannot be looked at; without r, not listed
      why = os.strerror(errno.EACCES)
    if why:
      yield here, path, st, why
      continue
    try:
      names = sorted(os.listdir(path), reverse=True)
    except OSError as exc:
      yield here, path, st, _why(exc)
      continue
    prefix = f"{here}/" if here else ""
    stack += [
        (path / name, prefix + os.fsdecode(name), level + 1)
        for name in names
        if name != ".git"
    ]


def regular_files(top: pathlib.Path) -> Iterator[pathlib.Path]:
  """Yields every regular file under `top` in sorted order.

  Does not follow symlinks and skips unreadable or overly deep folders.

  Args:
    top: Root directory to search.

  Yields:
    Paths to regular files under `top`.
  """
  for rel, path, st, why in walk(top, depth=DEPTH):
    if rel and not why and st is not None and stat.S_ISREG(st.st_mode):
      yield path


@dataclasses.dataclass(frozen=True)
class Entry:
  """One path of a repository as git records it.

  `mode` is git's (100644, 100755, 120000 for a link, 160000 for a submodule),
  "tree" for a folder found on disk, or "special" for a device, FIFO, socket or
  unreadable entry found on disk; `sha` is git's object id when known.
  """

  path: str
  mode: str
  size: int | None = None
  sha: str | None = None
  # a Git LFS pointer in place of the file (known only where the content was
  # read)
  lfs: bool = False
  # git ignores it, so a commit would not hold it (known only from a working
  # tree)
  ignored: bool = False
  # the index holds other content although git sees no change (a working tree
  # only)
  stale: bool = False
  # why this machine could not read it (a working tree only); see _blocking
  unreadable: str = ""

  @property
  def is_file(self) -> bool:
    return self.mode in FILE_MODES

  @property
  def executable(self) -> bool:
    return self.mode == "100755"

  @property
  def kind(self) -> str:
    if self.is_file:
      return "file"
    if self.mode == LINK:
      return "symbolic link"
    if self.mode == SUBMODULE:
      return "submodule"
    return "special file"


def from_git_tree(items: Iterable[dict[str, Any]]) -> list[Entry]:
  """Converts items from GitHub's recursive git trees API into `Entry` objects.

  Directory (`"tree"`) entries are omitted.

  Args:
    items: Iterable of tree entry dicts containing `path`, `mode`, `type`,
      `size`, and `sha`.

  Returns:
    A list of `Entry` objects for all non-tree items.
  """
  return [
      Entry(str(i["path"]), str(i["mode"]), i.get("size"), i.get("sha"))
      for i in items
      if i.get("type") != "tree"
  ]


def from_ls_tree(output: bytes) -> list[Entry]:
  r"""Parses the NUL-delimited output of `git ls-tree -r -l -z <commit>`.

  Each record has the format `"<mode> <type> <object> <size>\t<path>"`.
  Non-UTF-8 path bytes are decoded with surrogate escapes (`os.fsdecode`) and
  later rejected by `path_problem()`.

  Args:
    output: Raw stdout bytes from `git ls-tree -r -l -z`.

  Returns:
    A list of `Entry` objects for all non-tree entries.
  """
  out = []
  for record in output.split(b"\x00"):
    if not record:
      continue
    head, _, raw = record.partition(b"\t")
    mode, kind, sha, size = head.decode("ascii").split()
    if kind == "tree":
      continue
    out.append(
        Entry(os.fsdecode(raw), mode, None if size == "-" else int(size), sha)
    )
  return out


# from_directory's and worktree_entries' `git` when the caller has not asked git
# yet
_ASK = object()


def from_directory(
    repo: pathlib.Path,
    hashes: bool = True,
    git: GitRunner | None | object = _ASK,
) -> list[Entry]:
  """Lists the working tree entries of `repo` relevant to `review()`.

  Includes every top-level name and everything under any candidate artifacts
  folder (top-level `artifacts`-like folders and `runs/<run>/artifacts`). Does
  not follow symlinks and records nested git repositories as submodules. Inside
  a git work tree, records whether git ignores each entry, the committed file
  mode (`git_mode`), and, when `hashes` is True, the `git hash-object` blob ID.

  Args:
    repo: Path to the repository or entry directory on disk.
    hashes: Whether to compute git blob SHA-1 hashes for regular files.
    git: Optional pre-initialized `GitRunner` from `work_tree(repo)`.

  Returns:
    A list of `Entry` objects representing the inspected directory contents.
  """
  # (path, where it is, lstat, found inside an artifacts folder, why it cannot
  # be read)
  found: list[tuple[str, pathlib.Path, os.stat_result | None, bool, str]] = []

  def submodule(rel: str, path: pathlib.Path) -> str:
    del rel
    return (
        SUBMODULE_STOP
        if os.path.lexists(path / ".git") and _own_repository(path)
        else ""
    )

  def add(top: pathlib.Path, top_rel: str, listed: bool) -> None:
    # one name (a top-level one, or a .gitattributes): what it is, nothing
    # below it
    if not listed:
      try:
        st = os.lstat(top)
      except (OSError, ValueError) as exc:
        found.append((top_rel, top, None, False, _why(exc)))
        return
      if not stat.S_ISDIR(st.st_mode):
        # a .gitattributes above the artifacts decides what Git LFS stores: one
        # it cannot read stops the review
        locked = posixpath.basename(top_rel) == ATTRIBUTES and not os.access(
            top, os.R_OK
        )
        found.append((
            top_rel,
            top,
            st,
            False,
            os.strerror(errno.EACCES) if locked else "",
        ))
      elif submodule(top_rel, top):
        found.append((top_rel, top, None, False, ""))
      else:
        # a placeholder: find_root() sees the folder
        found.append((top_rel + "/", top, st, False, ""))
      return
    for rel, path, st, why in walk(top, top_rel, depth=DEPTH, stop=submodule):
      if why == SUBMODULE_STOP:
        found.append((rel, path, None, True, ""))
        continue
      if (
          not why
          and st is not None
          and stat.S_ISREG(st.st_mode)
          and not os.access(path, os.R_OK)
      ):
        # a file it cannot open is reported before any hashing
        why = os.strerror(errno.EACCES)
      found.append((rel, path, st, True, why))

  for child in sorted(repo.iterdir()):
    name = child.name
    if name == ".git":
      continue
    if is_folder(child) and name == "runs":
      # it can apply to a runs/<run>/artifacts folder (Git LFS)
      if is_regular(child / ATTRIBUTES):
        add(child / ATTRIBUTES, f"runs/{ATTRIBUTES}", False)
      for rel, where, why in _runs(child):
        # what it cannot list or enter there: a warning or a stop
        # (review_directory)
        if why:
          found.append((rel, where, lstat(where), False, why))
          continue
        if is_regular(where / ATTRIBUTES):
          add(where / ATTRIBUTES, f"{rel}/{ATTRIBUTES}", False)
        add(where / ROOT, f"{rel}/{ROOT}", True)
      continue
    add(child, name, is_folder(child) and bool(ROOT_ALIAS_RE.fullmatch(name)))
  git = work_tree(repo)[0] if git is _ASK else git
  ignored = (
      _ignored(git, [rel.rstrip("/") for rel, *_ in found]) if git else set()
  )
  unread = {
      rel: Entry(
          rel,
          "special",
          st.st_size if st else None,
          ignored=rel.rstrip("/") in ignored,
          unreadable=why,
      )
      for rel, path, st, listed, why in found
      if why
  }
  # when the review will stop (review_directory), skip hashing and git's index:
  # git could fail on what this machine cannot read. Whether a run folder it
  # cannot list stops it depends on the folder find_root picks, which needs only
  # the paths and which of them are folders.
  stopping = bool(_blocking(unread.values()))
  if not stopping and _unlisted_runs(unread.values()):
    stopping = bool(
        _runs_stop([
            unread.get(rel)
            or Entry(
                rel,
                "tree" if rel.endswith("/") else "100644",
                ignored=rel.rstrip("/") in ignored,
            )
            for rel, *_ in found
        ])
    )
  hashes = hashes and not stopping
  regular = [
      rel
      for rel, path, st, listed, why in found
      if listed and not why and st is not None and stat.S_ISREG(st.st_mode)
  ]
  ids = _hash_objects(git, regular, git.prefix) if git and hashes else {}
  roots = sorted({
      "/".join(rel.split("/")[:3])
      if rel.startswith("runs/")
      else rel.split("/", 1)[0]
      for rel, path, st, listed, why in found
      if listed
  })
  pointers, behind, indexed = (
      index_state(git, roots, ids)
      if git and not stopping
      else (set(), set(), {})
  )
  exec_bit = not git or git_flag(git, "core.fileMode")
  out = []
  for rel, path, st, listed, why in found:
    gone = rel.rstrip("/") in ignored
    if why:
      out.append(unread[rel])
    elif st is None:
      out.append(Entry(rel, SUBMODULE, ignored=gone))
    elif rel.endswith("/"):
      out.append(Entry(rel, "tree", ignored=gone))
    elif stat.S_ISLNK(st.st_mode):
      out.append(Entry(rel, LINK, st.st_size, ignored=gone))
    elif stat.S_ISREG(st.st_mode):
      mode = git_mode(st, indexed.get(rel), exec_bit)
      try:
        sha = (
            (ids.get(rel) or file_blob_id(path)) if hashes and listed else None
        )
      except OSError as exc:
        out.append(
            Entry(
                rel, "special", st.st_size, ignored=gone, unreadable=_why(exc)
            )
        )
        continue
      pointer = (
          listed and st.st_size < 1024 and _pointer(path)
      ) or rel in pointers
      out.append(
          Entry(
              rel,
              mode,
              st.st_size,
              sha,
              lfs=pointer,
              ignored=gone,
              stale=rel in behind,
          )
      )
    else:
      out.append(Entry(rel, "special", st.st_size, ignored=gone))
  return out


def _runs(runs: pathlib.Path) -> Iterator[tuple[str, pathlib.Path, str]]:
  """Yields candidate run folders under `runs` and any unreadable entries.

  Yields `("runs/<run>", path, "")` for run folders containing an `artifacts`
  subdirectory, and yields the relative path, filesystem path, and error reason
  for `runs/` itself, any run folder, or any `runs/<run>/artifacts` folder that
  cannot be listed or entered.

  Args:
    runs: Path to the top-level `runs/` directory.

  Yields:
    Tuples of `(relative_path, full_path, error_reason)`.
  """
  try:
    names = sorted(os.listdir(runs))
  except OSError as exc:
    yield "runs", runs, _why(exc)
    return
  # listed, but nothing in it can be looked at (r without x): runs/ blocks
  if not os.access(runs, os.X_OK):
    yield "runs", runs, os.strerror(errno.EACCES)
    return
  for name in names:
    run, rel = runs / name, f"runs/{os.fsdecode(name)}"
    try:
      st = os.lstat(run)
    except (OSError, ValueError) as exc:
      yield rel, run, _why(exc)
      continue
    if not stat.S_ISDIR(st.st_mode):
      continue
    try:
      inside = os.listdir(run)
    except OSError as exc:
      yield rel, run, _why(exc)
      continue
    if ROOT not in inside:
      continue
    # listed, but its artifacts folder cannot be looked at: the run blocks
    if not os.access(run, os.X_OK):
      yield rel, run, os.strerror(errno.EACCES)
      continue
    try:
      st = os.lstat(run / ROOT)
    except (OSError, ValueError) as exc:
      yield f"{rel}/{ROOT}", run / ROOT, _why(exc)
      continue
    if stat.S_ISDIR(st.st_mode):
      yield rel, run, ""


def index_state(
    git: Callable[..., bytes | None], roots: list[str], ids: dict[str, str]
) -> tuple[set[str], set[str], dict[str, str]]:
  """Inspects Git's index under `roots` for LFS pointers, stale blobs, and mode.

  Among files that Git sees as unchanged in the working tree (whose index entry
  would be used in a commit), identifies those whose staged blob is a Git LFS
  pointer and those whose staged blob ID differs from the on-disk hash in `ids`
  (e.g., after changing Git LFS or line-ending rules without touching the file).

  Args:
    git: Callable that runs a Git command in the repository.
    roots: Candidate artifact root paths to inspect in the index.
    ids: Mapping of relative file paths to their current `git hash-object` IDs.

  Returns:
    A `(lfs_pointer_paths, stale_index_paths, file_modes)` tuple.
  """
  if not roots:
    return set(), set(), {}
  spec = [f":(literal){r}" for r in roots]
  staged = git("ls-files", "-s", "-z", "--", *spec) or b""
  modified = git("ls-files", "-m", "-z", "--", *spec) or b""
  changed = {os.fsdecode(p) for p in modified.split(b"\x00") if p}
  clean: dict[str, str] = {}
  modes: dict[str, str] = {}
  for rec in staged.split(b"\x00"):
    head, _, raw = rec.partition(b"\t")
    fields, path = head.split(), os.fsdecode(raw)
    if len(fields) >= 2 and fields[0].decode() in FILE_MODES:
      modes[path] = fields[0].decode()
      if path not in changed:
        clean[path] = fields[1].decode()
  behind = {p for p, sha in clean.items() if ids.get(p, sha) != sha}
  if clean:
    sizes = git(
        "cat-file",
        "--batch-check=%(objectname) %(objectsize)",
        data="".join(f"{sha}\n" for sha in set(clean.values())).encode(),
    )
  else:
    sizes = b""
  small = [
      w[0]
      for w in (line.split() for line in (sizes or b"").decode().splitlines())
      if len(w) == 2 and w[1].isdigit() and int(w[1]) < 1024
  ]
  if small:
    heads = git(
        "cat-file",
        "--batch",
        data="".join(f"{sha}\n" for sha in small).encode(),
    )
  else:
    heads = b""
  pointer_ids, rest = set(), heads or b""
  while rest:  # "<id> <type> <size>\n<content>\n", one after another
    line, _, rest = rest.partition(b"\n")
    w = line.split()
    if len(w) != 3 or not w[2].isdigit():
      break
    n = int(w[2])
    if rest[:n].startswith(LFS_POINTER):
      pointer_ids.add(w[0].decode())
    rest = rest[n + 1 :]
  return {p for p, sha in clean.items() if sha in pointer_ids}, behind, modes


# what one git command of the kit may take (the listing of a large repository
# included)
GIT_SEC = 300


class GitSaid(str):
  """Error string from a failed Git command, tagged with a failure `kind`.

  `kind` is `"missing"`, `"timeout"`, or `"git"`, which `git_stop()` uses to
  choose its recovery advice.
  """

  def __new__(cls, text: str, kind: str = "git"):
    said = super().__new__(cls, text)
    said.kind = kind
    return said


def git_output(
    repo: pathlib.Path,
    *args: str,
    data: bytes | None = None,
    ok: tuple[int, ...] = (0,),
    timeout: float = GIT_SEC,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> bytes | None:
  """Runs a Git command in `repo` and returns its stdout bytes on success.

  Runs Git in the C locale with terminal prompts and optional locks disabled.
  When Git is missing, exits with a code not in `ok`, or exceeds `timeout`,
  appends a `GitSaid` reason to `errors` (if provided) and returns `None`. Any
  stderr lines produced on a successful run are appended to `warnings`.

  Args:
    repo: Working directory for `git -C`.
    *args: Command-line arguments passed to `git`.
    data: Optional stdin bytes to pass to the process.
    ok: Exit codes treated as successful.
    timeout: Maximum time in seconds to wait for the command.
    errors: Optional list to receive a `GitSaid` error description on failure.
    warnings: Optional list to receive stderr lines on success.

  Returns:
    The raw stdout bytes if the command exited with a code in `ok`, or `None` on
    failure.
  """
  try:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=data,
        capture_output=True,
        timeout=timeout,
        env={
            **os.environ,
            "LC_ALL": "C",
            "LANGUAGE": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        },
        check=False,
    )
  except subprocess.TimeoutExpired:
    if errors is not None:
      errors.append(
          GitSaid(
              f"git {args[0] if args else ''} took longer than {timeout:.0f} s",
              "timeout",
          )
      )
    return None
  except (OSError, subprocess.SubprocessError) as exc:
    if errors is not None:
      errors.append(
          GitSaid("git is not installed", "missing")
          if isinstance(exc, FileNotFoundError)
          else GitSaid(f"git could not run: {exc}")
      )
    return None
  if p.returncode in ok:
    # one line each, on "\n" alone: a file name may hold U+2028 and the like
    if warnings is not None:
      warnings.extend(
          line.removesuffix("\r")
          for line in os.fsdecode(p.stderr).split("\n")
          if line
      )
    return p.stdout
  if errors is not None:
    said = SAFE_DIRECTORY_RE.split(" ".join(os.fsdecode(p.stderr).split()), 1)[
        0
    ].strip()
    errors.append(
        GitSaid(
            said[-400:]
            or f"git {args[0] if args else ''} ended with code {p.returncode}"
        )
    )
  return None


class GitFailed(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when a Git command run in a work tree fails.

  Because the contents of a commit cannot be determined when Git fails, review,
  export, and build operations stop and report the error via `git_stop()`.
  """

  def __init__(self, said: str):
    super().__init__(said)
    self.said = said
    self.kind = getattr(said, "kind", "git")  # git_output's GitSaid


class Unreadable(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when the kit cannot read required files in the entry or work tree.

  Covers unreadable harness files on disk (for which `unreadable_stop()` prints
  the remediation command) as well as work-tree errors from `work_tree()`.
  """

  def __init__(self, problem: str):
    super().__init__(problem)
    self.problem = problem


# where git's advice to trust another user's repository starts: git_output cuts
# it off, since trusting the repository would run its owner's settings
SAFE_DIRECTORY_RE = re.compile(
    r"To add an exception|(?:git config \S+ \S+ )?safe\.directory"
)
# what `git ls-files --others` says, then goes on
COULD_NOT_OPEN = "warning: could not open directory "


def git_words(said: str) -> str:
  return printable(said or "no message")


def git_stop(repo: pathlib.Path, said: str, kind: str = "git") -> str:
  """Formats the user-facing stop message for a `GitFailed` error.

  Args:
    repo: Repository path where Git was invoked.
    said: Error text reported by Git or `git_output()`.
    kind: Failure category (`"missing"`, `"timeout"`, or `"git"`).

  Returns:
    A formatted actionable error message string.
  """
  if kind == "missing":
    return (
        "git is not installed, and the kit needs it to tell what a commit of"
        f" {repo} holds: install git, then run again"
    )
  if kind == "timeout":
    return (
        f"git cannot work in {repo} in time ({said}): run again, and if it"
        " keeps taking this long, look at the repository's size and the disk"
        " it is on"
    )
  return (
      f"git cannot work in {repo} (git says: {git_words(said)}): fix the"
      f" repository so that `git status` works in {repo}, then run again"
  )


@dataclasses.dataclass
class GitRunner:
  """Helper that runs Git commands in `repo` and raises `GitFailed` on error.

  `repo` lies `prefix` below the top of its work tree (`git rev-parse
  --show-prefix`). When `soft=True` is passed to a call, returns `None` instead
  of raising `GitFailed`. `attributes` holds the path to `.git/info/attributes`
  discovered by `work_tree()`.
  """

  repo: pathlib.Path
  prefix: str
  attributes: pathlib.Path | None = None

  def __call__(
      self,
      *args: str,
      data: bytes | None = None,
      ok: tuple[int, ...] = (0,),
      soft: bool = False,
      warnings: list[str] | None = None,
  ) -> bytes | None:
    errors: list[str] = []
    out = git_output(
        self.repo, *args, data=data, ok=ok, errors=errors, warnings=warnings
    )
    if out is None and not soft:
      raise GitFailed(
          errors[0]
          if errors
          else GitSaid(f"git {args[0] if args else ''} failed")
      )
    return out


def could_not_open(
    warnings: list[str], repo: pathlib.Path, git: Callable[..., bytes | None]
) -> list[tuple[str, str]]:
  """Finds non-ignored directories in `repo` that this machine cannot open.

  Triggered when `git ls-files --others` emits a `warning: could not open
  directory` message (after which Git continues with exit code 0, omitting the
  unreadable directory). Excludes directories ignored by Git or handled
  separately by `review()` (`bundle_excluded`, `runs/`, and run folders).

  Args:
    warnings: Stderr warning lines collected from `git ls-files`.
    repo: Root path of the repository to walk.
    git: Callable that runs Git commands in `repo`.

  Returns:
    A list of `(relative_folder_path, reason)` tuples for unreadable folders.
  """
  if not any(w.startswith(COULD_NOT_OPEN) for w in warnings):
    return []
  found = [
      (rel, why)
      for rel, path, st, why in walk(repo)
      if why
      and st is not None
      and stat.S_ISDIR(st.st_mode)
      and rel
      and not (bundle_excluded(rel) or RUNS_FOLDER_RE.fullmatch(rel))
  ]
  ignored = _ignored(git, [f"{rel}/" for rel, _ in found])
  return [(rel, why) for rel, why in found if f"{rel}/" not in ignored]


def _own_repository(path: pathlib.Path) -> bool:
  """Returns whether `path` is the top-level directory of its own Git repo.

  Git records such a directory as a submodule (or refuses to add it if it has no
  commits). A directory with a broken `.git` that Git cannot use is not treated
  as its own repository.

  Args:
    path: Directory path to test.

  Returns:
    True if `git rev-parse --show-toplevel` resolves to `path`.
  """
  top = git_output(path, "rev-parse", "--show-toplevel", timeout=30)
  return (
      top is not None
      and pathlib.Path(os.fsdecode(top.strip())).resolve() == path.resolve()
  )


WORK_TREE_ARGS = (
    "rev-parse",
    "--is-inside-work-tree",
    "--show-prefix",
    "--git-path",
    "info/attributes",
)


def work_tree(repo: pathlib.Path) -> tuple[GitRunner | None, str]:
  """Checks whether `repo` is inside a usable Git working tree.

  Treats `"not a git repository"` (when no `.git` exists at or above `repo` on
  the same filesystem) and `"detected dubious ownership"` (when `.git` is above
  `repo`) as outside Git without error. Any other Git failure returns an
  actionable stop message from `git_stop()`.

  Args:
    repo: Directory path to inspect.

  Returns:
    A `(runner, problem)` tuple where `runner` is a `GitRunner` if `repo` is in
    a usable work tree (else `None`), and `problem` is a non-empty stop message
    if Git failed in a broken repository (else `""`).
  """
  errors: list[str] = []
  out = git_output(repo, *WORK_TREE_ARGS, errors=errors)
  if out is None:
    said = errors[0] if errors else GitSaid("git failed")
    if said.kind == "git" and (
        "not a git repository" in said
        and not _dot_git_above(repo)
        or "detected dubious ownership" in said
        and not os.path.lexists(repo / ".git")
    ):
      return None, ""
    return None, git_stop(repo, said, said.kind)
  # one line each; a folder above may have "\n" in its name, so the prefix is
  # what lies between the first line and the last
  inside, _, rest = os.fsdecode(out).partition("\n")
  if inside != "true":
    return None, ""
  prefix, _, attributes = rest.removesuffix("\n").rpartition("\n")
  return (
      GitRunner(
          repo,
          prefix,
          pathlib.Path(os.path.abspath(repo.resolve() / attributes)),
      ),
      "",
  )


def _dot_git_above(repo: pathlib.Path) -> bool:
  """Returns whether `repo` or an ancestor on the same device contains `.git`.

  Args:
    repo: Starting directory path.

  Returns:
    True if a `.git` entry exists in `repo` or a parent on the same filesystem.
  """
  path = pathlib.Path(os.path.realpath(repo))
  start = lstat(path)
  for folder in (path, *path.parents):
    st = lstat(folder)
    if start is None or st is None or st.st_dev != start.st_dev:
      return False
    if os.path.lexists(folder / ".git"):
      return True
  return False


def read_regular(path: pathlib.Path, cap: int) -> bytes | None:
  """Reads up to `cap + 1` bytes from `path` without following symlinks.

  Opens `path` with `O_NOFOLLOW | O_NONBLOCK` so symlinks and blocking FIFOs are
  never followed or waited on.

  Args:
    path: Filesystem path to read.
    cap: Byte limit; up to `cap + 1` bytes are returned to detect truncation.

  Returns:
    The bytes read if `path` is a regular file, or `None` if it is a directory
    or special file.
  """
  fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
  try:
    if not stat.S_ISREG(os.fstat(fd).st_mode):
      return None
    chunks, left = [], cap + 1
    while left > 0:
      chunk = os.read(fd, min(left, 1 << 16))
      if not chunk:
        break
      chunks.append(chunk)
      left -= len(chunk)
    return b"".join(chunks)
  finally:
    os.close(fd)


def _ignored(git: Callable[..., bytes | None], paths: list[str]) -> set[str]:
  """The paths git ignores (a tracked file is never ignored)."""
  if not paths:
    return set()
  data = b"\x00".join(os.fsencode(p) for p in paths) + b"\x00"
  out = git("check-ignore", "-z", "--stdin", data=data, ok=(0, 1))
  return {os.fsdecode(p) for p in (out or b"").split(b"\x00") if p}


def _stdin_path(path: str) -> bool:
  """Returns whether `path` can be passed literally to `--stdin-paths`.

  `git hash-object --stdin-paths` reads one newline-delimited path per line and
  C-unquotes paths starting with `"` (so `'"a".txt'` would be misread).

  Args:
    path: Relative path string to check.

  Returns:
    True if `path` contains no CR/LF characters and does not start with `"` .
  """
  return "\n" not in path and "\r" not in path and not path.startswith('"')


def _hash_objects(
    git: Callable[..., bytes | None], paths: list[str], prefix: str
) -> dict[str, str]:
  """Computes `git hash-object` blob IDs for `paths` in batch.

  Prepends `prefix` to each path because `--stdin-paths` resolves paths relative
  to the work-tree root. Paths rejected by `_stdin_path()` are omitted so the
  caller can hash them individually.

  Args:
    git: Callable that executes Git commands in the repository.
    paths: Paths relative to `git`'s working directory.
    prefix: Work-tree prefix ending in `"/"` (or `""` at the root).

  Returns:
    A mapping from path in `paths` to its 40-character SHA-1 blob ID.

  Raises:
    GitFailed: If `git hash-object` fails or returns a mismatched count of IDs.
  """
  names = [p for p in paths if _stdin_path(prefix + p)]
  if not names:
    return {}
  out = git(
      "hash-object",
      "--stdin-paths",
      data=b"\n".join(os.fsencode(prefix + p) for p in names) + b"\n",
  )
  ids = (out or b"").decode("ascii", "replace").split()
  if len(ids) != len(names):
    raise GitFailed(
        f"git hash-object gave {len(ids)} ids for {len(names)} files"
    )
  return dict(zip(names, ids))


def _pointer(path: pathlib.Path) -> bool:
  try:
    with open(path, "rb") as fh:
      return fh.read(len(LFS_POINTER)) == LFS_POINTER
  except OSError:
    return False


def _byte_length(text: str) -> int:
  """Returns the byte length of `text`, counting surrogate escapes as 1 byte.

  `os.fsdecode` represents raw non-UTF-8 bytes as surrogate escapes (`0xDC80` to
  `0xDCFF`), each corresponding to one byte on disk. Any other lone surrogate is
  counted as the 3 bytes produced by `surrogatepass` UTF-8 encoding.

  Args:
    text: String to measure.

  Returns:
    The length of `text` in bytes.
  """
  try:
    return len(text.encode("utf-8", "surrogateescape"))
  except UnicodeEncodeError:
    return len(text.encode("utf-8", "surrogatepass"))


def path_problem(path: str) -> str | None:
  """Returns why `path` is invalid as a container file path, or `None` if valid.

  Non-UTF-8 names are reported first regardless of length, because GitHub's tree
  API replaces each invalid byte with a 3-byte `U+FFFD` replacement character.

  Args:
    path: Relative path string to validate.

  Returns:
    A short problem description string, or `None` if `path` is valid.
  """
  try:
    path.encode("utf-8")
  except UnicodeEncodeError:
    return "is not UTF-8"
  # how GitHub's tree API shows the bytes of a name that is not UTF-8
  if "\ufffd" in path:
    return "is not UTF-8"
  if path.count("/") > MAX_PATH_FOLDERS:
    return f"is nested deeper than {MAX_PATH_FOLDERS} folders"
  if _byte_length(path) > MAX_PATH_BYTES:
    return f"is longer than {MAX_PATH_BYTES:,} bytes"
  if any(_byte_length(s) > MAX_NAME_BYTES for s in path.split("/")):
    return (
        f"has a name longer than {MAX_NAME_BYTES} bytes, which no file system"
        " holds"
    )
  if any(s in ("", ".", "..") for s in path.split("/")):
    return "has an empty, '.' or '..' segment"
  if "\\" in path:
    return "has a backslash"
  if CONTROL_RE.search(path):
    return "has a control character"
  return None


def junk(rel: str) -> bool:
  """Returns whether `rel` is an OS metadata file or folder (e.g., `.DS_Store`).

  Args:
    rel: Relative path to check.

  Returns:
    True if any segment of `rel` is in `JUNK` or starts with `"._"`.
  """
  return any(s in JUNK or s.startswith("._") for s in rel.split("/"))


def deleted_lines(data: bytes) -> list[str]:
  """Parses non-empty, non-comment paths from `deleted.txt` bytes.

  Splits on LF (stripping a trailing CR for Windows files) rather than
  `str.splitlines()` so characters like `U+2028` inside filenames are preserved,
  and preserves trailing spaces.

  Args:
    data: Raw bytes of a `deleted.txt` file.

  Returns:
    A list of path strings listed in `data`.
  """
  lines = (
      line[:-1] if line.endswith("\r") else line
      for line in data.decode("utf-8", "replace").split("\n")
  )
  return [line for line in lines if line and not line.startswith("#")]


def git_own(rel: str) -> bool:
  """Returns whether `rel` is a Git metadata file or inside a `.git` directory.

  Files like `.gitignore`, `.gitattributes`, `.gitmodules`, and `.git/` are
  omitted during export and ignored by `review()` so they cannot alter how Git
  commits `artifacts/`.

  Args:
    rel: Relative path to check.

  Returns:
    True if `rel` is or lies under `.git` or is named in `GIT_OWN`.
  """
  parts = rel.split("/")
  return ".git" in parts or parts[-1] in GIT_OWN


def printable(text: str) -> str:
  r"""Formats `text` as valid UTF-8, escaping lone surrogates as hex escapes.

  `os.fsdecode` stores non-UTF-8 filename bytes as `0xDC80`..`0xDCFF`
  surrogates, which are formatted as `\xNN`; any other lone surrogate becomes
  `\uXXXX`.

  Args:
    text: String that may contain surrogate characters.

  Returns:
    A valid UTF-8 string safe for printing and JSON serialization.
  """
  try:
    text.encode("utf-8")
    return text
  except UnicodeEncodeError:
    out: list[str] = []
    for c in text:
      code = ord(c)
      if 0xDC80 <= code <= 0xDCFF:
        out.append(f"\\x{code - 0xDC00:02x}")
      elif 0xD800 <= code <= 0xDFFF:
        out.append(f"\\u{code:04x}")
      else:
        out.append(c)
    return "".join(out)


def _clip(path: str, keep: int = 200) -> str:
  """Truncates `path` in the middle if it exceeds `keep` characters.

  Args:
    path: Path string to format for a diagnostic message.
    keep: Maximum number of characters to retain from the start and end.

  Returns:
    The original `path` if within `keep`, or `"<start>...<end>"`.
  """
  return (
      path if len(path) <= keep else f"{path[:keep // 2]}...{path[-keep // 2:]}"
  )


def shown(paths: list[str]) -> str:
  head = ", ".join(printable(_clip(p)) for p in paths[:SHOWN])
  return head + (
      f" and {len(paths) - SHOWN} more" if len(paths) > SHOWN else ""
  )


def _printable_messages(rv: "Review") -> None:
  """Sanitizes all problem, warning, and note strings in `rv` via `printable()`.

  Args:
    rv: `Review` object whose message lists are updated in place.
  """
  rv.problems[:] = [printable(m) for m in rv.problems]
  rv.warnings[:] = [printable(m) for m in rv.warnings]
  rv.notes[:] = [printable(m) for m in rv.notes]
  for r in rv.tasks.values():
    r.problems[:] = [printable(m) for m in r.problems]
    r.warnings[:] = [printable(m) for m in r.warnings]


def _mib(n: int) -> str:
  """Formats `n` bytes in MiB, rounding up non-exact values to one decimal.

  Rounding up ensures a byte count just above a limit never formats as the limit
  itself (e.g., 95 MiB + 1 byte formats as `"95.1 MiB"`).

  Args:
    n: Size in bytes.

  Returns:
    A human-readable MiB string such as `"95 MiB"` or `"95.1 MiB"`.
  """
  if n % (1 << 20) == 0:
    return f"{n >> 20} MiB"
  return f"{-(-n * 10 // (1 << 20)) / 10:.1f} MiB"


@dataclasses.dataclass
class TaskReview:
  """Review outcome and discovered artifact entries for a single dev task."""

  id: str
  status: str = MISSING  # ok | empty | refused | missing
  folder: str | None = None  # the repository folder read for this task
  # container path -> its entry
  files: dict[str, Entry] = dataclasses.field(default_factory=dict)
  message: Entry | None = None
  deleted: Entry | None = None
  trajectory: Entry | None = None
  problems: list[str] = dataclasses.field(
      default_factory=list
  )  # why it is refused
  warnings: list[str] = dataclasses.field(default_factory=list)
  bytes: int = 0
  # the manifest's reward: the participant's own grade
  claimed: float | None = None
  edited: bool = False  # its files differ from what the manifest recorded

  def as_dict(self) -> dict[str, Any]:
    return {
        "id": self.id,
        "status": self.status,
        "folder": self.folder and printable(self.folder),
        "files": len(self.files),
        "bytes": self.bytes,
        "final_message": self.message is not None,
        "problems": self.problems,
        "warnings": self.warnings,
        "claimed": self.claimed,
        "edited": self.edited,
    }


@dataclasses.dataclass
class Review:
  """Overall review result for a submission's dev-set artifacts."""

  root: str | None
  tasks: dict[str, TaskReview]
  problems: list[str] = dataclasses.field(
      default_factory=list
  )  # the submission is refused
  warnings: list[str] = dataclasses.field(default_factory=list)
  # worth knowing, nothing to fix
  notes: list[str] = dataclasses.field(default_factory=list)
  # manifest, dev_set, harness, edited (_manifest)
  provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
  # the kit stopped (what this machine cannot read, git failing): nothing
  # reviewed
  unreadable: bool = False

  @property
  def absent(self) -> bool:
    """Returns whether no artifacts folder was found in the repository.

    Allows `kit check` to distinguish a repository that has not yet exported any
    dev-set runs from one whose artifacts violate a submission rule.
    """
    return (
        self.root is None
        and len(self.problems) == 1
        and self.problems[0].startswith(NO_FOLDER)
    )

  @property
  def accepted(self) -> bool:
    return not self.problems

  @property
  def usable(self) -> list[str]:
    return [t for t, r in self.tasks.items() if r.status == OK]

  def counts(self) -> dict[str, int]:
    out = {OK: 0, EMPTY: 0, REFUSED: 0, MISSING: 0}
    for r in self.tasks.values():
      out[r.status] += 1
    return out

  def as_dict(self) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "root": self.root and printable(self.root),
        "accepted": self.accepted,
        "absent": self.absent,
        "unreadable": self.unreadable,
        "problems": self.problems,
        "warnings": self.warnings,
        "notes": self.notes,
        "provenance": self.provenance,
        "counts": self.counts(),
        "tasks": [r.as_dict() for r in self.tasks.values()],
    }

  def lines(self) -> list[str]:
    """The review as text, the way `kit artifacts` prints it."""
    # nothing was reviewed: no count of tasks, which would read as a verdict
    if self.unreadable:
      return (
          ["the kit cannot review the artifacts on this machine:"]
          + [f"  STOP     {p}" for p in self.problems]
          + [f"  warning  {w}" for w in self.warnings]
      )
    c = self.counts()
    where = f"{printable(self.root)}/" if self.root else "no artifacts folder"
    out = [
        f"{where}: {c[OK]} of {len(self.tasks)} dev tasks usable"
        + "".join(f", {c[k]} {k}" for k in (EMPTY, REFUSED, MISSING) if c[k])
    ]
    out += [f"  REFUSED  {p}" for p in self.problems]
    out += [f"  warning  {w}" for w in self.warnings]
    out += [f"  note     {n}" for n in self.notes]
    for r in self.tasks.values():
      size = (
          f"{len(r.files)} file(s), {r.bytes / 2**20:.1f} MiB"
          if r.status != MISSING
          else ""
      )
      out.append(f"  {r.status:<8} {r.id}" + (f"  ({size})" if size else ""))
      out += [f"           refused: {p}" for p in r.problems]
      out += [f"           warning: {w}" for w in r.warnings]
    return out


def _top(path: str) -> str:
  return path.split("/", 1)[0]


NO_FOLDER = f"no {ROOT}/ folder"
NO_ARTIFACTS = (
    f"{NO_FOLDER}: run `uv run python eval/run.py <your entry>`, which writes"
    " it, then commit it (docs/artifacts.md)"
)


def find_root(entries: list[Entry]) -> tuple[str | None, list[str], list[str]]:
  """Locates the artifacts root folder in `entries`, with warnings and problems.

  Args:
    entries: Repository entries to inspect.

  Returns:
    A `(root_folder, warnings, problems)` tuple where `root_folder` is the
    selected relative folder path or `None` if no valid root was found.
  """
  warnings: list[str] = []
  paths = [e.path.rstrip("/") for e in entries]
  exact = [e for e in entries if e.path.rstrip("/") == ROOT]
  if exact and exact[0].mode != "tree":
    return None, [], [f"{ROOT} is a {exact[0].kind}; it must be a folder"]
  tops, aliases, runs = _candidates(entries)
  root = None
  if ROOT in tops:
    root = ROOT
  else:
    if len(aliases) > 1:
      return (
          None,
          [],
          [
              f"several folders could hold the artifacts ({shown(aliases)}):"
              f" keep one, named {ROOT}/"
          ],
      )
    if aliases:
      root = aliases[0]
      warnings.append(f"read {root}/ as {ROOT}/: rename it to {ROOT}/")
    elif len(runs) == 1:
      root = runs[0]
      warnings.append(
          f"read {root}/ as {ROOT}/: move it to {ROOT}/ at the top of the"
          " repository (and leave runs/ out of the commit)"
      )
    elif runs:
      return (
          None,
          [],
          [
              f"several runs/ folders hold artifacts ({shown(runs)}): copy the"
              f" one you mean to {ROOT}/ with `python -m kit artifacts <entry>"
              " --from <run>`"
          ],
      )
  # an archive at the top of the repository, or directly in the artifacts
  # folder, in place of task folders (one inside a task folder is a deliverable
  # like any other: some tasks ask for a zip)
  archives = [
      p
      for p in paths
      if ARCHIVE_RE.fullmatch(posixpath.basename(p))
      and ("/" not in p or (root is not None and posixpath.dirname(p) == root))
  ]
  if root is None:
    if archives:
      return (
          None,
          [],
          [
              f"{NO_FOLDER}, but {shown(archives)}: if that holds your"
              f" artifacts, commit the {ROOT}/ folder itself, unpacked (the"
              " server does not open archives); if not, run `uv run python"
              " eval/run.py <your entry>`, which writes it, then commit it"
          ],
      )
    return None, [], [NO_ARTIFACTS]
  archives = [p for p in archives if posixpath.dirname(p) == root]
  if archives:
    warnings.append(
        f"{shown(archives)}: an archive, ignored; the server reads only"
        " unpacked files"
    )
  return root, warnings, []


def _candidates(entries: list[Entry]) -> tuple[list[str], list[str], list[str]]:
  """Lists candidate artifact folders for `find_root`.

  Args:
    entries: Repository entries.

  Returns:
    A tuple `(tops, aliases, runs)` of top-level folders, artifacts-like
    top-level folders (including `artifacts/`), and `runs/<run>/artifacts`
    folders.
  """
  folders = {e.path.rstrip("/") for e in entries if e.mode == "tree"}
  paths = [e.path.rstrip("/") for e in entries]
  tops = sorted({_top(p) for p in paths if "/" in p or p in folders})
  runs = sorted({m.group(0) for p in paths if (m := RUN_ROOT_RE.match(p))})
  return tops, [t for t in tops if ROOT_ALIAS_RE.fullmatch(t)], runs


def _holds_artifacts(
    kids: dict[str, list[tuple[str, Entry]]], dev: dict[str, DevTask]
) -> bool:
  """Returns whether `kids` or its wrapper folder holds artifacts.

  Checks whether a folder whose children are `kids` (`_group`), or the one
  wrapper folder inside it that `_unwrap` would read, holds a dev task's folder
  or a `manifest.json`.

  Args:
    kids: Grouped child entries under the folder.
    dev: Mapping of task ID to `DevTask`.

  Returns:
    True if the folder holds a dev task folder or `manifest.json`.
  """
  if _wrapper(kids, dev) is not None:
    return True
  return MANIFEST in kids or any(
      task_for(name, dev)[0]
      for name, v in kids.items()
      if any(rest for rest, _ in v)
  )


def _other_roots(
    entries: list[Entry], root: str, dev: dict[str, DevTask]
) -> list[str]:
  """Returns candidate folders besides `root` that hold artifacts.

  A harness's own code package named `artifact/` holds neither dev tasks nor a
  manifest and is not included.

  Args:
    entries: Repository entries.
    root: Chosen artifacts root folder.
    dev: Mapping of task ID to `DevTask`.

  Returns:
    A list of other folder names (with trailing `/`) holding artifacts.
  """
  _, aliases, runs = _candidates(entries)
  others = {c for c in aliases + runs if c != root}
  if not others:
    return []
  below: dict[str, list[tuple[str, Entry]]] = {}
  for e in entries:
    p = e.path.rstrip("/")
    if e.mode == "tree" or "/" not in p:
      continue
    folder = _top(p)
    if folder not in others:
      m = RUN_ROOT_RE.match(p)
      folder = m.group(0) if m else ""
    if folder in others:
      below.setdefault(folder, []).append((p[len(folder) + 1 :], e))
  return [
      f"{c}/"
      for c in aliases + runs
      if c in below and _holds_artifacts(_group(below[c]), dev)
  ]


def reviewed_root(paths: Iterable[str]) -> str | None:
  """Returns the folder `find_root()` picks for a repository with `paths`.

  `review()` may then read a wrapper inside it. The builds count every other
  folder they leave out of the bundle (`bundle_excluded`) toward the
  repository's size limit, and not this one, which `review()` limits as a whole.

  Args:
    paths: Repository file paths.

  Returns:
    The chosen root folder name, or None if none is found.
  """
  root, _, _ = find_root([Entry(p, "100644") for p in paths])
  return root


def _spelled(name: str, dev: dict[str, DevTask]) -> tuple[str | None, str]:
  if name in dev:
    return name, "exact"
  spelled = re.sub(r"[\s_]+", "-", name.strip().lower())
  return (spelled, "spelling") if spelled in dev else (None, "")


def task_for(name: str, dev: dict[str, DevTask]) -> tuple[str | None, str]:
  """Resolves the dev task a folder name stands for and how it was matched.

  A trial folder's name is the task's and one suffix
  (`jobshop-scheduler__a1B2c3D`), taken off once, at the last `"__"`; a name
  over `MAX_NAME_BYTES` stands for no task.

  Args:
    name: Folder name to match.
    dev: Mapping of task ID to `DevTask`.

  Returns:
    A tuple `(task_id, how)` where `how` is `"exact"`, `"spelling"`, `"trial"`,
    or `""` (with `task_id` None) when none matches.
  """
  if _byte_length(name) > MAX_NAME_BYTES:
    return None, ""
  task, how = _spelled(name, dev)
  if task:
    return task, how
  head, sep, tail = name.rpartition("__")
  if head and sep and TRIAL_SUFFIX_RE.fullmatch(tail):
    task, _ = _spelled(head, dev)
    if task:
      return task, "trial"
  return None, ""


def _group(
    items: Iterable[tuple[str, Entry]],
) -> dict[str, list[tuple[str, Entry]]]:
  """Groups paths below one folder by their first path segment.

  Args:
    items: Pairs of `(relative_path, entry)` below a folder.

  Returns:
    A dictionary `{name: [(path_below_name, entry)]}`.
  """
  out: dict[str, list[tuple[str, Entry]]] = {}
  for below, e in items:
    head, _, rest = below.partition("/")
    out.setdefault(head, []).append((rest, e))
  return out


def _children(
    entries: list[Entry], root: str
) -> dict[str, list[tuple[str, Entry]]]:
  """Groups entries under `root` by the first segment below it.

  Args:
    entries: Repository entries.
    root: Root folder path.

  Returns:
    A dictionary `{name: [(path_below_name, entry)]}`.
  """
  prefix = root + "/"
  return _group(
      (p[len(prefix) :], e)
      for e in entries
      if e.mode != "tree" and (p := e.path.rstrip("/")).startswith(prefix)
  )


def _wrapper(
    kids: dict[str, list[tuple[str, Entry]]], dev: dict[str, DevTask]
) -> str | None:
  """Finds the single wrapper folder inside `kids` that holds task folders.

  Matches layouts such as `artifacts/my-harness/<task-id>/`. Returns None when
  the folder holds task folders itself, or when there is no single such wrapper.

  Args:
    kids: Grouped child entries (`_group`).
    dev: Mapping of task ID to `DevTask`.

  Returns:
    The wrapper folder name, or None.
  """
  kids = {k: v for k, v in kids.items() if not junk(k)}
  folders = [k for k, v in kids.items() if any(rest for rest, _ in v)]
  if any(task_for(k, dev)[0] for k in folders) or len(folders) != 1:
    return None
  inner = _group((rest, e) for rest, e in kids[folders[0]] if rest)
  if any(
      task_for(k, dev)[0]
      for k, v in inner.items()
      if any(rest for rest, _ in v)
  ):
    return folders[0]
  return None


def _unwrap(
    entries: list[Entry], root: str, dev: dict[str, DevTask]
) -> tuple[str, list[str]]:
  """Unwraps a single nested wrapper folder inside `root` if present.

  Args:
    entries: Repository entries.
    root: Candidate artifacts root folder.
    dev: Mapping of task ID to `DevTask`.

  Returns:
    A tuple `(effective_root, warnings)` for `root` or the wrapper folder inside
    it (`artifacts/my-harness/<task-id>/`).
  """
  wrapper = _wrapper(_children(entries, root), dev)
  if wrapper is None:
    return root, []
  inner = f"{root}/{wrapper}"
  return inner, [
      f"read {inner}/ as {ROOT}/: put the task folders directly in {ROOT}/"
  ]


def _place(rel: str, task: DevTask) -> tuple[str | None, str]:
  """Resolves where a file at `rel` inside a task folder goes in the container.

  Args:
    rel: Relative path inside the task folder.
    task: `DevTask` specification.

  Returns:
    A tuple `(container_path, how)` where `how` is `"files"` (the layout),
    `"path"` (a container path without `files/`), `"name"` (the one deliverable
    with that name), or `""` (with `container_path` None) when unplaced.
  """
  if rel.startswith(FILES + "/"):
    where = "/" + rel[len(FILES) + 1 :]
    if task.covers(where) or "/" in where[1:]:
      return where, "files"
    # files/init.py: a bare name, matched to a deliverable below
    rel = where[1:]
  firsts = {d.strip("/").split("/", 1)[0] for d in task.deliverables}
  if rel.split("/", 1)[0] in firsts:
    return "/" + rel, "path"
  if "/" not in rel:
    named = [d for d in task.deliverable_files if posixpath.basename(d) == rel]
    if len(named) == 1:
      return named[0], "name"
  return None, ""


def _review_task(
    task: DevTask, folder: str, items: list[tuple[str, Entry]]
) -> TaskReview:
  """Reviews the files and metadata inside a single task folder.

  Args:
    task: `DevTask` specification.
    folder: Repository path to the task folder.
    items: Pairs of `(relative_path, entry)` inside the task folder.

  Returns:
    A `TaskReview` with placed files, status, warnings, and problems.
  """
  r = TaskReview(id=task.id, folder=folder)
  placed: dict[str, tuple[str, Entry]] = {}  # container path -> (how, entry)
  messages: dict[str, Entry] = {}
  outside, unplaced, special, pointers, copies, own = [], [], [], [], [], []
  # placed by a fallback, in order (a later layout copy removes one)
  by_path: dict[str, None] = {}
  by_name: dict[str, None] = {}
  # the first SHOWN paths that cannot stand for a file, with why, and a count of
  # the rest: a commit may hold thousands
  bad: list[str] = []
  more_bad = 0
  for rel, e in items:
    if junk(rel):
      continue
    if git_own(rel):
      own.append(rel)
      continue
    problem = path_problem(rel)
    if problem:
      if len(bad) < SHOWN:
        bad.append(f"{_clip(f'{folder}/{rel}')} {problem}")
      else:
        more_bad += 1
      continue
    if not e.is_file:
      special.append(f"{folder}/{rel} ({e.kind})")
      continue
    if e.lfs:
      pointers.append(f"{folder}/{rel}")
      continue
    if rel == FINAL_MESSAGE or rel in MESSAGE_ALIASES:
      messages[rel] = e
      continue
    if rel == DELETED:
      r.deleted = e
      continue
    if rel == TRAJECTORY:
      if (e.size or 0) > MAX_TRAJECTORY_BYTES:
        r.warnings.append(
            f"{TRAJECTORY} is over {_mib(MAX_TRAJECTORY_BYTES)}: ignored"
        )
      else:
        r.trajectory = e
      continue
    where, how = _place(rel, task)
    if where is None and rel.startswith(FILES + "/"):
      # under files/, at a path no deliverable covers
      outside.append("/" + rel[len(FILES) + 1 :])
      continue
    if where is None:
      unplaced.append(rel)
      continue
    if not task.covers(where):
      outside.append(where)
      continue
    if where in placed:
      # the layout's own copy wins over a fallback's, and the first fallback
      # over a later one
      if placed[where][0] == "files" or how != "files":
        copies.append(rel)
        continue
      earlier = placed[where][1].path[len(folder) + 1 :]
      copies.append(earlier)
      by_path.pop(earlier, None)
      by_name.pop(earlier, None)
    placed[where] = (how, e)
    if how == "path":
      by_path[rel] = None
    elif how == "name":
      by_name[rel] = None
  r.problems += bad
  if more_bad:
    r.problems.append(
        f"and {more_bad} more paths that cannot stand for a file in the task"
        " container"
    )
  if special:
    r.problems.append(f"only regular files are graded: {shown(special)}")
  if pointers:
    r.problems.append(
        f"Git LFS pointers, not the files: {shown(pointers)}; the server clones"
        " without LFS, so commit the files themselves (where the file on disk"
        " is whole, git's index still holds the pointer: `git rm --cached` it,"
        " then `git add` it again)"
    )
  if messages:
    pick = (
        FINAL_MESSAGE
        if FINAL_MESSAGE in messages
        else next(m for m in MESSAGE_ALIASES if m in messages)
    )
    r.message = messages[pick]
    if pick != FINAL_MESSAGE:
      r.warnings.append(f"read {pick} as {FINAL_MESSAGE}: rename it")
    extra = [m for m in messages if m != pick]
    if extra:
      r.warnings.append(f"{shown(extra)}: ignored, {pick} is the final message")
    if (r.message.size or 0) > MAX_MESSAGE_BYTES:
      r.warnings.append(
          f"the final message is over {_mib(MAX_MESSAGE_BYTES)}: only its first"
          f" {_mib(MAX_MESSAGE_BYTES)} is graded"
      )
  if by_path:
    r.warnings.append(
        f"{shown(list(by_path))}: not under {FILES}/, placed by its path; the"
        f" layout is {FILES}/<container path>"
    )
  if by_name:
    r.warnings.append(
        f"{shown(list(by_name))}: placed as the deliverable with that name; the"
        f" layout is {FILES}/<container path>"
    )
  if copies:
    r.warnings.append(
        f"{shown(copies)}: a second copy of a file placed already, ignored (a"
        f" copy under {FILES}/ comes first)"
    )
  if own:
    r.warnings.append(
        f"{shown(own)}: git's own files (.git/, .gitignore, .gitattributes,"
        " .gitmodules) are never graded, ignored"
    )
  if unplaced:
    r.warnings.append(
        f"{shown(unplaced)}: not under {FILES}/ and not a deliverable, ignored"
    )
    if not placed and any(u.split("/", 1)[0] in TRIAL_PARTS for u in unplaced):
      r.warnings.append(
          "this looks like a run's log folder (agent/, verifier/, result.json),"
          " not artifacts: `python -m kit artifacts <entry> --from <run dir>`"
          " copies a run's artifacts in"
      )
  if outside:
    r.warnings.append(
        f"{shown(sorted(outside))}: outside this task's deliverables "
        f"({', '.join(task.deliverables)}), ignored"
    )
  r.files = {p: e for p, (_, e) in sorted(placed.items())}
  # the files and every folder above them: Out/x and out/y share one folder on a
  # case-insensitive file system
  folders = folders_of(r.files)
  folded: dict[str, set[str]] = {}
  for q in (*folders, *r.files):
    folded.setdefault(q.casefold(), set()).add(q)
  clashes = sorted(
      " and ".join(sorted(v)) for v in folded.values() if len(v) > 1
  )
  if clashes:
    r.problems.append(
        f"paths that differ only in letter case: {shown(clashes)}"
    )
  both = sorted(p for p in r.files if p in folders)
  if both:
    r.problems.append(f"both a file and a folder holding files: {shown(both)}")
  big = [
      f"{p} ({_mib(e.size or 0)})"
      for p, e in r.files.items()
      if (e.size or 0) > MAX_FILE_BYTES
  ]
  if big:
    r.problems.append(f"over {_mib(MAX_FILE_BYTES)}: {shown(big)}")
  parts = list(r.files.values()) + [
      e for e in (r.message, r.deleted, r.trajectory) if e is not None
  ]
  r.bytes = sum(e.size or 0 for e in parts)
  if len(r.files) > MAX_TASK_FILES:
    r.problems.append(f"{len(r.files)} files, more than {MAX_TASK_FILES}")
  if r.bytes > MAX_TASK_BYTES:
    r.problems.append(
        f"{_mib(r.bytes)} in all, more than {_mib(MAX_TASK_BYTES)}"
    )
  if r.problems:
    r.status = REFUSED
  elif not r.files and not (r.message and r.message.size != 0):
    r.status = EMPTY
    r.problems.append(
        f"nothing to grade: no file under {FILES}/ and no {FINAL_MESSAGE}"
    )
  else:
    r.status = OK
  if (
      r.status == OK
      and task.final_message
      and (r.message is None or r.message.size == 0)
  ):
    r.warnings.append(
        f"no {FINAL_MESSAGE}, which this task's grader reads"
        if r.message is None
        else f"{FINAL_MESSAGE} is empty, and this task's grader reads it"
    )
  return r


def folders_of(paths: Iterable[str]) -> set[str]:
  """Returns every folder above container `paths` (`/a/b/c` -> `/a`, `/a/b`)."""
  found: set[str] = set()
  for path in paths:
    cut = path.rfind("/")
    while cut > 0:
      folder = path[:cut]
      if folder in found:
        break
      found.add(folder)
      cut = path.rfind("/", 0, cut)
  return found


def unreadable_fix(paths: Iterable[pathlib.Path]) -> str:
  """Builds a shell remediation hint for unreadable `paths`.

  Produces one command for all `paths`, each made absolute so that it works
  wherever it is run. `--` comes before the operands, so that no path is read as
  an option by GNU and BSD `chmod` and `chown` alike. The `chmod` gives write
  access too, since a later export replaces what a folder holds.

  Args:
    paths: Unreadable filesystem paths.

  Returns:
    A human-readable remediation string with `chmod` and `chown` commands.
  """
  targets = sorted({p.absolute() for p in paths})
  q = " ".join(shlex.quote(str(p)) for p in targets)
  them = "it" if len(targets) == 1 else "them"
  return (
      f"make {them} readable (`chmod -R -- u+rwX {q}`, or `sudo chown -R --"
      f' "$USER" {q}` if a container left {them} owned by root), then run again'
  )


def unreadable_stop(
    base: pathlib.Path,
    blocked: list[tuple[str, str]],
    what: str,
    commit: bool = True,
) -> str:
  """Builds the error message when a kit command cannot read paths in `base`.

  The fix names each top-level file or folder of `base` that holds a blocked
  path, never anything above `base`.

  Args:
    base: Base folder the kit command works on.
    blocked: Pairs of `(relative_path, reason)` where `relative_path` is `""`
      for `base` itself.
    what: Description of the action that could not proceed.
    commit: Whether to add a note that the server reads the commit (true of an
      entry's files and not of a run's).

  Returns:
    A printable error message describing the blocked paths and remediation.
  """
  listed = shown([f"{rel or base} ({why})" for rel, why in blocked])
  fix = unreadable_fix(base / rel.split("/", 1)[0] for rel, _ in blocked)
  return printable(
      f"this machine cannot read {listed}, so {what}: {fix}"
      + (" (the server reads your commit, not this disk)" if commit else "")
  )


def unreadable_under(
    top: pathlib.Path, rel: str = ROOT
) -> list[tuple[str, str]]:
  """Lists paths that this machine cannot read under `top`.

  Includes folders it cannot list or enter, entries it cannot look at, and
  regular files it cannot open.

  Args:
    top: Filesystem path to the folder at `rel`.
    rel: Relative path prefix in the folder the kit works on.

  Returns:
    A list of `(path, reason)` tuples for unreadable entries.
  """
  out = []
  for path_rel, path, st, why in walk(top, rel, depth=DEPTH):
    if (
        not why
        and st is not None
        and stat.S_ISREG(st.st_mode)
        and not os.access(path, os.R_OK)
    ):
      why = os.strerror(errno.EACCES)
    if why and why != TOO_DEEP:
      out.append((path_rel, why))
  return out


def shell_path(path: str) -> str:
  """Formats `path` as the trailing argument of a printed git command.

  Returns `path` as written when it holds only letters, digits, and `._/-`, or
  after `--`, quoted and marked `:(literal)`, so that neither the shell nor
  git's pathspec rules read it as anything but itself.

  Args:
    path: Repository-relative path string.

  Returns:
    The shell-safe git path argument string.
  """
  return (
      path
      if re.fullmatch(r"[A-Za-z0-9._/-]+", path)
      else "-- " + shlex.quote(":(literal)" + path)
  )


def attributes_files(entries: list[Entry], root: str) -> list[Entry]:
  """Returns `.gitattributes` entries that can apply to paths under `root`.

  Includes the top level's `.gitattributes`, those of folders above `root`, and
  those inside `root`. The submit page reads these and no others, as `review()`
  does.

  Args:
    entries: Repository entries.
    root: Artifacts root folder path.

  Returns:
    A list of matching `.gitattributes` `Entry` objects.
  """
  above = {
      posixpath.join(a, ATTRIBUTES) if a else ATTRIBUTES
      for a in ["", *_folders_above(root)]
  }
  return [
      e
      for e in entries
      if e.is_file
      and posixpath.basename(e.path) == ATTRIBUTES
      and (e.path in above or e.path.startswith(root + "/"))
  ]


def _folders_above(path: str) -> list[str]:
  """Returns parent folders of `path` (`a/b/c` -> `["a", "a/b"]`)."""
  parts = path.split("/")
  return ["/".join(parts[:i]) for i in range(1, len(parts))]


LFS_CHECK_SEC = 30
# The most .gitattributes rules read from one file, and checked in all. Past
# these they are not checked (a warning, not a refusal), so that no repository
# can make git work long.
LFS_RULES_FILE_BYTES = 64 << 10
LFS_RULES_TOTAL_BYTES = 256 << 10
LFS_UNCHECKED = (
    "the Git LFS rules could not be checked here ({}); a Git LFS pointer is"
    " refused for its task when the server reads the files"
)
LFS_TOO_LONG = (
    "{}: too long to check for Git LFS rules (over"
    f" {LFS_RULES_FILE_BYTES >> 10} KiB), not read; a Git LFS pointer is"
    " refused for its task when the server reads the files"
)


def _lfs_review(
    paths: list[str],
    files: list[Entry],
    read: Callable[[str], bytes | None] | None,
    info: bytes | None = None,
) -> tuple[list[str] | None, str, list[str]]:
  """Runs `review()`'s Git LFS check over `.gitattributes` `files` and `info`.

  A file past `LFS_RULES_FILE_BYTES` is left out and named in the third return
  value; the others are still checked. When any of them mentions `lfs` and
  together they pass `LFS_RULES_TOTAL_BYTES`, the result is `(None, why, ...)`.

  Args:
    paths: Repository paths to check for LFS tracking.
    files: `.gitattributes` entries (`attributes_files`).
    read: Callback returning file bytes for a repository path, or None.
    info: Optional `.git/info/attributes` bytes.

  Returns:
    A tuple `(tracked_paths, reason, skipped_files)`.
  """
  attributes, skipped = {}, []
  for e in files:
    if (e.size or 0) > LFS_RULES_FILE_BYTES:
      skipped.append(e.path)
      continue
    data = read(e.path) if read is not None else None
    if data and len(data) > LFS_RULES_FILE_BYTES:
      skipped.append(e.path)
    elif data:
      attributes[e.path] = data
  if info is not None and len(info) > LFS_RULES_FILE_BYTES:
    skipped.append(".git/info/attributes")
    info = None
  named = any(b"lfs" in data for data in [*attributes.values(), info or b""])
  if (
      named
      and sum(map(len, attributes.values())) + len(info or b"")
      > LFS_RULES_TOTAL_BYTES
  ):
    return (
        None,
        (
            "its .gitattributes rules come to over"
            f" {LFS_RULES_TOTAL_BYTES >> 10} KiB"
        ),
        skipped,
    )
  tracked, why = lfs_check(paths, attributes, info)
  return tracked, why, skipped


def lfs_check(
    paths: list[str], attributes: dict[str, bytes], info: bytes | None = None
) -> tuple[list[str] | None, str]:
  """Returns paths among `paths` that Git LFS would store according to git.

  Runs `git check-attr filter` in a fresh repository holding only the
  `.gitattributes` files in `attributes` and `info` as `.git/info/attributes`,
  so that no system, global, or user attributes reach it.

  Args:
    paths: Repository paths to check.
    attributes: Mapping of repository `.gitattributes` path to raw bytes.
    info: Optional `.git/info/attributes` content bytes.

  Returns:
    A tuple `(tracked_paths, reason)` where `tracked_paths` is None and `reason`
    explains why when git is missing, takes too long, or fails.
  """
  if not paths or not any(
      b"lfs" in data for data in [*attributes.values(), info or b""]
  ):
    # no rule names lfs, so none can set filter=lfs (not even through a
    # macro)
    return [], ""
  with tempfile.TemporaryDirectory(prefix="plebench-attributes-") as tmp:
    work = pathlib.Path(tmp) / "repo"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": tmp,
        "XDG_CONFIG_HOME": tmp,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "LC_ALL": "C",
        "LANGUAGE": "C",
    }
    try:
      for rel, data in attributes.items():
        if (
            path_problem(rel)
            or posixpath.basename(rel) != ATTRIBUTES
            or ".git" in rel.split("/")[:-1]
        ):
          continue
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
      work.mkdir(exist_ok=True)
      subprocess.run(
          ["git", "init", "-q", "--template=", str(work)],
          env=env,
          capture_output=True,
          timeout=LFS_CHECK_SEC,
          check=True,
      )
      if info is not None:
        (work / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (work / ".git" / "info" / "attributes").write_bytes(info)
      p = subprocess.run(
          ["git", "-C", str(work), "check-attr", "-z", "--stdin", "filter"],
          env=env,
          input=b"\x00".join(os.fsencode(x) for x in paths) + b"\x00",
          capture_output=True,
          timeout=LFS_CHECK_SEC,
          check=True,
      )
    except FileNotFoundError:
      return None, "git is not available"
    except subprocess.TimeoutExpired:
      return None, f"git took longer than {LFS_CHECK_SEC} s"
    except (OSError, subprocess.SubprocessError):
      return None, "git failed"
  fields = p.stdout.split(b"\x00")
  return [
      os.fsdecode(fields[i])
      for i in range(0, len(fields) - 2, 3)
      if fields[i + 2] == b"lfs"
  ], ""


def _harness_rows(entries: Iterable[Entry]) -> list[tuple[bytes, Entry]]:
  return sorted(
      (e.path.encode("utf-8", "surrogateescape"), e)
      for e in entries
      if e.mode != "tree"
      and e.path != ENV_FILE
      and not e.path.startswith(".git/")
      and not bundle_excluded(e.path)
  )


def harness_tree(entries: Iterable[Entry]) -> str:
  """Computes a SHA-256 digest of a repository's harness entries.

  Digests git's mode, object ID, and path of every entry, leaving out every
  folder the server could read as the artifacts (`bundle_excluded`: neither
  build bundles them), `.env`, and `.git`. The kit computes it from the working
  tree (`worktree_entries`) when it exports, and the server from the submitted
  commit's tree; equal digests mean the same files.

  Args:
    entries: Repository entries.

  Returns:
    The lowercase hexadecimal SHA-256 digest string.
  """
  h = hashlib.sha256()
  for path, e in _harness_rows(entries):
    h.update(f"{e.mode} {e.sha or ''} ".encode() + path + b"\n")
  return h.hexdigest()


def harness_files(entries: Iterable[Entry]) -> dict[str, str] | None:
  """Returns the files `harness_tree()` digests, or None past limit.

  Produces the manifest's `harness.files` mapping, which lets `review()` name
  what changed since a run.

  Args:
    entries: Repository entries.

  Returns:
    A dictionary `{path: "<mode> <object id>"}`, or None if there are more than
    `MAX_HARNESS_FILES` entries.
  """
  rows = _harness_rows(entries)
  if len(rows) > MAX_HARNESS_FILES:
    return None
  return {e.path: f"{e.mode} {e.sha or ''}" for _, e in rows}


def _changed_files(
    recorded, current: dict[str, str] | None
) -> list[str] | None:
  """Lists paths whose mode or content differ between `recorded` and `current`.

  Includes added and removed paths.

  Args:
    recorded: Manifest's `harness.files` dictionary.
    current: Submitted harness's `harness_files()` dictionary, or None.

  Returns:
    A sorted list of changed paths, or None when either mapping is missing.
  """
  if not isinstance(recorded, dict) or current is None:
    return None
  return sorted(
      p
      for p in set(recorded) | set(current)
      if recorded.get(p) != current.get(p)
  )


def worktree_entries(
    repo: pathlib.Path, git: GitRunner | None | object = _ASK
) -> list[Entry] | None:
  """Returns the entries the working tree of `repo` would commit.

  Includes tracked and untracked files that are not ignored, as git would record
  them: the blob ID `git hash-object` gives (line-ending conversion included),
  the mode git takes (the owner's execute bit, or the index's mode where
  `core.fileMode` or `core.symlinks` is false, as on Windows), and the
  checked-out commit of each submodule. The folders `harness_tree()` leaves out
  (`bundle_excluded`) are not looked at at all.

  Args:
    repo: Path to the repository root.
    git: Optional `GitRunner` from `work_tree()`, or `_ASK` to query it.

  Returns:
    A list of `Entry` objects, or None when `repo` is not the top of a git
    repository.

  Raises:
    Unreadable: When this machine cannot read a harness file (or enter a folder
      above one) or `work_tree()` stops, before anything is hashed.
    GitFailed: When a git command fails.
  """
  if git is _ASK:
    git, problem = work_tree(repo)
    if problem:
      raise Unreadable(problem)
  if git is None or git.prefix:
    return None
  staged = git("ls-files", "-z", "-s") or b""
  warned: list[str] = []
  listed = (
      git(
          "ls-files",
          "-z",
          "--cached",
          "--others",
          "--exclude-standard",
          warnings=warned,
      )
      or b""
  )
  file_mode, symlinks = git_flag(git, "core.fileMode"), git_flag(
      git, "core.symlinks"
  )
  index: dict[str, tuple[str, str]] = {}  # path -> (mode, object id)
  for rec in staged.split(b"\x00"):
    if rec:
      head, _, raw = rec.partition(b"\t")
      mode, sha = head.split()[:2]
      index[os.fsdecode(raw)] = (mode.decode(), sha.decode())
  out: list[Entry] = []
  files: dict[str, os.stat_result] = {}
  # (path, why) of what this machine cannot read
  blocked = could_not_open(warned, repo, git)
  for raw in dict.fromkeys(r for r in listed.split(b"\x00") if r):
    rel = os.fsdecode(raw)
    if rel == ENV_FILE or bundle_excluded(rel):
      continue  # never part of the digest
    path = repo / rel
    mode = index.get(rel, ("", ""))[0]
    if rel.endswith("/"):
      # an untracked repository inside this one: `git add -A` records its commit
      # as a submodule (one without a commit cannot be added, so the server
      # never sees it either)
      head = (
          git("-C", rel, "rev-parse", "--verify", "-q", "HEAD", ok=(0, 1))
          if _own_repository(path)
          else None
      )
      if head:
        out.append(
            Entry(rel.rstrip("/"), SUBMODULE, None, head.decode().strip())
        )
      continue
    try:
      st = os.lstat(path)
      if mode == SUBMODULE:
        # a submodule that was never checked out is an empty folder, where `git
        # -C` would answer for this repository: its commit is then the one the
        # index records
        checked_out = stat.S_ISDIR(st.st_mode) and os.path.lexists(
            path / ".git"
        )
        head = (
            git("-C", rel, "rev-parse", "HEAD", soft=True)
            if checked_out
            else None
        )
        out.append(
            Entry(
                rel,
                SUBMODULE,
                None,
                head.decode().strip() if head else index[rel][1],
            )
        )
      elif stat.S_ISLNK(st.st_mode):
        out.append(
            Entry(rel, LINK, None, blob_id(os.fsencode(os.readlink(path))))
        )
      elif mode == LINK and not symlinks and stat.S_ISREG(st.st_mode):
        # a link checked out as a file
        out.append(Entry(rel, LINK, None, blob_id(path.read_bytes())))
      elif stat.S_ISREG(st.st_mode):
        if os.access(path, os.R_OK):
          files[rel] = st
        else:
          blocked.append((rel, os.strerror(errno.EACCES)))
    except (FileNotFoundError, NotADirectoryError):
      continue  # deleted since it was committed: a commit would not hold it
    except OSError as exc:
      blocked.append((rel, _why(exc)))
  if blocked:
    raise Unreadable(
        unreadable_stop(
            repo, blocked, "the kit cannot compute the digest of your harness"
        )
    )
  if files:
    ids = _hash_objects(git, list(files), git.prefix)
    for rel in files:
      # a name --stdin-paths cannot take as written: hashed on its own
      if rel not in ids:
        ids[rel] = (git("hash-object", "--", rel) or b"").decode().strip()
    for rel, st in files.items():
      out.append(
          Entry(
              rel,
              git_mode(st, index.get(rel, ("", ""))[0], file_mode),
              st.st_size,
              ids[rel],
          )
      )
  return out


def git_flag(git: Callable[..., bytes | None], key: str) -> bool:
  """Reads a boolean git setting that defaults to True unless set to false.

  `git config` exits with code 1 for a key that is not set (such as
  `core.fileMode`).

  Args:
    git: Git runner callable.
    key: Configuration key name.

  Returns:
    True unless the configuration value is `"false"`.
  """
  return (
      git("config", "--bool", key, ok=(0, 1)) or b"true"
  ).strip() != b"false"


def git_mode(st: os.stat_result, indexed: str | None, exec_bit: bool) -> str:
  """Returns the git mode string for a regular file.

  Uses the owner's execute bit when `exec_bit` is True, or the index's mode
  `indexed` (`100644` when absent from the index) when git ignores the execute
  bit (`core.fileMode` false).

  Args:
    st: File `os.stat_result`.
    indexed: Mode recorded in git's index, if any.
    exec_bit: Whether git honors the filesystem execute bit.

  Returns:
    `"100755"` or `"100644"`.
  """
  if exec_bit:
    return "100755" if st.st_mode & stat.S_IXUSR else "100644"
  return indexed if indexed in FILE_MODES else "100644"


def read_local(root: pathlib.Path) -> Callable[[str], bytes | None]:
  """Returns `review()`'s `read` callback for a working tree at `root`.

  Reads at most `MAX_MANIFEST_BYTES + 1` bytes of a regular file under `root`
  (`manifest.json`, `.gitattributes`) using `read_regular`, or None for anything
  else.

  Args:
    root: Working tree root directory.

  Returns:
    A callable taking a relative path string and returning bytes or None.
  """

  def read(rel: str) -> bytes | None:
    try:
      return read_regular(root / rel, MAX_MANIFEST_BYTES)
    except OSError:
      return None

  return read


def _blocking(entries: Iterable[Entry]) -> list[Entry]:
  """Returns unreadable entries that require stopping the kit's review.

  Includes what this machine cannot read in a folder the server could read as
  the artifacts (`bundle_excluded`), or a `.gitattributes` above them, which
  decides what Git LFS stores. Excludes what git ignores (a commit leaves it out
  anyway) and folders too deep to list (`review()` refuses their task).

  Args:
    entries: Repository entries.

  Returns:
    A list of blocking unreadable `Entry` objects.
  """
  return [
      e
      for e in entries
      if e.unreadable
      and e.unreadable != TOO_DEEP
      and not e.ignored
      and (
          bundle_excluded(e.path.rstrip("/"))
          or posixpath.basename(e.path.rstrip("/")) == ATTRIBUTES
      )
  ]


def _unlisted_runs(entries: Iterable[Entry]) -> list[Entry]:
  """Returns non-ignored `runs/` and run folders this machine cannot list.

  Args:
    entries: Repository entries.

  Returns:
    A sorted list of unreadable `runs/` or `runs/<run>` entries (`_runs`).
  """
  return sorted(
      (
          e
          for e in entries
          if e.unreadable
          and not e.ignored
          and RUNS_FOLDER_RE.fullmatch(e.path.rstrip("/"))
      ),
      key=lambda e: e.path,
  )


def _runs_stop(entries: list[Entry]) -> list[Entry]:
  """Returns the `_unlisted_runs` entries that the review must stop on.

  Stops on all of them when no other folder of artifacts is found, or when the
  one found is itself a `runs/<run>/artifacts`, since an unlisted run folder
  could hold the folder the server reads or a second candidate.

  Args:
    entries: Repository entries.

  Returns:
    A list of unreadable run folder entries that block review.
  """
  unlisted = _unlisted_runs(entries)
  root = (
      find_root([e for e in entries if not e.ignored])[0] if unlisted else None
  )
  return (
      unlisted
      if unlisted and (root is None or root.startswith("runs/"))
      else []
  )


def review_directory(
    entry: pathlib.Path,
    dev: dict[str, DevTask] | None = None,
    dev_file: pathlib.Path = DEV_SET_FILE,
    harness: bool = True,
) -> Review:
  """Reviews an entry's artifacts on disk using the server's commit rules.

  Computes the harness digest from what the working tree would commit
  (`worktree_entries`), unless `harness` is False (which skips hashing every
  file of the entry for callers that need only task statuses). Stops, reviewing
  nothing (`Review.unreadable`), on unreadable paths where artifacts could be,
  on unreadable harness files, and on git failures.

  Args:
    entry: Filesystem path to the entry directory.
    dev: Optional mapping of task ID to `DevTask`; loaded from `dev_file` when
      None.
    dev_file: Path to the dev-set YAML file.
    harness: Whether to compute and check the working-tree harness digest.

  Returns:
    The `Review` for `entry`.
  """
  dev = dev if dev is not None else load_dev_set(dev_file)
  # asked once, for the listing and .git/info/attributes alike
  git, problem = work_tree(entry)
  try:
    # the statuses need no blob id
    entries = [] if problem else from_directory(entry, hashes=harness, git=git)
    blocked = _blocking(entries) or _runs_stop(entries)
    if blocked:
      problem = unreadable_stop(
          entry,
          [(e.path.rstrip("/"), e.unreadable) for e in blocked],
          "the kit cannot review the artifacts",
      )
    if not problem:
      rv = review(
          entries,
          dev,
          read=read_local(entry),
          dev_digest=dev_set_digest(dev_file),
          harness_entries=worktree_entries(entry, git) if harness else None,
          info_attributes=_info_attributes(git),
      )
      unlisted = _unlisted_runs(entries)
      if unlisted:  # artifacts/ (or an alias) is read whatever runs/ holds
        listed = shown([f"{e.path}/ ({e.unreadable})" for e in unlisted])
        fix = unreadable_fix(entry / e.path.split("/", 1)[0] for e in unlisted)
        rv.warnings.append(
            printable(
                f"this machine cannot list or enter {listed}, so a"
                " runs/<run>/artifacts folder in it is not reviewed"
                f" ({rv.root}/ is read anyway): {fix}"
            )
        )
      return rv
  except GitFailed as exc:
    problem = git_stop(entry, exc.said, exc.kind)
  except Unreadable as exc:
    problem = exc.problem
  return Review(
      root=None,
      tasks={t: TaskReview(id=t) for t in dev},
      problems=[printable(problem)],
      unreadable=True,
  )


def _info_attributes(git: GitRunner | None) -> bytes | None:
  """Reads the working tree's `.git/info/attributes` file if available.

  Git applies `.git/info/attributes` on top of `.gitattributes` files (so a Git
  LFS rule there stores files as pointers just the same), reading through
  symlinks as git does.

  Args:
    git: Optional `GitRunner` for the working tree.

  Returns:
    The raw bytes of `.git/info/attributes`, or None when absent or unreadable.
  """
  if git is None or git.attributes is None:
    return None
  try:
    return read_regular(
        pathlib.Path(os.path.realpath(git.attributes)), LFS_RULES_FILE_BYTES
    )
  except OSError:
    return None


def _manifest(
    rv: Review,
    root_entries: dict[str, Entry],
    read: Callable[[str], bytes | None] | None,
    dev_digest: str | None,
    harness_digest: str | None,
    current_files: dict[str, str] | None = None,
) -> None:
  """Compares `manifest.json` against the files, dev set, and harness.

  Populates provenance, warnings, and notes on `rv` without adding refusals
  (the reproduction run checks whether the harness gets similar scores).

  Args:
    rv: `Review` object to update in place.
    root_entries: Mapping of file names directly under the artifacts root to
      their `Entry`.
    read: Callback returning file bytes for a repository path, or None.
    dev_digest: Digest of the current `dev_set.yaml`, or None.
    harness_digest: Digest of the submitted harness tree, or None.
    current_files: Optional `harness_files()` mapping of the submitted harness
      used to name what changed since the run.
  """
  prov = rv.provenance = {
      "manifest": "missing",
      "dev_set": "unknown",
      "harness": "unknown",
      "edited": [],
  }
  entry = root_entries.get(MANIFEST)
  if entry is None or read is None:
    rv.warnings.append(
        f"no {MANIFEST}: nothing records which run made these files (the kit's"
        " export writes it)"
    )
    return
  data = read(entry.path) if (entry.size or 0) <= MAX_MANIFEST_BYTES else None
  try:
    m = json.loads(data.decode("utf-8")) if data else None
  except (UnicodeDecodeError, ValueError, RecursionError):
    m = None
  if not isinstance(m, dict) or not isinstance(m.get("tasks"), dict):
    prov["manifest"] = "invalid"
    rv.warnings.append(f"{MANIFEST} is not a manifest the kit wrote: ignored")
    return
  prov["manifest"] = "ok"
  if m.get("format") != FORMAT:
    rv.warnings.append(
        f"{MANIFEST} has format {m.get('format')!r}, not {FORMAT}: export again"
        " with this kit"
    )
  if dev_digest:
    prov["dev_set"] = "same" if m.get("dev_set") == dev_digest else "other"
    if prov["dev_set"] == "other":
      rv.warnings.append(
          "made with another version of the dev set: run the dev set again with"
          " the competition repository's current eval/dev_set.yaml"
      )
  recorded = (
      (m.get("harness") or {}).get("tree")
      if isinstance(m.get("harness"), dict)
      else None
  )
  if not recorded:
    rv.warnings.append(
        f"{MANIFEST} records no harness (a run with --url, or of an entry"
        " outside a git repository): nothing ties these artifacts to the"
        " harness you submit"
    )
  # the last run's harness against the one submitted: the same, the same but for
  # documentation, or changed
  docs_changed = None
  if harness_digest and recorded and recorded != harness_digest:
    changed = _changed_files(m["harness"].get("files"), current_files)
    if changed and all(
        DOCS_RE.fullmatch(posixpath.basename(p)) for p in changed
    ):
      prov["harness"], docs_changed = "docs-only", changed
    else:
      prov["harness"] = "changed"
      what = (
          f"changed since the run: {shown(changed)}"
          if changed
          else "its files differ from the run's"
      )
      rv.warnings.append(
          f"the harness changed after these artifacts were made ({what}): the"
          " reproduction run checks whether this harness gets similar scores"
      )
  elif harness_digest and recorded:
    prov["harness"] = "same"
  older = []
  for tid, r in rv.tasks.items():
    t = m["tasks"].get(tid)
    if r.status == MISSING:
      continue
    if not isinstance(t, dict):
      r.warnings.append(f"not in {MANIFEST}")
      continue
    tree = t.get("tree")
    # a task kept from a run of an earlier version of the harness: its run's
    # tree is not the last run's (checked whenever the last run's harness is the
    # one submitted, documentation aside)
    if (
        prov["harness"] in ("same", "docs-only")
        and isinstance(tree, str)
        and tree != recorded
    ):
      older.append(tid)
    reward = t.get("reward")
    if (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and 0 <= reward <= 1
    ):
      r.claimed = float(reward)
    files = t.get("files") if isinstance(t.get("files"), dict) else {}
    have = {p.lstrip("/"): e.sha for p, e in r.files.items()}
    if any(sha is None for sha in have.values()):
      continue  # a listing without blob ids: nothing to compare
    changed = sorted(
        p for p in set(have) | set(files) if have.get(p) != files.get(p)
    )
    message = t.get("final_message")
    if (
        r.message is not None
        and r.message.sha
        and message
        and r.message.sha != message
    ):
      changed.append(FINAL_MESSAGE)
    if _deleted_changed(r, t.get("deleted"), read):
      changed.append(DELETED)
    if changed:
      r.edited = True
      prov["edited"].append(tid)
      r.warnings.append(
          f"differs from what the export wrote ({shown(changed)}): changed"
          " after the run"
      )
  if docs_changed is not None:
    rv.notes.append(
        f"only documentation changed since the last run ({shown(docs_changed)})"
        + (
            ""
            if older
            else ": the artifacts still come from the harness you submit"
        )
    )
  if older:
    prov["harness"] = "changed"
    for tid in older:
      rv.tasks[tid].warnings.append(
          "made by an earlier version of the harness than the one submitted:"
          " run this task again (`eval/run.py --tasks <id>`)"
      )


def _deleted_changed(
    r: TaskReview, recorded, read: Callable[[str], bytes | None] | None
) -> bool:
  """Returns whether `deleted.txt` differs from the manifest's `deleted` list.

  A missing `deleted.txt` differs when the export recorded deletions; returns
  False when `deleted.txt` cannot be read.

  Args:
    r: `TaskReview` for the task.
    recorded: Manifest's recorded `deleted` value for the task.
    read: Callback returning file bytes for a repository path, or None.

  Returns:
    True if the set of deleted paths differs from `recorded`.
  """
  if not isinstance(recorded, list):
    recorded = []
  if r.deleted is None:
    return bool(recorded)
  if read is None or (r.deleted.size or 0) > MAX_DELETED_BYTES:
    return False
  data = read(r.deleted.path)
  if data is None:
    return False
  return set(deleted_lines(data)) != {x for x in recorded if isinstance(x, str)}


def review(
    entries: list[Entry],
    dev: dict[str, DevTask],
    read: Callable[[str], bytes | None] | None = None,
    dev_digest: str | None = None,
    harness_digest: str | None = None,
    whole_tree: bool = False,
    harness_entries: list[Entry] | None = None,
    info_attributes: bytes | None = None,
) -> Review:
  """Reviews a submission's artifacts under the server's rules.

  Never raises on what a repository holds; every message in the returned
  `Review` is valid UTF-8 (`printable`).

  Args:
    entries: Repository entries (`from_git_tree`, `from_ls_tree`, or
      `from_directory` through `review_directory()`).
    dev: Mapping of task ID to `DevTask`.
    read: Optional callback `read(path)` returning a small file's bytes
      (`manifest.json`, `.gitattributes`) or None.
    dev_digest: Optional digest of the current `dev_set.yaml`.
    harness_digest: Optional digest of the submitted harness's tree; computed
      from `harness_entries` or `entries` (when `whole_tree` is True) if None.
    whole_tree: Whether `entries` covers the whole repository.
    harness_entries: Optional working-tree entries (`worktree_entries`) for
      computing `harness_digest`.
    info_attributes: Optional `.git/info/attributes` bytes from a working tree.

  Returns:
    The `Review` with sanitized printable messages.
  """
  rv = _review(
      entries,
      dev,
      read,
      dev_digest,
      harness_digest,
      whole_tree,
      harness_entries,
      info_attributes,
  )
  _printable_messages(rv)
  return rv


def _review(
    entries: list[Entry],
    dev: dict[str, DevTask],
    read: Callable[[str], bytes | None] | None,
    dev_digest: str | None,
    harness_digest: str | None,
    whole_tree: bool,
    harness_entries: list[Entry] | None,
    info_attributes: bytes | None,
) -> Review:
  """Performs the core artifact review before message sanitization.

  Args:
    entries: Repository entries.
    dev: Mapping of task ID to `DevTask`.
    read: Callback returning file bytes for a repository path, or None.
    dev_digest: Digest of the current `dev_set.yaml`, or None.
    harness_digest: Digest of the submitted harness tree, or None.
    whole_tree: Whether `entries` covers the whole repository.
    harness_entries: Working-tree entries for harness digest calculation, or
      None.
    info_attributes: `.git/info/attributes` bytes, or None.

  Returns:
    The unsanitized `Review` result.
  """
  ignored = [e.path for e in entries if e.ignored and e.mode != "tree"]
  entries = [e for e in entries if not e.ignored]
  root, warnings, problems = find_root(entries)
  rv = Review(
      root=None,
      tasks={t: TaskReview(id=t) for t in dev},
      problems=problems,
      warnings=warnings,
  )
  if root is None:
    hidden, _, _ = find_root([Entry(p, "100644") for p in ignored])
    if hidden is not None:
      rv.problems = [
          f"git ignores {hidden}/ (a .gitignore names it), so a commit holds"
          " none of it: take it out of .gitignore, or `git add -f"
          f" {shell_path(hidden)}`"
      ]
    return rv
  top = root
  root, more = _unwrap(entries, root, dev)
  rv.root = root
  rv.warnings += more
  # one wrapper folder read as the root: what lies beside it is not graded
  if root != top:
    inner = root[len(top) + 1 :]
    beside = []
    for name, v in _children(entries, top).items():
      if name != inner and not junk(name):
        for rel, _ in v:
          beside.append(f"{top}/{name}" + (f"/{rel}" if rel else ""))
    if beside:
      rv.warnings.append(f"beside {root}/, ignored: {shown(sorted(beside))}")
  behind = [
      e.path for e in entries if e.stale and e.path.startswith(root + "/")
  ]
  if behind:
    rv.warnings.append(
        f"the committed version of {shown(behind)} differs from the file on"
        " disk although git sees no change (an earlier Git LFS or line-ending"
        f" rule): `git rm -r --cached {shell_path(root)}`, then `git add"
        f" {shell_path(root)}` and commit"
    )
  left_out = [p for p in ignored if p.startswith(root + "/")]
  if left_out:
    rv.warnings.append(
        f"git ignores {len(left_out)} file(s) under {root}/"
        f" ({shown(left_out)}), so the server will not get them: `git add -f"
        f" {shell_path(root)}`"
    )
  kids = _children(entries, root)
  under = []
  for v in kids.values():
    for _, e in v:
      under.append(e)
  # the limits count the whole folder found, a wrapper's neighbours too: they
  # are in the commit all the same
  if root == top:
    everything = under
  else:
    everything = []
    for v in _children(entries, top).values():
      for _, e in v:
        everything.append(e)
  total = sum(e.size or 0 for e in everything)
  if len(everything) > MAX_TOTAL_FILES or total > MAX_TOTAL_BYTES:
    rv.problems.append(
        f"{top}/ holds {len(everything)} files and {_mib(total)}; at most"
        f" {MAX_TOTAL_FILES} files and {_mib(MAX_TOTAL_BYTES)}"
    )
    return rv
  # a stale artifacts/ and a fresh copy elsewhere: only one is read
  others = _other_roots(entries, top, dev)
  if others:
    rv.warnings.append(
        f"{shown(others)}: ignored, {top}/ is read (keep one folder of"
        " artifacts)"
    )
  tracked, cause, too_long = _lfs_review(
      [e.path for e in under],
      attributes_files(entries, root),
      read,
      info_attributes,
  )
  if too_long:
    rv.warnings.append(LFS_TOO_LONG.format(shown(too_long)))
  if tracked is None:
    rv.warnings.append(LFS_UNCHECKED.format(cause))
  elif tracked:
    rv.problems.append(
        f"Git LFS tracks {shown(tracked)}: the server clones without LFS, so"
        " commit these files directly (`git lfs untrack`, then add them again)"
    )
    return rv
  root_files = {
      name: v[0][1] for name, v in kids.items() if len(v) == 1 and not v[0][0]
  }
  # task id -> [(folder name, how)]
  folders: dict[str, list[tuple[str, str]]] = {}
  unknown, stray, leftover, earlier, not_folders = [], [], [], [], {}
  for name in kids:
    if junk(name):
      continue
    if name.startswith(KIT_STAGING):  # what an export cut short left
      leftover.append(f"{root}/{name}")
      continue
    kept = EARLIER_RE.fullmatch(name)
    # an earlier task folder an export could not put back
    if kept and kept.group("task") in dev:
      earlier.append(f"{root}/{name}")
      continue
    if name in root_files:
      task, _ = task_for(name, dev)
      if task and name != MANIFEST:
        # a link or a file named as a task
        not_folders.setdefault(task, (name, root_files[name]))
      elif (
          name not in LAYOUT_FILES
          and not ARCHIVE_RE.fullmatch(name)
          and name.lower() not in ("readme.md", "readme")
      ):
        stray.append(name)
      continue
    task, how = task_for(name, dev)
    if task is None:
      close = difflib.get_close_matches(
          name.lower(), list(dev), n=1, cutoff=0.75
      )
      unknown.append(f"{name} (did you mean {close[0]}?)" if close else name)
    else:
      folders.setdefault(task, []).append((name, how))
  if unknown:
    rv.warnings.append(f"not dev tasks, ignored: {shown(sorted(unknown))}")
  if stray:
    rv.warnings.append(
        f"files directly in {root}/ are ignored: {shown(sorted(stray))}"
    )
  if leftover:
    rv.warnings.append(
        f"{shown(sorted(leftover))}: left over from an export cut short,"
        " ignored: remove it before you commit (a .plebench-old-<task>-*"
        " folder is that task's earlier folder)"
    )
  if earlier:
    rv.warnings.append(
        f"{shown(sorted(earlier))}: an earlier task folder an export could not"
        " put back, ignored: rename it back to its task (once that folder is"
        " removed), or remove it"
    )
  for tid, found in folders.items():
    exact = [n for n, how in found if how == "exact"]
    if len(found) > 1 and not exact:
      r = rv.tasks[tid]
      r.status, r.problems = REFUSED, [
          f"several folders name this task ({shown([n for n, _ in found])}) and"
          f" none is spelled exactly {tid}: keep one"
      ]
      continue
    name = exact[0] if exact else found[0][0]
    r = rv.tasks[tid] = _review_task(dev[tid], f"{root}/{name}", kids[name])
    how = dict(found)[name]
    if how != "exact":
      r.warnings.insert(0, f"read folder {name} as {tid}: rename it")
    others = [n for n, _ in found if n != name]
    if others:
      r.warnings.append(f"{shown(others)}: also names this task, ignored")
  for tid, (name, e) in not_folders.items():
    r = rv.tasks[tid]
    if r.status == MISSING:
      r.status, r.folder = REFUSED, f"{root}/{name}"
      r.problems = [
          f"{root}/{name} is a {e.kind}, not a folder; the task's files go in a"
          f" folder named {tid}"
      ]
    else:
      r.warnings.append(
          f"{root}/{name} ({e.kind}) also names this task, ignored"
      )
  missing = [t for t, r in rv.tasks.items() if r.status == MISSING]
  if missing:
    rv.warnings.append(
        f"{len(missing)} of {len(dev)} dev tasks have no folder and count 0:"
        f" {shown(missing)}"
    )
  current_files = None
  if harness_digest is None and (harness_entries is not None or whole_tree):
    listed = harness_entries if harness_entries is not None else entries
    harness_digest = harness_tree(listed)
    current_files = harness_files(listed)
  _manifest(rv, root_files, read, dev_digest, harness_digest, current_files)
  if not rv.usable:
    rv.problems.append(
        f"no dev task in {root}/ has usable artifacts (see each task's reasons)"
    )
  return rv


def _drain(src: BinaryIO) -> None:
  """Reads and discards the remaining bytes of `src` to prevent SIGPIPE.

  Bounded, since the listing already passed the size checks.

  Args:
    src: Binary stream to drain.
  """
  for _ in iter(lambda: src.read(1 << 20), b""):
    pass


def _copy(src: BinaryIO, out: BinaryIO, limit: int) -> tuple[int, bytes]:
  """Copies at most `limit + 1` bytes from `src` to `out`.

  Reading up to `limit + 1` bytes allows a size over `limit` to show there was
  more.

  Args:
    src: Source binary stream.
    out: Destination binary stream.
    limit: Maximum allowed byte count.

  Returns:
    A tuple `(bytes_copied, first_1024_bytes)`.
  """
  n, head = 0, b""
  while n <= limit:
    chunk = src.read(min(1 << 20, limit + 1 - n))
    if not chunk:
      break
    if not head:
      head = chunk[:1024]
    out.write(chunk)
    n += len(chunk)
  return n, head


def _deleted_paths(data: bytes, task: DevTask, r: TaskReview) -> list[str]:
  """Filters `deleted.txt` lines to valid deliverable paths for `task`.

  Appends a warning to `r` for any lines that are not deliverable paths.

  Args:
    data: Raw bytes of `deleted.txt`.
    task: `DevTask` specification.
    r: `TaskReview` to record warnings on.

  Returns:
    A deduplicated list of valid deleted container paths.
  """
  out, bad = [], []
  for path in deleted_lines(data):
    if (
        not path.startswith("/")
        or path_problem(path[1:])
        or not task.covers(path)
    ):
      bad.append(path)
    else:
      out.append(path)
  if bad:
    r.warnings.append(
        f"{DELETED}: not deliverable paths, ignored: {shown(bad)}"
    )
  return list(dict.fromkeys(out))


def materialize(
    rv: Review,
    dev: dict[str, DevTask],
    open_file: Callable[[str], BinaryIO],
    dest: pathlib.Path,
    only: Iterable[str] | None = None,
) -> Review:
  """Copies each usable task's files into `dest/<task id>/` with `replay.json`.

  Writes files in the layout (`files/<path>`, `final_message.md`) along with
  `replay.json`, which the replay agents read. A task whose content breaks a
  rule the listing could not show (a Git LFS pointer, or a file or task over the
  size limits once read) is refused here; `rv` is updated in place and returned.

  Args:
    rv: `Review` object to materialize and update in place.
    dev: Mapping of task ID to `DevTask`.
    open_file: Callback `open_file(path)` opening a repository file for reading.
    dest: Destination directory for materialized task folders.
    only: Optional subset of task IDs to materialize.

  Returns:
    The updated `Review` object.

  Raises:
    OSError: If writing to `dest` fails for a reason other than an invalid
      filename (`NAME_ERRNOS`).
  """
  wanted = None if only is None else set(only)
  for tid in [t for t in rv.usable if wanted is None or t in wanted]:
    r, task = rv.tasks[tid], dev[tid]
    out = dest / tid
    out.mkdir(parents=True, exist_ok=False)
    files, total, problems, unwritable = [], 0, [], False
    for where, e in r.files.items():
      target = out / FILES / where.lstrip("/")
      try:
        target.parent.mkdir(parents=True, exist_ok=True)
        sink = open(target, "wb")
      except OSError as exc:
        if exc.errno not in NAME_ERRNOS:
          raise  # a full disk, say: this machine's fault, not the artifacts'
        # the task is refused here, with the reason, and the other tasks go on
        why = exc.strerror or type(exc).__name__
        problems.append(
            f"{_clip(where)} cannot be written on this machine ({why})"
        )
        unwritable = True
        break
      with sink, open_file(e.path) as fh:
        n, head = _copy(fh, sink, MAX_FILE_BYTES)
        _drain(fh)
      total += n
      if n > MAX_FILE_BYTES:
        problems.append(f"{where} is over {_mib(MAX_FILE_BYTES)}")
      elif n < 1024 and head.startswith(LFS_POINTER):
        problems.append(
            f"{where} is a Git LFS pointer, not the file: commit the file"
            " itself"
        )
      files.append({
          "path": where,
          "src": f"{FILES}/{where.lstrip('/')}",
          "mode": "755" if e.executable else "644",
      })
    if total > MAX_TASK_BYTES:
      problems.append(f"{_mib(total)} in all, more than {_mib(MAX_TASK_BYTES)}")
    if unwritable:  # its message and deleted.txt do not matter any more
      r.status, r.problems = REFUSED, problems
      shutil.rmtree(out)
      continue
    message = None
    if r.message is not None:
      with open_file(r.message.path) as fh:
        raw = fh.read(MAX_MESSAGE_BYTES)
        cut = bool(fh.read(1))
        _drain(fh)
      # a character the cap cut in two is left out, not replaced, and the text
      # written stays within the cap (each replaced byte takes three bytes in
      # UTF-8), so the kit's and the server's replays give graders one message
      text = codecs.getincrementaldecoder("utf-8")("replace").decode(
          raw, final=not cut
      )
      if "�" in text and b"\xef\xbf\xbd" not in raw:
        r.warnings.append(
            f"{FINAL_MESSAGE} is not all UTF-8: unreadable bytes were replaced"
        )
      data = text.encode("utf-8")
      if len(data) > MAX_MESSAGE_BYTES:
        text = data[:MAX_MESSAGE_BYTES].decode("utf-8", "ignore")
      (out / FINAL_MESSAGE).write_text(text, encoding="utf-8")
      message = FINAL_MESSAGE
    deleted = []
    if r.deleted is not None:
      with open_file(r.deleted.path) as fh:
        data = fh.read(MAX_DELETED_BYTES + 1)
        _drain(fh)
      if len(data) > MAX_DELETED_BYTES:
        # never act on a path cut in the middle: what follows the last whole
        # line is dropped
        data = data[:MAX_DELETED_BYTES].rpartition(b"\n")[0]
        r.warnings.append(
            f"{DELETED} is longer than {MAX_DELETED_BYTES >> 10} KiB: only its"
            f" first {MAX_DELETED_BYTES >> 10} KiB of whole lines is used"
        )
      deleted = _deleted_paths(data, task, r)
    if problems:
      r.status, r.problems = REFUSED, problems
      shutil.rmtree(out)
      continue
    (out / REPLAY).write_text(
        json.dumps(
            {
                "task": tid,
                "files": files,
                "deleted": deleted,
                "final_message": message,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
  if rv.accepted and not rv.usable:
    rv.problems.append(
        f"no dev task in {rv.root}/ has usable artifacts (see each task's"
        " reasons)"
    )
  _printable_messages(rv)
  return rv


def read_replay(folder: pathlib.Path) -> dict[str, Any]:
  """Reads and validates a materialized task folder's `replay.json`.

  Verifies that every listed source file and final message resolves inside
  `folder` and every container path is absolute.

  Args:
    folder: Path to the materialized task folder.

  Returns:
    The parsed `replay.json` dictionary.

  Raises:
    ValueError: If any file entry, deleted path, or `final_message` is invalid.
  """
  data = json.loads((folder / REPLAY).read_text(encoding="utf-8"))
  root = folder.resolve()

  def inside(rel) -> bool:
    return isinstance(rel, str) and root in (folder / rel).resolve().parents

  for f in data["files"]:
    if not inside(f["src"]) or not f["path"].startswith("/"):
      raise ValueError(f"replay.json: bad entry {f!r}")
  if any(not p.startswith("/") for p in data["deleted"]):
    raise ValueError("replay.json: a deleted path is not absolute")
  if data.get("final_message") is not None and not inside(
      data["final_message"]
  ):
    raise ValueError(
        f"replay.json: bad final_message {data['final_message']!r}"
    )
  return data
