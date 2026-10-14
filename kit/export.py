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

"""Exports and imports dev-set artifacts for PLEBench evaluation runs.

Formats each measured trial's capture (`capture.py`) according to
`docs/artifacts.md`, writes `manifest.json`, and copies task folders into the
entry's `artifacts/` directory. `eval/run.py` exports after every run, and
`python -m kit artifacts <entry> --from <run>` imports a run's artifacts the
same way (`import_run`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import pathlib
import re
import shutil
import stat
import tempfile
import time
from typing import Any
import uuid

from kit import artifacts
from kit import capture
from kit import spec

RECORD = "export.json"  # beside a run's artifacts/ folder (save_record)
# git history past this gets a note: the repository limit (1 GiB) counts it
HISTORY_WARN_BYTES = 512 << 20
# prefix of what an export writes beside the task folders before the swap
STAGING = artifacts.KIT_STAGING
# the manifest keeps this many of a task's `skipped` rows and counts the rest by
# reason
SKIPPED_KEPT = 50
# the path of a row that counts others, as capped() writes it
MORE_RE = re.compile(r"\((\d+) more\)")


@dataclasses.dataclass
class Exported:
  """Outcome of exporting a single task's artifacts."""

  task: str
  # the manifest's entry for the task, None when nothing was exported
  entry: dict[str, Any] | None
  reason: str = ""  # why it was not exported


def not_exported(d: Exported, entry: pathlib.Path | None) -> str:
  """Formats a status line for a task that was not exported.

  Reports why task `d` was not exported and whether `entry`'s `artifacts/`
  directory retains an earlier folder for it.

  Args:
    d: The unexported task result.
    entry: Path to the entry directory, or None.

  Returns:
    The formatted status line describing why the task was not exported.
  """
  kept = entry is not None and artifacts.is_folder(
      entry / artifacts.ROOT / d.task
  )
  return f"  {d.task}: not exported, {d.reason}" + (
      "; the entry keeps its earlier folder" if kept else ""
  )


def provenance(
    entry: pathlib.Path, name: str, model_api: str
) -> dict[str, Any]:
  """Builds the manifest `harness` metadata for a run of `entry`.

  Computes the digest of the files the working tree would commit
  (`artifacts.harness_tree`), the per-file digests (`artifacts.harness_files`,
  so a review can report what changed since), the HEAD commit, and whether
  harness files have uncommitted changes. Outside the root of a git repository,
  `tree`, `files`, and `commit` are None. Raises `artifacts.Unreadable` or
  `artifacts.GitFailed` when the repository state cannot be determined.

  Args:
    entry: Path to the entry directory.
    name: Name of the harness entry.
    model_api: Identifier of the model API used for the run.

  Returns:
    A dictionary containing the harness provenance metadata for the manifest.
  """
  entries = artifacts.worktree_entries(entry)
  if entries is None:
    return {
        "name": name,
        "model_api": model_api,
        "tree": None,
        "files": None,
        "commit": None,
        "dirty": False,
    }
  commit = (
      os.fsdecode(
          artifacts.git_output(entry, "rev-parse", "HEAD") or b""
      ).strip()
      or None
  )
  return {
      "name": name,
      "model_api": model_api,
      "tree": artifacts.harness_tree(entries),
      "files": artifacts.harness_files(entries),
      "commit": commit,
      "dirty": bool(commit) and _dirty(entry),
  }


def _dirty(entry: pathlib.Path) -> bool:
  """Checks whether `git status` shows uncommitted changes to harness files.

  Ignores `.env` and any folders excluded from the harness bundle (for example,
  an untracked `dev_artifacts/` directory leaves the harness clean).

  Args:
    entry: Path to the entry directory.

  Returns:
    True if any harness file covered by the digest is modified or untracked.
  """
  # every untracked file by name (git would show an untracked runs/ alone,
  # hiding runs/<run>/artifacts), whatever status.showUntrackedFiles says
  status = artifacts.GitRunner(entry, "")(
      "status", "--porcelain", "-z", "--untracked-files=all"
  )
  records = os.fsdecode(status or b"").split("\0")
  i = 0
  while i < len(records):
    record = records[i]
    i += 1
    if len(record) < 4:
      continue
    paths = [record[3:]]
    # a rename or a copy: the earlier path follows
    if record[0] in "RC" and i < len(records):
      paths.append(records[i])
      i += 1
    if any(
        p != artifacts.ENV_FILE and not artifacts.bundle_excluded(p)
        for p in paths
    ):
      return True
  return False


def cut_message(data: bytes) -> bytes:
  """Truncates `data` to `artifacts.MAX_MESSAGE_BYTES` at a UTF-8 boundary.

  Ensures the cut occurs where a UTF-8 character starts (a non-continuation
  byte) so truncation never introduces an invalid UTF-8 sequence.

  Args:
    data: Raw UTF-8 message bytes.

  Returns:
    The message bytes truncated to at most `artifacts.MAX_MESSAGE_BYTES`.
  """
  if len(data) <= artifacts.MAX_MESSAGE_BYTES:
    return data
  end = artifacts.MAX_MESSAGE_BYTES
  while end > 0 and data[end] & 0xC0 == 0x80:
    end -= 1
  return data[:end]


def task_folder(
    trial: pathlib.Path, dest: pathlib.Path
) -> tuple[dict[str, Any] | None, str]:
  """Populates a task artifact folder `dest` from `trial`'s capture.

  Writes `files/`, `final_message.md`, `deleted.txt`, and `trajectory.json`
  into `dest`. If the trial does not contain a complete capture, `dest` is
  removed and the task is not exported.

  Args:
    trial: Path to the trial directory containing the capture.
    dest: Destination path for the new task artifact folder.

  Returns:
    A tuple `(content, reason)` where `content` is the manifest entry's content
    dictionary on success, or `None` with a failure `reason` when the trial
    lacks a complete capture.
  """
  cap = trial / capture.FOLDER
  try:
    info = json.loads((cap / capture.INFO).read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return (
        None,
        (
            "its deliverables were not copied out of the task container (no"
            " capture.json)"
        ),
    )
  if not isinstance(info, dict) or not isinstance(info.get("files"), dict):
    return None, "its capture.json cannot be read"
  if info.get("error"):
    return None, f"copying its deliverables out failed: {info['error']}"
  dest.mkdir(parents=True)
  files, total = {}, 0
  inside = os.path.join(os.path.realpath(cap), artifacts.FILES)
  for rel, meta in sorted(info["files"].items()):
    src = cap / artifacts.FILES / rel
    # files/<rel> is the path review() measures (its 64-folder and 1,024-byte
    # bounds included). A capture holds no links, so a folder on the way that is
    # one (in a run folder made elsewhere) could lead out of the capture
    if (
        artifacts.path_problem(f"{artifacts.FILES}/{rel}")
        or not artifacts.is_regular(src)
        or os.path.realpath(src) != os.path.join(inside, rel)
    ):
      shutil.rmtree(dest)
      return None, f"its capture lacks {rel}"
    target = dest / artifacts.FILES / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    os.chmod(target, 0o755 if str((meta or {}).get("mode")) == "755" else 0o644)
    files[rel] = artifacts.file_blob_id(target)
    total += target.stat().st_size
  skipped = [s for s in info.get("skipped") or [] if isinstance(s, dict)]
  message = cap / artifacts.FINAL_MESSAGE
  whole = message.read_bytes() if artifacts.is_regular(message) else b""
  text = cut_message(whole)
  if len(text) < len(whole):
    skipped.append({
        "path": artifacts.FINAL_MESSAGE,
        "reason": (
            f"cut at {artifacts.MAX_MESSAGE_BYTES >> 20} MiB, "
            "as the server grades it"
        ),
    })
  (dest / artifacts.FINAL_MESSAGE).write_bytes(text)
  total += len(text)
  deleted = [p for p in info.get("deleted") or [] if isinstance(p, str)]
  if deleted:
    listing = "".join(f"{p}\n" for p in deleted)
    (dest / artifacts.DELETED).write_text(listing, encoding="utf-8")
    total += len(listing.encode("utf-8"))
  # the trajectory is evidence, never graded: it is what gives way when the task
  # would pass its limit
  trajectory = trial / "agent" / "trajectory.json"
  if artifacts.is_regular(trajectory):
    size = trajectory.stat().st_size
    if size > artifacts.MAX_TRAJECTORY_BYTES:
      skipped.append({
          "path": artifacts.TRAJECTORY,
          "reason": f"over {artifacts.MAX_TRAJECTORY_BYTES >> 20} MiB",
      })
    elif total + size > artifacts.MAX_TASK_BYTES:
      skipped.append({
          "path": artifacts.TRAJECTORY,
          "reason": (
              "left out: with it the task would pass "
              f"{artifacts.MAX_TASK_BYTES >> 20} MiB"
          ),
      })
    else:
      shutil.copyfile(trajectory, dest / artifacts.TRAJECTORY)
  if info.get("note"):
    skipped.append({"path": "(baseline)", "reason": info["note"]})
  return {
      "files": files,
      "final_message": artifacts.blob_id(text),
      "deleted": deleted,
      "skipped": skipped,
  }, ""


def _layout(root: pathlib.Path) -> None:
  """Writes the kit's `.gitattributes` and `.gitignore` in an artifacts folder.

  Overwrites files generated by the kit while preserving any file modified by
  the participant (one lacking the kit's header comment).

  Args:
    root: Path to the artifacts folder.
  """
  for name, text in (
      (artifacts.ATTRIBUTES, artifacts.ATTRIBUTES_TEXT),
      (artifacts.IGNORE, artifacts.IGNORE_TEXT),
  ):
    path = root / name
    st = artifacts.lstat(path)
    if st is not None and not stat.S_ISREG(st.st_mode):
      continue
    if st is None or path.read_bytes().startswith(b"# written by the kit"):
      path.write_text(text, encoding="utf-8")


def _staged(rel: str) -> bool:
  """Returns whether `rel` is a temporary staging path inside `artifacts/`.

  Staging paths hold folders before swap, replaced folders set aside, or
  manifests being written; they are cleaned up when the export finishes and are
  ignored by `review()`.

  Args:
    rel: Relative path inside the artifacts directory.

  Returns:
    True if the top-level component of `rel` starts with the staging prefix.
  """
  return rel.split("/", 1)[0].startswith(STAGING)


def _usage(
    root: pathlib.Path,
) -> tuple[int, int, dict[str, int], list[tuple[int, pathlib.Path]]]:
  """Measures file counts, byte sizes, and trajectory files under `root`.

  Counts all non-directory entries (without following symlinks) as `review()`
  will count them after the export completes, excluding staged paths and
  `manifest.json`.

  Args:
    root: Path to the artifacts directory.

  Returns:
    A tuple `(files, size, per, trajectories)` with total file count, total
    bytes, bytes per top-level task folder, and `(size, path)` pairs for each
    task's `trajectory.json`.
  """
  files = size = 0
  per: dict[str, int] = {}
  trajectories: list[tuple[int, pathlib.Path]] = []
  for rel, path, st, why in artifacts.walk(
      root, stop=lambda rel, _: "staged" if _staged(rel) else ""
  ):
    if why or st is None or _staged(rel) or rel == artifacts.MANIFEST:
      continue
    files, size = files + 1, size + st.st_size
    if "/" in rel:
      top, rest = rel.split("/", 1)
      per[top] = per.get(top, 0) + st.st_size
      if rest == artifacts.TRAJECTORY and stat.S_ISREG(st.st_mode):
        trajectories.append((st.st_size, path))
  return files, size, per, trajectories


def fit_total(
    root: pathlib.Path, manifest: dict[str, Any], notes: list[str]
) -> list[pathlib.Path]:
  """Selects `trajectory.json` files to omit so `root` fits server limits.

  Drops trajectories largest-first until `root` (including the serialized
  `manifest`) fits within `artifacts.MAX_TOTAL_BYTES` and
  `artifacts.MAX_TOTAL_FILES`. Records each dropped trajectory in its task's
  `skipped` list in `manifest` and appends summary messages to `notes`.

  Args:
    root: Path to the artifacts directory.
    manifest: Manifest dictionary to update with skipped trajectory entries.
    notes: List to which warning or informational notes are appended.

  Returns:
    List of `trajectory.json` paths that the caller should remove.
  """
  files, size, per, found = _usage(root)

  def total() -> tuple[int, int]:
    return files + 1, size + len(manifest_text(manifest).encode("utf-8"))

  def over() -> bool:
    count, used = total()
    return count > artifacts.MAX_TOTAL_FILES or used > artifacts.MAX_TOTAL_BYTES

  dropped: list[pathlib.Path] = []
  for n, path in sorted(found, key=lambda x: (-x[0], str(x[1]))):
    if not over():
      break
    dropped.append(path)
    files, size = files - 1, size - n
    per[path.parent.name] = per.get(path.parent.name, 0) - n
    entry = manifest["tasks"].get(path.parent.name)
    if isinstance(entry, dict):
      row = {
          "path": artifacts.TRAJECTORY,
          "reason": (
              f"left out: with it {artifacts.ROOT}/ would pass the "
              f"server's {artifacts.MAX_TOTAL_BYTES >> 20} MiB"
          ),
      }
      rows = entry.setdefault("skipped", [])
      if row not in rows:
        # first: the manifest keeps only the first SKIPPED_KEPT rows
        rows.insert(0, row)
  if dropped:
    notes.append(
        "left out the trajectories of"
        f" {', '.join(sorted(p.parent.name for p in dropped))} (never graded),"
        f" so that {artifacts.ROOT}/ fits the server's"
        f" {artifacts.MAX_TOTAL_BYTES >> 20} MiB and"
        f" {artifacts.MAX_TOTAL_FILES:,} files"
    )
  if over():
    largest = sorted(per.items(), key=lambda kv: -kv[1])[:3]
    count, used = total()
    notes.append(
        f"{artifacts.ROOT}/ still holds {count:,} files and {used / 2**20:.0f}"
        f" MiB, past the server's {artifacts.MAX_TOTAL_FILES:,} files and"
        f" {artifacts.MAX_TOTAL_BYTES >> 20} MiB, so it would refuse all of it;"
        " the largest task folders: "
        + ", ".join(f"{t} ({b / 2**20:.0f} MiB)" for t, b in largest)
    )
  return dropped


def export_run(
    rows: list[dict[str, Any]],
    out: pathlib.Path,
    harness: dict[str, Any],
    dev_digest: str,
    notes: list[str] | None = None,
) -> list[Exported]:
  """Populates a run's `artifacts/` directory `out` from trial result `rows`.

  Each row provides `task`, `trial` (relative to `out.parent` or absolute),
  `measured`, `reward`, `ended`, and `run`. Unmeasured tasks or trials with
  incomplete captures are recorded with an explanation in `Exported.reason`.
  Writes `out/manifest.json` after fitting total size via `fit_total()`.

  Args:
    rows: Per-task trial result dictionaries.
    out: Destination `artifacts/` directory for the run (recreated afresh).
    harness: Provenance metadata dictionary for the harness.
    dev_digest: Digest identifying the dev set.
    notes: Optional list to collect size-fitting notes.

  Returns:
    List of `Exported` results, one per input row.
  """
  if out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True)
  done, tasks = [], {}
  for row in rows:
    task, trial = row["task"], row.get("trial")
    if not row.get("measured") or not trial:
      done.append(Exported(task, None, "it has no measured result in this run"))
      continue
    trial = out.parent / trial
    content, why = task_folder(trial, out / task)
    if content is None:
      done.append(Exported(task, None, why))
      continue
    entry = {
        "run": row.get("run", ""),
        "trial": trial.name,
        "ended": row.get("ended", ""),
        "reward": row.get("reward"),
        "tree": harness.get("tree"),
        **content,
    }
    tasks[task] = entry
    done.append(Exported(task, entry))
  _layout(out)
  manifest = {
      "format": artifacts.FORMAT,
      "dev_set": dev_digest,
      "created": _now(),
      "harness": harness,
      "tasks": tasks,
  }
  for path in fit_total(out, manifest, notes if notes is not None else []):
    path.unlink()
  write_manifest(out, manifest)
  return done


def _now() -> str:
  """Returns the current UTC timestamp in ISO 8601 format."""
  return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_manifest(root: pathlib.Path) -> dict[str, Any] | None:
  """Reads and validates `root/manifest.json` if within the size limit.

  Args:
    root: Path to the directory containing `manifest.json`.

  Returns:
    The parsed manifest dictionary if it is a valid mapping with a `tasks`
    dictionary and within `artifacts.MAX_MANIFEST_BYTES`, or None otherwise.
  """
  path = root / artifacts.MANIFEST
  try:
    if (
        not artifacts.is_regular(path)
        or path.stat().st_size > artifacts.MAX_MANIFEST_BYTES
    ):
      return None
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError, RecursionError):
    return None
  return (
      data
      if isinstance(data, dict) and isinstance(data.get("tasks"), dict)
      else None
  )


def capped(skipped: list[Any]) -> list[dict[str, Any]]:
  """Caps a task's `skipped` list to `SKIPPED_KEPT` rows plus summary counts.

  Keeps the first `SKIPPED_KEPT` individual rows and aggregates any remaining
  entries by reason as `{"path": "(N more)", "reason": ...}` so large numbers
  of skipped files (such as a virtual environment) do not bloat `manifest.json`.
  Idempotent when applied to an already-capped list.

  Args:
    skipped: List of skipped-file dictionaries for a task.

  Returns:
    The capped list of skipped-file dictionaries.
  """
  rows: list[dict[str, Any]] = []
  rest: dict[str, int] = {}
  for s in skipped:
    if not isinstance(s, dict):
      continue
    more = MORE_RE.fullmatch(str(s.get("path", "")))
    if more is None and len(rows) < SKIPPED_KEPT:
      rows.append(s)
      continue
    skip_reason = str(s.get("reason", ""))
    rest[skip_reason] = rest.get(skip_reason, 0) + (
        int(more.group(1)) if more else 1
    )
  return rows + [{"path": f"({n} more)", "reason": r} for r, n in rest.items()]


def manifest_text(data: dict[str, Any]) -> str:
  """Serializes `data` as `manifest.json` within `artifacts.MAX_MANIFEST_BYTES`.

  Caps each task's `skipped` rows, formats with indentation when under a
  quarter of the size limit, switches to compact JSON above that, and drops
  `harness.files` if still over the limit. Does not mutate `data`.

  Args:
    data: Manifest dictionary to serialize.

  Returns:
    The serialized JSON string ending with a newline.

  Raises:
    ExportError: If the serialized manifest still exceeds
      `artifacts.MAX_MANIFEST_BYTES`.
  """
  data = json.loads(json.dumps(data))
  for entry in (data.get("tasks") or {}).values():
    if isinstance(entry, dict) and isinstance(entry.get("skipped"), list):
      entry["skipped"] = capped(entry["skipped"])
  text = json.dumps(data, indent=2) + "\n"
  if len(text.encode("utf-8")) > artifacts.MAX_MANIFEST_BYTES // 4:
    text = json.dumps(data, separators=(",", ":")) + "\n"
  if (
      len(text.encode("utf-8")) > artifacts.MAX_MANIFEST_BYTES
      and isinstance(data.get("harness"), dict)
      and data["harness"].get("files") is not None
  ):
    data["harness"]["files"] = None
    text = json.dumps(data, separators=(",", ":")) + "\n"
  if len(text.encode("utf-8")) > artifacts.MAX_MANIFEST_BYTES:
    raise ExportError(
        f"{artifacts.MANIFEST} would pass {artifacts.MAX_MANIFEST_BYTES >> 20}"
        " MiB (its tasks list too many files with long paths), so the server"
        " could not read it"
    )
  return text


def write_manifest(root: pathlib.Path, data: dict[str, Any]) -> None:
  """Atomically writes `data` to `root/manifest.json`.

  Args:
    root: Path to the artifacts directory.
    data: Manifest dictionary to write.
  """
  text = manifest_text(data)
  tmp = root / f"{STAGING}manifest-{uuid.uuid4().hex[:8]}.json"
  try:
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, root / artifacts.MANIFEST)
  except BaseException:
    # a full disk: no half-written manifest left among the artifacts
    tmp.unlink(missing_ok=True)
    raise


@contextlib.contextmanager
def entry_lock(entry: pathlib.Path):
  """Acquires an exclusive flock on `entry` while merging its artifacts.

  Ensures concurrent runs of the same entry (for example, two `--tasks`
  subsets finishing together) serialize their merges without leaving a lock
  file inside `entry`.

  Args:
    entry: Path to the entry directory to lock.
  """
  fd = os.open(entry, os.O_RDONLY)
  try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    yield
  finally:
    os.close(fd)  # closing the descriptor releases the lock


class ExportError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when exporting or importing artifacts fails."""


class RunUnreadable(ExportError):  # pylint: disable=g-bad-exception-name
  """Raised when the run's side of an export cannot be read.

  Indicates that nothing was written and the permission or missing-file fix is
  on the run folder, which the evaluation server never reads.
  """


def reason(exc: Exception) -> str:
  """Formats an exception for user-facing export failure messages.

  Args:
    exc: The exception raised during export.

  Returns:
    The message directly for `ExportError`, or `<Type>: <message>` otherwise.
  """
  return (
      str(exc)
      if isinstance(exc, ExportError)
      else f"{type(exc).__name__}: {exc}"
  )


def into_entry(
    src: pathlib.Path,
    entry: pathlib.Path,
    dev: dict[str, artifacts.DevTask],
    notes: list[str] | None = None,
    history: bool = True,
    run: bool = False,
) -> list[str]:
  """Copies task artifact folders from `src` into `entry/artifacts/`.

  Replaces exported task folders while preserving other tasks' folders and
  merging their manifests. Fits the combined folder to the server's limits via
  `fit_total()` and optionally checks git history size. Rolls back task folder
  changes if interrupted before `manifest.json` is replaced.

  Args:
    src: Source `artifacts/` directory of a run.
    entry: Destination entry directory.
    dev: Mapping of dev-set task names to `DevTask` definitions.
    notes: Optional list to append size and git history warnings.
    history: Whether to check the git history size of `entry`.
    run: Whether `entry` is a run directory rather than a committed entry.

  Returns:
    List of task names copied into `entry/artifacts/`.
  """
  with entry_lock(entry):
    return _into_entry(
        src, entry, dev, notes if notes is not None else [], history, run
    )


def _into_entry(
    src: pathlib.Path,
    entry: pathlib.Path,
    dev: dict[str, artifacts.DevTask],
    notes: list[str],
    history: bool,
    run: bool,
) -> list[str]:
  """Merges task folders from `src` into `entry/artifacts/` under `entry_lock`.

  Args:
    src: Source `artifacts/` directory of a run.
    entry: Destination entry directory.
    dev: Mapping of dev-set task names to `DevTask` definitions.
    notes: List to append size and git history warnings.
    history: Whether to check the git history size of `entry`.
    run: Whether `entry` is a run directory rather than a committed entry.

  Returns:
    List of task names copied into `entry/artifacts/`.

  Raises:
    RunUnreadable: If files in `src` cannot be read.
    ExportError: If `entry/artifacts/` is invalid, unreadable, or cannot be
      updated.
  """
  new = read_manifest(src) or {"tasks": {}}
  todo = [
      t
      for t in sorted(new["tasks"])
      if t in dev and artifacts.is_folder(src / t)
  ]
  unread: list[tuple[str, str]] = []
  for t in todo:
    unread.extend(artifacts.unreadable_under(src / t, f"{src.name}/{t}"))
  if unread:
    raise RunUnreadable(
        artifacts.unreadable_stop(
            src.parent,
            unread,
            f"the export cannot copy them from the run ({src.parent})",
            commit=False,
        )
    )
  root = entry / artifacts.ROOT
  st = artifacts.lstat(root)
  if st is not None and not stat.S_ISDIR(st.st_mode):
    raise ExportError(
        f"{root} is not a folder (a link or a file): remove it, and export"
        " again"
    )
  blocked = artifacts.unreadable_under(root) if st is not None else []
  if blocked:  # what it would replace or keep there is not all readable
    raise ExportError(
        artifacts.unreadable_stop(
            entry, blocked, "the export cannot write into them", commit=not run
        )
    )
  root.mkdir(exist_ok=True)
  old = read_manifest(root) or {"tasks": {}}
  # what a killed export left
  for name in sorted(n for n in os.listdir(root) if n.startswith(STAGING)):
    kind, _, rest = name.removeprefix(STAGING).partition("-")
    task = rest.rpartition("-")[0]
    if kind == "old":
      if task and artifacts.lstat(root / task) is None:
        # an earlier task folder it set aside goes back in its place
        os.replace(root / name, root / task)
    elif (
        kind == "drop"
        and task
        and artifacts.is_folder(root / task)
        and artifacts.lstat(root / task / artifacts.TRAJECTORY) is None
        and not _recorded(root, name, task, old)
    ):
      os.replace(root / name, root / task / artifacts.TRAJECTORY)
    else:
      capture.remove(root / name)
  _layout(root)
  kept = {
      t: e
      for t, e in old["tasks"].items()
      if t not in todo and artifacts.is_folder(root / t)
  }
  merged = {
      "format": artifacts.FORMAT,
      "dev_set": new.get("dev_set", old.get("dev_set")),
      "created": _now(),
      "harness": new.get("harness", old.get("harness")),
      "tasks": {**kept, **{t: new["tasks"][t] for t in todo}},
  }
  # one that cannot be written stops the export before any folder moves
  manifest_text(merged)
  xid = uuid.uuid4().hex[:8]
  before = _inode(root / artifacts.MANIFEST)
  # every change, logged before it is made: (path, where what was there waits),
  # or (path, None) for a path it makes
  log: list[tuple[pathlib.Path, pathlib.Path | None]] = []
  try:
    for task in todo:
      copy = root / f"{STAGING}new-{task}-{xid}"
      log.append((copy, None))
      try:
        shutil.copytree(src / task, copy, symlinks=True)
      # files of the run it could not copy, each with the reason
      except shutil.Error as exc:
        where, _, why = (
            exc.args[0][0] if exc.args and exc.args[0] else ("", "", str(exc))
        )
        raise ExportError(
            f"could not copy {where or src / task} from the run: {why}"
        ) from exc
      if artifacts.lstat(root / task) is not None:
        log.append((root / task, root / f"{STAGING}old-{task}-{xid}"))
        os.replace(*log[-1])
      log.append((root / task, None))
      os.replace(copy, root / task)
    added: list[str] = []
    for path in fit_total(root, merged, added):
      log.append((path, root / f"{STAGING}drop-{path.parent.name}-{xid}"))
      os.replace(*log[-1])
    write_manifest(root, merged)
  except BaseException:
    # manifest.json replaced: the export is done
    _settle(log, done=_inode(root / artifacts.MANIFEST) != before)
    raise
  _settle(log, done=True)
  notes += added
  size = _history_bytes(entry) if history else 0
  if size > HISTORY_WARN_BYTES:
    notes.append(
        f"the entry's git history is {size >> 20} MiB: every committed version"
        f" of {artifacts.ROOT}/ stays in the history, and the server refuses a"
        " repository over 1 GiB with its git history, so keep the artifacts"
        " small or squash the history before you submit"
    )
  return todo


def _recorded(
    root: pathlib.Path, name: str, task: str, manifest: dict[str, Any]
) -> bool:
  """Checks whether `manifest` already records omitting a staged trajectory.

  Verifies that `manifest` lists the task's trajectory in `skipped` and was
  written no earlier than the move that set `root / name` aside.

  Args:
    root: Path to the artifacts directory.
    name: Name of the staged trajectory file under `root`.
    task: Name of the task owning the trajectory.
    manifest: Parsed manifest dictionary from `root/manifest.json`.

  Returns:
    True if `manifest` reflects dropping the staged trajectory.
  """
  entry = manifest["tasks"].get(task)
  rows = entry.get("skipped") if isinstance(entry, dict) else None
  listed = isinstance(rows, list) and any(
      isinstance(r, dict) and r.get("path") == artifacts.TRAJECTORY
      for r in rows
  )
  aside, written = artifacts.lstat(root / name), artifacts.lstat(
      root / artifacts.MANIFEST
  )
  return (
      listed
      and bool(aside and written)
      and aside.st_ctime_ns <= written.st_mtime_ns
  )


def _inode(path: pathlib.Path) -> int | None:
  """Returns the inode number of `path`, or None if it does not exist."""
  st = artifacts.lstat(path)
  return st.st_ino if st else None


def _settle(
    log: list[tuple[pathlib.Path, pathlib.Path | None]], done: bool
) -> None:
  """Finalizes or rolls back the file replacements recorded in `log`.

  Once `manifest.json` is written (`done` is True), removes any replaced task
  folders and dropped trajectories waiting in staging. Otherwise, undoes each
  logged change in reverse order.

  Args:
    log: Sequence of `(target_path, staged_backup_path)` pairs.
    done: Whether the manifest replacement succeeded.
  """
  if done:
    for _, waiting in log:
      with contextlib.suppress(OSError):
        if waiting:
          capture.remove(waiting)
    return
  for path, waiting in reversed(log):
    with contextlib.suppress(OSError):
      if waiting:
        os.replace(waiting, path)
      else:
        capture.remove(path)


def _history_bytes(entry: pathlib.Path) -> int:
  """Estimates the on-disk size in bytes of git objects reachable from `HEAD`.

  Measures approximately what a clone of the pushed branch holds on the server,
  excluding other branches, stashes, reflog-only objects, unreachable objects,
  and the Git LFS cache.

  Args:
    entry: Path to the entry git repository.

  Returns:
    Total size in bytes of reachable git objects, or 0 if git cannot tell.
  """
  listed = artifacts.git_output(
      entry, "rev-list", "--objects", "HEAD", timeout=120
  )
  if listed is None:
    return 0
  ids = b"".join(
      line.split(b" ", 1)[0] + b"\n" for line in listed.splitlines() if line
  )
  sizes = artifacts.git_output(
      entry,
      "cat-file",
      "--batch-check=%(objectsize:disk)",
      data=ids,
      timeout=120,
  )
  return (
      sum(int(n) for n in sizes.split() if n.isdigit())
      if sizes is not None
      else 0
  )


def save_record(
    run_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    harness: dict[str, Any],
    dev_digest: str,
) -> None:
  """Writes `run_dir/export.json` before exporting a run's artifacts.

  Allows `kit artifacts --from` to reconstruct the run's `artifacts/` folder
  later if exporting at the end of the run failed.

  Args:
    run_dir: Path to the run directory.
    rows: Per-task trial result dictionaries passed to `export_run()`.
    harness: Provenance metadata dictionary for the harness.
    dev_digest: Digest identifying the dev set.
  """
  tmp = run_dir / f".{RECORD}.tmp"
  tmp.write_text(
      json.dumps({"rows": rows, "harness": harness, "dev_set": dev_digest})
      + "\n",
      encoding="utf-8",
  )
  os.replace(tmp, run_dir / RECORD)


def import_run(entry: pathlib.Path, run: pathlib.Path) -> list[str]:
  """Imports a run's artifacts into `entry/artifacts/`.

  Implements `kit artifacts <entry> --from <run>` using the same merge logic
  as `eval/run.py`.

  Args:
    entry: Path to the target entry directory.
    run: Path to the run directory or its `artifacts/` directory.

  Returns:
    Status and warning lines to print for the user.

  Raises:
    ExportError: If the run's or entry's artifacts cannot be read or written.
  """
  redone: list[Exported] = []
  try:
    # notes about the run's own folder are left out: it is not submitted
    src = run_artifacts(run, [], redone)
  except RunUnreadable as exc:
    raise ExportError(f"the run's artifacts cannot be read: {exc}") from None
  except (ExportError, OSError) as exc:
    raise ExportError(
        "the run's artifacts could not be written again from its trials:"
        f" {reason(exc)}"
    ) from None
  if src is None:
    raise ExportError(
        f"{run} holds no artifacts of a run (a runs/<name>-<time> folder of"
        " eval/run.py, or its artifacts/ folder, with its"
        f" {artifacts.MANIFEST})"
    )
  lines = []
  made_by = (read_manifest(src) or {}).get("harness") or {}
  try:
    name = spec.load(entry).name
  except (OSError, spec.SpecError):
    name = None
  if name and made_by.get("name") and made_by["name"] != name:
    lines.append(
        f"warning: these artifacts come from a run of {made_by['name']}, and"
        f" this entry is {name}"
    )
  notes: list[str] = []
  try:
    copied = into_entry(
        src, entry, artifacts.load_dev_set(artifacts.DEV_SET_FILE), notes
    )
  except RunUnreadable as exc:
    raise ExportError(f"the run's artifacts cannot be read: {exc}") from None
  except (ExportError, OSError) as exc:
    raise ExportError(
        f"the entry's artifacts/ could not be written: {reason(exc)}"
    ) from None
  lines.append(
      f"copied {', '.join(copied) or 'nothing'} into {entry / artifacts.ROOT}"
  )
  # what the run's export.json could not write
  lines += [not_exported(d, entry) for d in redone if d.entry is None]
  return lines + [f"note: {n}" for n in notes]


def run_artifacts(
    path: pathlib.Path,
    notes: list[str] | None = None,
    exported: list[Exported] | None = None,
) -> pathlib.Path | None:
  """Resolves and completes the `artifacts/` directory for a run at `path`.

  If `path` is a run directory (or its `artifacts/` subfolder), ensures any
  measured tasks listed in `export.json` are exported into its `artifacts/`
  folder, appending per-task outcomes to `exported`.

  Args:
    path: Path to a run directory or its `artifacts/` directory.
    notes: Optional list to collect size-fitting notes.
    exported: Optional list to collect `Exported` task results.

  Returns:
    Path to the run's `artifacts/` folder, or None if `path` is not a run or
    run artifacts folder.

  Raises:
    RunUnreadable: If permissions prevent inspecting `path` or its contents.
  """
  folder = next(
      (
          c
          for c in (path / artifacts.ROOT, path)
          if read_manifest(c) is not None
      ),
      None,
  )
  if folder is None:
    blocked = _run_blocked(path)
    if blocked:
      raise blocked
    record = _record(path)
    if record is None:
      return None
    folder = path / artifacts.ROOT
    done = export_run(
        record["rows"],
        folder,
        record.get("harness") or {},
        record.get("dev_set") or "",
        notes,
    )
  else:
    done = _add_missing(folder, notes)
  if exported is not None:
    exported.extend(done)
  return folder


def _add_missing(
    folder: pathlib.Path, notes: list[str] | None
) -> list[Exported]:
  """Exports measured tasks from `export.json` that are missing in `folder`.

  Stages missing tasks in a temporary directory first so the run directory is
  only modified if at least one task can be exported. If merging into `folder`
  fails (for example, on a read-only run folder), marks those tasks as not
  exported and leaves `folder` untouched.

  Args:
    folder: Path to the run's `artifacts/` directory.
    notes: Optional list to collect size-fitting notes.

  Returns:
    List of `Exported` results for the missing tasks.
  """
  # export_run would resolve a relative trial path from the temporary folder
  run = folder.parent.absolute()
  record = _record(run) if folder.name == artifacts.ROOT else None
  held = (read_manifest(folder) or {}).get("tasks") or {}
  rows = [
      {**r, "trial": r.get("trial") and str(run / r["trial"])}
      for r in (record or {}).get("rows", [])
      if r.get("measured") and r.get("task") not in held
  ]
  if not rows:
    return []
  with tempfile.TemporaryDirectory(prefix=f"{STAGING}import-") as staging:
    done = export_run(
        rows,
        pathlib.Path(staging),
        record.get("harness") or {},
        record.get("dev_set") or "",
        notes,
    )
    if any(d.entry for d in done):
      try:
        into_entry(
            pathlib.Path(staging),
            run,
            artifacts.load_dev_set(artifacts.DEV_SET_FILE),
            notes,
            history=False,
            run=True,
        )
      except (ExportError, OSError) as exc:
        why = (
            f"it could not be added to the run's {artifacts.ROOT}/"
            f" ({reason(exc)})"
        )
        return [Exported(d.task, None, why) if d.entry else d for d in done]
  return done


def _record(run: pathlib.Path) -> dict[str, Any] | None:
  """Reads and filters `run/export.json` if present and valid.

  Retains only rows for dev-set tasks whose `trial` value is a string path or
  None, keeping one entry per task.

  Args:
    run: Path to the run directory.

  Returns:
    The filtered export record dictionary, or None if `export.json` is missing
    or invalid.
  """
  try:
    record = (
        json.loads((run / RECORD).read_text(encoding="utf-8"))
        if artifacts.is_regular(run / RECORD)
        else None
    )
  except (OSError, ValueError):
    return None
  if not isinstance(record, dict) or not isinstance(record.get("rows"), list):
    return None
  dev = artifacts.load_dev_set(artifacts.DEV_SET_FILE)
  rows = {
      r["task"]: r
      for r in record["rows"]
      if isinstance(r, dict)
      and isinstance(r.get("task"), str)
      and r["task"] in dev
      and isinstance(r.get("trial") or "", str)
  }
  return {**record, "rows": list(rows.values())}


def _run_blocked(path: pathlib.Path) -> RunUnreadable | None:
  """Checks whether permissions prevent reading run artifacts at `path`.

  Inspects `path`, its `artifacts/` folder, `manifest.json`, and `export.json`.

  Args:
    path: Candidate run directory or `artifacts/` directory path.

  Returns:
    A `RunUnreadable` exception describing the inaccessible paths, or None if
    nothing is blocked (or `path` does not exist).
  """
  try:
    os.lstat(path)
  except PermissionError as exc:
    return RunUnreadable(
        f"this machine cannot reach {path} ({exc.strerror}), so the kit cannot"
        " tell whether it holds a run's artifacts: the kit does not change a"
        " folder above it; ask whoever owns it to let you in, or keep the run"
        " where you can read it, then run again"
    )
  except OSError:
    return None
  if not artifacts.is_folder(path):
    return None
  looked_at = (
      (artifacts.ROOT, True),
      (f"{artifacts.ROOT}/{artifacts.MANIFEST}", False),
      (artifacts.MANIFEST, False),
      (RECORD, False),
  )
  if not os.access(path, os.R_OK | os.X_OK):
    blocked = [""]
  else:
    blocked = []
    for rel, is_dir in looked_at:
      target = path / rel
      if is_dir:
        if artifacts.is_folder(target) and not os.access(
            target, os.R_OK | os.X_OK
        ):
          blocked.append(rel)
      elif artifacts.is_regular(target) and not os.access(target, os.R_OK):
        blocked.append(rel)
  if not blocked:
    return None
  return RunUnreadable(
      artifacts.unreadable_stop(
          path,
          [(rel, os.strerror(errno.EACCES)) for rel in blocked],
          "the kit cannot tell what the run holds",
          commit=False,
      )
  )
