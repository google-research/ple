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

"""Captures deliverable files modified by a harness before the verifier runs.

The kit's Harbor agent lists task deliverables once before the harness starts
(baseline) and once after the session ends, copying out regular files that are
new or modified within size limits. `<trial>/deliverables/` then holds
`files/<path>`, `deleted.txt`, `final_message.md`, and `capture.json`, which
`eval/run.py` exports into the entry's `artifacts/` directory (`export.py`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import re
import shlex
import shutil
import tarfile
import tempfile
from typing import Any, BinaryIO
import uuid

from kit import artifacts

FOLDER = "deliverables"  # in the trial directory
INFO = "capture.json"
LIST_SEC = 120  # one listing command in the task container
CAPTURE_SEC = 300  # the whole copy after the session, whatever it holds
MARK = "@@plebench-listing@@"
FIND_DONE = "@@plebench-find-done@@"
# Lists every entry under the deliverables except folders, never following a
# link, as JSON after MARK: path -> [kind, size, mode, mtime_ns, ctime_ns,
# inode]. A deliverable reached through a link is noted, not listed. The change
# time is the kernel's, so a file the harness rewrote, even with its old content
# and times, counts as changed.
LIST_PY = f"MARK = {MARK!r}\n" + r"""
import json, os, stat, sys
files, notes = {}, []
def kind(m):
    return "f" if stat.S_ISREG(m) else "l" if stat.S_ISLNK(m) else "o"
def add(p, s):
    files[p] = [kind(s.st_mode), s.st_size, s.st_mode & 0o7777, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]
for root in sys.argv[1:]:
    try:
        st = os.lstat(root)
    except OSError:
        continue
    if os.path.realpath(root) != root:
        notes.append([root, "reached through a symbolic link"])
        continue
    if not stat.S_ISDIR(st.st_mode):
        add(root, st)
        continue
    for folder, dirs, names in os.walk(root):
        for n in dirs + names:
            p = os.path.join(folder, n)
            try:
                s = os.lstat(p)
            except OSError:
                continue
            if not stat.S_ISDIR(s.st_mode):
                add(p, s)
print(MARK + json.dumps({"files": files, "notes": notes}))
"""
# The fallback listing with GNU find, where the task container has no python3:
# kinds f, l and others, times as text, one file system (-xdev), and no notes.
# Where find does not take -printf (BusyBox) or is missing, it prints nothing,
# not even FIND_DONE.
LIST_FIND = (
    r"""find / -maxdepth 0 -printf '' 2>/dev/null || exit; for p in "$@"; do """
    r"""find "$p" -xdev ! -type d -printf '%y\t%s\t%m\t%T@\t%C@\t%i\t%p\0' 2>/dev/null; """
    f"done; printf '\\n{FIND_DONE}'"
)


class CaptureError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when listing or copying deliverables out of the container fails."""


def exit_status(out: str) -> int | None:
  """Parses the `rc=N` exit status printed on the last line of `out`.

  Args:
    out: Command output string whose final line may be `rc=<int>`.

  Returns:
    The parsed integer exit code, or None if the last line does not match.
  """
  lines = (out or "").strip().splitlines()
  m = re.fullmatch(r"rc=(\d+)", lines[-1].strip()) if lines else None
  return int(m.group(1)) if m else None


def roots(deliverables: list[str]) -> list[str]:
  """Normalizes deliverable paths to deduplicated absolute paths without `/`.

  Stripping trailing slashes ensures `find` and `lstat` do not dereference
  symlinks at the end of a path.

  Args:
    deliverables: List of deliverable path strings.

  Returns:
    Deduplicated list of normalized absolute paths.
  """
  return list(
      dict.fromkeys(
          d.rstrip("/") or "/" for d in deliverables if d.startswith("/")
      )
  )


def _parse_find(out: str) -> dict[str, Any]:
  """Parses output from `LIST_FIND` into a deliverable listing dictionary.

  Args:
    out: Raw stdout produced by `LIST_FIND`.

  Returns:
    A dictionary with `files`, `notes`, and `tool` keys.
  """
  files = {}
  body, _, _ = out.rpartition("\n" + FIND_DONE)
  for record in body.split("\0"):
    fields = record.split("\t", 6)
    if len(fields) != 7:
      continue
    y, size, mode, mtime, ctime, inode, path = fields
    kind = "f" if y == "f" else "l" if y == "l" else "o"
    files[path] = [kind, int(size), int(mode, 8), mtime, ctime, inode]
  return {"files": files, "notes": [], "tool": "find"}


async def listing(environment, deliverables: list[str]) -> dict[str, Any]:
  """Lists all non-directory entries under `deliverables` in `environment`.

  Uses `python3` inside the container when available, falling back to GNU
  `find`.

  Args:
    environment: The Harbor task environment.
    deliverables: List of deliverable root paths in the container.

  Returns:
    A dictionary `{"files": {path: [kind, size, mode, mtime, ctime, inode]},
    "notes": [[path, reason]], "tool": str}`.

  Raises:
    CaptureError: If neither `python3` nor GNU `find` is available in the
      container.
  """
  paths = roots(deliverables)
  if not paths:
    return {"files": {}, "notes": [], "tool": "none"}
  r = await environment.exec(
      command=(
          f"python3 -I -S -c {shlex.join([LIST_PY])} {shlex.join(paths)}"
          " 2>/dev/null"
      ),
      cwd="/",
      timeout_sec=LIST_SEC,
  )
  out = r.stdout or ""
  if MARK in out:
    with contextlib.suppress(ValueError):
      data = json.loads(
          out[out.index(MARK) + len(MARK) :].strip().splitlines()[0]
      )
      return {
          "files": data["files"],
          "notes": data.get("notes") or [],
          "tool": "python3",
      }
  r = await environment.exec(
      command=f"sh -c {shlex.join([LIST_FIND])} sh {shlex.join(paths)}",
      cwd="/",
      timeout_sec=LIST_SEC,
  )
  out = r.stdout or ""
  if FIND_DONE in out:
    return _parse_find(out)
  raise CaptureError(
      "the task container has neither python3 nor GNU find to list the"
      " deliverables"
  )


def _git_part(path: str) -> str:
  """Returns the `.git` directory prefix of `path`, or `path` itself.

  Args:
    path: Container file path identified as a Git internal file.

  Returns:
    The path up to and including `.git`, or `path` if `.git` is not a segment.
  """
  parts = path.split("/")
  return "/".join(parts[: parts.index(".git") + 1]) if ".git" in parts else path


def compare(
    before: dict[str, Any] | None,
    after: dict[str, Any],
    deliverables: list[str],
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
  """Compares `before` and `after` deliverable listings to find changes.

  Without a baseline (`before` is None), every regular file counts as new. Only
  files accepted by `artifacts.review()` are kept; invalid paths, OS junk files,
  paths beneath a file-only deliverable, Git metadata files, non-regular files,
  and case-colliding paths are recorded in `skipped`.

  Args:
    before: Baseline listing dictionary from before the harness ran, or None.
    after: Post-run listing dictionary after the harness finished.
    deliverables: List of deliverable root paths configured for the task.

  Returns:
    A tuple `(changed, deleted, skipped)` where `changed` maps paths to listing
    entries, `deleted` lists removed regular file paths, and `skipped` lists
    `{"path": str, "reason": str}` dictionaries.
  """
  old = (before or {}).get("files") or {}
  # python3 and find write times differently, so after a change of tool every
  # file counts as changed
  same_tool = before is not None and before.get("tool") == after.get("tool")
  changed, skipped = {}, [
      {"path": p, "reason": why} for p, why in after.get("notes") or []
  ]
  own: dict[str, None] = {}
  for path, entry in sorted(after["files"].items()):
    rel = path.lstrip("/")
    # measured as review() measures it: on the path as committed under the task
    # folder (files/<container path>)
    problem = (
        artifacts.path_problem(f"{artifacts.FILES}/{rel}")
        if path.startswith("/")
        else "is not an absolute path"
    )
    if not artifacts.covered(deliverables, path):
      under = next(
          (
              d
              for d in deliverables
              if not d.endswith("/") and path.startswith(d + "/")
          ),
          None,
      )
      if under and (old.get(path) != entry or not same_tool):
        skipped.append({
            "path": path,
            "reason": (
                f"below {under}, which the task names as a file: only that "
                "file is graded"
            ),
        })
      continue
    if old.get(path) == entry and same_tool:
      continue
    if artifacts.git_own(rel):
      own.setdefault(_git_part(path))
      continue
    if artifacts.junk(rel):
      skipped.append({
          "path": path,
          "reason": (
              "an operating system's own file (.DS_Store, ._*, __MACOSX/, ...),"
              " never graded"
          ),
      })
      continue
    if problem:
      skipped.append({"path": path, "reason": f"the path {problem}"})
    elif entry[0] == "l":
      skipped.append({
          "path": path,
          "reason": "a symbolic link (only regular files are graded)",
      })
    elif entry[0] != "f":
      skipped.append({"path": path, "reason": "not a regular file"})
    else:
      changed[path] = entry
  skipped += [
      {"path": p, "reason": "git's own files are not exported"} for p in own
  ]
  seen: dict[str, str] = {}  # a case-folded file or folder -> the spelling kept
  for path in list(changed):
    # its folders, then itself
    names = [path[:i] for i in range(1, len(path)) if path[i] == "/"] + [path]
    kept = next(
        (seen[q.casefold()] for q in names if seen.get(q.casefold(), q) != q),
        None,
    )
    if kept is not None:
      del changed[path]
      skipped.append({
          "path": path,
          "reason": (
              f"differs only in letter case from {kept}; macOS and Windows"
              " would merge the two, so only that one is exported"
          ),
      })
      continue
    for q in names:
      seen.setdefault(q.casefold(), q)
  deleted = []
  for path, entry in sorted(old.items()):
    rel = path.lstrip("/")
    # deleted: a regular file now gone, or replaced by a link or a special file
    # (neither is exported)
    if (
        entry[0] != "f"
        or (after["files"].get(path) or ["gone"])[0] == "f"
        or artifacts.git_own(rel)
        or artifacts.junk(rel)
        or not artifacts.covered(deliverables, path)
    ):
      continue
    # deleted.txt holds one UTF-8 path per line, so a name review() would refuse
    # cannot go in it
    problem = (
        artifacts.path_problem(path.lstrip("/"))
        if path.startswith("/")
        else "is not an absolute path"
    )
    if problem:
      skipped.append({
          "path": path,
          "reason": (
              f"deleted, but the path {problem}, so deleted.txt cannot list it"
          ),
      })
    else:
      deleted.append(path)
  return changed, deleted, skipped


def within_limits(
    changed: dict[str, Any],
    skipped: list[dict[str, str]],
    room: int = artifacts.MAX_TASK_BYTES,
) -> dict[str, Any]:
  """Selects changed files that fit within per-task file and byte limits.

  Processes smallest files first to maximize the number of captured files,
  appending any files that exceed limits to `skipped`.

  Args:
    changed: Mapping of changed file paths to listing entries.
    skipped: List of skipped file records, mutated in place with rejected files.
    room: Remaining byte budget for files after accounting for
      `final_message.md` and `deleted.txt`.

  Returns:
    Subset of `changed` that fits within the task limits.
  """
  kept, total = {}, 0
  for path, entry in sorted(changed.items(), key=lambda kv: (kv[1][1], kv[0])):
    size = int(entry[1])
    if size > artifacts.MAX_FILE_BYTES:
      skipped.append(
          {"path": path, "reason": f"over {artifacts.MAX_FILE_BYTES >> 20} MiB"}
      )
    elif len(kept) >= artifacts.MAX_TASK_FILES or total + size > room:
      skipped.append({
          "path": path,
          "reason": (
              f"past the task's limit of {artifacts.MAX_TASK_FILES} files and "
              f"{artifacts.MAX_TASK_BYTES >> 20} MiB"
          ),
      })
    else:
      kept[path] = entry
      total += size
  return kept


def _mode(entry: list[Any]) -> str:
  """Returns `'755'` if `entry` has any execute bit set, else `'644'`.

  Args:
    entry: File listing entry `[kind, size, mode, ...]`.

  Returns:
    Normalized octal permission string (`'755'` or `'644'`).
  """
  return "755" if int(entry[2]) & 0o111 else "644"


def _save(
    src: BinaryIO,
    dest: pathlib.Path,
    path: str,
    skipped: list[dict[str, str]],
) -> bool:
  """Writes `src` to `dest/files/<path>` without following symlinks.

  Filenames rejected by the host OS (`artifacts.NAME_ERRNOS`) are appended to
  `skipped` and return False; any other `OSError` (such as a full disk) is
  raised so the capture fails cleanly without overwriting prior artifacts.

  Args:
    src: Binary stream of the file's contents.
    dest: Local capture destination directory.
    path: Container path of the file.
    skipped: List of skipped file records, mutated in place if the OS rejects
      the filename.

  Returns:
    True if the file was written, False if skipped due to a host filename error.

  Raises:
    OSError: If writing fails for a reason other than an invalid filename.
  """
  target = dest / artifacts.FILES / path.lstrip("/")
  try:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
    )
  except OSError as exc:
    if exc.errno not in artifacts.NAME_ERRNOS:
      raise
    skipped.append({
        "path": path,
        "reason": f"cannot be written on this machine ({exc.strerror})",
    })
    return False
  with os.fdopen(fd, "wb") as fh:
    shutil.copyfileobj(src, fh, 1 << 20)
  return True


def _extract(
    archive: pathlib.Path,
    wanted: dict[str, Any],
    dest: pathlib.Path,
    skipped: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
  """Extracts regular files in `wanted` from `archive` into `dest/files/`.

  Skips any archive member that is no longer a regular file or has grown larger
  than its listed size.

  Args:
    archive: Path to the downloaded tar archive.
    wanted: Mapping of container paths to expected listing entries.
    dest: Local capture destination directory.
    skipped: List of skipped file records, mutated in place.

  Returns:
    Mapping of written container paths to `{"bytes": int, "mode": str}`.

  Raises:
    CaptureError: If the tar archive cannot be read.
  """
  written = {}
  try:
    with tarfile.open(archive) as t:
      for m in t:
        path = "/" + m.name.lstrip("/")
        entry = wanted.get(path)
        if entry is None or path in written:
          continue
        if not m.isreg() or m.size > int(entry[1]):
          skipped.append(
              {"path": path, "reason": "changed while it was copied out"}
          )
          continue
        with contextlib.closing(t.extractfile(m)) as src:
          if _save(src, dest, path, skipped):
            written[path] = {"bytes": m.size, "mode": _mode(entry)}
  except tarfile.TarError as exc:
    raise CaptureError(f"the archive cannot be read ({exc})") from exc
  return written


async def _container(call):
  """Awaits a container operation, converting any exception to `CaptureError`.

  Args:
    call: Awaitable container operation.

  Returns:
    The result of awaiting `call`.

  Raises:
    CaptureError: If `call` raises an exception.
  """
  try:
    return await call
  # Docker's or the container's failure
  except Exception as exc:  # noqa: BLE001
    raise CaptureError(f"{type(exc).__name__}: {exc}") from exc


async def _copy_out(
    environment,
    wanted: dict[str, Any],
    dest: pathlib.Path,
    skipped: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
  """Copies `wanted` files out of `environment` into `dest/files/`.

  First attempts to pack all wanted files into a single tar archive inside the
  container and extract it locally; if the container tar step fails, falls back
  to downloading files individually via `_one_by_one`.

  Args:
    environment: The Harbor task environment.
    wanted: Mapping of container paths to listing entries to copy out.
    dest: Local capture destination directory.
    skipped: List of skipped file records, mutated in place.

  Returns:
    Mapping of written container paths to `{"bytes": int, "mode": str}`.
  """
  tag = uuid.uuid4().hex[:12]
  remote_list, remote_tar = (
      f"/tmp/.plebench-capture-{tag}.list",
      f"/tmp/.plebench-capture-{tag}.tar",
  )
  with tempfile.TemporaryDirectory(prefix="plebench-capture-") as tmp:
    names = pathlib.Path(tmp) / "list"
    names.write_bytes(
        b"\0".join(p.lstrip("/").encode("utf-8") for p in wanted) + b"\0"
    )
    try:
      await _container(environment.upload_file(str(names), remote_list))
      r = await _container(
          environment.exec(
              # --hard-dereference: a second hard link to a file is stored as a
              # file too, not as a link member
              command=(
                  f"tar -cf {remote_tar} --null --no-recursion"
                  f" --hard-dereference -C / -T {remote_list} 2>/dev/null; echo"
                  ' "rc=$?"'
              ),
              cwd="/",
              timeout_sec=CAPTURE_SEC,
          )
      )
      if exit_status(r.stdout) not in (0, 1):  # 1: a file changed meanwhile
        raise CaptureError("tar failed")
      local = pathlib.Path(tmp) / "capture.tar"
      await _container(environment.download_file(remote_tar, str(local)))
      noted: list[dict[str, str]] = []
      written = _extract(local, wanted, dest, noted)
      skipped += noted
      return written
    # no tar, no writable /tmp, an archive that cannot be read
    except CaptureError:
      shutil.rmtree(dest / artifacts.FILES, ignore_errors=True)
      return await _one_by_one(
          environment, wanted, dest, pathlib.Path(tmp), skipped
      )
    finally:
      with contextlib.suppress(Exception):
        await environment.exec(
            command=f"rm -f {remote_list} {remote_tar}", cwd="/", timeout_sec=30
        )


async def _one_by_one(
    environment,
    wanted: dict[str, Any],
    dest: pathlib.Path,
    tmp: pathlib.Path,
    skipped: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
  """Downloads `wanted` files one at a time when tar archiving is unavailable.

  Verifies each downloaded file as `_extract` does. If storing a file fails on
  the local machine (e.g., out of disk space), raises `CaptureError` so earlier
  task artifacts are preserved.

  Args:
    environment: The Harbor task environment.
    wanted: Mapping of container paths to listing entries.
    dest: Local capture destination directory.
    tmp: Temporary directory for staging individual downloads.
    skipped: List of skipped file records, mutated in place.

  Returns:
    Mapping of written container paths to `{"bytes": int, "mode": str}`.

  Raises:
    CaptureError: If a downloaded file cannot be stored on the local machine.
  """
  written = {}
  for n, (path, entry) in enumerate(wanted.items()):
    # a fresh name: what a failed copy left can never be taken for this one
    part = tmp / f"one-{n}"
    try:
      await environment.download_file(path, str(part))
      if part.is_symlink() or not part.is_file():
        why = "not a regular file"
      elif part.stat().st_size > int(entry[1]):
        why = "it changed while it was copied out"
      else:
        why = ""
    # this file's copy failed in the container, in Docker or here
    except Exception as exc:  # pylint: disable=broad-exception-caught
      local = re.search(re.escape(str(part)) + r": ([^\n.]+)", str(exc))
      if local or isinstance(exc, OSError):
        raise CaptureError(
            f"{path} could not be stored on this machine "
            f"({local[1] if local else exc.strerror or type(exc).__name__})"
        ) from exc
      why = f"{type(exc).__name__}: {exc}"
    if why:
      skipped.append(
          {"path": path, "reason": f"could not be copied out ({why})"}
      )
    else:
      with open(part, "rb") as src:
        if _save(src, dest, path, skipped):
          written[path] = {"bytes": part.stat().st_size, "mode": _mode(entry)}
    remove(part)
  return written


def remove(path: pathlib.Path) -> None:
  """Removes a file, symlink, or directory at `path` without following symlinks.

  Args:
    path: Filesystem path to remove.
  """
  if artifacts.is_folder(path):
    shutil.rmtree(path, ignore_errors=True)
  else:
    path.unlink(missing_ok=True)


def write_text(path: pathlib.Path, text: str) -> None:
  """Atomically replaces any file or symlink at `path` with UTF-8 `text`.

  Args:
    path: Target file path.
    text: String content to write as UTF-8.
  """
  path.unlink(missing_ok=True)
  fd = os.open(
      path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
  )
  with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
    fh.write(text)


def start(dest: pathlib.Path) -> None:
  """Creates a clean capture directory at `dest`, removing any prior contents.

  Args:
    dest: Path to the capture directory to initialize.
  """
  if dest.is_symlink() or dest.is_file():
    dest.unlink()
  elif dest.exists():
    shutil.rmtree(dest)
  dest.mkdir(parents=True)


async def capture(
    environment,
    deliverables: list[str],
    before: dict[str, Any] | None,
    dest: pathlib.Path,
    final_message: str,
    note: str = "",
) -> dict[str, Any]:
  """Copies modified deliverables, final message, and `capture.json` to `dest`.

  Only `asyncio.CancelledError` is propagated; any other error is recorded in
  `capture.json` under `"error"` so `eval/run.py` skips exporting artifacts for
  the task without failing the trial outcome.

  Args:
    environment: The Harbor task environment.
    deliverables: List of deliverable root paths in the container.
    before: Baseline listing from before the harness ran, or None.
    dest: Initialized capture directory prepared by `start()`.
    final_message: Final message text reported by the harness.
    note: Optional note explaining why no baseline was recorded.

  Returns:
    The capture summary dictionary written to `capture.json`.

  Raises:
    asyncio.CancelledError: If the capture task is cancelled while running.
  """
  info: dict[str, Any] = {
      "files": {},
      "deleted": [],
      "skipped": [],
      "baseline": before is not None,
      "note": note or None,
      "error": None,
  }
  try:
    write_text(dest / artifacts.FINAL_MESSAGE, final_message)
    after = await listing(environment, deliverables)
    changed, deleted, skipped = compare(before, after, deliverables)
    text = "".join(f"{p}\n" for p in deleted)
    if len(text.encode()) > artifacts.MAX_DELETED_BYTES:
      skipped.append({
          "path": "(deleted files)",
          "reason": (
              f"{len(deleted)} deleted files, more than "
              f"{artifacts.DELETED} can list"
          ),
      })
      text, deleted = "", []
    # the task's limit counts the final message (as the export cuts it) and
    # deleted.txt too
    message = min(
        len(final_message.encode("utf-8", "replace")),
        artifacts.MAX_MESSAGE_BYTES,
    )
    wanted = within_limits(
        changed,
        skipped,
        room=artifacts.MAX_TASK_BYTES - message - len(text.encode()),
    )
    written = (
        await _copy_out(environment, wanted, dest, skipped) if wanted else {}
    )
    info["files"] = {p.lstrip("/"): v for p, v in sorted(written.items())}
    missing = sorted(set(wanted) - set(written) - {s["path"] for s in skipped})
    skipped += [
        {"path": p, "reason": "not in the copy taken out of the container"}
        for p in missing
    ]
    if text:
      write_text(dest / artifacts.DELETED, text)
    info.update(deleted=deleted, skipped=skipped)
  except asyncio.CancelledError:
    info["error"] = "the copy was interrupted"
    raise
  # whatever failed, the trial's outcome stands
  except Exception as exc:  # pylint: disable=broad-exception-caught
    info["error"] = f"{type(exc).__name__}: {exc}"[:500]
  finally:
    with contextlib.suppress(OSError):
      write_text(dest / INFO, json.dumps(info, indent=1) + "\n")
  return info
