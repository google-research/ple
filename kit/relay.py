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

"""Evaluator-side relay for the serve protocol.

Connects to the harness over WebSocket, sends the task, executes requests in the
task container, and collects the final message. Used by `kit check` against a
local container and by `eval/` inside Harbor.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import dataclasses
import errno
import functools
import inspect
import json
import os
import re
import shlex
import tempfile
import time
from typing import Any, Awaitable, Callable, Protocol
import urllib.parse
import uuid

from kit import protocol as P
from kit import stack as stack_mod

# Our scripts run with the image's own python3 from the system directories,
# whatever PATH the task image sets; the commands they start get that PATH back
# (TASK_PATH_VAR).
PYTHON = (
    "env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    " python3 -I"
)
TASK_PATH_VAR = "PLEBENCH_TASK_PATH"
SIZE_MARK = "@@plebench-size@@"  # before a read's size: no base64 holds an "@"
# After an exec's output, the cutter's own line: this tag, the request's random
# token and the bytes it cut. Only that line says whether a result is truncated,
# since the command itself could print anything.
CUT_TAG = "@@plebench-cut@@"
EXEC_TAG = "PLEBENCH_EXEC_ID"  # in the environment of everything an exec starts
KILL_GRACE_SEC = 5
EXEC_MARGIN_SEC = 10
PROBE_TIMEOUT_SEC = 30
# Docker copies a staged file (at most a frame's size) into the task container
# in seconds. A copy still running after this is cut, so that it cannot hold its
# request until the time limit, and the request fails (Relay._failure). While
# the container still answers, the task goes on and the harness may send the
# request again; only Docker failing or the container being gone is the
# evaluator's failure.
UPLOAD_TIMEOUT_SEC = 60
WEDGED = "the task container is not answering: out of processes or memory?"
TIMEOUT_EXIT_CODE = 124
# `timeout -k`: 124 after its SIGTERM, 137 after the SIGKILL that follows
TIMED_OUT = (TIMEOUT_EXIT_CODE, 137)
# longer goes through a staged file: Linux caps one argument at 128 KiB
MAX_ARG_BYTES = 100_000
# What an exec's output keeps (a head and a tail, MAX_OUTPUT_BYTES in all); the
# container sends no more than that.
HEAD_BYTES = P.MAX_OUTPUT_BYTES * 2 // 3
# room for the cut marker and the cutter's own line
TAIL_BYTES = P.MAX_OUTPUT_BYTES - HEAD_BYTES - 64 - 80
OBSERVATION_CHARS = 20_000  # of each result, in the run's record
ARGUMENT_CHARS = 2_000  # of each request's command, path or URL, likewise
RECEIVE_QUEUE = 4  # frames websockets holds before the relay reads them
# wait for cut-short requests' commands to stop (an executor's stop takes up to
# 25 s)
STOPS_SEC = 30
# between attempts to connect to a server that does not accept yet
CONNECT_RETRY_SEC = 1
ALIVE_CHECK_SEC = 5  # how often such a server is asked whether it still runs
# A chatty harness (a log frame per line of its agent's stderr) or a long run
# must not grow the evaluator without bound: of the run's log and of its steps,
# about this many characters from the start and as many from the end stay.
LOG_KEEP_CHARS = 1_000_000
STEPS_KEEP_CHARS = 8_000_000
_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")

# Runs a command under `timeout` with stderr folded into stdout, keeping the
# head and tail of its output, until it exits and for one second more, for what
# a background child still writes. timeout and bash are found on this script's
# PATH; the command runs with the task container's own.
_CUTTER = """
import os, select, shutil, subprocess, sys, time
head_max, tail_max, kill, limit, how, what = sys.argv[1:7]
token = sys.argv[7] if len(sys.argv) > 7 else ""
head_max, tail_max = int(head_max), int(tail_max)
argv = [
    shutil.which("timeout") or "timeout",
    "-k",
    kill,
    limit,
    shutil.which("bash") or "bash",
]
argv += ["-c", what] if how == "-c" else [what]
env = dict(os.environ)
env["PATH"] = env.pop("PLEBENCH_TASK_PATH", env.get("PATH", ""))
p = subprocess.Popen(
    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
)
fd = p.stdout.fileno()
head, tail, dropped, until = bytearray(), bytearray(), 0, None
while until is None or time.monotonic() < until:
    if until is None and p.poll() is not None:
        until = time.monotonic() + 1
    if not select.select([fd], [], [], 0.2)[0]:
        continue
    chunk = os.read(fd, 65536)
    if not chunk:
        break
    n = min(len(chunk), head_max - len(head))
    head += chunk[:n]
    tail += chunk[n:]
    if len(tail) > tail_max:
        dropped += len(tail) - tail_max
        del tail[:-tail_max]
out = sys.stdout.buffer
out.write(head)
if dropped:
    out.write(b"\\n... [the middle of this output was cut] ...\\n")
out.write(tail)
rc = p.wait()
if token:
    out.write(b"\\n@@plebench-cut@@%s %d\\n" % (token.encode(), dropped))
out.flush()
sys.exit(rc if rc >= 0 else 128 - rc)
"""

# Stops every process in the task container whose environment holds an exec's
# tag (the command and all it started), TERM, then KILL: a request the time
# limit cut short must not go on changing the task's files. A tag matches as a
# prefix: tags have one length, so a whole tag matches only itself, and an empty
# one matches every exec's. It runs as the command's user, the one that may read
# those processes' environments.
_STOP_TAGGED = """
import os, signal, sys, time
tag = ("PLEBENCH_EXEC_ID=" + sys.argv[1]).encode()
def tagged():
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open("/proc/%s/environ" % pid, "rb") as f:
                    if any(v.startswith(tag) for v in f.read().split(b"\\0")):
                        yield int(pid)
            except OSError:
                pass
for sig in (signal.SIGTERM, signal.SIGKILL):
    pids = list(tagged())
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass
    if not pids:
        break
    time.sleep(2)
"""

# One `http` request from inside the task container, never through a proxy its
# environment names. A redirect is not followed: the 3xx goes back to the
# harness as it came, so every URL fetched is one mcp_allows has checked.
_HTTP = """
import base64, json, sys, urllib.error, urllib.request
arg = sys.argv[1]
raw_req = open(arg[1:], "rb").read() if arg.startswith("@") else arg
req = json.loads(base64.b64decode(raw_req))
cap = int(sys.argv[2])
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)
r = urllib.request.Request(
    req["url"],
    data=base64.b64decode(req["body"]) or None,
    method=req["method"],
    headers=req["headers"],
)
try:
    try:
        resp = opener.open(r, timeout=req["timeout"])
    except urllib.error.HTTPError as exc:
        resp = exc
    body = resp.read(cap + 1)
    out = {
        "status": getattr(resp, "status", None) or resp.code,
        "headers": dict(resp.headers.items()),
        "body_b64": base64.b64encode(body[:cap]).decode(),
        "truncated": len(body) > cap,
    }
except Exception as exc:
    out = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out))
"""


class Executor(Protocol):
  """Protocol for running bash commands and uploading files in a task container.

  The relay runs its commands under their own `timeout`, so `timeout_sec` is an
  outer bound: an exec that outlives it raises, as does one Docker cannot run at
  all (`ExecutorError` for Docker's own errors; see `docker_failure`), and
  `Relay._failure` decides whose failure that is. A command cut short (cancelled
  or past its time) is stopped inside the container too (`stop_command`).
  `upload` copies a local file in; the relay cancels one that outlives
  `UPLOAD_TIMEOUT_SEC`.

  Two optional methods may also be provided:
  - `async healthy() -> bool`: True when Docker answers and the task container
    is still present (running or OOM-killed). An executor without it is treated
    as unable to tell.
  - `async answers() -> bool`: The executor's own probe that the container can
    run a trivial command, which `Relay._reachable` uses instead of running
    `true` via `exec`.
  """

  async def exec(
      self, command: str, *, cwd: str, timeout_sec: int, user: str | None = None
  ) -> tuple[int, str]:
    ...

  async def upload(self, local: str, remote: str) -> None:
    ...


class ExecutorError(stack_mod.InfraError):
  """Raised when the task container could not run a request.

  When an executor raises this exception, the relay decides whose failure it is
  (`_failure`); when the relay raises it, it is the evaluator's failure.
  `reachable` is the result of a container probe the executor already ran
  (`None` if it did not run one), which `_failure` uses instead of probing
  again.
  """

  def __init__(self, message: str = "", *, reachable: bool | None = None):
    super().__init__(message)
    self.reachable = reachable


# The first line docker or docker compose prints when it could not run a command
# at all (the container stopped or gone, the daemon unreachable, the runtime
# refusing): the exit code then comes from docker, not the command.
DOCKER_FAILURE = re.compile(
    r"(Error response from daemon: |Error: No such container|OCI runtime exec"
    r" failed: |"
    r'service "[^"]+" is not running|Cannot connect to the Docker daemon|error'
    r" during connect)"
)


def docker_failure(code: int, output: str) -> str | None:
  """Extracts Docker's own error message when `docker exec` never ran a command.

  `output` must be text the command itself could not have written: either
  `docker exec`'s own stderr (since the cutter folds the command's stderr into
  stdout), or, when stdout and stderr are merged (Harbor), output verified by
  probing the container (`HarborExecutor`).

  Args:
    code: Exit code from the `docker exec` or `docker compose exec` process.
    output: Output text to inspect for a Docker daemon or runtime failure.

  Returns:
    The truncated Docker error line if `code` is non-zero and `output` matches a
    Docker failure pattern, or `None` otherwise.
  """
  if code == 0:
    return None
  first = next((line for line in output.splitlines() if line.strip()), "")
  return first.strip()[:500] if DOCKER_FAILURE.match(first.strip()) else None


def stop_command(tag: str) -> str:
  """Builds the command to terminate container processes matching `tag`.

  Stops every process inside the task container whose environment holds `tag` in
  `EXEC_TAG`, or every exec's processes when `tag` is empty. Must be run as the
  same user that started the exec.

  Args:
    tag: Execution identifier prefix to match against `PLEBENCH_EXEC_ID`.

  Returns:
    A shell command string that sends SIGTERM followed by SIGKILL to matching
    processes.
  """
  return f"{PYTHON} -c {shlex.quote(_STOP_TAGGED)} {shlex.quote(tag)}"


class RequestError(ValueError):
  """Raised when a request itself is invalid and cannot be executed.

  Covers harness-side issues such as an argument exceeding system limits or
  containing NUL bytes.
  """


class OutputOverrun(RequestError):  # pylint: disable=g-bad-exception-name
  """Raised when a command writes past `MAX_EXEC_READ_BYTES` to `docker exec`.

  The in-container cutter keeps normal command output bounded, but a background
  process started by the harness can bypass the cutter and write directly to
  `docker exec`'s stdout or stderr.
  """

  def __str__(self) -> str:
    return (
        f"the command sent more than {P.MAX_EXEC_READ_BYTES >> 20} MiB back,"
        " past the output cutter, so it was stopped"
    )


async def _read_both(proc) -> tuple[bytes, bytes]:
  """Reads stdout and stderr from `proc` up to `P.MAX_EXEC_READ_BYTES` total.

  If the combined output exceeds `P.MAX_EXEC_READ_BYTES`, kills the process,
  drains the remaining pipe output so `wait()` can complete, and raises
  `OutputOverrun`.

  Args:
    proc: An `asyncio.subprocess.Process` with piped stdout and stderr.

  Returns:
    A `(stdout, stderr)` tuple of raw bytes read from the process.

  Raises:
    OutputOverrun: If the combined stdout and stderr exceed
      `P.MAX_EXEC_READ_BYTES`.
  """
  left = [P.MAX_EXEC_READ_BYTES]

  async def take(stream) -> bytes:
    data = bytearray()
    while chunk := await stream.read(1 << 16):
      left[0] -= len(chunk)
      if left[0] >= 0:
        data += chunk
      elif proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
          proc.kill()
    return bytes(data)

  out, err = await asyncio.gather(take(proc.stdout), take(proc.stderr))
  await proc.wait()
  if left[0] < 0:
    raise OutputOverrun
  return out, err


def wrap(
    command: str, timeout_sec: int, *, file: bool = False, token: str = ""
) -> str:
  """Builds the in-container wrapper command that runs `command` with a timeout.

  Runs `command` (or the script file it names) under `timeout` with the task
  container's `PATH`, truncating output in the container to a head and tail.
  When `token` is provided, the cutter appends a trailer line (`CUT_TAG`, the
  token, and the number of dropped bytes) that `cut_trailer()` strips.

  Args:
    command: Shell command string or path to a script file inside the container.
    timeout_sec: Time limit in seconds passed to `timeout`.
    file: Whether `command` is a script file path rather than an inline `-c`
      command.
    token: Optional per-request token for the truncation trailer line.

  Returns:
    The shell command string to execute inside the container.
  """
  return (
      f'{TASK_PATH_VAR}="$PATH" {PYTHON} -c {shlex.quote(_CUTTER)} {HEAD_BYTES}'
      f" {TAIL_BYTES} "
      f"{KILL_GRACE_SEC} {timeout_sec} {'file' if file else '-c'}"
      f" {shlex.quote(command)}" + (f" {token}" if token else "")
  )


OUTPUT_CUT = b"\n... [output cut] ...\n"
# what _CUTTER writes between its head and tail
CUTTER_MARK = b"\n... [the middle of this output was cut] ...\n"
REPLACEMENT = "\ufffd".encode()


def cutter_mark(data: bytes, mark: bytes, head: int) -> int:
  """Finds the byte offset of the in-container cutter's `mark` within `data`.

  Because the command itself might print `mark` (e.g., `cat` of earlier output),
  this selects the occurrence whose prefix length is closest to `head` bytes,
  accounting for UTF-8 replacement characters (three bytes in UTF-8 representing
  what was typically one raw byte in the container).

  Args:
    data: UTF-8 encoded command output containing head, `mark`, and tail.
    mark: Marker byte sequence inserted by the cutter.
    head: Target head byte count kept by the cutter before `mark`.

  Returns:
    The starting byte index of `mark` in `data`, or -1 if `mark` is not present.
  """
  best, gap, seen, replaced = -1, None, 0, 0
  at = data.find(mark)
  while at >= 0:
    replaced += data.count(REPLACEMENT, seen, at)
    seen = at
    off = abs(at - 2 * replaced - head)
    if gap is None or off < gap:
      best, gap = at, off
    at = data.find(mark, at + 1)
  return best


def fit_output(data: bytes, cut: bool = False) -> bytes:
  """Trims valid UTF-8 `data` to fit within `P.MAX_OUTPUT_BYTES`.

  Keeps a head and tail separated by `OUTPUT_CUT`, splitting on UTF-8 character
  boundaries. Although the in-container cutter bounds raw byte length, non-UTF-8
  bytes expand to 3-byte replacement characters after decoding. When `cut` is
  True, `data` was already split by the cutter around `CUTTER_MARK`, so the head
  and tail are trimmed around a single marker rather than cutting a second time.

  Args:
    data: UTF-8 encoded output bytes to bound.
    cut: Whether `data` already contains `CUTTER_MARK` from the in-container
      cutter.

  Returns:
    Valid UTF-8 bytes of length at most `P.MAX_OUTPUT_BYTES`.
  """
  if len(data) <= P.MAX_OUTPUT_BYTES:
    return data
  room = P.MAX_OUTPUT_BYTES - len(OUTPUT_CUT)
  mark = cutter_mark(data, CUTTER_MARK, HEAD_BYTES) if cut else -1
  head, tail = (
      (data, data)
      if mark < 0
      else (data[:mark], data[mark + len(CUTTER_MARK) :])
  )
  keep_head = min(len(head), max(HEAD_BYTES, room - len(tail)))
  keep_tail = min(len(tail), room - keep_head)
  # a character cut in two at either edge is dropped rather than turned into
  # another replacement character
  return (
      head[:keep_head].decode("utf-8", errors="ignore").encode()
      + OUTPUT_CUT
      + tail[len(tail) - keep_tail :].decode("utf-8", errors="ignore").encode()
  )


def timed_out(code: int, elapsed: float, timeout: int) -> bool:
  """Returns whether `timeout -k` terminated a command after its time limit.

  `timeout -k` exits with 124 after SIGTERM or 137 after SIGKILL. The same exit
  code before `timeout` has elapsed comes from the command itself (or the
  kernel's OOM killer for 137).

  Args:
    code: Process exit code.
    elapsed: Wall-clock seconds the command ran.
    timeout: Configured timeout in seconds.

  Returns:
    True if `code` is in `TIMED_OUT` and `elapsed` reached `timeout`.
  """
  return code in TIMED_OUT and elapsed >= timeout - 0.5


def stopped_by(code: int, elapsed: float, timeout: int) -> str | None:
  """Describes why a read, write, or HTTP helper command was terminated.

  Args:
    code: Process exit code.
    elapsed: Wall-clock seconds the command ran.
    timeout: Configured timeout in seconds.

  Returns:
    A human-readable termination reason if the command timed out or was killed
    with exit code 137 (OOM), or `None` if it exited normally.
  """
  if timed_out(code, elapsed, timeout):
    return f"timed out after {timeout} s"
  return "killed (out of memory?)" if code == 137 else None


def read_script(path: str, mark: str) -> str:
  """Builds the shell script used to execute a `read_file` request.

  Validates that `path` exists, is not a directory, is a regular file after
  following symlinks (avoiding blocking FIFOs or infinite device nodes like
  `/dev/zero`), and is readable. Prints `mark` followed by the `stat -L` byte
  size, then up to `MAX_READ_BYTES + 1` bytes encoded in base64 so the caller
  can detect truncation even for growing or `/proc` files.

  Args:
    path: Container file path to read.
    mark: Marker string printed immediately before the file size.

  Returns:
    A bash command string that validates and base64-encodes the file.
  """
  q = shlex.quote(path)
  # pipefail: a file that passes the checks but cannot be read (an I/O error)
  # fails the read rather than reaching the harness as an empty file
  return (
      f"set -o pipefail; if [ ! -e {q} ]; then echo 'no such file or"
      f" directory'; exit 2; fi; if [ -d {q} ]; then echo 'is a directory';"
      f" exit 2; fi; if [ ! -f {q} ]; then echo 'not a regular file'; exit 2;"
      f" fi; if [ ! -r {q} ]; then echo 'permission denied'; exit 2; fi;"
      f" size=$(stat -L -c %s -- {q}) && printf '\\n%s %s\\n'"
      f' {shlex.quote(mark)} "$size" && head -c {P.MAX_READ_BYTES + 1} -- {q}'
      " | base64 -w0"
  )


def read_result(payload: str, size: int) -> tuple[str, int, bool]:
  """Processes the base64 payload and size returned by `read_script`.

  If `payload` decodes to more than `P.MAX_READ_BYTES` bytes, trims it to
  `P.MAX_READ_BYTES` by re-encoding only the final 3-byte group and marks the
  result as truncated.

  Args:
    payload: Base64-encoded file content of up to `P.MAX_READ_BYTES + 1` bytes.
    size: File size in bytes reported by `stat -L`.

  Returns:
    A `(content_b64, size, truncated)` tuple where `size` is at least the number
    of bytes read.
  """
  read = len(payload) // 4 * 3 - payload[-2:].count("=")
  if read > P.MAX_READ_BYTES:
    whole, extra = divmod(P.MAX_READ_BYTES, 3)
    # the bytes the cut leaves of it
    group = base64.b64decode(payload[whole * 4 : whole * 4 + 4])[:extra]
    payload = payload[: whole * 4] + base64.b64encode(group).decode()
  return payload, max(size, read), read > P.MAX_READ_BYTES


def read_error(out: str, mark: str, code: int) -> str:
  """Extracts a human-readable error message from a failed `read_script` run.

  Strips the `mark` size line and any partial base64 output (in case `head`
  failed mid-read), falling back to the exit code if no message remains.

  Args:
    out: Combined output from `read_script`.
    mark: Size marker string used in `read_script`.
    code: Exit code of the failed command.

  Returns:
    A cleaned error description string.
  """
  said = re.sub(re.escape(mark) + r" ?\d*", " ", out)
  said = re.sub(r"[A-Za-z0-9+/=]{16,}", " ", said)
  lines = (" ".join(line.split()) for line in said.splitlines())
  return (
      "\n".join(line for line in lines if line)[:2000]
      or f"the read failed (exit {code})"
  )


def cut_trailer(raw: str, token: str) -> tuple[str, int | None]:
  """Splits the in-container cutter's trailer line from `raw` command output.

  Only the final trailer line matching `token` is recognized, since the command
  output precedes it and each request uses a fresh random token.

  Args:
    raw: Raw stdout/stderr string returned from the container.
    token: Unique token passed to `wrap()` for this request.

  Returns:
    A `(output, dropped_bytes)` tuple, or `(raw, None)` if no valid trailer line
    was written (e.g., if the cutter was killed before finishing).
  """
  head, sep, line = raw.rpartition(f"\n{CUT_TAG}{token} ")
  if not sep:
    return raw, None
  try:
    return head, int(line.split()[0])
  except (IndexError, ValueError):
    return raw, None


@functools.cache
def _no_proxy() -> dict[str, Any]:
  """Returns keyword arguments disabling proxy usage in `websockets.connect`.

  `websockets` 15+ honours `HTTP(S)_PROXY` environment variables by default,
  whereas earlier versions do not accept a `proxy` argument.

  Returns:
    A dict containing `{"proxy": None}` if supported, or `{}` otherwise.
  """
  import websockets  # pylint: disable=g-import-not-at-top

  try:
    return (
        {"proxy": None}
        if "proxy" in inspect.signature(websockets.connect).parameters
        else {}
    )
  except (TypeError, ValueError):
    return {}


def _abort(ws) -> None:
  """Aborts the underlying transport of `ws` immediately.

  Used when the harness stops reading and a normal closing handshake cannot
  complete.

  Args:
    ws: WebSocket connection whose transport should be aborted.
  """
  transport = getattr(ws, "transport", None)
  if transport is not None:
    transport.abort()


def _host_port(url: str) -> str:
  """Formats the host and port of a WebSocket `url` for error messages.

  Args:
    url: WebSocket URL string to parse.

  Returns:
    A `"host:port"` or `"[ipv6]:port"` string, or `"the URL given"` if invalid.
  """
  try:
    parts = urllib.parse.urlsplit(url)
    host, port = parts.hostname, parts.port or (
        443 if parts.scheme == "wss" else 80
    )
  except ValueError:
    return "the URL given"
  if not host:
    return "the URL given"
  return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def safe(text) -> str:
  """Harness-sent text as storable UTF-8 (an unpaired surrogate becomes "?")."""
  return str(text).encode("utf-8", "replace").decode("utf-8")


def _local_copy(data: bytes) -> str:
  """Writes `data` to a temporary file with mode 0644 for `docker cp`.

  Args:
    data: Raw bytes to write to the local staging file.

  Returns:
    The filesystem path to the newly created temporary file.
  """
  fd, local = tempfile.mkstemp(prefix="serve-upload-")
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
    os.chmod(local, 0o644)
  except BaseException:
    os.unlink(local)
    raise
  return local


def _discard_local(writing: asyncio.Future[str]) -> None:
  """Remove the local copy an upload cut short still wrote."""
  if not writing.cancelled() and writing.exception() is None:
    with contextlib.suppress(OSError):
      os.unlink(writing.result())


def neutral_names(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Renames the task's MCP servers to neutral names (`apps`, `apps-2`, ...).

  Args:
    servers: List of MCP server configuration dicts with their original names.

  Returns:
    A new list of server dicts with `"name"` replaced and URLs unchanged.
  """
  return [
      {**s, "name": "apps" if n == 1 else f"apps-{n}"}
      for n, s in enumerate(servers, 1)
  ]


# The egress proxy carries only the harness's own traffic (its connections out
# and its websocket), so a proxy killed for its memory is the harness's fault,
# like a task container its commands wedged.
EGRESS_KILLED = (
    "the egress proxy was killed {when} for using more than {memory} of memory"
    " (only the harness's own traffic goes through it)"
)
# every request the time limit cut short or refused gets this
CANCELLED = "cancelled: the time is up"


class Kept:
  """Bounded buffer keeping head and tail items within character budgets.

  Retains items from the start of a run up to `budget` characters and the latest
  items up to `budget` characters (always keeping at least the very last item),
  while counting any dropped items in the middle.
  """

  def __init__(self, budget: int):
    self.budget, self.dropped = budget, 0
    self.head: list[Any] = []
    self.tail: collections.deque[tuple[Any, int]] = collections.deque()
    self._head_chars = self._tail_chars = 0

  def add(self, item, size: int) -> None:
    if not self.tail and self._head_chars + size <= self.budget:
      self.head.append(item)
      self._head_chars += size
      return
    self.tail.append((item, size))
    self._tail_chars += size
    while self._tail_chars > self.budget and len(self.tail) > 1:
      self._tail_chars -= self.tail.popleft()[1]
      self.dropped += 1

  def items(self, cut) -> list[Any]:
    """Returns kept items, calling `cut(dropped)` where items were omitted."""
    return [
        *self.head,
        *([cut(self.dropped)] if self.dropped else []),
        *(item for item, _ in self.tail),
    ]


@dataclasses.dataclass
class Outcome:
  """Records the status, final message, metrics, and trajectory of a run."""

  status: str = "pending"  # pending | running | completed | error | interrupted
  reason: str = "not started"
  final_message: str = ""
  requests: int = 0
  # the requests by type, as {"exec": 3, "http": 1}
  kinds: dict[str, int] = dataclasses.field(default_factory=dict)
  malformed: int = 0
  usage: dict[str, int] = dataclasses.field(
      default_factory=lambda: {
          "input_tokens": 0,
          "output_tokens": 0,
          "cache_read_tokens": 0,
      }
  )
  models: list[str] = dataclasses.field(default_factory=list)
  infra: str = ""
  connect_sec: float | None = None
  # the harness sent done in the grace after a cancel
  done_after_cancel: bool = False
  _log: Kept = dataclasses.field(
      default_factory=lambda: Kept(LOG_KEEP_CHARS), repr=False
  )
  _steps: Kept = dataclasses.field(
      default_factory=lambda: Kept(STEPS_KEEP_CHARS), repr=False
  )

  def note(self, line: str) -> None:
    self._log.add(line, len(line))

  def add_step(self, step: dict[str, Any]) -> None:
    self._steps.add(step, sum(len(str(v)) for v in step.values()))

  @property
  def log(self) -> list[str]:
    """Returns the run's log lines, bounded at the head and tail."""
    return self._log.items(lambda n: f"... [{n} log line(s) cut here] ...")

  @property
  def steps(self) -> list[dict[str, Any]]:
    """Returns the bounded task and request/result trajectory steps."""
    return self._steps.items(
        lambda n: {"source": "system", "message": f"[{n} step(s) cut here]"}
    )


class Relay:
  """Relays a task between a harness WebSocket and the task container.

  `stack` is the `kit.stack.Stack` when the evaluator started the server: a
  socket that closes early because the egress proxy (which also carries the
  WebSocket) stopped is then treated as an evaluator failure, matching server
  behavior.
  """

  def __init__(
      self,
      executor: Executor,
      *,
      workdir: str,
      mcp_servers: list[dict[str, Any]] | None = None,
      agent_timeout_sec: float | None = None,
      stack=None,
  ):
    self.executor = executor
    self.workdir = workdir
    self.mcp_servers = mcp_servers or []
    self.agent_timeout_sec = agent_timeout_sec
    self.stack = stack
    self.deadline: float | None = None
    self.outcome = Outcome()
    # (id, error) a cancel cut short, answered after the cancel frame
    self._due: list[tuple[str, str]] = []
    # requests a session cut short, stopping their commands
    self._stops: asyncio.Future[Any] | None = None
    # when the grace after a cancel ends, and the socket closes
    self._grace_ends: float | None = None

  def _left(self) -> float | None:
    return None if self.deadline is None else self.deadline - time.monotonic()

  async def _inspect(self, which: str) -> dict[str, Any] | None:
    """Runs `Stack.state(which)` in a worker thread to inspect container state.

    `None` is not treated as stopped nor blamed on the harness.

    Args:
      which: Stack service name to inspect (e.g., `"egress"` or `"serve"`).

    Returns:
      The container's `State` dict from `docker inspect`, or `None` if Docker
      could not report it.
    """
    try:
      return await asyncio.to_thread(self.stack.state, which)
    except stack_mod.InfraError:
      return None

  async def _egress(self, when: str) -> tuple[str, str]:
    """Checks the egress proxy state and attributes any failure.

    Both strings are empty while the proxy runs or when there is no stack (a
    participant-started server). The harness outcome is set when Docker reports
    the proxy was OOM-killed, since only harness traffic uses it
    (`EGRESS_KILLED`); otherwise a stopped proxy is an evaluator failure because
    it also carries the WebSocket.

    Args:
      when: Phrase describing when the check occurred for error messages.

    Returns:
      A `(evaluator_failure, harness_outcome)` tuple of strings.
    """
    if self.stack is None:
      return "", ""
    state = await self._inspect("egress")
    if state is None:
      return "Docker could not tell whether the egress proxy runs", ""
    if state.get("Running"):
      return "", ""
    if state.get("OOMKilled"):
      return "", EGRESS_KILLED.format(when=when, memory=stack_mod.EGRESS_MEMORY)
    return f"the egress proxy stopped {when}", ""

  def start_clock(self) -> None:
    """Starts the task's countdown timer once, including connection time."""
    if self.deadline is None and self.agent_timeout_sec is not None:
      self.deadline = time.monotonic() + self.agent_timeout_sec

  async def connect(
      self,
      url: str,
      alive: Callable[[], Awaitable[bool | None]] | None = None,
      start: float | None = None,
  ):
    """Connects to the harness WebSocket at `url` within `protocol.START_SEC`.

    Tries at least once and retries until `protocol.START_SEC` after `start`
    (defaulting to now). `alive()` checks whether the server container is still
    running; it is queried only after a failed connection attempt, at most every
    `ALIVE_CHECK_SEC`, and once more when the wait ends.

    Args:
      url: WebSocket URL to connect to.
      alive: Optional async callback returning `True` if running, `False` if
        stopped, or `None` if Docker could not tell.
      start: Monotonic timestamp when connection attempts began.

    Returns:
      The connected WebSocket client instance.

    Raises:
      ConnectionError: If the server container exited, the egress proxy was
        OOM-killed by harness traffic, or no WebSocket accepted within
        `protocol.START_SEC`.
      stack_mod.InfraError: If the egress proxy stopped or Docker could not
        determine whether the server container is running.
    """
    import websockets  # pylint: disable=g-import-not-at-top

    self.start_clock()
    start = start or time.monotonic()
    last: Exception | None = None
    # Docker could say whether the server runs, the last time it was asked
    known = True
    asked: float | None = None  # when alive() was last asked
    while True:
      try:
        # a small max_queue lets websocket flow control push back on the harness
        ws = await websockets.connect(
            url,
            max_size=P.MAX_FRAME_BYTES,
            max_queue=RECEIVE_QUEUE,
            ping_interval=None,
            open_timeout=10,
            close_timeout=P.CANCEL_GRACE_SEC,
            **_no_proxy(),
        )
        self.outcome.connect_sec = time.monotonic() - start
        return ws
      except (
          OSError,
          asyncio.TimeoutError,
          websockets.exceptions.WebSocketException,
      ) as exc:
        last = exc
      now = time.monotonic()
      over = now - start >= P.START_SEC
      if alive is not None and (
          over or asked is None or now - asked >= ALIVE_CHECK_SEC
      ):
        asked = now
        state = await alive()
        if state is not None and not state:
          raise ConnectionError(
              "the server exited before accepting a connection"
          )
        known = state is not None
      if over:
        break
      await asyncio.sleep(CONNECT_RETRY_SEC)
    # The egress proxy also carries the websocket, so with it gone the server
    # was never reachable: the evaluator's failure, as on the server, unless the
    # harness's own traffic got it killed. A wait Docker could not judge is the
    # evaluator's failure too.
    trouble, own = await self._egress("before the server could be reached")
    if own:
      raise ConnectionError(own)
    if trouble or not known:
      self.outcome.infra = (
          trouble or "Docker could not say whether the server runs"
      )
      raise stack_mod.InfraError(self.outcome.infra)
    # a server the evaluator started listens on PORT in its container; one the
    # participant started, at its URL
    where = (
        f"on port {P.PORT}"
        if self.stack is not None
        else f"at {_host_port(url)}"
    )
    raise ConnectionError(
        f"no websocket {where} within {P.START_SEC} s ({type(last).__name__}:"
        f" {last})"
    )

  async def run(
      self, ws, instruction: str, cancel_after: float | None = None
  ) -> Outcome:
    """Sends the task over `ws` and relays requests until completion or timeout.

    Args:
      ws: Connected WebSocket to the harness.
      instruction: Task instruction text to send in the initial `task` message.
      cancel_after: Optional override timeout in seconds before cancelling.

    Returns:
      The populated `Outcome` for the session.
    """
    import websockets  # pylint: disable=g-import-not-at-top

    o = self.outcome
    self.start_clock()
    left = self._left()
    o.add_step({"source": "user", "message": instruction})
    if left is not None and left <= 0:
      # connecting used the whole time limit: a task the harness has no time for
      # is not sent
      o.status, o.reason = (
          "interrupted",
          (
              "cancelled (timeout): the time limit ran out before the task was"
              " sent"
          ),
      )
      o.note(o.reason)
      await self._close(ws)
      return o
    if cancel_after is None:
      limit = left
    elif left is None:
      limit = cancel_after
    else:
      limit = min(left, cancel_after)
    try:
      # a socket that closes before the task arrives ends the task
      # (_closed_early says whose failure it is)
      await ws.send(
          json.dumps(
              P.task_message(
                  session=uuid.uuid4().hex[:16],
                  instruction=instruction,
                  workdir=self.workdir,
                  agent_timeout_sec=left,
                  mcp_servers=self.mcp_servers,
              )
          )
      )
      o.status = "running"
      o.note(
          f"task sent: workdir {self.workdir},"
          f" {'no time limit' if left is None else f'{left:.0f} s left'},"
          f" {len(self.mcp_servers)} MCP server(s)"
      )
      await asyncio.wait_for(self._session(ws), limit)
    except websockets.exceptions.ConnectionClosed as exc:
      await self._closed_early(
          f"the connection closed before the task was sent ({exc})"
      )
    except asyncio.TimeoutError:
      await self._cancel(ws, "timeout")
    except asyncio.CancelledError:
      await asyncio.shield(self._cancel(ws, "timeout"))
      raise
    finally:
      # Close the socket as the session ends (done, error, or the end of the
      # grace after a cancel), so the harness's later frames fail instead of
      # going unread. Meanwhile the commands of cut-short requests finish
      # stopping in the task container (at most STOPS_SEC). The verifier must
      # not start before they stop, so a cancel arriving now (Harbor's agent
      # timeout, just after a done near the limit) waits for them too, then
      # propagates.
      ending = asyncio.gather(self._close(ws), self._wait_for_stops())
      try:
        await asyncio.shield(ending)
      except asyncio.CancelledError:
        await stack_mod._outlast(ending)  # pylint: disable=protected-access
        raise
    return o

  async def _close(self, ws) -> None:
    """Closes `ws` within the remaining cancel grace period, else aborts it.

    `websockets` waits for its write buffer to drain before its own
    `close_timeout` applies, so closing a harness that stopped reading would
    otherwise block indefinitely.

    Args:
      ws: WebSocket connection to close.
    """
    left = (
        P.CANCEL_GRACE_SEC
        if self._grace_ends is None
        else self._grace_ends - time.monotonic()
    )
    try:
      await asyncio.wait_for(ws.close(), max(left, 0))
    except asyncio.CancelledError:
      _abort(ws)
      raise
    except Exception:  # pylint: disable=broad-exception-caught
      # the grace ran out, or the socket was already gone
      _abort(ws)

  async def _wait_for_stops(self) -> None:
    """Waits up to `STOPS_SEC` for cut-short commands to stop in the container.

    Best-effort wait so that interrupted commands do not continue modifying task
    files after the session ends.
    """
    if self._stops is None:
      return
    try:
      await asyncio.wait_for(asyncio.shield(self._stops), STOPS_SEC)
    except asyncio.TimeoutError:
      self.outcome.note(
          "requests cut short were still stopping their commands after"
          f" {STOPS_SEC} s"
      )
    self._stops = None

  async def _cancel(self, ws, reason: str) -> None:
    """Sends `cancel`, answers cut-short requests, and drains until `done`.

    Completes within `CANCEL_GRACE_SEC`. A harness that stopped reading its
    results may have filled the socket buffer, so sends are bounded by the same
    timeout. Cut-short commands stop inside the task container concurrently and
    are awaited in `run()` afterwards.

    Args:
      ws: WebSocket connection to the harness.
      reason: Cancellation reason string (e.g., `"timeout"`).
    """
    o = self.outcome
    self._grace_ends = time.monotonic() + P.CANCEL_GRACE_SEC
    o.note(f"cancel ({reason})")
    if o.status == "running":
      o.status, o.reason = "interrupted", f"cancelled ({reason})"
    due, self._due = self._due, []

    async def tell() -> None:
      await ws.send(json.dumps({"type": "cancel", "reason": reason}))
      for rid, error in due:
        await ws.send(json.dumps(P.error_result(rid, error)))
      await self._drain(ws)

    try:
      await asyncio.wait_for(tell(), P.CANCEL_GRACE_SEC)
    except Exception:  # pylint: disable=broad-exception-caught
      # best effort while stopping
      pass

  async def _closed_early(self, reason: str) -> None:
    """Records an early WebSocket closure and attributes responsibility.

    An early close is the harness's own outcome (annotated with how the server
    exited or whether its traffic caused the egress proxy to be OOM-killed),
    unless the egress proxy stopped for another reason or Docker could not
    inspect the containers, which are recorded on `Outcome.infra`.

    Args:
      reason: Initial description of why the connection closed early.
    """
    o = self.outcome
    # at most one inspect of each container (Stack.state)
    if self.stack is not None:
      trouble, own = await self._egress("during the run")
      if own:
        reason += f"; {own}"
      elif not trouble:
        server = await self._inspect("serve")
        # an inspect that failed, or a container removed outside the kit
        if server is None:
          trouble = "Docker could not tell whether the server still runs"
        elif not server.get("Running"):
          reason += (
              f"; the server {stack_mod.exit_wording(server, stack_mod.MEMORY)}"
          )
      if trouble:
        o.infra = o.infra or trouble
    o.status, o.reason = "error", reason
    o.note(reason)

  async def _drain(self, ws) -> None:
    """Handles frames received during the post-cancel grace period.

    `done` or `error` ends the grace period, `log` and `usage` notes are still
    recorded, and any new requests are refused with `CANCELLED`.

    Args:
      ws: WebSocket connection to read from and reply on.
    """
    o = self.outcome
    async for frame in ws:
      try:
        message = P.parse_frame(frame)
      except P.ProtocolError as exc:
        if exc.rid:  # a request of an unknown type: refused, and answered
          await ws.send(json.dumps(P.error_result(exc.rid, CANCELLED)))
        continue
      kind = message["type"]
      if kind == "done":
        o.final_message = safe(message.get("final_message") or "")
        o.done_after_cancel = True
        o.note("done after cancel")
        return
      if kind == "error":
        o.reason += (
            "; then the harness gave up: "
            + safe(message.get("message") or "")[:2000]
        )
        o.note(o.reason)
        return
      if kind in P.NOTES:
        try:
          self._notes(message)
        except P.ProtocolError as exc:
          if exc.rid:  # a malformed usage note with an id: answered, as above
            await ws.send(json.dumps(P.error_result(exc.rid, CANCELLED)))
      elif kind in P.REQUESTS and P.answerable_id(message.get("id")):
        await ws.send(json.dumps(P.error_result(message["id"], CANCELLED)))

  def _notes(self, message: dict[str, Any]) -> None:
    """Records a `log` or `usage` frame into `self.outcome`.

    Args:
      message: Parsed `log` or `usage` protocol frame.
    """
    o = self.outcome
    if message["type"] == "log":
      o.note("harness: " + safe(message.get("text") or "")[: P.MAX_LOG_CHARS])
      return
    usage = P.check_usage(message)
    for key in o.usage:
      o.usage[key] = min(o.usage[key] + usage[key], P.MAX_COUNT)
    if (
        usage["model"]
        and len(o.models) < P.MAX_MODELS
        and safe(usage["model"]) not in o.models
    ):
      o.models.append(safe(usage["model"]))

  async def _session(self, ws) -> None:
    """Reads and dispatches frames over `ws` until the session ends.

    Args:
      ws: Connected WebSocket for the active session.
    """
    import websockets  # pylint: disable=g-import-not-at-top

    o = self.outcome
    slots = asyncio.Semaphore(P.MAX_IN_FLIGHT)
    lock = asyncio.Lock()
    pending: set[asyncio.Task[Any]] = set()
    # each request's task and id, until its result is sent
    inflight: dict[asyncio.Task[Any], str] = {}
    # ids of requests read but not started: one waiting for a slot, and any the
    # socket closed on meanwhile
    waiting: list[str] = []
    # a malformed request's (id, problem), until its result is sent
    owed: list[tuple[str, str]] = []
    # the evaluator failed: the session ends (Outcome.infra)
    failed = asyncio.Event()
    # the socket's end, watched while a request waits for a slot
    closing: asyncio.Future[Any] | None = None
    cancelled = False

    async def reply(
        message: dict[str, Any],
        task: asyncio.Task[Any] | None = None,
        debt: tuple[str, str] | None = None,
    ) -> None:
      async with lock:
        # answered from here on, before the send: websockets hands the frame to
        # the socket before the send's first wait, so a cancel that interrupts
        # the send cannot answer the same id a second time
        inflight.pop(task, None)
        if debt in owed:
          owed.remove(debt)
        try:
          await ws.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
          pass

    async def relay(request: dict[str, Any]) -> None:
      """Runs one request in an acquired slot and frees the slot afterwards.

      Args:
        request: Validated request dict to dispatch.
      """
      try:
        started = time.monotonic()
        try:
          result = await self.dispatch(request)
        except RequestError as exc:
          result = P.error_result(request["id"], str(exc))
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # Docker, a full disk here, a bug: the evaluator's failure
          if isinstance(exc, ExecutorError):
            why = str(exc)
          else:
            why = (
                f"relaying {request['type']} failed: {type(exc).__name__}:"
                f" {exc}"
            )
          o.infra = o.infra or why
          result = P.error_result(
              request["id"], "the evaluator could not run this request"
          )
        try:
          self._record(request, result, time.monotonic() - started)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # a failed record must not keep the result from the harness
          o.note(
              f"recording {request['type']} {request['id']!r} failed:"
              f" {type(exc).__name__}: {exc}"
          )
        await reply(result, asyncio.current_task())
        if o.infra:
          failed.set()
      finally:
        slots.release()

    async def pay(debts: list[tuple[str, str]]) -> None:
      for debt in list(debts):
        await reply(P.error_result(*debt), debt=debt)

    async def slot() -> bool:
      """Acquires an in-flight execution slot or returns False if `ws` closes.

      Returns:
        True if a slot was acquired, or False if the WebSocket closed first.
      """
      nonlocal closing
      if not slots.locked():
        await slots.acquire()
        return True
      if closing is None:
        closing = asyncio.ensure_future(ws.wait_closed())
      acquire = asyncio.ensure_future(slots.acquire())
      try:
        await asyncio.wait(
            {acquire, closing}, return_when=asyncio.FIRST_COMPLETED
        )
      finally:
        acquire.cancel()  # a no-op once it has the slot
        await asyncio.gather(acquire, return_exceptions=True)
      if not closing.done():
        return True
      # the slot came as the socket closed: it goes back
      if not acquire.cancelled():
        slots.release()
      return False

    def malformed(problem: str, rid: str | None = None) -> bool:
      """Records a malformed frame and queues an error reply if it has `rid`.

      Args:
        problem: Description of why the frame was malformed.
        rid: Optional answerable request ID to send an error result for.

      Returns:
        True if the total malformed frame count exceeded `P.MAX_MALFORMED`.
      """
      o.malformed += 1
      o.note(f"malformed frame ({o.malformed}): {problem}")
      if rid:
        owed.append((rid, problem))
      if o.malformed > P.MAX_MALFORMED:
        o.status, o.reason = (
            "error",
            f"more than {P.MAX_MALFORMED} malformed frames",
        )
        return True
      return False

    async def frames() -> None:
      try:
        async for frame in ws:
          try:
            message = P.parse_frame(frame)
          except P.ProtocolError as exc:
            ended = malformed(str(exc), exc.rid)
            await pay(owed)
            if ended:
              break
            continue
          kind = message["type"]
          if kind in P.REQUESTS:
            # Frames are read as they arrive; only a request waits for a slot
            # before it is checked (its payload decoded) or started. A request
            # that arrives while MAX_IN_FLIGHT run is held (that one parsed
            # frame) and no further frame is read until a slot frees, so
            # websocket flow control holds back a harness that floods requests
            # instead of letting them pile up here. Until websockets has
            # RECEIVE_QUEUE frames queued it keeps reading the socket, so
            # keepalive pings are still answered and the socket closing is seen
            # at once.
            frame = None  # the parsed message is all that waits
            rid = P.answerable_id(message.get("id"))
            if rid:
              waiting.append(rid)
            if not await slot():
              # the socket closed meanwhile: the request never starts, and what
              # the harness sent before it closed (done among it) is still read
              continue
            if rid:
              waiting.pop()
            try:
              request = P.check_request(message)
              if kind == "http" and not P.mcp_allows(
                  request["url"], self.mcp_servers
              ):
                raise P.ProtocolError(
                    "http: only the task's MCP servers may be called",
                    P.answerable_id(request["id"]),
                )
            except P.ProtocolError as exc:
              slots.release()
              # one from inside mcp_allows has no id
              ended = malformed(str(exc), exc.rid or rid)
              await pay(owed)
              if ended:
                break
              continue
            o.requests += 1
            o.kinds[kind] = o.kinds.get(kind, 0) + 1
            task = asyncio.ensure_future(relay(request))
            pending.add(task)
            inflight[task] = request["id"]
            task.add_done_callback(pending.discard)
          elif kind in P.NOTES:
            try:
              self._notes(message)
            # a malformed usage note, answered when it carries an id
            except P.ProtocolError as exc:
              ended = malformed(str(exc), exc.rid)
              await pay(owed)
              if ended:
                break
              continue
          elif kind == "done":
            o.final_message = safe(message.get("final_message") or "")
            o.status, o.reason = "completed", "done"
            o.note(f"done after {o.requests} request(s)")
            break
          else:
            gave_up = safe(message.get("message") or "")[:2000]
            o.status, o.reason = "error", f"the harness gave up: {gave_up}"
            o.note(o.reason)
            break
        else:
          await self._closed_early(
              "the harness closed the connection before sending done"
          )
      except websockets.exceptions.ConnectionClosed as exc:
        codes = {
            frame.code for frame in (exc.rcvd, exc.sent) if frame is not None
        }
        if 1009 in codes:  # "message too big", whichever side noticed
          o.status, o.reason = (
              "error",
              f"a frame over {P.MAX_FRAME_BYTES} bytes",
          )
          o.note(o.reason)
        else:
          await self._closed_early(f"the connection closed before done ({exc})")

    reader = asyncio.ensure_future(frames())
    stopper = asyncio.ensure_future(failed.wait())
    try:
      await asyncio.wait({reader, stopper}, return_when=asyncio.FIRST_COMPLETED)
      if reader.done():
        reader.result()
      else:  # the evaluator failed: no use going on
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        o.status, o.reason = "error", f"the evaluator failed: {o.infra}"
        o.note(o.reason)
    except asyncio.CancelledError:
      cancelled = True
      raise
    finally:
      watchers = (
          (reader, stopper) if closing is None else (reader, stopper, closing)
      )
      for task in (*watchers, *pending):
        task.cancel()
      # All of this comes before the first await below, which a cancel can
      # interrupt (the time limit, or Harbor's agent timeout, arriving just as
      # the session ends). A request still running first stops its command in
      # the task container, which takes seconds; run() waits for that after the
      # cancel frame and the results, not before (self._stops). Every request
      # gets one result: a malformed one its problem (the reader may not have
      # sent it yet), and one still running or waiting for a slot a refusal.
      # After a cancel, _cancel sends them after its cancel frame (self._due,
      # the same list); otherwise they go now, within CANCEL_GRACE_SEC.
      self._stops = asyncio.gather(*pending, return_exceptions=True)
      ended = CANCELLED if cancelled else "cancelled: the task ended"
      due = self._due = list(owed) + [
          (rid, ended) for rid in [*inflight.values(), *waiting]
      ]
      owed.clear()
      inflight.clear()
      waiting.clear()

      async def settle() -> None:
        while due:
          # off the list before the send, as reply() does: an interrupted send
          # is not answered twice
          await reply(P.error_result(*due.pop(0)))

      try:
        # the reader and the stopper end at once
        await asyncio.gather(*watchers, return_exceptions=True)
        if not cancelled:
          try:
            await asyncio.wait_for(settle(), P.CANCEL_GRACE_SEC)
          except Exception:  # pylint: disable=broad-exception-caught
            # a harness that stopped reading: best effort while ending
            pass
      # a cancel struck meanwhile: _cancel answers the rest
      except asyncio.CancelledError:
        self._due = [
            (rid, CANCELLED)
            if error == "cancelled: the task ended"
            else (rid, error)
            for rid, error in due
        ]
        raise

  def _timeout(self, requested: int) -> int:
    left = self._left()
    # ends 2 s early
    cap = (
        P.EXEC_TIMEOUT_SEC
        if left is None
        else min(P.EXEC_TIMEOUT_SEC, int(left) - 2)
    )
    return max(1, min(requested, cap))

  async def _exec(
      self, command: str, timeout_sec: int, user: str | None = None
  ) -> tuple[int, str]:
    """Runs `command` in the task container via `self.executor.exec`.

    Args:
      command: Shell command to execute in `self.workdir`.
      timeout_sec: Inner command timeout in seconds (before grace and margin).
      user: Optional container user override.

    Returns:
      An `(exit_code, output)` tuple from the executor.

    Raises:
      RequestError: If the command cannot be encoded, exceeds argument limits,
        or fails while the container remains healthy.
      ExecutorError: If the container or Docker infrastructure failed.
    """
    try:
      return await self.executor.exec(
          command,
          cwd=self.workdir,
          user=user,
          timeout_sec=timeout_sec + KILL_GRACE_SEC + EXEC_MARGIN_SEC,
      )
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
      raise
    except (UnicodeError, ValueError) as exc:
      raise RequestError(f"the request cannot be run: {exc}") from exc
    except OSError as exc:
      if exc.errno == errno.E2BIG:
        raise RequestError(
            "the request is too long to run as one command"
        ) from exc
      raise await self._failure("exec", exc) from exc
    except Exception as exc:
      raise await self._failure("exec", exc) from exc

  async def _failure(self, what: str, exc: Exception) -> Exception:
    """Classifies a container error as `RequestError` or `ExecutorError`.

    A harness can cause container failures with its own commands: filling the
    disk or making `/tmp` unwritable (where the container still runs a trivial
    command), or wedging the container with a fork bomb or memory hog (where
    Docker still answers and the container exists, but no longer runs `true`).
    In both cases the request fails with `RequestError` and the task continues;
    only Docker failing or the container disappearing is an `ExecutorError`.

    Args:
      what: Short description of the operation that failed.
      exc: The underlying exception raised by the executor.

    Returns:
      A `RequestError` if the container is reachable or wedged, or an
      `ExecutorError` if the infrastructure failed.
    """
    # the executor probed already
    reachable = exc.reachable if isinstance(exc, ExecutorError) else None
    if reachable is None:
      reachable = await self._reachable()
    if reachable:
      return RequestError(
          f"the task container could not run this request ({what}: {exc}); the"
          " task goes on"
      )
    if await self._healthy():
      self.outcome.note(
          f"{what}: the task container no longer answers while Docker does "
          f"({type(exc).__name__}: {exc})"
      )
      return RequestError(WEDGED)
    return ExecutorError(
        f"{what} in the task container failed: {type(exc).__name__}: {exc}"
    )

  async def _healthy(self) -> bool:
    """Checks whether Docker, the task container, and egress are healthy.

    An executor without a `healthy` check is treated as unhealthy so the failure
    is attributed to the evaluator.

    Returns:
      True if the task container and serve stack egress proxy are healthy.
    """
    healthy = getattr(self.executor, "healthy", None)
    if healthy is None:
      return False
    try:
      if not await healthy():
        return False
      return not (await self._egress(""))[0]
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
      raise
    except Exception:  # pylint: disable=broad-exception-caught
      # cannot tell
      return False

  async def _reachable(self) -> bool:
    """Returns whether the task container can execute a trivial `true` command.

    Uses `self.executor.answers()` when available, falling back to running
    `true` via `self.executor.exec()`.

    Returns:
      True if the probe succeeded with exit code 0, or False otherwise.
    """
    answers = getattr(self.executor, "answers", None)
    try:
      if answers is not None:
        return await answers()
      code, _ = await self.executor.exec(
          "true", cwd="/", timeout_sec=PROBE_TIMEOUT_SEC
      )
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
      raise
    except Exception:  # pylint: disable=broad-exception-caught
      # an exception means it does not answer
      return False
    return code == 0

  async def _remove(self, upload: str) -> None:
    """Best effort: a staged file left in /tmp does not fail the request."""
    try:
      await self.executor.exec(
          f"rm -f {upload}", cwd="/", timeout_sec=30, user="root"
      )
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.outcome.note(f"could not remove {upload}: {exc}")

  def _path(self, path: str) -> str:
    return (
        path if path.startswith("/") else f"{self.workdir.rstrip('/')}/{path}"
    )

  async def _bounded(self, script: str, timeout: int) -> tuple[int, str]:
    """A short bash script under `timeout`, as every request is."""
    return await self._exec(
        f"timeout -k {KILL_GRACE_SEC} {timeout} bash -c {shlex.quote(script)}",
        timeout,
    )

  async def dispatch(self, r: dict[str, Any]) -> dict[str, Any]:
    """Dispatches a validated request dict `r` to its handler."""
    kind = r["type"]
    if kind == "exec":
      return await self._run(r)
    if kind == "read_file":
      return await self._read(r)
    if kind == "write_file":
      return await self._write(r)
    return await self._http(r)

  async def _run(self, r: dict[str, Any]) -> dict[str, Any]:
    """Executes an `exec` request in the task container and formats its result.

    Args:
      r: Validated `exec` request dict.

    Returns:
      The protocol result frame dict for the command execution.
    """
    timeout = self._timeout(r["timeout_sec"])
    token = uuid.uuid4().hex
    script, upload = wrap(r["command"], timeout, token=token), None
    if len(script.encode()) > MAX_ARG_BYTES:
      upload = await self._upload(r["command"].encode())
      script = wrap(upload, timeout, file=True, token=token)
    started = time.monotonic()
    try:
      code, raw = await self._exec(script, timeout)
      # before the cleanup, which can be slow
      elapsed = time.monotonic() - started
    finally:
      if upload is not None:
        await self._remove(upload)
    stopped = timed_out(code, elapsed, timeout)
    raw, cut = cut_trailer(raw, token)
    data = raw.encode("utf-8", errors="replace")
    truncated = bool(cut) or len(data) > P.MAX_OUTPUT_BYTES
    output = fit_output(data, cut=bool(cut)).decode("utf-8", errors="replace")
    return result_of(
        r["id"],
        {
            "exit_code": TIMEOUT_EXIT_CODE if stopped else code,
            "output": output,
            "truncated": truncated,
            "timed_out": stopped,
        },
        ok=None,
    )

  async def _read(self, r: dict[str, Any]) -> dict[str, Any]:
    """Executes a `read_file` request in the task container.

    Args:
      r: Validated `read_file` request dict.

    Returns:
      The protocol result frame dict containing base64 file content or an error.
    """
    path = self._path(r["path"])
    timeout, started = (
        self._timeout(P.DEFAULT_EXEC_TIMEOUT_SEC),
        time.monotonic(),
    )
    code, out = await self._bounded(read_script(path, SIZE_MARK), timeout)
    # what the shell printed before the marker (a locale warning on stderr) is
    # not part of the answer
    _, marked, rest = out.rpartition(SIZE_MARK)
    size_line, _, payload = rest.partition("\n")
    payload = payload.strip()
    try:
      size = int(size_line.strip()) if code == 0 and marked else -1
    except ValueError:
      size = -1
    # checked, not decoded: the content goes to the harness as it came (but for
    # the cut, read_result)
    if size < 0 or len(payload) % 4 or not _BASE64.fullmatch(payload):
      why = stopped_by(code, time.monotonic() - started, timeout) or read_error(
          out, SIZE_MARK, code
      )
      return P.error_result(r["id"], f"could not read {path}: {why}")
    content, size, truncated = read_result(payload, size)
    return result_of(
        r["id"], {"content_b64": content, "size": size, "truncated": truncated}
    )

  async def _upload(self, data: bytes) -> str:
    """Uploads `data` to a new temporary file in the task container's `/tmp`.

    Writes the local staging copy in a worker thread so large payloads do not
    block the shared event loop under Harbor.

    Args:
      data: Raw bytes to copy into the container.

    Returns:
      The remote path in `/tmp` inside the task container.
    """
    remote = f"/tmp/.serve-upload-{uuid.uuid4().hex}"
    writing = asyncio.ensure_future(asyncio.to_thread(_local_copy, data))
    try:
      local = await asyncio.shield(writing)
    except asyncio.CancelledError:
      # the thread finishes the file; remove it then
      writing.add_done_callback(_discard_local)
      raise
    try:
      try:
        await asyncio.wait_for(
            self.executor.upload(local, remote), UPLOAD_TIMEOUT_SEC
        )
      except asyncio.CancelledError:  # pylint: disable=try-except-raise
        raise
      except asyncio.TimeoutError:
        stuck = TimeoutError(
            f"the copy did not finish within {UPLOAD_TIMEOUT_SEC} s"
        )
        raise await self._failure(
            "staging the request in /tmp", stuck
        ) from None
      except Exception as exc:
        raise await self._failure("staging the request in /tmp", exc) from exc
    finally:
      os.unlink(local)
    return remote

  async def _write(self, r: dict[str, Any]) -> dict[str, Any]:
    """Executes a `write_file` request in the task container.

    Args:
      r: Validated `write_file` request dict.

    Returns:
      The protocol result frame dict with the written file size or an error.
    """
    path, content = self._path(r["path"]), r["content"]
    q, redirect = shlex.quote(path), ">>" if r["append"] else ">"
    payload = base64.b64encode(content).decode()
    upload = None
    if len(payload) <= MAX_ARG_BYTES:
      script = (
          f'set -o pipefail; mkdir -p "$(dirname {q})" && printf %s'
          f" {shlex.quote(payload)} | base64 -d {redirect} {q} && stat -L -c %s"
          f" -- {q}"
      )
    else:
      upload = await self._upload(content)
      script = (
          f'mkdir -p "$(dirname {q})" && cat {upload} {redirect} {q} && stat -L'
          f" -c %s -- {q}"
      )
    timeout, started = (
        self._timeout(P.DEFAULT_EXEC_TIMEOUT_SEC),
        time.monotonic(),
    )
    try:
      code, out = await self._bounded(script, timeout)
      elapsed = time.monotonic() - started
    finally:
      if upload is not None:
        await self._remove(upload)
    if code != 0:
      why = (
          stopped_by(code, elapsed, timeout)
          or out.strip()[:2000]
          or f"exit {code}"
      )
      return P.error_result(r["id"], f"could not write {path}: {why}")
    try:
      size = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
      size = -1
    return result_of(r["id"], {"size": size})

  async def _http(self, r: dict[str, Any]) -> dict[str, Any]:
    """Executes an `http` request from inside the task container.

    Args:
      r: Validated `http` request dict.

    Returns:
      The protocol result frame dict with the HTTP response or an error.
    """
    timeout = self._timeout(r["timeout_sec"])
    spec = base64.b64encode(
        json.dumps({
            "url": r["url"],
            "method": r["method"],
            "headers": r["headers"],
            "body": base64.b64encode(r["body"]).decode(),
            "timeout": timeout,
        }).encode()
    ).decode()
    upload = (
        await self._upload(spec.encode()) if len(spec) > MAX_ARG_BYTES else None
    )
    command = (
        f"timeout -k {KILL_GRACE_SEC} {timeout} {PYTHON} -c"
        f" {shlex.quote(_HTTP)} {shlex.quote('@' + upload if upload else spec)}"
        f" {P.MAX_HTTP_BYTES}"
    )
    started = time.monotonic()
    try:
      code, out = await self._exec(command, timeout)
      elapsed = time.monotonic() - started
    finally:
      if upload is not None:
        await self._remove(upload)
    why = stopped_by(code, elapsed, timeout)
    if why:
      return P.error_result(r["id"], f"http: {why}")
    try:
      reply = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
      return P.error_result(r["id"], f"http: no reply ({out[:2000]})")
    if isinstance(reply, dict) and "error" in reply:
      return P.error_result(r["id"], f"http: {reply['error']}")
    answer = http_answer(reply)
    if answer is None:
      return P.error_result(
          r["id"], "http: the task container's answer was not an HTTP response"
      )
    return result_of(r["id"], answer)

  def _record(
      self, request: dict[str, Any], result: dict[str, Any], seconds: float
  ) -> None:
    """Records `request` and `result` into the run's trajectory and log.

    Commands, paths, and URLs are truncated to `ARGUMENT_CHARS` before
    recording.

    Args:
      request: Dispatched request dict.
      result: Result frame dict produced for `request`.
      seconds: Elapsed wall-clock time in seconds for the request.
    """
    kind = request["type"]
    if kind == "exec":
      args = {
          "command": _cut(request["command"]),
          "timeout_sec": request["timeout_sec"],
      }
    elif kind == "write_file":
      args = {
          "path": _cut(request["path"]),
          "bytes": len(request["content"]),
          "append": request["append"],
      }
    elif kind == "http":
      args = {
          "method": request["method"],
          "url": _cut(request["url"]),
          "bytes": len(request["body"]),
      }
    else:
      args = {"path": _cut(request["path"])}
    observation = observe(result)
    self.outcome.add_step({
        "source": "agent",
        "tool": kind,
        "id": request["id"],
        "arguments": args,
        "observation": observation[:OBSERVATION_CHARS],
    })
    self.outcome.note(
        f"{kind} {json.dumps(args)[:500]} -> {observation[:300]!r}"
        f" ({seconds:.1f}s)"
    )


def result_of(
    rid: str, payload: dict[str, Any], ok: bool | None = True
) -> dict[str, Any]:
  """Builds a protocol `result` frame dict from `rid` and `payload`.

  Places `payload` first so container-supplied keys cannot overwrite `type`,
  `id`, or `ok`. When `ok` is `None` (as for `exec` results), omits `ok`.

  Args:
    rid: Request ID being answered.
    payload: Result fields specific to the request type.
    ok: Success flag, or `None` to omit the `ok` key.

  Returns:
    The assembled result frame dictionary.
  """
  out = {**payload, "type": "result", "id": rid, "ok": ok}
  if ok is None:
    del out["ok"]
  return out


def http_answer(reply) -> dict[str, Any] | None:
  """Validates and extracts HTTP result fields from the container's JSON output.

  Args:
    reply: Decoded JSON value printed by the in-container `_HTTP` helper.

  Returns:
    A dict with `status`, `headers`, `body_b64`, and `truncated`, or `None` if
    `reply` does not match the expected structure.
  """
  if not isinstance(reply, dict):
    return None
  status, headers, body, truncated = (
      reply.get(k) for k in ("status", "headers", "body_b64", "truncated")
  )
  if (
      isinstance(status, bool)
      or not isinstance(status, int)
      or not isinstance(headers, dict)
      or not all(
          isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
      )
      or not isinstance(body, str)
      or len(body) % 4
      or not _BASE64.fullmatch(body)
      or not isinstance(truncated, bool)
  ):
    return None
  return {
      "status": status,
      "headers": headers,
      "body_b64": body,
      "truncated": truncated,
  }


def _cut(text: str, max_chars: int = ARGUMENT_CHARS) -> str:
  return (
      text
      if len(text) <= max_chars
      else f"{text[:max_chars]}... [{len(text) - max_chars} more characters]"
  )


def _head(content_b64: str, max_chars: int = OBSERVATION_CHARS) -> str:
  """Decodes the prefix of `content_b64` into at most `max_chars` of text.

  Decodes only enough base64 groups to cover `max_chars` characters rather than
  decoding an entire multi-megabyte read.

  Args:
    content_b64: Base64-encoded content string.
    max_chars: Maximum number of decoded characters to return.

  Returns:
    The decoded UTF-8 prefix, or `"[not base64]"` if decoding fails.
  """
  try:
    return base64.b64decode(content_b64[: -(-4 * max_chars // 3) * 4]).decode(
        "utf-8", errors="replace"
    )[:max_chars]
  except ValueError:  # binascii.Error: not base64
    return "[not base64]"


def observe(result: dict[str, Any]) -> str:
  """Formats a request `result` dict as a summary string for the run trajectory.

  Args:
    result: Protocol result dictionary.

  Returns:
    A human-readable observation string (truncated by the caller to
    `OBSERVATION_CHARS`).
  """
  if result.get("ok") is not None and not result.get("ok"):
    return "error: " + str(result.get("error"))
  if "output" in result:
    timed = " (timed out)" if result.get("timed_out") else ""
    return f"exit_code={result['exit_code']}{timed}\n{result['output']}"
  if "content_b64" in result:
    cut = " (truncated)" if result.get("truncated") else ""
    return f"{result['size']} bytes{cut}\n{_head(result['content_b64'])}"
  if "status" in result:
    return f"HTTP {result['status']}\n" + _head(result.get("body_b64") or "")
  return json.dumps(
      {k: v for k, v in result.items() if k not in ("type", "id")}
  )


class DockerExecutor:
  """A plain container as the task container (kit check)."""

  def __init__(self, container: str, user: str | None = None):
    self.container, self.user = container, user

  async def exec(
      self, command: str, *, cwd: str, timeout_sec: int, user: str | None = None
  ) -> tuple[int, str]:
    """Runs `command` inside `self.container` via `docker exec`.

    Args:
      command: Bash command string to execute.
      cwd: Working directory inside the container.
      timeout_sec: Maximum seconds to wait for `docker exec` to complete.
      user: Optional user to run as, overriding `self.user`.

    Returns:
      An `(exit_code, combined_output)` tuple.

    Raises:
      OutputOverrun: If the command output exceeds `P.MAX_EXEC_READ_BYTES`.
      RuntimeError: If `docker exec` does not return within `timeout_sec`.
      ExecutorError: If `docker exec` fails due to a Docker daemon or container
        error.
    """
    who = ["-u", user or self.user] if (user or self.user) else []
    # in the environment of all the command starts, so _stop finds it
    tag = uuid.uuid4().hex
    argv = [
        "docker",
        "exec",
        "-w",
        cwd,
        "-e",
        f"{EXEC_TAG}={tag}",
        *who,
        self.container,
        "bash",
        "-c",
        command,
    ]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
      out, err = await asyncio.wait_for(_read_both(proc), timeout_sec)
    except OutputOverrun:
      await self._stop(proc, tag, who)
      raise
    except asyncio.TimeoutError:
      # The relay's commands run under their own `timeout`, which ends one that
      # runs too long well before this, so docker exec not back yet is a stall.
      # The relay handles it as it does Harbor's own timeout (_failure):
      # the request fails while the container still answers, and the task goes
      # on.
      await self._stop(proc, tag, who)
      raise RuntimeError(
          f"docker exec did not return within {timeout_sec} s"
      ) from None
    # a request the time limit cut short: stopped in the container too
    except asyncio.CancelledError:
      await self._stop(proc, tag, who)
      raise
    errors = err.decode("utf-8", errors="replace")
    # Docker's own errors come on docker exec's stderr. An exec request's
    # command writes nothing there (the cutter folds its stderr into stdout), so
    # what it printed, whatever it says, stays its result.
    failure = docker_failure(proc.returncode, errors)
    if failure:  # the relay decides whose failure it is
      raise ExecutorError(f"docker exec: {failure}")
    # stderr first: what a script's shell warns about at start-up stays ahead of
    # the markers its output ends with
    return proc.returncode, errors + out.decode("utf-8", errors="replace")

  async def _stop(self, proc, tag: str, who: list[str]) -> None:
    """Kills the local `docker exec` client and stops tagged processes.

    Killing the `docker exec` client alone leaves the in-container process
    running, so this also runs `stop_command(tag)` inside the container on a
    best-effort, time-bounded basis.

    Args:
      proc: Local `docker exec` subprocess to kill and reap.
      tag: `EXEC_TAG` value identifying processes started by this exec.
      who: Optional `["-u", user]` arguments for `docker exec`.
    """
    with contextlib.suppress(ProcessLookupError):
      proc.kill()
    with contextlib.suppress(asyncio.TimeoutError):
      await asyncio.wait_for(proc.wait(), 10)
    try:
      stopper = await asyncio.create_subprocess_exec(
          "docker",
          "exec",
          *who,
          self.container,
          "bash",
          "-c",
          stop_command(tag),
          stdout=asyncio.subprocess.DEVNULL,
          stderr=asyncio.subprocess.DEVNULL,
      )
    except OSError:
      return
    try:
      await asyncio.wait_for(stopper.wait(), 15)
    except asyncio.TimeoutError:
      stopper.kill()
      await stopper.wait()

  async def upload(self, local: str, remote: str) -> None:
    """Copies `local` into `self.container` at `remote` using `docker cp`.

    Args:
      local: Path to the source file on the host.
      remote: Destination path inside the container.

    Raises:
      RuntimeError: If `docker cp` exits with a non-zero status.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "cp",
        local,
        f"{self.container}:{remote}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
      _, err = await proc.communicate()
    # cut short (the relay's UPLOAD_TIMEOUT_SEC, the time limit)
    except asyncio.CancelledError:
      with contextlib.suppress(ProcessLookupError):
        proc.kill()
      with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), 10)
      raise
    if proc.returncode != 0:
      raise RuntimeError(err.decode("utf-8", errors="replace").strip())

  async def healthy(self) -> bool:
    """Returns whether the task container is running or was OOM-killed."""
    return await asyncio.to_thread(stack_mod.container_alive, self.container)
