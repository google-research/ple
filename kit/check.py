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

"""Runs a submission through the evaluation server checks stage by stage.

Validates `submission.yaml`, `.env`, staged files, build, model connectivity,
websocket listener, smoke/MCP/cancel tasks, and dev-set artifacts, reporting
any issues to fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import io
import ipaddress
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import urllib.parse
import uuid

from dotenv.main import dotenv_values
from dotenv.parser import parse_stream
from kit import artifacts
from kit import build
from kit import models
from kit import protocol
from kit import relay
from kit import spec
from kit import stack

TASK_IMAGE = "python:3.12-slim"
# The local task container gets the smallest dev task's CPUs and memory
# (task.toml: 2 CPUs, 4 GB), the server's file-size limit, a process limit and a
# 1 GB /app in memory, so that a runaway smoke task cannot take this machine
# down. eval/run.py gives the dev tasks' containers the same process limit.
TASK_PIDS = 1024
TASK_LIMITS = (
    "--cpus",
    "2",
    "--memory",
    "4g",
    "--ulimit",
    f"fsize={build.MAX_FILE_BYTES}",
    "--pids-limit",
    str(TASK_PIDS),
    "--tmpfs",
    "/app:rw,exec,size=1g",
)
SMOKE = (
    "Create a file named `hello.txt` in the current working directory (`/app`)"
    " whose entire content is the single line:\n\nHELLO\n\nDo not create any"
    " other files. When the file exists, reply with the word done and stop."
)
LONG = (
    "Write a thorough design memo, at least 3000 words, on sizing the cooling"
    " plant of a 10 MW data hall, into `/app/memo.md`. Work in several steps"
    " and check your numbers with shell commands as you go."
)
# The MCP stage's task needs the answer of a tool on the task's MCP server,
# which a held-out task would give its harness. MCP_STUB stands in for that
# server: streamable HTTP with one tool, answering in server-sent events as
# FastMCP does (a JSON-RPC batch gets a batch of answers), run in the task
# container at the address held-out tasks use. It picks its word itself and
# prints it once it listens, so the word is in no file, argument or variable a
# command in the container could read.
MCP_PORT = 8000
MCP_SERVERS = [{
    "name": "apps",
    "url": f"http://127.0.0.1:{MCP_PORT}/mcp/",
    "transport": "streamable-http",
}]
MCP_TASK = (
    "The task's MCP server `apps` has a tool named `secret_word`. Call it, then"
    " create a file named `mcp.txt` in the current working directory (`/app`)"
    " whose entire content is the single word the tool returns. Do not create"
    " any other files. When the file exists, reply with the word done and stop."
)
MCP_STUB = r"""
import json, secrets, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
WORD = "MCP-" + secrets.token_hex(4).upper()
TOOL = {"name": "secret_word", "description": "Returns the secret word.",
        "inputSchema": {"type": "object", "properties": {}}}
def answer(message):
    params = message.get("params") or {}
    results = {"initialize": {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                              "capabilities": {"tools": {}}, "serverInfo": {"name": "apps", "version": "1"}},
               "ping": {}, "tools/list": {"tools": [TOOL]}}
    if params.get("name") == TOOL["name"]:
        results["tools/call"] = {"content": [{"type": "text", "text": WORD}], "isError": False}
    result = results.get(message.get("method"))
    return {"jsonrpc": "2.0", "id": message["id"],
            **({"result": result} if result is not None else {"error": {"code": -32601, "message": "unknown"}})}
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        batch = body if isinstance(body, list) else [body]
        answers = [answer(m) for m in batch if isinstance(m, dict) and "id" in m]
        if not answers:
            return self.reply(202)
        data = answers if isinstance(body, list) else answers[0]
        self.reply(200, b"event: message\ndata: " + json.dumps(data).encode() + b"\n\n")
    def do_GET(self):
        self.reply(405)
    def do_DELETE(self):
        self.reply(200)
    def reply(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", "text/event-stream" if body else "application/json")
        self.send_header("Mcp-Session-Id", "plebench-check")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
print(WORD, flush=True)
server.serve_forever()
"""
SMOKE_TIMEOUT_SEC = 600
# 3 minutes of work before the samples' 4-minute wrap-up (WRAP_UP_SEC)
MCP_TIMEOUT_SEC = 420
PROBE_CALL_SEC = 150  # one model call of the probe, which makes two at most
# the docker exec that runs the probe: both calls fit in it
PROBE_EXEC_SEC = 2 * PROBE_CALL_SEC + 30
# A provider key's prefix and body (OpenAI sk-, sk-proj-, Anthropic sk-ant-,
# Google AIza, GitHub, Hugging Face, Slack), or a PEM private key. A body counts
# only when it holds an upper-case letter, a lower-case letter and a digit, as
# generated keys do and words do not (a URL path that starts with the sk- prefix
# and goes on in words is not a key). Every match is shorter than KEY_OVERLAP,
# so key_lines finds the same keys wherever its chunks end.
KEY_RE = re.compile(
    rb"(?<![A-Za-z0-9])(?:(?P<prefix>sk-(?:ant-|proj-)?|AIza|gh[pousr]_|hf_|xox[baprs]-)"
    rb"(?P<body>[A-Za-z0-9_-]{20,4000})|-----BEGIN [A-Z ]{0,64}PRIVATE"
    rb" KEY-----)"
)


# The key scan reads a staged file to its end whatever its size, in chunks,
# within one check's time (staging already caps the repository at
# build.MAX_SOURCE_BYTES). A file it could not read to the end is reported as
# not scanned and read again next time, never recorded clean.
KEY_SCAN_CHUNK = 1 << 20
# bytes held back at a chunk's end: longer than any KEY_RE match
KEY_OVERLAP = 4096
KEY_CONTEXT = 16  # bytes before the next search start, for KEY_RE's look-behind
KEY_SCAN_SEC = 120
# The version of the code that applies KEY_RE (_generated, key_lines): raise it
# whenever that code changes what is found, so that a saved scan (key_rules) is
# not reused.
KEY_SCAN_VERSION = 1
# Folder names that usually hold a dependency's code or its test data. Vendored
# test fixtures often hold sample keys, so what looks like a key under one of
# these is a warning instead of a failure.
VENDORED = frozenset({
    "test",
    "tests",
    "fixtures",
    "testdata",
    "vendor",
    "node_modules",
    "site-packages",
})


def key_rules() -> str:
  """Returns the SHA-256 fingerprint of the active key-scan detection rules.

  Combines `KEY_SCAN_VERSION` and `KEY_RE` so any change to either causes cached
  scan results to be invalidated and every file to be scanned again.

  Returns:
    The SHA-256 hex digest of the key detection rules.
  """
  return hashlib.sha256(
      b"%d %d " % (KEY_SCAN_VERSION, KEY_RE.flags) + KEY_RE.pattern
  ).hexdigest()


def vendored(rel: str) -> bool:
  """Returns True if `rel` lies under a directory name in `VENDORED`."""
  return any(part.lower() in VENDORED for part in rel.split("/")[:-1])


def _generated(m: re.Match[bytes]) -> bool:
  """Returns True if a `KEY_RE` match looks like a generated secret or PEM key.

  Matches either a PEM private key header or a key body containing uppercase
  letters, lowercase letters, and digits (unlike plain English words in URLs).

  Args:
    m: Regex match object from `KEY_RE`.

  Returns:
    True if the match represents a PEM header or generated key token.
  """
  body = m["body"]
  return body is None or bool(
      re.search(rb"[A-Z]", body)
      and re.search(rb"[a-z]", body)
      and re.search(rb"[0-9]", body)
  )


def key_lines(path: pathlib.Path, deadline: float) -> list[int] | None:
  """Returns 1-based line numbers in `path` containing suspected keys, or None.

  Streams `path` in `KEY_SCAN_CHUNK` chunks with `KEY_OVERLAP` overlap so files
  of any size or line length require bounded memory and each match is counted
  once. Returns None if `deadline` (`time.monotonic()`) expires before EOF.

  Args:
    path: File path to scan.
    deadline: Monotonic timestamp (`time.monotonic()`) at which to abort.

  Returns:
    Sorted 1-based line numbers with key matches, or None if the deadline
    passed before scanning finished.
  """
  lines: set[int] = set()
  # carry: `ctx` bytes of look-behind context, then bytes not yet searched;
  # newlines: how many come before carry[ctx:]
  carry, ctx, newlines = b"", 0, 0
  with path.open("rb") as fh:
    while True:
      if time.monotonic() > deadline:
        return None
      chunk = fh.read(KEY_SCAN_CHUNK)
      window = carry + chunk
      last = len(window) if not chunk else max(ctx, len(window) - KEY_OVERLAP)
      for m in KEY_RE.finditer(window, ctx):
        if m.start() >= last:
          break
        if _generated(m):
          lines.add(newlines + window.count(b"\n", ctx, m.start()) + 1)
      if not chunk:
        return sorted(lines)
      newlines += window.count(b"\n", ctx, last)
      ctx = min(KEY_CONTEXT, last)
      carry = window[last - ctx :]


PROBE = f"CALL_SEC = {PROBE_CALL_SEC}\n" + r"""
import json, os, time, urllib.error, urllib.request
class Refused(Exception):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code, self.body = code, body
def ask(limit):
    body = {"model": os.environ["MODEL_NAME"], "messages": [{"role": "user", "content": "Reply with the word OK."}],
            limit: 512}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("MODEL_API_KEY", "")}
    req = urllib.request.Request(os.environ["MODEL_BASE_URL"].rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=CALL_SEC) as r:
            return json.loads(r.read())["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        raise Refused(e.code, e.read().decode(errors="replace")) from None    # the body can be read only once
t = time.time()
try:
    try:
        msg = ask("max_completion_tokens")
    except Refused as e:
        if e.code != 400 or "max_completion_tokens" not in e.body:
            raise
        msg = ask("max_tokens")
    print(json.dumps({"ok": True, "seconds": round(time.time() - t, 1), "reply": (msg.get("content") or "")[:60]}))
except Refused as e:
    print(json.dumps({"ok": False, "error": str(e)}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
"""


@dataclasses.dataclass
class Stage:
  """Result of a single `kit check` stage."""

  name: str
  status: str  # pass | warn | fail | skip
  detail: str = ""


@dataclasses.dataclass
class Report:
  """Aggregated results across all stages of a `kit check` run."""

  submission: str
  stages: list[Stage] = dataclasses.field(default_factory=list)
  # stages asked for on the command line (cancel)
  requested: list[str] = dataclasses.field(default_factory=list)
  # why the entry is not ready to submit though every stage passes (an UNREADY
  # key), or ""
  unready: str = ""

  @property
  def ok(self) -> bool:
    """Returns True if every recorded stage passed, warned, or was skipped."""
    return all(s.status in ("pass", "warn", "skip") for s in self.stages)

  @property
  def smoke_skipped(self) -> bool:
    """Returns True if the smoke stage was skipped (`--no-smoke`).

    When the smoke task is skipped, the stages that ran can pass, but the entry
    is not marked ready for submission.
    """
    return any(s.name == "smoke" and s.status == "skip" for s in self.stages)

  @property
  def untested(self) -> list[str]:
    """Returns requested stages that did not run or were skipped."""
    last = {s.name: s for s in self.stages}
    return [
        f"{n} ({last[n].detail or 'skipped'})"
        if n in last
        else f"{n} (did not run)"
        for n in self.requested
        if n not in last or last[n].status == "skip"
    ]

  def add(self, name: str, status: str, detail: str = "") -> Stage:
    """Appends a stage result, prints its status line, and returns the `Stage`.

    Args:
      name: Stage name.
      status: Stage status (`"pass"`, `"warn"`, `"fail"`, or `"skip"`).
      detail: Optional detail message for the stage.

    Returns:
      The recorded `Stage` instance.
    """
    stage = Stage(name, status, detail)
    self.stages.append(stage)
    mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "skip"}[
        status
    ]
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return stage


# What check() says after "the stages that ran pass, but " for each
# Report.unready reason
UNREADY = {
    "clone": (
        "a folder of the competition repository is not submitted: make it a"
        " private repository of its own as README.md's Quick start does (copy"
        f" it out, `git init`, its {spec.ENV_FILE} there) and check that copy"
    ),
    "no-git": (
        "this folder is not a git repository, and the server builds a commit:"
        " make it the root of a private repository of its own (`git init`,"
        f" commit everything, its {spec.ENV_FILE} included, push it to GitHub)"
        " and check it again"
    ),
    "subfolder": (
        f"{spec.FILE} must be at the root of the repository you submit, and"
        " this folder is below its repository's root: make it a private"
        " repository of its own (copy it out, `git init`, commit it with its"
        f" {spec.ENV_FILE}, push it to GitHub) and check that copy"
    ),
    "uncommitted": (
        "the server builds the commit you submit, and the files stage lists"
        " what this one lacks: commit it (`git add -A && git commit`, and `git"
        f" add -f {spec.ENV_FILE}` for a {spec.ENV_FILE} git ignores), push,"
        " and submit that commit"
    ),
    "artifacts": (
        "a submission also holds the harness's dev-set artifacts: run `uv run"
        f" python eval/run.py <entry>`, which writes {artifacts.ROOT}/ into the"
        " entry, then commit it with the harness (docs/artifacts.md)"
    ),
    "model-override": (
        "the model stage called --model-base-url, so model_base_url itself,"
        " which the server calls, was not tested: once that public endpoint"
        " answers, run kit check again without the flag"
    ),
}


class SetupError(Exception):
  """Configuration error preventing the serve container from starting.

  `stage` identifies the check stage that failed: `.env` (for the submission's
  `.env` file) or `model` (for `MODEL_PROXY_URL`, `MODEL_PROXY_KEY`, or
  `--model-base-url`). Error messages never include secret values.
  """

  def __init__(self, stage: str, message: str):
    super().__init__(message)
    self.stage = stage


def read_env_file(path: str | None) -> dict[str, str]:
  """Parses `--env-file` settings for a local run using `python-dotenv`.

  Reads local run variables (`MODEL_PROXY_URL`, `MODEL_PROXY_KEY`, `JUDGE_*`),
  handling quotes, escapes, comments, `export`, and `${NAME}` interpolation.
  Ignores a leading UTF-8 BOM and omits keys without values.

  Args:
    path: Path to the env file, or None.

  Returns:
    A dictionary of parsed environment variable names and values.

  Raises:
    SystemExit: If the file cannot be read, is not valid UTF-8, or contains a
      line that `python-dotenv` cannot parse.
  """
  if not path:
    return {}
  try:
    text = pathlib.Path(path).read_text(encoding="utf-8-sig")
  except OSError as exc:
    raise SystemExit(f"--env-file {path}: {exc.strerror or exc}") from None
  except UnicodeDecodeError:
    raise SystemExit(f"--env-file {path} is not UTF-8 text") from None
  for binding in parse_stream(io.StringIO(text)):
    if binding.error:
      raise SystemExit(
          f"{path}:{binding.original.line}: not a NAME=value line (quote a"
          " value that holds a space followed by #)"
      )
  return {
      k: v
      for k, v in dotenv_values(stream=io.StringIO(text)).items()
      if v is not None
  }


def entry_env(src: pathlib.Path) -> dict[str, str]:
  """Reads and validates the submission's `.env` file in `src`.

  Args:
    src: Submission directory path.

  Returns:
    A dictionary of variables from `src/.env`, or `{}` if `.env` does not exist.

  Raises:
    SetupError: If `.env` is not a regular file, is not UTF-8 text, or fails
      `spec.read_env` validation.
  """
  path = pathlib.Path(src) / spec.ENV_FILE
  if not os.path.lexists(path):
    return {}
  if path.is_symlink() or not path.is_file():
    raise SetupError(
        spec.ENV_FILE,
        f"{spec.ENV_FILE} must be a regular file, not a link or a directory",
    )
  try:
    return spec.read_env(path.read_bytes().decode("utf-8"))
  except UnicodeDecodeError:
    raise SetupError(
        spec.ENV_FILE, f"{spec.ENV_FILE} is not UTF-8 text"
    ) from None
  except spec.SpecError as exc:
    raise SetupError(spec.ENV_FILE, "; ".join(exc.problems)) from None


# where the key goes for each setting that names a model server, instead of in
# its URL (url_problem)
URL_KEYS = {
    "MODEL_PROXY_URL": "MODEL_PROXY_KEY",
    "JUDGE_BASE_URL": "JUDGE_API_KEY",
    "--model-base-url": f"MODEL_API_KEY in the submission's {spec.ENV_FILE}",
}


# For each kind of address url_problem checks, what a message calls it and an
# example: a model server's (or a judge's), and a harness server's (eval/run.py
# --url)
URL_KINDS = {
    ("http", "https"): ("an http(s) URL with a host", "http://localhost:4000"),
    ("ws", "wss"): ("a websocket address", "ws://127.0.0.1:8765"),
}


def url_problem(
    url: str, name: str, schemes: tuple[str, ...] = ("http", "https")
) -> str | None:
  """Returns why `url` from setting `name` is not a valid server URL, or None.

  Requires one of `schemes` with a hostname and valid port, and forbids user,
  password, query, or fragment components so credentials are never embedded in
  the URL. Error messages never echo `url`.

  Args:
    url: URL string to validate.
    name: Name of the environment variable or CLI flag providing `url`.
    schemes: Allowed URL schemes (defaults to `("http", "https")`).

  Returns:
    An error message describing the problem, or None if `url` is valid.
  """
  what, example = URL_KINDS[schemes]
  try:
    parts = urllib.parse.urlsplit(url.strip())
    port = parts.port  # ValueError for a port that is not a number
  except ValueError:
    parts, port = None, None
  if (
      parts is None
      or parts.scheme not in schemes
      or not parts.hostname
      or port == 0
  ):
    return f"{name} is not {what}; write it like {example}"
  if "@" in parts.netloc or parts.query or parts.fragment:
    return (
        f"{name} must be the server's address alone, without a user, password,"
        " query or fragment"
        + (
            f" (its key goes in {URL_KEYS[name]})"
            if name in URL_KEYS
            else f"; write it like {example}"
        )
    )
  return None


def redact_url(url: str) -> str:
  """Strips user, password, query, and fragment components from `url`.

  Args:
    url: URL string to sanitize before displaying or storing.

  Returns:
    The sanitized URL containing only scheme, host, port, and path.
  """
  try:
    parts = urllib.parse.urlsplit(url.strip())
    host, port = parts.hostname or "", parts.port
  except ValueError:
    return "(a URL that does not parse)"
  host = f"[{host}]" if ":" in host else host
  return urllib.parse.urlunsplit(
      (parts.scheme, host + (f":{port}" if port else ""), parts.path, "", "")
  )


def loopback_tls_problem(url: str, name: str) -> str | None:
  """Returns a warning if `url` uses HTTPS on a loopback/local address, or None.

  The serve container reaches the host machine as `host.docker.internal`, which
  a local server's TLS certificate rarely covers.

  Args:
    url: Model server URL string.
    name: Name of the variable or flag that provided `url`.

  Returns:
    A warning message if `url` is a local HTTPS URL, or None otherwise.
  """
  parts = urllib.parse.urlsplit(url.strip())
  if parts.scheme != "https" or not this_machine(parts.hostname or ""):
    return None
  return (
      f"{name} is an https URL on this machine, which the serve container"
      " reaches as host.docker.internal: its certificate must cover that name,"
      " or the model call fails its TLS check. Serve a local model server over"
      " http (the traffic stays on this machine)"
  )


def this_machine(host: str) -> bool:
  """Returns True if `host` refers to the local machine (loopback/unspecified).

  Matches `localhost`, `*.localhost`, IPv4/IPv6 loopback or unspecified
  addresses, IPv4-mapped IPv6 loopback (`::ffff:127.0.0.1`), and short or
  numeric IPv4 forms accepted by the C resolver (`127.1`, `0x7f.1`).

  Args:
    host: Hostname or IP literal string to check.

  Returns:
    True if `host` resolves to this machine's loopback or unspecified address.
  """
  name = host.lower().rstrip(".")
  if name == "localhost" or name.endswith(".localhost"):
    return True
  try:
    address = ipaddress.ip_address(name)
  except ValueError:
    try:  # the short and numeric IPv4 forms (127.1), as a resolver reads them
      address = ipaddress.IPv4Address(
          socket.inet_aton(name)  # pylint: disable=g-socket-inet-aton
      )
    except (OSError, ValueError):
      return False
  # ::ffff:127.0.0.1: is_loopback ignores the mapping before 3.13
  if getattr(address, "ipv4_mapped", None):
    address = address.ipv4_mapped
  return address.is_loopback or address.is_unspecified


def model_endpoint(url: str, name: str = "MODEL_PROXY_URL") -> tuple[str, str]:
  """Rewrites `url` for the serve container and returns `(url, hostport)`.

  Replaces local hostnames (`this_machine`) with `host.docker.internal`, makes
  the port explicit, preserves brackets on IPv6 literals, and strips trailing
  slashes.

  Args:
    url: Model server URL string.
    name: Setting name for error reporting.

  Returns:
    A tuple `(served_url, allowed_hostport)` for the serve container and egress
    proxy.

  Raises:
    SetupError: If `url_problem` rejects `url`.
  """
  problem = url_problem(url, name)
  if problem:
    raise SetupError("model", problem)
  parts = urllib.parse.urlsplit(url.strip().rstrip("/"))
  host = parts.hostname or ""
  if this_machine(host):
    host = "host.docker.internal"
  port = parts.port or (443 if parts.scheme == "https" else 80)
  hostport = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
  return (
      urllib.parse.urlunsplit((parts.scheme, hostport, parts.path, "", "")),
      hostport,
  )


@dataclasses.dataclass
class Runtime:
  """Environment, egress allowlists, and warnings for a serve container.

  Holds the container's environment variables, allowed egress hosts, proxy host
  exceptions, and any pre-start model server warnings.
  """

  env: dict[str, str]
  allow: list[str]
  exceptions: list[str]
  notes: list[str] = dataclasses.field(default_factory=list)


def model_base_url_flag_problem(
    sub: spec.Submission, base_url: str | None
) -> str | None:
  """Returns why `--model-base-url` cannot be used with `sub.model`, or None.

  Args:
    sub: Parsed submission specification.
    base_url: Value passed to `--model-base-url`, or None.

  Returns:
    An error message if `base_url` is incompatible or invalid, or None.
  """
  if not base_url:
    return None
  if sub.model.provider == "openai-compatible":
    return url_problem(base_url, "--model-base-url")
  if sub.model.provider == "google-cloud":
    return (
        "--model-base-url replaces an openai-compatible model_base_url; a"
        " google-cloud model is reached through MODEL_PROXY_URL and"
        " MODEL_PROXY_KEY, so point those at your server instead"
    )
  return (
      "--model-base-url replaces model_base_url, so it needs model_api:"
      " openai-compatible/<model>"
  )


def model_server_problem(
    sub: spec.Submission, values: dict[str, str]
) -> str | None:
  """Returns why a `google-cloud` model's proxy settings are invalid, or None.

  Verifies that `MODEL_PROXY_URL` and `MODEL_PROXY_KEY` are present in `values`
  and contain no line breaks or NUL bytes.

  Args:
    sub: Parsed submission specification.
    values: Combined environment and `--env-file` variables.

  Returns:
    An error message if proxy variables are missing or malformed, or None.
  """
  proxy = (
      ("MODEL_PROXY_URL", "MODEL_PROXY_KEY")
      if sub.model.provider == "google-cloud"
      else ()
  )
  missing = [n for n in proxy if not values.get(n)]
  if missing:
    return (
        f"not set: {', '.join(missing)}: export them or put NAME=value lines in"
        " --env-file (google-cloud models run locally through a model server"
        " of your own: see kit/model_proxy; the submission's .env is not read"
        " for them)"
    )
  broken = [n for n in proxy if any(c in values[n] for c in "\r\n\x00")]
  if broken:
    return (
        f"{', '.join(broken)}: a value with a line break cannot reach the serve"
        " container; store it on one line"
    )
  return None


def runtime(
    sub: spec.Submission,
    dotenv: dict[str, str],
    values: dict[str, str],
    base_url: str | None = None,
) -> Runtime:
  """Constructs the serve container `Runtime` environment and egress settings.

  Combines model variables, `sub.env`, and variables from the submission's
  `.env` (`dotenv`). Routes `google-cloud` models through `MODEL_PROXY_URL` and
  `MODEL_PROXY_KEY` from `values`, and applies `base_url` when overriding an
  `openai-compatible` endpoint locally.

  Args:
    sub: Parsed submission specification.
    dotenv: Variables parsed from the submission's `.env` file.
    values: Local environment and `--env-file` variables.
    base_url: Optional `--model-base-url` override.

  Returns:
    A configured `Runtime` instance for the serve container.

  Raises:
    SetupError: If model settings (`stage="model"`) or `.env` variables
      (`stage=".env"`) are invalid.
  """
  proxy = sub.model.provider == "google-cloud"
  problem = model_base_url_flag_problem(sub, base_url)
  if not problem and proxy and values.get("MODEL_PROXY_URL"):
    problem = url_problem(values["MODEL_PROXY_URL"], "MODEL_PROXY_URL")
  if problem:
    raise SetupError("model", problem)
  problems = spec.env_problems(sub, dotenv)
  if problems:
    raise SetupError(spec.ENV_FILE, "; ".join(problems))
  problem = model_server_problem(sub, values)
  if problem:
    raise SetupError("model", problem)
  key = dotenv.get(sub.model.key_secret or "", "")
  name, url = (
      ("MODEL_PROXY_URL", values["MODEL_PROXY_URL"])
      if proxy
      else ("--model-base-url", base_url)
  )
  if url:
    served, hostport = model_endpoint(url, name)
    if proxy:
      env = models.environment(sub.model, values["MODEL_PROXY_KEY"], served)
    else:
      env = models.environment(
          dataclasses.replace(sub.model, base_url=served), key
      )
    exceptions, notes = [hostport], [
        n for n in [loopback_tls_problem(url, name)] if n
    ]
  else:
    env, exceptions, notes = models.environment(sub.model, key), [], []
  env.update({k: spec.fill(v, protocol.PORT) for k, v in sub.env.items()})
  env.update({k: v for k, v in dotenv.items() if k != sub.model.key_secret})
  return Runtime(env, sub.allowed_hosts, exceptions, notes)


def outcome_record(o: relay.Outcome) -> dict[str, Any]:
  """Serializes a `relay.Outcome` into a JSON-compatible dictionary.

  Includes all public dataclass fields of `o` followed by `steps` and `log`.

  Args:
    o: Relay outcome from a smoke or MCP task run.

  Returns:
    A dictionary representation suitable for `smoke.json` or `mcp.json`.
  """
  fields = {
      f.name: getattr(o, f.name)
      for f in dataclasses.fields(o)
      if not f.name.startswith("_")
  }
  return {**fields, "steps": o.steps, "log": o.log}


class Checker:
  """Runs all validation and execution stages for a submission entry."""

  def __init__(
      self,
      src: pathlib.Path,
      work: pathlib.Path,
      values: dict[str, str],
      *,
      smoke: bool = True,
      cancel_test: bool = False,
      base_url: str | None = None,
      keep: bool = False,
      rebuild: bool = False,
  ):
    self.src, self.work, self.values = src.resolve(), work, values
    self.smoke, self.cancel_test, self.base_url, self.keep = (
        smoke,
        cancel_test,
        base_url,
        keep,
    )
    self.rebuild = rebuild
    self.report = Report(str(src), requested=["cancel"] if cancel_test else [])
    # started and not yet removed: abort() removes them
    self.stacks: list[stack.Stack] = []
    self.tasks: list[str] = []  # task containers, likewise

  def kept(self) -> str:
    """Returns the `docker` command to remove containers kept with `--keep`.

    Returns:
      A shell command string to remove kept containers and networks, or `""`
      when nothing was kept.
    """
    if not self.keep or not (self.stacks or self.tasks):
      return ""
    containers: list[str] = []
    for st in self.stacks:
      containers.extend((st.serve, st.egress))
    containers.extend(self.tasks)
    networks = [st.network for st in self.stacks]
    removal = f"docker rm -f {' '.join(containers)}"
    return removal + (
        f"; docker network rm {' '.join(networks)}" if networks else ""
    )

  def abort(self) -> None:
    """Removes active stacks and task containers unless `--keep` is set.

    Cleans up any containers left running if an interrupt stops the check
    between container startup and stage teardown.
    """
    if self.keep:
      return
    for st in self.stacks:
      st.stop()  # never raises
    for name in self.tasks:
      with contextlib.suppress(stack.InfraError):
        stack.docker("rm", "-f", name, check=False)
    self.stacks, self.tasks = [], []

  def run(self) -> Report:
    """Executes all configured check stages and returns the final `Report`.

    Returns:
      The completed `Report` for the submission.
    """
    r = self.report
    try:
      sub = spec.load(self.src)
    except FileNotFoundError:
      r.add("submission.yaml", "fail", f"no {spec.FILE} in {self.src}")
      return r
    except spec.SpecError as exc:
      r.add("submission.yaml", "fail", "\n  - " + "\n  - ".join(exc.problems))
      return r
    r.submission = sub.name
    r.add(
        "submission.yaml",
        "pass",
        f"{sub.name}: {sub.model.api}, egress"
        f" {', '.join(sub.allowed_hosts) or 'none'}",
    )
    self._stages(sub)
    if r.ok:  # like every stage, only when none before it failed
      self._artifacts()
    if self.base_url and not r.unready:
      r.unready = "model-override"
    return r

  def _stages(self, sub: spec.Submission) -> None:
    """Runs stages from `.env` through `cancel`, stopping on the first failure.

    Args:
      sub: Parsed submission specification.
    """
    r = self.report
    # kit up's and eval/run.py's checks, in their order, before anything is
    # staged or built
    try:
      dotenv = entry_env(self.src)
      rt = runtime(sub, dotenv, self.values, self.base_url)
    except SetupError as exc:
      r.add(exc.stage, "fail", str(exc))
      return
    r.add(
        spec.ENV_FILE,
        "pass",
        ", ".join(dotenv) or f"none ({sub.model.api} needs no key)",
    )
    for note in rt.notes:
      r.add("model", "warn", note)
    # another kit run on this entry finishes its build first
    with build.locked(self.work):
      try:
        source = build.stage(self.src, self.work / "source")
      # a submodule not checked out, a repository over the limit, ...
      except build.BuildError as exc:
        r.add("files", "fail", str(exc))
        return
      self._files(sub, source)
      if not r.ok:
        return
      try:
        prep = build.prepare(sub, source, self.work, rebuild=self.rebuild)
      except build.BuildError as exc:
        # no such file when the build failed before it started
        log = self.work / "build.log"
        tail = (
            "\n".join(log.read_text(errors="replace").splitlines()[-25:])
            if log.is_file()
            else ""
        )
        r.add("build", "fail", f"{exc}\n{tail}".rstrip())
        return
      # Docker could not run the builder: not the install's doing
      except stack.InfraError as exc:
        r.add("build", "fail", f"the build could not run (Docker): {exc}")
        return
    if prep.reused:
      how = (
          "reused: files, install and builder unchanged since the last build"
          " (--rebuild to build anyway)"
      )
    else:
      how = f"built in {prep.bundle.seconds:.0f} s"
    r.add(
        "build",
        "pass",
        f"bundle {prep.bundle.size / 2**20:.0f} MiB, {how}; image {prep.image}",
    )
    asyncio.run(self._serve(sub, prep.image, rt))

  def _files(self, sub: spec.Submission, source: build.Source) -> None:
    """Validates repository structure, git state, submodules, and secrets.

    Args:
      sub: Parsed submission specification.
      source: Staged submission source files and metadata.
    """
    r = self.report
    # The server builds a commit of a repository with submission.yaml at its
    # root. In any other folder (one of the competition repository, one outside
    # git, one below its repository's root) the kit does not check what is
    # committed or whether git keeps .env, because committing there would be the
    # wrong advice.
    root, clone = build.repository_root(self.src), build.competition_clone(
        self.src
    )
    own = (
        f"what is committed, and whether git keeps your {spec.ENV_FILE}, is"
        " checked in a repository of its own"
    )
    if clone:
      r.unready = "clone"
      r.add(
          "files",
          "warn",
          "this entry is a folder of your clone of the competition repository"
          f" ({clone}), whose .gitignore keeps {spec.ENV_FILE} out of git:"
          f" {own}",
      )
    elif root is None:
      r.unready = "no-git"
      r.add(
          "files",
          "warn",
          "this folder is not a git repository, and the server builds a"
          f" commit: {own}",
      )
    elif root.resolve() != self.src.resolve():
      r.unready = "subfolder"
      r.add(
          "files",
          "warn",
          f"this folder is below the root of its git repository ({root}), and"
          f" the server builds a repository with {spec.FILE} at its root:"
          f" {own}",
      )
    # the artifacts stage reports what is uncommitted under artifacts/; the
    # other artifacts-like folders are neither graded nor bundled
    # (artifacts.bundle_excluded)
    uncommitted = (
        []
        if r.unready
        else [u for u in source.uncommitted if not artifacts.bundle_excluded(u)]
    )
    ignored = not r.unready and build.env_ignored(self.src)
    hidden = [] if r.unready else build.flagged(self.src)
    shallow = (
        not r.unready
        and build._git(  # pylint: disable=protected-access
            self.src, "rev-parse", "--is-shallow-repository"
        )
        == "true\n"
    )
    if uncommitted:
      r.add(
          "files",
          "warn",
          "not committed (the server builds a commit):"
          f" {artifacts.shown(uncommitted)}",
      )
    if ignored:
      r.add(
          "files",
          "warn",
          f"git ignores {spec.ENV_FILE}, so the server will not get it: commit"
          f" it with `git add -f {spec.ENV_FILE}` (in a private repository)",
      )
    if hidden:
      r.add(
          "files",
          "warn",
          f"git is set to skip changes to {artifacts.shown(hidden)}"
          " (assume-unchanged or skip-worktree), so it cannot tell whether the"
          " commit holds them as they are here: clear the flag (`git"
          " update-index --no-assume-unchanged --no-skip-worktree -- <file>`),"
          " then check again",
      )
    if shallow:
      r.add(
          "files",
          "warn",
          "this clone is shallow, so the kit measured only part of the git"
          " history the server clones, which counts toward the"
          f" {build.MAX_SOURCE_BYTES >> 30} GiB repository limit: run `git"
          " fetch --unshallow`, then check again",
      )
    if uncommitted or ignored or hidden:
      r.unready = "uncommitted"
    # spec.load refused a serve that does not split
    words = shlex.split(sub.serve, comments=True)
    missing = [
        t
        for t in words
        if t.startswith("{src}/")
        and not (source.path / t[len("{src}/") :]).exists()
    ]
    # install may make them (a build step's output): listen fails if it does
    # not
    if missing:
      r.add(
          "files",
          "warn",
          "serve names files that are not in the repository:"
          f" {', '.join(missing)}; install must create them",
      )
    if source.left_out:
      shown = artifacts.shown([f"{f}/" for f in source.left_out])
      r.add(
          "files",
          "warn",
          f"left out of the build, as on the server: {shown} (rename what your"
          " harness needs)",
      )
    if source.lfs:
      r.add(
          "files",
          "fail",
          f"Git LFS stores {artifacts.shown(source.lfs)}, and the server clones"
          " without Git LFS, so its build gets the pointer files: commit the"
          " files themselves, or download them in install",
      )
    submodules = [] if r.unready else build.submodule_problems(self.src)
    for status, line in submodules:
      r.add("files", status, line)
    found, unscanned = self._key_scan(source)
    # file:line, outside or inside a dependency's code or test data
    keys, advice = [], []
    for hit in found:
      (advice if vendored(hit.rsplit(":", 1)[0]) else keys).append(hit)
    if keys:
      r.add(
          "files",
          "fail",
          f"what looks like an API key or private key: {artifacts.shown(keys)} "
          f"(keys go in {spec.ENV_FILE} only)",
      )
    if advice:
      r.add(
          "files",
          "warn",
          "what looks like an API key or private key, in a dependency's code"
          f" or test data: {artifacts.shown(advice)} (advice: the server does"
          " not scan for keys; make sure none is yours, which go in"
          f" {spec.ENV_FILE})",
      )
    if unscanned:
      r.add(
          "files",
          "warn",
          f"not scanned for keys (unreadable, or past the scan's {KEY_SCAN_SEC}"
          f" s; the next check reads them again): {artifacts.shown(unscanned)}",
      )
    if not any(s.name == "files" for s in r.stages):
      r.add("files", "pass", f"{source.files} files")

  def _artifacts(self) -> None:
    """Reviews the entry's `artifacts/` directory under evaluation server rules.

    Checks `artifacts.review_directory`, uncommitted or git-ignored artifact
    files, and suspected API keys in graded files. Missing artifacts produce a
    warning and mark the report `unready`, while unreadable or rejected
    artifacts fail the stage.
    """
    r = self.report
    rv = artifacts.review_directory(self.src)
    # nothing was reviewed: a problem to fix on this machine, not the
    # server's verdict
    if rv.unreadable:
      r.add(
          "artifacts",
          "fail",
          "the kit cannot review them on this machine:"
          + "".join(f"\n  - {line}" for line in rv.problems + rv.warnings),
      )
      return
    if rv.absent:
      # keep any other problem (a hint about an archive elsewhere) and every
      # warning
      extra = [
          p for p in rv.problems if p != artifacts.NO_ARTIFACTS
      ] + rv.warnings
      run = f"uv run python eval/run.py {shlex.quote(str(self.src))}"
      r.add(
          "artifacts",
          "warn",
          f"no {artifacts.ROOT}/ yet: `{run}` writes it"
          + "".join(f"\n  - {line}" for line in extra),
      )
      if r.unready in ("", "uncommitted"):
        r.unready = "artifacts"
      return
    if not rv.accepted:
      # "no dev task has usable artifacts" says to see each task's reasons: name
      # them here, as the server does
      reasons = [
          f"{t.id}: {t.problems[0]}"
          for t in rv.tasks.values()
          if t.status in (artifacts.REFUSED, artifacts.EMPTY) and t.problems
      ]
      if len(reasons) > 12:
        more = (
            f"... and {len(reasons) - 12} more (python -m kit artifacts"
            f" {shlex.quote(str(self.src))})"
        )
      else:
        more = None
      lines = rv.problems + reasons[:12] + ([more] if more else [])
      r.add(
          "artifacts",
          "fail",
          "the server would refuse them:\n  - " + "\n  - ".join(lines),
      )
      return
    c = rv.counts()
    notes = list(rv.warnings)
    for t in rv.tasks.values():
      if t.status != artifacts.MISSING:
        for line in t.problems + t.warnings:
          notes.append(f"{t.id}: {line}")
    rooted = r.unready not in ("clone", "no-git", "subfolder")
    # what git ignores is in the review's warnings (from_directory asks git)
    if rooted:
      # the folder name is the participant's (runs/<run>/artifacts,
      # artifacts/<wrapper>): match it literally
      pending = (
          build._git(  # pylint: disable=protected-access
              self.src,
              "status",
              "--porcelain",
              "--untracked-files=all",
              "--",
              f":(literal){rv.root}",
          )
          or ""
      ).strip()
      if pending:
        notes.append(
            "not committed (the server reads the commit you submit):"
            f" {len(pending.splitlines())} change(s) under {rv.root}/"
        )
        r.unready = r.unready or "uncommitted"
    keys = self._artifact_keys(rv)
    if keys:
      notes.append(
          "what looks like an API key, in a file the graders read:"
          f" {', '.join(keys[:5])} (advice: make sure it is no key of yours)"
      )
    detail = f"{c[artifacts.OK]} of {len(rv.tasks)} dev tasks usable" + "".join(
        f", {c[k]} {k}"
        for k in (artifacts.EMPTY, artifacts.REFUSED, artifacts.MISSING)
        if c[k]
    )
    shown = notes[:15]
    if len(notes) > 15:
      shown.append(
          f"... and {len(notes) - 15} more (python -m kit artifacts"
          f" {shlex.quote(str(self.src))})"
      )
    detail += "".join(f"\n  - {n}" for n in shown) + "".join(
        f"\n  - note: {n}" for n in rv.notes
    )
    r.add("artifacts", "warn" if notes else "pass", detail)

  def _artifact_keys(self, rv: artifacts.Review) -> list[str]:
    """Scans regular files under the artifacts folder for suspected keys.

    Args:
      rv: Completed directory review containing the artifacts root path.

    Returns:
      A list of `"file:line"` strings where suspected keys were found.
    """
    deadline, found = time.monotonic() + KEY_SCAN_SEC, []
    for p in artifacts.regular_files(self.src / rv.root):
      try:
        found += [
            f"{p.relative_to(self.src).as_posix()}:{line}"
            for line in key_lines(p, deadline) or ()
        ]
      except OSError:
        continue
    return found

  def _key_scan(self, source: build.Source) -> tuple[list[str], list[str]]:
    """Scans staged files in `source` for suspected API or private keys.

    Caches per-file scan results keyed by SHA-256 digest and `key_rules()` in
    `key-scan.json` so unchanged files are not rescanned on subsequent checks.

    Args:
      source: Staged submission source files and digests.

    Returns:
      A tuple `(found_hits, unscanned_files)` where `found_hits` are
      `"file:line"` strings and `unscanned_files` are relative paths that could
      not be read before `KEY_SCAN_SEC`.
    """
    saved, rules = self.work / "key-scan.json", key_rules()
    try:
      data = json.loads(saved.read_text())
      last = data.get("files") if data.get("rules") == rules else None
    except (OSError, ValueError, AttributeError):
      last = None
    last = last if isinstance(last, dict) else {}
    files: dict[str, list[Any]] = {}
    deadline = time.monotonic() + KEY_SCAN_SEC
    for rel, digest in sorted(source.digests.items()):
      old = last.get(rel)
      if (
          digest
          and isinstance(old, list)
          and len(old) == 2
          and old[0] == digest
          and isinstance(old[1], list)
      ):
        files[rel] = old
        continue
      # a link has no content of its own to scan
      p, lines = source.path / rel, []
      if p.is_file() and not p.is_symlink():
        try:
          lines = key_lines(p, deadline)
        except OSError:
          lines = None
      files[rel] = [digest, lines]
    saved.write_text(json.dumps({"rules": rules, "files": files}))
    hits: list[str] = []
    for rel, (_, lines) in files.items():
      for n in lines or ():
        hits.append(f"{rel}:{n}")
    unscanned = [rel for rel, (_, lines) in files.items() if lines is None]
    return hits, unscanned

  async def _start(
      self,
      sub: spec.Submission,
      image: str,
      rt: Runtime,
      command: str | None = None,
  ) -> stack.Stack:
    """Starts a serve stack and registers it in `self.stacks`.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.
      command: Optional command override in place of `sub.serve`.

    Returns:
      The started `stack.Stack` instance.
    """
    # owner "keep": no age sweep removes what --keep keeps
    st = stack.Stack(keep=self.keep)
    command = command or spec.fill(sub.serve, protocol.PORT)
    # docker reads it once, as it creates the container
    with private_env_file(rt.env) as env_file:
      await stack.settled(
          functools.partial(
              st.start,
              image=image,
              command=command,
              env_file=str(env_file),
              allow=rt.allow,
              exceptions=rt.exceptions,
          ),
          st.stop,
      )
    self.stacks.append(st)
    return st

  async def _task_container(self, mcp: bool = False) -> tuple[str, str]:
    """Starts an isolated task container and returns `(container_name, word)`.

    Runs with `--network none` and `TASK_LIMITS`. When `mcp` is True, starts
    `MCP_STUB` inside the container and waits until it prints its secret word.

    Args:
      mcp: Whether to run the MCP stub server inside the task container.

    Returns:
      A tuple `(container_name, mcp_secret_word)` where `mcp_secret_word` is
      `""` when `mcp` is False.

    Raises:
      stack.InfraError: If Docker fails to start the container or the MCP stub
        does not start within 60 seconds.
    """
    name = f"plebench-task-{uuid.uuid4().hex[:12]}"
    command = (
        ("python3", "-I", "-c", MCP_STUB, str(MCP_PORT))
        if mcp
        else ("sleep", "infinity")
    )

    def remove():
      stack.docker("rm", "-f", name, check=False)

    # like its stack's: no age sweep removes a kept one
    owner = "keep" if self.keep else name
    try:
      await stack.settled(
          functools.partial(
              stack.docker,
              "run",
              "-d",
              "--name",
              name,
              "--label",
              f"{stack.LABEL}={owner}",
              "--network",
              "none",
              *TASK_LIMITS,
              *stack.LOG_OPTS,
              "-w",
              "/app",
              TASK_IMAGE,
              *command,
              timeout=600,
          ),
          remove,
      )
    # `docker run` can fail after it made the container
    except stack.InfraError:
      with contextlib.suppress(stack.InfraError):
        await asyncio.to_thread(remove)
      raise
    self.tasks.append(name)
    word, deadline = "", time.monotonic() + 60
    while mcp and not word:
      if time.monotonic() > deadline:
        raise stack.InfraError(
            "the MCP stub in the task container did not start within 60 s"
        )
      await asyncio.sleep(1)
      word = (
          await asyncio.to_thread(stack.docker, "logs", name, timeout=30)
      ).stdout.strip()
    return name, word

  @contextlib.asynccontextmanager
  async def _running(
      self,
      sub: spec.Submission,
      image: str,
      rt: Runtime,
      suffix: str = "",
      command: str | None = None,
  ):
    """Manages a serve stack for a single stage and tears it down afterward.

    Ensures `TASK_IMAGE` is present before starting the serve stack so image
    pull time never counts against `protocol.START_SEC`. Saves
    `serve<suffix>.log` and `egress<suffix>.log` on exit and removes the stack
    and task container unless `--keep` is set.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.
      suffix: Filename suffix for stage logs.
      command: Optional command override in place of `sub.serve`.

    Yields:
      The active `_Run` instance, or None if Docker could not start the stack.
    """
    await asyncio.to_thread(stack.ensure_image, TASK_IMAGE)
    try:
      st = await self._start(sub, image, rt, command)
    except stack.InfraError as exc:
      self.report.add(
          "serve",
          "fail",
          f"the serve container could not start (Docker): {exc}",
      )
      yield None
      return
    run = _Run(self, st, suffix)
    try:
      yield run
    finally:
      await asyncio.shield(self._teardown(run))

  async def _teardown(self, run: _Run) -> None:
    """Saves the stage's logs and removes its stack and task container.

    Args:
      run: Active stage `_Run` to tear down.
    """
    await run.save_log()
    (self.work / f"egress{run.suffix}.log").write_bytes(
        await asyncio.to_thread(run.st.logs, "egress")
    )
    if not self.keep:
      await asyncio.to_thread(run.st.stop)
      if run.st in self.stacks:
        self.stacks.remove(run.st)
      if run.task:
        await asyncio.to_thread(stack.docker, "rm", "-f", run.task, check=False)
        if run.task in self.tasks:
          self.tasks.remove(run.task)

  async def _serve(self, sub: spec.Submission, image: str, rt: Runtime) -> None:
    """Runs the model, listen, smoke, MCP, and cancel stages against `image`.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.
    """
    r = self.report
    if not await self._model(sub, image, rt):
      return
    async with self._running(sub, image, rt) as run:
      if run is None:
        return
      rl, ws = await self._listen(run)
      if not self.smoke:
        r.add("smoke", "skip")
      elif ws is not None:
        await self._task(run, rl, ws, "smoke", "HELLO")
    if self.smoke and r.ok:
      await self._mcp(sub, image, rt)
    if self.cancel_test and r.ok:
      await self._cancel(sub, image, rt)

  async def _model(self, sub: spec.Submission, image: str, rt: Runtime) -> bool:
    """Runs the model connectivity probe inside an isolated serve container.

    Probes the configured model endpoint through the egress proxy from a
    container running `sleep infinity` so harness startup behavior cannot mask
    model errors and `listen` timing starts from a fresh container.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.

    Returns:
      True if the model stage passed, False otherwise.

    Raises:
      stack.InfraError: If `docker exec` fails for a reason other than timeout.
    """
    r = self.report
    async with self._running(
        sub, image, rt, suffix="-model", command="exec sleep infinity"
    ) as run:
      if run is None:
        return False
      # isolated (-I): it starts in the entry's checkout, whose json.py
      # must not replace the stdlib's
      try:
        probe = await asyncio.to_thread(
            run.st.exec, "python3", "-I", "-c", PROBE, timeout=PROBE_EXEC_SEC
        )
      except stack.InfraError as exc:
        if not isinstance(exc.__cause__, subprocess.TimeoutExpired):
          raise
        r.add(
            "model",
            "fail",
            f"{sub.model.api} did not answer within {PROBE_EXEC_SEC} s through"
            " the egress proxy: check that its host is reachable and the model"
            f" is serving{await self._refusal(run)}",
        )
        return False
      # Docker's own error, and nothing the probe printed: Docker could not run
      # it, so the model was never asked
      refused = (
          None
          if probe.stdout.strip()
          else relay.docker_failure(probe.returncode, probe.stderr)
      )
      if refused:
        r.add(
            "model",
            "fail",
            "the probe could not run in the serve container, so it never"
            f" reached {sub.model.api}: {refused}",
        )
        return False
      try:
        result = json.loads(probe.stdout.strip().splitlines()[-1])
      except (ValueError, IndexError):
        result = None
      # not the probe's answer
      if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
        result = {
            "ok": False,
            "error": (
                (probe.stdout + probe.stderr).strip()[-300:] or "no answer"
            ),
        }
      if not result["ok"]:
        r.add(
            "model",
            "fail",
            f"{sub.model.api} through the egress proxy: {result.get('error')}\n"
            "  check the key, the model name and that its host is reachable"
            f"{await self._refusal(run)}",
        )
        return False
      r.add(
          "model",
          "pass",
          f"{sub.model.label} answered in {result.get('seconds')} s through the"
          " egress proxy",
      )
      return True

  async def _refusal(self, run: _Run) -> str:
    """Formats the egress proxy's last refused or failed connection log line.

    Args:
      run: Active stage `_Run` whose egress proxy log should be inspected.

    Returns:
      A formatted diagnostic suffix line, or `""` if no refusal was logged.
    """
    log, where = (
        await asyncio.to_thread(run.st.logs, "egress"),
        f"see {self.work / f'egress{run.suffix}.log'}",
    )
    for line in reversed(log.decode(errors="replace").splitlines()):
      try:
        entry = json.loads(line)
      except ValueError:
        continue
      if not isinstance(entry, dict):
        continue
      if entry.get("kind") == "refused":
        return (
            f"\n  the egress proxy refused a connection: {entry.get('reason')}"
            f" ({where})"
        )
      if entry.get("kind") == "failed":
        return (
            "\n  the egress proxy could not connect to"
            f" {entry.get('host')}:{entry.get('port')}: {entry.get('error')}"
            f" ({where})"
        )
    return ""

  async def _listen(self, run: _Run):
    """Checks that the server accepts a websocket within `protocol.START_SEC`.

    Args:
      run: Active stage `_Run` to connect to.

    Returns:
      A `(relay, websocket)` tuple, where `websocket` is None if connection
      failed.
    """
    rl, ws = await run.connect("listen")
    if ws is not None:
      self.report.add(
          "listen",
          "pass",
          f"websocket accepted {rl.outcome.connect_sec:.0f} s after start",
      )
      if not self.smoke:
        # the server sees a connection that closes before a task, and
        # serves the next
        await ws.close()
    return rl, ws

  async def _task(
      self, run: _Run, rl: relay.Relay, ws: Any, stage: str, want: str
  ) -> None:
    """Runs the smoke or MCP task over `ws` and verifies the output file.

    Task errors record a `FAIL` in the `smoke` stage and a `WARN` in the `mcp`
    stage because MCP tool support is optional for a harness.

    Args:
      run: Active stage `_Run`.
      rl: Configured `relay.Relay` instance.
      ws: Open websocket connection to the harness server.
      stage: Stage name (`"smoke"` or `"mcp"`).
      want: Expected stripped content of the output file created in `/app`.
    """
    r, (prompt, name) = (
        self.report,
        {"smoke": (SMOKE, "hello.txt"), "mcp": (MCP_TASK, "mcp.txt")}[stage],
    )
    started = time.monotonic()
    o = await rl.run(ws, prompt)
    code, content, unread = None, "", ""
    try:
      code, content = await rl.executor.exec(
          f"cat /app/{name}", cwd="/app", timeout_sec=30
      )
    # Docker's failure (an InfraError), or its exec not back in time
    except RuntimeError as exc:
      # decide whose failure it is by the relay's rule for requests: the
      # harness's (the container could not run it, or its commands wedged the
      # container) or the evaluator's (Docker failing, the container gone)
      failure = await rl._failure(  # pylint: disable=protected-access
          f"reading /app/{name}", exc
      )
      if isinstance(failure, relay.RequestError):
        unread = str(failure)
      else:
        o.infra = o.infra or str(failure)
    # written after the read, so the relay's note about a container that no
    # longer answers is in the log too
    log = self.work / f"relay{run.suffix}.log"
    log.write_text("\n".join(o.log) + "\n", encoding="utf-8", errors="replace")
    (self.work / f"{stage}.json").write_text(
        json.dumps(outcome_record(o), indent=1, default=str)
    )
    took = time.monotonic() - started
    if o.status != "completed":
      problem = (
          f"{o.reason} after {took:.0f} s, {o.requests} request(s); see {log}"
      )
    elif unread:
      problem = (
          f"done after {o.requests} request(s), but /app/{name} could not be"
          f" read: {unread}; see {log}"
      )
    elif code != 0 or content.strip() != want:
      problem = (
          f"done after {o.requests} request(s), but /app/{name} holds"
          f" {content.strip()[:80]!r}"
      )
    elif stage == "mcp" and not o.kinds.get("http"):
      problem = (
          f"/app/{name} is right, but no request was an http request: the"
          " harness found it another way"
      )
    else:
      problem = ""
    if o.infra:
      r.add(
          stage,
          "fail",
          "the evaluator's side failed here, not your harness (the local task"
          f" container or egress proxy): {o.infra}",
      )
    elif problem and stage == "smoke":
      r.add(stage, "fail", problem)
    elif problem:
      r.add(
          stage,
          "warn",
          f"{problem}; held-out tasks can expose MCP servers like this one"
          " (docs/protocol.md)",
      )
    else:
      if o.models:
        usage = (
            f", {o.usage['input_tokens']} in / {o.usage['output_tokens']} out"
            " tokens"
        )
      else:
        usage = ", no usage frames (optional)"
      r.add(
          stage,
          "pass",
          f"{name} written, done in {took:.0f} s with {o.requests}"
          f" request(s){usage}; final message {o.final_message[:60]!r}",
      )
      if stage == "smoke" and not o.final_message.strip():
        r.add(
            "final message",
            "warn",
            "empty: graders read it; end with a summary of what you delivered",
        )

  async def _mcp(self, sub: spec.Submission, image: str, rt: Runtime) -> None:
    """Runs the MCP stage using `MCP_STUB` in a fresh stack and task container.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.
    """
    async with self._running(sub, image, rt, suffix="-mcp") as run:
      if run is None:
        return
      rl, ws = await run.connect("mcp", mcp=True)
      if ws is not None:
        await self._task(run, rl, ws, "mcp", run.word)

  async def _cancel(
      self, sub: spec.Submission, image: str, rt: Runtime
  ) -> None:
    """Runs the optional `cancel` stage to verify graceful task cancellation.

    Args:
      sub: Parsed submission specification.
      image: Serve Docker image tag.
      rt: Container runtime environment and egress configuration.
    """
    r = self.report
    async with self._running(sub, image, rt, suffix="-cancel") as run:
      if run is None:
        return
      rl, ws = await run.connect("cancel")
      if ws is None:
        return
      # late enough that the harness is mid-task
      o = await rl.run(ws, LONG, cancel_after=45)
      log = self.work / f"relay{run.suffix}.log"
      log.write_text(
          "\n".join(o.log) + "\n", encoding="utf-8", errors="replace"
      )
      if o.infra:
        r.add(
            "cancel",
            "fail",
            "the evaluator's side failed here, not your harness (the local"
            f" task container or egress proxy): {o.infra}",
        )
      elif o.status == "completed":
        r.add(
            "cancel",
            "skip",
            "the task finished before the cancel was sent, so cancelling was"
            " not tested",
        )
      # the session ended (error, a closed socket) before the time was up
      elif o.status != "interrupted":
        r.add(
            "cancel",
            "skip",
            f"the harness ended before the cancel: {o.reason}; cancelling was"
            " not tested",
        )
      elif o.done_after_cancel:
        r.add(
            "cancel",
            "pass",
            f"done within {protocol.CANCEL_GRACE_SEC} s of the cancel",
        )
      else:
        r.add(
            "cancel",
            "fail",
            f"no done within {protocol.CANCEL_GRACE_SEC} s of the cancel"
            f" ({o.reason}): stop and send done, the final message still"
            " counts",
        )


class _Run:
  """Holds one stage's serve stack, lazy task container, and relay state.

  `suffix` names the stage's log files: `serve<suffix>.log`,
  `egress<suffix>.log`, and `relay<suffix>.log`.
  """

  def __init__(self, checker: Checker, st: stack.Stack, suffix: str):
    self.checker, self.st, self.suffix, self.task, self.word = (
        checker,
        st,
        suffix,
        None,
        "",
    )
    self.log = checker.work / f"serve{suffix}.log"

  async def save_log(self) -> bytes:
    """Writes the serve container's output so far to `self.log` and returns it.

    Returns:
      The raw log bytes from the serve container.
    """
    logs = await asyncio.to_thread(self.st.logs, "serve")
    self.log.write_bytes(logs)
    return logs

  async def connect(self, stage: str, mcp: bool = False):
    """Connects a `relay.Relay` to the serve container for `stage`.

    Starts the task container when the stage will run a task
    (`self.checker.smoke` or `stage != "listen"`). Records a `FAIL` on `stage`
    and returns `(relay, None)` if the server does not accept the websocket
    connection.

    Args:
      stage: Stage name (`"listen"`, `"mcp"`, or `"cancel"`).
      mcp: Whether to start the MCP stub in the task container.

    Returns:
      A `(relay, websocket)` tuple, where `websocket` is None on failure.
    """
    if self.task is None and (self.checker.smoke or stage != "listen"):
      # pylint: disable-next=protected-access
      self.task, self.word = await self.checker._task_container(mcp)
    rl = relay.Relay(
        relay.DockerExecutor(self.task or ""),
        workdir="/app",
        stack=self.st,
        mcp_servers=MCP_SERVERS if mcp else None,
        agent_timeout_sec=MCP_TIMEOUT_SEC if mcp else SMOKE_TIMEOUT_SEC,
    )
    try:
      ws = await rl.connect(
          self.st.url,
          alive=lambda: asyncio.to_thread(self.st.running),
          start=self.st.started_at,
      )
    except ConnectionError as exc:
      # the relay's verdict stands; how the server ended, and its last output,
      # are added when Docker says it stopped
      exited = await asyncio.to_thread(stack.exit_note, self.st)
      logs = await self.save_log()
      what = (
          ""
          if stage == "listen"
          else "the server did not accept a connection: "
      )
      if exited:
        end = (
            f" ({exited}); the end of its output"
            f" ({self.log}):\n{stack.output_tail(logs)}"
        )
      else:
        end = f"; see {self.log}"
      self.checker.report.add(stage, "fail", f"{what}{exc}{end}")
      return rl, None
    return rl, ws


# registry kind (stack.register): the stacks `kit up` keeps, by run id
UP = "up"
# registry kind: the private directories env files with keys and secrets are
# written in
SECRETS = "secrets"


def register_up(run: str) -> pathlib.Path:
  """Registers this `kit up` process and its stack's `run` ID for cleanup.

  Because kept stacks are exempt from age-based sweeps, `sweep_dead_ups()` uses
  this registration to remove the stack once the `kit up` process exits.

  Args:
    run: Stack run identifier.

  Returns:
    The path to the registration file created by `stack.register`.
  """
  return stack.register(UP, {"run": run}, key=run)


def sweep_dead_ups() -> int:
  """Removes stacks belonging to `kit up` processes that are no longer running.

  Returns:
    The number of containers and networks removed.
  """
  removed = [0]

  def gone(payload: dict[str, Any]) -> None:
    run = payload.get("run")
    if isinstance(run, str) and stack.RUN_RE.fullmatch(run):
      removed[0] += stack.sweep(run=run)

  stack.live(UP, gone=gone)
  return removed[0]


def private_root() -> pathlib.Path:
  """Returns the private directory where temporary Docker env files are stored.

  Located under `stack.kit_root()` rather than user-shared work directories.

  Returns:
    The `Path` to the `private_env` directory.
  """
  return stack.kit_root() / "private_env"


@contextlib.contextmanager
def private_env_file(values: dict[str, str]):
  """Writes `values` to a temporary `0600` Docker env file in a `0700` folder.

  Stores secrets under `private_root()` rather than a work or run directory that
  a user might share or archive, and removes the folder when the context exits
  (or on the next run via `sweep_secrets` if the process is killed).

  Args:
    values: Environment variable mapping to write.

  Yields:
    The `Path` to the temporary `serve.env` file.
  """
  key = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
  folder = private_root() / key
  # first, so a later run can sweep it
  registration = stack.register(SECRETS, {"dir": str(folder)}, key=key)
  try:
    private_root().mkdir(parents=True, exist_ok=True)
    os.chmod(private_root(), 0o700)
    folder.mkdir(mode=0o700)
    path = folder / "serve.env"
    stack.write_env_file(path, values)
    yield path
  finally:
    shutil.rmtree(folder, ignore_errors=True)
    stack.unregister(registration)


def sweep_secrets() -> None:
  """Removes temporary env-file directories left by dead kit processes.

  Only directories directly under `private_root()` are ever removed.
  """

  def gone(payload: dict[str, Any]) -> None:
    folder = pathlib.Path(str(payload.get("dir") or ""))
    if folder.name and folder.parent == private_root():
      shutil.rmtree(folder, ignore_errors=True)

  stack.live(SECRETS, gone=gone)


def sweep_leftovers() -> None:
  """Removes leftover secret env files, dead `kit up` stacks, and stale stacks.

  Silently ignores `stack.InfraError` if Docker is unreachable so the command's
  own stages can report it.
  """
  sweep_secrets()
  with contextlib.suppress(stack.InfraError):
    sweep_dead_ups()
    stack.sweep()


def tolerant_output() -> None:
  """Configures `stdout` and `stderr` to backslash-escape unencodable chars.

  Prevents `UnicodeEncodeError` when printing output from a harness or server on
  terminals with non-UTF-8 encodings.
  """
  for stream in (sys.stdout, sys.stderr):
    # a replaced stream (no reconfigure) or a closed one
    with contextlib.suppress(AttributeError, ValueError):
      stream.reconfigure(errors="backslashreplace")


@contextlib.contextmanager
def stops_on_signals():
  """Converts `SIGTERM` and `SIGHUP` into `KeyboardInterrupt` for cleanup.

  Restores the previous signal handlers when the context exits.
  """
  # only the main thread may set handlers
  main = threading.current_thread() is threading.main_thread()
  saved = (
      {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
      if main
      else {}
  )
  for sig in saved:
    signal.signal(sig, signal.default_int_handler)
  try:
    yield
  finally:
    for sig, handler in saved.items():
      signal.signal(sig, handler)


def check(src: str, work: str | None, env_file: str | None, **kw: Any) -> int:
  """Runs `kit check` on `src`, writes `check.json`, and returns an exit code.

  Args:
    src: Path to the submission directory or `submission.yaml`.
    work: Optional work directory path override.
    env_file: Optional `--env-file` path for local proxy/judge variables.
    **kw: Additional keyword arguments forwarded to `Checker`.

  Returns:
    `0` if all requested stages passed, `130` if interrupted, or `1` on failure.
  """
  try:
    sub_dir = spec.entry_dir(src)
  except spec.EntryError as exc:
    print(f"[FAIL] submission.yaml: {exc}")
    return 1
  values = {**os.environ, **read_env_file(env_file)}
  try:
    name = spec.load(sub_dir).name
  except Exception:  # pylint: disable=broad-exception-caught
    # the checker reports it
    name = sub_dir.resolve().name
  try:
    work_dir = build.work_dir(name, work, src=sub_dir)
  # inside the submission, or not writable: no check.json either
  except (build.BuildError, OSError) as exc:
    print(f"[FAIL] kit: {exc}")
    return 1
  checker = Checker(sub_dir, work_dir, values, **kw)
  interrupted = False
  with stops_on_signals():
    try:
      build.require_local_daemon()
      sweep_leftovers()  # with Docker down, the build stage says so
      report = checker.run()
    # whatever a stage raised becomes a FAIL line and check.json, not a
    # traceback
    except Exception as exc:  # pylint: disable=broad-exception-caught
      report, detail = checker.report, f"{type(exc).__name__}: {exc}"
      if not isinstance(
          exc, (stack.InfraError, build.BuildError, OSError, ValueError)
      ):
        with contextlib.suppress(OSError):
          (work_dir / "kit-error.log").write_text(
              traceback.format_exc(), encoding="utf-8", errors="replace"
          )
          detail += (
              " (a fault of the kit's own; its traceback is in"
              f" {work_dir / 'kit-error.log'})"
          )
      report.add("kit", "fail", detail)
    except KeyboardInterrupt:
      interrupted, report = True, checker.report
      report.add(
          "kit",
          "fail",
          "interrupted"
          + ("" if checker.keep else "; its containers are being removed"),
      )
    finally:
      # whatever an interrupt left between a start and its teardown
      checker.abort()
  removal = checker.kept()
  if removal:
    print(
        "kept for inspection, and no sweep removes them: remove them with"
        f" `{removal}`"
    )
  ready = (
      report.ok
      and not report.untested
      and not report.smoke_skipped
      and not report.unready
  )
  (work_dir / "check.json").write_text(
      json.dumps(
          {
              **dataclasses.asdict(report),
              "untested": report.untested,
              "ready": ready,
          },
          indent=1,
      )
  )
  if not report.ok:
    verdict = "not ready: fix the FAIL lines above"
  elif report.untested:
    verdict = f"not verified: you asked for {'; '.join(report.untested)}"
  elif report.unready:
    # what was asked for passed (exit 0), but the entry is not ready to submit
    verdict = "the stages that ran pass, but " + UNREADY[report.unready]
  elif report.smoke_skipped:
    # what was asked for passed (exit 0), but the server runs the smoke task
    # before anything else
    verdict = (
        "the stages that ran pass; the smoke task was skipped (--no-smoke), so"
        " run kit check without it before you submit"
    )
  else:
    verdict = "ready to submit"
  print(f"\n{verdict} (logs in {work_dir})")
  if interrupted:
    return 130
  return 0 if report.ok and not report.untested else 1
