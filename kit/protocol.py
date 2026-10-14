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

"""Serve protocol v1 limits and frame validation checks."""

from __future__ import annotations

import base64
import binascii
import functools
import json
import math
import re
from typing import Any
import urllib.parse

VERSION = 1
PORT = 8765
START_SEC = 120
MAX_FRAME_BYTES = 16 << 20
MAX_MALFORMED = 20
MAX_IN_FLIGHT = 8
CANCEL_GRACE_SEC = 10
DEFAULT_EXEC_TIMEOUT_SEC = 120
EXEC_TIMEOUT_SEC = 600
MAX_OUTPUT_BYTES = 100_000
MAX_READ_BYTES = MAX_OUTPUT_BYTES * 100
MAX_HTTP_BYTES = MAX_READ_BYTES
MAX_WRITE_BYTES = 10_000_000
# what the evaluator reads back from one docker exec (a 10 MB read is ~13.4 MB
MAX_EXEC_READ_BYTES = 32 << 20
# of base64): past it the command is stopped
MAX_ID_CHARS = 128
MAX_LOG_CHARS = 20_000
MAX_MODELS = 16
# the largest usage count: past it, JSON readers that parse doubles lose
# precision
MAX_COUNT = 2**53 - 1

REQUESTS = ("exec", "read_file", "write_file", "http")
NOTES = ("log", "usage")
ENDS = ("done", "error")
HTTP_METHODS = ("GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS")
# Framing and routing headers, which the task container's HTTP client sets
# itself or leaves out, so that a harness cannot frame a second request inside
# the body or send it elsewhere. They are dropped, with any Proxy-* header, and
# the request still runs.
HOP_HEADERS = frozenset({
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "te",
    "trailer",
})


class ProtocolError(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when a protocol frame from the harness is malformed.

  `rid` holds the request ID when a result frame can return it
  (`answerable_id`), ensuring that a frame with an unknown type or invalid field
  still receives a single error response.
  """

  def __init__(self, message: str, rid: str | None = None):
    """Initializes ProtocolError with an error message and optional request ID.

    Args:
      message: Description of the protocol violation.
      rid: Optional answerable request ID associated with the frame.
    """
    super().__init__(message)
    self.rid = rid


def harness_input(check):
  """Wraps a frame validation function so unexpected errors raise ProtocolError.

  Any exception raised by `check` other than `ProtocolError` or `MemoryError` is
  converted into a `ProtocolError` naming the exception type and carrying the
  frame's answerable ID, attributing unforeseen malformed inputs to the harness
  rather than the evaluator.

  Args:
    check: A pure validation function accepting a frame message and arguments.

  Returns:
    The wrapped validation function.
  """

  @functools.wraps(check)
  def checked(message, *args):
    try:
      return check(message, *args)
    except (ProtocolError, MemoryError):
      raise
    except Exception as exc:  # noqa: BLE001 - see the docstring
      rid = (
          answerable_id(message.get("id"))
          if isinstance(message, dict)
          else None
      )
      raise ProtocolError(
          f"malformed frame ({type(exc).__name__})", rid
      ) from None

  return checked


def answerable_id(rid) -> str | None:
  """Returns `rid` if it is a valid UTF-8 request ID within length limits.

  Args:
    rid: Candidate request ID from a frame.

  Returns:
    The request ID string if it is non-empty, at most `MAX_ID_CHARS` characters,
    and valid UTF-8; otherwise None.
  """
  if not isinstance(rid, str) or not rid or len(rid) > MAX_ID_CHARS:
    return None
  try:
    rid.encode("utf-8")
  except UnicodeEncodeError:
    return None
  return rid


def task_message(
    *,
    session: str,
    instruction: str,
    workdir: str,
    agent_timeout_sec: float | None,
    mcp_servers: list[dict[str, Any]],
) -> dict[str, Any]:
  """Builds the initial `task` frame sent to the harness.

  Time limits are rounded up to whole seconds with a minimum of 1 (since 0 would
  be read as no timeout).

  Args:
    session: Unique session identifier.
    instruction: Task instruction text for the agent.
    workdir: Working directory path inside the task container.
    agent_timeout_sec: Optional overall timeout in seconds for the agent phase.
    mcp_servers: List of MCP server descriptors available to the task.

  Returns:
    A dictionary representing the initial `task` protocol frame.
  """
  return {
      "type": "task",
      "protocol": VERSION,
      "session": session,
      "instruction": instruction,
      "workdir": workdir,
      "limits": {
          "agent_timeout_sec": (
              None
              if agent_timeout_sec is None
              else max(1, math.ceil(agent_timeout_sec))
          ),
          "exec_timeout_sec": EXEC_TIMEOUT_SEC,
          "max_output_bytes": MAX_OUTPUT_BYTES,
          "max_write_bytes": MAX_WRITE_BYTES,
      },
      "mcp_servers": mcp_servers,
  }


def _no_constant(name: str):
  """Rejects non-standard JSON constants such as NaN or Infinity.

  Args:
    name: The constant token encountered by `json.loads`.

  Raises:
    ProtocolError: Always raised because non-standard JSON constants are
      disallowed.
  """
  raise ProtocolError(f"{name} is not valid JSON")


@harness_input
def parse_frame(frame) -> dict[str, Any]:
  """Parses a raw WebSocket frame string into a validated JSON object.

  Args:
    frame: Raw WebSocket frame payload received from the harness.

  Returns:
    The parsed JSON object as a dictionary.

  Raises:
    ProtocolError: If `frame` is not a string, is invalid JSON, is not a JSON
      object, or has an unknown `type`.
  """
  if not isinstance(frame, str):
    raise ProtocolError("frames are text, one JSON object each")
  try:
    message = json.loads(frame, parse_constant=_no_constant)
  except RecursionError:
    raise ProtocolError("the JSON nests too deeply") from None
  except ValueError as exc:
    raise ProtocolError(f"not JSON: {exc}") from None
  if not isinstance(message, dict):
    raise ProtocolError("a frame holds one JSON object")
  if message.get("type") not in REQUESTS + NOTES + ENDS:
    raise ProtocolError(
        f"unknown type {message.get('type')!r}",
        answerable_id(message.get("id")),
    )
  return message


def _number(value) -> bool:
  """Returns whether `value` is a JSON number (int or finite float, not bool).

  Integers are checked without float conversion so large integers beyond float
  range are still handled safely.

  Args:
    value: Value to check.

  Returns:
    True if `value` is an int (excluding bool) or a finite float.
  """
  return not isinstance(value, bool) and (
      isinstance(value, int)
      or isinstance(value, float)
      and math.isfinite(value)
  )


def _whole(value) -> bool:
  """Returns whether `value` is a JSON number with no fractional part.

  Args:
    value: Value to check.

  Returns:
    True if `value` is a JSON number representing an integer value (e.g., 1 or
    1.0, but not 1.5).
  """
  return _number(value) and (isinstance(value, int) or value.is_integer())


def _int(
    message: dict[str, Any], key: str, default: int, low: int, high: int
) -> int:
  """Extracts a numeric field from `message` clamped to `[low, high]`.

  Args:
    message: Frame dictionary.
    key: Field name to look up.
    default: Default value when the field is absent or None.
    low: Minimum allowed integer value.
    high: Maximum allowed integer value.

  Returns:
    The clamped integer value.

  Raises:
    ProtocolError: If the field value is neither None nor a JSON number.
  """
  value = message.get(key, default)
  if value is None:
    return default
  if not _number(value):
    raise ProtocolError(f"{key}: a number")
  return min(max(int(value), low), high)


def _clean(key: str, value: str) -> str:
  """Validates that `value` contains no NUL bytes or unpaired surrogates.

  Args:
    key: Field name for error reporting.
    value: String value to validate.

  Returns:
    The validated `value` string.

  Raises:
    ProtocolError: If `value` contains a NUL byte or cannot be encoded as UTF-8.
  """
  if "\x00" in value:
    raise ProtocolError(f"{key}: contains a NUL byte")
  try:
    value.encode("utf-8")
  except UnicodeEncodeError:
    raise ProtocolError(
        f"{key}: not valid UTF-8 (an unpaired surrogate)"
    ) from None
  return value


def _text(message: dict[str, Any], key: str) -> str:
  """Extracts a required non-empty, UTF-8 clean string field from `message`.

  Args:
    message: Frame dictionary.
    key: Field name to extract.

  Returns:
    The validated string value.

  Raises:
    ProtocolError: If the field is missing, empty, not a string, or invalid
      UTF-8.
  """
  value = message.get(key)
  if not isinstance(value, str) or not value:
    raise ProtocolError(f"{key}: a non-empty string")
  return _clean(key, value)


def _b64(
    message: dict[str, Any], key: str, cap: int, required: bool = False
) -> bytes:
  """Decodes a base64 field from `message` within a maximum decoded byte size.

  A required field must be present (`""` represents an empty payload); an
  optional field that is missing or null decodes to empty bytes.

  Args:
    message: Frame dictionary.
    key: Field name holding base64 text.
    cap: Maximum allowed decoded length in bytes.
    required: Whether the field must be explicitly provided.

  Returns:
    The decoded bytes.

  Raises:
    ProtocolError: If the field is missing when required, not valid base64, or
      exceeds `cap` bytes.
  """
  value = message.get(key)
  if value is None:
    if required:
      raise ProtocolError(
          f'{key}: required, base64 text ("" for an empty file)'
      )
    value = ""
  # cap's base64 length, before decoding
  if not isinstance(value, str) or len(value) > (cap + 2) // 3 * 4 + 4:
    raise ProtocolError(f"{key}: base64 text of at most {cap} bytes")
  try:
    data = base64.b64decode(value, validate=True)
  except (binascii.Error, ValueError):
    raise ProtocolError(f"{key}: not valid base64") from None
  if len(data) > cap:
    raise ProtocolError(f"{key}: more than {cap} bytes")
  return data


@harness_input
def check_request(message: dict[str, Any]) -> dict[str, Any]:
  """Validates and normalizes a request frame from the harness.

  Optional fields set to null are treated the same as omitted fields.

  Args:
    message: Parsed request frame dictionary.

  Returns:
    A dictionary with normalized request fields.

  Raises:
    ProtocolError: If the request ID or any request field is invalid.
  """
  # a request whose id no result could carry back is not run
  rid = answerable_id(message.get("id"))
  if rid is None:
    raise ProtocolError(
        f"id: a non-empty string of at most {MAX_ID_CHARS} characters, valid"
        " UTF-8"
    )
  try:
    return _fields(message, rid)
  except ProtocolError as exc:
    raise ProtocolError(str(exc), rid) from None


def _fields(message: dict[str, Any], rid: str) -> dict[str, Any]:
  """Extracts and validates type-specific fields for a request frame.

  Args:
    message: Parsed request frame dictionary.
    rid: Validated request ID.

  Returns:
    Normalized request dictionary ready for execution.

  Raises:
    ProtocolError: If any field for the request type is invalid.
  """
  kind = message["type"]
  out: dict[str, Any] = {"type": kind, "id": rid}
  if kind == "exec":
    out["command"] = _text(message, "command")
    out["timeout_sec"] = _int(
        message, "timeout_sec", DEFAULT_EXEC_TIMEOUT_SEC, 1, EXEC_TIMEOUT_SEC
    )
  elif kind == "read_file":
    out["path"] = _text(message, "path")
  elif kind == "write_file":
    out["path"] = _text(message, "path")
    out["content"] = _b64(
        message, "content_b64", MAX_WRITE_BYTES, required=True
    )
    append = message.get("append")
    if append is not None and not isinstance(append, bool):
      raise ProtocolError("append: true or false")
    out["append"] = bool(append)
  else:
    method = str(message.get("method") or "GET").upper()
    if method not in HTTP_METHODS:
      raise ProtocolError(f"method: one of {', '.join(HTTP_METHODS)}")
    headers = message.get("headers") or {}
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
      raise ProtocolError("headers: an object of string values")
    for k, v in headers.items():
      _clean("headers", k + v)
    headers = {
        k: v
        for k, v in headers.items()
        if k.strip().lower() not in HOP_HEADERS
        and not k.strip().lower().startswith("proxy-")
    }
    out.update(
        method=method,
        url=_text(message, "url"),
        headers=headers,
        body=_b64(message, "body_b64", MAX_HTTP_BYTES),
        timeout_sec=_int(
            message,
            "timeout_sec",
            DEFAULT_EXEC_TIMEOUT_SEC,
            1,
            EXEC_TIMEOUT_SEC,
        ),
    )
  return out


@harness_input
def check_usage(message: dict[str, Any]) -> dict[str, Any]:
  """Validates and normalizes a `usage` note frame.

  Missing or null token counts default to 0; provided counts must be whole
  numbers in `[0, MAX_COUNT]`, and `model`, if provided, must be a string
  (truncated to 200 characters).

  Args:
    message: Parsed `usage` frame dictionary.

  Returns:
    Normalized usage dictionary with `model` and token counts.

  Raises:
    ProtocolError: If `model` is not a string or any token count is out of
      bounds.
  """
  rid = answerable_id(message.get("id"))
  model = message.get("model")
  if model is not None and not isinstance(model, str):
    raise ProtocolError("model: a string", rid)
  out = {"model": (model or "")[:200]}
  for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
    value = message.get(key)
    if value is None:
      value = 0
    if not _whole(value) or not 0 <= value <= MAX_COUNT:
      raise ProtocolError(f"{key}: a whole number from 0 to {MAX_COUNT}", rid)
    out[key] = int(value)
  return out


def error_result(rid: str, error: str) -> dict[str, Any]:
  """Builds an error `result` frame for request `rid`.

  Args:
    rid: Request ID being answered.
    error: Error message describing why the request failed.

  Returns:
    A dictionary representing the error `result` frame.
  """
  return {"type": "result", "id": rid, "ok": False, "error": error}


@harness_input
def mcp_allows(url: str, servers: list[dict[str, Any]]) -> bool:
  """Returns whether `url` is permitted by the task's configured MCP servers.

  A URL is allowed when it matches one of the task's MCP server URLs exactly,
  adds a query string to a server URL that has none (`http://h/mcp?session=1`),
  or accesses a sub-path under one. Paths that could traverse upward after
  decoding (`.`, `..`, `%2f`, `%5c`, backslashes, or `;` path parameters) are
  rejected.

  Args:
    url: Candidate HTTP URL requested by the harness.
    servers: List of MCP server dictionaries from the task configuration.

  Returns:
    True if `url` is allowed for the task's MCP servers, False otherwise.
  """
  try:
    path = urllib.parse.urlsplit(url).path
  # an unparseable URL (an unclosed IPv6 bracket) reaches nothing
  except ValueError:
    return False
  decoded = urllib.parse.unquote(path)
  if (
      re.search(r"%2f|%5c|\\", path, re.IGNORECASE)
      or ";" in decoded
      or {".", ".."} & set(decoded.split("/"))
  ):
    return False
  for server in servers:
    base = str(server.get("url") or "")
    if base and (
        url == base
        or ("?" not in base and url.startswith(base + "?"))
        or url.startswith(base.rstrip("/") + "/")
    ):
      return True
  return False
