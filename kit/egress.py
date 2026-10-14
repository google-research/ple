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

"""Egress proxy and port forwarder for a serve container.

Runs in its own container with this file mounted:

    python3 -I /egress.py --allow api.example.com:443,...
        [--except host:port] --forward serve:8765

Acts as an HTTP proxy (`CONNECT` and absolute-URI requests) that connects only
to allowed `host:port` pairs and only to addresses resolving as public IPs.
`--except` specifies private `host:port` endpoints that are permitted, such as
the evaluator's model server. Request heads (including `CONNECT`) are rejected
if a server could interpret them ambiguously (bare CR or LF, control characters,
folded or malformed header lines). Plain HTTP requests are routed to the host
in the request URL with `Host` rewritten from the URL, unambiguous body framing,
and strict validation of chunk-size lines and trailers. Outbound connections
idle for `IDLE_SEC` seconds in both directions are closed.

Also forwards a published port to the serve container's WebSocket using separate
connection slots. Accepting connections only when a slot is available bounds
proxy memory regardless of how many connections arrive. Connection events are
logged as JSON lines without payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import re
import socket
import sys
import time
from typing import Any
import urllib.parse

MAX_HEAD_BYTES = 64 << 10
HEAD_TIMEOUT_SEC = 30
CONNECT_TIMEOUT_SEC = 20
MAX_CONNECTIONS = 256  # the serve container's own, outbound
# the evaluator's into the serve container: one per task, and its retries
MAX_FORWARDS = 32
# connections waiting for a slot: the kernel holds them, and nothing reads them
BACKLOG = 64
# Close an outbound connection after this long with no bytes either way, as the
# evaluation server does. It equals the model server's own request timeout (1800
# s), so a call silent for that long fails anyway: stream long calls.
IDLE_SEC = 1800
CHUNK = 1 << 16
VERSIONS = (b"HTTP/1.0", b"HTTP/1.1")
_TOKEN = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_TARGET = re.compile(rb"[\x21-\x7e]+")
# control characters a header line may not hold (HTAB may)
_CTL = re.compile(rb"[\x00-\x08\x0a-\x1f\x7f]")
_CHUNK_LINE = re.compile(
    rb"([0-9A-Fa-f]{1,16})(;[^\x00-\x08\x0a-\x1f\x7f]*)?\r\n"
)
# The client's connection and proxy headers, and Host, which the proxy sets from
# the URL itself.
DROPPED_HEADERS = {
    b"host",
    b"connection",
    b"keep-alive",
    b"proxy-connection",
    b"proxy-authorization",
}
# Refused in a chunked body's trailers: a server that merges trailers into the
# headers could frame or route the request by them.
FORBIDDEN_TRAILERS = DROPPED_HEADERS | {
    b"content-length",
    b"transfer-encoding",
    b"te",
    b"trailer",
    b"upgrade",
}


def normal_host(host: str) -> str:
  """Normalizes a hostname to lowercase without trailing dots.

  Used consistently across the allowlist and upstream `Host` headers for both
  `CONNECT` and plain HTTP requests (e.g., `api.example.com.` becomes
  `api.example.com`).

  Args:
    host: Raw hostname string.

  Returns:
    Normalized lowercase hostname without a trailing dot.
  """
  return host.strip().lower().rstrip(".")


def host_port(text: str, default_port: int) -> tuple[str, int]:
  """Parses a `host[:port]` or `[ipv6][:port]` string into `(host, port)`.

  Args:
    text: Host and optional port specification.
    default_port: Port number to use when `text` omits a port.

  Returns:
    A tuple of `(normalized_host, port)`.
  """
  text = text.strip().lower()
  if text.startswith("["):  # [v6]:port
    host, _, rest = text[1:].partition("]")
    port = rest[1:] if rest.startswith(":") else ""
  elif text.count(":") == 1:
    host, _, port = text.partition(":")
  else:
    host, port = text, ""
  return normal_host(host), int(port) if port else default_port


def is_public(address: str) -> bool:
  """Returns whether `address` is a globally routable, non-multicast IP.

  Args:
    address: IPv4 or IPv6 address literal.

  Returns:
    True if `address` is global and not multicast, False otherwise.
  """
  ip = ipaddress.ip_address(address)
  if ip.version == 6 and ip.ipv4_mapped is not None:
    ip = ip.ipv4_mapped
  return ip.is_global and not ip.is_multicast


class Policy:
  """Determines which `host:port` targets are allowed and resolves their IPs."""

  def __init__(self, allow, exceptions=(), *, resolver=None, public=is_public):
    """Initializes Policy with allowed endpoints, private exceptions, and DNS.

    Args:
      allow: Iterable of allowed public `host[:port]` strings (default port
        443).
      exceptions: Iterable of allowed private `host[:port]` strings (default
        port 80).
      resolver: Optional async `(host, port) -> Iterable[str]` DNS resolver.
      public: Predicate checking if an IP address string is public.
    """
    self.allow = {host_port(a, 443) for a in allow}
    self.exceptions = {host_port(e, 80) for e in exceptions}
    self._resolver = resolver
    self._public = public

  async def _resolve(self, host: str, port: int) -> list[str]:
    """Resolves `host` and `port` into a deduplicated list of IP addresses.

    Args:
      host: Hostname to resolve.
      port: Destination port number.

    Returns:
      List of resolved IP address strings in resolution order.
    """
    if self._resolver is not None:
      return list(await self._resolver(host, port))
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    )
    return list(dict.fromkeys(info[4][0] for info in infos))

  async def decide(self, host: str, port: int) -> tuple[list[str], str]:
    """Decides whether a connection to `host:port` is permitted.

    Args:
      host: Normalized target hostname.
      port: Target port number.

    Returns:
      A tuple `(addresses, reason)` where `addresses` is the list of IP
      addresses or hostnames to connect to (empty when refused) and `reason`
      explains the decision.
    """
    if (host, port) in self.exceptions:
      return [host], "the evaluator's model server"
    if (host, port) not in self.allow:
      return [], f"{host}:{port} is not the model's endpoint"
    try:
      addresses = await self._resolve(host, port)
    except (OSError, UnicodeError) as exc:
      return [], f"{host} does not resolve ({exc})"
    if not addresses:
      return [], f"{host} does not resolve"
    private = [a for a in addresses if not self._public(a)]
    if private:
      return [], f"{host} resolves to a non-public address ({private[0]})"
    return addresses, "declared"


def log(**fields) -> None:
  """Writes a timestamped JSON event line to stdout.

  Args:
    **fields: Key-value pairs to include in the JSON log object.
  """
  print(json.dumps({"t": round(time.time(), 3), **fields}), flush=True)


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    counter: list[int],
    activity: list[float] | None = None,
) -> None:
  """Copies bytes from `reader` to `writer` until EOF and half-closes `writer`.

  Args:
    reader: Source stream reader.
    writer: Destination stream writer.
    counter: Single-element list accumulating total bytes transferred.
    activity: Optional single-element list updated with `time.monotonic()` on
      each chunk.
  """
  try:
    while True:
      chunk = await reader.read(CHUNK)
      if not chunk:
        break
      counter[0] += len(chunk)
      if activity is not None:
        activity[0] = time.monotonic()
      writer.write(chunk)
      await writer.drain()
  except (OSError, asyncio.IncompleteReadError):
    pass
  finally:
    try:
      if writer.can_write_eof():
        writer.write_eof()
    except OSError:
      pass


async def until_idle(
    copies: list[asyncio.Future[Any]],
    activity: list[float],
    idle: float | None,
) -> bool:
  """Waits for `copies` to finish, cancelling them if idle for `idle` seconds.

  Args:
    copies: Futures performing stream copy operations that update `activity`.
    activity: Single-element list holding the last activity monotonic timestamp.
    idle: Optional idle timeout in seconds, or None for no timeout.

  Returns:
    True if `copies` were cancelled due to reaching the idle timeout, False if
    they completed normally.
  """
  pending = set(copies)
  while pending:
    timeout = (
        None
        if idle is None
        else max(0.0, activity[0] + idle - time.monotonic())
    )
    _, pending = await asyncio.wait(pending, timeout=timeout)
    if pending and idle is not None and time.monotonic() - activity[0] >= idle:
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
      return True
  return False


async def splice(
    a_reader, a_writer, b_reader, b_writer, idle: float | None = None
) -> tuple[int, int, bool]:
  """Copies data bidirectionally between two streams until completion or idle.

  Args:
    a_reader: Reader for side A.
    a_writer: Writer for side A.
    b_reader: Reader for side B.
    b_writer: Writer for side B.
    idle: Optional idle timeout in seconds across both directions.

  Returns:
    A tuple `(bytes_a_to_b, bytes_b_to_a, went_idle)`.
  """
  up, down, activity = [0], [0], [time.monotonic()]
  copies = [
      asyncio.ensure_future(_pipe(a_reader, b_writer, up, activity)),
      asyncio.ensure_future(_pipe(b_reader, a_writer, down, activity)),
  ]
  idled = await until_idle(copies, activity, idle)
  for w in (a_writer, b_writer):
    w.close()
  return up[0], down[0], idled


async def _read_head(reader: asyncio.StreamReader) -> bytes:
  r"""Reads an HTTP request head up to `\r\n\r\n` within timeout/size limits.

  Args:
    reader: Client stream reader.

  Returns:
    The raw request head bytes including the trailing `\r\n\r\n`.

  Raises:
    ValueError: If the request head exceeds `MAX_HEAD_BYTES`.
  """
  head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_SEC)
  if len(head) > MAX_HEAD_BYTES:
    raise ValueError("request head too large")
  return head


def request_line(line: bytes) -> tuple[str, str, str]:
  """Parses and validates an HTTP request line into `(method, target, version)`.

  Rejects lines that upstream servers could interpret differently, such as
  control characters, non-token methods, non-printable ASCII targets, or HTTP
  versions other than `HTTP/1.0` and `HTTP/1.1`.

  Args:
    line: Raw request line bytes without the trailing CRLF.

  Returns:
    A tuple of `(method, target, version)` decoded as ASCII strings.

  Raises:
    ValueError: If the request line is malformed.
  """
  parts = line.split(b" ")
  if len(parts) != 3:
    raise ValueError("malformed request line")
  method, target, version = parts
  if (
      not _TOKEN.fullmatch(method)
      or not _TARGET.fullmatch(target)
      or version not in VERSIONS
  ):
    raise ValueError("malformed request line")
  return method.decode("ascii"), target.decode("ascii"), version.decode("ascii")


def header_fields(rest: bytes) -> list[tuple[bytes, bytes, bytes]]:
  """Parses HTTP header or trailer lines into `(raw_line, lower_name, value)`.

  Rejects lines containing bare CR/LF, NUL or other control characters (except
  HTAB), or non-token header names (including obsolete folded lines starting
  with whitespace).

  Args:
    rest: Raw header block bytes separated by CRLF.

  Returns:
    A list of `(raw_line, lowercase_name, stripped_value)` tuples.

  Raises:
    ValueError: If any header line is malformed.
  """
  fields = []
  for n, line in enumerate((h for h in rest.split(b"\r\n") if h), 1):
    name, colon, value = line.partition(b":")
    if _CTL.search(line) or not colon or not _TOKEN.fullmatch(name):
      # name the line by number only: a header value can hold an API key, and
      # the log keeps no payloads
      raise ValueError(
          f"malformed header line {n} (a bare CR or LF, a control character, or"
          " no name)"
      )
    fields.append((line, name.lower(), value.strip(b" \t")))
  return fields


def upstream_head(
    method: str, path: str, version: str, fields, host: str, port: int
) -> tuple[bytes, int]:
  """Builds the sanitized HTTP request head for an upstream plain HTTP request.

  Sets `Host` from the parsed URL rather than trusting the client's `Host`
  header, strips hop-by-hop and proxy headers, and validates `Content-Length`
  and `Transfer-Encoding` to prevent request smuggling.

  Args:
    method: HTTP method string.
    path: Request path and optional query string.
    version: HTTP version string (`HTTP/1.0` or `HTTP/1.1`).
    fields: Parsed header tuples from `header_fields`.
    host: Normalized target hostname.
    port: Target port number.

  Returns:
    A tuple `(raw_head_bytes, body_length)`, where `body_length` is `-1` for
    chunked transfer encoding or the non-negative byte length otherwise.

  Raises:
    ValueError: If `Content-Length` or `Transfer-Encoding` is ambiguous or
      invalid.
  """
  kept, length, chunked = [], None, False
  for line, key, value in fields:
    if key == b"content-length":
      if not value.isdigit() or (length is not None and int(value) != length):
        raise ValueError("an unclear Content-Length")
      length = int(value)
    elif key == b"transfer-encoding":
      if [c.strip(b" \t").lower() for c in value.split(b",")][-1] != b"chunked":
        raise ValueError("a Transfer-Encoding that does not end in chunked")
      chunked = True
    if key not in DROPPED_HEADERS:
      kept.append(line)
  if chunked and length is not None:
    raise ValueError("both Content-Length and Transfer-Encoding")
  netloc = (f"[{host}]" if ":" in host else host) + (
      "" if port == 80 else f":{port}"
  )
  head = [
      f"{method} {path} {version}".encode("ascii"),
      f"Host: {netloc}".encode("ascii"),
      *kept,
      b"Connection: close",
      b"",
      b"",
  ]
  return b"\r\n".join(head), -1 if chunked else length or 0


async def copy_body(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    length: int,
    counter: list[int],
    activity: list[float] | None = None,
) -> None:
  """Copies a single HTTP request body of `length` bytes or chunked encoding.

  When `length` is `-1`, validates each chunk-size line and trailer field,
  enforcing `MAX_HEAD_BYTES` on total trailer size and rejecting forbidden
  trailers. Stops reading after the body completes so a second pipelined request
  cannot be smuggled.

  Args:
    reader: Client stream reader.
    writer: Upstream stream writer.
    length: Non-negative content length in bytes, or `-1` for chunked encoding.
    counter: Single-element list accumulating total body bytes sent.
    activity: Optional single-element list updated with `time.monotonic()` on
      each write.

  Raises:
    ValueError: If a chunk size line, chunk terminator, or trailer is malformed.
    asyncio.IncompleteReadError: If the client stream ends before the body
      completes.
  """

  def sent(data: bytes) -> None:
    counter[0] += len(data)
    if activity is not None:
      activity[0] = time.monotonic()
    writer.write(data)

  async def copy(n: int) -> None:
    while n > 0:
      chunk = await reader.read(min(n, CHUNK))
      if not chunk:
        raise asyncio.IncompleteReadError(b"", n)
      sent(chunk)
      await writer.drain()
      n -= len(chunk)

  if length >= 0:
    await copy(length)
    return
  while True:
    # the chunk's size in hex, maybe ;extensions
    line = await reader.readuntil(b"\r\n")
    match = _CHUNK_LINE.fullmatch(line)
    if match is None:
      raise ValueError("a malformed chunk size line")
    sent(line)
    size = int(match.group(1), 16)
    if size == 0:
      trailers = 0
      while (trailer := await reader.readuntil(b"\r\n")) != b"\r\n":
        trailers += len(trailer)
        if trailers > MAX_HEAD_BYTES:
          raise ValueError(f"trailers over {MAX_HEAD_BYTES} bytes")
        if any(
            key in FORBIDDEN_TRAILERS
            for _, key, _ in header_fields(trailer[:-2])
        ):
          raise ValueError("a trailer that would frame or route the request")
        sent(trailer)
        # a host that stops reading holds the next one back
        await writer.drain()
      sent(b"\r\n")
      await writer.drain()
      return
    await copy(size)
    # every chunk's data ends in exactly CRLF
    if await reader.readexactly(2) != b"\r\n":
      raise ValueError("a chunk not ended by CRLF")
    sent(b"\r\n")


async def one_request(
    c_reader,
    c_writer,
    u_reader,
    u_writer,
    length: int,
    idle: float | None = None,
) -> tuple[int, int, bool]:
  """Forwards a plain HTTP request body upstream and streams the response back.

  Runs until the upstream server closes the connection (having received
  `Connection: close`) or until the connection is idle for `idle` seconds.

  Args:
    c_reader: Client stream reader.
    c_writer: Client stream writer.
    u_reader: Upstream stream reader.
    u_writer: Upstream stream writer.
    length: Request body length in bytes, or `-1` for chunked encoding.
    idle: Optional idle timeout in seconds.

  Returns:
    A tuple `(bytes_up, bytes_down, went_idle)`.
  """
  up, down, activity = [0], [0], [time.monotonic()]

  async def body() -> None:
    try:
      await copy_body(c_reader, u_writer, length, up, activity)
    except (
        OSError,
        ValueError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
    ) as exc:
      log(
          kind="cut",
          reason=(
              f"the request body went no further: {type(exc).__name__}: {exc}"
          ),
      )
      u_writer.close()

  sending = asyncio.ensure_future(body())
  answer = asyncio.ensure_future(_pipe(u_reader, c_writer, down, activity))
  try:
    idled = await until_idle([answer], activity, idle)
  finally:
    for task in (sending, answer):
      task.cancel()
    await asyncio.gather(sending, answer, return_exceptions=True)
    for w in (c_writer, u_writer):
      w.close()
  return up[0], down[0], idled


def _refuse(writer: asyncio.StreamWriter, status: str, reason: str) -> None:
  """Writes a plain-text HTTP error response to `writer`.

  Args:
    writer: Client stream writer.
    status: HTTP status code and reason phrase (e.g., `'403 Forbidden'`).
    reason: Error message body line.
  """
  body = (reason + "\n").encode()
  writer.write(
      f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length:"
      f" {len(body)}\r\nConnection: close\r\n\r\n".encode()
      + body
  )


def bound(port: int, host: str = "0.0.0.0") -> socket.socket:
  """Creates and returns a non-blocking listening socket bound to `host:port`.

  Args:
    port: Local port number to bind.
    host: Local interface address to bind (defaults to `'0.0.0.0'`).

  Returns:
    The bound non-blocking `socket.socket` instance.
  """
  sock = socket.create_server((host, port), backlog=BACKLOG)
  sock.setblocking(False)
  return sock


async def listen(
    sock: socket.socket, handler, slots: asyncio.Semaphore, limit: int = 1 << 16
) -> None:
  """Accepts connections on `sock` gated by `slots` and dispatches `handler`.

  Acquires a slot from `slots` before calling `sock_accept` so excess
  connections remain unread in the kernel's listen backlog, bounding proxy
  memory by the number of slots rather than incoming connection volume.

  Args:
    sock: Bound non-blocking listening socket.
    handler: Async `(reader, writer)` callback for each accepted connection.
    slots: Semaphore limiting concurrent active connections.
    limit: StreamReader buffer limit in bytes.
  """
  loop = asyncio.get_running_loop()
  running: set[asyncio.Task[Any]] = set()

  async def one(conn: socket.socket) -> None:
    try:
      try:
        reader, writer = await asyncio.open_connection(sock=conn, limit=limit)
      except OSError:
        conn.close()
        return
      await handler(reader, writer)
    finally:
      slots.release()

  while True:
    await slots.acquire()
    try:
      conn, _ = await loop.sock_accept(sock)
    # out of file descriptors, say: wait and accept again
    except OSError as exc:
      slots.release()
      log(kind="error", error=f"accept: {exc}")
      await asyncio.sleep(0.5)
      continue
    except BaseException:
      slots.release()
      raise
    task = asyncio.ensure_future(one(conn))
    running.add(task)
    task.add_done_callback(running.discard)


class Proxy:
  """HTTP egress proxy and inbound WebSocket port forwarder."""

  def __init__(
      self,
      policy: Policy,
      forward: tuple[str, int] | None = None,
      idle: float | None = IDLE_SEC,
  ):
    """Initializes the proxy with an egress policy and optional port forwarder.

    Args:
      policy: `Policy` instance governing outbound connections.
      forward: Optional `(host, port)` target for inbound WebSocket forwarding.
      idle: Optional timeout in seconds for idle outbound connections.
    """
    self.policy = policy
    self.forward = forward
    self.idle = idle
    self.slots = asyncio.Semaphore(MAX_CONNECTIONS)
    # slots of their own, so a pool of idle tunnels from the serve container
    # cannot block the evaluator
    self.forward_slots = asyncio.Semaphore(MAX_FORWARDS)

  async def handle_proxy(
      self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
  ) -> None:
    """Handles a single outbound proxy connection from the serve container.

    Args:
      reader: Client stream reader.
      writer: Client stream writer.
    """
    try:
      await self._proxy(reader, writer)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log(kind="error", error=f"{type(exc).__name__}: {exc}")
    finally:
      writer.close()

  async def _proxy(self, reader, writer) -> None:
    """Parses, validates, and relays an outbound HTTP or CONNECT request.

    Args:
      reader: Client stream reader.
      writer: Client stream writer.
    """
    try:
      head = await _read_head(reader)
    except (
        asyncio.TimeoutError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        ValueError,
    ):
      _refuse(writer, "400 Bad Request", "no complete request head")
      return
    line, _, rest = head.partition(b"\r\n")
    try:
      method, target, version = request_line(line)
      fields = header_fields(rest)
    except ValueError as exc:
      log(kind="refused", reason=str(exc))
      _refuse(writer, "400 Bad Request", f"egress refused: {exc}")
      return
    if method.upper() == "CONNECT":
      try:
        host, port = host_port(target, 443)
      except ValueError:
        _refuse(writer, "400 Bad Request", "CONNECT needs host:port")
        return
      path = None
    else:
      try:
        url = urllib.parse.urlsplit(target)
        host, port = normal_host(url.hostname or ""), url.port or 80
      except ValueError:
        url = host = None
      if url is None or url.scheme != "http" or not host:
        _refuse(
            writer,
            "400 Bad Request",
            "send https through CONNECT; plain requests need an http:// URL",
        )
        return
      path = (url.path or "/") + (f"?{url.query}" if url.query else "")
      try:
        up_head, length = upstream_head(
            method, path, version, fields, host, port
        )
      except ValueError as exc:
        log(
            kind="refused", method=method, host=host, port=port, reason=str(exc)
        )
        _refuse(writer, "400 Bad Request", f"egress refused: {exc}")
        return
    addresses, reason = await self.policy.decide(host, port)
    if not addresses:
      log(kind="refused", method=method, host=host, port=port, reason=reason)
      _refuse(writer, "403 Forbidden", f"egress refused: {reason}")
      return
    for address in addresses:
      try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(address, port), CONNECT_TIMEOUT_SEC
        )
        break
      except (OSError, asyncio.TimeoutError) as exc:
        log(
            kind="failed",
            method=method,
            host=host,
            port=port,
            address=address,
            error=str(exc),
        )
    else:
      _refuse(writer, "502 Bad Gateway", f"could not connect to {host}:{port}")
      return
    log(
        kind="allowed",
        method=method,
        host=host,
        port=port,
        address=address,
        reason=reason,
    )
    if path is None:
      # a tunnel's bytes (TLS, usually) are not inspected: the CONNECT host is
      # all the proxy checks
      writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
      await writer.drain()
      sent, received, idled = await splice(
          reader, writer, up_reader, up_writer, self.idle
      )
    else:
      up_writer.write(up_head)
      sent, received, idled = await one_request(
          reader, writer, up_reader, up_writer, length, self.idle
      )
    log(
        kind="closed",
        host=host,
        port=port,
        bytes_up=sent,
        bytes_down=received,
        **({"reason": f"idle for {self.idle:.0f} s"} if idled else {}),
    )

  async def handle_forward(
      self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
  ) -> None:
    """Forwards an inbound evaluator connection to the serve WebSocket.

    Args:
      reader: Inbound stream reader from the evaluator.
      writer: Inbound stream writer to the evaluator.
    """
    try:
      host, port = self.forward
      up_reader, up_writer = await asyncio.wait_for(
          asyncio.open_connection(host, port), 5
      )
    except (OSError, asyncio.TimeoutError, TypeError):
      writer.close()
      return
    await splice(reader, writer, up_reader, up_writer)


async def serve(args) -> None:
  """Starts the egress proxy and optional WebSocket forwarder listeners.

  Args:
    args: Parsed command-line arguments from `main`.
  """
  allow = [a for a in args.allow.split(",") if a]
  exceptions = [e for e in args.exceptions.split(",") if e]
  forward = host_port(args.forward, 0) if args.forward else None
  proxy = Proxy(Policy(allow, exceptions), forward)
  listening = [
      listen(
          bound(args.proxy_port),
          proxy.handle_proxy,
          proxy.slots,
          MAX_HEAD_BYTES,
      )
  ]
  if forward:
    listening.append(
        listen(
            bound(args.forward_port), proxy.handle_forward, proxy.forward_slots
        )
    )
  log(
      kind="start",
      allow=sorted(f"{h}:{p}" for h, p in proxy.policy.allow),
      exceptions=sorted(f"{h}:{p}" for h, p in proxy.policy.exceptions),
      forward=args.forward or None,
  )
  await asyncio.gather(*listening)


def main(argv=None) -> int:
  """Parses CLI arguments and runs the egress proxy event loop.

  Args:
    argv: Optional list of command-line arguments.

  Returns:
    Process exit status code (0).
  """
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument(
      "--allow",
      default="",
      help="comma-separated host:port pairs the serve container may reach",
  )
  ap.add_argument(
      "--except",
      dest="exceptions",
      default="",
      help=(
          "comma-separated address:port pairs allowed though private (the model"
          " proxy)"
      ),
  )
  ap.add_argument(
      "--forward",
      default="",
      help="host:port of the serve container's websocket",
  )
  ap.add_argument("--proxy-port", type=int, default=3128)
  ap.add_argument("--forward-port", type=int, default=8000)
  try:
    asyncio.run(serve(ap.parse_args(argv)))
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
