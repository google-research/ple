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

"""Submission specification (`submission.yaml`) parsing and validation rules."""

from __future__ import annotations

import dataclasses
import io
import pathlib
import re
import shlex

import yaml

from . import models

FILE = "submission.yaml"
MAX_BYTES = 16 * 1024
ENV_FILE = ".env"  # the keys, beside submission.yaml (read_env)
MAX_ENV_BYTES = 16 * 1024
MAX_ENV_VARS = 32
SRC = "/opt/harness/src"
# every task's agent time limit, in eval/run.py and on the evaluation server
AGENT_TIMEOUT_SEC = 6 * 3600
KEYS = ("name", "model_api", "model_base_url", "install", "serve", "env")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,38}")
ENV_KEY_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
HOST_RE = models.HOST_RE
PLACEHOLDERS = {
    "install": {"src"},
    "serve": {"src", "port"},
    "env": {"src", "port"},
}
KNOWN = {"src", "port"}
# A placeholder in braces, unless right after `$`: `${src}` and `${port}` are
# the shell's own expansion, left as is
PLACEHOLDER_RE = re.compile(r"(?<!\$)\{([a-z_]+)\}")
REMOVED = {
    "secrets": f"removed; put keys in {ENV_FILE} at the repository root",
    "egress": (
        "removed; a harness reaches its model's endpoint and no other host"
    ),
}
RESERVED = (
    frozenset({
        "PORT",
        "HOME",
        "USER",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "NODE_USE_ENV_PROXY",
    })
    | models.ENV_NAMES
)


class SpecError(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when `submission.yaml` or `.env` fails validation."""

  def __init__(self, problems: list[str]):
    """Initializes SpecError with a list of validation problems.

    Args:
      problems: Human-readable validation error messages.
    """
    super().__init__("; ".join(problems))
    self.problems = problems


@dataclasses.dataclass(frozen=True)
class Submission:
  """Validated submission configuration from `submission.yaml`."""

  name: str
  model: models.Model
  serve: str
  install: str = ""
  env: dict[str, str] = dataclasses.field(default_factory=dict)

  @property
  def allowed_hosts(self) -> list[str]:
    """Returns sorted host:port endpoints the serve container may reach."""
    return sorted(self.model.hosts)


def _reserved(name: str, model: models.Model | None) -> str:
  """Explains why a submission cannot set a reserved environment variable.

  A variable in `RESERVED` is either set directly by the evaluator or is a
  model variable unused by `model` (`models.environment` sets the active ones).

  Args:
    name: The reserved environment variable name.
    model: The parsed `Model`, or None if `model_api` was invalid.

  Returns:
    A human-readable explanation for why `name` cannot be set.
  """
  if (
      name not in models.ENV_NAMES
      or model is not None
      and name in models.environment(model, "")
  ):
    return "set by the evaluator"
  return "reserved: a model variable (docs/models.md)" + (
      " your model does not use" if model else ""
  )


def _placeholders(where: str, text: str) -> list[str]:
  """Returns validation errors for unsupported placeholders in `text`.

  Args:
    where: The configuration section name (`install`, `serve`, or `env`).
    text: The string to scan for `{placeholder}` tokens.

  Returns:
    A list of error messages for disallowed placeholders.
  """
  bad = sorted({
      m
      for m in PLACEHOLDER_RE.findall(text)
      if m in KNOWN and m not in PLACEHOLDERS[where]
  })
  return [f"{where}: {{{m}}} is not available here" for m in bad]


def _unencodable(text: str) -> bool:
  r"""Returns whether `text` contains an unencodable lone surrogate.

  YAML can produce lone surrogates from unpaired escapes such as `\ud83d`, which
  no file or environment variable can hold.

  Args:
    text: The string to check for UTF-8 encodability.

  Returns:
    True if `text` cannot be encoded as UTF-8, False otherwise.
  """
  try:
    text.encode()
  except UnicodeEncodeError:
    return True
  return False


def validate(data) -> list[str]:
  """Validates parsed YAML data against submission rules without raising.

  Args:
    data: Parsed YAML object from `submission.yaml`.

  Returns:
    A list of actionable problem descriptions, or an empty list when valid.
  """
  if not isinstance(data, dict):
    return ["the file must be a YAML mapping"]
  problems = [
      f"{k}: {REMOVED[k]}"
      if k in REMOVED
      else f"unknown key {models.brief(k)} (allowed: {', '.join(KEYS)})"
      for k in data
      if k not in KEYS
  ]
  if not isinstance(data.get("name"), str) or not NAME_RE.fullmatch(
      data["name"]
  ):
    problems.append(
        "name: 2-39 characters of a-z, 0-9 and '-', starting with a letter or"
        " digit"
    )
  model = None
  try:
    model = models.parse(data.get("model_api"), data.get("model_base_url"))
  except ValueError as exc:
    problems.append(str(exc))
  for key in ("install", "serve"):
    value = data.get(key)
    if value is None:  # `install:` left blank is the same as no install
      continue
    if not isinstance(value, str):
      problems.append(f"{key}: a shell command, as a string")
    elif "\x00" in value or _unencodable(value):
      problems.append(
          f"{key}: holds a NUL or an unpaired surrogate escape, which no shell"
          " command can"
      )
    elif value:
      problems += _placeholders(key, value)
  serve = data.get("serve")
  if not serve or isinstance(serve, str) and not serve.strip():
    problems.append(
        "serve: required, the command that starts your server on $PORT, "
        "e.g. /opt/harness/venv/bin/python {src}/serve.py"
    )
  elif isinstance(serve, str) and "\x00" not in serve:
    try:
      words = shlex.split(serve, comments=True)
    except ValueError as exc:
      problems.append(
          f"serve: not a shell command line ({exc}): check its quotes"
      )
    else:
      if not words:  # a comment alone: the server would exit at once
        problems.append(
            "serve: only a comment; it must be the command that starts your"
            " server on $PORT, e.g. /opt/harness/venv/bin/python {src}/serve.py"
        )
  env = data.get("env") or {}
  if not isinstance(env, dict):
    problems.append('env: a mapping of NAME: "value"')
    env = {}
  for k, v in env.items():
    if not isinstance(k, str) or not ENV_KEY_RE.fullmatch(k):
      problems.append(
          f"env: {models.brief(k)} is not an environment variable name"
      )
    elif k in RESERVED:
      problems.append(f"env.{k}: {_reserved(k, model)}; choose another name")
    elif not isinstance(v, str):
      problems.append(
          f'env.{k}: quote the value ({k}: "..."); YAML read it as '
          f"{type(v).__name__} {models.brief(v)}"
      )
    elif "${" in v or any(c in v for c in "\n\r\x00") or _unencodable(v):
      problems.append(
          f"env.{k}: one line, written out (no ${{...}}); API keys belong in"
          f" {ENV_FILE}"
      )
    else:
      problems += _placeholders("env", v)
  return list(dict.fromkeys(problems))


def parse(text: str) -> Submission:
  """Parses `submission.yaml` text into a `Submission`.

  Args:
    text: Raw YAML content of `submission.yaml`.

  Returns:
    The validated `Submission` instance.

  Raises:
    SpecError: If `text` exceeds size limits, is invalid YAML, or fails
      validation.
  """
  if len(text.encode("utf-8", "surrogatepass")) > MAX_BYTES:
    raise SpecError([f"{FILE} is larger than {MAX_BYTES // 1024} KiB"])
  try:
    data = yaml.safe_load(text)
  # ValueError: a number past int()'s digit limit, a 13th month
  except (yaml.YAMLError, ValueError) as exc:
    raise SpecError([f"not valid YAML: {exc}"]) from None
  except RecursionError:
    raise SpecError([f"{FILE} nests too deeply"]) from None
  problems = validate(data)
  if problems:
    raise SpecError(problems)
  return Submission(
      name=data["name"],
      model=models.parse(data["model_api"], data.get("model_base_url")),
      serve=data["serve"].strip(),
      install=(data.get("install") or "").strip(),
      env=dict(data.get("env") or {}),
  )


class EntryError(ValueError):
  """Raised when a path is not a submission directory or submission.yaml."""


def entry_dir(path: str | pathlib.Path) -> pathlib.Path:
  """Resolves a submission directory from a directory or `submission.yaml` path.

  Other files are rejected rather than staged as an empty tree because the
  evaluation server reads `submission.yaml` at the root of the build context.
  Non-existent paths not named `submission.yaml` are returned as-is for the
  caller to report.

  Args:
    path: Path to the submission directory or its `submission.yaml` file.

  Returns:
    The resolved submission directory path.

  Raises:
    EntryError: If `path` points to an existing file other than
      `submission.yaml`.
  """
  p = pathlib.Path(path)
  if p.name == FILE and not p.is_dir():
    return p.parent
  if p.exists() and not p.is_dir():
    raise EntryError(
        f"{p} is a file: give the directory that holds {FILE} (or {FILE}"
        " itself)"
    )
  return p


def load(path: str | pathlib.Path) -> Submission:
  """Loads and validates `submission.yaml` from a directory or file path.

  Args:
    path: Path to a directory containing `submission.yaml` or the file itself.

  Returns:
    The validated `Submission` instance.

  Raises:
    SpecError: If the file is not valid UTF-8 or fails `parse` validation.
  """
  path = pathlib.Path(path)
  try:
    text = (path / FILE if path.is_dir() else path).read_bytes().decode("utf-8")
  except UnicodeDecodeError:
    raise SpecError([f"{FILE} is not UTF-8 text"]) from None
  return parse(text)


def fill(text: str, port: int | None = None) -> str:
  """Substitutes `{src}` and `{port}` placeholders in `text`.

  Other braces and shell variable expansions (`${src}`, `${port}`) are left
  unchanged.

  Args:
    text: The command or environment value template.
    port: Optional port number to substitute for `{port}`.

  Returns:
    The string with placeholders substituted.
  """
  values = {"src": SRC, **({"port": str(port)} if port is not None else {})}
  return PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def read_env(text: str) -> dict[str, str]:
  """Parses `NAME=value` lines from `.env` content without variable expansion.

  Uses `python-dotenv` to handle quoting, escapes, comments, and `export`
  prefixes, while rejecting `${...}` interpolations since the evaluator has no
  outer variables to expand. Leading UTF-8 BOMs are stripped and valueless
  keys are ignored. Error messages report line numbers without quoting values.

  Args:
    text: Raw text content of the `.env` file.

  Returns:
    A dictionary mapping environment variable names to their string values.

  Raises:
    SpecError: If the `.env` content exceeds limits or contains invalid lines.
  """
  # the kit's and the portal's environments both have python-dotenv
  from dotenv.parser import parse_stream  # pylint: disable=g-import-not-at-top

  text = text.removeprefix("\ufeff")
  if len(text.encode("utf-8", "surrogatepass")) > MAX_ENV_BYTES:
    raise SpecError([f"{ENV_FILE} is larger than {MAX_ENV_BYTES // 1024} KiB"])
  problems: list[str] = []
  values: dict[str, str] = {}
  first: dict[str, int] = {}
  for b in parse_stream(io.StringIO(text)):
    where = f"{ENV_FILE} line {b.original.line}"
    if b.error:
      problems.append(
          f"{where}: not a NAME=value line (quote a value that holds a space"
          " followed by #)"
      )
    # a blank line, a comment, or a name without a value
    elif b.key is None or b.value is None:
      continue
    elif not ENV_KEY_RE.fullmatch(b.key):
      problems.append(
          f"{where}: {models.brief(b.key)} is not an environment variable name"
          " (A-Z, 0-9 and _)"
      )
    elif b.key in first:
      problems.append(
          f"{where}: {b.key} is set again (first on line {first[b.key]})"
      )
    elif any(c in b.value for c in "\r\n\x00") or _unencodable(b.value):
      problems.append(f"{where}: {b.key} must be one line")
    elif "${" in b.value:
      problems.append(f"{where}: {b.key}: write the value out (no ${{...}})")
    else:
      first[b.key] = b.original.line
      values[b.key] = b.value
  if len(values) > MAX_ENV_VARS:
    problems.append(f"{ENV_FILE}: at most {MAX_ENV_VARS} variables")
  if problems:
    raise SpecError(problems)
  return values


def env_problems(sub: Submission, values: dict[str, str]) -> list[str]:
  """Checks `.env` variables against a submission's model and `env` rules.

  Reports if the required model API key is missing, if a variable other than
  that key is reserved, or if a variable duplicates one in `sub.env`. Error
  messages name variables without exposing their values.

  Args:
    sub: The validated `Submission`.
    values: Environment variables parsed by `read_env`.

  Returns:
    A list of validation problem descriptions, or an empty list if valid.
  """
  key = sub.model.key_secret
  problems = (
      [f"add {key}=<your key> to {ENV_FILE}: {sub.model.api} needs it"]
      if key and not values.get(key)
      else []
  )
  for name in values:
    if name in RESERVED and name != key:
      problems.append(
          f"{name} in {ENV_FILE} is {_reserved(name, sub.model)}"
          + (
              f" (your model's key is {key})"
              if key
              else f" ({sub.model.api} needs no key)"
          )
      )
    elif name in sub.env:
      problems.append(f"{name} is in both {ENV_FILE} and env: keep one")
  return problems
