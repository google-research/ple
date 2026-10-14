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

"""Model API parsing and serve container environment configuration."""

from __future__ import annotations

import dataclasses
import ipaddress
import re
import reprlib
import urllib.parse

# Served by the organizers on Google Cloud, at their cost: the newest version of
# each model family.
GOOGLE_CLOUD = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "gemini-4-argon": "Gemini 4 Argon",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)",
    "gpt-oss-120b": "gpt-oss 120B",
    "qwen3-coder-480b": "Qwen3 Coder 480B A35B",
    "qwen3-next-80b": "Qwen3-Next 80B A3B Instruct",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "kimi-k2-thinking": "Kimi K2 Thinking",
    "minimax-m2": "MiniMax M2",
    "glm-5.2": "GLM 5.2",
    "gemma-4-26b": "Gemma 4 26B A4B",
    "llama-4-maverick": "Llama 4 Maverick 17B 128E",
}


@dataclasses.dataclass(frozen=True)
class Provider:
  """Configuration for a model API provider."""

  # OpenAI-compatible base URL (openai-compatible: the submission's
  # model_base_url)
  base_url: str
  key: str  # the secret holding the API key
  native: str = ""  # the provider's own API: anthropic | gemini
  native_url: str = ""


PROVIDERS = {
    "openai": Provider("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": Provider(
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "anthropic",
        "https://api.anthropic.com",
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini",
        "https://generativelanguage.googleapis.com",
    ),
    "openai-compatible": Provider("", "MODEL_API_KEY"),
}
ENV_NAMES = frozenset({
    "MODEL_API",
    "MODEL_NAME",
    "MODEL_BASE_URL",
    "MODEL_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GOOGLE_GEMINI_BASE_URL",
    "GEMINI_API_KEY",
})
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}")
# A DNS name of two or more labels. No label starts or ends with '-', and the
# last starts with a letter, so an IPv4 literal never matches.
HOST_RE = re.compile(
    r"(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(?:\.(?!-)[a-z0-9-]{1,63}(?<!-))*"
    r"\.[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
)
# names that resolve only inside a private network
LOCAL_SUFFIXES = (".local", ".internal", ".localhost")


def public_host_problem(host: str) -> str | None:
  """Returns why a lowercase host is not a valid public DNS name, or None.

  Only the syntax is checked here; the egress proxy rejects names that resolve
  to private addresses.

  Args:
    host: Lowercase hostname to validate.

  Returns:
    A description of why `host` is invalid, or None if it is valid.
  """
  if not HOST_RE.fullmatch(host):
    return (
        "not a public DNS name (no IP address, no single-label host, no label"
        " starting or ending with '-')"
    )
  if host.endswith(LOCAL_SUFFIXES):
    return (
        f"resolves only inside a local network ({', '.join(LOCAL_SUFFIXES)}"
        " names are never public)"
    )
  return None


@dataclasses.dataclass(frozen=True)
class Model:
  """Parsed model specification for a submission."""

  api: str
  provider: str
  name: str
  base_url: str = ""

  @property
  def family(self) -> str:
    """Returns the provider-native API family (anthropic, gemini, or openai)."""
    if self.provider == "google-cloud":
      if self.name.startswith("claude"):
        return "anthropic"
      if self.name.startswith("gemini"):
        return "gemini"
      return "openai"
    return PROVIDERS[self.provider].native or "openai"

  @property
  def key_secret(self) -> str | None:
    """Returns the environment variable name for the API key, or None."""
    return (
        None
        if self.provider == "google-cloud"
        else PROVIDERS[self.provider].key
    )

  @property
  def endpoint(self) -> str:
    """Returns the OpenAI-compatible base URL (empty for google-cloud)."""
    if self.provider == "google-cloud":
      return ""
    return (
        self.base_url
        if self.provider == "openai-compatible"
        else PROVIDERS[self.provider].base_url
    )

  @property
  def hosts(self) -> list[str]:
    """Returns host:port pairs the serve container must reach for this model."""
    urls = [self.endpoint] + (
        [PROVIDERS[self.provider].native_url]
        if self.provider in PROVIDERS
        else []
    )
    return sorted({host_port(u) for u in urls if u})

  @property
  def label(self) -> str:
    """Returns a human-readable display label for the model."""
    return (
        GOOGLE_CLOUD.get(self.name, self.name)
        if self.provider == "google-cloud"
        else f"{self.name} ({self.provider})"
    )


def host_port(url: str) -> str:
  """Extracts the host:port string from a URL.

  Args:
    url: The URL to parse.

  Returns:
    The formatted `host:port` string.
  """
  parts = urllib.parse.urlsplit(url)
  port = parts.port or (443 if parts.scheme == "https" else 80)
  return f"{parts.hostname}:{port}"


_REPR = reprlib.Repr()
_REPR.maxlevel, _REPR.maxstring, _REPR.maxother = 1, 80, 80


def brief(value) -> str:
  """Returns a truncated repr of `value` safe for error messages.

  A participant's value can be a 16 KiB string or a tree of YAML aliases whose
  full representation would be gigabytes.

  Args:
    value: Any value to format.

  Returns:
    A bounded string representation of `value`.
  """
  return _REPR.repr(value)


def parse(api, base_url=None) -> Model:
  """Parses `model_api` and `base_url` into a Model instance.

  Args:
    api: Model identifier in `<provider>/<model>` format.
    base_url: Optional base URL for `openai-compatible` models.

  Returns:
    The parsed `Model` instance.

  Raises:
    ValueError: If `api` or `base_url` is invalid.
  """
  if not isinstance(api, str) or "/" not in api:
    raise ValueError(
        "model_api: <provider>/<model>, e.g. google-cloud/claude-opus-5-5 or"
        " openai/gpt-6-astra"
    )
  provider, name = api.split("/", 1)
  if provider == "google-cloud":
    if name not in GOOGLE_CLOUD:
      raise ValueError(
          f"model_api: google-cloud serves {', '.join(GOOGLE_CLOUD)} (not"
          f" {brief(name)})"
      )
  elif provider not in PROVIDERS:
    raise ValueError(
        f"model_api: the provider is google-cloud, {', '.join(PROVIDERS)} (not"
        f" {brief(provider)})"
    )
  elif not _MODEL_RE.fullmatch(name):
    raise ValueError(f"model_api: {brief(name)} is not a model name")
  if provider == "openai-compatible":
    problem = base_url_problem(base_url)
    if problem:
      raise ValueError(problem)
    return Model(api, provider, name, base_url.rstrip("/"))
  if base_url:
    raise ValueError(
        "model_base_url: only with model_api: openai-compatible/<model>"
    )
  return Model(api, provider, name)


def base_url_problem(url) -> str | None:
  """Returns why `url` cannot be an openai-compatible model_base_url, or None.

  Error messages never repeat the full URL (which could contain credentials or
  keys in user, password, or query parameters); only the hostname may be shown.

  Args:
    url: The candidate base URL to validate.

  Returns:
    A description of the validation problem, or None if valid.
  """
  if not isinstance(url, str) or not url:
    return (
        "model_base_url: required with openai-compatible, e.g."
        " https://llm.example.org/v1"
    )
  # an unpaired surrogate is not printable either
  if any(c.isspace() or not c.isprintable() for c in url):
    return "model_base_url: must be a URL without spaces or control characters"
  try:
    parts = urllib.parse.urlsplit(url)
  except ValueError:  # urlsplit refuses an unbalanced [ in the host
    return "model_base_url: not a URL"
  if (
      parts.scheme != "https"
      or not parts.hostname
      or "@" in parts.netloc
      or parts.query
      or parts.fragment
  ):
    return (
        "model_base_url: must be an https:// URL without credentials, query or"
        " fragment"
    )
  # before the port check, which would read an IPv6 literal's colons as a bad
  # port
  try:
    ipaddress.ip_address(parts.hostname)
  except ValueError:
    pass
  else:
    return (
        f"model_base_url: {brief(parts.hostname)} is an IP address; it must be"
        " a public DNS name (the evaluation server reaches it over the"
        " internet)"
    )
  try:
    port = parts.port
  except ValueError:
    port = 0
  if port == 0 or ":" in parts.netloc and port is None:
    return "model_base_url: invalid port"
  if public_host_problem(parts.hostname):
    return (
        f"model_base_url: {brief(parts.hostname)} must be a public DNS name"
        " (the evaluation server reaches it over the internet)"
    )
  if parts.path.rstrip("/").endswith("/chat/completions"):
    return (
        "model_base_url: the base URL, without /chat/completions (e.g."
        " https://llm.example.org/v1)"
    )
  return None


def server_origin(url: str) -> str:
  """Returns a model server URL without a trailing slash or `/v1` suffix.

  Args:
    url: The model server URL.

  Returns:
    The normalized origin URL where native APIs (Anthropic, Gemini) are served.
  """
  return url.rstrip("/").removesuffix("/v1")


def openai_base(url: str) -> str:
  """Returns a model server URL normalized to end with `/v1` once.

  Args:
    url: The model server URL.

  Returns:
    The OpenAI-compatible base URL ending in `/v1`.
  """
  return server_origin(url) + "/v1"


def environment(model: Model, key: str, endpoint: str = "") -> dict[str, str]:
  """Builds the model environment variables for the serve container.

  Args:
    model: The parsed `Model` specification.
    key: The API key to present to the model server.
    endpoint: The evaluator's model server URL for google-cloud models (with or
      without `/v1`).

  Returns:
    A dictionary of environment variables for the serve container.
  """
  origin = server_origin(endpoint) if model.provider == "google-cloud" else ""
  base = openai_base(endpoint) if origin else model.endpoint
  env = {
      "MODEL_API": model.api,
      "MODEL_NAME": model.name,
      "MODEL_BASE_URL": base,
      "MODEL_API_KEY": key,
      "OPENAI_BASE_URL": base,
      "OPENAI_API_KEY": key,
  }
  native = origin or (
      PROVIDERS[model.provider].native_url
      if model.provider in PROVIDERS
      else ""
  )
  if model.family == "anthropic":
    env.update(ANTHROPIC_BASE_URL=native, ANTHROPIC_API_KEY=key)
  elif model.family == "gemini":
    env.update(GOOGLE_GEMINI_BASE_URL=native, GEMINI_API_KEY=key)
  return env
