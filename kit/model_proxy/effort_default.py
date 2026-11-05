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

"""Default reasoning effort and server-tool guard for the local model proxy.

Loaded by `config.yaml` as a LiteLLM callback to match the evaluation server:

- Chat Completions and Anthropic Messages requests that omit a reasoning level
  (`reasoning_effort`, `thinking`, `output_config`, `reasoning`, or
  `chat_template_kwargs`, either top-level or in `extra_body`) receive the
  model's highest `reasoning_effort`: `max` for Claude and GLM, and `high` for
  Gemini and gpt-oss. Models without effort levels (Llama, Gemma) are left
  unchanged.
- Gemini API requests (`generateContent`, `streamGenerateContent`) to Gemini 3
  and later models whose `thinkingConfig` sets neither `thinkingLevel` nor
  `thinkingBudget` receive `thinkingLevel` `HIGH`.
- Patches LiteLLM 1.90's Vertex Gemini adapter so Gemini 4+ models accept
  `reasoning_effort` and `thinking` as thinking levels (`low`, `medium`, `high`)
  instead of a token budget (which Gemini 4 Argon rejects with HTTP 400).
- Judge requests (`<model>-judge`, sent by `eval/run.py`) are routed to
  `<model>` without adding a reasoning level, matching the evaluation server's
  judge keys.
- Rejects provider-hosted tools (Google Search, URL context, web search/fetch,
  code execution/interpreter, file search, computer use, remote MCP) and
  every route outside the served set (`SERVED_CALL_TYPES`) with HTTP 400 so
  harnesses cannot access the internet through the model provider.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from litellm.integrations import custom_logger
from litellm.llms.vertex_ai.gemini import vertex_and_google_ai_studio_gemini as vertex_gemini

HIGHEST = (
    ("claude", "max"),
    ("glm", "max"),
    ("gemini", "high"),
    ("gpt-oss", "high"),
)
CALL_TYPES = frozenset({"completion", "acompletion", "anthropic_messages"})
# Gemini native generateContent routes
NATIVE_CALL_TYPES = frozenset({"agenerate_content", "agenerate_content_stream"})
# Allowed LiteLLM call types (Chat Completions, Anthropic Messages, Responses
# API, and Gemini generateContent). Every other call type is rejected:
# pass-through and realtime routes, for example, bypass pre-call tool
# inspection.
SERVED_CALL_TYPES = CALL_TYPES | NATIVE_CALL_TYPES | frozenset({
    "responses",
    "aresponses",
    "generate_content",
    "generate_content_stream",
})
CHOSEN = (
    "reasoning_effort",
    "thinking",
    "output_config",
    "reasoning",
    "chat_template_kwargs",
)
# Gemini 4+ models use thinking levels rather than token budgets
GEMINI_4 = re.compile(r"gemini-(?:[4-9]|[1-9]\d)")
# Suffix used by eval/run.py to request a model at its standard default effort
JUDGE_SUFFIX = "-judge"
# Provider-side tool parameters, tool types, and Gemini tool keys blocked on
# both the evaluation server and kit/model_proxy.
SERVER_TOOL_PARAMS = ("web_search_options", "mcp_servers")
SERVER_TOOL_TYPES = (
    "web_search",
    "web_fetch",
    "code_execution",
    "code_interpreter",
    "file_search",
    "computer_use",
    "google_search",
    "url_context",
    "enterprise_web_search",
    "mcp",
)
SERVER_TOOL_KEYS = frozenset({
    "googleSearch",
    "google_search",
    "googleSearchRetrieval",
    "google_search_retrieval",
    "enterpriseWebSearch",
    "enterprise_web_search",
    "urlContext",
    "url_context",
    "codeExecution",
    "code_execution",
    "googleMaps",
    "google_maps",
    "computerUse",
    "computer_use",
    "fileSearch",
    "file_search",
    "retrieval",
})
# LiteLLM 1.90's /v1/messages adapter takes a tool merely named web_search for
# its web search when the model is not Claude (Google Search for Gemini).
NAMED_WEB_SEARCH = "named:web_search"
# Reported name for an `extra_body` that is not an object and cannot be read
EXTRA_BODY = "extra_body"
NO_WEB_ACCESS = (
    "the competition gives the harness no web access through the model"
)
VertexGeminiConfig = vertex_gemini.VertexGeminiConfig
# pylint: disable=protected-access
_takes_level = VertexGeminiConfig._is_gemini_3_or_newer
_level = VertexGeminiConfig._map_reasoning_effort_to_thinking_level
# pylint: enable=protected-access
_supported = VertexGeminiConfig.get_supported_openai_params


def takes_level(model: Any) -> bool:
  """Returns whether `model` takes a thinking level rather than a budget.

  Args:
    model: The model name a request asks for.

  Returns:
    True for the Gemini 3 models LiteLLM knows and for Gemini 4 and later.
  """
  return _takes_level(model) or bool(GEMINI_4.search(str(model or "")))


def gemini_level(reasoning_effort: Any, model: Any = None) -> dict[str, Any]:
  """Maps a reasoning effort to the Gemini thinking config LiteLLM sends.

  This is LiteLLM's mapping, except that `medium` stays `medium` on Gemini 4
  (LiteLLM raises it to `high` for a model it does not know), and `max` and
  `xhigh` become `high` there: Argon's MAX thinks less than HIGH, and LiteLLM
  refuses both.

  Args:
    reasoning_effort: The requested effort, such as `low` or `max`.
    model: The model name a request asks for.

  Returns:
    The thinking config, such as `{"thinkingLevel": "high", ...}`.
  """
  if GEMINI_4.search(str(model or "")):
    if reasoning_effort in ("max", "xhigh"):
      reasoning_effort = "high"
    if reasoning_effort == "medium":
      return {"thinkingLevel": "medium", "includeThoughts": True}
  return _level(reasoning_effort, model)


def supported(self: Any, model: str) -> list[str]:
  """Returns LiteLLM's supported parameters, with effort for Gemini 4.

  Args:
    self: The `VertexGeminiConfig` instance.
    model: The model name a request asks for.

  Returns:
    LiteLLM's list, plus `reasoning_effort` and `thinking` for Gemini 4 and
    later.
  """
  params = _supported(self, model)
  if GEMINI_4.search(str(model or "")):
    params += [p for p in ("reasoning_effort", "thinking") if p not in params]
  return params


# LiteLLM may load this file more than once: patch its rules the first time.
if not getattr(_takes_level, "_gemini_4", False):
  takes_level._gemini_4 = True  # pylint: disable=protected-access
  # pylint: disable=protected-access
  VertexGeminiConfig._is_gemini_3_or_newer = staticmethod(takes_level)
  VertexGeminiConfig._map_reasoning_effort_to_thinking_level = staticmethod(
      gemini_level
  )
  # pylint: enable=protected-access
  VertexGeminiConfig.get_supported_openai_params = supported


def highest(model: Any) -> str | None:
  """Returns the highest reasoning level of `model`.

  Args:
    model: The model name a request asks for.

  Returns:
    The model's highest reasoning level, or None for a model without levels.
  """
  name = str(model or "").lower()
  return next((level for key, level in HIGHEST if key in name), None)


def with_default(data: dict[str, Any]) -> dict[str, Any]:
  """Adds the model's highest reasoning level to a request that sets none.

  Args:
    data: The request body, which is changed in place.

  Returns:
    `data`, with `reasoning_effort` set when the request chose no level and the
    model has one.
  """
  extra = data.get("extra_body")
  if not isinstance(extra, dict):
    extra = {}
  if any(data.get(k) is not None or k in extra for k in CHOSEN):
    return data
  level = highest(data.get("model"))
  if level:
    data["reasoning_effort"] = level
  return data


def native_with_default(data: dict[str, Any]) -> dict[str, Any]:
  """Adds thinkingLevel HIGH to a Gemini API request that sets no level.

  A request that sets a level or a budget (in camelCase or snake_case, as the
  REST API takes both) is sent as made, and so is a request to a model that
  takes no thinking level.

  Args:
    data: The Gemini API request body, which is changed in place.

  Returns:
    `data`, with `generationConfig.thinkingConfig.thinkingLevel` set to HIGH
    when the request chose none and the model takes levels.
  """
  if not takes_level(str(data.get("model") or "").lower()):
    return data
  # LiteLLM forwards only generationConfig
  if "generation_config" in data and "generationConfig" not in data:
    data["generationConfig"] = data.pop("generation_config")
  config = data.get("generationConfig") or {}
  if not isinstance(config, dict):
    return data
  tkey = (
      "thinking_config"
      if "thinking_config" in config and "thinkingConfig" not in config
      else "thinkingConfig"
  )
  thinking = config.get(tkey) or {}
  chosen = (
      "thinkingLevel",
      "thinking_level",
      "thinkingBudget",
      "thinking_budget",
  )
  if not isinstance(thinking, dict) or any(
      thinking.get(k) is not None for k in chosen
  ):
    return data
  data["generationConfig"] = {
      **config,
      tkey: {**thinking, "thinkingLevel": "HIGH"},
  }
  return data


def judged(data: dict[str, Any]) -> bool:
  """Returns whether `data` is the judge's request (`<model>-judge`).

  Args:
    data: The request body. A judge's request has its model set back to
      `<model>`, in place.

  Returns:
    True for a judge's request.
  """
  model = str(data.get("model") or "")
  if not model.endswith(JUDGE_SUFFIX) or model == JUDGE_SUFFIX:
    return False
  data["model"] = model[: -len(JUDGE_SUFFIX)]
  return True


def server_tools(data: dict[str, Any], call_type: str = "") -> list[str]:
  """Finds the provider-run parameters and tools in a request, and removes them.

  Reads names and types only, never message content. `extra_body` is read like
  the request itself, as LiteLLM merges it into the provider's body; one that is
  not an object cannot be read, and is refused as `EXTRA_BODY`.

  Args:
    data: The request body, which is changed in place: the parameters and tools
      found are taken out.
    call_type: The kind of call, such as `anthropic_messages`.

  Returns:
    Their names, each once: a parameter, a tool's `type`, or its Gemini key.
  """
  found = []
  extra = data.get("extra_body")
  if extra is not None and not isinstance(extra, dict):
    del data["extra_body"]
    found.append(EXTRA_BODY)
    extra = None
  bodies = [data, *([extra] if extra else [])]
  for body in bodies:
    for key in SERVER_TOOL_PARAMS:
      if body.pop(key, None) is not None:
        found.append(key)
  named_is_search = (
      call_type == "anthropic_messages"
      and "claude" not in str(data.get("model") or "").lower()
  )
  # Tools can appear at the top level or under the Gemini SDK's `config`, in
  # both the request and its `extra_body`.
  configs = [b["config"] for b in bodies if isinstance(b.get("config"), dict)]
  for holder in [*bodies, *configs]:
    tools = holder.get("tools")
    if not isinstance(tools, list):
      continue
    kept = []
    for tool in tools:
      names = []
      if isinstance(tool, dict):
        kind = tool.get("type")
        if isinstance(kind, str) and kind.startswith(SERVER_TOOL_TYPES):
          names.append(kind)
        names += sorted(SERVER_TOOL_KEYS.intersection(tool))
        if not names and named_is_search and tool.get("name") == "web_search":
          names.append(NAMED_WEB_SEARCH)
      if names:
        found += names
      else:
        kept.append(tool)
    if len(kept) != len(tools):
      holder["tools"] = kept
  return list(dict.fromkeys(found))


def no_web_access(found: list[str]) -> str:
  """Returns the error message stating the rule and what to remove.

  Args:
    found: The names `server_tools` returned.
  """
  params = [f for f in found if f in (*SERVER_TOOL_PARAMS, EXTRA_BODY)]
  tools = [
      "web_search" if f == NAMED_WEB_SEARCH else f
      for f in found
      if f not in params
  ]
  parts = [
      f"the {', '.join(names)} {what}{'s' if len(names) > 1 else ''}"
      for names, what in ((params, "parameter"), (tools, "tool"))
      if names
  ]
  message = f"{NO_WEB_ACCESS}; remove {' and '.join(parts)}"
  if NAMED_WEB_SEARCH in found:
    message += (
        " (a tool named web_search is LiteLLM's web search on /v1/messages for"
        " this model; rename it)"
    )
  return message


class EffortDefault(custom_logger.CustomLogger):
  """LiteLLM callback that sets the default reasoning level before each call."""

  async def async_pre_call_hook(
      self,
      user_api_key_dict: Any,
      cache: Any,
      data: dict[str, Any],
      call_type: str,
  ) -> dict[str, Any]:
    """Returns the request body to send, with the default reasoning level.

    Args:
      user_api_key_dict: The caller's key details (unused).
      cache: LiteLLM's cache (unused).
      data: The request body.
      call_type: The kind of call, such as `acompletion`.

    Returns:
      The request body, with the default reasoning level for model calls:
      `reasoning_effort` on the chat routes, thinkingLevel HIGH on the Gemini
      API route. A judge's request (`judged`) gets no level.

    Raises:
      fastapi.HTTPException: A 400 naming the provider-run tools a request
        carries (`server_tools`), or the route of a request outside
        `SERVED_CALL_TYPES`; LiteLLM's proxy answers the caller with it.
    """
    del user_api_key_dict, cache  # Unused.
    if call_type not in SERVED_CALL_TYPES:
      from fastapi import HTTPException  # pylint: disable=g-import-not-at-top

      print(
          f"unread-route refused route={call_type}", file=sys.stderr, flush=True
      )
      raise HTTPException(
          status_code=400,
          detail={
              "error": (
                  f"{NO_WEB_ACCESS}; the {call_type} route sends requests to"
                  " the provider unread, so this proxy does not serve it: use"
                  " the chat, Messages, Responses or Gemini API route"
              )
          },
      )
    found = server_tools(data, call_type)
    if found:
      # Raise rather than return: LiteLLM turns a string returned from this
      # hook into a 200 chat reply on the chat route.
      from fastapi import HTTPException  # pylint: disable=g-import-not-at-top

      print(
          f"server-tools refused={','.join(found)} route={call_type}"
          f" model={data.get('model')}",
          file=sys.stderr,
          flush=True,
      )
      raise HTTPException(
          status_code=400, detail={"error": no_web_access(found)}
      )
    if judged(data):
      return data
    if call_type in CALL_TYPES:
      return with_default(data)
    if call_type in NATIVE_CALL_TYPES:
      return native_with_default(data)
    return data


instance = EffortDefault()
