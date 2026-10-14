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

"""The evaluation server's default reasoning effort, for the local model proxy.

`config.yaml` loads this module as a LiteLLM callback. A request that sets no
reasoning level gets its model's highest: `max` for Claude and GLM, `high` for
Gemini, gpt-oss, Kimi and MiniMax (Gemini and gpt-oss refuse `max`). A request
that sets one (`reasoning_effort`, `thinking`, `output_config`, `reasoning` or
`chat_template_kwargs`, directly or in `extra_body`) is sent as made, and so is
any request to a model with no levels (Qwen, Llama) or with a thinking switch
of its own (DeepSeek, Gemma).

LiteLLM 1.90 knows no Gemini 4: it drops `reasoning_effort` and `thinking` for
one (an unknown model "does not support" them), and its rule for a Gemini it
does not count as Gemini 3 would send a thinking budget, which Gemini 4 Argon
refuses (400). Loading this module makes Gemini 4 and later take both, as a
thinking level (low, medium or high).

On the Gemini API route (generateContent), a request to a Gemini 3 or later
model that sets neither thinkingLevel nor thinkingBudget gets thinkingLevel
HIGH, because a Gemini model's own default there can be below its highest
(Gemini 4 Argon's is medium, and Gemini 3.8 Flash's thinks far less than high).
"""

from __future__ import annotations

import re
from typing import Any

from litellm.integrations import custom_logger
from litellm.llms.vertex_ai.gemini import vertex_and_google_ai_studio_gemini as vertex_gemini

HIGHEST = (
    ("claude", "max"),
    ("glm", "max"),
    ("gemini", "high"),
    ("gpt-oss", "high"),
    ("kimi", "high"),
    ("minimax", "high"),
)
CALL_TYPES = frozenset({"completion", "acompletion", "anthropic_messages"})
# the Gemini API
NATIVE_CALL_TYPES = frozenset({"agenerate_content", "agenerate_content_stream"})
CHOSEN = (
    "reasoning_effort",
    "thinking",
    "output_config",
    "reasoning",
    "chat_template_kwargs",
)
# Gemini 4 and later: thinking levels, no budget
GEMINI_4 = re.compile(r"gemini-(?:[4-9]|[1-9]\d)")
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
      API route.
    """
    del user_api_key_dict, cache  # Unused.
    if call_type in CALL_TYPES:
      return with_default(data)
    if call_type in NATIVE_CALL_TYPES:
      return native_with_default(data)
    return data


instance = EffortDefault()
