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

"""Judge refusal fallback for the local and evaluation-server model proxies.

Loaded as a LiteLLM callback. When a judge request to `claude-opus-5-5` returns
a refusal (`stop_reason: "refusal"`, mapped by LiteLLM to
`finish_reason: "content_filter"`), the proxy sends the request once to
`claude-opus-5` and returns that reply instead. If that call fails, the
original refusal is returned.

Only judge requests are re-sent, never harness requests: a `<model>-judge`
model name on `kit/model_proxy`, or a key alias ending in `-judge` on the
evaluation server.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from litellm.integrations import custom_logger

FALLBACK = {"claude-opus-5-5": "claude-opus-5"}
REFUSED = ("content_filter", "refusal")
JUDGE_SUFFIX = "-judge"
# Chat Completions parameters copied unchanged into the re-sent judge request
SENT = (
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "n",
    "response_format",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "thinking",
)


def _get(obj: Any, name: str) -> Any:
  """Returns the field `name` of a dictionary or of an object's attributes."""
  return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def judge_request(data: dict[str, Any], user_api_key_dict: Any) -> bool:
  """Returns whether the request is a judge's.

  Args:
    data: The request body.
    user_api_key_dict: The caller's key details.

  Returns:
    True if the key alias ends in `-judge` (evaluation server), or if the
    model name in the original client request ends in `-judge`
    (`kit/model_proxy`, where `effort_default` strips the suffix).
  """
  alias = str(_get(user_api_key_dict, "key_alias") or "")
  body = (data.get("proxy_server_request") or {}).get("body") or {}
  asked = str(body.get("model") or "")
  return alias.endswith(JUDGE_SUFFIX) or (
      asked.endswith(JUDGE_SUFFIX) and asked != JUDGE_SUFFIX
  )


def refused(response: Any) -> bool:
  """Returns whether the model refused, by its first choice's finish_reason."""
  choices = _get(response, "choices") or []
  return bool(choices) and _get(choices[0], "finish_reason") in REFUSED


def fallback_of(model: Any) -> str | None:
  """Returns the fallback for `model`.

  Args:
    model: The model name under any of the proxy's spellings of it
      (`claude-opus-5-5`, `anthropic/claude-opus-5-5`).

  Returns:
    The fallback model's name, or None for a model without one.
  """
  return FALLBACK.get(str(model or "").rsplit("/", 1)[-1])


async def send_again(data: dict[str, Any], model: str, to: str) -> Any:
  """Sends the judge's request to the model `to`.

  Uses the proxy's route for `to` if there is one; otherwise reuses the refused
  model's deployment with its model name replaced by `to`, since the kit's
  config lists only the competition's models.

  Args:
    data: The refused request body.
    model: The refused model, as the proxy named it.
    to: The fallback model.

  Returns:
    The fallback model's reply.

  Raises:
    LookupError: If the proxy has no route for `model` either.
  """
  # pylint: disable=g-import-not-at-top
  import litellm
  from litellm.proxy import proxy_server
  # pylint: enable=g-import-not-at-top

  router = proxy_server.llm_router
  request = {k: data[k] for k in SENT if data.get(k) is not None}
  if data.get("metadata"):
    # Keep the caller's metadata so spend is attributed as for the refused call.
    request["metadata"] = data["metadata"]
  if to in (getattr(router, "model_names", None) or ()):
    return await router.acompletion(model=to, **request)
  deployments = router.get_model_list(model_name=model) or []
  if not deployments:
    raise LookupError(f"the proxy has no route for {model}")
  params = dict(deployments[0]["litellm_params"])
  base = model.rsplit("/", 1)[-1]
  params["model"] = str(params["model"]).replace(base, to)
  return await litellm.acompletion(**{**request, **params})


class JudgeFallback(custom_logger.CustomLogger):
  """LiteLLM callback that sends a judge's refused request to its fallback."""

  async def async_post_call_success_hook(
      self,
      data: dict[str, Any],
      user_api_key_dict: Any,
      response: Any,
  ) -> Any:
    """Returns the fallback model's reply to a judge's refused request.

    Args:
      data: The request body.
      user_api_key_dict: The caller's key details.
      response: The model's reply.

    Returns:
      The fallback model's reply, or None to keep `response`: for a reply that
      is not a refusal, a request that is not a judge's, a model without a
      fallback, or a fallback that failed.
    """
    model = data.get("model")
    to = fallback_of(model)
    if (
        to is None
        or not refused(response)
        or not judge_request(data, user_api_key_dict)
    ):
      return None
    key = _get(user_api_key_dict, "key_alias") or "-"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
      again = await send_again(data, str(model), to)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Keep the original refusal; the grader reports it.
      print(
          f"judge-fallback {stamp}: {model} refused a judge request (key"
          f" {key}); {to} failed too ({type(exc).__name__}: {exc}), the"
          " refusal stands",
          file=sys.stderr,
          flush=True,
      )
      return None
    choices = _get(again, "choices") or [None]
    print(
        f"judge-fallback {stamp}: {model} refused a judge request (key {key});"
        f" re-sent once to {to}, finish_reason"
        f" {_get(choices[0], 'finish_reason')}",
        file=sys.stderr,
        flush=True,
    )
    return again


instance = JudgeFallback()
