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

"""Install-time patch so Hermes preserves Claude thinking blocks via the gateway.

By default, Hermes (`anthropic_thinking_route` in
`agent/anthropic_thinking_policy.py`) strips thinking blocks from prior
assistant turns whenever the base URL is not an official Anthropic or cloud
host. Behind the evaluation gateway, that forces Claude at max effort to
re-derive its reasoning on every turn.

This patch adds an early return: when `HERMES_NATIVE_THINKING_BASE_URL` matches
the active `base_url` for a Claude model, `anthropic_thinking_route` returns
`"native"` so signed thinking blocks are preserved.

Usage: python thinking_patch.py <hermes checkout>

Idempotent if run multiple times. Raises `SystemExit` if the anchor is not found
exactly once, so a Hermes upgrade cannot silently skip the patch.
"""

import pathlib
import sys

POLICY = pathlib.Path("agent") / "anthropic_thinking_policy.py"
MARKER = "# hermes-opus thinking_patch.py: native replay on our endpoint"
# the last line of anthropic_thinking_route's docstring (hermes v0.22.2)
ANCHOR = (
    "    signatures: strip all). The converter, accounting and"
    ' signature-rejection state share this."""\n'
)
INJECT = (
    f"    {MARKER}\n"
    "    import os as _patch_os\n"
    "    _native = _patch_os.environ.get('HERMES_NATIVE_THINKING_BASE_URL')\n"
    "    _native = (_native or '').strip().rstrip('/').lower()\n"
    "    _url = str(base_url or '').strip().rstrip('/').lower()\n"
    "    if _native and _url == _native and 'claude' in str(model).lower():\n"
    "        return 'native'\n"
)


def patch(checkout: pathlib.Path) -> str:
  """Adds the early return to the checkout's thinking-route policy.

  Args:
    checkout: The Hermes source checkout.

  Returns:
    "patched", or "already patched" when the marker is there.

  Raises:
    SystemExit: If the anchor is not found exactly once.
  """
  path = checkout / POLICY
  text = path.read_text(encoding="utf-8")
  if MARKER in text:
    return "already patched"
  if text.count(ANCHOR) != 1:
    raise SystemExit(
        f"thinking_patch: the anchor is in {path} {text.count(ANCHOR)} times,"
        " not once: did Hermes change?"
    )
  path.write_text(text.replace(ANCHOR, ANCHOR + INJECT), encoding="utf-8")
  return "patched"


if __name__ == "__main__":
  hermes = pathlib.Path(sys.argv[1])
  print(f"thinking_patch: {patch(hermes)} {hermes / POLICY}")
