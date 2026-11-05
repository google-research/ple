#!/bin/bash

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

# Build step: the pinned agy binary (checksum verified) and a Python venv for serve.py, all under /opt/harness.
set -euo pipefail
# agy 1.4.10
AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.4.10-5501573669978112/linux-x64/cli_linux_x64.tar.gz
AGY_SHA512=3a14c21697e86349ce8402e0b1889cbd1d3a5f4e2cb4dc43aae1c48c404240c2ccff9f775eec277e1f759234a33ec026645c63ed22ab6f6ebede561eda7c38b6
curl -fsSL "$AGY_URL" -o /tmp/agy.tar.gz
echo "$AGY_SHA512  /tmp/agy.tar.gz" | sha512sum -c -
mkdir -p /opt/harness/bin
tar -xzf /tmp/agy.tar.gz -C /opt/harness/bin antigravity
mv /opt/harness/bin/antigravity /opt/harness/bin/agy
rm /tmp/agy.tar.gz
uv venv --python 3.12 /opt/harness/venv
uv pip install --python /opt/harness/venv/bin/python -r "$(dirname "$0")/requirements.txt"
