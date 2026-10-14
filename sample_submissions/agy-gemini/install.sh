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
AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.3.2-5813501495738368/linux-x64/cli_linux_x64.tar.gz
AGY_SHA512=8dc2bb8471cfa6326e29e2896add767ad2d1f3714304b70fbae720d68af011182e7afde43964a7a3f54c604ab72c4fad772c8595608b059cfc582fb6798a25d2
curl -fsSL "$AGY_URL" -o /tmp/agy.tar.gz
echo "$AGY_SHA512  /tmp/agy.tar.gz" | sha512sum -c -
mkdir -p /opt/harness/bin
tar -xzf /tmp/agy.tar.gz -C /opt/harness/bin antigravity
mv /opt/harness/bin/antigravity /opt/harness/bin/agy
rm /tmp/agy.tar.gz
uv venv --python 3.12 /opt/harness/venv
uv pip install --python /opt/harness/venv/bin/python -r "$(dirname "$0")/requirements.txt"
