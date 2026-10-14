#!/usr/bin/env bash

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

# Serve the google-cloud/<id> models locally, from Vertex AI in your own Google Cloud project.
#   gcloud auth application-default login
#   export VERTEXAI_PROJECT=<your project> MODEL_PROXY_KEY=$(openssl rand -hex 24)
#   kit/model_proxy/start.sh &                   # listens on 0.0.0.0:4000 (MODEL_PROXY_HOST/PORT to change)
#   export MODEL_PROXY_URL=http://localhost:4000
#   uv run python -m kit check <dir>
set -euo pipefail
: "${VERTEXAI_PROJECT:?set VERTEXAI_PROJECT to a Google Cloud project with Vertex AI enabled}"
: "${MODEL_PROXY_KEY:?set MODEL_PROXY_KEY to the key clients will present}"
exec uv run --no-project --python 3.11 --with 'litellm[proxy,google]==1.90.0' litellm \
  --config "$(dirname "$0")/config.yaml" --host "${MODEL_PROXY_HOST:-0.0.0.0}" --port "${MODEL_PROXY_PORT:-4000}"
