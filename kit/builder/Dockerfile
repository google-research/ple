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

# The image submissions are built in and served from. Only /opt/harness is kept from a build, so uv, Node and the
# Pythons uv installs (UV_PYTHON_PREFERENCE) live there; a venv on Debian's python3 would point outside the bundle.
# uv copies from its cache (UV_LINK_MODE), which stays behind in /tmp.
FROM debian:bookworm
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils build-essential python3 \
 && rm -rf /var/lib/apt/lists/*
ARG UV_VERSION=0.11.23
ARG NODE_VERSION=v22.23.3
RUN mkdir -p /opt/harness/bin \
 && curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/opt/harness/bin INSTALLER_NO_MODIFY_PATH=1 sh \
 && curl -fsSL https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz | tar -xJ -C /opt/harness \
 && mv /opt/harness/node-${NODE_VERSION}-linux-x64 /opt/harness/node
ENV PATH=/opt/harness/bin:/opt/harness/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    UV_PYTHON_INSTALL_DIR=/opt/harness/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_TOOL_DIR=/opt/harness/uv-tools \
    UV_TOOL_BIN_DIR=/opt/harness/bin \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    npm_config_cache=/tmp/npm-cache \
    HOME=/tmp/build-home
