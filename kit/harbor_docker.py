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

"""Harbor Docker environment configured for PLEBench evaluation trials.

Harbor 0.20's egress sidecar closes connections when HTTP response headers take
more than 15 seconds (see compose/egress-gost.yaml for why that cuts long judge
calls). When egress control is enabled (as in the allowlisted verifier of a
rubric or checks-judge task), compose/docker-compose-egress-gost.yaml mounts
that gost config over the sidecar's config. Harbor builds a separate verifier
using the trial's environment class (without the job's extra compose files), so
the job file references this class to mirror the evaluation server.
"""

from __future__ import annotations

import pathlib

from harbor.environments.docker.docker import DockerEnvironment

COMPOSE_DIR = pathlib.Path(__file__).resolve().parent / "compose"
GOST_COMPOSE = COMPOSE_DIR / "docker-compose-egress-gost.yaml"
GOST_CONFIG = COMPOSE_DIR / "egress-gost.yaml"
GOST_CONFIG_ENV = "PLEBENCH_EGRESS_GOST_CONFIG"


class KitDockerEnvironment(DockerEnvironment):
  """Harbor Docker environment that overrides the egress sidecar timeout."""

  @property
  def _docker_compose_paths(self) -> list[pathlib.Path]:
    """Returns Docker Compose file paths, adding the egress gost override."""
    paths = super()._docker_compose_paths
    if self._enable_egress_control:
      # right after the sidecar's own file, before Harbor's generated service
      # overlay
      paths.insert(
          paths.index(self._DOCKER_COMPOSE_EGRESS_CONTROL_PATH) + 1,
          GOST_COMPOSE,
      )
    return paths

  def _compose_infra_env_vars(self) -> dict[str, str]:
    """Returns compose environment variables including the gost config path."""
    env = super()._compose_infra_env_vars()
    if self._enable_egress_control:
      env[GOST_CONFIG_ENV] = str(GOST_CONFIG)
    return env
