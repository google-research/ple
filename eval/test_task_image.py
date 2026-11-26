#!/usr/bin/env python3

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

"""Offline regressions for task image builds and evaluator preflight."""

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from eval import run
from kit import stack


class TaskImageTest(unittest.TestCase):

  def test_cached_image_on_expected_platform_is_reused(self):
    with mock.patch.object(run.subprocess, "run") as docker:
      docker.return_value = subprocess.CompletedProcess([], 0, "linux/amd64\n")
      self.assertEqual(
          run._built(pathlib.Path("task"), pathlib.Path("environment"), "tag"),
          "tag",
      )
      docker.assert_called_once()
      self.assertIn("{{.Os}}/{{.Architecture}}", docker.call_args.args[0])
      self.assertEqual(
          docker.call_args.kwargs["env"]["DOCKER_DEFAULT_PLATFORM"],
          "linux/amd64",
      )

  def test_missing_or_wrong_platform_image_is_built_for_harbor(self):
    with tempfile.TemporaryDirectory() as folder:
      task = pathlib.Path(folder)
      (task / "task.toml").write_text("[environment]\n", encoding="utf-8")
      for result in (
          subprocess.CompletedProcess([], 1, ""),
          subprocess.CompletedProcess([], 0, "linux/arm64\n"),
      ):
        with self.subTest(result=result):
          with (
              mock.patch.object(run.subprocess, "run") as docker,
              mock.patch.object(
                  stack, "build_env",
                  return_value={"DOCKER_DEFAULT_PLATFORM": "linux/amd64",
                                "BUILDX_BUILDER": "colima"},
              ),
          ):
            docker.side_effect = [result, subprocess.CompletedProcess([], 0)]
            self.assertEqual(run._built(task, task / "environment", "tag"), "tag")
            self.assertEqual(docker.call_count, 2)
            self.assertEqual(
                docker.call_args.kwargs["env"],
                {"DOCKER_DEFAULT_PLATFORM": "linux/amd64",
                 "BUILDX_BUILDER": "colima", "DOCKER_BUILDKIT": "1"},
            )

  def test_preflight_checks_both_plugins_before_cleanup(self):
    for missing in ("buildx", "compose", None):
      with self.subTest(missing=missing):
        with (
            mock.patch.object(run, "check_harbor"),
            mock.patch.object(run, "harbor", return_value="harbor"),
            mock.patch.object(run.build, "require_local_daemon"),
            mock.patch.object(run.check, "sweep_leftovers") as sweep,
            mock.patch.object(stack, "docker") as docker,
        ):
          docker.side_effect = lambda plugin, *args, **kwargs: (
              subprocess.CompletedProcess([], int(plugin == missing))
          )
          if missing:
            with self.assertRaisesRegex(stack.InfraError, f"Docker {missing}"):
              run.preflight()
            sweep.assert_not_called()
          else:
            run.preflight()
            sweep.assert_called_once_with()
            self.assertEqual(
                [call.args for call in docker.call_args_list],
                [("buildx", "version"), ("compose", "version")],
            )


if __name__ == "__main__":
  unittest.main()
