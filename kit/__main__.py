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

# pylint: disable=invalid-name
"""CLI entry point for the PLEBench submission kit.

Runs a submission entry locally through the evaluation server's steps. API keys
are read from the submission's `.env` file, matching the server. `google-cloud`
models run locally through a model proxy server configured via `MODEL_PROXY_URL`
and `MODEL_PROXY_KEY` in the environment or `--env-file` (see
`kit/model_proxy/`).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import uuid

from kit import artifacts
from kit import build
from kit import check
from kit import export
from kit import models
from kit import protocol
from kit import spec
from kit import stack

UP_POLL_SEC = 2  # how often `kit up` asks Docker whether the server still runs
# how long Docker may stay silent before `kit up` warns (once; it keeps waiting)
UP_SILENT_SEC = 30
COMMANDS = {
    "check": (
        "build, start and smoke-test the entry as the server does, then review"
        " its artifacts"
    ),
    "validate": "check submission.yaml and .env without Docker",
    "build": "build the bundle and serve image as the server builds them",
    "up": (
        "start the entry's server as the server runs it and print its ws://"
        " address"
    ),
    "artifacts": (
        "review the entry's dev-set artifacts with the server's rules, without"
        " Docker"
    ),
    "models": "list the google-cloud models",
}


def cmd_validate(a) -> int:
  """Validates `submission.yaml` and optional `.env` without Docker.

  A missing model API key produces a warning rather than an error (since CI
  validates sample entries that ship `.env.example`), whereas `kit check` and
  the evaluation server require the key.

  Args:
    a: Parsed CLI arguments containing `dir`.

  Returns:
    Exit status code (0 on success).

  Raises:
    check.SetupError: If `.env` contains invalid or disallowed variables.
  """
  sub = spec.load(a.dir)
  has_env = os.path.lexists(pathlib.Path(a.dir) / spec.ENV_FILE)
  dotenv = check.entry_env(pathlib.Path(a.dir))
  key = sub.model.key_secret
  missing_key = bool(key) and not dotenv.get(key)
  # a missing key only warns (below), so stand in a value for it while checking
  # the file's other names
  problems = spec.env_problems(
      sub, {**dotenv, key: "set"} if missing_key else dotenv
  )
  if problems:
    raise check.SetupError(spec.ENV_FILE, "; ".join(problems))
  model = sub.model.api + (
      f" ({sub.model.label})" if sub.model.provider == "google-cloud" else ""
  )
  print(
      f"ok: {sub.name}, {model}; egress"
      f" {', '.join(sub.allowed_hosts) or 'none'}; {spec.ENV_FILE}"
      f" {', '.join(dotenv) or 'none'}"
  )
  if missing_key:
    if has_env:
      todo = f"add {key}=<your key> to {spec.ENV_FILE}"
    else:
      todo = (
          f"put {key}=<your key> in a {spec.ENV_FILE} at the submission's root"
      )
    print(
        f"warning: {todo} before kit check or submitting: {sub.model.api}"
        " needs it"
    )
  return 0


def _build(
    a, sub: spec.Submission | None = None
) -> tuple[spec.Submission, str, pathlib.Path]:
  """Builds or reuses the submission bundle and serve Docker image.

  Args:
    a: Parsed CLI arguments with `dir`, `work`, and `rebuild`.
    sub: Optional pre-loaded `spec.Submission` instance.

  Returns:
    A tuple `(submission, serve_image_tag, work_dir_path)`.
  """
  sub = sub or spec.load(a.dir)
  build.require_local_daemon()
  work = build.work_dir(sub.name, a.work, src=a.dir)
  with build.locked(work):
    prep = build.prepare(
        sub,
        build.stage(pathlib.Path(a.dir).resolve(), work / "source"),
        work,
        rebuild=a.rebuild,
    )
  how = (
      "reused, unchanged since the last build"
      if prep.reused
      else f"built in {prep.bundle.seconds:.0f} s"
  )
  print(
      f"bundle {prep.bundle.size / 2**20:.0f} MiB, {how}; serve image"
      f" {prep.image}"
  )
  return sub, prep.image, work


BUILD_INTERRUPTED = (
    "interrupted: the build is stopped, and what it had started is removed"
)


def cmd_build(a) -> int:
  """Builds the submission bundle and serve image.

  Args:
    a: Parsed CLI arguments.

  Returns:
    Exit status code (0 on success, 130 if interrupted).
  """
  # `kill` or a closed terminal stops the build as Ctrl-C does (build.build)
  with check.stops_on_signals():
    try:
      _build(a)
    except KeyboardInterrupt:
      print(BUILD_INTERRUPTED, flush=True)
      return 130
  return 0


def cmd_up(a) -> int:
  """Builds and starts the submission's server stack until stopped.

  Args:
    a: Parsed CLI arguments.

  Returns:
    Exit status code (0 on clean Ctrl-C stop, 1 if the server exits, 130 if the
    build is interrupted).
  """
  sub = spec.load(a.dir)
  # check the .env and the model server before building, so a problem shows up
  # in seconds
  rt = check.runtime(
      sub,
      check.entry_env(pathlib.Path(a.dir)),
      {**os.environ, **check.read_env_file(a.env_file)},
      a.model_base_url,
  )
  for note in rt.notes:
    print(f"warning: {note}", flush=True)
  # On Ctrl-C, `kill` or a closed terminal, build.build removes the builder
  # container, Stack.start removes a stack it had half started, and the finally
  # below stops a running one.
  with check.stops_on_signals():
    try:
      _, image, _ = _build(a, sub)
    except KeyboardInterrupt:
      print(BUILD_INTERRUPTED, flush=True)
      return 130
    check.sweep_leftovers()
    run = f"up-{uuid.uuid4().hex[:12]}"
    st = stack.Stack(keep=True, run=run)
    # a later kit run removes the stack if this process is killed
    registration = check.register_up(run)
    try:
      # docker reads it once, as it creates the container
      with check.private_env_file(rt.env) as env_file:
        st.start(
            image=image,
            command=spec.fill(sub.serve, protocol.PORT),
            env_file=str(env_file),
            allow=rt.allow,
            exceptions=rt.exceptions,
        )
      print(
          f"{sub.name} is starting at {st.url}  (logs: docker logs -f"
          f" {st.serve}; Ctrl-C stops it)",
          flush=True,
      )
      print(_until_stopped(st, sub.name), flush=True)
      return 1
    except KeyboardInterrupt:
      pass
    finally:
      st.stop()
      stack.unregister(registration)
  return 0


def _until_stopped(st: stack.Stack, name: str) -> str:
  """Polls `st` until the serve or egress container stops or is removed.

  If Docker becomes temporarily unresponsive while the containers still exist,
  continues waiting and prints a warning once per silent stretch.

  Args:
    st: Running `stack.Stack` instance to monitor.
    name: Submission name for status messages.

  Returns:
    A human-readable message describing why the server stopped.
  """
  proxy = (
      f"its egress proxy {st.egress}, which carries {st.url} and the server's"
      " connections out,"
  )
  # since when Docker has not answered, and whether kit up said so
  silent, said = None, False
  while True:
    answered = True
    for which, what in (("serve", "the server"), ("egress", proxy)):
      try:
        state = st.state(which)
      except stack.InfraError:
        state = None
      if state is not None and not state.get("Running"):
        memory = stack.MEMORY if which == "serve" else stack.EGRESS_MEMORY
        return (
            f"{name} stopped: {what} {stack.exit_wording(state, memory)}; the"
            f" end of its output:\n{stack.output_tail(st.logs(which))}"
        )
      if state is None and st.gone(which):
        gone = f"its container {st.serve}" if which == "serve" else proxy
        return (
            f"{name} stopped: {gone} no longer exists (removed outside the kit,"
            " e.g. by `docker rm` or a reset of Docker)"
        )
      answered = answered and state is not None
    if answered:
      silent, said = None, False
    else:
      silent = time.monotonic() if silent is None else silent
      if not said and time.monotonic() - silent >= UP_SILENT_SEC:
        print(
            f"warning: Docker has not answered for {UP_SILENT_SEC} s; still"
            f" waiting for {name} (Ctrl-C stops it)",
            flush=True,
        )
        said = True
    time.sleep(UP_POLL_SEC)


def cmd_check(a) -> int:
  """Runs the full submission check (build, smoke test, and artifact review).

  Args:
    a: Parsed CLI arguments.

  Returns:
    Exit status code from `check.check`.
  """
  return check.check(
      a.dir,
      a.work,
      a.env_file,
      smoke=not a.no_smoke,
      cancel_test=a.cancel_test,
      base_url=a.model_base_url,
      keep=a.keep,
      rebuild=a.rebuild,
  )


def cmd_artifacts(a) -> int:
  """Reviews the entry's `artifacts/` directory using evaluation server rules.

  When `--from` is given, first imports artifacts from a run directory into the
  entry.

  Args:
    a: Parsed CLI arguments containing `dir`, `source`, and `json`.

  Returns:
    0 if the server would accept the artifacts (warnings allowed), or 1 if it
    would reject them.
  """
  entry = pathlib.Path(a.dir).resolve()
  # --json keeps standard output to the review alone
  said = sys.stderr if a.json else sys.stdout
  if a.source:
    try:
      print(
          "\n".join(export.import_run(entry, pathlib.Path(a.source))), file=said
      )
    except export.ExportError as exc:
      print(exc, file=said)
      return 1
  try:
    rv = artifacts.review_directory(
        entry, artifacts.load_dev_set(artifacts.DEV_SET_FILE)
    )
  # the entry itself cannot be listed; the review reports unreadable files
  # inside
  except OSError as exc:
    print(
        f"could not read {exc.filename or entry}: {exc.strerror or exc}",
        file=said,
    )
    return 1
  print(json.dumps(rv.as_dict(), indent=2) if a.json else "\n".join(rv.lines()))
  return 0 if rv.accepted else 1


def cmd_models(a) -> int:
  """Lists the available `google-cloud` models and supported custom providers.

  Args:
    a: Parsed CLI arguments (unused).

  Returns:
    Exit status code (0).
  """
  del a
  width = max(map(len, models.GOOGLE_CLOUD))
  for model_id, name in models.GOOGLE_CLOUD.items():
    print(f"google-cloud/{model_id:<{width}}  {name}")
  print(
      "\nWith your own key in .env: openai/<model>, anthropic/<model>,"
      " gemini/<model>, or openai-compatible/<model> with model_base_url"
      " (docs/models.md)"
  )
  return 0


def main(argv: list[str] | None = None) -> int:
  """Parses CLI arguments and dispatches the selected kit subcommand.

  Args:
    argv: Optional list of command-line arguments.

  Returns:
    Process exit status code.
  """
  check.tolerant_output()
  ap = argparse.ArgumentParser(
      prog="python -m kit",
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sp = ap.add_subparsers(dest="cmd", required=True, metavar="command")
  parsers = {
      name: sp.add_parser(
          name, help=text, description=text[0].upper() + text[1:] + "."
      )
      for name, text in COMMANDS.items()
  }
  for name, fn in (
      ("check", cmd_check),
      ("validate", cmd_validate),
      ("build", cmd_build),
      ("up", cmd_up),
      ("artifacts", cmd_artifacts),
      ("models", cmd_models),
  ):
    p = parsers[name]
    p.set_defaults(func=fn)
    if name != "models":
      p.add_argument(
          "dir",
          help=(
              "the submission: a directory holding submission.yaml (or that"
              " file)"
          ),
      )
    if name in ("check", "build", "up"):
      p.add_argument(
          "--work",
          help=(
              "where logs and the bundle go (default ~/.cache/plebench/<name>)"
          ),
      )
      p.add_argument(
          "--rebuild",
          action="store_true",
          help=(
              "build even if nothing changed since the last build (the bundle"
              " is reused otherwise)"
          ),
      )
    if name in ("check", "up"):
      p.add_argument(
          "--env-file",
          help=(
              "NAME=value lines for this run, such as MODEL_PROXY_URL and "
              "MODEL_PROXY_KEY (your keys go in the submission's .env)"
          ),
      )
      p.add_argument(
          "--model-base-url",
          help=(
              "for an openai-compatible model: call this server instead of "
              "model_base_url, e.g. vLLM on your machine (private addresses "
              "allowed; the entry is then not shown ready to submit)"
          ),
      )
  parsers["check"].add_argument(
      "--no-smoke",
      action="store_true",
      help="skip the smoke and MCP tasks (listen and --cancel-test still run)",
  )
  parsers["check"].add_argument(
      "--cancel-test",
      action="store_true",
      help="also check that a cancelled task still ends in done",
  )
  parsers["check"].add_argument(
      "--keep",
      action="store_true",
      help=(
          "leave the containers running for inspection; the check prints how to"
          " remove them"
      ),
  )
  parsers["artifacts"].add_argument(
      "--from",
      dest="source",
      metavar="RUN_DIR",
      help=(
          "first copy a run's artifacts (runs/<name>-<time>) into the entry, as"
          " eval/run.py does"
      ),
  )
  parsers["artifacts"].add_argument(
      "--json", action="store_true", help="print the review as JSON"
  )
  a = ap.parse_args(argv)
  try:
    if getattr(a, "dir", None):
      a.dir = str(spec.entry_dir(a.dir))
    return a.func(a)
  except spec.EntryError as exc:
    problem = str(exc)
  except spec.SpecError as exc:
    problem = "submission.yaml is not valid:\n  - " + "\n  - ".join(
        exc.problems
    )
  except FileNotFoundError as exc:
    missing_spec = (
        getattr(a, "dir", None)
        and pathlib.Path(exc.filename or "").name == spec.FILE
    )
    problem = (
        f"no {spec.FILE} in {a.dir}"
        if missing_spec
        else f"not found: {exc.filename}"
    )
  except check.SetupError as exc:
    problem = f"{exc.stage}: {exc}"
  except build.BuildError as exc:
    problem = f"build failed: {exc}"
  except export.ExportError as exc:
    problem = f"artifacts: {exc}"
  except stack.InfraError as exc:
    print(f"Docker failed: {exc}", file=sys.stderr)
    return 2
  # a file that cannot be read or written: name it, without a traceback
  except OSError as exc:
    problem = f"could not use {exc.filename or 'a file'}: {exc.strerror or exc}"
  print(problem, file=sys.stderr)
  return 1


if __name__ == "__main__":
  sys.exit(main())
