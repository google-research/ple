# Dev-set artifacts

A submission is a single commit of your repository holding two things:

1. Your harness: `submission.yaml`, `serve.py`, and the files they need ([`kit/README.md`](../kit/README.md)).
2. `artifacts/`: what your harness produced when you ran it on the public dev set ([`eval/README.md`](../eval/README.md)).

We grade the committed `artifacts/` with the dev tasks' graders to get your dev score ([How we grade them](#how-we-grade-them)). `eval/run.py` writes `artifacts/` into your entry when a run finishes:

```bash
uv run python eval/run.py ../my-harness               # runs the dev set and exports ../my-harness/artifacts/
git -C ../my-harness add -A && git -C ../my-harness commit -m "dev artifacts"
```

Commit the whole repository after a run (`git add -A`), harness and `artifacts/`
together. The exported `manifest.json` records a digest of the harness files
that made the artifacts, and the server compares it with the harness in the
commit you submit.

## Check before you submit

Three commands check your artifacts before the server does. Run them from the
root of this repository:

| Command | What it checks | What it needs |
|---|---|---|
| `uv run python -m kit artifacts ../my-harness` | The server rules below (what it refuses, what it warns about, accepted fallbacks) plus working-tree git state | `git` (takes a few seconds) |
| `uv run python -m kit check ../my-harness` | The same artifact review, plus whether `artifacts/` is committed and whether any file in it looks like an API key; ends with `ready to submit` | Docker (also builds and smoke-tests the harness) |
| `uv run python eval/run.py --artifacts ../my-harness` | Copies each task's committed files into a fresh task container and grades them just like the server's dev job, without running your harness | Docker + judge settings in [`eval/README.md`](../eval/README.md#before-you-run) |

`kit artifacts` and `eval/run.py --artifacts` read the files on disk, while the
server reads your commit. Commit once they pass, then run `kit check` to confirm
everything is committed.

## Layout

```
artifacts/
  manifest.json            written by the kit: which run produced each task's files (do not edit)
  .gitattributes           written by the kit: git stores these files byte for byte (no line-ending conversion,
                           no Git LFS filter, no $Id$ expansion, no re-encoding)
  .gitignore               written by the kit: git tracks these files even when a parent .gitignore would ignore them
                           (unless it ignores the artifacts/ folder itself)
  <task-id>/               one folder per dev task, named by its id in eval/dev_set.yaml
    final_message.md       your harness's final message for the task
    files/<path>           every file your harness created or changed under the task's deliverables, stored at its
                           container path without the leading "/"
    deleted.txt            optional: deliverable files your harness deleted, one container path per line
    trajectory.json        optional: the run's trajectory, kept for inspection and never graded
```

Each task in [`eval/dev_set.yaml`](../eval/dev_set.yaml) lists its
`deliverables`: the files and folders in the task container that its grader
reads. After a trial, the exporter copies every regular file under those paths
that your harness created or modified. For example, a run of `jobshop-scheduler`
that updated the baseline scheduler produces:

```
artifacts/jobshop-scheduler/final_message.md
artifacts/jobshop-scheduler/files/app/files/frontier__JobShop__ft/baseline/init.py
```

The exporter skips:

- Files the harness did not modify (grading starts from a clean task container)
- Git metadata (`.git/`, `.gitignore`, `.gitattributes`, `.gitmodules`) so nested files do not alter how git treats your repository
- OS metadata files (`.DS_Store`, `._*`, `__MACOSX/`)
- The second of two paths that differ only in letter case, which macOS and Windows checkouts would merge (it prints a warning)

### Re-running single tasks and recovering interrupted exports

Running a subset of tasks (`eval/run.py --tasks <id>`) updates only those task
folders and leaves the rest untouched. If a local fault (such as a full disk)
prevents a task from being measured or copied out of the container, its previous
folder is preserved. Likewise, if an export is interrupted before writing
`manifest.json` (e.g., via `Ctrl-C` or a full disk), `artifacts/` is left
unchanged. If a killed process leaves temporary `.plebench-` directories behind,
the next export cleans up any unfinished copies and restores the previous
`.plebench-old-<task>-<id>` folder (or leaves `.plebench-old-` alongside the new
folder for you to inspect or delete).

Passing `--no-export` leaves your entry's `artifacts/` untouched. Every finished
run also saves a copy in `runs/<name>-<time>/artifacts/`, and you can copy a
run's artifacts (including from an interrupted run) into your entry later:

```bash
uv run python -m kit artifacts <entry> --from <run dir>
```

## How we grade them

1. We build your harness and run the smoke task, just as `kit check` does.
2. For each dev task, we start a fresh task container, remove the paths listed
   in `deleted.txt`, copy your `files/` into place, record your
   `final_message.md`, and run the task's grader. Your harness does not run in
   this step. Your dev score is the mean over all dev tasks, and a task without
   usable artifacts scores `0`.
3. We run your harness on a few dev tasks to check that it reproduces your artifacts' scores ([reproduction check](rules.md#dev-score-and-reproduction-check)).

The server's build strips `artifacts/` (and every fallback folder name listed
below) from `/opt/harness`, so do not put harness code in folders with those
names. A harness that reads its own artifacts at build or run time (from disk,
git history, or the network) breaks the fair-play rules.

## Limits

| Scope | Limit |
|---|---|
| One deliverable file | 95 MiB (GitHub refuses files over 100 MiB) |
| One task folder | 256 MiB and 2,000 deliverable files |
| Whole `artifacts/` folder | 512 MiB and 10,000 files |
| `final_message.md` | 1 MiB (longer messages are truncated at 1 MiB) |
| `deleted.txt` | 256 KiB (only complete lines within the limit are used) |
| `trajectory.json` | 32 MiB (larger files are omitted by the exporter) |
| `manifest.json` | 4 MiB (ignored with a warning if larger) |

`trajectory.json` is never graded, so it is dropped first (largest trajectories
first, with a note) if a task folder or `artifacts/` would exceed its size cap.

The 1 GiB repository size limit ([`kit/README.md`](../kit/README.md#resource-and-protocol-limits)) includes your `.git` history. Every committed version of `artifacts/` stays in git history, so squash old commits if repeated dev runs make the repository grow. The exporter warns once your branch history passes 512 MiB.

## What the server refuses, and what it only warns about

The submit page, the server, and `kit artifacts` apply the same rules.

### Refused at submit time (no submission slot is used)

The server refuses the commit before queuing a run or counting a submission
when:

- There is no usable `artifacts/` folder (including the fallback names below).
- An archive (`artifacts.zip`, `artifacts.tar.gz`, etc.) was committed instead of a folder.
- Git LFS tracks the artifacts (the server clones without LFS, so commit the files directly).
- `artifacts/` holds more than 512 MiB or 10,000 files.
- No dev task has usable artifacts (if every task folder is invalid, the commit is refused rather than scored `0`).

### A task scores 0 (the other tasks are still graded)

A dev task scores `0` when its folder holds:

- A symbolic link or git submodule
- A deliverable file over 95 MiB, more than 2,000 deliverable files, or more than 256 MiB total
- A path with a `.` or `..` segment, a backslash, a control character, or non-UTF-8 bytes
- A path under `files/<container path>` nested deeper than 64 folders, longer than 1,024 bytes, or with a filename over 255 bytes
- Two deliverable paths that differ only in letter case
- A Git LFS pointer instead of the actual file
- Nothing to grade (no deliverable file and no non-empty `final_message.md`)

A file, symlink, or submodule placed where the task's folder should be also
scores `0`.

### Warned about, then graded as it is

The server warns, and still grades the submission, when:

- Some dev tasks have no folder in `artifacts/` (they score `0`).
- A task whose grader reads `final_message.md` has none, or an empty one.
- Files sit outside the task's `deliverables` or outside `files/` (ignored unless matched by a fallback below).
- A task folder contains `.gitignore`, `.gitattributes`, or `.gitmodules` (ignored when grading).
- `artifacts/` contains folders that are not dev tasks (ignored, with a suggestion for the closest task ID).
- A task folder looks like a raw Harbor trial folder (`agent/`, `verifier/`, `result.json`) rather than exported artifacts.
- `manifest.json` is missing or does not match the committed files (an artifact edited after export, artifacts from another version of the dev set, or tasks from an earlier version of your harness).

When only documentation (`README`, `LICENSE`, or `CHANGELOG`) changed after
you ran the dev set, the review prints a note instead of a warning.

### What only the kit checks

Because the kit reads your working tree, it also checks what a commit would
include:

- Ignored files inside `artifacts/` are skipped with a warning. If git ignores the entire `artifacts/` folder, the kit refuses it (stage it with `git add -f artifacts`).
- If a committed file on disk differs from git's object store even though
  `git status` shows no change (after changing Git LFS or line-ending settings),
  the kit warns and prints the git commands to fix it.
- If a file or folder is unreadable, the kit stops and prints the `chmod` or `chown` command needed to fix it.
- If `git status` fails in your entry (for example, repository ownership
  mismatch), the kit stops and quotes git. If the folder is not a git repository
  yet, the kit reviews the files on disk, and `kit check` waits until you
  initialize and commit the repo before reporting `ready to submit`.
- `kit check` warns if any file in `artifacts/` contains text that looks like an API key.

## Fallbacks

So that a small packaging mistake does not zero your dev score, the server
accepts these layouts with a warning (`eval/run.py` always writes the standard
layout):

| If you committed | We read it as |
|---|---|
| `Artifacts/`, `artifact/`, `dev_artifacts/`, `dev-artifacts/` | `artifacts/` |
| `runs/<run>/artifacts/` (when exactly one such folder exists) | `artifacts/` |
| One extra wrapper folder, e.g. `artifacts/my-harness/<task-id>/` | `artifacts/<task-id>/` |
| A Harbor trial folder name, e.g. `jobshop-scheduler__a1B2c3D` | `jobshop-scheduler` |
| A task ID with different letter case, underscores, or spaces in place of `-` | The matching `<task-id>` |
| `<task-id>/app/files/...` (without the `files/` level) | `<task-id>/files/app/files/...` |
| `<task-id>/init.py` or `<task-id>/files/init.py` (when the task has a single deliverable with that filename) | That deliverable's full path |
| `final_message.txt`, `final_message`, `final-message.md`, `message.md` | `final_message.md` |

If two folders map to the same task, the one spelled `<task-id>` wins; if
neither is exact, that task is refused as ambiguous. OS metadata files
(`.DS_Store`, `Thumbs.db`, `desktop.ini`, `._*`, `__MACOSX/`) are ignored
without a warning.

## `manifest.json`

`eval/run.py` and `kit artifacts --from` write and update
`artifacts/manifest.json`. Keep it as written so the server can verify which
run produced each file:

```json
{
  "format": 1,
  "dev_set": "sha256 of eval/dev_set.yaml",
  "created": "2026-10-10T12:00:00Z",
  "harness": {
    "name": "my-harness",
    "model_api": "google-cloud/claude-opus-5-5",
    "tree": "sha256 of the harness's files",
    "files": {"serve.py": "100644 git blob id"},
    "commit": "git commit of the run, or null",
    "dirty": false
  },
  "tasks": {
    "jobshop-scheduler": {
      "run": "my-harness-20261010-120000",
      "trial": "jobshop-scheduler__a1B2c3D",
      "ended": "ok",
      "reward": 0.83,
      "tree": "the harness digest of that run",
      "final_message": "git blob id",
      "files": {"app/files/frontier__JobShop__ft/baseline/init.py": "git blob id"},
      "deleted": [],
      "skipped": []
    }
  }
}
```

- `harness.tree` is a SHA-256 digest of your harness files as git records them
  (mode, blob ID, and path), excluding `.env` and artifact folders. When your
  submitted commit has the same digest, the server knows the submitted harness
  produced the export.
- `harness.files` maps each harness file to its mode and blob ID so the review can name any files modified after the run (`null` past 5,000 harness files or when the manifest would exceed 4 MiB).
- `tasks.<id>.files` and `final_message` hold the git blob IDs (`git hash-object`) of each exported file and final message, checked against the committed files.
- `tasks.<id>.reward` is your local grader's score, shown beside the server's dev score for reference and never used for ranking.
- `tasks.<id>.skipped` lists any files omitted by the exporter (such as a file over 95 MiB) and why: up to 50 rows, plus summary counts per reason.
