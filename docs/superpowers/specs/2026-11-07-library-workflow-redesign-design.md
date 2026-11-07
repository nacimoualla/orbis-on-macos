# Game library and developer workflow redesign

Date: 2026-11-07
Status: Proposed for final review

## Outcome

Keep the native AppKit application and make it the everyday interface for importing,
finding, configuring, and launching games. Keep `scripts/run_workflow.py` as the
terminal entry point and expose the same library and import operations to both.
Preserve existing CLI commands and the current uncommitted extraction changes.

## Approach

Use AppKit for presentation and a small shared Python service layer for persistent
library data, import jobs, and launch profiles. The GUI invokes workflow commands
with argument arrays and consumes versioned JSON output/events; human-readable
CLI output remains the default. Extraction diagnostics go to stderr in JSON mode.
This avoids implementing extraction and profile semantics twice. Keeping separate
GUI/CLI implementations would be initially simpler but permit drift; replacing
AppKit would add migration work without improving the requested workflows.

## AppKit interface

- Use Auto Layout and split views with sensible minimum sizes. Follow system
  appearance and standard macOS controls.
- Provide Library, Imports, Developer, and Diagnostics navigation.
- Library offers searchable grid and list views using local `sce_sys/icon0.png`
  artwork, with a fallback image. Search matches title and title ID. Show ready,
  importing, failed, and missing states, rather than claiming game compatibility.
- A selected game's details show metadata, location, last run, launch settings,
  Launch/Stop, and Reveal in Finder. One launch action owns running state.
- Save per-game JIT, window/headless, instruction budget, and Mono mode settings.
  Display effective settings before launch; reject invalid budgets. A profile is
  identified by the canonical game root, with content/title IDs as metadata.
- Keep a resizable, collapsible console with search, category filtering, copy,
  clear, and reveal-log actions. Retain bounded in-memory output while streaming
  complete logs to the repository's `Logs/` directory.
- Imports shows queued/active/completed/failed jobs and diagnostic details. Allow
  explicit retry and importing a chosen PKG, archive, or extracted folder.
- Developer provides build runner, build GUI/app bundle, build all, run all tests,
  filtered tests, and rerun failures. Stream output and show final exit status.
  Serialize jobs that mutate the build tree; disable conflicting actions.
- Diagnostics lists recent runs/builds/tests, exit status, log paths, detected
  crash signatures, and the existing collection action. Crash classification is
  a heuristic, not a root-cause claim.

## Shared data and commands

- Store versioned library roots, cached metadata, profiles, and import receipts in
  a machine-local state directory under `games/.library/` (already gitignored).
  Use atomic writes and an exclusive import lock shared by GUI and CLI.
- Reconcile persisted entries with disk on refresh; deduplicate canonical roots
  and symlink aliases. Missing games remain visible but cannot launch.
- Factor library/import/profile operations into focused Python modules. Keep
  `run_workflow.py` responsible for argument parsing, menus, and orchestration.
- Add `library` list/refresh operations, profile read/update operations, and a
  structured import operation. Document the actual CLI syntax during implementation.
- Menu startup shows useful library/build status and direct actions for library,
  launch, import, build, test, and diagnostics. Remove hard-coded Flappy Bird
  selection from the everyday quick-launch path.
- GUI startup, CLI menu startup, and library refresh trigger import discovery.
  While the GUI is open, periodically discover new PKGs without blocking AppKit.
  Pure help, build, test, and log commands do not implicitly extract games.
- Existing run commands resolve a selected library game and use its saved profile;
  explicit CLI flags override saved settings. Existing path-based launches remain.

## Automatic PKG import and deletion

1. Discover `.pkg` files recursively under `games/`, case-insensitively. Exclude
   internal state/staging directories and do not follow directory symlinks.
2. Queue only stable files: unchanged size and modification time across scans.
   Serialize extraction and do not enqueue an active source twice.
3. Inspect the package, extract into a unique staging directory, perform required
   boot conversion, and verify readable metadata plus an ELF boot executable.
   A boot check establishes launch eligibility, not that a game will run correctly.
4. Publish the completed title and atomically persist an import receipt tying the
   source identity to the validated destination. Never overwrite an existing title
   automatically. An existing title alone is insufficient evidence for deletion.
5. Delete the original PKG only after successful publication and receipt persistence,
   and only if source size/mtime still match the imported file. Record deletion
   success or failure; a deletion failure must not hide a successful import.
6. Keep source PKGs on extraction/conversion/validation/cancellation failure. Preserve
   enough job information to retry. Do not repeatedly retry an unchanged failure
   on every scan. A crash after publication can recover cleanup from the receipt.

Automatic deletion applies to successfully imported PKGs under `games/`, as
requested. Explicit imports from outside `games/` retain their source. Archive
imports retain the original archive; temporary nested PKGs may be cleaned after
successful import. Unsupported packages, updates, and DLC that cannot produce a
standalone validated title remain available with a clear failure/status message.

## Process handling

Run extraction, scans, builds, and tests off the main thread. Launch processes
without shell interpolation. Cancellation/timeout must stop child processes and
reap them even when they emit no output. Preserve the existing runner log location,
controls guide, and input behavior. Report missing tools, corrupt state, invalid
profiles, and file errors through actionable status messages and retained logs.

## Implementation order

1. Shared library/profile state, validated import receipts, automatic PKG discovery,
   and failure-safe cleanup; expose and verify CLI operations.
2. Workflow menu and launch integration, structured process output, and timeout /
   cancellation handling.
3. AppKit split-view shell, library grid/list/search, details, saved profiles, and
   launch/stop integration.
4. Import queue, developer actions, diagnostics, and documentation; native smoke
   verification and regression checks.

These are stages of this single redesign; all are included in scope.

## Verification

Use temporary fixture libraries and mocked extraction tools for import tests.
Cover successful source deletion, extraction/conversion/metadata failure retaining
sources, duplicate/version conflicts, interrupted receipts, changing source files,
symlink deduplication, profile round trips and overrides, concurrent imports, and
silent-child timeout/cancellation. Do not use real user PKGs for deletion tests.
Run existing extraction/import regressions and CLI help checks. Build `ps4-gui`,
`ps4-runner`, and the app bundle; manually check resize, selection/search, persisted
profiles, background import progress, launch/stop, build/test output, and diagnostics.
Report any native UI checks that cannot be performed in the available environment.
