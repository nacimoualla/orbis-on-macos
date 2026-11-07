# Library Workflow Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the Python service task and review its diff; integrate the native interface in this session.

**Goal:** Build a useful native game library with automatic imports, persistent profiles, developer tools, and a Desktop shortcut to the current app build.

**Architecture:** AppKit presents the library and streams subprocess output. A Python library service supplies JSON metadata, profiles, import receipts, and the CLI. CMake assembles the app and includes its icon.

**Tech Stack:** Objective-C++ / AppKit, Python standard library, CMake, Apple iconutil.

## Global Constraints

- Preserve existing uncommitted extraction changes.
- Delete PKGs in games/ only after validated import and receipt persistence.
- Keep external source files and original archives.
- Keep the AppKit main thread responsive and logs under Logs/.

## Task 1: Shared Python library and workflow

Files: scripts/game_library.py, scripts/run_workflow.py, tests/test_game_library.py.

- [ ] Implement atomic persisted library/profile state and recursive ELF/SFO discovery.
- [ ] Add serialized staged PKG import, stable-source checks, receipts and recoverable cleanup.
- [ ] Expose library --json, library --refresh, import --json, and profile commands.
- [ ] Replace Flappy-specific quick launch; support saved settings and explicit overrides.
- [ ] Make process timeout work for silent children and terminate process groups.
- [ ] Verify fixture-only deletion/failure/profile/timeout tests and existing extraction tests.

Interface: library JSON contains games with root, eboot, title, title_id, app_ver, icon_path, status and profile; import JSON contains jobs with source, status, message. JSON goes to stdout, diagnostics stderr. Profile fields: jit(bool), window(bool), budget(nonnegative int), hle_mono(bool).

## Task 2: AppKit shell and integration

Files: src/gui/game_library_gui.mm.

- [ ] Replace fixed frames with Auto Layout and resizable library/details/console regions.
- [ ] Add searchable grid/list library, saved profiles, one launch action, reveal and refresh.
- [ ] Add import jobs, file imports, periodic discovery, build/test controls and diagnostics.
- [ ] Use NSTask argument arrays and background work; preserve complete disk logs.
- [ ] Build native targets and smoke-check GUI with available native tools.

## Task 3: Packaging and verification

Files: cmake/Info.plist.in, CMakeLists.txt, resources/AppIcon.icns, scripts/create_app_icon.swift, docs/DEVELOPMENT.md.

- [ ] Generate a native controller icon and include it in the app bundle.
- [ ] Configure and build ps4-gui-app, run Python and native import regressions.
- [ ] Create a Desktop symlink to build/ps4-gui.app without replacing unrelated files.
- [ ] Review the integrated diff and document launch/build behavior and actual checks.
