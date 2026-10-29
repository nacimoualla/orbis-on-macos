# PS4 Dev Workflow Tool Enhancement Design

**Date:** 2026-10-20  
**Status:** Approved  
**Target:** `scripts/run_workflow.py`

## 1. Overview & Motivation

The `ps4-to-macos` emulator repository currently contains multiple build targets (`ps4-runner`, `ps4-gui`, and 17+ CTest test suites), sample games (such as `games/flappy_bird`), and log/diagnostic scripts (`scripts/collect_logs.py`).

The existing `scripts/run_workflow.py` is an early prototype that only looks for Python test files and root-level game files, missing integration with CMake/Ninja builds, CTest test phases, game launch configurations, and log diagnostics.

This enhancement redesigns `scripts/run_workflow.py` into a developer tool that provides:
1. **Full CLI subcommand support** for automation and scripted dev workflows.
2. **An interactive terminal menu** when invoked with zero arguments for rapid everyday use.
3. **Dedicated subsystems** for Building, Testing, Game Running, and Diagnostics.

---

## 2. Architecture & Components

The script will be organized in a modular structure within `scripts/run_workflow.py`:

```
scripts/run_workflow.py
├── TerminalUI (ANSI color palette, bold/status formatting, fallback)
├── BuildManager (CMake / Ninja integration, build types, targets, clean)
├── TestManager (CTest runner, phase regex filtering, failure output parsing)
├── GameRunner (Game asset discovery, ps4-runner / ps4-gui invocation, flags)
├── DiagnosticsManager (Recent log scans, error parsing, scripts/collect_logs.py)
├── InteractiveMenu (Numbered terminal prompt for zero-arg runs)
└── main() / CLI Argument Parser
```

### 2.1 TerminalUI
- Formats status banners (`>>> Running: ...`, `✓ Success`, `✗ Failed`, `⚠ Warning`).
- Provides clean colorized output using standard ANSI escape codes.
- Automatically disables color if stdout is not a TTY or if `NO_COLOR` environment variable is set.
- Gracefully handles optional rich styling if installed without hard dependencies.

### 2.2 BuildManager
- Interacts with `build/` directory and CMake/Ninja.
- Methods:
  - `ensure_configured(build_type="Debug")`: Checks if `build/build.ninja` or `build/Makefile` exists; if not, runs `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=<build_type>`.
  - `build(target=None, jobs=None, clean=False)`: Invokes `cmake --build build [--target TARGET] [-j JOBS]`.
  - `clean()`: Invokes `cmake --build build --target clean` or removes CMake cache if full clean requested.
- Supported targets: `ps4-runner`, `ps4-gui`, test binaries, or `all`.

### 2.3 TestManager
- Interacts with CTest and Python verification tests.
- Methods:
  - `run_tests(phase=None, verbose=False, rerun_failed=False)`:
    - Runs `ctest --test-dir build --output-on-failure`.
    - If `phase` is specified (e.g. `5`, `flappy`, `x87`), constructs regex: `ctest -R <regex>`.
    - Captures test results, summaries passed/failed counts, and displays failing test logs clearly.
  - `run_python_tests()`: Runs any Python test scripts such as `tests/test_verify_nids_adversarial.py`.

### 2.4 GameRunner
- Discovers and runs PS4 game titles or test binaries.
- Methods:
  - `discover_games()`: Scans `games/` directory and root for `eboot.bin`, `*.pkg`, or unpacked directory structures (specifically finding `games/flappy_bird`).
  - `launch_game(game_path=None, use_gui=False, headless=False, timeout=None, lldb=False)`:
    - Locates `build/ps4-runner` (or `build/ps4-gui` if `use_gui=True`).
    - Constructs execution command: `build/ps4-runner <game_path>`.
    - Supports optional timeout to prevent infinite hangs in automated testing.
    - If `lldb=True`, wraps command in `lldb -- <executable> <args>`.
    - Streams output to console and automatically logs to `Logs/runner_latest.log`.

### 2.5 DiagnosticsManager
- Methods:
  - `list_recent_logs(limit=5)`: Lists recently modified `.log` and `.txt` files in repo root and `Logs/`.
  - `show_log_summary(log_file, lines=30)`: Displays the tail and any crash signatures (e.g., `Fatal`, `SIGSEGV`, `Assertion`, `abort`) from the specified log.
  - `collect_diagnostics()`: Invokes `python3 scripts/collect_logs.py` to create a diagnostic bundle.

### 2.6 InteractiveMenu
- Activated when `scripts/run_workflow.py` is run with no arguments.
- Presents numbered choices:
  1. Quick Run (Build + Launch Flappy Bird)
  2. Build Targets (`all`, `ps4-runner`, `ps4-gui`, or tests)
  3. Run Tests (All CTests, or filter by Phase 1-17 / Flappy)
  4. Launch Game (Auto-detect from `games/` or prompt for custom path)
  5. Diagnostics & Logs (View latest log errors or run collect_logs)
  6. Clean Build Artifacts
  7. Exit
- Loops until exit, giving immediate feedback without needing to re-run the script.

---

## 3. CLI Subcommands & Options

```bash
# Build
python3 scripts/run_workflow.py build [--target TARGET] [--type {Debug,Release,RelWithDebInfo}] [--clean] [-j JOBS]

# Test
python3 scripts/run_workflow.py test [--phase PHASE] [--all] [--verbose] [--rerun-failed]

# Run / Game
python3 scripts/run_workflow.py run [--game PATH] [--gui] [--headless] [--timeout SECONDS] [--lldb]
python3 scripts/run_workflow.py game  # alias for run

# Diagnostics / Logs
python3 scripts/run_workflow.py logs [--recent] [--collect] [--show FILE]

# Clean
python3 scripts/run_workflow.py clean

# Full pipeline
python3 scripts/run_workflow.py all [--phase PHASE] [--game PATH]
```

---

## 4. Error Handling & Reliability

- **Missing Build Directory**: Automatically checks for `build/` and runs CMake configuration if needed instead of failing immediately.
- **Missing Game Assets**: Gracefully alerts user if `games/flappy_bird/eboot.bin` or requested game does not exist and lists available paths.
- **Test Failures**: Automatically surfaces `--output-on-failure` from CTest with clear colored markers rather than generic failure messages.
- **Timeouts**: Provides safe process termination and reports partial logs when tests or game runs exceed specified timeouts.

---

## 5. Verification & Testing

1. **CLI Flag Verification**:
   - Verify `--help` and subcommand `--help` for all commands.
   - Verify `build --target ps4-runner` works cleanly.
   - Verify `test --phase 1` executes only Phase 1 tests.
   - Verify `logs --recent` correctly scans and lists recent log files.
2. **Interactive Menu Verification**:
   - Verify menu renders cleanly, options respond as expected, and exit works properly.
3. **Regression Check**:
   - Verify backward compatibility for `python3 scripts/run_workflow.py test` and `python3 scripts/run_workflow.py all`.
