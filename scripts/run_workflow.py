#!/usr/bin/env python3
"""
PS4 Dev Workflow Tool

Unified developer automation script for ps4-to-macos:
- Building targets (ps4-runner, ps4-gui, test suites) via CMake & Ninja
- Running CTest test suites with phase filtering & failure log extraction
- Discovering and launching PS4 games (Flappy Bird, etc.) via ps4-runner or ps4-gui
- Inspecting recent crash logs, error analysis, and log collection
- Interactive terminal menu for quick everyday actions

Usage:
    python3 scripts/run_workflow.py                 # Interactive menu (in TTY)
    python3 scripts/run_workflow.py build [options]  # Build targets
    python3 scripts/run_workflow.py test [options]   # Run CTest / Python tests
    python3 scripts/run_workflow.py run [options]    # Launch game (alias: game)
    python3 scripts/run_workflow.py logs [options]   # Logs / diagnostics
    python3 scripts/run_workflow.py clean           # Clean build artifacts
    python3 scripts/run_workflow.py all [options]    # Full pipeline: Build -> Test -> Run
"""

import sys
import os
import subprocess
import argparse
import glob
import re
import time
import shutil
from typing import List, Dict, Optional, Tuple


# ==============================================================================
# Terminal UI & Formatting
# ==============================================================================

class TerminalUI:
    """Handles ANSI colored terminal output with graceful non-TTY fallbacks."""

    def __init__(self):
        use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        if use_color:
            self.GREEN = "\033[92m"
            self.RED = "\033[91m"
            self.YELLOW = "\033[93m"
            self.BLUE = "\033[94m"
            self.CYAN = "\033[96m"
            self.MAGENTA = "\033[95m"
            self.BOLD = "\033[1m"
            self.DIM = "\033[2m"
            self.RESET = "\033[0m"
        else:
            self.GREEN = ""
            self.RED = ""
            self.YELLOW = ""
            self.BLUE = ""
            self.CYAN = ""
            self.MAGENTA = ""
            self.BOLD = ""
            self.DIM = ""
            self.RESET = ""

    def header(self, title: str):
        width = 64
        print(f"\n{self.CYAN}{'=' * width}{self.RESET}")
        print(f"{self.BOLD}{self.CYAN} {title.center(width - 2)} {self.RESET}")
        print(f"{self.CYAN}{'=' * width}{self.RESET}")

    def section(self, title: str):
        print(f"\n{self.BOLD}── {title} ──{self.RESET}")

    def step(self, cmd_str: str, desc: str):
        print(f"\n{self.BLUE}>>> {self.BOLD}{desc}{self.RESET}")
        print(f"{self.DIM}Command: {cmd_str}{self.RESET}")

    def success(self, msg: str):
        print(f"{self.GREEN}✓ {msg}{self.RESET}")

    def fail(self, msg: str):
        print(f"{self.RED}✗ {msg}{self.RESET}")

    def warn(self, msg: str):
        print(f"{self.YELLOW}⚠ {msg}{self.RESET}")

    def info(self, msg: str):
        print(f"{self.CYAN}ℹ {msg}{self.RESET}")


ui = TerminalUI()


# ==============================================================================
# Command Execution Utility
# ==============================================================================

class CommandRunner:
    """Executes external commands with output streaming, logging, and timeouts."""

    @staticmethod
    def run(
        cmd: str,
        description: str,
        cwd: Optional[str] = None,
        log_file: Optional[str] = None,
        timeout: Optional[int] = None,
        capture: bool = False,
        env: Optional[dict] = None,
    ) -> Tuple[bool, int, str]:
        """
        Run a command, stream output to console, optionally tee to log_file,
        and report exit status.
        """
        ui.step(cmd, description)
        log_handle = None
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            log_handle = open(log_file, "a", encoding="utf-8")
            log_handle.write(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {description} ---\n")
            log_handle.write(f"Command: {cmd}\n\n")

        output_lines = []
        process = None
        try:
            # Guest code prints arbitrary bytes (Unity logs, printf of binary
            # data), so decode leniently: one stray byte must not kill the session.
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            start_time = time.time()
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    if not capture:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    output_lines.append(line)
                    if log_handle:
                        log_handle.write(line)
                        log_handle.flush()

                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    ui.warn(f"Process timed out after {timeout} seconds!")
                    if log_handle:
                        log_handle.write(f"\n[TIMEOUT after {timeout}s]\n")
                    break

            returncode = process.wait()
            success = (returncode == 0)
            if success:
                ui.success(f"{description} completed successfully.")
            else:
                ui.fail(f"{description} failed with return code {returncode}.")

            return success, returncode, "".join(output_lines)

        except Exception as e:
            ui.fail(f"Execution error: {e}")
            if process is not None and process.poll() is None:
                process.kill()  # don't leave the runner going without a reader
            return False, -1, str(e)
        finally:
            if log_handle:
                log_handle.close()


# ==============================================================================
# Build Subsystem (CMake / Ninja)
# ==============================================================================

class BuildManager:
    """Manages CMake configuration, building specific targets, and cleaning."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.build_dir = os.path.join(root_dir, "build")

    def is_configured(self) -> bool:
        """Check whether the build directory is already configured."""
        if not os.path.isdir(self.build_dir):
            return False
        has_ninja = os.path.isfile(os.path.join(self.build_dir, "build.ninja"))
        has_make = os.path.isfile(os.path.join(self.build_dir, "Makefile"))
        has_cache = os.path.isfile(os.path.join(self.build_dir, "CMakeCache.txt"))
        return (has_ninja or has_make) and has_cache

    def configure(self, build_type: str = "Debug", generator: str = "Ninja") -> bool:
        """Run CMake configuration in the build directory."""
        ui.info(f"Configuring CMake with generator {generator} ({build_type})...")
        os.makedirs(self.build_dir, exist_ok=True)
        # Check if ninja exists
        if generator == "Ninja" and not shutil.which("ninja"):
            ui.warn("Ninja not found in PATH, falling back to default generator.")
            cmd = f"cmake -B {self.build_dir} -DCMAKE_BUILD_TYPE={build_type}"
        else:
            cmd = f"cmake -B {self.build_dir} -G {generator} -DCMAKE_BUILD_TYPE={build_type}"
        
        success, _, _ = CommandRunner.run(cmd, "CMake Configure", cwd=self.root_dir)
        return success

    def clean(self) -> bool:
        """Clean build targets."""
        ui.section("Cleaning Build Artifacts")
        if not self.is_configured():
            ui.warn("Build directory not configured; nothing to clean.")
            return True
        cmd = f"cmake --build {self.build_dir} --target clean"
        success, _, _ = CommandRunner.run(cmd, "Clean Build", cwd=self.root_dir)
        return success

    def build(
        self,
        target: Optional[str] = None,
        clean_first: bool = False,
        jobs: Optional[int] = None,
        build_type: str = "Debug",
    ) -> bool:
        """Build target or all targets."""
        ui.header("Building PS4-to-macOS")

        if clean_first:
            self.clean()

        if not self.is_configured():
            if not self.configure(build_type=build_type):
                ui.fail("Failed to configure CMake.")
                return False

        cmd_parts = [f"cmake --build {self.build_dir}"]
        if target:
            cmd_parts.append(f"--target {target}")
        if jobs:
            cmd_parts.append(f"-j {jobs}")

        cmd = " ".join(cmd_parts)
        desc = f"Build target '{target}'" if target else "Build all targets"
        success, _, _ = CommandRunner.run(cmd, desc, cwd=self.root_dir)
        return success


# ==============================================================================
# Test Subsystem (CTest / Python Tests)
# ==============================================================================

class TestManager:
    """Manages test runs via CTest with phase regex filtering and log diagnostics."""

    PHASE_MAP = {
        "1": "Phase1_Types",
        "2": "Phase2_ElfLoader",
        "3": "Phase3_VirtualMemory",
        "4": "Phase4_SyscallTable",
        "5": "Phase5_CpuExecutor",
        "6": "Phase6_MetalBackend",
        "7": "Phase7_DynamicLinker",
        "8": "Phase8_DisplayWindow",
        "9": "Phase9_Threading",
        "10": "Phase10_GpuCommandProcessor",
        "11": "Phase11_AudioOut",
        "12": "Phase12_VirtualFileSystem",
        "13": "Phase13_JitCompiler",
        "14": "Phase14_MeshPipeline",
        "15": "Phase15_JitExecutorIntegration",
        "16": "Phase16_GameImport",
        "17": "Phase17_X87",
        "flappy": "Flappy_GpuSmoke",
        "smoke": "Flappy_GpuSmoke",
        "x87": "Phase17_X87",
    }

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.build_dir = os.path.join(root_dir, "build")

    def run_tests(
        self,
        phase: Optional[str] = None,
        verbose: bool = False,
        rerun_failed: bool = False,
        timeout: Optional[int] = None,
        run_python: bool = False,
    ) -> bool:
        """Run CTest suite with optional phase filter and failure outputs."""
        ui.header("Running Test Suite")

        if not os.path.isdir(self.build_dir):
            ui.fail(f"Build directory not found at {self.build_dir}. Please build first.")
            return False

        cmd_parts = [f"ctest --test-dir {self.build_dir}", "--output-on-failure"]

        if phase:
            phase_lower = phase.strip().lower()
            regex = self.PHASE_MAP.get(phase_lower, phase)
            cmd_parts.append(f"-R '{regex}'")
            ui.info(f"Targeting test filter: {regex}")

        if verbose:
            cmd_parts.append("-V")

        if rerun_failed:
            cmd_parts.append("--rerun-failed")

        cmd = " ".join(cmd_parts)
        ctest_log = os.path.join(self.root_dir, "Logs", "test_latest.log")
        ctest_success, _, _ = CommandRunner.run(
            cmd,
            "CTest Execution",
            cwd=self.root_dir,
            log_file=ctest_log,
            timeout=timeout,
        )

        py_success = True
        if run_python:
            py_success = self.run_python_tests()

        overall = ctest_success and py_success
        if overall:
            ui.success("All tests passed!")
        else:
            ui.fail("One or more tests failed.")
            log_candidate = None
            if os.path.isfile(ctest_log) and os.path.getsize(ctest_log) > 0:
                log_candidate = ctest_log
            else:
                last_test = os.path.join(self.build_dir, "Testing", "Temporary", "LastTest.log")
                if os.path.isfile(last_test):
                    log_candidate = last_test

            if log_candidate:
                self._triage_log(log_candidate, "test")
        return overall

    def _triage_log(self, log_path: str, context: str) -> None:
        try:
            scripts_dir = os.path.join(self.root_dir, "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from triage_crash import triage_log
            triage_res = triage_log(log_path, repo_dir=self.root_dir)
            card = triage_res.get("card", "")
            card_path = os.path.join(self.root_dir, "Logs", f"triage_last_{context}.txt")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(card)
            ui.section(f"Crash / Failure Triage for {context} (First 20 Lines)")
            for l in card.splitlines()[:20]:
                print(l)
            ui.info(f"Full triage card saved to: {os.path.relpath(card_path, self.root_dir)}")
        except Exception as e:
            ui.warn(f"Failed to run triage on {log_path}: {e}")

    def run_python_tests(self) -> bool:
        """Discover and execute Python verification tests."""
        ui.section("Running Python Verification Tests")
        tests_dir = os.path.join(self.root_dir, "tests")
        py_tests = glob.glob(os.path.join(tests_dir, "test_*.py"))
        if not py_tests:
            ui.info("No Python tests (test_*.py) found in tests/")
            return True

        all_ok = True
        for pt in py_tests:
            rel_pt = os.path.relpath(pt, self.root_dir)
            cmd = f"python3 {rel_pt}"
            ok, _, _ = CommandRunner.run(cmd, f"Python Test: {rel_pt}", cwd=self.root_dir)
            if not ok:
                all_ok = False
        return all_ok


# ==============================================================================
# Game Discovery & Runner Subsystem
# ==============================================================================

class GameRunner:
    """Discovers PS4 game assets and launches ps4-runner or ps4-gui."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.build_dir = os.path.join(root_dir, "build")
        self.logs_dir = os.path.join(root_dir, "Logs")

    def discover_games(self) -> List[Dict[str, str]]:
        """Search for eboot.bin, eboot.elf, or .pkg in games/ and root."""
        candidates = []
        games_dir = os.path.join(self.root_dir, "games")

        # 1. Search games/
        if os.path.isdir(games_dir):
            for root, _, files in os.walk(games_dir):
                for f in files:
                    if f.lower() in ("eboot.bin", "eboot.elf") or f.lower().endswith(".pkg"):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, self.root_dir)
                        # Infer game title from folder name
                        parent_name = os.path.basename(os.path.dirname(full_path))
                        if parent_name.startswith("ED") or parent_name.startswith("CUSA"):
                            # Check grandparent
                            grandparent = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                            title = f"{grandparent} ({parent_name})"
                        else:
                            title = parent_name if parent_name != "games" else f
                        candidates.append({
                            "title": title,
                            "path": rel_path,
                            "filename": f,
                        })

        # 2. Check root directory
        for f in ["eboot.bin", "eboot.elf"]:
            root_f = os.path.join(self.root_dir, f)
            if os.path.isfile(root_f):
                candidates.append({
                    "title": "Root Game",
                    "path": f,
                    "filename": f,
                })

        return candidates

    def find_executable(self, use_gui: bool = False) -> Optional[str]:
        """Find the path to ps4-runner or ps4-gui."""
        target_name = "ps4-gui" if use_gui else "ps4-runner"
        possible_paths = [
            os.path.join(self.build_dir, target_name),
            os.path.join(self.build_dir, "bin", target_name),
            os.path.join(self.root_dir, target_name),
        ]
        for p in possible_paths:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return None

    def launch(
        self,
        game_path: Optional[str] = None,
        use_gui: bool = False,
        headless: bool = False,
        window: bool = False,
        budget: Optional[int] = None,
        timeout: Optional[int] = None,
        lldb: bool = False,
        trace: bool = False,
        strict_hle: bool = False,
        no_jit: bool = False,
        hle_mono: bool = False,
    ) -> bool:
        """Launch game with runner. Unity titles run their own Mono PRX modules
        (PS4_LOAD_PRX=1) unless hle_mono asks for the HLE Mono shim."""
        ui.header("Launching PS4 Game / Application")

        runner_exe = self.find_executable(use_gui=use_gui)
        if not runner_exe:
            ui.fail(f"Executable for {'ps4-gui' if use_gui else 'ps4-runner'} not found in build directory.")
            ui.info("Please build the target first: python3 scripts/run_workflow.py build")
            return False

        if not game_path:
            # Auto-detect
            games = self.discover_games()
            if not games:
                ui.fail("No game assets (eboot.bin/pkg) found under games/ or root directory.")
                return False
            # Default to Flappy Bird if available
            flappy = next((g for g in games if "flappy" in g["path"].lower()), games[0])
            game_path = flappy["path"]
            ui.info(f"Auto-selected game: {flappy['title']} ({game_path})")

        if not os.path.exists(os.path.join(self.root_dir, game_path)):
            ui.fail(f"Game file does not exist: {game_path}")
            return False

        # Build runner flags
        cmd_parts = []
        if lldb:
            cmd_parts.extend(["lldb", "--"])

        cmd_parts.append(os.path.relpath(runner_exe, self.root_dir))

        if not use_gui:
            if window:
                cmd_parts.append("--window")
            elif headless:
                cmd_parts.append("--headless")
            
            if budget is not None:
                cmd_parts.append(f"--budget {budget}")

            if trace:
                cmd_parts.append("--trace")

            if strict_hle:
                cmd_parts.append("--strict-hle")

            if no_jit:
                cmd_parts.append("--no-jit")

        cmd_parts.append(f'"{game_path}"')
        cmd = " ".join(cmd_parts)

        env = os.environ.copy()
        if hle_mono:
            env.pop("PS4_LOAD_PRX", None)
        else:
            env["PS4_LOAD_PRX"] = "1"
        ui.info(f"Mono: {'HLE shim' if hle_mono else 'real (PS4_LOAD_PRX=1)'}")

        latest_log = os.path.join(self.logs_dir, "runner_latest.log")
        success, _, _ = CommandRunner.run(
            cmd,
            f"Launch {os.path.basename(game_path)}",
            cwd=self.root_dir,
            log_file=latest_log,
            timeout=timeout,
            env=env,
        )
        if success:
            ui.success(f"Game session ended. Full log teed to: {os.path.relpath(latest_log, self.root_dir)}")
        else:
            ui.fail("Game session exited with an error.")
            if os.path.isfile(latest_log) and os.path.getsize(latest_log) > 0:
                self._triage_log(latest_log, "run")
        return success

    def _triage_log(self, log_path: str, context: str) -> None:
        try:
            scripts_dir = os.path.join(self.root_dir, "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from triage_crash import triage_log
            triage_res = triage_log(log_path, repo_dir=self.root_dir)
            card = triage_res.get("card", "")
            card_path = os.path.join(self.logs_dir, f"triage_last_{context}.txt")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(card)
            ui.section(f"Crash / Failure Triage for {context} (First 20 Lines)")
            for l in card.splitlines()[:20]:
                print(l)
            ui.info(f"Full triage card saved to: {os.path.relpath(card_path, self.root_dir)}")
        except Exception as e:
            ui.warn(f"Failed to run triage on {log_path}: {e}")


# ==============================================================================
# Diagnostics & Log Inspection Subsystem
# ==============================================================================

class DiagnosticsManager:
    """Manages log searching, error inspection, and collect_logs invocation."""

    CRASH_KEYWORDS = [
        "SIGSEGV", "Fatal", "Assertion", "abort()", "Null pointer",
        "Unhandled opcode", "panic", "Segmentation fault", "Bus error",
    ]

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.logs_dir = os.path.join(root_dir, "Logs")

    def find_recent_logs(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Find the most recently modified log and text files."""
        candidates = []
        search_dirs = [self.root_dir, self.logs_dir]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.scandir(d):
                if entry.is_file() and entry.name.endswith((".log", ".txt")):
                    candidates.append((entry.path, entry.stat().st_mtime))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]

    def inspect_log(self, log_path: str, lines_to_show: int = 30):
        """Display summary and detected error patterns from a log file."""
        ui.section(f"Log Inspection: {os.path.relpath(log_path, self.root_dir)}")
        if not os.path.isfile(log_path):
            ui.fail(f"File not found: {log_path}")
            return

        size = os.path.getsize(log_path)
        ui.info(f"File Size: {size / 1024:.1f} KB")

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.readlines()

            # Search for crash signatures
            flagged = []
            for i, line in enumerate(content):
                for kw in self.CRASH_KEYWORDS:
                    if kw.lower() in line.lower():
                        flagged.append((i + 1, line.strip()))
                        break

            if flagged:
                ui.warn(f"Found {len(flagged)} potential issue/crash signature(s):")
                for line_num, line_str in flagged[:15]:
                    print(f"  {ui.RED}[Line {line_num}]{ui.RESET} {line_str}")
                if len(flagged) > 15:
                    print(f"  ... and {len(flagged) - 15} more.")
            else:
                ui.success("No common crash signatures found.")

            ui.section(f"Last {lines_to_show} Lines")
            tail = content[-lines_to_show:] if len(content) > lines_to_show else content
            for line in tail:
                sys.stdout.write(line)

        except Exception as e:
            ui.fail(f"Failed to read log file: {e}")

    def run_collector(self, log_file: Optional[str] = None):
        """Invoke scripts/collect_logs.py."""
        collector_script = os.path.join(self.root_dir, "scripts", "collect_logs.py")
        if not os.path.isfile(collector_script):
            ui.fail("scripts/collect_logs.py not found.")
            return

        cmd = f"python3 scripts/collect_logs.py"
        if log_file:
            cmd += f" --log {log_file}"

        CommandRunner.run(cmd, "Run PM4 Log Collector", cwd=self.root_dir)


# ==============================================================================
# Interactive Terminal Menu
# ==============================================================================

class InteractiveMenu:
    """Provides a user-friendly console menu for rapid everyday development."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.builder = BuildManager(root_dir)
        self.tester = TestManager(root_dir)
        self.runner = GameRunner(root_dir)
        self.diagnostics = DiagnosticsManager(root_dir)

    def run(self):
        """Display the interactive loop."""
        while True:
            ui.header("PS4-to-macOS Developer Workflow")
            print(f"  {ui.BOLD}[1]{ui.RESET} Quick Run: Build & Launch Flappy Bird (Budget: 10M)")
            print(f"  {ui.BOLD}[2]{ui.RESET} Build Targets (ps4-runner, ps4-gui, tests, clean)")
            print(f"  {ui.BOLD}[3]{ui.RESET} Run CTest Suite (All or choose Phase 1-17 / Flappy)")
            print(f"  {ui.BOLD}[4]{ui.RESET} Launch Game / App (Flappy Bird or custom eboot)")
            print(f"  {ui.BOLD}[5]{ui.RESET} Inspect / Collect Diagnostics & Logs")
            print(f"  {ui.BOLD}[6]{ui.RESET} Clean Build Artifacts")
            print(f"  {ui.BOLD}[q]{ui.RESET} Exit")
            print(f"{ui.CYAN}{'─' * 64}{ui.RESET}")

            try:
                choice = input(f"{ui.BOLD}Select an option [1-6, q]: {ui.RESET}").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if choice in ("q", "quit", "exit"):
                print("Goodbye!")
                break
            elif choice == "1":
                self._quick_run()
            elif choice == "2":
                self._menu_build()
            elif choice == "3":
                self._menu_test()
            elif choice == "4":
                self._menu_launch()
            elif choice == "5":
                self._menu_diagnostics()
            elif choice == "6":
                self.builder.clean()
            else:
                ui.warn("Invalid choice, please select 1-6 or q.")

            input(f"\n{ui.DIM}Press Enter to return to the menu...{ui.RESET}")

    def _quick_run(self):
        ui.section("Quick Run: Build & Launch Flappy Bird")
        if self.builder.build(target="ps4-runner"):
            self.runner.launch(budget=10000000)

    def _menu_build(self):
        ui.section("Build Targets")
        print("  [1] Build All")
        print("  [2] Build ps4-runner")
        print("  [3] Build ps4-gui")
        print("  [4] Rebuild (Clean + Build All)")
        sub = input("Select build option [1-4, Enter=1]: ").strip()
        if sub == "2":
            self.builder.build(target="ps4-runner")
        elif sub == "3":
            self.builder.build(target="ps4-gui")
        elif sub == "4":
            self.builder.build(clean_first=True)
        else:
            self.builder.build()

    def _menu_test(self):
        ui.section("Test Execution")
        print("  [1] Run All Tests")
        print("  [2] Run Flappy GPU Smoke Test")
        print("  [3] Run Specific Phase (1-17, x87)")
        print("  [4] Re-run Failed Tests Only")
        sub = input("Select test option [1-4, Enter=1]: ").strip()
        if sub == "2":
            self.tester.run_tests(phase="flappy")
        elif sub == "3":
            phase_num = input("Enter Phase number or name (e.g. 5, 10, x87): ").strip()
            if phase_num:
                self.tester.run_tests(phase=phase_num)
        elif sub == "4":
            self.tester.run_tests(rerun_failed=True)
        else:
            self.tester.run_tests()

    def _menu_launch(self):
        ui.section("Launch Game")
        games = self.runner.discover_games()
        if not games:
            ui.warn("No games detected.")
            return

        print("Available games:")
        for idx, g in enumerate(games, 1):
            print(f"  [{idx}] {g['title']} ({g['path']})")

        sel = input(f"Choose game [1-{len(games)}, Enter=1]: ").strip()
        chosen = games[0]
        if sel.isdigit() and 1 <= int(sel) <= len(games):
            chosen = games[int(sel) - 1]

        mode = input("Run mode: [1] macOS Window (default)  [2] Console Headless: ").strip()
        window = (mode != "2")
        budget_str = input("Instruction budget [default=unlimited for window, 10M for headless]: ").strip()
        budget = int(budget_str) if budget_str.isdigit() else (None if window else 10000000)
        mono = input("Mono: [1] Real Mono PRX (default)  [2] HLE shim: ").strip()

        self.runner.launch(game_path=chosen["path"], window=window, budget=budget, hle_mono=(mono == "2"))

    def _menu_diagnostics(self):
        ui.section("Diagnostics & Logs")
        logs = self.diagnostics.find_recent_logs(limit=6)
        if not logs:
            ui.warn("No log files found.")
            return

        print("Recent logs:")
        for idx, (path, mtime) in enumerate(logs, 1):
            rel = os.path.relpath(path, self.root_dir)
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            print(f"  [{idx}] {rel}  ({t_str})")

        sub = input(f"Select log to inspect [1-{len(logs)}, c=Run collect_logs, Enter=1]: ").strip()
        if sub.lower() == "c":
            self.diagnostics.run_collector()
        else:
            idx = int(sub) if (sub.isdigit() and 1 <= int(sub) <= len(logs)) else 1
            self.diagnostics.inspect_log(logs[idx - 1][0])


# ==============================================================================
# Pipeline Execution (All)
# ==============================================================================

def run_pipeline(root_dir: str, phase: Optional[str] = None, game_path: Optional[str] = None):
    """Executes the complete pipeline: Build -> Test -> Launch."""
    ui.header("Full Workflow: Build -> Test -> Launch")
    builder = BuildManager(root_dir)
    tester = TestManager(root_dir)
    runner = GameRunner(root_dir)

    ui.section("Step 1: Build")
    if not builder.build():
        ui.fail("Pipeline stopped: Build failed.")
        sys.exit(1)

    ui.section("Step 2: Test")
    if not tester.run_tests(phase=phase):
        ui.fail("Pipeline stopped: Tests failed.")
        sys.exit(1)

    ui.section("Step 3: Launch")
    runner.launch(game_path=game_path)
    ui.header("Pipeline Completed Successfully")


# ==============================================================================
# CLI Argument Parser & Entry Point
# ==============================================================================

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="PS4-to-macOS Developer Workflow Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--menu", "-m", action="store_true", help="Launch interactive menu")

    subparsers = parser.add_subparsers(dest="command", help="Workflow command to run")

    # --- Build subcommand ---
    p_build = subparsers.add_parser("build", help="Build targets via CMake & Ninja")
    p_build.add_argument("--target", "-t", type=str, default=None, help="Target name (e.g. ps4-runner, ps4-gui)")
    p_build.add_argument("--clean", "-c", action="store_true", help="Clean before building")
    p_build.add_argument("--type", choices=["Debug", "Release", "RelWithDebInfo"], default="Debug", help="Build type")
    p_build.add_argument("--jobs", "-j", type=int, default=None, help="Parallel build jobs")

    # --- Test subcommand ---
    p_test = subparsers.add_parser("test", help="Run CTest test suite")
    p_test.add_argument("--phase", "-p", type=str, default=None, help="Phase number or regex (e.g. 5, flappy, x87)")
    p_test.add_argument("--all", "-a", action="store_true", help="Run all tests including Python verification")
    p_test.add_argument("--verbose", "-v", action="store_true", help="Verbose test output")
    p_test.add_argument("--rerun-failed", action="store_true", help="Re-run only previously failed tests")
    p_test.add_argument("--timeout", type=int, default=None, help="CTest timeout in seconds")

    # --- Run / Game subcommand ---
    p_run = subparsers.add_parser("run", aliases=["game"], help="Launch PS4 game / app")
    p_run.add_argument("game_path", nargs="?", default=None, help="Path to eboot.bin, elf, or pkg")
    p_run.add_argument("--gui", action="store_true", help="Launch with ps4-gui instead of ps4-runner")
    p_run.add_argument("--window", "-w", action="store_true", help="Open macOS interactive display window")
    p_run.add_argument("--headless", action="store_true", help="Run in headless console mode")
    p_run.add_argument("--budget", "-b", type=int, default=None, help="Headless instruction budget (0=unlimited)")
    p_run.add_argument("--timeout", type=int, default=None, help="Execution timeout in seconds")
    p_run.add_argument("--lldb", action="store_true", help="Wrap runner in LLDB debugger")
    p_run.add_argument("--trace", "-t", action="store_true", help="Trace guest instruction execution")
    p_run.add_argument("--strict-hle", action="store_true", help="Fail-loud strict HLE mode")
    p_run.add_argument("--no-jit", action="store_true", help="Disable ARM64 JIT recompiler")
    p_run.add_argument("--hle-mono", action="store_true",
                       help="Use the HLE Mono shim instead of the game's own Mono PRX modules")

    # --- Logs subcommand ---
    p_logs = subparsers.add_parser("logs", aliases=["diagnose"], help="Inspect logs and crash diagnostics")
    p_logs.add_argument("--recent", "-r", action="store_true", help="List recent log files")
    p_logs.add_argument("--inspect", "-i", type=str, default=None, help="Inspect specific log file for errors")
    p_logs.add_argument("--collect", "-c", action="store_true", help="Run PM4 log collector script")

    # --- Clean subcommand ---
    subparsers.add_parser("clean", help="Clean build artifacts")

    # --- All subcommand ---
    p_all = subparsers.add_parser("all", help="Run pipeline: Build -> Test -> Launch")
    p_all.add_argument("--phase", type=str, default=None, help="Test phase filter for pipeline")
    p_all.add_argument("--game", type=str, default=None, help="Game path for pipeline")

    args = parser.parse_args()

    # Determine execution flow
    if args.menu or (not args.command and sys.stdin.isatty()):
        menu = InteractiveMenu(root_dir)
        menu.run()
        return

    if not args.command:
        parser.print_help()
        return

    if args.command == "build":
        builder = BuildManager(root_dir)
        success = builder.build(
            target=args.target,
            clean_first=args.clean,
            jobs=args.jobs,
            build_type=args.type,
        )
        sys.exit(0 if success else 1)

    elif args.command == "test":
        tester = TestManager(root_dir)
        success = tester.run_tests(
            phase=args.phase,
            verbose=args.verbose,
            rerun_failed=args.rerun_failed,
            timeout=args.timeout,
            run_python=args.all,
        )
        sys.exit(0 if success else 1)

    elif args.command in ("run", "game"):
        runner = GameRunner(root_dir)
        success = runner.launch(
            game_path=args.game_path,
            use_gui=args.gui,
            window=args.window,
            headless=args.headless,
            budget=args.budget,
            timeout=args.timeout,
            lldb=args.lldb,
            trace=args.trace,
            strict_hle=args.strict_hle,
            no_jit=args.no_jit,
            hle_mono=args.hle_mono,
        )
        sys.exit(0 if success else 1)

    elif args.command in ("logs", "diagnose"):
        diag = DiagnosticsManager(root_dir)
        if args.collect:
            diag.run_collector(log_file=args.inspect)
        elif args.inspect:
            diag.inspect_log(args.inspect)
        else:
            # Show recent logs
            recent = diag.find_recent_logs(limit=8)
            ui.section("Recent Logs & Diagnostics")
            for idx, (path, mtime) in enumerate(recent, 1):
                rel = os.path.relpath(path, root_dir)
                t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                size_kb = os.path.getsize(path) / 1024
                print(f"  [{idx}] {rel}  ({size_kb:.1f} KB, {t_str})")
            if recent:
                print("\nTo inspect a log file for crash signatures:")
                print(f"  python3 scripts/run_workflow.py logs --inspect {os.path.relpath(recent[0][0], root_dir)}")

    elif args.command == "clean":
        builder = BuildManager(root_dir)
        success = builder.clean()
        sys.exit(0 if success else 1)

    elif args.command == "all":
        run_pipeline(root_dir, phase=args.phase, game_path=args.game)


if __name__ == "__main__":
    main()
