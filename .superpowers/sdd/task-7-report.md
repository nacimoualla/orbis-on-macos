# Task 7 report: bounded Flappy Bird GPU smoke harness

## Status

Implementation is complete. The harness, CMake/CTest registration, GPU translator access needed to observe the runner-owned translator, and roadmap evidence are included. The focused build passes. The focused smoke test fails explicitly at guest frame 1 because the retail guest reaches an unsupported CPU VEX operation before it can complete the bounded interval; it does not pass with synthetic output.

## Changes

- Added `tests/test_flappy_gpu.cpp`.
  - Resolves the retail ELF relative to `CMAKE_SOURCE_DIR`.
  - Reuses ELF loading, guest memory, VFS, syscall, dynamic-linker, JIT, Metal, PM4, and headless display initialization paths.
  - Runs exactly 120 bounded guest batches, pumps events, submits asynchronously, presents, waits for the final GPU frame, and reads the center pixel.
  - Prints `frames`, `draws`, `packets`, `unsupported`, `uploads`, and `elapsed_ms`.
  - Fails clearly for missing target/dependencies, guest execution dependency failures, no packets/draws, or black output.
- Registered `test_flappy_gpu` and `Flappy_GpuSmoke` in `CMakeLists.txt`.
- Added `NidRegistry::gpu_translator()` so the smoke harness reports the same PM4 translator used by GNM HLE handlers.
- Added the smoke command, validated counters, and shader-signature/unsupported-PM4 evidence queue to `ROADMAP.md`.

## Validation

- `cmake --build build --target test_flappy_gpu` — **passed**.
- `ctest --test-dir build -R Flappy_GpuSmoke --output-on-failure` — **failed explicitly**, as required for an unavailable guest execution dependency.
  - Observed diagnostic: `[CPU UNIMPL VEX] map=1 op=0x7c at RIP=0x1061969`.
  - Harness result: guest execution stopped at frame 1 with `Illegal or unimplemented instruction`; no synthetic framebuffer or counter pass was accepted.
- `git diff --check` — **passed**.

## Concerns / follow-up queue

The current blocker is CPU instruction coverage (`VEX` map 1, opcode `0x7c`), before video-out/PM4 can produce a draw. Once that dependency is implemented, rerun the same CMake/CTest commands and use the emitted shader signatures and unsupported PM4 count to prioritize Flappy fidelity work. Existing unrelated changes in `src/cpu/executor.cpp` and `tests/test_phase5.cpp` were preserved and not staged.

## Commit

Task 7-only commit; final hash is reported in the completion summary.
