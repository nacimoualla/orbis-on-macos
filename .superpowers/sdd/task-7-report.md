# Task 7 report: bounded Flappy Bird GPU smoke harness

## Status

Implementation is complete. The harness, CMake/CTest registration, GPU translator access needed to observe the runner-owned translator, and roadmap evidence are included. The focused build passes. The focused smoke test fails explicitly at guest frame 1 because the retail guest reaches an unsupported CPU VEX operation before it can complete the bounded interval; it does not pass with synthetic output.

## Changes

- Added `tests/test_flappy_gpu.cpp`.
  - Resolves the retail ELF relative to `CMAKE_SOURCE_DIR`.
  - Reuses ELF loading, guest memory, VFS, syscall, dynamic-linker, JIT, Metal, PM4, and headless display initialization paths.
  - Runs exactly 120 bounded guest batches, pumps events, submits asynchronously, presents, waits for the final GPU frame, and reads the center pixel.
  - Prints exact required keys: `frames`, `draws`, `packets`,
    `unsupported_packets`, `upload_bytes`, and `elapsed_ms`.
  - Fails clearly for missing target/dependencies, guest execution dependency failures, no packets/draws, or black output.
- Registered `test_flappy_gpu` and `Flappy_GpuSmoke` in `CMakeLists.txt`.
- Added `NidRegistry::gpu_translator()` so the smoke harness reports the same PM4 translator used by GNM HLE handlers.
- Added dynamic-linker relocation diagnostics to failure paths, including
  loaded-module/trampoline state and unresolved NIDs, plus CPU/HLE/GPU runtime
  counters on bounded guest execution failures.
- Dynamic-linker relocation table read and guest-memory patch failures now
  propagate `false` with table name, table address, first failing address, and
  unread count in `m_last_error`.
- `load_module()` now checks relocation success, reports the diagnostics, and
  rolls back the module record, segment mappings, and import-slot bookkeeping
  before returning failure; exports and init are not reached.
- Added the smoke command, validated counters, and shader-signature/unsupported-PM4 evidence queue to `ROADMAP.md`.

## Validation

- `cmake --build build --target test_flappy_gpu` — **passed**.
- `ctest --test-dir build -R Flappy_GpuSmoke --output-on-failure` — **failed explicitly**, as required for an unavailable guest execution dependency.
  - Observed diagnostic: `[CPU UNIMPL VEX] map=1 op=0x7c at RIP=0x1061969`.
  - Harness result: guest execution stopped at frame 1 with `Illegal or unimplemented instruction`; no synthetic framebuffer or counter pass was accepted.
- `git diff --check` — **passed**.
- `ctest --test-dir build -R 'Flappy_GpuSmoke|Phase7_DynamicLinker' --output-on-failure`
  — **Phase7_DynamicLinker passed**; Flappy smoke failed explicitly on the
  known unsupported VEX operation.

## Review fix validation

The harness now names dynamic-linker failures as initialization versus
relocation failures and includes unresolved NIDs when available. Guest runtime
dependency failures retain the preceding CPU/HLE/GPU diagnostics and append
linker state plus PM4 packet/draw/unsupported counters. The focused CTest still
fails explicitly on the known unsupported VEX operation (`map=1 op=0x7c`).
Smoke output now uses `unsupported_packets=` and `upload_bytes=` exactly.
Phase 7 regression coverage was rerun after the module rollback fix.

## Concerns / follow-up queue

The current blocker is CPU instruction coverage (`VEX` map 1, opcode `0x7c` at
RIP `0x1061969`), before video-out/PM4 can produce a draw. The updated failure
also reports `trampolines=808`, `loaded_modules=0`, unresolved NIDs beginning
`0x3f7df43f774517af` and `0xcebd3de04437f56c`, and GPU
`packets=0 draws=0 unsupported=0`. Once that dependency is implemented, rerun
the same CMake/CTest commands and use the emitted shader signatures and
unsupported PM4 count to prioritize Flappy fidelity work. Existing unrelated
changes in `src/cpu/executor.cpp` and `tests/test_phase5.cpp` were preserved
and not staged.

## Commit

Task 7-only commit; final hash is reported in the completion summary.
