# Task 8 validation report

## Final whole-branch review fix pass

Implemented the required Critical/Important fixes:

- `src/main.cpp` now brackets each interactive guest slice with
  `Pm4Translator::begin_frame()` / `submit_frame(false)` and stops with an
  explicit failure status if either operation fails.
- `DisplayWindow::present_frame()` no longer waits for producer GPU work on the
  CPU. Producer and presenter command buffers remain ordered by their shared
  Metal queue; drawable starvation defers presentation.
- Production vertex/index and texture caches now compare guest-memory snapshots
  on every lookup, so direct guest writes invalidate stale resources without
  requiring a separate notification. Texture decoding now handles explicit
  RGBA8/BGRA8 layouts and row pitch, and rejects unsupported formats.
- Added VEX map 1 opcode `0x7c` (`VHADDPS`) with 128/256-bit lane behavior and a
  Phase 5 regression test. Other unsupported VEX instructions still fail
  explicitly.
- Added a Phase 14 regression proving cache refresh after an unannounced guest
  write. The strict Flappy smoke test was not weakened.

### Validation for this pass

- `cmake --build build --target test_phase5 test_phase6 test_phase8 test_phase10 test_phase14 test_flappy_gpu ps4-runner` — **passed**
- `ctest --test-dir build -R 'Phase5_CpuExecutor|Phase6_MetalBackend|Phase8_DisplayWindow|Phase10_GpuCommandProcessor|Phase14_MeshPipeline|Flappy_GpuSmoke' --output-on-failure` — **5 passed, 1 failed**; Flappy stopped at the next explicit dependency, VEX map 1 opcode `0x13` at RIP `0x106196d`, before any GPU packets/draws.
- `cmake --build build --target test_phase5 test_phase14 ps4-runner && ctest --test-dir build -R 'Phase5_CpuExecutor|Phase14_MeshPipeline' --output-on-failure` — **passed, 2/2**
- `ctest --test-dir build --output-on-failure` — **16 passed, 2 failed**. Remaining failures are strict Flappy smoke (unsupported VEX `0x13`, no GPU work reached) and the pre-existing machine-sensitive Phase 15 JIT benchmark (37.94 MIPS vs required 50 MIPS).
- `git diff --check` — **passed**

Date: 2026-10-20
Base: `997050c6dc9ef85b1d02b494c5c784bcb317d1b1` (`997050c`)

## Build

Command:

```sh
cmake --build build --target test_phase6 test_phase8 test_phase10 test_phase14 test_flappy_gpu ps4-runner
```

Result: **PASS** (all 7 build/link steps completed; no compiler warnings were emitted).

## Focused GPU suite

Command:

```sh
ctest --test-dir build -R 'Phase6_MetalBackend|Phase8_DisplayWindow|Phase10_GpuCommandProcessor|Phase14_MeshPipeline|Flappy_GpuSmoke' --output-on-failure
```

Result: **4 passed, 1 failed**.

- `Phase6_MetalBackend`: passed (0.54 s)
- `Phase8_DisplayWindow`: passed (0.61 s)
- `Phase10_GpuCommandProcessor`: passed (0.47 s)
- `Phase14_MeshPipeline`: passed (0.48 s)
- `Flappy_GpuSmoke`: failed at frame 1 (0.24 s)

The Flappy failure is the expected explicit missing-dependency path. The preserved diagnostic identifies `CPU UNIMPL VEX map=1 op=0x7c at RIP=0x1061969`; no GPU packets or draws were reached (`GPU packets=0 draws=0 unsupported=0`). The test correctly does not weaken the smoke interval or claim performance evidence when guest execution stops before rendering.

## Full CTest suite

Command:

```sh
ctest --test-dir build --output-on-failure
```

Result: **16 passed, 2 failed** (89%).

- All CPU, loader, memory, audio, display, linker, GPU, import, and x87 tests passed.
- `Phase15_JitExecutorIntegration`: pre-existing machine-sensitive throughput assertion failed: `JIT Throughput: 40.1206 MIPS`, while the test requires `> 50.0 MIPS`.
- `Flappy_GpuSmoke`: failed at frame 1 for the same explicit unsupported VEX dependency above.

## Flappy smoke performance evidence

Commands (two independent runs):

```sh
ctest --test-dir build -R '^Flappy_GpuSmoke$' --output-on-failure
ctest --test-dir build -R '^Flappy_GpuSmoke$' --output-on-failure
```

Results:

1. Failed in 0.27 s at frame 1. Runtime: `GPU packets=0 draws=0 unsupported=0`.
2. Failed in 0.26 s at frame 1. Runtime: `GPU packets=0 draws=0 unsupported=0`.

The smoke test did not print elapsed time, draw count, upload bytes, or unsupported packet count because it intentionally exits before the 120-frame interval. No frame-interval evidence can be claimed; the 16.67 ms interactive criterion is therefore not measurable. The next implementation item is the named CPU dependency: VEX map 1 opcode `0x7c` at guest RIP `0x1061969`.

## Source hygiene

Command:

```sh
git diff --check
git status --short
```

Result: **PASS** for whitespace checks. Existing unrelated worktree changes remain untouched:

```text
 M src/cpu/executor.cpp
 M tests/test_phase5.cpp
?? docs/superpowers/plans/2026-10-20-flappy-gpu-rendering.md
```

No GPU source or test changes were required by validation, and no validation fix was committed. Validation commit range: `997050c..997050c` (empty; HEAD remains `997050c`).
