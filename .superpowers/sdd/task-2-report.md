# Task 2 Report

## Scope

Implemented translator-owned frame lifecycle and batched PM4 indexed/auto draws through
the active Metal render encoder. Draws no longer create, commit, or wait on per-draw
command buffers. Added frame statistics and the two-draw regression test. Phase 10
tests were updated to use the required translator frame lifecycle.

## Validation

* `cmake --build build --target test_phase10 test_phase14` — passed.
* `ctest --test-dir build -R 'Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure` — passed; 2/2 tests passed.
* `git diff --check` — passed.
* Self-review confirmed `pm4_translator.mm` contains no per-draw `commandBuffer`,
  `commit`, `waitUntilCompleted`, or `endEncoding` calls. The regression test
  confirms two indexed draws submit exactly one command buffer and produce a
  non-black pixel.

## Commit

`b659754 feat: batch PM4 draws into one Metal frame`

## Concerns

The working tree retains unrelated pre-existing CPU/test changes and an untracked
`docs/superpowers/plans/` directory; they were not staged or modified by Task 2.

## Stats Fix Follow-up

Fixed `command_buffers_submitted` to report the per-frame delta by recording
`submitted_frame_count()` at `begin_frame()` and subtracting that baseline at
`submit_frame()`. Extended the multi-draw regression to submit a second frame and
assert that both frame stats report one command buffer while the backend total is two.

* `cmake --build build --target test_phase10 test_phase14` — passed.
* `ctest --test-dir build -R 'Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure` — passed; 2/2 tests passed.
