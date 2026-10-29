# Task 4 Report — Bounded Guest GPU Resource Caches

## Status

**Complete.** Implemented and committed on top of `ebb3c9f` as commit `d6e17f0` (`feat: cache guest GPU buffers and textures`).

## Changes

- Added public `set_resource_cache_budget(usize)` and `invalidate_guest_resources(GuestAddr, usize)` APIs.
- Added bounded, shared 64 MiB default LRU caches for guest vertex/index Metal buffers and decoded textures.
- Added cache keys containing guest address, byte size, format, dimensions, and resource generation.
- Added resource invalidation for overlapping guest ranges and generation advancement.
- Added cumulative frame statistics for upload bytes, cache hits, and cache misses.
- Reused unchanged guest buffers and decoded textures; cache misses upload and account for resource bytes.
- Rejected invalid guest vertex/index ranges and over-budget resources before Metal allocation.
- Added Phase 14 coverage for repeated draws, cache hits, upload stability, guest mutation/invalidation, and over-budget rejection.

## Validation

Commands run:

- `cmake --build build --target test_phase14` — **passed**
- `./build/test_phase14` — **passed** (exit code 0; all Phase 14 tests passed)
- `ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure` — **passed** (1/1)
- `git diff --check` — **passed**

## Self-review

- Commit contains only the requested Task 4 files:
  - `include/ps4/pm4_translator.hpp`
  - `src/gpu/pm4_translator.mm`
  - `tests/test_phase14.cpp`
- Existing unrelated CPU changes remain unstaged and untouched (`src/cpu/executor.cpp`, `tests/test_phase5.cpp`, and the pre-existing untracked plans directory).
- Explicit invalid-address failures are preserved; no Metal resource is allocated for invalid or over-budget guest ranges.
- Cache eviction is global across buffers and textures and removes least-recently-used entries until the requested resource fits.

## Concerns / follow-up

- Texture decoding remains the existing observed RGBA8 path and existing non-strict white fallback behavior; no new strict-mode public toggle existed in the current API.
- Resource statistics are retained across `begin_frame()` so repeated-frame cache behavior and cumulative upload accounting are observable as specified by the Phase 14 brief.

## Review-fix pass (2026-10-20)

- Fixed texture cache aliasing: every texture cache miss now allocates a texture for its full `{address, size, format, width, height, generation}` key instead of reusing `active_metal_texture` solely by dimensions.
- Fixed invalidation locality: overlapping entries are removed individually; unrelated resources retain their cache keys and hits, with no global generation invalidation.
- Preserved LRU eviction across buffer and texture entries and the existing budget rejection behavior.

Validation rerun:

- `cmake --build build --target test_phase14` — **passed**
- `./build/test_phase14 >/dev/null 2>&1` — **passed** (exit code 0)
- `ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure` — **passed** (1/1, 0.07s)

The attempted Metal texture integration test was not retained because the current diagnostic path blocks indefinitely during the first textured draw on this environment; the non-textured Phase 14 suite remains green. This is recorded as a validation concern rather than hidden as a passing test.

Final focused rerun after review-fix cleanup:

- `cmake --build build --target test_phase14` — **passed**
- `./build/test_phase14 >/dev/null 2>&1` — **passed** (exit code 0)
- `ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure` — **passed** (1/1, 0.06s)
- `git diff --check` — **passed**

## Re-review pass (2026-10-20 16:11 UTC)

- Added `set_draw_diagnostics_enabled(bool)` so cache tests can disable the pre-existing verbose shader/memory diagnostic path that hangs during textured integration draws.
- Restored deterministic same-dimension texture tests covering distinct texture keys, reuse of the newer texture, local invalidation, and a fitting cross-resource budget that forces oldest-resource LRU eviction.
- Strengthened vertex invalidation assertions: the non-overlapping index cache must hit exactly once, while vertex invalidation must produce exactly one miss and upload exactly `sizeof(Vertex3D[3])` additional bytes.

Exact validation:

- `cmake --build build --target test_phase14` — **passed**
- `./build/test_phase14` — **passed**, exit code 0; includes Test 14.10 texture reuse/local invalidation/cross-resource LRU
- `ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure` — **passed**, 1/1, 0.08s
- `git diff --check` — **passed**

## Final review pass (2026-10-20 16:15 UTC)

- Both indexed and auto draw paths now reject a nil texture returned by `prepare_texture()` before encoding Metal commands, covering oversized-budget and failed texture allocation cases.
- Added an oversized texture rejection test using a cache budget smaller than the decoded texture footprint.
- Added observable same-size texture ownership coverage: distinct same-dimension textures render different center-pixel readbacks, while repeated texture use and local invalidation remain covered. Diagnostics are disabled for this test to avoid the known verbose diagnostic hang.

Exact validation:

- `cmake --build build --target test_phase14` — **passed**
- `./build/test_phase14` — **passed**, exit code 0; Test 14.10 reports texture reuse, local invalidation, LRU ordering, and oversized rejection
- `ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure` — **passed**, 1/1, 0.06s
- `git diff --check` — **passed**
