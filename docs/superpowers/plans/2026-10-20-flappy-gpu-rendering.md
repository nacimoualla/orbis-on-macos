# Flappy Bird GPU Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the supplied Flappy Bird target use batched Apple GPU rendering with explicit PM4 state, cached guest resources, visible unsupported-feature diagnostics, and a paced 60 FPS presentation path.

**Architecture:** `MetalBackend` owns one frame command buffer and render encoder at a time. `Pm4Translator` decodes packets into persistent state, resolves guest buffers/textures through bounded caches, and encodes all draws into that frame instead of committing per draw. `DisplayWindow` presents the completed framebuffer after frame submission, while the runner uses an explicit frame boundary and monotonic 60 Hz deadline.

**Tech Stack:** C++20, Objective-C++, Apple Metal, AppKit/CAMetalLayer, CMake/CTest, existing `MemoryManager`, `GpuRingBuffer`, `Pm4Translator`, and Flappy Bird ELF/PRX files under `games/flappy_bird`.

## Global Constraints

- Target Apple Silicon macOS and use Metal for all normal rendering.
- Do not implement arbitrary GCN shader translation in this milestone; use target signatures plus the existing compatibility shader.
- Unsupported packets, formats, addresses, and Metal failures must be explicit failures or bounded diagnostics, never silent success.
- Do not call `waitUntilCompleted` in the normal interactive draw path; synchronous completion remains available for tests/readback.
- Preserve existing Phase 10 and Phase 14 behavior and test commands.
- Keep CPU readback restricted to tests and diagnostics; normal presentation must sample the GPU framebuffer directly.
- Do not modify unrelated existing CPU worktree changes.

---

## File map

- Modify: `include/ps4/metal_backend.hpp` — add frame lifecycle and frame statistics APIs.
- Modify: `src/gpu/metal_backend.mm` — own frame command buffer/encoder, completion state, and asynchronous submission.
- Modify: `include/ps4/pm4_translator.hpp` — expose frame boundaries and GPU statistics needed by the runner/tests.
- Modify: `src/gpu/pm4_translator.mm` — replace per-draw command buffers, add dirty state, explicit validation, resource caches, and shader-selection diagnostics.
- Modify: `include/ps4/display_window.hpp` — accept an already-submitted frame and expose presentation pacing state if needed.
- Modify: `src/gpu/display_window.mm` — preserve queue ordering and avoid presenting an incomplete framebuffer.
- Modify: `src/main.cpp` — run guest work inside explicit frame boundaries and pace presentation.
- Modify: `tests/test_phase10.cpp` — packet validation and state-transition coverage.
- Modify: `tests/test_phase14.cpp` — multi-draw single-submission and resource-cache coverage.
- Create: `tests/test_flappy_gpu.cpp` — bounded Flappy smoke harness and non-black output/counter checks.
- Modify: `CMakeLists.txt` — register the Flappy GPU smoke target and CTest entry.
- Modify: `ROADMAP.md` — record the completed GPU batching/presentation milestone and remaining shader/video-out scope.

### Task 1: Add a frame-scoped Metal submission API

**Files:**
- Modify: `include/ps4/metal_backend.hpp`
- Modify: `src/gpu/metal_backend.mm`
- Test: `tests/test_phase14.cpp`

**Interfaces:**
- Produces `bool begin_frame()`, `bool encode_clear(ColorRGBA)`, `bool submit_frame(bool wait_for_completion)`, `bool wait_for_last_frame()`, `bool frame_in_progress() const noexcept`, and `u64 submitted_frame_count() const noexcept`.
- `raw_command_queue()` and `raw_framebuffer()` remain available for `DisplayWindow`.
- Existing `clear_screen()` remains a synchronous compatibility wrapper implemented as `begin_frame()`, `encode_clear()`, `submit_frame(true)`.

- [ ] **Step 1: Write the failing frame lifecycle test**

Add this test near the existing Metal rendering test in `tests/test_phase14.cpp`:

```cpp
void test_metal_frame_submission_batches_clear() {
    MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(64, 64));
    assert(metal.begin_frame());
    assert(metal.frame_in_progress());
    assert(metal.encode_clear(ColorRGBA{0.1f, 0.2f, 0.3f, 1.0f}));
    assert(metal.submit_frame(true));
    assert(!metal.frame_in_progress());
    assert(metal.submitted_frame_count() == 1);

    u8 r = 0, g = 0, b = 0, a = 0;
    assert(metal.read_pixel(32, 32, r, g, b, a));
    assert(r >= 24 && r <= 27);
    assert(g >= 50 && g <= 53);
    assert(b >= 75 && b <= 78);
    assert(a == 255);
}
```

Register the call from `main()` in the same test file.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cmake --build build --target test_phase14
./build/test_phase14
```

Expected: compilation fails because `begin_frame()` and `encode_clear()` are not declared.

- [ ] **Step 3: Implement the backend frame state**

In `MetalBackend::Impl`, add:

```cpp
id<MTLCommandBuffer> active_command_buffer{nil};
id<MTLRenderCommandEncoder> active_render_encoder{nil};
u64 submitted_frames{0};
bool last_frame_completed{true};
```

Implement `begin_frame()` by rejecting an existing active frame, creating a command buffer, and leaving the encoder nil until the first render operation. Implement `encode_clear()` by creating a render pass against `framebuffer`, using `MTLLoadActionClear` and `MTLStoreActionStore`, ending the encoder immediately, and retaining the active command buffer for later submission. Implement `submit_frame()` by ending any active encoder, committing once, optionally waiting, updating `last_frame_completed`, incrementing `submitted_frames`, and clearing active Objective-C references. Implement `wait_for_last_frame()` only when a committed command buffer remains incomplete.

The existing `clear_screen()` must call the new methods and return the completion status. Return `false` with a diagnostic if frame methods are called before initialization, without a framebuffer, or in an invalid lifecycle order.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
cmake --build build --target test_phase14
./build/test_phase14
```

Expected: all Phase 14 tests pass, including the new frame submission test.

- [ ] **Step 5: Commit**

```bash
git add include/ps4/metal_backend.hpp src/gpu/metal_backend.mm tests/test_phase14.cpp
git commit -m "feat: add frame-scoped Metal submission"
```

### Task 2: Migrate PM4 draws to the active Metal frame

**Files:**
- Modify: `src/gpu/pm4_translator.mm`
- Modify: `include/ps4/pm4_translator.hpp`
- Modify: `tests/test_phase14.cpp`

**Interfaces:**
- Add `bool begin_frame()`, `bool submit_frame(bool wait_for_completion = false)`, and `bool frame_in_progress() const noexcept` to `Pm4Translator`.
- Add `struct GpuFrameStats { u64 frame_index; usize packets; usize draws; usize indexed_draws; usize unsupported_packets; usize upload_bytes; usize resource_cache_hits; usize resource_cache_misses; }` and `GpuFrameStats frame_stats() const noexcept`.
- `render_indexed_mesh()` and `render_auto_mesh()` must encode into `MetalBackend`’s active frame and must not create or commit a command buffer.

- [ ] **Step 1: Add the batching regression test**

Add a test that begins one frame, executes two PM4 indexed draws, submits once synchronously, and checks both draw counters and a non-black pixel:

```cpp
void test_pm4_multiple_draws_use_one_frame() {
    MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(256, 256));
    MemoryManager memory;
    assert(memory.initialize());

    const GuestAddr vertices = 0x120000;
    const GuestAddr indices = 0x121000;
    const Vertex3D quad[3] = {
        {{{-0.8f, -0.8f, 0.0f}}, {{1.0f, 0.0f, 0.0f, 1.0f}}},
        {{{ 0.0f,  0.8f, 0.0f}}, {{0.0f, 1.0f, 0.0f, 1.0f}}},
        {{{ 0.8f, -0.8f, 0.0f}}, {{0.0f, 0.0f, 1.0f, 1.0f}}},
    };
    const u16 triangle[3] = {0, 1, 2};
    memory.write_bytes(vertices, quad, sizeof(quad));
    memory.write_bytes(indices, triangle, sizeof(triangle));

    Pm4Translator translator(&metal);
    translator.set_memory_manager(&memory);
    translator.bind_vertex_buffer(vertices, sizeof(quad));
    assert(translator.begin_frame());
    assert(translator.execute_indexed_draw(indices, 3));
    assert(translator.execute_indexed_draw(indices, 3));
    assert(translator.submit_frame(true));
    assert(translator.draw_count() == 2);
    assert(metal.submitted_frame_count() == 1);
}
```

- [ ] **Step 2: Run the test and verify the current per-draw implementation fails**

Run:

```bash
cmake --build build --target test_phase14
./build/test_phase14
```

Expected: the new API does not compile before implementation; after temporary declarations are added, the behavioral assertion must expose more than one backend submission or a lifecycle error.

- [ ] **Step 3: Route draw encoding through `MetalBackend`**

At the start of each public translator frame, call `metal->begin_frame()`. In both render functions, replace the local `MTLRenderPassDescriptor`, command buffer, encoder, `commit`, and `waitUntilCompleted` sequence with an encoder acquired through a new backend method:

```cpp
id<MTLRenderCommandEncoder> active_render_encoder();
bool ensure_render_encoder(MTLLoadAction load_action, ColorRGBA clear);
```

`ensure_render_encoder()` must create the framebuffer pass once per frame, clear on the first draw, and load on subsequent draws. Add `end_render_encoder()` only inside `submit_frame()`. Preserve all existing pipeline, buffer, index, viewport, scissor, and draw-call setup on the shared encoder.

When no Metal backend is attached, keep the existing counter-only behavior. When a backend is attached but no frame was started, return `false` and log `[PM4] draw outside frame`.

- [ ] **Step 4: Add a single-submission counter assertion**

Expose `submitted_frame_count()` from `MetalBackend` and assert it remains `1` after the two-draw test. Add a diagnostic-only counter to `GpuFrameStats` for `command_buffers_submitted`, populated from the backend.

- [ ] **Step 5: Run focused and baseline GPU tests**

Run:

```bash
cmake --build build --target test_phase10 test_phase14
ctest --test-dir build -R 'Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure
```

Expected: both tests pass and the multi-draw test reports one command buffer for the frame.

- [ ] **Step 6: Commit**

```bash
git add include/ps4/pm4_translator.hpp src/gpu/pm4_translator.mm tests/test_phase14.cpp
git commit -m "feat: batch PM4 draws into one Metal frame"
```

### Task 3: Make PM4 state transitions explicit and reject malformed packets

**Files:**
- Modify: `src/gpu/pm4_translator.mm`
- Modify: `tests/test_phase10.cpp`
- Modify: `tests/test_phase14.cpp`

**Interfaces:**
- Add internal dirty bits for pipeline, vertex layout, texture, uniform, render target, and scissor state.
- Add `usize unsupported_packet_count() const noexcept` and `usize malformed_packet_count() const noexcept` to `Pm4Translator`.
- Every packet handler must validate payload length before indexing it and preserve prior state when validation fails.

- [ ] **Step 1: Add failing malformed-packet tests**

Add tests for short `IT_SET_CONTEXT_REG`, short `IT_DRAW_INDEX_2`, and an unknown Type 3 opcode:

```cpp
void test_pm4_rejects_short_payload_without_state_change() {
    Pm4Translator translator;
    const u32 payload[] = {PA_CL_VPORT_XSCALE};
    Pm4Packet packet{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_SET_CONTEXT_REG, 1)),
        .payload = {payload, payload + 1},
    };
    const GpuViewport before = translator.viewport();
    assert(!translator.execute_packet(packet));
    assert(translator.malformed_packet_count() == 1);
    assert(translator.viewport().width == before.width);
}
```

Add analogous coverage for unknown opcodes and assert `unsupported_packet_count()` increments.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cmake --build build --target test_phase10
./build/test_phase10
```

Expected: compilation fails until the counters/accessors exist, then malformed packets must fail before the implementation is corrected.

- [ ] **Step 3: Implement validation and dirty bits**

Add helper functions in `Pm4Translator::Impl`:

```cpp
bool require_payload(const Pm4Packet& packet, usize count, const char* operation);
void mark_pipeline_dirty();
void mark_resource_dirty();
```

Use `require_payload()` before every fixed-layout packet access. Increment `malformed_packet_count` for insufficient payloads, `unsupported_packet_count` for valid but unimplemented opcodes/state, and return `false` in both cases. Mark only the relevant dirty bit after successful state updates. Do not reset unrelated state when a packet fails.

- [ ] **Step 4: Run GPU regression tests**

Run:

```bash
cmake --build build --target test_phase10 test_phase14
ctest --test-dir build -R 'Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure
```

Expected: all existing packet/raster tests and new validation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/pm4_translator.mm tests/test_phase10.cpp tests/test_phase14.cpp
git commit -m "fix: validate PM4 payloads and track state dirtiness"
```

### Task 4: Add bounded GPU buffer and texture caches

**Files:**
- Modify: `src/gpu/pm4_translator.mm`
- Modify: `include/ps4/pm4_translator.hpp`
- Modify: `tests/test_phase14.cpp`

**Interfaces:**
- Add internal cache entries keyed by `{GuestAddr address, usize size, u32 format, u32 width, u32 height, u64 generation}`.
- Add `void set_resource_cache_budget(usize bytes) noexcept`.
- Add `void invalidate_guest_resources(GuestAddr address, usize size) noexcept`.
- Add counters to `GpuFrameStats`: `upload_bytes`, `resource_cache_hits`, and `resource_cache_misses`.

- [ ] **Step 1: Add cache behavior tests**

Add a test that draws twice from unchanged guest vertex/index data and asserts the second draw increases cache hits without increasing uploaded bytes. Then mutate the guest vertex range, call `invalidate_guest_resources()`, draw again, and assert a cache miss and additional upload.

```cpp
const GpuFrameStats first = translator.frame_stats();
assert(first.resource_cache_misses >= 2);
const usize first_uploads = first.upload_bytes;
// draw the same guest ranges in a second frame
const GpuFrameStats second = translator.frame_stats();
assert(second.resource_cache_hits > first.resource_cache_hits);
assert(second.upload_bytes == first_uploads);
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cmake --build build --target test_phase14
./build/test_phase14
```

Expected: the new cache API is initially absent and the current implementation allocates a fresh `MTLBuffer` for every draw.

- [ ] **Step 3: Implement buffer cache entries and LRU eviction**

Store `id<MTLBuffer>` plus byte size and last-used frame in an internal map. Before `newBufferWithBytes`, look up an unchanged guest range; on a hit reuse the Metal buffer. On a miss validate `MemoryManager::is_valid_address`, create the buffer, increment upload bytes, and evict least-recently-used entries until `current_cache_bytes + requested_bytes <= budget`. Use a default budget of 64 MiB and reject allocation if one resource itself exceeds the budget.

- [ ] **Step 4: Cache decoded textures**

Use the existing descriptor fields (`ps_texture_base`, width, height, format, pitch words) as the key. Keep the supported decode path for observed BGRA/RGBA 8-bit formats, return the cached `MTLTexture` when the key is unchanged, and increment a bounded diagnostic for unsupported formats. In non-strict mode create the existing white/checkerboard fallback; in strict mode return `nil` so the draw fails explicitly.

- [ ] **Step 5: Run Phase 14 and check diagnostics**

Run:

```bash
cmake --build build --target test_phase14
ctest --test-dir build -R Phase14_MeshPipeline --output-on-failure
```

Expected: rasterization remains unchanged, cache hits appear on repeated resources, and invalid guest ranges return `false` without a Metal allocation.

- [ ] **Step 6: Commit**

```bash
git add include/ps4/pm4_translator.hpp src/gpu/pm4_translator.mm tests/test_phase14.cpp
git commit -m "feat: cache guest GPU buffers and textures"
```

### Task 5: Add target-aware shader selection and diagnostics

**Files:**
- Modify: `src/gpu/pm4_translator.mm`
- Modify: `tests/test_phase10.cpp`
- Modify: `tests/test_phase14.cpp`
- Modify: `ROADMAP.md`

**Interfaces:**
- Add an internal `ShaderSignature { GuestAddr vertex_program; GuestAddr fragment_program; u32 vertex_resource_signature; u32 fragment_resource_signature; }`.
- Add `usize shader_fallback_count() const noexcept`.
- Add `std::string shader_status() const`.

- [ ] **Step 1: Add shader fallback test**

Set synthetic VS/PS program registers, execute a draw, and assert that the draw succeeds through the compatibility pipeline while `shader_fallback_count()` increments once for that signature, not once per repeated draw.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cmake --build build --target test_phase10 test_phase14
./build/test_phase10
./build/test_phase14
```

Expected: the new shader accessors are absent and current logging does not provide a stable per-signature fallback count.

- [ ] **Step 3: Implement signature-based pipeline lookup**

Build a signature from `vs_shader_addr`, `ps_shader_addr`, and the resource register values already tracked by `Pm4Translator::Impl`. Use a map from signature to `id<MTLRenderPipelineState>`. Return the existing mesh pipeline for unknown signatures, but insert a “compatibility fallback” record only on the first encounter. Keep pipeline compilation outside the draw loop and return `false` if Metal reports a pipeline error.

- [ ] **Step 4: Document the scope boundary**

Update `ROADMAP.md` to state that frame batching, resource caching, and explicit shader fallback are complete for the milestone, while arbitrary GCN-to-MSL translation and complete video-out remain future work.

- [ ] **Step 5: Run GPU regression tests**

Run:

```bash
cmake --build build --target test_phase10 test_phase14
ctest --test-dir build -R 'Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure
```

Expected: all tests pass and repeated unknown shader signatures produce one fallback diagnostic.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/pm4_translator.mm tests/test_phase10.cpp tests/test_phase14.cpp ROADMAP.md
git commit -m "feat: track target shader signatures and fallbacks"
```

### Task 6: Make presentation and runner frame pacing explicit

**Files:**
- Modify: `include/ps4/display_window.hpp`
- Modify: `src/gpu/display_window.mm`
- Modify: `src/main.cpp`
- Modify: `tests/test_phase8.cpp`

**Interfaces:**
- Keep `DisplayWindow::present_frame()` as the public presentation entry point.
- Add internal `u64 presented_frames`, `u64 skipped_presentations`, and `std::chrono::steady_clock::time_point next_frame_deadline`.
- Add a runner-side 60 Hz deadline using `std::chrono::steady_clock`, not wall-clock time.

- [ ] **Step 1: Add presentation pacing test seams**

Add a headless test seam that initializes `DisplayWindow` with `WindowConfig{.headless = true}`, calls `poll_events()` and `present_frame()` repeatedly, and verifies the window remains open while no drawable is available. Keep this test free of sleeps and use the existing headless configuration.

- [ ] **Step 2: Run the test before implementation**

Run:

```bash
cmake --build build --target test_phase8
ctest --test-dir build -R Phase8_DisplayWindow --output-on-failure
```

Expected: baseline behavior remains passing; this task must not introduce a requirement for a drawable in headless mode.

- [ ] **Step 3: Preserve queue ordering during presentation**

Before presenting, call `metal.wait_for_last_frame()` only when the backend reports that the framebuffer producer is not yet ordered before the presenter. Keep the presenter’s fullscreen triangle command buffer on the same command queue. Do not call `getBytes` or `waitUntilCompleted` in `present_frame()`. If `nextDrawable` is unavailable, poll events and return `true` without blocking.

- [ ] **Step 4: Add frame deadline logic to the runner**

At the frame loop boundary in `src/main.cpp`, use:

```cpp
constexpr auto frame_period = std::chrono::microseconds(16667);
auto next_deadline = std::chrono::steady_clock::now();
while (window.is_open()) {
    next_deadline += frame_period;
    window.poll_events();
    translator.begin_frame();
    run_guest_batch();
    translator.submit_frame(false);
    window.present_frame();
    std::this_thread::sleep_until(next_deadline);
}
```

Retain the existing close/input behavior and reset `next_deadline` when the loop falls more than two frame periods behind. Do not sleep in headless test mode.

- [ ] **Step 5: Run presentation and GPU tests**

Run:

```bash
cmake --build build --target test_phase8 test_phase10 test_phase14 ps4-runner
ctest --test-dir build -R 'Phase8_DisplayWindow|Phase10_GpuCommandProcessor|Phase14_MeshPipeline' --output-on-failure
```

Expected: tests pass; interactive mode has no per-draw GPU wait and headless mode does not block on pacing.

- [ ] **Step 6: Commit**

```bash
git add include/ps4/display_window.hpp src/gpu/display_window.mm src/main.cpp tests/test_phase8.cpp
git commit -m "feat: pace GPU presentation at 60 Hz"
```

### Task 7: Add a bounded Flappy Bird GPU smoke harness

**Files:**
- Create: `tests/test_flappy_gpu.cpp`
- Modify: `CMakeLists.txt`
- Modify: `ROADMAP.md`

**Interfaces:**
- The test must locate `games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/eboot.elf` relative to `CMAKE_SOURCE_DIR`.
- The test must run a bounded smoke interval and exit without requiring a human window interaction.
- The test must report `frames`, `draws`, `packets`, `unsupported_packets`, `upload_bytes`, and elapsed milliseconds.

- [ ] **Step 1: Write the smoke harness**

Use the existing ELF loader/runner initialization path and construct `MetalBackend`, `MemoryManager`, `Pm4Translator`, and a headless `DisplayWindow`. Fail with a clear message if the ELF path is missing or loading fails. Run exactly 120 bounded frame iterations, pump events, execute the existing guest batch entry point, submit asynchronously, and present. At completion, synchronously wait for the last frame, read the center pixel, and print:

```text
Flappy GPU smoke: frames=120 draws=<n> packets=<n> unsupported=<n> uploads=<n> elapsed_ms=<n>
```

Require `frames == 120`, `draws > 0`, `packets > 0`, and at least one non-black framebuffer pixel. Do not require zero unsupported packets; the test must expose their count for prioritization.

- [ ] **Step 2: Register the test target**

Add:

```cmake
add_executable(test_flappy_gpu tests/test_flappy_gpu.cpp)
target_link_libraries(test_flappy_gpu PRIVATE ps4_core)
add_test(NAME Flappy_GpuSmoke COMMAND test_flappy_gpu)
set_tests_properties(Flappy_GpuSmoke PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

- [ ] **Step 3: Run the smoke test**

Run:

```bash
cmake --build build --target test_flappy_gpu
ctest --test-dir build -R Flappy_GpuSmoke --output-on-failure
```

Expected: the test completes after 120 frames, prints the counters, and verifies non-black GPU output. If the retail target cannot reach a draw because a missing HLE/video-out dependency stops execution, the test must fail with that dependency named rather than passing with synthetic output.

- [ ] **Step 4: Update the roadmap**

Record the smoke command and the counters it validates in the GPU section of `ROADMAP.md`. State that shader signatures and unsupported PM4 counts remain the evidence-driven queue for subsequent Flappy fidelity work.

- [ ] **Step 5: Commit**

```bash
git add tests/test_flappy_gpu.cpp CMakeLists.txt ROADMAP.md
git commit -m "test: add Flappy GPU smoke coverage"
```

### Task 8: Full validation and performance evidence

**Files:**
- Modify: no source files unless a validation defect is found.
- Test: all GPU and existing CPU test targets.

- [ ] **Step 1: Build all relevant targets**

Run:

```bash
cmake --build build --target test_phase6 test_phase8 test_phase10 test_phase14 test_flappy_gpu ps4-runner
```

Expected: all targets compile with the project warning flags and no new warnings from the changed GPU files.

- [ ] **Step 2: Run the focused GPU suite**

Run:

```bash
ctest --test-dir build -R 'Phase6_MetalBackend|Phase8_DisplayWindow|Phase10_GpuCommandProcessor|Phase14_MeshPipeline|Flappy_GpuSmoke' --output-on-failure
```

Expected: every selected test passes, including the 120-frame Flappy smoke run.

- [ ] **Step 3: Run the complete CTest suite**

Run:

```bash
ctest --test-dir build --output-on-failure
```

Expected: no regression in CPU, loader, memory, audio, display, or linker tests. A pre-existing machine-sensitive Phase 15 MIPS threshold failure must be reported separately if it remains unchanged.

- [ ] **Step 4: Check source hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Only intended GPU implementation, test, roadmap, and plan/spec commits are present; unrelated existing CPU modifications remain unaltered.

- [ ] **Step 5: Record performance evidence**

Run the Flappy smoke test twice and record the printed elapsed time, draw count, upload bytes, and unsupported packet count. For interactive validation, require no sustained frame interval above 16.67 ms over the 120-frame interval. If the target stops before rendering, preserve the explicit failure and use its named missing dependency as the next implementation item rather than weakening the test.

