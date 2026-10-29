# Flappy Bird GPU Decoding and Rendering Design

## Goal

Make the supplied `games/flappy_bird` target render through the Apple GPU
smoothly on Apple Silicon macOS. The first milestone is a stable 60 FPS
interactive presentation path with correct PM4 state and draw behavior. This
does not attempt to implement the entire PS4 GCN ISA; unsupported GPU features
must be diagnosed explicitly rather than silently rendered incorrectly.

## Current gaps

- `Pm4Translator` creates and waits for a Metal command buffer for individual
  draws. This serializes CPU and GPU work and prevents efficient frame-level
  batching.
- The backend exposes a shared offscreen texture, but there is no explicit
  frame lifecycle (`begin_frame`/`submit_frame`) or GPU completion policy.
- PM4 state is only partially modeled. Draws currently use a compatibility
  mesh pipeline instead of a target-driven shader/resource description.
- Texture decoding is heuristic and lacks a stable cache keyed by guest
  address, dimensions, format, and content generation.
- The display presenter copies the completed framebuffer through a separate
  command buffer, and the runner presents only when a flip counter changes.

## Design

### 1. Frame-scoped Metal command submission

Add a frame lifecycle to `MetalBackend`:

- `begin_frame()` creates one command buffer and render encoder.
- PM4 draw and clear operations encode into the active encoder.
- `submit_frame(bool wait_for_completion)` ends the encoder and commits once.
- `wait_for_completion` is false for interactive presentation; tests may opt
  into synchronous completion before readback.

The backend owns command-buffer lifetime and reports command-buffer errors with
the frame number and operation label. No draw path calls
`waitUntilCompleted` during normal interactive execution.

The framebuffer remains a GPU render target. The presenter samples it directly
on the same command queue, with a completion/presentation ordering guarantee.
Shared storage is retained for diagnostic readback only.

### 2. Explicit PM4 state model

Extend `Pm4Translator::Impl` with frame and pipeline state:

- render target dimensions and scissor
- primitive topology and index type
- vertex/index buffer bindings
- shader program addresses and resource registers
- blend/depth/raster state where the command stream supplies them
- dirty bits for pipeline, vertex layout, textures, and uniforms

`execute_packet` updates state and marks the relevant dirty bits. Draw commands
snapshot the state and encode one Metal draw. Unsupported or malformed packet
payloads return `false` and emit one bounded diagnostic containing opcode,
payload size, guest addresses, and frame number.

### 3. GPU-backed resource uploads and texture cache

Create a small resource cache in the translator/backend boundary:

- vertex and index buffers use `MTLBuffer` objects backed by guest-memory
  snapshots, refreshed only when the guest range changes
- textures are cached by guest address, width, height, format, pitch, and a
  content generation counter
- supported PS4 formats initially include the formats already observed by the
  Flappy trace plus BGRA/RGBA 8-bit normalized formats
- unsupported formats fall back to a diagnostic checkerboard only in
  non-strict mode; strict mode rejects the draw

Uploads are staged before encoding the frame, not repeatedly inside each draw.
The cache has a bounded byte budget and evicts least-recently-used resources
after a frame completes.

### 4. Target-driven shader compatibility

Keep the existing compatibility mesh shader as a fallback, but add a shader
selection layer:

- identify the guest vertex/fragment program addresses from PM4 state
- use a cached Metal pipeline for known target signatures
- use the compatibility pipeline only when the target signature is not yet
  translated
- record shader address, resource signature, and fallback reason once per
  signature

This makes missing GCN-to-MSL work visible and allows adding only the shader
patterns required by Flappy Bird before attempting broad retail coverage.

### 5. Presentation and frame pacing

Move presentation to an explicit frame boundary:

1. poll input/events
2. execute a bounded guest command batch
3. drain PM4 commands into the current Metal frame
4. submit the frame without blocking
5. present the drawable in queue order
6. pace to 60 Hz using a monotonic deadline, allowing one frame in flight

If the GPU queue is behind, skip a presentation rather than blocking the guest
CPU. A diagnostic mode can wait for completion and report CPU encode time, GPU
completion time, draw count, upload bytes, cache hits/misses, and dropped
presentations.

## Data flow

Guest GNM/HLE calls write PM4 packets and resource data into guest memory.
The ring buffer parser validates and produces packets. The translator updates
explicit state, resolves cached guest buffers/textures, and encodes Metal
commands into one frame command buffer. The presenter samples the framebuffer
into a CAMetalLayer drawable. CPU readback is used only by tests and
diagnostics, never by the normal presentation path.

## Error handling

- malformed packet: reject packet, preserve prior state, increment a bounded
  diagnostic counter
- unsupported opcode/state/format: reject or checkerboard according to strict
  mode; never silently claim a successful draw
- Metal pipeline/resource/command-buffer failure: log the frame and operation,
  mark the frame failed, and return failure to the caller
- guest address or size outside memory: reject before creating an upload
- cache exhaustion: evict completed-frame resources, then fail explicitly if
  the requested resource still cannot be allocated

## Testing

1. Unit-test PM4 payload validation and state dirty-bit transitions.
2. Add headless tests for frame batching: many draws should produce one commit,
   and synchronous readback should still show the expected pixels.
3. Add texture-format/cache tests using guest-memory fixtures.
4. Preserve existing Phase 10 and Phase 14 packet/raster tests.
5. Add a Flappy smoke harness that loads the supplied ELF, runs a bounded
   number of frames, verifies non-black framebuffer output, and records frame
   time/draw/upload counters.
6. Run the interactive target with Metal HUD or internal counters and require
   no sustained frame-time spikes above the 16.67 ms 60 FPS budget in the
   smoke interval.

## Scope boundary

This milestone does not implement arbitrary GCN shader translation, full PS4
video-out semantics, or every PM4 packet. It establishes the GPU execution and
diagnostic architecture needed to add those features without serializing every
draw or hiding unsupported behavior.
