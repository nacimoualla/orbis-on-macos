# Two-Tier GCN to SPIR-V to MSL Shader Pipeline

To translate AMD GCN shader bytecode into Metal Shading Language (MSL), we chose a two-tier pipeline targeting SPIR-V intermediate representation before emitting MSL via SPIRV-Cross, backed by an on-disk pipeline cache. This avoids creating and maintaining a proprietary custom GCN-to-MSL text backend, allows standard SPIR-V optimization and validation passes, and prevents runtime shader stutter.

## Amendment (Phase 19): direct GCN -> MSL emitter

**Status:** accepted for the current implementation; the SPIR-V tier remains the long-term target.

**Deviation.** The first real translator (`src/gpu/gcn_decoder.cpp`, `src/gpu/gcn_to_msl.cpp`,
`src/gpu/gcn_pipeline.mm`, API in `include/ps4/gcn_shader.hpp` / `include/ps4/gcn_pipeline.hpp`)
emits MSL text directly instead of going GCN -> SPIR-V -> SPIRV-Cross -> MSL.

**Why.**
- SPIRV-Cross is not available in the toolchain (Homebrew only ships SPIRV-Headers/Tools/glslang
  here) and the build has no third-party C++ dependencies; vendoring it via FetchContent would add a
  network-dependent build step and ~60k lines for a cross-compiler whose only job would be to print MSL.
- The offline `metal` compiler is not installed, but `MTLDevice newLibraryWithSource:` compiles MSL at
  runtime on every Mac, so emitting MSL directly keeps the whole pipeline self-contained.
- GCN's execution model (scalar + vector register files, EXEC/VCC lane masks, scalar memory loads
  through descriptors) maps onto a per-lane MSL program with `uint s[128]`/`uint v[N]` arrays almost
  one-to-one; there is no structured-control-flow requirement to satisfy as there is for SPIR-V.

**Shape of the implementation (kept SPIR-V friendly).**
- Decoder and `Program` (with fetch-shader inlining through the `s_swappc_b64` user-SGPR pair) are
  backend-agnostic; a SPIR-V emitter can consume the same `Instruction` stream.
- Memory accessed by a draw is discovered on the CPU (`scan_resources`, evaluating the uniform scalar
  prefix with the real user SGPRs), copied into one arena buffer, and addressed from MSL through a
  small range table. The generated MSL therefore depends only on the ISA, so pipelines are cached by
  ISA hash (+ fetch shader hash, topology lowering, blend, write mask, PS input mapping, RT format).
- Unsupported instructions/state make the translator report an error once; the draw falls back to the
  fixed mesh shader in `pm4_translator.mm`. `PS4_GCN_SHADERS=0` disables the path.
- On-disk caching is not done yet (runtime MSL compile is ~ms per pipeline); `PS4_GCN_DUMP_DIR=<dir>`
  writes ISA binaries, disassembly and the generated MSL for each pipeline, which is the seed for a
  later binary-archive cache.
