# Two-Tier GCN to SPIR-V to MSL Shader Pipeline

To translate AMD GCN shader bytecode into Metal Shading Language (MSL), we chose a two-tier pipeline targeting SPIR-V intermediate representation before emitting MSL via SPIRV-Cross, backed by an on-disk pipeline cache. This avoids creating and maintaining a proprietary custom GCN-to-MSL text backend, allows standard SPIR-V optimization and validation passes, and prevents runtime shader stutter.
