# Embedded JIT Translation Core over Hand-Rolled Decoder

To accelerate development and guarantee robust handling of the complex x86_64 ISA (including AVX/SSE vector instructions, page cross-overs, and condition flag synthesis), we chose to embed an established open-source translation core rather than hand-writing an x86_64 decoder and ARM64 machine code emitter from scratch. This focuses our engineering effort on Orbis OS kernel emulation, PRX dynamic linking, and the GNM-to-Metal graphics pipeline while leveraging an battle-tested JIT compiler.
