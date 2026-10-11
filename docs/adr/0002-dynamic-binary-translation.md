# Dynamic Binary Translation (JIT) over Static Recompilation

To execute guest x86_64 binaries on Apple Silicon (ARM64), we chose a runtime dynamic binary translator (JIT) rather than static binary recompilation (AOT). Dynamic binary translation enables out-of-the-box loading of unmodified Orbis ELF executables and dynamic libraries (`.prx`), supporting runtime code generation and self-modifying code without per-binary reverse engineering.
