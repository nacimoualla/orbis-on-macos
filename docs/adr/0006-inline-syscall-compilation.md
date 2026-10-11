# Inline Syscall Compilation over Signal Traps

To minimize syscall overhead for high-frequency kernel operations (timing, mutexes, memory queries), we chose inline C++ function dispatch rather than relying on Darwin kernel exception traps (`EXC_BAD_INSTRUCTION`/`SIGSYS`). The dynamic binary translator compiles x86_64 `syscall` instructions into direct calls to an in-memory dispatch table, eliminating Mach exception handling context switches and preserving host register performance.
