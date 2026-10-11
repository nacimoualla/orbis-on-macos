# Userland Compatibility Layer over Full-System Emulation

To achieve maximum performance and native desktop integration on macOS, we chose a high-level userland compatibility layer over full-system hardware emulation or hypervisor virtualization. The runtime intercepts Orbis OS (FreeBSD) system calls, dynamically loads guest ELF binaries, and translates system services directly to Darwin/Mach-O APIs, bypassing the need to emulate proprietary hardware coprocessors (SAM, Southbridge) and kernel virtualization overhead.
