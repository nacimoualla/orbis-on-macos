# Orbis to macOS Compatibility Runtime

A high-level userland compatibility runtime enabling PlayStation 4 (Orbis OS) applications to execute natively on Apple Silicon macOS.

## Language

**Orbis OS**:
The FreeBSD-derived operating system running on the PlayStation 4.
_Avoid_: PS4 OS, system firmware

**Compatibility Layer**:
A userland runtime that intercepts guest Orbis system calls and dynamic libraries and translates them directly to host macOS Darwin/Mach APIs.
_Avoid_: Emulator, virtual machine, hypervisor

**Dynamic Binary Translator**:
The runtime JIT engine responsible for translating guest x86_64 CPU instructions into host ARM64 instructions.
_Avoid_: Static recompiler, interpreter

**GNM**:
The low-level proprietary GPU command buffer and hardware submission interface on Orbis OS.
_Avoid_: Graphics driver, Vulkan

**GNMX**:
The mid-level graphics library that wraps GNM with higher-level rendering state management on Orbis OS.
_Avoid_: Engine, middleware

**GCN Bytecode**:
AMD Graphics Core Next shader machine code used by the PS4 GPU, translated during runtime into Metal Shading Language (MSL).
_Avoid_: Host shader

**Unified Memory Map**:
The virtual memory subsystem mapping the guest 8GB flat address space directly into Apple Silicon unified virtual memory.
_Avoid_: Dedicated VRAM emulation, host paging

**PRX**:
PlayStation Relocatable eXecutable, an ELF64-derived dynamic shared library format used by Orbis OS.
_Avoid_: Shared library, DLL, dylib

**High-Level Emulation (HLE)**:
Clean-room implementation of guest system libraries directly in host C++ code rather than executing original binary code.
_Avoid_: Low-level emulation, firmware emulation

**Guest Pointer**:
A 64-bit virtual memory address within the guest's address space, resolved by adding the host memory base offset.
_Avoid_: Host pointer, physical address

**Thread Context**:
The virtual CPU register state and TLS segment bases (FS/GS) maintained per guest thread mapped to a host pthread.
_Avoid_: Process state, fiber

**NID**:
Numeric Identifier, a 64-bit cryptographic hash used by Orbis OS to uniquely identify exported and imported module symbols.
_Avoid_: Symbol name, ordinal

**Syscall Dispatch Table**:
An in-memory dispatch array mapping Orbis system call numbers directly to host C++ handler routines invoked inline by the JIT.
_Avoid_: Signal handler, trap table

**Shader Pipeline Cache**:
A persistent on-disk and in-memory cache storing translated Metal pipeline states keyed by GCN shader bytecode hashes.
_Avoid_: Shader bank, host cache

**Lock-Free Ring Buffer**:
A single-producer single-consumer circular buffer mediating between the guest audio thread and the host CoreAudio callback without mutex lock contention.
_Avoid_: Audio queue, pipe

**Headless Test Suite**:
Automated integration and regression tests that execute CPU translation, memory mapping, and offscreen Metal rendering without creating a GUI window.
_Avoid_: UI test, manual test
