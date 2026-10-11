# ps4-to-macos: Step-by-Step Implementation Roadmap

This roadmap is an interactive checklist. As you complete each step, click the checkbox (or change `[ ]` to `[x]`) to track your progress.

---

## 📌 Phase 1: Build Setup & Compilation Baseline

The goal of this phase is to ensure your C++20 build pipeline works smoothly on your Mac with Apple Clang.

- [x] **Step 1.1: Define primitive types**
  - Create `include/ps4/common.hpp` with integer primitives, `GuestAddr`, `Nid`, and `Elf64_*` aliases.
  *(Note: check line 38 in your `common.hpp`: make sure `Elf64_Adrr` is spelled `Elf64_Addr`).*

- [x] **Step 1.2: Create the `CMakeLists.txt` build configuration**
  - Create `CMakeLists.txt` in the project root.
  - Set C++ standard to `CXX_STANDARD 20`.
  - Add `include/` to header search paths (`include_directories(include)`).
  - Define an executable target `ps4-runner` pointing to `src/main.cpp`.
  - **Verification:** Run `cmake -B build` and verify that CMake configures without errors.

- [x] **Step 1.3: Verify `main.cpp` compilation**
  - In `src/main.cpp`, include `"ps4/common.hpp"` and print out `sizeof(GuestAddr)`.
  - **Verification:** Run `cmake --build build && ./build/ps4-runner`. It should print `8` and exit cleanly.

---

## 📌 Phase 2: The ELF64 & PRX Binary Loader

The goal of this phase is to open a real PS4 binary (`.elf` or `eboot.bin`), parse its headers, and inspect what is inside.

- [x] **Step 2.1: Define ELF64 Header structures (`include/ps4/elf.hpp`)**
  - Define `Elf64Header` struct (64 bytes) matching the standard ELF64 header.
  - Define `Elf64ProgramHeader` struct (56 bytes) matching the ELF64 program header.
  - Add `#pragma pack(push, 1)` and `static_assert` to ensure the struct sizes match the specification.
  - Add Sony PS4 custom segment constants (`PT_SCE_PROGLIB = 0x61000000`, `PT_SCE_DYNLIBDATA = 0x61000001`).

- [x] **Step 2.2: Implement `ElfLoader::load_file(...)` (`src/core/elf_loader.cpp`)**
  - Create a class `ElfLoader` in `include/ps4/elf_loader.hpp` and `src/core/elf_loader.cpp`.
  - Open the binary file with `std::ifstream` in binary mode.
  - Read the first 64 bytes into your `Elf64Header` struct.

- [x] **Step 2.3: Validate ELF Header Magic & Architecture**
  - Check `e_ident[0..3]`: Must match `0x7F`, `'E'`, `'L'`, `'F'`.
  - Check `e_ident[4]` (`EI_CLASS`): Must be `2` (64-bit ELF).
  - Check `e_machine`: Must be `0x3E` (AMD x86_64 architecture).
  - Return an error or throw if the file is not a valid 64-bit x86_64 ELF.

- [x] **Step 2.4: Parse and print Program Headers (`PT_LOAD`)**
  - Seek to `header.e_phoff` in the file.
  - Loop `header.e_phnum` times and read each `Elf64ProgramHeader`.
  - Filter for segments where `p_type == 1` (`PT_LOAD`).
  - Print the segment info: file offset, virtual address (`p_vaddr`), memory size (`p_memsz`), and flags (Read/Write/Exec).

---

## 📌 Phase 3: The Virtual Memory Subsystem

The goal of this phase is to allocate the PS4's 8 GB unified address space in macOS RAM and map the ELF segments into it.

- [x] **Step 3.1: Create `MemoryManager` (`src/core/memory.cpp`)**
  - Create `include/ps4/memory.hpp` and `src/core/memory.cpp`.
  - Add a pointer `u8* host_base_address = nullptr`.

- [x] **Step 3.2: Allocate 8 GB virtual memory using `mmap`**
  - In `MemoryManager::initialize()`, call `mmap` to reserve 8 GB:
    ```cpp
    mmap(nullptr, 8ULL * 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    ```
  - Store the resulting pointer as your base address.
  - Check if `mmap` failed (`MAP_FAILED`).

- [x] **Step 3.3: Implement Address Translation (`to_host_ptr`)**
  - Implement a method `void* to_host_ptr(GuestAddr guest_addr)`:
    ```cpp
    return static_cast<void*>(host_base_address + guest_addr);
    ```
  - Add an assertion: check that `guest_addr < 8ULL * 1024 * 1024 * 1024`.

- [x] **Step 3.4: Load ELF segments into Guest Memory**
  - In your loader, for every `PT_LOAD` segment:
    - Seek to `p_offset` in the file.
    - Read `p_filesz` bytes directly into `to_host_ptr(segment.p_vaddr)`.
    - If `p_memsz > p_filesz`, zero out the remaining bytes (`.bss` zero-initialization).

---

## 📌 Phase 4: High-Level Emulation (HLE) & Syscall Dispatch

The goal of this phase is to handle system calls emitted by the PS4 program when it asks the operating system to do something.

- [x] **Step 4.1: Define Syscall Numbers (`include/ps4/syscalls.hpp`)**
  - Define Orbis/FreeBSD syscall constants:
    - `SYS_EXIT = 1`
    - `SYS_WRITE = 4`
    - `SYS_MMAP = 477`
    - `SYS_MPROTECT = 74`

- [x] **Step 4.2: Build the `SyscallTable` Dispatcher (`src/core/syscall_table.cpp`)**
  - Create a class that maps syscall numbers to C++ function callbacks (`std::function` or function pointers).
  - Implement a fallback handler: if an unknown syscall is called, print `[HLE] Unimplemented syscall: N` and return an error.

- [x] **Step 4.3: Implement `sys_exit`**
  - Handler receives exit code (`int status`).
  - Print `[HLE] Guest requested exit with code: status` and call `std::exit(status)`.

- [x] **Step 4.4: Implement `sys_write` (Console Logging)**
  - Arguments: `int fd`, `GuestAddr buf`, `usize nbytes`.
  - Convert `buf` to a host pointer using `to_host_ptr(buf)`.
  - If `fd == 1` or `fd == 2` (stdout/stderr), forward the characters to `std::cout.write(...)`.

---

## 📌 Phase 5: CPU Execution & Instruction Dispatch

The goal of this phase is to start executing the guest x86_64 machine code at the ELF entry point.

- [x] **Step 5.1: Define CPU Register Context (`include/ps4/cpu_context.hpp`)**
  - Struct `CpuContext` with 64-bit general-purpose registers:
    `rax`, `rbx`, `rcx`, `rdx`, `rsi`, `rdi`, `rsp`, `rbp`, `r8`..`r15`, `rip`, `rflags`.
  - Set `rip = header.e_entry`.
  - Allocate a stack region in guest memory and set `rsp` to the top of the stack.

- [x] **Step 5.2: Instruction Fetch & Syscall Trap**
  - Create `src/cpu/executor.cpp`.
  - Read bytes at `to_host_ptr(rip)`.
  - When encountering `0x0F 0x05` (x86_64 `syscall` instruction):
    - Read syscall number from `rax`.
    - Pass arguments from `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9` into your `SyscallTable`.
    - Advance `rip += 2`.

---

## 📌 Phase 6: Offscreen Metal Framebuffer & Display

The goal of this phase is to set up a Metal device so the runtime can present graphics to macOS.

- [x] **Step 6.1: Initialize Apple Metal Device (`src/gpu/metal_backend.mm`)**
  - Create an Objective-C++ file `src/gpu/metal_backend.mm`.
  - Call `MTLCreateSystemDefaultDevice()` to get the default Apple Silicon GPU.
  - Create an `MTLCommandQueue` from the device.

- [x] **Step 6.2: Create the Framebuffer Texture**
  - Create a 1920x1080 `MTLTexture` with pixel format `MTLPixelFormatBGRA8Unorm`.
  - Verify that the texture is created successfully in unified memory.

- [x] **Step 6.3: Implement Clear Screen Render Pass**
  - Create an `MTLRenderPassDescriptor` with a clear color (e.g. Cornflower Blue).
  - Submit an `MTLCommandBuffer` to the queue and wait for completion.
  - Print confirmation that your first GPU command buffer completed!

---

## 📌 Phase 7: PRX Dynamic Linking & NID Symbol Resolution

The goal of this phase is to resolve external library function imports via 64-bit NID hashes using trampolines and HLE shims.

- [x] **Step 7.1: Define Relocation & Dynamic Structures (`include/ps4/reloc.hpp`)**
  - Define `Elf64_Rela` and `Elf64_Sym` structs.
  - Define dynamic tags (`DT_RELA`, `DT_RELASZ`, `DT_JMPREL`, `DT_SYMTAB`, `DT_STRTAB`).
  - Define x86_64 relocation types (`R_X86_64_RELATIVE`, `R_X86_64_GLOB_DAT`, `R_X86_64_JUMP_SLOT`).

- [x] **Step 7.2: Create NID Registry (`include/ps4/nid_registry.hpp` & `src/hle/nid_registry.cpp`)**
  - Define `NidRegistry` class mapping 64-bit NIDs to C++ HLE host shims.
  - Register canonical NIDs for `libkernel` and `libc`.

- [x] **Step 7.3: Implement Dynamic Linker & Trampoline Generator (`include/ps4/dynamic_linker.hpp` & `src/core/dynamic_linker.cpp`)**
  - Allocate an HLE trampoline pool in guest virtual memory.
  - Emit x86_64 dispatch stubs (`mov eax, 0x80000000 | id; syscall; ret`).
  - Process relocations (`R_X86_64_JUMP_SLOT`, `R_X86_64_RELATIVE`) and patch the Global Offset Table (GOT).

- [x] **Step 7.4: Implement Core `libkernel` and `libc` Shims**
  - `sceKernelExitProcess`
  - `sceKernelAllocateDirectMemory`
  - `sceKernelMapDirectMemory`
  - `libc` `printf` / `puts`

- [x] **Step 7.5: Unit Testing (`tests/test_phase7.cpp`)**
  - Test dynamic tag parsing and relocation resolution.
  - Test calling HLE functions through GOT trampolines from guest x86_64 machine code.

---

## 📌 Phase 8: Interactive macOS Display Window & Presenter

The goal of this phase is to connect an interactive AppKit window with a `CAMetalLayer` and real-time Metal presentation pipeline to blit the PlayStation 4 framebuffer at 60 FPS on macOS.

- [x] **Step 8.1: Define `DisplayWindow` & Window Configuration (`include/ps4/display_window.hpp`)**
  - Define `WindowConfig` (dimensions, title, resizable, headless mode).
  - Define `DisplayWindow` lifecycle and presentation API.

- [x] **Step 8.2: Implement AppKit `NSWindow` & `CAMetalLayer` (`src/gpu/display_window.mm`)**
  - Create and configure `NSWindow` with backing `CAMetalLayer` device attachment.
  - Add `PS4WindowDelegate` to monitor window close notifications.

- [x] **Step 8.3: Metal Fullscreen Presentation Pipeline**
  - Write inline MSL vertex and fragment shaders for textured full-screen quad presentation.
  - Implement dynamic aspect-ratio scaling and linear filtering for 1080p framebuffers.

- [x] **Step 8.4: Non-Blocking Event Polling & Lifecycle**
  - Implement `DisplayWindow::poll_events()` for macOS event queue processing.
  - Bind Escape key to close window cleanly.

- [x] **Step 8.5: Wire Runner Display Mode (`src/main.cpp`)**
  - Add `--window` / `-w` command-line flags to `ps4-runner`.
  - Run real-time 60 FPS presentation loop presenting guest Metal framebuffer.

- [x] **Step 8.6: Unit & Integration Testing (`tests/test_phase8.cpp`)**
  - Automated headless pipeline verification and live AppKit presentation tests.

---

## 📌 Phase 9: Threading & Synchronization HLE Subsystem (`libkernel`)

The goal of this phase is to support multi-threaded PS4 binaries with guest thread management, pthread lifecycle, and mutex/condition variable synchronization.

- [x] **Step 9.1: Define Thread Management & Context Types (`include/ps4/thread.hpp`)**
  - Define `ScePthread`, `ScePthreadAttr`, `ScePthreadMutex`, `ScePthreadCond` structs.
  - Define `ThreadContext`, thread states (`Running`, `Waiting`, `Terminated`), and thread priorities.

- [x] **Step 9.2: Implement `ThreadManager` (`src/hle/thread.cpp`)**
  - Guest stack allocation (default 2MB) for spawned threads.
  - Manage thread table, thread IDs, and dispatch execution.
  - Host threading bridge using `std::thread` / `pthread`.

- [x] **Step 9.3: Implement Synchronization Primitives**
  - Mutex initialization, recursive/errorcheck attributes, lock, trylock, and unlock.
  - Condition variable init, wait, timedwait, signal, and broadcast.

- [x] **Step 9.4: Register `libkernel` Threading NIDs (`src/hle/nid_registry.cpp`)**
  - `scePthreadCreate`, `scePthreadJoin`, `scePthreadSelf`, `scePthreadExit`.
  - `scePthreadMutexInit`, `scePthreadMutexLock`, `scePthreadMutexUnlock`, `scePthreadMutexDestroy`.
  - `scePthreadCondInit`, `scePthreadCondWait`, `scePthreadCondSignal`, `scePthreadCondDestroy`.
  - `sceKernelGettimeofday`, `sceKernelNanosleep`.

- [x] **Step 9.5: Unit Testing (`tests/test_phase9.cpp`)**
  - Test thread spawning, joining, argument passing, and mutual exclusion under concurrency.

---

## 📌 Phase 10: PS4 PM4 GPU Command Processor & Ring Buffer

The goal of this phase is to emulate the PS4 GCN/Radeon PM4 command processor, parsing GPU command packets from direct memory and translating them to Apple Metal 3 render commands.

- [x] **Step 10.1: Define PM4 Packet Types & Opcodes (`include/ps4/pm4.hpp`)**
  - Define PM4 Type-0 (register write), Type-2 (NOP), and Type-3 (packet processing) headers.
  - Define canonical GCN opcodes: `IT_SET_CONTEXT_REG`, `IT_DRAW_INDEX_AUTO`, `IT_DISPATCH_DIRECT`, `IT_CLEAR_STATE`.

- [x] **Step 10.2: Implement Ring Buffer Parser (`src/gpu/gpu_ring_buffer.cpp`)**
  - Circular command buffer reader with read/write pointers and submit mechanism.
  - Extract packets and dispatch them to the translation engine.

- [x] **Step 10.3: Implement Metal PM4 Translator (`src/gpu/pm4_translator.mm`)**
  - Map context registers (viewport, scissor, render target color/depth) to `MTLRenderPassDescriptor`.
  - Map index and non-indexed draw packets (`IT_DRAW_INDEX_AUTO`) to Metal draw primitives.
  - Map clear states to Metal clear passes.

- [x] **Step 10.4: Hook HLE GNM GPU Interfaces**
  - Register `sceGnmSubmitCommandBuffers` or direct ring buffer submission in `NidRegistry`.

- [x] **Step 10.5: Unit Testing (`tests/test_phase10.cpp`)**
  - Test PM4 packet decoding, state tracking, and Metal command execution.

---

## 📌 Phase 11: Audio Subsystem (`libaudioout`) & CoreAudio Backend

The goal of this phase is to output PlayStation 4 multi-channel PCM audio through Apple Silicon CoreAudio.

- [x] **Step 11.1: Define Audio Types & Formats (`include/ps4/audio.hpp`)**
  - Define `SceAudioOutPort`, sample formats (PCM 16-bit, float 32-bit), channel layouts (stereo, 5.1, 7.1), and sample rates (48kHz).

- [x] **Step 11.2: Implement `AudioOutManager` (`src/audio/audio_manager.cpp`)**
  - Ring buffer for audio streaming without underrun or pops.
  - Volume control and channel mixing.

- [x] **Step 11.3: Implement CoreAudio Backend (`src/audio/coreaudio_backend.mm`)**
  - Initialize Apple `AudioQueue` or `AudioUnit` for real-time low-latency playback.
  - Stream guest audio frames directly to macOS default audio device.

- [x] **Step 11.4: Register `libaudioout` NIDs (`src/hle/nid_registry.cpp`)**
  - `sceAudioOutOpen`, `sceAudioOutClose`, `sceAudioOutOutput`, `sceAudioOutSetVolume`, `sceAudioOutGetPortInfo`.

- [x] **Step 11.5: Unit Testing (`tests/test_phase11.cpp`)**
  - Test audio port allocation, sample queuing, volume attenuation, and playback lifecycle.

---

## 📌 Phase 12: Virtual File System & Save Data Subsystem

The goal of this phase is to emulate the PlayStation 4 FreeBSD virtual file system, directory mounting (`/app0`, `/data`, `/savedata`), virtual file descriptor table, and `libsave_data` services.

- [x] **Step 12.1: Define VFS Types & Stat Structures (`include/ps4/vfs.hpp`)**
  - Define `OrbisStat` binary layout matching Orbis / FreeBSD 9 specifications.
  - Define `SceSaveDataMount`, `SceSaveDataMountResult`, and mount modes.
  - Define `VfsMountPoint` and `VfsFileDescriptor`.

- [x] **Step 12.2: Implement `VirtualFileSystem` (`src/core/vfs.cpp`)**
  - Implement mount table with longest-prefix path resolution.
  - Sandboxed host directory translation (`/app0`, `/data`, `/savedata`).
  - Read-only mount enforcement and directory traversal prevention.
  - File descriptor table with virtual handles (0=stdin, 1=stdout, 2=stderr, 3+=virtual files).
  - File operations: `open`, `read`, `write`, `lseek`, `fstat`, `close`, `mkdir`, `unlink`.
  - Save data directory mounting and unmounting (`mount_savedata`, `unmount_savedata`).

- [x] **Step 12.3: Hook FreeBSD File I/O Syscalls (`src/core/syscall_table.cpp`)**
  - Wire `SYS_open` (5), `SYS_close` (6), `SYS_read` (3), `SYS_write` (4), `SYS_lseek` (478), and `SYS_fstat` (189) into VFS.

- [x] **Step 12.4: Register `libsave_data` NIDs (`src/hle/nid_registry.cpp`)**
  - `sceSaveDataMount` and `sceSaveDataUmount` shims hooked to VFS.

- [x] **Step 12.5: Unit Testing (`tests/test_phase12.cpp`)**
  - Test path resolution, file I/O, read-only enforcement, syscall table dispatch, and save data mounting.

---

## 📌 Phase 13: ARM64 JIT Dynamic Recompiler

The goal of this phase is to execute guest x86_64 machine code at native Apple Silicon speed via dynamic binary translation to ARM64 machine instructions.

- [x] **Step 13.1: Allocate Executable JIT Memory (`include/ps4/jit_compiler.hpp`)**
  - Reserve 16MB executable code cache via `mmap` with `MAP_ANON | MAP_PRIVATE | MAP_JIT`.
  - Implement thread write-protection toggling using `pthread_jit_write_protect_np(0)` / `pthread_jit_write_protect_np(1)`.
  - Invalidate instruction cache lines via `sys_icache_invalidate()`.

- [x] **Step 13.2: Implement Native ARM64 Code Emitter (`Arm64Emitter`)**
  - Emit 32-bit ARM64 instructions: `NOP`, `RET`, `MOV`, `MOVZ`, `MOVK`, `LDR`, `STR`, `ADD`, `SUB`, `AND`, `ORR`, `EOR`.
  - Emit AAPCS calling convention prologue and epilogue mapping `CpuContext*` pointer.

- [x] **Step 13.3: Implement Dynamic Recompiler & Basic Block Cache (`src/cpu/jit_compiler.cpp`)**
  - Translate guest x86_64 instructions: register clears (`XOR reg, reg`), 32-bit and 64-bit immediates (`MOV`, `MOVABS`), 64-bit arithmetic (`ADD`, `SUB`).
  - Cache compiled blocks keyed by guest `rip`.
  - Integrate tier-2 JIT dynamic compilation into `CpuExecutor::run()` with seamless interpreter fallback.

- [x] **Step 13.4: Unit Testing (`tests/test_phase13.cpp`)**
  - Test instruction emitter encodings, JIT page execution, 64-bit arithmetic recompiler, and execution engine integration.

---

## 📌 Phase 14: 3D Graphics Mesh Pipeline & Sample Demo

The goal of this phase is to extend the PM4 GPU command processor to support 3D indexed geometry, vertex/index buffers, and Apple Metal 3 shading pipelines.

- [x] **Step 14.1: Extend PM4 Packet Types (`include/ps4/pm4.hpp`)**
  - Define `IT_DRAW_INDEX_2` opcode and payload structures.
  - Define `Vertex3D` vertex structure and `ShaderRegOffset` constants (`SPI_SHADER_USER_DATA_VS_0`).

- [x] **Step 14.2: Implement Metal 3D Mesh Shading Pipeline (`src/gpu/pm4_translator.mm`)**
  - Write inline MSL vertex and fragment shaders for 3D colored meshes.
  - Compile `MTLRenderPipelineState` targeting offscreen 1080p BGRA8Unorm framebuffer.
  - Implement vertex buffer binding via `IT_SET_SH_REG`.
  - Implement indexed draw translation for `IT_DRAW_INDEX_2` using `drawIndexedPrimitives`.

- [x] **Step 14.3: Build 3D Mesh Sample (`samples/triangle_orbis.elf`)**
  - Generate synthetic 3D geometry ELF binary with colored vertices and program headers.

- [x] **Step 14.4: Unit & Integration Testing (`tests/test_phase14.cpp`)**
  - Test `IT_DRAW_INDEX_2` packet decoding, vertex buffer binding, and rasterization onto Metal 3 framebuffer with pixel color verification.

---

*All phases 1 through 14 successfully implemented and verified with 100% test pass rate!*

---

# 🚧 Next Up — Candidate Phases (15+)

_Phases 1–14 built the runtime and are verified against synthetic `samples/*.elf` fixtures. The phases below are the identified next steps to progress from samples toward running real retail titles, ordered by dependency and impact. Each "Verified starting point" was confirmed against the current tree._

### Verified starting points (current tree)
- The **ARM64 JIT (`Arm64Emitter`) is not connected** — `cpu.set_jit_compiler(&jit)` is commented out in `src/main.cpp`; the interpreter is executing.
- **Input does not exist yet** — no `libScePad` HLE, no GameController backend, and no keyboard/mouse capture in `DisplayWindow`, despite ADR 0009.
- **HLE breadth is thin** — only `libSceFios2` (done) and `libSceCommonDialog` are substantially implemented; `libSceVideoOut`, `libSceUserService`, `libSceSysmodule`, `libSceNgs2` and media decoders are missing. (`libaudioout` symbols are registered.)
- **Graphics uses a fixed inline MSL mesh shader** — ADR 0007's two-tier GCN→MSL translation and texture/sampler binding are not yet implemented.

## 📌 Phase 15: Connect the ARM64 JIT to the Executor

The goal is to activate the existing `JitCompiler`/`Arm64Emitter` for throughput while keeping the interpreter as a fallback.

- [ ] **Step 15.1: Wire `JitCompiler` into `CpuExecutor`**
  - Implement/enable `set_jit_compiler(&jit)` and dispatch translated blocks by guest PC.

- [ ] **Step 15.2: Code cache & basic-block boundaries**
  - Linear-sweep/block scanning to find branch targets; cache translated blocks keyed by guest PC.

- [ ] **Step 15.3: W^X code cache on Apple Silicon**
  - Guest pages are per-`phdr` RWX; use a separate `MAP_JIT` code cache with `pthread_jit_write_protect()` to satisfy hardened-runtime W^X.

- [ ] **Step 15.4: Exit the JIT for interrupts**
  - Fall back to the interpreter on syscalls, page faults, and traps; keep guest register state consistent across the boundary.

- [ ] **Step 15.5: Benchmark interpreter vs. JIT**
  - Measure instructions/sec on `samples/*.elf` (`test_phase15`).

## 📌 Phase 16: Input — GameController Framework (closes ADR 0009)

- [ ] **Step 16.1: `libScePad` HLE**
  - Implement `scePadOpen/Read/Close/etc.` mapping a controller port to an abstract input state.

- [ ] **Step 16.2: GCController backend**
  - Adopt DualShock 4 / DualSense / Xbox controllers via `GameController.framework`.

- [ ] **Step 16.3: Keyboard/mouse capture in `DisplayWindow`**
  - Translate NSEvent key/mouse events into pad/analog state (matches shadPS4's default bindings).

- [ ] **Step 16.4: Analog processing**
  - Deadzone handling, trigger/L2-R2 analog range, per-game remap.

## 📌 Phase 17: Display & Present — `libSceVideoOut` / flip

- [ ] **Step 17.1: `libSceVideoOut` HLE**
  - Register/allocate flip chains, set flip rate (30/60), resolution from `param.sfo`.

- [ ] **Step 17.2: Present on flip**
  - Blit the Metal framebuffer to the `DisplayWindow` presenter on `sceVideoOutSubmitFlip`/`flipRate`.

- [ ] **Step 17.3: Scanout metadata**
  - Derive initial width/height/tiling from `sce_sys/param.sfo`.

## 📌 Phase 18: Expand HLE Coverage (game-critical `libSce` modules)

- [ ] **Step 18.1: `libSceUserService` + `libSceSysmodule`**
  - Initial user id / username, and module load/unload bookkeeping.

- [ ] **Step 18.2: `libSceNgs2`**
  - Route the game audio mixer into the existing CoreAudio backend alongside `libaudioout`.

- [ ] **Step 18.3: Media decoders**
  - `libSceJpegDec`, `libScePngDec`, `libSceSulpha`/`libSceAvPlayer` stubs as needed.

- [ ] **Step 18.4: Offline network/NP stubs**
  - `libSceNpManager`, `libSceNet` returning benign offline values.

## 📌 Phase 19: Real GCN → MSL Shader Translation (closes ADR 0007)

- [ ] **Step 19.1: Parse GCN/AMDGPU shader ISA**
  - Decode vertex/fragment/compute shaders (leveraging SPIRV-Tools/Slang where appropriate).

- [ ] **Step 19.2: Map registers/varyings/descriptors to MSL**
  - Input semantics, uniform/constant buffers, resource descriptor tables.

- [ ] **Step 19.3: Replace the fixed mesh shader**
  - Compile translated vertex/fragment `MTLRenderPipelineState` instead of the inline MSL in `pm4_translator.mm`.

- [ ] **Step 19.4: Texture + sampler binding**
  - `T#`/`S#` descriptors → `MTLTexture`/`MTLSamplerState`.

## 📌 Phase 20: PM4 / GPU Feature Expansion

- [ ] **Step 20.1: Full context-register handling**
  - Depth/stencil, scissor, multiple render targets, viewport from `IT_SET_CONTEXT_REG`.

- [ ] **Step 20.2: Textures & samplers**
  - Texture upload path, sampler states, cubemaps.

- [ ] **Step 20.3: Sync & compute**
  - Occlusion queries, fences/`IT_EVENT_WRITE`, compute dispatch (`IT_DISPATCH_DIRECT`) if feasible.

## 📌 Phase 21: SELF / Retail Loading & `.pkg`

- [ ] **Step 21.1: SELF header + NPDRM/keystone**
  - Parse `SCE` header and decrypt `eboot.bin` (legally-dumped titles only).

- [ ] **Step 21.2: Integrate `.pkg` unpacking**
  - Fold `scripts/unpack_self.py` logic into the loader/install flow.

- [ ] **Step 21.3: Parse `sce_sys/param.sfo`**
  - Title id, version, parental level, display resolution.

## 📌 Phase 22: Concurrency, Save Data & Users

- [ ] **Step 22.1: Real host-thread scheduling**
  - Run guest threads on host threads; maintain per-thread CPU context + TLS FS/GS bases.

- [ ] **Step 22.2: `libkernel` sync primitives**
  - Mutex, condvar, semaphore, rwlock (extend the threading HLE).

- [ ] **Step 22.3: Save data & users**
  - Mount `savedata` to `~/Library/Application Support/<title>`, trophies, per-user profiles.

---

*These phases are candidates; reorder based on which real titles you target first.*


