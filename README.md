# orbis-on-macos

**A PlayStation 4 (PS4) emulator & Orbis compatibility layer for Apple Silicon macOS.**

[![Build](https://github.com/nacimoualla/orbis-on-macos/actions/workflows/ci.yml/badge.svg)](https://github.com/nacimoualla/orbis-on-macos/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/nacimoualla/orbis-on-macos?style=social)](https://github.com/nacimoualla/orbis-on-macos/stargazers)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)](https://github.com/nacimoualla/orbis-on-macos)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Build](https://img.shields.io/badge/build-CMake%203.25%2B-brightgreen)](CMakeLists.txt)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ouallanacim)

`orbis-on-macos` is an open-source **PlayStation 4 emulator** — a clean-room **userland compatibility layer & runtime** that runs PS4 (Orbis OS) games and apps **natively on Apple Silicon macOS**. Instead of emulating the whole console, it loads a PS4 `eboot.bin`/ELF or PRX module, maps the guest's 8 GB address space into the Mac's unified memory, dynamically recompiles guest `x86_64` code to native `ARM64`, translates the GNM/Metal GPU command stream and delivers audio through CoreAudio — all on top of Darwin/Mach and Apple frameworks.

> **Status:** Phases 1–15 implemented and verified (see [ROADMAP.md](ROADMAP.md)). CPU interpreter + ARM64 JIT, PRX/NID dynamic linker, Orbis syscall HLE, VFS, Metal graphics pipeline, CoreAudio, and threading are all in place.
>
> **Milestone:** a retail Unity title (Flappy Bird, `NPXX51362`) boots on the **real Mono runtime** (`mono-ps4.prx`, full-AOT, Boehm GC), loads its first scene, renders through the GCN → MSL path and is **playable at ~60 FPS** in the window — title screen, touch-pad input, gameplay, game over and restart. See [Running a real Unity title](#-running-a-real-unity-title).

## 🗺️ Progress & Roadmap

[![Progress](https://img.shields.io/badge/Roadmap-68%25%20complete-blue)](ROADMAP.md) [![Phases](https://img.shields.io/badge/Phases-15%2F22-8250df)](ROADMAP.md) [![Steps](https://img.shields.io/badge/Steps-64%2F88-success)](ROADMAP.md)

```
Phases   1–15 (runtime + JIT) ████████████████████  100%  ✅ built & tested
Phases   1–22 (full roadmap)  ██████████████░░░░░░   68%  15/22 phases · 64/88 steps
```

The **runtime foundation and ARM64 JIT are complete and tested** — ELF/PRX loader, 8 GB memory, x86_64 interpreter + ARM64 JIT dynamic binary translator, syscall HLE, NID linker, VFS, Metal pipeline, CoreAudio, and threading (Phases 1–15).

The remaining **Phases 16–22** are what move it from *synthetic `samples/*.elf`* to *real retail titles* — input, broader HLE, real GCN→MSL shaders, and retail loading. Tracked on the [Next-Up board (#1)](https://github.com/nacimoualla/orbis-on-macos/issues/1):

- **✅ Foundation + JIT (Phases 1–15):** done (no tracking issue — see [ROADMAP.md](ROADMAP.md))
- [ ] [#3](https://github.com/nacimoualla/orbis-on-macos/issues/3) · **P16** — Input (`libScePad` + GameController) — `ready-for-agent` ← *start here*
- [ ] [#4](https://github.com/nacimoualla/orbis-on-macos/issues/4) · **P17** — Display & present (`libSceVideoOut`) — `ready-for-human`
- [ ] [#5](https://github.com/nacimoualla/orbis-on-macos/issues/5) · **P18** — Expand HLE coverage — `needs-triage`
- [ ] [#6](https://github.com/nacimoualla/orbis-on-macos/issues/6) · **P19** — Real GCN→MSL shaders — `ready-for-human`
- [ ] [#7](https://github.com/nacimoualla/orbis-on-macos/issues/7) · **P20** — PM4/GPU expansion — `ready-for-human` *(blocked by #6)*
- [ ] [#8](https://github.com/nacimoualla/orbis-on-macos/issues/8) · **P21** — SELF/`.pkg` retail loading — `ready-for-human`
- [ ] [#9](https://github.com/nacimoualla/orbis-on-macos/issues/9) · **P22** — Concurrency + save data — `ready-for-human`

---

## ⚖️ Legal & Scope

- This project is a **clean-room reimplementation** of the documented *userland interfaces* of Orbis OS. It does **not** contain, and is not derived from, any Sony proprietary source code, firmware, keys, or BIOS.
- The runtime **does not ship or link any proprietary assets** — notably **no PS4 system modules** (`libSce*.prx` / `*.sprx`), **no keystones**, and **no game content**. Everything under [`games/`](games/) and [`build/`](build/) is **git-ignored and never published**.
- To run real titles you must **supply your own, legally obtained** game binaries. This project does not enable piracy and provides no copyrighted material.
- "PlayStation", "PS4", and "Orbis" are trademarks of Sony Interactive Entertainment. This project is not affiliated with or endorsed by Sony.

See [LICENSE](LICENSE) for terms.

---

## ✨ Features

- **ELF64 / PRX loader** for Orbis executables and relocatable modules (Sony `PT_SCE_*` segments handled).
- **8 GB unified virtual memory manager** — guest flat address space mapped directly into Apple Silicon unified memory.
- **x86_64 interpreter** with a full **ARM64 dynamic binary translator (JIT)** and inline syscall compilation.
- **Orbis syscall HLE table** + **NID registry / dynamic linker** for symbol resolution and trampolines.
- **Virtual File System** (`/app0`, `/data`, `/savedata`) and save-data subsystem.
- **Direct Metal graphics backend** with a **PM4 command processor**, ring buffer, and a **two-tier GCN → MSL shader** path; offscreen framebuffer + interactive window presenter.
- **CoreAudio** output with a lock-free SPSC ring between the guest audio thread and the host callback.
- **Threading HLE** (`libkernel`) mapping guest threads onto host pthreads.
- **Headless test suite** — CPU, memory, linker, and offscreen Metal rendering tested without a GUI.

## 🧭 How it works (architecture)

| Guest layer | Runtime translation |
|---|---|
| Guest `x86_64` CPU instructions | **ARM64 dynamic binary translator (JIT)** + interpreter fallback |
| Orbis **NID**-keyed symbols / PRX imports | **NID registry + dynamic linker** trampolines |
| Orbis **system calls** (`libkernel`/`libSce*`) | **Inline-compiled dispatch table** → host handlers (HLE) |
| **GNM / PM4** GPU command stream | **PM4 translator** → Metal pipeline states |
| **GCN** shader bytecode | **Two-tier** translation → Metal Shading Language |
| 8 GB guest address space | **Fixed-base** mapping into macOS unified memory |
| `libaudioout` | **CoreAudio** backend via lock-free ring |
| Input | **GameController framework** abstraction |

These decisions are documented as ADRs in [`docs/adr/`](docs/adr/):

- [0001 — Userland compatibility layer over full-system emulation](docs/adr/0001-userland-compatibility-layer.md)
- [0002 — Dynamic binary translation (JIT) over static recompilation](docs/adr/0002-dynamic-binary-translation.md)
- [0003 — Direct Metal graphics over MoltenVK](docs/adr/0003-direct-metal-graphics-pipeline.md)
- [0004 — High-level emulation for system libraries](docs/adr/0004-high-level-emulation-system-libraries.md)
- [0005 — Fixed-base virtual memory allocation](docs/adr/0005-fixed-base-virtual-memory.md)
- [0006 — Inline syscall compilation over signal traps](docs/adr/0006-inline-syscall-compilation.md)
- [0007 — Two-tier GCN → SPIR-V → MSL shader pipeline](docs/adr/0007-two-tier-shader-pipeline.md)
- [0008 — Embedded JIT translation core](docs/adr/0008-embedded-jit-core.md)
- [0009 — GameController framework for input](docs/adr/0009-gamecontroller-input-abstraction.md)

## 💻 Requirements

- **macOS** on **Apple Silicon** (arm64). Intel Macs are not supported (the JIT targets ARM64).
- **Xcode Command Line Tools** (for Apple Clang + frameworks).
- **CMake ≥ 3.25**.
- A C++20 compiler (Apple Clang) and Objective-C++ (ARC) support — Metal / AppKit / CoreAudio frameworks are required at build time.

## 🔨 Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds the `ps4_core` library, the `ps4-runner` CLI, the `test_phase*` binaries and the focused regression tests (guest exceptions, printf/scanf, Flappy GPU smoke, …).

## ▶️ Running

```bash
./build/ps4-runner [options] <path-to-eboot.bin-or-elf>
```

Options: `--window/-w`, `--headless`, `--budget N` (headless instruction budget,
default 100 M), `--no-jit`, `--trace/-t`, `--strict-hle`.

| Option | Meaning |
|---|---|
| `--window`, `-w` | Open an interactive macOS Metal window (60 FPS presentation loop). |
| `--headless` | Run console-only (default). |
| `--no-jit` | Disable ARM64 JIT dynamic recompiler (interpreter only). |
| `--trace`, `-t` | Trace guest instruction execution step-by-step. |
| `--help`, `-h` | Show usage. |

Example with the bundled synthetic sample ELF:

```bash
./build/ps4-runner --window samples/triangle_orbis.elf
```

The runner loads the binary, maps segments into guest memory, initializes the VFS (mounting the binary's parent directory as `/app0`), resolves NID symbols, initializes the Metal GPU, then executes guest code via the ARM64 JIT recompiler and reports progress.

Tip: overlay Apple's Metal performance HUD (FPS, frame time, GPU load) on the emulator window with:

```bash
MTL_HUD_ENABLED=1 ./build/ps4-runner --window <path-to-eboot>
```

**Bring your own titles.** Drop legally obtained game files under `games/<Title>/` (that directory is git-ignored) and point `ps4-runner` at an `eboot.bin`/`.elf`. No game data is included in this repository.

## 🎮 Running a real Unity title

Unity PS4 games ship their scripting runtime as PRX modules (`mono-ps4.prx`, `MonoAssembliesPS4.prx`). Set `PS4_LOAD_PRX=1` to load and run them for real instead of the HLE Mono shim:

```bash
PS4_LOAD_PRX=1 ./build/ps4-runner --window games/<Title>/eboot.elf
```

**Controls:** left click, `Space`, `Enter`, `↑` or `→` press **Cross** and the **touch pad** together (games that say "touch pad to start" read the touch pad). `Esc` closes the window.

What this path exercises: full-AOT Mono with Boehm GC stop-the-world (guest exception delivery via `sceKernelRaiseException`), Unity's preload/scene-load threads, FMOD audio paced by `sceAudioOutOutput`, the user-service login that opens the pad, Gnm command-buffer building (`sceGnmSet/Update*Shader*`, draws, EOP interrupts) and AVX-heavy engine math in the JIT.

**Headless smoke test.** `test_flappy_gpu` drives the same title offscreen; it expects the game at `games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/` and fails without it (exclude it with `ctest -E Flappy`):

```bash
PS4_LOAD_PRX=1 PS4_FRAMES=300 PS4_AUTO_TAP=130,2,1 PS4_GPU_READBACK=/tmp/frames ./build/test_flappy_gpu
```

| Variable | Effect |
|---|---|
| `PS4_LOAD_PRX=1` | Load and execute the game's own PRX modules (real Mono). |
| `PS4_FRAMES=N` | Host frames to run (`test_flappy_gpu`). |
| `PS4_AUTO_TAP=start,period[,hold]` | Press touch pad + Cross on a schedule so a headless run can play. |
| `PS4_GPU_READBACK=/dir` | Dump every drawn frame as `frame_N_gcn.ppm`. |
| `PS4_FRAME_TIMES=1` | Per-frame wall time vs. the main thread's guest slice. |
| `PS4_SAMPLE_MAIN=1` | One main-thread RIP + frame-pointer chain sample per frame. |
| `PS4_HLE_HIST=1` | HLE call histogram plus wall time spent in each handler, per guest thread. |
| `PS4_TRACE_RANGE=lo,hi` / `PS4_TRACE_AFTER=addr` | Single-step the main thread and print RIPs in a range / after an address. |
| `PS4_TRACE_TID=n` | Per-instruction trace of guest thread `n`. |
| `PS4_WATCH_ADDR=a` `PS4_WATCH_SIZE=n` | Record every store to a guest range (dumped on malformed GPU packets and at exit). |
| `PS4_DUMP_QWORDS=a,b,…` | Print guest qwords at exit. |
| `PS4_DUMP_TRAMPOLINES=1` | Map HLE trampoline addresses to NIDs/names. |
| `PS4_LOG_SEMA=1` | Log every `sceKernelWaitSema`/`SignalSema`. |
| `PS4_JIT_STATS=1` | JIT vs. interpreter coverage telemetry (see below). |

## 🧪 Testing

The suite is headless — it exercises CPU translation, memory mapping, the linker, and offscreen Metal rendering without opening a window (24 CTest targets):

```bash
cd build && ctest --output-on-failure
# or run an individual phase:
./build/test_phase15
```

| Phase | Focus |
|---|---|
| 1–3 | Types, ELF/PRX loader, 8 GB virtual memory |
| 4–5 | Orbis syscall table, x86_64 executor |
| 6, 8 | Offscreen Metal framebuffer, display window |
| 7 | Dynamic linker & NID resolution |
| 9–11 | Threading (libkernel), PM4 GPU processor, audio (CoreAudio) |
| 12–15 | Virtual File System, ARM64 JIT recompiler, 3D mesh pipeline, JIT-executor integration |

A standalone Python suite audits the HLE NID registry (hash vectors, tag round-trips, malformed inputs, registry + eboot cross-reference) without the CTest harness:

```bash
python3 tests/test_verify_nids_adversarial.py
```

### x86 differential check against real hardware

`scripts/x86_diff/run.sh` runs every case in `scripts/x86_diff/cases.h` (generated by `gen_cases.py`, 500+ instruction encodings) natively under Rosetta 2 and through the interpreter and the JIT from identical random state, and reports any register, flag or memory difference. Add a case whenever an instruction is implemented or fixed.

### CPU/JIT testing off-Mac

- `scripts/cpu_harness.sh [phase|tests/foo.cpp ...]` — cross-compiles the CPU sources plus the given tests to Linux/aarch64 and runs them under `qemu-aarch64` (default: phases 5, 13, 15), so JIT work can be tested on any Linux box.
- `scripts/cpu_harness.sh tests/test_jit_diff.cpp` — differential fuzzer: random x86_64 sequences (grouped templates: `alu_rr`, `alu_imm8`, `mov_mem`, `lea`, `stack`, …) run on the interpreter (reference) and with the JIT attached from identical state; GPRs, RIP, defined flags and scratch memory must match. Mismatches are shrunk to a minimal repro with hex bytes; per-group JIT coverage is reported. Knobs: `JIT_DIFF_SEED`, `JIT_DIFF_TRIALS`, `JIT_DIFF_GROUPS=alu_rr,mov_mem`, `JIT_DIFF_LEN`, `JIT_DIFF_CHECK_FLAGS=0` (full list in the file header; `--list` prints groups).
- `scripts/jit_fuzz.sh [seed ...]` — runs the fuzzer over several seeds and summarizes pass/fail, summed coverage and unique minimized failures.

### Debugging HLE coverage

The HLE registry is checked against [ps4libdoc](https://github.com/idc/ps4libdoc), whose symbol tables make a wrong NID *label* provable rather than a matter of taste. Three corpora are cached under `data/libdoc/`:

- `libkernel.sprx.json` — per-module dump: symbol NIDs plus the library that exports them (libkernel, libScePosix, libSceLibcInternal, libSceGnmDriver).
- `known_names.txt` — 42,010 symbol names across every PS4 module; resolves a NID to a name without a module dump.
- `mono_embedding_api.txt` — every `mono_*` identifier in the official Mono headers, for mono-ps4 NIDs.

Scripts, all stdlib-only:

- `python3 scripts/audit_nids_libdoc.py` — **the one to run after touching `nid_registry.cpp`.** Compares every registration against all three corpora, then hashes the label itself so NIDs ps4libdoc cannot name are still checked. Cross-references the guest's import tables (eboot + every PRX) to separate a live bug from a dead registration. Exits non-zero only on a mislabeled NID the guest actually imports.
- `python3 scripts/crack_mono_nids.py [--all] [--refresh]` — hash-verifies the `mono_*` labels against the Mono embedding API; `--all` lists every Mono NID the guest imports with its real API name.
- `python3 scripts/verify_nids.py [--eboot <eboot.bin|elf>] [--registry <nid_registry.cpp>]` — validates the Sony NID hash against test vectors, audits every `register_symbol` name/NID pair, and cross-references a game's imports.
- `python3 scripts/check_log_nids.py <run-log> [names.txt]` — flags every `[HLE CALL] label (NID=…)` line whose label doesn't hash to its NID.
- `python3 scripts/crack_nids.py` — hashes every string in the game's `eboot.bin` and Mono PRXs against target NIDs.

Why this matters: labels are load-bearing. `0x7501D612C26DA04E` was registered as `scePadInit` when it is `pthread_mutexattr_init`, which is why the trace showed a nonsensical "scePadInit ×42"; the same NID also carried a pad-state writer onto Mono's `sem_init`, overwriting guest semaphores. Run the audit after any registry edit.

### Debugging JIT coverage

Two tools rank the x86 instructions to implement next in the ARM64 JIT:

- **Runtime telemetry** (off by default): run with `PS4_JIT_STATS=1` (writes `./jit_stats.json` at exit), `PS4_JIT_STATS=/path/x.json`, or `PS4_JIT_STATS=-` (stderr). It counts guest instructions retired by JIT blocks vs the interpreter, every interpreter-executed opcode (weighted by execution count), and where/why the block compiler stopped (`terminator` vs `unsupported`). In code: `ps4::dump_jit_stats(std::ostream&)` (`include/ps4/jit_stats.hpp`).
- `python3 -I scripts/jit_coverage.py jit_stats.json [more.json ...] [--top N] [--markdown]` — merges runs and prints the interpreted opcodes ranked by dynamic count with mnemonic, % of interpreted instructions, cumulative %, and terminator/unsupported.
- `python3 -I scripts/opcode_census.py <elf> [...] [--top N] [--markdown] [--assume '8B mem']` — static census of an ELF's executable segments (samples, an unpacked eboot/PRX): mnemonic histogram, JIT coverage, and unsupported opcodes ranked by static count. `--assume` answers "what if the JIT handled X?". Uses `capstone` (`pip install capstone`) or falls back to `objdump`.

Opcode keys are normalized the same way in all three: prefixes/REX stripped (REX.W, fs/gs, lock reported as notes), 66/F3/F2 kept, `/n` = ModRM group extension, `reg`/`mem` = r/m form (e.g. `8B mem`, `83 /7 reg`, `66 0F 6F mem`). Off-Mac, `scripts/cpu_harness.sh tests/test_jit_stats.cpp` runs the telemetry test under qemu-aarch64.

At runtime, a stall watchdog in the thread manager dumps per-thread RIP/GPRs after ~2 s of per-thread HLE silence — look for `[WATCHDOG stall-a/b]` blocks in run logs when the game stops progressing.

## 📁 Project structure

```
.
├── CMakeLists.txt          # Build: ps4_core lib, ps4-runner, test_phase1..15
├── include/ps4/            # Public headers (per subsystem)
├── src/
│   ├── core/               # ELF loader, memory, syscall table, linker, VFS
│   ├── cpu/                # x86_64 interpreter + ARM64 JIT
│   ├── gpu/                # Metal backend, display window, GPU ring buffer, PM4
│   ├── audio/              # Audio manager + CoreAudio backend
│   └── hle/                # NID registry, threading
├── samples/                # Synthetic test ELF fixtures (hello/sample/dynamic/triangle)
├── tests/                  # test_phase1..15 (CTest)
├── scripts/                # NID audit vs ps4libdoc, Mono NID cracker, log checker, SELF unpacker, JIT coverage tools
├── data/libdoc/            # ps4libdoc symbol corpora used by the NID audit
├── vfs_root/               # Default VFS mount points (/app0, /data, /savedata)
├── docs/
│   ├── adr/                # Architecture Decision Records (0001–0009)
│   └── agents/             # Domain/issue-tracker docs for engineering agents
├── ROADMAP.md              # Phase-by-phase checklist (1–15)
├── GLOSSARY.md             # Orbis/macOS terminology
└── LICENSE
```

## 📚 Documentation

- [ROADMAP.md](ROADMAP.md) — interactive, phase-by-phase implementation plan.
- [GLOSSARY.md](GLOSSARY.md) — canonical terminology (Orbis, GNM, NID, PRX, HLE, …).
- [docs/adr/](docs/adr/) — architecture decision records.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to pick an issue, build, test, and open a PR.
- [SECURITY.md](SECURITY.md) — how to report vulnerabilities privately.

## 🤝 Contributing

Contributors welcome — the fastest way in is [#3 · Phase 16 Input](https://github.com/nacimoualla/orbis-on-macos/issues/3) (`ready-for-agent`), tracked under the [Next-Up board (#1)](https://github.com/nacimoualla/orbis-on-macos/issues/1). Five-minute path: fork → `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` → `cd build && ctest --output-on-failure` → open a focused PR. Full guide in [CONTRIBUTING.md](CONTRIBUTING.md). Clean-room only: never commit Sony code, keys, firmware, `*.prx`/`*.sprx`, `*.pkg`, `eboot.bin`, or game content.

## 🙏 Related projects

- **[shadPS4](https://github.com/shadps4-emu/shadPS4)** — the leading cross-platform PS4 emulator (Windows / Linux / macOS / FreeBSD), built on Vulkan. If you want an end-user GUI today, start there.

`orbis-on-macos` takes a **Mac-first** angle: an Apple Silicon–native, Metal/CoreAudio userland runtime with an `x86-64 → ARM64` dynamic binary translator, explored as a from-scratch, clean-room build.

## ☕ Support

If this project helps you, consider [buying me a coffee](https://buymeacoffee.com/ouallanacim) — it funds the late-night emulator debugging. Every cup is genuinely appreciated. 🙏

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ouallanacim)

## 📄 License

Released under the [MIT License](LICENSE). Provided "as is", without warranty of any kind.

**Reminder:** this repository contains no proprietary Sony software or game assets. You are responsible for ensuring any binaries you run with it are used in accordance with applicable laws and licenses.
