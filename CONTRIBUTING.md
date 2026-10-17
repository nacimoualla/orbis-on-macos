# Contributing to orbis-on-macos

Thanks for stopping by. This is a from-scratch, clean-room PS4 userland runtime for Apple Silicon macOS — ELF/PRX loader, 8 GB memory map, x86_64 interpreter + ARM64 JIT, HLE syscalls, Metal graphics, CoreAudio. Phases 1–15 are done; Phases 16–22 (input, video-out, HLE breadth, real shaders, retail loading) are where contributors matter most.

## Start here (5 minutes)

```bash
# Fork, then:
git clone https://github.com/<you>/orbis-on-macos.git
cd orbis-on-macos
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
```

Requirements: Apple Silicon Mac, Xcode Command Line Tools, CMake ≥ 3.25. Intel Macs are not supported (JIT targets ARM64).

Run the sample:

```bash
./build/ps4-runner --window samples/triangle_orbis.elf
```

## Where to help

- **Roadmap board:** [#1 — Next Up (Phases 15–22)](https://github.com/nacimoualla/orbis-on-macos/issues/1)
- **Best first issue:** [#3 — Phase 16 Input (`libScePad` + GameController)](https://github.com/nacimoualla/orbis-on-macos/issues/3) — labeled `ready-for-agent`, self-contained, closes ADR 0009.
- Also good: [#4](https://github.com/nacimoualla/orbis-on-macos/issues/4) (video-out), [#6](https://github.com/nacimoualla/orbis-on-macos/issues/6) (GCN→MSL shaders), [#8](https://github.com/nacimoualla/orbis-on-macos/issues/8) (SELF/pkg), [#9](https://github.com/nacimoualla/orbis-on-macos/issues/9) (threads/save data).
- Labels: `ready-for-agent` = fully specified, start now · `ready-for-human` = needs a human · `needs-triage` = under discussion · `good first issue` / `help wanted` = newcomer-friendly.

No Mac hardware for a hardware-only path (e.g. GameController)? The headless tests still count — note it in your PR and a maintainer will verify on device.

## Ground rules

1. **Clean-room only.** No Sony code, firmware, keys, BIOS, system modules (`*.prx`/`*.sprx`), keystones, or game content. Never commit anything from `games/`, `*.pkg`, `*.self`, `eboot.bin`, `sce_sys/`, `sce_module/`. Bring-your-own-titles only.
2. **Use the glossary.** `GLOSSARY.md` is canonical (NID, HLE, PRX, GNM…). Don't invent synonyms it avoids.
3. **Check ADRs.** `docs/adr/0001–0009` record why (userland over full-system, JIT over static, Metal over MoltenVK…). If your change contradicts one, say so explicitly in the PR.
4. **After touching `src/hle/nid_registry.cpp`, run the NID audit.** Labels are load-bearing — a wrong label attaches the wrong handler:
   ```bash
   python3 scripts/audit_nids_libdoc.py
   python3 tests/test_verify_nids_adversarial.py
   ```

## Pull requests

- One focused change per PR. Link the issue (`Closes #3`).
- Include: what changed, how you tested (`ctest` + which `test_phase*`), and — for JIT/GPU/audio — interpreter-vs-JIT or headless-vs-window result if relevant.
- Match surrounding C++20 style. Build is `-Wall -Wextra -Wpedantic`; keep it warning-free.
- Small PRs get reviewed fastest. Docs-only and test-only PRs are welcome.
- By contributing you agree your work is under the MIT License (see `LICENSE`).

## Getting help

- Open a [question issue](https://github.com/nacimoualla/orbis-on-macos/issues/new/choose) or comment on the phase issue you're working on before large changes — a 2-line "I'm taking X" avoids duplicate work.
- Be kind per [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security issues go to [`SECURITY.md`](SECURITY.md), never a public issue.
