## What this PR does

Closes #

## How it was tested

- [ ] `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
- [ ] `cd build && ctest --output-on-failure`
- [ ] NID audit (only if `src/hle/nid_registry.cpp` changed):
      `python3 scripts/audit_nids_libdoc.py`

## Checklist

- [ ] Clean-room: no Sony code, keys, firmware, `*.prx`/`*.sprx`, `*.pkg`, `eboot.bin`, or game content
- [ ] Glossary terms match `GLOSSARY.md`; ADRs checked (`docs/adr/`)
- [ ] No new compiler warnings (`-Wall -Wextra -Wpedantic`)
- [ ] One focused change; docs/ROADMAP updated if a phase step moved
