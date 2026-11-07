# Scenario Runner Implementation Plan

> **For agentic workers:** Execute inline, sequentially, using the approved scenario requirements. No builds overlap emulator runs.

**Goal:** Reach and measure Dragon Ball Xenoverse 2's title-menu sequence using reproducible event-driven recipes.

**Architecture:** A Python standard-library runner validates JSON recipes, starts one process, matches guest logs, schedules controller inputs, and writes run/repeat reports. An opt-in Objective-C++ debug service emits JSON milestone lines for configured HLE names and shader/texture events, receives JSON commands through a private FIFO, and reuses GPU capture. Existing main initialization installs only configured HLE wrappers and controller callbacks; CPU and game-fix files remain untouched.

**Tech Stack:** Python 3 standard library, C++20, Foundation JSON, POSIX FIFO, existing capture and golden-frame comparison APIs.

## Global Constraints

- Base `ad87475`, branch `codex/scenarios`, isolated existing worktree.
- Never edit scripts/run_workflow.py, scripts/extract_games.py, tests/test_extract_games.py, src/cpu/, src/hle/nid_registry.cpp or GPU implementation files.
- Build with `nice -n 10`, at most `-j4`. One emulator process, no concurrent build.
- At most three Dragon Ball runs, each at most 180 seconds; development repeats at most two.
- Event milestones, not frame-number triggers. Timing is host monotonic elapsed time.

### Task 1: Recipe validation and scheduling

Files: scripts/scenario/model.py, runner.py, test_scenario.py, recipes/flappy.json.

- [x] Test missing/duplicate milestone IDs, invalid trigger/action types, unsafe paths and negative durations, dependency ordering, repeated button timing, and failure classification.
- [x] Implement strict JSON loading, event predicates, a heap of milestone-relative actions, and pass/fail expectations.
- [x] Implement sequential process runs, timeout/group cleanup, complete logs, JSON reports, and an exclusive run lock.
- [x] Verify with `python3 -m unittest discover -s scripts/scenario -p 'test_*.py'` using fake processes/events; no real game needed.

### Task 2: Emulator debug event/control service

Files: include/ps4/debug/scenario.hpp, src/debug/scenario.mm, src/debug/capture.mm, src/main.cpp, CMakeLists.txt.

Interfaces: scenario_start(NidRegistry&), scenario_stop(), scenario_shader(stage,hash), scenario_texture(address), scenario_frame(frame), scenario_capture_requested(), scenario_capture_completed(path).

- [x] Load configured milestones only behind PS4_SCENARIO; emit `[milestone] {name,frame,t,source}` once per eligible milestone.
- [x] Install wrappers for selected named HLE symbols from main via the existing registry API. Preserve their original handlers; report unknown names.
- [x] Detect shader and sampled texture events through existing capture_resources hooks even when JSON capture is off.
- [x] Read private FIFO commands for buttons, observed log events, captures and graceful scenario stop. Emit acknowledgments and periodic frame/unsupported counters.
- [x] Reuse capture_begin/end and texture output; expose PPM alongside PNG for golden-frame API reuse.
- [x] Add optional 60 Hz debug pacing: normal throughout when fast-forward is off, unpaced until the selected milestone otherwise. Do not change guest clocks, omit GPU work or modify HLE sleeps.
- [x] Build `nice -n 10 cmake --build build-opt -j4`, then validate Flappy inputs, milestones and captures sequentially.

### Task 3: Reports, comparison and Dragon Ball discovery

Files: scripts/scenario/compare.py, recipes/dragon_ball_title.json, README.md, docs/agents/debug-switches.md.

- [x] Import downscale_image/load_reference/compute_mae from scripts/golden_frame_test.py; reject missing references and report comparison results without launching another emulator.
- [x] Report monotonic milestone times, furthest stage, missing expectations, return code/crash evidence, scene FPS from counter deltas, unsupported-packet deltas, capture paths, input acknowledgments and repeat pass rate.
- [x] Define Dragon Ball stages: logos, movie, auto-accepted save dialog, seven precaution pages, 29 EULA pages/Accept and next observed stage. Gate scripted navigation on actual matching evidence; do not infer UI arrival from elapsed time or NID calls alone.
- [x] Use up to three bounded Dragon Ball observations to refine event fingerprints and navigation. Record the actual furthest confirmed milestone, including incomplete or unknown stages.
- [x] Run ctest sequentially, document measured results and limitations, commit; do not push or merge.


## Execution notes

Implemented and reviewed inline on `codex/scenarios`. Runtime hooks stay in
`src/debug/` and main initialization; named HLE wrappers use the existing public
registry API. No CPU, HLE implementation, GPU implementation, or off-limits
workflow/extraction files changed.

Input delivery was strengthened after the third discovery run showed capture
stalls coalescing page turns: each state now waits for an actual guest pad read.
This fix passed Flappy repeat and synthetic delayed-poll tests. The final Dragon
Ball recipe uses the captured EULA Accept texture as its event fingerprint; its
full title-menu path remains unconfirmed. See `scripts/scenario/VALIDATION.md`.

The load-rule check failed in one respect: the newly merged GoldenFrameRegression
CTest entry implicitly launched an additional 25-second Dragon Ball run after the
three intentional discovery runs. This was not checked before invoking CTest.
The suite was not rerun, and no further games were launched afterward.
