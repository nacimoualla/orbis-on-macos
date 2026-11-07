# Scenario runner validation — 2026-11-07

Branch `codex/scenarios`, based on `ad87475`. Build type Investigate, with
`nice -n 10 cmake --build build-opt -j4`. Build commands and this agent's game
launches were sequential. No CPU, HLE implementation, GPU implementation, or
protected workflow/extraction files were changed.

## Tooling checks

- Eleven Python unittest methods pass. Tests cover recipe/dependency validation,
  event parsing, repeated/overlapping buttons, Escape's existing Options mapping,
  scene counters, actual runtime fault wording, golden comparison and missing or
  incorrectly sized references.
- A fake process exercises the full FIFO/log/report path. Delayed guest samples
  preserve all four press/release pairs; missing guest samples fail the run.
  These fixtures do not start an emulator or use game assets.
- A two-repeat Flappy run passed **2/2 (100%)**, including NID, shader and sampled
  texture milestones, Cross press/release acknowledgments from actual guest pad
  reads, next-frame JSON/PNG/PPM capture, graceful stop and golden comparison.
  Normalized image MAE was 0.0027593 and 0.0027287 against tolerance 0.08.
- The initial Flappy smoke run exercised pacing with fast-forward disabled;
  later runs enabled it until the rendering milestone. Differences in startup
  time also include shader-cache warming, so they are not a controlled speedup
  measurement.
- CTest's pre-existing 30-test suite: **29 passed, 1 failed**. The failing
  `GoldenFrameRegression` test matched Flappy, Sonic and Dragon Ball logos but
  did not capture Dragon Ball frame 810 within its fixed 25-second run. Its
  frame-based timing assumption is unrelated to the event-driven runner.
- The newly registered `ScenarioTools` CTest entry passed separately (11 unittest
  methods, no game launch). The entire suite was not rerun afterward.

A comparator fixture initially used a full-resolution PPM as a reference and
correctly failed the comparison. The reference was then downscaled through the
existing golden-frame API to its required 160x90 size. Original failure reports
were retained; the subsequent two passing runs use the corrected reference.

## Dragon Ball: what was actually reached

Three intentional observation runs used a 180-second process deadline each.
Timeout cleanup/report creation added about 0.1–0.2 seconds to their host duration.
They used different evolving recipes, so they are not a statistical repeat set.

1. **Observation only:** no navigation input. Captures showed Dimps/logo output
   and the opening sequence; no save dialog appeared before timeout.
2. **Skip discovery:** an early shader candidate proved to be pre-movie UI,
   rather than the movie. A later manual Escape probe was delivered over the
   control FIFO at 112.75 seconds and produced a transition into 2D UI. Its
   acknowledgments remain in the report. This run was not wholly automated.
3. **Event-driven navigation:** the corrected movie shader triggered Escape at
   31.843 seconds. The following events were measured:

   | Event | Seconds after debug-service initialization |
   | --- | ---: |
   | Logos shader | 4.097 |
   | Opening-sequence shader | 29.819 |
   | Post-movie UI texture | 43.440 |
   | First `sceMsgDialogOpen` call | 49.573 |
   | Save-data text in `[MsgDialog]` | 49.578 |

   The message was “DRAGON BALL XENOVERSE 2 save data could not be found. Do you
   want to create new save data?” The HLE auto-accept behavior was preserved.
   The first post-dialog capture at 52.314 seconds visibly showed **EULA page
   1/29**. Subsequent captures reached **page 11/29** and stayed there. **The
   title menu and EULA acceptance were not confirmed.** Unsupported-packet
   telemetry remained zero; the run timed out without guest-crash evidence.

The initial input implementation acknowledged host-state application only.
Several expensive captures overlapped its 28 short Right pulses; page turns were
lost while the guest was not polling. The final implementation waits for each
state to be read by a guest pad handler, and preserves minimum press/gap times.
This fix passed Flappy and delayed-poll fixtures **after** the Dragon Ball budget
was consumed. It has not been retested on Dragon Ball.

The final recipe uses the observed EULA Accept-button texture `0x67034200`, first
sampled at 49.801 seconds in run 3 and present in its verified EULA capture, to
gate the 28-page navigation. Its seven-page Precautions sequence remains anchored
to the post-movie UI backdrop and the user-confirmed ordering; Precautions text
was not independently recognized. The optional text milestone stays unmet rather
than being synthesized from completed button presses. The title-menu expectation
also requires independent evidence and will fail until that scene is observed.

## Load-budget mistake

After the three intentional Dragon Ball runs, invoking the full CTest suite
implicitly launched **one additional 25-second Dragon Ball run** through the
newly merged `GoldenFrameRegression` test. That test command should have been
inspected first. This exceeded the requested three-launch limit. No subsequent
games were launched, and the full CTest suite was not rerun. Use an explicit test
selection when preserving a game-specific run budget.

## Local evidence

Artifacts are machine-local under:

- `/tmp/ps4-scenarios-dragon-1/`, `dragon-2/`, `dragon-3/`
- `/tmp/ps4-scenarios-flappy-verified/` (passing two-run bundle)
- `/tmp/scenario-ctest.log` and `/tmp/scenario-python-ctest.log`

In Dragon Ball run 3, `captures/frame_1879/target_0000000000000000.png` shows EULA
1/29, and `captures/frame_2348/target_0000000000000000.png` shows EULA 11/29.
Report inputs, captures and fingerprints are retained; no game images or other
proprietary assets are committed.
