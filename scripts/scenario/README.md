# Event-driven scenario runner

Build once, then run one scenario process at a time:

```sh
cmake -B build-opt -DCMAKE_BUILD_TYPE=Investigate
nice -n 10 cmake --build build-opt --target ps4-runner -j4
python3 scripts/scenario/runner.py scripts/scenario/recipes/flappy.json
python3 scripts/scenario/runner.py scripts/scenario/recipes/flappy.json --repeat 2 --fast-forward
python3 scripts/scenario/runner.py scripts/scenario/recipes/dragon_ball_title.json --fast-forward
```

Do not run a build concurrently with a game. The runner takes a per-user machine
lock and refuses to launch if `pgrep -x ps4-runner` finds another runner, including
one launched outside this tool. It launches repeats sequentially, defaults to one,
and caps development repeats at two and timeouts at 180 seconds. It never builds
or invokes a second emulator for image comparison. The three-run Dragon Ball
budget used during development is documented in `VALIDATION.md`; do not repeat
those discovery runs casually on the user's machine.

`--output /absolute/new/directory` selects the artifact bundle; existing directories
are rejected. The default is `build-opt/scenarios/<name>-<UTC timestamp>/`.
`--runner path` selects another Investigate binary; `--timeout seconds` overrides
the recipe deadline. Normal runs exit 0 only when every repeat passes its
expectations. A missing milestone, timeout, observed guest fault, runtime control
error, incomplete capture, image mismatch, or unexpected process exit fails a run.
A timeout's supervisor-generated signal is reported separately from a guest crash.

## Recipe format (JSON, version 1)

Paths in `eboot` and capture `reference` are relative to the repository root;
absolute paths also work. Milestone names and step IDs must be unique identifiers.
`after` dependencies reference an earlier milestone in the array. All delays,
button durations and gaps are **minimum host monotonic seconds**, not guest time. If guest polling stalls, input delivery stretches the sequence to preserve each transition.

```json
{
  "version": 1,
  "name": "dialog-example",
  "eboot": "games/your_game/eboot.elf",
  "timeout": 120,
  "fast_forward_until": "eula_shown",
  "milestones": [
    {"name": "boot", "source": {"type": "start"}},
    {"name": "dialog_open", "after": "boot",
     "source": {"type": "nid", "name": "sceMsgDialogOpen"}},
    {"name": "eula_shown", "after": "dialog_open", "role": "scene",
     "source": {"type": "log", "pattern": "\\[MsgDialog\\].*LICENSE"}}
  ],
  "steps": [
    {"id": "pages", "after": "eula_shown", "delay": 2,
     "action": "press", "buttons": ["right"], "repeat": 28,
     "duration": 0.1, "gap": 0.3},
    {"id": "accept", "after": "eula_shown", "delay": 14,
     "action": "press", "buttons": ["cross"], "duration": 0.2},
    {"id": "eula_capture", "after": "eula_shown", "delay": 1,
     "action": "capture", "reference": "tests/golden_frames/eula.ppm",
     "tolerance": 0.08}
  ],
  "expected": {"milestones": ["eula_shown"],
               "forbid_milestones": [], "max_unsupported_packets": 0}
}
```

A `gap` is the released interval between presses, so the example's press starts
are 0.4 seconds apart. Overlapping button steps OR their masks and release only
their own buttons. A transition waits for a guest HLE pad read to observe its
mask before the next transition is delivered. A capture stall therefore delays
page turns instead of collapsing several pulses into one. Supported buttons are `cross`, `circle`, `square`, `triangle`,
`right`, `left`, `up`, `down`, `options`, `escape`, `touchpad`, `l1`, and `r1`.
**Escape maps to Options/Start (`0x0008`)**, matching `display_window.mm`; it is
not a host window-close command. Scenario input replaces `PS4_AUTO_PRESS` and
`PS4_AUTO_MOVE`. The runner removes automatic-input variables, and recipes cannot
reintroduce them through `env`.

Source types:

| Source | Required fields | Meaning |
| --- | --- | --- |
| `start` | none | Scenario service initialized, before CPU execution. |
| `nid` | `name` | First eligible call of a configured named HLE handler. Preserves the original handler. Names must already exist in the registry. |
| `shader` | `hash`, optional `stage` (`VS`, `PS`, `CS`) | First eligible translated draw/dispatch resource scan using that shader code hash. |
| `texture` | `address` | First eligible successful resource scan that samples this decoded T# base address. Image loads/stores alone do not qualify. |
| `log` | `pattern` | First eligible raw guest/runtime log line matching a Python regular expression, including `[MsgDialog]` output. |
| `input` | `id` | Acknowledgment that a guest pad read sampled a specific input command, e.g. `pages_27_release`. This is a navigation milestone, not proof of the resulting UI. |

A milestone emits once, after its dependency. An event that enables a child does
not simultaneously satisfy that child. Shader and texture fingerprints must be
verified against captures for the intended game/build; a shader can be reused
across scenes and addresses can move. A named HLE event proves a call, not that a
dialog was drawn or a call succeeded. The HLE name wrappers use the existing
`find_by_name` registry lookup: aliases under different names must be listed
separately, and direct guest PRX implementations bypass HLE wrappers.

The emulator emits one JSON object per line:

```text
[milestone] {"name":"eula_shown","source":"log","frame":123,"flip":122,"t":35.25}
[scenario] {"type":"input","id":"pages_0_press","buttons":32,"frame":124,"flip":123,"t":37.26}
```

`frame` is the backend submission number used by GPU captures; `flip` is the guest
video-out flip count. The epoch is service initialization, not process creation.
Log matching happens in the Python reader, which forwards the event to the
emulator over its private FIFO. Those milestone times include pipe/reader latency
(typically milliseconds); guest text is never interpreted as a control command.
NID, shader and texture events are observed directly inside the runtime.

## Actions, capture, and comparisons

`press` schedules a chord and release, optionally repeated. `[scenario]` events
distinguish `input` (applied to host controller state) from `input_sampled` (seen
in the guest pad buffer). Runtime `input` milestones fire on the sampled event.
If a game uses an unwrapped/direct PRX pad implementation, sampling cannot be
confirmed and the run fails rather than silently advancing its input sequence. `capture` queues the
existing debug GPU capture for the next frame that begins, preserving its JSON,
PNG, and pass-timing behavior. Scenario captures also write RGB PPMs alongside
PNGs. Several queued captures are serviced one per subsequent frame. If rendering
stops, a queued capture remains incomplete and the report fails instead of
inventing an image. `stop` requests a graceful stop at a main CPU batch boundary;
a blocked runtime gets up to eight seconds before forced cleanup.

A capture's optional `reference` uses `downscale_image`, `load_reference`, and
`compute_mae` from `scripts/golden_frame_test.py`; no duplicate comparator and no
second game launch. Supply an existing 160x90 reference as supported by that tool.
PPM references work without Pillow. Missing references fail; this runner never
creates or updates golden baselines automatically.

## Fast-forward limits

Fast-forward is **off by default**. `--fast-forward` runs without the debug
presentation limiter until `fast_forward_until` fires, then enables 60 Hz debug
pacing. `--no-fast-forward` paces throughout. Recipes still obey their input delays
and process deadlines. If the selected milestone never occurs, an enabled run
stays unpaced; the report records this choice and the missing expected milestone.

The scenario runner is headless, so it sets `PS4_NO_VSYNC=1` and supplies its own
switchable pacing from the debug frame-completion hook. It preserves rendering,
guest clocks, HLE sleeps, scheduling, file reads, audio, and all emulated work.
Games already pacing through guest timers may show little speedup. This is not a
checkpoint/resume facility or a promise of proportional acceleration.

## Reports and discovery

Each repeat contains a frozen `recipe.json`, full line-buffered `emulator.log`,
`report.json`, and `captures/frame_N/`. The outer `summary.json` records the pass
rate and furthest milestones. Reports include revision, dirty-worktree status, binary SHA-256, command, host duration,
return code, supervisor termination, crash evidence, milestone times, input
commands/acknowledgments, captured artifacts, golden comparisons, and unsupported
packet totals. `furthest_observed_scene` counts only milestones tagged
`"role": "scene"`; input acknowledgments do not upgrade it.

Scene FPS uses guest flip-counter deltas between telemetry samples inside each
`role: scene` interval, ending at the next scene milestone. Navigation and
bookkeeping milestones do not split scene intervals. A short scene with fewer than two samples has `fps: null`,
not zero. Reports state the measured interval and sample count. Unsupported
counts use the translator's existing total counter. Stops provide a final metric
sample; forced timeouts retain the most recent available sample.

`--discover` records the first observed shader hashes and sampled texture bases
(up to 2048 unique fingerprints) as `[scenario]` JSON events. Combine this with
explicit milestone-relative capture steps, inspect their images, then choose a
verified fingerprint. Discovery is opt-in and does not invent semantic scene
names. Inherited `PS4_*` switches are cleared for reproducibility. Recipe `env` can pass ordinary runtime settings such as
`PS4_GUEST_GC_DONT_GC`; control paths and automatic input are reserved.

## Tests

```sh
python3 -m unittest discover -s scripts/scenario -p 'test_*.py'
ctest --test-dir build-opt --output-on-failure -j1
```

The Python tests use synthetic events, a fake control-protocol process and tiny
PPM fixtures. They never start a game or consume the Dragon Ball run budget.
