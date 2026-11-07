# Xbox Controller Input Plan

**Goal:** Let a player use an Xbox Wireless / Series X|S controller (Bluetooth or USB) to play PS4 titles in the macOS runner, including analog sticks, analog triggers and rumble, while the keyboard/mouse path keeps working.

**Feasibility: yes.** Apple's GameController framework (macOS 11+, this machine runs 26.2) exposes Xbox controllers as `GCController` with a `GCExtendedGamepad` profile (`GCXboxGamepad` subclass) and `GCDeviceHaptics` for rumble. The emulator side needs only small changes. The guest already reads a full `ScePadData` through our HLE, and only the digital buttons are wired today. Sticks and triggers are hard-coded and rumble is a no-op. No change to the CPU, JIT or GPU paths.

**Status:** plan only. Nothing implemented. `src/hle/nid_registry.cpp` and `src/main.cpp` have other people's uncommitted work in the main checkout, so implement on a branch once that lands.

## How pad input works today

- **Host side.** `DisplayWindow::poll_events` (`src/gpu/display_window.mm:186`) maps Space, Enter, Up, Right and left click to Cross plus touch-pad click (`0x4000 | 0x100000`). It keeps `buttons` plus a `pressed_latch`, and `input_buttons()` returns `buttons | latch` and clears the latch.
- **Runner loop.** `src/main.cpp:325` runs on the AppKit main thread. It polls events, calls `nid_registry.set_pad_buttons(window->input_buttons())`, then presents a frame, or sleeps 2 ms when no frame is ready. So input is sampled roughly every 2 ms, or once per present. The guest runs on its own `std::thread`. `tests/test_flappy_gpu.cpp:133` does the same, and `PS4_AUTO_TAP` injects taps.
- **HLE side.** In `src/hle/nid_registry.cpp:483-516`:
  - `g_pad_buttons` is an atomic, and `g_pad_pressed_latch` reports a tap that started and ended between two guest reads.
  - `read_pad_buttons()` ORs them together and clears the latch.
  - `write_pad_data()` zeroes the 0x78-byte `ScePadData` and writes the buttons. It also writes sticks = 128, identity orientation, 1 g of acceleration, a fake centre touch when the touch-pad bit is set, `connected = 1` and a timestamp. `+0x08` (L2/R2 analog) stays 0.
- **Handlers.**
  - `scePadReadState` (`0x6277605EA41557B7`) and the unnamed `0xB445383A59BDC10B` write one sample. `scePadRead` (`0xAB570735F1B270B2`, at `:7415`) writes one sample and returns 1.
  - `scePadOpen` always returns handle `0x200`, whatever the `userId` or index.
  - `scePadGetControllerInformation` reports one connected DS4.
  - `scePadSetVibration`, `scePadClose` and the other setters are `SCE_OK` no-ops (`:7441`).
  - `sceUserServiceGetEvent` sends one LOGIN event for user `0x10000001`.
- **NIDs checked.** I recomputed the NIDs with the SHA-1 salt: the registered values for ReadState, Read, Open, Close, GetControllerInformation, SetLightBar and SetVibration are correct.
- **What Flappy Bird imports.** Its `eboot.bin` imports `scePadOpen`, `scePadRead`, `scePadClose`, `scePadSetLightBar`, `scePadSetVibration` and `scePadGetControllerInformation`. It does not import `scePadReadState`, `scePadOpenExt`, `scePadReadExt` or `scePadGetHandle`. So the guest reads through `scePadRead`, and it can call `scePadSetVibration`.

### ScePadData layout (0x78 bytes)

| Offset | Field | Today | After |
|---|---|---|---|
| 0x00 | `u32 buttons` | keyboard bits | keyboard OR pad bits |
| 0x04 / 0x05 | left stick x / y (u8, 128 = centre, y grows **down**) | 128 | from `leftThumbstick` |
| 0x06 / 0x07 | right stick x / y | 128 | from `rightThumbstick` |
| 0x08 / 0x09 | analog L2 / R2 (u8 0-255) | 0 | from `leftTrigger` / `rightTrigger` |
| 0x0C-0x33 | orientation, acceleration, angular velocity | identity / 1 g / 0 | unchanged (Xbox has no IMU) |
| 0x34-0x4B | touch data | centre finger while touch-pad bit set | unchanged |
| 0x4C | `connected` | 1 | 1 (see "No controller") |
| 0x50 | `u64 timestamp` | steady_clock µs | unchanged |
| 0x68 | `connectedCount` | 0 | 1, matching `GetControllerInformation` |

## Button and axis mapping

Positional mapping (the Xbox face layout matches the DS4 by position, not by label):

| Xbox (`GCExtendedGamepad`) | DS4 | `SCE_PAD_BUTTON_*` bit |
|---|---|---|
| A (`buttonA`) | Cross | `0x00004000` |
| B (`buttonB`) | Circle | `0x00002000` |
| X (`buttonX`) | Square | `0x00008000` |
| Y (`buttonY`) | Triangle | `0x00001000` |
| LB (`leftShoulder`) | L1 | `0x00000400` |
| RB (`rightShoulder`) | R1 | `0x00000800` |
| LT (`leftTrigger`) | L2 digital + analog `+0x08` | `0x00000100` |
| RT (`rightTrigger`) | R2 digital + analog `+0x09` | `0x00000200` |
| Left-stick click (`leftThumbstickButton`) | L3 | `0x00000002` |
| Right-stick click (`rightThumbstickButton`) | R3 | `0x00000004` |
| Menu ≡ (`buttonMenu`) | Options | `0x00000008` |
| View ⧉ (`buttonOptions`) | Touch-pad click (+ fake centre touch, as today) | `0x00100000` |
| D-pad up / right / down / left (`dpad`) | D-pad | `0x10` / `0x20` / `0x40` / `0x80` |
| Share (`GCXboxGamepad.buttonShare`, Series only) | unmapped (PS4 Share is system-reserved) | none |
| Xbox guide (`buttonHome`) | unmapped. macOS may grab it for Launchpad/Game Center. Set `preferredSystemGestureState = disabled` on it | none |

Axis conversion (GameController axes are floats in [-1, 1] with y **up**; DS4 bytes are 0-255 with 128 centre and y **down**):

```
stick_x = clamp(lround((x + 1.0f) * 127.5f), 0, 255)
stick_y = clamp(lround((1.0f - y) * 127.5f), 0, 255)
trigger = clamp(lround(value * 255.0f), 0, 255)
L2/R2 bit set when trigger.isPressed (framework threshold), mirroring the DS4 digital bit
```

GameController already applies a radial dead zone. `scePadGetControllerInformation` keeps reporting a dead zone of 13. That value is a hint the game applies itself, so no extra dead zone goes on the host side.

## Design

### Files

- **New `include/ps4/gamepad.hpp` + `src/input/gamepad_macos.mm`.** A `GamepadManager` class (pImpl, ARC) that:
  - subscribes to `GCControllerDidConnectNotification` / `GCControllerDidDisconnectNotification`. It does not call `startWirelessControllerDiscovery`, because pairing happens in System Settings;
  - sets `GCController.shouldMonitorBackgroundEvents = YES`, so input keeps flowing if the window loses focus (macOS 11.3+);
  - exposes `PadInput sample()`, which polls the active controller's `extendedGamepad` snapshot (no value-changed handlers), and `void apply_vibration(u8 large, u8 small)`.
- **`include/ps4/nid_registry.hpp`:**
  - add `struct PadInput { u32 buttons{0}; u8 lx{128}, ly{128}, rx{128}, ry{128}, l2{0}, r2{0}; };` (header-only, no ObjC);
  - add `void set_pad_state(const PadInput&)`. Keep `set_pad_buttons(u32)` as a wrapper with centred sticks, since `test_flappy_gpu` and `PS4_AUTO_TAP` use it;
  - add `PadVibration pad_vibration() const` so the host can read the guest's last rumble request.
- **`src/hle/nid_registry.cpp`:**
  - store the analog bytes in a `std::atomic<u64> g_pad_analog` (6 packed bytes) next to `g_pad_buttons`, with the button latch kept as it is;
  - `write_pad_data(mem, p, buttons, analog)` writes `+0x04..0x09` and `connectedCount`;
  - `scePadSetVibration(handle, const ScePadVibrationParam*)` reads `{u8 largeMotor; u8 smallMotor}` at `rsi`, validates the pointer, stores both in `std::atomic<u16> g_pad_vibration` and returns `SCE_OK`. Remove it from `kPadNoops`;
  - `scePadClose` resets the vibration to 0, so rumble never sticks after a game exits its pad loop.
- **`src/main.cpp` runner loop.** Build `PadInput in = gamepads.sample(); in.buttons |= window->input_buttons(); nid_registry.set_pad_state(in);` and then `gamepads.apply_vibration(nid_registry.pad_vibration())`.
- **`tests/test_flappy_gpu.cpp`.** Same merge when a window is present, so manual play with a pad works in the harness.
- **`CMakeLists.txt`.** Add `find_library(GAMECONTROLLER_FRAMEWORK GameController REQUIRED)` and `CoreHaptics`, add `src/input/gamepad_macos.mm` to `ps4_core` with `-fobjc-arc`, and add the new test target.
- **`include/ps4/display_window.hpp`.** Update the stale `input_buttons` comment. No other change.

### Threading

- **Everything GameController stays on the AppKit main thread** that already runs the runner loop. Connect/disconnect notifications are posted on the main thread. They are delivered while `poll_events()` spins the run loop through `nextEventMatchingMask:…inMode:NSDefaultRunLoopMode`. That also drains the main dispatch queue, which GameController uses as its default `handlerQueue`.
- **Polling, not handlers.** Each loop iteration (about 2 ms, or once per present) reads the current element values. This matches the existing keyboard sampling and needs no extra locks.
- **Host to guest.** `set_pad_state` does one relaxed store of the buttons atomic plus the press latch, as today, and one relaxed store of the 64-bit analog word. A guest read may pair buttons from one sample with axes from the next. That costs at most one 2 ms sample and is harmless for input, so a seqlock isn't worth it. The latch already guarantees that taps shorter than the guest's read interval are seen. Analog values are level-sampled with no latch.
- **Guest to host (rumble).** The `scePadSetVibration` handler runs on a guest thread. It must not call into CoreHaptics, so it only stores the atomic. The main loop compares against the last applied value and updates haptics only on change.

### Rumble

- `controller.haptics` returns a `GCDeviceHaptics`, or nil on old firmware or when unsupported. Create two `CHHapticEngine`s, one with `GCHapticsLocalityLeftHandle` and one with `GCHapticsLocalityRightHandle`. If either fails, fall back to a single `GCHapticsLocalityDefault` engine.
- On each engine, start one long continuous `CHHapticEvent` (`hapticIntensity` = 1, effectively infinite duration) through an `CHHapticAdvancedPatternPlayer`. Then drive it with `sendParameters:` `CHHapticDynamicParameterIDHapticIntensityControl`, using `largeMotor / 255` on the left (low-frequency) handle and `smallMotor / 255` on the right handle. Intensity 0 stops the player.
- The DS4 rumble semantics is "stay at this level until told otherwise". The continuous player plus a re-arm when it ends matches that.
- Stop and release the engines on disconnect, and when `scePadClose` resets the vibration.
- Xbox impulse triggers (`GCHapticsLocalityLeftTrigger` / `RightTrigger`) have no DS4 equivalent, so leave them unused.

### Hotplug and multiple pads

- **v1 (this plan): single player.** The active controller is the most recently connected extended gamepad. On disconnect, fall back to the next one in `GCController.controllers`, or to none.
- The controller's state is dropped the moment it disconnects (`sample()` returns a neutral `PadInput`), so no button sticks.
- Set `controller.playerIndex = GCControllerPlayerIndex1` so the controller's LED shows player 1.
- **v2 (deferred): local multiplayer.** This needs real per-user plumbing, which is a separate change:
  - `sceUserServiceGetLoginUserIdList` returns several users, and `sceUserServiceGetEvent` emits one LOGIN per controller;
  - `scePadOpen(userId, …)` hands out a distinct handle per user;
  - pad state becomes an array indexed by handle;
  - read handlers look up by `rdi`, and `scePadGetHandle` gets registered.

  Flappy Bird is single-player, so v1 is enough to play it.

### No controller connected

- Behaviour stays exactly as today: the keyboard/mouse virtual pad is always "connected" (`+0x4C = 1`), with sticks centred and triggers 0.
- Reporting `connected = 0` when the Xbox pad goes away would make many games pause on a "reconnect your controller" screen while the keyboard still works. So don't do it in v1.
- Keyboard and controller are OR-merged, so both work at once.
- Log one line on connect/disconnect (`[Pad] Xbox Wireless Controller connected`).

## Test strategy

1. **New CTest `PadInput` (`tests/test_pad_input.cpp`, links `ps4_core`).** Same harness shape as `tests/test_guest_exceptions.cpp`: `MemoryManager` → `SyscallTable` → `NidRegistry::register_default_libraries()`. It checks:
   - `set_pad_state({buttons = Cross|L1|R2, lx = 0, ly = 255, rx = 200, ry = 50, l2 = 10, r2 = 255})`, then `find_by_nid(0x6277605EA41557B7)` (scePadReadState) with `rsi = buf`. Assert the bytes at `+0x00` (u32), `+0x04..0x07`, `+0x08/0x09`, `+0x4C == 1` and `+0x68 == 1`;
   - the same through `scePadRead` (`0xAB570735F1B270B2`, `rdx = 1` returns `rax == 1`) and the unnamed `0xB445383A59BDC10B`;
   - the latch: `set_pad_state(Cross)` then `set_pad_state(0)` before a read still reports Cross once and 0 on the next read. Analog is not latched;
   - `set_pad_buttons(x)` still gives sticks 128 and triggers 0, so the old wrapper is unchanged;
   - `scePadSetVibration` with `{large = 200, small = 30}` at `rsi` makes `pad_vibration()` return `{200, 30}`; a bad pointer returns an error and leaves it unchanged; `scePadClose` resets it to `{0, 0}`.
2. **Mapping unit test, no hardware.** Put the Xbox-to-DS4 conversion in a pure C++ function, `PadInput map_xbox(const XboxSnapshot&)`, where `XboxSnapshot` is a plain struct of floats and bools. `gamepad_macos.mm` fills the struct from `GCExtendedGamepad`, and the test checks the table above plus the axis edge cases (-1, 0, +1, y inversion, clamp). The test can live in the same `test_pad_input.cpp`.
3. **Manual check (needs the user's controller).** Pair an Xbox controller in System Settings > Bluetooth and run Flappy Bird windowed:
   - A flaps;
   - the View button acts as touch-pad click;
   - unplugging or turning the pad off mid-game leaves no stuck button and the keyboard still works;
   - rumble fires if the game calls `scePadSetVibration` (log the first call to confirm it does);
   - optional `PS4_PAD_TRACE=1` prints the sampled `PadInput` once per second.

## Risks / things to verify in a 30-minute spike first

- **Unbundled binary.** GameController delivering controllers to an unbundled CLI binary (the runner is launched via `NSTask` from the library GUI, and has `NSApplicationActivationPolicyRegular` but no Info.plist). This is expected to work, but confirm it first with a 20-line probe that logs `GCController.controllers`. If it doesn't, we may need `GCSupportsControllerUserInteraction` / `GCSupportedGameControllers` in an embedded Info.plist (`-sectcreate __TEXT __info_plist`).
- **USB.** Bluetooth is the documented path for Xbox Wireless controllers on macOS. USB-wired Xbox pads are recognised on recent macOS but should be checked on the user's machine.

## Open questions for the user

1. Positional mapping (Xbox A = Cross, B = Circle) or label-swapped (B = Cross, Japanese-style)? The plan assumes positional and could add a `PS4_PAD_SWAP_AB=1` toggle.
2. Is the View button as touch-pad click right? Some players expect Back/View = Share, but PS4 Share is system-reserved, so touch pad is the more useful choice.
3. Is single player (one active controller) enough for now, or do you want local multiplayer (the v2 per-user plumbing) planned in?
4. Should the Xbox guide button do something emulator-side, such as quit the game (like Escape today) or pause?
5. Should the game library GUI also be navigable with the controller, or only in-game?
6. Which controller and connection will you use (Xbox Series X|S over Bluetooth, Xbox One, or USB)? That decides what the manual test covers.
