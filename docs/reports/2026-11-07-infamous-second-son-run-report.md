# inFAMOUS Second Son — latest run report

Examined: `Logs/gui_20261107-174116-345.log` (90,701 bytes, 1,596 text lines).
Run filename starts at 17:41:16 on 2026-11-07; file last modified at 17:43:01.
These filesystem times imply roughly 105 seconds, not an instrumented execution time.

## Result

The title loaded and ran initialization code, then repeatedly stalled in an HLE
sleep/poll loop. The log does not establish successful gameplay or a rendered game
frame. No fatal signal, assertion, or unhandled-instruction termination is recorded.
The runner ended with host exit status 0, but its shutdown watchdog had to hard-exit
because guest threads remained parked. That status cannot be used as a compatibility
or successful-gameplay result.

## Launch and successful initialization

- Windowed execution, instruction budget 0 (unlimited), no `--no-jit` flag.
- Valid ELF loaded at entry point `0x61dfd0`; three loadable segments mapped.
- 95,679 RELA relocations and 460 JMPREL relocations read, none unread.
- `/app0` mounted to the imported title directory.
- Apple M4 Metal backend initialized; 1920×1080 BGRA8 framebuffer and a clear pass
  executed; interactive display presentation loop opened.
- Guest registered two video buffers at `0xac568000` and `0xacd60000`.
- Guest opened/read `playgo.pgm` and its save icon. Save-data initialization and
  trophy setup ran. Five job-worker threads and one trophy thread started.
- Main execution reported 28,331,890 guest instructions; worker accounting reported
  40,001. JIT reported 6,164 cached blocks. Its zero invocation count is not proof
  that JIT was unused: `src/cpu/jit_compiler.cpp` increments that counter only when
  metrics collection is enabled.

## Primary observed stall

Eight watchdog snapshots (four `stall-a` / `stall-b` pairs) consistently put main
in `WaitingSleep`, blocked in HLE NID `0xd637d72d15738ac7`, caller `0x30612a`.
The local NID registry maps this to `sceKernelUsleep`.

Main's HLE count increases from 6,861 to 86,013 between snapshots. End-of-run
histograms show 90,498 `sceKernelUsleep` calls. This supports repeated polling /
sleeping, rather than one permanently blocked sleep call.

All five job workers are shown waiting for synchronization; trophy waits on an
event queue. These waits can be normal for idle threads. Combined with the repeated
main-thread wait and no recorded frame submissions, a startup dependency not
completing is a strong hypothesis, not a proven deadlock or identified root cause.

The last visible asset-related path is `/app0/art/cache/all_shaders.xpps`, followed
by unresolved calls and watchdog snapshots. The file exists and is 636,272 bytes,
so the log does not justify saying it is missing. The unresolved archive/path calls
near this path deserve investigation before attributing the stall to shader
translation itself.

## Unresolved HLE coverage

90 fallback warnings across 13 distinct NIDs are recorded. Fallbacks return zero,
which may falsely signal success without performing the requested operation.

| Unresolved NID | Calls | Observed context |
| --- | ---: | --- |
| `0xfd125634c2b77c2f` | 71 | Early initialization, caller `0x3c2f34` |
| `0x1757423fbaa4aa89` | 5 | Startup calls with timing-like arguments; symbol unconfirmed |
| `0x35490118e640462e` | 2 | Includes caller `0x30605b`, near the eventual polling loop |
| `0x2ac553734e2437d9` | 2 | Calls involving `delete_me.snt` test paths |
| `0xa57bdb0df721bba9` | 2 | `/app0/art/cache/all_shaders.xpps` and `/in_archive` arguments |
| `0x2e8b0b9473a936a4` | 1 | Startup |
| `0x8143c688e435b664` | 1 | Startup |
| `0x367ef32b09c0e6ad` | 1 | Startup |
| `0x4ac0bf9bf4bd2530` | 1 | Startup |
| `0xf6f036696f821ee0` | 1 | Before asset directory setup |
| `0xe11ebf3af2367040` | 1 | Trophy initialization vicinity |
| `0xdbda0abca5f3119a` | 1 | Later initialization |
| `0x7abe93910154be9d` | 1 | Immediately after shader-cache archive/path fallback |

Names for these unresolved symbols are not established by the local registry.
Their argument patterns alone are insufficient to name their APIs conclusively.

## Direct-memory allocation anomaly

The log contains:

```
Allocated 0x48000000 bytes at phys 0x40000000 (next=0x88000000)
Allocated 0x20000000 bytes at phys 0x88000000 (next=0xa8000000)
Allocated 0xffffffff97ff0000 bytes at phys 0xa8000000 (next=0x3fff0000)
```

The final requested length is implausibly large, and the allocation cursor moves
backward from `0xa8000000` to `0x3fff0000`. The local allocation handler checks
`cur + length` against its ceiling without first checking integer overflow, then
stores the wrapped sum and reports success when the output pointer is valid.
This is a concrete allocator arithmetic defect. It can permit overlapping later
allocations. The log alone cannot establish whether it caused this particular stall
or why the guest supplied that length.

## Exit interpretation

The log reports `Process exited (SYS_exit)`, process exit code 0, and GUI exit 0.
However, the runner also sets its exited flag when the display loop ends (including
window closure), so these messages do not prove the guest intentionally completed.
There is no corresponding logged `Guest called exit(...)` line in this session.

The final `[shutdown] watchdog: threads still parked after 2s, hard-exiting` confirms
an incomplete graceful teardown. Window closure is a plausible ending trigger;
the precise trigger is not recorded. No runner process remained at inspection.

## Rendering limits

Metal's initialization clear pass and display-loop message are confirmed. The
log contains no recorded PM4/draw/flip submissions or presented-game-frame count.
It therefore provides no evidence that a game scene rendered. Absence of those
messages is not proof of zero frames when relevant instrumentation is disabled.
The display-loop's “60 FPS” message describes its configured loop, not measured
in-game performance.

## Package and extraction context

The import receipt marks extraction complete and the source PKG removed after
validation. The extracted game remains present. Its SFO says title ID `CUSA00004`,
category `gd`, application version `01.00`, package version `01.07`, and content ID
`EP9000-CUSA00004_00-SECONDSONSHIP000`. A source filename containing “Update” alone
is insufficient to conclude the library contains an unusable update-only package.
ELF/SFO import validation establishes launch eligibility, not runtime compatibility.

## Recommended investigation order

1. Add overflow/range/alignment validation to direct-memory allocation. Reject
   impossible requests without advancing its cursor; capture caller/arguments.
2. Resolve the archive/path fallback NIDs around `all_shaders.xpps`, then the
   `0x35490118e640462e` call immediately before the polling region. Implement their
   actual contracts rather than returning success by default.
3. Capture guest disassembly and the condition around `0x30605b`–`0x30612a`, and
   determine which completion flag or job the main loop is waiting for.
4. Instrument worker wakeups and frame/flip counts to distinguish loading progress,
   idle workers, missing callbacks, and real presentation.
5. Distinguish user-stop/window-close, guest exit, crash, and forced teardown in
   GUI summaries; a green exit 0 should not imply the title is playable.

This report records observations and code-backed interpretations. No runtime
compatibility fixes were made as part of this log review.
