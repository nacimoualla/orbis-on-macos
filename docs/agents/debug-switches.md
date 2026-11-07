# Debug switches and investigation tools

Build with assertions and optimization:

```sh
cmake -B build-opt -DCMAKE_BUILD_TYPE=Investigate
cmake --build build-opt -j 6
ctest --test-dir build-opt --output-on-failure
```

Environment switches are read once at startup or first use. Unset a switch to
turn it off; presence switches remain enabled even when their value is `0`.
Guest addresses use C integer syntax (`0x…` for hexadecimal), except
`PS4_DUMP_CODE`, whose two fields are hexadecimal.

## Frame capture and unattended investigations

```sh
PS4_DEBUG_DIR=/tmp/flappy-debug PS4_GPU_CAPTURE_FRAME=120 \
  PS4_AT_FRAME=121:exit build-opt/ps4-runner --headless --budget 0 \
  games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/eboot.elf

PS4_DEBUG_DIR=/tmp/db-debug PS4_AT_FRAME=300:sample PS4_AUTO_PRESS=2 \
  build-opt/ps4-runner --headless --budget 0 \
  games/dragon_ball_xenoverse_2/CUSA05350/eboot.elf
```

| Switch | Behavior |
| --- | --- |
| `PS4_GPU_CAPTURE_FRAME=N` | Capture backend frame N into `frame_N/report.json` and `target_<guest-key>.png`. Frame numbers are **1-based Metal submissions**, including initialization clears and empty submissions, not wall-clock display refreshes or the game's own flip counter. Use a later frame to see actual rendering. |
| `PS4_GPU_CAPTURE_XCODE=N` | Capture frame N with `MTLCaptureManager` into `frame_N/frame.gputrace`. Set `MTL_CAPTURE_ENABLED=1` before starting the process; a missing capture facility is reported to stderr. Open the trace in Xcode. This can be combined with JSON capture. |
| `PS4_DEBUG_DIR=path` | Parent directory for frame folders and sample files. Default `debug-captures` relative to the working directory. Created automatically. Use separate directories per run; use an absolute path for predictable artifacts. |
| `PS4_AT_FRAME=N:dump` | Capture the same GPU JSON/PNGs as `PS4_GPU_CAPTURE_FRAME=N`; also print guest memory mappings and thread states to stderr at frame completion. |
| `PS4_AT_FRAME=N:sample` | Spawn `/usr/bin/sample` against the runtime's own PID for 8 seconds, writing `sample_frame_N.txt`. Execution continues during sampling. Keep the process alive for at least 8 seconds after the trigger. |
| `PS4_AT_FRAME=N:exit` | Flush output and terminate with status 0 immediately after frame completion. This deliberately skips normal runtime teardown and final statistics. |
| `PS4_PAGE_WATCH=addr[,size]` | Watch stores on all host pages overlapping the guest range; default size 8 bytes. Works with JIT-generated stores and host/HLE stores. Pages must already be readable/writable at startup. |

JSON operations preserve draw/dispatch order, counts/grid, VS/PS/CS code hashes,
all eight colour-target keys/dimensions/formats/write masks, viewport, scissor,
blend, depth/stencil state, and the raw context-register shadow. Resource entries
contain stage and SGPR slot, raw T# and S# words, decoded texture fields and sampler
fields. Shader-load and scalar-resource-scan failures are recorded explicitly;
unknown descriptors cannot be inferred reliably. Hashes use the pipeline's GCN
code/fetch-code hash. An independent scan also records resources on fallback draws.

Only the selected JSON frame commits and waits after each render/compute pass.
`passes` gives GPU elapsed milliseconds, pass order, actual Metal attachments, completion status and errors. Each encoded operation links to its pass and actual attachment keys/sizes/Metal formats; operations that never acquire an encoder retain a null pass. Timing covers the isolated pass command buffer, including any preceding blits.
A frame's final buffer is also waited before readback. The PNG set includes the
screen (key zero) and every initialized cached 8-bit colour target at frame end,
including targets retained from earlier frames. R8/RG8/RGBA8/BGRA8 variants are
supported; integer and signed targets show their raw byte patterns. Higher bit
depths and depth/stencil targets are not converted to PNG. Capture changes timing
on the selected frame and can be large.

Page-watch diagnostics include the faulting guest address, old value, deferred
post-store value, guest TID and guest RIP. With the watch enabled, the JIT publishes
the current instruction RIP before each translated instruction, including chained
blocks. This adds instrumentation overhead only in watch runs. A signal handler
removes write protection and writes a fixed-size event into a nonblocking pipe;
a timer drains it and reprotects the original page permissions every 10 ms. The
handler chains unrelated SIGBUS/SIGSEGV faults to the previous disposition.

Page granularity means writes outside the requested byte range but on the same
page are logged too. Stores within the unprotected 10 ms interval are coalesced;
the deferred value may include subsequent writes. A saturated event pipe drops
events. This is a diagnostic sampler, not an exhaustive store trace. Use
`PS4_WATCH_ADDR` with `--no-jit` when exact per-store old/new bytes are needed.
Do not watch pages the guest will unmap or change permissions on during the run.

## Existing runtime diagnostics

| Switch | Behavior |
| --- | --- |
| `PS4_FLIP_STATS=1` | Guest flip FPS, pacing/hitches and GPU busy time to stderr; window loop also reports presentation progress. |
| `PS4_JIT_STATS=1` or `path` | JIT coverage/block metrics JSON at normal exit; `1` selects `jit_stats.json`, `0` disables. Analyze with `scripts/jit_coverage.py`. |
| `PS4_RUN_SECONDS=N` | Stop the **windowed** host loop after N seconds. Headless runs use `--budget N` (0 means unlimited) or a frame-exit trigger. |
| `PS4_HLE_HIST=1` | Collect per-thread HLE call counts and elapsed time; dump at exit. |
| `PS4_HLE_HIST_EVERY=N` | In window mode, print HLE histograms every N seconds; enable `PS4_HLE_HIST` to collect data. |
| `PS4_GPU_READBACK=1` | Wait for selected frames that drew and log centre/quarter framebuffer pixels. An **absolute directory path** additionally writes RGB P6 PPMs; create this directory first. |
| `PS4_GPU_READBACK_FROM=N` | Begin readback sampling at backend frame N; default 0. |
| `PS4_GPU_READBACK_EVERY=N` | Read back only backend frames divisible by N; default 1. |
| `PS4_GPU_READBACK_KEY=addr` | Read back that guest colour-target key instead of the screen when available (RGBA8/BGRA8 UNORM). Requires path-valued `PS4_GPU_READBACK`. |
| `PS4_GCN_DUMP_DIR=path` | Dump GCN code, disassembly and generated MSL on pipeline compilation; create directory first. |
| `PS4_GCN_TRACE=1` | Log shader/scalar resource details per translated draw; very verbose. |
| `PS4_GCN_SHADERS=0` | Disable translated GCN shaders and use the fixed mesh fallback. Unset or any other value enables translation. |
| `PS4_AUTO_PRESS=N` | Every N seconds pulse Cross and touch-pad for roughly 100 ms; window and headless modes. |
| `PS4_AUTO_MOVE=1` | Hold d-pad right for 2 seconds, left for 1 second, repeating. |
| `PS4_WATCH_ADDR=addr` | Interpreter/MemoryManager-path store ring with old/new values, caller RIP and thread context; direct JIT stores bypass it. |
| `PS4_WATCH_SIZE=N` | Byte length for `PS4_WATCH_ADDR`; default 8. |
| `PS4_WATCH_FILTER=addr` | Filter the watch ring printed at normal exit to this address; defaults to the watched address. |
| `PS4_DUMP_CODE=addr:len` | Dump loaded guest bytes to `dump_code_<addr>.bin` at normal exit; both fields hexadecimal, length strictly below 4 MiB. |
| `PS4_LOG_PM4=1` | Log PM4 packets, draw/shader state and GNM calls. High output volume can stall execution. |
| `PS4_LOG_HLE=1` | Log HLE calls, rate-limiting hot non-GNM calls. |
| `PS4_LOG_HLE=all` | Request unthrottled HLE logging. This build's outer enable check accepts only literal `1`, so `all` alone does not enable the outer call logger. |
| `PS4_LOG_HLE_TID=N` | Restrict HLE call logging to a guest thread ID; default all threads. |
| `PS4_LOG_SEM=1` | Log semaphore waits/posts. |
| `PS4_LOG_SEMA=1` | Additional semaphore event diagnostics (separate spelling from `PS4_LOG_SEM`). |
| `PS4_SEMA_CHAIN=1` | Enable the experimental semaphore chaining diagnostic path. |
| `PS4_NO_VSYNC=1` | Disable guest flip pacing. |
| `PS4_LOAD_PRX=0` | Force HLE module mode. Unset or other values load real PRX modules when present, falling back to HLE when absent. |
| `PS4_GUEST_<name>=value` | Supply an explicit guest environment variable through HLE `getenv` (for example `PS4_GUEST_GC_DONT_GC=1`). |
| `PS4_DUMP_EXPORTS=1` | Dump export discovery diagnostics. |
| `PS4_DUMP_TRAMPOLINES=1` | Log HLE trampoline generation. |
| `PS4_RIP_HIST=1` | Collect interpreter RIP occupancy histogram; JIT blocks bypass interpreter instruction sampling. |
| `PS4_TRACE_AT=addr` | Interpreter one-shot register/stack-chain dump at a guest instruction address. Use `--no-jit` to guarantee observing it. |
| `PS4_TRACE_ITER=N` | Repeat the `PS4_TRACE_AT` dump after every N further hits. |
| `PS4_TRACE_TID=N` | Trace a selected guest worker thread's instructions. |
| `PS4_TRACE_MAX=N` | Maximum worker instruction trace lines; default 20000. |
| `PS4_TRACE_X87=1` | Interpreter x87 diagnostics. |
| `PS4_JIT_NO_CALLS=1` | Disable JIT call translation for comparison with interpreter call handling. |

## Standalone tools

All three tools use only Python's standard library:

```sh
python3 scripts/tools/sample_threads.py /tmp/db-debug/sample_frame_300.txt --top 8
python3 scripts/tools/nidcrack.py sceFiosInitialize sceKernelUsleep
python3 scripts/tools/nidcrack.py --target c00299fdd7adfb2a --wordlist symbols.txt
python3 scripts/tools/nidcrack.py --target wAKZ-det+yo --wordlist symbols.txt
python3 scripts/tools/ppm_to_png.py /tmp/frame_120_gcn.ppm /tmp/frame.png
```

The sample parser reports per-thread busy versus waiting **sample counts**, not
CPU utilization. Wait classification uses known Darwin blocking leaf functions;
unresolved JIT addresses and unknown blocking functions count as busy. It preserves
parent self samples and aggregates top leaf functions. The NID cracker hashes each
candidate's UTF-8 bytes plus salt `518D64A635DED8C1E6B039B1C3E55230`, interprets the
first eight SHA-1 bytes little-endian, and prints hex and Orbis encoded NIDs. It
reads stdin if no names/wordlist are supplied, returning 1 if no candidate matches.
The PPM converter supports binary P6, comments, RGB maxval 255, and preserves pixel
bytes (including whitespace-valued first pixels); other PPM variants are rejected.

## Validation on the debug-tooling branch

Tested with the Investigate build based on `1b046f3`:

- `ctest`: 29/29 passed. The asset-scanner test needs the existing local Thomas
  Was Alone dump as well as the requested game dumps in the worktree's `games/`.
- Flappy Bird and Dragon Ball Xenoverse 2: captured backend frame 120, checked
  ordered draws, completed GPU pass timings, actual attachment metadata, PNGs
  and `.gputrace` output; exited at frame 121 with status 0.
- Both games: frame-triggered 8-second samples, parsed per-thread totals,
  NID dictionary matches for symbols seen in their logs, and conversion of
  their actual framebuffer PPMs. Separate runs exercised frame-5 dumps and
  JIT page-watch hits with TID/RIP/value output.
- `PS4_GPU_CAPTURE_FRAME=2 build-opt/test_gcn_shader`: captured the existing PM4
  compute test's `(4,1,1)` dispatch and completed compute-pass timing.
- A temporary native harness exercised two JIT stores separated by rearming,
  verified the store instruction's RIP after a preceding NOP, and checked
  forwarding of an unrelated SIGSEGV to a preinstalled handler. Python fixtures
  checked sample parent self-time, a known NID vector, whitespace pixel bytes,
  and rejection of a truncated PPM.

Local validation artifacts are under `/tmp/ps4-debug-validation/`; these are
machine-local and are not part of the commit. The captured early Dragon Ball
frame exercises the boot/title draw, not a completed gameplay scene.
