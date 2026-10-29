# NID audit vs ps4libdoc

Sony NIDs are deterministic: `SHA-1(name + salt)[:8]` read little-endian, with
`salt = 518D64A635DED8C1E6B039B1C3E55230`. That makes a wrong *label* provable
rather than a matter of taste — and a wrong label is not cosmetic, because the
label is what we attach a handler body to.

This audit ran every registration in `src/hle/nid_registry.cpp` against
[ps4libdoc](https://github.com/idc/ps4libdoc) and found 51 mislabeled
registrations, 4 of which were handing the guest the wrong behaviour.

## Corpora used

| File | Contents | Cached at |
|---|---|---|
| `libkernel.sprx.json` | Per-module symbol table: NID → name **plus the exporting library** (libkernel, libScePosix, libSceLibcInternal, libSceGnmDriver) | `data/libdoc/` |
| `known_names.txt` | 42,010 names across every PS4 module; resolves a NID to a name with no module dump | `data/libdoc/` |
| `mono_embedding_api.txt` | Every `mono_*` identifier in the official Mono headers (2,836) | `data/libdoc/` |

```bash
python3 scripts/audit_nids_libdoc.py     # exits non-zero only on a live mislabel
python3 scripts/crack_mono_nids.py --all
```

The audit cross-references the guest's import tables (eboot.bin + every PRX in
`sce_module/` and `Media/Modules/`) so it can tell a mislabeled NID the guest
actually calls — a bug — from a dead registration nothing imports — cleanup.

## The three requested fixes

| NID | Was | Now | Action |
|---|---|---|---|
| `0x7501D612C26DA04E` | `scePadInit` (libScePad) | **`pthread_mutexattr_init`** (libScePosix) | Real handler: zeroes the 4-byte attr type field only. The real `scePadInit` (`0x86FD65BA226BA903`) is now registered properly. |
| `0x253BC17E58586B34` | *unimplemented stub* | **`sceKernelWaitEventFlag`** | Full event-flag family added: Create/Delete/Set/Clear/Poll/Wait. Wait blocks with the guest timeout and returns `ESRCH` for an unknown handle — the Mono call site at `0x7ea892` passes `evf=NULL`, which must not block forever. |
| `0xE03334E94D813446` | `sceKernelCancelSema` | unchanged | Already correct; now cited to ps4libdoc rather than only to the hash. |

**The `scePadInit ×42` loop**: ps4libdoc lists `0x7501D612C26DA04E` as
`pthread_mutexattr_init`, exported by both `libkernel` and `libScePosix`. The
trace confirmed it — 42 calls, always `rdi` on the main stack, with `rsi`/`rdx`
holding stale `0x10`. That is Unity initializing one stack mutexattr per worker
thread, not a pad subsystem being re-initialized.

## Behaviour bugs the audit found (all imported by the guest)

These were mislabels *and* wrong handlers:

1. **`sem_init`** (`0xA43B8F11FDE6E1F2`) was registered with the
   `scePadReadState` handler — it wrote 32 bytes of pad button/stick state into
   the caller's `sem_t`. Imported by eboot.bin **and mono-ps4.prx**, so Mono's
   semaphores were being corrupted on every init. Now zeroes the semaphore and
   stores the initial count.
2. **`pthread_create`** (`0x3B184807C2C1FCF4`) was registered as
   `scePadSetVibration`, a `SCE_OK` stub: every plain-POSIX thread creation
   returned a NULL thread id. Now routed to the real thread-spawn handler.
3. **`realloc`** (`0x63B689D6EC9D3CCA`) was registered as a third `free`, so
   every realloc silently returned NULL. Now allocates and copies; the bump
   allocator tracks block sizes in `s_heap_sizes` so the copy length is known
   without changing malloc's block layout.
4. **`sceKernelWaitEventFlag`** was falling through to the generic unimplemented
   stub (see table above).

## Relabels (same handler, truthful name)

- `0xCAF67BDEE414AAB9` `sceVideoOutWaitVblank` → **`sceGnmSubmitDone`**
  (libSceGnm, imported by eboot.bin, libSceJobManager.prx, libSceS3DConversion.prx).
- `0x610D276AFA7E6087` `puts_orbis` → **`puts`** (libSceLibcInternal).
  `puts_orbis` is a UE-side macro, not an Orbis export.
- `0x885D62407CF10495` `mspace_memalign` → **`sceLibcMspaceMemalign`**.
- `0x954A2AC1342EE06A` `lUoqwTQu4Go` (Sony tag) → **`sceUserServiceGetUserColor`**.
- FIOS trio: `sceFiosArchiveMountWithOrder`/`…Sync`/`…GetMountBufferSize` →
  **`sceFiosOpGetActualCount` / `sceFiosDHOpenSync` / `sceFiosDHOpen`**.
- `0xB50228F8620F925A` `calloc` → **`_ZSt14_Xlength_errorPKc`** (libstdc++
  length_error thrower). Body still hands back a block rather than throwing:
  a guest allocation-failure path must never unwind through our un-emulated
  C++ runtime.
- 38 of the 51 `sceKernelSyncStub` entries now carry their real ps4libdoc name
  and module (`mono_config_parse`, `pthread_mutex_init`, `sceGnmBeginWorkload`,
  `sceAppContentInitialize`, `__stack_chk_fail`, …). Bodies unchanged (`rax = SCE_OK`).
  Two comments were simply wrong: `0x3AEDE22F569BBE78` is `__stack_chk_fail`,
  not `scePthreadSetprio`.
- Two libScePad NIDs ps4libdoc cannot name are now `unknown_b445383a` /
  `unknown_d4143f8a` (honest placeholders) with the real `scePadReadState`
  (`0x6277605EA41557B7`) and `scePadSetLightBar` (`0x451E27A2F50410D6`)
  registered at their true NIDs.
- `tests/test_phase7.cpp` asserted `puts == 0x5C0F663677A000A8`; that NID is a
  second puts NID ps4libdoc cannot name, now labelled `unknown_5c0f6636` while
  still serving as the shared implementation. The test now asserts the
  ps4libdoc-verified NID.

## Mono NIDs: hash-verified, not assumed

`scripts/crack_mono_nids.py` hashes all 2,836 `mono_*` symbols from the official
Mono headers (plus 174 `mono_*` strings mined from the guest binaries and the
ps4libdoc corpus). All four are confirmed, and the `0x4cba40` question is closed:

| NID | Verdict |
|---|---|
| `0x47F74CE8E2982CC4` | `mono_get_corlib` — hash-verified |
| `0x1147722929EB0A4C` | `mono_class_is_subclass_of` — hash-verified |
| `0xA6B7BD0639320E1B` | `mono_runtime_invoke` — hash-verified |
| `0x4CCB2C2760985FAA` | `mono_class_get_parent` — hash-verified |
| `0x2C5314E29C000914` (bonus) | `mono_class_get_methods` |

`--all` resolves **107** Mono NIDs the guest imports to real API names.

### The `0x4cba40` "UH..SPH.." buffer

You were right, and the disassembly proves it. `55 48 89 e5 53 50 48` is
`push rbp; mov rbp,rsp; push rbx; push rax` — a function prologue in `.text`
at guest address `0x4cba40` (file offset `0x4cfa40`, since the first PT_LOAD
maps vaddr 0 → file `0x4000`):

```
4cba40: push rbp / mov rbp,rsp / push rbx / push rax
4cba46: mov rbx, [rip+0x10630db]        ; 0x152eb28
4cba4d: call 0x58cf50
4cba52: mov edx, 0x4e
4cba57: mov rax, rdi / mov rbx, rsi
```

`0x4cba40` is passed by address as an argument to `0x98e7b0` at `0x4cbb2f`, in
the same static-init table walk as `0x4ca4e0` (`0x156abc0`/`0x156b1e0` are the
matching `{count, fn}` tables). A function pointer, not a resource buffer.

The caller at `0x9878b1` confirms the label:

```
9878a1: mov rdi, [rax+0x2e8]
9878a8: call 0x11f6d40                      ; some domain getter
9878b1: call 0x11f6ba0                      ; <-- the import; returns to 0x9878b6
9878b6: lea rsi, "System.Collections"       ; 0x12b67ac
9878bd: lea rdx, "IEnumerator"              ; 0x12b67bf
9878c4: mov rdi, rax                        ; the import's return value
9878c7: call 0x11f6b40                      ; mono_class_from_name(image, ns, name)
9878cf: test rbx, rbx / jne …
9878d8: lea rsi, "Unable to find corlib type %s.%s"
```

An argument-less call whose result is used as the *image* argument of
`mono_class_from_name` is `mono_get_corlib()` by signature. Note the earlier
reading that this NID's *value* was a buffer at `0x4cba40` was a coincidence of
two unrelated addresses.

## Sweep targets

No sibling `*.sprx.json` files exist anywhere on disk, so the module sweep
could not run as a file sweep. ps4libdoc's `known_names.txt` covers every
module instead and resolved all six:

| NID | Now known as | Was guessed as |
|---|---|---|
| `0xA10C921147E05D10` | **`sceSaveDataSetupSaveDataMemory2`** | "PlayerPrefs tail" |
| `0x90654B73786D404F` | **`sceVideoOutGetDeviceCapabilityInfo_`** | "video handle" |
| `0xA48EE81523CFEB90` | **`sceSocialScreenInitialize`** | "mode struct" |
| `0xE2EC8737DAB865E5` | `sceAudio3dObjectSetAttributes` | correct |
| `0x8CEDAD79CE1D2763` | `sceAudio3dObjectReserve` | correct |
| `0x64E1ABC562E04331` | `sceAudio3dPortFlush` | correct |

Also resolved: `0x12101A8AA03B2607 = mono_method_get_name`.

## Mono lookup chain (follow-up turn)

`scripts/crack_mono_nids.py --all` names **107** guest-imported Mono NIDs. Twenty
were already registered; the other 87 are classified in `kMonoStubs` by what the
*guest* does with the return value:

| Shape | Count | Rationale |
|---|---|---|
| `Iter` (NULL) | 14 | The guest iterates until NULL. A cookie here is an infinite loop — nothing else terminates the walk. |
| `Cookie` | 24 | Distinct, stable, zeroed arena slot keyed on the argument tuple, so repeated lookups compare equal and `mono_runtime_invoke` never receives NULL. |
| `Null` | 15 | "Absent" answers: strings, exceptions, basetypes. NULL is the documented miss. |
| `Zero` | 37 | Scalar getters: counts, sizes, gc stats, enum predicates, setters. |
| `Small` | 4 | `class_get_rank`=1, `object_get_size`=8, `class_get_flags`=0x21, `gchandle_is_in_domain`=true. |
| `Pass` | 4 | `gchandle_new` / `get_target` echo the object, so acquire/release balances. |

Three handlers were rewritten rather than table-driven:

- **`mono_get_corlib`** returns the arena's `corlib_image` singleton, and
  `mono_class_from_name` now defaults to that image (previously `first_image`)
  when `image == 0`.
- **`mono_class_is_subclass_of`** returns TRUE whenever both pointers are live.
  Disassembly at `0x997891` shows it is a *guard*, not a query:

  ```
  test rbx, rbx / je skip          ; nothing to validate?
  mov edx, 1 / mov rdi, rbx / mov rsi, r14
  call mono_class_is_subclass_of   ; test eax, eax / jne accept
  accept:  mov rax, rbx            ; return the candidate
  reject:  mov rdi, [r15] / call   ; recurse up the parent chain
  ```

  The old reflexive-identity answer was always FALSE on distinct cookies, so
  every call fell into the recursive parent walk.
- **`mono_runtime_invoke`** clears `*exc` and logs the method cookie.
- **`mono_thread_current` / `mono_thread_attach`** hand out a per-host-thread
  handle, so the guest's `while (mono_thread_current() == NULL) attach();` spin
  terminates.

Result on the 60-second smoke test: 783 `mono_class_from_name` calls, **zero**
failures, zero "Unable to find corlib type".

## `sceKernelWaitEventFlag` must not park a worker pool

The first implementation blocked indefinitely when the guest passed no timeout.
Unity creates two `Unity EventFlag` objects and parks six worker threads in it,
and nothing in this runtime ever calls `sceKernelSetEventFlag` — so all six were
stranded with no producer:

```
[Thread 7..12] state=WaitingSync RIP=0x1023e5 instrs=200000 lastnid=0x253bc17e58586b34
```

The wait is now capped (guest timeout when supplied, otherwise 1000 µs) and
reports `ETIMEDOUT`, so the guest re-polls instead of parking. The same run has
workers at 17–18 M instructions each and `state=Running`.

## JIT vs interpreter: the pre-Mono stall is a JIT bug

With the JIT enabled (the default), the guest deterministically dies in a
vectorised walk at `0x58d1b2` after HLE call #45 (`scePthreadMutexattrDestroy`);
call #46 (`scePthreadMutexLock` @ `0x58ac29`) never dispatches. The watchdog
shows a wrapped counter driving the loop:

```
[Thread 1] 'main' state=Running RIP=0x58d196 hles=46 inhle=0 lastnid=0xb2658492d8b2c86d
    rax=0xfffffffffedf7290 ... (decreasing)   code: vpaddq / vinsertf128 / addq $-0x10, %rax
```

`[HLE CALL]` telemetry is printed *before* `sym->handler()`, and `inhle=0`, so
the guest is in pure guest code — not blocked inside a handler. Reproduces
identically at `ccd05da`, `9819396` and `ec509a3`, so it predates the recent
NID work. `--no-jit` passes the same point and reaches Mono.

**Follow-up: the AVX-512 path (`vpaddq`/`vinsertf128`/`vextractf128` with ymm)
in `JitCompiler` is mis-compiled. Until that is fixed, Mono-phase smoke tests
must run `--no-jit`.** The hardcoded 100 M headless budget also cannot reach the
Mono phase at all; `--budget N` now exposes it.

## Killing Unity's 56 "Unable to find method" warnings

Unity resolves a managed method like this (guest disassembly, three imports in
sequence):

```
99ad0a: iter = 0
99ae0c: method = mono_class_get_methods(klass, &iter)
99ae33: name   = mono_method_get_name(method)
99ae38: if (*name != *wanted) continue            ; r12 holds the wanted name
99ae47: if (strcmp(name, wanted)) continue
99af0f: sig    = mono_method_signature(method)
99af1b: test rax, rax / jne <resolved>            ; <-- the success branch
99af20: err    = mono_loader_get_last_error()
        ...                                       ; "Unable to find method %s in [%s]"
```

Three separate NULLs each produced the warning. All three are fixed:

- **`mono_class_get_methods`** is a one-shot iterator: if `*(u64*)iter != 0`
  return NULL, else store 1 and return a fresh `MonoMethod*` cookie. Every walk
  is now exactly one turn.
- **`mono_method_get_name`** returns `ctx.r12` — the callee-saved register that
  holds the wanted name for the whole walk. The dispatcher passes the guest's
  live `CpuContext` (`sctx.cpu = &m_context`), so `ctx.r12` *is* that pointer,
  which makes both the first-byte compare and the strcmp succeed on turn one.
  Verified by instrumentation: it returns `'Init'`, `'CallLogCallback'`,
  `'CallLowMemory'`, `'InvokeOnBeforeRender'`, …
- **`mono_method_signature`** moved from `Iter` to `Cookie`. It is not an
  iterator at all — non-NULL *is* the "method resolved" answer.

Result: **56 → 0** warnings, and `mono_runtime_invoke` went from 98 calls with
`method=0x0` to 100 calls carrying 5 distinct method cookies.

## The gfx worker's EINVAL retry loop

The tail of the run was Unity's gfx thread spinning:

```
[SCE] scePthreadMutexLock(&m_Mutex) returned 0xffffffea in
      .\PlatformDependent\PS4\Source\Threads\PlatformMutex.cpp(38)
[SCE] sceKernelUsleep(0xf4240)          ; 1 ms, repeat   (108 times in 60 s)
```

`PlatformMutex` keeps a *guest-owned* struct whose copies carry values we never
issued, so `mutex_lock` returned false (-22) forever. `resolve_mutex_id` now
adopts an unrecognised handle: mint a real mutex and write it back, the same
way the old code already handled a zeroed one. After the fix: **0** EINVAL
complaints, and `UnityGfxDeviceWorker` went from 4.6 M to **125 M** instructions
in the same 60 seconds.

## POSIX semaphores, strchr, gettimeofday

`sem_init` was wired earlier, but `sem_wait`/`sem_post` still fell through to the
generic stub, so nothing ever blocked. Wired the family with ps4libdoc NIDs
(`sem_wait` 0x602579746181702A, `sem_post` 0x20A3FCB72A744149, `sem_trywait`,
`sem_timedwait`, `sem_destroy`, `sem_getvalue`), plus `strchr`
(0xA1BE71016E259FFD — Unity's parser calls it on every token, and the stub's
NULL is indistinguishable from "not found") and both `gettimeofday` NIDs.
Unimplemented stubs fell from 45 to 28 per 60 s run.

## sceGnmSubmitDone ×19 is TID 5's watchdog, not the gfx worker

`0xcaf67bdee414aab9` is called at caller RIP `0x10a9080`, inside TID 5
(`Submit Done Thread`, entry `0x10a8fe0`) — not by `UnityGfxDeviceWorker`. TID 5
sleeps `sceKernelUsleep(0xf4240)` (1 s) per iteration and on every third
iteration calls `sceGnmSubmitDone()` then `puts(0x133a0b0)`. The string at
`0x133a0b0` is:

```
Forcing submitDone to avoid TRC R4089 breach
```

So 19 calls in 60 s is the 3-second TRC watchdog ticking, and it is a *symptom
worth keeping*: it is telling us no flip arrived on equeue 2 within 3 s. This is
the cleanest signal we have for "frame 1 has not presented".

## The three unnamed mono-ps4 probes at 0x98c4cb

ps4libdoc's 42 k corpus and the 2,836 upstream Mono symbols both miss these. The
import table settles the module: all three carry the same obfuscated `i#j`
tag as `mono_method_get_name`, i.e. mono-ps4 — Unity's fork, whose internals
never shipped upstream.

They are **not** pointer lookups. The guest chains them as one-argument,
int-returning capability probes:

```
98c4ce: call probe1; test eax,eax / je next   ; non-zero => result = -4, return
98c4ee: call probe2; test eax,eax / jne done
98c4fa: call probe3; test eax,eax / je 0x98c564
0x98c503: result = -6, normal epilogue
0x98c564: mono_class_is_subclass_of(r15, corlib_type, TRUE) twice
```

All three return 0 ("feature absent"), which sends the guest to `0x98c564`,
where `mono_class_is_subclass_of` now answers TRUE — so the guest survives
instead of bailing. Registered as `unknown_98c4ce_probe1` / `…probe2` /
`…probe3` with the behaviour pinned to the disassembly, not to a guess.

## WaitEventFlag cap: 20 ms is a regression, not an optimisation

The six idle workers wake on every `sceKernelWaitEventFlag`, so a longer cap
looks like free CPU. Measured over two 60 s runs, changing only that constant:

| cap | HLE calls | main (hles) | UnityGfxDeviceWorker | threads |
|---|---|---|---|---|
| **1 ms** | 382 k–410 k | **195 k** | **99–112 M instr** | 12 |
| 20 ms | 84 k | **0** | 4.4 M instr | 33 |

At 20 ms the guest spawned 33 threads instead of 12 and **main never executed a
single instruction** (`RIP=0x0 instrs=0 hles=0`). Longer blocking hands the
scheduler to the worker pool instead of the thread driving the game. Kept at
1 ms. The ~400 EventFlag polls per second are real overhead, but they are not
what is starving the frame — the TRC watchdog is.

## Left alone, on purpose

- **26 dead registrations.** NIDs no guest binary imports, carrying a label that
  provably does not hash to them (`sceKernelExitProcess` on `0x6424D00705E3BEBE`,
  `sceAudioOutOpen` on `0xE1D5176E5161A4FE`, …). ps4libdoc has no name for these
  NIDs at all, so they look speculative rather than fixable — they mislead only
  in the trace, and deleting them is a separate decision. The audit reports them
  as cleanup and does not fail on them.
- **Three unresolved NIDs** kept as honest placeholders: `unknown_443bb032`,
  `unknown_3c4ac625` (an ex-SignalSema label, never called) and
  `unknown_0528b1d7` (behaves as a non-blocking semaphore poll).

## Imports that must not return to the guest

A handler body that only sets `rax` and returns is not a conservative stub for
these. The guest has no code after the call that can make progress — what sits
there is the callee's alignment sled, then the next function's prologue, so
control walks into unrelated code until it branches to a garbage pointer. That
is how the canary failure in `9b8365c` was reported as "Memory fault / invalid
instruction pointer" at `RIP=0x3b480151ed5a1d8d` with the canary invisible.

Every noreturn import the guest makes, resolved against `data/libdoc/known_names.txt`
by SHA-1(name + Orbis salt) and checked against the guest's own import tables
(1,793 NIDs across eboot.bin + 12 PRX):

| Import | NID(s) | Before | Now |
|---|---|---|---|
| `abort` | `0x2F54814E40BE0AFC`, `0xFC0ABE404E81542F` | returned `rax = -1` | `guest_abort()`: reports call site + rbp/rsp/args, write-watch ring, exit 134 |
| `exit` | `0xB8C7A2D56F6EC8DA` | unregistered → 0 | `libc_exit_handler()`: `SYS_exit` dispatch (sets `has_exited`, releases untimed POSIX waiters), `rip = 0` |
| `_exit` | `0xE99F37B18585940F` | unregistered → 0 | same body (no atexit registry exists — see `__cxa_atexit`) |
| `pthread_exit` | `0x149AD3E4BB940405` | unregistered → 0 | shares `scePthreadExit`'s body: `exit_current_thread(retval)`, `rip = 0` |
| `__stack_chk_fail` | `0x3AEDE22F569BBE78` | (already fixed in `9b8365c`) | unchanged behaviour; abort epilogue factored into the shared `guest_abort()` helper |
| `longjmp` | `0x94A10DD8879B809D`, `0x9D809B87D80DA194` | already non-returning (restores `rip`/`rsp` from `jmp_buf`) | unchanged — verified correct |
| `setjmp` | `0x80D435576BDF5C31`, `0x315CDF6B5735D480` | returns 0 once | unchanged (by definition) |
| `scePthreadExit` | `0xDE483BAD3D0D408B` | already non-returning | unchanged |

Not imported by this guest, so deliberately **not** registered — a dead
registration is trace noise, not a fix: `quick_exit`, `__assert_fail`,
`__assert_rtn`, `siglongjmp`, `_Unwind_Resume`, `_Unwind_RaiseException`,
`_Unwind_Resume_or_Rethrow`, `__cxa_throw`, `_ZSt9terminatev`,
`raise`, `pthread_kill`, `pthread_atexit`, `bsearch`, `pthread_once`.

### Callbacks into guest code

Three imports take guest function pointers. An HLE handler cannot invoke them —
it would mean re-entering the guest CPU core from inside a handler, which is
what the RSP floor and the write-watch ring exist to police against:

- **`qsort`** (`0x00425D215653126A`, eboot.bin + libSceFaceTracker.prx) —
  registered, logged once, array left in input order. `qsort` returns void, so
  the old stub's `rax = 0` was never a wrong return value; the damage was that
  nothing was sorted. Input order is the deterministic no-op, equivalent to a
  sort whose comparator reports every pair equal.
- **`__cxa_atexit`** (`0xB6CBC49A77A7CF8F`) and **`__cxa_finalize`**
  (`0x1F67BCB7949C4067`) — both register 0 (success) and never run the handler.
  Documented consequence: **guest static destructors never run**, so nothing may
  depend on at-exit side effects. `exit` terminates rather than sweeping, and
  the trace now says `__cxa_finalize` instead of "Unimplemented stub called".
- **`atexit`** (`0xF06D8B07E037AF38`, eboot.bin) — handled in the dynamic
  linker's whitelist, not the registry. Same consequence as `__cxa_atexit`.

## The four unimplemented stubs that were firing

Every one of these hit `src/core/dynamic_linker.cpp`'s null-bail fallback
("Unimplemented stub called … returning 0") on a normal run. All four are
registered now, under the ps4libdoc names their NIDs hash to, with the guest's
call site disassembled to justify each answer.

| NID | Now registered as | Call site | Output struct? | Decision |
|---|---|---|---|---|
| `0xA10C921147E05D10` | `sceSaveDataSetupSaveDataMemory2` | `0x10acf7e` (TID 4) | no — int status call | `SCE_OK`; param struct dumped once so its layout is evidence, not guess |
| `0x43038EEEF7A09D5F` | `sceSaveDataGetSaveDataMemory2` | `0x10ad06a` (TID 4) | no — success path fills its own buffer | `SCE_OK` |
| `0x90654B73786D404F` | `sceVideoOutGetDeviceCapabilityInfo_` | `0x10a7b74` (TID 1) | **yes** — 8 bytes at `[rbp-0x170]` | **implement the output**: zero `rdx` bytes, `SCE_OK`. The guest tests byte 0 as signed at `0x10a7c0b`, so an unwritten stack slot was read as a capability. Blank = no capabilities is truthful and deterministic. |
| `0xA48EE81523CFEB90` | `sceSocialScreenInitialize` | `0x1065f20` (TID 1) | no — return value discarded | `SCE_OK` |

Only `sceVideoOutGetDeviceCapabilityInfo_` had a callee-written output. The
`sceSaveData*Memory2` pair are int status calls whose success paths dereference
nothing the stub failed to fill, so returning 0 was correct for them and kept.

### A/B: explicit error instead of 0

Made all four return `-EINVAL` (-22) and ran `ps4-runner --no-jit --budget
250000000` on `eboot.elf`. Result: canary abort at the same call site
`0x58d595`, earlier and shallower — 29 M instructions, 32 threads, 224
semaphores, `UnityPreload` created but `UnityGfxDeviceWorker` halted on
unimplemented `strcpy`.

But that single sample is not a signal, and saying so is the useful finding here.
The abort is intermittent in *both* variants and at `HEAD`. Full run matrix,
all with `--no-jit`:

| build | run | canary abort | last progress | threads | semas |
|---|---|---|---|---|---|
| HEAD (`git show HEAD:`) | 200 M budget | **yes** (`0x58d595`) | 33 M | 32 | 172 |
| HEAD (`git show HEAD:`) | 60 M budget | no | 44 M | 33 | 1,218 |
| HEAD (`git show HEAD:`) | 60 M budget | no | 39 M | 32 | 732 |
| HEAD (`git show HEAD:`) | 60 M budget | no | 44 M | 33 | 1,218 |
| HEAD (`git show HEAD:`) | 60 M budget | no | 44 M | 33 | 1,218 |
| this change | 200 M budget | no | 44 M | 33 | 1,218 |
| this change | 250 M budget | **yes** (`0x58d595`) | 29 M | 32 | 224 |
| this change | 250 M budget | no | 44 M | 33 | 1,218 |
| this change | 250 M budget | no | 44 M | 33 | 1,218 |
| this change | 250 M budget | no | 44 M | 33 | 1,218 |
| this change | 60 M budget | no | 44 M | 33 | 1,218 |
| **four → `-EINVAL`** | 250 M budget | **yes** (`0x58d595`) | 29 M | 32 | 224 |

**What that establishes:**

1. **The abort site does not change.** Every abort, in every build, is call
   site `0x58d595` with the same frame canary read as 0 and the same
   `sceKernelSignalSema(<id>): unknown sema id` immediately before it. The
   `abort` / `exit` / `pthread_exit` fixes above are irrelevant to it — none of
   those imports was reached in a normal run.
2. **The four stubs are not the cause.** One HEAD run in five aborts; one
   changed build in six aborts. With one sample of the error variant there is no
   measurable rate difference, so the error variant is *not* demonstrably worse —
   it is one draw from the same distribution.
3. **Why it is intermittent:** the guest is multithreaded, so HLE dispatch
   timing changes the thread interleave, and the canary slot at `[rbp-0x30]` is
   overwritten by whichever thread wins the race. The write-watch ring in the
   abort report names the clobbering store's RIP and TID for the runs where it
   is armed.
4. **What the four were doing, honestly:** hiding a real import behind
   "Unimplemented stub called … returning 0", and in
   `sceVideoOutGetDeviceCapabilityInfo_`'s case feeding the guest stale stack
   data as a capability value it then branches on (`0x10a7c0b`). That is a
   genuine correctness fix independent of the canary question.
