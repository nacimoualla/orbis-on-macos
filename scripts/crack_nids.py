#!/usr/bin/env python3
"""Targeted NID cracker: Mono/IL2CPP combinatorics + POSIX/SCE candidates +
C identifiers mined from guest binaries, checked big- and little-endian.

The log prints NIDs as little-endian uint64 (int.from_bytes(digest[:8]).
'LE' hits are the live ones; BE is a safety net in case a trace hex was
recorded big-endian.

Usage:
    python3 scripts/crack_nids.py
    # chase new unknowns: add their trace hex to TARGETS (either endianness)
"""
import glob
import hashlib
import itertools
import os
import re
import sys
from pathlib import Path

SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")

# trace-hex -> call-site notes
TARGETS = {
    "47f74ce8e2982cc4": "0x9878b6 (arena-cookie, 0x4cba40, 0x1adb)",
    "e2ec8737dab865e5": "0x117645a FMOD (-1, 0, 2)",
    "1147722929eb0a4c": "0x997891 (cookie, cookie, 1)",
    "a6b7bd0639320e1b": "0x98a780 (0, 0, stackbuf)",
    "8cedad79ce1d2763": "0x11763b0 FMOD (-1, path?, 0x38)",
    "253bc17e58586b34": "0x7ea892 (0, 1, 0x22)",
    "e03334e94d813446": "0x103fce1 (0x3f, -1, 0)",
    "64e1abc562e04331": "0x1177452 FMOD (-1, 0, 1)",
    "2c5314e29c000914": "0x99ad0a (class_cookie, stackbuf, -1)",
    "4ccb2c2760985faa": "0x99aeef (class_cookie, stackbuf, -1)",
    "443bb032833d9f45": "ex-SignalSema label, never called",
    "3c4ac6259839e5eb": "ex-SignalSema label, never called",
    "a10c921147e05d10": "0x10acf83 PlayerPrefs tail",
    "90654b73786d404f": "0x10a7b79 video handle",
    "a48ee81523cfeb90": "0x1065f25 mode struct",
}

HITS = []


def check_symbol(sym: str):
    h = hashlib.sha1(sym.encode("utf-8") + SALT).digest()[:8]
    be_hex = h.hex()
    le_hex = h[::-1].hex()
    if be_hex in TARGETS:
        HITS.append((be_hex, "BE", sym))
        print(f"[MATCH BE] {be_hex} -> {sym}  ({TARGETS[be_hex]})")
    if le_hex in TARGETS:
        HITS.append((le_hex, "LE", sym))
        print(f"[MATCH LE] {le_hex} -> {sym}  ({TARGETS[le_hex]})")


# 1. Scripting Bridge (Mono / IL2CPP) combinatorics
prefixes = ["mono_", "il2cpp_", "mono_unity_", "unity_mono_"]
nouns = [
    "class", "image", "domain", "assembly", "assembly_name", "method",
    "property", "field", "event", "type", "object", "string", "array",
    "thread", "vtable", "custom_attrs", "metadata", "signature", "gc",
    "runtime", "security", "profiler", "debug", "dllmap", "config",
]
verbs = [
    "get", "set", "from", "is", "has", "init", "open", "close", "load",
    "lookup", "find", "iterate", "enum", "alloc", "new", "free", "create",
    "destroy", "invoke", "compile", "resolve", "verify", "check", "register",
    "attach", "detach", "walk", "parse", "read", "write", "get_name",
    "get_namespace", "get_parent", "get_fields", "get_methods", "get_properties",
    "get_events", "get_interfaces", "get_nested_types", "get_type",
    "get_element_class", "get_rank", "get_flags", "get_image", "get_assembly",
    "get_userdata", "set_userdata", "is_subclass_of", "is_assignable_from",
    "is_valuetype", "is_enum", "is_inflated", "is_generic", "init_internal",
    "from_mono_type", "from_name", "from_name_case", "from_typeref",
    "get_method_from_name", "get_method_from_name_flags", "get_property_from_name",
    "get_field_from_name", "image_get_table_info", "image_get_table_rows",
    "metadata_decode_row", "metadata_string_heap", "metadata_blob_heap",
    "metadata_guid_heap", "metadata_user_string_heap", "get_resource",
    "get_manifest_resource", "get_data", "get_size", "get_table_info",
]

for p, n, v in itertools.product(prefixes, nouns, verbs):
    check_symbol(f"{p}{n}_{v}")
    check_symbol(f"{p}{v}_{n}")
    check_symbol(f"{p}{v}")

# 2. POSIX / Libc / Syscall / Pthread / FIOS / Kernel / Audio candidates
posix_and_sce = [
    "lseek", "_lseek", "__sys_lseek", "lseek64", "fseek", "fseeko", "ftell", "ftello",
    "read", "_read", "__sys_read", "pread", "pread64", "fread",
    "open", "_open", "__sys_open", "close", "_close", "fcntl", "_fcntl", "__sys_fcntl",
    "ioctl", "_ioctl", "__sys_ioctl", "fstat", "_fstat", "__sys_fstat", "stat", "lstat",
    "ftruncate", "fsync", "fdatasync", "mmap", "munmap", "mprotect", "madvise", "posix_madvise",
    "sceKernelLseek", "sceKernelRead", "sceKernelPread", "sceKernelOpen", "sceKernelClose",
    "sceKernelFstat", "sceKernelStat", "sceKernelIoctl", "sceKernelFcntl", "sceKernelMmap",
    "cpuset_setaffinity", "cpuset_getaffinity", "__sys_cpuset_setaffinity", "__sys_cpuset_getaffinity",
    "pthread_setaffinity_np", "pthread_getaffinity_np", "scePthreadSetaffinity", "scePthreadGetaffinity",
    "sceKernelSetThreadDtors", "sceKernelGetCpumode", "scePthreadAttrSetaffinity", "scePthreadAttrGetaffinity",
    "sched_setaffinity", "sched_getaffinity", "pthread_mutexattr_init", "pthread_mutexattr_settype",
    "pthread_mutexattr_setpshared", "pthread_condattr_init", "sem_init", "sem_open",
    "sceAudioOutOpen", "sceAudioOutOutput", "sceAudioOutOutputs", "sceAudioOutSetVolume",
    "sceAudioOutGetPortState", "sceAudioOutInit", "sceAudioOutClose", "sceAudioOutGetSystemState",
    "sceAudioOutSetMixLevelPadSpk", "sceAudioOutMasteringInit", "sceAudioOutMasteringSetParam",
    "sceAudioInOpen", "sceAudioInInput",
]

for s in posix_and_sce:
    check_symbol(s)

# 3. C identifiers mined from guest binaries (eboot, mono PRXs, bundled modules)
ROOT = (
    Path(__file__).resolve().parent.parent
    / "games"
    / "flappy_bird"
    / "ED1633-NPXX51362_00-TER1520000000000"
)
ident_re = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{2,120}")
seen = set()
rel_paths = ["eboot.bin", "Media/Modules/mono-ps4.prx", "Media/Modules/MonoAssembliesPS4.prx"]
try:
    rel_paths += [f"sce_module/{f}" for f in os.listdir(ROOT / "sce_module")]
except OSError:
    pass
for rel in rel_paths:
    try:
        with open(ROOT / rel, "rb") as f:
            data = f.read()
    except (FileNotFoundError, IsADirectoryError, OSError):
        continue
    n = 0
    for m in ident_re.finditer(data):
        s = m.group(0).decode("ascii", errors="ignore")
        if s not in seen:
            seen.add(s)
            n += 1
            check_symbol(s)
            check_symbol("_" + s)
            check_symbol("__sys_" + s)
    print(f"mined {rel}: {n} new strings")

print(f"\nDONE: {len(HITS)} total hits")
