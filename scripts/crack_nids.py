#!/usr/bin/env python3
"""Local NID cracker: hash every string in the game binaries against target NIDs.

Sony NID = first 8 bytes of SHA1(symbol_name + salt), little-endian uint64.
Sources: eboot.bin, mono-ps4.prx, MonoAssembliesPS4.prx (+ builtin candidate
names for mono/sce APIs not present as strings).

Usage:
    python3 scripts/crack_nids.py

To chase new unknowns, add their NIDs to TARGETS below (and extend EXTRA with
guessed API names from the relevant subsystem).
"""
import hashlib
import re
import sys
from pathlib import Path

KEY = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")

# Sanity anchors (must hold or the formula is wrong):
#   mono_jit_init_version -> 0x9b795b57a3b5e063
#   sceSysmoduleLoadModule -> 0x83c70cdfd11467aa

TARGETS = {
    0x12101A8AA03B2607,  # mono_method_get_name (resolved)
    0x253BC17E58586B34,
    0x443BB032833D9F45,
    0x3C4AC6259839E5EB,
    0xA10C921147E05D10,
    0x90654B73786D404F,
    0xA48EE81523CFEB90,
}

EXTRA = (
    "mono_class_get mono_class_init mono_class_is_enum mono_class_is_valuetype "
    "mono_class_get_rank mono_class_get_element_class mono_class_get_nesting "
    "mono_class_get_interfaces mono_class_get_fields mono_class_get_properties "
    "mono_class_get_events mono_class_get_method_from_name mono_class_get_field_from_name "
    "mono_class_get_property_from_name mono_field_get_name mono_field_get_type "
    "mono_method_get_name mono_method_get_class mono_method_get_header "
    "mono_method_signature mono_signature_get_params mono_signature_get_return_type "
    "mono_type_get_name mono_type_get_class mono_image_get_name "
    "mono_image_get_table_info mono_image_get_table_rows mono_image_get_assembly "
    "mono_assembly_get_object mono_domain_get_by_id mono_get_root_domain "
    "mono_jit_init mono_runtime_invoke mono_object_new mono_array_new "
    "mono_string_new mono_gchandle_new mono_thread_attach mono_thread_current "
    "mono_object_get_class mono_class_vtable mono_method_desc_new "
    "mono_method_desc_search_in_class mono_method_desc_search_in_image "
    "mono_property_get_name mono_property_get_get_method mono_property_get_set_method "
    "mono_event_get_name mono_custom_attrs_from_class mono_assembly_load_from "
    "mono_domain_set mono_config_parse mono_set_dirs mono_icall_add "
    "mono_marshal_get_native_wrapper mono_method_get_flags mono_class_get_flags "
    "mono_field_get_flags mono_property_get_flags mono_image_get_filename "
    "mono_assembly_get_main mono_get_exception_argument_null "
    "mono_get_exception_invalid_operation "
    "sceSaveDataGetMountInfo sceSaveDataDirNameSearchEx sceSaveDataMount5 "
    "sceVideoOutGetVblankStatus sceVideoOutGetFlipRate sceVideoOutAdjustColor "
    "sceVideoOutColorSettings sceVideoOutGetModeStatus sceKernelGetModuleInfo "
    "sceKernelGetModuleList sceKernelCancelSemaWait sceKernelDeleteSema "
    "sceKernelSetSemaAttr sceKernelCreateSemaInternal sceSysmoduleLoadModuleByName "
    "scePthreadMutexattrGettype"
).split()

ROOT = (
    Path(__file__).resolve().parent.parent
    / "games"
    / "flappy_bird"
    / "ED1633-NPXX51362_00-TER1520000000000"
)


def nid(s: bytes) -> int:
    return int.from_bytes(hashlib.sha1(s + KEY).digest()[:8], "little")


def main() -> int:
    assert nid(b"mono_jit_init_version") == 0x9B795B57A3B5E063
    assert nid(b"sceSysmoduleLoadModule") == 0x83C70CDFD11467AA
    print("formula OK")
    cands = set()
    for rel in (
        "eboot.bin",
        "Media/Modules/mono-ps4.prx",
        "Media/Modules/MonoAssembliesPS4.prx",
    ):
        path = ROOT / rel
        if not path.exists():
            print(f"missing: {rel}")
            continue
        data = path.read_bytes()
        print(f"{rel}: {len(data)} bytes")
        for m in re.finditer(rb"[A-Za-z0-9_]{3,}", data):
            cands.add(m.group(0))
    print("unique strings:", len(cands))
    cands.update(e.encode() for e in EXTRA)
    hits = {}
    for c in cands:
        h = nid(c)
        if h in TARGETS:
            hits.setdefault(h, []).append(c.decode("ascii", "replace"))
    for t in sorted(TARGETS):
        print(f"0x{t:016X}:", hits.get(t, ["UNRESOLVED"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
