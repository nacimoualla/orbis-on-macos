// libSceSaveData and libSceNpTrophy HLE, driven through the registered
// handlers with guest structs in guest memory: a save is created, written
// through sceKernelOpen/Write, unmounted, mounted again and read back; then
// search, delete, save-data memory, and a trophy context/handle/unlock round
// trip.

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/np_trophy.hpp"
#include "ps4/save_data.hpp"
#include "ps4/vfs.hpp"

using namespace ps4;

// Always evaluated, unlike assert: several checks have side effects.
#define CHECK(x)                                                                   \
    do {                                                                           \
        if (!(x)) {                                                                \
            std::cerr << "CHECK failed at line " << __LINE__ << ": " #x "\n";       \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

namespace {
constexpr i32 kUser = 0x10000001;
constexpr GuestAddr kArgs = 0x50000, kDirName = 0x51000, kResult = 0x52000, kBuf = 0x53000, kOut = 0x54000,
                    kOut2 = 0x55000, kPath = 0x56000;

struct Harness {
    std::filesystem::path root;
    MemoryManager mem;
    VirtualFileSystem vfs;
    NidRegistry registry;

    explicit Harness(const std::filesystem::path& r) : root(r), vfs(r) {
        CHECK(mem.initialize());
        registry.set_vfs(&vfs);
        registry.register_default_libraries();
    }
    ~Harness() { registry.set_vfs(nullptr); }

    u32 call(const char* name, std::initializer_list<u64> args) {
        const HleSymbol* sym = registry.find_by_name(name);
        if (!sym || !sym->handler) {
            std::cerr << "missing handler " << name << "\n";
            std::abort();
        }
        CpuContext ctx{};
        u64* regs[] = {&ctx.rdi, &ctx.rsi, &ctx.rdx, &ctx.rcx, &ctx.r8, &ctx.r9};
        usize i = 0;
        for (u64 a : args) *regs[i++] = a;
        sym->handler(ctx, mem);
        return static_cast<u32>(ctx.rax);
    }

    template <typename T>
    void put(GuestAddr a, const T& v) { CHECK(mem.write_bytes(a, &v, sizeof(T))); }
    template <typename T>
    T get(GuestAddr a) {
        T v{};
        CHECK(mem.read_bytes(a, &v, sizeof(T)));
        return v;
    }
    void put_str(GuestAddr a, const std::string& s) { CHECK(mem.write_bytes(a, s.c_str(), s.size() + 1)); }

    // sceSaveDataMount2 on `dir`; returns the status and leaves the result at kResult.
    u32 mount2(const char* dir, u32 mode) {
        SceSaveDataDirName dn{};
        std::strncpy(dn.data, dir, sizeof(dn.data) - 1);
        put(kDirName, dn);
        SceSaveDataMount2 m{};
        m.user_id = kUser;
        m.dir_name = kDirName;
        m.blocks = 96;
        m.mount_mode = mode;
        put(kArgs, m);
        put(kResult, SceSaveDataMountResult{});
        return call("sceSaveDataMount2", {kArgs, kResult});
    }
    std::string mount_point() { return get<SceSaveDataMountResult>(kResult).mount_point.data; }
    u32 umount() { return call("sceSaveDataUmount", {kResult}); }
};

void expect(u32 got, u32 want, const char* what) {
    if (got != want) {
        std::cerr << what << ": got 0x" << std::hex << got << ", want 0x" << want << std::dec << "\n";
        std::abort();
    }
}

// A minimal param.sfo with one TITLE_ID string entry.
void write_param_sfo(const std::filesystem::path& path, const std::string& title_id) {
    const std::string key = "TITLE_ID";
    std::vector<char> sfo(0x14 + 16);
    auto put32 = [&](usize off, u32 v) { std::memcpy(sfo.data() + off, &v, 4); };
    auto put16 = [&](usize off, u16 v) { std::memcpy(sfo.data() + off, &v, 2); };
    const u32 key_table = static_cast<u32>(sfo.size());
    const u32 data_table = key_table + 16;
    put32(0x00, 0x46535000);
    put32(0x04, 0x101);
    put32(0x08, key_table);
    put32(0x0C, data_table);
    put32(0x10, 1);
    put16(0x14, 0);                                          // key offset
    put16(0x16, 0x0204);                                     // utf8 string
    put32(0x18, static_cast<u32>(title_id.size() + 1));     // len
    put32(0x1C, 12);                                         // max len
    put32(0x20, 0);                                          // data offset
    sfo.resize(data_table + 12, 0);
    std::memcpy(sfo.data() + key_table, key.c_str(), key.size());
    std::memcpy(sfo.data() + data_table, title_id.c_str(), title_id.size());
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary).write(sfo.data(), static_cast<std::streamsize>(sfo.size()));
}

void test_save_data(Harness& h) {
    std::cout << "[SaveData] mount -> write -> unmount -> mount -> read back\n";
    expect(h.call("sceSaveDataInitialize3", {0}), SCE_OK, "Initialize3");

    // Missing save without a create flag.
    expect(h.mount2("SAVE0001", SCE_SAVE_DATA_MOUNT_MODE_RDONLY), SCE_SAVE_DATA_ERROR_NOT_FOUND, "mount missing");

    // CREATE2 creates it and says so.
    expect(h.mount2("SAVE0001", SCE_SAVE_DATA_MOUNT_MODE_RDWR | SCE_SAVE_DATA_MOUNT_MODE_CREATE2), SCE_OK, "mount create2");
    CHECK(h.mount_point() == "/savedata0");
    CHECK(h.get<SceSaveDataMountResult>(kResult).mount_status == SCE_SAVE_DATA_MOUNT_STATUS_CREATED);
    const auto host_dir = h.root / "savedata" / std::to_string(kUser) / "TEST00001" / "SAVE0001";
    CHECK(std::filesystem::is_directory(host_dir));

    // The guest writes its save through libkernel, into a subdirectory.
    const std::string payload = "level=7;coins=1234";
    h.put_str(kPath, "/savedata0/slots");
    expect(h.call("sceKernelMkdir", {kPath, 0777}), SCE_OK, "mkdir");
    h.put_str(kPath, "/savedata0/slots/slot0.bin");
    const u32 fd = h.call("sceKernelOpen", {kPath, ORBIS_O_WRONLY | ORBIS_O_CREAT | ORBIS_O_TRUNC, 0666});
    CHECK(static_cast<i32>(fd) >= 3);
    h.put_str(kBuf, payload);
    expect(h.call("sceKernelWrite", {fd, kBuf, payload.size()}), static_cast<u32>(payload.size()), "write");
    expect(h.call("sceKernelClose", {fd}), SCE_OK, "close");

    // Param and icon.
    h.put_str(kBuf, "Chapter 3");
    expect(h.call("sceSaveDataSetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_TITLE, kBuf, 10}), SCE_OK, "SetParam title");
    h.put<u32>(kBuf, 42);
    expect(h.call("sceSaveDataSetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_USER_PARAM, kBuf, 4}), SCE_OK, "SetParam user");
    const char png[] = "\x89PNG fake";
    CHECK(h.mem.write_bytes(kOut2, png, sizeof(png)));
    SceSaveDataIcon icon{};
    icon.buf = kOut2;
    icon.buf_size = sizeof(png);
    icon.data_size = sizeof(png);
    h.put(kOut, icon);
    expect(h.call("sceSaveDataSaveIcon", {kResult, kOut}), SCE_OK, "SaveIcon");
    CHECK(std::filesystem::file_size(host_dir / "sce_sys" / "icon0.png") == sizeof(png));

    expect(h.call("sceSaveDataGetMountInfo", {kResult, kOut}), SCE_OK, "GetMountInfo");
    const auto info = h.get<SceSaveDataMountInfo>(kOut);
    CHECK(info.blocks == 96 && info.free_blocks > 0 && info.free_blocks < 96);

    // A second mount of a mounted save is BUSY; unmount, then it is gone.
    const SceSaveDataMountResult first = h.get<SceSaveDataMountResult>(kResult);
    expect(h.mount2("SAVE0001", SCE_SAVE_DATA_MOUNT_MODE_RDWR), SCE_SAVE_DATA_ERROR_BUSY, "double mount");
    h.put(kResult, first);
    expect(h.umount(), SCE_OK, "umount");
    expect(h.umount(), SCE_SAVE_DATA_ERROR_NOT_FOUND, "umount twice");
    CHECK(!h.vfs.resolve_path("/savedata0/slots/slot0.bin"));

    // CREATE on an existing save fails; a plain read-only mount works.
    expect(h.mount2("SAVE0001", SCE_SAVE_DATA_MOUNT_MODE_RDWR | SCE_SAVE_DATA_MOUNT_MODE_CREATE),
           SCE_SAVE_DATA_ERROR_EXISTS, "create existing");
    expect(h.mount2("SAVE0001", SCE_SAVE_DATA_MOUNT_MODE_RDONLY), SCE_OK, "mount rdonly");
    CHECK(h.mount_point() == "/savedata0"); // the slot is reused
    CHECK(h.get<SceSaveDataMountResult>(kResult).mount_status == SCE_SAVE_DATA_MOUNT_STATUS_NOTHING);

    h.put_str(kPath, "/savedata0/slots/slot0.bin");
    const u32 rfd = h.call("sceKernelOpen", {kPath, ORBIS_O_RDONLY, 0});
    CHECK(static_cast<i32>(rfd) >= 3);
    CHECK(h.mem.write_bytes(kOut, std::string(64, '\0').data(), 64));
    expect(h.call("sceKernelRead", {rfd, kOut, 64}), static_cast<u32>(payload.size()), "read");
    CHECK(std::string(h.mem.to_host_ptr<char>(kOut)) == payload);
    expect(h.call("sceKernelClose", {rfd}), SCE_OK, "close read");
    // Read-only: writes are refused.
    expect(h.call("sceKernelOpen", {kPath, ORBIS_O_WRONLY, 0}) & 0xFFFF0000u, 0x80020000u, "open rdonly for write");
    expect(h.call("sceSaveDataSetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_TITLE, kBuf, 4}),
           SCE_SAVE_DATA_ERROR_BAD_MOUNTED, "SetParam rdonly");

    // Params persisted across the remount.
    u64 got = 0;
    expect(h.call("sceSaveDataGetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_TITLE, kOut, 128, kOut2}), SCE_OK, "GetParam title");
    CHECK(std::string(h.mem.to_host_ptr<char>(kOut)) == "Chapter 3");
    got = h.get<u64>(kOut2);
    CHECK(got == 10);
    expect(h.call("sceSaveDataGetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_ALL, kOut, sizeof(SceSaveDataParam), 0}),
           SCE_OK, "GetParam all");
    const auto param = h.get<SceSaveDataParam>(kOut);
    CHECK(param.user_param == 42 && std::string(param.title) == "Chapter 3" && param.mtime > 0);
    expect(h.umount(), SCE_OK, "umount rdonly");
    expect(h.call("sceSaveDataGetParam", {kResult, SCE_SAVE_DATA_PARAM_TYPE_TITLE, kOut, 128, 0}),
           SCE_SAVE_DATA_ERROR_NOT_MOUNTED, "GetParam unmounted");
}

void test_search_and_delete(Harness& h) {
    std::cout << "[SaveData] DirNameSearch reports what is on disk; Delete removes it\n";
    expect(h.mount2("SAVE0002", SCE_SAVE_DATA_MOUNT_MODE_RDWR | SCE_SAVE_DATA_MOUNT_MODE_CREATE), SCE_OK, "create 2");
    expect(h.umount(), SCE_OK, "umount 2");
    expect(h.mount2("AUTOSAVE", SCE_SAVE_DATA_MOUNT_MODE_RDWR | SCE_SAVE_DATA_MOUNT_MODE_CREATE), SCE_OK, "create auto");
    expect(h.umount(), SCE_OK, "umount auto");

    auto search = [&](const char* pattern, u32 order, u32 capacity) {
        SceSaveDataDirName pat{};
        if (pattern) std::strncpy(pat.data, pattern, sizeof(pat.data) - 1);
        h.put(kDirName, pat);
        SceSaveDataDirNameSearchCond cond{};
        cond.user_id = kUser;
        cond.dir_name = pattern ? kDirName : 0;
        cond.key = SCE_SAVE_DATA_SORT_KEY_DIRNAME;
        cond.order = order;
        h.put(kArgs, cond);
        SceSaveDataDirNameSearchResult res{};
        res.dir_names = kOut;
        res.dir_names_num = capacity;
        res.params = kOut2;
        h.put(kResult, res);
        expect(h.call("sceSaveDataDirNameSearch", {kArgs, kResult}), SCE_OK, "DirNameSearch");
        std::vector<std::string> names;
        res = h.get<SceSaveDataDirNameSearchResult>(kResult);
        for (u32 i = 0; i < res.set_num; ++i)
            names.push_back(h.get<SceSaveDataDirName>(kOut + i * sizeof(SceSaveDataDirName)).data);
        return std::make_pair(res.hit_num, names);
    };

    auto [hits, names] = search(nullptr, SCE_SAVE_DATA_SORT_ORDER_ASCENT, 8);
    CHECK(hits == 3);
    CHECK((names == std::vector<std::string>{"AUTOSAVE", "SAVE0001", "SAVE0002"}));
    CHECK(std::string(h.get<SceSaveDataParam>(kOut2 + sizeof(SceSaveDataParam)).title) == "Chapter 3");

    std::tie(hits, names) = search("SAVE%", SCE_SAVE_DATA_SORT_ORDER_DESCENT, 8);
    CHECK(hits == 2 && (names == std::vector<std::string>{"SAVE0002", "SAVE0001"}));

    std::tie(hits, names) = search("save000_", SCE_SAVE_DATA_SORT_ORDER_ASCENT, 1); // capacity 1
    CHECK(hits == 2 && (names == std::vector<std::string>{"SAVE0001"}));

    // Delete: refused while mounted, then removes the folder.
    SceSaveDataDirName dn{};
    std::strncpy(dn.data, "SAVE0002", sizeof(dn.data) - 1);
    h.put(kOut2, dn);
    SceSaveDataDelete del{};
    del.user_id = kUser;
    del.dir_name = kOut2;
    h.put(kBuf, del);
    expect(h.mount2("SAVE0002", SCE_SAVE_DATA_MOUNT_MODE_RDONLY), SCE_OK, "mount for delete");
    expect(h.call("sceSaveDataDelete", {kBuf}), SCE_SAVE_DATA_ERROR_BUSY, "delete mounted");
    expect(h.umount(), SCE_OK, "umount before delete");
    expect(h.call("sceSaveDataDelete", {kBuf}), SCE_OK, "delete");
    std::tie(hits, names) = search(nullptr, SCE_SAVE_DATA_SORT_ORDER_ASCENT, 8);
    CHECK(hits == 2 && (names == std::vector<std::string>{"AUTOSAVE", "SAVE0001"}));

    // Malformed dir names never reach the host filesystem.
    expect(h.mount2("../escape", SCE_SAVE_DATA_MOUNT_MODE_CREATE2), SCE_SAVE_DATA_ERROR_PARAMETER, "dir traversal");

    expect(h.call("sceSaveDataGetEventResult", {0, kOut}), SCE_SAVE_DATA_ERROR_NOT_FOUND, "GetEventResult");
}

void test_save_memory(Harness& h) {
    std::cout << "[SaveData] save-data memory setup / set / get\n";
    SceSaveDataMemoryGet2 get{};
    get.user_id = kUser;
    get.data = kArgs + 0x200;
    h.put(kArgs + 0x100, get);
    SceSaveDataMemoryData d{};
    d.buf = kOut;
    d.buf_size = 5;
    d.offset = 100;
    h.put(kArgs + 0x200, d);
    expect(h.call("sceSaveDataGetSaveDataMemory2", {kArgs + 0x100}), SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY, "get before setup");

    SceSaveDataMemorySetup2 setup{};
    setup.user_id = kUser;
    setup.memory_size = 4096;
    h.put(kArgs, setup);
    h.put(kResult, SceSaveDataMemorySetupResult{.existed_memory_size = 99});
    expect(h.call("sceSaveDataSetupSaveDataMemory2", {kArgs, kResult}), SCE_OK, "setup");
    CHECK(h.get<SceSaveDataMemorySetupResult>(kResult).existed_memory_size == 0);

    h.put_str(kBuf, "hello");
    d.buf = kBuf;
    h.put(kArgs + 0x200, d);
    SceSaveDataMemorySet2 set{};
    set.user_id = kUser;
    set.data = kArgs + 0x200;
    set.data_num = 1;
    h.put(kArgs + 0x300, set);
    expect(h.call("sceSaveDataSetSaveDataMemory2", {kArgs + 0x300}), SCE_OK, "set2");

    d.buf = kOut;
    h.put(kArgs + 0x200, d);
    expect(h.call("sceSaveDataGetSaveDataMemory2", {kArgs + 0x100}), SCE_OK, "get2");
    CHECK(std::string(h.mem.to_host_ptr<char>(kOut), 5) == "hello");

    // The v1 calls share slot 0; a second setup sees the existing memory.
    h.put_str(kBuf, "XY");
    expect(h.call("sceSaveDataSetSaveDataMemory", {static_cast<u64>(kUser), kBuf, 2, 4094}), SCE_OK, "set v1");
    expect(h.call("sceSaveDataGetSaveDataMemory", {static_cast<u64>(kUser), kOut, 2, 4094}), SCE_OK, "get v1");
    CHECK(std::string(h.mem.to_host_ptr<char>(kOut), 2) == "XY");
    expect(h.call("sceSaveDataSetupSaveDataMemory2", {kArgs, kResult}), SCE_OK, "setup again");
    CHECK(h.get<SceSaveDataMemorySetupResult>(kResult).existed_memory_size == 4096);
    const auto mem_file = h.root / "savedata" / std::to_string(kUser) / "TEST00001" / "sce_sdmemory" / "memory.dat";
    CHECK(std::filesystem::file_size(mem_file) == 4096);
    expect(h.call("sceSaveDataTerminate", {}), SCE_OK, "Terminate");
}

void test_trophy(Harness& h) {
    std::cout << "[NpTrophy] context / handle / register / unlock round trip\n";
    expect(h.call("sceNpTrophyCreateContext", {kOut, static_cast<u64>(kUser), 0, 1}),
           SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT, "options must be 0");
    expect(h.call("sceNpTrophyCreateContext", {kOut, static_cast<u64>(kUser), 0, 0}), SCE_OK, "CreateContext");
    const i32 context = h.get<i32>(kOut);
    CHECK(context == 1);
    expect(h.call("sceNpTrophyCreateContext", {kOut, static_cast<u64>(kUser), 0, 0}),
           SCE_NP_TROPHY_ERROR_CONTEXT_ALREADY_EXISTS, "duplicate context");
    expect(h.call("sceNpTrophyCreateHandle", {kOut}), SCE_OK, "CreateHandle");
    const i32 handle = h.get<i32>(kOut);
    CHECK(handle == 1);
    const u64 c = static_cast<u64>(context), hd = static_cast<u64>(handle);

    expect(h.call("sceNpTrophyUnlockTrophy", {c, hd, 3, kOut2}), SCE_NP_TROPHY_ERROR_NOT_REGISTERED, "unlock unregistered");
    expect(h.call("sceNpTrophyRegisterContext", {c, hd, 0}), SCE_OK, "RegisterContext");

    h.put<i32>(kOut2, 1234);
    expect(h.call("sceNpTrophyUnlockTrophy", {c, hd, 3, kOut2}), SCE_OK, "unlock 3");
    CHECK(h.get<i32>(kOut2) == SCE_NP_TROPHY_INVALID_TROPHY_ID); // no platinum
    expect(h.call("sceNpTrophyUnlockTrophy", {c, hd, 3, kOut2}), SCE_NP_TROPHY_ERROR_TROPHY_ALREADY_UNLOCKED, "unlock again");
    expect(h.call("sceNpTrophyUnlockTrophy", {c, hd, 200, kOut2}), SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID, "bad id");

    expect(h.call("sceNpTrophyGetTrophyUnlockState", {c, hd, kOut, kOut2}), SCE_OK, "GetTrophyUnlockState");
    const auto flags = h.get<SceNpTrophyFlagArray>(kOut);
    CHECK(flags.flag_bits[0] == (1u << 3) && flags.flag_bits[1] == 0);
    CHECK(h.get<u32>(kOut2) == 4);

    SceNpTrophyDetails details{};
    details.size = sizeof(details);
    SceNpTrophyData data{};
    data.size = sizeof(data);
    h.put(kOut, details);
    h.put(kOut2, data);
    expect(h.call("sceNpTrophyGetTrophyInfo", {c, hd, 3, kOut, kOut2}), SCE_OK, "GetTrophyInfo");
    CHECK(h.get<SceNpTrophyDetails>(kOut).trophy_id == 3 && h.get<SceNpTrophyDetails>(kOut).size == sizeof(details));
    data = h.get<SceNpTrophyData>(kOut2);
    CHECK(data.unlocked == 1 && data.trophy_id == 3 && data.timestamp > 62135596800ull * 1000000ull);
    expect(h.call("sceNpTrophyGetTrophyInfo", {c, hd, 4, 0, kOut2}), SCE_OK, "GetTrophyInfo locked");
    CHECK(h.get<SceNpTrophyData>(kOut2).unlocked == 0);

    SceNpTrophyGameDetails game{};
    game.size = sizeof(game);
    SceNpTrophyGameData game_data{};
    game_data.size = sizeof(game_data);
    h.put(kOut, game);
    h.put(kOut2, game_data);
    expect(h.call("sceNpTrophyGetGameInfo", {c, hd, kOut, kOut2}), SCE_OK, "GetGameInfo");
    CHECK(h.get<SceNpTrophyGameData>(kOut2).unlocked_trophies == 1);

    SceNpTrophyGroupData group_data{};
    group_data.size = sizeof(group_data);
    h.put(kOut2, group_data);
    expect(h.call("sceNpTrophyGetGroupInfo", {c, hd, static_cast<u64>(-1), 0, kOut2}), SCE_OK, "GetGroupInfo");
    CHECK(h.get<SceNpTrophyGroupData>(kOut2).group_id == -1);
    expect(h.call("sceNpTrophyGetGroupInfo", {c, hd, static_cast<u64>(-5), 0, kOut2}),
           SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID, "bad group");

    expect(h.call("sceNpTrophyShowTrophyList", {c, hd}), SCE_OK, "ShowTrophyList");
    expect(h.call("sceNpTrophyDestroyHandle", {hd}), SCE_OK, "DestroyHandle");
    expect(h.call("sceNpTrophyDestroyHandle", {hd}), SCE_NP_TROPHY_ERROR_INVALID_HANDLE, "DestroyHandle twice");
    expect(h.call("sceNpTrophyGetGameInfo", {c, hd, kOut, kOut2}), SCE_NP_TROPHY_ERROR_INVALID_HANDLE, "stale handle");
    expect(h.call("sceNpTrophyDestroyContext", {c}), SCE_OK, "DestroyContext");
    expect(h.call("sceNpTrophyDestroyContext", {c}), SCE_NP_TROPHY_ERROR_INVALID_CONTEXT, "DestroyContext twice");

    // Handles are limited to 4 and reused after destruction.
    for (int i = 1; i <= 4; ++i) {
        expect(h.call("sceNpTrophyCreateHandle", {kOut}), SCE_OK, "CreateHandle x4");
        CHECK(h.get<i32>(kOut) == i);
    }
    expect(h.call("sceNpTrophyCreateHandle", {kOut}), SCE_NP_TROPHY_ERROR_HANDLE_EXCEEDS_MAX, "5th handle");
}
} // namespace

int main() {
    std::cout << "=== libSceSaveData / libSceNpTrophy HLE ===\n";
    const auto root = std::filesystem::temp_directory_path() / "ps4_test_save_data_trophy";
    std::filesystem::remove_all(root);
    write_param_sfo(root / "app0" / "sce_sys" / "param.sfo", "TEST00001");
    {
        Harness h(root);
        test_save_data(h);
        test_search_and_delete(h);
        test_save_memory(h);
        test_trophy(h);
    }
    std::filesystem::remove_all(root);
    std::cout << "=== All save data / trophy tests passed ===\n";
    return 0;
}
