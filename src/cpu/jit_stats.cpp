#include "ps4/jit_stats.hpp"
#include "ps4/memory.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <utility>

namespace ps4 {

// =============================================================================
// Opcode normalization
// =============================================================================

namespace {

constexpr u32 KEY_BITS = 18;
constexpr u32 KEY_COUNT = 1u << KEY_BITS;

bool legacy_has_modrm(u8 op) {
  if (op < 0x40)
    return (op & 7) < 4; // ALU rows: 00-03, 08-0B, ..., 38-3B
  if (op == 0x62 || op == 0x63 || op == 0x69 || op == 0x6B)
    return true;
  if (op >= 0x80 && op <= 0x8F)
    return true;
  if (op == 0xC0 || op == 0xC1 || op == 0xC6 || op == 0xC7)
    return true;
  if (op >= 0xD0 && op <= 0xD3)
    return true;
  if (op >= 0xD8 && op <= 0xDF)
    return true; // x87
  return op == 0xF6 || op == 0xF7 || op == 0xFE || op == 0xFF;
}

bool twobyte_has_modrm(u8 op) {
  if (op >= 0x05 && op <= 0x0C)
    return false; // syscall, clts, sysret, invd, wbinvd, ud2, ...
  if (op == 0x0E || (op >= 0x30 && op <= 0x3F) || op == 0x77)
    return false; // femms, rdtsc/rdmsr/sysenter..., emms
  if (op >= 0x80 && op <= 0x8F)
    return false; // jcc rel32
  if (op == 0xA0 || op == 0xA1 || op == 0xA2 || op == 0xA8 || op == 0xA9 ||
      op == 0xAA)
    return false; // push/pop fs/gs, cpuid, rsm
  return !(op >= 0xC8 && op <= 0xCF); // bswap
}

bool is_legacy_group(u8 op) {
  switch (op) {
  case 0x80: case 0x81: case 0x82: case 0x83: case 0x8F:
  case 0xC0: case 0xC1: case 0xC6: case 0xC7:
  case 0xD0: case 0xD1: case 0xD2: case 0xD3:
  case 0xD8: case 0xD9: case 0xDA: case 0xDB:
  case 0xDC: case 0xDD: case 0xDE: case 0xDF:
  case 0xF6: case 0xF7: case 0xFE: case 0xFF:
    return true;
  default:
    return false;
  }
}

bool is_twobyte_group(u8 op) {
  switch (op) {
  case 0x00: case 0x01: case 0x0D: case 0x18: case 0x71: case 0x72:
  case 0x73: case 0xAE: case 0xB9: case 0xBA: case 0xC7:
    return true;
  default:
    return false;
  }
}

} // namespace

u32 OpcodeKey::index() const noexcept {
  return static_cast<u32>(op) | (static_cast<u32>(map & 3) << 8) |
         (static_cast<u32>(pfx & 3) << 10) | (static_cast<u32>(vex) << 12) |
         (static_cast<u32>(ext & 7) << 13) | (static_cast<u32>(mod & 3) << 16);
}

OpcodeKey OpcodeKey::from_index(u32 index) noexcept {
  OpcodeKey k;
  k.op = static_cast<u8>(index & 0xFF);
  k.map = static_cast<u8>((index >> 8) & 3);
  k.pfx = static_cast<u8>((index >> 10) & 3);
  k.vex = ((index >> 12) & 1) != 0;
  k.ext = static_cast<u8>((index >> 13) & 7);
  k.mod = static_cast<u8>((index >> 16) & 3);
  return k;
}

bool OpcodeKey::has_group_ext() const noexcept {
  if (mod == 0 || mod == 3)
    return false;
  if (map == 0)
    return is_legacy_group(op);
  if (map == 1)
    return is_twobyte_group(op);
  return false;
}

std::string OpcodeKey::to_string() const {
  if (mod == 3)
    return "<undecodable>";
  static constexpr const char *PFX[] = {"", "66", "F3", "F2"};
  static constexpr const char *MAP[] = {"", "0F", "0F 38", "0F 3A"};
  char buf[64];
  int n = 0;
  if (vex) {
    n = std::snprintf(buf, sizeof(buf), "VEX.%s%s%s %02X", PFX[pfx & 3],
                      pfx ? "." : "", map == 1 ? "0F" : map == 2 ? "0F38" : "0F3A",
                      op);
  } else {
    n = std::snprintf(buf, sizeof(buf), "%s%s%s%s%02X", PFX[pfx & 3],
                      pfx ? " " : "", MAP[map & 3], map ? " " : "", op);
  }
  std::string s(buf, static_cast<usize>(n));
  if (has_group_ext()) {
    s += " /";
    s += static_cast<char>('0' + (ext & 7));
  }
  if (mod == 1)
    s += " reg";
  else if (mod == 2)
    s += " mem";
  return s;
}

OpcodeKey normalize_opcode(const u8 *code, usize avail,
                           OpcodeNotes *notes) noexcept {
  OpcodeKey key;
  OpcodeNotes local;
  bool saw66 = false;
  u8 rep = 0;
  usize i = 0;

  for (; i < avail && i < 14; ++i) {
    const u8 b = code[i];
    if (b == 0x66)
      saw66 = true;
    else if (b == 0xF2 || b == 0xF3)
      rep = b;
    else if (b == 0xF0)
      local.lock = true;
    else if (b == 0x64 || b == 0x65)
      local.seg_fs_gs = true;
    else if (b == 0x2E || b == 0x36 || b == 0x3E || b == 0x26 || b == 0x67)
      continue;
    else if (b >= 0x40 && b <= 0x4F)
      local.rex_w = (b & 0x08) != 0;
    else
      break;
  }
  key.pfx = rep == 0xF3 ? 2 : rep == 0xF2 ? 3 : saw66 ? 1 : 0;

  auto undecodable = [&]() {
    OpcodeKey bad;
    bad.mod = 3;
    if (notes)
      *notes = local;
    return bad;
  };

  if (i >= avail)
    return undecodable();

  bool has_modrm = false;
  const u8 b0 = code[i];
  if (b0 == 0xC4 || b0 == 0xC5) {
    // VEX (always VEX in 64-bit mode). pp replaces the mandatory prefix.
    key.vex = true;
    u8 pp = 0;
    if (b0 == 0xC5) {
      if (i + 2 >= avail)
        return undecodable();
      pp = code[i + 1] & 3;
      key.map = 1;
      i += 2;
    } else {
      if (i + 3 >= avail)
        return undecodable();
      const u8 mmmmm = code[i + 1] & 0x1F;
      if (mmmmm < 1 || mmmmm > 3)
        return undecodable();
      key.map = mmmmm;
      local.rex_w = (code[i + 2] & 0x80) != 0;
      pp = code[i + 2] & 3;
      i += 3;
    }
    key.pfx = pp; // 0 none, 1 66, 2 F3, 3 F2 (same encoding as OpcodeKey)
    key.op = code[i++];
    has_modrm = !(key.map == 1 && key.op == 0x77); // vzeroupper/vzeroall
  } else if (b0 == 0x0F) {
    if (i + 1 >= avail)
      return undecodable();
    const u8 b1 = code[i + 1];
    if (b1 == 0x38 || b1 == 0x3A) {
      if (i + 2 >= avail)
        return undecodable();
      key.map = b1 == 0x38 ? 2 : 3;
      key.op = code[i + 2];
      i += 3;
      has_modrm = true;
    } else {
      key.map = 1;
      key.op = b1;
      i += 2;
      has_modrm = twobyte_has_modrm(b1);
    }
  } else {
    key.op = b0;
    i += 1;
    has_modrm = legacy_has_modrm(b0);
  }

  if (has_modrm) {
    if (i >= avail)
      return undecodable();
    const u8 modrm = code[i];
    key.mod = (modrm >> 6) == 3 ? 1 : 2;
    if (key.has_group_ext())
      key.ext = (modrm >> 3) & 7;
  }

  if (notes)
    *notes = local;
  return key;
}

// =============================================================================
// Counters
// =============================================================================

namespace jit_stats_detail {
std::atomic<int> g_state{0};
} // namespace jit_stats_detail

namespace {

struct Slot {
  std::atomic<u64> count{0};
  std::atomic<u64> rex_w{0};
  std::atomic<u64> seg{0};
  std::atomic<u64> lock{0};
};

struct BlockEndCount {
  u64 blocks{0};           // compiled blocks that ended here
  u64 failed_compiles{0};  // compile attempts with 0 instructions (retried each visit)
};

// Allocated once on first enable and never freed: hooks may still run on
// guest threads while atexit handlers write the dump.
Slot *g_interp = nullptr;
std::atomic<u64> g_jit_retired{0};
std::atomic<u64> g_interp_retired{0};
std::atomic<u64> g_jit_blocks{0};
std::atomic<u64> g_blocks_compiled{0};
std::atomic<u64> g_compile_failures{0};
std::atomic<bool> g_written{false}; // explicit write_jit_stats() supersedes atexit

std::mutex &block_end_mutex() {
  static std::mutex *m = new std::mutex;
  return *m;
}

std::map<std::pair<u32, JitStopReason>, BlockEndCount> &block_ends() {
  static auto *m = new std::map<std::pair<u32, JitStopReason>, BlockEndCount>;
  return *m;
}

std::string &output_path() {
  static auto *p = new std::string;
  return *p;
}

void ensure_tables() {
  static std::once_flag once;
  std::call_once(once, [] { g_interp = new Slot[KEY_COUNT]; });
}

usize readable_bytes(const MemoryManager &memory, GuestAddr rip) {
  usize n = 15;
  while (n > 0 && !memory.is_valid_address(rip, n))
    --n;
  return n;
}

const char *reason_name(JitStopReason r) {
  switch (r) {
  case JitStopReason::Terminator:
    return "terminator";
  case JitStopReason::Unsupported:
    return "unsupported";
  case JitStopReason::Limit:
    return "limit";
  }
  return "unknown";
}

void write_key_fields(std::ostream &os, const OpcodeKey &k) {
  static constexpr const char *PFX[] = {"", "66", "F3", "F2"};
  static constexpr const char *MAP[] = {"", "0F", "0F38", "0F3A"};
  static constexpr const char *MOD[] = {"", "reg", "mem", "bad"};
  char op[3];
  std::snprintf(op, sizeof(op), "%02X", k.op);
  os << "\"key\": \"" << k.to_string() << "\", \"map\": \"" << MAP[k.map & 3]
     << "\", \"op\": \"" << op << "\", \"pfx\": \"" << PFX[k.pfx & 3]
     << "\", \"vex\": " << (k.vex ? "true" : "false") << ", \"ext\": ";
  if (k.has_group_ext())
    os << static_cast<int>(k.ext);
  else
    os << "null";
  os << ", \"mod\": \"" << MOD[k.mod & 3] << "\"";
}

} // namespace

int jit_stats_detail::init_from_env() noexcept {
  static std::once_flag once;
  std::call_once(once, [] {
    const char *v = std::getenv("PS4_JIT_STATS");
    if (v && *v && std::strcmp(v, "0") != 0) {
      output_path() = std::strcmp(v, "1") == 0 ? "jit_stats.json" : v;
      ensure_tables();
      int expected = 0;
      g_state.compare_exchange_strong(expected, 2);
      std::atexit([] {
        if (!g_written.load())
          write_jit_stats();
      });
    } else {
      int expected = 0;
      g_state.compare_exchange_strong(expected, 1);
    }
  });
  return g_state.load(std::memory_order_relaxed);
}

void set_jit_stats_enabled(bool enabled) {
  if (enabled)
    ensure_tables();
  jit_stats_detail::g_state.store(enabled ? 2 : 1, std::memory_order_release);
}

void reset_jit_stats() {
  if (g_interp) {
    for (u32 i = 0; i < KEY_COUNT; ++i) {
      g_interp[i].count = 0;
      g_interp[i].rex_w = 0;
      g_interp[i].seg = 0;
      g_interp[i].lock = 0;
    }
  }
  g_jit_retired = 0;
  g_interp_retired = 0;
  g_jit_blocks = 0;
  g_blocks_compiled = 0;
  g_compile_failures = 0;
  std::lock_guard<std::mutex> lock(block_end_mutex());
  block_ends().clear();
}

void jit_stats_record_interp(const MemoryManager &memory,
                             GuestAddr rip) noexcept {
  if (!g_interp)
    return;
  g_interp_retired.fetch_add(1, std::memory_order_relaxed);
  const usize avail = readable_bytes(memory, rip);
  const u8 *code =
      avail ? static_cast<const u8 *>(memory.to_host_ptr(rip, avail)) : nullptr;
  OpcodeNotes notes;
  const OpcodeKey key =
      code ? normalize_opcode(code, avail, &notes) : normalize_opcode(nullptr, 0);
  Slot &slot = g_interp[key.index()];
  slot.count.fetch_add(1, std::memory_order_relaxed);
  if (notes.rex_w)
    slot.rex_w.fetch_add(1, std::memory_order_relaxed);
  if (notes.seg_fs_gs)
    slot.seg.fetch_add(1, std::memory_order_relaxed);
  if (notes.lock)
    slot.lock.fetch_add(1, std::memory_order_relaxed);
}

void jit_stats_record_jit_block(usize instructions) noexcept {
  g_jit_blocks.fetch_add(1, std::memory_order_relaxed);
  g_jit_retired.fetch_add(instructions, std::memory_order_relaxed);
}

void jit_stats_record_block_end(const MemoryManager &memory, GuestAddr stop_rip,
                                JitStopReason reason,
                                usize instructions) noexcept {
  const usize avail = readable_bytes(memory, stop_rip);
  const u8 *code = avail ? static_cast<const u8 *>(
                               memory.to_host_ptr(stop_rip, avail))
                         : nullptr;
  const OpcodeKey key =
      code ? normalize_opcode(code, avail) : normalize_opcode(nullptr, 0);
  if (instructions)
    g_blocks_compiled.fetch_add(1, std::memory_order_relaxed);
  else
    g_compile_failures.fetch_add(1, std::memory_order_relaxed);
  try {
    std::lock_guard<std::mutex> lock(block_end_mutex());
    BlockEndCount &c = block_ends()[{key.index(), reason}];
    if (instructions)
      c.blocks++;
    else
      c.failed_compiles++;
  } catch (...) {
    // Telemetry must never take down the emulator.
  }
}

JitStatsTotals jit_stats_totals() {
  JitStatsTotals t;
  t.jit_retired = g_jit_retired.load(std::memory_order_relaxed);
  t.interp_retired = g_interp_retired.load(std::memory_order_relaxed);
  t.jit_block_executions = g_jit_blocks.load(std::memory_order_relaxed);
  t.blocks_compiled = g_blocks_compiled.load(std::memory_order_relaxed);
  t.compile_failures = g_compile_failures.load(std::memory_order_relaxed);
  return t;
}

std::vector<OpcodeCount> jit_stats_interp_histogram() {
  std::vector<OpcodeCount> out;
  if (!g_interp)
    return out;
  for (u32 i = 0; i < KEY_COUNT; ++i) {
    const u64 c = g_interp[i].count.load(std::memory_order_relaxed);
    if (c)
      out.push_back({OpcodeKey::from_index(i), c});
  }
  std::sort(out.begin(), out.end(), [](const OpcodeCount &a, const OpcodeCount &b) {
    return a.count != b.count ? a.count > b.count : a.key.index() < b.key.index();
  });
  return out;
}

void dump_jit_stats(std::ostream &os) {
  const JitStatsTotals t = jit_stats_totals();
  const u64 total = t.jit_retired + t.interp_retired;
  char frac[32];
  std::snprintf(frac, sizeof(frac), "%.6f",
                total ? static_cast<double>(t.jit_retired) / static_cast<double>(total)
                      : 0.0);

  os << "{\n  \"version\": 1,\n  \"totals\": {\n"
     << "    \"guest_instructions\": " << total << ",\n"
     << "    \"jit_retired\": " << t.jit_retired << ",\n"
     << "    \"interp_retired\": " << t.interp_retired << ",\n"
     << "    \"jit_fraction\": " << frac << ",\n"
     << "    \"jit_block_executions\": " << t.jit_block_executions << ",\n"
     << "    \"blocks_compiled\": " << t.blocks_compiled << ",\n"
     << "    \"compile_failures\": " << t.compile_failures << "\n  },\n";

  os << "  \"interp_opcodes\": [";
  bool first = true;
  for (const OpcodeCount &e : jit_stats_interp_histogram()) {
    const Slot &s = g_interp[e.key.index()];
    os << (first ? "\n" : ",\n") << "    {";
    write_key_fields(os, e.key);
    os << ", \"count\": " << e.count
       << ", \"rex_w\": " << s.rex_w.load(std::memory_order_relaxed)
       << ", \"seg_fs_gs\": " << s.seg.load(std::memory_order_relaxed)
       << ", \"lock\": " << s.lock.load(std::memory_order_relaxed) << "}";
    first = false;
  }
  os << (first ? "],\n" : "\n  ],\n");

  std::vector<std::pair<std::pair<u32, JitStopReason>, BlockEndCount>> ends;
  {
    std::lock_guard<std::mutex> lock(block_end_mutex());
    ends.assign(block_ends().begin(), block_ends().end());
  }
  std::sort(ends.begin(), ends.end(), [](const auto &a, const auto &b) {
    const u64 wa = a.second.blocks + a.second.failed_compiles;
    const u64 wb = b.second.blocks + b.second.failed_compiles;
    return wa != wb ? wa > wb : a.first < b.first;
  });
  os << "  \"block_ends\": [";
  first = true;
  for (const auto &[k, c] : ends) {
    os << (first ? "\n" : ",\n") << "    {";
    write_key_fields(os, OpcodeKey::from_index(k.first));
    os << ", \"reason\": \"" << reason_name(k.second)
       << "\", \"blocks\": " << c.blocks
       << ", \"failed_compiles\": " << c.failed_compiles << "}";
    first = false;
  }
  os << (first ? "]\n}\n" : "\n  ]\n}\n");
}

void write_jit_stats() {
  if (!jit_stats_enabled())
    return;
  g_written = true;
  const std::string &path = output_path();
  if (path.empty() || path == "-") {
    dump_jit_stats(std::cerr);
    return;
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "[JIT-STATS] cannot write %s\n", path.c_str());
    return;
  }
  dump_jit_stats(out);
  std::fprintf(stderr, "[JIT-STATS] wrote %s\n", path.c_str());
}

} // namespace ps4
