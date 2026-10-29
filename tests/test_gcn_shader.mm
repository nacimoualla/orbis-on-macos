// Phase 19: GCN shader decode -> MSL translation -> Metal compile -> render,
// using the real shaders Flappy Bird (Unity) binds, captured from guest
// memory (see the [PM4 Draw] VS/PS ISA dumps in the smoke-test log).

#import <Metal/Metal.h>

#include "ps4/gcn_pipeline.hpp"
#include "ps4/gcn_shader.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ps4;

namespace {

// Unity sprite/particle VS at 0x1625ae300 (rsrc1=0x002c0183 rsrc2=0x20).
const u32 kSpriteVs[] = {
    0xbeeb03ff, 0x00000022, 0xbe802100, 0x7e000280, 0xc3080900, 0xc3100d44, 0xc2800500, 0xbf8c007f,
    0x10020810, 0x100e0811, 0x3e020a14, 0x101c0812, 0x3e020c18, 0x3e0e0a15, 0x0604021c, 0xd2820001,
    0x041e0c19, 0x101e0813, 0x3e1c0a16, 0x10060420, 0x0602021d, 0xd2820007, 0x043a0c1a, 0x101c0421,
    0x10080422, 0x3e1e0a17, 0x100a0423, 0x06040e1e, 0x3e060224, 0x3e1e0c1b, 0xd2820006, 0x04160227,
    0x3e1c0225, 0x3e080226, 0x06021e1f, 0x3e060428, 0x3e1c0429, 0x3e08042a, 0x3e0c042b, 0x3e06022c,
    0x3e1c022d, 0x3e08022e, 0x3e0c022f, 0xf80008cf, 0x06040e03, 0x7e040202, 0x7e020203, 0x3e041800,
    0x3e021a01, 0xf800021f, 0x00000102, 0xf800020f, 0x0b0a0908, 0xbf810000,
};
// Its fetch shader at 0x1625ae434 (called via s_swappc_b64 s[0:1]).
const u32 kSpriteFetch[] = {
    0xc0880300, 0xc08a0304, 0xc08c0308, 0xbf8c007f, 0xe0082000, 0x80040400, 0xe00c2000, 0x80050800,
    0xe0042000, 0x80060c00, 0xbf8c0000, 0xbe802000,
};
// Textured PS at 0x1625ae500 (rsrc1=0x002c0042 rsrc2=0x18).
const u32 kSpritePs[] = {
    0xbeeb03ff, 0x00000013, 0xbefc030c, 0xbeea047e, 0xbefe0a7e, 0xc8080400, 0xc80c0500, 0xc8090401,
    0xc80d0501, 0xf0800f00, 0x00400202, 0xc8200000, 0xc81c0100, 0xc8180200, 0xc8000300, 0xc8210001,
    0xc81d0101, 0xc8190201, 0xc8010301, 0xbf8c0f70, 0xd2100001, 0x08021102, 0xd2100002, 0x08020707,
    0xd2100003, 0x08020906, 0xd2100000, 0x08020b00, 0x5e020501, 0x5e000103, 0xbefe046a, 0xf8001c0f,
    0x00000001, 0xbf810000,
};
// Gnm clear VS (vertex-id rect) at 0x162554d00 and clear PS at 0x162554e00.
const u32 kClearVs[] = {
    0xbeeb03ff, 0x00000008, 0x360200c2, 0x36000081, 0xd2860002, 0x03050500, 0x4a0002c1, 0x7e020b02,
    0x7e000b00, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020001, 0xf800020f, 0x02020202, 0xbf810000,
};
const u32 kClearPs[] = {
    0xbeeb03ff, 0x00000007, 0xc2800100, 0xbf8c007f, 0x7e020201, 0x7e000203, 0x5e020200, 0x5e000002,
    0xf8001c0f, 0x00000001, 0xbf810000,
};

int g_failures = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

template <usize N>
gcn::Program decode(const u32 (&w)[N]) {
    gcn::Program p;
    std::string err;
    const bool ok = gcn::decode_program(w, N, p, err);
    if (!ok) std::fprintf(stderr, "decode error: %s\n", err.c_str());
    CHECK(ok);
    return p;
}

bool has_mnemonic(const gcn::Program& p, const std::string& m) {
    for (const auto& i : p.insts)
        if (gcn::mnemonic(i) == m) return true;
    return false;
}

void test_decode() {
    const gcn::Program vs = decode(kSpriteVs);
    const gcn::Program fetch = decode(kSpriteFetch);
    const gcn::Program ps = decode(kSpritePs);
    for (const auto* p : {&vs, &fetch, &ps})
        for (const auto& i : p->insts) CHECK(i.enc != gcn::Encoding::Unknown);
    CHECK(has_mnemonic(vs, "s_swappc_b64"));
    CHECK(has_mnemonic(vs, "s_buffer_load_dwordx16"));
    CHECK(has_mnemonic(vs, "v_mac_f32"));
    CHECK(has_mnemonic(vs, "v_mad_f32"));
    CHECK(has_mnemonic(vs, "exp"));
    CHECK(has_mnemonic(fetch, "s_load_dwordx4"));
    CHECK(has_mnemonic(fetch, "buffer_load_format_xyz"));
    CHECK(has_mnemonic(fetch, "buffer_load_format_xyzw"));
    CHECK(has_mnemonic(fetch, "buffer_load_format_xy"));
    CHECK(has_mnemonic(fetch, "s_setpc_b64"));
    CHECK(has_mnemonic(ps, "v_interp_p1_f32"));
    CHECK(has_mnemonic(ps, "v_interp_p2_f32"));
    CHECK(has_mnemonic(ps, "image_sample"));
    CHECK(has_mnemonic(ps, "v_cvt_pkrtz_f16_f32"));
    CHECK(has_mnemonic(ps, "s_wqm_b64"));
    const gcn::Program cvs = decode(kClearVs);
    CHECK(has_mnemonic(cvs, "v_mad_u32_u24"));
    CHECK(has_mnemonic(cvs, "v_cvt_f32_i32"));
    const gcn::Program cps = decode(kClearPs);
    CHECK(has_mnemonic(cps, "s_buffer_load_dwordx4"));

    // Exports: VS pos0 + param0/1, PS mrt0 compressed.
    u32 pos = 0, params = 0;
    for (const auto& i : vs.insts)
        if (i.enc == gcn::Encoding::EXP) { if (i.target == 12) ++pos; if (i.target >= 32) ++params; }
    CHECK(pos == 1 && params == 2);
    for (const auto& i : ps.insts)
        if (i.enc == gcn::Encoding::MIMG) {
            CHECK(i.srsrc == 0 && i.ssamp == 8 && i.dmask == 0xF && i.dst == 2 && i.vaddr == 2);
        }

    if (std::getenv("GCN_TEST_VERBOSE")) {
        for (const auto* p : {&vs, &fetch, &ps})
            for (const auto& i : p->insts) std::printf("  %s\n", gcn::disassemble(i).c_str());
    }
}

// Guest memory layout used by the end-to-end tests.
constexpr GuestAddr kVsAddr = 0x10000000;
constexpr GuestAddr kFetchAddr = kVsAddr + 0x134;
constexpr GuestAddr kPsAddr = 0x10000200;
constexpr GuestAddr kClearVsAddr = 0x10000400;
constexpr GuestAddr kClearPsAddr = 0x10000500;
constexpr GuestAddr kVTable = 0x10001000;
constexpr GuestAddr kVerts = 0x10002000;
constexpr GuestAddr kUvConst = 0x10003000;     // s[4:7]
constexpr GuestAddr kMvConst = 0x10003100;     // s[8:11]
constexpr GuestAddr kProjConst = 0x10003200;   // s[12:15]
constexpr GuestAddr kTexture = 0x10010000;
constexpr GuestAddr kClearColor = 0x10004000;

void write_words(MemoryManager& m, GuestAddr a, const u32* w, usize n) {
    CHECK(m.write_bytes(a, w, n * 4));
}

u32 vsharp_w1(GuestAddr base, u32 stride) { return static_cast<u32>(base >> 32) | (stride << 16); }

void setup_sprite_scene(MemoryManager& m, u32 vs_user[16], u32 ps_user[16]) {
    write_words(m, kVsAddr, kSpriteVs, std::size(kSpriteVs));
    write_words(m, kFetchAddr, kSpriteFetch, std::size(kSpriteFetch));
    write_words(m, kPsAddr, kSpritePs, std::size(kSpritePs));

    // Vertex buffer: 4 records of 88 bytes, pos(xyz) @0, color(rgba) @12, uv(xy) @28.
    // QuadList order (Flappy binds QUADLIST, VGT 0x13): TL, TR, BR, BL.
    const float px[4] = {-0.5f, 0.5f, 0.5f, -0.5f}, py[4] = {0.5f, 0.5f, -0.5f, -0.5f};
    const float uvx[4] = {0, 1, 1, 0}, uvy[4] = {0, 0, 1, 1};
    for (u32 v = 0; v < 4; ++v) {
        float rec[22] = {};
        rec[0] = px[v]; rec[1] = py[v]; rec[2] = 0.5f;
        rec[3] = 0.25f; rec[4] = 0.5f; rec[5] = 0.5f; rec[6] = 0.5f;
        rec[7] = uvx[v]; rec[8] = uvy[v];
        CHECK(m.write_bytes(kVerts + v * 88, rec, sizeof(rec)));
    }
    // V# table: dst_sel xyzw, nfmt float (7), dfmt 13 / 14 / 11.
    auto w3 = [](u32 dfmt) { return 0x00000facu | (7u << 12) | (dfmt << 15); };
    const u32 table[12] = {
        static_cast<u32>(kVerts), vsharp_w1(kVerts, 88), 4, w3(13),
        static_cast<u32>(kVerts + 12), vsharp_w1(kVerts + 12, 88), 4, w3(14),
        static_cast<u32>(kVerts + 28), vsharp_w1(kVerts + 28, 88), 4, w3(11),
    };
    write_words(m, kVTable, table, 12);

    // Constants: identity model-view (rows 0..3 of s[8:11]), identity
    // projection (rows 17..20 of s[12:15]), uv scale/offset (1,1,0,0).
    const float uv[4] = {1, 1, 0, 0};
    CHECK(m.write_bytes(kUvConst, uv, 16));
    float mv[40] = {};
    for (u32 k = 0; k < 4; ++k) mv[k * 4 + k] = 1.0f;
    CHECK(m.write_bytes(kMvConst, mv, sizeof(mv)));
    float proj[23 * 4] = {};
    for (u32 k = 0; k < 4; ++k) proj[(17 + k) * 4 + k] = 1.0f;
    CHECK(m.write_bytes(kProjConst, proj, sizeof(proj)));

    // 16x16 linear RGBA8 texture, solid (100, 50, 200, 255) with a white texel.
    std::vector<u8> tex(16 * 16 * 4);
    for (u32 k = 0; k < 256; ++k) { tex[k * 4] = 100; tex[k * 4 + 1] = 50; tex[k * 4 + 2] = 200; tex[k * 4 + 3] = 255; }
    CHECK(m.write_bytes(kTexture, tex.data(), tex.size()));

    const u32 user_vs[16] = {
        static_cast<u32>(kFetchAddr), static_cast<u32>(kFetchAddr >> 32),
        static_cast<u32>(kVTable), static_cast<u32>(kVTable >> 32),
        static_cast<u32>(kUvConst), 0x00100000, 1, 0x20077fac,
        static_cast<u32>(kMvConst), 0x00100000, 10, 0x20077fac,
        static_cast<u32>(kProjConst), 0x00100000, 23, 0x20077fac,
    };
    std::memcpy(vs_user, user_vs, sizeof(user_vs));
    // T# exactly as Flappy binds it (16x16 8_8_8_8 unorm, linear tiling
    // index 8), with only the base address changed; S# = point/wrap.
    const u32 user_ps[16] = {
        static_cast<u32>(kTexture >> 8), 0x00a00000, 0x7003c00f, 0x94800fac, 0x0001e000, 0, 0, 0,
        0x00000000, 0x00fff000, 0x01000000, 0x00000000, 0, 0, 0, 0,
    };
    std::memcpy(ps_user, user_ps, sizeof(user_ps));
}

gcn::MemReader reader(MemoryManager& m) {
    return [&m](u64 a, void* d, usize n) { return m.is_valid_address(a, n) && m.read_bytes(a, d, n); };
}

void test_translate_and_compile(MemoryManager& mem, id<MTLDevice> device) {
    u32 vs_user[16], ps_user[16];
    setup_sprite_scene(mem, vs_user, ps_user);
    gcn::Program vs, ps;
    std::string err;
    CHECK(gcn::load_program(reader(mem), kVsAddr, vs_user, 16, vs, err));
    CHECK(!vs.fetch_code.empty());
    CHECK(vs.fetch_code.size() == std::size(kSpriteFetch));
    CHECK(gcn::load_program(reader(mem), kPsAddr, ps_user, 12, ps, err));
    CHECK(vs.code.size() == std::size(kSpriteVs));
    CHECK(ps.code.size() == std::size(kSpritePs));

    gcn::TranslateOptions opts;
    gcn::TranslatedPipeline tp = gcn::translate_pipeline(vs, ps, nullptr, opts);
    if (!tp.ok) std::fprintf(stderr, "translate error: %s\n", tp.error.c_str());
    CHECK(tp.ok);
    CHECK(tp.ps_images.size() == 1);
    CHECK(tp.vs_param_count == 2);
    CHECK(tp.ps_writes_color);
    NSError* e = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:tp.msl.c_str()] options:nil error:&e];
    if (!lib) std::fprintf(stderr, "MSL compile error: %s\n%s\n", [[e localizedDescription] UTF8String], tp.msl.c_str());
    CHECK(lib != nil);
    CHECK([lib newFunctionWithName:@"gcn_vs_main"] != nil);
    CHECK([lib newFunctionWithName:@"gcn_ps_main"] != nil);

    // Resource scan resolves the fetch-shader V#s, constant buffers and T#/S#.
    gcn::ResourceScan vscan = gcn::scan_resources(vs, tp.vs_images, vs_user, 16, reader(mem), 4);
    gcn::ResourceScan pscan = gcn::scan_resources(ps, tp.ps_images, ps_user, 12, reader(mem), 0);
    if (!vscan.ok) std::fprintf(stderr, "vs scan: %s\n", vscan.error.c_str());
    CHECK(vscan.ok && pscan.ok);
    bool has_verts = false, has_table = false, has_proj = false;
    for (const auto& r : vscan.ranges) {
        if (r.addr <= kVerts && r.addr + r.size >= kVerts + 4 * 88) has_verts = true;
        if (r.addr <= kVTable && r.addr + r.size >= kVTable + 48) has_table = true;
        if (r.addr <= kProjConst + 272 && r.addr + r.size >= kProjConst + 336) has_proj = true;
    }
    CHECK(has_verts && has_table && has_proj);
    CHECK(pscan.images.size() == 1 && pscan.images[0].tsharp[1] == 0x00a00000 && pscan.images[0].sampler_known);
    const gcn::TextureDesc td = gcn::decode_tsharp(pscan.images[0].tsharp);
    CHECK(td.width == 16 && td.height == 16 && td.dfmt == 10 && td.nfmt == 0 && td.tiling_index == 8 && td.type == 9);

    // Clear pair (rect list remap).
    gcn::Program cvs = decode(kClearVs), cps = decode(kClearPs);
    gcn::TranslateOptions ro;
    ro.remap = gcn::TranslateOptions::VertexRemap::RectList;
    gcn::TranslatedPipeline ctp = gcn::translate_pipeline(cvs, cps, nullptr, ro);
    CHECK(ctp.ok);
    id<MTLLibrary> clib = [device newLibraryWithSource:[NSString stringWithUTF8String:ctp.msl.c_str()] options:nil error:&e];
    if (!clib) std::fprintf(stderr, "clear MSL compile error: %s\n", [[e localizedDescription] UTF8String]);
    CHECK(clib != nil);

    // Unsupported instructions are reported, never crash.
    const u32 bad[] = {0xd8340000, 0x00000000, 0xbf810000};   // DS op -> unsupported
    gcn::Program bp = decode(bad);
    gcn::TranslatedPipeline btp = gcn::translate_pipeline(bp, ps, nullptr, opts);
    CHECK(!btp.ok && !btp.error.empty());
}

void test_render(MemoryManager& mem) {
    MetalBackend metal;
    CHECK(metal.initialize());
    CHECK(metal.create_framebuffer(64, 64));
    u32 vs_user[16], ps_user[16];
    setup_sprite_scene(mem, vs_user, ps_user);
    const u32 clear_vs[16] = {};
    const u32 clear_ps[16] = {static_cast<u32>(kClearColor), 0x00100000, 1, 0x20077fac};
    write_words(mem, kClearVsAddr, kClearVs, std::size(kClearVs));
    write_words(mem, kClearPsAddr, kClearPs, std::size(kClearPs));
    const float clear_rgba[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    CHECK(mem.write_bytes(kClearColor, clear_rgba, 16));

    GcnPipelineCache cache(metal.raw_device(), &mem);
    CHECK(metal.begin_frame());
    CHECK(metal.ensure_render_encoder(true, ColorRGBA{0, 0, 0, 1}));
    void* enc = metal.active_render_encoder();
    std::vector<u32> ctx(0x400, 0);
    ctx[0x8E] = 0xF;
    for (u32 k = 0; k < 32; ++k) ctx[0x191 + k] = k;

    GcnDrawRequest clear{};
    clear.vs_addr = kClearVsAddr; clear.ps_addr = kClearPsAddr;
    clear.vs_rsrc2 = 0x00; clear.ps_rsrc2 = 0x08;
    clear.vs_user = clear_vs; clear.ps_user = clear_ps;
    clear.ctx_regs = ctx.data(); clear.ctx_reg_count = ctx.size();
    clear.prim = PrimitiveType::RectList; clear.count = 3;
    std::string why;
    const bool c_ok = cache.encode_draw(enc, MTLPixelFormatBGRA8Unorm, clear, &why);
    if (!c_ok) std::fprintf(stderr, "clear draw fallback: %s\n", why.c_str());
    CHECK(c_ok);

    ctx[0x1E0] = 0x61010504;   // Flappy's CB_BLEND0_CONTROL: src_alpha / 1-src_alpha
    GcnDrawRequest sprite{};
    sprite.vs_addr = kVsAddr; sprite.ps_addr = kPsAddr;
    sprite.vs_rsrc2 = 0x20; sprite.ps_rsrc2 = 0x18;
    sprite.vs_user = vs_user; sprite.ps_user = ps_user;
    sprite.ctx_regs = ctx.data(); sprite.ctx_reg_count = ctx.size();
    sprite.prim = PrimitiveType::QuadList; sprite.count = 4;
    const bool s_ok = cache.encode_draw(enc, MTLPixelFormatBGRA8Unorm, sprite, &why);
    if (!s_ok) std::fprintf(stderr, "sprite draw fallback: %s\n", why.c_str());
    CHECK(s_ok);
    CHECK(metal.submit_frame(true));

    u8 r, g, b, a;
    CHECK(metal.read_pixel(32, 32, r, g, b, a));
    std::printf("  center pixel (sprite): %u %u %u %u\n", r, g, b, a);
    // PS computes 2 * tex * vertex_color: tex (100,50,200,255)/255,
    // color (0.25,0.5,0.5,0.5) -> (50,50,200,255), alpha-blended over the
    // green clear with alpha 1.0.
    CHECK(std::abs(int(r) - 50) <= 2 && std::abs(int(g) - 50) <= 2 && std::abs(int(b) - 200) <= 2);
    CHECK(metal.read_pixel(2, 2, r, g, b, a));
    std::printf("  corner pixel (clear): %u %u %u %u\n", r, g, b, a);
    CHECK(r == 0 && g == 255 && b == 0);
    CHECK(cache.stats().translated_draws == 2);
    CHECK(cache.stats().pipelines_compiled == 2);
}

void test_texture_detile() {
    // 16x8 micro-tiled RGBA8: element (x,y) stored at tile*64 + interleave.
    gcn::TextureDesc td;
    td.width = 16; td.height = 8; td.pitch = 16; td.dfmt = 10; td.nfmt = 0; td.tiling_index = 13;
    std::vector<u8> src(16 * 8 * 4, 0);
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 16; ++x) {
            const u32 idx = (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
            const u32 e = (x / 8) * 64 + idx;
            src[e * 4] = static_cast<u8>(x); src[e * 4 + 1] = static_cast<u8>(y); src[e * 4 + 3] = 255;
        }
    std::vector<u8> out;
    std::string err;
    CHECK(gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
    bool ok = true;
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 16; ++x)
            ok = ok && out[(y * 16 + x) * 4] == x && out[(y * 16 + x) * 4 + 1] == y;
    CHECK(ok);
    td.tiling_index = 14;
    CHECK(!gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
}

} // namespace

int main() {
    @autoreleasepool {
        test_decode();
        test_texture_detile();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::printf("GCN shader test: no Metal device, skipping GPU parts\n");
            return g_failures ? 1 : 0;
        }
        MemoryManager mem;
        if (!mem.initialize()) {
            std::fprintf(stderr, "memory init failed\n");
            return 1;
        }
        test_translate_and_compile(mem, device);
        test_render(mem);
    }
    if (g_failures) {
        std::fprintf(stderr, "GCN shader test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("GCN shader test: all checks passed\n");
    return 0;
}
