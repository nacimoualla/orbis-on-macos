#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#include "ps4/debug/capture.hpp"
#include "ps4/gpu_state.hpp"
#include "ps4/memory.hpp"
#include <filesystem>
#include <spawn.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <sys/wait.h>
#include <cerrno>
extern char** environ;
namespace ps4::debug {
namespace {
NSMutableDictionary* report;
NSMutableArray* operations;
NSMutableArray* passes;
NSMutableDictionary* operation;
NSArray* attachments;
NSString* directory;
bool xcode = false;
std::function<void()> dump_callback;
u64 selected(const char* name) { const char* s = getenv(name); return s ? strtoull(s, nullptr, 0) : 0; }
NSArray* words(const u32* p, usize n) {
    NSMutableArray* a = [NSMutableArray array];
    for (usize i = 0; i < n; ++i) [a addObject:@(p[i])];
    return a;
}
NSString* hex(u64 n) { return [NSString stringWithFormat:@"0x%016llx", (unsigned long long)n]; }
void save_json(id object, NSString* path) {
    NSError* e = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:&e];
    if (!data || ![data writeToFile:path options:NSDataWritingAtomic error:&e])
        fprintf(stderr, "[debug] JSON write failed: %s\n", e.localizedDescription.UTF8String);
}
}
bool capture_active() { return report != nil; }
void capture_begin(u64 frame, void* device) {
    const char* trigger=getenv("PS4_AT_FRAME"); char* tail=nullptr;
    bool dump=trigger && strtoull(trigger,&tail,0)==frame && tail && strcmp(tail,":dump")==0;
    if (!dump && frame != selected("PS4_GPU_CAPTURE_FRAME") && frame != selected("PS4_GPU_CAPTURE_XCODE")) return;
    const char* root = getenv("PS4_DEBUG_DIR");
    directory = [NSString stringWithFormat:@"%s/frame_%llu", root ? root : "debug-captures", (unsigned long long)frame];
    NSError* error = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error]) {
        fprintf(stderr, "[debug] mkdir failed: %s\n", error.localizedDescription.UTF8String); return;
    }
    if (dump || frame == selected("PS4_GPU_CAPTURE_FRAME")) {
        operations = [NSMutableArray array]; passes = [NSMutableArray array];
        report = [@{@"frame":@(frame), @"operations":operations, @"passes":passes,
                    @"hash_algorithm":@"GCN FNV-1a code and fetch code"} mutableCopy];
    }
    if (frame == selected("PS4_GPU_CAPTURE_XCODE")) {
        MTLCaptureDescriptor* d = [MTLCaptureDescriptor new];
        d.captureObject = (__bridge id<MTLDevice>)device;
        d.destination = MTLCaptureDestinationGPUTraceDocument;
        d.outputURL = [NSURL fileURLWithPath:[directory stringByAppendingPathComponent:@"frame.gputrace"]];
        xcode = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&error];
        if (!xcode) fprintf(stderr, "[debug] Metal capture failed: %s (set MTL_CAPTURE_ENABLED=1)\n", error.localizedDescription.UTF8String);
    }
}
void capture_operation(const char* kind, u32 count, const u32* regs, MemoryManager* memory,
                       u32 width, u32 height, u32 x, u32 y, u32 z) {
    if (!report) return;
    const u32* ctx = regs + 0xA000;
    auto s = decode_render_state(ctx, ctx_reg::kCount, width, height);
    NSMutableArray* targets = [NSMutableArray array];
    for (u32 i=0; i<8; ++i) {
        auto c = s.color[i];
        u32 blend=ctx[ctx_reg::kCbBlend0Control+i]; bool separate=(blend>>29)&1;
        [targets addObject:@{@"slot":@(i), @"enabled":@(c.enabled), @"key":hex(c.base), @"width":@(c.width),
          @"height":@(c.height), @"pitch":@(c.pitch), @"format":@(c.format), @"number_type":@(c.num_format),
          @"component_swap":@(c.comp_swap), @"write_mask":@(c.write_mask), @"blend_raw":@(blend), @"blend":@{@"enabled":@((blend>>30)&1),@"separate_alpha":@(separate),
          @"src_rgb":@(blend&31),@"dst_rgb":@((blend>>8)&31),@"rgb_op":@((blend>>5)&7),
          @"src_alpha":@(separate?((blend>>16)&31):(blend&31)),@"dst_alpha":@(separate?((blend>>24)&31):((blend>>8)&31)),
          @"alpha_op":@(separate?((blend>>21)&7):((blend>>5)&7))}}];
    }
    operation = [@{@"order":@(operations.count), @"pass":[NSNull null], @"kind":@(kind), @"count":@(count), @"grid":@[@(x),@(y),@(z)],
        @"targets":targets, @"scissor":@[@(s.scissor.x),@(s.scissor.y),@(s.scissor.width),@(s.scissor.height)],
        @"viewport":@[@(s.viewport.x),@(s.viewport.y),@(s.viewport.width),@(s.viewport.height),@(s.viewport.znear),@(s.viewport.zfar)],
        @"depth_stencil":@{@"test":@(s.ds.depth_test),@"write":@(s.ds.depth_write),@"func":@(static_cast<u32>(s.ds.depth_func)),
          @"stencil_test":@(s.ds.stencil_test),@"depth_clear":@(s.ds.depth_clear),@"stencil_clear":@(s.ds.stencil_clear)},
        @"depth_stencil_raw":@{@"control":@(ctx[0x200]), @"stencil_control":@(ctx[0x10b]), @"front_ref_mask":@(ctx[0x10c]),
           @"back_ref_mask":@(ctx[0x10d]), @"render_control":@(ctx[0]), @"depth_key":hex(s.depth.base)},
        @"context_raw":words(ctx, ctx_reg::kCount), @"stages":[NSMutableDictionary dictionary]} mutableCopy];
    [operations addObject:operation];
    // Populate hashes even if translation later fails or falls back. Resource
    // descriptors below are the actual scalar scan used by the translated path.
    if (memory) {
        gcn::MemReader read = [memory](u64 a, void* p, usize n) { return memory->read_bytes(a,p,n); };
        const bool cs = strcmp(kind,"dispatch")==0;
        for (unsigned i=cs?2:0; i<(cs?3:2); ++i) {
            const u32 lo[] = {0x48,8,0x20c}, hi[] = {0x49,9,0x20d}, user[] = {0x4c,0xc,0x240}, rsrc[] = {0x4b,0xb,0x213};
            const u32* sh = regs+0x2c00;
            u64 addr = (u64(sh[hi[i]])<<40) | (u64(sh[lo[i]])<<8);
            gcn::Program p; std::string err;
            bool ok = addr && gcn::load_program(read,addr,sh+user[i],std::min(16u,(sh[rsrc[i]]>>1)&31),p,err);
            NSString* stage = @[@"VS",@"PS",@"CS"][i];
            operation[@"stages"][stage] = [@{@"address":hex(addr), @"hash":ok?hex(p.hash):(id)[NSNull null],
                                              @"load_error":@(err.c_str())} mutableCopy];
            if (ok) {
                std::vector<gcn::ImageSlot> slots;
                for (const auto& instruction : p.insts) {
                    if (instruction.enc != gcn::Encoding::MIMG) continue;
                    bool storage=instruction.op==8 || instruction.op==9;
                    bool sampled=instruction.op>=0x20;
                    gcn::ImageSlot slot{instruction.srsrc,sampled?instruction.ssamp:~0u,sampled,storage};
                    bool seen=false;
                    for (const auto& old : slots)
                        if (old.srsrc==slot.srsrc && old.ssamp==slot.ssamp && old.storage==slot.storage) seen=true;
                    if (!seen) slots.push_back(slot);
                }
                auto scan=gcn::scan_resources(p,slots,sh+user[i],std::min(16u,(sh[rsrc[i]]>>1)&31),read,count);
                capture_resources(stage.UTF8String,p.hash,scan,slots);
            }
        }
    }
}
void capture_resources(const char* stage, u64 hash, const gcn::ResourceScan& scan, const std::vector<gcn::ImageSlot>& slots) {
    if (!report || !operation) return;
    NSMutableArray* images = [NSMutableArray array];
    for (usize i=0;i<scan.images.size();++i) {
        auto& d=scan.images[i]; auto t=gcn::decode_tsharp(d.tsharp); auto s=gcn::decode_ssharp(d.ssharp);
        [images addObject:@{@"T_raw":words(d.tsharp,8), @"S_raw":words(d.ssharp,4), @"sampler_known":@(d.sampler_known),
         @"srsrc":@(i<slots.size()?slots[i].srsrc:0), @"ssamp":@(i<slots.size()?slots[i].ssamp:0),
         @"T":@{@"base":hex(t.base),@"width":@(t.width),@"height":@(t.height),@"depth":@(t.depth),@"pitch":@(t.pitch),
             @"dfmt":@(t.dfmt),@"nfmt":@(t.nfmt),@"dst_sel":words(t.dst_sel,4),@"tiling_index":@(t.tiling_index),
             @"type":@(t.type),@"base_level":@(t.base_level),@"last_level":@(t.last_level)},
         @"S":@{@"clamp_x":@(s.clamp_x),@"clamp_y":@(s.clamp_y),@"clamp_z":@(s.clamp_z),@"mag_filter":@(s.xy_mag_filter),
             @"min_filter":@(s.xy_min_filter),@"mip_filter":@(s.mip_filter),@"unnormalized":@(s.force_unnormalized)}}];
    }
    NSMutableDictionary* entry=operation[@"stages"][@(stage)];
    if (!entry) entry=[NSMutableDictionary dictionary];
    [entry addEntriesFromDictionary:@{@"hash":hex(hash),@"scan_ok":@(scan.ok),@"scan_error":@(scan.error.c_str()),@"images":images}];
    operation[@"stages"][@(stage)]=entry;
}
void capture_render_pass(void* descriptor, const u64* keys) {
    if (!report) return;
    MTLRenderPassDescriptor* pass=(__bridge MTLRenderPassDescriptor*)descriptor;
    NSMutableArray* a=[NSMutableArray array];
    for (u32 i=0;i<8;++i) {
        id<MTLTexture> t=pass.colorAttachments[i].texture;
        if (t) [a addObject:@{@"slot":@(i),@"key":hex(keys[i]),@"width":@(t.width),@"height":@(t.height),
                               @"metal_format":@(t.pixelFormat)}];
    }
    attachments=a;
}
void capture_bound() {
    if (!report || !operation) return;
    operation[@"pass"]=@(passes.count);
    operation[@"metal_targets"]=attachments ? attachments : @[];
}
void capture_pass(void* buffer, const char* kind) {
    if (!report) return;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)buffer;
    [cb commit]; [cb waitUntilCompleted];
    [passes addObject:@{@"index":@(passes.count),@"kind":@(kind),@"gpu_ms":@((cb.GPUEndTime-cb.GPUStartTime)*1000),
                        @"targets":attachments ? attachments : @[], @"status":@(cb.status),@"error":cb.error ? cb.error.localizedDescription : @""}];
    attachments=nil;
}
void capture_texture(u64 key, void* texture) {
    if (!report || !texture) return;
    id<MTLTexture> t = (__bridge id<MTLTexture>)texture;
    NSUInteger channels=0; bool bgra=false;
    switch (t.pixelFormat) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm: case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint: channels=1; break;
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint: channels=2; break;
    case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB: channels=4; bgra=true; break;
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint: channels=4; break;
    default: return;
    }
    std::vector<u8> pixels(t.width*t.height*channels);
    [t getBytes:pixels.data() bytesPerRow:t.width*channels fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];
    NSBitmapImageRep* image = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:nullptr pixelsWide:t.width pixelsHigh:t.height
        bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace bytesPerRow:t.width*4 bitsPerPixel:32];
    for (NSUInteger i=0;i<t.width*t.height;++i) {
        u8* p=image.bitmapData+4*i; const u8* q=pixels.data()+i*channels;
        p[0]=q[bgra?2:0]; p[1]=channels>1?q[1]:q[0]; p[2]=channels==4?q[bgra?0:2]:0; p[3]=channels==4?q[3]:255;
    }
    NSString* path=[directory stringByAppendingPathComponent:[NSString stringWithFormat:@"target_%016llx.png",(unsigned long long)key]];
    if (![[image representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:path atomically:YES])
        fprintf(stderr,"[debug] PNG write failed: %s\n",path.UTF8String);
}
void capture_end() {
    if (report) { save_json(report,[directory stringByAppendingPathComponent:@"report.json"]); fprintf(stderr,"[debug] captured %s\n",directory.UTF8String); }
    if (xcode) [[MTLCaptureManager sharedCaptureManager] stopCapture];
    report=nil; attachments=nil; operation=nil; operations=nil; passes=nil; xcode=false;
}
void set_frame_dump(std::function<void()> callback) { dump_callback=std::move(callback); }
void at_frame(u64 frame) {
    const char* spec=getenv("PS4_AT_FRAME"); if (!spec) return;
    char* end=nullptr; u64 n=strtoull(spec,&end,0); if (n!=frame || *end!=':') return;
    static std::atomic<bool> fired{false}; if (fired.exchange(true)) return;
    const char* action=end+1; const char* root=getenv("PS4_DEBUG_DIR");
    std::filesystem::path dir=root?root:"debug-captures"; std::error_code ec; std::filesystem::create_directories(dir,ec);
    if (strcmp(action,"exit")==0) { fprintf(stderr,"[debug] exit at frame %llu\n",(unsigned long long)frame); fflush(nullptr); _Exit(0); }
    if (strcmp(action,"sample")==0) {
        std::string pid=std::to_string(getpid()), file=(dir/("sample_frame_"+std::to_string(frame)+".txt")).string();
        char* args[]={const_cast<char*>("/usr/bin/sample"),pid.data(),const_cast<char*>("8"),const_cast<char*>("-file"),file.data(),nullptr};
        pid_t child; int rc=posix_spawn(&child,args[0],nullptr,nullptr,args,environ);
        if (rc==0) std::thread([child] { while (waitpid(child,nullptr,0)<0 && errno==EINTR) {} }).detach();
        fprintf(stderr,"[debug] sample frame %llu: %s (spawn=%d)\n",(unsigned long long)frame,file.c_str(),rc);
    } else if (strcmp(action,"dump")==0) {
        if (dump_callback) dump_callback();
        // Full GPU report for this frame is armed in capture_begin too.
        fprintf(stderr,"[debug] dump at frame %llu\n",(unsigned long long)frame);
    } else fprintf(stderr,"[debug] unknown PS4_AT_FRAME action: %s\n",action);
}
}
