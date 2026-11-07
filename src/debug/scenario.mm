#import <Foundation/Foundation.h>
#include "ps4/debug/scenario.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/pm4_translator.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <deque>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>

namespace ps4::debug {
namespace {
using Clock = std::chrono::steady_clock;
struct Rule { std::string name, after, type, value, stage; bool fired=false; };
std::vector<Rule> rules;
std::set<std::string> reached;
std::mutex mutex;
std::atomic<bool> active{false}, stopping{false}, quit{false}, paced{false};
std::atomic<u64> frame{0}, flips{0};
std::thread control;
Clock::time_point epoch, next_tick, last_metrics;
NidRegistry* registry=nullptr;
std::deque<std::string> capture_queue;
std::string current_capture, fast_until, pending_input;
u32 desired_buttons=0;
std::set<std::string> discovered;
bool discovery=false;
int input_fd=-1;
double elapsed() { return std::chrono::duration<double>(Clock::now()-epoch).count(); }
std::string str(id value) { return [value isKindOfClass:[NSString class]] ? std::string([value UTF8String]) : ""; }
void emit(const char* prefix, NSDictionary* fields) {
    NSMutableDictionary* event=[fields mutableCopy];
    event[@"frame"]=@(frame.load()); event[@"flip"]=@(registry?registry->get_flip_count():0); event[@"t"]=@(elapsed());
    NSData* json=[NSJSONSerialization dataWithJSONObject:event options:0 error:nil];
    if (json) fprintf(stderr,"%s %.*s\n",prefix,(int)json.length,(const char*)json.bytes);
}
bool eligible(const Rule& r) { return !r.fired && (r.after.empty() || reached.count(r.after)); }
void mark(Rule& r) {
    r.fired=true; reached.insert(r.name);
    if (r.name==fast_until) paced=true;
    emit("[milestone]",@{@"name":@(r.name.c_str()),@"source":@(r.type.c_str())});
}
void match(const std::string& type, const std::string& value, const std::string& stage="") {
    if (!active.load()) return;
    std::lock_guard lock(mutex);
    // Eligibility is evaluated before this event: one event cannot also
    // satisfy a descendant milestone that it has just enabled.
    std::vector<Rule*> candidates;
    for (auto& r:rules) if (eligible(r)) candidates.push_back(&r);
    for (Rule* rule:candidates) {
        auto& r=*rule;
        if (r.type!=type || (!r.stage.empty() && r.stage!=stage)) continue;
        bool equal=r.value==value;
        if (type=="shader" || type=="texture") equal=strtoull(r.value.c_str(),nullptr,0)==strtoull(value.c_str(),nullptr,0);
        if (equal) mark(r);
    }
}
void command(NSDictionary* cmd) {
    std::string op=str(cmd[@"op"]), id=str(cmd[@"id"]);
    if (op=="buttons") {
        u32 buttons=[cmd[@"buttons"] unsignedIntValue];
        {
            std::lock_guard lock(mutex);
            pending_input=id; desired_buttons=buttons;
        }
        registry->set_pad_buttons(buttons);
        {
            std::lock_guard lock(mutex);
            emit("[scenario]",@{@"type":@"input",@"id":@(id.c_str()),@"buttons":@(buttons)});
        }
    } else if (op=="log") {
        match("log",str(cmd[@"name"]));
    } else if (op=="capture") {
        std::lock_guard lock(mutex); capture_queue.push_back(id);
    } else if (op=="stop") {
        registry->set_pad_buttons(0); quit=true;
        std::lock_guard lock(mutex); emit("[scenario]",@{@"type":@"stop_requested"});
    } else {
        std::lock_guard lock(mutex); emit("[scenario]",@{@"type":@"error",@"message":@"unknown control command"});
    }
}
}
bool scenario_enabled() { return active.load(); }
bool scenario_should_stop() { return quit.load(); }
void scenario_start(NidRegistry& target) {
    const char* path=getenv("PS4_SCENARIO"); if (!path) return;
    @autoreleasepool {
        NSError* error=nil;
        NSData* bytes=[NSData dataWithContentsOfFile:@(path)];
        id config=bytes ? [NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error] : nil;
        if (![config isKindOfClass:[NSDictionary class]]) { fprintf(stderr,"[scenario-error] invalid recipe %s\n",path); std::exit(2); }
        const char* fifo=getenv("PS4_SCENARIO_CONTROL");
        input_fd=fifo ? open(fifo,O_RDONLY|O_NONBLOCK|O_CLOEXEC) : -1;
        if (input_fd<0) { fprintf(stderr,"[scenario-error] cannot open private control FIFO\n"); std::exit(2); }
        registry=&target; epoch=Clock::now(); last_metrics=epoch;
        discovery=getenv("PS4_SCENARIO_DISCOVERY") != nullptr;
        fast_until=str(config[@"fast_forward_until"]);
        const char* ff=getenv("PS4_SCENARIO_FAST_FORWARD");
        bool fast=ff && std::string(ff)=="1" && !fast_until.empty();
        paced=!fast;
        for (NSDictionary* definition in config[@"milestones"]) {
            NSDictionary* source=definition[@"source"];
            std::string type=str(source[@"type"]);
            std::string value=type=="input" ? str(source[@"id"]) : type=="nid" ? str(source[@"name"]) : type=="shader" ? str(source[@"hash"]) : type=="texture" ? str(source[@"address"]) : str(definition[@"name"]);
            rules.push_back({str(definition[@"name"]),str(definition[@"after"]),type,value,str(source[@"stage"])});
        }
        // Wrap only explicitly configured HLE names. Preserve all handler logic
        // and the registry's public interface; real PRX implementations bypass
        // these HLE wrappers and must use another observable milestone source.
        std::set<std::string> wanted;
        for (const auto& r:rules) if (r.type=="nid") wanted.insert(r.value);
        // Confirm delivery through the same public HLE symbols that games use.
        // No changes to the HLE implementations or CPU execution paths.
        const std::set<std::string> pad_names={"scePadRead", "scePadReadState", "unknown_b445383a"};
        wanted.insert(pad_names.begin(),pad_names.end());
        for (const auto& name:wanted) {
            const HleSymbol* symbol=target.find_by_name(name);
            if (!symbol) {
                if (pad_names.count(name)) continue;
                fprintf(stderr,"[scenario-error] unknown HLE name: %s\n",name.c_str()); std::exit(2);
            }
            HleSymbol copy=*symbol;
            const bool pad=pad_names.count(name)!=0;
            target.register_symbol(copy.nid,copy.name,copy.module_name,[handler=copy.handler,name=copy.name,pad](CpuContext& c,MemoryManager& m) {
                match("nid",name);
                const GuestAddr data=c.rsi;
                handler(c,m);
                if (pad && active.load() && data>0x1000 && m.is_valid_address(data,4)) {
                    u32 observed=0;
                    if (m.read_value(data,observed)) {
                        std::string sampled;
                        {
                            std::lock_guard lock(mutex);
                            if (!pending_input.empty() && observed==desired_buttons) {
                                sampled=std::move(pending_input); pending_input.clear();
                                emit("[scenario]",@{@"type":@"input_sampled",@"id":@(sampled.c_str()),@"buttons":@(observed)});
                            }
                        }
                        if (!sampled.empty()) match("input",sampled);
                    }
                }
            });
        }
        setvbuf(stdout,nullptr,_IOLBF,0); setvbuf(stderr,nullptr,_IOLBF,0);
        active=true;
        { std::lock_guard lock(mutex); emit("[scenario]",@{@"type":@"ready",@"fast_forward":@(fast)}); }
        for (const auto& r:rules) if (r.type=="start") match("start",r.value);
        control=std::thread([] {
            std::string pending;
            while (!stopping.load()) {
                @autoreleasepool {
                    char buffer[4096]; ssize_t n=read(input_fd,buffer,sizeof(buffer));
                    if (n>0) pending.append(buffer,n);
                    if (pending.size()>65536) { pending.clear(); fprintf(stderr,"[scenario-error] oversized control message\n"); }
                    size_t end;
                    while ((end=pending.find('\n'))!=std::string::npos) {
                        std::string line=pending.substr(0,end); pending.erase(0,end+1);
                        NSData* data=[NSData dataWithBytes:line.data() length:line.size()];
                        id cmd=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                        if ([cmd isKindOfClass:[NSDictionary class]]) command(cmd);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
}
void scenario_stop() {
    if (!active.load()) return;
    stopping=true; if (control.joinable()) control.join(); close(input_fd);
    registry->set_pad_buttons(0);
    { std::lock_guard lock(mutex);
        auto* gpu=registry->gpu_translator();
        emit("[scenario]",@{@"type":@"metrics",@"frames":@(registry->get_flip_count()),@"unsupported_packets":@(gpu?gpu->unsupported_packet_count():0)});
        emit("[scenario]",@{@"type":@"finished"}); }
    active=false;
}
void scenario_shader(const char* stage,u64 hash) {
    if (!active.load()) return;
    char text[32]; snprintf(text,sizeof(text),"0x%016llx",(unsigned long long)hash);
    match("shader",text,stage);
    if (discovery) {
        std::lock_guard lock(mutex);
        if (discovered.size()<2048 && discovered.insert(std::string(stage)+text).second)
            emit("[scenario]",@{@"type":@"shader_seen",@"hash":@(text),@"stage":@(stage)});
    }
}
void scenario_texture(u64 address) {
    if (!active.load()) return;
    char text[32]; snprintf(text,sizeof(text),"0x%llx",(unsigned long long)address);
    match("texture",text);
    if (discovery) {
        std::lock_guard lock(mutex);
        if (discovered.size()<2048 && discovered.insert(std::string("T")+text).second)
            emit("[scenario]",@{@"type":@"texture_seen",@"address":@(text)});
    }
}
void scenario_begin_frame(u64 index) { if (active.load()) frame=index; }
void scenario_frame(u64 index) {
    if (!active.load()) return;
    frame=index; ++flips;
    auto now=Clock::now();
    {
        std::lock_guard lock(mutex);
        if (now-last_metrics>=std::chrono::seconds(1)) {
            auto* gpu=registry->gpu_translator();
            emit("[scenario]",@{@"type":@"metrics",@"frames":@(registry->get_flip_count()),@"unsupported_packets":@(gpu?gpu->unsupported_packet_count():0)});
            last_metrics=now;
        }
    }
    if (paced.load()) {
        const auto interval=std::chrono::nanoseconds(16666667);
        if (next_tick.time_since_epoch().count()==0 || now>next_tick+interval) next_tick=now;
        else std::this_thread::sleep_until(next_tick);
        next_tick+=interval;
    }
}
bool scenario_capture_requested() {
    if (!active.load()) return false;
    std::lock_guard lock(mutex);
    if (capture_queue.empty()) return false;
    current_capture=capture_queue.front(); capture_queue.pop_front(); return true;
}
void scenario_capture_completed(const char* path) {
    if (!active.load()) return;
    std::lock_guard lock(mutex);
    if (current_capture.empty()) return;
    emit("[scenario]",@{@"type":@"capture",@"id":@(current_capture.c_str()),@"path":@(path)});
    current_capture.clear();
}
}
