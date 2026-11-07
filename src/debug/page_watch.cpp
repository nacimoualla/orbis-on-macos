#include "ps4/debug/page_watch.hpp"
#include "ps4/memory.hpp"
#include "ps4/cpu_context.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
namespace ps4::debug {
thread_local const CpuContext* watch_context = nullptr;
namespace {
struct Page { uintptr_t host; int protection; };
std::vector<Page> pages;
uintptr_t base=0, first=0, limit=0;
size_t page_size=0;
int events[2]={-1,-1};
struct sigaction previous_bus{}, previous_segv{};
std::atomic<bool> running{false};
std::thread timer;
struct Hit { u64 address, rip, old_value; int tid; };
void forward(int sig, siginfo_t* info, void* context) {
    const auto& prev=sig==SIGBUS?previous_bus:previous_segv;
    if (prev.sa_handler==SIG_IGN) return;
    if (prev.sa_handler==SIG_DFL) { sigaction(sig,&prev,nullptr); raise(sig); return; }
    if (prev.sa_flags&SA_SIGINFO) prev.sa_sigaction(sig,info,context);
    else prev.sa_handler(sig);
}
void fault(int sig, siginfo_t* info, void* context) {
    const uintptr_t addr=reinterpret_cast<uintptr_t>(info->si_addr);
    if (addr<first || addr>=limit) { forward(sig,info,context); return; }
    const size_t index=(addr-first)/page_size;
    // No allocation, stdio, locks, or symbol lookup in signal context. Log to
    // a nonblocking pipe; the timer owns formatting and the after-store sample.
    Hit hit{addr-base, watch_context?watch_context->rip:t_guest_rip,0,t_guest_tid};
    if (hit.tid==0) hit.tid=-1;
    const size_t n=std::min<size_t>(8,limit-addr);
    const volatile u8* p=reinterpret_cast<const volatile u8*>(addr);
    for (size_t i=0;i<n;++i) hit.old_value |= u64(p[i])<<(8*i);
    mprotect(reinterpret_cast<void*>(pages[index].host),page_size,pages[index].protection);
    (void)!write(events[1],&hit,sizeof(hit));
}
}
void start_page_watch(MemoryManager& memory) {
    const char* spec=getenv("PS4_PAGE_WATCH"); if (!spec || running) return;
    char* end=nullptr; u64 addr=strtoull(spec,&end,0), size=8;
    if (*end==',') { char* tail=nullptr; size=strtoull(end+1,&tail,0); end=tail; }
    if (*end || !size || !memory.is_valid_address(addr,size)) { fprintf(stderr,"[page-watch] invalid range: %s\n",spec); return; }
    page_size=getpagesize(); base=reinterpret_cast<uintptr_t>(memory.base_address());
    first=(base+addr)&~(page_size-1); limit=(base+addr+size+page_size-1)&~(page_size-1);
    for (uintptr_t p=first;p<limit;p+=page_size) {
        mach_vm_address_t region=p; mach_vm_size_t length=0; vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64; mach_port_t object=MACH_PORT_NULL;
        kern_return_t rc=mach_vm_region(mach_task_self(),&region,&length,VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info),&count,&object);
        if (object!=MACH_PORT_NULL) mach_port_deallocate(mach_task_self(),object);
        if (rc!=KERN_SUCCESS || region>p || !(info.protection&VM_PROT_WRITE) || !(info.protection&VM_PROT_READ)) {
            fprintf(stderr,"[page-watch] range must already be mapped read/write; page 0x%llx\n",(unsigned long long)(p-base)); pages.clear(); return;
        }
        pages.push_back({p,info.protection});
    }
    if (pipe(events)) { perror("page-watch pipe"); pages.clear(); return; }
    for (int fd:events) { fcntl(fd,F_SETFL,O_NONBLOCK); fcntl(fd,F_SETFD,FD_CLOEXEC); }
    struct sigaction action{}; action.sa_sigaction=fault; action.sa_flags=SA_SIGINFO; sigemptyset(&action.sa_mask);
    sigaction(SIGBUS,&action,&previous_bus); sigaction(SIGSEGV,&action,&previous_segv);
    running=true;
    timer=std::thread([] {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            Hit hit;
            while (read(events[0],&hit,sizeof(hit))==sizeof(hit)) {
                u64 after=0; const size_t n=std::min<size_t>(8,limit-(base+hit.address));
                memcpy(&after,reinterpret_cast<void*>(base+hit.address),n);
                fprintf(stderr,"[page-watch] guest=0x%llx old=0x%llx value=0x%llx TID=%d RIP=0x%llx (instruction RIP, deferred value)\n",
                    (unsigned long long)hit.address,(unsigned long long)hit.old_value,(unsigned long long)after,hit.tid,(unsigned long long)hit.rip);
            }
            for (const auto& p:pages) mprotect(reinterpret_cast<void*>(p.host),page_size,p.protection&~PROT_WRITE);
        }
    });
    for (const auto& p:pages) mprotect(reinterpret_cast<void*>(p.host),page_size,p.protection&~PROT_WRITE);
    fprintf(stderr,"[page-watch] armed 0x%llx+0x%llx (%zu host pages, 10ms rearm)\n",(unsigned long long)addr,(unsigned long long)size,pages.size());
}
void stop_page_watch() {
    if (!running.exchange(false)) return;
    timer.join();
    for (const auto& p:pages) mprotect(reinterpret_cast<void*>(p.host),page_size,p.protection);
    sigaction(SIGBUS,&previous_bus,nullptr); sigaction(SIGSEGV,&previous_segv,nullptr);
    close(events[0]);close(events[1]);pages.clear();
}
}
