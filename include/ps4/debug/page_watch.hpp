#pragma once
namespace ps4 { class MemoryManager; struct CpuContext; }
namespace ps4::debug {
void start_page_watch(MemoryManager& memory);
void stop_page_watch();
extern thread_local const CpuContext* watch_context;
}
