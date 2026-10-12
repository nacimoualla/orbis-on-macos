cmake_minimum_required(VERSION 3.25)
project(ps4-to-macos LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Find macOS frameworks for Metal, Display Window & Audio
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
find_library(COREAUDIO_FRAMEWORK CoreAudio REQUIRED)

# Core library
add_library(ps4_core
    src/core/elf_loader.cpp
    src/core/memory.cpp
    src/core/syscall_table.cpp
    src/core/dynamic_linker.cpp
    src/core/vfs.cpp
    src/cpu/executor.cpp
    src/cpu/jit_compiler.cpp
    src/gpu/metal_backend.mm
    src/gpu/display_window.mm
    src/gpu/gpu_ring_buffer.cpp
    src/gpu/pm4_translator.mm
    src/audio/audio_manager.cpp
    src/audio/coreaudio_backend.mm
    src/hle/nid_registry.cpp
    src/hle/thread.cpp
)
target_include_directories(ps4_core PUBLIC include)
target_link_libraries(ps4_core PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${APPKIT_FRAMEWORK}
    ${AUDIOTOOLBOX_FRAMEWORK}
    ${COREAUDIO_FRAMEWORK}
)
set_source_files_properties(src/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
set_source_files_properties(src/gpu/display_window.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
set_source_files_properties(src/gpu/pm4_translator.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
set_source_files_properties(src/audio/coreaudio_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Main executable runner
add_executable(ps4-runner
    src/main.cpp
)
target_link_libraries(ps4-runner PRIVATE ps4_core)

# Automated testing
enable_testing()

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE ps4_core)
add_test(NAME Phase1_Types COMMAND test_phase1)

add_executable(test_phase2 tests/test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE ps4_core)
add_test(NAME Phase2_ElfLoader COMMAND test_phase2)

add_executable(test_phase3 tests/test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE ps4_core)
add_test(NAME Phase3_VirtualMemory COMMAND test_phase3)

add_executable(test_phase4 tests/test_phase4.cpp)
target_link_libraries(test_phase4 PRIVATE ps4_core)
add_test(NAME Phase4_SyscallTable COMMAND test_phase4)

add_executable(test_phase5 tests/test_phase5.cpp)
target_link_libraries(test_phase5 PRIVATE ps4_core)
add_test(NAME Phase5_CpuExecutor COMMAND test_phase5)

add_executable(test_phase6 tests/test_phase6.cpp)
target_link_libraries(test_phase6 PRIVATE ps4_core)
add_test(NAME Phase6_MetalBackend COMMAND test_phase6)

add_executable(test_phase7 tests/test_phase7.cpp)
target_link_libraries(test_phase7 PRIVATE ps4_core)
add_test(NAME Phase7_DynamicLinker COMMAND test_phase7)

add_executable(test_phase8 tests/test_phase8.cpp)
target_link_libraries(test_phase8 PRIVATE ps4_core)
add_test(NAME Phase8_DisplayWindow COMMAND test_phase8)

add_executable(test_phase9 tests/test_phase9.cpp)
target_link_libraries(test_phase9 PRIVATE ps4_core)
add_test(NAME Phase9_Threading COMMAND test_phase9)

add_executable(test_phase10 tests/test_phase10.cpp)
target_link_libraries(test_phase10 PRIVATE ps4_core)
add_test(NAME Phase10_GpuCommandProcessor COMMAND test_phase10)

add_executable(test_phase11 tests/test_phase11.cpp)
target_link_libraries(test_phase11 PRIVATE ps4_core)
add_test(NAME Phase11_AudioOut COMMAND test_phase11)

add_executable(test_phase12 tests/test_phase12.cpp)
target_link_libraries(test_phase12 PRIVATE ps4_core)
add_test(NAME Phase12_VirtualFileSystem COMMAND test_phase12)

add_executable(test_phase13 tests/test_phase13.cpp)
target_link_libraries(test_phase13 PRIVATE ps4_core)
add_test(NAME Phase13_JitCompiler COMMAND test_phase13)

add_executable(test_phase14 tests/test_phase14.cpp)
target_link_libraries(test_phase14 PRIVATE ps4_core)
add_test(NAME Phase14_MeshPipeline COMMAND test_phase14)

add_executable(test_phase15 tests/test_phase15.cpp)
target_link_libraries(test_phase15 PRIVATE ps4_core)
add_test(NAME Phase15_JitExecutorIntegration COMMAND test_phase15)



