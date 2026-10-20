cmake_minimum_required(VERSION 3.25)
project(ps4-to-macos LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build. Without this a bare `cmake -B build ..` leaves
# CMAKE_BUILD_TYPE empty, and CMake's empty-config fallback adds no -O flag at
# all -- every translation unit compiles at -O0 with asserts live. That is what
# this tree was configured with (`CXX_FLAGS = -g -std=c++20 -arch arm64 ...`),
# and it cost roughly 2 us per guest instruction: the interpreter measured
# ~467 K guest instr/s, so main covered only 28 M instructions in a 60 s run and
# never reached first frame. Release resolves to -O3 -DNDEBUG here.
# Investigate build type: -O2 -g but *without* NDEBUG.
# The stock types are all wrong for this tree right now: Debug has no -O flag
# at all, and RelWithDebInfo/Release both define NDEBUG. That matters because
# tests/test_phase7.cpp has a failing malloc assertion at HEAD -- an allocator
# invariant violation that -DNDEBUG silently compiles away, in the very same
# change that made guest memory corruption appear. An allocator bug can corrupt
# a stack canary indirectly (a bad length field driving an oversized memcpy), so
# the assert is the best-supported lead and must stay live while we chase it.
set(CMAKE_CXX_FLAGS_INVESTIGATE "-O2 -g" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INVESTIGATE "-O2 -g" CACHE STRING "" FORCE)

# Only sets a default, so -DCMAKE_BUILD_TYPE=Debug/Release still work.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Investigate CACHE STRING "Build type" FORCE)
endif()

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
    src/gui/game_import.cpp
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
set_source_files_properties(src/gui/game_library_gui.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

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

add_executable(test_flappy_gpu tests/test_flappy_gpu.cpp)
target_link_libraries(test_flappy_gpu PRIVATE ps4_core)
target_compile_definitions(test_flappy_gpu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME Flappy_GpuSmoke COMMAND test_flappy_gpu)
set_tests_properties(Flappy_GpuSmoke PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})




# Game import / library GUI (AppKit). Separate binary so the runner stays usable
# headless; it locates ps4-runner next to itself at runtime.
add_executable(ps4-gui
    src/gui/game_library_gui.mm
)
target_include_directories(ps4-gui PRIVATE include)
target_link_libraries(ps4-gui PRIVATE ps4_core)
set_source_files_properties(src/gui/game_library_gui.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Scanner test for the game-import GUI: finds the real dump layout on disk.
add_executable(test_phase16 tests/test_phase16.cpp src/gui/game_import.cpp)
target_include_directories(test_phase16 PRIVATE include)
add_test(NAME Phase16_GameImport COMMAND test_phase16)
# The scanner test inspects the real games/ layout, so it must run from the
# source root, not the build directory.
set_tests_properties(Phase16_GameImport PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# x87 rounding / control-word vectors. Drives real instruction bytes through the
# interpreter so the test and the emulator cannot diverge silently.
add_executable(test_x87 tests/test_x87.cpp)
target_include_directories(test_x87 PRIVATE include)
target_link_libraries(test_x87 PRIVATE ps4_core)
add_test(NAME Phase17_X87 COMMAND test_x87)

# ---------------------------------------------------------------------------
# ps4-gui.app bundle
#
# A bare binary has no dock icon, no bundle identity and does not survive the
# terminal closing. Wrapping it gives a real double-clickable app. ps4-runner
# is copied in alongside it because the GUI resolves the runner relative to its
# own executable path, which inside a bundle is Contents/MacOS/.
# ---------------------------------------------------------------------------
set(GUI_BUNDLE "${CMAKE_BINARY_DIR}/ps4-gui.app")
set(GUI_BUNDLE_MACOS "${GUI_BUNDLE}/Contents/MacOS")
set(GUI_BUNDLE_RESOURCES "${GUI_BUNDLE}/Contents/Resources")

add_custom_target(ps4-gui-app ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GUI_BUNDLE_MACOS}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GUI_BUNDLE_RESOURCES}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ps4-gui>" "${GUI_BUNDLE_MACOS}/ps4-gui"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ps4-runner>" "${GUI_BUNDLE_MACOS}/ps4-runner"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/cmake/Info.plist.in" "${GUI_BUNDLE}/Contents/Info.plist"
    COMMAND ${CMAKE_COMMAND} -E echo "APPL????" > "${GUI_BUNDLE}/Contents/PkgInfo"
    DEPENDS ps4-gui ps4-runner
    COMMENT "Assembling ps4-gui.app"
    VERBATIM
)
