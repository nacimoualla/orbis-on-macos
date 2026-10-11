#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "ps4/audio.hpp"
#include "ps4/audio_manager.hpp"
#include "ps4/coreaudio_backend.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"

int main() {
    std::cout << "[PHASE 11] Starting Audio Subsystem (libaudioout) & CoreAudio tests...\n";

    ps4::AudioOutManager audio_mgr;

    // Test 1: Audio port allocation & lifecycle
    {
        assert(audio_mgr.active_port_count() == 0);
        const ps4::i32 port1 = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);
        assert(port1 > 0);
        assert(audio_mgr.is_port_open(port1));
        assert(audio_mgr.active_port_count() == 1);

        ps4::SceAudioOutPortInfo info{};
        assert(audio_mgr.get_port_info(port1, info));
        assert(info.status == 1);
        assert(info.queued_samples == 0);

        assert(audio_mgr.close_port(port1));
        assert(!audio_mgr.is_port_open(port1));
        assert(audio_mgr.active_port_count() == 0);

        std::cout << "  ✓ Test 1: Audio port open/close lifecycle and info query verified\n";
    }

    // Test 2: PCM 16-bit stereo sample submission & readback
    {
        const ps4::i32 port = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);

        // Generate 512 stereo samples (1024 total i16 values)
        std::vector<ps4::i16> samples(1024);
        for (ps4::usize i = 0; i < samples.size(); ++i) {
            samples[i] = static_cast<ps4::i16>((i % 256) * 100);
        }

        assert(audio_mgr.output_samples(port, samples));
        assert(audio_mgr.get_queued_sample_count(port) == 1024);

        // Read back into buffer
        std::vector<ps4::i16> read_buf(1024, 0);
        const ps4::usize read_count = audio_mgr.read_samples(port, read_buf);
        assert(read_count == 1024);
        assert(audio_mgr.get_queued_sample_count(port) == 0);

        // Verify data integrity
        for (ps4::usize i = 0; i < samples.size(); ++i) {
            assert(read_buf[i] == samples[i]);
        }

        assert(audio_mgr.close_port(port));
        std::cout << "  ✓ Test 2: 16-bit PCM stereo sample queueing and exact FIFO readback verified\n";
    }

    // Test 3: PCM 32-bit float conversion to 16-bit PCM
    {
        const ps4::i32 port = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);

        std::vector<ps4::f32> float_samples = {0.0f, 0.5f, 1.0f, -1.0f, -0.5f};
        assert(audio_mgr.output_samples_float(port, float_samples));
        assert(audio_mgr.get_queued_sample_count(port) == 5);

        std::vector<ps4::i16> read_buf(5, 0);
        audio_mgr.read_samples(port, read_buf);

        assert(read_buf[0] == 0);
        assert(std::abs(read_buf[1] - 16383) <= 1);
        assert(read_buf[2] == 32767);
        assert(read_buf[3] == -32767);

        assert(audio_mgr.close_port(port));
        std::cout << "  ✓ Test 3: 32-bit float to 16-bit PCM conversion and clipping verified\n";
    }

    // Test 4: Volume control and attenuation
    {
        const ps4::i32 port = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);

        // Set volume: Left channel = 50%, Right channel = 25%
        assert(audio_mgr.set_volume(port, 0.5f, 0.25f));

        std::vector<ps4::i16> input = {1000, 1000, 2000, 2000};
        assert(audio_mgr.output_samples(port, input));

        std::vector<ps4::i16> output(4, 0);
        audio_mgr.read_samples(port, output);

        assert(output[0] == 500);  // Left: 1000 * 0.5
        assert(output[1] == 250);  // Right: 1000 * 0.25
        assert(output[2] == 1000); // Left: 2000 * 0.5
        assert(output[3] == 500);  // Right: 2000 * 0.25

        assert(audio_mgr.close_port(port));
        std::cout << "  ✓ Test 4: Independent Left/Right channel volume attenuation verified\n";
    }

    // Test 5: Underflow silence padding
    {
        const ps4::i32 port = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);

        std::vector<ps4::i16> samples = {123, 456};
        assert(audio_mgr.output_samples(port, samples));

        // Read 6 samples (more than available)
        std::vector<ps4::i16> read_buf(6, 999);
        const ps4::usize read_count = audio_mgr.read_samples(port, read_buf);
        assert(read_count == 2);
        assert(read_buf[0] == 123);
        assert(read_buf[1] == 456);
        assert(read_buf[2] == 0); // Silence padded
        assert(read_buf[3] == 0);
        assert(read_buf[4] == 0);
        assert(read_buf[5] == 0);

        assert(audio_mgr.close_port(port));
        std::cout << "  ✓ Test 5: Underflow buffer exhaustion cleanly padded with silence\n";
    }

    // Test 6: CoreAudioBackend lifecycle
    {
        const ps4::i32 port = audio_mgr.open_port(ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN);
        ps4::CoreAudioBackend backend(audio_mgr, port);

        assert(backend.initialize(true)); // Headless mode
        assert(backend.is_headless());
        assert(!backend.is_running());

        assert(backend.start());
        assert(backend.is_running());

        std::vector<ps4::i16> pcm = {100, 200, 300, 400};
        assert(audio_mgr.output_samples(port, pcm));

        std::vector<ps4::i16> processed(4, 0);
        const ps4::usize count = backend.process_audio_buffer(processed);
        assert(count == 4);
        assert(processed[0] == 100);

        assert(backend.stop());
        assert(!backend.is_running());

        assert(audio_mgr.close_port(port));
        std::cout << "  ✓ Test 6: CoreAudioBackend initialization, streaming, and stop lifecycle verified\n";
    }

    // Test 7: HLE libaudioout NID resolution & handlers
    {
        ps4::MemoryManager memory;
        assert(memory.initialize());

        ps4::NidRegistry registry;
        registry.register_default_libraries();

        assert(registry.find_by_name("sceAudioOutOpen") != nullptr);
        assert(registry.find_by_name("sceAudioOutClose") != nullptr);
        assert(registry.find_by_name("sceAudioOutOutput") != nullptr);
        assert(registry.find_by_name("sceAudioOutSetVolume") != nullptr);
        assert(registry.find_by_name("sceAudioOutGetPortInfo") != nullptr);

        // Open port via HLE handler
        const auto* sym_open = registry.find_by_name("sceAudioOutOpen");
        ps4::CpuContext ctx{};
        ctx.rsi = ps4::SCE_AUDIO_OUT_PORT_TYPE_MAIN;
        sym_open->handler(ctx, memory);
        const ps4::i32 handle = static_cast<ps4::i32>(ctx.rax);
        assert(handle > 0);

        // Submit audio buffer via HLE handler
        const ps4::GuestAddr audio_guest_buf = 0x600000ULL;
        std::vector<ps4::i16> pcm_data(1024, 1234);
        assert(memory.write_bytes(audio_guest_buf, pcm_data.data(), pcm_data.size() * sizeof(ps4::i16)));

        const auto* sym_output = registry.find_by_name("sceAudioOutOutput");
        ctx.rdi = handle;
        ctx.rsi = audio_guest_buf;
        sym_output->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        // Close port via HLE handler
        const auto* sym_close = registry.find_by_name("sceAudioOutClose");
        ctx.rdi = handle;
        sym_close->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        std::cout << "  ✓ Test 7: libaudioout NID resolution and HLE dispatch verified\n";
    }

    std::cout << "[PHASE 11] PASS: All Audio Subsystem (libaudioout) & CoreAudio tests passed!\n";
    return 0;
}
