#include "ps4/audio_manager.hpp"
#include <algorithm>
#include <cmath>

namespace ps4 {

AudioOutManager::AudioOutManager() = default;
AudioOutManager::~AudioOutManager() = default;

i32 AudioOutManager::open_port(i32 type, const AudioPortConfig& config) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    const i32 handle = m_next_handle++;

    auto port = std::make_unique<AudioPort>();
    port->handle = handle;
    port->type = type;
    port->config = config;

    m_ports[handle] = std::move(port);
    return handle;
}

bool AudioOutManager::close_port(i32 handle) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    return m_ports.erase(handle) > 0;
}

bool AudioOutManager::output_samples(i32 handle, std::span<const i16> samples) {
    AudioPort* port = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_manager_mutex);
        auto it = m_ports.find(handle);
        if (it == m_ports.end()) return false;
        port = it->second.get();
    }

    std::lock_guard<std::mutex> lock(port->mutex);
    for (usize i = 0; i < samples.size(); ++i) {
        const f32 gain = (i % 2 == 0) ? port->config.volume_left : port->config.volume_right;
        const i32 scaled = static_cast<i32>(static_cast<f32>(samples[i]) * gain);
        const i16 clamped = static_cast<i16>(std::clamp(scaled, -32768, 32767));
        port->sample_queue.push_back(clamped);
    }
    port->total_samples_submitted += samples.size();
    return true;
}

bool AudioOutManager::output_samples_float(i32 handle, std::span<const f32> samples) {
    AudioPort* port = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_manager_mutex);
        auto it = m_ports.find(handle);
        if (it == m_ports.end()) return false;
        port = it->second.get();
    }

    std::lock_guard<std::mutex> lock(port->mutex);
    for (usize i = 0; i < samples.size(); ++i) {
        const f32 gain = (i % 2 == 0) ? port->config.volume_left : port->config.volume_right;
        const f32 scaled = std::clamp(samples[i] * gain, -1.0f, 1.0f);
        const i16 pcm16 = static_cast<i16>(scaled * 32767.0f);
        port->sample_queue.push_back(pcm16);
    }
    port->total_samples_submitted += samples.size();
    return true;
}

bool AudioOutManager::set_volume(i32 handle, f32 left, f32 right) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    auto it = m_ports.find(handle);
    if (it == m_ports.end()) return false;

    std::lock_guard<std::mutex> port_lock(it->second->mutex);
    it->second->config.volume_left = std::clamp(left, 0.0f, 1.0f);
    it->second->config.volume_right = std::clamp(right, 0.0f, 1.0f);
    return true;
}

usize AudioOutManager::read_samples(i32 handle, std::span<i16> out_buffer) {
    AudioPort* port = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_manager_mutex);
        auto it = m_ports.find(handle);
        if (it == m_ports.end()) {
            std::fill(out_buffer.begin(), out_buffer.end(), static_cast<i16>(0));
            return 0;
        }
        port = it->second.get();
    }

    std::lock_guard<std::mutex> lock(port->mutex);
    usize read_count = 0;
    for (usize i = 0; i < out_buffer.size(); ++i) {
        if (!port->sample_queue.empty()) {
            out_buffer[i] = port->sample_queue.front();
            port->sample_queue.pop_front();
            read_count++;
        } else {
            out_buffer[i] = 0; // Fill with silence if buffer exhausted
        }
    }
    port->total_samples_read += read_count;
    return read_count;
}

usize AudioOutManager::get_queued_sample_count(i32 handle) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    auto it = m_ports.find(handle);
    if (it == m_ports.end()) return 0;

    std::lock_guard<std::mutex> port_lock(it->second->mutex);
    return it->second->sample_queue.size();
}

bool AudioOutManager::get_port_info(i32 handle, SceAudioOutPortInfo& out_info) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    auto it = m_ports.find(handle);
    if (it == m_ports.end()) return false;

    std::lock_guard<std::mutex> port_lock(it->second->mutex);
    out_info.status = 1;
    out_info.volume_left = static_cast<i32>(it->second->config.volume_left * SCE_AUDIO_OUT_VOLUME_MAX);
    out_info.volume_right = static_cast<i32>(it->second->config.volume_right * SCE_AUDIO_OUT_VOLUME_MAX);
    out_info.queued_samples = static_cast<u32>(it->second->sample_queue.size());
    return true;
}

bool AudioOutManager::is_port_open(i32 handle) {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    return m_ports.find(handle) != m_ports.end();
}

usize AudioOutManager::active_port_count() {
    std::lock_guard<std::mutex> lock(m_manager_mutex);
    return m_ports.size();
}

} // namespace ps4
