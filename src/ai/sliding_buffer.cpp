#include "ai/sliding_buffer.h"

#include <algorithm>
#include <cstring>

namespace clow::ai {

SlidingWindowBuffer::SlidingWindowBuffer(size_t context_length, size_t feature_dim)
    : m_context_length(std::max(size_t{1}, context_length)),
      m_feature_dim(std::max(size_t{1}, feature_dim)),
      m_storage(m_context_length * m_feature_dim, 0.0f),
      m_head(0),
      m_count(0) {}

void SlidingWindowBuffer::push_bar(const std::vector<float>& features) {
    push_bar(features.data(), features.size());
}

void SlidingWindowBuffer::push_bar(const float* features, size_t dim) {
    if (features == nullptr || dim == 0) return;

    size_t copy_dim = std::min(dim, m_feature_dim);
    float* dest = m_storage.data() + (m_head * m_feature_dim);
    
    std::memcpy(dest, features, copy_dim * sizeof(float));
    if (copy_dim < m_feature_dim) {
        std::memset(dest + copy_dim, 0, (m_feature_dim - copy_dim) * sizeof(float));
    }

    m_head = (m_head + 1) % m_context_length;
    if (m_count < m_context_length) {
        m_count++;
    }
}

bool SlidingWindowBuffer::is_full() const {
    return m_count >= m_context_length;
}

size_t SlidingWindowBuffer::count() const {
    return m_count;
}

size_t SlidingWindowBuffer::context_length() const {
    return m_context_length;
}

size_t SlidingWindowBuffer::feature_dim() const {
    return m_feature_dim;
}

void SlidingWindowBuffer::clear() {
    m_head = 0;
    m_count = 0;
    std::fill(m_storage.begin(), m_storage.end(), 0.0f);
}

void SlidingWindowBuffer::get_chronological_data(std::vector<float>& out_buffer) const {
    out_buffer.resize(m_count * m_feature_dim);
    get_chronological_data(out_buffer.data());
}

void SlidingWindowBuffer::get_chronological_data(float* dst_ptr) const {
    if (dst_ptr == nullptr || m_count == 0) return;

    if (m_count < m_context_length) {
        // Not wrapped around yet: direct continuous copy from index 0
        std::memcpy(dst_ptr, m_storage.data(), m_count * m_feature_dim * sizeof(float));
    } else {
        // Buffer is full and has wrapped around:
        // Oldest item is at m_head up to m_context_length - 1
        // Newer items are from 0 up to m_head - 1
        size_t oldest_part_bars = m_context_length - m_head;
        size_t newest_part_bars = m_head;

        if (oldest_part_bars > 0) {
            std::memcpy(
                dst_ptr,
                m_storage.data() + (m_head * m_feature_dim),
                oldest_part_bars * m_feature_dim * sizeof(float)
            );
        }

        if (newest_part_bars > 0) {
            std::memcpy(
                dst_ptr + (oldest_part_bars * m_feature_dim),
                m_storage.data(),
                newest_part_bars * m_feature_dim * sizeof(float)
            );
        }
    }
}

} // namespace clow::ai
