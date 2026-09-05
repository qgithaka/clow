#pragma once

#include <cstddef>
#include <vector>

namespace clow::ai {

/**
 * @brief High-performance circular ring buffer for sliding window time-series features.
 * 
 * Provides zero-allocation continuous bar pushing and contiguous chronological export.
 */
class SlidingWindowBuffer {
public:
    /**
     * @brief Constructs buffer with fixed sequence length and feature dimension.
     * @param context_length Sequence length L (e.g. 64 bars).
     * @param feature_dim Feature dimension D (e.g. 8 or 20 features per bar).
     */
    SlidingWindowBuffer(size_t context_length, size_t feature_dim);

    ~SlidingWindowBuffer() = default;

    /**
     * @brief Pushes a new chronological bar into the circular buffer.
     * @param features Vector of feature values for current bar.
     */
    void push_bar(const std::vector<float>& features);

    /**
     * @brief Zero-allocation raw pointer push for incoming bar data.
     * @param features Pointer to float array of size feature_dim.
     * @param dim Feature dimension to copy.
     */
    void push_bar(const float* features, size_t dim);

    /**
     * @brief Returns whether the buffer has accumulated at least context_length bars.
     */
    [[nodiscard]] bool is_full() const;

    /**
     * @brief Number of bars currently held in the buffer (up to context_length).
     */
    [[nodiscard]] size_t count() const;

    /**
     * @brief Configured capacity in number of bars.
     */
    [[nodiscard]] size_t context_length() const;

    /**
     * @brief Feature dimension per bar.
     */
    [[nodiscard]] size_t feature_dim() const;

    /**
     * @brief Clears all stored bars.
     */
    void clear();

    /**
     * @brief Copies chronological sequence from oldest to newest into output vector.
     * @param out_buffer Pre-allocated or resized vector receiving [L * D] floats.
     */
    void get_chronological_data(std::vector<float>& out_buffer) const;

    /**
     * @brief Copies chronological sequence directly into raw pointer buffer.
     * @param dst_ptr Output memory destination of at least (count() * feature_dim()) floats.
     */
    void get_chronological_data(float* dst_ptr) const;

private:
    size_t m_context_length;
    size_t m_feature_dim;
    std::vector<float> m_storage; // Size: m_context_length * m_feature_dim
    size_t m_head{0};             // Next insertion slot
    size_t m_count{0};            // Total items inserted
};

} // namespace clow::ai
