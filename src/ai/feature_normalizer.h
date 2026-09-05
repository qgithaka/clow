#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace clow::ai {

enum class NormalizerType {
    ZScore,
    Robust,
    MinMax,
    Passthrough
};

/**
 * @brief SIMD-accelerated, zero-leakage feature normalizer for real-time inference.
 * 
 * Supports streaming online normalization and batch sliding window normalization.
 */
class FeatureNormalizer {
public:
    FeatureNormalizer(
        NormalizerType type = NormalizerType::ZScore,
        size_t feature_dim = 8,
        float clip_val = 5.0f,
        float eps = 1e-8f
    );

    ~FeatureNormalizer() = default;

    /**
     * @brief Configures normalization parameters per feature.
     * @param center Means or medians or mins (size matching feature_dim).
     * @param scale Standard deviations or IQRs or ranges (size matching feature_dim).
     */
    void set_parameters(const std::vector<float>& center, const std::vector<float>& scale);

    /**
     * @brief In-place normalization of a flattened sequence [num_bars * num_features].
     * @param data Pointer to continuous float buffer.
     * @param num_bars Number of bars.
     * @param num_features Feature dimension.
     */
    void normalize_in_place(float* data, size_t num_bars, size_t num_features) const;

    /**
     * @brief Normalizes source sequence into pre-allocated destination buffer.
     * @param src Source float buffer.
     * @param dst Destination float buffer.
     * @param num_bars Number of bars.
     * @param num_features Feature dimension.
     */
    void normalize_copy(const float* src, float* dst, size_t num_bars, size_t num_features) const;

    /**
     * @brief Normalizes a single incoming bar with fixed or streaming statistics.
     * @param current_bar Raw bar features.
     * @param normalized_bar Output normalized bar.
     * @param num_features Feature dimension.
     */
    void step_online(const float* current_bar, float* normalized_bar, size_t num_features);

    [[nodiscard]] NormalizerType type() const;
    [[nodiscard]] size_t feature_dim() const;
    [[nodiscard]] float clip_val() const;
    [[nodiscard]] float eps() const;

private:
    NormalizerType m_type;
    size_t m_feature_dim;
    float m_clip_val;
    float m_eps;
    std::vector<float> m_center; // mean / median / min
    std::vector<float> m_scale;  // std / iqr / range
};

} // namespace clow::ai
