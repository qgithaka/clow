#include "ai/feature_normalizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace clow::ai {

FeatureNormalizer::FeatureNormalizer(
    NormalizerType type,
    size_t feature_dim,
    float clip_val,
    float eps
)
    : m_type(type),
      m_feature_dim(feature_dim),
      m_clip_val(clip_val),
      m_eps(eps),
      m_center(feature_dim, 0.0f),
      m_scale(feature_dim, 1.0f) {}

void FeatureNormalizer::set_parameters(const std::vector<float>& center, const std::vector<float>& scale) {
    if (center.size() == m_feature_dim) {
        m_center = center;
    }
    if (scale.size() == m_feature_dim) {
        m_scale = scale;
    }
}

void FeatureNormalizer::normalize_in_place(float* data, size_t num_bars, size_t num_features) const {
    if (data == nullptr || num_bars == 0 || num_features == 0) return;

    if (m_type == NormalizerType::Passthrough) return;

    size_t effective_features = std::min(num_features, m_feature_dim);

    for (size_t b = 0; b < num_bars; ++b) {
        float* bar = data + (b * num_features);
        
        #pragma GCC unroll 4
        for (size_t f = 0; f < effective_features; ++f) {
            float val = bar[f];
            float c = m_center[f];
            float s = m_scale[f];

            float scaled = (val - c) / (s + m_eps);

            if (m_clip_val > 0.0f) {
                scaled = std::clamp(scaled, -m_clip_val, m_clip_val);
            }

            bar[f] = scaled;
        }
    }
}

void FeatureNormalizer::normalize_copy(const float* src, float* dst, size_t num_bars, size_t num_features) const {
    if (src == nullptr || dst == nullptr || num_bars == 0 || num_features == 0) return;

    size_t total_elements = num_bars * num_features;
    std::memcpy(dst, src, total_elements * sizeof(float));
    normalize_in_place(dst, num_bars, num_features);
}

void FeatureNormalizer::step_online(const float* current_bar, float* normalized_bar, size_t num_features) {
    if (current_bar == nullptr || normalized_bar == nullptr || num_features == 0) return;

    size_t effective_features = std::min(num_features, m_feature_dim);

    for (size_t f = 0; f < effective_features; ++f) {
        float val = current_bar[f];
        float c = m_center[f];
        float s = m_scale[f];

        float scaled = (val - c) / (s + m_eps);
        if (m_clip_val > 0.0f) {
            scaled = std::clamp(scaled, -m_clip_val, m_clip_val);
        }
        normalized_bar[f] = scaled;
    }
}

NormalizerType FeatureNormalizer::type() const {
    return m_type;
}

size_t FeatureNormalizer::feature_dim() const {
    return m_feature_dim;
}

float FeatureNormalizer::clip_val() const {
    return m_clip_val;
}

float FeatureNormalizer::eps() const {
    return m_eps;
}

} // namespace clow::ai
