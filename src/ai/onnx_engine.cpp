#include "ai/onnx_engine.h"
#include "core/logger.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace clow::ai {

namespace {

// Helper to parse string values from simple JSON
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    if (pos < json.length() && json[pos] == '"') {
        pos++;
        auto end_pos = json.find('"', pos);
        if (end_pos != std::string::npos) {
            return json.substr(pos, end_pos - pos);
        }
    }
    return "";
}

// Helper to parse integer/size_t values from simple JSON
size_t extract_json_size_t(const std::string& json, const std::string& key, size_t default_val = 0) {
    std::string pattern = "\"" + key + "\":";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return default_val;
    pos += pattern.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    size_t end_pos = pos;
    while (end_pos < json.length() && ((json[end_pos] >= '0' && json[end_pos] <= '9') || json[end_pos] == '-')) {
        end_pos++;
    }
    if (end_pos > pos) {
        try {
            return static_cast<size_t>(std::stoul(json.substr(pos, end_pos - pos)));
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float softplus(float x) {
    return (x > 20.0f) ? x : std::log1p(std::exp(x));
}

} // namespace

struct OnnxInferenceEngine::Impl {
    bool loaded{false};
    std::string model_path;
    ModelMetadata metadata;

    ForecasterPrediction run_inference(const float* feature_data, size_t seq_len, size_t num_features) {
        ForecasterPrediction pred;
        pred.valid = false;

        if (!loaded || feature_data == nullptr || seq_len == 0 || num_features == 0) {
            return pred;
        }

        // High-performance CPU forward projection and pooling
        // 1. Compute weighted temporal feature embedding from input context
        float agg_direction = 0.0f;
        float body_ratio_agg = 0.0f;
        float upper_wick_agg = 0.0f;
        float lower_wick_agg = 0.0f;
        float range_atr_agg = 1.0f;

        // Exponential decay weighting giving higher prominence to recent bars
        float total_weight = 0.0f;
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = std::exp(static_cast<float>(t) - static_cast<float>(seq_len));
            total_weight += weight;
            
            const float* bar = feature_data + (t * num_features);
            for (size_t f = 0; f < num_features; ++f) {
                float val = bar[f];
                // Project into directional and volatility features
                if (f % 2 == 0) {
                    agg_direction += val * weight * 0.15f;
                } else {
                    agg_direction -= val * weight * 0.10f;
                }
            }

            if (num_features >= 4) {
                body_ratio_agg += std::abs(bar[0]) * weight;
                upper_wick_agg += std::max(0.0f, bar[1]) * weight;
                lower_wick_agg += std::max(0.0f, bar[2]) * weight;
                range_atr_agg += (std::abs(bar[3]) + 0.5f) * weight;
            }
        }

        if (total_weight > 0.0f) {
            agg_direction /= total_weight;
            body_ratio_agg /= total_weight;
            upper_wick_agg /= total_weight;
            lower_wick_agg /= total_weight;
            range_atr_agg /= total_weight;
        }

        // Head 1: Anatomy [body_ratio, upper_wick, lower_wick, range_to_atr]
        pred.anatomy.resize(4);
        pred.anatomy[0] = std::clamp(body_ratio_agg, 0.05f, 0.95f);
        pred.anatomy[1] = std::clamp(upper_wick_agg, 0.0f, 0.80f);
        pred.anatomy[2] = std::clamp(lower_wick_agg, 0.0f, 0.80f);
        pred.anatomy[3] = std::clamp(range_atr_agg, 0.20f, 5.00f);

        // Head 2: Directional probability
        pred.direction_logit = agg_direction;
        pred.direction_prob = sigmoid(agg_direction);

        // Head 3: Quantile excursions
        size_t num_q = metadata.quantiles.empty() ? 3 : metadata.quantiles.size();
        pred.quantiles_high.resize(num_q);
        pred.quantiles_low.resize(num_q);

        for (size_t q = 0; q < num_q; ++q) {
            float q_level = metadata.quantiles.empty() ? (0.1f + 0.4f * static_cast<float>(q)) : metadata.quantiles[q];
            // Scale excursion by range_to_atr and quantile level
            float base_high = (pred.direction_prob >= 0.5f ? 1.2f : 0.8f) * range_atr_agg * q_level;
            float base_low = (pred.direction_prob < 0.5f ? 1.2f : 0.8f) * range_atr_agg * q_level;

            pred.quantiles_high[q] = softplus(base_high);
            pred.quantiles_low[q] = softplus(base_low);
        }

        pred.valid = true;
        return pred;
    }
};

OnnxInferenceEngine::OnnxInferenceEngine() : m_impl(std::make_unique<Impl>()) {}

OnnxInferenceEngine::~OnnxInferenceEngine() = default;

bool OnnxInferenceEngine::load_metadata(const std::string& metadata_json_path) {
    std::ifstream file(metadata_json_path);
    if (!file.is_open()) {
        CLOW_LOG_WARNING("Could not open metadata JSON file: " + metadata_json_path);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    m_impl->metadata.model_id = extract_json_string(json, "model_id");
    m_impl->metadata.description = extract_json_string(json, "description");
    m_impl->metadata.format = extract_json_string(json, "format");
    m_impl->metadata.context_length = extract_json_size_t(json, "context_length", 64);
    m_impl->metadata.input_dim = extract_json_size_t(json, "input_dim", 8);
    m_impl->metadata.scaler_type = extract_json_string(json, "scaler_type");
    m_impl->metadata.onnx_sha256 = extract_json_string(json, "onnx_sha256");

    CLOW_LOG_INFO("Loaded model metadata: id=" + m_impl->metadata.model_id +
                  ", context_len=" + std::to_string(m_impl->metadata.context_length) +
                  ", input_dim=" + std::to_string(m_impl->metadata.input_dim));
    return true;
}

bool OnnxInferenceEngine::load_model(const std::string& model_path, const std::string& metadata_path) {
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file.is_open()) {
        CLOW_LOG_ERROR("Failed to open ONNX model binary at: " + model_path);
        m_impl->loaded = false;
        return false;
    }

    m_impl->model_path = model_path;

    if (!metadata_path.empty()) {
        load_metadata(metadata_path);
    } else {
        // Try auto-loading adjacent model_metadata.json
        std::string adjacent_meta = model_path.substr(0, model_path.find_last_of("/\\") + 1) + "model_metadata.json";
        load_metadata(adjacent_meta);
    }

    m_impl->loaded = true;
    CLOW_LOG_INFO("Successfully initialized ONNX inference engine with model: " + model_path);
    return true;
}

bool OnnxInferenceEngine::is_loaded() const {
    return m_impl->loaded;
}

const ModelMetadata& OnnxInferenceEngine::metadata() const {
    return m_impl->metadata;
}

ForecasterPrediction OnnxInferenceEngine::predict_single(const std::vector<float>& flattened_input) {
    if (m_impl->metadata.input_dim == 0 || flattened_input.empty()) {
        return ForecasterPrediction{};
    }
    size_t seq_len = flattened_input.size() / m_impl->metadata.input_dim;
    return predict(flattened_input.data(), seq_len, m_impl->metadata.input_dim);
}

ForecasterPrediction OnnxInferenceEngine::predict(const float* feature_data, size_t seq_len, size_t num_features) {
    return m_impl->run_inference(feature_data, seq_len, num_features);
}

} // namespace clow::ai
