#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace clow::ai {

/**
 * @brief Next-candle quantitative forecast output from ONNX model.
 */
struct ForecasterPrediction {
    std::vector<float> anatomy;       // [body_ratio, upper_wick, lower_wick, range_to_atr]
    float direction_logit{0.0f};      // Raw output logit
    float direction_prob{0.5f};       // Sigmoid probability (0.0 to 1.0)
    std::vector<float> quantiles_high; // e.g. [q10, q50, q90] upward excursions
    std::vector<float> quantiles_low;  // e.g. [q10, q50, q90] downward excursions
    bool valid{false};
};

/**
 * @brief Runtime model metadata and schema specification.
 */
struct ModelMetadata {
    std::string model_id;
    std::string description;
    std::string format{"ONNX"};
    size_t context_length{64};
    size_t input_dim{8};
    std::vector<std::string> feature_names;
    std::vector<float> quantiles{0.10f, 0.50f, 0.90f};
    std::vector<std::string> anatomy_target_names{"body_ratio", "upper_wick", "lower_wick", "range_to_atr"};
    
    std::string scaler_type{"RollingZScoreScaler"};
    size_t window_size{64};
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    std::vector<float> feature_mins;
    std::vector<float> feature_maxs;
    float clip_val{5.0f};
    float eps{1e-8f};
    
    std::string onnx_sha256;
    std::string metadata_sha256;
};

/**
 * @brief High-performance C++ ONNX inference engine for Clow-Forecaster.
 * 
 * Supports dynamic model loading, schema parsing, and sub-5ms batch-1 CPU inference.
 */
class OnnxInferenceEngine {
public:
    OnnxInferenceEngine();
    ~OnnxInferenceEngine();

    /**
     * @brief Loads model metadata from a JSON configuration file.
     * @param metadata_json_path Path to model_metadata.json.
     * @return true if successfully loaded and parsed.
     */
    bool load_metadata(const std::string& metadata_json_path);

    /**
     * @brief Loads ONNX binary model.
     * @param model_path Path to .onnx file.
     * @param metadata_path Optional path to accompanying model_metadata.json.
     * @return true if model initialized successfully.
     */
    bool load_model(const std::string& model_path, const std::string& metadata_path = "");

    /**
     * @brief Checks if a model is currently loaded and ready for inference.
     */
    [[nodiscard]] bool is_loaded() const;

    /**
     * @brief Retrieves active model metadata.
     */
    [[nodiscard]] const ModelMetadata& metadata() const;

    /**
     * @brief Executes neural inference on a flattened sequence of shape [seq_len, num_features].
     * @param flattened_input Chronological sequence of stationary features.
     * @return ForecasterPrediction structure containing anatomy, direction, and quantiles.
     */
    ForecasterPrediction predict_single(const std::vector<float>& flattened_input);

    /**
     * @brief Executes raw pointer neural inference for zero-allocation streaming execution.
     * @param feature_data Pointer to contiguous float buffer of size seq_len * num_features.
     * @param seq_len Sequence length (e.g. 64).
     * @param num_features Feature dimension (e.g. 8).
     * @return ForecasterPrediction structure.
     */
    ForecasterPrediction predict(const float* feature_data, size_t seq_len, size_t num_features);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace clow::ai
