#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../src/ai/feature_normalizer.h"
#include "../../src/ai/inference_benchmark.h"
#include "../../src/ai/onnx_engine.h"
#include "../../src/ai/sliding_buffer.h"
#include "../../src/core/logger.h"

void test_sliding_window_buffer() {
    using namespace clow::ai;

    SlidingWindowBuffer buffer(4, 2); // 4 bars, 2 features
    assert(!buffer.is_full());
    assert(buffer.count() == 0);

    // Push 1st bar: [1.0, 2.0]
    std::vector<float> b1 = {1.0f, 2.0f};
    buffer.push_bar(b1);
    assert(buffer.count() == 1);
    assert(!buffer.is_full());

    // Push 2nd, 3rd, 4th bars
    buffer.push_bar(std::vector<float>{3.0f, 4.0f});
    buffer.push_bar(std::vector<float>{5.0f, 6.0f});
    buffer.push_bar(std::vector<float>{7.0f, 8.0f});
    assert(buffer.is_full());
    assert(buffer.count() == 4);

    std::vector<float> data;
    buffer.get_chronological_data(data);
    assert(data.size() == 8);
    assert(data[0] == 1.0f && data[1] == 2.0f);
    assert(data[6] == 7.0f && data[7] == 8.0f);

    // Push 5th bar (wrap around): [9.0, 10.0] -> oldest bar [1.0, 2.0] is replaced
    buffer.push_bar(std::vector<float>{9.0f, 10.0f});
    assert(buffer.is_full());
    assert(buffer.count() == 4);

    buffer.get_chronological_data(data);
    assert(data.size() == 8);
    // Chronological order should now be: b2 [3, 4], b3 [5, 6], b4 [7, 8], b5 [9, 10]
    assert(data[0] == 3.0f && data[1] == 4.0f);
    assert(data[2] == 5.0f && data[3] == 6.0f);
    assert(data[4] == 7.0f && data[5] == 8.0f);
    assert(data[6] == 9.0f && data[7] == 10.0f);

    std::cout << "[PASS] test_sliding_window_buffer" << std::endl;
}

void test_feature_normalizer() {
    using namespace clow::ai;

    FeatureNormalizer norm(NormalizerType::ZScore, 2, 3.0f, 1e-8f);
    // Center: [10.0, 20.0], Scale: [2.0, 5.0]
    norm.set_parameters({10.0f, 20.0f}, {2.0f, 5.0f});

    std::vector<float> raw_data = {
        12.0f, 25.0f,  // bar 0: (12-10)/2 = 1.0, (25-20)/5 = 1.0
        10.0f, 10.0f,  // bar 1: (10-10)/2 = 0.0, (10-20)/5 = -2.0
        20.0f, 50.0f   // bar 2: (20-10)/2 = 5.0 -> clipped to 3.0, (50-20)/5 = 6.0 -> clipped to 3.0
    };

    std::vector<float> normalized(6);
    norm.normalize_copy(raw_data.data(), normalized.data(), 3, 2);

    assert(std::abs(normalized[0] - 1.0f) < 1e-5f);
    assert(std::abs(normalized[1] - 1.0f) < 1e-5f);
    assert(std::abs(normalized[2] - 0.0f) < 1e-5f);
    assert(std::abs(normalized[3] - (-2.0f)) < 1e-5f);
    assert(std::abs(normalized[4] - 3.0f) < 1e-5f); // Clipped
    assert(std::abs(normalized[5] - 3.0f) < 1e-5f); // Clipped

    // Test streaming step_online
    float single_bar[2] = {14.0f, 15.0f}; // (14-10)/2 = 2.0, (15-20)/5 = -1.0
    float single_out[2] = {0.0f, 0.0f};
    norm.step_online(single_bar, single_out, 2);
    assert(std::abs(single_out[0] - 2.0f) < 1e-5f);
    assert(std::abs(single_out[1] - (-1.0f)) < 1e-5f);

    std::cout << "[PASS] test_feature_normalizer" << std::endl;
}

void test_onnx_inference_engine_execution() {
    using namespace clow::ai;

    // Create a temporary mock ONNX model binary and metadata JSON
    const std::string test_model_path = "test_mock_model.onnx";
    const std::string test_meta_path = "test_mock_metadata.json";

    {
        std::ofstream mf(test_model_path, std::ios::binary);
        mf << "ONNX_MOCK_BINARY_DATA";
    }

    {
        std::ofstream jf(test_meta_path);
        jf << "{\n"
           << "  \"model_id\": \"clow_forecaster_mock\",\n"
           << "  \"description\": \"Mock transformer forecaster\",\n"
           << "  \"context_length\": 4,\n"
           << "  \"input_dim\": 4,\n"
           << "  \"scaler_type\": \"RollingZScoreScaler\"\n"
           << "}\n";
    }

    OnnxInferenceEngine engine;
    assert(!engine.is_loaded());

    [[maybe_unused]] bool loaded = engine.load_model(test_model_path, test_meta_path);
    assert(loaded);
    assert(engine.is_loaded());
    assert(engine.metadata().model_id == "clow_forecaster_mock");
    assert(engine.metadata().context_length == 4);
    assert(engine.metadata().input_dim == 4);

    // Run prediction on mock 4 bars * 4 features
    std::vector<float> sample_seq(16, 0.5f);
    ForecasterPrediction pred = engine.predict_single(sample_seq);

    assert(pred.valid);
    assert(pred.anatomy.size() == 4);
    assert(pred.direction_prob >= 0.0f && pred.direction_prob <= 1.0f);
    assert(pred.quantiles_high.size() == 3);
    assert(pred.quantiles_low.size() == 3);
    // Quantiles strictly non-negative
    for ([[maybe_unused]] float qh : pred.quantiles_high) assert(qh >= 0.0f);
    for ([[maybe_unused]] float ql : pred.quantiles_low) assert(ql >= 0.0f);

    // Clean up temporary files
    std::remove(test_model_path.c_str());
    std::remove(test_meta_path.c_str());

    std::cout << "[PASS] test_onnx_inference_engine_execution" << std::endl;
}

void test_inference_benchmark_latency_gate() {
    using namespace clow::ai;

    OnnxInferenceEngine engine;
    // Load without external file to test fallback mode
    const std::string test_model_path = "test_bench_model.onnx";
    {
        std::ofstream mf(test_model_path, std::ios::binary);
        mf << "ONNX_BENCH_DATA";
    }
    engine.load_model(test_model_path);

    SlidingWindowBuffer buffer(64, 8); // 64 bars, 8 features
    FeatureNormalizer normalizer(NormalizerType::ZScore, 8);

    InferenceBenchmark benchmark(engine, buffer, normalizer);
    BenchmarkMetrics metrics = benchmark.run_end_to_end_benchmark(1000, 5000.0);

    assert(metrics.iterations == 1000);
    assert(metrics.p99_latency_us > 0.0);
    assert(metrics.p99_latency_us < 5000.0); // Must satisfy sub-5ms latency gate
    assert(metrics.passed_latency_gate);

    std::remove(test_model_path.c_str());
    std::cout << "[PASS] test_inference_benchmark_latency_gate (p99="
              << metrics.p99_latency_us << " us < 5000 us)" << std::endl;
}
