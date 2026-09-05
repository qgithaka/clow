#pragma once

#include "ai/feature_normalizer.h"
#include "ai/onnx_engine.h"
#include "ai/sliding_buffer.h"

#include <chrono>
#include <cstddef>
#include <vector>

namespace clow::ai {

struct BenchmarkMetrics {
    size_t iterations{0};
    double min_latency_us{0.0};
    double max_latency_us{0.0};
    double mean_latency_us{0.0};
    double p50_latency_us{0.0};
    double p95_latency_us{0.0};
    double p99_latency_us{0.0};
    bool passed_latency_gate{false}; // < 5000 us (5ms)
};

/**
 * @brief High-precision latency benchmark runner for Clow C++ inference pipeline.
 */
class InferenceBenchmark {
public:
    InferenceBenchmark(
        OnnxInferenceEngine& engine,
        SlidingWindowBuffer& buffer,
        FeatureNormalizer& normalizer
    );

    ~InferenceBenchmark() = default;

    /**
     * @brief Runs end-to-end streaming simulation benchmark across N iterations.
     * @param num_iterations Number of streaming iterations to run (e.g. 1,000).
     * @param max_permissible_p99_us Maximum permissible p99 latency in microseconds (default 5,000 us = 5ms).
     * @return BenchmarkMetrics struct.
     */
    BenchmarkMetrics run_end_to_end_benchmark(
        size_t num_iterations = 1000,
        double max_permissible_p99_us = 5000.0
    );

private:
    OnnxInferenceEngine& m_engine;
    SlidingWindowBuffer& m_buffer;
    FeatureNormalizer& m_normalizer;
};

} // namespace clow::ai
