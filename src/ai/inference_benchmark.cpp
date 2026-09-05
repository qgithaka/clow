#include "ai/inference_benchmark.h"
#include "core/logger.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace clow::ai {

InferenceBenchmark::InferenceBenchmark(
    OnnxInferenceEngine& engine,
    SlidingWindowBuffer& buffer,
    FeatureNormalizer& normalizer
)
    : m_engine(engine),
      m_buffer(buffer),
      m_normalizer(normalizer) {}

BenchmarkMetrics InferenceBenchmark::run_end_to_end_benchmark(
    size_t num_iterations,
    double max_permissible_p99_us
) {
    BenchmarkMetrics metrics;
    metrics.iterations = num_iterations;

    if (num_iterations == 0) return metrics;

    size_t feat_dim = m_buffer.feature_dim();
    size_t ctx_len = m_buffer.context_length();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // Pre-populate buffer to full capacity
    std::vector<float> sample_bar(feat_dim);
    for (size_t i = 0; i < ctx_len; ++i) {
        for (size_t f = 0; f < feat_dim; ++f) {
            sample_bar[f] = dist(rng);
        }
        m_buffer.push_bar(sample_bar);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(num_iterations);

    std::vector<float> continuous_data(ctx_len * feat_dim);
    std::vector<float> normalized_data(ctx_len * feat_dim);

    // Warm-up iterations
    for (size_t w = 0; w < 50; ++w) {
        for (size_t f = 0; f < feat_dim; ++f) sample_bar[f] = dist(rng);
        m_buffer.push_bar(sample_bar);
        m_buffer.get_chronological_data(continuous_data.data());
        m_normalizer.normalize_copy(continuous_data.data(), normalized_data.data(), ctx_len, feat_dim);
        m_engine.predict(normalized_data.data(), ctx_len, feat_dim);
    }

    // Benchmark loop
    for (size_t it = 0; it < num_iterations; ++it) {
        for (size_t f = 0; f < feat_dim; ++f) {
            sample_bar[f] = dist(rng);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 1. Step ring buffer
        m_buffer.push_bar(sample_bar);

        // 2. Extract chronological memory
        m_buffer.get_chronological_data(continuous_data.data());

        // 3. Apply feature normalizer
        m_normalizer.normalize_copy(continuous_data.data(), normalized_data.data(), ctx_len, feat_dim);

        // 4. Run model forward pass
        auto pred = m_engine.predict(normalized_data.data(), ctx_len, feat_dim);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);
    }

    std::sort(latencies_us.begin(), latencies_us.end());

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    metrics.min_latency_us = latencies_us.front();
    metrics.max_latency_us = latencies_us.back();
    metrics.mean_latency_us = sum / static_cast<double>(num_iterations);

    size_t idx_50 = static_cast<size_t>(0.50 * static_cast<double>(num_iterations));
    size_t idx_95 = static_cast<size_t>(0.95 * static_cast<double>(num_iterations));
    size_t idx_99 = static_cast<size_t>(0.99 * static_cast<double>(num_iterations));

    metrics.p50_latency_us = latencies_us[std::min(idx_50, num_iterations - 1)];
    metrics.p95_latency_us = latencies_us[std::min(idx_95, num_iterations - 1)];
    metrics.p99_latency_us = latencies_us[std::min(idx_99, num_iterations - 1)];

    metrics.passed_latency_gate = (metrics.p99_latency_us <= max_permissible_p99_us);

    CLOW_LOG_INFO("End-to-End Inference Benchmark (" + std::to_string(num_iterations) + " runs): " +
                  "Mean=" + std::to_string(metrics.mean_latency_us) + "us, " +
                  "P50=" + std::to_string(metrics.p50_latency_us) + "us, " +
                  "P95=" + std::to_string(metrics.p95_latency_us) + "us, " +
                  "P99=" + std::to_string(metrics.p99_latency_us) + "us, " +
                  "Gate=" + (metrics.passed_latency_gate ? "PASSED" : "FAILED"));

    return metrics;
}

} // namespace clow::ai
