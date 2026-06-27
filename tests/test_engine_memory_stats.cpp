#include "qus/core/device.h"
#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_u32(std::uint32_t actual, std::uint32_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_absent(const qus::ArenaMemoryStats& stats, std::string_view label) {
    int failures = 0;
    failures += !stats.present ? 0 : fail(std::string(label) + " present");
    failures += expect_size(stats.capacity_bytes, 0, std::string(label) + " capacity");
    failures += expect_size(stats.used_bytes, 0, std::string(label) + " used");
    failures += expect_size(stats.peak_used_bytes, 0, std::string(label) + " peak");
    return failures;
}

int expect_present(const qus::ArenaMemoryStats& stats, std::string_view label,
                   bool require_used) {
    int failures = 0;
    failures += stats.present ? 0 : fail(std::string(label) + " absent");
    failures += stats.capacity_bytes > 0 ? 0 : fail(std::string(label) + " zero capacity");
    if (require_used) {
        failures += stats.used_bytes > 0 ? 0 : fail(std::string(label) + " zero used");
    }
    failures += stats.peak_used_bytes >= stats.used_bytes
                    ? 0
                    : fail(std::string(label) + " peak smaller than used");
    return failures;
}

int expect_peak_reset(const qus::ArenaMemoryStats& stats, std::string_view label) {
    if (!stats.present) { return fail(std::string(label) + " absent after reset"); }
    return stats.peak_used_bytes == stats.used_bytes
               ? 0
               : fail(std::string(label) + " peak was not reset to used");
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_engine_memory_stats_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --profile model-bind --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

int test_unloaded_stats_do_not_need_cuda() {
    qus::EngineOptions options;
    options.device       = 7;
    options.max_ctx      = 123;
    options.weight_bytes = 4096;
    options.cache_bytes  = 8192;
    options.work_bytes   = 16ULL * kMiB;

    qus::Engine engine(options);
    int failures = 0;
    const qus::EngineMemoryStats stats = engine.memory_stats();
    failures += !stats.loaded ? 0 : fail("unloaded stats reported loaded");
    failures += stats.device == 7 ? 0 : fail("unloaded stats device mismatch");
    failures += expect_u32(stats.max_context, 123, "unloaded stats max_context");
    failures += expect_u32(stats.position, 0, "unloaded stats position");
    failures += expect_absent(stats.weights, "unloaded weights");
    failures += expect_absent(stats.cache, "unloaded cache");
    failures += expect_absent(stats.workspace, "unloaded workspace");
    failures += expect_size(stats.q5090_loaded_payload_bytes, 0, "unloaded loaded payload");
    failures += expect_size(stats.q5090_tensor_count, 0, "unloaded tensor count");
    failures += expect_size(stats.q5090_quant_count, 0, "unloaded quant count");

    engine.reset_memory_peaks();
    const qus::EngineMemoryStats after_reset = engine.memory_stats();
    failures += expect_absent(after_reset.weights, "unloaded weights after reset");
    failures += expect_absent(after_reset.cache, "unloaded cache after reset");
    failures += expect_absent(after_reset.workspace, "unloaded workspace after reset");
    return failures;
}

int test_loaded_stats_with_cuda() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    const std::filesystem::path fixture = make_fixture();
    qus::EngineOptions options;
    options.device     = 0;
    options.max_ctx    = 4;
    options.work_bytes = 64ULL * kMiB;

    qus::Engine engine(options);
    engine.load(fixture.string());

    int failures = 0;
    const qus::EngineMemoryStats stats = engine.memory_stats();
    failures += stats.loaded ? 0 : fail("loaded stats did not report loaded");
    failures += stats.device == 0 ? 0 : fail("loaded stats device mismatch");
    failures += expect_u32(stats.max_context, 4, "loaded stats max_context");
    failures += expect_u32(stats.position, 0, "loaded stats position");
    failures += expect_present(stats.weights, "loaded weights", true);
    failures += expect_present(stats.cache, "loaded cache", true);
    failures += expect_present(stats.workspace, "loaded workspace", false);
    failures += expect_size(stats.workspace.capacity_bytes, 64ULL * kMiB, "workspace capacity");
    failures += stats.q5090_loaded_payload_bytes > 0 ? 0 : fail("loaded payload bytes zero");
    failures += stats.q5090_tensor_count > 0 ? 0 : fail("q5090 tensor count zero");
    failures += stats.q5090_quant_count > 0 ? 0 : fail("q5090 quant count zero");

    engine.reset_memory_peaks();
    const qus::EngineMemoryStats after_reset = engine.memory_stats();
    failures += expect_peak_reset(after_reset.weights, "weights after reset");
    failures += expect_peak_reset(after_reset.cache, "cache after reset");
    failures += expect_peak_reset(after_reset.workspace, "workspace after reset");
    failures += expect_size(after_reset.q5090_loaded_payload_bytes, stats.q5090_loaded_payload_bytes,
                            "loaded payload stable after reset");
    failures += expect_size(after_reset.q5090_tensor_count, stats.q5090_tensor_count,
                            "tensor count stable after reset");
    failures += expect_size(after_reset.q5090_quant_count, stats.q5090_quant_count,
                            "quant count stable after reset");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_unloaded_stats_do_not_need_cuda();
    failures += test_loaded_stats_with_cuda();
    return failures == 0 ? 0 : fail("engine memory stats test failed");
}
