#include "ninfer/runtime/engine.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

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

int expect_absent(const ninfer::ArenaMemoryStats& stats, std::string_view label) {
    int failures = 0;
    failures += !stats.present ? 0 : fail(std::string(label) + " present");
    failures += expect_size(stats.capacity_bytes, 0, std::string(label) + " capacity");
    failures += expect_size(stats.used_bytes, 0, std::string(label) + " used");
    failures += expect_size(stats.peak_used_bytes, 0, std::string(label) + " peak");
    return failures;
}

int test_unloaded_stats_do_not_need_cuda() {
    ninfer::EngineOptions options;
    options.device      = 7;
    options.max_ctx     = 123;
    options.cache_bytes = 8192;
    options.work_bytes  = 16ULL * kMiB;

    ninfer::Engine engine(options);
    int failures                       = 0;
    const ninfer::EngineMemoryStats stats = engine.memory_stats();
    failures += !stats.loaded ? 0 : fail("unloaded stats reported loaded");
    failures += stats.device == 7 ? 0 : fail("unloaded stats device mismatch");
    failures += expect_u32(stats.max_context, 123, "unloaded stats max_context");
    failures += expect_u32(stats.position, 0, "unloaded stats position");
    failures += stats.kv_dtype == ninfer::DType::BF16 ? 0 : fail("unloaded kv dtype mismatch");
    failures += expect_size(stats.kv_quant_group, 0, "unloaded kv quant group");
    failures += expect_size(stats.kv_cache_payload_bytes, 0, "unloaded kv payload");
    failures += expect_absent(stats.weights, "unloaded weights");
    failures += expect_absent(stats.cache, "unloaded cache");
    failures += expect_absent(stats.workspace, "unloaded workspace");
    failures += expect_size(stats.q5090_loaded_payload_bytes, 0, "unloaded loaded payload");
    failures += expect_size(stats.q5090_tensor_count, 0, "unloaded tensor count");
    failures += expect_size(stats.q5090_quant_count, 0, "unloaded quant count");

    engine.reset_memory_peaks();
    const ninfer::EngineMemoryStats after_reset = engine.memory_stats();
    failures += expect_absent(after_reset.weights, "unloaded weights after reset");
    failures += expect_absent(after_reset.cache, "unloaded cache after reset");
    failures += expect_absent(after_reset.workspace, "unloaded workspace after reset");
    try {
        ninfer::EngineOptions bad_options;
        bad_options.mtp_draft_tokens = -1;
        ninfer::Engine bad_engine(bad_options);
        (void)bad_engine;
        failures += fail("negative mtp_draft_tokens did not throw");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::EngineOptions bad_options;
        bad_options.mtp_draft_tokens = ninfer::model::kMaxMtpDraftTokens + 1;
        ninfer::Engine bad_engine(bad_options);
        (void)bad_engine;
        failures += fail("too large mtp_draft_tokens did not throw");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::EngineOptions bad_options;
        bad_options.stop_token_ids = {ninfer::model::kCfg.vocab};
        ninfer::Engine bad_engine(bad_options);
        (void)bad_engine;
        failures += fail("reserved stop token id did not throw before load");
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main() {
    const int failures = test_unloaded_stats_do_not_need_cuda();
    // Loaded accounting is covered by ninfer_engine_real_file_test with the canonical artifact.
    return failures == 0 ? 0 : fail("engine memory stats test failed");
}
