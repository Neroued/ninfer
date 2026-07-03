#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kTextPayloadBytes = 16378329088ULL;
constexpr std::size_t kMtpPayloadBytes = 451267584ULL;
constexpr std::size_t kDefaultArenaSlackBytes = 256ULL * kMiB;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

bool enough_free_memory(std::size_t bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cout << "SKIP: cudaMemGetInfo failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    if (free_bytes < bytes) {
        std::cout << "SKIP: not enough free GPU memory; need " << bytes << ", free "
                  << free_bytes << '\n';
        return false;
    }
    return true;
}

void print_stats(const char* label, const qus::EngineMemoryStats& stats) {
    std::cout << "ENGINE_REAL " << label << " loaded_payload="
              << stats.q5090_loaded_payload_bytes << " weight_capacity="
              << stats.weights.capacity_bytes << " weight_used=" << stats.weights.used_bytes
              << " tensor_count=" << stats.q5090_tensor_count
              << " quant_count=" << stats.q5090_quant_count << '\n';
}

qus::EngineMemoryStats load_real_engine(const std::filesystem::path& weights_path,
                                        int mtp_draft_tokens) {
    qus::EngineOptions options;
    options.device = 0;
    options.max_ctx = 128;
    options.work_bytes = 64ULL * kMiB;
    options.mtp_draft_tokens = mtp_draft_tokens;

    qus::Engine engine(options);
    engine.load(weights_path.string());
    const qus::EngineMemoryStats stats = engine.memory_stats();
    cudaDeviceSynchronize();
    return stats;
}

int expect_stats(const qus::EngineMemoryStats& stats, std::size_t expected_loaded_payload,
                 std::size_t expected_weight_capacity, const char* label) {
    int failures = 0;
    failures += stats.loaded ? 0 : fail(std::string(label) + " did not report loaded");
    failures += stats.weights.present ? 0 : fail(std::string(label) + " missing weight arena");
    failures += stats.weights.capacity_bytes == expected_weight_capacity
                    ? 0
                    : fail(std::string(label) + " weight arena capacity mismatch");
    failures += stats.weights.used_bytes >= expected_loaded_payload
                    ? 0
                    : fail(std::string(label) + " weight arena used below loaded payload");
    failures += stats.weights.used_bytes <= stats.weights.capacity_bytes
                    ? 0
                    : fail(std::string(label) + " weight arena used above capacity");
    failures += stats.q5090_loaded_payload_bytes == expected_loaded_payload
                    ? 0
                    : fail(std::string(label) + " loaded payload mismatch");
    failures += stats.q5090_tensor_count == 1164
                    ? 0
                    : fail(std::string(label) + " tensor count mismatch");
    failures += stats.q5090_quant_count > 0 ? 0 : fail(std::string(label) + " quant count zero");
    return failures;
}

int expect_k5_8192_memory_budget(const std::filesystem::path& weights_path) {
    if (!enough_free_memory(24ULL * kGiB)) { return 0; }

    qus::EngineOptions options;
    options.device           = 0;
    options.max_ctx          = 8192;
    options.mtp_draft_tokens = 5;

    qus::Engine engine(options);
    engine.load(weights_path.string());
    const qus::EngineMemoryStats stats = engine.memory_stats();
    cudaDeviceSynchronize();

    const std::size_t total_capacity =
        stats.weights.capacity_bytes + stats.cache.capacity_bytes + stats.workspace.capacity_bytes;
    std::cout << "ENGINE_REAL mtp=5 max_ctx=8192"
              << " weight_capacity=" << stats.weights.capacity_bytes
              << " cache_capacity=" << stats.cache.capacity_bytes
              << " workspace_capacity=" << stats.workspace.capacity_bytes
              << " total_capacity=" << total_capacity << '\n';

    int failures = 0;
    failures += stats.loaded ? 0 : fail("k=5 max_ctx=8192 did not report loaded");
    failures += stats.weights.present ? 0 : fail("k=5 max_ctx=8192 missing weight arena");
    failures += stats.cache.present ? 0 : fail("k=5 max_ctx=8192 missing cache arena");
    failures += stats.workspace.present ? 0 : fail("k=5 max_ctx=8192 missing workspace arena");
    failures += total_capacity < 32ULL * kGiB
                    ? 0
                    : fail("k=5 max_ctx=8192 arena capacity exceeds 32 GiB");
    failures += stats.cache.used_bytes < stats.cache.capacity_bytes
                    ? 0
                    : fail("k=5 max_ctx=8192 cache used exhausted capacity");
    failures += stats.workspace.used_bytes < stats.workspace.capacity_bytes
                    ? 0
                    : fail("k=5 max_ctx=8192 workspace used exhausted capacity");
    return failures;
}

struct GenerateRun {
    std::vector<int> tokens;
    qus::EngineMtpStats mtp;
};

GenerateRun generate_real_run(const std::filesystem::path& weights_path, int mtp_draft_tokens,
                              bool strict, int max_new_tokens, std::uint32_t max_ctx = 32,
                              std::vector<int> stop_token_ids = {},
                              std::vector<int> prompt = std::vector<int>{1}) {
    qus::EngineOptions options;
    options.device                = 0;
    options.max_ctx               = max_ctx;
    options.mtp_draft_tokens      = mtp_draft_tokens;
    options.mtp_strict_sequential = strict;
    options.use_cuda_graph        = false;
    options.stop_token_ids        = std::move(stop_token_ids);

    qus::Engine engine(options);
    engine.load(weights_path.string());
    GenerateRun out;
    out.tokens = engine.generate(prompt, max_new_tokens);
    out.mtp    = engine.mtp_stats();
    return out;
}

std::vector<int> foundation_prompt_ids() {
    return {
        248045, 846,    198,    96220,  109841, 96125,  12654,  220,    103733,
        1510,   18479,  87682,  1494,   40798,  44646,  3709,   96719,  4960,
        198,    16,     13,     220,    99486,  95814,  1697,   50246,  18078,
        4891,   3709,   99448,  110167, 99516,  96932,  95793,  98162,  97889,
        113282, 24178,  198,    17,     13,     220,    109066, 96983,  119808,
        96348,  114727, 95726,  110167, 24178,  198,    18,     13,     220,
        99488,  96656,  109293, 96492,  96766,  110280, 95726,  110167, 101831,
        24178,  198,    19,     13,     220,    110334, 117443, 98682,  3709,
        96172,  111654, 97889,  1992,   220,    99449,  137029, 1710,   248046,
        198,    248045, 74455,  198,    248068, 271,    248069, 271,
    };
}

int expect_strict_mtp_matches_target(const std::filesystem::path& weights_path) {
    int failures = 0;
    constexpr int kMaxNew = 3;

    const GenerateRun baseline = generate_real_run(weights_path, 0, false, kMaxNew);
    std::vector<std::vector<int>> strict_tokens(qus::model::kMaxMtpDraftTokens + 1);

    for (int k = 1; k <= qus::model::kMaxMtpDraftTokens; ++k) {
        const GenerateRun strict = generate_real_run(weights_path, k, true, kMaxNew);
        strict_tokens[static_cast<std::size_t>(k)] = strict.tokens;
        if (strict.tokens != baseline.tokens) {
            failures += fail("strict MTP tokens differ from MTP-off baseline for k=" +
                             std::to_string(k));
        }
        failures += strict.mtp.rounds > 0
                        ? 0
                        : fail("strict MTP did not record MTP rounds for k=" + std::to_string(k));
        failures += strict.mtp.fallback_steps == 0
                        ? 0
                        : fail("strict MTP used fallback decode steps for k=" + std::to_string(k));
    }

    for (int k = 1; k <= qus::model::kMaxMtpDraftTokens; ++k) {
        const GenerateRun batched = generate_real_run(weights_path, k, false, kMaxNew);
        if (batched.tokens != strict_tokens[static_cast<std::size_t>(k)]) {
            failures += fail("batched MTP tokens differ from strict MTP for k=" +
                             std::to_string(k));
        }
        failures += batched.mtp.rounds > 0
                        ? 0
                        : fail("batched MTP did not record MTP rounds for k=" + std::to_string(k));
        failures += batched.mtp.fallback_steps == 0
                        ? 0
                        : fail("batched MTP used fallback decode steps for k=" + std::to_string(k));
    }
    return failures;
}

int expect_stop_and_capacity_controls(const std::filesystem::path& weights_path) {
    int failures = 0;
    constexpr int kMaxNew = 3;

    const GenerateRun baseline = generate_real_run(weights_path, 0, false, kMaxNew);
    if (baseline.tokens.size() < 3) {
        return fail("baseline did not produce enough tokens for stop/capacity checks");
    }

    const int stop_token = baseline.tokens[1];
    const GenerateRun stopped =
        generate_real_run(weights_path, 5, false, kMaxNew, 32, std::vector<int>{stop_token});
    const std::vector<int> expected_stopped{baseline.tokens[0], baseline.tokens[1]};
    if (stopped.tokens != expected_stopped) {
        failures += fail("MTP stop-token truncation did not stop at the in-round token");
    }
    failures += stopped.mtp.rounds > 0 ? 0 : fail("MTP stop check did not run an MTP round");
    failures += stopped.mtp.fallback_steps == 0
                    ? 0
                    : fail("MTP stop check unexpectedly used fallback steps");

    const GenerateRun fallback = generate_real_run(weights_path, 5, false, kMaxNew, 8);
    if (fallback.tokens != baseline.tokens) {
        failures += fail("capacity fallback tokens differ from MTP-off baseline");
    }
    failures += fallback.mtp.rounds == 0 ? 0 : fail("capacity fallback recorded MTP rounds");
    failures += fallback.mtp.fallback_steps > 0
                    ? 0
                    : fail("capacity fallback did not record fallback steps");

    return failures;
}

int expect_near_full_prefill_uses_decode_fallback(const std::filesystem::path& weights_path) {
    const std::vector<int> prompt{1, 2, 3, 4, 5, 6, 7};
    const GenerateRun baseline = generate_real_run(weights_path, 0, false, 2, 8, {}, prompt);
    const GenerateRun mtp      = generate_real_run(weights_path, 5, false, 2, 8, {}, prompt);

    int failures = 0;
    if (mtp.tokens != baseline.tokens) {
        failures += fail("near-full MTP fallback tokens differ from MTP-off baseline");
    }
    failures += mtp.mtp.rounds == 0 ? 0 : fail("near-full MTP fallback recorded MTP rounds");
    failures += mtp.mtp.fallback_steps > 0
                    ? 0
                    : fail("near-full MTP fallback did not record fallback steps");
    return failures;
}

int expect_generate_discards_pending_overshoot(const std::filesystem::path& weights_path) {
    qus::EngineOptions options;
    options.device           = 0;
    options.max_ctx          = 128;
    options.mtp_draft_tokens = 5;
    options.use_cuda_graph   = false;

    qus::Engine engine(options);
    engine.load(weights_path.string());
    const std::vector<int> generated = engine.generate(foundation_prompt_ids(), 2);
    if (generated.size() != 2) { return fail("MTP generate did not truncate to max_new=2"); }

    const qus::EngineMtpStats after_generate = engine.mtp_stats();
    (void)engine.decode_step();
    const qus::EngineMtpStats after_next_step = engine.mtp_stats();

    return after_next_step.rounds > after_generate.rounds ||
                   after_next_step.fallback_steps > after_generate.fallback_steps
               ? 0
               : fail("decode_step after generate drained stale pending sampled tokens");
}

} // namespace

int main() {
    const std::filesystem::path root(QUS_SOURCE_DIR);
    const std::filesystem::path weights_path =
        root / "out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus";
    if (!std::filesystem::exists(weights_path)) {
        std::cout << "SKIP: real q5090 file not present\n";
        return 0;
    }

    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device for real Engine load\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    const std::size_t file_size = std::filesystem::file_size(weights_path);
    const std::size_t expected_weight_capacity =
        file_size + kDefaultArenaSlackBytes + kMtpPayloadBytes;
    if (!enough_free_memory(expected_weight_capacity + kGiB)) { return 0; }

    int failures = 0;
    {
        const qus::EngineMemoryStats stats = load_real_engine(weights_path, 0);
        print_stats("mtp=0", stats);
        failures += expect_stats(stats, kTextPayloadBytes, expected_weight_capacity, "mtp=0");
    }
    {
        const qus::EngineMemoryStats stats = load_real_engine(weights_path, 1);
        print_stats("mtp=1", stats);
        failures += expect_stats(stats, kTextPayloadBytes + kMtpPayloadBytes,
                                 expected_weight_capacity, "mtp=1");
    }
    failures += expect_k5_8192_memory_budget(weights_path);
    failures += expect_strict_mtp_matches_target(weights_path);
    failures += expect_stop_and_capacity_controls(weights_path);
    failures += expect_near_full_prefill_uses_decode_fallback(weights_path);
    failures += expect_generate_discards_pending_overshoot(weights_path);
    return failures == 0 ? 0 : fail("real Engine load test failed");
}
