#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
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

std::filesystem::path real_weights_path() {
    const std::filesystem::path root(QUS_SOURCE_DIR);
    return root / "out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus";
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

struct Run {
    std::vector<int> tokens;
    qus::EngineMtpStats mtp;
};

Run generate(const std::filesystem::path& weights, qus::EngineOptions options,
             const std::vector<int>& prompt, int max_new_tokens) {
    options.device         = 0;
    options.use_cuda_graph = false;
    qus::Engine engine(options);
    engine.load(weights.string());
    Run out;
    out.tokens = engine.generate(prompt, max_new_tokens);
    out.mtp    = engine.mtp_stats();
    return out;
}

int scenario_batched(const std::filesystem::path& weights) {
    qus::EngineOptions options;
    options.max_ctx          = 128;
    options.mtp_draft_tokens = 5;
    const Run mtp = generate(weights, options, foundation_prompt_ids(), 8);

    int failures = 0;
    failures += mtp.tokens.size() == 8 ? 0 : fail("batched scenario token count mismatch");
    failures += mtp.mtp.rounds > 0 ? 0 : fail("batched scenario did not record MTP rounds");
    failures += mtp.mtp.draft_tokens > 0 ? 0 : fail("batched scenario did not draft tokens");
    failures += mtp.mtp.accepted_tokens > 0 ? 0 : fail("batched scenario accepted no tokens");
    failures += mtp.mtp.fallback_steps == 0 ? 0 : fail("batched scenario used fallback");
    return failures;
}

int scenario_capacity_fallback(const std::filesystem::path& weights) {
    qus::EngineOptions options;
    options.max_ctx          = 8;
    options.mtp_draft_tokens = 5;
    const Run mtp = generate(weights, options, {1, 2, 3, 4, 5, 6, 7}, 2);

    int failures = 0;
    failures += mtp.tokens.size() == 2 ? 0 : fail("capacity fallback token count mismatch");
    failures += mtp.mtp.rounds == 0 ? 0 : fail("capacity fallback recorded MTP rounds");
    failures += mtp.mtp.fallback_steps > 0 ? 0 : fail("capacity fallback did not record fallback");
    return failures;
}

int scenario_stop_truncation(const std::filesystem::path& weights) {
    qus::EngineOptions probe_options;
    probe_options.max_ctx          = 32;
    probe_options.mtp_draft_tokens = 5;
    const Run probe = generate(weights, probe_options, {1}, 3);
    if (probe.tokens.size() < 2) { return fail("stop probe token count mismatch"); }

    qus::EngineOptions options;
    options.max_ctx          = 32;
    options.mtp_draft_tokens = 5;
    options.stop_token_ids   = {probe.tokens[1]};
    const Run mtp = generate(weights, options, {1}, 3);

    int failures = 0;
    failures += mtp.tokens == std::vector<int>({probe.tokens[0], probe.tokens[1]})
                    ? 0
                    : fail("stop scenario did not truncate at stop token");
    failures += mtp.mtp.rounds > 0 ? 0 : fail("stop scenario did not record MTP rounds");
    failures += mtp.mtp.fallback_steps == 0 ? 0 : fail("stop scenario used fallback");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: qus_engine_mtp_e2e_test <batched|capacity_fallback|stop_truncation>\n";
        return 2;
    }
    const std::filesystem::path weights = real_weights_path();
    if (!std::filesystem::exists(weights)) {
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

    const std::size_t file_size = std::filesystem::file_size(weights);
    if (!enough_free_memory(file_size + kDefaultArenaSlackBytes + kMtpPayloadBytes + kGiB)) {
        return 0;
    }

    const std::string scenario = argc == 2 ? argv[1] : "batched";
    int failures = 0;
    if (scenario == "batched") {
        failures = scenario_batched(weights);
    } else if (scenario == "capacity_fallback") {
        failures = scenario_capacity_fallback(weights);
    } else if (scenario == "stop_truncation") {
        failures = scenario_stop_truncation(weights);
    } else {
        std::cerr << "unknown scenario: " << scenario << '\n';
        return 2;
    }
    if (failures != 0) { return fail("MTP E2E scenario failed"); }
    std::cout << "OK MTP E2E scenario " << scenario << '\n';
    return 0;
}
