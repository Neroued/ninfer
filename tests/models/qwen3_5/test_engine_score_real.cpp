// CausalScoring through the public Engine on NINFER_TEST_ARTIFACT: option normalization, the
// result shape, finite log-probabilities and fresh State/KV per window. With the argument "tp2" the
// Engine scores on two devices (INT8 KV, as ninfer-perplexity --tp 2), which covers the
// causal-score planning of a model without hyper-connections and, with the 1024-token prefill
// chunks of scoring, the half-chunk prefill overlap of Qwen4Forward::run_plain. Returns 77 without
// the artifact, for tp2 without two devices, and at tp 1 when the artifact does not fit one device.

#include "ninfer/engine.h"
#include "single_device_fit.h"
#include "models/load_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const bool tp2 = argc > 1 && std::strcmp(argv[1], "tp2") == 0;
    int devices    = 0;
    if (tp2 && (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2)) {
        std::cout << "SKIP: tensor parallelism needs two CUDA devices\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache =
        tp2 ? ninfer::KvCacheStorage::Int8Group64 : ninfer::KvCacheStorage::Fp8E4M3Row256;
    if (tp2) {
        options.tp      = 2;
        options.device  = 0;
        options.devices = {0, 1};
    } else if (const auto shortfall = ninfer::test::single_device_shortfall(
                   artifact, ninfer::models::load_options(options))) {
        std::cout << "SKIP: " << *shortfall << '\n';
        return 77;
    }
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != options.kv_cache) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}
