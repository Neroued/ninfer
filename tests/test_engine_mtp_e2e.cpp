#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kGiB                = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMtpPayloadBytes    = 451267584ULL;
constexpr std::size_t kTextPayloadBytes   = 16378329088ULL;
constexpr std::size_t kVisionPayloadBytes = 295719424ULL;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

bool enough_free_memory(std::size_t bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    const cudaError_t err   = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cout << "SKIP: cudaMemGetInfo failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    if (free_bytes < bytes) {
        std::cout << "SKIP: not enough free GPU memory; need " << bytes << ", free " << free_bytes
                  << '\n';
        return false;
    }
    return true;
}

std::filesystem::path real_weights_path() {
    const std::filesystem::path root(QUS_SOURCE_DIR);
    return root / "out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus";
}

std::vector<int> foundation_prompt_ids() {
    return {
        248045, 846,    198,    96220,  109841, 96125,  12654,  220,    103733, 1510,
        18479,  87682,  1494,   40798,  44646,  3709,   96719,  4960,   198,    16,
        13,     220,    99486,  95814,  1697,   50246,  18078,  4891,   3709,   99448,
        110167, 99516,  96932,  95793,  98162,  97889,  113282, 24178,  198,    17,
        13,     220,    109066, 96983,  119808, 96348,  114727, 95726,  110167, 24178,
        198,    18,     13,     220,    99488,  96656,  109293, 96492,  96766,  110280,
        95726,  110167, 101831, 24178,  198,    19,     13,     220,    110334, 117443,
        98682,  3709,   96172,  111654, 97889,  1992,   220,    99449,  137029, 1710,
        248046, 198,    248045, 74455,  198,    248068, 271,    248069, 271,
    };
}

struct Run {
    std::vector<int> tokens;
    qus::EngineMtpStats mtp;
};

Run generate(const std::filesystem::path& weights, qus::EngineOptions options,
             const std::vector<int>& prompt, int max_new_tokens) {
    options.device = 0;
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
    options.use_cuda_graph   = false;
    const Run mtp            = generate(weights, options, foundation_prompt_ids(), 8);

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
    options.use_cuda_graph   = false;
    const Run mtp            = generate(weights, options, {1, 2, 3, 4, 5, 6, 7}, 2);

    int failures = 0;
    failures += mtp.tokens.size() == 2 ? 0 : fail("capacity fallback token count mismatch");
    failures += mtp.mtp.rounds == 0 ? 0 : fail("capacity fallback recorded MTP rounds");
    failures += mtp.mtp.fallback_steps > 0 ? 0 : fail("capacity fallback did not record fallback");
    return failures;
}

int scenario_fallback_after_accept(const std::filesystem::path& weights) {
    const std::vector<int> prompt = foundation_prompt_ids();
    qus::EngineOptions options;
    options.max_ctx          = static_cast<std::uint32_t>(prompt.size() + 10);
    options.mtp_draft_tokens = 5;
    options.use_cuda_graph   = false;
    const Run mtp            = generate(weights, options, prompt, 8);

    int failures = 0;
    failures += mtp.tokens.size() == 8 ? 0 : fail("fallback-after-accept token count mismatch");
    failures +=
        mtp.mtp.rounds == 1 ? 0 : fail("fallback-after-accept did not run exactly one MTP round");
    failures += mtp.mtp.accepted_tokens > 0
                    ? 0
                    : fail("fallback-after-accept first round accepted no draft tokens");
    failures +=
        mtp.mtp.fallback_steps > 0 ? 0 : fail("fallback-after-accept did not record fallback");
    return failures;
}

int scenario_stop_truncation(const std::filesystem::path& weights) {
    qus::EngineOptions probe_options;
    probe_options.max_ctx          = 32;
    probe_options.mtp_draft_tokens = 5;
    probe_options.use_cuda_graph   = false;
    const Run probe                = generate(weights, probe_options, {1}, 3);
    if (probe.tokens.size() < 2) { return fail("stop probe token count mismatch"); }

    qus::EngineOptions options;
    options.max_ctx          = 32;
    options.mtp_draft_tokens = 5;
    options.use_cuda_graph   = false;
    options.stop_token_ids   = {probe.tokens[1]};
    const Run mtp            = generate(weights, options, {1}, 3);

    int failures = 0;
    failures += mtp.tokens == std::vector<int>({probe.tokens[0], probe.tokens[1]})
                    ? 0
                    : fail("stop scenario did not truncate at stop token");
    failures += mtp.mtp.rounds > 0 ? 0 : fail("stop scenario did not record MTP rounds");
    failures += mtp.mtp.fallback_steps == 0 ? 0 : fail("stop scenario used fallback");
    return failures;
}

// Drives one conversation turn to completion with greedy decode, honoring stop tokens the same way
// the text runner does. When use_cache is true it goes through the engine-level prefix cache.
std::vector<int> drive_turn(qus::Engine& engine, const std::vector<int>& prompt, int max_new,
                            const std::vector<int>& stop_ids, bool use_cache) {
    std::vector<int> out;
    if (max_new <= 0) { return out; }
    const auto is_stop = [&](int token) {
        return std::find(stop_ids.begin(), stop_ids.end(), token) != stop_ids.end();
    };
    // This scenario appends full (unstripped) generations, so it exercises the reuse-E append path;
    // pass the boundary at the prompt end (snapshot taken, no chunk split, generation unchanged).
    int token = use_cache ? engine.prefill_cached(prompt, static_cast<std::uint32_t>(prompt.size()))
                          : engine.prefill(prompt);
    out.push_back(token);
    if (is_stop(token)) { return out; }
    while (static_cast<int>(out.size()) < max_new) {
        token = engine.decode_step();
        out.push_back(token);
        if (is_stop(token)) { break; }
    }
    return out;
}

// Core correctness gate for engine-level prefix caching (plan whitelist 4). A two-turn greedy
// conversation must produce identical token ids whether the second turn re-prefills the whole
// prompt (cache OFF) or reuses the resident KV + GDN state and prefills only the new suffix
// (cache ON). Covers MTP-off and MTP-on. Turn 1 ends on a committed stop token (the reusable
// case) so turn 2 actually takes the append path.
int scenario_multiturn_prefix_cache(const std::filesystem::path& weights) {
    const std::vector<int> p1          = foundation_prompt_ids();
    const std::vector<int> user_suffix = {198, 248045, 74455, 198, 248068, 271};
    constexpr int kTurnNew             = 8;
    int failures                       = 0;
    for (int mtp : {0, 3}) {
        const std::string tag = mtp > 0 ? "mtp-on" : "mtp-off";

        // Probe the natural greedy continuation to pick a real token as the stop id, so turn 1
        // ends on a committed stop rather than a mid-round max-token cut.
        qus::EngineOptions probe_opt;
        probe_opt.max_ctx          = 256;
        probe_opt.mtp_draft_tokens = mtp;
        probe_opt.use_cuda_graph   = false;
        const Run probe            = generate(weights, probe_opt, p1, kTurnNew);
        if (probe.tokens.size() < 4) {
            failures += fail("multiturn probe too short (" + tag + ")");
            continue;
        }

        qus::EngineOptions options = probe_opt;
        options.stop_token_ids     = {probe.tokens[3]};

        // Baseline: full re-prefill each turn (cache OFF).
        qus::Engine base(options);
        base.load(weights.string());
        const std::vector<int> g1 = drive_turn(base, p1, kTurnNew, options.stop_token_ids, false);
        std::vector<int> p2       = p1;
        p2.insert(p2.end(), g1.begin(), g1.end());
        p2.insert(p2.end(), user_suffix.begin(), user_suffix.end());
        const std::vector<int> g2 = drive_turn(base, p2, kTurnNew, options.stop_token_ids, false);

        // Cached: one resident engine reuses the prefix across turns (cache ON).
        qus::Engine cached(options);
        cached.load(weights.string());
        const std::vector<int> c1 = drive_turn(cached, p1, kTurnNew, options.stop_token_ids, true);
        std::vector<int> p2c      = p1;
        p2c.insert(p2c.end(), c1.begin(), c1.end());
        p2c.insert(p2c.end(), user_suffix.begin(), user_suffix.end());
        const std::vector<int> c2 = drive_turn(cached, p2c, kTurnNew, options.stop_token_ids, true);

        failures += (c1 == g1) ? 0 : fail("multiturn prefix cache turn 1 differs (" + tag + ")");
        failures += (c2 == g2) ? 0 : fail("multiturn prefix cache turn 2 differs (" + tag + ")");
    }
    return failures;
}

// Regression for the reproduced garbage-output bug (plan whitelist 6). A content_boundary strictly
// inside the prompt caps a prefill chunk to end exactly at the boundary; the chunk loop must
// advance by the processed length, not the nominal chunk size, or it silently drops [t0+len,
// t0+chunk) and, when the boundary is near the end, never computes logits. This scenario also
// exercises branch-2 partial reuse (turn 2 shares only p1[0:boundary] then diverges), so cache-ON
// must equal an independent full (cache-OFF) prefill token-for-token, and turn 2 must actually
// reuse `boundary` resident tokens.
int scenario_partial_reuse_parity(const std::filesystem::path& weights) {
    const std::vector<int> p1    = foundation_prompt_ids();
    const std::uint32_t boundary = static_cast<std::uint32_t>(p1.size() / 2);
    std::vector<int> p2(p1.begin(), p1.begin() + static_cast<std::ptrdiff_t>(boundary));
    for (int i = 0; i < 12; ++i) {
        p2.push_back(1000 + i);
    } // divergent suffix after the shared prefix
    constexpr int kNew = 8;

    const auto run_from = [&](qus::Engine& eng, const std::vector<int>& prompt, bool cache,
                              std::uint32_t cb) {
        std::vector<int> out;
        int tok = cache ? eng.prefill_cached(prompt, cb) : eng.prefill(prompt);
        out.push_back(tok);
        while (static_cast<int>(out.size()) < kNew) { out.push_back(eng.decode_step()); }
        return out;
    };

    int failures = 0;
    for (int mtp : {0, 3}) {
        const std::string tag = mtp > 0 ? "mtp-on" : "mtp-off";
        qus::EngineOptions options;
        options.max_ctx          = 256;
        options.mtp_draft_tokens = mtp;
        options.use_cuda_graph   = false;

        // Cache OFF: independent full prefills (prefill() resets state between turns).
        qus::Engine base(options);
        base.load(weights.string());
        const std::vector<int> g1_off = run_from(base, p1, false, 0);
        const std::vector<int> g2_off = run_from(base, p2, false, 0);

        // Cache ON: turn 1 caps a chunk at `boundary` and snapshots GDN there; turn 2 reuses that
        // boundary via branch-2 rewind and re-prefills only the divergent suffix.
        qus::Engine cached(options);
        cached.load(weights.string());
        const std::vector<int> g1_on = run_from(cached, p1, true, boundary);
        const std::vector<int> g2_on =
            run_from(cached, p2, true, static_cast<std::uint32_t>(p2.size()));
        const std::uint32_t hit2 = cached.last_prefix_cache_hit();

        failures +=
            (g1_on == g1_off) ? 0 : fail("partial reuse turn 1 parity differs (" + tag + ")");
        failures +=
            (g2_on == g2_off) ? 0 : fail("partial reuse turn 2 parity differs (" + tag + ")");
        failures +=
            (hit2 == boundary)
                ? 0
                : fail("partial reuse turn 2 did not take the branch-2 boundary (" + tag + ")");
    }
    return failures;
}

int scenario_graph_parity(const std::filesystem::path& weights) {
    const std::vector<int> prompt = foundation_prompt_ids();

    qus::EngineOptions eager;
    eager.max_ctx          = 256;
    eager.mtp_draft_tokens = 5;
    eager.use_cuda_graph   = false;
    const Run a            = generate(weights, eager, prompt, 24);

    qus::EngineOptions graph;
    graph.max_ctx          = 256;
    graph.mtp_draft_tokens = 5;
    graph.use_cuda_graph   = true;
    const Run b            = generate(weights, graph, prompt, 24);

    int failures = 0;
    failures += a.tokens == b.tokens ? 0 : fail("graph/eager MTP token streams differ");
    failures += a.mtp.accepted_tokens == b.mtp.accepted_tokens
                    ? 0
                    : fail("graph/eager accepted-token counts differ");
    return failures;
}

int scenario_prefill_chunk_parity(const std::filesystem::path& weights) {
    const std::vector<int> seed = foundation_prompt_ids();
    std::vector<int> prompt;
    prompt.reserve(257);
    while (prompt.size() < 257) {
        const std::size_t remaining = 257 - prompt.size();
        prompt.insert(prompt.end(), seed.begin(),
                      seed.begin() + static_cast<std::ptrdiff_t>(std::min(remaining, seed.size())));
    }

    qus::EngineOptions chunked;
    chunked.max_ctx          = 384;
    chunked.prefill_chunk    = 128;
    chunked.mtp_draft_tokens = 3;
    chunked.use_cuda_graph   = false;
    const Run a              = generate(weights, chunked, prompt, 8);

    qus::EngineOptions single = chunked;
    single.prefill_chunk      = 512;
    const Run b               = generate(weights, single, prompt, 8);

    int failures = 0;
    failures += a.tokens == b.tokens ? 0 : fail("128/512 prefill chunk token streams differ");
    failures += a.mtp.rounds == b.mtp.rounds ? 0 : fail("128/512 MTP round counts differ");
    failures += a.mtp.draft_tokens == b.mtp.draft_tokens
                    ? 0
                    : fail("128/512 MTP draft-token counts differ");
    failures += a.mtp.accepted_tokens == b.mtp.accepted_tokens
                    ? 0
                    : fail("128/512 MTP accepted-token counts differ");
    failures += a.mtp.fallback_steps == b.mtp.fallback_steps
                    ? 0
                    : fail("128/512 MTP fallback counts differ");
    return failures;
}

int run_scenario(std::string_view scenario, const std::filesystem::path& weights) {
    if (scenario == "batched") { return scenario_batched(weights); }
    if (scenario == "capacity_fallback") { return scenario_capacity_fallback(weights); }
    if (scenario == "fallback_after_accept") { return scenario_fallback_after_accept(weights); }
    if (scenario == "stop_truncation") { return scenario_stop_truncation(weights); }
    if (scenario == "multiturn_prefix_cache") { return scenario_multiturn_prefix_cache(weights); }
    if (scenario == "partial_reuse_parity") { return scenario_partial_reuse_parity(weights); }
    if (scenario == "graph_parity") { return scenario_graph_parity(weights); }
    if (scenario == "prefill_chunk_parity") { return scenario_prefill_chunk_parity(weights); }
    std::cerr << "unknown scenario: " << scenario << '\n';
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: qus_engine_mtp_e2e_test "
                     "<batched|capacity_fallback|fallback_after_accept|stop_truncation|"
                     "multiturn_prefix_cache|partial_reuse_parity|graph_parity|"
                     "prefill_chunk_parity>\n";
        return 2;
    }
    const std::filesystem::path weights = real_weights_path();
    if (!std::filesystem::exists(weights)) {
        std::cout << "SKIP: real q5090 file not present\n";
        return 0;
    }

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device for real Engine load\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    if (!enough_free_memory(kTextPayloadBytes + kMtpPayloadBytes + kVisionPayloadBytes + kGiB)) {
        return 0;
    }

    int failures = 0;
    if (argc == 2) {
        failures = run_scenario(argv[1], weights);
        if (failures == 2) { return 2; }
        if (failures != 0) { return fail("MTP E2E scenario failed"); }
        std::cout << "OK MTP E2E scenario " << argv[1] << '\n';
    } else {
        for (std::string_view scenario :
             {"batched", "capacity_fallback", "fallback_after_accept", "stop_truncation",
              "multiturn_prefix_cache", "partial_reuse_parity", "graph_parity",
              "prefill_chunk_parity"}) {
            const int scenario_failures = run_scenario(scenario, weights);
            if (scenario_failures != 0) { failures += scenario_failures; }
        }
        if (failures != 0) { return fail("MTP E2E scenarios failed"); }
        std::cout << "OK MTP E2E scenarios\n";
    }
    return 0;
}
