// Qwen4Exp validation driver (T20): runs the public Engine on a fixed token sequence.
//
//   ninfer_qwen4_tf_driver score ART IDS [--context N] [--chunk N] [--kv bf16|int8|fp8|nvfp4|k8v4]
//   [--tp 1|2]
//       [--out F]
//       CausalScoring of the whole sequence (first target 1). Writes the per-token log-probs
//       (JSON). With NINFER_QWEN4_TFDUMP=<dir> the Program also writes logits / hidden taps.
//   ninfer_qwen4_tf_driver score-windows ART LIST [same options as score]
//       LIST = text file, one IDS path per line. One Engine scores every window in order (each
//       window is an independent sequence from position 0); one JSON line per window. With
//       NINFER_QWEN4_TFDUMP_SEQUENCES=1 each window's dump goes to its own seqNNN subdirectory.
//   ninfer_qwen4_tf_driver generate ART IDS --max-new N [--context N] [--chunk N]
//       [--no-cuda-graph] [--kv ...] [--tp 1|2] [--repeat R] [--reuse-prefix] [--mtp K] [--prompts]
//       [--temperature T --seed S] [--top-k K] [--top-p P] [--out F]
//       Generation (temperature 0 = greedy, no penalties, no stop tokens) from the sequence as
//       prompt, R times on one Engine (one JSON line per run and prompt: prompt + generated ids,
//       prefill and decode timing, MTP rounds/acceptance); with --reuse-prefix the context cache is
//       on and later runs may reuse the prompt's prefix. With NINFER_QWEN4_TFDUMP=<dir> the Program
//       writes the logits of every decode round. --mtp K decodes with MTP (K draft tokens, window
//       K+1) and reports the acceptance per draft position; --mtp-min-p P (T41) verifies only the
//       proposals whose chain probability (--mtp-window-rule chain, default) or own probability
//       (step) is >= P; --prompts reads one prompt per line of
//       IDS (one Engine for all); --temperature > 0 samples (rejection sampling under MTP).
//       --concurrency N (2-8, greedy only): max_concurrency N and N x max_context of KV; each run
//       submits every prompt at once (the Engine batches up to N of them), prints them in order.
// Elastic KV (T64, both modes): [--elastic] [--kv-grow TOKENS] [--kv-budget-gib G]
//   [--elastic-columns N (with --slots)]: the KV pool is backed on demand and the expert cache
//   gives up / takes back whole columns of slots as the KV grows / shrinks
//   (EngineOptions::elastic_kv).
// Routed experts (both modes): [--experts offload|mapped] [--slots N] [--slot-headroom-mib N]
//   [--cpu-threads N] [--insert-cap N] [--admit F] [--half-life F] [--evict-tau F] [--device-reads
//   N] [--embedding-host] [--kv-ram-active]
//   [--prefetch N] [--prefetch-min-score F] [--prefetch-first-layer L] [--prefetch-gbps G]
//   [--warm-start | --no-warm-start] (T25; warm start is on by default).
//   [--miss-split auto|off|<fraction>|count:<n>] [--split-path staged|zc] [--split-rank 0|1]
//   [--split-min N] [--split-max N]
//   [--split-costs
//   cpu_fixed,cpu_expert,cpu_column,cpu_read_slowdown,gpu_fixed,gpu_expert,gpu_column]. The
//   resolved slot count and the device memory budget are printed to stderr (memory: ...).
// IDS is a text file of whitespace-separated token ids.
#include "ninfer/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::vector<ninfer::TokenId>> read_prompts(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::vector<std::vector<ninfer::TokenId>> prompts;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::vector<ninfer::TokenId> ids;
        long long value = 0;
        while (row >> value) { ids.push_back(static_cast<ninfer::TokenId>(value)); }
        if (ids.size() >= 2) { prompts.push_back(std::move(ids)); }
    }
    if (prompts.empty()) { throw std::runtime_error("no prompt of at least two ids"); }
    return prompts;
}

std::vector<ninfer::TokenId> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::vector<ninfer::TokenId> ids;
    long long value = 0;
    while (in >> value) { ids.push_back(static_cast<ninfer::TokenId>(value)); }
    if (ids.size() < 2) { throw std::runtime_error("need at least two token ids"); }
    return ids;
}

ninfer::KvCacheStorage parse_kv(std::string_view text) {
    if (text == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (text == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return ninfer::KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return ninfer::KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid --kv");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: " << argv[0] << " score|generate ART IDS [options]\n";
            return 2;
        }
        const std::string mode = argv[1];
        ninfer::EngineOptions options;
        options.artifact_path = argv[2];
        bool many_prompts     = false;
        for (int i = 4; i < argc; ++i) {
            many_prompts = many_prompts || std::string_view(argv[i]) == "--prompts";
        }
        std::vector<std::string> window_paths;
        std::vector<std::vector<ninfer::TokenId>> windows;
        if (mode == "score-windows") {
            std::ifstream list(argv[3]);
            if (!list) { throw std::runtime_error(std::string("cannot read ") + argv[3]); }
            for (std::string line; std::getline(list, line);) {
                if (line.empty()) { continue; }
                window_paths.push_back(line);
                windows.push_back(read_ids(line));
            }
            if (windows.empty()) { throw std::runtime_error("empty window list"); }
        }
        const std::vector<std::vector<ninfer::TokenId>> prompts =
            !windows.empty() ? std::vector<std::vector<ninfer::TokenId>>{windows.front()}
            : many_prompts   ? read_prompts(argv[3])
                             : std::vector<std::vector<ninfer::TokenId>>{read_ids(argv[3])};
        const auto& ids       = prompts.front();
        std::uint32_t context = 0;
        std::uint32_t max_new = 0;
        std::uint32_t repeat  = 1;
        std::uint32_t lanes   = 1;
        bool reuse_prefix     = false;
        std::uint32_t mtp     = 0;
        float temperature     = 0.0F;
        std::int32_t top_k    = -1; // T107: < 0 = model default
        float top_p           = -1.0F;
        std::uint64_t seed    = 0;
        std::string out_path;
        for (int i = 4; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(arg) + " needs a value");
                }
                return argv[++i];
            };
            if (arg == "--context") {
                context = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--chunk") {
                options.prefill_chunk = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--kv") {
                options.kv_cache = parse_kv(next());
            } else if (arg == "--max-new") {
                max_new = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--repeat") {
                // generate: run the same request(s) N times in one Engine (speed samples).
                repeat = static_cast<std::uint32_t>(std::stoul(next()));
                if (repeat == 0) { throw std::invalid_argument("--repeat must be positive"); }
            } else if (arg == "--tp") {
                // Tensor-parallel width; 2 runs on devices 0 and 1 (the visible ones).
                options.tp = std::stoi(next());
                if (options.tp == 2) { options.devices = {0, 1}; }
            } else if (arg == "--mtp") {
                mtp = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--mtp-min-p") {
                // T41: adaptive MTP window (0 = fixed K).
                options.speculative.min_draft_probability = std::stof(next());
            } else if (arg == "--draft-shortlist") {
                // T43: proposal heads over a draft token shortlist (file of token IDs).
                options.speculative.draft_shortlist = next();
            } else if (arg == "--draft-rows") {
                options.speculative.draft_shortlist_rows =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--mtp-window-rule") {
                const std::string v = next();
                if (v == "chain") {
                    options.speculative.window_rule = ninfer::MtpWindowRule::Chain;
                } else if (v == "step") {
                    options.speculative.window_rule = ninfer::MtpWindowRule::Step;
                } else {
                    throw std::invalid_argument("--mtp-window-rule must be chain or step");
                }
            } else if (arg == "--concurrency") {
                lanes = static_cast<std::uint32_t>(std::stoul(next()));
                if (lanes == 0 || lanes > 8) {
                    throw std::invalid_argument("--concurrency must be in [1,8]");
                }
            } else if (arg == "--prompts") {
            } else if (arg == "--temperature") {
                temperature = std::stof(next());
            } else if (arg == "--top-k") {
                top_k = std::stoi(next());
            } else if (arg == "--top-p") {
                top_p = std::stof(next());
            } else if (arg == "--seed") {
                seed = std::stoull(next());
            } else if (arg == "--no-cuda-graph") {
                options.use_cuda_graph = false;
            } else if (arg == "--out") {
                out_path = next();
            } else if (arg == "--reuse-prefix") {
                reuse_prefix = true;
            } else if (arg == "--experts") {
                const std::string v = next();
                if (v == "offload") {
                    options.routed_experts.backend = ninfer::RoutedExpertBackend::Offload;
                } else if (v == "mapped") {
                    options.routed_experts.backend = ninfer::RoutedExpertBackend::Mapped;
                } else {
                    throw std::invalid_argument("--experts offload|mapped");
                }
            } else if (arg == "--slots") {
                options.routed_experts.slots_per_layer =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--slot-headroom-mib") {
                options.routed_experts.slot_headroom_bytes = std::stoull(next()) << 20;
            } else if (arg == "--cpu-threads") {
                options.routed_experts.cpu_threads = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--insert-cap") {
                options.routed_experts.insert_cap = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--admit") {
                // Admission threshold (default 2, T54); --admit 0 = plain LRU (the pre-T54 engine).
                options.routed_experts.admit_threshold = std::stof(next());
            } else if (arg == "--half-life") {
                options.routed_experts.half_life_tokens = std::stof(next());
            } else if (arg == "--evict-tau") {
                // T61: frequency x recency victim, time constant in windows (default 4); 0 = LRU
                // victim.
                options.routed_experts.evict_tau_windows = std::stof(next());
            } else if (arg == "--device-reads") {
                options.routed_experts.device_reads =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--prefetch") {
                // T25 router-ahead prefetch: staging copies per layer (0 = off).
                options.routed_experts.prefetch_slots =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--prefetch-min-score") {
                options.routed_experts.prefetch_min_score = std::stof(next());
            } else if (arg == "--prefetch-first-layer") {
                options.routed_experts.prefetch_first_layer =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--prefetch-gbps") {
                options.routed_experts.prefetch_link_gbps = std::stof(next());
            } else if (arg == "--warm-start") {
                options.routed_experts.warm_start = true;
            } else if (arg == "--no-warm-start") {
                options.routed_experts.warm_start = false;
            } else if (arg == "--stream-min") {
                options.routed_experts.stream_min_tokens =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--stream-cpu") {
                options.routed_experts.stream_cpu_tail = std::stoi(next()) != 0;
            } else if (arg == "--stream-cost") {
                // CPU us per expert / CPU us per token-expert / link us per expert
                const std::string v = next();
                const auto a = v.find('/'), b = v.rfind('/');
                options.routed_experts.stream_cpu_expert_us = std::stof(v.substr(0, a));
                options.routed_experts.stream_cpu_token_us  = std::stof(v.substr(a + 1, b - a - 1));
                options.routed_experts.stream_link_us       = std::stof(v.substr(b + 1));
            } else if (arg == "--prompt-chunk") {
                // T42: largest streamed prompt chunk on lent workspace (0 = off).
                options.routed_experts.prompt_chunk =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--stream-ring") {
                options.routed_experts.stream_ring_slots =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--miss-split") {
                // T18: auto | off | <fraction in [0,1]> | count:<n>
                auto& split         = options.routed_experts.miss_split;
                const std::string v = next();
                if (v == "auto") {
                    split.mode = ninfer::MissSplitMode::Auto;
                } else if (v == "off") {
                    split.mode = ninfer::MissSplitMode::Off;
                } else if (v.rfind("count:", 0) == 0) {
                    split.mode  = ninfer::MissSplitMode::Count;
                    split.count = static_cast<std::uint32_t>(std::stoul(v.substr(6)));
                } else {
                    split.mode     = ninfer::MissSplitMode::Fraction;
                    split.fraction = std::stof(v);
                }
            } else if (arg == "--split-path") {
                const std::string v = next();
                if (v != "staged" && v != "zc") {
                    throw std::invalid_argument("--split-path staged|zc");
                }
                options.routed_experts.miss_split.staged = v == "staged";
            } else if (arg == "--split-rank") {
                // T42, tp 2: which rank computes the GPU share (1 = rank 1, default; 0 = rank 0).
                options.routed_experts.miss_split.peer_share = std::stoi(next()) != 0;
            } else if (arg == "--split-min") {
                options.routed_experts.miss_split.min_misses =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--split-max") {
                options.routed_experts.miss_split.max_reads =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--split-costs") {
                // cpu_fixed,cpu_expert,cpu_column,cpu_read_slowdown,gpu_fixed,gpu_expert,gpu_column
                // (µs)
                const std::string v = next();
                std::vector<float> c;
                std::size_t at = 0;
                while (at <= v.size()) {
                    const auto comma = v.find(',', at);
                    c.push_back(std::stof(
                        v.substr(at, comma == std::string::npos ? std::string::npos : comma - at)));
                    if (comma == std::string::npos) { break; }
                    at = comma + 1;
                }
                if (c.size() != 7) { throw std::invalid_argument("--split-costs needs 7 values"); }
                auto& k = options.routed_experts.miss_split.costs;
                k       = {c[0], c[1], c[2], c[3], c[4], c[5], c[6]};
            } else if (arg == "--embedding-host") {
                options.embedding_host = true;
            } else if (arg == "--kv-ram-active") {
                options.kv_ram_active = true;
            } else if (arg == "--elastic") {
                // T64: elastic KV <-> expert slots (KV backed on demand, the cache shrinks/grows).
                options.elastic_kv.enabled = true;
            } else if (arg == "--kv-grow") {
                options.elastic_kv.grow_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--kv-budget-gib") {
                options.elastic_kv.kv_budget_bytes =
                    static_cast<std::size_t>(std::stod(next()) * 1073741824.0);
            } else if (arg == "--elastic-columns") {
                // With --slots: the elastic columns by hand (else both are resolved).
                options.routed_experts.elastic_columns =
                    static_cast<std::uint32_t>(std::stoul(next()));
            } else {
                throw std::invalid_argument("unknown option " + std::string(arg));
            }
        }
        const auto rounded = [](std::uint64_t n) {
            return static_cast<std::uint32_t>((n + 127) / 128 * 128);
        };
        const auto report_memory = [](const ninfer::Engine& engine) {
            const ninfer::MemorySummary m = engine.memory_summary();
            std::cerr << "memory: weights " << m.weights.capacity_bytes << " B, sequence "
                      << m.sequence.capacity_bytes << " B (expert slots " << m.routed_expert_slots
                      << "/layer = " << m.routed_expert_pool_bytes << " B), workspace "
                      << m.workspace.capacity_bytes << " B, graphs allowance "
                      << m.cuda_graph_allowance_bytes << " observed "
                      << m.cuda_graph_observed_bytes[0] << " B, reservation "
                      << m.runtime_reservation_bytes << " B, available after weights "
                      << m.available_after_weights_bytes << " B, after startup "
                      << m.available_after_startup_bytes << " B, kv " << m.kv_capacity
                      << " tokens, prefill chunk " << m.prefill_chunk << " (lent workspace "
                      << m.routed_expert_prompt_workspace_bytes << " B)\n";
        };
        std::ofstream file;
        std::ostream* out = &std::cout;
        if (!out_path.empty()) {
            file.open(out_path);
            out = &file;
        }
        if (mode == "score-windows") {
            std::size_t longest = 0;
            for (const auto& w : windows) { longest = std::max(longest, w.size()); }
            options.purpose     = ninfer::EnginePurpose::CausalScoring;
            options.max_context = context != 0 ? context : rounded(longest);
            options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
            ninfer::Engine engine(options);
            report_memory(engine);
            for (std::size_t w = 0; w < windows.size(); ++w) {
                const auto start                  = std::chrono::steady_clock::now();
                const std::vector<float> logprobs = engine.score_tokens(windows[w], 1);
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                double nll = 0;
                for (const float v : logprobs) { nll -= v; }
                nll /= static_cast<double>(logprobs.size());
                *out << "{\"mode\":\"score\",\"window\":\"" << window_paths[w]
                     << "\",\"index\":" << w << ",\"tokens\":" << windows[w].size()
                     << ",\"mean_nll\":" << nll << ",\"seconds\":" << seconds << ",\"logprobs\":[";
                for (std::size_t i = 0; i < logprobs.size(); ++i) {
                    *out << (i ? "," : "") << logprobs[i];
                }
                *out << "]}\n";
                out->flush();
                std::cerr << "score[" << w << "] " << window_paths[w] << ": " << windows[w].size()
                          << " tokens, mean NLL " << nll << ", " << seconds << " s\n";
            }
            return 0;
        }
        if (mode == "score") {
            options.purpose     = ninfer::EnginePurpose::CausalScoring;
            options.max_context = context != 0 ? context : rounded(ids.size());
            options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
            ninfer::Engine engine(options);
            report_memory(engine);
            const auto start                  = std::chrono::steady_clock::now();
            const std::vector<float> logprobs = engine.score_tokens(ids, 1);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double nll = 0;
            for (const float v : logprobs) { nll -= v; }
            nll /= static_cast<double>(logprobs.size());
            *out << "{\"mode\":\"score\",\"tokens\":" << ids.size() << ",\"mean_nll\":" << nll
                 << ",\"ppl\":" << std::exp(nll) << ",\"seconds\":" << seconds << ",\"logprobs\":[";
            for (std::size_t i = 0; i < logprobs.size(); ++i) {
                *out << (i ? "," : "") << logprobs[i];
            }
            *out << "]}\n";
            std::cerr << "score: " << ids.size() << " tokens, mean NLL " << nll << ", PPL "
                      << std::exp(nll) << ", " << seconds << " s\n";
            return 0;
        }
        if (mode == "generate") {
            if (max_new == 0) { throw std::invalid_argument("--max-new is required"); }
            if (lanes > 1 && temperature > 0.0F) {
                throw std::invalid_argument("--concurrency is greedy only");
            }
            std::size_t longest = 0;
            for (const auto& prompt : prompts) { longest = std::max(longest, prompt.size()); }
            options.purpose         = ninfer::EnginePurpose::Generation;
            options.max_context     = context != 0 ? context : rounded(longest + max_new + 1 + mtp);
            options.max_concurrency = lanes;
            options.kv_capacity =
                ninfer::KvCapacityPolicy::explicit_capacity(options.max_context * lanes);
            options.context_cache.enabled = reuse_prefix;
            if (reuse_prefix &&
                options.tp == 2) { // T109: as ninfer-serve at tp 2 (no Host tier on rank 1)
                options.context_cache.host_state_slots       = 0;
                options.context_cache.host_kv_capacity_bytes = 0;
            }
            if (mtp != 0) {
                options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
                options.speculative.draft_tokens = mtp;
            }
            ninfer::Engine engine(options);
            report_memory(engine);
            std::size_t index = 0;
            // T121: NINFER_DRIVER_TEMPS = one temperature per prompt line (index across runs;
            // missing = --temperature); NINFER_DRIVER_STOPS = stop at the default stop tokens of
            // the model (as ninfer-serve), not raw max-new.
            std::vector<float> temps;
            if (const char* tp = std::getenv("NINFER_DRIVER_TEMPS"); tp != nullptr && *tp != 0) {
                std::ifstream tin(tp);
                for (float v = 0.0F; tin >> v;) { temps.push_back(v); }
            }
            const bool model_stops = std::getenv("NINFER_DRIVER_STOPS") != nullptr;
            for (std::uint32_t run = 0; run < repeat; ++run) {
                // --concurrency: every prompt of the run is submitted before the first wait.
                std::vector<ninfer::GenerationHandle> handles;
                std::vector<std::chrono::steady_clock::time_point> starts;
                for (std::size_t p = 0; lanes > 1 && p < prompts.size(); ++p) {
                    ninfer::RequestOptions request;
                    request.execution.requested_output_tokens    = max_new;
                    request.execution.allow_prefix_reuse         = reuse_prefix;
                    request.execution.sampling.temperature       = 0.0F;
                    request.execution.sampling.presence_penalty  = 0.0F;
                    request.execution.sampling.frequency_penalty = 0.0F;
                    request.stop.include_model_defaults          = model_stops;
                    request.output.raw                           = true;
                    starts.push_back(std::chrono::steady_clock::now());
                    handles.push_back(
                        engine.submit(engine.prepare_tokens(prompts[p], false), request));
                }
                for (std::size_t p = 0; p < prompts.size(); ++p) {
                    const auto& prompt_ids = prompts[p];
                    ninfer::RequestOptions request;
                    request.execution.requested_output_tokens = max_new;
                    request.execution.allow_prefix_reuse      = reuse_prefix;
                    const float prompt_temperature =
                        index < temps.size() ? temps[index] : temperature;
                    request.execution.sampling.temperature       = prompt_temperature;
                    request.execution.sampling.presence_penalty  = 0.0F;
                    request.execution.sampling.frequency_penalty = 0.0F;
                    // T109: each repeat samples with its own seed (seed + run) so --repeat gives
                    // independent samples.
                    if (prompt_temperature > 0.0F) {
                        request.execution.sampling.seed = seed + run * 100003ULL + index;
                    }
                    if (top_k >= 0) { request.execution.sampling.top_k = top_k; }
                    if (top_p >= 0.0F) { request.execution.sampling.top_p = top_p; }
                    request.stop.include_model_defaults = model_stops;
                    request.output.raw                  = true;
                    const auto start = lanes > 1 ? starts[p] : std::chrono::steady_clock::now();
                    const ninfer::GenerationResult result =
                        lanes > 1
                            ? handles[p].wait()
                            : engine.generate(engine.prepare_tokens(prompt_ids, false), request);
                    const double wall =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                            .count();
                    const auto& g = result.generated_token_ids;
                    // decode_seconds covers the decode rounds after the prefill's first token.
                    const double tokens_per_second =
                        g.size() > 1 && result.timings.decode_seconds > 0
                            ? static_cast<double>(g.size() - 1) / result.timings.decode_seconds
                            : 0.0;
                    const auto& spec = result.speculative;
                    *out << "{\"mode\":\"generate\",\"run\":" << run << ",\"index\":" << index++
                         << ",\"mtp\":" << mtp << ",\"prompt_tokens\":" << prompt_ids.size()
                         << ",\"generated\":" << g.size()
                         << ",\"prefill_seconds\":" << result.timings.prefill_seconds
                         << ",\"prompt_wall_seconds\":" << result.timings.prompt_wall_seconds
                         << ",\"decode_seconds\":" << result.timings.decode_seconds
                         << ",\"generation_wall_seconds\":"
                         << result.timings.generation_wall_seconds << ",\"wall_seconds\":" << wall
                         << ",\"decode_tokens_per_second\":" << tokens_per_second
                         << ",\"tokens_per_window\":"
                         << (spec.rounds != 0 ? 1.0 + static_cast<double>(spec.accepted_tokens) /
                                                          static_cast<double>(spec.rounds)
                                              : 1.0)
                         << ",\"rounds\":" << spec.rounds << ",\"drafted\":" << spec.drafted_tokens
                         << ",\"accepted\":" << spec.accepted_tokens
                         << ",\"fallback_steps\":" << spec.fallback_steps
                         << ",\"accepted_per_position\":[";
                    for (std::size_t i = 0; i < spec.accepted_per_position.size(); ++i) {
                        *out << (i ? "," : "") << spec.accepted_per_position[i];
                    }
                    *out << "],\"ids\":[";
                    bool first = true;
                    for (const auto id : prompt_ids) {
                        *out << (first ? "" : ",") << id;
                        first = false;
                    }
                    for (const auto id : g) { *out << "," << id; }
                    *out << "]}\n";
                    out->flush();
                    std::cerr << "generate[" << run << "/" << index - 1 << "]: " << g.size()
                              << " tokens, prefill " << result.timings.prefill_seconds
                              << " s, decode " << tokens_per_second << " tok/s, wall " << wall
                              << " s";
                    if (spec.rounds != 0) {
                        std::cerr << ", rounds " << spec.rounds << ", accepted "
                                  << spec.accepted_tokens << "/" << spec.drafted_tokens
                                  << ", tokens/window "
                                  << 1.0 + static_cast<double>(spec.accepted_tokens) /
                                               static_cast<double>(spec.rounds);
                    }
                    std::cerr << "\n";
                }
            }
            return 0;
        }
        throw std::invalid_argument("mode must be score or generate");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
