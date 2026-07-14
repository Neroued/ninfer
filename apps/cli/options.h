#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new       = 128;
    std::uint32_t max_context   = 2048;
    std::uint32_t prefill_chunk = 1024;
    int device                  = 0;

    KvCacheStorage kv_cache        = KvCacheStorage::BFloat16;
    std::uint32_t mtp_draft_tokens = 0;
    ProposalHead proposal_head     = ProposalHead::Full;
    bool use_cuda_graph            = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Qwen3 thinking defaults. --greedy replaces these with exact argmax.
    SamplingParameters sampling{
        .temperature = 0.6F, .top_k = 20, .top_p = 0.95F, .min_p = 0.0F, .presence_penalty = 1.0F};
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
