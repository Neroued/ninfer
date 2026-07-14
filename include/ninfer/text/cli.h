#pragma once

#include "ninfer/core/dtype.h"
#include "ninfer/model/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::text {

enum class OutputMode {
    Clean,
    Raw,
};

struct CliOptions {
    bool help_requested = false;
    std::string weights_path;
    std::string prompt;
    std::string messages_path;
    int max_new                 = 128;
    std::uint32_t max_context   = 2048;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int device                  = 0;
    int mtp_draft_tokens        = 0;
    DType kv_dtype              = DType::BF16;
    OutputMode output_mode      = OutputMode::Clean;
    bool print_token_ids        = false;
    bool use_cuda_graph         = true;
    // Default thinking-ON, matching the Qwen3.6 template (--no-thinking opts out).
    bool enable_thinking   = true;
    bool use_lm_head_draft = false;
    std::vector<int> stop_token_ids;
    // Sampler defaults match the Qwen3 thinking recommendation so the CLI decodes
    // like real usage; --greedy forces exact argmax (temperature 0) for parity.
    // The default seed is fixed so demo runs are reproducible unless overridden.
    float temperature       = 0.6f;
    float top_p             = 0.95f;
    int top_k               = 20;
    float presence_penalty  = 1.0f;
    float frequency_penalty = 0.0f;
    std::uint64_t seed      = 0;
    bool greedy             = false;
};

CliOptions parse_cli(int argc, char** argv);
std::string usage_text(const char* argv0);

} // namespace ninfer::text
