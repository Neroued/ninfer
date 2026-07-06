#pragma once

#include "qus/model/config.h"

#include <cstdint>
#include <string>

namespace qus::serve {

struct ServeOptions {
    bool help_requested = false;
    std::string weights_path;
    std::string tokenizer_path;
    std::string host          = "127.0.0.1";
    int port                  = 8080;
    std::string api_key;                       // empty => no auth
    std::string model_id      = "qwen3.6-27b"; // reported by /v1/models
    std::uint32_t max_context = 8192;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int device                = 0;
    int mtp_draft_tokens      = 0;
    bool use_cuda_graph       = true;
    bool use_lm_head_draft    = false;
    bool enable_thinking      = false;  // default thinking mode for the generation prompt
    int default_max_tokens    = 512;    // used when a request omits max_tokens
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string serve_usage_text(const char* argv0);

} // namespace qus::serve
