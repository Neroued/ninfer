#pragma once

#include "qus/model/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qus::text {

enum class OutputMode {
    Clean,
    Raw,
};

struct CliOptions {
    bool help_requested = false;
    std::string weights_path;
    std::string tokenizer_path;
    std::string prompt;
    std::string messages_path;
    int max_new                 = 128;
    std::uint32_t max_context   = 2048;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int device                  = 0;
    OutputMode output_mode      = OutputMode::Clean;
    bool print_token_ids        = false;
    bool use_cuda_graph         = true;
    std::vector<int> stop_token_ids;
};

CliOptions parse_cli(int argc, char** argv);
std::string usage_text(const char* argv0);

} // namespace qus::text
