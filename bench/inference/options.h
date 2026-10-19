#pragma once

#include "product/config/options.h"

namespace ninfer::bench {

enum class InputKind { Prompt, Messages };
enum class OutputFormat { Table, Json, Csv };

struct InputSource {
    InputKind kind = InputKind::Prompt;
    std::string value;
};

struct Options {
    bool help_requested = false;
    product::LaunchOptions launch;
    std::vector<InputSource> inputs;
    std::uint32_t concurrency = 1;
    std::uint32_t repetitions = 3;
    std::uint32_t warmup      = 1;
    bool profile_measured     = false;
    OutputFormat output       = OutputFormat::Table;
    std::filesystem::path output_file;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(std::string_view executable);

} // namespace ninfer::bench
