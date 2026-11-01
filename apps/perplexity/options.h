#pragma once

#include "product/config/options.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace ninfer::perplexity {

struct Options {
    bool help_requested = false;
    product::LaunchOptions launch =
        product::default_launch_options(product::EntryPoint::Perplexity);
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context = 4096;
    std::uint32_t stride  = 2048;
    bool quick            = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text();

} // namespace ninfer::perplexity
