#pragma once

// Hybrid prefix-cache lookup keys of a prepared prompt (docs/maintainer/hybrid-prefix-cache.md
// §5.1): the chained 64-token block hashes and, for prompts with media, the cumulative Vision key
// each block carries. Preparation computes them once; the Program's quotes and admission read them.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct VisionTokenRange {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    std::uint64_t key   = 0;
};

// Every Vision item's covered token range and identity key, ordered by first token. The key binds
// content, modality, grid, patches, timing and token placement: positions after an item depend
// on its grid, so later text blocks carry it too.
[[nodiscard]] std::vector<VisionTokenRange> vision_ranges(const PreparedPromptData& prompt);

[[nodiscard]] std::uint64_t accumulate_vision(std::uint64_t cumulative, std::uint64_t key) noexcept;

// The block hashes of every full block of the prompt's tokens, and one cumulative Vision key per
// block (empty for a text-only prompt).
void prompt_block_keys(const PreparedPromptData& prompt, std::vector<std::uint64_t>& hashes,
                       std::vector<std::uint64_t>& extras);

} // namespace ninfer::models::qwen3_5::detail
