#pragma once

#include "models/qwen3_5/program/program_impl.h"
#include <chrono>

namespace ninfer::models::qwen3_5::execution {
struct MtpCausalAttentionEnvelopes;
struct DFlashEnvelopes;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

using execution::dimension;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept;

std::int32_t checked_i32(std::uint32_t value, const char* label);

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept;

execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity);

execution::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                            std::uint32_t k);

} // namespace ninfer::models::qwen3_5::detail
