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

std::size_t context_resource_index(runtime::ContextResourceClass resource);

runtime::PrefillWork validated_rebuild_work(runtime::PrefillWork work, std::uint32_t frontier);

void validate_long_anchor_ordinals(std::span<const LongAnchorCheckpoint> anchors,
                                   std::size_t capacity);

void advance_rebuild_work(SequenceState& sequence, std::uint32_t frontier,
                          std::uint32_t prefill_chunk);

std::optional<qwen3_5::TargetKVRequirement>
retained_requirement_after_drops(const qwen3_5::ContinuationSummary& summary,
                                 std::span<const runtime::CheckpointRef> dropped) noexcept;

runtime::ContextTransferRequirement
state_transfer_requirement(const StateImageHostLayout& layout,
                           runtime::ContextTransferDirection direction,
                           bool dflash_local_only = false);

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs = 1);

bool pressure_state_drops_host(qwen3_5::detail::PressureStateDecision change) noexcept;

bool pressure_state_demotes(qwen3_5::detail::PressureStateDecision change) noexcept;

std::optional<StateImageHandle> pressure_state_source(qwen3_5::detail::PressureStateDecision change,
                                                      const SequenceState* sequence,
                                                      const SharedPrefixState* shared);

detail::PhysicalResources checked_resource_sum(detail::PhysicalResources left,
                                               detail::PhysicalResources right);

detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,
                                                      detail::PhysicalResources removed);

detail::PhysicalResources positive_resource_difference(detail::PhysicalResources value,
                                                       detail::PhysicalResources removed) noexcept;

execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity);

execution::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                            std::uint32_t k);

} // namespace ninfer::models::qwen3_5::detail
