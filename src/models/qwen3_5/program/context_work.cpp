#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/rebuild_work.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::uint64_t elapsed_ns(Clock::time_point started) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::size_t context_resource_index(runtime::ContextResourceClass resource) {
    switch (resource) {
    case runtime::ContextResourceClass::State:
        return 0;
    case runtime::ContextResourceClass::MainKV:
        return 1;
    case runtime::ContextResourceClass::BackendKV:
        return 2;
    }
    throw std::logic_error("unknown context resource class");
}

runtime::PrefillWork validated_rebuild_work(runtime::PrefillWork work, std::uint32_t frontier) {
    if (work.tokens != frontier) {
        throw std::logic_error("checkpoint rebuild work does not match its frontier");
    }
    return work;
}

void validate_long_anchor_ordinals(std::span<const LongAnchorCheckpoint> anchors,
                                   std::size_t capacity) {
    if (anchors.size() > capacity) {
        throw std::logic_error("long-anchor set exceeds configured capacity");
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        const std::uint32_t ordinal = anchors[index].ordinal;
        if (ordinal == 0 || ordinal > capacity) {
            throw std::logic_error("long-anchor ordinal is outside configured slots");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (anchors[previous].ordinal == ordinal) {
                throw std::logic_error("long-anchor ordinals are not unique");
            }
        }
    }
}

void advance_rebuild_work(SequenceState& sequence, std::uint32_t frontier,
                          std::uint32_t prefill_chunk) {
    runtime_support::advance_segmented_rebuild_work(
        sequence.rebuild_work, sequence.rebuild_tail_begin, sequence.execution_frontier, frontier,
        prefill_chunk);
}

std::optional<qwen3_5::TargetKVRequirement>
retained_requirement_after_drops(const qwen3_5::ContinuationSummary& summary,
                                 std::span<const runtime::CheckpointRef> dropped) noexcept {
    if (dropped.empty()) { return std::nullopt; }
    qwen3_5::TargetKVRequirement requirement;
    std::size_t found  = 0;
    bool surviving     = false;
    const auto include = [&](const qwen3_5::CheckpointSummary& checkpoint) {
        const auto match = std::find(dropped.begin(), dropped.end(), checkpoint.ref);
        if (match != dropped.end()) {
            ++found;
            return;
        }
        surviving = true;
        requirement.main_frontier =
            std::max(requirement.main_frontier, checkpoint.required_kv.main_frontier);
        requirement.backend_frontier =
            std::max(requirement.backend_frontier, checkpoint.required_kv.backend_frontier);
        requirement.main_pages =
            std::max(requirement.main_pages, checkpoint.required_kv.main_pages);
        requirement.backend_pages =
            std::max(requirement.backend_pages, checkpoint.required_kv.backend_pages);
    };
    if (summary.endpoint) { include(*summary.endpoint); }
    if (summary.rewrite) { include(*summary.rewrite); }
    for (const qwen3_5::CheckpointSummary& anchor : summary.long_anchors) { include(anchor); }
    if (found != dropped.size() || !surviving || requirement.main_frontier == 0 ||
        requirement.main_pages == 0) {
        return std::nullopt;
    }
    return requirement;
}

runtime::ContextTransferRequirement
state_transfer_requirement(const StateImageHostLayout& layout,
                           runtime::ContextTransferDirection direction, bool dflash_local_only) {
    return runtime::ContextTransferRequirement{
        .resource   = runtime::ContextResourceClass::State,
        .direction  = direction,
        .units      = 1,
        .page_count = 0,
        .work       = dflash_local_only ? dflash_local_transfer_work(layout)
                                        : state_image_transfer_work(layout),
    };
}

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs) {
    const TransferWork work = direction == runtime::ContextTransferDirection::DeviceToDevice
                                  ? plan_device_kv_copy_work(layout, pages)
                                  : plan_host_kv_transfer_work(layout, pages, contiguous_runs);
    return runtime::ContextTransferRequirement{
        .resource   = resource,
        .direction  = direction,
        .units      = work.payload_bytes,
        .page_count = pages,
        .work       = work,
    };
}

bool pressure_state_drops_host(qwen3_5::detail::PressureStateDecision change) noexcept {
    return change == qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate ||
           change == qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate ||
           change == qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate;
}

bool pressure_state_demotes(qwen3_5::detail::PressureStateDecision change) noexcept {
    return change == qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost ||
           change == qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost ||
           change == qwen3_5::detail::PressureStateDecision::DemoteSharedToHost;
}

std::optional<StateImageHandle> pressure_state_source(qwen3_5::detail::PressureStateDecision change,
                                                      const SequenceState* sequence,
                                                      const SharedPrefixState* shared) {
    switch (change) {
    case qwen3_5::detail::PressureStateDecision::None:
        return std::nullopt;
    case qwen3_5::detail::PressureStateDecision::DropEndpointDeviceDuplicate:
    case qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost:
    case qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate:
        if (sequence == nullptr) {
            throw std::logic_error("private pressure State action targets a shared owner");
        }
        return sequence->state.read;
    case qwen3_5::detail::PressureStateDecision::DropRewriteDeviceDuplicate:
    case qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost:
    case qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate:
        if (sequence == nullptr || !sequence->rewrite_state) {
            throw std::logic_error("pressure rewrite StateImage disappeared");
        }
        return *sequence->rewrite_state;
    case qwen3_5::detail::PressureStateDecision::DropSharedDeviceDuplicate:
    case qwen3_5::detail::PressureStateDecision::DemoteSharedToHost:
    case qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate:
        if (shared == nullptr) {
            throw std::logic_error("shared pressure State action targets a private owner");
        }
        return shared->state;
    }
    throw std::logic_error("pressure StateImage action is invalid");
}

detail::PhysicalResources checked_resource_sum(detail::PhysicalResources left,
                                               detail::PhysicalResources right) {
    const auto add_u32 = [](std::uint32_t a, std::uint32_t b, const char* label) {
        if (b > std::numeric_limits<std::uint32_t>::max() - a) { throw std::overflow_error(label); }
        return static_cast<std::uint32_t>(a + b);
    };
    if (right.host.kv_bytes > std::numeric_limits<std::size_t>::max() - left.host.kv_bytes) {
        throw std::overflow_error("Qwen3.5 Host KV resource sum overflow");
    }
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes  = add_u32(left.device.active_lanes, right.device.active_lanes,
                                         "Qwen3.5 active-lane resource sum overflow"),
                .state_slots   = add_u32(left.device.state_slots, right.device.state_slots,
                                         "Qwen3.5 StateImage resource sum overflow"),
                .main_kv_pages = add_u32(left.device.main_kv_pages, right.device.main_kv_pages,
                                         "Qwen3.5 Main KV resource sum overflow"),
                .backend_kv_pages =
                    add_u32(left.device.backend_kv_pages, right.device.backend_kv_pages,
                            "Qwen3.5 Backend KV resource sum overflow"),
            },
        .host =
            {
                .state_slots = add_u32(left.host.state_slots, right.host.state_slots,
                                       "Qwen3.5 Host StateImage resource sum overflow"),
                .kv_bytes    = left.host.kv_bytes + right.host.kv_bytes,
            },
    };
}

detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,
                                                      detail::PhysicalResources removed) {
    if (removed.device.active_lanes > value.device.active_lanes ||
        removed.device.state_slots > value.device.state_slots ||
        removed.device.main_kv_pages > value.device.main_kv_pages ||
        removed.device.backend_kv_pages > value.device.backend_kv_pages ||
        removed.host.state_slots > value.host.state_slots ||
        removed.host.kv_bytes > value.host.kv_bytes) {
        throw std::logic_error("Qwen3.5 resource subtraction underflow");
    }
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes     = value.device.active_lanes - removed.device.active_lanes,
                .state_slots      = value.device.state_slots - removed.device.state_slots,
                .main_kv_pages    = value.device.main_kv_pages - removed.device.main_kv_pages,
                .backend_kv_pages = value.device.backend_kv_pages - removed.device.backend_kv_pages,
            },
        .host =
            {
                .state_slots = value.host.state_slots - removed.host.state_slots,
                .kv_bytes    = value.host.kv_bytes - removed.host.kv_bytes,
            },
    };
}

detail::PhysicalResources positive_resource_difference(detail::PhysicalResources value,
                                                       detail::PhysicalResources removed) noexcept {
    const auto positive_u32 = [](std::uint32_t left, std::uint32_t right) {
        return left > right ? left - right : 0U;
    };
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes =
                    positive_u32(value.device.active_lanes, removed.device.active_lanes),
                .state_slots = positive_u32(value.device.state_slots, removed.device.state_slots),
                .main_kv_pages =
                    positive_u32(value.device.main_kv_pages, removed.device.main_kv_pages),
                .backend_kv_pages =
                    positive_u32(value.device.backend_kv_pages, removed.device.backend_kv_pages),
            },
        .host =
            {
                .state_slots = positive_u32(value.host.state_slots, removed.host.state_slots),
                .kv_bytes    = value.host.kv_bytes > removed.host.kv_bytes
                                   ? value.host.kv_bytes - removed.host.kv_bytes
                                   : 0U,
            },
    };
}

execution::MtpCausalAttentionEnvelopes mtp_causal_attention_envelopes(std::uint32_t max_frontier,
                                                                      std::uint32_t k,
                                                                      std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    execution::MtpCausalAttentionEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

execution::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                            std::uint32_t k) {
    (void)min_frontier;
    return execution::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

} // namespace ninfer::models::qwen3_5::detail
