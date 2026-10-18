#include "models/qwen3_5/program/internal.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

MemorySummary RankProgramImpl::memory_summary() const {
    const auto& weights = parameters.model.storage_stats();
    MemorySummary out;
    out.device                      = device.device().index();
    out.max_context                 = capacity;
    out.prefill_chunk               = prefill_chunk;
    out.kv_capacity                 = kv_capacity;
    out.kv_cache                    = kv_storage;
    out.weights                     = {weights.device_capacity_bytes, weights.device_capacity_bytes,
                                       weights.device_capacity_bytes};
    out.sequence                    = {resources.persistent.size(), resources.persistent.size(),
                                       resources.persistent.size()};
    out.kv_payload_bytes            = kv_payload_bytes;
    out.cuda_graph                  = graph_memory;
    out.host_context_capacity_bytes = resources.host_context.size();
    std::size_t handoff             = 0;
    for (const auto& residency : residencies)
        if (residency.vision) handoff = std::max(handoff, residency.vision->active_handoff_bytes());
    auto used = resources.work.used();
    if (workspace_plan.vision && handoff)
        used = std::max(used, workspace_plan.vision->handoff_offset_bytes + handoff);
    out.workspace                    = {resources.workspace_storage.size(), used,
                                        std::max(resources.work.peak_used(), workspace_peak)};
    out.workspace_logical_peak_bytes = workspace_peak;
    if (workspace_plan.vision) {
        const auto& vision   = *workspace_plan.vision;
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, qwen::kMaximumPromptVisionTokens)),
            .max_item_tokens        = vision.max_merged_tokens,
            .general_capacity_bytes = vision.general_capacity_bytes,
            .encode_peak_bytes      = vision.encode_peak_bytes,
            .handoff_offset_bytes   = vision.handoff_offset_bytes,
            .handoff_capacity_bytes = vision.handoff_capacity_bytes,
            .handoff_active_bytes   = handoff,
            .handoff_peak_bytes     = vision_peak,
        };
    }
    if (resources.moe) {
        const auto& plan = resources.moe->description();
        const auto usage = resources.moe->cache_usage();
        MoeMemorySummary summary;
        summary.host_weights_bytes    = weights.host_capacity_bytes;
        summary.registered_host_bytes = weights.registered_host_bytes;
        summary.cache_bytes           = plan.cache_bytes;
        summary.fixed_bytes           = plan.fixed_bytes;
        summary.window_bytes          = plan.cache_bytes - plan.fixed_bytes;
        summary.window_count          = plan.window_count;
        summary.used_bytes            = usage.used_bytes;
        summary.target_bytes          = plan.target_bytes;
        summary.ready_bytes           = usage.ready_cache_bytes;
        summary.cpu_workers           = plan.cpu_workers;
        for (const auto& layer : plan.layers) {
            summary.fixed_experts_per_layer.push_back(layer.fixed_experts);
            summary.target_experts_per_layer.push_back(layer.cache_experts);
            summary.ready_experts_per_layer.push_back(
                usage.ready_experts_per_bank.at(layer.bank.index));
        }
        summary.bridge_device_bytes = plan.bridge_device_bytes;
        summary.bridge_pinned_bytes = plan.bridge_pinned_bytes;
        summary.cpu_workspace_bytes = plan.cpu_workspace_bytes;
        summary.uploaded_bytes      = resources.moe->uploaded_bytes();
        summary.prefetched_bytes    = usage.prefetched_bytes;
        out.moe                     = std::move(summary);
    }
    return out;
}

void RankProgramImpl::reset_memory_peaks() noexcept {
    resources.work.reset_peak();
    vision_peak = 0;
    for (const auto& residency : residencies)
        if (residency.vision)
            vision_peak = std::max(vision_peak, residency.vision->active_handoff_bytes());
    workspace_peak = resources.work.used();
    if (workspace_plan.vision && vision_peak)
        workspace_peak =
            std::max(workspace_peak, workspace_plan.vision->handoff_offset_bytes + vision_peak);
}

} // namespace ninfer::models::qwen3_5::detail
