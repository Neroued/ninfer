#include "models/qwen3_5/program/internal.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

RankExecutionObservation RankProgramImpl::retirement_observation() const noexcept {
    RankExecutionObservation out;
    const auto peak            = std::max(resources.work.peak_used(), workspace_peak);
    out.expert_upload_bytes    = resources.moe ? resources.moe->uploaded_bytes() : 0;
    out.workspace_peak_bytes   = std::max(workspace_lifetime_peak, peak);
    out.measurement_peak_bytes = peak;
    return out;
}

RankMemoryUsage RankProgramImpl::observe_resources() const {
    const auto& weights = parameters.model.storage_stats();
    RankMemoryUsage out;
    auto& a                    = out.allocations;
    a.device_weights_bytes     = weights.device_capacity_bytes;
    a.host_weights_bytes       = weights.host_capacity_bytes;
    a.registered_weights_bytes = weights.registered_host_bytes;
    a.context_device_bytes     = resources.context.size();
    a.persistent_device_bytes  = resources.persistent.size();
    a.workspace_device_bytes   = resources.workspace_storage.size();
    a.io_pinned_bytes          = resources.frame.host_storage.size() + context_counts.size() +
                        compute_actions->pinned_bytes() + context_actions->pinned_bytes();
    a.host_context_bytes = resources.host_context.size();
    if (resources.moe) {
        const auto& plan      = resources.moe->description();
        a.expert_device_bytes = plan.device_bytes;
        a.expert_pinned_bytes = plan.bridge_pinned_bytes;
        a.expert_host_bytes   = plan.cpu_workspace_bytes;
    }
    const auto peak = std::max(resources.work.peak_used(), workspace_peak);
    out.workspace   = {resources.work.used(), std::max(workspace_lifetime_peak, peak), peak};
    if (resources.vision) {
        out.vision = resources.vision->observe();
        a.persistent_device_bytes += resources.vision->capacity().output_bytes;
        a.io_pinned_bytes += resources.vision->capacity().encoder.host_staging_bytes;
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
        out.experts                 = std::move(summary);
    }
    return out;
}

void RankProgramImpl::reset_measurement_peaks() noexcept {
    workspace_lifetime_peak =
        std::max({workspace_lifetime_peak, resources.work.peak_used(), workspace_peak});
    resources.work.reset_peak();
    workspace_peak = 0;
    if (resources.vision) resources.vision->reset_peaks();
}

} // namespace ninfer::models::qwen3_5::detail
