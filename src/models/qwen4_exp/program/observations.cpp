#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/device_graphs.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

RankExecutionObservation RankProgramImpl::retirement_observation() const noexcept {
    RankExecutionObservation out;
    const auto peak            = std::max({resources.workspace.peak_used(),
                                resources.vision ? resources.vision->workspace_peak_bytes() : 0,
                                graphs ? graphs->workspace_peak() : 0,
                                device_graphs ? device_graphs->workspace_peak() : 0,
                                commit_graphs ? commit_graphs->workspace_peak() : 0});
    out.expert_upload_bytes    = resources.experts ? resources.experts->uploaded_bytes() : 0;
    out.ple_upload_bytes       = resources.ple_upload ? resources.ple_upload->uploaded_bytes() : 0;
    out.workspace_peak_bytes   = std::max(workspace_lifetime_peak, peak);
    out.measurement_peak_bytes = peak;
    return out;
}

RankMemoryUsage RankProgramImpl::observe_resources() const {
    const auto& plan    = resources.plan;
    const auto& weights = parameters.model.storage_stats();
    RankMemoryUsage out;
    auto& a                    = out.allocations;
    a.device_weights_bytes     = weights.device_capacity_bytes;
    a.host_weights_bytes       = weights.host_capacity_bytes;
    a.registered_weights_bytes = weights.registered_host_bytes;
    a.persistent_device_bytes  = plan.persistent.bytes;
    a.context_device_bytes     = plan.capacity.device_bytes;
    a.workspace_device_bytes   = plan.workspace.bytes;
    a.io_pinned_bytes = plan.pinned_draft_bytes + plan.pinned_io_bytes + plan.pinned_output_bytes +
                        plan.pinned_sampling_bytes + plan.pinned_mtp_bytes +
                        plan.pinned_context_bytes() + plan.pinned_aux_bytes() +
                        plan.pinned_context_counts_bytes();
    a.host_context_bytes = plan.host_context_bytes;
    a.ple_device_bytes   = plan.ple_upload.device_bytes;
    a.ple_pinned_bytes   = plan.ple_upload.pinned_host_bytes;
    a.ple_host_bytes     = plan.ple_prepare_bytes + plan.ple_upload.ordinary_host_bytes;
    if (plan.moe) {
        const auto& experts   = plan.moe->description();
        a.expert_device_bytes = experts.device_bytes;
        a.expert_pinned_bytes = experts.bridge_pinned_bytes;
        a.expert_host_bytes   = experts.cpu_workspace_bytes;
    }
    const auto peak = std::max({resources.workspace.peak_used(),
                                resources.vision ? resources.vision->workspace_peak_bytes() : 0,
                                graphs ? graphs->workspace_peak() : 0,
                                device_graphs ? device_graphs->workspace_peak() : 0,
                                commit_graphs ? commit_graphs->workspace_peak() : 0});
    out.workspace   = {resources.workspace.used(), std::max(workspace_lifetime_peak, peak), peak};
    if (resources.vision) {
        out.vision = resources.vision->observe();
        a.persistent_device_bytes += resources.vision->capacity().output_bytes;
        a.io_pinned_bytes += resources.vision->capacity().encoder.host_staging_bytes;
    }
    if (resources.ple_prepare) {
        const auto stats = resources.ple_prepare->stats();
        out.ple          = PleReadStats{stats.rows,
                               stats.cache_hits,
                               stats.shared_rows,
                               stats.shared_pages,
                               stats.reads,
                               stats.read_bytes,
                               resources.ple_upload->uploaded_bytes()};
    }
    if (!resources.experts) return out;
    const auto cache        = resources.experts->cache_usage();
    const auto& description = plan.moe->description();
    out.experts.emplace();
    auto& summary                 = *out.experts;
    summary.host_weights_bytes    = weights.host_capacity_bytes;
    summary.registered_host_bytes = weights.registered_host_bytes;
    summary.cache_bytes           = description.cache_bytes;
    summary.fixed_bytes           = description.fixed_bytes;
    summary.window_bytes          = summary.cache_bytes - summary.fixed_bytes;
    summary.window_count          = description.window_count;
    summary.used_bytes            = cache.used_bytes;
    summary.target_bytes          = description.target_bytes;
    summary.ready_bytes           = cache.ready_cache_bytes;
    summary.cpu_workers           = description.cpu_workers;
    summary.bridge_device_bytes   = description.bridge_device_bytes;
    summary.bridge_pinned_bytes   = description.bridge_pinned_bytes;
    summary.cpu_workspace_bytes   = description.cpu_workspace_bytes;
    summary.uploaded_bytes        = cache.data_bytes;
    summary.prefetched_bytes      = cache.prefetched_bytes;
    for (const auto& layer : description.layers) {
        summary.fixed_experts_per_layer.push_back(layer.fixed_experts);
        summary.target_experts_per_layer.push_back(layer.cache_experts);
        summary.ready_experts_per_layer.push_back(
            cache.ready_experts_per_bank.at(layer.bank.index));
    }
    return out;
}

void RankProgramImpl::reset_measurement_peaks() noexcept {
    workspace_lifetime_peak = std::max(
        {workspace_lifetime_peak, resources.workspace.peak_used(),
         resources.vision ? resources.vision->workspace_peak_bytes() : 0,
         graphs ? graphs->workspace_peak() : 0, device_graphs ? device_graphs->workspace_peak() : 0,
         commit_graphs ? commit_graphs->workspace_peak() : 0});
    resources.workspace.reset_peak();
    if (graphs) graphs->reset_workspace_peak();
    if (device_graphs) device_graphs->reset_workspace_peak();
    if (commit_graphs) commit_graphs->reset_workspace_peak();
    if (resources.vision) resources.vision->reset_peaks();
}

} // namespace ninfer::models::qwen4_exp::detail
