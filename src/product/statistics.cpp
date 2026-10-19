#include "product/statistics.h"

#include <nlohmann/json.hpp>
#include <cmath>

namespace ninfer::product {
namespace {
using Json = nlohmann::json;

Json allocations_json(const MemoryAllocations& m) {
    return {{"device_weights_bytes", m.device_weights_bytes},
            {"host_weights_bytes", m.host_weights_bytes},
            {"registered_weights_bytes", m.registered_weights_bytes},
            {"persistent_device_bytes", m.persistent_device_bytes},
            {"workspace_device_bytes", m.workspace_device_bytes},
            {"io_pinned_bytes", m.io_pinned_bytes},
            {"host_context_bytes", m.host_context_bytes},
            {"ple_device_bytes", m.ple_device_bytes},
            {"ple_pinned_bytes", m.ple_pinned_bytes},
            {"ple_host_bytes", m.ple_host_bytes},
            {"expert_device_bytes", m.expert_device_bytes},
            {"expert_pinned_bytes", m.expert_pinned_bytes},
            {"expert_host_bytes", m.expert_host_bytes},
            {"device_bytes", m.device_bytes()},
            {"host_bytes", m.host_bytes()}};
}

Json budget_json(const RankMemoryBudget& b) {
    Json graphs = nullptr;
    if (b.graphs) {
        const auto& g = *b.graphs;
        graphs        = {{"rebuild_estimate_bytes", g.rebuild_estimate_bytes},
                         {"preparation_increment_bytes", g.preparation_increment_bytes},
                         {"calibration_increment_bytes", g.calibration_increment_bytes},
                         {"calibration_retained_bytes", g.calibration_retained_bytes}};
    }
    return {{"limit_bytes", b.limit_bytes},
            {"runtime_reservation_bytes", b.runtime_reservation_bytes},
            {"minimum_runtime_reservation_bytes", b.minimum_runtime_reservation_bytes},
            {"bytes_per_page_group", b.bytes_per_page_group},
            {"available_after_weights_bytes", b.available_after_weights_bytes},
            {"available_after_startup_bytes", b.available_after_startup_bytes},
            {"headroom_bytes", b.headroom_bytes},
            {"planned_slack_bytes", b.planned_slack_bytes},
            {"retained_bytes", b.retained_bytes},
            {"graph_calibration", std::move(graphs)}};
}
} // namespace

double seconds(std::uint64_t ns) noexcept { return static_cast<double>(ns) * 1e-9; }

double rate(std::uint64_t count, double elapsed) noexcept {
    return std::isfinite(elapsed) && elapsed > 0 ? static_cast<double>(count) / elapsed : 0;
}

ExecutionWindow execution_window(const ExecutionCounters& before, const ExecutionCounters& after) {
    return {after.units - before.units, after.rows - before.rows,
            after.input_tokens - before.input_tokens,
            after.scheduled_positions - before.scheduled_positions,
            seconds(after.exposure.total.total_ns - before.exposure.total.total_ns)};
}

GenerationRates generation_rates(const RequestStats& stats, std::uint64_t output_tokens) {
    return {rate(stats.computed_prefill_tokens, stats.timings.prefill_seconds),
            rate(output_tokens ? output_tokens - 1 : 0, stats.timings.generation_wall_seconds)};
}

Json duration_json(const DurationCounter& value) {
    return {{"count", value.count}, {"total_ns", value.total_ns}, {"maximum_ns", value.maximum_ns}};
}

Json execution_json(const ExecutionExposure& value) {
    return {{"result", duration_json(value.result)},
            {"resolution", duration_json(value.resolution)},
            {"retirement", duration_json(value.retirement)},
            {"total", duration_json(value.total)}};
}

Json execution_json(const ExecutionSample& value) {
    const auto optional = [](auto v) -> Json { return v ? Json(*v) : Json(nullptr); };
    return {{"kind", static_cast<int>(value.work.kind)},
            {"rows", value.work.rows},
            {"input_tokens", value.work.input_tokens},
            {"scheduled_positions", value.work.scheduled_positions},
            {"result_ns", optional(value.latency.result_ns)},
            {"resolution_ns", optional(value.latency.resolution_ns)},
            {"retirement_ns", optional(value.latency.retirement_ns)},
            {"elapsed_ns", value.latency.total_ns}};
}

Json request_execution_json(const RequestStats& value) {
    constexpr const char* names[] = {"prefill", "decode", "replay", "control", "score", "context"};
    constexpr const char* intervals[] = {"result_stream", "commit_stream", "prefill_stream",
                                         "vision_stream"};
    Json out;
    for (std::size_t i = 0; i < value.executions.size(); ++i)
        out[names[i]] = execution_json(value.executions[i]);
    for (std::size_t i = 0; i < value.max_rank_gpu.size(); ++i)
        out["max_rank_gpu_intervals"][intervals[i]] = duration_json(value.max_rank_gpu[i]);
    return out;
}

MemoryAllocations sum_allocations(std::span<const RankSnapshot> ranks) {
    MemoryAllocations total;
    for (const auto& rank : ranks) {
        const auto& a = rank.memory.allocations;
        total.device_weights_bytes += a.device_weights_bytes;
        total.host_weights_bytes += a.host_weights_bytes;
        total.registered_weights_bytes += a.registered_weights_bytes;
        total.persistent_device_bytes += a.persistent_device_bytes;
        total.workspace_device_bytes += a.workspace_device_bytes;
        total.io_pinned_bytes += a.io_pinned_bytes;
        total.host_context_bytes += a.host_context_bytes;
        total.ple_device_bytes += a.ple_device_bytes;
        total.ple_pinned_bytes += a.ple_pinned_bytes;
        total.ple_host_bytes += a.ple_host_bytes;
        total.expert_device_bytes += a.expert_device_bytes;
        total.expert_pinned_bytes += a.expert_pinned_bytes;
        total.expert_host_bytes += a.expert_host_bytes;
    }
    return total;
}

Json resource_json(const RankSnapshot& rank) {
    const auto& m = rank.memory;
    Json out{{"rank", rank.rank},
             {"node", rank.node},
             {"hardware", rank.hardware},
             {"sample_sequence", rank.stamp.sequence},
             {"sampled_ns", rank.stamp.sampled_ns},
             {"received_ns", rank.received_ns},
             {"allocations", allocations_json(m.allocations)},
             {"kv_payload_bytes", m.kv_payload_bytes},
             {"workspace",
              {{"allocation_cursor_bytes", m.workspace.allocation_cursor_bytes},
               {"required_peak_bytes", m.workspace.required_peak_bytes},
               {"measurement_peak_bytes", m.workspace.measurement_peak_bytes}}},
             {"physical_memory", nullptr},
             {"vision_workspace", nullptr},
             {"experts", nullptr},
             {"ple", nullptr}};
    if (rank.physical_memory) {
        const auto& p          = *rank.physical_memory;
        out["physical_memory"] = {{"cuda_total_bytes", p.cuda_total_bytes},
                                  {"cuda_free_bytes", p.cuda_free_bytes},
                                  {"system_total_bytes", p.system_total_bytes},
                                  {"system_available_bytes", p.system_available_bytes},
                                  {"shared_physical_memory", p.shared_physical_memory},
                                  {"sampled_ns", p.sampled_ns}};
    }
    if (m.vision) {
        const auto& v           = *m.vision;
        out["vision_workspace"] = {{"handoff_in_persistent", v.handoff_in_persistent},
                                   {"aggregate_prompt_tokens", v.aggregate_prompt_tokens},
                                   {"max_item_tokens", v.max_item_tokens},
                                   {"general_capacity_bytes", v.general_capacity_bytes},
                                   {"encode_required_bytes", v.encode_required_bytes},
                                   {"handoff_offset_bytes", v.handoff_offset_bytes},
                                   {"handoff_capacity_bytes", v.handoff_capacity_bytes},
                                   {"handoff_active_bytes", v.handoff_active_bytes},
                                   {"handoff_peak_bytes", v.handoff_peak_bytes}};
    }
    if (m.experts) {
        const auto& e  = *m.experts;
        out["experts"] = {{"cache_bytes", e.cache_bytes},
                          {"fixed_bytes", e.fixed_bytes},
                          {"window_bytes", e.window_bytes},
                          {"window_count", e.window_count},
                          {"used_bytes", e.used_bytes},
                          {"target_bytes", e.target_bytes},
                          {"ready_bytes", e.ready_bytes},
                          {"cpu_workers", e.cpu_workers},
                          {"fixed_experts_per_layer", e.fixed_experts_per_layer},
                          {"target_experts_per_layer", e.target_experts_per_layer},
                          {"ready_experts_per_layer", e.ready_experts_per_layer},
                          {"uploaded_bytes", e.uploaded_bytes},
                          {"prefetched_bytes", e.prefetched_bytes}};
    }
    if (m.ple) {
        const auto& p = *m.ple;
        out["ple"]    = {{"rows", p.rows},
                         {"cache_hits", p.cache_hits},
                         {"shared_rows", p.shared_rows},
                         {"shared_pages", p.shared_pages},
                         {"reads", p.reads},
                         {"read_bytes", p.read_bytes},
                         {"uploaded_bytes", p.uploaded_bytes}};
    }
    return out;
}

Json startup_resources_json(const EngineDescription& description, const StartupReport& startup,
                            const EngineSnapshot& snapshot) {
    Json ranks = Json::array();
    for (std::size_t i = 0; i < description.ranks.size(); ++i) {
        const auto& rank = description.ranks[i];
        Json out{{"rank", rank.rank},
                 {"node", rank.node},
                 {"device", rank.device},
                 {"hardware", rank.hardware},
                 {"shared_physical_memory", rank.shared_physical_memory},
                 {"budget", budget_json(startup.ranks.at(i))}};
        for (const auto& sample : snapshot.ranks)
            if (sample.rank == rank.rank) out["resources"] = resource_json(sample);
        ranks.push_back(std::move(out));
    }
    return {{"ranks", std::move(ranks)},
            {"total_allocations", allocations_json(sum_allocations(snapshot.ranks))}};
}

} // namespace ninfer::product
