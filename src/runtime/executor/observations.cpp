#include "runtime/executor/protocol.h"

#include <tuple>
#include <type_traits>

namespace ninfer::runtime::protocol {
namespace {
namespace wire = distributed::wire;

// Diagnostics use a fixed field schema, independent of ABI padding and size_t layout.
template <class T>
auto fields(T& v) {
    using Value = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<Value, GraphMemorySummary>)
        return std::tie(v.rebuild_estimate_bytes, v.preparation_increment_bytes,
                        v.calibration_increment_bytes, v.calibration_retained_bytes);
    else if constexpr (std::is_same_v<Value, VisionMemorySummary>)
        return std::tie(v.budget_bytes, v.max_window_features, v.encoder_workspace_bytes,
                        v.output_capacity_bytes, v.input_staging_bytes, v.output_active_bytes,
                        v.output_peak_bytes);
    else if constexpr (std::is_same_v<Value, MoeMemorySummary>)
        return std::tie(v.host_weights_bytes, v.registered_host_bytes, v.cache_bytes, v.fixed_bytes,
                        v.window_bytes, v.window_count, v.used_bytes, v.target_bytes, v.ready_bytes,
                        v.cpu_workers, v.fixed_experts_per_layer, v.target_experts_per_layer,
                        v.ready_experts_per_layer, v.bridge_device_bytes, v.bridge_pinned_bytes,
                        v.cpu_workspace_bytes, v.uploaded_bytes, v.prefetched_bytes);
    else if constexpr (std::is_same_v<Value, MemoryAllocations>)
        return std::tie(v.device_weights_bytes, v.host_weights_bytes, v.registered_weights_bytes,
                        v.persistent_device_bytes, v.context_device_bytes, v.workspace_device_bytes,
                        v.io_pinned_bytes, v.host_context_bytes, v.ple_device_bytes,
                        v.ple_pinned_bytes, v.ple_host_bytes, v.expert_device_bytes,
                        v.expert_pinned_bytes, v.expert_host_bytes);
    else if constexpr (std::is_same_v<Value, WorkspaceUsage>)
        return std::tie(v.allocation_cursor_bytes, v.required_peak_bytes, v.measurement_peak_bytes);
    else if constexpr (std::is_same_v<Value, PleReadStats>)
        return std::tie(v.rows, v.cache_hits, v.shared_rows, v.shared_pages, v.reads, v.read_bytes,
                        v.uploaded_bytes);
    else if constexpr (std::is_same_v<Value, RankMemoryUsage>)
        return std::tie(v.allocations, v.workspace, v.vision, v.experts, v.ple);
    else if constexpr (std::is_same_v<Value, PhysicalMemorySample>)
        return std::tie(v.cuda_total_bytes, v.cuda_free_bytes, v.system_total_bytes,
                        v.system_available_bytes, v.shared_physical_memory, v.sampled_ns);
    else if constexpr (std::is_same_v<Value, SnapshotStamp>)
        return std::tie(v.sequence, v.sampled_ns);
    else if constexpr (std::is_same_v<Value, RankSnapshot>)
        return std::tie(v.rank, v.node, v.hardware, v.stamp, v.memory, v.host, v.communication,
                        v.physical_memory);
    else if constexpr (std::is_same_v<Value, CommunicationWork>)
        return std::tie(v.calls, v.payload_bytes);
    else if constexpr (std::is_same_v<Value, CommunicationStatistics>)
        return std::tie(v.group, v.size, v.eager, v.graph);
    else if constexpr (std::is_same_v<Value, RankHostStatistics>)
        return std::tie(v.submit_ns, v.progress_ns, v.supply_ns, v.commit_ns, v.discard_ns);
    else if constexpr (std::is_same_v<Value, GpuExecutionObservation>)
        return std::tie(v.result_stream_ns, v.commit_stream_ns, v.prefill_stream_ns,
                        v.vision_stream_ns);
    else if constexpr (std::is_same_v<Value, RankExecutionObservation>)
        return std::tie(v.path, v.host, v.gpu, v.communication_eager, v.communication_graph,
                        v.workspace_peak_bytes, v.measurement_peak_bytes, v.expert_upload_bytes,
                        v.ple_upload_bytes);
    else {
        static_assert(std::is_same_v<Value, RankMemoryBudget>);
        return std::tie(v.limit_bytes, v.runtime_reservation_bytes,
                        v.minimum_runtime_reservation_bytes, v.context_capacity_bytes,
                        v.minimum_context_capacity_bytes, v.context_control_budget_bytes,
                        v.available_after_weights_bytes, v.available_after_startup_bytes,
                        v.headroom_bytes, v.planned_slack_bytes, v.retained_bytes, v.graphs);
    }
}

template <class T>
void write(wire::Encoder& out, const T& v) {
    if constexpr (std::is_same_v<T, std::string>)
        out.string(v);
    else if constexpr (std::is_enum_v<T>)
        out.u32(static_cast<std::uint32_t>(v));
    else if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>)
            out.i64(v);
        else
            out.u64(v);
    } else if constexpr (requires { v.has_value(); }) {
        out.boolean(v.has_value());
        if (v) write(out, *v);
    } else if constexpr (requires {
                             v.size();
                             v.begin();
                         }) {
        out.u64(v.size());
        for (const auto& item : v) write(out, item);
    } else
        std::apply([&](const auto&... values) { (write(out, values), ...); }, fields(v));
}

template <class T>
void read(wire::Decoder& in, T& v) {
    if constexpr (std::is_same_v<T, std::string>)
        v = in.string();
    else if constexpr (std::is_enum_v<T>)
        v = static_cast<T>(in.u32());
    else if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>)
            v = static_cast<T>(in.i64());
        else
            v = static_cast<T>(in.u64());
    } else if constexpr (requires { v.has_value(); }) {
        if (in.boolean())
            read(in, v.emplace());
        else
            v.reset();
    } else if constexpr (requires { v.resize(0); }) {
        v.resize(in.count(8));
        for (auto& item : v) read(in, item);
    } else if constexpr (requires {
                             v.size();
                             v.begin();
                         }) {
        if (in.count(8) != v.size()) throw std::logic_error("observation array extent differs");
        for (auto& item : v) read(in, item);
    } else
        std::apply([&](auto&... values) { (read(in, values), ...); }, fields(v));
}
} // namespace

void encode(wire::Encoder& out, const RankMemoryBudget& summary) { write(out, summary); }

RankMemoryBudget memory_budget(wire::Decoder& in) {
    RankMemoryBudget result;
    read(in, result);
    return result;
}

void encode(wire::Encoder& out, const RankSnapshot& snapshot) { write(out, snapshot); }

RankSnapshot rank_snapshot(wire::Decoder& in) {
    RankSnapshot result;
    read(in, result);
    return result;
}

void encode(wire::Encoder& out, const RankExecutionObservation& value) { write(out, value); }

RankExecutionObservation rank_execution_observation(wire::Decoder& in) {
    RankExecutionObservation result;
    read(in, result);
    return result;
}
} // namespace ninfer::runtime::protocol
