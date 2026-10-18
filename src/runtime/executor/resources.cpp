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
    if constexpr (std::is_same_v<Value, ArenaMemorySummary>)
        return std::tie(v.capacity_bytes, v.used_bytes, v.peak_used_bytes);
    else if constexpr (std::is_same_v<Value, GraphMemorySummary>)
        return std::tie(v.rebuild_estimate_bytes, v.preparation_increment_bytes,
                        v.calibration_increment_bytes, v.calibration_retained_bytes);
    else if constexpr (std::is_same_v<Value, VisionWorkspaceMemorySummary>)
        return std::tie(v.aggregate_prompt_tokens, v.max_item_tokens, v.general_capacity_bytes,
                        v.encode_peak_bytes, v.handoff_offset_bytes, v.handoff_capacity_bytes,
                        v.handoff_active_bytes, v.handoff_peak_bytes);
    else if constexpr (std::is_same_v<Value, MoeMemorySummary>)
        return std::tie(v.host_weights_bytes, v.registered_host_bytes, v.cache_bytes, v.fixed_bytes,
                        v.window_bytes, v.window_count, v.used_bytes, v.target_bytes, v.ready_bytes,
                        v.cpu_workers, v.fixed_experts_per_layer, v.target_experts_per_layer,
                        v.ready_experts_per_layer, v.bridge_device_bytes, v.bridge_pinned_bytes,
                        v.cpu_workspace_bytes, v.uploaded_bytes, v.prefetched_bytes);
    else {
        static_assert(std::is_same_v<Value, MemorySummary>);
        return std::tie(
            v.device, v.max_context, v.prefill_chunk, v.device_budget_bytes, v.moe,
            v.kv_capacity_mode, v.kv_capacity, v.kv_capacity_page_groups,
            v.kv_capacity_max_page_groups, v.kv_cache, v.weights, v.sequence, v.workspace,
            v.vision_workspace, v.minimum_runtime_reservation_bytes, v.kv_capacity_increment_bytes,
            v.runtime_reservation_bytes, v.available_after_weights_bytes,
            v.available_after_startup_bytes, v.kv_capacity_headroom_bytes, v.planned_slack_bytes,
            v.workspace_logical_peak_bytes, v.cuda_graph, v.startup_retained_bytes,
            v.kv_payload_bytes, v.host_context_capacity_bytes, v.host_context_occupied_bytes,
            v.host_context_reserved_bytes, v.host_state_occupied_slots, v.host_kv_occupied_bytes);
    }
}

template <class T>
void write(wire::Encoder& out, const T& v) {
    if constexpr (std::is_enum_v<T>)
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
    if constexpr (std::is_enum_v<T>)
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
    } else
        std::apply([&](auto&... values) { (read(in, values), ...); }, fields(v));
}
} // namespace

void encode(wire::Encoder& out, const MemorySummary& summary) { write(out, summary); }

MemorySummary memory_summary(wire::Decoder& in) {
    MemorySummary result;
    read(in, result);
    return result;
}
} // namespace ninfer::runtime::protocol
