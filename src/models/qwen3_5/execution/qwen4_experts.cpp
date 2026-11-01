#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/qwen4_layer_profile.h"
#include <sstream>
#include <fstream>

#include "core/device.h"
#include "core/device_scope.h"
#include "ninfer/ops/moe_grouped.h"
#include "ninfer/ops/moe_slots.h"
#include "ops/sparse_moe/host/host_moe_mailbox.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <span>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <numeric>
#include <thread>
#include <cinttypes>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::int32_t kExperts        = 512;
constexpr std::int32_t kTopK           = ops::kMoeSlotsTopK;
constexpr std::int32_t kHidden         = ops::kMoeSlotsHidden;
constexpr std::int32_t kInter          = ops::kMoeSlotsIntermediate;
constexpr std::int32_t kGateUpDivisors = 2 * kExperts;
// Routing-log room of a later (MTP) layer beyond one call of the widest kind: a prompt window logs
// its chunk's call (<= prompt_tokens columns) and then one column per draft (<= 5); a decode window
// its alignment call ((K+1) x B <= 8 columns) and the AR drafts ((K-1) x B <= 8).
constexpr std::int32_t kExtraLayerLogTokens = 2 * ops::kMoeSlotsMaxTokens;
// Calls of a later (MTP) layer per window: one chunk or alignment call and up to 5 drafts (with
// margin; the expert trace keeps one record per call).
constexpr std::int64_t kExtraLayerCalls = 8;

Tensor divisor_tensor(const float* data, std::int32_t count) {
    return Tensor(const_cast<float*>(data), DType::FP32, {count});
}

ops::MoeSlotsLayer slots_layer(const Qwen4ExpertBank& bank, const float* gate_up_divisors,
                               const float* down_divisors, const Tensor& table) {
    return ops::MoeSlotsLayer{.slot_table       = table,
                              .gate_up_divisors = divisor_tensor(gate_up_divisors, kGateUpDivisors),
                              .down_divisors    = divisor_tensor(down_divisors, kExperts),
                              .gate_up_input_divisor = bank.gate_up_input_divisor,
                              .down_input_divisor    = bank.down_input_divisor};
}

void require_operands(const Tensor& x, const Tensor& ids, const Tensor& weights,
                      const Tensor& partial) {
    const std::int32_t T = x.ne[1];
    if (x.dtype != DType::BF16 || x.ne[0] != kHidden || ids.dtype != DType::I32 ||
        ids.ne[0] != kTopK || ids.ne[1] != T || weights.dtype != DType::FP32 ||
        weights.ne[0] != kTopK || weights.ne[1] != T || partial.dtype != DType::FP32 ||
        partial.ne[0] != kHidden || partial.ne[1] != T || T <= 0) {
        throw std::invalid_argument("Qwen4Exp routed experts: operands must be x [2560,T], "
                                    "ids/weights [10,T] and an FP32 [2560,T] partial");
    }
}

// The host executor reads the bank in place: per layer the gate|up matrix [512*1280, 2560] and
// the down matrix [512*2560, 640], each expert a contiguous row block (the bank's plane strides
// must equal the expert plane sizes).
ops::host::HostExpertLayer host_layer(const Qwen4ExpertBank& bank) {
    const ops::MoeSlotPool& b = bank.bank;
    if (b.gate_up_codes.stride != ops::kMoeSlotsGateUpCodeBytes ||
        b.gate_up_scales.stride != ops::kMoeSlotsGateUpScaleBytes ||
        b.down_codes.stride != ops::kMoeSlotsDownCodeBytes ||
        b.down_scales.stride != ops::kMoeSlotsDownScaleBytes || b.slots != kExperts) {
        throw std::invalid_argument("Qwen4Exp expert bank is not a contiguous stacked bank");
    }
    ops::host::HostExpertLayer layer;
    layer.gate_up = ops::host::Nvfp4Plane{
        .codes            = static_cast<const std::uint8_t*>(b.gate_up_codes.base),
        .scales           = static_cast<const std::uint8_t*>(b.gate_up_scales.base),
        .divisors         = bank.gate_up_divisors,
        .rows             = static_cast<std::int64_t>(kExperts) * 2 * kInter,
        .k                = kHidden,
        .rows_per_divisor = kInter,
    };
    layer.down = ops::host::Nvfp4Plane{
        .codes            = static_cast<const std::uint8_t*>(b.down_codes.base),
        .scales           = static_cast<const std::uint8_t*>(b.down_scales.base),
        .divisors         = bank.down_divisors,
        .rows             = static_cast<std::int64_t>(kExperts) * kHidden,
        .k                = kInter,
        .rows_per_divisor = kHidden,
    };
    layer.experts               = kExperts;
    layer.hidden                = kHidden;
    layer.intermediate          = kInter;
    layer.gate_up_input_divisor = bank.gate_up_input_divisor; // T98 W4A4 emulation only
    layer.down_input_divisor    = bank.down_input_divisor;
    return layer;
}

// A ggml block bank for the host executor (formats share the ggml type ids).
ops::host::HostExpertLayer host_gguf_layer(const Qwen4ExpertBank& bank) {
    ops::host::HostExpertLayer layer;
    layer.format                 = ops::host::HostExpertFormat::Gguf;
    layer.gguf.gate_up           = static_cast<ops::host::gguf::WeightFormat>(bank.gguf_gate_up);
    layer.gguf.down              = static_cast<ops::host::gguf::WeightFormat>(bank.gguf_down);
    layer.gguf.gate_up_rows      = reinterpret_cast<const std::uint8_t*>(bank.gguf_gate_up_rows);
    layer.gguf.down_rows         = reinterpret_cast<const std::uint8_t*>(bank.gguf_down_rows);
    layer.gguf.gate_up_row_bytes = static_cast<std::size_t>(bank.gguf_gate_up_row_bytes);
    layer.gguf.down_row_bytes    = static_cast<std::size_t>(bank.gguf_down_row_bytes);
    layer.experts                = kExperts;
    layer.hidden                 = kHidden;
    layer.intermediate           = kInter;
    return layer;
}

// The GGUF Op view of a layer: device slots (blob form) or the host bank read in place (planar).
ops::MoeGgufLayer gguf_view(const Tensor& table, const Qwen4ExpertBank& bank, std::int32_t slots,
                            const std::int64_t* slot_offset, const std::byte* base) {
    ops::MoeGgufLayer v;
    v.slot_table     = table;
    v.slots          = slots;
    v.slot_offset    = slot_offset;
    v.base           = base;
    v.gate_up_base   = bank.gguf_gate_up_rows;
    v.down_base      = bank.gguf_down_rows;
    v.gate_up_format = bank.gguf_gate_up;
    v.down_format    = bank.gguf_down;
    return v;
}

// Partial of the experts `view` holds, for any T: decode widths through moe_slots_gguf, wider
// calls grouped by expert (moe_grouped_gguf).
void gguf_partial(const Tensor& x, const Tensor& ids, const Tensor& weights,
                  const ops::MoeGgufLayer& view, Tensor& partial, WorkspaceArena& work,
                  cudaStream_t stream) {
    const std::int32_t T = x.ne[1];
    auto scope           = work.scope();
    if (T <= ops::kMoeSlotsMaxTokens) {
        WorkspaceArena scratch(work.alloc_bytes(ops::moe_slots_gguf_workspace_bytes(T)));
        ops::moe_slots_gguf(x, ids, weights, view, ops::MoeSlotsOutput::Store, partial, scratch,
                            stream);
    } else {
        WorkspaceArena scratch(work.alloc_bytes(ops::moe_grouped_gguf_workspace_bytes(T)));
        ops::moe_grouped_gguf(x, ids, weights, view, partial, scratch, stream);
    }
}

const char* window_name(Qwen4Window kind) {
    return kind == Qwen4Window::Decode ? "decode" : "prompt";
}

void add(Qwen4OffloadWindowStats& sum, const Qwen4OffloadWindowStats& w) {
    sum.tokens += w.tokens;
    sum.layers += w.layers;
    sum.accesses += w.accesses;
    sum.device_hits += w.device_hits;
    sum.policy_hits += w.policy_hits;
    sum.missing += w.missing;
    sum.inserts += w.inserts;
    sum.fills_issued += w.fills_issued;
    sum.fill_bytes += w.fill_bytes;
    sum.fills_pending += w.fills_pending;
    sum.wall_us += w.wall_us;
    sum.observe_us += w.observe_us;
    sum.cpu_us += w.cpu_us;
    sum.cpu_experts += w.cpu_experts;
    sum.gpu_resident_us += w.gpu_resident_us;
    sum.gpu_wait_us += w.gpu_wait_us;
    sum.doorbell_us += w.doorbell_us;
    sum.bell_after_flag_us += w.bell_after_flag_us;
    sum.prefetch_issued += w.prefetch_issued;
    sum.prefetch_landed += w.prefetch_landed;
    sum.prefetch_used += w.prefetch_used;
    sum.prefetch_used_routes += w.prefetch_used_routes;
    sum.streamed_experts += w.streamed_experts;
    sum.streamed_bytes += w.streamed_bytes;
    sum.streamed_layers += w.streamed_layers;
    sum.gpu_read_experts += w.gpu_read_experts;
    sum.gpu_read_routes += w.gpu_read_routes;
    sum.host_experts += w.host_experts;
}

void write_stats(std::FILE* f, const Qwen4OffloadWindowStats& w, double n) {
    std::fprintf(f,
                 "\"tokens\":%.3f,\"layers\":%.3f,\"accesses\":%.3f,\"device_hits\":%.3f,"
                 "\"policy_hits\":%.3f,\"missing\":%.3f,\"inserts\":%.3f,\"fills_issued\":%.3f,"
                 "\"fill_bytes\":%.1f,\"fills_pending\":%.3f,\"wall_us\":%.2f,\"observe_us\":%.2f,"
                 "\"cpu_us\":%.2f,\"cpu_experts\":%.3f,\"gpu_resident_us\":%.2f,"
                 "\"gpu_wait_us\":%.2f,\"doorbell_us\":%.2f,\"bell_after_flag_us\":%.2f",
                 w.tokens / n, w.layers / n, static_cast<double>(w.accesses) / n,
                 static_cast<double>(w.device_hits) / n, static_cast<double>(w.policy_hits) / n,
                 static_cast<double>(w.missing) / n, static_cast<double>(w.inserts) / n,
                 static_cast<double>(w.fills_issued) / n, static_cast<double>(w.fill_bytes) / n,
                 static_cast<double>(w.fills_pending) / n, w.wall_us / n, w.observe_us / n,
                 w.cpu_us / n, static_cast<double>(w.cpu_experts) / n, w.gpu_resident_us / n,
                 w.gpu_wait_us / n, w.doorbell_us / n, w.bell_after_flag_us / n);
    std::fprintf(
        f,
        ",\"prefetch_issued\":%.3f,\"prefetch_landed\":%.3f,\"prefetch_used\":%.3f,"
        "\"prefetch_used_routes\":%.3f",
        static_cast<double>(w.prefetch_issued) / n, static_cast<double>(w.prefetch_landed) / n,
        static_cast<double>(w.prefetch_used) / n, static_cast<double>(w.prefetch_used_routes) / n);
    std::fprintf(f, ",\"gpu_read_experts\":%.3f,\"gpu_read_routes\":%.3f,\"host_experts\":%.3f",
                 static_cast<double>(w.gpu_read_experts) / n,
                 static_cast<double>(w.gpu_read_routes) / n,
                 static_cast<double>(w.host_experts) / n);
    std::fprintf(f, ",\"streamed_experts\":%.3f,\"streamed_bytes\":%.1f,\"streamed_layers\":%.3f",
                 static_cast<double>(w.streamed_experts) / n,
                 static_cast<double>(w.streamed_bytes) / n,
                 static_cast<double>(w.streamed_layers) / n);
}

} // namespace

namespace {
// Arena allocations round up to this (core/arena.h kArenaAlign is 256).
constexpr std::size_t kAllocSlack = 256;

std::size_t partial_bytes(std::int32_t columns) {
    return static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(columns) * sizeof(float) +
           kAllocSlack;
}
} // namespace

std::size_t Qwen4RoutedExperts::workspace_bytes(std::int32_t tokens) {
    const std::int32_t T = std::clamp<std::int32_t>(tokens, 1, ops::kMoeSlotsMaxTokens);
    const std::size_t slots =
        2 * (ops::moe_slots_nvfp4_workspace_bytes(T) + kAllocSlack) + partial_bytes(T);
    // ggml block banks (capy-flash v0.7): the GGUF Ops of the same widths.
    const std::size_t gguf_slots =
        ops::moe_slots_gguf_workspace_bytes(T) + kAllocSlack + partial_bytes(T);
    // Wider calls may stream the layer and run it grouped (Qwen4OffloadExperts::stream_call).
    if (tokens <= ops::kMoeSlotsMaxTokens) { return std::max(slots, gguf_slots); }
    return std::max({slots, gguf_slots,
                     ops::moe_grouped_nvfp4_workspace_bytes(tokens) + kAllocSlack,
                     ops::moe_grouped_gguf_workspace_bytes(tokens) + kAllocSlack});
}

std::size_t Qwen4RoutedExperts::routed_workspace_bytes(std::int32_t tokens) {
    if (tokens > ops::kMoeSlotsMaxTokens) { return 0; }
    const std::int32_t T = std::max<std::int32_t>(tokens, 1);
    return ops::moe_slots_nvfp4_workspace_bytes(T) + kAllocSlack + partial_bytes(T);
}

// ---- miss split options ------------------------------------------------------------------------

MissSplitOptions effective_miss_split(const RoutedExpertOptions& experts) noexcept {
    MissSplitOptions split = experts.miss_split;
    if (split.mode == MissSplitMode::Off && experts.device_reads > 0) {
        split.mode   = MissSplitMode::Count;
        split.count  = experts.device_reads;
        split.staged = false;
    }
    return split;
}

ops::host::HostMoeSplit host_miss_split(const MissSplitOptions& split) noexcept {
    using Mode = ops::host::HostMoeSplit::Mode;
    ops::host::HostMoeSplit h;
    switch (split.mode) {
    case MissSplitMode::Off:
        h.mode = Mode::Off;
        break;
    case MissSplitMode::Auto:
        h.mode = Mode::Cost;
        break;
    case MissSplitMode::Fraction:
        h.mode = Mode::Fraction;
        break;
    case MissSplitMode::Count:
        h.mode = Mode::Count;
        break;
    }
    h.count              = static_cast<std::int32_t>(split.count);
    h.fraction           = split.fraction;
    h.min_misses         = static_cast<std::int32_t>(split.min_misses);
    h.max_reads          = static_cast<std::int32_t>(split.max_reads);
    h.host_fixed_us      = split.costs.cpu_fixed_us;
    h.host_expert_us     = split.costs.cpu_expert_us;
    h.host_column_us     = split.costs.cpu_column_us;
    h.host_read_slowdown = split.costs.cpu_read_slowdown;
    h.device_fixed_us    = split.costs.gpu_fixed_us;
    h.device_expert_us   = split.costs.gpu_expert_us;
    h.device_column_us   = split.costs.gpu_column_us;
    return h;
}

bool staged_miss_split(const RoutedExpertOptions& experts) noexcept {
    const MissSplitOptions split = effective_miss_split(experts);
    return split.mode != MissSplitMode::Off && split.staged;
}

// ---- Qwen4MappedExperts ------------------------------------------------------------------------

void Qwen4MappedExperts::run(std::int32_t, const Qwen4ExpertBank& bank, const Tensor& x,
                             const Tensor& ids, const Tensor& weights, Tensor& partial,
                             WorkspaceArena& work, cudaStream_t stream) {
    require_operands(x, ids, weights, partial);
    if (bank.gguf) {
        // The host bank read in place (planar form, slot = expert through the table).
        gguf_partial(x, ids, weights, gguf_view(table_, bank, kExperts, nullptr, nullptr), partial,
                     work, stream);
        return;
    }
    // The stacked divisor planes live in the host-mapped bank; the expert kernel reads the words
    // of the experts it runs in place (a few bytes per expert over PCIe).
    const ops::MoeSlotsLayer layer =
        slots_layer(bank, bank.gate_up_divisors, bank.down_divisors, table_);
    const std::int32_t T = x.ne[1];
    // The expert Op's scratch is a sub-arena: its output (in `work`) must not overlap it.
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(workspace_bytes(T)));
    for (std::int32_t begin = 0; begin < T; begin += ops::kMoeSlotsMaxTokens) {
        const std::int32_t count = std::min(ops::kMoeSlotsMaxTokens, T - begin);
        Tensor out               = partial.slice(1, begin, count);
        ops::moe_slots_nvfp4(x.slice(1, begin, count), ids.slice(1, begin, count),
                             weights.slice(1, begin, count), layer, bank.bank,
                             ops::MoeSlotsOutput::Store, out, scratch, stream);
    }
}

// ---- Qwen4OffloadExperts -----------------------------------------------------------------------

namespace {

std::array<std::int64_t, 4> planar_strides() {
    return {ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
            ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
}

ops::MoeSlotPool slot_pool(const std::array<std::byte*, 4>& base,
                           const std::array<std::int64_t, 4>& stride, std::int32_t slots) {
    return ops::MoeSlotPool{
        .gate_up_codes  = {base[0], stride[0]},
        .gate_up_scales = {base[1], stride[1]},
        .down_codes     = {base[2], stride[2]},
        .down_scales    = {base[3], stride[3]},
        .slots          = slots,
    };
}

} // namespace

std::array<std::int64_t, 4> Qwen4OffloadExperts::pool_plane_bytes(std::int32_t layers,
                                                                  std::int32_t slots_per_layer,
                                                                  std::int32_t extra_slots) {
    const std::int64_t slots = static_cast<std::int64_t>(layers) * slots_per_layer + extra_slots;
    return {slots * ops::kMoeSlotsGateUpCodeBytes, slots * ops::kMoeSlotsGateUpScaleBytes,
            slots * ops::kMoeSlotsDownCodeBytes, slots * ops::kMoeSlotsDownScaleBytes};
}

std::int64_t Qwen4OffloadExperts::plane_slot_bytes(std::int32_t plane) {
    return planar_strides().at(static_cast<std::size_t>(plane));
}

bool Qwen4OffloadExperts::gguf_banks(std::span<const Qwen4ExpertBank> banks) {
    const auto n =
        std::count_if(banks.begin(), banks.end(), [](const Qwen4ExpertBank& b) { return b.gguf; });
    if (n != 0 && n != static_cast<std::ptrdiff_t>(banks.size())) {
        throw std::invalid_argument(
            "Qwen4Exp offload experts: every routed-expert layer (the MTP layer included) must be "
            "NVFP4, or every one ggml blocks");
    }
    return n != 0;
}

std::int64_t Qwen4OffloadExperts::gguf_slot_bytes(const Qwen4ExpertBank& bank) noexcept {
    return (bank.gguf_expert_bytes() + 255) / 256 * 256;
}

Qwen4OffloadExperts::GgufPool
Qwen4OffloadExperts::gguf_pool(std::span<const std::int64_t> layer_slot_bytes,
                               std::span<const std::int32_t> layer_base,
                               std::int32_t elastic_columns) {
    if (layer_slot_bytes.size() != layer_base.size() || elastic_columns < 0) {
        throw std::invalid_argument("Qwen4Exp GGUF pool: one slot size and base count per layer");
    }
    const auto L = layer_base.size();
    GgufPool out;
    for (std::size_t l = 0; l < L; ++l) {
        for (std::int32_t s = 0; s < layer_base[l]; ++s) {
            out.slot_offset.push_back(out.base_bytes);
            out.base_bytes += layer_slot_bytes[l];
        }
        out.column_bytes += layer_slot_bytes[l];
    }
    for (std::int32_t c = 0; c < elastic_columns; ++c) {
        std::int64_t at = out.base_bytes + static_cast<std::int64_t>(c) * out.column_bytes;
        for (std::size_t l = 0; l < L; ++l) {
            out.slot_offset.push_back(at);
            at += layer_slot_bytes[l];
        }
    }
    out.total_bytes =
        out.base_bytes + static_cast<std::int64_t>(elastic_columns) * out.column_bytes;
    return out;
}

std::int64_t Qwen4OffloadExperts::elastic_column_offset(std::int32_t plane,
                                                        std::int32_t column) const {
    if (gguf_) {
        return plane == 0 ? gguf_pool_.base_bytes +
                                static_cast<std::int64_t>(column) * gguf_pool_.column_bytes
                          : 0;
    }
    const std::int64_t slot = static_cast<std::int64_t>(layers_) * options_.slots_per_layer +
                              staging_slots_ + static_cast<std::int64_t>(column) * layers_;
    return slot * plane_slot_bytes(plane);
}

std::int64_t Qwen4OffloadExperts::elastic_column_bytes(std::int32_t plane) const {
    if (gguf_) { return plane == 0 ? gguf_pool_.column_bytes : 0; }
    return static_cast<std::int64_t>(layers_) * plane_slot_bytes(plane);
}

void Qwen4OffloadExperts::set_active_columns(std::int32_t columns) {
    if (open_) {
        throw std::logic_error("Qwen4Exp offload experts: elastic resize inside a window");
    }
    const auto streams = window_streams(streams_[0]);
    cache_->set_active_columns(columns, streams);
}

std::int64_t Qwen4OffloadExperts::slot_bytes(std::int32_t layers) {
    const auto planes = pool_plane_bytes(layers, 1);
    return planes[0] + planes[1] + planes[2] + planes[3];
}

Qwen4OffloadExperts::Qwen4OffloadExperts(std::span<const Qwen4ExpertBank> banks,
                                         const Qwen4OffloadOptions& options, DeviceMemory memory,
                                         int device, cudaStream_t stream, const Peer* peer)
    : layers_(static_cast<std::int32_t>(banks.size())), partitions_(peer != nullptr ? 2 : 1),
      streams_{stream, peer != nullptr ? peer->stream : nullptr},
      peer_stream_(peer != nullptr ? peer->stream : nullptr),
      text_layers_(options.text_layers > 0 ? options.text_layers
                                           : static_cast<std::int32_t>(banks.size())),
      log_tokens_(
          std::max(options.prompt_tokens, ops::kMoeSlotsMaxTokens) +
          (options.text_layers > 0 && options.text_layers < static_cast<std::int32_t>(banks.size())
               ? kExtraLayerLogTokens
               : 0)),
      options_(options), memory_(memory) {
    if (text_layers_ > layers_) {
        throw std::invalid_argument("Qwen4Exp offload experts: more Text layers than banks");
    }
    extra_offset_.assign(static_cast<std::size_t>(layers_ - text_layers_), 0);
    if (layers_ <= 0 || options_.slots_per_layer < 1 || options_.slots_per_layer > kExperts ||
        options_.elastic_columns < 0 ||
        options_.slots_per_layer + options_.elastic_columns > kExperts ||
        options_.cpu_threads < 1 || options_.insert_cap < 0 || options_.admit_threshold < 0.0 ||
        !(options_.half_life_tokens > 0.0) || !(options_.evict_tau_windows >= 0.0) ||
        options_.prompt_tokens < 1 || options_.stream_min_tokens < 0 ||
        options_.stream_ring_slots < kExperts) {
        throw std::invalid_argument("Qwen4Exp offload experts: invalid options");
    }
    layer_base_ = layer_slot_profile(layers_, text_layers_, options_.slots_per_layer,
                                     options_.elastic_columns, partitions_);
    layer_offset_.assign(static_cast<std::size_t>(layers_) + 1, 0);
    for (std::int32_t l = 0; l < layers_; ++l) {
        layer_offset_[static_cast<std::size_t>(l) + 1] =
            layer_offset_[static_cast<std::size_t>(l)] + layer_base_[static_cast<std::size_t>(l)];
    }
    gguf_       = gguf_banks(banks);
    pool_slots_ = layers_ * (options_.slots_per_layer + options_.elastic_columns);
    if (gguf_) {
        if (options_.prefetch.slots_per_layer > 0 ||
            options_.split.mode != ops::host::HostMoeSplit::Mode::Off ||
            options_.stream_min_tokens > 0 || options_.base_chunk > 0) {
            std::fprintf(stderr,
                         "Qwen4Exp offload experts: ggml block banks: the NVFP4 prefill ring, the "
                         "router-ahead prefetch and the miss split are off (GGUF ring instead)\n");
        }
        options_.prefetch.slots_per_layer = 0;
        options_.split.mode               = ops::host::HostMoeSplit::Mode::Off;
        options_.split_staged             = false;
        options_.split_on_peer            = false;
        options_.stream_min_tokens        = 0; // the NVFP4 ring; GGUF: gguf_stream_min_tokens
        if (memory_.slot_offsets == nullptr || (peer != nullptr && peer->slot_offsets == nullptr)) {
            throw std::invalid_argument("Qwen4Exp offload experts: GGUF slot offsets are missing");
        }
        std::vector<std::int64_t> bytes;
        for (const Qwen4ExpertBank& b : banks) { bytes.push_back(gguf_slot_bytes(b)); }
        gguf_pool_       = gguf_pool(bytes, layer_base_, options_.elastic_columns);
        gguf_slot_bytes_ = bytes;
        if (options_.gguf_stream_min_tokens > 0 && options_.base_chunk > 0 &&
            options_.prompt_workspace_bytes > 0) {
            // The prefill ring: the last Text layers (as the NVFP4 ring), the borrowed workspace
            // below it; the planner lends with the same rule (startup.cpp).
            const std::int64_t ring_bytes = gguf_ring_bytes(bytes, text_layers_, partitions_);
            lend_count_ = gguf_layers_below(layer_base_, bytes, text_layers_, ring_bytes);
            lend_first_ = text_layers_ - lend_count_;
            workspace_layers_ =
                gguf_layers_below(layer_base_, bytes, lend_first_, options_.prompt_workspace_bytes);
            if (lend_count_ > text_layers_ ||
                !may_lend(layer_base_, text_layers_, lend_count_ + workspace_layers_,
                          partitions_)) {
                std::fprintf(stderr,
                             "Qwen4Exp offload experts: ggml block banks: no room for the "
                             "prefill ring and workspace; zero-copy prefill, base chunks\n");
                lend_count_       = 0;
                lend_first_       = text_layers_;
                workspace_layers_ = 0;
            } else {
                gguf_ring_slots_  = gguf_ring_slots(partitions_);
                gguf_ring_base_   = gguf_pool_.slot_offset[static_cast<std::size_t>(
                    layer_offset_[static_cast<std::size_t>(lend_first_)])];
                gguf_ring_stride_ = (ring_bytes / gguf_ring_slots_) / 256 * 256;
                for (std::int32_t k = 0; k < gguf_ring_slots_; ++k) {
                    gguf_pool_.slot_offset.push_back(gguf_ring_base_ + k * gguf_ring_stride_);
                }
            }
        }
        if (static_cast<std::int64_t>(gguf_pool_.slot_offset.size()) !=
            pool_slots_ + gguf_ring_slots_) {
            throw std::logic_error("Qwen4Exp offload experts: GGUF pool slot count");
        }
        std::fprintf(stderr,
                     "Qwen4Exp offload experts: ggml block banks, %d slots per GPU in %.2f GiB "
                     "(%.0f KB per slot on average)\n",
                     pool_slots_, static_cast<double>(gguf_pool_.total_bytes) / (1 << 30),
                     static_cast<double>(gguf_pool_.total_bytes) / std::max(pool_slots_, 1) /
                         1024.0);
    }
    if (layer_base_ != std::vector<std::int32_t>(layer_base_.size(), options_.slots_per_layer)) {
        const auto [lo, hi] = std::minmax_element(layer_base_.begin(), layer_base_.end());
        std::fprintf(stderr,
                     "Qwen4Exp offload experts: per-layer slot profile, %d..%d base slots per "
                     "layer (mean %d) + %d elastic\n",
                     *lo, *hi, options_.slots_per_layer, options_.elastic_columns);
    }
    {
        using Mode        = ops::host::HostMoeSplit::Mode;
        const auto& sp    = options_.split;
        const auto mode   = static_cast<std::int32_t>(sp.mode);
        const auto finite = [](float v) { return std::isfinite(v) && v >= 0.0F; };
        if (mode < 0 || mode > static_cast<std::int32_t>(Mode::Cost) || sp.count < 0 ||
            sp.count > ops::host::HostMoeSplit::kMaxReads || sp.max_reads < 0 ||
            sp.max_reads > ops::host::HostMoeSplit::kMaxReads ||
            !(sp.fraction >= 0.0F && sp.fraction <= 1.0F) || sp.min_misses < 0 ||
            !(sp.host_read_slowdown >= 1.0F) || !finite(sp.host_fixed_us) ||
            !finite(sp.host_expert_us) || !finite(sp.host_column_us) ||
            !finite(sp.device_fixed_us) || !finite(sp.device_expert_us) ||
            !finite(sp.device_column_us)) {
            throw std::invalid_argument("Qwen4Exp offload experts: invalid miss split");
        }
    }
    if (memory_.table == nullptr || memory_.divisors == nullptr || memory_.identity == nullptr ||
        memory_.reads == nullptr ||
        (options_.stream_min_tokens > 0 && memory_.stream_tables == nullptr) ||
        std::any_of(memory_.pool.begin(), memory_.pool.end(),
                    [](const std::byte* p) { return p == nullptr; })) {
        throw std::invalid_argument("Qwen4Exp offload experts: device memory is missing");
    }
    if (peer != nullptr &&
        (memory_.mirror == nullptr || peer->table == nullptr || peer->divisors == nullptr ||
         peer->stream == nullptr || peer->device == device ||
         std::any_of(peer->pool.begin(), peer->pool.end(),
                     [](const std::byte* p) { return p == nullptr; }))) {
        throw std::invalid_argument("Qwen4Exp offload experts: rank 1 memory is missing");
    }
    // v0.7 pinned fallback: banks beyond the page-locked limit live in pageable host memory. Only
    // the CPU and the expert cache (through pinned staging) read them; every path that has a
    // kernel read a bank in place or copies a whole layer is kept off them.
    pageable_.assign(static_cast<std::size_t>(layers_), 0);
    std::string pageable_list;
    for (std::int32_t l = 0; l < layers_; ++l) {
        const Qwen4ExpertBank& bank = banks[static_cast<std::size_t>(l)];
        const ops::MoeSlotPool& b   = bank.bank;
        // ggml block banks keep their rows outside the NVFP4 planes, which stay null (issue #22:
        // every GSQ-RCO layer was reported pageable although the rows are page-locked).
        const std::array<const void*, 4> bases =
            bank.gguf ? std::array<const void*, 4>{bank.gguf_gate_up_rows, bank.gguf_down_rows}
                      : std::array<const void*, 4>{b.gate_up_codes.base, b.gate_up_scales.base,
                                                   b.down_codes.base, b.down_scales.base};
        for (const void* base : std::span(bases).first(bank.gguf ? 2 : 4)) {
            if (!expert_cache::pinned_host_memory(base)) {
                pageable_[static_cast<std::size_t>(l)] = 1;
            }
        }
        if (pageable_[static_cast<std::size_t>(l)] != 0) {
            ++pageable_layers_;
            pageable_list += (pageable_list.empty() ? "" : ",") + std::to_string(l);
        }
    }
    if (pageable_layers_ > 0) {
        if (options_.prefetch.slots_per_layer > 0) {
            throw std::invalid_argument(
                "Qwen4Exp offload experts: the router-ahead prefetch copies from every bank; it "
                "needs every expert bank page-locked (raise --host-pinned-max-gib)");
        }
        if (splitting()) {
            std::fprintf(stderr,
                         "Qwen4Exp offload experts: miss split off (the GPU would read pageable "
                         "expert banks)\n");
            options_.split.mode   = ops::host::HostMoeSplit::Mode::Off;
            options_.split_staged = false;
        }
        std::fprintf(stderr,
                     "Qwen4Exp offload experts: %d of %d expert layers in pageable host memory "
                     "(layers %s): the CPU reads them in place, slot fills go through pinned "
                     "staging, no prefill streaming on them\n",
                     pageable_layers_, layers_, pageable_list.c_str());
    }
    if (splitting() && options_.split_staged &&
        (memory_.stage_table == nullptr || memory_.picks == nullptr ||
         std::any_of(memory_.staging.begin(), memory_.staging.end(),
                     [](const std::byte* p) { return p == nullptr; }))) {
        throw std::invalid_argument(
            "Qwen4Exp offload experts: the staged miss split needs scratch slots");
    }
    device_ids_[0] = device;
    banks_.assign(banks.begin(), banks.end());
    split_on_peer_ =
        peer != nullptr && options_.split_on_peer && splitting() && options_.split_staged;
    if (split_on_peer_ && (peer->mirror == nullptr || peer->reads == nullptr ||
                           peer->stage_table == nullptr || peer->picks == nullptr ||
                           std::any_of(peer->staging.begin(), peer->staging.end(),
                                       [](const std::byte* p) { return p == nullptr; }))) {
        throw std::invalid_argument(
            "Qwen4Exp offload experts: rank 1 miss-split memory is missing");
    }
    if (peer != nullptr) {
        peer_          = *peer;
        peer_pool_     = peer->pool;
        peer_divisors_ = peer->divisors;
        device_ids_[1] = peer->device;
    }
    if (!ops::host::host_experts_supported()) {
        throw std::runtime_error(
            "Qwen4Exp offload experts compute missing experts with AVX2 + FMA + F16C or AVX-512 "
            "VNNI, which this CPU lacks; use the mapped expert backend");
    }
    {
        const ops::host::HostIsa isa = ops::host::host_isa(); // throws on a bad NINFER_HOST_ISA
        const char* forced           = std::getenv("NINFER_HOST_ISA");
        std::fprintf(stderr, "qwen4 experts: host expert kernels %s%s\n",
                     ops::host::host_isa_name(isa),
                     forced != nullptr && *forced != '\0' ? " (NINFER_HOST_ISA)" : "");
    }

    // Divisors by expert id, once, on the device (the bank copy is host memory).
    std::vector<ops::host::HostExpertLayer> host_layers;
    host_layers.reserve(banks.size());
    for (std::int32_t l = 0; l < layers_; ++l) {
        const Qwen4ExpertBank& bank = banks[static_cast<std::size_t>(l)];
        if (gguf_) {
            host_layers.push_back(host_gguf_layer(bank));
            continue;
        }
        float* d = memory_.divisors + static_cast<std::size_t>(l) * kDivisorsPerLayer;
        CUDA_CHECK(cudaMemcpyAsync(d, bank.gate_up_divisors, sizeof(float) * kGateUpDivisors,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d + kGateUpDivisors, bank.down_divisors,
                                   sizeof(float) * kExperts, cudaMemcpyHostToDevice, stream));
        host_layers.push_back(host_layer(bank));
    }
    if (gguf_) {
        // One GPU's slot offsets, the same on both GPUs (same base slots and columns per layer).
        CUDA_CHECK(cudaMemcpyAsync(memory_.slot_offsets, gguf_pool_.slot_offset.data(),
                                   sizeof(std::int64_t) * gguf_pool_.slot_offset.size(),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (peer != nullptr && gguf_) {
        const ScopedCurrentDevice rank1(peer->device);
        CUDA_CHECK(cudaMemcpyAsync(peer->slot_offsets, gguf_pool_.slot_offset.data(),
                                   sizeof(std::int64_t) * gguf_pool_.slot_offset.size(),
                                   cudaMemcpyHostToDevice, peer->stream));
        CUDA_CHECK(cudaStreamSynchronize(peer->stream));
    } else if (peer != nullptr) {
        // Rank 1's copy of the divisors: its expert kernel reads them from its own memory.
        const ScopedCurrentDevice rank1(peer->device);
        CUDA_CHECK(
            cudaMemcpyAsync(peer->divisors, memory_.divisors,
                            sizeof(float) * kDivisorsPerLayer * static_cast<std::size_t>(layers_),
                            cudaMemcpyDefault, peer->stream));
        CUDA_CHECK(cudaStreamSynchronize(peer->stream));
    }

    expert_cache::PolicyConfig policy;
    policy.layers  = layers_;
    policy.experts = kExperts;
    // Partition p (GPU p) holds the experts e % partitions == p in its own slots_per_layer slots.
    policy.partitions      = partitions_;
    policy.slots_per_layer = options_.slots_per_layer * partitions_;
    // Every GPU holds the same base slots per layer (layer_slot_profile).
    for (const std::int32_t base : layer_base_) {
        policy.layer_slots.push_back(base * partitions_);
    }
    policy.admission         = options_.admit_threshold > 0.0;
    policy.admit_threshold   = options_.admit_threshold;
    policy.half_life_tokens  = options_.half_life_tokens;
    policy.evict_tau_windows = options_.evict_tau_windows;
    // T69 (NINFER_QWEN4_PROMPT_FREQ_WEIGHT=0: full-weight prompt counts, as before).
    policy.recency_weighted_prompts = true;
    if (const char* v = std::getenv("NINFER_QWEN4_PROMPT_FREQ_WEIGHT"); v != nullptr && *v != 0) {
        policy.recency_weighted_prompts = std::atoi(v) != 0;
    }
    // The cap is per GPU (each fills over its own link).
    policy.insert_cap      = options_.insert_cap * partitions_;
    policy.rule            = expert_cache::Rule::kEngine;
    policy.elastic_columns = options_.elastic_columns;

    expert_cache::ExpertLayout layout;
    if (gguf_) {
        // Two planes per expert (the gate|up rows and the down rows of the two host banks), back to
        // back in one slot of the arena: plane 0 at the slot's offset, plane 1 after it.
        layout.planes = 2;
        for (const Qwen4ExpertBank& bank : banks) {
            const std::int64_t gu = 1280 * bank.gguf_gate_up_row_bytes;
            const std::int64_t dn = 2560 * bank.gguf_down_row_bytes;
            layout.bank_base.push_back(bank.gguf_gate_up_rows);
            layout.bank_base.push_back(bank.gguf_down_rows);
            layout.layer_plane_bytes.insert(layout.layer_plane_bytes.end(), {gu, dn});
            layout.layer_bank_stride.insert(layout.layer_bank_stride.end(), {gu, dn});
            layout.layer_plane_offset.insert(layout.layer_plane_offset.end(), {0, gu});
        }
        layout.slot_offset = gguf_pool_.slot_offset;
    } else {
        layout.planes      = 4;
        layout.plane_bytes = {ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
                              ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
        layout.bank_stride = layout.plane_bytes;
        layout.pool_stride = layout.plane_bytes;
        layout.bank_base.reserve(static_cast<std::size_t>(layers_) * 4);
        for (const Qwen4ExpertBank& bank : banks) {
            for (const ops::MoeSlotPlane* plane :
                 {&bank.bank.gate_up_codes, &bank.bank.gate_up_scales, &bank.bank.down_codes,
                  &bank.bank.down_scales}) {
                layout.bank_base.push_back(static_cast<const std::byte*>(plane->base));
            }
        }
    }
    // T25 prefetch: staging slots after the cache's pool slots, same layout.
    if (options_.prefetch.slots_per_layer > 0) {
        if (peer != nullptr) {
            throw std::invalid_argument("Qwen4Exp offload experts: prefetch runs at one device");
        }
        staging_slots_ = Qwen4ExpertPrefetch::staging_slots(options_.prefetch);
        prefetch_      = std::make_unique<Qwen4ExpertPrefetch>(
            options_.prefetch, text_layers_, layout, memory_.pool,
            layers_ * options_.slots_per_layer, memory_.prefetch, device, stream);
    }
    if (options_.warm_start) {
        prompt_rank_.assign(static_cast<std::size_t>(layers_) * kExperts, kExperts);
    }
    std::array<expert_cache::GpuBinding, 2> bindings;
    bindings[0].device = device;
    for (std::size_t q = 0; q < 4; ++q) { bindings[0].pool_base[q] = memory_.pool[q]; }
    if (gguf_) { bindings[0].pool_base[1] = memory_.pool[0]; } // both planes in the arena
    bindings[0].table  = memory_.table;
    bindings[0].mirror = peer != nullptr ? memory_.mirror : nullptr;
    if (peer != nullptr) {
        bindings[1].device = peer->device;
        for (std::size_t q = 0; q < 4; ++q) { bindings[1].pool_base[q] = peer->pool[q]; }
        if (gguf_) { bindings[1].pool_base[1] = peer->pool[0]; }
        bindings[1].table = peer->table;
        if (split_on_peer_) { bindings[1].mirror = peer->mirror; }
    }
    // Elastic columns follow the base slots and the prefetch staging slots (T64).
    policy.elastic_gap = staging_slots_;
    cache_             = std::make_unique<expert_cache::ExpertCache>(
        policy, std::move(layout),
        std::span<const expert_cache::GpuBinding>(bindings.data(),
                                                              static_cast<std::size_t>(partitions_)));
    if (!options_.snapshot_load.empty()) {
        // T59: residency snapshots, one block per request (see Qwen4OffloadOptions).
        std::ifstream in(options_.snapshot_load);
        if (!in) {
            throw std::invalid_argument("NINFER_QWEN4_EXPERT_SNAPSHOT: cannot read " +
                                        options_.snapshot_load);
        }
        for (std::string line; std::getline(in, line);) {
            std::istringstream row(line);
            std::string first;
            if (!(row >> first)) { continue; }
            if (first == "snap" || snapshot_blocks_.empty()) {
                snapshot_blocks_.emplace_back(static_cast<std::size_t>(layers_));
                if (first == "snap") { continue; }
            }
            const std::int32_t layer = std::stoi(first);
            if (layer < 0 || layer >= layers_) {
                throw std::invalid_argument("NINFER_QWEN4_EXPERT_SNAPSHOT: layer out of range");
            }
            auto& list = snapshot_blocks_.back()[static_cast<std::size_t>(layer)];
            for (std::int32_t e = 0; row >> e;) { list.push_back(e); }
        }
        if (snapshot_blocks_.empty()) {
            throw std::invalid_argument("NINFER_QWEN4_EXPERT_SNAPSHOT: empty file");
        }
        cache_->initialize(snapshot_blocks_.front());
        std::int64_t listed = 0;
        for (const auto& l : snapshot_blocks_.front()) {
            listed += static_cast<std::int64_t>(l.size());
        }
        std::fprintf(stderr,
                     "qwen4 experts: snapshot %s, %zu blocks, block 0: %" PRId64
                     " experts listed, %" PRId64 " inserted\n",
                     options_.snapshot_load.c_str(), snapshot_blocks_.size(), listed,
                     cache_->counters().inserts);
    } else {
        cache_->initialize({});
    }
    if (!options_.snapshot_save.empty()) {
        if (std::FILE* f = std::fopen(options_.snapshot_save.c_str(), "w")) { std::fclose(f); }
    }
    cache_->set_frozen(options_.freeze_all);
    // The tables were reset outside the compute streams.
    CUDA_CHECK(cudaDeviceSynchronize());
    if (peer != nullptr) {
        const ScopedCurrentDevice rank1(peer->device);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const bool tracing = !options_.trace_path.empty();
    ops::host::HostMoeMailboxConfig config;
    config.device = device;
    config.layers = layers_;
    // Decode calls hand over <= 8 columns, prefill chunks their whole width in one call.
    config.max_tokens       = std::max(options_.prompt_tokens, ops::kMoeSlotsMaxTokens);
    config.top_k            = kTopK;
    config.hidden           = kHidden;
    config.intermediate     = kInter;
    config.cpus             = ops::host::spread_cpus(options_.cpu_threads);
    config.route_log_tokens = log_tokens_;
    config.gpu_stamps       = tracing;
    // One record per publish: the Text layers once per window, every later (MTP) layer once per
    // call (a prompt chunk and its drafts, or an alignment and the AR drafts of a round; T36 m10).
    config.trace_capacity =
        tracing ? static_cast<std::int64_t>(text_layers_) +
                      static_cast<std::int64_t>(layers_ - text_layers_) * kExtraLayerCalls
                : 0;
    mailbox_ =
        std::make_unique<ops::host::HostMoeMailbox>(std::move(config), std::move(host_layers));
    if (const char* v = std::getenv("NINFER_QWEN4_FILLS_AFTER_HOST"); v != nullptr && *v != 0) {
        options_.fills_after_host = std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("NINFER_QWEN4_REFILL_ORDER"); v != nullptr && *v != 0) {
        options_.refill_order = std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("NINFER_QWEN4_EARLY_RECLAIM"); v != nullptr && *v != 0) {
        options_.early_reclaim = std::atoi(v);
    }
    if (const char* v = std::getenv("NINFER_QWEN4_LEND_KEEP_ELASTIC"); v != nullptr && *v != 0) {
        options_.lend_keep_elastic = std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("NINFER_QWEN4_PROMPT_KEEP"); v != nullptr && *v != 0) {
        options_.keep_decode_residency = std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("NINFER_QWEN4_PROMPT_KEEP_MIN_CACHED");
        v != nullptr && *v != 0) {
        options_.keep_min_cached = std::atof(v);
    }
    if (const char* v = std::getenv("NINFER_QWEN4_PROMPT_NEW_CAP"); v != nullptr && *v != 0) {
        options_.new_prompt_cap = std::max(std::atoi(v), -1);
    }
    // The validation aid decides the residency by the prompt windows (T98): the old rule.
    if (options_.freeze_after_prompt) { options_.keep_decode_residency = false; }
    if (const char* v = std::getenv("NINFER_QWEN4_SCALE_WORKSPACE"); v != nullptr && *v != 0) {
        options_.scale_workspace = std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("NINFER_QWEN4_BOOST_WINDOWS"); v != nullptr && *v != 0) {
        options_.boost_windows = std::atoi(v);
    }
    if (const char* v = std::getenv("NINFER_QWEN4_BOOST_CAP"); v != nullptr && *v != 0) {
        options_.boost_cap = std::atoi(v);
    }
    served_at_begin_.assign(static_cast<std::size_t>(layers_), 0);
    pending_counts_.assign(static_cast<std::size_t>(layers_),
                           std::vector<std::int32_t>(kExperts, 0));
    pending_tokens_.assign(static_cast<std::size_t>(layers_), 0);
    // Prefill streaming: the ring is the slots of the last Text layers' cache rows (whole layers,
    // so lending empties them in the policy); at least half the Text layers keep their residents.
    // At tp 2 each GPU streams its partition (e % 2) into its own ring over its own link (T42).
    // The borrowed prompt workspace (T42) takes the layers just below the ring.
    if (options_.stream_min_tokens > 0) {
        const std::int32_t K = options_.slots_per_layer;
        // T56: at tp 2 a GPU streams only its partition (<= 256 experts a layer), so its ring
        // needs half the slots: fewer lent layers keep their residents during the prompt.
        const std::int32_t ring = ring_slots_for(options_.stream_ring_slots, partitions_);
        lend_count_ =
            ring_layers(options_.stream_ring_slots, layer_base_, text_layers_, partitions_);
        lend_first_ = text_layers_ - lend_count_;
        if (options_.base_chunk > 0 && options_.prompt_workspace_bytes > 0) {
            workspace_layers_ = layers_below(
                layer_base_, lend_first_,
                (options_.prompt_workspace_bytes + kLendPlaneBytes - 1) / kLendPlaneBytes);
            if (!may_lend(layer_base_, text_layers_, lend_count_ + workspace_layers_,
                          partitions_)) {
                std::fprintf(stderr,
                             "Qwen4Exp offload experts: no room to lend a %lld-byte prompt "
                             "workspace (%d slots per layer); prompt chunks "
                             "stay at %d columns\n",
                             static_cast<long long>(options_.prompt_workspace_bytes), K,
                             options_.base_chunk);
                workspace_layers_ = 0;
            } else if (!options_.prompt_workspace_steps.empty()) {
                std::fprintf(stderr,
                             "Qwen4Exp offload experts: prompt chunks up to %d columns, lending "
                             "%d workspace + %d ring layers of %d slots on average\n",
                             options_.prompt_workspace_steps.back().first, workspace_layers_,
                             lend_count_, K);
            }
        }
        const auto any_pageable = [&](std::int32_t first, std::int32_t end) {
            for (std::int32_t l = std::max(first, 0); l < end; ++l) {
                if (pageable_[static_cast<std::size_t>(l)] != 0) { return true; }
            }
            return false;
        };
        // v0.7 pinned fallback: the ring may be lent mid-window (open_stream), when the device
        // tables of its layers still name their slots: a ring layer must stream, so it must be
        // page-locked. The prompt workspace is lent at begin_window only, before the tables are
        // published: a pageable workspace layer then has no resident and runs all its routed
        // experts on the CPU during the chunk (correct, slower), its slots serving as workspace.
        const bool ring_pageable = any_pageable(lend_first_, text_layers_);
        if (!ring_pageable && workspace_layers_ > 0 &&
            any_pageable(lend_first_ - workspace_layers_, lend_first_)) {
            std::fprintf(stderr,
                         "Qwen4Exp offload experts: the prompt workspace lends layers with "
                         "pageable banks; their routed experts run on the CPU in wide prompt "
                         "chunks\n");
        }
        if (lend_count_ > text_layers_ / 2) {
            std::fprintf(stderr,
                         "Qwen4Exp offload experts: %d slots per layer are too few to lend a "
                         "%d-slot streaming ring; prefill streaming is off\n",
                         K, ring);
            workspace_layers_ = 0;
        } else if (ring_pageable) {
            // Turning streaming off here would also drop the lent workspace the Program's prompt
            // arena was planned with: refuse instead (the pin order keeps the top layers pinned
            // down to ~18 GiB, models/qwen3_5/qwen4_layer_profile.h).
            throw std::invalid_argument("Qwen4Exp offload experts: the streaming ring layers " +
                                        std::to_string(lend_first_) + ".." +
                                        std::to_string(text_layers_ - 1) +
                                        " have pageable expert banks; raise --host-pinned-max-gib");
        } else if (peer != nullptr && peer->stream_tables == nullptr) {
            throw std::invalid_argument(
                "Qwen4Exp offload experts: rank 1 streaming tables are missing");
        } else {
            const auto make_stream = [&](int rank) {
                Qwen4ExpertStream::Config sc;
                sc.device      = rank == 0 ? device : peer->device;
                sc.layers      = text_layers_;
                sc.experts     = kExperts;
                sc.plane_bytes = {ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
                                  ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
                sc.bank_base.reserve(static_cast<std::size_t>(text_layers_) * 4);
                for (std::int32_t l = 0; l < text_layers_; ++l) {
                    const Qwen4ExpertBank& bank = banks[static_cast<std::size_t>(l)];
                    for (const ops::MoeSlotPlane* plane :
                         {&bank.bank.gate_up_codes, &bank.bank.gate_up_scales,
                          &bank.bank.down_codes, &bank.bank.down_scales}) {
                        sc.bank_base.push_back(static_cast<const std::byte*>(plane->base));
                    }
                }
                sc.pool_base  = rank == 0 ? memory_.pool : peer->pool;
                sc.ring_first = static_cast<std::int32_t>(base_offset(lend_first_));
                sc.ring_slots =
                    static_cast<std::int32_t>(base_offset(text_layers_) - base_offset(lend_first_));
                sc.tables     = rank == 0 ? memory_.stream_tables : peer->stream_tables;
                sc.partitions = partitions_;
                sc.partition  = rank;
                if (pageable_layers_ > 0) {
                    sc.streamable.resize(static_cast<std::size_t>(text_layers_));
                    for (std::int32_t l = 0; l < text_layers_; ++l) {
                        sc.streamable[static_cast<std::size_t>(l)] =
                            pageable_[static_cast<std::size_t>(l)] != 0 ? 0 : 1;
                    }
                }
                return std::make_unique<Qwen4ExpertStream>(std::move(sc));
            };
            stream_ = make_stream(0);
            if (peer != nullptr) {
                peer_stream_obj_ = make_stream(1);
                // T56: at tp 2 the next layer's copies wait for this layer's MoE all-reduce
                // (Qwen4ExpertStream::set_defer_to_exchange); NINFER_QWEN4_STREAM_DEFER=0 keeps
                // them at release (A/B runs on one binary; same bits).
                const char* defer_env = std::getenv("NINFER_QWEN4_STREAM_DEFER");
                const bool defer      = defer_env == nullptr || std::string(defer_env) != "0";
                stream_->set_defer_to_exchange(defer);
                peer_stream_obj_->set_defer_to_exchange(defer);
                // The host computes nothing of a streamed chunk at tp 2: no CPU tail (its partial
                // would have to cover both GPUs' tables).
                options_.stream_cpu_tail = false;
            }
            lent_profile_.assign(static_cast<std::size_t>(text_layers_), {});
            prompt_counts_.assign(static_cast<std::size_t>(text_layers_),
                                  std::vector<std::int32_t>(kExperts, 0));
            prompt_counts_tokens_.assign(static_cast<std::size_t>(text_layers_), 0);
            stream_select_.assign(static_cast<std::size_t>(text_layers_),
                                  std::vector<std::uint8_t>(kExperts, 1));
        }
    }
    if (gguf_ && gguf_ring_slots_ > 0) {
        const char* gate_env = std::getenv("NINFER_GGUF_RING_GATE");
        gguf_ring_gate_ =
            partitions_ > 1 && !(gate_env != nullptr && std::string_view(gate_env) == "0");
        for (std::int32_t r = 0; r < partitions_; ++r) {
            const ScopedCurrentDevice scope(r == 0 ? device : peer->device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&gguf_copy_stream_[static_cast<std::size_t>(r)],
                                                 cudaStreamNonBlocking));
            for (std::size_t h = 0; h < 2; ++h) {
                CUDA_CHECK(cudaEventCreateWithFlags(&gguf_ready_[static_cast<std::size_t>(r)][h],
                                                    cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&gguf_free_[static_cast<std::size_t>(r)][h],
                                                    cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&gguf_moe_start_[static_cast<std::size_t>(r)],
                                                cudaEventDisableTiming));
        }
        std::fprintf(stderr,
                     "Qwen4Exp offload experts: ggml block banks: prefill ring of %d slots (%d "
                     "layers), prompt chunks up to %d columns on %d workspace layers\n",
                     gguf_ring_slots_, lend_count_,
                     options_.prompt_workspace_steps.empty()
                         ? options_.base_chunk
                         : options_.prompt_workspace_steps.back().first,
                     workspace_layers_);
        lent_profile_.assign(static_cast<std::size_t>(text_layers_), {});
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&enqueued_, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
    if (!options_.routing_path.empty()) {
        routing_ = std::fopen(options_.routing_path.c_str(), "wb");
        if (routing_ == nullptr) {
            throw std::runtime_error("Qwen4Exp routing dump: cannot write " +
                                     options_.routing_path);
        }
    }
    if (tracing) {
        trace_ = std::fopen(options_.trace_path.c_str(), "w");
        if (trace_ == nullptr) {
            throw std::runtime_error("Qwen4Exp expert trace: cannot write " + options_.trace_path);
        }
        clock_offset_ = mailbox_->calibrate_clock(stream);
        std::fprintf(
            trace_,
            "{\"config\":{\"layers\":%d,\"slots_per_layer\":%d,\"cpu_threads\":%d,"
            "\"insert_cap\":%d,\"admit_threshold\":%g,\"half_life_tokens\":%g,"
            "\"evict_tau_windows\":%g,"
            "\"prompt_tokens\":%d,\"slot_pool_bytes\":%" PRId64 ",\"prefetch_slots\":%d,"
            "\"prefetch_candidates\":%d,\"prefetch_min_score\":%g,"
            "\"prefetch_first_layer\":%d,\"prefetch_gbps\":%g,\"prefetch_lead\":%g,"
            "\"warm_start\":%d,\"prompt_keep\":%d,\"stream_min_tokens\":%d,\"split\":{\"mode\":%d,"
            "\"staged\":%d,\"count\":%d,\"fraction\":%g,\"min_misses\":%d,\"max_reads\":%d,"
            "\"host_us\":[%g,%g,%g,%g],\"device_us\":[%g,%g,%g]}}}\n",
            layers_, options_.slots_per_layer, options_.cpu_threads, options_.insert_cap,
            options_.admit_threshold, options_.half_life_tokens, options_.evict_tau_windows,
            options_.prompt_tokens, slot_bytes(layers_) * options_.slots_per_layer,
            options_.prefetch.slots_per_layer, options_.prefetch.candidates,
            static_cast<double>(options_.prefetch.min_score), options_.prefetch.first_layer,
            options_.prefetch.link_gbps, options_.prefetch.lead, options_.warm_start ? 1 : 0,
            options_.keep_decode_residency ? 1 : 0, options_.stream_min_tokens,
            static_cast<int>(options_.split.mode), options_.split_staged ? 1 : 0,
            options_.split.count, options_.split.fraction, options_.split.min_misses,
            options_.split.max_reads, options_.split.host_fixed_us, options_.split.host_expert_us,
            options_.split.host_column_us, options_.split.host_read_slowdown,
            options_.split.device_fixed_us, options_.split.device_expert_us,
            options_.split.device_column_us);
        std::fflush(trace_);
    }
    if (options_.split_calibrate) { setup_split_calibration(); }
}

// ---- --miss-split calibrate ---------------------------------------------------------------------

// The GPU share's cost per expert = the staged copy measured on this machine + the expert kernel
// on the copied slots at decode widths, which does not depend on the link: T18's 140 us per staged
// expert on the 2x RTX 5070 Ti (PCIe 5.0 x8) minus the copy measured there (102.7 us, 26.9 GB/s).
constexpr float kShareComputeUs = 37.0F;

void Qwen4OffloadExperts::setup_split_calibration() {
    if (!splitting() || !options_.split_staged ||
        options_.split.mode != ops::host::HostMoeSplit::Mode::Cost) {
        std::fprintf(stderr, "qwen4 experts: --miss-split calibrate needs the staged auto split "
                             "(off here); costs left as given\n");
        return;
    }
    // The share runs on rank 1 at tp 2 (its own link), else on this GPU. Every bank is page-locked
    // whenever the split is on, so layer 0's bank stands for all.
    const int rank                         = split_on_peer_ ? 1 : 0;
    std::int32_t* picks                    = rank == 1 ? peer_.picks : memory_.picks;
    const std::array<std::byte*, 4> stage  = rank == 1 ? peer_.staging : memory_.staging;
    const ops::MoeSlotPool& b              = banks_.front().bank;
    const std::array<std::int64_t, 4> unit = {
        ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes, ops::kMoeSlotsDownCodeBytes,
        ops::kMoeSlotsDownScaleBytes};
    {
        const ScopedCurrentDevice on(device_ids_[static_cast<std::size_t>(rank)]);
        cudaStream_t s = nullptr;
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        std::array<std::int32_t, kMissSplitSlots + 1> host_picks{};
        for (std::int32_t k = 0; k < kMissSplitSlots; ++k) { host_picks[k] = (k * 37) % kExperts; }
        const auto time_copy = [&](std::int32_t n) {
            host_picks[kMissSplitSlots] = n;
            CUDA_CHECK(
                cudaMemcpy(picks, host_picks.data(), sizeof host_picks, cudaMemcpyHostToDevice));
            float best = std::numeric_limits<float>::max();
            for (int rep = 0; rep < 14; ++rep) {
                CUDA_CHECK(cudaEventRecord(e0, s));
                ops::host::stage_device_share(s, picks,
                                              {static_cast<const std::byte*>(b.gate_up_codes.base),
                                               static_cast<const std::byte*>(b.gate_up_scales.base),
                                               static_cast<const std::byte*>(b.down_codes.base),
                                               static_cast<const std::byte*>(b.down_scales.base)},
                                              stage, {unit[0], unit[1], unit[2], unit[3]});
                CUDA_CHECK(cudaEventRecord(e1, s));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (rep >= 4) { best = std::min(best, 1e3F * ms); }
            }
            return best;
        };
        const float t16      = time_copy(kMissSplitSlots);
        const float t4       = time_copy(4);
        share_copy_us_       = std::max((t16 - t4) / static_cast<float>(kMissSplitSlots - 4), 0.0F);
        share_copy_fixed_us_ = std::max(t4 - 4.0F * share_copy_us_, 0.0F);
        host_picks[kMissSplitSlots] = 0;
        CUDA_CHECK(cudaMemcpy(picks, host_picks.data(), sizeof host_picks, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaStreamDestroy(s));
    }
    // Until the CPU is measured the publishes read a rule with no GPU share (max_reads 0).
    ops::host::HostMoeSplit measuring = options_.split;
    measuring.max_reads               = 0;
    for (int r = 0; r < (split_on_peer_ ? 2 : 1); ++r) {
        const ScopedCurrentDevice on(device_ids_[static_cast<std::size_t>(r)]);
        CUDA_CHECK(cudaMalloc(&split_live_[static_cast<std::size_t>(r)], sizeof measuring));
        CUDA_CHECK(cudaMemcpy(split_live_[static_cast<std::size_t>(r)], &measuring,
                              sizeof measuring, cudaMemcpyHostToDevice));
    }
    mailbox_->collect_cost_samples(kSplitCalibrationCalls);
    const double bytes = static_cast<double>(unit[0] + unit[1] + unit[2] + unit[3]);
    std::fprintf(stderr,
                 "qwen4 experts: miss split calibrate: GPU %d copies an expert in %.1f us "
                 "(%.1f GB/s, + %.1f us per copy); the first %" PRId64
                 " decode calls keep every miss on the CPU and time it\n",
                 device_ids_[static_cast<std::size_t>(rank)], share_copy_us_,
                 share_copy_us_ > 0.0F ? bytes / (1e3 * share_copy_us_) : 0.0, share_copy_fixed_us_,
                 kSplitCalibrationCalls);
}

void Qwen4OffloadExperts::finish_split_calibration(cudaStream_t stream) {
    const std::vector<ops::host::HostMoeCostSample> samples = mailbox_->cost_samples();
    if (static_cast<std::int64_t>(samples.size()) < kSplitCalibrationCalls) { return; }
    mailbox_->collect_cost_samples(0);
    split_calibrated_ = true;
    const ops::host::HostMoeCostFit fit =
        ops::host::fit_host_costs(samples, options_.split.host_column_us);
    ops::host::HostMoeSplit rule = options_.split;
    if (fit.samples > 0 && fit.expert_us > 0.0F) {
        rule.host_fixed_us  = fit.fixed_us;
        rule.host_expert_us = fit.expert_us;
        rule.host_column_us = fit.column_us;
    }
    rule.device_expert_us = share_copy_us_ + kShareComputeUs;
    rule.device_fixed_us  = std::max(options_.split.device_fixed_us, share_copy_fixed_us_);
    split_measured_       = rule;
    // Stream order on both ranks: this window's publish (rank 0) and share selection (rank 1) read
    // the same rule, so the two sides never disagree on which experts the GPU takes.
    for (int r = 0; r < (split_on_peer_ ? 2 : 1); ++r) {
        const ScopedCurrentDevice on(device_ids_[static_cast<std::size_t>(r)]);
        CUDA_CHECK(cudaMemcpyAsync(split_live_[static_cast<std::size_t>(r)], &split_measured_,
                                   sizeof split_measured_, cudaMemcpyHostToDevice,
                                   r == 0 ? stream : peer_stream_));
        CUDA_CHECK(cudaStreamSynchronize(r == 0 ? stream : peer_stream_));
    }
    options_.split = rule; // the exit line and the trace report the rule in use
    std::fprintf(stderr,
                 "qwen4 experts: miss split calibrated on %zu decode calls (%d kept, rms %.1f us): "
                 "CPU %.1f us + %.1f us per expert + %.1f us per extra column, GPU %.1f us + "
                 "%.1f us per expert%s\n",
                 samples.size(), fit.samples, fit.rms_us, rule.host_fixed_us, rule.host_expert_us,
                 rule.host_column_us, rule.device_fixed_us, rule.device_expert_us,
                 fit.samples > 0 && fit.expert_us > 0.0F ? ""
                                                         : " (CPU fit failed: CPU costs as given)");
}

Qwen4OffloadSplitCounters Qwen4OffloadExperts::split_counters() const noexcept {
    return split_totals_;
}

void Qwen4OffloadExperts::snapshot_save_block() {
    while (cache_->fills_pending() > 0) { // the residency the policy decided, landed
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        cache_->poll();
    }
    std::FILE* f = std::fopen(options_.snapshot_save.c_str(), "a");
    if (f == nullptr) {
        throw std::runtime_error("NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE: cannot open " +
                                 options_.snapshot_save);
    }
    std::fprintf(f, "snap %" PRId64 "\n", snapshot_saved_++);
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        std::fprintf(f, "%d", layer);
        for (const std::int32_t e : cache_->resident_experts(layer)) { std::fprintf(f, " %d", e); }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

Qwen4OffloadExperts::~Qwen4OffloadExperts() {
    for (std::size_t r = 0; r < split_live_.size(); ++r) {
        if (split_live_[r] == nullptr) { continue; }
        const ScopedCurrentDevice on(device_ids_[r]);
        (void)cudaFree(split_live_[r]);
    }
    for (std::size_t r = 0; r < 2; ++r) {
        if (gguf_copy_stream_[r] != nullptr) { (void)cudaStreamDestroy(gguf_copy_stream_[r]); }
        for (std::size_t h = 0; h < 2; ++h) {
            if (gguf_ready_[r][h] != nullptr) { (void)cudaEventDestroy(gguf_ready_[r][h]); }
            if (gguf_free_[r][h] != nullptr) { (void)cudaEventDestroy(gguf_free_[r][h]); }
            if (h == 0 && gguf_moe_start_[r] != nullptr) {
                (void)cudaEventDestroy(gguf_moe_start_[r]);
            }
        }
    }
    if (split_totals_.decode_windows > 0) {
        // One line per backend (stderr): the experts each side computed over the decode windows.
        const Qwen4OffloadSplitCounters c = split_counters();
        const double w                    = static_cast<double>(c.decode_windows);
        std::fprintf(
            stderr,
            "qwen4 experts: %" PRId64 " decode windows, per window %.1f routes, device hit "
            "%.4f, missing %.1f, GPU share %.1f experts (%.1f routes), CPU %.1f experts; "
            "split mode %d%s\n",
            c.decode_windows, static_cast<double>(c.accesses) / w,
            c.accesses ? static_cast<double>(c.device_hits) / static_cast<double>(c.accesses) : 0.0,
            static_cast<double>(c.missing) / w, static_cast<double>(c.gpu_experts) / w,
            static_cast<double>(c.gpu_routes) / w, static_cast<double>(c.cpu_experts) / w,
            static_cast<int>(options_.split.mode), options_.split_staged ? " staged" : "");
    }
    if (trace_ != nullptr) {
        if (decode_windows_ > 0) {
            std::fprintf(trace_, "{\"summary\":\"decode\",\"windows\":%" PRId64 ",",
                         decode_windows_);
            write_stats(trace_, decode_sum_, static_cast<double>(decode_windows_));
            const auto& c = cache_->counters();
            std::fprintf(trace_,
                         ",\"cache\":{\"windows\":%" PRId64 ",\"accesses\":%" PRId64
                         ",\"device_hits\":%" PRId64 ",\"policy_hits\":%" PRId64
                         ",\"fills_issued\":%" PRId64 ",\"fills_dropped\":%" PRId64 "}}\n",
                         c.windows, c.accesses, c.device_hits, c.policy_hits, c.fills_issued,
                         c.fills_dropped);
            std::fprintf(trace_, "{\"summary\":\"columns\",\"layer_windows\":[");
            for (std::size_t i = 0; i < column_layers_.size(); ++i) {
                std::fprintf(trace_, "%s%" PRId64, i ? "," : "", column_layers_[i]);
            }
            std::fprintf(trace_, "],\"distinct\":[");
            for (std::size_t i = 0; i < column_distinct_.size(); ++i) {
                std::fprintf(trace_, "%s%" PRId64, i ? "," : "", column_distinct_[i]);
            }
            std::fprintf(trace_, "],\"missing\":[");
            for (std::size_t i = 0; i < column_missing_.size(); ++i) {
                std::fprintf(trace_, "%s%" PRId64, i ? "," : "", column_missing_[i]);
            }
            std::fprintf(trace_, "]}\n");
        }
        if (prefetch_) {
            const auto& p = prefetch_->counters();
            std::fprintf(trace_,
                         "{\"summary\":\"prefetch\",\"windows\":%" PRId64 ",\"predicted\":%" PRId64
                         ",\"candidates\":%" PRId64 ",\"issued\":%" PRId64 ",\"landed\":%" PRId64
                         ",\"used\":%" PRId64 ",\"used_routes\":%" PRId64 ",\"budget_cut\":%" PRId64
                         ",\"layer_period_us\":%.1f,\"rate_scale\":%.3f}\n",
                         p.windows, p.predicted, p.candidates, p.issued, p.landed, p.used,
                         p.used_routes, p.budget_cut, prefetch_->layer_period_us(),
                         prefetch_->rate_scale());
        }
        std::fclose(trace_);
    }
    if (routing_ != nullptr) { std::fclose(routing_); }
    close_stream();
    stream_.reset();
    mailbox_.reset();
    if (side_ != nullptr) { (void)cudaStreamSynchronize(side_); }
    for (cudaEvent_t e : {enqueued_, fork_, join_}) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
    if (side_ != nullptr) { (void)cudaStreamDestroy(side_); }
}

Tensor Qwen4OffloadExperts::layer_table(std::int32_t layer) const {
    return Tensor(memory_.table + static_cast<std::size_t>(layer) * kExperts, DType::I32,
                  {kExperts});
}

const std::int32_t* Qwen4OffloadExperts::residency_table(std::int32_t layer) const {
    if (partitions_ == 1) { return static_cast<const std::int32_t*>(layer_table(layer).data); }
    return memory_.mirror + static_cast<std::size_t>(layer) * kExperts;
}

std::span<const cudaStream_t> Qwen4OffloadExperts::window_streams(cudaStream_t stream) {
    streams_[0] = stream;
    return std::span<const cudaStream_t>(streams_.data(), static_cast<std::size_t>(partitions_));
}

void Qwen4OffloadExperts::fork_device_reads(std::int32_t layer, const Qwen4ExpertBank& bank,
                                            const Tensor& x, const Tensor& weights,
                                            const Tensor& out, WorkspaceArena& work,
                                            cudaStream_t stream) {
    // The bank itself is the pool (slot = expert id through the identity table); the divisors are
    // the device copy.
    const std::int32_t T  = x.ne[1];
    const float* divisors = memory_.divisors + static_cast<std::size_t>(layer) * kDivisorsPerLayer;
    const bool staged     = options_.split_staged;
    const Tensor table(staged ? memory_.stage_table : memory_.identity, DType::I32, {kExperts});
    const ops::MoeSlotsLayer view = slots_layer(bank, divisors, divisors + kGateUpDivisors, table);
    const Tensor reads(memory_.reads, DType::I32, {kTopK, T});
    WorkspaceArena scratch(work.alloc_bytes(ops::moe_slots_nvfp4_workspace_bytes(T)));
    Tensor partial = out;
    CUDA_CHECK(cudaEventRecord(fork_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(side_, fork_, 0));
    if (staged) {
        // Every byte of the picked experts crosses the link once, into VRAM scratch slots; the
        // expert kernel then reads them there (scratch table: picked id -> slot, else -1).
        const ops::MoeSlotPool& b = bank.bank;
        ops::host::stage_device_share(side_, memory_.picks,
                                      {static_cast<const std::byte*>(b.gate_up_codes.base),
                                       static_cast<const std::byte*>(b.gate_up_scales.base),
                                       static_cast<const std::byte*>(b.down_codes.base),
                                       static_cast<const std::byte*>(b.down_scales.base)},
                                      memory_.staging,
                                      {ops::kMoeSlotsGateUpCodeBytes,
                                       ops::kMoeSlotsGateUpScaleBytes, ops::kMoeSlotsDownCodeBytes,
                                       ops::kMoeSlotsDownScaleBytes});
        const ops::MoeSlotPool pool{
            .gate_up_codes  = {memory_.staging[0], ops::kMoeSlotsGateUpCodeBytes},
            .gate_up_scales = {memory_.staging[1], ops::kMoeSlotsGateUpScaleBytes},
            .down_codes     = {memory_.staging[2], ops::kMoeSlotsDownCodeBytes},
            .down_scales    = {memory_.staging[3], ops::kMoeSlotsDownScaleBytes},
            .slots          = ops::host::HostMoeSplit::kMaxReads,
        };
        ops::moe_slots_nvfp4(x, reads, weights, view, pool, ops::MoeSlotsOutput::Store, partial,
                             scratch, side_);
    } else {
        ops::moe_slots_nvfp4(x, reads, weights, view, bank.bank, ops::MoeSlotsOutput::Store,
                             partial, scratch, side_);
    }
    CUDA_CHECK(cudaEventRecord(join_, side_));
}

void Qwen4OffloadExperts::ahead(std::int32_t layer, const Tensor& wide, const Tensor& next_norm,
                                const Tensor& next_router, float eps, cudaStream_t stream) {
    if (!prefetch_ || !prefetch_->predicts(layer) || wide.ne[1] > ops::kMoeSlotsMaxTokens) {
        return;
    }
    prefetch_->predict(layer, wide, next_norm, next_router, eps,
                       memory_.table + static_cast<std::size_t>(layer + 1) * kExperts, stream);
}

void Qwen4OffloadExperts::routed(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                                 const Tensor& ids, const Tensor& weights, WorkspaceArena& work,
                                 cudaStream_t stream) {
    if (layer < 0 || layer >= layers_) {
        throw std::invalid_argument("Qwen4Exp offload experts: layer out of range");
    }
    const std::int32_t T = x.ne[1];
    published_layer_     = -1;
    overlay_layer_       = -1;
    side_partial_        = Tensor{};
    // A Text forward begins: the later layers' routing logs restart (the offsets are fixed when a
    // round is captured, so they must follow the forward, not the window).
    if (layer == 0) { std::fill(extra_offset_.begin(), extra_offset_.end(), 0); }
    if (T > ops::kMoeSlotsMaxTokens) { return; } // run() publishes slice by slice
    // T25: the layer's effective row (table + staged experts that have landed) for both paths.
    // At tp 2 the publish reads rank 0's residency mirror of both GPUs (T33).
    const std::int32_t* row = residency_table(layer);
    if (prefetch_ && prefetch_->stages(layer)) {
        row = prefetch_->overlay(layer, row, static_cast<const std::int32_t*>(ids.data), kTopK * T,
                                 stream);
        overlay_layer_ = layer;
        overlay_row_   = row;
    }
    // The doorbell rings before the shared expert, so the host starts while the GPU works.
    mailbox_->publish_table(
        stream, layer, T, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(weights.data), row,
        kExperts, log_offset(layer, T), options_.split,
        ops::host::HostMoeDeviceShare{.reads = memory_.reads,
                                      .picks = options_.split_staged ? memory_.picks : nullptr,
                                      .stage_table =
                                          options_.split_staged ? memory_.stage_table : nullptr,
                                      .live = split_live_[0]});
    published_layer_ = layer;
    if (splitting() && !split_on_peer_) {
        side_partial_ = work.alloc(DType::FP32, {kHidden, T});
        fork_device_reads(layer, bank, x, weights, side_partial_, work, stream);
    }
}

void Qwen4OffloadExperts::run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                              const Tensor& ids, const Tensor& weights, Tensor& partial,
                              WorkspaceArena& work, cudaStream_t stream) {
    require_operands(x, ids, weights, partial);
    if (layer < 0 || layer >= layers_) {
        throw std::invalid_argument("Qwen4Exp offload experts: layer out of range");
    }
    const std::int32_t T = x.ne[1];
    if (gguf_) {
        run_gguf(layer, bank, x, ids, weights, partial, work, stream);
        return;
    }
    if (streams_call(layer, T)) {
        stream_call(0, layer, bank, x, ids, weights, partial, work, stream);
        return;
    }
    // A decode call of a staged layer reads the effective row routed() built (T25).
    const bool staged = T <= ops::kMoeSlotsMaxTokens && overlay_layer_ == layer;
    const Tensor table =
        staged ? Tensor(const_cast<std::int32_t*>(overlay_row_), DType::I32, {kExperts})
               : layer_table(layer);
    overlay_layer_        = -1;
    const float* divisors = memory_.divisors + static_cast<std::size_t>(layer) * kDivisorsPerLayer;
    const ops::MoeSlotsLayer view = slots_layer(bank, divisors, divisors + kGateUpDivisors, table);
    const ops::MoeSlotPool pool =
        slot_pool(memory_.pool, planar_strides(),
                  layers_ * (options_.slots_per_layer + options_.elastic_columns) + staging_slots_);
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(
        ops::moe_slots_nvfp4_workspace_bytes(std::min<std::int32_t>(T, ops::kMoeSlotsMaxTokens))));
    if (T <= ops::kMoeSlotsMaxTokens) {
        if (published_layer_ != layer) {
            throw std::logic_error("Qwen4Exp offload experts: run() without routed() of the layer");
        }
        published_layer_ = -1;
        ops::moe_slots_nvfp4(x, ids, weights, view, pool, ops::MoeSlotsOutput::Store, partial,
                             scratch, stream);
        mailbox_->stamp(stream, layer, ops::host::kStampDevice);
        const float* side = nullptr;
        if (side_partial_.data != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join_, 0));
            side          = static_cast<const float*>(side_partial_.data);
            side_partial_ = Tensor{};
        }
        mailbox_->accumulate(stream, layer, T, static_cast<float*>(partial.data), side);
        return;
    }
    // Wider calls (prefill chunks): one handshake for the whole call (the host reads each
    // missing expert once for all its columns), the resident experts in calls of <= 8 columns.
    mailbox_->publish_table(stream, layer, T, static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const std::int32_t*>(ids.data),
                            static_cast<const float*>(weights.data), residency_table(layer),
                            kExperts, log_offset(layer, T));
    for (std::int32_t begin = 0; begin < T; begin += ops::kMoeSlotsMaxTokens) {
        const std::int32_t count = std::min(ops::kMoeSlotsMaxTokens, T - begin);
        Tensor out               = partial.slice(1, begin, count);
        ops::moe_slots_nvfp4(x.slice(1, begin, count), ids.slice(1, begin, count),
                             weights.slice(1, begin, count), view, pool, ops::MoeSlotsOutput::Store,
                             out, scratch, stream);
    }
    mailbox_->accumulate(stream, layer, T, static_cast<float*>(partial.data));
}

void Qwen4OffloadExperts::upload_gguf_host_rows(cudaStream_t stream) {
    const auto cells = static_cast<std::size_t>(text_layers_) * kExperts;
    const bool ring  = gguf_ring_ready();
    for (std::int32_t r = 0; r < partitions_; ++r) {
        auto& rows   = gguf_host_rows_[static_cast<std::size_t>(r)];
        auto& copies = gguf_ring_copies_[static_cast<std::size_t>(r)];
        rows.assign(cells, -1);
        copies.assign(static_cast<std::size_t>(text_layers_), {});
        gguf_issued_[static_cast<std::size_t>(r)].assign(static_cast<std::size_t>(text_layers_), 0);
        for (std::int32_t l = 0; l < text_layers_; ++l) {
            const Qwen4ExpertBank& bank = banks_[static_cast<std::size_t>(l)];
            const std::int64_t gu       = 1280 * bank.gguf_gate_up_row_bytes;
            const std::int64_t dn       = 2560 * bank.gguf_down_row_bytes;
            std::int32_t k              = 0;
            for (std::int32_t e = r; e < kExperts; e += partitions_) {
                const std::int32_t slot = cache_->published_slot(l, e);
                auto& cell =
                    rows[static_cast<std::size_t>(l) * kExperts + static_cast<std::size_t>(e)];
                if (!ring) {
                    // Zero-copy: host_table row of a non-resident expert.
                    if (slot < 0) { cell = e; }
                    continue;
                }
                // Ring: the stream table holds the resident slot or the ring slot.
                if (slot >= 0) {
                    cell = slot;
                    continue;
                }
                const std::int32_t j  = (l % 2) * (gguf_ring_slots_ / 2) + k; // half l % 2
                cell                  = pool_slots_ + j;
                const std::int64_t at = gguf_ring_base_ + j * gguf_ring_stride_;
                auto& c               = copies[static_cast<std::size_t>(l)];
                c.dst.insert(c.dst.end(), {at, at + gu});
                c.src.insert(c.src.end(),
                             {bank.gguf_gate_up_rows + static_cast<std::int64_t>(e) * gu,
                              bank.gguf_down_rows + static_cast<std::int64_t>(e) * dn});
                c.bytes.insert(c.bytes.end(),
                               {static_cast<std::size_t>(gu), static_cast<std::size_t>(dn)});
                ++k;
            }
        }
    }
    // Stream-ordered before the window's chunk on each rank's compute stream.
    CUDA_CHECK(cudaMemcpyAsync(memory_.stream_tables, gguf_host_rows_[0].data(),
                               sizeof(std::int32_t) * cells, cudaMemcpyHostToDevice, stream));
    if (partitions_ == 2) {
        const ScopedCurrentDevice rank1(peer_.device);
        CUDA_CHECK(cudaMemcpyAsync(peer_.stream_tables, gguf_host_rows_[1].data(),
                                   sizeof(std::int32_t) * cells, cudaMemcpyHostToDevice,
                                   peer_.stream));
    }
}

void Qwen4OffloadExperts::gguf_issue_copies(int rank, std::int32_t layer, cudaEvent_t after) {
    auto& issued = gguf_issued_[static_cast<std::size_t>(rank)];
    if (layer >= text_layers_ || issued[static_cast<std::size_t>(layer)]) { return; }
    issued[static_cast<std::size_t>(layer)] = 1;
    const auto r                            = static_cast<std::size_t>(rank);
    const auto half                         = static_cast<std::size_t>(layer % 2);
    cudaStream_t copy                       = gguf_copy_stream_[r];
    // The ring's slots held lent experts: their fills issued before the lend must have landed;
    // the half must be free again (the grouped call of layer - 2 has read it).
    CUDA_CHECK(cudaStreamWaitEvent(copy, cache_->lend_events()[r], 0));
    CUDA_CHECK(cudaStreamWaitEvent(copy, gguf_free_[r][half], 0));
    if (after != nullptr) { CUDA_CHECK(cudaStreamWaitEvent(copy, after, 0)); }
    const auto& c = gguf_ring_copies_[r][static_cast<std::size_t>(layer)];
    if (!c.dst.empty()) {
        std::byte* arena = (rank == 0 ? memory_.pool : peer_pool_)[0];
        ring_dst_.clear();
        ring_src_.clear();
        ring_bytes_.clear();
        for (std::size_t i = 0; i < c.dst.size(); ++i) {
            ring_dst_.push_back(arena + c.dst[i]);
            ring_src_.push_back(c.src[i]);
            ring_bytes_.push_back(c.bytes[i]);
        }
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder    = cudaMemcpySrcAccessOrderStream;
        attr.flags             = cudaMemcpyFlagPreferOverlapWithCompute;
        std::size_t attr_index = 0;
        CUDA_CHECK(cudaMemcpyBatchAsync(ring_dst_.data(), ring_src_.data(), ring_bytes_.data(),
                                        ring_dst_.size(), &attr, &attr_index, 1, copy));
    }
    CUDA_CHECK(cudaEventRecord(gguf_ready_[r][half], copy));
}

bool Qwen4OffloadExperts::gguf_ring_copy(int rank, std::int32_t layer, cudaStream_t stream) {
    if (!gguf_ring_ready() || !open_ || kind_ != Qwen4Window::Prompt) { return false; }
    // This layer's copies (normally issued while the previous layer computed), then the next
    // layer's into the other half, overlapping this layer's work; the compute waits for its own.
    gguf_issue_copies(rank, layer);
    cudaEvent_t gate = nullptr;
    if (gguf_ring_gate_) {
        gate = gguf_moe_start_[static_cast<std::size_t>(rank)];
        CUDA_CHECK(cudaEventRecord(gate, stream));
    }
    gguf_issue_copies(rank, layer + 1, gate);
    CUDA_CHECK(cudaStreamWaitEvent(
        stream, gguf_ready_[static_cast<std::size_t>(rank)][static_cast<std::size_t>(layer % 2)],
        0));
    return true;
}

void Qwen4OffloadExperts::gguf_ring_done(int rank, std::int32_t layer, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(
        gguf_free_[static_cast<std::size_t>(rank)][static_cast<std::size_t>(layer % 2)], stream));
}

void Qwen4OffloadExperts::run_gguf(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                                   const Tensor& ids, const Tensor& weights, Tensor& partial,
                                   WorkspaceArena& work, cudaStream_t stream) {
    const std::int32_t T = x.ne[1];
    ops::MoeGgufLayer view =
        gguf_view(layer_table(layer), bank, pool_slots_, memory_.slot_offsets, memory_.pool[0]);
    if (gguf_streams(layer, T)) {
        // Prefill chunk on the GPU alone: the policy sees the routing, the CPU computes nothing;
        // the experts of this GPU's partition not resident come from the host bank in place.
        mailbox_->log_routes(stream, layer, T, static_cast<const std::int32_t*>(ids.data),
                             log_offset(layer, T));
        const std::int32_t* row =
            memory_.stream_tables + static_cast<std::size_t>(layer) * kExperts;
        const bool ring = gguf_ring_copy(0, layer, stream);
        if (ring) {
            // Ring: the layer's streamed experts copied into the ring (DMA), every expert of this
            // GPU's partition then read from VRAM through the stream table.
            view.slot_table = Tensor(const_cast<std::int32_t*>(row), DType::I32, {kExperts});
            view.slots      = gguf_stream_slots();
        } else {
            view.host_table = row; // zero-copy
        }
        gguf_partial(x, ids, weights, view, partial, work, stream);
        if (ring) { gguf_ring_done(0, layer, stream); }
        return;
    }
    if (T <= ops::kMoeSlotsMaxTokens) {
        if (published_layer_ != layer) {
            throw std::logic_error("Qwen4Exp offload experts: run() without routed() of the layer");
        }
        published_layer_ = -1;
        overlay_layer_   = -1;
        gguf_partial(x, ids, weights, view, partial, work, stream);
        mailbox_->stamp(stream, layer, ops::host::kStampDevice);
        mailbox_->accumulate(stream, layer, T, static_cast<float*>(partial.data), nullptr);
        return;
    }
    // Prefill chunks: one handshake for the whole call (the CPU reads each missing expert once
    // for all its columns), the resident experts grouped by expert on the GPU.
    mailbox_->publish_table(stream, layer, T, static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const std::int32_t*>(ids.data),
                            static_cast<const float*>(weights.data), residency_table(layer),
                            kExperts, log_offset(layer, T));
    gguf_partial(x, ids, weights, view, partial, work, stream);
    mailbox_->accumulate(stream, layer, T, static_cast<float*>(partial.data));
}

// A wide prefill call of a Text layer: every routed expert on the GPU, the non-resident ones
// streamed into the lent ring (Qwen4ExpertStream); the routing goes to the policy's log only.
bool Qwen4OffloadExperts::streams_call(std::int32_t layer, std::int32_t columns) const noexcept {
    return stream_ != nullptr && columns > ops::kMoeSlotsMaxTokens &&
           columns >= options_.stream_min_tokens && layer < text_layers_ &&
           pageable_[static_cast<std::size_t>(layer)] == 0;
}

void Qwen4OffloadExperts::open_stream(std::int32_t tokens) {
    // The first streamed call of the window: lend the ring when the prompt has not lent it yet
    // (mid-window: no kernel of this window reads the lent layers through the cache table, every
    // Text layer of a chunk takes this path), then open each GPU's stream window.
    if (!cache_->lending()) {
        save_lent_residency(lend_first_, lend_count_);
        (void)cache_->lend_layers(lend_first_, lend_count_, options_.lend_keep_elastic);
    }
    // The copies into the ring wait for every fill issued before the lend (begin_window may have
    // lent the layers already, for this chunk or an earlier one of the prompt).
    std::array<cudaEvent_t, 2> wait{};
    const auto events = cache_->lend_events();
    for (std::size_t p = 0; p < events.size(); ++p) { wait[p] = events[p]; }
    stream_->begin_window(cache_.get(), wait[0]);
    if (peer_stream_obj_) { peer_stream_obj_->begin_window(cache_.get(), wait[1]); }
    streaming_ = true;
    if (options_.stream_cpu_tail && tokens <= options_.stream_cpu_max_tokens) {
        select_cpu_tail(tokens);
    }
}

void Qwen4OffloadExperts::run_peer_streamed(std::int32_t layer, const Qwen4ExpertBank& bank,
                                            const Tensor& x, const Tensor& ids,
                                            const Tensor& weights, Tensor& partial,
                                            WorkspaceArena& work, cudaStream_t stream) {
    if (!peer_stream_obj_) {
        throw std::logic_error("Qwen4Exp offload experts: rank 1 does not stream");
    }
    stream_call(1, layer, bank, x, ids, weights, partial, work, stream);
}

void Qwen4OffloadExperts::stream_call(int rank, std::int32_t layer, const Qwen4ExpertBank& bank,
                                      const Tensor& x, const Tensor& ids, const Tensor& weights,
                                      Tensor& partial, WorkspaceArena& work, cudaStream_t stream) {
    if (!open_ || kind_ != Qwen4Window::Prompt) {
        throw std::logic_error(
            "Qwen4Exp offload experts: a streamed call needs an open prompt window");
    }
    const std::int32_t T = x.ne[1];
    if (!streaming_) { open_stream(T); }
    Qwen4ExpertStream& s = rank == 0 ? *stream_ : *peer_stream_obj_;
    s.acquire(layer, stream);
    // Experts of this GPU's partition the layer leaves to the host executor (table -1), counted
    // when the copies were issued; the host computes the routed ones among them in parallel with
    // the device (rank 0 only: at tp 2 there is no CPU tail).
    bool host_experts = false;
    if (rank == 0) {
        const std::int32_t* t = s.host_table(layer);
        for (std::int32_t e = 0; e < kExperts && !host_experts; e += partitions_) {
            host_experts = t[e] < 0;
        }
    }
    const Tensor table(const_cast<std::int32_t*>(s.table(layer)), DType::I32, {kExperts});
    const float* divisor_base = rank == 0 ? memory_.divisors : peer_divisors_;
    const float* divisors     = divisor_base + static_cast<std::size_t>(layer) * kDivisorsPerLayer;
    const ops::MoeSlotsLayer view = slots_layer(bank, divisors, divisors + kGateUpDivisors, table);
    const ops::MoeSlotPool pool =
        slot_pool(rank == 0 ? memory_.pool : peer_pool_, planar_strides(),
                  layers_ * (options_.slots_per_layer + options_.elastic_columns) +
                      (rank == 0 ? staging_slots_ : 0));
    if (rank == 0) {
        if (host_experts) {
            mailbox_->publish_table(
                stream, layer, T, static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(weights.data),
                static_cast<const std::int32_t*>(table.data), kExperts, log_offset(layer, T));
        } else {
            // The routing still reaches the cache policy (the prompt warms the decode cache).
            mailbox_->log_routes(stream, layer, T, static_cast<const std::int32_t*>(ids.data),
                                 log_offset(layer, T));
        }
    }
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(ops::moe_grouped_nvfp4_workspace_bytes(T)));
    Qwen4ColumnGroups* groups = qwen4_column_groups;
    const std::int32_t G      = groups != nullptr ? groups->group_columns : 0;
    if (G > 0 && !host_experts && T > G && groups->done == 0) {
        // T62: in column groups, each reported as soon as it is enqueued (its exchange overlaps
        // the next group). Groups are whole pieces of the grouped kernel: the same bits.
        for (std::int32_t first = 0; first < T; first += G) {
            const std::int32_t n = std::min(G, T - first);
            Tensor out           = partial.slice(1, first, n);
            ops::moe_grouped_nvfp4(x.slice(1, first, n), ids.slice(1, first, n),
                                   weights.slice(1, first, n), view, pool, out, scratch, stream);
            if (first + n < T) {
                groups->ready(first, n);
                groups->done = first + n;
            }
        }
    } else {
        ops::moe_grouped_nvfp4(x, ids, weights, view, pool, partial, scratch, stream);
    }
    s.release(layer, stream);
    if (host_experts) { mailbox_->accumulate(stream, layer, T, static_cast<float*>(partial.data)); }
}

void Qwen4OffloadExperts::exchanged(std::int32_t layer, cudaStream_t stream) {
    if (streaming_ && layer < text_layers_) { stream_->exchanged(layer, stream); }
}

void Qwen4OffloadExperts::exchanged_peer(std::int32_t layer, cudaStream_t stream) {
    if (streaming_ && peer_stream_obj_ && layer < text_layers_) {
        peer_stream_obj_->exchanged(layer, stream);
    }
}

void Qwen4OffloadExperts::select_cpu_tail(std::int32_t tokens) {
    // Per layer: the non-resident experts by predicted routes ascending (the previous prompt
    // window's counts scaled to this width); the CPU takes them in that order while its
    // estimated time (fixed cost weighted by the chance the expert is routed at all, plus a cost
    // per routed token) stays within the copy time of the experts still streamed.
    const double a = options_.stream_cpu_expert_us, b = options_.stream_cpu_token_us;
    const double link = options_.stream_link_us;
    std::vector<std::pair<double, std::int32_t>> order;
    for (std::int32_t l = 0; l < text_layers_; ++l) {
        auto& select = stream_select_[static_cast<std::size_t>(l)];
        std::fill(select.begin(), select.end(), 1);
        const std::int32_t previous = prompt_counts_tokens_[static_cast<std::size_t>(l)];
        if (previous <= 0) {
            stream_->set_selection(l, nullptr);
            continue;
        }
        const double scale = static_cast<double>(tokens) / previous;
        const auto& counts = prompt_counts_[static_cast<std::size_t>(l)];
        const bool lent    = cache_->lent(l);
        order.clear();
        for (std::int32_t e = 0; e < kExperts; ++e) {
            if (!lent && cache_->published_slot(l, e) >= 0) { continue; }
            order.emplace_back(scale * counts[static_cast<std::size_t>(e)], e);
        }
        std::stable_sort(order.begin(), order.end());
        double cpu    = 0.0;
        auto streamed = static_cast<double>(order.size());
        for (const auto& [routes, e] : order) {
            const double cost = a * std::min(1.0, routes) + b * routes;
            if (cpu + cost > (streamed - 1.0) * link) { break; }
            cpu += cost;
            streamed -= 1.0;
            select[static_cast<std::size_t>(e)] = 0;
        }
        stream_->set_selection(l, select.data());
    }
}

void Qwen4OffloadExperts::close_stream() noexcept {
    if (!streaming_) { return; }
    stream_->end_window();
    auto s = stream_->stats();
    if (peer_stream_obj_) {
        peer_stream_obj_->end_window();
        const auto& p = peer_stream_obj_->stats();
        s.streamed += p.streamed;
        s.bytes += p.bytes;
    }
    window_.streamed_experts = s.streamed;
    window_.streamed_bytes   = s.bytes;
    window_.streamed_layers  = s.layers;
    streaming_               = false;
}

std::int32_t Qwen4OffloadExperts::log_offset(std::int32_t layer, std::int32_t tokens) {
    if (layer < text_layers_) { return 0; }
    std::int32_t& next = extra_offset_[static_cast<std::size_t>(layer - text_layers_)];
    if (next + tokens > log_tokens_) { next = 0; }
    const std::int32_t at = next;
    next += tokens;
    return at;
}

void Qwen4OffloadExperts::require_executor() const {
    // A failed executor answers every doorbell with no partial (HostMoeMailbox), so every
    // non-resident expert would silently contribute nothing: fail the execution instead.
    if (mailbox_->failed()) {
        throw std::runtime_error("Qwen4Exp offload experts: the host expert executor failed: " +
                                 mailbox_->error());
    }
}

void Qwen4OffloadExperts::reclaim_lent(std::int32_t layers) {
    const auto first = static_cast<std::size_t>(cache_->lent_first());
    const auto count = static_cast<std::size_t>(layers);
    refill_lent(layers);
    if (options_.keep_decode_residency) {
        // T211: the saved residency is spent; the next lending saves the layers' new one.
        for (std::size_t i = first; i < first + count; ++i) { lent_profile_[i].clear(); }
    }
}

void Qwen4OffloadExperts::refill_lent(std::int32_t layers) {
    const auto first = static_cast<std::size_t>(cache_->lent_first());
    const auto count = static_cast<std::size_t>(layers);
    // The refill profile of each lent layer: T211 (keep_decode_residency) the residency it held
    // when lent, most recently used first; otherwise the last prompt window's routing, hottest
    // first.
    const std::span<const std::vector<std::int32_t>> lent(lent_profile_.data() + first, count);
    bool profiled = true;
    for (const auto& p : lent) { profiled = profiled && !p.empty(); }
    // A frozen cache (NINFER_QWEN4_EXPERT_FREEZE=all, validation) refills nothing either, so its
    // residency does not depend on how many elastic columns are active when the prompt ends.
    if (!profiled || cache_->frozen()) {
        cache_->reclaim_front(layers, {});
        return;
    }
    if (!options_.refill_order) {
        cache_->reclaim_front(layers, lent);
        return;
    }
    // T69: the refill goes hottest first across the reclaimed layers (rank of the expert in its
    // layer's profile), interleaved with the prompt's pending warm-start fills (their rank in the
    // targets), instead of layer by layer coldest first: the first decode windows find every
    // layer's hottest experts early.
    refill_rank_.assign(static_cast<std::size_t>(layers_) * kExperts, kExperts);
    if (options_.warm_start && !options_.keep_decode_residency &&
        prompt_rank_.size() == refill_rank_.size()) {
        refill_rank_ = prompt_rank_;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& profile = lent[i];
        std::int32_t* rank  = refill_rank_.data() + (first + i) * kExperts;
        std::fill(rank, rank + kExperts, kExperts);
        for (std::size_t r = 0; r < profile.size(); ++r) {
            rank[profile[r]] = static_cast<std::int32_t>(r);
        }
    }
    cache_->reclaim_front(layers, lent, refill_rank_);
}

std::int32_t Qwen4OffloadExperts::ring_slots_for(std::int32_t ring_slots, std::int32_t partitions) {
    if (partitions == 2) {
        // NINFER_QWEN4_TP2_HALF_RING=0 keeps the 1-GPU ring size (A/B runs; same bits).
        const char* v = std::getenv("NINFER_QWEN4_TP2_HALF_RING");
        if (v == nullptr || std::string(v) != "0") { return (ring_slots + 1) / 2; }
    }
    return ring_slots;
}

std::int32_t Qwen4OffloadExperts::ring_layers(std::int32_t ring_slots,
                                              std::span<const std::int32_t> layer_base,
                                              std::int32_t text_layers, std::int32_t partitions) {
    return layers_below(layer_base, text_layers, ring_slots_for(ring_slots, partitions));
}

std::int32_t Qwen4OffloadExperts::layers_below(std::span<const std::int32_t> layer_base,
                                               std::int32_t end, std::int64_t slots) {
    std::int64_t held = 0;
    std::int32_t n    = 0;
    while (held < slots) {
        // `end` <= 0 (a ring that takes every Text layer) leaves nothing below it.
        if (n >= end) { return std::max(end, 0) + 1; }
        ++n;
        held += layer_base[static_cast<std::size_t>(end - n)];
    }
    return n;
}

std::int64_t Qwen4OffloadExperts::gguf_ring_bytes(std::span<const std::int64_t> slot_bytes,
                                                  std::int32_t text_layers,
                                                  std::int32_t partitions) {
    std::int64_t largest = 0;
    for (std::int32_t l = 0; l < text_layers && l < static_cast<std::int32_t>(slot_bytes.size());
         ++l) {
        largest = std::max(largest, slot_bytes[static_cast<std::size_t>(l)]);
    }
    return static_cast<std::int64_t>(gguf_ring_slots(partitions)) * largest;
}

std::int32_t Qwen4OffloadExperts::gguf_layers_below(std::span<const std::int32_t> layer_base,
                                                    std::span<const std::int64_t> slot_bytes,
                                                    std::int32_t end, std::int64_t bytes) {
    std::int64_t held = 0;
    std::int32_t n    = 0;
    while (held < bytes) {
        if (n >= end) { return std::max(end, 0) + 1; }
        ++n;
        const auto l = static_cast<std::size_t>(end - n);
        held += static_cast<std::int64_t>(layer_base[l]) * slot_bytes[l];
    }
    return n;
}

bool Qwen4OffloadExperts::may_lend(std::span<const std::int32_t> layer_base,
                                   std::int32_t text_layers, std::int32_t layers,
                                   std::int32_t partitions) {
    if (layers < 0 || layers > text_layers) { return false; }
    std::int64_t lent  = 0;
    std::int64_t total = 0;
    for (std::int32_t l = 0; l < text_layers; ++l) {
        total += layer_base[static_cast<std::size_t>(l)];
        if (l >= text_layers - layers) { lent += layer_base[static_cast<std::size_t>(l)]; }
    }
    // One GPU: 0.9 (v0.7.2, a 8192-column prompt chunk next to the ring: 64k prefill +31 %, decode
    // and replay TTFT not worse); tp 2: 0.75 (16384 columns fit already).
    double fraction = partitions == 1 ? 0.9 : 0.75;
    if (const char* env = std::getenv("NINFER_QWEN4_PROMPT_LEND")) {
        const double v = std::atof(env);
        if (v > 0.0 && v <= 1.0) fraction = v;
    }
    return static_cast<double>(lent) <= fraction * static_cast<double>(total);
}

std::vector<std::int32_t> Qwen4OffloadExperts::layer_slot_profile(std::int32_t layers,
                                                                  std::int32_t text_layers,
                                                                  std::int32_t slots_per_layer,
                                                                  std::int32_t elastic_columns,
                                                                  std::int32_t partitions) {
    // The usage profile (models/qwen3_5/qwen4_layer_profile.h, prod/cache 17/10).
    const auto& kWeights = kQwen4LayerSlotWeights;
    std::vector<std::int32_t> base(static_cast<std::size_t>(std::max(layers, 0)), slots_per_layer);
    const char* env = std::getenv("NINFER_QWEN4_SLOT_PROFILE");
    const auto cap =
        static_cast<std::int64_t>(kExperts / std::max(partitions, 1)) - elastic_columns;
    if ((env != nullptr && std::string(env) == "0") || partitions != 2 || text_layers != 48 ||
        (layers != 48 && layers != 49) || slots_per_layer < 1 || cap < slots_per_layer) {
        return base;
    }
    // Scale the weights to layers * slots_per_layer, clamping at `cap` (water filling), then round
    // by largest remainder (ties to the lower layer) so the total is exact.
    const std::int64_t total = static_cast<std::int64_t>(layers) * slots_per_layer;
    std::vector<double> target(base.size(), 0.0);
    std::vector<bool> clamped(base.size(), false);
    for (bool changed = true; changed;) {
        changed          = false;
        double weights   = 0.0;
        double remaining = static_cast<double>(total);
        for (std::size_t l = 0; l < base.size(); ++l) {
            if (clamped[l]) {
                remaining -= static_cast<double>(cap);
            } else {
                weights += kWeights[l];
            }
        }
        for (std::size_t l = 0; l < base.size(); ++l) {
            if (clamped[l]) { continue; }
            target[l] = std::max(1.0, kWeights[l] * remaining / weights);
            if (target[l] > static_cast<double>(cap)) {
                target[l]  = static_cast<double>(cap);
                clamped[l] = true;
                changed    = true;
            }
        }
    }
    std::int64_t sum = 0;
    for (std::size_t l = 0; l < base.size(); ++l) {
        base[l] = static_cast<std::int32_t>(std::floor(target[l]));
        sum += base[l];
    }
    std::vector<std::size_t> order(base.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return target[a] - std::floor(target[a]) > target[b] - std::floor(target[b]);
    });
    for (std::size_t i = 0; sum < total && i < order.size(); ++i) {
        if (base[order[i]] < cap) {
            ++base[order[i]];
            ++sum;
        }
    }
    for (std::size_t i = 0; sum > total && i < order.size(); ++i) {
        const std::size_t l = order[order.size() - 1 - i];
        if (base[l] > 1) {
            --base[l];
            --sum;
        }
    }
    return base;
}

std::int32_t Qwen4OffloadExperts::workspace_layers_for(std::int32_t columns) const noexcept {
    if (!options_.scale_workspace || workspace_layers_ == 0) { return workspace_layers_; }
    for (const auto& [width, bytes] : options_.prompt_workspace_steps) {
        if (gguf_ && columns <= width) {
            return std::min(workspace_layers_,
                            gguf_layers_below(layer_base_, gguf_slot_bytes_, lend_first_, bytes));
        }
        if (columns <= width) {
            return std::min(workspace_layers_,
                            layers_below(layer_base_, lend_first_,
                                         (bytes + kLendPlaneBytes - 1) / kLendPlaneBytes));
        }
    }
    return workspace_layers_;
}

void Qwen4OffloadExperts::lend_prompt_layers(std::int32_t first) {
    if (cache_->lending() && cache_->lent_first() <= first) { return; }
    if (cache_->lending()) {
        // A wider chunk than the prompt's first: the lent layers are empty (their refill profile
        // comes from the last prompt window), so the lending restarts lower.
        cache_->reclaim_layers({});
    }
    if (options_.keep_decode_residency) {
        // T211: the layers lent earlier in this prompt keep the residency saved when they were
        // first lent; the newly lent ones save theirs now.
        save_lent_residency(first, text_layers_ - first);
    } else {
        for (auto& profile : lent_profile_) { profile.clear(); }
    }
    const auto events =
        cache_->lend_layers(first, text_layers_ - first, options_.lend_keep_elastic);
    for (std::size_t p = 0; p < events.size(); ++p) { lend_waited_[p] = false; }
}

void Qwen4OffloadExperts::save_lent_residency(std::int32_t first, std::int32_t count) {
    if (!options_.keep_decode_residency) { return; }
    prompt_lent_from_                             = std::min(prompt_lent_from_, first);
    const expert_cache::ExpertCachePolicy& policy = cache_->policy();
    for (std::int32_t l = first; l < first + count; ++l) {
        auto& profile = lent_profile_[static_cast<std::size_t>(l)];
        if (!profile.empty()) { continue; }
        // Most recently used first, the partitions interleaved (each GPU's refill queue then
        // ranks its experts by their own recency).
        std::array<std::vector<std::int32_t>, 2> lru;
        std::size_t longest = 0;
        for (std::int32_t p = 0; p < partitions_; ++p) {
            lru[static_cast<std::size_t>(p)] = policy.recency(l, p);
            longest = std::max(longest, lru[static_cast<std::size_t>(p)].size());
        }
        for (std::size_t r = 0; r < longest; ++r) {
            for (std::int32_t p = 0; p < partitions_; ++p) {
                const auto& v = lru[static_cast<std::size_t>(p)];
                if (r < v.size()) { profile.push_back(v[v.size() - 1 - r]); }
            }
        }
    }
}

void Qwen4OffloadExperts::prepare_prompt(std::int32_t columns, std::int64_t begin,
                                         std::int64_t prompt_tokens) {
    prompt_columns_ = columns;
    if (!text_prompt_seen_ || begin < prompt_next_begin_) {
        // T211b: the request's first text chunk (the first after a decode window, or one that
        // starts before the previous chunk ended: a request without a decode window came first)
        // decides how its prompt windows treat the cache. The update a request without a decode
        // window left pending is dropped.
        text_prompt_seen_ = true;
        prompt_new_       = options_.keep_decode_residency && prompt_tokens > 0 &&
                      static_cast<double>(begin) <
                          options_.keep_min_cached * static_cast<double>(prompt_tokens);
        pending_prompt_   = false;
        prompt_lent_from_ = std::numeric_limits<std::int32_t>::max();
        std::fill(pending_tokens_.begin(), pending_tokens_.end(), 0);
        for (auto& c : pending_counts_) { std::fill(c.begin(), c.end(), 0); }
    }
    prompt_next_begin_ = begin + columns;
}

void Qwen4OffloadExperts::apply_pending_prompt() {
    pending_prompt_ = false;
    for (std::int32_t l = 0; l < layers_; ++l) {
        auto& tokens = pending_tokens_[static_cast<std::size_t>(l)];
        auto& counts = pending_counts_[static_cast<std::size_t>(l)];
        if (tokens > 0 && (l >= text_layers_ || l < prompt_lent_from_)) {
            routed_.clear();
            for (std::int32_t e = 0; e < kExperts; ++e) {
                const std::int32_t c = counts[static_cast<std::size_t>(e)];
                if (c > 0) { routed_.push_back(expert_cache::Routed{e, c}); }
            }
            cache_->apply_prompt(l, routed_, tokens, options_.new_prompt_cap);
        }
        tokens = 0;
        std::fill(counts.begin(), counts.end(), 0);
    }
}

std::int32_t Qwen4OffloadExperts::prompt_arena_columns() const noexcept {
    return workspace_layers_ > 0 ? options_.base_chunk : std::numeric_limits<std::int32_t>::max();
}

std::array<DeviceSpan, 2>
Qwen4OffloadExperts::prompt_workspace(std::span<const cudaStream_t> streams) {
    if (!open_ || kind_ != Qwen4Window::Prompt || workspace_layers_ == 0) {
        throw std::logic_error("Qwen4Exp offload experts: no prompt workspace to lend");
    }
    const std::int32_t first = lend_first_ - chunk_workspace_layers_;
    if (chunk_workspace_layers_ == 0 || !cache_->lending() || cache_->lent_first() > first) {
        throw std::logic_error("Qwen4Exp offload experts: the prompt workspace is not lent "
                               "(prepare_prompt before begin_window)");
    }
    if (streams.size() != static_cast<std::size_t>(partitions_)) {
        throw std::invalid_argument("Qwen4Exp offload experts: one compute stream per GPU");
    }
    // The workspace: the base slots of layers [first, lend_first_) in each GPU's gate/up code
    // plane (planar pool: one contiguous region), the lent layers below the ring.
    std::int64_t begin = base_offset(first) * kLendPlaneBytes;
    std::int64_t end   = base_offset(lend_first_) * kLendPlaneBytes;
    if (gguf_) {
        // ggml block banks: the arena bytes of those base slots (base slots are layer-ordered).
        const auto at = [&](std::int32_t layer) {
            const auto slot = static_cast<std::size_t>(base_offset(layer));
            return slot < gguf_pool_.slot_offset.size() && layer < layers_
                       ? gguf_pool_.slot_offset[slot]
                       : gguf_pool_.base_bytes;
        };
        begin = at(first);
        end   = at(lend_first_);
    }
    const auto events = cache_->lend_events();
    std::array<DeviceSpan, 2> spans{};
    for (std::int32_t p = 0; p < partitions_; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        // The compute stream's first write waits for every fill issued before the lend.
        if (!lend_waited_[pi]) {
            const ScopedCurrentDevice scope(device_ids_[pi]);
            CUDA_CHECK(cudaStreamWaitEvent(streams[pi], events[pi], 0));
            lend_waited_[pi] = true;
        }
        std::byte* base = (p == 0 ? memory_.pool : peer_pool_)[0];
        spans[pi]       = DeviceSpan{base + begin, static_cast<std::size_t>(end - begin)};
    }
    return spans;
}

void Qwen4OffloadExperts::begin_window(cudaStream_t stream, Qwen4Window kind) {
    if (open_) { throw std::logic_error("Qwen4Exp offload experts: window already open"); }
    require_executor();
    if (split_live_[0] != nullptr && !split_calibrated_ && kind == Qwen4Window::Decode) {
        finish_split_calibration(stream);
    }
    if (kind == Qwen4Window::Prompt &&
        (!snapshot_any_window_ || snapshot_last_kind_ == Qwen4Window::Decode)) {
        // T59: a request begins (its first prompt window).
        if (!snapshot_blocks_.empty() && !cache_->lending()) {
            cache_->restore(snapshot_blocks_[static_cast<std::size_t>(
                snapshot_request_ % static_cast<std::int64_t>(snapshot_blocks_.size()))]);
        }
        ++snapshot_request_;
    }
    if (kind == Qwen4Window::Decode && snapshot_any_window_ &&
        snapshot_last_kind_ == Qwen4Window::Prompt && !options_.snapshot_save.empty()) {
        snapshot_save_block();
    }
    snapshot_any_window_ = true;
    snapshot_last_kind_  = kind;
    if (kind == Qwen4Window::Decode && after_prompt_ && options_.boost_windows > 0 &&
        options_.insert_cap > 0) {
        // T69: the decode after a prompt refills the cache with its own experts faster.
        boost_left_ = options_.boost_windows;
        cache_->set_insert_cap(options_.boost_cap * partitions_);
    } else if (kind == Qwen4Window::Decode && boost_left_ > 0 && --boost_left_ == 0) {
        cache_->set_insert_cap(options_.insert_cap * partitions_);
    } else if (kind == Qwen4Window::Prompt && boost_left_ > 0) {
        boost_left_ = 0;
        cache_->set_insert_cap(options_.insert_cap * partitions_);
    }
    after_prompt_ = kind == Qwen4Window::Prompt;
    if (kind == Qwen4Window::Decode) {
        text_prompt_seen_ = false;
        prompt_new_       = false;
    }
    if (kind == Qwen4Window::Decode && cache_->lending()) {
        // The prompt is over: the ring (and the borrowed workspace) goes back to the cache, its
        // layers refilled with the last prompt window's most routed experts.
        reclaim_lent(cache_->lent_count());
        // T98 (validation): under NINFER_QWEN4_EXPERT_FREEZE=1 the refill lands before the first
        // decode window, so the decode residency no longer depends on copy timing.
        if (cache_->frozen() && std::getenv("NINFER_QWEN4_FREEZE_FLUSH") != nullptr) {
            cache_->flush();
        }
    }
    if (kind == Qwen4Window::Decode && pending_prompt_) {
        // T211b: the new conversation's prompt updates the residency now that it is over (its
        // lent layers were just restored).
        apply_pending_prompt();
    }
    chunk_workspace_layers_ = 0;
    if (kind == Qwen4Window::Prompt && stream_ != nullptr && prompt_columns_ > 0) {
        // T42: a chunk wider than the Program's arena borrows the layers below the ring as its
        // workspace; a streamed chunk lends at least the ring. Lending now (between windows)
        // lets a later, wider chunk of the same prompt grow it. T69: the workspace is sized for
        // the chunk's width; `needed` = the first layer this chunk needs lent.
        const bool wide     = prompt_columns_ > prompt_arena_columns();
        const bool streamed = prompt_columns_ >= options_.stream_min_tokens &&
                              prompt_columns_ > ops::kMoeSlotsMaxTokens;
        if (wide) { chunk_workspace_layers_ = workspace_layers_for(prompt_columns_); }
        const std::int32_t needed = wide       ? lend_first_ - chunk_workspace_layers_
                                    : streamed ? lend_first_
                                               : text_layers_;
        if (cache_->lending() && options_.early_reclaim > 0 && cache_->lent_first() < needed) {
            // T69: layers lent for an earlier, wider chunk that this one does not need (e.g. the
            // short tail of a prompt) go back now, so their refill overlaps this chunk instead of
            // the first decode windows. Mode 2 waits for the refill first, so the chunk itself
            // finds the layers filled (published below).
            reclaim_lent(std::min(needed, text_layers_) - cache_->lent_first());
            if (options_.early_reclaim == 2) { cache_->flush(); }
        }
        if (wide) {
            lend_prompt_layers(needed);
        } else if (streamed && !cache_->lending()) {
            lend_prompt_layers(lend_first_);
        }
    }
    if (kind == Qwen4Window::Prompt && gguf_ && (workspace_layers_ > 0 || gguf_ring_slots_ > 0) &&
        prompt_columns_ > 0) {
        // T42 for ggml block banks: only the borrowed workspace (the zero-copy prefill has no
        // ring); the same early reclaim of layers a narrower chunk does not need.
        const bool wide = prompt_columns_ > prompt_arena_columns();
        if (wide) { chunk_workspace_layers_ = workspace_layers_for(prompt_columns_); }
        const std::int32_t needed = wide ? lend_first_ - chunk_workspace_layers_ : text_layers_;
        if (cache_->lending() && options_.early_reclaim > 0 && cache_->lent_first() < needed) {
            reclaim_lent(std::min(needed, text_layers_) - cache_->lent_first());
            if (options_.early_reclaim == 2) { cache_->flush(); }
        }
        if (wide) {
            lend_prompt_layers(needed);
        } else if (gguf_ring_slots_ > 0 && prompt_columns_ >= options_.gguf_stream_min_tokens &&
                   prompt_columns_ > ops::kMoeSlotsMaxTokens && !cache_->lending()) {
            lend_prompt_layers(lend_first_); // the ring alone
        }
    }
    prompt_columns_ = 0;
    cache_->begin_window(window_streams(stream));
    if (kind == Qwen4Window::Prompt && gguf_ && options_.gguf_stream_min_tokens > 0) {
        upload_gguf_host_rows(stream);
    }
    mailbox_->clear_route_flags();
    if (prefetch_) {
        prefetch_->begin_window(stream);
        prefetch_at_begin_ = prefetch_->counters();
    }
    if (trace_ != nullptr) {
        mailbox_->clear_trace();
        flag_seen_ns_.assign(static_cast<std::size_t>(layers_), 0);
    }
    // The later layers' routing logs restart with the window (a window may run the MTP layer
    // alone, without a Text forward: the prefix-reuse bridge); end_window observes [0, end).
    std::fill(extra_offset_.begin(), extra_offset_.end(), 0);
    for (std::int32_t l = 0; l < layers_; ++l) {
        served_at_begin_[static_cast<std::size_t>(l)] = mailbox_->served(l);
    }
    open_                 = true;
    kind_                 = kind;
    observed_             = 0;
    begin_ns_             = ops::host::HostMoeMailbox::now_ns();
    host_served_at_begin_ = mailbox_->host_experts_served();
    at_begin_             = cache_->counters();
    window_               = Qwen4OffloadWindowStats{};
    window_.kind          = kind;
}

void Qwen4OffloadExperts::observe_layer(std::int32_t layer) {
    const ops::host::HostMoeRouteLog log = mailbox_->route_log(layer);
    const std::int32_t tokens            = static_cast<std::int32_t>(*log.tokens);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (tokens < 1 || tokens > log_tokens_) {
        throw std::logic_error("Qwen4Exp offload experts: routing log of layer " +
                               std::to_string(layer) + " holds " + std::to_string(tokens) +
                               " tokens");
    }
    expert_cache::count_routed(
        std::span<const std::int32_t>(log.ids, static_cast<std::size_t>(tokens) * kTopK), kExperts,
        routed_);
    if (routing_ != nullptr) {
        const std::int32_t header[4] = {static_cast<std::int32_t>(routing_windows_),
                                        kind_ == Qwen4Window::Prompt ? 1 : 0, layer, tokens};
        std::fwrite(header, sizeof(header), 1, routing_);
        std::fwrite(log.ids, sizeof(std::int32_t), static_cast<std::size_t>(tokens) * kTopK,
                    routing_);
    }
    // Columns that route nothing (ids -1: the verify and MTP columns past a lane's valid count,
    // moe_route) are not tokens of the window for the policy.
    std::int32_t routed_tokens = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        routed_tokens += log.ids[static_cast<std::size_t>(t) * kTopK] >= 0 ? 1 : 0;
    }
    if (trace_ != nullptr && kind_ == Qwen4Window::Decode && layer < text_layers_) {
        // T41: marginal experts and misses of each verification column (routed columns first).
        std::array<bool, kExperts> seen{};
        std::int64_t distinct = 0;
        std::int64_t missing  = 0;
        const auto columns    = static_cast<std::int32_t>(column_layers_.size());
        for (std::int32_t t = 0; t < tokens && t < columns; ++t) {
            const std::int32_t* row = log.ids + static_cast<std::size_t>(t) * kTopK;
            if (row[0] < 0) { break; }
            for (std::int32_t k = 0; k < kTopK; ++k) {
                const std::int32_t id = row[k];
                if (id < 0 || id >= kExperts || seen[static_cast<std::size_t>(id)]) { continue; }
                seen[static_cast<std::size_t>(id)] = true;
                ++distinct;
                if (cache_->published_slot(layer, id) < 0) { ++missing; }
            }
            const auto c = static_cast<std::size_t>(t);
            ++column_layers_[c];
            column_distinct_[c] += distinct;
            column_missing_[c] += missing;
        }
    }
    if (kind_ == Qwen4Window::Prompt) {
        if (!options_.keep_decode_residency && cache_->lent(layer) &&
            layer < static_cast<std::int32_t>(lent_profile_.size())) {
            std::vector<expert_cache::Routed> hot(routed_.begin(), routed_.end());
            std::stable_sort(hot.begin(), hot.end(),
                             [](const auto& a, const auto& b) { return a.count > b.count; });
            auto& profile = lent_profile_[static_cast<std::size_t>(layer)];
            profile.clear();
            for (const auto& r : hot) { profile.push_back(r.expert); }
        }
        if (layer < static_cast<std::int32_t>(prompt_counts_.size())) {
            auto& counts = prompt_counts_[static_cast<std::size_t>(layer)];
            std::fill(counts.begin(), counts.end(), 0);
            for (const auto& r : routed_) { counts[static_cast<std::size_t>(r.expert)] = r.count; }
            prompt_counts_tokens_[static_cast<std::size_t>(layer)] = tokens;
        }
    }
    if (kind_ == Qwen4Window::Prompt) {
        if (options_.warm_start) {
            // Rank of each routed expert in the chunk's targets (count descending, ties to the
            // lower id, as decide_prompt): the fill priority of its insertion (T25).
            std::vector<expert_cache::Routed> order(routed_.begin(), routed_.end());
            std::stable_sort(order.begin(), order.end(),
                             [](const expert_cache::Routed& a, const expert_cache::Routed& b) {
                                 return a.count > b.count;
                             });
            std::int32_t* rank = prompt_rank_.data() + static_cast<std::size_t>(layer) * kExperts;
            std::fill(rank, rank + kExperts, kExperts);
            for (std::size_t r = 0; r < order.size(); ++r) {
                rank[order[r].expert] = static_cast<std::int32_t>(r);
            }
        }
        // T211: by default a prompt window inserts nothing (the decode after it routes much more
        // like the decode before it than like the prompt).
        // T211b: a new conversation's prompt (prompt_new_) is counted and applied when its
        // first decode window begins (apply_pending_prompt).
        if (prompt_new_ && layer < static_cast<std::int32_t>(pending_counts_.size())) {
            auto& counts = pending_counts_[static_cast<std::size_t>(layer)];
            for (const auto& r : routed_) { counts[static_cast<std::size_t>(r.expert)] += r.count; }
            pending_tokens_[static_cast<std::size_t>(layer)] += routed_tokens;
            pending_prompt_ = true;
        }
        (void)cache_->observe_prompt(layer, routed_, routed_tokens,
                                     !options_.keep_decode_residency);
    } else if (!options_.freeze_after_prompt) {
        (void)cache_->observe(layer, routed_, routed_tokens);
    }
    window_.tokens = std::max(window_.tokens, routed_tokens);
    ++window_.layers;
    observed_ = layer + 1;
}

void Qwen4OffloadExperts::observe_window(cudaStream_t stream) {
    if (!open_) { throw std::logic_error("Qwen4Exp offload experts: no open window"); }
    if (kind_ == Qwen4Window::Prompt && options_.warm_start) {
        observe_prompt_layers(stream);
        return;
    }
    if (kind_ == Qwen4Window::Prompt) {
        // A prefill chunk runs for seconds: keep the fill queue moving meanwhile (the policy acts
        // on the chunk's routing only once it completed), without competing with the host
        // executor's threads.
        CUDA_CHECK(cudaEventRecord(enqueued_, stream));
        for (;;) {
            cache_->poll();
            const cudaError_t status = cudaEventQuery(enqueued_);
            if (status == cudaSuccess) { return; }
            if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }
    // Every layer's routing is read while the GPU works on the following ones; the insertions of
    // the layers before it are copied as soon as a layer's routing is published (the GPU has left
    // those layers: their slots are no longer read by this window).
    CUDA_CHECK(cudaEventRecord(enqueued_, stream));
    // The Text layers only: a later layer (MTP) may publish again in this window, end_window
    // observes its whole log.
    for (std::int32_t layer = observed_; layer < text_layers_; ++layer) {
        const ops::host::HostMoeRouteLog log = mailbox_->route_log(layer);
        // T25: layer's prediction of layer+1 is published before layer's own routing (its kernel
        // runs before the router): copy the candidates as soon as they are there.
        bool predict = prefetch_ && prefetch_->predicts(layer);
        for (std::uint32_t spins = 1; *log.published == 0; ++spins) {
            if (predict && prefetch_->ready(layer)) {
                prefetch_->issue(layer, ops::host::HostMoeMailbox::now_ns());
                predict = false;
            }
            if ((spins & 1023U) == 0) {
                const cudaError_t status = cudaEventQuery(enqueued_);
                if (status == cudaSuccess) {
                    if (*log.published != 0) { break; }
                    return; // the execution ended without this layer: end_window finishes
                }
                if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
            }
            _mm_pause();
        }
        if (predict && prefetch_->ready(layer)) {
            prefetch_->issue(layer, ops::host::HostMoeMailbox::now_ns());
        }
        std::int64_t t0 = ops::host::HostMoeMailbox::now_ns();
        if (trace_ != nullptr) { flag_seen_ns_[static_cast<std::size_t>(layer)] = t0; }
        if (layer > 0) {
            if (options_.fills_after_host) {
                // The doorbell of this layer has rung: hold the copies until the host executor
                // has served it (or the execution ended, or the mailbox failed).
                const std::uint32_t before = served_at_begin_[static_cast<std::size_t>(layer)];
                for (std::uint32_t spins = 1; mailbox_->served(layer) == before; ++spins) {
                    if ((spins & 1023U) == 0) {
                        if (mailbox_->failed()) { break; }
                        const cudaError_t status = cudaEventQuery(enqueued_);
                        if (status == cudaSuccess) { break; }
                        if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
                    }
                    _mm_pause();
                }
                t0 = ops::host::HostMoeMailbox::now_ns(); // observe_us: policy work only
            }
            cache_->submit_fills_unfenced();
        }
        observe_layer(layer);
        window_.observe_us += 1e-3 * static_cast<double>(ops::host::HostMoeMailbox::now_ns() - t0);
    }
}

void Qwen4OffloadExperts::observe_prompt_layers(cudaStream_t stream) {
    // Warm start: each Text layer's prompt routing as soon as the chunk has published it; its
    // insertions are copied once the chunk has published a later layer (the GPU has left the
    // layer: its victims' slots are no longer read by this chunk), so the residency of the
    // prompt fills during the prefill. Between polls the fill queue keeps moving.
    CUDA_CHECK(cudaEventRecord(enqueued_, stream));
    for (;;) {
        cache_->poll();
        while (observed_ < text_layers_ && *mailbox_->route_log(observed_).published != 0) {
            const std::int64_t t0 = ops::host::HostMoeMailbox::now_ns();
            if (observed_ > 0) {
                cache_->submit_fills_unfenced();
                // T211: no prompt fills to rank; the queue holds the lent layers' restore.
                if (!options_.keep_decode_residency) { cache_->prioritize_queue(prompt_rank_); }
            }
            observe_layer(observed_);
            window_.observe_us +=
                1e-3 * static_cast<double>(ops::host::HostMoeMailbox::now_ns() - t0);
        }
        const cudaError_t status = cudaEventQuery(enqueued_);
        if (status == cudaSuccess) { return; } // end_window observes what is left
        if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void Qwen4OffloadExperts::end_window(cudaStream_t stream) {
    if (!open_) { throw std::logic_error("Qwen4Exp offload experts: no open window"); }
    open_ = false;
    close_stream();
    cache_->end_window(window_streams(stream));
    for (std::int32_t layer = observed_; layer < layers_; ++layer) {
        if (*mailbox_->route_log(layer).published != 0) { observe_layer(layer); }
    }
    cache_->submit_fills();
    if (options_.warm_start && kind_ == Qwen4Window::Prompt && !options_.keep_decode_residency) {
        cache_->prioritize_queue(prompt_rank_);
    }
    cache_->poll();
    if (prefetch_) {
        prefetch_->end_window();
        const auto& p                = prefetch_->counters();
        window_.prefetch_issued      = p.issued - prefetch_at_begin_.issued;
        window_.prefetch_landed      = p.landed - prefetch_at_begin_.landed;
        window_.prefetch_used        = p.used - prefetch_at_begin_.used;
        window_.prefetch_used_routes = p.used_routes - prefetch_at_begin_.used_routes;
    }
    if (options_.freeze_after_prompt && kind_ == Qwen4Window::Prompt) {
        // Validation: the prompt's residency lands before the next window begins (which
        // publishes it), whatever the copy timing.
        while (cache_->fills_pending() > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            cache_->poll();
        }
    }
    window_.fill_wait_us = 0;
    if (options_.sync_fills) {
        // T59: the window's fills land before the next window begins (deterministic residency).
        const std::int64_t t0 = ops::host::HostMoeMailbox::now_ns();
        while (cache_->fills_pending() > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
            cache_->poll();
        }
        window_.fill_wait_us = 1e-3 * static_cast<double>(ops::host::HostMoeMailbox::now_ns() - t0);
    }

    const auto& c    = cache_->counters();
    window_.wall_us  = 1e-3 * static_cast<double>(ops::host::HostMoeMailbox::now_ns() - begin_ns_);
    window_.accesses = c.accesses - at_begin_.accesses;
    window_.device_hits   = c.device_hits - at_begin_.device_hits;
    window_.policy_hits   = c.policy_hits - at_begin_.policy_hits;
    window_.missing       = c.missing - at_begin_.missing;
    window_.inserts       = c.inserts - at_begin_.inserts;
    window_.fills_issued  = c.fills_issued - at_begin_.fills_issued;
    window_.fill_bytes    = c.fill_bytes - at_begin_.fill_bytes;
    window_.fills_pending = cache_->fills_pending();
    // The service thread finished every doorbell of this window before the device completed it.
    window_.host_experts = mailbox_->host_experts_served() - host_served_at_begin_;
    if (kind_ == Qwen4Window::Decode) {
        split_totals_.cpu_experts += window_.host_experts;
        if (splitting()) {
            for (std::int32_t layer = 0; layer < layers_; ++layer) {
                window_.gpu_read_experts += mailbox_->device_share_experts(layer);
                window_.gpu_read_routes += mailbox_->device_share_routes(layer);
            }
        }
        ++split_totals_.decode_windows;
        split_totals_.accesses += window_.accesses;
        split_totals_.device_hits += window_.device_hits;
        split_totals_.missing += window_.missing;
        split_totals_.gpu_experts += window_.gpu_read_experts;
        split_totals_.gpu_routes += window_.gpu_read_routes;
    }
    ++routing_windows_;
    if (trace_ != nullptr) {
        finish_trace();
        // The device timer drifts against the host clock (tens of ppm): re-tie them after every
        // window while the device is idle (trace only; ~50 us).
        clock_offset_ = mailbox_->calibrate_clock(stream, 4);
    }
    // The window is closed; its outputs are wrong if the executor failed during it.
    require_executor();
}

void Qwen4OffloadExperts::abandon_window(cudaStream_t stream) noexcept {
    if (!open_) { return; }
    try {
        end_window(stream);
    } catch (...) {
        close_stream();
        open_ = false;
    }
}

void Qwen4OffloadExperts::run_peer_share(std::int32_t layer, const Tensor& x, const Tensor& ids,
                                         const Tensor& weights, Tensor& partial,
                                         WorkspaceArena& work, cudaStream_t stream) {
    if (!split_on_peer_ || layer < 0 || layer >= layers_) {
        throw std::logic_error("Qwen4Exp offload experts: no rank 1 miss-split share");
    }
    const std::int32_t T = x.ne[1];
    // The same rule on the same routing and residency as rank 0's publish (both mirrors take the
    // same updates at every begin_window), so rank 1 computes exactly the experts the host skips.
    ops::host::select_device_share(stream, T, kTopK, static_cast<const std::int32_t*>(ids.data),
                                   peer_.mirror + static_cast<std::size_t>(layer) * kExperts,
                                   kExperts, options_.split,
                                   ops::host::HostMoeDeviceShare{.reads       = peer_.reads,
                                                                 .picks       = peer_.picks,
                                                                 .stage_table = peer_.stage_table,
                                                                 .live        = split_live_[1]});
    const Qwen4ExpertBank& bank = banks_[static_cast<std::size_t>(layer)];
    const ops::MoeSlotPool& b   = bank.bank;
    ops::host::stage_device_share(stream, peer_.picks,
                                  {static_cast<const std::byte*>(b.gate_up_codes.base),
                                   static_cast<const std::byte*>(b.gate_up_scales.base),
                                   static_cast<const std::byte*>(b.down_codes.base),
                                   static_cast<const std::byte*>(b.down_scales.base)},
                                  peer_.staging,
                                  {ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
                                   ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes});
    const float* divisors = peer_divisors_ + static_cast<std::size_t>(layer) * kDivisorsPerLayer;
    const Tensor table(peer_.stage_table, DType::I32, {kExperts});
    const ops::MoeSlotsLayer view = slots_layer(bank, divisors, divisors + kGateUpDivisors, table);
    const Tensor reads(peer_.reads, DType::I32, {kTopK, T});
    const ops::MoeSlotPool pool{
        .gate_up_codes  = {peer_.staging[0], ops::kMoeSlotsGateUpCodeBytes},
        .gate_up_scales = {peer_.staging[1], ops::kMoeSlotsGateUpScaleBytes},
        .down_codes     = {peer_.staging[2], ops::kMoeSlotsDownCodeBytes},
        .down_scales    = {peer_.staging[3], ops::kMoeSlotsDownScaleBytes},
        .slots          = ops::host::HostMoeSplit::kMaxReads,
    };
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(ops::moe_slots_nvfp4_workspace_bytes(T)));
    ops::moe_slots_nvfp4(x, reads, weights, view, pool, ops::MoeSlotsOutput::Accumulate, partial,
                         scratch, stream);
}

// ---- Qwen4OffloadPeerExperts ------------------------------------------------------------------

void Qwen4OffloadPeerExperts::run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                                  const Tensor& ids, const Tensor& weights, Tensor& partial,
                                  WorkspaceArena& work, cudaStream_t stream) {
    require_operands(x, ids, weights, partial);
    if (layer < 0 || layer >= layers_) {
        throw std::invalid_argument("Qwen4Exp offload experts (rank 1): layer out of range");
    }
    const Tensor table(memory_.table + static_cast<std::size_t>(layer) * kExperts, DType::I32,
                       {kExperts});
    if (bank.gguf) {
        // ggml block banks: rank 1's resident experts of its partition, plus (zero-copy prefill
        // chunks) the rest of its partition read in place from the host bank.
        ops::MoeGgufLayer view = gguf_view(table, bank, layers_ * slots_per_layer_,
                                           memory_.slot_offsets, memory_.pool[0]);
        if (owner_ != nullptr && owner_->gguf_streams(layer, x.ne[1])) {
            const std::int32_t* row =
                memory_.stream_tables + static_cast<std::size_t>(layer) * kExperts;
            if (owner_->gguf_ring_copy(1, layer, stream)) {
                view.slot_table = Tensor(const_cast<std::int32_t*>(row), DType::I32, {kExperts});
                view.slots      = owner_->gguf_stream_slots();
                gguf_partial(x, ids, weights, view, partial, work, stream);
                owner_->gguf_ring_done(1, layer, stream);
                return;
            }
            view.host_table = row;
        }
        gguf_partial(x, ids, weights, view, partial, work, stream);
        return;
    }
    const float* divisors =
        memory_.divisors + static_cast<std::size_t>(layer) * Qwen4OffloadExperts::kDivisorsPerLayer;
    const ops::MoeSlotsLayer view = slots_layer(bank, divisors, divisors + kGateUpDivisors, table);
    const std::int32_t T          = x.ne[1];
    if (owner_ != nullptr && owner_->streams_call(layer, T)) {
        // A wide prefill call: rank 1 streams its partition into its own ring (T42).
        owner_->run_peer_streamed(layer, bank, x, ids, weights, partial, work, stream);
        return;
    }
    const ops::MoeSlotPool pool =
        slot_pool(memory_.pool, planar_strides(), layers_ * slots_per_layer_);
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(
        ops::moe_slots_nvfp4_workspace_bytes(std::min<std::int32_t>(T, ops::kMoeSlotsMaxTokens))));
    for (std::int32_t begin = 0; begin < T; begin += ops::kMoeSlotsMaxTokens) {
        const std::int32_t count = std::min(ops::kMoeSlotsMaxTokens, T - begin);
        Tensor out               = partial.slice(1, begin, count);
        ops::moe_slots_nvfp4(x.slice(1, begin, count), ids.slice(1, begin, count),
                             weights.slice(1, begin, count), view, pool, ops::MoeSlotsOutput::Store,
                             out, scratch, stream);
    }
    // A decode call's miss-split share computed here instead of rank 0 (T42 rank balance).
    if (owner_ != nullptr && owner_->peer_shares() && T <= ops::kMoeSlotsMaxTokens) {
        owner_->run_peer_share(layer, x, ids, weights, partial, work, stream);
    }
}

void Qwen4OffloadExperts::finish_trace() {
    const auto records = mailbox_->trace();
    std::vector<std::int64_t> seen(static_cast<std::size_t>(layers_), 0);
    for (const auto& r : records) {
        window_.cpu_us += 1e-3 * static_cast<double>(r.done_ns - r.seen_ns);
        window_.cpu_experts += r.experts;
        if (r.layer >= 0 && r.layer < layers_ && seen[static_cast<std::size_t>(r.layer)] == 0) {
            seen[static_cast<std::size_t>(r.layer)] = r.seen_ns;
        }
    }
    if (kind_ == Qwen4Window::Decode) {
        for (std::int32_t l = 0; l < layers_; ++l) {
            const volatile std::uint64_t* s = mailbox_->gpu_stamps(l);
            const auto pub = static_cast<std::int64_t>(s[ops::host::kStampPublish]);
            const auto dev = static_cast<std::int64_t>(s[ops::host::kStampDevice]);
            const auto acc = static_cast<std::int64_t>(s[ops::host::kStampAccumulate]);
            if (pub == 0 || dev < pub || acc < dev) { continue; }
            window_.gpu_resident_us += 1e-3 * static_cast<double>(dev - pub);
            window_.gpu_wait_us += 1e-3 * static_cast<double>(acc - dev);
            if (seen[static_cast<std::size_t>(l)] != 0) {
                window_.doorbell_us +=
                    1e-3 *
                    static_cast<double>(seen[static_cast<std::size_t>(l)] - (pub + clock_offset_));
                if (flag_seen_ns_[static_cast<std::size_t>(l)] != 0) {
                    window_.bell_after_flag_us +=
                        1e-3 * static_cast<double>(seen[static_cast<std::size_t>(l)] -
                                                   flag_seen_ns_[static_cast<std::size_t>(l)]);
                }
            }
        }
    }
    std::fprintf(trace_, "{\"window\":%" PRId64 ",\"kind\":\"%s\",", windows_++,
                 window_name(window_.kind));
    write_stats(trace_, window_, 1.0);
    std::fprintf(trace_, "}\n");
    if (kind_ == Qwen4Window::Decode) {
        add(decode_sum_, window_);
        ++decode_windows_;
    }
    if ((windows_ & 63) == 0) { std::fflush(trace_); }
}

} // namespace ninfer::models::qwen3_5::execution
