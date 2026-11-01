// Implements: include/ninfer/ops/tp2/hyper_connection.h (stream-split hyper-connection mix).
//
// Composition of the general route of hyper_connection_mix (elementwise.cu + Q8 Linear) over one
// rank's two streams, with the two cross-rank sums of allreduce_sum_f32: the projection p (rank
// partials of disjoint stream columns) and the collapsed block input (rank partials of disjoint
// streams). The element kernels below differ from elementwise.cu's only where the split needs it:
// p arrives in FP32 after the exchange, a rank keeps the inject gates of its own streams, and the
// collapse writes an FP32 partial instead of the rounded BF16 sum.
#include "ninfer/ops/tp2/hyper_connection.h"

#include "core/device_scope.h"
#include "ninfer/ops/linear.h"
#include "ops/common/math.cuh"
#include "ops/hyper_connection/launch.h"
#include "ops/hyper_connection/split_launch.h"
#include "ninfer/ops/peer_mailbox.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::kHcHidden;
using detail::kHcRank;
using detail::kHcStreams;

constexpr std::int32_t kLocalStreams = kHcStreams / 2;
constexpr std::int32_t kLocalWide    = kLocalStreams * kHcHidden;
constexpr int kBlock                 = 256;

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 65535LL * 8));
}

__global__ void hc_split_widen_kernel(const __nv_bfloat16* __restrict__ in, float* __restrict__ out,
                                      std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = __bfloat162float(in[i]);
    }
}

// p [rows, T] FP32 (the cross-rank sum) -> low [R, T] BF16 = silu(p/S); the rank's inject gates
// [2, T] FP32 = 2 sigmoid(p[R + first + s] / S).
__global__ void hc_split_gates_kernel(const float* __restrict__ p, __nv_bfloat16* __restrict__ low,
                                      float* __restrict__ inject, std::int32_t rows,
                                      std::int32_t first_stream, std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * columns;
    const float inv_streams  = 1.0F / static_cast<float>(kHcStreams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t row    = static_cast<std::int32_t>(i % rows);
        const std::int64_t column = i / rows;
        const float v             = p[i] * inv_streams;
        if (row < kHcRank) {
            low[column * kHcRank + row] = __float2bfloat16_rn(silu(v));
            continue;
        }
        const std::int32_t local = row - kHcRank - first_stream;
        if (inject != nullptr && local >= 0 && local < kLocalStreams) {
            inject[column * kLocalStreams + local] = 2.0F * sigmoid(v);
        }
    }
}

// up [2H, T] BF16 logits, normalized [2H, T] BF16 -> partial [H, T] FP32 =
// (1/S) sum_{s local} sigmoid(up) * n.
__global__ void hc_split_collapse_kernel(const __nv_bfloat16* __restrict__ up,
                                         const __nv_bfloat16* __restrict__ normalized,
                                         float* __restrict__ partial, std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(kHcHidden) * columns;
    const float inv_streams  = 1.0F / static_cast<float>(kHcStreams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t h      = static_cast<std::int32_t>(i % kHcHidden);
        const std::int64_t column = i / kHcHidden;
        const std::int64_t base   = column * kLocalWide + h;
        float sum                 = 0.0F;
        for (int s = 0; s < kLocalStreams; ++s) {
            const std::int64_t index = base + static_cast<std::int64_t>(s) * kHcHidden;
            sum += sigmoid(__bfloat162float(up[index])) * __bfloat162float(normalized[index]);
        }
        partial[i] = sum * inv_streams;
    }
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(std::string("hyper_connection_mix_split: ") + message);
    }
}

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.ne[0] == rows &&
                t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1,
            label);
}

void require_q8(const Weight& w, std::int32_t n, std::int32_t k, const char* label) {
    require(w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit &&
                w.scale_dtype == DType::FP16 && w.qdata != nullptr && w.scales != nullptr &&
                w.n == n && w.k == k && w.padded_shape[1] == (k + 127) / 128 * 128,
            label);
}

// Whether the bytes of two tensors intersect.
bool overlaps(const Tensor& a, const Tensor& b) {
    const auto begin_a = reinterpret_cast<std::uintptr_t>(a.data);
    const auto begin_b = reinterpret_cast<std::uintptr_t>(b.data);
    return begin_a < begin_b + b.bytes() && begin_b < begin_a + a.bytes();
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::size_t workspace_bytes(std::int32_t tokens, bool inject) {
    const auto t           = static_cast<std::size_t>(tokens);
    const std::size_t rows = kHcRank + (inject ? kHcStreams : 0);
    return align256(t * kLocalWide * 2)       // normalized
           + align256(t * rows * 2)           // projection partial (BF16)
           + 2 * align256(t * rows * 4)       // p FP32 + staging
           + align256(t * kHcRank * 2)        // low
           + align256(t * kLocalWide * 2)     // up logits
           + 2 * align256(t * kHcHidden * 4); // collapse partial FP32 + staging
}

struct RankScratch {
    Tensor normalized, projection, p, p_staging, low, up, partial, partial_staging;
};

// The fused route needs its kernels loaded and its grid known on both devices before a capture
// (a module cannot be loaded inside the capture that first launches it): every eager call prepares
// them, and a captured call takes the fused route only on devices an eager call prepared.
constexpr int kMaxDevices    = 16;
bool g_prepared[kMaxDevices] = {};

bool prepared(int device) { return device >= 0 && device < kMaxDevices && g_prepared[device]; }

// Prepares the devices of `ec` unless one of its streams is capturing (an occupancy query may load
// a module, which a capture rejects).
void prepare(const ExecutionContext& ec);

void prepare(int device) {
    if (device < 0 || device >= kMaxDevices || g_prepared[device]) { return; }
    ScopedCurrentDevice::select(device);
    (void)detail::hyper_connection_split_fused_grid(); // loads every T's kernel, occupancy query
    g_prepared[device] = true;
}

void prepare(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        ScopedCurrentDevice::select(ec.dev[rank]->device);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusNone) { return; }
    }
    for (int rank = 0; rank < 2; ++rank) { prepare(ec.dev[rank]->device); }
}

// NINFER_HC_SPLIT_ROUTE=composed keeps the composed route inside captures (A/B, tests).
bool fused_disabled() {
    const char* value = std::getenv("NINFER_HC_SPLIT_ROUTE");
    return value != nullptr && std::string(value) == "composed";
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 0xFU) == 0; }

struct Validated {
    std::int32_t tokens = 0;
    std::int32_t rows   = 0;
    bool inject         = false;
};

Validated validate(const std::array<Tensor, 2>& residual,
                   const std::array<HyperConnectionWeights, 2>& weights, float eps,
                   const std::array<Tensor, 2>& mixed, const std::array<Tensor*, 2>& inject,
                   const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value() &&
                ec.dev[0]->device != ec.dev[1]->device,
            "requires an ExecutionContext with two distinct devices");
    require(eps > 0.0F && std::isfinite(eps), "eps must be positive and finite");
    Validated v;
    v.tokens = residual[0].ne[1];
    require(v.tokens >= 1, "T must be positive");
    v.inject = inject[0] != nullptr;
    require((inject[1] != nullptr) == v.inject, "both ranks pass inject or neither");
    v.rows = kHcRank + (v.inject ? kHcStreams : 0);
    for (int rank = 0; rank < 2; ++rank) {
        require_matrix(residual[rank], DType::BF16, kLocalWide, v.tokens,
                       "residual must be BF16 [2*2560, T] on both ranks");
        require_matrix(mixed[rank], DType::BF16, kHcHidden, v.tokens,
                       "mixed must be BF16 [2560, T]");
        require(aligned16(mixed[rank].data), "mixed must be 16-byte aligned");
        const auto& w = weights[static_cast<std::size_t>(rank)];
        require(w.norm.dtype == DType::BF16 && w.norm.data != nullptr && w.norm.is_contiguous() &&
                    w.norm.numel() == kLocalWide,
                "norm must be contiguous BF16 [2*2560]");
        require_q8(w.down, v.rows, kLocalWide, "down must be Q8 row-split [R+I, 2*2560]");
        require_q8(w.up, kLocalWide, kHcRank, "up must be Q8 row-split [2*2560, 320]");
        if (v.inject) {
            require_matrix(*inject[rank], DType::FP32, kLocalStreams, v.tokens,
                           "inject must be FP32 [2, T]");
        }
        require(workspace[rank] != nullptr &&
                    workspace[rank]->capacity() - workspace[rank]->used() >=
                        std::max(workspace_bytes(v.tokens, v.inject),
                                 detail::hyper_connection_split_fused_workspace_bytes(v.tokens)),
                "workspace too small");
        require(!overlaps(mixed[rank], residual[rank]), "mixed must not overlap residual");
    }
    return v;
}

// The fused route's mailbox, or null for the composed route.
PeerMailbox* fused_mailbox(const Validated& v, const std::array<HyperConnectionWeights, 2>& weights,
                           const ExecutionContext& ec, const PeerEvents& events) {
    if (v.tokens > detail::kHcFusedMaxT || fused_disabled()) { return nullptr; }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& w = weights[static_cast<std::size_t>(rank)];
        if (!prepared(ec.dev[rank]->device) || !aligned16(w.down.qdata) || !aligned16(w.up.qdata)) {
            return nullptr;
        }
    }
    const std::size_t widest = static_cast<std::size_t>(v.tokens) * kHcHidden * sizeof(float);
    PeerMailbox* mailbox     = detail::captured_mailbox_for(widest, ec, events);
    if (mailbox == nullptr || mailbox->kernel() != PeerExchangeKernel::Pipelined) {
        return nullptr;
    }
    for (int rank = 0; rank < 2; ++rank) {
        ScopedCurrentDevice::select(ec.dev[rank]->device);
        if (detail::hyper_connection_split_fused_grid() == 0) { return nullptr; }
    }
    return mailbox;
}

void run_fused(PeerMailbox& mailbox, const Validated& v, const std::array<Tensor, 2>& residual,
               const std::array<Tensor, 2>* previous_y,
               const std::array<Tensor, 2>* previous_inject,
               const std::array<HyperConnectionWeights, 2>& weights, float eps,
               const std::array<Tensor, 2>& mixed, const std::array<Tensor*, 2>& inject,
               const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    const std::size_t bytes1 = static_cast<std::size_t>(v.tokens) * v.rows * sizeof(float);
    const std::size_t bytes2 = static_cast<std::size_t>(v.tokens) * kHcHidden * sizeof(float);
    const int slot1          = mailbox.take_capture_slot(bytes1);
    const int slot2          = mailbox.take_capture_slot(bytes2);
    const auto view          = [](const PeerExchangeSlot& s) {
        return detail::HcSplitExchange{s.mine_payload, s.peer_payload, s.mine_flags, s.peer_flags,
                                       s.epochs};
    };
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        ScopedCurrentDevice::select(local.device);
        auto& arena      = *workspace[rank];
        auto scope       = arena.scope();
        const auto t     = v.tokens;
        Tensor partials  = arena.alloc(DType::FP32, {detail::kHcRowsCap * kLocalStreams, t});
        Tensor squares   = arena.alloc(DType::FP32, {kLocalStreams, t});
        Tensor projected = arena.alloc(DType::FP32, {detail::kHcRowsCap, t});
        Tensor collapse  = arena.alloc(DType::FP32, {kHcHidden, t});
        const auto& w    = weights[static_cast<std::size_t>(rank)];
        const PeerExchangeSlot s1 = mailbox.exchange_slot(rank, slot1, bytes1);
        const PeerExchangeSlot s2 = mailbox.exchange_slot(rank, slot2, bytes2);
        detail::HcSplitFused params;
        params.residual = static_cast<__nv_bfloat16*>(residual[rank].data);
        if (previous_y != nullptr) {
            params.previous_y      = static_cast<const __nv_bfloat16*>((*previous_y)[rank].data);
            params.previous_inject = static_cast<const float*>((*previous_inject)[rank].data);
        }
        params.norm         = static_cast<const __nv_bfloat16*>(w.norm.data);
        params.down_codes   = static_cast<const std::int8_t*>(w.down.qdata);
        params.down_scales  = static_cast<const __half*>(w.down.scales);
        params.up_codes     = static_cast<const std::int8_t*>(w.up.qdata);
        params.up_scales    = static_cast<const __half*>(w.up.scales);
        params.mixed        = static_cast<__nv_bfloat16*>(mixed[rank].data);
        params.inject       = v.inject ? static_cast<float*>(inject[rank]->data) : nullptr;
        params.partials     = static_cast<float*>(partials.data);
        params.square_sums  = static_cast<float*>(squares.data);
        params.projection   = static_cast<float*>(projected.data);
        params.collapse     = static_cast<float*>(collapse.data);
        params.exchange1    = view(s1);
        params.exchange2    = view(s2);
        params.hang         = s1.hang;
        params.rows         = v.rows;
        params.tokens       = t;
        params.first_stream = rank * kLocalStreams;
        params.eps          = eps;
        detail::hyper_connection_split_fused_launch(params, local.stream);
    }
}

void run_composed(const Validated& v, const std::array<Tensor, 2>& residual,
                  const std::array<HyperConnectionWeights, 2>& weights, float eps,
                  const std::array<Tensor, 2>& mixed, const std::array<Tensor*, 2>& inject,
                  const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec,
                  const PeerEvents& events) {
    const std::int32_t tokens = v.tokens;
    const std::int32_t rows   = v.rows;
    std::array<RankScratch, 2> scratch;
    std::array<std::optional<DeviceArena::Scope>, 2> scopes;
    for (int rank = 0; rank < 2; ++rank) {
        auto& arena = *workspace[rank];
        scopes[rank].emplace(arena.scope());
        auto& s           = scratch[rank];
        s.normalized      = arena.alloc(DType::BF16, {kLocalWide, tokens});
        s.projection      = arena.alloc(DType::BF16, {rows, tokens});
        s.p               = arena.alloc(DType::FP32, {rows, tokens});
        s.p_staging       = arena.alloc(DType::FP32, {rows, tokens});
        s.low             = arena.alloc(DType::BF16, {kHcRank, tokens});
        s.up              = arena.alloc(DType::BF16, {kLocalWide, tokens});
        s.partial         = arena.alloc(DType::FP32, {kHcHidden, tokens});
        s.partial_staging = arena.alloc(DType::FP32, {kHcHidden, tokens});
    }
    // Phase 1, per rank: normalize its streams, project them onto the low rank (partial p).
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        ScopedCurrentDevice::select(local.device);
        auto& s       = scratch[rank];
        const auto& w = weights[static_cast<std::size_t>(rank)];
        detail::grouped_offset_rmsnorm_launch(residual[rank], w.norm, kLocalStreams, eps,
                                              s.normalized, local.stream);
        linear(s.normalized, w.down, s.projection, local.stream);
        hc_split_widen_kernel<<<grid_for(s.p.numel()), kBlock, 0, local.stream>>>(
            static_cast<const __nv_bfloat16*>(s.projection.data), static_cast<float*>(s.p.data),
            s.p.numel());
        CUDA_CHECK(cudaGetLastError());
    }
    // Exchange 1: p = p_0 + p_1 (FP32) on both ranks.
    allreduce_sum_f32({scratch[0].p, scratch[1].p}, {scratch[0].p_staging, scratch[1].p_staging},
                      nullptr, ec, events);
    // Phase 2, per rank: gates, Up rows of its streams, FP32 collapse partial.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        ScopedCurrentDevice::select(local.device);
        auto& s       = scratch[rank];
        const auto& w = weights[static_cast<std::size_t>(rank)];
        hc_split_gates_kernel<<<grid_for(s.p.numel()), kBlock, 0, local.stream>>>(
            static_cast<const float*>(s.p.data), static_cast<__nv_bfloat16*>(s.low.data),
            v.inject ? static_cast<float*>(inject[rank]->data) : nullptr, rows,
            rank * kLocalStreams, tokens);
        CUDA_CHECK(cudaGetLastError());
        linear(s.low, w.up, s.up, local.stream);
        hc_split_collapse_kernel<<<grid_for(s.partial.numel()), kBlock, 0, local.stream>>>(
            static_cast<const __nv_bfloat16*>(s.up.data),
            static_cast<const __nv_bfloat16*>(s.normalized.data),
            static_cast<float*>(s.partial.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
    // Exchange 2: mixed = RNE(q_0 + q_1).
    const std::array<Tensor, 2> out{mixed[0], mixed[1]};
    allreduce_sum_f32({scratch[0].partial, scratch[1].partial},
                      {scratch[0].partial_staging, scratch[1].partial_staging}, &out, ec, events);
}

} // namespace

std::size_t hyper_connection_mix_split_workspace_bytes(std::int32_t min_tokens,
                                                       std::int32_t max_tokens, bool inject) {
    require(min_tokens >= 1 && min_tokens <= max_tokens, "empty or non-positive token interval");
    return std::max(workspace_bytes(max_tokens, inject),
                    detail::hyper_connection_split_fused_workspace_bytes(max_tokens));
}

void hyper_connection_mix_split(const std::array<Tensor, 2>& residual,
                                const std::array<HyperConnectionWeights, 2>& weights, float eps,
                                const std::array<Tensor, 2>& mixed,
                                const std::array<Tensor*, 2>& inject,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& ec, const PeerEvents& events) {
    const Validated v = validate(residual, weights, eps, mixed, inject, workspace, ec);
    const ScopedCurrentDevice guard;
    if (PeerMailbox* mailbox = fused_mailbox(v, weights, ec, events)) {
        run_fused(*mailbox, v, residual, nullptr, nullptr, weights, eps, mixed, inject, workspace,
                  ec);
        return;
    }
    run_composed(v, residual, weights, eps, mixed, inject, workspace, ec, events);
    prepare(ec);
}

void hyper_connection_combine_mix_split(const std::array<Tensor, 2>& residual,
                                        const std::array<Tensor, 2>& previous_y,
                                        const std::array<Tensor, 2>& previous_inject,
                                        const std::array<HyperConnectionWeights, 2>& weights,
                                        float eps, const std::array<Tensor, 2>& mixed,
                                        const std::array<Tensor*, 2>& inject,
                                        const std::array<WorkspaceArena*, 2>& workspace,
                                        const ExecutionContext& ec, const PeerEvents& events) {
    const Validated v = validate(residual, weights, eps, mixed, inject, workspace, ec);
    for (int rank = 0; rank < 2; ++rank) {
        require_matrix(previous_y[rank], DType::BF16, kHcHidden, v.tokens,
                       "previous_y must be BF16 [2560, T]");
        require_matrix(previous_inject[rank], DType::FP32, kLocalStreams, v.tokens,
                       "previous_inject must be FP32 [2, T] (the rank's streams)");
        require(!overlaps(previous_y[rank], residual[rank]) &&
                    !overlaps(previous_inject[rank], residual[rank]),
                "previous_y and previous_inject must not overlap the residual");
        require(inject[rank] == nullptr || !overlaps(*inject[rank], previous_inject[rank]),
                "inject must not overlap previous_inject");
    }
    const ScopedCurrentDevice guard;
    if (PeerMailbox* mailbox = fused_mailbox(v, weights, ec, events)) {
        run_fused(*mailbox, v, residual, &previous_y, &previous_inject, weights, eps, mixed, inject,
                  workspace, ec);
        return;
    }
    for (int rank = 0; rank < 2; ++rank) {
        ScopedCurrentDevice::select(ec.dev[rank]->device);
        Tensor r = residual[rank];
        detail::hyper_connection_combine_launch(previous_y[rank], previous_inject[rank], r,
                                                ec.dev[rank]->stream);
    }
    run_composed(v, residual, weights, eps, mixed, inject, workspace, ec, events);
    prepare(ec);
}

} // namespace ninfer::ops
