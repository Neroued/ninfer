// ninfer::ops - hyper-connection wrappers: validate the public contract, then dispatch.
#include "ninfer/ops/hyper_connection.h"

#include "ninfer/ops/linear.h"
#include "ops/hyper_connection/launch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops {
namespace {

using detail::kHcFusedMaxT;
using detail::kHcHidden;
using detail::kHcRank;
using detail::kHcStreams;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_matrix(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype, op, (std::string(label) + " dtype mismatch").c_str());
    require(t.data != nullptr && t.is_contiguous(), op,
            (std::string(label) + " must be contiguous and non-null").c_str());
    require(t.ne[0] > 0 && t.ne[1] > 0 && t.ne[2] == 1 && t.ne[3] == 1, op,
            (std::string(label) + " must be a positive matrix").c_str());
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 0xFU) == 0; }

// Whether the bytes of two tensors intersect.
bool overlaps(const Tensor& a, const Tensor& b) {
    const auto begin_a = reinterpret_cast<std::uintptr_t>(a.data);
    const auto begin_b = reinterpret_cast<std::uintptr_t>(b.data);
    return begin_a < begin_b + b.bytes() && begin_b < begin_a + a.bytes();
}

void require_q8(const Weight& w, std::int32_t n, std::int32_t k, const char* op,
                const char* label) {
    require(w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit &&
                w.scale_dtype == DType::FP16 && w.qdata != nullptr && w.scales != nullptr,
            op, (std::string(label) + " must be a Q8_G32_FP16 row-split weight").c_str());
    require(w.n == n && w.k == k && w.padded_shape[1] == (k + 127) / 128 * 128, op,
            (std::string(label) + " has the wrong geometry").c_str());
}

// One mixer's geometry: S = kHcStreams streams of `hidden`, low rank `rank` (the K of Up, read
// from the weights), and `rows` = rank + inject Down rows. The general route contracts through
// linear(): with BF16 weights any hidden and rank that are whole numbers of 8-element chunks, with
// Q8 weights only the registered Q8 Linear shapes (validate_mix refuses the others before any
// launch). The small-T route is compiled for rank == kHcRank and hidden == kHcHidden (every
// format; the NVFP4 option has no other route) or, with Q8 weights, hidden == 2 kHcHidden
// (detail::hyper_connection_fused_compiled).
struct MixGeometry {
    std::int32_t hidden = 0;
    std::int32_t wide   = 0;
    std::int32_t rows   = 0;
    std::int32_t rank   = kHcRank;
};

MixGeometry validate_mix(const Tensor& residual, const HyperConnectionWeights& weights, float eps,
                         const Tensor& mixed, const Tensor* inject, const char* op) {
    require_matrix(residual, DType::BF16, op, "residual");
    require_matrix(mixed, DType::BF16, op, "mixed");
    const std::int32_t tokens = residual.ne[1];
    MixGeometry g;
    g.hidden = mixed.ne[0];
    g.wide   = kHcStreams * g.hidden;
    require(g.hidden % 8 == 0 && residual.ne[0] == g.wide && mixed.ne[1] == tokens, op,
            "residual must be [4*H,T] and mixed [H,T] with H a multiple of 8");
    require(weights.norm.dtype == DType::BF16 && weights.norm.data != nullptr &&
                weights.norm.is_contiguous() && weights.norm.numel() == g.wide,
            op, "norm must be contiguous BF16 [4*H]");
    require(eps > 0.0F && std::isfinite(eps), op, "eps must be positive and finite");
    g.rank = weights.up.k;
    require(g.rank > 0 && g.rank % 8 == 0, op,
            "the low rank (Up K) must be a positive multiple of 8");
    const std::int32_t rows = g.rank + (inject != nullptr ? kHcStreams : 0);
    g.rows                  = rows;
    if (weights.down.qtype == QType::NVFP4 || weights.up.qtype == QType::NVFP4) {
        // Option: NVFP4 Down stored with its rows padded to a multiple of 128 (the format's tile).
        require(weights.down.qtype == QType::NVFP4 && weights.up.qtype == QType::NVFP4, op,
                "down and up must share the weight format");
        (void)detail::validate_nvfp4_weight(weights.down, op);
        (void)detail::validate_nvfp4_weight(weights.up, op);
        require(g.hidden == kHcHidden && g.rank == kHcRank, op,
                "NVFP4 hyper-connection weights need H = 2560 and R = 320 (small-T route only)");
        require(weights.down.n == (rows + 127) / 128 * 128 && weights.down.k == g.wide, op,
                "NVFP4 down must be [round_up(R + I, 128), 4*2560]");
        require(weights.up.n == g.wide, op, "NVFP4 up must be [4*2560, 320]");
        require(residual.ne[1] <= kHcFusedMaxT, op,
                "NVFP4 hyper-connection weights are supported for T <= 8 only (no general route)");
    } else if (weights.down.qtype == QType::BF16 || weights.up.qtype == QType::BF16) {
        // BF16 Down(+inject)/Up: the small-T route for T <= 8 at H = 2560, the general route
        // otherwise.
        require(weights.down.qtype == QType::BF16 && weights.up.qtype == QType::BF16, op,
                "down and up must share the weight format");
        require(weights.down.qdata != nullptr && weights.up.qdata != nullptr, op,
                "BF16 down/up must be non-null");
        require(weights.down.n == rows && weights.down.k == g.wide, op,
                "BF16 down must be [R + I, 4*H]");
        require(weights.up.n == g.wide, op, "BF16 up must be [4*H, R]");
    } else {
        require_q8(weights.down, rows, g.wide, op, "down");
        require_q8(weights.up, g.wide, g.rank, op, "up");
        // Every T may take the general route, whose Q8 Linears serve registered shapes only.
        require(detail::q8_shape_registered(rows, g.wide) &&
                    detail::q8_shape_registered(g.wide, g.rank),
                op, "Q8 hyper-connection weights need H in {2560, 5120} and R in {320, 640}");
    }
    if (inject != nullptr) {
        require_matrix(*inject, DType::FP32, op, "inject");
        require(inject->ne[0] == kHcStreams && inject->ne[1] == tokens, op,
                "inject must be FP32 [4,T]");
        require(!overlaps(*inject, residual) && !overlaps(*inject, mixed), op,
                "inject must not overlap");
    }
    require(!overlaps(mixed, residual), op, "mixed must not overlap residual");
    return g;
}

void validate_combine(const Tensor& y, const Tensor& inject, const Tensor& residual,
                      const char* op) {
    require_matrix(y, DType::BF16, op, "y");
    require_matrix(inject, DType::FP32, op, "inject");
    require_matrix(residual, DType::BF16, op, "residual");
    require(residual.ne[0] == y.ne[0] * inject.ne[0] && residual.ne[1] == y.ne[1] &&
                inject.ne[1] == y.ne[1],
            op, "residual must be [S*H,T] for y [H,T] and inject [S,T]");
    require(!overlaps(y, residual) && !overlaps(inject, residual), op,
            "y and inject must not overlap residual");
}

// T103 diagnostic: NINFER_QWEN4_HC_GENERAL=1 sends small-T mixers through the general route too
// (the route of T > 8), to separate the route from the weight format.
bool general_route_forced() {
    static const bool forced = [] {
        const char* value = std::getenv("NINFER_QWEN4_HC_GENERAL");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return forced;
}

bool fused_route(std::int32_t tokens, const HyperConnectionWeights& weights,
                 const MixGeometry& geometry) {
    return !general_route_forced() && geometry.rank == kHcRank &&
           detail::hyper_connection_fused_compiled(geometry.hidden,
                                                   weights.down.qtype == QType::NVFP4,
                                                   weights.down.qtype == QType::BF16) &&
           tokens <= kHcFusedMaxT && aligned16(weights.down.qdata) && aligned16(weights.up.qdata) &&
           aligned16(weights.down.scales) && aligned16(weights.up.scales) &&
           aligned16(weights.norm.data) && detail::hyper_connection_fused_supported();
}

void run_fused(Tensor& residual, const Tensor* previous_y, const Tensor* previous_inject,
               const HyperConnectionWeights& weights, float eps, Tensor& mixed, Tensor* inject,
               const MixGeometry& geometry, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    Tensor squares            = workspace.alloc(DType::FP32, {kHcStreams, tokens});
    Tensor low                = workspace.alloc(DType::FP16, {kHcRank, tokens});
    detail::HcFusedMix params;
    params.residual = static_cast<__nv_bfloat16*>(residual.data);
    if (previous_y != nullptr) {
        params.previous_y      = static_cast<const __nv_bfloat16*>(previous_y->data);
        params.previous_inject = static_cast<const float*>(previous_inject->data);
    }
    params.norm       = static_cast<const __nv_bfloat16*>(weights.norm.data);
    params.down_codes = static_cast<const std::int8_t*>(weights.down.qdata);
    params.up_codes   = static_cast<const std::int8_t*>(weights.up.qdata);
    if (weights.down.qtype == QType::BF16) {
        params.bf16 = true;
    } else if (weights.down.qtype == QType::NVFP4) {
        params.fp4              = true;
        params.down_fp4_scales  = static_cast<const std::uint8_t*>(weights.down.scales);
        params.up_fp4_scales    = static_cast<const std::uint8_t*>(weights.up.scales);
        params.down_divisor     = weights.down.weight_scale_divisor;
        params.up_divisor       = weights.up.weight_scale_divisor;
        params.down_padded_rows = weights.down.n;
    } else {
        params.down_scales = static_cast<const __half*>(weights.down.scales);
        params.up_scales   = static_cast<const __half*>(weights.up.scales);
    }
    params.mixed       = static_cast<__nv_bfloat16*>(mixed.data);
    params.inject      = inject != nullptr ? static_cast<float*>(inject->data) : nullptr;
    params.square_sums = static_cast<float*>(squares.data);
    params.low         = static_cast<__half*>(low.data);
    params.rows        = geometry.rows;
    params.tokens      = tokens;
    params.hidden      = geometry.hidden;
    params.eps         = eps;
    detail::hyper_connection_fused_mix_launch(params, stream);
}

void run_general(const Tensor& residual, const HyperConnectionWeights& weights, float eps,
                 Tensor& mixed, Tensor* inject, const MixGeometry& geometry,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    Tensor normalized         = workspace.alloc(DType::BF16, {geometry.wide, tokens});
    Tensor projection         = workspace.alloc(DType::BF16, {geometry.rows, tokens});
    Tensor lowrank            = workspace.alloc(DType::BF16, {geometry.rank, tokens});
    Tensor up                 = workspace.alloc(DType::BF16, {geometry.wide, tokens});
    detail::grouped_offset_rmsnorm_launch(residual, weights.norm, kHcStreams, eps, normalized,
                                          stream);
    linear(normalized, weights.down, projection, stream);
    detail::hyper_connection_gates_launch(projection, kHcStreams, lowrank, inject, stream);
    // The fused epilogue reads `normalized` and writes `mixed` 16 bytes at a time; a misaligned
    // operand keeps the unfused route, whose collapse falls back to its scalar kernel.
    if (detail::hc_up_collapse_supported(weights.up, geometry.hidden, tokens) &&
        aligned16(normalized.data) && aligned16(mixed.data)) {
        detail::hc_up_collapse_launch(lowrank, weights.up, normalized, mixed, stream);
        return;
    }
    linear(lowrank, weights.up, up, stream);
    detail::hyper_connection_collapse_launch(up, normalized, kHcStreams, mixed, stream);
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::size_t general_bytes(std::int32_t tokens, bool inject, std::int32_t hidden,
                          std::int32_t rank) {
    const auto t           = static_cast<std::size_t>(tokens);
    const auto r           = static_cast<std::size_t>(rank);
    const std::size_t rows = r + (inject ? kHcStreams : 0);
    const std::size_t wide = static_cast<std::size_t>(kHcStreams) * hidden;
    return align256(t * wide * 2) + align256(t * rows * 2) + align256(t * r * 2) + t * wide * 2;
}

} // namespace

void grouped_offset_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "grouped_offset_rmsnorm";
    require_matrix(x, DType::BF16, op, "x");
    require_matrix(out, DType::BF16, op, "out");
    require(weight.dtype == DType::BF16 && weight.is_contiguous() && weight.data != nullptr &&
                weight.numel() == x.ne[0],
            op, "weight must be contiguous BF16 [G*H]");
    require(groups > 0 && x.ne[0] % groups == 0, op, "groups must divide the row width");
    require(out.ne[0] == x.ne[0] && out.ne[1] == x.ne[1], op, "x/out shapes differ");
    require(eps > 0 && std::isfinite(eps), op, "eps must be positive and finite");
    require(!overlaps(x, out), op, "x and out must not overlap");
    detail::grouped_offset_rmsnorm_launch(x, weight, groups, eps, out, stream);
}

void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_expand";
    require_matrix(x, DType::BF16, op, "x");
    require_matrix(residual, DType::BF16, op, "residual");
    require(streams > 0 && residual.ne[0] == x.ne[0] * streams && residual.ne[1] == x.ne[1], op,
            "residual must be [S*H,T]");
    require(!overlaps(x, residual), op, "x and residual must not overlap");
    detail::hyper_connection_expand_launch(x, streams, residual, stream);
}

void hyper_connection_combine(const Tensor& y, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_combine";
    validate_combine(y, inject, residual, op);
    detail::hyper_connection_combine_launch(y, inject, residual, stream);
}

void hyper_connection_mix(const Tensor& residual, const HyperConnectionWeights& weights, float eps,
                          Tensor& mixed, Tensor* inject, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    constexpr const char* op   = "hyper_connection_mix";
    const MixGeometry geometry = validate_mix(residual, weights, eps, mixed, inject, op);
    const std::int32_t tokens  = residual.ne[1];
    auto scope                 = workspace.scope();
    if (fused_route(tokens, weights, geometry)) {
        Tensor read_only = residual; // the route writes the residual only with a pending combine
        run_fused(read_only, nullptr, nullptr, weights, eps, mixed, inject, geometry, workspace,
                  stream);
        return;
    }
    run_general(residual, weights, eps, mixed, inject, geometry, workspace, stream);
}

void hyper_connection_combine_mix(Tensor& residual, const Tensor& previous_y,
                                  const Tensor& previous_inject,
                                  const HyperConnectionWeights& weights, float eps, Tensor& mixed,
                                  Tensor* inject, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op   = "hyper_connection_combine_mix";
    const MixGeometry geometry = validate_mix(residual, weights, eps, mixed, inject, op);
    validate_combine(previous_y, previous_inject, residual, op);
    require(previous_inject.ne[0] == kHcStreams, op, "previous_inject must be FP32 [4,T]");
    require(inject == nullptr || !overlaps(*inject, previous_inject), op,
            "inject must not overlap previous_inject");
    const std::int32_t tokens = residual.ne[1];
    auto scope                = workspace.scope();
    if (fused_route(tokens, weights, geometry)) {
        run_fused(residual, &previous_y, &previous_inject, weights, eps, mixed, inject, geometry,
                  workspace, stream);
        return;
    }
    detail::hyper_connection_combine_launch(previous_y, previous_inject, residual, stream);
    run_general(residual, weights, eps, mixed, inject, geometry, workspace, stream);
}

std::size_t hyper_connection_mix_workspace_bytes(std::int32_t min_tokens, std::int32_t max_tokens,
                                                 bool inject, std::int32_t hidden,
                                                 std::int32_t rank) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("hyper_connection_mix_workspace_bytes: invalid T interval");
    }
    if (hidden <= 0 || hidden % 8 != 0) {
        throw std::invalid_argument("hyper_connection_mix_workspace_bytes: invalid hidden width");
    }
    if (rank <= 0 || rank % 8 != 0) {
        throw std::invalid_argument("hyper_connection_mix_workspace_bytes: invalid low rank");
    }
    std::size_t bytes = general_bytes(max_tokens, inject, hidden, rank);
    // The small-T workspace does not depend on H or on the format.
    if (rank == kHcRank && detail::hyper_connection_fused_compiled(hidden, false, false) &&
        min_tokens <= kHcFusedMaxT) {
        bytes = std::max(bytes, detail::hyper_connection_fused_workspace_bytes(
                                    std::min(max_tokens, kHcFusedMaxT)));
    }
    return bytes;
}

} // namespace ninfer::ops
