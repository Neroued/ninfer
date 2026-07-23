#include "ninfer/ops/swa.h"

#include "core/layout.h"
#include "ops/launcher/swa.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQHeads  = 32;
constexpr std::int32_t kKVHeads = 8;
constexpr std::int32_t kWindow  = 4096;
constexpr float kExpectedScale  = 0.08838834764831844055f;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_context(const CyclicKVCacheLayerView& context, const char* op) {
    if (context.dtype != DType::BF16 || context.quant_group != 0 ||
        context.num_kv_heads != kKVHeads || context.head_dim != kHeadDim ||
        context.capacity != kWindow || context.padded_capacity < context.capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic context");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    if (context.k.dtype != DType::BF16 || context.v.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": context K/V must be BF16");
    }
    require_shape(context.k, kHeadDim, padded, kKVHeads, 1, op, "context k");
    require_shape(context.v, kHeadDim, padded, kKVHeads, 1, op, "context v");
    require_contiguous_nonnull(context.k, op, "context k");
    require_contiguous_nonnull(context.v, op, "context v");
    if (context.k_scale.data != nullptr || context.v_scale.data != nullptr) {
        throw std::invalid_argument(std::string(op) + ": BF16 context must not have scales");
    }
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

PartialWorkspace allocate_workspace(WorkspaceArena& workspace, std::int32_t tokens,
                                    std::int32_t splits) {
    return {
        workspace.alloc(DType::BF16, {kHeadDim, kQHeads, tokens, splits}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits}),
    };
}

} // namespace

std::size_t swa_workspace_bytes(std::int32_t tokens) {
    if (tokens < 1 || tokens > 16) { return 0; }
    const auto plan = detail::swa_resolve_plan(
        tokens, SwaContextExecutionEnvelope{1, std::numeric_limits<std::int32_t>::max()});
    const Tensor acc(nullptr, DType::BF16, {kHeadDim, kQHeads, tokens, plan.split_capacity});
    const Tensor stat(nullptr, DType::FP32, {kQHeads, tokens, plan.split_capacity});
    LayoutBuilder layout;
    (void)layout.add(acc.bytes(), 256, "SWA partial numerator");
    (void)layout.add(stat.bytes(), 256, "SWA partial max");
    (void)layout.add(stat.bytes(), 256, "SWA partial sum");
    return layout.finish(256, "SWA workspace");
}

void swa(const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
         float scale, const CyclicKVCacheLayerView& context, SwaContextExecutionEnvelope envelope,
         WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "swa";
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("swa: q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("swa: positions must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("swa: optimized domain is T=1..16");
    }
    require_shape(q, kHeadDim, kQHeads, tokens, 1, op, "q");
    require_shape(query_k, kHeadDim, kKVHeads, tokens, 1, op, "query k");
    require_shape(query_v, kHeadDim, kKVHeads, tokens, 1, op, "query v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, kQHeads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(query_k, op, "query k");
    require_contiguous_nonnull(query_v, op, "query v");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_context(context, op);
    if (envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("swa: invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument("swa: scale must be 1/sqrt(128)");
    }

    auto scope               = workspace.scope();
    const auto plan          = detail::swa_resolve_plan(tokens, envelope);
    PartialWorkspace partial = allocate_workspace(workspace, tokens, plan.split_capacity);
    detail::swa_launch(q, query_k, query_v, positions, scale, context, plan, partial.acc, partial.m,
                       partial.l, out, stream);
}

} // namespace ninfer::ops
