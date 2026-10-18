#include "core/platform.h"
#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/common/host/operands.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include "ops/moe/swiglu/dispatch.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace operand        = common::operand;
constexpr const char* op = "swiglu";

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void profile(const SwiGLUProfile& p) {
#if NINFER_TARGET_SPARK
    const bool nvfp4 =
        p.hidden == 4096 && p.intermediate == 6144 && p.gate_up_type == QType::NVFP4 &&
        p.down_type == QType::NVFP4 && p.gate_up_layout == QuantLayout::BlockScaleK16M128x4 &&
        p.down_layout == QuantLayout::BlockScaleK16M128x4 &&
        p.gate_up_policy == LinearPolicy::AllowA4 && p.down_policy == LinearPolicy::AllowA4;
    const bool bf16 = p.hidden == 4096 && p.intermediate == 1024 && p.gate_up_type == QType::BF16 &&
                      p.down_type == QType::BF16 && p.gate_up_layout == QuantLayout::Contiguous &&
                      p.down_layout == QuantLayout::Contiguous &&
                      p.gate_up_policy == LinearPolicy::A16Only &&
                      p.down_policy == LinearPolicy::A16Only;
    require(nvfp4 || bf16, "swiglu: unsupported profile");
    require(p.clamp_limit && std::isfinite(*p.clamp_limit) && *p.clamp_limit > 0,
            "swiglu: clamp limit must be finite and positive");
#else
    throw std::invalid_argument("swiglu: no registered profile for this target");
#endif
}

std::size_t capacity(const SwiGLUProfile& p, int tokens) {
    WorkspaceLayoutBuilder layout;
    if (p.gate_up_type == QType::BF16) {
        (void)layout.alloc_bytes(std::size_t(p.intermediate) * tokens * sizeof(float));
        return layout.peak_bytes(1);
    }
    if (tokens >= 129) {
        (void)detail::allocate_nvfp4_a4_workspace(layout, tokens, p.intermediate);
        (void)detail::allocate_nvfp4_a4_workspace(layout, tokens, p.hidden);
    } else {
        (void)layout.alloc(DType::FP32, {p.intermediate, tokens});
        if (tokens >= 3) (void)detail::allocate_nvfp4_a4_workspace(layout, tokens, p.intermediate);
    }
    return layout.peak_bytes(1);
}
} // namespace

std::size_t swiglu_workspace_capacity_bytes(const SwiGLUProfile& p, int first, int last) {
    profile(p);
    require(first > 0 && last >= first, "swiglu: invalid token interval");
    auto result = std::max(capacity(p, first), capacity(p, last));
    // Switching from the FP32 small-call product to a direct A4 product reduces live storage.
    if (first <= 128 && last >= 128) result = std::max(result, capacity(p, 128));
    return result;
}

void swiglu(const Tensor& x, const SwiGLUProfile& p, const SwiGLUWeights& weights, Tensor& out,
            MemoryArena& ws, CudaExecutionView execution) {
    profile(p);
    const int tokens = x.ne[1];
    require(tokens > 0, "swiglu: token count must be positive");
    operand::tensor(x, DType::BF16, {p.hidden, tokens}, 16, op, "x", execution.device());
    operand::tensor(out, DType::BF16, {p.hidden, tokens}, 16, op, "destination",
                    execution.device());
    auto expected        = swiglu_profile(weights);
    expected.clamp_limit = p.clamp_limit;
    require(expected == p, "swiglu: prepared weights differ from profile");
    operand::device(ws.device(), execution.device(), op, "workspace");
    const auto bytes = capacity(p, tokens);
    require(bytes <= ws.capacity() - ws.used(), "swiglu: insufficient workspace");
    auto* cursor = static_cast<std::byte*>(ws.storage().data) + ws.used();
    require(operand::aligned(cursor, 256), "swiglu: workspace cursor must be 256-byte aligned");
    const auto output = operand::range(out, op), scratch = operand::range(cursor, bytes, op);
    operand::disjoint(output, scratch, op);
    const auto read = [&](const void* data, std::size_t size) {
        const auto r = operand::range(data, size, op);
        operand::disjoint(output, r, op);
        operand::disjoint(scratch, r, op);
    };
    read(x.data, x.bytes());
    for (const Weight* w : {&weights.gate_up.weight, &weights.down.weight}) {
        operand::device(w->device, execution.device(), op, "weight");
        if (w->qtype == QType::BF16) {
            require(w->ndim == 2 && w->shape[0] == w->n && w->shape[1] == w->k &&
                        w->padded_shape[0] == w->n && w->padded_shape[1] == w->k &&
                        w->layout == QuantLayout::Contiguous && w->group == 0 &&
                        w->group_size == 0 && !w->scales && !w->qhigh && !w->high_plane_bytes &&
                        operand::aligned(w->qdata, 16) &&
                        w->payload_bytes >= std::uint64_t(w->n) * w->k * 2,
                    "swiglu: invalid BF16 weight");
            read(w->qdata, std::size_t(w->n) * w->k * 2);
        } else {
            (void)common::validate_nvfp4_weight(*w, op);
            read(w->qdata, std::size_t(w->n) * w->k / 2);
            read(w->scales, std::size_t(w->n) * w->k / 16);
        }
    }
#if NINFER_TARGET_SPARK
    if (p.gate_up_type == QType::BF16) {
        detail::swiglu_ffn::launch_bf16_ffn(x, weights, out, *p.clamp_limit, ws,
                                            execution.stream());
        return;
    }
    if (tokens >= 129) {
        detail::swiglu_ffn::launch_nvfp4_quantized_ffn(x, weights, out, *p.clamp_limit, ws,
                                                       execution.stream());
        return;
    }
    auto scope   = ws.scope();
    auto product = ws.alloc(DType::FP32, {p.intermediate, tokens});
    detail::swiglu_ffn::launch_nvfp4_gu_float(x, weights.gate_up.weight,
                                              static_cast<float*>(product.data), *p.clamp_limit, ws,
                                              execution.stream());
    detail::swiglu_ffn::launch_nvfp4_down(static_cast<const float*>(product.data),
                                          weights.down.weight, out, tokens, ws, execution.stream());
#endif
}
} // namespace ninfer::ops
