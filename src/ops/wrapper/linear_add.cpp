#include "core/weight.h"
#include "ninfer/ops/linear_add.h"

#include "ops/linear_add/bf16/bf16_linear_add_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_add/q4/q4_linear_add_dispatch.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t columns,
                    const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_add: invalid ") + name);
    }
}

void require_q4(const Weight& w) {
    if (w.qtype != QType::Q4_G64_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.group != 64 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh != nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q4_G64_FP16 row-split");
    }
}

void require_q5(const Weight& w) {
    if (w.qtype != QType::Q5_G64_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.group != 64 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q5_G64_FP16 row-split");
    }
}

void require_q8(const Weight& w) {
    if (w.qtype != QType::Q8_G32_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 32 || w.group != 32 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh != nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q8_G32_FP16 row-split");
    }
}

void require_bf16(const Weight& w) {
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous || w.qdata == nullptr) {
        throw std::invalid_argument("linear_add: weight must be contiguous BF16");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_add: invalid compute policy");
}

} // namespace

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    return linear_add_workspace_capacity_bytes(qtype, output_rows, input_rows,
                                               LinearPolicy::A16Only, min_tokens, max_tokens);
}

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_add workspace: invalid token interval");
    }
    if (qtype == QType::BF16) {
        (void)detail::bf16_linear_add_select(output_rows, input_rows, min_tokens);
        (void)detail::bf16_linear_add_select(output_rows, input_rows, max_tokens);
        return 0;
    }
    if (qtype == QType::Q4_G64_FP16) {
        (void)detail::select_q4_linear_add(output_rows, input_rows, min_tokens);
        (void)detail::select_q4_linear_add(output_rows, input_rows, max_tokens);
        return 0;
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_add_resolve_plan({output_rows, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_add_resolve_plan({output_rows, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q5_G64_FP16) {
        return detail::q5_linear_add_capacity_workspace_bytes(output_rows, input_rows, input_rows,
                                                              min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4) {
        const bool supported = (output_rows == detail::Nvfp4N5120K6144::kOutputRows &&
                                input_rows == detail::Nvfp4N5120K6144::kInputRows) ||
                               (output_rows == detail::Nvfp4N5120K17408::kOutputRows &&
                                input_rows == detail::Nvfp4N5120K17408::kInputRows);
        if (!supported) {
            throw std::invalid_argument("linear_add workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_add_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                                 min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        const bool supported = (output_rows == detail::Fp8N5120K6144::kOutputRows &&
                                input_rows == detail::Fp8N5120K6144::kInputRows) ||
                               (output_rows == detail::Fp8N5120K17408::kOutputRows &&
                                input_rows == detail::Fp8N5120K17408::kInputRows);
        if (!supported) {
            throw std::invalid_argument("linear_add workspace: unsupported FP8 profile");
        }
        return detail::fp8_linear_add_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                               min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_add workspace: unsupported weight format");
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    linear_add(x, w, residual_out, LinearPolicy::A16Only, ws, stream);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t t = x.ne[1];
    if (t <= 0) { throw std::invalid_argument("linear_add: T must be positive"); }
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");
    if (overlaps(x, residual_out)) {
        throw std::invalid_argument("linear_add: x and residual_out must not overlap");
    }

    if (w.qtype == QType::BF16) {
        require_bf16(w);
        if (!detail::bf16_linear_add_admits(w.n, w.k, t)) {
            throw std::invalid_argument("linear_add: unsupported BF16 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16)) {
            throw std::invalid_argument(
                "linear_add: BF16 requires 16-byte x/residual/weight alignment");
        }
        (void)ws;
        detail::bf16_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::Q4_G64_FP16) {
        require_q4(w);
        const auto launch = detail::select_q4_linear_add(w.n, w.k, t);
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: Q4 requires 16-byte x/residual/code/scale alignment");
        }
        launch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::Q5_G64_FP16) {
        require_q5(w);
        const bool supported_shape = (w.n == 5120 && w.k == 17408) || (w.n == 5120 && w.k == 6144);
        if (!supported_shape) { throw std::invalid_argument("linear_add: unsupported Q5 shape"); }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: Q5 requires 16-byte x/residual/code/high/scale alignment");
        }
        detail::q5_linear_add_dispatch(x, w, residual_out, ws, stream);
        return;
    }

    if (w.qtype == QType::Q8_G32_FP16) {
        require_q8(w);
        if (!detail::q8_linear_add_admits({w.n, w.k, w.padded_shape[1], t})) {
            throw std::invalid_argument("linear_add: unsupported Q8 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: Q8 requires 16-byte x/residual/code/scale alignment");
        }
        (void)ws;
        detail::q8_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::NVFP4) {
        detail::validate_nvfp4_weight(w, "nvfp4 linear_add");
        const bool supported_shape = (w.n == detail::Nvfp4N5120K6144::kOutputRows &&
                                      w.k == detail::Nvfp4N5120K6144::kInputRows) ||
                                     (w.n == detail::Nvfp4N5120K17408::kOutputRows &&
                                      w.k == detail::Nvfp4N5120K17408::kInputRows);
        if (!supported_shape) {
            throw std::invalid_argument("nvfp4 linear_add: unsupported weight shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16)) {
            throw std::invalid_argument("linear_add: NVFP4 requires 16-byte x/residual alignment");
        }
        detail::nvfp4_linear_add_dispatch(x, w, residual_out, policy, ws, stream);
        return;
    }

    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        (void)detail::validate_fp8_weight(w, "fp8 linear_add");
        const bool supported_shape = (w.n == detail::Fp8N5120K6144::kOutputRows &&
                                      w.k == detail::Fp8N5120K6144::kInputRows) ||
                                     (w.n == detail::Fp8N5120K17408::kOutputRows &&
                                      w.k == detail::Fp8N5120K17408::kInputRows);
        if (!supported_shape) {
            throw std::invalid_argument("fp8 linear_add: unsupported weight shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16)) {
            throw std::invalid_argument("linear_add: FP8 requires 16-byte x/residual alignment");
        }
        detail::fp8_linear_add_dispatch(x, w, residual_out, policy, ws, stream);
        return;
    }

    throw std::invalid_argument("linear_add: unsupported weight format");
}

} // namespace ninfer::ops
