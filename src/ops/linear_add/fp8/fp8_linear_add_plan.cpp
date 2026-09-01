#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    LinearThenAdd,
#endif
    A8,
};

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != Fp8Residual6144Geometry::kOutputRows ||
        (input_rows != Fp8Residual6144Geometry::kInputRows &&
         input_rows != Fp8Residual17408Geometry::kInputRows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        return tokens == 1 ? Fp8LinearAddRoute::A16 : Fp8LinearAddRoute::LinearThenAdd;
#else
        return Fp8LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_add: unsupported policy");
    }
    const std::int32_t first_a8 = input_rows == Fp8Residual6144Geometry::kInputRows ? 22 : 25;
    return tokens >= first_a8 ? Fp8LinearAddRoute::A8 : Fp8LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kFp8LastSmallT) {
        const std::int32_t active = std::min(kFp8LastSmallT, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            fp8_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            fp8_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

void launch_linear_then_add(const Tensor& x, const Weight& weight, Tensor& residual,
                            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope      = workspace.scope();
    Tensor projected = allocate_projected(workspace, weight.n, x.ne[1]);
    if (x.ne[1] >= 33) {
        fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    } else {
        const std::size_t linear_bytes = std::max<std::size_t>(
            linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, weight.n, weight.k,
                                            LinearPolicy::A16Only, x.ne[1], x.ne[1]),
            256);
        WorkspaceArena linear_workspace(workspace.alloc_bytes(linear_bytes, 256));
        linear(x, weight, projected, LinearPolicy::A16Only, linear_workspace, stream);
    }
    residual_add(projected, residual, stream);
}

std::size_t linear_then_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes =
        tokens >= 33
            ? fp8_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens)
            : std::max<std::size_t>(
                  linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, output_rows,
                                                  input_rows, LinearPolicy::A16Only, tokens,
                                                  tokens),
                  256);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return layout.peak_bytes(1);
}
#endif

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Fp8LinearAddRoute route = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (route == Fp8LinearAddRoute::A8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Fp8LinearAddRoute::LinearThenAdd) {
        return linear_then_add_workspace_bytes(output_rows, input_rows, max_tokens);
    }
#endif
    return 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Fp8LinearAddRoute::LinearThenAdd) {
        launch_linear_then_add(x, weight, residual, workspace, stream);
        return;
    }
#endif
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
}

} // namespace ninfer::ops::detail
