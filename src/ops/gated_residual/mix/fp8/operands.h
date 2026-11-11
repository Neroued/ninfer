#pragma once
#include "ninfer/ops/gated_residual.h"
#include "core/layout.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::hc::mix::fp8 {
inline constexpr int kWidth = 5120, kRank = 640, kStreams = 4, kTotal = kWidth * kStreams;

struct Weights {
    const SingleProjectionWeight& down;
    const SingleProjectionWeight& up;
    const Weight* injection;
};

enum class UpSchedule { Thin, Medium, WideRows, WideTokens };
int up_token_tile(UpSchedule);

struct Plan {
    bool down_a8, up_a8;
    int down_tile;
    UpSchedule up;
};

struct Scratch {
    __nv_bfloat16* u;
    Fp8A8Workspace encoded;
};

struct Layout {
    LayoutRegion u, codes, scales, partials;
    std::size_t bytes;
};

Plan plan(LinearPolicy down, LinearPolicy up, int columns);
Layout layout(int columns, Plan plan);
void launch(Weights weights, const Tensor& z, Tensor& output, Tensor* injection, Scratch scratch,
            Plan plan, CudaExecutionView execution);
} // namespace ninfer::ops::detail::hc::mix::fp8
