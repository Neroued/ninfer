#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

#include <span>

namespace ninfer::ops::detail {

struct Mxfp8BiasRoute {
    int last_tokens;
    void (*launch)(const Tensor&, const Weight&, const Tensor&, Tensor&, Mxfp8A8Workspace,
                   cudaStream_t);
    bool quantize;
    int code_token_alignment = 1;
    bool tensor_core         = true;

    int code_tokens(int tokens) const {
        return mxfp8_a8_padded_tokens(tokens, code_token_alignment);
    }
};

struct Mxfp8BiasShape {
    int rows, k;
    std::span<const Mxfp8BiasRoute> a16, a8;
};

extern const Mxfp8BiasShape kMxfp8BiasN1024K1024;
Mxfp8BiasRoute select_mxfp8_bias_route(int rows, int k, int tokens, LinearPolicy);
std::size_t mxfp8_bias_workspace_capacity_bytes(int rows, int k, LinearPolicy, int first, int last);
void mxfp8_bias_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&, LinearPolicy,
                         MemoryArena*, cudaStream_t);

} // namespace ninfer::ops::detail
