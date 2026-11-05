#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

#include <span>

namespace ninfer::ops::detail {

struct Mxfp8SwiGluRoute {
    int last_tokens;
    void (*launch)(const Tensor&, const Weight&, Tensor&, float limit, bool clamped,
                   Mxfp8A8Workspace, cudaStream_t);
    bool quantize;
    int code_token_alignment = 1;
    bool tensor_core         = true;
    bool padded_scales       = false;

    int code_tokens(int tokens) const {
        return mxfp8_a8_padded_tokens(tokens, code_token_alignment);
    }

    int scale_tokens(int tokens) const { return padded_scales ? code_tokens(tokens) : tokens; }
};

struct Mxfp8SwiGluShape {
    int rows, k;
    std::span<const Mxfp8SwiGluRoute> a16, a8;
};

extern const Mxfp8SwiGluShape kMxfp8SwiGluN12288K4096;
extern const Mxfp8SwiGluShape kMxfp8SwiGluN2048K4096;
Mxfp8SwiGluRoute select_mxfp8_swiglu_route(int rows, int k, int tokens, LinearPolicy);
std::size_t mxfp8_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy, int first,
                                                  int last);
void mxfp8_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, float limit, bool clamped,
                           LinearPolicy, MemoryArena&, cudaStream_t);

} // namespace ninfer::ops::detail
