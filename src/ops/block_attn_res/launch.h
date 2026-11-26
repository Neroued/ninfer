#pragma once

#include "ninfer/ops/block_attn_res.h"

#include <cstdint>
#include <initializer_list>

namespace ninfer::ops::attn_res_detail {

struct Mix {
    const std::uint16_t* stack;
    const std::uint16_t* addend;
    const float* queries;
    const std::uint16_t* norm_weight;
    void* out;
    int d, columns, stacked, q;
    float inv_d;
    BlockAttnResParameters params;
};

// Partial/addend shape can have a batch axis in the fused entry. All other
// attention operands use the same flattened column layout.
Mix prepare_mix(const Tensor& stack, const Tensor& addend, const Tensor& queries,
                const Tensor& norm_weight, const BlockAttnResParameters& params, Tensor& out, int d,
                int width, int batch, Device device, const char* op);

void require_disjoint(const Tensor& output, std::initializer_list<const Tensor*> inputs,
                      const char* op);

void launch(Mix mix, const std::uint16_t* partial, CudaExecutionView execution);

} // namespace ninfer::ops::attn_res_detail
