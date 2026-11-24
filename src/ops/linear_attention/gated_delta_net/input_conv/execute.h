#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"

namespace ninfer::ops::detail::gdn_input {

struct Scratch;

struct ProjectedOutputs {
    Tensor qkv, z;
};

void launch_projection(const Tensor& x, const WeightOperands& weights, ProjectedOutputs outputs,
                       ProjectionPlan plan, const Scratch& scratch, cudaStream_t stream);
void launch_snapshot(const Tensor& x, const WeightOperands& weights, const Tensor& conv_weight,
                     ConstStateFieldView state, HistoryDestination destination, const Tensor& valid,
                     GdnConvOutputs outputs, Batch batch, ConvPlan plan, const Scratch& scratch,
                     cudaStream_t stream);
void launch_record(const Tensor& x, const WeightOperands& weights, const Tensor& conv_weight,
                   ConstStateFieldView state, const Tensor& valid, GdnConvOutputs outputs,
                   Tensor& record, Batch batch, ConvPlan plan, const Scratch& scratch,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail::gdn_input
