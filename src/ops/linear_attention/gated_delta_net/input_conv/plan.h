#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/operands.h"

namespace ninfer::ops::detail::gdn_input {

inline constexpr int kQ8AggregateColumns = 32;

enum class Compute { A16, A8, A4 };
enum class ConvKernel {
    Materialized,
    Q4Q5Fused,
    Q8Decode,
    Q8SlicedK,
    Q8Window,
    Q8Aggregate,
    Fp8Fused,
    Nvfp4Decode,
    Nvfp4Window,
    Q8TmaBatch,
};
enum class ProjectedConvKernel { Sequential, TokenParallel, TokenParallel8 };

struct ProjectionPlan {
    Compute compute = Compute::A16;
};

struct ConvPlan {
    ProjectionPlan projection;
    ConvKernel kernel        = ConvKernel::Materialized;
    ProjectedConvKernel post = ProjectedConvKernel::Sequential;

    [[nodiscard]] bool materialized() const { return kernel == ConvKernel::Materialized; }
};

ProjectionPlan select_projection_plan(const WeightProfile& weights, std::int32_t columns);
// Publication does not select arithmetic. Snapshot and record consume the same physical block.
ConvPlan select_q8_conv_plan(const WeightProfile& weights, Batch batch);
ConvPlan select_conv_plan(const WeightProfile& weights, Batch batch);
std::size_t convolution_capacity(const WeightProfile& weights, Publication publication,
                                 std::int32_t batch, std::int32_t first, std::int32_t last);

} // namespace ninfer::ops::detail::gdn_input
