#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"
#include "ops/linear_attention/gated_delta_net/input_conv/workspace.h"
#include <algorithm>

namespace ninfer::ops::detail::gdn_input {

ProjectionPlan select_projection_plan(const WeightProfile& weights, std::int32_t columns) {
    if (weights.format == Format::Fp8 && allows_a8(weights.policy) && columns >= 17)
        return {Compute::A8};
    if (weights.format == Format::Nvfp4 && allows_a4(weights.policy)) return {Compute::A4};
    return {Compute::A16};
}

ConvPlan select_conv_plan(const WeightProfile& weights, Batch batch) {
    ConvPlan plan{select_projection_plan(weights, batch.columns())};
    switch (weights.format) {
    case Format::Q4Q5:
        if (batch.rows == 1 && (batch.tokens <= 3 || batch.tokens == 5 || batch.tokens == 6))
            plan.kernel = ConvKernel::Q4Q5Fused;
        break;
    case Format::Q8:
        return select_q8_conv_plan(weights, batch);
    case Format::Fp8:
        if (batch.rows == 1 && batch.tokens <= 3) plan.kernel = ConvKernel::Fp8Fused;
        break;
    case Format::Nvfp4:
        if (batch.rows == 1) {
            if (batch.tokens == 1)
                plan.kernel = ConvKernel::Nvfp4Decode;
            else if (batch.tokens <= (allows_a4(weights.policy) ? 3 : 16))
                plan.kernel = ConvKernel::Nvfp4Window;
            // This strategy consumes BF16 projected values, independently of weight format.
            plan.post = ProjectedConvKernel::TokenParallel;
        }
        break;
    }
    if (!plan.materialized()) plan.projection.compute = Compute::A16;
    return plan;
}

std::size_t convolution_capacity(const WeightProfile& weights, Publication publication,
                                 std::int32_t batch, std::int32_t first, std::int32_t last) {
    validate_batch_interval(publication, batch, first, last);
    const auto at = [&](std::int32_t tokens) {
        const auto plan = select_conv_plan(weights, {tokens, batch});
        return workspace_bytes(weights, plan.projection, tokens * batch,
                               publication == Publication::Snapshot && plan.materialized());
    };
    std::size_t capacity = 0;
    // Short-window routes are not monotone. Above 16, B=1 always materializes and the
    // allocation recipe grows monotonically, so the upper endpoint covers that whole region.
    for (int tokens = first; tokens <= std::min(last, 16); ++tokens)
        capacity = std::max(capacity, at(tokens));
    if (last > 16) capacity = std::max(capacity, at(last));
    return capacity;
}

} // namespace ninfer::ops::detail::gdn_input
