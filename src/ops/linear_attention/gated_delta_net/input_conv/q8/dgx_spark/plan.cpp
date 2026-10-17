#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"

namespace ninfer::ops::detail::gdn_input {
ConvPlan select_q8_conv_plan(const WeightProfile& weights, Batch batch) {
    ConvPlan plan{{Compute::A16}};
    if (weights.geometry.input_rows == 2560) {
        if (batch.rows == 1 && batch.tokens <= 16)
            plan.kernel = ConvKernel::Q8Window;
        else if (batch.columns() <= 16 && !(batch.tokens == 1 && batch.rows >= 6) &&
                 !(batch.tokens == 2 && batch.rows == 4))
            plan.kernel = ConvKernel::Q8Aggregate;
        if (batch.rows > 1 && batch.columns() > 64 && batch.columns() <= 128)
            plan.kernel = ConvKernel::Q8TmaBatch;
        // Q8 B=1 materialized windows can consume token tiles independently;
        // every channel's initial history is loaded before that CTA publishes.
        if (batch.rows == 1 && batch.tokens > 16) plan.post = ProjectedConvKernel::TokenParallel8;
    } else if (batch.rows == 1) {
        if (batch.tokens == 1)
            plan.kernel = ConvKernel::Q8Decode;
        else if (batch.tokens <= 16)
            plan.kernel = ConvKernel::Q8SlicedK;
    }

    return plan;
}
} // namespace ninfer::ops::detail::gdn_input
