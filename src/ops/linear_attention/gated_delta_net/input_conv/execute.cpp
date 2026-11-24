#include "core/platform.h"
#include "ops/linear_attention/gated_delta_net/input_conv/execute.h"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/projected_conv.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q4_q5/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/workspace.h"
#include <stdexcept>

namespace ninfer::ops::detail::gdn_input {

void launch_projection(const Tensor& x, const WeightOperands& weights, ProjectedOutputs out,
                       ProjectionPlan plan, const Scratch& scratch, cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (weights.format) {
    case Format::Q4Q5:
        q4_q5_gdn_input_dispatch(x, w, *weights.second, out.qkv, out.z, stream);
        break;
    case Format::Q8:
        q8_gdn_input_dispatch(x, w, out.qkv, out.z, stream);
        break;
    case Format::Fp8:
        if (plan.compute == Compute::A8)
            launch_fp8_gdn_input_a8(x, w, out.qkv, out.z, scratch.fp8, stream);
        else if (x.ne[1] == 1)
            launch_fp8_gdn_input_decode(x, w, out.qkv, out.z, stream);
        else
            launch_fp8_gdn_input_matrix(x, w, out.qkv, out.z, stream);
        break;
    case Format::Nvfp4:
        if (plan.compute == Compute::A4)
            launch_nvfp4_gdn_input_a4(x, w, out.qkv, out.z, scratch.nvfp4, stream);
        else
            launch_nvfp4_gdn_input_a16(x, w, out.qkv, out.z, stream);
        break;
    }
}

void launch_snapshot(const Tensor& x, const WeightOperands& weights, const Tensor& conv,
                     ConstStateFieldView state, HistoryDestination destination, const Tensor& valid,
                     GdnConvOutputs out, Batch batch, ConvPlan plan, const Scratch& scratch,
                     cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (plan.kernel) {
    case ConvKernel::Q4Q5Fused:
        launch_q4_q5_gdn_input_conv_snapshot(x, w, *weights.second, conv, state, valid, destination,
                                             out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8Decode:
        launch_q8_gdn_input_decode_conv_snapshot(x, w, conv, state, valid, destination, out.query,
                                                 out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8SlicedK:
        launch_q8_gdn_input_splitk_conv_snapshot(x, w, conv, state, valid, destination, out.query,
                                                 out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8Window:
    case ConvKernel::Q8Aggregate:
    case ConvKernel::Q8TmaBatch:
#if NINFER_TARGET_SPARK
        if (weights.geometry.query_rows == 1024) {
            launch_q8_gdn_conv_n8192_k2560_snapshot(x, w, conv, state, valid, destination,
                                                    out.query, out.key, out.value, out.z,
                                                    plan.kernel, stream);
            return;
        }
#endif
        launch_q8_gdn_conv_n16384_k2560_snapshot(x, w, conv, state, valid, destination, out.query,
                                                 out.key, out.value, out.z, plan.kernel, stream);
        return;
    case ConvKernel::Fp8Fused:
        launch_fp8_gdn_snapshot_fused(x, w, conv, state, valid, destination, out.query, out.key,
                                      out.value, out.z, stream);
        return;
    case ConvKernel::Nvfp4Decode:
        launch_nvfp4_gdn_snapshot_decode(x, w, conv, state, valid, destination, out.query, out.key,
                                         out.value, out.z, stream);
        return;
    case ConvKernel::Nvfp4Window:
        launch_nvfp4_gdn_snapshot_small_t(x, w, conv, state, valid, destination, out.query, out.key,
                                          out.value, out.z, stream);
        return;
    case ConvKernel::Materialized:
        break;
    }
    const Tensor flat_x = x.view({weights.geometry.input_rows, batch.columns()});
    Tensor flat_z       = out.z.view({weights.geometry.value_rows, batch.columns()});
    launch_projection(flat_x, weights, {scratch.projected, flat_z}, plan.projection, scratch,
                      stream);
    const Tensor projected =
        scratch.projected.view({weights.geometry.channels(), batch.tokens, batch.rows});
    if (plan.post != ProjectedConvKernel::Sequential)
        launch_gdn_projected_conv_parallel_snapshot(projected, conv, state, valid, destination,
                                                    out.query, out.key, out.value, plan.post,
                                                    stream);
    else
        launch_gdn_projected_conv_snapshot(projected, conv, state, valid, destination, out.query,
                                           out.key, out.value, stream);
}

void launch_record(const Tensor& x, const WeightOperands& weights, const Tensor& conv,
                   ConstStateFieldView state, const Tensor& valid, GdnConvOutputs out,
                   Tensor& record, Batch batch, ConvPlan plan, const Scratch& scratch,
                   cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (plan.kernel) {
    case ConvKernel::Q4Q5Fused:
        launch_q4_q5_gdn_input_conv_record(x, w, *weights.second, conv, state, valid, record,
                                           out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8SlicedK:
        launch_q8_gdn_input_splitk_conv_record(x, w, conv, state, valid, record, out.query, out.key,
                                               out.value, out.z, stream);
        return;
    case ConvKernel::Q8Window:
    case ConvKernel::Q8Aggregate:
    case ConvKernel::Q8TmaBatch:
#if NINFER_TARGET_SPARK
        if (weights.geometry.query_rows == 1024) {
            launch_q8_gdn_conv_n8192_k2560_record(x, w, conv, state, valid, record, out.query,
                                                  out.key, out.value, out.z, plan.kernel, stream);
            return;
        }
#endif
        launch_q8_gdn_conv_n16384_k2560_record(x, w, conv, state, valid, record, out.query, out.key,
                                               out.value, out.z, plan.kernel, stream);
        return;
    case ConvKernel::Fp8Fused:
        launch_fp8_gdn_record_fused(x, w, conv, state, valid, record, out.query, out.key, out.value,
                                    out.z, stream);
        return;
    case ConvKernel::Nvfp4Window:
        launch_nvfp4_gdn_record_small_t(x, w, conv, state, valid, record, out.query, out.key,
                                        out.value, out.z, stream);
        return;
    case ConvKernel::Q8Decode:
    case ConvKernel::Nvfp4Decode:
        throw std::logic_error("GDN record selected a one-token kernel");
    case ConvKernel::Materialized:
        break;
    }
    const Tensor flat_x = x.view({weights.geometry.input_rows, batch.columns()});
    const ProjectedOutputs projected{record.view({weights.geometry.channels(), batch.columns()}),
                                     out.z.view({weights.geometry.value_rows, batch.columns()})};
    launch_projection(flat_x, weights, projected, plan.projection, scratch, stream);
    if (plan.post != ProjectedConvKernel::Sequential)
        launch_gdn_projected_conv_parallel_record(record, conv, state, valid, out.query, out.key,
                                                  out.value, plan.post, stream);
    else
        launch_gdn_projected_conv_record(record, conv, state, valid, out.query, out.key, out.value,
                                         stream);
}

} // namespace ninfer::ops::detail::gdn_input
