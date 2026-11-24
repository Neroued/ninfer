#pragma once

#include "core/record_view.h"
#include "ninfer/ops/projection_weights.h"

namespace ninfer::ops {
struct KdaConvOutputs {
    Tensor query, key, value;
};

struct KdaConvSourceState {
    ConstStateFieldView history;
};

struct KdaConvUpdateState {
    ConstStateFieldView source;
    StateFieldView destination;
};

/**
 * Project p=[Wq;Wk;Wv]*x, apply four-tap causal channelwise convolution to
 * [source.history || p], then SiLU and split Q/K/V. Projection and convolution form one
 * computation: private p has no BF16 rounding boundary. Published outputs/history are BF16.
 *
 * x BF16 [K,T,B], Q/K/V BF16 [N/3,T,B]; registered Q8-G32-FP16 RowSplit A16 parent [N,K].
 * conv_weight FP32 [N,4], taps oldest to current. Source/destination fields BF16 [N,3],
 * selected by independent Device I64 [B] byte offsets, feature-contiguous, 16-byte aligned.
 * T>0, B=1..8; valid_columns is empty (n=T) or Device I32 [B], 1<=n[b]<=T.
 * Outputs are zero after n[b]; unselected fields are unchanged. Destination is the final
 * three represented projected columns, including old columns when n<3. Same-row exact
 * in-place is allowed; destinations are distinct and cannot overwrite another row's source.
 * Other writes do not overlap accessed operands/metadata. Inputs and weights are preserved.
 *
 * Caller supplies the queried scoped workspace. All operands share execution.device().
 * Capture/replay reads current offsets/lengths; no hidden allocation or Host readback.
 */
void kda_input_proj_conv_update(const Tensor& x, const SingleProjectionWeight& qkv,
                                const Tensor& conv_weight, KdaConvUpdateState state,
                                const Tensor& valid_columns, KdaConvOutputs outputs,
                                MemoryArena& workspace, CudaExecutionView execution);

/**
 * Same physical computation and readouts as update, but source stays read-only and each
 * valid newest BF16 projected column is stored in conv_record [N,T,B]. Record suffixes remain
 * unchanged. These words define tail restoration for accepted prefixes of this physical call;
 * recomputing a shorter projection is not the reference. Outputs and records are disjoint.
 */
void kda_input_proj_conv_record(const Tensor& x, const SingleProjectionWeight& qkv,
                                const Tensor& conv_weight, KdaConvSourceState state,
                                const Tensor& valid_columns, KdaConvOutputs outputs,
                                Tensor& conv_record, MemoryArena& workspace,
                                CudaExecutionView execution);

[[nodiscard]] std::size_t
kda_input_proj_conv_update_workspace_capacity_bytes(const LinearProfile& qkv, int batch,
                                                    int min_tokens, int max_tokens);
[[nodiscard]] std::size_t
kda_input_proj_conv_record_workspace_capacity_bytes(const LinearProfile& qkv, int batch,
                                                    int min_tokens, int max_tokens);
} // namespace ninfer::ops
