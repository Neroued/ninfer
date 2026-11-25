#pragma once

#include "core/record_view.h"
#include "ninfer/ops/projection_weights.h"
#include "ninfer/ops/ragged.h"

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
 * x BF16 [K,T,B], Q/K/V BF16 [N/3,T,B]. The projection accepts Q8-G32-FP16 RowSplit
 * with A16Only and FP32 convolution weights, or MXFP8 RowGroupScale with A16Only/AllowA8
 * and BF16 convolution weights. conv_weight [N,4] stores taps oldest to current.
 * Source/destination fields BF16 [N,3],
 * selected by independent Device I64 [B] byte offsets, feature-contiguous, 16-byte aligned.
 * T>0, B>0; valid_columns is empty (n=T) or Device I32 [B], 0<=n[b]<=T.
 * Outputs are zero after n[b]; zero-length rows read no State selectors/history and write
 * no history or records. Unselected fields are unchanged. For positive rows destination is the
 * final three represented projected columns, including old columns when n<3. Same-row exact
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
 * Packed ragged prefill form over S = ragged.num_seqs() non-empty segments and
 * N = ragged.total_tokens() columns. x is contiguous BF16 [K,N], Q/K/V are contiguous
 * BF16 [N_proj/3,N], and source/destination select S independent BF16 [N_proj,3] histories.
 * Each segment independently applies the exact formula and trailing-3 history update above,
 * with zero cross-segment convolution carry.
 */
void kda_input_proj_conv_update(const Tensor& x, const SingleProjectionWeight& qkv,
                                const Tensor& conv_weight, KdaConvUpdateState state,
                                const RaggedSequenceView& ragged, KdaConvOutputs outputs,
                                MemoryArena& workspace, CudaExecutionView execution);

/**
 * Approximate the projection, four-tap causal channelwise convolution and SiLU defined
 * above, with read-only source history. Each valid newest projected column is stored as BF16
 * in conv_record [N,T,B]. These words define tail restoration for accepted prefixes.
 * Invalid output tails are exact BF16 +0 and invalid record suffixes are unchanged.
 * Zero-length rows access neither State selectors nor history. Outputs and records are disjoint.
 */
void kda_input_proj_conv_record(const Tensor& x, const SingleProjectionWeight& qkv,
                                const Tensor& conv_weight, KdaConvSourceState state,
                                const Tensor& valid_columns, KdaConvOutputs outputs,
                                Tensor& conv_record, MemoryArena& workspace,
                                CudaExecutionView execution);

[[nodiscard]] std::size_t
kda_input_proj_conv_update_workspace_capacity_bytes(const LinearProfile& qkv, int batch,
                                                    int min_tokens, int max_tokens);
[[nodiscard]] std::size_t kda_input_proj_conv_update_ragged_workspace_capacity_bytes(
    const LinearProfile& qkv, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t min_segments, std::int32_t max_segments);
[[nodiscard]] std::size_t
kda_input_proj_conv_record_workspace_capacity_bytes(const LinearProfile& qkv, int batch,
                                                    int min_tokens, int max_tokens);
} // namespace ninfer::ops
