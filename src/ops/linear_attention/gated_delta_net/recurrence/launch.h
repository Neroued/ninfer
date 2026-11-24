#pragma once

#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "core/record_view.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net {
struct RaggedSegments {
    const int* cu_seqlens;
    const int* lengths;
    int count;
};

namespace chunked {
struct ChunkIndexEntry;
}

void launch_recurrent_ragged(const RecurrenceOperands& args, RaggedSegments segments,
                             bool normalize_qk, int recurrent_before,
                             chunked::ChunkIndexEntry* build_index, cudaStream_t stream);

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream, const std::int64_t* source_offset = nullptr,
                            const std::int64_t* destination_offset = nullptr);

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, const Tensor& valid_columns,
                                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
