#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mtp_prepare_verify_inputs_launch(const Tensor& token, const Tensor& drafts,
                                      const Tensor& length, Tensor& window_base, Tensor& verify_ids,
                                      Tensor& positions, cudaStream_t stream);

void mtp_accept_tokens_launch(const Tensor& target_tokens, const Tensor& logits,
                              const Tensor& drafts, Tensor& length, Tensor& token,
                              Tensor& sampled_out, Tensor& num_sampled, Tensor& accepted,
                              Tensor& ar_pos, Tensor& stats, std::int32_t token_domain,
                              const SamplingConfig* config, DeviceSpan workspace,
                              cudaStream_t stream);

void mtp_prepare_shifted_ids_launch(const Tensor& verify_ids, const Tensor& token,
                                    const Tensor& accepted, Tensor& shifted_ids,
                                    cudaStream_t stream);

void mtp_gather_hidden_row_launch(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                                  cudaStream_t stream);

void mtp_remap_draft_token_launch(Tensor& draft_token, const std::int32_t* id_map, std::int32_t n,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
