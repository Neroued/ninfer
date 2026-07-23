#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& token, const Tensor& drafts,
                                              const Tensor& length, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream);

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, Tensor& length, Tensor& token,
                                             Tensor& sampled_out, Tensor& num_sampled,
                                             Tensor& accepted, Tensor& stats,
                                             std::int32_t token_domain,
                                             const SamplingConfig* config, DeviceSpan workspace,
                                             cudaStream_t stream);

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& accepted,
                                               Tensor& out, cudaStream_t stream);

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream);

} // namespace ninfer::ops::detail
