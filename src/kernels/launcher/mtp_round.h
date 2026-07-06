#pragma once

#include "qus/core/tensor.h"
#include "qus/kernels/sampling.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void mtp_prepare_verify_inputs_launch(const Tensor& token, const Tensor& drafts,
                                      const Tensor& length, Tensor& window_base,
                                      Tensor& verify_ids, Tensor& positions,
                                      cudaStream_t stream);

void mtp_accept_tokens_launch(const Tensor& target_tokens, const Tensor& logits,
                              const Tensor& drafts, Tensor& length, Tensor& token,
                              Tensor& sampled_out, Tensor& num_sampled, Tensor& accepted,
                              Tensor& ar_pos, Tensor& stats, const SamplingConfig* config,
                              cudaStream_t stream);

void mtp_prepare_shifted_ids_launch(const Tensor& verify_ids, const Tensor& token,
                                    const Tensor& accepted, Tensor& shifted_ids,
                                    cudaStream_t stream);

void mtp_gather_hidden_row_launch(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                                  cudaStream_t stream);

void mtp_remap_draft_token_launch(Tensor& draft_token, const std::int32_t* id_map, std::int32_t n,
                                  cudaStream_t stream);

void mtp_increment_i32_launch(Tensor& scalar, cudaStream_t stream);

void mtp_count_fallback_step_launch(Tensor& stats, cudaStream_t stream);

void mtp_reset_gdn_initial_slot_launch(Tensor& gdn_initial_slot, cudaStream_t stream);

void mtp_set_gdn_initial_slot_from_accepted_launch(const Tensor& accepted, Tensor& gdn_initial_slot,
                                                   cudaStream_t stream);

} // namespace qus::kernels::detail
