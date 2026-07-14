#pragma once

#include "core/tensor.h"
#include "kernels/sampling/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::kernels {

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);
void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);
void mtp_prepare_verify_inputs(const Tensor& token, const Tensor& drafts, const Tensor& length,
                               Tensor& window_base, Tensor& verify_ids, Tensor& positions,
                               cudaStream_t stream);
void mtp_accept_tokens(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
                       Tensor& length, Tensor& token, Tensor& sampled_out, Tensor& num_sampled,
                       Tensor& accepted, Tensor& ar_pos, Tensor& stats, std::int32_t token_domain,
                       const SamplingConfig* config, cudaStream_t stream);
void mtp_prepare_shifted_ids(const Tensor& verify_ids, const Tensor& token, const Tensor& accepted,
                             Tensor& shifted_ids, cudaStream_t stream);
void mtp_gather_hidden_row(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                           cudaStream_t stream);
void mtp_remap_draft_token(Tensor& draft_token, const std::int32_t* id_map, std::int32_t count,
                           cudaStream_t stream);
void mtp_increment_i32(Tensor& scalar, cudaStream_t stream);
void mtp_count_fallback_step(Tensor& stats, cudaStream_t stream);
void mtp_reset_gdn_initial_slot(Tensor& gdn_initial_slot, cudaStream_t stream);
void mtp_set_gdn_initial_slot_from_accepted(const Tensor& accepted, Tensor& gdn_initial_slot,
                                            cudaStream_t stream);

} // namespace ninfer::kernels
