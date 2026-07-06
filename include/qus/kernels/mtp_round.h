#pragma once

// qus::kernels - fixed-shape MTP round state helpers.

#include "qus/core/tensor.h"
#include "qus/kernels/sampling.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void mtp_prepare_verify_inputs(const Tensor& token, const Tensor& drafts, const Tensor& length,
                               Tensor& window_base, Tensor& verify_ids, Tensor& positions,
                               cudaStream_t stream);

// Commits one MTP round. `logits` is the [vocab, k+1] verify-column logits and
// `config` is the device-resident sampler config: temperature <= 0 keeps the
// exact greedy argmax accept (using `target_tokens`), temperature > 0 runs
// distribution-correct rejection sampling over `logits`.
void mtp_accept_tokens(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
                       Tensor& length, Tensor& token, Tensor& sampled_out, Tensor& num_sampled,
                       Tensor& accepted, Tensor& ar_pos, Tensor& stats,
                       const SamplingConfig* config, cudaStream_t stream);

void mtp_prepare_shifted_ids(const Tensor& verify_ids, const Tensor& token,
                             const Tensor& accepted, Tensor& shifted_ids, cudaStream_t stream);

void mtp_gather_hidden_row(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                           cudaStream_t stream);

// Remap an in-place draft token id: draft_token holds a shortlist index in
// [0,n) produced by argmax over the draft head; replace it with the real vocab
// id via the device id_map. Out-of-range indices are left unchanged.
void mtp_remap_draft_token(Tensor& draft_token, const std::int32_t* id_map, std::int32_t n,
                           cudaStream_t stream);

void mtp_increment_i32(Tensor& scalar, cudaStream_t stream);

void mtp_count_fallback_step(Tensor& stats, cudaStream_t stream);

void mtp_reset_gdn_initial_slot(Tensor& gdn_initial_slot, cudaStream_t stream);

void mtp_set_gdn_initial_slot_from_accepted(const Tensor& accepted, Tensor& gdn_initial_slot,
                                            cudaStream_t stream);

} // namespace qus::kernels
