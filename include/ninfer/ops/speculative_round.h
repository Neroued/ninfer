#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   verify_ids[0] = token; verify_ids[i+1] = drafts[i];
 *   positions[i] = length + i for 0 <= i <= K.
 *
 * Logical shapes:
 *   All tensors are contiguous I32. token/length are scalars, drafts is [K] with K>=1, and
 *   verify_ids/positions are [K+1]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes the full verify_ids and positions vectors. Inputs remain unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& token, const Tensor& drafts,
                                       const Tensor& length, Tensor& verify_ids, Tensor& positions,
                                       cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Greedy mode accepts the longest draft prefix matching target_tokens and commits the target
 *   token at the first mismatch (or the bonus column). Sampling mode applies SamplingConfig to
 *   each verification column, accepts draft i with target probability p_i(draft_i), samples from
 *   the residual distribution on first rejection, and samples a bonus from column K when every
 *   draft is accepted. The draft proposal distribution is one-hot at each greedy draft token.
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens and sampled_out are I32 [K+1], drafts is I32
 *   [K], logits is BF16 [physical_rows,C] with C>=K+1, and
 *   length/token/num_sampled/accepted are I32 scalars. stats is I64 with at least 4+K elements.
 *   token_domain is in [1,physical_rows], K>=1, and config points to one device-resident
 *   SamplingConfig. Tensor arguments, config, and config->token_counts do not overlap except for
 *   the explicitly mutated objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   Let A be the accepted draft count and produced=A+1. sampled_out[0:A] receives accepted drafts,
 *   sampled_out[A] receives the correction/bonus token, and the remaining slots are zero.
 *   num_sampled=produced; accepted=A; token=sampled_out[A]; length'=length+produced;
 *   stats[0]+=K, stats[1]+=A, stats[2]+=1, and stats[4+i]+=1 for 0<=i<A; stats[3] and all other
 *   entries are unchanged. In sampling mode, each produced
 *   token increments config->token_counts when that pointer is non-null. Greedy mode does not
 *   update token_counts. Inputs, logits, and drafts remain unchanged.
 *
 * Workspace:
 *   Caller-owned transient storage reported by sampling_workspace_bytes(token_domain,K+1).
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, Tensor& length, Tensor& token,
                                      Tensor& sampled_out, Tensor& num_sampled, Tensor& accepted,
                                      Tensor& stats, std::int32_t token_domain,
                                      const SamplingConfig* config, WorkspaceArena& workspace,
                                      cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,0] = hidden[:,accepted].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T], accepted is a contiguous I32 scalar in [0,T), and out is a
 *   distinct contiguous BF16 [D,1]. The Op exactly copies BF16 bits, writes all of out, and uses
 *   no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

} // namespace ninfer::ops
