#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: mtp_prepare_verify_inputs
 *
 * Math / indexing:
 *   window_base = length; verify_ids[0] = token; verify_ids[i+1] = drafts[i];
 *   positions[i] = length + i for 0 <= i <= K.
 *
 * Logical shapes:
 *   All tensors are contiguous I32. token/length/window_base are scalars, drafts is [K] with
 *   1<=K<=5, and verify_ids/positions are [K+1]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes window_base and the full verify_ids and positions vectors.
 *
 * Workspace:
 *   None. Only window_base, verify_ids, and positions are written.
 */
void mtp_prepare_verify_inputs(const Tensor& token, const Tensor& drafts, const Tensor& length,
                               Tensor& window_base, Tensor& verify_ids, Tensor& positions,
                               cudaStream_t stream);

/**
 * Op: mtp_accept_tokens
 *
 * Algorithm:
 *   Greedy mode accepts the longest draft prefix matching target_tokens and commits the target
 *   token at the first mismatch (or the bonus column). Sampling mode applies SamplingConfig to
 *   each verification column, accepts draft i with target probability p_i(draft_i), samples from
 *   the residual distribution on first rejection, and samples a bonus from column K when every
 *   draft is accepted. RNG domains are the MTP accept/resample/bonus SamplePurpose values and
 *   logical positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens and sampled_out are I32 [K+1], drafts is I32
 *   [K], logits is BF16 [physical_rows,C] with C>=K+1, length/token/num_sampled/accepted/ar_pos
 *   are I32 scalars, and stats is I64 with at least 4+min(K,5) elements. token_domain is in
 *   [1,physical_rows], with 1<=K<=5. config points to one device-resident SamplingConfig. Tensor
 *   arguments, config, and config->token_counts do not overlap except for the explicitly mutated
 *   objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   Let A be the accepted draft count and produced=A+1. sampled_out[0:A] receives accepted drafts,
 *   sampled_out[A] receives the correction/bonus token, and the remaining slots are zero.
 *   num_sampled=produced; accepted=A; token=sampled_out[A]; length'=length+produced;
 *   ar_pos'=length'. stats[0]+=K, stats[1]+=A, stats[2]+=1, and stats[4+i]+=1 for
 *   0<=i<min(A,5); stats[3] and all other entries are unchanged. In sampling mode, each produced
 *   token increments config->token_counts when that pointer is non-null. Greedy mode does not
 *   update token_counts. Inputs, logits, and drafts remain unchanged.
 *
 * Workspace:
 *   Caller-owned transient storage reported by sampling_workspace_bytes(token_domain,K+1).
 */
void mtp_accept_tokens(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
                       Tensor& length, Tensor& token, Tensor& sampled_out, Tensor& num_sampled,
                       Tensor& accepted, Tensor& ar_pos, Tensor& stats, std::int32_t token_domain,
                       const SamplingConfig* config, WorkspaceArena& workspace,
                       cudaStream_t stream);

/**
 * Op: mtp_prepare_shifted_ids
 *
 * Math / indexing:
 *   For T=K+1, shifted_ids[i]=verify_ids[i+1] for 0<=i<K, then
 *   shifted_ids[accepted]=token. shifted_ids[K] is written only when accepted=K; otherwise its
 *   previous value is retained.
 *
 * Logical shapes / effects:
 *   verify_ids and shifted_ids are distinct contiguous I32 [K+1], token and accepted are
 *   contiguous I32 scalars, and 0<=accepted<=K. No workspace or other state is used.
 */
void mtp_prepare_shifted_ids(const Tensor& verify_ids, const Tensor& token, const Tensor& accepted,
                             Tensor& shifted_ids, cudaStream_t stream);

/**
 * Op: mtp_gather_hidden_row
 *
 * Math / indexing:
 *   out[:,0] = hidden[:,accepted].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T], accepted is a contiguous I32 scalar in [0,T), and out is a
 *   distinct contiguous BF16 [D,1]. The Op exactly copies BF16 bits, writes all of out, and uses
 *   no workspace or other state.
 */
void mtp_gather_hidden_row(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                           cudaStream_t stream);

/**
 * Op: mtp_remap_draft_token
 *
 * Math / indexing:
 *   draft_token' = id_map[draft_token] for 0 <= draft_token < count.
 *
 * Effects:
 *   Updates the contiguous I32 draft_token scalar in place; id_map is a distinct device I32 array
 *   [count]. There is no workspace or other state side effect.
 */
void mtp_remap_draft_token(Tensor& draft_token, const std::int32_t* id_map, std::int32_t count,
                           cudaStream_t stream);

} // namespace ninfer::ops
