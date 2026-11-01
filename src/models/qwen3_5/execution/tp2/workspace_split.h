#pragma once

// Two-device (tp 2) workspace recipes, next to the single-device ones of execution/workspace.h.
// Our file: the upstream header carries no hook.

#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/tp2/argmax.h"
#include "ninfer/ops/tp2/linear_topk.h"

#include <cstdint>

namespace ninfer::models::qwen3_5::execution::workspace {

// Tensor-parallel (tp == 2) calls run the workspace.h recipes on each rank's own arena with the
// rank's share of the config (execution::shard_text_config): the attention, GDN, FFN and vocabulary
// extents are halved and the hidden/residual extent is not. The recipes below are tp-only.

// The all-reduce staging of one call: every row-parallel projection of every layer reuses it, one
// per rank, because each all-reduce leaves its source and staging free on return.
struct TensorParallelCallRoots {
    Tensor staging;
};

template <class Allocator>
TensorParallelCallRoots tp_call_roots(Allocator& allocator, const TextConfig& config,
                                      std::int32_t tokens) {
    return {matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens)};
}

// The vocabulary-split logits of `columns` final hidden columns on one rank: this rank's rows of
// the output head and, on rank 0 when it gathers the complete logits alone, the staging that
// receives rank 1's `peer_rows` before they are interleaved into the caller's [V, columns] logits.
struct TensorParallelLogitsRoots {
    Tensor partial;
    Tensor staging;
};

template <class Allocator>
TensorParallelLogitsRoots tp_logits(Allocator& allocator, std::int32_t shard_rows,
                                    std::int32_t peer_rows, std::int32_t columns,
                                    bool gather_staging) {
    TensorParallelLogitsRoots out;
    out.partial = matrix(allocator, DType::BF16, shard_rows, columns);
    if (gather_staging) { out.staging = matrix(allocator, DType::BF16, peer_rows, columns); }
    return out;
}

// The vocabulary-split argmax of the optimized MTP proposal head over `columns` hidden columns on
// one rank: this rank's `shard_rows` logits, their argmax, and the packed candidates with the
// all-reduce staging they are exchanged through (ops::argmax_split_pack).
struct TensorParallelProposalRoots {
    Tensor partial;
    Tensor local;
    Tensor candidates;
    Tensor staging;
};

template <class Allocator>
TensorParallelProposalRoots tp_proposal_argmax(Allocator& allocator, std::int32_t shard_rows,
                                               std::int32_t columns) {
    TensorParallelProposalRoots out;
    out.partial    = matrix(allocator, DType::BF16, shard_rows, columns);
    out.local      = vector(allocator, DType::I32, columns);
    out.candidates = matrix(allocator, DType::BF16, ops::kArgmaxSplitCandidateRows, columns);
    out.staging    = matrix(allocator, DType::BF16, ops::kArgmaxSplitCandidateRows, columns);
    return out;
}

// The vocabulary-split DFlash2 candidate ranking over `columns` final drafter hidden columns on one
// rank (execution::dflash2_candidates_split): with `broadcast`, the hidden rank 1 receives from
// rank 0 and the all-reduce staging it arrives through (rank 1's `hidden` is its copy, rank 0's is
// unused); then the rank's top sixteen, its packed candidates and their all-reduce staging.
struct TensorParallelCandidateRoots {
    Tensor hidden;
    Tensor hidden_staging;
    Tensor ids;
    Tensor scores;
    Tensor candidates;
    Tensor staging;
};

template <class Allocator>
TensorParallelCandidateRoots tp_dflash2_candidates(Allocator& allocator, std::int32_t hidden,
                                                   std::int32_t columns, bool broadcast) {
    constexpr std::int32_t top_k = 16;
    TensorParallelCandidateRoots out;
    if (broadcast) {
        out.hidden         = matrix(allocator, DType::BF16, hidden, columns);
        out.hidden_staging = matrix(allocator, DType::BF16, hidden, columns);
    }
    out.ids        = matrix(allocator, DType::I32, top_k, columns);
    out.scores     = matrix(allocator, DType::FP32, top_k, columns);
    out.candidates = matrix(allocator, DType::BF16, ops::kTopKSplitCandidateRows, columns);
    out.staging    = matrix(allocator, DType::BF16, ops::kTopKSplitCandidateRows, columns);
    return out;
}

// One rank's roots of the two-device MTP stem. Rank 0 contracts the normalized token-embedding
// half of the packed input projection and rank 1 the normalized hidden half, so each rank
// normalizes only its own half into `normalized_input` and the packed input is never formed.
// Only rank 0 embeds tokens.
struct MtpStemSplitRoots {
    Tensor embedding;
    Tensor normalized_input;
    Tensor residual;
    Tensor attention_hidden;
};

template <class Allocator>
MtpStemSplitRoots mtp_stem_split(Allocator& allocator, const TextConfig& config,
                                 std::int32_t tokens, bool allocate_embedding) {
    MtpStemSplitRoots out;
    if (allocate_embedding) {
        out.embedding = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    }
    out.normalized_input = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.residual         = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.attention_hidden = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    return out;
}

} // namespace ninfer::models::qwen3_5::execution::workspace
