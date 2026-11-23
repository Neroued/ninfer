#include "models/common/vocabulary.h"

namespace ninfer::models {

template <bool RelativeLse>
Tensor DistributedVocabulary::collect(const Tensor& logits, const ops::SamplingConfig* configs,
                                      const Tensor& valid, const Tensor& drafts) const {
    const int t = logits.ne[1], b = logits.ne[2];
    constexpr int width = RelativeLse ? candidates_with_lse : candidates;
    Tensor keys(local, DType::I64, {width, t, b});
    Tensor combined(gathered, DType::I64, {width, t, b, static_cast<int>(peer.size())});
    if constexpr (RelativeLse)
        ops::vocabulary_candidates_with_lse(logits, begin, domain, configs, valid, drafts, keys,
                                            workspace, cuda);
    else
        ops::vocabulary_candidates(logits, begin, domain, configs, valid, drafts, keys, workspace,
                                   cuda);
    if constexpr (RelativeLse)
        peer.gather_unsigned_words(keys, combined, cuda);
    else
        peer.gather(keys, combined, cuda);
    return combined;
}

void DistributedVocabulary::sample(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& live_rows,
                                   const Tensor& positions, int purpose, Tensor& ids,
                                   Tensor& top1_probabilities) const {
    const auto combined = collect<true>(logits, configs, valid, {});
    ops::sample_vocabulary_candidates(combined, configs, positions, purpose, ids,
                                      top1_probabilities, live_rows, cuda);
}

void DistributedVocabulary::sample(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& live_rows,
                                   const Tensor& positions, int purpose, Tensor& ids) const {
    const auto combined = collect(logits, configs, valid, {});
    ops::sample_vocabulary_candidates(combined, configs, positions, purpose, ids, live_rows, cuda);
}

void DistributedVocabulary::accept(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& live_rows,
                                   const Tensor& drafts, const Tensor& extents,
                                   const Tensor& positions, Tensor& licensed,
                                   Tensor& counts) const {
    const auto combined = collect(logits, configs, valid, drafts);
    ops::accept_vocabulary_candidates(combined, configs, drafts, extents, positions, licensed,
                                      counts, live_rows, cuda);
}

void DistributedVocabulary::accept_sparse(const Tensor& logits, const ops::SamplingConfig* configs,
                                          const Tensor& valid, const Tensor& live_rows,
                                          const Tensor& drafts, const Tensor& candidate_ids,
                                          const Tensor& q, const Tensor& extents,
                                          const Tensor& positions, Tensor& licensed,
                                          Tensor& counts) const {
    const auto combined = collect(logits, configs, valid, drafts);
    ops::accept_vocabulary_candidates_sparse(combined, configs, drafts, candidate_ids, q, extents,
                                             positions, licensed, counts, live_rows, cuda);
}

} // namespace ninfer::models
