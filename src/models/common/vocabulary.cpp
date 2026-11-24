#include "models/common/vocabulary.h"

namespace ninfer::models {

Tensor DistributedVocabulary::collect(const Tensor& logits, const ops::SamplingConfig* configs,
                                      const Tensor& valid, const Tensor& drafts) const {
    const int t = logits.ne[1], b = logits.ne[2];
    Tensor keys(local, DType::I64, {candidates, t, b});
    Tensor combined(gathered, DType::I64, {candidates, t, b, static_cast<int>(peer.size())});
    ops::vocabulary_candidates(logits, begin, domain, configs, valid, drafts, keys, workspace,
                               cuda);
    peer.gather(keys, combined, cuda);
    return combined;
}

void DistributedVocabulary::sample(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& positions, int purpose,
                                   Tensor& ids) const {
    const auto combined = collect(logits, configs, valid, {});
    ops::sample_vocabulary_candidates(combined, configs, positions, purpose, ids, cuda);
}

void DistributedVocabulary::accept(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& drafts, const Tensor& extents,
                                   const Tensor& positions, Tensor& licensed,
                                   Tensor& counts) const {
    const auto combined = collect(logits, configs, valid, drafts);
    ops::accept_vocabulary_candidates(combined, configs, drafts, extents, positions, licensed,
                                      counts, cuda);
}

} // namespace ninfer::models
