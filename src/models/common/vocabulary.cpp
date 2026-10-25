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
    if (peer.output_owner())
        ops::sample_vocabulary_candidates(combined, configs, positions, purpose, ids, cuda);
    peer.broadcast(ids, cuda);
}

void DistributedVocabulary::accept(const Tensor& logits, const ops::SamplingConfig* configs,
                                   const Tensor& valid, const Tensor& drafts, const Tensor& extents,
                                   Tensor& frontiers, Tensor& anchors, Tensor& licensed,
                                   Tensor& counts, Tensor& accepted) const {
    const auto combined = collect(logits, configs, valid, drafts);
    if (peer.output_owner())
        ops::accept_vocabulary_candidates(combined, configs, drafts, extents, frontiers, anchors,
                                          licensed, counts, accepted, cuda);
    peer.broadcast(licensed, cuda);
    peer.broadcast(counts, cuda);
    peer.broadcast(accepted, cuda);
}

} // namespace ninfer::models
