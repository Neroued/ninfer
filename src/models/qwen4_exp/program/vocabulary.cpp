#include "models/qwen4_exp/program/internal.h"

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::vocabulary_draft(const Tensor& logits, const Tensor& valid, Tensor& ids,
                                       Tensor& confidences) {
    const auto& io   = resources.buffers();
    const auto batch = static_cast<std::int32_t>(ids.numel());
    auto local       = packed_tensor(*io.vocabulary_local, {2, 1, batch});
    auto gathered =
        packed_tensor(*io.vocabulary_gathered, {2, 1, batch, dimension(parallel->size())});
    const auto configs = device_tensor(*io.proposal_sampling);
    auto scope         = resources.workspace.scope();
    ops::vocabulary_candidates_with_lse(
        logits.view({logits.ne[0], 1, batch}), parameters.profile->local.vocabulary_begin,
        parameters.profile->public_tokens, static_cast<const ops::SamplingConfig*>(configs.data),
        valid, {}, local, resources.workspace, cuda);
    parallel->gather_unsigned_words(local, gathered, cuda);
    auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    ops::sample_vocabulary_candidates(
        gathered, static_cast<const ops::SamplingConfig*>(configs.data), positions,
        ops::kSamplePurposeDecode, ids, confidences, valid, cuda);
}

DistributedVocabulary RankProgramImpl::vocabulary() {
    const auto& io = resources.buffers();
    return {parameters.profile->local.vocabulary_begin,
            parameters.profile->public_tokens,
            io.vocabulary_local->region.bind(resources.persistent()),
            io.vocabulary_gathered->region.bind(resources.persistent()),
            *parallel,
            resources.workspace,
            cuda};
}

void RankProgramImpl::vocabulary_sample(std::int32_t batch, bool first_token) {
    const auto& io = resources.buffers();
    auto logits =
        packed_tensor(*io.logits, {dimension(parameters.profile->local.vocabulary), 1, batch});
    auto configs         = device_tensor(*io.sampling);
    const auto* sampling = static_cast<const ops::SamplingConfig*>(configs.data);
    auto live            = device_tensor(io.batch.live_rows).slice(0, 0, batch);
    auto positions       = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    auto ids             = packed_tensor(*io.sampled_ids, {batch});
    if (!first_token && io.top1_probabilities) {
        auto probabilities = packed_tensor(*io.top1_probabilities, {batch});
        vocabulary().sample(logits, sampling, live, live, positions, ops::kSamplePurposeDecode, ids,
                            probabilities);
    } else {
        vocabulary().sample(logits, sampling, live, live, positions,
                            first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode,
                            ids);
    }
}

void RankProgramImpl::vocabulary_accept(std::int32_t batch, std::int32_t width) {
    const auto& io  = resources.buffers();
    const auto& mtp = *io.mtp;
    auto logits =
        packed_tensor(*io.logits, {dimension(parameters.profile->local.vocabulary), width, batch});
    auto drafts        = draft_proposals(batch);
    const auto configs = device_tensor(*io.sampling);
    const auto row     = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto extents = row(mtp.extents), positions = row(io.batch.sampling_positions);
    auto licensed = packed_tensor(*io.sampled_ids, {width, batch});
    auto counts   = row(*io.output_counts);
    vocabulary().accept(logits, static_cast<const ops::SamplingConfig*>(configs.data),
                        row(io.batch.valid_columns), row(io.batch.live_rows), drafts, extents,
                        positions, licensed, counts);
}

} // namespace ninfer::models::qwen4_exp::detail
