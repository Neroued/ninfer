#include "models/qwen4_exp/program/internal.h"

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::vocabulary_draft(const Tensor& logits, Tensor& ids) {
    const auto& io     = resources.plan.persistent.buffers;
    const auto batch   = static_cast<std::int32_t>(ids.numel());
    auto keys          = packed_tensor(*io.vocabulary_local, {1, 1, batch});
    const auto configs = device_tensor(*io.proposal_sampling);
    const auto valid   = device_tensor(*io.proposal_columns).slice(0, 0, batch);
    auto scope         = resources.workspace.scope();
    ops::vocabulary_candidates(
        logits.view({logits.ne[0], 1, batch}), parameters.profile->local.vocabulary_begin,
        parameters.profile->public_tokens, static_cast<const ops::SamplingConfig*>(configs.data),
        valid, {}, keys, resources.workspace, cuda);
    parallel->maximum(keys, cuda);
    auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    ops::sample_vocabulary_candidates(keys.view({1, 1, batch, 1}),
                                      static_cast<const ops::SamplingConfig*>(configs.data),
                                      positions, ops::kSamplePurposeDecode, ids, cuda);
}

DistributedVocabulary RankProgramImpl::vocabulary() {
    const auto& io = resources.plan.persistent.buffers;
    return {parameters.profile->local.vocabulary_begin,
            parameters.profile->public_tokens,
            io.vocabulary_local->region.bind(resources.persistent()),
            io.vocabulary_gathered->region.bind(resources.persistent()),
            *parallel,
            resources.workspace,
            cuda};
}

void RankProgramImpl::vocabulary_sample(std::int32_t batch, bool first_token) {
    const auto& io = resources.plan.persistent.buffers;
    auto logits =
        packed_tensor(*io.logits, {dimension(parameters.profile->local.vocabulary), 1, batch});
    auto configs         = device_tensor(*io.sampling);
    const auto* sampling = static_cast<const ops::SamplingConfig*>(configs.data);
    auto valid = device_tensor(first_token ? io.batch.accepted_columns : io.batch.valid_columns)
                     .slice(0, 0, batch);
    auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    auto ids       = packed_tensor(*io.sampled_ids, {batch});
    vocabulary().sample(logits, sampling, valid, positions,
                        first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode, ids);
}

void RankProgramImpl::vocabulary_accept(std::int32_t batch) {
    const auto& io  = resources.plan.persistent.buffers;
    const auto& mtp = *io.mtp;
    const int width = shape.draft_tokens + 1;
    auto logits =
        packed_tensor(*io.logits, {dimension(parameters.profile->local.vocabulary), width, batch});
    auto drafts        = packed_tensor(mtp.drafts, {width - 1, batch});
    const auto configs = device_tensor(*io.sampling);
    const auto row     = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto extents = row(mtp.extents), lengths = row(mtp.frontiers), anchors = row(mtp.anchors);
    auto licensed = packed_tensor(*io.sampled_ids, {width, batch});
    auto counts = row(*io.output_counts), accepted = row(*io.accepted_drafts);
    vocabulary().accept(logits, static_cast<const ops::SamplingConfig*>(configs.data),
                        row(io.batch.valid_columns), drafts, extents, lengths, anchors, licensed,
                        counts, accepted);
}

} // namespace ninfer::models::qwen4_exp::detail
