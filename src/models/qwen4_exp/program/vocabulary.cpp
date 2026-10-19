#include "models/qwen4_exp/program/internal.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::vocabulary_draft(const Tensor& logits, Tensor& ids, const Tensor& valid) {
    const auto& io     = resources.plan.persistent.buffers;
    const auto batch   = static_cast<std::int32_t>(ids.numel());
    auto keys          = packed_tensor(*io.vocabulary_local, {1, 1, batch});
    const auto configs = device_tensor(*io.proposal_sampling);
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

void RankProgramImpl::vocabulary_sample(std::int32_t batch, bool first_token, std::int32_t k) {
    const auto& io       = resources.plan.persistent.buffers;
    const auto& local    = parameters.profile->local;
    const bool greedy    = k == 1;
    auto logits          = packed_tensor(*io.logits, {dimension(local.vocabulary), 1, batch});
    auto keys            = packed_tensor(*io.vocabulary_local, {k, 1, batch});
    auto configs         = device_tensor(*io.sampling);
    const auto* sampling = static_cast<const ops::SamplingConfig*>(configs.data);
    auto valid = device_tensor(first_token ? io.batch.accepted_columns : io.batch.valid_columns)
                     .slice(0, 0, batch);
    auto scope = resources.workspace.scope();
    ops::vocabulary_candidates(logits, local.vocabulary_begin, parameters.profile->public_tokens,
                               sampling, valid, {}, keys, resources.workspace, cuda);
    Tensor combined;
    if (greedy) {
        parallel->maximum(keys, cuda);
        combined = keys.view({1, 1, batch, 1});
    } else {
        combined = packed_tensor(*io.vocabulary_gathered, {k, 1, batch, dimension(local.size)});
        parallel->gather(keys, combined, cuda);
    }
    auto ids = packed_tensor(*io.sampled_ids, {batch});
    if (greedy || parallel->output_owner()) {
        auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
        ops::sample_vocabulary_candidates(
            combined, sampling, positions,
            first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode, ids, cuda);
    }
    if (!greedy) parallel->broadcast(ids, cuda);
}

void RankProgramImpl::vocabulary_accept(std::int32_t batch, std::int32_t k) {
    const auto& io       = resources.plan.persistent.buffers;
    const auto& mtp      = *io.mtp;
    const auto& local    = parameters.profile->local;
    const auto width     = static_cast<int>(shape.draft_tokens + 1);
    const bool greedy    = k == 1;
    auto logits          = packed_tensor(*io.logits, {dimension(local.vocabulary), width, batch});
    auto keys            = packed_tensor(*io.vocabulary_local, {k, width, batch});
    auto drafts          = packed_tensor(mtp.drafts, {width - 1, batch});
    auto configs         = device_tensor(*io.sampling);
    const auto* sampling = static_cast<const ops::SamplingConfig*>(configs.data);
    auto valid           = device_tensor(io.batch.valid_columns).slice(0, 0, batch);
    auto scope           = resources.workspace.scope();
    ops::vocabulary_candidates(logits, local.vocabulary_begin, parameters.profile->public_tokens,
                               sampling, valid, drafts, keys, resources.workspace, cuda);
    Tensor combined;
    if (greedy) {
        parallel->maximum(keys, cuda);
        combined = keys.view({1, width, batch, 1});
    } else {
        combined = packed_tensor(*io.vocabulary_gathered, {k, width, batch, dimension(local.size)});
        parallel->gather(keys, combined, cuda);
    }
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto extents = row(mtp.extents), lengths = row(mtp.frontiers), anchors = row(mtp.anchors);
    auto licensed = packed_tensor(*io.sampled_ids, {width, batch});
    auto counts = row(*io.output_counts), accepted = row(*io.accepted_drafts);
    if (greedy || parallel->output_owner())
        ops::accept_vocabulary_candidates(combined, sampling, drafts, extents, lengths, anchors,
                                          licensed, counts, accepted, cuda);
    if (!greedy) {
        parallel->broadcast(licensed, cuda);
        parallel->broadcast(counts, cuda);
        parallel->broadcast(accepted, cuda);
    }
}

} // namespace ninfer::models::qwen4_exp::detail
