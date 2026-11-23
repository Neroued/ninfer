#include "models/glm5next/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

void RankProgramImpl::copy_result(int rows, bool verify, SampleKind kind) {
    const auto& io = frame;
    if (verify)
        copy_bytes_async(resources.egress().subspan(0, io.result_egress.bytes),
                         io.result_egress.bind(resources.fixed()), cuda);
    else {
        const auto begin = io.sampled_ids->region.offset;
        auto bytes       = std::size_t(rows) * sizeof(TokenId);
        if (kind == SampleKind::NextToken && io.top1_probabilities) {
            const auto& last = io.top1_probabilities->region;
            bytes            = last.offset + last.bytes - begin;
        }
        copy_bytes_async(resources.egress().subspan(0, bytes),
                         resources.fixed().subspan(begin, bytes), cuda);
    }
}

DistributedVocabulary RankProgramImpl::vocabulary() {
    const auto& io = frame;
    return {parameters.profile->local.vocabulary.begin,
            parameters.profile->public_tokens,
            io.candidates->region.bind(resources.fixed()),
            io.gathered_candidates->region.bind(resources.fixed()),
            *parallel,
            resources.workspace,
            cuda};
}

void RankProgramImpl::sample(int b, SampleKind kind) {
    const auto& io    = frame;
    const auto& local = parameters.profile->local;
    auto logits       = resources.packed(*io.logits, {dimension(local.vocabulary.size), 1, b});
    auto ids          = resources.packed(*io.sampled_ids, {b});
    auto positions    = resources.tensor(io.rows.sampling_positions).slice(0, 0, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    const int purpose =
        kind == SampleKind::FirstToken ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode;
    const auto live  = row(io.rows.live_rows, b);
    const auto valid = row(io.rows.sampling_columns, b);
    if (kind == SampleKind::NextToken && io.top1_probabilities) {
        auto probabilities = resources.packed(*io.top1_probabilities, {b});
        if (parallel)
            vocabulary().sample(logits, configs, valid, live, positions, purpose, ids,
                                probabilities);
        else
            ops::sample(logits.view({logits.ne[0], b}), ids, probabilities,
                        parameters.profile->public_tokens, configs, positions, purpose, live,
                        resources.workspace, cuda);
    } else if (parallel) {
        vocabulary().sample(logits, configs, valid, live, positions, purpose, ids);
    } else {
        ops::sample(logits.view({logits.ne[0], b}), ids, parameters.profile->public_tokens, configs,
                    positions, purpose, live, resources.workspace, cuda);
    }
}

void RankProgramImpl::score() {
    auto& f          = *compute;
    const auto& work = std::get<ScoreWork>(f.request.work.operation);
    if (f.score_at == work.targets.size()) {
        timer.observe_result(f.gpu);
        if (resources.compute_actions->occupied()) resources.compute_actions->finish();
        events.emplace_back(runtime::ResultReady<ExecutionTypes>{
            f.request.ticket, Result(ScoreResult{static_cast<std::uint32_t>(work.row.ids.size()),
                                                 std::move(f.logprobs)})});
        f.phase = ComputePhase::PendingCommit;
        return;
    }
    const int n       = std::min<std::size_t>(1024, work.targets.size() - f.score_at);
    const auto& local = parameters.profile->local;
    auto scope        = resources.workspace.scope();
    auto logits  = resources.workspace.alloc(DType::BF16, {dimension(local.vocabulary.size), n});
    auto targets = resources.workspace.alloc(DType::I32, {n});
    auto output  = resources.workspace.alloc(DType::FP32, {n});
    auto* input  = static_cast<TokenId*>(host(frame.main.ids).data);
    std::copy_n(work.targets.begin() + f.score_at, n, input);
    copy_bytes_async({targets.data, targets.bytes(), cuda.device()},
                     {input, targets.bytes(), Device::cpu(), true}, cuda);
    const auto hidden = f.hidden.slice(1, work.first_target_column + f.score_at, n);
    ops::linear(hidden, parameters.text.head.weight, logits, parameters.text.head.policy,
                resources.workspace, cuda);
    if (parallel) {
        auto stats    = resources.workspace.alloc(DType::FP32, {3, n});
        auto gathered = resources.workspace.alloc(DType::FP32, {3, n, dimension(local.size)});
        ops::vocabulary_logprob_stats(logits, targets, local.vocabulary.begin,
                                      parameters.profile->public_tokens, stats, resources.workspace,
                                      cuda);
        parallel->gather(stats, gathered, cuda);
        ops::target_logprobs_from_stats(gathered, output, cuda);
    } else
        ops::target_logprobs(logits, targets, parameters.profile->public_tokens, output, cuda);
    copy_bytes_async(resources.egress().subspan(0, output.bytes()),
                     {output.data, output.bytes(), cuda.device()}, cuda);
    f.score_count = n;
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitScore;
}

} // namespace ninfer::models::glm5next::detail
