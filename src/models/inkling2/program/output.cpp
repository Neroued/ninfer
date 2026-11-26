#include "models/inkling2/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/logit_softcap.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::inkling2::detail {

void RankProgramImpl::copy_result(std::int32_t rows) {
    const auto begin = frame.sampled_ids->region.offset;
    const auto bytes = std::size_t(rows) * sizeof(TokenId);
    copy_bytes_async(resources.egress().subspan(0, bytes), resources.fixed().subspan(begin, bytes),
                     cuda);
}

void RankProgramImpl::sample(std::int32_t b, SampleKind kind) {
    const auto& io    = frame;
    const auto& local = parameters.profile->local;
    auto logits       = resources.packed(*io.logits, {dimension(local.vocabulary.size), b});
    auto ids          = resources.packed(*io.sampled_ids, {b});
    auto positions    = resources.tensor(io.rows.sampling_positions).slice(0, 0, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    const int purpose =
        kind == SampleKind::FirstToken ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode;
    const auto live = row(io.rows.live_rows, b);
    ops::sample(logits, ids, parameters.profile->public_tokens, configs, positions, purpose, live,
                resources.workspace, cuda);
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
    const int n       = static_cast<int>(std::min<std::size_t>(1024, work.targets.size() - f.score_at));
    const auto& local = parameters.profile->local;
    auto scope        = resources.workspace.scope();
    auto logits       = resources.workspace.alloc(DType::BF16, {dimension(local.vocabulary.size), n});
    auto targets      = resources.workspace.alloc(DType::I32, {n});
    auto output       = resources.workspace.alloc(DType::FP32, {n});
    auto* input       = static_cast<TokenId*>(host(frame.main.ids).data);
    std::copy_n(work.targets.begin() + f.score_at, n, input);
    copy_bytes_async({targets.data, targets.bytes(), cuda.device()},
                     {input, targets.bytes(), Device::cpu(), true}, cuda);
    const auto hidden = f.hidden.slice(1, work.first_target_column + f.score_at, n);
    ops::linear(hidden, parameters.text.head.weight, logits, parameters.text.head.policy,
                resources.workspace, cuda);
    ops::logit_softcap(logits, parameters.profile->config.text.final_logit_softcapping, logits,
                       cuda);
    ops::target_logprobs(logits, targets, parameters.profile->public_tokens, output, cuda);
    copy_bytes_async(resources.egress().subspan(0, output.bytes()),
                     {output.data, output.bytes(), cuda.device()}, cuda);
    f.score_count = static_cast<std::uint32_t>(n);
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitScore;
}

} // namespace ninfer::models::inkling2::detail
