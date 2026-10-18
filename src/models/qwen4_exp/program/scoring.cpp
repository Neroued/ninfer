#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::score_next() {
    auto& f          = *compute;
    const auto& work = std::get<ScoreWork>(f.request.work.operation);
    if (f.score_at == work.targets.size()) {
        if (!compute_done.ready()) return;
        ScoreResult result;
        result.processed = work.row.ids.size();
        result.logprobs  = std::move(f.logprobs);
        result.timing    = f.timing;
        if (f.timer_started)
            result.timing.gpu_elapsed_ns +=
                static_cast<std::uint64_t>(double(prefill_timer.elapsed_ms()) * 1.0e6 + 0.5);
        collect_vision_timing(binding(work.row.member.residency),
                              work.row.main_frontier + result.processed);
        events.emplace_back(
            runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(std::move(result))});
        f.phase = ComputePhase::PendingCommit;
        if (resources.compute_actions->occupied()) resources.compute_actions->finish();
        return;
    }
    const auto n = std::min<std::size_t>(1024, work.targets.size() - f.score_at);
    auto scope   = resources.workspace.scope();
    auto buffers = execution::workspace::score(resources.workspace,
                                               parameters.profile->config.text.vocabulary, n);
    auto* host   = static_cast<TokenId*>(resources.compute_aux().data);
    std::copy_n(work.targets.begin() + f.score_at, n, host);
    copy_bytes_async({buffers.targets.data, n * sizeof(TokenId), cuda.device()},
                     {host, n * sizeof(TokenId), Device::cpu(), true}, cuda);
    execution::project_head(f.sample.slice(1, work.first_target_column + f.score_at, n),
                            parameters.text.head, buffers.logits, resources.workspace, cuda);
    ops::target_logprobs(buffers.logits, buffers.targets, parameters.profile->public_tokens,
                         buffers.logprobs, cuda);
    copy_bytes_async(resources.egress().subspan(0, n * sizeof(float)),
                     {buffers.logprobs.data, n * sizeof(float), cuda.device()}, cuda);
    f.score_count = n;
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitScore;
}

} // namespace ninfer::models::qwen4_exp::detail
