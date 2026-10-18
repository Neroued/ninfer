#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::launch_score(ComputeOperation& operation) {
    const auto& work      = std::get<ScoreWork>(operation.request.work.operation);
    auto& frame           = resources.frame;
    operation.score_count = std::min<std::uint32_t>(
        frame.score_targets.ne[0],
        static_cast<std::uint32_t>(work.targets.size()) - operation.score_cursor);
    if (!operation.score_count) {
        result_ready.record(device.stream());
        operation.stage = ComputeStage::Readback;
        return;
    }
    const int count = static_cast<int>(operation.score_count);
    // The History/readback fence retires the prior use of this pinned region.
    auto* host_targets = static_cast<std::int32_t*>(frame.host_input_ids.data);
    std::copy_n(work.targets.begin() + operation.score_cursor, count, host_targets);
    Tensor targets = frame.score_targets.slice(0, 0, count);
    CUDA_CHECK(cudaMemcpyAsync(targets.data, host_targets, targets.bytes(), cudaMemcpyHostToDevice,
                               device.stream()));
    resources.work.reset();
    Tensor hidden =
        frame.input_view(static_cast<int>(operation.width), 1)
            .hidden.slice(1, static_cast<int>(work.first_column + operation.score_cursor), count)
            .view({frame.spec.hidden, count});
    Tensor logits = resources.work.alloc(DType::BF16, {frame.spec.output_rows, count});
    execution::project(hidden, parameters.text.output_head, logits, resources.work,
                       device.execution_view());
    Tensor scores = frame.score_logprobs.slice(0, 0, count);
    ops::target_logprobs(logits, targets, execution::dimension(parameters.profile->public_tokens),
                         scores, device.execution_view());
    CUDA_CHECK(cudaMemcpyAsync(frame.host_score_logprobs.data, scores.data, scores.bytes(),
                               cudaMemcpyDeviceToHost, device.stream()));
    result_ready.record(device.stream());
    operation.stage = ComputeStage::Readback;
}

} // namespace ninfer::models::qwen3_5::detail
