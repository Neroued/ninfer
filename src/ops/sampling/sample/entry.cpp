#include "ninfer/ops/sampling.h"

#include "ops/sampling/sample/plan.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::size_t sample_capacity(int token_domain, int min_lanes, int max_lanes, bool probability) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    return sample_detail::capacity(token_domain, min_lanes, max_lanes, probability);
}

void sample_impl(const Tensor& logits, Tensor& out, Tensor* probabilities, int token_domain,
                 const SamplingConfig* configs, const Tensor& logical_positions, int purpose,
                 const Tensor& live_rows, MemoryArena& workspace, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("sample: workspace belongs to another device");
    }
    if (logits.device != execution.device() || out.device != execution.device() ||
        logical_positions.device != execution.device()) {
        throw std::invalid_argument("sample: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [B]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) { throw std::invalid_argument("sample: out shape must be [B]"); }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { throw std::invalid_argument("sample: B must be positive"); }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out/logical_positions must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr) {
        throw std::invalid_argument("sample: tensor data must be non-null");
    }
    if (configs == nullptr) { throw std::invalid_argument("sample: configs must be non-null"); }
    if (live_rows.data) {
        common::operand::tensor(live_rows, DType::I32, {logits.ne[1]}, 4, "sample", "live rows",
                                execution.device());
        common::operand::disjoint(common::operand::range(live_rows, "sample"),
                                  common::operand::range(out, "sample"), "sample");
    }

    auto scratch_scope = workspace.scope();
    const auto plan    = sample_detail::select_plan(token_domain, logits.ne[1], probabilities);
    const MemorySpan scratch = plan.bytes == 0 ? MemorySpan{} : workspace.alloc_bytes(plan.bytes);
    if (live_rows.data && scratch.data)
        common::operand::disjoint(common::operand::range(live_rows, "sample"),
                                  common::operand::range(scratch.data, scratch.bytes, "sample"),
                                  "sample");
    if (probabilities) {
        using namespace common::operand;
        constexpr const char* op = "sample with probability";
        tensor(*probabilities, DType::FP32, {logits.ne[1]}, 4, op, "top1 probabilities",
               execution.device());
        const auto destination = range(*probabilities, op);
        for (const auto* input :
             {&logits, static_cast<const Tensor*>(&out), &logical_positions, &live_rows})
            if (input->data) disjoint(destination, range(*input, op), op);
        disjoint(destination, range(configs, sizeof(SamplingConfig) * logits.ne[1], op), op);
        if (scratch.data) disjoint(destination, range(scratch.data, scratch.bytes, op), op);
        // Count and mask pointers are Device metadata. Their disjointness is a caller contract.
    }
    sample_detail::launch(
        logits, out, probabilities ? static_cast<float*>(probabilities->data) : nullptr,
        token_domain, configs, logical_positions, purpose, live_rows, plan, scratch, stream);
}
} // namespace

std::size_t sampling_workspace_capacity_bytes(int token_domain, int min_lanes, int max_lanes) {
    return sample_capacity(token_domain, min_lanes, max_lanes, false);
}

std::size_t sampling_with_probability_workspace_capacity_bytes(int token_domain, int min_lanes,
                                                               int max_lanes) {
    return sample_capacity(token_domain, min_lanes, max_lanes, true);
}

void sample(const Tensor& logits, Tensor& out, int token_domain, const SamplingConfig* configs,
            const Tensor& logical_positions, int purpose, const Tensor& live_rows,
            MemoryArena& workspace, CudaExecutionView execution) {
    sample_impl(logits, out, nullptr, token_domain, configs, logical_positions, purpose, live_rows,
                workspace, execution);
}

void sample(const Tensor& logits, Tensor& out, Tensor& top1_probabilities, int token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, int purpose,
            const Tensor& live_rows, MemoryArena& workspace, CudaExecutionView execution) {
    sample_impl(logits, out, &top1_probabilities, token_domain, configs, logical_positions, purpose,
                live_rows, workspace, execution);
}

} // namespace ninfer::ops
