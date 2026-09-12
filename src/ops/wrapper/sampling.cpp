// ninfer::ops - sample wrapper: public api validation and dispatch.
#include "ninfer/ops/sampling.h"

#include "ops/common/sampling_workspace.h"
#include "ops/launcher/sampling.h" // detail::sample_batch_launch

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    if (token_domain <= kSamplerTileItems || min_lanes > kSamplerMaxColumns) { return 0; }
    return detail::sampling_workspace_exact_bytes(token_domain,
                                                  std::min(max_lanes, kSamplerMaxColumns));
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream) {
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

    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(token_domain, logits.ne[1], logits.ne[1]);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_batch_launch(logits, out, token_domain, configs, logical_positions, purpose,
                                scratch, stream);
}

void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream) {
    if (token_ids.dtype != DType::I32 || token_ids.ne[0] <= 0 || token_ids.ne[1] != 1 ||
        token_ids.ne[2] != 1 || token_ids.ne[3] != 1 || !token_ids.is_contiguous() ||
        token_ids.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_ids must be a contiguous non-empty I32 vector");
    }
    if (token_counts.dtype != DType::I32 || token_counts.ne[0] <= 0 || token_counts.ne[1] != 1 ||
        token_counts.ne[2] != 1 || token_counts.ne[3] != 1 || !token_counts.is_contiguous() ||
        token_counts.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_counts must be a contiguous non-empty I32 vector");
    }
    detail::increment_token_counts_launch(token_ids, token_counts, stream);
}

} // namespace ninfer::ops
