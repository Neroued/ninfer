#include "ops/common/host/operands.h"
#include "ninfer/ops/sequence.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include "core/device.h"
#include <cstdint>

namespace ninfer::ops {
using namespace common;

namespace {

void require_row_pitched_matrix(const Tensor& t, std::int32_t rows, std::int32_t cols,
                                const char* op, const char* name, Device device) {
    operand::device(t.device, device, op, name);
    constexpr std::int64_t element = sizeof(std::int32_t);
    if (rows <= 0 || cols <= 0 || t.dtype != DType::I32 || t.ne[0] != rows || t.ne[1] != cols ||
        t.ne[2] != 1 || t.ne[3] != 1 || t.nb[0] != element || t.nb[1] < rows * element ||
        t.nb[1] % element != 0 || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid row-pitched matrix for " + name);
    }
}

__global__ void prepare_autoregressive_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int a         = accepted[row];
    a             = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        alignment_ids[row * T + j] = j < a ? verify_ids[row * T + j + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int licensed       = licensed_counts[row];
        const int remaining      = remaining_budgets[row] - licensed;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            const int offset          = s * ar_step_stride + row;
            const int position        = updated_frontiers[row] + s;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            ar_valid_columns[offset]  = s + 1 < next ? 1 : 0;
        }
    }
}

void prepare_autoregressive_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                         const Tensor& accepted, const Tensor& updated_frontiers,
                                         const Tensor& remaining_budgets,
                                         const Tensor& licensed_counts, const Tensor& rope_deltas,
                                         Tensor& alignment_ids, Tensor& next_extents,
                                         Tensor& ar_positions, Tensor& ar_rope_positions,
                                         Tensor& ar_valid_columns, std::int32_t max_context,
                                         cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = verify_ids.ne[0] - 1;
    const int batch      = verify_ids.ne[1];
    const int ar_step_stride =
        static_cast<int>(ar_positions.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((k + kBlock) / kBlock),
                    static_cast<unsigned int>(batch));
    prepare_autoregressive_round_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(next_anchors.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(updated_frontiers.data),
        static_cast<const std::int32_t*>(remaining_budgets.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(alignment_ids.data),
        static_cast<std::int32_t*>(next_extents.data),
        static_cast<std::int32_t*>(ar_positions.data),
        static_cast<std::int32_t*>(ar_rope_positions.data),
        static_cast<std::int32_t*>(ar_valid_columns.data), k, ar_step_stride, max_context);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void prepare_autoregressive_round(const Tensor& verify_ids, const Tensor& next_anchors,
                                  const Tensor& accepted, const Tensor& updated_frontiers,
                                  const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                  const Tensor& rope_deltas, Tensor& alignment_ids,
                                  Tensor& next_extents, Tensor& ar_positions,
                                  Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                  std::int32_t max_context, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "prepare_autoregressive_round";
    const std::int32_t T     = verify_ids.ne[0];
    const std::int32_t batch = verify_ids.ne[1];
    if (T < 2 || T > 6) {
        throw std::invalid_argument("prepare_autoregressive_round: T must be in [2,6]");
    }
    if (batch < 1) {
        throw std::invalid_argument("prepare_autoregressive_round: B must be positive");
    }
    if (max_context <= 0) {
        throw std::invalid_argument("prepare_autoregressive_round: max_context must be positive");
    }
    common::operand::tensor(verify_ids, DType::I32, {T, batch}, 1, op, "verify_ids",
                            execution.device());
    common::operand::tensor(next_anchors, DType::I32, {batch}, 1, op, "next_anchors",
                            execution.device());
    common::operand::tensor(accepted, DType::I32, {batch}, 1, op, "accepted", execution.device());
    common::operand::tensor(updated_frontiers, DType::I32, {batch}, 1, op, "updated_frontiers",
                            execution.device());
    common::operand::tensor(remaining_budgets, DType::I32, {batch}, 1, op, "remaining_budgets",
                            execution.device());
    common::operand::tensor(licensed_counts, DType::I32, {batch}, 1, op, "licensed_counts",
                            execution.device());
    common::operand::tensor(rope_deltas, DType::I32, {batch}, 1, op, "rope_deltas",
                            execution.device());
    common::operand::tensor(alignment_ids, DType::I32, {T, batch}, 1, op, "alignment_ids",
                            execution.device());
    common::operand::tensor(next_extents, DType::I32, {batch}, 1, op, "next_extents",
                            execution.device());
    const std::int32_t steps = std::max(T - 2, 1);
    require_row_pitched_matrix(ar_positions, batch, steps, op, "ar_positions", execution.device());
    require_row_pitched_matrix(ar_rope_positions, batch, steps, op, "ar_rope_positions",
                               execution.device());
    require_row_pitched_matrix(ar_valid_columns, batch, steps, op, "ar_valid_columns",
                               execution.device());
    if (ar_rope_positions.nb[1] != ar_positions.nb[1] ||
        ar_valid_columns.nb[1] != ar_positions.nb[1]) {
        throw std::invalid_argument(
            "prepare_autoregressive_round: AR outputs must share one step stride");
    }
    prepare_autoregressive_round_launch(verify_ids, next_anchors, accepted, updated_frontiers,
                                        remaining_budgets, licensed_counts, rope_deltas,
                                        alignment_ids, next_extents, ar_positions,
                                        ar_rope_positions, ar_valid_columns, max_context, stream);
}

} // namespace ninfer::ops
