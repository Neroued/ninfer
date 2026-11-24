#include "ninfer/ops/indexed88_q8.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/indexing.h"
#include "ops/common/host/launch.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q8/q8_a16_sliced_k_mma.cuh"
#include "ops/linear/kernels/q8/q8_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

struct Indexed88Rows {
    static constexpr int kOutputRowsPerCta = 16;
    const std::int32_t* device_row_ids;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return device_row_ids[output_row0 + local_row];
    }
};

using BaseSchedule = detail::Q8A16SlicedKMmaSchedule<
    8, 8, 1, 2, detail::Q8ScaleAccess::Shared,
    Cache::ca, Cache::cg, detail::Q8ActivationStage::ActiveOnly>;
using Schedule = BaseSchedule::with_problem<5120, 8>;

static_assert(kThinkingCapIndexed88Rows % Indexed88Rows::kOutputRowsPerCta == 0);
static_assert(Schedule::kBlockRows == Indexed88Rows::kOutputRowsPerCta);
static_assert(Schedule::kStaticK == 5120 && Schedule::kTokenCapacity == 8);

void require_tensor(const Tensor& t, DType dtype, int rows, int tokens, Device device) {
    if (!t.data || !t.is_contiguous() || t.dtype != dtype || t.device != device ||
        t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("Indexed88: invalid tensor shape, dtype, or device");
    }
}

} // namespace

void indexed88_q8_propose(const Tensor& hidden, const Weight& physical_q8_head,
                          const std::int32_t* selected_ids_device, Tensor& logits,
                          Tensor& proposal_ids, CudaExecutionView execution) {
    const int tokens = hidden.ne[1];
    if (tokens <= 0 || tokens > 8) {
        throw std::invalid_argument("Indexed88: only proposal widths 1..8 are supported");
    }
    if (!selected_ids_device) {
        throw std::invalid_argument("Indexed88: selection IDs are not initialized");
    }
    require_tensor(hidden, DType::BF16, 5120, tokens, execution.device());
    require_tensor(logits, DType::BF16, kThinkingCapIndexed88Rows, tokens,
                   execution.device());
    require_tensor(proposal_ids, DType::I32, tokens, 1, execution.device());
    if (physical_q8_head.device != execution.device() ||
        physical_q8_head.n != 135168 || physical_q8_head.k != 5120 ||
        physical_q8_head.padded_shape[1] != 5120 ||
        physical_q8_head.qtype != QType::Q8_G32_FP16 ||
        physical_q8_head.layout != QuantLayout::RowSplit ||
        physical_q8_head.group_size != 32) {
        throw std::invalid_argument("Indexed88: expected physical Q8 G32 135168x5120 head");
    }

    const detail::Q8LinearOperands operands =
        detail::q8_linear_operands(hidden, physical_q8_head);
    detail::validate_q8_operands(operands);

    using Output = detail::LinearBf16Output;
    using Epilogue = detail::LinearIdentityEpilogue;
    constexpr auto kernel =
        detail::q8_a16_sliced_k_mma_kernel<Schedule, true, Output, Epilogue, Indexed88Rows>;
    const int shared = common::prepare_shared<Schedule::kSharedBytes, kernel>();
    constexpr int blocks = kThinkingCapIndexed88Rows / Indexed88Rows::kOutputRowsPerCta;

    kernel<<<blocks, Schedule::kThreads, shared, execution.stream()>>>(
        operands,
        Output{static_cast<__nv_bfloat16*>(logits.data), kThinkingCapIndexed88Rows},
        Epilogue{}, Indexed88Rows{selected_ids_device}, 0);
    CUDA_CHECK(cudaGetLastError());

    // The selection is sorted ascending, so argmax ties retain the smallest
    // physical token ID exactly as on the full-head path (within selected rows).
    argmax(logits, proposal_ids, kThinkingCapIndexed88Rows, execution);
    remap_ids_inplace(proposal_ids, selected_ids_device, kThinkingCapIndexed88Rows, execution);
}

} // namespace ninfer::ops
