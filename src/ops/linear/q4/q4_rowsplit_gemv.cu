#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_kernels.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

template <class Schedule>
void require_gemv_problem(const Tensor& x, const Weight& w, const Tensor& out) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];

    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || !x.is_contiguous() ||
        !out.is_contiguous() || x.ne[1] != 1 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("q4 GEMV: x/out must be contiguous BF16 column vectors");
    }
    if (w.qtype != QType::Q4G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != Q4RowSplitStorage::kGroupK ||
        w.group_size != Q4RowSplitStorage::kGroupK || w.qhigh != nullptr ||
        w.high_plane_bytes != 0) {
        throw std::invalid_argument("q4 GEMV: weight must be Q4G64 RowSplit");
    }
    if (rows <= 0 || k <= 0 || w.ndim != 2 || w.n != rows || w.k != k || w.shape[0] != rows ||
        w.shape[1] != k || w.padded_shape[0] != rows || w.padded_shape[1] != k || (k % 128) != 0 ||
        out.ne[0] != rows || (rows % Schedule::kRowsPerCta) != 0) {
        throw std::invalid_argument(
            "q4 GEMV: requires full CTA rows, runtime Kpad == K, and K divisible by 128");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.scales, 4)) {
        throw std::invalid_argument("q4 GEMV: required pointer alignment is missing");
    }
    if constexpr (Schedule::kStaticGroupsPerRow > 0) {
        if (k / Q4RowSplitStorage::kGroupK != Schedule::kStaticGroupsPerRow) {
            throw std::invalid_argument(
                "q4 GEMV: K does not match the fixed groups-per-row schedule");
        }
    }
    if constexpr (Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK) {
        if (k > 16 * 1024) {
            throw std::invalid_argument(
                "q4 GEMV: full-K shared activation schedule supports K <= 16384");
        }
    }
}

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_gemv_problem<Schedule>(x, w, out);

    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    const std::size_t dynamic_shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(k) * sizeof(__nv_bfloat16)
            : 0u;

    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, dynamic_shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_rowsplit_gemv_r4_w1_shared_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_gemv<Q4GemvR4W1SharedSchedule>(x, w, out, stream);
}

void q4_rowsplit_gemv_r4_w1_direct_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_gemv<Q4GemvR4W1DirectSchedule>(x, w, out, stream);
}

void q4_rowsplit_gemv_r1_w8_direct_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_gemv<Q4GemvR1W8DirectSchedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
