#include "ops/linear/q4/q4_rowsplit_launch.h"

#include "ops/linear/q4/q4_rowsplit_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* ptr, std::uintptr_t alignment) {
    return ptr != nullptr && (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

bool is_known_schedule(Q4ScheduleId schedule) {
    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Shared:
    case Q4ScheduleId::GemvR4W1Direct:
    case Q4ScheduleId::GemvR1W8Direct:
    case Q4ScheduleId::SimtR8C4:
    case Q4ScheduleId::SimtR8C8:
    case Q4ScheduleId::MmaR64C64:
    case Q4ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

bool is_gemv(Q4ScheduleId schedule) {
    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Shared:
    case Q4ScheduleId::GemvR4W1Direct:
    case Q4ScheduleId::GemvR1W8Direct:
        return true;
    case Q4ScheduleId::SimtR8C4:
    case Q4ScheduleId::SimtR8C8:
    case Q4ScheduleId::MmaR64C64:
    case Q4ScheduleId::MmaR64C128:
        return false;
    }
    return false;
}

bool is_simt(Q4ScheduleId schedule) {
    return schedule == Q4ScheduleId::SimtR8C4 || schedule == Q4ScheduleId::SimtR8C8;
}

int schedule_cols(Q4ScheduleId schedule) {
    switch (schedule) {
    case Q4ScheduleId::SimtR8C4:
        return 4;
    case Q4ScheduleId::SimtR8C8:
        return 8;
    case Q4ScheduleId::MmaR64C64:
        return 64;
    case Q4ScheduleId::MmaR64C128:
        return 128;
    case Q4ScheduleId::GemvR4W1Shared:
    case Q4ScheduleId::GemvR4W1Direct:
    case Q4ScheduleId::GemvR1W8Direct:
        return 1;
    }
    return 0;
}

void require_candidate_operands(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("q4 candidate: x/out must be contiguous BF16 matrices");
    }
    if (w.qtype != QType::Q4G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 64 || w.group_size != 64 || w.qhigh != nullptr ||
        w.high_plane_bytes != 0) {
        throw std::invalid_argument("q4 candidate: weight must be Q4G64 RowSplit");
    }
    if (w.ndim != 2 || w.n <= 0 || w.k <= 0 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q4 candidate: inconsistent matrix shapes");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.scales, 4)) {
        throw std::invalid_argument("q4 candidate: required pointer alignment is missing");
    }
}

} // namespace

const char* q4_schedule_name(Q4ScheduleId schedule) {
    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Shared:
        return "q4.gemv.r4.w1.g16.s1.groups_runtime.x_shared_full_k."
               "lane_q4x8.decode_fp16mantissa."
               "code_async_vec16_ca.scale_shared_pair32.lb1";
    case Q4ScheduleId::GemvR4W1Direct:
        return "q4.gemv.r4.w1.g16.s1.groups_runtime.x_direct."
               "lane_q4x8.decode_fp16mantissa."
               "code_async_vec16_ca.scale_shared_pair32.lb1";
    case Q4ScheduleId::GemvR1W8Direct:
        return "q4.gemv.r1.w8.g16.s1.groups_static80.x_direct."
               "lane_q4x2.decode_scalar_integer."
               "code_sync_vec16_ca.scale_scalar16_shuffle.lb1";
    case Q4ScheduleId::SimtR8C4:
        return "q4.simt.r8.c4.g16.s2.code_ca.lb1";
    case Q4ScheduleId::SimtR8C8:
        return "q4.simt.r8.c8.g16.s2.code_ca.lb1";
    case Q4ScheduleId::MmaR64C64:
        return "q4.mma.r64.c64.k64.wr32.wc32.s2.frag_pingpong.q_ca.x_ca.scale_scalar16.lb3";
    case Q4ScheduleId::MmaR64C128:
        return "q4.mma.r64.c128.k64.wr64.wc32.s2.frag_serial.q_cg.x_cg.scale_pair32.lb1";
    }
    return "q4.unknown";
}

const char* q4_kernel_variant_name(Q4KernelVariant variant) {
    switch (variant) {
    case Q4KernelVariant::None:
        return "none";
    case Q4KernelVariant::Full:
        return "full";
    case Q4KernelVariant::Predicated:
        return "predicated";
    }
    return "unknown";
}

bool q4_schedule_uses_mma(Q4ScheduleId schedule) {
    return schedule == Q4ScheduleId::MmaR64C64 || schedule == Q4ScheduleId::MmaR64C128;
}

bool q4_candidate_is_legal(Q4ScheduleId schedule, Q4KernelVariant variant,
                           const Q4Problem& problem) {
    if (!is_known_schedule(schedule) || problem.rows <= 0 || problem.k <= 0 || problem.cols <= 0 ||
        problem.padded_k != problem.k || (problem.k % 128) != 0 || (problem.rows % 16) != 0) {
        return false;
    }
    if (is_gemv(schedule)) {
        if (variant != Q4KernelVariant::None || problem.cols != 1) { return false; }
        if (schedule == Q4ScheduleId::GemvR4W1Shared && problem.k > 16 * 1024) { return false; }
        if (schedule == Q4ScheduleId::GemvR1W8Direct && problem.k != 5120) { return false; }
        return true;
    }
    if (variant == Q4KernelVariant::None) { return false; }
    const int tile_cols = schedule_cols(schedule);
    if (variant == Q4KernelVariant::Full) {
        if ((problem.cols % tile_cols) != 0) { return false; }
        if (is_simt(schedule)) { return (problem.rows % 8) == 0 && ((problem.k / 64) % 16) == 0; }
        return (problem.rows % 64) == 0;
    }
    return variant == Q4KernelVariant::Predicated;
}

void q4_rowsplit_launch_candidate(Q4ScheduleId schedule, Q4KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream) {
    require_candidate_operands(x, w, out);
    const Q4Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (!q4_candidate_is_legal(schedule, variant, problem)) {
        throw std::invalid_argument(std::string("q4 candidate: illegal ") +
                                    q4_schedule_name(schedule) + "." +
                                    q4_kernel_variant_name(variant));
    }

    switch (schedule) {
    case Q4ScheduleId::GemvR4W1Shared:
        q4_rowsplit_gemv_r4_w1_shared_launch(x, w, out, stream);
        return;
    case Q4ScheduleId::GemvR4W1Direct:
        q4_rowsplit_gemv_r4_w1_direct_launch(x, w, out, stream);
        return;
    case Q4ScheduleId::GemvR1W8Direct:
        q4_rowsplit_gemv_r1_w8_direct_launch(x, w, out, stream);
        return;
    case Q4ScheduleId::SimtR8C4:
        q4_rowsplit_gemm_simt_r8_c4_launch(variant, x, w, out, stream);
        return;
    case Q4ScheduleId::SimtR8C8:
        q4_rowsplit_gemm_simt_r8_c8_launch(variant, x, w, out, stream);
        return;
    case Q4ScheduleId::MmaR64C64:
        q4_rowsplit_gemm_mma_r64_c64_launch(variant, x, w, out, stream);
        return;
    case Q4ScheduleId::MmaR64C128:
        q4_rowsplit_gemm_mma_r64_c128_launch(variant, x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 candidate: unknown schedule");
}

} // namespace ninfer::ops::detail
