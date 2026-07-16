#include "ops/linear/q5/q5_rowsplit_launch.h"

#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool is_known_schedule(Q5ScheduleId schedule) {
    switch (schedule) {
    case Q5ScheduleId::GemvR16S2X:
    case Q5ScheduleId::SimtR8C4:
    case Q5ScheduleId::SimtR8C8:
    case Q5ScheduleId::SimtSplit2Exact:
    case Q5ScheduleId::SimtSplit4Exact:
    case Q5ScheduleId::MmaR64C64:
    case Q5ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

bool is_simt(Q5ScheduleId schedule) {
    return schedule == Q5ScheduleId::SimtR8C4 || schedule == Q5ScheduleId::SimtR8C8 ||
           schedule == Q5ScheduleId::SimtSplit2Exact || schedule == Q5ScheduleId::SimtSplit4Exact;
}

int schedule_cols(Q5ScheduleId schedule) {
    switch (schedule) {
    case Q5ScheduleId::SimtR8C4:
        return 4;
    case Q5ScheduleId::SimtR8C8:
        return 8;
    case Q5ScheduleId::MmaR64C64:
        return 64;
    case Q5ScheduleId::MmaR64C128:
        return 128;
    case Q5ScheduleId::GemvR16S2X:
    case Q5ScheduleId::SimtSplit2Exact:
    case Q5ScheduleId::SimtSplit4Exact:
        return 0;
    }
    return 0;
}

bool split2_shape(const Q5Problem& problem) {
    return (problem.rows == 5120 && problem.k == 6144 && problem.padded_k == 6144) ||
           (problem.rows == 5120 && problem.k == 17408 && problem.padded_k == 17408);
}

bool split4_shape(const Q5Problem& problem) {
    return problem.rows == 6144 && problem.k == 5120 && problem.padded_k == 5120;
}

void require_candidate_operands(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("q5 candidate: x/out must be contiguous BF16 matrices");
    }
    if (w.qtype != QType::Q5G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 64 || w.group_size != 64 || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("q5 candidate: weight must be Q5G64 RowSplit");
    }
    if (w.ndim != 2 || w.n <= 0 || w.k <= 0 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q5 candidate: inconsistent matrix shapes");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 16)) {
        throw std::invalid_argument("q5 candidate: required 16-byte pointer alignment is missing");
    }
}

} // namespace

const char* q5_schedule_name(Q5ScheduleId schedule) {
    switch (schedule) {
    case Q5ScheduleId::GemvR16S2X:
        return "q5.gemv.r16.s2.x_shared.tile_g16";
    case Q5ScheduleId::SimtR8C4:
        return "q5.simt.r8.c4.slab1024.s2";
    case Q5ScheduleId::SimtR8C8:
        return "q5.simt.r8.c8.slab1024.s2";
    case Q5ScheduleId::SimtSplit2Exact:
        return "q5.simt.row1.splitk2.c_exact";
    case Q5ScheduleId::SimtSplit4Exact:
        return "q5.simt.row1.splitk4.c_exact";
    case Q5ScheduleId::MmaR64C64:
        return "q5.mma.r64.c64.k64.wr32.wc32.s2.frag_pingpong";
    case Q5ScheduleId::MmaR64C128:
        return "q5.mma.r64.c128.k64.wr64.wc32.s2.frag_serial";
    }
    return "q5.unknown";
}

const char* q5_kernel_variant_name(Q5KernelVariant variant) {
    switch (variant) {
    case Q5KernelVariant::None:
        return "none";
    case Q5KernelVariant::Full:
        return "full";
    case Q5KernelVariant::Predicated:
        return "predicated";
    }
    return "unknown";
}

bool q5_schedule_uses_mma(Q5ScheduleId schedule) {
    return schedule == Q5ScheduleId::MmaR64C64 || schedule == Q5ScheduleId::MmaR64C128;
}

bool q5_candidate_is_legal(Q5ScheduleId schedule, Q5KernelVariant variant,
                           const Q5Problem& problem) {
    if (!is_known_schedule(schedule) || problem.rows <= 0 || problem.k <= 0 || problem.cols <= 0 ||
        problem.padded_k < problem.k || (problem.padded_k % 128) != 0) {
        return false;
    }
    if (schedule == Q5ScheduleId::GemvR16S2X) {
        return variant == Q5KernelVariant::None && problem.cols == 1 && problem.rows == 6144 &&
               problem.k == 5120 && problem.padded_k == 5120;
    }
    if (schedule == Q5ScheduleId::SimtSplit2Exact) {
        return variant == Q5KernelVariant::None && problem.cols >= 2 && problem.cols <= 6 &&
               split2_shape(problem);
    }
    if (schedule == Q5ScheduleId::SimtSplit4Exact) {
        return variant == Q5KernelVariant::None && problem.cols >= 2 && problem.cols <= 6 &&
               split4_shape(problem);
    }
    if (is_simt(schedule)) { return variant == Q5KernelVariant::None; }
    if (variant == Q5KernelVariant::None || (problem.k % 8) != 0 || (problem.padded_k % 64) != 0) {
        return false;
    }
    if (variant == Q5KernelVariant::Full) {
        return problem.rows % 64 == 0 && problem.cols % schedule_cols(schedule) == 0 &&
               problem.k == problem.padded_k && problem.k % 64 == 0;
    }
    return variant == Q5KernelVariant::Predicated;
}

void q5_rowsplit_launch_fixed(Q5Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    require_candidate_operands(x, w, out);
    const Q5Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (!q5_candidate_is_legal(plan.schedule, plan.variant, problem)) {
        throw std::invalid_argument(std::string("q5 fixed launch: illegal ") +
                                    q5_schedule_name(plan.schedule) + "." +
                                    q5_kernel_variant_name(plan.variant));
    }

    switch (plan.schedule) {
    case Q5ScheduleId::GemvR16S2X:
        q5_rowsplit_gemv_r16_s2_x_launch(x, w, out, stream);
        return;
    case Q5ScheduleId::SimtR8C4:
        q5_rowsplit_simt_r8_c4_launch(x, w, out, stream);
        return;
    case Q5ScheduleId::SimtR8C8:
        q5_rowsplit_simt_r8_c8_launch(x, w, out, stream);
        return;
    case Q5ScheduleId::SimtSplit2Exact:
        q5_rowsplit_simt_split2_exact_launch(x, w, out, stream);
        return;
    case Q5ScheduleId::SimtSplit4Exact:
        q5_rowsplit_simt_split4_exact_launch(x, w, out, stream);
        return;
    case Q5ScheduleId::MmaR64C64:
        q5_rowsplit_mma_r64_c64_launch(plan.variant, x, w, out, stream);
        return;
    case Q5ScheduleId::MmaR64C128:
        q5_rowsplit_mma_r64_c128_launch(plan.variant, x, w, out, stream);
        return;
    }
    throw std::logic_error("q5 fixed launch: unknown schedule");
}

void q5_rowsplit_launch_candidate(Q5ScheduleId schedule, Q5KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream) {
    q5_rowsplit_launch_fixed({schedule, variant}, x, w, out, stream);
}

} // namespace ninfer::ops::detail
