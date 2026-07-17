#include "ops/linear/q6/q6_rowsplit_launch.h"

#include "ops/linear/q6/q6_rowsplit_kernels.h"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool is_known_schedule(Q6ScheduleId schedule) {
    switch (schedule) {
    case Q6ScheduleId::SimtR8C4:
    case Q6ScheduleId::SimtR8C8:
    case Q6ScheduleId::MmaR64C64:
    case Q6ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

bool is_simt(Q6ScheduleId schedule) {
    return schedule == Q6ScheduleId::SimtR8C4 || schedule == Q6ScheduleId::SimtR8C8;
}

int schedule_cols(Q6ScheduleId schedule) {
    switch (schedule) {
    case Q6ScheduleId::SimtR8C4:
        return 4;
    case Q6ScheduleId::SimtR8C8:
        return 8;
    case Q6ScheduleId::MmaR64C64:
        return 64;
    case Q6ScheduleId::MmaR64C128:
        return 128;
    }
    return 0;
}

void require_candidate_operands(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("q6 candidate: x/out must be contiguous BF16 matrices");
    }
    if (w.qtype != QType::Q6G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 64 || w.group_size != 64 || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("q6 candidate: weight must be Q6G64 RowSplit");
    }
    if (w.ndim != 2 || w.n <= 0 || w.k <= 0 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q6 candidate: inconsistent matrix shapes");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 4)) {
        throw std::invalid_argument("q6 candidate: required pointer alignment is missing");
    }
}

} // namespace

const char* q6_schedule_name(Q6ScheduleId schedule) {
    switch (schedule) {
    case Q6ScheduleId::SimtR8C4:
        return "q6.simt.r8.c4.slab1024.s2.code_ca.high_ca.scale_pair32";
    case Q6ScheduleId::SimtR8C8:
        return "q6.simt.r8.c8.slab1024.s2.code_ca.high_ca.scale_pair32";
    case Q6ScheduleId::MmaR64C64:
        return "q6.mma.r64.c64.k64.wr32.wc32.s2.frag_pingpong.q_ca.x_ca.scale_scalar16.lb3";
    case Q6ScheduleId::MmaR64C128:
        return "q6.mma.r64.c128.k64.wr64.wc32.s2.frag_serial.q_cg.x_cg.scale_pair32.lb1";
    }
    return "q6.unknown";
}

const char* q6_kernel_variant_name(Q6KernelVariant variant) {
    switch (variant) {
    case Q6KernelVariant::None:
        return "none";
    case Q6KernelVariant::Full:
        return "full";
    case Q6KernelVariant::Predicated:
        return "predicated";
    }
    return "unknown";
}

bool q6_schedule_uses_mma(Q6ScheduleId schedule) {
    return schedule == Q6ScheduleId::MmaR64C64 || schedule == Q6ScheduleId::MmaR64C128;
}

bool q6_candidate_is_legal(Q6ScheduleId schedule, Q6KernelVariant variant,
                           const Q6Problem& problem) {
    if (!is_known_schedule(schedule) || problem.rows <= 0 || problem.k <= 0 || problem.cols <= 0 ||
        problem.padded_k < problem.k || (problem.padded_k % 128) != 0) {
        return false;
    }
    if (is_simt(schedule)) { return variant == Q6KernelVariant::None; }
    if (variant == Q6KernelVariant::None || (problem.k % 8) != 0 || (problem.padded_k % 64) != 0) {
        return false;
    }
    if (variant == Q6KernelVariant::Full) {
        return problem.rows % 64 == 0 && problem.cols % schedule_cols(schedule) == 0 &&
               problem.k == problem.padded_k && problem.k % 64 == 0;
    }
    return variant == Q6KernelVariant::Predicated;
}

void q6_rowsplit_launch_fixed(Q6Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    require_candidate_operands(x, w, out);
    const Q6Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (!q6_candidate_is_legal(plan.schedule, plan.variant, problem)) {
        throw std::invalid_argument(std::string("q6 fixed launch: illegal ") +
                                    q6_schedule_name(plan.schedule) + "." +
                                    q6_kernel_variant_name(plan.variant));
    }

    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            switch (plan.schedule) {
            case Q6ScheduleId::SimtR8C4:
                q6_rowsplit_simt_r8_c4_launch(x_slice, w, out_slice, stream);
                return;
            case Q6ScheduleId::SimtR8C8:
                q6_rowsplit_simt_r8_c8_launch(x_slice, w, out_slice, stream);
                return;
            case Q6ScheduleId::MmaR64C64:
                q6_rowsplit_mma_r64_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case Q6ScheduleId::MmaR64C128:
                q6_rowsplit_mma_r64_c128_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            }
            throw std::logic_error("q6 fixed launch: unknown schedule");
        });
}

void q6_rowsplit_launch_candidate(Q6ScheduleId schedule, Q6KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream) {
    q6_rowsplit_launch_fixed({schedule, variant}, x, w, out, stream);
}

} // namespace ninfer::ops::detail
