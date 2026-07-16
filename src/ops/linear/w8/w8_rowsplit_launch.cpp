#include "ops/linear/w8/w8_rowsplit_launch.h"

#include "ops/linear/w8/w8_rowsplit_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool is_known_schedule(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
    case W8ScheduleId::SimtR8C8:
    case W8ScheduleId::MmaR32C128:
    case W8ScheduleId::MmaR64C128:
        return true;
    }
    return false;
}

bool is_simt(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::SimtR8C4 || schedule == W8ScheduleId::SimtR8C8;
}

int schedule_rows(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
    case W8ScheduleId::SimtR8C8:
        return 8;
    case W8ScheduleId::MmaR32C128:
        return 32;
    case W8ScheduleId::MmaR64C128:
        return 64;
    }
    return 0;
}

int schedule_cols(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
        return 4;
    case W8ScheduleId::SimtR8C8:
        return 8;
    case W8ScheduleId::MmaR32C128:
    case W8ScheduleId::MmaR64C128:
        return 128;
    }
    return 0;
}

void require_candidate_operands(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("w8 candidate: x/out must be contiguous BF16 matrices");
    }
    if (w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 32 || w.group_size != 32 || w.qdata == nullptr ||
        w.qhigh != nullptr || w.scales == nullptr) {
        throw std::invalid_argument("w8 candidate: weight must be W8G32 RowSplit");
    }
    if (w.ndim != 2 || w.n <= 0 || w.k <= 0 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("w8 candidate: inconsistent matrix shapes");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16) || !aligned_to(w.qdata, 16) ||
        !aligned_to(w.scales, 4)) {
        throw std::invalid_argument("w8 candidate: required pointer alignment is missing");
    }
}

} // namespace

const char* w8_schedule_name(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::SimtR8C4:
        return "w8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case W8ScheduleId::SimtR8C8:
        return "w8.simt.r8.c8.slab1024.s2.code_ca.scale_pair32";
    case W8ScheduleId::MmaR32C128:
        return "w8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case W8ScheduleId::MmaR64C128:
        return "w8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    }
    return "w8.unknown";
}

const char* w8_kernel_variant_name(W8KernelVariant variant) {
    switch (variant) {
    case W8KernelVariant::None:
        return "none";
    case W8KernelVariant::Full:
        return "full";
    case W8KernelVariant::Predicated:
        return "predicated";
    }
    return "unknown";
}

bool w8_schedule_uses_mma(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::MmaR32C128 || schedule == W8ScheduleId::MmaR64C128;
}

bool w8_candidate_is_legal(W8ScheduleId schedule, W8KernelVariant variant,
                           const W8Problem& problem) {
    if (!is_known_schedule(schedule) || problem.rows <= 0 || problem.k <= 0 || problem.cols <= 0 ||
        problem.padded_k < problem.k || (problem.padded_k % 128) != 0) {
        return false;
    }
    if (variant == W8KernelVariant::None) { return false; }
    if (is_simt(schedule)) {
        if (variant == W8KernelVariant::Full) {
            return problem.rows % schedule_rows(schedule) == 0 &&
                   problem.cols % schedule_cols(schedule) == 0;
        }
        return variant == W8KernelVariant::Predicated;
    }
    if ((problem.k % 8) != 0 || (problem.padded_k % 256) != 0) { return false; }
    if (variant == W8KernelVariant::Full) {
        return problem.rows % schedule_rows(schedule) == 0 &&
               problem.cols % schedule_cols(schedule) == 0 && problem.k == problem.padded_k &&
               (problem.k % 64) == 0;
    }
    return variant == W8KernelVariant::Predicated;
}

void w8_rowsplit_launch_fixed(W8Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    require_candidate_operands(x, w, out);
    const W8Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (!w8_candidate_is_legal(plan.schedule, plan.variant, problem)) {
        throw std::invalid_argument(std::string("w8 fixed launch: illegal ") +
                                    w8_schedule_name(plan.schedule) + "." +
                                    w8_kernel_variant_name(plan.variant));
    }
    if (w8_schedule_uses_mma(plan.schedule) && !aligned_to(w.scales, 16)) {
        throw std::invalid_argument("w8 MMA candidate: scale plane must be 16-byte aligned");
    }

    switch (plan.schedule) {
    case W8ScheduleId::SimtR8C4:
        w8_rowsplit_gemm_simt_r8_c4_launch(plan.variant, x, w, out, stream);
        return;
    case W8ScheduleId::SimtR8C8:
        w8_rowsplit_gemm_simt_r8_c8_launch(plan.variant, x, w, out, stream);
        return;
    case W8ScheduleId::MmaR32C128:
        w8_rowsplit_gemm_mma_r32_c128_launch(plan.variant, x, w, out, stream);
        return;
    case W8ScheduleId::MmaR64C128:
        w8_rowsplit_gemm_mma_r64_c128_launch(plan.variant, x, w, out, stream);
        return;
    }
    throw std::logic_error("w8 fixed launch: unknown schedule");
}

void w8_rowsplit_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream) {
    w8_rowsplit_launch_fixed({schedule, variant}, x, w, out, stream);
}

} // namespace ninfer::ops::detail
