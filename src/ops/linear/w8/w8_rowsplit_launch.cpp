#include "ops/linear/w8/w8_rowsplit_launch.h"

#include "ops/linear/w8/w8_rowsplit_kernels.h"
#include "ops/common/token_slices.h"

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
    case W8ScheduleId::DecodeR4:
    case W8ScheduleId::DecodeR8:
    case W8ScheduleId::DecodeR16:
    case W8ScheduleId::SplitKMmaExactT:
    case W8ScheduleId::SplitKMma32PlusTail:
    case W8ScheduleId::SplitKMediumC48:
    case W8ScheduleId::SplitKMediumC64:
    case W8ScheduleId::SplitKMediumC96:
    case W8ScheduleId::SplitKMediumC128:
    case W8ScheduleId::SplitKMediumC144:
    case W8ScheduleId::SplitKMediumC160:
    case W8ScheduleId::SimtR8C4:
    case W8ScheduleId::SimtR8C8:
    case W8ScheduleId::MmaR32C32:
    case W8ScheduleId::MmaR32C48:
    case W8ScheduleId::MmaR32C64:
    case W8ScheduleId::MmaR32C80:
    case W8ScheduleId::MmaR32C96:
    case W8ScheduleId::MmaR32C112:
    case W8ScheduleId::MmaR32C128:
    case W8ScheduleId::MmaR48C64:
    case W8ScheduleId::MmaR48C80:
    case W8ScheduleId::MmaR48C96:
    case W8ScheduleId::MmaR48C112:
    case W8ScheduleId::MmaR48C128:
    case W8ScheduleId::MmaR64C32:
    case W8ScheduleId::MmaR64C48:
    case W8ScheduleId::MmaR64C64:
    case W8ScheduleId::MmaR64C80:
    case W8ScheduleId::MmaR64C96:
    case W8ScheduleId::MmaR64C112:
    case W8ScheduleId::MmaR64C128:
    case W8ScheduleId::MmaR96C64:
    case W8ScheduleId::MmaR96C80:
    case W8ScheduleId::MmaR96C96:
    case W8ScheduleId::MmaR96C112:
    case W8ScheduleId::MmaR128C64:
    case W8ScheduleId::MmaR128C80:
        return true;
    }
    return false;
}

bool is_simt(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::SimtR8C4 || schedule == W8ScheduleId::SimtR8C8;
}

bool is_decode(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::DecodeR4 || schedule == W8ScheduleId::DecodeR8 ||
           schedule == W8ScheduleId::DecodeR16;
}

bool is_exact(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::SplitKMmaExactT ||
           schedule == W8ScheduleId::SplitKMma32PlusTail;
}

bool is_medium_splitk(W8ScheduleId schedule) {
    return schedule == W8ScheduleId::SplitKMediumC48 || schedule == W8ScheduleId::SplitKMediumC64 ||
           schedule == W8ScheduleId::SplitKMediumC96 ||
           schedule == W8ScheduleId::SplitKMediumC128 ||
           schedule == W8ScheduleId::SplitKMediumC144 || schedule == W8ScheduleId::SplitKMediumC160;
}

int schedule_rows(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::DecodeR4:
        return 4;
    case W8ScheduleId::DecodeR8:
        return 8;
    case W8ScheduleId::DecodeR16:
    case W8ScheduleId::SplitKMmaExactT:
    case W8ScheduleId::SplitKMma32PlusTail:
    case W8ScheduleId::SplitKMediumC48:
    case W8ScheduleId::SplitKMediumC64:
    case W8ScheduleId::SplitKMediumC96:
    case W8ScheduleId::SplitKMediumC128:
    case W8ScheduleId::SplitKMediumC144:
    case W8ScheduleId::SplitKMediumC160:
        return 16;
    case W8ScheduleId::SimtR8C4:
    case W8ScheduleId::SimtR8C8:
        return 8;
    case W8ScheduleId::MmaR32C32:
    case W8ScheduleId::MmaR32C48:
    case W8ScheduleId::MmaR32C64:
    case W8ScheduleId::MmaR32C80:
    case W8ScheduleId::MmaR32C96:
    case W8ScheduleId::MmaR32C112:
    case W8ScheduleId::MmaR32C128:
        return 32;
    case W8ScheduleId::MmaR48C64:
    case W8ScheduleId::MmaR48C80:
    case W8ScheduleId::MmaR48C96:
    case W8ScheduleId::MmaR48C112:
    case W8ScheduleId::MmaR48C128:
        return 48;
    case W8ScheduleId::MmaR64C32:
    case W8ScheduleId::MmaR64C48:
    case W8ScheduleId::MmaR64C64:
    case W8ScheduleId::MmaR64C80:
    case W8ScheduleId::MmaR64C96:
    case W8ScheduleId::MmaR64C112:
    case W8ScheduleId::MmaR64C128:
        return 64;
    case W8ScheduleId::MmaR96C64:
    case W8ScheduleId::MmaR96C80:
    case W8ScheduleId::MmaR96C96:
    case W8ScheduleId::MmaR96C112:
        return 96;
    case W8ScheduleId::MmaR128C64:
    case W8ScheduleId::MmaR128C80:
        return 128;
    }
    return 0;
}

int schedule_cols(W8ScheduleId schedule) {
    switch (schedule) {
    case W8ScheduleId::DecodeR4:
    case W8ScheduleId::DecodeR8:
    case W8ScheduleId::DecodeR16:
        return 1;
    case W8ScheduleId::SplitKMmaExactT:
        return 32;
    case W8ScheduleId::SplitKMma32PlusTail:
        return 32;
    case W8ScheduleId::SplitKMediumC48:
        return 48;
    case W8ScheduleId::SplitKMediumC64:
        return 64;
    case W8ScheduleId::SplitKMediumC96:
        return 96;
    case W8ScheduleId::SplitKMediumC128:
        return 128;
    case W8ScheduleId::SplitKMediumC144:
        return 144;
    case W8ScheduleId::SplitKMediumC160:
        return 160;
    case W8ScheduleId::SimtR8C4:
        return 4;
    case W8ScheduleId::SimtR8C8:
        return 8;
    case W8ScheduleId::MmaR32C32:
    case W8ScheduleId::MmaR64C32:
        return 32;
    case W8ScheduleId::MmaR32C48:
    case W8ScheduleId::MmaR64C48:
        return 48;
    case W8ScheduleId::MmaR32C64:
    case W8ScheduleId::MmaR48C64:
    case W8ScheduleId::MmaR64C64:
    case W8ScheduleId::MmaR96C64:
    case W8ScheduleId::MmaR128C64:
        return 64;
    case W8ScheduleId::MmaR32C80:
    case W8ScheduleId::MmaR48C80:
    case W8ScheduleId::MmaR64C80:
    case W8ScheduleId::MmaR96C80:
    case W8ScheduleId::MmaR128C80:
        return 80;
    case W8ScheduleId::MmaR32C96:
    case W8ScheduleId::MmaR48C96:
    case W8ScheduleId::MmaR64C96:
    case W8ScheduleId::MmaR96C96:
        return 96;
    case W8ScheduleId::MmaR32C112:
    case W8ScheduleId::MmaR48C112:
    case W8ScheduleId::MmaR64C112:
    case W8ScheduleId::MmaR96C112:
        return 112;
    case W8ScheduleId::MmaR32C128:
    case W8ScheduleId::MmaR48C128:
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
    case W8ScheduleId::DecodeR4:
        return "w8.decode.k16384.r4";
    case W8ScheduleId::DecodeR8:
        return "w8.decode.k16384.r8";
    case W8ScheduleId::DecodeR16:
        return "w8.decode.k16384.r16";
    case W8ScheduleId::SplitKMmaExactT:
        return "w8.splitk8.mma.r16.exact_t";
    case W8ScheduleId::SplitKMma32PlusTail:
        return "w8.splitk8.mma.r16.exact32_plus_tail";
    case W8ScheduleId::SplitKMediumC48:
        return "w8.splitk4.mma.r16.c48";
    case W8ScheduleId::SplitKMediumC64:
        return "w8.splitk4.mma.r16.c64";
    case W8ScheduleId::SplitKMediumC96:
        return "w8.splitk2.mma.r16.c96";
    case W8ScheduleId::SplitKMediumC128:
        return "w8.splitk2.mma.r16.c128";
    case W8ScheduleId::SplitKMediumC144:
        return "w8.splitk2.mma.r16.c144";
    case W8ScheduleId::SplitKMediumC160:
        return "w8.splitk2.mma.r16.c160";
    case W8ScheduleId::SimtR8C4:
        return "w8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case W8ScheduleId::SimtR8C8:
        return "w8.simt.r8.c8.slab1024.s2.code_ca.scale_pair32";
    case W8ScheduleId::MmaR32C32:
        return "w8.mma.r32.c32";
    case W8ScheduleId::MmaR32C48:
        return "w8.mma.r32.c48";
    case W8ScheduleId::MmaR32C64:
        return "w8.mma.r32.c64";
    case W8ScheduleId::MmaR32C80:
        return "w8.mma.r32.c80";
    case W8ScheduleId::MmaR32C96:
        return "w8.mma.r32.c96";
    case W8ScheduleId::MmaR32C112:
        return "w8.mma.r32.c112";
    case W8ScheduleId::MmaR32C128:
        return "w8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case W8ScheduleId::MmaR48C64:
        return "w8.mma.r48.c64";
    case W8ScheduleId::MmaR48C80:
        return "w8.mma.r48.c80";
    case W8ScheduleId::MmaR48C96:
        return "w8.mma.r48.c96";
    case W8ScheduleId::MmaR48C112:
        return "w8.mma.r48.c112";
    case W8ScheduleId::MmaR48C128:
        return "w8.mma.r48.c128";
    case W8ScheduleId::MmaR64C32:
        return "w8.mma.r64.c32";
    case W8ScheduleId::MmaR64C48:
        return "w8.mma.r64.c48";
    case W8ScheduleId::MmaR64C64:
        return "w8.mma.r64.c64";
    case W8ScheduleId::MmaR64C80:
        return "w8.mma.r64.c80";
    case W8ScheduleId::MmaR64C96:
        return "w8.mma.r64.c96";
    case W8ScheduleId::MmaR64C112:
        return "w8.mma.r64.c112";
    case W8ScheduleId::MmaR64C128:
        return "w8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case W8ScheduleId::MmaR96C64:
        return "w8.mma.r96.c64";
    case W8ScheduleId::MmaR96C80:
        return "w8.mma.r96.c80";
    case W8ScheduleId::MmaR96C96:
        return "w8.mma.r96.c96";
    case W8ScheduleId::MmaR96C112:
        return "w8.mma.r96.c112";
    case W8ScheduleId::MmaR128C64:
        return "w8.mma.r128.c64";
    case W8ScheduleId::MmaR128C80:
        return "w8.mma.r128.c80";
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
    return !is_decode(schedule) && !is_simt(schedule);
}

bool w8_candidate_is_legal(W8ScheduleId schedule, W8KernelVariant variant,
                           const W8Problem& problem) {
    if (!is_known_schedule(schedule) || problem.rows <= 0 || problem.k <= 0 || problem.cols <= 0 ||
        problem.padded_k < problem.k || (problem.padded_k % 128) != 0) {
        return false;
    }
    const bool conditioning_shape =
        problem.rows == 2048 && problem.k == 16384 && problem.padded_k == 16384;
    if (is_decode(schedule)) {
        return conditioning_shape && problem.cols == 1 && variant == W8KernelVariant::None;
    }
    if (schedule == W8ScheduleId::SplitKMmaExactT) {
        return conditioning_shape && problem.cols >= 2 && problem.cols <= 32 &&
               variant == W8KernelVariant::None;
    }
    if (schedule == W8ScheduleId::SplitKMma32PlusTail) {
        return conditioning_shape && problem.cols >= 33 && problem.cols <= 127 &&
               variant == W8KernelVariant::None;
    }
    if (is_medium_splitk(schedule)) {
        const int tile_cols = schedule_cols(schedule);
        return conditioning_shape && problem.cols >= 33 && problem.cols <= tile_cols &&
               variant == W8KernelVariant::None;
    }
    if (variant == W8KernelVariant::None) { return false; }
    if (is_simt(schedule)) {
        if (variant == W8KernelVariant::Full) {
            return problem.rows % schedule_rows(schedule) == 0 &&
                   problem.cols % schedule_cols(schedule) == 0;
        }
        return variant == W8KernelVariant::Predicated;
    }
    if (is_exact(schedule) || (problem.k % 8) != 0 || (problem.padded_k % 256) != 0) {
        return false;
    }
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
    const bool conditioning_exact_tail = plan.tail_policy == W8TailPolicy::ConditioningExact;
    if (conditioning_exact_tail &&
        (problem.rows != 2048 || problem.k != 16384 || problem.padded_k != 16384 ||
         plan.variant == W8KernelVariant::None || !w8_schedule_uses_mma(plan.schedule))) {
        throw std::invalid_argument(
            "w8 conditioning exact-tail policy requires a tiled [2048,16384] MMA route");
    }
    switch (plan.schedule) {
    case W8ScheduleId::DecodeR4:
        w8_rowsplit_decode_r4_launch(x, w, out, stream);
        return;
    case W8ScheduleId::DecodeR8:
        w8_rowsplit_decode_r8_launch(x, w, out, stream);
        return;
    case W8ScheduleId::DecodeR16:
        w8_rowsplit_decode_r16_launch(x, w, out, stream);
        return;
    case W8ScheduleId::SplitKMmaExactT:
        w8_rowsplit_exact_t_splitk_launch(x, w, out, stream);
        return;
    case W8ScheduleId::SplitKMma32PlusTail:
        w8_rowsplit_exact_t_composite_launch(x, w, out, stream);
        return;
    case W8ScheduleId::SplitKMediumC48:
    case W8ScheduleId::SplitKMediumC64:
    case W8ScheduleId::SplitKMediumC96:
    case W8ScheduleId::SplitKMediumC128:
    case W8ScheduleId::SplitKMediumC144:
    case W8ScheduleId::SplitKMediumC160:
        w8_rowsplit_medium_t_splitk_launch(plan.schedule, x, w, out, stream);
        return;
    default:
        break;
    }

    if (conditioning_exact_tail) {
        const std::int32_t tile_cols = schedule_cols(plan.schedule);
        const std::int32_t full_cols = (x.ne[1] / tile_cols) * tile_cols;
        if (full_cols > 0) {
            const Tensor x_prefix = x.slice(1, 0, full_cols);
            Tensor out_prefix     = out.slice(1, 0, full_cols);
            const W8Problem prefix_problem{problem.rows, problem.k, problem.padded_k, full_cols};
            const W8KernelVariant prefix_variant =
                w8_candidate_is_legal(plan.schedule, W8KernelVariant::Full, prefix_problem)
                    ? W8KernelVariant::Full
                    : W8KernelVariant::Predicated;
            w8_rowsplit_launch_fixed(
                {plan.schedule, prefix_variant, W8TailPolicy::Homogeneous}, x_prefix, w, out_prefix,
                stream);
        }

        const std::int32_t tail = x.ne[1] - full_cols;
        if (tail == 0) { return; }
        const Tensor x_tail = x.slice(1, full_cols, tail);
        Tensor out_tail     = out.slice(1, full_cols, tail);
        if (tail == 1) {
            w8_rowsplit_decode_r4_launch(x_tail, w, out_tail, stream);
        } else if (tail <= 32) {
            w8_rowsplit_exact_t_splitk_launch(x_tail, w, out_tail, stream);
        } else if (tail <= 88) {
            w8_rowsplit_exact_t_composite_launch(x_tail, w, out_tail, stream);
        } else if (tail <= 96) {
            w8_rowsplit_medium_t_splitk_launch(W8ScheduleId::SplitKMediumC96, x_tail, w, out_tail,
                                               stream);
        } else {
            w8_rowsplit_medium_t_splitk_launch(W8ScheduleId::SplitKMediumC128, x_tail, w, out_tail,
                                               stream);
        }
        return;
    }

    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            switch (plan.schedule) {
            case W8ScheduleId::SimtR8C4:
                w8_rowsplit_gemm_simt_r8_c4_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::SimtR8C8:
                w8_rowsplit_gemm_simt_r8_c8_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C32:
                w8_rowsplit_gemm_mma_r32_c32_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C48:
                w8_rowsplit_gemm_mma_r32_c48_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C64:
                w8_rowsplit_gemm_mma_r32_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C80:
                w8_rowsplit_gemm_mma_r32_c80_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C96:
                w8_rowsplit_gemm_mma_r32_c96_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C112:
                w8_rowsplit_gemm_mma_r32_c112_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR32C128:
                w8_rowsplit_gemm_mma_r32_c128_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR48C64:
                w8_rowsplit_gemm_mma_r48_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR48C80:
                w8_rowsplit_gemm_mma_r48_c80_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR48C96:
                w8_rowsplit_gemm_mma_r48_c96_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR48C112:
                w8_rowsplit_gemm_mma_r48_c112_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR48C128:
                w8_rowsplit_gemm_mma_r48_c128_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C32:
                w8_rowsplit_gemm_mma_r64_c32_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C48:
                w8_rowsplit_gemm_mma_r64_c48_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C64:
                w8_rowsplit_gemm_mma_r64_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C80:
                w8_rowsplit_gemm_mma_r64_c80_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C96:
                w8_rowsplit_gemm_mma_r64_c96_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C112:
                w8_rowsplit_gemm_mma_r64_c112_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR64C128:
                w8_rowsplit_gemm_mma_r64_c128_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR96C64:
                w8_rowsplit_gemm_mma_r96_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR96C80:
                w8_rowsplit_gemm_mma_r96_c80_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR96C96:
                w8_rowsplit_gemm_mma_r96_c96_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR96C112:
                w8_rowsplit_gemm_mma_r96_c112_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR128C64:
                w8_rowsplit_gemm_mma_r128_c64_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::MmaR128C80:
                w8_rowsplit_gemm_mma_r128_c80_launch(plan.variant, x_slice, w, out_slice, stream);
                return;
            case W8ScheduleId::DecodeR4:
            case W8ScheduleId::DecodeR8:
            case W8ScheduleId::DecodeR16:
            case W8ScheduleId::SplitKMmaExactT:
            case W8ScheduleId::SplitKMma32PlusTail:
            case W8ScheduleId::SplitKMediumC48:
            case W8ScheduleId::SplitKMediumC64:
            case W8ScheduleId::SplitKMediumC96:
            case W8ScheduleId::SplitKMediumC128:
            case W8ScheduleId::SplitKMediumC144:
            case W8ScheduleId::SplitKMediumC160:
                throw std::logic_error("w8 unsliced schedule reached sliced launch");
            }
            throw std::logic_error("w8 fixed launch: unknown schedule");
        });
}

void w8_rowsplit_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream,
                                  W8TailPolicy tail_policy) {
    w8_rowsplit_launch_fixed({schedule, variant, tail_policy}, x, w, out, stream);
}

} // namespace ninfer::ops::detail
