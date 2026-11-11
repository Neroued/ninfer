#include "ops/gated_residual/mix/fp8/kernel/output.cuh"
#include "ops/gated_residual/mix/fp8/kernel/input.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/common/host/operands.h"

namespace ninfer::ops::detail::hc::mix::fp8 {
using DownThin   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<32, 32, 128, 1, 2, 3, 2>, 340, 16, 1>;
using DownMedium = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 64, 128, 2, 2, 3, 1>, 340, 8, 2>;
using DownWide = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 8, 2>;
using UpThin   = Fp8A8MmaSchedule<32, 64, 128, 1, 4, 2, 4, Cache::cg, Cache::cg,
                                  Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using UpMedium =
    Fp8ScheduleInstance<Fp8A8MmaSchedule<64, 64, 64, 2, 2, 2, 4, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>,
                        kRank>;
using UpWide =
    Fp8ScheduleInstance<Fp8A8MmaSchedule<128, 64, 64, 2, 2, 2, 3, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>,
                        kRank>;

using UpWideRows =
    Fp8ScheduleInstance<Fp8A8MmaSchedule<64, 128, 64, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>,
                        kRank>;

int up_token_tile(UpSchedule up) {
    return up == UpSchedule::Thin ? 32 : up == UpSchedule::WideTokens ? 128 : 64;
}

Plan plan(LinearPolicy down, LinearPolicy up, int m) {
    return {allows_a8(down), allows_a8(up) && m >= 33,
            m <= 64    ? 32
            : m <= 320 ? 64
                       : 128,
            m <= 64     ? UpSchedule::Thin
            : m <= 128  ? UpSchedule::Medium
            : m <= 1024 ? UpSchedule::WideRows
                        : UpSchedule::WideTokens};
}

int encoded_columns(int m, int tile) {
    const auto rounded = common::round_up(std::int64_t(m), std::int64_t(tile));
    // The generic MMA tail remains available when a rounded int32 extent is
    // unrepresentable. Ordinary allocations use complete private tiles.
    return rounded <= std::numeric_limits<int>::max() ? int(rounded) : m;
}

Layout layout(int m, Plan p) {
    LayoutBuilder b;
    const auto u =
        b.add(common::operand::bytes(std::uint64_t(kRank) * m, 2, "hc_mix"), 256, "HC u");
    LayoutRegion codes, scales, partials;
    if (p.down_a8 || p.up_a8) {
        const int up_columns = p.up_a8 ? encoded_columns(m, up_token_tile(p.up)) : m;
        const auto code_bytes =
            std::max(p.down_a8 ? fp8_a8_checked_bytes(m, kTotal) : std::size_t(0),
                     p.up_a8 ? fp8_a8_checked_bytes(up_columns, kRank) : std::size_t(0));
        codes  = b.add(code_bytes, 256, "HC A8 codes");
        scales = b.add(fp8_a8_checked_bytes(std::max(m, up_columns), 4), 256, "HC A8 scales");
    }
    if (p.down_a8) {
        const auto bytes = p.down_tile == 32   ? DownThin::kPartialBytes
                           : p.down_tile == 64 ? DownMedium::kPartialBytes
                                               : DownWide::kPartialBytes;
        partials         = b.add(bytes, 256, "HC down partials");
    }
    return {u, codes, scales, partials, b.finish(1)};
}

void launch(Weights w, const Tensor& z, Tensor& out, Tensor* j, Scratch s, Plan p,
            CudaExecutionView e) {
    const auto stream = e.stream();
    const int m       = z.ne[2];
    const Tensor input(z.data, z.device, DType::BF16, {kTotal, m});
    const LinearBf16Output down{s.u, kRank};
    if (j && !p.down_a8) {
        injection<<<dim3(m, 4), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(z.data),
            static_cast<const __nv_bfloat16*>(w.injection->qdata), static_cast<float*>(j->data), m);
        CUDA_CHECK(cudaGetLastError());
    }
    if (p.down_a8) {
        if (j) {
            encode_injection<<<m, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(z.data),
                static_cast<const __nv_bfloat16*>(w.injection->qdata), s.encoded.codes,
                s.encoded.scales, static_cast<float*>(j->data));
            CUDA_CHECK(cudaGetLastError());
        } else
            launch_fp8_a8_quantize(input, w.down.weight, s.encoded, stream);
        const auto a      = fp8_a8_operands(w.down.weight, s.encoded, m);
        const auto submit = [&]<class Schedule>() {
            launch_fp8_a8_tma_mma<Schedule>(a, down, DownEpilogue{}, stream, s.encoded.partials);
        };
        if (p.down_tile == 32)
            submit.template operator()<DownThin>();
        else if (p.down_tile == 64)
            submit.template operator()<DownMedium>();
        else
            submit.template operator()<DownWide>();
    } else {
        const auto a = fp8_a16_operands(input, w.down.weight);
        if (m == 1)
            launch_fp8_a16_gemv<Fp8A16GemvSchedule<4, 1, 8, 4, Fp8CodeCache::Default, 2, 2>>(
                a, down, DownEpilogue{}, stream);
        else
            launch_fp8_a16_simt<
                Fp8A16SimtSchedule<4, 1, 8, 4, 4, Fp8SimtActivationAccess::TokenPacked,
                                   Fp8CodeCache::Default, 2, Fp8SimtBlockOrder::RowsContiguous, 2>>(
                a, down, DownEpilogue{}, stream);
    }
    const Tensor u(s.u, z.device, DType::BF16, {kRank, m});
    const UpOutput output{static_cast<const __nv_bfloat16*>(z.data),
                          static_cast<__nv_bfloat16*>(out.data), m};
    if (p.up_a8) {
        const int columns = encoded_columns(m, up_token_tile(p.up));
        encode_up<<<columns, 256, 0, stream>>>(s.u, s.encoded.codes, s.encoded.scales, m);
        CUDA_CHECK(cudaGetLastError());
        const auto a      = fp8_a8_operands(w.up.weight, s.encoded, columns);
        const auto submit = [&]<class Schedule>() {
            launch_fp8_a8_mma<Schedule>(a, output, A8UpEpilogue{}, stream,
                                        StreamRows<Schedule::kBlockRows / 4>{});
        };
        if (p.up == UpSchedule::Thin)
            submit.template operator()<UpThin>();
        else if (p.up == UpSchedule::Medium)
            submit.template operator()<UpMedium>();
        else if (p.up == UpSchedule::WideRows)
            submit.template operator()<UpWideRows>();
        else
            submit.template operator()<UpWide>();
    } else {
        const auto a = fp8_a16_operands(u, w.up.weight);
        if (m == 1) {
            up_gemv<<<kWidth / 16, 128, 0, stream>>>(a, output);
            CUDA_CHECK(cudaGetLastError());
        } else {
            using Up = Fp8A16MmaSchedule<64, 16, 64, 16, 8, 2, 2>;
            launch_fp8_a16_mma<Up>(a, output, A16UpEpilogue{}, stream, StreamRows<16>{});
        }
    }
}
} // namespace ninfer::ops::detail::hc::mix::fp8
