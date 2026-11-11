#include "ops/gated_residual/mix/fp8/kernel/output.cuh"
#include "ops/gated_residual/mix/fp8/kernel/input.cuh"
#include "ops/gated_residual/mix/fp8/kernel/down.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/common/host/operands.h"
#include "small.cuh"

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
    const auto down_schedule = m <= kDownVectorColumns ? DownSchedule::SplitVector
                               : !allows_a8(down)      ? DownSchedule::Vector
                               : m <= kSmallColumns    ? DownSchedule::A8Segmented
                               : m <= 64               ? DownSchedule::A8Thin
                               : m <= 320              ? DownSchedule::A8Medium
                                                       : DownSchedule::A8Wide;
    const bool grouped =
        allows_a8(up) && m >= 3 && m <= kSmallColumns && down_schedule != DownSchedule::Vector;
    const auto encoding = grouped                    ? UpEncoding::Group
                          : allows_a8(up) && m >= 33 ? UpEncoding::Row
                                                     : UpEncoding::None;
    return {down_schedule, encoding,
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

Layout layout(int m, Plan p, bool injection) {
    LayoutBuilder b;
    LayoutRegion u;
    if (!p.grouped_up())
        u = b.add(common::operand::bytes(std::uint64_t(kRank) * m, 2, "hc_mix"), 256, "HC u");
    LayoutRegion codes, scales, partials;
    if (p.down_a8() || p.up_a8()) {
        const int up_columns =
            p.up_a8() && !p.grouped_up() ? encoded_columns(m, up_token_tile(p.up)) : m;
        const auto code_bytes =
            std::max(p.down_a8() ? fp8_a8_checked_bytes(m, kTotal) : std::size_t(0),
                     p.up_a8() ? fp8_a8_checked_bytes(up_columns, kRank) : std::size_t(0));
        codes                  = b.add(code_bytes, 256, "HC A8 codes");
        const auto scale_bytes = p.down == DownSchedule::A8Segmented
                                     ? std::size_t(down_control_elements(m, injection) +
                                                   (p.grouped_up() ? m * kRank / kUpGroup : 0)) *
                                           sizeof(float)
                                 : p.grouped_up()
                                     ? std::size_t(m * kRank / kUpGroup) * sizeof(float)
                                     : fp8_a8_checked_bytes(std::max(m, up_columns), 4);
        scales                 = b.add(scale_bytes, 256, "HC A8 scales");
    }
    if (p.down == DownSchedule::SplitVector) {
        partials =
            b.add(common::operand::bytes(std::uint64_t(kDownPartitions) * m * (kRank + kStreams),
                                         sizeof(float), "hc_mix"),
                  256, "HC down partials");
    } else if (p.down_a8()) {
        const auto bytes = p.down == DownSchedule::A8Segmented ? SegmentedDown::kPartialBytes
                           : p.down == DownSchedule::A8Thin    ? DownThin::kPartialBytes
                           : p.down == DownSchedule::A8Medium  ? DownMedium::kPartialBytes
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
    if (j && p.down == DownSchedule::Vector) {
        injection<<<dim3(m, 4), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(z.data),
            static_cast<const __nv_bfloat16*>(w.injection->qdata), static_cast<float*>(j->data), m);
        CUDA_CHECK(cudaGetLastError());
    }
    if (p.down == DownSchedule::SplitVector) {
        const auto submit = [&]<int Tokens, int Values>() {
            constexpr int Rows = 2;
            down_split_vector<Tokens, Rows, Values>
                <<<dim3((kRank + (j ? kStreams : 0) + 4 * Rows - 1) / (4 * Rows),
                        (m + Tokens - 1) / Tokens, kDownPartitions),
                   128, 0, stream>>>(static_cast<const __nv_bfloat16*>(z.data),
                                     static_cast<const std::uint8_t*>(w.down.weight.qdata),
                                     j ? static_cast<const __nv_bfloat16*>(w.injection->qdata)
                                       : nullptr,
                                     s.encoded.partials, m, kDownPartitions);
        };
        if (m == 1)
            submit.template operator()<1, 8>();
        else
            submit.template operator()<4, 4>();
        CUDA_CHECK(cudaGetLastError());
        if (p.grouped_up()) {
            const auto finish = [&]<bool Injection>() {
                finish_small_down<false, true, Injection>
                    <<<dim3(kRank / kUpGroup, m), kUpGroup, 0, stream>>>(
                        s.encoded.partials, static_cast<const __nv_bfloat16*>(w.down.weight.scales),
                        nullptr, nullptr, s.encoded.codes, s.encoded.scales,
                        j ? static_cast<float*>(j->data) : nullptr, m);
            };
            if (j)
                finish.template operator()<true>();
            else
                finish.template operator()<false>();
        } else {
            down_split_finish<<<dim3((kRank + (j ? kStreams : 0) + 127) / 128, m), 128, 0,
                                stream>>>(
                s.encoded.partials, static_cast<const __nv_bfloat16*>(w.down.weight.scales), s.u,
                j ? static_cast<float*>(j->data) : nullptr, m, kDownPartitions);
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (p.down == DownSchedule::A8Segmented) {
        const auto encode = [&]<bool Injection>() {
            encode_segments<Injection><<<dim3(m, kDownPartitions), kSegmentThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(z.data),
                j ? static_cast<const __nv_bfloat16*>(w.injection->qdata) : nullptr,
                s.encoded.codes, s.encoded.scales, m);
        };
        if (j)
            encode.template operator()<true>();
        else
            encode.template operator()<false>();
        CUDA_CHECK(cudaGetLastError());
        auto a = fp8_a8_operands(w.down.weight, s.encoded, m);
        // This route publishes only raw partials; partition scales belong to the finish kernel.
        a.x_scales = nullptr;
        const Fp8TmaDescriptors descriptors{
            fp8_tma_map(a.x, m, kTotal, SegmentedDown::kBlockTokens, SegmentedDown::kBlockK),
            fp8_tma_map(a.codes, kRank, kTotal, SegmentedDown::kBlockRows, SegmentedDown::kBlockK)};
        constexpr int tiles = kRank / SegmentedDown::kBlockRows;
        static_assert(SegmentedDown::kMaxParts == kDownPartitions &&
                      tiles * kDownPartitions <= SegmentedDown::kSplitWaveCtas);
        const auto split      = fp8_tma_split_k_plan<SegmentedDown>(tiles, kTotal);
        constexpr auto kernel = fp8_a8_tma_mma_kernel<SegmentedDown, false, LinearBf16Output,
                                                      DownEpilogue, true, Fp8IdentityRows>;
        constexpr int bytes =
            fp8_tma_scratch_bytes<SegmentedDown, DownEpilogue> + SegmentedDown::kBarrierBytes;
        const int shared = common::prepare_shared<bytes, kernel, true>();
        kernel<<<split.split_ctas, SegmentedDown::kThreads, shared, stream>>>(
            descriptors, a, down, DownEpilogue{}, Fp8IdentityRows{}, 0, m, split,
            s.encoded.partials);
        CUDA_CHECK(cudaGetLastError());
        const auto finish = [&]<bool Encoded, bool Injection>() {
            finish_small_down<true, Encoded, Injection>
                <<<dim3(kRank / kUpGroup, m), kUpGroup, 0, stream>>>(
                    s.encoded.partials, static_cast<const __nv_bfloat16*>(w.down.weight.scales),
                    s.encoded.scales, s.u, s.encoded.codes,
                    s.encoded.scales + down_control_elements(m, Injection),
                    j ? static_cast<float*>(j->data) : nullptr, m);
        };
        if (p.grouped_up()) {
            if (j)
                finish.template operator()<true, true>();
            else
                finish.template operator()<true, false>();
        } else {
            if (j)
                finish.template operator()<false, true>();
            else
                finish.template operator()<false, false>();
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (p.down_a8()) {
        if (j) {
            const auto encode = [&]<int Threads>() {
                encode_injection<Threads><<<m, Threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(z.data),
                    static_cast<const __nv_bfloat16*>(w.injection->qdata), s.encoded.codes,
                    s.encoded.scales, static_cast<float*>(j->data));
            };
            if (m <= 128)
                encode.template operator()<1024>();
            else
                encode.template operator()<256>();
            CUDA_CHECK(cudaGetLastError());
        } else
            launch_fp8_a8_quantize(input, w.down.weight, s.encoded, stream);
        const auto a      = fp8_a8_operands(w.down.weight, s.encoded, m);
        const auto submit = [&]<class Schedule>() {
            launch_fp8_a8_tma_mma<Schedule>(a, down, DownEpilogue{}, stream, s.encoded.partials);
        };
        if (p.down == DownSchedule::A8Thin)
            submit.template operator()<DownThin>();
        else if (p.down == DownSchedule::A8Medium)
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
    if (p.grouped_up()) {
        const int control =
            p.down == DownSchedule::A8Segmented ? down_control_elements(m, j != nullptr) : 0;
        const UpOutput output{static_cast<const __nv_bfloat16*>(z.data),
                              static_cast<__nv_bfloat16*>(out.data), m};
        const auto submit = [&]<int Tokens>() {
            using S               = GroupedUpSchedule<Tokens>;
            constexpr auto kernel = grouped_up_kernel<Tokens>;
            const int bytes       = common::prepare_shared<S::kSharedBytes, kernel>();
            kernel<<<kTotal / S::kBlockRows, S::kThreads, bytes, stream>>>(
                s.encoded.codes, s.encoded.scales + control,
                static_cast<const std::uint8_t*>(w.up.weight.qdata),
                static_cast<const __nv_bfloat16*>(w.up.weight.scales), output);
        };
        if (m <= 16)
            submit.template operator()<16>();
        else
            submit.template operator()<32>();
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const Tensor u(s.u, z.device, DType::BF16, {kRank, m});
    const UpOutput output{static_cast<const __nv_bfloat16*>(z.data),
                          static_cast<__nv_bfloat16*>(out.data), m};
    if (p.up_a8()) {
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
        } else if (m >= 17 && m <= 32) {
            using Up = Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 32, 64, 16, 16, 2, 2>, kRank>;
            launch_fp8_a16_mma<Up>(a, output, A16UpEpilogue{}, stream, StreamRows<16>{});
        } else {
            using Up = Fp8A16MmaSchedule<64, 16, 64, 16, 8, 2, 2>;
            launch_fp8_a16_mma<Up>(a, output, A16UpEpilogue{}, stream, StreamRows<16>{});
        }
    }
}
} // namespace ninfer::ops::detail::hc::mix::fp8
