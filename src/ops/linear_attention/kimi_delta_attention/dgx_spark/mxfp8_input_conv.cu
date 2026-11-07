#include "ops/linear_attention/kimi_delta_attention/input_conv/mxfp8.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"

namespace ninfer::ops::detail::kda_input {
namespace {

bool window_route(int width, int batch) {
    const auto columns = std::int64_t(width) * batch;
    if (columns <= 32) return true;
    // Every tile owns complete request windows. Narrower tiles reuse weights in L2
    // while keeping enough resident warps for the projection and convolution.
    return width <= 8 && columns <= 64;
}

bool single_tile_route(int width, int batch, LinearPolicy policy) {
    return policy == LinearPolicy::AllowA8 && std::int64_t(width) * batch <= 128;
}

template <int Width, int Rows, int Columns, int Warps, int Stages, int WarpK = 128>
void window(const Mxfp8A16Operands<>& p, const Arguments& a, cudaStream_t stream) {
    using Schedule =
        NativeFp8ScheduleInstance<NativeFp8A16SlicedKSchedule<Rows, Columns, Warps, WarpK, Stages>,
                                  kInputFeatures>;
    launch_native_fp8_a16_sliced_k<Schedule>(p, FloatOutput{}, ParallelWindowEpilogue<Width>{{a}},
                                             stream);
}

template <int Width>
void window(const Mxfp8A16Operands<>& p, const Arguments& a, cudaStream_t stream) {
    // A token tile may own several requests, but never divides a request window.
    // Small updates publish complete 32-byte history sectors. Keep the same K
    // partition as record so the projected values and history rounding agree exactly.
    if (a.destination && p.tokens >= 3 && p.tokens <= 5) {
        if constexpr (Width <= 4)
            if (p.tokens <= 4) return window<Width, 16, 4, 8, 1, 256>(p, a, stream);
        if constexpr (Width == 1 || Width == 5)
            if (p.tokens == 5) return window<Width, 16, 5, 8, 1, 256>(p, a, stream);
    }
    if constexpr (Width <= 2)
        if (p.tokens <= 2) return window<Width, 8, 2, 8, 2, 256>(p, a, stream);
    if constexpr (Width <= 4)
        if (p.tokens <= 4) return window<Width, 8, 4, 8, 1, 256>(p, a, stream);
    if constexpr (Width <= 8)
        if (p.tokens <= 8) return window<Width, 8, 8, 8, 1, 256>(p, a, stream);
    if constexpr (Width == 3 || Width == 6)
        if (p.tokens <= 24) return window<Width, 16, 12, 8, 1>(p, a, stream);
    if constexpr (Width == 5) {
        if (p.tokens <= 30) return window<Width, 16, 15, 8, 1>(p, a, stream);
        return window<Width, 16, 20, 4, 1>(p, a, stream);
    }
    if constexpr (Width == 7) {
        if (p.tokens <= 28) return window<Width, 16, 14, 8, 1>(p, a, stream);
        return window<Width, 32, 28, 4, 1>(p, a, stream);
    }
    if constexpr (Width <= 16)
        if (p.tokens <= 16) return window<Width, 16, 16, 8, 1>(p, a, stream);
    if constexpr (Width > 0 && Width <= 16 && 16 % Width == 0)
        if (p.tokens <= 32) return window<Width, 16, 16, 8, 1>(p, a, stream);
    if constexpr (Width <= 24) {
        if (p.tokens <= 24) {
            window<Width, 16, 24, 4, 1>(p, a, stream);
            return;
        }
    }
    if constexpr (Width > 0 && Width <= 8 && 24 % Width == 0)
        if (p.tokens <= 48) return window<Width, 32, 24, 4, 1>(p, a, stream);
    window<Width, 32, 32, 4, 1>(p, a, stream);
}

template <int Columns>
void single_tile(const Tensor& x, const Weight& w, const Arguments& a, void* scratch,
                 cudaStream_t stream) {
    using Schedule = ConvolutionSchedule<NativeFp8ScheduleInstance<
        Mxfp8A8TmaSchedule<128, Columns, 128, 32, 32, Columns == 64 ? 1 : 2>, kInputFeatures>>;
    const auto plan =
        mxfp8_workspace(a.width, a.batch, a.destination != nullptr, LinearPolicy::AllowA8, false);
    const MemorySpan storage{scratch, plan.bytes, x.device};
    const Mxfp8A8Workspace packed{static_cast<std::uint8_t*>(plan.codes.bind(storage).data),
                                  static_cast<std::uint8_t*>(plan.scales.bind(storage).data)};
    const int columns = a.width * a.batch;
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), kInputFeatures, columns,
                             packed, stream);
    launch_mxfp8_a8_tma<Schedule>(mxfp8_a8_operands(w, packed, columns),
                                  TiledConvolution{a, nullptr, nullptr},
                                  PackedConvolutionEpilogue<int, true, true>{}, stream);
}

using A16Schedule = ConvolutionSchedule<
    NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 128, 128, 32, 64, 2>, kInputFeatures>>;
using A8Schedule = ConvolutionSchedule<
    NativeFp8ScheduleInstance<Mxfp8A8TmaSchedule<128, 128, 128, 32, 32, 2>, kInputFeatures>>;

} // namespace

Mxfp8Workspace mxfp8_execution_workspace(int width, int batch, bool update, LinearPolicy policy) {
    return window_route(width, batch) ? Mxfp8Workspace{}
                                      : mxfp8_workspace(width, batch, update, policy,
                                                        !single_tile_route(width, batch, policy));
}

std::size_t mxfp8_workspace_capacity(int batch, int, int last, bool update, LinearPolicy policy) {
    // Whole-request tiles cover every width <= 8. For a fixed batch the zero-scratch
    // domain is a prefix; the later tiled workspace grows monotonically.
    return mxfp8_execution_workspace(last, batch, update, policy).bytes;
}

void launch_mxfp8(const Tensor& x, const Weight& w, LinearPolicy policy, const Arguments& a,
                  void* scratch, cudaStream_t stream) {
    if (window_route(a.width, a.batch)) {
        using Window = void (*)(const Mxfp8A16Operands<>&, const Arguments&, cudaStream_t);
        static constexpr Window windows[]{window<1>,  window<2>,  window<3>,  window<4>,
                                          window<5>,  window<6>,  window<7>,  window<8>,
                                          window<9>,  window<10>, window<11>, window<12>,
                                          window<13>, window<14>, window<15>, window<16>};
        const auto p = mxfp8_a16_operands(x.view({kInputFeatures, a.width * a.batch}), w);
        if (a.width <= 16)
            windows[a.width - 1](p, a, stream);
        else
            window<0>(p, a, stream);
    } else if (single_tile_route(a.width, a.batch, policy)) {
        if (std::int64_t(a.width) * a.batch <= 64)
            single_tile<64>(x, w, a, scratch, stream);
        else
            single_tile<128>(x, w, a, scratch, stream);
    } else if (policy == LinearPolicy::AllowA8 && std::int64_t(a.width) * a.batch > 128) {
        const auto columns = std::int64_t(a.width) * a.batch;
        const auto launch  = [&]<class Base>() {
            using Schedule = ConvolutionSchedule<NativeFp8ScheduleInstance<Base, kInputFeatures>>;
            launch_mxfp8_tiled<A16Schedule, Schedule, true, true, true, true>(x, w, policy, a,
                                                                               scratch, stream);
        };
        if (columns <= 192)
            launch.template operator()<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 8, true>>();
        else if (columns <= 384 || columns > 2048)
            launch
                .template operator()<Mxfp8A8TmaSchedule<192, 128, 128, 96, 32, 2, 1, 8, true, 1>>();
        else if (columns <= 512)
            launch
                .template operator()<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 2, true, 1>>();
        else
            launch
                .template operator()<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>();
    } else {
        launch_mxfp8_tiled<A16Schedule, A8Schedule, true>(x, w, policy, a, scratch, stream);
    }
}

} // namespace ninfer::ops::detail::kda_input
