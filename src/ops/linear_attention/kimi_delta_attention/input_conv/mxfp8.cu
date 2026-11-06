#include "ops/linear_attention/kimi_delta_attention/input_conv/mxfp8.cuh"

namespace ninfer::ops::detail::kda_input {
namespace {

bool window_route(int width, int batch) { return width <= 8 && std::int64_t(width) * batch <= 8; }

template <int Width, int Columns>
void window(const Mxfp8A16Operands<>& p, const Arguments& a, cudaStream_t stream) {
    using Base     = NativeFp8A16SimtSchedule<Columns, 4, 1, 8, 4>;
    using Schedule = NativeFp8ScheduleInstance<Base, kInputFeatures>;
    launch_native_fp8_a16_simt<Schedule>(p, FloatOutput{}, WindowEpilogue<Width, __nv_bfloat16>{a},
                                         stream);
}

template <int Width>
void window(const Mxfp8A16Operands<>& p, const Arguments& a, cudaStream_t stream) {
    if (a.batch == 1) {
        window<Width, Width>(p, a, stream);
    } else if constexpr (Width == 1) {
        if (a.batch == 2)
            window<1, 2>(p, a, stream);
        else if (a.batch <= 4)
            window<1, 4>(p, a, stream);
        else
            window<1, 8>(p, a, stream);
    } else {
        window<Width, (8 / Width) * Width>(p, a, stream);
    }
}

struct MmaSchedule
    : NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 128, 128, 32, 64, 2>, kInputFeatures> {
    static constexpr int kWarpGridTokens = kWarpsTokens;
};

} // namespace

Mxfp8Workspace mxfp8_execution_workspace(int width, int batch, bool update, LinearPolicy policy) {
    return window_route(width, batch) ? Mxfp8Workspace{}
                                      : mxfp8_workspace(width, batch, update, policy, true);
}

std::size_t mxfp8_workspace_capacity(int batch, int, int last, bool update, LinearPolicy policy) {
    return mxfp8_execution_workspace(last, batch, update, policy).bytes;
}

void launch_mxfp8(const Tensor& x, const Weight& w, LinearPolicy policy, const Arguments& a,
                  void* scratch, cudaStream_t stream) {
    if (window_route(a.width, a.batch)) {
        using Window = void (*)(const Mxfp8A16Operands<>&, const Arguments&, cudaStream_t);
        static constexpr Window windows[]{window<1>, window<2>, window<3>, window<4>,
                                          window<5>, window<6>, window<7>, window<8>};
        const auto p = mxfp8_a16_operands(x.view({kInputFeatures, a.width * a.batch}), w);
        windows[a.width - 1](p, a, stream);
        return;
    }
    launch_mxfp8_tiled<MmaSchedule, MmaSchedule, false>(x, w, policy, a, scratch, stream);
}

} // namespace ninfer::ops::detail::kda_input
