#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 9216;
constexpr int kHidden         = 2048;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 16;
using Output                  = W8SplitOutput4<4096, 512, 4096, 512>;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                    Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                        Tensor& v, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols>
        <<<kRows / kRowsPerCta, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers =
    make_projection_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_attn_input_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                     cudaStream_t stream) {
    if (variant != W8KernelVariant::None || x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 attention input split-K MMA requires exact T=2..16");
    }
    kProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
