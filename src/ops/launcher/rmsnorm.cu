// ninfer::ops - RMSNorm launcher: finite semantic dispatch over general row geometries.
#include "ops/launcher/rmsnorm.h"

#include "ops/kernel/rmsnorm.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Past this many blocks the gated epilogue gives up its hoisted loads. Below one block per SM the
// prefetch is the only source of overlap and those kernels run at 0.83x to 0.96x; above it a
// second resident block already supplies that overlap and only the register cost is left (35 -> 50
// on the warp kernel), which measures 1.02x to 1.14x. Swept over grid size on both gated shapes
// the crossing sits between 176 and 192 blocks; this is the 170 SMs of this part, a literal
// because nothing in the tree queries the device, so it is not portable.
constexpr std::int64_t kRmsPrefetchBlocks = 170;

template <RmsEpilogue Epilogue>
void launch_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor* z, Tensor& out,
                    std::int32_t d, std::int64_t rows, float eps, bool aligned2,
                    cudaStream_t stream) {
    const auto* x_bf16 = static_cast<const __nv_bfloat16*>(x.data);
    const auto* w_bf16 = static_cast<const __nv_bfloat16*>(weight.data);
    const auto* z_bf16 = z != nullptr ? static_cast<const __nv_bfloat16*>(z->data) : nullptr;
    auto* out_bf16     = static_cast<__nv_bfloat16*>(out.data);

    // The hoisted weight and gate cost registers. Measured at 512-thread blocks, only the
    // gated epilogue loses a block per SM for them: warp 35 -> 50 and the wide row 38 -> 48, both
    // three blocks to two, while every other instantiation stays at three. So only that epilogue
    // consults the grid, and the un-prefetched instantiation is compiled only where it is reached.
    constexpr bool kGateOnGrid = Epilogue == RmsEpilogue::Gated;

    if constexpr (Epilogue != RmsEpilogue::Gated) {
        if (aligned2 && d == 5120) {
            // Fixed width removes the dynamic pair-count predicates. Ten pairs per thread
            // with hoisted gains wins the hidden-row sweep through prefill.
            rmsnorm_cta_bf16x2_kernel<Epilogue, 256, 10, true, 5120>
                <<<static_cast<unsigned>(rows), 256, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                    reinterpret_cast<const __nv_bfloat162*>(w_bf16), nullptr,
                    reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
            return;
        }
    }
    if constexpr (Epilogue == RmsEpilogue::Offset) {
        if (aligned2 && d == 256) {
            // One warp per row, four rows per CTA across both query and key projections.
            constexpr int block = 128;
            rmsnorm_warp_bf16x2_kernel<Epilogue, block, true, 256>
                <<<static_cast<unsigned>((rows + 3) / 4), block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                    reinterpret_cast<const __nv_bfloat162*>(w_bf16), nullptr,
                    reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
            return;
        }
    }
    if (aligned2 && d == 128 && Epilogue == RmsEpilogue::Plain) {
        // Plain per-head D128 is latency-bound for Q32/KV8 rows. Four rows per CTA follows the
        // measured lower envelope from T=1 through the 1024-token prefill point.
        constexpr int kBlock         = 128;
        constexpr int kWarpsPerBlock = kBlock / kWarpSize;
        const auto blocks = static_cast<unsigned int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        rmsnorm_d128_bf16x2_kernel<Epilogue, kBlock>
            <<<blocks, kBlock, 0, stream>>>(reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                                            reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                                            reinterpret_cast<const __nv_bfloat162*>(z_bf16),
                                            reinterpret_cast<__nv_bfloat162*>(out_bf16), rows, eps);
    } else if (aligned2 && d >= 64 && d <= 256 && d % 64 == 0) {
        constexpr int kBlock         = 512;
        constexpr int kWarpsPerBlock = kBlock / kWarpSize;
        const auto blocks = static_cast<unsigned int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        if constexpr (kGateOnGrid) {
            if (blocks > kRmsPrefetchBlocks) {
                rmsnorm_warp_bf16x2_kernel<Epilogue, kBlock, false><<<blocks, kBlock, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                    reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                    reinterpret_cast<const __nv_bfloat162*>(z_bf16),
                    reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
                return;
            }
        }
        rmsnorm_warp_bf16x2_kernel<Epilogue, kBlock, true><<<blocks, kBlock, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(x_bf16),
            reinterpret_cast<const __nv_bfloat162*>(w_bf16),
            reinterpret_cast<const __nv_bfloat162*>(z_bf16),
            reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
    } else if (aligned2 && d == 2048 && Epilogue == RmsEpilogue::Plain) {
        // The 512-thread CTA is faster than 128/256-thread alternatives at every measured DFlash
        // extent and reaches the same-topology payload floor at the prefill endpoint.
        rmsnorm_d2048_bf16x2_kernel<Epilogue><<<static_cast<unsigned int>(rows), 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(x_bf16),
            reinterpret_cast<const __nv_bfloat162*>(w_bf16),
            reinterpret_cast<const __nv_bfloat162*>(z_bf16),
            reinterpret_cast<__nv_bfloat162*>(out_bf16), rows, eps);
    } else if (aligned2 && d >= 512 && d <= 3072 && d % 512 == 0) {
        // 30 -> 34 registers at Block=256, which is 6 blocks per SM either way under the
        // 1536-thread cap, so this route pays no occupancy for the prefetch and is not gated.
        rmsnorm_cta_bf16x2_kernel<Epilogue, 256, 6, true>
            <<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                reinterpret_cast<const __nv_bfloat162*>(z_bf16),
                reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
    } else if (aligned2 && d > 3072 && d <= 8192 && d % 1024 == 0) {
        if constexpr (kGateOnGrid) {
            if (rows > kRmsPrefetchBlocks) {
                rmsnorm_cta_bf16x2_kernel<Epilogue, 512, 8, false>
                    <<<static_cast<unsigned int>(rows), 512, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                        reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                        reinterpret_cast<const __nv_bfloat162*>(z_bf16),
                        reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
                return;
            }
        }
        rmsnorm_cta_bf16x2_kernel<Epilogue, 512, 8, true>
            <<<static_cast<unsigned int>(rows), 512, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat162*>(x_bf16),
                reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                reinterpret_cast<const __nv_bfloat162*>(z_bf16),
                reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
    } else {
        rmsnorm_generic_kernel<Epilogue><<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
            x_bf16, w_bf16, z_bf16, out_bf16, d, rows, eps);
    }
}

} // namespace

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream) {
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("rmsnorm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto z_addr = z != nullptr ? reinterpret_cast<std::uintptr_t>(z->data) : 0;
    const auto o_addr = reinterpret_cast<std::uintptr_t>(out.data);
    const bool aligned2 =
        ((x_addr | w_addr | z_addr | o_addr) & (alignof(__nv_bfloat162) - 1)) == 0;

    if (z != nullptr) {
        launch_rmsnorm<RmsEpilogue::Gated>(x, weight, z, out, d, rows, eps, aligned2, stream);
    } else if (unit_offset) {
        launch_rmsnorm<RmsEpilogue::Offset>(x, weight, nullptr, out, d, rows, eps, aligned2,
                                            stream);
    } else {
        launch_rmsnorm<RmsEpilogue::Plain>(x, weight, nullptr, out, d, rows, eps, aligned2, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
