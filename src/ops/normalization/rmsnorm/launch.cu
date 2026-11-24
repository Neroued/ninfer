#include "ops/normalization/rmsnorm/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/normalization/rmsnorm/kernel/warp.cuh"
#include "ops/normalization/rmsnorm/kernel/cta.cuh"
#include "ops/normalization/rmsnorm/kernel/generic.cuh"
#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops::detail::rmsnorm {
namespace {

template <Epilogue Kind, int Block>
void launch_d128(const __nv_bfloat162* x, const __nv_bfloat162* weight, const __nv_bfloat162* gate,
                 __nv_bfloat162* out, std::int64_t rows, float eps, cudaStream_t stream) {
    const auto blocks =
        static_cast<unsigned>((rows + Block / common::kWarpSize - 1) / (Block / common::kWarpSize));
    rmsnorm_d128_bf16x2_kernel<Kind, Block>
        <<<blocks, Block, 0, stream>>>(x, weight, gate, out, rows, eps);
}

template <Epilogue Kind>
void launch_instance(const Tensor& x, const Tensor& weight, const Tensor* gate, Tensor& out,
                     Operands operands, float eps, Instance instance, cudaStream_t stream) {
    const auto d    = operands.dimension;
    const auto rows = operands.rows;
    const auto* x2  = static_cast<const __nv_bfloat162*>(x.data);
    const auto* w2  = static_cast<const __nv_bfloat162*>(weight.data);
    const auto* g2  = gate ? static_cast<const __nv_bfloat162*>(gate->data) : nullptr;
    auto* out2      = static_cast<__nv_bfloat162*>(out.data);
    switch (instance) {
    case Instance::D128Block128:
        if constexpr (is_gated<Kind> || Kind == Epilogue::Plain) {
            launch_d128<Kind, 128>(x2, w2, g2, out2, rows, eps, stream);
            return;
        }
        break;
    case Instance::D128Block256:
        if constexpr (is_gated<Kind>) {
            launch_d128<Kind, 256>(x2, w2, g2, out2, rows, eps, stream);
            return;
        }
        break;
    case Instance::D128Block512:
        if constexpr (is_gated<Kind>) {
            launch_d128<Kind, 512>(x2, w2, g2, out2, rows, eps, stream);
            return;
        }
        break;
    case Instance::D256:
        if constexpr (Kind == Epilogue::Offset) {
            rmsnorm_warp_bf16x2_kernel<Kind, 128, true, 256>
                <<<static_cast<unsigned>((rows + 3) / 4), 128, 0, stream>>>(x2, w2, nullptr, out2,
                                                                            d, rows, eps);
            return;
        }
        break;
    case Instance::D2048:
        if constexpr (Kind == Epilogue::Plain) {
            rmsnorm_d2048_bf16x2_kernel<Kind>
                <<<static_cast<unsigned>(rows), 512, 0, stream>>>(x2, w2, g2, out2, rows, eps);
            return;
        }
        break;
    case Instance::D5120:
        if constexpr (!is_gated<Kind>) {
            rmsnorm_cta_bf16x2_kernel<Kind, 256, 10, true, 5120>
                <<<static_cast<unsigned>(rows), 256, 0, stream>>>(x2, w2, nullptr, out2, d, rows,
                                                                  eps);
            return;
        }
        break;
    case Instance::D10240:
        if constexpr (Kind == Epilogue::Offset) {
            rmsnorm_cta_bf16x2_kernel<Kind, 512, 10, true, 10240>
                <<<static_cast<unsigned>(rows), 512, 0, stream>>>(x2, w2, nullptr, out2, d, rows,
                                                                  eps);
            return;
        }
        break;
    case Instance::Warp:
        if constexpr (is_gated<Kind>) {
            rmsnorm_warp_bf16x2_kernel<Kind, 512, false>
                <<<static_cast<unsigned>((rows + 15) / 16), 512, 0, stream>>>(x2, w2, g2, out2, d,
                                                                              rows, eps);
            return;
        }
        break;
    case Instance::WarpPrefetch:
        rmsnorm_warp_bf16x2_kernel<Kind, 512, true>
            <<<static_cast<unsigned>((rows + 15) / 16), 512, 0, stream>>>(x2, w2, g2, out2, d, rows,
                                                                          eps);
        return;
    case Instance::CtaMedium:
        rmsnorm_cta_bf16x2_kernel<Kind, 256, 6, true>
            <<<static_cast<unsigned>(rows), 256, 0, stream>>>(x2, w2, g2, out2, d, rows, eps);
        return;
    case Instance::CtaWide:
        if constexpr (is_gated<Kind>) {
            rmsnorm_cta_bf16x2_kernel<Kind, 512, 8, false>
                <<<static_cast<unsigned>(rows), 512, 0, stream>>>(x2, w2, g2, out2, d, rows, eps);
            return;
        }
        break;
    case Instance::CtaWidePrefetch:
        rmsnorm_cta_bf16x2_kernel<Kind, 512, 8, true>
            <<<static_cast<unsigned>(rows), 512, 0, stream>>>(x2, w2, g2, out2, d, rows, eps);
        return;
    case Instance::Generic:
        rmsnorm_generic_kernel<Kind><<<static_cast<unsigned>(rows), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            gate ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr,
            static_cast<__nv_bfloat16*>(out.data), d, rows, eps);
        return;
    }
    throw std::logic_error("rmsnorm: instance does not support selected epilogue");
}

} // namespace

void launch(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps, Epilogue kind,
            Tensor& out, Operands operands, Instance instance, cudaStream_t stream) {
    switch (kind) {
    case Epilogue::Plain:
        launch_instance<Epilogue::Plain>(x, weight, gate, out, operands, eps, instance, stream);
        break;
    case Epilogue::Offset:
        launch_instance<Epilogue::Offset>(x, weight, gate, out, operands, eps, instance, stream);
        break;
    case Epilogue::SiluGated:
        launch_instance<Epilogue::SiluGated>(x, weight, gate, out, operands, eps, instance, stream);
        break;
    case Epilogue::SigmoidGated:
        launch_instance<Epilogue::SigmoidGated>(x, weight, gate, out, operands, eps, instance,
                                                stream);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::rmsnorm
