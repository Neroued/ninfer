#pragma once
#include "core/buffer.h"
#include "options.h"
#include "ninfer/ops/gated_residual.h"

namespace ninfer::bench::hc {
inline constexpr float kEpsilon = 1.e-6F;

struct Fixture {
    Fixture(const Options& options, int tokens)
        : op(options.op), in_place(options.in_place), m(columns(tokens, options.batch)),
          input(make_bf16(std::size_t(kWidth) * m * (op == Op::ExpandNorm ? 1 : kStreams), 6101,
                          -1.F, 1.F)),
          block(combines(op) ? make_bf16(std::size_t(kWidth) * m, 6103, -0.75F, 0.75F)
                             : DeviceBuffer{}),
          weight(op != Op::Combine ? make_bf16(kWidth * kStreams, 6107, -0.25F, 0.25F)
                                   : DeviceBuffer{}),
          injection(combines(op) && op != Op::AddNorm ? make_f32(std::size_t(kStreams) * m, 6109,
                                                                 op == Op::AddThenNorm ? 1.F : 0.F,
                                                                 op == Op::AddThenNorm ? 1.F : 2.F)
                                                      : DeviceBuffer{}),
          backup(in_place ? make_zeros(input.size()) : DeviceBuffer{}),
          residual(op != Op::GroupedNorm && !in_place
                       ? make_zeros(std::size_t(kWidth) * kStreams * m * sizeof(__nv_bfloat16))
                       : DeviceBuffer{}),
          normalized(op != Op::Combine
                         ? make_zeros(std::size_t(kWidth) * kStreams * m * sizeof(__nv_bfloat16))
                         : DeviceBuffer{}) {
        h = op == Op::ExpandNorm ? Tensor(input.span(), DType::BF16, {kWidth, m})
                                 : Tensor(input.span(), DType::BF16, {kWidth, kStreams, m});
        if (block.size()) b = Tensor(block.span(), DType::BF16, {kWidth, m});
        if (weight.size()) w = Tensor(weight.span(), DType::BF16, {kWidth, kStreams});
        if (injection.size()) j = Tensor(injection.span(), DType::FP32, {kStreams, m});
        if (in_place || residual.size())
            r = Tensor(in_place ? input.span() : residual.span(), DType::BF16,
                       {kWidth, kStreams, m});
        if (normalized.size()) z = Tensor(normalized.span(), DType::BF16, {kWidth, kStreams, m});
        if (in_place)
            CUDA_CHECK(
                cudaMemcpy(backup.data(), input.data(), input.size(), cudaMemcpyDeviceToDevice));
    }

    void restore(CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        if (in_place)
            CUDA_CHECK(cudaMemcpyAsync(input.data(), backup.data(), input.size(),
                                       cudaMemcpyDeviceToDevice, stream));
    }

    std::size_t workspace_capacity() const { return 0; }

    std::size_t workspace_peak() const { return 0; }

    void launch(CudaExecutionView e) {
        switch (op) {
        case Op::ExpandNorm:
            ops::hc_expand_norm(h, w, kEpsilon, r, z, e);
            break;
        case Op::GroupedNorm:
            ops::hc_grouped_rmsnorm(h, w, kEpsilon, z, e);
            break;
        case Op::Combine:
            ops::hc_combine(h, b, j, r, e);
            break;
        case Op::CombineNorm:
            ops::hc_combine_norm(h, b, j, w, kEpsilon, r, z, e);
            break;
        case Op::AddNorm:
            ops::hc_combine_norm(h, b, w, kEpsilon, r, z, e);
            break;
        case Op::CombineThenNorm:
        case Op::AddThenNorm:
            ops::hc_combine(h, b, j, r, e);
            ops::hc_grouped_rmsnorm(r, w, kEpsilon, z, e);
            break;
        case Op::BlockMix:
        case Op::FinalMix:
            throw std::logic_error("Mix requires its projection fixture");
        }
    }

    Op op;
    bool in_place;
    int m;
    DeviceBuffer input;
    DeviceBuffer block;
    DeviceBuffer weight;
    DeviceBuffer injection;
    DeviceBuffer backup;
    DeviceBuffer residual;
    DeviceBuffer normalized;
    Tensor h, b, w, j, r, z;
};
} // namespace ninfer::bench::hc
