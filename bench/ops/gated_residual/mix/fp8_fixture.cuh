#pragma once
#include "../options.h"
#include "quantized_weight.cuh"
#include "ninfer/ops/gated_residual.h"
#include "bf16_fixture.cuh"

namespace ninfer::bench::hc {
struct Fp8MixWeights {
    Fp8MixWeights(Op op, ops::LinearPolicy p)
        : policy(p), down(make_fp8_weight(640, 20480, 6203, 2.F / std::sqrt(20480.F))),
          up(make_fp8_weight(20480, 640, 6207, 8.F / std::sqrt(640.F))),
          inject(op == Op::BlockMix ? make_bf16(std::size_t(4) * 20480, 6211,
                                                -1.F / std::sqrt(20480.F), 1.F / std::sqrt(20480.F))
                                    : DeviceBuffer{}),
          injection(inject.size() ? bf16_weight(inject, 4, 20480) : Weight{}) {}

    ops::LinearPolicy policy;
    PackedQuantizedWeight down, up;
    DeviceBuffer inject;
    Weight injection;

    ops::HcFp8MixWeights block() const {
        return {{down.weight, policy}, {up.weight, policy}, injection};
    }

    ops::HcFp8FinalMixWeights final() const { return {{down.weight, policy}, {up.weight, policy}}; }
};

struct Fp8MixFixture {
    static constexpr int width = 5120;

    Fp8MixFixture(const Options& options, int tokens, const Fp8MixWeights& w)
        : weights(w), op(options.op), m(columns(tokens, options.batch)),
          input(make_bf16(std::size_t(20480) * m, 6201, -1.F, 1.F)),
          output(make_zeros(std::size_t(5120) * m * 2)),
          injection(op == Op::BlockMix ? make_zeros(std::size_t(4) * m * 4) : DeviceBuffer{}),
          z(input.span(), DType::BF16, {5120, 4, m}), x(output.span(), DType::BF16, {5120, m}),
          j(injection.size() ? Tensor(injection.span(), DType::FP32, {4, m}) : Tensor{}),
          capacity(op == Op::BlockMix ? ops::hc_mix_workspace_capacity_bytes(w.block(), m, m)
                                      : ops::hc_mix_workspace_capacity_bytes(w.final(), m, m)),
          scratch(make_zeros(capacity)), arena(scratch.span()) {}

    void launch(CudaExecutionView e) {
        if (op == Op::BlockMix)
            ops::hc_mix(z, weights.block(), arena, x, j, e);
        else
            ops::hc_mix(z, weights.final(), arena, x, e);
    }

    void restore(CudaExecutionView) {}

    std::size_t workspace_capacity() const { return capacity; }

    std::size_t workspace_peak() const { return arena.peak_used(); }

    const Fp8MixWeights& weights;
    Op op;
    int m;
    DeviceBuffer input, output, injection;
    Tensor z, x, j;
    std::size_t capacity;
    DeviceBuffer scratch;
    MemoryArena arena;
};
} // namespace ninfer::bench::hc
