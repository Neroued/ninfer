#pragma once

#include "core/buffer.h"
#include "options.h"
#include "ninfer/ops/gated_residual.h"

namespace ninfer::bench::hc {
inline Weight bf16_weight(const DeviceBuffer& buffer, int n, int k) {
    Weight w;
    w.device  = buffer.device();
    w.payload = w.qdata = buffer.data();
    w.payload_bytes     = buffer.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

// The represented weights remain resident while M varies, as they do in a Program.
struct MixWeights {
    explicit MixWeights(Op form)
        : op(form),
          down(make_bf16(
              std::size_t(kRank + (op == Op::BlockMix ? kStreams : 0)) * kWidth * kStreams, 6203,
              -std::sqrt(3.F / (kWidth * kStreams)), std::sqrt(3.F / (kWidth * kStreams)))),
          up(make_bf16(std::size_t(kWidth) * kStreams * kRank, 6207, -4.F * std::sqrt(3.F / kRank),
                       4.F * std::sqrt(3.F / kRank))),
          wd(bf16_weight(down, kRank + (op == Op::BlockMix ? kStreams : 0), kWidth * kStreams)),
          wu(bf16_weight(up, kWidth * kStreams, kRank)) {}

    Op op;
    DeviceBuffer down, up;
    Weight wd, wu;
};

struct MixFixture {
    MixFixture(const Options& options, int tokens, const MixWeights& resident)
        : weights(resident), op(options.op), m(columns(tokens, options.batch)),
          input(make_bf16(std::size_t(kWidth) * kStreams * m, 6201, -1.F, 1.F)),
          output(make_zeros(std::size_t(kWidth) * m * sizeof(__nv_bfloat16))),
          injection(op == Op::BlockMix ? make_zeros(std::size_t(kStreams) * m * sizeof(float))
                                       : DeviceBuffer{}),
          z(input.span(), DType::BF16, {kWidth, kStreams, m}),
          x(output.span(), DType::BF16, {kWidth, m}),
          j(injection.size() ? Tensor(injection.span(), DType::FP32, {kStreams, m}) : Tensor{}),
          capacity(op == Op::BlockMix ? ops::hc_mix_workspace_capacity_bytes(
                                            ops::HcMixWeights{weights.wd, weights.wu}, m, m)
                                      : ops::hc_mix_workspace_capacity_bytes(
                                            ops::HcFinalMixWeights{weights.wd, weights.wu}, m, m)),
          arena_storage(make_buffer(capacity)), arena(arena_storage.span()) {}

    void launch(CudaExecutionView execution) {
        if (op == Op::BlockMix)
            ops::hc_mix(z, ops::HcMixWeights{weights.wd, weights.wu}, arena, x, j, execution);
        else
            ops::hc_mix(z, ops::HcFinalMixWeights{weights.wd, weights.wu}, arena, x, execution);
    }

    void restore(CudaExecutionView) {}

    std::size_t workspace_capacity() const { return capacity; }

    std::size_t workspace_peak() const { return arena.peak_used(); }

    const MixWeights& weights;
    Op op;
    int m;
    DeviceBuffer input;
    DeviceBuffer output;
    DeviceBuffer injection;
    Tensor z, x, j;
    std::size_t capacity;
    DeviceBuffer arena_storage;
    MemoryArena arena;
};
} // namespace ninfer::bench::hc
