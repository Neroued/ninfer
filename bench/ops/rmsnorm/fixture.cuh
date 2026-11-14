#pragma once
#include "core/buffer.h"
#include "options.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::bench::rmsnorm {
struct Fixture {
    Fixture(const Options& o, int columns)
        : options(o),
          count(Tensor::metadata(DType::BF16, {o.dimension, o.heads, columns, o.batch}).numel()),
          input(make_bf16(count, 101, -1.F, 1.F)),
          weight(make_bf16(o.dimension, 107, -0.25F, 0.25F)),
          gate(o.op == Op::Silu || o.op == Op::Sigmoid ? make_bf16(count, 109, -4.F, 4.F)
                                                       : DeviceBuffer{}),
          output(make_zeros(count * 2)),
          inverse_output(writes_inverse(o.op) ? make_zeros(count / o.dimension * sizeof(float))
                                              : DeviceBuffer{}),
          x(input.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch}),
          w(weight.span(), DType::BF16, {o.dimension}),
          z(gate.size() ? Tensor(gate.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch})
                        : Tensor{}),
          out(output.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch}),
          inverse(inverse_output.size()
                      ? Tensor(inverse_output.span(), DType::FP32, {o.heads, columns, o.batch})
                      : Tensor{}) {}

    void reset_columns(int columns) {
        const auto& o = options;
        count = Tensor::metadata(DType::BF16, {o.dimension, o.heads, columns, o.batch}).numel();
        if (count * 2 > input.size()) throw std::invalid_argument("RMSNorm fixture capacity");
        x = Tensor(input.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch});
        if (gate.size())
            z = Tensor(gate.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch});
        out = Tensor(output.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch});
        if (inverse_output.size())
            inverse = Tensor(inverse_output.span(), DType::FP32, {o.heads, columns, o.batch});
    }

    void launch(CudaExecutionView e) {
        switch (options.op) {
        case Op::Plain:
            ops::rmsnorm(x, w, options.epsilon, false, out, e);
            return;
        case Op::Offset:
            ops::rmsnorm(x, w, options.epsilon, true, out, e);
            return;
        case Op::PlainInverse:
            ops::rmsnorm(x, w, options.epsilon, false, out, inverse, e);
            return;
        case Op::OffsetInverse:
            ops::rmsnorm(x, w, options.epsilon, true, out, inverse, e);
            return;
        case Op::Silu:
            ops::silu_gated_rmsnorm(x, w, z, options.epsilon, out, e);
            return;
        case Op::Sigmoid:
            ops::sigmoid_gated_rmsnorm(x, w, z, options.epsilon, out, e);
            return;
        }
    }

    const Options& options;
    std::size_t count;
    DeviceBuffer input;
    DeviceBuffer weight;
    DeviceBuffer gate;
    DeviceBuffer output;
    DeviceBuffer inverse_output;
    Tensor x, w, z, out, inverse;
};
} // namespace ninfer::bench::rmsnorm
