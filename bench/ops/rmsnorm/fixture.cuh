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
          x(input.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch}),
          w(weight.span(), DType::BF16, {o.dimension}),
          z(gate.size() ? Tensor(gate.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch})
                        : Tensor{}),
          out(output.span(), DType::BF16, {o.dimension, o.heads, columns, o.batch}) {}

    void launch(CudaExecutionView e) {
        switch (options.op) {
        case Op::Plain:
            ops::rmsnorm(x, w, 1.0e-6F, false, out, e);
            return;
        case Op::Offset:
            ops::rmsnorm(x, w, 1.0e-6F, true, out, e);
            return;
        case Op::Silu:
            ops::silu_gated_rmsnorm(x, w, z, 1.0e-6F, out, e);
            return;
        case Op::Sigmoid:
            ops::sigmoid_gated_rmsnorm(x, w, z, 1.0e-6F, out, e);
            return;
        }
    }

    const Options& options;
    std::size_t count;
    DeviceBuffer input;
    DeviceBuffer weight;
    DeviceBuffer gate;
    DeviceBuffer output;
    Tensor x, w, z, out;
};
} // namespace ninfer::bench::rmsnorm
