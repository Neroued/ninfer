#pragma once
#include "core/buffer.h"
#include "options.h"
#include "ninfer/ops/gdn_control.h"
#include <optional>

namespace ninfer::bench::gdn_control {
inline Weight weight(ConstMemorySpan storage, int n, int k) {
    Weight w{};
    const auto* data = storage.data;
    w.device         = storage.device;
    w.qtype          = QType::BF16;
    w.layout         = QuantLayout::Contiguous;
    w.payload = w.qdata = data;
    w.payload_bytes     = std::uint64_t(n) * k * 2;
    w.ndim              = 2;
    w.shape[0] = w.padded_shape[0] = w.n = n;
    w.shape[1] = w.padded_shape[1] = w.k = k;
    return w;
}

struct Weights {
    explicit Weights(const Options& o) {
        first =
            make_bf16(std::size_t(o.heads) * o.input_rows * (o.storage == Storage::Parent ? 2 : 1),
                      103, -0.015F, 0.015F);
        if (o.storage == Storage::Parent)
            native = ops::SingleProjectionWeight{weight(first.span(), 2 * o.heads, o.input_rows)};
        else {
            second = make_bf16(std::size_t(o.heads) * o.input_rows, 105, -0.015F, 0.015F);
            native = ops::PairedProjectionWeights{weight(first.span(), o.heads, o.input_rows),
                                                  weight(second.span(), o.heads, o.input_rows)};
        }
    }

    DeviceBuffer first;
    DeviceBuffer second;
    ops::ProjectionWeights native;
};

struct Fixture {
    Fixture(const Options& o, const ops::ProjectionWeights* w, int tokens)
        : options(o), weights(w), column_count(columns(tokens, o.batch)),
          input_rows(o.op == Op::Control ? o.heads : o.input_rows),
          count(std::size_t(input_rows) * column_count),
          capacity(o.op == Op::Control      ? 0
                   : o.op == Op::Projection ? ops::gdn_control_proj_workspace_capacity_bytes(
                                                  *w, column_count, column_count)
                                            : ops::gdn_norm_control_proj_workspace_capacity_bytes(
                                                  *w, column_count, column_count)),
          input(make_bf16(count, 101, -1.F, 1.F)), a_log(make_f32(o.heads, 111, -2.F, 1.F)),
          dt_bias(make_f32(o.heads, 113, -1.F, 1.F)),
          g(make_zeros(std::size_t(o.heads) * column_count * 4)),
          beta(make_zeros(std::size_t(o.heads) * column_count * 4)),
          x(input.span(), DType::BF16, {input_rows, column_count}),
          a_log_tensor(a_log.span(), DType::FP32, {o.heads}),
          dt_bias_tensor(dt_bias.span(), DType::FP32, {o.heads}),
          outputs{Tensor(g.span(), DType::FP32, {o.heads, column_count}),
                  Tensor(beta.span(), DType::FP32, {o.heads, column_count})} {
        if (o.op != Op::Control) {
            arena_storage = make_buffer(std::max<std::size_t>(capacity, 256));
            arena.emplace(arena_storage.span());
        }
        if (o.op == Op::Control) {
            gate       = make_bf16(count, 109, -4.F, 4.F);
            gate_input = Tensor(gate.span(), DType::BF16, {input_rows, column_count});
        } else if (o.op == Op::NormProjection) {
            norm_gain         = make_bf16(input_rows, 107, -0.25F, 0.25F);
            normalized        = make_zeros(count * 2);
            norm_weight       = Tensor(norm_gain.span(), DType::BF16, {input_rows});
            normalized_tensor = Tensor(normalized.span(), DType::BF16, {input_rows, column_count});
        }
    }

    void launch(CudaExecutionView e) {
        switch (options.op) {
        case Op::Control:
            ops::gdn_control(x, gate_input, a_log_tensor, dt_bias_tensor, outputs, e);
            return;
        case Op::Projection:
            ops::gdn_control_proj(x, *weights, a_log_tensor, dt_bias_tensor, outputs, *arena, e);
            return;
        case Op::NormProjection:
            ops::gdn_norm_control_proj(x, norm_weight, 1.0e-6F, *weights, a_log_tensor,
                                       dt_bias_tensor, {normalized_tensor, outputs}, *arena, e);
            return;
        }
    }

    const Options& options;
    const ops::ProjectionWeights* weights;
    int column_count, input_rows;
    std::size_t count, capacity;
    DeviceBuffer input;
    DeviceBuffer norm_gain;
    DeviceBuffer gate;
    DeviceBuffer a_log;
    DeviceBuffer dt_bias;
    DeviceBuffer normalized;
    DeviceBuffer g;
    DeviceBuffer beta;
    Tensor x, norm_weight, gate_input, a_log_tensor, dt_bias_tensor, normalized_tensor;
    ops::GdnControlOutputs outputs;
    DeviceBuffer arena_storage;
    std::optional<MemoryArena> arena;
};
} // namespace ninfer::bench::gdn_control
