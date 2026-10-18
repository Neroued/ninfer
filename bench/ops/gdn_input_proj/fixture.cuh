#pragma once

#include "core/buffer.h"
#include "options.h"
#include "quantized_weight.cuh"
#include <memory>

namespace ninfer::bench::gdn_input {
struct Weights {
    explicit Weights(const Options& o) {
        if (o.format == "q4q5") {
            first  = make_row_split_weight(QType::Q4_G64_FP16, 4096, o.k, o.k, 501U);
            second = make_row_split_weight(QType::Q5_G64_FP16, 12288, o.k, o.k, 503U);
            native = ops::PairedProjectionWeights{first.weight, second.weight};
        } else {
            if (o.format == "q8")
                first = make_row_split_weight(QType::Q8_G32_FP16, o.n, o.k, o.k, 501U);
            else if (o.format == "fp8")
                first = make_fp8_weight(o.n, o.k, 501U);
            else
                first = make_nvfp4_weight(o.n, o.k, 501U);
            native = ops::SingleProjectionWeight{first.weight, policy(o)};
        }
    }

    PackedQuantizedWeight first, second;
    ops::ProjectionWeights native;
};

struct Fixture {
    Fixture(const Options& o, const ops::ProjectionWeights& weights, int t)
        : options(o), weights(weights), tokens(t), query_rows(o.query_rows),
          value_rows((o.n - 2 * query_rows) / 2), channels(2 * query_rows + value_rows),
          columns(t * o.batch), slots(columns + o.batch),
          x(make_bf16(std::size_t(o.k) * columns, 101U)),
          z(make_zeros(std::size_t(value_rows) * columns * 2)),
          conv(make_bf16(std::size_t(channels) * 4, 103U)),
          history(make_bf16(std::size_t(channels) * 3 * slots, 107U)),
          saved(o.op == "snapshot" ? std::make_unique<SavedBuffer>(history) : nullptr),
          q(make_zeros(std::size_t(query_rows) * columns * 2)),
          k(make_zeros(std::size_t(query_rows) * columns * 2)),
          v(make_zeros(std::size_t(value_rows) * columns * 2)),
          records(o.op == "record" ? make_zeros(std::size_t(channels) * columns * 2)
                                   : DeviceBuffer{}),
          capacity(workspace_capacity()), arena_storage(make_buffer(256 + capacity)),
          arena(arena_storage.span()) {
        (void)arena.alloc_bytes(256);
        std::vector<int> sources(o.batch), bases(o.batch);
        for (int b = 0; b < o.batch; ++b) {
            sources[b] = columns + b;
            bases[b]   = b * t;
        }
        source = measurement::make_i32(sources);
        base   = measurement::make_i32(bases);
        if (!o.valid.empty()) valid = measurement::make_i32(o.valid);
    }

    std::size_t workspace_capacity() const {
        if (options.op == "snapshot")
            return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                weights, options.batch, tokens, tokens);
        return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(weights, options.batch,
                                                                        tokens, tokens);
    }

    void restore(CudaExecutionView cuda) const {
        if (saved) saved->restore(cuda);
    }

    void launch(CudaExecutionView cuda) {
        const int b = options.batch;
        Tensor input(x.span(), DType::BF16, {options.k, tokens, b}),
            weight(conv.span(), DType::BF16, {channels, 4});
        Tensor state(history.span(), DType::BF16, {channels, 3, slots}),
            src(source.span(), DType::I32, {b}), dst(base.span(), DType::I32, {b});
        Tensor mask;
        if (valid.data()) mask = Tensor(valid.span(), DType::I32, {b});
        ops::GdnConvOutputs out{Tensor(q.span(), DType::BF16, {query_rows, tokens, b}),
                                Tensor(k.span(), DType::BF16, {query_rows, tokens, b}),
                                Tensor(v.span(), DType::BF16, {value_rows, tokens, b}),
                                Tensor(z.span(), DType::BF16, {value_rows, tokens, b})};
        if (options.op == "snapshot")
            ops::gdn_input_proj_conv_snapshot(input, weights, weight, {state, src, dst}, mask, out,
                                              arena, cuda);
        else {
            Tensor record(records.span(), DType::BF16, {channels, tokens, b});
            ops::gdn_input_proj_conv_record(input, weights, weight, {state, src}, mask, out, record,
                                            arena, cuda);
        }
    }

    const Options& options;
    const ops::ProjectionWeights& weights;
    int tokens, query_rows, value_rows, channels, columns, slots;
    DeviceBuffer x;
    DeviceBuffer z;
    DeviceBuffer conv;
    DeviceBuffer history;
    std::unique_ptr<SavedBuffer> saved;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer records;
    DeviceBuffer source;
    DeviceBuffer base;
    DeviceBuffer valid;
    std::size_t capacity;
    DeviceBuffer arena_storage;
    MemoryArena arena;
};
} // namespace ninfer::bench::gdn_input
