#pragma once

#include "core/buffer.h"
#include "options.h"
#include "direct_bf16_weight.cuh"
#include "quantized_weight.cuh"

#include <numeric>

namespace ninfer::bench::qsa {

inline constexpr int kInputRows = 2560;

struct Weights {
    std::vector<int> rows;
    int parent_rows;
    DeviceBuffer storage;
    Weight weight;

    Weights(Projection projection, int kv_heads)
        : rows(segments(projection, kv_heads)),
          parent_rows(std::accumulate(rows.begin(), rows.end(), 0)) {
        if (projection == Projection::QueryGate) {
            auto packed = make_row_split_weight(QType::Q8_G32_FP16, parent_rows, kInputRows,
                                                kInputRows, 701U);
            weight      = packed.weight;
            storage     = std::move(packed.storage);
        } else {
            auto packed = make_direct_bf16_weight(parent_rows, kInputRows, 701U);
            weight      = packed.weight;
            storage     = std::move(packed.storage);
        }
    }
};

struct Fixture {
    DeviceBuffer input;
    DeviceBuffer output;
    Tensor x, packed;
    std::vector<Tensor> split;

    Fixture(const Weights& weights, int tokens)
        : input(make_bf16(std::size_t(kInputRows) * tokens, 101U)),
          output(make_zeros(std::size_t(weights.parent_rows) * tokens * 2)),
          x(input.span(), DType::BF16, {kInputRows, tokens}),
          packed(output.span(), DType::BF16, {weights.parent_rows, tokens}) {
        std::size_t offset = 0;
        for (int rows : weights.rows) {
            const auto bytes = std::size_t(rows) * tokens * sizeof(std::uint16_t);
            split.emplace_back(output.span().subspan(offset, bytes), DType::BF16,
                               std::initializer_list<std::int32_t>{rows, tokens});
            offset += bytes;
        }
    }
};

} // namespace ninfer::bench::qsa
