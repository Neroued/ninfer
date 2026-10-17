#pragma once

#include <optional>

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ninfer/ops/qsa.h"

#include <memory>
#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

struct IndexerFixture {
    DeviceBuffer query_data;
    DeviceBuffer key_data;
    DeviceBuffer positions;
    DeviceBuffer valid;
    DeviceBuffer table_rows;
    DeviceBuffer tables;
    DeviceBuffer ids;
    DeviceBuffer counts;
    std::vector<int> contexts, lengths;
    Tensor query;
    ops::QsaIndexCacheView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::QsaIndexerExecutionEnvelope envelope;
    DeviceBuffer workspace_storage;
    std::optional<MemoryArena> workspace;
    std::size_t workspace_bytes;

    IndexerFixture(const Options& o, int tokens, CudaExecutionView execution)
        : contexts(broadcast(o.contexts, o.batch)),
          lengths(o.valid == "dense" ? std::vector<int>(o.batch, tokens)
                                     : broadcast(measurement::integers(o.valid, 0), o.batch)),
          envelope{std::uint32_t(o.context_capacity_explicit
                                     ? o.context_capacity
                                     : *std::max_element(contexts.begin(), contexts.end()))} {
        if (envelope.max_visible_tokens > 262144)
            throw std::invalid_argument("indexer context capacity exceeds 262144");
        const int B = o.batch, logical = std::max(1, (int(envelope.max_visible_tokens) + 63) / 64);
        int pages = 2;
        for (int b = 0; b < B; ++b) {
            if (contexts[b] > int(envelope.max_visible_tokens) || lengths[b] > tokens ||
                lengths[b] > contexts[b])
                throw std::invalid_argument("invalid indexer context/valid length");
            pages += (contexts[b] + 63) / 64;
        }
        const int row_stride  = o.layout == "unaligned" ? 129 : 128;
        const int page_stride = 16 * row_stride * (o.layout == "layered" ? 2 : 1);
        const int leading     = o.layout == "unaligned" ? 1 : 0;
        query_data            = make_bf16(std::size_t(512) * tokens * B, 101);
        if (o.score_pattern == "ties") query_data.fill(0);
        key_data = make_bf16(std::size_t(page_stride) * pages + leading, 103);
        std::vector<int> position(tokens * B, -1), selectors(B), table(logical * B, -1),
            permutation(pages);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937 random(313);
        std::shuffle(permutation.begin(), permutation.end(), random);
        int physical = 0;
        for (int b = 0; b < B; ++b) {
            selectors[b] = lengths[b] ? B - 1 - b : -1;
            for (int t = 0; t < lengths[b]; ++t)
                position[t + tokens * b] = contexts[b] - lengths[b] + t;
            for (int p = 0; p < (contexts[b] + 63) / 64; ++p)
                table[p + logical * (B - 1 - b)] = permutation[physical++];
        }
        positions  = measurement::make_i32(position);
        valid      = measurement::make_i32(lengths);
        table_rows = measurement::make_i32(selectors);
        tables     = measurement::make_i32(table);
        ids        = make_zeros(std::size_t(512) * tokens * B * 4);
        counts     = make_zeros(std::size_t(tokens) * B * 4);
        query      = Tensor(query_data.span(), DType::BF16, {128, 4, tokens, B});
        Tensor keys(key_data.span().subspan(leading * 2), DType::BF16, {128, 16, 1, pages});
        keys.nb[1]      = row_stride * 2;
        keys.nb[2]      = 16 * keys.nb[1];
        keys.nb[3]      = page_stride * 2;
        cache           = {keys, Tensor(tables.span(), DType::I32, {logical, B})};
        batch           = {Tensor(positions.span(), DType::I32, {tokens, B}),
                           Tensor(valid.span(), DType::I32, {B}), Tensor(table_rows.span(), DType::I32, {B})};
        selected        = {Tensor(ids.span(), DType::I32, {512, tokens, B}),
                           Tensor(counts.span(), DType::I32, {tokens, B})};
        workspace_bytes = ops::qsa_indexer_workspace_capacity_bytes(tokens, tokens, B, envelope);
        workspace_storage =
            DeviceBuffer(execution.device(), std::max<std::size_t>(256, workspace_bytes));
        workspace.emplace(workspace_storage.span());
    }

    void launch(CudaExecutionView execution) {
        ops::qsa_indexer(query, cache, batch, envelope, *workspace, selected, execution);
    }
};
} // namespace ninfer::bench::qsa
