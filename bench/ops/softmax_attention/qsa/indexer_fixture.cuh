#pragma once

#include <optional>

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ragged_fixture.cuh"
#include "ninfer/ops/qsa.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <memory>
#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

struct IndexerFixture {
    DeviceBuffer query_data;
    std::unique_ptr<tool::PackedKVFixture> keys;
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
    ops::ContextBucket bucket;
    int shared_prefix_tokens = 0;
    const RaggedBatchFixture* ragged_batch;
    DeviceBuffer workspace_storage;
    std::optional<MemoryArena> workspace;
    std::size_t workspace_bytes;

    IndexerFixture(const Options& o, int tokens, CudaExecutionView execution,
                   const RaggedBatchFixture* packed_batch)
        : ragged_batch(packed_batch) {
        const int B     = packed_batch ? 1 : o.batch;
        int max_visible = 0;
        std::optional<RaggedPageMap> mapping;
        if (packed_batch) {
            contexts    = packed_batch->ends;
            lengths     = packed_batch->spec.lengths;
            max_visible = *std::max_element(contexts.begin(), contexts.end());
            const int S = static_cast<int>(lengths.size());
            bucket      = ragged_bucket(o, max_visible);
            workspace_bytes =
                ops::qsa_indexer_ragged_workspace_capacity_bytes(bucket, tokens, tokens, S, S);
            shared_prefix_tokens = o.shared_prefix_tokens;
            mapping.emplace(*packed_batch, shared_prefix_tokens, false, 313);
        } else {
            contexts = broadcast(o.contexts, B);
            if (o.explicit_valid) {
                lengths = o.valid == "dense" ? std::vector<int>(B, tokens)
                                             : broadcast(measurement::integers(o.valid, 0), B);
            } else {
                lengths = workload::generate_batch_workload(
                              tokens, tokens, B, workload::LengthDist::Uniform, o.valid_dist)
                              .valid_columns;
            }
            if (o.contexts.size() == 1 && B > 1 && o.contexts.front() > 0) {
                const int target     = o.contexts.front();
                const auto generated = workload::generate_batch_workload(
                    target, 1, B, o.length_dist, workload::ValidDist::Full);
                shared_prefix_tokens = generated.shared_prefix_tokens;
                for (int b = 0; b < B; ++b)
                    contexts[b] =
                        lengths[b] ? std::clamp(lengths[b], generated.row_contexts[b], target) : 0;
            }

            for (int b = 0; b < B; ++b) {
                if (lengths[b] > tokens || lengths[b] > contexts[b])
                    throw std::invalid_argument("invalid indexer context/valid length");
                if (lengths[b]) max_visible = std::max(max_visible, contexts[b]);
            }
            bucket          = ragged_bucket(o, max_visible);
            workspace_bytes = ops::qsa_indexer_workspace_capacity_bytes(bucket, B, tokens, tokens);
        }
        const int logical       = mapping ? mapping->columns : std::max(1, (max_visible + 63) / 64);
        const int table_height  = packed_batch ? packed_batch->table_rows() : B;
        const int shared_pages  = max_visible ? shared_prefix_tokens / 64 : 0;
        std::int64_t page_count = shared_pages + 2;
        if (mapping) {
            page_count = mapping->pages;
        } else {
            for (int b = 0; b < B; ++b)
                if (lengths[b]) page_count += (contexts[b] + 63) / 64 - shared_pages;
        }
        if (page_count > std::numeric_limits<int>::max())
            throw std::invalid_argument("too many indexer cache pages");
        const int pages       = static_cast<int>(page_count);
        const int row_stride  = 128;
        const int page_stride = 16 * row_stride;
        query_data            = make_bf16(std::size_t(512) * tokens * B, 101);
        if (o.score_pattern == "ties") query_data.fill(0);
        auto key_data = make_bf16(std::size_t(page_stride) * pages, 103);
        keys          = std::make_unique<tool::PackedKVFixture>(
            execution.device(), KVPageGeometry{64, {{DType::BF16, 128, 1, 256, 4}}}, pages);
        keys->upload_plane(0, Tensor(key_data.span(), DType::BF16, {128, 16, 1, pages}));
        std::vector<int> table;
        if (mapping) {
            table = std::move(mapping->table);
        } else {
            std::vector<int> position(std::size_t(tokens) * B, -1), selectors(B),
                permutation(pages);
            table.resize(std::size_t(logical) * B, -1);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::mt19937 random(313);
            std::shuffle(permutation.begin(), permutation.end(), random);
            int physical = shared_pages;
            for (int b = 0; b < B; ++b) {
                selectors[b] = lengths[b] ? B - 1 - b : -1;
                if (!lengths[b]) continue;
                for (int t = 0; t < lengths[b]; ++t)
                    position[t + std::size_t(tokens) * b] = contexts[b] - lengths[b] + t;
                for (int p = 0; p < (contexts[b] + 63) / 64; ++p)
                    table[p + std::size_t(logical) * (B - 1 - b)] =
                        permutation[p < shared_pages ? p : physical++];
            }
            positions  = measurement::make_i32(position);
            valid      = measurement::make_i32(lengths);
            table_rows = measurement::make_i32(selectors);
            batch      = {Tensor(positions.span(), DType::I32, {tokens, B}),
                          Tensor(valid.span(), DType::I32, {B}),
                          Tensor(table_rows.span(), DType::I32, {B})};
        }
        const auto offsets = keys->table_offsets(table);
        tables = DeviceBuffer(execution.device(), offsets.size() * sizeof(std::int64_t));
        tables.copy_from_host(offsets.data(), tables.size());
        ids      = make_zeros(std::size_t(512) * tokens * B * 4);
        counts   = make_zeros(std::size_t(tokens) * B * 4);
        query    = Tensor(query_data.span(), DType::BF16, {128, 4, tokens, B});
        cache    = {keys->plane(0, Tensor(tables.span(), DType::I64, {logical, table_height}))};
        selected = {Tensor(ids.span(), DType::I32, {512, tokens, B}),
                    Tensor(counts.span(), DType::I32, {tokens, B})};
        workspace_storage = DeviceBuffer(execution.device(), workspace_bytes);
        workspace.emplace(workspace_storage.span());
    }

    void launch(CudaExecutionView execution) {
        if (ragged_batch)
            ops::qsa_indexer(query, cache, ragged_batch->view, bucket, *workspace, selected,
                             execution);
        else
            ops::qsa_indexer(query, cache, batch, bucket, *workspace, selected, execution);
    }
};
} // namespace ninfer::bench::qsa
