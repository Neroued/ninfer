#pragma once

#include <optional>

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/qsa.h"

#include <memory>
#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

// Every storage profile starts with the same represented BF16 source values. The public append
// codec initializes persistent bytes outside the timer; the benchmark owns only plane copying.
__global__ void initialize_sparse_source(__nv_bfloat16* words, std::size_t count, bool value) {
    const unsigned seed = value ? 1311u : 701u;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const auto token = i / 512;
        unsigned x =
            unsigned((token / 64 * 2 + i / 256 % 2) * 64 * 256 + token % 64 * 256 + i % 256) + seed;
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        words[i] = __float2bfloat16_rn((int(x % 10001) - 5000) / 3000.0f);
    }
}

__global__ void copy_sparse_plane(const std::uint8_t* src, std::uint8_t* dst, int row_bytes,
                                  std::int64_t row_stride, std::int64_t head_stride,
                                  std::int64_t page_stride, std::size_t bytes) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const auto row     = i / row_bytes;
        dst[row / 128 * page_stride + row / 64 % 2 * head_stride + row % 64 * row_stride +
            i % row_bytes] = src[i];
    }
}

struct SparseGqaFixture {
    DeviceBuffer query_data;
    DeviceBuffer gate_data;
    DeviceBuffer output_data;
    DeviceBuffer key_data;
    DeviceBuffer value_data;
    DeviceBuffer positions;
    DeviceBuffer valid;
    DeviceBuffer table_rows;
    DeviceBuffer tables;
    DeviceBuffer ids;
    DeviceBuffer counts;
    DeviceBuffer key_scales;
    DeviceBuffer value_scales;
    std::vector<int> contexts, lengths;
    Tensor query, gate, out;
    PagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::QsaSparseGqaExecutionEnvelope envelope;
    DeviceBuffer workspace_storage;
    std::optional<MemoryArena> workspace;
    std::size_t workspace_bytes, selected_tokens = 0, unique_tokens = 0, token_bytes = 0;
    std::size_t active_queries = 0, selected_ids = 0, table_entries = 0, active_rows = 0;

    SparseGqaFixture(const Options& o, int tokens, CudaExecutionView execution)
        : contexts(broadcast(o.contexts, o.batch)),
          lengths(o.valid == "dense" ? std::vector<int>(o.batch, tokens)
                                     : broadcast(measurement::integers(o.valid, 0), o.batch)) {
        envelope.max_visible_tokens = o.context_capacity_explicit
                                          ? o.context_capacity
                                          : *std::max_element(contexts.begin(), contexts.end());
        const int B                 = o.batch,
                  logical =
                      std::max(1, (*std::max_element(contexts.begin(), contexts.end()) + 63) / 64);
        int pages = 2;
        for (int b = 0; b < B; ++b) {
            if (contexts[b] > int(envelope.max_visible_tokens) || lengths[b] > tokens ||
                lengths[b] > contexts[b])
                throw std::invalid_argument("invalid sparse context/valid length");
            pages += (contexts[b] + 63) / 64;
        }
        query_data          = make_bf16(std::size_t(6144) * tokens * B, 101, -1.5f, 1.5f);
        gate_data           = make_bf16(std::size_t(6144) * tokens * B, 103, -4.f, 4.f);
        output_data         = make_zeros(query_data.size());
        const auto layout   = paged_kv_storage_layout(o.storage, 256);
        token_bytes         = 2 * layout.physical_bytes_per_token_head();
        const auto allocate = [&](DeviceBuffer& storage, DType dtype, int features) {
            if (!features) return Tensor{};
            const int element = dtype_size(dtype);
            const int leading = o.layout == "unaligned" ? element : 0;
            const int row     = features * element + leading;
            const int head = 64 * row, page = 2 * head * (o.layout == "layered" ? 2 : 1);
            storage = DeviceBuffer(ninfer::bench::device_context().device(),
                                   std::size_t(page) * pages + leading);
            Tensor view(storage.span().subspan(leading), dtype, {features, 64, 2, pages});
            view.nb[1] = row;
            view.nb[2] = head;
            view.nb[3] = page;
            return view;
        };
        const auto key = allocate(key_data, layout.key.data_dtype, layout.key.data_leading_extent);
        const auto value =
            allocate(value_data, layout.value.data_dtype, layout.value.data_leading_extent);
        const auto ks =
            allocate(key_scales, layout.key.scale_dtype, layout.key.scale_leading_extent);
        const auto vs =
            allocate(value_scales, layout.value.scale_dtype, layout.value.scale_leading_extent);
        {
            const int rows = pages * 64;
            DeviceBuffer raw_key(ninfer::bench::device_context().device(),
                                 std::size_t(rows) * 512 * 2);
            DeviceBuffer raw_value(ninfer::bench::device_context().device(), raw_key.size());
            initialize_sparse_source<<<1024, 256, 0, execution.stream()>>>(
                static_cast<__nv_bfloat16*>(raw_key.data()), raw_key.size() / 2, false);
            initialize_sparse_source<<<1024, 256, 0, execution.stream()>>>(
                static_cast<__nv_bfloat16*>(raw_value.data()), raw_value.size() / 2, true);
            std::vector<int> coordinates(rows), mapping(pages);
            std::iota(coordinates.begin(), coordinates.end(), 0);
            std::iota(mapping.begin(), mapping.end(), 0);
            auto source_positions = measurement::make_i32(coordinates),
                 identity_table   = measurement::make_i32(mapping);
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
            Tensor contiguous[4];
            DeviceBuffer buffers[4];
            const Tensor planes[]{key, value, ks, vs};
            for (int i = 0; i < 4; ++i) {
                if (!planes[i].data) continue;
                buffers[i] =
                    DeviceBuffer(ninfer::bench::device_context().device(), planes[i].bytes());
                contiguous[i] =
                    Tensor(buffers[i].span(), planes[i].dtype, {planes[i].ne[0], 64, 2, pages});
            }
            ops::kv_cache_append(Tensor(raw_key.span(), DType::BF16, {256, 2, rows}),
                                 Tensor(raw_value.span(), DType::BF16, {256, 2, rows}),
                                 Tensor(source_positions.span(), DType::I32, {rows}),
                                 {contiguous[0], contiguous[1], contiguous[2], contiguous[3],
                                  Tensor(identity_table.span(), DType::I32, {pages}), 256, 2,
                                  o.storage},
                                 execution);
            for (int i = 0; i < 4; ++i) {
                if (!planes[i].data) continue;
                copy_sparse_plane<<<1024, 256, 0, execution.stream()>>>(
                    static_cast<const std::uint8_t*>(contiguous[i].data),
                    static_cast<std::uint8_t*>(planes[i].data),
                    planes[i].ne[0] * dtype_size(planes[i].dtype), planes[i].nb[1], planes[i].nb[2],
                    planes[i].nb[3], contiguous[i].bytes());
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        }
        std::vector<int> pos(tokens * B, -1), selectors(B, -1), table(logical * B, -1),
            physical(pages), block_ids(std::size_t(512) * tokens * B, -1), block_counts(tokens * B);
        std::iota(physical.begin(), physical.end(), 0);
        std::mt19937 random(731);
        std::shuffle(physical.begin(), physical.end(), random);
        int next = 0;
        for (int b = 0; b < B; ++b) {
            selectors[b] = lengths[b] ? B - 1 - b : -1;
            for (int p = 0; p < (contexts[b] + 63) / 64; ++p)
                table[(B - 1 - b) * logical + p] = physical[next++];
            std::vector<int> order(contexts[b] / 4);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), random);
            std::vector<int> random_order(order.size());
            std::iota(random_order.begin(), random_order.end(), 0);
            std::vector<bool> seen(contexts[b]);
            for (int t = 0; t < lengths[b]; ++t) {
                const int m = b * tokens + t, p = contexts[b] - lengths[b] + t, C = (p + 1) / 4,
                          c     = std::min(o.selected_blocks, C);
                pos[m]          = p;
                block_counts[m] = c;
                std::vector<int> chosen;
                if (o.selection == "shared") {
                    for (int g : order) {
                        if (g < C && int(chosen.size()) < c) chosen.push_back(g);
                        if (int(chosen.size()) == c) break;
                    }
                } else if (o.selection == "contiguous") {
                    for (int i = 0; i < c; ++i) chosen.push_back(C - 1 - i);
                } else {
                    // The prefix remains a permutation of [0,C) as causal C grows.
                    // Partial Fisher-Yates gives an ordered uniform sample without replacement.
                    for (int i = 0; i < c; ++i) {
                        const int j = std::uniform_int_distribution<int>(i, C - 1)(random);
                        std::swap(random_order[i], random_order[j]);
                        chosen.push_back(random_order[i]);
                    }
                }
                for (int i = 0; i < c; ++i) {
                    block_ids[std::size_t(m) * 512 + i] = chosen[i];
                    for (int j = 0; j < 4; ++j) seen[chosen[i] * 4 + j] = true;
                }
                for (int j = C * 4; j <= p; ++j) seen[j] = true;
                const auto support = c * 4 + (p + 1) % 4;
                selected_tokens += support;
                active_queries += support != 0;
                // A full selection is known from its count without inspecting ids.
                if (c < C) selected_ids += c;
            }
            unique_tokens += std::count(seen.begin(), seen.end(), true);
            bool active = false;
            for (std::size_t first = 0; first < seen.size(); first += 64) {
                const bool used = std::any_of(seen.begin() + first,
                                              seen.begin() + std::min(first + 64, seen.size()),
                                              [](bool v) { return v; });
                table_entries += used;
                active |= used;
            }
            active_rows += active;
        }
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        positions  = measurement::make_i32(pos);
        valid      = measurement::make_i32(lengths);
        table_rows = measurement::make_i32(selectors);
        tables     = measurement::make_i32(table);
        ids        = measurement::make_i32(block_ids);
        counts     = measurement::make_i32(block_counts);
        query      = Tensor(query_data.span(), DType::BF16, {256, 24, tokens, B});
        gate       = Tensor(gate_data.span(), DType::BF16, {256, 24, tokens, B});
        out        = Tensor(output_data.span(), DType::BF16, {256, 24, tokens, B});
        cache      = {key, value, ks,       vs, Tensor(tables.span(), DType::I32, {logical, B}),
                      256, 2,     o.storage};
        batch      = {Tensor(positions.span(), DType::I32, {tokens, B}),
                      Tensor(valid.span(), DType::I32, {B}), Tensor(table_rows.span(), DType::I32, {B})};
        selected   = {Tensor(ids.span(), DType::I32, {512, tokens, B}),
                      Tensor(counts.span(), DType::I32, {tokens, B})};
        workspace_bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
            cache.storage, tokens, tokens, B, envelope, execution.multiprocessor_count());
        workspace_storage =
            DeviceBuffer(execution.device(), std::max<std::size_t>(256, workspace_bytes));
        workspace.emplace(workspace_storage.span());
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void launch(CudaExecutionView execution) {
        ops::qsa_sparse_gqa(query, gate, cache, batch, selected, envelope, *workspace, out,
                            execution);
    }
};
} // namespace ninfer::bench::qsa
