#pragma once

#include <optional>

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/qsa.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <memory>
#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

// Every storage profile starts with the same represented BF16 source values. Existing row
// codecs initialize persistent bytes outside timing; the benchmark owns plane copying.
__global__ void initialize_sparse_source(__nv_bfloat16* words, std::size_t count, bool value,
                                         int heads) {
    const unsigned seed = value ? 1311u : 701u;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const auto token = i / (256 * heads);
        unsigned x = unsigned((token / 64 * heads + i / 256 % heads) * 64 * 256 + token % 64 * 256 +
                              i % 256) +
                     seed;
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        words[i] = __float2bfloat16_rn((int(x % 10001) - 5000) / 3000.0f);
    }
}

__global__ void copy_sparse_plane(const std::uint8_t* src, std::size_t src_record_stride,
                                  std::uint8_t* dst, std::size_t dst_record_stride, int row_bytes,
                                  std::size_t bytes, int heads) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const auto row         = i / row_bytes;
        const auto encoded_row = heads == 1 ? row / 128 * 128 + row % 2 * 64 + row / 2 % 64 : row;
        dst[row / (64 * heads) * dst_record_stride + row % (64 * heads) * row_bytes +
            i % row_bytes]     = src[encoded_row / 128 * src_record_stride +
                                 encoded_row % 128 * row_bytes + i % row_bytes];
    }
}

inline std::vector<int> sparse_contexts(const Options& o, int tokens) {
    if (o.prefixes.empty()) return broadcast(o.contexts, o.batch);
    auto result      = broadcast(o.prefixes, o.batch);
    const auto valid = o.valid == "dense" ? std::vector<int>(o.batch, tokens)
                                          : broadcast(measurement::integers(o.valid, 0), o.batch);
    for (int b = 0; b < o.batch; ++b) {
        if (valid[b] > 262144 || result[b] > 262144 - valid[b])
            throw std::invalid_argument("prefix/valid exceeds maximum context");
        result[b] += valid[b];
    }
    return result;
}

struct SparseGqaFixture {
    DeviceBuffer query_data;
    DeviceBuffer gate_data;
    DeviceBuffer output_data;
    std::unique_ptr<tool::PackedKVFixture> packed;
    DeviceBuffer positions;
    DeviceBuffer valid;
    DeviceBuffer table_rows;
    DeviceBuffer tables;
    DeviceBuffer ids;
    DeviceBuffer counts;
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
        : contexts(sparse_contexts(o, tokens)),
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
        query_data  = make_bf16(std::size_t(256 * o.query_heads) * tokens * B, 101, -1.5f, 1.5f);
        gate_data   = make_bf16(std::size_t(256 * o.query_heads) * tokens * B, 103, -4.f, 4.f);
        output_data = make_zeros(query_data.size());
        const auto layout = paged_kv_storage_layout(o.storage, 256);
        token_bytes       = o.kv_heads * layout.physical_bytes_per_token_head();
        KVPageGeometry geometry;
        const auto add = [&](DType dtype, int features) {
            if (features) geometry.planes.push_back({dtype, features, o.kv_heads, 256, 1});
        };
        add(layout.key.data_dtype, layout.key.data_leading_extent);
        add(layout.value.data_dtype, layout.value.data_leading_extent);
        add(layout.key.scale_dtype, layout.key.scale_leading_extent);
        add(layout.value.scale_dtype, layout.value.scale_leading_extent);
        packed = std::make_unique<tool::PackedKVFixture>(execution.device(), geometry, pages);
        {
            const int rows = pages * 64;
            DeviceBuffer raw_key(ninfer::bench::device_context().device(),
                                 std::size_t(rows) * 256 * o.kv_heads * 2);
            DeviceBuffer raw_value(ninfer::bench::device_context().device(), raw_key.size());
            initialize_sparse_source<<<1024, 256, 0, execution.stream()>>>(
                static_cast<__nv_bfloat16*>(raw_key.data()), raw_key.size() / 2, false, o.kv_heads);
            initialize_sparse_source<<<1024, 256, 0, execution.stream()>>>(
                static_cast<__nv_bfloat16*>(raw_value.data()), raw_value.size() / 2, true,
                o.kv_heads);
            std::vector<int> coordinates(rows);
            std::iota(coordinates.begin(), coordinates.end(), 0);
            auto source_positions    = measurement::make_i32(coordinates);
            const int encoded_tokens = rows * o.kv_heads / 2;
            const int encoded_pages  = (encoded_tokens + 63) / 64;
            auto encoded_geometry    = geometry;
            for (auto& plane : encoded_geometry.planes) plane.head_extent = 2;
            tool::PackedKVFixture encoded_cache(execution.device(), encoded_geometry,
                                                encoded_pages);
            std::vector<int> mapping(encoded_pages);
            std::iota(mapping.begin(), mapping.end(), 0);
            const auto encoded_offsets = encoded_cache.table_offsets(mapping);
            DeviceBuffer identity_table(execution.device(), encoded_offsets.size() * 8);
            identity_table.copy_from_host(encoded_offsets.data(), identity_table.size());
            Tensor table(identity_table.span(), DType::I64, {encoded_pages});
            const PagedKVLayerView encoded{
                encoded_cache.plane(0, table),
                encoded_cache.plane(1, table),
                layout.key.has_scale() ? encoded_cache.plane(2, table) : PagedPlaneView{},
                layout.value.has_scale() ? encoded_cache.plane(3, table) : PagedPlaneView{},
                256,
                2,
                o.storage};
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
            ops::kv_cache_append(Tensor(raw_key.span(), DType::BF16, {256, 2, encoded_tokens}),
                                 Tensor(raw_value.span(), DType::BF16, {256, 2, encoded_tokens}),
                                 Tensor(source_positions.span(), DType::I32, {encoded_tokens}),
                                 encoded, execution);
            for (std::size_t i = 0; i < geometry.planes.size(); ++i) {
                const auto& source  = encoded_cache.layout().planes[i];
                const auto& target  = packed->layout().planes[i];
                const int row_bytes = target.row_stride_bytes();
                copy_sparse_plane<<<1024, 256, 0, execution.stream()>>>(
                    static_cast<const std::uint8_t*>(encoded_cache.storage().data) +
                        encoded_cache.physical_offset(0) + source.tensor.region.offset,
                    encoded_cache.record_stride(),
                    static_cast<std::uint8_t*>(packed->storage().data) +
                        packed->physical_offset(0) + target.tensor.region.offset,
                    packed->record_stride(), row_bytes, std::size_t(rows) * o.kv_heads * row_bytes,
                    o.kv_heads);
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
        positions                = measurement::make_i32(pos);
        valid                    = measurement::make_i32(lengths);
        table_rows               = measurement::make_i32(selectors);
        const auto table_offsets = packed->table_offsets(table);
        tables                   = DeviceBuffer(execution.device(), table_offsets.size() * 8);
        tables.copy_from_host(table_offsets.data(), tables.size());
        ids    = measurement::make_i32(block_ids);
        counts = measurement::make_i32(block_counts);
        query  = Tensor(query_data.span(), DType::BF16, {256, o.query_heads, tokens, B});
        gate   = Tensor(gate_data.span(), DType::BF16, {256, o.query_heads, tokens, B});
        out    = Tensor(output_data.span(), DType::BF16, {256, o.query_heads, tokens, B});
        Tensor page_offsets(tables.span(), DType::I64, {logical, B});
        cache           = {packed->plane(0, page_offsets),
                           packed->plane(1, page_offsets),
                 layout.key.has_scale() ? packed->plane(2, page_offsets) : PagedPlaneView{},
                 layout.value.has_scale() ? packed->plane(3, page_offsets) : PagedPlaneView{},
                 256,
                           o.kv_heads,
                           o.storage};
        batch           = {Tensor(positions.span(), DType::I32, {tokens, B}),
                           Tensor(valid.span(), DType::I32, {B}), Tensor(table_rows.span(), DType::I32, {B})};
        selected        = {Tensor(ids.span(), DType::I32, {512, tokens, B}),
                           Tensor(counts.span(), DType::I32, {tokens, B})};
        workspace_bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
            cache.storage, o.query_heads, o.kv_heads, tokens, tokens, B, envelope,
            execution.multiprocessor_count());
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
