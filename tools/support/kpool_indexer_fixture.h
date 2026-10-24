#pragma once

#include "kpool_fixture.h"
#include "paged_kv_fixture.h"
#include "ninfer/ops/kpool_mla_layout.h"

#include <memory>
#include <numeric>

namespace ninfer::tool::kpool {

// Optional application workload: logical table capacity, actual per-row history and
// physical reservation are independent. Shared pages are read-only prefix pages.
struct IndexerCase {
    std::vector<int> ends, valid, reserve_ends;
    int envelope = 0, shared_prefix = 0;
    bool contiguous_pages = false;

    int table_end(int fallback) const {
        return std::max(fallback, std::max(envelope, reserve_ends.empty()
                                                         ? 0
                                                         : *std::max_element(reserve_ends.begin(),
                                                                             reserve_ends.end())));
    }

    int pages(int batch, int fallback) const {
        if (ends.empty()) return batch * std::max(1, (fallback + 63) / 64);
        int count = shared_prefix / 64;
        for (int b = 0; b < batch; ++b)
            count +=
                (reserve_ends.empty() ? (ends.at(b) + 63) / 64 : (reserve_ends.at(b) + 63) / 64) -
                shared_prefix / 64;
        return std::max(1, count);
    }
};

// Public storage/input fixture only. Dense keys represent physical pages, independently
// mapped by a permuted, padded I64 table. Never mirror the full multilayer backing on Host.
struct IndexerFixture {
    int T, B, E, P, layers, layer, physical_pages;
    Buffer<std::uint16_t> query;
    Buffer<float> coeff;
    Buffer<int> positions, valid, selectors, ids, counts, ends;
    Buffer<std::int64_t> tables;
    ops::KpoolMlaCacheLayout layout;
    PackedKVFixture packed;
    ops::ConstKpoolIndexCacheView cache;
    ops::KpoolBlockSelection selection;
    ops::KpoolExecutionEnvelope envelope;
    std::vector<std::uint16_t> keys;
    DeviceBuffer scratch;
    std::unique_ptr<MemoryArena> workspace;
    std::size_t workspace_bytes;
    int plane_index;

    IndexerFixture(CudaExecutionView cuda, int tokens, int batch, int end, int layer_count = 11,
                   bool ragged                  = false,
                   ops::KpoolMlaStorage storage = ops::KpoolMlaStorage::BFloat16,
                   const IndexerCase& shape = {}, bool initialize_history = true)
        : T(tokens), B(batch), E(end), P(std::max(1, (shape.table_end(end) + 63) / 64)),
          layers(layer_count), layer(layers / 2), physical_pages(shape.pages(batch, end)),
          query(cuda.device(), DType::BF16, {128, 32, T, B}),
          coeff(cuda.device(), DType::FP32, {32, T, B}),
          positions(cuda.device(), DType::I32, {T, B}), valid(cuda.device(), DType::I32, {B}),
          selectors(cuda.device(), DType::I32, {B}), ids(cuda.device(), DType::I32, {512, T * B}),
          counts(cuda.device(), DType::I32, {T * B}), ends(cuda.device(), DType::I32, {T * B}),
          tables(cuda.device(), DType::I64, {P, B}, true),
          layout(ops::plan_kpool_mla_cache_layout({storage, 512, 128, layers})),
          packed(cuda.device(), layout.page.geometry, physical_pages),
          cache(layout.layer(PagedRecordView{packed.storage(), tables.view, 64}, layer).index),
          selection{ids.view, counts.view, ends.view},
          envelope{std::uint32_t(shape.envelope ? shape.envelope : end)},
          keys(initialize_history ? std::size_t(physical_pages) * 16 * 128 : 0),
          workspace_bytes(ops::kpool_indexer_workspace_capacity_bytes(
              {32, 128}, envelope, B, 1, T, cuda.multiprocessor_count())) {
        if (T <= 0 || B < 1 || B > 8 || E < 0 || E > 1048576 || (E && E < T))
            throw std::invalid_argument("Indexer fixture T/B/end");
        plane_index    = (storage == ops::KpoolMlaStorage::BFloat16 ? 2 : 3) * (layer + 1) - 1;
        int multiplier = 17;
        while (std::gcd(multiplier, physical_pages) != 1) multiplier += 2;
        int next_page = shape.shared_prefix / 64;
        if (shape.shared_prefix % 64)
            throw std::invalid_argument("shared prefix must contain whole pages");
        for (int b = 0; b < B; ++b) {
            const int row_end = shape.ends.empty() ? E : shape.ends.at(b);
            const int n       = shape.valid.empty() ? (!E || (ragged && b == B - 1) ? 0
                                                       : ragged ? std::max(1, T - b)
                                                                : T)
                                                    : shape.valid.at(b);
            if (n < 0 || n > T || row_end < n || row_end > int(envelope.max_visible_tokens) ||
                (n && row_end - n < shape.shared_prefix))
                throw std::invalid_argument("invalid per-row history");
            valid.at(b)     = n;
            selectors.at(b) = n ? B - 1 - b : -1;
            for (int t = 0; t < T; ++t)
                positions.at(t, b) = t < n ? (shape.ends.empty() ? E - T : row_end - n) + t : -1;
            const int row_pages =
                shape.ends.empty()
                    ? P
                    : ((shape.reserve_ends.empty() ? row_end : shape.reserve_ends.at(b)) + 63) / 64;
            for (int p = 0; p < P; ++p) {
                if (p >= row_pages) {
                    tables.at(p, B - 1 - b) = -1;
                    continue;
                }
                const int logical = p < shape.shared_prefix / 64 ? p : next_page++;
                const int physical =
                    shape.contiguous_pages
                        ? logical
                        : (std::int64_t(logical) * multiplier + physical_pages / 2) %
                              physical_pages;
                tables.at(p, B - 1 - b) = packed.physical_offset(physical);
            }
        }
        for (std::size_t i = 0; i < query.host.size(); ++i)
            query.host[i] = sample(i + 719, 1.F / 512);
        for (std::size_t i = 0; i < coeff.host.size(); ++i)
            coeff.host[i] = (int(hash(i + 2071) % 1025) - 512) * (1.F / 512);
        for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = sample(i + 3307, 1.F / 512);
        query.upload();
        coeff.upload();
        upload_metadata();
        if (initialize_history) upload_keys();
        scratch = DeviceBuffer(cuda.device(), workspace_bytes + 512);
        scratch.fill(0xa5);
        workspace = std::make_unique<MemoryArena>(scratch.span().subspan(256, workspace_bytes));
    }

    void upload_metadata() {
        positions.upload();
        valid.upload();
        selectors.upload();
        tables.upload();
    }

    void upload_keys() {
        Tensor dense({keys.data(), keys.size() * 2, Device::cpu()}, DType::BF16,
                     {128, 16, 1, physical_pages});
        packed.upload_plane(plane_index, dense);
    }

    const std::uint16_t* key(int b, int pool) const {
        const auto offset = tables.at(pool / 16, selectors.at(b));
        const auto page   = (offset - packed.physical_offset(0)) / packed.record_stride();
        return keys.data() + (page * 16 + pool % 16) * 128;
    }

    void launch(CudaExecutionView cuda) {
        ops::kpool_indexer(query.view, coeff.view, cache,
                           {positions.view, valid.view, selectors.view}, envelope, *workspace,
                           selection, cuda);
    }

    bool scratch_guards() const {
        std::array<std::byte, 256> a, b;
        scratch.copy_to_host(a.data(), a.size());
        scratch.copy_to_host(b.data(), b.size(), scratch.size() - b.size());
        const auto ok = [](auto x) { return x == std::byte{0xa5}; };
        return std::all_of(a.begin(), a.end(), ok) && std::all_of(b.begin(), b.end(), ok);
    }
};
} // namespace ninfer::tool::kpool
