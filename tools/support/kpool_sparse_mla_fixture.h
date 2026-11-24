#pragma once

#include "kpool_fixture.h"
#include "kpool_fp8_fixture.h"
#include "paged_kv_fixture.h"
#include "ninfer/ops/kpool_mla_layout.h"

#include <memory>
#include <map>
#include <numeric>
#include <set>

namespace ninfer::tool::kpool {

// Allocate only referenced logical pages, each as a complete native Main/MTP page
// group. Unreferenced table entries stay invalid; no dense or single-layer cache proxy.
struct SparseMlaFixture {
    int T, B, E, P, layers, layer, plane_index;
    Buffer<std::uint16_t> query, key_up, value_up, output;
    Buffer<int> positions, valid, selectors, selection_rows, ids, counts, ends;
    Buffer<std::int64_t> tables;
    ops::KpoolMlaCacheLayout layout;
    std::unique_ptr<PackedKVFixture> packed;
    ops::ConstKpoolMlaLatentCacheView cache;
    ops::KpoolExecutionEnvelope envelope;
    std::vector<std::uint16_t> latent;
    std::vector<std::uint8_t> latent_codes, latent_scales;
    int scale_plane_index = -1;
    DeviceBuffer scratch;
    std::unique_ptr<MemoryArena> workspace;
    std::size_t workspace_bytes;
    float scale = 1.F / 16;

    SparseMlaFixture(CudaExecutionView cuda, int tokens, int batch, int end, int cap = 512,
                     int layer_count = 11, bool ragged = false, bool frozen = false,
                     float magnitude = 1, int selection_span = 1,
                     ops::KpoolMlaStorage storage     = ops::KpoolMlaStorage::BFloat16,
                     KvCacheVectorFormat index_format = KvCacheVectorFormat::BFloat16,
                     int shared_prefix = 0, int max_visible = 0)
        : T(tokens), B(batch), E(end), P(std::max(1, (std::max(end, max_visible) + 63) / 64)),
          layers(layer_count), layer(layers / 2),
          query(cuda.device(), DType::BF16, {256, 64, T, B}),
          key_up(cuda.device(), DType::BF16, {512, 256, 64}),
          value_up(cuda.device(), DType::BF16, {512, 256, 64}),
          output(cuda.device(), DType::BF16, {256, 64, T, B}),
          positions(cuda.device(), DType::I32, {T, B}), valid(cuda.device(), DType::I32, {B}),
          selectors(cuda.device(), DType::I32, {B}),
          selection_rows(cuda.device(), DType::I32, {T, B}),
          ids(cuda.device(), DType::I32, {512, T * B}), counts(cuda.device(), DType::I32, {T * B}),
          ends(cuda.device(), DType::I32, {T * B}), tables(cuda.device(), DType::I64, {P, B}, true),
          layout(ops::plan_kpool_mla_cache_layout({storage, index_format, 512, 128, layers})),
          envelope{std::uint32_t(max_visible ? max_visible : end)},
          workspace_bytes(ops::kpool_sparse_mla_workspace_capacity_bytes(
              {64, 256, 256, 512}, storage, envelope, B, 1, T, cuda.multiprocessor_count())) {
        if (E < 0 || E > 1048576 || (E && E < T) || cap < 0 || cap > 512 || selection_span < 1 ||
            shared_prefix < 0 || shared_prefix > E || shared_prefix % 64 ||
            envelope.max_visible_tokens < unsigned(E))
            throw std::invalid_argument("Sparse MLA fixture T/end/cap");
        std::fill(ids.host.begin(), ids.host.end(), -1);
        std::fill(counts.host.begin(), counts.host.end(), -1);
        std::fill(ends.host.begin(), ends.host.end(), -1);
        std::fill(tables.host.begin(), tables.host.end(), -1);
        std::set<std::pair<int, int>> pages;
        for (int b = 0; b < B; ++b) {
            valid.at(b)     = !E || (ragged && b == B - 1) ? 0 : ragged ? std::max(1, T - b) : T;
            selectors.at(b) = valid.at(b) ? B - 1 - b : -1;
            for (int t = 0; t < T; ++t) {
                positions.at(t, b)      = t < valid.at(b) ? E - T + t : -1;
                const int source        = t / selection_span * selection_span;
                const int row           = b * T + T - 1 - source;
                selection_rows.at(t, b) = t < valid.at(b) ? row : -1;
                if (t >= valid.at(b)) continue;
                if (t != source) continue;
                const int e = frozen ? E - T : E - T + t + 1, c = e / 4;
                counts.at(row) = std::min(cap, c);
                ends.at(row)   = e;
                int stride     = 503;
                while (c && std::gcd(stride, c) != 1) stride += 2;
                for (int i = 0; i < counts.at(row); ++i) {
                    const int g    = (std::int64_t(i) * stride + t + 17 * b) % c;
                    ids.at(i, row) = g;
                    pages.emplace(b, g / 16);
                }
                if (e % 4) pages.emplace(b, e / 64);
            }
        }
        std::map<std::pair<int, int>, int> physical;
        for (auto [b, p] : pages)
            physical.emplace(std::pair{p < shared_prefix / 64 ? 0 : b + 1, p}, 0);
        const int physical_pages = std::max<std::size_t>(1, physical.size());
        packed =
            std::make_unique<PackedKVFixture>(cuda.device(), layout.page.geometry, physical_pages);
        int multiplier = 17;
        while (std::gcd(multiplier, physical_pages) != 1) multiplier += 2;
        int ordinal = 0;
        for (auto& [key, page] : physical)
            page = (std::int64_t(ordinal++) * multiplier + physical_pages / 2) % physical_pages;
        for (auto [b, p] : pages)
            tables.at(p, B - 1 - b) =
                packed->physical_offset(physical.at({p < shared_prefix / 64 ? 0 : b + 1, p}));
        cache = layout.layer(PagedRecordView{packed->storage(), tables.view, 64}, layer).latent;
        plane_index = packed->plane_index(cache.data.plane);
        if (storage == ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0)
            scale_plane_index = packed->plane_index(cache.scales.plane);
        latent.resize(std::size_t(physical_pages) * 64 * 512);
        for (std::size_t i = 0; i < latent.size(); ++i) latent[i] = sample(i + 3351, 1.F / 512);
        for (std::size_t i = 0; i < query.host.size(); ++i)
            query.host[i] = sample(i + 719, magnitude / 512);
        for (std::size_t i = 0; i < key_up.host.size(); ++i) {
            key_up.host[i]   = sample(i + 1157, 1.F / 8192);
            value_up.host[i] = sample(i + 7759, 1.F / 8192);
        }
        query.upload();
        key_up.upload();
        value_up.upload();
        upload_metadata();
        upload_latent();
        scratch = DeviceBuffer(cuda.device(), workspace_bytes + 512);
        scratch.fill(0xa5);
        workspace = std::make_unique<MemoryArena>(scratch.span().subspan(256, workspace_bytes));
    }

    void upload_metadata() {
        positions.upload();
        valid.upload();
        selectors.upload();
        selection_rows.upload();
        ids.upload();
        counts.upload();
        ends.upload();
        tables.upload();
    }

    void upload_latent() {
        const int pages = latent.size() / (512 * 64);
        if (cache.storage == ops::KpoolMlaStorage::BFloat16) {
            Tensor dense({latent.data(), latent.size() * 2, Device::cpu()}, DType::BF16,
                         {512, 64, 1, pages});
            packed->upload_plane(plane_index, dense);
        } else {
            encode_fp8_latent(latent, latent_codes, latent_scales);
            Tensor codes({latent_codes.data(), latent_codes.size(), Device::cpu()},
                         DType::FP8_E4M3FN, {512, 64, 1, pages});
            Tensor scales({latent_scales.data(), latent_scales.size(), Device::cpu()}, DType::U8,
                          {4, 64, 1, pages});
            packed->upload_plane(plane_index, codes);
            packed->upload_plane(scale_plane_index, scales);
        }
    }

    std::vector<int> selected_tokens(int t, int b) const {
        std::vector<int> result;
        if (t >= valid.at(b)) return result;
        const int row = selection_rows.at(t, b), end = ends.at(row);
        for (int i = 0; i < counts.at(row); ++i)
            for (int j = 0; j < 4; ++j) result.push_back(ids.at(i, row) * 4 + j);
        for (int i = end / 4 * 4; i < end; ++i) result.push_back(i);
        return result;
    }

    std::size_t physical_token(int b, int token) const {
        const auto offset = tables.at(token / 64, selectors.at(b));
        return (offset - packed->physical_offset(0)) / packed->record_stride() * 64 + token % 64;
    }

    const std::uint16_t* latent_row(int b, int token) const {
        return latent.data() + physical_token(b, token) * 512;
    }

    void launch(CudaExecutionView cuda) {
        ops::kpool_sparse_mla(query.view, {key_up.view, value_up.view}, cache,
                              {positions.view, valid.view, selectors.view},
                              {ids.view, counts.view, ends.view}, selection_rows.view, scale,
                              envelope, *workspace, output.view, cuda);
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
