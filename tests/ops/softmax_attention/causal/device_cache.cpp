#include "fixture.h"

namespace ninfer::test::causal_attention {
namespace {

KVPageGeometry cache_geometry(const TestCacheLayout& layout, const Geometry& geometry) {
    KVPageGeometry result;
    result.planes.reserve(4);
    const auto add = [&](DType dtype, int extent) {
        result.planes.push_back({dtype, extent, geometry.kv_heads});
    };
    add(layout.key.code_dtype, layout.key.code_extent);
    add(layout.value.code_dtype, layout.value.code_extent);
    if (layout.key.scale_extent) add(layout.key.scale_dtype, layout.key.scale_extent);
    if (layout.value.scale_extent) add(layout.value.scale_dtype, layout.value.scale_extent);
    return result;
}

void upload_plane(tool::PackedKVFixture& packed, std::size_t index, const void* data,
                  std::size_t bytes) {
    const auto& plane = packed.layout().planes.at(index).tensor;
    const auto pages  = static_cast<int>(bytes / plane.region.bytes);
    Tensor dense({const_cast<void*>(data), bytes, Device::cpu()}, plane.dtype,
                 {plane.shape[0], plane.shape[1], plane.shape[2], pages});
    packed.upload_plane(index, dense);
}

template <class T>
std::vector<T> snapshot_plane(const tool::PackedKVFixture& packed, std::size_t index,
                              std::size_t elements) {
    if (!elements) return {};
    const auto bytes = packed.download_plane(index);
    std::vector<T> values(elements);
    if (bytes.size() != elements * sizeof(T)) throw std::logic_error("cache plane size mismatch");
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

PagedKVLayerView layer_view(tool::PackedKVFixture& packed, const Tensor& table,
                            const Geometry& geometry, KvCacheStorage storage) {
    PagedKVLayerView result{
        .k_pages      = packed.plane(0, table),
        .v_pages      = packed.plane(1, table),
        .head_dim     = kHeadDim,
        .num_kv_heads = geometry.kv_heads,
        .storage      = storage,
    };
    if (packed.layout().planes.size() > 2) {
        result.k_scale_pages = packed.plane(2, table);
        result.v_scale_pages = packed.plane(3, table);
    }
    return result;
}

int verify_record_guards(const std::string& label, const tool::PackedKVFixture& packed) {
    if (packed.guards_intact()) return 0;
    std::cerr << label << ": cache record guards changed\n";
    return 1;
}

} // namespace

DeviceCache::DeviceCache(const HostCache& cache, MappingPattern mapping,
                         std::int32_t table_capacity)
    : geometry_(cache.geometry), storage_(cache.storage), layout_(test_cache_layout(storage_)),
      max_context_(cache.max_context), logical_capacity_(cache.logical_capacity),
      logical_pages_(align_up_page(std::max(logical_capacity_, table_capacity)) / kPagedKVPageSize),
      physical_pages_(physical_page_count(logical_capacity_ / kPagedKVPageSize, mapping)),
      block_table_host_(
          make_block_table(logical_capacity_ / kPagedKVPageSize, logical_pages_, mapping)),
      k_code_elements_(
          physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
      v_code_elements_(
          physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
      k_scale_elements_(
          physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
      v_scale_elements_(
          physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
      packed_(cuda_environment().context.device(), cache_geometry(layout_, geometry_),
              physical_pages_),
      block_table_(block_table_host_.size() * sizeof(std::int64_t)) {
    const auto offsets = packed_.table_offsets(block_table_host_);
    block_table_.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
    if (storage_ == KvCacheStorage::BFloat16) {
        const auto k_physical = scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical = scatter_paged(cache.v_fp16, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        upload_plane(packed_, 0, k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
        upload_plane(packed_, 1, v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
    } else if (storage_ == KvCacheStorage::Int8Group64) {
        const auto k_physical = scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical = scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        upload_plane(packed_, 0, k_physical.data(), k_physical.size() * sizeof(std::int8_t));
        upload_plane(packed_, 1, v_physical.data(), v_physical.size() * sizeof(std::int8_t));
        upload_plane(packed_, 2, ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
        upload_plane(packed_, 3, vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
    } else if (storage_ == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        const auto k_physical = scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical = scatter_paged(cache.v_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        upload_plane(packed_, 0, k_physical.data(), k_physical.size());
        upload_plane(packed_, 1, v_physical.data(), v_physical.size());
        upload_plane(packed_, 2, ks_physical.data(), ks_physical.size() * sizeof(std::uint8_t));
        upload_plane(packed_, 3, vs_physical.data(), vs_physical.size() * sizeof(std::uint8_t));
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto k_physical = scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical =
            scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        upload_plane(packed_, 0, k_physical.data(), k_physical.size());
        upload_plane(packed_, 1, v_physical.data(), v_physical.size());
        upload_plane(packed_, 2, ks_physical.data(), ks_physical.size() * sizeof(std::uint8_t));
        upload_plane(packed_, 3, vs_physical.data(), vs_physical.size());
    } else {
        const auto k_physical =
            scatter_paged(cache.k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto v_physical =
            scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        upload_plane(packed_, 0, k_physical.data(), k_physical.size());
        upload_plane(packed_, 1, v_physical.data(), v_physical.size());
        upload_plane(packed_, 2, ks_physical.data(), ks_physical.size());
        upload_plane(packed_, 3, vs_physical.data(), vs_physical.size());
    }
}

PagedKVLayerView DeviceCache::view() {
    return layer_view(packed_, Tensor(block_table_.span(), DType::I64, {logical_pages_}), geometry_,
                      storage_);
}

PagedKVBatchLayerView DeviceCache::batch_view() { return single_row_paged_kv_batch_view(view()); }

HostCache DeviceCache::snapshot() const {
    HostCache cache{geometry_, storage_, max_context_, logical_capacity_};
    if (storage_ == KvCacheStorage::BFloat16) {
        const auto k_physical = snapshot_plane<std::uint16_t>(packed_, 0, k_code_elements_);
        const auto v_physical = snapshot_plane<std::uint16_t>(packed_, 1, v_code_elements_);
        cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                            logical_capacity_, block_table_host_);
        cache.v_fp16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                            logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Int8Group64) {
        const auto k_physical  = snapshot_plane<std::int8_t>(packed_, 0, k_code_elements_);
        const auto v_physical  = snapshot_plane<std::int8_t>(packed_, 1, v_code_elements_);
        const auto ks_physical = snapshot_plane<std::uint16_t>(packed_, 2, k_scale_elements_);
        const auto vs_physical = snapshot_plane<std::uint16_t>(packed_, 3, v_scale_elements_);
        cache.k_i8 = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                               block_table_host_);
        cache.v_i8 = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_, logical_capacity_,
                                               block_table_host_);
        cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
        cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        const auto k_physical  = snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_);
        const auto v_physical  = snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_);
        const auto ks_physical = snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_);
        const auto vs_physical = snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_);
        cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.v_fp8 = gather_paged<std::uint8_t>(v_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.k_fp8_scale = gather_paged<std::uint8_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                       logical_capacity_, block_table_host_);
        cache.v_fp8_scale = gather_paged<std::uint8_t>(vs_physical, kFp8QuantGroups, geometry_,
                                                       logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto k_physical  = snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_);
        const auto v_physical  = snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_);
        const auto ks_physical = snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_);
        const auto vs_physical = snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_);
        cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.v_nvfp4       = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                         logical_capacity_, block_table_host_);
        cache.k_fp8_scale   = gather_paged<std::uint8_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                         logical_capacity_, block_table_host_);
        cache.v_nvfp4_scale = gather_paged<std::uint8_t>(vs_physical, kNvfp4QuantGroups, geometry_,
                                                         logical_capacity_, block_table_host_);
    } else {
        const auto k_physical  = snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_);
        const auto v_physical  = snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_);
        const auto ks_physical = snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_);
        const auto vs_physical = snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_);
        cache.k_nvfp4          = gather_paged<std::uint8_t>(k_physical, kNvfp4CodeBytes, geometry_,
                                                            logical_capacity_, block_table_host_);
        cache.v_nvfp4          = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                            logical_capacity_, block_table_host_);
        cache.k_nvfp4_scale = gather_paged<std::uint8_t>(ks_physical, kNvfp4QuantGroups, geometry_,
                                                         logical_capacity_, block_table_host_);
        cache.v_nvfp4_scale = gather_paged<std::uint8_t>(vs_physical, kNvfp4QuantGroups, geometry_,
                                                         logical_capacity_, block_table_host_);
    }
    return cache;
}

int DeviceCache::verify_guards(const std::string& label) const {
    int failures = verify_record_guards(label, packed_);
    failures += block_table_.verify_guards((label + " page offsets").c_str());
    failures +=
        verify_exact((label + " page offsets unchanged").c_str(),
                     copy_from_guarded<std::int64_t>(block_table_, block_table_host_.size()),
                     packed_.table_offsets(block_table_host_));
    return failures;
}

BatchDeviceCache::BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping,
                                   std::int32_t table_capacity)
    : geometry_(rows.front().geometry), storage_(rows.front().storage),
      layout_(test_cache_layout(storage_)), rows_(rows.size()),
      logical_capacity_(rows.front().logical_capacity),
      logical_pages_(align_up_page(std::max(logical_capacity_, table_capacity)) / kPagedKVPageSize),
      physical_pages_(
          mapping == MappingPattern::Fragmented
              ? 2 * static_cast<std::int32_t>(rows_) * (logical_capacity_ / kPagedKVPageSize) + 1
              : static_cast<std::int32_t>(rows_) * (logical_capacity_ / kPagedKVPageSize)),
      block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_), -1),
      k_code_elements_(
          physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
      v_code_elements_(
          physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
      k_scale_elements_(
          physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
      v_scale_elements_(
          physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
      packed_(cuda_environment().context.device(), cache_geometry(layout_, geometry_),
              physical_pages_),
      block_tables_(block_tables_host_.size() * sizeof(std::int64_t)) {
    for (std::size_t row = 0; row < rows_; ++row) {
        const HostCache& cache = rows[row];
        if (cache.geometry.q_heads != geometry_.q_heads ||
            cache.geometry.kv_heads != geometry_.kv_heads || cache.storage != storage_ ||
            cache.logical_capacity != logical_capacity_) {
            throw std::invalid_argument("batch cache rows must share one physical geometry");
        }
        for (std::int32_t logical = 0; logical < logical_capacity_ / kPagedKVPageSize; ++logical) {
            const std::int32_t linear =
                static_cast<std::int32_t>(row) * (logical_capacity_ / kPagedKVPageSize) + logical;
            block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
        }
    }
    const auto offsets = packed_.table_offsets(block_tables_host_);
    block_tables_.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
    upload_rows(rows);
}

PagedKVBatchLayerView BatchDeviceCache::view() {
    const auto table =
        Tensor(block_tables_.span(), DType::I64, {logical_pages_, static_cast<int>(rows_)});
    const auto direct = layer_view(packed_, table, geometry_, storage_);
    return {direct.k_pages,  direct.v_pages,      direct.k_scale_pages, direct.v_scale_pages,
            direct.head_dim, direct.num_kv_heads, direct.storage};
}

BatchDeviceCache::Bytes BatchDeviceCache::snapshot_bytes() const {
    return {snapshot_plane<std::uint8_t>(packed_, 0,
                                         k_code_elements_ * dtype_size(layout_.key.code_dtype)),
            snapshot_plane<std::uint8_t>(packed_, 1,
                                         v_code_elements_ * dtype_size(layout_.value.code_dtype)),
            snapshot_plane<std::uint8_t>(packed_, 2,
                                         k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
            snapshot_plane<std::uint8_t>(
                packed_, 3, v_scale_elements_ * dtype_size(layout_.value.scale_dtype))};
}

void BatchDeviceCache::copy_from(const BatchDeviceCache& source) {
    CUDA_CHECK(cudaMemcpy(packed_.storage().data, source.packed_.storage().data,
                          packed_.storage().bytes, cudaMemcpyDeviceToDevice));
}

PagedKVLayerView BatchDeviceCache::single_view(int row) {
    const auto table =
        Tensor(block_tables_.span(), DType::I64, {logical_pages_, static_cast<int>(rows_)})
            .slice(1, row, 1)
            .view({logical_pages_});
    return layer_view(packed_, table, geometry_, storage_);
}

int BatchDeviceCache::verify_untouched(const std::string& label, const Bytes& before,
                                       std::span<const int> positions, std::span<const int> lanes,
                                       std::span<const int> valid, int width) const {
    const auto after        = snapshot_bytes();
    const int physical_rows = physical_pages_ * geometry_.kv_heads * kPagedKVPageSize;
    std::vector<bool> writable(physical_rows, false);
    for (std::size_t b = 0; b < valid.size(); ++b)
        for (int j = 0; j < valid[b]; ++j) {
            const int pos  = positions[b * width + j];
            const int page = block_tables_host_[lanes[b] * logical_pages_ + pos / kPagedKVPageSize];
            for (int h = 0; h < geometry_.kv_heads; ++h)
                writable[paged_index(1, geometry_, page, h, pos, 0)] = true;
        }
    const std::array<std::size_t, 4> row_bytes{
        layout_.key.code_extent * dtype_size(layout_.key.code_dtype),
        layout_.value.code_extent * dtype_size(layout_.value.code_dtype),
        layout_.key.scale_extent * dtype_size(layout_.key.scale_dtype),
        layout_.value.scale_extent * dtype_size(layout_.value.scale_dtype)};
    int failures = 0;
    for (int plane = 0; plane < 4; ++plane) {
        if (row_bytes[plane] == 0) {
            failures +=
                verify_exact((label + " unused scale").c_str(), after[plane], before[plane]);
            continue;
        }
        for (int row = 0; row < physical_rows; ++row)
            if (!writable[row]) {
                const auto first = std::size_t(row) * row_bytes[plane];
                if (!std::equal(before[plane].begin() + first,
                                before[plane].begin() + first + row_bytes[plane],
                                after[plane].begin() + first)) {
                    std::cerr << label << ": unrelated cache bytes changed in plane " << plane
                              << " row " << row << '\n';
                    ++failures;
                    break;
                }
            }
    }
    return failures;
}

int BatchDeviceCache::verify(const std::string& label, std::span<const HostCache> expected) const {
    if (expected.size() != rows_) {
        std::cerr << label << ": expected cache row count mismatch\n";
        return 1;
    }
    int failures = 0;
    if (storage_ == KvCacheStorage::BFloat16) {
        std::vector<std::uint16_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint16_t> expected_v(v_code_elements_, 0);
        scatter_bf16_rows(expected, expected_k, expected_v);
        failures +=
            verify_exact((label + " cache-k").c_str(),
                         snapshot_plane<std::uint16_t>(packed_, 0, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-v").c_str(),
                         snapshot_plane<std::uint16_t>(packed_, 1, v_code_elements_), expected_v);
    } else if (storage_ == KvCacheStorage::Int8Group64) {
        std::vector<std::int8_t> expected_k(k_code_elements_, 0);
        std::vector<std::int8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_k);
            scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_v);
            scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_, logical_capacity_,
                               table, expected_ks);
            scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_, logical_capacity_,
                               table, expected_vs);
        }
        failures +=
            verify_exact((label + " cache-k-code").c_str(),
                         snapshot_plane<std::int8_t>(packed_, 0, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-k-scale").c_str(),
                         snapshot_plane<std::uint16_t>(packed_, 2, k_scale_elements_), expected_ks);
        failures +=
            verify_exact((label + " cache-v-code").c_str(),
                         snapshot_plane<std::int8_t>(packed_, 1, v_code_elements_), expected_v);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         snapshot_plane<std::uint16_t>(packed_, 3, v_scale_elements_), expected_vs);
    } else if (storage_ == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint8_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_k);
            scatter_paged_into(expected[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_v);
            scatter_paged_into(expected[row].k_fp8_scale, kFp8QuantGroups, geometry_,
                               logical_capacity_, table, expected_ks);
            scatter_paged_into(expected[row].v_fp8_scale, kFp8QuantGroups, geometry_,
                               logical_capacity_, table, expected_vs);
        }
        failures +=
            verify_exact((label + " cache-k-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-v-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_), expected_v);
        failures +=
            verify_exact((label + " cache-k-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_), expected_ks);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_), expected_vs);
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint8_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_k);
            scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, expected_v);
            scatter_paged_into(expected[row].k_fp8_scale, kFp8QuantGroups, geometry_,
                               logical_capacity_, table, expected_ks);
            scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, expected_vs);
        }
        failures +=
            verify_exact((label + " cache-k-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-v-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_), expected_v);
        failures +=
            verify_exact((label + " cache-k-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_), expected_ks);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_), expected_vs);
    } else {
        std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint8_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, expected_k);
            scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, expected_v);
            scatter_paged_into(expected[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, expected_ks);
            scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, expected_vs);
        }
        failures +=
            verify_exact((label + " cache-k-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 0, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-v-code").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 1, v_code_elements_), expected_v);
        failures +=
            verify_exact((label + " cache-k-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 2, k_scale_elements_), expected_ks);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         snapshot_plane<std::uint8_t>(packed_, 3, v_scale_elements_), expected_vs);
    }
    failures +=
        verify_exact((label + " block tables unchanged").c_str(),
                     copy_from_guarded<std::int64_t>(block_tables_, block_tables_host_.size()),
                     packed_.table_offsets(block_tables_host_));
    failures += verify_record_guards(label, packed_);
    failures += block_tables_.verify_guards((label + " block tables guard").c_str());
    return failures;
}

std::span<const std::int32_t> BatchDeviceCache::row_table(std::size_t row) const {
    return std::span<const std::int32_t>(block_tables_host_.data() +
                                             row * static_cast<std::size_t>(logical_pages_),
                                         static_cast<std::size_t>(logical_pages_));
}

void BatchDeviceCache::scatter_bf16_rows(std::span<const HostCache> rows,
                                         std::vector<std::uint16_t>& k,
                                         std::vector<std::uint16_t>& v) const {
    for (std::size_t row = 0; row < rows_; ++row) {
        const std::span<const std::int32_t> table = row_table(row);
        scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
        scatter_paged_into(rows[row].v_fp16, kHeadDim, geometry_, logical_capacity_, table, v);
    }
}

void BatchDeviceCache::upload_rows(std::span<const HostCache> rows) {
    if (storage_ == KvCacheStorage::BFloat16) {
        std::vector<std::uint16_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint16_t> physical_v(v_code_elements_, 0);
        scatter_bf16_rows(rows, physical_k, physical_v);
        upload_plane(packed_, 0, physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
        upload_plane(packed_, 1, physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
        return;
    }
    if (storage_ == KvCacheStorage::Int8Group64) {
        std::vector<std::int8_t> physical_k(k_code_elements_, 0);
        std::vector<std::int8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        upload_plane(packed_, 0, physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        upload_plane(packed_, 1, physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        upload_plane(packed_, 2, physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        upload_plane(packed_, 3, physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
        return;
    }
    if (storage_ == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint8_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_ks);
            scatter_paged_into(rows[row].v_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_vs);
        }
        upload_plane(packed_, 0, physical_k.data(), physical_k.size());
        upload_plane(packed_, 1, physical_v.data(), physical_v.size());
        upload_plane(packed_, 2, physical_ks.data(), physical_ks.size() * sizeof(std::uint8_t));
        upload_plane(packed_, 3, physical_vs.data(), physical_vs.size() * sizeof(std::uint8_t));
        return;
    }
    if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint8_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_v);
            scatter_paged_into(rows[row].k_fp8_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_ks);
            scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_vs);
        }
        upload_plane(packed_, 0, physical_k.data(), physical_k.size());
        upload_plane(packed_, 1, physical_v.data(), physical_v.size());
        upload_plane(packed_, 2, physical_ks.data(), physical_ks.size() * sizeof(std::uint8_t));
        upload_plane(packed_, 3, physical_vs.data(), physical_vs.size());
        return;
    }
    std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
    std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
    std::vector<std::uint8_t> physical_ks(k_scale_elements_, 0);
    std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
    for (std::size_t row = 0; row < rows_; ++row) {
        const std::span<const std::int32_t> table = row_table(row);
        scatter_paged_into(rows[row].k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_, table,
                           physical_k);
        scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_, table,
                           physical_v);
        scatter_paged_into(rows[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                           table, physical_ks);
        scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                           table, physical_vs);
    }
    upload_plane(packed_, 0, physical_k.data(), physical_k.size());
    upload_plane(packed_, 1, physical_v.data(), physical_v.size());
    upload_plane(packed_, 2, physical_ks.data(), physical_ks.size());
    upload_plane(packed_, 3, physical_vs.data(), physical_vs.size());
}

} // namespace ninfer::test::causal_attention
