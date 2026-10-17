#include "fixture.h"

namespace ninfer::test::causal_attention {

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
      k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
      v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
      k_scale_(layout_.key.scale_extent == 0
                   ? 1
                   : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
      v_scale_(layout_.value.scale_extent == 0
                   ? 1
                   : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
      block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
    block_table_.copy_from_host(block_table_host_.data(),
                                block_table_host_.size() * sizeof(std::int32_t));
    if (storage_ == KvCacheStorage::BFloat16) {
        const auto k_physical = scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical = scatter_paged(cache.v_fp16, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
        v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
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
        k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
        v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
    } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
        const auto k_physical = scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical = scatter_paged(cache.v_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        k_.copy_from_host(k_physical.data(), k_physical.size());
        v_.copy_from_host(v_physical.data(), v_physical.size());
        k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto k_physical = scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                                              block_table_host_, physical_pages_);
        const auto v_physical =
            scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto ks_physical =
            scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        const auto vs_physical =
            scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                          block_table_host_, physical_pages_);
        k_.copy_from_host(k_physical.data(), k_physical.size());
        v_.copy_from_host(v_physical.data(), v_physical.size());
        k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
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
        k_.copy_from_host(k_physical.data(), k_physical.size());
        v_.copy_from_host(v_physical.data(), v_physical.size());
        k_scale_.copy_from_host(ks_physical.data(), ks_physical.size());
        v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
    }
}

PagedKVLayerView DeviceCache::view() {
    PagedKVLayerView result;
    result.k_pages =
        Tensor(k_.span(), layout_.key.code_dtype,
               {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    result.v_pages =
        Tensor(v_.span(), layout_.value.code_dtype,
               {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    result.block_table  = Tensor(block_table_.span(), DType::I32, {logical_pages_});
    result.num_kv_heads = geometry_.kv_heads;
    result.head_dim     = kHeadDim;
    result.storage      = storage_;
    if (layout_.key.scale_extent != 0) {
        result.k_scale_pages = Tensor(
            k_scale_.span(), layout_.key.scale_dtype,
            {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    }
    if (layout_.value.scale_extent != 0) {
        result.v_scale_pages = Tensor(
            v_scale_.span(), layout_.value.scale_dtype,
            {layout_.value.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    }
    return result;
}

PagedKVBatchLayerView DeviceCache::batch_view() {
    const PagedKVLayerView direct = view();
    return {
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = direct.block_table.view({logical_pages_, 1}),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
    };
}

HostCache DeviceCache::snapshot() const {
    HostCache cache{geometry_, storage_, max_context_, logical_capacity_};
    if (storage_ == KvCacheStorage::BFloat16) {
        const auto k_physical = copy_from_guarded<std::uint16_t>(k_, k_code_elements_);
        const auto v_physical = copy_from_guarded<std::uint16_t>(v_, v_code_elements_);
        cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                            logical_capacity_, block_table_host_);
        cache.v_fp16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                            logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Int8Group64) {
        const auto k_physical  = copy_from_guarded<std::int8_t>(k_, k_code_elements_);
        const auto v_physical  = copy_from_guarded<std::int8_t>(v_, v_code_elements_);
        const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
        const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
        cache.k_i8 = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                               block_table_host_);
        cache.v_i8 = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_, logical_capacity_,
                                               block_table_host_);
        cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
        cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
        const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
        const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
        const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
        const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
        cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.v_fp8 = gather_paged<std::uint8_t>(v_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
        cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kFp8QuantGroups, geometry_,
                                                    logical_capacity_, block_table_host_);
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
        const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
        const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
        const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
        cache.k_fp8 = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_, logical_capacity_,
                                                 block_table_host_);
        cache.v_nvfp4       = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                         logical_capacity_, block_table_host_);
        cache.k_scale       = gather_paged<std::uint16_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                          logical_capacity_, block_table_host_);
        cache.v_nvfp4_scale = gather_paged<std::uint8_t>(vs_physical, kNvfp4QuantGroups, geometry_,
                                                         logical_capacity_, block_table_host_);
    } else {
        const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
        const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
        const auto ks_physical = copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_);
        const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
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
    int failures = 0;
    failures += k_.verify_guards((label + " cache-k").c_str());
    failures += v_.verify_guards((label + " cache-v").c_str());
    if (layout_.key.scale_extent != 0) {
        failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
    }
    if (layout_.value.scale_extent != 0) {
        failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
    }
    failures += block_table_.verify_guards((label + " block-table").c_str());
    failures += verify_exact(
        (label + " block-table unchanged").c_str(),
        copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()), block_table_host_);
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
      k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
      v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
      k_scale_(layout_.key.scale_extent == 0
                   ? 1
                   : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
      v_scale_(layout_.value.scale_extent == 0
                   ? 1
                   : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
      block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
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
    block_tables_.copy_from_host(block_tables_host_.data(),
                                 block_tables_host_.size() * sizeof(std::int32_t));
    upload_rows(rows);
}

PagedKVBatchLayerView BatchDeviceCache::view() {
    PagedKVBatchLayerView result;
    result.k_pages =
        Tensor(k_.span(), layout_.key.code_dtype,
               {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    result.v_pages =
        Tensor(v_.span(), layout_.value.code_dtype,
               {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    result.block_tables = Tensor(block_tables_.span(), DType::I32,
                                 {logical_pages_, static_cast<std::int32_t>(rows_)});
    result.num_kv_heads = geometry_.kv_heads;
    result.head_dim     = kHeadDim;
    result.storage      = storage_;
    if (layout_.key.scale_extent != 0) {
        result.k_scale_pages = Tensor(
            k_scale_.span(), layout_.key.scale_dtype,
            {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    }
    if (layout_.value.scale_extent != 0) {
        result.v_scale_pages = Tensor(
            v_scale_.span(), layout_.value.scale_dtype,
            {layout_.value.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
    }
    return result;
}

BatchDeviceCache::Bytes BatchDeviceCache::snapshot_bytes() const {
    return {copy_from_guarded<std::uint8_t>(k_, k_.bytes()),
            copy_from_guarded<std::uint8_t>(v_, v_.bytes()),
            copy_from_guarded<std::uint8_t>(k_scale_, k_scale_.bytes()),
            copy_from_guarded<std::uint8_t>(v_scale_, v_scale_.bytes())};
}

void BatchDeviceCache::copy_from(const BatchDeviceCache& source) {
    const std::array<const GuardedDeviceBuffer*, 4> from{&source.k_, &source.v_, &source.k_scale_,
                                                         &source.v_scale_};
    const std::array<GuardedDeviceBuffer*, 4> to{&k_, &v_, &k_scale_, &v_scale_};
    for (int i = 0; i < 4; ++i)
        CUDA_CHECK(
            cudaMemcpy(to[i]->data(), from[i]->data(), to[i]->bytes(), cudaMemcpyDeviceToDevice));
}

PagedKVLayerView BatchDeviceCache::single_view(int row) {
    auto batch = view();
    return {.k_pages       = batch.k_pages,
            .v_pages       = batch.v_pages,
            .k_scale_pages = batch.k_scale_pages,
            .v_scale_pages = batch.v_scale_pages,
            .block_table   = batch.block_tables.slice(1, row, 1).view({logical_pages_}),
            .head_dim      = kHeadDim,
            .num_kv_heads  = geometry_.kv_heads,
            .storage       = storage_};
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
                         copy_from_guarded<std::uint16_t>(k_, k_code_elements_), expected_k);
        failures +=
            verify_exact((label + " cache-v").c_str(),
                         copy_from_guarded<std::uint16_t>(v_, v_code_elements_), expected_v);
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
        failures += verify_exact((label + " cache-k-code").c_str(),
                                 copy_from_guarded<std::int8_t>(k_, k_code_elements_), expected_k);
        failures += verify_exact((label + " cache-k-scale").c_str(),
                                 copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                 expected_ks);
        failures += verify_exact((label + " cache-v-code").c_str(),
                                 copy_from_guarded<std::int8_t>(v_, v_code_elements_), expected_v);
        failures += verify_exact((label + " cache-v-scale").c_str(),
                                 copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                 expected_vs);
    } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
        std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_k);
            scatter_paged_into(expected[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_v);
            scatter_paged_into(expected[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, expected_ks);
            scatter_paged_into(expected[row].v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, expected_vs);
        }
        failures += verify_exact((label + " cache-k-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
        failures += verify_exact((label + " cache-v-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
        failures += verify_exact((label + " cache-k-scale").c_str(),
                                 copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                 expected_ks);
        failures += verify_exact((label + " cache-v-scale").c_str(),
                                 copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                 expected_vs);
    } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
        std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
        std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               expected_k);
            scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, expected_v);
            scatter_paged_into(expected[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, expected_ks);
            scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, expected_vs);
        }
        failures += verify_exact((label + " cache-k-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
        failures += verify_exact((label + " cache-v-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
        failures += verify_exact((label + " cache-k-scale").c_str(),
                                 copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                 expected_ks);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_), expected_vs);
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
        failures += verify_exact((label + " cache-k-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
        failures += verify_exact((label + " cache-v-code").c_str(),
                                 copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
        failures +=
            verify_exact((label + " cache-k-scale").c_str(),
                         copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_), expected_ks);
        failures +=
            verify_exact((label + " cache-v-scale").c_str(),
                         copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_), expected_vs);
    }
    failures +=
        verify_exact((label + " block tables unchanged").c_str(),
                     copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                     block_tables_host_);
    failures += k_.verify_guards((label + " cache-k guard").c_str());
    failures += v_.verify_guards((label + " cache-v guard").c_str());
    if (layout_.key.scale_extent != 0) {
        failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
    }
    if (layout_.value.scale_extent != 0) {
        failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
    }
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
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
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
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
        return;
    }
    if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_ks);
            scatter_paged_into(rows[row].v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size());
        v_.copy_from_host(physical_v.data(), physical_v.size());
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
        return;
    }
    if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_v);
            scatter_paged_into(rows[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                               table, physical_ks);
            scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size());
        v_.copy_from_host(physical_v.data(), physical_v.size());
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
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
    k_.copy_from_host(physical_k.data(), physical_k.size());
    v_.copy_from_host(physical_v.data(), physical_v.size());
    k_scale_.copy_from_host(physical_ks.data(), physical_ks.size());
    v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
}

} // namespace ninfer::test::causal_attention
