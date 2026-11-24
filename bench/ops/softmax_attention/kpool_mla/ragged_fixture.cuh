#pragma once

#include "jobs.cuh"
#include "ops/ragged_workload.h"
#include "ninfer/ops/kpool_mla_layout.h"
#include "../../../../tools/support/kpool_prepare_fixture.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <bit>
#include <numeric>
#include <type_traits>

namespace ninfer::bench::kpool {
namespace kp = ninfer::tool::kpool;

inline constexpr int page_tokens = 64;
inline constexpr int max_context = ops::kContextBucketTokens.back();

struct Partition {
    ragged::SegmentSpec lengths, histories;

    int segments() const { return static_cast<int>(lengths.size()); }

    int tokens() const { return std::accumulate(lengths.begin(), lengths.end(), 0); }

    int end(int s) const { return histories[s] + lengths[s]; }

    int max_end() const {
        int result = 0;
        for (int s = 0; s < segments(); ++s) result = std::max(result, end(s));
        return result;
    }

    std::string csv(const measurement::Result& result) const {
        return graph_csv(result) + ',' + ragged::format_segments(lengths) + ',' +
               ragged::format_segments(histories);
    }

    void validate_shared_prefix(int shared) const {
        if (shared % page_tokens || shared > *std::min_element(histories.begin(), histories.end()))
            throw std::invalid_argument(
                "--shared-prefix must contain whole pages within every history");
    }

    ops::ContextBucket bucket(std::optional<ops::ContextBucket> override, int envelope = 0) const {
        const auto chosen =
            override.value_or(ops::select_context_bucket(envelope ? envelope : max_end()));
        if (ops::context_bucket_tokens(chosen) < std::uint32_t(max_end()))
            throw std::invalid_argument("--bucket or --envelope is smaller than the live context");
        return chosen;
    }
};

struct PackedOptions {
    std::optional<std::string> segments, histories;

    bool parse(std::string_view flag, std::string_view value) {
        if (flag == "--ragged-segments")
            segments = value;
        else if (flag == "--ragged-histories")
            histories = value;
        else
            return false;
        return true;
    }

    std::vector<Partition> partitions() const {
        if (!segments) {
            if (histories)
                throw std::invalid_argument("--ragged-histories requires --ragged-segments");
            return {};
        }
        auto lengths             = ragged::parse_ragged_specs(*segments);
        std::size_t max_segments = 0;
        for (const auto& length : lengths) max_segments = std::max(max_segments, length.size());
        std::vector<ragged::SegmentSpec> prefixes;
        std::string_view text = histories ? std::string_view(*histories) : "0";
        for (;;) {
            const auto semi = text.find(';');
            auto list       = text.substr(0, semi);
            auto& prefix    = prefixes.emplace_back();
            for (;;) {
                const auto comma = list.find(',');
                const auto item  = list.substr(0, comma);
                const auto x     = item.find('x');
                const int count  = x == item.npos ? 1 : measurement::integer(item.substr(0, x));
                const int value  = measurement::integer(x == item.npos ? item : item.substr(x + 1),
                                                       0, max_context);
                if (std::size_t(count) > max_segments - prefix.size())
                    throw std::invalid_argument("too many ragged histories");
                prefix.insert(prefix.end(), count, value);
                if (comma == list.npos) break;
                list.remove_prefix(comma + 1);
            }
            if (semi == text.npos) break;
            text.remove_prefix(semi + 1);
        }
        if (prefixes.size() != 1 && prefixes.size() != lengths.size())
            throw std::invalid_argument("supply one ragged history list or one per partition");

        std::vector<Partition> result;
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            auto prefix = prefixes[prefixes.size() == 1 ? 0 : i];
            if (prefix.size() == 1) prefix.resize(lengths[i].size(), prefix.front());
            if (prefix.size() != lengths[i].size())
                throw std::invalid_argument("ragged history count must match the segment count");
            for (std::size_t s = 0; s < prefix.size(); ++s)
                if (lengths[i][s] > max_context - prefix[s])
                    throw std::invalid_argument("ragged history plus segment exceeds 1048576");
            result.push_back({std::move(lengths[i]), std::move(prefix)});
        }
        return result;
    }
};

class PackedBatch {
public:
    PackedBatch(Device device, const Partition& partition) : sequence_(partition.lengths, device) {
        const auto seq = sequence_.view();
        std::vector<int> positions(seq.total_tokens()), rows(seq.num_seqs());
        for (int s = 0; s < seq.num_seqs(); ++s) {
            std::iota(positions.begin() + seq.segment_begin(s),
                      positions.begin() + seq.segment_begin(s + 1), partition.histories[s]);
            rows[s] = seq.num_seqs() - 1 - s;
        }
        positions_ = ragged::upload(device, std::span<const int>(positions));
        rows_      = ragged::upload(device, std::span<const int>(rows));
        view       = {seq, Tensor(positions_.span(), DType::I32, {seq.total_tokens()}),
                      Tensor(rows_.span(), DType::I32, {seq.num_seqs()})};
    }

    ops::KpoolMlaRaggedBatchView view;

private:
    ragged::RaggedDeviceFixture sequence_;
    DeviceBuffer positions_, rows_;
};

struct DeviceTensor {
    DeviceBuffer storage;
    Tensor view;

    DeviceTensor(Device device, DType type, std::initializer_list<int> shape)
        : storage(device, Tensor::metadata(type, shape).bytes()),
          view(storage.span(), type, shape) {}
};

using PageLists = std::vector<std::vector<int>>;

inline PageLists history_pages(const Partition& partition, bool write_only = false) {
    PageLists lists(partition.segments());
    for (int s = 0; s < partition.segments(); ++s) {
        const int begin = write_only ? partition.histories[s] / page_tokens : 0;
        const int end   = (partition.end(s) + page_tokens - 1) / page_tokens;
        lists[s].resize(end - begin);
        std::iota(lists[s].begin(), lists[s].end(), begin);
    }
    return lists;
}

// Keep native multilayer page geometry, and populate only the layer planes the measured Op reads.
class PackedCache {
public:
    PackedCache(Device device, ops::KpoolMlaCacheSpec spec, int end, int shared_prefix,
                const PageLists& lists)
        : layout_(ops::plan_kpool_mla_cache_layout(spec)),
          table_(device, DType::I64,
                 {(end + page_tokens - 1) / page_tokens, static_cast<int>(lists.size())}),
          stride_(layout_.page.page_bytes + layout_.page.alignment) {
        const int shared = shared_prefix / page_tokens;
        int pages        = shared;
        for (const auto& list : lists)
            pages +=
                static_cast<int>(list.end() - std::lower_bound(list.begin(), list.end(), shared));
        pages_   = std::max(1, pages);
        storage_ = DeviceBuffer(device, layout_.page.alignment + stride_ * pages_);
        std::fill(table_.host.begin(), table_.host.end(), kInvalidRecordOffset);
        int multiplier = 17;
        while (std::gcd(multiplier, pages_) != 1) multiplier += 2;
        int next = shared;
        for (std::size_t s = 0; s < lists.size(); ++s) {
            for (int p : lists[s]) {
                const int ordinal  = p < shared ? p : next++;
                const int physical = (std::int64_t(ordinal) * multiplier + pages_ / 2) % pages_;
                table_.at(p, lists.size() - 1 - s) = layout_.page.alignment + stride_ * physical;
            }
        }
        table_.upload();
        view = layout_.layer(PagedRecordView{storage_.span(), table_.view, page_tokens},
                             spec.layers / 2);
    }

    ops::KpoolMlaCacheView view;

    int pages() const { return pages_; }

    std::size_t stride() const { return stride_; }

    std::size_t first_offset() const { return layout_.page.alignment; }

private:
    ops::KpoolMlaCacheLayout layout_;
    kp::Buffer<std::int64_t> table_;
    std::size_t stride_;
    int pages_;
    DeviceBuffer storage_;
};

struct Arena {
    DeviceBuffer storage;
    MemoryArena arena;

    Arena(Device device, std::size_t bytes) : storage(device, bytes), arena(storage.span()) {}
};

struct DeviceSelection {
    DeviceBuffer storage;
    ops::KpoolBlockSelection view;

    DeviceSelection(Device device, int n) {
        LayoutBuilder builder;
        const auto layout = ops::plan_kpool_block_selection(builder, n);
        storage           = DeviceBuffer(device, builder.finish());
        view              = layout.bind(storage.span());
    }

    void upload(std::span<const int> ids, std::span<const int> counts, std::span<const int> ends) {
        for (const auto& [tensor, values] : {std::pair{view.pool_ids, ids},
                                             {view.pool_counts, counts},
                                             {view.selection_end, ends}})
            storage.copy_from_host(values.data(), values.size_bytes(),
                                   static_cast<const std::byte*>(tensor.data) -
                                       static_cast<const std::byte*>(storage.data()));
    }
};

namespace fixture_detail {
// Same deterministic BF16 samples as the rectangular tool fixtures, generated on device.
__device__ inline int sample_int(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return int(x % 1025) - 512;
}

template <class T>
static __global__ void fill_samples(T* output, std::size_t size, std::uint32_t seed, float scale) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const auto value = sample_int(i + seed) * scale;
        if constexpr (std::is_same_v<T, __nv_bfloat16>)
            output[i] = __float2bfloat16_rn(value);
        else
            output[i] = value;
    }
}

static __global__ void fill_dense_pages(std::byte* plane, std::size_t stride, int words, int pages,
                                        std::uint32_t seed) {
    for (int p = blockIdx.x; p < pages; p += gridDim.x) {
        auto* row = reinterpret_cast<__nv_bfloat16*>(plane + p * stride);
        for (int d = threadIdx.x; d < words; d += blockDim.x)
            row[d] = __float2bfloat16_rn(sample_int(std::uint64_t(p) * words + d + seed) / 512.F);
    }
}

template <bool Hadamard>
static __global__ void fill_fp8_pages(std::byte* plane, std::byte* scales, std::size_t stride,
                                      int groups, int pages, std::uint32_t seed) {
    __shared__ float warp_values[128];
    const int lane = threadIdx.x;
    for (int p = blockIdx.x; p < pages; p += gridDim.x) {
        auto* codes     = reinterpret_cast<std::uint8_t*>(plane + p * stride);
        auto* exponents = reinterpret_cast<std::uint8_t*>(scales + p * stride);
        for (int group = blockIdx.y; group < groups; group += gridDim.y) {
            const int d = group * 128 + lane;
            float value = __bfloat162float(__float2bfloat16_rn(
                sample_int((std::uint64_t(p) * groups + group) * 128 + lane + seed) / 512.F));
            if constexpr (Hadamard) {
                for (int step = 1; step < 32; step *= 2) {
                    const float other = __shfl_xor_sync(0xffffffff, value, step);
                    value             = lane & step ? other - value : value + other;
                }
                for (int step = 32; step < 128; step *= 2) {
                    warp_values[lane] = value;
                    __syncthreads();
                    const float other = warp_values[lane ^ step];
                    __syncthreads();
                    value = lane & step ? other - value : value + other;
                }
                value *= 0.08838834764831845F;
            }
            float maximum = fabsf(value);
            for (int step = 16; step; step /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
            if (lane % 32 == 0) warp_values[lane / 32] = maximum;
            __syncthreads();
            maximum =
                fmaxf(fmaxf(warp_values[0], warp_values[1]), fmaxf(warp_values[2], warp_values[3]));
            __syncthreads();
            int exponent = 0;
            if (maximum) {
                int e;
                const auto fraction = frexp(double(maximum) / 448.0, &e);
                exponent            = max(-127, e - (fraction == .5));
            }
            if (!lane) exponents[group] = exponent + 127;
            codes[d] = __nv_cvt_float_to_fp8(ldexpf(value, -exponent), __NV_SATFINITE, __NV_E4M3);
        }
    }
}
} // namespace fixture_detail

template <class T = __nv_bfloat16>
inline void fill_samples(Tensor tensor, std::uint32_t seed, float scale, CudaExecutionView cuda) {
    const auto count = tensor.bytes() / sizeof(T);
    fixture_detail::fill_samples<<<fixture::grid(count), 256, 0, cuda.stream()>>>(
        static_cast<T*>(tensor.data), count, seed, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Hadamard>
inline void fill_cache_plane(const PackedCache& cache, PagedPlaneView data, PagedPlaneView scales,
                             int words, std::uint32_t seed, CudaExecutionView cuda) {
    const auto alignment = cache.first_offset();
    auto* plane          = static_cast<std::byte*>(data.data()) + alignment;
    const int grid       = std::min(cache.pages(), 65535);
    if (scales.data()) {
        const int groups = words / 128;
        fixture_detail::fill_fp8_pages<Hadamard>
            <<<dim3(grid, std::min(groups, 32)), 128, 0, cuda.stream()>>>(
                plane, static_cast<std::byte*>(scales.data()) + alignment, cache.stride(), groups,
                cache.pages(), seed);
    } else {
        fixture_detail::fill_dense_pages<<<grid, 256, 0, cuda.stream()>>>(
            plane, cache.stride(), words, cache.pages(), seed);
    }
    CUDA_CHECK(cudaGetLastError());
}

enum class StateAlias { Disjoint, Inplace, Mixed };

inline StateAlias parse_state_alias(std::string_view alias) {
    if (alias == "disjoint") return StateAlias::Disjoint;
    if (alias == "inplace") return StateAlias::Inplace;
    if (alias == "mixed") return StateAlias::Mixed;
    throw std::invalid_argument("--state-alias must be disjoint, inplace or mixed");
}

inline const char* name(StateAlias alias) {
    switch (alias) {
    case StateAlias::Disjoint:
        return "disjoint";
    case StateAlias::Inplace:
        return "inplace";
    case StateAlias::Mixed:
        return "mixed";
    }
    throw std::invalid_argument("invalid packed state alias");
}

class TailState {
public:
    TailState(Device device, const Partition& partition, StateAlias alias) {
        StateImageBuilder builder;
        const auto fields = ops::append_kpool_tail_fields(builder, 128, 11);
        const auto layout = builder.finish();
        const auto form   = alias == StateAlias::Mixed     ? ragged::StateOffsetForm::Mixed
                            : alias == StateAlias::Inplace ? ragged::StateOffsetForm::Alias
                                                           : ragged::StateOffsetForm::Distinct;
        const auto offsets =
            ragged::make_state_offsets(partition.segments(), form, layout.image_bytes, 256, 256);
        source_offsets_ = ragged::upload_offsets(device, offsets.source_offsets_host);
        dest_offsets_   = ragged::upload_offsets(device, offsets.destination_offsets_host);
        state_          = DeviceBuffer(device, offsets.backing_bytes());
        StateImageStorage images(state_.span(), layout);
        const auto bind = [&](DeviceBuffer& offsets_buffer) {
            const auto tensor = ragged::offset_tensor(offsets_buffer);
            return ops::KpoolTailStateView{images.field(fields.raw_keys, tensor, 5),
                                           images.field(fields.raw_gates, tensor, 5)};
        };
        before = bind(source_offsets_);
        after  = bind(dest_offsets_);
        initial_.resize(offsets.source_bytes() / sizeof(std::uint16_t));
        for (int s = 0; s < partition.segments(); ++s) {
            for (int kind = 0; kind < 2; ++kind) {
                const auto& field = kind ? before.raw_gates.field : before.raw_keys.field;
                for (int r = 0; r < partition.histories[s] % 4; ++r) {
                    const auto offset = offsets.source_offsets_host[s] +
                                        field.tensor.region.offset + r * field.byte_strides[1];
                    for (int d = 0; d < 128; ++d)
                        initial_[offset / 2 + d] =
                            kp::sample(d + 128 * r + 777 * s + 99 * kind, 1.F / 512);
                }
            }
        }
        state_.copy_from_host(initial_.data(), initial_.size() * sizeof(std::uint16_t));
        mutates_source_ = offsets.mutates_source;
    }

    void restore(CudaExecutionView cuda) {
        if (mutates_source_)
            CUDA_CHECK(cudaMemcpyAsync(state_.data(), initial_.data(),
                                       initial_.size() * sizeof(std::uint16_t),
                                       cudaMemcpyHostToDevice, cuda.stream()));
    }

    ops::KpoolTailStateView before, after;

private:
    DeviceBuffer source_offsets_, dest_offsets_, state_;
    std::vector<std::uint16_t> initial_;
    bool mutates_source_;
};

struct PrepareFixture {
    PackedBatch batch;
    PackedCache cache;
    TailState tail;
    DeviceTensor x;
    std::optional<DeviceTensor> q, coeff;
    Arena workspace;
    kp::Weights& weights;

    PrepareFixture(CudaExecutionView cuda, const Partition& partition, StateAlias alias,
                   kp::Weights& w, ops::KpoolMlaStorage storage, KvCacheVectorFormat index_format)
        : batch(cuda.device(), partition),
          cache(cuda.device(), {storage, index_format, 512, 128, 11}, partition.max_end(), 0,
                history_pages(partition, true)),
          tail(cuda.device(), partition, alias),
          x(cuda.device(), DType::BF16, {4096, partition.tokens()}),
          workspace(cuda.device(),
                    ops::kpool_mla_prepare_update_ragged_workspace_capacity_bytes(
                        w.profile(), storage, index_format, partition.tokens(), partition.tokens(),
                        partition.segments(), partition.segments(), cuda.multiprocessor_count())),
          weights(w) {
        const int n = partition.tokens();
        if (w.q) q.emplace(cuda.device(), DType::BF16, std::initializer_list<int>{w.q, n});
        if (w.heads)
            coeff.emplace(cuda.device(), DType::FP32, std::initializer_list<int>{w.heads, n});
        fill_samples(x.view, 331, 1.F / 256, cuda);
    }

    void launch(CudaExecutionView cuda) {
        ops::kpool_mla_prepare_update(
            x.view, weights.params, batch.view, tail.before, cache.view, tail.after,
            {q ? q->view : Tensor{}, coeff ? coeff->view : Tensor{}}, workspace.arena, cuda);
    }
};

struct IndexerFixture {
    PackedBatch batch;
    PackedCache cache;
    DeviceTensor query, coeff;
    DeviceSelection selection;
    Arena workspace;
    ops::ContextBucket bucket;

    IndexerFixture(CudaExecutionView cuda, const Partition& partition, int layers, int shared,
                   ops::KpoolMlaStorage storage, KvCacheVectorFormat format,
                   ops::ContextBucket context_bucket, bool ties)
        : batch(cuda.device(), partition),
          cache(cuda.device(), {storage, format, 512, 128, layers}, partition.max_end(), shared,
                history_pages(partition)),
          query(cuda.device(), DType::BF16, {128, 32, partition.tokens()}),
          coeff(cuda.device(), DType::FP32, {32, partition.tokens()}),
          selection(cuda.device(), partition.tokens()),
          workspace(cuda.device(), ops::kpool_indexer_ragged_workspace_capacity_bytes(
                                       {32, 128}, format, ops::ContextBucketRange{context_bucket},
                                       partition.tokens(), partition.tokens(), partition.segments(),
                                       partition.segments(), cuda.multiprocessor_count())),
          bucket(context_bucket) {
        fill_samples(query.view, 719, 1.F / 512, cuda);
        fill_samples<float>(coeff.view, 2071, ties ? 0.F : 1.F / 512, cuda);
        fill_cache_plane<true>(cache, cache.view.index.keys, cache.view.index.scales, 128 * 16,
                               3307, cuda);
    }

    void launch(CudaExecutionView cuda) {
        ops::kpool_indexer(query.view, coeff.view, cache.view.index, batch.view, bucket,
                           workspace.arena, selection.view, cuda);
    }
};

struct SparseSelection {
    std::vector<int> rows, ids, counts, ends;
    PageLists pages;
    std::uint64_t nonempty = 0, nontrivial = 0, visits = 0, attention_visits = 0;
    std::uint64_t unique_tokens = 0, metadata_bytes = 0;

    SparseSelection(const Partition& partition, int cap, bool frozen, int span, int shared)
        : rows(partition.tokens()), ids(512ULL * rows.size(), -1), counts(rows.size(), -1),
          ends(rows.size(), -1), pages(partition.segments()),
          metadata_bytes(4ULL * (rows.size() + partition.segments())) {
        const int shared_pages = shared / page_tokens;
        std::vector<std::uint64_t> shared_masks(shared_pages);
        int base = 0;
        for (int s = 0; s < partition.segments(); ++s) {
            const int length = partition.lengths[s], history = partition.histories[s];
            std::vector<std::uint64_t> masks((partition.end(s) + page_tokens - 1) / page_tokens);
            bool has_selection = false;
            for (int t = 0; t < length;) {
                const int num_queries = std::min(span, length - t);
                const int row         = base + length - 1 - t;
                const int e           = frozen ? history : history + t + 1;
                const int pools = e / 4, count = std::min(cap, pools), tail = e % 4;
                std::fill_n(rows.begin() + base + t, num_queries, row);
                counts[row] = count;
                ends[row]   = e;
                int stride  = 503;
                while (pools && std::gcd(stride, pools) != 1) stride += 2;
                for (int i = 0; i < count; ++i) {
                    const int pool        = (std::int64_t(i) * stride + t + 17LL * s) % pools;
                    ids[512ULL * row + i] = pool;
                    masks[pool / 16] |= UINT64_C(15) << (4 * (pool % 16));
                }
                if (tail) masks[e / page_tokens] |= ((UINT64_C(1) << tail) - 1) << (e % 64 - tail);
                const int tokens = count * 4 + tail;
                if (tokens > 0) {
                    nonempty += num_queries;
                    has_selection = true;
                }
                const auto token_visits = std::uint64_t(tokens) * num_queries;
                visits += token_visits;
                if (tokens > 1) {
                    nontrivial += num_queries;
                    attention_visits += token_visits;
                }
                metadata_bytes += 8 + (count < pools ? 4ULL * count : 0);
                t += num_queries;
            }
            if (has_selection) metadata_bytes += 4;
            for (int p = 0; p < std::ssize(masks); ++p) {
                const auto mask = masks[p];
                if (p < shared_pages) {
                    pages[s].push_back(p);
                    shared_masks[p] |= mask;
                } else if (mask) {
                    pages[s].push_back(p);
                    unique_tokens += std::popcount(mask);
                }
                if (mask) metadata_bytes += 8;
            }
            base += length;
        }
        for (auto mask : shared_masks) unique_tokens += std::popcount(mask);
    }
};

struct SparseFactors {
    kp::Buffer<std::uint16_t> key, value;
    std::unique_ptr<kp::EncodedWeight> key_encoded, value_encoded;
    bool mxfp8;

    SparseFactors(Device device, bool mx)
        : key(device, DType::BF16, {512, 256, 64}), value(device, DType::BF16, {512, 256, 64}),
          mxfp8(mx) {
        for (std::size_t i = 0; i < key.host.size(); ++i) {
            key.host[i]   = kp::sample(i + 1157, 1.F / 8192);
            value.host[i] = kp::sample(i + 7759, 1.F / 8192);
        }
        if (mxfp8) {
            const auto encode = [&](const auto& matrix) {
                std::vector<float> words(matrix.host.size());
                for (std::size_t i = 0; i < words.size(); ++i)
                    words[i] =
                        std::ldexp(kp::decode(matrix.host[i]), int((i / 512 + i / 32) % 5) - 2);
                return std::make_unique<kp::EncodedWeight>(device, QType::MXFP8, 16384, 512, words);
            };
            key_encoded   = encode(key);
            value_encoded = encode(value);
        } else {
            key.upload();
            value.upload();
        }
    }

    ops::KpoolMlaFactors view() const {
        return {key_encoded ? key_encoded->weight : kp::dense_weight(key.view, 16384, 512),
                value_encoded ? value_encoded->weight : kp::dense_weight(value.view, 16384, 512)};
    }

    std::size_t key_bytes() const { return key_encoded ? key_encoded->bytes() : key.view.bytes(); }

    std::size_t value_bytes() const {
        return value_encoded ? value_encoded->bytes() : value.view.bytes();
    }
};

struct SparseFixture {
    PackedBatch batch;
    PackedCache cache;
    DeviceTensor query, output, rows;
    DeviceSelection selection;
    Arena workspace;
    ops::KpoolMlaFactors factors;
    ops::ContextBucket bucket;

    SparseFixture(CudaExecutionView cuda, const Partition& partition, const SparseSelection& input,
                  const PageLists& pages, int layers, int shared, ops::KpoolMlaStorage storage,
                  KvCacheVectorFormat index, ops::ContextBucket context_bucket,
                  ops::KpoolMlaFactors weights)
        : batch(cuda.device(), partition), cache(cuda.device(), {storage, index, 512, 128, layers},
                                                 partition.max_end(), shared, pages),
          query(cuda.device(), DType::BF16, {256, 64, partition.tokens()}),
          output(cuda.device(), DType::BF16, {256, 64, partition.tokens()}),
          rows(cuda.device(), DType::I32, {partition.tokens()}),
          selection(cuda.device(), partition.tokens()),
          workspace(cuda.device(), ops::kpool_sparse_mla_ragged_workspace_capacity_bytes(
                                       ops::kpool_sparse_mla_profile(weights, 64), storage,
                                       ops::ContextBucketRange{context_bucket}, partition.tokens(),
                                       partition.tokens(), partition.segments(),
                                       partition.segments(), cuda.multiprocessor_count())),
          factors(weights), bucket(context_bucket) {
        rows.storage.copy_from_host(input.rows.data(), rows.storage.size());
        selection.upload(input.ids, input.counts, input.ends);
        fill_samples(query.view, 719, 1.F / 512, cuda);
        fill_cache_plane<false>(cache, cache.view.latent.data, cache.view.latent.scales, 512 * 64,
                                3351, cuda);
    }

    void launch(CudaExecutionView cuda) {
        ops::kpool_sparse_mla(query.view, factors, cache.view.latent, batch.view, selection.view,
                              rows.view, 1.F / 16, bucket, workspace.arena, output.view, cuda);
    }
};

} // namespace ninfer::bench::kpool
