// Public-Op benchmark for causal cache Softmax Attention.
//
// Benchmark the public dense and packed append-and-attend and dense cached-only contracts.

#include "ninfer/ops/softmax_attention.h"
#include "core/buffer.h"
#include "../../tools/support/paged_kv_fixture.h"
#include "ninfer/ops/kv_cache_append.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "context_bucket_workload.h"
#include "op_measurement.cuh"
#include "ragged_workload.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

namespace workload = bench::workload;
namespace ragged   = bench::ragged;

constexpr std::int32_t kHeadDim   = 256;
constexpr float kScale            = 0.0625F;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;


enum class Entry : std::uint8_t { Append, Cached, Both };
enum class GeometryChoice : std::uint8_t { H24Kv4, H16Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, Nvfp4, K8V4, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };
enum class PageMapping : std::uint8_t { Identity, Fragmented };

struct Geometry {
    const char* name;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};

constexpr Geometry kH24Kv4{"d256-h24-kv4", 24, 4};
constexpr Geometry kH16Kv2{"d256-h16-kv2", 16, 2};

struct Options {
    Entry entry             = Entry::Both;
    GeometryChoice geometry = GeometryChoice::All;
    KvChoice kv             = KvChoice::All;
    Execution execution     = Execution::Graph;
    CacheMode cache         = CacheMode::Cold;
    PageMapping mapping     = PageMapping::Identity;
    std::vector<std::int32_t> batches{1};
    std::vector<std::int32_t> tokens{1, 2, 4, 6, 8, 12, 16, 1024};
    std::vector<std::int32_t> contexts{0, 128, 2048, 8192};
    std::vector<std::int32_t> row_contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    std::vector<ragged::SegmentSpec> ragged_partitions;
    std::vector<std::int32_t> ragged_histories;
    std::optional<int> shared_prefix_tokens;
    workload::ContextQueryOptions workload;
    int graph_calls = 1;
    int warmup      = 5;
    int repeat      = 30;
    bool profile    = false;
    std::string csv_out;
};

struct Result {
    Entry entry;
    Geometry geometry;
    KvCacheStorage storage;
    Execution execution;
    CacheState cache;
    PageMapping mapping;
    std::int32_t batch;
    std::int32_t tokens;
    std::string row_contexts;
    std::string valid_columns;
    std::string table_rows;
    std::size_t workspace_bytes;
    double logical_bytes;
    double physical_bytes;
    double logical_cache_bytes;
    double key_vector_bytes;
    double value_vector_bytes;
    double physical_cache_bytes;
    double qk_flops;
    double pv_flops;
    double unique_kv_bytes;
    bench::ColdTiming timing;
    std::size_t graph_nodes = 0, workspace_peak = 0;
    int graph_calls = 1;
    ops::ContextBucket bucket{};
    std::string_view length_dist;
    std::string_view valid_dist;
    int shared_prefix_tokens      = 0;
    bool packed                   = false;
    std::int64_t causal_key_reads = 0;
    std::size_t kernel_count      = 0;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_causal_softmax_attention_bench "
                 "[--entry append|cached|both] "
                 "[--geometry d256-h24-kv4|d256-h16-kv2|all] "
                 "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] [--batch B,...] [--tokens W,...] "
                 "[--context L,...] [--row-contexts L0,...] [--valid-columns V0,...] "
                 "[--table-rows R0,...] "
                 "[--ragged-segments 'SPEC[;SPEC...]' [--ragged-histories 'P0,NxP,...'] "
                 "[--shared-prefix-tokens K]] "
                 "[--bucket auto|2k|8k|16k|32k|64k|128k|256k] [--envelope-max N] "
                 "[--length-dist uniform|skewed-1long|staggered|shared-prefix] "
                 "[--valid-dist full|ragged|with-inert] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--mapping identity|fragmented] "
                 "[--warmup N] [--repeat N] [--graph-calls N] [--profile] [--csv-out PATH]\n"
                 "Packed ragged runs use append, committed history per segment, and "
                 "page-aligned shared prefixes.\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, std::int32_t minimum, std::int32_t maximum,
                                     const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
        if (remaining.empty()) { usage(flag); }
    }
    if (result.empty()) { usage(flag); }
    return result;
}

std::vector<std::int32_t> parse_histories(std::string_view spec, std::size_t segments) {
    std::vector<std::int32_t> histories;
    for (;;) {
        const auto comma = spec.find(',');
        const auto item  = spec.substr(0, comma);
        const auto times = item.find('x');
        const auto count = times == item.npos ? 1
                                              : parse_i32(item.substr(0, times), 1, INT32_MAX,
                                                          "--ragged-histories count");
        const auto history =
            parse_i32(times == item.npos ? item : item.substr(times + 1), 0,
                      ops::kCausalAttentionMaximumVisibleKeys, "--ragged-histories");
        if (static_cast<std::size_t>(count) > segments - histories.size())
            usage("ragged history count must equal the segment count");
        histories.insert(histories.end(), count, history);
        if (comma == spec.npos) break;
        spec.remove_prefix(comma + 1);
    }
    if (histories.size() != segments) usage("ragged history count must equal the segment count");
    return histories;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool dense_shape = false;
    std::optional<std::string_view> histories;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "append")
                options.entry = Entry::Append;
            else if (value == "cached")
                options.entry = Entry::Cached;
            else if (value == "both")
                options.entry = Entry::Both;
            else
                usage("--entry expects append, cached, or both");
        } else if (argument == "--geometry") {
            const std::string_view value(next("--geometry requires a value"));
            if (value == "d256-h24-kv4")
                options.geometry = GeometryChoice::H24Kv4;
            else if (value == "d256-h16-kv2")
                options.geometry = GeometryChoice::H16Kv2;
            else if (value == "all")
                options.geometry = GeometryChoice::All;
            else
                usage("--geometry expects d256-h24-kv4, d256-h16-kv2, or all");
        } else if (argument == "--kv-dtype") {
            const std::string_view value(next("--kv-dtype requires a value"));
            if (value == "bf16")
                options.kv = KvChoice::Bf16;
            else if (value == "int8")
                options.kv = KvChoice::Int8;
            else if (value == "fp8")
                options.kv = KvChoice::Fp8;
            else if (value == "nvfp4")
                options.kv = KvChoice::Nvfp4;
            else if (value == "k8v4")
                options.kv = KvChoice::K8V4;
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, fp8, nvfp4, k8v4, or all");
        } else if (argument == "--tokens") {
            dense_shape    = true;
            options.tokens = parse_list(next("--tokens requires a value"), 1, 262144, "--tokens");
        } else if (argument == "--batch") {
            dense_shape     = true;
            options.batches = parse_list(next("--batch requires a value"), 1, INT32_MAX, "--batch");
        } else if (argument == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, 262144, "--context");
        } else if (argument == "--row-contexts") {
            options.row_contexts =
                parse_list(next("--row-contexts requires a value"), 0, 262144, "--row-contexts");
        } else if (argument == "--valid-columns") {
            dense_shape = true;
            options.valid_columns =
                parse_list(next("--valid-columns requires a value"), 0, 16, "--valid-columns");
        } else if (argument == "--ragged-segments") {
            options.ragged_partitions =
                ragged::parse_ragged_specs(next("--ragged-segments requires a value"));
        } else if (argument == "--ragged-histories") {
            histories = next("--ragged-histories requires a value");
        } else if (argument == "--shared-prefix-tokens") {
            options.shared_prefix_tokens =
                parse_i32(next("--shared-prefix-tokens requires a value"), 0,
                          ops::kCausalAttentionMaximumVisibleKeys, "--shared-prefix-tokens");
            if (*options.shared_prefix_tokens % kPagedKVPageSize != 0)
                usage("--shared-prefix-tokens must be a multiple of the KV page size");
        } else if (argument == "--table-rows") {
            options.table_rows =
                parse_list(next("--table-rows requires a value"), 0, INT32_MAX, "--table-rows");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--mapping") {
            const std::string_view value(next("--mapping requires a value"));
            if (value == "identity")
                options.mapping = PageMapping::Identity;
            else if (value == "fragmented")
                options.mapping = PageMapping::Fragmented;
            else
                usage("--mapping expects identity or fragmented");
        } else if (argument == "--bucket") {
            options.workload.bucket =
                workload::parse_bucket_override(next("--bucket requires a value"));
            if (options.workload.bucket)
                ops::ContextBucketRange{*options.workload.bucket}.validate(ops::ContextBucket::K256,
                                                                           "--bucket");
        } else if (argument == "--length-dist") {
            options.workload.length_dist =
                workload::parse_length_dist(next("--length-dist requires a value"));
        } else if (argument == "--valid-dist") {
            options.workload.valid_dist =
                workload::parse_valid_dist(next("--valid-dist requires a value"));
        } else if (argument == "--envelope-max") {
            options.workload.envelope_max =
                parse_i32(next("--envelope-max requires a value"), 1, 262144, "--envelope-max");
        } else if (argument == "--graph-calls") {
            options.graph_calls =
                parse_i32(next("--graph-calls requires a value"), 1, 128, "--graph-calls");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.graph_calls > 1 &&
        (options.cache != CacheMode::Warm || options.execution != Execution::Graph))
        usage("repeated graphs require --cache warm --execution graph");
    if (!options.ragged_partitions.empty()) {
        if (dense_shape || options.entry == Entry::Cached)
            usage("--ragged-segments conflicts with --tokens, --batch, --valid-columns, and "
                  "--entry cached");
        if (options.workload.valid_dist.value_or(workload::ValidDist::Full) !=
            workload::ValidDist::Full)
            usage("packed ragged segments require --valid-dist full");
        if (histories || !options.row_contexts.empty()) {
            if (options.ragged_partitions.size() != 1 ||
                (histories && !options.row_contexts.empty()))
                usage("use one explicit history option with a single ragged partition");
            const auto segments = options.ragged_partitions.front().size();
            options.ragged_histories =
                histories ? parse_histories(*histories, segments) : std::move(options.row_contexts);
            if (options.ragged_histories.size() != segments)
                usage("ragged history count must equal the segment count");
        }
        if (!options.table_rows.empty()) {
            auto sorted = options.table_rows;
            std::sort(sorted.begin(), sorted.end());
            if (sorted.back() == INT32_MAX ||
                std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                usage("ragged table rows must be distinct and fit the table row capacity");
            for (const auto& partition : options.ragged_partitions) {
                if (partition.size() != sorted.size())
                    usage("table row count must equal the segment count of each partition");
            }
        }
        if (options.profile &&
            (options.geometry == GeometryChoice::All || options.kv == KvChoice::All ||
             options.ragged_partitions.size() != 1 ||
             (options.ragged_histories.empty() && options.contexts.size() != 1) ||
             options.execution == Execution::Both || options.cache == CacheMode::Both))
            usage("ragged --profile requires one geometry, dtype, partition, history profile, "
                  "execution, and cache");
        options.entry = Entry::Append;
        return options;
    }
    if (histories || options.shared_prefix_tokens)
        usage("--ragged-histories and --shared-prefix-tokens require --ragged-segments");
    for (const std::int32_t tokens : options.tokens) {
        for (const std::int32_t context : options.contexts) {
            if (context > std::numeric_limits<std::int32_t>::max() - tokens) {
                usage("context + tokens exceeds int32");
            }
        }
    }
    const bool exact_profile = !options.row_contexts.empty() || !options.valid_columns.empty() ||
                               !options.table_rows.empty();
    if (exact_profile && options.entry != Entry::Append) {
        usage("exact row profiles require --entry append");
    }
    if (exact_profile && (options.batches.size() != 1 || options.tokens.size() != 1)) {
        usage("exact row profiles require one batch size and one W");
    }
    if (exact_profile) {
        const auto batch = static_cast<std::size_t>(options.batches.front());
        if ((!options.row_contexts.empty() && options.row_contexts.size() != batch) ||
            (!options.valid_columns.empty() && options.valid_columns.size() != batch) ||
            (!options.table_rows.empty() && options.table_rows.size() != batch)) {
            usage("exact row profile length must equal B");
        }
        for (const std::int32_t valid : options.valid_columns) {
            if (valid > options.tokens.front()) { usage("valid column count exceeds W"); }
        }
        if (!options.table_rows.empty()) {
            std::vector<bool> seen(batch, false);
            for (const std::int32_t row : options.table_rows) {
                if (row < 0 || static_cast<std::size_t>(row) >= batch || seen[row]) {
                    usage("table rows must be a permutation of [0,B)");
                }
                seen[row] = true;
            }
        }
    }
    for (const std::int32_t batch : options.batches) {
        if (batch > 1 && std::any_of(options.tokens.begin(), options.tokens.end(),
                                     [](std::int32_t width) { return width > 16; })) {
            usage("B>1 only supports W<=16");
        }
    }
    if (options.entry == Entry::Cached &&
        std::any_of(options.batches.begin(), options.batches.end(),
                    [](std::int32_t batch) { return batch > 1; })) {
        usage("cached entry is B=1 only");
    }
    if (options.profile &&
        (options.entry == Entry::Both || options.geometry == GeometryChoice::All ||
         options.kv == KvChoice::All || options.batches.size() != 1 || options.tokens.size() != 1 ||
         (options.row_contexts.empty() && options.contexts.size() != 1) ||
         options.execution == Execution::Both || options.cache == CacheMode::Both)) {
        usage("--profile requires one entry, geometry, dtype, B, W, row profile, execution, and "
              "cache");
    }
    return options;
}

std::int32_t align_context(std::int32_t visible) { return ((visible + 127) / 128) * 128; }

KVPageGeometry cache_geometry(const Geometry& geometry, KvCacheStorage storage) {
    const auto layout = paged_kv_storage_layout(storage, kHeadDim);
    KVPageGeometry result;
    result.planes.reserve(4);
    result.planes.push_back(
        {layout.key.data_dtype, layout.key.data_leading_extent, geometry.kv_heads});
    result.planes.push_back(
        {layout.value.data_dtype, layout.value.data_leading_extent, geometry.kv_heads});
    if (layout.key.has_scale())
        result.planes.push_back(
            {layout.key.scale_dtype, layout.key.scale_leading_extent, geometry.kv_heads});
    if (layout.value.has_scale())
        result.planes.push_back(
            {layout.value.scale_dtype, layout.value.scale_leading_extent, geometry.kv_heads});
    return result;
}

PagedKVLayerView make_cache_view(tool::PackedKVFixture& packed, const Tensor& offsets,
                                 const Geometry& geometry, KvCacheStorage storage) {
    PagedKVLayerView result{.k_pages      = packed.plane(0, offsets),
                            .v_pages      = packed.plane(1, offsets),
                            .head_dim     = kHeadDim,
                            .num_kv_heads = geometry.kv_heads,
                            .storage      = storage};
    if (packed.layout().planes.size() > 2) {
        result.k_scale_pages = packed.plane(2, offsets);
        result.v_scale_pages = packed.plane(3, offsets);
    }
    return result;
}

PagedKVBatchLayerView make_batch_cache_view(const PagedKVLayerView& direct) {
    return {direct.k_pages,  direct.v_pages,      direct.k_scale_pages, direct.v_scale_pages,
            direct.head_dim, direct.num_kv_heads, direct.storage};
}

std::size_t workspace_capacity(const Geometry& geometry, KvCacheStorage storage,
                               std::int32_t tokens, std::int32_t batch, ops::ContextBucket bucket,
                               CudaExecutionView execution) {
    return ops::causal_softmax_attention_workspace_capacity_bytes(
        {kHeadDim, geometry.query_heads, geometry.kv_heads}, storage, bucket, batch, tokens, tokens,
        execution.multiprocessor_count());
}

std::int32_t profile_visible(std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns) {
    std::int32_t visible = 0;
    for (std::size_t row = 0; row < contexts.size(); ++row)
        if (valid_columns[row] > 0) visible = std::max(visible, contexts[row] + valid_columns[row]);
    return visible;
}

DeviceBuffer varied_values(std::size_t count, unsigned seed, float scale) {
    return bench::make_bf16(count, seed, -scale, scale);
}

class Case {
public:
    Case(Geometry geometry, KvCacheStorage storage, std::int32_t tokens,
         std::span<const std::int32_t> contexts, std::span<const std::int32_t> valid_columns,
         std::span<const std::int32_t> table_rows, PageMapping mapping, int shared_prefix_tokens,
         const workload::ContextQueryOptions& workload_options, CudaExecutionView execution)
        : execution_(execution), batch_(static_cast<std::int32_t>(contexts.size())),
          masked_(std::any_of(valid_columns.begin(), valid_columns.end(),
                              [tokens](std::int32_t valid) { return valid != tokens; })),
          visible_(profile_visible(contexts, valid_columns)),
          bucket_(workload_options.select_bucket(visible_)),
          padded_(align_context(std::max(1, visible_))), mapping_(mapping),
          logical_pages_(padded_ / kPagedKVPageSize),
          shared_pages_(shared_prefix_tokens / kPagedKVPageSize),
          physical_pages_(shared_pages_ + batch_ * (logical_pages_ - shared_pages_)),
          q_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.query_heads * tokens *
                               batch_,
                           101, .25f)),
          k_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens * batch_,
                           103, .25f)),
          v_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens * batch_,
                           107, 1.f)),
          positions_(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(tokens) * batch_ * sizeof(std::int32_t)),
          valid_columns_(ninfer::bench::device_context().device(),
                         static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          table_rows_(ninfer::bench::device_context().device(),
                      static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          packed_(bench::device_context().device(), cache_geometry(geometry, storage),
                  mapping == PageMapping::Identity ? physical_pages_ : 2 * physical_pages_ + 1),
          block_table_(ninfer::bench::device_context().device(),
                       static_cast<std::size_t>(logical_pages_) * batch_ * sizeof(std::int64_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry.query_heads *
                                    tokens * batch_ * 2)),
          workspace_bytes_(
              workspace_capacity(geometry, storage, tokens, batch_, bucket_, execution_)),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(q_.span(), DType::BF16, {kHeadDim, geometry.query_heads, tokens, batch_}),
          k_tensor_(k_.span(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          v_tensor_(v_.span(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          positions_tensor_(positions_.span(), DType::I32, {tokens, batch_}),
          valid_columns_tensor_(valid_columns_.span(), DType::I32, {batch_}),
          table_rows_tensor_(table_rows_.span(), DType::I32, {batch_}),
          output_tensor_(output_.span(), DType::BF16,
                         {kHeadDim, geometry.query_heads, tokens, batch_}),
          cache_view_(make_cache_view(packed_,
                                      Tensor(block_table_.span(), DType::I64, {logical_pages_}),
                                      geometry, storage)),
          batch_cache_view_(make_batch_cache_view(make_cache_view(
              packed_, Tensor(block_table_.span(), DType::I64, {logical_pages_, batch_}), geometry,
              storage))) {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens) * batch_, 0);
        for (std::int32_t row = 0; row < batch_; ++row) {
            const std::int32_t valid = valid_columns[static_cast<std::size_t>(row)];
            for (std::int32_t token = 0; token < valid; ++token) {
                host_positions[static_cast<std::size_t>(row) * tokens + token] =
                    contexts[static_cast<std::size_t>(row)] + token;
            }
            const std::int32_t padding_position =
                valid == 0 ? 0 : contexts[static_cast<std::size_t>(row)] + valid - 1;
            for (std::int32_t token = valid; token < tokens; ++token) {
                host_positions[static_cast<std::size_t>(row) * tokens + token] = padding_position;
            }
        }
        std::vector<std::int64_t> host_table(static_cast<std::size_t>(logical_pages_) * batch_);
        for (std::int32_t row = 0; row < batch_; ++row) {
            for (std::int32_t page = 0; page < logical_pages_; ++page) {
                const std::int32_t linear =
                    page < shared_pages_ ? page : row * (logical_pages_ - shared_pages_) + page;
                host_table[static_cast<std::size_t>(row) * logical_pages_ + page] =
                    packed_.physical_offset(mapping_ == PageMapping::Identity ? linear
                                                                              : 2 * linear + 1);
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.data(), host_positions.data(), positions_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.data(), host_table.data(), block_table_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid_columns_.data(), valid_columns.data(), valid_columns_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(table_rows_.data(), table_rows.data(), table_rows_.size(),
                              cudaMemcpyHostToDevice));
        // Populate every represented cache row through the public codec, outside measurement.
        // A row view selects its own table while all physical planes remain shared.
        std::vector<std::int32_t> initial_positions(padded_);
        for (int pos = 0; pos < padded_; ++pos) initial_positions[pos] = pos;
        DeviceBuffer dp = bench::make_buffer(std::size_t(padded_) * 4);
        CUDA_CHECK(
            cudaMemcpy(dp.data(), initial_positions.data(), dp.size(), cudaMemcpyHostToDevice));
        const Tensor positions(dp.span(), DType::I32, {padded_});
        for (int row = 0; row < batch_; ++row) {
            const int start = row == 0 ? 0 : shared_prefix_tokens;
            const int count = padded_ - start;
            auto initial_k =
                varied_values(std::size_t(kHeadDim) * geometry.kv_heads * count, 211 + row, .25f);
            auto initial_v =
                varied_values(std::size_t(kHeadDim) * geometry.kv_heads * count, 311 + row, 1.f);
            Tensor k(initial_k.span(), DType::BF16, {kHeadDim, geometry.kv_heads, count});
            Tensor v(initial_v.span(), DType::BF16, {kHeadDim, geometry.kv_heads, count});
            auto row_cache = make_cache_view(
                packed_,
                Tensor(block_table_.span().subspan(std::size_t(row) * logical_pages_ * 8),
                       DType::I64, {logical_pages_}),
                geometry, storage);
            ops::kv_cache_append(k, v, positions.slice(0, start, count), row_cache,
                                 bench::default_execution());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    void launch(Entry entry, CudaExecutionView cuda) {
        if (entry == Entry::Append) {
            const Tensor validity = masked_ ? valid_columns_tensor_ : Tensor{};
            ops::causal_softmax_attention(
                q_tensor_, k_tensor_, v_tensor_, positions_tensor_, validity, table_rows_tensor_,
                {kHeadDim, q_tensor_.ne[1], k_tensor_.ne[1]}, kScale, batch_cache_view_, bucket_,
                workspace_, output_tensor_, cuda);
        } else {
            ops::causal_softmax_attention_cached(
                q_tensor_, positions_tensor_, {kHeadDim, q_tensor_.ne[1], cache_view_.num_kv_heads},
                kScale, cache_view_, bucket_, workspace_, output_tensor_, cuda);
        }
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    [[nodiscard]] ops::ContextBucket bucket() const { return bucket_; }

    [[nodiscard]] std::size_t workspace_peak() const {
        if (workspace_.used() != 0 || workspace_.peak_used() > workspace_bytes_)
            throw std::runtime_error("attention workspace query mismatch");
        return workspace_.peak_used();
    }

private:
    CudaExecutionView execution_;
    std::int32_t batch_;
    bool masked_;
    std::int32_t visible_;
    ops::ContextBucket bucket_;
    std::int32_t padded_;
    PageMapping mapping_;
    std::int32_t logical_pages_;
    std::int32_t shared_pages_;
    std::int32_t physical_pages_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer valid_columns_;
    DeviceBuffer table_rows_;
    tool::PackedKVFixture packed_;
    DeviceBuffer block_table_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor valid_columns_tensor_;
    Tensor table_rows_tensor_;
    Tensor output_tensor_;
    PagedKVLayerView cache_view_;
    PagedKVBatchLayerView batch_cache_view_;
};

struct RowProfile {
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    int shared_prefix_tokens = 0;
};

class RaggedCase {
public:
    RaggedCase(Geometry geometry, KvCacheStorage storage, const RowProfile& rows,
               const Options& options, CudaExecutionView cuda)
        : visible_(profile_visible(rows.contexts, rows.valid_columns)),
          bucket_(options.workload.select_bucket(visible_)),
          logical_pages_((visible_ + kPagedKVPageSize - 1) / kPagedKVPageSize),
          shared_pages_(rows.shared_prefix_tokens / kPagedKVPageSize),
          table_capacity_(*std::max_element(rows.table_rows.begin(), rows.table_rows.end()) + 1),
          geometry_{kHeadDim, geometry.query_heads, geometry.kv_heads},
          ragged_(rows.valid_columns, cuda.device()), view_(ragged_.view()),
          q_(varied_values(elements(geometry.query_heads), 101, .25f)),
          k_(varied_values(elements(geometry.kv_heads), 103, .25f)),
          v_(varied_values(elements(geometry.kv_heads), 107, 1.f)),
          positions_(bench::make_buffer(std::size_t(view_.total_tokens()) * sizeof(std::int32_t))),
          table_rows_(
              ragged::upload(cuda.device(), std::span<const std::int32_t>(rows.table_rows))),
          packed_(cuda.device(), cache_geometry(geometry, storage),
                  physical_page_count(options.mapping)),
          block_table_(bench::make_buffer(std::size_t(logical_pages_) * table_capacity_ *
                                          sizeof(std::int64_t))),
          output_(bench::make_zeros(elements(geometry.query_heads) * sizeof(__nv_bfloat16))),
          workspace_bytes_(ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
              geometry_, storage, bucket_, view_.total_tokens(), view_.total_tokens(),
              view_.num_seqs(), view_.num_seqs(), cuda.multiprocessor_count())),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(q_.span(), DType::BF16, {kHeadDim, geometry.query_heads, view_.total_tokens()}),
          k_tensor_(k_.span(), DType::BF16, {kHeadDim, geometry.kv_heads, view_.total_tokens()}),
          v_tensor_(v_.span(), DType::BF16, {kHeadDim, geometry.kv_heads, view_.total_tokens()}),
          positions_tensor_(positions_.span(), DType::I32, {view_.total_tokens()}),
          table_rows_tensor_(table_rows_.span(), DType::I32, {view_.num_seqs()}),
          output_tensor_(output_.span(), DType::BF16,
                         {kHeadDim, geometry.query_heads, view_.total_tokens()}),
          block_table_tensor_(block_table_.span(), DType::I64, {logical_pages_, table_capacity_}),
          cache_(make_batch_cache_view(
              make_cache_view(packed_, block_table_tensor_, geometry, storage))) {
        std::vector<std::int32_t> positions(view_.total_tokens());
        for (int s = 0; s < view_.num_seqs(); ++s) {
            const auto begin = positions.begin() + view_.segment_begin(s);
            std::iota(begin, begin + view_.segment_length(s), rows.contexts[s]);
        }
        positions_.copy_from_host(positions.data(), positions_.size());

        std::vector<std::int64_t> table(block_table_.size() / sizeof(std::int64_t),
                                        kInvalidRecordOffset);
        auto active_rows = rows.table_rows;
        std::sort(active_rows.begin(), active_rows.end());
        int next_page = shared_pages_;
        for (int row : active_rows) {
            for (int page = 0; page < logical_pages_; ++page) {
                const int record = page < shared_pages_ ? page : next_page++;
                const int physical =
                    options.mapping == PageMapping::Identity ? record : 2 * record + 1;
                table[std::size_t(row) * logical_pages_ + page] = packed_.physical_offset(physical);
            }
        }
        block_table_.copy_from_host(table.data(), block_table_.size());
        populate_history(rows, geometry, storage, cuda);
    }

    void launch(CudaExecutionView cuda) {
        ops::causal_softmax_attention(q_tensor_, k_tensor_, v_tensor_, view_, positions_tensor_,
                                      table_rows_tensor_, geometry_, kScale, cache_, bucket_,
                                      workspace_, output_tensor_, cuda);
    }

    [[nodiscard]] std::size_t workspace_bytes() const { return workspace_bytes_; }

    [[nodiscard]] std::int32_t total_tokens() const { return view_.total_tokens(); }

    [[nodiscard]] ops::ContextBucket bucket() const { return bucket_; }

    [[nodiscard]] std::size_t workspace_peak() const {
        if (workspace_.used() != 0 || workspace_.peak_used() > workspace_bytes_)
            throw std::runtime_error("ragged attention workspace query mismatch");
        return workspace_.peak_used();
    }

private:
    [[nodiscard]] std::size_t elements(int heads) const {
        return std::size_t(kHeadDim) * heads * view_.total_tokens();
    }

    [[nodiscard]] int physical_page_count(PageMapping mapping) const {
        const auto pages =
            shared_pages_ + std::int64_t(view_.num_seqs()) * (logical_pages_ - shared_pages_);
        const auto capacity = mapping == PageMapping::Identity ? pages : 2 * pages + 1;
        if (capacity > INT32_MAX) throw std::invalid_argument("too many physical KV pages");
        return static_cast<int>(capacity);
    }

    void populate_history(const RowProfile& rows, Geometry geometry, KvCacheStorage storage,
                          CudaExecutionView cuda) {
        const int max_history = *std::max_element(rows.contexts.begin(), rows.contexts.end());
        if (max_history == 0) return;
        std::vector<std::int32_t> host_positions(max_history);
        std::iota(host_positions.begin(), host_positions.end(), 0);
        auto positions =
            ragged::upload(cuda.device(), std::span<const std::int32_t>(host_positions));
        const Tensor p(positions.span(), DType::I32, {max_history});
        const auto kv_elements = std::size_t(kHeadDim) * geometry.kv_heads;
        auto keys   = bench::make_buffer(kv_elements * max_history * sizeof(__nv_bfloat16));
        auto values = bench::make_buffer(keys.size());
        for (int s = 0; s < view_.num_seqs(); ++s) {
            // The first segment initializes the common pages. Later segments only write their
            // tails.
            const int start = s == 0 ? 0 : rows.shared_prefix_tokens;
            const int count = rows.contexts[s] - start;
            if (count == 0) continue;
            CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(keys.data()),
                                                   kv_elements * count, 211U + s, -.25f, .25f,
                                                   cuda.stream()));
            CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(values.data()),
                                                   kv_elements * count, 311U + s, -1.f, 1.f,
                                                   cuda.stream()));
            const Tensor k(keys.span(), DType::BF16, {kHeadDim, geometry.kv_heads, count});
            const Tensor v(values.span(), DType::BF16, {kHeadDim, geometry.kv_heads, count});
            auto row_cache = make_cache_view(
                packed_, block_table_tensor_.slice(1, rows.table_rows[s], 1), geometry, storage);
            ops::kv_cache_append(k, v, p.slice(0, start, count), row_cache, cuda);
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    }

    int visible_;
    ops::ContextBucket bucket_;
    int logical_pages_, shared_pages_, table_capacity_;
    ops::AttentionHeadGeometry geometry_;
    ragged::RaggedDeviceFixture ragged_;
    ops::RaggedSequenceView view_;
    DeviceBuffer q_, k_, v_, positions_, table_rows_;
    tool::PackedKVFixture packed_;
    DeviceBuffer block_table_, output_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    Tensor q_tensor_, k_tensor_, v_tensor_, positions_tensor_, table_rows_tensor_, output_tensor_;
    Tensor block_table_tensor_;
    PagedKVBatchLayerView cache_;
};

const char* entry_name(Entry entry) { return entry == Entry::Append ? "append" : "cached"; }

const char* storage_name(KvCacheStorage storage) { return kv_cache_profile_info(storage).name; }

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

const char* mapping_name(PageMapping mapping) {
    return mapping == PageMapping::Identity ? "identity" : "fragmented";
}

std::string profile_name(std::span<const std::int32_t> values) {
    std::string result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) { result.push_back(';'); }
        result += std::to_string(values[index]);
    }
    return result;
}

double cache_vector_bytes(const PagedKVVectorLayout& layout) {
    return static_cast<double>(layout.physical_bytes());
}

std::int64_t causal_key_sum(std::span<const std::int32_t> contexts,
                            std::span<const std::int32_t> valid_columns) {
    std::int64_t total = 0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        const std::int64_t tokens = valid_columns[row];
        total += tokens * contexts[row] + tokens * (tokens + 1) / 2;
    }
    return total;
}

double modeled_bytes(Entry entry, const Geometry& geometry, KvCacheStorage storage,
                     std::span<const std::int32_t> contexts,
                     std::span<const std::int32_t> valid_columns, bool physical_cache) {
    const auto layout              = paged_kv_storage_layout(storage, kHeadDim);
    std::int64_t valid_token_count = 0;
    for (const std::int32_t valid : valid_columns) { valid_token_count += valid; }
    const double valid_tokens = static_cast<double>(valid_token_count);
    const double q_and_output = 2.0 * kHeadDim * geometry.query_heads * valid_tokens * 2.0;
    const double key_bytes =
        physical_cache ? cache_vector_bytes(layout.key) : layout.logical_vector_bytes();
    const double value_bytes =
        physical_cache ? cache_vector_bytes(layout.value) : layout.logical_vector_bytes();
    const double cache_reads =
        causal_key_sum(contexts, valid_columns) * geometry.kv_heads * (key_bytes + value_bytes);
    if (entry == Entry::Cached) { return q_and_output + cache_reads; }
    const double input_kv     = 2.0 * kHeadDim * geometry.kv_heads * valid_tokens * 2.0;
    const double cache_writes = geometry.kv_heads * valid_tokens * (key_bytes + value_bytes);
    return q_and_output + cache_reads + input_kv + cache_writes;
}

double visible_cache_vectors(std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns) {
    double visible_rows = 0.0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        if (valid_columns[row] > 0) { visible_rows += contexts[row] + valid_columns[row]; }
    }
    return visible_rows;
}

double cache_footprint_bytes(const Geometry& geometry, KvCacheStorage storage,
                             std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns, bool physical) {
    const auto layout = paged_kv_storage_layout(storage, kHeadDim);
    const double key_bytes =
        physical ? cache_vector_bytes(layout.key) : layout.logical_vector_bytes();
    const double value_bytes =
        physical ? cache_vector_bytes(layout.value) : layout.logical_vector_bytes();
    return visible_cache_vectors(contexts, valid_columns) * geometry.kv_heads *
           (key_bytes + value_bytes);
}

double contraction_flops(const Geometry& geometry, std::span<const std::int32_t> contexts,
                         std::span<const std::int32_t> valid_columns) {
    return 2.0 * kHeadDim * geometry.query_heads * causal_key_sum(contexts, valid_columns);
}

double unique_kv_bytes(const Geometry& geometry, KvCacheStorage storage,
                       std::span<const std::int32_t> contexts,
                       std::span<const std::int32_t> valid_columns, int shared_prefix_tokens) {
    const auto layout        = paged_kv_storage_layout(storage, kHeadDim);
    const auto live          = std::count_if(valid_columns.begin(), valid_columns.end(),
                                             [](int count) { return count > 0; });
    const double shared_rows = live > 1 ? double(live - 1) * shared_prefix_tokens : 0.0;
    return cache_footprint_bytes(geometry, storage, contexts, valid_columns, true) -
           shared_rows * geometry.kv_heads *
               (cache_vector_bytes(layout.key) + cache_vector_bytes(layout.value));
}

std::string_view length_distribution(const Options& options) {
    return options.row_contexts.empty() && options.ragged_histories.empty()
               ? options.workload.length_name()
               : "explicit";
}

std::string_view valid_distribution(const Options& options) {
    return options.valid_columns.empty() ? options.workload.valid_name() : "explicit";
}

bench::ColdTiming measure(const auto& launch, Execution execution, CacheState cache,
                          const bench::measurement::BodyTimedGraph& graph,
                          bench::L2FlushBuffer& flush, CudaExecutionView cuda, int warmup,
                          int repeat) {
    if (execution == Execution::Eager) {
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, cuda, warmup, repeat)
                   : bench::measure_launch(launch, cuda, warmup, repeat);
    }
    return cache == CacheState::Cold ? bench::measure_cold_graph(graph, flush, cuda, warmup, repeat)
                                     : bench::measure_graph(graph, cuda, warmup, repeat);
}

void report(const Result& result) {
    const double seconds       = result.timing.median_us * 1.0e-6;
    const double logical_gbps  = result.logical_bytes / seconds / 1.0e9;
    const double physical_gbps = result.physical_bytes / seconds / 1.0e9;
    const double cache_gbps    = result.unique_kv_bytes / seconds / 1.0e9;
    const double tflops        = (result.qk_flops + result.pv_flops) / seconds / 1.0e12;
    const double qk_tflops     = result.qk_flops / seconds / 1.0e12;
    const double pv_tflops     = result.pv_flops / seconds / 1.0e12;
    std::printf(
        "entry=%-6s geometry=%-14s kv=%-6s mapping=%-10s execution=%-5s cache=%-4s mode=%s "
        "B=%d W=%d contexts=%s valid=%s rows=%s bucket=%s length_dist=%s valid_dist=%s "
        "shared_prefix=%d "
        "workspace=%9zu peak=%9zu nodes=%zu calls=%d median=%10.3f us min=%10.3f us p95=%10.3f us "
        "logical_payload=%8.1f GB/s physical_payload=%8.1f GB/s math=%7.2f TFLOP/s\n",
        entry_name(result.entry), result.geometry.name, storage_name(result.storage),
        mapping_name(result.mapping), execution_name(result.execution), cache_name(result.cache),
        result.packed ? "ragged" : "dense", result.batch, result.tokens,
        result.row_contexts.c_str(), result.valid_columns.c_str(), result.table_rows.c_str(),
        ops::context_bucket_name(result.bucket).data(), result.length_dist.data(),
        result.valid_dist.data(), result.shared_prefix_tokens, result.workspace_bytes,
        result.workspace_peak, result.graph_nodes, result.graph_calls, result.timing.median_us,
        result.timing.min_us, result.timing.p95_us, logical_gbps, physical_gbps, tflops);
    std::printf("  vectors K=%.0f V=%.0f bytes cache logical=%.0f physical=%.0f bytes "
                "qk_flops=%.0f pv_flops=%.0f qk_full_op=%7.2f TFLOP/s "
                "pv_full_op=%7.2f TFLOP/s unique_kv=%.0f bytes "
                "unique_cache_rate=%8.1f GB/s\n",
                result.key_vector_bytes, result.value_vector_bytes, result.logical_cache_bytes,
                result.physical_cache_bytes, result.qk_flops, result.pv_flops, qk_tflops, pv_tflops,
                result.unique_kv_bytes, cache_gbps);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,geometry,kv_dtype,mapping,execution,cache,B,W,row_contexts,valid_columns,"
              "table_rows,workspace_bytes,logical_bytes,physical_bytes,logical_cache_bytes,"
              "key_vector_bytes,value_vector_bytes,physical_cache_bytes,qk_flops,pv_flops,"
              "qk_full_op_tflops,pv_full_op_tflops,"
              "unique_kv_bytes,"
              "unique_kv_gbps,median_us,min_us,p95_us,graph_nodes,workspace_peak_bytes,graph_calls,"
              "bucket,bucket_tokens,length_dist,valid_dist,shared_prefix_tokens,"
              "mode,causal_key_reads,kernel_count\n";
    for (const Result& result : results) {
        output << std::setprecision(result.packed ? std::numeric_limits<double>::max_digits10 : 6);
        output << entry_name(result.entry) << ',' << result.geometry.name << ','
               << storage_name(result.storage) << ',' << mapping_name(result.mapping) << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.batch << ',' << result.tokens << ',' << result.row_contexts << ','
               << result.valid_columns << ',' << result.table_rows << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.physical_bytes << ','
               << result.logical_cache_bytes << ',' << result.key_vector_bytes << ','
               << result.value_vector_bytes << ',' << result.physical_cache_bytes << ','
               << result.qk_flops << ',' << result.pv_flops << ',';
        const double seconds = result.timing.median_us * 1.0e-6;
        output << result.qk_flops / seconds / 1.0e12 << ',' << result.pv_flops / seconds / 1.0e12
               << ',' << result.unique_kv_bytes << ',' << result.unique_kv_bytes / seconds / 1.0e9;
        output << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << result.graph_nodes << ',' << result.workspace_peak
               << ',' << result.graph_calls << ',' << ops::context_bucket_name(result.bucket) << ','
               << ops::context_bucket_tokens(result.bucket) << ',' << result.length_dist << ','
               << result.valid_dist << ',' << result.shared_prefix_tokens << ','
               << (result.packed ? "ragged" : "dense") << ',' << result.causal_key_reads << ','
               << result.kernel_count << '\n';
    }
}

void profile(const auto& launch, ops::ContextBucket bucket, Entry entry, const Geometry& geometry,
             KvCacheStorage storage, const Options& options, std::int32_t batch, std::int32_t width,
             std::string_view contexts, std::string_view valid_columns, std::string_view table_rows,
             bench::L2FlushBuffer& flush, CudaExecutionView cuda) {
    const auto stream         = cuda.stream();
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::measurement::BodyTimedGraph graph;
    if (execution == Execution::Graph) {
        launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) {
            for (int call = 0; call < options.graph_calls; ++call) launch(cuda);
        });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(cuda); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { launch(cuda); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf(
        "PROFILE entry=%s geometry=%s kv=%s mapping=%s dispatch=public execution=%s cache=%s "
        "B=%d W=%d contexts=%.*s valid=%.*s rows=%.*s bucket=%s length_dist=%s valid_dist=%s "
        "graph_calls=%d\n",
        entry_name(entry), geometry.name, storage_name(storage), mapping_name(options.mapping),
        execution_name(execution), cache_name(cache), batch, width,
        static_cast<int>(contexts.size()), contexts.data(), static_cast<int>(valid_columns.size()),
        valid_columns.data(), static_cast<int>(table_rows.size()), table_rows.data(),
        ops::context_bucket_name(bucket).data(), length_distribution(options).data(),
        valid_distribution(options).data(), options.graph_calls);
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(cuda);
    else
        launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<Geometry> selected_geometries(GeometryChoice choice) {
    if (choice == GeometryChoice::H24Kv4) { return {kH24Kv4}; }
    if (choice == GeometryChoice::H16Kv2) { return {kH16Kv2}; }
    return {kH24Kv4, kH16Kv2};
}

std::vector<KvCacheStorage> selected_storages(KvChoice choice) {
    if (choice == KvChoice::Bf16) { return {KvCacheStorage::BFloat16}; }
    if (choice == KvChoice::Int8) { return {KvCacheStorage::Int8Group64}; }
    if (choice == KvChoice::Fp8) { return {KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}; }
    if (choice == KvChoice::Nvfp4) { return {KvCacheStorage::Nvfp4Group16}; }
    if (choice == KvChoice::K8V4) { return {KvCacheStorage::Fp8KeyNvfp4Value}; }
    return {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
            KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
            KvCacheStorage::Fp8KeyNvfp4Value};
}

RowProfile make_row_profile(const Options& options, std::int32_t batch, std::int32_t width,
                            std::int32_t uniform_context) {
    RowProfile profile;
    auto settings = options.workload;
    if (!options.valid_columns.empty()) settings.valid_dist.reset();
    auto generated = workload::generate_context_query_workload(uniform_context, width, batch,
                                                               settings, kPagedKVPageSize);
    profile.contexts =
        options.row_contexts.empty() ? std::move(generated.lengths) : options.row_contexts;
    profile.valid_columns =
        options.valid_columns.empty() ? std::move(generated.valid_columns) : options.valid_columns;
    if (options.row_contexts.empty() &&
        std::any_of(profile.valid_columns.begin(), profile.valid_columns.end(),
                    [](int count) { return count > 0; })) {
        profile.shared_prefix_tokens = generated.shared_prefix_tokens;
    }
    if (options.table_rows.empty()) {
        profile.table_rows.resize(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            profile.table_rows[static_cast<std::size_t>(row)] = row;
        }
    } else {
        profile.table_rows = options.table_rows;
    }
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t context = profile.contexts[static_cast<std::size_t>(row)];
        const std::int32_t valid   = profile.valid_columns[static_cast<std::size_t>(row)];
        if (valid < 0 || valid > width || context < 0 ||
            (valid > 0 &&
             context >
                 static_cast<std::int32_t>(ops::kCausalAttentionMaximumVisibleKeys) - valid)) {
            throw std::invalid_argument("row profile exceeds the public causal-attention domain");
        }
    }
    return profile;
}

RowProfile make_ragged_profile(const Options& options, const ragged::SegmentSpec& lengths,
                               int context) {
    RowProfile profile;
    profile.valid_columns = lengths;
    profile.contexts      = options.ragged_histories;
    if (profile.contexts.empty()) {
        profile.contexts.assign(lengths.size(), context);
        switch (options.workload.length_dist.value_or(workload::LengthDist::Uniform)) {
        case workload::LengthDist::Uniform:
            break;
        case workload::LengthDist::Skewed1Long:
            std::fill(profile.contexts.begin() + 1, profile.contexts.end(), std::min(context, 256));
            break;
        case workload::LengthDist::Staggered:
            for (std::size_t s = 1; s < lengths.size(); ++s)
                profile.contexts[s] = std::int64_t(context) * (lengths.size() - s) / lengths.size();
            break;
        case workload::LengthDist::SharedPrefix:
            profile.shared_prefix_tokens = (context / 2) / kPagedKVPageSize * kPagedKVPageSize;
            break;
        }
    }
    profile.shared_prefix_tokens =
        options.shared_prefix_tokens.value_or(profile.shared_prefix_tokens);
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        if (std::int64_t(profile.contexts[s]) + lengths[s] >
            ops::kCausalAttentionMaximumVisibleKeys)
            throw std::invalid_argument(
                "ragged segment exceeds the public causal-attention domain");
        if (profile.shared_prefix_tokens > profile.contexts[s])
            throw std::invalid_argument("shared prefix exceeds a segment's committed history");
    }
    profile.table_rows = options.table_rows;
    if (profile.table_rows.empty()) {
        profile.table_rows.resize(lengths.size());
        std::iota(profile.table_rows.begin(), profile.table_rows.end(), 0);
    }
    return profile;
}

void run_case(const Options& options, const auto& data, const auto& launch, Entry entry,
              Geometry geometry, KvCacheStorage storage, const RowProfile& rows, std::int32_t width,
              bench::L2FlushBuffer& flush, CudaExecutionView cuda, std::vector<Result>& results) {
    const bool packed = !options.ragged_partitions.empty();
    const auto batch  = static_cast<std::int32_t>(rows.contexts.size());
    const auto format = [packed, batch](const auto& values) {
        return packed && batch > 8 ? ragged::format_segments(values) : profile_name(values);
    };
    const auto contexts = format(rows.contexts);
    const auto valid    = format(rows.valid_columns);
    const auto table    = profile_name(rows.table_rows);
    if (options.profile) {
        profile(launch, data.bucket(), entry, geometry, storage, options, batch, width, contexts,
                valid, table, flush, cuda);
        return;
    }

    bench::measurement::BodyTimedGraph graph;
    if (packed || options.execution != Execution::Eager) {
        launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        graph.capture(cuda, [&](CudaExecutionView cuda) {
            for (int call = 0; call < options.graph_calls; ++call) launch(cuda);
        });
    }

    const auto storage_layout = paged_kv_storage_layout(storage, kHeadDim);
    const auto flops          = contraction_flops(geometry, rows.contexts, rows.valid_columns);
    const Result metrics{
        .entry           = entry,
        .geometry        = geometry,
        .storage         = storage,
        .mapping         = options.mapping,
        .batch           = batch,
        .tokens          = width,
        .row_contexts    = contexts,
        .valid_columns   = valid,
        .table_rows      = table,
        .workspace_bytes = data.workspace_bytes(),
        .logical_bytes =
            modeled_bytes(entry, geometry, storage, rows.contexts, rows.valid_columns, false),
        .physical_bytes =
            modeled_bytes(entry, geometry, storage, rows.contexts, rows.valid_columns, true),
        .logical_cache_bytes =
            cache_footprint_bytes(geometry, storage, rows.contexts, rows.valid_columns, false),
        .key_vector_bytes   = cache_vector_bytes(storage_layout.key),
        .value_vector_bytes = cache_vector_bytes(storage_layout.value),
        .physical_cache_bytes =
            cache_footprint_bytes(geometry, storage, rows.contexts, rows.valid_columns, true),
        .qk_flops        = flops,
        .pv_flops        = flops,
        .unique_kv_bytes = unique_kv_bytes(geometry, storage, rows.contexts, rows.valid_columns,
                                           rows.shared_prefix_tokens),
        .bucket          = data.bucket(),
        .length_dist     = length_distribution(options),
        .valid_dist      = valid_distribution(options),
        .shared_prefix_tokens = rows.shared_prefix_tokens,
        .packed               = packed,
        .causal_key_reads     = causal_key_sum(rows.contexts, rows.valid_columns),
    };
    for (const Execution execution : {Execution::Eager, Execution::Graph}) {
        if (options.execution != Execution::Both && options.execution != execution) continue;
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            auto result      = metrics;
            result.execution = execution;
            result.cache     = cache;
            result.timing    = measure(launch, execution, cache, graph, flush, cuda, options.warmup,
                                       options.repeat);
            result.workspace_peak = data.workspace_peak();
            if (packed || execution == Execution::Graph) {
                result.graph_calls = execution == Execution::Graph ? options.graph_calls : 1;
                // With --execution both, the graph probe always contains one call.
                result.graph_nodes  = graph.body_nodes();
                result.kernel_count = graph.kernels();
            }
            result.timing.median_us /= result.graph_calls;
            result.timing.min_us /= result.graph_calls;
            result.timing.p95_us /= result.graph_calls;
            report(result);
            results.push_back(std::move(result));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        auto& device          = ninfer::bench::device_context();
        const auto cuda       = device.execution_view();
        const auto stream     = cuda.stream();
        bench::L2FlushBuffer flush(kFlushBytes);
        const std::vector<Geometry> geometries     = selected_geometries(options.geometry);
        const std::vector<KvCacheStorage> storages = selected_storages(options.kv);

        std::vector<Result> results;
        const auto& contexts = options.ragged_histories.empty() && options.row_contexts.empty()
                                   ? options.contexts
                                   : std::vector<std::int32_t>{0};
        for (const Geometry& geometry : geometries) {
            for (const KvCacheStorage storage : storages) {
                if (!options.ragged_partitions.empty()) {
                    for (const auto context : contexts) {
                        for (const auto& lengths : options.ragged_partitions) {
                            const auto rows = make_ragged_profile(options, lengths, context);
                            RaggedCase data(geometry, storage, rows, options, cuda);
                            const auto launch = [&](CudaExecutionView cuda) { data.launch(cuda); };
                            run_case(options, data, launch, Entry::Append, geometry, storage, rows,
                                     data.total_tokens(), flush, cuda, results);
                        }
                    }
                    continue;
                }
                for (const auto batch : options.batches) {
                    for (const auto context : contexts) {
                        for (const auto width : options.tokens) {
                            const auto rows = make_row_profile(options, batch, width, context);
                            Case data(geometry, storage, width, rows.contexts, rows.valid_columns,
                                      rows.table_rows, options.mapping, rows.shared_prefix_tokens,
                                      options.workload, cuda);
                            for (const Entry entry : {Entry::Append, Entry::Cached}) {
                                if ((options.entry != Entry::Both && options.entry != entry) ||
                                    (entry == Entry::Cached && batch != 1))
                                    continue;
                                const auto launch = [&](CudaExecutionView cuda) {
                                    data.launch(entry, cuda);
                                };
                                run_case(options, data, launch, entry, geometry, storage, rows,
                                         width, flush, cuda, results);
                            }
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_causal_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
