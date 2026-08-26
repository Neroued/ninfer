// Public-Op benchmark for causal cache Softmax Attention.
//
// The two benchmark entries map directly to the public append-and-attend and cached-only
// contracts. Decode, small-T, prompt, split-KV, and kernel selection remain private production
// implementation details and never enter this benchmark's dispatch or output schema.

#include "ninfer/ops/softmax_attention.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kKvGroup          = 64;
constexpr std::int32_t kFp8KvGroup       = 256;
constexpr float kScale                   = 0.0625F;
constexpr std::size_t kFlushBytes        = std::size_t{256} << 20;
constexpr double kDenseF16Bf16TcTflops   = 209.5;
constexpr double kDenseFp8TcTflops       = 419.0;
constexpr double kMixedFp8F16TcTflops    = 279.333;
constexpr double kRtx5090DramGBs         = 1792.0;
constexpr double kColdPureReadCeilingGBs = 1674.5;

enum class Entry : std::uint8_t { Append, Cached, Both };
enum class GeometryChoice : std::uint8_t { H24Kv4, H16Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, All };
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
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    std::string csv_out;
};

struct Result {
    Entry entry;
    Geometry geometry;
    DType kv_dtype;
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
    double qk_flops;
    double pv_flops;
    double physical_kv_read_bytes;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_causal_softmax_attention_bench "
                 "[--entry append|cached|both] "
                 "[--geometry d256-h24-kv4|d256-h16-kv2|all] "
                 "[--kv-dtype bf16|int8|fp8|all] [--batch B,...] [--tokens W,...] "
                 "[--context L,...] [--row-contexts L0,...] [--valid-columns V0,...] "
                 "[--table-rows R0,...] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--mapping identity|fragmented] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
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
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
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
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, fp8, or all");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 262144, "--tokens");
        } else if (argument == "--batch") {
            options.batches = parse_list(next("--batch requires a value"), 1, 8, "--batch");
        } else if (argument == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, 262144, "--context");
        } else if (argument == "--row-contexts") {
            options.row_contexts =
                parse_list(next("--row-contexts requires a value"), 0, 262144, "--row-contexts");
        } else if (argument == "--valid-columns") {
            options.valid_columns =
                parse_list(next("--valid-columns requires a value"), 0, 16, "--valid-columns");
        } else if (argument == "--table-rows") {
            options.table_rows =
                parse_list(next("--table-rows requires a value"), 0, 7, "--table-rows");
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

std::size_t cache_plane_bytes(const Geometry& geometry, DType dtype, std::int32_t physical_pages) {
    return static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * kPagedKVPageSize *
           physical_pages * dtype_size(dtype);
}

std::int32_t scale_groups(DType dtype) {
    if (dtype == DType::I8) return kHeadDim / kKvGroup;
    if (dtype == DType::FP8_E4M3FN) return kHeadDim / kFp8KvGroup;
    return 0;
}

std::size_t scale_plane_bytes(const Geometry& geometry, DType dtype, std::int32_t physical_pages) {
    return static_cast<std::size_t>(scale_groups(dtype)) * geometry.kv_heads * kPagedKVPageSize *
           physical_pages * dtype_size(DType::FP16);
}

PagedKVLayerView make_cache_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                 DeviceBuffer& v_scale, DeviceBuffer& block_table,
                                 const Geometry& geometry, DType dtype, std::int32_t padded) {
    const bool quantized              = dtype != DType::BF16;
    const std::int32_t groups         = scale_groups(dtype);
    const std::int32_t logical_pages  = padded / kPagedKVPageSize;
    const std::int32_t physical_pages = static_cast<std::int32_t>(
        k.bytes / (static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * kPagedKVPageSize *
                   dtype_size(dtype)));
    return {
        .k_pages =
            Tensor(k.p, dtype, {kHeadDim, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .v_pages =
            Tensor(v.p, dtype, {kHeadDim, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .k_scale_pages = quantized
                             ? Tensor(k_scale.p, DType::FP16,
                                      {groups, kPagedKVPageSize, geometry.kv_heads, physical_pages})
                             : Tensor(),
        .v_scale_pages = quantized
                             ? Tensor(v_scale.p, DType::FP16,
                                      {groups, kPagedKVPageSize, geometry.kv_heads, physical_pages})
                             : Tensor(),
        .block_table   = Tensor(block_table.p, DType::I32, {logical_pages}),
        .head_dim      = kHeadDim,
        .num_kv_heads  = geometry.kv_heads,
        .dtype         = dtype,
        .quant_group   = dtype == DType::I8           ? kKvGroup
                         : dtype == DType::FP8_E4M3FN ? kFp8KvGroup
                                                      : 0,
    };
}

PagedKVBatchLayerView make_batch_cache_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
    };
}

PagedKVBatchLayerView make_batch_cache_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                            DeviceBuffer& v_scale, DeviceBuffer& block_tables,
                                            const Geometry& geometry, DType dtype,
                                            std::int32_t padded, std::int32_t table_rows) {
    const PagedKVLayerView direct =
        make_cache_view(k, v, k_scale, v_scale, block_tables, geometry, dtype, padded);
    PagedKVBatchLayerView result = make_batch_cache_view(direct);
    result.block_tables =
        Tensor(block_tables.p, DType::I32, {padded / kPagedKVPageSize, table_rows});
    return result;
}

std::size_t workspace_capacity(const Geometry& geometry, DType dtype, std::int32_t tokens,
                               std::int32_t batch, std::int32_t visible) {
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                                         static_cast<std::uint32_t>(visible)};
    return ops::causal_softmax_attention_workspace_capacity_bytes(
        {kHeadDim, geometry.query_heads, geometry.kv_heads}, dtype, envelope, batch, tokens,
        tokens);
}

std::int32_t profile_visible(std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns) {
    std::int32_t visible = 1;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        visible = std::max(visible, contexts[row] + valid_columns[row]);
    }
    return visible;
}

class Case {
public:
    Case(Geometry geometry, DType dtype, std::int32_t tokens,
         std::span<const std::int32_t> contexts, std::span<const std::int32_t> valid_columns,
         std::span<const std::int32_t> table_rows, PageMapping mapping)
        : batch_(static_cast<std::int32_t>(contexts.size())),
          masked_(std::any_of(valid_columns.begin(), valid_columns.end(),
                              [tokens](std::int32_t valid) { return valid != tokens; })),
          visible_(profile_visible(contexts, valid_columns)), padded_(align_context(visible_)),
          mapping_(mapping), logical_pages_(padded_ / kPagedKVPageSize),
          physical_pages_(mapping == PageMapping::Identity ? batch_ * logical_pages_
                                                           : 2 * batch_ * logical_pages_ + 1),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.query_heads * tokens *
                              batch_)),
          k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens *
                              batch_)),
          v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens *
                              batch_)),
          positions_(static_cast<std::size_t>(tokens) * batch_ * sizeof(std::int32_t)),
          valid_columns_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          table_rows_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          cache_k_(bench::make_zeros(cache_plane_bytes(geometry, dtype, physical_pages_))),
          cache_v_(bench::make_zeros(cache_plane_bytes(geometry, dtype, physical_pages_))),
          cache_k_scale_(bench::make_zeros(dtype != DType::BF16
                                               ? scale_plane_bytes(geometry, dtype, physical_pages_)
                                               : std::size_t{1})),
          cache_v_scale_(bench::make_zeros(dtype != DType::BF16
                                               ? scale_plane_bytes(geometry, dtype, physical_pages_)
                                               : std::size_t{1})),
          block_table_(static_cast<std::size_t>(logical_pages_) * batch_ * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry.query_heads *
                                    tokens * batch_ * 2)),
          workspace_bytes_(workspace_capacity(geometry, dtype, tokens, batch_, visible_)),
          workspace_(std::max<std::size_t>(workspace_bytes_, 1)),
          q_tensor_(q_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens, batch_}),
          k_tensor_(k_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          v_tensor_(v_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          positions_tensor_(positions_.p, DType::I32, {tokens, batch_}),
          valid_columns_tensor_(valid_columns_.p, DType::I32, {batch_}),
          table_rows_tensor_(table_rows_.p, DType::I32, {batch_}),
          output_tensor_(output_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens, batch_}),
          cache_view_(make_cache_view(cache_k_, cache_v_, cache_k_scale_, cache_v_scale_,
                                      block_table_, geometry, dtype, padded_)),
          batch_cache_view_(make_batch_cache_view(cache_k_, cache_v_, cache_k_scale_,
                                                  cache_v_scale_, block_table_, geometry, dtype,
                                                  padded_, batch_)),
          envelope_{static_cast<std::uint32_t>(visible_), static_cast<std::uint32_t>(visible_)} {
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
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(logical_pages_) * batch_);
        for (std::int32_t row = 0; row < batch_; ++row) {
            for (std::int32_t page = 0; page < logical_pages_; ++page) {
                const std::int32_t linear = row * logical_pages_ + page;
                host_table[static_cast<std::size_t>(row) * logical_pages_ + page] =
                    mapping_ == PageMapping::Identity ? linear : 2 * linear + 1;
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid_columns_.p, valid_columns.data(), valid_columns_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(table_rows_.p, table_rows.data(), table_rows_.bytes,
                              cudaMemcpyHostToDevice));
    }

    void launch(Entry entry, cudaStream_t stream) {
        if (entry == Entry::Append) {
            const Tensor validity = masked_ ? valid_columns_tensor_ : Tensor{};
            ops::causal_softmax_attention(
                q_tensor_, k_tensor_, v_tensor_, positions_tensor_, validity, table_rows_tensor_,
                {kHeadDim, q_tensor_.ne[1], k_tensor_.ne[1]}, kScale, batch_cache_view_, envelope_,
                workspace_, output_tensor_, stream);
        } else {
            ops::causal_softmax_attention_cached(
                q_tensor_, positions_tensor_, {kHeadDim, q_tensor_.ne[1], cache_view_.num_kv_heads},
                kScale, cache_view_, envelope_, workspace_, output_tensor_, stream);
        }
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    std::int32_t batch_;
    bool masked_;
    std::int32_t visible_;
    std::int32_t padded_;
    PageMapping mapping_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer valid_columns_;
    DeviceBuffer table_rows_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor valid_columns_tensor_;
    Tensor table_rows_tensor_;
    Tensor output_tensor_;
    PagedKVLayerView cache_view_;
    PagedKVBatchLayerView batch_cache_view_;
    ops::CausalAttentionExecutionEnvelope envelope_;
};

const char* entry_name(Entry entry) { return entry == Entry::Append ? "append" : "cached"; }

const char* dtype_name(DType dtype) {
    if (dtype == DType::BF16) return "bf16";
    if (dtype == DType::I8) return "int8";
    return "fp8";
}

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

double cache_vector_bytes(DType dtype) {
    if (dtype == DType::BF16) { return static_cast<double>(kHeadDim * dtype_size(DType::BF16)); }
    return static_cast<double>(kHeadDim * dtype_size(dtype) +
                               scale_groups(dtype) * dtype_size(DType::FP16));
}

double causal_key_sum(std::int32_t tokens, std::int32_t context) {
    const double t = static_cast<double>(tokens);
    return t * context + t * static_cast<double>(tokens + 1) * 0.5;
}

double causal_key_sum(std::span<const std::int32_t> contexts,
                      std::span<const std::int32_t> valid_columns) {
    double total = 0.0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        total += causal_key_sum(valid_columns[row], contexts[row]);
    }
    return total;
}

double logical_bytes(Entry entry, const Geometry& geometry, DType dtype,
                     std::span<const std::int32_t> contexts,
                     std::span<const std::int32_t> valid_columns) {
    std::int64_t valid_token_count = 0;
    for (const std::int32_t valid : valid_columns) { valid_token_count += valid; }
    const double valid_tokens = static_cast<double>(valid_token_count);
    const double q_and_output = 2.0 * kHeadDim * geometry.query_heads * valid_tokens * 2.0;
    const double cache_reads  = causal_key_sum(contexts, valid_columns) * geometry.kv_heads * 2.0 *
                               cache_vector_bytes(dtype);
    if (entry == Entry::Cached) { return q_and_output + cache_reads; }
    const double input_kv     = 2.0 * kHeadDim * geometry.kv_heads * valid_tokens * 2.0;
    const double cache_writes = 2.0 * geometry.kv_heads * valid_tokens * cache_vector_bytes(dtype);
    return q_and_output + cache_reads + input_kv + cache_writes;
}

double contraction_flops(const Geometry& geometry, std::span<const std::int32_t> contexts,
                         std::span<const std::int32_t> valid_columns) {
    return 2.0 * kHeadDim * geometry.query_heads * causal_key_sum(contexts, valid_columns);
}

double physical_kv_read_bytes(const Geometry& geometry, DType dtype,
                              std::span<const std::int32_t> contexts,
                              std::span<const std::int32_t> valid_columns) {
    double visible_rows = 0.0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        if (valid_columns[row] > 0) { visible_rows += contexts[row] + valid_columns[row]; }
    }
    return visible_rows * geometry.kv_heads * 2.0 * cache_vector_bytes(dtype);
}

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, DeviceBuffer& flush, cudaStream_t stream,
                          int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.logical_bytes / seconds / 1.0e9;
    const double tflops  = (result.qk_flops + result.pv_flops) / seconds / 1.0e12;
    const double tensor_core_roofline =
        result.kv_dtype == DType::FP8_E4M3FN ? kMixedFp8F16TcTflops : kDenseF16Bf16TcTflops;
    std::printf("entry=%-6s geometry=%-14s kv=%-4s mapping=%-10s execution=%-5s cache=%-4s "
                "B=%d W=%d contexts=%s valid=%s rows=%s "
                "workspace=%9zu median=%10.3f us min=%10.3f us p95=%10.3f us "
                "logical=%8.1f GB/s (%5.1f%% of %.0f) math=%7.2f TFLOP/s (%5.1f%% of %.1f)\n",
                entry_name(result.entry), result.geometry.name, dtype_name(result.kv_dtype),
                mapping_name(result.mapping), execution_name(result.execution),
                cache_name(result.cache), result.batch, result.tokens, result.row_contexts.c_str(),
                result.valid_columns.c_str(), result.table_rows.c_str(), result.workspace_bytes,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps,
                gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs, tflops,
                tflops / tensor_core_roofline * 100.0, tensor_core_roofline);
    if (result.kv_dtype == DType::FP8_E4M3FN) {
        const double mixed_floor_us = (result.qk_flops / (kDenseFp8TcTflops * 1.0e12) +
                                       result.pv_flops / (kDenseF16Bf16TcTflops * 1.0e12)) *
                                      1.0e6;
        const double physical_gbps = result.physical_kv_read_bytes / seconds / 1.0e9;
        std::printf("  fp8 qk_flops=%.0f pv_flops=%.0f mixed_tc_floor=%8.3f us "
                    "mixed_tc_roofline=%5.1f%% physical_kv_read=%.0f bytes "
                    "physical_kv_read=%8.1f GB/s (%5.1f%% of %.1f)\n",
                    result.qk_flops, result.pv_flops, mixed_floor_us,
                    mixed_floor_us / result.timing.median_us * 100.0, result.physical_kv_read_bytes,
                    physical_gbps, physical_gbps / kColdPureReadCeilingGBs * 100.0,
                    kColdPureReadCeilingGBs);
    }
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,geometry,kv_dtype,mapping,execution,cache,B,W,row_contexts,valid_columns,"
              "table_rows,workspace_bytes,logical_bytes,qk_flops,pv_flops,"
              "mixed_tc_floor_us,mixed_tc_roofline_pct,physical_kv_read_bytes,"
              "physical_kv_read_gbps,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << entry_name(result.entry) << ',' << result.geometry.name << ','
               << dtype_name(result.kv_dtype) << ',' << mapping_name(result.mapping) << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.batch << ',' << result.tokens << ',' << result.row_contexts << ','
               << result.valid_columns << ',' << result.table_rows << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.qk_flops << ',' << result.pv_flops
               << ',';
        if (result.kv_dtype == DType::FP8_E4M3FN) {
            const double seconds        = result.timing.median_us * 1.0e-6;
            const double mixed_floor_us = (result.qk_flops / (kDenseFp8TcTflops * 1.0e12) +
                                           result.pv_flops / (kDenseF16Bf16TcTflops * 1.0e12)) *
                                          1.0e6;
            output << mixed_floor_us << ',' << mixed_floor_us / result.timing.median_us * 100.0
                   << ',' << result.physical_kv_read_bytes << ','
                   << result.physical_kv_read_bytes / seconds / 1.0e9;
        } else {
            output << ",,,";
        }
        output << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
    }
}

void profile(Case& data, Entry entry, const Geometry& geometry, DType dtype, const Options& options,
             std::int32_t batch, std::int32_t width, std::string_view contexts,
             std::string_view valid_columns, std::string_view table_rows, DeviceBuffer& flush,
             cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream,
                      [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf(
        "PROFILE entry=%s geometry=%s kv=%s mapping=%s dispatch=public execution=%s cache=%s "
        "B=%d W=%d contexts=%.*s valid=%.*s rows=%.*s\n",
        entry_name(entry), geometry.name, dtype_name(dtype), mapping_name(options.mapping),
        execution_name(execution), cache_name(cache), batch, width,
        static_cast<int>(contexts.size()), contexts.data(), static_cast<int>(valid_columns.size()),
        valid_columns.data(), static_cast<int>(table_rows.size()), table_rows.data());
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(entry, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<Geometry> selected_geometries(GeometryChoice choice) {
    if (choice == GeometryChoice::H24Kv4) { return {kH24Kv4}; }
    if (choice == GeometryChoice::H16Kv2) { return {kH16Kv2}; }
    return {kH24Kv4, kH16Kv2};
}

std::vector<DType> selected_dtypes(KvChoice choice) {
    if (choice == KvChoice::Bf16) { return {DType::BF16}; }
    if (choice == KvChoice::Int8) { return {DType::I8}; }
    if (choice == KvChoice::Fp8) { return {DType::FP8_E4M3FN}; }
    return {DType::BF16, DType::I8, DType::FP8_E4M3FN};
}

struct RowProfile {
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
};

RowProfile make_row_profile(const Options& options, std::int32_t batch, std::int32_t width,
                            std::int32_t uniform_context) {
    RowProfile profile;
    profile.contexts =
        options.row_contexts.empty()
            ? std::vector<std::int32_t>(static_cast<std::size_t>(batch), uniform_context)
            : options.row_contexts;
    profile.valid_columns = options.valid_columns.empty()
                                ? std::vector<std::int32_t>(static_cast<std::size_t>(batch), width)
                                : options.valid_columns;
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
            context > static_cast<std::int32_t>(ops::kCausalAttentionMaximumVisibleKeys) - valid) {
            throw std::invalid_argument("row profile exceeds the public causal-attention domain");
        }
    }
    return profile;
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
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        const std::vector<Geometry> geometries = selected_geometries(options.geometry);
        const std::vector<DType> dtypes        = selected_dtypes(options.kv);

        if (options.profile) {
            const Entry entry        = options.entry;
            const Geometry geometry  = geometries.front();
            const DType dtype        = dtypes.front();
            const std::int32_t batch = options.batches.front();
            const std::int32_t width = options.tokens.front();
            const std::int32_t context =
                options.row_contexts.empty() ? options.contexts.front() : 0;
            const RowProfile rows = make_row_profile(options, batch, width, context);
            Case data(geometry, dtype, width, rows.contexts, rows.valid_columns, rows.table_rows,
                      options.mapping);
            const std::string context_name = profile_name(rows.contexts);
            const std::string valid_name   = profile_name(rows.valid_columns);
            const std::string table_name   = profile_name(rows.table_rows);
            profile(data, entry, geometry, dtype, options, batch, width, context_name, valid_name,
                    table_name, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        const std::vector<std::int32_t> context_profiles =
            options.row_contexts.empty() ? options.contexts : std::vector<std::int32_t>{0};
        for (const Geometry& geometry : geometries) {
            for (const DType dtype : dtypes) {
                for (const std::int32_t batch : options.batches) {
                    for (const std::int32_t context : context_profiles) {
                        for (const std::int32_t tokens : options.tokens) {
                            const RowProfile rows =
                                make_row_profile(options, batch, tokens, context);
                            Case data(geometry, dtype, tokens, rows.contexts, rows.valid_columns,
                                      rows.table_rows, options.mapping);
                            for (const Entry entry : {Entry::Append, Entry::Cached}) {
                                if ((options.entry == Entry::Append && entry != Entry::Append) ||
                                    (options.entry == Entry::Cached && entry != Entry::Cached) ||
                                    (entry == Entry::Cached && batch != 1)) {
                                    continue;
                                }
                                bench::TimedGraph graph;
                                if (options.execution != Execution::Eager) {
                                    data.launch(entry, stream);
                                    CUDA_CHECK(cudaStreamSynchronize(stream));
                                    graph.capture(stream, [&](cudaStream_t launch_stream) {
                                        data.launch(entry, launch_stream);
                                    });
                                }
                                for (const Execution execution :
                                     {Execution::Eager, Execution::Graph}) {
                                    if ((options.execution == Execution::Eager &&
                                         execution != Execution::Eager) ||
                                        (options.execution == Execution::Graph &&
                                         execution != Execution::Graph)) {
                                        continue;
                                    }
                                    for (const CacheState cache :
                                         {CacheState::Cold, CacheState::Warm}) {
                                        if ((options.cache == CacheMode::Cold &&
                                             cache != CacheState::Cold) ||
                                            (options.cache == CacheMode::Warm &&
                                             cache != CacheState::Warm)) {
                                            continue;
                                        }
                                        Result result{
                                            entry,
                                            geometry,
                                            dtype,
                                            execution,
                                            cache,
                                            options.mapping,
                                            batch,
                                            tokens,
                                            profile_name(rows.contexts),
                                            profile_name(rows.valid_columns),
                                            profile_name(rows.table_rows),
                                            data.workspace_bytes(),
                                            logical_bytes(entry, geometry, dtype, rows.contexts,
                                                          rows.valid_columns),
                                            contraction_flops(geometry, rows.contexts,
                                                              rows.valid_columns),
                                            contraction_flops(geometry, rows.contexts,
                                                              rows.valid_columns),
                                            physical_kv_read_bytes(geometry, dtype, rows.contexts,
                                                                   rows.valid_columns),
                                            measure(data, entry, execution, cache, &graph, flush,
                                                    stream, options.warmup, options.repeat)};
                                        report(result);
                                        results.push_back(result);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_causal_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
