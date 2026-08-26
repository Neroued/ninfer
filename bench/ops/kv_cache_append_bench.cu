// Public-Op benchmark for full and device-count-prefix KV cache append contracts.
// Cache encoding, launch geometry, and route selection remain private to the public wrappers.

#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/kv_cache_append.h"

#include "core/device.h"
#include "core/cyclic_kv_cache.h"
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

constexpr std::int32_t kFullHeadDim   = 256;
constexpr std::int32_t kPrefixHeadDim = 128;
constexpr std::int32_t kPrefixKvHeads = 8;
constexpr std::int32_t kKvGroup       = 64;
constexpr std::int32_t kFp8KvGroup    = 256;
constexpr std::int32_t kRingCapacity  = 4096;
constexpr std::size_t kFlushBytes     = std::size_t{256} << 20;
constexpr double kRtx5090DramGBs      = 1792.0;

enum class Mode : std::uint8_t { Full, Prefix, All };
enum class FullGeometryChoice : std::uint8_t { Kv4, Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, All };
enum class LayoutChoice : std::uint8_t { Paged, Cyclic, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct FullGeometry {
    const char* name;
    std::int32_t kv_heads;
};

constexpr FullGeometry kFullKv4{"d256-kv4", 4};
constexpr FullGeometry kFullKv2{"d256-kv2", 2};

struct Options {
    Mode mode                        = Mode::All;
    FullGeometryChoice full_geometry = FullGeometryChoice::All;
    KvChoice kv                      = KvChoice::All;
    LayoutChoice layout              = LayoutChoice::All;
    Execution execution              = Execution::Graph;
    CacheMode cache                  = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 1024};
    std::vector<std::int32_t> counts{0, 1, 4, 8, 16};
    std::int32_t context = 128;
    int warmup           = 5;
    int repeat           = 50;
    bool profile         = false;
    std::string csv_out;
};

struct Result {
    Mode mode;
    const char* geometry;
    DType kv_dtype;
    const char* layout;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::int32_t committed;
    double useful_bytes;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_kv_cache_append_bench [--mode full|prefix|all] "
                 "[--full-geometry d256-kv4|d256-kv2|all] [--kv-dtype bf16|int8|fp8|all] "
                 "[--layout paged|cyclic|all] [--tokens T,...] [--counts C,...] "
                 "[--context L] [--execution eager|graph|both] [--cache cold|warm|both] "
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
        if (argument == "--mode") {
            const std::string_view value(next("--mode requires a value"));
            if (value == "full")
                options.mode = Mode::Full;
            else if (value == "prefix")
                options.mode = Mode::Prefix;
            else if (value == "all")
                options.mode = Mode::All;
            else
                usage("--mode expects full, prefix, or all");
        } else if (argument == "--full-geometry") {
            const std::string_view value(next("--full-geometry requires a value"));
            if (value == "d256-kv4")
                options.full_geometry = FullGeometryChoice::Kv4;
            else if (value == "d256-kv2")
                options.full_geometry = FullGeometryChoice::Kv2;
            else if (value == "all")
                options.full_geometry = FullGeometryChoice::All;
            else
                usage("--full-geometry expects d256-kv4, d256-kv2, or all");
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
        } else if (argument == "--layout") {
            const std::string_view value(next("--layout requires a value"));
            if (value == "paged")
                options.layout = LayoutChoice::Paged;
            else if (value == "cyclic")
                options.layout = LayoutChoice::Cyclic;
            else if (value == "all")
                options.layout = LayoutChoice::All;
            else
                usage("--layout expects paged, cyclic, or all");
        } else if (argument == "--tokens") {
            options.tokens =
                parse_list(next("--tokens requires a value"), 1, kRingCapacity, "--tokens");
        } else if (argument == "--counts") {
            options.counts =
                parse_list(next("--counts requires a value"), 0, kRingCapacity, "--counts");
        } else if (argument == "--context") {
            options.context = parse_i32(next("--context requires a value"), 0, 262144, "--context");
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
    if (options.context > std::numeric_limits<std::int32_t>::max() -
                              *std::max_element(options.tokens.begin(), options.tokens.end())) {
        usage("context + tokens exceeds int32");
    }
    if (options.profile &&
        (options.mode == Mode::All || options.tokens.size() != 1 ||
         options.execution == Execution::Both || options.cache == CacheMode::Both)) {
        usage("--profile requires one mode, one T, one execution, and one cache state");
    }
    if (options.profile && options.mode == Mode::Full &&
        (options.full_geometry == FullGeometryChoice::All || options.kv == KvChoice::All)) {
        usage("full --profile requires one full geometry and one KV dtype");
    }
    if (options.profile && options.mode == Mode::Prefix &&
        (options.layout == LayoutChoice::All || options.counts.size() != 1)) {
        usage("prefix --profile requires one layout and one count");
    }
    return options;
}

std::int32_t align_context(std::int32_t context) {
    return ((std::max(context, 1) + 127) / 128) * 128;
}

std::size_t full_cache_bytes(const FullGeometry& geometry, DType dtype, std::int32_t padded) {
    return static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * padded * dtype_size(dtype);
}

std::int32_t full_scale_groups(DType dtype) {
    if (dtype == DType::I8) return kFullHeadDim / kKvGroup;
    if (dtype == DType::FP8_E4M3FN) return kFullHeadDim / kFp8KvGroup;
    return 0;
}

std::size_t full_scale_bytes(const FullGeometry& geometry, DType dtype, std::int32_t padded) {
    return static_cast<std::size_t>(full_scale_groups(dtype)) * geometry.kv_heads * padded *
           dtype_size(DType::FP16);
}

PagedKVLayerView make_full_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                DeviceBuffer& v_scale, DeviceBuffer& block_table,
                                const FullGeometry& geometry, DType dtype, std::int32_t padded) {
    const bool quantized            = dtype != DType::BF16;
    const std::int32_t scale_groups = full_scale_groups(dtype);
    const std::int32_t pages        = padded / kPagedKVPageSize;
    return {
        .k_pages = Tensor(k.p, dtype, {kFullHeadDim, kPagedKVPageSize, geometry.kv_heads, pages}),
        .v_pages = Tensor(v.p, dtype, {kFullHeadDim, kPagedKVPageSize, geometry.kv_heads, pages}),
        .k_scale_pages = quantized
                             ? Tensor(k_scale.p, DType::FP16,
                                      {scale_groups, kPagedKVPageSize, geometry.kv_heads, pages})
                             : Tensor(),
        .v_scale_pages = quantized
                             ? Tensor(v_scale.p, DType::FP16,
                                      {scale_groups, kPagedKVPageSize, geometry.kv_heads, pages})
                             : Tensor(),
        .block_table   = Tensor(block_table.p, DType::I32, {pages}),
        .head_dim      = kFullHeadDim,
        .num_kv_heads  = geometry.kv_heads,
        .dtype         = dtype,
        .quant_group   = dtype == DType::I8           ? kKvGroup
                         : dtype == DType::FP8_E4M3FN ? kFp8KvGroup
                                                      : 0,
    };
}

class FullCase {
public:
    FullCase(FullGeometry geometry, DType dtype, std::int32_t tokens, std::int32_t context)
        : geometry_(geometry), dtype_(dtype), tokens_(tokens), capacity_(context + tokens),
          padded_(align_context(capacity_)),
          k_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens)),
          v_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          cache_k_(bench::make_zeros(full_cache_bytes(geometry, dtype, padded_))),
          cache_v_(bench::make_zeros(full_cache_bytes(geometry, dtype, padded_))),
          cache_k_scale_(bench::make_zeros(
              dtype != DType::BF16 ? full_scale_bytes(geometry, dtype, padded_) : std::size_t{1})),
          cache_v_scale_(bench::make_zeros(
              dtype != DType::BF16 ? full_scale_bytes(geometry, dtype, padded_) : std::size_t{1})),
          block_table_(static_cast<std::size_t>(padded_ / kPagedKVPageSize) * sizeof(std::int32_t)),
          k_tensor_(k_.p, DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          v_tensor_(v_.p, DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          positions_tensor_(positions_.p, DType::I32, {tokens}),
          cache_view_(make_full_view(cache_k_, cache_v_, cache_k_scale_, cache_v_scale_,
                                     block_table_, geometry, dtype, padded_)) {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            host_positions[static_cast<std::size_t>(token)] = context + token;
        }
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(padded_ / kPagedKVPageSize));
        for (std::int32_t page = 0; page < static_cast<std::int32_t>(host_table.size()); ++page) {
            host_table[static_cast<std::size_t>(page)] = page;
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
    }

    void launch(cudaStream_t stream) {
        ops::kv_cache_append(k_tensor_, v_tensor_, positions_tensor_, cache_view_, stream);
    }

private:
    FullGeometry geometry_;
    DType dtype_;
    std::int32_t tokens_;
    std::int32_t capacity_;
    std::int32_t padded_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    PagedKVLayerView cache_view_;
};

PagedKVBatchLayerView make_prefix_paged_view(DeviceBuffer& k, DeviceBuffer& v,
                                             DeviceBuffer& block_tables) {
    return {
        .k_pages = Tensor(
            k.p, DType::BF16,
            {kPrefixHeadDim, kPagedKVPageSize, kRingCapacity / kPagedKVPageSize, kPrefixKvHeads}),
        .v_pages = Tensor(
            v.p, DType::BF16,
            {kPrefixHeadDim, kPagedKVPageSize, kRingCapacity / kPagedKVPageSize, kPrefixKvHeads}),
        .block_tables = Tensor(block_tables.p, DType::I32, {kRingCapacity / kPagedKVPageSize, 1}),
        .head_dim     = kPrefixHeadDim,
        .num_kv_heads = kPrefixKvHeads,
        .dtype        = DType::BF16,
        .quant_group  = 0,
    };
}

CyclicKVCacheLayerView make_prefix_cyclic_view(DeviceBuffer& k, DeviceBuffer& v) {
    return {
        .k        = Tensor(k.p, DType::BF16, {kPrefixHeadDim, kRingCapacity, kPrefixKvHeads, 1}),
        .v        = Tensor(v.p, DType::BF16, {kPrefixHeadDim, kRingCapacity, kPrefixKvHeads, 1}),
        .capacity = kRingCapacity,
        .padded_capacity = kRingCapacity,
        .num_kv_heads    = kPrefixKvHeads,
        .head_dim        = kPrefixHeadDim,
        .lane_capacity   = 1,
    };
}

class PrefixCase {
public:
    PrefixCase(std::int32_t tokens, std::int32_t committed, bool cyclic)
        : tokens_(tokens), committed_(committed), cyclic_(cyclic),
          k_(bench::make_bf16(static_cast<std::size_t>(kPrefixHeadDim) * kPrefixKvHeads * tokens)),
          v_(bench::make_bf16(static_cast<std::size_t>(kPrefixHeadDim) * kPrefixKvHeads * tokens)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          commit_count_(sizeof(std::int32_t)), selector_(sizeof(std::int32_t)),
          cache_k_(bench::make_zeros(static_cast<std::size_t>(kPrefixHeadDim) * kRingCapacity *
                                     kPrefixKvHeads * 2)),
          cache_v_(bench::make_zeros(static_cast<std::size_t>(kPrefixHeadDim) * kRingCapacity *
                                     kPrefixKvHeads * 2)),
          block_table_(static_cast<std::size_t>(kRingCapacity / kPagedKVPageSize) *
                       sizeof(std::int32_t)),
          k_tensor_(k_.p, DType::BF16, {kPrefixHeadDim, kPrefixKvHeads, tokens, 1}),
          v_tensor_(v_.p, DType::BF16, {kPrefixHeadDim, kPrefixKvHeads, tokens, 1}),
          positions_tensor_(positions_.p, DType::I32, {tokens, 1}),
          count_tensor_(commit_count_.p, DType::I32, {1}),
          selector_tensor_(selector_.p, DType::I32, {1}),
          paged_view_(make_prefix_paged_view(cache_k_, cache_v_, block_table_)),
          cyclic_view_(make_prefix_cyclic_view(cache_k_, cache_v_)),
          envelope_{0, static_cast<std::uint32_t>(tokens)} {
        const std::int32_t start = cyclic ? 2 * kRingCapacity - 3 : 0;
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            host_positions[static_cast<std::size_t>(token)] = start + token;
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        std::vector<std::int32_t> host_table(kRingCapacity / kPagedKVPageSize);
        for (std::int32_t page = 0; page < static_cast<std::int32_t>(host_table.size()); ++page) {
            host_table[static_cast<std::size_t>(page)] = page;
        }
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(commit_count_.p, &committed_, sizeof(committed_), cudaMemcpyHostToDevice));
        const std::int32_t selector = 0;
        CUDA_CHECK(cudaMemcpy(selector_.p, &selector, sizeof(selector), cudaMemcpyHostToDevice));
    }

    void launch(cudaStream_t stream) {
        if (cyclic_) {
            ops::kv_cache_append_prefix(k_tensor_, v_tensor_, positions_tensor_, count_tensor_,
                                        selector_tensor_, envelope_, cyclic_view_, stream);
        } else {
            ops::kv_cache_append_prefix(k_tensor_, v_tensor_, positions_tensor_, count_tensor_,
                                        selector_tensor_, envelope_, paged_view_, stream);
        }
    }

private:
    std::int32_t tokens_;
    std::int32_t committed_;
    bool cyclic_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer commit_count_;
    DeviceBuffer selector_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer block_table_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor count_tensor_;
    Tensor selector_tensor_;
    PagedKVBatchLayerView paged_view_;
    CyclicKVCacheLayerView cyclic_view_;
    ops::KVCacheAppendPrefixExecutionEnvelope envelope_;
};

const char* mode_name(Mode mode) { return mode == Mode::Full ? "full" : "prefix"; }

const char* dtype_name(DType dtype) {
    if (dtype == DType::BF16) return "bf16";
    if (dtype == DType::I8) return "int8";
    return "fp8";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

double full_vector_bytes(DType dtype) {
    if (dtype == DType::BF16) {
        return static_cast<double>(kFullHeadDim * dtype_size(DType::BF16));
    }
    return static_cast<double>(kFullHeadDim * dtype_size(dtype) +
                               full_scale_groups(dtype) * dtype_size(DType::FP16));
}

double full_useful_bytes(const FullGeometry& geometry, DType dtype, std::int32_t tokens) {
    const double input  = 2.0 * kFullHeadDim * geometry.kv_heads * tokens * 2.0;
    const double output = 2.0 * geometry.kv_heads * tokens * full_vector_bytes(dtype);
    return input + output;
}

double prefix_useful_bytes(std::int32_t committed) {
    return static_cast<double>(committed) * 8192.0;
}

template <class Case>
bench::ColdTiming measure(Case& data, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, DeviceBuffer& flush, cudaStream_t stream,
                          int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(launch_stream); };
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
    const double gbps    = result.useful_bytes / seconds / 1.0e9;
    std::printf("mode=%-6s geometry=%-9s kv=%-4s layout=%-6s execution=%-5s cache=%-4s "
                "T=%4d C=%4d median=%8.3f us min=%8.3f us p95=%8.3f us "
                "useful=%8.1f GB/s (%5.1f%% of %.0f)\n",
                mode_name(result.mode), result.geometry, dtype_name(result.kv_dtype), result.layout,
                execution_name(result.execution), cache_name(result.cache), result.tokens,
                result.committed, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "mode,geometry,kv_dtype,layout,execution,cache,T,committed,useful_bytes,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << mode_name(result.mode) << ',' << result.geometry << ','
               << dtype_name(result.kv_dtype) << ',' << result.layout << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.tokens << ',' << result.committed << ',' << result.useful_bytes << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
    }
}

template <class Case>
void profile_case(Case& data, const char* label, const Options& options, DeviceBuffer& flush,
                  cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream, [&](cudaStream_t launch_stream) { data.launch(launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE %s dispatch=public execution=%s cache=%s\n", label,
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<FullGeometry> selected_geometries(FullGeometryChoice choice) {
    if (choice == FullGeometryChoice::Kv4) return {kFullKv4};
    if (choice == FullGeometryChoice::Kv2) return {kFullKv2};
    return {kFullKv4, kFullKv2};
}

std::vector<DType> selected_dtypes(KvChoice choice) {
    if (choice == KvChoice::Bf16) return {DType::BF16};
    if (choice == KvChoice::Int8) return {DType::I8};
    if (choice == KvChoice::Fp8) return {DType::FP8_E4M3FN};
    return {DType::BF16, DType::I8, DType::FP8_E4M3FN};
}

template <class Case>
void collect_case(Case& data, Mode mode, const char* geometry, DType dtype, const char* layout,
                  std::int32_t tokens, std::int32_t committed, double bytes, const Options& options,
                  DeviceBuffer& flush, cudaStream_t stream, std::vector<Result>& results) {
    bench::TimedGraph graph;
    if (options.execution != Execution::Eager) {
        data.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream, [&](cudaStream_t launch_stream) { data.launch(launch_stream); });
    }
    for (const Execution execution : {Execution::Eager, Execution::Graph}) {
        if ((options.execution == Execution::Eager && execution != Execution::Eager) ||
            (options.execution == Execution::Graph && execution != Execution::Graph)) {
            continue;
        }
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                continue;
            }
            Result result{
                mode,
                geometry,
                dtype,
                layout,
                execution,
                cache,
                tokens,
                committed,
                bytes,
                measure(data, execution, cache, &graph, flush, stream, options.warmup, options.repeat)};
            report(result);
            results.push_back(result);
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
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        const std::vector<FullGeometry> geometries = selected_geometries(options.full_geometry);
        const std::vector<DType> dtypes            = selected_dtypes(options.kv);

        if (options.profile) {
            if (options.mode == Mode::Full) {
                const FullGeometry geometry = geometries.front();
                const DType dtype           = dtypes.front();
                FullCase data(geometry, dtype, options.tokens.front(), options.context);
                const std::string label =
                    std::string("mode=full geometry=") + geometry.name + " kv=" + dtype_name(dtype);
                profile_case(data, label.c_str(), options, flush, stream);
            } else {
                const bool cyclic = options.layout == LayoutChoice::Cyclic;
                if (options.counts.front() > options.tokens.front()) {
                    usage("prefix count exceeds T");
                }
                PrefixCase data(options.tokens.front(), options.counts.front(), cyclic);
                const std::string label =
                    std::string("mode=prefix layout=") + (cyclic ? "cyclic" : "paged");
                profile_case(data, label.c_str(), options, flush, stream);
            }
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        if (options.mode != Mode::Prefix) {
            for (const FullGeometry& geometry : geometries) {
                for (const DType dtype : dtypes) {
                    for (const std::int32_t tokens : options.tokens) {
                        FullCase data(geometry, dtype, tokens, options.context);
                        collect_case(data, Mode::Full, geometry.name, dtype, "paged", tokens,
                                     tokens, full_useful_bytes(geometry, dtype, tokens), options,
                                     flush, stream, results);
                    }
                }
            }
        }
        if (options.mode != Mode::Full) {
            for (const bool cyclic : {false, true}) {
                if ((options.layout == LayoutChoice::Paged && cyclic) ||
                    (options.layout == LayoutChoice::Cyclic && !cyclic)) {
                    continue;
                }
                for (const std::int32_t tokens : options.tokens) {
                    for (const std::int32_t committed : options.counts) {
                        if (committed > tokens) { continue; }
                        PrefixCase data(tokens, committed, cyclic);
                        collect_case(data, Mode::Prefix, "d128-kv8", DType::BF16,
                                     cyclic ? "cyclic" : "paged", tokens, committed,
                                     prefix_useful_bytes(committed), options, flush, stream,
                                     results);
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_kv_cache_append_bench: %s\n", error.what());
        return 1;
    }
}
