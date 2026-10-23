// Public-Op benchmark for full and device-count-prefix KV cache append contracts.
// Cache encoding, launch geometry, and route selection remain private to the public wrappers.

#include "common/roofline.h"
#include "../../tools/support/paged_kv_fixture.h"
#include <optional>
#include "ninfer/ops/softmax_attention.h"
#include "core/buffer.h"
#include "ninfer/ops/kv_cache_append.h"

#include "core/device.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
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

constexpr std::int32_t kFullHeadDim         = 256;
constexpr std::int32_t kPrefixHeadDim       = 128;
constexpr std::int32_t kPrefixKvHeads       = 8;
constexpr std::int32_t kPagedPrefixCapacity = 4096;
constexpr std::size_t kFlushBytes           = std::size_t{256} << 20;

enum class Mode : std::uint8_t { Full, Prefix, All };
enum class FullGeometryChoice : std::uint8_t { Kv4, Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, Nvfp4, K8V4, All };
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
    std::int32_t cyclic_capacity = 4096;
    std::int32_t batch           = 1;
    std::int32_t context         = 128;
    int warmup                   = 5;
    int repeat                   = 50;
    int graph_calls              = 1;
    int max_count                = -1;
    bool profile                 = false;
    std::string csv_out;
};

struct Result {
    Mode mode;
    const char* geometry;
    KvCacheStorage storage;
    const char* layout;
    std::int32_t batch;
    std::int32_t cyclic_capacity;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::int32_t committed;
    std::int32_t max_count;
    std::size_t graph_nodes;
    int graph_calls;
    double logical_cache_bytes;
    double key_vector_bytes;
    double value_vector_bytes;
    double physical_cache_bytes;
    double useful_bytes;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_kv_cache_append_bench [--mode full|prefix|all] "
                 "[--full-geometry d256-kv4|d256-kv2|all] "
                 "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] "
                 "[--layout paged|cyclic|all] [--tokens T,...] [--counts C,...] "
                 "[--cyclic-capacity 2048|4096] [--batch B] "
                 "[--context L] [--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--graph-calls N] [--max-count N] [--profile] "
                 "[--csv-out PATH]\n",
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
            else if (value == "nvfp4")
                options.kv = KvChoice::Nvfp4;
            else if (value == "k8v4")
                options.kv = KvChoice::K8V4;
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, fp8, nvfp4, k8v4, or all");
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
                parse_list(next("--tokens requires a value"), 1, kPagedPrefixCapacity, "--tokens");
        } else if (argument == "--counts") {
            options.counts =
                parse_list(next("--counts requires a value"), 0, kPagedPrefixCapacity, "--counts");
        } else if (argument == "--cyclic-capacity") {
            options.cyclic_capacity = parse_i32(next("--cyclic-capacity requires a value"), 2048,
                                                4096, "--cyclic-capacity");
            if (options.cyclic_capacity != 2048 && options.cyclic_capacity != 4096) {
                usage("--cyclic-capacity expects 2048 or 4096");
            }
        } else if (argument == "--batch") {
            options.batch = parse_i32(next("--batch requires a value"), 1, 8, "--batch");
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
        } else if (argument == "--graph-calls") {
            options.graph_calls =
                parse_i32(next("--graph-calls requires a value"), 1, 64, "--graph-calls");
        } else if (argument == "--max-count") {
            options.max_count =
                parse_i32(next("--max-count requires a value"), 0, 4096, "--max-count");
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
    if (options.batch != 1 &&
        (options.mode != Mode::Prefix || options.layout != LayoutChoice::Cyclic)) {
        usage("--batch greater than one requires --mode prefix --layout cyclic");
    }
    if (options.graph_calls > 1 && (options.execution != Execution::Graph || options.profile))
        usage("graph bundles require --execution graph and no --profile");
    return options;
}

std::int32_t align_context(std::int32_t context) {
    return ((std::max(context, 1) + 127) / 128) * 128;
}

KVPageGeometry full_page_geometry(const FullGeometry& geometry, KvCacheStorage storage) {
    const auto layout = paged_kv_storage_layout(storage, kFullHeadDim);
    KVPageGeometry result;
    result.planes = {
        {layout.key.data_dtype, layout.key.data_leading_extent, geometry.kv_heads},
        {layout.value.data_dtype, layout.value.data_leading_extent, geometry.kv_heads}};
    if (layout.key.has_scale())
        result.planes.push_back(
            {layout.key.scale_dtype, layout.key.scale_leading_extent, geometry.kv_heads});
    if (layout.value.has_scale())
        result.planes.push_back(
            {layout.value.scale_dtype, layout.value.scale_leading_extent, geometry.kv_heads});
    return result;
}

class FullCase {
public:
    FullCase(FullGeometry geometry, KvCacheStorage storage, std::int32_t tokens,
             std::int32_t context)
        : k_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens,
                              101U)),
          v_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens,
                              103U)),
          positions_(bench::device_context().device(), tokens * sizeof(std::int32_t)),
          packed_(bench::device_context().device(), full_page_geometry(geometry, storage),
                  align_context(context + tokens) / kPagedKVPageSize),
          table_(bench::device_context().device(),
                 align_context(context + tokens) / kPagedKVPageSize * sizeof(std::int64_t)),
          k_tensor_(k_.span(), DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          v_tensor_(v_.span(), DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          positions_tensor_(positions_.span(), DType::I32, {tokens}) {
        std::vector<std::int32_t> host_positions(tokens);
        for (int token = 0; token < tokens; ++token) host_positions[token] = context + token;
        positions_.copy_from_host(host_positions.data(), positions_.size());
        const auto pages = align_context(context + tokens) / kPagedKVPageSize;
        std::vector<std::int32_t> mapping(pages);
        for (int page = 0; page < pages; ++page) mapping[page] = page;
        const auto offsets = packed_.table_offsets(mapping);
        table_.copy_from_host(offsets.data(), table_.size());
        const Tensor table(table_.span(), DType::I64, {pages});
        const bool scaled = storage != KvCacheStorage::BFloat16;
        cache_view_       = {packed_.plane(0, table),
                             packed_.plane(1, table),
                       scaled ? packed_.plane(2, table) : PagedPlaneView{},
                       scaled ? packed_.plane(3, table) : PagedPlaneView{},
                       kFullHeadDim,
                             geometry.kv_heads,
                             storage};
    }

    void launch(CudaExecutionView cuda) {
        ops::kv_cache_append(k_tensor_, v_tensor_, positions_tensor_, cache_view_, cuda);
    }

private:
    DeviceBuffer k_, v_, positions_;
    tool::PackedKVFixture packed_;
    DeviceBuffer table_;
    Tensor k_tensor_, v_tensor_, positions_tensor_;
    PagedKVLayerView cache_view_;
};

class PrefixCase {
public:
    PrefixCase(std::int32_t tokens, std::int32_t committed, bool cyclic,
               std::int32_t cyclic_capacity, std::int32_t batch, int maximum)
        : cyclic_(cyclic),
          k_(bench::make_bf16(
              static_cast<std::size_t>(kPrefixHeadDim) * kPrefixKvHeads * tokens * batch, 105U)),
          v_(bench::make_bf16(
              static_cast<std::size_t>(kPrefixHeadDim) * kPrefixKvHeads * tokens * batch, 107U)),
          positions_(bench::device_context().device(), tokens * batch * sizeof(std::int32_t)),
          commit_count_(bench::device_context().device(), batch * sizeof(std::int32_t)),
          selector_(bench::device_context().device(), batch * (cyclic ? 8 : 4)),
          k_tensor_(k_.span(), DType::BF16, {kPrefixHeadDim, kPrefixKvHeads, tokens, batch}),
          v_tensor_(v_.span(), DType::BF16, {kPrefixHeadDim, kPrefixKvHeads, tokens, batch}),
          positions_tensor_(positions_.span(), DType::I32, {tokens, batch}),
          count_tensor_(commit_count_.span(), DType::I32, {batch}),
          selector_tensor_(selector_.span(), cyclic ? DType::I64 : DType::I32, {batch}),
          envelope_{0, static_cast<std::uint32_t>(maximum)} {
        if (maximum > tokens || committed > maximum)
            throw std::invalid_argument("prefix count/envelope exceeds extent");
        if (!cyclic && batch != 1)
            throw std::invalid_argument("paged prefix benchmark requires B=1");
        if (cyclic && (maximum > cyclic_capacity || committed > cyclic_capacity))
            throw std::invalid_argument("cyclic prefix extent exceeds capacity");
        const std::int32_t start = cyclic ? 2 * cyclic_capacity - 3 : 0;
        std::vector<std::int32_t> host_positions(tokens * batch);
        for (int b = 0; b < batch; ++b)
            for (int token = 0; token < tokens; ++token)
                host_positions[b * tokens + token] = start + token;
        positions_.copy_from_host(host_positions.data(), positions_.size());
        const std::vector<std::int32_t> host_counts(batch, committed);
        commit_count_.copy_from_host(host_counts.data(), commit_count_.size());
        if (cyclic) {
            StateImageBuilder builder;
            const auto cache_layout =
                plan_cyclic_kv_cache(builder, 1, cyclic_capacity, kPrefixKvHeads, kPrefixHeadDim);
            const auto state_layout = builder.finish();
            state_ =
                DeviceBuffer(bench::device_context().device(), state_layout.image_bytes * batch);
            state_.fill();
            const StateImageStorage images(state_.span(), state_layout);
            const CyclicKVCacheView view(images, cache_layout);
            cyclic_view_ = view.layer_view(0, selector_tensor_);
            std::vector<std::int64_t> offsets(batch);
            for (int b = 0; b < batch; ++b) offsets[b] = b * state_layout.image_bytes;
            selector_.copy_from_host(offsets.data(), selector_.size());
        } else {
            constexpr int pages = kPagedPrefixCapacity / kPagedKVPageSize;
            packed_.emplace(bench::device_context().device(),
                            KVPageGeometry{kPagedKVPageSize,
                                           {{DType::BF16, kPrefixHeadDim, kPrefixKvHeads},
                                            {DType::FP16, kPrefixHeadDim, kPrefixKvHeads}}},
                            pages);
            table_ = DeviceBuffer(bench::device_context().device(), pages * sizeof(std::int64_t));
            std::vector<std::int32_t> mapping(pages);
            for (int page = 0; page < pages; ++page) mapping[page] = page;
            const auto offsets = packed_->table_offsets(mapping);
            table_.copy_from_host(offsets.data(), table_.size());
            const Tensor table(table_.span(), DType::I64, {pages, 1});
            paged_view_ = {
                packed_->plane(0, table), packed_->plane(1, table), {}, {}, kPrefixHeadDim,
                kPrefixKvHeads,           KvCacheStorage::BFloat16};
            selector_.fill();
        }
    }

    void launch(CudaExecutionView cuda) {
        if (cyclic_)
            ops::kv_cache_append_prefix(k_tensor_, v_tensor_, positions_tensor_, count_tensor_,
                                        envelope_, cyclic_view_, cuda);
        else
            ops::kv_cache_append_prefix(k_tensor_, v_tensor_, positions_tensor_, count_tensor_,
                                        selector_tensor_, envelope_, paged_view_, cuda);
    }

private:
    bool cyclic_;
    DeviceBuffer k_, v_, positions_, commit_count_, selector_, state_, table_;
    std::optional<tool::PackedKVFixture> packed_;
    Tensor k_tensor_, v_tensor_, positions_tensor_, count_tensor_, selector_tensor_;
    PagedKVBatchLayerView paged_view_;
    CyclicKVCacheLayerView cyclic_view_;
    ops::KVCacheAppendPrefixExecutionEnvelope envelope_;
};

const char* mode_name(Mode mode) { return mode == Mode::Full ? "full" : "prefix"; }

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

double full_vector_bytes(const PagedKVVectorLayout& layout) {
    return static_cast<double>(layout.physical_bytes());
}

double full_logical_cache_bytes(const FullGeometry& geometry, std::int32_t tokens) {
    const auto layout = paged_kv_storage_layout(KvCacheStorage::BFloat16, kFullHeadDim);
    return static_cast<double>(layout.logical_bytes_per_token_head()) * geometry.kv_heads * tokens;
}

double full_physical_cache_bytes(const FullGeometry& geometry, KvCacheStorage storage,
                                 std::int32_t tokens) {
    const auto layout = paged_kv_storage_layout(storage, kFullHeadDim);
    return geometry.kv_heads * tokens *
           (full_vector_bytes(layout.key) + full_vector_bytes(layout.value));
}

double full_useful_bytes(const FullGeometry& geometry, KvCacheStorage storage,
                         std::int32_t tokens) {
    const auto layout   = paged_kv_storage_layout(storage, kFullHeadDim);
    const double input  = 2.0 * kFullHeadDim * geometry.kv_heads * tokens * 2.0;
    const double output = geometry.kv_heads * tokens *
                          (full_vector_bytes(layout.key) + full_vector_bytes(layout.value));
    return input + output;
}

double prefix_useful_bytes(std::int32_t committed, std::int32_t batch) {
    return static_cast<double>(committed) * batch * 8192.0;
}

template <class Case>
bench::ColdTiming measure(Case& data, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, bench::L2FlushBuffer& flush,
                          CudaExecutionView cuda, int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](CudaExecutionView cuda) { data.launch(cuda); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, cuda, warmup, repeat)
                   : bench::measure_launch(launch, cuda, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, cuda, warmup, repeat)
               : bench::measure_graph(*graph, cuda, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = seconds > 0 ? result.useful_bytes / seconds / 1.0e9 : 0.0;
    std::printf("mode=%-6s geometry=%-9s kv=%-6s layout=%-6s B=%d ring=%4d "
                "execution=%-5s cache=%-4s T=%4d C=%4d max_count=%d graph_nodes=%zu graph_calls=%d "
                "median=%8.3f us min=%8.3f us p95=%8.3f us "
                "logical=%.0f physical=%.0f (K=%.0f V=%.0f) useful=%8.1f GB/s "
                "(%5.1f%% of %.0f)\n",
                mode_name(result.mode), result.geometry, storage_name(result.storage),
                result.layout, result.batch, result.cyclic_capacity,
                execution_name(result.execution), cache_name(result.cache), result.tokens,
                result.committed, result.max_count, result.graph_nodes, result.graph_calls,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us,
                result.logical_cache_bytes, result.physical_cache_bytes, result.key_vector_bytes,
                result.value_vector_bytes, gbps, gbps / bench::kRoofline.memory_gbs * 100.0,
                bench::kRoofline.memory_gbs);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "mode,geometry,kv_dtype,layout,batch,cyclic_capacity,execution,cache,T,committed,max_"
              "count,graph_nodes,graph_calls,"
              "logical_cache_bytes,key_vector_bytes,value_vector_bytes,physical_cache_bytes,"
              "useful_bytes,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << mode_name(result.mode) << ',' << result.geometry << ','
               << storage_name(result.storage) << ',' << result.layout << ',' << result.batch << ','
               << result.cyclic_capacity << ',' << execution_name(result.execution) << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.committed << ','
               << result.max_count << ',' << result.graph_nodes << ',' << result.graph_calls << ','
               << result.logical_cache_bytes << ',' << result.key_vector_bytes << ','
               << result.value_vector_bytes << ',' << result.physical_cache_bytes << ','
               << result.useful_bytes << ',' << result.timing.median_us << ','
               << result.timing.min_us << ',' << result.timing.p95_us << '\n';
    }
}

template <class Case>
void profile_case(Case& data, const char* label, const Options& options,
                  bench::L2FlushBuffer& flush, CudaExecutionView cuda) {
    const auto stream         = cuda.stream();
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    const bool empty       = options.mode == Mode::Prefix && options.max_count == 0;
    bench::TimedGraph graph;
    if (execution == Execution::Graph && !empty) {
        data.launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) { data.launch(cuda); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(cuda); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(cuda); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE %s dispatch=public execution=%s cache=%s\n", label,
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph && !empty)
        graph.launch(cuda);
    else
        data.launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<FullGeometry> selected_geometries(FullGeometryChoice choice) {
    if (choice == FullGeometryChoice::Kv4) return {kFullKv4};
    if (choice == FullGeometryChoice::Kv2) return {kFullKv2};
    return {kFullKv4, kFullKv2};
}

std::vector<KvCacheStorage> selected_storages(KvChoice choice) {
    if (choice == KvChoice::Bf16) return {KvCacheStorage::BFloat16};
    if (choice == KvChoice::Int8) return {KvCacheStorage::Int8Group64};
    if (choice == KvChoice::Fp8) return {KvCacheStorage::Fp8E4M3Row256};
    if (choice == KvChoice::Nvfp4) return {KvCacheStorage::Nvfp4Group16};
    if (choice == KvChoice::K8V4) return {KvCacheStorage::Fp8KeyNvfp4Value};
    return {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
            KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value};
}

template <class Case>
void collect_case(Case& data, Mode mode, const char* geometry, KvCacheStorage storage,
                  const char* layout, std::int32_t batch, std::int32_t cyclic_capacity,
                  std::int32_t tokens, std::int32_t committed, double logical_cache_bytes,
                  double key_vector_bytes, double value_vector_bytes, double physical_cache_bytes,
                  double useful_bytes, const Options& options, bench::L2FlushBuffer& flush,
                  CudaExecutionView cuda, std::vector<Result>& results) {
    const auto stream = cuda.stream();
    const bool empty  = mode == Mode::Prefix && options.max_count == 0;
    bench::TimedGraph graph;
    if (empty) {
        data.launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (options.execution != Execution::Eager && !empty) {
        data.launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) {
            for (int call = 0; call < options.graph_calls; ++call) data.launch(cuda);
        });
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
            Result result{mode,
                          geometry,
                          storage,
                          layout,
                          batch,
                          cyclic_capacity,
                          execution,
                          cache,
                          tokens,
                          committed,
                          mode == Mode::Prefix
                              ? (options.max_count < 0 ? tokens : options.max_count)
                              : tokens,
                          execution == Execution::Graph && !empty ? graph.nodes() : 0,
                          options.graph_calls,
                          logical_cache_bytes,
                          key_vector_bytes,
                          value_vector_bytes,
                          physical_cache_bytes,
                          useful_bytes,
                          empty ? bench::ColdTiming{}
                                : measure(data, execution, cache, &graph, flush, cuda,
                                          options.warmup, options.repeat)};
            result.timing.median_us /= options.graph_calls;
            result.timing.min_us /= options.graph_calls;
            result.timing.p95_us /= options.graph_calls;
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
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(kFlushBytes);
        const std::vector<FullGeometry> geometries = selected_geometries(options.full_geometry);
        const std::vector<KvCacheStorage> storages = selected_storages(options.kv);

        if (options.profile) {
            if (options.mode == Mode::Full) {
                const FullGeometry geometry  = geometries.front();
                const KvCacheStorage storage = storages.front();
                FullCase data(geometry, storage, options.tokens.front(), options.context);
                const std::string label = std::string("mode=full geometry=") + geometry.name +
                                          " kv=" + storage_name(storage);
                profile_case(data, label.c_str(), options, flush, cuda);
            } else {
                const bool cyclic = options.layout == LayoutChoice::Cyclic;
                if (options.counts.front() > options.tokens.front()) {
                    usage("prefix count exceeds T");
                }
                if (cyclic &&
                    ((options.max_count < 0 ? options.tokens.front() : options.max_count) >
                         options.cyclic_capacity ||
                     options.counts.front() > options.cyclic_capacity)) {
                    usage("cyclic prefix extent exceeds capacity");
                }
                PrefixCase data(options.tokens.front(), options.counts.front(), cyclic,
                                options.cyclic_capacity, options.batch,
                                options.max_count < 0 ? options.tokens.front() : options.max_count);
                const std::string label =
                    std::string("mode=prefix layout=") +
                    (cyclic ? "cyclic capacity=" + std::to_string(options.cyclic_capacity) +
                                  " B=" + std::to_string(options.batch)
                            : "paged");
                profile_case(data, label.c_str(), options, flush, cuda);
            }
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        if (options.mode != Mode::Prefix) {
            for (const FullGeometry& geometry : geometries) {
                for (const KvCacheStorage storage : storages) {
                    for (const std::int32_t tokens : options.tokens) {
                        const auto storage_layout = paged_kv_storage_layout(storage, kFullHeadDim);
                        const double key_vector_bytes   = full_vector_bytes(storage_layout.key);
                        const double value_vector_bytes = full_vector_bytes(storage_layout.value);
                        const double physical_cache_bytes =
                            full_physical_cache_bytes(geometry, storage, tokens);
                        FullCase data(geometry, storage, tokens, options.context);
                        collect_case(data, Mode::Full, geometry.name, storage, "paged", 1, 0,
                                     tokens, tokens, full_logical_cache_bytes(geometry, tokens),
                                     key_vector_bytes, value_vector_bytes, physical_cache_bytes,
                                     full_useful_bytes(geometry, storage, tokens), options, flush,
                                     cuda, results);
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
                        if (committed > tokens ||
                            (options.max_count >= 0 && committed > options.max_count)) {
                            continue;
                        }
                        if (cyclic && ((options.max_count < 0 ? tokens : options.max_count) >
                                           options.cyclic_capacity ||
                                       committed > options.cyclic_capacity)) {
                            continue;
                        }
                        PrefixCase data(tokens, committed, cyclic, options.cyclic_capacity,
                                        options.batch,
                                        options.max_count < 0 ? tokens : options.max_count);
                        constexpr double kPrefixVectorBytes = kPrefixHeadDim * 2.0;
                        const double cache_bytes = static_cast<double>(committed) * options.batch *
                                                   kPrefixKvHeads * (2.0 * kPrefixVectorBytes);
                        collect_case(data, Mode::Prefix, "d128-kv8", KvCacheStorage::BFloat16,
                                     cyclic ? "cyclic" : "paged", options.batch,
                                     cyclic ? options.cyclic_capacity : 0, tokens, committed,
                                     cache_bytes, kPrefixVectorBytes, kPrefixVectorBytes,
                                     cache_bytes, prefix_useful_bytes(committed, options.batch),
                                     options, flush, cuda, results);
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
