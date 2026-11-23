// Public-Op benchmark for dense context-plus-query Softmax Attention.
//
// Fixture construction, cache conditioning, graph construction, and profiling control remain
// outside the measured body. Every measured eager launch and every captured graph contains one
// call to the public context_softmax_attention contract. Production dispatch is opaque here.

#include "common/roofline.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/buffer.h"
#include "../../tools/support/paged_kv_fixture.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "context_bucket_workload.h"
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
namespace workload = ninfer::bench::workload;

namespace {

constexpr std::int32_t kHeadDim        = 128;
constexpr std::int32_t kQueryHeads     = 32;
constexpr std::int32_t kKvHeads        = 8;
constexpr std::int32_t kMaximumContext = ops::context_bucket_tokens(ops::ContextBucket::K256);
constexpr float kScale                 = 0.08838834764831844055F;
constexpr ops::AttentionHeadGeometry kGeometry{kHeadDim, kQueryHeads, kKvHeads};
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    std::vector<std::int32_t> batches{1};
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16};
    std::vector<std::int32_t> contexts{0, 128, 2048, 8192, 32768, 131072, 262144};
    workload::ContextQueryOptions workload;
    Execution execution = Execution::Graph;
    CacheMode cache     = CacheMode::Cold;
    int warmup          = 5;
    int repeat          = 50;
    bool profile        = false;
    std::string csv_out;
};

struct Result {
    std::int32_t batch;
    std::int32_t tokens;
    std::int32_t context;
    ops::ContextBucket bucket;
    std::string_view lengths, valid;
    std::size_t graph_nodes;
    Execution execution;
    CacheState cache;
    std::size_t workspace_bytes;
    double useful_bytes;
    double useful_flops;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_context_softmax_attention_bench "
                 "[--batch B,...] [--tokens 1,...,16] [--context 0,...,262144] [--envelope-max N] "
                 "[--bucket auto|2k|8k|16k|32k|64k|128k|256k] "
                 "[--length-dist uniform|skewed-1long|staggered|shared-prefix] "
                 "[--valid-dist full|ragged|with-inert] [--valid full|half|mixed|zero] "
                 "[--lengths uniform|mixed] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n"
                 "--context is committed history and excludes the query block.\n"
                 "--envelope-max sets the auto bucket input when supplied.\n",
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
        if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 16, "--tokens");
        } else if (argument == "--batch" || argument == "--batches") {
            options.batches = parse_list(next("--batch requires a value"), 1, INT32_MAX, "--batch");
        } else if (argument == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, kMaximumContext, "--context");
        } else if (argument == "--envelope-max") {
            options.workload.envelope_max = parse_i32(next("--envelope-max requires a value"), 0,
                                                      kMaximumContext, "--envelope-max");
        } else if (argument == "--bucket") {
            options.workload.bucket =
                workload::parse_bucket_override(next("--bucket requires a value"));
        } else if (argument == "--length-dist") {
            options.workload.length_dist =
                workload::parse_length_dist(next("--length-dist requires a value"));
        } else if (argument == "--valid-dist") {
            options.workload.valid_dist =
                workload::parse_valid_dist(next("--valid-dist requires a value"));
        } else if (argument == "--lengths") {
            const std::string_view lengths(next("--lengths requires a value"));
            if (lengths != "uniform" && lengths != "mixed") usage("lengths mode");
            options.workload.lengths = lengths;
        } else if (argument == "--valid") {
            const std::string_view valid(next("--valid requires a value"));
            if (valid != "full" && valid != "half" && valid != "mixed" && valid != "zero")
                usage("valid mode");
            options.workload.valid = valid;
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
    if (options.profile && (options.tokens.size() != 1 || options.contexts.size() != 1 ||
                            options.batches.size() != 1 || options.execution == Execution::Both ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one B, one T, one context, one execution, and one cache state");
    }
    return options;
}

std::int32_t page_count(std::int32_t context) {
    return (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
}

struct PageMapping {
    std::vector<std::int32_t> table;
    std::int32_t logical_pages;
    std::int32_t physical_pages;
};

PageMapping make_page_mapping(const workload::ContextQueryWorkload& shape) {
    const int logical_pages = std::max(page_count(shape.max_history), 1);
    const int shared_pages  = shape.shared_prefix_tokens / kPagedKVPageSize;
    int next_page           = shared_pages;
    std::vector<std::int32_t> table(shape.lengths.size() * logical_pages, -1);
    for (std::size_t row = 0; row < shape.lengths.size(); ++row) {
        if (shape.valid_columns[row] == 0) continue;
        const int count = page_count(shape.lengths[row]);
        if (count - shared_pages > INT32_MAX - next_page)
            throw std::overflow_error("context fixture page count exceeds int32");
        for (int page = 0; page < count; ++page)
            table[row * logical_pages + page] = page < shared_pages ? page : next_page++;
    }
    return {std::move(table), logical_pages, std::max(next_page, 1)};
}

KVPageGeometry context_geometry() {
    return {kPagedKVPageSize,
            {{DType::BF16, kHeadDim, kKvHeads}, {DType::FP16, kHeadDim, kKvHeads}}};
}

PagedKVBatchLayerView make_context_view(tool::PackedKVFixture& packed, DeviceBuffer& offsets,
                                        std::int32_t pages, std::int32_t batch) {
    const Tensor table(offsets.span(), DType::I64, {pages, batch});
    return {.k_pages      = packed.plane(0, table),
            .v_pages      = packed.plane(1, table),
            .head_dim     = kHeadDim,
            .num_kv_heads = kKvHeads,
            .storage      = KvCacheStorage::BFloat16};
}

class Case {
public:
    Case(std::int32_t tokens, std::int32_t context, std::int32_t batch, const Options& options)
        : workload_(workload::generate_context_query_workload(context, tokens, batch,
                                                              options.workload, kPagedKVPageSize)),
          bucket_(options.workload.select_bucket(workload_.max_history)),
          workspace_bytes_(ops::context_softmax_attention_workspace_capacity_bytes(
              kGeometry, bucket_, batch, tokens, tokens)),
          mapping_(make_page_mapping(workload_)),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * batch,
                              101U)),
          query_k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens * batch,
                                    103U)),
          query_v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens * batch,
                                    105U)),
          packed_(bench::device_context().device(), context_geometry(), mapping_.physical_pages),
          block_table_(ninfer::bench::device_context().device(),
                       mapping_.table.size() * sizeof(std::int64_t)),
          context_length_(ninfer::bench::device_context().device(), sizeof(std::int32_t) * batch),
          valid_(ninfer::bench::device_context().device(), sizeof(std::int32_t) * batch),
          table_row_(ninfer::bench::device_context().device(), sizeof(std::int32_t) * batch),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens *
                                    batch * 2)),
          workspace_storage_(bench::make_buffer(std::max<std::size_t>(workspace_bytes_, 1))),
          workspace_(workspace_storage_.span()),
          q_tensor_(q_.span(), DType::BF16, {kHeadDim, kQueryHeads, tokens, batch}),
          query_k_tensor_(query_k_.span(), DType::BF16, {kHeadDim, kKvHeads, tokens, batch}),
          query_v_tensor_(query_v_.span(), DType::BF16, {kHeadDim, kKvHeads, tokens, batch}),
          length_tensor_(context_length_.span(), DType::I32, {batch}),
          valid_tensor_(valid_.span(), DType::I32, {batch}),
          table_row_tensor_(table_row_.span(), DType::I32, {batch}),
          output_tensor_(output_.span(), DType::BF16, {kHeadDim, kQueryHeads, tokens, batch}),
          context_view_(make_context_view(packed_, block_table_, mapping_.logical_pages, batch)) {
        std::vector<std::int32_t> rows(batch);
        for (int b = 0; b < batch; ++b) rows[b] = workload_.valid_columns[b] ? b : -1;
        CUDA_CHECK(cudaMemcpy(context_length_.data(), workload_.lengths.data(),
                              context_length_.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid_.data(), workload_.valid_columns.data(), valid_.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(table_row_.data(), rows.data(), table_row_.size(), cudaMemcpyHostToDevice));
        const auto pages    = mapping_.physical_pages;
        const auto elements = std::size_t(kHeadDim) * kPagedKVPageSize * kKvHeads * pages;
        auto context_k      = bench::make_bf16(elements, 211U, -.25F, .25F);
        auto context_v      = bench::make_bf16(elements, 311U, -1.F, 1.F);
        packed_.upload_plane(
            0, Tensor(context_k.span(), DType::BF16, {kHeadDim, kPagedKVPageSize, pages, kKvHeads}),
            true);
        packed_.upload_plane(
            1, Tensor(context_v.span(), DType::FP16, {kHeadDim, kPagedKVPageSize, pages, kKvHeads}),
            true);
        const auto table = packed_.table_offsets(mapping_.table);
        CUDA_CHECK(cudaMemcpy(block_table_.data(), table.data(), block_table_.size(),
                              cudaMemcpyHostToDevice));
    }

    void launch(CudaExecutionView cuda) {
        ops::context_softmax_attention(q_tensor_, query_k_tensor_, query_v_tensor_, length_tensor_,
                                       valid_tensor_, table_row_tensor_, kGeometry, kScale,
                                       context_view_, bucket_, workspace_, output_tensor_, cuda);
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    [[nodiscard]] ops::ContextBucket bucket() const noexcept { return bucket_; }

    [[nodiscard]] double useful_bytes() const {
        constexpr double q_bytes  = 2. * kHeadDim * kQueryHeads;
        constexpr double kv_bytes = 4. * kHeadDim * kKvHeads;
        double bytes              = q_bytes * output_tensor_.ne[2] * output_tensor_.ne[3];
        int live_rows             = 0;
        for (std::size_t row = 0; row < workload_.lengths.size(); ++row) {
            const int live = workload_.valid_columns[row];
            if (live == 0) continue;
            ++live_rows;
            bytes += kv_bytes * workload_.lengths[row] + (q_bytes + kv_bytes) * live;
        }
        return bytes - kv_bytes * std::max(0, live_rows - 1) * workload_.shared_prefix_tokens;
    }

    [[nodiscard]] double useful_flops() const {
        double visits = 0;
        for (std::size_t row = 0; row < workload_.lengths.size(); ++row) {
            const int live = workload_.valid_columns[row];
            visits += double(live) * (workload_.lengths[row] + live);
        }
        return 4. * kHeadDim * kQueryHeads * visits;
    }

private:
    workload::ContextQueryWorkload workload_;
    ops::ContextBucket bucket_;
    std::size_t workspace_bytes_;
    PageMapping mapping_;
    DeviceBuffer q_;
    DeviceBuffer query_k_;
    DeviceBuffer query_v_;
    tool::PackedKVFixture packed_;
    DeviceBuffer block_table_;
    DeviceBuffer context_length_;
    DeviceBuffer valid_;
    DeviceBuffer table_row_;
    DeviceBuffer output_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    Tensor q_tensor_;
    Tensor query_k_tensor_;
    Tensor query_v_tensor_;
    Tensor length_tensor_;
    Tensor valid_tensor_;
    Tensor table_row_tensor_;
    Tensor output_tensor_;
    PagedKVBatchLayerView context_view_;
};

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

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
    const double gbps    = result.useful_bytes / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=context execution=%-5s cache=%-4s B=%d T=%2d L=%6d "
                "bucket=%s lengths=%s valid=%s graph_nodes=%zu workspace=%8zu "
                "median=%9.3f us min=%9.3f us p95=%9.3f us useful=%8.1f GB/s "
                "(%5.1f%% of %.0f) math=%7.2f TFLOP/s (%5s%% of %s)\n",
                execution_name(result.execution), cache_name(result.cache), result.batch,
                result.tokens, result.context, ops::context_bucket_name(result.bucket).data(),
                result.lengths.data(), result.valid.data(), result.graph_nodes,
                result.workspace_bytes, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, gbps / bench::kRoofline.memory_gbs * 100.0,
                bench::kRoofline.memory_gbs, tflops,
                bench::format_metric(tflops / bench::kRoofline.bf16_f32_tflops * 100.0, 1).c_str(),
                bench::format_metric(bench::kRoofline.bf16_f32_tflops, 1).c_str());
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,execution,cache,B,T,context,workspace_bytes,useful_bytes,useful_flops,"
              "median_us,min_us,p95_us,bucket,lengths,valid,graph_nodes\n";
    for (const Result& result : results) {
        output << "context," << execution_name(result.execution) << ',' << cache_name(result.cache)
               << ',' << result.batch << ',' << result.tokens << ',' << result.context << ','
               << result.workspace_bytes << ',' << result.useful_bytes << ',' << result.useful_flops
               << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << ops::context_bucket_name(result.bucket) << ','
               << result.lengths << ',' << result.valid << ',' << result.graph_nodes << '\n';
    }
}

void profile(Case& data, const Options& options, bench::L2FlushBuffer& flush,
             CudaExecutionView cuda) {
    const auto stream         = cuda.stream();
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
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

    std::printf("PROFILE entry=context dispatch=public bucket=%s execution=%s cache=%s\n",
                ops::context_bucket_name(data.bucket()).data(), execution_name(execution),
                cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(cuda);
    else
        data.launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
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

        if (options.profile) {
            Case data(options.tokens.front(), options.contexts.front(), options.batches.front(),
                      options);
            profile(data, options, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const auto batch : options.batches) {
            for (const std::int32_t context : options.contexts) {
                for (const std::int32_t tokens : options.tokens) {
                    Case data(tokens, context, batch, options);
                    bench::TimedGraph graph;
                    if (options.execution != Execution::Eager) {
                        data.launch(cuda);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        graph.capture(cuda, [&](CudaExecutionView cuda) { data.launch(cuda); });
                    }
                    for (const Execution execution : {Execution::Eager, Execution::Graph}) {
                        if ((options.execution == Execution::Eager &&
                             execution != Execution::Eager) ||
                            (options.execution == Execution::Graph &&
                             execution != Execution::Graph)) {
                            continue;
                        }
                        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
                            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                                continue;
                            }
                            Result result{batch,
                                          tokens,
                                          context,
                                          data.bucket(),
                                          options.workload.length_name(),
                                          options.workload.valid_name(),
                                          execution == Execution::Graph ? graph.nodes() : 0,
                                          execution,
                                          cache,
                                          data.workspace_bytes(),
                                          data.useful_bytes(),
                                          data.useful_flops(),
                                          measure(data, execution, cache, &graph, flush, cuda,
                                                  options.warmup, options.repeat)};
                            report(result);
                            results.push_back(result);
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_context_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
