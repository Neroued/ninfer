#pragma once

#include "core/buffer.h"
#include "ninfer_bench_common.h"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace ninfer::bench::measurement {

struct Options {
    int warmup = 10, repeat = 100;
    std::size_t flush_bytes = 256ULL << 20;
    std::string execution = "graph", cache = "both", csv_out;
};

template <class Integer = int>
inline Integer integer(std::string_view text, Integer minimum = 1,
                       Integer maximum = std::numeric_limits<Integer>::max()) {
    Integer value           = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum ||
        value > maximum)
        throw std::invalid_argument("invalid integer: " + std::string(text));
    return value;
}

inline std::string_view take(int argc, char** argv, int& index) {
    if (++index == argc) throw std::invalid_argument("missing option value");
    return argv[index];
}

inline std::vector<int> integers(std::string_view text, int minimum = 1,
                                 int maximum = std::numeric_limits<int>::max()) {
    std::vector<int> values;
    while (!text.empty()) {
        const auto comma = text.find(',');
        values.push_back(integer(text.substr(0, comma), minimum, maximum));
        if (comma == text.npos) return values;
        text.remove_prefix(comma + 1);
    }
    throw std::invalid_argument("empty integer list");
}

struct Sweep {
    int begin = 0, end = 0, step = 1;
};

inline Sweep parse_sweep(std::string_view text) {
    const auto a = text.find(':');
    if (a == text.npos) throw std::invalid_argument("sweep expects START:END[:STEP]");
    const auto b    = text.find(':', a + 1);
    const int first = integer(text.substr(0, a));
    const int last  = integer(text.substr(a + 1, b == text.npos ? text.npos : b - a - 1));
    const int step  = b == text.npos ? 1 : integer(text.substr(b + 1));
    if (first > last) throw std::invalid_argument("reversed sweep");
    return {first, last, step};
}

inline std::vector<int> sweep(std::string_view text) {
    const auto range = parse_sweep(text);
    std::vector<int> result;
    for (std::int64_t i = range.begin; i <= range.end; i += range.step) result.push_back(int(i));
    return result;
}

inline bool parse(Options& o, std::string_view flag, std::string_view value) {
    if (flag == "--warmup")
        o.warmup = integer(value, 0);
    else if (flag == "--repeat")
        o.repeat = integer(value);
    else if (flag == "--flush-mib")
        o.flush_bytes = std::size_t(integer(value)) << 20;
    else if (flag == "--execution") {
        if (value != "graph" && value != "eager" && value != "both")
            throw std::invalid_argument("execution: graph|eager|both");
        o.execution = value;
    } else if (flag == "--cache") {
        if (value != "cold" && value != "warm" && value != "both")
            throw std::invalid_argument("cache: cold|warm|both");
        o.cache = value;
    } else if (flag == "--csv-out")
        o.csv_out = value;
    else
        return false;
    return true;
}

inline void help() {
    std::cout
        << "  --execution graph|eager|both  --cache cold|warm|both\n"
           "  --warmup N --repeat N --flush-mib N --csv-out PATH\n"
           "Graph GPU timing brackets the captured Op body. Eager timing includes launch gaps.\n"
           "State restore and cold-cache eviction precede the timer. Host submission is "
           "separate.\n";
}

inline std::vector<std::string> selections(const std::string& value, const char* a, const char* b) {
    return value == "both" ? std::vector<std::string>{a, b} : std::vector<std::string>{value};
}

// This timer is deliberately separate from TimedGraph's graph-launch elapsed scope.
class BodyTimedGraph {
public:
    BodyTimedGraph() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~BodyTimedGraph() {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    BodyTimedGraph(const BodyTimedGraph&)            = delete;
    BodyTimedGraph& operator=(const BodyTimedGraph&) = delete;

    template <class Launch>
    void capture(CudaExecutionView cuda, Launch&& launch, bool allow_empty = false) {
        const auto stream = cuda.stream();
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            CUDA_CHECK(cudaEventRecordWithFlags(start_, stream, cudaEventRecordExternal));
            launch(cuda);
            CUDA_CHECK(cudaEventRecordWithFlags(stop_, stream, cudaEventRecordExternal));
        } catch (...) {
            cudaGraph_t discarded = nullptr;
            cudaStreamEndCapture(stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
            throw;
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        std::size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &n));
        nodes_ = n;
        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &n));
        for (const auto node : nodes) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeKernel) ++kernels_;
        }
        if (!kernels_ && !allow_empty) throw std::runtime_error("captured Op has no kernels");
    }

    void launch(CudaExecutionView cuda) const {
        const auto stream = cuda.stream();
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    }

    double elapsed_us() const {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms * 1000.0;
    }

    std::size_t kernels() const { return kernels_; }

    std::size_t nodes() const { return nodes_; }
private:
    cudaGraph_t graph_    = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
    std::size_t kernels_ = 0;
    std::size_t nodes_   = 0;
};

struct Result {
    ColdTiming gpu, host;
    std::size_t kernels = 0;
    std::size_t nodes   = 0;
};

template <class Launch, class Restore>
Result measure(Launch&& launch, Restore&& restore, L2FlushBuffer& flush, CudaExecutionView cuda,
               const Options& options, const std::string& execution, const std::string& cache,
               bool allow_empty = false) {
    const auto stream = cuda.stream();
    // Complete fixture uploads and poison fills before using the capture stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    restore(cuda);
    launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    BodyTimedGraph graph;
    // Capture also counts the actual Op kernels for an eager row, without timing setup.
    graph.capture(cuda, launch, allow_empty);
    const bool captured = execution == "graph";
    const auto invoke   = [&] {
        if (captured)
            graph.launch(cuda);
        else
            launch(cuda);
    };
    const auto prepare = [&] {
        restore(cuda);
        if (cache == "cold") flush_l2(flush, cuda);
    };
    for (int i = 0; i < options.warmup; ++i) {
        prepare();
        invoke();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> gpu, host;
    for (int i = 0; i < options.repeat; ++i) {
        prepare();
        if (!captured) CUDA_CHECK(cudaEventRecord(start, stream));
        const auto begin = std::chrono::steady_clock::now();
        invoke();
        const auto end = std::chrono::steady_clock::now();
        if (!captured) CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double us = 0;
        if (captured)
            us = graph.elapsed_us();
        else {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            us = ms * 1000.0;
        }
        gpu.push_back(us);
        host.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {summarize_timings(std::move(gpu)), summarize_timings(std::move(host)), graph.kernels(),
            graph.nodes()};
}

struct Row {
    std::string op, format = "none", policy = "none", valid = "dense", commits = "none";
    int n = 0, k = 0, conv_channels = 0, qk_heads = 0, value_heads = 0, layers = 1, tokens = 1,
        batch                   = 1;
    std::size_t workspace_bytes = 0, workspace_peak = 0;
};

class Csv {
public:
    explicit Csv(const std::string& path, std::string_view extra_columns = {}) {
        if (!path.empty()) {
            auto parent = std::filesystem::path(path).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            file_.open(path);
            if (!file_) throw std::runtime_error("cannot open CSV: " + path);
            out_ = &file_;
        }
        *out_ << "op,format,policy,n,k,conv_channels,qk_heads,value_heads,layers,tokens,batch,"
                 "valid,commits,"
                 "execution,cache,"
                 "gpu_median_us,gpu_min_us,gpu_p95_us,host_median_us,host_min_us,host_p95_us,"
                 "workspace_bytes,workspace_peak,kernels";
        if (!extra_columns.empty()) *out_ << ',' << extra_columns;
        *out_ << '\n';
    }

    void write(const Row& row, const Result& r, const std::string& execution,
               const std::string& cache, std::string_view extra_values = {}) {
        *out_ << row.op << ',' << row.format << ',' << row.policy << ',' << row.n << ',' << row.k
              << ',' << row.conv_channels << ',' << row.qk_heads << ',' << row.value_heads << ','
              << row.layers << ',' << row.tokens << ',' << row.batch << ',' << row.valid << ','
              << row.commits << ',' << execution << ',' << cache << ',' << r.gpu.median_us << ','
              << r.gpu.min_us << ',' << r.gpu.p95_us << ',' << r.host.median_us << ','
              << r.host.min_us << ',' << r.host.p95_us << ',' << row.workspace_bytes << ','
              << row.workspace_peak << ',' << r.kernels;
        if (!extra_values.empty()) *out_ << ',' << extra_values;
        *out_ << '\n';
        out_->flush();
    }
private:
    std::ofstream file_;
    std::ostream* out_ = &std::cout;
};

// One complete, stateless or restored Op with a logical-byte roofline model.
class LogicalBenchmark {
public:
    explicit LogicalBenchmark(const Options& options)
        : options_(options), flush_(options.flush_bytes),
          csv_(options.csv_out, "logical_bytes,logical_gb_s,bandwidth_reference_us") {}

    template <class Launch, class Restore>
    void run(const Row& row, std::size_t logical_bytes, Launch&& launch, Restore&& restore) {
        for (const auto& execution : selections(options_.execution, "graph", "eager"))
            for (const auto& cache : selections(options_.cache, "cold", "warm")) {
                const auto result =
                    measure(launch, restore, flush_, device_context().execution_view(), options_,
                            execution, cache);
                csv_.write(row, result, execution, cache,
                           std::to_string(logical_bytes) + ',' +
                               format_metric(logical_bytes / (result.gpu.median_us * 1000), 3, "") +
                               ',' +
                               format_metric(logical_bytes / (kRoofline.memory_gbs * 1000), 3, ""));
            }
    }

private:
    Options options_;
    L2FlushBuffer flush_;
    Csv csv_;
};

inline DeviceBuffer make_i32(const std::vector<int>& values) {
    DeviceBuffer result(ninfer::bench::device_context().device(), values.size() * sizeof(int));
    result.copy_from_host(values.data(), result.size());
    return result;
}

inline void hardware() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    std::cerr << p.name << " CUDA=" << CUDART_VERSION << " SMs=" << p.multiProcessorCount << '\n';
}
} // namespace ninfer::bench::measurement
