// Cold-cache per-op GEMV measurement rig for q5090 v2 ROW_SPLIT low-bit
// linears. This is measurement-only: it dispatches through qus::kernels::linear
// so the current generic ROW_SPLIT plan remains the measured implementation.
//
// Examples:
//   ./build/bench/qus_linear_op_bench --all-targets --csv-out profiles/ncu-linear-v2/baseline.csv
//   ./build/bench/qus_linear_op_bench --shape MlpGateUp17408x5120 --qtype Q4 --repeat 200

#include "qus/kernels/linear.h"
#include "qus/core/device.h"
#include "qus_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace qus;

namespace {

constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20; // 256 MiB > RTX 5090 L2.
constexpr std::uint64_t kDefaultCopyBytes  = 512ULL << 20; // Counted as read + write bytes.
constexpr int           kDefaultWarmup     = 3;
constexpr int           kDefaultRepeat     = 20;
constexpr int           kDefaultCopyRepeat = 8;

struct DeviceBuffer {
    void*         p     = nullptr;
    std::uint64_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::uint64_t nbytes) : bytes(nbytes) {
        if (bytes > 0) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    }
    ~DeviceBuffer() {
        if (p) { cudaFree(p); }
    }
    DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
        other.p     = nullptr;
        other.bytes = 0;
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct ShapeSpec {
    const char*  name;
    std::int32_t n;
    std::int32_t k;
};

struct TargetSpec {
    ShapeSpec shape;
    QType     qtype;
};

constexpr ShapeSpec kShapes[] = {
    {"MlpGateUp17408x5120", 17408, 5120},
    {"MlpDown5120x17408", 5120, 17408},
    {"LmHead248320x5120", 248320, 5120},
    {"Proj6144x5120", 6144, 5120},
    {"Out5120x6144", 5120, 6144},
    {"GdnQK2048x5120", 2048, 5120},
    {"AttnKV1024x5120", 1024, 5120},
};

constexpr TargetSpec kTask2Targets[] = {
    {{"MlpGateUp17408x5120", 17408, 5120}, QType::Q4G64_F16S},
    {{"MlpDown5120x17408", 5120, 17408}, QType::Q5G64_F16S},
    {{"LmHead248320x5120", 248320, 5120}, QType::Q6G64_F16S},
    {{"Proj6144x5120", 6144, 5120}, QType::Q5G64_F16S},
    {{"Proj6144x5120", 6144, 5120}, QType::Q4G64_F16S},
    {{"Out5120x6144", 5120, 6144}, QType::Q5G64_F16S},
    {{"GdnQK2048x5120", 2048, 5120}, QType::Q4G64_F16S},
    {{"AttnKV1024x5120", 1024, 5120}, QType::Q5G64_F16S},
    {{"AttnKV1024x5120", 1024, 5120}, QType::Q4G64_F16S},
};

struct Options {
    bool          all_targets        = true;
    bool          have_shape         = false;
    bool          have_qtype         = false;
    ShapeSpec     shape              = kTask2Targets[0].shape;
    QType         qtype              = kTask2Targets[0].qtype;
    int           warmup             = kDefaultWarmup;
    int           repeat             = kDefaultRepeat;
    int           copy_repeat        = kDefaultCopyRepeat;
    std::uint64_t flush_bytes        = kDefaultFlushBytes;
    std::uint64_t copy_bytes         = kDefaultCopyBytes;
    double        stream_ceiling_gbs = 0.0;
    std::string   csv_out;
};

struct TimingStats {
    int    samples   = 0;
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
    double mean_us   = 0.0;
};

struct RowSplitPayload {
    DeviceBuffer  data;
    std::uint64_t code_bytes   = 0;
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes  = 0;
    std::uint64_t payload_bytes() const { return scale_offset + scale_bytes; }
};

struct RunResult {
    const char*   shape_name          = "";
    const char*   qtype_name          = "";
    std::int32_t  n                   = 0;
    std::int32_t  k                   = 0;
    std::uint64_t weight_payload_bytes = 0;
    std::uint64_t x_bytes             = 0;
    std::uint64_t out_bytes           = 0;
    std::uint64_t bytes_streamed      = 0;
    double        cold_median_us      = 0.0;
    double        cold_min_us         = 0.0;
    double        cold_p95_us         = 0.0;
    double        achieved_gbs        = 0.0;
    double        achieved_dram_pct   = 0.0;
    double        stream_ceiling_gbs  = 0.0;
    double        roofline_us         = 0.0;
    int           repeat              = 0;
    int           warmup              = 0;
    std::uint64_t flush_bytes         = 0;
};

__global__ void fill_codes_kernel(std::uint8_t* codes, std::uint64_t nbytes) {
    const std::uint64_t start =
        blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < nbytes; i += stride) {
        codes[i] = static_cast<std::uint8_t>((i * 17ULL + 31ULL) & 0xffULL);
    }
}

__global__ void fill_scales_kernel(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t start =
        blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < groups; i += stride) {
        const std::uint64_t off = i * 2ULL;
        scales[off]             = 0x00u;
        scales[off + 1]         = 0x3cu; // binary16 1.0, little-endian.
    }
}

__global__ void stream_copy_kernel(const uint4* src, uint4* dst, std::uint64_t words) {
    const std::uint64_t start =
        blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < words; i += stride) {
        dst[i] = src[i];
    }
}

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t align) {
    if (align == 0) { throw std::invalid_argument("align must be nonzero"); }
    if (value > std::numeric_limits<std::uint64_t>::max() - (align - 1)) {
        throw std::overflow_error("align_up overflow");
    }
    return ((value + align - 1) / align) * align;
}

std::string lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) { out.push_back(static_cast<char>(std::tolower(c))); }
    return out;
}

const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return "Q4";
    case QType::Q5G64_F16S: return "Q5";
    case QType::Q6G64_F16S: return "Q6";
    default:                return "unsupported";
    }
}

std::int32_t bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return 32;
    case QType::Q5G64_F16S: return 40;
    case QType::Q6G64_F16S: return 48;
    default:                throw std::invalid_argument("unsupported qtype for ROW_SPLIT bench");
    }
}

QType parse_qtype(std::string_view raw) {
    const std::string q = lower(raw);
    if (q == "q4" || q == "q4g64_f16s") { return QType::Q4G64_F16S; }
    if (q == "q5" || q == "q5g64_f16s") { return QType::Q5G64_F16S; }
    if (q == "q6" || q == "q6g64_f16s") { return QType::Q6G64_F16S; }
    throw std::invalid_argument("unknown qtype: " + std::string(raw));
}

ShapeSpec parse_shape(std::string_view raw) {
    const std::string wanted = lower(raw);
    for (const ShapeSpec& shape : kShapes) {
        if (lower(shape.name) == wanted) { return shape; }
    }
    throw std::invalid_argument("unknown shape: " + std::string(raw));
}

std::uint64_t parse_u64(std::string_view raw, const char* label) {
    char*      end = nullptr;
    const auto str = std::string(raw);
    errno          = 0;
    const unsigned long long value = std::strtoull(str.c_str(), &end, 10);
    if (errno != 0 || end == str.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + str);
    }
    return static_cast<std::uint64_t>(value);
}

int parse_int(std::string_view raw, const char* label) {
    const std::uint64_t value = parse_u64(raw, label);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<int>(value);
}

double parse_double(std::string_view raw, const char* label) {
    char*      end = nullptr;
    const auto str = std::string(raw);
    errno          = 0;
    const double value = std::strtod(str.c_str(), &end);
    if (errno != 0 || end == str.c_str() || *end != '\0' || value <= 0.0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + str);
    }
    return value;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --all-targets [--warmup N] [--repeat N] [--copy-repeat N]\n"
                 "  %s --shape ShapeFamily --qtype Q4|Q5|Q6 [--repeat N]\n\n"
                 "Options:\n"
                 "  --all-targets              Run the Task-2 target shape/qtype rows (default).\n"
                 "  --shape NAME               One ShapeFamily string, e.g. MlpGateUp17408x5120.\n"
                 "  --qtype Q4|Q5|Q6           Low-bit ROW_SPLIT qtype for --shape.\n"
                 "  --warmup N                 Cold-cache warmup GEMV launches (default %d).\n"
                 "  --repeat N                 Cold-cache measured GEMV launches (default %d).\n"
                 "  --copy-repeat N            Cold-cache copy-ceiling samples (default %d).\n"
                 "  --flush-mib N              L2 flush buffer size in MiB (default 256).\n"
                 "  --copy-mib N               Copy-ceiling buffer size in MiB (default 512).\n"
                 "  --stream-ceiling-gbs X     Use a premeasured copy ceiling instead of measuring it.\n"
                 "  --csv-out PATH             Write the result table as CSV.\n",
                 argv0, argv0, kDefaultWarmup, kDefaultRepeat, kDefaultCopyRepeat);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto next = [&](const char* label) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++i];
        };
        if (arg == "--all-targets") {
            opt.all_targets = true;
        } else if (arg == "--shape") {
            opt.shape       = parse_shape(next("shape"));
            opt.have_shape  = true;
            opt.all_targets = false;
        } else if (arg == "--qtype") {
            opt.qtype       = parse_qtype(next("qtype"));
            opt.have_qtype  = true;
            opt.all_targets = false;
        } else if (arg == "--warmup") {
            opt.warmup = parse_int(next("warmup"), "warmup");
        } else if (arg == "--repeat") {
            opt.repeat = parse_int(next("repeat"), "repeat");
        } else if (arg == "--copy-repeat") {
            opt.copy_repeat = parse_int(next("copy-repeat"), "copy-repeat");
        } else if (arg == "--flush-mib") {
            opt.flush_bytes = parse_u64(next("flush-mib"), "flush-mib") << 20;
        } else if (arg == "--copy-mib") {
            opt.copy_bytes = parse_u64(next("copy-mib"), "copy-mib") << 20;
        } else if (arg == "--stream-ceiling-gbs") {
            opt.stream_ceiling_gbs = parse_double(next("stream-ceiling-gbs"), "stream-ceiling-gbs");
        } else if (arg == "--csv-out") {
            opt.csv_out = next("csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!opt.all_targets && (!opt.have_shape || !opt.have_qtype)) {
        throw std::invalid_argument("--shape and --qtype must be provided together");
    }
    if (opt.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (opt.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (opt.copy_repeat <= 0) { throw std::invalid_argument("--copy-repeat must be positive"); }
    if (opt.flush_bytes == 0) { throw std::invalid_argument("--flush-mib must be positive"); }
    if (opt.copy_bytes == 0 || (opt.copy_bytes % sizeof(uint4)) != 0) {
        throw std::invalid_argument("--copy-mib must produce a positive uint4-aligned byte count");
    }
    return opt;
}

int grid_for(std::uint64_t elements, int block = 256) {
    const std::uint64_t grid = (elements + block - 1) / block;
    return static_cast<int>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(grid, 65535)));
}

void flush_l2(DeviceBuffer& flush, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
}

template <class Launch>
TimingStats measure_cold(Launch&& launch, DeviceBuffer& flush, cudaStream_t stream, int warmup,
                         int repeat) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        flush_l2(flush, stream);
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        flush_l2(flush, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms) * 1000.0);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double sample : samples) { sum += sample; }

    TimingStats stats;
    stats.samples   = static_cast<int>(samples.size());
    stats.median_us = sorted[sorted.size() / 2];
    stats.min_us    = sorted.front();
    stats.p95_us    = sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(0.95 * sorted.size()))];
    stats.mean_us   = sum / static_cast<double>(samples.size());
    return stats;
}

double measure_stream_copy_ceiling_gbs(std::uint64_t copy_bytes, int repeat, DeviceBuffer& flush,
                                       cudaStream_t stream) {
    DeviceBuffer src(copy_bytes);
    DeviceBuffer dst(copy_bytes);
    CUDA_CHECK(cudaMemsetAsync(src.p, 0x5a, src.bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst.p, 0x00, dst.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::uint64_t words = copy_bytes / sizeof(uint4);
    const int           block = 256;
    const int           grid  = grid_for(words, block);
    const TimingStats stats = measure_cold(
        [&](cudaStream_t s) {
            stream_copy_kernel<<<grid, block, 0, s>>>(static_cast<const uint4*>(src.p),
                                                      static_cast<uint4*>(dst.p), words);
            CUDA_CHECK(cudaGetLastError());
        },
        flush, stream, 1, repeat);

    const double moved_bytes = static_cast<double>(copy_bytes) * 2.0;
    const double sec         = stats.median_us * 1e-6;
    return sec > 0.0 ? moved_bytes / sec / 1e9 : 0.0;
}

DeviceBuffer make_bf16_device(std::uint64_t n) {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
        h[static_cast<std::size_t>(i)] =
            qus::bench::f32_to_bf16(0.5f - static_cast<float>(i % 251ULL) / 250.0f);
    }
    DeviceBuffer d(n * 2ULL);
    CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
    return d;
}

RowSplitPayload make_row_split_payload(QType qtype, std::int32_t n, std::int32_t k,
                                       cudaStream_t stream) {
    const std::int32_t padded_k = static_cast<std::int32_t>(align_up_u64(k, 64));
    const std::int32_t kg       = padded_k / 64;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg);
    const std::uint64_t code_bytes =
        groups * static_cast<std::uint64_t>(bytes_per_group(qtype));
    const std::uint64_t scale_offset = align_up_u64(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2ULL;

    RowSplitPayload payload{DeviceBuffer(scale_offset + scale_bytes), code_bytes, scale_offset,
                            scale_bytes};
    constexpr int block = 256;
    fill_codes_kernel<<<grid_for(code_bytes, block), block, 0, stream>>>(
        static_cast<std::uint8_t*>(payload.data.p), code_bytes);
    CUDA_CHECK(cudaGetLastError());
    fill_scales_kernel<<<grid_for(groups, block), block, 0, stream>>>(
        static_cast<std::uint8_t*>(payload.data.p) + scale_offset, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return payload;
}

Weight make_weight(const RowSplitPayload& payload, QType qtype, std::int32_t n, std::int32_t k) {
    const std::int32_t padded_k = static_cast<std::int32_t>(align_up_u64(k, 64));
    Weight             w{};
    w.payload             = payload.data.p;
    w.payload_bytes       = payload.payload_bytes();
    w.qtype               = qtype;
    w.layout              = QuantLayout::RowSplit;
    w.q5090_scale_dtype   = ScaleDType::FP16;
    w.group_size          = 64;
    w.shape[0]            = n;
    w.shape[1]            = k;
    w.padded_shape[0]     = n;
    w.padded_shape[1]     = padded_k;
    w.ndim                = 2;
    w.qdata               = payload.data.p;
    w.scales              = static_cast<const std::uint8_t*>(payload.data.p) + payload.scale_offset;
    w.n                   = n;
    w.k                   = k;
    w.group               = 64;
    return w;
}

RunResult run_target(const TargetSpec& target, const Options& opt, double stream_ceiling_gbs,
                     DeviceBuffer& flush, cudaStream_t stream) {
    const ShapeSpec& shape = target.shape;
    DeviceBuffer     x     = make_bf16_device(static_cast<std::uint64_t>(shape.k));
    RowSplitPayload  wbuf  = make_row_split_payload(target.qtype, shape.n, shape.k, stream);
    DeviceBuffer     out(static_cast<std::uint64_t>(shape.n) * 2ULL);
    CUDA_CHECK(cudaMemsetAsync(out.p, 0, out.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor tx(x.p, DType::BF16, {shape.k, 1});
    Tensor tout(out.p, DType::BF16, {shape.n, 1});
    Weight w = make_weight(wbuf, target.qtype, shape.n, shape.k);
    WorkspaceArena ws(64ULL << 20);

    const TimingStats stats = measure_cold(
        [&](cudaStream_t s) { kernels::linear(tx, w, tout, ws, s); }, flush, stream, opt.warmup,
        opt.repeat);

    const std::uint64_t x_bytes        = static_cast<std::uint64_t>(shape.k) * 2ULL;
    const std::uint64_t out_bytes      = static_cast<std::uint64_t>(shape.n) * 2ULL;
    const std::uint64_t bytes_streamed = w.payload_bytes + x_bytes + out_bytes;
    const double        sec            = stats.median_us * 1e-6;
    const double        achieved_gbs =
        sec > 0.0 ? static_cast<double>(bytes_streamed) / sec / 1e9 : 0.0;
    const double achieved_dram_pct =
        stream_ceiling_gbs > 0.0 ? achieved_gbs / stream_ceiling_gbs * 100.0 : 0.0;
    const double roofline_us =
        stream_ceiling_gbs > 0.0
            ? static_cast<double>(bytes_streamed) / (stream_ceiling_gbs * 1e9) * 1e6
            : 0.0;

    return RunResult{shape.name,
                     qtype_name(target.qtype),
                     shape.n,
                     shape.k,
                     w.payload_bytes,
                     x_bytes,
                     out_bytes,
                     bytes_streamed,
                     stats.median_us,
                     stats.min_us,
                     stats.p95_us,
                     achieved_gbs,
                     achieved_dram_pct,
                     stream_ceiling_gbs,
                     roofline_us,
                     opt.repeat,
                     opt.warmup,
                     opt.flush_bytes};
}

void print_table(const std::vector<RunResult>& rows, double stream_ceiling_gbs,
                 std::uint64_t copy_bytes) {
    std::printf("# stream_ceiling_gbs=%.3f\n", stream_ceiling_gbs);
    std::printf("# stream_ceiling_method=cold L2-flushed uint4 copy kernel, bytes_per_sample=%llu "
                "(read+write counted as 2x copy bytes)\n",
                static_cast<unsigned long long>(copy_bytes));
    std::printf("%-24s %-3s %8s %8s %15s %12s %12s %9s %10s\n", "shape", "qt", "N", "K",
                "bytes_streamed", "cold_us", "ach_GB/s", "DRAM_%", "roof_us");
    for (const RunResult& r : rows) {
        std::printf("%-24s %-3s %8d %8d %15llu %12.3f %12.3f %9.3f %10.3f\n",
                    r.shape_name, r.qtype_name, r.n, r.k,
                    static_cast<unsigned long long>(r.bytes_streamed), r.cold_median_us,
                    r.achieved_gbs, r.achieved_dram_pct, r.roofline_us);
    }
}

void write_csv(const std::filesystem::path& path, const std::vector<RunResult>& rows) {
    if (path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV output: " + path.string()); }
    out << "shape,qtype,N,K,weight_payload_bytes,x_bytes,out_bytes,bytes_streamed,"
           "cold_median_us,cold_min_us,cold_p95_us,achieved_gbs,achieved_dram_pct,"
           "stream_ceiling_gbs,roofline_us,warmup,repeat,flush_bytes\n";
    for (const RunResult& r : rows) {
        out << r.shape_name << ',' << r.qtype_name << ',' << r.n << ',' << r.k << ','
            << r.weight_payload_bytes << ',' << r.x_bytes << ',' << r.out_bytes << ','
            << r.bytes_streamed << ',' << r.cold_median_us << ',' << r.cold_min_us << ','
            << r.cold_p95_us << ',' << r.achieved_gbs << ',' << r.achieved_dram_pct << ','
            << r.stream_ceiling_gbs << ',' << r.roofline_us << ',' << r.warmup << ','
            << r.repeat << ',' << r.flush_bytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(opt.flush_bytes);

        double stream_ceiling_gbs = opt.stream_ceiling_gbs;
        if (stream_ceiling_gbs <= 0.0) {
            stream_ceiling_gbs =
                measure_stream_copy_ceiling_gbs(opt.copy_bytes, opt.copy_repeat, flush, stream);
        }

        std::vector<TargetSpec> targets;
        if (opt.all_targets) {
            targets.assign(std::begin(kTask2Targets), std::end(kTask2Targets));
        } else {
            targets.push_back(TargetSpec{opt.shape, opt.qtype});
        }

        std::vector<RunResult> rows;
        rows.reserve(targets.size());
        for (const TargetSpec& target : targets) {
            rows.push_back(run_target(target, opt, stream_ceiling_gbs, flush, stream));
        }

        print_table(rows, stream_ceiling_gbs, opt.copy_bytes);
        if (!opt.csv_out.empty()) { write_csv(opt.csv_out, rows); }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qus_linear_op_bench: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }
}
