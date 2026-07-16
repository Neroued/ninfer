// Cold-cache per-op GEMV measurement rig for native row-split low-bit
// linears. This is measurement-only: it dispatches through ninfer::ops::linear
// so the current registered ROW_SPLIT plan remains the measured implementation.
//
// Examples:
//   ./build/bench/ninfer_linear_op_bench --all-targets --csv-out profiles/ncu/linear-baseline.csv
//   ./build/bench/ninfer_linear_op_bench --shape MlpGateUp34816x5120 --qtype Q4 --repeat 200

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"
#include "ops/linear/q4/q4_rowsplit_plan.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"
#include "ops/linear/q6/q6_rowsplit_launch.h"
#include "ops/linear/q6/q6_rowsplit_plan.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"
#include "ops/linear/w8/w8_rowsplit_plan.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

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

using namespace ninfer;

namespace {

constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20; // 256 MiB > RTX 5090 L2.
constexpr std::uint64_t kDefaultCopyBytes  = 512ULL << 20; // Counted as read + write bytes.
constexpr int kDefaultWarmup               = 3;
constexpr int kDefaultRepeat               = 20;
constexpr int kDefaultCopyRepeat           = 8;

struct DeviceBuffer {
    void* p             = nullptr;
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
    const char* name;
    std::int32_t n;
    std::int32_t k;
};

struct TargetSpec {
    ShapeSpec shape;
    QType qtype;
};

enum class CandidateKind {
    Auto,
    Q4Fixed,
    Q5Fixed,
    Q6Fixed,
    W8Fixed,
};

constexpr ShapeSpec kShapes[] = {
    {"MtpFc5120x10240", 5120, 10240},
    {"MtpKV1024x5120", 1024, 5120},
    {"MtpAttnIn14336x5120", 14336, 5120},
    {"MtpOProj5120x6144", 5120, 6144},
    {"MtpMlpGateUp34816x5120", 34816, 5120},
    {"MtpMlpDown5120x17408", 5120, 17408},
    {"MlpGateUp34816x5120", 34816, 5120},
    {"AttnInQKV7168x5120", 7168, 5120},
    {"GdnInQK4096x5120", 4096, 5120},
    {"MlpDown5120x17408", 5120, 17408},
    {"LmHead248320x5120", 248320, 5120},
    {"LmHeadDraft65536x5120", 65536, 5120},
    {"LmHeadDraft98304x5120", 98304, 5120},
    {"LmHeadDraft131072x5120", 131072, 5120},
    {"Proj6144x5120", 6144, 5120},
    {"Out5120x6144", 5120, 6144},
    {"VisionPatch1152x1536", 1152, 1536},
    {"VisionQKV3456x1152", 3456, 1152},
    {"VisionFC14304x1152", 4304, 1152},
    {"AttnV1024x5120", 1024, 5120},
    {"VisionOut1152x1152", 1152, 1152},
    {"VisionFC2_1152x4304", 1152, 4304},
    {"MtpQGate6144x5120", 6144, 5120},
    {"VisionMergerFC1_4608x4608", 4608, 4608},
    {"VisionMergerFC2_5120x4608", 5120, 4608},
};

constexpr TargetSpec kTask2Targets[] = {
    {{"AttnK1024x5120", 1024, 5120}, QType::Q4G64_F16S},
    {{"GdnInQK4096x5120", 4096, 5120}, QType::Q4G64_F16S},
    {{"AttnQ6144x5120", 6144, 5120}, QType::Q4G64_F16S},
    {{"MlpGateUp34816x5120", 34816, 5120}, QType::Q4G64_F16S},
    {{"LmHeadDraft131072x5120", 131072, 5120}, QType::Q4G64_F16S},
    {{"VisionQKV3456x1152", 3456, 1152}, QType::Q4G64_F16S},
    {{"VisionFC14304x1152", 4304, 1152}, QType::Q4G64_F16S},
    {{"AttnV1024x5120", 1024, 5120}, QType::Q5G64_F16S},
    {{"MlpDown5120x17408", 5120, 17408}, QType::Q5G64_F16S},
    {{"LmHead248320x5120", 248320, 5120}, QType::Q6G64_F16S},
    {{"VisionPatch1152x1536", 1152, 1536}, QType::Q6G64_F16S},
    {{"Proj6144x5120", 6144, 5120}, QType::Q5G64_F16S},
    {{"Out5120x6144", 5120, 6144}, QType::Q5G64_F16S},
    {{"VisionOut1152x1152", 1152, 1152}, QType::Q5G64_F16S},
    {{"VisionFC2_1152x4304", 1152, 4304}, QType::Q5G64_F16S},
    {{"MtpFc5120x10240", 5120, 10240}, QType::W8G32_F16S},
    {{"MtpKV1024x5120", 1024, 5120}, QType::W8G32_F16S},
    {{"MtpAttnIn14336x5120", 14336, 5120}, QType::W8G32_F16S},
    {{"MtpOProj5120x6144", 5120, 6144}, QType::W8G32_F16S},
    {{"MtpMlpGateUp34816x5120", 34816, 5120}, QType::W8G32_F16S},
    {{"MtpMlpDown5120x17408", 5120, 17408}, QType::W8G32_F16S},
    {{"MtpQGate6144x5120", 6144, 5120}, QType::W8G32_F16S},
    {{"VisionMergerFC1_4608x4608", 4608, 4608}, QType::W8G32_F16S},
    {{"VisionMergerFC2_5120x4608", 5120, 4608}, QType::W8G32_F16S},
};

struct Options {
    bool all_targets                        = true;
    bool have_shape                         = false;
    bool have_qtype                         = false;
    bool paired_kv                          = false;
    bool have_rows                          = false;
    bool have_k                             = false;
    ShapeSpec shape                         = kTask2Targets[0].shape;
    QType qtype                             = kTask2Targets[0].qtype;
    std::int32_t rows                       = 0;
    std::int32_t k                          = 0;
    CandidateKind candidate                 = CandidateKind::Auto;
    ops::detail::Q4ScheduleId q4_schedule   = ops::detail::Q4ScheduleId::SimtR8C4;
    ops::detail::Q4KernelVariant q4_variant = ops::detail::Q4KernelVariant::Predicated;
    bool q4_variant_auto                    = true;
    ops::detail::Q5ScheduleId q5_schedule   = ops::detail::Q5ScheduleId::SimtR8C4;
    ops::detail::Q5KernelVariant q5_variant = ops::detail::Q5KernelVariant::Predicated;
    bool q5_variant_auto                    = true;
    ops::detail::Q6ScheduleId q6_schedule   = ops::detail::Q6ScheduleId::SimtR8C4;
    ops::detail::Q6KernelVariant q6_variant = ops::detail::Q6KernelVariant::Predicated;
    bool q6_variant_auto                    = true;
    ops::detail::W8ScheduleId w8_schedule   = ops::detail::W8ScheduleId::SimtR8C4;
    ops::detail::W8KernelVariant w8_variant = ops::detail::W8KernelVariant::Predicated;
    bool w8_variant_auto                    = true;
    int warmup                              = kDefaultWarmup;
    int repeat                              = kDefaultRepeat;
    int copy_repeat                         = kDefaultCopyRepeat;
    std::uint64_t flush_bytes               = kDefaultFlushBytes;
    std::uint64_t copy_bytes                = kDefaultCopyBytes;
    double stream_ceiling_gbs               = 0.0;
    std::vector<int> t_sweep; // activation columns to sweep; empty => default set
    std::string csv_out;
};

struct TimingStats {
    int samples      = 0;
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
    double mean_us   = 0.0;
};

struct RowSplitPayload {
    DeviceBuffer data;
    std::uint64_t nibble_bytes = 0;
    std::uint64_t high_offset  = 0;
    std::uint64_t high_bytes   = 0;
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes  = 0;

    std::uint64_t payload_bytes() const { return scale_offset + scale_bytes; }
};

struct RunResult {
    const char* shape_name = "";
    const char* qtype_name = "";
    std::string candidate_name;
    const char* kernel_variant = "";
    const char* build_type     = "";
    std::string gpu_name;
    int cuda_runtime_version                      = 0;
    int cuda_driver_version                       = 0;
    int compute_major                             = 0;
    int compute_minor                             = 0;
    std::int32_t n                                = 0;
    std::int32_t k                                = 0;
    std::int32_t t                                = 1;
    std::uint64_t weight_payload_bytes            = 0;
    std::uint64_t x_bytes                         = 0;
    std::uint64_t out_bytes                       = 0;
    std::uint64_t useful_bytes                    = 0;
    std::uint64_t weight_replay_lower_bound_bytes = 0;
    double cold_median_us                         = 0.0;
    double cold_min_us                            = 0.0;
    double cold_p95_us                            = 0.0;
    double warm_median_us                         = 0.0;
    double useful_gbs                             = 0.0;
    double weight_replay_lower_bound_gbs          = 0.0;
    double useful_copy_ceiling_pct                = 0.0;
    double useful_tflops                          = 0.0;
    double executed_tflops                        = std::numeric_limits<double>::quiet_NaN();
    double tc_ceiling_tflops                      = 0.0;
    double executed_tc_pct                        = std::numeric_limits<double>::quiet_NaN();
    double stream_ceiling_gbs                     = 0.0;
    double useful_copy_floor_us                   = 0.0;
    int repeat                                    = 0;
    int warmup                                    = 0;
    std::uint64_t flush_bytes                     = 0;
};

__global__ void fill_codes_kernel(std::uint8_t* codes, std::uint64_t nbytes) {
    const std::uint64_t start  = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < nbytes; i += stride) {
        codes[i] = static_cast<std::uint8_t>((i * 17ULL + 31ULL) & 0xffULL);
    }
}

__global__ void fill_scales_kernel(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t start  = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < groups; i += stride) {
        const std::uint64_t off = i * 2ULL;
        scales[off]             = 0x00u;
        scales[off + 1]         = 0x3cu; // binary16 1.0, little-endian.
    }
}

__global__ void stream_copy_kernel(const uint4* src, uint4* dst, std::uint64_t words) {
    const std::uint64_t start  = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = start; i < words; i += stride) { dst[i] = src[i]; }
}

// Dense bf16 mma.sync (m16n8k16) throughput probe: pure tensor-core FLOPs, no
// memory traffic. Used as the *measured* compute roofline for the T-sweep, the
// TC analogue of the stream-copy bandwidth ceiling (we do not quote a datasheet
// number). NACC independent accumulator chains supply the ILP to approach peak.
template <int NACC>
__global__ void tc_peak_kernel(float* sink, int iters) {
    const unsigned a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    const unsigned b0 = 0x3f803f80u, b1 = 0x3f803f80u; // bf16 1.0 packed pairs
    float c[NACC][4];
#pragma unroll
    for (int n = 0; n < NACC; ++n) {
        c[n][0] = 0.0f;
        c[n][1] = 0.0f;
        c[n][2] = 0.0f;
        c[n][3] = 0.0f;
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int n = 0; n < NACC; ++n) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int n = 0; n < NACC; ++n) { s += c[n][0] + c[n][1] + c[n][2] + c[n][3]; }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s; // force the chain to be live
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
    case QType::Q4G64_F16S:
        return "Q4";
    case QType::Q5G64_F16S:
        return "Q5";
    case QType::Q6G64_F16S:
        return "Q6";
    case QType::W8G32_F16S:
        return "W8G32";
    default:
        return "unsupported";
    }
}

const char* build_type_name() {
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

void fill_environment(RunResult& result) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&result.cuda_runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&result.cuda_driver_version));
    result.build_type    = build_type_name();
    result.gpu_name      = props.name;
    result.compute_major = props.major;
    result.compute_minor = props.minor;
}

std::int32_t group_size(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 64;
    case QType::W8G32_F16S:
        return 32;
    default:
        throw std::invalid_argument("unsupported qtype for ROW_SPLIT bench");
    }
}

std::int32_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 32;
    case QType::Q5G64_F16S:
        return 32;
    case QType::Q6G64_F16S:
        return 32;
    case QType::W8G32_F16S:
        return 32;
    default:
        throw std::invalid_argument("unsupported qtype for ROW_SPLIT bench");
    }
}

std::int32_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 0;
    case QType::W8G32_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        throw std::invalid_argument("unsupported qtype for ROW_SPLIT bench");
    }
}

QType parse_qtype(std::string_view raw) {
    const std::string q = lower(raw);
    if (q == "q4" || q == "q4g64_f16s") { return QType::Q4G64_F16S; }
    if (q == "q5" || q == "q5g64_f16s") { return QType::Q5G64_F16S; }
    if (q == "q6" || q == "q6g64_f16s") { return QType::Q6G64_F16S; }
    if (q == "w8g32" || q == "w8g32_f16s") { return QType::W8G32_F16S; }
    throw std::invalid_argument("unknown qtype: " + std::string(raw));
}

ops::detail::Q4Plan resolve_auto_q4_plan(const ShapeSpec& shape, std::int32_t t) {
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return ops::detail::q4_rowsplit_resolve_plan({shape.n, shape.k, padded_k, t});
}

ops::detail::Q5Plan resolve_auto_q5_plan(const ShapeSpec& shape, std::int32_t t) {
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return ops::detail::q5_rowsplit_resolve_plan({shape.n, shape.k, padded_k, t});
}

ops::detail::Q6Plan resolve_auto_q6_plan(const ShapeSpec& shape, std::int32_t t) {
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return ops::detail::q6_rowsplit_resolve_plan({shape.n, shape.k, padded_k, t});
}

ops::detail::W8Plan resolve_auto_w8_plan(const ShapeSpec& shape, std::int32_t t) {
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return ops::detail::w8_rowsplit_resolve_plan({shape.n, shape.k, padded_k, t});
}

std::string candidate_name(const Options& opt, QType qtype, const ShapeSpec& shape,
                           std::int32_t t) {
    switch (opt.candidate) {
    case CandidateKind::Auto:
        if (qtype == QType::Q4G64_F16S) {
            return ops::detail::q4_schedule_name(resolve_auto_q4_plan(shape, t).schedule);
        }
        if (qtype == QType::Q5G64_F16S) {
            return ops::detail::q5_schedule_name(resolve_auto_q5_plan(shape, t).schedule);
        }
        if (qtype == QType::Q6G64_F16S) {
            return ops::detail::q6_schedule_name(resolve_auto_q6_plan(shape, t).schedule);
        }
        if (qtype == QType::W8G32_F16S) {
            return ops::detail::w8_schedule_name(resolve_auto_w8_plan(shape, t).schedule);
        }
        return "unsupported";
    case CandidateKind::Q4Fixed:
        return ops::detail::q4_schedule_name(opt.q4_schedule);
    case CandidateKind::Q5Fixed:
        return ops::detail::q5_schedule_name(opt.q5_schedule);
    case CandidateKind::Q6Fixed:
        return ops::detail::q6_schedule_name(opt.q6_schedule);
    case CandidateKind::W8Fixed:
        return ops::detail::w8_schedule_name(opt.w8_schedule);
    }
    return "unknown";
}

void parse_candidate(Options& opt, std::string_view raw) {
    const std::string candidate = lower(raw);
    if (candidate == "auto") {
        opt.candidate = CandidateKind::Auto;
        return;
    }
    opt.candidate = CandidateKind::Q4Fixed;
    if (candidate == "q4-gemv-r4w1-direct") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::GemvR4W1Direct;
    } else if (candidate == "q4-gemv-r1w8-direct") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::GemvR1W8Direct;
    } else if (candidate == "q4-simt-r8c4") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::SimtR8C4;
    } else if (candidate == "q4-simt-r8c8") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::SimtR8C8;
    } else if (candidate == "q4-mma-r64c64") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::MmaR64C64;
    } else if (candidate == "q4-mma-r64c128") {
        opt.q4_schedule = ops::detail::Q4ScheduleId::MmaR64C128;
    } else if (candidate == "q5-gemv-r16s2x") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::GemvR16S2X;
    } else if (candidate == "q5-simt-r8c4") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::SimtR8C4;
    } else if (candidate == "q5-simt-r8c8") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::SimtR8C8;
    } else if (candidate == "q5-simt-split2-exact") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::SimtSplit2Exact;
    } else if (candidate == "q5-simt-split4-exact") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::SimtSplit4Exact;
    } else if (candidate == "q5-mma-r64c64") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::MmaR64C64;
    } else if (candidate == "q5-mma-r64c128") {
        opt.candidate   = CandidateKind::Q5Fixed;
        opt.q5_schedule = ops::detail::Q5ScheduleId::MmaR64C128;
    } else if (candidate == "q6-simt-r8c4") {
        opt.candidate   = CandidateKind::Q6Fixed;
        opt.q6_schedule = ops::detail::Q6ScheduleId::SimtR8C4;
    } else if (candidate == "q6-mma-r64c64") {
        opt.candidate   = CandidateKind::Q6Fixed;
        opt.q6_schedule = ops::detail::Q6ScheduleId::MmaR64C64;
    } else if (candidate == "q6-mma-r64c128") {
        opt.candidate   = CandidateKind::Q6Fixed;
        opt.q6_schedule = ops::detail::Q6ScheduleId::MmaR64C128;
    } else if (candidate == "w8-simt-r8c4") {
        opt.candidate   = CandidateKind::W8Fixed;
        opt.w8_schedule = ops::detail::W8ScheduleId::SimtR8C4;
    } else if (candidate == "w8-simt-r8c8") {
        opt.candidate   = CandidateKind::W8Fixed;
        opt.w8_schedule = ops::detail::W8ScheduleId::SimtR8C8;
    } else if (candidate == "w8-mma-r32c128") {
        opt.candidate   = CandidateKind::W8Fixed;
        opt.w8_schedule = ops::detail::W8ScheduleId::MmaR32C128;
    } else if (candidate == "w8-mma-r64c128") {
        opt.candidate   = CandidateKind::W8Fixed;
        opt.w8_schedule = ops::detail::W8ScheduleId::MmaR64C128;
    } else {
        throw std::invalid_argument("unknown candidate: " + std::string(raw));
    }
}

ops::detail::Q4KernelVariant parse_q4_variant(std::string_view raw, bool& is_auto) {
    const std::string variant = lower(raw);
    if (variant == "auto") {
        is_auto = true;
        return ops::detail::Q4KernelVariant::Predicated;
    }
    is_auto = false;
    if (variant == "none") { return ops::detail::Q4KernelVariant::None; }
    if (variant == "full") { return ops::detail::Q4KernelVariant::Full; }
    if (variant == "predicated") { return ops::detail::Q4KernelVariant::Predicated; }
    throw std::invalid_argument("unknown Q4 kernel variant: " + std::string(raw));
}

ops::detail::Q5KernelVariant parse_q5_variant(std::string_view raw, bool& is_auto) {
    const std::string variant = lower(raw);
    if (variant == "auto") {
        is_auto = true;
        return ops::detail::Q5KernelVariant::Predicated;
    }
    is_auto = false;
    if (variant == "none") { return ops::detail::Q5KernelVariant::None; }
    if (variant == "full") { return ops::detail::Q5KernelVariant::Full; }
    if (variant == "predicated") { return ops::detail::Q5KernelVariant::Predicated; }
    throw std::invalid_argument("unknown Q5 kernel variant: " + std::string(raw));
}

ops::detail::Q6KernelVariant parse_q6_variant(std::string_view raw, bool& is_auto) {
    const std::string variant = lower(raw);
    if (variant == "auto") {
        is_auto = true;
        return ops::detail::Q6KernelVariant::Predicated;
    }
    is_auto = false;
    if (variant == "none") { return ops::detail::Q6KernelVariant::None; }
    if (variant == "full") { return ops::detail::Q6KernelVariant::Full; }
    if (variant == "predicated") { return ops::detail::Q6KernelVariant::Predicated; }
    throw std::invalid_argument("unknown Q6 kernel variant: " + std::string(raw));
}

ops::detail::W8KernelVariant parse_w8_variant(std::string_view raw, bool& is_auto) {
    const std::string variant = lower(raw);
    if (variant == "auto") {
        is_auto = true;
        return ops::detail::W8KernelVariant::Predicated;
    }
    is_auto = false;
    if (variant == "none") { return ops::detail::W8KernelVariant::None; }
    if (variant == "full") { return ops::detail::W8KernelVariant::Full; }
    if (variant == "predicated") { return ops::detail::W8KernelVariant::Predicated; }
    throw std::invalid_argument("unknown W8 kernel variant: " + std::string(raw));
}

ops::detail::Q4KernelVariant resolve_q4_variant(ops::detail::Q4ScheduleId schedule,
                                                std::int32_t rows, std::int32_t k,
                                                std::int32_t padded_k, std::int32_t cols) {
    const ops::detail::Q4Problem problem{rows, k, padded_k, cols};
    using V = ops::detail::Q4KernelVariant;
    for (const V variant : {V::None, V::Full, V::Predicated}) {
        if (ops::detail::q4_candidate_is_legal(schedule, variant, problem)) { return variant; }
    }
    throw std::invalid_argument(std::string("no legal variant for ") +
                                ops::detail::q4_schedule_name(schedule));
}

ops::detail::Q5KernelVariant resolve_q5_variant(ops::detail::Q5ScheduleId schedule,
                                                std::int32_t rows, std::int32_t k,
                                                std::int32_t padded_k, std::int32_t cols) {
    const ops::detail::Q5Problem problem{rows, k, padded_k, cols};
    using V = ops::detail::Q5KernelVariant;
    for (const V variant : {V::None, V::Full, V::Predicated}) {
        if (ops::detail::q5_candidate_is_legal(schedule, variant, problem)) { return variant; }
    }
    throw std::invalid_argument(std::string("no legal variant for ") +
                                ops::detail::q5_schedule_name(schedule));
}

ops::detail::Q6KernelVariant resolve_q6_variant(ops::detail::Q6ScheduleId schedule,
                                                std::int32_t rows, std::int32_t k,
                                                std::int32_t padded_k, std::int32_t cols) {
    const ops::detail::Q6Problem problem{rows, k, padded_k, cols};
    using V = ops::detail::Q6KernelVariant;
    for (const V variant : {V::None, V::Full, V::Predicated}) {
        if (ops::detail::q6_candidate_is_legal(schedule, variant, problem)) { return variant; }
    }
    throw std::invalid_argument(std::string("no legal variant for ") +
                                ops::detail::q6_schedule_name(schedule));
}

ops::detail::W8KernelVariant resolve_w8_variant(ops::detail::W8ScheduleId schedule,
                                                std::int32_t rows, std::int32_t k,
                                                std::int32_t padded_k, std::int32_t cols) {
    const ops::detail::W8Problem problem{rows, k, padded_k, cols};
    using V = ops::detail::W8KernelVariant;
    for (const V variant : {V::Full, V::Predicated}) {
        if (ops::detail::w8_candidate_is_legal(schedule, variant, problem)) { return variant; }
    }
    throw std::invalid_argument(std::string("no legal variant for ") +
                                ops::detail::w8_schedule_name(schedule));
}

ShapeSpec parse_shape(std::string_view raw) {
    const std::string wanted = lower(raw);
    for (const ShapeSpec& shape : kShapes) {
        if (lower(shape.name) == wanted) { return shape; }
    }
    throw std::invalid_argument("unknown shape: " + std::string(raw));
}

std::uint64_t parse_u64(std::string_view raw, const char* label) {
    char* end                      = nullptr;
    const auto str                 = std::string(raw);
    errno                          = 0;
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

std::vector<int> parse_int_list(std::string_view raw) {
    std::vector<int> out;
    const std::string s(raw);
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string tok =
            s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            const int v = parse_int(tok, "t-sweep");
            if (v <= 0) { throw std::invalid_argument("--t-sweep values must be positive"); }
            out.push_back(v);
        }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    if (out.empty()) { throw std::invalid_argument("--t-sweep must list at least one value"); }
    return out;
}

double parse_double(std::string_view raw, const char* label) {
    char* end          = nullptr;
    const auto str     = std::string(raw);
    errno              = 0;
    const double value = std::strtod(str.c_str(), &end);
    if (errno != 0 || end == str.c_str() || *end != '\0' || value <= 0.0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + str);
    }
    return value;
}

void usage(const char* argv0) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s --all-targets [--warmup N] [--repeat N] [--copy-repeat N]\n"
        "  %s --shape ShapeFamily --qtype Q4|Q5|Q6|W8G32 [--repeat N]\n"
        "  %s --rows N --k K --qtype Q4|Q5|Q6|W8G32 --candidate NAME "
        "[--variant auto|none|full|predicated]\n\n"
        "Options:\n"
        "  --all-targets              Run the registered target shape/qtype rows (default).\n"
        "  --shape NAME               One ShapeFamily string, e.g. MlpGateUp34816x5120.\n"
        "  --rows N --k K             Numeric matrix geometry for fixed-candidate work.\n"
        "  --qtype Q4|Q5|Q6|W8G32     Low-bit ROW_SPLIT qtype for --shape.\n"
        "  --candidate NAME           auto, q4-gemv-r4w1-direct,\n"
        "                             q4-gemv-r1w8-direct, q4-simt-r8c4, q4-simt-r8c8,\n"
        "                             q4-mma-r64c64, q4-mma-r64c128,\n"
        "                             q5-gemv-r16s2x, q5-simt-r8c4, q5-simt-r8c8,\n"
        "                             q5-simt-split2-exact, q5-simt-split4-exact,\n"
        "                             q5-mma-r64c64, q5-mma-r64c128,\n"
        "                             q6-simt-r8c4,\n"
        "                             q6-mma-r64c64, q6-mma-r64c128,\n"
        "                             w8-simt-r8c4, w8-simt-r8c8,\n"
        "                             w8-mma-r32c128, or w8-mma-r64c128.\n"
        "  --variant NAME             Fixed kernel variant; auto is the default.\n"
        "  --paired-kv                Benchmark paired MTP K/V (requires MtpKV + W8G32).\n"
        "  --warmup N                 Cold-cache warmup GEMV launches (default %d).\n"
        "  --repeat N                 Cold-cache measured GEMV launches (default %d).\n"
        "  --copy-repeat N            Cold-cache copy-ceiling samples (default %d).\n"
        "  --flush-mib N              L2 flush buffer size in MiB (default 256).\n"
        "  --copy-mib N               Copy-ceiling buffer size in MiB (default 512).\n"
        "  --stream-ceiling-gbs X     Use a premeasured copy ceiling instead of measuring it.\n"
        "  --t-sweep L                Comma list of activation columns T, e.g. 1,8,64,512.\n"
        "                             Fixed GEMV defaults to 1; other paths default to\n"
        "                             1,2,4,8,16,32,64,128,256,512,1024,2048.\n"
        "  --csv-out PATH             Write the result table as CSV.\n",
        argv0, argv0, argv0, kDefaultWarmup, kDefaultRepeat, kDefaultCopyRepeat);
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
        } else if (arg == "--rows") {
            opt.rows        = parse_int(next("rows"), "rows");
            opt.have_rows   = true;
            opt.all_targets = false;
        } else if (arg == "--k") {
            opt.k           = parse_int(next("k"), "k");
            opt.have_k      = true;
            opt.all_targets = false;
        } else if (arg == "--qtype") {
            opt.qtype       = parse_qtype(next("qtype"));
            opt.have_qtype  = true;
            opt.all_targets = false;
        } else if (arg == "--candidate") {
            parse_candidate(opt, next("candidate"));
            opt.all_targets = false;
        } else if (arg == "--variant") {
            const std::string_view raw = next("variant");
            opt.q4_variant             = parse_q4_variant(raw, opt.q4_variant_auto);
            opt.q5_variant             = parse_q5_variant(raw, opt.q5_variant_auto);
            opt.q6_variant             = parse_q6_variant(raw, opt.q6_variant_auto);
            opt.w8_variant             = parse_w8_variant(raw, opt.w8_variant_auto);
            opt.all_targets            = false;
        } else if (arg == "--paired-kv") {
            opt.paired_kv   = true;
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
        } else if (arg == "--t-sweep") {
            opt.t_sweep = parse_int_list(next("t-sweep"));
        } else if (arg == "--csv-out") {
            opt.csv_out = next("csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    const bool numeric_shape = opt.have_rows || opt.have_k;
    if (numeric_shape && (!opt.have_rows || !opt.have_k)) {
        throw std::invalid_argument("--rows and --k must be provided together");
    }
    if (numeric_shape && opt.have_shape) {
        throw std::invalid_argument("--rows/--k cannot be combined with --shape");
    }
    if (!opt.all_targets && !numeric_shape && (!opt.have_shape || !opt.have_qtype)) {
        throw std::invalid_argument("--shape and --qtype must be provided together");
    }
    if (!opt.all_targets && numeric_shape && !opt.have_qtype) {
        throw std::invalid_argument("--rows/--k require --qtype");
    }
    if (opt.candidate != CandidateKind::Auto && (opt.paired_kv || opt.all_targets)) {
        throw std::invalid_argument("fixed candidates require one non-paired shape");
    }
    if (opt.candidate == CandidateKind::Q4Fixed && !numeric_shape && !opt.have_shape) {
        throw std::invalid_argument("Q4 fixed candidate requires one shape");
    }
    if (opt.candidate == CandidateKind::Q4Fixed && opt.qtype != QType::Q4G64_F16S) {
        throw std::invalid_argument("Q4 fixed candidate requires Q4");
    }
    if (opt.candidate == CandidateKind::Q5Fixed && !numeric_shape && !opt.have_shape) {
        throw std::invalid_argument("Q5 fixed candidate requires one shape");
    }
    if (opt.candidate == CandidateKind::Q5Fixed && opt.qtype != QType::Q5G64_F16S) {
        throw std::invalid_argument("Q5 fixed candidate requires Q5");
    }
    if (opt.candidate == CandidateKind::Q6Fixed && !numeric_shape && !opt.have_shape) {
        throw std::invalid_argument("Q6 fixed candidate requires one shape");
    }
    if (opt.candidate == CandidateKind::Q6Fixed && opt.qtype != QType::Q6G64_F16S) {
        throw std::invalid_argument("Q6 fixed candidate requires Q6");
    }
    if (opt.candidate == CandidateKind::W8Fixed && !numeric_shape && !opt.have_shape) {
        throw std::invalid_argument("W8 fixed candidate requires one shape");
    }
    if (opt.candidate == CandidateKind::W8Fixed && opt.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("W8 fixed candidate requires W8G32");
    }
    if (opt.paired_kv &&
        (std::string_view(opt.shape.name) != "MtpKV1024x5120" || opt.qtype != QType::W8G32_F16S)) {
        throw std::invalid_argument("--paired-kv requires --shape MtpKV1024x5120 --qtype W8G32");
    }
    if (opt.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (opt.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (opt.copy_repeat <= 0) { throw std::invalid_argument("--copy-repeat must be positive"); }
    if (numeric_shape && (opt.rows <= 0 || opt.k <= 0)) {
        throw std::invalid_argument("--rows and --k must be positive");
    }
    if (opt.flush_bytes == 0) { throw std::invalid_argument("--flush-mib must be positive"); }
    if (opt.copy_bytes == 0 || (opt.copy_bytes % sizeof(uint4)) != 0) {
        throw std::invalid_argument("--copy-mib must produce a positive uint4-aligned byte count");
    }
    return opt;
}

std::vector<int> default_t_sweep(const TargetSpec& target, const Options& opt) {
    if (opt.paired_kv) { return {1, 4, 5, 16, 17, 56, 57, 128}; }
    if (opt.candidate == CandidateKind::Q4Fixed) {
        const bool gemv = opt.q4_schedule == ops::detail::Q4ScheduleId::GemvR4W1Direct ||
                          opt.q4_schedule == ops::detail::Q4ScheduleId::GemvR1W8Direct;
        return gemv ? std::vector<int>{1}
                    : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    }
    if (opt.candidate == CandidateKind::Q5Fixed) {
        if (opt.q5_schedule == ops::detail::Q5ScheduleId::GemvR16S2X) { return {1}; }
        if (opt.q5_schedule == ops::detail::Q5ScheduleId::SimtSplit2Exact ||
            opt.q5_schedule == ops::detail::Q5ScheduleId::SimtSplit4Exact) {
            return {2, 3, 4, 5, 6};
        }
        return {1, 2, 4, 8, 16, 24, 25, 32, 64, 128, 256, 512, 1024, 2048};
    }
    if (opt.candidate == CandidateKind::Q6Fixed) {
        return {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    }
    if (opt.candidate == CandidateKind::W8Fixed) {
        return {1, 2, 4, 5, 8, 9, 16, 17, 32, 56, 57, 64, 128, 256, 512, 1024, 2048};
    }
    if (target.qtype == QType::Q4G64_F16S) {
        if (target.shape.n == 131072 && target.shape.k == 5120) { return {1}; }
        if (target.shape.n == 34816 && target.shape.k == 5120) { return {2, 4, 5, 8, 16}; }
        if (target.shape.k == 5120) { return {1, 2, 4, 5, 8, 16}; }
        if (target.shape.k == 1152) {
            return {4, 8, 12, 16, 24, 28, 36, 40, 64, 128, 320, 324, 512, 2048};
        }
    }
    if (target.qtype == QType::Q6G64_F16S) {
        if (target.shape.n == 248320 && target.shape.k == 5120) { return {1, 2, 3, 4, 5, 6}; }
        if (target.shape.n == 1152 && target.shape.k == 1536) {
            return {4,   36,  40,  64,  128,  704,  708,  768,  772,  832, 836,
                    896, 900, 960, 964, 1024, 1028, 1088, 1092, 1152, 2048};
        }
    }
    if (target.qtype == QType::Q5G64_F16S) {
        if (target.shape.n == 1024 && target.shape.k == 5120) { return {1, 2, 4, 5, 8, 16}; }
        if (target.shape.n == 6144 && target.shape.k == 5120) {
            return {1, 2, 4, 6, 7, 16, 24, 25, 64, 65, 128, 512, 1024, 2048};
        }
        if (target.shape.n == 5120 && (target.shape.k == 6144 || target.shape.k == 17408)) {
            return {2, 4, 6, 7, 16, 24};
        }
        if (target.shape.n == 1152 && target.shape.k == 1152) {
            return {4, 8, 56, 60, 64, 128, 512, 2048};
        }
        if (target.shape.n == 1152 && target.shape.k == 4304) {
            return {4, 8, 84, 88, 92, 128, 512, 2048};
        }
    }
    if (target.qtype == QType::W8G32_F16S) {
        if (target.shape.n == 14336 || target.shape.n == 34816) { return {1, 4, 5, 8, 9, 16, 128}; }
        if (target.shape.k == 4608) { return {1, 4, 5, 6, 16, 128}; }
        return {1, 4, 5, 8, 16, 17, 128};
    }
    return {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
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
    stats.p95_us =
        sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(0.95 * sorted.size()))];
    stats.mean_us = sum / static_cast<double>(samples.size());
    return stats;
}

// Warm (no L2 flush between samples): diagnostic for intra-call L2 reuse across
// T-tiles. For weights larger than L2 this converges to the cold number.
template <class Launch>
TimingStats measure_warm(Launch&& launch, cudaStream_t stream, int warmup, int repeat) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) { launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
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
    stats.p95_us =
        sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(0.95 * sorted.size()))];
    stats.mean_us = sum / static_cast<double>(samples.size());
    return stats;
}

double measure_tc_ceiling_tflops(cudaStream_t stream) {
    constexpr int kGrid  = 1024;
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / 32;
    constexpr int kNacc  = 8;
    constexpr int kIters = 4096;
    DeviceBuffer sink(static_cast<std::uint64_t>(kGrid) * kBlock * sizeof(float));

    tc_peak_kernel<kNacc><<<kGrid, kBlock, 0, stream>>>(static_cast<float*>(sink.p), kIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double best_ms = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        tc_peak_kernel<kNacc><<<kGrid, kBlock, 0, stream>>>(static_cast<float*>(sink.p), kIters);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best_ms = std::min(best_ms, static_cast<double>(ms));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 2*M*N*K flops per mma (m16n8k16), per warp per accumulator per iteration.
    const double flops =
        static_cast<double>(kGrid) * kWarps * kNacc * kIters * 2.0 * 16.0 * 8.0 * 16.0;
    const double sec = best_ms * 1e-3;
    return sec > 0.0 ? flops / sec / 1e12 : 0.0;
}

double measure_stream_copy_ceiling_gbs(std::uint64_t copy_bytes, int repeat, DeviceBuffer& flush,
                                       cudaStream_t stream) {
    DeviceBuffer src(copy_bytes);
    DeviceBuffer dst(copy_bytes);
    CUDA_CHECK(cudaMemsetAsync(src.p, 0x5a, src.bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst.p, 0x00, dst.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::uint64_t words = copy_bytes / sizeof(uint4);
    const int block           = 256;
    const int grid            = grid_for(words, block);
    const TimingStats stats   = measure_cold(
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
            ninfer::bench::f32_to_bf16(0.5f - static_cast<float>(i % 251ULL) / 250.0f);
    }
    DeviceBuffer d(n * 2ULL);
    CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
    return d;
}

RowSplitPayload make_row_split_payload(QType qtype, std::int32_t n, std::int32_t k,
                                       cudaStream_t stream) {
    const std::int32_t padded_k = static_cast<std::int32_t>(align_up_u64(k, 128));
    const std::int32_t group    = group_size(qtype);
    const std::int32_t kg       = padded_k / group;
    const std::uint64_t groups  = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg);
    const std::uint64_t nibble_bytes =
        groups * static_cast<std::uint64_t>(nibble_bytes_per_group(qtype));
    const std::uint64_t high_bytes =
        groups * static_cast<std::uint64_t>(high_bytes_per_group(qtype));
    const std::uint64_t high_offset  = align_up_u64(nibble_bytes, 256);
    const std::uint64_t scale_offset = high_offset + align_up_u64(high_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2ULL;

    RowSplitPayload payload{DeviceBuffer(scale_offset + scale_bytes),
                            nibble_bytes,
                            high_offset,
                            high_bytes,
                            scale_offset,
                            scale_bytes};
    constexpr int block = 256;
    fill_codes_kernel<<<grid_for(nibble_bytes, block), block, 0, stream>>>(
        static_cast<std::uint8_t*>(payload.data.p), nibble_bytes);
    CUDA_CHECK(cudaGetLastError());
    if (high_bytes != 0) {
        fill_codes_kernel<<<grid_for(high_bytes, block), block, 0, stream>>>(
            static_cast<std::uint8_t*>(payload.data.p) + high_offset, high_bytes);
    }
    CUDA_CHECK(cudaGetLastError());
    fill_scales_kernel<<<grid_for(groups, block), block, 0, stream>>>(
        static_cast<std::uint8_t*>(payload.data.p) + scale_offset, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return payload;
}

Weight make_weight(const RowSplitPayload& payload, QType qtype, std::int32_t n, std::int32_t k) {
    const std::int32_t padded_k = static_cast<std::int32_t>(align_up_u64(k, 128));
    const std::int32_t group    = group_size(qtype);
    Weight w{};
    w.payload          = payload.data.p;
    w.payload_bytes    = payload.payload_bytes();
    w.high_plane_bytes = payload.high_bytes;
    w.qtype            = qtype;
    w.layout           = QuantLayout::RowSplit;
    w.scale_dtype      = DType::FP16;
    w.group_size       = static_cast<std::uint32_t>(group);
    w.shape[0]         = n;
    w.shape[1]         = k;
    w.padded_shape[0]  = n;
    w.padded_shape[1]  = padded_k;
    w.ndim             = 2;
    w.qdata            = payload.data.p;
    w.qhigh            = payload.high_bytes == 0
                             ? nullptr
                             : static_cast<const std::uint8_t*>(payload.data.p) + payload.high_offset;
    w.scales           = static_cast<const std::uint8_t*>(payload.data.p) + payload.scale_offset;
    w.n                = n;
    w.k                = k;
    w.group            = group;
    return w;
}

ops::detail::Q4KernelVariant selected_q4_variant(const Options& opt, const ShapeSpec& shape,
                                                 std::int32_t t) {
    if (opt.candidate != CandidateKind::Q4Fixed) { return ops::detail::Q4KernelVariant::None; }
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return opt.q4_variant_auto ? resolve_q4_variant(opt.q4_schedule, shape.n, shape.k, padded_k, t)
                               : opt.q4_variant;
}

ops::detail::Q5KernelVariant selected_q5_variant(const Options& opt, const ShapeSpec& shape,
                                                 std::int32_t t) {
    if (opt.candidate != CandidateKind::Q5Fixed) { return ops::detail::Q5KernelVariant::None; }
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return opt.q5_variant_auto ? resolve_q5_variant(opt.q5_schedule, shape.n, shape.k, padded_k, t)
                               : opt.q5_variant;
}

ops::detail::Q6KernelVariant selected_q6_variant(const Options& opt, const ShapeSpec& shape,
                                                 std::int32_t t) {
    if (opt.candidate != CandidateKind::Q6Fixed) { return ops::detail::Q6KernelVariant::None; }
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return opt.q6_variant_auto ? resolve_q6_variant(opt.q6_schedule, shape.n, shape.k, padded_k, t)
                               : opt.q6_variant;
}

ops::detail::W8KernelVariant selected_w8_variant(const Options& opt, const ShapeSpec& shape,
                                                 std::int32_t t) {
    if (opt.candidate != CandidateKind::W8Fixed) { return ops::detail::W8KernelVariant::None; }
    const auto padded_k = static_cast<std::int32_t>(align_up_u64(shape.k, 128));
    return opt.w8_variant_auto ? resolve_w8_variant(opt.w8_schedule, shape.n, shape.k, padded_k, t)
                               : opt.w8_variant;
}

void launch_candidate(const Options& opt, const ShapeSpec& shape, const Tensor& x, const Weight& w,
                      Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    switch (opt.candidate) {
    case CandidateKind::Auto:
        ops::linear(x, w, out, ws, stream);
        return;
    case CandidateKind::Q4Fixed: {
        const auto variant = selected_q4_variant(opt, shape, x.ne[1]);
        ops::detail::q4_rowsplit_launch_candidate(opt.q4_schedule, variant, x, w, out, stream);
        return;
    }
    case CandidateKind::Q5Fixed: {
        const auto variant = selected_q5_variant(opt, shape, x.ne[1]);
        ops::detail::q5_rowsplit_launch_candidate(opt.q5_schedule, variant, x, w, out, stream);
        return;
    }
    case CandidateKind::Q6Fixed: {
        const auto variant = selected_q6_variant(opt, shape, x.ne[1]);
        ops::detail::q6_rowsplit_launch_candidate(opt.q6_schedule, variant, x, w, out, stream);
        return;
    }
    case CandidateKind::W8Fixed: {
        const auto variant = selected_w8_variant(opt, shape, x.ne[1]);
        ops::detail::w8_rowsplit_launch_candidate(opt.w8_schedule, variant, x, w, out, stream);
        return;
    }
    }
    throw std::invalid_argument("unknown Linear benchmark candidate");
}

int schedule_col_tile(ops::detail::Q4ScheduleId schedule) {
    using S = ops::detail::Q4ScheduleId;
    switch (schedule) {
    case S::GemvR4W1Direct:
    case S::GemvR1W8Direct:
        return 1;
    case S::SimtR8C4:
        return 4;
    case S::SimtR8C8:
        return 8;
    case S::MmaR64C64:
        return 64;
    case S::MmaR64C128:
        return 128;
    }
    throw std::invalid_argument("unknown Q4 schedule tile");
}

int schedule_col_tile(ops::detail::Q6ScheduleId schedule) {
    using S = ops::detail::Q6ScheduleId;
    switch (schedule) {
    case S::SimtR8C4:
        return 4;
    case S::MmaR64C64:
        return 64;
    case S::MmaR64C128:
        return 128;
    }
    throw std::invalid_argument("unknown Q6 schedule tile");
}

int schedule_col_tile(ops::detail::Q5ScheduleId schedule, std::int32_t exact_cols) {
    using S = ops::detail::Q5ScheduleId;
    switch (schedule) {
    case S::GemvR16S2X:
        return 1;
    case S::SimtR8C4:
        return 4;
    case S::SimtR8C8:
        return 8;
    case S::SimtSplit2Exact:
    case S::SimtSplit4Exact:
        return exact_cols;
    case S::MmaR64C64:
        return 64;
    case S::MmaR64C128:
        return 128;
    }
    throw std::invalid_argument("unknown Q5 schedule tile");
}

int schedule_col_tile(ops::detail::W8ScheduleId schedule) {
    using S = ops::detail::W8ScheduleId;
    switch (schedule) {
    case S::SimtR8C4:
        return 4;
    case S::SimtR8C8:
        return 8;
    case S::MmaR32C128:
    case S::MmaR64C128:
        return 128;
    }
    throw std::invalid_argument("unknown W8 schedule tile");
}

int candidate_col_tile(const Options& opt, QType qtype, const ShapeSpec& shape, std::int32_t t) {
    switch (opt.candidate) {
    case CandidateKind::Auto:
        if (qtype == QType::Q4G64_F16S) {
            return schedule_col_tile(resolve_auto_q4_plan(shape, t).schedule);
        }
        if (qtype == QType::Q5G64_F16S) {
            return schedule_col_tile(resolve_auto_q5_plan(shape, t).schedule, t);
        }
        if (qtype == QType::Q6G64_F16S) {
            return schedule_col_tile(resolve_auto_q6_plan(shape, t).schedule);
        }
        if (qtype == QType::W8G32_F16S) {
            return schedule_col_tile(resolve_auto_w8_plan(shape, t).schedule);
        }
        throw std::invalid_argument("unsupported qtype for auto candidate tile");
    case CandidateKind::Q4Fixed:
        return schedule_col_tile(opt.q4_schedule);
    case CandidateKind::Q5Fixed:
        return schedule_col_tile(opt.q5_schedule, t);
    case CandidateKind::Q6Fixed:
        return schedule_col_tile(opt.q6_schedule);
    case CandidateKind::W8Fixed:
        return schedule_col_tile(opt.w8_schedule);
    }
    throw std::invalid_argument("unknown Linear benchmark candidate tile");
}

bool candidate_uses_mma(const Options& opt, QType qtype, const ShapeSpec& shape, std::int32_t t) {
    if (opt.candidate == CandidateKind::Q4Fixed) {
        return ops::detail::q4_schedule_uses_mma(opt.q4_schedule);
    }
    if (opt.candidate == CandidateKind::Q5Fixed) {
        return ops::detail::q5_schedule_uses_mma(opt.q5_schedule);
    }
    if (opt.candidate == CandidateKind::Q6Fixed) {
        return ops::detail::q6_schedule_uses_mma(opt.q6_schedule);
    }
    if (opt.candidate == CandidateKind::W8Fixed) {
        return ops::detail::w8_schedule_uses_mma(opt.w8_schedule);
    }
    if (opt.candidate != CandidateKind::Auto) { return false; }
    if (qtype == QType::Q4G64_F16S) {
        return ops::detail::q4_schedule_uses_mma(resolve_auto_q4_plan(shape, t).schedule);
    }
    if (qtype == QType::Q5G64_F16S) {
        return ops::detail::q5_schedule_uses_mma(resolve_auto_q5_plan(shape, t).schedule);
    }
    if (qtype == QType::Q6G64_F16S) {
        return ops::detail::q6_schedule_uses_mma(resolve_auto_q6_plan(shape, t).schedule);
    }
    return qtype == QType::W8G32_F16S &&
           ops::detail::w8_schedule_uses_mma(resolve_auto_w8_plan(shape, t).schedule);
}

int candidate_mma_row_tile(const Options& opt, QType qtype, const ShapeSpec& shape,
                           std::int32_t t) {
    ops::detail::W8ScheduleId schedule{};
    if (opt.candidate == CandidateKind::W8Fixed) {
        schedule = opt.w8_schedule;
    } else if (opt.candidate == CandidateKind::Auto && qtype == QType::W8G32_F16S) {
        schedule = resolve_auto_w8_plan(shape, t).schedule;
    } else {
        return 64;
    }
    switch (schedule) {
    case ops::detail::W8ScheduleId::MmaR32C128:
        return 32;
    case ops::detail::W8ScheduleId::MmaR64C128:
        return 64;
    case ops::detail::W8ScheduleId::SimtR8C4:
    case ops::detail::W8ScheduleId::SimtR8C8:
        throw std::logic_error("SIMT W8 schedule has no MMA row tile");
    }
    throw std::logic_error("unknown W8 schedule row tile");
}

RunResult run_target(const TargetSpec& target, const Options& opt, double stream_ceiling_gbs,
                     double tc_ceiling_tflops, std::int32_t t, DeviceBuffer& flush,
                     cudaStream_t stream) {
    const ShapeSpec& shape = target.shape;
    const std::uint64_t tt = static_cast<std::uint64_t>(t);
    DeviceBuffer x         = make_bf16_device(static_cast<std::uint64_t>(shape.k) * tt);
    RowSplitPayload wbuf   = make_row_split_payload(target.qtype, shape.n, shape.k, stream);
    DeviceBuffer out(static_cast<std::uint64_t>(shape.n) * tt * 2ULL);
    CUDA_CHECK(cudaMemsetAsync(out.p, 0, out.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor tx(x.p, DType::BF16, {shape.k, t});
    Tensor tout(out.p, DType::BF16, {shape.n, t});
    Weight w = make_weight(wbuf, target.qtype, shape.n, shape.k);
    WorkspaceArena ws(64ULL << 20);

    const auto launch = [&](cudaStream_t s) { launch_candidate(opt, shape, tx, w, tout, ws, s); };
    const TimingStats cold = measure_cold(launch, flush, stream, opt.warmup, opt.repeat);
    const TimingStats warm = measure_warm(launch, stream, opt.warmup, opt.repeat);

    const std::uint64_t x_bytes       = static_cast<std::uint64_t>(shape.k) * tt * 2ULL;
    const std::uint64_t out_bytes     = static_cast<std::uint64_t>(shape.n) * tt * 2ULL;
    const std::uint64_t useful_bytes  = w.payload_bytes + x_bytes + out_bytes;
    const int col_tile                = candidate_col_tile(opt, target.qtype, shape, t);
    const std::uint64_t weight_passes = static_cast<std::uint64_t>((t + col_tile - 1) / col_tile);
    const std::uint64_t weight_replay_lower_bound_bytes =
        w.payload_bytes * weight_passes + x_bytes + out_bytes;
    const double sec        = cold.median_us * 1e-6;
    const double useful_gbs = sec > 0.0 ? static_cast<double>(useful_bytes) / sec / 1e9 : 0.0;
    const double weight_replay_lower_bound_gbs =
        sec > 0.0 ? static_cast<double>(weight_replay_lower_bound_bytes) / sec / 1e9 : 0.0;
    const double useful_copy_ceiling_pct =
        stream_ceiling_gbs > 0.0 ? useful_gbs / stream_ceiling_gbs * 100.0 : 0.0;
    const double useful_flops =
        2.0 * static_cast<double>(shape.n) * static_cast<double>(shape.k) * static_cast<double>(t);
    double executed_tflops = std::numeric_limits<double>::quiet_NaN();
    double executed_tc_pct = std::numeric_limits<double>::quiet_NaN();
    if (candidate_uses_mma(opt, target.qtype, shape, t)) {
        const std::int64_t executed_rows =
            align_up_u64(shape.n, candidate_mma_row_tile(opt, target.qtype, shape, t));
        const std::int64_t executed_cols = align_up_u64(t, col_tile);
        const double executed_flops      = 2.0 * static_cast<double>(executed_rows) *
                                      static_cast<double>(shape.k) *
                                      static_cast<double>(executed_cols);
        executed_tflops = sec > 0.0 ? executed_flops / sec / 1e12 : 0.0;
        executed_tc_pct =
            tc_ceiling_tflops > 0.0 ? executed_tflops / tc_ceiling_tflops * 100.0 : 0.0;
    }
    const double useful_tflops = sec > 0.0 ? useful_flops / sec / 1e12 : 0.0;
    const double useful_copy_floor_us =
        stream_ceiling_gbs > 0.0
            ? static_cast<double>(useful_bytes) / (stream_ceiling_gbs * 1e9) * 1e6
            : 0.0;

    RunResult r;
    r.shape_name     = shape.name;
    r.qtype_name     = qtype_name(target.qtype);
    r.candidate_name = candidate_name(opt, target.qtype, shape, t);
    if (opt.candidate == CandidateKind::Q4Fixed) {
        r.kernel_variant = ops::detail::q4_kernel_variant_name(selected_q4_variant(opt, shape, t));
    } else if (opt.candidate == CandidateKind::Q5Fixed) {
        r.kernel_variant = ops::detail::q5_kernel_variant_name(selected_q5_variant(opt, shape, t));
    } else if (opt.candidate == CandidateKind::Q6Fixed) {
        r.kernel_variant = ops::detail::q6_kernel_variant_name(selected_q6_variant(opt, shape, t));
    } else if (opt.candidate == CandidateKind::W8Fixed) {
        r.kernel_variant = ops::detail::w8_kernel_variant_name(selected_w8_variant(opt, shape, t));
    } else if (opt.candidate == CandidateKind::Auto && target.qtype == QType::Q4G64_F16S) {
        r.kernel_variant =
            ops::detail::q4_kernel_variant_name(resolve_auto_q4_plan(shape, t).variant);
    } else if (opt.candidate == CandidateKind::Auto && target.qtype == QType::Q5G64_F16S) {
        r.kernel_variant =
            ops::detail::q5_kernel_variant_name(resolve_auto_q5_plan(shape, t).variant);
    } else if (opt.candidate == CandidateKind::Auto && target.qtype == QType::Q6G64_F16S) {
        r.kernel_variant =
            ops::detail::q6_kernel_variant_name(resolve_auto_q6_plan(shape, t).variant);
    } else if (opt.candidate == CandidateKind::Auto && target.qtype == QType::W8G32_F16S) {
        r.kernel_variant =
            ops::detail::w8_kernel_variant_name(resolve_auto_w8_plan(shape, t).variant);
    } else {
        r.kernel_variant = "n/a";
    }
    r.n                               = shape.n;
    r.k                               = shape.k;
    r.t                               = t;
    r.weight_payload_bytes            = w.payload_bytes;
    r.x_bytes                         = x_bytes;
    r.out_bytes                       = out_bytes;
    r.useful_bytes                    = useful_bytes;
    r.weight_replay_lower_bound_bytes = weight_replay_lower_bound_bytes;
    r.cold_median_us                  = cold.median_us;
    r.cold_min_us                     = cold.min_us;
    r.cold_p95_us                     = cold.p95_us;
    r.warm_median_us                  = warm.median_us;
    r.useful_gbs                      = useful_gbs;
    r.weight_replay_lower_bound_gbs   = weight_replay_lower_bound_gbs;
    r.useful_copy_ceiling_pct         = useful_copy_ceiling_pct;
    r.useful_tflops                   = useful_tflops;
    r.executed_tflops                 = executed_tflops;
    r.tc_ceiling_tflops               = tc_ceiling_tflops;
    r.executed_tc_pct                 = executed_tc_pct;
    r.stream_ceiling_gbs              = stream_ceiling_gbs;
    r.useful_copy_floor_us            = useful_copy_floor_us;
    r.repeat                          = opt.repeat;
    r.warmup                          = opt.warmup;
    r.flush_bytes                     = opt.flush_bytes;
    fill_environment(r);
    return r;
}

RunResult run_paired_kv(const Options& opt, double stream_ceiling_gbs, double tc_ceiling_tflops,
                        std::int32_t t, DeviceBuffer& flush, cudaStream_t stream) {
    constexpr ShapeSpec shape{"MtpKVPair1024x5120", 1024, 5120};
    const std::uint64_t tt = static_cast<std::uint64_t>(t);
    DeviceBuffer x         = make_bf16_device(static_cast<std::uint64_t>(shape.k) * tt);
    RowSplitPayload kbuf   = make_row_split_payload(QType::W8G32_F16S, shape.n, shape.k, stream);
    RowSplitPayload vbuf   = make_row_split_payload(QType::W8G32_F16S, shape.n, shape.k, stream);
    DeviceBuffer kout(static_cast<std::uint64_t>(shape.n) * tt * 2ULL);
    DeviceBuffer vout(static_cast<std::uint64_t>(shape.n) * tt * 2ULL);
    CUDA_CHECK(cudaMemsetAsync(kout.p, 0, kout.bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(vout.p, 0, vout.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor tx(x.p, DType::BF16, {shape.k, t});
    Tensor tk(kout.p, DType::BF16, {shape.n, t});
    Tensor tv(vout.p, DType::BF16, {shape.n, t});
    Weight wk = make_weight(kbuf, QType::W8G32_F16S, shape.n, shape.k);
    Weight wv = make_weight(vbuf, QType::W8G32_F16S, shape.n, shape.k);
    WorkspaceArena ws(64ULL << 20);

    const auto launch      = [&](cudaStream_t s) { ops::linear_pair(tx, wk, wv, tk, tv, ws, s); };
    const TimingStats cold = measure_cold(launch, flush, stream, opt.warmup, opt.repeat);
    const TimingStats warm = measure_warm(launch, stream, opt.warmup, opt.repeat);

    const std::uint64_t weight_bytes = wk.payload_bytes + wv.payload_bytes;
    const std::uint64_t x_bytes      = static_cast<std::uint64_t>(shape.k) * tt * 2ULL;
    const std::uint64_t out_bytes    = 2ULL * static_cast<std::uint64_t>(shape.n) * tt * 2ULL;
    const std::uint64_t useful_bytes = weight_bytes + x_bytes + out_bytes;
    const double sec                 = cold.median_us * 1e-6;
    const double flops =
        4.0 * static_cast<double>(shape.n) * static_cast<double>(shape.k) * static_cast<double>(t);
    const ops::detail::W8PairPlan plan =
        ops::detail::w8_pair_resolve_plan({shape.n, shape.k, shape.k, t});
    const int col_tile = plan.schedule == ops::detail::W8PairScheduleId::TwoSimtR8C4   ? 4
                         : plan.schedule == ops::detail::W8PairScheduleId::TwoSimtR8C8 ? 8
                                                                                       : 128;
    const std::uint64_t weight_passes = static_cast<std::uint64_t>((t + col_tile - 1) / col_tile);
    const std::uint64_t weight_replay_lower_bound_bytes =
        weight_bytes * weight_passes + x_bytes + out_bytes;

    RunResult r;
    r.shape_name                      = shape.name;
    r.qtype_name                      = "W8G32x2";
    r.candidate_name                  = ops::detail::w8_pair_schedule_name(plan.schedule);
    r.kernel_variant                  = ops::detail::w8_kernel_variant_name(plan.variant);
    r.n                               = shape.n * 2;
    r.k                               = shape.k;
    r.t                               = t;
    r.weight_payload_bytes            = weight_bytes;
    r.x_bytes                         = x_bytes;
    r.out_bytes                       = out_bytes;
    r.useful_bytes                    = useful_bytes;
    r.weight_replay_lower_bound_bytes = weight_replay_lower_bound_bytes;
    r.cold_median_us                  = cold.median_us;
    r.cold_min_us                     = cold.min_us;
    r.cold_p95_us                     = cold.p95_us;
    r.warm_median_us                  = warm.median_us;
    r.useful_gbs = sec > 0.0 ? static_cast<double>(useful_bytes) / sec / 1e9 : 0.0;
    r.weight_replay_lower_bound_gbs =
        sec > 0.0 ? static_cast<double>(weight_replay_lower_bound_bytes) / sec / 1e9 : 0.0;
    r.useful_copy_ceiling_pct =
        stream_ceiling_gbs > 0.0 ? r.useful_gbs / stream_ceiling_gbs * 100.0 : 0.0;
    r.useful_tflops = sec > 0.0 ? flops / sec / 1e12 : 0.0;
    if (plan.schedule == ops::detail::W8PairScheduleId::DualMmaR32C128) {
        const auto executed_rows    = align_up_u64(shape.n, 32);
        const auto executed_cols    = align_up_u64(t, 128);
        const double executed_flops = 4.0 * static_cast<double>(executed_rows) *
                                      static_cast<double>(shape.k) *
                                      static_cast<double>(executed_cols);
        r.executed_tflops = sec > 0.0 ? executed_flops / sec / 1e12 : 0.0;
        r.executed_tc_pct =
            tc_ceiling_tflops > 0.0 ? r.executed_tflops / tc_ceiling_tflops * 100.0 : 0.0;
    }
    r.tc_ceiling_tflops    = tc_ceiling_tflops;
    r.stream_ceiling_gbs   = stream_ceiling_gbs;
    r.useful_copy_floor_us = stream_ceiling_gbs > 0.0 ? static_cast<double>(useful_bytes) /
                                                            (stream_ceiling_gbs * 1e9) * 1e6
                                                      : 0.0;
    r.repeat               = opt.repeat;
    r.warmup               = opt.warmup;
    r.flush_bytes          = opt.flush_bytes;
    fill_environment(r);
    return r;
}

void print_table(const std::vector<RunResult>& rows, double stream_ceiling_gbs,
                 double tc_ceiling_tflops, std::uint64_t copy_bytes) {
    std::printf("# stream_ceiling_gbs=%.3f tc_ceiling_tflops=%.3f\n", stream_ceiling_gbs,
                tc_ceiling_tflops);
    if (!rows.empty()) {
        const RunResult& env = rows.front();
        std::printf("# build_type=%s gpu=%s sm=%d%d cuda_runtime=%d cuda_driver=%d\n",
                    env.build_type, env.gpu_name.c_str(), env.compute_major, env.compute_minor,
                    env.cuda_runtime_version, env.cuda_driver_version);
    }
    std::printf("# stream_ceiling_method=cold L2-flushed uint4 copy kernel, bytes_per_sample=%llu "
                "(read+write counted as 2x copy bytes); "
                "tc_ceiling_method=dense bf16 mma.sync m16n8k16 probe\n",
                static_cast<unsigned long long>(copy_bytes));
    std::printf("%-22s %-3s %8s %8s %6s %12s %11s %11s %11s %11s %7s %12s %-12s %s\n", "shape",
                "qt", "N", "K", "T", "cold_us", "use_GB/s", "wreplay_GB/s", "use_TFLOP",
                "exec_TFLOP", "TC_%", "warm_us", "variant", "candidate");
    for (const RunResult& r : rows) {
        std::printf("%-22s %-3s %8d %8d %6d %12.3f %11.3f %11.3f %11.2f %11.2f %7.2f %12.3f "
                    "%-12s %s\n",
                    r.shape_name, r.qtype_name, r.n, r.k, r.t, r.cold_median_us, r.useful_gbs,
                    r.weight_replay_lower_bound_gbs, r.useful_tflops, r.executed_tflops,
                    r.executed_tc_pct, r.warm_median_us, r.kernel_variant,
                    r.candidate_name.c_str());
    }
}

void write_csv(const std::filesystem::path& path, const std::vector<RunResult>& rows) {
    if (path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV output: " + path.string()); }
    // Each row is one (shape, qtype, T, physical candidate) point. Useful counters,
    // column-tiled weight-replay lower bounds, and MMA-resolved Q4/Q5/Q6 executed counters remain
    // distinct; physical traffic and SOL percentages come only from NCU.
    out << "shape,qtype,N,K,weight_payload_bytes,x_bytes,out_bytes,useful_bytes,"
           "weight_replay_lower_bound_bytes,cold_median_us,cold_min_us,cold_p95_us,useful_gbs,"
           "weight_replay_lower_bound_gbs,useful_copy_ceiling_pct,stream_ceiling_gbs,"
           "useful_copy_floor_us,warmup,repeat,flush_bytes,T,useful_tflops,executed_tflops,"
           "tc_ceiling_tflops,executed_tc_pct,warm_median_us,candidate,kernel_variant,build_type,"
           "gpu_name,cuda_runtime_version,cuda_driver_version,compute_major,compute_minor\n";
    for (const RunResult& r : rows) {
        out << r.shape_name << ',' << r.qtype_name << ',' << r.n << ',' << r.k << ','
            << r.weight_payload_bytes << ',' << r.x_bytes << ',' << r.out_bytes << ','
            << r.useful_bytes << ',' << r.weight_replay_lower_bound_bytes << ',' << r.cold_median_us
            << ',' << r.cold_min_us << ',' << r.cold_p95_us << ',' << r.useful_gbs << ','
            << r.weight_replay_lower_bound_gbs << ',' << r.useful_copy_ceiling_pct << ','
            << r.stream_ceiling_gbs << ',' << r.useful_copy_floor_us << ',' << r.warmup << ','
            << r.repeat << ',' << r.flush_bytes << ',' << r.t << ',' << r.useful_tflops << ','
            << r.executed_tflops << ',' << r.tc_ceiling_tflops << ',' << r.executed_tc_pct << ','
            << r.warm_median_us << ',' << r.candidate_name << ',' << r.kernel_variant << ','
            << r.build_type << ',' << r.gpu_name << ',' << r.cuda_runtime_version << ','
            << r.cuda_driver_version << ',' << r.compute_major << ',' << r.compute_minor << '\n';
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
        const double tc_ceiling_tflops = measure_tc_ceiling_tflops(stream);

        std::vector<TargetSpec> targets;
        if (opt.all_targets) {
            targets.assign(std::begin(kTask2Targets), std::end(kTask2Targets));
        } else if (opt.have_rows) {
            targets.push_back(TargetSpec{{"Numeric", opt.rows, opt.k}, opt.qtype});
        } else {
            targets.push_back(TargetSpec{opt.shape, opt.qtype});
        }

        std::vector<RunResult> rows;
        for (const TargetSpec& target : targets) {
            const std::vector<int> sweep =
                opt.t_sweep.empty() ? default_t_sweep(target, opt) : opt.t_sweep;
            for (int t : sweep) {
                if (opt.paired_kv) {
                    rows.push_back(run_paired_kv(opt, stream_ceiling_gbs, tc_ceiling_tflops, t,
                                                 flush, stream));
                } else {
                    rows.push_back(run_target(target, opt, stream_ceiling_gbs, tc_ceiling_tflops, t,
                                              flush, stream));
                }
            }
        }

        print_table(rows, stream_ceiling_gbs, tc_ceiling_tflops, opt.copy_bytes);
        if (!opt.csv_out.empty()) { write_csv(opt.csv_out, rows); }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_linear_op_bench: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }
}
