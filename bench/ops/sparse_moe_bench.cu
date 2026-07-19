#include "ninfer/ops/sparse_moe.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHidden             = 2048;
constexpr int kExperts            = 256;
constexpr int kTopK               = 8;
constexpr int kIntermediate       = 512;
constexpr int kRouterRows         = kExperts + 1;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct DeviceBuffer {
    void* data        = nullptr;
    std::size_t bytes = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t size) : bytes(size) {
        if (bytes != 0) { CUDA_CHECK(cudaMalloc(&data, bytes)); }
    }

    ~DeviceBuffer() {
        if (data != nullptr) { cudaFree(data); }
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data  = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct QuantGeometry {
    int group;
    int code_bytes_per_group;
    int high_bytes_per_group;
};

QuantGeometry geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 32, 0};
    case QType::Q5G64_F16S:
        return {64, 32, 8};
    case QType::Q6G64_F16S:
        return {64, 32, 16};
    case QType::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("unsupported benchmark codec");
    }
}

__device__ __forceinline__ std::uint32_t payload_hash(std::size_t index, std::uint32_t seed) {
    std::uint32_t value = static_cast<std::uint32_t>(index) ^
                          static_cast<std::uint32_t>(index >> 32) ^ seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__global__ void fill_payload_kernel(std::uint8_t* payload, std::size_t count,
                                    std::uint32_t seed) {
    const std::size_t start  = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t stride = gridDim.x * static_cast<std::size_t>(blockDim.x);
    for (std::size_t index = start; index < count; index += stride) {
        payload[index] = static_cast<std::uint8_t>(payload_hash(index, seed));
    }
}

__global__ void fill_scale_kernel(std::uint16_t* scales, std::size_t count, std::uint32_t seed) {
    const std::size_t start  = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t stride = gridDim.x * static_cast<std::size_t>(blockDim.x);
    for (std::size_t index = start; index < count; index += stride) {
        // Positive finite FP16 scales in [2^-10, 2^-9), with ten varied mantissa bits.
        scales[index] = static_cast<std::uint16_t>(0x1400u | (payload_hash(index, seed) & 0x3ffu));
    }
}

__device__ __forceinline__ std::uint32_t warp_xor(std::uint32_t value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value ^= __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void sparse_moe_d1_payload_control_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ router,
                                                     float* __restrict__ sink) {
    __shared__ std::uint32_t partial[8];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = static_cast<int>(blockIdx.x);
    const uint4 w =
        reinterpret_cast<const uint4*>(router + static_cast<std::int64_t>(row) * kHidden)[tid];
    const uint4 xv         = reinterpret_cast<const uint4*>(x)[tid];
    std::uint32_t checksum = w.x ^ w.y ^ w.z ^ w.w ^ xv.x ^ xv.y ^ xv.z ^ xv.w;
    checksum               = warp_xor(checksum);
    if (lane == 0) { partial[warp] = checksum; }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int index = 1; index < 8; ++index) { partial[0] ^= partial[index]; }
        reinterpret_cast<std::uint32_t*>(sink)[row] = partial[0];
    }
}

template <int GroupK, int HighBytesPerGroup, bool SingleW8 = false>
__device__ __forceinline__ std::uint32_t
payload_group_checksum(const std::uint8_t* codes, const std::uint8_t* high,
                       const std::uint8_t* scales, std::int64_t group_index, int lane) {
    constexpr int kActiveLanes = SingleW8 ? 32 : GroupK / 2;
    if (lane >= kActiveLanes) { return 0; }
    const std::uint8_t* code = codes + group_index * 32;
    std::uint32_t checksum;
    if constexpr (GroupK == 32 && !SingleW8) {
        checksum = *reinterpret_cast<const std::uint16_t*>(code + lane * 2);
    } else {
        checksum = code[lane];
    }
    checksum ^= static_cast<std::uint32_t>(
                    *reinterpret_cast<const std::uint16_t*>(scales + group_index * 2))
                << 8;
    if constexpr (HighBytesPerGroup != 0) {
        checksum ^= static_cast<std::uint32_t>(
                        high[group_index * HighBytesPerGroup + lane / (32 / HighBytesPerGroup)])
                    << 24;
    }
    return checksum;
}

template <int GroupK>
__device__ __forceinline__ std::uint32_t d3_two_row_payload(const std::uint8_t* codes,
                                                            const std::uint8_t* scales, int row0,
                                                            int row1, int lane) {
    constexpr int kGroups        = kHidden / GroupK;
    constexpr int kCodeRowBytes  = kGroups * 32;
    constexpr int kScaleRowBytes = kGroups * 2;
    constexpr int kCodeVectors   = kCodeRowBytes / (32 * sizeof(uint4));
    std::uint32_t checksum0      = 0;
    std::uint32_t checksum1      = 0;
#pragma unroll
    for (int vector = 0; vector < kCodeVectors; ++vector) {
        const int offset  = (vector * 32 + lane) * sizeof(uint4);
        const uint4 code0 = *reinterpret_cast<const uint4*>(
            codes + static_cast<std::int64_t>(row0) * kCodeRowBytes + offset);
        const uint4 code1 = *reinterpret_cast<const uint4*>(
            codes + static_cast<std::int64_t>(row1) * kCodeRowBytes + offset);
        checksum0 ^= code0.x ^ code0.y ^ code0.z ^ code0.w;
        checksum1 ^= code1.x ^ code1.y ^ code1.z ^ code1.w;
    }
    if constexpr (kScaleRowBytes == 64) {
        checksum0 ^= reinterpret_cast<const std::uint16_t*>(
            scales + static_cast<std::int64_t>(row0) * kScaleRowBytes)[lane];
        checksum1 ^= reinterpret_cast<const std::uint16_t*>(
            scales + static_cast<std::int64_t>(row1) * kScaleRowBytes)[lane];
    } else {
        checksum0 ^= reinterpret_cast<const std::uint32_t*>(
            scales + static_cast<std::int64_t>(row0) * kScaleRowBytes)[lane];
        checksum1 ^= reinterpret_cast<const std::uint32_t*>(
            scales + static_cast<std::int64_t>(row1) * kScaleRowBytes)[lane];
    }
    return checksum0 ^ checksum1;
}

template <int RoutedGroupK>
__global__ void sparse_moe_d3_payload_control_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const int* __restrict__ ids,
                                                     const std::uint8_t* __restrict__ routed_codes,
                                                     const std::uint8_t* __restrict__ routed_scales,
                                                     const std::uint8_t* __restrict__ shared_codes,
                                                     const std::uint8_t* __restrict__ shared_scales,
                                                     float* __restrict__ sink) {
    __shared__ uint4 x_stage[256];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < 256) { x_stage[tid] = reinterpret_cast<const uint4*>(x)[tid]; }
    __syncthreads();

    const int row = static_cast<int>(blockIdx.x);
    std::uint32_t checksum;
    if (warp < kTopK) {
        const int row_base = ids[warp] * 1024;
        checksum = d3_two_row_payload<RoutedGroupK>(routed_codes, routed_scales, row_base + row,
                                                    row_base + kIntermediate + row, lane);
    } else {
        checksum =
            d3_two_row_payload<32>(shared_codes, shared_scales, row, kIntermediate + row, lane);
    }
    const uint4 xv = x_stage[(warp * 32 + lane) & 255];
    checksum ^= xv.x ^ xv.y ^ xv.z ^ xv.w;
    checksum = warp_xor(checksum);
    if (lane == 0) {
        reinterpret_cast<std::uint32_t*>(
            sink)[static_cast<std::int64_t>(warp) * kIntermediate + row] = checksum;
    }
}

template <int GroupK, int HighBytesPerGroup>
__device__ __forceinline__ std::uint32_t
d4_row_payload(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               int row, const float* activation, int lane) {
    constexpr int kGroups  = kIntermediate / GroupK;
    std::uint32_t checksum = 0;
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        checksum ^= payload_group_checksum<GroupK, HighBytesPerGroup>(
            codes, high, scales, static_cast<std::int64_t>(row) * kGroups + group, lane);
        if (lane < GroupK / 2) {
            const float2 value =
                reinterpret_cast<const float2*>(activation)[group * (GroupK / 2) + lane];
            checksum ^= __float_as_uint(value.x) ^ __float_as_uint(value.y);
        }
    }
    return checksum;
}

template <int RoutedGroupK, int RoutedHighBytesPerGroup>
__global__ void sparse_moe_d4_payload_control_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ activation,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination) {
    __shared__ std::uint32_t path_checksum[kTopK + 1];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x);
    std::uint32_t checksum;
    if (warp < kTopK) {
        checksum = d4_row_payload<RoutedGroupK, RoutedHighBytesPerGroup>(
            routed_codes, routed_high, routed_scales, ids[warp] * kHidden + row,
            activation + static_cast<std::int64_t>(warp) * kIntermediate, lane);
    } else {
        checksum = d4_row_payload<32, 0>(
            shared_codes, nullptr, shared_scales, row,
            activation + static_cast<std::int64_t>(kTopK) * kIntermediate, lane);
    }
    checksum = warp_xor(checksum);
    if (lane == 0) { path_checksum[warp] = checksum; }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float value = __bfloat162float(destination[row]);
#pragma unroll
        for (int path = 0; path < kTopK; ++path) {
            value += alpha[path] * static_cast<float>(path_checksum[path] & 0xffu) * 1.0e-7f;
        }
        value += *shared_scale * static_cast<float>(path_checksum[kTopK] & 0xffu) * 1.0e-7f;
        destination[row] = __float2bfloat16_rn(value);
    }
}

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, int rows, int columns, std::uint32_t seed)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_bytes_(static_cast<std::size_t>(rows) * groups_per_row_ *
                      geometry_.code_bytes_per_group),
          high_bytes_(static_cast<std::size_t>(rows) * groups_per_row_ *
                      geometry_.high_bytes_per_group),
          scale_bytes_(static_cast<std::size_t>(rows) * groups_per_row_ * 2), codes_(code_bytes_),
          high_(high_bytes_ == 0 ? nullptr : std::make_unique<DeviceBuffer>(high_bytes_)),
          scales_(scale_bytes_) {
        const auto launch_payload = [](DeviceBuffer& buffer, std::uint32_t payload_seed) {
            fill_payload_kernel<<<std::min<std::size_t>(4096, (buffer.bytes + 255) / 256), 256>>>(
                static_cast<std::uint8_t*>(buffer.data), buffer.bytes, payload_seed);
        };
        launch_payload(codes_, seed ^ 0x243f6a88u);
        if (high_) { launch_payload(*high_, seed ^ 0x85a308d3u); }
        const std::size_t count = scale_bytes_ / 2;
        fill_scale_kernel<<<std::min<std::size_t>(4096, (count + 255) / 256), 256>>>(
            static_cast<std::uint16_t*>(scales_.data), count, seed ^ 0x13198a2eu);
        CUDA_CHECK(cudaGetLastError());
    }

    Weight weight() const {
        Weight out{};
        out.payload          = codes_.data;
        out.payload_bytes    = code_bytes_ + high_bytes_ + scale_bytes_;
        out.high_plane_bytes = high_bytes_;
        out.qtype            = qtype_;
        out.group_size       = geometry_.group;
        out.qdata            = codes_.data;
        out.qhigh            = high_ ? high_->data : nullptr;
        out.scales           = scales_.data;
        out.n                = rows_;
        out.k                = columns_;
        out.group            = geometry_.group;
        out.layout           = QuantLayout::RowSplit;
        out.scale_dtype      = DType::FP16;
        out.ndim             = 2;
        out.shape[0]         = rows_;
        out.shape[1]         = columns_;
        out.padded_shape[0]  = rows_;
        out.padded_shape[1]  = columns_;
        return out;
    }

private:
    QType qtype_;
    int rows_;
    int columns_;
    QuantGeometry geometry_;
    int groups_per_row_;
    std::size_t code_bytes_;
    std::size_t high_bytes_;
    std::size_t scale_bytes_;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

Weight dense_weight(void* data, int rows, int columns) {
    Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2ULL;
    out.qtype           = QType::BF16_CTRL;
    out.qdata           = data;
    out.n               = rows;
    out.k               = columns;
    out.layout          = QuantLayout::Contiguous;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

enum class CodecProfile {
    Q4Q5,
    Q4Q6,
    W8W8,
};

enum class ExpertDistribution {
    TraceLike,
    Independent,
    Same,
};

const char* distribution_name(ExpertDistribution distribution) {
    switch (distribution) {
    case ExpertDistribution::TraceLike:
        return "trace-like";
    case ExpertDistribution::Independent:
        return "independent";
    case ExpertDistribution::Same:
        return "same";
    }
    return "unknown";
}

ExpertDistribution parse_distribution(std::string_view value) {
    if (value == "trace-like") return ExpertDistribution::TraceLike;
    if (value == "independent") return ExpertDistribution::Independent;
    if (value == "same") return ExpertDistribution::Same;
    throw std::invalid_argument("--distribution must be trace-like, independent, or same");
}

struct RoutePattern {
    std::vector<std::array<int, kTopK>> selected;
    int unique_experts      = 0;
    double adjacent_overlap = 0.0;
};

RoutePattern make_route_pattern(int tokens, ExpertDistribution distribution, std::uint32_t seed) {
    std::mt19937 random(seed);
    RoutePattern result;
    result.selected.resize(tokens);

    // Preserve the original decode benchmark route so T=1 comparisons remain
    // directly comparable with the pre-Small-T baseline.
    if (tokens == 1) {
        result.selected[0]      = {0, 32, 64, 96, 128, 160, 224, 255};
        result.unique_experts   = kTopK;
        result.adjacent_overlap = kTopK;
        return result;
    }

    const auto sample_unique = [&](const std::vector<int>& excluded) {
        std::array<int, kTopK> selected{};
        std::array<bool, kExperts> unavailable{};
        for (int expert : excluded) { unavailable[expert] = true; }
        std::uniform_int_distribution<int> expert_distribution(0, kExperts - 1);
        for (int rank = 0; rank < kTopK; ++rank) {
            int expert = 0;
            do { expert = expert_distribution(random); } while (unavailable[expert]);
            selected[rank]      = expert;
            unavailable[expert] = true;
        }
        return selected;
    };

    // Trace-like MTP traffic combines a modest hot-expert bias with partial
    // adjacency correlation. Independent and same-expert controls bound the
    // sensitivity to assignment grouping and weight reuse.
    std::array<int, 24> hot_pool{};
    {
        std::array<int, kExperts> experts{};
        for (int expert = 0; expert < kExperts; ++expert) { experts[expert] = expert; }
        std::shuffle(experts.begin(), experts.end(), random);
        std::copy_n(experts.begin(), hot_pool.size(), hot_pool.begin());
    }
    result.selected[0] = sample_unique({});
    for (int token = 1; token < tokens; ++token) {
        if (distribution == ExpertDistribution::Same) {
            result.selected[token] = result.selected[0];
            continue;
        }
        if (distribution == ExpertDistribution::Independent) {
            result.selected[token] = sample_unique({});
            continue;
        }

        std::array<int, kTopK> selected{};
        std::array<bool, kExperts> used{};
        std::array<int, kTopK> previous = result.selected[token - 1];
        std::shuffle(previous.begin(), previous.end(), random);
        int count = 0;
        for (; count < 3; ++count) {
            selected[count]       = previous[count];
            used[selected[count]] = true;
        }
        std::uniform_int_distribution<int> hot_distribution(0, hot_pool.size() - 1);
        while (count < 5) {
            const int expert = hot_pool[hot_distribution(random)];
            if (used[expert]) { continue; }
            selected[count++] = expert;
            used[expert]      = true;
        }
        std::uniform_int_distribution<int> expert_distribution(0, kExperts - 1);
        while (count < kTopK) {
            const int expert = expert_distribution(random);
            if (used[expert]) { continue; }
            selected[count++] = expert;
            used[expert]      = true;
        }
        std::shuffle(selected.begin(), selected.end(), random);
        result.selected[token] = selected;
    }

    std::array<bool, kExperts> present{};
    double overlap_sum = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int expert : result.selected[token]) { present[expert] = true; }
        if (token == 0) { continue; }
        int overlap = 0;
        for (int expert : result.selected[token]) {
            overlap +=
                std::find(result.selected[token - 1].begin(), result.selected[token - 1].end(),
                          expert) != result.selected[token - 1].end();
        }
        overlap_sum += overlap;
    }
    result.unique_experts   = static_cast<int>(std::count(present.begin(), present.end(), true));
    result.adjacent_overlap = tokens > 1 ? overlap_sum / (tokens - 1) : kTopK;
    return result;
}

const char* codec_name(CodecProfile profile) {
    switch (profile) {
    case CodecProfile::Q4Q5:
        return "q4-q5";
    case CodecProfile::Q4Q6:
        return "q4-q6";
    case CodecProfile::W8W8:
        return "w8-w8";
    }
    return "unknown";
}

QType gate_codec(CodecProfile profile) {
    return profile == CodecProfile::W8W8 ? QType::W8G32_F16S : QType::Q4G64_F16S;
}

QType down_codec(CodecProfile profile) {
    switch (profile) {
    case CodecProfile::Q4Q5:
        return QType::Q5G64_F16S;
    case CodecProfile::Q4Q6:
        return QType::Q6G64_F16S;
    case CodecProfile::W8W8:
        return QType::W8G32_F16S;
    }
    throw std::logic_error("unknown codec profile");
}

enum class Scope {
    Full,
    D1,
    D2,
    D3,
    D4,
};

const char* scope_name(Scope scope) {
    switch (scope) {
    case Scope::Full:
        return "full";
    case Scope::D1:
        return "d1";
    case Scope::D2:
        return "d2";
    case Scope::D3:
        return "d3";
    case Scope::D4:
        return "d4";
    }
    return "unknown";
}

struct Candidate {
    Scope scope;
    const char* name;
    ops::detail::SparseMoeDecodePlan plan;
    ops::detail::SparseMoeSmallTPlan small_t_plan;
    bool small_t                = false;
    bool decode_loop            = false;
    bool public_production      = false;
    bool payload_control        = false;
    int grid                    = 0;
    int block                   = 0;
    const char* matched_control = "";
};

struct Stats {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Payload {
    std::uint64_t code         = 0;
    std::uint64_t high         = 0;
    std::uint64_t scale        = 0;
    std::uint64_t useful_fma   = 0;
    std::uint64_t executed_fma = 0;
};

struct Result {
    CodecProfile codec;
    int tokens;
    ExpertDistribution distribution;
    std::uint32_t seed;
    int unique_experts;
    double adjacent_overlap;
    Candidate candidate;
    Stats cold;
    Stats warm;
    Payload payload;
    std::size_t workspace_bytes;
    double matched_cold_us = std::numeric_limits<double>::quiet_NaN();
    double matched_warm_us = std::numeric_limits<double>::quiet_NaN();
};

struct Options {
    int warmup = 3;
    int repeat = 30;
    std::string csv_out;
    bool matrix                     = true;
    Scope scope                     = Scope::Full;
    std::string candidate           = "production";
    CodecProfile codec              = CodecProfile::Q4Q5;
    int tokens                      = 1;
    ExpertDistribution distribution = ExpertDistribution::TraceLike;
    std::uint32_t seed              = 20260718U;
};

Scope parse_scope(std::string_view value) {
    if (value == "full") return Scope::Full;
    if (value == "d1") return Scope::D1;
    if (value == "d2") return Scope::D2;
    if (value == "d3") return Scope::D3;
    if (value == "d4") return Scope::D4;
    throw std::invalid_argument("--scope must be full, d1, d2, d3, or d4");
}

CodecProfile parse_codec(std::string_view value) {
    if (value == "q4-q5") return CodecProfile::Q4Q5;
    if (value == "q4-q6") return CodecProfile::Q4Q6;
    if (value == "w8-w8") return CodecProfile::W8W8;
    throw std::invalid_argument("--codec must be q4-q5, q4-q6, or w8-w8");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--matrix") {
            options.matrix = true;
        } else if (argument == "--scope") {
            options.scope  = parse_scope(next("--scope value"));
            options.matrix = false;
        } else if (argument == "--candidate") {
            options.candidate = next("--candidate value");
            options.matrix    = false;
        } else if (argument == "--codec") {
            options.codec  = parse_codec(next("--codec value"));
            options.matrix = false;
        } else if (argument == "--tokens") {
            options.tokens = std::stoi(std::string(next("--tokens value")));
        } else if (argument == "--distribution") {
            options.distribution = parse_distribution(next("--distribution value"));
        } else if (argument == "--seed") {
            options.seed =
                static_cast<std::uint32_t>(std::stoul(std::string(next("--seed value"))));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out value");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--matrix] [--scope full|d1|d2|d3|d4] "
                        "[--candidate NAME] [--codec q4-q5|q4-q6|w8-w8] [--tokens 1..16384] "
                        "[--distribution trace-like|independent|same] [--seed N] [--warmup N] "
                        "[--repeat N] [--csv-out PATH]\n\n"
                        "Candidates: production, small-t-tiled, decode-loop, payload-control, "
                        "row-cta-4w, row-cta-8w, serial-control, "
                        "warp-register, nine-warp-control, balanced-eight-warp, nine-warp-r1, "
                        "balanced-eight-warp-r4.\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    if (options.tokens < 1 || options.tokens > 16384) {
        throw std::invalid_argument("--tokens must be in [1,16384]");
    }
    return options;
}

std::size_t private_workspace_bytes(int tokens) {
    if (tokens >= ops::detail::kSparseMoeSmallTMin && tokens <= ops::detail::kSparseMoeSmallTMax) {
        return ops::detail::sparse_moe_small_t_workspace_bytes(tokens);
    }
    return ops::detail::sparse_moe_decode_workspace_bytes();
}

class BenchmarkState {
public:
    BenchmarkState(CodecProfile profile, int tokens, ExpertDistribution distribution,
                   std::uint32_t seed)
        : profile_(profile), tokens_(tokens),
          route_pattern_(make_route_pattern(tokens, distribution, seed)),
          input_(static_cast<std::size_t>(tokens) * kHidden * 2),
          residual_seed_(static_cast<std::size_t>(tokens) * kHidden * 2),
          destination_(static_cast<std::size_t>(tokens) * kHidden * 2),
          router_(static_cast<std::size_t>(kRouterRows) * kHidden * 2),
          routed_gate_(gate_codec(profile), kExperts * 1024, kHidden, seed ^ 0xa4093822u),
          routed_down_(down_codec(profile), kExperts * kHidden, kIntermediate,
                       seed ^ 0x299f31d0u),
          shared_gate_(QType::W8G32_F16S, 1024, kHidden, seed ^ 0x082efa98u),
          shared_down_(QType::W8G32_F16S, kHidden, kIntermediate, seed ^ 0xec4e6c89u),
          flush_(kFlushBytes),
          private_arena_(private_workspace_bytes(tokens)),
          decode_arena_(ops::detail::sparse_moe_decode_workspace_bytes()),
          public_workspace_(ops::sparse_moe_workspace_bytes(tokens)) {
        std::vector<std::uint16_t> input(static_cast<std::size_t>(tokens) * kHidden);
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(tokens) * kHidden);
        const bool basis_router = tokens > ops::detail::kSparseMoeSmallTMax;
        const int route_markers = tokens <= 8    ? 8
                                  : tokens <= 16 ? 16
                                  : tokens <= 32 ? 32
                                                 : ops::detail::kSparseMoeSmallTMax;
        for (int token = 0; token < tokens; ++token) {
            for (int index = 0; index < kHidden; ++index) {
                const int centered = (index * 17 + token * 29 + (index ^ token)) % 81 - 40;
                input[static_cast<std::size_t>(token) * kHidden + index] =
                    bench::f32_to_bf16(static_cast<float>(centered) * 0.001f);
                residual[static_cast<std::size_t>(token) * kHidden + index] = bench::f32_to_bf16(
                    0.125f + static_cast<float>((index + 11 * token) % 17) * 0.002f);
            }
            if (basis_router) {
                for (int expert = 0; expert < kExperts; ++expert) {
                    input[static_cast<std::size_t>(token) * kHidden + expert] =
                        bench::f32_to_bf16(-0.25f);
                }
                for (int rank = 0; rank < kTopK; ++rank) {
                    const int expert = route_pattern_.selected[token][rank];
                    input[static_cast<std::size_t>(token) * kHidden + expert] =
                        bench::f32_to_bf16(0.25f - static_cast<float>(rank) / 64.0f);
                }
                input[static_cast<std::size_t>(token) * kHidden + kExperts] =
                    bench::f32_to_bf16(0.0625f);
            } else {
                for (int marker = 0; marker < route_markers; ++marker) {
                    input[static_cast<std::size_t>(token) * kHidden + kHidden - route_markers +
                          marker] = bench::f32_to_bf16(0.0f);
                }
                input[static_cast<std::size_t>(token) * kHidden] = bench::f32_to_bf16(1.0f);
                input[static_cast<std::size_t>(token) * kHidden + kHidden - route_markers + token] =
                    bench::f32_to_bf16(1.0f);
            }
        }
        CUDA_CHECK(cudaMemcpy(input_.data, input.data(), input_.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(residual_seed_.data, residual.data(), residual_seed_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(destination_.data, residual_seed_.data, destination_.bytes,
                              cudaMemcpyDeviceToDevice));
        std::vector<float> router_values(static_cast<std::size_t>(kRouterRows) * kHidden, 0.0f);
        if (basis_router) {
            for (int expert = 0; expert < kExperts; ++expert) {
                router_values[static_cast<std::size_t>(expert) * kHidden + expert] = 16.0f;
            }
            router_values[static_cast<std::size_t>(kExperts) * kHidden + kExperts] = 4.0f;
        } else {
            for (int expert = 0; expert < kExperts; ++expert) {
                router_values[static_cast<std::size_t>(expert) * kHidden] = -8.0f;
            }
            for (int token = 0; token < tokens; ++token) {
                for (int rank = 0; rank < kTopK; ++rank) {
                    const int expert = route_pattern_.selected[token][rank];
                    const int marker = kHidden - route_markers + token;
                    router_values[static_cast<std::size_t>(expert) * kHidden + marker] =
                        12.0f - 0.25f * rank;
                }
            }
            router_values[static_cast<std::size_t>(kExperts) * kHidden] = 0.25f;
        }
        std::vector<std::uint16_t> router_bits(router_values.size());
        for (std::size_t index = 0; index < router_values.size(); ++index) {
            router_bits[index] = bench::f32_to_bf16(router_values[index]);
        }
        CUDA_CHECK(
            cudaMemcpy(router_.data, router_bits.data(), router_.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(flush_.data, 0xa5, flush_.bytes));

        weights_ = {
            dense_weight(router_.data, kRouterRows, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_.weight(),
        };
        x_                  = Tensor(input_.data, DType::BF16, {kHidden, tokens});
        destination_tensor_ = Tensor(destination_.data, DType::BF16, {kHidden, tokens});
        if (tokens == 1 || tokens > ops::detail::kSparseMoeSmallTMax) {
            workspace_ = ops::detail::allocate_sparse_moe_decode_workspace(private_arena_);
        } else {
            small_t_workspace_ =
                ops::detail::allocate_sparse_moe_small_t_workspace(private_arena_, tokens);
            workspace_ = ops::detail::allocate_sparse_moe_decode_workspace(decode_arena_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] const RoutePattern& route_pattern() const noexcept { return route_pattern_; }

    void reset_destination(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(destination_.data, residual_seed_.data, destination_.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.data, 0xa5, flush_.bytes, stream));
    }

    void prepare(const Candidate& candidate, bool cold, cudaStream_t stream) {
        if (cold) { flush(stream); }
        if (candidate.scope == Scope::Full || candidate.scope == Scope::D4) {
            reset_destination(stream);
        }
        if (candidate.small_t) {
            switch (candidate.scope) {
            case Scope::Full:
            case Scope::D1:
                return;
            case Scope::D2:
                ops::detail::sparse_moe_small_t_launch_s1(x_, weights_.router_shared_gate,
                                                          small_t_workspace_, stream);
                return;
            case Scope::D3:
                launch_small_t_front(candidate.small_t_plan, stream);
                return;
            case Scope::D4:
                launch_small_t_front(candidate.small_t_plan, stream);
                ops::detail::sparse_moe_small_t_launch_s3(x_, weights_, candidate.small_t_plan,
                                                          small_t_workspace_, stream);
                return;
            }
        }
        const auto& plan = candidate.plan;
        switch (candidate.scope) {
        case Scope::Full:
        case Scope::D1:
            return;
        case Scope::D2:
            ops::detail::sparse_moe_decode_launch_d1(x_, weights_.router_shared_gate, workspace_,
                                                     plan.d1, stream);
            return;
        case Scope::D3:
            launch_front(plan, stream);
            return;
        case Scope::D4:
            launch_front(plan, stream);
            ops::detail::sparse_moe_decode_launch_d3(x_, weights_, workspace_, plan.d3, stream);
            return;
        }
    }

    void launch(const Candidate& candidate, cudaStream_t stream) {
        if (candidate.decode_loop) {
            for (int token = 0; token < tokens_; ++token) {
                const Tensor x_column     = x_.slice(1, token, 1);
                Tensor destination_column = destination_tensor_.slice(1, token, 1);
                ops::detail::sparse_moe_decode_launch(x_column, weights_, destination_column,
                                                      workspace_, candidate.plan, stream);
            }
            return;
        }
        if (candidate.small_t) {
            switch (candidate.scope) {
            case Scope::Full:
                if (candidate.public_production) {
                    ops::sparse_moe(x_, weights_, ops::SparseMoeEpilogue::AddResidual,
                                    destination_tensor_, public_workspace_, stream);
                } else {
                    ops::detail::sparse_moe_small_t_launch(x_, weights_, destination_tensor_,
                                                           candidate.small_t_plan,
                                                           small_t_workspace_, stream);
                }
                return;
            case Scope::D1:
                ops::detail::sparse_moe_small_t_launch_s1(x_, weights_.router_shared_gate,
                                                          small_t_workspace_, stream);
                return;
            case Scope::D2:
                ops::detail::sparse_moe_small_t_launch_s2(candidate.small_t_plan,
                                                          small_t_workspace_, stream);
                return;
            case Scope::D3:
                ops::detail::sparse_moe_small_t_launch_s3(x_, weights_, candidate.small_t_plan,
                                                          small_t_workspace_, stream);
                return;
            case Scope::D4:
                ops::detail::sparse_moe_small_t_launch_s4(weights_, destination_tensor_,
                                                          candidate.small_t_plan,
                                                          small_t_workspace_, stream);
                return;
            }
        }
        const auto& plan = candidate.plan;
        if (candidate.payload_control) {
            launch_payload_control(candidate.scope, stream);
            return;
        }
        switch (candidate.scope) {
        case Scope::Full:
            if (candidate.public_production) {
                ops::sparse_moe(x_, weights_, ops::SparseMoeEpilogue::AddResidual,
                                destination_tensor_, public_workspace_, stream);
            } else {
                ops::detail::sparse_moe_decode_launch(x_, weights_, destination_tensor_, workspace_,
                                                      plan, stream);
            }
            return;
        case Scope::D1:
            ops::detail::sparse_moe_decode_launch_d1(x_, weights_.router_shared_gate, workspace_,
                                                     plan.d1, stream);
            return;
        case Scope::D2:
            ops::detail::sparse_moe_decode_launch_d2(workspace_, plan.d2, stream);
            return;
        case Scope::D3:
            ops::detail::sparse_moe_decode_launch_d3(x_, weights_, workspace_, plan.d3, stream);
            return;
        case Scope::D4:
            ops::detail::sparse_moe_decode_launch_d4(weights_, destination_tensor_, workspace_,
                                                     plan.d4, stream);
            return;
        }
    }

private:
    void launch_payload_control(Scope scope, cudaStream_t stream) {
        const auto* input = static_cast<const __nv_bfloat16*>(x_.data);
        if (scope == Scope::D1) {
            sparse_moe_d1_payload_control_kernel<<<kRouterRows, 256, 0, stream>>>(
                input, static_cast<const __nv_bfloat16*>(weights_.router_shared_gate.qdata),
                static_cast<float*>(workspace_.scratch.data));
            return;
        }
        if (scope == Scope::D3) {
            const auto* ids = static_cast<const int*>(workspace_.ids.data);
            auto* sink      = static_cast<float*>(workspace_.scratch.data);
            if (profile_ == CodecProfile::W8W8) {
                sparse_moe_d3_payload_control_kernel<32><<<kIntermediate, 288, 0, stream>>>(
                    input, ids, static_cast<const std::uint8_t*>(weights_.routed_gate_up.qdata),
                    static_cast<const std::uint8_t*>(weights_.routed_gate_up.scales),
                    static_cast<const std::uint8_t*>(weights_.shared_gate_up.qdata),
                    static_cast<const std::uint8_t*>(weights_.shared_gate_up.scales), sink);
            } else {
                sparse_moe_d3_payload_control_kernel<64><<<kIntermediate, 288, 0, stream>>>(
                    input, ids, static_cast<const std::uint8_t*>(weights_.routed_gate_up.qdata),
                    static_cast<const std::uint8_t*>(weights_.routed_gate_up.scales),
                    static_cast<const std::uint8_t*>(weights_.shared_gate_up.qdata),
                    static_cast<const std::uint8_t*>(weights_.shared_gate_up.scales), sink);
            }
            return;
        }
        if (scope == Scope::D4) {
            const auto* ids          = static_cast<const int*>(workspace_.ids.data);
            const auto* alpha        = static_cast<const float*>(workspace_.alpha.data);
            const auto* shared_scale = static_cast<const float*>(workspace_.shared_scale.data);
            const auto* activation   = static_cast<const float*>(workspace_.scratch.data);
            const auto* routed_codes = static_cast<const std::uint8_t*>(weights_.routed_down.qdata);
            const auto* routed_high  = static_cast<const std::uint8_t*>(weights_.routed_down.qhigh);
            const auto* routed_scales =
                static_cast<const std::uint8_t*>(weights_.routed_down.scales);
            const auto* shared_codes = static_cast<const std::uint8_t*>(weights_.shared_down.qdata);
            const auto* shared_scales =
                static_cast<const std::uint8_t*>(weights_.shared_down.scales);
            auto* destination = static_cast<__nv_bfloat16*>(destination_tensor_.data);
            switch (profile_) {
            case CodecProfile::Q4Q5:
                sparse_moe_d4_payload_control_kernel<64, 8><<<kHidden, 288, 0, stream>>>(
                    ids, alpha, shared_scale, activation, routed_codes, routed_high, routed_scales,
                    shared_codes, shared_scales, destination);
                return;
            case CodecProfile::Q4Q6:
                sparse_moe_d4_payload_control_kernel<64, 16><<<kHidden, 288, 0, stream>>>(
                    ids, alpha, shared_scale, activation, routed_codes, routed_high, routed_scales,
                    shared_codes, shared_scales, destination);
                return;
            case CodecProfile::W8W8:
                sparse_moe_d4_payload_control_kernel<32, 0><<<kHidden, 288, 0, stream>>>(
                    ids, alpha, shared_scale, activation, routed_codes, routed_high, routed_scales,
                    shared_codes, shared_scales, destination);
                return;
            }
        }
        throw std::logic_error("payload control is not defined for this scope");
    }

    void launch_front(const ops::detail::SparseMoeDecodePlan& plan, cudaStream_t stream) {
        ops::detail::sparse_moe_decode_launch_d1(x_, weights_.router_shared_gate, workspace_,
                                                 plan.d1, stream);
        ops::detail::sparse_moe_decode_launch_d2(workspace_, plan.d2, stream);
    }

    void launch_small_t_front(const ops::detail::SparseMoeSmallTPlan& plan, cudaStream_t stream) {
        ops::detail::sparse_moe_small_t_launch_s1(x_, weights_.router_shared_gate,
                                                  small_t_workspace_, stream);
        ops::detail::sparse_moe_small_t_launch_s2(plan, small_t_workspace_, stream);
    }

    CodecProfile profile_;
    int tokens_;
    RoutePattern route_pattern_;
    DeviceBuffer input_;
    DeviceBuffer residual_seed_;
    DeviceBuffer destination_;
    DeviceBuffer router_;
    DeviceRowSplit routed_gate_;
    DeviceRowSplit routed_down_;
    DeviceRowSplit shared_gate_;
    DeviceRowSplit shared_down_;
    DeviceBuffer flush_;
    DeviceArena private_arena_;
    DeviceArena decode_arena_;
    WorkspaceArena public_workspace_;
    ops::SparseMoeWeights weights_{};
    Tensor x_;
    Tensor destination_tensor_;
    ops::detail::SparseMoeDecodeWorkspace workspace_;
    ops::detail::SparseMoeSmallTWorkspace small_t_workspace_;
};

Stats measure(BenchmarkState& state, const Candidate& candidate, bool cold, int warmup,
              int repeat) {
    cudaStream_t stream = nullptr;
    cudaEvent_t start   = nullptr;
    cudaEvent_t stop    = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) {
        state.prepare(candidate, cold, stream);
        state.launch(candidate, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(repeat);
    for (int index = 0; index < repeat; ++index) {
        state.prepare(candidate, cold, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        state.launch(candidate, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(),
            samples[std::min(samples.size() - 1, static_cast<std::size_t>(0.95 * samples.size()))]};
}

ops::detail::SparseMoeDecodePlan base_plan(CodecProfile profile) {
    return ops::detail::resolve_sparse_moe_decode_plan(gate_codec(profile), down_codec(profile));
}

Candidate make_candidate(CodecProfile profile, Scope scope, std::string_view name, int tokens) {
    Candidate result;
    result.scope = scope;
    result.plan  = base_plan(profile);

    if (tokens > ops::detail::kSparseMoeSmallTMax) {
        if (scope != Scope::Full) {
            throw std::invalid_argument("prefill benchmark currently measures the whole op");
        }
        if (name == "decode-loop") {
            result.name        = "decode-loop";
            result.decode_loop = true;
        } else if (name == "production") {
            result.name              = "production";
            result.public_production = true;
        } else {
            throw std::invalid_argument("unknown prefill candidate");
        }
        return result;
    }

    if (tokens > 1) {
        if (name == "decode-loop" && scope == Scope::Full) {
            result.name                 = "decode-loop";
            result.decode_loop          = true;
            result.plan.workspace_bytes = ops::detail::sparse_moe_decode_workspace_bytes();
            return result;
        }
        if (name != "production" && name != "small-t-tiled") {
            throw std::invalid_argument("unknown T > 1 candidate");
        }
        if (name == "production") {
            result.name = "production";
        } else {
            result.name = "small-t-tiled";
        }
        result.small_t      = true;
        result.small_t_plan = ops::detail::resolve_sparse_moe_small_t_plan(
            tokens, gate_codec(profile), down_codec(profile));
        result.public_production = name == "production" && scope == Scope::Full;
        switch (scope) {
        case Scope::D1:
            result.grid  = kRouterRows * 4;
            result.block = 128;
            break;
        case Scope::D2:
            result.grid  = 1;
            result.block = std::min(tokens, 32) * 32;
            break;
        case Scope::D3:
            result.grid  = kIntermediate;
            result.block = 9 * 32;
            break;
        case Scope::D4:
            result.grid  = kHidden;
            result.block = 9 * 32;
            break;
        case Scope::Full:
            result.grid  = 0;
            result.block = 0;
            break;
        }
        return result;
    }

    if (name == "production") {
        result.name              = "production";
        result.public_production = scope == Scope::Full;
    } else if (name == "payload-control" &&
               (scope == Scope::D1 || scope == Scope::D3 || scope == Scope::D4)) {
        result.name            = "payload-control";
        result.payload_control = true;
    } else if (name == "row-cta-4w" && scope == Scope::D1) {
        result.name    = "row-cta-4w";
        result.plan.d1 = ops::detail::SparseMoeD1Schedule::RowCta4;
    } else if (name == "row-cta-8w" && scope == Scope::D1) {
        result.name    = "row-cta-8w";
        result.plan.d1 = ops::detail::SparseMoeD1Schedule::RowCta8;
    } else if (name == "serial-control" && scope == Scope::D2) {
        result.name    = "serial-control";
        result.plan.d2 = ops::detail::SparseMoeD2Schedule::SerialControl;
    } else if (name == "warp-register" && scope == Scope::D2) {
        result.name    = "warp-register";
        result.plan.d2 = ops::detail::SparseMoeD2Schedule::WarpRegister;
    } else if (name == "nine-warp-control" && scope == Scope::D3) {
        result.name    = "nine-warp-control";
        result.plan.d3 = ops::detail::SparseMoeD3Schedule::NineWarp;
    } else if (name == "balanced-eight-warp" && scope == Scope::D3) {
        result.name    = "balanced-eight-warp";
        result.plan.d3 = ops::detail::SparseMoeD3Schedule::BalancedEightWarp;
    } else if (name == "nine-warp-r1" && scope == Scope::D4) {
        result.name    = "nine-warp-r1";
        result.plan.d4 = ops::detail::SparseMoeD4Schedule::NineWarpRows1;
    } else if (name == "balanced-eight-warp-r4" && scope == Scope::D4) {
        result.name    = "balanced-eight-warp-r4";
        result.plan.d4 = ops::detail::SparseMoeD4Schedule::BalancedEightWarpRows4;
    } else {
        throw std::invalid_argument("candidate does not belong to selected scope");
    }

    switch (scope) {
    case Scope::D1:
        result.grid            = kRouterRows;
        result.block           = result.payload_control                                        ? 256
                                 : result.plan.d1 == ops::detail::SparseMoeD1Schedule::RowCta4 ? 128
                                                                                               : 256;
        result.matched_control = result.payload_control ? ""
                                 : result.name == std::string_view("row-cta-4w")
                                     ? "row-cta-8w"
                                     : "payload-control";
        break;
    case Scope::D2:
        result.grid  = 1;
        result.block = result.plan.d2 == ops::detail::SparseMoeD2Schedule::SerialControl ? 256 : 32;
        result.matched_control =
            result.name == std::string_view("warp-register") ? "serial-control" : "warp-register";
        break;
    case Scope::D3:
        result.grid            = kIntermediate;
        result.block           = result.payload_control ? 288
                                 : result.plan.d3 == ops::detail::SparseMoeD3Schedule::NineWarp ? 288
                                                                                                : 256;
        result.matched_control = result.payload_control ? ""
                                 : result.name == std::string_view("balanced-eight-warp")
                                     ? "nine-warp-control"
                                     : "payload-control";
        break;
    case Scope::D4:
        result.grid            = result.payload_control ? kHidden
                                 : result.plan.d4 == ops::detail::SparseMoeD4Schedule::NineWarpRows1
                                     ? kHidden
                                     : kHidden / 4;
        result.block           = result.payload_control ? 288
                                 : result.plan.d4 == ops::detail::SparseMoeD4Schedule::BalancedEightWarpRows4
                                     ? 256
                                     : 288;
        result.matched_control = result.payload_control ? ""
                                 : result.name == std::string_view("balanced-eight-warp-r4")
                                     ? "nine-warp-r1"
                                     : "payload-control";
        break;
    case Scope::Full:
        result.grid            = 0;
        result.block           = 0;
        result.matched_control = "";
        break;
    }
    result.plan.workspace_bytes = ops::detail::sparse_moe_decode_workspace_bytes();
    return result;
}

Payload payload_for(CodecProfile profile, Scope scope, bool payload_control, int tokens,
                    bool joint_route) {
    const auto add = [](Payload a, const Payload& b) {
        a.code += b.code;
        a.high += b.high;
        a.scale += b.scale;
        a.useful_fma += b.useful_fma;
        a.executed_fma += b.executed_fma;
        return a;
    };

    if (tokens > 1 && !joint_route) {
        Payload result = payload_for(profile, scope, payload_control, 1, false);
        result.code *= tokens;
        result.high *= tokens;
        result.scale *= tokens;
        result.useful_fma *= tokens;
        result.executed_fma *= tokens;
        return result;
    }

    if (tokens > 1) {
        Payload s1{static_cast<std::uint64_t>(kRouterRows) * kHidden * 2ULL, 0, 0,
                   static_cast<std::uint64_t>(kRouterRows) * kHidden * tokens,
                   static_cast<std::uint64_t>(kRouterRows) * kHidden * tokens};
        Payload s2{static_cast<std::uint64_t>(kRouterRows) * sizeof(float) * tokens, 0, 0, 0, 0};
        Payload s3;
        if (profile == CodecProfile::W8W8) {
            s3.code  = 18ULL * 1024 * 1024 * tokens;
            s3.scale = (9ULL * 1024 * 1024 / 8) * tokens;
        } else {
            s3.code  = 10ULL * 1024 * 1024 * tokens;
            s3.scale = (5ULL * 1024 * 1024 / 8) * tokens;
        }
        s3.useful_fma = s3.executed_fma = 18874368ULL * tokens;

        Payload s4;
        if (profile == CodecProfile::Q4Q5) {
            s4.code  = 5ULL * 1024 * 1024 * tokens;
            s4.high  = 1ULL * 1024 * 1024 * tokens;
            s4.scale = (5ULL * 1024 * 1024 / 16) * tokens;
        } else if (profile == CodecProfile::Q4Q6) {
            s4.code  = 5ULL * 1024 * 1024 * tokens;
            s4.high  = 2ULL * 1024 * 1024 * tokens;
            s4.scale = (5ULL * 1024 * 1024 / 16) * tokens;
        } else {
            s4.code  = 9ULL * 1024 * 1024 * tokens;
            s4.scale = (9ULL * 1024 * 1024 / 16) * tokens;
        }
        s4.useful_fma = s4.executed_fma = 9437184ULL * tokens;

        switch (scope) {
        case Scope::D1:
            return s1;
        case Scope::D2:
            return s2;
        case Scope::D3:
            return s3;
        case Scope::D4:
            return s4;
        case Scope::Full:
            return add(add(add(s1, s2), s3), s4);
        }
    }

    Payload d1{static_cast<std::uint64_t>(kRouterRows) * kHidden * 2ULL, 0, 0,
               static_cast<std::uint64_t>(kRouterRows) * kHidden,
               static_cast<std::uint64_t>(kRouterRows) * kHidden};
    Payload d2{static_cast<std::uint64_t>(kRouterRows) * sizeof(float), 0, 0, 0, 0};
    const bool w8_gate = profile == CodecProfile::W8W8;
    Payload d3;
    d3.code       = w8_gate ? 18ULL * 1024 * 1024 : 10ULL * 1024 * 1024;
    d3.scale      = w8_gate ? (9ULL * 1024 * 1024) / 8 : (5ULL * 1024 * 1024) / 8;
    d3.useful_fma = d3.executed_fma = 18874368ULL;

    Payload d4;
    if (profile == CodecProfile::Q4Q5) {
        d4.code  = 5ULL * 1024 * 1024;
        d4.high  = 1ULL * 1024 * 1024;
        d4.scale = (5ULL * 1024 * 1024) / 16;
    } else if (profile == CodecProfile::Q4Q6) {
        d4.code  = 5ULL * 1024 * 1024;
        d4.high  = 2ULL * 1024 * 1024;
        d4.scale = (5ULL * 1024 * 1024) / 16;
    } else {
        d4.code  = 9ULL * 1024 * 1024;
        d4.scale = (9ULL * 1024 * 1024) / 16;
    }
    d4.useful_fma = d4.executed_fma = 9437184ULL;

    Payload result;
    switch (scope) {
    case Scope::D1:
        result = d1;
        break;
    case Scope::D2:
        result = d2;
        break;
    case Scope::D3:
        result = d3;
        break;
    case Scope::D4:
        result = d4;
        break;
    case Scope::Full:
        result = add(add(add(d1, d2), d3), d4);
        break;
    }
    if (payload_control) {
        result.useful_fma   = 0;
        result.executed_fma = 0;
    }
    return result;
}

std::vector<std::pair<Scope, const char*>> matrix_candidates(CodecProfile profile, int tokens) {
    std::vector<std::pair<Scope, const char*>> result;
    if (tokens > ops::detail::kSparseMoeSmallTMax) { return {{Scope::Full, "production"}}; }
    if (tokens > 1) {
        return {{Scope::D1, "small-t-tiled"},   {Scope::D2, "small-t-tiled"},
                {Scope::D3, "small-t-tiled"},   {Scope::D4, "small-t-tiled"},
                {Scope::Full, "small-t-tiled"}, {Scope::Full, "decode-loop"},
                {Scope::Full, "production"}};
    }
    if (profile == CodecProfile::Q4Q5) {
        result.insert(result.end(), {{Scope::D1, "row-cta-4w"},
                                     {Scope::D1, "row-cta-8w"},
                                     {Scope::D1, "payload-control"},
                                     {Scope::D2, "serial-control"},
                                     {Scope::D2, "warp-register"}});
    }
    result.insert(result.end(), {{Scope::D3, "nine-warp-control"},
                                 {Scope::D3, "balanced-eight-warp"},
                                 {Scope::D3, "payload-control"},
                                 {Scope::D4, "nine-warp-r1"},
                                 {Scope::D4, "balanced-eight-warp-r4"},
                                 {Scope::D4, "payload-control"},
                                 {Scope::Full, "production"}});
    return result;
}

void print_result(const Result& result) {
    const double useful_tflops =
        2.0 * static_cast<double>(result.payload.useful_fma) / (result.warm.median_us * 1.0e6);
    std::printf("%-6s T=%d %-11s U=%-2d ov=%4.1f %-5s %-16s grid=%-4d block=%-3d "
                "cold=%8.3f us warm=%8.3f us useful=%6.1f TFLOP/s",
                codec_name(result.codec), result.tokens, distribution_name(result.distribution),
                result.unique_experts, result.adjacent_overlap, scope_name(result.candidate.scope),
                result.candidate.name, result.candidate.grid, result.candidate.block,
                result.cold.median_us, result.warm.median_us, useful_tflops);
    if (!std::isnan(result.matched_cold_us)) {
        std::printf(" control=%8.3f us", result.matched_cold_us);
    }
    std::printf("\n");
}

void attach_matched_controls(std::vector<Result>& results) {
    for (Result& result : results) {
        if (result.candidate.matched_control[0] == '\0') { continue; }
        const auto control = std::find_if(results.begin(), results.end(), [&](const Result& other) {
            return other.codec == result.codec && other.tokens == result.tokens &&
                   other.distribution == result.distribution &&
                   other.candidate.scope == result.candidate.scope &&
                   std::string_view(other.candidate.name) == result.candidate.matched_control;
        });
        if (control != results.end()) {
            result.matched_cold_us = control->cold.median_us;
            result.matched_warm_us = control->warm.median_us;
        }
    }
}

void write_csv(const std::string& path, const std::vector<Result>& results,
               const Options& options) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV output"); }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    stream << "codec,tokens,distribution,seed,unique_experts,adjacent_overlap,scope,candidate,"
              "matched_control,matched_cold_us,matched_warm_us,grid,block,"
              "cold_median_us,cold_min_us,cold_p95_us,warm_median_us,warm_min_us,warm_p95_us,"
              "code_bytes,high_bytes,scale_bytes,useful_fma,executed_fma,workspace_bytes,warmup,"
              "repeat,build_type,gpu,cuda_runtime\n";
    for (const Result& result : results) {
        stream << codec_name(result.codec) << ',' << result.tokens << ','
               << distribution_name(result.distribution) << ',' << result.seed << ','
               << result.unique_experts << ',' << result.adjacent_overlap << ','
               << scope_name(result.candidate.scope) << ',' << result.candidate.name << ','
               << result.candidate.matched_control << ',' << result.matched_cold_us << ','
               << result.matched_warm_us << ',' << result.candidate.grid << ','
               << result.candidate.block << ',' << result.cold.median_us << ','
               << result.cold.min_us << ',' << result.cold.p95_us << ',' << result.warm.median_us
               << ',' << result.warm.min_us << ',' << result.warm.p95_us << ','
               << result.payload.code << ',' << result.payload.high << ',' << result.payload.scale
               << ',' << result.payload.useful_fma << ',' << result.payload.executed_fma << ','
               << result.workspace_bytes << ',' << options.warmup << ',' << options.repeat << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << properties.name << ',' << runtime << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::vector<Result> results;
        const std::array<CodecProfile, 3> profiles = {CodecProfile::Q4Q5, CodecProfile::Q4Q6,
                                                      CodecProfile::W8W8};
        for (CodecProfile profile : profiles) {
            if (!options.matrix && profile != options.codec) { continue; }
            BenchmarkState state(profile, options.tokens, options.distribution, options.seed);
            std::vector<std::pair<Scope, const char*>> candidates;
            if (options.matrix) {
                candidates = matrix_candidates(profile, options.tokens);
            } else {
                candidates.push_back({options.scope, options.candidate.c_str()});
            }
            for (const auto& [scope, name] : candidates) {
                Candidate candidate         = make_candidate(profile, scope, name, options.tokens);
                const RoutePattern& pattern = state.route_pattern();
                const std::size_t workspace_bytes =
                    candidate.public_production ? ops::sparse_moe_workspace_bytes(options.tokens)
                    : candidate.small_t         ? candidate.small_t_plan.workspace_bytes
                                                : candidate.plan.workspace_bytes;
                Result result{profile,
                              options.tokens,
                              options.distribution,
                              options.seed,
                              pattern.unique_experts,
                              pattern.adjacent_overlap,
                              candidate,
                              measure(state, candidate, true, options.warmup, options.repeat),
                              measure(state, candidate, false, options.warmup, options.repeat),
                              payload_for(profile, scope, candidate.payload_control, options.tokens,
                                          candidate.small_t),
                              workspace_bytes};
                results.push_back(std::move(result));
            }
        }
        attach_matched_controls(results);
        for (const Result& result : results) { print_result(result); }
        write_csv(options.csv_out, results, options);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sparse_moe_bench: %s\n", error.what());
        return 1;
    }
}
