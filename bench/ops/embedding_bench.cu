// Public embedding benchmark: normal IDs and mask-heavy draft blocks at explicit matrix extents.
#include "ninfer/ops/embedding.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kVocab       = 248320;
constexpr std::size_t kL2FlushBytes = 256ULL << 20;

enum class Profile {
    Q6D5120,
    W8D5120,
    W8D2048,
    Fp8D5120,
};

struct ProfileSpec {
    const char* name;
    QType qtype;
    std::int32_t d;
    std::int32_t group;
};

constexpr ProfileSpec profile_spec(Profile profile) {
    switch (profile) {
    case Profile::Q6D5120:
        return {"q6-d5120", QType::Q6G64_F16S, 5120, 64};
    case Profile::W8D5120:
        return {"w8-d5120", QType::W8G32_F16S, 5120, 32};
    case Profile::W8D2048:
        return {"w8-d2048", QType::W8G32_F16S, 2048, 32};
    case Profile::Fp8D5120:
        return {"fp8-d5120", QType::FP8_E4M3FN_ROW_BF16S, 5120, 5120};
    }
    throw std::logic_error("unknown embedding profile");
}

Profile parse_profile(const char* raw) {
    if (!std::strcmp(raw, "q6-d5120")) return Profile::Q6D5120;
    if (!std::strcmp(raw, "w8-d5120")) return Profile::W8D5120;
    if (!std::strcmp(raw, "w8-d2048")) return Profile::W8D2048;
    if (!std::strcmp(raw, "fp8-d5120")) return Profile::Fp8D5120;
    throw std::invalid_argument("--format must be q6-d5120, w8-d5120, w8-d2048, or fp8-d5120");
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

struct PackedLayout {
    std::int32_t padded_d            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t high_plane_offset  = 0;
    std::uint64_t high_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t payload_bytes      = 0;
};

PackedLayout packed_layout(const ProfileSpec& spec) {
    PackedLayout layout;
    layout.padded_d            = static_cast<std::int32_t>(align_up(spec.d, 128));
    const std::uint64_t groups = static_cast<std::uint64_t>(layout.padded_d / spec.group);
    if (spec.qtype == QType::Q6G64_F16S) {
        layout.code_plane_bytes  = static_cast<std::uint64_t>(kVocab) * groups * 32;
        layout.high_plane_offset = align_up(layout.code_plane_bytes, 256);
        layout.high_plane_bytes  = static_cast<std::uint64_t>(kVocab) * groups * 16;
        layout.scale_plane_offset =
            layout.high_plane_offset + align_up(layout.high_plane_bytes, 256);
    } else if (spec.qtype == QType::W8G32_F16S) {
        layout.code_plane_bytes =
            static_cast<std::uint64_t>(kVocab) * groups * static_cast<std::uint64_t>(spec.group);
        layout.scale_plane_offset = align_up(layout.code_plane_bytes, 256);
    } else {
        layout.code_plane_bytes   = static_cast<std::uint64_t>(kVocab) * spec.d;
        layout.scale_plane_offset = align_up(layout.code_plane_bytes, 256);
    }
    layout.scale_plane_bytes = static_cast<std::uint64_t>(kVocab) * groups * 2;
    layout.payload_bytes     = layout.scale_plane_offset + layout.scale_plane_bytes;
    return layout;
}

Weight make_weight(const ProfileSpec& spec, const PackedLayout& layout, void* payload) {
    auto* bytes = static_cast<std::uint8_t*>(payload);
    Weight table{};
    table.payload          = payload;
    table.payload_bytes    = layout.payload_bytes;
    table.high_plane_bytes = layout.high_plane_bytes;
    table.qtype            = spec.qtype;
    table.layout =
        spec.qtype == QType::FP8_E4M3FN_ROW_BF16S ? QuantLayout::RowScale : QuantLayout::RowSplit;
    table.scale_dtype     = spec.qtype == QType::FP8_E4M3FN_ROW_BF16S ? DType::BF16 : DType::FP16;
    table.group_size      = spec.group;
    table.shape[0]        = kVocab;
    table.shape[1]        = spec.d;
    table.padded_shape[0] = kVocab;
    table.padded_shape[1] = layout.padded_d;
    table.ndim            = 2;
    table.qdata           = payload;
    table.qhigh  = spec.qtype == QType::Q6G64_F16S ? bytes + layout.high_plane_offset : nullptr;
    table.scales = bytes + layout.scale_plane_offset;
    table.n      = kVocab;
    table.k      = spec.d;
    table.group  = spec.group;
    if (spec.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        table.scale_ne[0] = kVocab;
        table.scale_nb[0] = 2;
        table.scale_nb[1] = static_cast<std::int64_t>(kVocab) * 2;
        table.scale_nb[2] = table.scale_nb[1];
        table.scale_nb[3] = table.scale_nb[1];
    }
    return table;
}

__device__ std::uint32_t fixture_hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

// Mode 0/1: Q6 low/high planes; 2: W8; 3: finite E4M3FN. All codes obey the numeric format.
__global__ void initialize_codes(std::uint32_t* words, std::size_t size, int mode) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const int bits   = mode == 0 ? 4 : mode == 1 ? 2 : 8;
    const int values = 32 / bits;
    unsigned packed  = 0;
    for (int j = 0; j < values; ++j) {
        const auto h = fixture_hash(static_cast<std::uint32_t>(i * values + j));
        unsigned code;
        if (mode < 2) {
            const int q = static_cast<int>(h % 63) - 31;
            code        = mode == 0 ? (q & 15) : ((q & 63) >> 4);
        } else if (mode == 2) {
            code = static_cast<std::uint8_t>(static_cast<int>(h % 255) - 127);
        } else {
            code = h & 255;
            if ((code & 127) == 127) --code;
        }
        packed |= code << (j * bits);
    }
    words[i] = packed;
}

__global__ void initialize_scales(std::uint16_t* scales, std::size_t count, bool bf16) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value =
        0.001f * (1.0f + (fixture_hash(static_cast<std::uint32_t>(i)) % 13) * 0.0625f);
    scales[i] = bf16 ? __bfloat16_as_ushort(__float2bfloat16_rn(value))
                     : __half_as_ushort(__float2half_rn(value));
}

void initialize_payload(const ProfileSpec& spec, const PackedLayout& layout,
                        DeviceBuffer& payload) {
    auto* base      = static_cast<std::uint8_t*>(payload.p);
    const int mode  = spec.qtype == QType::Q6G64_F16S ? 0 : spec.qtype == QType::W8G32_F16S ? 2 : 3;
    const auto fill = [&](std::size_t offset, std::size_t bytes, int selected) {
        const auto words = bytes / 4;
        initialize_codes<<<(words + 255) / 256, 256>>>(
            reinterpret_cast<std::uint32_t*>(base + offset), words, selected);
        CUDA_CHECK(cudaGetLastError());
    };
    fill(0, layout.code_plane_bytes, mode);
    if (layout.high_plane_bytes) fill(layout.high_plane_offset, layout.high_plane_bytes, 1);
    const auto scales = layout.scale_plane_bytes / 2;
    initialize_scales<<<(scales + 255) / 256, 256>>>(
        reinterpret_cast<std::uint16_t*>(base + layout.scale_plane_offset), scales, mode == 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct Options {
    std::vector<Profile> profiles;
    std::vector<int> tokens;
    int warmup = 10, repeat = 100, graph_calls = 1, block_width = 8;
    std::string execution = "graph", cache = "cold", pattern = "normal", csv;
    bool profile = false;
};

int parse_integer(const std::string& s, int low, int high) {
    std::size_t end;
    const int value = std::stoi(s, &end);
    if (end != s.size() || value < low || value > high)
        throw std::invalid_argument("integer outside supported range");
    return value;
}

std::vector<int> parse_tokens(const std::string& text) {
    std::set<int> result;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        result.insert(parse_integer(text.substr(start, end - start), 1, 262144));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (result.empty()) throw std::invalid_argument("empty tokens list");
    return {result.begin(), result.end()};
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            o.profile = true;
            continue;
        }
        if (arg == "--help") {
            std::puts(
                "usage: ninfer_embedding_bench [--format q6-d5120|w8-d5120|w8-d2048|fp8-d5120] "
                "[--tokens T,...] [--id-pattern normal|masked] [--block-width 2..16] [--execution "
                "eager|graph] [--cache cold|warm] [--graph-calls 1..64] [--warmup N] [--repeat N] "
                "[--csv-out PATH] [--profile]");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--format")
            o.profiles.push_back(parse_profile(value.c_str()));
        else if (arg == "--tokens")
            o.tokens = parse_tokens(value);
        else if (arg == "--id-pattern")
            o.pattern = value;
        else if (arg == "--block-width")
            o.block_width = parse_integer(value, 2, 16);
        else if (arg == "--execution")
            o.execution = value;
        else if (arg == "--cache")
            o.cache = value;
        else if (arg == "--graph-calls")
            o.graph_calls = parse_integer(value, 1, 64);
        else if (arg == "--warmup")
            o.warmup = parse_integer(value, 0, 10000);
        else if (arg == "--repeat")
            o.repeat = parse_integer(value, 1, 10000);
        else if (arg == "--csv-out")
            o.csv = value;
        else
            throw std::invalid_argument("unknown option " + arg);
    }
    if (o.profiles.empty())
        o.profiles = {Profile::Q6D5120, Profile::W8D5120, Profile::W8D2048, Profile::Fp8D5120};
    if (o.tokens.empty())
        for (int t = 1; t <= 128; ++t) o.tokens.push_back(t);
    if ((o.execution != "eager" && o.execution != "graph") ||
        (o.cache != "cold" && o.cache != "warm") ||
        (o.pattern != "normal" && o.pattern != "masked"))
        throw std::invalid_argument("invalid execution or ID pattern");
    if (o.graph_calls > 1 && (o.execution != "graph" || o.profile))
        throw std::invalid_argument("bundles require Graph and no profiling");
    if (o.profile && (o.profiles.size() != 1 || o.tokens.size() != 1))
        throw std::invalid_argument("profile one format and one shape");
    return o;
}

double logical_bytes_per_column(const ProfileSpec& spec, const PackedLayout& layout) {
    const double encoded =
        spec.qtype == QType::Q6G64_F16S ? layout.padded_d * 0.75 : layout.padded_d;
    return encoded + (layout.padded_d / spec.group) * 2 + spec.d * 2 + 4;
}

void run_profile(Profile profile, const Options& o, std::ofstream& csv) {
    const auto spec   = profile_spec(profile);
    const auto layout = packed_layout(spec);
    DeviceBuffer payload(layout.payload_bytes);
    initialize_payload(spec, layout, payload);
    const auto weight = make_weight(spec, layout, payload.p);
    const int max_t   = *std::max_element(o.tokens.begin(), o.tokens.end());
    std::vector<int> host_ids(max_t);
    const int mask = spec.d == 2048 ? 248077 : 248070;
    for (int i = 0; i < max_t; ++i) {
        const int normal = (static_cast<std::int64_t>(i) * 9973 + 12345) % 248077;
        host_ids[i]      = o.pattern == "masked" && i % o.block_width ? mask : normal;
    }
    DeviceBuffer ids(static_cast<std::size_t>(max_t) * 4),
        output(static_cast<std::size_t>(max_t) * spec.d * 2), flush(kL2FlushBytes);
    ids.copy_from_host(host_ids.data(), ids.bytes);
    DeviceContext device;
    for (int t : o.tokens) {
        Tensor input(ids.p, DType::I32, {t}), out(output.p, DType::BF16, {spec.d, t});
        const auto launch = [&](cudaStream_t stream) {
            ops::embedding(input, weight, out, stream);
        };
        TimedGraph graph;
        if (o.execution == "graph") {
            launch(device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            graph.capture(device.stream, [&](cudaStream_t stream) {
                for (int i = 0; i < o.graph_calls; ++i) launch(stream);
            });
        }
        if (o.profile) {
            for (int i = 0; i < o.warmup; ++i) {
                if (o.execution == "graph")
                    graph.launch(device.stream);
                else
                    launch(device.stream);
            }
            if (o.cache == "cold") flush_l2(flush, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            CUDA_CHECK(cudaProfilerStart());
            if (o.execution == "graph")
                graph.launch(device.stream);
            else
                launch(device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            CUDA_CHECK(cudaProfilerStop());
            return;
        }
        ColdTiming result;
        if (o.execution == "graph")
            result = o.cache == "cold"
                         ? measure_cold_graph(graph, flush, device.stream, o.warmup, o.repeat)
                         : measure_graph(graph, device.stream, o.warmup, o.repeat);
        else
            result = o.cache == "cold"
                         ? measure_cold_launch(launch, flush, device.stream, o.warmup, o.repeat)
                         : measure_launch(launch, device.stream, o.warmup, o.repeat);
        result.median_us /= o.graph_calls;
        result.min_us /= o.graph_calls;
        result.p95_us /= o.graph_calls;
        const auto unique          = std::set<int>(host_ids.begin(), host_ids.begin() + t).size();
        const auto nodes           = o.execution == "graph" ? graph.nodes() : 0;
        const double logical_bytes = logical_bytes_per_column(spec, layout) * t;
        std::printf("embedding %s T=%d ids=%s W=%d unique=%zu execution=%s cache=%s nodes=%zu "
                    "calls=%d scratch=0 median=%.3f min=%.3f p95=%.3f us\n",
                    spec.name, t, o.pattern.c_str(), o.block_width, unique, o.execution.c_str(),
                    o.cache.c_str(), nodes, o.graph_calls, result.median_us, result.min_us,
                    result.p95_us);
        if (csv)
            csv << spec.name << ',' << t << ',' << o.pattern << ',' << o.block_width << ',' << mask
                << ',' << unique << ',' << o.execution << ',' << o.cache << ',' << nodes << ','
                << o.graph_calls << ",0," << logical_bytes << ',' << result.median_us << ','
                << result.min_us << ',' << result.p95_us << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        int devices     = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU=%s CUDART=%d\n", prop.name, CUDART_VERSION);
        std::ofstream csv;
        if (!o.csv.empty()) {
            csv.open(o.csv);
            if (!csv) throw std::runtime_error("cannot open CSV output");
            csv << "format,T,id_pattern,block_width,mask_id,unique_ids,execution,cache,graph_nodes,"
                   "graph_calls,workspace_bytes,logical_bytes,median_us,min_us,p95_us\n";
        }
        for (const auto profile : o.profiles) run_profile(profile, o, csv);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "embedding_bench: %s\n", e.what());
        return 1;
    }
}
