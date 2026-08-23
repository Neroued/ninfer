// Public aggregate-column benchmark for the registered Qwen3.6 embedding profiles.
//
// Every measurement is exactly one ninfer::ops::embedding call. L2 eviction
// completes before the timed interval.
#include "ninfer/ops/embedding.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kVocab             = 248320;
constexpr std::size_t kL2FlushBytes       = 256ULL << 20;
constexpr double kRtx5090SustainedReadGBs = 1674.5;

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
    std::int32_t max_proposal_k;
};

constexpr ProfileSpec profile_spec(Profile profile) {
    switch (profile) {
    case Profile::Q6D5120:
        return {"q6-d5120", QType::Q6G64_F16S, 5120, 64, 5};
    case Profile::W8D5120:
        return {"w8-d5120", QType::W8G32_F16S, 5120, 32, 5};
    case Profile::W8D2048:
        return {"w8-d2048", QType::W8G32_F16S, 2048, 32, 15};
    case Profile::Fp8D5120:
        return {"fp8-d5120", QType::FP8_E4M3FN_ROW_BF16S, 5120, 5120, 5};
    }
    throw std::logic_error("unknown embedding profile");
}

Profile parse_profile(const char* raw) {
    if (!std::strcmp(raw, "q6-d5120")) return Profile::Q6D5120;
    if (!std::strcmp(raw, "w8-d5120")) return Profile::W8D5120;
    if (!std::strcmp(raw, "w8-d2048")) return Profile::W8D2048;
    if (!std::strcmp(raw, "fp8-d5120")) return Profile::Fp8D5120;
    throw std::invalid_argument("--profile must be q6-d5120, w8-d5120, w8-d2048, or fp8-d5120");
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

DeviceBuffer make_ids(std::int32_t t) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(t));
    for (std::int32_t index = 0; index < t; ++index) {
        host[static_cast<std::size_t>(index)] = (index * 9973 + 12345) % kVocab;
    }
    DeviceBuffer device(host.size() * sizeof(std::int32_t));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

std::vector<std::int32_t> parse_tokens(const char* raw) {
    std::set<std::int32_t> unique;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const long long value  = std::stoll(item);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("tokens must be positive int32 values");
        }
        unique.insert(static_cast<std::int32_t>(value));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (unique.empty()) throw std::invalid_argument("tokens must not be empty");
    return {unique.begin(), unique.end()};
}

std::vector<std::int32_t> production_tokens(const ProfileSpec& spec) {
    std::set<std::int32_t> values;
    for (std::int32_t batch = 1; batch <= 8; ++batch) values.insert(batch);
    for (std::int32_t batch = 1; batch <= 8; ++batch) {
        for (std::int32_t k = 1; k <= spec.max_proposal_k; ++k) { values.insert(batch * (k + 1)); }
    }
    return {values.begin(), values.end()};
}

ColdTiming measure_point(const ProfileSpec& spec, const Weight& table, DeviceBuffer& ids,
                         DeviceBuffer& out, DeviceBuffer& flush, std::int32_t t,
                         cudaStream_t stream, int warmup, int repeat) {
    Tensor all_ids(ids.p, DType::I32, {t});
    Tensor all_out(out.p, DType::BF16, {spec.d, t});
    const auto launch = [&](cudaStream_t selected_stream) {
        ops::embedding(all_ids, table, all_out, selected_stream);
    };
    return measure_cold_launch(launch, flush, stream, warmup, repeat);
}

double logical_bytes_per_column(const ProfileSpec& spec, const PackedLayout& layout) {
    const double groups       = static_cast<double>(layout.padded_d / spec.group);
    const double weight_bytes = spec.qtype == QType::Q6G64_F16S
                                    ? groups * static_cast<double>(32 + 16 + 2)
                                    : static_cast<double>(layout.padded_d) + groups * 2.0;
    return weight_bytes + static_cast<double>(spec.d) * 2.0 + sizeof(std::int32_t);
}

void run_profile(Profile profile, const std::vector<std::int32_t>* requested_tokens, int warmup,
                 int repeat, bool csv) {
    const ProfileSpec spec = profile_spec(profile);
    const std::vector<std::int32_t> tokens =
        requested_tokens == nullptr ? production_tokens(spec) : *requested_tokens;
    const std::int32_t max_t  = *std::max_element(tokens.begin(), tokens.end());
    const PackedLayout layout = packed_layout(spec);

    DeviceBuffer payload(static_cast<std::size_t>(layout.payload_bytes));
    DeviceBuffer ids = make_ids(max_t);
    DeviceBuffer out(static_cast<std::size_t>(spec.d) * static_cast<std::size_t>(max_t) *
                     sizeof(std::uint16_t));
    DeviceBuffer flush(kL2FlushBytes);
    payload.fill(0x3c);
    out.fill();
    const Weight table = make_weight(spec, layout, payload.p);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const std::int32_t t : tokens) {
        const ColdTiming result =
            measure_point(spec, table, ids, out, flush, t, stream, warmup, repeat);
        const double logical_bytes = logical_bytes_per_column(spec, layout) * t;
        const double effective_gbs = logical_bytes / (result.median_us * 1.0e-6) / 1.0e9;
        const double read_pct      = effective_gbs / kRtx5090SustainedReadGBs * 100.0;
        if (csv) {
            std::printf("%s,%d,%.3f,%.3f,%.3f,%.3f\n", spec.name, t, result.median_us,
                        result.p95_us, effective_gbs, read_pct);
        } else {
            std::printf("embedding %-10s T=%3d  median=%8.3f us  p95=%8.3f us  "
                        "effective=%8.1f GB/s  READ=%6.2f%%\n",
                        spec.name, t, result.median_us, result.p95_us, effective_gbs, read_pct);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<Profile> profiles;
    std::vector<std::int32_t> requested_tokens;
    bool has_requested_tokens = false;
    bool csv                  = false;
    int warmup                = 10;
    int repeat                = 61;
    for (int index = 1; index < argc; ++index) {
        if (!std::strcmp(argv[index], "--profile") && index + 1 < argc) {
            profiles.push_back(parse_profile(argv[++index]));
        } else if (!std::strcmp(argv[index], "--tokens") && index + 1 < argc) {
            requested_tokens     = parse_tokens(argv[++index]);
            has_requested_tokens = true;
        } else if (!std::strcmp(argv[index], "--warmup") && index + 1 < argc) {
            warmup = std::stoi(argv[++index]);
        } else if (!std::strcmp(argv[index], "--repeat") && index + 1 < argc) {
            repeat = std::stoi(argv[++index]);
        } else if (!std::strcmp(argv[index], "--csv")) {
            csv = true;
        } else {
            throw std::invalid_argument("unknown or incomplete embedding benchmark argument");
        }
    }
    if (profiles.empty()) {
        profiles = {Profile::Q6D5120, Profile::W8D5120, Profile::W8D2048, Profile::Fp8D5120};
    }
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    std::sort(profiles.begin(), profiles.end());
    profiles.erase(std::unique(profiles.begin(), profiles.end()), profiles.end());

    if (csv) { std::printf("profile,T,median_us,p95_us,effective_gbs,sustained_read_pct\n"); }
    for (const Profile profile : profiles) {
        run_profile(profile, has_requested_tokens ? &requested_tokens : nullptr, warmup, repeat,
                    csv);
    }
    return 0;
}
