// Public pure Linear benchmark with explicit cache conditioning.
//
// Examples:
//   ./build/rtx5090-dev/bench/ninfer_linear_bench --qtype q4 --n 4096 --k 5120 --t 8
//   ./build/rtx5090-dev/bench/ninfer_linear_bench --qtype q4 --n 4096 --k 5120 --sweep 1:32:1
//   ./build/rtx5090-dev/bench/ninfer_linear_bench --qtype fp8 --policy a8 --n 14336 --k 5120 --t 1
//   ./build/rtx5090-dev/bench/ninfer_linear_bench --suite qwen3_6_27b
//   ncu --profile-from-start off ./build/rtx5090-dev/bench/ninfer_linear_bench \
//       --qtype q4 --n 4096 --k 5120 --t 8 --profile

#include "common/roofline.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "core/buffer.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;
using ninfer::ops::LinearPolicy;

namespace {

constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;
constexpr int kDefaultWarmup               = 3;
constexpr int kDefaultRepeat               = 20;

enum class TClass : std::uint8_t {
    Continuous,
    VisionStep4,
};

struct SuiteEntry {
    const char* label;
    QType qtype;
    std::int32_t n;
    std::int32_t k;
    TClass t_class;
};

constexpr SuiteEntry kQwen27bEntries[] = {
    {"27b.output_head", QType::Q6_G64_FP16, 248320, 5120, TClass::Continuous},
    {"27b.draft_head", QType::Q4_G64_FP16, 131072, 5120, TClass::Continuous},
    {"27b.gdn_output_gate", QType::Q5_G64_FP16, 6144, 5120, TClass::Continuous},
    {"27b.mtp_input", QType::Q8_G32_FP16, 5120, 10240, TClass::Continuous},
    {"27b.mtp_attention", QType::Q8_G32_FP16, 14336, 5120, TClass::Continuous},
    {"27b.mtp_gate_up", QType::Q8_G32_FP16, 34816, 5120, TClass::Continuous},
    {"27b.mtp_down", QType::Q8_G32_FP16, 5120, 17408, TClass::Continuous},
    {"27b.vision_patch", QType::Q6_G64_FP16, 1152, 1536, TClass::VisionStep4},
    {"27b.vision_qkv", QType::Q4_G64_FP16, 3456, 1152, TClass::VisionStep4},
    {"27b.vision_attn_out", QType::Q5_G64_FP16, 1152, 1152, TClass::VisionStep4},
    {"27b.vision_fc1", QType::Q4_G64_FP16, 4304, 1152, TClass::VisionStep4},
    {"27b.vision_fc2", QType::Q5_G64_FP16, 1152, 4304, TClass::VisionStep4},
    {"27b.vision_merger_fc1", QType::Q8_G32_FP16, 4608, 4608, TClass::VisionStep4},
    {"27b.vision_merger_fc2", QType::Q8_G32_FP16, 5120, 4608, TClass::VisionStep4},
};

constexpr SuiteEntry kQwen35bEntries[] = {
    {"35b.output_head", QType::Q6_G64_FP16, 248320, 2048, TClass::Continuous},
    {"35b.draft_head", QType::Q4_G64_FP16, 131072, 2048, TClass::Continuous},
    {"35b.mtp_projection", QType::Q8_G32_FP16, 2048, 4096, TClass::Continuous},
    {"35b.dflash_feature", QType::Q8_G32_FP16, 2048, 16384, TClass::Continuous},
    {"35b.vision_patch", QType::Q6_G64_FP16, 1152, 1536, TClass::VisionStep4},
    {"35b.vision_qkv", QType::Q4_G64_FP16, 3456, 1152, TClass::VisionStep4},
    {"35b.vision_attn_out", QType::Q5_G64_FP16, 1152, 1152, TClass::VisionStep4},
    {"35b.vision_fc1", QType::Q4_G64_FP16, 4304, 1152, TClass::VisionStep4},
    {"35b.vision_fc2", QType::Q5_G64_FP16, 1152, 4304, TClass::VisionStep4},
    {"35b.vision_merger_fc1", QType::Q8_G32_FP16, 4608, 4608, TClass::VisionStep4},
    {"35b.vision_merger_fc2", QType::Q8_G32_FP16, 2048, 4608, TClass::VisionStep4},
};

struct Options {
    bool have_qtype     = false;
    bool have_n         = false;
    bool have_k         = false;
    bool have_t         = false;
    bool have_sweep     = false;
    bool have_tokens    = false;
    bool have_suite     = false;
    bool profile        = false;
    bool graph          = false;
    bool cold           = true;
    bool body_timer     = false;
    int graph_calls     = 1;
    QType qtype         = QType::Q4_G64_FP16;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::int32_t n      = 0;
    std::int32_t k      = 0;
    std::int32_t t      = 0;
    bench::measurement::Sweep sweep;
    std::vector<std::int32_t> tokens;
    std::string suite;
    int warmup                = kDefaultWarmup;
    int repeat                = kDefaultRepeat;
    std::uint64_t flush_bytes = kDefaultFlushBytes;
    std::string csv_out;
};

struct BenchPoint {
    QType qtype;
    LinearPolicy policy;
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    std::vector<std::string> labels;
    bool sweep_point = false;
};

struct PointGroup {
    QType qtype;
    LinearPolicy policy;
    std::int32_t n;
    std::int32_t k;
    std::vector<BenchPoint> points;
};

struct Result {
    const char* timer_scope = "eager-launch";
    const char* cache       = "cold";
    const char* execution   = "eager";
    std::size_t graph_nodes = 0;
    int graph_calls         = 1;
    std::string labels;
    const char* qtype_name               = "";
    const char* policy_name              = "";
    std::int32_t n                       = 0;
    std::int32_t k                       = 0;
    std::int32_t t                       = 0;
    std::uint64_t weight_bytes           = 0;
    std::uint64_t x_bytes                = 0;
    std::uint64_t out_bytes              = 0;
    std::uint64_t model_bytes            = 0;
    double useful_flops                  = 0.0;
    double median_us                     = 0.0;
    double min_us                        = 0.0;
    double p95_us                        = 0.0;
    double effective_gbs                 = 0.0;
    double dram_spec_pct                 = 0.0;
    double sustained_read_pct            = 0.0;
    double useful_tflops                 = 0.0;
    const char* tensor_profile           = "";
    double tensor_peak_tflops            = std::numeric_limits<double>::quiet_NaN();
    double tensor_peak_pct               = std::numeric_limits<double>::quiet_NaN();
    double memory_floor_us               = 0.0;
    double memory_floor_pct              = 0.0;
    double t1_linear_extrapolation       = std::numeric_limits<double>::quiet_NaN();
    double delta_pct                     = std::numeric_limits<double>::quiet_NaN();
    int warmup                           = 0;
    int repeat                           = 0;
    std::uint64_t flush_bytes            = 0;
    std::size_t workspace_capacity_bytes = 0;
};

struct LinearBenchWeight {
    DeviceBuffer storage;
    Weight weight{};
    std::uint64_t model_bytes = 0;

    [[nodiscard]] std::uint64_t model_weight_bytes() const noexcept { return model_bytes; }
};

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        throw std::overflow_error(std::string(label) + " overflows uint64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflows uint64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) { throw std::invalid_argument("alignment must be positive"); }
    return checked_mul((checked_add(value, alignment - 1, "aligned size") / alignment), alignment,
                       "aligned size");
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
    }
    return out;
}

const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return "Q4";
    case QType::Q5_G64_FP16:
        return "Q5";
    case QType::Q6_G64_FP16:
        return "Q6";
    case QType::Q8_G32_FP16:
        return "Q8";
    case QType::BF16:
        return "BF16";
    case QType::NVFP4:
        return "NVFP4";
    case QType::FP8_E4M3FN_ROW_BF16:
        return "FP8";
    default:
        break;
    }
    throw std::invalid_argument("unsupported Linear benchmark qtype");
}

const char* policy_name(LinearPolicy policy) {
    if (policy == LinearPolicy::A16Only) { return "A16"; }
    if (policy == LinearPolicy::AllowA8) { return "A8"; }
    if (policy == LinearPolicy::AllowA4) { return "A4"; }
    throw std::invalid_argument("unsupported Linear benchmark policy");
}

QType parse_qtype(std::string_view text) {
    const std::string value = lower(text);
    if (value == "q4" || value == "q4_g64_fp16") { return QType::Q4_G64_FP16; }
    if (value == "q5" || value == "q5_g64_fp16") { return QType::Q5_G64_FP16; }
    if (value == "q6" || value == "q6_g64_fp16") { return QType::Q6_G64_FP16; }
    if (value == "q8" || value == "q8_g32_fp16") { return QType::Q8_G32_FP16; }
    if (value == "bf16") { return QType::BF16; }
    if (value == "nvfp4") { return QType::NVFP4; }
    if (value == "fp8" || value == "fp8_e4m3fn_row_bf16") { return QType::FP8_E4M3FN_ROW_BF16; }
    throw std::invalid_argument("unknown qtype: " + std::string(text));
}

LinearPolicy parse_policy(std::string_view text) {
    const std::string value = lower(text);
    if (value == "a16" || value == "a16only") { return LinearPolicy::A16Only; }
    if (value == "a8" || value == "allowa8") { return LinearPolicy::AllowA8; }
    if (value == "a4" || value == "allowa4") { return LinearPolicy::AllowA4; }
    throw std::invalid_argument("Linear benchmark policy must be a16, a8, or a4");
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --qtype Q4|Q5|Q6|Q8|BF16|NVFP4|FP8 --n N --k K --t T [options]\n"
                 "  %s --qtype Q4|Q5|Q6|Q8|BF16|NVFP4|FP8 --n N --k K --sweep START:END[:STEP] "
                 "[options]\n"
                 "  %s --qtype Q4|Q5|Q6|Q8|BF16|NVFP4|FP8 --n N --k K --tokens T1,T2,... "
                 "[options]\n"
                 "  %s --suite qwen3_6_27b|qwen3_6_35b_a3b|all [options]\n\n"
                 "Options:\n"
                 "  --policy a16|a8|a4 Activation-compute policy (default a16).\n"
                 "  --execution MODE   eager (default) or graph; time the complete Op.\n"
                 "  --cache MODE       cold (default) or warm; report separately.\n"
                 "  --graph-calls N    Calls per timed graph (1..64, default 1); report per call.\n"
                 "  --graph-timer MODE launch (default) or body; body brackets captured GPU work.\n"
                 "  --profile          Capture exactly one post-warmup public Linear call.\n"
                 "  --warmup N         Warmup calls per point (default %d).\n"
                 "  --repeat N         Measured samples per point (default %d).\n"
                 "  --flush-mib N      L2 eviction buffer size (default 256 MiB).\n"
                 "  --csv-out PATH     Write all ordinary measurement rows as CSV.\n"
                 "  -h, --help         Show this text.\n",
                 argv0, argv0, argv0, argv0, kDefaultWarmup, kDefaultRepeat);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++i];
        };
        if (arg == "--qtype") {
            opt.qtype      = parse_qtype(next("qtype"));
            opt.have_qtype = true;
        } else if (arg == "--policy") {
            opt.policy = parse_policy(next("policy"));
        } else if (arg == "--n") {
            opt.n      = bench::measurement::integer(next("N"));
            opt.have_n = true;
        } else if (arg == "--k") {
            opt.k      = bench::measurement::integer(next("K"));
            opt.have_k = true;
        } else if (arg == "--t") {
            opt.t      = bench::measurement::integer(next("T"));
            opt.have_t = true;
        } else if (arg == "--sweep") {
            opt.sweep      = bench::measurement::parse_sweep(next("sweep"));
            opt.have_sweep = true;
        } else if (arg == "--tokens") {
            opt.tokens      = bench::measurement::integers(next("tokens"));
            opt.have_tokens = true;
        } else if (arg == "--suite") {
            opt.suite      = lower(next("suite"));
            opt.have_suite = true;
        } else if (arg == "--execution") {
            const std::string_view mode(next("execution"));
            if (mode != "eager" && mode != "graph")
                throw std::invalid_argument("execution must be eager or graph");
            opt.graph = mode == "graph";
        } else if (arg == "--graph-calls") {
            opt.graph_calls = bench::measurement::integer(next("graph-calls"), 0);
        } else if (arg == "--graph-timer") {
            const std::string_view mode(next("graph-timer"));
            if (mode != "launch" && mode != "body")
                throw std::invalid_argument("graph-timer must be launch or body");
            opt.body_timer = mode == "body";
        } else if (arg == "--cache") {
            const std::string_view mode(next("cache"));
            if (mode != "cold" && mode != "warm")
                throw std::invalid_argument("cache must be cold or warm");
            opt.cold = mode == "cold";
        } else if (arg == "--profile") {
            opt.profile = true;
        } else if (arg == "--warmup") {
            opt.warmup = bench::measurement::integer(next("warmup"), 0);
        } else if (arg == "--repeat") {
            opt.repeat = bench::measurement::integer(next("repeat"), 0);
        } else if (arg == "--flush-mib") {
            opt.flush_bytes =
                checked_mul(bench::measurement::integer<std::uint64_t>(next("flush-mib"), 0),
                            1ULL << 20, "flush bytes");
        } else if (arg == "--csv-out") {
            opt.csv_out = next("csv output path");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (argc == 1) { throw std::invalid_argument("select one exact point, sweep, or suite"); }
    if (opt.body_timer && (!opt.graph || opt.profile))
        throw std::invalid_argument("body timer requires graph timing without --profile");
    if (opt.graph_calls < 1 || opt.graph_calls > 64 ||
        (opt.graph_calls != 1 && (!opt.graph || opt.profile)))
        throw std::invalid_argument(
            "graph-calls must be 1..64; multiple calls require graph timing without --profile");
    if (opt.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (opt.flush_bytes == 0) { throw std::invalid_argument("--flush-mib must be positive"); }
    if (int(opt.have_t) + int(opt.have_sweep) + int(opt.have_tokens) > 1) {
        throw std::invalid_argument("--t, --sweep and --tokens are mutually exclusive");
    }
    if (opt.have_suite) {
        if (opt.suite != "qwen3_6_27b" && opt.suite != "qwen3_6_35b_a3b" && opt.suite != "all") {
            throw std::invalid_argument("--suite must be qwen3_6_27b, qwen3_6_35b_a3b, or all");
        }
        if (opt.have_qtype || opt.have_n || opt.have_k || opt.have_t || opt.have_sweep ||
            opt.have_tokens) {
            throw std::invalid_argument("--suite cannot be combined with an explicit point");
        }
        if (opt.profile) { throw std::invalid_argument("--profile does not accept a suite"); }
    } else {
        if (!opt.have_qtype || !opt.have_n || !opt.have_k) {
            throw std::invalid_argument("explicit mode requires --qtype, --n, and --k");
        }
        if (!opt.have_t && !opt.have_sweep && !opt.have_tokens) {
            throw std::invalid_argument("explicit mode requires --t, --sweep or --tokens");
        }
        if (opt.profile && !opt.have_t) {
            throw std::invalid_argument("--profile requires one exact --t");
        }
    }
    if (opt.profile && !opt.csv_out.empty()) {
        throw std::invalid_argument("--profile does not write timing CSV");
    }
    return opt;
}

const std::vector<std::int32_t>& default_t_values(TClass t_class) {
    static const std::vector<std::int32_t> continuous{1, 16, 128, 1024};
    static const std::vector<std::int32_t> vision{4, 128, 1024};
    return t_class == TClass::Continuous ? continuous : vision;
}

bool same_point(const BenchPoint& a, const BenchPoint& b) {
    return a.qtype == b.qtype && a.policy == b.policy && a.n == b.n && a.k == b.k && a.t == b.t;
}

void append_point(std::vector<BenchPoint>& points, BenchPoint point) {
    for (BenchPoint& existing : points) {
        if (!same_point(existing, point)) { continue; }
        for (std::string& label : point.labels) {
            if (std::find(existing.labels.begin(), existing.labels.end(), label) ==
                existing.labels.end()) {
                existing.labels.push_back(std::move(label));
            }
        }
        existing.sweep_point = existing.sweep_point || point.sweep_point;
        return;
    }
    points.push_back(std::move(point));
}

template <std::size_t N>
void append_suite(std::vector<BenchPoint>& points, const SuiteEntry (&entries)[N]) {
    for (const SuiteEntry& entry : entries) {
        for (const std::int32_t t : default_t_values(entry.t_class)) {
            append_point(
                points,
                {entry.qtype, LinearPolicy::A16Only, entry.n, entry.k, t, {entry.label}, false});
        }
    }
}

std::vector<BenchPoint> expand_points(const Options& opt) {
    std::vector<BenchPoint> points;
    if (opt.have_suite) {
        if (opt.suite == "qwen3_6_27b" || opt.suite == "all") {
            append_suite(points, kQwen27bEntries);
        }
        if (opt.suite == "qwen3_6_35b_a3b" || opt.suite == "all") {
            append_suite(points, kQwen35bEntries);
        }
        return points;
    }
    if (opt.have_t) {
        points.push_back({opt.qtype, opt.policy, opt.n, opt.k, opt.t, {"explicit"}, false});
        return points;
    }
    if (opt.have_tokens) {
        for (const auto t : opt.tokens)
            append_point(points, {opt.qtype, opt.policy, opt.n, opt.k, t, {"explicit"}, false});
        return points;
    }
    for (std::int64_t t = opt.sweep.begin; t <= opt.sweep.end; t += opt.sweep.step) {
        points.push_back(
            {opt.qtype, opt.policy, opt.n, opt.k, static_cast<std::int32_t>(t), {"sweep"}, true});
        if (t > static_cast<std::int64_t>(opt.sweep.end) - opt.sweep.step) { break; }
    }
    return points;
}

std::vector<PointGroup> group_points(const std::vector<BenchPoint>& points) {
    std::vector<PointGroup> groups;
    for (const BenchPoint& point : points) {
        auto it = std::find_if(groups.begin(), groups.end(), [&](const PointGroup& group) {
            return group.qtype == point.qtype && group.policy == point.policy &&
                   group.n == point.n && group.k == point.k;
        });
        if (it == groups.end()) {
            groups.push_back({point.qtype, point.policy, point.n, point.k, {point}});
        } else {
            it->points.push_back(point);
        }
    }
    for (PointGroup& group : groups) {
        std::sort(group.points.begin(), group.points.end(),
                  [](const BenchPoint& a, const BenchPoint& b) { return a.t < b.t; });
    }
    return groups;
}

LinearBenchWeight make_weight(QType qtype, std::int32_t n, std::int32_t k) {
    if (qtype == QType::BF16) {
        if (n == 16384 && k == 1536) {
            auto direct = bench::make_direct_bf16_weight(20480, k);
            const std::array parent_shape{std::uint64_t(20480), std::uint64_t(k)};
            const WeightParent parent{
                weight_geometry(QType::BF16, QuantLayout::Contiguous, parent_shape),
                direct.storage.device(), static_cast<const std::byte*>(direct.storage.data())};
            const auto elements = std::uint64_t(n) * k;
            const auto prefix =
                native_weight({{std::uint64_t(n), std::uint64_t(k)}, {{&parent, 0, elements}}});
            return {std::move(direct.storage), prefix, elements * 2};
        }
        bench::DirectBf16Weight direct  = bench::make_direct_bf16_weight(n, k);
        const std::uint64_t model_bytes = direct.model_weight_bytes();
        return {std::move(direct.storage), direct.weight, model_bytes};
    }
    if (qtype == QType::NVFP4) {
        bench::PackedQuantizedWeight packed = bench::make_nvfp4_weight(n, k);
        const std::uint64_t model_bytes     = packed.model_weight_bytes();
        return {std::move(packed.storage), packed.weight, model_bytes};
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        bench::PackedQuantizedWeight packed = bench::make_fp8_weight(n, k);
        const std::uint64_t model_bytes     = packed.model_weight_bytes();
        return {std::move(packed.storage), packed.weight, model_bytes};
    }
    const std::uint64_t padded_k_u64 = align_up(static_cast<std::uint64_t>(k), 128);
    if (padded_k_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("padded K does not fit int32");
    }
    bench::PackedQuantizedWeight packed =
        bench::make_row_split_weight(qtype, n, k, static_cast<std::int32_t>(padded_k_u64), 501U);
    const std::uint64_t model_bytes = packed.model_weight_bytes();
    return {std::move(packed.storage), packed.weight, model_bytes};
}

void fill_activation(DeviceBuffer& buffer, std::uint64_t elements, CudaExecutionView cuda) {
    const auto stream = cuda.stream();
    CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(buffer.data()), elements,
                                           101U, -0.5F, 0.5F, stream));
}

std::string join_labels(const std::vector<std::string>& labels) {
    std::string out;
    for (const std::string& label : labels) {
        if (!out.empty()) { out += '|'; }
        out += label;
    }
    return out;
}

double registered_tensor_peak_tflops(const BenchPoint& point, const char*& profile) {
    // Report Tensor Core utilization only when the exact registered problem and extent determine
    // that the public route executes the named MMA profile.
    if (point.qtype == QType::Q4_G64_FP16 && point.k == 2560 &&
        (point.n == 248320 || point.n == 124160) && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::Q8_G32_FP16 && point.n == 4096 &&
        ((point.k == 1024 && point.t > 1) || (point.k == 16384 && point.t > 4))) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::Q4_G64_FP16 && point.n == 77440 && point.k == 4096 && point.t > 4) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 20480 && point.k == 1536 && point.t > 0) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 16384 && point.k == 1536 && point.t > 0) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 4096 && point.k == 8192 && point.t > 8) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 1024 && point.k == 1176 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 8192 && point.k == 1024 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 1024 && point.k == 4096 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 4096 && point.k == 4096 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 20480 && point.k == 4096 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 4096 && point.k == 10240 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 1024 && point.k == 1024 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::BF16 && point.n == 3072 && point.k == 1024 && point.t > 1) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    const bool nvfp4_problem =
        (point.n == 12288 && point.k == 4096) || (point.n == 4096 && point.k == 6144) ||
        (point.k == 5120 &&
         (point.n == 14336 || point.n == 16384 || point.n == 28672 || point.n == 34816)) ||
        (point.n == 5120 && (point.k == 6144 || point.k == 14336 || point.k == 17408));
    if (point.qtype == QType::NVFP4 && nvfp4_problem) {
        const bool a4 =
            point.policy == LinearPolicy::AllowA4 &&
            ((point.n == 12288 && point.k == 4096 && point.t >= 17) ||
             (point.n == 4096 && point.k == 6144 && point.t >= 9) ||
             (point.n == 14336 && point.t >= 4) || (point.n == 28672 && point.t >= 8) ||
             point.n == 16384 || point.n == 34816 ||
             (point.k == 6144 && point.t >= 17) ||
             (point.k == 14336 && point.t >= 5) || (point.k == 17408 && point.t >= 8));
        if (a4) {
            profile = "NVFP4_F32ACC";
            return bench::kRoofline.nvfp4_f32_tflops;
        }
        if (point.t >= (point.k == 14336 ? 1 : point.n == 28672 ? 2 :
                       point.k == 17408 ? 6 : 3)) {
            profile = "BF16_F32ACC";
            return bench::kRoofline.bf16_f32_tflops;
        }
    }
    if (point.qtype == QType::Q8_G32_FP16 &&
        (((point.n == 12288 || point.n == 4096 || point.n == 2048) && point.k == 4096) ||
         (point.n == 2560 && (point.k == 3072 || point.k == 6144)) ||
         (point.n == 6144 && point.k == 2560) || (point.n == 10240 && point.k == 2560) ||
         (point.n == 12288 && point.k == 2560) || (point.n == 16384 && point.k == 2560) ||
         (point.n == 8192 && point.k == 2560)) &&
        point.t > 1 && !(point.n == 12288 && point.t <= 8 && point.k == 4096)) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        (((point.n == 14336 || point.n == 16384) && point.k == 5120) ||
         (point.n == 5120 && (point.k == 6144 || point.k == 17408))) &&
        point.t > 1 && (point.policy == LinearPolicy::A16Only || point.t <= 16)) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    if (point.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        (point.n == 14976 || point.n == 5120) && point.k == 5120) {
        if (point.policy != LinearPolicy::A16Only && point.t >= 17) {
            profile = "MXFP8_F32ACC";
            return bench::kRoofline.mxfp8_f32_tflops;
        }
        if (point.t > 1) {
            profile = "BF16_F32ACC";
            return bench::kRoofline.bf16_f32_tflops;
        }
    }
    const bool fp8_problem =
        (point.n == 14336 && point.k == 5120) || (point.n == 16384 && point.k == 5120) ||
        (point.n == 34816 && point.k == 5120) || (point.n == 5120 && point.k == 6144) ||
        (point.n == 5120 && point.k == 17408);
    const bool fp8_tensor_route =
        (point.n == 14336 && point.k == 5120 && point.t >= 17) ||
        (point.n == 16384 && point.k == 5120 && point.t >= 17) ||
        (point.n == 34816 && point.k == 5120 && point.t >= 5) ||
        (point.n == 5120 && (point.k == 6144 || point.k == 17408) && point.t >= 17);
    if (point.qtype == QType::FP8_E4M3FN_ROW_BF16 && point.policy == LinearPolicy::AllowA8 &&
        fp8_problem && fp8_tensor_route) {
        profile = "MXFP8_F32ACC";
        return bench::kRoofline.mxfp8_f32_tflops;
    }
    if (point.qtype == QType::BF16 &&
        (((point.n == 1152 || point.n == 7296 || point.n == 640) && point.k == 2560 &&
          point.t >= 2) ||
         (point.n == 2560 && point.k == 320) || (point.n == 4096 && point.k == 128) ||
         (point.n == 288 && point.k == 4096 && point.t >= 2) ||
         (point.n == 2048 && point.k == 4096 && point.t >= 2) ||
         (point.n == 4096 && point.k == 1024 && point.t >= 2) ||
         (point.n == 77440 && point.k == 4096 && point.t >= 9) ||
         (point.n == 256 && point.k == 5120) ||
         ((point.n == 48 || point.n == 96) && point.k == 2560 && point.t >= 17) ||
         (point.n == 1664 && point.k == 2560 && point.t >= 2) ||
         (point.n == 2560 && point.k == 4608 && point.t >= 3) ||
         (point.n == 4608 && point.k == 4608 && point.t >= 3) ||
         (point.n == 4304 && point.k == 1152 && point.t >= 4) ||
         (point.n == 1152 && point.k == 4304 && point.t >= 4) ||
         (point.n == 1152 && point.k == 1152 && point.t >= 8) ||
         (point.n == 1152 && point.k == 1536 && point.t >= 4) ||
         (point.n == 3456 && point.k == 1152 && point.t >= 4) ||
         (point.n == 2560 && point.k == 2560 && point.t >= 2) ||
         (point.n == 2560 && point.k == 3072 && point.t >= 2) ||
         (point.n == 2560 && point.k == 6144 && point.t >= 2) ||
         (point.n == 13952 && point.k == 2560 && point.t >= 2) ||
         (point.n == 248320 && point.k == 2560 && point.t >= 2) ||
         (point.n == 124160 && point.k == 2560 && point.t >= 9) ||
         (point.n == 12800 && point.k == 2560 && point.t >= 2) ||
         (point.n == 10240 && point.k == 320 && point.t >= 2) ||
         (point.n == 320 && point.k == 10240 && point.t >= 9) ||
         (point.n == 324 && point.k == 10240 && point.t >= 9))) {
        profile = "BF16_F32ACC";
        return bench::kRoofline.bf16_f32_tflops;
    }
    profile = "";
    return std::numeric_limits<double>::quiet_NaN();
}

Result make_result(const BenchPoint& point, const LinearBenchWeight& weight,
                   const bench::ColdTiming& timing, const Options& opt) {
    const std::uint64_t x_elements =
        checked_mul(static_cast<std::uint64_t>(point.k), point.t, "activation elements");
    const std::uint64_t out_elements =
        checked_mul(static_cast<std::uint64_t>(point.n), point.t, "output elements");
    const std::uint64_t x_bytes     = checked_mul(x_elements, 2, "activation bytes");
    const std::uint64_t out_bytes   = checked_mul(out_elements, 2, "output bytes");
    const std::uint64_t model_bytes = checked_add(
        checked_add(weight.model_weight_bytes(), x_bytes, "model bytes"), out_bytes, "model bytes");
    const double useful_flops = 2.0 * static_cast<double>(point.n) * static_cast<double>(point.k) *
                                static_cast<double>(point.t);
    const double seconds = timing.median_us * 1.0e-6;
    const double memory_floor_us =
        static_cast<double>(model_bytes) / (bench::kRoofline.memory_gbs * 1.0e9) * 1.0e6;

    Result result;
    result.labels                   = join_labels(point.labels);
    result.qtype_name               = qtype_name(point.qtype);
    result.policy_name              = policy_name(point.policy);
    result.n                        = point.n;
    result.k                        = point.k;
    result.t                        = point.t;
    result.weight_bytes             = weight.model_weight_bytes();
    result.x_bytes                  = x_bytes;
    result.out_bytes                = out_bytes;
    result.model_bytes              = model_bytes;
    result.workspace_capacity_bytes = ops::linear_workspace_capacity_bytes(
        point.qtype, point.n, point.k, point.policy, point.t, point.t);
    result.useful_flops       = useful_flops;
    result.median_us          = timing.median_us;
    result.min_us             = timing.min_us;
    result.p95_us             = timing.p95_us;
    result.effective_gbs      = static_cast<double>(model_bytes) / seconds / 1.0e9;
    result.dram_spec_pct      = result.effective_gbs / bench::kRoofline.memory_gbs * 100.0;
    result.sustained_read_pct = result.effective_gbs / bench::kRoofline.sustained_read_gbs * 100.0;
    result.useful_tflops      = useful_flops / seconds / 1.0e12;
    result.tensor_peak_tflops = registered_tensor_peak_tflops(point, result.tensor_profile);
    if (std::isfinite(result.tensor_peak_tflops)) {
        result.tensor_peak_pct = result.useful_tflops / result.tensor_peak_tflops * 100.0;
    }
    result.memory_floor_us  = memory_floor_us;
    result.memory_floor_pct = memory_floor_us / timing.median_us * 100.0;
    if (!opt.cold || opt.graph_calls > 1) {
        result.dram_spec_pct = result.sustained_read_pct = result.memory_floor_us =
            result.memory_floor_pct                      = std::numeric_limits<double>::quiet_NaN();
    }
    result.warmup      = opt.warmup;
    result.repeat      = opt.repeat;
    result.flush_bytes = opt.cold ? opt.flush_bytes : 0;
    result.cache = !opt.cold ? "warm" : opt.graph_calls == 1 ? "cold" : "cold-before-graph-bundle";
    return result;
}

std::vector<Result> run_group(const PointGroup& group, const Options& opt,
                              bench::L2FlushBuffer& flush, CudaExecutionView cuda) {
    const auto stream = cuda.stream();
    const std::int32_t max_t =
        std::max_element(group.points.begin(), group.points.end(),
                         [](const BenchPoint& a, const BenchPoint& b) { return a.t < b.t; })
            ->t;
    const std::int32_t min_t =
        std::min_element(group.points.begin(), group.points.end(),
                         [](const BenchPoint& a, const BenchPoint& b) { return a.t < b.t; })
            ->t;
    const std::uint64_t x_elements =
        checked_mul(static_cast<std::uint64_t>(group.k), max_t, "activation allocation");
    const std::uint64_t out_elements =
        checked_mul(static_cast<std::uint64_t>(group.n), max_t, "output allocation");

    LinearBenchWeight weight = make_weight(group.qtype, group.n, group.k);
    DeviceBuffer x(ninfer::bench::device_context().device(),
                   checked_mul(x_elements, 2, "activation allocation bytes"));
    DeviceBuffer out(ninfer::bench::device_context().device(),
                     checked_mul(out_elements, 2, "output allocation bytes"));
    const std::size_t workspace_capacity = ops::linear_workspace_capacity_bytes(
        group.qtype, group.n, group.k, group.policy, min_t, max_t);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(workspace_capacity, 256));
    MemoryArena workspace(workspace_storage.span());
    fill_activation(x, x_elements, cuda);
    CUDA_CHECK(cudaMemsetAsync(out.data(), 0, out.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Result> results;
    results.reserve(group.points.size());
    double t1_median       = std::numeric_limits<double>::quiet_NaN();
    double previous_median = std::numeric_limits<double>::quiet_NaN();

    for (const BenchPoint& point : group.points) {
        Tensor activation(x.span(), DType::BF16, {group.k, point.t});
        Tensor output(out.span(), DType::BF16, {group.n, point.t});
        const auto launch = [&](CudaExecutionView cuda) {
            ops::linear(activation, weight.weight, output, group.policy, workspace, cuda);
        };
        bench::TimedGraph graph;
        if (opt.graph && !opt.body_timer)
            graph.capture(cuda, [&](CudaExecutionView cuda) {
                for (int call = 0; call < opt.graph_calls; ++call) launch(cuda);
            });
        bench::ColdTiming timing;
        std::size_t graph_nodes = graph.nodes();
        if (opt.body_timer) {
            bench::measurement::Options measurement;
            measurement.warmup  = opt.warmup;
            measurement.repeat  = opt.repeat;
            const auto measured = bench::measurement::measure(
                [&](CudaExecutionView execution) {
                    for (int call = 0; call < opt.graph_calls; ++call) launch(execution);
                },
                [](auto) {}, flush, cuda, measurement, "graph", opt.cold ? "cold" : "warm");
            timing      = measured.gpu;
            graph_nodes = measured.nodes;
        } else if (opt.cold)
            timing = opt.graph
                         ? bench::measure_cold_graph(graph, flush, cuda, opt.warmup, opt.repeat)
                         : bench::measure_cold_launch(launch, flush, cuda, opt.warmup, opt.repeat);
        else
            timing = opt.graph ? bench::measure_graph(graph, cuda, opt.warmup, opt.repeat)
                               : bench::measure_launch(launch, cuda, opt.warmup, opt.repeat);
        timing.median_us /= opt.graph_calls;
        timing.min_us /= opt.graph_calls;
        timing.p95_us /= opt.graph_calls;
        Result result      = make_result(point, weight, timing, opt);
        result.timer_scope = !opt.graph       ? "eager-launch"
                             : opt.body_timer ? "gpu-body"
                                              : "graph-launch";
        result.execution   = opt.graph ? "graph" : "eager";
        result.graph_nodes = graph_nodes;
        result.graph_calls = opt.graph_calls;
        if (point.t == 1) { t1_median = result.median_us; }
        if (std::isfinite(t1_median)) {
            result.t1_linear_extrapolation =
                static_cast<double>(point.t) * t1_median / result.median_us;
        }
        if (point.sweep_point && std::isfinite(previous_median)) {
            result.delta_pct = (result.median_us / previous_median - 1.0) * 100.0;
        }
        if (point.sweep_point) { previous_median = result.median_us; }
        results.push_back(std::move(result));
    }
    return results;
}

void run_profile(const BenchPoint& point, const Options& opt, bench::L2FlushBuffer& flush,
                 CudaExecutionView cuda) {
    const auto stream = cuda.stream();
    const std::uint64_t x_elements =
        checked_mul(static_cast<std::uint64_t>(point.k), point.t, "activation allocation");
    const std::uint64_t out_elements =
        checked_mul(static_cast<std::uint64_t>(point.n), point.t, "output allocation");
    LinearBenchWeight weight = make_weight(point.qtype, point.n, point.k);
    DeviceBuffer x(ninfer::bench::device_context().device(),
                   checked_mul(x_elements, 2, "activation allocation bytes"));
    DeviceBuffer out(ninfer::bench::device_context().device(),
                     checked_mul(out_elements, 2, "output allocation bytes"));
    const std::size_t workspace_capacity = ops::linear_workspace_capacity_bytes(
        point.qtype, point.n, point.k, point.policy, point.t, point.t);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(workspace_capacity, 256));
    MemoryArena workspace(workspace_storage.span());
    fill_activation(x, x_elements, cuda);
    CUDA_CHECK(cudaMemsetAsync(out.data(), 0, out.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor activation(x.span(), DType::BF16, {point.k, point.t});
    Tensor output(out.span(), DType::BF16, {point.n, point.t});
    const auto body = [&](CudaExecutionView cuda) {
        ops::linear(activation, weight.weight, output, point.policy, workspace, cuda);
    };
    bench::TimedGraph graph;
    if (opt.graph) graph.capture(cuda, body);
    const auto launch = [&] {
        if (opt.graph)
            graph.launch(cuda);
        else
            body(cuda);
    };
    std::printf("# execution=%s graph_nodes=%zu\n", opt.graph ? "graph" : "eager", graph.nodes());
    for (int i = 0; i < opt.warmup; ++i) {
        if (opt.cold) bench::flush_l2(flush, cuda);
        launch();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (opt.cold) bench::flush_l2(flush, cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::uint64_t x_bytes     = checked_mul(x_elements, 2, "activation bytes");
    const std::uint64_t out_bytes   = checked_mul(out_elements, 2, "output bytes");
    const std::uint64_t model_bytes = checked_add(
        checked_add(weight.model_weight_bytes(), x_bytes, "model bytes"), out_bytes, "model bytes");
    const double useful_flops = 2.0 * static_cast<double>(point.n) * static_cast<double>(point.k) *
                                static_cast<double>(point.t);
    std::printf("PROFILE linear qtype=%s policy=%s route=public N=%d K=%d T=%d model_bytes=%llu "
                "useful_flops=%.0f\n",
                qtype_name(point.qtype), policy_name(point.policy), point.n, point.k, point.t,
                static_cast<unsigned long long>(model_bytes), useful_flops);
    std::fflush(stdout);

    CUDA_CHECK(cudaProfilerStart());
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

void print_header(const Options& opt) {
    std::printf("# execution=%s graph_calls=%d cuda_runtime=%d\n", opt.graph ? "graph" : "eager",
                opt.graph_calls, CUDART_VERSION);
    std::printf("# timer_scope=%s\n", !opt.graph       ? "eager-launch"
                                      : opt.body_timer ? "gpu-body"
                                                       : "graph-launch");
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("# actual_gpu=%s sm=%d%d reference_target=%s\n", properties.name, properties.major,
                properties.minor, bench::kRoofline.target);
    std::printf("# dram_spec_gbs=%.1f sustained_read_gbs=%s cache=%s\n",
                bench::kRoofline.memory_gbs,
                bench::format_metric(bench::kRoofline.sustained_read_gbs).c_str(),
                !opt.cold              ? "warm"
                : opt.graph_calls == 1 ? "cold"
                                       : "cold-before-graph-bundle");
    std::printf("# dense_fp8_tensor_tflops fp16_acc=%s fp32_acc=%s mxfp8_fp32_acc=%s\n",
                bench::format_metric(bench::kRoofline.fp8_f16_tflops).c_str(),
                bench::format_metric(bench::kRoofline.fp8_f32_tflops).c_str(),
                bench::format_metric(bench::kRoofline.mxfp8_f32_tflops).c_str());
    std::printf("# dense_bf16_tensor_tflops fp32_acc=%s\n",
                bench::format_metric(bench::kRoofline.bf16_f32_tflops).c_str());
    std::printf("# dense_nvfp4_tensor_tflops fp32_acc=%s\n",
                bench::format_metric(bench::kRoofline.nvfp4_f32_tflops).c_str());
}

void print_results(const std::vector<Result>& results) {
    std::printf("%-44s %5s %3s %8s %8s %6s %11s %11s %11s %10s %7s %7s %10s %11s %8s "
                "%9s %9s %8s\n",
                "label", "qt", "pol", "N", "K", "T", "median_us", "min_us", "p95_us", "eff_GB/s",
                "DRAM_%", "READ_%", "TFLOP/s", "TC_profile", "TC_%", "mem_%", "T1_lin_x",
                "delta_%");
    for (const Result& result : results) {
        std::printf("%-44s %5s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %7s "
                    "%7s %10.2f %11s %8s %9s %9s %8s\n",
                    result.labels.c_str(), result.qtype_name, result.policy_name, result.n,
                    result.k, result.t, result.median_us, result.min_us, result.p95_us,
                    result.effective_gbs, bench::format_metric(result.dram_spec_pct).c_str(),
                    bench::format_metric(result.sustained_read_pct).c_str(), result.useful_tflops,
                    result.tensor_profile[0] == '\0' ? "-" : result.tensor_profile,
                    bench::format_metric(result.tensor_peak_pct).c_str(),
                    bench::format_metric(result.memory_floor_pct).c_str(),
                    bench::format_metric(result.t1_linear_extrapolation).c_str(),
                    bench::format_metric(result.delta_pct).c_str());
    }
}

std::string csv_quote(std::string_view value) {
    std::string out{"\""};
    for (const char c : value) {
        if (c == '"') { out.push_back('"'); }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

void write_csv(const std::filesystem::path& path, const std::vector<Result>& results) {
    if (path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV output: " + path.string()); }
    out << "label,qtype,policy,N,K,T,weight_bytes,x_bytes,out_bytes,model_bytes,"
           "useful_flops,median_us,min_us,p95_us,effective_gbs,dram_spec_gbs,dram_spec_pct,"
           "sustained_read_gbs,sustained_read_pct,useful_tflops,tensor_profile,"
           "tensor_peak_tflops,tensor_peak_pct,memory_floor_us,memory_floor_pct,"
           "t1_linear_extrapolation,delta_pct,"
           "warmup,repeat,flush_bytes,execution,graph_nodes,graph_calls,cache,timer_scope,"
           "workspace_capacity_bytes\n";
    for (const Result& result : results) {
        out << csv_quote(result.labels) << ',' << result.qtype_name << ',' << result.policy_name
            << ',' << result.n << ',' << result.k << ',' << result.t << ',' << result.weight_bytes
            << ',' << result.x_bytes << ',' << result.out_bytes << ',' << result.model_bytes << ','
            << result.useful_flops << ',' << result.median_us << ',' << result.min_us << ','
            << result.p95_us << ',' << result.effective_gbs << ',' << bench::kRoofline.memory_gbs
            << ',' << bench::format_metric(result.dram_spec_pct, 6, "") << ','
            << bench::format_metric(bench::kRoofline.sustained_read_gbs, 2, "") << ','
            << bench::format_metric(result.sustained_read_pct, 6, "") << ',' << result.useful_tflops
            << ',' << result.tensor_profile << ',';
        if (std::isfinite(result.tensor_peak_tflops)) { out << result.tensor_peak_tflops; }
        out << ',';
        if (std::isfinite(result.tensor_peak_pct)) { out << result.tensor_peak_pct; }
        out << ',' << bench::format_metric(result.memory_floor_us, 6, "") << ','
            << bench::format_metric(result.memory_floor_pct, 6, "") << ',';
        if (std::isfinite(result.t1_linear_extrapolation)) {
            out << result.t1_linear_extrapolation;
        }
        out << ',';
        if (std::isfinite(result.delta_pct)) { out << result.delta_pct; }
        out << ',' << result.warmup << ',' << result.repeat << ',' << result.flush_bytes << ','
            << result.execution << ',';
        if (result.graph_nodes) out << result.graph_nodes;
        out << ',' << result.graph_calls << ',' << result.cache << ',' << result.timer_scope << ','
            << result.workspace_capacity_bytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        int device_count  = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        bench::L2FlushBuffer flush(opt.flush_bytes);
        const std::vector<BenchPoint> points = expand_points(opt);

        print_header(opt);
        if (opt.profile) {
            run_profile(points.front(), opt, flush, cuda);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const PointGroup& group : group_points(points)) {
            std::vector<Result> group_results = run_group(group, opt, flush, cuda);
            results.insert(results.end(), std::make_move_iterator(group_results.begin()),
                           std::make_move_iterator(group_results.end()));
        }
        print_results(results);
        if (!opt.csv_out.empty()) { write_csv(opt.csv_out, results); }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_bench: %s\n", error.what());
        usage(argv[0]);
        return 2;
    }
}
