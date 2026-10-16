#include "common/roofline.h"
#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/moe.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "timing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {
using bench::moe::BodyTimedGraph;
using bench::moe::Stats;
using bench::moe::summarize;

constexpr std::int32_t kHidden             = 2048;
constexpr std::int32_t kExperts            = 256;
constexpr std::int32_t kTopK               = 8;
constexpr std::int32_t kIntermediate       = 512;
constexpr std::int32_t kRouterRows         = kExperts + 1;
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;

enum class CodecProfile : std::uint8_t {
    Q4Q5,
    Q4Q6,
    Q8Q8,
};

enum class ExpertDistribution : std::uint8_t {
    TraceLike,
    Independent,
    Same,
};

enum class Execution : std::uint8_t {
    Graph,
    Eager,
    Both,
};

enum class CacheMode : std::uint8_t {
    Cold,
    Warm,
    Both,
};

enum class CacheState : std::uint8_t {
    Cold,
    Warm,
};

struct TokenSweep {
    std::int32_t begin = 1;
    std::int32_t end   = 1;
    std::int32_t step  = 1;
};

enum class Operation { Complete, Route, Routed, RoutedShared };

const char* operation_name(Operation op) {
    switch (op) {
    case Operation::Complete:
        return "complete";
    case Operation::Route:
        return "route";
    case Operation::Routed:
        return "routed";
    case Operation::RoutedShared:
        return "routed-shared";
    }
    return "unknown";
}

struct Options {
    Operation operation          = Operation::Complete;
    ops::ExpertSubsetKind subset = ops::ExpertSubsetKind::All;
    std::string codec            = "q4-q5";
    TokenSweep tokens;
    ExpertDistribution distribution = ExpertDistribution::TraceLike;
    Execution execution             = Execution::Graph;
    CacheMode cache                 = CacheMode::Both;
    std::uint32_t seed              = 20260718U;
    int warmup                      = 5;
    int repeat                      = 50;
    std::uint64_t flush_bytes       = kDefaultFlushBytes;
    std::string csv_out;
};

struct RoutePattern {
    std::vector<std::array<std::int32_t, kTopK>> selected;
    std::int32_t unique_experts = 0;
    double adjacent_overlap     = 0.0;
};

struct Result {
    Operation operation;
    ops::ExpertSubsetKind subset;
    int selected_assignments;
    CodecProfile codec;
    std::int32_t tokens;
    ExpertDistribution distribution;
    std::uint32_t seed;
    std::int32_t unique_experts;
    double adjacent_overlap;
    Execution execution;
    CacheState cache;
    Stats stats;
    std::size_t workspace_bytes;
};

const char* codec_name(CodecProfile profile) {
    switch (profile) {
    case CodecProfile::Q4Q5:
        return "q4-q5";
    case CodecProfile::Q4Q6:
        return "q4-q6";
    case CodecProfile::Q8Q8:
        return "q8-q8";
    }
    return "unknown";
}

QType gate_codec(CodecProfile profile) {
    return profile == CodecProfile::Q8Q8 ? QType::Q8_G32_FP16 : QType::Q4_G64_FP16;
}

QType down_codec(CodecProfile profile) {
    switch (profile) {
    case CodecProfile::Q4Q5:
        return QType::Q5_G64_FP16;
    case CodecProfile::Q4Q6:
        return QType::Q6_G64_FP16;
    case CodecProfile::Q8Q8:
        return QType::Q8_G32_FP16;
    }
    throw std::logic_error("unknown MoE codec profile");
}

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

const char* execution_name(Execution execution) {
    switch (execution) {
    case Execution::Graph:
        return "graph_replay";
    case Execution::Eager:
        return "eager";
    case Execution::Both:
        break;
    }
    return "unknown";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

std::uint64_t packed_weight_bytes(QType qtype, std::int32_t rows, std::int32_t columns) {
    const std::int32_t group   = qtype == QType::Q8_G32_FP16 ? 32 : 64;
    const std::uint64_t groups = static_cast<std::uint64_t>(rows) * columns / group;
    const std::uint64_t low    = qtype == QType::Q8_G32_FP16
                                     ? static_cast<std::uint64_t>(rows) * columns
                                     : static_cast<std::uint64_t>(rows) * columns / 2;
    const std::uint64_t high   = qtype == QType::Q5_G64_FP16   ? groups * 8
                                 : qtype == QType::Q6_G64_FP16 ? groups * 16
                                                               : 0;
    return low + high + groups * sizeof(std::uint16_t);
}

double unique_weight_bytes(const Result& result) {
    const bool router =
        result.operation == Operation::Route || result.operation == Operation::Complete;
    const bool shared =
        result.operation == Operation::RoutedShared || result.operation == Operation::Complete;
    const std::uint64_t fixed =
        (router ? std::uint64_t(kRouterRows) * kHidden * 2 : 0) +
        (shared ? packed_weight_bytes(QType::Q8_G32_FP16, 1024, kHidden) +
                      packed_weight_bytes(QType::Q8_G32_FP16, kHidden, kIntermediate)
                : 0);
    const std::uint64_t per_expert =
        packed_weight_bytes(gate_codec(result.codec), 1024, kHidden) +
        packed_weight_bytes(down_codec(result.codec), kHidden, kIntermediate);
    return static_cast<double>(fixed) + (result.operation == Operation::Route
                                             ? 0.0
                                             : double(result.unique_experts) * double(per_expert));
}

double logical_flops(const Result& result) {
    const double ffn = 2.0 * 1024 * kHidden + 2.0 * kHidden * kIntermediate;
    const bool router =
        result.operation == Operation::Route || result.operation == Operation::Complete;
    const bool shared =
        result.operation == Operation::RoutedShared || result.operation == Operation::Complete;
    return result.tokens * ((router ? 2.0 * kRouterRows * kHidden : 0) + (shared ? ffn : 0)) +
           (result.operation == Operation::Route ? 0 : result.selected_assignments * ffn);
}

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end                       = nullptr;
    errno                           = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::int32_t parse_positive_i32(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must be in [1, INT32_MAX]");
    }
    return static_cast<std::int32_t>(value);
}

int parse_nonnegative_int(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<int>(value);
}

TokenSweep parse_sweep(std::string_view text) {
    const std::string value(text);
    const std::size_t first = value.find(':');
    if (first == std::string::npos) {
        throw std::invalid_argument("--sweep must be START:END or START:END:STEP");
    }
    const std::size_t second = value.find(':', first + 1);
    if (second != std::string::npos && value.find(':', second + 1) != std::string::npos) {
        throw std::invalid_argument("--sweep has too many fields");
    }
    TokenSweep sweep;
    sweep.begin = parse_positive_i32(value.substr(0, first), "sweep start");
    sweep.end   = parse_positive_i32(value.substr(first + 1, second == std::string::npos
                                                                 ? std::string::npos
                                                                 : second - first - 1),
                                     "sweep end");
    if (second != std::string::npos) {
        sweep.step = parse_positive_i32(value.substr(second + 1), "sweep step");
    }
    if (sweep.begin > sweep.end) {
        throw std::invalid_argument("--sweep start must not exceed end");
    }
    return sweep;
}

ExpertDistribution parse_distribution(std::string_view value) {
    if (value == "trace-like") return ExpertDistribution::TraceLike;
    if (value == "independent") return ExpertDistribution::Independent;
    if (value == "same") return ExpertDistribution::Same;
    throw std::invalid_argument("--distribution must be trace-like, independent, or same");
}

Execution parse_execution(std::string_view value) {
    if (value == "graph") return Execution::Graph;
    if (value == "eager") return Execution::Eager;
    if (value == "both") return Execution::Both;
    throw std::invalid_argument("--execution must be graph, eager, or both");
}

CacheMode parse_cache(std::string_view value) {
    if (value == "cold") return CacheMode::Cold;
    if (value == "warm") return CacheMode::Warm;
    if (value == "both") return CacheMode::Both;
    throw std::invalid_argument("--cache must be cold, warm, or both");
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n\n"
                 "Public workload:\n"
                 "  --op complete|route|routed|routed-shared (default complete).\n"
                 "  --subset all|mask-ones|mask-zeros|none (routed Ops only).\n"
                 "  Complete adds to BF16 residual; routed Ops write FP32.\n"
                 "  --codec q4-q5|q4-q6|q8-q8|all  Routed weight profile (default q4-q5).\n"
                 "  --tokens T                       Exact token extent (default 1).\n"
                 "  --sweep START:END[:STEP]         Public token-extent sweep.\n"
                 "  --distribution trace-like|independent|same\n"
                 "  --seed N                         Fixture seed.\n\n"
                 "Measurement:\n"
                 "  --execution graph|eager|both     Default graph.\n"
                 "  --cache cold|warm|both           Default both; cold is authoritative.\n"
                 "  --warmup N                       Warmup replays per point (default 5).\n"
                 "  --repeat N                       Measured samples per point (default 50).\n"
                 "  --flush-mib N                    L2 eviction storage (default 256 MiB).\n"
                 "  --csv-out PATH                   Write result rows as CSV.\n"
                 "  -h, --help                       Show this text.\n",
                 argv0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool have_tokens = false;
    bool have_sweep  = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--op") {
            const auto value = next("op");
            if (value == "complete") {
                options.operation = Operation::Complete;
            } else if (value == "route") {
                options.operation = Operation::Route;
            } else if (value == "routed") {
                options.operation = Operation::Routed;
            } else if (value == "routed-shared") {
                options.operation = Operation::RoutedShared;
            } else {
                throw std::invalid_argument("invalid --op");
            }
        } else if (argument == "--subset") {
            const auto value = next("subset");
            if (value == "all") {
                options.subset = ops::ExpertSubsetKind::All;
            } else if (value == "mask-ones") {
                options.subset = ops::ExpertSubsetKind::MaskOnes;
            } else if (value == "mask-zeros") {
                options.subset = ops::ExpertSubsetKind::MaskZeros;
            } else if (value == "none") {
                options.subset = ops::ExpertSubsetKind::None;
            } else {
                throw std::invalid_argument("invalid --subset");
            }
        } else if (argument == "--codec") {
            options.codec = next("codec");
        } else if (argument == "--tokens") {
            const std::int32_t tokens = parse_positive_i32(next("tokens"), "tokens");
            options.tokens            = {tokens, tokens, 1};
            have_tokens               = true;
        } else if (argument == "--sweep") {
            options.tokens = parse_sweep(next("sweep"));
            have_sweep     = true;
        } else if (argument == "--distribution") {
            options.distribution = parse_distribution(next("distribution"));
        } else if (argument == "--execution") {
            options.execution = parse_execution(next("execution"));
        } else if (argument == "--cache") {
            options.cache = parse_cache(next("cache"));
        } else if (argument == "--seed") {
            const std::uint64_t seed = parse_u64(next("seed"), "seed");
            if (seed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("seed exceeds uint32");
            }
            options.seed = static_cast<std::uint32_t>(seed);
        } else if (argument == "--warmup") {
            options.warmup = parse_nonnegative_int(next("warmup"), "warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_nonnegative_int(next("repeat"), "repeat");
        } else if (argument == "--flush-mib") {
            const std::uint64_t mib = parse_u64(next("flush-mib"), "flush-mib");
            if (mib == 0 || mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("flush-mib is out of range");
            }
            options.flush_bytes = mib << 20;
        } else if (argument == "--csv-out") {
            options.csv_out = next("CSV output path");
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (have_tokens && have_sweep) {
        throw std::invalid_argument("--tokens and --sweep are mutually exclusive");
    }
    if (options.codec != "q4-q5" && options.codec != "q4-q6" && options.codec != "q8-q8" &&
        options.codec != "all") {
        throw std::invalid_argument("--codec must be q4-q5, q4-q6, q8-q8, or all");
    }
    if (options.subset != ops::ExpertSubsetKind::All &&
        (options.operation == Operation::Complete || options.operation == Operation::Route)) {
        throw std::invalid_argument("--subset requires a routed Op");
    }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (options.flush_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("flush buffer does not fit size_t");
    }
    return options;
}

std::vector<CodecProfile> selected_profiles(const std::string& codec) {
    if (codec == "all") { return {CodecProfile::Q4Q5, CodecProfile::Q4Q6, CodecProfile::Q8Q8}; }
    if (codec == "q4-q5") return {CodecProfile::Q4Q5};
    if (codec == "q4-q6") return {CodecProfile::Q4Q6};
    return {CodecProfile::Q8Q8};
}

std::vector<std::int32_t> selected_tokens(const TokenSweep& sweep) {
    std::vector<std::int32_t> result;
    for (std::int64_t tokens = sweep.begin; tokens <= sweep.end; tokens += sweep.step) {
        result.push_back(static_cast<std::int32_t>(tokens));
        if (tokens > static_cast<std::int64_t>(sweep.end) - sweep.step) { break; }
    }
    return result;
}

std::vector<Execution> selected_executions(Execution execution) {
    if (execution == Execution::Both) { return {Execution::Graph, Execution::Eager}; }
    return {execution};
}

std::vector<CacheState> selected_caches(CacheMode cache) {
    if (cache == CacheMode::Both) { return {CacheState::Cold, CacheState::Warm}; }
    return {cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm};
}

RoutePattern make_route_pattern(std::int32_t tokens, ExpertDistribution distribution,
                                std::uint32_t seed) {
    std::mt19937 random(seed);
    RoutePattern result;
    result.selected.resize(static_cast<std::size_t>(tokens));

    if (tokens == 1) {
        result.selected[0]      = {0, 32, 64, 96, 128, 160, 224, 255};
        result.unique_experts   = kTopK;
        result.adjacent_overlap = kTopK;
        return result;
    }

    const auto sample_unique = [&](const std::array<bool, kExperts>& excluded) {
        std::array<std::int32_t, kTopK> selected{};
        std::array<bool, kExperts> unavailable = excluded;
        std::uniform_int_distribution<std::int32_t> expert_distribution(0, kExperts - 1);
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            std::int32_t expert = 0;
            do { expert = expert_distribution(random); } while (unavailable[expert]);
            selected[rank]      = expert;
            unavailable[expert] = true;
        }
        return selected;
    };

    std::array<std::int32_t, 24> hot_pool{};
    {
        std::array<std::int32_t, kExperts> experts{};
        for (std::int32_t expert = 0; expert < kExperts; ++expert) { experts[expert] = expert; }
        std::shuffle(experts.begin(), experts.end(), random);
        std::copy_n(experts.begin(), hot_pool.size(), hot_pool.begin());
    }

    result.selected[0] = sample_unique({});
    for (std::int32_t token = 1; token < tokens; ++token) {
        if (distribution == ExpertDistribution::Same) {
            result.selected[token] = result.selected[0];
            continue;
        }
        if (distribution == ExpertDistribution::Independent) {
            result.selected[token] = sample_unique({});
            continue;
        }

        std::array<std::int32_t, kTopK> selected{};
        std::array<bool, kExperts> used{};
        std::array<std::int32_t, kTopK> previous = result.selected[token - 1];
        std::shuffle(previous.begin(), previous.end(), random);
        std::int32_t count = 0;
        for (; count < 3; ++count) {
            selected[count]       = previous[count];
            used[selected[count]] = true;
        }
        std::uniform_int_distribution<std::int32_t> hot_distribution(
            0, static_cast<std::int32_t>(hot_pool.size()) - 1);
        while (count < 5) {
            const std::int32_t expert = hot_pool[hot_distribution(random)];
            if (used[expert]) { continue; }
            selected[count++] = expert;
            used[expert]      = true;
        }
        std::uniform_int_distribution<std::int32_t> expert_distribution(0, kExperts - 1);
        while (count < kTopK) {
            const std::int32_t expert = expert_distribution(random);
            if (used[expert]) { continue; }
            selected[count++] = expert;
            used[expert]      = true;
        }
        std::shuffle(selected.begin(), selected.end(), random);
        result.selected[token] = selected;
    }

    std::array<bool, kExperts> present{};
    double overlap_sum = 0.0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t expert : result.selected[token]) { present[expert] = true; }
        if (token == 0) { continue; }
        std::int32_t overlap = 0;
        for (std::int32_t expert : result.selected[token]) {
            overlap +=
                std::find(result.selected[token - 1].begin(), result.selected[token - 1].end(),
                          expert) != result.selected[token - 1].end();
        }
        overlap_sum += overlap;
    }
    result.unique_experts =
        static_cast<std::int32_t>(std::count(present.begin(), present.end(), true));
    result.adjacent_overlap = overlap_sum / static_cast<double>(tokens - 1);
    return result;
}

Weight dense_weight(ConstMemorySpan storage, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.device          = storage.device;
    result.payload         = storage.data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2ULL;
    result.qtype           = QType::BF16;
    result.qdata           = storage.data;
    result.n               = rows;
    result.k               = columns;
    result.layout          = QuantLayout::Contiguous;
    result.ndim            = 2;
    result.shape[0]        = rows;
    result.shape[1]        = columns;
    result.padded_shape[0] = rows;
    result.padded_shape[1] = columns;
    return result;
}

class BenchmarkWeights {
public:
    BenchmarkWeights(CodecProfile profile, std::uint32_t seed, std::size_t flush_bytes)
        : router_(ninfer::bench::device_context().device(),
                  static_cast<std::size_t>(kRouterRows) * kHidden * 2),
          routed_gate_(bench::make_row_split_weight(gate_codec(profile), kExperts * 1024, kHidden,
                                                    kHidden, seed ^ 501U)),
          routed_down_(bench::make_row_split_weight(down_codec(profile), kExperts * kHidden,
                                                    kIntermediate, kIntermediate, seed ^ 503U)),
          shared_gate_(bench::make_row_split_weight(QType::Q8_G32_FP16, 1024, kHidden, kHidden,
                                                    seed ^ 505U)),
          shared_down_(bench::make_row_split_weight(QType::Q8_G32_FP16, kHidden, kIntermediate,
                                                    kIntermediate, seed ^ 507U)),
          flush_(flush_bytes) {
        std::vector<std::uint16_t> router(static_cast<std::size_t>(kRouterRows) * kHidden,
                                          bench::f32_to_bf16(0.0F));
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            router[static_cast<std::size_t>(expert) * kHidden + expert] = bench::f32_to_bf16(16.0F);
        }
        router[static_cast<std::size_t>(kExperts) * kHidden + kExperts] = bench::f32_to_bf16(4.0F);
        router_.copy_from_host(router.data(), router_.size());
        bench::flush_l2(flush_, bench::default_execution());

        const auto affine = [](const Weight& w) {
            const auto groups = std::uint64_t(w.n / kExperts) * (w.k / w.group);
            const auto high   = w.qtype == QType::Q5_G64_FP16   ? 8
                                : w.qtype == QType::Q6_G64_FP16 ? 16
                                                                : 0;
            return ExpertMatrixDeviceView{static_cast<const std::byte*>(w.qdata),
                                          static_cast<const std::byte*>(w.qhigh),
                                          static_cast<const std::byte*>(w.scales),
                                          groups * 32,
                                          groups * high,
                                          groups * 2};
        };
        weights_ = {
            dense_weight(router_.span(), kRouterRows, kHidden),
            {{affine(routed_gate_.weight), affine(routed_down_.weight), kExperts, router_.device()},
             {}},
            {{shared_gate_.weight}, {shared_down_.weight}}};
        profile_ = {{{kHidden, kIntermediate, gate_codec(profile), down_codec(profile),
                      QuantLayout::RowSplit, QuantLayout::RowSplit},
                     kExperts,
                     kTopK},
                    ops::swiglu_profile(weights_.shared)};
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] const ops::MoeWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const ops::MoeProfile& profile() const noexcept { return profile_; }

    void flush(CudaExecutionView cuda) { bench::flush_l2(flush_, cuda); }

private:
    DeviceBuffer router_;
    bench::PackedQuantizedWeight routed_gate_;
    bench::PackedQuantizedWeight routed_down_;
    bench::PackedQuantizedWeight shared_gate_;
    bench::PackedQuantizedWeight shared_down_;
    bench::L2FlushBuffer flush_;
    ops::MoeWeights weights_{};
    ops::MoeProfile profile_{};
};

class BenchmarkState {
public:
    BenchmarkState(BenchmarkWeights& fixture, std::int32_t tokens, const Options& options)
        : fixture_(fixture), operation_(options.operation), subset_kind_(options.subset),
          route_pattern_(make_route_pattern(tokens, options.distribution, options.seed)),
          input_(ninfer::bench::device_context().device(),
                 static_cast<std::size_t>(tokens) * kHidden * 2),
          residual_(ninfer::bench::device_context().device(),
                    static_cast<std::size_t>(tokens) * kHidden * 2),
          destination_(ninfer::bench::device_context().device(),
                       static_cast<std::size_t>(tokens) * kHidden *
                           (operation_ == Operation::Complete ? 2 : 4)),
          workspace_bytes_(workspace_bytes(fixture.profile(), tokens, options)),
          workspace_storage_(bench::make_buffer(workspace_bytes_)),
          workspace_(workspace_storage_.span()),
          ids_(bench::make_buffer(std::size_t(tokens) * kTopK * 4)),
          alpha_(bench::make_buffer(std::size_t(tokens) * kTopK * 4)),
          beta_(bench::make_buffer(std::size_t(tokens) * 4)),
          mask_(bench::make_buffer(kExperts * 4)) {
        const auto seed = options.seed;
        std::vector<std::uint16_t> input(static_cast<std::size_t>(tokens) * kHidden);
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(tokens) * kHidden);
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t index = 0; index < kHidden; ++index) {
                const auto offset = static_cast<std::size_t>(token) * kHidden + index;
                input[offset] =
                    bench::f32_to_bf16(bench::fixture::uniform(offset, seed ^ 101U, -.04F, .04F));
                residual[offset] =
                    bench::f32_to_bf16(bench::fixture::uniform(offset, seed ^ 103U, -.2F, .2F));
            }
            for (std::int32_t expert = 0; expert < kExperts; ++expert) {
                input[static_cast<std::size_t>(token) * kHidden + expert] =
                    bench::f32_to_bf16(-0.25F);
            }
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const std::int32_t expert = route_pattern_.selected[token][rank];
                input[static_cast<std::size_t>(token) * kHidden + expert] =
                    bench::f32_to_bf16(0.25F - static_cast<float>(rank) / 64.0F);
            }
            input[static_cast<std::size_t>(token) * kHidden + kExperts] =
                bench::f32_to_bf16(0.0625F);
        }
        input_.copy_from_host(input.data(), input_.size());
        residual_.copy_from_host(residual.data(), residual_.size());
        if (operation_ == Operation::Complete) {
            CUDA_CHECK(cudaMemcpy(destination_.data(), residual_.data(), residual_.size(),
                                  cudaMemcpyDeviceToDevice));
        }

        x_                  = Tensor(input_.span(), DType::BF16, {kHidden, tokens});
        destination_tensor_ = Tensor(destination_.span(),
                                     operation_ == Operation::Complete ? DType::BF16 : DType::FP32,
                                     {kHidden, tokens});
        routes_             = {Tensor(ids_.span(), DType::I32, {kTopK, tokens}),
                               Tensor(alpha_.span(), DType::FP32, {kTopK, tokens})};
        beta_tensor_        = Tensor(beta_.span(), DType::FP32, {tokens});
        std::array<int, kExperts> membership;
        for (int e = 0; e < kExperts; ++e) { membership[e] = e % 2; }
        mask_.copy_from_host(membership.data(), sizeof(membership));
        subset_ = {subset_kind_, Tensor(mask_.span(), DType::I32, {kExperts})};
        std::array<bool, kExperts> touched{};
        for (const auto& row : route_pattern_.selected) {
            for (int e : row) {
                if (subset_kind_ == ops::ExpertSubsetKind::All ||
                    (subset_kind_ == ops::ExpertSubsetKind::MaskOnes && membership[e]) ||
                    (subset_kind_ == ops::ExpertSubsetKind::MaskZeros && !membership[e])) {
                    touched[e] = true;
                    ++selected_assignments_;
                }
            }
        }
        route_pattern_.unique_experts = std::count(touched.begin(), touched.end(), true);
        if (operation_ == Operation::Routed || operation_ == Operation::RoutedShared) {
            DeviceBuffer scratch =
                bench::make_buffer(ops::topk_softmax_route_workspace_capacity_bytes(
                    {kHidden, kExperts, kTopK}, tokens, tokens));
            MemoryArena arena(scratch.span());
            ops::topk_softmax_route(x_, {kHidden, kExperts, kTopK}, fixture.weights().control,
                                    routes_, beta_tensor_, arena, bench::default_execution());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] const RoutePattern& route_pattern() const noexcept { return route_pattern_; }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    int selected_assignments() const { return selected_assignments_; }

    void prepare(CacheState cache, CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        if (cache == CacheState::Cold) { fixture_.flush(cuda); }
        if (operation_ == Operation::Complete) {
            CUDA_CHECK(cudaMemcpyAsync(destination_.data(), residual_.data(), residual_.size(),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    void launch(CudaExecutionView cuda) {
        const auto& profile = fixture_.profile();
        const auto& weights = fixture_.weights();
        switch (operation_) {
        case Operation::Complete:
            ops::topk_softmax_swiglu_moe(x_, profile, weights, ops::MoeEpilogue::AddResidual,
                                         destination_tensor_, workspace_, cuda);
            break;
        case Operation::Route:
            ops::topk_softmax_route(x_, {kHidden, kExperts, kTopK}, weights.control, routes_,
                                    beta_tensor_, workspace_, cuda);
            break;
        case Operation::Routed:
            ops::routed_swiglu(x_, profile.routed, weights.routed, routes_, subset_,
                               ops::MoeEpilogue::Write, destination_tensor_, workspace_, cuda);
            break;
        case Operation::RoutedShared:
            ops::routed_swiglu(x_, profile.routed, weights.routed, routes_, subset_,
                               {weights.shared, beta_tensor_}, ops::MoeEpilogue::Write,
                               destination_tensor_, workspace_, cuda);
            break;
        }
    }

private:
    static std::size_t workspace_bytes(const ops::MoeProfile& p, int tokens,
                                       const Options& options) {
        switch (options.operation) {
        case Operation::Complete:
            return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                p, ops::MoeEpilogue::AddResidual, tokens, tokens);
        case Operation::Route:
            return ops::topk_softmax_route_workspace_capacity_bytes({kHidden, kExperts, kTopK},
                                                                    tokens, tokens);
        case Operation::Routed:
            return ops::routed_swiglu_workspace_capacity_bytes(
                p.routed, options.subset, DType::FP32, ops::MoeEpilogue::Write, tokens, tokens);
        case Operation::RoutedShared:
            return ops::routed_swiglu_workspace_capacity_bytes(p.routed, p.shared, options.subset,
                                                               DType::FP32, ops::MoeEpilogue::Write,
                                                               tokens, tokens);
        }
        throw std::invalid_argument("invalid Op");
    }

    BenchmarkWeights& fixture_;
    Operation operation_;
    ops::ExpertSubsetKind subset_kind_;
    RoutePattern route_pattern_;
    DeviceBuffer input_, residual_, destination_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    MemoryArena workspace_;
    DeviceBuffer ids_, alpha_, beta_, mask_;
    Tensor x_, destination_tensor_, beta_tensor_;
    ops::ExpertRoutes routes_;
    ops::ExpertSubset subset_;
    int selected_assignments_ = 0;
};

Stats measure_eager(BenchmarkState& state, CacheState cache, CudaExecutionView cuda, int warmup,
                    int repeat) {
    const auto stream = cuda.stream();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, cuda);
        state.launch(cuda);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, cuda);
        CUDA_CHECK(cudaEventRecord(start, stream));
        state.launch(cuda);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize(std::move(samples));
}

Stats measure_graph(BenchmarkState& state, const BodyTimedGraph& graph, CacheState cache,
                    CudaExecutionView cuda, int warmup, int repeat) {
    const auto stream = cuda.stream();
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, cuda);
        graph.launch(cuda);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, cuda);
        samples.push_back(graph.launch_body_timed(cuda));
    }
    return summarize(std::move(samples));
}

std::vector<Result> run_point(BenchmarkWeights& fixture, CodecProfile profile, std::int32_t tokens,
                              const Options& options, CudaExecutionView device_execution) {
    const auto cuda   = device_execution;
    const auto stream = cuda.stream();
    BenchmarkState state(fixture, tokens, options);

    // Match the production graph lifecycle: materialize eagerly, capture from a
    // reset state, instantiate, then prime one replay before configured warmup.
    state.prepare(CacheState::Warm, cuda);
    state.launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BodyTimedGraph graph;
    if (options.execution != Execution::Eager) {
        state.prepare(CacheState::Warm, cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(cuda, [&](CudaExecutionView cuda) { state.launch(cuda); });
        state.prepare(CacheState::Warm, cuda);
        graph.launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<Result> results;
    const RoutePattern& pattern = state.route_pattern();
    for (Execution execution : selected_executions(options.execution)) {
        for (CacheState cache : selected_caches(options.cache)) {
            const Stats stats =
                execution == Execution::Graph
                    ? measure_graph(state, graph, cache, cuda, options.warmup, options.repeat)
                    : measure_eager(state, cache, cuda, options.warmup, options.repeat);
            results.push_back({options.operation, options.subset, state.selected_assignments(),
                               profile, tokens, options.distribution, options.seed,
                               pattern.unique_experts, pattern.adjacent_overlap, execution, cache,
                               stats, state.workspace_bytes()});
        }
    }
    return results;
}

void print_result(const Result& result, double peak_memory_gbps) {
    const double seconds            = result.stats.median_us * 1.0e-6;
    const double useful_weight_gbps = unique_weight_bytes(result) / seconds / 1.0e9;
    const double memory_lower_bound = unique_weight_bytes(result) / (peak_memory_gbps * 1000.0);
    const double tflops             = logical_flops(result) / seconds / 1.0e12;
    std::printf("%-6s T=%-5d %-11s U=%-3d ov=%4.1f %-12s %-4s "
                "median=%8.3f us min=%8.3f us p95=%8.3f us "
                "unique_weight=%7.1f GB/s memory_lower_bound=%7.3f us logical=%6.2f TFLOP/s "
                "workspace=%zu\n",
                codec_name(result.codec), result.tokens, distribution_name(result.distribution),
                result.unique_experts, result.adjacent_overlap, execution_name(result.execution),
                cache_name(result.cache), result.stats.median_us, result.stats.min_us,
                result.stats.p95_us, useful_weight_gbps, memory_lower_bound, tflops,
                result.workspace_bytes);
}

void write_csv(const std::string& path, const std::vector<Result>& results, const Options& options,
               const DeviceContext& context) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV output"); }
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    const double peak_memory_gbps = bench::kRoofline.memory_gbs;
    stream << "codec,tokens,distribution,seed,unique_experts,adjacent_overlap,execution,"
              "timed_scope,cache,median_us,min_us,p95_us,unique_weight_gbps,"
              "memory_lower_bound_us,logical_tflops,workspace_bytes,warmup,repeat,"
              "flush_bytes,build_type,gpu,cuda_runtime,op,subset,selected_assignments\n";
    for (const Result& result : results) {
        const double seconds            = result.stats.median_us * 1.0e-6;
        const double useful_weight_gbps = unique_weight_bytes(result) / seconds / 1.0e9;
        stream << codec_name(result.codec) << ',' << result.tokens << ','
               << distribution_name(result.distribution) << ',' << result.seed << ','
               << result.unique_experts << ',' << result.adjacent_overlap << ','
               << execution_name(result.execution) << ",public_op_device_body,"
               << cache_name(result.cache) << ',' << result.stats.median_us << ','
               << result.stats.min_us << ',' << result.stats.p95_us << ',' << useful_weight_gbps
               << ',' << unique_weight_bytes(result) / (peak_memory_gbps * 1000.0) << ','
               << logical_flops(result) / seconds / 1.0e12 << ',' << result.workspace_bytes << ','
               << options.warmup << ',' << options.repeat << ',' << options.flush_bytes << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << context.properties().name << ',' << runtime << ','
               << operation_name(result.operation) << ',' << int(result.subset) << ','
               << result.selected_assignments << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options         = parse_options(argc, argv);
        auto& context                 = ninfer::bench::device_context();
        const double peak_memory_gbps = bench::kRoofline.memory_gbs;
        std::printf("# gpu=%s compute_capability=%d execution=%s "
                    "timed_scope=public_op_device_body cold_flush_mib=%llu "
                    "reference_target=%s theoretical_memory=%.1f GB/s\n",
                    context.properties().name, context.compute_capability(),
                    options.execution == Execution::Both ? "both"
                                                         : execution_name(options.execution),
                    static_cast<unsigned long long>(options.flush_bytes >> 20),
                    bench::kRoofline.target, peak_memory_gbps);

        std::printf(
            "# op=%s subset=%d; memory bound uses unique weights and theoretical bandwidth\n",
            operation_name(options.operation), int(options.subset));
        std::vector<Result> results;
        for (CodecProfile profile : selected_profiles(options.codec)) {
            BenchmarkWeights fixture(profile, options.seed,
                                     static_cast<std::size_t>(options.flush_bytes));
            for (std::int32_t tokens : selected_tokens(options.tokens)) {
                std::vector<Result> point =
                    run_point(fixture, profile, tokens, options, context.execution_view());
                for (const Result& result : point) { print_result(result, peak_memory_gbps); }
                results.insert(results.end(), std::make_move_iterator(point.begin()),
                               std::make_move_iterator(point.end()));
            }
        }
        write_csv(options.csv_out, results, options, context);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_moe_bench: %s\n", error.what());
        return 1;
    }
}
