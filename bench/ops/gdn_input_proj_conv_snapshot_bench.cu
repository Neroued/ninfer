// Public Qwen3-family GDN projection/convolution Snapshot and ReplaySSM Record benchmark.
//
// The timed body is exactly one selected gdn_input_proj_conv_*() public Op call.
// Production dispatch, kernel topology, and workspace use remain behind that contract.

#include "ninfer/ops/gdn_input_proj.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden             = 5120;
constexpr std::int32_t kQueryRows          = 2048;
constexpr std::int32_t kKeyRows            = 2048;
constexpr std::int32_t kValueRows          = 6144;
constexpr std::int32_t kZRows              = 6144;
constexpr std::int32_t kQkRows             = kQueryRows + kKeyRows;
constexpr std::int32_t kValueZRows         = kValueRows + kZRows;
constexpr std::int32_t kChannels           = kQueryRows + kKeyRows + kValueRows;
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;

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

enum class Format : std::uint8_t {
    Q4Q5,
    Nvfp4,
    Fp8,
    W8,
    All,
};

enum class Form : std::uint8_t {
    Snapshot,
    Record,
    Both,
};

struct GdnGeometry {
    std::int32_t hidden;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t z_rows;

    [[nodiscard]] std::int32_t channels() const { return query_rows + key_rows + value_rows; }
};

struct TokenSweep {
    std::int32_t begin = 1;
    std::int32_t end   = 6;
    std::int32_t step  = 1;
};

struct Options {
    TokenSweep tokens;
    Format format                  = Format::Q4Q5;
    Form form                      = Form::Snapshot;
    ops::LinearPolicy nvfp4_policy = ops::LinearPolicy::AllowA4;
    ops::LinearPolicy fp8_policy   = ops::LinearPolicy::AllowA8;
    Execution execution            = Execution::Graph;
    CacheMode cache                = CacheMode::Both;
    int warmup                     = 10;
    int repeat                     = 100;
    std::uint64_t flush_bytes      = kDefaultFlushBytes;
    std::int32_t batch             = 1;
    std::vector<std::int32_t> valid_columns;
    std::string csv_out;
};

struct Stats {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Result {
    const char* profile;
    Form form;
    std::int32_t tokens;
    std::int32_t batch;
    Execution execution;
    CacheState cache;
    Stats stats;
    std::size_t workspace_bytes;
};

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

Format parse_format(std::string_view value) {
    if (value == "q4q5") return Format::Q4Q5;
    if (value == "nvfp4") return Format::Nvfp4;
    if (value == "fp8") return Format::Fp8;
    if (value == "w8") return Format::W8;
    if (value == "all") return Format::All;
    throw std::invalid_argument("--format must be q4q5, nvfp4, fp8, w8, or all");
}

Form parse_form(std::string_view value) {
    if (value == "snapshot") return Form::Snapshot;
    if (value == "record") return Form::Record;
    if (value == "both") return Form::Both;
    throw std::invalid_argument("--form must be snapshot, record, or both");
}

std::vector<std::int32_t> parse_valid_columns(std::string_view text) {
    std::vector<std::int32_t> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        values.push_back(parse_positive_i32(text.substr(0, comma), "valid column"));
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return values;
}

ops::LinearPolicy parse_nvfp4_policy(std::string_view value) {
    if (value == "a16") return ops::LinearPolicy::A16Only;
    if (value == "a4") return ops::LinearPolicy::AllowA4;
    throw std::invalid_argument("--nvfp4-policy must be a16 or a4");
}

ops::LinearPolicy parse_fp8_policy(std::string_view value) {
    if (value == "a16") return ops::LinearPolicy::A16Only;
    if (value == "a8") return ops::LinearPolicy::AllowA8;
    throw std::invalid_argument("--fp8-policy must be a16 or a8");
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n\n"
                 "Public workload:\n"
                 "  --format q4q5|nvfp4|fp8|w8|all  Default q4q5.\n"
                 "  --form snapshot|record|both  Default snapshot.\n"
                 "  --nvfp4-policy a16|a4        Default a4.\n"
                 "  --fp8-policy a16|a8          Default a8.\n"
                 "  --tokens T                    Exact token extent.\n"
                 "  --sweep START:END[:STEP]      Token sweep (default 1:6).\n\n"
                 "  --batch B                     Exact batch in [1,8] (default 1).\n"
                 "  --valid-columns V0,V1,...     Mixed valid prefixes; requires exact tokens.\n\n"
                 "Measurement:\n"
                 "  --execution graph|eager|both  Default graph.\n"
                 "  --cache cold|warm|both        Default both; cold matches layer-to-layer use.\n"
                 "  --warmup N                    Warmup replays per point (default 10).\n"
                 "  --repeat N                    Measured samples per point (default 100).\n"
                 "  --flush-mib N                 L2 eviction storage (default 256 MiB).\n"
                 "  --csv-out PATH                Write result rows as CSV.\n"
                 "  -h, --help                    Show this text.\n",
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
        if (argument == "--format") {
            options.format = parse_format(next("format"));
        } else if (argument == "--form") {
            options.form = parse_form(next("form"));
        } else if (argument == "--nvfp4-policy") {
            options.nvfp4_policy = parse_nvfp4_policy(next("NVFP4 policy"));
        } else if (argument == "--fp8-policy") {
            options.fp8_policy = parse_fp8_policy(next("FP8 policy"));
        } else if (argument == "--tokens") {
            const std::int32_t tokens = parse_positive_i32(next("tokens"), "tokens");
            options.tokens            = {tokens, tokens, 1};
            have_tokens               = true;
        } else if (argument == "--sweep") {
            options.tokens = parse_sweep(next("sweep"));
            have_sweep     = true;
        } else if (argument == "--batch") {
            options.batch = parse_positive_i32(next("batch"), "batch");
        } else if (argument == "--valid-columns") {
            options.valid_columns = parse_valid_columns(next("valid columns"));
        } else if (argument == "--execution") {
            options.execution = parse_execution(next("execution"));
        } else if (argument == "--cache") {
            options.cache = parse_cache(next("cache"));
        } else if (argument == "--warmup") {
            options.warmup = parse_nonnegative_int(next("warmup"), "warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_nonnegative_int(next("repeat"), "repeat");
        } else if (argument == "--flush-mib") {
            const std::uint64_t mib = parse_u64(next("flush-mib"), "flush-mib");
            if (mib == 0 || mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--flush-mib is out of range");
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
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (options.flush_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("flush buffer does not fit size_t");
    }
    if (options.batch > 8 || (options.batch > 1 && options.tokens.end > 16) ||
        (!options.valid_columns.empty() &&
         (options.batch == 1 || !have_tokens ||
          options.valid_columns.size() != static_cast<std::size_t>(options.batch)))) {
        throw std::invalid_argument("invalid batch or valid-column workload");
    }
    for (const std::int32_t valid : options.valid_columns) {
        if (valid > options.tokens.begin) {
            throw std::invalid_argument("valid column exceeds exact width");
        }
    }
    return options;
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

std::vector<Form> selected_forms(Form form, std::int32_t tokens) {
    if (form == Form::Snapshot) return {Form::Snapshot};
    if (form == Form::Record) {
        if (tokens < 2) { throw std::invalid_argument("ReplaySSM Record requires T>=2"); }
        return {Form::Record};
    }
    if (tokens == 1) return {Form::Snapshot};
    return {Form::Snapshot, Form::Record};
}

const char* form_name(Form form) {
    switch (form) {
    case Form::Snapshot:
        return "snapshot";
    case Form::Record:
        return "record";
    case Form::Both:
        break;
    }
    return "unknown";
}

std::vector<CacheState> selected_caches(CacheMode cache) {
    if (cache == CacheMode::Both) { return {CacheState::Cold, CacheState::Warm}; }
    return {cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm};
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

const char* policy_name(ops::LinearPolicy policy) {
    if (policy == ops::LinearPolicy::AllowA4) return "a4";
    if (policy == ops::LinearPolicy::AllowA8) return "a8";
    return "a16";
}

class Q4Q5Fixture {
public:
    explicit Q4Q5Fixture(std::size_t flush_bytes)
        : qk_(bench::make_row_split_weight(QType::Q4G64_F16S, kQkRows, kHidden, kHidden,
                                           {0x53, 0x00, 0x3400})),
          value_z_(bench::make_row_split_weight(QType::Q5G64_F16S, kValueZRows, kHidden, kHidden,
                                                {0x53, 0x55, 0x3400})),
          conv_weight_(bench::make_bf16(static_cast<std::size_t>(kChannels) * 4)),
          flush_(flush_bytes) {
        CUDA_CHECK(cudaMemset(flush_.p, 0xa5, flush_.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] Tensor conv_weight() const {
        return Tensor(conv_weight_.p, DType::BF16, {kChannels, 4});
    }

    [[nodiscard]] const char* profile() const noexcept { return "q4-q5"; }

    [[nodiscard]] GdnGeometry geometry() const noexcept {
        return {kHidden, kQueryRows, kKeyRows, kValueRows, kZRows};
    }

    [[nodiscard]] std::size_t workspace_capacity(Form form, std::int32_t batch,
                                                 std::int32_t tokens) const {
        if (form == Form::Record) {
            return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, tokens, tokens);
        }
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, tokens, tokens);
    }

    void launch(Form form, const Tensor& x, Tensor& conv_states, const Tensor& initial,
                const Tensor& snapshot_base, const Tensor& valid_columns, Tensor& conv_record,
                Tensor& query, Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                cudaStream_t stream) {
        Tensor convolution_weight = conv_weight();
        if (form == Form::Record) {
            ops::gdn_input_proj_conv_record(x, qk_.weight, value_z_.weight, convolution_weight,
                                            conv_states, valid_columns, initial, conv_record, query,
                                            key, value, z, workspace, stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(x, qk_.weight, value_z_.weight, convolution_weight,
                                              conv_states, valid_columns, initial, snapshot_base,
                                              query, key, value, z, workspace, stream);
        }
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, stream));
    }

private:
    bench::PackedQuantizedWeight qk_;
    bench::PackedQuantizedWeight value_z_;
    DeviceBuffer conv_weight_;
    DeviceBuffer flush_;
};

class Nvfp4Fixture {
public:
    Nvfp4Fixture(std::size_t flush_bytes, ops::LinearPolicy policy)
        : parent_(bench::make_nvfp4_weight(kChannels + kZRows, kHidden)),
          conv_weight_(bench::make_bf16(static_cast<std::size_t>(kChannels) * 4)),
          flush_(flush_bytes), policy_(policy) {
        CUDA_CHECK(cudaMemset(flush_.p, 0xa5, flush_.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] Tensor conv_weight() const {
        return Tensor(conv_weight_.p, DType::BF16, {kChannels, 4});
    }

    [[nodiscard]] const char* profile() const noexcept {
        return policy_ == ops::LinearPolicy::AllowA4 ? "nvfp4-a4" : "nvfp4-a16";
    }

    [[nodiscard]] GdnGeometry geometry() const noexcept {
        return {kHidden, kQueryRows, kKeyRows, kValueRows, kZRows};
    }

    [[nodiscard]] std::size_t workspace_capacity(Form form, std::int32_t batch,
                                                 std::int32_t tokens) const {
        if (form == Form::Record) {
            return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::NVFP4, kChannels + kZRows, kHidden, policy_, batch, tokens, tokens);
        }
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, kChannels + kZRows, kHidden, policy_, batch, tokens, tokens);
    }

    void launch(Form form, const Tensor& x, Tensor& conv_states, const Tensor& initial,
                const Tensor& snapshot_base, const Tensor& valid_columns, Tensor& conv_record,
                Tensor& query, Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                cudaStream_t stream) {
        Tensor convolution_weight = conv_weight();
        if (form == Form::Record) {
            ops::gdn_input_proj_conv_record(x, parent_.weight, convolution_weight, conv_states,
                                            valid_columns, initial, conv_record, query, key, value,
                                            z, policy_, workspace, stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(x, parent_.weight, convolution_weight, conv_states,
                                              valid_columns, initial, snapshot_base, query, key,
                                              value, z, policy_, workspace, stream);
        }
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, stream));
    }

private:
    bench::PackedQuantizedWeight parent_;
    DeviceBuffer conv_weight_;
    DeviceBuffer flush_;
    ops::LinearPolicy policy_;
};

class Fp8Fixture {
public:
    Fp8Fixture(std::size_t flush_bytes, ops::LinearPolicy policy)
        : parent_(bench::make_fp8_weight(kChannels + kZRows, kHidden)),
          conv_weight_(bench::make_bf16(static_cast<std::size_t>(kChannels) * 4)),
          flush_(flush_bytes), policy_(policy) {
        CUDA_CHECK(cudaMemset(flush_.p, 0xa5, flush_.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] Tensor conv_weight() const {
        return Tensor(conv_weight_.p, DType::BF16, {kChannels, 4});
    }

    [[nodiscard]] const char* profile() const noexcept {
        return policy_ == ops::LinearPolicy::AllowA8 ? "fp8-a8" : "fp8-a16";
    }

    [[nodiscard]] GdnGeometry geometry() const noexcept {
        return {kHidden, kQueryRows, kKeyRows, kValueRows, kZRows};
    }

    [[nodiscard]] std::size_t workspace_capacity(Form form, std::int32_t batch,
                                                 std::int32_t tokens) const {
        if (form == Form::Record) {
            return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kChannels + kZRows, kHidden, policy_, batch, tokens,
                tokens);
        }
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kChannels + kZRows, kHidden, policy_, batch, tokens,
            tokens);
    }

    void launch(Form form, const Tensor& x, Tensor& conv_states, const Tensor& initial,
                const Tensor& snapshot_base, const Tensor& valid_columns, Tensor& conv_record,
                Tensor& query, Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                cudaStream_t stream) {
        Tensor convolution_weight = conv_weight();
        if (form == Form::Record) {
            ops::gdn_input_proj_conv_record(x, parent_.weight, convolution_weight, conv_states,
                                            valid_columns, initial, conv_record, query, key, value,
                                            z, policy_, workspace, stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(x, parent_.weight, convolution_weight, conv_states,
                                              valid_columns, initial, snapshot_base, query, key,
                                              value, z, policy_, workspace, stream);
        }
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, stream));
    }

private:
    bench::PackedQuantizedWeight parent_;
    DeviceBuffer conv_weight_;
    DeviceBuffer flush_;
    ops::LinearPolicy policy_;
};

class W8Fixture {
public:
    explicit W8Fixture(std::size_t flush_bytes)
        : parent_(bench::make_row_split_weight(QType::W8G32_F16S, 12288, 2048, 2048,
                                               {0x03, 0x00, 0x3c00})),
          conv_weight_(bench::make_bf16(static_cast<std::size_t>(8192) * 4)), flush_(flush_bytes) {
        CUDA_CHECK(cudaMemset(flush_.p, 0xa5, flush_.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] Tensor conv_weight() const {
        return Tensor(conv_weight_.p, DType::BF16, {8192, 4});
    }

    [[nodiscard]] const char* profile() const noexcept { return "w8"; }

    [[nodiscard]] GdnGeometry geometry() const noexcept { return {2048, 2048, 2048, 4096, 4096}; }

    [[nodiscard]] std::size_t workspace_capacity(Form form, std::int32_t batch,
                                                 std::int32_t tokens) const {
        if (form == Form::Record) {
            return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, 4096, batch,
                                                                            tokens, tokens);
        }
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, batch,
                                                                          tokens, tokens);
    }

    void launch(Form form, const Tensor& x, Tensor& conv_states, const Tensor& initial,
                const Tensor& snapshot_base, const Tensor& valid_columns, Tensor& conv_record,
                Tensor& query, Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                cudaStream_t stream) {
        Tensor convolution_weight = conv_weight();
        if (form == Form::Record) {
            ops::gdn_input_proj_conv_record(x, parent_.weight, convolution_weight, conv_states,
                                            valid_columns, initial, conv_record, query, key, value,
                                            z, workspace, stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(x, parent_.weight, convolution_weight, conv_states,
                                              valid_columns, initial, snapshot_base, query, key,
                                              value, z, workspace, stream);
        }
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, stream));
    }

private:
    bench::PackedQuantizedWeight parent_;
    DeviceBuffer conv_weight_;
    DeviceBuffer flush_;
};

template <class Fixture>
class BenchmarkState {
public:
    BenchmarkState(Fixture& fixture, Form form, std::int32_t tokens, const Options& options)
        : fixture_(fixture), geometry_(fixture.geometry()), form_(form), batch_(options.batch),
          slots_(batch_ == 1 ? tokens + 1 : batch_ * tokens + batch_),
          input_(bench::make_bf16(static_cast<std::size_t>(geometry_.hidden) * tokens * batch_)),
          states_(bench::make_bf16(static_cast<std::size_t>(geometry_.channels()) * 3 * slots_)),
          conv_record_(static_cast<std::size_t>(geometry_.channels()) * tokens * batch_ * 2),
          initial_slot_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          snapshot_base_slot_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          query_(static_cast<std::size_t>(geometry_.query_rows) * tokens * batch_ * 2),
          key_(static_cast<std::size_t>(geometry_.key_rows) * tokens * batch_ * 2),
          value_(static_cast<std::size_t>(geometry_.value_rows) * tokens * batch_ * 2),
          z_(static_cast<std::size_t>(geometry_.z_rows) * tokens * batch_ * 2),
          workspace_bytes_(fixture.workspace_capacity(form_, batch_, tokens)),
          workspace_(std::max<std::size_t>(1, workspace_bytes_)),
          x_(input_.p, DType::BF16, {geometry_.hidden, tokens, batch_}),
          conv_states_(states_.p, DType::BF16, {geometry_.channels(), 3, slots_}),
          conv_record_tensor_(conv_record_.p, DType::BF16, {geometry_.channels(), tokens, batch_}),
          initial_(initial_slot_.p, DType::I32, {batch_}),
          snapshot_base_(snapshot_base_slot_.p, DType::I32, {batch_}),
          query_tensor_(query_.p, DType::BF16, {geometry_.query_rows, tokens, batch_}),
          key_tensor_(key_.p, DType::BF16, {geometry_.key_rows, tokens, batch_}),
          value_tensor_(value_.p, DType::BF16, {geometry_.value_rows, tokens, batch_}),
          z_tensor_(z_.p, DType::BF16, {geometry_.z_rows, tokens, batch_}) {
        std::vector<std::int32_t> initial(static_cast<std::size_t>(batch_));
        std::vector<std::int32_t> snapshot_base(static_cast<std::size_t>(batch_));
        if (batch_ == 1) {
            initial[0] = tokens;
        } else {
            for (std::int32_t row = 0; row < batch_; ++row) {
                snapshot_base[static_cast<std::size_t>(row)] = row * tokens;
                initial[static_cast<std::size_t>(row)]       = batch_ * tokens + row;
            }
        }
        initial_slot_.copy_from_host(initial.data(), initial_slot_.bytes);
        snapshot_base_slot_.copy_from_host(snapshot_base.data(), snapshot_base_slot_.bytes);
        if (!options.valid_columns.empty()) {
            valid_columns_ = DeviceBuffer(options.valid_columns.size() * sizeof(std::int32_t));
            valid_columns_.copy_from_host(options.valid_columns.data(), valid_columns_.bytes);
            valid_ = Tensor(valid_columns_.p, DType::I32, {batch_});
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] const char* profile() const noexcept { return fixture_.profile(); }

    [[nodiscard]] Form form() const noexcept { return form_; }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    void prepare(CacheState cache, cudaStream_t stream) {
        if (cache == CacheState::Cold) { fixture_.flush(stream); }
    }

    void launch(cudaStream_t stream) {
        fixture_.launch(form_, x_, conv_states_, initial_, snapshot_base_, valid_,
                        conv_record_tensor_, query_tensor_, key_tensor_, value_tensor_, z_tensor_,
                        workspace_, stream);
    }

private:
    Fixture& fixture_;
    GdnGeometry geometry_;
    Form form_;
    std::int32_t batch_;
    std::int32_t slots_;
    DeviceBuffer input_;
    DeviceBuffer states_;
    DeviceBuffer conv_record_;
    DeviceBuffer initial_slot_;
    DeviceBuffer snapshot_base_slot_;
    DeviceBuffer valid_columns_;
    DeviceBuffer query_;
    DeviceBuffer key_;
    DeviceBuffer value_;
    DeviceBuffer z_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor x_;
    Tensor conv_states_;
    Tensor conv_record_tensor_;
    Tensor initial_;
    Tensor snapshot_base_;
    Tensor valid_;
    Tensor query_tensor_;
    Tensor key_tensor_;
    Tensor value_tensor_;
    Tensor z_tensor_;
};

class BodyTimedGraph {
public:
    BodyTimedGraph() {
        CUDA_CHECK(cudaEventCreate(&body_start_));
        CUDA_CHECK(cudaEventCreate(&body_stop_));
        CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
    }

    ~BodyTimedGraph() {
        if (exec_ != nullptr) { cudaGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
        if (body_start_ != nullptr) { cudaEventDestroy(body_start_); }
        if (body_stop_ != nullptr) { cudaEventDestroy(body_stop_); }
        if (completion_ != nullptr) { cudaEventDestroy(completion_); }
    }

    BodyTimedGraph(const BodyTimedGraph&)            = delete;
    BodyTimedGraph& operator=(const BodyTimedGraph&) = delete;

    template <class Launch>
    void capture(cudaStream_t stream, Launch&& launch) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(cudaEventRecordWithFlags(body_start_, stream, cudaEventRecordExternal));
        launch(stream);
        CUDA_CHECK(cudaEventRecordWithFlags(body_stop_, stream, cudaEventRecordExternal));
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        std::size_t nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
        if (nodes < 3) { throw std::runtime_error("GDN conv capture produced an empty graph"); }
    }

    void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

    double launch_body_timed(cudaStream_t stream) const {
        launch(stream);
        CUDA_CHECK(cudaEventRecord(completion_, stream));
        CUDA_CHECK(cudaEventSynchronize(completion_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, body_start_, body_stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

private:
    cudaGraph_t graph_      = nullptr;
    cudaGraphExec_t exec_   = nullptr;
    cudaEvent_t body_start_ = nullptr;
    cudaEvent_t body_stop_  = nullptr;
    cudaEvent_t completion_ = nullptr;
};

Stats summarize(std::vector<double> samples) {
    if (samples.empty()) { throw std::invalid_argument("cannot summarize an empty sample set"); }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

template <class Fixture>
Stats measure_eager(BenchmarkState<Fixture>& state, CacheState cache, cudaStream_t stream,
                    int warmup, int repeat) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, stream);
        state.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        state.launch(stream);
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

template <class Fixture>
Stats measure_graph(BenchmarkState<Fixture>& state, const BodyTimedGraph& graph, CacheState cache,
                    cudaStream_t stream, int warmup, int repeat) {
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, stream);
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, stream);
        samples.push_back(graph.launch_body_timed(stream));
    }
    return summarize(std::move(samples));
}

template <class Fixture>
std::vector<Result> run_point(Fixture& fixture, Form form, std::int32_t tokens,
                              const Options& options, cudaStream_t stream) {
    BenchmarkState<Fixture> state(fixture, form, tokens, options);

    // Match production graph lifecycle: materialize once, capture, instantiate,
    // and prime one replay before the configured measurements.
    state.prepare(CacheState::Warm, stream);
    state.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BodyTimedGraph graph;
    if (options.execution != Execution::Eager) {
        graph.capture(stream, [&](cudaStream_t launch_stream) { state.launch(launch_stream); });
        graph.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<Result> results;
    for (Execution execution : selected_executions(options.execution)) {
        for (CacheState cache : selected_caches(options.cache)) {
            const Stats stats =
                execution == Execution::Graph
                    ? measure_graph(state, graph, cache, stream, options.warmup, options.repeat)
                    : measure_eager(state, cache, stream, options.warmup, options.repeat);
            results.push_back({state.profile(), state.form(), tokens, options.batch, execution,
                               cache, stats, state.workspace_bytes()});
        }
    }
    return results;
}

void print_result(const Result& result) {
    std::printf("%-10s %-8s T=%-3d B=%-2d %-12s %-4s median=%8.3f us min=%8.3f us "
                "p95=%8.3f us workspace=%zu\n",
                result.profile, form_name(result.form), result.tokens, result.batch,
                execution_name(result.execution), cache_name(result.cache), result.stats.median_us,
                result.stats.min_us, result.stats.p95_us, result.workspace_bytes);
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
    stream << "profile,form,tokens,batch,execution,timed_scope,cache,median_us,min_us,p95_us,"
              "workspace_bytes,warmup,repeat,flush_bytes,build_type,gpu,cuda_runtime\n";
    for (const Result& result : results) {
        stream << result.profile << ',' << form_name(result.form) << ',' << result.tokens << ','
               << result.batch << ',' << execution_name(result.execution)
               << ",full_gdn_input_proj_conv_device_body," << cache_name(result.cache) << ','
               << result.stats.median_us << ',' << result.stats.min_us << ',' << result.stats.p95_us
               << ',' << result.workspace_bytes << ',' << options.warmup << ',' << options.repeat
               << ',' << options.flush_bytes << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << context.props.name << ',' << runtime << '\n';
    }
}

template <class Fixture>
void run_fixture(Fixture& fixture, const Options& options, cudaStream_t stream,
                 std::vector<Result>& results) {
    for (std::int32_t tokens : selected_tokens(options.tokens)) {
        for (const Form form : selected_forms(options.form, tokens)) {
            std::vector<Result> point = run_point(fixture, form, tokens, options, stream);
            for (const Result& result : point) { print_result(result); }
            results.insert(results.end(), std::make_move_iterator(point.begin()),
                           std::make_move_iterator(point.end()));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        const char* configured_format = options.format == Format::Q4Q5    ? "q4q5"
                                        : options.format == Format::Nvfp4 ? "nvfp4"
                                        : options.format == Format::Fp8   ? "fp8"
                                        : options.format == Format::W8    ? "w8"
                                                                          : "all";
        std::printf(
            "# op=gdn_input_proj_conv form=%s format=%s nvfp4_policy=%s fp8_policy=%s gpu=%s "
            "compute_capability=%d batch=%d execution=%s "
            "timed_scope=full_public_op_device_body cold_flush_mib=%llu\n",
            options.form == Form::Both ? "both" : form_name(options.form), configured_format,
            policy_name(options.nvfp4_policy), policy_name(options.fp8_policy), context.props.name,
            context.compute_capability(), options.batch,
            options.execution == Execution::Both ? "both" : execution_name(options.execution),
            static_cast<unsigned long long>(options.flush_bytes >> 20));

        std::vector<Result> results;
        if (options.format == Format::Q4Q5 || options.format == Format::All) {
            Q4Q5Fixture fixture(static_cast<std::size_t>(options.flush_bytes));
            run_fixture(fixture, options, context.stream, results);
        }
        if (options.format == Format::Nvfp4 || options.format == Format::All) {
            Nvfp4Fixture fixture(static_cast<std::size_t>(options.flush_bytes),
                                 options.nvfp4_policy);
            run_fixture(fixture, options, context.stream, results);
        }
        if (options.format == Format::Fp8 || options.format == Format::All) {
            Fp8Fixture fixture(static_cast<std::size_t>(options.flush_bytes), options.fp8_policy);
            run_fixture(fixture, options, context.stream, results);
        }
        if (options.format == Format::W8 || options.format == Format::All) {
            W8Fixture fixture(static_cast<std::size_t>(options.flush_bytes));
            run_fixture(fixture, options, context.stream, results);
        }
        write_csv(options.csv_out, results, options, context);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_input_proj_conv_bench: %s\n", error.what());
        return 1;
    }
}
