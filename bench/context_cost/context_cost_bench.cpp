#include "context_cost_measure.h"
#include "qwen3_6_context_fixture.h"

#include "core/device.h"
#include "runtime/engine/context_cost.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json     = nlohmann::json;
namespace cost = ninfer::bench::context_cost;

constexpr double kMaximumTransferP95   = 0.35;
constexpr double kMaximumPrefillP95    = 0.15;
constexpr double kOrderingSignificance = 0.20;

enum class Suite : std::uint8_t {
    All,
    Transfer,
    Prefill,
};

const char* suite_name(Suite suite) noexcept {
    switch (suite) {
    case Suite::All:
        return "all";
    case Suite::Transfer:
        return "transfer";
    case Suite::Prefill:
        return "prefill";
    }
    return "unknown";
}

struct Options {
    cost::MeasurementOptions measurement;
    Suite suite = Suite::All;
    std::filesystem::path json_output;
    std::filesystem::path preset_output;
    bool help_requested = false;
};

std::uint64_t parse_u64(std::string_view value, const char* option) {
    std::size_t consumed = 0;
    std::uint64_t parsed = 0;
    try {
        parsed = std::stoull(std::string(value), &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires an unsigned integer");
    }
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string(option) + " requires an unsigned integer");
    }
    return parsed;
}

int parse_nonnegative_int(std::string_view value, const char* option) {
    const std::uint64_t parsed = parse_u64(value, option);
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(option) + " is out of range");
    }
    return static_cast<int>(parsed);
}

std::uint32_t parse_u32(std::string_view value, const char* option) {
    const std::uint64_t parsed = parse_u64(value, option);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(option) + " is out of range");
    }
    return static_cast<std::uint32_t>(parsed);
}

std::string usage(const char* executable) {
    std::ostringstream out;
    out << "usage: " << executable << " [options]\n\n"
        << "Measure the machine transfer roofline and per-artifact prefill cost.\n\n"
        << "  --suite <all|transfer|prefill> (default: all)\n"
        << "  --artifact <path>          required only by all/prefill; transfer loads no model\n"
        << "  --corpus <path>            prefill token corpus (default: "
           "bench/fixtures/bench_corpus.ids)\n"
        << "  --device <id>              CUDA device (default: 0)\n"
        << "  --max-context <tokens>     prefill measurement context (default: 8192)\n"
        << "  --prefill-chunk <tokens>   prefill measurement chunk (default: 1024)\n"
        << "  --transfer-warmup <n>      warmups per transfer point (default: 2)\n"
        << "  --transfer-reps <n>        samples per transfer point (default: 9)\n"
        << "  --prefill-reps <n>         samples per prefill point (default: 5)\n"
        << "  --json <path>              complete measurement report; stdout if omitted\n"
        << "  --preset-out <path>        atomically update measured preset components\n"
        << "  -h, --help                 show this help\n";
    return out.str();
}

Options parse_options(int argc, char** argv) {
    Options options;
    options.measurement.corpus = "bench/fixtures/bench_corpus.ids";
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.measurement.artifact = value("--artifact");
        } else if (argument == "--corpus") {
            options.measurement.corpus = value("--corpus");
        } else if (argument == "--suite") {
            const std::string_view selected = value("--suite");
            if (selected == "all") {
                options.suite = Suite::All;
            } else if (selected == "transfer") {
                options.suite = Suite::Transfer;
            } else if (selected == "prefill") {
                options.suite = Suite::Prefill;
            } else {
                throw std::invalid_argument("--suite must be all, transfer, or prefill");
            }
        } else if (argument == "--device") {
            options.measurement.device = parse_nonnegative_int(value("--device"), "--device");
        } else if (argument == "--max-context") {
            options.measurement.max_context = parse_u32(value("--max-context"), "--max-context");
        } else if (argument == "--prefill-chunk") {
            options.measurement.prefill_chunk =
                parse_u32(value("--prefill-chunk"), "--prefill-chunk");
        } else if (argument == "--transfer-warmup") {
            options.measurement.transfer_warmup =
                parse_nonnegative_int(value("--transfer-warmup"), "--transfer-warmup");
        } else if (argument == "--transfer-reps") {
            options.measurement.transfer_repetitions =
                parse_nonnegative_int(value("--transfer-reps"), "--transfer-reps");
        } else if (argument == "--prefill-reps") {
            options.measurement.prefill_repetitions =
                parse_nonnegative_int(value("--prefill-reps"), "--prefill-reps");
        } else if (argument == "--json") {
            options.json_output = value("--json");
        } else if (argument == "--preset-out") {
            options.preset_output = value("--preset-out");
        } else if (argument == "-h" || argument == "--help") {
            options.help_requested = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.help_requested) { return options; }
    const bool measures_prefill = options.suite != Suite::Transfer;
    if (measures_prefill && options.measurement.artifact.empty()) {
        throw std::invalid_argument("--artifact is required by all/prefill");
    }
    if (options.measurement.max_context == 0 || options.measurement.prefill_chunk == 0 ||
        options.measurement.prefill_chunk % 128U != 0U) {
        throw std::invalid_argument(
            "max-context must be positive and prefill-chunk aligned to 128");
    }
    if (options.measurement.transfer_repetitions <= 0 ||
        options.measurement.prefill_repetitions <= 0) {
        throw std::invalid_argument("measurement repetitions must be positive");
    }
    if (!options.preset_output.empty() && options.preset_output == options.json_output) {
        throw std::invalid_argument("--preset-out and --json must be different files");
    }
    return options;
}

std::string cuda_version(int value) {
    return value <= 0 ? std::string{}
                      : std::to_string(value / 1000) + "." + std::to_string((value % 1000) / 10);
}

struct Hardware {
    std::string gpu;
    std::string runtime;
    std::string driver;
    int major = 0;
    int minor = 0;
};

Hardware inspect_hardware(int device) {
    Hardware out;
    int runtime = 0;
    int driver  = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    out.runtime = cuda_version(runtime);
    out.driver  = cuda_version(driver);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    out.gpu   = properties.name;
    out.major = properties.major;
    out.minor = properties.minor;
    return out;
}

Json sample_json(const cost::SampleSummary& sample) {
    return Json{{"median_ns", sample.median_ns},
                {"mad_ns", sample.mad_ns},
                {"minimum_ns", sample.minimum_ns},
                {"maximum_ns", sample.maximum_ns},
                {"samples_ns", sample.elapsed_ns}};
}

Json metrics_json(const cost::FitMetrics& metrics) {
    return Json{{"median_relative_error", metrics.median_relative_error},
                {"p95_relative_error", metrics.p95_relative_error},
                {"maximum_relative_error", metrics.maximum_relative_error},
                {"ordering_failures", metrics.ordering_failures},
                {"ordering_ties", metrics.ordering_ties},
                {"ordering_pairs", metrics.ordering_pairs}};
}

Json fit_json(const cost::FitResult& fit) {
    Json coefficients = Json::array();
    for (std::size_t index = 0; index < fit.coefficients.size(); ++index) {
        coefficients.push_back(Json{{"name", fit.coefficient_names[index]},
                                    {"encoding", cost::encoding_name(fit.encodings[index])},
                                    {"floating", fit.coefficients[index]},
                                    {"quantized", fit.quantized_coefficients[index]}});
    }
    Json predictions = Json::array();
    for (const cost::PointPrediction& point : fit.predictions) {
        predictions.push_back(Json{{"label", point.label},
                                   {"elapsed_ns", point.elapsed_ns},
                                   {"predicted_ns", point.predicted_ns},
                                   {"relative_error", point.relative_error},
                                   {"validation", point.validation}});
    }
    return Json{{"accepted", fit.accepted},
                {"coefficients", std::move(coefficients)},
                {"training", metrics_json(fit.training)},
                {"validation", metrics_json(fit.validation)},
                {"predictions", std::move(predictions)}};
}

std::array<cost::FitResult, 3> fit_transfers(const cost::TransferSuiteResult& samples) {
    std::array<cost::FitResult, 3> result;
    constexpr std::array<cost::TransferDirection, 3> directions{
        cost::TransferDirection::DeviceToHost,
        cost::TransferDirection::HostToDevice,
        cost::TransferDirection::DeviceToDevice,
    };
    for (std::size_t direction = 0; direction < directions.size(); ++direction) {
        std::vector<cost::TransferRegressionPoint> points;
        for (const cost::TransferMeasurement& sample : samples.measurements) {
            if (sample.direction != directions[direction]) { continue; }
            points.push_back(cost::TransferRegressionPoint{
                .label = sample.label + "-p" + std::to_string(sample.page_count) + "-r" +
                         std::to_string(sample.contiguous_runs),
                .payload_bytes   = sample.work.payload_bytes,
                .copy_operations = sample.work.copy_operations,
                .elapsed_ns      = sample.timing.median_ns,
                .validation      = sample.validation,
            });
        }
        result[direction] =
            cost::fit_transfer_roofline(points, kMaximumTransferP95, kOrderingSignificance);
    }
    return result;
}

cost::FitResult fit_prefill(const cost::PrefillSuiteResult& samples) {
    std::vector<cost::RegressionPoint> points;
    points.reserve(samples.measurements.size());
    for (const cost::PrefillMeasurement& sample : samples.measurements) {
        points.push_back(cost::RegressionPoint{
            .label      = sample.label,
            .features   = {static_cast<double>(sample.chunks),
                           static_cast<double>(sample.suffix_tokens),
                           static_cast<double>(sample.attention_pairs),
                           static_cast<double>(sample.vision_items),
                           static_cast<double>(sample.vision_patches)},
            .elapsed_ns = sample.timing.median_ns,
            .validation = sample.validation,
        });
    }
    return cost::fit_nonnegative_model(
        points,
        {"chunk_ns", "token_ns_q32", "attention_pair_ns_q32", "vision_item_ns",
         "vision_patch_ns_q32"},
        {cost::CoefficientEncoding::IntegerNanoseconds, cost::CoefficientEncoding::Q32Nanoseconds,
         cost::CoefficientEncoding::Q32Nanoseconds, cost::CoefficientEncoding::IntegerNanoseconds,
         cost::CoefficientEncoding::Q32Nanoseconds},
        kMaximumPrefillP95, kOrderingSignificance);
}

std::array<ninfer::runtime::ContextTransferCost, 3>
runtime_transfer_cost(const std::array<cost::FitResult, 3>& fits) {
    std::array<ninfer::runtime::ContextTransferCost, 3> result;
    for (std::size_t index = 0; index < fits.size(); ++index) {
        const auto& values = fits[index].quantized_coefficients;
        if (values.size() != 3) { throw std::logic_error("incomplete transfer roofline fit"); }
        result[index] = ninfer::runtime::ContextTransferCost{
            .batch_ns = values[0], .operation_ns = values[1], .ns_per_byte_q32 = values[2]};
    }
    return result;
}

ninfer::runtime::ContextPrefillCost runtime_prefill_cost(const cost::FitResult& fit) {
    const auto& values = fit.quantized_coefficients;
    if (values.size() != 5) { throw std::logic_error("incomplete prefill fit"); }
    return ninfer::runtime::ContextPrefillCost{
        .chunk_ns              = values[0],
        .token_ns_q32          = values[1],
        .attention_pair_ns_q32 = values[2],
        .vision_item_ns        = values[3],
        .vision_patch_ns_q32   = values[4],
    };
}

void write_report(const std::filesystem::path& path, const std::string& text) {
    if (path.empty()) {
        std::cout << text << '\n';
        return;
    }
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open report path: " + path.string()); }
    output << text << '\n';
    if (!output) { throw std::runtime_error("failed to write report path: " + path.string()); }
    std::cerr << "wrote " << path.string() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer_context_cost_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << usage(argc > 0 ? argv[0] : "ninfer_context_cost_bench");
        return 0;
    }

    try {
        const Hardware hardware          = inspect_hardware(options.measurement.device);
        const std::string hardware_class = ninfer::runtime::context_cost_hardware_class(
            hardware.gpu, hardware.major, hardware.minor);
        std::optional<cost::ArtifactProfile> artifact;
        if (options.suite != Suite::Transfer) {
            artifact = cost::inspect_artifact(options.measurement.artifact);
        }

        std::optional<cost::TransferSuiteResult> transfer_samples;
        std::optional<cost::PrefillSuiteResult> prefill_samples;
        std::optional<std::array<cost::FitResult, 3>> transfer_fits;
        std::optional<cost::FitResult> prefill_fit;
        if (options.suite != Suite::Prefill) {
            transfer_samples = cost::measure_context_transfers(options.measurement);
            transfer_fits    = fit_transfers(*transfer_samples);
        }
        if (options.suite != Suite::Transfer) {
            prefill_samples = cost::measure_prefill(*artifact, options.measurement);
            prefill_fit     = fit_prefill(*prefill_samples);
        }

        bool accepted = !transfer_fits || std::all_of(transfer_fits->begin(), transfer_fits->end(),
                                                      [](const auto& fit) { return fit.accepted; });
        accepted      = accepted && (!prefill_fit || prefill_fit->accepted);

        Json report{
            {"schema_version", 2},
            {"artifact_type", "ninfer_context_cost_calibration"},
            {"accepted", accepted},
            {"hardware", Json{{"gpu", hardware.gpu},
                              {"hardware_class", hardware_class},
                              {"compute_capability", std::to_string(hardware.major) + "." +
                                                         std::to_string(hardware.minor)},
                              {"cuda_runtime", hardware.runtime},
                              {"cuda_driver", hardware.driver},
                              {"device", options.measurement.device}}},
            {"profile", Json{{"suite", suite_name(options.suite)},
                             {"transfer_warmup", options.measurement.transfer_warmup},
                             {"transfer_repetitions", options.measurement.transfer_repetitions},
                             {"prefill_repetitions", options.measurement.prefill_repetitions},
                             {"prefill_chunk", options.measurement.prefill_chunk},
                             {"max_context", options.measurement.max_context}}},
            {"acceptance", Json{{"maximum_transfer_p95_relative_error", kMaximumTransferP95},
                                {"maximum_prefill_p95_relative_error", kMaximumPrefillP95},
                                {"ordering_significance", kOrderingSignificance}}},
        };
        if (artifact) {
            report["artifact"] = Json{{"path", artifact->path.string()},
                                      {"model_id", artifact->model_id},
                                      {"weights_id", artifact->weights_id},
                                      {"target", artifact->target_key},
                                      {"canonical_kv_cache", "bf16"},
                                      {"canonical_speculative_backend", "none"},
                                      {"corpus", options.measurement.corpus.string()}};
        }
        if (transfer_samples && transfer_fits) {
            Json measurements = Json::array();
            for (const cost::TransferMeasurement& sample : transfer_samples->measurements) {
                measurements.push_back(
                    Json{{"label", sample.label},
                         {"direction", cost::transfer_direction_name(sample.direction)},
                         {"payload_bytes", sample.work.payload_bytes},
                         {"copy_operations", sample.work.copy_operations},
                         {"page_count", sample.page_count},
                         {"contiguous_runs", sample.contiguous_runs},
                         {"validation", sample.validation},
                         {"timing", sample_json(sample.timing)}});
            }
            Json fits = Json::object();
            for (std::size_t index = 0; index < transfer_fits->size(); ++index) {
                fits[cost::transfer_direction_name(static_cast<cost::TransferDirection>(index))] =
                    fit_json((*transfer_fits)[index]);
            }
            report["transfer"] = Json{{"model", "max(batch_ns + copy_operations * operation_ns, "
                                                "payload_bytes * ns_per_byte)"},
                                      {"measurements", std::move(measurements)},
                                      {"fits", std::move(fits)}};
        }
        if (prefill_samples && prefill_fit) {
            Json measurements = Json::array();
            for (const cost::PrefillMeasurement& sample : prefill_samples->measurements) {
                measurements.push_back(Json{{"label", sample.label},
                                            {"prefix_tokens", sample.prefix_tokens},
                                            {"suffix_tokens", sample.suffix_tokens},
                                            {"chunks", sample.chunks},
                                            {"attention_pairs", sample.attention_pairs},
                                            {"vision_items", sample.vision_items},
                                            {"vision_patches", sample.vision_patches},
                                            {"validation", sample.validation},
                                            {"timing", sample_json(sample.timing)}});
            }
            report["prefill"] = Json{
                {"measurements", std::move(measurements)},
                {"fit", fit_json(*prefill_fit)},
                {"load", Json{{"target", prefill_samples->load.target},
                              {"model_id", prefill_samples->load.model_id},
                              {"weights_id", prefill_samples->load.weights_id},
                              {"load_seconds", prefill_samples->load.load_seconds}}},
            };
        }

        write_report(options.json_output, report.dump(2));
        if (!accepted) {
            std::cerr << "context-cost calibration rejected: training/held-out error or ordering "
                         "exceeded its limit\n";
            return 3;
        }
        if (!options.preset_output.empty()) {
            const Json base_provenance{
                {"gpu", hardware.gpu},
                {"compute_capability",
                 std::to_string(hardware.major) + "." + std::to_string(hardware.minor)},
                {"cuda_runtime", hardware.runtime},
                {"cuda_driver", hardware.driver},
                {"device", options.measurement.device},
            };
            if (transfer_fits) {
                Json provenance                    = base_provenance;
                provenance["suite"]                = "transfer";
                provenance["transfer_warmup"]      = options.measurement.transfer_warmup;
                provenance["transfer_repetitions"] = options.measurement.transfer_repetitions;
                ninfer::runtime::upsert_context_transfer_cost_atomic(
                    options.preset_output, hardware_class, runtime_transfer_cost(*transfer_fits),
                    provenance.dump());
            }
            if (prefill_fit) {
                Json provenance                   = base_provenance;
                provenance["suite"]               = "prefill";
                provenance["artifact_path"]       = artifact->path.string();
                provenance["target"]              = artifact->target_key;
                provenance["corpus"]              = options.measurement.corpus.string();
                provenance["prefill_chunk"]       = options.measurement.prefill_chunk;
                provenance["max_context"]         = options.measurement.max_context;
                provenance["prefill_repetitions"] = options.measurement.prefill_repetitions;
                ninfer::runtime::upsert_context_prefill_cost_atomic(
                    options.preset_output,
                    ninfer::runtime::ContextCostIdentity{.hardware_class = hardware_class,
                                                         .model_id       = artifact->model_id,
                                                         .weights_id     = artifact->weights_id},
                    runtime_prefill_cost(*prefill_fit), provenance.dump());
            }
            std::cerr << "wrote preset " << options.preset_output.string() << '\n';
        }
        std::cerr << "context-cost calibration accepted\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_context_cost_bench: " << error.what() << '\n';
        return 1;
    }
}
