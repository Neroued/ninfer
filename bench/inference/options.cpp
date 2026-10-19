#include "options.h"

#include <charconv>
#include <limits>
#include <stdexcept>

namespace ninfer::bench {
namespace {

constexpr product::CommandLineOption kOptions[] = {
    {"--prompt"}, {"--messages"}, {"--concurrency"}, {"--repetitions"},
    {"--warmup"}, {"--output"},   {"--output-file"}, {"--profile-measured", false},
};

std::uint32_t count(const product::CommandLineArgument& argument, bool allow_zero = false) {
    std::uint32_t value     = 0;
    const auto& text        = argument.value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || (!value && !allow_zero))
        throw std::invalid_argument(argument.flag + " requires " +
                                    (allow_zero ? "a nonnegative" : "a positive") + " integer");
    return value;
}

} // namespace

Options parse_options(int argc, char** argv) {
    auto parsed = product::parse_launch_options(argc, argv, product::EntryPoint::Bench, kOptions);
    Options options;
    options.help_requested = parsed.help_requested;
    options.launch         = std::move(parsed.launch);
    if (options.help_requested) return options;
    std::optional<std::uint32_t> concurrency;
    for (const auto& argument : parsed.extra_arguments) {
        const auto& flag = argument.flag;
        if (flag == "--prompt" || flag == "--messages") {
            if (argument.value.empty()) throw std::invalid_argument(flag + " must not be empty");
            options.inputs.push_back(
                {flag == "--prompt" ? InputKind::Prompt : InputKind::Messages, argument.value});
        } else if (flag == "--concurrency")
            concurrency = count(argument);
        else if (flag == "--repetitions")
            options.repetitions = count(argument);
        else if (flag == "--warmup")
            options.warmup = count(argument, true);
        else if (flag == "--profile-measured")
            options.profile_measured = true;
        else if (flag == "--output-file") {
            if (argument.value.empty()) throw std::invalid_argument(flag + " must not be empty");
            options.output_file = argument.value;
        } else if (flag == "--output") {
            if (argument.value == "table")
                options.output = OutputFormat::Table;
            else if (argument.value == "json")
                options.output = OutputFormat::Json;
            else if (argument.value == "csv")
                options.output = OutputFormat::Csv;
            else
                throw std::invalid_argument("--output must be table, json or csv");
        }
    }
    if (options.inputs.empty()) throw std::invalid_argument("provide --prompt or --messages");
    if (options.inputs.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("too many inputs");
    options.concurrency = concurrency.value_or(static_cast<std::uint32_t>(options.inputs.size()));
    if (options.inputs.size() != 1 && options.inputs.size() != options.concurrency)
        throw std::invalid_argument("multiple inputs require --concurrency to equal their count");
    if (!options.launch.provenance.fields.contains("engine.max_concurrency"))
        options.launch.engine.max_concurrency = options.concurrency;
    if (options.profile_measured && options.repetitions != 1)
        throw std::invalid_argument("--profile-measured requires --repetitions 1");
    if (options.launch.generation.max_tokens == 0)
        throw std::invalid_argument(
            "benchmark --max-tokens must be positive; use 1 for prefill only");
    return options;
}

std::string usage_text(std::string_view executable) {
    return product::startup_usage_text(executable, product::EntryPoint::Bench) + R"(
benchmark:
  --prompt TEXT / --messages FILE
      Repeat to supply independent requests, in order. Messages use the CLI conversation format.
      Inputs are explicit command-line arguments; the TOML [cli] input section is not used.
  --concurrency N
      Default: input count. A single input is copied N times; otherwise N must equal input count.
      Engine max-concurrency defaults to N unless explicitly configured. Native scheduling applies.
  --max-tokens N
      Native per-request output limit, not a decode-step count. EOS is honored. Use 1 for prefill.
  --warmup N          Full, unmeasured generations (default 1).
  --repetitions N     Full measured generations (default 3).
  --output table|json|csv
  --output-file FILE  Write the report to FILE instead of stdout.
  --profile-measured  CUDA profiler range around one measured run; requires --repetitions 1.

Each run starts with the same inputs and sampling seeds; runtime caches stay warm.
Preparation is outside the measured run. Prefix reuse defaults off; explicit configuration wins.
Decode step latency covers one native batch execution, not one output token.
Result/resolution/retirement and Host times use existing Engine observations.
For external profiling, use nsys --capture-range=cudaProfilerApi with --profile-measured.
)";
}

} // namespace ninfer::bench
