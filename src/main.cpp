#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/cli.h"
#include "qus/text/text_runner.h"
#include "qus/text/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::string format_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds << " s";
    return out.str();
}

std::string format_tok_s(double tokens, double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (seconds > 0.0 && tokens > 0.0) {
        out << (tokens / seconds) << " tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string format_percent(std::uint64_t done, std::uint64_t total) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (total == 0) {
        out << "100.0%";
    } else {
        out << (100.0 * static_cast<double>(done) / static_cast<double>(total)) << '%';
    }
    return out.str();
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(10) << group << std::setw(24) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_progress(std::string_view detail, std::uint64_t done, std::uint64_t total,
                    double seconds) {
    std::cerr << std::left << std::setw(10) << "load" << std::setw(24) << detail << std::right
              << std::setw(8) << format_percent(done, total) << std::setw(14)
              << format_bytes(done) << " / " << std::setw(14) << format_bytes(total)
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(10) << "summary" << std::setw(24) << label << value
              << '\n';
}

struct ProgressState {
    Clock::time_point start;
    std::uint64_t last_done = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last_total = std::numeric_limits<std::uint64_t>::max();
};

} // namespace

int main(int argc, char** argv) {
    try {
        const qus::text::CliOptions cli = qus::text::parse_cli(argc, argv);
        if (cli.help_requested) {
            std::cout << qus::text::usage_text(argv[0]);
            return 0;
        }

        std::cerr << "phase     detail                    elapsed/progress\n";
        const auto tokenizer_start = Clock::now();
        qus::text::QwenTokenizer tokenizer(cli.tokenizer_path);
        print_stage("load", "tokenizer", std::chrono::duration<double>(Clock::now() -
                                                                        tokenizer_start)
                                             .count());

        const std::vector<int> stop_token_ids =
            qus::text::resolve_stop_token_ids(tokenizer, cli.stop_token_ids);

        std::map<std::string, ProgressState> progress_states;
        qus::Q5090Progress progress;
        progress.min_interval_bytes = 1024ULL * 1024ULL * 1024ULL;
        progress.callback = [&](std::string_view phase, std::uint64_t done,
                                std::uint64_t total) {
            const std::string key(phase);
            const auto now = Clock::now();
            auto [it, inserted] = progress_states.emplace(key, ProgressState{now});
            if (!inserted && it->second.last_done == done && it->second.last_total == total) {
                return;
            }
            it->second.last_done  = done;
            it->second.last_total = total;
            print_progress(key, done, total,
                           std::chrono::duration<double>(now - it->second.start).count());
        };

        qus::EngineOptions engine_options;
        engine_options.device         = cli.device;
        engine_options.max_ctx        = cli.max_context;
        engine_options.stop_token_ids = stop_token_ids;
        engine_options.progress       = &progress;
        engine_options.use_cuda_graph = cli.use_cuda_graph;
        qus::Engine engine(engine_options);

        const auto engine_load_start = Clock::now();
        engine.load(cli.weights_path);
        print_stage("load", "engine total", std::chrono::duration<double>(Clock::now() -
                                                                          engine_load_start)
                                               .count());

        const std::vector<qus::text::ChatMessage> messages =
            cli.messages_path.empty() ? qus::text::messages_from_prompt(cli.prompt)
                                      : qus::text::read_messages_json(cli.messages_path);

        qus::text::TextGenerationOptions generation_options;
        generation_options.max_new_tokens = cli.max_new;
        generation_options.raw_output     = cli.output_mode == qus::text::OutputMode::Raw;
        generation_options.stop_token_ids = stop_token_ids;
        generation_options.stream_callback = [](const qus::text::TextStreamChunk& chunk) {
            std::cout << chunk.text;
            std::cout.flush();
        };

        qus::text::TextGenerationRunner runner(tokenizer, engine);

        const qus::text::TextGenerationResult result = runner.generate(messages, generation_options);

        if (result.text.empty() || result.text.back() != '\n') { std::cout << '\n'; }
        std::cout.flush();

        print_stage("prompt", "render/tokenize", result.timings.render_tokenize_seconds);
        print_stage("generate", "prefill", result.timings.prefill_seconds);
        print_stage("generate", "decode", result.timings.decode_seconds);
        print_stage("generate", "total", result.timings.total_seconds);

        if (cli.print_token_ids) {
            std::cerr << std::left << std::setw(10) << "tokens" << std::setw(24)
                      << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }

        // Prefill processes the whole prompt and emits the first token; the decode
        // loop emits the remaining (generated - 1) tokens. Report the two phases
        // separately so prefill (prompt ingestion) and decode (generation) speeds
        // are not blended into one number.
        const std::size_t prompt_tokens    = result.prompt_token_ids.size();
        const std::size_t generated_tokens = result.generated_token_ids.size();
        const std::size_t decode_tokens    = generated_tokens > 0 ? generated_tokens - 1 : 0;
        const double      prefill_seconds  = result.timings.prefill_seconds;
        const double      decode_seconds   = result.timings.decode_seconds;
        const double      seconds          = prefill_seconds + decode_seconds;

        print_metric("prompt tokens", std::to_string(prompt_tokens));
        print_metric("generated tokens", std::to_string(generated_tokens));
        print_metric("model elapsed", format_seconds(seconds));
        print_metric("prefill speed",
                     format_tok_s(static_cast<double>(prompt_tokens), prefill_seconds));
        print_metric("decode speed",
                     format_tok_s(static_cast<double>(decode_tokens), decode_seconds));
        print_metric("throughput (overall)",
                     format_tok_s(static_cast<double>(generated_tokens), seconds));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << qus::text::usage_text(argv[0]);
        return 1;
    }

    return 0;
}
