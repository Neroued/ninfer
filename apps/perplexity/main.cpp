#include "corpus.h"
#include "evaluation.h"
#include "options.h"
#include "report.h"

#include "ninfer/engine.h"
#include "product/logging/configuration.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace ninfer::perplexity;

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    auto engine_options                   = options.launch.engine;
    engine_options.startup_observer       = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const RunDescription description{ninfer::product::configuration_view(
                                         options.launch, ninfer::product::EntryPoint::Perplexity),
                                     engine.description(), engine.startup_report(),
                                     engine.snapshot()};
    startup_log.engine_ready(description.startup.load);
    ninfer::product::log_resolved_configuration(*logger, description.engine);
    logger->info("scoring | window {} | stride {}", options.context, options.stride);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory =
        prepare_output_directory(options, description, corpus);
    const Clock::time_point scoring_started = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    EvaluationResult result;
    result.preflight_seconds = preflight_seconds;
    result.streams.reserve(streams.size());
    auto& overall                   = result.overall;
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        StreamResult stream_result{.id           = stream.source.id,
                                   .domain       = stream.source.domain,
                                   .path         = stream.source.path,
                                   .input_tokens = stream.tokens.size()};
        stream_result.windows.reserve(stream.windows.size());
        auto& stream_score = stream_result.score;
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            try {
                logprobs = engine.score_tokens(std::move(input), window.first_target);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            result.domains[stream.source.domain].add(window_score);
            ++completed_windows;
            stream_result.windows.push_back({window, window_score, seconds_since(window_started)});

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        stream_result.seconds = stream_seconds;
        result.streams.push_back(std::move(stream_result));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    result.scoring_seconds = scoring_seconds;
    result.total_seconds   = seconds_since(total_started);
    write_report(output_directory, make_report(options, description, corpus, result), std::cout);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::perplexity::Options options;
    try {
        options = ninfer::perplexity::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        std::cerr << ninfer::perplexity::usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::perplexity::usage_text();
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-perplexity",
         .level        = options.launch.logging.level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        ninfer::product::log_configuration(
            *logger, ninfer::product::configuration_view(options.launch,
                                                         ninfer::product::EntryPoint::Perplexity));
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
