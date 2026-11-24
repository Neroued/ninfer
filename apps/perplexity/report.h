#pragma once

#include "corpus.h"
#include "evaluation.h"
#include "options.h"

#include <nlohmann/json.hpp>

#include <iosfwd>
#include <map>

namespace ninfer::perplexity {

struct WindowResult {
    WindowPlan window;
    ScoreAggregate score;
    double seconds = 0;
};

struct StreamResult {
    std::string id, domain;
    std::filesystem::path path;
    std::size_t input_tokens = 0;
    ScoreAggregate score;
    double seconds = 0;
    std::vector<WindowResult> windows;
};

struct EvaluationResult {
    std::vector<StreamResult> streams;
    std::map<std::string, ScoreAggregate> domains;
    ScoreAggregate overall;
    double preflight_seconds = 0, scoring_seconds = 0, total_seconds = 0;
};

struct RunDescription {
    product::StartupConfigurationView configuration;
    EngineDescription engine;
    StartupReport startup;
    EngineSnapshot initial;
};

[[nodiscard]] std::filesystem::path prepare_output_directory(const Options&, const RunDescription&,
                                                             const CorpusSelection&);
[[nodiscard]] nlohmann::json make_report(const Options&, const RunDescription&,
                                         const CorpusSelection&, const EvaluationResult&);
void write_report(const std::filesystem::path& directory, const nlohmann::json& report,
                  std::ostream& summary);

} // namespace ninfer::perplexity
