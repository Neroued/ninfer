#include "report.h"

#include "product/statistics.h"

#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ninfer::perplexity {
namespace {

using Json = nlohmann::json;

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value)
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

Json aggregate_json(const ScoreAggregate& value) {
    return {{"scored_tokens", value.scored_tokens},
            {"total_nll", value.total_nll},
            {"mean_nll", value.mean_nll()},
            {"perplexity", value.ppl()}};
}

} // namespace

std::filesystem::path prepare_output_directory(const Options& options, const RunDescription& run,
                                               const CorpusSelection& corpus) {
    const auto& load = run.startup.load;
    auto output      = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_cache_profile_info(run.engine.kv_cache).name /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output))
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
    } else if (!std::filesystem::create_directories(output))
        throw std::runtime_error("cannot create output directory: " + output.string());
    return std::filesystem::absolute(output).lexically_normal();
}

Json make_report(const Options& options, const RunDescription& run, const CorpusSelection& corpus,
                 const EvaluationResult& result) {
    const auto& load = run.startup.load;
    Json streams = Json::array(), domains = Json::array();
    for (const auto& stream : result.streams) {
        auto item               = aggregate_json(stream.score);
        item["id"]              = stream.id;
        item["domain"]          = stream.domain;
        item["path"]            = stream.path.string();
        item["input_tokens"]    = stream.input_tokens;
        item["unscored_tokens"] = 1;
        item["seconds"]         = stream.seconds;
        auto& windows = item["windows"] = Json::array();
        for (std::size_t i = 0; i < stream.windows.size(); ++i) {
            const auto& value = stream.windows[i];
            const auto& w     = value.window;
            auto window       = aggregate_json(value.score);
            window.update({{"index", i},
                           {"input_begin", w.input_begin},
                           {"input_end", w.input_end},
                           {"target_begin", w.target_begin},
                           {"target_end", w.target_end},
                           {"first_target", w.first_target},
                           {"seconds", value.seconds}});
            windows.push_back(std::move(window));
        }
        streams.push_back(std::move(item));
    }
    for (const auto& [domain, score] : result.domains) {
        auto item      = aggregate_json(score);
        item["domain"] = domain;
        domains.push_back(std::move(item));
    }
    Json configuration = Json::object();
    for (const auto& item : run.configuration.items)
        std::visit([&](const auto& value) { configuration[item.key] = value; }, item.value);
    return {
        {"schema_version", 2},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", options.launch.engine.artifact_path.string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"argv", run.configuration.argv},
        {"config_path", run.configuration.config_path.string()},
        {"configuration", std::move(configuration)},
        {"environment", run.configuration.environment},
        {"startup", product::startup_resources_json(run.engine, run.startup, run.initial)},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", result.streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"context_tokens", options.context},
          {"stride_tokens", options.stride},
          {"max_context_tokens", run.engine.max_context},
          {"prefill_chunk_tokens", run.engine.prefill_chunk},
          {"kv_dtype", kv_cache_profile_info(run.engine.kv_cache).name},
          {"context_pools", product::context_capacity_json(run.engine.context_pools)},
          {"context_token_capacity_estimate", run.engine.context_token_capacity_estimate}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", result.preflight_seconds},
          {"score_seconds", result.scoring_seconds},
          {"total_seconds", result.total_seconds},
          {"scored_tokens_per_second",
           product::rate(result.overall.scored_tokens, result.scoring_seconds)}}},
        {"streams", std::move(streams)},
        {"domains", std::move(domains)},
        {"overall", aggregate_json(result.overall)},
    };
}

void write_report(const std::filesystem::path& directory, const Json& report,
                  std::ostream& summary) {
    const auto temporary = directory / "report.json.tmp";
    const auto final     = directory / "report.json";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("cannot create report: " + temporary.string());
        file << std::setw(2) << report << '\n';
        file.close();
        if (!file) throw std::runtime_error("cannot write report: " + temporary.string());
    }
    std::filesystem::rename(temporary, final);

    const auto& execution = report.at("execution");
    summary << "Perplexity result\nartifact: "
            << report.at("artifact").at("name").get<std::string>()
            << "\nkv: " << execution.at("kv_dtype").get<std::string>()
            << ", corpus: " << report.at("corpus").at("id").get<std::string>() << " / "
            << report.at("corpus").at("mode").get<std::string>()
            << ", context/stride: " << execution.at("context_tokens") << '/'
            << execution.at("stride_tokens")
            << "\nranks: " << report.at("startup").at("ranks").size() << "\n\n";
    summary << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
            << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    const auto row = [&](std::string_view label, const Json& score) {
        summary << std::left << std::setw(24) << label << std::right << std::setw(16)
                << score.at("scored_tokens").get<std::uint64_t>() << std::setw(16) << std::fixed
                << std::setprecision(6) << score.at("mean_nll").get<double>() << std::setw(16)
                << score.at("perplexity").get<double>() << '\n';
    };
    for (const auto& domain : report.at("domains"))
        row(domain.at("domain").get<std::string>(), domain);
    row("overall", report.at("overall"));
    summary << "\nscore rate: " << std::setprecision(1)
            << report.at("timing").at("scored_tokens_per_second").get<double>()
            << " tok/s\nreport: " << final << '\n';
}

} // namespace ninfer::perplexity
