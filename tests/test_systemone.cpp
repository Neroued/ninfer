#include "serve/systemone.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using ninfer::decision::Json;
using ninfer::serve::parse_systemone_request;
using ninfer::serve::format_systemone_response;

namespace {
int checks = 0;

void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) { throw std::runtime_error(message); }
}

template <typename F>
void rejects(F&& action, const char* message) {
    bool failed = false;
    try {
        action();
    } catch (const std::exception&) { failed = true; }
    expect(failed, message);
}

bool near(double actual, double expected) { return std::abs(actual - expected) < 1e-12; }

Json example() {
    return Json::parse(R"({"state":{"message":"Charged twice","amount":42},"questions":{
      "route_private_id":{"type":"choice","instructions":{"ask":"Which team?"},
          "criteria":{"billing":"Money","technical":{"area":"Software"},"other":null}},
      "urgent_private_id":{"type":"noul","instructions":"Urgent?","criteria":{"false":"No rush"}},
      "score_private_id":{"type":"score","instructions":"Severity?","criteria":["Low","Medium","High"]}
    }})");
}

Json scored(const ninfer::serve::SystemOneRequest& request,
            const std::vector<std::vector<double>>& p) {
    Json fields = Json::object();
    for (std::size_t i = 0; i < request.questions.size(); ++i) {
        fields[request.questions[i].id] = {{"probabilities", p.at(i)}};
    }
    return {{"results", Json::array({{{"fields", fields}}})}, {"usage", {{"prompt_tokens", 123}}}};
}
} // namespace

int main() {
    try {
        const auto body    = example();
        const auto request = parse_systemone_request(body);
        expect(request.questions.size() == 3 && request.decision.contexts.size() == 1,
               "mixed questions compile into one state");
        expect(request.decision.mode == "tree" && request.decision.retain_probabilities,
               "full distributions always use exhaustive scoring");
        expect(request.decision.contexts[0] == "State:\n" + body["state"].dump(),
               "structured state preserved");
        expect(request.decision.fields[0].isolated_question.find("Software") != std::string::npos,
               "structured choice descriptions reach the question");
        const auto response = format_systemone_response(
            request, scored(request, {{.6, .3, .1}, {.1, .9}, {0, .6, .4}}), "local-27b");
        const auto& answers = response.at("answers");
        expect(response.at("model") == "local-27b" && response.at("usage").at("output_tokens") == 0,
               "report the actual model and no generated tokens");
        expect(response.at("usage").at("input_tokens") == 123, "native usage preserved");
        expect(answers.at("route_private_id").at("choice") == "billing", "choice argmax");
        expect(near(answers.at("route_private_id").at("confidence"), .4),
               "published choice confidence");
        expect(answers.at("route_private_id").at("probabilities").size() == 3,
               "full choice distribution");
        expect(near(answers.at("urgent_private_id").at("noul"), .1),
               "noul is p(true), not winning probability");
        expect(answers.at("urgent_private_id").size() == 2, "no invented noul confidence field");
        expect(near(answers.at("score_private_id").at("score"), 1.4),
               "score mean is not grid rounded");
        expect(near(answers.at("score_private_id").at("confidence"), .4),
               "published score confidence");
        expect(answers.at("score_private_id").at("legend").at("2") == "High",
               "score legend ordering");

        for (const auto& [p, expected] :
             std::vector<std::pair<std::vector<double>, double>>{{{0, .5, .5}, .25},
                                                                 {{.5, 0, .5}, 0},
                                                                 {{1.0 / 3, 1.0 / 3, 1.0 / 3}, 0},
                                                                 {{0, 0, 1}, 1}}) {
            const auto value = format_systemone_response(
                request, scored(request, {{1, 0, 0}, {.5, .5}, p}), "local");
            expect(near(value["answers"]["score_private_id"]["confidence"], expected),
                   "score spread oracle");
        }
        auto single         = body;
        single["questions"] = {{"pick", {{"type", "choice"}, {"criteria", {{"only", nullptr}}}}}};
        const auto one      = parse_systemone_request(single);
        expect(format_systemone_response(one, scored(one, {{1}}),
                                         "local")["answers"]["pick"]["confidence"] == 1,
               "singleton choice confidence does not divide by zero");
        auto reserved_id            = single;
        reserved_id["questions"]    = {{"properties", single["questions"]["pick"]}};
        const auto reserved_request = parse_systemone_request(reserved_id);
        expect(format_systemone_response(reserved_request, scored(reserved_request, {{1}}),
                                         "local")["answers"]["properties"]["choice"] == "only",
               "question IDs cannot be mistaken for schema keywords");
        auto structured = body;
        structured["questions"]["score_private_id"]["criteria"] =
            Json::array({Json{{"level", "low"}}, nullptr});
        const auto nested          = parse_systemone_request(structured);
        const auto nested_response = format_systemone_response(
            nested, scored(nested, {{1, 0, 0}, {1, 0}, {.5, .5}}), "local");
        expect(nested_response["answers"]["score_private_id"]["legend"]["0"] ==
                   Json{{"level", "low"}},
               "SDK structured score legend preserved");
        expect(nested_response["answers"]["score_private_id"]["legend"]["1"].is_null(),
               "null level preserved");

        auto changed     = body;
        changed["state"] = Json::array({"first", Json{{"second", 2}}});
        expect(parse_systemone_request(changed).decision.contexts.size() == 1,
               "array state is not a context batch");
        changed["state"]                                          = nullptr;
        changed["questions"]["urgent_private_id"]["instructions"] = nullptr;
        changed["questions"]["urgent_private_id"]["criteria"]     = nullptr;
        expect(parse_systemone_request(changed).decision.contexts[0] == "State:\nnull",
               "SDK null entries accepted");
        for (const auto& value : {Json("description"), Json{{"what", "a condition"}},
                                  Json::array({"first", "second"}), Json(nullptr)}) {
            changed = body;
            for (auto& question : changed["questions"]) { question["instructions"] = value; }
            changed["questions"]["route_private_id"]["criteria"]["billing"] = value;
            changed["questions"]["urgent_private_id"]["criteria"]["true"]   = value;
            changed["questions"]["score_private_id"]["criteria"][0]         = value;
            const auto structured_request = parse_systemone_request(changed);
            for (const auto& field : structured_request.decision.fields) {
                const auto rubric = Json::parse(field.isolated_question);
                expect(rubric["instructions"] == value,
                       "every instruction entry shape survives protocol translation");
                expect(rubric["criteria"] == changed["questions"][field.name]["criteria"],
                       "every criterion entry shape survives protocol translation");
            }
        }

        // The real compiler path uses identical prompts when a caller renames question IDs,
        // and adding an unrelated question cannot inject its text into another field.
        ninfer::decision::Backend backend;
        std::vector<std::string> prompts;
        std::vector<ninfer::decision::Tokens> scored_prefixes;
        std::vector<std::vector<std::uint32_t>> scored_frontiers;
        backend.tokenize = [&](const std::string& system, const std::string& context,
                               const std::string& continuation) {
            prompts.push_back(system + "\n" + context + "\n" + continuation);
            return ninfer::decision::Tokens(prompts.back().begin(), prompts.back().end());
        };
        backend.check_cancelled = [] {};
        std::size_t batches     = 0;
        backend.score           = [&](std::vector<ninfer::decision::ScoreRow> rows, bool) {
            ++batches;
            ninfer::decision::ScoreBatch result;
            for (const auto& row : rows) {
                scored_prefixes.emplace_back(row.prefix.begin(), row.prefix.end());
                scored_frontiers.push_back(row.cache_frontiers);
                result.logits.emplace_back(row.candidates.size(), 0.0f);
            }
            return result;
        };
        auto native = ninfer::decision::evaluate(request.decision, backend);
        expect(batches == 1, "questions still batch in one host scoring submission");
        expect(format_systemone_response(request, native, "local")["answers"].size() == 3,
               "compiler-to-formatter integration");
        const std::string state_root =
            request.decision.system + "\n" + request.decision.contexts.front() + "\n";
        for (std::size_t i = 0; i < scored_frontiers.size(); ++i) {
            expect(scored_frontiers[i] ==
                       std::vector<std::uint32_t>{
                           static_cast<std::uint32_t>(state_root.size()),
                           static_cast<std::uint32_t>(scored_prefixes[i].size())},
                   "reuse the state root and entire isolated question prefix");
            expect(std::string(scored_prefixes[i].begin(),
                               scored_prefixes[i].begin() + state_root.size()) == state_root,
                   "root token identity shared by every question");
        }
        const auto original_prompts = prompts;
        for (const auto& prompt : prompts) {
            expect(prompt.find("private_id") == std::string::npos,
                   "caller IDs leaked into inference");
            expect(prompt.find("Urgent?") == std::string::npos ||
                       prompt.find("Which team?") == std::string::npos,
                   "another question leaked into input");
        }
        changed              = body;
        changed["questions"] = Json::object();
        int index            = 0;
        for (const auto& question : body["questions"]) {
            changed["questions"][std::to_string(index++)] = question;
        }
        prompts.clear();
        (void)ninfer::decision::evaluate(parse_systemone_request(changed).decision, backend);
        expect(prompts == original_prompts, "renaming IDs changes tokenization");
        changed["questions"].erase("1");
        changed["questions"].erase("2");
        prompts.clear();
        (void)ninfer::decision::evaluate(parse_systemone_request(changed).decision, backend);
        expect(std::equal(prompts.begin(), prompts.end(), original_prompts.begin()),
               "removing unrelated questions changes input");
        const auto old_request  = ninfer::decision::parse_request(Json::parse(
            R"({"contexts":["hello"],"schema":{"yes":{"type":"boolean","description":"A greeting?"}}})"));
        const auto old_response = ninfer::decision::evaluate(old_request, backend);
        expect(!old_response["results"][0]["fields"]["yes"].contains("probabilities"),
               "ordinary decision response has not gained private formatter fields");

        auto branching = parse_systemone_request(Json::parse(R"({"state":"state text","questions":{
          "pick":{"type":"choice","instructions":"Select a label","criteria":{
            "aa":"First","ab":"Second","ba":"Third","bb":"Fourth"}}}})"));
        scored_prefixes.clear();
        scored_frontiers.clear();
        const auto cached = ninfer::decision::evaluate(branching.decision, backend);
        expect(scored_frontiers.size() == 3, "multi-token labels exercise three divergences");
        expect(scored_frontiers[0] == scored_frontiers[1] &&
                   scored_frontiers[1] == scored_frontiers[2] &&
                   scored_frontiers[0].back() == scored_prefixes[0].size(),
               "each divergence reuses one complete rubric rather than recomputing it");
        branching.decision.cache_prompt = false;
        scored_prefixes.clear();
        scored_frontiers.clear();
        const auto uncached = ninfer::decision::evaluate(branching.decision, backend);
        expect(std::all_of(scored_frontiers.begin(), scored_frontiers.end(),
                           [](const auto& points) { return points.empty(); }),
               "cache opt-out also disables isolated-question frontiers");
        expect(cached["results"] == uncached["results"], "reuse hints cannot change host answers");

        auto invalid = body;
        for (const auto& bad : {Json(123), Json(true)}) {
            invalid["state"] = bad;
            rejects([&] { (void)parse_systemone_request(invalid); }, "reject scalar state");
        }
        invalid = body;
        invalid.erase("state");
        rejects([&] { (void)parse_systemone_request(invalid); }, "state required");
        invalid         = body;
        invalid["mode"] = "greedy";
        rejects([&] { (void)parse_systemone_request(invalid); }, "no incomplete distribution mode");
        for (const auto& criteria : {Json::array({"one"}), Json::array(), Json::object()}) {
            invalid                                              = body;
            invalid["questions"]["score_private_id"]["criteria"] = criteria;
            rejects([&] { (void)parse_systemone_request(invalid); }, "invalid score rubric");
        }
        invalid = body;
        invalid["questions"]["score_private_id"]["criteria"] =
            std::vector<std::string>(11, "level");
        rejects([&] { (void)parse_systemone_request(invalid); }, "score maximum ten levels");
        invalid                                              = body;
        invalid["questions"]["route_private_id"]["criteria"] = Json::object();
        for (int i = 0; i < 255; ++i) {
            invalid["questions"]["route_private_id"]["criteria"][std::to_string(i)] = nullptr;
        }
        expect(parse_systemone_request(invalid).decision.fields[0].values.size() == 255,
               "255 choices supported exactly");
        invalid["questions"]["route_private_id"]["criteria"]["overflow"] = nullptr;
        rejects([&] { (void)parse_systemone_request(invalid); }, "choice bound");
        invalid                                               = body;
        invalid["questions"]["urgent_private_id"]["criteria"] = {{"yes", "Wrong key"}};
        rejects([&] { (void)parse_systemone_request(invalid); },
                "noul criteria are boolean outcomes");
        invalid                                                  = body;
        invalid["questions"]["route_private_id"]["instructions"] = 3;
        rejects([&] { (void)parse_systemone_request(invalid); }, "entry type validated");
        invalid              = body;
        invalid["questions"] = Json::object();
        rejects([&] { (void)parse_systemone_request(invalid); }, "nonempty questions");
        for (int i = 0; i < 32; ++i) {
            invalid["questions"][std::to_string(i)] = {{"type", "noul"}};
        }
        expect(parse_systemone_request(invalid).questions.size() == 32,
               "maximum supported question batch accepted");
        invalid["questions"]["overflow"] = {{"type", "noul"}};
        rejects([&] { (void)parse_systemone_request(invalid); }, "question bound");
        for (const auto& p :
             {std::vector<double>{.5}, std::vector<double>{.5, .6, 0},
              std::vector<double>{std::numeric_limits<double>::quiet_NaN(), 0, 1}}) {
            rejects(
                [&] {
                    (void)format_systemone_response(
                        request, scored(request, {p, {1, 0}, {1, 0, 0}}), "local");
                },
                "invalid backend distribution cannot become a successful response");
        }
        std::cout << checks << " systemone contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "systemone test failed: " << error.what() << '\n';
        return 1;
    }
}
