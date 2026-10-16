#include "decision/decision.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ninfer::decision;

#ifndef NINFER_SOURCE_DIR
#define NINFER_SOURCE_DIR "."
#endif

namespace {
void expect(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

Tokens bytes(const std::string& text) { return Tokens(text.begin(), text.end()); }
} // namespace

int main(int argc, char** argv) {
    try {
        std::ifstream input(argc > 1 ? argv[1] :
            NINFER_SOURCE_DIR "/tests/fixtures/decision/codacus.json");
        expect(input.good(), "cannot open Codacus fixtures");
        const auto corpus = Json::parse(input);
        const auto& fixtures = corpus.at("presets");
        expect(fixtures.size() == 8, "expected all eight Codacus presets");
        const std::vector<std::size_t> counts = {5, 3, 3, 12, 18, 10, 8, 30};
        std::size_t cases = 0;
        Backend backend;
        // A byte tokenizer and uniform next-edge logits test the compiler/assembler only.
        // They are deliberately not a language model or semantic answer oracle.
        backend.tokenize = [](const std::string& system, const std::string& context,
                              const std::string& assistant) {
            return bytes(system + "\n<context>" + context + "\n<assistant>" + assistant);
        };
        backend.check_cancelled = [] {};
        backend.score = [](std::vector<ScoreRow> rows, bool) {
            ScoreBatch batch;
            for (const auto& row : rows) {
                batch.logits.emplace_back(row.candidates.size(), 0.0f);
                batch.computed_tokens += row.prompt_size();
            }
            return batch;
        };
        for (std::size_t i = 0; i < fixtures.size(); ++i) {
            for (const auto* mode : {"auto", "tree", "greedy"}) {
                auto body = fixtures[i].at("request");
                body["mode"] = mode;
                const auto request = parse_request(body);
                expect(request.fields.size() == counts[i], "fixture field count changed");
                expect(request.contexts.size() == (i == 2 ? 8 : 1), "fixture context count changed");
                const auto response = evaluate(request, backend);
                expect(response.at("results").size() == request.contexts.size(), "lost a context");
                for (const auto& result : response.at("results")) {
                    expect(result.at("decision").size() == request.fields.size(), "lost a field");
                    expect(result.at("fields").size() == request.fields.size(), "lost field details");
                    for (const auto& field : request.fields) {
                        const auto& value = result.at("decision").at(field.name);
                        expect(std::find(field.values.begin(), field.values.end(), value) !=
                                   field.values.end(), "value escaped finite domain");
                        const auto& detail = result.at("fields").at(field.name);
                        expect(detail.at("value") == value, "field detail disagrees with decision");
                        const double probability = detail.at("probability");
                        expect(std::isfinite(probability) && probability >= 0 && probability <= 1,
                               "invalid conditional probability");
                        const bool tree = request.mode == "tree" ||
                            (request.mode == "auto" && field.values.size() <= request.tree_max);
                        expect(detail.at("tree") == tree, "wrong tree/greedy selection");
                        expect(detail.contains("interval_p10_p90") ==
                                   (tree && !field.numbers.empty()), "wrong numeric interval contract");
                    }
                }
                ++cases;
            }
        }
        std::cout << cases << " Codacus fixture/mode cases passed (host contract only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision fixture test failed: " << error.what() << '\n';
        return 1;
    }
}
