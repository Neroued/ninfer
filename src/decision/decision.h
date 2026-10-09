#pragma once

#include "ninfer/types.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ninfer {
class Engine;
}

namespace ninfer::decision {

using Json   = nlohmann::ordered_json;
using Tokens = std::vector<TokenId>;

struct Field {
    std::string name;
    std::string description;
    // System One questions append their own rubric after the shared state. Other questions
    // and the caller's answer-map key are never included in this field's model input.
    std::string isolated_question;
    std::string aggregate = "mode";
    std::vector<Json> values;
    std::vector<std::string> encoded;
    std::vector<long double> numbers;
};

struct Request {
    std::string model;
    std::string system;
    std::vector<Field> fields;
    std::vector<std::string> contexts;
    std::string mode     = "auto";
    std::size_t tree_max = 128;
    bool cache_prompt    = true;
    // Internal formatter input only, not a /v1/decision request or response extension.
    bool retain_probabilities = false;
};

// Host-side protocol and finite-tree mathematics, independent of CUDA and serving transport.
[[nodiscard]] Request parse_request(const Json& body);
[[nodiscard]] std::vector<double> log_softmax(std::span<const float> logits);

struct Node {
    Tokens suffix;
    Tokens options;
    std::vector<std::vector<std::size_t>> members;
    // Nonnegative: another divergence node; negative: -(candidate index + 1).
    std::vector<int> children;
};

class CandidateTrie {
public:
    // Complete prompt+value+terminator tokenizations. The common prefix is factored on each
    // insertion, so long shared prompts are stored once, not once per candidate.
    void add(Tokens complete);
    void build();
    [[nodiscard]] Tokens prompt(std::size_t node) const;

    [[nodiscard]] const Tokens& prefix() const noexcept { return prefix_; }

    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }

    [[nodiscard]] std::size_t size() const noexcept { return paths_.size(); }

    [[nodiscard]] std::vector<double>
    distribution(const std::vector<std::vector<float>>& scores) const;

private:
    int build_node(const std::vector<std::size_t>& members, std::size_t depth);
    Tokens prefix_;
    std::vector<Tokens> paths_;
    std::vector<Node> nodes_;
};

struct FieldResult {
    std::size_t winner = 0;
    double probability = 1.0;
    std::vector<double> probabilities;
    std::size_t scored_nodes = 0;
    bool tree                = true;
};

[[nodiscard]] Json assemble(const std::vector<Field>& fields,
                            const std::vector<FieldResult>& results,
                            bool retain_probabilities = false);

struct ScoreRow {
    // Borrowed only for the duration of Backend::score. Shared trunks are not duplicated for
    // every divergence; the native Program packs suffix tiles directly from these spans.
    std::span<const TokenId> prefix;
    std::span<const TokenId> suffix;
    std::span<const TokenId> candidates;
    std::vector<std::uint32_t> cache_frontiers;

    [[nodiscard]] std::size_t prompt_size() const noexcept { return prefix.size() + suffix.size(); }

    [[nodiscard]] Tokens prompt() const {
        Tokens tokens(prefix.begin(), prefix.end());
        tokens.insert(tokens.end(), suffix.begin(), suffix.end());
        return tokens;
    }
};

struct ScoreBatch {
    std::vector<std::vector<float>> logits;
    std::uint64_t computed_tokens = 0;
    std::uint64_t reused_tokens   = 0;
    double prefill_ms             = 0;
    std::uint64_t forward_batches = 0;
    std::uint32_t largest_batch   = 0;
};

// The native adapter supplies artifact-template tokenization and worker-owned branch batches.
// Keeping this host algorithm explicit also allows deterministic tests without model weights.
struct Backend {
    std::function<Tokens(const std::string&, const std::string&, const std::string&)> tokenize;
    std::function<ScoreBatch(std::vector<ScoreRow>, bool)> score;
    std::function<void()> check_cancelled;
};

[[nodiscard]] Json evaluate(const Request& request, const Backend& backend);

class DecisionEngine {
public:
    explicit DecisionEngine(Engine& engine) : engine_(engine) {}

    [[nodiscard]] Json run(const Request& request, const PreparationControl& control = {});
private:
    Engine& engine_;
};

} // namespace ninfer::decision
