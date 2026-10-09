#include "decision/decision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ninfer::decision {
namespace {

constexpr std::size_t kMaximumTextBytes = 1U << 20;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void keys(const Json& object, std::initializer_list<std::string_view> allowed,
          std::string_view where) {
    require(object.is_object(), std::string(where) + " must be an object");
    for (const auto& [key, value] : object.items()) {
        (void)value;
        require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(),
                std::string(where) + ": unsupported member " + key);
    }
}

std::string text(const Json& object, const char* key, std::string fallback = {}) {
    if (!object.contains(key)) { return fallback; }
    require(object.at(key).is_string(), std::string(key) + " must be a string");
    return object.at(key).get<std::string>();
}

bool boolean(const Json& object, const char* key, bool fallback) {
    if (!object.contains(key)) { return fallback; }
    require(object.at(key).is_boolean(), std::string(key) + " must be a boolean");
    return object.at(key).get<bool>();
}

std::int64_t integer(const Json& object, const char* key) {
    require(object.contains(key) && object.at(key).is_number_integer(),
            std::string(key) + " must be an integer");
    const auto& value = object.at(key);
    require(!value.is_number_unsigned() ||
                value.get<std::uint64_t>() <= static_cast<std::uint64_t>(INT64_MAX),
            std::string(key) + " exceeds int64");
    return value.get<std::int64_t>();
}

double number(const Json& object, const char* key) {
    require(object.contains(key) && object.at(key).is_number(),
            std::string(key) + " must be a finite number");
    const auto value = object.at(key).get<double>();
    require(std::isfinite(value), std::string(key) + " must be finite");
    return value;
}

int decimal_places(double value) {
    for (int digits = 0; digits <= 9; ++digits) {
        const auto scaled = value * std::pow(10.0, digits);
        const auto tolerance =
            8 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(scaled));
        if (std::isfinite(scaled) && std::abs(scaled - std::round(scaled)) <= tolerance) {
            return digits;
        }
    }
    throw std::invalid_argument("number grids support at most nine decimal places");
}

std::size_t common_size(std::span<const TokenId> a, std::span<const TokenId> b) {
    std::size_t count = 0;
    while (count < std::min(a.size(), b.size()) && a[count] == b[count]) { ++count; }
    return count;
}

Field compile_field(const std::string& name, const Json& spec, bool json_schema) {
    keys(spec,
         {"type", "description", "title", "enum", "choices", "minimum", "maximum", "step",
          "multipleOf", "aggregate", "x-aggregate"},
         "schema field");
    require(!name.empty() && name.size() <= 4096, "field names must contain 1-4096 bytes");
    Field field;
    field.name        = name;
    field.description = text(spec, "description");
    require(json_schema || !field.description.empty(), "field " + name + " needs a description");
    std::string type = text(spec, "type");
    if (spec.contains("enum")) {
        require(type.empty() || type == "string" || type == "enum", "enum fields must be strings");
        type = "enum";
    }
    const bool choice = type == "enum" || type == "choice" || type == "selection";
    require(!(spec.contains("enum") && spec.contains("choices")), "use enum or choices, not both");
    require(choice || !spec.contains("choices"), "choices require an enum field");
    require(type == "integer" || type == "number" ||
                (!spec.contains("minimum") && !spec.contains("maximum")),
            "bounds require a numeric field");
    require(type == "number" || (!spec.contains("step") && !spec.contains("multipleOf")),
            "step and multipleOf require a number field");
    require(!(spec.contains("step") && spec.contains("multipleOf")),
            "use step or multipleOf, not both");
    require(!(spec.contains("aggregate") && spec.contains("x-aggregate")),
            "use aggregate or x-aggregate, not both");
    if (type == "boolean") {
        field.values = {true, false};
    } else if (type == "enum" || type == "choice" || type == "selection") {
        const auto* key = spec.contains("enum") ? "enum" : "choices";
        require(spec.contains(key) && spec.at(key).is_array(), "enum fields need choices");
        require(!spec.at(key).empty() && spec.at(key).size() <= 255,
                "enum fields need 1-255 choices");
        for (const auto& value : spec.at(key)) {
            require(value.is_string() && value.get_ref<const std::string&>().size() <= 4096,
                    "enum choices must be strings of at most 4096 bytes");
            field.values.push_back(value);
        }
    } else if (type == "integer") {
        const auto lo = integer(spec, "minimum"), hi = integer(spec, "maximum");
        const auto distance = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
        require(hi >= lo && distance <= 254, "integer bounds must define 1-255 values");
        for (std::uint64_t offset = 0; offset <= distance; ++offset) {
            const auto value = lo + static_cast<std::int64_t>(offset);
            field.values.push_back(value);
            field.numbers.push_back(static_cast<long double>(value));
        }
    } else if (type == "number") {
        require(!spec.contains(json_schema ? "step" : "multipleOf"),
                json_schema ? "JSON Schema numbers use multipleOf" : "compact numbers use step");
        const auto lo = number(spec, "minimum"), hi = number(spec, "maximum");
        const auto step = number(spec, json_schema ? "multipleOf" : "step");
        require(step > 0 && hi >= lo, "number grid needs ordered bounds and a positive step");
        const auto count = (hi - lo) / step;
        require(std::isfinite(count) && count >= 0 && std::round(count) <= 254 &&
                    std::abs(count - std::round(count)) < 1e-7,
                "number grid must include both bounds and define 1-255 values");
        if (json_schema) {
            const auto multiples = lo / step;
            require(std::isfinite(multiples) && std::abs(multiples - std::round(multiples)) < 1e-7,
                    "minimum must be on the multipleOf grid");
        }
        const int digits = std::max({decimal_places(lo), decimal_places(hi), decimal_places(step)});
        for (int index = 0; index <= static_cast<int>(std::round(count)); ++index) {
            std::ostringstream encoded;
            encoded.imbue(std::locale::classic());
            encoded << std::fixed << std::setprecision(digits) << lo + index * step;
            const auto value = Json::parse(encoded.str());
            field.values.push_back(value);
            field.encoded.push_back(encoded.str());
            field.numbers.push_back(value.get<long double>());
        }
    } else {
        throw std::invalid_argument("supported field types: boolean, enum, integer, number");
    }
    if (field.encoded.empty()) {
        for (const auto& value : field.values) { field.encoded.push_back(value.dump()); }
    }
    for (std::size_t i = 0; i < field.values.size(); ++i) {
        require(std::find(field.values.begin(), field.values.begin() + i, field.values[i]) ==
                    field.values.begin() + i,
                "field " + name + " has duplicate values");
    }
    field.aggregate = text(spec, "aggregate", text(spec, "x-aggregate", "mode"));
    require(field.aggregate == "mode" || (!field.numbers.empty() && (field.aggregate == "median" ||
                                                                     field.aggregate == "mean")),
            "aggregate must be mode, or median/mean for a numeric field");
    return field;
}

} // namespace

Request parse_request(const Json& body) {
    keys(body, {"model", "instructions", "schema", "contexts", "mode", "tree_max", "cache_prompt"},
         "decision request");
    Request request;
    request.model = text(body, "model");
    request.mode  = text(body, "mode", "auto");
    require(request.mode == "auto" || request.mode == "tree" || request.mode == "greedy",
            "mode must be auto, tree or greedy");
    if (body.contains("tree_max")) {
        const auto limit = integer(body, "tree_max");
        require(limit >= 1 && limit <= 255, "tree_max must be in [1,255]");
        request.tree_max = static_cast<std::size_t>(limit);
    }
    request.cache_prompt = boolean(body, "cache_prompt", true);
    require(body.contains("contexts") && body.at("contexts").is_array() &&
                !body.at("contexts").empty() && body.at("contexts").size() <= 256,
            "contexts must contain 1-256 strings");
    std::size_t bytes = 0;
    for (const auto& value : body.at("contexts")) {
        require(value.is_string(), "every context must be a string");
        request.contexts.push_back(value.get<std::string>());
        bytes += request.contexts.back().size();
        require(bytes <= kMaximumTextBytes, "contexts exceed 1 MiB in aggregate");
    }
    require(body.contains("schema") && body.at("schema").is_object(), "schema must be an object");
    const auto& schema     = body.at("schema");
    const bool json_schema = schema.contains("properties");
    if (json_schema) {
        keys(schema,
             {"$schema", "type", "properties", "required", "additionalProperties", "title",
              "description"},
             "schema");
        require(text(schema, "type", "object") == "object", "schema type must be object");
        if (schema.contains("additionalProperties")) {
            (void)boolean(schema, "additionalProperties", false);
        }
    }
    const auto& fields = json_schema ? schema.at("properties") : schema;
    require(fields.is_object() && !fields.empty() && fields.size() <= 32,
            "schema must define 1-32 fields");
    if (json_schema && schema.contains("required")) {
        require(schema.at("required").is_array(), "required must be an array of property names");
        for (const auto& name : schema.at("required")) {
            require(name.is_string() && fields.contains(name.get<std::string>()),
                    "required references an unknown property");
        }
    }
    std::string catalogue;
    for (const auto& [name, spec] : fields.items()) {
        auto field = compile_field(name, spec, json_schema);
        require(request.mode != "greedy" || field.aggregate == "mode",
                "median and mean require an exact tree distribution");
        catalogue += Json(name).dump() + ": " + field.description + "\nAllowed values: ";
        for (std::size_t i = 0; i < field.encoded.size(); ++i) {
            catalogue += (i ? ", " : "") + field.encoded[i];
        }
        catalogue += "\n";
        require(catalogue.size() <= kMaximumTextBytes, "schema catalogue exceeds 1 MiB");
        request.fields.push_back(std::move(field));
    }
    const auto instructions = text(body, "instructions");
    require(instructions.size() <= kMaximumTextBytes, "instructions exceed 1 MiB");
    request.system =
        "Select the requested field value from its allowed values, based on the context. "
        "Respond with the JSON value only.\n\nFields:\n" +
        catalogue + "\n" + instructions;
    return request;
}

std::vector<double> log_softmax(std::span<const float> logits) {
    if (logits.empty()) { throw std::logic_error("empty decision logit row"); }
    for (const auto logit : logits) {
        if (!std::isfinite(logit)) { throw std::runtime_error("non-finite decision logit"); }
    }
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double sum           = 0;
    for (const auto logit : logits) { sum += std::exp(static_cast<double>(logit) - maximum); }
    const double normalizer = std::log(sum);
    std::vector<double> result;
    for (const auto logit : logits) {
        result.push_back((static_cast<double>(logit) - maximum) - normalizer);
    }
    return result;
}

void CandidateTrie::add(Tokens complete) {
    require(paths_.size() < 255 && !complete.empty(), "invalid candidate token path");
    if (paths_.empty()) {
        prefix_ = std::move(complete);
        paths_.emplace_back();
        return;
    }
    const auto shared = common_size(prefix_, complete);
    if (shared < prefix_.size()) {
        Tokens moved(prefix_.begin() + shared, prefix_.end());
        for (auto& path : paths_) { path.insert(path.begin(), moved.begin(), moved.end()); }
        prefix_.resize(shared);
    }
    paths_.emplace_back(complete.begin() + shared, complete.end());
}

void CandidateTrie::build() {
    require(!paths_.empty() && !prefix_.empty(), "candidates must have a nonempty prompt prefix");
    nodes_.clear();
    std::vector<std::size_t> members(paths_.size());
    std::iota(members.begin(), members.end(), 0);
    (void)build_node(members, 0);
}

int CandidateTrie::build_node(const std::vector<std::size_t>& members, std::size_t depth) {
    if (members.size() == 1) { return -static_cast<int>(members.front()) - 1; }
    for (;;) {
        for (const auto member : members) {
            require(depth < paths_[member].size(),
                    "candidate token paths collide or lack a terminator");
        }
        const auto token = paths_[members.front()][depth];
        if (!std::all_of(members.begin(), members.end(),
                         [&](auto i) { return paths_[i][depth] == token; })) {
            break;
        }
        ++depth;
    }
    Node node;
    node.suffix.assign(paths_[members.front()].begin(), paths_[members.front()].begin() + depth);
    for (const auto member : members) {
        const auto token = paths_[member][depth];
        auto found       = std::find(node.options.begin(), node.options.end(), token);
        if (found == node.options.end()) {
            node.options.push_back(token);
            node.members.emplace_back();
            found = std::prev(node.options.end());
        }
        node.members[found - node.options.begin()].push_back(member);
    }
    const auto index = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));
    // Recursion only at divergences (depth <= candidate count), not at every token.
    const auto groups = nodes_[index].members;
    for (const auto& group : groups) {
        const int child = build_node(group, depth + 1);
        nodes_[index].children.push_back(child);
    }
    return index;
}

Tokens CandidateTrie::prompt(std::size_t node) const {
    Tokens result      = prefix_;
    const auto& suffix = nodes_.at(node).suffix;
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}

std::vector<double>
CandidateTrie::distribution(const std::vector<std::vector<float>>& scores) const {
    if (scores.size() != nodes_.size()) {
        throw std::logic_error("decision tree row count mismatch");
    }
    std::vector<double> mass(paths_.size(), 0);
    for (std::size_t node = 0; node < nodes_.size(); ++node) {
        if (scores[node].size() != nodes_[node].options.size()) {
            throw std::logic_error("decision tree logit count mismatch");
        }
        const auto logp = log_softmax(scores[node]);
        for (std::size_t edge = 0; edge < logp.size(); ++edge) {
            for (const auto member : nodes_[node].members[edge]) { mass[member] += logp[edge]; }
        }
    }
    const double maximum = *std::max_element(mass.begin(), mass.end());
    double sum           = 0;
    for (auto& value : mass) {
        value = std::exp(value - maximum);
        sum += value;
    }
    for (auto& value : mass) { value /= sum; }
    return mass;
}

Json assemble(const std::vector<Field>& fields, const std::vector<FieldResult>& results,
              bool retain_probabilities) {
    if (fields.size() != results.size()) {
        throw std::logic_error("decision field count mismatch");
    }
    Json values = Json::object(), details = Json::object();
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& field  = fields[i];
        const auto& result = results[i];
        auto winner        = result.winner;
        Json detail        = Json::object();
        if (!field.numbers.empty() && result.probabilities.size() == field.values.size()) {
            const auto quantile = [&](double q) {
                double cumulative = 0;
                for (std::size_t j = 0; j < field.values.size(); ++j) {
                    cumulative += result.probabilities[j];
                    if (cumulative >= q) { return j; }
                }
                return field.values.size() - 1;
            };
            if (field.aggregate == "median") { winner = quantile(0.5); }
            if (field.aggregate == "mean") {
                long double mean = 0;
                for (std::size_t j = 0; j < field.numbers.size(); ++j) {
                    mean += field.numbers[j] * result.probabilities[j];
                }
                winner = 0;
                for (std::size_t j = 1; j < field.numbers.size(); ++j) {
                    if (std::abs(field.numbers[j] - mean) <
                        std::abs(field.numbers[winner] - mean)) {
                        winner = j;
                    }
                }
            }
            detail["interval_p10_p90"] = {field.values[quantile(0.1)], field.values[quantile(0.9)]};
        }
        values[field.name] = field.values.at(winner);
        detail["value"]    = field.values.at(winner);
        detail["probability"] =
            result.probabilities.empty() ? result.probability : result.probabilities.at(winner);
        detail["scored_nodes"] = result.scored_nodes;
        detail["tree"]         = result.tree;
        if (retain_probabilities) {
            if (!result.tree || result.probabilities.size() != field.values.size()) {
                throw std::logic_error("full decision probabilities require exact tree scoring");
            }
            detail["probabilities"] = result.probabilities;
        }
        details[field.name]    = std::move(detail);
    }
    return {{"decision", std::move(values)}, {"fields", std::move(details)}};
}

Json evaluate(const Request& request, const Backend& backend) {
    require(!request.contexts.empty() && request.contexts.size() <= 256 &&
                !request.fields.empty() && request.fields.size() <= 32,
            "decision request needs 1-256 contexts and 1-32 compiled fields");
    for (const auto& field : request.fields) {
        require(!field.values.empty() && field.values.size() <= 255 &&
                    field.values.size() == field.encoded.size() &&
                    (field.numbers.empty() || field.numbers.size() == field.values.size()),
                "invalid compiled decision field");
    }
    const auto started     = std::chrono::steady_clock::now();
    Json results           = Json::array();
    std::uint64_t computed = 0, reused = 0, rows = 0, rounds = 0, context_tokens = 0;
    double prefill_ms = 0;
    std::uint64_t forward_batches       = 0;
    std::uint32_t largest_batch         = 0;
    constexpr std::size_t context_batch = 8;
    const auto fields                   = request.fields.size();
    const auto static_prompt            = backend.tokenize(request.system, "", "{\n");
    const bool isolated_state = request.contexts.size() == 1 &&
        std::all_of(request.fields.begin(), request.fields.end(),
                    [](const Field& field) { return !field.isolated_question.empty(); });
    const auto state_prompt = isolated_state && request.cache_prompt
        ? backend.tokenize(request.system, request.contexts.front(), "{\n") : Tokens{};
    for (std::size_t begin = 0; begin < request.contexts.size(); begin += context_batch) {
        backend.check_cancelled();
        const auto contexts = std::min(context_batch, request.contexts.size() - begin);
        std::vector<CandidateTrie> tries(fields * contexts);
        std::vector<FieldResult> answers(tries.size());
        std::vector<std::vector<std::uint32_t>> frontiers(tries.size());
        std::vector<std::uint64_t> context_lengths(contexts), context_rows(contexts);
        for (std::size_t c = 0; c < contexts; ++c) {
            Tokens trunk;
            for (std::size_t f = 0; f < fields; ++f) {
                const auto index  = c * fields + f;
                const auto& field = request.fields[f];
                const auto context = field.isolated_question.empty()
                    ? request.contexts[begin + c]
                    : request.contexts[begin + c] + "\n\nQuestion:\n" + field.isolated_question;
                const auto key = field.isolated_question.empty() ? field.name : std::string("answer");
                for (const auto& value : field.encoded) {
                    backend.check_cancelled();
                    tries[index].add(
                        backend.tokenize(request.system, context,
                                         "{\n  " + Json(key).dump() + ": " + value + "\n}"));
                }
                tries[index].build();
                answers[index].tree =
                    request.mode == "tree" || field.aggregate != "mode" ||
                    (request.mode == "auto" && field.values.size() <= request.tree_max);
                if (f == 0) {
                    trunk = tries[index].prefix();
                } else {
                    trunk.resize(common_size(trunk, tries[index].prefix()));
                }
            }
            const auto shared  = common_size(trunk, static_prompt);
            context_lengths[c] = trunk.size() - shared;
            context_tokens += context_lengths[c];
            if (request.cache_prompt) {
                const auto root = isolated_state ? common_size(trunk, state_prompt) : shared;
                for (std::size_t f = 0; f < fields; ++f) {
                    const auto index = c * fields + f;
                    auto& points = frontiers[index];
                    if (isolated_state) {
                        // One state, many isolated questions: prefill the state once, then
                        // each rubric once. Every divergence in a question reuses its rubric.
                        // Compare complete template tokenizations; never concatenate BPE IDs.
                        points = {static_cast<std::uint32_t>(root),
                                  static_cast<std::uint32_t>(tries[index].prefix().size())};
                    } else {
                        if (shared) { points.push_back(static_cast<std::uint32_t>(shared)); }
                        if (trunk.size() > shared) {
                            points.push_back(static_cast<std::uint32_t>(trunk.size()));
                        }
                    }
                }
            }
        }
        std::vector<std::vector<std::vector<float>>> tree_scores(tries.size());
        std::vector<int> current(tries.size(), 0);
        std::vector<double> greedy_logp(tries.size(), 0);
        for (std::size_t f = 0; f < tries.size(); ++f) {
            tree_scores[f].resize(tries[f].nodes().size());
            if (tries[f].nodes().empty()) { current[f] = -1; }
        }
        for (;;) {
            std::vector<ScoreRow> pending;
            std::vector<std::pair<std::size_t, std::size_t>> locations;
            for (std::size_t f = 0; f < tries.size(); ++f) {
                if (current[f] < 0) { continue; }
                const auto append = [&](std::size_t node) {
                    pending.push_back({tries[f].prefix(), tries[f].nodes()[node].suffix,
                                       tries[f].nodes()[node].options, frontiers[f]});
                    locations.emplace_back(f, node);
                };
                if (answers[f].tree) {
                    for (std::size_t n = 0; n < tries[f].nodes().size(); ++n) { append(n); }
                } else {
                    append(static_cast<std::size_t>(current[f]));
                }
            }
            if (pending.empty()) { break; }
            backend.check_cancelled();
            const auto scored = backend.score(std::move(pending), request.cache_prompt);
            if (scored.logits.size() != locations.size()) {
                throw std::logic_error("decision batch size mismatch");
            }
            computed += scored.computed_tokens;
            reused += scored.reused_tokens;
            prefill_ms += scored.prefill_ms;
            rows += locations.size();
            forward_batches += scored.forward_batches;
            largest_batch = std::max(largest_batch, scored.largest_batch);
            ++rounds;
            for (std::size_t i = 0; i < locations.size(); ++i) {
                const auto [f, n] = locations[i];
                const auto& node  = tries[f].nodes()[n];
                if (scored.logits[i].size() != node.options.size()) {
                    throw std::logic_error("decision row size mismatch");
                }
                ++answers[f].scored_nodes;
                ++context_rows[f / fields];
                if (answers[f].tree) {
                    tree_scores[f][n] = scored.logits[i];
                    current[f]        = -1;
                } else {
                    const auto probabilities = log_softmax(scored.logits[i]);
                    const auto edge          = static_cast<std::size_t>(
                        std::max_element(probabilities.begin(), probabilities.end()) -
                        probabilities.begin());
                    greedy_logp[f] += probabilities[edge];
                    current[f] = node.children[edge];
                    if (current[f] < 0) {
                        answers[f].winner      = static_cast<std::size_t>(-current[f] - 1);
                        answers[f].probability = std::exp(greedy_logp[f]);
                    }
                }
            }
        }
        for (std::size_t f = 0; f < tries.size(); ++f) {
            if (answers[f].tree) {
                answers[f].probabilities = tries[f].distribution(tree_scores[f]);
                answers[f].winner =
                    static_cast<std::size_t>(std::max_element(answers[f].probabilities.begin(),
                                                              answers[f].probabilities.end()) -
                                             answers[f].probabilities.begin());
            }
        }
        for (std::size_t c = 0; c < contexts; ++c) {
            std::vector<FieldResult> selected(answers.begin() + c * fields,
                                              answers.begin() + (c + 1) * fields);
            auto result     = assemble(request.fields, selected, request.retain_probabilities);
            result["usage"] = {{"context_tokens", context_lengths[c]},
                               {"scored_rows", context_rows[c]}};
            results.push_back(std::move(result));
        }
    }
    const auto total =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return {{"object", "decision"},
            {"results", std::move(results)},
            {"usage",
             {{"prompt_tokens", computed + reused},
              {"cached_tokens", reused},
              {"computed_tokens", computed},
              {"context_tokens", context_tokens},
              {"scored_rows", rows}}},
            {"timings",
             {{"prefill_ms", prefill_ms},
              {"scoring_ms", std::max(0.0, total - prefill_ms)},
              {"total_ms", total},
              {"rounds", rounds},
              {"forward_batches", forward_batches},
              {"largest_batch", largest_batch},
              {"per_decision_ms", total / request.contexts.size()}}}};
}

} // namespace ninfer::decision
