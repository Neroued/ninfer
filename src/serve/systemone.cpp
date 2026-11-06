#include "serve/systemone.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace ninfer::serve {
namespace {
using decision::Json;
constexpr std::size_t kMaximumTextBytes = 1U << 20;
constexpr std::array kEdges{"top", "bottom", "left", "right"};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void keys(const Json& value, std::initializer_list<std::string_view> allowed,
          const std::string& where) {
    require(value.is_object(), where + " must be an object");
    for (const auto& [key, unused] : value.items()) {
        require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(),
                where + ": unsupported member " + key);
    }
}

void entry(const Json& value, const std::string& where) {
    // SDK EntryType permits null, with arbitrary JSON nested inside objects/arrays.
    require(value.is_string() || value.is_object() || value.is_array() || value.is_null(),
            where + " must be text, an object, an array, or null");
}

double confidence(const std::vector<double>& probabilities, bool score) {
    const auto n = probabilities.size();
    if (n == 1) { return 1.0; }
    const auto winner = std::max_element(probabilities.begin(), probabilities.end());
    // TypeSafe's published Choice formula: (p_max - 1/n) / (1 - 1/n).
    if (!score) { return std::clamp((*winner * n - 1.0) / (n - 1.0), 0.0, 1.0); }
    const auto mode     = static_cast<double>(winner - probabilities.begin());
    const double middle = (n - 1.0) / 2.0;
    double spread = 0, uniform_spread = 0;
    for (std::size_t i = 0; i < n; ++i) {
        spread += probabilities[i] * std::abs(static_cast<double>(i) - mode);
        uniform_spread += std::abs(static_cast<double>(i) - middle) / n;
    }
    return std::clamp(1.0 - spread / uniform_spread, 0.0, 1.0);
}
} // namespace

SystemOneRequest parse_systemone_request(const Json& body) {
    keys(body, {"state", "questions", "model", "images"}, "systemone request");
    require(body.contains("state"), "state is required");
    entry(body.at("state"), "state");
    require(body.contains("questions") && body.at("questions").is_object() &&
                !body.at("questions").empty() && body.at("questions").size() <= 32,
            "questions must contain 1-32 named questions");
    const auto& state         = body.at("state");
    const std::string context = state.is_string() ? state.get<std::string>() : state.dump();
    require(context.size() <= kMaximumTextBytes, "state exceeds 1 MiB");
    Json translated = {{"contexts", Json::array({"State:\n" + context})},
                       {"schema", {{"type", "object"}, {"properties", Json::object()}}},
                       {"mode", "tree"}};
    if (body.contains("model")) {
        require(body.at("model").is_string() &&
                    !body.at("model").get_ref<const std::string&>().empty(),
                "model must be a nonempty served model alias");
        translated["model"] = body.at("model");
    }
    SystemOneRequest request;
    const auto images = body.value("images", Json::array());
    require(images.is_array() && images.size() <= 8, "images must contain at most 8 sources");
    for (const auto& image : images) {
        require(image.is_string() && !image.get_ref<const std::string&>().empty(),
                "each image must be a nonempty source string");
        request.images.push_back(image.get<std::string>());
    }
    std::size_t bytes = context.size();
    const auto append = [&](const std::string& name, Json spec, const Json& rubric) {
        auto isolated = rubric.dump();
        bytes += isolated.size();
        require(bytes <= kMaximumTextBytes, "state and questions exceed 1 MiB");
        spec["description"]                       = "Answer the isolated question.";
        translated["schema"]["properties"]        = Json{{name, std::move(spec)}};
        auto compiled                             = decision::parse_request(translated);
        compiled.fields.front().isolated_question = std::move(isolated);
        request.decision.fields.push_back(std::move(compiled.fields.front()));
    };
    for (const auto& [id, question] : body.at("questions").items()) {
        require(!id.empty() && id.size() <= 4096, "question names must contain 1-4096 bytes");
        const auto where = "questions[" + Json(id).dump() + "]";
        keys(question, {"type", "instructions", "criteria"}, where);
        require(question.contains("type") && question.at("type").is_string(),
                where + ".type is required");
        const auto type         = question.at("type").get<std::string>();
        const auto instructions = question.value("instructions", Json(nullptr));
        entry(instructions, where + ".instructions");
        auto criteria = question.value("criteria", Json(nullptr));
        Json spec;
        SystemOneType kind;
        const auto first_field = request.decision.fields.size();
        std::string aggregate  = "mode";
        if (type == "choice") {
            require(criteria.is_object() && !criteria.empty() && criteria.size() <= 255,
                    where + ".criteria must contain 1-255 options");
            spec = {{"type", "enum"}, {"choices", Json::array()}};
            for (const auto& [label, description] : criteria.items()) {
                require(!label.empty(), where + ".criteria option names must not be empty");
                entry(description, where + ".criteria[" + Json(label).dump() + "]");
                spec["choices"].push_back(label);
            }
            kind = SystemOneType::Choice;
        } else if (type == "noul") {
            if (criteria.is_null()) { criteria = Json::object(); }
            keys(criteria, {"true", "false"}, where + ".criteria");
            for (const auto& [label, description] : criteria.items()) {
                entry(description, where + ".criteria." + label);
            }
            spec = {{"type", "boolean"}};
            kind = SystemOneType::Noul;
        } else if (type == "score") {
            require(criteria.is_array() && criteria.size() >= 2 && criteria.size() <= 10,
                    where + ".criteria must contain 2-10 ordered levels");
            for (const auto& description : criteria) {
                entry(description, where + ".criteria level");
            }
            spec = {{"type", "integer"}, {"minimum", 0}, {"maximum", criteria.size() - 1}};
            kind = SystemOneType::Score;
        } else if (type == "box") {
            require(!request.images.empty(), where + ": box requires images");
            if (criteria.is_null()) { criteria = Json::object(); }
            keys(criteria, {"step", "aggregate"}, where + ".criteria");
            require(!criteria.contains("aggregate") || criteria["aggregate"].is_string(),
                    where + ".criteria.aggregate must be mode, mean, or median");
            aggregate = criteria.value("aggregate", std::string("mode"));
            require(aggregate == "mode" || aggregate == "mean" || aggregate == "median",
                    where + ".criteria.aggregate must be mode, mean, or median");
            const auto step = criteria.value("step", Json(0.01));
            for (const auto* edge : kEdges) {
                auto name = std::string(1, '\0') + std::to_string(first_field) + ":" + edge;
                while (body["questions"].contains(name)) name += '\0';
                append(name,
                       {{"type", "number"}, {"minimum", 0}, {"maximum", 1}, {"multipleOf", step}},
                       {{"type", "box"},
                        {"edge", edge},
                        {"instructions", instructions},
                        {"grid", "a normalized coordinate, 0 at the picture's top-left edge and "
                                 "1 at its bottom-right one, on a " +
                                     step.dump() + " grid"}});
            }
            request.questions.push_back(
                {id, SystemOneType::Box, std::move(criteria), first_field, std::move(aggregate)});
            continue;
        } else {
            throw std::invalid_argument(where + ".type must be choice, noul, score, or box");
        }
        const Json rubric = {
            {"type", type}, {"instructions", instructions}, {"criteria", criteria}};
        append(id, std::move(spec), rubric);
        request.questions.push_back({id, kind, std::move(criteria), first_field});
    }
    request.decision.model    = translated.value("model", std::string{});
    request.decision.contexts = translated["contexts"].get<std::vector<std::string>>();
    request.decision.mode     = "tree";
    request.decision.system =
        "Evaluate the single question against the supplied state. Follow its instructions and "
        "criteria. For choice, select a criterion label; for noul, select true or false; "
        "for score, select a zero-based criterion level. Return only the JSON answer field.";
    request.decision.retain_probabilities = true;
    if (std::any_of(request.questions.begin(), request.questions.end(),
                    [](const auto& q) { return q.type == SystemOneType::Box; })) {
        request.decision.system +=
            " For box, select the requested normalized coordinate: top/bottom measure downward "
            "and left/right measure rightward, from 0 to 1 across the whole picture.";
    }
    return request;
}

Json format_systemone_response(const SystemOneRequest& request, const Json& result,
                               const std::string& served_model) {
    if (result.at("results").size() != 1) { throw std::logic_error("systemone expects one state"); }
    const auto& details = result.at("results").at(0).at("fields");
    Json answers        = Json::object();
    for (std::size_t q = 0; q < request.questions.size(); ++q) {
        const auto& question = request.questions[q];
        const auto read_distribution = [&](std::size_t index) {
            const auto& field = request.decision.fields.at(index);
            auto probabilities =
                details.at(field.name).at("probabilities").get<std::vector<double>>();
            if (probabilities.size() != field.values.size() || probabilities.empty() ||
                std::any_of(probabilities.begin(), probabilities.end(),
                            [](double p) { return !std::isfinite(p) || p < 0 || p > 1; }) ||
                std::abs(std::accumulate(probabilities.begin(), probabilities.end(), 0.0) - 1.0) >
                    1e-8) {
                throw std::logic_error("invalid systemone decision distribution");
            }
            return probabilities;
        };
        if (question.type == SystemOneType::Box) {
            Json box = Json::object(), edges = Json::object();
            double certainty = 1;
            std::array<double, 4> selected{};
            for (std::size_t e = 0; e < kEdges.size(); ++e) {
                const auto ps      = read_distribution(question.first_field + e);
                const auto& values = request.decision.fields.at(question.first_field + e).numbers;
                const auto mode    = std::max_element(ps.begin(), ps.end()) - ps.begin();
                double mean = 0, cumulative = 0, spread = 0, uniform = 0;
                std::size_t median = ps.size() - 1;
                bool found         = false;
                const auto middle  = (values.front() + values.back()) / 2;
                for (std::size_t i = 0; i < ps.size(); ++i) {
                    mean += ps[i] * static_cast<double>(values[i]);
                    cumulative += ps[i];
                    if (!found && cumulative >= 0.5) {
                        median = i;
                        found  = true;
                    }
                    spread += ps[i] * static_cast<double>(std::abs(values[i] - values[mode]));
                    uniform += static_cast<double>(std::abs(values[i] - middle)) / ps.size();
                }
                const auto nearest =
                    std::min_element(values.begin(), values.end(), [mean](auto a, auto b) {
                        return std::abs(a - mean) < std::abs(b - mean);
                    });
                selected[e] = static_cast<double>(question.aggregate == "mean"     ? *nearest
                                                  : question.aggregate == "median" ? values[median]
                                                                                   : values[mode]);
                box[kEdges[e]]   = selected[e];
                certainty        = std::min(certainty, std::max(0.0, 1.0 - spread / uniform));
                edges[kEdges[e]] = {{"probability", ps[mode]},
                                    {"modal", values[mode]},
                                    {"mean", *nearest},
                                    {"median", values[median]}};
            }
            if (selected[0] > selected[1]) std::swap(box["top"], box["bottom"]);
            if (selected[2] > selected[3]) std::swap(box["left"], box["right"]);
            for (std::size_t e = 0; e < kEdges.size(); ++e)
                edges[kEdges[e]]["moved"] = selected[e] != box[kEdges[e]].get<double>();
            answers[question.id] = {{"type", "box"},
                                    {"aggregate", question.aggregate},
                                    {"box", std::move(box)},
                                    {"confidence", certainty},
                                    {"edges", std::move(edges)}};
            continue;
        }
        const auto& field        = request.decision.fields.at(question.first_field);
        const auto probabilities = read_distribution(question.first_field);
        if (question.type == SystemOneType::Noul) {
            // Native boolean domains are [true, false]; never use the winning probability here.
            answers[question.id] = {{"type", "noul"},
                                    {"noul", probabilities.at(0)},
                                    {"confidence", probabilities.at(0)}};
            continue;
        }
        Json distribution = Json::object(), legend = Json::object();
        double mean      = 0;
        const bool score = question.type == SystemOneType::Score;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            const auto key    = score ? std::to_string(i) : field.values[i].get<std::string>();
            distribution[key] = probabilities[i];
            if (score) { legend[key] = question.criteria.at(i); }
            mean += static_cast<double>(i) * probabilities[i];
        }
        Json answer = {{"type", score ? "score" : "choice"},
                       {"probabilities", std::move(distribution)},
                       {"confidence", confidence(probabilities, score)}};
        if (score) {
            answer["score"]  = mean;
            answer["legend"] = std::move(legend);
        } else {
            const auto winner = std::max_element(probabilities.begin(), probabilities.end()) -
                                probabilities.begin();
            answer["choice"]  = field.values.at(static_cast<std::size_t>(winner));
        }
        answers[question.id] = std::move(answer);
    }
    Json response = {
        {"model", served_model},
        {"answers", std::move(answers)},
        {"usage",
         {{"input_tokens", result.at("usage").at("prompt_tokens")}, {"output_tokens", 0}}}};
    for (const auto* key : {"execution", "timings"})
        if (result.contains(key)) response[key] = result.at(key);
    return response;
}

} // namespace ninfer::serve
