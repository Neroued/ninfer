#include "text/structured_output.h"
#include "text/schema_normalization.h"
#include "text/unique_strings.h"
#include <bit>
#include <xgrammar/xgrammar.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;

bool schema_value_equal(const Json& left, const Json& right) {
    if (left.is_number() && right.is_number())
        return left.get<long double>() == right.get<long double>();
    if (left.is_object() && right.is_object()) {
        if (left.size() != right.size()) return false;
        for (const auto& [key, value] : left.items()) {
            if (!right.contains(key) || !schema_value_equal(value, right.at(key))) return false;
        }
        return true;
    }
    if (left.is_array() && right.is_array()) {
        if (left.size() != right.size()) return false;
        for (std::size_t index = 0; index < left.size(); ++index)
            if (!schema_value_equal(left[index], right[index])) return false;
        return true;
    }
    return left == right;
}

bool contains_schema_reference(const Json& schema) {
    if (!schema.is_object()) return false;
    if (schema.contains("$ref") || schema.contains("$dynamicRef") || schema.contains("$recursiveRef")) return true;
    for (const auto& [key, value] : schema.items()) {
        if (key == "properties" || key == "$defs" || key == "definitions" || key == "patternProperties" || key == "dependentSchemas") {
            if (value.is_object()) for (const auto& child : value)
                if (contains_schema_reference(child)) return true;
        } else if (key == "anyOf" || key == "allOf" || key == "oneOf" || key == "prefixItems") {
            if (value.is_array()) for (const auto& child : value)
                if (contains_schema_reference(child)) return true;
        } else if (key == "items" || key == "additionalProperties" || key == "propertyNames" || key == "not" || key == "contains" || key == "if" || key == "then" || key == "else") {
            if (contains_schema_reference(value)) return true;
        }
    }
    return false;
}

void schema_check(const Json& s) {
    if (s.is_boolean()) { return; }
    if (!s.is_object()) { throw std::invalid_argument("JSON schema must be an object or boolean"); }
    static const std::unordered_set<std::string> annotations = {
        "$schema",  "title",    "description", "default",
        "examples", "$comment", "$defs",       "definitions", "$id"};
    static const std::unordered_set<std::string> supported = {"type",
                                                              "properties",
                                                              "required",
                                                              "additionalProperties",
                                                              "items",
                                                              "uniqueItems",
                                                              "prefixItems",
                                                              "minItems",
                                                              "maxItems",
                                                              "maxProperties",
                                                              "pattern",
                                                              "format",
                                                              "minLength",
                                                              "maxLength",
                                                              "enum",
                                                              "const",
                                                              "anyOf",
                                                              "$ref",
                                                              "minimum",
                                                              "maximum",
                                                              "exclusiveMinimum",
                                                              "exclusiveMaximum"};
    for (const auto& [key, value] : s.items()) {
        if (key == "$schema" && value != "https://json-schema.org/draft/2020-12/schema" &&
            value != "http://json-schema.org/draft-07/schema#") {
            throw std::invalid_argument("unsupported JSON Schema dialect");
        }
        if ((key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems" || key == "maxProperties") &&
            (!value.is_number_integer() || value < 0 || value > 2147483647)) {
            throw std::invalid_argument(key + " must be a nonnegative 32-bit integer");
        }
        if (key == "minimum" || key == "maximum" || key == "exclusiveMinimum" ||
            key == "exclusiveMaximum") {
            // The pinned compiler represents bounds as doubles. Keep integer bounds exact
            // and reject non-finite values before they reach its range arithmetic.
            if (!value.is_number() || !std::isfinite(value.get<double>()) ||
                std::abs(value.get<long double>()) > 9007199254740991.0L) {
                throw std::invalid_argument(key + " requires a finite bound within +/- (2^53-1)");
            }
            if (!s.contains("type") || (s.at("type") != "integer" && s.at("type") != "number")) {
                throw std::invalid_argument(
                    "numeric bounds require explicit integer or number type");
            }
        }
        if (!annotations.contains(key) && !supported.contains(key)) {
            throw std::invalid_argument("unsupported JSON schema keyword: " + key);
        }
        if (key == "$id") {
            if (!value.is_string() || contains_schema_reference(s))
                throw std::invalid_argument("schema resource identifiers with references need explicit scoped resolution");
        }
        if (key == "maxProperties") {
            const bool object_type = s.contains("type") &&
                (s.at("type") == "object" || (s.at("type").is_array() &&
                 std::any_of(s.at("type").begin(), s.at("type").end(), [](const auto& type) { return type == "object"; })));
            if (!object_type && !s.contains("properties") && !s.contains("additionalProperties"))
                throw std::invalid_argument("maxProperties requires a represented object branch");
        }
        if (key == "uniqueItems" && !value.is_boolean())
            throw std::invalid_argument("uniqueItems must be boolean");
        if (key == "required") {
            if (!value.is_array()) throw std::invalid_argument("required must be an array of unique property names");
            std::unordered_set<std::string> names;
            for (const auto& name : value) {
                if (!name.is_string() || !names.insert(name.get<std::string>()).second)
                    throw std::invalid_argument("required must be an array of unique property names");
                const bool declared = s.contains("properties") && s.at("properties").is_object() &&
                                      s.at("properties").contains(name.get<std::string>());
                if (!declared && s.contains("additionalProperties") && s.at("additionalProperties") == false)
                    throw std::invalid_argument("required property is forbidden by additionalProperties: " + name.get<std::string>());
            }
        }
        if (key == "pattern" || key == "format") {
            if (!value.is_string() || !s.contains("type") || s.at("type") != "string") {
                throw std::invalid_argument(key + " requires a string value and explicit string type");
            }
            // The pinned converter chooses one string representation. Refuse conjunctions
            // until their intersection is implemented; never silently discard a bound.
            if (s.contains("minLength") || s.contains("maxLength") ||
                (s.contains("pattern") && s.contains("format"))) {
                throw std::invalid_argument("pattern/format cannot be combined with length bounds or each other");
            }
            if (key == "format" && value != "date" && value != "date-time" && value != "time") {
                throw std::invalid_argument("unsupported asserted string format: " + value.get<std::string>());
            }
        }
        if (key == "$ref" &&
            (!value.is_string() || (value != "#" && !value.get<std::string>().starts_with("#/")))) {
            throw std::invalid_argument("JSON schema supports only local fragment $ref values");
        }
        if (key == "$ref" && value != "#") {
            const auto ref = value.get<std::string>();
            // The pinned compiler interprets literal object paths, not RFC 6901 escapes.
            // Reject ambiguous spellings rather than resolving a different schema silently.
            if (ref.find_first_of("~%") != std::string::npos || ref.ends_with('/') ||
                ref.find("//") != std::string::npos) {
                throw std::invalid_argument(
                    "$ref requires nonempty, unescaped object path segments");
            }
        }
        if (key == "properties" || key == "$defs" || key == "definitions") {
            if (!value.is_object()) { throw std::invalid_argument(key + " must be an object"); }
            for (const auto& child : value) { schema_check(child); }
        } else if (key == "items" || key == "additionalProperties") {
            schema_check(value);
        } else if (key == "anyOf" || key == "prefixItems") {
            if (!value.is_array()) { throw std::invalid_argument(key + " must be an array"); }
            for (const auto& child : value) { schema_check(child); }
        }
    }
    // XGrammar prioritizes these branches over their siblings. Disallow combinations that
    // would otherwise silently discard constraints (annotations and definitions are harmless).
    for (const char* branch : {"$ref", "const", "enum", "anyOf"}) {
        if (!s.contains(branch)) { continue; }
        for (const auto& [key, value] : s.items()) {
            if (key != branch && !annotations.contains(key)) {
                // Pydantic Literal may emit both assertions. Compile the constant only
                // when it satisfies enum as well; object order and number kinds are semantic.
                if (((std::string_view(branch) == "const" && key == "enum") ||
                     (std::string_view(branch) == "enum" && key == "const")) &&
                    s.at("enum").is_array() &&
                    std::any_of(s.at("enum").begin(), s.at("enum").end(),
                                [&](const auto& candidate) { return schema_value_equal(s.at("const"), candidate); })) {
                    continue;
                }
                if (key == "type" &&
                    (std::string_view(branch) == "enum" || std::string_view(branch) == "const")) {
                    const auto matches = [&](const Json& v) {
                        if (!value.is_string()) { return false; }
                        const auto type = value.get<std::string>();
                        return (type == "string" && v.is_string()) ||
                               (type == "integer" && v.is_number_integer()) ||
                               (type == "number" && v.is_number()) ||
                               (type == "boolean" && v.is_boolean()) ||
                               (type == "null" && v.is_null()) ||
                               (type == "array" && v.is_array()) ||
                               (type == "object" && v.is_object());
                    };
                    if (s.contains("const") && matches(s.at("const"))) { continue; }
                    if (s.contains("enum") && s.at("enum").is_array() &&
                        std::all_of(s.at("enum").begin(), s.at("enum").end(), matches)) {
                        continue;
                    }
                }
                throw std::invalid_argument(std::string(branch) + " cannot be combined with " +
                                            key);
            }
        }
    }
}
} // namespace

void validate_structured_output(const StructuredOutputOptions& options) {
    if (options.kind == StructuredOutputKind::JsonSchema) {
        try {
            const auto normalized_schema = normalize_json_schema(options.schema);
            schema_check(Json::parse(normalized_schema));
            UniqueStringState::AnalyzeSchema(normalized_schema);
        } catch (const Json::exception& e) {
            throw std::invalid_argument(std::string("invalid JSON schema: ") + e.what());
        }
    } else if (!options.schema.empty()) {
        throw std::invalid_argument("schema requires JsonSchema mode");
    }
}

struct GrammarState::Impl {
    xgrammar::GrammarMatcher matcher;
    int vocab_size;
    std::shared_ptr<const std::vector<std::string>> vocabulary;
    std::shared_ptr<const std::vector<std::uint8_t>> stops;
    std::unique_ptr<UniqueStringState> unique;

    Impl(xgrammar::GrammarMatcher matcher, int vocab_size,
         std::shared_ptr<const std::vector<std::string>> vocabulary,
         std::shared_ptr<const std::vector<std::uint8_t>> stops,
         std::unique_ptr<UniqueStringState> unique)
        : matcher(std::move(matcher)), vocab_size(vocab_size),
          vocabulary(std::move(vocabulary)), stops(std::move(stops)), unique(std::move(unique)) {}
    std::string_view semantic_bytes(TokenId token) const {
        return (*stops)[token] ? std::string_view{} : std::string_view((*vocabulary)[token]);
    }
};

GrammarState::GrammarState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GrammarState::~GrammarState() = default;
GrammarState::GrammarState(GrammarState&&) noexcept = default;
GrammarState& GrammarState::operator=(GrammarState&&) noexcept = default;

std::unique_ptr<GrammarState> GrammarState::fork() const {
    auto unique = impl_->unique ? std::make_unique<UniqueStringState>(*impl_->unique) : nullptr;
    return std::unique_ptr<GrammarState>(new GrammarState(std::make_unique<Impl>(
        impl_->matcher.Fork(), impl_->vocab_size, impl_->vocabulary, impl_->stops, std::move(unique))));
}

void GrammarState::accept(std::span<const TokenId> tokens) {
    for (TokenId token : tokens) {
        if (token < 0 || token >= impl_->vocab_size || impl_->matcher.IsTerminated())
            throw std::logic_error("generated token violates the structured output grammar");
        auto next_unique = impl_->unique ? std::make_unique<UniqueStringState>(*impl_->unique) : nullptr;
        if ((next_unique && !next_unique->accept(impl_->semantic_bytes(token))) ||
            !impl_->matcher.AcceptToken(token))
            throw std::logic_error("generated token " + std::to_string(token) +
                                   " violates the structured output grammar");
        if (next_unique) impl_->unique = std::move(next_unique);
    }
}

void GrammarState::fill_masks(std::span<std::uint32_t> masks,
                              std::span<const TokenId> drafts) const {
    const int words = xgrammar::GetBitmaskSize(impl_->vocab_size);
    if (masks.size() != (drafts.size() + 1) * words)
        throw std::logic_error("incorrect grammar mask shape");
    auto matcher = impl_->matcher.Fork();
    auto unique = impl_->unique ? std::make_unique<UniqueStringState>(*impl_->unique) : nullptr;
    std::fill(masks.begin(), masks.end(), ~std::uint32_t{0});
    for (std::size_t col = 0; col <= drafts.size(); ++col) {
        if (matcher.IsTerminated()) break;
        std::int64_t shape[2] = {1, words};
        DLTensor tensor{};
        tensor.data = masks.data() + col * words;
        tensor.device = {kDLCPU, 0}; tensor.ndim = 2;
        tensor.dtype = {kDLInt, 32, 1}; tensor.shape = shape;
        matcher.FillNextTokenBitmask(&tensor);
        if (impl_->vocab_size % 32)
            masks[(col + 1) * words - 1] &= (1U << (impl_->vocab_size % 32)) - 1;
        auto row = masks.subspan(col * words, words);
        // Semantic work is absent for ordinary/trivially unique schemas.
        if (unique) {
            for (int word = 0; word < words; ++word) {
                auto candidates = row[word];
                while (candidates) {
                    const int bit = std::countr_zero(candidates);
                    const TokenId token = word * 32 + bit;
                    const auto bytes = impl_->semantic_bytes(token);
                    if (!(*impl_->stops)[token] && unique->needs_check(bytes) && !unique->accepts(bytes))
                        row[word] &= ~(1U << bit);
                    candidates &= candidates - 1;
                }
            }
        }
        if (std::none_of(row.begin(), row.end(), [](auto word) { return word != 0; }))
            throw std::runtime_error("structured output grammar has no admissible next token");
        if (col < drafts.size()) {
            const TokenId token = drafts[col];
            if (token < 0 || token >= impl_->vocab_size || !(row[token / 32] & (1U << (token % 32)))) break;
            if (!matcher.AcceptToken(token)) break;
            if (unique && !unique->accept(impl_->semantic_bytes(token)))
                throw std::logic_error("admissible draft violated semantic grammar state");
        }
    }
}

struct StructuredCompiler::Impl {
    std::shared_ptr<const std::vector<std::string>> vocabulary;
    std::shared_ptr<const std::vector<std::uint8_t>> stops;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
    std::mutex mutex;

    Impl(std::vector<std::string> vocab, std::vector<int> stop_ids)
        : vocabulary(std::make_shared<const std::vector<std::string>>(std::move(vocab))),
          tokenizer(*vocabulary, xgrammar::VocabType::RAW, static_cast<int>(vocabulary->size()), stop_ids),
          compiler(tokenizer, 4, true, 256 * 1024 * 1024) {
        auto flags = std::make_shared<std::vector<std::uint8_t>>(vocabulary->size());
        for (int token : stop_ids) {
            if (token < 0 || static_cast<std::size_t>(token) >= flags->size())
                throw std::invalid_argument("stop token is outside the vocabulary");
            (*flags)[token] = 1;
        }
        stops = std::move(flags);
    }
};

StructuredCompiler::StructuredCompiler(std::vector<std::string> vocab, std::vector<int> stops)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(stops))) {}

StructuredCompiler::~StructuredCompiler() = default;

std::shared_ptr<GrammarState>
StructuredCompiler::compile(const StructuredOutputOptions& options,
                            const StructuredOutputEnvelope& envelope) {
    validate_structured_output(options);
    if (options.kind == StructuredOutputKind::None) { return {}; }
    std::lock_guard lock(impl_->mutex);
    try {
        const auto normalized_schema = options.kind == StructuredOutputKind::JsonSchema
            ? normalize_json_schema(options.schema) : std::string("{\"type\":\"object\"}");
        auto unique = std::make_unique<UniqueStringState>(normalized_schema, envelope.reasoning_close);
        if (!unique->has_constraints()) unique.reset();
        // strict_mode=false retains JSON Schema defaults for additional properties/items.
        auto grammar = impl_->compiler.CompileJSONSchema(
            normalized_schema,
            true, std::nullopt, std::nullopt, false, 8);
        if (!envelope.reasoning_close.empty() || !envelope.alternative_format.empty()) {
            std::ostringstream ebnf;
            ebnf << grammar.GetGrammar();
            Json content{{"type", "grammar"}, {"grammar", ebnf.str()}};
            // The schema root starts at the JSON value; Qwen's reasoning close is followed by
            // whitespace (including the canonical budget-control suffix's two newlines).
            content = Json{
                {"type", "sequence"},
                {"elements", Json::array({Json{{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}},
                                          content})}};
            if (!envelope.alternative_format.empty()) {
                content = Json{
                    {"type", "or"},
                    {"elements", Json::array({content, Json::parse(envelope.alternative_format)})}};
            }
            if (!envelope.reasoning_close.empty()) {
                // any_text excludes the first closing delimiter, including split-token and
                // overlapping prefixes. A wildcard repetition would allow reasoning to consume
                // the delimiter and bypass the final-content constraint.
                content =
                    Json{{"type", "sequence"},
                         {"elements",
                          Json::array(
                              {Json{{"type", "any_text"},
                                    {"excludes", Json::array({envelope.reasoning_close})}},
                               Json{{"type", "const_string"}, {"value", envelope.reasoning_close}},
                               content})}};
            }
            grammar = impl_->compiler.CompileStructuralTag(
                Json{{"type", "structural_tag"}, {"format", content}}.dump());
        }
        return std::shared_ptr<GrammarState>(new GrammarState(std::make_unique<GrammarState::Impl>(
            xgrammar::GrammarMatcher(grammar), impl_->tokenizer.GetVocabSize(),
            impl_->vocabulary, impl_->stops, std::move(unique))));
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("cannot compile JSON schema: ") + e.what());
    }
}
} // namespace ninfer::text
