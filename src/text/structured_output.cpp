#include "text/structured_output.h"
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

void schema_check(const Json& s) {
    if (s.is_boolean()) { return; }
    if (!s.is_object()) { throw std::invalid_argument("JSON schema must be an object or boolean"); }
    static const std::unordered_set<std::string> annotations = {
        "$schema",  "title",    "description", "default",
        "examples", "$comment", "$defs",       "definitions"};
    static const std::unordered_set<std::string> supported = {"type",
                                                              "properties",
                                                              "required",
                                                              "additionalProperties",
                                                              "items",
                                                              "prefixItems",
                                                              "minItems",
                                                              "maxItems",
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
        if ((key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems") &&
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
            schema_check(Json::parse(options.schema));
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

    Impl(xgrammar::GrammarMatcher matcher, int vocab_size)
        : matcher(std::move(matcher)), vocab_size(vocab_size) {}
};

GrammarState::GrammarState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GrammarState::~GrammarState()                                  = default;
GrammarState::GrammarState(GrammarState&&) noexcept            = default;
GrammarState& GrammarState::operator=(GrammarState&&) noexcept = default;

std::unique_ptr<GrammarState> GrammarState::fork() const {
    return std::unique_ptr<GrammarState>(
        new GrammarState(std::make_unique<Impl>(impl_->matcher.Fork(), impl_->vocab_size)));
}

void GrammarState::accept(std::span<const TokenId> tokens) {
    for (TokenId token : tokens) {
        if (impl_->matcher.IsTerminated() || !impl_->matcher.AcceptToken(token)) {
            throw std::logic_error("generated token " + std::to_string(token) +
                                   " violates the structured output grammar");
        }
    }
}

void GrammarState::fill_masks(std::span<std::uint32_t> masks,
                              std::span<const TokenId> drafts) const {
    const int words = xgrammar::GetBitmaskSize(impl_->vocab_size);
    if (masks.size() != (drafts.size() + 1) * words) {
        throw std::logic_error("incorrect grammar mask shape");
    }
    auto matcher = impl_->matcher.Fork();
    std::fill(masks.begin(), masks.end(), ~std::uint32_t{0});
    for (std::size_t col = 0; col <= drafts.size(); ++col) {
        if (matcher.IsTerminated()) { break; } // no later column can be published past EOS
        std::int64_t shape[2] = {1, words};
        DLTensor tensor{};
        tensor.data   = masks.data() + col * words;
        tensor.device = {kDLCPU, 0};
        tensor.ndim   = 2;
        tensor.dtype  = {kDLInt, 32, 1};
        tensor.shape  = shape;
        matcher.FillNextTokenBitmask(&tensor);
        if (impl_->vocab_size % 32) {
            masks[(col + 1) * words - 1] &= (1U << (impl_->vocab_size % 32)) - 1;
        }
        const auto row = masks.subspan(col * words, words);
        if (std::none_of(row.begin(), row.end(), [](auto word) { return word != 0; })) {
            throw std::runtime_error("structured output grammar has no admissible next token");
        }
        if (col < drafts.size() && !matcher.AcceptToken(drafts[col])) { break; }
    }
}

struct StructuredCompiler::Impl {
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
    std::mutex mutex;

    Impl(std::vector<std::string> vocab, std::vector<int> stops)
        : tokenizer(vocab, xgrammar::VocabType::RAW, static_cast<int>(vocab.size()), stops),
          compiler(tokenizer, 4, true, 256 * 1024 * 1024) {}
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
        // strict_mode=false retains JSON Schema defaults for additional properties/items.
        auto grammar = impl_->compiler.CompileJSONSchema(
            options.kind == StructuredOutputKind::JsonObject ? "{\"type\":\"object\"}"
                                                             : options.schema,
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
            xgrammar::GrammarMatcher(grammar), impl_->tokenizer.GetVocabSize())));
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("cannot compile JSON schema: ") + e.what());
    }
}
} // namespace ninfer::text
