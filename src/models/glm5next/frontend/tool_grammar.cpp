#include "models/glm5next/frontend/tool_grammar.h"
#include "text/json_schema.h"
#include "text/unicode.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "json_schema_converter_ext.h"
#include "json_string_grammar.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::glm5next {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

xgrammar::Grammar strict_arguments(const Contract::Tool& tool) {
    try {
        return xgrammar::JSONSchemaToGrammar(
            text::prepare_json_schema(tool.schema_json), false, std::nullopt,
            std::pair<std::string, std::string>{", ", ": "}, false, std::nullopt, false,
            xgrammar::JSONFormat::kGlmXML, {});
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
                              ? RequestErrorKind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? RequestErrorKind::UnsatisfiableJsonSchema
                              : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer,
                           RequestErrorSource::Tools);
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    }
}

// A trie excludes exactly the declared names. StringConstraints also excludes the closing tag,
// including occurrences straddling a trie edge and the unrestricted suffix.
xgrammar::Grammar open_names(const Contract::Tool& tool) {
    struct Node {
        std::map<std::int32_t, std::size_t> children;
        bool declared = false;
    };

    std::vector<Node> nodes(1);
    for (const auto& parameter : tool.parameters) {
        if (!valid_parameter_name(parameter.name)) continue;
        std::size_t node = 0;
        for (const auto cp : text::unicode_internal::utf8_codepoints(parameter.name, "parameter")) {
            const auto found = nodes[node].children.find(cp.value);
            if (found != nodes[node].children.end())
                node = found->second;
            else {
                const auto next = nodes.size();
                nodes[node].children.emplace(cp.value, next);
                nodes.emplace_back();
                node = next;
            }
        }
        nodes[node].declared = true;
    }
    const auto escape = [](std::int32_t cp) {
        auto value = text::unicode_internal::codepoint_to_utf8(cp);
        if (cp < 128 && std::string_view("\\.^$|?*+()[]{}-").find(static_cast<char>(cp)) !=
                            std::string_view::npos)
            value.insert(0, "\\");
        return value;
    };
    std::vector<std::string> rules(nodes.size());
    for (std::size_t i = nodes.size(); i-- > 0;) {
        std::string choices, excluded;
        for (const auto& [cp, child] : nodes[i].children) {
            if (!choices.empty()) choices += '|';
            choices += escape(cp) + rules[child];
            excluded += escape(cp);
        }
        if (!choices.empty()) choices += '|';
        choices +=
            (excluded.empty() ? "[\\s\\S]" : "[^" + excluded + "]") + std::string("[\\s\\S]*");
        if (!nodes[i].declared) choices += '|';
        rules[i] = "(?:" + choices + ")";
    }
    return xgrammar::StringConstraints({"^" + rules.front() + "$"}, 0, -1, {std::string(kKeyClose)},
                                       false);
}

int32_t basic_arguments(xgrammar::GrammarBuilder& builder, const Contract::Tool& tool, int32_t raw,
                        int32_t json) {
    const auto parameter = [&](int32_t name, int32_t value) {
        return builder.AddSequence(
            {builder.AddByteString(std::string(kKeyOpen)), name,
             builder.AddByteString(std::string(kKeyClose) + std::string(kValueOpen)),
             builder.AddRuleRef(value), builder.AddByteString(std::string(kValueClose))});
    };
    std::vector<int32_t> alternatives;
    for (const auto& item : tool.parameters)
        if (valid_parameter_name(item.name))
            alternatives.push_back(
                parameter(builder.AddByteString(item.name),
                          item.encoding == Contract::Encoding::JsonValue ? json : raw));
    const auto other = xgrammar::SubGrammarAdder::Apply(&builder, open_names(tool));
    alternatives.push_back(parameter(builder.AddRuleRef(other), raw));
    return builder.AddRuleWithHint(
        "parameters",
        builder.AddRepeatFromExpr("parameter", builder.AddChoices(alternatives), 0, -1));
}

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body) {
    std::optional<xgrammar::Grammar> json;
    if (body) json = text::build_json_grammar(*body);
    if (json && contract.tools.empty()) return *json;
    xgrammar::GrammarBuilder builder;
    if (contract.tools.empty())
        return builder.Get(builder.AddRuleWithHint(
            "root", builder.AddTagDispatch({{}, false, {std::string(kToolOpen)}})));
    std::optional<int32_t> raw_value, json_value;
    std::vector<int32_t> choices;
    for (const auto& tool : contract.tools) {
        if (!tool.strict && !raw_value) {
            raw_value = xgrammar::SubGrammarAdder::Apply(
                &builder,
                xgrammar::StringConstraints({}, 0, -1, {std::string(kValueClose)}, false));
            xgrammar::XMLToolCallingConverter codec(std::nullopt, std::nullopt, true, std::nullopt,
                                                    {}, xgrammar::JSONFormat::kGlmXML);
            json_value = xgrammar::SubGrammarAdder::Apply(&builder, codec.ConvertJSONValue());
        }
        const auto parameters =
            tool.strict ? xgrammar::SubGrammarAdder::Apply(&builder, strict_arguments(tool))
                        : basic_arguments(builder, tool, *raw_value, *json_value);
        choices.push_back(
            builder.AddSequence({builder.AddByteString(tool.name), builder.AddRuleRef(parameters),
                                 builder.AddByteString(std::string(kToolClose))}));
    }
    const auto call = builder.AddRuleWithHint("call", builder.AddChoices(choices));
    const auto next = builder.AddSequence(
        {builder.AddByteString(std::string(kToolOpen)), builder.AddRuleRef(call)});
    const auto tail =
        contract.parallel ? builder.AddRepeatFromExpr("calls", next, 0, -1) : builder.AddEmptyStr();
    const auto calls =
        builder.AddRuleWithHint("calls", builder.AddSequence({builder.AddRuleRef(call), tail}));
    const auto root = contract.required || body
                          ? builder.AddSequence({builder.AddByteString(std::string(kToolOpen)),
                                                 builder.AddRuleRef(calls)})
                          : builder.AddTagDispatch({{{std::string(kToolOpen), calls}}, false, {}});
    auto grammar =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, grammar}) : grammar;
}

bool match(std::string_view marker, std::size_t& matched, char byte) {
    while (matched && marker[matched] != byte) {
        auto n = matched - 1;
        while (n && marker.substr(0, n) != marker.substr(matched - n, n)) --n;
        matched = n;
    }
    if (marker[matched] == byte) ++matched;
    if (matched != marker.size()) return false;
    matched = 0;
    return true;
}

// Static rules own values. This small transactional scanner rejects duplicate argument names.
// It follows the selected codec so a closing tag inside a JSON string stays value data.
class UniqueNames final : public text::GrammarStateConstraint {
    enum class Phase {
        Reasoning,
        Branch,
        Body,
        Text,
        Function,
        Parameters,
        Name,
        ValueStart,
        Value
    };

    struct Cursor {
        Phase phase         = Phase::Text;
        std::size_t matched = 0, closing = 0, call = 0, begin = 0;
        const Contract::Tool* tool = nullptr;
        bool json = false, quoted = false, escaped = false;
    };

    struct Inserted {
        std::size_t call;
        const std::string* name;
    };

    struct Mark {
        Cursor cursor;
        std::size_t bytes, inserted;
    };
public:
    UniqueNames(std::shared_ptr<const text::GrammarVocabulary> vocabulary,
                std::shared_ptr<const Contract> contract, std::string_view close, bool combined)
        : vocabulary_(std::move(vocabulary)), endings_(vocabulary_->containing('>')),
          contract_(std::move(contract)), close_(close), combined_(combined) {
        cursor_.phase = close.empty() ? body_phase() : Phase::Reasoning;
        committed_    = mark();
    }

    void filter(std::span<std::uint32_t> words) override {
        if (cursor_.phase == Phase::Body) return;
        const auto before = mark();
        for (auto id : endings_) {
            if (!(words[id / 32] & (1u << (id % 32)))) continue;
            const auto& bytes = vocabulary_->pieces()[id];
            if (cursor_.phase != Phase::Name &&
                !((cursor_.phase == Phase::Parameters || cursor_.phase == Phase::Function) &&
                  cursor_.matched) &&
                bytes.find(kKeyOpen) == std::string::npos)
                continue;
            try {
                if (!consume(bytes)) words[id / 32] &= ~(1u << (id % 32));
            } catch (...) {
                restore(before);
                throw;
            }
            restore(before);
        }
    }

    bool accept(std::string_view bytes) override {
        const auto before = mark();
        try {
            if (consume(bytes)) return true;
        } catch (...) {
            restore(before);
            throw;
        }
        restore(before);
        return false;
    }

    void confirm() noexcept override {
        inserted_.clear();
        for (auto it = names_.begin(); it != names_.end();)
            if (it->first != cursor_.call)
                it = names_.erase(it);
            else
                ++it;
        if (cursor_.phase == Phase::Name || cursor_.phase == Phase::Function)
            buffer_.erase(0, cursor_.begin);
        else
            buffer_.clear();
        cursor_.begin = 0;
        committed_    = mark();
    }

    void discard() noexcept override { restore(committed_); }
private:
    Phase body_phase() const { return combined_ ? Phase::Branch : Phase::Text; }

    Mark mark() const { return {cursor_, buffer_.size(), inserted_.size()}; }

    void restore(const Mark& before) noexcept {
        while (inserted_.size() > before.inserted) {
            const auto item = inserted_.back();
            auto found      = names_.find(item.call);
            found->second.erase(*item.name);
            if (found->second.empty()) names_.erase(found);
            inserted_.pop_back();
        }
        buffer_.resize(before.bytes);
        cursor_ = before.cursor;
    }

    bool consume(std::string_view bytes) {
        for (char c : bytes) {
            switch (cursor_.phase) {
            case Phase::Reasoning:
                if (match(close_, cursor_.matched, c)) cursor_.phase = body_phase();
                break;
            case Phase::Branch:
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') break;
                cursor_.phase = c == '<' ? Phase::Text : Phase::Body;
                if (cursor_.phase == Phase::Body) break;
                [[fallthrough]];
            case Phase::Text:
                if (match(kToolOpen, cursor_.matched, c)) {
                    ++cursor_.call;
                    cursor_.begin = buffer_.size();
                    cursor_.phase = Phase::Function;
                }
                break;
            case Phase::Function:
                buffer_ += c;
                if (match(kKeyOpen, cursor_.matched, c)) {
                    const auto name = std::string_view(buffer_).substr(
                        cursor_.begin, buffer_.size() - cursor_.begin - kKeyOpen.size());
                    const auto found =
                        std::find_if(contract_->tools.begin(), contract_->tools.end(),
                                     [&](const auto& tool) { return tool.name == name; });
                    if (found == contract_->tools.end()) return false;
                    cursor_.tool    = &*found;
                    cursor_.begin   = buffer_.size();
                    cursor_.closing = 0;
                    cursor_.phase   = Phase::Name;
                } else if (match(kToolClose, cursor_.closing, c)) {
                    cursor_.matched = 0;
                    cursor_.phase   = Phase::Text;
                }
                break;
            case Phase::Parameters:
                if (match(kKeyOpen, cursor_.matched, c)) {
                    cursor_.begin   = buffer_.size();
                    cursor_.closing = 0;
                    cursor_.phase   = Phase::Name;
                } else if (match(kToolClose, cursor_.closing, c)) {
                    cursor_.matched = 0;
                    cursor_.phase   = Phase::Text;
                }
                break;
            case Phase::Name:
                buffer_ += c;
                if (match(kKeyClose, cursor_.matched, c)) {
                    const auto name = buffer_.substr(cursor_.begin, buffer_.size() - cursor_.begin -
                                                                        kKeyClose.size());
                    auto& names     = names_[cursor_.call];
                    const auto [entry, added] = names.emplace(name);
                    if (!added) return false;
                    inserted_.push_back({cursor_.call, &*entry});
                    const auto found = std::find_if(
                        cursor_.tool->parameters.begin(), cursor_.tool->parameters.end(),
                        [&](const auto& parameter) { return parameter.name == name; });
                    cursor_.json = found != cursor_.tool->parameters.end() &&
                                   found->encoding == Contract::Encoding::JsonValue;
                    cursor_.quoted = cursor_.escaped = false;
                    cursor_.phase                    = Phase::ValueStart;
                }
                break;
            case Phase::ValueStart:
                if (match(kValueOpen, cursor_.matched, c)) cursor_.phase = Phase::Value;
                break;
            case Phase::Value:
                if (cursor_.json && cursor_.quoted) {
                    if (cursor_.escaped)
                        cursor_.escaped = false;
                    else if (c == '\\')
                        cursor_.escaped = true;
                    else if (c == '"')
                        cursor_.quoted = false;
                } else if (cursor_.json && c == '"') {
                    cursor_.quoted  = true;
                    cursor_.matched = 0;
                } else if (match(kValueClose, cursor_.matched, c)) {
                    cursor_.closing = 0;
                    cursor_.phase   = Phase::Parameters;
                }
                break;
            case Phase::Body:
                break;
            }
        }
        return true;
    }

    std::shared_ptr<const text::GrammarVocabulary> vocabulary_;
    const std::vector<std::int32_t>& endings_;
    std::shared_ptr<const Contract> contract_;
    std::string close_;
    bool combined_;
    Cursor cursor_;
    std::string buffer_;
    std::unordered_map<std::size_t, std::unordered_set<std::string>> names_;
    std::vector<Inserted> inserted_;
    Mark committed_;
};
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, std::shared_ptr<const Contract> bound,
                     std::string_view close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body) {
    const auto& contract = *bound;
    Json identity{{"glm_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel}};
    if (body && !body->choices.empty())
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    if (body) identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"]       = Json::parse(tool.schema_json);
            entry["source_index"] = tool.declaration_index;
        } else {
            entry["encodings"] = Json::array();
            for (const auto& item : tool.parameters)
                if (valid_parameter_name(item.name))
                    entry["encodings"].push_back({item.name, static_cast<int>(item.encoding)});
        }
        identity["glm_tools"].push_back(std::move(entry));
    }
    try {
        return compiler.compile_model(
            identity.dump(), [&] { return build(contract, body); }, close, continuation,
            contract.tools.empty() ? nullptr
                                   : std::make_unique<UniqueNames>(compiler.vocabulary(), bound,
                                                                   close, body.has_value()));
    } catch (const RequestError&) { throw; } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint, error.what(), {},
                           RequestErrorSource::Tools);
    } catch (const std::invalid_argument& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint, error.what(), {},
                           RequestErrorSource::Tools);
    }
}
} // namespace ninfer::models::glm5next
