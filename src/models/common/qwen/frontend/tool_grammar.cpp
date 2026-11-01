#include "models/common/qwen/frontend/tool_grammar.h"
#include "models/common/qwen/frontend/tool_format.h"
#include "models/common/qwen/frontend/tool_name_constraint.h"
#include "text/json_schema.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "json_schema_converter_ext.h"
#include "json_string_grammar.h"

#include <nlohmann/json.hpp>

#include <optional>

namespace ninfer::models::qwen::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

xgrammar::Grammar strict_arguments(const Contract::Tool& tool) {
    try {
        const std::string source = text::prepare_json_schema(tool.schema_json);
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, false, xgrammar::JSONFormat::kQwenXML, {});
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
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
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters",
                           RequestErrorSource::Tools);
    }
}

int32_t basic_arguments(xgrammar::GrammarBuilder& builder, const Contract::Tool& tool,
                        xgrammar::StringDispatch& names, int32_t json) {
    std::vector<std::pair<std::string, int32_t>> cases;
    for (const auto& item : tool.parameters)
        if (valid_parameter_name(item.name) && item.encoding == Contract::Encoding::JsonValue)
            cases.emplace_back(item.name, json);
    const auto parameter = builder.AddSequence(
        {builder.AddByteString(std::string(kParamOpen)), builder.AddRuleRef(names.Add(cases))});
    return builder.AddRuleWithHint("parameters",
                                   builder.AddRepeatFromExpr("parameter", parameter, 0, -1));
}

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body) {
    std::optional<xgrammar::Grammar> json;
    if (body) json = text::build_json_grammar(*body);
    if (json && contract.tools.empty()) return *json;
    xgrammar::GrammarBuilder builder;
    if (contract.tools.empty())
        return builder.Get(
            builder.AddRuleWithHint("root", builder.AddTagDispatch({{}, false, {"<tool_call>"}})));
    std::vector<int32_t> choices;
    std::optional<xgrammar::StringDispatch> names;
    int32_t json_value = -1;
    for (const auto& tool : contract.tools) {
        if (!tool.strict && !names) {
            const auto raw_value = xgrammar::SubGrammarAdder::Apply(
                &builder,
                xgrammar::StringConstraints({}, 0, -1, {std::string(kParameterEnd)}, false));
            xgrammar::XMLToolCallingConverter json_codec(std::nullopt, std::nullopt, true,
                                                         std::nullopt);
            const auto json =
                xgrammar::SubGrammarAdder::Apply(&builder, json_codec.ConvertJSONValue());
            const auto value = [&](int32_t body) {
                return builder.AddRuleWithHint(
                    "parameter_value",
                    builder.AddSequence(
                        {builder.AddRuleRef(body),
                         builder.AddByteString(std::string(kParameterEnd) + "\n")}));
            };
            json_value = value(json);
            names.emplace(builder, ">\n", value(raw_value), "<>\r\n", true);
        }
        const auto parameters =
            tool.strict ? xgrammar::SubGrammarAdder::Apply(&builder, strict_arguments(tool))
                        : basic_arguments(builder, tool, *names, json_value);
        choices.push_back(builder.AddSequence(
            {builder.AddByteString("\n<function=" + tool.name + ">\n"),
             builder.AddRuleRef(parameters), builder.AddByteString("</function>\n</tool_call>")}));
    }
    const auto call_body = builder.AddRuleWithHint("call_body", builder.AddChoices(choices));
    const auto next_call = builder.AddSequence(
        {builder.AddByteString("\n<tool_call>"), builder.AddRuleRef(call_body)});
    const auto tail  = contract.parallel ? builder.AddRepeatFromExpr("calls", next_call, 0, -1)
                                         : builder.AddEmptyStr();
    const auto first = builder.AddRuleWithHint(
        "calls", builder.AddSequence({builder.AddRuleRef(call_body), tail}));
    const auto root =
        (contract.required || body)
            ? builder.AddSequence({builder.AddByteString("<tool_call>"), builder.AddRuleRef(first)})
            : builder.AddTagDispatch({{{"<tool_call>", first}}, false, {}});
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
}
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel}};
    if (body && !body->choices.empty())
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    if (body) identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"] = Json::parse(tool.schema_json);
            // Failed compilations are cached too, including their declaration error location.
            entry["source_index"] = tool.declaration_index;
        } else {
            entry["encodings"] = Json::array();
            for (const auto& parameter : tool.parameters)
                if (valid_parameter_name(parameter.name))
                    entry["encodings"].push_back(
                        {parameter.name, static_cast<int>(parameter.encoding)});
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body); }, reasoning_close, continuation,
        contract.tools.empty()
            ? nullptr
            : make_tool_name_constraint(compiler.vocabulary(), reasoning_close, body.has_value()));
}
} // namespace ninfer::models::qwen::frontend
