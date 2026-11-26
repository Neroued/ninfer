#include "models/inkling2/frontend/tool_grammar.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "text/json_schema.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {
namespace {
using Json = nlohmann::ordered_json;

xgrammar::Grammar tool_arguments_grammar(const models::frontend::ToolDeclaration& tool) {
    if (!tool.strict) {
        return text::build_json_grammar(OutputConstraint{.kind = OutputConstraintKind::JsonObject});
    }
    try {
        return text::build_json_grammar(
            OutputConstraint{.kind = OutputConstraintKind::JsonSchema, .source = tool.schema_json});
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    }
}

xgrammar::Grammar build_grammar(const text::GrammarCompiler& compiler, const FramingTokens& framing,
                                InitialBlock initial_block,
                                const std::optional<OutputConstraint>& body,
                                const std::shared_ptr<const ToolCallOutputContract>& tools,
                                bool allow_thinking, bool thinking_budget_active) {
    const std::vector<int32_t> all_framing = {
        framing.message_model,
        framing.content_thinking,
        framing.content_text,
        framing.content_invoke_tool_json,
        framing.end_message,
    };

    xgrammar::GrammarBuilder builder;
    const auto tok = [&](TokenId id) { return builder.AddTokenSet({id}); };

    const int32_t free_body = builder.AddRuleWithHint(
        "free_body",
        builder.AddRepeatFromExpr("free_tok", builder.AddExcludeTokenSet(all_framing), 0, -1));

    const int32_t thinking_after_m = builder.AddRuleWithHint(
        "thinking_after_m",
        builder.AddSequence({tok(framing.content_thinking), builder.AddRuleRef(free_body),
                             tok(framing.end_message)}));
    const int32_t thinking_block = builder.AddRuleWithHint(
        "thinking_block",
        builder.AddSequence({tok(framing.message_model), builder.AddRuleRef(thinking_after_m)}));

    const int32_t free_text_after_m = builder.AddRuleWithHint(
        "free_text_after_m",
        builder.AddSequence(
            {tok(framing.content_text), builder.AddRuleRef(free_body), tok(framing.end_message)}));
    const int32_t free_text_block = builder.AddRuleWithHint(
        "free_text_block",
        builder.AddSequence({tok(framing.message_model), builder.AddRuleRef(free_text_after_m)}));

    const bool has_tools     = tools && !tools->tools.empty();
    const bool tools_forbid  = tools && tools->tools.empty();
    const bool tools_require = has_tools && tools->required;
    const bool single_think  = thinking_budget_active && tools_require && !body;

    int32_t body_rule                 = -1;
    int32_t constrained_text_after_m  = -1;
    int32_t constrained_text_block    = -1;
    if (body) {
        body_rule = xgrammar::SubGrammarAdder::Apply(&builder, compiler.build_constraint(*body));
        constrained_text_after_m = builder.AddRuleWithHint(
            "constrained_text_after_m",
            builder.AddSequence({tok(framing.content_text), builder.AddRuleRef(body_rule),
                                 tok(framing.end_message)}));
        constrained_text_block = builder.AddRuleWithHint(
            "constrained_text_block",
            builder.AddSequence(
                {tok(framing.message_model), builder.AddRuleRef(constrained_text_after_m)}));
    }

    int32_t tool_calls_between = -1;
    int32_t tool_calls_header  = -1;
    if (has_tools) {
        std::vector<int32_t> call_choices;
        call_choices.reserve(tools->tools.size());
        for (const auto& tool : tools->tools) {
            const int32_t args_rule =
                xgrammar::SubGrammarAdder::Apply(&builder, tool_arguments_grammar(tool));
            const std::string name_json = Json(tool.name).dump();
            call_choices.push_back(builder.AddSequence(
                {builder.AddByteString(tool.name), tok(framing.content_invoke_tool_json),
                 builder.AddByteString("{\"name\":" + name_json + ",\"args\":"),
                 builder.AddRuleRef(args_rule), builder.AddByteString("}"),
                 tok(framing.end_message)}));
        }
        const int32_t tool_after_m =
            builder.AddRuleWithHint("tool_after_m", builder.AddChoices(call_choices));
        const int32_t tool_block = builder.AddRuleWithHint(
            "tool_block",
            builder.AddSequence({tok(framing.message_model), builder.AddRuleRef(tool_after_m)}));
        if (tools->parallel) {
            tool_calls_between = builder.AddRuleWithHint(
                "tool_calls_between",
                builder.AddRepeatFromExpr("tool_call_rep", builder.AddRuleRef(tool_block), 1, -1));
            tool_calls_header = builder.AddRuleWithHint(
                "tool_calls_header",
                builder.AddSequence(
                    {builder.AddRuleRef(tool_after_m),
                     builder.AddRepeatFromExpr("tool_call_tail", builder.AddRuleRef(tool_block), 0,
                                               -1)}));
        } else {
            tool_calls_between = tool_block;
            tool_calls_header  = tool_after_m;
        }
    }

    const auto think_or_text_between = [&]() {
        if (!allow_thinking) {
            return builder.AddRepeatFromExpr("text_star", builder.AddRuleRef(free_text_block), 0,
                                             -1);
        }
        const int32_t block = builder.AddRuleWithHint(
            "think_or_text_block",
            builder.AddChoices(
                {builder.AddRuleRef(thinking_block), builder.AddRuleRef(free_text_block)}));
        return builder.AddRepeatFromExpr("think_or_text_star", builder.AddRuleRef(block), 0, -1);
    };

    const auto build_between_root = [&]() -> int32_t {
        if (body && !has_tools) {
            if (!allow_thinking) return builder.AddRuleRef(constrained_text_block);
            return builder.AddSequence(
                {builder.AddRepeatFromExpr("think_star", builder.AddRuleRef(thinking_block), 0, -1),
                 builder.AddRuleRef(constrained_text_block)});
        }
        if (tools_forbid && !body) { return think_or_text_between(); }
        if (tools_require) {
            if (!allow_thinking) return builder.AddRuleRef(tool_calls_between);
            const int32_t think_prefix = builder.AddRepeatFromExpr(
                "think_prefix", builder.AddRuleRef(thinking_block), 0, single_think ? 1 : -1);
            return builder.AddSequence({think_prefix, builder.AddRuleRef(tool_calls_between)});
        }
        if (has_tools && body) {
            const int32_t tail = builder.AddChoices(
                {builder.AddRuleRef(constrained_text_block), builder.AddRuleRef(tool_calls_between)});
            if (!allow_thinking) return tail;
            return builder.AddSequence(
                {builder.AddRepeatFromExpr("think_star", builder.AddRuleRef(thinking_block), 0, -1),
                 tail});
        }
        if (has_tools) {
            return builder.AddSequence(
                {think_or_text_between(),
                 builder.AddRepeatFromExpr("opt_tools", builder.AddRuleRef(tool_calls_between), 0,
                                           1)});
        }
        throw std::logic_error("Inkling2 grammar requested without constraint or tools");
    };

    const auto build_header_root = [&]() -> int32_t {
        const int32_t think_then_m = builder.AddRuleWithHint(
            "think_then_m",
            builder.AddSequence({builder.AddRuleRef(thinking_after_m), tok(framing.message_model)}));
        if (body && !has_tools) {
            if (!allow_thinking) return builder.AddRuleRef(constrained_text_after_m);
            return builder.AddSequence(
                {builder.AddRepeatFromExpr("think_m_star", builder.AddRuleRef(think_then_m), 0, -1),
                 builder.AddRuleRef(constrained_text_after_m)});
        }
        if (tools_forbid && !body) {
            const int32_t first = allow_thinking
                                      ? builder.AddChoices({builder.AddRuleRef(thinking_after_m),
                                                            builder.AddRuleRef(free_text_after_m)})
                                      : builder.AddRuleRef(free_text_after_m);
            return builder.AddSequence({first, think_or_text_between()});
        }
        if (tools_require) {
            if (!allow_thinking) return builder.AddRuleRef(tool_calls_header);
            const int32_t think_prefix = builder.AddRepeatFromExpr(
                "think_m_prefix", builder.AddRuleRef(think_then_m), 0, single_think ? 1 : -1);
            return builder.AddSequence({think_prefix, builder.AddRuleRef(tool_calls_header)});
        }
        if (has_tools && body) {
            const int32_t tail = builder.AddChoices({builder.AddRuleRef(constrained_text_after_m),
                                                     builder.AddRuleRef(tool_calls_header)});
            if (!allow_thinking) return tail;
            return builder.AddSequence(
                {builder.AddRepeatFromExpr("think_m_star", builder.AddRuleRef(think_then_m), 0, -1),
                 tail});
        }
        if (has_tools) {
            const int32_t first = allow_thinking
                                      ? builder.AddChoices({builder.AddRuleRef(thinking_after_m),
                                                            builder.AddRuleRef(free_text_after_m)})
                                      : builder.AddRuleRef(free_text_after_m);
            const int32_t non_tool_first = builder.AddSequence(
                {first, think_or_text_between(),
                 builder.AddRepeatFromExpr("opt_tools", builder.AddRuleRef(tool_calls_between), 0,
                                           1)});
            return builder.AddChoices({builder.AddRuleRef(tool_calls_header), non_tool_first});
        }
        throw std::logic_error("Inkling2 grammar requested without constraint or tools");
    };

    int32_t root_expr = -1;
    switch (initial_block) {
    case InitialBlock::BetweenBlocks:
        root_expr = build_between_root();
        break;
    case InitialBlock::MessageHeader:
        root_expr = build_header_root();
        break;
    case InitialBlock::Text:
        if (body) {
            root_expr =
                builder.AddSequence({builder.AddRuleRef(body_rule), tok(framing.end_message)});
        } else if (tools_forbid) {
            root_expr = builder.AddSequence(
                {builder.AddRuleRef(free_body), tok(framing.end_message), think_or_text_between()});
        } else if (tools_require) {
            root_expr = builder.AddSequence({builder.AddRuleRef(free_body),
                                             tok(framing.end_message),
                                             builder.AddRuleRef(tool_calls_between)});
        } else if (has_tools) {
            root_expr = builder.AddSequence(
                {builder.AddRuleRef(free_body), tok(framing.end_message), think_or_text_between(),
                 builder.AddRepeatFromExpr("opt_tools", builder.AddRuleRef(tool_calls_between), 0,
                                           1)});
        } else {
            throw std::logic_error("Inkling2 grammar requested without constraint or tools");
        }
        break;
    case InitialBlock::Thinking:
        root_expr = builder.AddSequence(
            {builder.AddRuleRef(free_body), tok(framing.end_message), build_between_root()});
        break;
    }

    return xgrammar::GrammarNormalizer::Apply(
        builder.Get(builder.AddRuleWithHint("root", root_expr)));
}

} // namespace

std::unique_ptr<text::GrammarSession>
compile_inkling_grammar(text::GrammarCompiler& compiler, const FramingTokens& framing,
                        InitialBlock initial_block, std::string_view continuation,
                        const std::optional<OutputConstraint>& body,
                        const std::shared_ptr<const ToolCallOutputContract>& tools,
                        bool allow_thinking, bool thinking_budget_active) {
    Json identity{
        {"inkling2_grammar", 1},
        {"initial_block", static_cast<int>(initial_block)},
        {"framing",
         {framing.message_model, framing.content_thinking, framing.content_text,
          framing.content_invoke_tool_json, framing.end_message}},
        {"allow_thinking", allow_thinking},
        {"budget", thinking_budget_active},
    };
    if (body) {
        if (body->kind == OutputConstraintKind::Choice) {
            auto choices = body->choices;
            std::sort(choices.begin(), choices.end());
            choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
            identity["body"] = {
                {"kind", static_cast<int>(body->kind)}, {"choices", std::move(choices)}};
        } else {
            identity["body"] = {
                {"kind", static_cast<int>(body->kind)}, {"source", body->source}};
        }
    }
    if (tools) {
        Json tool_list = Json::array();
        for (const auto& tool : tools->tools) {
            Json entry{{"name", tool.name}, {"strict", tool.strict}};
            if (tool.strict) {
                entry["schema"]       = Json::parse(tool.schema_json);
                entry["source_index"] = tool.declaration_index;
            }
            tool_list.push_back(std::move(entry));
        }
        identity["tools"] = {{"list", std::move(tool_list)},
                             {"required", tools->required},
                             {"parallel", tools->parallel}};
    }

    // Only text continuation against an active body constraint needs AcceptString prefixing.
    const std::string_view grammar_continuation =
        (initial_block == InitialBlock::Text && body) ? continuation : std::string_view{};

    try {
        return compiler.compile_model(
            identity.dump(),
            [&] {
                return build_grammar(compiler, framing, initial_block, body, tools, allow_thinking,
                                     thinking_budget_active);
            },
            /*reasoning_close=*/{}, grammar_continuation);
    } catch (const RequestError&) {
        throw;
    } catch (const xgrammar::LogFatalError& error) {
        const auto kind =
            body ? text::constraint_error_kind(body->kind) : RequestErrorKind::InvalidToolConstraint;
        throw RequestError(kind, error.what());
    } catch (const std::invalid_argument& error) {
        const auto kind =
            body ? text::constraint_error_kind(body->kind) : RequestErrorKind::InvalidToolConstraint;
        throw RequestError(kind, error.what());
    }
}

} // namespace ninfer::models::inkling2
