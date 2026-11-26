#include "models/inkling2/frontend/output.h"

#include "models/inkling2/frontend/block_decoder.h"
#include "models/inkling2/frontend/tool_grammar.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2 {

OutputResources::OutputResources(const FrontendResources& resources, std::size_t cache_bytes)
    : text(resources.text, cache_bytes,
           std::vector<TokenId>{resources.framing.message_model,
                                resources.framing.content_thinking,
                                resources.framing.content_text,
                                resources.framing.content_invoke_tool_json,
                                resources.framing.end_message}),
      framing(resources.framing) {
    if (!resources.control_to_text || resources.control_to_text->empty() ||
        !resources.control_to_header || resources.control_to_header->empty()) {
        throw std::invalid_argument("Inkling2 thinking control tokens have not been prepared");
    }
    to_text = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kEndMessage), resources.control_to_text, false);
    to_header = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kEndMessage), resources.control_to_header, false);
}

models::frontend::OutputSession
create_output_session(const OutputResources& resources, const PreparedOutput& prompt,
                      const StopPolicy& stop, OutputOptions output, ThinkingControlOptions thinking,
                      const std::optional<OutputConstraint>& constraint, const ToolChoice& choice) {
    const bool may_think =
        prompt.thinking_enabled && prompt.initial_block != InitialBlock::Text;
    if (!may_think) thinking.budget.reset();

    auto policy = resources.text.stop_policy(stop);
    if (output.raw) policy.publish_stop_token = true;

    auto tools = select_tool_call_contract(prompt.tools, choice);
    if (constraint && tools && tools->tools.empty()) tools.reset();
    const bool combined = constraint && tools && !tools->tools.empty();
    if (combined) {
        if (constraint->kind != OutputConstraintKind::JsonObject &&
            constraint->kind != OutputConstraintKind::JsonSchema) {
            throw RequestError(
                RequestErrorKind::UnsupportedOutputCombination,
                "active Inkling2 tools can only be combined with JSON output constraints");
        }
        if (!tools->constrained) {
            auto selected         = std::make_shared<ToolCallOutputContract>(*tools);
            selected->constrained = true;
            tools                 = std::move(selected);
        }
    }

    const bool tool_constraint = tools && tools->constrained;
    const bool tool_required   = tools && !tools->tools.empty() && tools->required;
    std::unique_ptr<text::GrammarSession> matcher;
    if (constraint || tool_constraint) {
        if (!stop.strings.empty()) {
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "output constraints require model EOS and cannot use custom stops",
                               constraint ? std::string{} : std::string("stop"));
        }
        if (resources.text.defaults.token_ids.empty() || !stop.token_ids.empty() ||
            !stop.include_model_defaults || stop.publish_stop_token || output.raw ||
            output.preserve_special_tokens) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               "constraints require default EOS, text output, no custom stops, and "
                               "one output language");
        }
        if (tool_constraint) {
            for (const auto& tool : tools->tools) {
                if (!valid_tool_name(tool.name, output.tool_name_max_length)) {
                    throw RequestError(
                        RequestErrorKind::InvalidToolConstraint,
                        "tool name is outside the Inkling2 output contract: " + tool.name,
                        "/" + std::to_string(tool.declaration_index) + "/name",
                        RequestErrorSource::Tools);
                }
            }
        }
        matcher = compile_inkling_grammar(resources.text.grammars(), resources.framing,
                                          prompt.initial_block, prompt.continuation_content,
                                          constraint, tool_constraint ? tools : nullptr,
                                          thinking.budget.has_value());
    }

    std::unique_ptr<models::frontend::StructuredOutputDecoder> structured;
    if (!output.raw) {
        structured = std::make_unique<InklingBlockDecoder>(
            resources.framing, prompt.initial_block,
            // Ordinary constrained text without active tools remains literal.
            (constraint && !combined) ? nullptr : std::move(tools), output.tool_name_max_length,
            output.preserve_special_tokens, may_think);
    }

    const auto reasoning_format =
        (tool_required && !constraint) ? resources.to_header : resources.to_text;
    const bool starts_in_reasoning = prompt.initial_block == InitialBlock::Thinking;

    return models::frontend::OutputSession(
        {.tokenizer           = resources.text.tokenizer,
         .reasoning           = reasoning_format,
         .stop                = std::move(policy),
         .presentation        = output,
         .starts_in_reasoning = starts_in_reasoning,
         .has_content_prefix  = !prompt.continuation_content.empty(),
         .thinking            = thinking},
        /*body=*/nullptr, std::move(matcher), std::move(structured));
}

} // namespace ninfer::models::inkling2
