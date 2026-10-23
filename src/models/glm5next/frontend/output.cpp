#include "models/glm5next/frontend/output.h"
#include "models/glm5next/frontend/tool_decoder.h"
#include "models/glm5next/frontend/tool_grammar.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {
constexpr std::string_view kThinkClose = "</think>";
}

OutputResources::OutputResources(const models::frontend::TextResources& resources,
                                 std::size_t cache_bytes)
    : text(resources, cache_bytes) {
    const auto control =
        std::make_shared<const std::vector<TokenId>>(text.tokenizer->encode(kThinkClose));
    if (control->empty() || text.tokenizer->decode(*control) != kThinkClose ||
        text.tokenizer->decode(*control, {.skip_special_tokens = true}) != kThinkClose)
        throw std::invalid_argument("GLM tokenizer cannot present the thinking control suffix");
    for (auto token : *control)
        if (std::find(text.defaults.token_ids.begin(), text.defaults.token_ids.end(), token) !=
            text.defaults.token_ids.end())
            throw std::invalid_argument("GLM thinking control contains a terminal token");
    normal = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kThinkClose), std::string(kThinkClose), control, false, std::string(kToolOpen));
    constrained = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kThinkClose), std::string(kThinkClose), control, false);
}

models::frontend::OutputSession
create_output_session(const OutputResources& resources, const PreparedOutput& prompt,
                      const StopPolicy& stop, OutputOptions output, ThinkingControlOptions thinking,
                      const std::optional<OutputConstraint>& constraint, const ToolChoice& choice) {
    if (!prompt.starts_in_reasoning) thinking.budget.reset();
    auto policy = resources.text.stop_policy(stop);
    if (output.raw) policy.publish_stop_token = true;
    auto tools = select_tool_call_contract(prompt.tools, choice);
    if (constraint && tools && tools->tools.empty()) tools.reset();
    const bool combined = constraint && tools && !tools->tools.empty();
    if (combined) {
        if (constraint->kind != OutputConstraintKind::JsonObject &&
            constraint->kind != OutputConstraintKind::JsonSchema)
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "active GLM tools can be combined with JSON output constraints");
        if (!tools->constrained) {
            auto selected         = std::make_shared<ToolCallOutputContract>(*tools);
            selected->constrained = true;
            tools                 = std::move(selected);
        }
    }
    const bool tool_constraint = tools && tools->constrained;
    std::unique_ptr<text::GrammarSession> matcher;
    if (constraint || tool_constraint) {
        if (!stop.strings.empty())
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "output constraints require model EOS and cannot use custom stops",
                               constraint ? std::string{} : std::string("stop"));
        if (resources.text.defaults.token_ids.empty() || !stop.token_ids.empty() ||
            !stop.include_model_defaults || stop.publish_stop_token || output.raw ||
            output.preserve_special_tokens)
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               "constraints require default EOS, text output, no custom stops, and "
                               "one output language");
        const auto close = prompt.starts_in_reasoning ? kThinkClose : std::string_view{};
        if (tool_constraint) {
            for (const auto& tool : tools->tools)
                if (!valid_tool_name(tool.name, output.tool_name_max_length))
                    throw RequestError(RequestErrorKind::InvalidToolConstraint,
                                       "tool name is outside the GLM output contract: " + tool.name,
                                       "/" + std::to_string(tool.declaration_index) + "/name",
                                       RequestErrorSource::Tools);
            matcher = compile_tool_grammar(resources.text.grammars(), tools, close,
                                           prompt.continuation_content, constraint);
        } else
            matcher =
                resources.text.grammars().compile(*constraint, close, prompt.continuation_content);
    }
    std::unique_ptr<models::frontend::BodyDecoder> decoder;
    // Ordinary constrained text must remain literal even if it contains a tool marker.
    if (!output.raw && tools) {
        auto body = std::make_unique<ToolCallOutputDecoder>(std::move(tools),
                                                            output.tool_name_max_length, combined);
        if (!prompt.continuation_content.empty())
            body->initialize_continuation(prompt.continuation_content);
        decoder = std::move(body);
    }
    return models::frontend::OutputSession(
        {.tokenizer           = resources.text.tokenizer,
         .reasoning           = matcher ? resources.constrained : resources.normal,
         .stop                = std::move(policy),
         .presentation        = output,
         .starts_in_reasoning = prompt.starts_in_reasoning,
         .has_content_prefix  = !prompt.continuation_content.empty(),
         .thinking            = thinking},
        std::move(decoder), std::move(matcher));
}
} // namespace ninfer::models::glm5next
