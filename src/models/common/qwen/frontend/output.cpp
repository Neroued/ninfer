#include "models/common/qwen/frontend/output.h"
#include "models/common/qwen/frontend/tool_call_parser.h"
#include "models/common/qwen/frontend/tool_grammar.h"
#include "text/grammar.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen::frontend {
namespace {
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
static_assert(kThinkingControl.ends_with(kCanonicalReasoningCloseSerialization));

} // namespace

OutputResources::OutputResources(const models::frontend::TextResources& resources,
                                 std::size_t cache_bytes)
    : text(resources, cache_bytes) {
    auto control =
        std::make_shared<const std::vector<TokenId>>(text.tokenizer->encode(kThinkingControl));
    if (control->empty() || text.tokenizer->decode(*control) != kThinkingControl ||
        text.tokenizer->decode(*control, {.skip_special_tokens = true}).find(kThinkClose) ==
            std::string::npos)
        throw std::invalid_argument(
            "Qwen tokenizer cannot present the canonical thinking control suffix");
    for (auto token : *control)
        if (std::find(text.defaults.token_ids.begin(), text.defaults.token_ids.end(), token) !=
            text.defaults.token_ids.end())
            throw std::invalid_argument(
                "canonical thinking control suffix contains a default terminal token");
    normal = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kThinkClose), std::string(kCanonicalReasoningCloseSerialization), control,
        true);
    constrained = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kCanonicalReasoningCloseSerialization),
        std::string(kCanonicalReasoningCloseSerialization), control, false);
}

models::frontend::OutputSession create_output_session(
    const OutputResources& resources, const PreparedOutput& prompt, const StopPolicy& caller_stop,
    OutputOptions output, ThinkingControlOptions thinking,
    const std::optional<OutputConstraint>& constraint, const ToolChoice& tool_choice) {
    if (!prompt.starts_in_reasoning || prompt.enable_thinking == false) thinking.budget.reset();
    StopPolicy policy = resources.text.stop_policy(caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    auto tool_contract = select_tool_call_contract(prompt.tools, tool_choice);
    if (constraint && tool_contract && tool_contract->tools.empty()) tool_contract.reset();
    const bool combined = constraint && tool_contract && !tool_contract->tools.empty();
    if (combined) {
        if (constraint->kind != OutputConstraintKind::JsonObject &&
            constraint->kind != OutputConstraintKind::JsonSchema)
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "active tools can be combined with JSON output constraints");
        if (!tool_contract->constrained) {
            auto structured         = std::make_shared<ToolCallOutputContract>(*tool_contract);
            structured->constrained = true;
            tool_contract           = std::move(structured);
        }
    }
    const bool tool_constraint = tool_contract && tool_contract->constrained;
    std::unique_ptr<text::GrammarSession> matcher;
    if (constraint || tool_constraint) {
        if (!caller_stop.strings.empty())
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "output constraints require model EOS and cannot use custom stops",
                               constraint ? std::string{} : std::string("stop"));
        if (resources.text.defaults.token_ids.empty() || !caller_stop.token_ids.empty() ||
            !caller_stop.strings.empty() || !caller_stop.include_model_defaults ||
            caller_stop.publish_stop_token || output.raw || output.preserve_special_tokens) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               "constraints require default EOS, text output, no custom stops, and "
                               "one output language");
        }
        try {
            const auto close = prompt.starts_in_reasoning ? kCanonicalReasoningCloseSerialization
                                                          : std::string_view{};
            if (tool_constraint) {
                for (const auto& tool : tool_contract->tools) {
                    if (tool.name.size() > output.tool_name_max_length ||
                        !std::all_of(tool.name.begin(), tool.name.end(), [](unsigned char c) {
                            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '_' || c == '-';
                        }))
                        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                                           "tool name is outside the output contract: " +
                                               tool.name);
                }
                matcher = compile_tool_grammar(resources.text.grammars(), *tool_contract, close,
                                               prompt.continuation_content, constraint);
            } else {
                matcher = resources.text.grammars().compile(*constraint, close,
                                                            prompt.continuation_content);
            }
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               error.what());
        }
    }
    if (!constraint && !tool_constraint &&
        ((tool_contract && !tool_contract->tools.empty()) || prompt.has_tool_history))
        output.preserve_special_tokens = true;
    std::unique_ptr<models::frontend::BodyDecoder> decoder;
    if (!output.raw) {
        auto tools = std::make_unique<ToolCallOutputDecoder>(std::move(tool_contract),
                                                             output.tool_name_max_length, combined);
        if (!prompt.continuation_content.empty())
            tools->initialize_continuation(prompt.continuation_content);
        decoder = std::move(tools);
    }
    const auto reasoning = matcher ? resources.constrained : resources.normal;
    return models::frontend::OutputSession(
        {.tokenizer           = resources.text.tokenizer,
         .reasoning           = reasoning,
         .stop                = std::move(policy),
         .presentation        = output,
         .starts_in_reasoning = prompt.starts_in_reasoning,
         .has_content_prefix  = !prompt.continuation_content.empty(),
         .thinking            = thinking},
        std::move(decoder), std::move(matcher));
}

} // namespace ninfer::models::qwen::frontend
