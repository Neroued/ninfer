#include "models/common/qwen/frontend/output.h"
#include "models/common/qwen/frontend/tool_call_parser.h"
#include "models/common/qwen/frontend/tool_grammar.h"
#include "text/grammar.h"
#include "text/unicode.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen::frontend {
namespace {
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
static_assert(kThinkingControl.ends_with(kCanonicalReasoningCloseSerialization));

StopPolicy merge_stop_policy(const text::Tokenizer& tokenizer, std::span<const TokenId> defaults,
                             const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : defaults) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

} // namespace

OutputResources::OutputResources(const models::frontend::TextResources& resources,
                                 std::size_t cache_bytes)
    : tokenizer(resources.tokenizer), defaults{.token_ids = resources.stop_token_ids},
      cache_bytes_(cache_bytes) {
    auto control =
        std::make_shared<const std::vector<TokenId>>(tokenizer->encode(kThinkingControl));
    if (control->empty() || tokenizer->decode(*control) != kThinkingControl ||
        tokenizer->decode(*control, {.skip_special_tokens = true}).find(kThinkClose) ==
            std::string::npos)
        throw std::invalid_argument(
            "Qwen tokenizer cannot present the canonical thinking control suffix");
    for (auto token : *control)
        if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
            defaults.token_ids.end())
            throw std::invalid_argument(
                "canonical thinking control suffix contains a default terminal token");
    normal = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kThinkClose), std::string(kCanonicalReasoningCloseSerialization), control,
        true);
    constrained = std::make_shared<const models::frontend::ReasoningFormat>(
        std::string(kCanonicalReasoningCloseSerialization),
        std::string(kCanonicalReasoningCloseSerialization), control, false);
}

OutputResources::~OutputResources() = default;

text::GrammarCompiler& OutputResources::grammars() const {
    std::call_once(grammar_once_, [&] {
        std::vector<std::string> vocab(tokenizer->vocab_size());
        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (!tokenizer->is_valid_token(static_cast<TokenId>(id))) continue;
            const auto decoded = tokenizer->decoded_token(static_cast<TokenId>(id));
            if (!decoded.special) vocab[id] = decoded.bytes;
        }
        grammar_ = std::make_unique<text::GrammarCompiler>(std::move(vocab), defaults.token_ids,
                                                           cache_bytes_);
    });
    return *grammar_;
}

models::frontend::OutputSession create_output_session(
    const OutputResources& resources, const PreparedOutput& prompt, const StopPolicy& caller_stop,
    OutputOptions output, ThinkingControlOptions thinking,
    const std::optional<OutputConstraint>& constraint, const ToolChoice& tool_choice) {
    if (!prompt.starts_in_reasoning || prompt.enable_thinking == false) thinking.budget.reset();
    StopPolicy policy =
        merge_stop_policy(*resources.tokenizer, resources.defaults.token_ids, caller_stop);
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
        if (resources.defaults.token_ids.empty() || !caller_stop.token_ids.empty() ||
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
                matcher = compile_tool_grammar(resources.grammars(), *tool_contract, close,
                                               prompt.continuation_content, constraint);
            } else {
                matcher =
                    resources.grammars().compile(*constraint, close, prompt.continuation_content);
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
        {.tokenizer           = resources.tokenizer,
         .reasoning           = reasoning,
         .stop                = std::move(policy),
         .presentation        = output,
         .starts_in_reasoning = prompt.starts_in_reasoning,
         .has_content_prefix  = !prompt.continuation_content.empty(),
         .thinking            = thinking},
        std::move(decoder), std::move(matcher));
}

} // namespace ninfer::models::qwen::frontend
