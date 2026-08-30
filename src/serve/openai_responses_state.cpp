#include "serve/openai_responses.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

[[noreturn]] void invalid_tool_history(std::string message) {
    bad_request(std::move(message), "input", "invalid_tool_history");
}

[[noreturn]] void invalid_input(std::string message) {
    bad_request(std::move(message), "input", "invalid_value");
}

[[noreturn]] void response_not_found(const std::string& id) {
    ApiError error{.status  = 404,
                   .type    = "invalid_request_error",
                   .message = "response '" + id + "' not found",
                   .param   = "previous_response_id",
                   .code    = "response_not_found"};
    throw ApiException(std::move(error));
}

bool has_user_query(const std::vector<ChatTurn>& turns) {
    return std::any_of(turns.begin(), turns.end(), [](const ChatTurn& turn) {
        if (turn.role != ChatRole::User) { return false; }
        return std::any_of(turn.content.begin(), turn.content.end(), [](const ContentPart& part) {
            return part.kind != ContentKind::Text || !part.text.empty();
        });
    });
}

std::vector<ChatTurn> normalize_call_graph(std::vector<ChatTurn> turns) {
    std::unordered_map<std::string, std::size_t> call_owner;
    for (std::size_t turn_index = 0; turn_index < turns.size(); ++turn_index) {
        const ChatTurn& turn = turns[turn_index];
        if (!turn.tool_calls.empty() && turn.role != ChatRole::Assistant) {
            invalid_tool_history("function calls must belong to an assistant Item");
        }
        for (const ToolCall& call : turn.tool_calls) {
            if (call.id.empty()) { invalid_tool_history("function call_id must not be empty"); }
            if (!call_owner.emplace(call.id, turn_index).second) {
                invalid_tool_history("duplicate function call_id '" + call.id + "'");
            }
        }
        if (turn.role == ChatRole::Tool && turn.tool_call_id.empty()) {
            invalid_tool_history("function_call_output call_id must not be empty");
        }
    }

    std::vector<ChatTurn> normalized;
    normalized.reserve(turns.size());
    for (std::size_t index = 0; index < turns.size();) {
        ChatTurn& turn = turns[index];
        if (turn.role == ChatRole::Tool) {
            invalid_tool_history("function_call_output call_id '" + turn.tool_call_id +
                                 "' does not immediately follow its assistant function_call");
        }
        if (turn.tool_calls.empty()) {
            normalized.push_back(std::move(turn));
            ++index;
            continue;
        }

        std::unordered_map<std::string, std::size_t> positions;
        positions.reserve(turn.tool_calls.size());
        for (std::size_t call_index = 0; call_index < turn.tool_calls.size(); ++call_index) {
            positions.emplace(turn.tool_calls[call_index].id, call_index);
        }
        normalized.push_back(std::move(turn));

        std::vector<std::optional<ChatTurn>> results(positions.size());
        std::size_t result_count = 0;
        std::size_t cursor       = index + 1;
        while (cursor < turns.size() && turns[cursor].role == ChatRole::Tool) {
            ChatTurn& result    = turns[cursor];
            const auto position = positions.find(result.tool_call_id);
            if (position == positions.end()) {
                const auto owner = call_owner.find(result.tool_call_id);
                if (owner == call_owner.end()) {
                    invalid_tool_history("function_call_output references unknown call_id '" +
                                         result.tool_call_id + "'");
                }
                invalid_tool_history("function_call_output call_id '" + result.tool_call_id +
                                     "' does not immediately follow its assistant call group");
            }
            const ToolCall& call = normalized.back().tool_calls[position->second];
            if (result.tool_result_name && *result.tool_result_name != call.name) {
                invalid_tool_history("function_call_output identity does not match call_id '" +
                                     result.tool_call_id + "'");
            }
            if (results[position->second]) {
                invalid_tool_history("duplicate function_call_output for call_id '" +
                                     result.tool_call_id + "'");
            }
            results[position->second] = std::move(result);
            ++result_count;
            ++cursor;
        }

        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            if (!results[result_index]) {
                invalid_tool_history(
                    "partial function results must cover a prefix of the assistant call group");
            }
        }
        for (std::size_t result_index = result_count; result_index < results.size();
             ++result_index) {
            if (results[result_index]) {
                invalid_tool_history(
                    "partial function results must cover a prefix of the assistant call group");
            }
        }
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            normalized.push_back(std::move(*results[result_index]));
        }
        index = cursor;
    }
    return normalized;
}

ChatTurn instruction_turn(const std::string& text) {
    ChatTurn turn;
    turn.role = ChatRole::Developer;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = text;
    turn.content.push_back(std::move(part));
    return turn;
}

} // namespace

OpenAIResponsesResolvedPrompt
resolve_openai_responses_prompt(const OpenAIResponsesPromptRequest& request,
                                OpenAIResponsesStore& store, std::optional<std::string> response_id,
                                bool store_response) {
    OpenAIResponsesResolvedPrompt resolved;
    resolved.generation = request.generation;

    std::shared_ptr<const StoredOpenAIResponse> parent_record;
    if (request.previous_response_id) {
        parent_record = store.get(*request.previous_response_id);
        if (!parent_record) { response_not_found(*request.previous_response_id); }
        resolved.parent = parent_record->context;
    }

    const bool parent_preserve = parent_record ? parent_record->preserve_thinking : false;
    if (resolved.generation.preserve_thinking) {
        resolved.preserve_thinking_semantic_change =
            parent_record && *resolved.generation.preserve_thinking != parent_preserve;
    } else if (parent_record) {
        resolved.generation.preserve_thinking = parent_preserve;
    }

    std::vector<ChatTurn> context = flatten_openai_response_context(resolved.parent);
    context.insert(context.end(), request.input_turns.begin(), request.input_turns.end());
    if (!has_user_query(context)) {
        invalid_input("input must provide a user query when no previous response contains one");
    }
    context = normalize_call_graph(std::move(context));

    resolved.generation.messages.reserve(context.size() + (request.instructions ? 1U : 0U));
    if (request.instructions) {
        resolved.generation.messages.push_back(instruction_turn(*request.instructions));
    }
    resolved.generation.messages.insert(resolved.generation.messages.end(),
                                        std::make_move_iterator(context.begin()),
                                        std::make_move_iterator(context.end()));

    if (response_id) {
        if (parent_record) {
            resolved.session_key = parent_record->session_key;
        } else if (store_response) {
            resolved.session_key = *response_id;
        }
        resolved.cache_hints.session_key = resolved.session_key;
        resolved.cache_hints.retention =
            store_response ? CacheRetentionHint::LiveSession : CacheRetentionHint::Disposable;
        resolved.cache_hints.update_session_index = store_response;
    }
    return resolved;
}

} // namespace ninfer::serve
