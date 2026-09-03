#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
    // Issue #142: additionally publish a shared-prefix candidate at the
    // leading system/developer frontier so agent sibling sessions share a
    // long head even without a client marker. The last-content implicit
    // candidate stays untouched.
    bool auto_system_shared_prefix = true;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const nlohmann::json& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const nlohmann::json& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
