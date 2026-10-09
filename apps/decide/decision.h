#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::decide {

using Json = nlohmann::ordered_json;

inline constexpr std::uint32_t kMaxOptions = 255;

class InvalidDecision final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// A validated decision row: the option keys, the option descriptions shown to the model, and the
// prompt text (without the chat template) built from them.
struct Decision {
    Json question;
    std::vector<std::string> keys;
    std::vector<Json> descriptions;
    std::string prompt;
};

// Mirrors Python json.dumps(value, ensure_ascii=False): ", " and ": " separators, insertion order.
[[nodiscard]] std::string python_dumps(const Json& value);

[[nodiscard]] Decision parse_decision(const Json& row, const std::vector<std::string>& codes);

// The full chat-templated prompt (thinking disabled, generation prompt appended).
[[nodiscard]] std::string render_chat(const std::string& user_prompt);

// Probabilities are the engine's truncated softmax; the answer renormalizes them.
[[nodiscard]] Json answer(const Decision& decision, const std::vector<float>& probabilities);

} // namespace ninfer::decide
