#include "product/prompt_input/prompt_input.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {

class TemporaryInput final {
public:
    TemporaryInput()
        : path_(std::filesystem::temp_directory_path() /
                ("ninfer-prompt-input-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".json")) {}

    ~TemporaryInput() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TemporaryInput(const TemporaryInput&)            = delete;
    TemporaryInput& operator=(const TemporaryInput&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace

int main() {
    TemporaryInput input_file;
    const std::string tool =
        R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}})";
    const std::string arguments = R"({"zeta":"last","alpha":1})";
    {
        std::ofstream output(input_file.path());
        if (!output) {
            std::cerr << "failed to create prompt-input fixture\n";
            return 1;
        }
        output << "{\"tools\":[" << tool
               << "],\"messages\":[{\"role\":\"user\",\"content\":\"probe\"},"
                  "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\","
                  "\"type\":\"function\",\"function\":{\"name\":\"probe\",\"arguments\":"
               << arguments << "}}]}]}";
    }

    const ninfer::PromptInput prompt =
        ninfer::product::prompt_from_messages(input_file.path(), false, false);
    if (prompt.options.tool_jsons != std::vector<std::string>{tool} ||
        prompt.messages.size() != 2 || prompt.messages[1].tool_calls.size() != 1 ||
        prompt.messages[1].tool_calls.front().arguments_json != arguments) {
        std::cerr << "local messages JSON changed prompt-bearing object member order\n";
        return 1;
    }
    auto constrained = prompt;
    constrained.context_cache.markers.push_back(
        {.after_message_count      = 1,
         .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
         .after_message_part_count = 1});
    ninfer::product::apply_structured_output_instruction(
        constrained, {ninfer::StructuredOutputKind::JsonObject, {}});
    if (constrained.messages.size() != 3 ||
        constrained.messages.front().role != ninfer::ChatRole::System ||
        constrained.messages[1].parts[0].text != prompt.messages[0].parts[0].text ||
        constrained.context_cache.markers[0].after_message_count != 2 ||
        constrained.context_cache.markers[0].after_message_part_count != 1) {
        std::cerr << "structured instruction changed caller content or cache boundary\n";
        return 1;
    }
    auto leading                  = ninfer::product::prompt_from_text("answer", false);
    leading.messages.front().role = ninfer::ChatRole::System;
    leading.context_cache.markers.push_back(
        {.location                  = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
         .leading_instruction_bytes = 3});
    ninfer::product::apply_structured_output_instruction(
        leading, {ninfer::StructuredOutputKind::JsonObject, {}});
    if (leading.messages.size() != 1 || leading.messages[0].parts[0].text != "answer" ||
        leading.context_cache.markers[0].leading_instruction_bytes != 3) {
        std::cerr << "structured instruction moved caller instruction boundary\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
