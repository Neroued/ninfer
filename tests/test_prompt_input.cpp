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
        : path_(std::filesystem::path(NINFER_SOURCE_DIR) / ".agent-tmp" /
                ("ninfer-prompt-input-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) /
                "messages.json") {
        std::filesystem::create_directories(path_.parent_path());
    }

    ~TemporaryInput() {
        std::error_code ignored;
        std::filesystem::remove_all(path_.parent_path(), ignored);
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
               << arguments
               << "}}]},"
                  "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"failed\","
                  "\"tool_result_is_error\":true}]}";
    }

    const ninfer::PromptInput prompt =
        ninfer::product::prompt_from_messages(input_file.path(), false, false);
    if (prompt.options.tool_jsons != std::vector<std::string>{tool} ||
        prompt.messages.size() != 3 || prompt.messages[1].tool_calls.size() != 1 ||
        prompt.messages[1].tool_calls.front().arguments_json != arguments ||
        !prompt.messages[2].tool_result_is_error || prompt.messages[2].parts[0].text != "failed") {
        std::cerr << "local messages JSON changed prompt-bearing object member order\n";
        return 1;
    }
    {
        std::ofstream asset(input_file.path().parent_path() / "asset.bin", std::ios::binary);
        asset << "test";
        std::ofstream messages(input_file.path());
        messages << R"([{"role":"user","content":[{"type":"image","image":"asset.bin"}]}])";
    }
    const auto media = ninfer::product::prompt_from_messages(input_file.path(), false, true);
    if (media.messages.front().parts.front().media.bytes !=
        std::vector<std::uint8_t>{'t', 'e', 's', 't'}) {
        std::cerr << "media paths were not resolved relative to the messages file\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
