#include "qus/text/text_runner.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 16; ++i) {
            path = std::filesystem::temp_directory_path() /
                   ("qus_text_runner_" + std::to_string(stamp) + "_" + std::to_string(i));
            if (std::filesystem::create_directory(path)) { return; }
        }
        throw std::runtime_error("failed to create text runner temp directory");
    }

    ~TempDir() { std::filesystem::remove_all(path); }
};

void write_file(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to write " + path.string()); }
    out << text;
}

std::string minimal_tokenizer_json() {
    return R"({"model":{"type":"BPE","vocab":{"a":0,"b":1}},"added_tokens":[]})";
}

qus::text::QwenTokenizer make_tokenizer_with_stop_ids(const std::vector<int>& ids) {
    TempDir dir;
    write_file(dir.path / "tokenizer.json", minimal_tokenizer_json());

    std::string config = R"({"eos_token_id":[)";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { config += ","; }
        config += std::to_string(ids[i]);
    }
    config += "]})";
    write_file(dir.path / "generation_config.json", config);

    return qus::text::QwenTokenizer(dir.path);
}

bool throws_invalid_argument_for_negative_override(const qus::text::QwenTokenizer& tokenizer) {
    try {
        (void)qus::text::resolve_stop_token_ids(tokenizer, {4, -1});
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

int test_resolve_stop_token_ids() {
    const qus::text::QwenTokenizer tokenizer = make_tokenizer_with_stop_ids({248046, 248044});

    int failures = 0;
    failures += check(qus::text::resolve_stop_token_ids(tokenizer, {}) ==
                          std::vector<int>{248046, 248044},
                      "empty stop id overrides did not use tokenizer defaults");
    failures += check(qus::text::resolve_stop_token_ids(tokenizer, {9, 9, 8}) ==
                          std::vector<int>{9, 8},
                      "stop id overrides were not deduplicated in first-occurrence order");
    failures += check(throws_invalid_argument_for_negative_override(tokenizer),
                      "negative stop id override did not throw invalid_argument");
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = test_resolve_stop_token_ids();
        return failures == 0 ? 0 : fail("qwen text runner test failed");
    } catch (const std::exception& ex) {
        std::cerr << "qwen text runner test failed: " << ex.what() << '\n';
        return 1;
    }
}
