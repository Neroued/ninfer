#include "qus/text/tokenizer.h"

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
                   ("qus_tokenizer_" + std::to_string(stamp) + "_" + std::to_string(i));
            if (std::filesystem::create_directory(path)) { return; }
        }
        throw std::runtime_error("failed to create tokenizer temp directory");
    }

    ~TempDir() { std::filesystem::remove_all(path); }
};

void write_file(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to write " + path.string()); }
    out << text;
}

void write_tokenizer_json(const std::filesystem::path& dir, std::string_view text) {
    write_file(dir / "tokenizer.json", text);
}

void write_generation_config_json(const std::filesystem::path& dir, std::string_view text) {
    write_file(dir / "generation_config.json", text);
}

std::string minimal_tokenizer_json(
    std::string_view vocab,
    std::string_view added_tokens =
        R"([{"id":2,"content":"<extra>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}])") {
    std::string json =
        R"({"model":{"type":"BPE","vocab":)" + std::string(vocab) + R"(},"added_tokens":)";
    json += added_tokens;
    json += "}";
    return json;
}

bool throws_invalid_containing(const std::filesystem::path& dir, std::string_view expected) {
    try {
        (void)qus::text::QwenTokenizer(dir);
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        return message.find(expected) != std::string::npos &&
               message.find((dir / "tokenizer.json").string()) != std::string::npos;
    }
    return false;
}

bool throws_generation_invalid_containing(const std::filesystem::path& dir,
                                          std::string_view expected) {
    try {
        (void)qus::text::QwenTokenizer(dir);
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        return message.find(expected) != std::string::npos &&
               message.find((dir / "generation_config.json").string()) != std::string::npos;
    }
    return false;
}

int test_valid_minimal_metadata() {
    TempDir dir;
    write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0,"b":1})"));
    write_generation_config_json(dir.path, R"({"eos_token_id":[4,5]})");

    const qus::text::QwenTokenizer tokenizer(dir.path);

    int failures = 0;
    failures += check(tokenizer.default_stop_token_ids() == std::vector<int>{4, 5},
                      "minimal default stop ids mismatch");
    failures += check(tokenizer.added_tokens().size() == 2, "minimal added token count mismatch");
    failures += check(tokenizer.added_tokens().at(0).id == 2, "minimal added token id mismatch");
    failures += check(tokenizer.added_tokens().at(0).content == "<extra>",
                      "minimal added token content mismatch");
    failures +=
        check(tokenizer.added_tokens().at(0).special, "minimal added token special flag mismatch");
    failures += check(tokenizer.is_special_token(2), "minimal special token lookup mismatch");
    failures += check(!tokenizer.is_special_token(3), "minimal non-special token lookup mismatch");
    return failures;
}

int test_rejects_empty_vocab() {
    TempDir dir;
    write_tokenizer_json(dir.path, minimal_tokenizer_json("{}"));
    return check(throws_invalid_containing(dir.path, "model.vocab"), "empty model.vocab accepted");
}

int test_rejects_duplicate_vocab_id() {
    TempDir dir;
    write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0,"b":0})"));
    return check(throws_invalid_containing(dir.path, "model.vocab"),
                 "duplicate model.vocab id accepted");
}

int test_rejects_added_token_duplicate_or_overlap_id() {
    int failures = 0;
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"a":0,"b":1})",
                R"([{"id":1,"content":"<overlap>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
        failures += check(throws_invalid_containing(dir.path, "added_tokens"),
                          "added token overlapping model.vocab id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"a":0,"":1})",
                R"([{"id":1,"content":"<empty-overlap>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
        failures += check(throws_invalid_containing(dir.path, "added_tokens"),
                          "added token overlapping empty model.vocab token id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"a":0,"b":1})",
                R"([{"id":2,"content":"<x>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<y>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
        failures += check(throws_invalid_containing(dir.path, "added_tokens"),
                          "duplicate added token id accepted");
    }
    return failures;
}

int test_rejects_negative_or_out_of_range_token_ids() {
    int failures = 0;
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":-1})"));
        failures += check(throws_invalid_containing(dir.path, "model.vocab"),
                          "negative model.vocab id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":2147483648})"));
        failures += check(throws_invalid_containing(dir.path, "model.vocab"),
                          "out-of-range model.vocab id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"a":0})",
                R"([{"id":2147483648,"content":"<x>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
        failures += check(throws_invalid_containing(dir.path, "added_tokens"),
                          "out-of-range added token id accepted");
    }
    return failures;
}

int test_rejects_malformed_added_token_fields() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"a":0})",
            R"([{"id":1,"content":"<x>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":"yes"}])"));
    return check(throws_invalid_containing(dir.path, "special"),
                 "malformed added token field accepted");
}

int test_rejects_invalid_generation_config_eos_token_id() {
    int failures = 0;
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0})"));
        write_generation_config_json(dir.path, R"({})");
        failures += check(throws_generation_invalid_containing(dir.path, "eos_token_id"),
                          "missing eos_token_id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0})"));
        write_generation_config_json(dir.path, R"({"eos_token_id":[]})");
        failures += check(throws_generation_invalid_containing(dir.path, "eos_token_id"),
                          "empty eos_token_id array accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0})"));
        write_generation_config_json(dir.path, R"({"eos_token_id":-1})");
        failures += check(throws_generation_invalid_containing(dir.path, "eos_token_id"),
                          "negative eos_token_id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0})"));
        write_generation_config_json(dir.path, R"({"eos_token_id":[2147483648]})");
        failures += check(throws_generation_invalid_containing(dir.path, "eos_token_id"),
                          "out-of-range eos_token_id accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0})"));
        write_generation_config_json(dir.path, R"({"eos_token_id":"4"})");
        failures += check(throws_generation_invalid_containing(dir.path, "eos_token_id"),
                          "string eos_token_id accepted");
    }
    return failures;
}

int test_load_real_tokenizer_metadata() {
    const std::filesystem::path tokenizer_dir =
        "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16";
    if (!std::filesystem::exists(tokenizer_dir / "tokenizer.json")) {
        std::cout << "SKIP qwen tokenizer metadata smoke test: missing "
                  << (tokenizer_dir / "tokenizer.json") << '\n';
        return 0;
    }

    const qus::text::QwenTokenizer tokenizer(tokenizer_dir);

    int failures = 0;
    failures += check(tokenizer.default_stop_token_ids() == std::vector<int>{248046, 248044},
                      "real default stop token ids mismatch");

    bool found_im_start = false;
    bool found_think    = false;
    for (const qus::text::AddedToken& token : tokenizer.added_tokens()) {
        if (token.id == 248045) {
            found_im_start = token.content == "<|im_start|>" && token.special;
        }
        if (token.id == 248068) { found_think = token.content == "<think>" && !token.special; }
    }
    failures += check(found_im_start, "missing real <|im_start|> added token metadata");
    failures += check(found_think, "missing real <think> added token metadata");
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        failures += test_valid_minimal_metadata();
        failures += test_rejects_empty_vocab();
        failures += test_rejects_duplicate_vocab_id();
        failures += test_rejects_added_token_duplicate_or_overlap_id();
        failures += test_rejects_negative_or_out_of_range_token_ids();
        failures += test_rejects_malformed_added_token_fields();
        failures += test_rejects_invalid_generation_config_eos_token_id();
        failures += test_load_real_tokenizer_metadata();
        return failures == 0 ? 0 : fail("qwen tokenizer metadata test failed");
    } catch (const std::exception& ex) {
        std::cerr << "qwen tokenizer metadata test failed: " << ex.what() << '\n';
        return 1;
    }
}
