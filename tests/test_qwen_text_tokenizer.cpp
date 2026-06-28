#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path repo_file(std::string_view relative) {
#ifdef QUS_SOURCE_DIR
    return std::filesystem::path(QUS_SOURCE_DIR) / relative;
#else
    return std::filesystem::current_path() / relative;
#endif
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

void print_ids(std::string_view label, const std::vector<int>& ids) {
    std::cerr << label << ": [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { std::cerr << ", "; }
        std::cerr << ids[i];
    }
    std::cerr << "]\n";
}

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

bool throws_invalid_with_file_containing(const std::filesystem::path& dir,
                                         const std::filesystem::path& file,
                                         std::string_view expected) {
    try {
        (void)qus::text::QwenTokenizer(dir);
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        return message.find(expected) != std::string::npos &&
               message.find(file.string()) != std::string::npos;
    }
    return false;
}

bool decode_throws_out_of_range_containing(const qus::text::QwenTokenizer& tokenizer,
                                           const std::vector<int>& ids, std::string_view expected) {
    try {
        (void)tokenizer.decode(ids);
    } catch (const std::out_of_range& ex) {
        return std::string_view(ex.what()).find(expected) != std::string_view::npos;
    }
    return false;
}

bool decode_throws_invalid_containing(const qus::text::QwenTokenizer& tokenizer,
                                      const std::vector<int>& ids, std::string_view expected) {
    try {
        (void)tokenizer.decode(ids);
    } catch (const std::invalid_argument& ex) {
        return std::string_view(ex.what()).find(expected) != std::string_view::npos;
    }
    return false;
}

bool encode_throws_logic_containing(const qus::text::QwenTokenizer& tokenizer,
                                    std::string_view text, std::string_view expected) {
    try {
        (void)tokenizer.encode(text);
    } catch (const std::logic_error& ex) {
        return std::string_view(ex.what()).find(expected) != std::string_view::npos;
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

int test_minimal_added_token_encode_decode() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"!":0,"A":1})",
            R"([{"id":2,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":4,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},{"id":5,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}])"));
    const qus::text::QwenTokenizer tokenizer(dir.path);

    const std::vector<int> ids = tokenizer.encode("<|im_start|><|im_end|><think></think>");
    int failures               = 0;
    failures += check(ids == std::vector<int>{2, 3, 4, 5}, "minimal added token ids mismatch");

    const std::string raw = tokenizer.decode(ids, qus::text::DecodeOptions{false, {}});
    failures +=
        check(raw == "<|im_start|><|im_end|><think></think>", "minimal raw decode mismatch");

    const std::string clean = tokenizer.decode(ids, qus::text::DecodeOptions{true, {}});
    failures += check(clean == "<think></think>", "minimal clean decode mismatch");

    const std::vector<int> with_stop{4, 5, 3};
    const std::string stop_trimmed =
        tokenizer.decode(with_stop, qus::text::DecodeOptions{false, {3}});
    failures += check(stop_trimmed == "<think></think>", "minimal terminal stop decode mismatch");

    const std::vector<int> with_interior_stop{3, 4, 3};
    const std::string interior_stop_preserved =
        tokenizer.decode(with_interior_stop, qus::text::DecodeOptions{false, {3}});
    failures += check(interior_stop_preserved == "<|im_end|><think>",
                      "minimal interior stop decode mismatch");
    return failures;
}

int test_byte_level_vocab_decode() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"Ġ":0,"Ċ":1,"Ã©":2,"€":3})",
            R"([{"id":4,"content":"<extra>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
    const qus::text::QwenTokenizer tokenizer(dir.path);

    int failures = 0;
    failures += check(tokenizer.decode(std::vector<int>{0}) == " ", "space byte decode mismatch");
    failures +=
        check(tokenizer.decode(std::vector<int>{1}) == "\n", "newline byte decode mismatch");
    failures += check(tokenizer.decode(std::vector<int>{2}) == "é", "utf-8 byte decode mismatch");
    failures += check(decode_throws_invalid_containing(tokenizer, {3}, "3"),
                      "invalid byte alphabet token accepted");
    return failures;
}

int test_decode_rejects_reconstructed_invalid_utf8() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"Ã":0})",
            R"([{"id":1,"content":"<extra>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
    const qus::text::QwenTokenizer tokenizer(dir.path);

    return check(decode_throws_invalid_containing(tokenizer, {0}, "valid UTF-8"),
                 "decode returned reconstructed invalid UTF-8");
}

int test_rejects_nul_in_merges_txt_symbols() {
    TempDir dir;
    write_tokenizer_json(dir.path, minimal_tokenizer_json(R"({"a":0,"b":1})", "[]"));
    std::string merges = "#version: 0.2\n";
    merges += std::string("a\0 b\n", 5);
    write_file(dir.path / "merges.txt", merges);

    return check(throws_invalid_with_file_containing(dir.path, dir.path / "merges.txt", "NUL"),
                 "merges.txt symbol containing NUL accepted");
}

int test_local_bpe_split_merge_cases() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"c":0,"a":1,"n":2,"'":3,"t":4,"ca":5,"can":6,"'t":7,"Ġ":8,"Ċ":9,"ĠĠ":10,"ĊĊ":11,"ĠĠĊĊ":12,"ä":13,"½":14,"ł":15,"ä½":16,"ä½ł":17,"å":18,"¥":19,"å¥":20,"å¥½":21,"!":22,"ð":23,"Ł":24,"ĺ":25,"Ģ":26,"ðŁ":27,"ðŁĺ":28,"ðŁĺĢ":29,"ĠðŁĺĢ":30,"C":31,"f":32,"Ã":33,"©":34,"Ca":35,"Caf":36,"Ã©":37,"ĠCaf":38,"ĠCafÃ©":39})",
            "[]"));
    write_file(dir.path / "merges.txt", "#version: 0.2\n"
                                        "c a\n"
                                        "ca n\n"
                                        "' t\n"
                                        "Ġ Ġ\n"
                                        "Ċ Ċ\n"
                                        "ĠĠ ĊĊ\n"
                                        "ä ½\n"
                                        "ä½ ł\n"
                                        "å ¥\n"
                                        "å¥ ½\n"
                                        "ð Ł\n"
                                        "ðŁ ĺ\n"
                                        "ðŁĺ Ģ\n"
                                        "Ġ ðŁĺĢ\n"
                                        "C a\n"
                                        "Ca f\n"
                                        "Ã ©\n"
                                        "Ġ Caf\n"
                                        "ĠCaf Ã©\n");

    const qus::text::QwenTokenizer tokenizer(dir.path);

    int failures = 0;
    failures += check(tokenizer.encode("can't") == std::vector<int>{6, 7},
                      "local contraction BPE ids mismatch");
    failures += check(tokenizer.decode(std::vector<int>{6, 7}) == "can't",
                      "local contraction BPE decode mismatch");
    failures += check(tokenizer.encode("  \n\n") == std::vector<int>{12},
                      "local whitespace/newline BPE ids mismatch");
    failures += check(tokenizer.decode(std::vector<int>{12}) == "  \n\n",
                      "local whitespace/newline BPE decode mismatch");
    failures += check(tokenizer.encode("你好") == std::vector<int>{17, 21},
                      "local non-Latin BPE ids mismatch");
    failures += check(tokenizer.decode(std::vector<int>{17, 21}) == "你好",
                      "local non-Latin BPE decode mismatch");
    failures += check(tokenizer.encode("! 😀") == std::vector<int>{22, 30},
                      "local punctuation/emoji BPE ids mismatch");
    failures += check(tokenizer.decode(std::vector<int>{22, 30}) == "! 😀",
                      "local punctuation/emoji BPE decode mismatch");
    failures +=
        check(tokenizer.encode(" Café") == std::vector<int>{39}, "local NFC mark BPE ids mismatch");
    failures += check(tokenizer.decode(std::vector<int>{39}) == " Café",
                      "local NFC mark BPE decode mismatch");
    return failures;
}

int test_added_token_same_position_first_loaded_wins() {
    int failures = 0;
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"!":0})",
                R"([{"id":1,"content":"<tag>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},{"id":2,"content":"<tag>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}])"));
        const qus::text::QwenTokenizer tokenizer(dir.path);

        failures += check(tokenizer.encode("<tag>") == std::vector<int>{1},
                          "same-position added token tie did not choose first loaded token");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"x":0})",
                R"([{"id":1,"content":"<tag>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},{"id":2,"content":"<tag>x","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}])"));
        write_file(dir.path / "merges.txt", "#version: 0.2\n");
        const qus::text::QwenTokenizer tokenizer(dir.path);

        failures +=
            check(tokenizer.encode("<tag>x") == std::vector<int>{1, 0},
                  "same-position overlapping added token did not choose first loaded token");
    }
    return failures;
}

int test_encode_rejects_unsupported_added_token_flags() {
    int failures = 0;
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"!":0})",
                R"([{"id":1,"content":"<x>","single_word":true,"lstrip":false,"rstrip":false,"normalized":false,"special":false}])"));
        const qus::text::QwenTokenizer tokenizer(dir.path);
        failures += check(encode_throws_logic_containing(tokenizer, "<x>", "single_word=false"),
                          "single_word=true added token encode accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"!":0})",
                R"([{"id":1,"content":"<x>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":false}])"));
        const qus::text::QwenTokenizer tokenizer(dir.path);
        failures += check(encode_throws_logic_containing(tokenizer, "<x>", "lstrip=false"),
                          "lstrip=true added token encode accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"!":0})",
                R"([{"id":1,"content":"<x>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":false}])"));
        const qus::text::QwenTokenizer tokenizer(dir.path);
        failures += check(encode_throws_logic_containing(tokenizer, "<x>", "rstrip=false"),
                          "rstrip=true added token encode accepted");
    }
    {
        TempDir dir;
        write_tokenizer_json(
            dir.path,
            minimal_tokenizer_json(
                R"({"!":0})",
                R"([{"id":1,"content":"<x>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}])"));
        const qus::text::QwenTokenizer tokenizer(dir.path);
        failures += check(encode_throws_logic_containing(tokenizer, "<x>", "normalized=false"),
                          "normalized=true added token encode accepted");
    }
    return failures;
}

int test_decode_rejects_sparse_invalid_id() {
    TempDir dir;
    write_tokenizer_json(
        dir.path,
        minimal_tokenizer_json(
            R"({"":0})",
            R"([{"id":5,"content":"<extra>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])"));
    const qus::text::QwenTokenizer tokenizer(dir.path);

    int failures = 0;
    failures += check(tokenizer.decode(std::vector<int>{0}).empty(),
                      "valid empty vocab token decode mismatch");
    failures += check(decode_throws_out_of_range_containing(tokenizer, {3}, "3"),
                      "sparse invalid token id accepted");
    return failures;
}

int test_added_token_encode_decode() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping added token test: local tokenizer not present\n";
        return 0;
    }
    const qus::text::QwenTokenizer tok(tokenizer_path);
    const std::vector<int> ids = tok.encode("<|im_start|><|im_end|><think></think>");
    if (ids != std::vector<int>{248045, 248046, 248068, 248069}) {
        return fail("added token ids mismatch");
    }
    const std::string raw = tok.decode(ids, qus::text::DecodeOptions{false, {}});
    if (raw != "<|im_start|><|im_end|><think></think>") { return fail("raw decode mismatch"); }
    const std::string clean = tok.decode(ids, qus::text::DecodeOptions{true, {}});
    if (clean != "<think></think>") { return fail("clean decode mismatch"); }
    return 0;
}

int test_hf_golden_text_cases() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping golden text cases: local tokenizer not present\n";
        return 0;
    }
    const auto fixture_path = repo_file("tests/fixtures/text/qwen36_text_golden.json");
    std::ifstream in(fixture_path);
    const auto fixture = nlohmann::json::parse(in);
    const qus::text::QwenTokenizer tok(tokenizer_path);
    for (const auto& item : fixture.at("text_cases")) {
        const std::string name          = item.at("name").get<std::string>();
        const std::string text          = item.at("text").get<std::string>();
        const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
        const std::vector<int> actual   = tok.encode(text);
        if (actual != expected) {
            std::cerr << "golden encode mismatch for " << name << '\n';
            print_ids("expected", expected);
            print_ids("actual", actual);
            return 1;
        }
        const std::string raw = tok.decode(expected, qus::text::DecodeOptions{false, {}});
        if (raw != item.at("raw_decoded").get<std::string>()) {
            std::cerr << "golden raw decode mismatch for " << name << '\n';
            return 1;
        }
    }
    return 0;
}

int test_hf_golden_message_cases() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping golden message cases: local tokenizer not present\n";
        return 0;
    }
    const auto fixture_path = repo_file("tests/fixtures/text/qwen36_text_golden.json");
    std::ifstream in(fixture_path);
    const auto fixture = nlohmann::json::parse(in);
    const qus::text::QwenTokenizer tok(tokenizer_path);
    for (const auto& item : fixture.at("message_cases")) {
        std::vector<qus::text::ChatMessage> messages;
        for (const auto& msg : item.at("messages")) {
            messages.push_back(
                {msg.at("role").get<std::string>(), msg.at("content").get<std::string>()});
        }
        const std::string rendered = qus::text::render_qwen_chat(messages);
        if (rendered != item.at("rendered").get<std::string>()) {
            std::cerr << "rendered prompt mismatch for " << item.at("name").get<std::string>()
                      << '\n';
            return 1;
        }
        const std::vector<int> actual   = tok.encode(rendered);
        const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
        if (actual != expected) {
            std::cerr << "message ids mismatch for " << item.at("name").get<std::string>() << '\n';
            return 1;
        }
    }
    return 0;
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
        failures += test_minimal_added_token_encode_decode();
        failures += test_byte_level_vocab_decode();
        failures += test_decode_rejects_reconstructed_invalid_utf8();
        failures += test_rejects_nul_in_merges_txt_symbols();
        failures += test_local_bpe_split_merge_cases();
        failures += test_added_token_same_position_first_loaded_wins();
        failures += test_encode_rejects_unsupported_added_token_flags();
        failures += test_decode_rejects_sparse_invalid_id();
        failures += test_added_token_encode_decode();
        failures += test_hf_golden_text_cases();
        failures += test_hf_golden_message_cases();
        return failures == 0 ? 0 : fail("qwen tokenizer metadata test failed");
    } catch (const std::exception& ex) {
        std::cerr << "qwen tokenizer metadata test failed: " << ex.what() << '\n';
        return 1;
    }
}
