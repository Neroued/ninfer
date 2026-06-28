#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qus::text {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::filesystem::path& tokenizer_dir);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;

    [[nodiscard]] const std::vector<AddedToken>& added_tokens() const noexcept {
        return added_tokens_;
    }

private:
    std::filesystem::path tokenizer_dir_;
    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::vector<AddedToken> added_tokens_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace qus::text
