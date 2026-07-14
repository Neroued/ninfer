#include "ninfer/text/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer::text {
namespace {

using Json    = nlohmann::json;
namespace uni = unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        std::string(label));
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    std::string(label));
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + std::string(label));
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", label);
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

std::vector<AddedToken> load_added_tokens(const Json& root, std::string_view label,
                                          std::vector<std::string>& id_to_token,
                                          const std::unordered_set<int>& occupied_vocab_ids) {
    const Json& added = require_array_field(root, "added_tokens", label);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, label);
        if (token.single_word || token.lstrip || token.rstrip || token.normalized) {
            throw std::invalid_argument(
                "QwenTokenizer only supports added tokens with single_word=false, "
                "lstrip=false, rstrip=false, and normalized=false in " +
                std::string(label));
        }
        const auto index = static_cast<std::size_t>(token.id);
        if (occupied_vocab_ids.contains(token.id)) {
            throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                        std::string(label));
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " +
                                        std::string(label));
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument(
                "field eos_token_id must not be empty in generation_config.json");
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or array in generation_config.json");
}

std::string merge_pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + 1 + right.size());
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

std::unordered_map<std::string, int> load_bpe_merge_ranks(std::string_view contents) {
    std::istringstream in{std::string(contents)};

    std::unordered_map<std::string, int> ranks;
    std::string line;
    int rank        = 0;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.empty()) { continue; }
        if (!saw_header && line.starts_with("#version")) {
            saw_header = true;
            continue;
        }
        saw_header = true;

        const std::size_t first_space = line.find(' ');
        if (first_space == std::string::npos || first_space == 0 ||
            first_space + 1 >= line.size() ||
            line.find(' ', first_space + 1) != std::string::npos) {
            throw std::invalid_argument("malformed merges.txt line: " + line);
        }
        const std::string left  = line.substr(0, first_space);
        const std::string right = line.substr(first_space + 1);
        if (left.find('\0') != std::string::npos || right.find('\0') != std::string::npos) {
            throw std::invalid_argument("field merges.txt must not contain NUL in merge symbols: " +
                                        line);
        }
        const auto inserted = ranks.emplace(merge_pair_key(left, right), rank);
        if (!inserted.second) {
            throw std::invalid_argument("duplicate merge pair in merges.txt: " + line);
        }
        ++rank;
    }
    return ranks;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(codepoint, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

std::unordered_map<unsigned char, std::string> build_byte_level_encoder() {
    std::unordered_map<unsigned char, std::string> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder.emplace(static_cast<unsigned char>(byte),
                        uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoint)));
    }
    return encoder;
}

bool is_newline(std::int32_t codepoint) noexcept { return codepoint == '\r' || codepoint == '\n'; }

bool is_letter_or_mark(std::int32_t codepoint) noexcept {
    return uni::is_letter(codepoint) || uni::is_mark(codepoint);
}

bool is_non_newline_non_letter_non_number(std::int32_t codepoint) noexcept {
    return !is_newline(codepoint) && !uni::is_letter(codepoint) && !uni::is_number(codepoint);
}

bool is_non_space_non_letter_mark_number(std::int32_t codepoint) noexcept {
    return !uni::is_whitespace(codepoint) && !uni::is_letter(codepoint) &&
           !uni::is_mark(codepoint) && !uni::is_number(codepoint);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
        const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
    }
    return true;
}

std::size_t span_end_offset(std::string_view text, const std::vector<uni::CodepointSpan>& spans,
                            std::size_t end) {
    if (end == spans.size()) { return text.size(); }
    return spans.at(end).offset;
}

std::vector<std::string_view> qwen_split_words(std::string_view text) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "QwenTokenizer::encode input");
    std::vector<std::string_view> words;
    for (std::size_t i = 0; i < spans.size();) {
        const std::size_t begin_offset = spans[i].offset;
        const std::int32_t cp          = spans[i].value;

        if (cp == '\'') {
            constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (std::string_view suffix : suffixes) {
                if (ascii_ci_matches(text, begin_offset + 1, suffix)) {
                    std::size_t end = i + 1;
                    while (end < spans.size() &&
                           spans[end].offset < begin_offset + 1 + suffix.size()) {
                        ++end;
                    }
                    words.emplace_back(text.substr(begin_offset, span_end_offset(text, spans, end) -
                                                                     begin_offset));
                    i = end;
                    goto next_word;
                }
            }
        }

        if (is_letter_or_mark(cp) ||
            (is_non_newline_non_letter_non_number(cp) && i + 1 < spans.size() &&
             is_letter_or_mark(spans[i + 1].value))) {
            std::size_t end = i;
            if (!is_letter_or_mark(spans[end].value)) { ++end; }
            while (end < spans.size() && is_letter_or_mark(spans[end].value)) { ++end; }
            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            goto next_word;
        }

        if (uni::is_number(cp)) {
            words.emplace_back(text.substr(begin_offset, spans[i].length));
            ++i;
            goto next_word;
        }

        if ((cp == ' ' && i + 1 < spans.size() &&
             is_non_space_non_letter_mark_number(spans[i + 1].value)) ||
            is_non_space_non_letter_mark_number(cp)) {
            std::size_t end = i;
            if (spans[end].value == ' ') { ++end; }
            while (end < spans.size() && is_non_space_non_letter_mark_number(spans[end].value)) {
                ++end;
            }
            while (end < spans.size() && is_newline(spans[end].value)) { ++end; }
            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            goto next_word;
        }

        if (uni::is_whitespace(cp)) {
            std::size_t run_end      = i;
            std::size_t last_newline = std::string_view::npos;
            while (run_end < spans.size() && uni::is_whitespace(spans[run_end].value)) {
                if (is_newline(spans[run_end].value)) { last_newline = run_end; }
                ++run_end;
            }
            if (last_newline != std::string_view::npos) {
                const std::size_t end = last_newline + 1;
                words.emplace_back(
                    text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                goto next_word;
            }

            if (run_end == spans.size()) {
                words.emplace_back(text.substr(begin_offset));
                i = run_end;
                goto next_word;
            }

            if (run_end - i >= 2) {
                const std::size_t end = run_end - 1;
                words.emplace_back(
                    text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                goto next_word;
            }

            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, run_end) - begin_offset));
            i = run_end;
            goto next_word;
        }

        words.emplace_back(text.substr(begin_offset, spans[i].length));
        ++i;

    next_word:;
    }
    return words;
}

std::string byte_level_encode(std::string_view text) {
    static const std::unordered_map<unsigned char, std::string> byte_encoder =
        build_byte_level_encoder();
    std::string encoded;
    encoded.reserve(text.size());
    for (const unsigned char byte : text) { encoded += byte_encoder.at(byte); }
    return encoded;
}

std::vector<std::string> byte_level_symbols(std::string_view text) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "QwenTokenizer::encode byte-level text");
    std::vector<std::string> symbols;
    symbols.reserve(spans.size());
    for (const uni::CodepointSpan& span : spans) {
        symbols.emplace_back(text.substr(span.offset, span.length));
    }
    return symbols;
}

bool is_added_token_id(const std::vector<AddedToken>& added_tokens, int id) {
    return std::any_of(added_tokens.begin(), added_tokens.end(),
                       [id](const AddedToken& token) { return token.id == id; });
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), id) != stop_token_ids.end();
}

std::size_t valid_utf8_prefix_size(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto lead         = static_cast<unsigned char>(bytes[offset]);
        std::size_t length      = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum   = 0;
        if (lead <= 0x7F) {
            length    = 1;
            codepoint = lead;
            minimum   = 0;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length    = 2;
            codepoint = lead & 0x1FU;
            minimum   = 0x80U;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length    = 3;
            codepoint = lead & 0x0FU;
            minimum   = 0x800U;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length    = 4;
            codepoint = lead & 0x07U;
            minimum   = 0x10000U;
        } else {
            throw std::invalid_argument("invalid UTF-8 leading byte in token stream");
        }

        if (offset + length > bytes.size()) { return offset; }
        for (std::size_t i = 1; i < length; ++i) {
            const auto byte = static_cast<unsigned char>(bytes[offset + i]);
            if ((byte & 0xC0U) != 0x80U) {
                throw std::invalid_argument("invalid UTF-8 continuation byte in token stream");
            }
            codepoint = (codepoint << 6U) | (byte & 0x3FU);
        }
        if (codepoint < minimum || (codepoint >= 0xD800U && codepoint <= 0xDFFFU) ||
            codepoint > 0x10FFFFU) {
            throw std::invalid_argument("invalid UTF-8 codepoint in token stream");
        }
        offset += length;
    }
    return offset;
}

void append_symbol_id(std::vector<int>& ids,
                      const std::unordered_map<std::string, int>& token_to_id,
                      std::string_view symbol) {
    const auto direct = token_to_id.find(std::string(symbol));
    if (direct != token_to_id.end()) {
        ids.push_back(direct->second);
        return;
    }

    const std::vector<std::string> bytes = byte_level_symbols(symbol);
    if (bytes.size() <= 1) {
        throw std::invalid_argument("QwenTokenizer::encode produced token outside vocabulary: " +
                                    std::string(symbol));
    }
    for (const std::string& byte_symbol : bytes) {
        const auto byte_id = token_to_id.find(byte_symbol);
        if (byte_id == token_to_id.end()) {
            throw std::invalid_argument(
                "QwenTokenizer::encode produced byte symbol outside vocabulary: " + byte_symbol);
        }
        ids.push_back(byte_id->second);
    }
}

void append_bpe_ids(std::vector<int>& ids, std::string_view text, bool has_bpe_merges,
                    const std::unordered_map<std::string, int>& merge_ranks,
                    const std::unordered_map<std::string, int>& token_to_id) {
    if (text.empty()) { return; }
    if (!has_bpe_merges) {
        throw std::invalid_argument(
            "QwenTokenizer::encode ordinary BPE text requires embedded merges.txt");
    }

    const std::string normalized = uni::normalize_nfc(text);
    for (const std::string_view word : qwen_split_words(normalized)) {
        std::vector<std::string> symbols = byte_level_symbols(byte_level_encode(word));
        while (symbols.size() > 1) {
            int best_rank              = std::numeric_limits<int>::max();
            std::size_t best_pair_left = symbols.size();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto rank = merge_ranks.find(merge_pair_key(symbols[i], symbols[i + 1]));
                if (rank != merge_ranks.end() && rank->second < best_rank) {
                    best_rank      = rank->second;
                    best_pair_left = i;
                }
            }
            if (best_pair_left == symbols.size()) { break; }
            symbols[best_pair_left] += symbols[best_pair_left + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_pair_left + 1));
        }
        for (const std::string& symbol : symbols) { append_symbol_id(ids, token_to_id, symbol); }
    }
}

} // namespace

QwenTokenizer::QwenTokenizer(Q5090TokenizerBundle bundle) {
    if (bundle.empty()) { throw std::invalid_argument("embedded q5090 tokenizer bundle is empty"); }
    constexpr std::string_view tokenizer_label = "tokenizer.json";
    const Json root   = read_json_asset(bundle.tokenizer_json, tokenizer_label);
    const Json& model = require_object_field(root, "model", tokenizer_label);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_label);
    id_to_token_                 = std::move(vocab_metadata.id_to_token);
    vocab_token_to_id_           = std::move(vocab_metadata.token_to_id);
    valid_token_ids_.resize(id_to_token_.size());
    for (const int id : vocab_metadata.occupied_ids) {
        valid_token_ids_.at(static_cast<std::size_t>(id)) = true;
    }
    added_tokens_ =
        load_added_tokens(root, tokenizer_label, id_to_token_, vocab_metadata.occupied_ids);
    if (valid_token_ids_.size() < id_to_token_.size()) {
        valid_token_ids_.resize(id_to_token_.size());
    }
    for (const AddedToken& token : added_tokens_) {
        valid_token_ids_.at(static_cast<std::size_t>(token.id)) = true;
    }
    bpe_merge_ranks_        = load_bpe_merge_ranks(bundle.merges_txt);
    has_bpe_merges_         = true;
    default_stop_token_ids_ = load_default_stop_token_ids(bundle.generation_config_json);
}

std::vector<int> QwenTokenizer::encode(std::string_view text, EncodeOptions options) const {
    if (text.empty()) { return {}; }
    if (!options.parse_added_tokens) {
        std::vector<int> ids;
        append_bpe_ids(ids, text, has_bpe_merges_, bpe_merge_ranks_, vocab_token_to_id_);
        return ids;
    }

    std::vector<int> ids;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t match_pos         = std::string_view::npos;
        const AddedToken* match_token = nullptr;
        for (const AddedToken& token : added_tokens_) {
            if (token.content.empty()) { continue; }
            const std::size_t found = text.find(token.content, pos);
            if (found == std::string_view::npos) { continue; }
            if (match_token == nullptr || found < match_pos) {
                match_pos   = found;
                match_token = &token;
            }
        }

        if (match_token == nullptr) {
            append_bpe_ids(ids, text.substr(pos), has_bpe_merges_, bpe_merge_ranks_,
                           vocab_token_to_id_);
            break;
        }
        if (match_pos > pos) {
            append_bpe_ids(ids, text.substr(pos, match_pos - pos), has_bpe_merges_,
                           bpe_merge_ranks_, vocab_token_to_id_);
        }

        ids.push_back(match_token->id);
        pos = match_pos + match_token->content.size();
    }
    return ids;
}

std::string QwenTokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(id, options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "QwenTokenizer::decode reconstructed output");
    return text;
}

std::string QwenTokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    static const std::unordered_map<std::uint32_t, char> byte_decoder = build_byte_level_decoder();

    if (skip_special_tokens && is_special_token(id)) { return {}; }
    if (id < 0) {
        throw std::invalid_argument("QwenTokenizer::decode received negative token id " +
                                    std::to_string(id));
    }
    const auto index = static_cast<std::size_t>(id);
    if (index >= id_to_token_.size() || index >= valid_token_ids_.size() ||
        !valid_token_ids_.at(index)) {
        throw std::out_of_range("QwenTokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }

    const std::string& token = id_to_token_.at(index);
    if (is_added_token_id(added_tokens_, id)) { return token; }

    std::string bytes;
    const std::vector<uni::CodepointSpan> codepoints =
        uni::utf8_codepoints(token, "QwenTokenizer::decode token id " + std::to_string(id));
    for (const uni::CodepointSpan& codepoint : codepoints) {
        const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
        if (byte == byte_decoder.end()) {
            throw std::invalid_argument("QwenTokenizer::decode token id " + std::to_string(id) +
                                        " contains a character outside the byte-level alphabet");
        }
        bytes.push_back(byte->second);
    }
    return bytes;
}

bool QwenTokenizer::is_special_token(int id) const noexcept {
    return std::any_of(added_tokens_.begin(), added_tokens_.end(),
                       [id](const AddedToken& token) { return token.id == id && token.special; });
}

TokenStreamDecoder::TokenStreamDecoder(const QwenTokenizer& tokenizer, DecodeOptions options)
    : tokenizer_(tokenizer), options_(std::move(options)) {}

std::string TokenStreamDecoder::append(int token_id) {
    if (stopped_) { return {}; }
    if (is_stop_token_id(options_.stop_token_ids, token_id)) {
        stopped_ = true;
        return {};
    }

    pending_bytes_ += tokenizer_.decode_token_bytes(token_id, options_.skip_special_tokens);
    const std::size_t prefix_size = valid_utf8_prefix_size(pending_bytes_);
    if (prefix_size == 0) { return {}; }

    std::string text = pending_bytes_.substr(0, prefix_size);
    pending_bytes_.erase(0, prefix_size);
    return text;
}

std::string TokenStreamDecoder::finish() {
    if (pending_bytes_.empty()) { return {}; }
    (void)uni::utf8_codepoints(pending_bytes_, "QwenTokenizer stream decoder pending bytes");
    std::string text = std::move(pending_bytes_);
    pending_bytes_.clear();
    return text;
}

} // namespace ninfer::text
