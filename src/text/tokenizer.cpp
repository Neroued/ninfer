#include "qus/text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace qus::text {
namespace {

using Json = nlohmann::json;

constexpr std::int64_t kMaxTokenId = 1'000'000;

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_set<int> occupied_ids;
};

Json read_json_file(const std::filesystem::path& path, const char* label) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument(std::string("missing ") + label + ": " + path.string());
    }

    Json root;
    try {
        in >> root;
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument(std::string("malformed ") + label + " at " + path.string() +
                                    ": " + ex.what());
    }
    return root;
}

const Json& require_object_field(const Json& object, const char* field,
                                 const std::filesystem::path& path) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " + path.string());
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    path.string());
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field,
                                const std::filesystem::path& path) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " + path.string());
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    path.string());
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, const std::filesystem::path& path) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    path.string());
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        path.string());
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    path.string());
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    path.string());
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field,
                                 const std::filesystem::path& path) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    path.string());
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, const std::filesystem::path& path) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    path.string());
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, const std::filesystem::path& path) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + path.string());
    }
    const Json& vocab = require_object_field(model, "vocab", path);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + path.string());
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", path);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " + path.string());
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        metadata.id_to_token.at(static_cast<std::size_t>(
            parse_token_id(item.value(), "model.vocab", path))) = item.key();
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, const std::filesystem::path& path) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " + path.string());
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + path.string());
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", path);
    token.content     = require_string_field(item, "content", path);
    token.single_word = require_bool_field(item, "single_word", path);
    token.lstrip      = require_bool_field(item, "lstrip", path);
    token.rstrip      = require_bool_field(item, "rstrip", path);
    token.normalized  = require_bool_field(item, "normalized", path);
    token.special     = require_bool_field(item, "special", path);
    return token;
}

std::vector<AddedToken> load_added_tokens(const Json& root, const std::filesystem::path& path,
                                          std::vector<std::string>& id_to_token,
                                          const std::unordered_set<int>& occupied_vocab_ids) {
    const Json& added = require_array_field(root, "added_tokens", path);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, path);
        const auto index = static_cast<std::size_t>(token.id);
        if (occupied_vocab_ids.contains(token.id)) {
            throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                        path.string());
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " + path.string());
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<int> load_default_stop_token_ids(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) { return {248046, 248044}; }

    const Json root = read_json_file(path, "generation_config.json");
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in " + path.string());
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", path)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument("field eos_token_id must not be empty in " + path.string());
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", path)); }
        return ids;
    }
    throw std::invalid_argument("field eos_token_id must be integer or array in " + path.string());
}

} // namespace

QwenTokenizer::QwenTokenizer(const std::filesystem::path& tokenizer_dir)
    : tokenizer_dir_(tokenizer_dir) {
    const std::filesystem::path tokenizer_json = tokenizer_dir_ / "tokenizer.json";
    const Json root                            = read_json_file(tokenizer_json, "tokenizer.json");
    const Json& model = require_object_field(root, "model", tokenizer_json);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_json);
    id_to_token_                 = std::move(vocab_metadata.id_to_token);
    added_tokens_ =
        load_added_tokens(root, tokenizer_json, id_to_token_, vocab_metadata.occupied_ids);
    default_stop_token_ids_ =
        load_default_stop_token_ids(tokenizer_dir_ / "generation_config.json");
}

std::vector<int> QwenTokenizer::encode(std::string_view text, EncodeOptions options) const {
    (void)text;
    (void)options;
    throw std::logic_error("QwenTokenizer::encode is not implemented yet");
}

std::string QwenTokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    (void)ids;
    (void)options;
    throw std::logic_error("QwenTokenizer::decode is not implemented yet");
}

bool QwenTokenizer::is_special_token(int id) const noexcept {
    return std::any_of(added_tokens_.begin(), added_tokens_.end(),
                       [id](const AddedToken& token) { return token.id == id && token.special; });
}

} // namespace qus::text
