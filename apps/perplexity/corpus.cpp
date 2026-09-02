#include "corpus.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::perplexity {
namespace {

using json = nlohmann::json;

std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open text file: " + path.string()); }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size <= 0) { throw std::runtime_error("text file is empty: " + path.string()); }
    input.seekg(0, std::ios::beg);
    std::string bytes(static_cast<std::size_t>(size), '\0');
    input.read(bytes.data(), size);
    if (!input) { throw std::runtime_error("cannot read text file: " + path.string()); }
    return bytes;
}

bool valid_utf8(std::string_view text) noexcept {
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t i     = 0;
    while (i < text.size()) {
        const unsigned char lead = bytes[i++];
        if (lead <= 0x7fU) {
            if (lead == 0) { return false; }
            continue;
        }
        std::uint32_t value      = 0;
        std::size_t continuation = 0;
        if (lead >= 0xc2U && lead <= 0xdfU) {
            value        = lead & 0x1fU;
            continuation = 1;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            value        = lead & 0x0fU;
            continuation = 2;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            value        = lead & 0x07U;
            continuation = 3;
        } else {
            return false;
        }
        if (continuation > text.size() - i) { return false; }
        for (std::size_t j = 0; j < continuation; ++j) {
            const unsigned char byte = bytes[i++];
            if ((byte & 0xc0U) != 0x80U) { return false; }
            value = (value << 6U) | (byte & 0x3fU);
        }
        if ((continuation == 1 && value < 0x80U) || (continuation == 2 && value < 0x800U) ||
            (continuation == 3 && value < 0x10000U) || value > 0x10ffffU ||
            (value >= 0xd800U && value <= 0xdfffU)) {
            return false;
        }
    }
    return true;
}

void validate_text(const std::string& text, const std::filesystem::path& path) {
    if (!valid_utf8(text)) {
        throw std::runtime_error("text file is not nonempty NUL-free UTF-8: " + path.string());
    }
}

std::filesystem::path checked_relative_path(std::string_view value) {
    std::filesystem::path path(value);
    if (path.empty() || path.is_absolute()) {
        throw std::runtime_error("corpus stream path must be relative");
    }
    for (const auto& component : path) {
        if (component == "..") { throw std::runtime_error("corpus stream path escapes manifest"); }
    }
    return path.lexically_normal();
}

std::string required_string(const json& value, const char* key) {
    if (!value.contains(key) || !value.at(key).is_string()) {
        throw std::runtime_error(std::string("manifest field '") + key + "' must be a string");
    }
    std::string out = value.at(key).get<std::string>();
    if (out.empty()) {
        throw std::runtime_error(std::string("manifest field '") + key + "' is empty");
    }
    return out;
}

} // namespace

CorpusSelection load_corpus(const std::filesystem::path& manifest, bool quick) {
    std::ifstream input(manifest);
    if (!input) { throw std::runtime_error("cannot open corpus manifest: " + manifest.string()); }
    json root;
    try {
        input >> root;
    } catch (const json::exception& error) {
        throw std::runtime_error(std::string("invalid corpus manifest JSON: ") + error.what());
    }
    if (!root.is_object() || !root.contains("streams") || !root.at("streams").is_array() ||
        !root.contains("modes") || !root.at("modes").is_object()) {
        throw std::runtime_error("corpus manifest must contain streams and modes");
    }

    struct Metadata {
        std::string domain;
        std::filesystem::path path;
    };

    std::unordered_map<std::string, Metadata> metadata;
    for (const json& stream : root.at("streams")) {
        if (!stream.is_object()) { throw std::runtime_error("corpus stream must be an object"); }
        const std::string id = required_string(stream, "id");
        Metadata item;
        item.domain = required_string(stream, "domain");
        item.path   = checked_relative_path(required_string(stream, "path"));
        if (!metadata.emplace(id, std::move(item)).second) {
            throw std::runtime_error("corpus stream identity is empty or duplicated");
        }
    }

    const auto mode_ids = [&](std::string_view name) {
        if (!root.at("modes").contains(name) || !root.at("modes").at(name).is_array()) {
            throw std::runtime_error("corpus manifest mode is missing");
        }
        std::vector<std::string> ids;
        std::unordered_set<std::string> unique;
        for (const json& value : root.at("modes").at(name)) {
            if (!value.is_string()) { throw std::runtime_error("corpus mode ID must be a string"); }
            std::string id = value.get<std::string>();
            if (!metadata.contains(id) || !unique.insert(id).second) {
                throw std::runtime_error("corpus mode contains an unknown or duplicate stream");
            }
            ids.push_back(std::move(id));
        }
        if (ids.empty()) { throw std::runtime_error("corpus mode is empty"); }
        return ids;
    };
    (void)mode_ids("quick");
    (void)mode_ids("full");
    const std::string mode                  = quick ? "quick" : "full";
    const std::vector<std::string> selected = mode_ids(mode);

    CorpusSelection result;
    result.corpus_id = required_string(root, "corpus_id");
    result.mode      = mode;
    result.source    = std::filesystem::absolute(manifest).lexically_normal();
    const std::filesystem::path root_directory = manifest.parent_path();
    for (const std::string& id : selected) {
        const Metadata& item             = metadata.at(id);
        const std::filesystem::path path = root_directory / item.path;
        std::string text                 = read_bytes(path);
        validate_text(text, path);
        result.streams.push_back(CorpusStream{.id     = id,
                                              .domain = item.domain,
                                              .path   = item.path,
                                              .text   = std::move(text)});
    }
    return result;
}

CorpusSelection load_custom_text(const std::filesystem::path& path) {
    std::string text = read_bytes(path);
    validate_text(text, path);
    CorpusSelection result;
    result.corpus_id = "custom-" + path.filename().string();
    result.mode      = "custom";
    result.source    = std::filesystem::absolute(path).lexically_normal();
    result.streams.push_back(CorpusStream{.id     = path.filename().string(),
                                          .domain = "custom",
                                          .path   = result.source,
                                          .text   = std::move(text)});
    return result;
}

} // namespace ninfer::perplexity
