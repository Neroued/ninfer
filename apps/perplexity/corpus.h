#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::perplexity {

struct CorpusStream {
    std::string id;
    std::string domain;
    std::filesystem::path path;
    std::string text;
};

struct CorpusSelection {
    std::string corpus_id;
    std::string mode;
    std::filesystem::path source;
    std::vector<CorpusStream> streams;
};

[[nodiscard]] CorpusSelection load_corpus(const std::filesystem::path& manifest, bool quick);
[[nodiscard]] CorpusSelection load_custom_text(const std::filesystem::path& path);

} // namespace ninfer::perplexity
