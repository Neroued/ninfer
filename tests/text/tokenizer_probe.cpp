#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
using Json = nlohmann::json;

std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

Json run(const ninfer::text::Tokenizer& tokenizer, const Json& input) {
    const auto text       = input.at("text").get<std::string>();
    const auto boundaries = input.value("boundaries", std::vector<std::size_t>{});
    std::vector<ninfer::text::ByteSpan> literal;
    for (const auto& span : input.value("literal_spans", Json::array()))
        literal.push_back({span.at(0).get<std::size_t>(), span.at(1).get<std::size_t>()});
    ninfer::text::EncodeOptions options;
    options.parse_added_tokens = input.value("parse_added_tokens", true);
    const auto encoded = tokenizer.encode_with_boundaries(text, boundaries, options, literal);
    Json result{
        {"ids", encoded.input_ids},
        {"decoded", tokenizer.decode(encoded.input_ids)},
        {"skip_special", tokenizer.decode(encoded.input_ids, {.skip_special_tokens = true})},
        {"boundaries", Json::array()},
        {"limited", Json::array()}};
    for (const auto& boundary : encoded.boundaries)
        result["boundaries"].push_back(
            {{"exact", boundary.exact_frontier ? Json(*boundary.exact_frontier) : Json(nullptr)},
             {"stable", boundary.stable_frontier}});
    for (const auto limit : input.value("limits", std::vector<std::size_t>{})) {
        options.max_tokens = limit;
        result["limited"].push_back(
            tokenizer.encode_with_boundaries(text, {}, options, literal).input_ids);
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: tokenizer_probe RESOURCE_DIR");
        const std::filesystem::path root(argv[1]);
        const auto source = read(root / "tokenizer.json");
        const auto config = read(root / "tokenizer_config.json");
        const ninfer::text::Tokenizer tokenizer({source, config});
        for (std::string line; std::getline(std::cin, line);) {
            try {
                std::cout << run(tokenizer, Json::parse(line)).dump() << '\n';
            } catch (const std::exception& error) {
                std::cout << Json{{"error", error.what()}}.dump() << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
