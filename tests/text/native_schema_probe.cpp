// CPU-only stdin JSONL probe for StructuredCompiler. No model, tokenizer, CUDA or HTTP.
#include "text/structured_output.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ninfer;
    using namespace ninfer::text;
    // Byte alphabet and EOS are protocol fixtures; mask width follows the public API.
    constexpr std::size_t byte_alphabet = 256;
    constexpr TokenId eos = byte_alphabet;
    constexpr std::size_t bits_per_mask_word = 32;
    std::vector<std::string> vocabulary(byte_alphabet + 1);
    for (std::size_t i = 0; i < byte_alphabet; ++i) {
        vocabulary[i] = std::string(1, static_cast<char>(i));
    }
    StructuredCompiler compiler(vocabulary, {eos});
    std::map<std::string, std::shared_ptr<GrammarState>> compiled;
    std::map<std::string, std::string> rejected;
    const auto mask_words = (vocabulary.size() + bits_per_mask_word - 1) / bits_per_mask_word;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) { continue; }
        nlohmann::json result;
        try {
            // Preserve schema declaration order, as the server's request parser does.
            auto item = nlohmann::ordered_json::parse(line);
            result["id"] = item.at("id");
            result["compile_only"] = item.value("compile_only", false);
            if (item.contains("name")) { result["name"] = item.at("name"); }
            const auto schema = item.at("schema").dump();
            if (!compiled.contains(schema) && !rejected.contains(schema)) {
                try {
                    compiled[schema] = compiler.compile({StructuredOutputKind::JsonSchema, schema});
                } catch (const std::exception& error) {
                    rejected[schema] = error.what();
                }
            }
            if (rejected.contains(schema)) {
                result["compile_error"] = rejected.at(schema);
                result["accepted"] = false;
            } else if (item.value("compile_only", false)) {
                result["compiled"] = true;
            } else {
                auto trial = compiled.at(schema)->fork();
                std::vector<TokenId> tokens;
                if (item.contains("wire_bytes")) {
                    for (const auto& value : item.at("wire_bytes")) {
                        if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
                            value.get<std::int64_t>() >= static_cast<std::int64_t>(byte_alphabet)) {
                            throw std::invalid_argument("wire_bytes entries must be bytes");
                        }
                        tokens.push_back(value.get<TokenId>());
                    }
                } else {
                    for (unsigned char byte : item.at("wire").get<std::string>()) {
                        tokens.push_back(byte);
                    }
                }
                tokens.push_back(eos);
                bool accepted = true;
                result["accepted_prefix_tokens"] = 0;
                std::vector<std::uint32_t> masks(mask_words);
                for (auto token : tokens) {
                    try {
                        trial->fill_masks(masks, {});
                    } catch (const std::exception& error) {
                        result["internal_error"] = error.what();
                        result["failure_stage"] = "fill_masks";
                        result["next_token"] = token;
                        accepted = false;
                        break;
                    }
                    if (!(masks[token / bits_per_mask_word] & (1U << (token % bits_per_mask_word)))) {
                        accepted = false;
                        break;
                    }
                    // Mask permission and committed matcher acceptance must agree.
                    try {
                        trial->accept(std::vector<TokenId>{token});
                    } catch (const std::exception& error) {
                        result["internal_error"] = error.what();
                        result["failure_stage"] = "accept";
                        result["next_token"] = token;
                        accepted = false;
                        break;
                    }
                    result["accepted_prefix_tokens"] = result["accepted_prefix_tokens"].get<std::size_t>() + 1;
                }
                result["accepted"] = accepted;
            }
        } catch (const std::exception& error) {
            result["internal_error"] = error.what();
            result["accepted"] = false;
        }
        std::cout << result.dump() << '\n';
    }
}
