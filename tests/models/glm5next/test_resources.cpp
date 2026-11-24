#include "models/glm5next/frontend/resources.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using ninfer::models::glm5next::FrontendResources;

std::string read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), {}};
}

void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

template <class Action>
void rejects(Action action, const char* message) {
    try {
        action();
    } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

void synthetic() {
    const std::string tokenizer = R"json({
      "normalizer":null,
      "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
      "decoder":{"type":"ByteLevel"},
      "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[],"ignore_merges":true},
      "added_tokens":[
        {"id":2,"content":"<eos>","special":true,"single_word":false,"normalized":false,"lstrip":false,"rstrip":false},
        {"id":3,"content":"</think>","special":false,"single_word":false,"normalized":false,"lstrip":false,"rstrip":false}]
    })json";
    std::string generation      = R"({"eos_token_id":[2],"temperature":1.0,"top_p":0.95})";
    FrontendResources resources{.text = {.tokenizer_json         = tokenizer,
                                         .tokenizer_config_json  = "{}",
                                         .chat_template_jinja    = "template bytes",
                                         .generation_config_json = generation}};
    ninfer::models::glm5next::parse_resources(resources, 8);
    require(resources.text.public_token_count == 4 &&
                resources.text.stop_token_ids == std::vector<int>{2},
            "public token domain or default EOS changed");
    require(!resources.text.tokenizer->is_valid_token(4) &&
                resources.text.tokenizer->is_special_token(2) &&
                !resources.text.tokenizer->is_special_token(3),
            "padding rows or ordinary added tokens acquired special-token semantics");
    require(resources.sampling.temperature == 1.0F && resources.sampling.top_p == 0.95F &&
                resources.sampling.top_k == 0,
            "generation defaults changed");
    require(resources.text.chat_template_jinja == "template bytes",
            "resource parsing changed the template");
    resources.text.tokenizer_config_json = "null";
    rejects([&] { ninfer::models::glm5next::parse_resources(resources, 8); },
            "non-object tokenizer config was accepted");
    resources.text.tokenizer_config_json = "{}";
    rejects([&] { ninfer::models::glm5next::parse_resources(resources, 3); },
            "token domain exceeded embedding rows");
    generation                            = R"({"eos_token_id":[4]})";
    resources.text.generation_config_json = generation;
    rejects([&] { ninfer::models::glm5next::parse_resources(resources, 8); },
            "weight padding was accepted as EOS");
    generation = R"({"eos_token_id":2,"do_sample":false,"temperature":1.5,"top_p":0.0,"top_k":7})";
    resources.text.generation_config_json = generation;
    ninfer::models::glm5next::parse_resources(resources, 8);
    require(resources.sampling.temperature == 0.0F && resources.sampling.top_p == 0.0F &&
                resources.sampling.top_k == 7,
            "greedy or explicit zero defaults changed");
    for (const auto* invalid :
         {R"({"eos_token_id":2,"top_k":1.5})", R"({"eos_token_id":2,"top_p":1e100})"}) {
        resources.text.generation_config_json = invalid;
        rejects([&] { ninfer::models::glm5next::parse_resources(resources, 8); },
                "generation defaults changed numeric representation silently");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            synthetic();
            std::cout << "GLM frontend resources passed\n";
        } else if (argc == 3) {
            const std::filesystem::path root(argv[1]);
            const auto tokenizer  = read(root / "tokenizer.json");
            const auto config     = read(root / "tokenizer_config.json");
            const auto generation = read(root / "generation_config.json");
            FrontendResources resources{.text = {.tokenizer_json         = tokenizer,
                                                 .tokenizer_config_json  = config,
                                                 .generation_config_json = generation}};
            ninfer::models::glm5next::parse_resources(resources, std::stoul(argv[2]));
            std::cout << Json{{"public_tokens", resources.text.public_token_count},
                              {"eos", resources.text.stop_token_ids},
                              {"temperature", resources.sampling.temperature},
                              {"top_p", resources.sampling.top_p},
                              {"top_k", resources.sampling.top_k}}
                             .dump()
                      << '\n';
        } else
            throw std::invalid_argument("usage: resources_test [RESOURCE_DIR EMBEDDING_ROWS]");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
