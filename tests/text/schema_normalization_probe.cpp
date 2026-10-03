// Standalone CPU-only harness; no model, CUDA, XGrammar or provider dependency.
#include "text/schema_normalization.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

int main() {
    using Json = nlohmann::ordered_json;
    std::string line;
    while (std::getline(std::cin, line)) {
        Json output = Json::object();
        try {
            const Json input = Json::parse(line);
            output["id"] = input.at("id");
            output["schema"] = Json::parse(ninfer::text::normalize_json_schema(input.at("schema").dump()));
        } catch (const std::exception& error) {
            output["error"] = error.what();
        }
        std::cout << output.dump() << '\n';
    }
}
