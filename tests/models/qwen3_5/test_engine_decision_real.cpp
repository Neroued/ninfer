#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Reads whitespace-separated token ids from NINFER_TEST_TOKENS and prints the BF16 bits of the
// noncausal last-token hidden state, one hex word per line.
int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    const char* tokens   = std::getenv("NINFER_TEST_TOKENS");
    if (artifact == nullptr || *artifact == '\0' || tokens == nullptr || *tokens == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT or NINFER_TEST_TOKENS is not set\n";
        return 77;
    }
    std::ifstream in(tokens);
    std::vector<ninfer::TokenId> ids;
    for (long long value = 0; in >> value;) { ids.push_back(static_cast<ninfer::TokenId>(value)); }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::Decision;
    options.max_context   = 8192;
    ninfer::Engine engine(options);
    if (const char* count = std::getenv("NINFER_TEST_OPTIONS"); count != nullptr && *count != '\0') {
        const auto probabilities =
            engine.decide(ids, static_cast<std::uint32_t>(std::strtoul(count, nullptr, 10)));
        std::cout.precision(9);
        for (const float value : probabilities) { std::cout << value << '\n'; }
        return 0;
    }
    const std::vector<std::uint16_t> hidden = engine.decision_hidden(ids);
    for (const std::uint16_t word : hidden) { std::cout << std::hex << word << '\n'; }
    return 0;
}
