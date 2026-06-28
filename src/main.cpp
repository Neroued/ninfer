#include "qus/runtime/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " <weights.qus> [--max-context N] [--max-new N] [--stop-token-id N]... "
                 "<token-id> [token-id...]\n";
}

int parse_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    try {
        const std::string path = argv[1];
        int max_new            = 16;
        std::uint32_t max_ctx  = 2048;
        std::vector<int> stop_tokens;
        std::vector<int> prompt;

        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--max-new") {
                if (++i >= argc) { throw std::invalid_argument("--max-new needs a value"); }
                max_new = parse_int(argv[i], "max-new");
            } else if (arg == "--stop-token-id") {
                if (++i >= argc) { throw std::invalid_argument("--stop-token-id needs a value"); }
                stop_tokens.push_back(parse_int(argv[i], "stop-token-id"));
            } else if (arg == "--max-context") {
                if (++i >= argc) { throw std::invalid_argument("--max-context needs a value"); }
                max_ctx = static_cast<std::uint32_t>(parse_int(argv[i], "max-context"));
            } else {
                prompt.push_back(parse_int(argv[i], "token-id"));
            }
        }

        if (prompt.empty()) {
            throw std::invalid_argument("at least one prompt token is required");
        }

        qus::EngineOptions options;
        options.max_ctx        = max_ctx;
        options.stop_token_ids = stop_tokens;
        qus::Engine engine(options);
        engine.load(path);

        const auto start        = std::chrono::steady_clock::now();
        std::vector<int> tokens = engine.generate(std::span<const int>(prompt), max_new);
        const auto end          = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i != 0) { std::cout << ' '; }
            std::cout << tokens[i];
        }
        std::cout << '\n';

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double tok_s   = seconds > 0.0 ? static_cast<double>(tokens.size()) / seconds : 0.0;
        std::cerr << "generated=" << tokens.size() << " elapsed_s=" << seconds << " tok_s=" << tok_s
                  << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
