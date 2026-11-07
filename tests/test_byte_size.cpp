#include "core/byte_size.h"

#include <iostream>
#include <limits>

int main() {
    int failures     = 0;
    const auto check = [&](bool condition, const char* label) {
        if (!condition) {
            std::cerr << "FAIL: " << label << '\n';
            ++failures;
        }
    };

    struct Valid {
        const char* text;
        std::uint64_t bytes;
    };

    const Valid valid[] = {
        {"1", 1ULL << 30},
        {"0.5", 1ULL << 29},
        {"1.25", 5ULL << 28},
        {"+1", 1ULL << 30},
        {" 1", 1ULL << 30},
        {"1e1", 10ULL << 30},
        {"0x1p-30", 1},
        {"0x1.8p-30", 1},
        {"0x1.fffffffffffffp33", std::numeric_limits<std::uint64_t>::max() - 2047},
    };
    for (const auto& test : valid) {
        check(ninfer::parse_positive_gib(test.text) == test.bytes, test.text);
    }
    for (const char* text :
         {"", " ", "0", "-0", "-1", "1e-12", "1e-9999", "0x1.fffffffffffffp-31", "17179869184",
          "1e100", "1e9999", "nan", "inf", "-inf", "1garbage", "1 GiB", "1 ", "."}) {
        check(!ninfer::parse_positive_gib(text), text);
    }
    check(!ninfer::parse_positive_gib(nullptr), "null input");
    std::cout << (failures ? "FAIL" : "PASS") << " byte-size parsing\n";
    return failures ? 1 : 0;
}
