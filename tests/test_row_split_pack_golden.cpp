#include "ops/row_split_pack.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    error "NINFER_SOURCE_DIR must be defined for ninfer_row_split_pack_golden_test"
#endif

namespace {

constexpr std::int32_t kN = 70;
constexpr std::int32_t kK = 130;

std::string shell_quote(std::string_view value) {
#ifdef _WIN32
    std::string out = "\"";
    out.append(value);
    out += '\"';
#else
    std::string out = "'";
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
#endif
    return out;
}

std::filesystem::path unique_temp_path(std::string_view stem, std::string_view suffix) {
    static int counter = 0;
    const auto ticks   = std::chrono::steady_clock::now().time_since_epoch().count() +
                       static_cast<long long>(counter++);
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + std::to_string(ticks) + std::string(suffix));
}

std::vector<float> known_matrix() {
    std::vector<float> w(static_cast<std::size_t>(kN) * kK);
    for (std::int32_t row = 0; row < kN; ++row) {
        for (std::int32_t col = 0; col < kK; ++col) {
            const int base     = ((row * 37 + col * 17) % 211) - 105;
            const double sign  = ((row + col) & 1) ? -1.0 : 1.0;
            const double value = sign * (static_cast<double>(base) / 37.0) +
                                 static_cast<double>((row % 5) - 2) * 0.03125;
            w[static_cast<std::size_t>(row) * kK + col] = static_cast<float>(value);
        }
    }
    return w;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open converter golden payload"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

int run_converter(const std::filesystem::path& script, const std::filesystem::path& output_dir) {
    std::string command = shell_quote(NINFER_PYTHON_EXECUTABLE) + " " +
                          shell_quote(script.string()) + " " + shell_quote(output_dir.string());
#ifdef _WIN32
    // std::system dispatches through cmd.exe, which consumes the first quoted token unless the
    // complete command line has its own outer quote pair.
    command = '"' + command + '"';
#endif
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        std::cerr << "converter golden command failed: " << command << '\n';
        return 1;
    }
    return 0;
}

int compare_payload(std::string_view label, const std::vector<std::uint8_t>& actual,
                    const std::vector<std::uint8_t>& expected) {
    if (actual.size() != expected.size()) {
        std::cerr << label << " payload size mismatch: actual=" << actual.size()
                  << " expected=" << expected.size() << '\n';
        return 1;
    }
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first == actual.end()) { return 0; }

    const auto index = static_cast<std::size_t>(mismatch.first - actual.begin());
    std::cerr << label << " payload mismatch at byte " << index
              << ": actual=" << static_cast<int>(*mismatch.first)
              << " expected=" << static_cast<int>(*mismatch.second) << '\n';
    return 1;
}

struct Case {
    ninfer::QType qtype;
    const char* python_name;
    const char* label;
};

} // namespace

int main() {
    const std::filesystem::path script_path =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests/helpers/row_split_golden.py";
    const std::filesystem::path output_dir = unique_temp_path("ninfer_row_split_pack_golden_", "");
    std::error_code cleanup_error;

    try {
        std::filesystem::create_directory(output_dir);
        const std::vector<float> source = known_matrix();

        int failures = run_converter(script_path, output_dir);
        if (failures == 0) {
            for (const Case& test_case : {Case{ninfer::QType::Q4G64_F16S, "q4", "Q4G64_F16S"},
                                          Case{ninfer::QType::Q5G64_F16S, "q5", "Q5G64_F16S"},
                                          Case{ninfer::QType::Q6G64_F16S, "q6", "Q6G64_F16S"},
                                          Case{ninfer::QType::W8G32_F16S, "w8g32", "W8G32_F16S"}}) {
                const std::filesystem::path payload_path =
                    output_dir / (std::string(test_case.python_name) + ".bin");
                const std::vector<std::uint8_t> expected = read_bytes(payload_path);
                const ninfer::test::row_split::PackedWeight actual =
                    ninfer::test::row_split::pack_row_split_lowbit(source, kN, kK, test_case.qtype);
                failures += compare_payload(test_case.label, actual.payload, expected);
            }
        }

        std::filesystem::remove_all(output_dir, cleanup_error);
        std::cout << (failures ? "FAIL" : "OK") << " row-split pack golden\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(output_dir, cleanup_error);
        std::cerr << "row-split pack golden: " << e.what() << '\n';
        return 1;
    }
}
