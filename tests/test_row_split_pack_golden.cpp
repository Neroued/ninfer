#include "kernels/row_split_pack.h"

#include <cuda_runtime.h>

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
    std::string out = "'";
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
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

void write_python_script(const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to create converter golden script"); }
    out << R"PY(
import sys
from pathlib import Path

import torch

source_dir = Path(sys.argv[1])
qtype_name = sys.argv[2]
out_path = Path(sys.argv[3])
sys.path.insert(0, str(source_dir))

from tools.artifact.layouts import row_split_geometry
from tools.convert.common.quantize import quantize_and_encode

N = 70
K = 130

def known_matrix():
    w = torch.empty((N, K), dtype=torch.float32)
    for row in range(N):
        for col in range(K):
            base = ((row * 37 + col * 17) % 211) - 105
            sign = -1.0 if ((row + col) & 1) else 1.0
            value = sign * (base / 37.0) + ((row % 5) - 2) * 0.03125
            w[row, col] = value
    return w

formats = {
    "q4": "Q4G64_F16S",
    "q5": "Q5G64_F16S",
    "q6": "Q6G64_F16S",
    "w8g32": "W8G32_F16S",
}

expected_group = 32 if qtype_name == "w8g32" else 64
geometry = row_split_geometry(formats[qtype_name], (N, K))
if geometry.n != N or geometry.k != K or geometry.k_pad != 256 or 256 // geometry.groups_per_row != expected_group:
    raise SystemExit(
        f"unexpected row-split geometry: {geometry}"
    )

out_path.write_bytes(
    quantize_and_encode(known_matrix(), formats[qtype_name], device="cpu")
)
)PY";
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open converter golden payload"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

int run_converter(const std::filesystem::path& script, std::string_view qtype_name,
                  const std::filesystem::path& out_path) {
    const std::string command = shell_quote(NINFER_PYTHON_EXECUTABLE) + " " +
                                shell_quote(script.string()) + " " +
                                shell_quote(NINFER_SOURCE_DIR) + " " +
                                shell_quote(qtype_name) + " " +
                                shell_quote(out_path.string());
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        std::cerr << "converter golden command failed for " << qtype_name << ": " << command
                  << '\n';
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

void touch_cuda_runtime_for_sanitizer() {
    // This byte-golden test is host-only, but the sanitizer gate expects a CUDA API call.
    int device_count = 0;
    (void)cudaGetDeviceCount(&device_count);
}

} // namespace

int main() {
    touch_cuda_runtime_for_sanitizer();

    const std::filesystem::path script_path = unique_temp_path("ninfer_row_split_pack_golden_", ".py");
    std::error_code cleanup_error;

    try {
        write_python_script(script_path);
        const std::vector<float> source = known_matrix();

        int failures = 0;
        for (const Case& test_case : {Case{ninfer::QType::Q4G64_F16S, "q4", "Q4G64_F16S"},
                                      Case{ninfer::QType::Q5G64_F16S, "q5", "Q5G64_F16S"},
                                      Case{ninfer::QType::Q6G64_F16S, "q6", "Q6G64_F16S"},
                                      Case{ninfer::QType::W8G32_F16S, "w8g32",
                                           "W8G32_F16S"}}) {
            const std::filesystem::path payload_path = unique_temp_path(
                "ninfer_row_split_pack_golden_" + std::string(test_case.python_name) + "_", ".bin");
            int case_failures = run_converter(script_path, test_case.python_name, payload_path);
            if (case_failures == 0) {
                const std::vector<std::uint8_t> expected = read_bytes(payload_path);
                const ninfer::test::row_split::PackedWeight actual =
                    ninfer::test::row_split::pack_row_split_lowbit(source, kN, kK, test_case.qtype);
                case_failures += compare_payload(test_case.label, actual.payload, expected);
            }
            failures += case_failures;
            std::filesystem::remove(payload_path, cleanup_error);
        }

        std::filesystem::remove(script_path, cleanup_error);
        std::cout << (failures ? "FAIL" : "OK") << " row-split pack golden\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(script_path, cleanup_error);
        std::cerr << "row-split pack golden: " << e.what() << '\n';
        return 1;
    }
}
