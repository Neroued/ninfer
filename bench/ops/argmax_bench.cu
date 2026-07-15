// Qualification benchmark for the Qwen3.6-35B G1 argmax domains.
//
//   ./ninfer_argmax_bench
//   ./ninfer_argmax_bench --shape full --cols 1
//   ./ninfer_argmax_bench --shape shortlist --cols 1 --control
#include "ninfer/ops/argmax.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kFullPhysicalRows = 248320;
constexpr std::int32_t kFullValidRows    = 248077;
constexpr std::int32_t kShortlistRows    = 131072;
constexpr int kControlBlock              = 512;
constexpr int kWindowStride              = 8;
constexpr int kWindowCount               = 32;
constexpr int kLogitSlots                = kWindowStride * kWindowCount;

struct Options {
    std::string shape;
    int cols     = 0;
    bool control = false;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--shape full|shortlist --cols N] [--control]\n", argv0);
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--shape") {
            options.shape = need_value("--shape");
        } else if (arg == "--cols") {
            options.cols = parse_int(need_value("--cols"), "--cols");
        } else if (arg == "--control") {
            options.control = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_argmax_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.shape.empty()) {
        if (options.cols != 0) { throw std::invalid_argument("--cols requires --shape"); }
        return options;
    }
    if (options.shape != "full" && options.shape != "shortlist") {
        throw std::invalid_argument("--shape must be full or shortlist");
    }
    if (options.cols == 0) { options.cols = 1; }
    if (options.cols < 1 || options.cols > 6) {
        throw std::invalid_argument("--cols must be in [1,6]");
    }
    if (options.shape == "shortlist" && options.cols != 1) {
        throw std::invalid_argument("shortlist supports exactly --cols 1");
    }
    return options;
}

__global__ void argmax_payload_control_kernel(const std::uint16_t* logits, std::uint32_t* out,
                                              std::int32_t valid_rows, std::int32_t physical_rows) {
    const int col = static_cast<int>(blockIdx.y);
    const int v   = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint32_t value =
        v < valid_rows ? logits[static_cast<std::int64_t>(col) * physical_rows + v] : 0u;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value ^= __shfl_down_sync(0xffffffffu, value, offset);
    }
    __shared__ std::uint32_t warp_values[kControlBlock / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_values[warp] = value; }
    __syncthreads();
    if (warp != 0) { return; }
    value = lane < kControlBlock / 32 ? warp_values[lane] : 0u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value ^= __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) { atomicXor(out + col, value); }
}

void run_shape(std::int32_t physical_rows, std::int32_t valid_rows, int cols, bool control,
               const char* shape) {
    DBuf logits       = make_bf16(static_cast<std::size_t>(physical_rows) * kLogitSlots);
    DBuf out          = make_zeros(static_cast<std::size_t>(cols) * sizeof(std::int32_t));
    auto* logits_base = static_cast<std::uint16_t*>(logits.p);
    Tensor tout(out.p, DType::I32, {cols});

    const double bytes  = static_cast<double>(valid_rows) * 2.0 * static_cast<double>(cols);
    int launch          = 0;
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            const int slot = (launch++ & (kWindowCount - 1)) * kWindowStride;
            auto* window   = logits_base + static_cast<std::size_t>(slot) * physical_rows;
            if (control) {
                CUDA_CHECK(cudaMemsetAsync(
                    out.p, 0, static_cast<std::size_t>(cols) * sizeof(std::int32_t), stream));
                const int blocks = (valid_rows + kControlBlock - 1) / kControlBlock;
                const dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(cols));
                argmax_payload_control_kernel<<<grid, kControlBlock, 0, stream>>>(
                    window, static_cast<std::uint32_t*>(out.p), valid_rows, physical_rows);
                CUDA_CHECK(cudaGetLastError());
            } else {
                Tensor tlogits(window, DType::BF16, {physical_rows, cols});
                ops::argmax(tlogits, tout, valid_rows, stream);
            }
        },
        bytes);

    const std::string label = std::string(control ? "argmax payload control " : "argmax ") + shape +
                              " rows=" + std::to_string(valid_rows) + " C=" + std::to_string(cols);
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        if (options.shape.empty()) {
            for (int cols = 1; cols <= 6; ++cols) {
                run_shape(kFullPhysicalRows, kFullValidRows, cols, options.control, "full");
            }
            run_shape(kShortlistRows, kShortlistRows, 1, options.control, "shortlist");
        } else if (options.shape == "full") {
            run_shape(kFullPhysicalRows, kFullValidRows, options.cols, options.control, "full");
        } else {
            run_shape(kShortlistRows, kShortlistRows, 1, options.control, "shortlist");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_argmax_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
