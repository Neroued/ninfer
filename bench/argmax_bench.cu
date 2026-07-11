// Performance bench for argmax over the real Qwen3.6 vocabulary decode shape.
// The printed GB/s is informational only; the gate is ncu sustained DRAM %
// (see docs/kernel-development.md §8).
//   ./qus_argmax_bench [--decode]   (default: decode)
#include "qus/kernels/argmax.h"
#include "qus/core/device.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kVocab = 248320;
constexpr int kLogitSlots = 256;

void run_decode() {
    DBuf logits =
        make_bf16(static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kLogitSlots));
    DBuf out = make_zeros(sizeof(std::int32_t));

    auto* logits_base = static_cast<std::uint16_t*>(logits.p);
    Tensor tout(out.p, DType::I32, {1});

    const double bytes = static_cast<double>(kVocab) * 2.0;
    int launch = 0;
    const Result r = bench_loop(
        [&](cudaStream_t s) {
            const int slot = launch++ & (kLogitSlots - 1);
            Tensor tlogits(logits_base + static_cast<std::size_t>(slot) *
                                             static_cast<std::size_t>(kVocab),
                           DType::BF16, {kVocab, 1});
            kernels::argmax(tlogits, tout, s);
        },
        bytes);
    print_result("argmax decode logits=[248320,1]", r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) decode = true;
    }
    if (!decode) { decode = true; }

    if (decode) { run_decode(); }
    return 0;
}
