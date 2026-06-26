// Performance bench for embed_gather Q6 RowGroupedG64 at the real Qwen3.6
// embedding shape. The printed GB/s is informational only; the gate is ncu
// sustained DRAM percent (see docs/l1-op-test-standard.md section 2).
//   ./qus_embed_gather_bench [--decode] [--t64|--prefill]   (default: both)
#include "qus/kernels/embed_gather.h"
#include "qus/core/device.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kVocab = 248320;
constexpr std::int32_t kD     = 5120;
constexpr std::int32_t kGroup = 64;
constexpr std::int32_t kBpr   = 48;
constexpr std::int32_t kRoww  = 2 + kBpr;
constexpr std::int32_t kKg    = kD / kGroup;

DBuf make_ids(std::int32_t t) {
    std::vector<std::int32_t> h(static_cast<std::size_t>(t));
    for (std::int32_t i = 0; i < t; ++i) {
        h[static_cast<std::size_t>(i)] = (i * 9973 + 12345) % kVocab;
    }
    DBuf d(h.size() * sizeof(std::int32_t));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

Weight q6_weight(void* payload) {
    Weight w{};
    w.payload            = payload;
    w.payload_bytes      = static_cast<std::uint64_t>(kVocab) * kKg * kRoww;
    w.qtype              = QType::Q6G64_F16S;
    w.layout             = QuantLayout::RowGroupedG64;
    w.q5090_scale_dtype  = ScaleDType::FP16;
    w.group_size         = kGroup;
    w.shape[0]           = kVocab;
    w.shape[1]           = kD;
    w.padded_shape[0]    = kVocab;
    w.padded_shape[1]    = kD;
    w.ndim               = 2;
    w.qdata              = payload;
    w.scales             = nullptr;
    w.n                  = kVocab;
    w.k                  = kD;
    w.group              = kGroup;
    return w;
}

void run(std::int32_t t, const char* tag) {
    constexpr std::size_t payload_bytes =
        static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kKg) *
        static_cast<std::size_t>(kRoww);
    const std::size_t out_elems =
        static_cast<std::size_t>(kD) * static_cast<std::size_t>(t);

    DBuf payload(payload_bytes);
    DBuf ids = make_ids(t);
    DBuf out = make_zeros(out_elems * sizeof(std::uint16_t));

    // Deterministic nonzero RowGroupedG64 bytes. Correctness values do not
    // matter for this bandwidth bench; tests cover signed LSB-first semantics.
    CUDA_CHECK(cudaMemset(payload.p, 0x3c, payload.bytes));

    Weight table = q6_weight(payload.p);
    Tensor tids(ids.p, DType::I32, {t});
    Tensor tout(out.p, DType::BF16, {kD, t});

    // Task-defined traffic: Q6 row read plus BF16 output write.
    const double bytes = static_cast<double>(t) * static_cast<double>(kD) * 0.875 +
                         static_cast<double>(t) * static_cast<double>(kD) * 2.0;
    const Result r = bench_loop(
        [&](cudaStream_t s) { kernels::embed_gather(tids, table, tout, s); }, bytes);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode = false, t64 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) decode = true;
        else if (!std::strcmp(argv[i], "--t64") || !std::strcmp(argv[i], "--prefill")) t64 = true;
    }
    if (!decode && !t64) { decode = t64 = true; }

    if (decode) run(1, "embed_gather q6 decode [248320,5120] T=1");
    if (t64) run(64, "embed_gather q6 t64    [248320,5120] T=64");
    return 0;
}
