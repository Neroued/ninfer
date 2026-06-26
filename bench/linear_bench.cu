// Performance bench for dense linear at the real Qwen3.6 dense seam shape
// in_a/in_b = [48,5120]. The printed GB/s is informational only; the gate is
// deferred for this correctness task (see docs/plans/l1-tier3-linear-attention.md).
//   ./qus_linear_bench [--decode] [--prefill] [--bf16] [--fp32]
#include "qus/kernels/linear.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kN = 48;
constexpr std::int32_t kK = 5120;

DBuf make_f32(std::size_t n) {
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) { h[i] = 0.25f - static_cast<float>(i % 251) / 1000.0f; }
    DBuf d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

DBuf make_weight(QType qtype, std::size_t n) {
    if (qtype == QType::FP32_CTRL) { return make_f32(n); }
    return make_bf16(n);
}

Weight dense_weight(void* data, QType qtype) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(kN) * kK * ((qtype == QType::FP32_CTRL) ? 4u : 2u);
    w.qtype         = qtype;
    w.layout        = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.group_size        = 0;
    w.shape[0]          = kN;
    w.shape[1]          = kK;
    w.padded_shape[0]   = kN;
    w.padded_shape[1]   = kK;
    w.ndim              = 2;
    w.qdata             = data;
    w.scales            = nullptr;
    w.n                 = kN;
    w.k                 = kK;
    w.group             = 0;
    return w;
}

const char* qtype_name(QType qtype) { return (qtype == QType::FP32_CTRL) ? "fp32" : "bf16"; }

void run(std::int32_t t, QType qtype, const char* phase) {
    const std::size_t x_elems   = static_cast<std::size_t>(kK) * t;
    const std::size_t w_elems   = static_cast<std::size_t>(kN) * kK;
    const std::size_t out_elems = static_cast<std::size_t>(kN) * t;

    DBuf x    = make_bf16(x_elems);
    DBuf wbuf = make_weight(qtype, w_elems);
    DBuf out  = make_zeros(out_elems * sizeof(std::uint16_t));

    Tensor tx(x.p, DType::BF16, {kK, t});
    Tensor tout(out.p, DType::BF16, {kN, t});
    Weight w = dense_weight(wbuf.p, qtype);

    const double weight_bytes = static_cast<double>(w_elems) *
                                ((qtype == QType::FP32_CTRL) ? 4.0 : 2.0) * static_cast<double>(t);
    const double bytes =
        static_cast<double>(x_elems) * 2.0 + weight_bytes + static_cast<double>(out_elems) * 2.0;
    const Result r = bench_loop([&](cudaStream_t s) { kernels::linear(tx, w, tout, s); }, bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "linear %s %s [%d,%d] T=%d", qtype_name(qtype), phase, kN, kK,
                  t);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode = false, prefill = false, bf16 = false, fp32 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--bf16"))
            bf16 = true;
        else if (!std::strcmp(argv[i], "--fp32"))
            fp32 = true;
    }
    if (!decode && !prefill) { decode = prefill = true; }
    if (!bf16 && !fp32) { bf16 = fp32 = true; }

    if (decode && bf16) run(1, QType::BF16_CTRL, "decode");
    if (decode && fp32) run(1, QType::FP32_CTRL, "decode");
    if (prefill && bf16) run(7, QType::BF16_CTRL, "prefill");
    if (prefill && fp32) run(7, QType::FP32_CTRL, "prefill");
    return 0;
}
