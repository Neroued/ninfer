// Performance bench for linear at real Qwen3.6 shapes. Dense in_a/in_b use
// [48,5120]; Q4 decode uses mlp_gate/mlp_up [17408,5120]. The printed GB/s is
// informational only; the gate is deferred for this correctness task.
//   ./qus_linear_bench [--decode] [--prefill] [--bf16] [--fp32] [--q4]
#include "qus/kernels/linear.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kDenseN = 48;
constexpr std::int32_t kDenseK = 5120;
constexpr std::int32_t kQ4N    = 17408;
constexpr std::int32_t kQ4K    = 5120;
constexpr std::int32_t kQ4Tile = 2176;

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

Weight dense_weight(void* data, QType qtype, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * k * ((qtype == QType::FP32_CTRL) ? 4u : 2u);
    w.qtype         = qtype;
    w.layout        = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.group_size        = 0;
    w.shape[0]          = n;
    w.shape[1]          = k;
    w.padded_shape[0]   = n;
    w.padded_shape[1]   = k;
    w.ndim              = 2;
    w.qdata             = data;
    w.scales            = nullptr;
    w.n                 = n;
    w.k                 = k;
    w.group             = 0;
    return w;
}

const char* qtype_name(QType qtype) { return (qtype == QType::FP32_CTRL) ? "fp32" : "bf16"; }

void run_dense(std::int32_t t, QType qtype, const char* phase) {
    const std::size_t x_elems   = static_cast<std::size_t>(kDenseK) * t;
    const std::size_t w_elems   = static_cast<std::size_t>(kDenseN) * kDenseK;
    const std::size_t out_elems = static_cast<std::size_t>(kDenseN) * t;

    DBuf x    = make_bf16(x_elems);
    DBuf wbuf = make_weight(qtype, w_elems);
    DBuf out  = make_zeros(out_elems * sizeof(std::uint16_t));

    Tensor tx(x.p, DType::BF16, {kDenseK, t});
    Tensor tout(out.p, DType::BF16, {kDenseN, t});
    Weight w = dense_weight(wbuf.p, qtype, kDenseN, kDenseK);

    const double weight_bytes = static_cast<double>(w_elems) *
                                ((qtype == QType::FP32_CTRL) ? 4.0 : 2.0) * static_cast<double>(t);
    const double bytes =
        static_cast<double>(x_elems) * 2.0 + weight_bytes + static_cast<double>(out_elems) * 2.0;
    const Result r = bench_loop([&](cudaStream_t s) { kernels::linear(tx, w, tout, s); }, bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "linear %s %s [%d,%d] T=%d", qtype_name(qtype), phase, kDenseN,
                  kDenseK, t);
    print_result(tag, r);
}

DBuf make_q4_payload() {
    constexpr std::uint16_t kScaleOne = 0x3c00u;
    const std::int32_t nt             = kQ4N / 64;
    const std::int32_t kg             = kQ4K / 64;
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(nt) * kg * kQ4Tile);
    for (std::int32_t tile = 0; tile < nt; ++tile) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t base =
                (static_cast<std::size_t>(tile) * kg + g) * static_cast<std::size_t>(kQ4Tile);
            for (std::int32_t row = 0; row < 64; ++row) {
                payload[base + static_cast<std::size_t>(row) * 2] =
                    static_cast<std::uint8_t>(kScaleOne & 0xffu);
                payload[base + static_cast<std::size_t>(row) * 2 + 1] =
                    static_cast<std::uint8_t>(kScaleOne >> 8);
            }
            for (std::int32_t i = 64 * 2; i < kQ4Tile; ++i) {
                payload[base + i] = static_cast<std::uint8_t>((i + tile + g) & 0xff);
            }
        }
    }
    DBuf d(payload.size());
    cudaMemcpy(d.p, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    return d;
}

Weight q4_weight(void* payload) {
    Weight w{};
    w.payload           = payload;
    w.payload_bytes     = static_cast<std::uint64_t>(kQ4N / 64) * (kQ4K / 64) * kQ4Tile;
    w.qtype             = QType::Q4G64_F16S;
    w.layout            = QuantLayout::TileN64K64;
    w.q5090_scale_dtype = ScaleDType::FP16;
    w.group_size        = 64;
    w.shape[0]          = kQ4N;
    w.shape[1]          = kQ4K;
    w.padded_shape[0]   = kQ4N;
    w.padded_shape[1]   = kQ4K;
    w.ndim              = 2;
    w.qdata             = payload;
    w.scales            = nullptr;
    w.n                 = kQ4N;
    w.k                 = kQ4K;
    w.group             = 64;
    return w;
}

void run_q4_decode() {
    constexpr std::int32_t t    = 1;
    const std::size_t x_elems   = static_cast<std::size_t>(kQ4K) * t;
    const std::size_t out_elems = static_cast<std::size_t>(kQ4N) * t;

    DBuf x    = make_bf16(x_elems);
    DBuf wbuf = make_q4_payload();
    DBuf out  = make_zeros(out_elems * sizeof(std::uint16_t));

    Tensor tx(x.p, DType::BF16, {kQ4K, t});
    Tensor tout(out.p, DType::BF16, {kQ4N, t});
    Weight w = q4_weight(wbuf.p);

    const double bytes = static_cast<double>(x_elems) * 2.0 + static_cast<double>(w.payload_bytes) +
                         static_cast<double>(out_elems) * 2.0;
    const Result r = bench_loop([&](cudaStream_t s) { kernels::linear(tx, w, tout, s); }, bytes);
    print_result("linear q4 decode [17408,5120] T=1", r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode = false, prefill = false, bf16 = false, fp32 = false, q4 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--bf16"))
            bf16 = true;
        else if (!std::strcmp(argv[i], "--fp32"))
            fp32 = true;
        else if (!std::strcmp(argv[i], "--q4"))
            q4 = true;
    }
    if (!decode && !prefill) { decode = prefill = true; }
    if (!bf16 && !fp32 && !q4) { bf16 = fp32 = q4 = true; }

    if (decode && bf16) run_dense(1, QType::BF16_CTRL, "decode");
    if (decode && fp32) run_dense(1, QType::FP32_CTRL, "decode");
    if (prefill && bf16) run_dense(7, QType::BF16_CTRL, "prefill");
    if (prefill && fp32) run_dense(7, QType::FP32_CTRL, "prefill");
    if (decode && q4) run_q4_decode();
    return 0;
}
