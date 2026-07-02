// Performance bench for the Qwen3.6-27B fused GDN in_a/in_b prefill path:
// two dense BF16 [48,5120] projections fused with gdn_gating.
//   ./qus_gdn_in_ab_prefill_bench -p 128,256,512,1024,2048,4096,8192,16384
#include "qus/kernels/gdn_in_ab.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kHeads  = 48;
constexpr double kTcPeakTflops = 220.0;

DBuf make_f32(std::size_t n, std::uint32_t seed) {
    std::vector<float> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i] = 2.0f * u - 1.0f;
    }
    DBuf d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

Weight dense_bf16_weight(void* data) {
    Weight w{};
    w.qtype             = QType::BF16_CTRL;
    w.layout            = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.payload           = data;
    w.payload_bytes     = static_cast<std::uint64_t>(kHeads) * static_cast<std::uint64_t>(kHidden) *
                      2ULL;
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = kHeads;
    w.shape[1]        = kHidden;
    w.padded_shape[0] = kHeads;
    w.padded_shape[1] = kHidden;
    w.n               = kHeads;
    w.k               = kHidden;
    return w;
}

std::vector<std::int32_t> parse_tokens(const char* csv) {
    std::vector<std::int32_t> out;
    const char* p = csv;
    while (*p != '\0') {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 1'000'000) {
            std::fprintf(stderr, "invalid token length list: %s\n", csv);
            std::exit(2);
        }
        out.push_back(static_cast<std::int32_t>(v));
        p = (*end == ',') ? end + 1 : end;
    }
    return out;
}

void run(std::int32_t T, int warmup, int repeat, int min_time_ms) {
    const std::size_t x_elems   = static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(T);
    const std::size_t w_elems   = static_cast<std::size_t>(kHeads) * kHidden;
    const std::size_t out_elems = static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T);

    DBuf x       = make_bf16(x_elems);
    DBuf aw      = make_bf16(w_elems);
    DBuf bw      = make_bf16(w_elems);
    DBuf A_log   = make_f32(kHeads, 0x1234abcdU);
    DBuf dt_bias = make_f32(kHeads, 0x9876fedcU);
    DBuf g       = make_zeros(out_elems * sizeof(float));
    DBuf beta    = make_zeros(out_elems * sizeof(float));

    Tensor tx(x.p, DType::BF16, {kHidden, T});
    Tensor tA_log(A_log.p, DType::FP32, {kHeads});
    Tensor tdt_bias(dt_bias.p, DType::FP32, {kHeads});
    Tensor tg(g.p, DType::FP32, {kHeads, T});
    Tensor tbeta(beta.p, DType::FP32, {kHeads, T});
    Weight wa = dense_bf16_weight(aw.p);
    Weight wb = dense_bf16_weight(bw.p);

    const double useful_flops =
        4.0 * static_cast<double>(kHeads) * static_cast<double>(kHidden) *
        static_cast<double>(T);
    const double bytes =
        2.0 * static_cast<double>(x_elems) * 2.0 +
        2.0 * static_cast<double>(w_elems) * 2.0 * static_cast<double>(T) +
        2.0 * static_cast<double>(out_elems) * 2.0 +
        2.0 * static_cast<double>(out_elems) * 2.0 +
        2.0 * static_cast<double>(out_elems) * 4.0;

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            kernels::gdn_in_ab_gated_prefill(tx, wa, wb, tA_log, tdt_bias, tg, tbeta, s);
        },
        bytes, warmup, repeat, min_time_ms);

    const double sec          = r.median_us * 1e-6;
    const double useful_tflop = (sec > 0.0) ? useful_flops / sec / 1e12 : 0.0;
    const double tc_pct       = useful_tflop / kTcPeakTflops * 100.0;
    std::printf("fused,%d,%.3f,%.3f,%.3f,%.4f,%.2f,%d,%d\n", T, r.median_us, r.min_us,
                r.p95_us, useful_tflop, tc_pct, r.n_runs, r.inner_iters);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<std::int32_t> tokens = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int warmup                       = 20;
    int repeat                       = 100;
    int min_time_ms                  = 500;

    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-p") || !std::strcmp(argv[i], "--tokens")) && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--min-ms") && i + 1 < argc) {
            min_time_ms = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "usage: %s [-p 128,256,...] [--warmup N] [--repeat N] [--min-ms N]\n",
                         argv[0]);
            return 2;
        }
    }

    std::printf("mode,T,median_us,min_us,p95_us,useful_tflops,tc_peak_pct,n_runs,inner_iters\n");
    for (std::int32_t T : tokens) { run(T, warmup, repeat, min_time_ms); }
    return 0;
}
