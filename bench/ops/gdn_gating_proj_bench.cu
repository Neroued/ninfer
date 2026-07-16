// Performance bench for the Qwen3.6-27B fused GDN in_a/in_b prefill path:
// two contiguous BF16 [48,5120] projections fused with gdn_gating.
//   ./ninfer_gdn_gating_proj_bench -p 128,256,512,1024,2048,4096,8192,16384
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kHeads        = 48;
constexpr std::int32_t kSmallTMax    = 8;
constexpr std::int32_t kSmallTSplits = 10;

int bf16_prefill_split_k(std::int32_t tokens) {
    if (tokens <= 128) { return 40; }
    if (tokens <= 256) { return 20; }
    if (tokens <= 512) { return 10; }
    if (tokens <= 1024) { return 8; }
    if (tokens <= 2048) { return 4; }
    if (tokens <= 4096) { return 2; }
    return 1;
}

DBuf make_f32(std::size_t n, std::uint32_t seed) {
    std::vector<float> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = 2.0f * u - 1.0f;
    }
    DBuf d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

Weight bf16_weight(void* data) {
    Weight w{};
    w.qtype   = QType::BF16_CTRL;
    w.layout  = QuantLayout::Contiguous;
    w.payload = data;
    w.payload_bytes =
        static_cast<std::uint64_t>(kHeads) * static_cast<std::uint64_t>(kHidden) * 2ULL;
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
        char* end    = nullptr;
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
    WorkspaceArena ws(64u * 1024u * 1024u);

    Tensor tx(x.p, DType::BF16, {kHidden, T});
    Tensor tA_log(A_log.p, DType::FP32, {kHeads});
    Tensor tdt_bias(dt_bias.p, DType::FP32, {kHeads});
    Tensor tg(g.p, DType::FP32, {kHeads, T});
    Tensor tbeta(beta.p, DType::FP32, {kHeads, T});
    Weight wa = bf16_weight(aw.p);
    Weight wb = bf16_weight(bw.p);

    const int split_k = (T >= 2 && T <= kSmallTMax) ? kSmallTSplits : bf16_prefill_split_k(T);
    const char* route = (T == 1)            ? "decode-row"
                        : (T <= kSmallTMax) ? "smallt-splitk"
                        : (split_k > 1)     ? "mma-coop-splitk"
                                            : "mma-prefill";
    const double weight_bytes  = 2.0 * static_cast<double>(w_elems) * sizeof(std::uint16_t);
    const double x_bytes       = static_cast<double>(x_elems) * sizeof(std::uint16_t);
    const double out_bytes     = 2.0 * static_cast<double>(out_elems) * sizeof(float);
    const double scratch_bytes = (T >= 2 && split_k > 1)
                                     ? 2.0 * static_cast<double>(split_k) * static_cast<double>(T) *
                                           static_cast<double>(2 * kHeads) * sizeof(float)
                                     : 0.0;
    const double useful_bytes  = weight_bytes + x_bytes + out_bytes;
    const double bench_bytes   = useful_bytes + scratch_bytes;

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            ops::gdn_gating_proj(tx, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, s);
        },
        bench_bytes, warmup, repeat, min_time_ms);

    const double sec        = r.median_us * 1e-6;
    const double useful_gbs = (sec > 0.0) ? useful_bytes / sec / 1e9 : 0.0;
    const double bench_gbs  = (sec > 0.0) ? bench_bytes / sec / 1e9 : 0.0;
    const double roof_pct   = useful_gbs / kRooflineGBs * 100.0;
    std::printf("fused,%d,%s,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.0f,%d,%d\n", T, route, r.median_us,
                r.min_us, r.p95_us, useful_gbs, bench_gbs, roof_pct, scratch_bytes, r.n_runs,
                r.inner_iters);
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

    std::printf("mode,T,route,median_us,min_us,p95_us,useful_gbps,bench_gbps,roof_pct,"
                "scratch_bytes,n_runs,inner_iters\n");
    for (std::int32_t T : tokens) { run(T, warmup, repeat, min_time_ms); }
    return 0;
}
