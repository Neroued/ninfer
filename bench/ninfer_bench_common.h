#pragma once
//
// ninfer_bench_common.h — shared bench harness for L1 op performance binaries.
//
// Adapted from ~/chunked_gdn/bench/bench_common.h. Timing uses CUDA events
// with inner-iter batching to amortize the per-sample host sync; throughput is
// reported from the MEDIAN per-launch time (robust to host scheduling spikes).
//
// IMPORTANT (docs/kernel-development.md §8): the GB/s printed here is a
// convenience readout, NOT the acceptance gate. The gate is ncu
// dram__throughput.avg.pct_of_peak_sustained_elapsed >= 85%. Always profile.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace ninfer::bench {

constexpr double kRooflineGBs = 1792.0;  // RTX 5090 GDDR7 bandwidth roofline.

struct DBuf {
    void* p           = nullptr;
    std::size_t bytes = 0;
    explicit DBuf(std::size_t b) : bytes(b) { cudaMalloc(&p, b); }
    ~DBuf() { if (p) cudaFree(p); }
    DBuf(DBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
    DBuf& operator=(DBuf&& o) noexcept {
        if (this != &o) {
            if (p) cudaFree(p);
            p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0;
        }
        return *this;
    }
    DBuf(const DBuf&)            = delete;
    DBuf& operator=(const DBuf&) = delete;
};

inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}

// Device bf16 buffer filled with a small varied ramp (avoids all-zero special
// paths; exact values are irrelevant to bandwidth). Returns an owning DBuf.
inline DBuf make_bf16(std::size_t n) {
    std::vector<std::uint16_t> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(0.5f - float(i % 251) / 250.0f);
    DBuf d(n * 2);
    cudaMemcpy(d.p, h.data(), n * 2, cudaMemcpyHostToDevice);
    return d;
}
inline DBuf make_zeros(std::size_t bytes) {
    DBuf d(bytes);
    cudaMemset(d.p, 0, bytes);
    return d;
}

// cudaDeviceProp memory-clock fields were removed in CUDA 13; the in-process
// GB/s is informational anyway (ncu is the acceptance gate), so report against
// the known RTX 5090 roofline constant.
inline double device_peak_bw_gbs(int /*dev*/ = 0) { return kRooflineGBs; }

struct Result {
    int n_runs      = 0;
    int inner_iters = 1;
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
    double mean_us   = 0.0;
    double gbs       = 0.0;  // from median
};

using launch_fn = std::function<void(cudaStream_t)>;

inline Result bench_loop(const launch_fn& launch, double bytes_moved, int warmup = 20,
                         int repeat = 100, int min_time_ms = 500) {
    cudaStream_t stream = nullptr;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    for (int i = 0; i < warmup; ++i) launch(stream);
    cudaStreamSynchronize(stream);

    // Auto-size inner_iters so each timed batch is ~500us (amortize sync wait).
    int inner = 0;
    {
        constexpr int probe = 4;
        cudaEventRecord(a, stream);
        for (int i = 0; i < probe; ++i) launch(stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        const double per_us = double(ms) * 1000.0 / probe;
        inner = std::max(1, std::min(1024, int(std::ceil(500.0 / std::max(per_us, 1.0)))));
    }

    std::vector<double> samples;
    long long total_us = 0;
    while (int(samples.size()) < repeat || total_us < (long long) min_time_ms * 1000) {
        cudaEventRecord(a, stream);
        for (int i = 0; i < inner; ++i) launch(stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        const double batch_us = double(ms) * 1000.0;
        samples.push_back(batch_us / inner);
        total_us += (long long) batch_us;
        if (samples.size() > 100000) break;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        const std::size_t idx = std::min(sorted.size() - 1, std::size_t(q * sorted.size()));
        return sorted[idx];
    };
    double sum = 0.0;
    for (double v : samples) sum += v;

    Result r;
    r.n_runs      = int(samples.size());
    r.inner_iters = inner;
    r.median_us   = pct(0.50);
    r.min_us      = sorted.front();
    r.p95_us      = pct(0.95);
    r.mean_us     = sum / samples.size();
    const double sec = r.median_us * 1e-6;
    r.gbs            = (sec > 0.0) ? bytes_moved / sec / 1e9 : 0.0;
    return r;
}

inline void print_result(const char* tag, const Result& r) {
    std::printf("%-32s median=%8.2f us  min=%8.2f us  p95=%8.2f us  %8.1f GB/s  (%.1f%% of %.0f GB/s "
                "roofline)\n",
                tag, r.median_us, r.min_us, r.p95_us, r.gbs, r.gbs / kRooflineGBs * 100.0,
                kRooflineGBs);
}

} // namespace ninfer::bench
