#pragma once
//
// op_tester.h — device buffers, seeded honest input generators, and the
// fp64-golden verify step for L1 op correctness tests. Pairs with op_check.h
// (the frozen tolerance presets). See docs/kernel-development.md.
//
// Typical per-op test flow (one shape):
//   std::vector<float> x(n); fill_uniform(x, seed, -8.f, 8.f);
//   round_to_bf16(x);                       // what the kernel will read
//   std::vector<double> ref(n);             // compute in double from x
//   ... cpu_ref(x, ref) ...
//   DBuf dx = to_device_bf16(x), dout(n*2);
//   ... launch kernel on dx -> dout ...
//   failures += verify("op n=...", from_device_bf16(dout, n), ref,
//                      Tolerance::bf16_elementwise());

#include "core/tensor.h"   // ninfer::DType, ninfer::Tensor (for op call sites)
#include "kernels/op_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace ninfer::test {

// --- environment ------------------------------------------------------------
inline bool cuda_unavailable() {
    int n               = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || e != cudaSuccess || n == 0;
}

// --- bf16 <-> f32 (round-to-nearest-even) -----------------------------------
inline float bf16_to_f32(std::uint16_t h) {
    std::uint32_t u = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((u >> 16) | 0x0040u);  // keep NaN
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}
// Round an fp32 buffer to the bf16 grid in place, so a CPU reference computes
// from EXACTLY the values the kernel will read.
inline void round_to_bf16(std::vector<float>& v) {
    for (auto& x : v) x = bf16_to_f32(f32_to_bf16(x));
}

// --- device buffers (RAII) --------------------------------------------------
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

// --- seeded honest inputs ---------------------------------------------------
inline void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo, float hi) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v) x = d(g);
}
inline void fill_iota_i32(std::vector<int>& v, int start = 0) {
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = start + static_cast<int>(i);
}
// L2-normalize each contiguous `d`-element row (matches RMSNorm/F.normalize
// upstream of GDN q/k; keeps recurrence test inputs from overflowing).
inline void l2_normalize_rows(std::vector<float>& v, int d, long long rows) {
    constexpr double eps = 1e-12;
    for (long long r = 0; r < rows; ++r) {
        float* row   = v.data() + r * d;
        double sumsq = 0.0;
        for (int i = 0; i < d; ++i) sumsq += double(row[i]) * double(row[i]);
        const float inv = float(1.0 / std::sqrt(sumsq + eps));
        for (int i = 0; i < d; ++i) row[i] *= inv;
    }
}

// --- host -> device ---------------------------------------------------------
inline DBuf to_device_bf16(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    DBuf d(h.size() * 2);
    cudaMemcpy(d.p, b.data(), h.size() * 2, cudaMemcpyHostToDevice);
    return d;
}
inline DBuf to_device_f32(const std::vector<float>& h) {
    DBuf d(h.size() * 4);
    cudaMemcpy(d.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    return d;
}
inline DBuf to_device_i32(const std::vector<int>& h) {
    DBuf d(h.size() * 4);
    cudaMemcpy(d.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    return d;
}

// --- device -> host (upcast to double for the comparison) -------------------
inline std::vector<double> from_device_bf16(const DBuf& d, std::size_t n) {
    std::vector<std::uint16_t> b(n);
    cudaMemcpy(b.data(), d.p, n * 2, cudaMemcpyDeviceToHost);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(bf16_to_f32(b[i]));
    return o;
}
inline std::vector<double> from_device_f32(const DBuf& d, std::size_t n) {
    std::vector<float> f(n);
    cudaMemcpy(f.data(), d.p, n * 4, cudaMemcpyDeviceToHost);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(f[i]);
    return o;
}
inline std::vector<int> from_device_i32(const DBuf& d, std::size_t n) {
    std::vector<int> o(n);
    cudaMemcpy(o.data(), d.p, n * 4, cudaMemcpyDeviceToHost);
    return o;
}

// --- verdict ----------------------------------------------------------------
// Returns 0 (pass) / 1 (fail). Prints the diff line with the chosen preset.
inline int verify(const char* label, const std::vector<double>& got,
                  const std::vector<double>& ref, const Tolerance& tol) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size() << '\n';
        return 1;
    }
    const DiffStats s = compute_diff(got.data(), ref.data(), (long long) got.size(), tol);
    print_diff(label, s, tol);
    const bool ok = diff_passes(s, tol);
    if (!ok) std::cerr << label << ": FAIL (see diff above)\n";
    return ok ? 0 : 1;
}

} // namespace ninfer::test
