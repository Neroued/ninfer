// Performance bench for partial NeoX RoPE at the real Qwen3.6-27B attention
// q/k shapes. The printed GB/s is informational only; the gate is ncu sustained
// DRAM % (see docs/kernel-development.md §8).
//   ./qus_rope_bench [--decode] [--prefill]   (default: both)
#include "qus/kernels/rope.h"
#include "qus/core/device.h"
#include "qus_bench_common.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr int kHeadDim   = 256;
constexpr int kQHeads    = 24;
constexpr int kKHeads    = 4;
constexpr int kRotaryDim = 64;
constexpr float kTheta   = 1.0e7f;

DBuf make_positions(int t) {
    std::vector<std::int32_t> h(static_cast<std::size_t>(t));
    for (int i = 0; i < t; ++i) h[static_cast<std::size_t>(i)] = i;
    DBuf d(h.size() * sizeof(std::int32_t));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

DBuf make_positions(int t, int axes) {
    std::vector<std::int32_t> h(static_cast<std::size_t>(t) * axes);
    for (int axis = 0; axis < axes; ++axis) {
        for (int i = 0; i < t; ++i) { h[static_cast<std::size_t>(axis) * t + i] = i + axis * 17; }
    }
    DBuf d(h.size() * sizeof(std::int32_t));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

__device__ __forceinline__ void copy_vec8(__nv_bfloat16* data, std::int64_t idx) {
    void* ptr = data + idx;
    unsigned int x, y, z, w;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(ptr)
                 : "memory");
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}

__global__ void rope_copy_baseline_kernel(__nv_bfloat16* q, __nv_bfloat16* k, std::int64_t q_vecs,
                                          std::int64_t total_vecs) {
    constexpr int kVecsPerThread = 4;
    const std::int64_t tid       = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x * kVecsPerThread;
    for (std::int64_t v = tid * kVecsPerThread; v < total_vecs; v += stride) {
        for (int i = 0; i < kVecsPerThread; ++i) {
            const std::int64_t idx = v + i;
            if (idx >= total_vecs) { break; }
            if (idx < q_vecs) {
                copy_vec8(q, idx * 8);
            } else {
                copy_vec8(k, (idx - q_vecs) * 8);
            }
        }
    }
}

void launch_copy_baseline(DBuf& q, DBuf& k, std::size_t qn, std::size_t kn, cudaStream_t stream) {
    constexpr int kBlock  = 256;
    const auto q_vecs     = static_cast<std::int64_t>(qn / 8);
    const auto total_vecs = static_cast<std::int64_t>((qn + kn) / 8);
    const auto blocks     = (total_vecs + kBlock * 4 - 1) / (kBlock * 4);
    const int grid        = static_cast<int>(
        std::min<std::int64_t>(blocks, static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    rope_copy_baseline_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q.p), static_cast<__nv_bfloat16*>(k.p), q_vecs, total_vecs);
    CUDA_CHECK(cudaGetLastError());
}

void run(int t, const char* tag, bool copy_baseline) {
    const auto qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                    static_cast<std::size_t>(t);
    const auto kn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kKHeads) *
                    static_cast<std::size_t>(t);

    DBuf positions = make_positions(t);
    DBuf q         = make_bf16(qn);
    DBuf k         = make_bf16(kn);

    Tensor tpos(positions.p, DType::I32, {t});
    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, t});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKHeads, t});

    // Task-defined HBM traffic: q/k read and written in place. Positions are
    // tiny by comparison and intentionally excluded from this byte count.
    const double bytes = 2.0 * static_cast<double>(qn + kn) * 2.0;
    const Result r =
        copy_baseline
            ? bench_loop([&](cudaStream_t s) { launch_copy_baseline(q, k, qn, kn, s); }, bytes)
            : bench_loop(
                  [&](cudaStream_t s) { kernels::rope(tpos, kRotaryDim, kTheta, tq, tk, s); },
                  bytes);
    print_result(tag, r);
}

void run_mrope(int t) {
    const auto qn  = static_cast<std::size_t>(kHeadDim) * kQHeads * t;
    const auto kn  = static_cast<std::size_t>(kHeadDim) * kKHeads * t;
    DBuf positions = make_positions(t, 3);
    DBuf q         = make_bf16(qn);
    DBuf k         = make_bf16(kn);
    Tensor tpos(positions.p, DType::I32, {t, 3});
    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, t});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKHeads, t});
    const double bytes  = 4.0 * static_cast<double>(qn + kn);
    const Result result = bench_loop(
        [&](cudaStream_t stream) { kernels::rope(tpos, kRotaryDim, kTheta, tq, tk, stream); },
        bytes);
    print_result("rope MRoPE [3,4096]", result);
}

void run_vision(int patches) {
    constexpr int head_dim     = 72;
    constexpr int heads        = 16;
    constexpr int hidden       = head_dim * heads;
    constexpr int qkv          = hidden * 3;
    const std::size_t elements = static_cast<std::size_t>(patches) * qkv;
    DBuf positions             = make_positions(patches, 2);
    DBuf packed                = make_bf16(elements);
    Tensor tq(packed.p, DType::BF16, {head_dim, heads, patches});
    tq.nb[2]  = qkv * 2;
    Tensor tk = tq;
    tk.data   = static_cast<unsigned char*>(packed.p) + hidden * 2;
    Tensor tpos(positions.p, DType::I32, {patches, 2});
    const double bytes  = 8.0 * static_cast<double>(hidden) * patches;
    const Result result = bench_loop(
        [&](cudaStream_t stream) { kernels::rope(tpos, head_dim, 10000.0f, tq, tk, stream); },
        bytes);
    print_result("rope Vision packed P=4096", result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool prefill = false, decode = false, copy_baseline = false, mrope = false, vision = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else if (!std::strcmp(argv[i], "--copy-baseline"))
            copy_baseline = true;
        else if (!std::strcmp(argv[i], "--mrope"))
            mrope = true;
        else if (!std::strcmp(argv[i], "--vision"))
            vision = true;
    }
    if (!prefill && !decode && !mrope && !vision) { prefill = decode = true; }

    if (decode) {
        run(1,
            copy_baseline ? "rope copy baseline decode  q=[256,24,1] k=[256,4,1]"
                          : "rope decode  q=[256,24,1] k=[256,4,1]",
            copy_baseline);
    }
    if (prefill) {
        run(4096,
            copy_baseline ? "rope copy baseline prefill q=[256,24,4096] k=[256,4,4096]"
                          : "rope prefill q=[256,24,4096] k=[256,4,4096]",
            copy_baseline);
    }
    if (mrope) run_mrope(4096);
    if (vision) run_vision(4096);
    return 0;
}
