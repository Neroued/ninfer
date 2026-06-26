// Performance bench for gated_delta_rule_recurrent at the real Qwen3.6-27B
// decode shape ([S,Hqk,Hv,T] = [128,16,48,1]). The public-wrapper row is the
// catalog API timing. The kernel-only-fp32 row is an apples-to-apples diagnostic
// against ~/chunked_gdn/bench_ar, which also times preallocated fp32 buffers.
//   ./qus_gated_delta_rule_bench --decode
//   ./qus_gated_delta_rule_bench --decode --kernel-only
#include "qus/kernels/gated_delta_rule.h"
#include "kernels/launcher/gated_delta_rule.h"
#include "qus_bench_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr int kS       = 128;
constexpr int kHqk     = 16;
constexpr int kHv      = 48;
constexpr int kT       = 1;
constexpr float kScale = 0.08838834764831845f;

DBuf make_f32(const std::vector<float>& h) {
    DBuf d(h.size() * sizeof(float));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> make_normalized_qk(std::size_t rows, std::uint32_t seed) {
    std::vector<float> h(rows * kS);
    std::uint32_t state = seed;
    for (float& x : h) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        x             = 2.0f * u - 1.0f;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        float* row = h.data() + r * kS;
        double ss  = 0.0;
        for (int i = 0; i < kS; ++i) { ss += static_cast<double>(row[i]) * row[i]; }
        const float inv = static_cast<float>(1.0 / std::sqrt(ss + 1.0e-12));
        for (int i = 0; i < kS; ++i) { row[i] *= inv; }
    }
    return h;
}

std::vector<float> make_ramp(std::size_t n, float scale) {
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        h[i] = scale * (0.5f - static_cast<float>(i % 251) / 250.0f);
    }
    return h;
}

void run_public_wrapper() {
    const std::size_t qk_n    = static_cast<std::size_t>(kS) * kHqk * kT;
    const std::size_t v_n     = static_cast<std::size_t>(kS) * kHv * kT;
    const std::size_t gb_n    = static_cast<std::size_t>(kHv) * kT;
    const std::size_t state_n = static_cast<std::size_t>(kS) * kS * kHv;

    DBuf q     = make_bf16(qk_n);
    DBuf k     = make_bf16(qk_n);
    DBuf v     = make_bf16(v_n);
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * 2u);

    Tensor tq(q.p, DType::BF16, {kS, kHqk, kT});
    Tensor tk(k.p, DType::BF16, {kS, kHqk, kT});
    Tensor tv(v.p, DType::BF16, {kS, kHv, kT});
    Tensor tg(g.p, DType::FP32, {kHv, kT});
    Tensor tbeta(beta.p, DType::FP32, {kHv, kT});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::BF16, {kS, kHv, kT});

    const double bytes = static_cast<double>((qk_n * 2u + qk_n * 2u + v_n * 2u + gb_n * 8u) +
                                             (state_n * 8u + v_n * 2u));
    const Result r     = bench_loop(
        [&](cudaStream_t s) {
            kernels::gated_delta_rule_recurrent(tq, tk, tv, tg, tbeta, kScale, tstate, tout, s);
        },
        bytes, 20, 100, 1000);
    print_result("gdn recurrent public-wrapper [S128,Hqk16,Hv48,L1]", r);
}

void run_kernel_only_fp32() {
    const std::size_t qk_n    = static_cast<std::size_t>(kS) * kHqk * kT;
    const std::size_t v_n     = static_cast<std::size_t>(kS) * kHv * kT;
    const std::size_t gb_n    = static_cast<std::size_t>(kHv) * kT;
    const std::size_t state_n = static_cast<std::size_t>(kS) * kS * kHv;

    DBuf q     = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * kT, 0x12345678u));
    DBuf k     = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * kT, 0x87654321u));
    DBuf v     = make_f32(make_ramp(v_n, 0.5f));
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(float));

    Tensor tq(q.p, DType::FP32, {kS, kHqk, kT});
    Tensor tk(k.p, DType::FP32, {kS, kHqk, kT});
    Tensor tv(v.p, DType::FP32, {kS, kHv, kT});
    Tensor tg(g.p, DType::FP32, {kHv, kT});
    Tensor tbeta(beta.p, DType::FP32, {kHv, kT});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::FP32, {kS, kHv, kT});

    const double bytes = static_cast<double>((qk_n * 4u + qk_n * 4u + v_n * 4u + gb_n * 8u) +
                                             (state_n * 8u + v_n * 4u));
    const Result r     = bench_loop(
        [&](cudaStream_t s) {
            kernels::detail::gated_delta_rule_recurrent_launch(tq, tk, tv, tg, tbeta, kScale,
                                                                   tstate, tout, s);
        },
        bytes, 20, 100, 1000);
    print_result("gdn recurrent kernel-only-fp32 [S128,Hqk16,Hv48,L1]", r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode      = false;
    bool kernel_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) {
            decode = true;
        } else if (!std::strcmp(argv[i], "--kernel-only")) {
            kernel_only = true;
        }
    }
    if (!decode) { decode = true; }

    if (decode && kernel_only) {
        run_kernel_only_fp32();
    } else if (decode) {
        run_public_wrapper();
    }
    return 0;
}
