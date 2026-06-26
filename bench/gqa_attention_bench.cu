// Performance bench for gqa_attention at Qwen3.6-27B decode/prefill shapes.
// The GB/s readout is informational; correctness is covered by
// tests/kernels/test_gqa_attention.cpp.
//   ./qus_gqa_attention_bench --decode
//   ./qus_gqa_attention_bench --prefill
#include "qus/kernels/gqa_attention.h"
#include "qus_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKVHeads = 4;
constexpr float kScale          = 0.0625f;

DBuf make_i32(std::int32_t value) {
    DBuf d(sizeof(std::int32_t));
    cudaMemcpy(d.p, &value, sizeof(std::int32_t), cudaMemcpyHostToDevice);
    return d;
}

void run_decode(KVCache& kv, const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
                std::int32_t pos_value) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});

    const double window      = static_cast<double>(pos_value) + 1.0;
    const double q_elements  = static_cast<double>(kHeadDim) * static_cast<double>(kQHeads);
    const double kv_elements = static_cast<double>(kHeadDim) * static_cast<double>(kKVHeads);
    const double bytes = ((4.0 * window * q_elements) + q_elements + (4.0 * kv_elements)) * 2.0;

    const Result r = bench_loop(
        [&](cudaStream_t s) { kernels::gqa_attention_decode(q, k, v, pos, kScale, kv, 0, out, s); },
        bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention decode pos=%d", pos_value);
    print_result(tag, r);
}

void run_prefill(KVCache& kv, std::int32_t tokens) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const double causal_pairs = static_cast<double>(tokens) * static_cast<double>(tokens + 1) * 0.5;
    const double q_elements   = static_cast<double>(qn);
    const double kv_elements  = static_cast<double>(kvn);
    const double bytes =
        (4.0 * kv_elements + q_elements + q_elements +
         2.0 * causal_pairs * static_cast<double>(kQHeads) * static_cast<double>(kHeadDim)) *
        2.0;

    const Result r = bench_loop(
        [&](cudaStream_t s) { kernels::gqa_attention_prefill(tq, tk, tv, kScale, kv, 0, tout, s); },
        bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention prefill T=%d", tokens);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode  = false;
    bool prefill = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) decode = true;
        if (!std::strcmp(argv[i], "--prefill")) prefill = true;
    }
    if (!decode && !prefill) { decode = true; }

    constexpr std::int32_t max_context = 32769;
    const std::size_t layer_elements   = static_cast<std::size_t>(kKVHeads) *
                                       static_cast<std::size_t>(kHeadDim) *
                                       static_cast<std::size_t>(max_context);
    const std::size_t layer_bytes = layer_elements * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, max_context, kKVHeads, kHeadDim, DType::BF16);
    cudaMemset(kv.k[0].data, 0x3e, layer_bytes);
    cudaMemset(kv.v[0].data, 0x3d, layer_bytes);

    constexpr std::size_t qn  = static_cast<std::size_t>(kHeadDim) * kQHeads;
    constexpr std::size_t kvn = static_cast<std::size_t>(kHeadDim) * kKVHeads;
    DBuf q                    = make_bf16(qn);
    DBuf k                    = make_bf16(kvn);
    DBuf v                    = make_bf16(kvn);
    DBuf out                  = make_zeros(qn * sizeof(std::uint16_t));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, 1});

    if (decode) {
        run_decode(kv, tq, tk, tv, tout, 2048);
        run_decode(kv, tq, tk, tv, tout, 32768);
    }
    if (prefill) {
        run_prefill(kv, 128);
        run_prefill(kv, 2048);
    }
    return 0;
}
