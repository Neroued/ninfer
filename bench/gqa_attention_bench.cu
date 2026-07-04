// Performance bench for gqa_attention at Qwen3.6-27B decode/prefill shapes.
// Prefill reports useful causal attention FLOP/s against the RTX 5090
// bf16/FP32-accumulate dense tensor-core roofline. Correctness is covered by
// tests/kernels/test_gqa_attention.cpp.
//   ./qus_gqa_attention_bench --decode
//   ./qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
//   ./qus_gqa_attention_bench --append-small-t --tokens 6 --context 32768
//   ./qus_gqa_attention_bench --prefill --tokens 4096
//   ./qus_gqa_attention_bench --prefill --tokens 4096 --expect-tflops-pct-min 80
#include "qus/core/device.h"
#include "qus/kernels/gqa_attention.h"
#include "qus_bench_common.h"
#include "../src/kernels/launcher/gqa_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace qus;
using namespace qus::bench;
using qus::kernels::kGqaDecodeSplits;

namespace {

constexpr std::int32_t kHeadDim                  = 256;
constexpr std::int32_t kQHeads                   = 24;
constexpr std::int32_t kKVHeads                  = 4;
constexpr float kScale                           = 0.0625f;
constexpr std::int32_t kDChunk                   = 64;
constexpr std::size_t kColdCacheBytes            = std::size_t(256) << 20;
constexpr std::int32_t kDefaultDecodePositions[] = {2048, 2882, 8192, 32768};
constexpr std::int32_t kDefaultPrefillTokens[]   = {512, 1024, 2048, 4096};
constexpr double kDenseTcPeakTflops              = 209.5;
constexpr double kDramPeakGBs                    = 1792.0;

constexpr std::int32_t align_up_128(std::int32_t value) { return ((value + 127) / 128) * 128; }

std::int32_t ceil_div_i32(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::size_t ceil_div_size(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1u) / divisor;
}

std::size_t align_overhead(std::size_t allocations) { return allocations * 255u; }

std::int32_t decode_kps(std::int32_t pos_value) {
    const std::int32_t window = pos_value + 1;
    return ceil_div_i32(window, kGqaDecodeSplits);
}

std::int32_t small_t_active_splits(std::int32_t tokens, std::int32_t context) {
    if (tokens <= 1) { return kGqaDecodeSplits; }
    const std::int32_t target_keys_per_split = (tokens <= 5) ? 480 : 512;
    constexpr std::int32_t kMinSplits        = 4;
    const std::int32_t window                = context + tokens;
    std::int32_t splits                      = ceil_div_i32(window, target_keys_per_split);
    splits                                   = std::max(kMinSplits, splits);
    return std::min(kGqaDecodeSplits, splits);
}

struct DecodeBytes {
    std::size_t useful_kv = 0;
    std::size_t scratch   = 0;
    std::size_t total     = 0;
};

DecodeBytes decode_bytes(std::int32_t pos_value) {
    const auto window          = static_cast<std::size_t>(pos_value) + 1u;
    constexpr auto split_count = static_cast<std::size_t>(kGqaDecodeSplits);

    const std::size_t k_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t v_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t new_kv_writes = kKVHeads * kHeadDim * sizeof(std::uint16_t) * 2u;
    const std::size_t q_reads       = kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t output_writes = kQHeads * kHeadDim * sizeof(std::uint16_t);

    const std::size_t partial_acc_writes_reads =
        2u * split_count * kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t partial_ml_writes = split_count * kQHeads * 2u * sizeof(float);
    const std::size_t partial_ml_reads =
        ceil_div_i32(kHeadDim, kDChunk) * split_count * kQHeads * 2u * sizeof(float);

    DecodeBytes bytes;
    bytes.useful_kv = k_cache_reads + v_cache_reads;
    bytes.scratch   = partial_acc_writes_reads + partial_ml_writes + partial_ml_reads;
    bytes.total     = bytes.useful_kv + new_kv_writes + q_reads + output_writes + bytes.scratch;
    return bytes;
}

DecodeBytes append_small_t_bytes(std::int32_t tokens, std::int32_t context) {
    const auto window      = static_cast<std::size_t>(context + tokens);
    const auto token_count = static_cast<std::size_t>(tokens);
    const auto split_count = static_cast<std::size_t>(small_t_active_splits(tokens, context));

    const std::size_t k_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t v_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t new_kv_writes =
        token_count * kKVHeads * kHeadDim * sizeof(std::uint16_t) * 2u;
    const std::size_t q_reads       = token_count * kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t output_writes = q_reads;

    const std::size_t partial_acc_writes_reads =
        2u * split_count * token_count * kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t partial_ml_writes = split_count * token_count * kQHeads * 2u * sizeof(float);
    const std::size_t partial_ml_reads =
        ceil_div_i32(kHeadDim, kDChunk) * split_count * token_count * kQHeads * 2u * sizeof(float);

    DecodeBytes bytes;
    bytes.useful_kv = k_cache_reads + v_cache_reads;
    bytes.scratch   = partial_acc_writes_reads + partial_ml_writes + partial_ml_reads;
    bytes.total     = bytes.useful_kv + new_kv_writes + q_reads + output_writes + bytes.scratch;
    return bytes;
}

std::size_t decode_workspace_bytes_for_pos(std::int32_t) {
    constexpr auto split_count = static_cast<std::size_t>(kGqaDecodeSplits);
    const std::size_t partial_acc_bytes =
        static_cast<std::size_t>(kHeadDim) * kQHeads * split_count * sizeof(std::uint16_t);
    const std::size_t partial_m_bytes =
        static_cast<std::size_t>(kQHeads) * split_count * sizeof(float);
    const std::size_t partial_l_bytes =
        static_cast<std::size_t>(kQHeads) * split_count * sizeof(float);
    return partial_acc_bytes + partial_m_bytes + partial_l_bytes + align_overhead(3);
}

std::size_t small_t_workspace_bytes(std::int32_t tokens) {
    constexpr auto split_count          = static_cast<std::size_t>(kGqaDecodeSplits);
    const auto token_count              = static_cast<std::size_t>(tokens);
    const std::size_t partial_acc_bytes = static_cast<std::size_t>(kHeadDim) * kQHeads *
                                          token_count * split_count * sizeof(std::uint16_t);
    const std::size_t partial_m_bytes =
        static_cast<std::size_t>(kQHeads) * token_count * split_count * sizeof(float);
    const std::size_t partial_l_bytes =
        static_cast<std::size_t>(kQHeads) * token_count * split_count * sizeof(float);
    return partial_acc_bytes + partial_m_bytes + partial_l_bytes + align_overhead(3);
}

std::size_t decode_workspace_bytes(const std::vector<std::int32_t>& positions) {
    std::size_t bytes = 1u;
    for (const std::int32_t pos : positions) {
        bytes = std::max(bytes, decode_workspace_bytes_for_pos(pos));
    }
    return bytes;
}

std::size_t cache_arena_bytes(std::uint32_t layers, std::int32_t max_context) {
    const auto padded_context = static_cast<std::size_t>(align_up_128(max_context));
    const std::size_t layer_elements =
        static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(kHeadDim) * padded_context;
    const std::size_t layer_bytes = layer_elements * sizeof(std::uint16_t);
    return static_cast<std::size_t>(layers) * 2u * (layer_bytes + 255u) + 4096u;
}

DBuf make_i32(std::int32_t value) {
    DBuf d(sizeof(std::int32_t));
    cudaMemcpy(d.p, &value, sizeof(std::int32_t), cudaMemcpyHostToDevice);
    return d;
}

DBuf make_i32_sequence(std::int32_t start, std::int32_t count) {
    std::vector<std::int32_t> values(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { values[static_cast<std::size_t>(i)] = start + i; }
    DBuf d(values.size() * sizeof(std::int32_t));
    cudaMemcpy(d.p, values.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

__global__ void bench_cold_cache_touch_kernel(std::uint32_t* data, std::size_t words) {
    const std::size_t start = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t step  = blockDim.x * static_cast<std::size_t>(gridDim.x);
    for (std::size_t i = start; i < words; i += step) { data[i] = data[i] + 1u; }
}

void touch_cold_cache(DBuf& buf, cudaStream_t stream) {
    constexpr int kBlock    = 256;
    const std::size_t words = buf.bytes / sizeof(std::uint32_t);
    if (words == 0) { return; }
    const int grid = static_cast<int>(std::min<std::size_t>(4096u, ceil_div_size(words, kBlock)));
    bench_cold_cache_touch_kernel<<<grid, kBlock, 0, stream>>>(static_cast<std::uint32_t*>(buf.p),
                                                               words);
    CUDA_CHECK(cudaGetLastError());
}

Result bench_cold_cache_loop(const launch_fn& launch, DBuf& cold_cache, double bytes_moved,
                             int warmup = 5, int repeat = 100) {
    cudaStream_t stream = nullptr;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    for (int i = 0; i < warmup; ++i) {
        touch_cold_cache(cold_cache, stream);
        launch(stream);
    }
    cudaStreamSynchronize(stream);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        touch_cold_cache(cold_cache, stream);
        cudaEventRecord(a, stream);
        launch(stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(static_cast<double>(ms) * 1000.0);
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
    for (double v : samples) { sum += v; }

    Result r;
    r.n_runs         = static_cast<int>(samples.size());
    r.inner_iters    = 1;
    r.median_us      = pct(0.50);
    r.min_us         = sorted.front();
    r.p95_us         = pct(0.95);
    r.mean_us        = sum / static_cast<double>(samples.size());
    const double sec = r.median_us * 1e-6;
    r.gbs            = (sec > 0.0) ? bytes_moved / sec / 1e9 : 0.0;
    return r;
}

void print_decode_result(const char* tag, const Result& r, const DecodeBytes& bytes,
                         std::int32_t pos_value, std::uint32_t round_robin_layers,
                         const char* suffix) {
    const double sec       = r.median_us * 1.0e-6;
    const double total_gbs = (sec > 0.0) ? static_cast<double>(bytes.total) / sec / 1.0e9 : 0.0;
    const double useful_kv_gbs =
        (sec > 0.0) ? static_cast<double>(bytes.useful_kv) / sec / 1.0e9 : 0.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  total_model=%8.1f GB/s  "
                "useful_kv=%8.1f GB/s  bytes useful_kv=%.2f MiB scratch=%.2f MiB total=%.2f MiB  "
                "splits=%d kps=%d round_robin_layers=%u%s\n",
                tag, r.median_us, r.min_us, r.p95_us, total_gbs, useful_kv_gbs,
                static_cast<double>(bytes.useful_kv) / kMiB,
                static_cast<double>(bytes.scratch) / kMiB, static_cast<double>(bytes.total) / kMiB,
                kGqaDecodeSplits, decode_kps(pos_value), round_robin_layers, suffix);
}

void print_append_small_t_result(const char* tag, const Result& r, const DecodeBytes& bytes,
                                 std::int32_t tokens, std::int32_t context, const char* suffix) {
    const double sec       = r.median_us * 1.0e-6;
    const double total_gbs = (sec > 0.0) ? static_cast<double>(bytes.total) / sec / 1.0e9 : 0.0;
    const double useful_kv_gbs =
        (sec > 0.0) ? static_cast<double>(bytes.useful_kv) / sec / 1.0e9 : 0.0;
    const double redundancy = bytes.useful_kv > 0 ? static_cast<double>(bytes.total) /
                                                        static_cast<double>(bytes.useful_kv)
                                                  : 0.0;
    constexpr double kMiB   = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  total_model=%8.1f GB/s  "
                "useful_kv=%8.1f GB/s  redundancy=%5.2f  bytes useful_kv=%.2f MiB "
                "scratch=%.2f MiB total=%.2f MiB  T=%d context=%d splits=%d%s\n",
                tag, r.median_us, r.min_us, r.p95_us, total_gbs, useful_kv_gbs, redundancy,
                static_cast<double>(bytes.useful_kv) / kMiB,
                static_cast<double>(bytes.scratch) / kMiB, static_cast<double>(bytes.total) / kMiB,
                tokens, context, small_t_active_splits(tokens, context), suffix);
}

void print_append_prompt_baseline_result(const Result& r, const DecodeBytes& bytes,
                                         std::int32_t tokens, std::int32_t context) {
    const double sec = r.median_us * 1.0e-6;
    const double useful_kv_gbs =
        (sec > 0.0) ? static_cast<double>(bytes.useful_kv) / sec / 1.0e9 : 0.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  "
                "ideal_useful_kv=%8.1f GB/s  bytes useful_kv=%.2f MiB  T=%d context=%d\n",
                "gqa_attention append prompt baseline", r.median_us, r.min_us, r.p95_us,
                useful_kv_gbs, static_cast<double>(bytes.useful_kv) / kMiB, tokens, context);
}

void run_decode(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k, const Tensor& v,
                Tensor& out, std::int32_t pos_value, std::uint32_t round_robin_layers) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});

    const DecodeBytes bytes  = decode_bytes(pos_value);
    std::uint32_t next_layer = 0;

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            kv.pos          = static_cast<std::uint32_t>(pos_value);
            const int layer = static_cast<int>(next_layer);
            next_layer      = (next_layer + 1u) % round_robin_layers;
            kernels::gqa_attention(q, k, v, pos, kScale, kv, layer, ws, out, s);
        },
        static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention decode combined pos=%d", pos_value);
    print_decode_result(tag, r, bytes, pos_value, round_robin_layers,
                        (round_robin_layers == 1u) ? " hot_cache_info" : "");
}

void run_profile_once(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k,
                      const Tensor& v, Tensor& out, std::int32_t pos_value, DBuf* cold_cache) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});
    const DecodeBytes bytes = decode_bytes(pos_value);

    cudaStream_t stream = nullptr;
    if (cold_cache != nullptr) {
        const Result r = bench_cold_cache_loop(
            [&](cudaStream_t s) {
                kv.pos = static_cast<std::uint32_t>(pos_value);
                kernels::gqa_attention(q, k, v, pos, kScale, kv, 0, ws, out, s);
            },
            *cold_cache, static_cast<double>(bytes.total));

        char tag[96];
        std::snprintf(tag, sizeof(tag), "PROFILE_COLD gqa_attention decode pos=%d", pos_value);
        print_decode_result(tag, r, bytes, pos_value, 1u, " cold_cache");
        std::printf("PROFILE_COLD_METADATA pos=%d splits=%d kps=%d cold_cache_bytes=%zu "
                    "useful_kv_bytes=%zu scratch_bytes=%zu total_modeled_bytes=%zu repeats=%d\n",
                    pos_value, kGqaDecodeSplits, decode_kps(pos_value), cold_cache->bytes,
                    bytes.useful_kv, bytes.scratch, bytes.total, r.n_runs);
        return;
    }

    kv.pos = static_cast<std::uint32_t>(pos_value);
    kernels::gqa_attention(q, k, v, pos, kScale, kv, 0, ws, out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("PROFILE_ONCE gqa_attention decode combined pos=%d splits=%d kps=%d "
                "useful_kv_bytes=%zu scratch_bytes=%zu total_modeled_bytes=%zu\n",
                pos_value, kGqaDecodeSplits, decode_kps(pos_value), bytes.useful_kv, bytes.scratch,
                bytes.total);
}

void run_append_small_t(KVCache& kv, std::int32_t tokens, std::int32_t context) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(context, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(small_t_workspace_bytes(tokens));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const DecodeBytes bytes = append_small_t_bytes(tokens, context);
    const Result r          = bench_loop(
        [&](cudaStream_t s) {
            kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, s);
        },
        static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention append-small-T");
    print_append_small_t_result(tag, r, bytes, tokens, context, "");
}

void run_append_small_t_profile_once(KVCache& kv, std::int32_t tokens, std::int32_t context,
                                     DBuf* cold_cache) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(context, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(small_t_workspace_bytes(tokens));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const DecodeBytes bytes = append_small_t_bytes(tokens, context);
    cudaStream_t stream     = nullptr;
    if (cold_cache != nullptr) {
        const Result r = bench_cold_cache_loop(
            [&](cudaStream_t s) {
                kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, s);
            },
            *cold_cache, static_cast<double>(bytes.total));
        print_append_small_t_result("PROFILE_COLD gqa_attention append-small-T", r, bytes, tokens,
                                    context, " cold_cache");
        std::printf("PROFILE_COLD_METADATA mode=append-small-t T=%d context=%d splits=%d "
                    "cold_cache_bytes=%zu useful_kv_bytes=%zu scratch_bytes=%zu "
                    "total_modeled_bytes=%zu redundancy=%.6f repeats=%d "
                    "ncu_kernel_regex='gqa_attention_small_t_(tc_)?partial_kernel'\n",
                    tokens, context, small_t_active_splits(tokens, context), cold_cache->bytes,
                    bytes.useful_kv, bytes.scratch, bytes.total,
                    bytes.useful_kv > 0
                        ? static_cast<double>(bytes.total) / static_cast<double>(bytes.useful_kv)
                        : 0.0,
                    r.n_runs);
        return;
    }

    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("PROFILE_ONCE gqa_attention append-small-T T=%d context=%d splits=%d "
                "useful_kv_bytes=%zu scratch_bytes=%zu total_model_bytes=%zu redundancy=%.6f "
                "ncu_kernel_regex='gqa_attention_small_t_(tc_)?partial_kernel'\n",
                tokens, context, small_t_active_splits(tokens, context), bytes.useful_kv,
                bytes.scratch, bytes.total,
                bytes.useful_kv > 0
                    ? static_cast<double>(bytes.total) / static_cast<double>(bytes.useful_kv)
                    : 0.0);
}

void run_append_prompt_baseline(KVCache& kv, std::int32_t tokens, std::int32_t context) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(context, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const DecodeBytes bytes = append_small_t_bytes(tokens, context);
    const Result r          = bench_loop(
        [&](cudaStream_t s) {
            kernels::detail::gqa_attention_prompt_launch(tq, tk, tv, tpos, kScale, kv, 0, tout, s);
        },
        static_cast<double>(bytes.useful_kv));

    print_append_prompt_baseline_result(r, bytes, tokens, context);
}

double prefill_useful_flops(std::int32_t tokens) {
    return 2.0 * static_cast<double>(kHeadDim) * static_cast<double>(kQHeads) *
           static_cast<double>(tokens) * static_cast<double>(tokens + 1);
}

double prefill_model_floor_bytes(std::int32_t tokens) {
    const double q_bytes = static_cast<double>(tokens) * static_cast<double>(kQHeads) *
                           static_cast<double>(kHeadDim) *
                           static_cast<double>(sizeof(std::uint16_t));
    const double out_bytes = q_bytes;
    const double k_bytes   = static_cast<double>(tokens) * static_cast<double>(kKVHeads) *
                           static_cast<double>(kHeadDim) *
                           static_cast<double>(sizeof(std::uint16_t));
    const double v_bytes = k_bytes;
    return q_bytes + out_bytes + k_bytes + v_bytes;
}

struct PrefillTimingOptions {
    int warmup      = 3;
    int repeat      = 10;
    int min_time_ms = 0;
};

struct PrefillMetrics {
    std::int32_t tokens      = 0;
    int runs                 = 0;
    int inner_iters          = 1;
    double median_ms         = 0.0;
    double min_ms            = 0.0;
    double p95_ms            = 0.0;
    double mean_ms           = 0.0;
    double useful_flops      = 0.0;
    double model_floor_bytes = 0.0;
    double tflops            = 0.0;
    double tflops_pct        = 0.0;
    double gbps_model        = 0.0;
    double gbps_model_pct    = 0.0;
    double roofline_tflops   = 0.0;
    double roofline_eff_pct  = 0.0;
    const char* bound        = "tc";
};

PrefillMetrics prefill_metrics_from_result(std::int32_t tokens, const Result& r) {
    PrefillMetrics m;
    m.tokens            = tokens;
    m.runs              = r.n_runs;
    m.inner_iters       = r.inner_iters;
    m.median_ms         = r.median_us * 1.0e-3;
    m.min_ms            = r.min_us * 1.0e-3;
    m.p95_ms            = r.p95_us * 1.0e-3;
    m.mean_ms           = r.mean_us * 1.0e-3;
    m.useful_flops      = prefill_useful_flops(tokens);
    m.model_floor_bytes = prefill_model_floor_bytes(tokens);

    const double sec = r.median_us * 1.0e-6;
    if (sec > 0.0) {
        m.tflops     = m.useful_flops / sec / 1.0e12;
        m.gbps_model = m.model_floor_bytes / sec / 1.0e9;
    }
    m.tflops_pct     = m.tflops / kDenseTcPeakTflops * 100.0;
    m.gbps_model_pct = m.gbps_model / kDramPeakGBs * 100.0;

    const double intensity = m.model_floor_bytes > 0.0 ? m.useful_flops / m.model_floor_bytes : 0.0;
    const double dram_roof_tflops = kDramPeakGBs * intensity / 1000.0;
    m.roofline_tflops             = std::min(kDenseTcPeakTflops, dram_roof_tflops);
    m.bound                       = (kDenseTcPeakTflops <= dram_roof_tflops) ? "tc" : "dram";
    m.roofline_eff_pct = (m.roofline_tflops > 0.0) ? (m.tflops / m.roofline_tflops * 100.0) : 0.0;
    return m;
}

void print_prefill_result(const PrefillMetrics& m) {
    std::printf("gqa_attention prefill T=%-6d median=%9.3f ms  min=%9.3f ms  p95=%9.3f ms  "
                "useful=%9.2f TFLOP/s  tc=%6.2f%% of %.1f  model_floor=%8.1f GB/s "
                "(%5.2f%% of %.0f)  bound=%s  roofline_eff=%6.2f%%  runs=%d inner=%d\n",
                m.tokens, m.median_ms, m.min_ms, m.p95_ms, m.tflops, m.tflops_pct,
                kDenseTcPeakTflops, m.gbps_model, m.gbps_model_pct, kDramPeakGBs, m.bound,
                m.roofline_eff_pct, m.runs, m.inner_iters);
}

PrefillMetrics run_prefill(KVCache& kv, std::int32_t tokens, const PrefillTimingOptions& timing) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(0, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(tokens <= 6 ? small_t_workspace_bytes(tokens) : 1u);

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, s);
        },
        prefill_model_floor_bytes(tokens), timing.warmup, timing.repeat, timing.min_time_ms);

    PrefillMetrics metrics = prefill_metrics_from_result(tokens, r);
    print_prefill_result(metrics);
    return metrics;
}

void run_prefill_profile_once(KVCache& kv, std::int32_t tokens) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(0, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(tokens <= 6 ? small_t_workspace_bytes(tokens) : 1u);

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    cudaStream_t stream = nullptr;
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("PROFILE_ONCE gqa_attention prefill T=%d useful_flops=%.0f "
                "model_floor_bytes=%.0f tc_peak_tflops=%.1f dram_peak_gbps=%.0f "
                "ncu_kernel_regex='gqa_attention_prefill_kernel'\n",
                tokens, prefill_useful_flops(tokens), prefill_model_floor_bytes(tokens),
                kDenseTcPeakTflops, kDramPeakGBs);
}

std::string json_number(double value) {
    if (!std::isfinite(value)) { return "null"; }
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string format_prefill_csv(const std::vector<PrefillMetrics>& results) {
    std::ostringstream out;
    out << "T,ms,tflops,tflops_pct,gbps_model,gbps_model_pct,gbps_dram,gbps_dram_pct,"
           "gbps_pct,bound,"
           "roofline_tflops,roofline_eff_pct,model_floor_bytes,useful_flops,"
           "median_ms,min_ms,p95_ms,mean_ms,runs,inner_iters\n";
    for (const PrefillMetrics& m : results) {
        out << m.tokens << ',' << json_number(m.median_ms) << ',' << json_number(m.tflops) << ','
            << json_number(m.tflops_pct) << ',' << json_number(m.gbps_model) << ','
            << json_number(m.gbps_model_pct) << ",,," << json_number(m.gbps_model_pct) << ','
            << m.bound << ',' << json_number(m.roofline_tflops) << ','
            << json_number(m.roofline_eff_pct) << ',' << json_number(m.model_floor_bytes) << ','
            << json_number(m.useful_flops) << ',' << json_number(m.median_ms) << ','
            << json_number(m.min_ms) << ',' << json_number(m.p95_ms) << ','
            << json_number(m.mean_ms) << ',' << m.runs << ',' << m.inner_iters << '\n';
    }
    return out.str();
}

std::string format_prefill_json(const std::vector<PrefillMetrics>& results,
                                const PrefillTimingOptions& timing) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact_type\": \"qus_gqa_attention_prefill_bench\",\n"
        << "  \"tc_peak_tflops\": " << json_number(kDenseTcPeakTflops) << ",\n"
        << "  \"dram_peak_gbps\": " << json_number(kDramPeakGBs) << ",\n"
        << "  \"flops_definition\": \"useful_causal_2*d*Hq*T*(T+1)\",\n"
        << "  \"gbps_pct_definition\": \"model_floor_bytes_per_second / dram_peak_gbps\",\n"
        << "  \"timing\": {\"warmup\": " << timing.warmup << ", \"repeat\": " << timing.repeat
        << ", \"min_time_ms\": " << timing.min_time_ms << "},\n"
        << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const PrefillMetrics& m = results[i];
        out << "    {\"T\": " << m.tokens << ", \"ms\": " << json_number(m.median_ms)
            << ", \"tflops\": " << json_number(m.tflops)
            << ", \"tflops_pct\": " << json_number(m.tflops_pct)
            << ", \"gbps_model\": " << json_number(m.gbps_model)
            << ", \"gbps_model_pct\": " << json_number(m.gbps_model_pct) << ", \"gbps_dram\": null"
            << ", \"gbps_dram_pct\": null" << ", \"gbps_pct\": " << json_number(m.gbps_model_pct)
            << ", \"bound\": \"" << m.bound
            << "\", \"roofline_tflops\": " << json_number(m.roofline_tflops)
            << ", \"roofline_eff_pct\": " << json_number(m.roofline_eff_pct)
            << ", \"model_floor_bytes\": " << json_number(m.model_floor_bytes)
            << ", \"useful_flops\": " << json_number(m.useful_flops)
            << ", \"median_ms\": " << json_number(m.median_ms)
            << ", \"min_ms\": " << json_number(m.min_ms)
            << ", \"p95_ms\": " << json_number(m.p95_ms)
            << ", \"mean_ms\": " << json_number(m.mean_ms) << ", \"runs\": " << m.runs
            << ", \"inner_iters\": " << m.inner_iters << "}" << (i + 1 < results.size() ? "," : "")
            << "\n";
    }
    out << "  ]\n"
        << "}\n";
    return out.str();
}

void write_text_file(const std::string& path, const std::string& text) {
    if (path.empty()) { return; }
    const std::filesystem::path p(path);
    if (!p.parent_path().empty()) { std::filesystem::create_directories(p.parent_path()); }
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "error: failed to open output file: %s\n", path.c_str());
        std::exit(2);
    }
    out << text;
    std::printf("wrote %s\n", path.c_str());
}

struct Options {
    bool decode                      = false;
    bool prefill                     = false;
    bool append_small_t              = false;
    bool append_prompt_baseline      = false;
    bool decode_pos_set              = false;
    bool context_set                 = false;
    bool profile_once                = false;
    bool cold_cache                  = false;
    std::int32_t decode_pos          = 0;
    std::int32_t context             = 0;
    std::uint32_t round_robin_layers = 1;
    std::vector<std::int32_t> tokens;
    PrefillTimingOptions prefill_timing;
    double expect_tflops_pct_min = -1.0;
    std::string csv_out;
    std::string json_out;
};

void fail_usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: qus_gqa_attention_bench [--prefill] [--tokens T[,T...]] "
                 "[--expect-tflops-pct-min PCT] [--csv-out path] [--json-out path]\n"
                 "       qus_gqa_attention_bench --prefill --tokens 4096 --profile-once\n"
                 "       qus_gqa_attention_bench --append-small-t --tokens T --context N "
                 "[--profile-once] [--cold-cache]\n"
                 "       qus_gqa_attention_bench --append-prompt-baseline --tokens T --context N\n"
                 "       qus_gqa_attention_bench --decode [--decode-pos N] [--profile-once] "
                 "[--cold-cache] [--round-robin-layers 16]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32_arg(const char* text, const char* flag) {
    errno            = 0;
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s expects a non-negative int32", flag);
        fail_usage(msg);
    }
    return static_cast<std::int32_t>(value);
}

double parse_double_arg(const char* text, const char* flag) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s expects a finite number", flag);
        fail_usage(msg);
    }
    return value;
}

std::vector<std::int32_t> parse_i32_list_arg(const char* text, const char* flag) {
    std::vector<std::int32_t> out;
    const char* start = text;
    while (*start != '\0') {
        const char* comma = std::strchr(start, ',');
        std::string piece(start, comma == nullptr ? std::strlen(start)
                                                  : static_cast<std::size_t>(comma - start));
        const std::int32_t value = parse_i32_arg(piece.c_str(), flag);
        if (value <= 0) { fail_usage("--tokens expects positive lengths"); }
        out.push_back(value);
        if (comma == nullptr) { break; }
        start = comma + 1;
    }
    if (out.empty()) { fail_usage("--tokens expects at least one length"); }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) {
            opt.decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            opt.prefill = true;
        } else if (!std::strcmp(argv[i], "--append-small-t")) {
            opt.append_small_t = true;
        } else if (!std::strcmp(argv[i], "--append-prompt-baseline")) {
            opt.append_prompt_baseline = true;
        } else if (!std::strcmp(argv[i], "--tokens") || !std::strcmp(argv[i], "--prefill-tokens")) {
            if (++i >= argc) { fail_usage("--tokens requires a value"); }
            opt.tokens = parse_i32_list_arg(argv[i], "--tokens");
        } else if (!std::strcmp(argv[i], "--decode-pos")) {
            if (++i >= argc) { fail_usage("--decode-pos requires a value"); }
            opt.decode_pos     = parse_i32_arg(argv[i], "--decode-pos");
            opt.decode_pos_set = true;
            opt.decode         = true;
        } else if (!std::strcmp(argv[i], "--context")) {
            if (++i >= argc) { fail_usage("--context requires a value"); }
            opt.context     = parse_i32_arg(argv[i], "--context");
            opt.context_set = true;
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            opt.profile_once = true;
        } else if (!std::strcmp(argv[i], "--cold-cache")) {
            opt.cold_cache = true;
        } else if (!std::strcmp(argv[i], "--round-robin-layers")) {
            if (++i >= argc) { fail_usage("--round-robin-layers requires a value"); }
            opt.round_robin_layers =
                static_cast<std::uint32_t>(parse_i32_arg(argv[i], "--round-robin-layers"));
            if (opt.round_robin_layers == 0) {
                fail_usage("--round-robin-layers expects a positive value");
            }
        } else if (!std::strcmp(argv[i], "--warmup")) {
            if (++i >= argc) { fail_usage("--warmup requires a value"); }
            opt.prefill_timing.warmup = parse_i32_arg(argv[i], "--warmup");
        } else if (!std::strcmp(argv[i], "--repeat")) {
            if (++i >= argc) { fail_usage("--repeat requires a value"); }
            opt.prefill_timing.repeat = parse_i32_arg(argv[i], "--repeat");
            if (opt.prefill_timing.repeat <= 0) { fail_usage("--repeat expects a positive value"); }
        } else if (!std::strcmp(argv[i], "--min-time-ms")) {
            if (++i >= argc) { fail_usage("--min-time-ms requires a value"); }
            opt.prefill_timing.min_time_ms = parse_i32_arg(argv[i], "--min-time-ms");
        } else if (!std::strcmp(argv[i], "--expect-tflops-pct-min")) {
            if (++i >= argc) { fail_usage("--expect-tflops-pct-min requires a value"); }
            opt.expect_tflops_pct_min = parse_double_arg(argv[i], "--expect-tflops-pct-min");
        } else if (!std::strcmp(argv[i], "--csv-out")) {
            if (++i >= argc) { fail_usage("--csv-out requires a value"); }
            opt.csv_out = argv[i];
        } else if (!std::strcmp(argv[i], "--json-out")) {
            if (++i >= argc) { fail_usage("--json-out requires a value"); }
            opt.json_out = argv[i];
        } else {
            fail_usage("unknown argument");
        }
    }

    if (!opt.decode && !opt.prefill && !opt.append_small_t && !opt.append_prompt_baseline) {
        opt.prefill = true;
    }
    if (opt.prefill && opt.tokens.empty()) {
        if (opt.profile_once) {
            opt.tokens = {4096};
        } else {
            opt.tokens.assign(std::begin(kDefaultPrefillTokens), std::end(kDefaultPrefillTokens));
        }
    }
    if (opt.append_small_t && opt.tokens.empty()) { opt.tokens = {1}; }
    if (opt.append_prompt_baseline && opt.tokens.empty()) { opt.tokens = {6}; }
    if ((opt.append_small_t || opt.append_prompt_baseline) && opt.tokens.size() != 1u) {
        fail_usage("append modes require exactly one --tokens value");
    }
    if (opt.append_small_t && (opt.tokens[0] <= 0 || opt.tokens[0] > 6)) {
        fail_usage("--append-small-t currently supports T in [1,6]");
    }
    if (opt.append_prompt_baseline && (opt.tokens[0] <= 0 || opt.tokens[0] > 6)) {
        fail_usage("--append-prompt-baseline currently supports T in [1,6]");
    }
    if ((opt.append_small_t || opt.append_prompt_baseline) && !opt.context_set) {
        fail_usage("append modes require --context");
    }
    if (static_cast<int>(opt.decode) + static_cast<int>(opt.prefill) +
            static_cast<int>(opt.append_small_t) + static_cast<int>(opt.append_prompt_baseline) >
        1) {
        fail_usage("select exactly one benchmark mode");
    }
    if (opt.profile_once && opt.decode && !opt.prefill && !opt.decode_pos_set) {
        fail_usage("--decode --profile-once requires --decode-pos");
    }
    if (opt.decode_pos_set && opt.decode_pos == std::numeric_limits<std::int32_t>::max()) {
        fail_usage("--decode-pos exceeds the maximum supported decode window");
    }
    if (opt.profile_once && static_cast<int>(opt.decode) + static_cast<int>(opt.prefill) +
                                    static_cast<int>(opt.append_small_t) +
                                    static_cast<int>(opt.append_prompt_baseline) >
                                1) {
        fail_usage("--profile-once must target one mode");
    }
    if (opt.profile_once && opt.prefill && opt.tokens.size() != 1u) {
        fail_usage("--prefill --profile-once requires exactly one --tokens length");
    }
    if (opt.profile_once && opt.decode && opt.round_robin_layers != 1u) {
        fail_usage("--profile-once cannot be combined with --round-robin-layers");
    }
    if (opt.cold_cache && (!opt.profile_once || (!opt.decode && !opt.append_small_t))) {
        fail_usage("--cold-cache is only valid with decode or append-small-t --profile-once");
    }
    if (opt.profile_once && opt.append_prompt_baseline) {
        fail_usage("--append-prompt-baseline does not support --profile-once");
    }
    if (opt.round_robin_layers != 1u && !opt.decode) {
        fail_usage("--round-robin-layers is only valid with --decode");
    }
    if (opt.round_robin_layers != 1u && opt.round_robin_layers != 16u) {
        fail_usage("--round-robin-layers currently supports only 16");
    }
    if (opt.expect_tflops_pct_min >= 0.0 && !opt.prefill) {
        fail_usage("--expect-tflops-pct-min is only valid with --prefill");
    }
    return opt;
}

std::vector<std::int32_t> selected_decode_positions(const Options& opt) {
    if (!opt.decode) { return {}; }
    if (opt.decode_pos_set) { return {opt.decode_pos}; }
    return std::vector<std::int32_t>(std::begin(kDefaultDecodePositions),
                                     std::end(kDefaultDecodePositions));
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    const Options opt                                = parse_options(argc, argv);
    const std::vector<std::int32_t> decode_positions = selected_decode_positions(opt);

    std::int32_t max_context = 2048;
    for (const std::int32_t pos : decode_positions) {
        max_context = std::max(max_context, pos + 1);
    }
    for (const std::int32_t tokens : opt.tokens) { max_context = std::max(max_context, tokens); }
    if (opt.append_small_t) { max_context = std::max(max_context, opt.context + opt.tokens[0]); }
    if (opt.append_prompt_baseline) {
        max_context = std::max(max_context, opt.context + opt.tokens[0]);
    }
    const std::uint32_t cache_layers = opt.round_robin_layers;
    DeviceArena cache_arena(cache_arena_bytes(cache_layers, max_context));
    WorkspaceArena work_arena(decode_workspace_bytes(decode_positions));
    KVCache kv(cache_arena, cache_layers, static_cast<std::uint32_t>(max_context), kKVHeads,
               kHeadDim, DType::BF16);
    for (std::uint32_t layer = 0; layer < kv.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemset(kv.k[layer].data, 0x3e, kv.k[layer].bytes()));
        CUDA_CHECK(cudaMemset(kv.v[layer].data, 0x3d, kv.v[layer].bytes()));
    }

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

    if (opt.decode) {
        if (opt.profile_once) {
            DBuf cold_cache = make_zeros(opt.cold_cache ? kColdCacheBytes : 1u);
            run_profile_once(kv, work_arena, tq, tk, tv, tout, opt.decode_pos,
                             opt.cold_cache ? &cold_cache : nullptr);
        } else {
            for (const std::int32_t pos : decode_positions) {
                run_decode(kv, work_arena, tq, tk, tv, tout, pos, opt.round_robin_layers);
            }
        }
    }
    if (opt.append_small_t) {
        const std::int32_t tokens = opt.tokens[0];
        if (opt.profile_once) {
            DBuf cold_cache = make_zeros(opt.cold_cache ? kColdCacheBytes : 1u);
            run_append_small_t_profile_once(kv, tokens, opt.context,
                                            opt.cold_cache ? &cold_cache : nullptr);
        } else {
            run_append_small_t(kv, tokens, opt.context);
        }
    }
    if (opt.append_prompt_baseline) {
        run_append_prompt_baseline(kv, opt.tokens[0], opt.context);
    }
    if (opt.prefill) {
        if (opt.profile_once) {
            for (const std::int32_t tokens : opt.tokens) { run_prefill_profile_once(kv, tokens); }
        } else {
            std::vector<PrefillMetrics> prefill_results;
            prefill_results.reserve(opt.tokens.size());
            for (const std::int32_t tokens : opt.tokens) {
                prefill_results.push_back(run_prefill(kv, tokens, opt.prefill_timing));
            }
            write_text_file(opt.csv_out, format_prefill_csv(prefill_results));
            write_text_file(opt.json_out, format_prefill_json(prefill_results, opt.prefill_timing));

            if (opt.expect_tflops_pct_min >= 0.0) {
                int failures = 0;
                for (const PrefillMetrics& m : prefill_results) {
                    if (m.tflops_pct < opt.expect_tflops_pct_min) {
                        std::fprintf(stderr,
                                     "FAIL: prefill T=%d tflops_pct=%.3f below expected %.3f\n",
                                     m.tokens, m.tflops_pct, opt.expect_tflops_pct_min);
                        ++failures;
                    }
                }
                if (failures != 0) { return 1; }
                std::printf("PASS: all prefill tflops_pct >= %.3f\n", opt.expect_tflops_pct_min);
            }
        }
    }
    return 0;
}
