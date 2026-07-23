// Performance bench for gqa_attention at the registered Qwen3.6 27B/35B geometries.
// Prefill reports useful causal attention FLOP/s against the RTX 5090
// bf16/FP32-accumulate dense tensor-core roofline. Correctness is covered by
// tests/ops/test_gqa_attention.cpp.
// Canonical 35B matrix: prefill uses a fixed 1024-token chunk; decode/verify
// sweeps T=1..16. Both sweep the same live-context list.
//   ./ninfer_gqa_attention_bench --append-prompt-baseline --geometry 35b --tokens 1024 \
//       --context 0,128,512,2048,8192,32768,131072,261120
//   ./ninfer_gqa_attention_bench --append-small-t \
//       --geometry 35b --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
//       --context 0,128,512,2048,8192,32768,131072,261120
//   ./ninfer_gqa_attention_bench --cached-small-t \
//       --geometry 35b --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
//       --context 0,128,512,2048,8192,32768,131072,261120
//   ./ninfer_gqa_attention_bench --kv-append --geometry 35b --tokens 1,2,3,4,5,6,1024 \
//       --context 0,128,512,2048,8192,32768,131072,261120
// Add --kv-dtype int8 to measure the INT8-G64 KV route.
//   ./ninfer_gqa_attention_bench --decode --geometry 35b
//   ./ninfer_gqa_attention_bench --decode --geometry 35b --decode-pos 2882 --profile-once \
//       --cold-cache
//   ./ninfer_gqa_attention_bench --copy-ceiling --tokens 1 --context 32768
//   ./ninfer_gqa_attention_bench --prefill --tokens 4096
//   ./ninfer_gqa_attention_bench --prefill --tokens 4096 --expect-tflops-pct-min 80
#include "core/device.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "core/kv_cache.h"
#include "ninfer_bench_common.h"

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

using namespace ninfer;
using namespace ninfer::bench;

namespace {

using KVCache = ninfer::KVCache;

constexpr std::int32_t kHeadDim                  = 256;
std::int32_t kQHeads                             = 24;
std::int32_t kKVHeads                            = 4;
std::int32_t kGqaDecodeSplits                    = 85;
const char* kGeometryName                        = "27b";
constexpr float kScale                           = 0.0625f;
constexpr std::int32_t kDChunk                   = 64;
constexpr std::size_t kColdCacheBytes            = std::size_t(256) << 20;
constexpr std::int32_t kDefaultDecodePositions[] = {2048, 2882, 8192, 32768};
constexpr std::int32_t kDefaultPrefillTokens[]   = {512, 1024, 2048, 4096};
constexpr double kDenseTcPeakTflops              = 209.5;
constexpr double kDramPeakGBs                    = 1792.0;
constexpr std::int32_t kPromptQBlock             = 64;
constexpr std::int32_t kPromptKBlock             = 32;
constexpr std::int32_t kBenchKvQuantGroup        = ninfer::kKvQuantGroup;

ops::GqaExecutionEnvelope exact_envelope(std::uint32_t visible_keys) {
    return {visible_keys, visible_keys};
}

constexpr std::int32_t align_up_128(std::int32_t value) { return ((value + 127) / 128) * 128; }

std::int32_t ceil_div_i32(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::int64_t ceil_div_i64(std::int64_t value, std::int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::size_t ceil_div_size(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1u) / divisor;
}

std::size_t align_overhead(std::size_t allocations) { return allocations * 255u; }

std::int32_t small_t_active_splits(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    const std::int32_t window      = context + tokens;
    const std::int32_t split_scale = 4 / kKVHeads;
    // Keep the byte/scratch model identical to the dtype-aware production
    // schedule so reported bandwidth does not undercount specialized splits.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return ceil_div_i32(window, 32 / split_scale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return ceil_div_i32(window, 24 / split_scale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        return std::min(42 * split_scale,
                        std::max(4 * split_scale, ceil_div_i32(window, 192 / split_scale)));
    }
    std::int32_t target_keys_per_split = 480;
    if (window <= 4096) {
        target_keys_per_split = 64 / split_scale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / split_scale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / split_scale;
    } else {
        target_keys_per_split = 480 / split_scale;
    }
    const std::int32_t kMinSplits = 4 * split_scale;
    std::int32_t splits           = ceil_div_i32(window, target_keys_per_split);
    splits                        = std::max(kMinSplits, splits);
    return std::min(kGqaDecodeSplits, splits);
}

std::int32_t decode_active_splits(std::int32_t pos_value, DType kv_dtype) {
    return small_t_active_splits(1, pos_value, kv_dtype);
}

std::int32_t decode_kps(std::int32_t pos_value, DType kv_dtype) {
    const std::int32_t window = pos_value + 1;
    return ceil_div_i32(window, decode_active_splits(pos_value, kv_dtype));
}

const char* kv_dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "bf16";
    case DType::I8:
        return "int8";
    default:
        return "unknown";
    }
}

const char* small_t_ncu_kernel_regex(DType dtype) {
    return dtype == DType::I8 ? "gqa_attention_decode_i8_tiled_kernel"
                              : "gqa_attention_small_t_tc_partial_bf16_kernel";
}

const char* prefill_ncu_kernel_regex(DType dtype) {
    return dtype == DType::I8 ? "gqa_attention_prefill_i8_kernel"
                              : "gqa_attention_prefill_bf16_kernel";
}

double kv_cache_vector_bytes(DType dtype) {
    if (dtype == DType::I8) {
        return static_cast<double>(kHeadDim) + static_cast<double>(kHeadDim) /
                                                   static_cast<double>(kBenchKvQuantGroup) *
                                                   static_cast<double>(dtype_size(DType::FP16));
    }
    return static_cast<double>(kHeadDim) * static_cast<double>(dtype_size(DType::BF16));
}

double kv_cache_pair_bytes_per_head(DType dtype) { return 2.0 * kv_cache_vector_bytes(dtype); }

double kv_input_pair_bytes_per_head() {
    return 2.0 * static_cast<double>(kHeadDim) * static_cast<double>(dtype_size(DType::BF16));
}

struct DecodeBytes {
    std::size_t useful_kv = 0;
    std::size_t scratch   = 0;
    std::size_t total     = 0;
};

double append_prompt_logical_kv_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype);
double append_prompt_global_floor_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype);

DecodeBytes decode_bytes(std::int32_t pos_value, DType kv_dtype) {
    const auto window      = static_cast<std::size_t>(pos_value) + 1u;
    const auto split_count = static_cast<std::size_t>(decode_active_splits(pos_value, kv_dtype));

    const std::size_t k_cache_reads = static_cast<std::size_t>(
        static_cast<double>(window * kKVHeads) * kv_cache_vector_bytes(kv_dtype));
    const std::size_t v_cache_reads = k_cache_reads;
    const std::size_t new_kv_writes = static_cast<std::size_t>(
        static_cast<double>(kKVHeads) * kv_cache_pair_bytes_per_head(kv_dtype));
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

DecodeBytes small_t_stage_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype,
                                bool append) {
    const auto window      = static_cast<std::size_t>(context + tokens);
    const auto token_count = static_cast<std::size_t>(tokens);
    const auto split_count =
        static_cast<std::size_t>(small_t_active_splits(tokens, context, kv_dtype));

    const std::size_t k_cache_reads = static_cast<std::size_t>(
        static_cast<double>(window * kKVHeads) * kv_cache_vector_bytes(kv_dtype));
    const std::size_t v_cache_reads = k_cache_reads;
    const std::size_t new_kv_writes = static_cast<std::size_t>(
        static_cast<double>(token_count * kKVHeads) * kv_cache_pair_bytes_per_head(kv_dtype));
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
    bytes.total =
        bytes.useful_kv + (append ? new_kv_writes : 0u) + q_reads + output_writes + bytes.scratch;
    return bytes;
}

using VerifyRoute = ops::detail::GqaAttentionRoute;

VerifyRoute verify_route(std::int32_t tokens, std::int32_t context) {
    const auto visible = static_cast<std::uint32_t>(context + tokens);
    return ops::detail::gqa_attention_resolve_route(kQHeads, tokens, {visible, visible});
}

const char* verify_route_ncu_kernel_regex(VerifyRoute route, DType dtype) {
    return route == VerifyRoute::Prompt ? prefill_ncu_kernel_regex(dtype)
                                        : small_t_ncu_kernel_regex(dtype);
}

std::int32_t verify_route_chunks(std::int32_t tokens, VerifyRoute route) {
    return route == VerifyRoute::ChunkedSmallT ? ceil_div_i32(tokens, 6) : 1;
}

DecodeBytes verify_route_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype,
                               bool append) {
    const VerifyRoute route = verify_route(tokens, context);
    if (route == VerifyRoute::SmallT) {
        return small_t_stage_bytes(tokens, context, kv_dtype, append);
    }
    if (route == VerifyRoute::ChunkedSmallT) {
        DecodeBytes total;
        for (std::int32_t begin = 0; begin < tokens; begin += 6) {
            const DecodeBytes chunk =
                small_t_stage_bytes(std::min(6, tokens - begin), context + begin, kv_dtype, append);
            total.useful_kv += chunk.useful_kv;
            total.scratch += chunk.scratch;
            total.total += chunk.total;
        }
        return total;
    }

    DecodeBytes bytes;
    bytes.useful_kv =
        static_cast<std::size_t>(append_prompt_logical_kv_bytes(tokens, context, kv_dtype));
    bytes.total =
        static_cast<std::size_t>(append_prompt_global_floor_bytes(tokens, context, kv_dtype));
    if (!append) {
        const double token_count = static_cast<double>(tokens);
        bytes.total -= static_cast<std::size_t>(
            token_count * static_cast<double>(kKVHeads) *
            (kv_input_pair_bytes_per_head() + kv_cache_pair_bytes_per_head(kv_dtype)));
    }
    return bytes;
}

DecodeBytes append_small_t_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    return verify_route_bytes(tokens, context, kv_dtype, true);
}

DecodeBytes cached_small_t_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    return verify_route_bytes(tokens, context, kv_dtype, false);
}

struct KvAppendBytes {
    std::size_t input_read  = 0;
    std::size_t cache_write = 0;
    std::size_t total       = 0;
};

KvAppendBytes kv_append_bytes(std::int32_t tokens, DType kv_dtype) {
    const auto token_count = static_cast<std::size_t>(tokens);
    KvAppendBytes bytes;
    bytes.input_read  = static_cast<std::size_t>(static_cast<double>(token_count * kKVHeads) *
                                                 kv_input_pair_bytes_per_head());
    bytes.cache_write = static_cast<std::size_t>(static_cast<double>(token_count * kKVHeads) *
                                                 kv_cache_pair_bytes_per_head(kv_dtype));
    bytes.total       = bytes.input_read + bytes.cache_write;
    return bytes;
}

std::int32_t kv_append_ctas(std::int32_t tokens, DType kv_dtype) {
    if (kv_dtype == DType::I8) {
        constexpr std::int32_t kWarpsPerCta = 8;
        return ceil_div_i32(tokens * kKVHeads * (kHeadDim / kBenchKvQuantGroup), kWarpsPerCta);
    }
    constexpr std::int32_t kThreadsPerCta = 256;
    constexpr std::int32_t kVecElems      = 8;
    return ceil_div_i32(tokens * kKVHeads * (kHeadDim / kVecElems), kThreadsPerCta);
}

const char* kv_append_ncu_kernel_regex(DType dtype) {
    return dtype == DType::I8 ? "gqa_attention_prefill_fill_i8_kernel"
                              : "gqa_attention_prefill_fill_bf16_kernel";
}

const char* kv_append_control_ncu_kernel_regex(DType dtype) {
    return dtype == DType::I8 ? "bench_kv_append_i8_payload_control_kernel"
                              : "bench_kv_append_bf16_control_kernel";
}

std::int32_t append_prompt_q_blocks(std::int32_t tokens) {
    return ceil_div_i32(tokens, kPromptQBlock);
}

double append_prompt_key_sum(std::int32_t tokens, std::int32_t context) {
    const double t = static_cast<double>(tokens);
    const double c = static_cast<double>(context);
    return t * c + t * static_cast<double>(tokens + 1) * 0.5;
}

double append_prompt_qblock_key_rows(std::int32_t tokens, std::int32_t context) {
    double rows = 0.0;
    for (std::int32_t q0 = 0; q0 < tokens; q0 += kPromptQBlock) {
        const std::int32_t tile_rows = std::min(kPromptQBlock, tokens - q0);
        rows += static_cast<double>(context + q0 + tile_rows);
    }
    return rows;
}

std::int64_t append_prompt_key_tiles_per_head(std::int32_t tokens, std::int32_t context) {
    std::int64_t tiles = 0;
    for (std::int32_t q0 = 0; q0 < tokens; q0 += kPromptQBlock) {
        const std::int32_t tile_rows = std::min(kPromptQBlock, tokens - q0);
        const std::int64_t key_count =
            static_cast<std::int64_t>(context) + static_cast<std::int64_t>(q0) + tile_rows;
        tiles += ceil_div_i64(key_count, kPromptKBlock);
    }
    return tiles;
}

double append_prompt_useful_flops(std::int32_t tokens, std::int32_t context) {
    return 4.0 * static_cast<double>(kHeadDim) * static_cast<double>(kQHeads) *
           append_prompt_key_sum(tokens, context);
}

double append_prompt_logical_kv_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    return append_prompt_key_sum(tokens, context) * static_cast<double>(kKVHeads) *
           kv_cache_pair_bytes_per_head(kv_dtype);
}

double append_prompt_tile_kv_read_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    return append_prompt_qblock_key_rows(tokens, context) * static_cast<double>(kQHeads) *
           kv_cache_pair_bytes_per_head(kv_dtype);
}

double append_prompt_global_floor_bytes(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    const double token_count = static_cast<double>(tokens);
    const double q_bytes     = token_count * static_cast<double>(kQHeads) *
                           static_cast<double>(kHeadDim) *
                           static_cast<double>(sizeof(std::uint16_t));
    const double out_bytes = q_bytes;
    const double kv_input_bytes =
        token_count * static_cast<double>(kKVHeads) * kv_input_pair_bytes_per_head();
    const double kv_cache_write_bytes =
        token_count * static_cast<double>(kKVHeads) * kv_cache_pair_bytes_per_head(kv_dtype);
    return q_bytes + out_bytes + kv_input_bytes + kv_cache_write_bytes +
           append_prompt_tile_kv_read_bytes(tokens, context, kv_dtype);
}

std::size_t decode_workspace_bytes_for_pos(std::int32_t) {
    const auto split_count = static_cast<std::size_t>(kGqaDecodeSplits);
    const std::size_t partial_acc_bytes =
        static_cast<std::size_t>(kHeadDim) * kQHeads * split_count * sizeof(std::uint16_t);
    const std::size_t partial_m_bytes =
        static_cast<std::size_t>(kQHeads) * split_count * sizeof(float);
    const std::size_t partial_l_bytes =
        static_cast<std::size_t>(kQHeads) * split_count * sizeof(float);
    return partial_acc_bytes + partial_m_bytes + partial_l_bytes + align_overhead(3);
}

std::size_t small_t_workspace_bytes(std::int32_t tokens) {
    return ops::gqa_attention_workspace_bytes(kQHeads, tokens);
}

std::size_t decode_workspace_bytes(const std::vector<std::int32_t>& positions) {
    std::size_t bytes = 1u;
    for (const std::int32_t pos : positions) {
        bytes = std::max(bytes, decode_workspace_bytes_for_pos(pos));
    }
    return bytes;
}

std::size_t cache_arena_bytes(std::uint32_t layers, std::int32_t max_context, DType kv_dtype) {
    const auto padded_context = static_cast<std::size_t>(align_up_128(max_context));
    const std::size_t layer_elements =
        static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(kHeadDim) * padded_context;
    if (kv_dtype == DType::I8) {
        const std::size_t code_bytes     = layer_elements * dtype_size(DType::I8);
        const std::size_t scale_elements = static_cast<std::size_t>(kKVHeads) *
                                           static_cast<std::size_t>(kHeadDim / kBenchKvQuantGroup) *
                                           padded_context;
        const std::size_t scale_bytes = scale_elements * dtype_size(DType::FP16);
        return static_cast<std::size_t>(layers) *
                   (2u * (code_bytes + 255u) + 2u * (scale_bytes + 255u)) +
               4096u;
    }
    const std::size_t layer_bytes = layer_elements * dtype_size(DType::BF16);
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

__global__ void bench_stream_copy_kernel(const uint4* src, uint4* dst, std::size_t words) {
    const std::size_t start = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t step  = blockDim.x * static_cast<std::size_t>(gridDim.x);
    for (std::size_t i = start; i < words; i += step) { dst[i] = src[i]; }
}

__global__ void bench_kv_append_bf16_control_kernel(const uint4* k, const uint4* v, uint4* cache_k,
                                                    uint4* cache_v, std::size_t vectors) {
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= vectors) { return; }
    cache_k[idx] = k[idx];
    cache_v[idx] = v[idx];
}

__global__ void bench_kv_append_i8_payload_control_kernel(
    const std::uint32_t* k, const std::uint32_t* v, std::uint16_t* cache_k, std::uint16_t* cache_v,
    std::uint16_t* scale_k, std::uint16_t* scale_v, std::size_t groups) {
    const std::size_t group = blockIdx.x * 8u + threadIdx.x / 32u;
    const std::size_t lane  = threadIdx.x % 32u;
    if (group >= groups) { return; }

    const std::size_t pair = group * 32u + lane;
    const std::uint32_t pk = k[pair];
    const std::uint32_t pv = v[pair];
    cache_k[pair]          = static_cast<std::uint16_t>((pk & 0xffu) | ((pk >> 8u) & 0xff00u));
    cache_v[pair]          = static_cast<std::uint16_t>((pv & 0xffu) | ((pv >> 8u) & 0xff00u));
    if (lane == 0u) {
        scale_k[group] = 0x3c00u;
        scale_v[group] = 0x3c00u;
    }
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

void print_copy_ceiling_result(const char* tag, const Result& r, std::size_t payload_bytes,
                               std::int32_t tokens, std::int32_t context) {
    const double sec         = r.median_us * 1.0e-6;
    const double payload_gbs = (sec > 0.0) ? static_cast<double>(payload_bytes) / sec / 1.0e9 : 0.0;
    const double copy_gbs =
        (sec > 0.0) ? static_cast<double>(payload_bytes * 2u) / sec / 1.0e9 : 0.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  C_copy=%8.1f GB/s  "
                "payload_rate=%8.1f GB/s  bytes payload=%.2f MiB copy_total=%.2f MiB  "
                "T=%d context=%d\n",
                tag, r.median_us, r.min_us, r.p95_us, copy_gbs, payload_gbs,
                static_cast<double>(payload_bytes) / kMiB,
                static_cast<double>(payload_bytes * 2u) / kMiB, tokens, context);
}

void print_decode_result(const char* tag, const Result& r, const DecodeBytes& bytes,
                         std::int32_t pos_value, DType kv_dtype, std::uint32_t round_robin_layers,
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
                decode_active_splits(pos_value, kv_dtype), decode_kps(pos_value, kv_dtype),
                round_robin_layers, suffix);
}

void print_append_small_t_result(const char* tag, const Result& r, const DecodeBytes& bytes,
                                 std::int32_t tokens, std::int32_t context, DType kv_dtype,
                                 const char* suffix) {
    const double sec       = r.median_us * 1.0e-6;
    const double total_gbs = (sec > 0.0) ? static_cast<double>(bytes.total) / sec / 1.0e9 : 0.0;
    const double useful_kv_gbs =
        (sec > 0.0) ? static_cast<double>(bytes.useful_kv) / sec / 1.0e9 : 0.0;
    const double redundancy   = bytes.useful_kv > 0 ? static_cast<double>(bytes.total) /
                                                        static_cast<double>(bytes.useful_kv)
                                                    : 0.0;
    const VerifyRoute route   = verify_route(tokens, context);
    const std::int32_t chunks = verify_route_chunks(tokens, route);
    const std::int32_t final_chunk =
        route == VerifyRoute::ChunkedSmallT ? tokens - 6 * (chunks - 1) : tokens;
    const std::int32_t final_context =
        route == VerifyRoute::ChunkedSmallT ? context + 6 * (chunks - 1) : context;
    const std::int32_t splits = route == VerifyRoute::Prompt
                                    ? 0
                                    : small_t_active_splits(final_chunk, final_context, kv_dtype);
    constexpr double kMiB     = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  total_model=%8.1f GB/s  "
                "useful_kv=%8.1f GB/s  redundancy=%5.2f  bytes useful_kv=%.2f MiB "
                "scratch=%.2f MiB total=%.2f MiB  T=%d context=%d route=%s chunks=%d "
                "final_splits=%d%s\n",
                tag, r.median_us, r.min_us, r.p95_us, total_gbs, useful_kv_gbs, redundancy,
                static_cast<double>(bytes.useful_kv) / kMiB,
                static_cast<double>(bytes.scratch) / kMiB, static_cast<double>(bytes.total) / kMiB,
                tokens, context, ops::detail::gqa_attention_route_name(route), chunks, splits,
                suffix);
}

struct AppendPromptMetrics {
    std::int32_t tokens          = 0;
    std::int32_t context         = 0;
    std::int32_t end_context     = 0;
    DType kv_dtype               = DType::BF16;
    std::int32_t q_blocks        = 0;
    std::int64_t attention_ctas  = 0;
    std::int64_t key_tiles       = 0;
    int runs                     = 0;
    int inner_iters              = 1;
    const char* math_mode        = "bf16_qk_bf16_pv";
    const char* qk_mma_dtype     = "bf16";
    const char* pv_mma_dtype     = "bf16";
    std::int64_t qk_mma_count    = 0;
    std::int64_t pv_mma_count    = 0;
    double median_ms             = 0.0;
    double min_ms                = 0.0;
    double p95_ms                = 0.0;
    double mean_ms               = 0.0;
    double key_sum               = 0.0;
    double avg_keys_per_query    = 0.0;
    double qblock_key_rows       = 0.0;
    double tile_reuse_queries    = 0.0;
    double useful_flops          = 0.0;
    double logical_kv_bytes      = 0.0;
    double tile_kv_read_bytes    = 0.0;
    double global_floor_bytes    = 0.0;
    double tflops                = 0.0;
    double tflops_pct            = 0.0;
    double logical_kv_gbps       = 0.0;
    double tile_kv_read_gbps     = 0.0;
    double global_floor_gbps     = 0.0;
    double global_floor_gbps_pct = 0.0;
    double ns_per_key_query      = 0.0;
    double us_per_token          = 0.0;
    double roofline_tflops       = 0.0;
    double roofline_eff_pct      = 0.0;
    const char* bound            = "tc";
};

struct PrefillTimingOptions {
    int warmup      = 3;
    int repeat      = 10;
    int min_time_ms = 0;
};

struct KvAppendMetrics {
    std::int32_t tokens               = 0;
    std::int32_t context              = 0;
    DType kv_dtype                    = DType::BF16;
    std::int32_t ctas                 = 0;
    int runs                          = 0;
    int inner_iters                   = 1;
    std::size_t input_read_bytes      = 0;
    std::size_t cache_write_bytes     = 0;
    std::size_t total_bytes           = 0;
    double median_us                  = 0.0;
    double min_us                     = 0.0;
    double p95_us                     = 0.0;
    double mean_us                    = 0.0;
    double total_gbps                 = 0.0;
    double bandwidth_pct              = 0.0;
    double control_median_us          = 0.0;
    double control_interval_ratio_pct = 0.0;
};

KvAppendMetrics kv_append_metrics_from_result(std::int32_t tokens, std::int32_t context,
                                              DType kv_dtype, const Result& r) {
    const KvAppendBytes bytes = kv_append_bytes(tokens, kv_dtype);
    KvAppendMetrics m;
    m.tokens            = tokens;
    m.context           = context;
    m.kv_dtype          = kv_dtype;
    m.ctas              = kv_append_ctas(tokens, kv_dtype);
    m.runs              = r.n_runs;
    m.inner_iters       = r.inner_iters;
    m.input_read_bytes  = bytes.input_read;
    m.cache_write_bytes = bytes.cache_write;
    m.total_bytes       = bytes.total;
    m.median_us         = r.median_us;
    m.min_us            = r.min_us;
    m.p95_us            = r.p95_us;
    m.mean_us           = r.mean_us;
    const double sec    = r.median_us * 1.0e-6;
    if (sec > 0.0) { m.total_gbps = static_cast<double>(bytes.total) / sec / 1.0e9; }
    m.bandwidth_pct = m.total_gbps / kDramPeakGBs * 100.0;
    return m;
}

void print_kv_append_result(const KvAppendMetrics& m, const char* suffix = "") {
    constexpr double kKiB = 1024.0;
    std::printf("gqa_kv_append T=%-4d C=%-7d kv=%s median=%8.2f us  min=%8.2f us  "
                "p95=%8.2f us  payload=%8.1f GB/s (%5.2f%% of %.0f)  "
                "control=%8.2f us interval_ratio=%6.2f%%  read=%.2f KiB write=%.2f KiB "
                "total=%.2f KiB  CTAs=%d runs=%d inner=%d%s\n",
                m.tokens, m.context, kv_dtype_name(m.kv_dtype), m.median_us, m.min_us, m.p95_us,
                m.total_gbps, m.bandwidth_pct, kDramPeakGBs, m.control_median_us,
                m.control_interval_ratio_pct, static_cast<double>(m.input_read_bytes) / kKiB,
                static_cast<double>(m.cache_write_bytes) / kKiB,
                static_cast<double>(m.total_bytes) / kKiB, m.ctas, m.runs, m.inner_iters, suffix);
}

AppendPromptMetrics append_prompt_metrics_from_result(std::int32_t tokens, std::int32_t context,
                                                      DType kv_dtype, const Result& r) {
    AppendPromptMetrics m;
    m.tokens         = tokens;
    m.context        = context;
    m.end_context    = context + tokens;
    m.kv_dtype       = kv_dtype;
    m.q_blocks       = append_prompt_q_blocks(tokens);
    m.attention_ctas = static_cast<std::int64_t>(m.q_blocks) * kQHeads;
    m.key_tiles      = append_prompt_key_tiles_per_head(tokens, context) * kQHeads;
    m.qk_mma_count   = m.key_tiles * (kv_dtype == DType::I8 ? 256 : 512);
    m.pv_mma_count   = m.key_tiles * 512;
    m.runs           = r.n_runs;
    m.inner_iters    = r.inner_iters;
    if (kv_dtype == DType::I8) {
        m.math_mode    = "s8_qk_f16_pv";
        m.qk_mma_dtype = "s8";
        m.pv_mma_dtype = "f16";
    }
    m.median_ms          = r.median_us * 1.0e-3;
    m.min_ms             = r.min_us * 1.0e-3;
    m.p95_ms             = r.p95_us * 1.0e-3;
    m.mean_ms            = r.mean_us * 1.0e-3;
    m.key_sum            = append_prompt_key_sum(tokens, context);
    m.avg_keys_per_query = m.key_sum / static_cast<double>(tokens);
    m.qblock_key_rows    = append_prompt_qblock_key_rows(tokens, context);
    m.tile_reuse_queries = (m.qblock_key_rows > 0.0) ? m.key_sum / m.qblock_key_rows : 0.0;
    m.useful_flops       = append_prompt_useful_flops(tokens, context);
    m.logical_kv_bytes   = append_prompt_logical_kv_bytes(tokens, context, kv_dtype);
    m.tile_kv_read_bytes = append_prompt_tile_kv_read_bytes(tokens, context, kv_dtype);
    m.global_floor_bytes = append_prompt_global_floor_bytes(tokens, context, kv_dtype);

    const double sec = r.median_us * 1.0e-6;
    if (sec > 0.0) {
        m.tflops            = m.useful_flops / sec / 1.0e12;
        m.logical_kv_gbps   = m.logical_kv_bytes / sec / 1.0e9;
        m.tile_kv_read_gbps = m.tile_kv_read_bytes / sec / 1.0e9;
        m.global_floor_gbps = m.global_floor_bytes / sec / 1.0e9;
        m.ns_per_key_query  = r.median_us * 1000.0 / m.key_sum;
    }
    m.us_per_token          = r.median_us / static_cast<double>(tokens);
    m.tflops_pct            = m.tflops / kDenseTcPeakTflops * 100.0;
    m.global_floor_gbps_pct = m.global_floor_gbps / kDramPeakGBs * 100.0;

    const double intensity =
        m.global_floor_bytes > 0.0 ? m.useful_flops / m.global_floor_bytes : 0.0;
    const double dram_roof_tflops = kDramPeakGBs * intensity / 1000.0;
    m.roofline_tflops             = std::min(kDenseTcPeakTflops, dram_roof_tflops);
    m.bound                       = (kDenseTcPeakTflops <= dram_roof_tflops) ? "tc" : "tile";
    m.roofline_eff_pct = (m.roofline_tflops > 0.0) ? (m.tflops / m.roofline_tflops * 100.0) : 0.0;
    return m;
}

void print_append_prompt_result(const char* tag, const AppendPromptMetrics& m) {
    constexpr double kMiB = 1024.0 * 1024.0;
    std::printf("%-38s T=%-6d C=%-7d kv=%s median=%9.3f ms  min=%9.3f ms  "
                "p95=%9.3f ms  useful=%9.2f TFLOP/s  tc=%6.2f%% of %.1f  "
                "tile_floor=%8.1f GB/s (%5.2f%% of %.0f)  logical_kv=%8.1f GB/s  "
                "ns/key=%6.3f  q_blocks=%d key_tiles=%lld tile_kv=%.2f MiB  "
                "bound=%s roofline_eff=%6.2f%% runs=%d inner=%d\n",
                tag, m.tokens, m.context, kv_dtype_name(m.kv_dtype), m.median_ms, m.min_ms,
                m.p95_ms, m.tflops, m.tflops_pct, kDenseTcPeakTflops, m.global_floor_gbps,
                m.global_floor_gbps_pct, kDramPeakGBs, m.logical_kv_gbps, m.ns_per_key_query,
                m.q_blocks, static_cast<long long>(m.key_tiles), m.tile_kv_read_bytes / kMiB,
                m.bound, m.roofline_eff_pct, m.runs, m.inner_iters);
}

void run_decode(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k, const Tensor& v,
                Tensor& out, std::int32_t pos_value, std::uint32_t round_robin_layers) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});

    const DecodeBytes bytes  = decode_bytes(pos_value, kv.dtype);
    std::uint32_t next_layer = 0;

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            const int layer = static_cast<int>(next_layer);
            next_layer      = (next_layer + 1u) % round_robin_layers;
            ops::gqa_attention(q, k, v, pos, kScale, kv.layer_view(layer),
                               exact_envelope(static_cast<std::uint32_t>(pos_value + 1)), ws, out,
                               s);
        },
        static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention decode combined pos=%d kv=%s", pos_value,
                  kv_dtype_name(kv.dtype));
    print_decode_result(tag, r, bytes, pos_value, kv.dtype, round_robin_layers,
                        (round_robin_layers == 1u) ? " hot_cache_info" : "");
}

void run_profile_once(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k,
                      const Tensor& v, Tensor& out, std::int32_t pos_value, DBuf* cold_cache) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});
    const DecodeBytes bytes = decode_bytes(pos_value, kv.dtype);

    cudaStream_t stream = nullptr;
    if (cold_cache != nullptr) {
        const Result r = bench_cold_cache_loop(
            [&](cudaStream_t s) {
                ops::gqa_attention(q, k, v, pos, kScale, kv.layer_view(0),
                                   exact_envelope(static_cast<std::uint32_t>(pos_value + 1)), ws,
                                   out, s);
            },
            *cold_cache, static_cast<double>(bytes.total));

        char tag[96];
        std::snprintf(tag, sizeof(tag), "PROFILE_COLD gqa_attention decode pos=%d", pos_value);
        print_decode_result(tag, r, bytes, pos_value, kv.dtype, 1u, " cold_cache");
        std::printf("PROFILE_COLD_METADATA pos=%d kv_dtype=%s splits=%d kps=%d "
                    "cold_cache_bytes=%zu "
                    "useful_kv_bytes=%zu scratch_bytes=%zu total_modeled_bytes=%zu repeats=%d\n",
                    pos_value, kv_dtype_name(kv.dtype), decode_active_splits(pos_value, kv.dtype),
                    decode_kps(pos_value, kv.dtype), cold_cache->bytes, bytes.useful_kv,
                    bytes.scratch, bytes.total, r.n_runs);
        return;
    }

    ops::gqa_attention(q, k, v, pos, kScale, kv.layer_view(0),
                       exact_envelope(static_cast<std::uint32_t>(pos_value + 1)), ws, out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("PROFILE_ONCE gqa_attention decode combined pos=%d kv_dtype=%s splits=%d kps=%d "
                "useful_kv_bytes=%zu scratch_bytes=%zu total_modeled_bytes=%zu\n",
                pos_value, kv_dtype_name(kv.dtype), decode_active_splits(pos_value, kv.dtype),
                decode_kps(pos_value, kv.dtype), bytes.useful_kv, bytes.scratch, bytes.total);
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

    const DecodeBytes bytes = append_small_t_bytes(tokens, context, kv.dtype);
    const Result r          = bench_loop(
        [&](cudaStream_t s) {
            ops::gqa_attention(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                                        exact_envelope(static_cast<std::uint32_t>(context + tokens)), ws,
                                        tout, s);
        },
        static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention append-small-T kv=%s", kv_dtype_name(kv.dtype));
    print_append_small_t_result(tag, r, bytes, tokens, context, kv.dtype, "");
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

    const DecodeBytes bytes = append_small_t_bytes(tokens, context, kv.dtype);
    cudaStream_t stream     = nullptr;
    if (cold_cache != nullptr) {
        const Result r = bench_cold_cache_loop(
            [&](cudaStream_t s) {
                ops::gqa_attention(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                                   exact_envelope(static_cast<std::uint32_t>(context + tokens)), ws,
                                   tout, s);
            },
            *cold_cache, static_cast<double>(bytes.total));
        char tag[96];
        std::snprintf(tag, sizeof(tag), "PROFILE_COLD gqa_attention append-small-T kv=%s",
                      kv_dtype_name(kv.dtype));
        print_append_small_t_result(tag, r, bytes, tokens, context, kv.dtype, " cold_cache");
        const VerifyRoute route = verify_route(tokens, context);
        std::printf("PROFILE_COLD_METADATA mode=append-small-t T=%d context=%d kv_dtype=%s "
                    "route=%s chunks=%d "
                    "cold_cache_bytes=%zu useful_kv_bytes=%zu scratch_bytes=%zu "
                    "total_modeled_bytes=%zu redundancy=%.6f repeats=%d "
                    "ncu_kernel_regex='%s'\n",
                    tokens, context, kv_dtype_name(kv.dtype),
                    ops::detail::gqa_attention_route_name(route),
                    verify_route_chunks(tokens, route), cold_cache->bytes, bytes.useful_kv,
                    bytes.scratch, bytes.total,
                    bytes.useful_kv > 0
                        ? static_cast<double>(bytes.total) / static_cast<double>(bytes.useful_kv)
                        : 0.0,
                    r.n_runs, verify_route_ncu_kernel_regex(route, kv.dtype));
        return;
    }

    ops::gqa_attention(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                       exact_envelope(static_cast<std::uint32_t>(context + tokens)), ws, tout,
                       stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const VerifyRoute route = verify_route(tokens, context);
    std::printf("PROFILE_ONCE gqa_attention append-small-T T=%d context=%d kv_dtype=%s route=%s "
                "chunks=%d "
                "useful_kv_bytes=%zu scratch_bytes=%zu total_model_bytes=%zu redundancy=%.6f "
                "ncu_kernel_regex='%s'\n",
                tokens, context, kv_dtype_name(kv.dtype),
                ops::detail::gqa_attention_route_name(route), verify_route_chunks(tokens, route),
                bytes.useful_kv, bytes.scratch, bytes.total,
                bytes.useful_kv > 0
                    ? static_cast<double>(bytes.total) / static_cast<double>(bytes.useful_kv)
                    : 0.0,
                verify_route_ncu_kernel_regex(route, kv.dtype));
}

void run_cached_small_t(KVCache& kv, std::int32_t tokens, std::int32_t context) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf pos = make_i32_sequence(context, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(small_t_workspace_bytes(tokens));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const DecodeBytes bytes = cached_small_t_bytes(tokens, context, kv.dtype);
    const Result r          = bench_loop(
        [&](cudaStream_t s) {
            ops::gqa_attention_cached(tq, tpos, kScale, kv.layer_view(0),
                                               exact_envelope(static_cast<std::uint32_t>(context + tokens)),
                                               ws, tout, s);
        },
        static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention cached-small-T kv=%s", kv_dtype_name(kv.dtype));
    print_append_small_t_result(tag, r, bytes, tokens, context, kv.dtype, "");
}

void run_cached_small_t_profile_once(KVCache& kv, std::int32_t tokens, std::int32_t context,
                                     DBuf* cold_cache) {
    const std::size_t qn = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                           static_cast<std::size_t>(tokens);
    DBuf q   = make_bf16(qn);
    DBuf pos = make_i32_sequence(context, tokens);
    DBuf out = make_zeros(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(small_t_workspace_bytes(tokens));

    Tensor tq(q.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    const DecodeBytes bytes = cached_small_t_bytes(tokens, context, kv.dtype);

    const auto launch = [&](cudaStream_t s) {
        ops::gqa_attention_cached(tq, tpos, kScale, kv.layer_view(0),
                                  exact_envelope(static_cast<std::uint32_t>(context + tokens)), ws,
                                  tout, s);
    };
    if (cold_cache != nullptr) {
        const Result r =
            bench_cold_cache_loop(launch, *cold_cache, static_cast<double>(bytes.total));
        char tag[96];
        std::snprintf(tag, sizeof(tag), "PROFILE_COLD gqa_attention cached-small-T kv=%s",
                      kv_dtype_name(kv.dtype));
        print_append_small_t_result(tag, r, bytes, tokens, context, kv.dtype, " cold_cache");
        const VerifyRoute route = verify_route(tokens, context);
        std::printf(
            "PROFILE_COLD_METADATA mode=cached-small-t T=%d context=%d kv_dtype=%s "
            "route=%s chunks=%d cold_cache_bytes=%zu useful_kv_bytes=%zu scratch_bytes=%zu "
            "total_modeled_bytes=%zu repeats=%d ncu_kernel_regex='%s'\n",
            tokens, context, kv_dtype_name(kv.dtype), ops::detail::gqa_attention_route_name(route),
            verify_route_chunks(tokens, route), cold_cache->bytes, bytes.useful_kv, bytes.scratch,
            bytes.total, r.n_runs, verify_route_ncu_kernel_regex(route, kv.dtype));
        return;
    }

    launch(nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const VerifyRoute route = verify_route(tokens, context);
    std::printf("PROFILE_ONCE gqa_attention cached-small-T T=%d context=%d kv_dtype=%s "
                "route=%s chunks=%d useful_kv_bytes=%zu scratch_bytes=%zu total_model_bytes=%zu "
                "ncu_kernel_regex='%s'\n",
                tokens, context, kv_dtype_name(kv.dtype),
                ops::detail::gqa_attention_route_name(route), verify_route_chunks(tokens, route),
                bytes.useful_kv, bytes.scratch, bytes.total,
                verify_route_ncu_kernel_regex(route, kv.dtype));
}

void launch_kv_append_control(DType kv_dtype, const DBuf& k, const DBuf& v, DBuf& control_k,
                              DBuf& control_v, std::int32_t tokens, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int grid       = kv_append_ctas(tokens, kv_dtype);
    if (kv_dtype == DType::I8) {
        const std::size_t groups =
            static_cast<std::size_t>(tokens) * kKVHeads * (kHeadDim / kBenchKvQuantGroup);
        const std::size_t code_pairs = static_cast<std::size_t>(tokens) * kKVHeads * kHeadDim / 2u;
        auto* control_k_data         = static_cast<std::uint16_t*>(control_k.p);
        auto* control_v_data         = static_cast<std::uint16_t*>(control_v.p);
        bench_kv_append_i8_payload_control_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const std::uint32_t*>(k.p), static_cast<const std::uint32_t*>(v.p),
            control_k_data, control_v_data, control_k_data + code_pairs,
            control_v_data + code_pairs, groups);
    } else {
        const std::size_t vectors = k.bytes / sizeof(uint4);
        bench_kv_append_bf16_control_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const uint4*>(k.p), static_cast<const uint4*>(v.p),
            static_cast<uint4*>(control_k.p), static_cast<uint4*>(control_v.p), vectors);
    }
    CUDA_CHECK(cudaGetLastError());
}

KvAppendMetrics run_kv_append(KVCache& kv, std::int32_t tokens, std::int32_t context,
                              const PrefillTimingOptions& timing) {
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(context, tokens);
    const std::size_t control_bytes =
        kv.dtype == DType::I8 ? kv_append_bytes(tokens, kv.dtype).cache_write / 2u : k.bytes;
    DBuf control_k = make_zeros(control_bytes);
    DBuf control_v = make_zeros(control_bytes);
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});

    const KvAppendBytes bytes = kv_append_bytes(tokens, kv.dtype);
    const Result r            = bench_loop(
        [&](cudaStream_t s) { ops::gqa_kv_append(tk, tv, tpos, kv.layer_view(0), s); },
        static_cast<double>(bytes.total), timing.warmup, timing.repeat, timing.min_time_ms);
    const Result control = bench_loop(
        [&](cudaStream_t s) {
            launch_kv_append_control(kv.dtype, k, v, control_k, control_v, tokens, s);
        },
        static_cast<double>(bytes.total), timing.warmup, timing.repeat, timing.min_time_ms);
    KvAppendMetrics metrics   = kv_append_metrics_from_result(tokens, context, kv.dtype, r);
    metrics.control_median_us = control.median_us;
    metrics.control_interval_ratio_pct =
        r.median_us > 0.0 ? control.median_us / r.median_us * 100.0 : 0.0;
    print_kv_append_result(metrics);
    return metrics;
}

void run_kv_append_profile_once(KVCache& kv, std::int32_t tokens, std::int32_t context,
                                DBuf* cold_cache) {
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) *
                            static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(tokens);
    DBuf k   = make_bf16(kvn);
    DBuf v   = make_bf16(kvn);
    DBuf pos = make_i32_sequence(context, tokens);
    const std::size_t control_bytes =
        kv.dtype == DType::I8 ? kv_append_bytes(tokens, kv.dtype).cache_write / 2u : k.bytes;
    DBuf control_k = make_zeros(control_bytes);
    DBuf control_v = make_zeros(control_bytes);
    Tensor tk(k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(pos.p, DType::I32, {tokens});
    const KvAppendBytes bytes = kv_append_bytes(tokens, kv.dtype);

    cudaStream_t stream = nullptr;
    if (cold_cache != nullptr) {
        const Result r = bench_cold_cache_loop(
            [&](cudaStream_t s) { ops::gqa_kv_append(tk, tv, tpos, kv.layer_view(0), s); },
            *cold_cache, static_cast<double>(bytes.total));
        const KvAppendMetrics metrics = kv_append_metrics_from_result(tokens, context, kv.dtype, r);
        print_kv_append_result(metrics, " cold_cache");
        std::printf("PROFILE_COLD_METADATA mode=kv-append T=%d context=%d kv_dtype=%s CTAs=%d "
                    "cold_cache_bytes=%zu input_read_bytes=%zu cache_write_bytes=%zu "
                    "total_modeled_bytes=%zu repeats=%d ncu_kernel_regex='%s'\n",
                    tokens, context, kv_dtype_name(kv.dtype), metrics.ctas, cold_cache->bytes,
                    bytes.input_read, bytes.cache_write, bytes.total, r.n_runs,
                    kv_append_ncu_kernel_regex(kv.dtype));
        return;
    }

    ops::gqa_kv_append(tk, tv, tpos, kv.layer_view(0), stream);
    launch_kv_append_control(kv.dtype, k, v, control_k, control_v, tokens, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("PROFILE_ONCE mode=kv-append T=%d context=%d kv_dtype=%s CTAs=%d "
                "input_read_bytes=%zu cache_write_bytes=%zu total_model_bytes=%zu "
                "ncu_kernel_regex='%s' control_ncu_kernel_regex='%s'\n",
                tokens, context, kv_dtype_name(kv.dtype), kv_append_ctas(tokens, kv.dtype),
                bytes.input_read, bytes.cache_write, bytes.total,
                kv_append_ncu_kernel_regex(kv.dtype), kv_append_control_ncu_kernel_regex(kv.dtype));
}

void run_copy_ceiling(std::int32_t tokens, std::int32_t context, DType kv_dtype) {
    const DecodeBytes bytes = append_small_t_bytes(tokens, context, kv_dtype);
    DBuf src                = make_zeros(bytes.useful_kv);
    DBuf dst                = make_zeros(bytes.useful_kv);
    DBuf cold_cache         = make_zeros(kColdCacheBytes);

    const std::size_t words = bytes.useful_kv / sizeof(uint4);
    constexpr int kBlock    = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(4096u, ceil_div_size(words, kBlock)));

    auto launch = [&](cudaStream_t s) {
        bench_stream_copy_kernel<<<grid, kBlock, 0, s>>>(static_cast<const uint4*>(src.p),
                                                         static_cast<uint4*>(dst.p), words);
        CUDA_CHECK(cudaGetLastError());
    };

    char hot_tag[96];
    std::snprintf(hot_tag, sizeof(hot_tag), "gqa_attention copy ceiling hot kv=%s",
                  kv_dtype_name(kv_dtype));
    const Result hot = bench_loop(launch, static_cast<double>(bytes.useful_kv * 2u));
    print_copy_ceiling_result(hot_tag, hot, bytes.useful_kv, tokens, context);

    const Result cold =
        bench_cold_cache_loop(launch, cold_cache, static_cast<double>(bytes.useful_kv * 2u));
    char cold_tag[96];
    std::snprintf(cold_tag, sizeof(cold_tag), "gqa_attention copy ceiling cold kv=%s",
                  kv_dtype_name(kv_dtype));
    print_copy_ceiling_result(cold_tag, cold, bytes.useful_kv, tokens, context);
}

AppendPromptMetrics run_append_prompt_baseline(KVCache& kv, std::int32_t tokens,
                                               std::int32_t context,
                                               const PrefillTimingOptions& timing) {
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

    const double bytes = append_prompt_global_floor_bytes(tokens, context, kv.dtype);
    const Result r     = bench_loop(
        [&](cudaStream_t s) {
            ops::detail::gqa_attention_prompt_launch(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                                                         tout, s);
        },
        bytes, timing.warmup, timing.repeat, timing.min_time_ms);

    AppendPromptMetrics metrics = append_prompt_metrics_from_result(tokens, context, kv.dtype, r);
    print_append_prompt_result("gqa_attention append-prompt", metrics);
    return metrics;
}

AppendPromptMetrics run_append_prompt_attention_only(KVCache& kv, std::int32_t tokens,
                                                     std::int32_t context,
                                                     const PrefillTimingOptions& timing) {
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

    cudaStream_t stream = nullptr;
    ops::detail::gqa_attention_prompt_launch(tq, tk, tv, tpos, kScale, kv.layer_view(0), tout,
                                             stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double bytes = append_prompt_global_floor_bytes(tokens, context, kv.dtype);
    const Result r     = bench_loop(
        [&](cudaStream_t s) {
            ops::detail::gqa_attention_prompt_attention_launch(tq, tpos, kScale, kv.layer_view(0),
                                                                   tout, s);
        },
        bytes, timing.warmup, timing.repeat, timing.min_time_ms);

    AppendPromptMetrics metrics = append_prompt_metrics_from_result(tokens, context, kv.dtype, r);
    print_append_prompt_result("gqa_attention append-attn-only", metrics);
    return metrics;
}

double prefill_useful_flops(std::int32_t tokens) {
    return 2.0 * static_cast<double>(kHeadDim) * static_cast<double>(kQHeads) *
           static_cast<double>(tokens) * static_cast<double>(tokens + 1);
}

double prefill_model_floor_bytes(std::int32_t tokens, DType kv_dtype) {
    const double q_bytes = static_cast<double>(tokens) * static_cast<double>(kQHeads) *
                           static_cast<double>(kHeadDim) *
                           static_cast<double>(sizeof(std::uint16_t));
    const double out_bytes      = q_bytes;
    const double kv_input_bytes = static_cast<double>(tokens) * static_cast<double>(kKVHeads) *
                                  kv_input_pair_bytes_per_head();
    const double kv_cache_write_bytes = static_cast<double>(tokens) *
                                        static_cast<double>(kKVHeads) *
                                        kv_cache_pair_bytes_per_head(kv_dtype);
    return q_bytes + out_bytes + kv_input_bytes + kv_cache_write_bytes;
}

struct PrefillMetrics {
    std::int32_t tokens      = 0;
    DType kv_dtype           = DType::BF16;
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

PrefillMetrics prefill_metrics_from_result(std::int32_t tokens, DType kv_dtype, const Result& r) {
    PrefillMetrics m;
    m.tokens            = tokens;
    m.kv_dtype          = kv_dtype;
    m.runs              = r.n_runs;
    m.inner_iters       = r.inner_iters;
    m.median_ms         = r.median_us * 1.0e-3;
    m.min_ms            = r.min_us * 1.0e-3;
    m.p95_ms            = r.p95_us * 1.0e-3;
    m.mean_ms           = r.mean_us * 1.0e-3;
    m.useful_flops      = prefill_useful_flops(tokens);
    m.model_floor_bytes = prefill_model_floor_bytes(tokens, kv_dtype);

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
    std::printf("gqa_attention prefill T=%-6d kv=%s median=%9.3f ms  min=%9.3f ms  p95=%9.3f ms  "
                "useful=%9.2f TFLOP/s  tc=%6.2f%% of %.1f  model_floor=%8.1f GB/s "
                "(%5.2f%% of %.0f)  bound=%s  roofline_eff=%6.2f%%  runs=%d inner=%d\n",
                m.tokens, kv_dtype_name(m.kv_dtype), m.median_ms, m.min_ms, m.p95_ms, m.tflops,
                m.tflops_pct, kDenseTcPeakTflops, m.gbps_model, m.gbps_model_pct, kDramPeakGBs,
                m.bound, m.roofline_eff_pct, m.runs, m.inner_iters);
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
            ops::gqa_attention(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                               exact_envelope(static_cast<std::uint32_t>(tokens)), ws, tout, s);
        },
        prefill_model_floor_bytes(tokens, kv.dtype), timing.warmup, timing.repeat,
        timing.min_time_ms);

    PrefillMetrics metrics = prefill_metrics_from_result(tokens, kv.dtype, r);
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
    ops::gqa_attention(tq, tk, tv, tpos, kScale, kv.layer_view(0),
                       exact_envelope(static_cast<std::uint32_t>(tokens)), ws, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("PROFILE_ONCE gqa_attention prefill T=%d kv_dtype=%s useful_flops=%.0f "
                "model_floor_bytes=%.0f tc_peak_tflops=%.1f dram_peak_gbps=%.0f "
                "ncu_kernel_regex='%s'\n",
                tokens, kv_dtype_name(kv.dtype), prefill_useful_flops(tokens),
                prefill_model_floor_bytes(tokens, kv.dtype), kDenseTcPeakTflops, kDramPeakGBs,
                prefill_ncu_kernel_regex(kv.dtype));
}

std::string json_number(double value) {
    if (!std::isfinite(value)) { return "null"; }
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string format_kv_append_csv(const std::vector<KvAppendMetrics>& results) {
    std::ostringstream out;
    out << "T,context,kv_dtype,ctas,input_read_bytes,cache_write_bytes,total_bytes,"
           "median_us,min_us,p95_us,mean_us,total_gbps,bandwidth_pct,control_median_us,"
           "control_interval_ratio_pct,runs,inner_iters\n";
    for (const KvAppendMetrics& m : results) {
        out << m.tokens << ',' << m.context << ',' << kv_dtype_name(m.kv_dtype) << ',' << m.ctas
            << ',' << m.input_read_bytes << ',' << m.cache_write_bytes << ',' << m.total_bytes
            << ',' << json_number(m.median_us) << ',' << json_number(m.min_us) << ','
            << json_number(m.p95_us) << ',' << json_number(m.mean_us) << ','
            << json_number(m.total_gbps) << ',' << json_number(m.bandwidth_pct) << ','
            << json_number(m.control_median_us) << ',' << json_number(m.control_interval_ratio_pct)
            << ',' << m.runs << ',' << m.inner_iters << '\n';
    }
    return out.str();
}

std::string format_kv_append_json(const std::vector<KvAppendMetrics>& results,
                                  const PrefillTimingOptions& timing) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact_type\": \"ninfer_gqa_kv_append_bench\",\n"
        << "  \"geometry\": \"" << kGeometryName << "\",\n"
        << "  \"q_heads\": " << kQHeads << ",\n"
        << "  \"kv_heads\": " << kKVHeads << ",\n"
        << "  \"head_dim\": " << kHeadDim << ",\n"
        << "  \"dram_peak_gbps\": " << json_number(kDramPeakGBs) << ",\n"
        << "  \"bytes_definition\": \"BF16 K/V input reads plus encoded K/V cache writes; "
           "control-position traffic excluded\",\n"
        << "  \"control_definition\": \"same-grid, same-payload K/V transfer; the INT8 "
           "control packs bytes and writes scales but omits absmax, scaling, and rounding\",\n"
        << "  \"timing\": {\"warmup\": " << timing.warmup << ", \"repeat\": " << timing.repeat
        << ", \"min_time_ms\": " << timing.min_time_ms << "},\n"
        << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const KvAppendMetrics& m = results[i];
        out << "    {\"T\": " << m.tokens << ", \"context\": " << m.context << ", \"kv_dtype\": \""
            << kv_dtype_name(m.kv_dtype) << "\", \"ctas\": " << m.ctas
            << ", \"input_read_bytes\": " << m.input_read_bytes
            << ", \"cache_write_bytes\": " << m.cache_write_bytes
            << ", \"total_bytes\": " << m.total_bytes
            << ", \"median_us\": " << json_number(m.median_us)
            << ", \"min_us\": " << json_number(m.min_us)
            << ", \"p95_us\": " << json_number(m.p95_us)
            << ", \"mean_us\": " << json_number(m.mean_us)
            << ", \"total_gbps\": " << json_number(m.total_gbps)
            << ", \"bandwidth_pct\": " << json_number(m.bandwidth_pct)
            << ", \"control_median_us\": " << json_number(m.control_median_us)
            << ", \"control_interval_ratio_pct\": " << json_number(m.control_interval_ratio_pct)
            << ", \"runs\": " << m.runs << ", \"inner_iters\": " << m.inner_iters << "}"
            << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n"
        << "}\n";
    return out.str();
}

std::string format_prefill_csv(const std::vector<PrefillMetrics>& results) {
    std::ostringstream out;
    out << "T,kv_dtype,ms,tflops,tflops_pct,gbps_model,gbps_model_pct,gbps_dram,gbps_dram_pct,"
           "gbps_pct,bound,"
           "roofline_tflops,roofline_eff_pct,model_floor_bytes,useful_flops,"
           "median_ms,min_ms,p95_ms,mean_ms,runs,inner_iters\n";
    for (const PrefillMetrics& m : results) {
        out << m.tokens << ',' << kv_dtype_name(m.kv_dtype) << ',' << json_number(m.median_ms)
            << ',' << json_number(m.tflops) << ',' << json_number(m.tflops_pct) << ','
            << json_number(m.gbps_model) << ',' << json_number(m.gbps_model_pct) << ",,,"
            << json_number(m.gbps_model_pct) << ',' << m.bound << ','
            << json_number(m.roofline_tflops) << ',' << json_number(m.roofline_eff_pct) << ','
            << json_number(m.model_floor_bytes) << ',' << json_number(m.useful_flops) << ','
            << json_number(m.median_ms) << ',' << json_number(m.min_ms) << ','
            << json_number(m.p95_ms) << ',' << json_number(m.mean_ms) << ',' << m.runs << ','
            << m.inner_iters << '\n';
    }
    return out.str();
}

std::string format_prefill_json(const std::vector<PrefillMetrics>& results,
                                const PrefillTimingOptions& timing) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact_type\": \"ninfer_gqa_attention_prefill_bench\",\n"
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
            << ", \"kv_dtype\": \"" << kv_dtype_name(m.kv_dtype) << "\""
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

std::string format_append_prompt_csv(const std::vector<AppendPromptMetrics>& results) {
    std::ostringstream out;
    out << "T,context,end_context,kv_dtype,math_mode,qk_mma_dtype,pv_mma_dtype,qk_mma_count,"
           "pv_mma_count,ms,tflops,tflops_pct,global_floor_gbps,"
           "global_floor_gbps_pct,tile_kv_read_gbps,logical_kv_gbps,avg_keys_per_query,"
           "ns_per_key_query,us_per_token,q_blocks,attention_ctas,key_tiles,tile_reuse_queries,"
           "bound,roofline_tflops,roofline_eff_pct,global_floor_bytes,tile_kv_read_bytes,"
           "logical_kv_bytes,useful_flops,key_sum,qblock_key_rows,median_ms,min_ms,p95_ms,"
           "mean_ms,runs,inner_iters\n";
    for (const AppendPromptMetrics& m : results) {
        out << m.tokens << ',' << m.context << ',' << m.end_context << ','
            << kv_dtype_name(m.kv_dtype) << ',' << m.math_mode << ',' << m.qk_mma_dtype << ','
            << m.pv_mma_dtype << ',' << m.qk_mma_count << ',' << m.pv_mma_count << ','
            << json_number(m.median_ms) << ',' << json_number(m.tflops) << ','
            << json_number(m.tflops_pct) << ',' << json_number(m.global_floor_gbps) << ','
            << json_number(m.global_floor_gbps_pct) << ',' << json_number(m.tile_kv_read_gbps)
            << ',' << json_number(m.logical_kv_gbps) << ',' << json_number(m.avg_keys_per_query)
            << ',' << json_number(m.ns_per_key_query) << ',' << json_number(m.us_per_token) << ','
            << m.q_blocks << ',' << m.attention_ctas << ',' << m.key_tiles << ','
            << json_number(m.tile_reuse_queries) << ',' << m.bound << ','
            << json_number(m.roofline_tflops) << ',' << json_number(m.roofline_eff_pct) << ','
            << json_number(m.global_floor_bytes) << ',' << json_number(m.tile_kv_read_bytes) << ','
            << json_number(m.logical_kv_bytes) << ',' << json_number(m.useful_flops) << ','
            << json_number(m.key_sum) << ',' << json_number(m.qblock_key_rows) << ','
            << json_number(m.median_ms) << ',' << json_number(m.min_ms) << ','
            << json_number(m.p95_ms) << ',' << json_number(m.mean_ms) << ',' << m.runs << ','
            << m.inner_iters << '\n';
    }
    return out.str();
}

std::string format_append_prompt_json(const std::vector<AppendPromptMetrics>& results,
                                      const PrefillTimingOptions& timing) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 2,\n"
        << "  \"artifact_type\": \"ninfer_gqa_attention_append_prompt_bench\",\n"
        << "  \"tc_peak_tflops\": " << json_number(kDenseTcPeakTflops) << ",\n"
        << "  \"dram_peak_gbps\": " << json_number(kDramPeakGBs) << ",\n"
        << "  \"q_block\": " << kPromptQBlock << ",\n"
        << "  \"k_block\": " << kPromptKBlock << ",\n"
        << "  \"flops_definition\": \"useful_causal_4*d*Hq*sum_{i=0}^{T-1}(context+i+1)\",\n"
        << "  \"tflops_pct_definition\": \"useful_flops_per_second / dense_bf16_tc_peak; not "
           "mixed-kernel hardware utilization\",\n"
        << "  \"logical_kv_bytes_definition\": \"per-query unique GQA K/V references, no "
           "query-block reuse\",\n"
        << "  \"global_floor_bytes_definition\": \"Q read + output write + input K/V read + "
           "cache K/V write + current prompt-kernel per-q_head K/V tile reads\",\n"
        << "  \"tile_reuse_queries_definition\": \"logical token-key pairs divided by loaded "
           "q-block key rows before q_head multiplication\",\n"
        << "  \"timing\": {\"warmup\": " << timing.warmup << ", \"repeat\": " << timing.repeat
        << ", \"min_time_ms\": " << timing.min_time_ms << "},\n"
        << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const AppendPromptMetrics& m = results[i];
        out << "    {\"T\": " << m.tokens << ", \"context\": " << m.context
            << ", \"end_context\": " << m.end_context << ", \"ms\": " << json_number(m.median_ms)
            << ", \"kv_dtype\": \"" << kv_dtype_name(m.kv_dtype) << "\"" << ", \"math_mode\": \""
            << m.math_mode << "\"" << ", \"qk_mma_dtype\": \"" << m.qk_mma_dtype << "\""
            << ", \"pv_mma_dtype\": \"" << m.pv_mma_dtype << "\""
            << ", \"qk_mma_count\": " << m.qk_mma_count << ", \"pv_mma_count\": " << m.pv_mma_count
            << ", \"tflops\": " << json_number(m.tflops)
            << ", \"tflops_pct\": " << json_number(m.tflops_pct)
            << ", \"global_floor_gbps\": " << json_number(m.global_floor_gbps)
            << ", \"global_floor_gbps_pct\": " << json_number(m.global_floor_gbps_pct)
            << ", \"tile_kv_read_gbps\": " << json_number(m.tile_kv_read_gbps)
            << ", \"logical_kv_gbps\": " << json_number(m.logical_kv_gbps)
            << ", \"avg_keys_per_query\": " << json_number(m.avg_keys_per_query)
            << ", \"ns_per_key_query\": " << json_number(m.ns_per_key_query)
            << ", \"us_per_token\": " << json_number(m.us_per_token)
            << ", \"q_blocks\": " << m.q_blocks << ", \"attention_ctas\": " << m.attention_ctas
            << ", \"key_tiles\": " << m.key_tiles
            << ", \"tile_reuse_queries\": " << json_number(m.tile_reuse_queries)
            << ", \"bound\": \"" << m.bound
            << "\", \"roofline_tflops\": " << json_number(m.roofline_tflops)
            << ", \"roofline_eff_pct\": " << json_number(m.roofline_eff_pct)
            << ", \"global_floor_bytes\": " << json_number(m.global_floor_bytes)
            << ", \"tile_kv_read_bytes\": " << json_number(m.tile_kv_read_bytes)
            << ", \"logical_kv_bytes\": " << json_number(m.logical_kv_bytes)
            << ", \"useful_flops\": " << json_number(m.useful_flops)
            << ", \"key_sum\": " << json_number(m.key_sum)
            << ", \"qblock_key_rows\": " << json_number(m.qblock_key_rows)
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
    bool cached_small_t              = false;
    bool kv_append                   = false;
    bool append_prompt_baseline      = false;
    bool append_prompt_attention     = false;
    bool copy_ceiling                = false;
    bool decode_pos_set              = false;
    bool context_set                 = false;
    bool profile_once                = false;
    bool cold_cache                  = false;
    std::int32_t decode_pos          = 0;
    std::int32_t context             = 0;
    std::uint32_t round_robin_layers = 1;
    DType kv_dtype                   = DType::BF16;
    bool geometry_35b                = false;
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> contexts;
    PrefillTimingOptions prefill_timing;
    double expect_tflops_pct_min = -1.0;
    std::string csv_out;
    std::string json_out;
};

void fail_usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninfer_gqa_attention_bench [--prefill] [--tokens T[,T...]] "
        "[--geometry 27b|35b] [--kv-dtype bf16|int8] "
        "[--expect-tflops-pct-min PCT] [--csv-out path] [--json-out path]\n"
        "       ninfer_gqa_attention_bench --prefill --tokens 4096 --profile-once\n"
        "       ninfer_gqa_attention_bench --append-small-t --tokens T[,T...] "
        "--context N[,N...] [--geometry 27b|35b] [--kv-dtype bf16|int8] "
        "[--profile-once] [--cold-cache]\n"
        "       ninfer_gqa_attention_bench --cached-small-t --tokens T[,T...] "
        "--context N[,N...] [--geometry 27b|35b] [--kv-dtype bf16|int8] "
        "[--profile-once] [--cold-cache]\n"
        "       ninfer_gqa_attention_bench --kv-append --tokens T[,T...] "
        "--context N[,N...] [--geometry 27b|35b] [--kv-dtype bf16|int8] "
        "[--profile-once] [--cold-cache] [--csv-out path] [--json-out path]\n"
        "       ninfer_gqa_attention_bench --append-prompt-baseline --tokens T[,T...] "
        "--context N[,N...] [--geometry 27b|35b] [--kv-dtype bf16|int8] "
        "[--csv-out path] [--json-out path]\n"
        "       ninfer_gqa_attention_bench --append-prompt-attention-only --tokens T[,T...] "
        "--context N[,N...] [--geometry 27b|35b] [--kv-dtype bf16|int8]\n"
        "       ninfer_gqa_attention_bench --copy-ceiling --tokens T --context N "
        "[--geometry 27b|35b] [--kv-dtype bf16|int8]\n"
        "       ninfer_gqa_attention_bench --decode [--decode-pos N] [--profile-once] "
        "[--cold-cache] [--round-robin-layers 16] [--geometry 27b|35b] "
        "[--kv-dtype bf16|int8]\n",
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

DType parse_kv_dtype_arg(const char* text, const char* flag) {
    if (!std::strcmp(text, "bf16")) { return DType::BF16; }
    if (!std::strcmp(text, "int8")) { return DType::I8; }
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s expects bf16 or int8", flag);
    fail_usage(msg);
    return DType::BF16;
}

std::vector<std::int32_t> parse_i32_list_arg(const char* text, const char* flag, bool allow_zero) {
    std::vector<std::int32_t> out;
    const char* start = text;
    while (*start != '\0') {
        const char* comma = std::strchr(start, ',');
        std::string piece(start, comma == nullptr ? std::strlen(start)
                                                  : static_cast<std::size_t>(comma - start));
        const std::int32_t value = parse_i32_arg(piece.c_str(), flag);
        if (!allow_zero && value <= 0) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "%s expects positive values", flag);
            fail_usage(msg);
        }
        out.push_back(value);
        if (comma == nullptr) { break; }
        start = comma + 1;
    }
    if (out.empty()) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s expects at least one value", flag);
        fail_usage(msg);
    }
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
        } else if (!std::strcmp(argv[i], "--cached-small-t")) {
            opt.cached_small_t = true;
        } else if (!std::strcmp(argv[i], "--kv-append")) {
            opt.kv_append = true;
        } else if (!std::strcmp(argv[i], "--append-prompt-baseline")) {
            opt.append_prompt_baseline = true;
        } else if (!std::strcmp(argv[i], "--append-prompt-attention-only")) {
            opt.append_prompt_attention = true;
        } else if (!std::strcmp(argv[i], "--copy-ceiling")) {
            opt.copy_ceiling = true;
        } else if (!std::strcmp(argv[i], "--tokens") || !std::strcmp(argv[i], "--prefill-tokens")) {
            if (++i >= argc) { fail_usage("--tokens requires a value"); }
            opt.tokens = parse_i32_list_arg(argv[i], "--tokens", false);
        } else if (!std::strcmp(argv[i], "--decode-pos")) {
            if (++i >= argc) { fail_usage("--decode-pos requires a value"); }
            opt.decode_pos     = parse_i32_arg(argv[i], "--decode-pos");
            opt.decode_pos_set = true;
            opt.decode         = true;
        } else if (!std::strcmp(argv[i], "--context")) {
            if (++i >= argc) { fail_usage("--context requires a value"); }
            opt.contexts    = parse_i32_list_arg(argv[i], "--context", true);
            opt.context     = opt.contexts[0];
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
        } else if (!std::strcmp(argv[i], "--kv-dtype")) {
            if (++i >= argc) { fail_usage("--kv-dtype requires a value"); }
            opt.kv_dtype = parse_kv_dtype_arg(argv[i], "--kv-dtype");
        } else if (!std::strcmp(argv[i], "--geometry")) {
            if (++i >= argc) { fail_usage("--geometry requires a value"); }
            if (!std::strcmp(argv[i], "27b")) {
                opt.geometry_35b = false;
            } else if (!std::strcmp(argv[i], "35b")) {
                opt.geometry_35b = true;
            } else {
                fail_usage("--geometry expects 27b or 35b");
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

    if (!opt.decode && !opt.prefill && !opt.append_small_t && !opt.cached_small_t &&
        !opt.kv_append && !opt.append_prompt_baseline && !opt.append_prompt_attention &&
        !opt.copy_ceiling) {
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
    if (opt.cached_small_t && opt.tokens.empty()) { opt.tokens = {1}; }
    if (opt.kv_append && opt.tokens.empty()) { opt.tokens = {1, 2, 3, 4, 5, 6, 1024}; }
    if ((opt.append_prompt_baseline || opt.append_prompt_attention) && opt.tokens.empty()) {
        opt.tokens = {1024};
    }
    if (opt.copy_ceiling && opt.tokens.empty()) { opt.tokens = {1}; }
    if (opt.copy_ceiling && opt.tokens.size() != 1u) {
        fail_usage("--copy-ceiling requires exactly one --tokens value");
    }
    if (opt.append_small_t || opt.cached_small_t) {
        for (const std::int32_t tokens : opt.tokens) {
            if (tokens <= 0 || tokens > 16) {
                fail_usage("small-T attention supports T values in [1,16]");
            }
        }
    }
    if (opt.kv_append) {
        for (const std::int32_t tokens : opt.tokens) {
            if (tokens <= 0 || tokens > 1024) {
                fail_usage("--kv-append supports T values in [1,1024]");
            }
        }
    }
    if (opt.copy_ceiling && (opt.tokens[0] <= 0 || opt.tokens[0] > 6)) {
        fail_usage("--copy-ceiling currently supports T in [1,6]");
    }
    if ((opt.append_small_t || opt.cached_small_t || opt.kv_append || opt.append_prompt_baseline ||
         opt.append_prompt_attention || opt.copy_ceiling) &&
        !opt.context_set) {
        fail_usage("append/copy modes require --context");
    }
    if (opt.copy_ceiling && opt.contexts.size() != 1u) {
        fail_usage("--copy-ceiling requires exactly one --context value");
    }
    if ((opt.append_prompt_baseline || opt.append_prompt_attention) && opt.contexts.empty()) {
        fail_usage("--append-prompt modes require --context");
    }
    if (opt.append_small_t || opt.cached_small_t || opt.kv_append || opt.copy_ceiling ||
        opt.append_prompt_baseline || opt.append_prompt_attention) {
        for (const std::int32_t tokens : opt.tokens) {
            for (const std::int32_t context : opt.contexts) {
                if (context > std::numeric_limits<std::int32_t>::max() - tokens) {
                    fail_usage("--context + --tokens exceeds the maximum supported window");
                }
            }
        }
    }
    if (static_cast<int>(opt.decode) + static_cast<int>(opt.prefill) +
            static_cast<int>(opt.append_small_t) + static_cast<int>(opt.cached_small_t) +
            static_cast<int>(opt.kv_append) + static_cast<int>(opt.append_prompt_baseline) +
            static_cast<int>(opt.append_prompt_attention) + static_cast<int>(opt.copy_ceiling) >
        1) {
        fail_usage("select exactly one benchmark mode");
    }
    if (opt.profile_once && opt.decode && !opt.prefill && !opt.decode_pos_set) {
        fail_usage("--decode --profile-once requires --decode-pos");
    }
    if (opt.decode_pos_set && opt.decode_pos == std::numeric_limits<std::int32_t>::max()) {
        fail_usage("--decode-pos exceeds the maximum supported decode window");
    }
    if (opt.profile_once &&
        static_cast<int>(opt.decode) + static_cast<int>(opt.prefill) +
                static_cast<int>(opt.append_small_t) + static_cast<int>(opt.cached_small_t) +
                static_cast<int>(opt.kv_append) + static_cast<int>(opt.append_prompt_baseline) +
                static_cast<int>(opt.append_prompt_attention) >
            1) {
        fail_usage("--profile-once must target one mode");
    }
    if (opt.profile_once && opt.prefill && opt.tokens.size() != 1u) {
        fail_usage("--prefill --profile-once requires exactly one --tokens length");
    }
    if (opt.profile_once && opt.append_small_t &&
        (opt.tokens.size() != 1u || opt.contexts.size() != 1u)) {
        fail_usage("--append-small-t --profile-once requires one T and one context");
    }
    if (opt.profile_once && opt.cached_small_t &&
        (opt.tokens.size() != 1u || opt.contexts.size() != 1u)) {
        fail_usage("--cached-small-t --profile-once requires one T and one context");
    }
    if (opt.profile_once && opt.kv_append &&
        (opt.tokens.size() != 1u || opt.contexts.size() != 1u)) {
        fail_usage("--kv-append --profile-once requires one T and one context");
    }
    if (opt.profile_once && opt.decode && opt.round_robin_layers != 1u) {
        fail_usage("--profile-once cannot be combined with --round-robin-layers");
    }
    if (opt.cold_cache && (!opt.profile_once || (!opt.decode && !opt.append_small_t &&
                                                 !opt.cached_small_t && !opt.kv_append))) {
        fail_usage("--cold-cache is only valid with decode or small-T/append --profile-once");
    }
    if (opt.profile_once && opt.copy_ceiling) {
        fail_usage("--copy-ceiling does not support --profile-once");
    }
    if (opt.profile_once && (opt.append_prompt_baseline || opt.append_prompt_attention)) {
        fail_usage("--append-prompt modes do not support --profile-once");
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

void select_geometry(bool geometry_35b) {
    if (geometry_35b) {
        kQHeads       = 16;
        kKVHeads      = 2;
        kGeometryName = "35b";
    } else {
        kQHeads       = 24;
        kKVHeads      = 4;
        kGeometryName = "27b";
    }
    kGqaDecodeSplits = ops::detail::gqa_attention_decode_splits(kQHeads, kKVHeads);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    const Options opt = parse_options(argc, argv);
    select_geometry(opt.geometry_35b);
    std::printf("gqa_attention geometry=%s q_heads=%d kv_heads=%d decode_splits=%d\n",
                kGeometryName, kQHeads, kKVHeads, kGqaDecodeSplits);
    const std::vector<std::int32_t> decode_positions = selected_decode_positions(opt);

    std::int32_t max_context = 2048;
    for (const std::int32_t pos : decode_positions) {
        max_context = std::max(max_context, pos + 1);
    }
    for (const std::int32_t tokens : opt.tokens) { max_context = std::max(max_context, tokens); }
    if (opt.append_small_t || opt.cached_small_t || opt.kv_append || opt.append_prompt_baseline ||
        opt.append_prompt_attention || opt.copy_ceiling) {
        for (const std::int32_t tokens : opt.tokens) {
            for (const std::int32_t context : opt.contexts) {
                max_context = std::max(max_context, context + tokens);
            }
        }
    }
    const std::uint32_t cache_layers = opt.round_robin_layers;
    DeviceArena cache_arena(cache_arena_bytes(cache_layers, max_context, opt.kv_dtype));
    WorkspaceArena work_arena(decode_workspace_bytes(decode_positions));
    LayoutBuilder cache_layout_builder;
    auto cache_layout = ninfer::plan_kv_cache(
        cache_layout_builder, cache_layers, static_cast<std::uint32_t>(max_context), kKVHeads,
        kHeadDim, opt.kv_dtype, opt.kv_dtype == DType::I8 ? kBenchKvQuantGroup : 0);
    const DeviceSpan cache_backing = cache_arena.alloc_bytes(cache_layout_builder.finish(256));
    KVCache kv(cache_backing, cache_layout);
    for (std::uint32_t layer = 0; layer < kv.layer_count(); ++layer) {
        if (kv.dtype == DType::I8) {
            CUDA_CHECK(cudaMemset(kv.k[layer].data, 0, kv.k[layer].bytes()));
            CUDA_CHECK(cudaMemset(kv.v[layer].data, 0, kv.v[layer].bytes()));
            CUDA_CHECK(cudaMemset(kv.k_scale[layer].data, 0, kv.k_scale[layer].bytes()));
            CUDA_CHECK(cudaMemset(kv.v_scale[layer].data, 0, kv.v_scale[layer].bytes()));
        } else {
            CUDA_CHECK(cudaMemset(kv.k[layer].data, 0x3e, kv.k[layer].bytes()));
            CUDA_CHECK(cudaMemset(kv.v[layer].data, 0x3d, kv.v[layer].bytes()));
        }
    }

    const std::size_t qn  = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kvn = static_cast<std::size_t>(kHeadDim) * kKVHeads;
    DBuf q                = make_bf16(qn);
    DBuf k                = make_bf16(kvn);
    DBuf v                = make_bf16(kvn);
    DBuf out              = make_zeros(qn * sizeof(std::uint16_t));

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
        if (opt.profile_once) {
            DBuf cold_cache = make_zeros(opt.cold_cache ? kColdCacheBytes : 1u);
            run_append_small_t_profile_once(kv, opt.tokens[0], opt.context,
                                            opt.cold_cache ? &cold_cache : nullptr);
        } else {
            for (const std::int32_t tokens : opt.tokens) {
                for (const std::int32_t context : opt.contexts) {
                    run_append_small_t(kv, tokens, context);
                }
            }
        }
    }
    if (opt.cached_small_t) {
        if (opt.profile_once) {
            DBuf cold_cache = make_zeros(opt.cold_cache ? kColdCacheBytes : 1u);
            run_cached_small_t_profile_once(kv, opt.tokens[0], opt.context,
                                            opt.cold_cache ? &cold_cache : nullptr);
        } else {
            for (const std::int32_t tokens : opt.tokens) {
                for (const std::int32_t context : opt.contexts) {
                    run_cached_small_t(kv, tokens, context);
                }
            }
        }
    }
    if (opt.kv_append) {
        if (opt.profile_once) {
            DBuf cold_cache = make_zeros(opt.cold_cache ? kColdCacheBytes : 1u);
            run_kv_append_profile_once(kv, opt.tokens[0], opt.context,
                                       opt.cold_cache ? &cold_cache : nullptr);
        } else {
            std::vector<KvAppendMetrics> results;
            results.reserve(opt.tokens.size() * opt.contexts.size());
            for (const std::int32_t tokens : opt.tokens) {
                for (const std::int32_t context : opt.contexts) {
                    results.push_back(run_kv_append(kv, tokens, context, opt.prefill_timing));
                }
            }
            write_text_file(opt.csv_out, format_kv_append_csv(results));
            write_text_file(opt.json_out, format_kv_append_json(results, opt.prefill_timing));
        }
    }
    if (opt.append_prompt_baseline) {
        std::vector<AppendPromptMetrics> append_results;
        append_results.reserve(opt.tokens.size() * opt.contexts.size());
        for (const std::int32_t tokens : opt.tokens) {
            for (const std::int32_t context : opt.contexts) {
                append_results.push_back(
                    run_append_prompt_baseline(kv, tokens, context, opt.prefill_timing));
            }
        }
        write_text_file(opt.csv_out, format_append_prompt_csv(append_results));
        write_text_file(opt.json_out,
                        format_append_prompt_json(append_results, opt.prefill_timing));
    }
    if (opt.append_prompt_attention) {
        std::vector<AppendPromptMetrics> append_results;
        append_results.reserve(opt.tokens.size() * opt.contexts.size());
        for (const std::int32_t tokens : opt.tokens) {
            for (const std::int32_t context : opt.contexts) {
                append_results.push_back(
                    run_append_prompt_attention_only(kv, tokens, context, opt.prefill_timing));
            }
        }
        write_text_file(opt.csv_out, format_append_prompt_csv(append_results));
        write_text_file(opt.json_out,
                        format_append_prompt_json(append_results, opt.prefill_timing));
    }
    if (opt.copy_ceiling) { run_copy_ceiling(opt.tokens[0], opt.context, opt.kv_dtype); }
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
