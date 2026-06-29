// Performance bench for gqa_attention at Qwen3.6-27B decode/prefill shapes.
// The GB/s readout is informational; correctness is covered by
// tests/kernels/test_gqa_attention.cpp.
//   ./qus_gqa_attention_bench --decode
//   ./qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
//   ./qus_gqa_attention_bench --prefill
#include "qus/core/device.h"
#include "qus/kernels/gqa_attention.h"
#include "qus_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKVHeads = 4;
constexpr float kScale          = 0.0625f;
constexpr std::int32_t kDChunk  = 32;
constexpr std::size_t kColdCacheBytes = std::size_t(256) << 20;
constexpr std::int32_t kDefaultDecodePositions[] = {2048, 2882, 8192, 32768};

constexpr std::int32_t align_up_128(std::int32_t value) {
    return ((value + 127) / 128) * 128;
}

std::int32_t ceil_div_i32(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::size_t ceil_div_size(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1u) / divisor;
}

std::size_t align_overhead(std::size_t allocations) {
    return allocations * 255u;
}

std::int32_t decode_tile_n(std::int32_t pos_value) {
    const std::int32_t window = pos_value + 1;
    return (window <= 4096) ? 16 : (window <= 16384) ? 64 : 128;
}

std::int32_t decode_tile_count(std::int32_t pos_value) {
    const std::int32_t window = pos_value + 1;
    return ceil_div_i32(window, decode_tile_n(pos_value));
}

struct DecodeBytes {
    std::size_t useful_kv = 0;
    std::size_t scratch = 0;
    std::size_t total = 0;
};

DecodeBytes decode_bytes(std::int32_t pos_value) {
    const auto window     = static_cast<std::size_t>(pos_value) + 1u;
    const auto tile_count = static_cast<std::size_t>(decode_tile_count(pos_value));

    const std::size_t k_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t v_cache_reads = window * kKVHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t new_kv_writes = kKVHeads * kHeadDim * sizeof(std::uint16_t) * 2u;
    const std::size_t q_reads       = kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t output_writes = kQHeads * kHeadDim * sizeof(std::uint16_t);

    const std::size_t partial_acc_writes_reads =
        2u * tile_count * kQHeads * kHeadDim * sizeof(std::uint16_t);
    const std::size_t partial_ml_writes = tile_count * kQHeads * 2u * sizeof(float);
    const std::size_t partial_ml_reads =
        ceil_div_i32(kHeadDim, kDChunk) * tile_count * kQHeads * 2u * sizeof(float);

    DecodeBytes bytes;
    bytes.useful_kv = k_cache_reads + v_cache_reads;
    bytes.scratch   = partial_acc_writes_reads + partial_ml_writes + partial_ml_reads;
    bytes.total = bytes.useful_kv + new_kv_writes + q_reads + output_writes + bytes.scratch;
    return bytes;
}

std::size_t decode_workspace_bytes_for_pos(std::int32_t pos_value) {
    const auto tile_count = static_cast<std::size_t>(decode_tile_count(pos_value));
    const std::size_t partial_acc_bytes =
        static_cast<std::size_t>(kHeadDim) * kQHeads * tile_count * sizeof(std::uint16_t);
    const std::size_t partial_m_bytes = static_cast<std::size_t>(kQHeads) * tile_count * sizeof(float);
    const std::size_t partial_l_bytes = static_cast<std::size_t>(kQHeads) * tile_count * sizeof(float);
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

__global__ void bench_cold_cache_touch_kernel(std::uint32_t* data, std::size_t words) {
    const std::size_t start = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t step  = blockDim.x * static_cast<std::size_t>(gridDim.x);
    for (std::size_t i = start; i < words; i += step) { data[i] = data[i] + 1u; }
}

void touch_cold_cache(DBuf& buf, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::size_t words = buf.bytes / sizeof(std::uint32_t);
    if (words == 0) { return; }
    const int grid = static_cast<int>(std::min<std::size_t>(4096u, ceil_div_size(words, kBlock)));
    bench_cold_cache_touch_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::uint32_t*>(buf.p), words);
    CUDA_CHECK(cudaGetLastError());
}

void print_decode_result(const char* tag, const Result& r, const DecodeBytes& bytes,
                         std::int32_t pos_value, std::uint32_t round_robin_layers) {
    const double sec = r.median_us * 1.0e-6;
    const double total_gbs = (sec > 0.0) ? static_cast<double>(bytes.total) / sec / 1.0e9 : 0.0;
    const double useful_kv_gbs =
        (sec > 0.0) ? static_cast<double>(bytes.useful_kv) / sec / 1.0e9 : 0.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::printf("%-38s median=%8.2f us  min=%8.2f us  p95=%8.2f us  total_model=%8.1f GB/s  "
                "useful_kv=%8.1f GB/s  bytes useful_kv=%.2f MiB scratch=%.2f MiB total=%.2f MiB  "
                "tile_n=%d tile_count=%d round_robin_layers=%u%s\n",
                tag, r.median_us, r.min_us, r.p95_us, total_gbs, useful_kv_gbs,
                static_cast<double>(bytes.useful_kv) / kMiB,
                static_cast<double>(bytes.scratch) / kMiB,
                static_cast<double>(bytes.total) / kMiB, decode_tile_n(pos_value),
                decode_tile_count(pos_value), round_robin_layers,
                (round_robin_layers == 1u) ? " hot_cache_info" : "");
}

void run_decode(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k, const Tensor& v,
                Tensor& out, std::int32_t pos_value, std::uint32_t round_robin_layers) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});

    const DecodeBytes bytes = decode_bytes(pos_value);
    std::uint32_t next_layer = 0;

    const Result r = bench_loop([&](cudaStream_t s) {
        kv.pos = static_cast<std::uint32_t>(pos_value);
        const int layer = static_cast<int>(next_layer);
        next_layer      = (next_layer + 1u) % round_robin_layers;
        kernels::gqa_attention_decode(q, k, v, pos, kScale, kv, layer, ws, out, s);
    }, static_cast<double>(bytes.total));

    char tag[96];
    std::snprintf(tag, sizeof(tag), "gqa_attention decode combined pos=%d", pos_value);
    print_decode_result(tag, r, bytes, pos_value, round_robin_layers);
}

void run_profile_once(KVCache& kv, WorkspaceArena& ws, const Tensor& q, const Tensor& k,
                      const Tensor& v, Tensor& out, std::int32_t pos_value, DBuf* cold_cache) {
    DBuf pos_buf = make_i32(pos_value);
    Tensor pos(pos_buf.p, DType::I32, {1});
    const DecodeBytes bytes = decode_bytes(pos_value);

    cudaStream_t stream = nullptr;
    if (cold_cache != nullptr) { touch_cold_cache(*cold_cache, stream); }
    kv.pos = static_cast<std::uint32_t>(pos_value);
    kernels::gqa_attention_decode(q, k, v, pos, kScale, kv, 0, ws, out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("PROFILE_ONCE gqa_attention decode combined pos=%d tile_n=%d tile_count=%d "
                "useful_kv_bytes=%zu scratch_bytes=%zu total_modeled_bytes=%zu\n",
                pos_value, decode_tile_n(pos_value), decode_tile_count(pos_value), bytes.useful_kv,
                bytes.scratch, bytes.total);
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

struct Options {
    bool decode = false;
    bool prefill = false;
    bool decode_pos_set = false;
    bool profile_once = false;
    bool cold_cache = false;
    std::int32_t decode_pos = 0;
    std::uint32_t round_robin_layers = 1;
};

void fail_usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: qus_gqa_attention_bench [--decode] [--prefill] [--decode-pos N] "
                 "[--profile-once] [--cold-cache] [--round-robin-layers 16]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32_arg(const char* text, const char* flag) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s expects a non-negative int32", flag);
        fail_usage(msg);
    }
    return static_cast<std::int32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) {
            opt.decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            opt.prefill = true;
        } else if (!std::strcmp(argv[i], "--decode-pos")) {
            if (++i >= argc) { fail_usage("--decode-pos requires a value"); }
            opt.decode_pos = parse_i32_arg(argv[i], "--decode-pos");
            opt.decode_pos_set = true;
            opt.decode = true;
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            opt.profile_once = true;
            opt.decode = true;
        } else if (!std::strcmp(argv[i], "--cold-cache")) {
            opt.cold_cache = true;
        } else if (!std::strcmp(argv[i], "--round-robin-layers")) {
            if (++i >= argc) { fail_usage("--round-robin-layers requires a value"); }
            opt.round_robin_layers = static_cast<std::uint32_t>(
                parse_i32_arg(argv[i], "--round-robin-layers"));
            if (opt.round_robin_layers == 0) {
                fail_usage("--round-robin-layers expects a positive value");
            }
        } else {
            fail_usage("unknown argument");
        }
    }

    if (!opt.decode && !opt.prefill) { opt.decode = true; }
    if (opt.profile_once && !opt.decode_pos_set) {
        fail_usage("--profile-once requires --decode-pos");
    }
    if (opt.decode_pos_set && opt.decode_pos == std::numeric_limits<std::int32_t>::max()) {
        fail_usage("--decode-pos exceeds the maximum supported decode window");
    }
    if (opt.profile_once && opt.prefill) {
        fail_usage("--profile-once cannot be combined with --prefill");
    }
    if (opt.profile_once && opt.round_robin_layers != 1u) {
        fail_usage("--profile-once cannot be combined with --round-robin-layers");
    }
    if (opt.cold_cache && !opt.profile_once) {
        fail_usage("--cold-cache is only valid with --profile-once");
    }
    if (opt.round_robin_layers != 1u && !opt.decode) {
        fail_usage("--round-robin-layers is only valid with --decode");
    }
    if (opt.round_robin_layers != 1u && opt.round_robin_layers != 16u) {
        fail_usage("--round-robin-layers currently supports only 16");
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

    const Options opt = parse_options(argc, argv);
    const std::vector<std::int32_t> decode_positions = selected_decode_positions(opt);

    std::int32_t max_context = 2048;
    for (const std::int32_t pos : decode_positions) { max_context = std::max(max_context, pos + 1); }
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
    if (opt.prefill) {
        run_prefill(kv, 128);
        run_prefill(kv, 2048);
    }
    return 0;
}
