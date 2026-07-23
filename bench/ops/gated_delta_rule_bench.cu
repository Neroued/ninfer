// Performance bench for the native gated_delta_rule geometry. Defaults are the Qwen3.6-27B shape;
// pass --H_v 32 for Qwen3.6-35B-A3B.
//
// Default keeps the original decode behavior:
//   ./ninfer_gated_delta_rule_bench
//   ./ninfer_gated_delta_rule_bench --decode --kernel-only
//
// Chunked prefill is explicit:
//   ./ninfer_gated_delta_rule_bench --prefill
//   ./ninfer_gated_delta_rule_bench --prefill --kernel-only
//   ./ninfer_gated_delta_rule_bench --prefill --sweep --csv
//
// The public-wrapper chunked row includes WorkspaceArena scratch. The
// kernel-only row times the detail chunked launcher with preallocated
// workspace, matching ~/chunked_gdn/bench_chunked's allocation policy.
#include "ninfer/ops/gated_delta_rule.h"
#include "ninfer/ops/l2norm.h"
#include "ops/launcher/gated_delta_rule.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kS         = 128;
constexpr int kHqk       = 16;
constexpr int kHv        = 48;
constexpr int kB         = 1;
constexpr int kDecodeT   = 1;
constexpr int kPrefillT  = 4096;
constexpr int kChunkSize = 64;

struct Options {
    bool decode      = false;
    bool prefill     = false;
    bool small_t     = false;
    bool kernel_only = false;
    bool sweep       = false;
    bool csv         = false;
    bool help        = false;

    int S               = kS;
    int H_qk            = kHqk;
    int H_v             = kHv;
    int L               = kPrefillT;
    int B               = kB;
    int warmup          = 20;
    int repeat          = 100;
    int min_time_ms     = 1000;
    std::string qk_norm = "fused";
    std::vector<int> t_sweep{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
};

struct BenchRow {
    const char* mode            = "";
    const char* path            = "";
    int L                       = 0;
    std::size_t workspace_bytes = 0;
    Result result{};
};

[[noreturn]] void fail(const char* msg) { throw std::invalid_argument(msg); }

int parse_positive_int(const char* flag, const char* value) {
    if (value == nullptr) { fail((std::string("missing argument for ") + flag).c_str()); }
    char* end   = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
        fail((std::string("invalid positive integer for ") + flag + ": " + value).c_str());
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_t_sweep(const char* value) {
    if (value == nullptr || *value == '\0') { fail("--t-sweep requires a comma-separated list"); }
    std::vector<int> sweep;
    const std::string spec(value);
    std::size_t begin = 0;
    while (begin < spec.size()) {
        const std::size_t end   = spec.find(',', begin);
        const std::string token = spec.substr(begin, end == std::string::npos ? end : end - begin);
        const int T             = parse_positive_int("--t-sweep", token.c_str());
        if (T > 16) { fail("--t-sweep supports exact T in [1,16]"); }
        sweep.push_back(T);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (sweep.empty()) { fail("--t-sweep requires at least one T"); }
    return sweep;
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto take       = [&]() -> const char* {
            if (i + 1 >= argc) { fail((std::string("missing argument for ") + arg).c_str()); }
            return argv[++i];
        };

        if (!std::strcmp(arg, "-h") || !std::strcmp(arg, "--help")) {
            opt.help = true;
        } else if (!std::strcmp(arg, "--decode")) {
            opt.decode = true;
        } else if (!std::strcmp(arg, "--prefill") || !std::strcmp(arg, "--chunked")) {
            opt.prefill = true;
        } else if (!std::strcmp(arg, "--small-t")) {
            opt.small_t = true;
        } else if (!std::strcmp(arg, "--kernel-only")) {
            opt.kernel_only = true;
        } else if (!std::strcmp(arg, "--sweep")) {
            opt.sweep = true;
        } else if (!std::strcmp(arg, "--csv")) {
            opt.csv = true;
        } else if (!std::strcmp(arg, "--S")) {
            opt.S = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--H_qk")) {
            opt.H_qk = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--H_v")) {
            opt.H_v = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--L")) {
            opt.L = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--B")) {
            opt.B = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--warmup")) {
            opt.warmup = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--repeat")) {
            opt.repeat = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--min-time-ms")) {
            opt.min_time_ms = parse_positive_int(arg, take());
        } else if (!std::strcmp(arg, "--qk-norm")) {
            opt.qk_norm = take();
            if (opt.qk_norm != "fused" && opt.qk_norm != "composed") {
                fail("--qk-norm must be fused or composed");
            }
        } else if (!std::strcmp(arg, "--t-sweep")) {
            opt.t_sweep = parse_t_sweep(take());
        } else {
            fail((std::string("unknown argument: ") + arg + " (try --help)").c_str());
        }
    }

    if (!opt.decode && !opt.prefill && !opt.small_t) {
        opt.prefill = opt.sweep;
        opt.decode  = !opt.sweep;
    }
    if (opt.small_t && (opt.decode || opt.prefill || opt.kernel_only || opt.sweep)) {
        fail("--small-t cannot be combined with decode, prefill, kernel-only, or sweep");
    }
    if (opt.sweep && opt.decode && !opt.prefill) {
        fail("--sweep is a chunked prefill option; use --prefill --sweep");
    }
    const bool supported_head_dim = opt.S == 16 || opt.S == 32 || opt.S == 64 || opt.S == 128;
    if (!supported_head_dim || opt.H_v < opt.H_qk || (opt.H_v % opt.H_qk) != 0) {
        fail("gated_delta_rule requires S in {16,32,64,128} and divisible H_v >= H_qk");
    }
    if (opt.B != kB) { fail("gated_delta_rule NInfer bench supports the product batch B=1"); }
    if (opt.kernel_only && opt.prefill && (opt.L % kChunkSize) != 0 && !opt.sweep) {
        fail("--prefill --kernel-only requires L to be a multiple of 64");
    }
    return opt;
}

void print_help(const char* prog) {
    std::printf("Usage: %s [options]\n"
                "\n"
                "Modes:\n"
                "  --decode           run recurrent decode at L=1 (default when no mode is given)\n"
                "  --prefill          run chunked prefill at L=4096 by default\n"
                "  --chunked          alias for --prefill\n"
                "  --small-t          sweep snapshot T over exact 1..16\n"
                "  --t-sweep LIST     comma-separated small-T subset in [1,16]\n"
                "  --qk-norm MODE     small-T route: fused or composed (default: fused)\n"
                "  --kernel-only      use the native-bf16 detail launcher path instead of the "
                "public wrapper\n"
                "  --sweep            with prefill, sweep L over {64,128,256,512,1024,4096}\n"
                "\n"
                "Shape:\n"
                "  --S N              head dimension in {16,32,64,128} (default: 128)\n"
                "  --H_qk N           Q/K heads (default: 16)\n"
                "  --H_v N            divisible value heads >= H_qk (default: 48)\n"
                "  --L N              prefill length (default: 4096)\n"
                "  --B N              fixed to 1\n"
                "\n"
                "Bench loop:\n"
                "  --warmup N         warmup launches (default: 20)\n"
                "  --repeat N         minimum timed samples (default: 100)\n"
                "  --min-time-ms N    minimum total timed batch duration (default: 1000)\n"
                "\n"
                "Output:\n"
                "  --csv              emit CSV rows\n"
                "  -h, --help         show this help\n",
                prog);
}

float scale_for(const Options& opt) { return 1.0f / std::sqrt(static_cast<float>(opt.S)); }

std::size_t wrapper_workspace_bytes(const Options& opt, int T, bool normalize_qk = false) {
    const std::size_t required =
        ops::gated_delta_rule_workspace_bytes(opt.S, opt.H_qk, opt.H_v, T, normalize_qk);
    return std::max<std::size_t>(required, 256);
}

double estimate_user_bytes(const Options& opt, int T, std::size_t qkv_elem_size) {
    const double t       = static_cast<double>(T);
    const double qk_n    = static_cast<double>(opt.S) * opt.H_qk * t;
    const double v_n     = static_cast<double>(opt.S) * opt.H_v * t;
    const double gb_n    = static_cast<double>(opt.H_v) * t;
    const double state_n = static_cast<double>(opt.S) * opt.S * opt.H_v;
    return (2.0 * qk_n + 2.0 * v_n) * static_cast<double>(qkv_elem_size) +
           (2.0 * gb_n + 2.0 * state_n) * static_cast<double>(sizeof(float));
}

double estimate_snapshot_user_bytes(const Options& opt, int T) {
    const double t       = static_cast<double>(T);
    const double qk_n    = static_cast<double>(opt.S) * opt.H_qk * t;
    const double v_n     = static_cast<double>(opt.S) * opt.H_v * t;
    const double gb_n    = static_cast<double>(opt.H_v) * t;
    const double state_n = static_cast<double>(opt.S) * opt.S * opt.H_v;
    return (2.0 * qk_n + 2.0 * v_n) * static_cast<double>(sizeof(std::uint16_t)) +
           2.0 * gb_n * static_cast<double>(sizeof(float)) +
           (1.0 + t) * state_n * static_cast<double>(sizeof(float)) +
           static_cast<double>(sizeof(std::int32_t));
}

DBuf make_f32(const std::vector<float>& h) {
    DBuf d(h.size() * sizeof(float));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

DBuf make_bf16_from_f32(const std::vector<float>& h) {
    std::vector<std::uint16_t> bf16(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) { bf16[i] = f32_to_bf16(h[i]); }
    DBuf d(bf16.size() * sizeof(std::uint16_t));
    cudaMemcpy(d.p, bf16.data(), bf16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> make_normalized_qk(std::size_t rows, int head_dim, std::uint32_t seed) {
    std::vector<float> h(rows * static_cast<std::size_t>(head_dim));
    std::uint32_t state = seed;
    for (float& x : h) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        x             = 2.0f * u - 1.0f;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        float* row = h.data() + r * head_dim;
        double ss  = 0.0;
        for (int i = 0; i < head_dim; ++i) { ss += static_cast<double>(row[i]) * row[i]; }
        const float inv = static_cast<float>(1.0 / std::sqrt(ss + 1.0e-12));
        for (int i = 0; i < head_dim; ++i) { row[i] *= inv; }
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

std::vector<int> prefill_lengths(const Options& opt) {
    if (opt.sweep) { return {64, 128, 256, 512, 1024, 4096}; }
    return {opt.L};
}

Result graph_cold_loop(const launch_fn& launch, double bytes_moved, int warmup, int repeat) {
    constexpr std::size_t FlushBytes = 256ULL << 20;
    cudaStream_t stream              = nullptr;
    cudaGraph_t graph                = nullptr;
    cudaGraphExec_t exec             = nullptr;
    cudaEvent_t start                = nullptr;
    cudaEvent_t stop                 = nullptr;
    const auto require               = [](cudaError_t status, const char* label) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
        }
    };

    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    launch(stream);
    require(cudaStreamSynchronize(stream), "resolve small-T route");
    require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch(stream);
    require(cudaStreamEndCapture(stream, &graph), "end capture");
    require(cudaGraphInstantiate(&exec, graph, 0), "instantiate graph");
    require(cudaEventCreate(&start), "create start event");
    require(cudaEventCreate(&stop), "create stop event");
    DBuf flush(FlushBytes);

    for (int i = 0; i < warmup; ++i) {
        require(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream), "flush L2");
        require(cudaGraphLaunch(exec, stream), "warm graph launch");
    }
    require(cudaStreamSynchronize(stream), "warm graph sync");

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        require(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream), "flush L2");
        require(cudaEventRecord(start, stream), "record start");
        require(cudaGraphLaunch(exec, stream), "timed graph launch");
        require(cudaEventRecord(stop, stream), "record stop");
        require(cudaEventSynchronize(stop), "timed graph sync");
        float milliseconds = 0.0f;
        require(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    require(cudaEventDestroy(start), "destroy start event");
    require(cudaEventDestroy(stop), "destroy stop event");
    require(cudaGraphExecDestroy(exec), "destroy graph exec");
    require(cudaGraphDestroy(graph), "destroy graph");
    require(cudaStreamDestroy(stream), "destroy stream");

    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double q) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(q * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    double sum = 0.0;
    for (double sample : samples) { sum += sample; }
    Result result;
    result.n_runs      = repeat;
    result.inner_iters = 1;
    result.median_us   = percentile(0.50);
    result.min_us      = samples.front();
    result.p95_us      = percentile(0.95);
    result.mean_us     = sum / static_cast<double>(samples.size());
    result.gbs         = bytes_moved / (result.median_us * 1.0e3);
    return result;
}

BenchRow run_small_t_snapshot(const Options& opt, int T) {
    constexpr int Slots       = 17;
    constexpr int InitialSlot = 16;
    const std::size_t qk_n    = static_cast<std::size_t>(opt.S) * opt.H_qk * T;
    const std::size_t v_n     = static_cast<std::size_t>(opt.S) * opt.H_v * T;
    const std::size_t gb_n    = static_cast<std::size_t>(opt.H_v) * T;
    const std::size_t state_n = static_cast<std::size_t>(opt.S) * opt.S * opt.H_v;

    DBuf q = make_bf16(qk_n), k = make_bf16(qk_n), v = make_bf16(v_n);
    DBuf q_norm = make_zeros(qk_n * sizeof(std::uint16_t));
    DBuf k_norm = make_zeros(qk_n * sizeof(std::uint16_t));
    DBuf g      = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta   = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf states = make_zeros(state_n * Slots * sizeof(float));
    DBuf out    = make_zeros(v_n * sizeof(std::uint16_t));
    DBuf initial(sizeof(std::int32_t));
    cudaMemcpy(initial.p, &InitialSlot, sizeof(InitialSlot), cudaMemcpyHostToDevice);

    Tensor tq(q.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tk(k.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tq_norm(q_norm.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tk_norm(k_norm.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tv(v.p, DType::BF16, {opt.S, opt.H_v, T});
    Tensor tg(g.p, DType::FP32, {opt.H_v, T});
    Tensor tbeta(beta.p, DType::FP32, {opt.H_v, T});
    Tensor tstates(states.p, DType::FP32, {opt.S, opt.S, opt.H_v, Slots});
    Tensor tinitial(initial.p, DType::I32, {1});
    Tensor tout(out.p, DType::BF16, {opt.S, opt.H_v, T});
    WorkspaceArena ws(256);
    const bool fused = opt.qk_norm == "fused";

    BenchRow row{"small-t",
                 fused ? "gated_delta_rule_snapshot.bf16.recurrent.qk_fused.w4"
                       : "l2norm_x2+gated_delta_rule_snapshot.bf16.recurrent.qk_pre_normalized.w4",
                 T,
                 0,
                 {}};
    row.result = graph_cold_loop(
        [&](cudaStream_t s) {
            const Tensor* q_recurrent = &tq;
            const Tensor* k_recurrent = &tk;
            if (!fused) {
                ops::l2norm(tq, 1.0e-6f, tq_norm, s);
                ops::l2norm(tk, 1.0e-6f, tk_norm, s);
                q_recurrent = &tq_norm;
                k_recurrent = &tk_norm;
            }
            ops::gated_delta_rule_snapshot(*q_recurrent, *k_recurrent, tv, tg, tbeta,
                                           scale_for(opt), fused, ws, tstates, tinitial, tout, s);
        },
        estimate_snapshot_user_bytes(opt, T), opt.warmup, opt.repeat);
    return row;
}

BenchRow run_decode_public(const Options& opt) {
    const int T               = kDecodeT;
    const std::size_t qk_n    = static_cast<std::size_t>(opt.S) * opt.H_qk * T;
    const std::size_t v_n     = static_cast<std::size_t>(opt.S) * opt.H_v * T;
    const std::size_t gb_n    = static_cast<std::size_t>(opt.H_v) * T;
    const std::size_t state_n = static_cast<std::size_t>(opt.S) * opt.S * opt.H_v;

    DBuf q     = make_bf16(qk_n);
    DBuf k     = make_bf16(qk_n);
    DBuf v     = make_bf16(v_n);
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(std::uint16_t));

    Tensor tq(q.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tk(k.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tv(v.p, DType::BF16, {opt.S, opt.H_v, T});
    Tensor tg(g.p, DType::FP32, {opt.H_v, T});
    Tensor tbeta(beta.p, DType::FP32, {opt.H_v, T});
    Tensor tstate(state.p, DType::FP32, {opt.S, opt.S, opt.H_v});
    Tensor tout(out.p, DType::BF16, {opt.S, opt.H_v, T});
    WorkspaceArena ws(wrapper_workspace_bytes(opt, T));

    BenchRow row{"decode", "public-wrapper", T, 0, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            ops::gated_delta_rule(tq, tk, tv, tg, tbeta, scale_for(opt), false, ws, tstate, tout,
                                  s);
        },
        estimate_user_bytes(opt, T, sizeof(std::uint16_t)), opt.warmup, opt.repeat,
        opt.min_time_ms);
    return row;
}

BenchRow run_decode_kernel_only(const Options& opt) {
    const int T               = kDecodeT;
    const std::size_t v_n     = static_cast<std::size_t>(opt.S) * opt.H_v * T;
    const std::size_t gb_n    = static_cast<std::size_t>(opt.H_v) * T;
    const std::size_t state_n = static_cast<std::size_t>(opt.S) * opt.S * opt.H_v;

    DBuf q =
        make_f32(make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x12345678u));
    DBuf k =
        make_f32(make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x87654321u));
    DBuf v     = make_f32(make_ramp(v_n, 0.5f));
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(float));

    Tensor tq(q.p, DType::FP32, {opt.S, opt.H_qk, T});
    Tensor tk(k.p, DType::FP32, {opt.S, opt.H_qk, T});
    Tensor tv(v.p, DType::FP32, {opt.S, opt.H_v, T});
    Tensor tg(g.p, DType::FP32, {opt.H_v, T});
    Tensor tbeta(beta.p, DType::FP32, {opt.H_v, T});
    Tensor tstate(state.p, DType::FP32, {opt.S, opt.S, opt.H_v});
    Tensor tout(out.p, DType::FP32, {opt.S, opt.H_v, T});

    BenchRow row{"decode", "kernel-only-fp32", T, 0, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            ops::detail::gated_delta_rule_recurrent_launch(tq, tk, tv, tg, tbeta, scale_for(opt),
                                                           tstate, tout, s);
        },
        estimate_user_bytes(opt, T, sizeof(float)), opt.warmup, opt.repeat, opt.min_time_ms);
    return row;
}

BenchRow run_prefill_public(const Options& opt, int T) {
    const std::size_t v_n     = static_cast<std::size_t>(opt.S) * opt.H_v * T;
    const std::size_t gb_n    = static_cast<std::size_t>(opt.H_v) * T;
    const std::size_t state_n = static_cast<std::size_t>(opt.S) * opt.S * opt.H_v;

    DBuf q = make_bf16_from_f32(
        make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x12345678u));
    DBuf k = make_bf16_from_f32(
        make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x87654321u));
    DBuf v                     = make_bf16_from_f32(make_ramp(v_n, 0.5f));
    DBuf g                     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta                  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state                 = make_zeros(state_n * sizeof(float));
    DBuf out                   = make_zeros(v_n * sizeof(std::uint16_t));
    const std::size_t ws_bytes = wrapper_workspace_bytes(opt, T);
    WorkspaceArena ws(ws_bytes);

    Tensor tq(q.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tk(k.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tv(v.p, DType::BF16, {opt.S, opt.H_v, T});
    Tensor tg(g.p, DType::FP32, {opt.H_v, T});
    Tensor tbeta(beta.p, DType::FP32, {opt.H_v, T});
    Tensor tstate(state.p, DType::FP32, {opt.S, opt.S, opt.H_v});
    Tensor tout(out.p, DType::BF16, {opt.S, opt.H_v, T});

    BenchRow row{"prefill", "public-wrapper", T, ws_bytes, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            ops::gated_delta_rule(tq, tk, tv, tg, tbeta, scale_for(opt), false, ws, tstate, tout,
                                  s);
        },
        estimate_user_bytes(opt, T, sizeof(std::uint16_t)), opt.warmup, opt.repeat,
        opt.min_time_ms);
    return row;
}

BenchRow run_prefill_kernel_only(const Options& opt, int T) {
    if ((T % kChunkSize) != 0) {
        fail("--prefill --kernel-only requires L to be a multiple of 64");
    }
    const std::size_t v_n     = static_cast<std::size_t>(opt.S) * opt.H_v * T;
    const std::size_t gb_n    = static_cast<std::size_t>(opt.H_v) * T;
    const std::size_t state_n = static_cast<std::size_t>(opt.S) * opt.S * opt.H_v;
    const std::size_t ws_bytes =
        ops::detail::gdn_chunked_workspace_bytes(opt.S, opt.H_qk, opt.H_v, T);

    DBuf q = make_bf16_from_f32(
        make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x12345678u));
    DBuf k = make_bf16_from_f32(
        make_normalized_qk(static_cast<std::size_t>(opt.H_qk) * T, opt.S, 0x87654321u));
    DBuf v         = make_bf16_from_f32(make_ramp(v_n, 0.5f));
    DBuf g         = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta      = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state     = make_zeros(state_n * sizeof(float));
    DBuf out       = make_zeros(v_n * sizeof(std::uint16_t));
    DBuf workspace = make_zeros(ws_bytes);

    Tensor tq(q.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tk(k.p, DType::BF16, {opt.S, opt.H_qk, T});
    Tensor tv(v.p, DType::BF16, {opt.S, opt.H_v, T});
    Tensor tg(g.p, DType::FP32, {opt.H_v, T});
    Tensor tbeta(beta.p, DType::FP32, {opt.H_v, T});
    Tensor tstate(state.p, DType::FP32, {opt.S, opt.S, opt.H_v});
    Tensor tout(out.p, DType::BF16, {opt.S, opt.H_v, T});

    BenchRow row{"prefill", "kernel-only-bf16", T, ws_bytes, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            ops::detail::gated_delta_rule_chunked_launch(tq, tk, tv, tg, tbeta, scale_for(opt),
                                                         tstate, tstate, tout, workspace.p,
                                                         workspace.bytes, s);
        },
        estimate_user_bytes(opt, T, sizeof(std::uint16_t)), opt.warmup, opt.repeat,
        opt.min_time_ms);
    return row;
}

void print_csv_header() {
    std::printf("mode,path,S,H_qk,H_v,L,B,workspace_bytes,runs,inner,median_us,min_us,p95_us,mean_"
                "us,gbs\n");
}

void print_row(const BenchRow& row, const Options& opt) {
    if (opt.csv) {
        std::printf("%s,%s,%d,%d,%d,%d,%d,%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", row.mode, row.path,
                    opt.S, opt.H_qk, opt.H_v, row.L, opt.B, row.workspace_bytes, row.result.n_runs,
                    row.result.inner_iters, row.result.median_us, row.result.min_us,
                    row.result.p95_us, row.result.mean_us, row.result.gbs);
        return;
    }

    char tag[128];
    std::snprintf(tag, sizeof(tag), "gdn %s %s [S%d,Hqk%d,Hv%d,L%d,ws=%.1fMiB]", row.mode, row.path,
                  opt.S, opt.H_qk, opt.H_v, row.L,
                  static_cast<double>(row.workspace_bytes) / (1024.0 * 1024.0));
    print_result(tag, row.result);
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_options(argc, argv);
        if (opt.help) {
            print_help(argv[0]);
            return 0;
        }

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }

        if (opt.csv) { print_csv_header(); }

        if (opt.decode) {
            print_row(opt.kernel_only ? run_decode_kernel_only(opt) : run_decode_public(opt), opt);
        }

        if (opt.small_t) {
            for (int T : opt.t_sweep) { print_row(run_small_t_snapshot(opt, T), opt); }
        }

        if (opt.prefill) {
            for (int L : prefill_lengths(opt)) {
                print_row(opt.kernel_only ? run_prefill_kernel_only(opt, L)
                                          : run_prefill_public(opt, L),
                          opt);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
