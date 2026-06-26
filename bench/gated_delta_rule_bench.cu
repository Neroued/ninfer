// Performance bench for gated_delta_rule at the real Qwen3.6-27B GDN shapes.
//
// Default keeps the original decode behavior:
//   ./qus_gated_delta_rule_bench
//   ./qus_gated_delta_rule_bench --decode --kernel-only
//
// Chunked prefill is explicit:
//   ./qus_gated_delta_rule_bench --prefill
//   ./qus_gated_delta_rule_bench --prefill --kernel-only
//   ./qus_gated_delta_rule_bench --prefill --sweep --csv
//
// The public-wrapper chunked row includes bf16 q/k/v/out boundary casts and
// WorkspaceArena scratch. The kernel-only-fp32 row times the detail chunked
// launcher with fp32 q/k/v/out/state and preallocated workspace, matching
// ~/chunked_gdn/bench_chunked's allocation policy.
#include "qus/kernels/gated_delta_rule.h"
#include "kernels/launcher/gated_delta_rule.h"
#include "qus_bench_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::bench;

namespace {

constexpr int kS             = 128;
constexpr int kHqk           = 16;
constexpr int kHv            = 48;
constexpr int kB             = 1;
constexpr int kDecodeT       = 1;
constexpr int kPrefillT      = 4096;
constexpr int kChunkSize     = 64;
constexpr float kScale       = 0.08838834764831845f;
constexpr std::size_t kAlign = 256;

struct Options {
    bool decode      = false;
    bool prefill     = false;
    bool kernel_only = false;
    bool sweep       = false;
    bool csv         = false;
    bool help        = false;

    int S           = kS;
    int H_qk        = kHqk;
    int H_v         = kHv;
    int L           = kPrefillT;
    int B           = kB;
    int warmup      = 20;
    int repeat      = 100;
    int min_time_ms = 1000;
};

struct BenchRow {
    const char* mode = "";
    const char* path = "";
    int L            = 0;
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
        } else {
            fail((std::string("unknown argument: ") + arg + " (try --help)").c_str());
        }
    }

    if (!opt.decode && !opt.prefill) {
        opt.prefill = opt.sweep;
        opt.decode  = !opt.sweep;
    }
    if (opt.sweep && opt.decode && !opt.prefill) {
        fail("--sweep is a chunked prefill option; use --prefill --sweep");
    }
    if (opt.S != kS || opt.H_qk != kHqk || opt.H_v != kHv || opt.B != kB) {
        fail("gated_delta_rule bench supports only S=128 H_qk=16 H_v=48 B=1");
    }
    if (opt.kernel_only && opt.prefill && (opt.L % kChunkSize) != 0 && !opt.sweep) {
        fail("--prefill --kernel-only requires L to be a multiple of 64");
    }
    return opt;
}

void print_help(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Modes:\n"
        "  --decode           run recurrent decode at L=1 (default when no mode is given)\n"
        "  --prefill          run chunked prefill at L=4096 by default\n"
        "  --chunked          alias for --prefill\n"
        "  --kernel-only      use the FP32 detail launcher path instead of the public wrapper\n"
        "  --sweep            with prefill, sweep L over {64,128,256,512,1024,4096}\n"
        "\n"
        "Shape:\n"
        "  --S N              fixed to 128\n"
        "  --H_qk N           fixed to 16\n"
        "  --H_v N            fixed to 48\n"
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

std::size_t align_up(std::size_t value, std::size_t align = kAlign) {
    return (value + align - 1) & ~(align - 1);
}

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > static_cast<std::size_t>(-1) - a) {
        throw std::overflow_error("gated_delta_rule_bench: size overflow");
    }
    return a + b;
}

std::size_t tensor_bytes(std::size_t n, std::size_t elem_size) { return n * elem_size; }

std::size_t wrapper_workspace_bytes(int T) {
    const std::size_t t      = static_cast<std::size_t>(T);
    const std::size_t qk_n   = static_cast<std::size_t>(kS) * kHqk * t;
    const std::size_t v_n    = static_cast<std::size_t>(kS) * kHv * t;
    const int T_full         = (T / kChunkSize) * kChunkSize;
    const std::size_t stages = kernels::detail::gdn_chunked_workspace_bytes(T_full);
    std::size_t bytes        = 0;
    bytes                    = checked_add(bytes, align_up(tensor_bytes(qk_n, sizeof(float))));
    bytes                    = checked_add(bytes, align_up(tensor_bytes(qk_n, sizeof(float))));
    bytes                    = checked_add(bytes, align_up(tensor_bytes(v_n, sizeof(float))));
    bytes                    = checked_add(bytes, align_up(tensor_bytes(v_n, sizeof(float))));
    bytes                    = checked_add(bytes, align_up(stages));
    return checked_add(bytes, 4u * 1024u * 1024u);
}

double estimate_user_bytes(int T, std::size_t qkv_elem_size) {
    const double t       = static_cast<double>(T);
    const double qk_n    = static_cast<double>(kS) * kHqk * t;
    const double v_n     = static_cast<double>(kS) * kHv * t;
    const double gb_n    = static_cast<double>(kHv) * t;
    const double state_n = static_cast<double>(kS) * kS * kHv;
    return (2.0 * qk_n + 2.0 * v_n) * static_cast<double>(qkv_elem_size) +
           (2.0 * gb_n + 2.0 * state_n) * static_cast<double>(sizeof(float));
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

std::vector<int> prefill_lengths(const Options& opt) {
    if (opt.sweep) { return {64, 128, 256, 512, 1024, 4096}; }
    return {opt.L};
}

BenchRow run_decode_public(const Options& opt) {
    const int T               = kDecodeT;
    const std::size_t qk_n    = static_cast<std::size_t>(kS) * kHqk * T;
    const std::size_t v_n     = static_cast<std::size_t>(kS) * kHv * T;
    const std::size_t gb_n    = static_cast<std::size_t>(kHv) * T;
    const std::size_t state_n = static_cast<std::size_t>(kS) * kS * kHv;

    DBuf q     = make_bf16(qk_n);
    DBuf k     = make_bf16(qk_n);
    DBuf v     = make_bf16(v_n);
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(std::uint16_t));

    Tensor tq(q.p, DType::BF16, {kS, kHqk, T});
    Tensor tk(k.p, DType::BF16, {kS, kHqk, T});
    Tensor tv(v.p, DType::BF16, {kS, kHv, T});
    Tensor tg(g.p, DType::FP32, {kHv, T});
    Tensor tbeta(beta.p, DType::FP32, {kHv, T});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::BF16, {kS, kHv, T});

    BenchRow row{"decode", "public-wrapper", T, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            kernels::gated_delta_rule_recurrent(tq, tk, tv, tg, tbeta, kScale, tstate, tout, s);
        },
        estimate_user_bytes(T, sizeof(std::uint16_t)), opt.warmup, opt.repeat, opt.min_time_ms);
    return row;
}

BenchRow run_decode_kernel_only(const Options& opt) {
    const int T               = kDecodeT;
    const std::size_t v_n     = static_cast<std::size_t>(kS) * kHv * T;
    const std::size_t gb_n    = static_cast<std::size_t>(kHv) * T;
    const std::size_t state_n = static_cast<std::size_t>(kS) * kS * kHv;

    DBuf q     = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x12345678u));
    DBuf k     = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x87654321u));
    DBuf v     = make_f32(make_ramp(v_n, 0.5f));
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(float));

    Tensor tq(q.p, DType::FP32, {kS, kHqk, T});
    Tensor tk(k.p, DType::FP32, {kS, kHqk, T});
    Tensor tv(v.p, DType::FP32, {kS, kHv, T});
    Tensor tg(g.p, DType::FP32, {kHv, T});
    Tensor tbeta(beta.p, DType::FP32, {kHv, T});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::FP32, {kS, kHv, T});

    BenchRow row{"decode", "kernel-only-fp32", T, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            kernels::detail::gated_delta_rule_recurrent_launch(tq, tk, tv, tg, tbeta, kScale,
                                                               tstate, tout, s);
        },
        estimate_user_bytes(T, sizeof(float)), opt.warmup, opt.repeat, opt.min_time_ms);
    return row;
}

BenchRow run_prefill_public(const Options& opt, int T) {
    const std::size_t v_n     = static_cast<std::size_t>(kS) * kHv * T;
    const std::size_t gb_n    = static_cast<std::size_t>(kHv) * T;
    const std::size_t state_n = static_cast<std::size_t>(kS) * kS * kHv;

    DBuf q =
        make_bf16_from_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x12345678u));
    DBuf k =
        make_bf16_from_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x87654321u));
    DBuf v     = make_bf16_from_f32(make_ramp(v_n, 0.5f));
    DBuf g     = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta  = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state = make_zeros(state_n * sizeof(float));
    DBuf out   = make_zeros(v_n * sizeof(std::uint16_t));
    WorkspaceArena ws(wrapper_workspace_bytes(T));

    Tensor tq(q.p, DType::BF16, {kS, kHqk, T});
    Tensor tk(k.p, DType::BF16, {kS, kHqk, T});
    Tensor tv(v.p, DType::BF16, {kS, kHv, T});
    Tensor tg(g.p, DType::FP32, {kHv, T});
    Tensor tbeta(beta.p, DType::FP32, {kHv, T});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::BF16, {kS, kHv, T});

    BenchRow row{"prefill", "public-wrapper", T, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            kernels::gated_delta_rule_chunked(tq, tk, tv, tg, tbeta, kScale, kChunkSize, ws, tstate,
                                              tout, s);
        },
        estimate_user_bytes(T, sizeof(std::uint16_t)), opt.warmup, opt.repeat, opt.min_time_ms);
    return row;
}

BenchRow run_prefill_kernel_only(const Options& opt, int T) {
    if ((T % kChunkSize) != 0) {
        fail("--prefill --kernel-only requires L to be a multiple of 64");
    }
    const std::size_t v_n      = static_cast<std::size_t>(kS) * kHv * T;
    const std::size_t gb_n     = static_cast<std::size_t>(kHv) * T;
    const std::size_t state_n  = static_cast<std::size_t>(kS) * kS * kHv;
    const std::size_t ws_bytes = kernels::detail::gdn_chunked_workspace_bytes(T);

    DBuf q         = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x12345678u));
    DBuf k         = make_f32(make_normalized_qk(static_cast<std::size_t>(kHqk) * T, 0x87654321u));
    DBuf v         = make_f32(make_ramp(v_n, 0.5f));
    DBuf g         = make_f32(std::vector<float>(gb_n, -1.0f));
    DBuf beta      = make_f32(std::vector<float>(gb_n, 0.5f));
    DBuf state     = make_zeros(state_n * sizeof(float));
    DBuf out       = make_zeros(v_n * sizeof(float));
    DBuf workspace = make_zeros(ws_bytes);

    Tensor tq(q.p, DType::FP32, {kS, kHqk, T});
    Tensor tk(k.p, DType::FP32, {kS, kHqk, T});
    Tensor tv(v.p, DType::FP32, {kS, kHv, T});
    Tensor tg(g.p, DType::FP32, {kHv, T});
    Tensor tbeta(beta.p, DType::FP32, {kHv, T});
    Tensor tstate(state.p, DType::FP32, {kS, kS, kHv});
    Tensor tout(out.p, DType::FP32, {kS, kHv, T});

    BenchRow row{"prefill", "kernel-only-fp32", T, {}};
    row.result = bench_loop(
        [&](cudaStream_t s) {
            kernels::detail::gated_delta_rule_chunked_launch(tq, tk, tv, tg, tbeta, kScale, tstate,
                                                             tout, workspace.p, workspace.bytes, s);
        },
        estimate_user_bytes(T, sizeof(float)), opt.warmup, opt.repeat, opt.min_time_ms);
    return row;
}

void print_csv_header() {
    std::printf("mode,path,S,H_qk,H_v,L,B,runs,inner,median_us,min_us,p95_us,mean_us,gbs\n");
}

void print_row(const BenchRow& row, bool csv) {
    if (csv) {
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", row.mode, row.path, kS,
                    kHqk, kHv, row.L, kB, row.result.n_runs, row.result.inner_iters,
                    row.result.median_us, row.result.min_us, row.result.p95_us, row.result.mean_us,
                    row.result.gbs);
        return;
    }

    char tag[128];
    std::snprintf(tag, sizeof(tag), "gdn %s %s [S128,Hqk16,Hv48,L%d]", row.mode, row.path, row.L);
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
            print_row(opt.kernel_only ? run_decode_kernel_only(opt) : run_decode_public(opt),
                      opt.csv);
        }

        if (opt.prefill) {
            for (int L : prefill_lengths(opt)) {
                print_row(opt.kernel_only ? run_prefill_kernel_only(opt, L)
                                          : run_prefill_public(opt, L),
                          opt.csv);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
