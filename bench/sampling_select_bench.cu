// Kernel-level benchmark for Qwen3.6 sampler candidate selection at the real
// decode shape. The reported GB/s uses the intended one-pass payload:
// BF16 logits once, plus i32 token_counts once when penalties are active.
#include "ninfer/core/device.h"
#include "ninfer/core/tensor.h"
#include "ninfer/kernels/argmax.h"
#include "ninfer/kernels/sampling.h"
#include "model/mtp_ops.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kVocab = 248320;
constexpr std::int32_t kMtpK  = 3;
constexpr std::int32_t kStats = 9;

struct Options {
    int cols    = 1;
    bool sample = false;
    bool mtp    = false;
    bool argmax = false;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--sample] [--mtp] [--argmax] [--cols 1|4]\n", argv0);
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(name) + " needs value"); }
            return argv[++i];
        };
        if (arg == "--sample") {
            options.sample = true;
        } else if (arg == "--mtp") {
            options.mtp = true;
        } else if (arg == "--argmax") {
            options.argmax = true;
        } else if (arg == "--cols") {
            options.cols = parse_int(need_value("--cols"), "--cols");
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_sampling_select_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!options.sample && !options.mtp && !options.argmax) {
        options.sample = true;
        options.mtp    = true;
        options.argmax = true;
    }
    if (options.cols != 1 && options.cols != 4) {
        throw std::invalid_argument("--cols must be 1 or 4");
    }
    return options;
}

DBuf make_logits(int cols) {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(kVocab) * cols);
    for (int c = 0; c < cols; ++c) {
        const int hot = (17 + c * 7919) % kVocab;
        for (int v = 0; v < kVocab; ++v) {
            float x = -8.0f + static_cast<float>((v * 17 + c * 31) % 4096) * (1.0f / 4096.0f);
            if (v == hot) { x = 8.0f; }
            h[static_cast<std::size_t>(c) * kVocab + v] = f32_to_bf16(x);
        }
    }
    DBuf d(h.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
    return d;
}

DBuf make_i32(const std::vector<std::int32_t>& h) {
    DBuf d(h.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
    return d;
}

DBuf make_i64_zero(std::size_t n) {
    DBuf d(n * sizeof(std::int64_t));
    CUDA_CHECK(cudaMemset(d.p, 0, d.bytes));
    return d;
}

DBuf make_config(DBuf& counts) {
    kernels::SamplingConfig cfg;
    cfg.temperature      = 0.6f;
    cfg.top_k            = 20;
    cfg.top_p            = 0.95f;
    cfg.min_p            = 0.0f;
    cfg.presence_penalty = 1.0f;
    cfg.seed             = 20260706ull;
    cfg.token_counts     = static_cast<std::int32_t*>(counts.p);

    DBuf d(sizeof(kernels::SamplingConfig));
    CUDA_CHECK(cudaMemcpy(d.p, &cfg, sizeof(cfg), cudaMemcpyHostToDevice));
    return d;
}

double payload_bytes(int cols, bool counts) {
    const double per_col = static_cast<double>(kVocab) * (2.0 + (counts ? 4.0 : 0.0));
    return per_col * static_cast<double>(cols);
}

void run_argmax(DBuf& logits, int cols) {
    DBuf out(static_cast<std::size_t>(cols) * sizeof(std::int32_t));
    Tensor tlogits(logits.p, DType::BF16, {kVocab, cols});
    Tensor tout(out.p, DType::I32, {cols});
    const Result r = bench_loop([&](cudaStream_t s) { kernels::argmax(tlogits, tout, s); },
                                payload_bytes(cols, false));
    print_result(cols == 1 ? "argmax_tiled_atomic [vocab,1]" : "argmax_tiled_atomic [vocab,4]", r);
}

void run_sample(DBuf& logits, DBuf& cfg, int cols) {
    DBuf out(static_cast<std::size_t>(cols) * sizeof(std::int32_t));
    DBuf pos(sizeof(std::int32_t));
    CUDA_CHECK(cudaMemset(pos.p, 0, pos.bytes));
    Tensor tlogits(logits.p, DType::BF16, {kVocab, cols});
    Tensor tout(out.p, DType::I32, {cols});
    auto* cfg_ptr  = static_cast<const kernels::SamplingConfig*>(cfg.p);
    auto* pos_ptr  = static_cast<const std::int32_t*>(pos.p);
    const Result r = bench_loop(
        [&](cudaStream_t s) {
            kernels::sample(tlogits, tout, cfg_ptr, pos_ptr, kernels::kSamplePurposeDecode, s);
        },
        payload_bytes(cols, true));
    print_result(cols == 1 ? "sample [vocab,1]" : "sample [vocab,4]", r);
}

void run_mtp(DBuf& logits, DBuf& cfg) {
    auto targets = make_i32({0, 0, 0, 0});
    auto drafts  = make_i32({17, 17 + 7919, 17 + 2 * 7919});
    auto length  = make_i32({128});
    auto token   = make_i32({-1});
    DBuf sampled(static_cast<std::size_t>(kMtpK + 1) * sizeof(std::int32_t));
    DBuf num(sizeof(std::int32_t));
    DBuf accepted(sizeof(std::int32_t));
    DBuf ar_pos(sizeof(std::int32_t));
    DBuf stats = make_i64_zero(kStats);

    Tensor ttargets(targets.p, DType::I32, {kMtpK + 1});
    Tensor tlogits(logits.p, DType::BF16, {kVocab, kMtpK + 1});
    Tensor tdrafts(drafts.p, DType::I32, {kMtpK});
    Tensor tlength(length.p, DType::I32, {1});
    Tensor ttoken(token.p, DType::I32, {1});
    Tensor tsampled(sampled.p, DType::I32, {kMtpK + 1});
    Tensor tnum(num.p, DType::I32, {1});
    Tensor taccepted(accepted.p, DType::I32, {1});
    Tensor tar_pos(ar_pos.p, DType::I32, {1});
    Tensor tstats(stats.p, DType::I64, {kStats});
    auto* cfg_ptr = static_cast<const kernels::SamplingConfig*>(cfg.p);

    const Result r = bench_loop(
        [&](cudaStream_t s) {
            kernels::mtp_accept_tokens(ttargets, tlogits, tdrafts, tlength, ttoken, tsampled, tnum,
                                       taccepted, tar_pos, tstats, cfg_ptr, s);
        },
        payload_bytes(kMtpK + 1, true));
    print_result("mtp_accept k=3 [vocab,4]", r);
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sampling_select_bench: %s\n", e.what());
        return 2;
    }

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    DBuf logits = make_logits(options.mtp ? 4 : options.cols);
    DBuf counts(static_cast<std::size_t>(kVocab) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));
    DBuf cfg = make_config(counts);

    std::printf("payload: vocab=%d logits=%.3f MiB/col counts=%.3f MiB/col sampling=%.3f MiB/col\n",
                kVocab, static_cast<double>(kVocab * 2) / 1048576.0,
                static_cast<double>(kVocab * 4) / 1048576.0, payload_bytes(1, true) / 1048576.0);
    if (options.argmax) { run_argmax(logits, options.mtp ? 4 : options.cols); }
    if (options.sample) { run_sample(logits, cfg, options.cols); }
    if (options.mtp) { run_mtp(logits, cfg); }
    return 0;
}
