// Qualification benchmark for Qwen3.6-35B G2 sampling and G3 MTP accept.
//
//   ./ninfer_sampling_select_bench --sample --mode stochastic
//   ./ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5
//   ./ninfer_sampling_select_bench --matrix
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;
constexpr std::int32_t kStats        = 9;

enum class Mode {
    Greedy,
    Stochastic,
};

struct Options {
    bool sample        = false;
    bool mtp           = false;
    bool matrix        = false;
    bool counts_active = true;
    Mode mode          = Mode::Stochastic;
    int mtp_k          = 3;
    int top_k          = 20;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--sample|--mtp|--matrix] [--mode greedy|stochastic] "
                "[--mtp-k 1..5] [--top-k 1..20] [--no-counts]\n",
                argv0);
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
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--sample") {
            options.sample = true;
        } else if (arg == "--mtp") {
            options.mtp = true;
        } else if (arg == "--matrix") {
            options.matrix = true;
        } else if (arg == "--mode") {
            const std::string_view mode = need_value("--mode");
            if (mode == "greedy") {
                options.mode = Mode::Greedy;
            } else if (mode == "stochastic") {
                options.mode = Mode::Stochastic;
            } else {
                throw std::invalid_argument("--mode must be greedy or stochastic");
            }
        } else if (arg == "--mtp-k") {
            options.mtp_k = parse_int(need_value("--mtp-k"), "--mtp-k");
        } else if (arg == "--top-k") {
            options.top_k = parse_int(need_value("--top-k"), "--top-k");
        } else if (arg == "--no-counts") {
            options.counts_active = false;
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_sampling_select_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!options.sample && !options.mtp && !options.matrix) { options.matrix = true; }
    if (options.matrix && (options.sample || options.mtp)) {
        throw std::invalid_argument("--matrix cannot be combined with --sample or --mtp");
    }
    if (options.mtp_k < 1 || options.mtp_k > 5) {
        throw std::invalid_argument("--mtp-k must be in [1,5]");
    }
    if (options.top_k < 1 || options.top_k > 20) {
        throw std::invalid_argument("--top-k must be in [1,20]");
    }
    return options;
}

DBuf make_logits(int cols) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kPhysicalRows) * cols);
    for (int col = 0; col < cols; ++col) {
        const int hot = (17 + col * 7919) % kTokenDomain;
        for (int row = 0; row < kPhysicalRows; ++row) {
            float value = -8.0f + static_cast<float>((row * 17 + col * 31) % 4096) / 4096.0f;
            if (row == hot) { value = 8.0f; }
            if (row >= kTokenDomain) { value = 20.0f; }
            host[static_cast<std::size_t>(col) * kPhysicalRows + row] = f32_to_bf16(value);
        }
    }
    DBuf device(host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

DBuf make_i32(const std::vector<std::int32_t>& host) {
    DBuf device(host.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

DBuf make_i64_zero(std::size_t count) {
    DBuf device(count * sizeof(std::int64_t));
    CUDA_CHECK(cudaMemset(device.p, 0, device.bytes));
    return device;
}

DBuf make_config(DBuf& counts, Mode mode, bool counts_active, int top_k) {
    ops::SamplingConfig config;
    config.temperature      = mode == Mode::Greedy ? 0.0f : 0.6f;
    config.top_k            = top_k;
    config.top_p            = 0.95f;
    config.presence_penalty = mode == Mode::Stochastic && counts_active ? 1.0f : 0.0f;
    config.seed             = 20260716ull;
    config.token_counts =
        mode == Mode::Stochastic && counts_active ? static_cast<std::int32_t*>(counts.p) : nullptr;

    DBuf device(sizeof(ops::SamplingConfig));
    CUDA_CHECK(cudaMemcpy(device.p, &config, sizeof(config), cudaMemcpyHostToDevice));
    return device;
}

double stochastic_payload_bytes(int cols, bool counts_active) {
    const double per_col = static_cast<double>(kTokenDomain) * 2.0 +
                           (counts_active ? static_cast<double>(kTokenDomain) * 4.0 : 0.0);
    return per_col * static_cast<double>(cols);
}

void run_sample(DBuf& logits, DBuf& counts, Mode mode, bool counts_active, int top_k) {
    CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));
    DBuf config = make_config(counts, mode, counts_active, top_k);
    DBuf out(sizeof(std::int32_t));
    DBuf pos(sizeof(std::int32_t));
    CUDA_CHECK(cudaMemset(pos.p, 0, pos.bytes));
    Tensor tlogits(logits.p, DType::BF16, {kPhysicalRows, 1});
    Tensor tout(out.p, DType::I32, {1});
    WorkspaceArena workspace(ops::sampling_workspace_bytes(kTokenDomain, 1));
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(config.p);
    const auto* pos_ptr    = static_cast<const std::int32_t*>(pos.p);

    const double bytes  = mode == Mode::Greedy ? static_cast<double>(kTokenDomain) * 2.0
                                               : stochastic_payload_bytes(1, counts_active);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::sample(tlogits, tout, kTokenDomain, config_ptr, pos_ptr, ops::kSamplePurposeDecode,
                        workspace, stream);
        },
        bytes);
    const std::string label = std::string("G2 ") +
                              (mode == Mode::Greedy ? "greedy" : "stochastic") +
                              (counts_active && mode == Mode::Stochastic ? " counts" : "") +
                              " top_k=" + std::to_string(top_k);
    print_result(label.c_str(), result);
}

void run_mtp(DBuf& logits, DBuf& counts, int k, Mode mode, bool counts_active, int top_k) {
    CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));
    DBuf config = make_config(counts, mode, counts_active, top_k);
    std::vector<std::int32_t> target_host(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> draft_host(static_cast<std::size_t>(k));
    for (int col = 0; col <= k; ++col) {
        target_host[static_cast<std::size_t>(col)] = (17 + col * 7919) % kTokenDomain;
        if (col < k) {
            draft_host[static_cast<std::size_t>(col)] = target_host[static_cast<std::size_t>(col)];
        }
    }
    DBuf targets = make_i32(target_host);
    DBuf drafts  = make_i32(draft_host);
    DBuf length  = make_i32({128});
    DBuf token   = make_i32({-1});
    DBuf sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    DBuf num(sizeof(std::int32_t));
    DBuf accepted(sizeof(std::int32_t));
    DBuf stats = make_i64_zero(kStats);

    Tensor ttargets(targets.p, DType::I32, {k + 1});
    Tensor tlogits(logits.p, DType::BF16, {kPhysicalRows, k + 1});
    Tensor tdrafts(drafts.p, DType::I32, {k});
    Tensor tlength(length.p, DType::I32, {1});
    Tensor ttoken(token.p, DType::I32, {1});
    Tensor tsampled(sampled.p, DType::I32, {k + 1});
    Tensor tnum(num.p, DType::I32, {1});
    Tensor taccepted(accepted.p, DType::I32, {1});
    Tensor tstats(stats.p, DType::I64, {kStats});
    WorkspaceArena workspace(ops::sampling_workspace_bytes(kTokenDomain, k + 1));
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(config.p);

    const double bytes  = mode == Mode::Greedy ? static_cast<double>((k + 1) * 4 + k * 4)
                                               : stochastic_payload_bytes(k + 1, counts_active);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::speculative_accept_greedy_drafts(ttargets, tlogits, tdrafts, tlength, ttoken,
                                                  tsampled, tnum, taccepted, tstats, kTokenDomain,
                                                  config_ptr, workspace, stream);
        },
        bytes);
    const std::string label = std::string("G3 K=") + std::to_string(k) + " " +
                              (mode == Mode::Greedy ? "greedy" : "stochastic") +
                              (counts_active && mode == Mode::Stochastic ? " counts" : "") +
                              " top_k=" + std::to_string(top_k);
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        const int max_cols    = options.matrix ? 6 : (options.mtp ? options.mtp_k + 1 : 1);
        DBuf logits           = make_logits(max_cols);
        DBuf counts(static_cast<std::size_t>(kTokenDomain) * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));

        std::printf("payload: physical_rows=%d token_domain=%d logits=%.3f MiB/col "
                    "counts=%.3f MiB/col\n",
                    kPhysicalRows, kTokenDomain, static_cast<double>(kTokenDomain * 2) / 1048576.0,
                    static_cast<double>(kTokenDomain * 4) / 1048576.0);
        std::printf("workspace: C=1 %.1f KiB, C=6 %.1f KiB\n",
                    static_cast<double>(ops::sampling_workspace_bytes(kTokenDomain, 1)) / 1024.0,
                    static_cast<double>(ops::sampling_workspace_bytes(kTokenDomain, 6)) / 1024.0);
        if (options.matrix) {
            run_sample(logits, counts, Mode::Greedy, false, 1);
            run_sample(logits, counts, Mode::Stochastic, false, 20);
            run_sample(logits, counts, Mode::Stochastic, true, 20);
            for (int k = 1; k <= 5; ++k) {
                run_mtp(logits, counts, k, Mode::Greedy, false, 1);
                run_mtp(logits, counts, k, Mode::Stochastic, true, 20);
            }
        } else {
            if (options.sample) {
                run_sample(logits, counts, options.mode, options.counts_active, options.top_k);
            }
            if (options.mtp) {
                run_mtp(logits, counts, options.mtp_k, options.mode, options.counts_active,
                        options.top_k);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_sampling_select_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
