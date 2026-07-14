#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/kv_cache.h"
#include "ninfer/core/state_store.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/core/weight_store_parser.h"
#include "ninfer/model/config.h"
#include "ninfer/model/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

struct Options {
    std::string weights = "out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus";
    int device          = 0;
    int warmup          = 1;
    int reps            = 3;
};

void usage(const char* argv0) {
    std::cout << "usage: " << argv0 << " [--weights <path>] [--device <id>] [--warmup <n>]"
              << " [--reps <n>]\n";
}

Options parse_args(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(name) + " needs value"); }
            return argv[++i];
        };
        if (arg == "--weights") {
            out.weights = need_value("--weights");
        } else if (arg == "--device") {
            out.device = std::stoi(need_value("--device"));
        } else if (arg == "--warmup") {
            out.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--reps") {
            out.reps = std::stoi(need_value("--reps"));
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_target_verify_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (out.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (out.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (out.reps <= 0) { throw std::invalid_argument("--reps must be positive"); }
    return out;
}

ninfer::model::StepState make_step_state(ninfer::DeviceArena& arena, int window_cols, int prefill_chunk) {
    const int draft_cols = std::max(1, window_cols - 1);
    return ninfer::model::StepState{
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::BF16, {ninfer::model::kCfg.vocab, window_cols}),
        arena.alloc(ninfer::DType::BF16, {ninfer::model::kCfg.hidden, window_cols}),
        arena.alloc(ninfer::DType::BF16, {ninfer::model::kCfg.hidden, prefill_chunk}),
        arena.alloc(ninfer::DType::I32, {window_cols}),
        arena.alloc(ninfer::DType::I32, {draft_cols}),
        arena.alloc(ninfer::DType::I32, {window_cols}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {window_cols}),
        arena.alloc(ninfer::DType::I32, {window_cols}),
        arena.alloc(ninfer::DType::I32, {window_cols}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::I32, {1}),
        arena.alloc(ninfer::DType::BF16, {ninfer::model::kCfg.hidden, 1}),
        arena.alloc(ninfer::DType::I64, {ninfer::model::kStepStatsCounters}),
    };
}

void copy_i32_scalar(int value, ninfer::Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
}

void copy_i32_vector(const std::vector<int>& value, ninfer::Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, value.data(), value.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
}

double mean_ms(const std::vector<float>& ms) {
    return std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "target_verify_bench: " << e.what() << '\n';
        return 2;
    }

    const std::filesystem::path weights_path(options.weights);
    if (!std::filesystem::exists(weights_path)) {
        std::cout << "SKIP: weights file not present: " << weights_path << '\n';
        return 0;
    }

    std::optional<ninfer::DeviceContext> ctx;
    ninfer::WeightStore weights;
    try {
        weights.prepare(weights_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "target_verify_bench: " << e.what() << '\n';
        return 1;
    }

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ctx.emplace(options.device);
        constexpr int kMaxVerifyT   = 6;
        constexpr int kMaxContext   = 128;
        constexpr int kPrefillChunk = 128;
        ninfer::DeviceArena cache_arena(3ULL * kGiB);
        ninfer::WorkspaceArena workspace(3ULL * kGiB);
        weights.upload(*ctx);

        ninfer::KVCache kv(cache_arena, ninfer::model::kCfg.n_full(), kMaxContext, ninfer::model::kCfg.n_kv,
                        ninfer::model::kCfg.head_dim);
        ninfer::GdnState state(cache_arena, ninfer::model::kCfg.n_gdn(), ninfer::model::kCfg.conv_dim,
                            ninfer::model::kCfg.gdn_conv_state_width, ninfer::model::kCfg.gdn_v_heads,
                            ninfer::model::kCfg.gdn_v_dim, ninfer::model::kCfg.gdn_k_dim, kMaxVerifyT);
        ninfer::model::StepState io = make_step_state(cache_arena, kMaxVerifyT, kPrefillChunk);
        ninfer::model::Qwen3_6_27B card(*ctx, weights, workspace, kv, state, io, kPrefillChunk);

        ninfer::Tensor ids       = cache_arena.alloc(ninfer::DType::I32, {kMaxVerifyT});
        ninfer::Tensor positions = cache_arena.alloc(ninfer::DType::I32, {kMaxVerifyT});
        copy_i32_vector({1, 2, 3, 4, 5, 6}, ids, ctx->stream);
        copy_i32_vector({0, 1, 2, 3, 4, 5}, positions, ctx->stream);
        ctx->synchronize();

        auto reset_decode_state = [&] {
            kv.reset();
            state.reset(ctx->stream);
            workspace.reset();
            copy_i32_scalar(1, io.token, ctx->stream);
            copy_i32_scalar(0, io.pos, ctx->stream);
            copy_i32_scalar(0, io.rope_pos, ctx->stream);
            copy_i32_scalar(0, io.rope_delta, ctx->stream);
            copy_i32_scalar(0, io.gdn_initial_slot, ctx->stream);
            ctx->synchronize();
        };
        auto reset_verify_state = [&] {
            kv.reset();
            state.reset(ctx->stream);
            workspace.reset();
            copy_i32_scalar(0, io.rope_delta, ctx->stream);
            copy_i32_scalar(0, io.gdn_initial_slot, ctx->stream);
            ctx->synchronize();
        };
        auto time_decode = [&] {
            reset_decode_state();
            ninfer::CudaEventTimer timer(*ctx);
            timer.start();
            card.decode_step();
            return timer.stop_ms();
        };
        auto time_verify = [&](int T) {
            reset_verify_state();
            ninfer::Tensor ids_t       = ids.slice(0, 0, T);
            ninfer::Tensor positions_t = positions.slice(0, 0, T);
            ninfer::CudaEventTimer timer(*ctx);
            timer.start();
            card.target_verify(ids_t, positions_t);
            return timer.stop_ms();
        };

        for (int i = 0; i < options.warmup; ++i) {
            (void)time_decode();
            for (int T = 2; T <= kMaxVerifyT; ++T) { (void)time_verify(T); }
        }

        std::vector<float> decode_ms;
        decode_ms.reserve(static_cast<std::size_t>(options.reps));
        for (int i = 0; i < options.reps; ++i) { decode_ms.push_back(time_decode()); }
        const double decode_mean = mean_ms(decode_ms);

        std::cout << "target_verify_bench\n";
        std::cout << "weights," << weights_path << '\n';
        std::cout << "device," << ctx->props.name << '\n';
        std::cout << "warmup," << options.warmup << '\n';
        std::cout << "reps," << options.reps << '\n';
        std::cout << "decode_t1_ms," << decode_mean << '\n';
        std::cout << "T,verify_ms,ratio_vs_decode,epsilon\n";
        for (int T = 2; T <= kMaxVerifyT; ++T) {
            std::vector<float> verify_ms;
            verify_ms.reserve(static_cast<std::size_t>(options.reps));
            for (int i = 0; i < options.reps; ++i) { verify_ms.push_back(time_verify(T)); }
            const double verify_mean = mean_ms(verify_ms);
            const double ratio       = verify_mean / decode_mean;
            std::cout << T << ',' << verify_mean << ',' << ratio << ',' << (ratio - 1.0) << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "target_verify_bench: " << e.what() << '\n';
        return 1;
    }
}
