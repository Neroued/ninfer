#include "ninfer/ops/loop_message.h"
#include "op_measurement.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <memory>

using namespace ninfer;
using namespace ninfer::bench;
using namespace ninfer::bench::measurement;

namespace {
using Bf16                       = __nv_bfloat16;
constexpr int kThreads           = 256;
constexpr float kEps             = 1.e-6F;
constexpr std::size_t kBlasBytes = 32ULL << 20;

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS: ") + cublasGetStatusString(status));
}

// The unfused reference pipeline uses two separate RMS kernels and two cuBLAS
// BF16 contractions. The second GEMM publishes FP32 logits.
__global__ void pair_rms(const float* dest, const float* msg, Bf16* pair, int d, int m,
                         bool broadcast) {
    const int row = blockIdx.x, half = blockIdx.y;
    const auto* src =
        half ? msg + std::int64_t(row) * d : dest + std::int64_t(broadcast ? row % m : row) * d;
    auto* dst = pair + (std::int64_t(row) * 2 + half) * d;
    __shared__ float sums[kThreads / 32], scale;
    float sum = 0;
    for (int i = threadIdx.x; i < d; i += kThreads) sum = fmaf(src[i], src[i], sum);
    sum = ops::common::block_reduce_sum<kThreads>(sum, sums);
    if (threadIdx.x == 0) scale = rsqrtf(__fdiv_rn(sum, float(d)) + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += kThreads) dst[i] = __float2bfloat16_rn(src[i] * scale);
}

__global__ void blend_rms(const float* dest, const float* msg, const float* logits,
                          const Bf16* gamma, Bf16* out, int d, int m, bool broadcast) {
    const int row  = blockIdx.x;
    const auto* dx = dest + std::int64_t(broadcast ? row % m : row) * d;
    const auto* mx = msg + std::int64_t(row) * d;
    const auto* g  = logits + std::int64_t(row) * (2 * d);
    auto* y        = out + std::int64_t(row) * d;
    extern __shared__ float blended[];
    __shared__ float sums[kThreads / 32], scale;
    float ss = 0;
    for (int i = threadIdx.x; i < d; i += kThreads) {
        const float x =
            __fadd_rn(__fmul_rn(1.F - tanhf(g[i + d]), dx[i]), __fmul_rn(tanhf(g[i]), mx[i]));
        blended[i] = x;
        ss         = fmaf(x, x, ss);
    }
    ss = ops::common::block_reduce_sum<kThreads>(ss, sums);
    if (threadIdx.x == 0) scale = rsqrtf(__fdiv_rn(ss, float(d)) + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += kThreads)
        y[i] = __float2bfloat16_rn(blended[i] * scale * float(gamma[i]));
}

struct Blas {
    Blas() : workspace(make_zeros(kBlasBytes)) {
        check_blas(cublasCreate(&handle));
        try {
            check_blas(cublasSetStream(handle, device_context().stream()));
            check_blas(cublasSetWorkspace(handle, workspace.data(), workspace.size()));
        } catch (...) {
            cublasDestroy(handle);
            throw;
        }
    }

    ~Blas() { cublasDestroy(handle); }

    void gemm(int n, int m, int k, int s, const void* w, const void* x, void* y,
              cudaDataType_t output) const {
        constexpr float alpha = 1.F, beta = 0.F;
        check_blas(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k,
            std::int64_t(k) * n, x, CUDA_R_16BF, k, std::int64_t(k) * m, &beta, y, output, n,
            std::int64_t(n) * m, s, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    DeviceBuffer workspace;
    cublasHandle_t handle{};
};

struct BenchOptions {
    Options timing;
    int d = 2304, r = 64;
    std::vector<int> streams{1, 3};
    std::vector<int> tokens;
    std::string variant = "both";
    bool broadcast      = false;
};

struct Fixture {
    Fixture(int d, int r, int s, int m, bool broadcast)
        : d(d), r(r), s(s), m(m), sd(broadcast ? 1 : s),
          x(make_f32(std::size_t(d) * m * sd, 101, -1.F, 1.F)),
          msg(make_f32(std::size_t(d) * m * s, 103, -1.F, 1.F)),
          w(make_bf16(std::size_t(2) * d * r * s, 107, -0.04F, 0.04F)),
          u(make_bf16(std::size_t(2) * d * r * s, 109, -0.2F, 0.2F)),
          gamma(make_bf16(d, 113, 0.75F, 1.25F)), y(make_zeros(std::size_t(d) * m * s * 2)),
          scratch(device_context().device(),
                  ops::loop_message_blend_rmsnorm_workspace_capacity_bytes(d, r, s, m, m)),
          dest(x.span(), DType::FP32, {d, m, sd}), message(msg.span(), DType::FP32, {d, m, s}),
          norm(gamma.span(), DType::BF16, {d}), out(y.span(), DType::BF16, {d, m, s}),
          weights{Tensor(w.span(), DType::BF16, {2 * d, r, s}),
                  Tensor(u.span(), DType::BF16, {r, 2 * d, s})},
          arena(scratch.span()) {}

    void fused(CudaExecutionView cuda) {
        ops::loop_message_blend_rmsnorm(dest, message, weights, norm, kEps, kEps, arena, out, cuda);
    }

    int d, r, s, m, sd;
    DeviceBuffer x, msg, w, u, gamma, y, scratch;
    Tensor dest, message, norm, out;
    ops::LoopMessageWeights weights;
    MemoryArena arena;
};

void run(const BenchOptions& opt, int s, int m, const Blas* blas, L2FlushBuffer& flush, Csv& csv) {
    if (std::int64_t(m) * s > std::numeric_limits<int>::max())
        throw std::invalid_argument("benchmark row count exceeds the CUDA grid");
    Fixture f(opt.d, opt.r, s, m, opt.broadcast);
    const int rows = m * s, d = f.d;
    const std::size_t latent_bytes = std::size_t(f.r) * rows * 2;
    const std::size_t logical_bytes =
        f.x.size() + f.msg.size() + f.w.size() + f.u.size() + f.gamma.size() + f.y.size();
    // Use the same compulsory public I/O floor for both implementations.
    const double flops = 8.0 * d * f.r * rows;
    const auto cuda    = device_context().execution_view();
    auto record = [&](const char* policy, std::size_t workspace, const std::string& pipeline_bytes,
                      auto launch) {
        for (const auto& execution : selections(opt.timing.execution, "graph", "eager")) {
            for (const auto& cache : selections(opt.timing.cache, "cold", "warm")) {
                const auto result = measure(
                    launch, [](CudaExecutionView) {}, flush, cuda, opt.timing, execution, cache);
                const double mem_us  = logical_bytes / (kRoofline.sustained_read_gbs * 1000);
                const double math_us = flops / (kRoofline.bf16_f32_tflops * 1.e6);
                const double floor   = std::max(mem_us, math_us);
                const std::size_t peak =
                    policy == std::string_view("fused") ? f.arena.peak_used() : workspace;
                csv.write({.op              = "loop_message_blend_rmsnorm",
                           .format          = "bf16",
                           .policy          = policy,
                           .n               = d,
                           .k               = f.r,
                           .tokens          = m,
                           .batch           = s,
                           .workspace_bytes = workspace,
                           .workspace_peak  = peak},
                          result, execution, cache,
                          std::to_string(f.sd) + ',' + std::to_string(logical_bytes) + ',' +
                              pipeline_bytes + ',' + format_metric(flops, 0) + ',' +
                              format_metric(floor, 3) + ',' +
                              format_metric(100 * floor / result.gpu.median_us, 2));
            }
        }
    };
    if (opt.variant != "unfused")
        record("fused", f.arena.capacity(), "", [&](CudaExecutionView v) { f.fused(v); });
    if (blas) {
        auto pair                = make_zeros(std::size_t(d) * rows * 4);
        auto h                   = make_zeros(latent_bytes);
        auto logits              = make_zeros(std::size_t(d) * rows * 8);
        const auto intermediates = pair.size() + h.size() + logits.size();
        // Count each stage's input and output tensors, including broadcast
        // inputs at each row that consumes them. Actual memory traffic is
        // affected by cache reuse and the tile schedule inside each kernel.
        const std::size_t row_elements = std::size_t(rows) * d;
        const auto activation_reads    = 2 * (2 * row_elements * sizeof(float));
        const auto gamma_reads         = row_elements * sizeof(Bf16);
        const auto pipeline_bytes      = activation_reads + gamma_reads + f.w.size() + f.u.size() +
                                    f.y.size() + 2 * intermediates;
        record("unfused", intermediates + kBlasBytes, std::to_string(pipeline_bytes),
               [&](CudaExecutionView v) {
                   const auto* x   = static_cast<const float*>(f.x.data());
                   const auto* msg = static_cast<const float*>(f.msg.data());
                   pair_rms<<<dim3(rows, 2), kThreads, 0, v.stream()>>>(
                       x, msg, static_cast<Bf16*>(pair.data()), d, m, opt.broadcast);
                   blas->gemm(f.r, m, 2 * d, s, f.w.data(), pair.data(), h.data(), CUDA_R_16BF);
                   blas->gemm(2 * d, m, f.r, s, f.u.data(), h.data(), logits.data(), CUDA_R_32F);
                   blend_rms<<<rows, kThreads, d * sizeof(float), v.stream()>>>(
                       x, msg, static_cast<const float*>(logits.data()),
                       static_cast<const Bf16*>(f.gamma.data()), static_cast<Bf16*>(f.y.data()), d,
                       m, opt.broadcast);
                   CUDA_CHECK(cudaGetLastError());
               });
    }
}
} // namespace

int main(int argc, char** argv) try {
    BenchOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "loop_message_bench [--d D] [--rank R] [--streams 1,3] "
                         "[--tokens LIST | --sweep FIRST:LAST[:STEP]] "
                         "[--variant fused|unfused|both] [--broadcast 0|1]\n";
            help();
            return 0;
        }
        const auto arg = take(argc, argv, i);
        if (flag == "--d")
            opt.d = integer(arg, 8, 8192);
        else if (flag == "--rank")
            opt.r = integer(arg);
        else if (flag == "--streams")
            opt.streams = integers(arg);
        else if (flag == "--tokens")
            opt.tokens = integers(arg);
        else if (flag == "--sweep")
            opt.tokens = sweep(arg);
        else if (flag == "--broadcast")
            opt.broadcast = integer(arg, 0, 1) != 0;
        else if (flag == "--variant" && (arg == "fused" || arg == "unfused" || arg == "both"))
            opt.variant = arg;
        else if (!parse(opt.timing, flag, arg))
            throw std::invalid_argument("unknown or invalid option " + std::string(flag));
    }
    if (opt.tokens.empty())
        opt.tokens = {1,  2,   3,   4,   7,   8,   15,   16,   17,   31,   32,   63,  64,
                      65, 128, 129, 256, 509, 512, 1021, 1024, 2048, 2053, 4093, 4096};
    hardware();
    std::unique_ptr<Blas> blas;
    if (opt.variant != "fused") blas = std::make_unique<Blas>();
    L2FlushBuffer flush(opt.timing.flush_bytes);
    Csv csv(opt.timing.csv_out,
            "dest_streams,logical_bytes,pipeline_bytes,useful_flops,roofline_us,roofline_pct");
    for (int s : opt.streams)
        for (int m : opt.tokens) run(opt, s, m, blas.get(), flush, csv);
    return 0;
} catch (const std::exception& error) {

    std::cerr << "ninfer_loop_message_bench: " << error.what() << '\n';
    return 2;
}
