#include "ninfer/ops/mhc.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"
#include "common/gpu_activity.cuh"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
constexpr int D = 4096, S = 4, K = D * S, N = 24;

struct Options {
    measurement::Options timing;
    std::string op = "pre";
    std::vector<int> columns{1,  2,  3,  4,  5,   6,   7,   8,    12,   16,  24,
                             32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096};
    bool alias       = false;
    int sites        = 1;
    int calls        = 0;
    int condition_us = 200;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "mHC complete Op benchmark, D=4096 S=4\n"
                         "  --op pre|post|expand-pre|post-pre|post-mean-norm\n"
                         "  --columns M[,M...] --alias disjoint|inplace --sites N --calls N\n"
                         "  --condition-us N "
                         "(default 200)\n"
                         "Calls defaults to sites; it must be a positive multiple of sites.\n"
                         "Times are per call; sites selects the number of distinct F weights.\n"
                         "Rotation uses preserved inputs and separate outputs; cache eviction is "
                         "before each cycle.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--condition-us")
            o.condition_us = measurement::integer(value, 0);
        else if (flag == "--sites")
            o.sites = measurement::integer(value);
        else if (flag == "--calls")
            o.calls = measurement::integer(value);
        else if (flag == "--columns")
            o.columns = measurement::integers(value);
        else if (flag == "--alias") {
            if (value != "disjoint" && value != "inplace")
                throw std::invalid_argument("invalid alias");
            o.alias = value == "inplace";
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.op != "pre" && o.op != "post" && o.op != "expand-pre" && o.op != "post-pre" &&
        o.op != "post-mean-norm")
        throw std::invalid_argument("unknown mHC Op");
    if (o.alias && o.op != "post" && o.op != "post-pre")
        throw std::invalid_argument("inplace requires post or post-pre");
    if (o.sites > 1 && (o.alias || o.op == "post" || o.op == "post-mean-norm"))
        throw std::invalid_argument("site rotation requires a pre form with disjoint output");
    if (o.calls == 0) o.calls = o.sites;
    if (o.calls % o.sites != 0 || (o.alias && o.calls > 1))
        throw std::invalid_argument("calls must be a multiple of sites; inplace requires one call");
    return o;
}

std::size_t capacity(const std::string& op, int m) {
    if (op == "post") return 0;
    auto query = ops::mhc_pre_workspace_capacity_bytes;
    if (op == "expand-pre") query = ops::mhc_expand_pre_workspace_capacity_bytes;
    if (op == "post-pre") query = ops::mhc_post_pre_workspace_capacity_bytes;
    if (op == "post-mean-norm") query = ops::mhc_post_mean_norm_workspace_capacity_bytes;
    return query({D, S}, device_context().multiprocessor_count(), m, m);
}

std::size_t semantic_bytes(const std::string& op, int m) {
    const std::size_t block = std::size_t(D) * m * 2, residual = block * S,
                      coeff   = std::size_t(20) * m * 4;
    const std::size_t weights = std::size_t(N) * K * 2 + (D + N + 3) * 2;
    if (op == "post") return residual * 2 + block + coeff;
    if (op == "post-mean-norm") return residual + 2 * block + coeff + D * 2;
    if (op == "expand-pre") return weights + residual + 2 * block + coeff;
    if (op == "pre") return weights + residual + block + coeff;
    return weights + 2 * residual + 2 * block + 2 * coeff;
}

struct Fixture {
    Fixture(const Options& o, int m, const std::vector<DirectBf16Weight>& projections)
        : options(o), projections(projections), m(m),
          x(make_bf16(std::size_t(K) * m, 7101, -2.F, 2.F)),
          y(make_bf16(std::size_t(D) * m, 7103, -1.F, 1.F)), base(make_bf16(N, 7107, -0.5F, 0.5F)),
          scale(make_bf16(3, 7109, 0.5F, 1.5F)), gamma(make_bf16(D, 7111, -1.F, 1.F)),
          q(make_f32(S * m, 7113, 0.F, 2.F)), c(make_f32(S * S * m, 7117, 0.01F, 0.5F)),
          r(make_zeros(std::size_t(K) * m * 2)), u(make_zeros(std::size_t(D) * m * 2)),
          nq(make_zeros(S * m * 4)), nc(make_zeros(S * S * m * 4)),
          backup(o.alias ? make_zeros(x.size()) : DeviceBuffer{}), capacity(::capacity(o.op, m)),
          scratch(device_context().device(), capacity), arena(scratch.span()),
          tx(x.span(), DType::BF16, {D, S, m}), ty(y.span(), DType::BF16, {D, m}),
          tr(o.alias ? x.span() : r.span(), DType::BF16, {D, S, m}),
          weights{projections[0].weight, Tensor(base.span(), DType::BF16, {N}),
                  Tensor(scale.span(), DType::BF16, {3}), Tensor(gamma.span(), DType::BF16, {D})},
          coefficients{Tensor(q.span(), DType::FP32, {S, m}),
                       Tensor(c.span(), DType::FP32, {S, S, m})},
          output{
              Tensor(u.span(), DType::BF16, {D, m}),
              {Tensor(nq.span(), DType::FP32, {S, m}), Tensor(nc.span(), DType::FP32, {S, S, m})}} {
        if (o.alias)
            CUDA_CHECK(cudaMemcpy(backup.data(), x.data(), x.size(), cudaMemcpyDeviceToDevice));
    }

    void restore(CudaExecutionView e) {
        condition_gpu(e, options.condition_us);
        if (options.alias)
            CUDA_CHECK(cudaMemcpyAsync(x.data(), backup.data(), x.size(), cudaMemcpyDeviceToDevice,
                                       e.stream()));
    }

    void launch(CudaExecutionView e) {
        for (int call = 0; call < options.calls; ++call) {
            weights.projection = projections[call % options.sites].weight;
            if (options.op == "pre")
                ops::mhc_pre(tx, weights, parameters, output, arena, e);
            else if (options.op == "post")
                ops::mhc_post(tx, ty, coefficients, tr, e);
            else if (options.op == "expand-pre")
                ops::mhc_expand_pre(ty, weights, parameters, tr, output, arena, e);
            else if (options.op == "post-pre")
                ops::mhc_post_pre(tx, ty, coefficients, weights, parameters, tr, output, arena, e);
            else if (options.op == "post-mean-norm")
                ops::mhc_post_mean_norm(tx, ty, coefficients, weights.gamma, 1.e-5F,
                                        output.block_input, arena, e);
        }
    }

    const Options& options;
    const std::vector<DirectBf16Weight>& projections;
    int m;
    DeviceBuffer x, y, base, scale, gamma, q, c, r, u, nq, nc, backup;
    std::size_t capacity;
    DeviceBuffer scratch;
    MemoryArena arena;
    Tensor tx, ty, tr;
    ops::MhcPreWeights weights;
    ops::MhcCoefficients coefficients;
    ops::MhcPreOutputs output;
    ops::MhcPreParameters parameters{1.e-5F, 1.e-5F, 1.e-6F, 1.e-6F, 2.F, 20};
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        measurement::hardware();
        std::vector<DirectBf16Weight> projections;
        for (int i = 0; i < o.sites; ++i)
            projections.push_back(make_direct_bf16_weight(N, K, 7100 + i, 0.015625F));
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out,
                             "alias,semantic_bytes,logical_gb_s,bandwidth_reference_us,projection_"
                             "flops,bf16_mma_reference_us,calls,sites");
        for (int m : o.columns) {
            Fixture f(o, m, projections);
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    auto result = measurement::measure([&](CudaExecutionView e) { f.launch(e); },
                                                       [&](CudaExecutionView e) { f.restore(e); },
                                                       flush, device_context().execution_view(),
                                                       o.timing, execution, cache);
                    measurement::Row row;
                    row.op              = "mhc_" + o.op;
                    row.format          = "bf16";
                    row.n               = N;
                    row.k               = K;
                    row.value_heads     = S;
                    row.tokens          = m;
                    row.workspace_bytes = f.capacity;
                    row.workspace_peak  = f.arena.peak_used();
                    row.layers          = o.sites;
                    for (auto* timing : {&result.gpu, &result.host}) {
                        timing->median_us /= o.calls;
                        timing->min_us /= o.calls;
                        timing->p95_us /= o.calls;
                    }
                    result.kernels /= o.calls;
                    const std::size_t bytes = semantic_bytes(o.op, m);
                    const double flops =
                        (o.op == "post" || o.op == "post-mean-norm") ? 0 : double(2) * N * K * m;
                    csv.write(
                        row, result, execution,
                        o.sites > 1   ? "rotation-" + cache
                        : o.calls > 1 ? "repeat-" + cache
                                      : cache,
                        std::string(o.alias ? "inplace," : "disjoint,") + std::to_string(bytes) +
                            "," + format_metric(bytes / (result.gpu.median_us * 1000), 3, "") +
                            "," + format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") +
                            "," + format_metric(flops, 0, "") + "," +
                            format_metric(flops / (kRoofline.bf16_f32_tflops * 1.e6), 3, "") + "," +
                            std::to_string(o.calls) + "," + std::to_string(o.sites));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
