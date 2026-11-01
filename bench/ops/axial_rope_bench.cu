#include "op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "common/roofline.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        measurement::Options timing;
        bool norm = true;
        int calls = 1;
        std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help") {
                std::cout << "D64/H16 axial RoPE public Op benchmark\n"
                             "  --op norm-rope|rope --tokens N[,N...] --calls N\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (flag == "--op") {
                if (value != "norm-rope" && value != "rope")
                    throw std::invalid_argument("unknown Op");
                norm = value == "norm-rope";
            } else if (flag == "--tokens")
                tokens = measurement::integers(value);
            else if (flag == "--calls")
                calls = measurement::integer(value);
            else if (!measurement::parse(timing, flag, value))
                throw std::invalid_argument("unknown option");
        }
        if (calls > 1 && timing.cache != "warm")
            throw std::invalid_argument("--calls > 1 requires --cache warm");
        measurement::hardware();
        L2FlushBuffer flush(timing.flush_bytes);
        measurement::Csv csv(timing.csv_out,
                             "semantic_bytes,logical_gb_s,bandwidth_reference_us,calls");
        for (int n : tokens) {
            const size_t count = std::size_t(n) * 1024;
            auto original = make_bf16(count, 6601, -2.F, 2.F), qdata = make_zeros(count * 2),
                 kdata = make_zeros(count * 2), gain = make_bf16(64, 6607, 1.F, 1.F),
                 k_gain = make_bf16(64, 6619, 1.F, 1.F), positions = make_zeros(n * 8ULL);
            std::vector<int> host_positions(n * 2);
            for (int t = 0; t < n; ++t) {
                host_positions[t]     = t % 64;
                host_positions[t + n] = t / 64;
            }
            CUDA_CHECK(cudaMemcpy(positions.data(), host_positions.data(), n * 8ULL,
                                  cudaMemcpyHostToDevice));
            Tensor q(qdata.span(), DType::BF16, {64, 16, n}),
                k(kdata.span(), DType::BF16, {64, 16, n}), g(gain.span(), DType::BF16, {64}),
                kg(k_gain.span(), DType::BF16, {64}), p(positions.span(), DType::I32, {n, 2});
            auto launch = [&](CudaExecutionView e) {
                for (int i = 0; i < calls; ++i)
                    if (norm)
                        ops::rmsnorm_rope_2d(p, g, kg, 1.e-5F, 10000.F, q, k, e);
                    else
                        ops::rope(p, 64, 10000.F, q, k, e);
            };
            auto restore = [&](CudaExecutionView e) {
                CUDA_CHECK(cudaMemcpyAsync(q.data, original.data(), count * 2,
                                           cudaMemcpyDeviceToDevice, e.stream()));
                CUDA_CHECK(cudaMemcpyAsync(k.data, original.data(), count * 2,
                                           cudaMemcpyDeviceToDevice, e.stream()));
                condition_gpu(e, 200);
            };
            for (const auto& execution :
                 measurement::selections(timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(timing.cache, "cold", "warm")) {
                    auto measured = measurement::measure(launch, restore, flush,
                                                         device_context().execution_view(), timing,
                                                         execution, cache);
                    for (auto* time : {&measured.gpu, &measured.host}) {
                        time->median_us /= calls;
                        time->min_us /= calls;
                        time->p95_us /= calls;
                    }
                    measured.kernels /= calls;
                    measurement::Row row;
                    row.op           = norm ? "rmsnorm_rope_2d" : "rope_2d";
                    row.format       = "bf16";
                    row.n            = 64;
                    row.tokens       = n;
                    const auto bytes = count * 8 + n * 8 + (norm ? 256 : 0);
                    csv.write(row, measured, execution, cache,
                              std::to_string(bytes) + "," +
                                  format_metric(bytes / (measured.gpu.median_us * 1000), 3, "") +
                                  "," +
                                  format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") +
                                  "," + std::to_string(calls));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
