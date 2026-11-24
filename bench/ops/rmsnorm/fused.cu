#include "op_measurement.cuh"
#include "common/roofline.h"
#include "common/gpu_activity.cuh"
#include "ninfer/ops/paired_rmsnorm_concat.h"
#include "ninfer/ops/residual_add_rmsnorm.h"

#include <optional>

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        measurement::Options timing;
        bool paired = true, alias = false;
        int d = 4096, calls = 1;
        std::vector<int> columns{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help") {
                std::cout << "Fused normalization public Op benchmark\n"
                             "  --op paired|residual --dim D --columns M[,M...] --calls N\n"
                             "  --alias disjoint|inplace (residual only)\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (flag == "--op") {
                if (value != "paired" && value != "residual")
                    throw std::invalid_argument("unknown Op");
                paired = value == "paired";
            } else if (flag == "--dim")
                d = measurement::integer(value);
            else if (flag == "--alias") {
                if (value != "disjoint" && value != "inplace")
                    throw std::invalid_argument("alias: disjoint|inplace");
                alias = value == "inplace";
            } else if (flag == "--columns")
                columns = measurement::integers(value);
            else if (flag == "--calls")
                calls = measurement::integer(value);
            else if (!measurement::parse(timing, flag, value))
                throw std::invalid_argument("unknown option");
        }
        if (calls > 1 && timing.cache != "warm")
            throw std::invalid_argument("--calls > 1 requires --cache warm");
        if (alias && (paired || calls != 1))
            throw std::invalid_argument("inplace requires one residual call");
        measurement::hardware();
        L2FlushBuffer flush(timing.flush_bytes);
        measurement::Csv csv(timing.csv_out,
                             "semantic_bytes,logical_gb_s,bandwidth_reference_us,calls,alias");
        for (int m : columns) {
            const auto count = std::size_t(d) * m;
            auto x = make_bf16(count, 3901, -2.F, 2.F), y = make_bf16(count, 3903, -1.F, 1.F),
                 gx = make_bf16(d, 3907, -1.F, 1.F), gy = make_bf16(d, 3909, -1.F, 1.F),
                 out = make_zeros(count * (paired ? 4 : 2)), residual = make_zeros(count * 2);
            auto enabled = make_zeros(std::size_t(m) * sizeof(int));
            const std::vector<int> host_enabled(m, 1);
            CUDA_CHECK(cudaMemcpy(enabled.data(), host_enabled.data(),
                                  host_enabled.size() * sizeof(int), cudaMemcpyHostToDevice));
            Tensor tx(x.span(), DType::BF16, {d, m}), ty(y.span(), DType::BF16, {d, m}),
                wx(gx.span(), DType::BF16, {d}), wy(gy.span(), DType::BF16, {d}),
                mask(enabled.span(), DType::I32, {m}),
                result(out.span(), DType::BF16, {paired ? 2 * d : d, m}),
                r(alias ? x.span() : residual.span(), DType::BF16, {d, m});
            std::optional<SavedBuffer> saved;
            if (alias) saved.emplace(x);
            const auto launch = [&](CudaExecutionView e) {
                for (int i = 0; i < calls; ++i)
                    if (paired)
                        ops::paired_rmsnorm_concat(tx, ty, wx, wy, mask, 1.e-5F, result, e);
                    else
                        ops::residual_add_rmsnorm(tx, ty, wx, 1.e-5F, r, result, e);
            };
            for (const auto& execution :
                 measurement::selections(timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(timing.cache, "cold", "warm")) {
                    auto measured = measurement::measure(
                        launch,
                        [&](auto e) {
                            if (saved) saved->restore(e);
                            condition_gpu(e, 200);
                        },
                        flush, device_context().execution_view(), timing, execution, cache);
                    for (auto* time : {&measured.gpu, &measured.host}) {
                        time->median_us /= calls;
                        time->min_us /= calls;
                        time->p95_us /= calls;
                    }
                    measured.kernels /= calls;
                    measurement::Row row;
                    row.op           = paired ? "paired_rmsnorm_concat" : "residual_add_rmsnorm";
                    row.format       = "bf16";
                    row.n            = d;
                    row.tokens       = m;
                    const auto bytes = count * 8 + d * (paired ? 4 : 2) + (paired ? m * 4 : 0);
                    csv.write(
                        row, measured, execution, cache,
                        std::to_string(bytes) + "," +
                            format_metric(bytes / (measured.gpu.median_us * 1000), 3, "") + "," +
                            format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") + "," +
                            std::to_string(calls) + "," + (alias ? "inplace" : "disjoint"));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
