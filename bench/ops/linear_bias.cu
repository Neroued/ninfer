#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        measurement::Options timing;
        std::vector<int> tokens{1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        int n = 1024, k = 1176;
        std::string form = "projection";
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help") {
                std::cout << "LinearBias BF16/A16: --n N --k K --tokens T[,T...] --sweep A:B[:S]\n "
                             " --form projection|split|swiglu\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (flag == "--form")
                form = value;
            else if (flag == "--n")
                n = measurement::integer(value);
            else if (flag == "--k")
                k = measurement::integer(value);
            else if (flag == "--tokens")
                tokens = measurement::integers(value);
            else if (flag == "--sweep")
                tokens = measurement::sweep(value);
            else if (!measurement::parse(timing, flag, value))
                throw std::invalid_argument("unknown option");
        }
        if (form != "projection" && form != "split" && form != "swiglu")
            throw std::invalid_argument("unknown form");
        measurement::hardware();
        const auto maximum = *std::max_element(tokens.begin(), tokens.end());
        auto weight        = make_direct_bf16_weight(n, k);
        auto input         = make_bf16(std::size_t(k) * maximum, 17);
        const int rows     = form == "swiglu" ? n / 2 : n;
        const int segments = form == "split" ? 3 : 1;
        std::vector<DeviceBuffer> outputs;
        for (int s = 0; s < segments; ++s)
            outputs.push_back(make_buffer(std::size_t(rows / segments) * maximum * 2));
        auto bias = make_bf16(n, 19);
        Tensor b(bias.span(), DType::BF16, {n});
        L2FlushBuffer flush(timing.flush_bytes);
        measurement::Csv csv(timing.csv_out,
                             "operand_bytes,operand_gb_s,useful_tflops,bandwidth_reference_us,"
                             "compute_reference_us,roofline_reference_us");
        for (int t : tokens) {
            Tensor x(input.span(), DType::BF16, {k, t});
            std::vector<Tensor> y;
            for (auto& output : outputs)
                y.emplace_back(output.span(), DType::BF16,
                               std::initializer_list<int>{rows / segments, t});
            measurement::Row row;
            row.op               = form == "projection" ? "linear_bias"
                                   : form == "split"    ? "linear_split_bias"
                                                        : "linear_swiglu_bias";
            row.format           = "bf16";
            row.policy           = "a16";
            row.n                = n;
            row.k                = k;
            row.tokens           = t;
            const auto bytes     = weight.model_weight_bytes() + 2ULL * n + 2ULL * (rows + k) * t;
            const double flops   = 2.0 * n * k * t;
            const double compute = t > 1 ? flops / (kRoofline.bf16_f32_tflops * 1e6)
                                         : std::numeric_limits<double>::quiet_NaN();
            for (const auto& execution :
                 measurement::selections(timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](auto cuda) {
                            if (form == "projection")
                                ops::linear_bias(x, weight.weight, b, y[0], cuda);
                            else if (form == "split")
                                ops::linear_split(x, weight.weight, b, y, cuda);
                            else
                                ops::linear_swiglu(x, weight.weight, b, y[0], 10.F, cuda);
                        },
                        [](auto) {}, flush, device_context().execution_view(), timing, execution,
                        cache);
                    const double memory   = cache == "cold"
                                                ? bytes / (kRoofline.sustained_read_gbs * 1000)
                                                : std::numeric_limits<double>::quiet_NaN();
                    const double roofline = cache == "cold" && t > 1
                                                ? std::max(memory, compute)
                                                : std::numeric_limits<double>::quiet_NaN();
                    csv.write(row, result, execution, cache,
                              std::to_string(bytes) + ',' +
                                  format_metric(bytes / (result.gpu.median_us * 1000), 3, "") +
                                  ',' + format_metric(flops / (result.gpu.median_us * 1e6), 3, "") +
                                  ',' + format_metric(memory, 3, "") + ',' +
                                  format_metric(compute, 3, "") + ',' +
                                  format_metric(roofline, 3, ""));
                }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
