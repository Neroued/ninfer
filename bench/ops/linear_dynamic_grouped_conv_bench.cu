#include "ninfer/ops/dynamic_grouped_conv.h"
#include "direct_bf16_weight.cuh"
#include "op_measurement.cuh"
#include "common/roofline.h"
using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    int k = 2048, width = 0, batch = 0;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Linear dynamic contribution: --k K --width W --batch B\nOmit width/batch "
                         "to scan W2..16,B1..8.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--k")
            o.k = measurement::integer(value);
        else if (flag == "--width")
            o.width = measurement::integer(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.width < 0 || o.width == 1 || o.width > 16 || o.batch < 0 || o.batch > 8)
        throw std::invalid_argument("invalid W/B");
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o    = parse(argc, argv);
        constexpr int H = 4096, G = H / 16;
        auto input = make_bf16(std::size_t(o.k) * 128, 101), base = make_bf16(H * 4, 103),
             delta = make_bf16(G * 2 * 128, 105), output = make_buffer(H * 128 * 2);
        auto weight = make_direct_bf16_weight(H, o.k, 107);
        Tensor b(base.span(), DType::BF16, {H, 2, 2});
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::hardware();
        measurement::Csv csv(
            o.timing.csv_out,
            "width,columns,necessary_bytes,useful_flops,effective_gbs,roofline_us");
        for (int w = 2; w <= 16; ++w) {
            if (o.width && w != o.width) continue;
            for (int batch = 1; batch <= 8; ++batch) {
                if (o.batch && batch != o.batch) continue;
                const int m = w * batch;
                Tensor x(input.span(), DType::BF16, {o.k, w, batch}),
                    d(delta.span(), DType::BF16, {G, 2, w, batch}),
                    y(output.span(), DType::BF16, {H, w, batch});
                const auto bytes = weight.model_weight_bytes() + 2ULL * (o.k + H) * m + 4ULL * H +
                                   4ULL * G * m - 2ULL * G * batch;
                const double flops = 2. * H * o.k * m;
                measurement::Row row;
                row.op     = "linear_dynamic_grouped_conv";
                row.format = "bf16";
                row.policy = "a16";
                row.n      = H;
                row.k      = o.k;
                row.tokens = w;
                row.batch  = batch;
                for (const auto& exec :
                     measurement::selections(o.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(o.timing.cache, "cold", "warm")) {
                        const auto timing = measurement::measure(
                            [&](auto cuda) {
                                ops::linear_dynamic_grouped_conv(x, weight.weight, b, d, y, cuda);
                            },
                            [](auto) {}, flush, device_context().execution_view(), o.timing, exec,
                            cache);
                        const auto floor = std::max(bytes / (1000 * kRoofline.sustained_read_gbs),
                                                    flops / (1.e6 * kRoofline.bf16_f32_tflops));
                        csv.write(row, timing, exec, cache,
                                  std::to_string(w) + ',' + std::to_string(m) + ',' +
                                      std::to_string(bytes) + ',' + std::to_string(flops) + ',' +
                                      format_metric(bytes / timing.gpu.median_us / 1000, 3, "") +
                                      ',' + format_metric(floor, 3, ""));
                    }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
