#include "ninfer/ops/dynamic_grouped_conv.h"
#include "direct_bf16_weight.cuh"
#include "op_measurement.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    int hidden = 5120, width = 0, batch = 0;
    float epsilon = 1.e-6F;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "Dynamic prepare: --hidden H --epsilon E --width W --batch B\n"
                         "Omit width/batch to scan W2..16,B1..8. Defaults H5120, epsilon1e-6.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--hidden")
            o.hidden = measurement::integer(value);
        else if (flag == "--epsilon")
            o.epsilon = std::stof(std::string(value));
        else if (flag == "--width")
            o.width = measurement::integer(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown dynamic prepare option");
    }
    if (o.width < 0 || o.width == 1 || o.width > 16 || o.batch < 0 || o.batch > 8 ||
        !std::isfinite(o.epsilon) || o.epsilon <= 0)
        throw std::invalid_argument("invalid width, batch or epsilon");
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o        = parse(argc, argv);
        const auto capacity = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            o.hidden, 2, 16, 1, 8);
        const int groups = o.hidden / 16, rows = groups * 4;
        auto residual   = make_bf16(std::size_t(o.hidden) * 128, 101U);
        auto norm       = make_bf16(o.hidden, 103U, .8F, 1.2F);
        auto base       = make_bf16(std::size_t(o.hidden) * 4, 105U);
        auto projection = make_direct_bf16_weight(rows, o.hidden, 0x31U);
        auto prepared   = make_buffer(std::size_t(o.hidden) * 128 * 2);
        auto finish     = make_buffer(std::size_t(groups) * 2 * 128 * 2);
        auto storage    = make_buffer(capacity);
        MemoryArena arena(storage.span());
        Tensor norm_weight(norm.span(), DType::BF16, {o.hidden});
        Tensor base_kernel(base.span(), DType::BF16, {o.hidden, 2, 2});
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::hardware();
        measurement::Csv csv(
            o.timing.csv_out,
            "width,columns,necessary_bytes,useful_flops,effective_gbs,roofline_us");
        for (int width = 2; width <= 16; ++width) {
            if (o.width && width != o.width) continue;
            for (int batch = 1; batch <= 8; ++batch) {
                if (o.batch && batch != o.batch) continue;
                const int m = width * batch;
                Tensor x(residual.span(), DType::BF16, {o.hidden, width, batch});
                Tensor y(prepared.span(), DType::BF16, {o.hidden, width, batch});
                Tensor d(finish.span(), DType::BF16, {groups, 2, width, batch});
                measurement::Row row;
                row.op                    = "rmsnorm_dynamic_grouped_conv_prepare";
                row.format                = "bf16";
                row.policy                = "a16";
                row.n                     = rows;
                row.k                     = o.hidden;
                row.tokens                = width;
                row.batch                 = batch;
                row.workspace_bytes       = capacity;
                const std::uint64_t bytes = projection.model_weight_bytes() +
                                            std::uint64_t(4) * o.hidden * m + 6ULL * o.hidden +
                                            4ULL * groups * m;
                const double flops = 2.0 * rows * o.hidden * m;
                for (const auto& execution :
                     measurement::selections(o.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(o.timing.cache, "cold", "warm")) {
                        arena.reset_peak();
                        const auto result = measurement::measure(
                            [&](auto cuda) {
                                ops::rmsnorm_dynamic_grouped_conv_prepare(
                                    x, norm_weight, o.epsilon, base_kernel, projection.weight, y, d,
                                    arena, cuda);
                            },
                            [](auto) {}, flush, device_context().execution_view(), o.timing,
                            execution, cache);
                        row.workspace_peak = arena.peak_used();
                        const double floor = std::max(bytes / (1000 * kRoofline.sustained_read_gbs),
                                                      flops / (1.e6 * kRoofline.bf16_f32_tflops));
                        csv.write(row, result, execution, cache,
                                  std::to_string(width) + ',' + std::to_string(m) + ',' +
                                      std::to_string(bytes) + ',' + std::to_string(flops) + ',' +
                                      format_metric(bytes / result.gpu.median_us / 1000, 3, "") +
                                      ',' + format_metric(floor, 3, ""));
                    }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "dynamic prepare bench: " << e.what() << '\n';
        return 1;
    }
}
