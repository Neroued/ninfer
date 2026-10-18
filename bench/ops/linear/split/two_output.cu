#include "ninfer/ops/linear_split.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"

#include <array>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::string format = "q8", policy = "a16";
    int n = 16384, k = 2560;
    std::vector<int> tokens{1};
};

Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "LinearSplit two-output benchmark\n"
                         "  --format q4q5|q8|fp8|nvfp4 --n N --k K --policy a16|allow-a8|allow-a4\n"
                         "  --tokens M[,M...] or --sweep START:END[:STEP]\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--format")
            o.format = value;
        else if (flag == "--n")
            o.n = measurement::integer(value);
        else if (flag == "--k")
            o.k = measurement::integer(value);
        else if (flag == "--policy")
            o.policy = value;
        else if (flag == "--tokens" || flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens =
                flag == "--tokens" ? measurement::integers(value) : measurement::sweep(value);
            explicit_tokens = true;
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.policy != "a16" && o.policy != "allow-a8" && o.policy != "allow-a4")
        throw std::invalid_argument("unknown activation policy");
    if (o.format == "q4q5" && o.policy != "a16")
        throw std::invalid_argument("paired Q4/Q5 weights use A16");
    const bool q8_geometry = (o.n == 12288 && o.k == 2048) || (o.n == 16384 && o.k == 2560) ||
                             (o.n == 8192 && o.k == 2560) || (o.n == 6144 && o.k == 2560);
    const bool supported = o.format == "q8"
                               ? q8_geometry
                               : (o.format == "q4q5" || o.format == "fp8" || o.format == "nvfp4") &&
                                     o.n == 16384 && o.k == 5120;
    if (!supported) throw std::invalid_argument("unsupported format/N/K");
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto policy  = options.policy == "a16"        ? ops::LinearPolicy::A16Only
                             : options.policy == "allow-a8" ? ops::LinearPolicy::AllowA8
                                                            : ops::LinearPolicy::AllowA4;
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        PackedQuantizedWeight first, second;
        ops::ProjectionWeights weights;
        if (options.format == "q4q5") {
            first   = make_row_split_weight(QType::Q4_G64_FP16, 4096, options.k, options.k, 501U);
            second  = make_row_split_weight(QType::Q5_G64_FP16, 12288, options.k, options.k, 503U);
            weights = ops::PairedProjectionWeights{first.weight, second.weight};
        } else {
            if (options.format == "q8")
                first = make_row_split_weight(QType::Q8_G32_FP16, options.n, options.k, options.k,
                                              501U);
            else if (options.format == "fp8")
                first = make_fp8_weight(options.n, options.k, 501U);
            else
                first = make_nvfp4_weight(options.n, options.k, 501U);
            weights = ops::SingleProjectionWeight{first.weight, policy};
        }
        const int tail  = (options.n - 4096) / 2;
        const auto rows = options.n == 6144   ? std::array{3072, 3072}
                          : options.n == 8192 ? std::array{5120, 3072}
                                              : std::array{4096 + tail, tail};
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out);
        for (int t : options.tokens) {
            auto input          = make_bf16(std::size_t(options.k) * t, 101U);
            auto output0        = make_zeros(std::size_t(rows[0]) * t * 2);
            auto output1        = make_zeros(std::size_t(rows[1]) * t * 2);
            const auto capacity = ops::linear_split_workspace_capacity_bytes(weights, rows, t, t);
            ninfer::DeviceBuffer arena_storage(ninfer::bench::device_context().device(),
                                               256 + capacity);
            MemoryArena arena(arena_storage.span());
            (void)arena.alloc_bytes(256);
            Tensor x(input.span(), DType::BF16, {options.k, t});
            const std::array outputs{Tensor(output0.span(), DType::BF16, {rows[0], t}),
                                     Tensor(output1.span(), DType::BF16, {rows[1], t})};
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) {
                            ops::linear_split(x, weights, outputs, arena, cuda);
                        },
                        [](auto) {}, flush, device.execution_view(), options.timing, execution,
                        cache);
                    measurement::Row row;
                    row.op              = "linear_split";
                    row.format          = options.format;
                    row.policy          = options.policy;
                    row.n               = options.n;
                    row.k               = options.k;
                    row.tokens          = t;
                    row.batch           = 1;
                    row.workspace_bytes = capacity;
                    row.workspace_peak  = arena.peak_used() - 256;
                    csv.write(row, result, execution, cache);
                }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
