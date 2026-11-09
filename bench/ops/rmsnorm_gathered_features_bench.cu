#include "ninfer/ops/rmsnorm_gathered_features.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/rmsnorm.h"
#include "op_measurement.cuh"

using namespace ninfer;
namespace m = ninfer::bench::measurement;

namespace {
struct Options {
    m::Options timing;
    std::vector<int> columns{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::string form = "fused";
    int calls        = 1;
};

Options parse(int argc, char** argv) {
    Options o;
    o.timing.cache = "warm";
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "gathered features RMSNorm [2048,M,2] -> [4096,M], epsilon=1e-5\n"
                         "  --columns M,... --sweep START:END[:STEP]\n"
                         "  --form fused|composed|both --calls N (warm only if N>1)\n";
            m::help();
            std::exit(0);
        }
        const auto value = m::take(argc, argv, i);
        if (flag == "--columns")
            o.columns = m::integers(value);
        else if (flag == "--sweep")
            o.columns = m::sweep(value);
        else if (flag == "--calls")
            o.calls = m::integer(value, 1, 64);
        else if (flag == "--form") {
            if (value != "fused" && value != "composed" && value != "both")
                throw std::invalid_argument("form: fused|composed|both");
            o.form = value;
        } else if (!m::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.calls > 1 && o.timing.cache != "warm")
        throw std::invalid_argument("multiple calls require warm cache");
    if (o.timing.profile && (o.columns.size() != 1 || o.form == "both"))
        throw std::invalid_argument("profile requires one extent and form");
    return o;
}

void run(const Options& o, int columns, m::Csv& csv, bench::L2FlushBuffer& flush) {
    const auto e = bench::device_context().execution_view();
    auto input   = bench::make_bf16(std::size_t(4096) * columns, 51101U);
    auto gain    = bench::make_bf16(4096, 51103U, .25F, 1.75F);
    auto output  = bench::make_buffer(std::size_t(4096) * columns * 2);
    auto concat  = bench::make_buffer(output.size());
    Tensor x(input.span(), DType::BF16, {2048, columns, 2}),
        gamma(gain.span(), DType::BF16, {4096}), y(output.span(), DType::BF16, {4096, columns}),
        joined(concat.span(), DType::BF16, {4096, columns});
    const auto first  = x.slice(2, 0, 1).view({2048, columns});
    const auto second = x.slice(2, 1, 1).view({2048, columns});
    for (const auto& form : m::selections(o.form, "fused", "composed")) {
        const auto launch = [&](CudaExecutionView cuda) {
            for (int i = 0; i < o.calls; ++i) {
                if (form == "fused")
                    ops::rmsnorm_gathered_features(x, gamma, 1.e-5F, y, cuda);
                else {
                    ops::concat_bf16_features(first, second, joined, cuda);
                    ops::rmsnorm(joined, gamma, 1.e-5F, false, y, cuda);
                }
            }
        };
        m::Row row{.op     = form == "fused" ? "rmsnorm_gathered_features" : "concat_rmsnorm",
                   .format = "bf16",
                   .n      = 4096,
                   .tokens = columns};
        const std::size_t bytes = std::size_t(4096) * (4LL * columns + 2);
        for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
            for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
                const auto result = m::measure(
                    launch, [](CudaExecutionView) {}, flush, e, o.timing, execution, cache);
                csv.write(
                    row, result, execution, cache,
                    std::to_string(o.calls) + ',' +
                        bench::format_metric(result.gpu.median_us / o.calls, 3, "") + ',' +
                        std::to_string(bytes) + ',' +
                        std::to_string(form == "fused" ? 0 : 16384ULL * columns) + ',' +
                        std::to_string(16384ULL * columns) + ',' +
                        bench::format_metric(bytes / (bench::kRoofline.memory_gbs * 1000), 3, ""));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out,
                   "calls,per_call_median_us,logical_bytes_per_call,extra_intermediate_bytes_per_"
                   "call,useful_flops_per_call,bandwidth_reference_us_per_call");
        for (int columns : o.columns) run(o, columns, csv, flush);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
