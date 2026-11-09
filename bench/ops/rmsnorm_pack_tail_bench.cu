// Complete fused normalization and packing, with one public Op per sample.
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "op_measurement.cuh"
#include <iomanip>
#include <sstream>
using namespace ninfer;
namespace m = ninfer::bench::measurement;

namespace {
struct Options {
    m::Options timing;
    int hidden = 5120, calls = 1;
    float epsilon           = 1.e-6F;
    std::vector<int> widths = m::sweep("2:16"), batches = m::sweep("1:8");
};

Options parse(int argc, char** argv) {
    Options o;
    o.timing.cache = "warm";
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "usage: ninfer_rmsnorm_pack_tail_bench [--hidden 4096|5120] [--epsilon E]\n"
                   "  --widths W,... (2..16) --batches B,... (1..8)\n"
                   "  --calls N (1..64): repeated read-only inputs, warm cache only\n";
            m::help();
            std::exit(0);
        }
        const auto value = m::take(argc, argv, i);
        if (flag == "--hidden")
            o.hidden = m::integer(value, 4096, 5120);
        else if (flag == "--calls")
            o.calls = m::integer(value, 1, 64);
        else if (flag == "--epsilon") {
            std::size_t end;
            o.epsilon = std::stof(std::string(value), &end);
            if (end != value.size() || !(o.epsilon > 0) || !std::isfinite(o.epsilon))
                throw std::invalid_argument("epsilon must be positive and finite");
        } else if (flag == "--widths")
            o.widths = m::integers(value, 2, 16);
        else if (flag == "--batches")
            o.batches = m::integers(value, 1, 8);
        else if (!m::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown flag: " + std::string(flag));
    }
    if (o.hidden != 4096 && o.hidden != 5120) throw std::invalid_argument("hidden: 4096|5120");
    if (o.timing.profile && o.widths.size() * o.batches.size() != 1)
        throw std::invalid_argument("profile requires one extent");
    if (o.calls > 1 && o.timing.cache != "warm")
        throw std::invalid_argument("multiple calls require --cache warm");
    return o;
}

void run(const Options& o, int width, int batch, bench::L2FlushBuffer& flush, m::Csv& csv,
         CudaExecutionView e) {
    const int columns = (width - 1) * batch;
    auto input        = bench::make_bf16(std::size_t(o.hidden) * width * batch, 101U);
    auto weight       = bench::make_bf16(o.hidden, 103U, .8F, 1.2F);
    auto output       = bench::make_buffer(std::size_t(o.hidden) * columns * 2);
    Tensor x(input.span(), DType::BF16, {o.hidden, width, batch}),
        w(weight.span(), DType::BF16, {o.hidden}),
        y(output.span(), DType::BF16, {o.hidden, columns});
    const auto launch = [&](CudaExecutionView cuda) {
        for (int i = 0; i < o.calls; ++i) ops::rmsnorm_pack_tail(x, w, o.epsilon, y, cuda);
    };
    const std::size_t bytes = std::size_t(o.hidden) * (4 * columns + 2);
    m::Row row{.op     = "rmsnorm_pack_tail",
               .format = "bf16",
               .n      = o.hidden,
               .tokens = width,
               .batch  = batch};
    for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
            const auto result =
                m::measure(launch, [](CudaExecutionView) {}, flush, e, o.timing, execution, cache);
            std::ostringstream metadata;
            metadata << std::setprecision(9) << o.epsilon << ',' << bytes << ','
                     << bench::format_metric(bytes / (bench::kRoofline.memory_gbs * 1000), 3, "")
                     << ',' << o.calls << ','
                     << bench::format_metric(result.gpu.median_us / o.calls, 3, "");
            csv.write(row, result, execution, cache, metadata.str());
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out, "epsilon,logical_bytes_per_call,bandwidth_reference_us_per_"
                                     "call,calls,per_call_median_us");
        const auto e = bench::device_context().execution_view();
        for (int w : o.widths)
            for (int b : o.batches) run(o, w, b, flush, csv, e);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "rmsnorm_pack_tail: " << ex.what() << '\n';
        return 1;
    }
}
