#include "fixture.cuh"
#include "mix_fixture.cuh"
#include "conditioning.cuh"
#include "common/roofline.h"

#include <optional>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
std::optional<std::size_t> logical_bytes(hc::Op op, int columns) {
    const auto block     = std::size_t(hc::kWidth) * columns * 2;
    const auto residual  = block * hc::kStreams;
    const auto weight    = std::size_t(hc::kWidth) * hc::kStreams * 2;
    const auto injection = std::size_t(hc::kStreams) * columns * sizeof(float);
    switch (op) {
    case hc::Op::ExpandNorm:
        return block + 2 * residual + weight;
    case hc::Op::GroupedNorm:
        return 2 * residual + weight;
    case hc::Op::Combine:
        return 2 * residual + block + injection;
    case hc::Op::CombineNorm:
        return 3 * residual + block + injection + weight;
    case hc::Op::AddNorm:
        return 3 * residual + block + weight;
    case hc::Op::CombineThenNorm:
    case hc::Op::AddThenNorm:
        return 4 * residual + block + injection + weight;
    case hc::Op::BlockMix:
    case hc::Op::FinalMix:
        return std::nullopt;
    }
    throw std::logic_error("unknown HC Op");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = hc::parse(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(
            options.timing.csv_out,
            "alias,condition_us,logical_bytes,logical_gb_s,bandwidth_reference_us");
        for (int tokens : options.tokens) {
            const auto measure = [&](auto& f) {
                for (const auto& execution :
                     measurement::selections(options.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(options.timing.cache, "cold", "warm")) {
                        const auto result = measurement::measure(
                            [&](CudaExecutionView cuda) { f.launch(cuda); },
                            [&](CudaExecutionView cuda) {
                                hc::condition(cuda, options.condition_us);
                                f.restore(cuda);
                            },
                            flush, device.execution_view(), options.timing, execution, cache);
                        measurement::Row row;
                        row.op              = hc::name(options.op);
                        row.format          = "bf16";
                        row.n               = hc::kWidth;
                        row.value_heads     = hc::kStreams;
                        row.tokens          = tokens;
                        row.batch           = options.batch;
                        row.workspace_bytes = f.workspace_capacity();
                        row.workspace_peak  = f.workspace_peak();
                        auto extra = std::string(options.in_place ? "inplace," : "disjoint,") +
                                     std::to_string(options.condition_us);
                        if (const auto bytes =
                                logical_bytes(options.op, hc::columns(tokens, options.batch)))
                            extra +=
                                ',' + std::to_string(*bytes) + ',' +
                                format_metric(*bytes / (result.gpu.median_us * 1000.0), 3, "") +
                                ',' +
                                format_metric(*bytes / (kRoofline.memory_gbs * 1000.0), 3, "");
                        else
                            extra += ",,,";
                        csv.write(row, result, execution, cache, extra);
                    }
            };
            if (hc::mixes(options.op)) {
                hc::MixFixture f(options, tokens);
                measure(f);
            } else {
                hc::Fixture f(options, tokens);
                measure(f);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
