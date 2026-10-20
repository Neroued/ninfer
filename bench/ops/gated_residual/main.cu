#include "fixture.cuh"
#include "mix_fixture.cuh"
#include "conditioning.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
std::size_t mix_weight_bytes(hc::Op op) {
    const int down_rows = hc::kRank + (op == hc::Op::BlockMix ? hc::kStreams : 0);
    return std::size_t(hc::kWidth) * hc::kStreams * (down_rows + hc::kRank) * 2;
}

std::size_t logical_bytes(hc::Op op, int columns) {
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
        return 4 * residual + block + injection + weight;
    case hc::Op::AddThenNorm:
        return 4 * residual + block + weight;
    case hc::Op::BlockMix:
    case hc::Op::FinalMix:
        // Public operand traffic is a lower bound: private staging, repeated reads and
        // cache/writeback traffic depend on the selected implementation.
        return mix_weight_bytes(op) + residual + block + (op == hc::Op::BlockMix ? injection : 0);
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
        measurement::Csv csv(options.timing.csv_out,
                             "alias,condition_us,logical_bytes,logical_gb_s,bandwidth_reference_us,"
                             "weight_bytes,weight_gb_s,contraction_flops,contraction_tflops");
        const auto measure = [&](int tokens, auto& f) {
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
                    row.k               = hc::mixes(options.op) ? hc::kRank : 0;
                    row.value_heads     = hc::kStreams;
                    row.tokens          = tokens;
                    row.batch           = options.batch;
                    row.workspace_bytes = f.workspace_capacity();
                    row.workspace_peak  = f.workspace_peak();
                    auto extra          = std::string(options.in_place ? "inplace," : "disjoint,") +
                                 std::to_string(options.condition_us);
                    const int m      = hc::columns(tokens, options.batch);
                    const auto bytes = logical_bytes(options.op, m);
                    extra += ',' + std::to_string(bytes) + ',' +
                             format_metric(bytes / (result.gpu.median_us * 1000.0), 3, "") + ',' +
                             format_metric(bytes / (kRoofline.memory_gbs * 1000.0), 3, "");
                    if (hc::mixes(options.op)) {
                        const auto weights = mix_weight_bytes(options.op);
                        // Two FLOPs per BF16 matrix element and column, over both projections.
                        const double flops = double(weights) * m;
                        extra += ',' + std::to_string(weights) + ',' +
                                 format_metric(weights / (result.gpu.median_us * 1000.0), 3, "") +
                                 ',' + format_metric(flops, 0, "") + ',' +
                                 format_metric(flops / (result.gpu.median_us * 1.e6), 3, "");
                    } else
                        extra += ",,,,";
                    csv.write(row, result, execution, cache, extra);
                }
        };
        if (hc::mixes(options.op)) {
            const hc::MixWeights weights(options.op);
            for (int tokens : options.tokens) {
                hc::MixFixture f(options, tokens, weights);
                measure(tokens, f);
            }
        } else {
            for (int tokens : options.tokens) {
                hc::Fixture f(options, tokens);
                measure(tokens, f);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
