#include "ninfer/ops/block_attn_res.h"
#include "op_measurement.cuh"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> columns{1};
    int d = 2304, stacked = 3, q = 1;
    bool addend = true, norm = true, normalize_values = true;
};

void run(const Options& o, int m, measurement::LogicalBenchmark& benchmark) {
    const auto d = o.d, q = o.q;
    const auto count = std::size_t(d) * m;
    auto p           = make_bf16(count, 451);
    auto a           = o.addend ? make_bf16(count, 452) : DeviceBuffer{};
    auto s           = o.stacked ? make_bf16(count * o.stacked, 453) : DeviceBuffer{};
    auto v           = make_f32(std::size_t(d) * q, 454, -1.0f, 1.0f);
    auto w           = o.norm ? make_bf16(std::size_t(d) * q, 455, 0.5f, 1.5f) : DeviceBuffer{};
    auto y           = make_buffer(count * q * (o.norm ? 2 : 4));
    const Tensor partial(p.span(), DType::BF16, {d, m});
    const auto stack  = o.stacked ? Tensor(s.span(), DType::BF16, {d, m, o.stacked}) : Tensor{};
    const auto addend = o.addend ? Tensor(a.span(), DType::BF16, {d, m}) : Tensor{};
    const Tensor queries(v.span(), DType::FP32, {d, q});
    const auto weight = o.norm ? Tensor(w.span(), DType::BF16, {d, q}) : Tensor{};
    Tensor out(y.span(), o.norm ? DType::BF16 : DType::FP32, {d, m, q});
    const ops::BlockAttnResParameters params{o.normalize_values ? 1.0f / std::sqrt(float(d)) : 1.0f,
                                             1.0e-6f, 1.0e-6f, o.normalize_values};
    measurement::Row row;
    row.op            = "block_attn_res";
    row.format        = o.norm ? "bf16" : "fp32";
    row.policy        = o.normalize_values ? "normalized" : "raw";
    row.n             = o.stacked;
    row.k             = q;
    row.conv_channels = d;
    row.tokens        = m;
    row.valid         = o.addend ? "addend" : "no-addend";
    const auto bytes  = p.size() + a.size() + s.size() + v.size() + w.size() + y.size();
    benchmark.run(
        row, bytes,
        [&](CudaExecutionView cuda) {
            ops::block_attn_res(stack, partial, addend, queries, weight, params, out, cuda);
        },
        [](CudaExecutionView) {});
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "block_attn_res [--d D] [--tokens M[,M...]] [--stack N] [--queries Q]\n"
                       "  --addend 0|1 --norm 0|1 --normalize-values 0|1\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (flag == "--d")
                o.d = measurement::integer(value);
            else if (flag == "--tokens")
                o.columns = measurement::integers(value);
            else if (flag == "--stack")
                o.stacked = measurement::integer(value, 0);
            else if (flag == "--queries")
                o.q = measurement::integer(value);
            else if (flag == "--norm")
                o.norm = measurement::integer(value, 0, 1);
            else if (flag == "--addend")
                o.addend = measurement::integer(value, 0, 1);
            else if (flag == "--normalize-values")
                o.normalize_values = measurement::integer(value, 0, 1);
            else if (!measurement::parse(o.timing, flag, value))
                throw std::invalid_argument("unknown option " + std::string(flag));
        }
        measurement::LogicalBenchmark benchmark(o.timing);
        for (int m : o.columns) run(o, m, benchmark);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
