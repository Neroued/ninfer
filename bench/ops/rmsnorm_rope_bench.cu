// Complete public-Op timing; in-place operands are restored before each sample.
#include "ninfer/ops/rmsnorm_rope.h"
#include "op_measurement.cuh"
#include <iomanip>
#include <sstream>
#include <climits>

using namespace ninfer;
namespace m = ninfer::bench::measurement;

namespace {
struct Options {
    m::Options timing;
    std::string form        = "pair";
    std::vector<int> widths = m::sweep("2:16"), batches = m::sweep("1:8");
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int key_heads = 8, position = 1'048'560, calls = 1;
    float epsilon = 1.e-6F, theta = 1.e7F;
};

float positive_float(std::string_view text) {
    std::size_t end;
    const float value = std::stof(std::string(text), &end);
    if (end != text.size() || !(value > 0) || !std::isfinite(value))
        throw std::invalid_argument("expected positive finite float");
    return value;
}

Options parse(int argc, char** argv) {
    Options o;
    o.timing.cache = "warm";
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "usage: ninfer_rmsnorm_rope_bench [--form pair|single] "
                         "[--key-heads 4|8] [--epsilon E] [--theta T] [--position P]\n"
                         "  --widths W,... (2..16) --batches B,... (1..8) "
                         "--tokens T,... (1..2048)\n"
                         "  --calls N (1..64): independent restored operands, warm cache only\n"
                         "Pair uses Hq=4*Hkv. Defaults: Hkv=8, epsilon=1e-6, theta=1e7.\n";
            m::help();
            std::exit(0);
        }
        const auto value = m::take(argc, argv, i);
        if (flag == "--form") {
            if (value != "pair" && value != "single")
                throw std::invalid_argument("form: pair|single");
            o.form = value;
        } else if (flag == "--key-heads") {
            o.key_heads = m::integer(value, 4, 8);
            if (o.key_heads != 4 && o.key_heads != 8) throw std::invalid_argument("key heads: 4|8");
        } else if (flag == "--calls")
            o.calls = m::integer(value, 1, 64);
        else if (flag == "--epsilon")
            o.epsilon = positive_float(value);
        else if (flag == "--theta")
            o.theta = positive_float(value);
        else if (flag == "--position")
            o.position = m::integer(value, 0, INT_MAX - 2048);
        else if (flag == "--widths")
            o.widths = m::integers(value, 2, 16);
        else if (flag == "--batches")
            o.batches = m::integers(value, 1, 8);
        else if (flag == "--tokens")
            o.tokens = m::integers(value, 1, 2048);
        else if (!m::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown flag: " + std::string(flag));
    }
    if (o.timing.profile &&
        (o.form == "pair" ? o.widths.size() * o.batches.size() != 1 : o.tokens.size() != 1))
        throw std::invalid_argument("profile requires one extent");
    if (o.calls > 1 && o.timing.cache != "warm")
        throw std::invalid_argument("multiple calls require --cache warm");
    return o;
}

void run(const Options& o, int width, int batch, bench::L2FlushBuffer& flush, m::Csv& csv,
         CudaExecutionView e) {
    const bool pair  = o.form == "pair";
    const int tokens = width * batch, qh = pair ? 4 * o.key_heads : 0;
    const std::size_t q_count = std::size_t(128) * qh * tokens;
    const std::size_t k_count = std::size_t(128) * o.key_heads * tokens;
    auto q                    = bench::make_bf16(std::max(q_count * o.calls, std::size_t(1)), 101U);
    auto k                    = bench::make_bf16(k_count * o.calls, 103U);
    auto qw                   = bench::make_bf16(128, 105U, .8F, 1.2F);
    auto kw                   = bench::make_bf16(128, 107U, .8F, 1.2F);
    DeviceBuffer p(e.device(), std::size_t(tokens) * sizeof(int));
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) positions[t] = o.position + t;
    p.copy_from_host(positions.data(), p.size());
    bench::SavedBuffer initial_q(q), initial_k(k);
    std::vector<Tensor> queries, keys;
    for (int i = 0; i < o.calls; ++i) {
        if (pair)
            queries.emplace_back(q.span().subspan(i * q_count * 2, q_count * 2), DType::BF16,
                                 std::initializer_list<int>{128, qh, width, batch});
        keys.emplace_back(k.span().subspan(i * k_count * 2, k_count * 2), DType::BF16,
                          std::initializer_list<int>{128, o.key_heads, width, batch});
    }
    Tensor twq(qw.span(), DType::BF16, {128}), twk(kw.span(), DType::BF16, {128});
    Tensor tp(p.span(), DType::I32,
              pair ? std::initializer_list<int>{width, batch} : std::initializer_list<int>{tokens});
    const auto restore = [&](CudaExecutionView cuda) {
        if (pair) initial_q.restore(cuda);
        initial_k.restore(cuda);
    };
    const auto launch = [&](CudaExecutionView cuda) {
        for (int i = 0; i < o.calls; ++i) {
            if (pair)
                ops::rmsnorm_rope(tp, twq, twk, o.epsilon, o.theta, queries[i], keys[i], cuda);
            else
                ops::rmsnorm_rope(tp, twk, o.epsilon, o.theta, keys[i], cuda);
        }
    };
    // Represented activations read+write, norm weights once, and Device positions.
    const std::size_t bytes = 4 * (q_count + k_count) + (pair ? 512 : 256) + 4 * tokens;
    m::Row row{.op          = "rmsnorm_rope_" + o.form,
               .format      = "bf16",
               .n           = 128,
               .qk_heads    = qh,
               .value_heads = o.key_heads,
               .tokens      = width,
               .batch       = batch};
    for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
            const auto result = m::measure(launch, restore, flush, e, o.timing, execution, cache);
            std::ostringstream metadata;
            metadata << std::setprecision(9) << o.epsilon << ',' << o.theta << ',' << o.position
                     << ',' << bytes << ','
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
        m::Csv csv(o.timing.csv_out, "epsilon,theta,base_position,logical_bytes_per_call,bandwidth_"
                                     "reference_us_per_call,calls,per_call_median_us");
        const auto e = bench::device_context().execution_view();
        if (o.form == "pair")
            for (int w : o.widths)
                for (int b : o.batches) run(o, w, b, flush, csv, e);
        else
            for (int t : o.tokens) run(o, t, 1, flush, csv, e);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "rmsnorm_rope: " << ex.what() << '\n';
        return 1;
    }
}
