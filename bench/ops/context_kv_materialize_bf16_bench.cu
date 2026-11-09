#include "ninfer/ops/context_kv_materialize.h"
#include "../../tools/support/bf16_context_fixture.h"
#include "op_measurement.cuh"
#include <numeric>

using namespace ninfer;
namespace m   = ninfer::bench::measurement;
using Fixture = tool::Bf16ContextFixture;

namespace {
struct Options {
    m::Options timing;
    std::vector<int> widths = m::sweep("1:16"), batches = m::sweep("1:8");
    std::string counts = "full";
    bool broad         = false;
    int position       = 1'048'570;
};

Options parse(int argc, char** argv) {
    Options o;
    o.timing.cache = "cold";
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "Five-layer BF16 context KV materialization: H4096 D128/Hkv4\n"
                         "  --widths C,... --batches B,... --counts full|one|half|ragged|zero\n"
                         "  --broad-envelope 0|1 --position P\n";
            m::help();
            std::exit(0);
        }
        const auto value = m::take(argc, argv, i);
        if (flag == "--widths")
            o.widths = m::integers(value, 1, 2048);
        else if (flag == "--batches")
            o.batches = m::integers(value, 1, 8);
        else if (flag == "--counts") {
            if (value != "full" && value != "one" && value != "half" && value != "ragged" &&
                value != "zero")
                throw std::invalid_argument("counts mode");
            o.counts = value;
        } else if (flag == "--broad-envelope")
            o.broad = m::integer(value, 0, 1);
        else if (flag == "--position")
            o.position = m::integer(value, 0, INT32_MAX - 65536);
        else if (!m::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.timing.profile && (o.widths.size() * o.batches.size() != 1))
        throw std::invalid_argument("profile one geometry");
    return o;
}

void run(const Options& o, Fixture& fixture, int c, int b, m::Csv& csv,
         bench::L2FlushBuffer& flush) {
    const auto e = bench::device_context().execution_view();
    auto input   = bench::make_bf16(4096ULL * c * b, 51501U);
    DeviceBuffer positions(e.device(), c * b * 4), counts(e.device(), b * 4),
        offsets(e.device(), b * 8);
    std::vector<int> cp(b), pp(c * b, -1), slots(b);
    for (int j = 0; j < b; ++j) {
        cp[j]    = o.counts == "zero"     ? 0
                   : o.counts == "one"    ? 1
                   : o.counts == "half"   ? (c + 1) / 2
                   : o.counts == "ragged" ? (j * 7 + c) % (c + 1)
                                          : c;
        slots[j] = cp[j] ? 7 - j : -1;
        for (int i = 0; i < cp[j]; ++i) pp[j * c + i] = o.position + j * 4096 + i;
    }
    const int actual = std::accumulate(cp.begin(), cp.end(), 0);
    const auto off   = fixture.records->record_offsets(slots);
    positions.copy_from_host(pp.data(), pp.size() * 4);
    counts.copy_from_host(cp.data(), cp.size() * 4);
    offsets.copy_from_host(off.data(), off.size() * 8);
    Tensor x(input.span(), DType::BF16, {4096, c, b}), p(positions.span(), DType::I32, {c, b}),
        n(counts.span(), DType::I32, {b}), offset(offsets.span(), DType::I64, {b});
    for (int l = 0; l < 5; ++l) fixture.bind(l, offset);
    const ops::ContextKVMaterializeExecutionEnvelope envelope{
        o.broad ? 0U : std::uint32_t(*std::min_element(cp.begin(), cp.end())),
        o.broad ? std::uint32_t(c) : std::uint32_t(*std::max_element(cp.begin(), cp.end()))};
    const auto capacity =
        ops::context_kv_materialize_workspace_capacity_bytes(Fixture::Profile, b, c, c);
    auto backing = bench::make_buffer(capacity);
    MemoryArena arena(backing.span());
    const auto launch = [&](CudaExecutionView cuda) {
        ops::context_kv_materialize(x, p, n, fixture.layers, Fixture::Profile, envelope, arena,
                                    cuda);
    };
    // The cache is write-only for this Op: repeated identical calls are idempotent and
    // do not evolve a consumed input. No cache restore belongs in its timing interval.
    const auto weights  = actual ? 5ULL * 1024 * 4096 * 2 : 0;
    const int live_rows = std::count_if(cp.begin(), cp.end(), [](int n) { return n > 0; });
    const auto bytes    = envelope.max_count == 0 ? 0ULL
                                                  : weights + 8192ULL * actual + 5ULL * 2048 * actual +
                                                     (actual ? 5ULL * 256 : 0) + 4ULL * actual +
                                                     4ULL * b + 8ULL * live_rows;
    m::Row row{.op              = "context_kv_materialize",
               .format          = "bf16",
               .policy          = "a16",
               .commits         = o.counts,
               .n               = 1024,
               .k               = 4096,
               .qk_heads        = 4,
               .layers          = 5,
               .tokens          = c,
               .batch           = b,
               .workspace_bytes = capacity};
    for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
            arena.reset_peak();
            const auto result = m::measure(
                launch, [](CudaExecutionView) {}, flush, e, o.timing, execution, cache,
                envelope.max_count == 0);
            row.workspace_peak = arena.peak_used();
            csv.write(
                row, result, execution, cache,
                std::to_string(actual) + ',' + std::to_string(envelope.max_count) + ',' +
                    std::to_string(bytes) + ',' + std::to_string(2ULL * 5 * 1024 * 4096 * actual) +
                    ',' + std::to_string(weights) + ',' +
                    bench::format_metric(weights / (result.gpu.median_us * 1000), 3, "") + ',' +
                    bench::format_metric(bytes / (bench::kRoofline.memory_gbs * 1000), 3, "") +
                    ',' +
                    bench::format_metric(2. * 5 * 1024 * 4096 * actual /
                                             (bench::kRoofline.bf16_f32_tflops * 1.e6),
                                         3, ""));
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        Fixture f(bench::device_context().device());
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out,
                   "actual_columns,envelope_max_count,logical_bytes,useful_flops,necessary_weight_"
                   "bytes,weight_gb_s,memory_reference_us,compute_reference_us");
        for (int c : o.widths)
            for (int b : o.batches)
                if (c <= 16 || b == 1) run(o, f, c, b, csv, flush);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
