#include "ninfer/ops/linear_topk.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "quantized_weight.cuh"
#include "common/roofline.h"
#include "ops/linear/topk/keys/dispatch.h"
using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::string format = "bf16", policy;
    std::vector<int> columns{1, 7, 14, 28, 56, 120};
    int begin = 77440, domain = 154856, drafts = 0, batch = 1;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Vocabulary shard top-16: --format bf16|mxfp8|q4 --policy a16|a8|a4\n"
                         "--columns M,... | --sweep START:END[:STEP] | --drafts D --batch B\n"
                         "--token-begin ID --token-domain N (defaults 77440/154856)\n";
            measurement::help();
            std::exit(0);
        }
        const auto v = measurement::take(argc, argv, i);
        if (flag == "--format")
            o.format = v;
        else if (flag == "--policy")
            o.policy = v;
        else if (flag == "--columns")
            o.columns = measurement::integers(v);
        else if (flag == "--sweep")
            o.columns = measurement::sweep(v);
        else if (flag == "--drafts")
            o.drafts = measurement::integer(v);
        else if (flag == "--batch")
            o.batch = measurement::integer(v);
        else if (flag == "--token-begin")
            o.begin = measurement::integer(v, 0);
        else if (flag == "--token-domain")
            o.domain = measurement::integer(v);
        else if (!measurement::parse(o.timing, flag, v))
            throw std::invalid_argument("unknown key topk option");
    }
    if (o.policy.empty()) o.policy = o.format == "mxfp8" ? "a8" : "a16";
    if (o.policy != "a16" && o.policy != "a8" && o.policy != "a4")
        throw std::invalid_argument("invalid policy");
    if (o.batch < 1 || o.batch > 8) throw std::invalid_argument("invalid batch");
    if (o.drafts)
        o.columns = {o.drafts * o.batch};
    else if (o.batch != 1)
        throw std::invalid_argument("--batch requires --drafts");
    return o;
}

template <class Fixture>
void run(const Options& o, Fixture& fixture) {
    constexpr int N = 77440, K = 4096;
    const auto policy = o.policy == "a16"  ? ops::LinearPolicy::A16Only
                        : o.policy == "a8" ? ops::LinearPolicy::AllowA8
                                           : ops::LinearPolicy::AllowA4;
    const int max     = *std::max_element(o.columns.begin(), o.columns.end());
    const auto capacity =
        ops::linear_topk_workspace_capacity_bytes(fixture.weight.qtype, N, K, policy, 1, max);
    auto x = make_bf16(std::size_t(K) * max, 101), keys = make_buffer(std::size_t(16) * max * 8),
         ws = make_buffer(capacity);
    MemoryArena arena(ws.span());
    L2FlushBuffer flush(o.timing.flush_bytes);
    measurement::hardware();
    measurement::Csv csv(o.timing.csv_out,
                         "columns,token_begin,token_domain,valid_rows,necessary_bytes,useful_flops,"
                         "effective_gbs,roofline_us,workspace_traffic_bytes,activation_encoding_"
                         "bytes,producer_rows,tensor_profile");
    for (int m : o.columns) {
        Tensor input(x.span(), DType::BF16, {K, m}),
            output(keys.span(), DType::I64, {16, o.drafts ? o.drafts : m, o.batch});
        const int valid      = std::clamp(o.domain - o.begin, 0, N);
        const auto row_bytes = fixture.model_weight_bytes() / N;
        const auto bytes =
            (valid ? std::uint64_t(valid) * row_bytes + 2ULL * K * m : 0) + 128ULL * m;
        const double flops       = 2. * valid * K * m;
        std::uint64_t activation = 0, traffic = 0;
        double compute_floor = 0;
        bool a8 = false, a16 = false;
        int producer_rows = 0;
        for (int first = 0; valid && first < m;) {
            const int columns = std::min(m - first, 128);
            const auto plan =
                ops::detail::topk_keys::select(fixture.weight.qtype, N, K, policy, columns);
            WorkspaceLayoutBuilder layout;
            const auto work    = ops::detail::topk_keys::allocate(layout, plan, N, K, columns);
            const auto encoded = plan.a8 ? std::uint64_t(columns) * (K + K / 32) : 0;
            activation += encoded;
            traffic +=
                2 * (work.candidates.partial_keys.bytes() + work.candidates.group_keys.bytes() +
                     work.candidates.secondary_keys.bytes() + encoded);
            compute_floor +=
                2. * valid * K * columns /
                (1.e6 * (plan.a8 ? kRoofline.mxfp8_f32_tflops : kRoofline.bf16_f32_tflops));
            a8 |= plan.a8;
            a16 |= !plan.a8;
            if (!first)
                producer_rows = plan.rows;
            else if (producer_rows != plan.rows)
                producer_rows = 0;
            first += columns;
        }
        const char* profile = a8 ? (a16 ? "MIXED_A16_A8_F32ACC" : "MXFP8_F32ACC") : "BF16_F32ACC";
        measurement::Row row;
        row.op              = "linear_topk_keys";
        row.format          = o.format;
        row.policy          = o.policy;
        row.n               = N;
        row.k               = K;
        row.tokens          = o.drafts ? o.drafts : m;
        row.batch           = o.batch;
        row.workspace_bytes = capacity;
        for (const auto& exec : measurement::selections(o.timing.execution, "graph", "eager"))
            for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto timing = measurement::measure(
                    [&](auto cuda) {
                        ops::linear_topk(input, fixture.weight, o.begin, o.domain, policy, output,
                                         arena, cuda);
                    },
                    [](auto) {}, flush, device_context().execution_view(), o.timing, exec, cache,
                    valid == 0);
                row.workspace_peak = arena.peak_used();
                const double floor =
                    std::max(bytes / (1000 * kRoofline.sustained_read_gbs), compute_floor);
                csv.write(row, timing, exec, cache,
                          std::to_string(m) + ',' + std::to_string(o.begin) + ',' +
                              std::to_string(o.domain) + ',' + std::to_string(valid) + ',' +
                              std::to_string(bytes) + ',' + std::to_string(flops) + ',' +
                              format_metric(bytes / timing.gpu.median_us / 1000, 3, "") + ',' +
                              format_metric(floor, 3, "") + ',' + std::to_string(traffic) + ',' +
                              std::to_string(activation) + ',' + std::to_string(producer_rows) +
                              ',' + profile);
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto o = parse(argc, argv);
        if (o.format == "bf16") {
            auto w = make_direct_bf16_weight(77440, 4096, 107);
            run(o, w);
        } else if (o.format == "mxfp8") {
            auto w = make_mxfp8_weight(77440, 4096, 107);
            run(o, w);
        } else if (o.format == "q4") {
            auto w = make_row_split_weight(QType::Q4_G64_FP16, 77440, 4096, 4096, 107);
            run(o, w);
        } else
            throw std::invalid_argument("unsupported format");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
