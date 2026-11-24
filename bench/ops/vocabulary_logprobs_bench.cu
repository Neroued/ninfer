#include "ninfer/ops/target_logprobs.h"
#include "op_measurement.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::string op = "stats";
    std::vector<int> columns{1};
    int rows = 124160, ranks = 2, rank = 0, domain = 248077;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Vocabulary log-probability benchmark\n  --op stats|merge --columns "
                         "C[,C...] --rows V --ranks R --rank R --domain D\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--columns")
            o.columns = measurement::integers(value);
        else if (flag == "--rows")
            o.rows = measurement::integer(value);
        else if (flag == "--ranks")
            o.ranks = measurement::integer(value, 1, 16);
        else if (flag == "--rank")
            o.rank = measurement::integer(value, 0, 15);
        else if (flag == "--domain")
            o.domain = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if ((o.op != "stats" && o.op != "merge") || o.rank >= o.ranks ||
        std::int64_t(o.rows) * o.ranks < o.domain)
        throw std::invalid_argument("invalid Op/partition");
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        measurement::hardware();
        const auto cuda = device_context().execution_view();
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out,
                             "rank,ranks,logical_bytes,logical_gb_s,bandwidth_reference_us");
        for (int c : o.columns) {
            auto logits = make_bf16(std::size_t(o.rows) * c, 611, -16, 16);
            std::vector<int> target_ids(c);
            for (int i = 0; i < c; ++i) target_ids[i] = std::int64_t(i) * 7919 % o.domain;
            auto targets = measurement::make_i32(target_ids),
                 stats = make_buffer(std::size_t(12) * c * o.ranks), output = make_buffer(c * 4);
            const auto capacity = ops::vocabulary_logprob_stats_workspace_capacity_bytes(o.rows, c);
            auto storage        = make_buffer(std::max<std::size_t>(256, capacity));
            MemoryArena workspace(storage.span());
            Tensor x(logits.span(), DType::BF16, {o.rows, c}), ids(targets.span(), DType::I32, {c});
            Tensor partials(stats.span(), DType::FP32, {3, c, o.ranks}),
                out(output.span(), DType::FP32, {c});
            const auto extract = [&](int rank, CudaExecutionView execution) {
                Tensor part(stats.span().subspan(std::size_t(rank) * c * 12, std::size_t(c) * 12),
                            DType::FP32, {3, c});
                ops::vocabulary_logprob_stats(x, ids, rank * o.rows, o.domain, part, workspace,
                                              execution);
            };
            if (o.op == "merge")
                for (int rank = 0; rank < o.ranks; ++rank) extract(rank, cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            const auto bytes =
                o.op == "stats"
                    ? std::size_t(c) *
                          (std::max(0, std::min(o.rows, o.domain - o.rank * o.rows)) * 2ULL + 16)
                    : stats.size() + output.size();
            measurement::Row row;
            row.op     = o.op;
            row.format = "bf16";
            row.n      = o.rows;
            row.tokens = c;
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    const auto timing = measurement::measure(
                        [&](auto view) {
                            if (o.op == "stats")
                                extract(o.rank, view);
                            else
                                ops::target_logprobs_from_stats(partials, out, view);
                        },
                        [](auto) {}, flush, cuda, o.timing, execution, cache);
                    row.workspace_bytes = o.op == "stats" ? capacity : 0;
                    row.workspace_peak  = o.op == "stats" ? workspace.peak_used() : 0;
                    csv.write(row, timing, execution, cache,
                              std::to_string(o.rank) + ',' + std::to_string(o.ranks) + ',' +
                                  std::to_string(bytes) + ',' +
                                  format_metric(bytes / (timing.gpu.median_us * 1000), 3, "") +
                                  ',' +
                                  format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, ""));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
