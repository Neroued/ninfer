#include "ninfer/ops/candidate_selector.h"
#include "../../tools/support/gathered_selector_fixture.h"
#include "op_measurement.cuh"

#include <set>

using namespace ninfer;
namespace m = bench::measurement;
using F     = tool::GatheredSelectorFixture;

namespace {
struct Options {
    m::Options timing;
    std::vector<int> steps = m::sweep("1:15"), batches = m::sweep("1:8"), modes{0, 1, 2};
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "Gathered top16 + BF16 selector path (rank256, V154880/domain154856)\n"
                         "  --steps D,... --batches B,... --mode greedy|stochastic|mixed|all\n";
            m::help();
            std::exit(0);
        }
        const auto v = m::take(argc, argv, i);
        if (flag == "--steps")
            o.steps = m::integers(v, 1, 15);
        else if (flag == "--batches")
            o.batches = m::integers(v, 1, 8);
        else if (flag == "--mode") {
            if (v == "greedy")
                o.modes = {0};
            else if (v == "stochastic")
                o.modes = {1};
            else if (v == "mixed")
                o.modes = {2};
            else if (v == "all")
                o.modes = {0, 1, 2};
            else
                throw std::invalid_argument("mode");
        } else if (!m::parse(o.timing, flag, v))
            throw std::invalid_argument("unknown option");
    }
    if (o.timing.profile && (o.steps.size() != 1 || o.batches.size() != 1 || o.modes.size() != 1))
        throw std::invalid_argument("profile one geometry/mode");
    return o;
}

const char* mode_name(int mode) {
    return mode == 0 ? "greedy" : mode == 1 ? "stochastic" : "mixed";
}

void run(const Options& o, F& f, int mode, m::Csv& csv, bench::L2FlushBuffer& flush) {
    const auto e = bench::device_context().execution_view();
    f.set_mode(mode);
    auto ib = bench::make_buffer(16ULL * f.steps * f.batch * 4),
         db = bench::make_buffer(f.steps * f.batch * 4),
         qb = bench::make_buffer(16ULL * f.steps * f.batch * 4);
    Tensor ids(ib.span(), DType::I32, {16, f.steps, f.batch}),
        drafts(db.span(), DType::I32, {f.steps, f.batch}),
        q(qb.span(), DType::FP32, {16, f.steps, f.batch});
    const auto capacity = ops::candidate_selector_path_workspace_capacity_bytes(
        ops::CandidateSelectorInput::GatheredKeys, f.steps, f.steps, f.batch, f.batch);
    auto wb = bench::make_buffer(std::max<std::size_t>(capacity, 1));
    MemoryArena work(wb.span());
    auto launch = [&](CudaExecutionView cuda) {
        ops::candidate_selector_path(
            f.key_tensor(), f.hidden_tensor(), f.anchor_tensor(), f.predecessor_tensor(),
            f.successor_tensor(), F::Domain, f.position_tensor(),
            static_cast<const ops::SamplingConfig*>(f.configs.data()), ids, drafts, q, work, cuda);
    };
    m::Row row;
    row.op              = "candidate_selector_path";
    row.format          = "gathered_bf16";
    row.policy          = "a16";
    row.tokens          = f.steps;
    row.batch           = f.batch;
    row.n               = 16;
    row.k               = 256;
    row.commits         = mode_name(mode);
    row.workspace_bytes = capacity;
    for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
            work.reset_peak();
            const auto result =
                m::measure(launch, [](CudaExecutionView) {}, flush, e, o.timing, execution, cache);
            row.workspace_peak = work.peak_used();
            std::vector<int> chosen(f.steps * f.batch);
            db.copy_to_host(chosen.data(), chosen.size() * 4);
            std::set<int> pred(f.anchors.begin(), f.anchors.end()), succ;
            for (int b = 0; b < f.batch; ++b)
                for (int i = 0; i < f.steps - 1; ++i) pred.insert(chosen[b * f.steps + i]);
            for (int id : f.ids)
                if (id != INT_MAX) succ.insert(id);
            const auto book_bytes   = (pred.size() + succ.size()) * 512ULL;
            const auto key_bytes    = 256ULL * f.steps * f.batch,
                       output_bytes = 132ULL * f.steps * f.batch;
            csv.write(row, result, execution, cache,
                      std::to_string(pred.size()) + ',' + std::to_string(succ.size()) + ',' +
                          std::to_string(book_bytes) + ',' + std::to_string(key_bytes) + ',' +
                          std::to_string(output_bytes));
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out, "actual_predecessor_rows,selected_successor_rows,mandatory_"
                                     "codebook_bytes,input_key_bytes,output_bytes");
        for (int d : o.steps)
            for (int b : o.batches) {
                F f(bench::device_context().device(), d, b);
                for (int mode : o.modes)
                    if (mode != 2 || b > 1) run(o, f, mode, csv, flush);
            }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
