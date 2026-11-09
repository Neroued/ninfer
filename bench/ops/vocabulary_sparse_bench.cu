#include "../../tools/support/sparse_vocabulary_fixture.h"
#include "op_measurement.cuh"
#include <numeric>

using namespace ninfer;
namespace m = bench::measurement;
using F     = tool::SparseVocabularyFixture;

struct Options {
    m::Options timing;
    std::vector<int> steps = m::sweep("1:15"), batches = m::sweep("1:8"), modes{0, 1, 2},
                     accepted{0, 1, 2};
    std::string extent = "full";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view f = argv[i];
        if (f == "--help") {
            std::cout << "Sparse-q gathered target acceptance\n --steps D,... --batches B,... "
                         "--mode greedy|stochastic|mixed|all\n --accepted first|half|all|sweep "
                         "--extent full|zero|mixed\n";
            m::help();
            std::exit(0);
        }
        auto v = m::take(argc, argv, i);
        if (f == "--steps")
            o.steps = m::integers(v, 1, 15);
        else if (f == "--batches")
            o.batches = m::integers(v, 1, 8);
        else if (f == "--mode") {
            if (v == "greedy")
                o.modes = {0};
            else if (v == "stochastic")
                o.modes = {1};
            else if (v == "mixed")
                o.modes = {2};
            else if (v != "all")
                throw std::invalid_argument("mode");
        } else if (f == "--accepted") {
            if (v == "first")
                o.accepted = {0};
            else if (v == "half")
                o.accepted = {1};
            else if (v == "all")
                o.accepted = {2};
            else if (v != "sweep")
                throw std::invalid_argument("accepted");
        } else if (f == "--extent") {
            if (v != "full" && v != "zero" && v != "mixed") throw std::invalid_argument("extent");
            o.extent = v;
        } else if (!m::parse(o.timing, f, v))
            throw std::invalid_argument("option");
    }
    if (o.timing.profile && (o.steps.size() != 1 || o.batches.size() != 1 || o.modes.size() != 1 ||
                             o.accepted.size() != 1))
        throw std::invalid_argument("profile one point");
    return o;
}

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        auto e       = bench::device_context().execution_view();
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out, "extent,active_columns,accepted_columns,min_count,max_count,"
                                     "input_payload_bytes,output_bytes");
        for (int d : o.steps)
            for (int b : o.batches)
                for (int mode : o.modes)
                    for (int a : o.accepted) {
                        if (mode == 2 && b == 1) continue;
                        const int accepted = a == 0 ? 0 : a == 1 ? d / 2 : d;
                        F f(e.device(), d, b, 2, mode, accepted);
                        if (o.extent == "mixed") f.refresh(0, true);
                        if (o.extent == "zero") {
                            std::fill(f.extents.begin(), f.extents.end(), 0);
                            f.extent_buffer.copy_from_host(f.extents.data(), b * 4);
                        }
                        auto ob = bench::make_buffer((d + 1) * b * 4),
                             cb = bench::make_buffer(b * 4);
                        Tensor out(ob.span(), DType::I32, {d + 1, b}),
                            counts(cb.span(), DType::I32, {b});
                        auto launch = [&](CudaExecutionView cuda) {
                            ops::accept_vocabulary_candidates_sparse(
                                f.keys(),
                                static_cast<const ops::SamplingConfig*>(f.config_buffer.data()),
                                f.draft_view(), f.id_view(), f.q_view(), f.extent_view(),
                                f.position_view(), out, counts, cuda);
                        };
                        m::Row row;
                        row.op      = "accept_vocabulary_candidates_sparse";
                        row.format  = "keys_fp32_q";
                        row.tokens  = d;
                        row.batch   = b;
                        row.n       = 20;
                        row.k       = 16;
                        row.commits = std::string(mode == 0   ? "greedy"
                                                  : mode == 1 ? "stochastic"
                                                              : "mixed") +
                                      "_" +
                                      (a == 0   ? "first"
                                       : a == 1 ? "half"
                                                : "all");
                        const int active = std::accumulate(f.extents.begin(), f.extents.end(), b);
                        for (const auto& execution :
                             m::selections(o.timing.execution, "graph", "eager"))
                            for (const auto& cache :
                                 m::selections(o.timing.cache, "cold", "warm")) {
                                auto timing = m::measure(
                                    launch, [](CudaExecutionView) {}, flush, e, o.timing, execution,
                                    cache);
                                std::vector<int> actual(b);
                                cb.copy_to_host(actual.data(), b * 4);
                                const auto [lo, hi] =
                                    std::minmax_element(actual.begin(), actual.end());
                                const std::size_t payload = 320ULL * active +
                                                            132ULL * (active - b) +
                                                            b * (8 + sizeof(ops::SamplingConfig));
                                csv.write(row, timing, execution, cache,
                                          o.extent + ',' + std::to_string(active) + ',' +
                                              std::to_string(std::accumulate(actual.begin(),
                                                                             actual.end(), -b)) +
                                              ',' + std::to_string(*lo) + ',' +
                                              std::to_string(*hi) + ',' + std::to_string(payload) +
                                              ',' + std::to_string(4ULL * (d + 2) * b));
                            }
                    }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
