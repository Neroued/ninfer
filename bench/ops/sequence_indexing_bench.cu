#include "ninfer/ops/sequence.h"
#include "ninfer/ops/indexing.h"
#include "op_measurement.cuh"
#include <numeric>

using namespace ninfer;
namespace m = bench::measurement;

namespace {
struct Options {
    m::Options timing;
    std::vector<std::string> ops{"masked",  "verify", "verify-ids",   "ragged",
                                 "scatter", "gather", "state-gather", "state-scatter"};
    std::vector<int> widths = m::sweep("2:16"), batches = m::sweep("1:8");
    int features = 4096, column_features = 0;
    std::string counts = "all";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Sequence/indexing public Op benchmark\n --op "
                         "masked|verify|verify-ids|ragged|scatter|gather|state-gather|state-"
                         "scatter|all\n --widths W,... --batches B,... --features D\n"
                         " --column-features P (default D) --counts "
                         "all|half|mixed|zero\n";
            m::help();
            std::exit(0);
        }
        const auto value = m::take(argc, argv, i);
        if (flag == "--op") {
            if (value != "all") {
                if (std::find(o.ops.begin(), o.ops.end(), value) == o.ops.end())
                    throw std::invalid_argument("unknown Op");
                o.ops = {std::string(value)};
            }
        } else if (flag == "--column-features")
            o.column_features = m::integer(value, 8);
        else if (flag == "--features")
            o.features = m::integer(value, 8);
        else if (flag == "--widths")
            o.widths = m::integers(value, 2, 4096);
        else if (flag == "--batches")
            o.batches = m::integers(value, 1, 8);
        else if (flag == "--counts")
            o.counts = value;
        else if (!m::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if ((o.column_features && (o.column_features < o.features || o.column_features % 8)) ||
        o.features % 8 ||
        (o.counts != "all" && o.counts != "half" && o.counts != "mixed" && o.counts != "zero"))
        throw std::invalid_argument("invalid features/counts");
    if (o.timing.profile && (o.ops.size() != 1 || o.widths.size() != 1 || o.batches.size() != 1))
        throw std::invalid_argument("profile one point");
    return o;
}

void run(const Options& o, const std::string& op, int w, int b, m::Csv& csv,
         bench::L2FlushBuffer& flush) {
    if ((op == "masked" || op == "verify" || op == "verify-ids") && w > 16)
        throw std::invalid_argument("query width must be <=16");
    if (op == "masked" && o.counts == "zero") return;
    const int d     = o.features;
    const int pitch = o.column_features ? o.column_features : d;
    const auto e    = bench::device_context().execution_view();
    std::vector<int> anchors(b), positions(b), valid(b), extents(b), lanes(b), starts(b), ends(b),
        columns(b);
    std::vector<std::int64_t> offsets(b);
    const int record_stride = d * 2 + 64;
    for (int r = 0; r < b; ++r) {
        anchors[r]   = 17 + r * 7919;
        positions[r] = 1'048'560 + r * 97;
        lanes[r]     = (r * 3 + 1) % 8;
        valid[r]     = o.counts == "all"    ? w
                       : o.counts == "half" ? w / 2
                       : o.counts == "zero" ? 0
                                            : r % (w + 1);
        if (op == "masked") valid[r] = std::max(1, valid[r]);
        extents[r] = std::min(w - 1, valid[r]);
        starts[r]  = positions[r];
        ends[r]    = starts[r] + valid[r];
        columns[r] = valid[r] ? std::min(w - 1, valid[r] - 1) : -1;
        offsets[r] = valid[r] ? std::int64_t(lanes[r]) * record_stride : -1;
    }
    auto ab = m::make_i32(anchors), pb = m::make_i32(positions), vb = m::make_i32(valid),
         eb = m::make_i32(extents), lb = m::make_i32(lanes), sb = m::make_i32(starts),
         endb = m::make_i32(ends), cb = m::make_i32(columns);
    auto ob = bench::make_buffer(b * 8);
    ob.copy_from_host(offsets.data(), b * 8);
    auto ids = bench::make_buffer(w * b * 4), pos = bench::make_buffer(w * b * 4),
         nb     = bench::make_buffer(b * 4);
    auto drafts = m::make_i32(std::vector<int>((w - 1) * b, 12345));
    Tensor a(ab.span(), DType::I32, {b}), p(pb.span(), DType::I32, {b}),
        n(vb.span(), DType::I32, {b}), extent(eb.span(), DType::I32, {b}),
        lane(lb.span(), DType::I32, {b}), start(sb.span(), DType::I32, {b}),
        end(endb.span(), DType::I32, {b}), selected(cb.span(), DType::I32, {b}),
        off(ob.span(), DType::I64, {b});
    Tensor output_ids(ids.span(), DType::I32, {w, b}),
        output_positions(pos.span(), DType::I32, {w, b}), output_counts(nb.span(), DType::I32, {b}),
        draft(drafts.span(), DType::I32, {w - 1, b});
    const bool feature_op = op != "masked" && op != "verify" && op != "verify-ids";
    auto source = feature_op ? bench::make_bf16(std::size_t(pitch) * w * 8, 6101) : DeviceBuffer{};
    auto dense  = feature_op ? bench::make_bf16(std::size_t(d) * w * b, 6103) : DeviceBuffer{};
    auto destination =
        feature_op ? bench::make_zeros(std::size_t(pitch) * w * 8 * 2) : DeviceBuffer{};
    auto gathered = feature_op ? bench::make_zeros(std::size_t(d) * b * 2) : DeviceBuffer{};
    auto records =
        feature_op ? bench::make_bf16(std::size_t(record_stride / 2) * 8, 6109) : DeviceBuffer{};
    Tensor x, y, z, g;
    StateFieldView state;
    if (feature_op) {
        x       = Tensor(source.span(), DType::BF16, {d, w, 8});
        x.nb[1] = pitch * 2;
        x.nb[2] = x.nb[1] * w;
        x.nb[3] = x.nb[2] * 8;
        y       = Tensor(dense.span(), DType::BF16, {d, w, b});
        z       = Tensor(destination.span(), DType::BF16, {d, w, 8});
        z.nb[1] = x.nb[1];
        z.nb[2] = x.nb[2];
        z.nb[3] = x.nb[3];
        g       = Tensor(gathered.span(), DType::BF16, {d, b});
        TensorRegion field{{0, std::size_t(d) * 2, 16}, DType::BF16, {d, 1, 1, 1}};
        state = {{records.span(), off}, StateFieldLayout{field, 1, 0}};
    }
    auto choose = columns;
    for (auto& v : choose) v = std::max(0, v);
    auto chooser = m::make_i32(choose);
    Tensor selection(chooser.span(), DType::I32, {b});
    const auto launch = [&](CudaExecutionView cuda) {
        if (op == "masked") {
            // This Op's nonempty query contract excludes zero valid columns.
            ops::prepare_masked_block(a, p, n, 154856, output_ids, output_positions, cuda);
        } else if (op == "verify")
            ops::prepare_verify_inputs(a, draft, p, extent, output_ids, output_positions, cuda);
        else if (op == "verify-ids")
            ops::prepare_verify_ids(a, draft, extent, output_ids, cuda);
        else if (op == "ragged")
            ops::prepare_ragged_prefix(x, lane, start, end, y, output_positions, output_counts,
                                       cuda);
        else if (op == "scatter")
            ops::scatter_bf16_batch(y, lane, n, z, cuda);
        else if (op == "gather")
            ops::gather_bf16_batch_columns(y, selection, g, cuda);
        else if (op == "state-gather") {
            const ops::StateGather field{state, n, g};
            ops::gather_state_fields(std::span(&field, 1), cuda);
        } else {
            const ops::StateColumnScatter field{y, selected, state};
            ops::scatter_state_fields(std::span(&field, 1), cuda);
        }
    };
    const auto live   = std::accumulate(valid.begin(), valid.end(), std::int64_t(0));
    const int active  = std::count_if(valid.begin(), valid.end(), [](int v) { return v > 0; });
    std::size_t bytes = 0;
    if (op == "masked")
        bytes = 12 * b + 8 * w * b;
    else if (op == "verify")
        bytes = 12 * b + 4 * std::accumulate(extents.begin(), extents.end(), 0) + 8 * w * b;
    else if (op == "verify-ids")
        bytes = 8 * b + 4 * std::accumulate(extents.begin(), extents.end(), 0) + 4 * w * b;
    else if (op == "ragged")
        bytes = std::size_t(d) * 2 * (live + std::int64_t(w) * b) + 4 * w * b + 16 * b;
    else if (op == "scatter")
        bytes = std::size_t(d) * 4 * live + 8 * b;
    else if (op == "gather")
        bytes = std::size_t(d) * 4 * b + 4 * b;
    else if (op == "state-gather")
        bytes = std::size_t(d) * 2 * (active + b) + 4 * b + 8 * active;
    else
        bytes = std::size_t(d) * 4 * active + 4 * b + 8 * active;
    m::Row row;
    row.op     = op;
    row.format = feature_op ? "bf16" : "i32";
    row.n      = feature_op ? d : 1;
    row.tokens = w;
    row.batch  = b;
    row.valid  = o.counts;
    for (const auto& execution : m::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : m::selections(o.timing.cache, "cold", "warm")) {
            auto result =
                m::measure(launch, [](CudaExecutionView) {}, flush, e, o.timing, execution, cache);
            csv.write(row, result, execution, cache,
                      std::to_string(live) + ',' + std::to_string(bytes) + ',' +
                          std::to_string(pitch) + ',' + std::to_string(record_stride));
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        bench::L2FlushBuffer flush(o.timing.flush_bytes);
        m::Csv csv(o.timing.csv_out,
                   "configured_prefix_columns,semantic_bytes,column_features,record_stride_bytes");
        for (const auto& op : o.ops)
            for (int w : o.widths)
                for (int b : o.batches) run(o, op, w, b, csv, flush);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
