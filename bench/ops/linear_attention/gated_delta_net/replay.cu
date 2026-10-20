#include "replay_fixture.cuh"
#include "traffic.h"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto options = gdn_replay::parse_options(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out,
                             std::string(gdn::Traffic::columns) + ",row_metadata");
        for (int t : options.tokens) {
            measurement::Row row;
            row.op            = options.op;
            row.tokens        = t;
            row.batch         = options.batch;
            row.conv_channels = options.op == "fold" ? options.profile.conv_channels : 0;
            row.qk_heads      = options.profile.qk_heads;
            row.value_heads   = options.profile.value_heads;
            row.layers        = options.op == "fold" ? options.profile.layers : 1;
            const auto run    = [&](auto&& launch, auto&& restore, const gdn::Traffic& traffic) {
                for (const auto& execution :
                     measurement::selections(options.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(options.timing.cache, "cold", "warm")) {
                        const auto r = measurement::measure(
                            launch, restore, flush, device.execution_view(), options.timing,
                            execution, cache, traffic.active_rows == 0);
                        csv.write(row, r, execution, cache,
                                     traffic.csv(r.gpu.median_us) + ',' +
                                         (options.op == "fold" ? options.row_metadata : "none"));
                    }
            };
            if (options.op == "fold") {
                gdn_replay::FoldFixture f(options.profile, t);
                const auto counts = gdn_replay::extents(options.commits, t, options.batch, true);
                constexpr int slots[] = {10, 2, 8, 0, 6, 4, 9, 1};
                std::vector<ops::GdnReplayFoldRow> rows;
                for (int b = 0; b < options.batch; ++b)
                    rows.push_back({slots[b], slots[b], counts[b]});
                const bool on_device = options.row_metadata == "device";
                auto selected =
                    on_device
                        ? measurement::make_i32(std::vector<int>(slots, slots + options.batch))
                        : DeviceBuffer{};
                auto prefixes = on_device ? measurement::make_i32(counts) : DeviceBuffer{};
                Tensor selectors, commits;
                if (on_device) {
                    selectors = Tensor(selected.span(), DType::I32, {options.batch});
                    commits   = Tensor(prefixes.span(), DType::I32, {options.batch});
                }
                row.commits = gdn_replay::csv_extent(options.commits);
                run(
                    [&](CudaExecutionView cuda) {
                        if (on_device)
                            f.fold_plan().execute(selectors, selectors, commits, cuda);
                        else
                            f.fold_plan().execute(rows, cuda);
                    },
                    [&](CudaExecutionView cuda) { f.restore(cuda); },
                    gdn::fold_traffic(options.profile.layers, options.profile.qk_heads,
                                      options.profile.value_heads, options.profile.conv_channels,
                                      counts));
            } else {
                auto valid = gdn_replay::extents(options.valid, t, options.batch, false);
                const auto traffic =
                    gdn::record_traffic(options.profile.qk_heads, options.profile.value_heads, t,
                                        valid, options.valid != "dense");
                if (options.valid == "dense") valid.clear();
                gdn_replay::RecordFixture f(options.profile, t, options.batch, valid);
                row.valid = gdn_replay::csv_extent(options.valid);
                run([&](CudaExecutionView cuda) { f.launch_record(cuda); }, [](auto) {}, traffic);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
