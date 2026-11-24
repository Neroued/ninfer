#include "ops/suite.h"
#include "fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/ple.h"
#include "core/decode_graph.h"

#include <cstring>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::ple;

namespace {
int run(DeviceContext& device, int batch, int layout, bool inplace, bool sliced,
        bool masked = false) {
    constexpr int capacity = 19;
    const int slots        = batch + 3;
    Words source({D, 9, slots}, layout == 1, history_strides(slots, layout));
    Words destination({D, 9, batch + 1}, layout == 1, history_strides(batch + 1, layout));
    Words records({D, capacity, batch}, layout == 1, column_strides(capacity, batch, layout));
    source.fill(211, 1.f, true);
    destination.fill(223, 1.f, true);
    records.fill(227, 1.f, true);
    std::vector<int> sources(batch), destinations(batch), accepted(batch), valid(batch, capacity);
    DeviceBuffer ds(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer dd(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer da(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer dv = to_device(valid);
    Tensor before = source.tensor, after = inplace ? source.tensor : destination.tensor;
    if (sliced) after = before.slice(2, 2, slots - 2);
    ops::PleConvStateView input{
        history_field(before, source.device->span(), Tensor(ds.span(), DType::I64, {batch}))};
    ops::PleConvStateView output{
        history_field(after, inplace ? source.device->span() : destination.device->span(),
                      Tensor(dd.span(), DType::I64, {batch}))};
    ops::PleConvRecordReadView read{records.tensor, Tensor(dv.span(), DType::I32, {batch})};
    Tensor counts(da.span(), DType::I32, {batch});
    std::vector<int> active(batch, 1);
    auto live_storage = to_device(active);
    Tensor live       = masked ? Tensor(live_storage.span(), DType::I32, {batch}) : Tensor{};
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int extent = 0; extent <= capacity; ++extent) {
        if (masked && extent != 0 && extent != 2 && extent != 9 && extent != 19) continue;
        source.upload();
        destination.upload();
        records.upload();
        for (int b = 0; b < batch; ++b) {
            sources[b]      = inplace ? b + 2 : (extent % 2 ? 1 : b + 2);
            destinations[b] = inplace ? (sliced ? b : b + 2) : batch - b;
            accepted[b]     = (extent + b) % (capacity + 1);
            active[b]       = !masked || (b + extent) % 3 != 0;
        }
        auto source_offsets      = history_offsets(before, sources);
        auto destination_offsets = history_offsets(after, destinations);
        for (int b = 0; b < batch; ++b)
            if (!active[b]) source_offsets[b] = destination_offsets[b] = -1;
        ds.copy_from_host(source_offsets.data(), ds.size());
        dd.copy_from_host(destination_offsets.data(), dd.size());
        upload_i32(live_storage, active);
        upload_i32(da, accepted);
        auto expected = inplace ? source.host : destination.host;
        for (int b = 0; b < batch; ++b)
            if (active[b])
                for (int c = 0; c < D; ++c) {
                    std::array<std::uint16_t, capacity + 9> sequence;
                    for (int j = 0; j < 9; ++j)
                        sequence[j] = source.host[source.index(c, j, sources[b])];
                    for (int j = 0; j < accepted[b]; ++j)
                        sequence[9 + j] = records.host[records.index(c, j, b)];
                    for (int j = 0; j < 9; ++j) {
                        const auto dst = inplace ? source.index(c, j, b + 2)
                                                 : destination.index(c, j, destinations[b]);
                        expected[dst]  = sequence[accepted[b] + j];
                    }
                }
        cuda_synchronize();
        if (!extent) {
            definition.capture(device.stream(), [&] {
                ops::ple_state_extract(input, read, counts, output, live, device.execution_view());
            });
            graph.instantiate(definition);
        }
        if (extent % 2)
            ops::ple_state_extract(input, read, counts, output, live, device.execution_view());
        else
            graph.launch(device.stream());
        device.synchronize();
        const auto label =
            "PLE extract B=" + std::to_string(batch) + " layout=" + std::to_string(layout) +
            " inplace=" + std::to_string(inplace) + " sliced=" + std::to_string(sliced) +
            " extent=" + std::to_string(extent);
        failures += verify_exact(label.c_str(),
                                 inplace ? source.download() : destination.download(), expected);
        failures += records.preserved("extract preserves records");
        if (!inplace) failures += source.preserved("extract preserves source");
        failures += source.device->verify_guards(label) + destination.device->verify_guards(label);
        failures += verify_exact("accepted preserved", from_device<int>(da, batch), accepted);
        failures += verify_exact("source slots preserved", from_device<std::int64_t>(ds, batch),
                                 source_offsets);
        failures += verify_exact("destination slots preserved",
                                 from_device<std::int64_t>(dd, batch), destination_offsets);
        failures += verify_exact("valid preserved", from_device<int>(dv, batch), valid);
    }
    return failures;
}

int ragged_validation() {
    const RaggedFixture ragged({1, 1});
    Words history({D, 9, 2}), values({D, 2});
    auto offsets = to_device(history_offsets(history.tensor, {0, 1}));
    history.upload();
    values.upload();
    cuda_synchronize();
    const auto state  = history_field(history.tensor, history.device->span(),
                                      Tensor(offsets.span(), DType::I64, {2}));
    auto& device      = cuda_environment().context;
    const auto launch = [&](const StateFieldView& after) {
        ops::ple_state_extract(ops::PleConvStateView{state}, values.tensor, ragged.view, {after},
                               device.execution_view());
    };
    auto invalid         = state;
    invalid.field.layers = 2;
    const int failures   = rejects("ragged extract history layer", [&] { launch(invalid); });
    invalid              = state;
    invalid.field.tensor.region.offset += 1;
    return failures + rejects("ragged extract history base alignment", [&] { launch(invalid); });
}

int ragged_extract(int layout, std::vector<int> source, std::vector<int> destination, bool graph,
                   std::vector<int> lengths = {1, 2, 8, 9, 10, 65}) {
    RaggedFixture ragged(std::move(lengths));
    constexpr int slots = 13;
    const int rows = ragged.segments(), n = ragged.tokens();
    const bool inplace = source == destination;
    Words history({D, 9, slots}, layout == 1, history_strides(slots, layout == 1)),
        values({D, n}, layout == 1, column_strides(n, 1, layout));
    auto& device = cuda_environment().context;
    DeviceBuffer src(device.device(), rows * sizeof(std::int64_t));
    DeviceBuffer dst(device.device(), rows * sizeof(std::int64_t));
    const auto input = history_field(history.tensor, history.device->span(),
                                     Tensor(src.span(), DType::I64, {rows}));
    const auto output =
        history_field(history.tensor, history.device->span(),
                      Tensor(inplace ? src.span() : dst.span(), DType::I64, {rows}));
    const auto launch = [&] {
        ops::ple_state_extract(ops::PleConvStateView{input}, values.tensor, ragged.view, {output},
                               device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) rotate_replay_partition(ragged);
        history.fill(1001 + phase, 1.0f, true);
        values.fill(1013 + phase, 1.0f, true);
        for (int s = 0; s < slots; ++s)
            for (int h = 0; h < 9; ++h) history.host[history.index(0, h, s)] = 0x8000;
        for (int t = 0; t < n; ++t) values.host[values.index(1, t)] = 0x8000;
        if (phase) {
            for (int& slot : source) slot = (slot + 1) % slots;
            for (int& slot : destination) slot = (slot + 1) % slots;
        }
        history.upload();
        values.upload();
        upload_offsets(src, history.tensor, source);
        if (!inplace) upload_offsets(dst, history.tensor, destination);
        auto expected = history.host;
        for (int s = 0; s < rows; ++s)
            for (int h = 0; h < 9; ++h) {
                const int local = ragged.lengths[s] - 9 + h;
                const auto* from =
                    local < 0 ? history.host.data() + history.index(0, local + 9, source[s])
                              : values.host.data() + values.index(0, ragged.prefix[s] + local);
                std::copy_n(from, D, expected.begin() + history.index(0, h, destination[s]));
            }
        cuda_synchronize();
        if (graph && phase == 0) {
            definition.capture(device.stream(), launch);
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream());
        else
            launch();
        device.synchronize();
        failures +=
            verify_exact("ragged updated histories and neighbours", history.download(), expected);
        failures += history.device->verify_guards("ragged PLE state") +
                    values.preserved("ragged state records");
        failures += verify_exact("ragged source offsets", from_device<std::int64_t>(src, rows),
                                 history_offsets(history.tensor, source));
        if (!inplace)
            failures +=
                verify_exact("ragged destination offsets", from_device<std::int64_t>(dst, rows),
                             history_offsets(history.tensor, destination));
    }
    return failures;
}

int ragged_shared_storage() {
    constexpr int history_columns = 9;
    const RaggedFixture ragged({1, 3});
    const int rows = ragged.segments(), n = ragged.tokens();
    const int first_record = 1, first_history = first_record + n;
    Words pool({D, first_history + rows * history_columns});
    const auto storage = pool.device->span();
    const auto records = pool.tensor.slice(1, first_record, n);
    const auto history = pool.tensor.slice(1, first_history, rows * history_columns)
                             .view({D, history_columns, rows});
    const auto offsets                  = history_offsets(history, {0, 1});
    const auto offsets_bytes            = offsets.size() * sizeof(std::int64_t);
    auto state                          = history_field(history, storage,
                                                        Tensor(storage.subspan(0, offsets_bytes), DType::I64, {rows}));
    state.field.tensor.region.alignment = 1;
    auto layout                         = ragged.view;
    const auto mapping_bytes            = ragged.mapping.size() * sizeof(int);
    layout.token_to_seq = Tensor(storage.subspan(offsets_bytes, mapping_bytes), DType::I32, {n});
    pool.fill(1237, 1.f, true);
    auto* metadata = reinterpret_cast<std::byte*>(pool.host.data());
    std::memcpy(metadata, offsets.data(), offsets_bytes);
    std::memcpy(metadata + offsets_bytes, ragged.mapping.data(), mapping_bytes);
    pool.upload();
    cuda_synchronize();

    auto expected = pool.host;
    for (int s = 0; s < rows; ++s) {
        const int dst       = first_history + s * history_columns;
        const int length    = ragged.lengths[s];
        const auto kept     = pool.host.begin() + pool.index(0, dst + length);
        const auto appended = pool.host.begin() + pool.index(0, first_record + ragged.prefix[s]);
        auto output         = expected.begin() + pool.index(0, dst);
        output              = std::copy_n(kept, D * (history_columns - length), output);
        std::copy_n(appended, D * length, output);
    }
    auto& device = cuda_environment().context;
    const ops::PleConvStateView in_place{state};
    ops::ple_state_extract(in_place, records, layout, in_place, device.execution_view());
    device.synchronize();
    return verify_exact("extract preserves shared records and metadata", pool.download(),
                        expected) +
           pool.device->verify_guards("ragged shared storage");
}

int ragged_extract_grid_slice() {
    constexpr int rows = 65'536, history_columns = 9;
    constexpr std::size_t record_bytes  = D * sizeof(std::uint16_t);
    constexpr std::size_t history_bytes = history_columns * record_bytes;
    auto& device                        = cuda_environment().context;
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < (16ULL << 30)) throw OpTestSkip("requires 16 GiB of free CUDA memory");

    const RaggedFixture ragged(std::vector<int>(rows, 1));
    DeviceBuffer history(device.device(), rows * history_bytes);
    DeviceBuffer records(device.device(), rows * record_bytes);
    std::vector<std::int64_t> offsets(rows);
    for (int row = 0; row < rows; ++row) offsets[row] = row * history_bytes;
    auto d_offsets = to_device(offsets);
    const auto state =
        history_field(Tensor(history.span(), DType::BF16, {D, history_columns, rows}),
                      history.span(), Tensor(d_offsets.span(), DType::I64, {rows}));
    history.fill(0x31);
    records.fill(0x11);
    cuda_synchronize();

    constexpr std::array boundary_rows{0, 1, rows - 2, rows - 1};
    std::array<std::vector<std::uint16_t>, boundary_rows.size()> expected;
    std::vector<std::uint16_t> before(D * history_columns), appended(D);
    for (int i = 0; i < int(boundary_rows.size()); ++i) {
        const int row = boundary_rows[i];
        std::iota(before.begin(), before.end(), std::uint16_t(0x8000 + i * D));
        std::iota(appended.begin(), appended.end(), std::uint16_t(i * D));
        history.copy_from_host(before.data(), history_bytes, offsets[row]);
        records.copy_from_host(appended.data(), record_bytes, row * record_bytes);
        expected[i].assign(before.begin() + D, before.end());
        expected[i].insert(expected[i].end(), appended.begin(), appended.end());
    }
    const ops::PleConvStateView in_place{state};
    ops::ple_state_extract(in_place, Tensor(records.span(), DType::BF16, {D, rows}), ragged.view,
                           in_place, device.execution_view());
    device.synchronize();

    int failures = 0;
    std::vector<std::uint16_t> actual(D * history_columns);
    for (int i = 0; i < int(boundary_rows.size()); ++i) {
        const int row = boundary_rows[i];
        history.copy_to_host(actual.data(), history_bytes, offsets[row]);
        const auto label = "extract grid slice row " + std::to_string(row);
        failures += verify_exact(label.c_str(), actual, expected[i]);
    }
    return failures;
}
} // namespace

void add_ple_state_extract_cases(ninfer::test::OpSuite& suite) {
    suite.add("ple_state_extract/ragged/contract", ragged_validation);
    suite.add("ple_state_extract/ragged/shared_storage", ragged_shared_storage);
    suite.add("ple_state_extract/ragged/grid_slice", ragged_extract_grid_slice,
              {{"suite", "concurrency"}});
    suite.add("ple_state_extract/ragged/single",
              [] { return ragged_extract(0, {3}, {3}, false, {2}); });
    suite.add("ple_state_extract/ragged/paired_alias",
              [] { return ragged_extract(0, {3, 1, 6, 0, 7, 9}, {3, 1, 6, 0, 7, 9}, true); },
              {{"execution", "graph"}});
    suite.add("ple_state_extract/ragged/scalar_mixed",
              [] { return ragged_extract(1, {3, 1, 6, 0, 7, 9}, {3, 4, 6, 2, 8, 9}, true); },
              {{"execution", "graph"}});
    suite.add("ple_state_extract/ragged/padded_shared_source",
              [] { return ragged_extract(2, {3, 1, 1, 0, 7, 3}, {12, 4, 11, 2, 8, 9}, false); });
    const auto add = [&](int batch, int layout, bool inplace, bool sliced) {
        suite.add("ple_state_extract/b" + std::to_string(batch) + "/layout" +
                      std::to_string(layout) + (inplace ? "/alias" : "/disjoint") +
                      (sliced ? "/sliced" : "/full"),
                  [=] { return run(cuda_environment().context, batch, layout, inplace, sliced); },
                  {{"execution", "graph"}});
    };
    suite.add("ple_state_extract/b17/masked",
              [] { return run(cuda_environment().context, 17, 0, false, false, true); },
              {{"execution", "graph"}, {"suite", "concurrency"}});
    for (int layout : {0, 1, 2}) {
        add(1, layout, false, false);
        add(8, layout, false, false);
        add(3, layout, true, false);
        add(8, layout, true, true);
    }
}
