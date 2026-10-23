#include "ops/suite.h"
#include "kv_cache_storage.h"
#include "tail_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

#include <limits>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa;

namespace {

enum class Binding { Separate, InPlace, ShiftedInPlace, SamePool, RawInPlace };

int run(DeviceContext& device, int batch, int layout, Binding binding, int capacity,
        bool exhaustive = true) {
    const int slots = 2 * batch + 3;
    Plane<std::uint16_t> before(128, 3, slots, layout, true),
        after(128, 3, batch + 2, layout, true), records(128, capacity, batch, layout);
    Plane<std::int32_t> first(3, slots, 1, layout, true, true),
        last(3, binding == Binding::RawInPlace ? slots : batch + 2, 1, layout, true, true),
        positions(3, capacity, batch, layout);
    const bool raw_alias  = binding != Binding::Separate;
    const bool rope_alias = raw_alias && binding != Binding::RawInPlace;
    Tensor raw_out        = raw_alias ? before.view : after.view;
    Tensor rope_out       = rope_alias ? first.view : last.view;
    if (binding == Binding::ShiftedInPlace) {
        raw_out  = raw_out.slice(2, 1, slots - 1);
        rope_out = rope_out.slice(1, 1, slots - 1);
    }
    DeviceBuffer source(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer source_rope(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer destination(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer destination_rope(ninfer::test::cuda_environment().context.device(), batch * 8);
    DeviceBuffer base(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer valid(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer accepted(ninfer::test::cuda_environment().context.device(), batch * 4);
    const ops::QsaTailStateView input{
        state_field(before.view, before.device->span(), Tensor(source.span(), DType::I64, {batch})),
        state_field(first.view, first.device->span(),
                    Tensor(source_rope.span(), DType::I64, {batch}), true)};
    const ops::QsaTailStateView output{
        state_field(raw_out, raw_alias ? before.device->span() : after.device->span(),
                    Tensor(destination.span(), DType::I64, {batch})),
        state_field(rope_out, rope_alias ? first.device->span() : last.device->span(),
                    Tensor(destination_rope.span(), DType::I64, {batch}), true)};
    const ops::QsaRawRecordReadView read{records.view, positions.view,
                                         Tensor(base.span(), DType::I32, {batch}),
                                         Tensor(valid.span(), DType::I32, {batch})};
    const Tensor count(accepted.span(), DType::I32, {batch});
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures    = 0;
    const int cases = exhaustive ? 4 * (capacity + 1) : 8;
    for (int phase = 0; phase < cases; ++phase) {
        before.fill(101 + phase);
        first.fill(103 + phase);
        after.fill(107 + phase);
        last.fill(109 + phase);
        records.fill(113 + phase);
        positions.fill(127 + phase);
        std::vector<int> sources(batch), destinations(batch), bases(batch), counts(batch),
            available(batch);
        for (int b = 0; b < batch; ++b) {
            const bool shared_source = binding == Binding::Separate && phase % 2;
            sources[b]               = shared_source ? 1 : 1 + (b + phase) % batch;
            const int remainder      = exhaustive ? phase / (capacity + 1) : (phase + b) % 4;
            // Exercise both small positions and frontiers close to the I32 domain boundary.
            bases[b] = capacity < 1024 ? 60 + remainder
                                       : std::numeric_limits<int>::max() - capacity - 3 + remainder;
            if (!exhaustive && shared_source) bases[b] = bases[0];
            counts[b]       = exhaustive ? (phase % (capacity + 1) + b) % (capacity + 1)
                                         : std::array{0,
                                                1,
                                                std::min(2, capacity),
                                                std::min(3, capacity),
                                                capacity / 2,
                                                std::max(0, capacity - 2),
                                                capacity - 1,
                                                capacity}[(phase + b) % 8];
            available[b]    = counts[b] + (capacity - counts[b]) / (1 + phase % 3);
            const int dst   = binding == Binding::SamePool ? batch + 1 + b
                              : raw_alias                  ? sources[b]
                                                           : batch - b;
            destinations[b] = dst - (binding == Binding::ShiftedInPlace ? 1 : 0);
            for (int j = bases[b] % 4; j < 3; ++j)
                for (int c = 0; c < 128; ++c) before.host[before.index(c, j, sources[b])] = 0;
            if (bases[b] % 4 == 0)
                for (int axis = 0; axis < 3; ++axis) first.host[first.index(axis, sources[b])] = 0;
        }
        auto expected_raw  = raw_alias ? before.host : after.host;
        auto expected_rope = rope_alias ? first.host : last.host;
        for (int b = 0; b < batch; ++b) {
            const auto result =
                reference(before, first, records, positions, sources[b], b, bases[b], counts[b]);
            const int dst    = destinations[b] + (binding == Binding::ShiftedInPlace ? 1 : 0);
            const auto& raw  = raw_alias ? before : after;
            const auto& rope = rope_alias ? first : last;
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < 128; ++c) expected_raw[raw.index(c, j, dst)] = result.raw[j][c];
            for (int axis = 0; axis < 3; ++axis)
                expected_rope[rope.index(axis, dst)] = result.rope[axis];
        }
        before.upload();
        first.upload();
        after.upload();
        last.upload();
        records.upload();
        positions.upload();
        const auto source_offsets           = state_offsets(before.view, sources);
        const auto source_rope_offsets      = state_offsets(first.view, sources, true);
        const auto destination_offsets      = state_offsets(raw_out, destinations);
        const auto destination_rope_offsets = state_offsets(rope_out, destinations, true);
        source.copy_from_host(source_offsets.data(), batch * 8);
        source_rope.copy_from_host(source_rope_offsets.data(), batch * 8);
        destination.copy_from_host(destination_offsets.data(), batch * 8);
        destination_rope.copy_from_host(destination_rope_offsets.data(), batch * 8);
        base.copy_from_host(bases.data(), batch * 4);
        valid.copy_from_host(available.data(), batch * 4);
        accepted.copy_from_host(counts.data(), batch * 4);
        cuda_synchronize();
        if (phase == 0) {
            definition.capture(device.stream(), [&] {
                ops::qsa_tail_extract(input, read, count, output, device.execution_view());
            });
            graph.instantiate(definition);
        }
        if (phase % 2)
            ops::qsa_tail_extract(input, read, count, output, device.execution_view());
        else
            graph.launch(device.stream());
        device.synchronize();
        const auto label = "QSA tail B=" + std::to_string(batch) +
                           " layout=" + std::to_string(layout) +
                           " binding=" + std::to_string(int(binding)) +
                           " R=" + std::to_string(capacity) + " phase=" + std::to_string(phase);
        failures += before.check(label + " before raw", raw_alias ? expected_raw : before.host);
        failures += first.check(label + " before rope", rope_alias ? expected_rope : first.host);
        failures += after.check(label + " after raw", raw_alias ? after.host : expected_raw);
        failures += last.check(label + " after rope", rope_alias ? last.host : expected_rope);
        failures += records.check(label + " records", records.host) +
                    positions.check(label + " positions", positions.host);
        failures += verify_exact("source metadata", from_device<std::int64_t>(source, batch),
                                 source_offsets);
        failures +=
            verify_exact("destination metadata", from_device<std::int64_t>(destination, batch),
                         destination_offsets);
        failures +=
            verify_exact("source rope metadata", from_device<std::int64_t>(source_rope, batch),
                         source_rope_offsets);
        failures += verify_exact("destination rope metadata",
                                 from_device<std::int64_t>(destination_rope, batch),
                                 destination_rope_offsets);
        failures += verify_exact("base metadata", from_device<int>(base, batch), bases);
        failures += verify_exact("valid metadata", from_device<int>(valid, batch), available);
        failures += verify_exact("accepted metadata", from_device<int>(accepted, batch), counts);
    }
    return failures;
}

// Only the actually consumed record columns are initialized. Initcheck therefore catches
// reads of rejected suffixes, earlier complete blocks, and unused coordinate columns.
int read_footprint(DeviceContext& device) {
    int failures = 0;
    for (auto [f0, a] : {std::pair{6, 0}, {6, 1}, {6, 2}, {6, 9}, {8, 3}, {0, 0}}) {
        constexpr int capacity = 16;
        Plane<std::uint16_t> before(128, 3, 1, 0), records(128, capacity, 1, 0);
        Plane<std::int32_t> first(3, 1, 1, 0), positions(3, capacity, 1, 0);
        before.fill(501);
        records.fill(503);
        first.fill(507);
        positions.fill(509);
        for (int j = f0 % 4; j < 3; ++j)
            for (int c = 0; c < 128; ++c) before.host[before.index(c, j)] = 0;
        if (f0 % 4 == 0)
            for (int axis = 0; axis < 3; ++axis) first.host[first.index(axis, 0)] = 0;
        before.upload();
        first.upload();
        DeviceBuffer raw(ninfer::test::cuda_environment().context.device(), capacity * 128 * 2);
        DeviceBuffer rope(ninfer::test::cuda_environment().context.device(), capacity * 3 * 4);
        DeviceBuffer out(ninfer::test::cuda_environment().context.device(), 128 * 3 * 2);
        DeviceBuffer out_rope(ninfer::test::cuda_environment().context.device(), 3 * 4);
        const int end = f0 + a, start = end - end % 4;
        for (int p = std::max(f0, start); p < end; ++p)
            cuda_check(cudaMemcpy(static_cast<std::uint16_t*>(raw.data()) + (p - f0) * 128,
                                  records.host.data() + records.index(0, p - f0), 256,
                                  cudaMemcpyHostToDevice),
                       "record column upload");
        if (end % 4 && start >= f0)
            cuda_check(cudaMemcpy(static_cast<std::int32_t*>(rope.data()) + (start - f0) * 3,
                                  positions.host.data() + positions.index(0, start - f0), 12,
                                  cudaMemcpyHostToDevice),
                       "coordinate upload");
        auto zero = to_device(std::vector<std::int64_t>{0});
        auto base = to_device_i32({f0}), valid = to_device_i32({capacity}),
             accepted = to_device_i32({a});
        cuda_synchronize();
        ops::qsa_tail_extract(
            {state_field(before.view, before.device->span(), Tensor(zero.span(), DType::I64, {1})),
             state_field(first.view, first.device->span(), Tensor(zero.span(), DType::I64, {1}),
                         true)},
            {Tensor(raw.span(), DType::BF16, {128, capacity, 1}),
             Tensor(rope.span(), DType::I32, {3, capacity, 1}),
             Tensor(base.span(), DType::I32, {1}), Tensor(valid.span(), DType::I32, {1})},
            Tensor(accepted.span(), DType::I32, {1}),
            {state_field(Tensor(out.span(), DType::BF16, {128, 3, 1}), out.span(),
                         Tensor(zero.span(), DType::I64, {1})),
             state_field(Tensor(out_rope.span(), DType::I32, {3, 1}), out_rope.span(),
                         Tensor(zero.span(), DType::I64, {1}), true)},
            device.execution_view());
        device.synchronize();
        const auto expected = reference(before, first, records, positions, 0, 0, f0, a);
        std::vector<std::uint16_t> words;
        for (const auto& column : expected.raw)
            words.insert(words.end(), column.begin(), column.end());
        failures +=
            verify_exact("tail read footprint", from_device<std::uint16_t>(out, 384), words);
        failures +=
            verify_exact("coordinate read footprint", from_device<std::int32_t>(out_rope, 3),
                         std::vector<std::int32_t>(expected.rope.begin(), expected.rope.end()));
    }
    return failures;
}

template <class F>
int rejects(const char* label, F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": accepted invalid view\n";
    return 1;
}

int validation(DeviceContext& device) {
    Plane<std::uint16_t> raw(128, 3, 3, 0), records(128, 5, 2, 0);
    Plane<std::int32_t> rope(3, 3, 1, 0), positions(3, 5, 2, 0);
    DeviceBuffer metadata(ninfer::test::cuda_environment().context.device(), 8);
    const Tensor m(metadata.span(), DType::I32, {2});
    DeviceBuffer selected(ninfer::test::cuda_environment().context.device(), 16);
    const Tensor offsets(selected.span(), DType::I64, {2});
    const ops::QsaTailStateView state{state_field(raw.view, raw.device->span(), offsets),
                                      state_field(rope.view, rope.device->span(), offsets, true)};
    const ops::QsaRawRecordReadView read{records.view, positions.view, m, m};
    int failures             = 0;
    const auto invalid_state = [&](const char* name, auto change) {
        auto s = state;
        change(s);
        failures += rejects(
            name, [&] { ops::qsa_tail_extract(s, read, m, state, device.execution_view()); });
    };
    invalid_state("wrong raw dtype", [](auto& s) { s.raw_keys.field.tensor.dtype = DType::FP16; });
    invalid_state("wrong tail length", [](auto& s) { s.raw_keys.field.tensor.shape[1] = 4; });
    invalid_state("wrong coordinate shape",
                  [](auto& s) { s.first_rope.field.tensor.shape[0] = 2; });
    invalid_state("empty state arena", [](auto& s) { s.raw_keys.records.storage.bytes = 0; });
    invalid_state("truncated field", [](auto& s) { s.raw_keys.field.tensor.region.bytes = 128; });
    invalid_state("address overflow", [](auto& s) {
        s.raw_keys.field.tensor.region.offset = std::numeric_limits<std::size_t>::max();
    });
    invalid_state("unaligned coordinates",
                  [](auto& s) { s.first_rope.field.tensor.region.offset += 2; });
    invalid_state("strided metadata", [](auto& s) { s.raw_keys.records.byte_offsets.nb[0] = 16; });
    auto bad                 = read;
    bad.rope_positions.ne[1] = 4;
    failures += rejects("record capacity mismatch", [&] {
        ops::qsa_tail_extract(state, bad, m, state, device.execution_view());
    });
    bad                = read;
    bad.raw_keys.ne[1] = 0;
    failures += rejects("empty records", [&] {
        ops::qsa_tail_extract(state, bad, m, state, device.execution_view());
    });
    auto count  = m;
    count.ne[0] = 9;
    failures += rejects("too many rows", [&] {
        ops::qsa_tail_extract(state, read, count, state, device.execution_view());
    });
    count.ne[0] = 0;
    failures += rejects("empty batch", [&] {
        ops::qsa_tail_extract(state, read, count, state, device.execution_view());
    });
    invalid_state("invalid state device",
                  [](auto& s) { s.raw_keys.records.byte_offsets.device = Device::cpu(); });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("qsa_tail/contract", [] { return validation(cuda_environment().context); });
    suite.add("qsa_tail/read_footprint", [] { return read_footprint(cuda_environment().context); });
    const auto add = [&](int b, int layout, Binding binding, int capacity, bool exhaustive = true) {
        suite.add(
            "qsa_tail/b" + std::to_string(b) + "/layout" + std::to_string(layout) + "/binding" +
                std::to_string(int(binding)) + "/capacity" + std::to_string(capacity),
            [=] {
                return run(cuda_environment().context, b, layout, binding, capacity, exhaustive);
            },
            {{"execution", "graph"}});
    };
    add(1, 0, Binding::Separate, 9);
    add(8, 0, Binding::Separate, 9);
    add(8, 1, Binding::InPlace, 9);
    add(3, 2, Binding::InPlace, 9);
    add(8, 3, Binding::ShiftedInPlace, 9);
    add(8, 2, Binding::SamePool, 9);
    add(4, 1, Binding::RawInPlace, 9);
    for (int b = 1; b <= 8; ++b) add(b, b % 4, Binding::Separate, 1, false);
    for (int r : {2, 3, 16, 4096}) add(8, r % 4, Binding::Separate, r, false);

    return suite.main(argc, argv);
}
