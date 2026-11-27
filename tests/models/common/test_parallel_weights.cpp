#include "models/common/parallel_weights.h"

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "models/common/bindings.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models;
using test::artifact_fixture::Fixture;
using test::artifact_fixture::Json;
using test::artifact_fixture::put_word;
using test::artifact_fixture::rejects;
using test::artifact_fixture::require;

void test_vocabulary_partition_laws() {
    // 1. Prime 128-block count: 151936 = 1187 * 128 across tp=2, 4, 8
    for (const std::uint32_t tp : {1U, 2U, 4U, 8U}) {
        std::uint32_t cursor = 0;
        for (std::uint32_t r = 0; r < tp; ++r) {
            const auto range = partition_vocabulary_range(151936, tp, r);
            require(range.begin == cursor && range.size > 0, "151936 partition gap or empty");
            if (tp > 1) require(range.begin % 128 == 0 && range.size % 128 == 0, "128 alignment");
            cursor += range.size;
        }
        require(cursor == 151936, "151936 partition total mismatch");
    }

    // 2. Non-128 tail with vocabulary >= 128 * tp: vocab = 300, tp = 2 -> [0, 128), [128, 300)
    {
        const auto r0 = partition_vocabulary_range(300, 2, 0);
        const auto r1 = partition_vocabulary_range(300, 2, 1);
        require(r0.begin == 0 && r0.size == 128 && r1.begin == 128 && r1.size == 172,
                "300-vocab tp2 must keep interior boundary 128-aligned and absorb tail on rank 1");
    }

    // 3. Small-vocabulary fallback: vocab = 19 < 128 * 4 -> {5, 5, 5, 4}
    {
        std::uint32_t cursor = 0;
        const std::array<std::uint32_t, 4> expected{5, 5, 5, 4};
        for (std::uint32_t r = 0; r < 4; ++r) {
            const auto range = partition_vocabulary_range(19, 4, r);
            require(range.begin == cursor && range.size == expected[r], "small-vocab fallback");
            cursor += range.size;
        }
        require(cursor == 19, "small-vocab total mismatch");
    }

    // 4. Invalid inputs rejected
    rejects<std::invalid_argument>([] { (void)partition_vocabulary_range(0, 1, 0); },
                                   "accepted empty vocabulary");
    rejects<std::invalid_argument>([] { (void)partition_vocabulary_range(3, 4, 0); },
                                   "accepted vocabulary < parts");
    rejects<std::invalid_argument>([] { (void)partition_vocabulary_range(256, 2, 2); },
                                   "accepted rank >= parts");
}

struct WeightFixture {
    Fixture file;
    static constexpr std::uint32_t kVocab  = 300;
    static constexpr std::uint32_t kHidden = 16;

    WeightFixture() {
        // Build clean contiguous BF16/FP32 tensors in one file:
        // - head: [300, 16] bf16 = 9600 bytes @ offset 0
        // - conv: [4, 12] bf16   = 96 bytes   @ offset 9728
        // - proj: [8, 16] bf16   = 256 bytes  @ offset 9984
        // - norm: [16] bf16      = 32 bytes   @ offset 10368
        // - div:  [16] fp32      = 64 bytes   @ offset 10496
        file.payload.assign(12288, std::byte{0});
        for (std::uint32_t row = 0; row < kVocab; ++row)
            for (std::uint32_t col = 0; col < kHidden; ++col)
                put_word(file.payload, (row * kHidden + col) * 2,
                         static_cast<std::uint16_t>((row * 17U + col) & 0x7fffU), 2);
        for (std::uint32_t r = 0; r < 4; ++r)
            for (std::uint32_t c = 0; c < 12; ++c)
                put_word(file.payload, 9728 + (r * 12 + c) * 2,
                         static_cast<std::uint16_t>(100 + r * 12 + c), 2);
        for (std::uint32_t i = 0; i < 8 * 16; ++i)
            put_word(file.payload, 9984 + i * 2, static_cast<std::uint16_t>(500 + i), 2);
        for (std::uint32_t i = 0; i < 16; ++i)
            put_word(file.payload, 10240 + i * 2, static_cast<std::uint16_t>(900 + i), 2);
        for (std::uint32_t i = 0; i < 2; ++i) put_word(file.payload, 10496 + i * 4, 0x3f800000U, 4);
        for (std::uint32_t i = 0; i < 2 * 4 * 16; ++i)
            put_word(file.payload, 10752 + i * 2, static_cast<std::uint16_t>(1100 + i), 2);

        file.root = {
            {"components", {{"text", {{"config", test::artifact_fixture::Json::object()}}}}},
            {"objects", test::artifact_fixture::Json::array({{{"id", "head_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {kVocab, kHidden}},
                                                              {"format", "bf16"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 0},
                                                              {"bytes", kVocab * kHidden * 2}},
                                                             {{"id", "conv_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {4, 12}},
                                                              {"format", "bf16"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 9728},
                                                              {"bytes", 96}},
                                                             {{"id", "proj_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {8, 16}},
                                                              {"format", "bf16"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 9984},
                                                              {"bytes", 256}},
                                                             {{"id", "norm_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {16}},
                                                              {"format", "bf16"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 10240},
                                                              {"bytes", 32}},
                                                             {{"id", "div_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {2}},
                                                              {"format", "fp32"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 10496},
                                                              {"bytes", 8}},
                                                             {{"id", "bank_obj"},
                                                              {"kind", "tensor"},
                                                              {"shape", {2, 4, 16}},
                                                              {"format", "bf16"},
                                                              {"layout", "contiguous_le_v1"},
                                                              {"offset", 10752},
                                                              {"bytes", 256}}})},
            {"bindings",
             {{"head", {{"object", "head_obj"}}},
              {"conv", {{"object", "conv_obj"}}},
              {"key", {{"object", "proj_obj"}}},
              {"context_key", {{"object", "proj_obj"}}},
              {"norm", {{"object", "norm_obj"}}},
              {"active_bank", {{"object", "bank_obj"}}},
              {"omitted_proj", {{"object", "bank_obj"}}}}},
            {"uses",
             test::artifact_fixture::Json::array(
                 {{{"parameter", "head"}, {"input", "hidden"}, {"activation_policy", "AllowA8"}},
                  {{"parameter", "key"}, {"input", "q_in"}, {"activation_policy", "AllowA8"}},
                  {{"parameter", "context_key"},
                   {"input", "c_in"},
                   {"activation_policy", "AllowA8"}},
                  {{"parameter", "active_bank"},
                   {"input", "active_in"},
                   {"activation_policy", "AllowA8"},
                   {"auxiliaries", {{"activation_input_divisor", {{"object", "div_obj"}}}}}},
                  {{"parameter", "omitted_proj"},
                   {"input", "omitted_in"},
                   {"activation_policy", "AllowA8"},
                   {"auxiliaries", {{"activation_input_divisor", {{"object", "div_obj"}}}}}}})},
            {"files",
             test::artifact_fixture::Json::array({{{"path", nullptr}, {"payload_bytes", 12288}}})},
            {"metadata", {{"name", "tristate-weight-fixture"}}}};
        file.write();
    }
};

void test_tristate_selection_omission_and_head_reconstruction() {
    WeightFixture wf;
    artifact::Reader reader(wf.file.entry);
    artifact::Binder binder(reader);
    Bindings b(binder);

    const auto head_id =
        b.parameter("head", {WeightFixture::kVocab, WeightFixture::kHidden}, {"hidden"});
    const auto conv_id    = b.direct("conv", {4, 12});
    const auto key_id     = b.parameter("key", {8, 16}, {"q_in"});
    const auto ctx_key_id = b.parameter("context_key", {8, 16}, {"c_in"});
    const auto norm_id    = b.direct("norm", {16});
    const auto active_id  = b.parameter("active_bank", {2, 4, 16}, {"active_in"});
    const auto omit_id    = b.parameter("omitted_proj", {2, 4, 16}, {"omitted_in"});

    std::vector<artifact::ParameterReference> refs;
    std::vector<WeightParent> dummy_parents(reader.directory().objects.size());
    std::byte dummy_byte{0};
    std::vector<BoundWeight> descriptions;
    for (auto& p : b.weights) {
        WeightView pre_view{p.reference.shape, {}};
        for (const auto& part : p.reference.binding.parts) {
            auto& parent    = dummy_parents.at(part.object.index);
            parent.geometry = reader.geometry(part.object);
            parent.device   = Device::cpu();
            parent.data     = &dummy_byte;
            pre_view.parts.push_back({&parent, part.begin, part.end});
        }
        for (auto& use : p.uses) use.calibration_device = {&dummy_byte, 64, Device::cpu()};
        descriptions.push_back({p.reference.name, p.source_objects, std::move(pre_view), p.uses});
        refs.push_back(p.reference);
    }

    // 1. Missing classification rejected (no implicit whole_tensor fallback!)
    rejects<std::logic_error>(
        [&] {
            const std::array whole{head_id, conv_id, key_id, ctx_key_id, norm_id, active_id};
            (void)select_weights(reader, refs, descriptions, whole, {}, {}, {}, Device::cpu());
        },
        "unclassified parameter was implicitly materialized");

    // 2. Duplicate classification rejected
    rejects<std::logic_error>(
        [&] {
            const std::array whole{head_id, conv_id,   key_id, ctx_key_id,
                                   norm_id, active_id, omit_id};
            const std::array omitted{omit_id};
            (void)select_weights(reader, refs, descriptions, whole, {}, {}, omitted, Device::cpu());
        },
        "parameter classified in both whole and omitted was accepted");

    // 3. Multi-slice conv + aliased key/context_key + omitted parameter + 128-aligned 300-row head
    std::vector<std::uint16_t> reconstructed_head(WeightFixture::kVocab * WeightFixture::kHidden,
                                                  0xffffU);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto vocab = partition_vocabulary_range(WeightFixture::kVocab, 2, rank);
        std::vector<ParameterAssembly> assemblies;

        // Vocabulary head shard
        assemblies.push_back({{vocab.size, WeightFixture::kHidden},
                              {{head_id, vocab.begin, 0, vocab.size, WeightFixture::kHidden, 0, 0}},
                              {{head_id,
                                {vocab.size, WeightFixture::kHidden},
                                0,
                                std::uint64_t(vocab.size) * WeightFixture::kHidden}}});

        // 3 slices from conv_id into one assembly (like KDA/GDN convolution)
        ParameterAssembly conv;
        conv.shape = {4, 6};
        for (std::uint64_t f = 0; f < 3; ++f)
            conv.slices.push_back({conv_id, 0, f * 4 + rank * 2, 4, 2, 0, f * 2});
        conv.bindings.push_back({conv_id, {4, 6}, 0, 24});
        assemblies.push_back(std::move(conv));

        // Aliased key + context_key sharing one slice range (like DFlash2 attention)
        ParameterAssembly kv;
        kv.shape    = {4, 16};
        kv.slices   = {{key_id, rank * 4ULL, 0, 4, 16, 0, 0}};
        kv.bindings = {{key_id, {4, 16}, 0, 64}, {ctx_key_id, {4, 16}, 0, 64}};
        assemblies.push_back(std::move(kv));

        const std::array whole{norm_id, active_id};
        const std::array omitted{omit_id};
        const auto selected = select_weights(reader, refs, descriptions, whole, assemblies, {},
                                             omitted, Device::cpu());

        // Only active_id's calibration should be requested; omit_id's calibration MUST be skipped!
        require(selected.calibrations.size() == 1 &&
                    selected.calibrations[0].parameter == active_id.index,
                "omitted parameter leaked activation calibration request");

        auto mat_plan      = artifact::plan_materialization(reader, selected.requests);
        const auto req_ids = mat_plan.requests;
        auto backing       = artifact::materialize(reader, std::move(mat_plan));
        std::vector<WeightParent> parents;
        std::size_t max_storage = 0;
        for (const auto sid : req_ids) max_storage = std::max(max_storage, sid.index + 1);
        for (std::size_t i = 0; i < max_storage; ++i)
            parents.push_back(backing.parent({i}, Device::cpu()));

        const auto bound = bind_selected_weights(descriptions, selected, req_ids, parents);

        // Omitted weight MUST have empty view and cleared calibration_device!
        require(
            bound[omit_id.index].view.parts.empty() && bound[omit_id.index].view.shape.empty() &&
                bound[omit_id.index].uses[0].calibration_device.data == nullptr &&
                bound[omit_id.index].uses[0].calibration_device.bytes == 0,
            "bind_selected_weights retained stale view or calibration_device on Omitted weight");

        // Active 3D weight calibration MUST be bound (2 FP32 values = 8 bytes)
        require(bound[active_id.index].uses[0].calibration_device.data != nullptr &&
                    bound[active_id.index].uses[0].calibration_device.bytes == 8,
                "non-omitted calibration_device was not materialized");

        // Aliased bindings MUST point to identical region
        require(bound[key_id.index].view.parts.size() == 1 &&
                    bound[ctx_key_id.index].view.parts.size() == 1 &&
                    bound[key_id.index].view.parts[0].parent ==
                        bound[ctx_key_id.index].view.parts[0].parent &&
                    bound[key_id.index].view.parts[0].begin ==
                        bound[ctx_key_id.index].view.parts[0].begin,
                "aliased bindings lost shared slice storage");

        // Copy materialized head shard rows into reconstructed_head
        const auto& head_part = bound[head_id.index].view.parts.at(0);
        const auto* src       = reinterpret_cast<const std::uint16_t*>(
            static_cast<const std::byte*>(head_part.parent->data) + head_part.begin * 2);
        std::memcpy(reconstructed_head.data() + vocab.begin * WeightFixture::kHidden, src,
                    vocab.size * WeightFixture::kHidden * sizeof(std::uint16_t));
    }

    for (std::uint32_t row = 0; row < WeightFixture::kVocab; ++row) {
        for (std::uint32_t col = 0; col < WeightFixture::kHidden; ++col) {
            const auto expected = static_cast<std::uint16_t>((row * 17U + col) & 0x7fffU);
            require(reconstructed_head[row * WeightFixture::kHidden + col] == expected,
                    "reconstructed 128-aligned head matrix row/col mismatch");
        }
    }
}

void test_row_split_expert_selection() {
    constexpr std::uint32_t expert_count = 4, hidden = 128, intermediate = 96;
    constexpr std::array<std::uint64_t, 2> rows{2 * intermediate, hidden};
    constexpr std::array<std::uint64_t, 2> columns{hidden, intermediate};
    constexpr std::array<std::uint64_t, 3> row_bytes{64, 16, 4}; // Two K64 groups per row.
    const auto align_plane   = [](std::uint64_t bytes) { return (bytes + 255) / 256 * 256; };
    const auto plane_offsets = [&](std::uint64_t row_count) {
        const auto high = align_plane(row_count * row_bytes[0]);
        return std::array{std::uint64_t{0}, high, high + align_plane(row_count * row_bytes[1])};
    };

    Fixture fixture;
    fixture.payload.clear();
    fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
                    {"objects", Json::array()},
                    {"bindings", Json::object()},
                    {"uses", Json::array()}};
    const std::array names{"gate-up", "down"};
    std::array<std::uint64_t, 2> object_offsets;
    for (std::size_t matrix = 0; matrix < names.size(); ++matrix) {
        const auto total_rows  = expert_count * rows[matrix];
        const auto offsets     = plane_offsets(total_rows);
        const auto bytes       = offsets.back() + total_rows * row_bytes.back();
        const auto base        = align_plane(fixture.payload.size());
        object_offsets[matrix] = base;
        fixture.payload.resize(base + bytes);
        fixture.root["objects"].push_back({{"id", names[matrix]},
                                           {"kind", "tensor"},
                                           {"shape", {total_rows, columns[matrix]}},
                                           {"format", "q5_g64_fp16"},
                                           {"layout", "row_split_k128_v1"},
                                           {"offset", base},
                                           {"bytes", bytes}});
        for (std::size_t plane = 0; plane < row_bytes.size(); ++plane) {
            for (std::uint64_t row = 0; row < total_rows; ++row) {
                for (std::uint64_t i = 0; i < row_bytes[plane]; ++i) {
                    fixture.payload[base + offsets[plane] + row * row_bytes[plane] + i] =
                        std::byte((29 * matrix + 67 * plane + 13 * row + row / 128 + i) & 255);
                }
            }
        }
    }
    fixture.root["files"] =
        Json::array({{{"path", nullptr}, {"payload_bytes", fixture.payload.size()}}});
    fixture.write();

    const artifact::Reader reader(fixture.entry);
    const std::array objects{reader.find(names[0]), reader.find(names[1])};
    const auto source = [&](std::size_t matrix) -> moe::MatrixSource {
        return {objects[matrix],
                describe_weight_bank(reader.geometry(objects[matrix]),
                                     {0, expert_count, rows[matrix], rows[matrix]})};
    };
    ExpertWeightPartition partition{.bank      = {source(0), source(1)},
                                    .partition = {.intermediate = {0, intermediate}}};
    const auto cpu = Device::cpu();
    for (const auto& ids : {std::vector<std::uint32_t>{0, 1, 2, 3}, {1, 2}}) {
        partition.partition.ids = ids;
        const auto selection =
            select_weights(reader, {}, {}, {}, {}, std::span(&partition, 1), {}, cpu);
        const auto& selected = selection.experts.at(0);
        auto plan            = artifact::plan_materialization(reader, selection.requests);
        const auto requests  = plan.requests;
        const auto backing   = artifact::materialize(reader, std::move(plan));
        const auto verify    = [&](std::size_t request, std::size_t matrix, std::uint32_t first,
                                std::uint32_t count, bool packed) {
            const auto n       = rows[matrix];
            const auto k       = columns[matrix];
            const auto src     = plane_offsets(expert_count * n);
            const auto dst     = plane_offsets(count * n);
            const auto& tensor = selection.requests.at(request).tensor;
            const auto shape   = packed ? artifact::Shape{count, n, k} : artifact::Shape{n, k};
            require(tensor.geometry.shape == shape &&
                           tensor.geometry.layout == QuantLayout::RowSplit &&
                           tensor.geometry.format == QType::Q5_G64_FP16 && tensor.copies.size() == 3,
                       "Q5 expert selection must emit one bulk copy per plane");
            const auto& parent = backing.parent(requests.at(request), cpu);
            const auto layout  = describe_weight_bank(parent.geometry, {0, count, n, n});
            const auto bank    = bind_weight_bank(parent, layout);
            const std::array planes{layout.codes, layout.high, layout.scales};
            for (std::size_t p = 0; p < planes.size(); ++p) {
                const auto stride = n * row_bytes[p];
                const auto begin  = src[p] + first * stride;
                require(tensor.copies[p] ==
                               artifact::EncodedCopy{objects[matrix], begin, dst[p], count * stride},
                           "Q5 expert plane copy has the wrong byte range");
                for (std::uint32_t slot = 0; slot < count; ++slot) {
                    const auto actual = bank.plane(planes[p], slot);
                    const auto expected =
                        fixture.payload.data() + object_offsets[matrix] + begin + slot * stride;
                    const auto* data = static_cast<const std::byte*>(actual.data);
                    require(actual.bytes == stride && std::equal(data, data + stride, expected),
                               "Q5 expert plane bytes differ from the source");
                }
            }
            const auto last = matrix_weight(bank, count - 1);
            require(last.qdata == bank.plane(layout.codes, count - 1).data &&
                           last.qhigh == bank.plane(layout.high, count - 1).data &&
                           last.scales == bank.plane(layout.scales, count - 1).data &&
                           weight_scale_offset(parent.geometry, count * n - 1, 1) ==
                               dst[2] + (count * n - 1) * row_bytes[2] + 2,
                       "Q5 native matrix or scale offset lost the flattened expert row");
        };
        if (ids.size() == expert_count) {
            require(selected.packed && selected.packed->logical_ids.empty() &&
                        selection.requests.size() == 2,
                    "complete Q5 expert bank was not packed");
            verify(selected.packed->gate_up, 0, 0, expert_count, true);
            verify(selected.packed->down, 1, 0, expert_count, true);
        } else {
            require(!selected.packed && selected.entries.size() == ids.size(),
                    "Q5 subset lost indexed expert addressing");
            for (std::size_t slot = 0; slot < ids.size(); ++slot) {
                const auto& entry = selected.entries[slot];
                require(entry.expert == ids[slot], "Q5 subset lost a logical expert ID");
                verify(entry.gate_up, 0, entry.expert, 1, false);
                verify(entry.down, 1, entry.expert, 1, false);
            }
        }
    }
}

// Reconstruct both halves of two non-adjacent experts from their encoded parents.
// FP8 uses a 64-column cut inside an original 128x128 scale block.
void test_mixed_expert_selection(bool nvfp4, std::optional<CudaExecutionView> execution = {}) {
    const Device target           = execution ? execution->device() : Device::cpu();
    constexpr std::uint32_t count = 8, hidden = 128;
    const std::uint32_t intermediate = nvfp4 ? 256 : 128;
    const auto format                = nvfp4 ? QType::NVFP4 : QType::FP8_E4M3FN_BLOCK128X128_BF16;
    const auto layout =
        nvfp4 ? QuantLayout::BatchedBlockScaleK16M128x4 : QuantLayout::BlockScaleM128K128;
    const std::array shapes{artifact::Shape{count, 2 * intermediate, hidden},
                            artifact::Shape{count, hidden, intermediate}};
    const std::array names{"gu", "down"};
    Fixture f;
    f.payload.clear();
    f.root = {{"components", {{"text", {{"config", Json::object()}}}}},
              {"objects", Json::array()},
              {"bindings", Json::object()},
              {"uses", Json::array()}};
    std::array<std::size_t, 2> bases;
    for (unsigned m = 0; m < 2; ++m) {
        const auto g    = weight_geometry(format, layout, shapes[m]);
        const auto base = (f.payload.size() + 255) / 256 * 256;
        bases[m]        = base;
        f.payload.resize(base + g.bytes);
        for (std::size_t i = 0; i < g.bytes; ++i)
            f.payload[base + i] = std::byte((i * 37 + i / 73 + m * 13) % 251);
        for (std::size_t e = 0; e < g.matrix_scale_count; ++e)
            put_word(f.payload, base + g.matrix_scale_offset + 4 * e,
                     std::bit_cast<std::uint32_t>(float(e + 1 + m * count)), 4);
        f.root["objects"].push_back(
            {{"id", names[m]},
             {"kind", "tensor"},
             {"shape", shapes[m]},
             {"format", nvfp4 ? "nvfp4" : "fp8_e4m3fn_block128x128_bf16"},
             {"layout", nvfp4 ? "batched_block_scale_k16_m128x4_v1" : "block_scale_m128_k128_v1"},
             {"offset", base},
             {"bytes", g.bytes}});
    }
    f.root["files"] = Json::array({{{"path", nullptr}, {"payload_bytes", f.payload.size()}}});
    f.write();
    artifact::Reader reader(f.entry);
    const auto source = [&](unsigned m) -> moe::MatrixSource {
        const auto obj  = reader.find(names[m]);
        const auto rows = shapes[m][1];
        return {obj, describe_weight_bank(reader.geometry(obj), {0, count, rows, rows})};
    };
    const auto width = intermediate / 2;
    const std::vector<std::uint32_t> ids{1, 6};
    for (const auto begin : {0U, width}) {
        const ExpertWeightPartition part{{source(0), source(1)}, {}, {ids, {begin, width}}};
        const auto selected = select_weights(reader, {}, {}, {}, {}, {&part, 1}, {}, target);
        const auto& bank    = selected.experts.at(0).packed.value();
        require(bank.logical_ids == ids && selected.expert_descriptions[0].count == count,
                "compact slots or global routing domain differ");
        auto plan           = artifact::plan_materialization(reader, selected.requests);
        const auto requests = plan.requests;
        const auto expected_bytes =
            execution ? plan.device_capacity_bytes : plan.host_capacity_bytes;
        const auto parent_count = plan.tensors.size();
        auto storage            = artifact::materialize(reader, std::move(plan), execution);
        const auto& stats       = storage.stats();
        require((execution ? stats.device_capacity_bytes : stats.host_capacity_bytes) ==
                    expected_bytes,
                "mixed bank backing was misbudgeted");
        if (execution) {
            std::vector<WeightParent> parents;
            for (std::size_t i = 0; i < parent_count; ++i)
                parents.push_back(storage.parent({i}, target));
            const auto resident = bind_selected_experts(selected, requests, parents, target);
            const auto& view    = resident.front().view();
            require(view.count == count && view.gate_up.physical_count == ids.size() &&
                        view.gate_up.expert_indices &&
                        view.down.expert_indices == view.gate_up.expert_indices,
                    "device bank lost the distinction between logical experts and physical slots");
            std::array<std::uint32_t, count> map;
            std::array<std::int32_t, count> owned;
            require(cudaMemcpy(map.data(), view.gate_up.expert_indices, sizeof(map),
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(owned.data(), resident.front().subset().mask.data, sizeof(owned),
                                   cudaMemcpyDeviceToHost) == cudaSuccess,
                    "could not read device expert metadata");
            for (std::uint32_t e = 0; e < count; ++e) {
                const auto it = std::ranges::find(ids, e);
                require(map[e] == (it == ids.end()
                                       ? UINT32_MAX
                                       : static_cast<std::uint32_t>(it - ids.begin())) &&
                            owned[e] == (it != ids.end()),
                        "incorrect mixed bank slot or ownership mask");
            }
        }
        for (unsigned m = 0; m < 2; ++m) {
            const auto& parent = storage.parent(requests.at(m ? bank.down : bank.gate_up), target);
            const auto& dst    = parent.geometry;
            std::vector<std::byte> download;
            const auto* words = parent.data;
            if (execution) {
                download.resize(dst.bytes);
                require(cudaMemcpy(download.data(), words, download.size(),
                                   cudaMemcpyDeviceToHost) == cudaSuccess,
                        "could not read mixed expert payload");
                words = download.data();
            }
            const auto& src = reader.geometry(reader.find(names[m]));
            const auto rows = m ? hidden : 2 * width;
            const auto cols = m ? width : hidden;
            require(dst.shape == artifact::Shape{ids.size(), rows, cols} &&
                        (nvfp4 || dst.format == QType::FP8_E4M3FN_BLOCK64X64_BF16),
                    "mixed bank geometry differs");
            const auto packing   = nvfp4 ? 2U : 1U;
            const auto* original = f.payload.data() + bases[m];
            for (std::size_t slot = 0; slot < ids.size(); ++slot) {
                const auto expert = ids[slot];
                if (nvfp4)
                    require(parent.matrix_scales[slot] == float(expert + 1 + m * count),
                            "matrix divisor no longer belongs to the selected expert");
                for (std::uint32_t row = 0; row < rows; ++row) {
                    const auto dr = slot * rows + row;
                    const auto sr = expert * shapes[m][1] +
                                    (m ? row : begin + row % width + (row / width) * intermediate);
                    const auto sc = m ? begin : 0U;
                    require(std::memcmp(words + dr * dst.code_bytes_per_row,
                                        original + sr * src.code_bytes_per_row + sc / packing,
                                        cols / packing) == 0,
                            "mixed expert code row differs");
                    for (std::uint32_t col = 0; col < cols; col += dst.group_size) {
                        const auto target_scale =
                            weight_scale_offset(dst, dr, col / dst.group_size);
                        const auto source_scale =
                            weight_scale_offset(src, sr, (col + sc) / src.group_size);
                        require(std::memcmp(words + target_scale, original + source_scale,
                                            dst.scale_word_bytes) == 0,
                                "mixed expert scale differs");
                    }
                }
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            require(argc == 2 && std::string_view(argv[1]) == "--gpu", "unknown test mode");
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
            DeviceContext device;
            test_mixed_expert_selection(true, device.transfer_view());
            test_mixed_expert_selection(false, device.transfer_view());
            std::cout << "PASS mixed expert device bindings\n";
            return 0;
        }
        test_vocabulary_partition_laws();
        test_tristate_selection_omission_and_head_reconstruction();
        test_row_split_expert_selection();
        test_mixed_expert_selection(true);
        test_mixed_expert_selection(false);
        std::cout << "PASS: parallel weight selection and RowSplit expert banks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
