#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/fixture.h"
#include "core/weight_view.h"
#include "core/nf4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void file_set_and_bindings() {
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto bytes = reader.read_object(reader.find("q5"));
    require(bytes.size() == 528 &&
                std::equal(bytes.begin(), bytes.end(), fixture.payload.begin() + 256),
            "cross-file parent bytes changed");
    require(reader.read_range(1344, 0).empty(), "empty terminal range failed");
    rejects([&] { (void)reader.read_range(1344, 1); }, "out-of-range read accepted");
    rejects([&] { (void)reader.geometry(reader.find("unused")); },
            "unknown selected encoding accepted");
    rejects([&] { (void)reader.read_range(1280, 64); }, "required missing part accepted");

    Binder binder(reader);
    const auto full = binder.parameter("matrix", {2, 130});
    const auto row  = binder.parameter("row", {1, 130});
    require(full.binding.parts[0].object == row.binding.parts[0].object, "shared parent lost");
    rejects([&] { (void)binder.parameter("matrix", {1, 260}); },
            "whole-object shape was silently reshaped");
    const auto reshaped = binder.parameter("row", {130});
    require(reshaped.binding.elements == 130, "Part reshape changed logical elements");
    const auto context =
        binder.values(binder.use("row", "context").auxiliaries.at("input_divisor"));
    const auto query = binder.values(binder.use("row", "query").auxiliaries.at("input_divisor"));
    require(std::bit_cast<std::uint32_t>(query.scalar_f32()) == 0x40000000 &&
                std::bit_cast<std::uint32_t>(context.scalar_f32()) == 0x40400000,
            "Use values were mixed");
    require(binder.use("row", "query").activation_policy == ActivationPolicy::AllowA4 &&
                binder.use("row", "context").activation_policy == ActivationPolicy::AllowA8,
            "Use policy changed");
    (void)binder.resource("text", "tokenizer.json");
    const std::array requests{PlacementRequest{full.binding.parts[0].object, Device::cuda(0)},
                              PlacementRequest{row.binding.parts[0].object, Device::cuda(0)}};
    auto plan = plan_materialization(reader, requests, std::move(binder).finish());
    require(plan.device_objects.size() == 1 && plan.device_capacity_bytes == 528 &&
                plan.resources.objects.size() == 1,
            "selected parent deduplication or residency failed");
}

void geometry_and_views() {
    const std::array<std::uint16_t, 12> direct_words{
        0, 0x8000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120};
    const auto* direct_bytes = reinterpret_cast<const std::byte*>(direct_words.data());
    const WeightParent direct_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                                     std::array<std::uint64_t, 3>{2, 2, 3}),
                                     Device::cpu(), direct_bytes};
    const WeightView direct_view{{2, 3}, {{&direct_parent, 3, 6}, {&direct_parent, 6, 9}}};
    const auto direct = native_weight(direct_view);
    require(direct.n == 2 && direct.k == 3 && direct.padded_shape[1] == 3 &&
                direct.qdata == direct_bytes + 6 && direct.payload == direct_bytes &&
                direct.payload_bytes == sizeof(direct_words),
            "Direct reshape lost its logical shape, element offset or owning parent");
    Fixture fixture;
    WeightParent parent{weight_geometry(QType::Q5_G64_FP16, QuantLayout::RowSplit,
                                        std::array<std::uint64_t, 2>{2, 130}),
                        Device::cpu(), fixture.payload.data() + 256};
    const WeightView row{{1, 130}, {{&parent, 130, 260}}};
    const auto planes = weight_row_planes(row.parts.front());
    require(parent.geometry.bytes == 528 && planes.codes == parent.data + 128 &&
                planes.high == parent.data + 288 && planes.scales == parent.data + 520,
            "Q5 view was interpreted as a standalone payload");
    const auto weight = native_weight(row);
    rejects<std::invalid_argument>([&] { (void)native_weight(WeightView{{2, 65}, row.parts}); },
                                   "quantized view changed K without preserving encoding groups");
    require(weight.qdata == planes.codes && weight.qhigh == planes.high &&
                weight.scales == planes.scales,
            "native RowSplit view lost its planes");
    require(std::to_integer<int>(planes.codes[0]) == 0x52 &&
                std::to_integer<int>(planes.scales[1]) == 0x40,
            "row view selected the wrong represented values");

    std::vector<std::byte> fp8(16896);
    WeightParent fp8_parent{weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                            std::array<std::uint64_t, 2>{256, 64}),
                            Device::cpu(), fp8.data()};
    WeightView fp8_rows{{128, 64}, {{&fp8_parent, 0, 128 * 64}}};
    require(weight_row_planes(fp8_rows.parts.front()).scales == fp8.data() + 16384,
            "FP8 view recomputed the parent scale base");
    rejects<std::invalid_argument>([&] { (void)native_weight(fp8_rows); },
                                   "complete-parent ABI accepted an FP8 submatrix");

    std::vector<std::byte> nvfp4(4612);
    const std::array divisors{2.0F};
    WeightParent nv_parent{weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                           std::array<std::uint64_t, 2>{128, 64}),
                           Device::cpu(), nvfp4.data(), divisors};
    require(weight_scale_offset(nv_parent.geometry, 31, 0) == 4096 + 496 &&
                weight_scale_offset(nv_parent.geometry, 32, 0) == 4096 + 4,
            "NVFP4 parent swizzle origin was lost");
    const auto nv_rows = weight_row_planes({&nv_parent, 31 * 64, 33 * 64});
    require(nv_rows.swizzled_scales && nv_rows.row_begin == 31 &&
                nv_rows.scales == nvfp4.data() + 4096,
            "NVFP4 submatrix lost its parent coordinates");
    const WeightView complete{{128, 64}, {{&nv_parent, 0, 128 * 64}}};
    const auto query   = native_weight(complete, 3.0F);
    const auto context = native_weight(complete, 4.0F);
    require(query.weight_scale_divisor == 2 && query.input_scale_divisor == 3 &&
                context.input_scale_divisor == 4 && nv_parent.weight_divisors[0] == 2,
            "per-use native parameters changed the parent");

    const auto paged = weight_geometry(QType::NF4_G16_E4M3FN, QuantLayout::RowPagedG16_4K,
                                       std::array<std::uint64_t, 2>{46, 160});
    require(paged.alignment == 4096 && paged.row_bytes == 90 && paged.rows_per_page == 45 &&
                paged.page_count == 2 && paged.divisor_offset == 8192 && paged.bytes == 8196,
            "NF4 paged geometry differs from its wire representation");
    require(weight_paged_row_offset(paged, 44) == 3960 &&
                weight_paged_row_offset(paged, 45) == 4096 &&
                weight_scale_offset(paged, 45, 9) == 4185,
            "NF4 lookup crossed a page or scale boundary incorrectly");
    require(nf4_code_value(0) == -1 && nf4_code_value(7) == 0 && nf4_code_value(15) == 1 &&
                std::bit_cast<std::uint32_t>(nf4_code_value(8)) == 0x3da2faff,
            "NF4 codebook changed");
    WeightParent paged_parent{paged, Device::cpu(), nvfp4.data(), divisors};
    rejects<std::invalid_argument>([&] { (void)weight_row_planes({&paged_parent, 0, 160}); },
                                   "paged lookup was exposed as constant-stride planes");
    rejects<std::invalid_argument>([&] { (void)weight_paged_row_offset(paged, 46); },
                                   "out-of-bounds paged row accepted");
}

void bank_geometry_and_views() {
    const std::array<std::uint64_t, 3> shape{3, 256, 128};
    const auto geometry =
        weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4, shape);
    require(geometry.code_bytes == 49152 && geometry.scale_offset == 49152 &&
                geometry.scale_bytes == 6144 && geometry.divisor_offset == 55296 &&
                geometry.divisor_count == 3 && geometry.bytes == 55308,
            "NVFP4 bank planes or divisor count differ from the wire layout");
    std::vector<std::byte> payload(geometry.bytes);
    const std::array divisors{2.0F, 3.0F, 5.0F};
    const WeightParent parent{geometry, Device::cpu(), payload.data(), divisors};
    const auto second_up = native_weight({{128, 128}, {{&parent, 384 * 128, 512 * 128}}}, 7);
    require(second_up.qdata == payload.data() + 24576 &&
                second_up.scales == payload.data() + 52224 && second_up.weight_scale_divisor == 3 &&
                second_up.input_scale_divisor == 7 && second_up.payload == payload.data() &&
                second_up.payload_bytes == 55308 &&
                second_up.layout == QuantLayout::BlockScaleK16M128x4,
            "NVFP4 bank slice lost local planes, divisor or backing extent");
    require(weight_scale_offset(geometry, 384 + 32, 5) == 52224 + 512 + 4 + 1,
            "NVFP4 bank scale addressing lost the local tile origin");
    rejects<std::invalid_argument>(
        [&] { (void)native_weight({{128, 128}, {{&parent, 64 * 128, 192 * 128}}}); },
        "NVFP4 accepted a view cutting a scale tile");
    rejects<std::invalid_argument>(
        [&] { (void)native_weight({{256, 128}, {{&parent, 128 * 128, 384 * 128}}}); },
        "NVFP4 merged independent matrix divisor domains");
    rejects<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape); },
        "single-matrix NVFP4 layout accepted a bank");
    auto invalid            = parent;
    invalid.weight_divisors = std::span(divisors).first(1);
    rejects<std::invalid_argument>(
        [&] { (void)native_weight({{256, 128}, {{&invalid, 256 * 128, 512 * 128}}}); },
        "NVFP4 accepted missing bank divisor metadata");

    const auto fp8_geometry =
        weight_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128,
                        std::array<std::uint64_t, 3>{3, 256, 384});
    require(fp8_geometry.code_bytes == 294912 && fp8_geometry.scale_bytes == 36 &&
                fp8_geometry.bytes == 294948 && fp8_geometry.scale_block_rows == 128,
            "block-scaled FP8 bank expanded its scale grid");
    std::vector<std::byte> fp8_payload(fp8_geometry.bytes);
    const WeightParent fp8{fp8_geometry, Device::cpu(), fp8_payload.data()};
    const auto fp8_up = native_weight({{128, 384}, {{&fp8, 384 * 384, 512 * 384}}});
    require(fp8_up.qdata == fp8_payload.data() + 147456 &&
                fp8_up.scales == fp8_payload.data() + 294930 && fp8_up.scale_ne[0] == 3 &&
                fp8_up.scale_ne[1] == 1 && fp8_up.scale_nb[0] == 2 && fp8_up.scale_nb[1] == 6 &&
                weight_scale_offset(fp8_geometry, 383, 2) == 294928,
            "block-scaled FP8 view selected a different expert or block row");
    rejects<std::invalid_argument>(
        [&] { (void)native_weight({{128, 384}, {{&fp8, 1 * 384, 129 * 384}}}); },
        "block-scaled FP8 native view split a scale block");
}

void invalid_directories() {
    Fixture fixture;
    const auto bad = [&](auto mutate) {
        Json root = fixture.root;
        mutate(root);
        rejects([&] { (void)parse_directory(root, "model.ninfer"); }, "invalid directory accepted");
    };
    bad([](Json& root) { root["bindings"]["row"]["parts"][0]["range"] = {129, 261}; });
    bad([](Json& root) { root["uses"].push_back(root["uses"][0]); });
    bad([](Json& root) { root["objects"][1]["offset"] = 4; });
    bad([](Json& root) { root["objects"][2]["id"] = "q5"; });
    bad([](Json& root) { root["files"][1]["path"] = "../other"; });
    bad([](Json& root) { root["files"][1]["payload_bytes"] = 1.5; });
    bad([](Json& root) { root["components"]["text"]["resources"]["tokenizer.json"] = "q5"; });
    rejects([] { (void)parse_json("{\"a\":{\"x\":1,\"x\":2}}", "duplicate"); },
            "duplicate JSON key accepted");
    fixture.write();
    {
        std::fstream file(fixture.entry, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(7);
        file.put(2);
    }
    rejects([&] { Reader reader(fixture.entry); }, "v2 magic accepted");
    fixture.write();
    {
        std::fstream file(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(16);
        file.put(0);
    }
    Reader wrong_part(fixture.entry);
    rejects([&] { (void)wrong_part.read_object(wrong_part.find("q5")); },
            "foreign continuation accepted");
    fixture.write();
    Reader shortened(fixture.entry);
    std::filesystem::resize_file(fixture.entry, std::filesystem::file_size(fixture.entry) - 1);
    rejects([&] { (void)shortened.read_range(499, 1); }, "premature EOF accepted");
}

} // namespace

int main(int argc, char** argv) {
    try {
        file_set_and_bindings();
        geometry_and_views();
        bank_geometry_and_views();
        invalid_directories();
        // Optional production-writer fixture or explicitly selected real artifact.
        if (argc == 2) {
            Reader reader(argv[1]);
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
                reader.validate_object({i});
                bytes =
                    checked_add(bytes, object_bytes(reader.directory().objects[i]), "object bytes");
            }
            std::cout << "v3 objects=" << reader.directory().objects.size() << " bytes=" << bytes
                      << '\n';
        }
        std::cout << "artifact reader and binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
