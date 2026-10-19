#include "artifact/encoded_slice.h"
#include "artifact/fixture.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <bit>
#include <cstring>

namespace ninfer::test {
namespace {
using namespace artifact;
using namespace artifact_fixture;

void selection(QType format, QuantLayout layout, const char* name, const char* encoded_layout,
               std::optional<CudaExecutionView> execution) {
    Fixture f;
    const bool nv     = format == QType::NVFP4;
    const bool bank   = layout == QuantLayout::BatchedBlockScaleK16M128x4;
    const Shape shape = bank ? Shape{2, 384, 256} : Shape{384, 256};
    const auto g      = weight_geometry(format, layout, shape);
    f.payload.resize(g.bytes);
    for (std::size_t i = 0; i < f.payload.size(); ++i)
        f.payload[i] = std::byte((i * 37 + i / 73) % 251);
    for (std::uint64_t i = 0; i < g.divisor_count; ++i)
        put_word(f.payload, g.divisor_offset + i * 4, std::bit_cast<std::uint32_t>(2.5F + i), 4);
    f.root["components"] = {{"text", {{"config", Json::object()}}}};
    f.root["objects"]    = Json::array({{{"id", "source"},
                                         {"kind", "tensor"},
                                         {"shape", shape},
                                         {"format", name},
                                         {"layout", encoded_layout},
                                         {"offset", 0},
                                         {"bytes", g.bytes}}});
    f.root["bindings"]   = {{"source", {{"object", "source"}}}};
    f.root["uses"]       = Json::array();
    f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", 19001}},
                                        {{"path", "second.bin"}, {"payload_bytes", g.bytes - 19001}}});
    f.write();
    Reader reader(f.entry);
    const auto object = reader.find("source");
    const auto origin = bank ? 384U : 0U;
    const std::array slices{MatrixSlice{object, origin + 256, 128, 128, 128, 0, 0},
                            MatrixSlice{object, origin, 128, 128, 128, 128, 0}};
    const Shape target_shape{256, 128};
    const auto tensor = slice_tensor(reader, target_shape, slices);
    std::vector<TensorRequest> requests{{tensor, Device::cpu()},
                                        {tensor, Device::cpu()},
                                        {whole_tensor(reader, object), Device::cpu()},
                                        {tensor, Device::cpu(), HostAccess::CpuOnly, 7}};
    if (execution) requests.push_back({tensor, execution->device()});
    auto plan     = plan_materialization(reader, requests);
    const auto id = plan.requests[0];
    require(plan.requests[1] == id && plan.requests[3] != id &&
                (!execution || plan.requests[4] == id),
            "selection storage deduplication");
    auto backing       = materialize(reader, std::move(plan), execution);
    const auto& parent = backing.parent(id, Device::cpu());
    const auto& d      = parent.geometry;
    const auto equal   = [&](std::uint64_t destination, std::uint64_t source, std::uint64_t bytes) {
        require(std::memcmp(parent.data + destination, f.payload.data() + source, bytes) == 0,
                  "selected encoded words differ from source coordinates");
    };
    for (std::uint64_t row = 0; row < 256; ++row) {
        const auto src = origin + (row < 128 ? row + 256 : row - 128);
        if (layout == QuantLayout::Contiguous) {
            const auto word = g.bytes / g.elements;
            equal(row * 128 * word, (src * 256 + 128) * word, 128 * word);
            continue;
        }
        equal(row * d.code_bytes_per_row, src * g.code_bytes_per_row + g.code_bytes_per_row / 2,
              d.code_bytes_per_row);
        if (g.high_bytes)
            equal(d.high_offset + row * d.high_bytes_per_row,
                  g.high_offset + src * g.high_bytes_per_row + g.high_bytes_per_row / 2,
                  d.high_bytes_per_row);
        if (nv) {
            // Independent encoding equation: 128x4 scale tiles, row-major tile ordering.
            for (std::uint64_t group = 0; group < 8; ++group) {
                const auto offset = [](std::uint64_t r, std::uint64_t k, std::uint64_t groups) {
                    return ((r / 128) * (groups / 4) + k / 4) * 512 + (r % 32) * 16 +
                           ((r % 128) / 32) * 4 + k % 4;
                };
                equal(d.scale_offset + offset(row, group, 8),
                      g.scale_offset + offset(src, group + 8, 16), 1);
            }
        } else if (layout == QuantLayout::RowScale) {
            equal(d.scale_offset + row * 2, g.scale_offset + src * 2, 2);
        } else {
            const auto block = layout == QuantLayout::BlockScaleM128K128 ? 128 : 1;
            equal(d.scale_offset + row / block * d.scale_row_bytes,
                  g.scale_offset + src / block * g.scale_row_bytes + g.scale_row_bytes / 2,
                  d.scale_row_bytes);
        }
    }
    if (nv) {
        equal(d.divisor_offset, g.divisor_offset + (bank ? 4 : 0), 4);
        require(parent.weight_divisors[0] == (bank ? 3.5F : 2.5F), "selected divisor metadata");
    }
    if (execution) {
        std::vector<std::byte> actual(d.bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), backing.parent(id, execution->device()).data,
                              actual.size(), cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), parent.data, d.bytes) == 0,
                "selected GPU/CPU encoded bytes differ");
    }
    auto invalid               = slices;
    invalid[1].destination_row = 0;
    rejects([&] { (void)slice_tensor(reader, target_shape, invalid); },
            "overlapping destination accepted");
}

void assembly_and_padding(std::optional<CudaExecutionView> execution) {
    Fixture f;
    f.payload.assign(192, std::byte{});
    for (std::uint64_t i = 0; i < 96; ++i) put_word(f.payload, i * 2, i, 2);
    f.root["components"] = {{"text", {{"config", Json::object()}}}};
    f.root["objects"]    = Json::array({{{"id", "conv"},
                                         {"kind", "tensor"},
                                         {"shape", {4, 24}},
                                         {"format", "bf16"},
                                         {"layout", "contiguous_le_v1"},
                                         {"offset", 0},
                                         {"bytes", 192}}});
    f.root["bindings"]   = {{"conv", {{"object", "conv"}}}};
    f.root["uses"]       = Json::array();
    f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", 77}},
                                        {{"path", "second.bin"}, {"payload_bytes", 115}}});
    f.write();
    Reader reader(f.entry);
    const auto object = reader.find("conv");
    const std::array slices{MatrixSlice{object, 0, 2, 4, 2, 0, 0},
                            MatrixSlice{object, 0, 6, 4, 2, 0, 2},
                            MatrixSlice{object, 0, 16, 4, 8, 0, 4}};
    const auto selected = slice_tensor(reader, Shape{4, 12}, slices);
    const auto device   = execution ? execution->device() : Device::cpu();
    const std::array requests{TensorRequest{selected, device}};
    auto plan     = plan_materialization(reader, requests);
    const auto id = plan.requests[0];
    auto backing  = materialize(reader, std::move(plan), execution);
    std::array<std::uint16_t, 48> words{};
    const auto* source = backing.parent(id, device).data;
    if (execution)
        CUDA_CHECK(cudaMemcpy(words.data(), source, sizeof(words), cudaMemcpyDeviceToHost));
    else
        std::memcpy(words.data(), source, sizeof(words));
    const std::array channels{2, 3, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    for (unsigned tap = 0; tap < 4; ++tap)
        for (unsigned channel = 0; channel < 12; ++channel)
            require(words[tap * 12 + channel] == tap * 24 + channels[channel],
                    "tap-major assembly changed channel order");

    Fixture padded;
    std::fill(padded.payload.begin() + 256, padded.payload.begin() + 784, std::byte{0x5a});
    padded.write();
    Reader qr(padded.entry);
    const MatrixSlice part{qr.find("q5"), 0, 64, 2, 64};
    const auto target = slice_tensor(qr, Shape{2, 64}, std::span(&part, 1));
    const std::array request{TensorRequest{target, Device::cpu()}};
    auto qp            = plan_materialization(qr, request);
    const auto qid     = qp.requests[0];
    auto q             = materialize(qr, std::move(qp));
    const auto& parent = q.parent(qid, Device::cpu());
    const auto& g      = parent.geometry;
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (const auto [base, stride, width] :
             {std::array<std::uint64_t, 3>{0, g.code_bytes_per_row, 32},
              std::array<std::uint64_t, 3>{g.high_offset, g.high_bytes_per_row, 8},
              std::array<std::uint64_t, 3>{g.scale_offset, g.scale_row_bytes, 2}}) {
            for (std::uint64_t i = 0; i < stride; ++i)
                require(parent.data[base + row * stride + i] ==
                            (i < width ? std::byte{0x5a} : std::byte{}),
                        "new quantized padding retained source words");
        }
    }
}
} // namespace

void encoded_slice_checks(std::optional<CudaExecutionView> execution) {
    assembly_and_padding(execution);
    selection(QType::BF16, QuantLayout::Contiguous, "bf16", "contiguous_le_v1", execution);
    selection(QType::Q4_G64_FP16, QuantLayout::RowSplit, "q4_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q5_G64_FP16, QuantLayout::RowSplit, "q5_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q6_G64_FP16, QuantLayout::RowSplit, "q6_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q8_G32_FP16, QuantLayout::RowSplit, "q8_g32_fp16", "row_split_k128_v1",
              execution);
    selection(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, "fp8_e4m3fn_row_bf16",
              "row_scale_v1", execution);
    selection(QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128,
              "fp8_e4m3fn_block128x128_bf16", "block_scale_m128_k128_v1", execution);
    selection(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, "nvfp4", "block_scale_k16_m128x4_v1",
              execution);
    selection(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4, "nvfp4",
              "batched_block_scale_k16_m128x4_v1", execution);
}
} // namespace ninfer::test
