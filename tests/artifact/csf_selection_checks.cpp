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

// Independent CSF reference ordering: 128x4 scale tiles, column pairs before 32-row groups.
template <class Fn>
void blocks(std::uint64_t rows, std::uint64_t columns, Fn&& fn) {
    const auto tile_columns = (columns + 3) / 4;
    std::uint64_t ordinal = 0, packed = 0;
    for (std::uint64_t tr = 0; tr < (rows + 127) / 128; ++tr)
        for (std::uint64_t tc = 0; tc < tile_columns; ++tc)
            for (std::uint64_t pair = 0; pair < 2; ++pair)
                for (std::uint64_t row = 0; row < 4; ++row) {
                    const auto r = tr * 128 + row * 32, c = tc * 4 + pair * 2;
                    if (r >= rows || c >= columns) continue;
                    const auto width = std::min<std::uint64_t>(2, columns - c);
                    fn(ordinal++, (tr * tile_columns + tc) / 32, r, c, width, packed);
                    packed += width * 16;
                }
}

std::uint16_t u16(const std::byte* p) {
    return std::to_integer<std::uint16_t>(p[0]) | (std::to_integer<std::uint16_t>(p[1]) << 8);
}

std::uint32_t u32(const std::byte* p) { return u16(p) | (std::uint32_t(u16(p + 2)) << 16); }

std::vector<std::byte> decode(const WeightGeometry& g, const std::byte* payload) {
    const auto rows = g.shape[g.shape.size() - 2], columns = g.shape.back() / 16;
    std::vector<std::byte> values(g.matrix_scale_count * rows * columns);
    for (std::uint64_t matrix = 0; matrix < g.matrix_scale_count; ++matrix) {
        blocks(
            rows, columns,
            [&](auto ordinal, auto page, auto row, auto column, auto width, auto packed) {
                const auto* nibbles = payload + g.csf.nibbles_offset +
                                      matrix * g.csf.nibble_bytes_per_matrix + packed;
                for (std::uint64_t r = 0; r < 32; ++r)
                    for (std::uint64_t c = 0; c < width; ++c) {
                        const auto index = r * width + c;
                        const auto code  = std::to_integer<unsigned>(nibbles[index / 2]);
                        const auto base  = std::to_integer<unsigned>(
                            payload[g.csf.bases_offset + matrix * rows + row + r]);
                        values[(matrix * rows + row + r) * columns + column + c] =
                            std::byte(base + ((code >> (4 * (index % 2))) & 15));
                    }
                const auto* offsets = payload + g.csf.offsets_offset +
                                      (matrix * g.csf.offsets_per_matrix + ordinal + page) * 2;
                const auto base = u32(payload + g.csf.page_bases_offset +
                                      (matrix * g.csf.pages_per_matrix + page) * 4);
                for (auto index = u16(offsets); index < u16(offsets + 2); ++index) {
                    const auto word     = u16(payload + g.csf.exceptions_offset + base + index * 2);
                    const auto position = word & 255;
                    values[(matrix * rows + row + position / 2) * columns + column + position % 2] =
                        std::byte(word >> 8);
                }
            });
    }
    return values;
}

void check_csf(std::uint64_t rows, std::uint64_t k, std::optional<CudaExecutionView> execution,
               bool no_exceptions = false) {
    constexpr std::uint64_t matrices = 3;
    const Shape shape{matrices, rows, k};
    const auto zero = weight_geometry(QType::NVFP4_MUL, QuantLayout::Csf32, shape, {0});
    std::vector<std::byte> payload(zero.bytes);
    std::vector<std::uint16_t> exceptions;
    for (std::uint64_t matrix = 0; matrix < matrices; ++matrix) {
        put_word(payload, zero.matrix_scale_offset + matrix * 4,
                 std::bit_cast<std::uint32_t>(0.75F + matrix), 4);
        for (std::uint64_t row = 0; row < rows; ++row) {
            payload[zero.csf.bases_offset + matrix * rows + row] = std::byte(32 + row % 7 + matrix);
            for (std::uint64_t column = 0; column < k / 2; ++column)
                payload[(matrix * rows + row) * (k / 2) + column] =
                    std::byte((matrix * 13 + row * 7 + column) % 256);
        }
        std::uint64_t previous_page = UINT64_MAX;
        std::uint16_t relative      = 0;
        blocks(rows, k / 16,
               [&](auto ordinal, auto page, auto row, auto column, auto width, auto packed) {
                   if (page != previous_page) {
                       put_word(payload,
                                zero.csf.page_bases_offset +
                                    (matrix * zero.csf.pages_per_matrix + page) * 4,
                                exceptions.size() * 2, 4);
                       relative      = 0;
                       previous_page = page;
                   }
                   auto* nibbles = payload.data() + zero.csf.nibbles_offset +
                                   matrix * zero.csf.nibble_bytes_per_matrix + packed;
                   for (std::uint64_t index = 0; index < width * 16; ++index)
                       nibbles[index] = std::byte((matrix + row + column + index * 17) % 256);
                   const auto offset = zero.csf.offsets_offset +
                                       (matrix * zero.csf.offsets_per_matrix + ordinal + page) * 2;
                   put_word(payload, offset, relative, 2);
                   const auto count = no_exceptions ? 0 : (matrix + row / 32 + column / 2) % 3;
                   for (std::uint64_t index = 0; index < count; ++index)
                       exceptions.push_back(
                           static_cast<std::uint16_t>((96 + index) << 8 | index * 2));
                   relative += count;
                   put_word(payload, offset + 2, relative, 2);
               });
    }
    const auto geometry =
        weight_geometry(QType::NVFP4_MUL, QuantLayout::Csf32, shape, {exceptions.size()});
    payload.resize(geometry.bytes);
    for (std::size_t i = 0; i < exceptions.size(); ++i)
        put_word(payload, geometry.csf.exceptions_offset + i * 2, exceptions[i], 2);
    const auto expected = decode(geometry, payload.data());
    Fixture fixture;
    fixture.payload            = payload;
    fixture.root["components"] = {{"text", {{"config", Json::object()}}}};
    fixture.root["objects"] =
        Json::array({{{"id", "csf"},
                      {"kind", "tensor"},
                      {"shape", shape},
                      {"format", "nvfp4_mul"},
                      {"layout", "csf_32x2_v1"},
                      {"layout_parameters", {{"exception_count", exceptions.size()}}},
                      {"offset", 0},
                      {"bytes", geometry.bytes}}});
    fixture.root["bindings"] = {{"csf", {{"object", "csf"}}}};
    fixture.root["uses"]     = Json::array();
    const auto split =
        k == 4096 ? geometry.csf.exceptions_offset + 3 : geometry.csf.offsets_offset + 3;
    fixture.root["files"] =
        Json::array({{{"path", nullptr}, {"payload_bytes", split}},
                     {{"path", "part.bin"}, {"payload_bytes", geometry.bytes - split}}});
    fixture.write();
    Reader reader(fixture.entry);
    const auto object      = reader.find("csf");
    const auto columns     = k == 4096 ? 1024U : k;
    const auto column      = k == 4096 ? 1024U : 0U;
    const auto target_rows = k == 4096 ? 256U : 128U;
    const auto half        = target_rows / 2;
    const std::array slices{
        MatrixSlice{object, 2 * rows + rows - half, column, half, columns, 0, 0},
        MatrixSlice{object, 2 * rows, column, half, columns, half, 0},
        MatrixSlice{object, 0, column, target_rows, columns, target_rows, 0}};
    const Shape target{2, target_rows, columns};
    (void)reader.read_range(0, 7);
    const auto preparation_begin = reader.payload_read_bytes();
    const auto selected          = slice_tensor(reader, target, slices);
    require(selected.preparation &&
                selected.preparation->read_bytes == reader.payload_read_bytes() - preparation_begin,
            "CSF preparation included earlier Reader I/O");
    const std::array whole_request{TensorRequest{whole_tensor(reader, object), Device::cpu()}};
    auto whole = materialize(reader, plan_materialization(reader, whole_request));
    require(std::memcmp(whole.host_parent(object).data, payload.data(), payload.size()) == 0,
            "whole CSF materialization changed compressed bytes");
    const auto independently_prepared = slice_tensor(reader, target, slices);
    std::vector<TensorRequest> requests{{selected, Device::cpu()},
                                        {selected, Device::cpu()},
                                        {independently_prepared, Device::cpu()}};
    if (execution) requests.push_back({selected, execution->device()});
    auto plan = plan_materialization(reader, requests);
    require(plan.requests[0] == plan.requests[1], "CSF selection was not deduplicated");
    require(plan.requests[0] == plan.requests[2] &&
                plan.preparation_read_bytes == selected.preparation->read_bytes +
                                                   independently_prepared.preparation->read_bytes,
            "CSF preparation reads were deduplicated with storage or duplicated by placement");
    const auto id      = plan.requests[0];
    auto output        = materialize(reader, std::move(plan), execution);
    const auto& parent = output.parent(id, Device::cpu());
    require(parent.geometry.layout == QuantLayout::Csf32 &&
                parent.geometry.bytes ==
                    parent.geometry.csf.exceptions_offset + parent.geometry.csf.exception_count * 2,
            "CSF materialization expanded or padded its exception arena");
    const auto actual = decode(parent.geometry, parent.data);
    for (const auto& slice : slices)
        for (std::uint64_t row = 0; row < slice.rows; ++row) {
            require(std::memcmp(parent.data + (slice.destination_row + row) * columns / 2,
                                payload.data() + (slice.row + row) * k / 2 + column / 2,
                                columns / 2) == 0,
                    "selected CSF codes differ");
            require(std::memcmp(actual.data() + (slice.destination_row + row) * (columns / 16),
                                expected.data() + (slice.row + row) * (k / 16) + column / 16,
                                columns / 16) == 0,
                    "selected CSF scale bytes differ");
        }
    require(parent.matrix_scales[0] == 2.75F && parent.matrix_scales[1] == 0.75F,
            "CSF matrix multiplier domain changed");
    if (execution) {
        std::vector<std::byte> gpu(parent.geometry.bytes);
        CUDA_CHECK(cudaMemcpy(gpu.data(), output.parent(id, execution->device()).data, gpu.size(),
                              cudaMemcpyDeviceToHost));
        require(std::memcmp(gpu.data(), parent.data, gpu.size()) == 0,
                "CSF Host/device payload differs");
    }
    auto invalid   = slices;
    invalid[1].row = 0;
    rejects([&] { (void)slice_tensor(reader, target, invalid); },
            "CSF accepted mixed matrix multiplier domains");
}
} // namespace

void csf_selection_checks(std::optional<CudaExecutionView> execution) {
    check_csf(256, 4096, execution);
    check_csf(160, 2064, execution);
    check_csf(160, 2064, execution, true);
}
} // namespace ninfer::test
