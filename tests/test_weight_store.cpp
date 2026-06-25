#include "qus/core/device.h"
#include "qus/core/weight_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)

struct TestHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian_tag;
    std::uint32_t header_size;
    std::uint32_t tensor_entry_size;
    std::uint32_t tensor_count;
    std::uint32_t reserved0;
    std::uint64_t tensor_table_offset;
    std::uint64_t payload_offset;
    std::uint64_t file_size;
    char model_id[64];
    std::uint32_t hidden_size;
    std::uint32_t intermediate_size;
    std::uint32_t num_layers;
    std::uint32_t full_attention_layers;
    std::uint32_t gdn_layers;
    std::uint32_t attention_heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t gdn_key_heads;
    std::uint32_t gdn_value_heads;
    std::uint32_t gdn_value_head_dim;
    std::uint32_t gdn_key_head_dim;
    std::uint32_t gdn_conv_width;
    std::uint32_t vocab_size;
    std::uint32_t max_position_embeddings;
    std::uint8_t reserved1[76];
};

struct TestEntry {
    char name[96];
    char role[48];
    std::int32_t layer;
    std::uint8_t kind;
    std::uint8_t dtype;
    std::uint8_t quant_layout;
    std::uint8_t rank;
    std::uint32_t shape[4];
    std::uint64_t data_offset;
    std::uint64_t data_nbytes;
    std::uint8_t scale_dtype;
    std::uint8_t scale_rank;
    std::uint16_t reserved0;
    std::uint32_t scale_shape[4];
    std::uint64_t scale_offset;
    std::uint64_t scale_nbytes;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t group;
    std::uint8_t reserved1[24];
};

#pragma pack(pop)

static_assert(sizeof(TestHeader) == 256);
static_assert(sizeof(TestEntry) == 256);

constexpr std::uint64_t kHeaderSize    = 256;
constexpr std::uint64_t kEntrySize     = 256;
constexpr std::uint64_t kPayloadOffset = 768;
constexpr std::uint64_t kDenseOffset   = 768;
constexpr std::uint64_t kQDataOffset   = 1024;
constexpr std::uint64_t kScaleOffset   = 1280;
constexpr std::uint64_t kFileSize      = 1296;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

void set_fixed(char* dest, std::size_t size, const char* text) {
    std::memset(dest, 0, size);
    std::memcpy(dest, text, std::min(size - 1, std::strlen(text)));
}

TestHeader valid_header() {
    TestHeader h{};
    std::memcpy(h.magic, "QUSWGT01", 8);
    h.version             = 1;
    h.endian_tag          = 0x01020304U;
    h.header_size         = kHeaderSize;
    h.tensor_entry_size   = kEntrySize;
    h.tensor_count        = 2;
    h.tensor_table_offset = kHeaderSize;
    h.payload_offset      = kPayloadOffset;
    h.file_size           = kFileSize;
    set_fixed(h.model_id, sizeof(h.model_id), "tiny-qwen-fixture");
    h.hidden_size             = 4;
    h.intermediate_size       = 8;
    h.num_layers              = 1;
    h.full_attention_layers   = 1;
    h.gdn_layers              = 0;
    h.attention_heads         = 1;
    h.kv_heads                = 1;
    h.head_dim                = 4;
    h.gdn_key_heads           = 0;
    h.gdn_value_heads         = 0;
    h.gdn_value_head_dim      = 0;
    h.gdn_key_head_dim        = 0;
    h.gdn_conv_width          = 0;
    h.vocab_size              = 16;
    h.max_position_embeddings = 32;
    return h;
}

TestEntry dense_entry() {
    TestEntry e{};
    set_fixed(e.name, sizeof(e.name), "model.norm.weight");
    set_fixed(e.role, sizeof(e.role), "norm");
    e.layer        = -1;
    e.kind         = 0;
    e.dtype        = 1;
    e.quant_layout = 255;
    e.rank         = 1;
    e.shape[0]     = 4;
    e.shape[1]     = 1;
    e.shape[2]     = 1;
    e.shape[3]     = 1;
    e.data_offset  = kDenseOffset;
    e.data_nbytes  = 16;
    e.scale_dtype  = 255;
    return e;
}

TestEntry quant_entry() {
    TestEntry e{};
    set_fixed(e.name, sizeof(e.name), "layers.0.q_proj.weight");
    set_fixed(e.role, sizeof(e.role), "q_proj");
    e.layer          = 0;
    e.kind           = 1;
    e.dtype          = 3;
    e.quant_layout   = 0;
    e.rank           = 1;
    e.shape[0]       = 8;
    e.shape[1]       = 1;
    e.shape[2]       = 1;
    e.shape[3]       = 1;
    e.data_offset    = kQDataOffset;
    e.data_nbytes    = 8;
    e.scale_dtype    = 1;
    e.scale_rank     = 2;
    e.scale_shape[0] = 2;
    e.scale_shape[1] = 2;
    e.scale_shape[2] = 1;
    e.scale_shape[3] = 1;
    e.scale_offset   = kScaleOffset;
    e.scale_nbytes   = 16;
    e.n              = 2;
    e.k              = 8;
    e.group          = 4;
    return e;
}

template <typename T>
void write_object(std::vector<unsigned char>& file, std::uint64_t offset, const T& object) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&object);
    std::copy(bytes, bytes + sizeof(T), file.begin() + static_cast<std::ptrdiff_t>(offset));
}

std::filesystem::path write_weight_file(
    const std::string& stem,
    const std::function<void(TestHeader&, std::array<TestEntry, 2>&, std::vector<unsigned char>&)>&
        mutate = {}) {
    TestHeader header = valid_header();
    std::array<TestEntry, 2> entries{dense_entry(), quant_entry()};
    std::vector<unsigned char> file(kFileSize, 0);

    const float dense_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(file.data() + kDenseOffset, dense_values, sizeof(dense_values));
    for (int i = 0; i < 8; ++i) { file[kQDataOffset + i] = static_cast<unsigned char>(0xa0 + i); }
    const float scales[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    std::memcpy(file.data() + kScaleOffset, scales, sizeof(scales));

    if (mutate) { mutate(header, entries, file); }

    if (file.size() < header.file_size) {
        file.resize(static_cast<std::size_t>(header.file_size), 0);
    }
    write_object(file, 0, header);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        write_object(file, kHeaderSize + i * kEntrySize, entries[i]);
    }

    const auto path = std::filesystem::temp_directory_path() / (stem + ".qusweights");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(file.data()),
              static_cast<std::streamsize>(file.size()));
    return path;
}

qus::WeightFileExpectations expectations() {
    qus::WeightFileExpectations e;
    e.model_id                = "tiny-qwen-fixture";
    e.hidden_size             = 4;
    e.intermediate_size       = 8;
    e.num_layers              = 1;
    e.full_attention_layers   = 1;
    e.gdn_layers              = 0;
    e.attention_heads         = 1;
    e.kv_heads                = 1;
    e.head_dim                = 4;
    e.vocab_size              = 16;
    e.max_position_embeddings = 32;
    return e;
}

template <typename Exception>
int expect_load_throws(const std::filesystem::path& path, const char* label) {
    qus::DeviceContext ctx(0);
    qus::DeviceArena arena(4096);
    const std::size_t before = arena.used();
    try {
        qus::WeightStore store(expectations());
        store.load(path.c_str(), arena, ctx);
    } catch (const Exception&) {
        if (arena.used() != before) {
            std::cerr << label << " consumed arena before failing\n";
            return 1;
        }
        return 0;
    }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

int expect_bytes(const void* device, const std::vector<unsigned char>& expected,
                 const char* label) {
    std::vector<unsigned char> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
    if (actual != expected) {
        std::cerr << label << " payload mismatch\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures          = 0;
    const auto valid_path = write_weight_file("qus_weight_store_valid");
    qus::DeviceContext ctx(0);
    qus::DeviceArena arena(4096);
    qus::WeightStore store(expectations());
    store.load(valid_path.c_str(), arena, ctx);
    failures += store.dense_count() == 1 ? 0 : fail("dense_count mismatch");
    failures += store.quant_count() == 1 ? 0 : fail("quant_count mismatch");

    const qus::Tensor dense = store.weight("norm", -1);
    failures += dense.dtype == qus::DType::FP32 ? 0 : fail("dense dtype mismatch");
    failures += dense.ne[0] == 4 && dense.ne[1] == 1 ? 0 : fail("dense shape mismatch");
    std::vector<unsigned char> dense_bytes(16);
    const float dense_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(dense_bytes.data(), dense_values, sizeof(dense_values));
    failures += expect_bytes(dense.data, dense_bytes, "dense");

    const qus::QuantWeight qw = store.qweight("q_proj", 0);
    failures += qw.n == 2 && qw.k == 8 && qw.group == 4 ? 0 : fail("quant dims mismatch");
    failures +=
        qw.layout == qus::QuantLayout::W4A16KernelPackedV1 ? 0 : fail("quant layout mismatch");
    failures += qw.scale_dtype == qus::DType::FP32 ? 0 : fail("scale dtype mismatch");
    failures += qw.scale_ne[0] == 2 && qw.scale_ne[1] == 2 ? 0 : fail("scale shape mismatch");
    failures += expect_bytes(qw.qdata, {0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7}, "qdata");
    std::vector<unsigned char> scale_bytes(16);
    const float scales[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    std::memcpy(scale_bytes.data(), scales, sizeof(scales));
    failures += expect_bytes(qw.scales, scale_bytes, "scales");

    try {
        (void)store.weight("missing", -1);
        failures += fail("missing dense lookup did not throw");
    } catch (const std::out_of_range&) {}
    try {
        (void)store.qweight("norm", -1);
        failures += fail("wrong-kind quant lookup did not throw");
    } catch (const std::out_of_range&) {}

    const auto shared_role_path =
        write_weight_file("qus_weight_shared_role", [](auto&, auto& e, auto&) {
            set_fixed(e[1].role, sizeof(e[1].role), "norm");
            e[1].layer = -1;
        });
    qus::DeviceArena shared_arena(4096);
    qus::WeightStore shared_store(expectations());
    shared_store.load(shared_role_path.c_str(), shared_arena, ctx);
    failures += shared_store.dense_count() == 1 ? 0 : fail("shared role dense_count mismatch");
    failures += shared_store.quant_count() == 1 ? 0 : fail("shared role quant_count mismatch");
    failures +=
        shared_store.qweight("norm", -1).k == 8 ? 0 : fail("shared role quant lookup mismatch");

    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_magic", [](auto& h, auto&, auto&) { h.magic[0] = 'X'; }),
        "bad magic");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_version", [](auto& h, auto&, auto&) { h.version = 2; }),
        "bad version");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_endian",
                          [](auto& h, auto&, auto&) { h.endian_tag = 0x04030201U; }),
        "bad endian");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_hidden_mismatch",
                          [](auto& h, auto&, auto&) { h.hidden_size = 5; }),
        "hidden mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_zero_mismatch",
                          [](auto& h, auto&, auto&) { h.gdn_layers = 1; }),
        "zero-valued expectation mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_header_reserved",
                          [](auto& h, auto&, auto&) { h.reserved0 = 1; }),
        "header reserved");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_dtype", [](auto&, auto& e, auto&) { e[0].dtype = 99; }),
        "bad dtype");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_layout",
                          [](auto&, auto& e, auto&) { e[1].quant_layout = 42; }),
        "bad layout");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_bad_rank", [](auto&, auto& e, auto&) { e[0].rank = 5; }),
        "bad rank");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file(
            "qus_weight_unterminated",
            [](auto&, auto& e, auto&) { std::memset(e[0].role, 'x', sizeof(e[0].role)); }),
        "unterminated role");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file(
            "qus_weight_unterminated_name",
            [](auto&, auto& e, auto&) { std::memset(e[0].name, 'y', sizeof(e[0].name)); }),
        "unterminated name");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_entry_reserved",
                          [](auto&, auto& e, auto&) { e[0].reserved1[0] = 1; }),
        "entry reserved");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_unaligned",
                          [](auto&, auto& e, auto&) { e[0].data_offset = kDenseOffset + 1; }),
        "unaligned offset");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_out_of_range",
                          [](auto&, auto& e, auto&) { e[0].data_offset = kFileSize + 256; }),
        "out-of-range offset");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_before_payload",
                          [](auto&, auto& e, auto&) { e[0].data_offset = kHeaderSize; }),
        "payload before payload_offset");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_table_overlap",
                          [](auto& h, auto&, auto&) { h.payload_offset = 512; }),
        "table payload overlap");
    failures += expect_load_throws<std::runtime_error>(write_weight_file("qus_weight_duplicate",
                                                                         [](auto&, auto& e, auto&) {
                                                                             e[1] = e[0];
                                                                             e[1].data_offset =
                                                                                 kQDataOffset;
                                                                         }),
                                                       "duplicate kind role layer");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_overlap",
                          [](auto&, auto& e, auto&) { e[1].data_offset = kDenseOffset; }),
        "overlap");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_missing_scales",
                          [](auto&, auto& e, auto&) {
                              e[1].scale_offset = 0;
                              e[1].scale_nbytes = 0;
                              e[1].scale_rank   = 0;
                              std::fill(std::begin(e[1].scale_shape), std::end(e[1].scale_shape),
                                        0);
                          }),
        "missing scales");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_qbytes", [](auto&, auto& e, auto&) { e[1].data_nbytes = 7; }),
        "qdata byte mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_qshape", [](auto&, auto& e, auto&) { e[1].shape[0] = 7; }),
        "qdata shape byte mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_scale_shape",
                          [](auto&, auto& e, auto&) { e[1].scale_shape[1] = 1; }),
        "scale shape mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_scale_nbytes",
                          [](auto&, auto& e, auto&) { e[1].scale_nbytes = 12; }),
        "scale byte mismatch");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_quant_bound",
                          [](auto&, auto& e, auto&) {
                              e[1].group = static_cast<std::uint32_t>(
                                               std::numeric_limits<std::int32_t>::max()) +
                                           1U;
                              e[1].scale_shape[1] = 1;
                              e[1].scale_nbytes   = 8;
                          }),
        "quant signed bound");
    failures += expect_load_throws<std::runtime_error>(
        write_weight_file("qus_weight_dense_bytes",
                          [](auto&, auto& e, auto&) { e[0].data_nbytes = 12; }),
        "dense byte mismatch");

    qus::DeviceArena small_arena(64);
    const std::size_t before = small_arena.used();
    try {
        qus::WeightStore small_store(expectations());
        small_store.load(valid_path.c_str(), small_arena, ctx);
        failures += fail("small weights arena did not throw");
    } catch (const std::bad_alloc&) {
        failures += small_arena.used() == before ? 0 : fail("small arena was consumed");
    }

    return failures == 0 ? 0 : fail("weight store test failed");
}
