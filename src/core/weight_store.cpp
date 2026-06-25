#include "qus/core/weight_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace qus {
namespace {

#pragma pack(push, 1)

struct RawHeader {
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

struct RawEntry {
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

static_assert(sizeof(RawHeader) == 256);
static_assert(sizeof(RawEntry) == 256);

constexpr std::uint32_t kVersion         = 1;
constexpr std::uint32_t kEndianTag       = 0x01020304U;
constexpr std::uint32_t kHeaderSize      = 256;
constexpr std::uint32_t kEntrySize       = 256;
constexpr std::uint64_t kPayloadAlign    = 256;
constexpr std::size_t kStagingChunkBytes = 1U << 20;
constexpr std::uint8_t kNoLayout         = 255;
constexpr std::uint8_t kNoScaleDType     = 255;

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
};

struct ParsedEntry {
    std::string name;
    std::string role;
    int layer            = -1;
    WeightEntryKind kind = WeightEntryKind::Dense;
    DType dtype          = DType::BF16;
    QuantLayout layout   = QuantLayout::W4A16KernelPackedV1;
    std::uint8_t rank    = 0;
    std::array<std::int32_t, 4> shape{1, 1, 1, 1};
    std::uint64_t data_offset = 0;
    std::uint64_t data_nbytes = 0;
    DType scale_dtype         = DType::FP32;
    std::uint8_t scale_rank   = 0;
    std::array<std::int32_t, 4> scale_shape{1, 1, 1, 1};
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_nbytes = 0;
    std::uint32_t n            = 0;
    std::uint32_t k            = 0;
    std::uint32_t group        = 0;
};

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void cuda_throw(cudaError_t err, const char* what) {
    if (err != cudaSuccess) { throw std::runtime_error(cuda_error_message(what, err)); }
}

std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::runtime_error("weight file range overflow");
    }
    return a + b;
}

std::size_t checked_add_size(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("weight store size overflow");
    }
    return a + b;
}

std::size_t checked_mul_size(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("weight store size overflow");
    }
    return a * b;
}

std::uint64_t ceil_div_u64(std::uint64_t a, std::uint64_t b) {
    if (b == 0) { throw std::runtime_error("division by zero"); }
    if (a > std::numeric_limits<std::uint64_t>::max() - (b - 1)) {
        throw std::runtime_error("ceil division overflow");
    }
    return (a + b - 1) / b;
}

std::string fixed_string(const char* data, std::size_t size, const char* field) {
    const void* nul = std::memchr(data, '\0', size);
    if (nul == nullptr) { throw std::runtime_error(std::string(field) + " is not NUL-terminated"); }
    return std::string(data, static_cast<const char*>(nul));
}

bool all_zero(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    return std::all_of(bytes, bytes + size, [](std::uint8_t b) { return b == 0; });
}

DType dtype_from_tag(std::uint8_t tag) {
    switch (tag) {
    case 0:
        return DType::BF16;
    case 1:
        return DType::FP32;
    case 2:
        return DType::I32;
    case 3:
        return DType::U8;
    default:
        throw std::runtime_error("invalid dtype tag");
    }
}

WeightEntryKind kind_from_tag(std::uint8_t tag) {
    switch (tag) {
    case 0:
        return WeightEntryKind::Dense;
    case 1:
        return WeightEntryKind::QuantW4;
    default:
        throw std::runtime_error("invalid entry kind");
    }
}

QuantLayout layout_from_tag(std::uint8_t tag) {
    if (tag != 0) { throw std::runtime_error("invalid quant layout"); }
    return QuantLayout::W4A16KernelPackedV1;
}

std::array<std::int32_t, 4> parse_shape(const std::uint32_t raw[4], std::uint8_t rank,
                                        const char* field) {
    if (rank < 1 || rank > 4) { throw std::runtime_error(std::string(field) + " rank is invalid"); }
    std::array<std::int32_t, 4> out{1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
        if (i < rank) {
            if (raw[i] == 0 ||
                raw[i] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error(std::string(field) + " shape is invalid");
            }
            out[i] = static_cast<std::int32_t>(raw[i]);
        } else if (raw[i] != 1) {
            throw std::runtime_error(std::string(field) + " trailing shape dims must be one");
        }
    }
    return out;
}

Tensor make_tensor(void* data, DType dtype, const std::array<std::int32_t, 4>& shape,
                   std::uint8_t rank) {
    switch (rank) {
    case 1:
        return Tensor(data, dtype, {shape[0]});
    case 2:
        return Tensor(data, dtype, {shape[0], shape[1]});
    case 3:
        return Tensor(data, dtype, {shape[0], shape[1], shape[2]});
    case 4:
        return Tensor(data, dtype, {shape[0], shape[1], shape[2], shape[3]});
    default:
        throw std::runtime_error("invalid tensor rank");
    }
}

Tensor alloc_tensor(DeviceArena& arena, DType dtype, const std::array<std::int32_t, 4>& shape,
                    std::uint8_t rank) {
    switch (rank) {
    case 1:
        return arena.alloc(dtype, {shape[0]});
    case 2:
        return arena.alloc(dtype, {shape[0], shape[1]});
    case 3:
        return arena.alloc(dtype, {shape[0], shape[1], shape[2]});
    case 4:
        return arena.alloc(dtype, {shape[0], shape[1], shape[2], shape[3]});
    default:
        throw std::runtime_error("invalid tensor rank");
    }
}

void validate_payload_range(std::uint64_t offset, std::uint64_t nbytes,
                            std::uint64_t payload_offset, std::uint64_t file_size,
                            std::vector<Range>& ranges) {
    if (nbytes == 0) { throw std::runtime_error("payload byte count is zero"); }
    if (offset < payload_offset) {
        throw std::runtime_error("payload begins before payload region");
    }
    if (offset % kPayloadAlign != 0) { throw std::runtime_error("payload offset is not aligned"); }
    if (offset > file_size || nbytes > file_size - offset) {
        throw std::runtime_error("payload range is outside file");
    }
    ranges.push_back(Range{offset, offset + nbytes});
}

void validate_non_overlapping(std::vector<Range> ranges) {
    std::sort(ranges.begin(), ranges.end(),
              [](const Range& a, const Range& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].begin < ranges[i - 1].end) {
            throw std::runtime_error("payload ranges overlap");
        }
    }
}

void validate_expected(std::uint32_t actual, const std::optional<std::uint32_t>& expected,
                       const char* name) {
    if (expected.has_value() && actual != *expected) {
        throw std::runtime_error(std::string("model metadata mismatch: ") + name);
    }
}

std::uint64_t stream_size(std::ifstream& in) {
    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < 0) { throw std::runtime_error("failed to size weight file"); }
    in.seekg(0, std::ios::beg);
    return static_cast<std::uint64_t>(end);
}

RawHeader read_header(std::ifstream& in, std::uint64_t file_size) {
    if (file_size < sizeof(RawHeader)) { throw std::runtime_error("weight file is too small"); }
    RawHeader h{};
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in) { throw std::runtime_error("failed to read weight file header"); }
    return h;
}

std::vector<unsigned char> read_table(std::ifstream& in, const RawHeader& h) {
    const std::size_t table_nbytes =
        checked_mul_size(static_cast<std::size_t>(h.tensor_count), kEntrySize);
    std::vector<unsigned char> table(table_nbytes);
    if (!table.empty()) {
        in.seekg(static_cast<std::streamoff>(h.tensor_table_offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(table.data()), static_cast<std::streamsize>(table.size()));
        if (!in) { throw std::runtime_error("failed to read weight tensor table"); }
    }
    return table;
}

std::ifstream open_weight_stream(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open weight file"); }
    return in;
}

void validate_header(const RawHeader& h, std::uint64_t actual_size,
                     const WeightFileExpectations& expected) {
    if (std::memcmp(h.magic, "QUSWGT01", 8) != 0) {
        throw std::runtime_error("bad weight file magic");
    }
    if (h.version != kVersion) { throw std::runtime_error("unsupported weight file version"); }
    if (h.endian_tag != kEndianTag) { throw std::runtime_error("weight file endian mismatch"); }
    if (h.header_size != kHeaderSize || h.tensor_entry_size != kEntrySize) {
        throw std::runtime_error("weight file structure size mismatch");
    }
    if (h.reserved0 != 0 || !all_zero(h.reserved1, sizeof(h.reserved1))) {
        throw std::runtime_error("weight file reserved header fields are nonzero");
    }
    if (h.file_size != actual_size) { throw std::runtime_error("weight file size mismatch"); }
    const std::uint64_t table_nbytes =
        checked_mul_size(static_cast<std::size_t>(h.tensor_count), kEntrySize);
    const std::uint64_t table_end = checked_add_u64(h.tensor_table_offset, table_nbytes);
    if (h.tensor_table_offset < h.header_size || table_end > h.payload_offset ||
        h.payload_offset > h.file_size || h.payload_offset % kPayloadAlign != 0) {
        throw std::runtime_error("invalid weight file table/payload layout");
    }
    const std::string model_id = fixed_string(h.model_id, sizeof(h.model_id), "model_id");
    if (!expected.model_id.empty() && model_id != expected.model_id) {
        throw std::runtime_error("model id mismatch");
    }
    validate_expected(h.hidden_size, expected.hidden_size, "hidden_size");
    validate_expected(h.intermediate_size, expected.intermediate_size, "intermediate_size");
    validate_expected(h.num_layers, expected.num_layers, "num_layers");
    validate_expected(h.full_attention_layers, expected.full_attention_layers,
                      "full_attention_layers");
    validate_expected(h.gdn_layers, expected.gdn_layers, "gdn_layers");
    validate_expected(h.attention_heads, expected.attention_heads, "attention_heads");
    validate_expected(h.kv_heads, expected.kv_heads, "kv_heads");
    validate_expected(h.head_dim, expected.head_dim, "head_dim");
    validate_expected(h.gdn_key_heads, expected.gdn_key_heads, "gdn_key_heads");
    validate_expected(h.gdn_value_heads, expected.gdn_value_heads, "gdn_value_heads");
    validate_expected(h.gdn_value_head_dim, expected.gdn_value_head_dim, "gdn_value_head_dim");
    validate_expected(h.gdn_key_head_dim, expected.gdn_key_head_dim, "gdn_key_head_dim");
    validate_expected(h.gdn_conv_width, expected.gdn_conv_width, "gdn_conv_width");
    validate_expected(h.vocab_size, expected.vocab_size, "vocab_size");
    validate_expected(h.max_position_embeddings, expected.max_position_embeddings,
                      "max_position_embeddings");
}

ParsedEntry parse_entry(const RawEntry& raw, std::uint64_t payload_offset, std::uint64_t file_size,
                        std::vector<Range>& ranges) {
    if (raw.reserved0 != 0 || !all_zero(raw.reserved1, sizeof(raw.reserved1))) {
        throw std::runtime_error("entry reserved fields are nonzero");
    }
    ParsedEntry e;
    e.name  = fixed_string(raw.name, sizeof(raw.name), "tensor name");
    e.role  = fixed_string(raw.role, sizeof(raw.role), "tensor role");
    e.layer = raw.layer;
    if (e.layer < -1) { throw std::runtime_error("invalid tensor layer"); }
    e.kind        = kind_from_tag(raw.kind);
    e.dtype       = dtype_from_tag(raw.dtype);
    e.rank        = raw.rank;
    e.shape       = parse_shape(raw.shape, raw.rank, "tensor");
    e.data_offset = raw.data_offset;
    e.data_nbytes = raw.data_nbytes;
    validate_payload_range(e.data_offset, e.data_nbytes, payload_offset, file_size, ranges);

    if (e.kind == WeightEntryKind::Dense) {
        if (raw.quant_layout != kNoLayout || raw.n != 0 || raw.k != 0 || raw.group != 0 ||
            raw.scale_dtype != kNoScaleDType || raw.scale_rank != 0 || raw.scale_offset != 0 ||
            raw.scale_nbytes != 0) {
            throw std::runtime_error("invalid dense tensor metadata");
        }
        for (std::uint32_t dim : raw.scale_shape) {
            if (dim != 0) { throw std::runtime_error("dense tensor has scale shape"); }
        }
        const Tensor tmp = make_tensor(nullptr, e.dtype, e.shape, e.rank);
        if (tmp.bytes() != e.data_nbytes) {
            throw std::runtime_error("dense tensor byte count mismatch");
        }
    } else {
        e.layout = layout_from_tag(raw.quant_layout);
        if (e.dtype != DType::U8 || raw.n == 0 || raw.k == 0 || raw.group == 0 ||
            raw.n > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            raw.k > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            raw.group > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("invalid quant tensor metadata");
        }
        e.n     = raw.n;
        e.k     = raw.k;
        e.group = raw.group;
        const std::uint64_t packed_bytes =
            checked_mul_size(e.n, static_cast<std::size_t>(ceil_div_u64(e.k, 2)));
        if (packed_bytes != e.data_nbytes ||
            e.data_nbytes > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("quant qdata byte count mismatch");
        }
        const Tensor qdata_shape = make_tensor(nullptr, e.dtype, e.shape, e.rank);
        if (qdata_shape.bytes() != e.data_nbytes) {
            throw std::runtime_error("quant serialized shape byte count mismatch");
        }
        e.scale_dtype = dtype_from_tag(raw.scale_dtype);
        if (e.scale_dtype != DType::FP32 || raw.scale_rank != 2) {
            throw std::runtime_error("invalid quant scale metadata");
        }
        e.scale_rank      = raw.scale_rank;
        e.scale_shape     = parse_shape(raw.scale_shape, raw.scale_rank, "scale");
        const auto groups = static_cast<std::int32_t>(ceil_div_u64(e.k, e.group));
        if (e.scale_shape[0] != static_cast<std::int32_t>(e.n) || e.scale_shape[1] != groups) {
            throw std::runtime_error("quant scale shape mismatch");
        }
        const Tensor scale_tmp = make_tensor(nullptr, e.scale_dtype, e.scale_shape, e.scale_rank);
        e.scale_offset         = raw.scale_offset;
        e.scale_nbytes         = raw.scale_nbytes;
        if (scale_tmp.bytes() != e.scale_nbytes) {
            throw std::runtime_error("quant scale byte count mismatch");
        }
        validate_payload_range(e.scale_offset, e.scale_nbytes, payload_offset, file_size, ranges);
    }

    return e;
}

std::vector<ParsedEntry> parse_entries(const std::vector<unsigned char>& table,
                                       const RawHeader& h) {
    std::vector<Range> ranges;
    std::vector<ParsedEntry> entries;
    entries.reserve(h.tensor_count);
    std::set<std::tuple<WeightEntryKind, std::string, int>> seen_kind_role_layer;
    for (std::uint32_t i = 0; i < h.tensor_count; ++i) {
        const std::uint64_t offset = static_cast<std::uint64_t>(i) * kEntrySize;
        if (offset > table.size() || kEntrySize > table.size() - offset) {
            throw std::runtime_error("tensor table entry outside file");
        }
        RawEntry raw{};
        std::memcpy(&raw, table.data() + offset, sizeof(raw));
        ParsedEntry entry = parse_entry(raw, h.payload_offset, h.file_size, ranges);
        auto key          = std::make_tuple(entry.kind, entry.role, entry.layer);
        if (!seen_kind_role_layer.insert(key).second) {
            throw std::runtime_error("duplicate tensor kind/role/layer");
        }
        entries.push_back(entry);
    }
    validate_non_overlapping(ranges);
    return entries;
}

void preflight_weights_arena(DeviceArena& arena, const std::vector<ParsedEntry>& entries) {
    if (arena.used() > arena.capacity()) {
        throw std::runtime_error("weights arena used exceeds capacity");
    }
    std::size_t required = 0;
    for (const ParsedEntry& e : entries) {
        required = checked_add_size(required,
                                    checked_add_size(static_cast<std::size_t>(e.data_nbytes), 255));
        if (e.kind == WeightEntryKind::QuantW4) {
            required = checked_add_size(
                required, checked_add_size(static_cast<std::size_t>(e.scale_nbytes), 255));
        }
    }
    if (required > arena.capacity() - arena.used()) { throw std::bad_alloc(); }
}

void upload_payload(std::ifstream& in, std::uint64_t offset, std::uint64_t nbytes, void* dst,
                    cudaStream_t stream) {
    if (nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("payload too large for host staging");
    }
    if (nbytes == 0) { return; }
    const std::size_t staging_size =
        std::min<std::size_t>(kStagingChunkBytes, static_cast<std::size_t>(nbytes));
    PinnedHostBuffer staging(staging_size);
    std::uint64_t copied = 0;
    while (copied < nbytes) {
        const auto chunk =
            static_cast<std::size_t>(std::min<std::uint64_t>(staging_size, nbytes - copied));
        in.seekg(static_cast<std::streamoff>(offset + copied), std::ios::beg);
        in.read(static_cast<char*>(staging.data()), static_cast<std::streamsize>(chunk));
        if (!in) { throw std::runtime_error("failed to read weight payload"); }
        auto* dst_bytes = static_cast<unsigned char*>(dst) + copied;
        cuda_throw(
            cudaMemcpyAsync(dst_bytes, staging.data(), chunk, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync");
        cuda_throw(cudaStreamSynchronize(stream), "cudaStreamSynchronize(load_stream)");
        copied += chunk;
    }
}

} // namespace

WeightStore::WeightStore(WeightFileExpectations expected) : expected_(std::move(expected)) {}

void WeightStore::load(const char* path, DeviceArena& weights, DeviceContext& ctx) {
    std::ifstream in              = open_weight_stream(path);
    const std::uint64_t file_size = stream_size(in);
    const RawHeader header        = read_header(in, file_size);
    validate_header(header, file_size, expected_);
    const std::vector<unsigned char> table = read_table(in, header);
    std::vector<ParsedEntry> entries       = parse_entries(table, header);
    preflight_weights_arena(weights, entries);

    std::vector<DenseRecord> dense_records;
    std::vector<QuantRecord> quant_records;
    dense_records.reserve(entries.size());
    quant_records.reserve(entries.size());

    for (const ParsedEntry& e : entries) {
        if (e.kind == WeightEntryKind::Dense) {
            Tensor tensor = alloc_tensor(weights, e.dtype, e.shape, e.rank);
            upload_payload(in, e.data_offset, e.data_nbytes, tensor.data, ctx.load_stream);
            dense_records.push_back(DenseRecord{e.name, e.role, e.layer, tensor});
        } else {
            Tensor qdata  = weights.alloc(DType::U8, {static_cast<std::int32_t>(e.data_nbytes)});
            Tensor scales = alloc_tensor(weights, e.scale_dtype, e.scale_shape, e.scale_rank);
            upload_payload(in, e.data_offset, e.data_nbytes, qdata.data, ctx.load_stream);
            upload_payload(in, e.scale_offset, e.scale_nbytes, scales.data, ctx.load_stream);
            QuantWeight qw{};
            qw.qdata       = qdata.data;
            qw.scales      = scales.data;
            qw.n           = static_cast<std::int32_t>(e.n);
            qw.k           = static_cast<std::int32_t>(e.k);
            qw.group       = static_cast<std::int32_t>(e.group);
            qw.layout      = e.layout;
            qw.scale_dtype = e.scale_dtype;
            for (int i = 0; i < 4; ++i) {
                qw.scale_ne[i] = scales.ne[i];
                qw.scale_nb[i] = scales.nb[i];
            }
            quant_records.push_back(QuantRecord{e.name, e.role, e.layer, qw});
        }
    }
    cuda_throw(cudaStreamSynchronize(ctx.load_stream), "cudaStreamSynchronize(load_stream)");

    dense_ = std::move(dense_records);
    quant_ = std::move(quant_records);
}

QuantWeight WeightStore::qweight(std::string_view role, int layer) const {
    for (const QuantRecord& record : quant_) {
        if (record.role == role && record.layer == layer) { return record.weight; }
    }
    throw std::out_of_range("quant weight not found");
}

Tensor WeightStore::weight(std::string_view role, int layer) const {
    for (const DenseRecord& record : dense_) {
        if (record.role == role && record.layer == layer) { return record.tensor; }
    }
    throw std::out_of_range("dense weight not found");
}

std::size_t WeightStore::dense_count() const noexcept { return dense_.size(); }

std::size_t WeightStore::quant_count() const noexcept { return quant_.size(); }

void WeightStore::clear() noexcept {
    dense_.clear();
    quant_.clear();
}

} // namespace qus
