#include "qus/core/weight_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qus {
namespace {

constexpr std::uint64_t kProgressChunkBytes = 256ULL * 1024ULL * 1024ULL;

class ProgressReporter {
public:
    explicit ProgressReporter(Q5090Progress* progress) : progress_(progress) {}

    void report(std::string_view phase, std::uint64_t done, std::uint64_t total,
                bool force = false) {
        if (progress_ == nullptr || !progress_->callback) { return; }
        if (!force && done < last_done_ + progress_->min_interval_bytes && done < total) { return; }
        last_done_ = done;
        progress_->callback(phase, done, total);
    }

private:
    Q5090Progress* progress_ = nullptr;
    std::uint64_t last_done_ = 0;
};

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void cuda_throw(cudaError_t err, const char* what) {
    if (err != cudaSuccess) { throw std::runtime_error(cuda_error_message(what, err)); }
}

std::vector<std::byte> read_file(const char* path, Q5090Progress* progress) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open q5090 weight file"); }
    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < 0) { throw std::runtime_error("failed to size q5090 weight file"); }
    in.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    ProgressReporter reporter(progress);
    std::uint64_t done        = 0;
    const std::uint64_t total = bytes.size();
    reporter.report("read file", 0, total, true);
    auto* out = reinterpret_cast<char*>(bytes.data());
    while (done < total) {
        const std::uint64_t chunk = std::min(kProgressChunkBytes, total - done);
        in.read(out + done, static_cast<std::streamsize>(chunk));
        if (!in) { throw std::runtime_error("failed to read q5090 weight file"); }
        done += chunk;
        reporter.report("read file", done, total);
    }
    reporter.report("read file", total, total, true);
    return bytes;
}

bool should_load(ModuleKind module, const LoadOptions& options) {
    switch (module) {
    case ModuleKind::TextCore:
        return options.load_text;
    case ModuleKind::MtpDraft:
        return options.load_mtp;
    case ModuleKind::VisionEncoder:
        return options.load_vision;
    }
    return false;
}

std::size_t module_index(ModuleKind module) { return static_cast<std::size_t>(module); }

DType dtype_for_contiguous(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL:
        return DType::BF16;
    case QType::FP32_CTRL:
        return DType::FP32;
    case QType::I32_CTRL:
        return DType::I32;
    default:
        throw std::runtime_error("q5090 contiguous tensor has non-control qtype");
    }
}

Tensor make_tensor_view(void* data, DType dtype, const std::array<std::uint32_t, 4>& shape,
                        std::uint32_t ndim) {
    switch (ndim) {
    case 1:
        return Tensor(data, dtype, {static_cast<std::int32_t>(shape[0])});
    case 2:
        return Tensor(data, dtype,
                      {static_cast<std::int32_t>(shape[0]), static_cast<std::int32_t>(shape[1])});
    case 3:
        return Tensor(data, dtype,
                      {static_cast<std::int32_t>(shape[0]), static_cast<std::int32_t>(shape[1]),
                       static_cast<std::int32_t>(shape[2])});
    case 4:
        return Tensor(data, dtype,
                      {static_cast<std::int32_t>(shape[0]), static_cast<std::int32_t>(shape[1]),
                       static_cast<std::int32_t>(shape[2]), static_cast<std::int32_t>(shape[3])});
    default:
        throw std::runtime_error("invalid q5090 tensor rank");
    }
}

void check_int32_shape(const ParsedQ5090Tensor& tensor) {
    for (std::uint32_t i = 0; i < tensor.ndim; ++i) {
        if (tensor.shape[i] >
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            tensor.padded_shape[i] >
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("q5090 tensor shape exceeds Tensor descriptor range");
        }
    }
}

Tensor alloc_payload(DeviceArena& arena, std::uint64_t payload_bytes) {
    if (payload_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("q5090 payload too large for Tensor descriptor");
    }
    return arena.alloc(DType::U8, {static_cast<std::int32_t>(payload_bytes)});
}

void upload_payload(const std::vector<std::byte>& file, const ParsedQ5090Tensor& tensor, void* dst,
                    cudaStream_t stream, ProgressReporter* reporter, std::uint64_t& uploaded,
                    std::uint64_t upload_total) {
    if (tensor.payload_bytes == 0) { return; }
    const auto* src = file.data() + tensor.payload_offset;
    if (reporter == nullptr) {
        cuda_throw(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(tensor.payload_bytes),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(q5090 weight payload)");
        return;
    }

    auto* dst_bytes      = static_cast<std::byte*>(dst);
    std::uint64_t copied = 0;
    while (copied < tensor.payload_bytes) {
        const std::uint64_t chunk = std::min(kProgressChunkBytes, tensor.payload_bytes - copied);
        cuda_throw(cudaMemcpyAsync(dst_bytes + copied, src + copied,
                                   static_cast<std::size_t>(chunk), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(q5090 weight payload)");
        cuda_throw(cudaStreamSynchronize(stream), "cudaStreamSynchronize(q5090 weight payload)");
        copied += chunk;
        uploaded += chunk;
        reporter->report("upload payloads", uploaded, upload_total);
    }
}

bool is_quant_layout(QuantLayout layout) {
    return layout == QuantLayout::RowSplit;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t align) {
    const std::uint64_t add = align - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - add) {
        throw std::overflow_error("q5090 row-split plane offset overflow");
    }
    return ((value + add) / align) * align;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error("q5090 row-split byte offset overflow");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("q5090 row-split byte offset overflow");
    }
    return a + b;
}

std::uint64_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::W8G32_F16S:
        return 32;
    case QType::W8G128_F16S:
        return 128;
    default:
        throw std::runtime_error("q5090 row-split qtype has no nibble bytes per group");
    }
}

std::uint64_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::W8G128_F16S:
    case QType::W8G32_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        throw std::runtime_error("q5090 row-split qtype has no high bytes per group");
    }
}

Weight make_quant_descriptor(const ParsedQ5090Tensor& tensor, const ParsedQ5090Segment& segment,
                             void* payload) {
    check_int32_shape(tensor);
    if (tensor.ndim != 2 || tensor.layout != QuantLayout::RowSplit) {
        throw std::runtime_error("q5090 quant descriptor requires ROW_SPLIT matrix");
    }
    if (segment.row_count >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("q5090 segment row_count exceeds Weight descriptor range");
    }
    Weight weight{};
    weight.payload           = payload;
    weight.payload_bytes     = tensor.payload_bytes;
    weight.high_plane_bytes  = tensor.high_plane_bytes;
    weight.qtype             = tensor.qtype;
    weight.layout            = tensor.layout;
    weight.module            = tensor.module_kind;
    weight.q5090_scale_dtype = tensor.scale_dtype;
    weight.group_size        = tensor.group_size;
    weight.source_layer      = segment.source_layer;
    weight.source_kind       = segment.source_kind;
    weight.ndim              = 2;

    const std::uint64_t groups = tensor.padded_shape[1] / tensor.group_size;
    const std::uint64_t nibble_group_bytes = nibble_bytes_per_group(tensor.qtype);
    const std::uint64_t high_group_bytes = high_bytes_per_group(tensor.qtype);
    const std::uint64_t nibble_row_bytes = checked_mul_u64(groups, nibble_group_bytes);
    const std::uint64_t high_row_bytes = checked_mul_u64(groups, high_group_bytes);
    const std::uint64_t scale_row_bytes = checked_mul_u64(groups, 2);
    const std::uint64_t expected_nibble =
        checked_mul_u64(static_cast<std::uint64_t>(tensor.shape[0]), nibble_row_bytes);
    const std::uint64_t expected_high =
        checked_mul_u64(static_cast<std::uint64_t>(tensor.shape[0]), high_row_bytes);
    const std::uint64_t expected_scale =
        checked_mul_u64(static_cast<std::uint64_t>(tensor.shape[0]), scale_row_bytes);
    if (tensor.nibble_plane_bytes != expected_nibble || tensor.high_plane_bytes != expected_high ||
        tensor.scale_plane_bytes != expected_scale) {
        throw std::runtime_error("q5090 row-split plane byte fields do not match descriptor math");
    }

    const std::uint64_t high_rel = align_up(tensor.nibble_plane_bytes, 256);
    const std::uint64_t scale_rel = checked_add(high_rel, align_up(tensor.high_plane_bytes, 256));
    const std::uint64_t nibble_offset = checked_mul_u64(segment.row_begin, nibble_row_bytes);
    const std::uint64_t high_offset =
        checked_add(high_rel, checked_mul_u64(segment.row_begin, high_row_bytes));
    const std::uint64_t scale_offset =
        checked_add(scale_rel, checked_mul_u64(segment.row_begin, scale_row_bytes));
    if (payload != nullptr) {
        const auto* bytes = static_cast<const std::byte*>(payload);
        weight.qdata      = bytes + nibble_offset;
        weight.qhigh      = high_group_bytes == 0 ? nullptr : bytes + high_offset;
        weight.scales     = bytes + scale_offset;
    } else {
        weight.qdata  = nullptr;
        weight.qhigh  = nullptr;
        weight.scales = nullptr;
    }

    weight.n           = static_cast<std::int32_t>(segment.row_count);
    weight.k           = static_cast<std::int32_t>(tensor.shape[1]);
    weight.group       = static_cast<std::int32_t>(tensor.group_size);
    weight.scale_dtype = DType::BF16;
    weight.shape[0]        = static_cast<std::int32_t>(segment.row_count);
    weight.shape[1]        = static_cast<std::int32_t>(tensor.shape[1]);
    weight.shape[2]        = 1;
    weight.shape[3]        = 1;
    weight.padded_shape[0] = static_cast<std::int32_t>(segment.row_count);
    weight.padded_shape[1] = static_cast<std::int32_t>(tensor.padded_shape[1]);
    weight.padded_shape[2] = 1;
    weight.padded_shape[3] = 1;
    return weight;
}

bool contains_text_name(const ParsedQ5090File& parsed, std::string_view name) {
    if (std::any_of(parsed.tensors.begin(), parsed.tensors.end(),
                    [name](const ParsedQ5090Tensor& tensor) {
        return tensor.module_kind == ModuleKind::TextCore && tensor.name == name;
    })) {
        return true;
    }
    for (std::size_t i = 0; i < parsed.segments.size(); ++i) {
        if (parsed.segments[i].name != name) { continue; }
        for (const ParsedQ5090Tensor& tensor : parsed.tensors) {
            const std::uint64_t begin = tensor.segment_begin;
            const std::uint64_t end   = begin + tensor.segment_count;
            if (i >= begin && i < end && tensor.module_kind == ModuleKind::TextCore) {
                return true;
            }
        }
    }
    return false;
}

const ParsedQ5090FusionGroup& require_fusion_group_record(const ParsedQ5090File& parsed,
                                                          const ParsedQ5090Tensor& tensor,
                                                          std::size_t tensor_index) {
    for (const ParsedQ5090FusionGroup& group : parsed.fusion_groups) {
        if (group.group_id != tensor.fusion_group_id) { continue; }
        const std::uint64_t first = group.first_block_tensor_index;
        const std::uint64_t end   = first + group.block_count;
        if (tensor_index < first || tensor_index >= end) { continue; }
        if (tensor.fusion_index >= group.block_count) {
            throw std::runtime_error("q5090 fused block fusion_index exceeds group block_count");
        }
        if (first + tensor.fusion_index != tensor_index) {
            throw std::runtime_error("q5090 fused block index does not match fusion_index");
        }
        if (tensor.source_layer != group.source_layer) {
            throw std::runtime_error("q5090 fused block source_layer does not match group");
        }
        if (tensor.shape[1] != group.shared_k) {
            throw std::runtime_error("q5090 fused block K does not match group shared_k");
        }
        return group;
    }
    throw std::runtime_error("q5090 fused block references no FusionGroupRecord");
}

} // namespace

WeightStore::WeightStore(Q5090Expectations expected) : expected_(std::move(expected)) {}

void WeightStore::load(const char* path, DeviceArena& weights, DeviceContext& ctx,
                       const LoadOptions& options) {
    const std::vector<std::byte> file = read_file(path, options.progress);
    const ParsedQ5090File parsed      = parse_q5090_file(file, expected_, options.progress);

    for (const std::string& required : options.required_text_tensors) {
        if (!contains_text_name(parsed, required)) {
            throw std::runtime_error("required TEXT_CORE tensor is missing: " + required);
        }
    }

    const std::size_t mark = weights.mark();
    std::vector<TensorRecord> next_tensors;
    std::vector<QuantRecord> next_quant;
    std::vector<FusedBlockRecord> next_fused;
    ModuleState next_modules[3]{};
    std::size_t next_loaded_payload_bytes = 0;

    try {
        for (const ParsedQ5090Module& module : parsed.modules) {
            ModuleState& state  = next_modules[module_index(module.module_kind)];
            state.present       = true;
            state.loaded        = should_load(module.module_kind, options);
            state.tensor_count  = module.tensor_index_count;
            state.payload_bytes = module.payload_bytes;
            if (state.loaded) {
                next_loaded_payload_bytes += static_cast<std::size_t>(module.payload_bytes);
            }
        }

        next_tensors.reserve(parsed.tensors.size());
        next_quant.reserve(parsed.segments.size());
        next_fused.reserve(parsed.tensors.size());
        std::uint64_t upload_total = 0;
        for (const ParsedQ5090Tensor& tensor : parsed.tensors) {
            if (should_load(tensor.module_kind, options)) { upload_total += tensor.payload_bytes; }
        }
        ProgressReporter upload_reporter(options.progress);
        ProgressReporter* upload_progress =
            options.progress != nullptr ? &upload_reporter : nullptr;
        std::uint64_t uploaded = 0;
        upload_reporter.report("upload payloads", 0, upload_total, true);
        for (std::size_t tensor_index = 0; tensor_index < parsed.tensors.size(); ++tensor_index) {
            const ParsedQ5090Tensor& tensor = parsed.tensors[tensor_index];
            const bool load_payload = should_load(tensor.module_kind, options);
            void* payload           = nullptr;
            if (load_payload) {
                Tensor raw = alloc_payload(weights, tensor.payload_bytes);
                payload    = raw.data;
                upload_payload(file, tensor, payload, ctx.load_stream, upload_progress, uploaded,
                               upload_total);
            }

            if (tensor.layout == QuantLayout::Contiguous) {
                if (tensor.segment_count != 1) {
                    throw std::runtime_error("q5090 contiguous block must have one segment");
                }
                const ParsedQ5090Segment& segment =
                    parsed.segments[static_cast<std::size_t>(tensor.segment_begin)];
                check_int32_shape(tensor);
                Tensor view = make_tensor_view(payload, dtype_for_contiguous(tensor.qtype),
                                               tensor.shape, tensor.ndim);
                next_tensors.push_back(TensorRecord{segment.name, tensor.module_kind,
                                                    segment.source_kind, segment.source_layer,
                                                    tensor.qtype, tensor.payload_bytes, view});
            } else if (is_quant_layout(tensor.layout)) {
                for (std::uint32_t j = 0; j < tensor.segment_count; ++j) {
                    const ParsedQ5090Segment& segment =
                        parsed.segments[static_cast<std::size_t>(tensor.segment_begin + j)];
                    next_quant.push_back(
                        QuantRecord{segment.name, tensor.name, tensor.module_kind,
                                    segment.source_kind, segment.source_layer, segment.row_begin,
                                    segment.row_count,
                                    make_quant_descriptor(tensor, segment, payload)});
                }
                if (tensor.fusion_group_id != 0) {
                    const ParsedQ5090FusionGroup& group =
                        require_fusion_group_record(parsed, tensor, tensor_index);
                    ParsedQ5090Segment block_seg{};
                    block_seg.source_kind  = static_cast<std::uint32_t>(SourceKind::Other);
                    block_seg.source_layer = group.source_layer;
                    block_seg.row_begin    = 0;
                    block_seg.row_count    = tensor.shape[0];
                    next_fused.push_back(FusedBlockRecord{
                        tensor.name, tensor.module_kind,
                        static_cast<std::uint16_t>(tensor.fusion_group_id),
                        static_cast<std::uint16_t>(tensor.fusion_index), group.source_layer,
                        make_quant_descriptor(tensor, block_seg, payload)});
                }
            } else {
                throw std::runtime_error("unsupported q5090 tensor layout");
            }
        }
        upload_reporter.report("upload payloads", upload_total, upload_total, true);
        cuda_throw(cudaStreamSynchronize(ctx.load_stream), "cudaStreamSynchronize(load_stream)");
    } catch (...) {
        weights.rewind(mark);
        weights.reset_peak();
        throw;
    }

    tensors_              = std::move(next_tensors);
    quant_                = std::move(next_quant);
    fused_                = std::move(next_fused);
    total_tensor_count_   = parsed.tensors.size();
    loaded_payload_bytes_ = next_loaded_payload_bytes;
    for (std::size_t i = 0; i < 3; ++i) { modules_[i] = next_modules[i]; }
}

const Tensor* WeightStore::tensor(std::string_view name) const noexcept {
    for (const TensorRecord& record : tensors_) {
        if (record.name == name) { return &record.tensor; }
    }
    return nullptr;
}

const Weight* WeightStore::qweight(std::string_view name) const noexcept {
    for (const QuantRecord& record : quant_) {
        if (record.name == name) { return &record.weight; }
    }
    return nullptr;
}

const Tensor* WeightStore::tensor(ModuleKind module, std::uint32_t source_kind,
                                  std::uint32_t source_layer) const noexcept {
    for (const TensorRecord& record : tensors_) {
        if (record.module == module && record.source_kind == source_kind &&
            record.source_layer == source_layer) {
            return &record.tensor;
        }
    }
    return nullptr;
}

const Weight* WeightStore::qweight(ModuleKind module, std::uint32_t source_kind,
                                   std::uint32_t source_layer) const noexcept {
    for (const QuantRecord& record : quant_) {
        if (record.module == module && record.source_kind == source_kind &&
            record.source_layer == source_layer) {
            return &record.weight;
        }
    }
    return nullptr;
}

const Weight* WeightStore::qfused(ModuleKind module, std::uint16_t group_id,
                                  std::uint16_t fusion_index,
                                  std::uint32_t source_layer) const noexcept {
    for (const FusedBlockRecord& record : fused_) {
        if (record.module == module && record.group_id == group_id &&
            record.fusion_index == fusion_index && record.source_layer == source_layer) {
            return &record.weight;
        }
    }
    return nullptr;
}

std::size_t WeightStore::tensor_count() const noexcept { return total_tensor_count_; }

std::size_t WeightStore::quant_count() const noexcept { return quant_.size(); }

std::size_t WeightStore::loaded_payload_bytes() const noexcept { return loaded_payload_bytes_; }

std::size_t WeightStore::module_tensor_count(ModuleKind module) const noexcept {
    const ModuleState& state = modules_[module_index(module)];
    return state.present ? static_cast<std::size_t>(state.tensor_count) : 0;
}

bool WeightStore::module_loaded(ModuleKind module) const noexcept {
    return modules_[module_index(module)].loaded;
}

void WeightStore::require_mtp_module_expectations() const {
    const ModuleState& mtp = modules_[module_index(ModuleKind::MtpDraft)];
    if (!mtp.present) { throw std::runtime_error("q5090 MTP_DRAFT module is missing"); }
    if (!mtp.loaded) { throw std::runtime_error("q5090 MTP_DRAFT module is not loaded"); }
    if (mtp.tensor_count != 12) {
        throw std::runtime_error("q5090 MTP_DRAFT tensor count mismatch");
    }

    std::size_t mtp_quant_segments = 0;
    for (const QuantRecord& record : quant_) {
        if (record.module == ModuleKind::MtpDraft) { ++mtp_quant_segments; }
    }
    std::size_t mtp_dense_segments = 0;
    for (const TensorRecord& record : tensors_) {
        if (record.module == ModuleKind::MtpDraft) { ++mtp_dense_segments; }
    }
    std::size_t mtp_fused_blocks = 0;
    for (const FusedBlockRecord& record : fused_) {
        if (record.module == ModuleKind::MtpDraft) { ++mtp_fused_blocks; }
    }

    if (mtp_quant_segments + mtp_dense_segments != 16) {
        throw std::runtime_error("q5090 MTP_DRAFT segment count mismatch");
    }
    if (mtp_dense_segments != 7) {
        throw std::runtime_error("q5090 MTP_DRAFT BF16 control count mismatch");
    }
    if (mtp_fused_blocks != 2) {
        throw std::runtime_error("q5090 MTP_DRAFT fusion group count mismatch");
    }

    auto require_w8_metadata = [](const Weight& weight, const char* label) {
        const std::string prefix = std::string("q5090 invalid MTP W8G32 weight: ") + label;
        if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
            weight.group_size != 32 || weight.group != 32 || weight.high_plane_bytes != 0 ||
            weight.qhigh != nullptr || weight.q5090_scale_dtype != ScaleDType::FP16 ||
            weight.ndim != 2 || weight.n <= 0 || weight.k <= 0 || weight.shape[0] != weight.n ||
            weight.shape[1] != weight.k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
            weight.padded_shape[0] != weight.n || weight.padded_shape[1] < weight.k ||
            weight.padded_shape[2] != 1 || weight.padded_shape[3] != 1) {
            throw std::runtime_error(prefix);
        }
    };
    auto require_equal_u32 = [](std::uint32_t got, std::uint32_t want, const char* label) {
        if (got != want) {
            throw std::runtime_error(std::string("q5090 invalid MTP layout: ") + label);
        }
    };
    auto require_equal_i32 = [](std::int32_t got, std::uint32_t want, const char* label) {
        if (got < 0 || static_cast<std::uint32_t>(got) != want) {
            throw std::runtime_error(std::string("q5090 invalid MTP layout: ") + label);
        }
    };
    auto checked_add_u32 = [](std::uint32_t a, std::uint32_t b, const char* label) {
        if (b > std::numeric_limits<std::uint32_t>::max() - a) {
            throw std::overflow_error(label);
        }
        return a + b;
    };
    auto checked_mul_u32 = [](std::uint32_t a, std::uint32_t b, const char* label) {
        if (b != 0 && a > std::numeric_limits<std::uint32_t>::max() / b) {
            throw std::overflow_error(label);
        }
        return a * b;
    };
    auto require_w8_segment = [&](const char* block_name, const char* segment_name,
                                  SourceKind source_kind, std::uint32_t source_layer,
                                  const char* label) -> const QuantRecord& {
        for (const QuantRecord& record : quant_) {
            if (record.module == ModuleKind::MtpDraft && record.block_name == block_name &&
                record.name == segment_name &&
                record.source_kind == static_cast<std::uint32_t>(source_kind) &&
                record.source_layer == source_layer) {
                require_w8_metadata(record.weight, label);
                return record;
            }
        }
        throw std::runtime_error(std::string("q5090 missing MTP W8G32 segment: ") + label);
    };
    auto require_segment_range = [&](const QuantRecord& record, std::uint32_t row_begin,
                                     std::uint32_t row_count, const char* label) {
        require_equal_u32(record.row_begin, row_begin, label);
        require_equal_u32(record.row_count, row_count, label);
        require_equal_i32(record.weight.n, row_count, label);
    };
    auto require_fused_w8 = [&](const char* block_name, std::uint16_t group_id,
                                std::uint32_t total_n, std::uint32_t k,
                                const char* label) {
        for (const FusedBlockRecord& record : fused_) {
            if (record.module == ModuleKind::MtpDraft && record.name == block_name &&
                record.group_id == group_id && record.fusion_index == 0 &&
                record.source_layer == 0) {
                require_w8_metadata(record.weight, label);
                if (record.weight.source_kind != static_cast<std::uint32_t>(SourceKind::Other)) {
                    throw std::runtime_error(std::string("q5090 invalid MTP fused source: ") +
                                             label);
                }
                require_equal_i32(record.weight.n, total_n, label);
                require_equal_i32(record.weight.k, k, label);
                return;
            }
        }
        throw std::runtime_error(std::string("q5090 missing MTP W8G32 fused block: ") + label);
    };
    auto require_bf16 = [&](const char* name, SourceKind source_kind,
                            std::uint32_t source_layer) -> const TensorRecord& {
        for (const TensorRecord& record : tensors_) {
            if (record.module == ModuleKind::MtpDraft && record.name == name &&
                record.source_kind == static_cast<std::uint32_t>(source_kind) &&
                record.source_layer == source_layer) {
                if (record.qtype != QType::BF16_CTRL || record.tensor.dtype != DType::BF16) {
                    throw std::runtime_error(std::string("q5090 invalid MTP BF16 control: ") +
                                             name);
                }
                return record;
            }
        }
        throw std::runtime_error(std::string("q5090 missing MTP BF16 control: ") + name);
    };
    auto require_bf16_1d = [&](const char* name, SourceKind source_kind,
                               std::uint32_t source_layer, std::uint32_t size) {
        const TensorRecord& record = require_bf16(name, source_kind, source_layer);
        if (record.tensor.ne[0] < 0 || static_cast<std::uint32_t>(record.tensor.ne[0]) != size ||
            record.tensor.ne[1] != 1 || record.tensor.ne[2] != 1 || record.tensor.ne[3] != 1) {
            throw std::runtime_error(std::string("q5090 invalid MTP BF16 shape: ") + name);
        }
    };

    const QuantRecord& fc = require_w8_segment(
        "mtp.fc.weight", "mtp.fc.weight", SourceKind::MtpFc, kQ5090NoLayer, "mtp.fc.weight");
    require_segment_range(fc, 0, fc.row_count, "mtp.fc.weight");
    if (fc.row_count == 0) { throw std::runtime_error("q5090 invalid MTP hidden size"); }
    const std::uint32_t hidden = fc.row_count;
    require_equal_i32(fc.weight.k, checked_mul_u32(hidden, 2, "q5090 MTP fc K overflow"),
                      "mtp.fc.weight K");

    require_bf16_1d("mtp.pre_fc_norm_embedding.weight", SourceKind::MtpPreFcNormEmb,
                    kQ5090NoLayer, hidden);
    require_bf16_1d("mtp.pre_fc_norm_hidden.weight", SourceKind::MtpPreFcNormHid,
                    kQ5090NoLayer, hidden);
    require_bf16_1d("mtp.layers.0.input_layernorm.weight", SourceKind::InputLayernorm, 0,
                    hidden);

    const char* attn_block = "mtp.layers.0.attn_in.w8";
    const QuantRecord& attn_q =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.q_proj.q", SourceKind::AttnQ, 0,
                           "mtp.layers.0.attn_in.w8 ATTN_Q");
    const std::uint32_t q_rows = attn_q.row_count;
    require_segment_range(attn_q, 0, q_rows, "mtp.layers.0.attn_in.w8 ATTN_Q");
    require_equal_i32(attn_q.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_k = require_w8_segment(attn_block,
                                                   "mtp.layers.0.self_attn.k_proj.weight",
                                                   SourceKind::AttnK, 0,
                                                   "mtp.layers.0.attn_in.w8 ATTN_K");
    const std::uint32_t k_rows = attn_k.row_count;
    require_segment_range(attn_k, q_rows, k_rows, "mtp.layers.0.attn_in.w8 ATTN_K");
    require_equal_i32(attn_k.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_gate =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.q_proj.gate",
                           SourceKind::AttnGate, 0, "mtp.layers.0.attn_in.w8 ATTN_GATE");
    const std::uint32_t gate_begin =
        checked_add_u32(q_rows, k_rows, "q5090 MTP attn gate row overflow");
    require_segment_range(attn_gate, gate_begin, q_rows, "mtp.layers.0.attn_in.w8 ATTN_GATE");
    require_equal_i32(attn_gate.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_v = require_w8_segment(attn_block,
                                                   "mtp.layers.0.self_attn.v_proj.weight",
                                                   SourceKind::AttnV, 0,
                                                   "mtp.layers.0.attn_in.w8 ATTN_V");
    const std::uint32_t v_begin =
        checked_add_u32(gate_begin, q_rows, "q5090 MTP attn v row overflow");
    require_segment_range(attn_v, v_begin, k_rows, "mtp.layers.0.attn_in.w8 ATTN_V");
    require_equal_i32(attn_v.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");
    const std::uint32_t attn_total =
        checked_add_u32(v_begin, k_rows, "q5090 MTP attn total row overflow");
    require_fused_w8(attn_block, /*ATTN_IN*/ 1, attn_total, hidden,
                     "mtp.layers.0.attn_in.w8");

    const TensorRecord& q_norm =
        require_bf16("mtp.layers.0.self_attn.q_norm.weight", SourceKind::AttnQNorm, 0);
    const TensorRecord& k_norm =
        require_bf16("mtp.layers.0.self_attn.k_norm.weight", SourceKind::AttnKNorm, 0);
    if (q_norm.tensor.ne[0] <= 0 || k_norm.tensor.ne[0] != q_norm.tensor.ne[0] ||
        q_norm.tensor.ne[1] != 1 || q_norm.tensor.ne[2] != 1 || q_norm.tensor.ne[3] != 1 ||
        k_norm.tensor.ne[1] != 1 || k_norm.tensor.ne[2] != 1 || k_norm.tensor.ne[3] != 1 ||
        (q_rows % static_cast<std::uint32_t>(q_norm.tensor.ne[0])) != 0 ||
        (k_rows % static_cast<std::uint32_t>(q_norm.tensor.ne[0])) != 0) {
        throw std::runtime_error("q5090 invalid MTP q/k norm shape");
    }

    const QuantRecord& o_proj = require_w8_segment(
        "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
        SourceKind::AttnO, 0, "mtp.layers.0.self_attn.o_proj.weight");
    require_segment_range(o_proj, 0, hidden, "mtp.layers.0.self_attn.o_proj.weight");
    require_equal_i32(o_proj.weight.k, q_rows, "mtp.layers.0.self_attn.o_proj.weight K");
    require_bf16_1d("mtp.layers.0.post_attention_layernorm.weight",
                    SourceKind::PostAttnLayernorm, 0, hidden);

    const char* gateup_block = "mtp.layers.0.mlp.gateup.w8";
    const QuantRecord& mlp_gate =
        require_w8_segment(gateup_block, "mtp.layers.0.mlp.gate_proj.weight",
                           SourceKind::MlpGate, 0, "mtp.layers.0.mlp.gateup.w8 MLP_GATE");
    const std::uint32_t intermediate = mlp_gate.row_count;
    require_segment_range(mlp_gate, 0, intermediate, "mtp.layers.0.mlp.gateup.w8 MLP_GATE");
    require_equal_i32(mlp_gate.weight.k, hidden, "mtp.layers.0.mlp.gateup.w8 K");

    const QuantRecord& mlp_up =
        require_w8_segment(gateup_block, "mtp.layers.0.mlp.up_proj.weight", SourceKind::MlpUp, 0,
                           "mtp.layers.0.mlp.gateup.w8 MLP_UP");
    require_segment_range(mlp_up, intermediate, intermediate,
                          "mtp.layers.0.mlp.gateup.w8 MLP_UP");
    require_equal_i32(mlp_up.weight.k, hidden, "mtp.layers.0.mlp.gateup.w8 K");
    require_fused_w8(gateup_block, /*MLP_GATEUP*/ 3,
                     checked_mul_u32(intermediate, 2, "q5090 MTP gateup row overflow"), hidden,
                     "mtp.layers.0.mlp.gateup.w8");

    const QuantRecord& down = require_w8_segment(
        "mtp.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
        SourceKind::MlpDown, 0, "mtp.layers.0.mlp.down_proj.weight");
    require_segment_range(down, 0, hidden, "mtp.layers.0.mlp.down_proj.weight");
    require_equal_i32(down.weight.k, intermediate, "mtp.layers.0.mlp.down_proj.weight K");
    require_bf16_1d("mtp.norm.weight", SourceKind::MtpNorm, kQ5090NoLayer, hidden);

    if (mtp.payload_bytes == 451267584ULL) {
        require_equal_u32(hidden, 5120, "real MTP hidden size");
        require_equal_u32(q_rows, 6144, "real MTP q rows");
        require_equal_u32(k_rows, 1024, "real MTP kv rows");
        require_equal_u32(intermediate, 17408, "real MTP intermediate size");
        require_equal_i32(q_norm.tensor.ne[0], 256, "real MTP q/k norm size");
    }
}

void WeightStore::clear() noexcept {
    tensors_.clear();
    quant_.clear();
    fused_.clear();
    for (ModuleState& module : modules_) { module = ModuleState{}; }
    total_tensor_count_   = 0;
    loaded_payload_bytes_ = 0;
}

} // namespace qus
