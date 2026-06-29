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

std::uint64_t bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 32;
    case QType::Q5G64_F16S:
        return 40;
    case QType::Q6G64_F16S:
        return 48;
    case QType::W8G128_F16S:
        return 128;
    default:
        throw std::runtime_error("q5090 row-split qtype has no packed bytes per group");
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
    weight.qtype             = tensor.qtype;
    weight.layout            = tensor.layout;
    weight.module            = tensor.module_kind;
    weight.q5090_scale_dtype = tensor.scale_dtype;
    weight.group_size        = tensor.group_size;
    weight.source_layer      = segment.source_layer;
    weight.source_kind       = segment.source_kind;
    weight.ndim              = 2;

    const std::uint64_t groups = tensor.padded_shape[1] / tensor.group_size;
    const std::uint64_t code_row_bytes = checked_mul_u64(groups, bytes_per_group(tensor.qtype));
    const std::uint64_t scale_row_bytes = checked_mul_u64(groups, 2);
    const std::uint64_t code_offset = checked_mul_u64(segment.row_begin, code_row_bytes);
    const std::uint64_t scale_offset =
        align_up(tensor.code_plane_bytes, 256) + checked_mul_u64(segment.row_begin, scale_row_bytes);
    if (payload != nullptr) {
        const auto* bytes = static_cast<const std::byte*>(payload);
        weight.qdata      = bytes + code_offset;
        weight.scales     = bytes + scale_offset;
    } else {
        weight.qdata  = nullptr;
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
        std::uint64_t upload_total = 0;
        for (const ParsedQ5090Tensor& tensor : parsed.tensors) {
            if (should_load(tensor.module_kind, options)) { upload_total += tensor.payload_bytes; }
        }
        ProgressReporter upload_reporter(options.progress);
        ProgressReporter* upload_progress =
            options.progress != nullptr ? &upload_reporter : nullptr;
        std::uint64_t uploaded = 0;
        upload_reporter.report("upload payloads", 0, upload_total, true);
        for (const ParsedQ5090Tensor& tensor : parsed.tensors) {
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
                                                    tensor.payload_bytes, view});
            } else if (is_quant_layout(tensor.layout)) {
                for (std::uint32_t j = 0; j < tensor.segment_count; ++j) {
                    const ParsedQ5090Segment& segment =
                        parsed.segments[static_cast<std::size_t>(tensor.segment_begin + j)];
                    next_quant.push_back(
                        QuantRecord{segment.name, tensor.module_kind, segment.source_kind,
                                    segment.source_layer,
                                    make_quant_descriptor(tensor, segment, payload)});
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

void WeightStore::clear() noexcept {
    tensors_.clear();
    quant_.clear();
    for (ModuleState& module : modules_) { module = ModuleState{}; }
    total_tensor_count_   = 0;
    loaded_payload_bytes_ = 0;
}

} // namespace qus
