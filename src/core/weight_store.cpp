#include "ninfer/core/weight_store.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

constexpr std::uint64_t kHeaderBytes   = 4096;
constexpr std::size_t kPinnedSlotCount = 2;
constexpr std::size_t kPinnedSlotBytes = 64ULL * 1024ULL * 1024ULL;

class ProgressReporter {
public:
    explicit ProgressReporter(Q5090Progress* progress) : progress_(progress) {}

    void report(std::string_view phase, std::uint64_t done, std::uint64_t total,
                bool force = false) {
        if (progress_ == nullptr || !progress_->callback) { return; }
        if (phase_ != phase) {
            phase_     = std::string(phase);
            last_done_ = 0;
        }
        if (!force && done < last_done_ + progress_->min_interval_bytes && done < total) { return; }
        last_done_ = done;
        progress_->callback(phase, done, total);
    }

private:
    Q5090Progress* progress_ = nullptr;
    std::string phase_;
    std::uint64_t last_done_ = 0;
};

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void cuda_throw(cudaError_t err, const char* what) {
    if (err != cudaSuccess) { throw std::runtime_error(cuda_error_message(what, err)); }
}

bool should_load(ModuleKind module, const LoadOptions& options) {
    switch (module) {
    case ModuleKind::TextCore:
        return true;
    case ModuleKind::MtpDraft:
        return options.load_mtp;
    case ModuleKind::VisionEncoder:
        return options.load_vision;
    case ModuleKind::LmHeadDraft:
        return options.load_lm_head_draft;
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

bool is_quant_layout(QuantLayout layout) { return layout == QuantLayout::RowSplit; }

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
    default:
        throw std::runtime_error("q5090 row-split qtype has no nibble bytes per group");
    }
}

std::uint64_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
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
    if (segment.row_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
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

    const std::uint64_t groups             = tensor.padded_shape[1] / tensor.group_size;
    const std::uint64_t nibble_group_bytes = nibble_bytes_per_group(tensor.qtype);
    const std::uint64_t high_group_bytes   = high_bytes_per_group(tensor.qtype);
    const std::uint64_t nibble_row_bytes   = checked_mul_u64(groups, nibble_group_bytes);
    const std::uint64_t high_row_bytes     = checked_mul_u64(groups, high_group_bytes);
    const std::uint64_t scale_row_bytes    = checked_mul_u64(groups, 2);
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

    const std::uint64_t high_rel  = align_up(tensor.nibble_plane_bytes, 256);
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

    weight.n               = static_cast<std::int32_t>(segment.row_count);
    weight.k               = static_cast<std::int32_t>(tensor.shape[1]);
    weight.group           = static_cast<std::int32_t>(tensor.group_size);
    weight.scale_dtype     = DType::BF16;
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

struct WeightStore::PreparedArtifact {
    int fd = -1;

    struct stat identity {};

    ParsedQ5090File parsed;
    LoadOptions options;
    std::optional<Q5090Progress> owned_progress;

    ~PreparedArtifact() {
        if (fd >= 0) { ::close(fd); }
    }

    void read_exact(std::uint64_t offset, std::span<std::byte> output) const {
        std::size_t done = 0;
        while (done < output.size()) {
            const std::size_t request = std::min<std::size_t>(
                output.size() - done,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const std::uint64_t absolute = offset + done;
            if (absolute > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
                throw std::overflow_error("q5090 pread offset exceeds off_t");
            }
            const ssize_t got =
                ::pread(fd, output.data() + done, request, static_cast<off_t>(absolute));
            if (got < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::string("q5090 pread failed: ") +
                                         std::strerror(errno));
            }
            if (got == 0) { throw std::runtime_error("q5090 unexpected EOF"); }
            done += static_cast<std::size_t>(got);
        }
    }

    void verify_identity() const {
        struct stat current {};

        if (::fstat(fd, &current) != 0) {
            throw std::runtime_error(std::string("q5090 fstat failed: ") + std::strerror(errno));
        }
        const bool same = current.st_dev == identity.st_dev && current.st_ino == identity.st_ino &&
                          current.st_size == identity.st_size &&
                          current.st_mtim.tv_sec == identity.st_mtim.tv_sec &&
                          current.st_mtim.tv_nsec == identity.st_mtim.tv_nsec &&
                          current.st_ctim.tv_sec == identity.st_ctim.tv_sec &&
                          current.st_ctim.tv_nsec == identity.st_ctim.tv_nsec;
        if (!same) { throw std::runtime_error("q5090 artifact changed during load"); }
    }
};

namespace {

class UploadSlot {
public:
    explicit UploadSlot(std::size_t bytes) : buffer(bytes) {
        cuda_throw(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
                   "cudaEventCreate(q5090 upload slot)");
    }

    ~UploadSlot() {
        if (done != nullptr) {
            if (in_flight) { (void)cudaEventSynchronize(done); }
            (void)cudaEventDestroy(done);
        }
    }

    UploadSlot(const UploadSlot&)            = delete;
    UploadSlot& operator=(const UploadSlot&) = delete;

    PinnedHostBuffer buffer;
    cudaEvent_t done = nullptr;
    bool in_flight   = false;
};

class UploadDrainGuard {
public:
    explicit UploadDrainGuard(cudaStream_t stream) : stream_(stream) {}

    ~UploadDrainGuard() noexcept {
        if (!armed_) { return; }
        // If the stream cannot be proven quiescent, freeing pinned/device storage is unsafe. Treat
        // this as CUDA-fatal instead of returning a recoverable exception into a poisoned process.
        if (cudaStreamSynchronize(stream_) != cudaSuccess) { std::terminate(); }
    }

    void arm() noexcept { armed_ = true; }

    void disarm() noexcept { armed_ = false; }

private:
    cudaStream_t stream_ = nullptr;
    bool armed_          = false;
};

template <typename Artifact>
void validate_selected_draft_idmap(const Artifact& artifact) {
    const bool selected = artifact.options.load_lm_head_draft;
    if (!selected) { return; }
    const ParsedQ5090Tensor* idmap = nullptr;
    for (const ParsedQ5090Tensor& tensor : artifact.parsed.tensors) {
        if (tensor.module_kind == ModuleKind::LmHeadDraft &&
            tensor.source_kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraftIdmap)) {
            idmap = &tensor;
            break;
        }
    }
    if (idmap == nullptr) { throw std::runtime_error("q5090 selected draft id-map is missing"); }
    std::vector<std::byte> bytes(static_cast<std::size_t>(idmap->payload_bytes));
    artifact.read_exact(idmap->payload_offset, bytes);
    std::set<std::uint32_t> ids;
    for (std::uint32_t i = 0; i < idmap->shape[0]; ++i) {
        std::uint32_t id = 0;
        std::memcpy(&id, bytes.data() + 4ULL * i, sizeof(id));
        if (id >= artifact.parsed.header.vocab_size) {
            throw std::runtime_error("q5090 draft-head id-map contains out-of-range vocab id");
        }
        if (!ids.insert(id).second) {
            throw std::runtime_error("q5090 draft-head id-map contains duplicate id");
        }
    }
}

} // namespace

WeightStore::WeightStore() = default;

WeightStore::~WeightStore() = default;

void WeightStore::prepare(const char* path, const LoadOptions& options) {
    if (options.load_lm_head_draft && !options.load_mtp) {
        throw std::invalid_argument("q5090 LM_HEAD_DRAFT residency requires MTP_DRAFT");
    }
    clear();
    for (std::size_t i = 0; i < stats_.modules.size(); ++i) {
        stats_.modules[i].module = static_cast<ModuleKind>(i);
    }

    auto artifact     = std::make_unique<PreparedArtifact>();
    artifact->options = options;
    if (options.progress != nullptr) {
        artifact->owned_progress.emplace(*options.progress);
        artifact->options.progress = &*artifact->owned_progress;
    }
    stats_.fail_stage = "header";
    artifact->fd      = ::open(path, O_RDONLY | O_CLOEXEC);
    if (artifact->fd < 0) {
        throw std::runtime_error(std::string("failed to open q5090 artifact: ") +
                                 std::strerror(errno));
    }
    if (::fstat(artifact->fd, &artifact->identity) != 0) {
        throw std::runtime_error(std::string("q5090 fstat failed: ") + std::strerror(errno));
    }
    if (artifact->identity.st_size < static_cast<off_t>(kHeaderBytes)) {
        throw std::runtime_error("q5090 file too small for header");
    }
    const std::uint64_t file_size = static_cast<std::uint64_t>(artifact->identity.st_size);

    ProgressReporter reporter(artifact->options.progress);
    std::array<std::byte, kHeaderBytes> header_bytes{};
    const auto header_begin = std::chrono::steady_clock::now();
    reporter.report("read header", 0, kHeaderBytes, true);
    artifact->read_exact(0, header_bytes);
    reporter.report("read header", kHeaderBytes, kHeaderBytes, true);
    const ParsedQ5090Header header = parse_q5090_header(header_bytes, file_size);
    stats_.header_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - header_begin).count();

    stats_.fail_stage = "catalog";
    if (header.payload_offset >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("q5090 catalog size exceeds size_t");
    }
    std::vector<std::byte> metadata(static_cast<std::size_t>(header.payload_offset));
    std::memcpy(metadata.data(), header_bytes.data(), header_bytes.size());
    const std::uint64_t catalog_bytes = header.tokenizer_index_offset - kHeaderBytes;
    const auto catalog_begin          = std::chrono::steady_clock::now();
    reporter.report("read catalog", 0, catalog_bytes, true);
    artifact->read_exact(kHeaderBytes, std::span<std::byte>(metadata).subspan(
                                           static_cast<std::size_t>(kHeaderBytes),
                                           static_cast<std::size_t>(catalog_bytes)));
    reporter.report("read catalog", catalog_bytes, catalog_bytes, true);
    stats_.catalog_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - catalog_begin).count();

    stats_.fail_stage                   = "tokenizer";
    const std::uint64_t tokenizer_bytes = header.payload_offset - header.tokenizer_index_offset;
    const auto tokenizer_begin          = std::chrono::steady_clock::now();
    reporter.report("read tokenizer", 0, tokenizer_bytes, true);
    artifact->read_exact(header.tokenizer_index_offset,
                         std::span<std::byte>(metadata).subspan(
                             static_cast<std::size_t>(header.tokenizer_index_offset),
                             static_cast<std::size_t>(tokenizer_bytes)));
    reporter.report("read tokenizer", tokenizer_bytes, tokenizer_bytes, true);
    artifact->parsed = parse_q5090_catalog(metadata, file_size);
    stats_.tokenizer_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tokenizer_begin).count();
    artifact->verify_identity();
    stats_.header_read_bytes     = kHeaderBytes;
    stats_.catalog_read_bytes    = catalog_bytes;
    stats_.tokenizer_read_bytes  = tokenizer_bytes;
    stats_.total_file_read_bytes = header.payload_offset;

    for (const std::string& required : options.required_text_tensors) {
        if (!contains_text_name(artifact->parsed, required)) {
            throw std::runtime_error("required TEXT_CORE tensor is missing: " + required);
        }
    }

    stats_.fail_stage = "plan";
    Q5090LoadPlan next_plan;
    next_plan.file_read_bytes = header.payload_offset;
    for (const ParsedQ5090Module& module : artifact->parsed.modules) {
        const std::size_t index            = module_index(module.module_kind);
        const bool selected                = should_load(module.module_kind, options);
        Q5090ModuleLoadStats& module_stats = stats_.modules[index];
        module_stats.available             = true;
        module_stats.selected              = selected;
        module_stats.file_bytes            = module.payload_bytes;
        if (!selected) { continue; }
        next_plan.selected[index] = true;
        next_plan.modules.push_back(PlannedQ5090Module{
            module.module_kind, module.payload_offset, module.payload_bytes, module.payload_bytes,
            module.tensor_index_begin, module.tensor_index_count});
        next_plan.file_read_bytes = checked_add(next_plan.file_read_bytes, module.payload_bytes);
        next_plan.h2d_bytes       = checked_add(next_plan.h2d_bytes, module.payload_bytes);
        next_plan.device_bytes    = checked_add(next_plan.device_bytes, module.payload_bytes);
        for (std::uint64_t tensor_index = module.tensor_index_begin;
             tensor_index < module.tensor_index_begin + module.tensor_index_count; ++tensor_index) {
            const ParsedQ5090Tensor& tensor =
                artifact->parsed.tensors[static_cast<std::size_t>(tensor_index)];
            next_plan.tensors.push_back(PlannedQ5090Tensor{
                tensor_index, module.module_kind, tensor.payload_offset, tensor.payload_bytes,
                tensor.payload_offset - module.payload_offset});
        }
        module_stats.h2d_bytes            = 0;
        module_stats.arena_capacity_bytes = module.payload_bytes;
    }
    for (ModuleKind requested : {ModuleKind::TextCore, ModuleKind::MtpDraft,
                                 ModuleKind::VisionEncoder, ModuleKind::LmHeadDraft}) {
        if (!should_load(requested, options)) { continue; }
        if (!stats_.modules[module_index(requested)].available) {
            throw std::runtime_error("requested q5090 module is not available");
        }
    }
    if (options.load_lm_head_draft) {
        for (const ParsedQ5090Tensor& tensor : artifact->parsed.tensors) {
            if (tensor.module_kind == ModuleKind::LmHeadDraft &&
                tensor.source_kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraftIdmap)) {
                next_plan.file_read_bytes =
                    checked_add(next_plan.file_read_bytes, tensor.payload_bytes);
                break;
            }
        }
    }
    plan_     = std::move(next_plan);
    prepared_ = std::move(artifact);
    stats_.fail_stage.clear();
}

void WeightStore::upload(DeviceContext& ctx) {
    if (!prepared_) { throw std::runtime_error("q5090 upload requires a prepared artifact"); }
    PreparedArtifact& artifact = *prepared_;
    ParsedQ5090File& parsed    = artifact.parsed;
    stats_.fail_stage          = "identity";
    artifact.verify_identity();

    std::vector<TensorRecord> next_tensors;
    std::vector<QuantRecord> next_quant;
    std::vector<FusedBlockRecord> next_fused;
    ModuleState next_modules[4]{};
    std::array<std::optional<DeviceArena>, 4> next_arenas;
    std::array<double, 4> upload_seconds{};
    std::size_t next_loaded_payload_bytes = 0;
    std::uint64_t total_file_read_bytes   = parsed.header.payload_offset;
    std::vector<void*> payloads(parsed.tensors.size(), nullptr);
    stats_.fail_stage = "selected payload validation";
    validate_selected_draft_idmap(artifact);
    if (artifact.options.load_lm_head_draft) {
        total_file_read_bytes = plan_.file_read_bytes - plan_.h2d_bytes;
    }

    for (const ParsedQ5090Module& module : parsed.modules) {
        ModuleState& state  = next_modules[module_index(module.module_kind)];
        state.present       = true;
        state.loaded        = plan_.selected[module_index(module.module_kind)];
        state.tensor_count  = module.tensor_index_count;
        state.payload_bytes = module.payload_bytes;
        if (state.loaded) {
            next_loaded_payload_bytes += static_cast<std::size_t>(module.payload_bytes);
        }
    }

    next_tensors.reserve(parsed.tensors.size());
    next_quant.reserve(parsed.segments.size());
    next_fused.reserve(parsed.tensors.size());

    stats_.fail_stage            = "allocate";
    std::uint64_t largest_module = 0;
    for (const PlannedQ5090Module& module : plan_.modules) {
        const std::size_t index = module_index(module.module);
        next_arenas[index].emplace(static_cast<std::size_t>(module.arena_bytes));
        DeviceArena& arena = *next_arenas[index];
        if (reinterpret_cast<std::uintptr_t>(arena.base()) % 256 != 0) {
            throw std::runtime_error("q5090 module arena base is not 256-byte aligned");
        }
        for (const PlannedQ5090Tensor& planned : plan_.tensors) {
            if (planned.module != module.module) { continue; }
            Tensor raw     = alloc_payload(arena, planned.file_bytes);
            auto* expected = static_cast<std::byte*>(arena.base()) + planned.device_offset;
            if (raw.data != expected) {
                throw std::runtime_error("q5090 planned device offset mismatch");
            }
            payloads[static_cast<std::size_t>(planned.tensor_index)] = raw.data;
        }
        if (arena.used() != module.arena_bytes || arena.capacity() != module.arena_bytes) {
            throw std::runtime_error("q5090 exact module arena accounting mismatch");
        }
        largest_module = std::max(largest_module, module.file_bytes);
    }

    stats_.fail_stage = "upload";
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kPinnedSlotBytes, largest_module));
    if (slot_bytes == 0) { throw std::runtime_error("q5090 load plan selected no payload"); }
    std::array<std::unique_ptr<UploadSlot>, kPinnedSlotCount> slots;
    for (auto& slot : slots) { slot = std::make_unique<UploadSlot>(slot_bytes); }
    UploadDrainGuard drain_guard(ctx.load_stream);
    stats_.pinned_slot_count       = kPinnedSlotCount;
    stats_.pinned_slot_bytes       = slot_bytes;
    stats_.host_peak_staging_bytes = kPinnedSlotCount * slot_bytes;
    ProgressReporter upload_reporter(artifact.options.progress);
    std::uint64_t uploaded_total = 0;
    upload_reporter.report("upload selected payloads", 0, plan_.h2d_bytes, true);
    std::size_t next_slot = 0;
    for (const PlannedQ5090Module& module : plan_.modules) {
        const auto module_begin = std::chrono::steady_clock::now();
        DeviceArena& arena      = *next_arenas[module_index(module.module)];
        std::uint64_t copied    = 0;
        while (copied < module.file_bytes) {
            UploadSlot& slot = *slots[next_slot];
            if (slot.in_flight) {
                cuda_throw(cudaEventSynchronize(slot.done),
                           "cudaEventSynchronize(q5090 upload slot)");
                slot.in_flight = false;
            }
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(slot.buffer.size(), module.file_bytes - copied));
            artifact.read_exact(
                module.file_offset + copied,
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), chunk));
            total_file_read_bytes = checked_add(total_file_read_bytes, chunk);
            auto* dst             = static_cast<std::byte*>(arena.base()) + copied;
            cuda_throw(cudaMemcpyAsync(dst, slot.buffer.data(), chunk, cudaMemcpyHostToDevice,
                                       ctx.load_stream),
                       "cudaMemcpyAsync(q5090 staged payload)");
            // The pinned slot and device arena are DMA dependencies immediately after enqueue.
            // Arm before accounting or any other potentially-throwing work.
            drain_guard.arm();
            cuda_throw(cudaEventRecord(slot.done, ctx.load_stream),
                       "cudaEventRecord(q5090 upload slot)");
            slot.in_flight = true;
            copied += chunk;
            uploaded_total += chunk;
            upload_reporter.report("upload selected payloads", uploaded_total, plan_.h2d_bytes);
            next_slot = (next_slot + 1) % slots.size();
        }
        cuda_throw(cudaStreamSynchronize(ctx.load_stream),
                   "cudaStreamSynchronize(q5090 module upload)");
        drain_guard.disarm();
        for (auto& slot : slots) { slot->in_flight = false; }
        upload_seconds[module_index(module.module)] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - module_begin).count();
    }
    upload_reporter.report("upload selected payloads", plan_.h2d_bytes, plan_.h2d_bytes, true);
    stats_.h2d_bytes             = plan_.h2d_bytes;
    stats_.total_file_read_bytes = total_file_read_bytes;
    stats_.device_resident_bytes = plan_.device_bytes;
    for (const PlannedQ5090Module& module : plan_.modules) {
        Q5090ModuleLoadStats& module_stats = stats_.modules[module_index(module.module)];
        module_stats.h2d_bytes             = module.file_bytes;
        module_stats.upload_seconds        = upload_seconds[module_index(module.module)];
    }
    artifact.verify_identity();

    stats_.fail_stage = "publish";
    for (std::size_t tensor_index = 0; tensor_index < parsed.tensors.size(); ++tensor_index) {
        const ParsedQ5090Tensor& tensor = parsed.tensors[tensor_index];
        void* payload                   = payloads[tensor_index];
        if (payload == nullptr) { continue; }

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
                next_quant.push_back(QuantRecord{segment.name, tensor.name, tensor.module_kind,
                                                 segment.source_kind, segment.source_layer,
                                                 segment.row_begin, segment.row_count,
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

    tensors_              = std::move(next_tensors);
    quant_                = std::move(next_quant);
    fused_                = std::move(next_fused);
    total_tensor_count_   = parsed.tensors.size();
    loaded_payload_bytes_ = next_loaded_payload_bytes;
    for (std::size_t i = 0; i < 4; ++i) { modules_[i] = next_modules[i]; }
    module_arenas_ = std::move(next_arenas);
    tokenizer_     = std::move(parsed.tokenizer);
    prepared_.reset();
    stats_.fail_stage.clear();
}

void WeightStore::load(const char* path, DeviceContext& ctx, const LoadOptions& options) {
    prepare(path, options);
    upload(ctx);
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

const Q5090LoadPlan& WeightStore::load_plan() const {
    if (plan_.modules.empty()) { throw std::runtime_error("q5090 load plan is not available"); }
    return plan_;
}

const Q5090LoadStats& WeightStore::load_stats() const noexcept { return stats_; }

const DeviceArena* WeightStore::module_arena(ModuleKind module) const noexcept {
    const auto& arena = module_arenas_[module_index(module)];
    return arena ? &*arena : nullptr;
}

void WeightStore::reset_arena_peaks() noexcept {
    for (auto& arena : module_arenas_) {
        if (arena) { arena->reset_peak(); }
    }
}

Q5090TokenizerBundle WeightStore::take_tokenizer_bundle() {
    if (tokenizer_.empty()) { throw std::runtime_error("q5090 tokenizer bundle is not available"); }
    Q5090TokenizerBundle bundle = std::move(tokenizer_);
    tokenizer_                  = {};
    return bundle;
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
        if (b > std::numeric_limits<std::uint32_t>::max() - a) { throw std::overflow_error(label); }
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
                                std::uint32_t total_n, std::uint32_t k, const char* label) {
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
    auto require_bf16_1d = [&](const char* name, SourceKind source_kind, std::uint32_t source_layer,
                               std::uint32_t size) {
        const TensorRecord& record = require_bf16(name, source_kind, source_layer);
        if (record.tensor.ne[0] < 0 || static_cast<std::uint32_t>(record.tensor.ne[0]) != size ||
            record.tensor.ne[1] != 1 || record.tensor.ne[2] != 1 || record.tensor.ne[3] != 1) {
            throw std::runtime_error(std::string("q5090 invalid MTP BF16 shape: ") + name);
        }
    };

    const QuantRecord& fc = require_w8_segment("mtp.fc.weight", "mtp.fc.weight", SourceKind::MtpFc,
                                               kQ5090NoLayer, "mtp.fc.weight");
    require_segment_range(fc, 0, fc.row_count, "mtp.fc.weight");
    if (fc.row_count == 0) { throw std::runtime_error("q5090 invalid MTP hidden size"); }
    const std::uint32_t hidden = fc.row_count;
    require_equal_i32(fc.weight.k, checked_mul_u32(hidden, 2, "q5090 MTP fc K overflow"),
                      "mtp.fc.weight K");

    require_bf16_1d("mtp.pre_fc_norm_embedding.weight", SourceKind::MtpPreFcNormEmb, kQ5090NoLayer,
                    hidden);
    require_bf16_1d("mtp.pre_fc_norm_hidden.weight", SourceKind::MtpPreFcNormHid, kQ5090NoLayer,
                    hidden);
    require_bf16_1d("mtp.layers.0.input_layernorm.weight", SourceKind::InputLayernorm, 0, hidden);

    const char* attn_block = "mtp.layers.0.attn_in.w8";
    const QuantRecord& attn_q =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.q_proj.q", SourceKind::AttnQ, 0,
                           "mtp.layers.0.attn_in.w8 ATTN_Q");
    const std::uint32_t q_rows = attn_q.row_count;
    require_segment_range(attn_q, 0, q_rows, "mtp.layers.0.attn_in.w8 ATTN_Q");
    require_equal_i32(attn_q.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_k =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.k_proj.weight", SourceKind::AttnK, 0,
                           "mtp.layers.0.attn_in.w8 ATTN_K");
    const std::uint32_t k_rows = attn_k.row_count;
    require_segment_range(attn_k, q_rows, k_rows, "mtp.layers.0.attn_in.w8 ATTN_K");
    require_equal_i32(attn_k.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_gate =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.q_proj.gate", SourceKind::AttnGate,
                           0, "mtp.layers.0.attn_in.w8 ATTN_GATE");
    const std::uint32_t gate_begin =
        checked_add_u32(q_rows, k_rows, "q5090 MTP attn gate row overflow");
    require_segment_range(attn_gate, gate_begin, q_rows, "mtp.layers.0.attn_in.w8 ATTN_GATE");
    require_equal_i32(attn_gate.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");

    const QuantRecord& attn_v =
        require_w8_segment(attn_block, "mtp.layers.0.self_attn.v_proj.weight", SourceKind::AttnV, 0,
                           "mtp.layers.0.attn_in.w8 ATTN_V");
    const std::uint32_t v_begin =
        checked_add_u32(gate_begin, q_rows, "q5090 MTP attn v row overflow");
    require_segment_range(attn_v, v_begin, k_rows, "mtp.layers.0.attn_in.w8 ATTN_V");
    require_equal_i32(attn_v.weight.k, hidden, "mtp.layers.0.attn_in.w8 K");
    const std::uint32_t attn_total =
        checked_add_u32(v_begin, k_rows, "q5090 MTP attn total row overflow");
    require_fused_w8(attn_block, /*ATTN_IN*/ 1, attn_total, hidden, "mtp.layers.0.attn_in.w8");

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
    require_bf16_1d("mtp.layers.0.post_attention_layernorm.weight", SourceKind::PostAttnLayernorm,
                    0, hidden);

    const char* gateup_block = "mtp.layers.0.mlp.gateup.w8";
    const QuantRecord& mlp_gate =
        require_w8_segment(gateup_block, "mtp.layers.0.mlp.gate_proj.weight", SourceKind::MlpGate,
                           0, "mtp.layers.0.mlp.gateup.w8 MLP_GATE");
    const std::uint32_t intermediate = mlp_gate.row_count;
    require_segment_range(mlp_gate, 0, intermediate, "mtp.layers.0.mlp.gateup.w8 MLP_GATE");
    require_equal_i32(mlp_gate.weight.k, hidden, "mtp.layers.0.mlp.gateup.w8 K");

    const QuantRecord& mlp_up =
        require_w8_segment(gateup_block, "mtp.layers.0.mlp.up_proj.weight", SourceKind::MlpUp, 0,
                           "mtp.layers.0.mlp.gateup.w8 MLP_UP");
    require_segment_range(mlp_up, intermediate, intermediate, "mtp.layers.0.mlp.gateup.w8 MLP_UP");
    require_equal_i32(mlp_up.weight.k, hidden, "mtp.layers.0.mlp.gateup.w8 K");
    require_fused_w8(gateup_block, /*MLP_GATEUP*/ 3,
                     checked_mul_u32(intermediate, 2, "q5090 MTP gateup row overflow"), hidden,
                     "mtp.layers.0.mlp.gateup.w8");

    const QuantRecord& down =
        require_w8_segment("mtp.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
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
    prepared_.reset();
    tensors_.clear();
    quant_.clear();
    fused_.clear();
    for (auto& arena : module_arenas_) { arena.reset(); }
    plan_      = {};
    stats_     = {};
    tokenizer_ = {};
    for (ModuleState& module : modules_) { module = ModuleState{}; }
    total_tensor_count_   = 0;
    loaded_payload_bytes_ = 0;
}

} // namespace ninfer
