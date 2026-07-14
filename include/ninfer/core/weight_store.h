#pragma once

#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/tensor.h"
#include "ninfer/core/weight_store_parser.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

struct LoadOptions {
    bool load_mtp           = false;
    bool load_vision        = false;
    bool load_lm_head_draft = false;
    Q5090Progress* progress = nullptr;
    std::vector<std::string> required_text_tensors;
};

struct PlannedQ5090Module {
    ModuleKind module                = ModuleKind::TextCore;
    std::uint64_t file_offset        = 0;
    std::uint64_t file_bytes         = 0;
    std::uint64_t arena_bytes        = 0;
    std::uint64_t tensor_index_begin = 0;
    std::uint64_t tensor_index_count = 0;
};

struct PlannedQ5090Tensor {
    std::uint64_t tensor_index  = 0;
    ModuleKind module           = ModuleKind::TextCore;
    std::uint64_t file_offset   = 0;
    std::uint64_t file_bytes    = 0;
    std::uint64_t device_offset = 0;
};

struct Q5090LoadPlan {
    std::vector<PlannedQ5090Module> modules;
    std::vector<PlannedQ5090Tensor> tensors;
    std::array<bool, 4> selected{};
    std::uint64_t file_read_bytes = 0;
    std::uint64_t h2d_bytes       = 0;
    std::uint64_t device_bytes    = 0;
};

struct Q5090ModuleLoadStats {
    ModuleKind module                  = ModuleKind::TextCore;
    bool available                     = false;
    bool selected                      = false;
    std::uint64_t file_bytes           = 0;
    std::uint64_t h2d_bytes            = 0;
    std::uint64_t arena_capacity_bytes = 0;
    double upload_seconds              = 0.0;
};

struct Q5090LoadStats {
    // Byte counters describe the latest prepare/upload attempt. Payload reads and H2D enqueue
    // operations are accounted incrementally, including work completed before an exception.
    std::uint64_t header_read_bytes       = 0;
    std::uint64_t catalog_read_bytes      = 0;
    std::uint64_t tokenizer_read_bytes    = 0;
    std::uint64_t total_file_read_bytes   = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t host_peak_staging_bytes = 0;
    std::uint64_t device_resident_bytes   = 0;
    std::uint32_t pinned_slot_count       = 0;
    std::uint64_t pinned_slot_bytes       = 0;
    double header_seconds                 = 0.0;
    double catalog_seconds                = 0.0;
    double tokenizer_seconds              = 0.0;
    std::array<Q5090ModuleLoadStats, 4> modules{};
    std::string fail_stage;
};

struct FusedBlockRecord {
    std::string name;
    ModuleKind module          = ModuleKind::TextCore;
    std::uint16_t group_id     = 0;
    std::uint16_t fusion_index = 0;
    std::uint32_t source_layer = kQ5090NoLayer;
    Weight weight;
};

class WeightStore {
public:
    WeightStore();
    ~WeightStore();

    WeightStore(const WeightStore&)            = delete;
    WeightStore& operator=(const WeightStore&) = delete;

    // CPU-only phase. Opens one stable fd, reads/validates Header first, then the bounded
    // catalog/tokenizer region, and derives the exact residency plan. No CUDA call is made.
    void prepare(const char* path, const LoadOptions& options = {});
    // GPU phase. Allocates one exact arena per selected module and transactionally publishes
    // descriptors only after staged uploads and the final file-identity check succeed.
    void upload(DeviceContext& ctx);
    void load(const char* path, DeviceContext& ctx, const LoadOptions& options = {});

    const Tensor* tensor(std::string_view name) const noexcept;
    const Weight* qweight(std::string_view name) const noexcept;
    const Tensor* tensor(ModuleKind module, std::uint32_t source_kind,
                         std::uint32_t source_layer) const noexcept;
    const Weight* qweight(ModuleKind module, std::uint32_t source_kind,
                          std::uint32_t source_layer) const noexcept;
    const Weight* qfused(ModuleKind module, std::uint16_t group_id, std::uint16_t fusion_index,
                         std::uint32_t source_layer) const noexcept;

    std::size_t tensor_count() const noexcept;
    std::size_t quant_count() const noexcept;
    std::size_t loaded_payload_bytes() const noexcept;
    std::size_t module_tensor_count(ModuleKind module) const noexcept;
    bool module_loaded(ModuleKind module) const noexcept;
    const Q5090LoadPlan& load_plan() const;
    const Q5090LoadStats& load_stats() const noexcept;
    const DeviceArena* module_arena(ModuleKind module) const noexcept;
    void reset_arena_peaks() noexcept;
    Q5090TokenizerBundle take_tokenizer_bundle();
    void require_mtp_module_expectations() const;
    void clear() noexcept;

private:
    struct TensorRecord {
        std::string name;
        ModuleKind module           = ModuleKind::TextCore;
        std::uint32_t source_kind   = 0;
        std::uint32_t source_layer  = kQ5090NoLayer;
        QType qtype                 = QType::BF16_CTRL;
        std::uint64_t payload_bytes = 0;
        Tensor tensor;
    };

    struct QuantRecord {
        std::string name;
        std::string block_name;
        ModuleKind module          = ModuleKind::TextCore;
        std::uint32_t source_kind  = 0;
        std::uint32_t source_layer = kQ5090NoLayer;
        std::uint32_t row_begin    = 0;
        std::uint32_t row_count    = 0;
        Weight weight;
    };

    struct ModuleState {
        bool present                = false;
        bool loaded                 = false;
        std::uint64_t tensor_count  = 0;
        std::uint64_t payload_bytes = 0;
    };

    struct PreparedArtifact;
    std::unique_ptr<PreparedArtifact> prepared_;
    std::vector<TensorRecord> tensors_;
    std::vector<QuantRecord> quant_;
    std::vector<FusedBlockRecord> fused_;
    ModuleState modules_[4];
    std::array<std::optional<DeviceArena>, 4> module_arenas_;
    Q5090LoadPlan plan_;
    Q5090LoadStats stats_;
    Q5090TokenizerBundle tokenizer_;
    std::size_t total_tensor_count_   = 0;
    std::size_t loaded_payload_bytes_ = 0;
};

} // namespace ninfer
