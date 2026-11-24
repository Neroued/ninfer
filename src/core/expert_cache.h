#pragma once

#include "core/device.h"
#include "core/layout.h"
#include "core/weight_bank.h"

#include <memory>
#include <optional>

namespace ninfer {

struct ExpertBankCacheLayout {
    ExpertMatrixEntryLayout gate_up, down;
    std::size_t entry_bytes   = 0;
    std::size_t payload_bytes = 0;
    std::size_t first_entry   = 0;
    std::uint32_t count       = 0;
    bool resident             = false;
    LayoutRegion offsets, gate_up_divisors, down_divisors;

    friend bool operator==(const ExpertBankCacheLayout&, const ExpertBankCacheLayout&) = default;
};

struct ExpertCacheConfig {
    std::size_t capacity_bytes       = 0;
    std::size_t fixed_bytes          = 0;
    std::size_t max_batch_bytes      = 96 * 1024 * 1024;
    std::size_t max_batch_copies     = 4096;
    std::uint32_t upload_slots       = 3;
    std::uint32_t max_in_flight_uses = 8;

    friend bool operator==(const ExpertCacheConfig&, const ExpertCacheConfig&) = default;
};

struct ExpertCacheRequirements {
    ExpertCacheConfig config;
    std::vector<ExpertBankCacheLayout> banks;
    std::size_t entry_count   = 0;
    std::size_t offset_bytes  = 0;
    std::size_t divisor_bytes = 0;
    std::size_t pinned_bytes  = 0;

    friend bool operator==(const ExpertCacheRequirements&,
                           const ExpertCacheRequirements&) = default;
};

struct ExpertBankCacheSpec {
    WeightBankLayout gate_up, down;
    bool resident = false;
};

class ExpertCache;

struct ExpertCacheUse {
    const ExpertCache* owner = nullptr;
    std::uint32_t slot       = 0;
    std::uint64_t generation = 0;
};

struct ExpertCacheProgress {
    bool required_published       = false;
    bool prefetch_ready           = false;
    bool waiting_for_uploads      = false;
    std::size_t submitted_batches = 0;
    std::size_t submitted_bytes   = 0;
};

struct ExpertCacheRequest {
    std::span<const ExpertKey> required;
    std::span<const ExpertKey> prefetch;
};

struct ExpertPlacement {
    ExpertKey key;
    std::size_t offset = 0; // Relative to the containing region; 256-byte aligned.
};

struct ExpertUploadBudget {
    std::size_t max_bytes   = SIZE_MAX;
    std::size_t max_batches = 4;
};

struct ExpertCacheUsage {
    std::size_t used_bytes        = 0;
    std::size_t published_experts = 0;
    std::uint64_t data_bytes = 0, index_bytes = 0;
    std::size_t in_flight_bytes   = 0;
    std::uint64_t data_copy_calls = 0, index_copy_calls = 0;
    std::uint64_t planning_nanoseconds = 0, submit_nanoseconds = 0;
    std::uint64_t prefetched_bytes = 0;
    std::size_t ready_cache_bytes  = 0;
    std::vector<std::uint32_t> ready_experts_per_bank;
};

[[nodiscard]] ExpertCacheRequirements
describe_expert_cache(std::span<const ExpertBankSource> sources, const ExpertCacheConfig& config);
[[nodiscard]] ExpertCacheRequirements
describe_expert_cache(std::span<const ExpertBankCacheSpec> banks, const ExpertCacheConfig& config);

// Program-owned, one Host writer. Sources and DeviceContext outlive the cache.
// Upload and consumer views use explicit non-null streams, which also outlive the cache.
// All handoffs run outside capture. Every acquired use must be released, including resident banks.
// Fixed entries occupy [0, fixed_bytes) for the owner's lifetime. replace rebuilds an explicit
// mutable region, ordering writes after its released readers. It never moves retained entries.
// prepare selects already placed entries; progress submits bounded uploads without GPU waits.
// acquire protects exactly its keys and adds the upload dependency to the consumer stream.
// release follows the last real use, including Graph replay, on that same stream.
class ExpertCache {
public:
    ExpertCache(std::span<const ExpertBankSource> sources,
                const ExpertCacheRequirements& requirements, const DeviceContext& device,
                CudaExecutionView upload, std::span<const ExpertPlacement> fixed = {});
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    void replace(LayoutRegion region, std::span<const ExpertPlacement> entries);
    void prepare(const ExpertCacheRequest& request);
    ExpertCacheProgress progress(ExpertUploadBudget budget = {});
    void ready_keys(ExpertBankId bank, std::vector<ExpertKey>& keys);
    void wait_for_progress();
    // nullopt means unpublished weights or a pending released use occupying the bounded ring.
    [[nodiscard]] std::optional<ExpertCacheUse> acquire(std::span<const ExpertKey> required,
                                                        CudaExecutionView execution);
    void release(ExpertCacheUse use, CudaExecutionView execution);
    // Only experts protected by a live use may be consumed. Encoded planes are 16-byte aligned.
    [[nodiscard]] const ExpertBankDeviceView& view(ExpertBankId bank) const;
    [[nodiscard]] ExpertCacheUsage usage() const;
    [[nodiscard]] std::uint64_t uploaded_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
