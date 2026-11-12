#pragma once

#include "models/common/context/recovery.h"
#include "models/common/context/directory.h"
#include "runtime/contract/recovery.h"

namespace ninfer::models::context {

inline std::vector<runtime::ContextTransferRequirement>
maintenance_work(const ContextAssessment& assessment) {
    std::vector<runtime::ContextTransferRequirement> result;
    for (const auto& domain : assessment.placement.domains) {
        if (!domain.placement.move_count) continue;
        runtime::ContextTransferRequirement work;
        work.direction = domain.domain.memory == ContextMemory::Device
                             ? runtime::ContextTransferDirection::DeviceToDevice
                             : runtime::ContextTransferDirection::HostToHost;
        work.rank      = domain.domain.rank;
        work.work = {domain.placement.move_bytes, static_cast<std::uint32_t>(std::min<std::size_t>(
                                                      UINT32_MAX, domain.placement.move_count))};
        result.push_back(work);
    }
    return result;
}

inline std::vector<runtime::ContextTransferRequirement>
demotion_work(const ContextDirectory& directory, const ContextReleaseSet& released,
              bool restore = false) {
    auto unique = released;
    unique.normalize();
    std::vector<runtime::ContextTransferRequirement> result;
    for (const auto& record : unique.records) {
        if (record.domain.memory != ContextMemory::Device) continue;
        const auto rank = record.domain.rank;
        runtime::ContextTransferRequirement work;
        work.direction = restore ? runtime::ContextTransferDirection::HostToDevice
                                 : runtime::ContextTransferDirection::DeviceToHost;
        work.rank      = rank;
        if (const auto image = directory.states().image_for_allocation(record.allocation)) {
            const auto allocations = directory.states().allocations(*image);
            if (!restore &&
                std::any_of(allocations.begin(), allocations.end(), [&](const auto& copy) {
                    return copy.domain.rank == rank && copy.domain.memory == ContextMemory::Host;
                }))
                continue;
            work.image_count = 1;
            work.work        = {directory.states().device_binding(*image, rank).bytes, 1};
            result.push_back(work);
            continue;
        }
        for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
            const auto& pages = directory.pool(pool).pages();
            const auto page   = pages.page_for_allocation(record.allocation);
            if (!page) continue;
            if (restore || !pages.host_covers(*page, rank, pages.committed_columns(*page))) {
                work.resource.pool = runtime::KvPoolId{pool};
                work.page_groups   = 1;
                work.work          = {pages.physical(*page, rank).bytes, 1};
                result.push_back(work);
            }
            break;
        }
    }
    return result;
}

inline std::vector<runtime::ContextTransferRequirement>
host_write_work(const ContextDirectory& directory, const HostWriteRequirements& write) {
    std::vector<runtime::ContextTransferRequirement> out;
    if (write.state_images)
        for (auto rank : directory.states().ranks(write.state_variant))
            out.push_back({.direction   = runtime::ContextTransferDirection::DeviceToHost,
                           .image_count = write.state_images,
                           .work        = {write.state_images *
                                               directory.states().image_bytes(rank, write.state_variant),
                                           write.state_images},
                           .rank        = rank});
    for (std::uint16_t pool = 0; pool < write.kv_pages.size(); ++pool) {
        const auto count = write.kv_pages[pool];
        if (!count) continue;
        for (const auto& part : directory.pool(pool).pages().layout().parts)
            out.push_back({.resource    = {runtime::KvPoolId{pool}},
                           .direction   = runtime::ContextTransferDirection::DeviceToHost,
                           .page_groups = count,
                           .work        = {count * part.record.bytes, count},
                           .rank        = part.rank});
    }
    return out;
}

// Filter physical preparation releases (unused writer/capture destinations) from the
// contents that a paused snapshot actually needs on restore.
inline std::vector<runtime::ContextTransferRequirement>
pause_restore_work(const ContextDirectory& directory, StateId image,
                   std::span<const HistoryId> histories, const ContextReleaseSet& released) {
    auto contents = directory.state_release(image);
    for (std::uint16_t pool = 0; pool < histories.size(); ++pool) {
        const auto& mapping = directory.pool(pool).histories();
        for (std::uint32_t page = 0; page < mapping.mapped_pages(histories[pool]); ++page)
            contents.merge(directory.page_release(mapping.logical_page(histories[pool], page)));
    }
    std::erase_if(contents.records, [&](const auto& record) {
        return std::find(released.records.begin(), released.records.end(), record) ==
               released.records.end();
    });
    return demotion_work(directory, contents, true);
}

// Charge owned quote metadata for exactly as long as a decision or evaluation retains it.
struct RecoveryDescription final : runtime::RecoveryDescription {
    RecoveryDescription(MetadataBudget& budget, runtime::RecoveryDescription value)
        : runtime::RecoveryDescription(std::move(value)), account(budget) {
        auto bytes = sizeof(*this) + restores.capacity() * sizeof(runtime::RecoveryRestore);
        for (const auto& restore : restores)
            bytes += (restore.transfers.capacity() + restore.replay_transfers.capacity()) *
                     sizeof(runtime::ContextTransferRequirement);
        account.acquire(bytes);
    }

    MetadataAccount account;
};

inline std::shared_ptr<const runtime::RecoveryDescription>
recovery_description(MetadataBudget& budget, runtime::RecoveryDescription value) {
    return std::make_shared<RecoveryDescription>(budget, std::move(value));
}

// Descriptor and workspace limits are checked by their concrete implementations.

inline std::size_t recovery_value_storage_bytes(std::size_t capacity, std::size_t concurrency,
                                                std::size_t ranks = 1, std::size_t placements = 1) {
    if (!concurrency) return 0;
    // Required service can add up to one newly paused target per resident request.
    const auto records = std::max<std::size_t>(1, capacity) + concurrency;
    const auto native  = sizeof(RecoveryDescription) +
                        sizeof(runtime::RecoveryPoint<CheckpointHandle>) +
                        runtime::kRecoveryIdentityMetadataLimit +
                        placements * (sizeof(runtime::RecoveryRestore) +
                                      12 * ranks * sizeof(runtime::ContextTransferRequirement));
    const auto priced = runtime::kRecoveryPricedDescriptorLimit +
                        placements * sizeof(runtime::RecoveryRestoreCost) +
                        runtime::kRecoveryIdentityMetadataLimit +
                        2 * sizeof(runtime::RecoveryPoint<CheckpointHandle>) +
                        2 * sizeof(CheckpointHandle);
    // Capture and execution may both wait per lane; admission/pause and synchronous price
    // views add a bounded overlap. Raw transfer inventories exist only during quotation.
    return records * (2 * concurrency + runtime::kRecoveryAuxiliaryViews) * priced +
           (2 * records + concurrency + 3) * native +
           (2 * records + 1) * runtime::kRecoveryEvaluationEntryLimit +
           RecoveryIndex<runtime::RecoveryHandle>::metadata_bytes(2 * records);
}

inline std::size_t recovery_value_metadata_bytes(std::size_t records, std::size_t concurrency,
                                                 std::size_t ranks      = 1,
                                                 std::size_t placements = 1) {
    return recovery_value_storage_bytes(records + 255, concurrency, ranks, placements);
}

// The checkpoint store reserves this workspace together with descriptor growth, before
// publication. Catalog entries cannot spend the last bytes needed to evaluate their eviction.
class RecoveryMetadata final : public MetadataCapacity {
public:
    RecoveryMetadata(MetadataBudget& owner, std::size_t concurrency, std::size_t ranks = 1,
                     std::size_t placements = 1, std::size_t identity_bytes = 0)
        : budget_(recovery_value_storage_bytes(0, concurrency, ranks, placements), &owner),
          identities_(identity_bytes, &owner), concurrency_(concurrency), ranks_(ranks),
          placements_(placements) {}

    std::size_t metadata_requirement(std::size_t capacity) const override {
        const auto bytes = required(capacity);
        return bytes > budget_.limit() ? bytes - budget_.limit() : 0;
    }

    void reserve_metadata(std::size_t capacity) override { budget_.grow(required(capacity)); }

    void trim_metadata(std::size_t capacity) noexcept override {
        budget_.trim_when_idle(required(capacity));
    }

    MetadataBudget& budget() const { return budget_; }

    MetadataBudget& identity_budget() const { return identities_; }
private:
    std::size_t required(std::size_t capacity) const {
        return recovery_value_storage_bytes(capacity, concurrency_, ranks_, placements_);
    }

    mutable MetadataBudget budget_;
    mutable MetadataBudget identities_;
    std::size_t concurrency_, ranks_, placements_;
};

} // namespace ninfer::models::context
