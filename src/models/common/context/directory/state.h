#pragma once

#include "core/transfer_work.h"
#include "models/common/context/directory/records.h"
#include "models/common/context/directory/metadata.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::context {

// IDs are scoped to one execution session. Only the directory owns their allocation.
struct StateId {
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return generation != 0; }

    friend bool operator==(StateId, StateId) = default;
};

enum class StateRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
    Writing,
};

enum class StateResidency : std::uint8_t {
    None,
    DeviceOnly,
    HostOnly,
    Both,
};

enum class StateCopyDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
};

struct StateSelectors {
    std::int64_t source      = -1;
    std::int64_t destination = -1;
};

struct StatePartSpec {
    std::uint32_t rank                          = 0;
    std::size_t image_bytes                     = 0;
    std::size_t alignment                       = 256;
    bool operator==(const StatePartSpec&) const = default;
};

struct StateLayoutSpec {
    std::uint32_t variant = 0;
    std::vector<StatePartSpec> parts;
    bool operator==(const StateLayoutSpec&) const = default;
};

struct StateDirectorySpec {
    std::uint32_t logical_capacity = 0;
    std::vector<StateLayoutSpec> layouts;
    bool operator==(const StateDirectorySpec&) const = default;
};

[[nodiscard]] std::uint32_t state_record_layout(std::uint32_t variant);

struct StateTransferDescription {
    StateCopyDirection direction = StateCopyDirection::DeviceToHost;
    StateId source{};
    StateId destination{};
    ContextRecordBinding source_record;
    ContextRecordBinding destination_record;
    std::uint64_t epoch = 0;
};

// Owns logical images and rank-local record replicas. Physical storage and execution
// are external; callers retain leases and transfers until their device consumers have retired.
class StateDirectory : public MetadataOwner {
public:
    class ReadLease {
    public:
        ReadLease() noexcept = default;
        ~ReadLease();
        ReadLease(ReadLease&& other) noexcept;
        ReadLease& operator=(ReadLease&& other) noexcept;
        ReadLease(const ReadLease&)            = delete;
        ReadLease& operator=(const ReadLease&) = delete;

        [[nodiscard]] StateId image() const noexcept { return image_; }

        [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

        [[nodiscard]] std::int64_t device_offset(std::uint32_t rank = 0) const;
        [[nodiscard]] ContextRecordBinding host_binding(std::uint32_t rank = 0) const;
        void reset() noexcept;

    private:
        ReadLease(StateDirectory& directory, StateId image, std::uint64_t epoch,
                  ContextMemory memory, ContextUse use) noexcept
            : directory_(&directory), image_(image), epoch_(epoch), memory_(memory), use_(use) {}

        StateDirectory* directory_ = nullptr;
        StateId image_;
        std::uint64_t epoch_  = 0;
        ContextMemory memory_ = ContextMemory::Device;
        ContextUse use_       = ContextUse::Cache;
        friend class StateDirectory;
    };

    // Owns a destination until the native continuation adopts it.
    class Destination {
    public:
        Destination(StateDirectory& directory, StateId image);
        ~Destination();
        Destination(Destination&& other) noexcept;
        Destination& operator=(Destination&&)      = delete;
        Destination(const Destination&)            = delete;
        Destination& operator=(const Destination&) = delete;

        [[nodiscard]] StateId image() const noexcept { return image_; }

        [[nodiscard]] StateSelectors begin_write(StateId source, std::uint32_t rank = 0);
        [[nodiscard]] StateId release() noexcept;

    private:
        StateDirectory* directory_ = nullptr;
        StateId image_;
        StateId write_source_;
    };

    // Pins the source and owns the destination reservation. Publish only after copy completion;
    // destruction rolls back the reservation and requires all physical consumers to have retired.
    class Transfer {
    public:
        Transfer() noexcept = default;
        ~Transfer();
        Transfer(Transfer&& other) noexcept;
        Transfer& operator=(Transfer&&)      = delete;
        Transfer(const Transfer&)            = delete;
        Transfer& operator=(const Transfer&) = delete;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

        [[nodiscard]] StateCopyDirection direction() const noexcept { return direction_; }

        [[nodiscard]] StateTransferDescription description(std::uint32_t rank = 0) const;
        [[nodiscard]] std::vector<std::uint32_t> ranks() const;
        [[nodiscard]] TransferWork work() const;
        void publish(bool keep_source_replica);

    private:
        Transfer(StateDirectory& owner, std::uint64_t id, StateCopyDirection direction,
                 ReadLease source, StateId destination) noexcept;

        StateDirectory* owner_        = nullptr;
        std::uint64_t id_             = 0;
        StateCopyDirection direction_ = StateCopyDirection::DeviceToHost;
        ReadLease source_;
        StateId destination_;
        std::vector<RecordLease> records_;
        friend class StateDirectory;
    };

    StateDirectory(StateDirectorySpec spec, RecordDomains& records, MetadataBudget& metadata);
    ~StateDirectory() override = default;

    [[nodiscard]] static std::size_t metadata_bytes(const StateDirectorySpec& spec);
    [[nodiscard]] static std::size_t object_metadata_bytes(std::size_t parts);
    [[nodiscard]] std::size_t allocated_metadata_bytes() const;
    [[nodiscard]] bool reserve_metadata(std::uint32_t count);
    [[nodiscard]] std::size_t metadata_requirement(std::uint32_t count) const;
    [[nodiscard]] std::uint32_t available_descriptors() const noexcept;
    [[nodiscard]] MetadataReservation reserve_descriptors(std::uint32_t count);
    void release_metadata(std::span<const std::uint32_t> ids) noexcept override;

    StateDirectory(const StateDirectory&)            = delete;
    StateDirectory& operator=(const StateDirectory&) = delete;
    StateDirectory(StateDirectory&&)                 = delete;
    StateDirectory& operator=(StateDirectory&&)      = delete;

    [[nodiscard]] std::size_t image_bytes(std::uint32_t rank = 0, std::uint32_t variant = 0) const;
    [[nodiscard]] std::vector<std::uint32_t> ranks(std::uint32_t variant = 0) const;
    [[nodiscard]] std::uint32_t variant(StateId image) const;
    [[nodiscard]] std::optional<StateId> image_for_allocation(AllocationId allocation) const;
    [[nodiscard]] ReadLease pin_binding(StateId image);
    [[nodiscard]] std::size_t reclaimable_metadata(std::span<const StateId> released = {}) const;
    [[nodiscard]] std::size_t reclaim_metadata();

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t occupied() const noexcept;
    [[nodiscard]] std::uint32_t device_occupied() const noexcept;
    [[nodiscard]] std::uint32_t host_occupied() const noexcept;
    [[nodiscard]] std::optional<StateId> reserve_destination(RecordReservation reservation,
                                                             std::uint32_t variant = 0);
    [[nodiscard]] std::optional<StateId> reserve_destination(std::uint32_t variant = 0);
    [[nodiscard]] std::optional<StateId> reserve_logical_destination(std::uint32_t variant = 0);
    [[nodiscard]] std::optional<StateId> reserve_logical_destination(RecordReservation& reservation,
                                                                     std::uint32_t variant = 0);

    // Adopting initialized or copied content preserves its epoch; native writes advance it.
    // Publication relies on the caller's execution dependency, not a synchronous device wait.
    void complete_initialization(StateId image);
    // The caller owns the completed field mapping and retains the immutable source through copy
    // retirement. A layout conversion adopts the source epoch without a mathematical write.
    void complete_conversion(const ReadLease& source, StateId destination);

    [[nodiscard]] bool valid(StateId image) const noexcept;
    [[nodiscard]] std::uint32_t descriptor_index(StateId image) const;
    [[nodiscard]] StateRole role(StateId image) const;
    [[nodiscard]] StateResidency residency(StateId image) const;
    [[nodiscard]] bool device_resident(StateId image) const;
    [[nodiscard]] bool host_resident(StateId image) const;
    [[nodiscard]] std::uint64_t content_epoch(StateId image) const;
    [[nodiscard]] std::uint32_t source_pins(StateId image) const;
    [[nodiscard]] std::uint32_t checkpoint_references(StateId image) const;
    [[nodiscard]] std::int64_t physical_offset(StateId image, std::uint32_t rank = 0) const;
    [[nodiscard]] ContextRecordBinding device_binding(StateId image, std::uint32_t rank = 0) const;
    [[nodiscard]] ContextRecordBinding host_binding(StateId image, std::uint32_t rank = 0) const;
    [[nodiscard]] std::vector<RecordRelease> allocations(StateId image) const;
    [[nodiscard]] ReadLease read(StateId image, ContextMemory memory = ContextMemory::Device,
                                 ContextUse use = ContextUse::Request);

    void retain_checkpoint_reference(StateId image);

    // Last-owner release is deferred while readers or transfers still borrow the image.
    [[nodiscard]] bool release_checkpoint_owner(StateId image) noexcept;

    void make_mutable(StateId image);
    void consume_checkpoint(StateId image);
    [[nodiscard]] bool drop_device_replica(StateId image) noexcept;
    [[nodiscard]] bool drop_host_replica(StateId image) noexcept;

    // A Both source retains its immutable Host identity; the destination adopts its Device records.
    void split_device_replica_identity(StateId source, StateId destination);

    void freeze(StateId image);
    [[nodiscard]] StateSelectors begin_write(StateId source, StateId destination,
                                             std::uint32_t rank = 0);
    void complete_write(StateId source, StateId destination);
    void abort_write(StateId source, StateId destination);
    [[nodiscard]] bool can_abort_write(StateId source, StateId destination) const noexcept;
    [[nodiscard]] StateSelectors selectors(StateId source, StateId destination,
                                           std::uint32_t rank = 0) const;

    [[nodiscard]] std::optional<Transfer> reserve_device_to_host(StateId source,
                                                                 RecordReservation reserved = {});

    [[nodiscard]] std::optional<Transfer> reserve_host_to_device(StateId source);
    [[nodiscard]] std::optional<Transfer>
    reserve_host_to_device(StateId source, StateId destination, RecordReservation reserved = {});
    [[nodiscard]] bool release(StateId image) noexcept;
    [[nodiscard]] bool can_release(StateId image) const noexcept;
    [[nodiscard]] bool
    can_release_after_checkpoint_references(StateId image, std::uint32_t released_references) const;

private:
    struct Object {
        std::uint32_t generation    = 0;
        std::uint32_t variant       = 0;
        std::uint64_t content_epoch = 0;

        struct Part {
            std::uint32_t rank = 0;
            std::optional<RecordLease> device;
            std::optional<RecordLease> host;
        };

        std::vector<Part> parts;
        std::uint64_t transfer_id           = 0;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t source_pins           = 0;
        std::array<std::uint32_t, 2> request_references{};
        StateId write_source{};
        bool destination_pinned = false;
        bool release_pending    = false;
        StateRole role          = StateRole::Free;
    };

    void validate_records(const std::vector<RecordLease>& records, ContextMemory memory,
                          std::uint32_t variant) const;
    [[nodiscard]] const StateLayoutSpec& layout_spec(std::uint32_t variant) const;
    void retire_released_owner(StateId image) noexcept;
    static void refresh_usage(Object&) noexcept;
    void retain_request_reference(StateId image, ContextMemory memory);
    void release_request_reference(StateId image, ContextMemory memory) noexcept;
    template <class Metadata, class Identity>
    friend class ContextSnapshot;
    [[nodiscard]] bool can_release_checkpoint_owner(StateId image) const noexcept;
    [[nodiscard]] static bool has_pending_replica(const Object& object) noexcept;
    [[nodiscard]] std::optional<StateId> allocate(std::uint32_t index, StateRole role,
                                                  std::uint32_t variant,
                                                  std::vector<RecordLease> records = {});
    [[nodiscard]] static Object::Part& part(Object&, std::uint32_t rank);
    [[nodiscard]] static const Object::Part& part(const Object&, std::uint32_t rank);
    [[nodiscard]] std::uint64_t next_epoch() noexcept;
    [[nodiscard]] std::uint64_t next_transfer() noexcept;
    [[nodiscard]] Object& require(StateId image);
    [[nodiscard]] const Object& require(StateId image) const;
    [[nodiscard]] bool valid_transfer(const Transfer& transfer) const noexcept;
    void validate_transfer(const Transfer& transfer) const;
    void publish_transfer(Transfer& transfer, bool keep_source_replica);
    void abort_transfer(Transfer& transfer) noexcept;
    static void consume(Transfer& transfer) noexcept;

    StateDirectorySpec spec_;
    RecordDomains* records_ = nullptr;
    MetadataAccount metadata_account_;
    MetadataBudget* metadata_budget_;
    std::size_t max_parts_ = 0;
    DirectorySlots<Object> objects_;
    std::uint32_t initialized_objects_ = 0;

    std::vector<std::uint32_t> free_objects_;
    std::uint32_t free_object_count_  = 0;
    std::uint32_t device_images_      = 0; // Reserved transfers and published replicas.
    std::uint32_t host_images_        = 0; // Reserved transfers and published replicas.
    std::uint32_t next_generation_    = 0;
    std::uint64_t next_content_epoch_ = 0;
    std::uint64_t next_transfer_id_   = 0;
};

template <class Metadata>
struct Continuation {
    StateId image;
    Metadata metadata;
};

} // namespace ninfer::models::context
