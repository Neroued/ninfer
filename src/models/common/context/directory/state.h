#pragma once

#include "core/transfer_work.h"
#include "models/common/context/directory/allocation.h"

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
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

struct StateDirectorySpec {
    std::uint32_t logical_capacity = 0;
    std::uint32_t device_slots     = 0;
    std::size_t image_bytes        = 0;
};

struct StateTransferDescription {
    StateCopyDirection direction = StateCopyDirection::DeviceToHost;
    StateId source{};
    StateId destination{};
    StateSelectors slots{};
    HostExtent host{};
    std::size_t bytes   = 0;
    std::uint64_t epoch = 0;
};

// Owns logical images, device slot assignments and Host extents. Physical storage and execution
// are external; callers retain leases and transfers until their device consumers have retired.
class StateDirectory {
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

        [[nodiscard]] std::int32_t device_slot() const;
        [[nodiscard]] HostExtent host_extent() const;
        void reset() noexcept;

    private:
        ReadLease(StateDirectory& directory, StateId image, std::uint64_t epoch) noexcept
            : directory_(&directory), image_(image), epoch_(epoch) {}

        StateDirectory* directory_ = nullptr;
        StateId image_;
        std::uint64_t epoch_ = 0;
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

        [[nodiscard]] StateSelectors begin_write(StateId source);
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

        [[nodiscard]] StateTransferDescription description() const;
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
        std::optional<std::int32_t> device_slot_;
        std::optional<HostAllocation> host_slot_;
        friend class StateDirectory;
    };

    StateDirectory(StateDirectorySpec spec, HostAllocator* host);
    StateDirectory(const StateDirectory&)            = delete;
    StateDirectory& operator=(const StateDirectory&) = delete;
    StateDirectory(StateDirectory&&)                 = delete;
    StateDirectory& operator=(StateDirectory&&)      = delete;

    [[nodiscard]] std::size_t image_bytes() const noexcept { return image_bytes_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t occupied() const noexcept;
    [[nodiscard]] std::uint32_t device_occupied() const noexcept;
    [[nodiscard]] std::uint32_t device_capacity() const noexcept;
    [[nodiscard]] std::uint32_t host_occupied() const noexcept;
    [[nodiscard]] std::optional<StateId> reserve_destination() noexcept;
    [[nodiscard]] std::optional<StateId> reserve_logical_destination() noexcept;

    // Adopting initialized or copied content preserves its epoch; native writes advance it.
    // Publication relies on the caller's execution dependency, not a synchronous device wait.
    void complete_initialization(StateId image);

    [[nodiscard]] bool valid(StateId image) const noexcept;
    [[nodiscard]] std::uint32_t descriptor_index(StateId image) const;
    [[nodiscard]] StateRole role(StateId image) const;
    [[nodiscard]] StateResidency residency(StateId image) const;
    [[nodiscard]] bool device_resident(StateId image) const;
    [[nodiscard]] bool host_resident(StateId image) const;
    [[nodiscard]] std::uint64_t content_epoch(StateId image) const;
    [[nodiscard]] std::uint32_t source_pins(StateId image) const;
    [[nodiscard]] std::uint32_t checkpoint_references(StateId image) const;
    [[nodiscard]] std::int32_t physical_slot(StateId image) const;
    [[nodiscard]] HostExtent host_extent(StateId image) const;
    [[nodiscard]] ReadLease read(StateId image);

    void retain_checkpoint_reference(StateId image);

    // Last-owner release is deferred while readers or transfers still borrow the image.
    [[nodiscard]] bool release_checkpoint_owner(StateId image) noexcept;

    void make_mutable(StateId image);
    void consume_checkpoint(StateId image);
    [[nodiscard]] bool drop_device_replica(StateId image) noexcept;
    [[nodiscard]] bool drop_host_replica(StateId image) noexcept;

    // A Both source retains its immutable Host identity; the destination adopts its device slot.
    void split_device_replica_identity(StateId source, StateId destination);

    void freeze(StateId image);
    [[nodiscard]] StateSelectors begin_write(StateId source, StateId destination);
    void complete_write(StateId source, StateId destination);
    void abort_write(StateId source, StateId destination);
    [[nodiscard]] bool can_abort_write(StateId source, StateId destination) const noexcept;
    [[nodiscard]] StateSelectors selectors(StateId source, StateId destination) const;

    [[nodiscard]] std::optional<Transfer> reserve_device_to_host(StateId source,
                                                                 HostAllocation reserved = {});

    [[nodiscard]] std::optional<Transfer> reserve_host_to_device(StateId source);
    [[nodiscard]] std::optional<Transfer> reserve_host_to_device(StateId source,
                                                                 StateId destination);
    [[nodiscard]] bool release(StateId image) noexcept;
    [[nodiscard]] bool can_release(StateId image) const noexcept;
    [[nodiscard]] bool
    can_release_after_checkpoint_references(StateId image, std::uint32_t released_references) const;

private:
    struct Object {
        std::uint32_t generation    = 1;
        std::uint64_t content_epoch = 0;
        std::optional<std::int32_t> device_slot{};
        std::optional<HostAllocation> host_slot{};
        std::uint64_t transfer_id           = 0;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t source_pins           = 0;
        StateId write_source{};
        bool destination_pinned = false;
        bool release_pending    = false;
        StateRole role          = StateRole::Free;
    };

    void retire_released_owner(StateId image) noexcept;
    [[nodiscard]] bool can_release_checkpoint_owner(StateId image) const noexcept;
    [[nodiscard]] static bool has_pending_replica(const Object& object) noexcept;
    [[nodiscard]] std::optional<StateId> allocate(StateRole role, bool with_device) noexcept;
    [[nodiscard]] std::optional<std::int32_t> take_device_slot() noexcept;
    void return_device_slot(std::int32_t slot) noexcept;
    [[nodiscard]] std::uint64_t next_epoch() noexcept;
    [[nodiscard]] std::uint64_t next_transfer() noexcept;
    [[nodiscard]] Object& require(StateId image);
    [[nodiscard]] const Object& require(StateId image) const;
    [[nodiscard]] bool valid_transfer(const Transfer& transfer) const noexcept;
    void validate_transfer(const Transfer& transfer) const;
    void publish_transfer(Transfer& transfer, bool keep_source_replica);
    void abort_transfer(Transfer& transfer) noexcept;
    void release_host_slot(std::optional<HostAllocation>& slot) noexcept;
    static void consume(Transfer& transfer) noexcept;

    std::size_t image_bytes_;
    HostAllocator* host_ = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::vector<std::int32_t> free_device_slots_;
    std::uint32_t free_object_count_  = 0;
    std::uint32_t free_device_count_  = 0;
    std::uint32_t host_images_        = 0; // Reserved transfers and published replicas.
    std::uint64_t next_content_epoch_ = 0;
    std::uint64_t next_transfer_id_   = 0;
};

template <class Metadata>
struct Continuation {
    StateId image;
    Metadata metadata;
};

} // namespace ninfer::models::context
