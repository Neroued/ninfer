#include "models/common/context/storage.h"

#include "core/platform.h"
#include "core/transfer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::context {
namespace {

void require_execution(Device device, CudaExecutionView execution) {
    if (device != execution.device()) {
        throw std::invalid_argument("Context storage requires its CUDA execution device");
    }
}

void validate_staging(ConstMemorySpan staging, std::size_t bytes) {
    if (bytes == 0) return;
    validate_memory(staging);
    if (staging.device != Device::cpu() || !staging.host_pinned || staging.bytes < bytes ||
        reinterpret_cast<std::uintptr_t>(staging.data) % alignof(std::int64_t)) {
        throw std::invalid_argument("Context mapping staging requires a sufficient pinned range");
    }
}

bool host_only(const ContextAction& action) noexcept {
    if (const auto* clear = std::get_if<RecordClear>(&action)) {
        return clear->destination.memory == ContextMemory::Host;
    }
    if (const auto* copy = std::get_if<RecordCopy>(&action)) {
        return copy->source.memory == ContextMemory::Host &&
               copy->destination.memory == ContextMemory::Host;
    }
    return false;
}

void validate_copy(ContextRange source, ContextRange destination) {
    if (source.bytes != destination.bytes) {
        throw std::invalid_argument("Context copy ranges must have equal byte lengths");
    }
    if (source.memory == destination.memory && source.offset != destination.offset) {
        const auto distance = source.offset < destination.offset
                                  ? destination.offset - source.offset
                                  : source.offset - destination.offset;
        if (static_cast<std::uint64_t>(distance) < source.bytes) {
            throw std::invalid_argument("Context copy records must not overlap");
        }
    }
}

} // namespace

std::size_t host_context_capacity(std::optional<std::size_t> requested, std::size_t image_bytes) {
    if constexpr (NINFER_SHARED_PHYSICAL_MEMORY != 0) {
        if (requested.value_or(0) != 0) {
            throw std::invalid_argument(
                "Host Context storage is not supported on this target; use 0B or default");
        }
        return 0;
    }
    if (requested) return *requested;
    constexpr std::size_t base = 8ULL << 30;
    if (image_bytes > (std::numeric_limits<std::size_t>::max() - base) / 8) {
        throw std::overflow_error("Host Context default capacity overflow");
    }
    return base + 8 * image_bytes;
}

ExecutionTableStorage::ExecutionTableStorage(MemorySpan backing,
                                             const KVExecutionTableLayout& layout,
                                             MemorySpan context_arena, std::size_t record_bytes)
    : spec_(layout.spec), page_offsets_(layout.page_offsets.bind(backing)),
      context_arena_(context_arena), record_bytes_(record_bytes) {
    validate_memory(context_arena_);
    if (backing.device.type() != DeviceType::CUDA || backing.device != context_arena_.device) {
        throw std::invalid_argument("Execution tables require their Context arena's CUDA device");
    }
    if (record_bytes_ == 0 || record_bytes_ > context_arena_.bytes ||
        spec_.logical_page_capacity == 0 || spec_.logical_page_capacity > INT32_MAX ||
        spec_.table_rows <= 0 || page_offsets_.dtype != DType::I64 ||
        page_offsets_.ne[0] != static_cast<std::int32_t>(spec_.logical_page_capacity) ||
        page_offsets_.ne[1] != spec_.table_rows || page_offsets_.ne[2] != 1 ||
        page_offsets_.ne[3] != 1 || !page_offsets_.is_contiguous()) {
        throw std::invalid_argument("Execution table layout has inconsistent geometry");
    }
}

Tensor ExecutionTableStorage::row(std::int32_t index) const {
    if (index < 0 || index >= row_count()) {
        throw std::out_of_range("Execution table row is out of range");
    }
    return page_offsets_.slice(1, index, 1)
        .view({static_cast<std::int32_t>(logical_page_capacity())});
}

void ExecutionTableStorage::validate(std::int32_t row_index, std::uint32_t logical_begin,
                                     std::span<const std::int64_t> offsets) const {
    if (row_index < 0 || row_index >= row_count() || logical_begin > logical_page_capacity() ||
        offsets.size() > logical_page_capacity() - logical_begin) {
        throw std::out_of_range("Context mapping exceeds its execution table row");
    }
    for (const auto offset : offsets) {
        if (offset == -1) continue;
        if (offset < 0 ||
            static_cast<std::uint64_t>(offset) > context_arena_.bytes - record_bytes_) {
            throw std::out_of_range("Context mapping names an out-of-range record");
        }
    }
}

void ExecutionTableStorage::publish(std::int32_t row_index, std::uint32_t logical_begin,
                                    std::span<const std::int64_t> offsets,
                                    MemorySpan pinned_staging, CudaExecutionView execution) const {
    require_execution(page_offsets_.device, execution);
    validate(row_index, logical_begin, offsets);
    validate_staging(pinned_staging, offsets.size_bytes());
    if (offsets.empty()) return;
    std::memmove(pinned_staging.data, offsets.data(), offsets.size_bytes());
    const auto target = row(row_index);
    MemorySpan destination{static_cast<std::int64_t*>(target.data) + logical_begin,
                           offsets.size_bytes(), target.device};
    copy_bytes_async(destination, pinned_staging.subspan(0, offsets.size_bytes()), execution);
}

ContextStorage::ContextStorage(MemorySpan device_arena, MemorySpan host_arena,
                               std::span<const ContextTableStorage> tables,
                               std::uint32_t logical_rank)
    : logical_rank_(logical_rank), device_arena_(device_arena), host_arena_(host_arena),
      tables_(tables.begin(), tables.end()) {
    validate_memory(device_arena_);
    validate_memory(host_arena_);
    if (device_arena_.device.type() != DeviceType::CUDA || device_arena_.bytes > INT64_MAX) {
        throw std::invalid_argument("Context Device arena requires I64-addressable CUDA storage");
    }
    if (host_arena_.bytes && (host_arena_.device != Device::cpu() || !host_arena_.host_pinned ||
                              host_arena_.bytes > INT64_MAX)) {
        throw std::invalid_argument("Context Host arena requires I64-addressable pinned storage");
    }
    std::sort(tables_.begin(), tables_.end(),
              [](const auto& left, const auto& right) { return left.pool_id < right.pool_id; });
    for (std::size_t i = 0; i < tables_.size(); ++i) {
        const auto& table = tables_[i];
        if (!table.tables || table.tables->matrix().device != device_arena_.device ||
            table.tables->context_arena_.data != device_arena_.data ||
            table.tables->context_arena_.bytes != device_arena_.bytes ||
            (i != 0 && tables_[i - 1].pool_id == table.pool_id)) {
            throw std::invalid_argument("Context execution tables are inconsistent");
        }
        const auto elements = static_cast<std::size_t>(table.tables->logical_page_capacity()) *
                              static_cast<std::size_t>(table.tables->row_count());
        if (elements > SIZE_MAX / sizeof(std::int64_t) - mapping_capacity_) {
            throw std::overflow_error("Context mapping capacity overflow");
        }
        mapping_capacity_ += elements;
    }
}

MemorySpan ContextStorage::range(ContextRange record) const {
    if (record.offset < 0) throw std::out_of_range("Context record offset is negative");
    const MemorySpan* backing = nullptr;
    switch (record.memory) {
    case ContextMemory::Device:
        backing = &device_arena_;
        break;
    case ContextMemory::Host:
        backing = &host_arena_;
        break;
    default:
        throw std::invalid_argument("Unknown Context memory domain");
    }
    return backing->subspan(static_cast<std::size_t>(record.offset), record.bytes);
}

const ExecutionTableStorage& ContextStorage::tables(std::uint32_t pool_id) const {
    const auto found =
        std::lower_bound(tables_.begin(), tables_.end(), pool_id,
                         [](const auto& table, auto id) { return table.pool_id < id; });
    if (found == tables_.end() || found->pool_id != pool_id) {
        throw std::out_of_range("Context execution table is not present on this rank");
    }
    return *found->tables;
}

std::size_t ContextStorage::mapping_capacity_elements() const noexcept { return mapping_capacity_; }

std::size_t ContextStorage::copy_segment_capacity() const noexcept {
    std::size_t result = 4 * (tables_.size() + 1);
    for (const auto& table : tables_) result += 4 * table.tables->row_count();
    return result;
}

std::size_t ContextStorage::mapping_elements(const ContextActions& actions) {
    std::size_t result = 0;
    for (const auto& action : actions.operations) {
        if (const auto* update = std::get_if<MappingUpdate>(&action)) {
            if (update->offsets.size() > SIZE_MAX / sizeof(std::int64_t) - result) {
                throw std::overflow_error("Context mapping staging size overflow");
            }
            result += update->offsets.size();
        }
    }
    return result;
}

bool ContextStorage::uses_cuda(const ContextActions& actions) noexcept {
    return std::any_of(actions.operations.begin(), actions.operations.end(),
                       [](const auto& action) { return !host_only(action); });
}

runtime::ContextTransferObservation ContextStorage::describe(const ContextAction& action) const {
    const auto* copy = std::get_if<RecordCopy>(&action);
    if (!copy || copy->source == copy->destination || copy->source.bytes == 0) return {};
    using Direction = runtime::ContextTransferDirection;
    const auto direction =
        copy->source.memory == ContextMemory::Device
            ? (copy->destination.memory == ContextMemory::Device ? Direction::DeviceToDevice
                                                                 : Direction::DeviceToHost)
            : (copy->destination.memory == ContextMemory::Device ? Direction::HostToDevice
                                                                 : Direction::HostToHost);
    return {.resource    = copy->resource,
            .direction   = direction,
            .image_count = copy->image_count,
            .page_groups = copy->page_groups,
            .work        = {copy->source.bytes, 1},
            .rank        = logical_rank_};
}

void ContextStorage::validate(const ContextActions& actions, ConstMemorySpan staging) const {
    validate_staging(staging, mapping_elements(actions) * sizeof(std::int64_t));
    bool host_actions = false;
    bool cuda_actions = false;
    for (const auto& action : actions.operations) {
        host_actions |= host_only(action);
        cuda_actions |= !host_only(action);
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, RecordClear>) {
                    (void)range(value.destination);
                } else if constexpr (std::is_same_v<T, RecordCopy>) {
                    (void)range(value.source);
                    (void)range(value.destination);
                    validate_copy(value.source, value.destination);
                } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                    tables(value.pool_id).validate(value.row, value.logical_begin, value.offsets);
                }
            },
            action);
    }
    if (host_actions && cuda_actions) {
        throw std::invalid_argument("Host Context maintenance requires a separate retired batch");
    }
}

std::size_t ContextStorage::execute(const ContextAction& action, MemorySpan staging,
                                    CudaExecutionView execution) const {
    require_execution(device(), execution);
    return std::visit(
        [&](const auto& value) -> std::size_t {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, RecordClear>) {
                const auto destination = range(value.destination);
                if (destination.bytes == 0) return 0;
                if (value.destination.memory == ContextMemory::Host) {
                    std::memset(destination.data, 0, destination.bytes);
                } else {
                    CudaDeviceScope scope(execution.device());
                    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes,
                                               execution.stream()));
                }
            } else if constexpr (std::is_same_v<T, RecordCopy>) {
                if (value.source == value.destination || value.source.bytes == 0) return 0;
                const auto source      = range(value.source);
                const auto destination = range(value.destination);
                if (host_only(action)) {
                    copy_bytes(destination, source);
                } else {
                    copy_bytes_async(destination, source, execution);
                }
            } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                tables(value.pool_id)
                    .publish(value.row, value.logical_begin, value.offsets, staging, execution);
                return value.offsets.size() * sizeof(std::int64_t);
            }
            return 0;
        },
        action);
}

void ContextStorage::execute(const ContextActions& actions, MemorySpan staging,
                             CudaExecutionView execution) const {
    validate(actions, staging);
    for (const auto& action : actions.operations) {
        const auto consumed = execute(action, staging, execution);
        if (consumed) staging = staging.subspan(consumed);
    }
}

} // namespace ninfer::models::context
