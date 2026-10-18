#include "models/common/context/storage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::models::context {
namespace {

void require_execution(Device device, CudaExecutionView execution) {
    if (device != execution.device()) {
        throw std::invalid_argument("Context storage requires its CUDA execution device");
    }
}

void add_work(TransferWork& total, TransferWork part) {
    if (part.payload_bytes > UINT64_MAX - total.payload_bytes ||
        part.copy_operations > UINT32_MAX - total.copy_operations) {
        throw std::overflow_error("KV transfer work overflow");
    }
    total.payload_bytes += part.payload_bytes;
    total.copy_operations += part.copy_operations;
}

std::size_t run_end(std::span<const std::int32_t> pages, std::size_t begin) {
    std::size_t end = begin + 1;
    while (end < pages.size() && pages[end] == pages[end - 1] + 1) { ++end; }
    return end;
}

} // namespace

DevicePageStorage::DevicePageStorage(MemorySpan backing, const DeviceKVPagePoolLayout& layout)
    : spec_(layout.spec) {
    if (backing.device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("Device page storage requires CUDA backing");
    }
    if (spec_.page_group_count == 0 || spec_.page_group_count > INT32_MAX ||
        spec_.geometry.page_tokens != kPagedKVPageSize || layout.planes.empty() ||
        layout.planes.size() != spec_.geometry.planes.size()) {
        throw std::invalid_argument("Device page storage has inconsistent geometry");
    }
    const auto physical_pages = static_cast<std::int32_t>(spec_.page_group_count);
    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const auto& planned  = layout.planes[index];
        const auto& expected = geometry().planes[index];
        if (planned.geometry != expected || expected.leading_extent <= 0 ||
            expected.head_extent <= 0) {
            throw std::invalid_argument("Device page plane layout disagrees with its geometry");
        }
        Tensor plane = planned.storage.bind(backing);
        if (plane.dtype != expected.dtype || plane.ne[0] != expected.leading_extent ||
            plane.ne[1] != kv_plane_rows_per_page(geometry(), expected)) {
            throw std::invalid_argument("Device page plane tensor has inconsistent geometry");
        }
        const bool page_major = geometry().device_plane_order == PagedKVPlaneOrder::PageMajor;
        if (plane.ne[2] != (page_major ? expected.head_extent : physical_pages) ||
            plane.ne[3] != (page_major ? physical_pages : expected.head_extent)) {
            throw std::invalid_argument("Device page plane tensor has inconsistent ordering");
        }
        planes_.push_back(plane);
    }
    host_layout_ = plan_host_kv_page_layout(geometry());
    initialize_host_transfer_plan();
    validation_marks_.resize(capacity_pages(), 0);
}

void DevicePageStorage::initialize_host_transfer_plan() {
    int maximum_pitch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maximum_pitch, cudaDevAttrMaxPitch, device().index()));
    const auto pitch_limit = static_cast<std::size_t>(maximum_pitch);
    const bool page_major  = geometry().device_plane_order == PagedKVPlaneOrder::PageMajor;
    std::size_t maximum_short_dimension = 1;
    host_transfer_planes_.reserve(planes_.size());
    for (std::size_t index = 0; index < planes_.size(); ++index) {
        const auto& packed = host_layout_.planes[index];
        const auto& plane  = planes_[index];
        HostTransferPlane transfer{
            .host_offset = packed.offset,
            .page_bytes  = packed.page_payload_bytes,
            .head_bytes  = packed.head_payload_bytes,
            .heads       = static_cast<std::uint32_t>(geometry().planes[index].head_extent),
            .page_copies_supported = !page_major &&
                                     static_cast<std::size_t>(plane.nb[3]) <= pitch_limit &&
                                     packed.head_payload_bytes <= pitch_limit,
        };
        page_payload_bytes_ += packed.page_payload_bytes;
        if (transfer.page_copies_supported) {
            maximum_short_dimension =
                std::max(maximum_short_dimension, static_cast<std::size_t>(transfer.heads));
        }
        host_transfer_planes_.push_back(transfer);
    }
    if (page_major) {
        host_transfer_groups_.reserve(planes_.size());
        for (std::size_t first = 0; first < planes_.size();) {
            HostTransferGroup group{.first_plane = first};
            const auto width = host_transfer_planes_[first].page_bytes;
            for (std::size_t next = first + 1; next < planes_.size(); ++next) {
                if (host_transfer_planes_[next].page_bytes != width ||
                    planes_[next].nb[3] != planes_[first].nb[3]) {
                    break;
                }
                const auto current  = reinterpret_cast<std::uintptr_t>(planes_[next].data);
                const auto previous = reinterpret_cast<std::uintptr_t>(planes_[next - 1].data);
                if (current <= previous) { break; }
                const std::size_t device_pitch = current - previous;
                const std::size_t host_pitch   = host_transfer_planes_[next].host_offset -
                                               host_transfer_planes_[next - 1].host_offset;
                if (device_pitch < width || device_pitch > pitch_limit || host_pitch < width ||
                    host_pitch > pitch_limit) {
                    break;
                }
                if (group.plane_count > 1 &&
                    (group.device_pitch != device_pitch || group.host_pitch != host_pitch)) {
                    break;
                }
                group.device_pitch = device_pitch;
                group.host_pitch   = host_pitch;
                ++group.plane_count;
            }
            maximum_host_plane_group_ = std::max(maximum_host_plane_group_, group.plane_count);
            host_transfer_groups_.push_back(group);
            first += group.plane_count;
        }
        maximum_short_dimension = maximum_host_plane_group_;
    }

    host_transfer_operations_.resize(maximum_short_dimension + 1, 0);
    for (std::size_t pages = 1; pages <= maximum_short_dimension; ++pages) {
        std::uint64_t operations = 0;
        if (page_major) {
            for (const auto& group : host_transfer_groups_) {
                operations += group.copy_by_page(pages) ? pages : group.plane_count;
            }
        } else {
            for (const auto& plane : host_transfer_planes_) {
                operations += plane.copy_by_page(pages) ? pages : plane.heads;
            }
        }
        if (operations > UINT32_MAX) {
            throw std::overflow_error("Host KV transfer operation count exceeds uint32");
        }
        host_transfer_operations_[pages] = static_cast<std::uint32_t>(operations);
    }
}

void DevicePageStorage::validate_page(std::int32_t page) const {
    if (page < 0 || static_cast<std::uint32_t>(page) >= capacity_pages()) {
        throw std::out_of_range("Device page index is out of range");
    }
}

void DevicePageStorage::validate_distinct_pages(std::span<const std::int32_t> pages) const {
    ++validation_stamp_;
    if (validation_stamp_ == 0) {
        std::fill(validation_marks_.begin(), validation_marks_.end(), 0);
        validation_stamp_ = 1;
    }
    for (const auto page : pages) {
        validate_page(page);
        auto& stamp = validation_marks_[static_cast<std::size_t>(page)];
        if (stamp == validation_stamp_) {
            throw std::invalid_argument("Device page destinations contain duplicates");
        }
        stamp = validation_stamp_;
    }
}

void DevicePageStorage::validate_host_span(ConstMemorySpan memory, const HostKVPageLayout& layout,
                                           std::size_t pages) const {
    validate_memory(memory);
    if (layout != host_layout_ || memory.device != Device::cpu() || !memory.host_pinned ||
        pages > SIZE_MAX / host_layout_.page_stride ||
        memory.bytes != pages * host_layout_.page_stride) {
        throw std::invalid_argument("Host KV transfer requires matching pinned records");
    }
}

TransferWork DevicePageStorage::host_transfer_run_work(std::uint32_t pages) const {
    if (pages == 0) { return {}; }
    if (page_payload_bytes_ > UINT64_MAX / pages) {
        throw std::overflow_error("Host KV transfer payload overflow");
    }
    return {.payload_bytes   = page_payload_bytes_ * pages,
            .copy_operations = host_transfer_operations_[std::min<std::size_t>(
                pages, host_transfer_operations_.size() - 1)]};
}

TransferWork DevicePageStorage::host_transfer_work(std::span<const std::int32_t> pages) const {
    for (const auto page : pages) { validate_page(page); }
    TransferWork result;
    for (std::size_t begin = 0; begin < pages.size();) {
        const auto end = run_end(pages, begin);
        add_work(result, host_transfer_run_work(static_cast<std::uint32_t>(end - begin)));
        begin = end;
    }
    return result;
}

TransferWork DevicePageStorage::device_copy_work(std::uint32_t pages) const {
    if (pages == 0) { return {}; }
    if (page_payload_bytes_ > UINT64_MAX / pages || planes_.size() > UINT32_MAX / pages) {
        throw std::overflow_error("Device KV copy work overflow");
    }
    return {.payload_bytes   = page_payload_bytes_ * pages,
            .copy_operations = static_cast<std::uint32_t>(planes_.size()) * pages};
}

void DevicePageStorage::zero_pages(std::span<const std::int32_t> pages,
                                   CudaExecutionView execution) const {
    require_execution(device(), execution);
    validate_distinct_pages(pages);
    CudaDeviceScope scope(execution.device());
    for (std::size_t begin = 0; begin < pages.size();) {
        const auto end   = run_end(pages, begin);
        const auto first = pages[begin];
        const auto count = end - begin;
        for (const Tensor& plane : planes_) {
            auto* base = static_cast<std::byte*>(plane.data);
            if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                           count * plane.nb[3], execution.stream()));
            } else {
                CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                             plane.nb[3], 0, count * plane.nb[2], plane.ne[3],
                                             execution.stream()));
            }
        }
        begin = end;
    }
}

TransferWork DevicePageStorage::copy_page(std::int32_t source, std::int32_t destination,
                                          CudaExecutionView execution) const {
    require_execution(device(), execution);
    validate_page(source);
    validate_page(destination);
    if (source == destination) { return {}; }
    const auto work = device_copy_work(1);
    CudaDeviceScope scope(execution.device());
    for (const Tensor& plane : planes_) {
        auto* base = static_cast<std::byte*>(plane.data);
        if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
            CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::int64_t>(destination) * plane.nb[3],
                                       base + static_cast<std::int64_t>(source) * plane.nb[3],
                                       plane.nb[3], cudaMemcpyDeviceToDevice, execution.stream()));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                base + static_cast<std::int64_t>(destination) * plane.nb[2], plane.nb[3],
                base + static_cast<std::int64_t>(source) * plane.nb[2], plane.nb[3], plane.nb[2],
                plane.ne[3], cudaMemcpyDeviceToDevice, execution.stream()));
        }
    }
    return work;
}

template <bool ToHost>
TransferWork
DevicePageStorage::copy_host_pages(std::span<const std::int32_t> pages,
                                   std::conditional_t<ToHost, std::byte*, const std::byte*> host,
                                   CudaExecutionView execution) const {
    // Validate the complete quotation before enqueueing any transfer.
    const auto expected = host_transfer_work(pages);
    TransferWork work;
    CudaDeviceScope scope(execution.device());
    const auto copy = [&](std::byte* device_base, std::size_t device_pitch, auto* host_base,
                          std::size_t host_pitch, std::size_t width, std::size_t height) {
        if constexpr (ToHost) {
            CUDA_CHECK(cudaMemcpy2DAsync(host_base, host_pitch, device_base, device_pitch, width,
                                         height, cudaMemcpyDeviceToHost, execution.stream()));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(device_base, device_pitch, host_base, host_pitch, width,
                                         height, cudaMemcpyHostToDevice, execution.stream()));
        }
        ++work.copy_operations;
    };
    const auto host_stride = host_layout_.page_stride;
    for (std::size_t begin = 0; begin < pages.size();) {
        const auto end   = run_end(pages, begin);
        const auto count = end - begin;
        const auto first = pages[begin];
        if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
            if (count < maximum_host_plane_group_) {
                for (const auto& group : host_transfer_groups_) {
                    if (group.copy_by_page(count)) {
                        const auto& plane  = planes_[group.first_plane];
                        const auto& packed = host_transfer_planes_[group.first_plane];
                        for (std::size_t page = begin; page < end; ++page) {
                            copy(static_cast<std::byte*>(plane.data) +
                                     static_cast<std::int64_t>(pages[page]) * plane.nb[3],
                                 group.device_pitch, host + page * host_stride + packed.host_offset,
                                 group.host_pitch, packed.page_bytes, group.plane_count);
                        }
                    } else {
                        for (std::size_t index = group.first_plane;
                             index < group.first_plane + group.plane_count; ++index) {
                            const auto& plane  = planes_[index];
                            const auto& packed = host_transfer_planes_[index];
                            copy(static_cast<std::byte*>(plane.data) +
                                     static_cast<std::int64_t>(first) * plane.nb[3],
                                 plane.nb[3], host + begin * host_stride + packed.host_offset,
                                 host_stride, packed.page_bytes, count);
                        }
                    }
                }
            } else {
                for (std::size_t index = 0; index < planes_.size(); ++index) {
                    const auto& plane  = planes_[index];
                    const auto& packed = host_transfer_planes_[index];
                    copy(static_cast<std::byte*>(plane.data) +
                             static_cast<std::int64_t>(first) * plane.nb[3],
                         plane.nb[3], host + begin * host_stride + packed.host_offset, host_stride,
                         packed.page_bytes, count);
                }
            }
        } else {
            for (std::size_t index = 0; index < planes_.size(); ++index) {
                const auto& plane  = planes_[index];
                const auto& packed = host_transfer_planes_[index];
                auto* device_base  = static_cast<std::byte*>(plane.data);
                auto* host_base    = host + begin * host_stride + packed.host_offset;
                if (packed.copy_by_page(count)) {
                    for (std::size_t page = begin; page < end; ++page) {
                        copy(device_base + static_cast<std::int64_t>(pages[page]) * plane.nb[2],
                             plane.nb[3], host + page * host_stride + packed.host_offset,
                             packed.head_bytes, packed.head_bytes, packed.heads);
                    }
                } else {
                    for (std::uint32_t head = 0; head < packed.heads; ++head) {
                        copy(device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                                 static_cast<std::int64_t>(first) * plane.nb[2],
                             plane.nb[2], host_base + head * packed.head_bytes, host_stride,
                             packed.head_bytes, count);
                    }
                }
            }
        }
        work.payload_bytes += page_payload_bytes_ * count;
        begin = end;
    }
    if (work != expected) { throw std::logic_error("KV transfer disagrees with its work plan"); }
    return work;
}

TransferWork DevicePageStorage::copy_to_host(std::span<const std::int32_t> source,
                                             MemorySpan destination, const HostKVPageLayout& layout,
                                             CudaExecutionView execution) const {
    require_execution(device(), execution);
    validate_host_span(destination, layout, source.size());
    return copy_host_pages<true>(source, static_cast<std::byte*>(destination.data), execution);
}

TransferWork DevicePageStorage::copy_from_host(ConstMemorySpan source,
                                               const HostKVPageLayout& layout,
                                               std::span<const std::int32_t> destination,
                                               CudaExecutionView execution) const {
    require_execution(device(), execution);
    validate_host_span(source, layout, destination.size());
    validate_distinct_pages(destination);
    return copy_host_pages<false>(destination, static_cast<const std::byte*>(source.data),
                                  execution);
}

ExecutionTableStorage::ExecutionTableStorage(MemorySpan backing,
                                             const KVExecutionTableLayout& layout,
                                             const DevicePageStorage& pages)
    : spec_(layout.spec), block_tables_(layout.block_tables.bind(backing)),
      page_capacity_(pages.capacity_pages()) {
    if (backing.device != pages.device()) {
        throw std::invalid_argument("Execution tables must share their page storage device");
    }
    if (spec_.logical_page_capacity == 0 || spec_.logical_page_capacity > INT32_MAX ||
        spec_.table_rows <= 0 || block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] != static_cast<std::int32_t>(spec_.logical_page_capacity) ||
        block_tables_.ne[1] != spec_.table_rows || block_tables_.ne[2] != 1 ||
        block_tables_.ne[3] != 1) {
        throw std::invalid_argument("Execution table layout has inconsistent geometry");
    }
}

Tensor ExecutionTableStorage::row(std::int32_t index) const {
    if (index < 0 || index >= row_count()) {
        throw std::out_of_range("Execution table row is out of range");
    }
    return block_tables_.slice(1, index, 1)
        .view({static_cast<std::int32_t>(logical_page_capacity())});
}

void ExecutionTableStorage::publish(std::int32_t row_index, std::uint32_t logical_begin,
                                    std::span<const std::int32_t> pages, MemorySpan pinned_staging,
                                    CudaExecutionView execution) const {
    require_execution(block_tables_.device, execution);
    const auto destination_row = row(row_index);
    if (logical_begin > logical_page_capacity() ||
        pages.size() > logical_page_capacity() - logical_begin) {
        throw std::out_of_range("Mapping publication exceeds its execution table row");
    }
    for (const auto page : pages) {
        if (page < 0 || static_cast<std::uint32_t>(page) >= page_capacity_) {
            throw std::out_of_range("Mapping publication names an out-of-range device page");
        }
    }
    if (pages.empty()) { return; }
    validate_memory(pinned_staging);
    if (pinned_staging.device != Device::cpu() || !pinned_staging.host_pinned ||
        pinned_staging.bytes < pages.size_bytes()) {
        throw std::invalid_argument("Mapping publication requires caller-owned pinned staging");
    }
    std::memmove(pinned_staging.data, pages.data(), pages.size_bytes());
    auto* destination = static_cast<std::int32_t*>(destination_row.data) + logical_begin;
    CudaDeviceScope scope(execution.device());
    CUDA_CHECK(cudaMemcpyAsync(destination, pinned_staging.data, pages.size_bytes(),
                               cudaMemcpyHostToDevice, execution.stream()));
}

ContextStorage::ContextStorage(StateImageStorage& state, std::span<const ContextPoolStorage> pools,
                               std::uint32_t host_pool_id, MemorySpan host_backing)
    : state_(&state), pools_(pools.begin(), pools.end()), host_pool_id_(host_pool_id),
      host_backing_(host_backing) {
    std::size_t maximum = 0;
    for (std::size_t i = 0; i < pools_.size(); ++i) {
        const auto& pool = pools_[i];
        if (pool.pool_id != i || !pool.pages || !pool.tables)
            throw std::invalid_argument("context physical pools are inconsistent");
        maximum = std::max(maximum, std::size_t(pool.pages->capacity_pages()));
    }
    if (host_backing.bytes && (host_backing.device != Device::cpu() || !host_backing.host_pinned))
        throw std::invalid_argument("context Host backing must be pinned CPU storage");
    indices_.resize(maximum);
}

const ContextPoolStorage& ContextStorage::pool(std::uint32_t id) const { return pools_.at(id); }

MemorySpan ContextStorage::host_extent(HostExtent extent) const {
    if (!extent.valid() || extent.pool_id != host_pool_id_ || extent.offset > host_backing_.bytes ||
        extent.bytes > host_backing_.bytes - extent.offset)
        throw std::invalid_argument("context Host extent is outside its backing");
    return {static_cast<std::byte*>(host_backing_.data) + extent.offset, extent.bytes,
            host_backing_.device, host_backing_.host_pinned};
}

std::size_t ContextStorage::mapping_capacity_elements() const noexcept {
    std::size_t result = 0;
    for (const auto& pool : pools_)
        result += std::size_t(pool.tables->logical_page_capacity()) * pool.tables->row_count();
    return result;
}

std::size_t ContextStorage::copy_segment_capacity() const noexcept {
    std::size_t result = 3 * (pools_.size() + 1);
    for (const auto& pool : pools_) result += 3 * pool.tables->row_count();
    return result;
}

std::size_t ContextStorage::mapping_elements(const ContextActions& actions) {
    std::size_t result = 0;
    for (const auto& action : actions.operations) {
        if (const auto* update = std::get_if<MappingUpdate>(&action)) {
            if (update->pages.size() > (SIZE_MAX / sizeof(std::int32_t)) - result)
                throw std::overflow_error("context mapping staging size overflow");
            result += update->pages.size();
        }
    }
    return result;
}

void ContextStorage::validate_slot(PageSlot slot) const {
    if (!slot.valid() || slot.index >= pool(slot.pool_id).pages->capacity_pages() ||
        slot.index > INT32_MAX)
        throw std::invalid_argument("context page slot is outside its pool");
}

std::span<const std::int32_t> ContextStorage::page_indices(std::span<const PageSlot> pages) const {
    if (pages.size() > indices_.size())
        throw std::out_of_range("context page batch exceeds capacity");
    for (std::size_t i = 0; i < pages.size(); ++i) {
        validate_slot(pages[i]);
        if (pages[i].pool_id != pages.front().pool_id)
            throw std::invalid_argument("context page run spans different pools");
        indices_[i] = static_cast<std::int32_t>(pages[i].index);
    }
    return {indices_.data(), pages.size()};
}

runtime::ContextTransferObservation ContextStorage::describe(const ContextAction& action) const {
    using Direction = runtime::ContextTransferDirection;
    runtime::ContextTransferObservation result;
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, StateTransferDescription>) {
                result.direction   = value.direction == StateCopyDirection::DeviceToHost
                                         ? Direction::DeviceToHost
                                         : Direction::HostToDevice;
                result.work        = state_->layout().transfer_work();
                result.image_count = 1;
            } else if constexpr (std::is_same_v<T, StateCopy>) {
                result.direction = Direction::DeviceToDevice;
                result.work =
                    value.bytes ? TransferWork{value.bytes, 1} : state_->layout().transfer_work();
                result.image_count = 1;
            } else if constexpr (std::is_same_v<T, PageCopy>) {
                validate_slot(value.source);
                validate_slot(value.destination);
                if (value.source.pool_id != value.destination.pool_id)
                    throw std::invalid_argument("context page copy spans different pools");
                result.resource = {
                    runtime::KvPoolId{static_cast<std::uint16_t>(value.source.pool_id)}};
                result.direction   = Direction::DeviceToDevice;
                result.work        = value.source.index == value.destination.index
                                         ? TransferWork{}
                                         : pool(value.source.pool_id).pages->device_copy_work(1);
                result.page_groups = 1;
            } else if constexpr (std::is_same_v<T, PageTransfer>) {
                if (value.pages.empty())
                    throw std::invalid_argument("context page transfer is empty");
                const auto id      = value.pages.front().pool_id;
                result.resource    = {runtime::KvPoolId{static_cast<std::uint16_t>(id)}};
                result.direction   = value.direction == StateCopyDirection::DeviceToHost
                                         ? Direction::DeviceToHost
                                         : Direction::HostToDevice;
                result.work        = pool(id).pages->host_transfer_work(page_indices(value.pages));
                result.page_groups = value.pages.size();
            }
        },
        action);
    return result;
}

void ContextStorage::validate(const ContextActions& actions, ConstMemorySpan staging) const {
    const auto bytes = mapping_elements(actions) * sizeof(std::int32_t);
    if (bytes && (!staging.data || staging.bytes < bytes || staging.device != Device::cpu() ||
                  !staging.host_pinned ||
                  reinterpret_cast<std::uintptr_t>(staging.data) % alignof(std::int32_t)))
        throw std::invalid_argument("context mapping staging is not a sufficient pinned range");
    const auto state_slot = [&](std::int32_t slot) { (void)state_->image(slot); };
    for (const auto& action : actions.operations) {
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, StateClear>) {
                    state_slot(value.slot);
                } else if constexpr (std::is_same_v<T, StateCopy>) {
                    state_slot(value.source);
                    state_slot(value.destination);
                    if ((!value.bytes && value.offset) || value.offset > state_->image_bytes() ||
                        value.bytes > state_->image_bytes() - value.offset)
                        throw std::invalid_argument(
                            "context State copy group is outside the image");
                } else if constexpr (std::is_same_v<T, StateTransferDescription>) {
                    state_slot(value.direction == StateCopyDirection::DeviceToHost
                                   ? value.slots.source
                                   : value.slots.destination);
                    if (host_extent(value.host).bytes != state_->image_bytes() ||
                        value.bytes != state_->image_bytes())
                        throw std::invalid_argument("context State transfer has inconsistent size");
                } else if constexpr (std::is_same_v<T, PageTransfer>) {
                    (void)describe(action);
                    const auto& storage = *pool(value.pages.front().pool_id).pages;
                    if (host_extent(value.host).bytes !=
                        value.pages.size() * storage.host_layout().page_stride)
                        throw std::invalid_argument("context page transfer has inconsistent size");
                } else if constexpr (std::is_same_v<T, PageCopy>) {
                    (void)describe(action);
                } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                    const auto& tables = *pool(value.row.pool_id).tables;
                    if (!value.row.valid() ||
                        value.row.index >= std::uint32_t(tables.row_count()) ||
                        value.logical_begin > tables.logical_page_capacity() ||
                        value.pages.size() > tables.logical_page_capacity() - value.logical_begin)
                        throw std::invalid_argument("context mapping exceeds its table row");
                    for (auto page : value.pages) {
                        validate_slot(page);
                        if (page.pool_id != value.row.pool_id)
                            throw std::invalid_argument("context mapping names a foreign pool");
                    }
                }
            },
            action);
    }
}

std::size_t ContextStorage::execute(const ContextAction& action, MemorySpan staging,
                                    CudaExecutionView execution) const {
    std::size_t consumed = 0;
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, StateClear>) {
                state_->clear(value.slot, execution);
            } else if constexpr (std::is_same_v<T, StateCopy>) {
                if (value.bytes)
                    state_->copy_group({value.offset, value.bytes}, value.source, value.destination,
                                       execution);
                else
                    state_->copy(value.source, value.destination, execution);
            } else if constexpr (std::is_same_v<T, StateTransferDescription>) {
                if (value.direction == StateCopyDirection::DeviceToHost)
                    state_->copy_to_host(value.slots.source, host_extent(value.host), execution);
                else
                    state_->copy_from_host(host_extent(value.host), value.slots.destination,
                                           execution);
            } else if constexpr (std::is_same_v<T, PageCopy>) {
                (void)pool(value.source.pool_id)
                    .pages->copy_page(value.source.index, value.destination.index, execution);
            } else if constexpr (std::is_same_v<T, PageTransfer>) {
                const auto& storage = *pool(value.pages.front().pool_id).pages;
                const auto ids      = page_indices(value.pages);
                if (value.direction == StateCopyDirection::DeviceToHost)
                    (void)storage.copy_to_host(ids, host_extent(value.host), storage.host_layout(),
                                               execution);
                else
                    (void)storage.copy_from_host(host_extent(value.host), storage.host_layout(),
                                                 ids, execution);
            } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                const auto ids = page_indices(value.pages);
                consumed       = ids.size_bytes();
                pool(value.row.pool_id)
                    .tables->publish(value.row.index, value.logical_begin, ids, staging, execution);
            }
        },
        action);
    return consumed;
}

void ContextStorage::execute(const ContextActions& actions, MemorySpan staging,
                             CudaExecutionView execution) const {
    validate(actions, staging);
    for (const auto& action : actions.operations) {
        const auto consumed = execute(action, staging, execution);
        if (consumed) {
            staging.data = static_cast<std::byte*>(staging.data) + consumed;
            staging.bytes -= consumed;
        }
    }
}

} // namespace ninfer::models::context
