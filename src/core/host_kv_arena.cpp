#include "core/host_kv_arena.h"

#include "core/dtype.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

constexpr std::size_t kHostKVAlignment = 256;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(std::string(label) + " alignment overflow");
    }
    return (value + mask) & ~mask;
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

} // namespace

HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry) {
    if (geometry.page_tokens == 0 || geometry.planes.empty()) {
        throw std::invalid_argument("Host KV page geometry is empty");
    }

    HostKVPageLayout out;
    out.geometry = geometry;
    out.planes.reserve(geometry.planes.size());
    std::size_t cursor = 0;
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Host KV plane geometry must be positive");
        }
        // Device slab alignment is not part of the canonical packed Host representation.
        cursor = align_up(cursor, kHostKVAlignment, "Host KV plane");
        const std::size_t head_bytes =
            checked_mul(checked_mul(static_cast<std::size_t>(plane.leading_extent),
                                    geometry.page_tokens, "Host KV head payload overflow"),
                        dtype_size(plane.dtype), "Host KV head payload overflow");
        const std::size_t page_bytes =
            checked_mul(head_bytes, static_cast<std::size_t>(plane.head_extent),
                        "Host KV plane payload overflow");
        out.planes.push_back(HostKVPlaneLayout{
            .offset             = cursor,
            .page_payload_bytes = page_bytes,
            .head_payload_bytes = head_bytes,
        });
        cursor = checked_add(cursor, page_bytes, "Host KV page payload overflow");
    }
    out.page_stride = align_up(cursor, kHostKVAlignment, "Host KV page record");
    return out;
}

TransferWork plan_host_kv_transfer_work(const HostKVPageLayout& layout, std::uint32_t pages,
                                        std::uint32_t contiguous_runs) {
    if (pages == 0) { return {}; }
    if (contiguous_runs == 0 || contiguous_runs > pages || layout.planes.empty() ||
        layout.planes.size() != layout.geometry.planes.size()) {
        throw std::invalid_argument("Host KV transfer geometry is invalid");
    }

    std::size_t bytes_per_page     = 0;
    std::size_t operations_per_run = 0;
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        bytes_per_page = checked_add(bytes_per_page, layout.planes[index].page_payload_bytes,
                                     "Host KV transfer payload overflow");
        const KVPlaneGeometry& plane = layout.geometry.planes[index];
        const std::size_t operations =
            layout.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor
                ? 1U
                : static_cast<std::size_t>(plane.head_extent);
        operations_per_run = checked_add(operations_per_run, operations,
                                         "Host KV transfer operation count overflow");
    }

    const std::size_t payload =
        checked_mul(bytes_per_page, pages, "Host KV transfer payload overflow");
    const std::size_t operations = checked_mul(operations_per_run, contiguous_runs,
                                               "Host KV transfer operation count overflow");
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork plan_device_kv_copy_work(const HostKVPageLayout& layout, std::uint32_t pages) {
    if (pages == 0) { return {}; }
    if (layout.planes.empty() || layout.planes.size() != layout.geometry.planes.size()) {
        throw std::invalid_argument("Device KV copy geometry is invalid");
    }

    std::size_t bytes_per_page = 0;
    for (const HostKVPlaneLayout& plane : layout.planes) {
        bytes_per_page = checked_add(bytes_per_page, plane.page_payload_bytes,
                                     "Device KV copy payload overflow");
    }
    const std::size_t payload =
        checked_mul(bytes_per_page, pages, "Device KV copy payload overflow");
    const std::size_t operations =
        checked_mul(layout.planes.size(), pages, "Device KV copy operation count overflow");
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Device KV copy operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

bool HostKVAllocationView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationView HostKVAllocationView::subview(std::uint32_t begin, std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationView(
        handle_, data_ + static_cast<std::size_t>(begin) * layout_->page_stride, layout_, count);
}

bool HostKVAllocationConstView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationConstView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationConstView HostKVAllocationConstView::subview(std::uint32_t begin,
                                                             std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationConstView(
        handle_, data_ + static_cast<std::size_t>(begin) * layout_->page_stride, layout_, count);
}

HostKVAllocation::~HostKVAllocation() { (void)release(); }

HostKVAllocation::HostKVAllocation(HostKVAllocation&& other) noexcept
    : owner_(other.owner_), descriptor_(other.descriptor_), generation_(other.generation_) {
    other.disarm();
}

HostKVAllocation& HostKVAllocation::operator=(HostKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_      = other.owner_;
    descriptor_ = other.descriptor_;
    generation_ = other.generation_;
    other.disarm();
    return *this;
}

HostKVAllocationHandle HostKVAllocation::handle() const noexcept {
    return valid() ? HostKVAllocationHandle(owner_, descriptor_, generation_)
                   : HostKVAllocationHandle();
}

std::uint32_t HostKVAllocation::page_count() const noexcept {
    if (!valid() || descriptor_ >= owner_->descriptors_.size()) { return 0; }
    const HostKVArena::Descriptor& descriptor = owner_->descriptors_[descriptor_];
    return descriptor.active && descriptor.generation == generation_ ? descriptor.pages : 0;
}

bool HostKVAllocation::release() noexcept {
    if (!valid()) { return false; }
    const bool released = owner_->release_descriptor(descriptor_, generation_);
    disarm();
    return released;
}

void HostKVAllocation::disarm() noexcept {
    owner_      = nullptr;
    descriptor_ = 0;
    generation_ = 0;
}

HostKVArena::HostKVArena(std::size_t capacity_bytes,
                         std::span<const HostKVPageLayout> supported_layouts)
    : capacity_bytes_(capacity_bytes),
      layouts_(supported_layouts.begin(), supported_layouts.end()) {
    for (std::size_t index = 0; index < layouts_.size(); ++index) {
        const HostKVPageLayout planned = plan_host_kv_page_layout(layouts_[index].geometry);
        if (planned != layouts_[index]) {
            throw std::invalid_argument("Host KV arena received an inconsistent page layout");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (layouts_[previous] == layouts_[index]) {
                throw std::invalid_argument("Host KV arena contains a duplicate page layout");
            }
        }
    }
    if (capacity_bytes_ == 0) { return; }
    if (layouts_.empty()) {
        throw std::invalid_argument("Non-empty Host KV arena requires supported page layouts");
    }

    backing_.emplace(capacity_bytes_);
    const auto smallest = std::min_element(
        layouts_.begin(), layouts_.end(), [](const HostKVPageLayout& a, const HostKVPageLayout& b) {
            return a.page_stride < b.page_stride;
        });
    const std::size_t maximum_descriptors = capacity_bytes_ / smallest->page_stride;
    if (maximum_descriptors > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV arena descriptor capacity exceeds uint32");
    }
    descriptors_.resize(maximum_descriptors);
    free_descriptors_.reserve(maximum_descriptors);
    for (std::size_t index = maximum_descriptors; index > 0; --index) {
        free_descriptors_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    free_extents_.reserve(maximum_descriptors + 1);
    free_extents_.push_back({0, capacity_bytes_});
}

std::optional<std::uint32_t>
HostKVArena::find_layout(const HostKVPageLayout& layout) const noexcept {
    const auto it = std::find(layouts_.begin(), layouts_.end(), layout);
    if (it == layouts_.end()) { return std::nullopt; }
    return static_cast<std::uint32_t>(it - layouts_.begin());
}

const HostKVPageLayout* HostKVArena::layout_for(const KVPageGeometry& geometry) const noexcept {
    const auto layout =
        std::find_if(layouts_.begin(), layouts_.end(), [&](const HostKVPageLayout& candidate) {
            return candidate.geometry == geometry;
        });
    return layout == layouts_.end() ? nullptr : &*layout;
}

std::optional<std::size_t> HostKVArena::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        if (free_extents_[index].bytes >= bytes) { return index; }
    }
    return std::nullopt;
}

bool HostKVArena::can_allocate(const HostKVPageLayout& layout, std::uint32_t pages) const noexcept {
    if (pages == 0 || free_descriptors_.empty() || !find_layout(layout) ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return false;
    }
    return find_free_extent(layout.page_stride * static_cast<std::size_t>(pages)).has_value();
}

std::optional<HostKVAllocation> HostKVArena::allocate(const HostKVPageLayout& layout,
                                                      std::uint32_t pages) noexcept {
    const std::optional<std::uint32_t> layout_index = find_layout(layout);
    if (!layout_index || pages == 0 || free_descriptors_.empty() ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return std::nullopt;
    }
    const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(pages);
    const std::optional<std::size_t> free_index = find_free_extent(bytes);
    if (!free_index) { return std::nullopt; }

    const std::uint32_t descriptor_index = take_descriptor();
    if (descriptor_index == std::numeric_limits<std::uint32_t>::max()) { return std::nullopt; }
    FreeExtent& free         = free_extents_[*free_index];
    const std::size_t offset = free.offset;
    free.offset += bytes;
    free.bytes -= bytes;
    if (free.bytes == 0) {
        free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(*free_index));
    }

    Descriptor& descriptor = descriptors_[descriptor_index];
    descriptor.offset      = offset;
    descriptor.bytes       = bytes;
    descriptor.layout      = *layout_index;
    descriptor.pages       = pages;
    descriptor.active      = true;
    occupied_bytes_ += bytes;
    bump_revision();
    return HostKVAllocation(*this, descriptor_index, descriptor.generation);
}

std::optional<HostKVAllocationRecipe> HostKVArena::plan_after_releases(
    std::span<const HostKVAllocationHandle> proposed_releases,
    std::span<const HostKVAllocationRequest> target_allocations) const {
    if (proposed_releases.empty() && target_allocations.empty()) { return std::nullopt; }
    if (target_allocations.size() > free_descriptors_.size() + proposed_releases.size()) {
        return std::nullopt;
    }

    HostKVAllocationRecipe recipe;
    recipe.owner_          = this;
    recipe.arena_revision_ = revision_;
    recipe.releases_.reserve(proposed_releases.size());
    recipe.targets_.reserve(target_allocations.size());

    std::vector<FreeExtent> simulated = free_extents_;
    const auto insert_extent          = [&](FreeExtent extent) {
        const auto position = std::lower_bound(simulated.begin(), simulated.end(), extent.offset,
                                                        [](const FreeExtent& candidate, std::size_t offset) {
                                                   return candidate.offset < offset;
                                               });
        auto inserted       = simulated.insert(position, extent);
        if (inserted != simulated.begin()) {
            auto previous = inserted - 1;
            if (previous->offset + previous->bytes == inserted->offset) {
                previous->bytes += inserted->bytes;
                inserted = simulated.erase(inserted);
                inserted = previous;
            }
        }
        const auto next = inserted + 1;
        if (next != simulated.end() && inserted->offset + inserted->bytes == next->offset) {
            inserted->bytes += next->bytes;
            simulated.erase(next);
        }
    };

    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocationHandle handle = proposed_releases[index];
        if (!valid_handle(handle) ||
            std::find(proposed_releases.begin(),
                      proposed_releases.begin() + static_cast<std::ptrdiff_t>(index),
                      handle) != proposed_releases.begin() + static_cast<std::ptrdiff_t>(index)) {
            return std::nullopt;
        }
        const Descriptor& descriptor = descriptors_[handle.descriptor_];
        insert_extent({descriptor.offset, descriptor.bytes});
        recipe.releases_.push_back(handle);
    }

    for (const HostKVAllocationRequest& request : target_allocations) {
        if (request.layout == nullptr || request.pages == 0) { return std::nullopt; }
        const std::optional<std::uint32_t> layout_index = find_layout(*request.layout);
        if (!layout_index ||
            request.layout->page_stride > std::numeric_limits<std::size_t>::max() / request.pages) {
            return std::nullopt;
        }
        const std::size_t bytes =
            request.layout->page_stride * static_cast<std::size_t>(request.pages);
        const auto extent =
            std::find_if(simulated.begin(), simulated.end(),
                         [&](const FreeExtent& free) { return free.bytes >= bytes; });
        if (extent == simulated.end()) { return std::nullopt; }
        const std::size_t offset = extent->offset;
        extent->offset += bytes;
        extent->bytes -= bytes;
        if (extent->bytes == 0) { simulated.erase(extent); }
        recipe.targets_.push_back(HostKVAllocationRecipe::Target{
            .layout = *layout_index,
            .pages  = request.pages,
            .offset = offset,
            .bytes  = bytes,
        });
    }
    return recipe;
}

bool HostKVArena::can_allocate_after_suballocation_releases(
    std::span<const HostKVSuballocationRelease> proposed_releases,
    std::span<const HostKVAllocationRequest> target_allocations) const {
    if (proposed_releases.empty() && target_allocations.empty()) { return true; }

    std::vector<FreeExtent> simulated = free_extents_;
    const auto insert_extent          = [&](FreeExtent extent) {
        const auto position = std::lower_bound(simulated.begin(), simulated.end(), extent.offset,
                                                        [](const FreeExtent& candidate, std::size_t offset) {
                                                   return candidate.offset < offset;
                                               });
        auto inserted       = simulated.insert(position, extent);
        if (inserted != simulated.begin()) {
            auto previous = inserted - 1;
            if (previous->offset + previous->bytes == inserted->offset) {
                previous->bytes += inserted->bytes;
                inserted = simulated.erase(inserted);
                inserted = previous;
            }
        }
        const auto next = inserted + 1;
        if (next != simulated.end() && inserted->offset + inserted->bytes == next->offset) {
            inserted->bytes += next->bytes;
            simulated.erase(next);
        }
    };

    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVSuballocationRelease& release = proposed_releases[index];
        if (!valid_handle(release.allocation) || release.page_count == 0) { return false; }
        const Descriptor& descriptor = descriptors_[release.allocation.descriptor_];
        if (release.begin_page > descriptor.pages ||
            release.page_count > descriptor.pages - release.begin_page) {
            return false;
        }
        const std::uint32_t end = release.begin_page + release.page_count;
        for (std::size_t prior = 0; prior < index; ++prior) {
            const HostKVSuballocationRelease& other = proposed_releases[prior];
            if (other.allocation != release.allocation) { continue; }
            const std::uint32_t other_end = other.begin_page + other.page_count;
            if (release.begin_page < other_end && other.begin_page < end) { return false; }
        }
        const std::size_t stride = layouts_[descriptor.layout].page_stride;
        insert_extent(FreeExtent{
            .offset = checked_add(descriptor.offset,
                                  checked_mul(static_cast<std::size_t>(release.begin_page), stride,
                                              "Host KV suballocation release offset overflow"),
                                  "Host KV suballocation release offset overflow"),
            .bytes  = checked_mul(static_cast<std::size_t>(release.page_count), stride,
                                  "Host KV suballocation release size overflow"),
        });
    }

    std::size_t available_descriptors = free_descriptors_.size();
    std::size_t required_descriptors  = target_allocations.size();
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocationHandle allocation = proposed_releases[index].allocation;
        bool first                              = true;
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (proposed_releases[prior].allocation == allocation) {
                first = false;
                break;
            }
        }
        if (!first) { continue; }

        std::vector<std::pair<std::uint32_t, std::uint32_t>> intervals;
        for (const HostKVSuballocationRelease& release : proposed_releases) {
            if (release.allocation == allocation) {
                intervals.emplace_back(release.begin_page, release.begin_page + release.page_count);
            }
        }
        std::sort(intervals.begin(), intervals.end());
        const Descriptor& descriptor = descriptors_[allocation.descriptor_];
        std::uint32_t cursor         = 0;
        std::size_t retained_runs    = 0;
        for (const auto [begin, end] : intervals) {
            if (begin > cursor) { ++retained_runs; }
            cursor = end;
        }
        if (cursor < descriptor.pages) { ++retained_runs; }
        if (retained_runs == 0) {
            ++available_descriptors;
        } else if (retained_runs > 1) {
            required_descriptors += retained_runs - 1U;
        }
    }
    if (required_descriptors > available_descriptors) { return false; }

    for (const HostKVAllocationRequest& request : target_allocations) {
        if (request.layout == nullptr || request.pages == 0) { return false; }
        const std::optional<std::uint32_t> layout_index = find_layout(*request.layout);
        if (!layout_index ||
            request.layout->page_stride > std::numeric_limits<std::size_t>::max() / request.pages) {
            return false;
        }
        const std::size_t bytes =
            request.layout->page_stride * static_cast<std::size_t>(request.pages);
        const auto extent =
            std::find_if(simulated.begin(), simulated.end(),
                         [&](const FreeExtent& free) { return free.bytes >= bytes; });
        if (extent == simulated.end()) { return false; }
        extent->offset += bytes;
        extent->bytes -= bytes;
        if (extent->bytes == 0) { simulated.erase(extent); }
    }
    return true;
}

bool HostKVArena::apply_recipe(HostKVAllocationRecipe&& recipe,
                               std::span<HostKVAllocation* const> proposed_releases,
                               std::span<HostKVAllocation> target_allocations) noexcept {
    if (recipe.owner_ != this || recipe.arena_revision_ != revision_ ||
        recipe.releases_.size() != proposed_releases.size() ||
        recipe.targets_.size() != target_allocations.size()) {
        return false;
    }
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocation* allocation = proposed_releases[index];
        if (allocation == nullptr || allocation->handle() != recipe.releases_[index] ||
            !valid_handle(recipe.releases_[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < target_allocations.size(); ++index) {
        const HostKVAllocationRecipe::Target& target = recipe.targets_[index];
        if (target_allocations[index].valid() || target.layout >= layouts_.size() ||
            target.pages == 0 ||
            layouts_[target.layout].page_stride >
                std::numeric_limits<std::size_t>::max() / target.pages ||
            layouts_[target.layout].page_stride * static_cast<std::size_t>(target.pages) !=
                target.bytes) {
            return false;
        }
    }

    // All generations and outputs are validated before the first mutation. The recipe was minted
    // from this exact revision, so every operation below is an invariant-preserving adoption.
    for (HostKVAllocation* allocation : proposed_releases) {
        if (!allocation->release()) { std::terminate(); }
    }
    for (std::size_t index = 0; index < recipe.targets_.size(); ++index) {
        const HostKVAllocationRecipe::Target& target = recipe.targets_[index];
        std::optional<HostKVAllocation> allocation =
            allocate(layouts_[target.layout], target.pages);
        if (!allocation) { std::terminate(); }
        const Descriptor& descriptor = descriptors_[allocation->descriptor_];
        if (descriptor.offset != target.offset || descriptor.bytes != target.bytes) {
            std::terminate();
        }
        target_allocations[index] = std::move(*allocation);
    }
    recipe.owner_          = nullptr;
    recipe.arena_revision_ = 0;
    recipe.releases_.clear();
    recipe.targets_.clear();
    return true;
}

std::pair<HostKVAllocation, HostKVAllocation> HostKVArena::split(HostKVAllocation&& allocation,
                                                                 std::uint32_t page_offset) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot split a stale Host KV allocation");
    }
    Descriptor& original = descriptors_[allocation.descriptor_];
    if (page_offset == 0 || page_offset >= original.pages) {
        throw std::out_of_range("Host KV split must leave two non-empty allocations");
    }
    const std::uint32_t right_index = take_descriptor();
    if (right_index == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("Host KV descriptor capacity invariant was violated");
    }

    const std::size_t stride        = layouts_[original.layout].page_stride;
    const std::uint32_t right_pages = original.pages - page_offset;
    Descriptor& right               = descriptors_[right_index];
    right.offset = original.offset + static_cast<std::size_t>(page_offset) * stride;
    right.bytes  = static_cast<std::size_t>(right_pages) * stride;
    right.layout = original.layout;
    right.pages  = right_pages;
    right.active = true;

    increment_generation(original.generation);
    original.pages                       = page_offset;
    original.bytes                       = static_cast<std::size_t>(page_offset) * stride;
    const std::uint32_t left_generation  = original.generation;
    const std::uint32_t right_generation = right.generation;
    const std::uint32_t left_index       = allocation.descriptor_;
    allocation.disarm();
    bump_revision();
    return {HostKVAllocation(*this, left_index, left_generation),
            HostKVAllocation(*this, right_index, right_generation)};
}

HostKVAllocationView HostKVArena::writable_view(HostKVAllocation& allocation) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    return HostKVAllocationView(allocation.handle(), allocation_data(descriptor),
                                &layouts_[descriptor.layout], descriptor.pages);
}

HostKVAllocationConstView HostKVArena::view(const HostKVAllocation& allocation) const {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    return HostKVAllocationConstView(allocation.handle(), allocation_data(descriptor),
                                     &layouts_[descriptor.layout], descriptor.pages);
}

bool HostKVArena::valid_handle(HostKVAllocationHandle handle) const noexcept {
    if (handle.owner_ != this || handle.descriptor_ >= descriptors_.size()) { return false; }
    const Descriptor& descriptor = descriptors_[handle.descriptor_];
    return descriptor.active && descriptor.generation == handle.generation_;
}

std::uint32_t HostKVArena::take_descriptor() noexcept {
    if (free_descriptors_.empty()) { return std::numeric_limits<std::uint32_t>::max(); }
    const std::uint32_t out = free_descriptors_.back();
    free_descriptors_.pop_back();
    return out;
}

bool HostKVArena::release_descriptor(std::uint32_t descriptor_index,
                                     std::uint32_t generation) noexcept {
    if (descriptor_index >= descriptors_.size()) { return false; }
    Descriptor& descriptor = descriptors_[descriptor_index];
    if (!descriptor.active || descriptor.generation != generation) { return false; }

    const FreeExtent released{descriptor.offset, descriptor.bytes};
    occupied_bytes_ -= descriptor.bytes;
    descriptor.active = false;
    descriptor.offset = 0;
    descriptor.bytes  = 0;
    descriptor.pages  = 0;
    increment_generation(descriptor.generation);
    free_descriptors_.push_back(descriptor_index);
    insert_free_extent(released);
    bump_revision();
    return true;
}

void HostKVArena::insert_free_extent(FreeExtent extent) noexcept {
    const auto position = std::lower_bound(
        free_extents_.begin(), free_extents_.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = free_extents_.insert(position, extent);
    if (inserted != free_extents_.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            inserted = free_extents_.erase(inserted);
            inserted = previous;
        }
    }
    auto next = inserted + 1;
    if (next != free_extents_.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        free_extents_.erase(next);
    }
}

std::byte* HostKVArena::allocation_data(const Descriptor& descriptor) const noexcept {
    if (!backing_) { return nullptr; }
    return static_cast<std::byte*>(backing_->data()) + descriptor.offset;
}

void HostKVArena::bump_revision() noexcept {
    ++revision_;
    if (revision_ == 0) { ++revision_; }
}

} // namespace ninfer
