#include "core/layout.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::size_t checked_add(std::size_t a, std::size_t b, std::string_view label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string(label) + " size overflow");
    }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, std::string_view label) {
    if (!is_power_of_two(alignment)) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    return checked_add(value, mask, label) & ~mask;
}

} // namespace

ConstMemorySpan LayoutRegion::bind(ConstMemorySpan backing) const {
    validate_memory(backing);
    if (backing.data == nullptr) { throw std::invalid_argument("layout backing is null"); }
    if (!is_power_of_two(alignment)) {
        throw std::logic_error("layout region has an invalid alignment");
    }
    if (offset > backing.bytes || bytes > backing.bytes - offset) {
        throw std::out_of_range("layout region exceeds its backing allocation");
    }
    const auto part = backing.subspan(offset, bytes);
    if (reinterpret_cast<std::uintptr_t>(part.data) % alignment != 0) {
        throw std::invalid_argument("layout backing does not satisfy region alignment");
    }
    return part;
}

MemorySpan LayoutRegion::bind(MemorySpan backing) const {
    const auto part = bind(static_cast<ConstMemorySpan>(backing));
    return {const_cast<void*>(part.data), part.bytes, part.device, part.host_pinned};
}

Tensor TensorRegion::bind(MemorySpan backing) const {
    const auto expected = Tensor::metadata(dtype, {shape[0], shape[1], shape[2], shape[3]});
    if (expected.bytes() != region.bytes) {
        throw std::logic_error("tensor layout has an inconsistent byte size");
    }
    return Tensor(region.bind(backing), dtype, {shape[0], shape[1], shape[2], shape[3]});
}

LayoutBuilder::Scope::Scope(LayoutBuilder& builder) noexcept
    : builder_(&builder), saved_cursor_(builder.cursor_) {}

LayoutBuilder::Scope::~Scope() noexcept {
    if (builder_ != nullptr) { builder_->cursor_ = saved_cursor_; }
}

LayoutBuilder::Scope::Scope(Scope&& other) noexcept
    : builder_(other.builder_), saved_cursor_(other.saved_cursor_) {
    other.builder_ = nullptr;
}

LayoutRegion LayoutBuilder::add(std::size_t bytes, std::size_t alignment, std::string_view label) {
    if (bytes == 0) { throw std::invalid_argument(std::string(label) + " must not be empty"); }
    const std::size_t offset = align_up(cursor_, alignment, label);
    cursor_                  = checked_add(offset, bytes, label);
    if (cursor_ > peak_) { peak_ = cursor_; }
    return LayoutRegion{offset, bytes, alignment};
}

TensorRegion LayoutBuilder::add_tensor(DType dtype, std::initializer_list<std::int32_t> shape,
                                       std::size_t alignment, std::string_view label) {
    const auto tensor = Tensor::metadata(dtype, shape);
    TensorRegion out;
    out.region = add(tensor.bytes(), alignment, label);
    out.dtype  = dtype;
    std::copy(shape.begin(), shape.end(), out.shape.begin());
    return out;
}

LayoutBuilder::Scope LayoutBuilder::scope() noexcept { return Scope(*this); }

std::size_t LayoutBuilder::finish(std::size_t alignment, std::string_view label) const {
    return align_up(peak_, alignment, label);
}

WorkspaceLayoutBuilder::Scope::Scope(WorkspaceLayoutBuilder& builder) noexcept
    : builder_(&builder), saved_cursor_(builder.cursor_) {}

WorkspaceLayoutBuilder::Scope::~Scope() noexcept {
    if (builder_ != nullptr) { builder_->cursor_ = saved_cursor_; }
}

WorkspaceLayoutBuilder::Scope::Scope(Scope&& other) noexcept
    : builder_(other.builder_), saved_cursor_(other.saved_cursor_) {
    other.builder_ = nullptr;
}

Tensor WorkspaceLayoutBuilder::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                                     std::size_t alignment) {
    const auto tensor = Tensor::metadata(dtype, shape);
    const auto offset = align_up(cursor_, alignment, "workspace layout");
    cursor_           = checked_add(offset, tensor.bytes(), "workspace layout");
    if (cursor_ > peak_) { peak_ = cursor_; }
    return tensor;
}

MemorySpan WorkspaceLayoutBuilder::alloc_bytes(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) { return {}; }
    const auto offset = align_up(cursor_, alignment, "workspace layout");
    cursor_           = checked_add(offset, bytes, "workspace layout");
    if (cursor_ > peak_) { peak_ = cursor_; }
    return MemorySpan{nullptr, bytes, Device{}};
}

WorkspaceLayoutBuilder::Scope WorkspaceLayoutBuilder::scope() noexcept { return Scope(*this); }

std::size_t WorkspaceLayoutBuilder::peak_bytes(std::size_t alignment) const {
    return align_up(peak_, alignment, "workspace layout");
}

} // namespace ninfer
