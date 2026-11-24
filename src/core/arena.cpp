#include "core/arena.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace ninfer {

MemoryArena::MemoryArena(MemorySpan storage) : storage_(storage) { validate_memory(storage); }

MemoryArena::MemoryArena(MemoryArena&& other) noexcept
    : storage_(std::exchange(other.storage_, {})), offset_(std::exchange(other.offset_, 0)),
      peak_(std::exchange(other.peak_, 0)) {}

MemoryArena& MemoryArena::operator=(MemoryArena&& other) noexcept {
    if (this != &other) {
        storage_ = std::exchange(other.storage_, {});
        offset_  = std::exchange(other.offset_, 0);
        peak_    = std::exchange(other.peak_, 0);
    }
    return *this;
}

MemorySpan MemoryArena::alloc_bytes(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }
    const auto base  = reinterpret_cast<std::uintptr_t>(storage_.data);
    const auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (offset_ > limit - base || alignment - 1 > limit - base - offset_) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    const auto start = ((base + offset_ + alignment - 1) & ~(alignment - 1)) - base;
    if (start > storage_.bytes || bytes > storage_.bytes - start) { throw std::bad_alloc(); }
    const auto result = storage_.subspan(start, bytes);
    offset_           = start + bytes;
    peak_             = std::max(peak_, offset_);
    return result;
}

Tensor MemoryArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t alignment) {
    return Tensor(alloc_bytes(Tensor::metadata(dtype, shape).bytes(), alignment), dtype, shape);
}

MemoryArena::Scope::Scope(MemoryArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.offset_) {}

MemoryArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->offset_) { arena_->offset_ = saved_offset_; }
}

MemoryArena::Scope::Scope(Scope&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)), saved_offset_(other.saved_offset_) {}

MemoryArena::Scope MemoryArena::scope() noexcept { return Scope(*this); }

void MemoryArena::reset() noexcept { offset_ = 0; }

void MemoryArena::reset_peak() noexcept { peak_ = offset_; }

} // namespace ninfer
