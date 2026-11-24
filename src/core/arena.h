#pragma once

#include "core/memory.h"
#include "core/tensor.h"

#include <cstddef>
#include <initializer_list>

namespace ninfer {

// Non-owning bump allocator. The backing owner and asynchronous consumers must outlive
// every allocation. Scope rewinds only the cursor; it does not wait for outstanding work.
class MemoryArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class MemoryArena;
        explicit Scope(MemoryArena& arena) noexcept;
        MemoryArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit MemoryArena(MemorySpan storage);
    MemoryArena(const MemoryArena&)            = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&& other) noexcept;
    MemoryArena& operator=(MemoryArena&& other) noexcept;

    [[nodiscard]] MemorySpan alloc_bytes(std::size_t bytes, std::size_t alignment = 256);
    [[nodiscard]] Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                               std::size_t alignment = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    [[nodiscard]] MemorySpan storage() const noexcept { return storage_; }

    [[nodiscard]] Device device() const noexcept { return storage_.device; }

    [[nodiscard]] std::size_t used() const noexcept { return offset_; }

    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.bytes; }

    [[nodiscard]] std::size_t peak_used() const noexcept { return peak_; }

    void reset_peak() noexcept;

private:
    MemorySpan storage_;
    std::size_t offset_ = 0;
    std::size_t peak_   = 0;
};

} // namespace ninfer
