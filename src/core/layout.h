#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <array>
#include <string_view>

namespace ninfer {

struct LayoutRegion {
    std::size_t offset    = 0;
    std::size_t bytes     = 0;
    std::size_t alignment = 1;

    [[nodiscard]] DeviceSpan bind(DeviceSpan backing) const;
};

struct TensorRegion {
    LayoutRegion region;
    DType dtype = DType::BF16;
    std::array<std::int32_t, 4> shape{1, 1, 1, 1};

    [[nodiscard]] Tensor bind(DeviceSpan backing) const;
};

// Checked offset builder for project-owned backing allocations. Model-specific code owns the
// region list; this mechanism only guarantees that the total and every later binding share the
// exact offsets produced here.
class LayoutBuilder {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class LayoutBuilder;
        explicit Scope(LayoutBuilder& builder) noexcept;
        LayoutBuilder* builder_   = nullptr;
        std::size_t saved_cursor_ = 0;
    };

    [[nodiscard]] LayoutRegion add(std::size_t bytes, std::size_t alignment,
                                   std::string_view label);
    [[nodiscard]] TensorRegion add_tensor(DType dtype,
                                          std::initializer_list<std::int32_t> shape,
                                          std::size_t alignment, std::string_view label);
    [[nodiscard]] Scope scope() noexcept;
    [[nodiscard]] std::size_t finish(std::size_t alignment = 1,
                                     std::string_view label = "layout") const;

private:
    std::size_t cursor_ = 0;
    std::size_t peak_   = 0;
};

// Dry-run counterpart of WorkspaceArena. Target allocation helpers can run against this builder
// and the real arena, including identical nested scope lifetimes, without maintaining byte formulas.
class WorkspaceLayoutBuilder {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class WorkspaceLayoutBuilder;
        explicit Scope(WorkspaceLayoutBuilder& builder) noexcept;
        WorkspaceLayoutBuilder* builder_ = nullptr;
        std::size_t saved_cursor_        = 0;
    };

    [[nodiscard]] Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                               std::size_t alignment = 256);
    void alloc_bytes(std::size_t bytes, std::size_t alignment = 256);
    [[nodiscard]] Scope scope() noexcept;
    [[nodiscard]] std::size_t peak_bytes(std::size_t alignment = 256) const;

private:
    std::size_t cursor_ = 0;
    std::size_t peak_   = 0;
};

} // namespace ninfer
