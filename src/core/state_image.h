#pragma once

#include "core/device.h"
#include "core/record_view.h"
#include "core/transfer_work.h"

#include <cstdint>
#include <vector>

namespace ninfer {

struct StateField {
    std::uint32_t index = 0;
};

struct StateImageLayout {
    std::size_t image_bytes = 0;
    std::size_t alignment   = 256;
    std::vector<StateFieldLayout> fields;

    [[nodiscard]] TransferWork transfer_work() const noexcept;
};

// Declares one image. Device and Host records use the same internal field offsets.
class StateImageBuilder {
public:
    [[nodiscard]] StateField add(DType dtype, std::initializer_list<std::int32_t> shape,
                                 std::uint32_t layers, std::string_view label);
    [[nodiscard]] LayoutRegion group(StateField first, StateField last) const;
    [[nodiscard]] StateImageLayout finish() const;

private:
    LayoutBuilder image_;
    std::vector<StateFieldLayout> fields_;
};

// Borrows an arena. The owner supplies record offsets and governs their asynchronous lifetime.
class StateImageStorage {
public:
    StateImageStorage(MemorySpan backing, const StateImageLayout& layout);

    [[nodiscard]] const StateImageLayout& layout() const noexcept { return layout_; }

    [[nodiscard]] MemorySpan backing() const noexcept { return storage_; }

    [[nodiscard]] std::size_t image_bytes() const noexcept { return layout_.image_bytes; }

    [[nodiscard]] StateFieldView field(StateField field, const Tensor& byte_offsets) const;
    [[nodiscard]] StateFieldView field(StateField field, const Tensor& byte_offsets,
                                       std::uint32_t layer) const;
    [[nodiscard]] Tensor tensor(StateField field, std::int64_t byte_offset,
                                std::uint32_t layer = 0) const;
    [[nodiscard]] std::int64_t layer_stride(StateField field) const;
    [[nodiscard]] MemorySpan image(std::int64_t byte_offset) const;

    void clear(std::int64_t byte_offset, CudaExecutionView execution) const;
    void copy(std::int64_t source, std::int64_t destination, CudaExecutionView execution) const;
    void copy_group(LayoutRegion group, std::int64_t source, std::int64_t destination,
                    CudaExecutionView execution) const;
    void copy_to_host(std::int64_t source, MemorySpan destination,
                      CudaExecutionView execution) const;
    void copy_from_host(ConstMemorySpan source, std::int64_t destination,
                        CudaExecutionView execution) const;

private:
    StateImageLayout layout_;
    MemorySpan storage_;
};

} // namespace ninfer
