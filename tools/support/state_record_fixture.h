#pragma once

#include "core/buffer.h"
#include "core/device.h"
#include "core/state_image.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::tool {

// Keeps the oracle's dense record order while placing complete images at nonuniform offsets.
class StateRecordFixture {
public:
    StateRecordFixture(std::span<const Tensor> dense_fields, int record_axis)
        : dense_(dense_fields.begin(), dense_fields.end()), axis_(record_axis),
          layout_(plan(dense_fields, record_axis)),
          offsets_(make_offsets(layout_.image_bytes, dense_.front().ne[axis_])),
          backing_(dense_.front().device, offsets_.back() + layout_.image_bytes + 256),
          images_(backing_.span(), layout_) {
        backing_.fill(0xa5);
        upload();
    }

    [[nodiscard]] std::int64_t byte_offset(std::int32_t record) const {
        if (record < 0 || static_cast<std::size_t>(record) >= offsets_.size())
            throw std::out_of_range("fixture state record");
        return offsets_[record];
    }

    [[nodiscard]] std::vector<std::int64_t>
    record_offsets(std::span<const std::int32_t> records) const {
        std::vector<std::int64_t> result;
        result.reserve(records.size());
        for (const auto record : records)
            result.push_back(record < 0 ? kInvalidRecordOffset : byte_offset(record));
        return result;
    }

    [[nodiscard]] StateFieldView field(std::uint32_t index, const Tensor& offsets = {}) const {
        if (index >= dense_.size()) throw std::out_of_range("fixture state field");
        return images_.field(StateField{2 * index + 1}, offsets);
    }

    [[nodiscard]] Tensor field_tensor(std::uint32_t index, std::int32_t record) const {
        return field(index).tensor(byte_offset(record));
    }

    void upload() const { copy(true); }

    void copy_back() const { copy(false); }

    [[nodiscard]] bool guards_intact() const {
        const auto intact = [&](std::size_t begin, std::size_t end) {
            std::vector<std::byte> bytes(end - begin);
            backing_.copy_to_host(bytes.data(), bytes.size(), begin);
            return std::all_of(bytes.begin(), bytes.end(),
                               [](auto byte) { return byte == std::byte{0xa5}; });
        };
        std::size_t previous = 0;
        for (const auto offset : offsets_) {
            for (std::size_t index = 0; index < dense_.size(); ++index) {
                const auto& region = layout_.fields[2 * index + 1].tensor.region;
                const auto begin   = static_cast<std::size_t>(offset) + region.offset;
                if (!intact(previous, begin)) return false;
                previous = begin + region.bytes;
            }
        }
        return intact(previous, backing_.size());
    }

private:
    static StateImageLayout plan(std::span<const Tensor> fields, int axis) {
        if (fields.empty() || axis < 1 || axis > 3 || fields.front().ne[axis] <= 0)
            throw std::invalid_argument("fixture state geometry");
        StateImageBuilder builder;
        (void)builder.add(DType::I32, {64}, 1, "fixture guard");
        for (const auto& tensor : fields) {
            if (!tensor.slice(axis, 0, 1).is_contiguous() ||
                tensor.ne[axis] != fields.front().ne[axis])
                throw std::invalid_argument("fixture state field shape");
            if (axis == 1)
                (void)builder.add(tensor.dtype, {tensor.ne[0]}, 1, "fixture field");
            else if (axis == 2)
                (void)builder.add(tensor.dtype, {tensor.ne[0], tensor.ne[1]}, 1, "fixture field");
            else
                (void)builder.add(tensor.dtype, {tensor.ne[0], tensor.ne[1], tensor.ne[2]}, 1,
                                  "fixture field");
            (void)builder.add(DType::I32, {64}, 1, "fixture guard");
        }
        return builder.finish();
    }

    static std::vector<std::int64_t> make_offsets(std::size_t image_bytes, int count) {
        std::vector<std::int64_t> result(count);
        std::size_t next = 256;
        for (int record = 0; record < count; ++record) {
            result[record] = static_cast<std::int64_t>(next);
            next += image_bytes + static_cast<std::size_t>((record % 3) + 1) * 256;
        }
        return result;
    }

    void copy(bool upload) const {
        for (std::size_t index = 0; index < dense_.size(); ++index) {
            const auto& dense = dense_[index];
            for (int record = 0; record < dense.ne[axis_]; ++record) {
                const auto selected = field_tensor(index, record);
                auto* value = static_cast<std::byte*>(dense.data) + record * dense.nb[axis_];
                CUDA_CHECK(cudaMemcpy(upload ? selected.data : value,
                                      upload ? value : selected.data, selected.bytes(),
                                      cudaMemcpyDefault));
            }
        }
    }

    std::vector<Tensor> dense_;
    int axis_;
    StateImageLayout layout_;
    std::vector<std::int64_t> offsets_;
    DeviceBuffer backing_;
    StateImageStorage images_;
};

} // namespace ninfer::tool
