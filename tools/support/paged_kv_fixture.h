#pragma once

#include "core/buffer.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::tool {

// Packed records with guard gaps. Dense plane inputs remain independent oracle/initialization data.
class PackedKVFixture {
public:
    PackedKVFixture(Device device, const KVPageGeometry& geometry, std::int32_t physical_pages)
        : layout_(plan_kv_page_layout(geometry)), pages_(physical_pages),
          stride_(layout_.page_bytes + layout_.alignment),
          storage_(device, capacity(layout_, physical_pages)) {
        storage_.fill(0xa5);
    }

    [[nodiscard]] const KVPageLayout& layout() const noexcept { return layout_; }

    [[nodiscard]] MemorySpan storage() noexcept { return storage_.span(); }

    [[nodiscard]] ConstMemorySpan storage() const noexcept { return storage_.span(); }

    [[nodiscard]] std::size_t record_stride() const noexcept { return stride_; }

    [[nodiscard]] std::int64_t physical_offset(std::int32_t page) const {
        if (page < 0 || page >= pages_) throw std::out_of_range("fixture physical page");
        return static_cast<std::int64_t>(layout_.alignment + stride_ * page);
    }

    [[nodiscard]] std::vector<std::int64_t>
    table_offsets(std::span<const std::int32_t> pages) const {
        std::vector<std::int64_t> result;
        result.reserve(pages.size());
        for (const auto page : pages)
            result.push_back(page < 0 ? kInvalidRecordOffset : physical_offset(page));
        return result;
    }

    [[nodiscard]] PagedPlaneView plane(std::size_t index, const Tensor& table) {
        return {{storage_.span(), table, layout_.geometry.page_tokens}, layout_.planes.at(index)};
    }

    void upload_plane(std::size_t index, const Tensor& dense, bool head_major = false) {
        copy_plane(index, dense, head_major, true);
    }

    void download_plane(std::size_t index, const Tensor& dense, bool head_major = false) const {
        copy_plane(index, dense, head_major, false);
    }

    [[nodiscard]] std::vector<std::byte> download_plane(std::size_t index) const {
        const auto& plane = layout_.planes.at(index);
        std::vector<std::byte> bytes(plane.tensor.region.bytes * pages_);
        Tensor dense({bytes.data(), bytes.size(), Device::cpu()}, plane.tensor.dtype,
                     {plane.tensor.shape[0], plane.tensor.shape[1], plane.tensor.shape[2], pages_});
        download_plane(index, dense);
        return bytes;
    }

    [[nodiscard]] bool guards_intact() const {
        const auto intact = [](const auto& bytes) {
            return std::all_of(bytes.begin(), bytes.end(),
                               [](auto byte) { return byte == std::byte{0xa5}; });
        };
        std::vector<std::byte> prefix(layout_.alignment);
        storage_.copy_to_host(prefix.data(), prefix.size());
        if (!intact(prefix)) return false;
        std::size_t previous_end = 0;
        const auto check_gap     = [&](std::size_t begin, std::size_t end) {
            if (begin == end) return true;
            const auto width = end - begin;
            std::vector<std::byte> bytes(width * pages_);
            CUDA_CHECK(cudaMemcpy2D(bytes.data(), width,
                                        static_cast<const std::byte*>(storage_.data()) +
                                            physical_offset(0) + begin,
                                        stride_, width, pages_, cudaMemcpyDeviceToHost));
            return intact(bytes);
        };
        for (const auto& plane : layout_.planes) {
            const auto& region = plane.tensor.region;
            if (!check_gap(previous_end, region.offset)) return false;
            previous_end = region.offset + region.bytes;
        }
        return check_gap(previous_end, stride_);
    }

private:
    static std::size_t capacity(const KVPageLayout& layout, std::int32_t pages) {
        if (pages <= 0) throw std::invalid_argument("fixture requires physical pages");
        return layout.alignment + (layout.page_bytes + layout.alignment) * pages;
    }

    void copy_plane(std::size_t index, const Tensor& dense, bool head_major, bool upload) const {
        const auto& plane = layout_.planes.at(index);
        const auto& shape = plane.tensor.shape;
        const auto expected =
            head_major ? std::array<std::int32_t, 4>{shape[0], shape[1], pages_, shape[2]}
                       : std::array<std::int32_t, 4>{shape[0], shape[1], shape[2], pages_};
        if (dense.dtype != plane.tensor.dtype ||
            !std::equal(std::begin(dense.ne), std::end(dense.ne), expected.begin()) ||
            !dense.is_contiguous())
            throw std::invalid_argument("fixture plane shape or dtype mismatch");
        const auto width = head_major ? plane.head_stride_bytes : plane.tensor.region.bytes;
        const auto count = head_major ? shape[2] : 1;
        auto* packed     = const_cast<std::byte*>(static_cast<const std::byte*>(storage_.data())) +
                       physical_offset(0) + plane.tensor.region.offset;
        auto* contiguous = static_cast<std::byte*>(dense.data);
        for (int head = 0; head < count; ++head) {
            auto* record_plane = packed + head * plane.head_stride_bytes;
            auto* dense_plane  = contiguous + head * plane.head_stride_bytes * pages_;
            if (upload)
                CUDA_CHECK(cudaMemcpy2D(record_plane, stride_, dense_plane, width, width, pages_,
                                        cudaMemcpyDefault));
            else
                CUDA_CHECK(cudaMemcpy2D(dense_plane, width, record_plane, stride_, width, pages_,
                                        cudaMemcpyDefault));
        }
        // Device-to-device cudaMemcpy2D may return before completion. Callers launch
        // on independent nonblocking streams, so complete this fixture-owned copy.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    KVPageLayout layout_;
    std::int32_t pages_;
    std::size_t stride_;
    DeviceBuffer storage_;
};

} // namespace ninfer::tool
