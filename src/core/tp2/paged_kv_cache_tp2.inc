// Translation-unit fragment of src/core/paged_kv_cache.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) KV mirrors: rank 1's page and execution-table pools replay rank 0's physical
// mutations (attach_mirror, the page runs both pools share, the mirror's execution-row lease).

#include "core/device_scope.h"
#include "core/uva_copy.h"

namespace ninfer {

void DeviceKVPagePool::attach_mirror(const DeviceKVPagePool& mirror, DeviceKVMirror where) {
    if (&mirror == this || mirror.capacity_pages() != capacity_pages() ||
        mirror.planes_.size() != planes_.size() ||
        mirror.spec_.geometry.device_plane_order != spec_.geometry.device_plane_order ||
        where.device < 0 || where.stream == nullptr) {
        throw std::invalid_argument("Paged KV mirror pool does not match the origin pool");
    }
    for (std::size_t index = 0; index < planes_.size(); ++index) {
        const Tensor& origin  = planes_[index];
        const Tensor& replica = mirror.planes_[index];
        if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor
                ? origin.nb[3] != replica.nb[3]
                : (origin.nb[2] != replica.nb[2] || origin.nb[3] != replica.nb[3] ||
                   origin.ne[3] != replica.ne[3])) {
            throw std::invalid_argument("Paged KV mirror pool plane geometry differs");
        }
    }
    mirror_       = &mirror;
    mirror_where_ = where;
}

void DeviceKVPagePool::require_no_mirror(const char* what) const {
    if (mirror_ != nullptr) {
        throw std::logic_error(std::string(what) +
                               " is not available while a tensor-parallel mirror is attached");
    }
}

void DeviceKVPagePool::zero_run(std::int32_t first, std::int32_t count, cudaStream_t stream) const {
    notify_write(stream);
    for (std::size_t index = 0; index < planes_.size(); ++index) {
        const Tensor& plane = planes_[index];
        auto* base          = static_cast<unsigned char*>(plane.data);
        if (spec_.geometry.planes[index].host) {
            // Host plane (active KV in RAM, page-major): a stream-ordered kernel fill.
            uva_fill_async(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                           static_cast<std::size_t>(count) * plane.nb[3], stream);
        } else if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                       static_cast<std::size_t>(count) * plane.nb[3], stream));
        } else {
            CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                         plane.nb[3], 0,
                                         static_cast<std::size_t>(count) * plane.nb[2],
                                         static_cast<std::size_t>(plane.ne[3]), stream));
        }
    }
}

void DeviceKVPagePool::copy_run(std::int32_t source_index, std::int32_t destination_index,
                                cudaStream_t stream, std::int32_t count) const {
    notify_write(stream);
    for (std::size_t index = 0; index < planes_.size(); ++index) {
        const Tensor& plane = planes_[index];
        auto* base          = static_cast<unsigned char*>(plane.data);
        if (spec_.geometry.planes[index].host) {
            // Host plane (active KV in RAM): a Host-to-Host cudaMemcpyAsync would run at once on
            // the CPU, unordered with the stream; a kernel copy keeps the stream order.
            uva_copy_async(base + static_cast<std::int64_t>(destination_index) * plane.nb[3],
                           base + static_cast<std::int64_t>(source_index) * plane.nb[3],
                           static_cast<std::size_t>(count) * plane.nb[3], stream);
        } else if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            CUDA_CHECK(cudaMemcpyAsync(
                base + static_cast<std::int64_t>(destination_index) * plane.nb[3],
                base + static_cast<std::int64_t>(source_index) * plane.nb[3],
                static_cast<std::size_t>(count) * plane.nb[3], cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                base + static_cast<std::int64_t>(destination_index) * plane.nb[2], plane.nb[3],
                base + static_cast<std::int64_t>(source_index) * plane.nb[2], plane.nb[3],
                static_cast<std::size_t>(count) * plane.nb[2],
                static_cast<std::size_t>(plane.ne[3]), cudaMemcpyDeviceToDevice, stream));
        }
    }
}

void KVExecutionTablePool::attach_mirror(KVExecutionTablePool& mirror, DeviceKVMirror where) {
    if (&mirror == this || mirror.logical_page_capacity() != logical_page_capacity() ||
        mirror.row_count() != row_count() || where.device < 0 || where.stream == nullptr) {
        throw std::invalid_argument("Paged KV mirror table pool does not match the origin pool");
    }
    const auto has_bound_row = [](const KVExecutionTablePool& pool) {
        return std::find(pool.row_in_use_.begin(), pool.row_in_use_.end(), true) !=
               pool.row_in_use_.end();
    };
    if (has_bound_row(*this) || has_bound_row(mirror)) {
        throw std::logic_error(
            "Paged KV mirror cannot be attached while an execution row is bound");
    }
    mirror_       = &mirror;
    mirror_where_ = where;
}

const KVExecutionRowLease& KVExecutionTablePool::mirror_row(KVExecutionRowHandle handle) const {
    if (mirror_ == nullptr) { throw std::logic_error("Paged KV execution table has no mirror"); }
    if (!valid_handle(handle)) { throw std::invalid_argument("Paged KV execution row is stale"); }
    return *mirror_rows_[static_cast<std::size_t>(handle.row_)];
}

} // namespace ninfer
