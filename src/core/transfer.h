#pragma once

#include "core/memory.h"

#include <vector>

namespace ninfer {

class CudaExecutionView;

// Exact byte copies between equally sized spans. CPU copies allow overlapping ranges.
// CUDA copies reject partial overlap. The synchronous overload waits for its execution stream.
void copy_bytes(MemorySpan destination, ConstMemorySpan source);
void copy_bytes(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution);

// Host endpoints must be pinned; CUDA endpoints must belong to execution.device().
// The caller keeps both spans alive and orders all consumers through the stream or an event.
void copy_bytes_async(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution);

// Copies unchanged rows between disjoint spans. Pitches include row gaps; both spans cover
// every addressed row. Device and pinned-Host requirements follow copy_bytes_async.
void copy_rows_async(MemorySpan destination, std::size_t destination_pitch, ConstMemorySpan source,
                     std::size_t source_pitch, std::size_t width, std::size_t rows,
                     CudaExecutionView execution);

// Copies rows of unchanged bytes. Both pitches include the gap between rows;
// source and destination spans cover all addressed rows.
void copy_host_2d_async(MemorySpan destination, std::size_t destination_pitch, CudaHostView source,
                        std::size_t source_pitch, std::size_t width, std::size_t rows,
                        CudaExecutionView execution);

// Bounded CUDA copies between Device and pinned Host spans. Adjacent source/destination
// runs are coalesced. try_append returns false on capacity or an ordering dependency;
// submit and clear before appending that copy. Backing remains alive until stream retirement.
class CudaCopyBatch {
public:
    CudaCopyBatch(Device device, std::size_t capacity);

    void clear() noexcept { count_ = 0; }

    [[nodiscard]] bool try_append(MemorySpan destination, ConstMemorySpan source);
    void submit(CudaExecutionView execution) const;

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    Device device_;
    std::vector<ConstMemorySpan> source_spans_;
    std::vector<MemorySpan> destination_spans_;
    std::vector<const void*> sources_;
    std::vector<void*> destinations_;
    std::vector<std::size_t> sizes_;
    std::size_t count_ = 0;
};

// Preallocated descriptors for independent H2D copies. Destinations do not overlap;
// source backing and mappings outlive completion. Calls within a batch are unordered.
class CudaHostCopyBatch {
public:
    CudaHostCopyBatch(Device device, std::size_t capacity);

    void clear() noexcept { count_ = 0; }

    void append(MemorySpan destination, CudaHostView source);
    void submit(CudaExecutionView execution) const;

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    [[nodiscard]] std::size_t capacity() const noexcept { return sources_.size(); }

private:
    Device device_;
    std::vector<const void*> sources_;
    std::vector<void*> destinations_;
    std::vector<std::size_t> sizes_;
    std::size_t count_ = 0;
};

} // namespace ninfer
