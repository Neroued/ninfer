#pragma once

// Stream-ordered copies and fills between any device-addressable memory (device memory, pinned
// and device-mapped Host memory) done by kernels. cudaMemcpyAsync between two Host addresses is
// synchronous with respect to the host and not ordered after earlier work of the stream, so the
// Host-resident KV planes of active KV in RAM (KVPlaneGeometry::host) use these instead. The
// kernels run on `stream`'s device (made current for the launch).

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

void uva_copy_async(void* destination, const void* source, std::size_t bytes, cudaStream_t stream);
// `height` rows of `width` bytes; row r is at destination + r * destination_pitch.
void uva_copy_2d_async(void* destination, std::size_t destination_pitch, const void* source,
                       std::size_t source_pitch, std::size_t width, std::size_t height,
                       cudaStream_t stream);
void uva_fill_async(void* destination, unsigned char value, std::size_t bytes, cudaStream_t stream);

} // namespace ninfer
