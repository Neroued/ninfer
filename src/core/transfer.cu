#include "core/transfer.h"
#include "core/device.h"

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

cudaMemcpyKind validate_copy(MemorySpan destination, ConstMemorySpan source, Device device,
                             bool asynchronous) {
    validate_memory(destination);
    validate_memory(source);
    if (destination.bytes != source.bytes) {
        throw std::invalid_argument("copy spans must have equal byte lengths");
    }
    for (const auto memory : {static_cast<ConstMemorySpan>(destination), source}) {
        if (memory.device != Device::cpu() && memory.device != device) {
            throw std::invalid_argument("copy memory does not belong to the execution device");
        }
        if (asynchronous && memory.device == Device::cpu() && !memory.host_pinned) {
            throw std::invalid_argument("asynchronous copy requires pinned host memory");
        }
    }
    if (source.device == Device::cpu() && destination.device == Device::cpu()) {
        throw std::invalid_argument("use CPU copy for host-to-host transfer");
    }
    if (source.device == destination.device) {
        const auto src = reinterpret_cast<std::uintptr_t>(source.data);
        const auto dst = reinterpret_cast<std::uintptr_t>(destination.data);
        if (src != dst && (src < dst ? dst - src : src - dst) < source.bytes) {
            throw std::invalid_argument("CUDA copy spans must not partially overlap");
        }
        return cudaMemcpyDeviceToDevice;
    }
    return source.device == Device::cpu() ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
}

void enqueue(MemorySpan destination, ConstMemorySpan source, cudaMemcpyKind kind,
             CudaExecutionView execution) {
    if (source.bytes == 0 || source.data == destination.data) { return; }
    const auto error =
        cudaMemcpyAsync(destination.data, source.data, source.bytes, kind, execution.stream());
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA copy: ") + cudaGetErrorString(error));
    }
}

void validate_host_copy(MemorySpan destination, CudaHostView source, Device device) {
    validate_memory(destination);
    validate_memory(source.host);
    if (destination.device != device || source.cuda_device != device ||
        source.host.device != Device::cpu() || !source.host.host_pinned ||
        (source.host.bytes && !source.device_address)) {
        throw std::invalid_argument("H2D copy requires a registered source for its CUDA device");
    }
}

void check_copy(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA host copy: ") + cudaGetErrorString(error));
    }
}

} // namespace

void copy_bytes_async(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution) {
    const auto kind = validate_copy(destination, source, execution.device(), true);
    CudaDeviceScope scope(execution.device());
    enqueue(destination, source, kind, execution);
}

void copy_rows_async(MemorySpan destination, std::size_t destination_pitch, ConstMemorySpan source,
                     std::size_t source_pitch, std::size_t width, std::size_t rows,
                     CudaExecutionView execution) {
    validate_memory(destination);
    validate_memory(source);
    if (!width || !rows) return;
    const auto fits = [&](std::size_t bytes, std::size_t pitch) {
        return width <= pitch && width <= bytes && rows - 1 <= (bytes - width) / pitch;
    };
    if (!fits(destination.bytes, destination_pitch) || !fits(source.bytes, source_pitch))
        throw std::out_of_range("pitched copy exceeds backing");
    const auto kind = validate_copy(destination.subspan(0, width), source.subspan(0, width),
                                    execution.device(), true);
    if (destination.device == source.device) {
        const auto dst       = reinterpret_cast<std::uintptr_t>(destination.data);
        const auto src       = reinterpret_cast<std::uintptr_t>(source.data);
        const auto dst_bytes = (rows - 1) * destination_pitch + width;
        const auto src_bytes = (rows - 1) * source_pitch + width;
        if (src <= dst ? dst - src < src_bytes : src - dst < dst_bytes)
            throw std::invalid_argument("pitched copy spans overlap");
    }
    CudaDeviceScope scope(execution.device());
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, destination_pitch, source.data, source_pitch,
                                 width, rows, kind, execution.stream()));
}

void copy_bytes(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution) {
    const auto kind = validate_copy(destination, source, execution.device(), false);
    CudaDeviceScope scope(execution.device());
    enqueue(destination, source, kind, execution);
    const auto error = cudaStreamSynchronize(execution.stream());
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA copy completion: ") + cudaGetErrorString(error));
    }
}

void copy_host_2d_async(MemorySpan destination, std::size_t destination_pitch, CudaHostView source,
                        std::size_t source_pitch, std::size_t width, std::size_t rows,
                        CudaExecutionView execution) {
    validate_host_copy(destination, source, execution.device());
    if (!width || !rows) { return; }
    const auto fits = [&](std::size_t bytes, std::size_t pitch) {
        return width <= pitch && width <= bytes && rows - 1 <= (bytes - width) / pitch;
    };
    if (!fits(destination.bytes, destination_pitch) || !fits(source.host.bytes, source_pitch)) {
        throw std::out_of_range("pitched H2D copy exceeds backing");
    }
    CudaDeviceScope scope(execution.device());
    // Explicit H2D consumes the CPU address; the batch API below consumes CUDA mappings.
    check_copy(cudaMemcpy2DAsync(destination.data, destination_pitch, source.host.data,
                                 source_pitch, width, rows, cudaMemcpyHostToDevice,
                                 execution.stream()));
}

CudaCopyBatch::CudaCopyBatch(Device device, std::size_t capacity)
    : device_(device), source_spans_(capacity), destination_spans_(capacity), sources_(capacity),
      destinations_(capacity), sizes_(capacity) {
    if (device.type() != DeviceType::CUDA || capacity == 0) {
        throw std::invalid_argument("CUDA copy batch requires a CUDA device and positive capacity");
    }
}

bool CudaCopyBatch::try_append(MemorySpan destination, ConstMemorySpan source) {
    (void)validate_copy(destination, source, device_, true);
    if (source.bytes == 0 || source.data == destination.data) return true;
    const auto overlaps = [](ConstMemorySpan left, ConstMemorySpan right) {
        if (left.device != right.device) return false;
        const auto l = reinterpret_cast<std::uintptr_t>(left.data);
        const auto r = reinterpret_cast<std::uintptr_t>(right.data);
        return l <= r ? r - l < left.bytes : l - r < right.bytes;
    };
    for (std::size_t i = 0; i < count_; ++i) {
        if (overlaps(destination, source_spans_[i]) ||
            overlaps(destination, destination_spans_[i]) ||
            overlaps(source, destination_spans_[i])) {
            return false;
        }
    }
    if (count_) {
        const auto last                  = count_ - 1;
        const auto& previous_source      = source_spans_[last];
        const auto& previous_destination = destination_spans_[last];
        if (previous_source.device == source.device &&
            previous_destination.device == destination.device &&
            static_cast<const std::byte*>(previous_source.data) + previous_source.bytes ==
                source.data &&
            static_cast<const std::byte*>(previous_destination.data) + previous_destination.bytes ==
                destination.data) {
            source_spans_[last].bytes += source.bytes;
            destination_spans_[last].bytes += destination.bytes;
            sizes_[last] += source.bytes;
            return true;
        }
    }
    if (count_ == sources_.size()) return false;
    source_spans_[count_]      = source;
    destination_spans_[count_] = destination;
    sources_[count_]           = source.data;
    destinations_[count_]      = destination.data;
    sizes_[count_++]           = source.bytes;
    return true;
}

void CudaCopyBatch::submit(CudaExecutionView execution) const {
    if (execution.device() != device_ || !execution.stream()) {
        throw std::invalid_argument("CUDA copy batch requires its device and an explicit stream");
    }
    if (count_ == 0) return;
    if (count_ == 1) {
        copy_bytes_async(destination_spans_[0], source_spans_[0], execution);
        return;
    }
    CudaDeviceScope scope(device_);
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first         = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(destinations_.data(), sources_.data(), sizes_.data(), count_,
                                    &attributes, &first, 1, execution.stream()));
}

CudaHostCopyBatch::CudaHostCopyBatch(Device device, std::size_t capacity)
    : device_(device), sources_(capacity), destinations_(capacity), sizes_(capacity) {
    if (device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("H2D batch requires a CUDA destination device");
    }
}

void CudaHostCopyBatch::append(MemorySpan destination, CudaHostView source) {
    validate_host_copy(destination, source, device_);
    if (destination.bytes != source.host.bytes) {
        throw std::invalid_argument("H2D copy spans must have equal byte lengths");
    }
    if (!source.host.bytes) { return; }
    if (count_ == capacity()) { throw std::length_error("H2D batch descriptor capacity exceeded"); }
    destinations_[count_] = destination.data;
    sources_[count_]      = source.device_address;
    sizes_[count_++]      = source.host.bytes;
}

void CudaHostCopyBatch::submit(CudaExecutionView execution) const {
    if (execution.device() != device_ || !execution.stream()) {
        throw std::invalid_argument("H2D batch requires its CUDA device and an explicit stream");
    }
    if (!count_) { return; }
    CudaDeviceScope scope(device_);
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first         = 0;
    check_copy(cudaMemcpyBatchAsync(destinations_.data(), sources_.data(), sizes_.data(), count_,
                                    &attributes, &first, 1, execution.stream()));
}

} // namespace ninfer
