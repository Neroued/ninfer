#include "core/buffer.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace ninfer {

HostBuffer::HostBuffer(std::size_t bytes, std::size_t alignment)
    : bytes_(bytes), alignment_(alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("host alignment must be a nonzero power of two");
    }
    alignment_ = std::max(alignment, alignof(std::max_align_t));
    if (bytes != 0) { data_ = ::operator new(bytes, std::align_val_t{alignment_}); }
}

HostBuffer::~HostBuffer() { ::operator delete(data_, std::align_val_t{alignment_}); }

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
      alignment_(other.alignment_) {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
        HostBuffer previous(std::move(*this));
        data_      = std::exchange(other.data_, nullptr);
        bytes_     = std::exchange(other.bytes_, 0);
        alignment_ = other.alignment_;
    }
    return *this;
}

} // namespace ninfer
