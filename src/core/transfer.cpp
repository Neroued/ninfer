#include "core/transfer.h"

#include <cstring>
#include <stdexcept>

namespace ninfer {

void copy_bytes(MemorySpan destination, ConstMemorySpan source) {
    validate_memory(destination);
    validate_memory(source);
    if (destination.device != Device::cpu() || source.device != Device::cpu()) {
        throw std::invalid_argument("CPU copy requires CPU memory");
    }
    if (destination.bytes != source.bytes) {
        throw std::invalid_argument("copy spans must have equal byte lengths");
    }
    if (source.bytes != 0) { std::memmove(destination.data, source.data, source.bytes); }
}

} // namespace ninfer
