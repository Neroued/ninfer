#pragma once

#include "core/device.h"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail {

template <int Bytes, auto Kernel, bool AlwaysDynamic = false, class... Metadata>
int prepare_causal_shared(const Metadata&... metadata) {
    if constexpr (sizeof...(Metadata) == 0) {
        return common::prepare_shared<Bytes, Kernel, AlwaysDynamic>();
    } else if constexpr (Bytes > 48 * 1024 || (AlwaysDynamic && Bytes > 0)) {
        // CUDA function attributes are per device. The execution owner binds the thread.
        static thread_local int prepared_device = -1;
        const int device                        = (metadata.segments.device_id, ...);
        if (prepared_device != device) {
            CUDA_CHECK(
                cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
            prepared_device = device;
        }
        return Bytes;
    } else {
        return AlwaysDynamic ? Bytes : 0;
    }
}

} // namespace ninfer::ops::detail
