#pragma once

#include "core/device.h"

namespace ninfer::test {

struct CudaEnvironment {
    DeviceContext context;
    CudaExecutionView default_execution = context.execution_view(nullptr);
};

// Shared GPU 0 allocation and default-stream fixture. Tests that exercise independent
// execution owners may create their own DeviceContext.
inline CudaEnvironment& cuda_environment() {
    static CudaEnvironment environment;
    return environment;
}

} // namespace ninfer::test
