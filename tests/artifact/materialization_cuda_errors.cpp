#include "artifact/binder.h"
#include "artifact/fixture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace {

enum class Failure { None, EventCreation, EventRecord };

struct Trace {
    Failure failure                = Failure::None;
    bool injected                  = false;
    bool upload_pending            = false;
    bool released_during_upload    = false;
    std::size_t uploads            = 0;
    std::size_t host_allocations   = 0;
    std::size_t host_releases      = 0;
    std::size_t device_allocations = 0;
    std::size_t device_releases    = 0;
};

Trace trace;

bool active() { return trace.failure != Failure::None; }

} // namespace

// Link-time wrappers affect only this test executable. The production path has no fault hooks.
extern "C" {
cudaError_t CUDARTAPI __real_cudaMalloc(void**, std::size_t);
cudaError_t CUDARTAPI __real_cudaMallocHost(void**, std::size_t);
cudaError_t CUDARTAPI __real_cudaFree(void*);
cudaError_t CUDARTAPI __real_cudaFreeHost(void*);
cudaError_t CUDARTAPI __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
cudaError_t CUDARTAPI __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                             cudaStream_t);
cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);

cudaError_t CUDARTAPI __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
    const auto status = __real_cudaMalloc(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.device_allocations; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaMallocHost(void** pointer, std::size_t bytes) {
    const auto status = __real_cudaMallocHost(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.host_allocations; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaFree(void* pointer) {
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = __real_cudaFree(pointer);
    if (active() && status == cudaSuccess) { ++trace.device_releases; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaFreeHost(void* pointer) {
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = __real_cudaFreeHost(pointer);
    if (active() && status == cudaSuccess) { ++trace.host_releases; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (trace.failure == Failure::EventCreation && !trace.injected) {
        trace.injected = true;
        *event         = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaEventCreateWithFlags(event, flags);
}

cudaError_t CUDARTAPI __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (trace.failure == Failure::EventRecord && !trace.injected) {
        trace.injected = true;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaEventRecord(event, stream);
}

cudaError_t CUDARTAPI __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                             std::size_t bytes, cudaMemcpyKind kind,
                                             cudaStream_t stream) {
    const auto status = __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (active() && status == cudaSuccess) {
        trace.upload_pending = true;
        ++trace.uploads;
    }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    if (active() && status == cudaSuccess) { trace.upload_pending = false; }
    return status;
}
}

namespace ninfer::test {

void materialization_cuda_errors(DeviceContext& device) {
    using namespace artifact;
    using namespace artifact_fixture;
    Fixture fixture;
    fixture.write();
    for (const auto failure : {Failure::EventCreation, Failure::EventRecord}) {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        trace       = {.failure = failure};
        bool caught = false;
        try {
            (void)materialize(reader, std::move(binder).finish(), device);
        } catch (const ArtifactError&) { caught = true; }
        const auto result = std::exchange(trace, {});
        require(result.injected && caught, "CUDA loading error did not propagate as an exception");
        require(result.host_allocations > 0 && result.device_allocations > 0 &&
                    result.host_allocations == result.host_releases &&
                    result.device_allocations == result.device_releases,
                "failed loading leaked pinned or device allocations");
        require(!result.upload_pending && !result.released_during_upload,
                "failed loading released storage before completing submitted transfers");
        if (failure == Failure::EventRecord) {
            require(result.uploads > 0, "event failure did not exercise an in-flight upload");
        }
    }
    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("matrix", {2, 130});
    auto recovered = materialize(reader, std::move(binder).finish(), device);
    std::array<std::byte, 528> received{};
    CUDA_CHECK(cudaMemcpy(received.data(), recovered.device_parent(reader.find("q5")).data,
                          received.size(), cudaMemcpyDeviceToHost));
    require(std::equal(received.begin(), received.end(), fixture.payload.begin() + 256),
            "successful loading after CUDA failure changed weight bytes");
}

} // namespace ninfer::test
