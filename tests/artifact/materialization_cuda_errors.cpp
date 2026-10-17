#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/fixture.h"
#include "symbol_interpose.h"

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

extern "C" {

cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t bytes) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(void**, std::size_t)>("cudaMalloc");
    const auto status = real(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.device_allocations; }
    return status;
}

cudaError_t CUDARTAPI cudaMallocHost(void** pointer, std::size_t bytes) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(void**, std::size_t)>("cudaMallocHost");
    const auto status = real(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.host_allocations; }
    return status;
}

cudaError_t CUDARTAPI cudaFree(void* pointer) {
    static const auto real = ninfer::test::next_symbol<cudaError_t (*)(void*)>("cudaFree");
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = real(pointer);
    if (active() && status == cudaSuccess) { ++trace.device_releases; }
    return status;
}

cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    static const auto real = ninfer::test::next_symbol<cudaError_t (*)(void*)>("cudaFreeHost");
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = real(pointer);
    if (active() && status == cudaSuccess) { ++trace.host_releases; }
    return status;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    static const auto real = ninfer::test::next_symbol<cudaError_t (*)(cudaEvent_t*, unsigned int)>(
        "cudaEventCreateWithFlags");
    if (trace.failure == Failure::EventCreation && !trace.injected) {
        trace.injected = true;
        *event         = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return real(event, flags);
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(cudaEvent_t, cudaStream_t)>("cudaEventRecord");
    if (trace.failure == Failure::EventRecord && !trace.injected) {
        trace.injected = true;
        return cudaErrorInvalidResourceHandle;
    }
    return real(event, stream);
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source, std::size_t bytes,
                                      cudaMemcpyKind kind, cudaStream_t stream) {
    static const auto real = ninfer::test::next_symbol<cudaError_t (*)(
        void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t)>("cudaMemcpyAsync");
    const auto status      = real(destination, source, bytes, kind, stream);
    if (active() && status == cudaSuccess) {
        trace.upload_pending = true;
        ++trace.uploads;
    }
    return status;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(cudaStream_t)>("cudaStreamSynchronize");
    const auto status = real(stream);
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
            (void)materialize(reader,
                              plan_materialization(
                                  reader,
                                  std::array{PlacementRequest{reader.find("q5"), device.device()}},
                                  std::move(binder).finish()),
                              device.transfer_view());
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
    auto recovered =
        materialize(reader,
                    plan_materialization(
                        reader, std::array{PlacementRequest{reader.find("q5"), device.device()}},
                        std::move(binder).finish()),
                    device.transfer_view());
    std::array<std::byte, 528> received{};
    CUDA_CHECK(cudaMemcpy(received.data(), recovered.device_parent(reader.find("q5")).data,
                          received.size(), cudaMemcpyDeviceToHost));
    require(std::equal(received.begin(), received.end(), fixture.payload.begin() + 256),
            "successful loading after CUDA failure changed weight bytes");
}

} // namespace ninfer::test
