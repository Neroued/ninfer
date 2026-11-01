#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/fixture.h"
#include "symbol_interpose.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>

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

struct Alignment {
    unsigned memory = 0, offset = 0;
    bool shift_pinned = false;
};

Alignment alignment;
std::vector<std::pair<void*, void*>> pinned_views;
std::size_t pinned_bytes = 0;

struct AlignmentScope {
    AlignmentScope(unsigned memory, unsigned offset, bool shift_pinned = false) {
        ninfer::test::artifact_fixture::require(!alignment.memory, "nested alignment injection");
        alignment    = {memory, offset, shift_pinned};
        pinned_bytes = 0;
    }

    ~AlignmentScope() { alignment = {}; }
};

} // namespace

extern "C" {

int statx(int descriptor, const char* path, int flags, unsigned mask,
          struct statx* status) noexcept {
    static const auto real = ninfer::test::next_symbol<decltype(&statx)>("statx");
    const auto result      = real(descriptor, path, flags, mask, status);
    if (!result && alignment.memory && (mask & STATX_DIOALIGN) && (flags & AT_EMPTY_PATH)) {
        status->stx_mask |= STATX_DIOALIGN;
        status->stx_dio_mem_align    = alignment.memory;
        status->stx_dio_offset_align = alignment.offset;
    }
    return result;
}

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
    const auto status = real(pointer, bytes + (alignment.shift_pinned ? 8192 : 0));
    if (status == cudaSuccess && alignment.memory) {
        pinned_bytes += bytes;
        if (alignment.shift_pinned) {
            auto* original     = *pointer;
            const auto address = reinterpret_cast<std::uintptr_t>(original);
            const auto padding = (4096 - address % 4096) % 4096 + 256;
            *pointer           = static_cast<std::byte*>(original) + padding;
            pinned_views.emplace_back(*pointer, original);
        }
    }
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
    const auto found       = std::find_if(pinned_views.begin(), pinned_views.end(),
                                          [&](const auto& entry) { return entry.first == pointer; });
    if (found != pinned_views.end()) {
        pointer = found->second;
        pinned_views.erase(found);
    }
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

void materialization_alignment_checks(std::optional<CudaExecutionView> execution) {
    using namespace artifact;
    using namespace artifact_fixture;
    {
        AlignmentScope scope(8192, 128 * 1024, execution.has_value());
        Fixture fixture;
        fixture.write();
        Reader reader(fixture.entry);
        const auto matrix = reader.find("q5"), divisors = reader.find("divisors");
        std::vector<PlacementRequest> requests{{matrix, Device::cpu()}, {divisors, Device::cpu()}};
        if (execution) requests.push_back({matrix, execution->device()});
        const auto backing = materialize(reader, plan_materialization(reader, requests), execution);
        require(std::equal(backing.host_parent(matrix).data, backing.host_parent(matrix).data + 528,
                           fixture.payload.begin() + 256),
                "aligned loading changed weight bytes");
        require(std::equal(backing.host_parent(divisors).data,
                           backing.host_parent(divisors).data + 8, fixture.payload.begin() + 1024),
                "aligned loading changed a neighboring tensor");
        require(backing.stats().read_bytes ==
                    std::filesystem::file_size(fixture.entry) +
                        std::filesystem::file_size(fixture.directory / "weights-second.bin"),
                "overlapping aligned windows were read more than once or lost EOF bytes");
        if (execution) {
            std::array<std::byte, 528> actual{};
            CUDA_CHECK(cudaMemcpy(actual.data(), backing.device_parent(matrix).data, actual.size(),
                                  cudaMemcpyDeviceToHost));
            require(std::equal(actual.begin(), actual.end(), fixture.payload.begin() + 256),
                    "aligned fanout upload changed weight bytes");
            require(pinned_bytes == backing.stats().peak_staging_bytes && pinned_views.empty(),
                    "staging accounting or aligned pinned ownership is incorrect");
        }
    }
    {
        AlignmentScope scope(8192, 4096);
        Fixture fixture;
        const std::size_t bytes = 1024 * 1024 + 258;
        fixture.payload.resize(bytes);
        for (std::size_t i = 0; i < bytes; ++i)
            fixture.payload[i] = std::byte((i * 17 + i / 73) % 251);
        fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
                        {"objects", Json::array({{{"id", "large"},
                                                  {"kind", "tensor"},
                                                  {"shape", {bytes / 2}},
                                                  {"format", "bf16"},
                                                  {"layout", "contiguous_le_v1"},
                                                  {"offset", 0},
                                                  {"bytes", bytes}}})},
                        {"bindings", {{"large", {{"object", "large"}}}}},
                        {"uses", Json::array()},
                        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
        fixture.write();
        Reader reader(fixture.entry);
        const auto object = reader.find("large");
        const std::array requests{PlacementRequest{object, Device::cpu()}};
        const auto backing = materialize(reader, plan_materialization(reader, requests));
        require(std::equal(backing.host_parent(object).data,
                           backing.host_parent(object).data + bytes, fixture.payload.begin()),
                "Host in-place loading changed weight bytes");
        require(backing.stats().peak_staging_bytes <= 8192 && backing.stats().read_bytes == bytes,
                "Host interior was staged or boundary reads were duplicated");
    }
}

void materialization_cuda_errors(DeviceContext& device) {
    using namespace artifact;
    using namespace artifact_fixture;
    Fixture fixture;
    fixture.write();
    AlignmentScope scope(8192, 128 * 1024, true);
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
        require(pinned_views.empty(), "failed loading leaked the original pinned allocation");
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
