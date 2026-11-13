#pragma once

// ninfer::models::qwen3_5::execution - side streams for the tp 2 GDN gating (PR #64).
//
// At tp 2 and T <= 32 (decode and verification windows, where the gating kernel takes no
// workspace) the plain block's GDN gating runs on these streams, forked from the main streams
// after the input norm and joined before the recurrence, so its small GEMM (~4 us, once per GDN
// layer and round) overlaps the Q|K|V|Z projection instead of preceding it. Same kernels and
// inputs: bit-identical. The Program owns one instance over its two devices, declared before the
// decode graph families: captured graphs reference these streams and events.

#include "core/device.h"
#include "core/device_scope.h"

#include <cuda_runtime.h>

#include <array>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

struct GdnGatingOverlap {
    ExecutionContext side;
    std::array<cudaEvent_t, 2> fork{};
    std::array<cudaEvent_t, 2> join{};

    // Leaves the caller's current device as it found it (the Program builds it inside a rank-1
    // scope; the ExecutionContext constructor makes its device 0 current).
    explicit GdnGatingOverlap(const ExecutionContext& main)
        : GdnGatingOverlap(main, ScopedCurrentDevice{}) {}

    ~GdnGatingOverlap() {
        const ScopedCurrentDevice restore;
        for (std::size_t r = 0; r < 2; ++r) {
            if (cudaSetDevice(side.dev[r]->device) != cudaSuccess) { continue; }
            if (fork[r] != nullptr) { cudaEventDestroy(fork[r]); }
            if (join[r] != nullptr) { cudaEventDestroy(join[r]); }
        }
    }

    GdnGatingOverlap(const GdnGatingOverlap&)            = delete;
    GdnGatingOverlap& operator=(const GdnGatingOverlap&) = delete;

private:
    // `restore` lives until the constructor returns: it is a parameter, so it is built before
    // `side` and destroyed after the body.
    GdnGatingOverlap(const ExecutionContext& main, ScopedCurrentDevice&& restore)
        : side(std::vector<int>{main.dev[0]->device, main.dev[1]->device}) {
        (void)restore;
        for (std::size_t r = 0; r < 2; ++r) {
            CUDA_CHECK(cudaSetDevice(side.dev[r]->device));
            CUDA_CHECK(cudaEventCreateWithFlags(&fork[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming));
        }
    }
};

} // namespace ninfer::models::qwen3_5::execution
