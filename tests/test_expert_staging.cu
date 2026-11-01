// Staging slots of the expert cache (core/expert_staging.h, T25): lifecycle and
// race safety of the prefetch protocol, with copies landing while windows run.
//
// A synthetic "model": L layers per window, 4 routed tokens x top-10 per layer from 64 experts of
// 2 764 800 B (the NVFP4 expert size; every 32-bit word of expert e holds a hash of (e, word)) in
// pinned host memory, B staging slots per parity set (2 sets). Per window the compute stream runs
// the epoch write and, per layer j: a "predictor" kernel that sets mapped flag j, the overlay of
// layer j (effective row from a device table with residents + the landed staged experts), and a
// reader that, for every routed expert the row maps to a staging slot, verifies the slot's words
// repeatedly for ~1.5 ms (a copy into a slot being read would be caught as bad words). The host,
// like the engine's execution thread, waits for flag j and copies layer j+1's chosen experts into
// set (j+1)%2 on a separate stream, each followed by its tag write; some layers get a 48 MB copy
// queued first so their experts land late (after the layer looked), into the next layers or the
// next window. Checks:
//   1. no bad word in any read (the slot was never rewritten while a layer read it);
//   2. every staged entry of an effective row names an expert copied for that layer in that window,
//      the table's residents keep their slot, a resident expert is never overridden;
//   3. per-layer stats = what the rows show (landed <= issued, used / used routes recounted);
//   4. both regimes happened: copies landed in time and copies landed late;
//   5. control: the same run issuing layer j+2's copies already at flag j (the set is rewritten
//      while layer j still reads it) IS caught (bad words > 0), so check 1 has teeth.
#include "core/device.h"
#include "core/expert_staging.h"

#include <cuda_runtime.h>

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace ninfer;
using namespace ninfer::expert_cache;

namespace {

constexpr int kLayers              = 8;
constexpr int kExperts             = kStagingExperts; // table width
constexpr int kBank                = 64;              // experts that exist in the bank
constexpr int kTokens              = 4;
constexpr int kRoutes              = 10 * kTokens;
constexpr int kB                   = 4; // staging slots per set
constexpr int kMainSlots           = 8; // dummy resident slots (never read here)
constexpr std::size_t kExpertBytes = 2764800;
constexpr std::size_t kWords       = kExpertBytes / 4;

__host__ __device__ inline std::uint32_t pattern(std::uint32_t e, std::uint32_t i) {
    std::uint32_t h = e * 0x9E3779B1U ^ (i * 0x85EBCA77U + 0x165667B1U);
    h ^= h >> 15;
    return h * 0x2C1B3C6DU;
}

__global__ void flag_kernel(std::uint32_t* flag) {
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(flag) = 1;
}

__device__ inline std::uint64_t now_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Every block verifies a stripe of every staged slot the row uses, until `duration_ns` passed
// (a layer of fixed duration).
__global__ void reader_kernel(const std::int32_t* row, const std::int32_t* ids, int routes,
                              const std::uint32_t* pool, int first_slot, std::uint64_t duration_ns,
                              unsigned long long* bad, unsigned* reads) {
    __shared__ int staged[kRoutes];
    __shared__ int n;
    if (threadIdx.x == 0) {
        n = 0;
        for (int i = 0; i < routes; ++i) {
            const int e = ids[i];
            const int s = row[e];
            if (s < first_slot) continue;
            bool dup = false;
            for (int k = 0; k < n; ++k) dup |= staged[k] == e;
            if (!dup) staged[n++] = e;
        }
    }
    __syncthreads();
    // The layer lasts `duration_ns` whether or not it uses staged experts (a model layer's time
    // does not depend on them), so later layers' copies have time to land.
    const std::uint64_t start = now_ns();
    unsigned long long local  = 0;
    unsigned passes           = 0;
    do {
        for (int k = 0; k < n; ++k) {
            const int e = staged[k];
            const std::uint32_t* slot =
                pool + static_cast<std::size_t>(row[e] - first_slot) * kWords;
            for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kWords;
                 i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                const std::uint32_t v = __ldcg(slot + i);
                local += v != pattern(static_cast<std::uint32_t>(e), static_cast<std::uint32_t>(i));
            }
        }
        ++passes;
    } while (now_ns() - start < duration_ns);
    if (local) atomicAdd(bad, local);
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(reads, passes);
}

struct Result {
    unsigned long long bad = 0;
    int failures           = 0;
    long issued = 0, landed = 0, late = 0, used = 0;
};

Result run(bool unsafe, std::uint32_t seed, int windows) {
    Result r;
    // Bank (pinned) and staging pool (device). Staging slot s of the pool = pool + s * kWords
    // (first_slot = kMainSlots: slots 0..7 are the "cache", never read by this test).
    std::uint32_t* bank = nullptr;
    CUDA_CHECK(cudaMallocHost(&bank, kBank * kExpertBytes));
    for (std::uint32_t e = 0; e < kBank; ++e) {
        for (std::size_t i = 0; i < kWords; ++i)
            bank[e * kWords + i] = pattern(e, static_cast<std::uint32_t>(i));
    }
    std::uint32_t *pool = nullptr, *tags = nullptr, *epoch = nullptr, *stats = nullptr,
                  *flags = nullptr;
    std::int32_t *table = nullptr, *rows = nullptr, *ids = nullptr;
    void* dummy                  = nullptr;
    void* dummy_src              = nullptr;
    constexpr std::size_t kDummy = 48u << 20;
    CUDA_CHECK(cudaMalloc(&pool, 2 * kB * kExpertBytes));
    CUDA_CHECK(cudaMalloc(&tags, 2 * kB * 4));
    CUDA_CHECK(cudaMalloc(&epoch, 4));
    CUDA_CHECK(cudaMalloc(&table, kLayers * kExperts * 4));
    CUDA_CHECK(cudaMalloc(&rows, kLayers * kExperts * 4));
    CUDA_CHECK(cudaMalloc(&ids, kLayers * kRoutes * 4));
    CUDA_CHECK(cudaMalloc(&dummy, kDummy));
    CUDA_CHECK(cudaMallocHost(&dummy_src, kDummy));
    CUDA_CHECK(cudaHostAlloc(&stats, kLayers * kStagingStatsWords * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&flags, kLayers * 4, cudaHostAllocMapped));
    unsigned long long* bad = nullptr;
    unsigned* reads         = nullptr;
    CUDA_CHECK(cudaMalloc(&bad, 8));
    CUDA_CHECK(cudaMalloc(&reads, 4));
    CUDA_CHECK(cudaMemset(bad, 0, 8));
    CUDA_CHECK(cudaMemset(reads, 0, 4));
    CUDA_CHECK(cudaMemset(tags, 0xFF, 2 * kB * 4));
    CUDA_CHECK(cudaMemset(pool, 0, 2 * kB * kExpertBytes));

    std::mt19937 rng(seed);
    std::vector<std::int32_t> host_table(kLayers * kExperts, -1);
    for (int l = 0; l < kLayers; ++l) {
        for (int k = 0; k < kMainSlots; ++k)
            host_table[l * kExperts + static_cast<int>(rng() % kBank)] = k;
    }
    CUDA_CHECK(cudaMemcpy(table, host_table.data(), host_table.size() * 4, cudaMemcpyHostToDevice));
    // The pageable upload may return before its DMA lands, and the non-blocking streams below do
    // not wait for the default stream: staging_overlay could read the table before it is written.
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t compute, copy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    std::vector<std::int32_t> host_ids(kLayers * kRoutes), host_rows(kLayers * kExperts);

    for (int w = 0; w < windows; ++w) {
        const std::uint32_t ep = static_cast<std::uint32_t>(w + 1);
        for (auto& v : host_ids) v = static_cast<std::int32_t>(rng() % kBank);
        CUDA_CHECK(cudaMemcpyAsync(ids, host_ids.data(), host_ids.size() * 4,
                                   cudaMemcpyHostToDevice, compute));
        for (int l = 0; l < kLayers; ++l) reinterpret_cast<volatile std::uint32_t*>(flags)[l] = 0;
        write_staging_word(compute, epoch, ep);
        for (int j = 0; j < kLayers; ++j) {
            flag_kernel<<<1, 1, 0, compute>>>(flags + j);
            staging_overlay(table + j * kExperts, tags + (j & 1) * kB, kB,
                            kMainSlots + (j & 1) * kB, epoch, j, ids + j * kRoutes, kRoutes,
                            rows + j * kExperts, stats + j * kStagingStatsWords, compute);
            reader_kernel<<<64, 256, 0, compute>>>(
                rows + j * kExperts, ids + j * kRoutes, kRoutes,
                pool + static_cast<std::size_t>(j & 1) * kB * kWords, kMainSlots + (j & 1) * kB,
                1500000ULL, bad, reads);
        }
        CUDA_CHECK(cudaGetLastError());
        // Host: on flag j, copy layer t's experts (t = j+1; the control also copies j+2 early).
        std::vector<std::vector<int>> issued(kLayers);
        std::vector<int> slot_expert(2 * kB, -1);
        for (int j = 0; j < kLayers; ++j) {
            while (reinterpret_cast<volatile std::uint32_t*>(flags)[j] == 0) _mm_pause();
            std::vector<int> targets;
            if (j + 1 < kLayers) targets.push_back(j + 1);
            if (unsafe && j + 2 < kLayers) targets.push_back(j + 2);
            for (const int t : targets) {
                if (!issued[t].empty()) continue;
                if (t % 3 == 0) {
                    CUDA_CHECK(
                        cudaMemcpyAsync(dummy, dummy_src, kDummy, cudaMemcpyHostToDevice, copy));
                }
                // Half from the layer's routed non-resident experts, the rest other non-resident
                // ones.
                std::vector<int> pick;
                for (int i = 0; i < kRoutes && static_cast<int>(pick.size()) < kB / 2; ++i) {
                    const int e = host_ids[t * kRoutes + i];
                    if (host_table[t * kExperts + e] < 0 &&
                        std::find(pick.begin(), pick.end(), e) == pick.end())
                        pick.push_back(e);
                }
                while (static_cast<int>(pick.size()) < kB) {
                    const int e = static_cast<int>(rng() % kBank);
                    if (host_table[t * kExperts + e] < 0 &&
                        std::find(pick.begin(), pick.end(), e) == pick.end())
                        pick.push_back(e);
                }
                for (int i = 0; i < kB; ++i) {
                    const int local = (t & 1) * kB + i;
                    CUDA_CHECK(cudaMemcpyAsync(pool + static_cast<std::size_t>(local) * kWords,
                                               bank + static_cast<std::size_t>(pick[i]) * kWords,
                                               kExpertBytes, cudaMemcpyHostToDevice, copy));
                    write_staging_word(copy, tags + local, staging_tag(ep, t, pick[i]));
                }
                issued[t] = pick;
                r.issued += kB;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(
            cudaMemcpy(host_rows.data(), rows, host_rows.size() * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < kLayers; ++j) {
            const std::int32_t* row = host_rows.data() + j * kExperts;
            const int first         = kMainSlots + (j & 1) * kB;
            int landed = 0, used = 0, used_routes = 0;
            std::set<int> used_set;
            for (int e = 0; e < kExperts; ++e) {
                const int base = host_table[j * kExperts + e];
                if (base >= 0) {
                    if (row[e] != base) {
                        std::cerr << "window " << w << " layer " << j << ": resident " << e
                                  << " overridden\n";
                        ++r.failures;
                    }
                    continue;
                }
                if (row[e] < 0) continue;
                const int i = row[e] - first;
                if (i < 0 || i >= kB || static_cast<int>(issued[j].size()) <= i ||
                    issued[j][i] != e) {
                    std::cerr << "window " << w << " layer " << j << ": row maps " << e
                              << " to slot " << row[e] << ", not a copy of this window and layer\n";
                    ++r.failures;
                    continue;
                }
                ++landed;
            }
            for (int i = 0; i < kRoutes; ++i) {
                const int e = host_ids[j * kRoutes + i];
                if (host_table[j * kExperts + e] < 0 && row[e] >= 0) {
                    ++used_routes;
                    used_set.insert(e);
                }
            }
            used                   = static_cast<int>(used_set.size());
            const std::uint32_t* s = stats + j * kStagingStatsWords;
            if (s[0] != ep || static_cast<int>(s[1]) != landed || static_cast<int>(s[2]) != used ||
                static_cast<int>(s[3]) != used_routes) {
                std::cerr << "window " << w << " layer " << j << ": stats {" << s[0] << "," << s[1]
                          << "," << s[2] << "," << s[3] << "} vs rows {" << ep << "," << landed
                          << "," << used << "," << used_routes << "}\n";
                ++r.failures;
            }
            r.landed += landed;
            r.used += used;
            r.late += static_cast<long>(issued[j].size()) - landed;
        }
        // Let the next window start while this window's late copies may still be in flight.
    }
    CUDA_CHECK(cudaStreamSynchronize(copy));
    CUDA_CHECK(cudaMemcpy(&r.bad, bad, 8, cudaMemcpyDeviceToHost));
    unsigned passes = 0;
    CUDA_CHECK(cudaMemcpy(&passes, reads, 4, cudaMemcpyDeviceToHost));
    std::cout << (unsafe ? "control" : "protocol") << ": windows " << windows << " issued "
              << r.issued << " landed in time " << r.landed << " late " << r.late << " used "
              << r.used << " reader passes " << passes << " bad words " << r.bad << '\n';
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(copy));
    for (void* p : {static_cast<void*>(pool), static_cast<void*>(tags), static_cast<void*>(epoch),
                    static_cast<void*>(table), static_cast<void*>(rows), static_cast<void*>(ids),
                    dummy, static_cast<void*>(bad), static_cast<void*>(reads)}) {
        CUDA_CHECK(cudaFree(p));
    }
    CUDA_CHECK(cudaFreeHost(bank));
    CUDA_CHECK(cudaFreeHost(dummy_src));
    CUDA_CHECK(cudaFreeHost(stats));
    CUDA_CHECK(cudaFreeHost(flags));
    return r;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Tag encoding.
    if (staging_tag(0xFFFE, 127, 511) == kStagingTagEmpty ||
        staging_tag(1, 2, 3) != ((1u << 16) | (2u << 9) | 3u)) {
        std::cerr << "staging_tag encoding\n";
        ++failures;
    }
    try {
        const Result ok = run(false, 7U, 40);
        failures += ok.failures;
        if (ok.bad != 0) {
            std::cerr << "protocol: " << ok.bad << " bad words read from staging slots\n";
            ++failures;
        }
        if (ok.landed == 0 || ok.late == 0 || ok.used == 0) {
            std::cerr << "protocol: the run did not exercise both in-time and late copies\n";
            ++failures;
        }
        const Result control = run(true, 7U, 20);
        if (control.bad == 0) {
            std::cerr << "control: rewriting a set one layer early was not detected\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "expert staging test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "expert staging: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK expert staging lifecycle and race safety\n";
    return 0;
}
