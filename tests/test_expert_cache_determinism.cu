// GPU test of deterministic publication in ExpertCache (NINFER_QWEN4_FILL_PUBLISH_LAG). The same
// routing history runs twice: once with the host waiting for every copy (A), once with the copies
// slowed by a concurrent host-to-device stream and the host never waiting for the fill stream (B).
// With a publish lag the cache must publish exactly the same tables after every begin_window in
// both runs (the residency is a function of the routing alone), and must have actually waited for
// copies in B. In timed mode (lag -1) the runs must differ, which shows the perturbation is strong
// enough to matter. In every window a kernel reads each published slot through the device table
// while fills run and checks its bytes. Exit 77 when no GPU is usable.

#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace ninfer::expert_cache;

constexpr std::int32_t kLayers   = 3;
constexpr std::int32_t kExperts  = 64;
constexpr std::int32_t kSlots    = 16; // one GPU, per layer
constexpr std::int64_t kBytes    = 1 << 20;
constexpr std::int64_t kWords    = kBytes / 4;
constexpr int kWindows           = 160;
constexpr std::int64_t kHogBytes = std::int64_t(128) << 20;

int failures = 0;

__host__ __device__ std::uint32_t word(std::int32_t l, std::int32_t e, std::int64_t i) {
    std::uint32_t h = 2166136261U;
    h               = (h ^ static_cast<std::uint32_t>(l)) * 16777619U;
    h               = (h ^ static_cast<std::uint32_t>(e)) * 16777619U;
    h               = (h ^ static_cast<std::uint32_t>(i)) * 16777619U;
    return h;
}

void write_expert(std::byte* to, std::int32_t l, std::int32_t e) {
    auto* w = reinterpret_cast<std::uint32_t*>(to);
    for (std::int64_t i = 0; i < kWords; ++i) { w[i] = word(l, e, i); }
}

bool intact(const std::byte* at, std::int32_t l, std::int32_t e) {
    const auto* w = reinterpret_cast<const std::uint32_t*>(at);
    for (std::int64_t i = 0; i < kWords; ++i) {
        if (w[i] != word(l, e, i)) { return false; }
    }
    return true;
}

void must(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// One block per table entry: a published slot must hold its expert's bytes.
__global__ void verify_kernel(const std::int32_t* table, const std::uint32_t* pool,
                              unsigned long long* bad) {
    const std::int32_t cell = blockIdx.x;
    const std::int32_t slot = table[cell];
    if (slot < 0) { return; }
    const std::int32_t l   = cell / kExperts;
    const std::int32_t e   = cell % kExperts;
    const std::uint32_t* w = pool + static_cast<std::int64_t>(slot) * kWords;
    unsigned long long n   = 0;
    for (std::int64_t i = threadIdx.x; i < kWords; i += blockDim.x) {
        n += w[i] != word(l, e, i) ? 1ULL : 0ULL;
    }
    if (n != 0) { atomicAdd(bad, n); }
}

__global__ void spin_kernel(unsigned long long ns) {
    unsigned long long start = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    for (;;) {
        unsigned long long now = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - start >= ns) { break; }
    }
}

struct Result {
    std::vector<std::int32_t> tables; // published_slot of every cell after every begin_window
    CacheCounters counters;
    unsigned long long bad_words = 0;
};

Result run(std::int32_t lag, bool perturbed) {
    PolicyConfig policy;
    policy.layers          = kLayers;
    policy.experts         = kExperts;
    policy.partitions      = 1;
    policy.slots_per_layer = kSlots;

    void* bank_memory    = nullptr;
    const auto bank_size = static_cast<std::size_t>(kLayers) * kExperts * kBytes;
    must(cudaHostAlloc(&bank_memory, bank_size, cudaHostAllocPortable));
    auto* bank = static_cast<std::byte*>(bank_memory);

    ExpertLayout layout;
    layout.planes         = 1;
    layout.plane_bytes[0] = kBytes;
    layout.bank_stride[0] = kBytes;
    layout.pool_stride[0] = kBytes;
    for (std::int32_t l = 0; l < kLayers; ++l) {
        layout.bank_base.push_back(bank + static_cast<std::int64_t>(l) * kExperts * kBytes);
        for (std::int32_t e = 0; e < kExperts; ++e) {
            write_expert(bank + (static_cast<std::int64_t>(l) * kExperts + e) * kBytes, l, e);
        }
    }

    const auto device_slots = static_cast<std::size_t>(ExpertCache::pool_slots(policy, 0));
    void* device_pool       = nullptr;
    void* device_table      = nullptr;
    void* hog_dst           = nullptr;
    void* hog_src           = nullptr;
    unsigned long long* bad = nullptr;
    must(cudaMalloc(&device_pool, device_slots * kBytes));
    must(cudaMalloc(&device_table,
                    static_cast<std::size_t>(kLayers) * kExperts * sizeof(std::int32_t)));
    must(cudaMalloc(&hog_dst, kHogBytes));
    must(cudaHostAlloc(&hog_src, kHogBytes, cudaHostAllocPortable));
    must(cudaMalloc(&bad, sizeof(unsigned long long)));
    must(cudaMemset(bad, 0, sizeof(unsigned long long)));
    cudaStream_t stream = nullptr;
    cudaStream_t hog    = nullptr;
    must(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    must(cudaStreamCreateWithFlags(&hog, cudaStreamNonBlocking));
    cudaEvent_t verified = nullptr;
    must(cudaEventCreateWithFlags(&verified, cudaEventDisableTiming));

    GpuBinding binding;
    binding.device       = 0;
    binding.pool_base[0] = static_cast<std::byte*>(device_pool);
    binding.table        = static_cast<std::int32_t*>(device_table);
    const std::array<GpuBinding, 1> bindings{binding};
    ExpertCache cache(policy, layout, bindings, kBytes * 4);
    cache.set_publish_lag(lag);

    std::vector<std::vector<std::int32_t>> profile(kLayers);
    for (std::int32_t l = 0; l < kLayers; ++l) {
        for (std::int32_t e = kExperts - kSlots; e < kExperts; ++e) { profile[l].push_back(e); }
    }
    cache.initialize(profile);

    const std::array<cudaStream_t, 1> streams{stream};
    std::mt19937 routing(11);                // the same history in every run
    std::mt19937 polls(perturbed ? 5U : 7U); // a different number of poll() calls
    const auto route = [&](std::int32_t window) {
        std::vector<Routed> routed;
        std::vector<std::int32_t> ids;
        for (int i = 0; i < 12; ++i) {
            const auto u = static_cast<double>(routing()) / std::mt19937::max();
            ids.push_back((window / 8 + static_cast<std::int32_t>(u * u * 40)) % kExperts);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (const std::int32_t e : ids) { routed.push_back({e, 2}); }
        return routed;
    };
    const auto some_polls = [&] {
        for (int i = static_cast<int>(polls() % 6); i > 0; --i) { cache.poll(); }
    };

    Result result;
    for (int window = 0; window < kWindows; ++window) {
        cache.begin_window(streams);
        for (std::int32_t l = 0; l < kLayers; ++l) {
            for (std::int32_t e = 0; e < kExperts; ++e) {
                result.tables.push_back(cache.published_slot(l, e));
            }
        }
        verify_kernel<<<kLayers * kExperts, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(device_table),
            static_cast<const std::uint32_t*>(device_pool), bad);
        must(cudaGetLastError());
        must(cudaEventRecord(verified, stream));
        if (perturbed) {
            must(cudaMemcpyAsync(hog_dst, hog_src, kHogBytes, cudaMemcpyHostToDevice, hog));
            spin_kernel<<<1, 1, 0, stream>>>(1000000ULL);
            must(cudaGetLastError());
        }
        // The window has left the layers once their routing is published (as the engine's
        // mailbox): their insertions may be copied mid-window.
        must(cudaEventSynchronize(verified));
        (void)cache.observe(0, route(window), 8);
        some_polls();
        cache.submit_fills_unfenced();
        (void)cache.observe(1, route(window), 8);
        some_polls();
        cache.submit_fills_unfenced();
        must(cudaStreamSynchronize(stream)); // the engine synchronizes before end_window
        cache.end_window(streams);
        if (!perturbed) { must(cudaDeviceSynchronize()); }
        (void)cache.observe(2, route(window), 8);
        cache.submit_fills();
        if (!perturbed) {
            must(cudaDeviceSynchronize());
            cache.poll();
        }
        some_polls();
    }
    cache.flush();
    cache.begin_window(streams); // publishes what flush landed
    cache.end_window(streams);
    must(cudaDeviceSynchronize());
    must(cudaMemcpy(&result.bad_words, bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Every expert intact where the cache says.
    std::vector<std::byte> copy(kBytes);
    for (std::int32_t l = 0; l < kLayers; ++l) {
        for (std::int32_t e = 0; e < kExperts; ++e) {
            const std::int32_t s = cache.published_slot(l, e);
            bool ok              = false;
            if (s >= 0) {
                must(cudaMemcpy(copy.data(), static_cast<std::byte*>(device_pool) + s * kBytes,
                                kBytes, cudaMemcpyDeviceToHost));
                ok = intact(copy.data(), l, e);
            } else {
                const std::byte* host = layout.bank(l, 0, e);
                ok                    = host != nullptr && intact(host, l, e);
            }
            if (!ok) {
                std::fprintf(stderr, "lag %d %s: layer %d expert %d lost (slot %d)\n", lag,
                             perturbed ? "B" : "A", l, e, s);
                ++failures;
            }
        }
    }
    result.counters = cache.counters();

    must(cudaEventDestroy(verified));
    must(cudaStreamDestroy(stream));
    must(cudaStreamDestroy(hog));
    must(cudaFree(bad));
    must(cudaFreeHost(hog_src));
    must(cudaFree(hog_dst));
    must(cudaFree(device_pool));
    must(cudaFree(device_table));
    must(cudaFreeHost(bank_memory));
    return result;
}

bool same_history(const Result& a, const Result& b) {
    return a.tables == b.tables && a.counters.fills_issued == b.counters.fills_issued &&
           a.counters.fills_dropped == b.counters.fills_dropped &&
           a.counters.published == b.counters.published &&
           a.counters.unpublished == b.counters.unpublished;
}

void report(const char* name, const Result& r) {
    std::printf("%s: fills %lld, dropped %lld, published %lld, waits %lld (%.0f us), "
                "bad words %llu\n",
                name, static_cast<long long>(r.counters.fills_issued),
                static_cast<long long>(r.counters.fills_dropped),
                static_cast<long long>(r.counters.published),
                static_cast<long long>(r.counters.publish_waits), r.counters.publish_wait_us,
                r.bad_words);
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    must(cudaSetDevice(0));

    const Result lag_a   = run(1, false);
    const Result lag_b   = run(1, true);
    const Result timed_a = run(-1, false);
    const Result timed_b = run(-1, true);
    report("lag 1, A", lag_a);
    report("lag 1, B", lag_b);
    report("timed, A", timed_a);
    report("timed, B", timed_b);

    for (const Result* r : {&lag_a, &lag_b, &timed_a, &timed_b}) {
        if (r->bad_words != 0) {
            std::fprintf(stderr, "a window read a slot while it was rewritten\n");
            ++failures;
        }
    }
    if (lag_a.counters.fills_issued < 200) {
        std::fprintf(stderr, "too few fills to mean anything\n");
        ++failures;
    }
    if (!same_history(lag_a, lag_b)) {
        std::fprintf(stderr, "lag 1: the residency depends on copy timing\n");
        ++failures;
    }
    if (lag_b.counters.publish_waits == 0) {
        std::fprintf(stderr, "lag 1: B never waited for a copy (perturbation too weak)\n");
        ++failures;
    }
    if (same_history(timed_a, timed_b)) {
        std::fprintf(stderr, "timed: A and B agree (perturbation too weak to test anything)\n");
        ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::puts("ok");
    return 0;
}
