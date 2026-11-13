#include "core/expert_cache.h"

#include "core/device.h"       // CUDA_CHECK
#include "core/device_scope.h" // ScopedCurrentDevice

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::expert_cache {

namespace {

// SM fills (expert_cache.h): one kernel per fill batch, the batch's (dst, src, 16-byte units)
// as a kernel parameter (copied at launch: 3 KiB for 16 experts x 8 planes).
constexpr int kSmFillThreads   = 256;
constexpr int kSmFillUnroll    = 4; // 16-byte loads in flight per thread
constexpr int kMaxSmFillCopies = ExpertCache::kFillChunkExperts * kMaxPlanes;

struct SmFillCopies {
    std::int32_t count = 0;
    const uint4* src[kMaxSmFillCopies];
    uint4* dst[kMaxSmFillCopies];
    std::int64_t units[kMaxSmFillCopies];
};

// Copies every (src, dst, units) of the batch in order: each thread keeps kSmFillUnroll loads in
// flight, a warp reads 512 contiguous bytes per load, the grid strides over the copy. Streaming
// loads and stores: neither the host bank nor the slot just written belongs in L2.
__global__ __launch_bounds__(kSmFillThreads) void sm_fill_kernel(const SmFillCopies c) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first  = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int k = 0; k < c.count; ++k) {
        const uint4* src  = c.src[k];
        uint4* dst        = c.dst[k];
        const long long n = c.units[k];
        for (long long base = first; base < n; base += stride * kSmFillUnroll) {
            uint4 v[kSmFillUnroll];
#pragma unroll
            for (int u = 0; u < kSmFillUnroll; ++u) {
                const long long i = base + u * stride;
                if (i < n) { v[u] = __ldcs(src + i); }
            }
#pragma unroll
            for (int u = 0; u < kSmFillUnroll; ++u) {
                const long long i = base + u * stride;
                if (i < n) { __stcs(dst + i, v[u]); }
            }
        }
    }
}

// Host memory a kernel on the current device reads at the same address (mapped, unified
// addressing).
bool device_reads_in_place(const void* p) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost && attr.devicePointer == p;
}

// Table updates travel as kernel parameters (copied at launch), so the host never shares a
// buffer with an update in flight. 8 KiB of parameters per launch (CUDA 12.1+ allows 32 KiB).
constexpr std::int32_t kMaxUpdates = 1020;

struct TableUpdates {
    std::int32_t count = 0;
    std::int32_t index[kMaxUpdates];
    std::int32_t value[kMaxUpdates];
};

__global__ void apply_table_updates_kernel(std::int32_t* __restrict__ table,
                                           const TableUpdates updates) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < updates.count) { table[updates.index[i]] = updates.value[i]; }
}

__global__ void map_slots_kernel(const std::int32_t* __restrict__ ids, std::int32_t count,
                                 const std::int32_t* __restrict__ layer_table, std::int32_t experts,
                                 std::int32_t* __restrict__ slot_ids) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) { return; }
    const std::int32_t e = ids[i];
    slot_ids[i]          = (e >= 0 && e < experts) ? layer_table[e] : -1;
}

void launch_updates(std::int32_t* table, const std::vector<std::int32_t>& index,
                    const std::vector<std::int32_t>& value, cudaStream_t stream,
                    std::int64_t& launches) {
    TableUpdates u;
    for (std::size_t first = 0; first < index.size(); first += kMaxUpdates) {
        const std::size_t n = std::min<std::size_t>(kMaxUpdates, index.size() - first);
        u.count             = static_cast<std::int32_t>(n);
        std::copy_n(index.begin() + static_cast<std::ptrdiff_t>(first), n, u.index);
        std::copy_n(value.begin() + static_cast<std::ptrdiff_t>(first), n, u.value);
        apply_table_updates_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
            table, u);
        CUDA_CHECK(cudaGetLastError());
        ++launches;
    }
}

bool pinned_host(const void* p) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost;
}

} // namespace

bool pinned_host_memory(const void* p) noexcept { return pinned_host(p); }

std::int64_t ExpertCache::pool_slots(const PolicyConfig& policy, std::int32_t gpu) {
    return ExpertCachePolicy::pool_slot_count(policy, gpu);
}

ExpertCache::ExpertCache(const PolicyConfig& policy, ExpertLayout layout,
                         std::span<const GpuBinding> gpus, std::int64_t max_in_flight_bytes)
    : policy_(policy), layout_(std::move(layout)), max_in_flight_bytes_(max_in_flight_bytes),
      gpus_(static_cast<std::int32_t>(gpus.size())), layers_(policy.layers),
      experts_(policy.experts) {
    if (gpus_ < 1 || gpus_ > kMaxGpus || gpus_ != policy.partitions) {
        throw std::invalid_argument("expert cache: one or two GPUs, one partition per GPU");
    }
    if (layout_.planes < 1 || layout_.planes > kMaxPlanes ||
        layout_.bank_base.size() != static_cast<std::size_t>(layers_ * layout_.planes)) {
        throw std::invalid_argument(
            "expert cache: 1..8 planes and one bank base per layer and plane");
    }
    if (layout_.variable()) {
        const auto cells = static_cast<std::size_t>(layers_ * layout_.planes);
        if (layout_.layer_plane_bytes.size() != cells ||
            layout_.layer_bank_stride.size() != cells ||
            layout_.layer_plane_offset.size() != cells) {
            throw std::invalid_argument("expert cache: variable layout needs every layer's planes");
        }
        for (std::int32_t l = 0; l < layers_; ++l) {
            std::int64_t bytes = 0;
            for (std::int32_t q = 0; q < layout_.planes; ++q) {
                const auto i = static_cast<std::size_t>(l * layout_.planes + q);
                if (layout_.layer_plane_bytes[i] <= 0 ||
                    layout_.layer_bank_stride[i] < layout_.layer_plane_bytes[i]) {
                    throw std::invalid_argument("expert cache: variable layout plane sizes");
                }
                bytes += layout_.layer_plane_bytes[i];
            }
            expert_bytes_ = std::max(expert_bytes_, bytes); // the in-flight budget's unit
        }
    } else {
        for (std::int32_t q = 0; q < layout_.planes; ++q) {
            if (layout_.plane_bytes[q] <= 0 || layout_.bank_stride[q] < layout_.plane_bytes[q] ||
                layout_.pool_stride[q] <= 0) {
                throw std::invalid_argument(
                    "expert cache: plane " + std::to_string(q) +
                    " needs bytes > 0, bank stride >= bytes, pool stride > 0");
            }
        }
        for (std::int32_t q = 0; q < layout_.planes; ++q) {
            expert_bytes_ += layout_.plane_bytes[q];
        }
    }
    if (max_in_flight_bytes_ < expert_bytes_) {
        throw std::invalid_argument("expert cache: the fill depth must hold at least one expert");
    }
    // v0.7 pinned fallback: a layer with any plane outside page-locked memory fills through the
    // staging ring.
    pageable_.assign(static_cast<std::size_t>(layers_), 0);
    bool any_pageable = false;
    for (std::int32_t l = 0; l < layers_; ++l) {
        for (std::int32_t q = 0; q < layout_.planes; ++q) {
            const std::byte* base =
                layout_.bank_base[static_cast<std::size_t>(l * layout_.planes + q)];
            if (base == nullptr) { throw std::invalid_argument("expert cache: null host bank"); }
            if (!pinned_host(base)) { pageable_[static_cast<std::size_t>(l)] = 1; }
        }
        any_pageable = any_pageable || pageable_[static_cast<std::size_t>(l)] != 0;
    }
    if (any_pageable) {
        std::int64_t staging = kDefaultStagingBytes;
        if (const char* v = std::getenv("NINFER_HOST_STAGING_MIB"); v != nullptr && *v != 0) {
            staging = std::max<std::int64_t>(0, std::atoll(v)) << 20;
        }
        staging_slots_ = static_cast<std::int32_t>(std::clamp<std::int64_t>(
            staging / expert_bytes_, 1, max_in_flight_bytes_ / expert_bytes_));
    }
    const std::size_t cells =
        static_cast<std::size_t>(layers_) * static_cast<std::size_t>(experts_);
    published_.assign(cells, -1);
    target_.assign(cells, -1);
    dirty_.assign(cells, 0);
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g    = gpu_[static_cast<std::size_t>(p)];
        g.binding = gpus[static_cast<std::size_t>(p)];
        if (g.binding.table == nullptr) { throw std::invalid_argument("expert cache: null table"); }
        for (std::int32_t q = 0; q < layout_.planes; ++q) {
            if (g.binding.pool_base[q] == nullptr) {
                throw std::invalid_argument("expert cache: null pool plane");
            }
        }
        const auto slots = static_cast<std::size_t>(pool_slots(policy, p));
        g.generation.assign(slots, 0);
        g.staged_at.assign(slots, -1);
        ScopedCurrentDevice scope(g.binding.device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&g.fill_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&g.fence, cudaEventDisableTiming));
        if (staging_slots_ > 0) {
            void* ring = nullptr;
            const auto bytes =
                static_cast<std::size_t>(staging_slots_) * static_cast<std::size_t>(expert_bytes_);
            const cudaError_t err = cudaHostAlloc(&ring, bytes, cudaHostAllocPortable);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "expert cache: pinned staging ring of " + std::to_string(bytes >> 20) +
                    " MiB for the pageable expert banks: cudaHostAlloc failed: " +
                    cudaGetErrorString(err) +
                    " (lower --host-pinned-max-gib or NINFER_HOST_STAGING_MIB)");
            }
            g.staging = static_cast<std::byte*>(ring);
        }
    }
    if (const char* v = std::getenv("NINFER_QWEN4_FILL_PUBLISH_LAG"); v != nullptr && *v != 0) {
        char* end        = nullptr;
        const long value = std::strtol(v, &end, 10);
        if (*end != 0 || (value != -1 && (value < 1 || value > 3))) {
            throw std::invalid_argument("NINFER_QWEN4_FILL_PUBLISH_LAG must be -1 or 1..3");
        }
        publish_lag_ = static_cast<std::int32_t>(value);
    }
    if (const char* v = std::getenv("NINFER_QWEN4_FILL_SM"); v != nullptr && *v != 0) {
        sm_fill_ctas_ = std::clamp(std::atoi(v), 0, 1024);
    }
    if (sm_fill_ctas_ > 0) {
        // Every bank must be readable in place by a kernel on every GPU; otherwise the copy
        // engine stays (a bank that is pinned but not mapped).
        bool in_place = true;
        for (std::int32_t p = 0; p < gpus_ && in_place; ++p) {
            ScopedCurrentDevice scope(gpu_[static_cast<std::size_t>(p)].binding.device);
            for (const std::byte* base : layout_.bank_base) {
                if (!device_reads_in_place(base)) {
                    in_place = false;
                    break;
                }
            }
        }
        if (!in_place) {
            std::fprintf(stderr, "expert cache: NINFER_QWEN4_FILL_SM ignored, the host banks are "
                                 "not mapped; fills use the copy engine\n");
            sm_fill_ctas_ = 0;
        } else {
            std::fprintf(stderr,
                         "expert cache: fills copied by a %d-block kernel (NINFER_QWEN4_FILL_SM)\n",
                         sm_fill_ctas_);
        }
    }
}

ExpertCache::~ExpertCache() {
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        if (cudaSetDevice(g.binding.device) != cudaSuccess) { continue; }
        // Fills may still be writing Program-owned memory.
        if (g.fill_stream != nullptr) { (void)cudaStreamSynchronize(g.fill_stream); }
        for (FillBatch& b : g.in_flight) { (void)cudaEventDestroy(b.done); }
        for (cudaEvent_t e : g.spare_events) { (void)cudaEventDestroy(e); }
        if (g.fence != nullptr) { (void)cudaEventDestroy(g.fence); }
        if (lend_events_[static_cast<std::size_t>(p)] != nullptr) {
            (void)cudaEventDestroy(lend_events_[static_cast<std::size_t>(p)]);
        }
        if (g.fill_stream != nullptr) { (void)cudaStreamDestroy(g.fill_stream); }
        if (g.staging != nullptr) { (void)cudaFreeHost(g.staging); }
    }
}

void ExpertCache::initialize(std::span<const std::vector<std::int32_t>> profile) {
    if (initialized_) { throw std::logic_error("expert cache: initialize called twice"); }
    if (!profile.empty() && profile.size() != static_cast<std::size_t>(layers_)) {
        throw std::invalid_argument("expert cache: profile needs one list per layer");
    }
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        ScopedCurrentDevice scope(g.binding.device);
        CUDA_CHECK(cudaMemset(g.binding.table, 0xFF,
                              static_cast<std::size_t>(table_entries(policy_.config())) *
                                  sizeof(std::int32_t)));
        if (g.binding.mirror != nullptr) {
            CUDA_CHECK(cudaMemset(g.binding.mirror, 0xFF,
                                  static_cast<std::size_t>(table_entries(policy_.config())) *
                                      sizeof(std::int32_t)));
        }
        // cudaMemset returns before the fill lands, and the windows publish on the caller's
        // non-blocking streams: finish the reset here, as the contract promises.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!profile.empty()) {
        std::vector<Insert> inserts;
        for (std::int32_t l = 0; l < layers_; ++l) {
            inserts.clear();
            policy_.prefill(l, profile[static_cast<std::size_t>(l)], inserts);
            for (const Insert& ins : inserts) { stage(ins, l); }
            counters_.inserts += static_cast<std::int64_t>(inserts.size());
        }
        submit_fills();
        flush();
    }
    initialized_ = true;
}

void ExpertCache::set_publish_lag(std::int32_t lag) {
    if (initialized_) { throw std::logic_error("expert cache: set_publish_lag after initialize"); }
    if (lag != -1 && (lag < 1 || lag > 3)) {
        throw std::invalid_argument("expert cache: publish lag must be -1 or 1..3");
    }
    publish_lag_ = lag;
}

void ExpertCache::check_streams(std::span<const cudaStream_t> streams) const {
    if (!initialized_) {
        throw std::logic_error("expert cache: initialize before the first window");
    }
    if (streams.size() != static_cast<std::size_t>(gpus_)) {
        throw std::invalid_argument("expert cache: one compute stream per GPU");
    }
}

void ExpertCache::begin_window(std::span<const cudaStream_t> streams) {
    check_streams(streams);
    if (fenced_ != launched_) {
        throw std::logic_error("expert cache: end_window missing for the previous window");
    }
    if (!deterministic()) {
        poll();
        publish_targets(streams);
        ++launched_;
        return;
    }
    // Deterministic publication: the batches due now (waited for) and their updates, then the
    // fill queue.
    retire_due();
    publish_targets(streams);
    // Fills issued from now on wait for a point after this publication on the compute stream, so
    // after anything that could still read a slot unpublished here.
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        ScopedCurrentDevice scope(g.binding.device);
        CUDA_CHECK(cudaEventRecord(g.fence, streams[static_cast<std::size_t>(p)]));
    }
    for (std::int32_t p = 0; p < gpus_; ++p) { pump(gpu_[static_cast<std::size_t>(p)]); }
    ++launched_;
}

void ExpertCache::retire_due() {
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g        = gpu_[static_cast<std::size_t>(p)];
        std::size_t n = 0;
        while (n < g.in_flight.size() && g.in_flight[n].due <= launched_) { ++n; }
        if (n == 0) { continue; }
        // One fill stream per GPU records the batches in order: the last one due covers the rest.
        ScopedCurrentDevice scope(g.binding.device);
        const cudaEvent_t last   = g.in_flight[n - 1].done;
        const cudaError_t status = cudaEventQuery(last);
        if (status == cudaErrorNotReady) {
            const auto t0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventSynchronize(last));
            ++counters_.publish_waits;
            counters_.publish_wait_us +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count();
        } else {
            CUDA_CHECK(status);
        }
        // Their experts are published by this begin_window, before the window it opens.
        for (std::size_t i = 0; i < n; ++i) { retire_front(g); }
    }
}

void ExpertCache::schedule(FillBatch& batch) const noexcept {
    // Inside window k (launched_ == k + 1, fenced_ == k): due at the next begin_window. Between
    // windows (fenced_ == launched_, in end_window or begin_window before ++launched_): one later.
    // Non-decreasing along the FIFO of batches, as retire_due needs.
    batch.due = launched_ + (publish_lag_ - 1) + (fenced_ < launched_ ? 0 : 1);
}

void ExpertCache::publish_targets(std::span<const cudaStream_t> streams) {
    std::array<std::vector<std::int32_t>, kMaxGpus> index;
    std::array<std::vector<std::int32_t>, kMaxGpus> value;
    std::vector<std::int32_t> mirror_index;
    std::vector<std::int32_t> mirror_value;
    // Residency mirrors of every partition's table: GPU 0's (the CPU path's publish), and at tp 2
    // optionally GPU 1's (its own selection of a miss-split share, T42); updated identically.
    bool any_mirror = false;
    for (std::int32_t p = 0; p < gpus_; ++p) {
        any_mirror = any_mirror || gpu_[static_cast<std::size_t>(p)].binding.mirror != nullptr;
    }
    std::int32_t* const mirror = any_mirror ? gpu_[0].binding.mirror : nullptr;
    for (const std::int32_t c : dirty_list_) {
        const auto ci        = static_cast<std::size_t>(c);
        const std::int32_t v = target_[ci];
        if (v == published_[ci]) { continue; }
        const auto p = static_cast<std::size_t>((c % experts_) % gpus_);
        index[p].push_back(c);
        value[p].push_back(v);
        if (any_mirror) {
            mirror_index.push_back(c);
            mirror_value.push_back(v);
        }
    }
    for (std::int32_t p = 0; p < gpus_; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        if (index[pi].empty()) { continue; }
        ScopedCurrentDevice scope(gpu_[pi].binding.device);
        launch_updates(gpu_[pi].binding.table, index[pi], value[pi], streams[pi],
                       counters_.table_launches);
    }
    for (std::int32_t p = 0; p < gpus_ && !mirror_index.empty(); ++p) {
        const auto pi = static_cast<std::size_t>(p);
        if (gpu_[pi].binding.mirror == nullptr) { continue; }
        ScopedCurrentDevice scope(gpu_[pi].binding.device);
        launch_updates(gpu_[pi].binding.mirror, mirror_index, mirror_value, streams[pi],
                       counters_.table_launches);
    }
    (void)mirror;
    // The host mirror follows only once every update (device tables and rank 0's residency mirror)
    // is launched: if a launch fails, nothing is committed and the next begin_window issues the
    // updates again (they are idempotent), so the host never believes a victim unpublished while a
    // device table still maps it.
    for (std::int32_t p = 0; p < gpus_; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        for (std::size_t i = 0; i < index[pi].size(); ++i) {
            const std::int32_t v                               = value[pi][i];
            published_[static_cast<std::size_t>(index[pi][i])] = v;
            ++(v >= 0 ? counters_.published : counters_.unpublished);
        }
    }
    for (const std::int32_t c : dirty_list_) { dirty_[static_cast<std::size_t>(c)] = 0; }
    dirty_list_.clear();
}

void ExpertCache::end_window(std::span<const cudaStream_t> streams) {
    check_streams(streams);
    if (launched_ != fenced_ + 1) {
        throw std::logic_error("expert cache: end_window without begin_window");
    }
    for (std::int32_t p = 0; p < gpus_; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        ScopedCurrentDevice scope(gpu_[pi].binding.device);
        CUDA_CHECK(cudaEventRecord(gpu_[pi].fence, streams[pi]));
    }
    ++fenced_;
    ++counters_.windows;
}

LayerView ExpertCache::observe(std::int32_t layer, std::span<const Routed> routed,
                               std::int32_t tokens) {
    LayerView view = classify(layer, routed);
    if (lent(layer) && !lent_keep_) {
        lent_decision(routed);
    } else if (frozen_) {
        frozen_decision(routed, view);
    } else {
        policy_.decide(layer, routed, tokens, decision_);
    }
    return finish_observe(layer, view);
}

LayerView ExpertCache::observe_prompt(std::int32_t layer, std::span<const Routed> routed,
                                      std::int32_t tokens, bool insert, std::int32_t target_cap) {
    LayerView view = classify(layer, routed);
    if (lent(layer) && !lent_keep_) {
        lent_decision(routed);
    } else if (frozen_) {
        frozen_decision(routed, view);
    } else if (!insert) {
        policy_.count_prompt(layer, routed, tokens, decision_);
    } else if (target_cap > 0) {
        policy_.decide_prompt(layer, routed, tokens, target_cap, decision_);
    } else {
        policy_.decide_prompt(layer, routed, tokens, decision_);
    }
    return finish_observe(layer, view);
}

void ExpertCache::apply_prompt(std::int32_t layer, std::span<const Routed> routed,
                               std::int32_t tokens, std::int32_t target_cap) {
    if (!initialized_) { throw std::logic_error("expert cache: initialize before apply_prompt"); }
    if (fenced_ != launched_) {
        throw std::logic_error("expert cache: apply_prompt inside a window");
    }
    if (frozen_ || lent(layer)) { return; }
    if (target_cap < 0) {
        policy_.count_prompt(layer, routed, tokens, decision_, /*frequencies=*/true);
        return;
    }
    policy_.decide_prompt(layer, routed, tokens,
                          target_cap > 0 ? target_cap : std::numeric_limits<std::int32_t>::max(),
                          decision_);
    for (const Insert& ins : decision_.inserts) { stage(ins, layer); }
    counters_.inserts += static_cast<std::int64_t>(decision_.inserts.size());
    queue_staged(true, /*issue_now=*/false);
}

// T59: a frozen cache inserts nothing; the policy's hits are the published residents.
void ExpertCache::frozen_decision(std::span<const Routed> routed, const LayerView& view) {
    decision_.accesses = 0;
    decision_.hits     = view.device_hits;
    decision_.distinct = 0;
    decision_.missing.clear();
    decision_.inserts.clear();
    for (const Routed& r : routed) {
        if (r.expert < 0 || r.expert >= experts_) { break; }
        decision_.accesses += r.count;
        ++decision_.distinct;
    }
    decision_.missing.assign(missing_.begin(), missing_.end());
    std::sort(decision_.missing.begin(), decision_.missing.end());
}

void ExpertCache::restore(std::span<const std::vector<std::int32_t>> profile) {
    if (!initialized_) { throw std::logic_error("expert cache: initialize before restore"); }
    if (fenced_ != launched_) { throw std::logic_error("expert cache: restore inside a window"); }
    if (lent_count_ > 0) { throw std::logic_error("expert cache: restore while layers are lent"); }
    if (profile.size() != static_cast<std::size_t>(layers_)) {
        throw std::invalid_argument("expert cache: restore needs one list per layer");
    }
    submit_fills();
    flush();
    std::vector<std::int32_t> evicted;
    std::vector<Insert> inserts;
    for (std::int32_t l = 0; l < layers_; ++l) {
        evicted.clear();
        policy_.clear_layer(l, evicted);
        for (std::int32_t e = 0; e < experts_; ++e) {
            const std::size_t c = cell(l, e);
            if (published_[c] >= 0 || target_[c] >= 0) { set_target(c, -1); }
        }
        inserts.clear();
        policy_.prefill(l, profile[static_cast<std::size_t>(l)], inserts);
        for (const Insert& ins : inserts) { stage(ins, l); }
        counters_.inserts += static_cast<std::int64_t>(inserts.size());
    }
    submit_fills();
    flush();
}

std::vector<std::int32_t> ExpertCache::resident_experts(std::int32_t layer) const {
    std::vector<std::int32_t> out;
    for (std::int32_t e = 0; e < experts_; ++e) {
        if (target_[cell(layer, e)] >= 0) { out.push_back(e); }
    }
    return out;
}

// A lent layer holds nothing and inserts nothing.
void ExpertCache::lent_decision(std::span<const Routed> routed) {
    decision_.accesses = 0;
    decision_.hits     = 0;
    decision_.distinct = static_cast<std::int32_t>(routed.size());
    decision_.missing.clear();
    decision_.inserts.clear();
    for (const Routed& r : routed) {
        decision_.accesses += r.count;
        decision_.missing.push_back(r.expert);
    }
}

std::span<const cudaEvent_t> ExpertCache::lend_layers(std::int32_t first, std::int32_t count,
                                                      bool keep_elastic) {
    if (!initialized_) { throw std::logic_error("expert cache: initialize before lending"); }
    if (lent_count_ > 0) { throw std::logic_error("expert cache: layers are already lent"); }
    if (first < 0 || count <= 0 || first + count > layers_) {
        throw std::out_of_range("expert cache: lent layers out of range");
    }
    keep_elastic = keep_elastic && policy_.config().elastic_columns > 0;
    std::vector<std::int32_t> evicted;
    std::vector<SlotEviction> base_evicted;
    for (std::int32_t l = first; l < first + count; ++l) {
        if (keep_elastic) {
            // T69: only the base slots go; a base slot's expert may still be published (or its
            // fill in flight), so its table entry is cleared like an eviction.
            base_evicted.clear();
            policy_.block_base(l, base_evicted);
            for (const SlotEviction& v : base_evicted) { set_target(cell(l, v.expert), -1); }
            continue;
        }
        evicted.clear();
        policy_.clear_layer(l, evicted);
        for (std::int32_t e = 0; e < experts_; ++e) {
            const std::size_t c = cell(l, e);
            if (published_[c] >= 0 || target_[c] >= 0) { set_target(c, -1); }
        }
    }
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        // A new generation drops every staged, queued or in-flight fill of a lent slot (the
        // layers' base slots and, unless kept, their elastic slots).
        for (std::int32_t s = policy_.base_offset(p, first);
             s < policy_.base_offset(p, first + count); ++s) {
            ++g.generation[static_cast<std::size_t>(s)];
        }
        for (std::int32_t c = 0; c < policy_.config().elastic_columns && !keep_elastic; ++c) {
            for (std::int32_t l = first; l < first + count; ++l) {
                ++g.generation[static_cast<std::size_t>(
                    policy_.pool_slot(p, l, policy_.base_slots(p, l) + c))];
            }
        }
        ScopedCurrentDevice scope(g.binding.device);
        cudaEvent_t& e = lend_events_[static_cast<std::size_t>(p)];
        if (e == nullptr) { CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
        CUDA_CHECK(cudaEventRecord(e, g.fill_stream));
    }
    lent_first_ = first;
    lent_count_ = count;
    lent_keep_  = keep_elastic;
    return {lend_events_.data(), static_cast<std::size_t>(gpus_)};
}

void ExpertCache::reclaim_layers(std::span<const std::vector<std::int32_t>> profile,
                                 std::span<const std::int32_t> rank) {
    reclaim_front(lent_count_, profile, rank);
}

void ExpertCache::reclaim_front(std::int32_t count,
                                std::span<const std::vector<std::int32_t>> profile,
                                std::span<const std::int32_t> rank) {
    if (lent_count_ == 0) { return; }
    if (fenced_ != launched_) {
        throw std::logic_error("expert cache: reclaim_layers inside a window");
    }
    if (count <= 0 || count > lent_count_) {
        throw std::out_of_range("expert cache: reclaimed layers out of the lent range");
    }
    if (!profile.empty() && profile.size() != static_cast<std::size_t>(count)) {
        throw std::invalid_argument("expert cache: one profile per lent layer");
    }
    const std::int32_t first = lent_first_;
    const bool kept          = lent_keep_;
    lent_first_ += count;
    lent_count_ -= count;
    if (lent_count_ == 0) {
        lent_first_ = 0;
        lent_keep_  = false;
    }
    if (kept) {
        for (std::int32_t l = first; l < first + count; ++l) { policy_.unblock_base(l); }
    }
    if (profile.empty()) { return; }
    std::vector<Insert> inserts;
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(profile.size()); ++i) {
        inserts.clear();
        if (kept) {
            policy_.top_up(first + i, profile[static_cast<std::size_t>(i)], inserts);
        } else {
            policy_.prefill(first + i, profile[static_cast<std::size_t>(i)], inserts);
        }
        for (const Insert& ins : inserts) { stage(ins, first + i); }
        counters_.inserts += static_cast<std::int64_t>(inserts.size());
    }
    if (rank.empty()) {
        queue_staged(true);
        return;
    }
    queue_staged(true, /*issue_now=*/false);
    prioritize_queue(rank);
    poll();
}

void ExpertCache::set_active_columns(std::int32_t columns, std::span<const cudaStream_t> streams) {
    check_streams(streams);
    if (fenced_ != launched_) {
        throw std::logic_error("expert cache: set_active_columns inside a window");
    }
    if (columns >= policy_.active_columns()) {
        std::vector<SlotEviction> none_evicted;
        std::vector<SlotMove> none_moved;
        policy_.set_active_columns(columns, none_evicted, none_moved);
        return;
    }
    // Every policy-resident expert's bytes are in its slot once the fills submitted so far have
    // landed (insertions staged mid-window are submitted first).
    queue_staged(true);
    flush();
    std::vector<SlotEviction> evicted;
    std::vector<SlotMove> moves;
    policy_.set_active_columns(columns, evicted, moves);
    for (const SlotEviction& v : evicted) {
        Gpu& g = gpu_[static_cast<std::size_t>(v.partition)];
        ++g.generation[static_cast<std::size_t>(v.slot)];
        set_target(cell(v.layer, v.expert), -1);
    }
    // Moves: device-to-device on the fill stream, after the last launched window (no window
    // reads the destination: it held an evicted expert, which no window of the idle device
    // reads any more once the updates below are applied before the next one).
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        ScopedCurrentDevice scope(g.binding.device);
        if (fenced_ > 0) { CUDA_CHECK(cudaStreamWaitEvent(g.fill_stream, g.fence, 0)); }
        for (const SlotMove& m : moves) {
            if (m.partition != p) { continue; }
            for (std::int32_t q = 0; q < layout_.planes; ++q) {
                CUDA_CHECK(cudaMemcpyAsync(
                    g.binding.pool_base[q] + layout_.pool_offset(m.layer, q, m.to),
                    g.binding.pool_base[q] + layout_.pool_offset(m.layer, q, m.from),
                    static_cast<std::size_t>(layout_.bytes(m.layer, q)), cudaMemcpyDeviceToDevice,
                    g.fill_stream));
            }
            ++g.generation[static_cast<std::size_t>(m.from)];
            ++g.generation[static_cast<std::size_t>(m.to)];
        }
        CUDA_CHECK(cudaStreamSynchronize(g.fill_stream));
    }
    for (const SlotMove& m : moves) { set_target(cell(m.layer, m.expert), m.to); }
    counters_.moved += static_cast<std::int64_t>(moves.size());
    counters_.shrink_evicted += static_cast<std::int64_t>(evicted.size());
    // The tables (and residency mirrors) name no released slot once these complete.
    publish_targets(streams);
    for (std::int32_t p = 0; p < gpus_; ++p) {
        ScopedCurrentDevice scope(gpu_[static_cast<std::size_t>(p)].binding.device);
        CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(p)]));
    }
}

LayerView ExpertCache::classify(std::int32_t layer, std::span<const Routed> routed) {
    if (!initialized_) { throw std::logic_error("expert cache: initialize before observe"); }
    if (layer < 0 || layer >= layers_) {
        throw std::out_of_range("expert cache: layer " + std::to_string(layer));
    }
    resident_.clear();
    missing_.clear();
    LayerView view;
    for (const Routed& r : routed) {
        if (r.expert < 0 || r.expert >= experts_) { break; } // the policy rejects the input
        const std::int32_t s = published_[cell(layer, r.expert)];
        if (s >= 0) {
            resident_.push_back(ResidentExpert{r.expert, r.expert % gpus_, s});
            view.device_hits += r.count;
        } else {
            missing_.push_back(r.expert);
        }
    }
    return view;
}

LayerView ExpertCache::finish_observe(std::int32_t layer, LayerView view) {
    for (const Insert& ins : decision_.inserts) { stage(ins, layer); }
    view.resident    = resident_;
    view.missing     = missing_;
    view.fills       = decision_.inserts;
    view.accesses    = decision_.accesses;
    view.policy_hits = decision_.hits;
    counters_.accesses += view.accesses;
    counters_.policy_hits += view.policy_hits;
    counters_.device_hits += view.device_hits;
    counters_.missing += static_cast<std::int64_t>(missing_.size());
    counters_.inserts += static_cast<std::int64_t>(decision_.inserts.size());
    return view;
}

void ExpertCache::set_target(std::size_t c, std::int32_t value) {
    target_[c] = value;
    if (dirty_[c] == 0) {
        dirty_[c] = 1;
        dirty_list_.push_back(static_cast<std::int32_t>(c));
    }
}

void ExpertCache::stage(const Insert& ins, std::int32_t layer) {
    Gpu& g = gpu_[static_cast<std::size_t>(ins.partition)];
    if (ins.victim >= 0) { set_target(cell(layer, ins.victim), -1); }
    const auto s = static_cast<std::size_t>(ins.slot);
    // A new owner cancels the publication of any earlier fill of this slot still in flight.
    const std::uint32_t generation = ++g.generation[s];
    const PendingFill fill{static_cast<std::int32_t>(cell(layer, ins.expert)), ins.slot,
                           generation};
    if (g.staged_at[s] >= 0) {
        // Reassigned twice before submission: copy only the last owner (copies inside one batch
        // are unordered, so two writes of one slot must never share a batch).
        g.staged[static_cast<std::size_t>(g.staged_at[s])] = fill;
        ++counters_.fills_dropped;
    } else {
        g.staged_at[s] = static_cast<std::int32_t>(g.staged.size());
        g.staged.push_back(fill);
    }
}

void ExpertCache::submit_fills() {
    if (fenced_ != launched_) {
        throw std::logic_error(
            "expert cache: submit_fills before end_window of the launched window");
    }
    queue_staged(true);
}

void ExpertCache::submit_fills_unfenced() {
    if (!initialized_) { throw std::logic_error("expert cache: initialize before submit"); }
    queue_staged(false);
}

void ExpertCache::queue_staged(bool fenced, bool issue_now) {
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        // Mid-window (demand) insertions go ahead of the queued backlog, in their own order: a
        // queued fill of the same slot is older, hence stale (dropped at issue), and one already
        // issued precedes them on the fill stream.
        std::size_t front = 0;
        for (PendingFill f : g.staged) {
            g.staged_at[static_cast<std::size_t>(f.slot)] = -1;
            f.fenced                                      = fenced;
            if (fenced) {
                g.queue.push_back(f);
            } else {
                g.queue.insert(g.queue.begin() + static_cast<std::ptrdiff_t>(front++), f);
            }
        }
        g.staged.clear();
        if (issue_now) { pump(g); }
    }
}

void ExpertCache::prioritize_queue(std::span<const std::int32_t> rank) {
    const auto key = [&](const PendingFill& f) {
        const auto c = static_cast<std::size_t>(f.cell);
        return c < rank.size() ? rank[c] : std::numeric_limits<std::int32_t>::max();
    };
    for (std::int32_t p = 0; p < gpus_; ++p) {
        Gpu& g = gpu_[static_cast<std::size_t>(p)];
        std::stable_sort(
            g.queue.begin(), g.queue.end(),
            [&](const PendingFill& a, const PendingFill& b) { return key(a) < key(b); });
        if (deterministic()) { pump(g); } // in place of the poll() that follows in timed mode
    }
}

// Issues queued fills while the in-flight bytes stay within the limit. A batch with a fill from
// submit_fills waits for the fence of the last launched window: the victims of any queued fill
// were unpublished no later than the begin_window following its submission, and submit_fills runs
// only after that window's fence, so every window that could still read a victim is covered (a
// later fence only waits longer). Fills from submit_fills_unfenced carry their caller's guarantee
// and wait for nothing. Insertions whose slot was reassigned since are dropped without a copy.
void ExpertCache::pump(Gpu& g) {
    if (!deterministic()) { retire(g, false); }
    std::vector<PendingFill> batch;
    std::int32_t staged = 0; // pageable fills in `batch`: each needs a free staging slot
    while (!g.queue.empty() &&
           g.in_flight_bytes + static_cast<std::int64_t>(batch.size() + 1) * expert_bytes_ <=
               max_in_flight_bytes_) {
        const PendingFill f = g.queue.front();
        if (g.generation[static_cast<std::size_t>(f.slot)] != f.generation) {
            g.queue.pop_front();
            ++counters_.fills_dropped;
            continue;
        }
        if (pageable_[static_cast<std::size_t>(f.cell / experts_)] != 0) {
            if (g.staging_in_use + staged >= staging_slots_) { break; } // stays queued
            ++staged;
        }
        g.queue.pop_front();
        batch.push_back(f);
    }
    issue(g, batch);
}

void ExpertCache::issue(Gpu& g, std::span<const PendingFill> fills) {
    if (fills.empty()) { return; }
    ScopedCurrentDevice scope(g.binding.device);
    const bool fenced =
        std::any_of(fills.begin(), fills.end(), [](const PendingFill& f) { return f.fenced; });
    if (fenced && fenced_ > 0) { CUDA_CHECK(cudaStreamWaitEvent(g.fill_stream, g.fence, 0)); }
    const std::int32_t planes = layout_.planes;
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder    = cudaMemcpySrcAccessOrderStream;
    attr.flags             = cudaMemcpyFlagPreferOverlapWithCompute;
    std::size_t attr_index = 0;
    for (std::size_t first = 0; first < fills.size(); first += kFillChunkExperts) {
        const std::size_t n = std::min<std::size_t>(kFillChunkExperts, fills.size() - first);
        copy_dst_.clear();
        copy_src_.clear();
        copy_bytes_.clear();
        FillBatch batch;
        for (std::size_t i = first; i < first + n; ++i) {
            const PendingFill& f = fills[i];
            const std::int32_t l = f.cell / experts_;
            const std::int32_t e = f.cell % experts_;
            // A pageable bank's expert first goes into a free staging slot (pump checked there is
            // one); the slot is held until the batch's event completes (retire).
            std::byte* stage          = nullptr;
            std::int64_t staged_bytes = 0; // planes packed back to back in the staging slot
            if (pageable_[static_cast<std::size_t>(l)] != 0) {
                const std::int64_t at = g.staging_head++ % staging_slots_;
                stage                 = g.staging + at * expert_bytes_;
                ++g.staging_in_use;
                ++batch.staged;
                ++counters_.staged_fills;
            }
            for (std::int32_t q = 0; q < planes; ++q) {
                const std::byte* src     = layout_.bank(l, q, e);
                const std::int64_t bytes = layout_.bytes(l, q);
                if (stage != nullptr) {
                    std::memcpy(stage + staged_bytes, src, static_cast<std::size_t>(bytes));
                    src = stage + staged_bytes;
                    staged_bytes += bytes;
                }
                copy_dst_.push_back(g.binding.pool_base[q] + layout_.pool_offset(l, q, f.slot));
                copy_src_.push_back(src);
                copy_bytes_.push_back(static_cast<std::size_t>(bytes));
                counters_.fill_bytes += bytes;
            }
            batch.fills.push_back(f);
        }
        bool by_kernel =
            sm_fill_ctas_ > 0 && copy_dst_.size() <= static_cast<std::size_t>(kMaxSmFillCopies);
        if (by_kernel) {
            SmFillCopies c;
            c.count = static_cast<std::int32_t>(copy_dst_.size());
            for (std::size_t i = 0; i < copy_dst_.size() && by_kernel; ++i) {
                const auto d = reinterpret_cast<std::uintptr_t>(copy_dst_[i]);
                const auto s = reinterpret_cast<std::uintptr_t>(copy_src_[i]);
                by_kernel    = d % 16 == 0 && s % 16 == 0 && copy_bytes_[i] % 16 == 0;
                c.dst[i]     = static_cast<uint4*>(copy_dst_[i]);
                c.src[i]     = static_cast<const uint4*>(copy_src_[i]);
                c.units[i]   = static_cast<std::int64_t>(copy_bytes_[i] / 16);
            }
            if (by_kernel) {
                sm_fill_kernel<<<static_cast<unsigned>(sm_fill_ctas_), kSmFillThreads, 0,
                                 g.fill_stream>>>(c);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (!by_kernel) {
            CUDA_CHECK(cudaMemcpyBatchAsync(copy_dst_.data(), copy_src_.data(), copy_bytes_.data(),
                                            copy_dst_.size(), &attr, &attr_index, 1,
                                            g.fill_stream));
        }
        if (!g.spare_events.empty()) {
            batch.done = g.spare_events.back();
            g.spare_events.pop_back();
        } else {
            CUDA_CHECK(cudaEventCreateWithFlags(&batch.done, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(batch.done, g.fill_stream));
        counters_.fills_issued += static_cast<std::int64_t>(n);
        g.in_flight_bytes += static_cast<std::int64_t>(n) * expert_bytes_;
        if (deterministic()) { schedule(batch); }
        g.in_flight.push_back(std::move(batch));
    }
}

void ExpertCache::retire(Gpu& g, bool wait) {
    while (!g.in_flight.empty()) {
        FillBatch& b = g.in_flight.front();
        if (wait) {
            ScopedCurrentDevice scope(g.binding.device);
            CUDA_CHECK(cudaEventSynchronize(b.done));
        } else {
            const cudaError_t status = cudaEventQuery(b.done);
            if (status == cudaErrorNotReady) { break; }
            CUDA_CHECK(status);
        }
        retire_front(g);
    }
}

void ExpertCache::retire_front(Gpu& g) {
    FillBatch& b = g.in_flight.front();
    for (const PendingFill& f : b.fills) {
        if (g.generation[static_cast<std::size_t>(f.slot)] == f.generation) {
            set_target(static_cast<std::size_t>(f.cell), f.slot);
        }
    }
    g.in_flight_bytes -= static_cast<std::int64_t>(b.fills.size()) * expert_bytes_;
    g.staging_in_use -= b.staged;
    g.spare_events.push_back(b.done);
    g.in_flight.pop_front();
}

void ExpertCache::poll() {
    if (deterministic()) { return; } // every state change happens at a fixed point of the calls
    for (std::int32_t p = 0; p < gpus_; ++p) { pump(gpu_[static_cast<std::size_t>(p)]); }
}

void ExpertCache::flush() {
    if (!deterministic()) {
        for (std::int32_t p = 0; p < gpus_; ++p) {
            Gpu& g = gpu_[static_cast<std::size_t>(p)];
            while (!g.queue.empty() || !g.in_flight.empty()) {
                retire(g, true);
                pump(g);
            }
        }
        return;
    }
    // Deterministic publication: round by round over the GPUs, so each one's next batches copy
    // while another's are waited for (one GPU after the other doubled a prompt chunk's refill time
    // on the link, and with the lag this flush runs in every prompt chunk).
    std::array<bool, kMaxGpus> done{};
    for (bool busy = true; busy;) {
        busy = false;
        for (std::int32_t p = 0; p < gpus_; ++p) {
            Gpu& g = gpu_[static_cast<std::size_t>(p)];
            if (done[static_cast<std::size_t>(p)]) { continue; }
            if (g.queue.empty() && g.in_flight.empty()) {
                done[static_cast<std::size_t>(p)] = true;
                continue;
            }
            retire(g, true);
            pump(g);
            busy = true;
        }
    }
}

std::int64_t ExpertCache::fills_pending() const noexcept {
    std::int64_t n = 0;
    for (std::int32_t p = 0; p < gpus_; ++p) {
        const Gpu& g = gpu_[static_cast<std::size_t>(p)];
        n += static_cast<std::int64_t>(g.staged.size() + g.queue.size());
        for (const FillBatch& b : g.in_flight) { n += static_cast<std::int64_t>(b.fills.size()); }
    }
    return n;
}

void expert_cache_map_slots(const std::int32_t* ids, std::int32_t count, std::int32_t layer,
                            std::int32_t experts, const std::int32_t* table, std::int32_t* slot_ids,
                            cudaStream_t stream) {
    if (count <= 0) { return; }
    map_slots_kernel<<<static_cast<unsigned>((count + 127) / 128), 128, 0, stream>>>(
        ids, count, table + static_cast<std::ptrdiff_t>(layer) * experts, experts, slot_ids);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::expert_cache
