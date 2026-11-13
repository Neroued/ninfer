// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ninfer/ops/peer_mailbox.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04),
// which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

// CAPTURED MAILBOX SELECTION. Returns the mailbox attached to `events` when every predicate holds:
//
//   * the mailbox serves this ExecutionContext's device pair;
//   * the payload is whole 16-byte vectors within one slot (ordinary or wide), at 16-byte aligned
//     addresses (the exchange kernel's vector contract);
//   * BOTH ranks' streams are capturing, into the SAME capture. The exchange has no host reset
//     point and no event ordering: its two kernels must be nodes of one graph that every launch
//     runs on both devices. A rank whose stream is not in that capture would run its half
//     eagerly, spin without a partner and report a hang.
//
// Every predicate the mailbox fails is a collective that runs the staged path, unchanged. Leaves
// the current device changed; the caller holds a CurrentDeviceGuard.
PeerMailbox* captured_mailbox(const std::array<Tensor, 2>& buffer, std::size_t bytes,
                              const ExecutionContext& ec, const PeerEvents& events) {
    for (int rank = 0; rank < 2; ++rank) {
        if ((reinterpret_cast<std::uintptr_t>(buffer[rank].data) % 16) != 0) { return nullptr; }
    }
    return detail::captured_mailbox_for(bytes, ec, events);
}

// EAGER HOST EXCHANGE SELECTION (T56). Returns the HostExchange attached to `events` when it serves
// this pair and neither rank's stream is capturing; otherwise the driver-staged pull stays.
HostExchange* eager_host_exchange(const ExecutionContext& ec, const PeerEvents& events) {
    HostExchange* exchange = events.host_exchange();
    if (exchange == nullptr || !exchange->serves(ec)) { return nullptr; }
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusNone) { return nullptr; }
    }
    return exchange;
}

// EAGER PEER LANES SELECTION (v07/p2p-overlap). Returns the PeerLanes attached to `events` when
// they serve this pair and neither rank's stream is capturing; otherwise null, and the split-phase
// sums keep the HostExchange lanes exactly as before (no PeerLanes are attached without peer
// access unless NINFER_TP_PEER_OVERLAP=force).
PeerLanes* eager_peer_lanes(const ExecutionContext& ec, const PeerEvents& events) {
    PeerLanes* lanes = events.peer_lanes();
    if (lanes == nullptr || !lanes->serves(ec)) { return nullptr; }
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusNone) { return nullptr; }
    }
    return lanes;
}

} // namespace

PeerMailbox* detail::captured_mailbox_for(std::size_t bytes, const ExecutionContext& ec,
                                          const PeerEvents& events) {
    PeerMailbox* mailbox = events.mailbox();
    if (mailbox == nullptr || events.staged_only() || !mailbox->serves(ec)) { return nullptr; }
    if ((bytes % 16) != 0 || bytes > mailbox->max_bytes()) { return nullptr; }
    unsigned long long capture[2] = {0, 0};
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamGetCaptureInfo(local.stream, &status, &capture[rank]));
        if (status != cudaStreamCaptureStatusActive) { return nullptr; }
    }
    return capture[0] == capture[1] ? mailbox : nullptr;
}

namespace {

// The staged path's local combine of allreduce_sum_f32: one IEEE FP32 addition per element (the
// mailbox kernel's arithmetic, so the transports agree bit for bit) and the optional single BF16
// rounding of the sum.
__global__ void allreduce_f32_combine_kernel(float* __restrict__ partial,
                                             const float* __restrict__ peer,
                                             __nv_bfloat16* __restrict__ bf16_out,
                                             std::size_t count, unsigned int dither_key,
                                             unsigned int dither_bit) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        float sum = __fadd_rn(partial[i], peer[i]);
        if (dither_key != 0U) {
            // T128 diagnostic (NINFER_TP_AR_DITHER): flip one low mantissa bit of half the sums.
            unsigned int h = static_cast<unsigned int>(i) ^ dither_key;
            h ^= h >> 16;
            h *= 0x7feb352dU;
            h ^= h >> 15;
            h *= 0x846ca68bU;
            h ^= h >> 16;
            if ((h & 1U) != 0U) {
                sum = __uint_as_float(__float_as_uint(sum) ^ (1U << dither_bit));
            }
        }
        partial[i] = sum;
        if (bf16_out != nullptr) { bf16_out[i] = __float2bfloat16_rn(sum); }
    }
}

void allreduce_f32_combine(float* partial, const float* peer, __nv_bfloat16* bf16_out,
                           std::size_t count, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const std::size_t want = (count + kThreads - 1) / kThreads;
    const int blocks       = static_cast<int>(want < 1024 ? (want < 1 ? 1 : want) : 1024);
    // T128: NINFER_TP_AR_DITHER=<seed> (diagnostic), keyed per call so the layers differ.
    static const unsigned int seed = [] {
        const char* v = std::getenv("NINFER_TP_AR_DITHER");
        return v != nullptr ? static_cast<unsigned int>(std::strtoul(v, nullptr, 10)) : 0U;
    }();
    static const unsigned int bit = [] {
        const char* v = std::getenv("NINFER_TP_AR_DITHER_BIT");
        return v != nullptr ? static_cast<unsigned int>(std::strtoul(v, nullptr, 10)) & 15U : 0U;
    }();
    // The callers combine rank 0 then rank 1 of each exchange: both ranks get the same key (the
    // ranks must keep identical bits).
    static unsigned int calls = 0;
    const unsigned int call   = (calls++) >> 1;
    const unsigned int key = seed == 0U ? 0U : ((seed * 0x9E3779B9U) ^ (call * 0x85EBCA6BU)) | 1U;
    allreduce_f32_combine_kernel<<<blocks, kThreads, 0, stream>>>(partial, peer, bf16_out, count,
                                                                  key, bit);
    CUDA_CHECK(cudaGetLastError());
}

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    int forward = 0;
    int reverse = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]));
    // Asymmetric support is not a usable transport for a symmetric collective: fall back to the
    // staged path rather than enabling one direction only.
    if (forward == 0 || reverse == 0) { return false; }

    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(pair[rank]);
        const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // Already enabled by an earlier call; clear the sticky runtime error so the next
            // cudaGetLastError() in an unrelated launcher does not observe it.
            cudaGetLastError();
            continue;
        }
        // #41: the driver may advertise peer access and still fail to map the peer's memory
        // (a VM whose GPUs expose a 256 MiB BAR1: cudaErrorMapBufferObjectFailed). Undo the
        // direction already enabled and use the staged path instead of aborting startup.
        if (status == cudaErrorMapBufferObjectFailed || status == cudaErrorMemoryAllocation ||
            status == cudaErrorTooManyPeers) {
            cudaGetLastError();
            for (int done = 0; done < rank; ++done) {
                CurrentDeviceGuard::set(pair[done]);
                (void)cudaDeviceDisablePeerAccess(pair[1 - done]);
                cudaGetLastError();
            }
            std::fprintf(stderr,
                         "tensor parallel: peer access between devices %d and %d failed (%s); "
                         "using the host-staged path (p2p off). Under VFIO, grow the GPUs' BAR1 "
                         "on the host (see docs/serving.md)\n",
                         pair[0], pair[1], cudaGetErrorString(status));
            return false;
        }
        CUDA_CHECK(status);
    }
    return true;
}

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    cudaEvent_t created[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 4; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    pull_done_    = {created[2], created[3]};
}

PeerEvents::~PeerEvents() {
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_), mailbox_(other.mailbox_),
      host_exchange_(other.host_exchange_), peer_lanes_(other.peer_lanes_),
      staged_scopes_(other.staged_scopes_) {
    other.inputs_ready_  = {nullptr, nullptr};
    other.pull_done_     = {nullptr, nullptr};
    other.mailbox_       = nullptr;
    other.host_exchange_ = nullptr;
    other.peer_lanes_    = nullptr;
    other.staged_scopes_ = 0;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    std::swap(mailbox_, other.mailbox_);
    std::swap(host_exchange_, other.host_exchange_);
    std::swap(peer_lanes_, other.peer_lanes_);
    std::swap(staged_scopes_, other.staged_scopes_);
    return *this;
}

void detail::require_allreduce_sum_arguments(const std::array<Tensor, 2>& buffer,
                                             const std::array<Tensor, 2>& staging,
                                             const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");
#ifndef NDEBUG
    const std::size_t bytes = buffer[0].bytes();
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif
}

namespace {
// On a pair without peer access the T62 exchange's host-to-device leg is read by a kernel, off the
// copy engine (default since v0.11.1; NINFER_TP_XIN_KERNEL=0 keeps the copy; with peer access the
// copy stays as it is). A GPU's host-to-device copies run
// one after another from a single queue (an 8 MiB copy issued during a 512 MiB upload on another
// stream finishes with the upload: 16.6 ms instead of 0.31 ms, whatever the stream priority or the
// upload's piece size), so in a streamed prefill each exchange copy waits behind the next layer's
// expert weights (93-650 MiB per layer and card). A kernel reading the peer's pinned buffer over
// PCIe does not queue there: 24 GB/s from 4 blocks, the same 8 MiB in 0.6 ms during the upload.
bool xin_kernel_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TP_XIN_KERNEL");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

// The same rule as enable_peer_access(): P2P only when both directions report support.
bool pair_can_access_peer(int a, int b) {
    int forward = 0, reverse = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, a, b));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, b, a));
    return forward != 0 && reverse != 0;
}

// Blocks of 512 threads. Alone 4 blocks already read at the link's 24 GB/s, but beside the
// prefill's kernels more blocks win: 100k-token prefill (Qwen3.8 Flash, 8192-column groups, RTX
// 5080 + 5070 Ti, three interleaved runs) 2 / 4 / 8 / 16 / 32 blocks: 3.90 / 4.01 / 4.30 / 4.35
// / 4.33k tok/s; copy engine 3.88k. The large kernels of the prefill (dense q8, attention) keep
// their times with 16 blocks beside them (nsys).
constexpr int kXinBlocks = 16;

// Ordering and visibility come from the stream's wait on the peer's device-to-host event;
// ld.global.cv is defensive only (the same host lines are rewritten by the peer every call).
__global__ void xin_read_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src,
                                std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        dst[i] = __ldcv(src + i);
    }
}

// The T62 receive copy: by kernel when the exchange reads by kernel and the piece is 16-byte
// aligned (odd rows x payload sizes keep the copy).
void receive_to_device(void* dst, const void* src, std::size_t bytes, cudaStream_t stream,
                       bool by_kernel) {
    if (by_kernel && reinterpret_cast<std::uintptr_t>(dst) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(src) % 16 == 0 && bytes % 16 == 0) {
        xin_read_kernel<<<kXinBlocks, 512, 0, stream>>>(static_cast<uint4*>(dst),
                                                        static_cast<const uint4*>(src), bytes / 16);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
}

// With the kernel, the receive stream gets the device's greatest priority: its blocks are
// dispatched ahead of the waiting blocks of the compute kernels (8 blocks: 4.29 / 4.34k tok/s with
// it, 4.21-4.24 / 4.27-4.28k without).
cudaStream_t create_receive_stream(bool by_kernel) {
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t s = nullptr;
    CUDA_CHECK(
        cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, by_kernel ? greatest : least));
    return s;
}
} // namespace

HostExchange::HostExchange(const ExecutionContext& ec, std::size_t chunk_bytes)
    : chunk_bytes_(chunk_bytes) {
    require_two_devices(ec, "HostExchange: requires an ExecutionContext with two distinct devices");
    require(chunk_bytes_ > 0 && chunk_bytes_ % 256 == 0,
            "HostExchange: chunk must be a multiple of 256 bytes");
    const CurrentDeviceGuard guard;
    receive_by_kernel_ =
        xin_kernel_enabled() && !pair_can_access_peer(ec.dev[0]->device, ec.dev[1]->device);
    for (int rank = 0; rank < 2; ++rank) {
        devices_[static_cast<std::size_t>(rank)] = ec.dev[rank]->device;
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&side_[static_cast<std::size_t>(rank)],
                                             cudaStreamNonBlocking));
        const auto r = static_cast<std::size_t>(rank);
        receive_[r]  = create_receive_stream(receive_by_kernel_);
        CUDA_CHECK(cudaEventCreateWithFlags(&begun_[r], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&received_[r], cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&lane_send_[r], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&lane_receive_[r], cudaStreamNonBlocking));
        for (Lane& lane : lanes_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.begun[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.received[r], cudaEventDisableTiming));
        }
    }
}

HostExchange::~HostExchange() {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) { previous = -1; }
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] < 0 || cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        if (side_[rank] != nullptr) {
            (void)cudaStreamSynchronize(side_[rank]);
            (void)cudaStreamDestroy(side_[rank]);
        }
        if (receive_[rank] != nullptr) {
            (void)cudaStreamSynchronize(receive_[rank]);
            (void)cudaStreamDestroy(receive_[rank]);
        }
        for (cudaEvent_t e : chunk_done_[rank]) { (void)cudaEventDestroy(e); }
        for (cudaEvent_t e : group_ready_[rank]) { (void)cudaEventDestroy(e); }
        for (cudaEvent_t e : pipe_chunk_[rank]) { (void)cudaEventDestroy(e); }
        for (cudaEvent_t e : {begun_[rank], received_[rank]}) {
            if (e != nullptr) { (void)cudaEventDestroy(e); }
        }
        if (stats_[rank] != nullptr) { (void)cudaFree(stats_[rank]); }
        for (cudaStream_t s : {lane_send_[rank], lane_receive_[rank]}) {
            if (s != nullptr) {
                (void)cudaStreamSynchronize(s);
                (void)cudaStreamDestroy(s);
            }
        }
        for (Lane& lane : lanes_) {
            for (cudaEvent_t e : lane.chunk_done[rank]) { (void)cudaEventDestroy(e); }
            for (cudaEvent_t e : {lane.begun[rank], lane.received[rank]}) {
                if (e != nullptr) { (void)cudaEventDestroy(e); }
            }
        }
    }
    // The peers' host-to-device reads ran on the ranks' own streams: drain both devices before
    // the buffers go.
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] >= 0 && cudaSetDevice(devices_[rank]) == cudaSuccess) {
            (void)cudaDeviceSynchronize();
        }
    }
    for (std::byte* h : host_) {
        if (h != nullptr) { (void)cudaFreeHost(h); }
    }
    for (Lane& lane : lanes_) {
        for (std::byte* h : lane.host) {
            if (h != nullptr) { (void)cudaFreeHost(h); }
        }
    }
    if (previous >= 0) { (void)cudaSetDevice(previous); }
}

bool HostExchange::serves(const ExecutionContext& ec) const noexcept {
    return ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value() &&
           ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

void HostExchange::reserve(std::size_t bytes, std::size_t chunks, const ExecutionContext& ec) {
    if (bytes > capacity_) {
        // Earlier exchanges may still read or write the old buffers: drain both devices.
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (std::byte*& h : host_) {
            if (h != nullptr) { CUDA_CHECK(cudaFreeHost(h)); }
            h = nullptr;
        }
        capacity_ = 0;
        for (std::byte*& h : host_) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
            h = static_cast<std::byte*>(p);
        }
        capacity_ = bytes;
    }
    for (int rank = 0; rank < 2; ++rank) {
        auto& events = chunk_done_[static_cast<std::size_t>(rank)];
        if (events.size() >= chunks) { continue; }
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        while (events.size() < chunks) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            events.push_back(e);
        }
    }
}

void HostExchange::pull(const std::array<const void*, 2>& source,
                        const std::array<void*, 2>& destination, std::size_t bytes,
                        const ExecutionContext& ec, const PeerEvents& events) {
    require(serves(ec), "HostExchange: serves another device pair");
    const std::size_t chunks = (bytes + chunk_bytes_ - 1) / chunk_bytes_;
    reserve(bytes, chunks, ec);
    // Leg 1, both ranks: operand -> own host buffer, on the side stream, one event per chunk.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamWaitEvent(side_[r], events.inputs_ready(rank), 0));
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * chunk_bytes_;
            const std::size_t n      = std::min(chunk_bytes_, bytes - offset);
            CUDA_CHECK(cudaMemcpyAsync(host_[r] + offset,
                                       static_cast<const std::byte*>(source[r]) + offset, n,
                                       cudaMemcpyDeviceToHost, side_[r]));
            CUDA_CHECK(cudaEventRecord(chunk_done_[r][c], side_[r]));
        }
    }
    // Leg 2, both ranks: the peer's host buffer -> own staging, on the rank's stream, chunk by
    // chunk behind the peer's events (the two legs of a pair overlap on two different links).
    for (int rank = 0; rank < 2; ++rank) {
        const auto r    = static_cast<std::size_t>(rank);
        const auto peer = static_cast<std::size_t>(1 - rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const cudaStream_t stream = ec.dev[rank]->stream;
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * chunk_bytes_;
            const std::size_t n      = std::min(chunk_bytes_, bytes - offset);
            CUDA_CHECK(cudaStreamWaitEvent(stream, chunk_done_[peer][c], 0));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(destination[r]) + offset,
                                       host_[peer] + offset, n, cudaMemcpyHostToDevice, stream));
        }
    }
}

HostExchange* eager_exchange(const ExecutionContext& ec, const PeerEvents& events) {
    return eager_host_exchange(ec, events);
}

void HostExchange::copy(int from, const void* source, void* destination, std::size_t bytes,
                        const ExecutionContext& ec, const PeerEvents& events) {
    require(serves(ec), "HostExchange: serves another device pair");
    require(from == 0 || from == 1, "HostExchange: copy from rank 0 or 1");
    if (bytes == 0) { return; }
    const std::size_t chunks = (bytes + chunk_bytes_ - 1) / chunk_bytes_;
    reserve(bytes, chunks, ec);
    const auto f = static_cast<std::size_t>(from);
    const int to = 1 - from;
    CurrentDeviceGuard::set(ec.dev[from]->device);
    CUDA_CHECK(cudaStreamWaitEvent(side_[f], events.inputs_ready(from), 0));
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t offset = c * chunk_bytes_;
        const std::size_t n      = std::min(chunk_bytes_, bytes - offset);
        CUDA_CHECK(cudaMemcpyAsync(host_[f] + offset,
                                   static_cast<const std::byte*>(source) + offset, n,
                                   cudaMemcpyDeviceToHost, side_[f]));
        CUDA_CHECK(cudaEventRecord(chunk_done_[f][c], side_[f]));
    }
    CurrentDeviceGuard::set(ec.dev[to]->device);
    const cudaStream_t stream = ec.dev[to]->stream;
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t offset = c * chunk_bytes_;
        const std::size_t n      = std::min(chunk_bytes_, bytes - offset);
        CUDA_CHECK(cudaStreamWaitEvent(stream, chunk_done_[f][c], 0));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(destination) + offset, host_[f] + offset,
                                   n, cudaMemcpyHostToDevice, stream));
    }
}

// ---- T62: the prefill MoE exchange by column groups, optional narrow payload ------------------

namespace {

constexpr int kPayloadF16 = 1; // HostExchange::Payload::FP16; anything else narrows to BF16

// T128b: the FP16 payload carries v * 2^-k (exact), saturated to the FP16 range (NaN stays NaN);
// the receiver multiplies by 2^k (exact). down = 2^-k, up = 2^k; the BF16 payload ignores them.
__device__ __forceinline__ std::uint16_t narrow_bits(float v, int payload, float down) {
    if (payload != kPayloadF16) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
    float s = v * down;
    if (fabsf(s) > 65504.0F) { s = copysignf(65504.0F, s); }
    return __half_as_ushort(__float2half_rn(s));
}

__device__ __forceinline__ float widen_bits(std::uint16_t b, int payload, float up) {
    return payload == kPayloadF16 ? __half2float(__ushort_as_half(b)) * up
                                  : __bfloat162float(__ushort_as_bfloat16(b));
}

// NINFER_TP_AR_SHIFT = k (0..14, default 2).
int exchange_shift() {
    static const int k = [] {
        const char* v = std::getenv("NINFER_TP_AR_SHIFT");
        const int s   = v != nullptr ? std::atoi(v) : 2;
        return s < 0 ? 0 : (s > 14 ? 14 : s);
    }();
    return k;
}

// dst[i] = narrow(src[i]), RNE.
__global__ void exchange_narrow_kernel(const float* __restrict__ src,
                                       std::uint16_t* __restrict__ dst, std::size_t count,
                                       int payload, float down) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        dst[i] = narrow_bits(src[i], payload, down);
    }
}

// S = fl32(widen(narrow(own)) + widen(peer)); own <- S; out <- RNE_bf16(S). The own operand is
// rounded exactly as the copy the peer received, so both ranks add the same two values.
__global__ void exchange_narrow_combine_kernel(float* __restrict__ partial,
                                               const std::uint16_t* __restrict__ peer,
                                               __nv_bfloat16* __restrict__ bf16_out,
                                               std::size_t count, int payload, float down,
                                               float up) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        const float own = widen_bits(narrow_bits(partial[i], payload, down), payload, up);
        const float sum = __fadd_rn(own, widen_bits(peer[i], payload, up));
        partial[i]      = sum;
        if (bf16_out != nullptr) { bf16_out[i] = __float2bfloat16_rn(sum); }
    }
}

// ---- INT8 payload (allreduce.h, Payload::INT8, NINFER_TP_AR_PAYLOAD=int8) ----
// Per column `rows` int8 codes then rows / 32 binary16 scales. Both ranks dequantize the same bytes
// to the same floats and add the same two values, so their sums are bitwise equal. q * s is exact
// in FP32 (8-bit code times an 11-bit significand), so contraction could not change it; __fmul_rn /
// __fadd_rn state the intent.
constexpr int kInt8Block = 32;

__host__ __device__ constexpr std::size_t int8_column_bytes(std::int64_t rows) {
    return static_cast<std::size_t>(rows + rows / 16);
}

// One warp per 32-row block of `columns` columns starting at src (column-major, `rows` per column).
__global__ void exchange_int8_narrow_kernel(const float* __restrict__ src,
                                            std::byte* __restrict__ dst, std::int64_t rows,
                                            std::int64_t columns) {
    const std::int64_t per_column = rows / kInt8Block;
    const std::int64_t blocks     = per_column * columns;
    const int lane                = static_cast<int>(threadIdx.x & 31U);
    const std::int64_t warps      = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / 32);
    for (std::int64_t w =
             static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
         w < blocks; w += warps) {
        const std::int64_t column = w / per_column, block = w % per_column;
        const float v  = src[column * rows + block * kInt8Block + lane];
        const bool bad = __any_sync(0xFFFFFFFFU, !isfinite(v));
        float amax     = fabsf(v);
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, o));
        }
        __half scale = __float2half(0.0F);
        int q        = 0;
        if (bad) {
            scale =
                __ushort_as_half(static_cast<unsigned short>(0x7E00)); // NaN: the block becomes NaN
        } else if (amax > 0.0F) {
            const float s   = fminf(fmaxf(__fdiv_rn(amax, 127.0F), 6.103515625e-05F), 65504.0F);
            scale           = __float2half_ru(s);
            const float inv = __fdiv_rn(1.0F, __half2float(scale));
            q = static_cast<int>(fminf(fmaxf(rintf(__fmul_rn(v, inv)), -127.0F), 127.0F));
        }
        std::byte* col = dst + column * static_cast<std::int64_t>(int8_column_bytes(rows));
        reinterpret_cast<std::int8_t*>(col)[block * kInt8Block + lane] =
            static_cast<std::int8_t>(q);
        if (lane == 0) { reinterpret_cast<__half*>(col + rows)[block] = scale; }
    }
}

__device__ __forceinline__ float int8_value(const std::byte* base, std::int64_t rows,
                                            std::size_t i) {
    const auto column    = static_cast<std::int64_t>(i) / rows;
    const auto row       = static_cast<std::int64_t>(i) % rows;
    const std::byte* col = base + column * static_cast<std::int64_t>(int8_column_bytes(rows));
    const float s = __half2float(reinterpret_cast<const __half*>(col + rows)[row / kInt8Block]);
    return __fmul_rn(static_cast<float>(reinterpret_cast<const std::int8_t*>(col)[row]), s);
}

// S = fl32(value(own) + value(peer)); partial <- S; out <- RNE_bf16(S). The own operand is read
// back from the bytes the peer received.
__global__ void exchange_int8_combine_kernel(float* __restrict__ partial,
                                             const std::byte* __restrict__ own,
                                             const std::byte* __restrict__ peer,
                                             __nv_bfloat16* __restrict__ bf16_out,
                                             std::int64_t rows, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        const float sum = __fadd_rn(int8_value(own, rows, i), int8_value(peer, rows, i));
        partial[i]      = sum;
        if (bf16_out != nullptr) { bf16_out[i] = __float2bfloat16_rn(sum); }
    }
}

// stats[0] = max |v| bits (non-negative floats order as unsigned ints), [1] = |v| > 65504,
// [2] = 0 < |v| < 2^-14 (FP16 subnormal), [3] = non-finite.
__global__ void exchange_stats_kernel(const float* __restrict__ v, std::size_t count,
                                      unsigned long long* __restrict__ stats) {
    unsigned int max_bits   = 0;
    unsigned long long over = 0, sub = 0, bad = 0;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        const float a = fabsf(v[i]);
        if (!isfinite(a)) {
            ++bad;
            continue;
        }
        max_bits = max(max_bits, __float_as_uint(a));
        over += a > 65504.0F ? 1ULL : 0ULL;
        sub += (a > 0.0F && a < 6.103515625e-05F) ? 1ULL : 0ULL;
    }
    atomicMax(stats, static_cast<unsigned long long>(max_bits));
    if (over != 0) { atomicAdd(stats + 1, over); }
    if (sub != 0) { atomicAdd(stats + 2, sub); }
    if (bad != 0) { atomicAdd(stats + 3, bad); }
}

int grid_for(std::size_t count) {
    const std::size_t want = (count + 255) / 256;
    return static_cast<int>(want < 1 ? 1 : (want > 1024 ? 1024 : want));
}

bool exchange_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TP_AR_MAXABS");
        return v != nullptr && std::string(v) == "1";
    }();
    return on;
}

} // namespace

void HostExchange::grow(std::vector<cudaEvent_t>& events, std::size_t count, int device) {
    if (events.size() >= count) { return; }
    CurrentDeviceGuard::set(device);
    while (events.size() < count) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        events.push_back(e);
    }
}

bool HostExchange::begin_sum_f32(const std::array<Tensor, 2>& partial,
                                 const std::array<Tensor, 2>& staging,
                                 const std::array<Tensor, 2>* bf16_out, Payload payload,
                                 const ExecutionContext& ec) {
    require(serves(ec), "HostExchange: serves another device pair");
    require(!pending_.active, "HostExchange: begin_sum_f32 while a call is open");
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusNone) { return false; }
    }
    for (int rank = 0; rank < 2; ++rank) {
        require(partial[rank].dtype == DType::FP32 && staging[rank].dtype == DType::FP32,
                "begin_sum_f32: partial/staging must be FP32");
        require(partial[rank].data != nullptr && staging[rank].data != nullptr &&
                    partial[rank].is_contiguous() && staging[rank].is_contiguous(),
                "begin_sum_f32: partial/staging must be contiguous non-null");
        require(partial[rank].data != staging[rank].data, "begin_sum_f32: staging aliases partial");
        for (int d = 0; d < 4; ++d) {
            require(partial[rank].ne[d] == partial[0].ne[d] &&
                        staging[rank].ne[d] == partial[0].ne[d],
                    "begin_sum_f32: partial/staging shapes must match on both devices");
        }
        if (bf16_out != nullptr) {
            const Tensor& out = (*bf16_out)[rank];
            require(out.dtype == DType::BF16 && out.data != nullptr && out.is_contiguous() &&
                        out.numel() == partial[0].numel(),
                    "begin_sum_f32: bf16_out must be contiguous BF16 of the partial's size");
        }
    }
    const CurrentDeviceGuard guard;
    Pending& p = pending_;
    p          = Pending{};
    p.ec       = &ec;
    p.partial  = partial;
    p.staging  = staging;
    p.has_out  = bf16_out != nullptr;
    if (p.has_out) { p.out = *bf16_out; }
    p.payload = payload;
    p.rows    = partial[0].ne[0];
    p.columns = static_cast<std::int64_t>(partial[0].numel()) / std::max<std::int64_t>(p.rows, 1);
    if (p.payload == Payload::INT8 && p.rows % kInt8Block != 0) { p.payload = Payload::FP16; }
    const std::size_t esize = payload == Payload::FP32 ? 4 : 2;
    const std::size_t bytes = static_cast<std::size_t>(partial[0].numel()) * esize;
    reserve(bytes, 0, ec);
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        // staging[r] (and the narrow copy in it) is free once rank r's stream got here.
        CUDA_CHECK(cudaEventRecord(begun_[r], ec.dev[rank]->stream));
        CUDA_CHECK(cudaStreamWaitEvent(receive_[r], begun_[r], 0));
    }
    p.active = true;
    return true;
}

void HostExchange::columns_ready(int rank, std::int64_t first, std::int64_t count) {
    Pending& p = pending_;
    require(p.active, "HostExchange: columns_ready without begin_sum_f32");
    require(rank == 0 || rank == 1, "HostExchange: rank must be 0 or 1");
    const auto r    = static_cast<std::size_t>(rank);
    const auto peer = static_cast<std::size_t>(1 - rank);
    require(first == p.done[r] && count >= 0 && first + count <= p.columns,
            "HostExchange: column groups must be consecutive within [0, T)");
    if (count == 0) { return; }
    const ExecutionContext& ec = *p.ec;
    const CurrentDeviceGuard guard;
    const std::size_t total = static_cast<std::size_t>(p.rows * p.columns);
    const std::size_t begin = static_cast<std::size_t>(p.rows * first);
    const std::size_t n     = static_cast<std::size_t>(p.rows * count);
    // Bytes per column on the links: FP32 4 * rows, FP16 / BF16 2 * rows, INT8 rows * 17 / 16.
    std::size_t column_bytes = 0;
    switch (p.payload) {
    case Payload::FP32:
        column_bytes = 4 * static_cast<std::size_t>(p.rows);
        break;
    case Payload::FP16:
    case Payload::BF16:
        column_bytes = 2 * static_cast<std::size_t>(p.rows);
        break;
    case Payload::INT8:
        column_bytes = int8_column_bytes(p.rows);
        break;
    default:
        require(false, "HostExchange: unknown payload");
    }
    const std::size_t offset  = static_cast<std::size_t>(first) * column_bytes;
    const std::size_t bytes   = static_cast<std::size_t>(count) * column_bytes;
    const cudaStream_t stream = ec.dev[rank]->stream;
    CurrentDeviceGuard::set(ec.dev[rank]->device);
    const std::byte* source = nullptr;
    std::byte* destination  = nullptr; // on the peer
    if (p.payload == Payload::FP32) {
        source      = static_cast<const std::byte*>(p.partial[r].data) + offset;
        destination = static_cast<std::byte*>(p.staging[peer].data) + offset;
    } else if (p.payload == Payload::INT8) {
        // Own bytes in the first part of staging[r]; the peer receives at byte 2 * total of
        // staging[peer] (where the 16-bit payloads' second half starts; staging holds 4 * total
        // bytes).
        std::byte* own = static_cast<std::byte*>(p.staging[r].data) + offset;
        exchange_int8_narrow_kernel<<<grid_for(n), 256, 0, stream>>>(
            static_cast<const float*>(p.partial[r].data) + begin, own, p.rows, count);
        CUDA_CHECK(cudaGetLastError());
        source      = own;
        destination = static_cast<std::byte*>(p.staging[peer].data) + 2 * total + offset;
    } else {
        auto* own = static_cast<std::uint16_t*>(p.staging[r].data) + begin;
        exchange_narrow_kernel<<<grid_for(n), 256, 0, stream>>>(
            static_cast<const float*>(p.partial[r].data) + begin, own, n,
            static_cast<int>(p.payload), ldexpf(1.0F, -exchange_shift()));
        CUDA_CHECK(cudaGetLastError());
        source      = reinterpret_cast<const std::byte*>(own);
        destination = reinterpret_cast<std::byte*>(
            static_cast<std::uint16_t*>(p.staging[peer].data) + total + begin);
    }
    const std::size_t chunks = (bytes + chunk_bytes_ - 1) / chunk_bytes_;
    grow(group_ready_[r], p.groups[r] + 1, ec.dev[rank]->device);
    grow(pipe_chunk_[r], p.chunks[r] + chunks, ec.dev[rank]->device);
    CurrentDeviceGuard::set(ec.dev[rank]->device);
    const cudaEvent_t ready = group_ready_[r][p.groups[r]++];
    CUDA_CHECK(cudaEventRecord(ready, stream));
    CUDA_CHECK(cudaStreamWaitEvent(side_[r], ready, 0));
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t o = c * chunk_bytes_;
        const std::size_t m = std::min(chunk_bytes_, bytes - o);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaMemcpyAsync(host_[r] + offset + o, source + o, m, cudaMemcpyDeviceToHost,
                                   side_[r]));
        const cudaEvent_t done = pipe_chunk_[r][p.chunks[r]++];
        CUDA_CHECK(cudaEventRecord(done, side_[r]));
        CurrentDeviceGuard::set(ec.dev[1 - rank]->device);
        CUDA_CHECK(cudaStreamWaitEvent(receive_[peer], done, 0));
        receive_to_device(destination + o, host_[r] + offset + o, m, receive_[peer],
                          receive_by_kernel_);
    }
    p.done[r] += count;
}

void HostExchange::finish_sum_f32() {
    Pending& p = pending_;
    require(p.active, "HostExchange: finish_sum_f32 without begin_sum_f32");
    require(p.done[0] == p.columns && p.done[1] == p.columns,
            "HostExchange: finish_sum_f32 before every column was reported");
    const ExecutionContext& ec = *p.ec;
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(received_[static_cast<std::size_t>(rank)],
                                   receive_[static_cast<std::size_t>(rank)]));
    }
    const std::size_t count = static_cast<std::size_t>(p.rows * p.columns);
    const bool stats        = exchange_stats_enabled();
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const cudaStream_t stream = ec.dev[rank]->stream;
        // Inbound data complete, and the peer has read rank r's host buffer (its receive copies
        // follow rank r's device-to-host copies): partial[r] and the buffers may be reused.
        CUDA_CHECK(cudaStreamWaitEvent(stream, received_[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, received_[1 - r], 0));
        if (stats) {
            if (stats_[r] == nullptr) {
                CUDA_CHECK(cudaMalloc(&stats_[r], 4 * sizeof(unsigned long long)));
            }
            auto* st = static_cast<unsigned long long*>(stats_[r]);
            CUDA_CHECK(cudaMemsetAsync(st, 0, 4 * sizeof(unsigned long long), stream));
            exchange_stats_kernel<<<grid_for(count), 256, 0, stream>>>(
                static_cast<const float*>(p.partial[r].data), count, st);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long h[4] = {0, 0, 0, 0};
            CUDA_CHECK(cudaMemcpyAsync(h, st, sizeof(h), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto bits = static_cast<std::uint32_t>(h[0]);
            float max_abs   = 0.0F;
            std::memcpy(&max_abs, &bits, sizeof(max_abs));
            std::fprintf(stderr,
                         "T62 maxabs call=%lld rank=%d T=%lld max=%.6g over_fp16=%llu "
                         "sub_fp16=%llu nonfinite=%llu n=%zu\n",
                         static_cast<long long>(calls_), rank, static_cast<long long>(p.columns),
                         static_cast<double>(max_abs), h[1], h[2], h[3], count);
        }
        auto* out = p.has_out ? static_cast<__nv_bfloat16*>(p.out[r].data) : nullptr;
        if (p.payload == Payload::FP32) {
            allreduce_f32_combine(static_cast<float*>(p.partial[r].data),
                                  static_cast<const float*>(p.staging[r].data), out, count, stream);
        } else if (p.payload == Payload::INT8) {
            const auto* staged = static_cast<const std::byte*>(p.staging[r].data);
            exchange_int8_combine_kernel<<<grid_for(count), 256, 0, stream>>>(
                static_cast<float*>(p.partial[r].data), staged, staged + 2 * count, out, p.rows,
                count);
            CUDA_CHECK(cudaGetLastError());
        } else {
            exchange_narrow_combine_kernel<<<grid_for(count), 256, 0, stream>>>(
                static_cast<float*>(p.partial[r].data),
                static_cast<const std::uint16_t*>(p.staging[r].data) + count, out, count,
                static_cast<int>(p.payload), ldexpf(1.0F, -exchange_shift()),
                ldexpf(1.0F, exchange_shift()));
            CUDA_CHECK(cudaGetLastError());
        }
    }
    ++calls_;
    p.active = false;
}

void HostExchange::abandon_sum_f32() noexcept {
    Pending& p = pending_;
    if (!p.active) { return; }
    p.active                   = false;
    const ExecutionContext& ec = *p.ec;
    int previous               = 0;
    const bool restore         = cudaGetDevice(&previous) == cudaSuccess;
    for (std::size_t r = 0; r < 2; ++r) {
        (void)cudaSetDevice(ec.dev[r]->device);
        (void)cudaEventRecord(received_[r], receive_[r]);
    }
    for (std::size_t r = 0; r < 2; ++r) {
        (void)cudaSetDevice(ec.dev[r]->device);
        (void)cudaStreamWaitEvent(ec.dev[r]->stream, received_[r], 0);
        (void)cudaStreamWaitEvent(ec.dev[r]->stream, received_[1 - r], 0);
    }
    if (restore) { (void)cudaSetDevice(previous); }
}

namespace {
// Chunk of a lane's transfer: the peer's host-to-device copy of chunk c overlaps the
// device-to-host copy of chunk c+1 (the two directions run on the two boards' links).
constexpr std::size_t kLaneChunkBytes = std::size_t{2} << 20;
} // namespace

void HostExchange::reserve_lane(Lane& lane, std::size_t bytes, std::size_t chunks,
                                const ExecutionContext& ec) {
    if (bytes > lane.capacity) {
        // An earlier call of this lane may still read or write the old buffers.
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (std::byte*& h : lane.host) {
            if (h != nullptr) { CUDA_CHECK(cudaFreeHost(h)); }
            h = nullptr;
        }
        lane.capacity = 0;
        for (std::byte*& h : lane.host) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
            h = static_cast<std::byte*>(p);
        }
        lane.capacity = bytes;
    }
    for (int rank = 0; rank < 2; ++rank) {
        grow(lane.chunk_done[static_cast<std::size_t>(rank)], chunks, ec.dev[rank]->device);
    }
}

void HostExchange::sum_begin(int lane_index, const std::array<Tensor, 2>& buffer,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec) {
    require(serves(ec), "HostExchange: serves another device pair");
    require(lane_index >= 0 && lane_index < kSumLanes, "HostExchange: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(!lane.active, "HostExchange: sum_begin on an open lane");
    const std::size_t bytes = buffer[0].bytes();
    require(bytes > 0 && buffer[1].bytes() == bytes && staging[0].bytes() == bytes &&
                staging[1].bytes() == bytes,
            "HostExchange: sum operands must share one size");
    const CurrentDeviceGuard guard;
    const std::size_t chunks = (bytes + kLaneChunkBytes - 1) / kLaneChunkBytes;
    reserve_lane(lane, bytes, chunks, ec);
    // Both operands are complete here on the ranks' streams; staging[r] and the lane's host
    // buffers are free, since the lane's previous sum_finish (earlier on the same streams)
    // waited for every copy of its call.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(lane.begun[r], ec.dev[rank]->stream));
        CUDA_CHECK(cudaStreamWaitEvent(lane_send_[r], lane.begun[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(lane_receive_[r], lane.begun[r], 0));
    }
    // Leg 1, both ranks: operand -> the lane's host buffer, one event per chunk.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const auto* source = static_cast<const std::byte*>(buffer[r].data);
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * kLaneChunkBytes;
            const std::size_t n      = std::min(kLaneChunkBytes, bytes - offset);
            CUDA_CHECK(cudaMemcpyAsync(lane.host[r] + offset, source + offset, n,
                                       cudaMemcpyDeviceToHost, lane_send_[r]));
            CUDA_CHECK(cudaEventRecord(lane.chunk_done[r][c], lane_send_[r]));
        }
    }
    // Leg 2, both ranks: the peer's host buffer -> own staging on the receive stream, chunk by
    // chunk behind the peer's events; the lane's completion is recorded right after its last
    // copy, so a later lane's copies on the same stream do not delay this lane's finish.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r    = static_cast<std::size_t>(rank);
        const auto peer = static_cast<std::size_t>(1 - rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        auto* destination = static_cast<std::byte*>(staging[r].data);
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * kLaneChunkBytes;
            const std::size_t n      = std::min(kLaneChunkBytes, bytes - offset);
            CUDA_CHECK(cudaStreamWaitEvent(lane_receive_[r], lane.chunk_done[peer][c], 0));
            CUDA_CHECK(cudaMemcpyAsync(destination + offset, lane.host[peer] + offset, n,
                                       cudaMemcpyHostToDevice, lane_receive_[r]));
        }
        CUDA_CHECK(cudaEventRecord(lane.received[r], lane_receive_[r]));
    }
    lane.buffer  = buffer;
    lane.staging = staging;
    lane.active  = true;
}

void HostExchange::sum_finish(int lane_index, const ExecutionContext& ec) {
    require(serves(ec), "HostExchange: serves another device pair");
    require(lane_index >= 0 && lane_index < kSumLanes, "HostExchange: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(lane.active, "HostExchange: sum_finish without sum_begin");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const cudaStream_t stream = ec.dev[rank]->stream;
        // Inbound data complete, and the peer has read rank r's host buffer (its receive copies
        // follow rank r's device-to-host reads of buffer[r]): buffer[r] may be overwritten and
        // every buffer of the lane reused.
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[1 - r], 0));
        Tensor accumulator = lane.buffer[r];
        detail::residual_add_launch(lane.staging[r], accumulator, stream);
    }
    lane.buffer  = {};
    lane.staging = {};
    lane.active  = false;
}

// ---- v07/p2p-overlap: split-phase lanes over the direct cross-device copy ----------------------

PeerLanes::PeerLanes(const ExecutionContext& ec, bool direct) : direct_(direct) {
    require_two_devices(ec, "PeerLanes: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        devices_[r]  = ec.dev[rank]->device;
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&pull_[r], cudaStreamNonBlocking));
        for (Lane& lane : lanes_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.begun[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.received[r], cudaEventDisableTiming));
        }
    }
}

PeerLanes::~PeerLanes() {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) { previous = -1; }
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] < 0 || cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        if (pull_[rank] != nullptr) {
            (void)cudaStreamSynchronize(pull_[rank]);
            (void)cudaStreamDestroy(pull_[rank]);
        }
        for (Lane& lane : lanes_) {
            for (cudaEvent_t e : {lane.begun[rank], lane.received[rank]}) {
                if (e != nullptr) { (void)cudaEventDestroy(e); }
            }
        }
    }
    // The side streams read the peer's memory and the combines of an open lane may still be
    // queued on the ranks' streams: drain both devices before the operands can go (PR #15,
    // DirectExchange's destructor).
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] >= 0 && cudaSetDevice(devices_[rank]) == cudaSuccess) {
            (void)cudaDeviceSynchronize();
        }
    }
    if (previous >= 0) { (void)cudaSetDevice(previous); }
}

bool PeerLanes::serves(const ExecutionContext& ec) const noexcept {
    return ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value() &&
           ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

void PeerLanes::sum_begin(int lane_index, const std::array<Tensor, 2>& buffer,
                          const std::array<Tensor, 2>& staging, const ExecutionContext& ec) {
    require(serves(ec), "PeerLanes: serves another device pair");
    require(lane_index >= 0 && lane_index < kSumLanes, "PeerLanes: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(!lane.active, "PeerLanes: sum_begin on an open lane");
    const std::size_t bytes = buffer[0].bytes();
    require(bytes > 0 && buffer[1].bytes() == bytes && staging[0].bytes() == bytes &&
                staging[1].bytes() == bytes,
            "PeerLanes: sum operands must share one size");
    const CurrentDeviceGuard guard;
    // Phase A, both ranks, before any wait observes it: operand r is complete and staging[r] is
    // free (the lane's previous sum_finish, earlier on the same stream, waited for its copies).
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(lane.begun[r], ec.dev[rank]->stream));
    }
    if (!direct_) {
        // Driver-staged test mode: the staged copy does not order itself after the source
        // device's work (T65), so both operands are retired on the host first.
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaEventSynchronize(lane.begun[static_cast<std::size_t>(rank)]));
        }
    }
    // Phase B, both ranks: the pull of the one-call path, on rank r's side stream.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r    = static_cast<std::size_t>(rank);
        const auto peer = static_cast<std::size_t>(1 - rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamWaitEvent(pull_[r], lane.begun[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(pull_[r], lane.begun[peer], 0));
        CUDA_CHECK(pull_peer(staging[r].data, buffer[peer].data, bytes, pull_[r]));
        if (!direct_) { CUDA_CHECK(cudaStreamSynchronize(pull_[r])); }
        CUDA_CHECK(cudaEventRecord(lane.received[r], pull_[r]));
    }
    lane.buffer  = buffer;
    lane.staging = staging;
    lane.active  = true;
}

void PeerLanes::sum_finish(int lane_index, const ExecutionContext& ec) {
    require(serves(ec), "PeerLanes: serves another device pair");
    require(lane_index >= 0 && lane_index < kSumLanes, "PeerLanes: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(lane.active, "PeerLanes: sum_finish without sum_begin");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const cudaStream_t stream = ec.dev[rank]->stream;
        // Inbound data complete, and the peer has finished reading buffer[r]: buffer[r] may be
        // overwritten and staging[r] reused.
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[1 - r], 0));
        Tensor accumulator = lane.buffer[r];
        detail::residual_add_launch(lane.staging[r], accumulator, stream);
    }
    lane.buffer  = {};
    lane.staging = {};
    lane.active  = false;
}

bool allreduce_sum_split_available(const ExecutionContext& ec, const PeerEvents& events) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() || !events.live()) {
        return false;
    }
    const CurrentDeviceGuard guard;
    if (eager_peer_lanes(ec, events) != nullptr) { return true; }
    return eager_host_exchange(ec, events) != nullptr;
}

void allreduce_sum_begin(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events, int lane) {
    detail::require_allreduce_sum_arguments(buffer, staging, ec, events);
    PeerLanes* lanes = nullptr;
    {
        const CurrentDeviceGuard guard;
        lanes = eager_peer_lanes(ec, events);
    }
    if (lanes != nullptr) {
        lanes->sum_begin(lane, buffer, staging, ec);
        return;
    }
    HostExchange* exchange = nullptr;
    {
        const CurrentDeviceGuard guard;
        exchange = eager_host_exchange(ec, events);
    }
    require(exchange != nullptr,
            "allreduce_sum_begin: needs an eager HostExchange and uncaptured streams");
    exchange->sum_begin(lane, buffer, staging, ec);
}

void allreduce_sum_finish(const ExecutionContext& ec, const PeerEvents& events, int lane) {
    if (PeerLanes* lanes = events.peer_lanes(); lanes != nullptr && lanes->sum_open(lane)) {
        lanes->sum_finish(lane, ec);
        return;
    }
    HostExchange* exchange = events.host_exchange();
    require(exchange != nullptr && exchange->sum_open(lane),
            "allreduce_sum_finish: the lane has no open call");
    exchange->sum_finish(lane, ec);
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    detail::require_allreduce_sum_arguments(buffer, staging, ec, events);
    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }

    const CurrentDeviceGuard guard;

    // MAILBOX TRANSPORT (see captured_mailbox()): one exchange kernel per device. Both ranks
    // publish their partial into their own pinned host slot, release an epoch flag, wait for the
    // peer's, and combine locally with the same arithmetic as residual_add_launch below.
    if (PeerMailbox* mailbox = captured_mailbox(buffer, bytes, ec, events)) {
        const int slot = mailbox->take_capture_slot(bytes);
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            mailbox->enqueue_exchange_sum(rank, slot, buffer[rank].data, bytes, local.stream);
        }
        return;
    }

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer's operand into storage only it owns. Both inbound copies
    // are issued before either rank waits again, so the two directions overlap. Eagerly with a
    // HostExchange attached the pull goes through its pinned buffers (T56: no host block).
    if (HostExchange* exchange = eager_host_exchange(ec, events)) {
        exchange->pull({buffer[0].data, buffer[1].data}, {staging[0].data, staging[1].data}, bytes,
                       ec, events);
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allreduce_sum_f32(const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& staging,
                       const std::array<Tensor, 2>* bf16_out, const ExecutionContext& ec,
                       const PeerEvents& events) {
    require_two_devices(
        ec, "allreduce_sum_f32: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(partial[rank].dtype == DType::FP32 && staging[rank].dtype == DType::FP32,
                "allreduce_sum_f32: partial/staging must be FP32");
        require(partial[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum_f32: partial/staging data must be non-null");
        require(partial[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum_f32: partial/staging must be contiguous");
        require(partial[rank].data != staging[rank].data,
                "allreduce_sum_f32: staging must not alias the partial");
        for (int d = 0; d < 4; ++d) {
            require(partial[rank].ne[d] == partial[0].ne[d] &&
                        staging[rank].ne[d] == partial[0].ne[d],
                    "allreduce_sum_f32: partial/staging shapes must match on both devices");
        }
        if (bf16_out != nullptr) {
            const Tensor& out = (*bf16_out)[rank];
            require(out.dtype == DType::BF16 && out.data != nullptr && out.is_contiguous(),
                    "allreduce_sum_f32: bf16_out must be contiguous non-null BF16");
            require((reinterpret_cast<std::uintptr_t>(out.data) % 16) == 0,
                    "allreduce_sum_f32: bf16_out must be 16-byte aligned");
            for (int d = 0; d < 4; ++d) {
                require(out.ne[d] == partial[0].ne[d],
                        "allreduce_sum_f32: bf16_out must have the partial's shape");
            }
            require(out.data != partial[rank].data && out.data != staging[rank].data,
                    "allreduce_sum_f32: bf16_out must not alias the partial or staging");
        }
    }
    require(events.live(), "allreduce_sum_f32: events must be live");
    const std::size_t count = partial[0].numel();
    const std::size_t bytes = count * sizeof(float);
    if (bytes == 0) { return; }

    const CurrentDeviceGuard guard;
    const auto out_data = [&](int rank) -> __nv_bfloat16* {
        return bf16_out != nullptr ? static_cast<__nv_bfloat16*>((*bf16_out)[rank].data) : nullptr;
    };

    PeerMailbox* mailbox = captured_mailbox(partial, bytes, ec, events);
    if (mailbox != nullptr && mailbox->kernel() == PeerExchangeKernel::Pipelined) {
        const int slot = mailbox->take_capture_slot(bytes);
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            mailbox->enqueue_exchange_sum_f32(rank, slot, partial[rank].data, bytes, out_data(rank),
                                              local.stream);
        }
        return;
    }

    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }
    if (HostExchange* exchange = eager_host_exchange(ec, events)) {
        exchange->pull({partial[0].data, partial[1].data}, {staging[0].data, staging[1].data},
                       bytes, ec, events);
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(pull_peer(staging[rank].data, partial[1 - rank].data, bytes, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        allreduce_f32_combine(static_cast<float*>(partial[rank].data),
                              static_cast<const float*>(staging[rank].data), out_data(rank), count,
                              local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer's block, both on its own
    // stream, so destination[r] has exactly one writer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                             part[1 - rank].data, block[1 - rank], local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

} // namespace ninfer::ops
